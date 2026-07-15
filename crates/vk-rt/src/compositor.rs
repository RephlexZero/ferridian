//! Running a loaded pack over the intercepted frame (the M4 seam).
//!
//! At the end of the classified pass that finishes the world
//! ([`ferridian_engine::frame::composite_trigger`]), the layer hands the
//! [`PackCompositor`] the pass's resolved attachments. The compositor copies
//! them into its own *tap* images (the pack's `game_color`/`game_depth`
//! inputs — copies, because the composite also writes the color attachment,
//! and sampling an image while rendering to it is a feedback loop), then
//! records the [`PackExecutor`] with the application's own color attachment
//! as the swapchain output, left in the layout the app's render pass had
//! already promised. Hand and GUI passes then draw on top, untouched.
//!
//! Runtime code inside someone's game: nothing panics, every failure logs
//! and disables compositing (the game must keep rendering vanilla), and
//! device objects built against the app's views are torn down the moment the
//! app destroys one of them ([`PackCompositor::invalidate_view`]).

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use ash::vk;
use ferridian_contract::CameraUniforms;
use ferridian_engine::exec::{ExecutionPlan, WireError, plan_execution};
use ferridian_engine::pack::{LoadedPack, PassModules};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::exec::{ExecContext, ExecError, OutputTarget, PackExecutor, vk_err};
use crate::tap::{ResolvedAttachment, ResolvedFrame};

#[derive(Debug, thiserror::Error)]
pub enum CompositorError {
    #[error(transparent)]
    Exec(#[from] ExecError),
    #[error("pack input {0:?} has no source attachment in the intercepted pass")]
    MissingSource(String),
}

/// One pack input tapped from the intercepted pass: the source attachment
/// and the compositor-owned copy the executor actually samples.
struct Tap {
    name: String,
    source: ResolvedAttachment,
    source_is_depth: bool,
    image: vk::Image,
    allocation: Allocation,
    view: vk::ImageView,
}

/// Everything built against one concrete frame geometry (extent + attachment
/// identities). Rebuilt from scratch after [`PackCompositor::invalidate_view`].
struct Built {
    extent: vk::Extent2D,
    output: ResolvedAttachment,
    taps: Vec<Tap>,
    executor: PackExecutor,
}

/// A loaded pack, planned once, instantiated lazily against the first
/// triggering frame's geometry.
pub struct PackCompositor {
    /// Down-chain device table: recording through it never re-enters the layer.
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    allocator: Arc<Mutex<Allocator>>,
    plan: ExecutionPlan,
    modules: BTreeMap<String, PassModules>,
    built: Option<Built>,
    /// Building failed once — stay off rather than failing every frame.
    disabled: bool,
    mismatch_warned: bool,
}

/// Which resolved attachment feeds a pack external input, and whether it is
/// the depth attachment (barrier stages/aspects differ).
fn source_for(name: &str, frame: &ResolvedFrame) -> Option<(ResolvedAttachment, bool)> {
    match name {
        "game_color" => Some((frame.color, false)),
        "game_depth" => frame.depth.map(|depth| (depth, true)),
        _ => None,
    }
}

/// Every aspect the format has — layout transitions must cover the whole
/// image, including stencil the pack never samples.
fn full_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    match format {
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT => {
            vk::ImageAspectFlags::DEPTH
        }
        vk::Format::S8_UINT => vk::ImageAspectFlags::STENCIL,
        vk::Format::D16_UNORM_S8_UINT
        | vk::Format::D24_UNORM_S8_UINT
        | vk::Format::D32_SFLOAT_S8_UINT => {
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        }
        _ => vk::ImageAspectFlags::COLOR,
    }
}

/// The single aspect the pack samples (and the copy moves): depth for
/// depth-bearing formats, color otherwise.
fn sample_aspect(format: vk::Format) -> vk::ImageAspectFlags {
    if full_aspect(format).contains(vk::ImageAspectFlags::DEPTH) {
        vk::ImageAspectFlags::DEPTH
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

impl PackCompositor {
    /// Plan the pack (pure — the same name-wiring the tests exercise). A pack
    /// that doesn't wire is rejected here, before any device object exists.
    pub fn new(
        device: ash::Device,
        queue: vk::Queue,
        queue_family_index: u32,
        allocator: Arc<Mutex<Allocator>>,
        pack: &LoadedPack,
    ) -> Result<PackCompositor, WireError> {
        let plan = plan_execution(pack)?;
        Ok(PackCompositor {
            device,
            queue,
            queue_family_index,
            allocator,
            plan,
            modules: pack.modules.clone(),
            built: None,
            disabled: false,
            mismatch_warned: false,
        })
    }

    pub fn pass_count(&self) -> usize {
        self.plan.passes.len()
    }

    /// Upload this frame's camera state — the layer calls this once per
    /// composited frame, immediately before [`PackCompositor::record_over`],
    /// with whatever the shim → layer transport last published (see
    /// `ferridian_vk_layer::camera_transport`). A no-op until the compositor
    /// has built (nothing to upload into yet).
    ///
    /// Unsynchronized against a still-in-flight previous frame's read of the
    /// same buffer — the same frame-pacing assumption `record_over` already
    /// makes for the tap images it rewrites every frame without a fence
    /// wait. Real per-swapchain-image resources are the same M2-remainder
    /// follow-up as the taps' (needs a real game to prove against).
    pub fn update_camera(&mut self, camera: &CameraUniforms) {
        let Some(built) = &self.built else {
            return;
        };
        // SAFETY: see the doc comment above.
        if let Err(error) = unsafe { built.executor.update_camera(camera) } {
            tracing::warn!(%error, "camera update failed; the frame keeps its previous values");
        }
    }

    /// Record the pack over the intercepted frame: tap copies, then the
    /// executor's schedule, leaving the app's color attachment in the layout
    /// its own render pass had declared. Called after the app's render pass
    /// has ended in `command_buffer`.
    ///
    /// Returns whether the pack was actually recorded — `false` covers every
    /// silent early-out (disabled after a failed build, geometry mismatch),
    /// so the caller's success accounting can't drift from reality.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state and *outside* a render
    /// pass instance, on this device's recording thread; the frame's handles
    /// must be live.
    #[must_use]
    pub unsafe fn record_over(
        &mut self,
        command_buffer: vk::CommandBuffer,
        frame: &ResolvedFrame,
    ) -> bool {
        if self.disabled {
            return false;
        }
        if self.built.is_none() {
            match self.build(frame) {
                Ok(built) => {
                    tracing::info!(
                        passes = self.plan.passes.len(),
                        width = frame.extent.width,
                        height = frame.extent.height,
                        "pack compositor built over intercepted frame"
                    );
                    self.built = Some(built);
                }
                Err(error) => {
                    tracing::warn!(%error, "pack compositing disabled: build failed");
                    self.disabled = true;
                    return false;
                }
            }
        }
        let Some(built) = &self.built else {
            return false;
        };
        let geometry_matches = built.extent == frame.extent
            && built.output.view == frame.color.view
            && built.taps.iter().all(|tap| {
                source_for(&tap.name, frame)
                    .is_some_and(|(source, _)| source.view == tap.source.view)
            });
        if !geometry_matches {
            // Rebuilding here would destroy objects possibly still referenced
            // by in-flight frames; the safe rebuild point is invalidate_view
            // (the app destroying its old attachments proves they are idle).
            if !self.mismatch_warned {
                tracing::warn!(
                    "pack compositing skipped: frame geometry changed without the old \
                     attachments being destroyed (multiple render targets in flight?)"
                );
                self.mismatch_warned = true;
            }
            return false;
        }

        // Tap copies. One barrier batch in, the copies, one barrier batch out;
        // stage masks are the union over the batch, access masks are per image.
        let mut acquire: Vec<vk::ImageMemoryBarrier<'_>> = Vec::new();
        let mut release: Vec<vk::ImageMemoryBarrier<'_>> = Vec::new();
        let barrier = |image: vk::Image,
                       old_layout: vk::ImageLayout,
                       new_layout: vk::ImageLayout,
                       src_access: vk::AccessFlags,
                       dst_access: vk::AccessFlags,
                       aspect: vk::ImageAspectFlags| {
            vk::ImageMemoryBarrier::default()
                .image(image)
                .old_layout(old_layout)
                .new_layout(new_layout)
                .src_access_mask(src_access)
                .dst_access_mask(dst_access)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(aspect)
                        .level_count(1)
                        .layer_count(1),
                )
        };
        for tap in &built.taps {
            let source_access = if tap.source_is_depth {
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            } else {
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            };
            acquire.push(barrier(
                tap.source.image,
                tap.source.final_layout,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                source_access,
                vk::AccessFlags::TRANSFER_READ,
                full_aspect(tap.source.format),
            ));
            // The tap was sampled by the previous composite; UNDEFINED
            // discards it (fully rewritten by the copy).
            acquire.push(barrier(
                tap.image,
                vk::ImageLayout::UNDEFINED,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::AccessFlags::empty(),
                vk::AccessFlags::TRANSFER_WRITE,
                full_aspect(tap.source.format),
            ));
            // Executor sampling reads the tap; the transfer write must be
            // visible to fragment shaders.
            release.push(barrier(
                tap.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                vk::AccessFlags::TRANSFER_WRITE,
                vk::AccessFlags::SHADER_READ,
                full_aspect(tap.source.format),
            ));
            if tap.source_is_depth {
                // Give the depth attachment back exactly as the app left it.
                release.push(barrier(
                    tap.source.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    tap.source.final_layout,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                    full_aspect(tap.source.format),
                ));
            } else {
                // The color attachment is also the executor's output. No
                // transition (its render pass begins from UNDEFINED), but the
                // tap read must finish before the composite overwrites it —
                // the executor's own entry dependency only orders against
                // prior color-attachment writes, not transfers.
                release.push(barrier(
                    tap.source.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                    full_aspect(tap.source.format),
                ));
            }
        }

        // SAFETY: caller guarantees a recording command buffer outside a
        // render pass; every handle below was created on this device and is
        // kept alive by `built` until invalidation, which the caller only
        // reaches once the app proves the frame idle.
        unsafe {
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                    | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS
                    // The previous composite's fragment *and compute* passes
                    // sampled the taps — order the rewrite behind both.
                    | vk::PipelineStageFlags::FRAGMENT_SHADER
                    | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &acquire,
            );
            for tap in &built.taps {
                let subresource = vk::ImageSubresourceLayers::default()
                    .aspect_mask(sample_aspect(tap.source.format))
                    .layer_count(1);
                let region = vk::ImageCopy::default()
                    .src_subresource(subresource)
                    .dst_subresource(subresource)
                    .extent(built.extent.into());
                self.device.cmd_copy_image(
                    command_buffer,
                    tap.source.image,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    tap.image,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    &[region],
                );
            }
            self.device.cmd_pipeline_barrier(
                command_buffer,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &release,
            );
            built.executor.record(&self.device, command_buffer);
        }
        true
    }

    /// Create the taps and instantiate the executor against one frame
    /// geometry. Anything half-built is destroyed before the error returns.
    fn build(&self, frame: &ResolvedFrame) -> Result<Built, CompositorError> {
        let device = &self.device;
        let mut taps: Vec<Tap> = Vec::new();
        let result = (|| {
            for resource in &self.plan.external_inputs {
                let name = resource.0.clone();
                let (source, source_is_depth) = source_for(&name, frame)
                    .ok_or_else(|| CompositorError::MissingSource(name.clone()))?;
                // Slot pushed with null handles first, filled as each create
                // succeeds, so a failure leaves nothing unreachable.
                taps.push(Tap {
                    name,
                    source,
                    source_is_depth,
                    image: vk::Image::null(),
                    allocation: Allocation::default(),
                    view: vk::ImageView::null(),
                });
                let Some(tap) = taps.last_mut() else {
                    unreachable!("pushed above");
                };
                let image_info = vk::ImageCreateInfo::default()
                    .image_type(vk::ImageType::TYPE_2D)
                    .format(source.format)
                    .extent(frame.extent.into())
                    .mip_levels(1)
                    .array_layers(1)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .tiling(vk::ImageTiling::OPTIMAL)
                    .usage(vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::SAMPLED)
                    .initial_layout(vk::ImageLayout::UNDEFINED);
                // SAFETY: valid create infos over locals; handles land in the
                // tap slot and are destroyed by destroy_taps on any failure.
                unsafe {
                    tap.image = device
                        .create_image(&image_info, None)
                        .map_err(vk_err("create tap image"))
                        .map_err(CompositorError::Exec)?;
                    let requirements = device.get_image_memory_requirements(tap.image);
                    tap.allocation = self
                        .allocator
                        .lock()
                        .expect("allocator poisoned")
                        .allocate(&AllocationCreateDesc {
                            name: &format!("ferridian tap {}", tap.name),
                            requirements,
                            location: MemoryLocation::GpuOnly,
                            linear: false,
                            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                        })
                        .map_err(ExecError::from)
                        .map_err(CompositorError::Exec)?;
                    device
                        .bind_image_memory(
                            tap.image,
                            tap.allocation.memory(),
                            tap.allocation.offset(),
                        )
                        .map_err(vk_err("bind tap memory"))
                        .map_err(CompositorError::Exec)?;
                    let view_info = vk::ImageViewCreateInfo::default()
                        .image(tap.image)
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(source.format)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(sample_aspect(source.format))
                                .level_count(1)
                                .layer_count(1),
                        );
                    tap.view = device
                        .create_image_view(&view_info, None)
                        .map_err(vk_err("create tap view"))
                        .map_err(CompositorError::Exec)?;
                }
            }
            Ok(())
        })();
        if let Err(error) = result {
            // SAFETY: the taps were never recorded anywhere yet.
            unsafe { destroy_taps(device, &self.allocator, &mut taps) };
            return Err(error);
        }

        let external_inputs: BTreeMap<String, vk::ImageView> = taps
            .iter()
            .map(|tap| (tap.name.clone(), tap.view))
            .collect();
        let ctx = ExecContext {
            device,
            queue: self.queue,
            queue_family_index: self.queue_family_index,
            allocator: self.allocator.clone(),
        };
        let output = OutputTarget {
            view: frame.color.view,
            format: frame.color.format,
            final_layout: frame.color.final_layout,
        };
        match PackExecutor::new(
            &ctx,
            &self.plan,
            &self.modules,
            frame.extent,
            &external_inputs,
            // The very first triggering frame builds before the layer's
            // per-frame `update_camera` call can reach an executor to write
            // into (see `PackCompositor::update_camera`) — one frame of the
            // placeholder, then whatever the shim has published since.
            &CameraUniforms::placeholder(),
            &output,
        ) {
            Ok(executor) => Ok(Built {
                extent: frame.extent,
                output: frame.color,
                taps,
                executor,
            }),
            Err(error) => {
                // SAFETY: as above — nothing has been submitted.
                unsafe { destroy_taps(device, &self.allocator, &mut taps) };
                Err(CompositorError::Exec(error))
            }
        }
    }

    /// The application is destroying `view`. If anything we built references
    /// it, tear that down now — the app destroying a view it rendered with
    /// proves no submitted work still uses it (vkDestroyImageView's own
    /// rules), and our recorded work rode the app's submissions.
    ///
    /// # Safety
    /// Must be called from the layer's `vkDestroyImageView` hook, before the
    /// destroy is forwarded.
    pub unsafe fn invalidate_view(&mut self, view: vk::ImageView) {
        let references = self.built.as_ref().is_some_and(|built| {
            built.output.view == view || built.taps.iter().any(|tap| tap.source.view == view)
        });
        if references {
            // SAFETY: per this function's contract.
            unsafe { self.destroy_built() };
            // The next trigger rebuilds against the new geometry.
            self.mismatch_warned = false;
        }
    }

    /// Destroy everything. Idempotent.
    ///
    /// # Safety
    /// No submitted work may still reference this compositor's objects (true
    /// at vkDestroyDevice time, where the app must have idled the device).
    pub unsafe fn destroy(&mut self) {
        // SAFETY: per this function's contract.
        unsafe { self.destroy_built() };
    }

    unsafe fn destroy_built(&mut self) {
        if let Some(mut built) = self.built.take() {
            // SAFETY: per the callers' contracts — the objects are idle.
            unsafe {
                built.executor.destroy(&self.device);
                destroy_taps(&self.device, &self.allocator, &mut built.taps);
            }
        }
    }
}

/// # Safety
/// The tap objects must not be referenced by submitted work.
unsafe fn destroy_taps(device: &ash::Device, allocator: &Mutex<Allocator>, taps: &mut Vec<Tap>) {
    let mut allocator = allocator.lock().expect("allocator poisoned");
    for tap in taps.drain(..) {
        // SAFETY: created on `device`; destroying nulls is a no-op.
        unsafe {
            device.destroy_image_view(tap.view, None);
            device.destroy_image(tap.image, None);
        }
        // As in PackExecutor::destroy: nothing to recover into on failure.
        let _ = allocator.free(tap.allocation);
    }
}

#[cfg(test)]
mod tests {
    use ash::vk::Handle;

    use super::*;

    fn attachment(view: u64, format: vk::Format) -> ResolvedAttachment {
        ResolvedAttachment {
            image: vk::Image::from_raw(view * 10),
            view: vk::ImageView::from_raw(view),
            format,
            final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        }
    }

    #[test]
    fn pack_inputs_map_to_the_expected_attachments() {
        let frame = ResolvedFrame {
            extent: vk::Extent2D {
                width: 8,
                height: 8,
            },
            color: attachment(1, vk::Format::R8G8B8A8_UNORM),
            depth: Some(attachment(2, vk::Format::D32_SFLOAT)),
        };
        let (color, color_is_depth) = source_for("game_color", &frame).unwrap();
        assert_eq!(color.view, vk::ImageView::from_raw(1));
        assert!(!color_is_depth);
        let (depth, depth_is_depth) = source_for("game_depth", &frame).unwrap();
        assert_eq!(depth.view, vk::ImageView::from_raw(2));
        assert!(depth_is_depth);
        assert!(source_for("swapchain", &frame).is_none());

        let no_depth = ResolvedFrame {
            depth: None,
            ..frame
        };
        assert!(
            source_for("game_depth", &no_depth).is_none(),
            "a pass without a depth attachment cannot feed game_depth"
        );
    }

    #[test]
    fn aspects_cover_transitions_but_sample_one_plane() {
        assert_eq!(
            full_aspect(vk::Format::D24_UNORM_S8_UINT),
            vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
        );
        assert_eq!(
            sample_aspect(vk::Format::D24_UNORM_S8_UINT),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            full_aspect(vk::Format::D32_SFLOAT),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            sample_aspect(vk::Format::R8G8B8A8_UNORM),
            vk::ImageAspectFlags::COLOR
        );
    }
}
