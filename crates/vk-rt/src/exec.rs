//! Pack execution on a device: where `ferridian-engine`'s pure
//! [`ExecutionPlan`] meets real Vulkan objects (M4).
//!
//! [`PackExecutor::new`] creates everything a wired plan needs — one
//! RGBA16F image per intermediate resource, a descriptor set per pass wired
//! binding-slot → resource view, and one pipeline per pass (a
//! fullscreen-triangle graphics pipeline, or a compute pipeline dispatched
//! over the frame) — and [`PackExecutor::record`] replays the schedule into
//! a command buffer, writers before readers. Graphics writes become sampleable
//! through each render pass's final layout and an external subpass
//! dependency; compute writes go through explicit `GENERAL` ↔
//! `SHADER_READ_ONLY` image barriers around each dispatch.
//!
//! This is runtime code (it will run inside someone's game), so nothing here
//! panics: every Vulkan failure is an [`ExecError`], and a half-built
//! executor tears down what it created before returning the error. Memory
//! comes from a shared `gpu-allocator` instance (owned by whoever builds the
//! [`ExecContext`]) rather than one dedicated allocation per image — the
//! reload-churny pack lifecycle can create and destroy a lot of these.

use std::collections::BTreeMap;
use std::ffi::CString;
use std::sync::{Arc, Mutex};

use ash::vk;
use ferridian_contract::CameraUniforms;
use ferridian_engine::exec::{ExecutionPlan, Filter, PassPlan, SWAPCHAIN, StagePlan};
use gpu_allocator::MemoryLocation;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

/// Format of every executor-created intermediate resource. Float, because
/// passes exchange scene-referred values (the reference pack's `lit`/`fog`
/// exceed [0,1]); universally attachment+sampled capable, portability floor
/// included.
pub const INTERMEDIATE_FORMAT: vk::Format = vk::Format::R16G16B16A16_SFLOAT;

const EXECUTE_TIMEOUT_NS: u64 = 10_000_000_000;

/// The device-side handles execution needs. Deliberately a plain struct of
/// borrowed/copied handles rather than [`crate::VkRuntime`]: inside the layer
/// these come from the intercepted application's device, not our own.
pub struct ExecContext<'a> {
    pub device: &'a ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    /// Shared across every executor/compositor on this device; an `Arc` so
    /// it can outlive this borrowed context (executors keep it for their own
    /// teardown, without needing a context passed back in later).
    pub allocator: Arc<Mutex<Allocator>>,
}

/// Where the pass writing `swapchain` lands. The caller owns the image; the
/// executor's last render pass leaves it in `final_layout`, visible to any
/// later stage (the exit dependency is deliberately broad).
pub struct OutputTarget {
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub final_layout: vk::ImageLayout,
}

#[derive(Debug, thiserror::Error)]
pub enum ExecError {
    #[error("{what} failed: {result}")]
    Vk {
        what: &'static str,
        result: vk::Result,
    },
    #[error("plan samples {0:?} but no such external input was supplied")]
    MissingExternalInput(String),
    #[error("plan schedules pass {0:?} but no module of that name was supplied")]
    MissingModule(String),
    #[error(transparent)]
    Allocation(#[from] gpu_allocator::AllocationError),
    #[error("entry point name {0:?} has an interior NUL byte")]
    BadEntryPointName(String),
    #[error("pass {0:?} binds the camera but no camera buffer was created")]
    CameraMissing(String),
    #[error("camera buffer allocation is not host-mapped")]
    CameraNotMapped,
}

pub(crate) fn vk_err(what: &'static str) -> impl FnOnce(vk::Result) -> ExecError {
    move |result| ExecError::Vk { what, result }
}

/// The contract camera uniform block, one buffer shared by every pass that
/// binds it (host-visible, persistently mapped by gpu-allocator, rewritten
/// via [`PackExecutor::update_camera`]).
struct CameraBuffer {
    buffer: vk::Buffer,
    allocation: Allocation,
}

/// One executor-owned image (an intermediate pass resource).
struct Intermediate {
    name: String,
    image: vk::Image,
    allocation: Allocation,
    view: vk::ImageView,
}

/// A compute pass's dispatch parameters: how many groups cover the frame,
/// and the written image [`PackExecutor::record`] must barrier between
/// `GENERAL` (dispatch) and `SHADER_READ_ONLY_OPTIMAL` (consumers).
struct ComputeDispatch {
    group_counts: [u32; 3],
    output_image: vk::Image,
}

/// Per-pass device objects, in the order [`PackExecutor::record`] replays.
/// `render_pass`/`framebuffer` stay null for compute passes; `compute` is
/// `None` for graphics passes.
struct PassResources {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    /// Freed with the pool.
    descriptor_set: vk::DescriptorSet,
    /// One clear per color attachment (empty for compute passes) — the
    /// render pass begin must supply exactly as many as it declared.
    clear_values: Vec<vk::ClearValue>,
    compute: Option<ComputeDispatch>,
}

/// A plan instantiated on a device, reusable across frames until destroyed.
pub struct PackExecutor {
    extent: vk::Extent2D,
    /// Cloned from the [`ExecContext`] that built this executor, so
    /// [`PackExecutor::destroy`] can free every allocation without needing a
    /// context passed back in.
    allocator: Arc<Mutex<Allocator>>,
    sampler_nearest: vk::Sampler,
    sampler_linear: vk::Sampler,
    camera: Option<CameraBuffer>,
    intermediates: Vec<Intermediate>,
    descriptor_pool: vk::DescriptorPool,
    passes: Vec<PassResources>,
    command_pool: vk::CommandPool,
    fence: vk::Fence,
}

impl PackExecutor {
    /// Instantiate `plan` on the device. `modules` are the loaded pack's
    /// SPIR-V words per pass; `external_inputs` must contain a view per
    /// resource in `plan.external_inputs`, already in
    /// `SHADER_READ_ONLY_OPTIMAL` and visible to fragment sampling; `camera`
    /// is the initial value of the contract camera block (updated later via
    /// [`PackExecutor::update_camera`]) — ignored if no pass binds it.
    pub fn new(
        ctx: &ExecContext<'_>,
        plan: &ExecutionPlan,
        modules: &BTreeMap<String, Vec<u32>>,
        extent: vk::Extent2D,
        external_inputs: &BTreeMap<String, vk::ImageView>,
        camera: &CameraUniforms,
        output: &OutputTarget,
    ) -> Result<PackExecutor, ExecError> {
        let mut executor = PackExecutor {
            extent,
            allocator: ctx.allocator.clone(),
            sampler_nearest: vk::Sampler::null(),
            sampler_linear: vk::Sampler::null(),
            camera: None,
            intermediates: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            passes: Vec::new(),
            command_pool: vk::CommandPool::null(),
            fence: vk::Fence::null(),
        };
        match executor.build(ctx, plan, modules, external_inputs, camera, output) {
            Ok(()) => Ok(executor),
            Err(error) => {
                // SAFETY: nothing has been submitted yet, so no in-flight
                // work references the partially built objects.
                unsafe { executor.destroy(ctx.device) };
                Err(error)
            }
        }
    }

    fn build(
        &mut self,
        ctx: &ExecContext<'_>,
        plan: &ExecutionPlan,
        modules: &BTreeMap<String, Vec<u32>>,
        external_inputs: &BTreeMap<String, vk::ImageView>,
        camera: &CameraUniforms,
        output: &OutputTarget,
    ) -> Result<(), ExecError> {
        let device = ctx.device;

        // One sampler per manifest filter mode, shared by every binding that
        // asks for it. NEAREST is the default (same-extent reads at texel
        // centers are filter-invariant, and depth-format taps are legal to
        // sample NEAREST everywhere, while linear filtering of depth is an
        // optional format feature); LINEAR is opt-in per binding via the
        // pass's `filters` table.
        let sampler_info = |filter: vk::Filter| {
            vk::SamplerCreateInfo::default()
                .mag_filter(filter)
                .min_filter(filter)
                .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
                .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE)
        };
        // SAFETY: create infos below only borrow locals that outlive the
        // call they are passed to; every created handle is stored in `self`,
        // whose destroy() pairs each with the matching destroy call.
        self.sampler_nearest =
            unsafe { device.create_sampler(&sampler_info(vk::Filter::NEAREST), None) }
                .map_err(vk_err("create nearest sampler"))?;
        // SAFETY: as above.
        self.sampler_linear =
            unsafe { device.create_sampler(&sampler_info(vk::Filter::LINEAR), None) }
                .map_err(vk_err("create linear sampler"))?;

        // How an intermediate is written decides its usage: compute outputs
        // are storage images, graphics outputs are color attachments (the
        // graph guarantees exactly one writer). RGBA16F supports both
        // everywhere — storage on it is in Vulkan's required-format table.
        let compute_written: Vec<&str> = plan
            .passes
            .iter()
            .filter(|pass| matches!(pass.stage, StagePlan::Compute { .. }))
            .filter_map(|pass| pass.outputs.first().map(|output| output.0.as_str()))
            .collect();
        for resource in &plan.intermediates {
            let write_usage = if compute_written.contains(&resource.0.as_str()) {
                vk::ImageUsageFlags::STORAGE
            } else {
                vk::ImageUsageFlags::COLOR_ATTACHMENT
            };
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(INTERMEDIATE_FORMAT)
                .extent(self.extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(write_usage | vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // The slot is pushed first with null handles and filled as each
            // create succeeds, so a mid-loop failure leaves nothing destroy()
            // can't reach (destroying nulls is a no-op).
            self.intermediates.push(Intermediate {
                name: resource.0.clone(),
                image: vk::Image::null(),
                allocation: Allocation::default(),
                view: vk::ImageView::null(),
            });
            let Some(slot) = self.intermediates.last_mut() else {
                unreachable!("pushed above");
            };
            // SAFETY: see the block comment on sampler creation.
            unsafe {
                slot.image = device
                    .create_image(&image_info, None)
                    .map_err(vk_err("create intermediate image"))?;
                let requirements = device.get_image_memory_requirements(slot.image);
                slot.allocation = self
                    .allocator
                    .lock()
                    .expect("allocator poisoned")
                    .allocate(&AllocationCreateDesc {
                        name: &format!("ferridian intermediate {}", slot.name),
                        requirements,
                        location: MemoryLocation::GpuOnly,
                        linear: false,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })?;
                device
                    .bind_image_memory(
                        slot.image,
                        slot.allocation.memory(),
                        slot.allocation.offset(),
                    )
                    .map_err(vk_err("bind intermediate memory"))?;
                let view_info = vk::ImageViewCreateInfo::default()
                    .image(slot.image)
                    .view_type(vk::ImageViewType::TYPE_2D)
                    .format(INTERMEDIATE_FORMAT)
                    .subresource_range(
                        vk::ImageSubresourceRange::default()
                            .aspect_mask(vk::ImageAspectFlags::COLOR)
                            .level_count(1)
                            .layer_count(1),
                    );
                slot.view = device
                    .create_image_view(&view_info, None)
                    .map_err(vk_err("create intermediate view"))?;
            }
        }

        // Resource name -> sampled view, for descriptor writes. Owns its
        // keys so it doesn't hold a borrow of `self.intermediates`.
        let mut views: BTreeMap<String, vk::ImageView> = self
            .intermediates
            .iter()
            .map(|i| (i.name.clone(), i.view))
            .collect();
        for (name, &view) in external_inputs {
            views.insert(name.clone(), view);
        }
        // Resource name -> image, for compute passes' dispatch barriers.
        let images: BTreeMap<String, vk::Image> = self
            .intermediates
            .iter()
            .map(|i| (i.name.clone(), i.image))
            .collect();

        // One camera buffer serves every pass that binds the block.
        let camera_bindings: u32 = plan
            .passes
            .iter()
            .filter(|pass| pass.camera_binding.is_some())
            .count() as u32;
        if camera_bindings > 0 {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(CameraUniforms::STD140_SIZE as u64)
                .usage(vk::BufferUsageFlags::UNIFORM_BUFFER);
            // Slot-first like the intermediates: destroy() reaches whatever
            // was created if a later step fails.
            self.camera = Some(CameraBuffer {
                buffer: vk::Buffer::null(),
                allocation: Allocation::default(),
            });
            let Some(slot) = self.camera.as_mut() else {
                unreachable!("assigned above");
            };
            // SAFETY: see the block comment on sampler creation.
            unsafe {
                slot.buffer = device
                    .create_buffer(&buffer_info, None)
                    .map_err(vk_err("create camera buffer"))?;
                let requirements = device.get_buffer_memory_requirements(slot.buffer);
                // CpuToGpu: rewritten from the CPU each camera update, read as
                // a 32-byte uniform block — device-local placement buys
                // nothing at this size, and gpu-allocator persistently maps
                // host-visible allocations for us.
                slot.allocation = self
                    .allocator
                    .lock()
                    .expect("allocator poisoned")
                    .allocate(&AllocationCreateDesc {
                        name: "ferridian camera uniform buffer",
                        requirements,
                        location: MemoryLocation::CpuToGpu,
                        linear: true,
                        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                    })?;
                device
                    .bind_buffer_memory(
                        slot.buffer,
                        slot.allocation.memory(),
                        slot.allocation.offset(),
                    )
                    .map_err(vk_err("bind camera memory"))?;
                // Nothing is submitted yet, so the initial write is safe.
                self.update_camera(camera)?;
            }
        }

        let total_bindings: u32 = plan
            .passes
            .iter()
            .map(|pass| pass.bindings.len() as u32)
            .sum();
        let mut pool_sizes = vec![
            vk::DescriptorPoolSize::default()
                .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .descriptor_count(total_bindings.max(1)),
        ];
        if camera_bindings > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(camera_bindings),
            );
        }
        // One storage-image descriptor per compute pass (its output).
        let storage_bindings: u32 = plan
            .passes
            .iter()
            .filter(|pass| pass.output_binding.is_some())
            .count() as u32;
        if storage_bindings > 0 {
            pool_sizes.push(
                vk::DescriptorPoolSize::default()
                    .ty(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(storage_bindings),
            );
        }
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(plan.passes.len().max(1) as u32)
            .pool_sizes(&pool_sizes);
        // SAFETY: see the block comment on sampler creation.
        self.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(vk_err("create descriptor pool"))?;

        for pass in &plan.passes {
            let resources = self.build_pass(ctx, pass, modules, &views, &images, output)?;
            self.passes.push(resources);
        }

        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(ctx.queue_family_index);
        // SAFETY: see the block comment on sampler creation.
        self.command_pool = unsafe { device.create_command_pool(&pool_info, None) }
            .map_err(vk_err("create command pool"))?;
        // SAFETY: see the block comment on sampler creation.
        self.fence = unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }
            .map_err(vk_err("create fence"))?;
        Ok(())
    }

    fn build_pass(
        &mut self,
        ctx: &ExecContext<'_>,
        pass: &PassPlan,
        modules: &BTreeMap<String, Vec<u32>>,
        views: &BTreeMap<String, vk::ImageView>,
        images: &BTreeMap<String, vk::Image>,
        output: &OutputTarget,
    ) -> Result<PassResources, ExecError> {
        let device = ctx.device;
        let writes_swapchain = pass.outputs.iter().any(|resource| resource.0 == SWAPCHAIN);
        // One (view, format) per output — the color attachments of a
        // graphics pass, in the plan's location order. The planner keeps the
        // swapchain writer single-output, so mixed formats never happen.
        let targets: Vec<(vk::ImageView, vk::Format)> = pass
            .outputs
            .iter()
            .map(|resource| {
                if resource.0 == SWAPCHAIN {
                    return Ok((output.view, output.format));
                }
                views
                    .get(resource.0.as_str())
                    .copied()
                    // Unreachable for a plan from `plan_execution` (every
                    // intermediate was just created), kept as an error
                    // because this is runtime code.
                    .ok_or_else(|| ExecError::MissingExternalInput(resource.0.clone()))
                    .map(|view| (view, INTERMEDIATE_FORMAT))
            })
            .collect::<Result<_, ExecError>>()?;
        let shader_stages = match pass.stage {
            StagePlan::Graphics { .. } => vk::ShaderStageFlags::FRAGMENT,
            StagePlan::Compute { .. } => vk::ShaderStageFlags::COMPUTE,
        };

        let mut resources = PassResources {
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_set: vk::DescriptorSet::null(),
            clear_values: Vec::new(),
            compute: None,
        };
        // On failure, hand the partial resources to `self` so destroy()
        // reaches them; a closure-based try block keeps that in one place.
        let result = (|| {
            let mut layout_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = pass
                .bindings
                .iter()
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding.binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(shader_stages)
                })
                .collect();
            if let Some(slot) = pass.camera_binding {
                let camera_stages = match pass.stage {
                    // Fragment-only today, but vertex access is free to
                    // declare and packs will want it.
                    StagePlan::Graphics { .. } => {
                        vk::ShaderStageFlags::FRAGMENT | vk::ShaderStageFlags::VERTEX
                    }
                    StagePlan::Compute { .. } => vk::ShaderStageFlags::COMPUTE,
                };
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(slot)
                        .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                        .descriptor_count(1)
                        .stage_flags(camera_stages),
                );
            }
            if let Some(slot) = pass.output_binding {
                layout_bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(slot)
                        .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            let set_layout_info =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&layout_bindings);
            // SAFETY: one linear sequence of creates against a live device;
            // create infos only borrow locals that outlive each call, and
            // every handle lands in `resources`, destroyed by destroy().
            unsafe {
                resources.set_layout = device
                    .create_descriptor_set_layout(&set_layout_info, None)
                    .map_err(vk_err("create descriptor set layout"))?;

                let set_layouts = [resources.set_layout];
                let pipeline_layout_info =
                    vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
                resources.pipeline_layout = device
                    .create_pipeline_layout(&pipeline_layout_info, None)
                    .map_err(vk_err("create pipeline layout"))?;

                let words = modules
                    .get(&pass.name)
                    .ok_or_else(|| ExecError::MissingModule(pass.name.clone()))?;
                let module_info = vk::ShaderModuleCreateInfo::default().code(words);
                resources.module = device
                    .create_shader_module(&module_info, None)
                    .map_err(vk_err("create shader module"))?;

                match &pass.stage {
                    StagePlan::Graphics {
                        vertex_entry,
                        fragment_entry,
                    } => self.build_graphics_pipeline(
                        device,
                        &mut resources,
                        vertex_entry,
                        fragment_entry,
                        &targets,
                        writes_swapchain,
                        output,
                    )?,
                    StagePlan::Compute {
                        entry,
                        workgroup_size,
                    } => {
                        let entry_name = CString::new(entry.as_str())
                            .map_err(|_| ExecError::BadEntryPointName(entry.clone()))?;
                        let stage_info = vk::PipelineShaderStageCreateInfo::default()
                            .stage(vk::ShaderStageFlags::COMPUTE)
                            .module(resources.module)
                            .name(&entry_name);
                        let pipeline_info = vk::ComputePipelineCreateInfo::default()
                            .stage(stage_info)
                            .layout(resources.pipeline_layout);
                        resources.pipeline = device
                            .create_compute_pipelines(
                                vk::PipelineCache::null(),
                                &[pipeline_info],
                                None,
                            )
                            .map_err(|(_, result)| vk_err("create compute pipeline")(result))?[0];
                        let output_image = pass
                            .outputs
                            .first()
                            .and_then(|resource| images.get(resource.0.as_str()))
                            .copied()
                            // Unreachable as for targets above: a compute
                            // output is always a single intermediate (the
                            // planner rejects compute→swapchain and MRT).
                            .ok_or_else(|| ExecError::MissingExternalInput(pass.name.clone()))?;
                        resources.compute = Some(ComputeDispatch {
                            group_counts: [
                                self.extent.width.div_ceil(workgroup_size[0].max(1)),
                                self.extent.height.div_ceil(workgroup_size[1].max(1)),
                                1,
                            ],
                            output_image,
                        });
                    }
                }

                let set_layouts = [resources.set_layout];
                let alloc_info = vk::DescriptorSetAllocateInfo::default()
                    .descriptor_pool(self.descriptor_pool)
                    .set_layouts(&set_layouts);
                resources.descriptor_set = device
                    .allocate_descriptor_sets(&alloc_info)
                    .map_err(vk_err("allocate descriptor set"))?[0];

                let image_infos: Vec<[vk::DescriptorImageInfo; 1]> =
                    pass.bindings
                        .iter()
                        .map(|binding| {
                            let view = views.get(binding.resource.0.as_str()).copied().ok_or_else(
                                || ExecError::MissingExternalInput(binding.resource.0.clone()),
                            )?;
                            let sampler = match binding.filter {
                                Filter::Nearest => self.sampler_nearest,
                                Filter::Linear => self.sampler_linear,
                            };
                            Ok([vk::DescriptorImageInfo::default()
                                .sampler(sampler)
                                .image_view(view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)])
                        })
                        .collect::<Result<_, ExecError>>()?;
                let camera_info = match pass.camera_binding {
                    Some(_) => {
                        let camera = self
                            .camera
                            .as_ref()
                            // Unreachable: build() created the buffer from
                            // the same plan. Runtime code, so still an error.
                            .ok_or_else(|| ExecError::CameraMissing(pass.name.clone()))?;
                        Some([vk::DescriptorBufferInfo::default()
                            .buffer(camera.buffer)
                            .offset(0)
                            .range(CameraUniforms::STD140_SIZE as u64)])
                    }
                    None => None,
                };
                let mut writes: Vec<vk::WriteDescriptorSet<'_>> = pass
                    .bindings
                    .iter()
                    .zip(&image_infos)
                    .map(|(binding, info)| {
                        vk::WriteDescriptorSet::default()
                            .dst_set(resources.descriptor_set)
                            .dst_binding(binding.binding)
                            .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                            .image_info(info)
                    })
                    .collect();
                if let (Some(slot), Some(info)) = (pass.camera_binding, camera_info.as_ref()) {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(resources.descriptor_set)
                            .dst_binding(slot)
                            .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                            .buffer_info(info),
                    );
                }
                // The compute output: written in GENERAL (record() puts the
                // image there before the dispatch).
                let output_info = pass
                    .output_binding
                    .map(|_| {
                        let view = pass
                            .outputs
                            .first()
                            .and_then(|resource| views.get(resource.0.as_str()))
                            .copied()
                            .ok_or_else(|| ExecError::MissingExternalInput(pass.name.clone()))?;
                        Ok::<_, ExecError>([vk::DescriptorImageInfo::default()
                            .image_view(view)
                            .image_layout(vk::ImageLayout::GENERAL)])
                    })
                    .transpose()?;
                if let (Some(slot), Some(info)) = (pass.output_binding, output_info.as_ref()) {
                    writes.push(
                        vk::WriteDescriptorSet::default()
                            .dst_set(resources.descriptor_set)
                            .dst_binding(slot)
                            .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                            .image_info(info),
                    );
                }
                device.update_descriptor_sets(&writes, &[]);
            }
            Ok(())
        })();
        match result {
            Ok(()) => Ok(resources),
            Err(error) => {
                self.passes.push(resources);
                Err(error)
            }
        }
    }

    /// The graphics half of [`PackExecutor::build_pass`]: render pass,
    /// framebuffer, and the fullscreen-triangle pipeline over one color
    /// attachment per target (fragment location order). `resources.module`
    /// and `resources.pipeline_layout` must already be created.
    #[expect(
        clippy::too_many_arguments,
        reason = "private continuation of build_pass, not an API"
    )]
    fn build_graphics_pipeline(
        &self,
        device: &ash::Device,
        resources: &mut PassResources,
        vertex_entry: &str,
        fragment_entry: &str,
        targets: &[(vk::ImageView, vk::Format)],
        writes_swapchain: bool,
        output: &OutputTarget,
    ) -> Result<(), ExecError> {
        // Each target is written by exactly one pass (graph-guaranteed),
        // then only sampled — the render pass carries the transition, the
        // exit dependency makes the writes visible to consumers.
        let final_layout = if writes_swapchain {
            output.final_layout
        } else {
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        };
        let (exit_stage, exit_access) = if writes_swapchain {
            (
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
            )
        } else {
            (
                // The consuming pass may sample from a fragment shader or a
                // compute dispatch.
                vk::PipelineStageFlags::FRAGMENT_SHADER | vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::AccessFlags::SHADER_READ,
            )
        };
        // SAFETY: as in build_pass — create infos borrow locals that outlive
        // each call; every handle lands in `resources`.
        unsafe {
            let attachments: Vec<vk::AttachmentDescription> = targets
                .iter()
                .map(|&(_, format)| {
                    vk::AttachmentDescription::default()
                        .format(format)
                        .samples(vk::SampleCountFlags::TYPE_1)
                        .load_op(vk::AttachmentLoadOp::CLEAR)
                        .store_op(vk::AttachmentStoreOp::STORE)
                        .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                        .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                        .initial_layout(vk::ImageLayout::UNDEFINED)
                        .final_layout(final_layout)
                })
                .collect();
            let color_refs: Vec<vk::AttachmentReference> = (0..targets.len() as u32)
                .map(|location| {
                    vk::AttachmentReference::default()
                        .attachment(location)
                        .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                })
                .collect();
            let subpasses = [vk::SubpassDescription::default()
                .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
                .color_attachments(&color_refs)];
            let dependencies = [
                vk::SubpassDependency::default()
                    .src_subpass(vk::SUBPASS_EXTERNAL)
                    .dst_subpass(0)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE),
                vk::SubpassDependency::default()
                    .src_subpass(0)
                    .dst_subpass(vk::SUBPASS_EXTERNAL)
                    .src_stage_mask(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT)
                    .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                    .dst_stage_mask(exit_stage)
                    .dst_access_mask(exit_access),
            ];
            let render_pass_info = vk::RenderPassCreateInfo::default()
                .attachments(&attachments)
                .subpasses(&subpasses)
                .dependencies(&dependencies);
            resources.render_pass = device
                .create_render_pass(&render_pass_info, None)
                .map_err(vk_err("create render pass"))?;

            let framebuffer_views: Vec<vk::ImageView> =
                targets.iter().map(|&(view, _)| view).collect();
            let framebuffer_info = vk::FramebufferCreateInfo::default()
                .render_pass(resources.render_pass)
                .attachments(&framebuffer_views)
                .width(self.extent.width)
                .height(self.extent.height)
                .layers(1);
            resources.framebuffer = device
                .create_framebuffer(&framebuffer_info, None)
                .map_err(vk_err("create framebuffer"))?;

            let vertex_entry = CString::new(vertex_entry)
                .map_err(|_| ExecError::BadEntryPointName(vertex_entry.to_owned()))?;
            let fragment_entry = CString::new(fragment_entry)
                .map_err(|_| ExecError::BadEntryPointName(fragment_entry.to_owned()))?;
            let stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(resources.module)
                    .name(&vertex_entry),
                vk::PipelineShaderStageCreateInfo::default()
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(resources.module)
                    .name(&fragment_entry),
            ];
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let viewports = [vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.extent.width as f32,
                height: self.extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            }];
            let scissors = [vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: self.extent,
            }];
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewports(&viewports)
                .scissors(&scissors);
            let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let multisample = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let blend_attachments = vec![
                vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA);
                targets.len()
            ];
            let color_blend =
                vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
            let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&stages)
                .vertex_input_state(&vertex_input)
                .input_assembly_state(&input_assembly)
                .viewport_state(&viewport_state)
                .rasterization_state(&rasterization)
                .multisample_state(&multisample)
                .color_blend_state(&color_blend)
                .layout(resources.pipeline_layout)
                .render_pass(resources.render_pass)
                .subpass(0);
            resources.pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_, result)| vk_err("create graphics pipeline")(result))?[0];
        }
        resources.clear_values = vec![
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            };
            targets.len()
        ];
        Ok(())
    }

    /// Overwrite the contract camera block. A no-op when the pack never
    /// binds the camera.
    ///
    /// # Safety
    /// No submitted work may still be reading the buffer — either wait the
    /// frame that sampled it (as [`PackExecutor::execute`]'s fence does) or
    /// only call this before work is submitted.
    pub unsafe fn update_camera(&self, camera: &CameraUniforms) -> Result<(), ExecError> {
        let Some(buffer) = &self.camera else {
            return Ok(());
        };
        let bytes = camera.to_std140_bytes();
        // gpu-allocator persistently maps CpuToGpu allocations, so the
        // pointer is already valid — no map/unmap needed (and calling
        // vkMapMemory ourselves on already-mapped memory would be invalid).
        let ptr = buffer
            .allocation
            .mapped_ptr()
            .ok_or(ExecError::CameraNotMapped)?;
        // SAFETY: the mapping is exactly STD140_SIZE bytes host-visible
        // memory; the caller guarantees no in-flight readers.
        unsafe {
            std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr().cast::<u8>(), bytes.len());
        }
        Ok(())
    }

    /// Record the whole schedule into `command_buffer`: for each graphics
    /// pass one render pass, pipeline + descriptor set, one fullscreen
    /// triangle; for each compute pass a `GENERAL` transition, the dispatch,
    /// and a release into `SHADER_READ_ONLY_OPTIMAL`. This is the seam the
    /// layer calls inside an intercepted frame.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state, on the queue family
    /// this executor was built for, and the executor must stay alive until
    /// the buffer's execution completes.
    pub unsafe fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        for pass in &self.passes {
            match &pass.compute {
                None => {
                    let begin = vk::RenderPassBeginInfo::default()
                        .render_pass(pass.render_pass)
                        .framebuffer(pass.framebuffer)
                        .render_area(vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: self.extent,
                        })
                        .clear_values(&pass.clear_values);
                    // SAFETY: all handles were created together on `device` by
                    // build(); the caller guarantees the command buffer state.
                    unsafe {
                        device.cmd_begin_render_pass(
                            command_buffer,
                            &begin,
                            vk::SubpassContents::INLINE,
                        );
                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pass.pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::GRAPHICS,
                            pass.pipeline_layout,
                            0,
                            &[pass.descriptor_set],
                            &[],
                        );
                        device.cmd_draw(command_buffer, 3, 1, 0, 0);
                        device.cmd_end_render_pass(command_buffer);
                    }
                }
                Some(dispatch) => {
                    let range = vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1);
                    // UNDEFINED discards last frame's contents (the dispatch
                    // rewrites every texel); the source stages order the
                    // write-after-read against the previous frame's samplers.
                    let acquire = vk::ImageMemoryBarrier::default()
                        .image(dispatch.output_image)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .new_layout(vk::ImageLayout::GENERAL)
                        .src_access_mask(vk::AccessFlags::empty())
                        .dst_access_mask(
                            vk::AccessFlags::SHADER_WRITE | vk::AccessFlags::SHADER_READ,
                        )
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .subresource_range(range);
                    let release = vk::ImageMemoryBarrier::default()
                        .image(dispatch.output_image)
                        .old_layout(vk::ImageLayout::GENERAL)
                        .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                        .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                        .dst_access_mask(vk::AccessFlags::SHADER_READ)
                        .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                        .subresource_range(range);
                    // SAFETY: as in the graphics arm.
                    unsafe {
                        device.cmd_pipeline_barrier(
                            command_buffer,
                            vk::PipelineStageFlags::FRAGMENT_SHADER
                                | vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[acquire],
                        );
                        device.cmd_bind_pipeline(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pass.pipeline,
                        );
                        device.cmd_bind_descriptor_sets(
                            command_buffer,
                            vk::PipelineBindPoint::COMPUTE,
                            pass.pipeline_layout,
                            0,
                            &[pass.descriptor_set],
                            &[],
                        );
                        device.cmd_dispatch(
                            command_buffer,
                            dispatch.group_counts[0],
                            dispatch.group_counts[1],
                            dispatch.group_counts[2],
                        );
                        device.cmd_pipeline_barrier(
                            command_buffer,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::FRAGMENT_SHADER
                                | vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[],
                            &[],
                            &[release],
                        );
                    }
                }
            }
        }
    }

    /// Run the schedule once on the context's queue and wait for completion.
    pub fn execute(&self, ctx: &ExecContext<'_>) -> Result<(), ExecError> {
        let device = ctx.device;
        // SAFETY: pool/fence belong to this executor on this device; the
        // fence wait below means nothing is in flight when the pool is reset
        // on the next call; the recorded buffer lives until that reset.
        unsafe {
            device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .map_err(vk_err("reset command pool"))?;
            let alloc_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            let command_buffer = device
                .allocate_command_buffers(&alloc_info)
                .map_err(vk_err("allocate command buffer"))?[0];
            let begin_info = vk::CommandBufferBeginInfo::default()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device
                .begin_command_buffer(command_buffer, &begin_info)
                .map_err(vk_err("begin command buffer"))?;
            self.record(device, command_buffer);
            device
                .end_command_buffer(command_buffer)
                .map_err(vk_err("end command buffer"))?;
            let command_buffers = [command_buffer];
            let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
            device
                .queue_submit(ctx.queue, &[submit], self.fence)
                .map_err(vk_err("submit"))?;
            device
                .wait_for_fences(&[self.fence], true, EXECUTE_TIMEOUT_NS)
                .map_err(vk_err("wait for execution"))?;
            device
                .reset_fences(&[self.fence])
                .map_err(vk_err("reset fence"))?;
        }
        Ok(())
    }

    /// Destroy every device object this executor created. Idempotent
    /// (destroying nulls is a no-op).
    ///
    /// # Safety
    /// No submitted work may still reference this executor's objects —
    /// [`PackExecutor::execute`] waits its fence, so after it returns this is
    /// safe unless the caller recorded via [`PackExecutor::record`] and has
    /// its own submission in flight.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: every handle was created on `device` by build() and is
        // destroyed exactly once (nulled by take/drain semantics below).
        unsafe {
            device.destroy_fence(self.fence, None);
            self.fence = vk::Fence::null();
            device.destroy_command_pool(self.command_pool, None);
            self.command_pool = vk::CommandPool::null();
            for pass in self.passes.drain(..) {
                device.destroy_pipeline(pass.pipeline, None);
                device.destroy_shader_module(pass.module, None);
                device.destroy_framebuffer(pass.framebuffer, None);
                device.destroy_render_pass(pass.render_pass, None);
                device.destroy_pipeline_layout(pass.pipeline_layout, None);
                device.destroy_descriptor_set_layout(pass.set_layout, None);
            }
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.descriptor_pool = vk::DescriptorPool::null();
            let mut allocator = self.allocator.lock().expect("allocator poisoned");
            for intermediate in self.intermediates.drain(..) {
                device.destroy_image_view(intermediate.view, None);
                device.destroy_image(intermediate.image, None);
                // A free() failure here would only mean a leaked block in an
                // allocator we're not tearing down (the device may outlive
                // this executor) — nothing to recover into, so it's dropped.
                let _ = allocator.free(intermediate.allocation);
            }
            if let Some(camera) = self.camera.take() {
                device.destroy_buffer(camera.buffer, None);
                let _ = allocator.free(camera.allocation);
            }
            drop(allocator);
            device.destroy_sampler(self.sampler_nearest, None);
            self.sampler_nearest = vk::Sampler::null();
            device.destroy_sampler(self.sampler_linear, None);
            self.sampler_linear = vk::Sampler::null();
        }
    }
}
