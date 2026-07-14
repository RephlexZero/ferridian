//! Pack execution on a device: where `ferridian-engine`'s pure
//! [`ExecutionPlan`] meets real Vulkan objects (M4).
//!
//! [`PackExecutor::new`] creates everything a wired plan needs — one
//! RGBA16F image per intermediate resource, a descriptor set per pass wired
//! binding-slot → resource view, and one fullscreen-triangle pipeline per
//! pass — and [`PackExecutor::record`] replays the schedule into a command
//! buffer, writers before readers, with the write→sample transition carried
//! by each render pass's final layout and an external subpass dependency.
//!
//! This is runtime code (it will run inside someone's game), so nothing here
//! panics: every Vulkan failure is an [`ExecError`], and a half-built
//! executor tears down what it created before returning the error. Dedicated
//! allocations per image are fine at reference-pack scale; a real allocator
//! (`gpu-allocator`) arrives when pack resources get bigger and reload-churny.

use std::collections::BTreeMap;
use std::ffi::CString;

use ash::vk;
use ferridian_engine::exec::{ExecutionPlan, PassPlan, SWAPCHAIN};

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
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
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
    #[error("no memory type fits type bits {type_bits:#x} for {what}")]
    NoMemoryType { what: &'static str, type_bits: u32 },
    #[error("entry point name {0:?} has an interior NUL byte")]
    BadEntryPointName(String),
}

pub(crate) fn vk_err(what: &'static str) -> impl FnOnce(vk::Result) -> ExecError {
    move |result| ExecError::Vk { what, result }
}

/// One executor-owned image (an intermediate pass resource).
struct Intermediate {
    name: String,
    image: vk::Image,
    memory: vk::DeviceMemory,
    view: vk::ImageView,
}

/// Per-pass device objects, in the order [`PackExecutor::record`] replays.
struct PassResources {
    set_layout: vk::DescriptorSetLayout,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
    framebuffer: vk::Framebuffer,
    module: vk::ShaderModule,
    pipeline: vk::Pipeline,
    /// Freed with the pool.
    descriptor_set: vk::DescriptorSet,
}

/// A plan instantiated on a device, reusable across frames until destroyed.
pub struct PackExecutor {
    extent: vk::Extent2D,
    sampler: vk::Sampler,
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
    /// `SHADER_READ_ONLY_OPTIMAL` and visible to fragment sampling.
    pub fn new(
        ctx: &ExecContext<'_>,
        plan: &ExecutionPlan,
        modules: &BTreeMap<String, Vec<u32>>,
        extent: vk::Extent2D,
        external_inputs: &BTreeMap<String, vk::ImageView>,
        output: &OutputTarget,
    ) -> Result<PackExecutor, ExecError> {
        let mut executor = PackExecutor {
            extent,
            sampler: vk::Sampler::null(),
            intermediates: Vec::new(),
            descriptor_pool: vk::DescriptorPool::null(),
            passes: Vec::new(),
            command_pool: vk::CommandPool::null(),
            fence: vk::Fence::null(),
        };
        match executor.build(ctx, plan, modules, external_inputs, output) {
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
        output: &OutputTarget,
    ) -> Result<(), ExecError> {
        let device = ctx.device;

        // NEAREST, not LINEAR: every sampled resource is a same-extent tap or
        // intermediate read at texel centers, where the two are identical —
        // and depth-format taps are legal to sample NEAREST everywhere, while
        // linear filtering of depth is an optional format feature. Per-binding
        // filter configuration arrives with executor v1.
        let sampler_info = vk::SamplerCreateInfo::default()
            .mag_filter(vk::Filter::NEAREST)
            .min_filter(vk::Filter::NEAREST)
            .address_mode_u(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_v(vk::SamplerAddressMode::CLAMP_TO_EDGE)
            .address_mode_w(vk::SamplerAddressMode::CLAMP_TO_EDGE);
        // SAFETY: create infos below only borrow locals that outlive the
        // call they are passed to; every created handle is stored in `self`,
        // whose destroy() pairs each with the matching destroy call.
        self.sampler = unsafe { device.create_sampler(&sampler_info, None) }
            .map_err(vk_err("create sampler"))?;

        for resource in &plan.intermediates {
            let image_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(INTERMEDIATE_FORMAT)
                .extent(self.extent.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::SAMPLED)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            // The slot is pushed first with null handles and filled as each
            // create succeeds, so a mid-loop failure leaves nothing destroy()
            // can't reach (destroying nulls is a no-op).
            self.intermediates.push(Intermediate {
                name: resource.0.clone(),
                image: vk::Image::null(),
                memory: vk::DeviceMemory::null(),
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
                slot.memory = allocate(
                    device,
                    &ctx.memory_properties,
                    requirements,
                    "intermediate image",
                )?;
                device
                    .bind_image_memory(slot.image, slot.memory, 0)
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

        let total_bindings: u32 = plan
            .passes
            .iter()
            .map(|pass| pass.bindings.len() as u32)
            .sum();
        let pool_sizes = [vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
            .descriptor_count(total_bindings.max(1))];
        let pool_info = vk::DescriptorPoolCreateInfo::default()
            .max_sets(plan.passes.len().max(1) as u32)
            .pool_sizes(&pool_sizes);
        // SAFETY: see the block comment on sampler creation.
        self.descriptor_pool = unsafe { device.create_descriptor_pool(&pool_info, None) }
            .map_err(vk_err("create descriptor pool"))?;

        for pass in &plan.passes {
            let resources = self.build_pass(ctx, pass, modules, &views, output)?;
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
        output: &OutputTarget,
    ) -> Result<PassResources, ExecError> {
        let device = ctx.device;
        let writes_swapchain = pass.output.0 == SWAPCHAIN;
        let (target_view, target_format) = if writes_swapchain {
            (output.view, output.format)
        } else {
            let view = views
                .get(pass.output.0.as_str())
                .copied()
                // Unreachable for a plan from `plan_execution` (every
                // intermediate was just created), kept as an error because
                // this is runtime code.
                .ok_or_else(|| ExecError::MissingExternalInput(pass.output.0.clone()))?;
            (view, INTERMEDIATE_FORMAT)
        };

        let mut resources = PassResources {
            set_layout: vk::DescriptorSetLayout::null(),
            pipeline_layout: vk::PipelineLayout::null(),
            render_pass: vk::RenderPass::null(),
            framebuffer: vk::Framebuffer::null(),
            module: vk::ShaderModule::null(),
            pipeline: vk::Pipeline::null(),
            descriptor_set: vk::DescriptorSet::null(),
        };
        // On failure, hand the partial resources to `self` so destroy()
        // reaches them; a closure-based try block keeps that in one place.
        let result = (|| {
            let layout_bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = pass
                .bindings
                .iter()
                .map(|binding| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(binding.binding)
                        .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                        .descriptor_count(1)
                        .stage_flags(vk::ShaderStageFlags::FRAGMENT)
                })
                .collect();
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

                // Written by exactly one pass (graph-guaranteed), then only
                // sampled — the render pass carries the transition, the exit
                // dependency makes the write visible to consumers.
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
                        vk::PipelineStageFlags::FRAGMENT_SHADER,
                        vk::AccessFlags::SHADER_READ,
                    )
                };
                let attachments = [vk::AttachmentDescription::default()
                    .format(target_format)
                    .samples(vk::SampleCountFlags::TYPE_1)
                    .load_op(vk::AttachmentLoadOp::CLEAR)
                    .store_op(vk::AttachmentStoreOp::STORE)
                    .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
                    .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
                    .initial_layout(vk::ImageLayout::UNDEFINED)
                    .final_layout(final_layout)];
                let color_refs = [vk::AttachmentReference::default()
                    .attachment(0)
                    .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)];
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

                let framebuffer_views = [target_view];
                let framebuffer_info = vk::FramebufferCreateInfo::default()
                    .render_pass(resources.render_pass)
                    .attachments(&framebuffer_views)
                    .width(self.extent.width)
                    .height(self.extent.height)
                    .layers(1);
                resources.framebuffer = device
                    .create_framebuffer(&framebuffer_info, None)
                    .map_err(vk_err("create framebuffer"))?;

                let words = modules
                    .get(&pass.name)
                    .ok_or_else(|| ExecError::MissingModule(pass.name.clone()))?;
                let module_info = vk::ShaderModuleCreateInfo::default().code(words);
                resources.module = device
                    .create_shader_module(&module_info, None)
                    .map_err(vk_err("create shader module"))?;

                let vertex_entry = CString::new(pass.vertex_entry.as_str())
                    .map_err(|_| ExecError::BadEntryPointName(pass.vertex_entry.clone()))?;
                let fragment_entry = CString::new(pass.fragment_entry.as_str())
                    .map_err(|_| ExecError::BadEntryPointName(pass.fragment_entry.clone()))?;
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
                let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
                    .color_write_mask(vk::ColorComponentFlags::RGBA)];
                let color_blend = vk::PipelineColorBlendStateCreateInfo::default()
                    .attachments(&blend_attachments);
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
                            Ok([vk::DescriptorImageInfo::default()
                                .sampler(self.sampler)
                                .image_view(view)
                                .image_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)])
                        })
                        .collect::<Result<_, ExecError>>()?;
                let writes: Vec<vk::WriteDescriptorSet<'_>> = pass
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

    /// Record the whole schedule into `command_buffer`: for each pass, one
    /// render pass, its pipeline + descriptor set, one fullscreen triangle.
    /// This is the seam the layer will call inside an intercepted frame.
    ///
    /// # Safety
    /// `command_buffer` must be in the recording state, on the queue family
    /// this executor was built for, and the executor must stay alive until
    /// the buffer's execution completes.
    pub unsafe fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        for pass in &self.passes {
            let clear_values = [vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 0.0],
                },
            }];
            let begin = vk::RenderPassBeginInfo::default()
                .render_pass(pass.render_pass)
                .framebuffer(pass.framebuffer)
                .render_area(vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent: self.extent,
                })
                .clear_values(&clear_values);
            // SAFETY: all handles were created together on `device` by
            // build(); the caller guarantees the command buffer state.
            unsafe {
                device.cmd_begin_render_pass(command_buffer, &begin, vk::SubpassContents::INLINE);
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
            for intermediate in self.intermediates.drain(..) {
                device.destroy_image_view(intermediate.view, None);
                device.destroy_image(intermediate.image, None);
                device.free_memory(intermediate.memory, None);
            }
            device.destroy_sampler(self.sampler, None);
            self.sampler = vk::Sampler::null();
        }
    }
}

/// Dedicated device-local allocation with fallback to any compatible type
/// (lavapipe advertises everything host-visible).
pub(crate) fn allocate(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    requirements: vk::MemoryRequirements,
    what: &'static str,
) -> Result<vk::DeviceMemory, ExecError> {
    let pick = |must_have: vk::MemoryPropertyFlags| {
        memory_properties.memory_types[..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .position(|(index, memory_type)| {
                requirements.memory_type_bits & (1 << index) != 0
                    && memory_type.property_flags.contains(must_have)
            })
    };
    let type_index = pick(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        .or_else(|| pick(vk::MemoryPropertyFlags::empty()))
        .ok_or(ExecError::NoMemoryType {
            what,
            type_bits: requirements.memory_type_bits,
        })? as u32;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    // SAFETY: valid allocate info; the caller stores and later frees the
    // memory via destroy().
    unsafe { device.allocate_memory(&allocate_info, None) }.map_err(vk_err("allocate memory"))
}
