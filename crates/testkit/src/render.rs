//! Offscreen rendering for golden tests: draw a shader into an RGBA8 target
//! and read the pixels back. Deliberately classic Vulkan (render pass +
//! framebuffer, no 1.3-only features) so it runs on the portability floor.
//!
//! Everything panics on failure: this runs inside tests, in an environment
//! (pinned lavapipe) where a Vulkan error is a broken harness, not a
//! condition to recover from. Leaks on the panic path are fine — the test
//! process is already dead.

use std::ffi::CString;

use ash::vk;
use ferridian_vk_rt::VkRuntime;

use crate::image::RgbaImage;

/// A compiled shader module holding both stages of a draw.
pub struct ShaderSpec<'a> {
    /// SPIR-V words containing both entry points (one slangc module).
    pub spirv: &'a [u32],
    pub vertex_entry: &'a str,
    pub fragment_entry: &'a str,
}

/// One offscreen draw: `vertex_count` vertices, no vertex buffers (fixtures
/// synthesize positions from the vertex index), cleared to `clear_color`.
pub struct RenderSpec<'a> {
    pub width: u32,
    pub height: u32,
    pub clear_color: [f32; 4],
    pub vertex_count: u32,
    pub shader: ShaderSpec<'a>,
    /// Wrap the render pass in a `vkCmd{Begin,End}DebugUtilsLabelEXT` pair —
    /// how layer tests stand in for Blaze3D's debug groups. Requires a
    /// runtime with validation enabled (that's what enables debug utils).
    pub pass_label: Option<&'a str>,
}

const READBACK_TIMEOUT_NS: u64 = 10_000_000_000;

/// Render the spec on the runtime's graphics queue and read back the target.
pub fn render_offscreen(runtime: &VkRuntime, spec: &RenderSpec<'_>) -> RgbaImage {
    let device = runtime.device();
    let extent = vk::Extent2D {
        width: spec.width,
        height: spec.height,
    };
    // SAFETY: one linear sequence of Vulkan calls against a live device; every
    // create is paired with a destroy in reverse order at the end, all work is
    // fence-waited before readback and destruction, and create infos only
    // borrow locals that outlive the calls they're passed to.
    unsafe {
        let memory_properties = runtime
            .instance()
            .get_physical_device_memory_properties(runtime.physical_device());

        // Render target.
        let image_info = vk::ImageCreateInfo::default()
            .image_type(vk::ImageType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .extent(extent.into())
            .mip_levels(1)
            .array_layers(1)
            .samples(vk::SampleCountFlags::TYPE_1)
            .tiling(vk::ImageTiling::OPTIMAL)
            .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC)
            .initial_layout(vk::ImageLayout::UNDEFINED);
        let image = device
            .create_image(&image_info, None)
            .expect("create render target image");
        let image_requirements = device.get_image_memory_requirements(image);
        let image_memory = allocate(
            device,
            &memory_properties,
            image_requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        device
            .bind_image_memory(image, image_memory, 0)
            .expect("bind render target memory");

        let view_info = vk::ImageViewCreateInfo::default()
            .image(image)
            .view_type(vk::ImageViewType::TYPE_2D)
            .format(vk::Format::R8G8B8A8_UNORM)
            .subresource_range(
                vk::ImageSubresourceRange::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .level_count(1)
                    .layer_count(1),
            );
        let view = device
            .create_image_view(&view_info, None)
            .expect("create render target view");

        // Render pass: clear -> draw -> leave the image ready for transfer.
        let attachments = [vk::AttachmentDescription::default()
            .format(vk::Format::R8G8B8A8_UNORM)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)];
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
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];
        let render_pass_info = vk::RenderPassCreateInfo::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        let render_pass = device
            .create_render_pass(&render_pass_info, None)
            .expect("create render pass");

        let framebuffer_views = [view];
        let framebuffer_info = vk::FramebufferCreateInfo::default()
            .render_pass(render_pass)
            .attachments(&framebuffer_views)
            .width(spec.width)
            .height(spec.height)
            .layers(1);
        let framebuffer = device
            .create_framebuffer(&framebuffer_info, None)
            .expect("create framebuffer");

        // Pipeline.
        let module_info = vk::ShaderModuleCreateInfo::default().code(spec.shader.spirv);
        let module = device
            .create_shader_module(&module_info, None)
            .expect("create shader module");
        let vertex_entry = CString::new(spec.shader.vertex_entry).expect("vertex entry point name");
        let fragment_entry =
            CString::new(spec.shader.fragment_entry).expect("fragment entry point name");
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(module)
                .name(&vertex_entry),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(module)
                .name(&fragment_entry),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: spec.width as f32,
            height: spec.height as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent,
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
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let layout_info = vk::PipelineLayoutCreateInfo::default();
        let pipeline_layout = device
            .create_pipeline_layout(&layout_info, None)
            .expect("create pipeline layout");
        let pipeline_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .color_blend_state(&color_blend)
            .layout(pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        let pipelines = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .expect("create graphics pipeline");
        let pipeline = pipelines[0];

        // Readback buffer.
        let readback_size = u64::from(spec.width) * u64::from(spec.height) * 4;
        let buffer_info = vk::BufferCreateInfo::default()
            .size(readback_size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        let buffer = device
            .create_buffer(&buffer_info, None)
            .expect("create readback buffer");
        let buffer_requirements = device.get_buffer_memory_requirements(buffer);
        let buffer_memory = allocate(
            device,
            &memory_properties,
            buffer_requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        device
            .bind_buffer_memory(buffer, buffer_memory, 0)
            .expect("bind readback memory");

        // Record: clear + draw + copy out.
        let pool_info =
            vk::CommandPoolCreateInfo::default().queue_family_index(runtime.queue_family_index());
        let pool = device
            .create_command_pool(&pool_info, None)
            .expect("create command pool");
        let alloc_info = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = device
            .allocate_command_buffers(&alloc_info)
            .expect("allocate command buffer")[0];
        let begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        device
            .begin_command_buffer(command_buffer, &begin_info)
            .expect("begin command buffer");

        let clear_values = [vk::ClearValue {
            color: vk::ClearColorValue {
                float32: spec.clear_color,
            },
        }];
        let render_pass_begin = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            })
            .clear_values(&clear_values);
        let debug_utils = spec.pass_label.map(|label| {
            let name = CString::new(label).expect("pass label has no interior NUL");
            (
                ash::ext::debug_utils::Device::new(runtime.instance(), device),
                name,
            )
        });
        if let Some((debug_utils, name)) = &debug_utils {
            let label = vk::DebugUtilsLabelEXT::default().label_name(name);
            debug_utils.cmd_begin_debug_utils_label(command_buffer, &label);
        }
        device.cmd_begin_render_pass(
            command_buffer,
            &render_pass_begin,
            vk::SubpassContents::INLINE,
        );
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_draw(command_buffer, spec.vertex_count, 1, 0, 0);
        device.cmd_end_render_pass(command_buffer);
        if let Some((debug_utils, _)) = &debug_utils {
            debug_utils.cmd_end_debug_utils_label(command_buffer);
        }

        let copy = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(extent.into());
        device.cmd_copy_image_to_buffer(
            command_buffer,
            image,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            buffer,
            &[copy],
        );
        let host_barrier = vk::BufferMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::HOST_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(buffer)
            .size(vk::WHOLE_SIZE);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::HOST,
            vk::DependencyFlags::empty(),
            &[],
            &[host_barrier],
            &[],
        );
        device
            .end_command_buffer(command_buffer)
            .expect("end command buffer");

        let fence = device
            .create_fence(&vk::FenceCreateInfo::default(), None)
            .expect("create fence");
        let command_buffers = [command_buffer];
        let submit = vk::SubmitInfo::default().command_buffers(&command_buffers);
        device
            .queue_submit(runtime.queue(), &[submit], fence)
            .expect("submit render");
        device
            .wait_for_fences(&[fence], true, READBACK_TIMEOUT_NS)
            .expect("render did not complete within 10s");

        // Read back.
        let mapped = device
            .map_memory(buffer_memory, 0, readback_size, vk::MemoryMapFlags::empty())
            .expect("map readback memory");
        let mut pixels = vec![0u8; readback_size as usize];
        std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), pixels.as_mut_ptr(), pixels.len());
        device.unmap_memory(buffer_memory);

        // Teardown, reverse order.
        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(buffer_memory, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(pipeline_layout, None);
        device.destroy_shader_module(module, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        device.destroy_image_view(view, None);
        device.destroy_image(image, None);
        device.free_memory(image_memory, None);

        RgbaImage::from_pixels(spec.width, spec.height, pixels)
    }
}

/// Allocate memory for `requirements`, preferring `wanted` properties but
/// falling back to any compatible type (lavapipe advertises everything as
/// host-visible anyway).
unsafe fn allocate(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    requirements: vk::MemoryRequirements,
    wanted: vk::MemoryPropertyFlags,
) -> vk::DeviceMemory {
    let pick = |must_have: vk::MemoryPropertyFlags| {
        memory_properties.memory_types[..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .position(|(index, memory_type)| {
                requirements.memory_type_bits & (1 << index) != 0
                    && memory_type.property_flags.contains(must_have)
            })
    };
    let type_index = pick(wanted)
        .or_else(|| pick(vk::MemoryPropertyFlags::empty()))
        .expect("no compatible memory type") as u32;
    let allocate_info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    // SAFETY: valid allocate info; caller frees the memory after fence-waited work.
    unsafe {
        device
            .allocate_memory(&allocate_info, None)
            .expect("allocate memory")
    }
}
