//! GPU texture fixtures: upload a CPU image as a sampleable texture, create
//! render targets, and read images back. Same conventions as `render`:
//! classic Vulkan, panics on failure (tests only), fence-waited submissions.

use ash::vk;
use ferridian_vk_rt::VkRuntime;

use crate::image::RgbaImage;
use crate::render::allocate;

const SUBMIT_TIMEOUT_NS: u64 = 10_000_000_000;

/// An image + its memory and a full view, owned by the test.
pub struct GpuImage {
    pub image: vk::Image,
    pub view: vk::ImageView,
    memory: vk::DeviceMemory,
}

impl GpuImage {
    /// Destroy the image, view, and memory. Call after all GPU work using
    /// them has completed.
    pub fn destroy(&mut self, device: &ash::Device) {
        // SAFETY: handles were created together on `device` by this module
        // and are destroyed exactly once (then nulled, making this idempotent).
        unsafe {
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
            device.free_memory(self.memory, None);
        }
        self.view = vk::ImageView::null();
        self.image = vk::Image::null();
        self.memory = vk::DeviceMemory::null();
    }
}

/// Create an RGBA8 image in `UNDEFINED` layout with the given usage — e.g. a
/// `COLOR_ATTACHMENT | TRANSFER_SRC` target for an executor's swapchain slot.
pub fn create_target(
    runtime: &VkRuntime,
    width: u32,
    height: u32,
    usage: vk::ImageUsageFlags,
) -> GpuImage {
    let device = runtime.device();
    let info = vk::ImageCreateInfo::default()
        .image_type(vk::ImageType::TYPE_2D)
        .format(vk::Format::R8G8B8A8_UNORM)
        .extent(vk::Extent2D { width, height }.into())
        .mip_levels(1)
        .array_layers(1)
        .samples(vk::SampleCountFlags::TYPE_1)
        .tiling(vk::ImageTiling::OPTIMAL)
        .usage(usage)
        .initial_layout(vk::ImageLayout::UNDEFINED);
    // SAFETY: a linear create-allocate-bind-view sequence on a live device;
    // create infos borrow locals that outlive each call; the caller owns the
    // returned handles via GpuImage::destroy.
    unsafe {
        let memory_properties = runtime
            .instance()
            .get_physical_device_memory_properties(runtime.physical_device());
        let image = device.create_image(&info, None).expect("create image");
        let requirements = device.get_image_memory_requirements(image);
        let memory = allocate(
            device,
            &memory_properties,
            requirements,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        );
        device
            .bind_image_memory(image, memory, 0)
            .expect("bind image memory");
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
            .expect("create image view");
        GpuImage {
            image,
            view,
            memory,
        }
    }
}

/// Upload `pixels` as an RGBA8 sampled texture, left in
/// `SHADER_READ_ONLY_OPTIMAL` visible to fragment sampling — the shape
/// executor external inputs (`game_color`, `game_depth`) expect.
pub fn upload_texture(runtime: &VkRuntime, pixels: &RgbaImage) -> GpuImage {
    let device = runtime.device();
    let target = create_target(
        runtime,
        pixels.width,
        pixels.height,
        vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
    );
    // SAFETY: one fence-waited upload submission; every transient object is
    // destroyed after the wait; create infos borrow locals that outlive the
    // calls they're passed to.
    unsafe {
        let memory_properties = runtime
            .instance()
            .get_physical_device_memory_properties(runtime.physical_device());

        let staging_info = vk::BufferCreateInfo::default()
            .size(pixels.pixels.len() as u64)
            .usage(vk::BufferUsageFlags::TRANSFER_SRC);
        let staging = device
            .create_buffer(&staging_info, None)
            .expect("create staging buffer");
        let requirements = device.get_buffer_memory_requirements(staging);
        let staging_memory = allocate(
            device,
            &memory_properties,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        device
            .bind_buffer_memory(staging, staging_memory, 0)
            .expect("bind staging memory");
        let mapped = device
            .map_memory(
                staging_memory,
                0,
                pixels.pixels.len() as u64,
                vk::MemoryMapFlags::empty(),
            )
            .expect("map staging memory");
        std::ptr::copy_nonoverlapping(
            pixels.pixels.as_ptr(),
            mapped.cast::<u8>(),
            pixels.pixels.len(),
        );
        device.unmap_memory(staging_memory);

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
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin command buffer");

        let range = vk::ImageSubresourceRange::default()
            .aspect_mask(vk::ImageAspectFlags::COLOR)
            .level_count(1)
            .layer_count(1);
        let to_transfer = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::empty())
            .dst_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .old_layout(vk::ImageLayout::UNDEFINED)
            .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(range);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_transfer],
        );
        let copy = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(
                vk::Extent2D {
                    width: pixels.width,
                    height: pixels.height,
                }
                .into(),
            );
        device.cmd_copy_buffer_to_image(
            command_buffer,
            staging,
            target.image,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            &[copy],
        );
        let to_sampled = vk::ImageMemoryBarrier::default()
            .src_access_mask(vk::AccessFlags::TRANSFER_WRITE)
            .dst_access_mask(vk::AccessFlags::SHADER_READ)
            .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
            .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .image(target.image)
            .subresource_range(range);
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::FRAGMENT_SHADER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[to_sampled],
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
            .expect("submit upload");
        device
            .wait_for_fences(&[fence], true, SUBMIT_TIMEOUT_NS)
            .expect("upload did not complete within 10s");

        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_buffer(staging, None);
        device.free_memory(staging_memory, None);
    }
    target
}

/// Read back an RGBA8 image currently in `TRANSFER_SRC_OPTIMAL` layout.
pub fn read_back(runtime: &VkRuntime, image: vk::Image, width: u32, height: u32) -> RgbaImage {
    let device = runtime.device();
    let size = u64::from(width) * u64::from(height) * 4;
    // SAFETY: one fence-waited copy submission against a live device; all
    // transient objects are destroyed after the wait.
    unsafe {
        let memory_properties = runtime
            .instance()
            .get_physical_device_memory_properties(runtime.physical_device());
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        let buffer = device
            .create_buffer(&buffer_info, None)
            .expect("create readback buffer");
        let requirements = device.get_buffer_memory_requirements(buffer);
        let memory = allocate(
            device,
            &memory_properties,
            requirements,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
        );
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind readback memory");

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
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin command buffer");
        let copy = vk::BufferImageCopy::default()
            .image_subresource(
                vk::ImageSubresourceLayers::default()
                    .aspect_mask(vk::ImageAspectFlags::COLOR)
                    .layer_count(1),
            )
            .image_extent(vk::Extent2D { width, height }.into());
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
            .expect("submit readback");
        device
            .wait_for_fences(&[fence], true, SUBMIT_TIMEOUT_NS)
            .expect("readback did not complete within 10s");

        let mapped = device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .expect("map readback memory");
        let mut out = vec![0u8; size as usize];
        std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), out.as_mut_ptr(), out.len());
        device.unmap_memory(memory);

        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);

        RgbaImage::from_pixels(width, height, out)
    }
}
