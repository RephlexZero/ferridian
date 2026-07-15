//! Real-loader coverage for the two render paths beyond the classic
//! render-pass/framebuffer one that `composited.rs` proves:
//! `VK_KHR_create_renderpass2` (promoted to core 1.2) and
//! `VK_KHR_dynamic_rendering` (promoted to core 1.3). Both must classify
//! against the contract and composite through the pack exactly like the
//! classic path — and neither test requests `TRANSFER_SRC` on its own
//! attachment images, so a passing run is also end-to-end proof that the
//! layer's `vkCreateImage` hook (which forces the bit onto attachment-usage
//! images) is doing its job for these paths too.
//!
//! nextest runs each test in its own process, so setting env vars here is
//! safe. The staging helpers mirror `composited.rs` (test binaries don't
//! share modules).

use std::ffi::CString;
use std::path::PathBuf;

use ash::vk;
use ferridian_contract::{Contract, GamePassKind};
use ferridian_pack_compiler::{SlangCompiler, compile_pack, write_artifact};
use ferridian_testkit::require_gpu;
use ferridian_vk_rt::{RuntimeOptions, VkRuntime};

const SIZE: u32 = 128;
const COLOR_FORMAT: vk::Format = vk::Format::R8G8B8A8_UNORM;
const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// The unhashed cdylib cargo links into `target/<profile>/` when building
/// this crate's lib target.
fn layer_dylib_path() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let deps_dir = exe.parent().expect("test executable lives in deps/");
    let name = if cfg!(windows) {
        "ferridian_vk_layer.dll"
    } else if cfg!(target_os = "macos") {
        "libferridian_vk_layer.dylib"
    } else {
        "libferridian_vk_layer.so"
    };
    let candidates = [
        deps_dir.join(name),
        deps_dir.parent().expect("deps/ has a parent").join(name),
    ];
    candidates
        .iter()
        .find(|path| path.is_file())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "layer cdylib not found at {} — did the lib target build?",
                candidates[0].display()
            )
        })
}

/// Stage the committed manifest with its library_path rewritten to the
/// freshly built cdylib, and point `VK_LAYER_PATH` at it (preserving every
/// entry of any existing multi-path value — VVL has to stay findable).
fn stage_layer() -> PathBuf {
    let dylib = layer_dylib_path();
    let manifest_dir =
        std::env::temp_dir().join(format!("ferridian-modern-render-{}", std::process::id()));
    std::fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    let mut manifest: serde_json::Value =
        serde_json::from_str(include_str!("../manifest/VkLayer_FERRIDIAN_overlay.json"))
            .expect("committed manifest is JSON");
    manifest["layer"]["library_path"] =
        serde_json::Value::from(dylib.to_str().expect("dylib path is valid UTF-8"));
    std::fs::write(
        manifest_dir.join("VkLayer_FERRIDIAN_overlay.json"),
        manifest.to_string(),
    )
    .expect("write manifest");
    let system_layers = std::env::var("VK_LAYER_PATH")
        .unwrap_or_else(|_| "/usr/share/vulkan/explicit_layer.d".to_owned());
    let layer_path = std::env::join_paths(
        std::iter::once(manifest_dir.clone()).chain(std::env::split_paths(&system_layers)),
    )
    .expect("join VK_LAYER_PATH entries");
    // SAFETY: nextest gives this test its own process; nothing else is
    // reading the environment concurrently.
    unsafe { std::env::set_var("VK_LAYER_PATH", layer_path) };
    manifest_dir
}

/// Compile and publish the reference pack, and point `FERRIDIAN_PACK` at the
/// artifact — the layer picks it up at device creation.
fn stage_reference_pack() -> PathBuf {
    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");
    let pack_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packs/reference");
    let artifact = compile_pack(&pack_dir, &compiler).expect("reference pack compiles");
    let out = std::env::temp_dir().join(format!(
        "ferridian-modern-render-pack-{}",
        std::process::id()
    ));
    write_artifact(&artifact, &out).expect("write artifact");
    // SAFETY: as in stage_layer — this process owns its environment.
    unsafe { std::env::set_var(ferridian_vk_layer::PACK_ENV, &out) };
    out
}

fn boot_layered_runtime() -> VkRuntime {
    let layer_name = ferridian_vk_layer::LAYER_NAME
        .to_str()
        .expect("layer name is ASCII")
        .to_owned();
    VkRuntime::new(&RuntimeOptions {
        app_name: "ferridian-modern-render-test".to_owned(),
        enable_validation: true,
        prefer_software_device: true,
        extra_layers: vec![layer_name],
    })
    .expect("boot lavapipe with the Ferridian layer + validation enabled")
}

/// Device-local memory for `requirements` — lavapipe advertises everything
/// as host-visible too, but every real GPU's attachment memory is
/// device-local, so that's the honest choice here.
unsafe fn allocate(
    device: &ash::Device,
    memory_properties: &vk::PhysicalDeviceMemoryProperties,
    requirements: vk::MemoryRequirements,
) -> vk::DeviceMemory {
    let type_index = memory_properties.memory_types[..memory_properties.memory_type_count as usize]
        .iter()
        .enumerate()
        .position(|(index, memory_type)| {
            requirements.memory_type_bits & (1 << index) != 0
                && memory_type
                    .property_flags
                    .contains(vk::MemoryPropertyFlags::DEVICE_LOCAL)
        })
        .expect("a device-local memory type exists on lavapipe") as u32;
    let info = vk::MemoryAllocateInfo::default()
        .allocation_size(requirements.size)
        .memory_type_index(type_index);
    // SAFETY: valid allocate info; the caller frees this after fence-waited work.
    unsafe {
        device
            .allocate_memory(&info, None)
            .expect("allocate memory")
    }
}

/// The color + depth attachment images under test. Deliberately requests
/// only `COLOR_ATTACHMENT`/`DEPTH_STENCIL_ATTACHMENT` usage — no
/// `TRANSFER_SRC` — so a successful tap-copy and final readback prove the
/// layer's `vkCreateImage` hook added it.
struct Scene {
    color_image: vk::Image,
    color_memory: vk::DeviceMemory,
    color_view: vk::ImageView,
    depth_image: vk::Image,
    depth_memory: vk::DeviceMemory,
    depth_view: vk::ImageView,
}

impl Scene {
    fn new(runtime: &VkRuntime) -> Scene {
        let device = runtime.device();
        // SAFETY: one linear sequence of Vulkan calls against a live
        // device; every create is paired with a destroy, borrowed create
        // infos outlive the calls they're passed to.
        unsafe {
            let memory_properties = runtime
                .instance()
                .get_physical_device_memory_properties(runtime.physical_device());
            let extent = vk::Extent3D {
                width: SIZE,
                height: SIZE,
                depth: 1,
            };

            let color_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(COLOR_FORMAT)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let color_image = device
                .create_image(&color_info, None)
                .expect("create color image");
            let color_requirements = device.get_image_memory_requirements(color_image);
            let color_memory = allocate(device, &memory_properties, color_requirements);
            device
                .bind_image_memory(color_image, color_memory, 0)
                .expect("bind color memory");
            let color_view_info = vk::ImageViewCreateInfo::default()
                .image(color_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(COLOR_FORMAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .level_count(1)
                        .layer_count(1),
                );
            let color_view = device
                .create_image_view(&color_view_info, None)
                .expect("create color view");

            let depth_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(DEPTH_FORMAT)
                .extent(extent)
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            let depth_image = device
                .create_image(&depth_info, None)
                .expect("create depth image");
            let depth_requirements = device.get_image_memory_requirements(depth_image);
            let depth_memory = allocate(device, &memory_properties, depth_requirements);
            device
                .bind_image_memory(depth_image, depth_memory, 0)
                .expect("bind depth memory");
            let depth_view_info = vk::ImageViewCreateInfo::default()
                .image(depth_image)
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(DEPTH_FORMAT)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                );
            let depth_view = device
                .create_image_view(&depth_view_info, None)
                .expect("create depth view");

            Scene {
                color_image,
                color_memory,
                color_view,
                depth_image,
                depth_memory,
                depth_view,
            }
        }
    }

    /// # Safety
    /// No submitted work may still reference these objects.
    unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: created on `device`, per this method's contract.
        unsafe {
            device.destroy_image_view(self.color_view, None);
            device.destroy_image(self.color_image, None);
            device.free_memory(self.color_memory, None);
            device.destroy_image_view(self.depth_view, None);
            device.destroy_image(self.depth_image, None);
            device.free_memory(self.depth_memory, None);
        }
    }
}

/// The scene fixture's two stage modules, compiled once per test — never one
/// module mixing both entry points.
struct SceneShader {
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
}

impl SceneShader {
    fn compile(runtime: &VkRuntime) -> SceneShader {
        let compiler = SlangCompiler::from_environment().expect("GPU tests need slangc");
        let fixture =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/scene.slang");
        let vertex_spirv = compiler
            .compile_stage(&fixture, "scene", "vs_main", "vertex")
            .expect("compile scene fixture vertex stage");
        let fragment_spirv = compiler
            .compile_stage(&fixture, "scene", "fs_main", "fragment")
            .expect("compile scene fixture fragment stage");
        let device = runtime.device();
        // SAFETY: valid create infos over compiler-verified SPIR-V.
        unsafe {
            let vertex_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&vertex_spirv),
                    None,
                )
                .expect("create vertex shader module");
            let fragment_module = device
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo::default().code(&fragment_spirv),
                    None,
                )
                .expect("create fragment shader module");
            SceneShader {
                vertex_module,
                fragment_module,
            }
        }
    }

    /// # Safety
    /// No submitted work may still reference these objects.
    unsafe fn destroy(self, device: &ash::Device) {
        // SAFETY: created on `device`, per this method's contract.
        unsafe {
            device.destroy_shader_module(self.vertex_module, None);
            device.destroy_shader_module(self.fragment_module, None);
        }
    }
}

fn shader_stages<'a>(
    shader: &SceneShader,
    vs_name: &'a CString,
    fs_name: &'a CString,
) -> [vk::PipelineShaderStageCreateInfo<'a>; 2] {
    [
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::VERTEX)
            .module(shader.vertex_module)
            .name(vs_name),
        vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::FRAGMENT)
            .module(shader.fragment_module)
            .name(fs_name),
    ]
}

fn pixel(image: &ferridian_testkit::RgbaImage, x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * SIZE + x) * 4) as usize;
    image.pixels[offset..offset + 4]
        .try_into()
        .expect("4 bytes per pixel")
}

/// The composited frame must show the same physics `composited.rs` proves
/// for the classic path: albedo split surviving lit in the near half.
fn assert_composited_frame(image: &ferridian_testkit::RgbaImage) {
    assert!(
        image.pixels.chunks_exact(4).all(|px| px[3] == 255),
        "composite must cover the frame with alpha 1.0"
    );
    let (mut left, mut right) = (0u64, 0u64);
    for y in 0..SIZE / 2 {
        for x in 0..SIZE {
            let red = pixel(image, x, y)[0] as u64;
            if x < SIZE / 2 {
                left += red;
            } else {
                right += red;
            }
        }
    }
    assert!(
        left > right,
        "warm half should stay redder than cool half (left {left} vs right {right})"
    );
}

/// Read `image` back (already in `TRANSFER_SRC_OPTIMAL`) as an offscreen
/// RGBA8 buffer.
fn read_back_color(runtime: &VkRuntime, image: vk::Image) -> ferridian_testkit::RgbaImage {
    let device = runtime.device();
    // SAFETY: physical_device comes from this instance and is live.
    let memory_properties = unsafe {
        runtime
            .instance()
            .get_physical_device_memory_properties(runtime.physical_device())
    };
    let size = u64::from(SIZE) * u64::from(SIZE) * 4;
    // SAFETY: one linear sequence of Vulkan calls; every create is
    // destroyed before returning, all work is fence-waited first.
    unsafe {
        let buffer_info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST);
        let buffer = device
            .create_buffer(&buffer_info, None)
            .expect("create readback buffer");
        let requirements = device.get_buffer_memory_requirements(buffer);
        let type_index = memory_properties.memory_types
            [..memory_properties.memory_type_count as usize]
            .iter()
            .enumerate()
            .position(|(index, memory_type)| {
                requirements.memory_type_bits & (1 << index) != 0
                    && memory_type.property_flags.contains(
                        vk::MemoryPropertyFlags::HOST_VISIBLE
                            | vk::MemoryPropertyFlags::HOST_COHERENT,
                    )
            })
            .expect("a host-visible memory type exists") as u32;
        let memory = device
            .allocate_memory(
                &vk::MemoryAllocateInfo::default()
                    .allocation_size(requirements.size)
                    .memory_type_index(type_index),
                None,
            )
            .expect("allocate readback memory");
        device
            .bind_buffer_memory(buffer, memory, 0)
            .expect("bind readback memory");

        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(runtime.queue_family_index()),
                None,
            )
            .expect("create command pool");
        let command_buffer = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
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
            .image_extent(vk::Extent3D {
                width: SIZE,
                height: SIZE,
                depth: 1,
            });
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
            .wait_for_fences(&[fence], true, 10_000_000_000)
            .expect("readback did not complete within 10s");

        let mapped = device
            .map_memory(memory, 0, size, vk::MemoryMapFlags::empty())
            .expect("map readback memory");
        let mut pixels = vec![0u8; size as usize];
        std::ptr::copy_nonoverlapping(mapped.cast::<u8>(), pixels.as_mut_ptr(), pixels.len());
        device.unmap_memory(memory);

        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_buffer(buffer, None);
        device.free_memory(memory, None);

        ferridian_testkit::RgbaImage::from_pixels(SIZE, SIZE, pixels)
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "test helper bundling one graphics pipeline's fixed state; a struct would just move the field count elsewhere"
)]
fn common_pipeline_state<'a>(
    stages: &'a [vk::PipelineShaderStageCreateInfo<'a>; 2],
    vertex_input: &'a vk::PipelineVertexInputStateCreateInfo<'a>,
    input_assembly: &'a vk::PipelineInputAssemblyStateCreateInfo<'a>,
    viewport_state: &'a vk::PipelineViewportStateCreateInfo<'a>,
    rasterization: &'a vk::PipelineRasterizationStateCreateInfo<'a>,
    multisample: &'a vk::PipelineMultisampleStateCreateInfo<'a>,
    depth_stencil: &'a vk::PipelineDepthStencilStateCreateInfo<'a>,
    color_blend: &'a vk::PipelineColorBlendStateCreateInfo<'a>,
    layout: vk::PipelineLayout,
) -> vk::GraphicsPipelineCreateInfo<'a> {
    vk::GraphicsPipelineCreateInfo::default()
        .stages(stages)
        .vertex_input_state(vertex_input)
        .input_assembly_state(input_assembly)
        .viewport_state(viewport_state)
        .rasterization_state(rasterization)
        .multisample_state(multisample)
        .depth_stencil_state(depth_stencil)
        .color_blend_state(color_blend)
        .layout(layout)
}

fn anchor(kind: GamePassKind) -> String {
    Contract::current()
        .passes
        .iter()
        .find(|pass| pass.kind == kind)
        .expect("contract tracks the kind")
        .game_anchor
        .clone()
}

/// The world-final pass through `vkCreateRenderPass2`/`vkCmdBeginRenderPass2`
/// classifies and composites exactly like the classic path.
#[test]
fn renderpass2_path_classifies_and_composites() {
    require_gpu!();
    let manifest_dir = stage_layer();
    let pack_out = stage_reference_pack();
    let translucent = anchor(GamePassKind::Translucent);

    let runtime = boot_layered_runtime();
    let device = runtime.device();
    let scene = Scene::new(&runtime);
    let shader = SceneShader::compile(&runtime);

    // SAFETY: one linear sequence of Vulkan calls against a live device;
    // every create is destroyed before returning, all work fence-waited
    // before readback and teardown.
    let image = unsafe {
        let color_attachment = vk::AttachmentDescription2::default()
            .format(COLOR_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL);
        let depth_attachment = vk::AttachmentDescription2::default()
            .format(DEPTH_FORMAT)
            .samples(vk::SampleCountFlags::TYPE_1)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .stencil_load_op(vk::AttachmentLoadOp::DONT_CARE)
            .stencil_store_op(vk::AttachmentStoreOp::DONT_CARE)
            .initial_layout(vk::ImageLayout::UNDEFINED)
            .final_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL);
        let attachments = [color_attachment, depth_attachment];
        let color_ref = vk::AttachmentReference2::default()
            .attachment(0)
            .layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .aspect_mask(vk::ImageAspectFlags::COLOR);
        let color_refs = [color_ref];
        let depth_ref = vk::AttachmentReference2::default()
            .attachment(1)
            .layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .aspect_mask(vk::ImageAspectFlags::DEPTH);
        let subpass = vk::SubpassDescription2::default()
            .pipeline_bind_point(vk::PipelineBindPoint::GRAPHICS)
            .color_attachments(&color_refs)
            .depth_stencil_attachment(&depth_ref);
        let subpasses = [subpass];
        let dependencies = [
            vk::SubpassDependency2::default()
                .src_subpass(vk::SUBPASS_EXTERNAL)
                .dst_subpass(0)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .dst_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .dst_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                ),
            vk::SubpassDependency2::default()
                .src_subpass(0)
                .dst_subpass(vk::SUBPASS_EXTERNAL)
                .src_stage_mask(
                    vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                        | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                )
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                )
                .dst_stage_mask(vk::PipelineStageFlags::TRANSFER)
                .dst_access_mask(vk::AccessFlags::TRANSFER_READ),
        ];
        let render_pass_info = vk::RenderPassCreateInfo2::default()
            .attachments(&attachments)
            .subpasses(&subpasses)
            .dependencies(&dependencies);
        let render_pass = device
            .create_render_pass2(&render_pass_info, None)
            .expect("create render pass 2");

        let framebuffer_views = [scene.color_view, scene.depth_view];
        let framebuffer = device
            .create_framebuffer(
                &vk::FramebufferCreateInfo::default()
                    .render_pass(render_pass)
                    .attachments(&framebuffer_views)
                    .width(SIZE)
                    .height(SIZE)
                    .layers(1),
                None,
            )
            .expect("create framebuffer");

        let vs_name = c"vs_main".to_owned();
        let fs_name = c"fs_main".to_owned();
        let stages = shader_stages(&shader, &vs_name, &fs_name);
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: SIZE as f32,
            height: SIZE as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: SIZE,
                height: SIZE,
            },
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
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::ALWAYS);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .expect("create pipeline layout");
        let pipeline_info = common_pipeline_state(
            &stages,
            &vertex_input,
            &input_assembly,
            &viewport_state,
            &rasterization,
            &multisample,
            &depth_stencil,
            &color_blend,
            layout,
        )
        .render_pass(render_pass)
        .subpass(0);
        let pipeline = device
            .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
            .expect("create graphics pipeline")[0];

        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(runtime.queue_family_index()),
                None,
            )
            .expect("create command pool");
        let command_buffer = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("allocate command buffer")[0];
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin command buffer");

        let debug_utils = ash::ext::debug_utils::Device::new(runtime.instance(), device);
        let label_name = CString::new(translucent.as_str()).expect("anchor has no interior NUL");
        let label = vk::DebugUtilsLabelEXT::default().label_name(&label_name);
        debug_utils.cmd_begin_debug_utils_label(command_buffer, &label);

        let clear_values = [
            vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            },
            vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            },
        ];
        let begin_info = vk::RenderPassBeginInfo::default()
            .render_pass(render_pass)
            .framebuffer(framebuffer)
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: SIZE,
                    height: SIZE,
                },
            })
            .clear_values(&clear_values);
        let subpass_begin = vk::SubpassBeginInfo::default().contents(vk::SubpassContents::INLINE);
        device.cmd_begin_render_pass2(command_buffer, &begin_info, &subpass_begin);
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_draw(command_buffer, 3, 1, 0, 0);
        let subpass_end = vk::SubpassEndInfo::default();
        device.cmd_end_render_pass2(command_buffer, &subpass_end);

        debug_utils.cmd_end_debug_utils_label(command_buffer);

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
            .wait_for_fences(&[fence], true, 10_000_000_000)
            .expect("render did not complete within 10s");

        let image = read_back_color(&runtime, scene.color_image);

        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        device.destroy_framebuffer(framebuffer, None);
        device.destroy_render_pass(render_pass, None);
        image
    };

    assert_composited_frame(&image);

    let messages = runtime.validation_messages();
    assert!(
        messages.is_empty(),
        "validation reported {} message(s) over RenderPass2:\n{}",
        messages.len(),
        messages.join("\n")
    );

    // SAFETY: submissions above were fence-waited; nothing references these.
    unsafe {
        shader.destroy(device);
        scene.destroy(device);
    }
    drop(runtime);
    std::fs::remove_dir_all(manifest_dir).ok();
    std::fs::remove_dir_all(pack_out).ok();
}

/// The world-final pass through `vkCmdBeginRendering` (dynamic rendering, no
/// `VkRenderPass`/`VkFramebuffer` object at all) classifies and composites
/// exactly like the classic path.
#[test]
fn dynamic_rendering_path_classifies_and_composites() {
    require_gpu!();
    let manifest_dir = stage_layer();
    let pack_out = stage_reference_pack();
    let translucent = anchor(GamePassKind::Translucent);

    let runtime = boot_layered_runtime();
    let device = runtime.device();
    let scene = Scene::new(&runtime);
    let shader = SceneShader::compile(&runtime);

    // SAFETY: as in the RenderPass2 test.
    let image = unsafe {
        let vs_name = c"vs_main".to_owned();
        let fs_name = c"fs_main".to_owned();
        let stages = shader_stages(&shader, &vs_name, &fs_name);
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        let viewports = [vk::Viewport {
            x: 0.0,
            y: 0.0,
            width: SIZE as f32,
            height: SIZE as f32,
            min_depth: 0.0,
            max_depth: 1.0,
        }];
        let scissors = [vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: SIZE,
                height: SIZE,
            },
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
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default()
            .depth_test_enable(true)
            .depth_write_enable(true)
            .depth_compare_op(vk::CompareOp::ALWAYS);
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let layout = device
            .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
            .expect("create pipeline layout");
        let color_formats = [COLOR_FORMAT];
        let mut rendering_info = vk::PipelineRenderingCreateInfo::default()
            .color_attachment_formats(&color_formats)
            .depth_attachment_format(DEPTH_FORMAT);
        let pipeline_info = common_pipeline_state(
            &stages,
            &vertex_input,
            &input_assembly,
            &viewport_state,
            &rasterization,
            &multisample,
            &depth_stencil,
            &color_blend,
            layout,
        )
        .push_next(&mut rendering_info);
        let pipeline = match device.create_graphics_pipelines(
            vk::PipelineCache::null(),
            &[pipeline_info],
            None,
        ) {
            Ok(pipelines) => pipelines[0],
            Err((_, result)) => {
                let messages = runtime.validation_messages();
                panic!(
                    "create graphics pipeline: {result}\n{}",
                    messages.join("\n")
                );
            }
        };

        let pool = device
            .create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(runtime.queue_family_index()),
                None,
            )
            .expect("create command pool");
        let command_buffer = device
            .allocate_command_buffers(
                &vk::CommandBufferAllocateInfo::default()
                    .command_pool(pool)
                    .level(vk::CommandBufferLevel::PRIMARY)
                    .command_buffer_count(1),
            )
            .expect("allocate command buffer")[0];
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("begin command buffer");

        let debug_utils = ash::ext::debug_utils::Device::new(runtime.instance(), device);
        let label_name = CString::new(translucent.as_str()).expect("anchor has no interior NUL");
        let label = vk::DebugUtilsLabelEXT::default().label_name(&label_name);
        debug_utils.cmd_begin_debug_utils_label(command_buffer, &label);

        // Dynamic rendering has no automatic layout transition machinery
        // (that's what a classic render pass's initial/final layouts give
        // you) — the app transitions before vkCmdBeginRendering itself.
        let range = |aspect: vk::ImageAspectFlags| {
            vk::ImageSubresourceRange::default()
                .aspect_mask(aspect)
                .level_count(1)
                .layer_count(1)
        };
        let acquire = [
            vk::ImageMemoryBarrier::default()
                .image(scene.color_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(range(vk::ImageAspectFlags::COLOR)),
            vk::ImageMemoryBarrier::default()
                .image(scene.depth_image)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                .dst_access_mask(vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE)
                .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
                .subresource_range(range(vk::ImageAspectFlags::DEPTH)),
        ];
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &acquire,
        );

        let color_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(scene.color_view)
            .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                color: vk::ClearColorValue {
                    float32: [0.0, 0.0, 0.0, 1.0],
                },
            });
        let color_attachments = [color_attachment];
        let depth_attachment = vk::RenderingAttachmentInfo::default()
            .image_view(scene.depth_view)
            .image_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
            .load_op(vk::AttachmentLoadOp::CLEAR)
            .store_op(vk::AttachmentStoreOp::STORE)
            .clear_value(vk::ClearValue {
                depth_stencil: vk::ClearDepthStencilValue {
                    depth: 1.0,
                    stencil: 0,
                },
            });
        let rendering_info = vk::RenderingInfo::default()
            .render_area(vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: SIZE,
                    height: SIZE,
                },
            })
            .layer_count(1)
            .color_attachments(&color_attachments)
            .depth_attachment(&depth_attachment);
        device.cmd_begin_rendering(command_buffer, &rendering_info);
        device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
        device.cmd_draw(command_buffer, 3, 1, 0, 0);
        device.cmd_end_rendering(command_buffer);

        debug_utils.cmd_end_debug_utils_label(command_buffer);

        // The layer's compositor left color in `COLOR_ATTACHMENT_OPTIMAL`
        // (whatever `RenderingAttachmentInfo.image_layout` declared above —
        // dynamic rendering never transitions it further on its own); one
        // more barrier gets it to the layout the final readback needs.
        let final_barrier = vk::ImageMemoryBarrier::default()
            .image(scene.color_image)
            .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
            .new_layout(vk::ImageLayout::TRANSFER_SRC_OPTIMAL)
            .src_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
            .dst_access_mask(vk::AccessFlags::TRANSFER_READ)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .subresource_range(range(vk::ImageAspectFlags::COLOR));
        device.cmd_pipeline_barrier(
            command_buffer,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TRANSFER,
            vk::DependencyFlags::empty(),
            &[],
            &[],
            &[final_barrier],
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
            .wait_for_fences(&[fence], true, 10_000_000_000)
            .expect("render did not complete within 10s");

        let image = read_back_color(&runtime, scene.color_image);

        device.destroy_fence(fence, None);
        device.destroy_command_pool(pool, None);
        device.destroy_pipeline(pipeline, None);
        device.destroy_pipeline_layout(layout, None);
        image
    };

    assert_composited_frame(&image);

    let messages = runtime.validation_messages();
    assert!(
        messages.is_empty(),
        "validation reported {} message(s) over dynamic rendering:\n{}",
        messages.len(),
        messages.join("\n")
    );

    // SAFETY: submissions above were fence-waited; nothing references these.
    unsafe {
        shader.destroy(device);
        scene.destroy(device);
    }
    drop(runtime);
    std::fs::remove_dir_all(manifest_dir).ok();
    std::fs::remove_dir_all(pack_out).ok();
}
