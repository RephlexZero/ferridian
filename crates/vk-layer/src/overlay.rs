//! The composite-over-frame walking skeleton (M2): draw the layer's own
//! geometry *inside the game's render pass*, but only into passes the
//! contract classified — Unknown passes are forwarded untouched.
//!
//! The overlay pipeline is built lazily against the application's own
//! `VkRenderPass` handle (that is what makes it render-pass-compatible) from
//! the committed `shaders/overlay.{vertex,fragment}.spv` — two modules,
//! never one mixing both entry points (that's exactly the shape
//! GPU-assisted validation can't instrument); a layer living inside a game
//! process has no shader compiler. Every Vulkan call here goes through a
//! *down-chain*
//! `ash::Device` loaded from the next layer's GDPA, so the layer never
//! re-enters itself.
//!
//! Skeleton limitations, all caught by VVL in the golden environment rather
//! than guessed at: the pipeline assumes a single-sampled subpass 0 with
//! exactly one color attachment (true of vanilla's main targets). Pipeline
//! creation failure disables compositing for that render pass and is cached,
//! never retried per frame. TODO(M2): derive multisample/attachment state
//! from intercepted vkCreateRenderPass.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

use ash::vk;
use ash::vk::Handle;
use ferridian_contract::GamePassKind;

// Two modules, never one mixing both entry points — that's exactly the shape
// GPU-assisted validation can't instrument.
static OVERLAY_VERTEX_SPV: &[u8] = include_bytes!("../shaders/overlay.vertex.spv");
static OVERLAY_FRAGMENT_SPV: &[u8] = include_bytes!("../shaders/overlay.fragment.spv");

fn words_of(bytes: &'static [u8]) -> Vec<u32> {
    bytes
        .chunks_exact(4)
        .map(|word| u32::from_le_bytes(word.try_into().expect("chunks_exact(4)")))
        .collect()
}

fn overlay_vertex_words() -> &'static [u32] {
    static WORDS: OnceLock<Vec<u32>> = OnceLock::new();
    WORDS.get_or_init(|| words_of(OVERLAY_VERTEX_SPV))
}

fn overlay_fragment_words() -> &'static [u32] {
    static WORDS: OnceLock<Vec<u32>> = OnceLock::new();
    WORDS.get_or_init(|| words_of(OVERLAY_FRAGMENT_SPV))
}

/// A render pass currently being recorded into a command buffer, keyed by
/// the command buffer's raw handle (unique among live command buffers).
pub(crate) struct ActivePass {
    pub render_pass: vk::RenderPass,
    pub framebuffer: vk::Framebuffer,
    pub render_area: vk::Rect2D,
    /// How the contract classified this pass; the compositors only ever
    /// touch classified passes — Unknown is forwarded untouched.
    pub kind: GamePassKind,
}

impl ActivePass {
    pub(crate) fn classified(&self) -> bool {
        self.kind != GamePassKind::Unknown
    }
}

static PASSES_IN_FLIGHT: Mutex<BTreeMap<u64, ActivePass>> = Mutex::new(BTreeMap::new());

pub(crate) fn pass_begun(command_buffer: vk::CommandBuffer, pass: ActivePass) {
    PASSES_IN_FLIGHT
        .lock()
        .expect("active pass registry poisoned")
        .insert(command_buffer.as_raw(), pass);
}

pub(crate) fn pass_ending(command_buffer: vk::CommandBuffer) -> Option<ActivePass> {
    PASSES_IN_FLIGHT
        .lock()
        .expect("active pass registry poisoned")
        .remove(&command_buffer.as_raw())
}

/// Per-device overlay state: the down-chain device table and one pipeline
/// per application render pass. Lives inside the device's dispatch entry.
pub(crate) struct Overlay {
    device: ash::Device,
    vertex_module: vk::ShaderModule,
    fragment_module: vk::ShaderModule,
    pipeline_layout: vk::PipelineLayout,
    /// `None` records a pipeline that failed to build (incompatible pass) so
    /// the failure is not retried every frame.
    pipelines: BTreeMap<u64, Option<vk::Pipeline>>,
}

impl Overlay {
    /// Create the pass-independent objects on the *down-chain* device table
    /// (calls through it never re-enter the layer). `None` (logged) disables
    /// the overlay for this device.
    pub(crate) fn new(device: ash::Device) -> Option<Overlay> {
        let vertex_info = vk::ShaderModuleCreateInfo::default().code(overlay_vertex_words());
        // SAFETY: valid create info over the embedded (build-verified) SPIR-V.
        let vertex_module = match unsafe { device.create_shader_module(&vertex_info, None) } {
            Ok(module) => module,
            Err(error) => {
                tracing::warn!(%error, "overlay disabled: vertex shader module creation failed");
                return None;
            }
        };
        let fragment_info = vk::ShaderModuleCreateInfo::default().code(overlay_fragment_words());
        // SAFETY: as above.
        let fragment_module = match unsafe { device.create_shader_module(&fragment_info, None) } {
            Ok(module) => module,
            Err(error) => {
                // SAFETY: destroying the module created above, unused elsewhere.
                unsafe { device.destroy_shader_module(vertex_module, None) };
                tracing::warn!(%error, "overlay disabled: fragment shader module creation failed");
                return None;
            }
        };
        // SAFETY: valid (empty) layout create info.
        let pipeline_layout = match unsafe {
            device.create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)
        } {
            Ok(layout) => layout,
            Err(error) => {
                // SAFETY: destroying the modules created above, unused elsewhere.
                unsafe {
                    device.destroy_shader_module(vertex_module, None);
                    device.destroy_shader_module(fragment_module, None);
                }
                tracing::warn!(%error, "overlay disabled: pipeline layout creation failed");
                return None;
            }
        };
        Some(Overlay {
            device,
            vertex_module,
            fragment_module,
            pipeline_layout,
            pipelines: BTreeMap::new(),
        })
    }

    fn pipeline_for(&mut self, render_pass: vk::RenderPass) -> Option<vk::Pipeline> {
        if let Some(&cached) = self.pipelines.get(&render_pass.as_raw()) {
            return cached;
        }
        let stages = [
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::VERTEX)
                .module(self.vertex_module)
                .name(c"vs_main"),
            vk::PipelineShaderStageCreateInfo::default()
                .stage(vk::ShaderStageFlags::FRAGMENT)
                .module(self.fragment_module)
                .name(c"fs_main"),
        ];
        let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo::default()
            .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
        // Viewport/scissor are dynamic: one pipeline serves any render area.
        let viewport_state = vk::PipelineViewportStateCreateInfo::default()
            .viewport_count(1)
            .scissor_count(1);
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic_state =
            vk::PipelineDynamicStateCreateInfo::default().dynamic_states(&dynamic_states);
        let rasterization = vk::PipelineRasterizationStateCreateInfo::default()
            .polygon_mode(vk::PolygonMode::FILL)
            .cull_mode(vk::CullModeFlags::NONE)
            .line_width(1.0);
        let multisample = vk::PipelineMultisampleStateCreateInfo::default()
            .rasterization_samples(vk::SampleCountFlags::TYPE_1);
        // Depth test off; ignored by Vulkan if the subpass has no depth
        // attachment, harmless (draw-on-top) if it has one.
        let depth_stencil = vk::PipelineDepthStencilStateCreateInfo::default();
        let blend_attachments = [vk::PipelineColorBlendAttachmentState::default()
            .color_write_mask(vk::ColorComponentFlags::RGBA)];
        let color_blend =
            vk::PipelineColorBlendStateCreateInfo::default().attachments(&blend_attachments);
        let create_info = vk::GraphicsPipelineCreateInfo::default()
            .stages(&stages)
            .vertex_input_state(&vertex_input)
            .input_assembly_state(&input_assembly)
            .viewport_state(&viewport_state)
            .rasterization_state(&rasterization)
            .multisample_state(&multisample)
            .depth_stencil_state(&depth_stencil)
            .color_blend_state(&color_blend)
            .dynamic_state(&dynamic_state)
            .layout(self.pipeline_layout)
            .render_pass(render_pass)
            .subpass(0);
        // SAFETY: valid create info; the application's render pass handle is
        // live for the duration of the pass we were called from.
        let pipeline = match unsafe {
            self.device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None)
        } {
            Ok(pipelines) => Some(pipelines[0]),
            Err((_, error)) => {
                tracing::warn!(
                    %error,
                    render_pass = render_pass.as_raw(),
                    "overlay pipeline creation failed; compositing disabled for this pass"
                );
                None
            }
        };
        self.pipelines.insert(render_pass.as_raw(), pipeline);
        pipeline
    }

    /// Record the overlay draw into the application's open render pass.
    ///
    /// This clobbers bound-pipeline/viewport/scissor state, which is why it
    /// runs at `vkCmdEndRenderPass` time — after every application draw in
    /// the pass, before the pass closes.
    pub(crate) fn composite(&mut self, command_buffer: vk::CommandBuffer, pass: &ActivePass) {
        let Some(pipeline) = self.pipeline_for(pass.render_pass) else {
            return;
        };
        let viewport = vk::Viewport::default()
            .x(pass.render_area.offset.x as f32)
            .y(pass.render_area.offset.y as f32)
            .width(pass.render_area.extent.width as f32)
            .height(pass.render_area.extent.height as f32)
            .max_depth(1.0);
        // SAFETY: recording core-1.0 commands into a command buffer that is
        // inside a render pass instance (we are called from the app's own
        // vkCmdEndRenderPass, on the app's recording thread).
        unsafe {
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline,
            );
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);
            self.device
                .cmd_set_scissor(command_buffer, 0, &[pass.render_area]);
            self.device.cmd_draw(command_buffer, 3, 1, 0, 0);
        }
    }

    /// Destroy every child object; must run before the device itself is
    /// destroyed (VVL reports leaked children at vkDestroyDevice otherwise).
    pub(crate) fn destroy(&mut self) {
        // SAFETY: the device is still live (we run before forwarding its
        // destroy) and these objects are used by no in-flight work — the
        // application must have idled the device per vkDestroyDevice's rules.
        unsafe {
            for pipeline in self.pipelines.values().flatten() {
                self.device.destroy_pipeline(*pipeline, None);
            }
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_shader_module(self.vertex_module, None);
            self.device
                .destroy_shader_module(self.fragment_module, None);
        }
        self.pipelines.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_overlay_spirv_is_wellformed() {
        for (words, bytes) in [
            (overlay_vertex_words(), OVERLAY_VERTEX_SPV),
            (overlay_fragment_words(), OVERLAY_FRAGMENT_SPV),
        ] {
            assert_eq!(words[0], 0x0723_0203, "SPIR-V magic");
            assert_eq!(bytes.len() % 4, 0, "whole words");
            assert!(words.len() > 5, "more than a bare header");
        }
    }
}
