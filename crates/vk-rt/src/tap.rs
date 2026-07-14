//! Frame-geometry bookkeeping for the layer: which image views, framebuffers,
//! and render passes the intercepted application has created, and how a
//! `(render pass, framebuffer)` pair at `vkCmdBeginRenderPass` time resolves
//! to concrete color/depth attachments the compositor can tap.
//!
//! Pure data over raw handles — the layer feeds it from its create/destroy
//! hooks and never calls Vulkan here, so all of it is unit-testable without a
//! device. Final layouts come from the *begun* render pass (compatible render
//! passes may declare different final layouts, and the instance's own pass is
//! the one whose layouts the images actually end up in).

use std::collections::BTreeMap;

use ash::vk;
use ash::vk::Handle;

/// What the layer learned from one `vkCreateImageView`.
#[derive(Debug, Clone, Copy)]
pub struct ViewRecord {
    pub image: vk::Image,
    pub format: vk::Format,
    pub aspect: vk::ImageAspectFlags,
}

#[derive(Debug)]
struct FramebufferRecord {
    attachments: Vec<vk::ImageView>,
    extent: vk::Extent2D,
}

#[derive(Debug)]
struct RenderPassRecord {
    final_layouts: Vec<vk::ImageLayout>,
}

/// One attachment of the intercepted pass, resolved to everything the
/// compositor needs: the image (for barriers and copies), the view (for
/// output-target identity), and the layout the pass leaves it in.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedAttachment {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub format: vk::Format,
    pub final_layout: vk::ImageLayout,
}

/// The intercepted pass's attachments: the first color attachment and, if
/// present, the first depth/stencil attachment.
#[derive(Debug, Clone, Copy)]
pub struct ResolvedFrame {
    pub extent: vk::Extent2D,
    pub color: ResolvedAttachment,
    pub depth: Option<ResolvedAttachment>,
}

/// Registry of the application's live views/framebuffers/render passes,
/// keyed by raw handle (unique among live objects of a type).
#[derive(Debug, Default)]
pub struct TapRegistry {
    views: BTreeMap<u64, ViewRecord>,
    framebuffers: BTreeMap<u64, FramebufferRecord>,
    render_passes: BTreeMap<u64, RenderPassRecord>,
}

impl TapRegistry {
    pub fn record_view(&mut self, view: vk::ImageView, record: ViewRecord) {
        self.views.insert(view.as_raw(), record);
    }

    pub fn forget_view(&mut self, view: vk::ImageView) {
        self.views.remove(&view.as_raw());
    }

    pub fn record_framebuffer(
        &mut self,
        framebuffer: vk::Framebuffer,
        attachments: Vec<vk::ImageView>,
        extent: vk::Extent2D,
    ) {
        self.framebuffers.insert(
            framebuffer.as_raw(),
            FramebufferRecord {
                attachments,
                extent,
            },
        );
    }

    pub fn forget_framebuffer(&mut self, framebuffer: vk::Framebuffer) {
        self.framebuffers.remove(&framebuffer.as_raw());
    }

    pub fn record_render_pass(
        &mut self,
        render_pass: vk::RenderPass,
        final_layouts: Vec<vk::ImageLayout>,
    ) {
        self.render_passes
            .insert(render_pass.as_raw(), RenderPassRecord { final_layouts });
    }

    pub fn forget_render_pass(&mut self, render_pass: vk::RenderPass) {
        self.render_passes.remove(&render_pass.as_raw());
    }

    /// Resolve a begun pass to its attachments. `None` whenever anything is
    /// unknown or mismatched (imageless framebuffer, an unintercepted create,
    /// attachment-count disagreement, no color attachment) — the caller then
    /// simply doesn't composite, it never guesses.
    pub fn resolve(
        &self,
        render_pass: vk::RenderPass,
        framebuffer: vk::Framebuffer,
    ) -> Option<ResolvedFrame> {
        let framebuffer = self.framebuffers.get(&framebuffer.as_raw())?;
        let render_pass = self.render_passes.get(&render_pass.as_raw())?;
        if framebuffer.attachments.len() != render_pass.final_layouts.len() {
            return None;
        }
        let mut color = None;
        let mut depth = None;
        for (&view, &final_layout) in framebuffer
            .attachments
            .iter()
            .zip(&render_pass.final_layouts)
        {
            let record = self.views.get(&view.as_raw())?;
            let resolved = ResolvedAttachment {
                image: record.image,
                view,
                format: record.format,
                final_layout,
            };
            if record.aspect.contains(vk::ImageAspectFlags::COLOR) && color.is_none() {
                color = Some(resolved);
            } else if record.aspect.contains(vk::ImageAspectFlags::DEPTH) && depth.is_none() {
                depth = Some(resolved);
            }
        }
        Some(ResolvedFrame {
            extent: framebuffer.extent,
            color: color?,
            depth,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXTENT: vk::Extent2D = vk::Extent2D {
        width: 64,
        height: 32,
    };

    fn view(raw: u64) -> vk::ImageView {
        vk::ImageView::from_raw(raw)
    }

    fn registry_with_scene() -> TapRegistry {
        let mut registry = TapRegistry::default();
        registry.record_view(
            view(1),
            ViewRecord {
                image: vk::Image::from_raw(10),
                format: vk::Format::R8G8B8A8_UNORM,
                aspect: vk::ImageAspectFlags::COLOR,
            },
        );
        registry.record_view(
            view(2),
            ViewRecord {
                image: vk::Image::from_raw(20),
                format: vk::Format::D32_SFLOAT,
                aspect: vk::ImageAspectFlags::DEPTH,
            },
        );
        registry.record_framebuffer(
            vk::Framebuffer::from_raw(100),
            vec![view(1), view(2)],
            EXTENT,
        );
        registry.record_render_pass(
            vk::RenderPass::from_raw(200),
            vec![
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            ],
        );
        registry
    }

    #[test]
    fn resolves_color_and_depth_with_the_begun_pass_layouts() {
        let registry = registry_with_scene();
        let frame = registry
            .resolve(
                vk::RenderPass::from_raw(200),
                vk::Framebuffer::from_raw(100),
            )
            .expect("fully recorded pass resolves");
        assert_eq!(frame.extent, EXTENT);
        assert_eq!(frame.color.image, vk::Image::from_raw(10));
        assert_eq!(
            frame.color.final_layout,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
        let depth = frame.depth.expect("depth attachment present");
        assert_eq!(depth.image, vk::Image::from_raw(20));
        assert_eq!(depth.format, vk::Format::D32_SFLOAT);
        assert_eq!(
            depth.final_layout,
            vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
        );
    }

    #[test]
    fn depth_only_pass_resolves_to_none_not_a_guess() {
        let mut registry = TapRegistry::default();
        registry.record_view(
            view(2),
            ViewRecord {
                image: vk::Image::from_raw(20),
                format: vk::Format::D32_SFLOAT,
                aspect: vk::ImageAspectFlags::DEPTH,
            },
        );
        registry.record_framebuffer(vk::Framebuffer::from_raw(100), vec![view(2)], EXTENT);
        registry.record_render_pass(
            vk::RenderPass::from_raw(200),
            vec![vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL],
        );
        assert!(
            registry
                .resolve(
                    vk::RenderPass::from_raw(200),
                    vk::Framebuffer::from_raw(100)
                )
                .is_none(),
            "a pass with no color attachment has nothing to composite into"
        );
    }

    #[test]
    fn anything_unknown_resolves_to_none() {
        let registry = registry_with_scene();
        // Unknown render pass / framebuffer handles.
        assert!(
            registry
                .resolve(
                    vk::RenderPass::from_raw(999),
                    vk::Framebuffer::from_raw(100)
                )
                .is_none()
        );
        assert!(
            registry
                .resolve(
                    vk::RenderPass::from_raw(200),
                    vk::Framebuffer::from_raw(999)
                )
                .is_none()
        );

        // Attachment-count mismatch between the pass and the framebuffer.
        let mut mismatched = registry_with_scene();
        mismatched.record_render_pass(
            vk::RenderPass::from_raw(200),
            vec![vk::ImageLayout::TRANSFER_SRC_OPTIMAL],
        );
        assert!(
            mismatched
                .resolve(
                    vk::RenderPass::from_raw(200),
                    vk::Framebuffer::from_raw(100)
                )
                .is_none()
        );

        // A view the layer never saw created (or already saw destroyed).
        let mut forgotten = registry_with_scene();
        forgotten.forget_view(view(2));
        assert!(
            forgotten
                .resolve(
                    vk::RenderPass::from_raw(200),
                    vk::Framebuffer::from_raw(100)
                )
                .is_none()
        );
    }

    #[test]
    fn forgetting_framebuffers_and_passes_unresolves_them() {
        let mut registry = registry_with_scene();
        registry.forget_framebuffer(vk::Framebuffer::from_raw(100));
        assert!(
            registry
                .resolve(
                    vk::RenderPass::from_raw(200),
                    vk::Framebuffer::from_raw(100)
                )
                .is_none()
        );
        let mut registry = registry_with_scene();
        registry.forget_render_pass(vk::RenderPass::from_raw(200));
        assert!(
            registry
                .resolve(
                    vk::RenderPass::from_raw(200),
                    vk::Framebuffer::from_raw(100)
                )
                .is_none()
        );
    }
}
