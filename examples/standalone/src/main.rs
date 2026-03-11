use anyhow::{Context, Result};
use ferridian_core::SurfaceRenderer;
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

fn main() -> Result<()> {
    let event_loop = EventLoop::new().context("failed to create event loop")?;
    let mut app = StandaloneApp::default();
    event_loop
        .run_app(&mut app)
        .context("standalone example exited unexpectedly")
}

#[derive(Default)]
struct StandaloneApp {
    window: Option<Arc<Window>>,
    renderer: Option<SurfaceRenderer>,
    started_at: Option<Instant>,
}

impl ApplicationHandler for StandaloneApp {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        let attributes = WindowAttributes::default()
            .with_title("Ferridian Standalone")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .expect("failed to create standalone window"),
        );
        let size = window.inner_size();
        let renderer = pollster::block_on(SurfaceRenderer::new(
            window.clone(),
            size.width,
            size.height,
            wgpu::Color {
                r: 0.04,
                g: 0.08,
                b: 0.11,
                a: 1.0,
            },
        ))
        .expect("failed to initialize standalone renderer");

        self.started_at = Some(Instant::now());
        window.request_redraw();
        self.renderer = Some(renderer);
        self.window = Some(window);
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(window) = self.window.as_ref() else {
            return;
        };

        if window.id() != window_id {
            return;
        }

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                if let Some(renderer) = self.renderer.as_mut() {
                    renderer.resize(size.width, size.height);
                }
                window.request_redraw();
            }
            WindowEvent::RedrawRequested => {
                let Some(renderer) = self.renderer.as_mut() else {
                    return;
                };
                let elapsed = self
                    .started_at
                    .as_ref()
                    .map(Instant::elapsed)
                    .map(|duration| duration.as_secs_f32())
                    .unwrap_or(0.0);

                match renderer.render(elapsed) {
                    Ok(()) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        renderer.reconfigure();
                        window.request_redraw();
                    }
                    Err(wgpu::SurfaceError::OutOfMemory) => event_loop.exit(),
                    Err(wgpu::SurfaceError::Timeout) => window.request_redraw(),
                    Err(wgpu::SurfaceError::Other) => window.request_redraw(),
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_ref() {
            window.request_redraw();
        }
    }
}
