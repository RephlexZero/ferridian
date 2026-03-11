use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use ferridian_utils::WorkspaceMetadata;
use glam::Vec3;
use wgpu::util::DeviceExt;

pub struct RenderBackendPlan {
    pub primary_backend: &'static str,
    pub fallback_backend: &'static str,
    pub target_passes: &'static [&'static str],
    pub sun_direction: Vec3,
}

impl Default for RenderBackendPlan {
    fn default() -> Self {
        Self {
            primary_backend: "wgpu-vulkan",
            fallback_backend: "wgpu-metal-dx12",
            target_passes: &[
                "shadow-map",
                "gbuffer-fill",
                "deferred-lighting",
                "translucency",
                "volumetrics",
                "taa",
                "final-compose",
            ],
            sun_direction: Vec3::new(0.3, 0.85, 0.2).normalize(),
        }
    }
}

pub fn workspace_metadata() -> WorkspaceMetadata {
    WorkspaceMetadata::new("ferridian", "minecraft-vulkan-shader-engine")
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TriangleVertex {
    position: [f32; 2],
    color: [f32; 3],
}

impl TriangleVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 2] =
        wgpu::vertex_attr_array![0 => Float32x2, 1 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct TriangleUniform {
    time_seconds: f32,
    aspect_ratio: f32,
    _padding: [f32; 2],
}

impl TriangleUniform {
    fn new(width: u32, height: u32) -> Self {
        Self {
            time_seconds: 0.0,
            aspect_ratio: aspect_ratio(width, height),
            _padding: [0.0; 2],
        }
    }
}

pub struct FrameStats {
    pub width: u32,
    pub height: u32,
    pub aspect_ratio: f32,
}

pub struct SurfaceRenderer {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    clear_color: wgpu::Color,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    vertex_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    uniform: TriangleUniform,
}

impl SurfaceRenderer {
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        clear_color: wgpu::Color,
    ) -> Result<Self> {
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(target)
            .context("failed to create wgpu surface")?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .context("failed to find a compatible GPU adapter")?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ferridian Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create logical device")?;

        let capabilities = surface.get_capabilities(&adapter);
        let format = capabilities
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .or_else(|| capabilities.formats.first().copied())
            .ok_or_else(|| anyhow!("surface reported no supported formats"))?;
        let present_mode = capabilities
            .present_modes
            .iter()
            .copied()
            .find(|mode| *mode == wgpu::PresentMode::AutoVsync)
            .or_else(|| capabilities.present_modes.first().copied())
            .ok_or_else(|| anyhow!("surface reported no supported present modes"))?;
        let alpha_mode = capabilities
            .alpha_modes
            .first()
            .copied()
            .ok_or_else(|| anyhow!("surface reported no supported alpha modes"))?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: width.max(1),
            height: height.max(1),
            present_mode,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        let uniform = TriangleUniform::new(config.width, config.height);
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ferridian Triangle Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Ferridian Triangle Uniform Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ferridian Triangle Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Ferridian Standalone Shader"),
            source: wgpu::ShaderSource::Wgsl(
                include_str!("../../../shaders/wgsl/standalone.wgsl").into(),
            ),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ferridian Triangle Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ferridian Triangle Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[TriangleVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::REPLACE),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertices = triangle_vertices();
        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ferridian Triangle Vertex Buffer"),
            contents: bytemuck::cast_slice(&vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });

        Ok(Self {
            _instance: instance,
            surface,
            device,
            queue,
            config,
            clear_color,
            render_pipeline,
            vertex_buffer,
            vertex_count: vertices.len() as u32,
            uniform_buffer,
            uniform_bind_group,
            uniform,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.uniform.aspect_ratio = aspect_ratio(width, height);
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));
        self.surface.configure(&self.device, &self.config);
    }

    pub fn render(&mut self, time_seconds: f32) -> std::result::Result<(), wgpu::SurfaceError> {
        self.uniform.time_seconds = time_seconds;
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&self.uniform));

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ferridian Clear Encoder"),
            });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Ferridian Clear Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.draw(0..self.vertex_count, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    pub fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
    }

    pub fn frame_stats(&self) -> FrameStats {
        FrameStats {
            width: self.config.width,
            height: self.config.height,
            aspect_ratio: self.uniform.aspect_ratio,
        }
    }
}

fn triangle_vertices() -> [TriangleVertex; 3] {
    [
        TriangleVertex {
            position: [0.0, 0.72],
            color: [0.95, 0.52, 0.18],
        },
        TriangleVertex {
            position: [-0.7, -0.56],
            color: [0.15, 0.72, 0.92],
        },
        TriangleVertex {
            position: [0.7, -0.56],
            color: [0.44, 0.89, 0.46],
        },
    ]
}

fn aspect_ratio(width: u32, height: u32) -> f32 {
    width.max(1) as f32 / height.max(1) as f32
}
