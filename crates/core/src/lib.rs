use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use ferridian_shader::{ShaderAsset, ShaderPipelineMetadata};
use ferridian_utils::WorkspaceMetadata;
use glam::{Mat4, Quat, Vec3};
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

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub target: Vec3,
    pub distance: f32,
    pub yaw_radians: f32,
    pub pitch_radians: f32,
    pub fov_y_radians: f32,
    pub z_near: f32,
    pub z_far: f32,
}

impl Camera {
    pub fn orbiting(time_seconds: f32) -> Self {
        Self {
            target: Vec3::new(1.5, 0.45, 1.5),
            distance: 9.5,
            yaw_radians: time_seconds * 0.35,
            pitch_radians: 0.58,
            fov_y_radians: 50.0_f32.to_radians(),
            z_near: 0.1,
            z_far: 100.0,
        }
    }

    pub fn eye(&self) -> Vec3 {
        let yaw = self.yaw_radians;
        let pitch = self.pitch_radians.clamp(-1.2, 1.2);
        let direction = Vec3::new(
            yaw.cos() * pitch.cos(),
            pitch.sin(),
            yaw.sin() * pitch.cos(),
        );
        self.target + direction.normalize() * self.distance
    }

    pub fn view_projection_matrix(&self, aspect_ratio: f32) -> Mat4 {
        let view = Mat4::look_at_rh(self.eye(), self.target, Vec3::Y);
        let projection =
            Mat4::perspective_rh(self.fov_y_radians, aspect_ratio, self.z_near, self.z_far);
        projection * view
    }
}

impl Default for Camera {
    fn default() -> Self {
        Self::orbiting(0.0)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub base_color_factor: Vec3,
}

impl Default for Material {
    fn default() -> Self {
        Self {
            base_color_factor: Vec3::ONE,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }
}

impl Default for Transform {
    fn default() -> Self {
        Self {
            translation: Vec3::ZERO,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MeshVertex {
    position: [f32; 3],
    normal: [f32; 3],
    color: [f32; 3],
}

impl MeshVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x3];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub material: Material,
    pub transform: Transform,
}

impl Mesh {
    pub fn demo_voxel_chunk() -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();

        for x in 0..4 {
            for z in 0..4 {
                let height = ((x + z) % 3) + 1;
                for y in 0..height {
                    let tint = voxel_tint(x as f32, y as f32, z as f32);
                    append_cube(
                        &mut vertices,
                        &mut indices,
                        Vec3::new(x as f32, y as f32, z as f32),
                        tint,
                    );
                }
            }
        }

        Self {
            vertices,
            indices,
            material: Material {
                base_color_factor: Vec3::new(0.98, 0.96, 1.0),
            },
            transform: Transform {
                translation: Vec3::new(-1.5, -1.0, -1.5),
                ..Transform::default()
            },
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct SceneUniform {
    view_projection: [[f32; 4]; 4],
    model: [[f32; 4]; 4],
    light_direction: [f32; 4],
    tint_and_time: [f32; 4],
}

impl SceneUniform {
    fn new(width: u32, height: u32, mesh: &Mesh, time_seconds: f32) -> Self {
        let camera = Camera::orbiting(time_seconds);
        let aspect = aspect_ratio(width, height);
        Self {
            view_projection: camera.view_projection_matrix(aspect).to_cols_array_2d(),
            model: mesh.transform.matrix().to_cols_array_2d(),
            light_direction: [0.35, 0.8, 0.28, 0.0],
            tint_and_time: [
                mesh.material.base_color_factor.x,
                mesh.material.base_color_factor.y,
                mesh.material.base_color_factor.z,
                time_seconds,
            ],
        }
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct FrameUniform {
    time_seconds: f32,
    aspect_ratio: f32,
    _padding: [f32; 2],
}

impl FrameUniform {
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
    pub index_count: u32,
}

struct DepthTarget {
    _texture: wgpu::Texture,
    view: wgpu::TextureView,
}

impl DepthTarget {
    const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;

    fn new(device: &wgpu::Device, config: &wgpu::SurfaceConfiguration) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Ferridian Depth Target"),
            size: wgpu::Extent3d {
                width: config.width.max(1),
                height: config.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: Self::FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Self {
            _texture: texture,
            view,
        }
    }
}

pub struct SurfaceRenderer {
    _instance: wgpu::Instance,
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    clear_color: wgpu::Color,
    shader_asset: ShaderAsset,
    render_pipeline: wgpu::RenderPipeline,
    vertex_buffer: wgpu::Buffer,
    index_buffer: wgpu::Buffer,
    index_count: u32,
    uniform_buffer: wgpu::Buffer,
    uniform_bind_group: wgpu::BindGroup,
    mesh: Mesh,
    depth_target: DepthTarget,
    frame_uniform: FrameUniform,
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

        let mesh = Mesh::demo_voxel_chunk();
        let scene_uniform = SceneUniform::new(config.width, config.height, &mesh, 0.0);
        let frame_uniform = FrameUniform::new(config.width, config.height);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ferridian Scene Uniform Buffer"),
            contents: bytemuck::bytes_of(&scene_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Ferridian Scene Uniform Layout"),
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Ferridian Scene Uniform Bind Group"),
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniform_buffer.as_entire_binding(),
            }],
        });

        let shader_asset = ShaderAsset::load_workspace_wgsl(
            "shaders/wgsl/standalone.wgsl",
            ShaderPipelineMetadata::standalone_voxel(),
        )?;
        let shader = shader_asset.create_module(&device);
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Ferridian Voxel Pipeline Layout"),
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Ferridian Voxel Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some(&shader_asset.metadata.vertex_entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[MeshVertex::layout()],
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some(&shader_asset.metadata.fragment_entry),
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
            depth_stencil: Some(wgpu::DepthStencilState {
                format: DepthTarget::FORMAT,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ferridian Voxel Vertex Buffer"),
            contents: bytemuck::cast_slice(mesh.vertices.as_slice()),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Ferridian Voxel Index Buffer"),
            contents: bytemuck::cast_slice(mesh.indices.as_slice()),
            usage: wgpu::BufferUsages::INDEX,
        });
        let depth_target = DepthTarget::new(&device, &config);

        Ok(Self {
            _instance: instance,
            surface,
            device,
            queue,
            config,
            clear_color,
            shader_asset,
            render_pipeline,
            vertex_buffer,
            index_buffer,
            index_count: mesh.indices.len() as u32,
            uniform_buffer,
            uniform_bind_group,
            mesh,
            depth_target,
            frame_uniform,
        })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }

        self.config.width = width;
        self.config.height = height;
        self.frame_uniform.aspect_ratio = aspect_ratio(width, height);
        self.surface.configure(&self.device, &self.config);
        self.depth_target = DepthTarget::new(&self.device, &self.config);
    }

    pub fn render(&mut self, time_seconds: f32) -> std::result::Result<(), wgpu::SurfaceError> {
        self.frame_uniform.time_seconds = time_seconds;
        let scene_uniform = SceneUniform::new(
            self.config.width,
            self.config.height,
            &Mesh {
                transform: Transform {
                    rotation: Quat::from_rotation_y(time_seconds * 0.25),
                    ..self.mesh.transform
                },
                material: self.mesh.material,
                vertices: Vec::new(),
                indices: Vec::new(),
            },
            time_seconds,
        );
        self.queue
            .write_buffer(&self.uniform_buffer, 0, bytemuck::bytes_of(&scene_uniform));

        let _ = self.shader_asset.reload_if_changed();

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
                label: Some("Ferridian Voxel Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_target.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.render_pipeline);
            pass.set_bind_group(0, &self.uniform_bind_group, &[]);
            pass.set_vertex_buffer(0, self.vertex_buffer.slice(..));
            pass.set_index_buffer(self.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..self.index_count, 0, 0..1);
        }

        self.queue.submit(std::iter::once(encoder.finish()));
        frame.present();
        Ok(())
    }

    pub fn reconfigure(&mut self) {
        self.surface.configure(&self.device, &self.config);
        self.depth_target = DepthTarget::new(&self.device, &self.config);
    }

    pub fn frame_stats(&self) -> FrameStats {
        FrameStats {
            width: self.config.width,
            height: self.config.height,
            aspect_ratio: self.frame_uniform.aspect_ratio,
            index_count: self.index_count,
        }
    }
}

fn aspect_ratio(width: u32, height: u32) -> f32 {
    width.max(1) as f32 / height.max(1) as f32
}

fn voxel_tint(x: f32, y: f32, z: f32) -> [f32; 3] {
    [0.32 + x * 0.11, 0.45 + y * 0.16, 0.58 + z * 0.08]
}

fn append_cube(
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
    offset: Vec3,
    color: [f32; 3],
) {
    const FACE_DEFINITIONS: [([f32; 3], [[f32; 3]; 4]); 6] = [
        (
            [0.0, 0.0, 1.0],
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 1.0],
                [0.0, 1.0, 1.0],
            ],
        ),
        (
            [0.0, 0.0, -1.0],
            [
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
        ),
        (
            [1.0, 0.0, 0.0],
            [
                [1.0, 0.0, 1.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [1.0, 1.0, 1.0],
            ],
        ),
        (
            [-1.0, 0.0, 0.0],
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 1.0, 0.0],
            ],
        ),
        (
            [0.0, 1.0, 0.0],
            [
                [0.0, 1.0, 1.0],
                [1.0, 1.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        ),
        (
            [0.0, -1.0, 0.0],
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
        ),
    ];

    for (normal, corners) in FACE_DEFINITIONS {
        let base_index = vertices.len() as u32;
        for corner in corners {
            vertices.push(MeshVertex {
                position: [
                    corner[0] + offset.x,
                    corner[1] + offset.y,
                    corner[2] + offset.z,
                ],
                normal,
                color,
            });
        }
        indices.extend_from_slice(&[
            base_index,
            base_index + 1,
            base_index + 2,
            base_index,
            base_index + 2,
            base_index + 3,
        ]);
    }
}

#[cfg(test)]
mod tests {
    use super::{Camera, Mesh};

    #[test]
    fn demo_voxel_chunk_contains_indexed_geometry() {
        let mesh = Mesh::demo_voxel_chunk();
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn orbit_camera_produces_finite_eye_position() {
        let camera = Camera::orbiting(1.25);
        let eye = camera.eye();
        assert!(eye.is_finite());
    }
}
