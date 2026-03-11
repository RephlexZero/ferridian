use anyhow::{Context, Result, anyhow};
use bytemuck::{Pod, Zeroable};
use ferridian_shader::{ShaderAsset, ShaderPipelineMetadata};
use ferridian_utils::WorkspaceMetadata;
use glam::{Mat4, Quat, Vec3};
use rayon::prelude::*;
use wgpu::util::DeviceExt;

// ---------------------------------------------------------------------------
// Backend isolation — wraps wgpu adapter/device creation so the rest of the
// codebase does not spread version-specific configuration everywhere.
// ---------------------------------------------------------------------------

/// Configuration for the wgpu backend, centralised so upgrades only touch one
/// place.
#[derive(Clone, Debug)]
pub struct BackendConfig {
    pub power_preference: wgpu::PowerPreference,
    pub required_features: wgpu::Features,
    pub required_limits: wgpu::Limits,
    pub experimental: bool,
}

impl Default for BackendConfig {
    fn default() -> Self {
        Self {
            power_preference: wgpu::PowerPreference::HighPerformance,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            experimental: false,
        }
    }
}

impl BackendConfig {
    pub async fn request_device(
        &self,
        adapter: &wgpu::Adapter,
    ) -> Result<(wgpu::Device, wgpu::Queue)> {
        let experimental_features = if self.experimental {
            // SAFETY: experimental features are opt-in through BackendConfig.
            unsafe { wgpu::ExperimentalFeatures::enabled() }
        } else {
            wgpu::ExperimentalFeatures::disabled()
        };
        adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("Ferridian Device"),
                required_features: self.required_features,
                required_limits: self.required_limits.clone(),
                experimental_features,
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .context("failed to create logical device")
    }

    pub fn adapter_options<'a>(
        &self,
        surface: &'a wgpu::Surface<'static>,
    ) -> wgpu::RequestAdapterOptions<'a, 'a> {
        wgpu::RequestAdapterOptions {
            power_preference: self.power_preference,
            compatible_surface: Some(surface),
            force_fallback_adapter: false,
        }
    }

    /// Returns adapter options that force a software/fallback adapter.
    /// Useful for validation testing and CI headless environments.
    pub fn fallback_adapter_options<'a>(
        &self,
        surface: &'a wgpu::Surface<'static>,
    ) -> wgpu::RequestAdapterOptions<'a, 'a> {
        wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::LowPower,
            compatible_surface: Some(surface),
            force_fallback_adapter: true,
        }
    }

    /// Returns adapter options for headless operation (no surface).
    pub fn headless_adapter_options(&self) -> wgpu::RequestAdapterOptions<'static, 'static> {
        wgpu::RequestAdapterOptions {
            power_preference: self.power_preference,
            compatible_surface: None,
            force_fallback_adapter: false,
        }
    }
}

/// Describes the capabilities of the active backend, discovered at init time.
#[derive(Clone, Debug)]
pub struct BackendCapabilities {
    pub adapter_name: String,
    pub backend: wgpu::Backend,
    pub features: wgpu::Features,
}

impl BackendCapabilities {
    pub fn from_adapter(adapter: &wgpu::Adapter) -> Self {
        let info = adapter.get_info();
        Self {
            adapter_name: info.name.clone(),
            backend: info.backend,
            features: adapter.features(),
        }
    }
}

/// A shared renderer configuration that can drive either standalone or
/// JNI-hosted entry points identically.
#[derive(Clone, Debug)]
pub struct SharedRendererConfig {
    pub width: u32,
    pub height: u32,
    pub clear_color: wgpu::Color,
    pub backend: BackendConfig,
}

#[derive(Clone, Debug)]
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

// ---------------------------------------------------------------------------
// PBR material definitions – evolving toward LabPBR resource-pack conventions.
// ---------------------------------------------------------------------------

/// A full PBR material definition that can express standard metallic-roughness
/// properties as well as LabPBR-extended fields used by advanced shader packs.
#[derive(Clone, Debug)]
pub struct MaterialDefinition {
    pub name: String,
    pub albedo: [f32; 4],
    pub pbr: PbrProperties,
    pub lab_pbr: Option<LabPbrExtension>,
}

#[derive(Clone, Copy, Debug)]
pub struct PbrProperties {
    pub metallic: f32,
    pub roughness: f32,
    pub reflectance: f32,
    pub emissive_intensity: f32,
}

impl Default for PbrProperties {
    fn default() -> Self {
        Self {
            metallic: 0.0,
            roughness: 0.5,
            reflectance: 0.5,
            emissive_intensity: 0.0,
        }
    }
}

/// LabPBR extension fields for shader packs that consume the LabPBR convention.
#[derive(Clone, Copy, Debug)]
pub struct LabPbrExtension {
    pub porosity: f32,
    pub subsurface_scattering: f32,
    pub emission: f32,
}

impl Default for LabPbrExtension {
    fn default() -> Self {
        Self {
            porosity: 0.0,
            subsurface_scattering: 0.0,
            emission: 0.0,
        }
    }
}

impl MaterialDefinition {
    pub fn opaque(name: impl Into<String>, albedo: [f32; 3]) -> Self {
        Self {
            name: name.into(),
            albedo: [albedo[0], albedo[1], albedo[2], 1.0],
            pbr: PbrProperties::default(),
            lab_pbr: None,
        }
    }

    pub fn emissive(name: impl Into<String>, albedo: [f32; 3], intensity: f32) -> Self {
        Self {
            name: name.into(),
            albedo: [albedo[0], albedo[1], albedo[2], 1.0],
            pbr: PbrProperties {
                emissive_intensity: intensity,
                ..PbrProperties::default()
            },
            lab_pbr: Some(LabPbrExtension {
                emission: intensity,
                ..LabPbrExtension::default()
            }),
        }
    }

    pub fn is_emissive(&self) -> bool {
        self.pbr.emissive_intensity > 0.0
    }

    pub fn is_translucent(&self) -> bool {
        self.albedo[3] < 1.0
    }
}

// ---------------------------------------------------------------------------
// G-buffer layout — explicit render target configuration for deferred shading.
// ---------------------------------------------------------------------------

/// Describes the texture formats and sizes for a G-buffer used by the deferred
/// shading pipeline.
#[derive(Clone, Debug)]
pub struct GBufferLayout {
    pub albedo_format: wgpu::TextureFormat,
    pub normal_format: wgpu::TextureFormat,
    pub material_format: wgpu::TextureFormat,
    pub depth_format: wgpu::TextureFormat,
}

impl Default for GBufferLayout {
    fn default() -> Self {
        Self {
            albedo_format: wgpu::TextureFormat::Rgba8UnormSrgb,
            normal_format: wgpu::TextureFormat::Rgba16Float,
            material_format: wgpu::TextureFormat::Rgba8Unorm,
            depth_format: wgpu::TextureFormat::Depth24Plus,
        }
    }
}

/// Holds the actual GPU textures and views backing a G-buffer.
pub struct GBufferTargets {
    pub albedo_view: wgpu::TextureView,
    pub normal_view: wgpu::TextureView,
    pub material_view: wgpu::TextureView,
    pub depth_view: wgpu::TextureView,
    pub width: u32,
    pub height: u32,
}

impl GBufferTargets {
    pub fn new(device: &wgpu::Device, width: u32, height: u32, layout: &GBufferLayout) -> Self {
        let w = width.max(1);
        let h = height.max(1);
        let albedo_view =
            Self::create_target(device, w, h, layout.albedo_format, "G-Buffer Albedo");
        let normal_view =
            Self::create_target(device, w, h, layout.normal_format, "G-Buffer Normal");
        let material_view =
            Self::create_target(device, w, h, layout.material_format, "G-Buffer Material");
        let depth_view = Self::create_target(device, w, h, layout.depth_format, "G-Buffer Depth");
        Self {
            albedo_view,
            normal_view,
            material_view,
            depth_view,
            width: w,
            height: h,
        }
    }

    fn create_target(
        device: &wgpu::Device,
        width: u32,
        height: u32,
        format: wgpu::TextureFormat,
        label: &str,
    ) -> wgpu::TextureView {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        texture.create_view(&wgpu::TextureViewDescriptor::default())
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
    packed_position: u32,
    packed_normal: u32,
    packed_color: u32,
}

impl MeshVertex {
    const ATTRIBUTES: [wgpu::VertexAttribute; 3] =
        wgpu::vertex_attr_array![0 => Uint32, 1 => Uint32, 2 => Uint32];

    fn layout() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Self>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &Self::ATTRIBUTES,
        }
    }

    fn new(position: [u32; 3], normal_index: u32, color: [f32; 3]) -> Self {
        Self {
            packed_position: pack_position(position),
            packed_normal: normal_index,
            packed_color: pack_color(color),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BlockKind {
    Air,
    Grass,
    Dirt,
    Stone,
}

impl BlockKind {
    fn is_solid(self) -> bool {
        !matches!(self, Self::Air)
    }

    fn color(self, y: usize) -> [f32; 3] {
        match self {
            Self::Air => [0.0, 0.0, 0.0],
            Self::Grass => [0.22, 0.55 + y as f32 * 0.01, 0.25],
            Self::Dirt => [0.42, 0.29, 0.18],
            Self::Stone => [0.48 + y as f32 * 0.002, 0.5, 0.53],
        }
    }
}

#[derive(Debug, Clone)]
pub struct ChunkSection {
    width: usize,
    height: usize,
    depth: usize,
    blocks: Vec<BlockKind>,
}

impl ChunkSection {
    pub fn new(width: usize, height: usize, depth: usize) -> Self {
        Self {
            width,
            height,
            depth,
            blocks: vec![BlockKind::Air; width * height * depth],
        }
    }

    pub fn sample_terrain() -> Self {
        let mut section = Self::new(16, 16, 16);

        for x in 0..section.width {
            for z in 0..section.depth {
                let ridge = ((x as i32 - 8).abs() + (z as i32 - 8).abs()) as usize / 3;
                let wave = ((x * 13 + z * 7) % 5) + ((x + z) % 3);
                let top = (3 + wave)
                    .saturating_sub(ridge.min(3))
                    .min(section.height - 1);

                for y in 0..=top {
                    let block = if y == top {
                        BlockKind::Grass
                    } else if y + 2 >= top {
                        BlockKind::Dirt
                    } else {
                        BlockKind::Stone
                    };
                    section.set_block(x, y, z, block);
                }
            }
        }

        section
    }

    pub fn dimensions(&self) -> (usize, usize, usize) {
        (self.width, self.height, self.depth)
    }

    pub fn block(&self, x: usize, y: usize, z: usize) -> BlockKind {
        self.blocks[self.index(x, y, z)]
    }

    pub fn set_block(&mut self, x: usize, y: usize, z: usize, block: BlockKind) {
        let index = self.index(x, y, z);
        self.blocks[index] = block;
    }

    fn index(&self, x: usize, y: usize, z: usize) -> usize {
        debug_assert!(x < self.width);
        debug_assert!(y < self.height);
        debug_assert!(z < self.depth);
        (y * self.depth + z) * self.width + x
    }

    fn block_or_air(&self, x: isize, y: isize, z: isize) -> BlockKind {
        if x < 0
            || y < 0
            || z < 0
            || x as usize >= self.width
            || y as usize >= self.height
            || z as usize >= self.depth
        {
            BlockKind::Air
        } else {
            self.block(x as usize, y as usize, z as usize)
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct FaceDefinition {
    normal_index: u32,
    corners: [[u32; 3]; 4],
    neighbor_offset: [isize; 3],
}

const FACE_DEFINITIONS: [FaceDefinition; 6] = [
    FaceDefinition {
        normal_index: 0,
        corners: [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]],
        neighbor_offset: [0, 0, 1],
    },
    FaceDefinition {
        normal_index: 1,
        corners: [[1, 0, 0], [0, 0, 0], [0, 1, 0], [1, 1, 0]],
        neighbor_offset: [0, 0, -1],
    },
    FaceDefinition {
        normal_index: 2,
        corners: [[1, 0, 1], [1, 0, 0], [1, 1, 0], [1, 1, 1]],
        neighbor_offset: [1, 0, 0],
    },
    FaceDefinition {
        normal_index: 3,
        corners: [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],
        neighbor_offset: [-1, 0, 0],
    },
    FaceDefinition {
        normal_index: 4,
        corners: [[0, 1, 1], [1, 1, 1], [1, 1, 0], [0, 1, 0]],
        neighbor_offset: [0, 1, 0],
    },
    FaceDefinition {
        normal_index: 5,
        corners: [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],
        neighbor_offset: [0, -1, 0],
    },
];

#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<MeshVertex>,
    pub indices: Vec<u32>,
    pub material: Material,
    pub transform: Transform,
}

impl Mesh {
    #[profiling::function]
    pub fn from_chunk_section(section: &ChunkSection) -> Self {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let (width, height, depth) = section.dimensions();

        for y in 0..height {
            for z in 0..depth {
                for x in 0..width {
                    let block = section.block(x, y, z);
                    if !block.is_solid() {
                        continue;
                    }

                    let offset = [x as u32, y as u32, z as u32];
                    let color = block.color(y);

                    for face in FACE_DEFINITIONS {
                        let neighbor = section.block_or_air(
                            x as isize + face.neighbor_offset[0],
                            y as isize + face.neighbor_offset[1],
                            z as isize + face.neighbor_offset[2],
                        );
                        if neighbor.is_solid() {
                            continue;
                        }
                        append_face(&mut vertices, &mut indices, offset, color, face);
                    }
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

    /// Parallel version of `from_chunk_section` using Rayon.
    /// Meshes each column (x, z) in parallel, then merges results.
    #[profiling::function]
    pub fn from_chunk_section_parallel(section: &ChunkSection) -> Self {
        let (width, height, depth) = section.dimensions();
        let columns: Vec<(usize, usize)> = (0..width)
            .flat_map(|x| (0..depth).map(move |z| (x, z)))
            .collect();

        let column_meshes: Vec<(Vec<MeshVertex>, Vec<u32>)> = columns
            .par_iter()
            .map(|&(x, z)| {
                let mut verts = Vec::new();
                let mut idxs = Vec::new();
                for y in 0..height {
                    let block = section.block(x, y, z);
                    if !block.is_solid() {
                        continue;
                    }
                    let offset = [x as u32, y as u32, z as u32];
                    let color = block.color(y);
                    for face in FACE_DEFINITIONS {
                        let neighbor = section.block_or_air(
                            x as isize + face.neighbor_offset[0],
                            y as isize + face.neighbor_offset[1],
                            z as isize + face.neighbor_offset[2],
                        );
                        if neighbor.is_solid() {
                            continue;
                        }
                        append_face(&mut verts, &mut idxs, offset, color, face);
                    }
                }
                (verts, idxs)
            })
            .collect();

        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        for (col_verts, col_idxs) in column_meshes {
            let base = vertices.len() as u32;
            vertices.extend(col_verts);
            indices.extend(col_idxs.iter().map(|i| i + base));
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

    pub fn terrain_chunk_demo() -> Self {
        Self::from_chunk_section(&ChunkSection::sample_terrain())
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
    pub pass_count: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RenderPassKind {
    OpaqueTerrain,
    GBufferFill,
    DeferredLighting,
    Translucent,
    ShadowMap,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScheduledPass {
    pub kind: RenderPassKind,
    pub label: &'static str,
}

#[derive(Clone, Debug)]
pub struct PassScheduler {
    passes: Vec<ScheduledPass>,
}

impl PassScheduler {
    pub fn standalone_voxel() -> Self {
        Self {
            passes: vec![ScheduledPass {
                kind: RenderPassKind::OpaqueTerrain,
                label: "Ferridian Opaque Terrain Pass",
            }],
        }
    }

    /// A deferred rendering pipeline schedule:
    /// 1. G-buffer fill — write albedo, normal, material to MRT
    /// 2. Deferred lighting — read G-buffer, evaluate lights
    /// 3. Opaque terrain — forward composite (current implementation)
    /// 4. Translucent — alpha-blended geometry (water, glass, particles)
    pub fn deferred() -> Self {
        Self {
            passes: vec![
                ScheduledPass {
                    kind: RenderPassKind::GBufferFill,
                    label: "Ferridian G-Buffer Fill Pass",
                },
                ScheduledPass {
                    kind: RenderPassKind::DeferredLighting,
                    label: "Ferridian Deferred Lighting Pass",
                },
                ScheduledPass {
                    kind: RenderPassKind::Translucent,
                    label: "Ferridian Translucent Pass",
                },
            ],
        }
    }

    /// A full deferred pipeline with shadow maps up front.
    pub fn deferred_with_shadows(cascade_count: u32) -> Self {
        let mut passes = Vec::with_capacity(3 + cascade_count as usize);
        for i in 0..cascade_count {
            passes.push(ScheduledPass {
                kind: RenderPassKind::ShadowMap,
                label: if i == 0 {
                    "Ferridian Shadow Cascade 0"
                } else if i == 1 {
                    "Ferridian Shadow Cascade 1"
                } else {
                    "Ferridian Shadow Cascade N"
                },
            });
        }
        passes.push(ScheduledPass {
            kind: RenderPassKind::GBufferFill,
            label: "Ferridian G-Buffer Fill Pass",
        });
        passes.push(ScheduledPass {
            kind: RenderPassKind::DeferredLighting,
            label: "Ferridian Deferred Lighting Pass",
        });
        passes.push(ScheduledPass {
            kind: RenderPassKind::Translucent,
            label: "Ferridian Translucent Pass",
        });
        Self { passes }
    }

    pub fn passes(&self) -> &[ScheduledPass] {
        &self.passes
    }
}

// ---------------------------------------------------------------------------
// Shadow mapping — cascade configuration and shadow atlas types.
// ---------------------------------------------------------------------------

/// Configuration for cascaded shadow maps.
#[derive(Clone, Debug)]
pub struct ShadowCascadeConfig {
    pub cascade_count: u32,
    pub resolution: u32,
    pub depth_format: wgpu::TextureFormat,
    pub cascade_splits: Vec<f32>,
    pub depth_bias: f32,
    pub normal_bias: f32,
}

impl Default for ShadowCascadeConfig {
    fn default() -> Self {
        Self {
            cascade_count: 3,
            resolution: 2048,
            depth_format: wgpu::TextureFormat::Depth32Float,
            cascade_splits: vec![0.05, 0.15, 0.5, 1.0],
            depth_bias: 1.25,
            normal_bias: 0.75,
        }
    }
}

impl ShadowCascadeConfig {
    pub fn high_quality() -> Self {
        Self {
            cascade_count: 4,
            resolution: 4096,
            cascade_splits: vec![0.03, 0.08, 0.25, 0.6, 1.0],
            ..Self::default()
        }
    }

    pub fn low_quality() -> Self {
        Self {
            cascade_count: 2,
            resolution: 1024,
            cascade_splits: vec![0.1, 0.5, 1.0],
            ..Self::default()
        }
    }
}

/// Per-cascade shadow data uploaded as a uniform for the shadow pass.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowCascadeUniform {
    pub light_view_projection: [[f32; 4]; 4],
    pub split_near: f32,
    pub split_far: f32,
    pub texel_size: f32,
    pub _padding: f32,
}

/// Describes the kind of shadow-filtering algorithm to use.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShadowFilterMode {
    Hard,
    #[default]
    Pcf4,
    Pcf16,
    Pcss,
}

// ---------------------------------------------------------------------------
// Post-processing stack — ordered list of screen-space effects.
// ---------------------------------------------------------------------------

/// An individual post-processing effect and its configuration.
#[derive(Clone, Debug)]
pub enum PostProcessEffect {
    Ssao {
        radius: f32,
        sample_count: u32,
    },
    Ssr {
        max_steps: u32,
        thickness: f32,
    },
    VolumetricLight {
        sample_count: u32,
        density: f32,
    },
    Bloom {
        threshold: f32,
        intensity: f32,
        iterations: u32,
    },
    TemporalAntialiasing {
        feedback_factor: f32,
    },
    Tonemapping {
        operator: TonemapOperator,
    },
}

/// Tonemap operators supported by the post-processing pipeline.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TonemapOperator {
    #[default]
    Aces,
    Reinhard,
    AgX,
    Uncharted2,
}

/// An ordered stack of post-processing effects applied after deferred lighting.
#[derive(Clone, Debug)]
pub struct PostProcessStack {
    pub effects: Vec<PostProcessEffect>,
    pub enabled: bool,
}

impl PostProcessStack {
    pub fn minimal() -> Self {
        Self {
            effects: vec![
                PostProcessEffect::TemporalAntialiasing {
                    feedback_factor: 0.9,
                },
                PostProcessEffect::Tonemapping {
                    operator: TonemapOperator::Aces,
                },
            ],
            enabled: true,
        }
    }

    pub fn full() -> Self {
        Self {
            effects: vec![
                PostProcessEffect::Ssao {
                    radius: 0.5,
                    sample_count: 16,
                },
                PostProcessEffect::Ssr {
                    max_steps: 64,
                    thickness: 0.1,
                },
                PostProcessEffect::VolumetricLight {
                    sample_count: 32,
                    density: 0.02,
                },
                PostProcessEffect::Bloom {
                    threshold: 1.0,
                    intensity: 0.2,
                    iterations: 6,
                },
                PostProcessEffect::TemporalAntialiasing {
                    feedback_factor: 0.9,
                },
                PostProcessEffect::Tonemapping {
                    operator: TonemapOperator::Aces,
                },
            ],
            enabled: true,
        }
    }

    pub fn disabled() -> Self {
        Self {
            effects: Vec::new(),
            enabled: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Dense block-material metadata encoding — compact per-block data for GPU
// upload. Each block packs material index, light level, and flags into 32 bits.
// ---------------------------------------------------------------------------

/// Dense per-block metadata packed into a single u32 for GPU upload.
///
/// Layout: `[material_id: 12 bits][block_light: 4 bits][sky_light: 4 bits]
///          [flags: 8 bits][ao: 4 bits]`
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackedBlockMeta(pub u32);

impl PackedBlockMeta {
    pub fn new(
        material_id: u16,
        block_light: u8,
        sky_light: u8,
        flags: u8,
        ambient_occlusion: u8,
    ) -> Self {
        let mat = (material_id as u32) & 0xFFF;
        let bl = ((block_light & 0xF) as u32) << 12;
        let sl = ((sky_light & 0xF) as u32) << 16;
        let f = ((flags) as u32) << 20;
        let ao = ((ambient_occlusion & 0xF) as u32) << 28;
        Self(mat | bl | sl | f | ao)
    }

    pub fn material_id(self) -> u16 {
        (self.0 & 0xFFF) as u16
    }

    pub fn block_light(self) -> u8 {
        ((self.0 >> 12) & 0xF) as u8
    }

    pub fn sky_light(self) -> u8 {
        ((self.0 >> 16) & 0xF) as u8
    }

    pub fn flags(self) -> u8 {
        ((self.0 >> 20) & 0xFF) as u8
    }

    pub fn ambient_occlusion(self) -> u8 {
        ((self.0 >> 28) & 0xF) as u8
    }
}

// ---------------------------------------------------------------------------
// Texture array material loading — configuration for texture atlas / array
// binding used by the material system.
// ---------------------------------------------------------------------------

/// Configuration for binding texture arrays used by the material system.
#[derive(Clone, Debug)]
pub struct TextureArrayConfig {
    pub max_layers: u32,
    pub layer_width: u32,
    pub layer_height: u32,
    pub format: wgpu::TextureFormat,
    pub generate_mipmaps: bool,
}

impl Default for TextureArrayConfig {
    fn default() -> Self {
        Self {
            max_layers: 256,
            layer_width: 16,
            layer_height: 16,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            generate_mipmaps: true,
        }
    }
}

impl TextureArrayConfig {
    /// Mip levels for the configured layer dimensions.
    pub fn mip_levels(&self) -> u32 {
        if !self.generate_mipmaps {
            return 1;
        }
        (self.layer_width.max(self.layer_height) as f32)
            .log2()
            .floor() as u32
            + 1
    }
}

// ---------------------------------------------------------------------------
// Performance measurement — lightweight frame timing helpers.
// ---------------------------------------------------------------------------

/// Lightweight frame-timing accumulator for basic performance tracking.
#[derive(Clone, Debug)]
pub struct FrameTimings {
    samples: Vec<f32>,
    capacity: usize,
    cursor: usize,
    filled: bool,
}

impl FrameTimings {
    pub fn new(capacity: usize) -> Self {
        Self {
            samples: vec![0.0; capacity.max(1)],
            capacity: capacity.max(1),
            cursor: 0,
            filled: false,
        }
    }

    pub fn push(&mut self, frame_time_ms: f32) {
        self.samples[self.cursor] = frame_time_ms;
        self.cursor += 1;
        if self.cursor >= self.capacity {
            self.cursor = 0;
            self.filled = true;
        }
    }

    pub fn average_ms(&self) -> f32 {
        let count = if self.filled {
            self.capacity
        } else {
            self.cursor
        };
        if count == 0 {
            return 0.0;
        }
        self.samples[..count].iter().sum::<f32>() / count as f32
    }

    pub fn fps(&self) -> f32 {
        let avg = self.average_ms();
        if avg > 0.0 { 1000.0 / avg } else { 0.0 }
    }

    pub fn worst_ms(&self) -> f32 {
        let count = if self.filled {
            self.capacity
        } else {
            self.cursor
        };
        self.samples[..count]
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
    }
}

// ---------------------------------------------------------------------------
// GPU-resident chunk buffers — configuration for indirect draw and compute
// visibility culling on the GPU side.
// ---------------------------------------------------------------------------

/// Configuration for GPU-resident chunk buffer management.
#[derive(Clone, Debug)]
pub struct GpuChunkBufferConfig {
    pub max_chunks: u32,
    pub vertices_per_chunk: u32,
    pub indices_per_chunk: u32,
    pub use_indirect_draw: bool,
}

impl Default for GpuChunkBufferConfig {
    fn default() -> Self {
        Self {
            max_chunks: 1024,
            vertices_per_chunk: 65536,
            indices_per_chunk: 98304,
            use_indirect_draw: true,
        }
    }
}

/// An indirect draw command matching `wgpu::DrawIndexedIndirect` layout.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct IndirectDrawCommand {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

/// Describes a chunk region slot in the GPU-resident buffer. The compute
/// visibility pass reads these to decide which indirect draw commands to emit.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ChunkDrawSlot {
    pub chunk_x: i32,
    pub chunk_y: i32,
    pub chunk_z: i32,
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub _padding: [u32; 2],
}

/// Configuration for compute-driven visibility culling.
#[derive(Clone, Debug)]
pub struct VisibilityCullConfig {
    pub frustum_cull: bool,
    pub occlusion_cull: bool,
    pub workgroup_size: u32,
}

impl Default for VisibilityCullConfig {
    fn default() -> Self {
        Self {
            frustum_cull: true,
            occlusion_cull: false,
            workgroup_size: 64,
        }
    }
}

// ---------------------------------------------------------------------------
// Screen-space effect configurations — individual effect parameters that the
// post-processing stack references.
// ---------------------------------------------------------------------------

/// SSAO configuration matching XeGTAO-class parameters.
#[derive(Clone, Debug)]
pub struct SsaoConfig {
    pub radius: f32,
    pub bias: f32,
    pub intensity: f32,
    pub sample_count: u32,
    pub half_resolution: bool,
}

impl Default for SsaoConfig {
    fn default() -> Self {
        Self {
            radius: 0.5,
            bias: 0.025,
            intensity: 1.5,
            sample_count: 16,
            half_resolution: false,
        }
    }
}

/// Screen-space reflections configuration.
#[derive(Clone, Debug)]
pub struct SsrConfig {
    pub max_ray_steps: u32,
    pub ray_step_size: f32,
    pub thickness: f32,
    pub hi_z_enabled: bool,
    pub max_roughness: f32,
}

impl Default for SsrConfig {
    fn default() -> Self {
        Self {
            max_ray_steps: 64,
            ray_step_size: 0.1,
            thickness: 0.05,
            hi_z_enabled: true,
            max_roughness: 0.6,
        }
    }
}

/// Froxel-based volumetric lighting configuration.
#[derive(Clone, Debug)]
pub struct VolumetricConfig {
    pub froxel_grid: [u32; 3],
    pub max_distance: f32,
    pub scatter_intensity: f32,
    pub density: f32,
    pub temporal_reprojection: bool,
}

impl Default for VolumetricConfig {
    fn default() -> Self {
        Self {
            froxel_grid: [160, 90, 64],
            max_distance: 128.0,
            scatter_intensity: 1.0,
            density: 0.02,
            temporal_reprojection: true,
        }
    }
}

/// Bloom configuration.
#[derive(Clone, Debug)]
pub struct BloomConfig {
    pub threshold: f32,
    pub soft_threshold: f32,
    pub intensity: f32,
    pub iterations: u32,
}

impl Default for BloomConfig {
    fn default() -> Self {
        Self {
            threshold: 1.0,
            soft_threshold: 0.5,
            intensity: 0.2,
            iterations: 6,
        }
    }
}

/// Temporal anti-aliasing configuration.
#[derive(Clone, Debug)]
pub struct TaaConfig {
    pub feedback_factor: f32,
    pub jitter_scale: f32,
    pub motion_vector_scale: f32,
    pub clamp_method: TaaClampMethod,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TaaClampMethod {
    #[default]
    VarianceClip,
    MinMax,
    None,
}

impl Default for TaaConfig {
    fn default() -> Self {
        Self {
            feedback_factor: 0.9,
            jitter_scale: 1.0,
            motion_vector_scale: 1.0,
            clamp_method: TaaClampMethod::VarianceClip,
        }
    }
}

/// Color grading and tonemapping configuration.
#[derive(Clone, Debug)]
pub struct ColorGradingConfig {
    pub tonemap_operator: TonemapOperator,
    pub exposure: f32,
    pub saturation: f32,
    pub contrast: f32,
    pub gamma: f32,
}

impl Default for ColorGradingConfig {
    fn default() -> Self {
        Self {
            tonemap_operator: TonemapOperator::Aces,
            exposure: 1.0,
            saturation: 1.0,
            contrast: 1.0,
            gamma: 2.2,
        }
    }
}

/// Water shading configuration for the translucent forward pass.
#[derive(Clone, Debug)]
pub struct WaterShadingConfig {
    pub absorption_color: [f32; 3],
    pub absorption_density: f32,
    pub wave_amplitude: f32,
    pub wave_frequency: f32,
    pub refraction_strength: f32,
    pub caustics_enabled: bool,
}

impl Default for WaterShadingConfig {
    fn default() -> Self {
        Self {
            absorption_color: [0.02, 0.08, 0.12],
            absorption_density: 0.5,
            wave_amplitude: 0.03,
            wave_frequency: 2.0,
            refraction_strength: 0.15,
            caustics_enabled: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Voxel GI types — scaffolding for voxel cone tracing and path tracing modes.
// ---------------------------------------------------------------------------

/// Configuration for voxel cone tracing GI.
#[derive(Clone, Debug)]
pub struct VoxelGiConfig {
    pub grid_resolution: u32,
    pub cascade_count: u32,
    pub cone_count: u32,
    pub cone_angle_degrees: f32,
    pub max_trace_distance: f32,
    pub temporal_filtering: bool,
}

impl Default for VoxelGiConfig {
    fn default() -> Self {
        Self {
            grid_resolution: 256,
            cascade_count: 3,
            cone_count: 6,
            cone_angle_degrees: 30.0,
            max_trace_distance: 64.0,
            temporal_filtering: true,
        }
    }
}

/// High-end rendering mode selection.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GiMode {
    #[default]
    None,
    VoxelConeTracing,
    PathTracing,
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
    uniform_bind_group_layout: wgpu::BindGroupLayout,
    uniform_bind_group: wgpu::BindGroup,
    mesh: Mesh,
    depth_target: DepthTarget,
    frame_uniform: FrameUniform,
    pass_scheduler: PassScheduler,
    #[allow(dead_code)]
    gbuffer_layout: GBufferLayout,
    #[allow(dead_code)]
    _capabilities: BackendCapabilities,
}

impl SurfaceRenderer {
    pub async fn new(
        target: impl Into<wgpu::SurfaceTarget<'static>>,
        width: u32,
        height: u32,
        clear_color: wgpu::Color,
    ) -> Result<Self> {
        let backend_config = BackendConfig::default();
        let instance = wgpu::Instance::default();
        let surface = instance
            .create_surface(target)
            .context("failed to create wgpu surface")?;
        let adapter = instance
            .request_adapter(&backend_config.adapter_options(&surface))
            .await
            .context("failed to find a compatible GPU adapter")?;
        let backend_capabilities = BackendCapabilities::from_adapter(&adapter);
        let (device, queue) = backend_config.request_device(&adapter).await?;

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

        let mesh = Mesh::terrain_chunk_demo();
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
        let render_pipeline = create_render_pipeline(
            &device,
            config.format,
            &uniform_bind_group_layout,
            &shader_asset,
        );

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
        let pass_scheduler = PassScheduler::standalone_voxel();
        let gbuffer_layout = GBufferLayout::default();

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
            uniform_bind_group_layout,
            uniform_bind_group,
            mesh,
            depth_target,
            frame_uniform,
            pass_scheduler,
            gbuffer_layout,
            _capabilities: backend_capabilities,
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

    #[profiling::function]
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

        if matches!(self.shader_asset.reload_if_changed(), Ok(true)) {
            self.render_pipeline = create_render_pipeline(
                &self.device,
                self.config.format,
                &self.uniform_bind_group_layout,
                &self.shader_asset,
            );
        }

        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("Ferridian Clear Encoder"),
            });

        for scheduled_pass in self.pass_scheduler.passes() {
            match scheduled_pass.kind {
                RenderPassKind::OpaqueTerrain => {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some(scheduled_pass.label),
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
                RenderPassKind::GBufferFill
                | RenderPassKind::DeferredLighting
                | RenderPassKind::Translucent
                | RenderPassKind::ShadowMap => {
                    // Deferred passes are registered in the scheduler but
                    // execute as no-ops until the deferred pipeline shaders
                    // and render targets are wired.
                }
            }
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
            pass_count: self.pass_scheduler.passes().len(),
        }
    }
}

fn aspect_ratio(width: u32, height: u32) -> f32 {
    width.max(1) as f32 / height.max(1) as f32
}

fn create_render_pipeline(
    device: &wgpu::Device,
    surface_format: wgpu::TextureFormat,
    uniform_bind_group_layout: &wgpu::BindGroupLayout,
    shader_asset: &ShaderAsset,
) -> wgpu::RenderPipeline {
    let shader = shader_asset.create_module(device);
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Ferridian Voxel Pipeline Layout"),
        bind_group_layouts: &[uniform_bind_group_layout],
        push_constant_ranges: &[],
    });

    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
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
                format: surface_format,
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
    })
}

fn append_face(
    vertices: &mut Vec<MeshVertex>,
    indices: &mut Vec<u32>,
    offset: [u32; 3],
    color: [f32; 3],
    face: FaceDefinition,
) {
    let base_index = vertices.len() as u32;
    for corner in face.corners {
        vertices.push(MeshVertex::new(
            [
                corner[0] + offset[0],
                corner[1] + offset[1],
                corner[2] + offset[2],
            ],
            face.normal_index,
            color,
        ));
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

fn pack_position(position: [u32; 3]) -> u32 {
    debug_assert!(position[0] < 32);
    debug_assert!(position[1] < 32);
    debug_assert!(position[2] < 32);

    position[0] | (position[1] << 5) | (position[2] << 10)
}

fn pack_color(color: [f32; 3]) -> u32 {
    let r = (color[0].clamp(0.0, 1.0) * 255.0).round() as u32;
    let g = (color[1].clamp(0.0, 1.0) * 255.0).round() as u32;
    let b = (color[2].clamp(0.0, 1.0) * 255.0).round() as u32;
    r | (g << 8) | (b << 16) | (255 << 24)
}

#[cfg(test)]
mod tests {
    use super::{
        BackendConfig, BlockKind, BloomConfig, Camera, ChunkSection, ColorGradingConfig,
        FrameTimings, GBufferLayout, GiMode, GpuChunkBufferConfig, LabPbrExtension,
        MaterialDefinition, Mesh, PackedBlockMeta, PassScheduler, PbrProperties, PostProcessStack,
        RenderPassKind, ShadowCascadeConfig, ShadowFilterMode, SharedRendererConfig, SsaoConfig,
        SsrConfig, TaaConfig, TextureArrayConfig, TonemapOperator, VisibilityCullConfig,
        VolumetricConfig, VoxelGiConfig, WaterShadingConfig,
    };
    use std::mem::size_of;

    #[test]
    fn terrain_chunk_mesh_contains_indexed_geometry() {
        let mesh = Mesh::terrain_chunk_demo();
        assert!(!mesh.vertices.is_empty());
        assert!(!mesh.indices.is_empty());
        assert_eq!(mesh.indices.len() % 3, 0);
    }

    #[test]
    fn chunk_mesher_culls_hidden_interior_faces() {
        let mut section = ChunkSection::new(2, 1, 1);
        section.set_block(0, 0, 0, BlockKind::Stone);
        section.set_block(1, 0, 0, BlockKind::Stone);

        let mesh = Mesh::from_chunk_section(&section);

        assert_eq!(mesh.vertices.len(), 40);
        assert_eq!(mesh.indices.len(), 60);
    }

    #[test]
    fn terrain_vertices_use_compact_stride() {
        assert_eq!(size_of::<super::MeshVertex>(), 12);
    }

    #[test]
    fn orbit_camera_produces_finite_eye_position() {
        let camera = Camera::orbiting(1.25);
        let eye = camera.eye();
        assert!(eye.is_finite());
    }

    #[test]
    fn standalone_scheduler_starts_with_opaque_terrain_pass() {
        let scheduler = PassScheduler::standalone_voxel();

        assert_eq!(scheduler.passes().len(), 1);
        assert_eq!(scheduler.passes()[0].kind, RenderPassKind::OpaqueTerrain);
    }

    #[test]
    fn deferred_scheduler_has_gbuffer_lighting_translucent() {
        let scheduler = PassScheduler::deferred();

        assert_eq!(scheduler.passes().len(), 3);
        assert_eq!(scheduler.passes()[0].kind, RenderPassKind::GBufferFill);
        assert_eq!(scheduler.passes()[1].kind, RenderPassKind::DeferredLighting);
        assert_eq!(scheduler.passes()[2].kind, RenderPassKind::Translucent);
    }

    #[test]
    fn parallel_mesher_produces_same_geometry_as_sequential() {
        let section = ChunkSection::sample_terrain();
        let sequential = Mesh::from_chunk_section(&section);
        let parallel = Mesh::from_chunk_section_parallel(&section);

        assert_eq!(sequential.vertices.len(), parallel.vertices.len());
        assert_eq!(sequential.indices.len(), parallel.indices.len());
    }

    #[test]
    fn backend_config_defaults_to_stable_features() {
        let config = BackendConfig::default();
        assert!(!config.experimental);
        assert_eq!(config.required_features, wgpu::Features::empty());
    }

    #[test]
    fn gbuffer_layout_has_sensible_defaults() {
        let layout = GBufferLayout::default();
        assert_eq!(layout.albedo_format, wgpu::TextureFormat::Rgba8UnormSrgb);
        assert_eq!(layout.normal_format, wgpu::TextureFormat::Rgba16Float);
        assert_eq!(layout.depth_format, wgpu::TextureFormat::Depth24Plus);
    }

    #[test]
    fn material_definition_opaque_creates_non_translucent() {
        let mat = MaterialDefinition::opaque("stone", [0.5, 0.5, 0.5]);
        assert!(!mat.is_translucent());
        assert!(!mat.is_emissive());
    }

    #[test]
    fn material_definition_emissive_has_lab_pbr() {
        let mat = MaterialDefinition::emissive("glowstone", [1.0, 0.9, 0.4], 2.0);
        assert!(mat.is_emissive());
        assert!(mat.lab_pbr.is_some());
        assert_eq!(mat.lab_pbr.unwrap().emission, 2.0);
    }

    #[test]
    fn pbr_properties_defaults_are_dielectric() {
        let pbr = PbrProperties::default();
        assert_eq!(pbr.metallic, 0.0);
        assert!((pbr.roughness - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn lab_pbr_defaults_are_zero() {
        let ext = LabPbrExtension::default();
        assert_eq!(ext.porosity, 0.0);
        assert_eq!(ext.subsurface_scattering, 0.0);
        assert_eq!(ext.emission, 0.0);
    }

    #[test]
    fn shared_renderer_config_can_be_constructed() {
        let _config = SharedRendererConfig {
            width: 1280,
            height: 720,
            clear_color: wgpu::Color::BLACK,
            backend: BackendConfig::default(),
        };
    }

    #[test]
    fn shadow_cascade_config_defaults() {
        let config = ShadowCascadeConfig::default();
        assert_eq!(config.cascade_count, 3);
        assert_eq!(config.resolution, 2048);
        assert_eq!(config.cascade_splits.len(), 4);
    }

    #[test]
    fn shadow_cascade_quality_presets() {
        let high = ShadowCascadeConfig::high_quality();
        assert_eq!(high.cascade_count, 4);
        assert_eq!(high.resolution, 4096);

        let low = ShadowCascadeConfig::low_quality();
        assert_eq!(low.cascade_count, 2);
        assert_eq!(low.resolution, 1024);
    }

    #[test]
    fn shadow_filter_defaults_to_pcf4() {
        assert_eq!(ShadowFilterMode::default(), ShadowFilterMode::Pcf4);
    }

    #[test]
    fn deferred_with_shadows_scheduler() {
        let scheduler = PassScheduler::deferred_with_shadows(3);
        assert_eq!(scheduler.passes().len(), 6);
        assert_eq!(scheduler.passes()[0].kind, RenderPassKind::ShadowMap);
        assert_eq!(scheduler.passes()[1].kind, RenderPassKind::ShadowMap);
        assert_eq!(scheduler.passes()[2].kind, RenderPassKind::ShadowMap);
        assert_eq!(scheduler.passes()[3].kind, RenderPassKind::GBufferFill);
    }

    #[test]
    fn post_process_stack_minimal() {
        let stack = PostProcessStack::minimal();
        assert!(stack.enabled);
        assert_eq!(stack.effects.len(), 2);
    }

    #[test]
    fn post_process_stack_full() {
        let stack = PostProcessStack::full();
        assert!(stack.enabled);
        assert_eq!(stack.effects.len(), 6);
    }

    #[test]
    fn tonemap_operator_defaults_to_aces() {
        assert_eq!(TonemapOperator::default(), TonemapOperator::Aces);
    }

    #[test]
    fn packed_block_meta_roundtrips() {
        let meta = PackedBlockMeta::new(2047, 14, 11, 0xFF, 9);
        assert_eq!(meta.material_id(), 2047);
        assert_eq!(meta.block_light(), 14);
        assert_eq!(meta.sky_light(), 11);
        assert_eq!(meta.flags(), 0xFF);
        assert_eq!(meta.ambient_occlusion(), 9);
    }

    #[test]
    fn texture_array_config_defaults() {
        let config = TextureArrayConfig::default();
        assert_eq!(config.max_layers, 256);
        assert_eq!(config.layer_width, 16);
        assert!(config.generate_mipmaps);
    }

    #[test]
    fn texture_array_mip_levels() {
        let config = TextureArrayConfig::default();
        assert_eq!(config.mip_levels(), 5); // log2(16) + 1

        let no_mips = TextureArrayConfig {
            generate_mipmaps: false,
            ..TextureArrayConfig::default()
        };
        assert_eq!(no_mips.mip_levels(), 1);
    }

    #[test]
    fn frame_timings_average_and_fps() {
        let mut timings = FrameTimings::new(4);
        timings.push(16.0);
        timings.push(16.0);
        timings.push(16.0);
        timings.push(16.0);
        assert!((timings.average_ms() - 16.0).abs() < f32::EPSILON);
        assert!((timings.fps() - 62.5).abs() < 0.1);
    }

    #[test]
    fn frame_timings_worst_ms() {
        let mut timings = FrameTimings::new(4);
        timings.push(10.0);
        timings.push(20.0);
        timings.push(15.0);
        assert!((timings.worst_ms() - 20.0).abs() < f32::EPSILON);
    }

    #[test]
    fn gpu_chunk_buffer_config_defaults() {
        let config = GpuChunkBufferConfig::default();
        assert_eq!(config.max_chunks, 1024);
        assert!(config.use_indirect_draw);
    }

    #[test]
    fn visibility_cull_config_defaults() {
        let config = VisibilityCullConfig::default();
        assert!(config.frustum_cull);
        assert!(!config.occlusion_cull);
    }

    #[test]
    fn ssao_config_defaults() {
        let config = SsaoConfig::default();
        assert_eq!(config.sample_count, 16);
        assert!(!config.half_resolution);
    }

    #[test]
    fn ssr_config_defaults() {
        let config = SsrConfig::default();
        assert_eq!(config.max_ray_steps, 64);
        assert!(config.hi_z_enabled);
    }

    #[test]
    fn volumetric_config_defaults() {
        let config = VolumetricConfig::default();
        assert_eq!(config.froxel_grid, [160, 90, 64]);
        assert!(config.temporal_reprojection);
    }

    #[test]
    fn bloom_config_defaults() {
        let config = BloomConfig::default();
        assert_eq!(config.iterations, 6);
    }

    #[test]
    fn taa_config_defaults() {
        let config = TaaConfig::default();
        assert!((config.feedback_factor - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn color_grading_config_defaults() {
        let config = ColorGradingConfig::default();
        assert_eq!(config.tonemap_operator, TonemapOperator::Aces);
        assert!((config.gamma - 2.2).abs() < f32::EPSILON);
    }

    #[test]
    fn water_shading_config_defaults() {
        let config = WaterShadingConfig::default();
        assert!(config.caustics_enabled);
    }

    #[test]
    fn voxel_gi_config_defaults() {
        let config = VoxelGiConfig::default();
        assert_eq!(config.grid_resolution, 256);
        assert!(config.temporal_filtering);
    }

    #[test]
    fn gi_mode_defaults_to_none() {
        assert_eq!(GiMode::default(), GiMode::None);
    }

    // -----------------------------------------------------------------------
    // Camera tests
    // -----------------------------------------------------------------------

    #[test]
    fn camera_default_matches_orbiting_zero() {
        let default_cam = Camera::default();
        let orbit_cam = Camera::orbiting(0.0);
        assert_eq!(default_cam.target, orbit_cam.target);
        assert_eq!(default_cam.distance, orbit_cam.distance);
        assert_eq!(default_cam.yaw_radians, orbit_cam.yaw_radians);
    }

    #[test]
    fn camera_view_projection_is_finite() {
        let camera = Camera::orbiting(2.5);
        let vp = camera.view_projection_matrix(16.0 / 9.0);
        for col in vp.to_cols_array() {
            assert!(col.is_finite(), "view-projection element is not finite");
        }
    }

    #[test]
    fn camera_orbiting_different_times_produce_different_eyes() {
        let a = Camera::orbiting(0.0).eye();
        let b = Camera::orbiting(1.5).eye();
        assert_ne!(a, b);
    }

    #[test]
    fn camera_pitch_is_clamped() {
        let mut cam = Camera::orbiting(0.0);
        cam.pitch_radians = 10.0; // exceeds clamp
        let eye = cam.eye();
        assert!(eye.is_finite());
    }

    // -----------------------------------------------------------------------
    // Material and Transform tests
    // -----------------------------------------------------------------------

    #[test]
    fn material_default_has_white_base_color() {
        let mat = super::Material::default();
        assert_eq!(mat.base_color_factor, glam::Vec3::ONE);
    }

    #[test]
    fn transform_default_is_identity() {
        let t = super::Transform::default();
        let m = t.matrix();
        let identity = glam::Mat4::IDENTITY;
        assert!((m - identity).abs_diff_eq(glam::Mat4::ZERO, 1e-6));
    }

    #[test]
    fn transform_translation_affects_matrix() {
        let t = super::Transform {
            translation: glam::Vec3::new(5.0, 3.0, 1.0),
            ..super::Transform::default()
        };
        let m = t.matrix();
        // The translation shows up in the last column of the 4x4 matrix
        let col3 = m.col(3);
        assert!((col3.x - 5.0).abs() < f32::EPSILON);
        assert!((col3.y - 3.0).abs() < f32::EPSILON);
        assert!((col3.z - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn transform_scale_affects_matrix() {
        let t = super::Transform {
            scale: glam::Vec3::new(2.0, 2.0, 2.0),
            ..super::Transform::default()
        };
        let m = t.matrix();
        assert!((m.col(0).x - 2.0).abs() < f32::EPSILON);
        assert!((m.col(1).y - 2.0).abs() < f32::EPSILON);
        assert!((m.col(2).z - 2.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // BlockKind tests
    // -----------------------------------------------------------------------

    #[test]
    fn block_kind_air_is_not_solid() {
        assert!(!super::BlockKind::Air.is_solid());
    }

    #[test]
    fn block_kind_all_non_air_are_solid() {
        assert!(super::BlockKind::Grass.is_solid());
        assert!(super::BlockKind::Dirt.is_solid());
        assert!(super::BlockKind::Stone.is_solid());
    }

    #[test]
    fn block_kind_air_color_is_black() {
        let c = super::BlockKind::Air.color(0);
        assert_eq!(c, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn block_kind_grass_color_varies_with_y() {
        let c0 = super::BlockKind::Grass.color(0);
        let c5 = super::BlockKind::Grass.color(5);
        assert!(c5[1] > c0[1], "grass should get greener with height");
    }

    // -----------------------------------------------------------------------
    // ChunkSection tests
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_section_dimensions_match_constructor() {
        let section = super::ChunkSection::new(8, 4, 6);
        assert_eq!(section.dimensions(), (8, 4, 6));
    }

    #[test]
    fn chunk_section_initially_all_air() {
        let section = super::ChunkSection::new(4, 4, 4);
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    assert_eq!(section.block(x, y, z), super::BlockKind::Air);
                }
            }
        }
    }

    #[test]
    fn chunk_section_set_block_roundtrip() {
        let mut section = super::ChunkSection::new(4, 4, 4);
        section.set_block(2, 3, 1, super::BlockKind::Stone);
        assert_eq!(section.block(2, 3, 1), super::BlockKind::Stone);
        assert_eq!(section.block(0, 0, 0), super::BlockKind::Air);
    }

    #[test]
    fn chunk_section_sample_terrain_has_solid_blocks() {
        let section = super::ChunkSection::sample_terrain();
        assert_eq!(section.dimensions(), (16, 16, 16));
        // The sample terrain should have at least some solid blocks
        let has_solid = (0..16).any(|x| section.block(x, 0, 0).is_solid());
        assert!(has_solid, "sample terrain should have solid blocks at y=0");
    }

    // -----------------------------------------------------------------------
    // Mesh tests
    // -----------------------------------------------------------------------

    #[test]
    fn mesh_from_single_block_has_six_faces() {
        let mut section = super::ChunkSection::new(1, 1, 1);
        section.set_block(0, 0, 0, super::BlockKind::Stone);
        let mesh = Mesh::from_chunk_section(&section);
        // Single isolated block: 6 faces * 4 verts = 24, 6 faces * 6 indices = 36
        assert_eq!(mesh.vertices.len(), 24);
        assert_eq!(mesh.indices.len(), 36);
    }

    #[test]
    fn mesh_from_empty_section_is_empty() {
        let section = super::ChunkSection::new(4, 4, 4);
        let mesh = Mesh::from_chunk_section(&section);
        assert!(mesh.vertices.is_empty());
        assert!(mesh.indices.is_empty());
    }

    #[test]
    fn mesh_terrain_chunk_demo_has_correct_transform() {
        let mesh = Mesh::terrain_chunk_demo();
        assert!((mesh.transform.translation.x - (-1.5)).abs() < f32::EPSILON);
        assert_eq!(mesh.transform.rotation, glam::Quat::IDENTITY);
    }

    #[test]
    fn mesh_indices_are_valid_vertex_references() {
        let mesh = Mesh::terrain_chunk_demo();
        let vcount = mesh.vertices.len() as u32;
        for &idx in &mesh.indices {
            assert!(idx < vcount, "index {idx} out of bounds (vcount={vcount})");
        }
    }

    // -----------------------------------------------------------------------
    // SceneUniform and FrameUniform tests
    // -----------------------------------------------------------------------

    #[test]
    fn scene_uniform_is_pod() {
        let mesh = Mesh::terrain_chunk_demo();
        let uniform = super::SceneUniform::new(1920, 1080, &mesh, 0.0);
        let bytes: &[u8] = bytemuck::bytes_of(&uniform);
        assert_eq!(bytes.len(), size_of::<super::SceneUniform>());
    }

    #[test]
    fn frame_uniform_is_pod() {
        let uniform = super::FrameUniform::new(1920, 1080);
        let bytes: &[u8] = bytemuck::bytes_of(&uniform);
        assert_eq!(bytes.len(), size_of::<super::FrameUniform>());
    }

    #[test]
    fn frame_uniform_aspect_ratio() {
        let uniform = super::FrameUniform::new(1920, 1080);
        let expected = 1920.0 / 1080.0;
        assert!((uniform.aspect_ratio - expected).abs() < 0.01);
    }

    // -----------------------------------------------------------------------
    // Packing helpers tests
    // -----------------------------------------------------------------------

    #[test]
    fn pack_position_roundtrip() {
        let packed = super::pack_position([7, 15, 20]);
        let x = packed & 0x1F;
        let y = (packed >> 5) & 0x1F;
        let z = (packed >> 10) & 0x1F;
        assert_eq!((x, y, z), (7, 15, 20));
    }

    #[test]
    fn pack_color_clamps_and_encodes_rgba() {
        let packed = super::pack_color([1.0, 0.0, 0.5]);
        let r = packed & 0xFF;
        let g = (packed >> 8) & 0xFF;
        let b = (packed >> 16) & 0xFF;
        let a = (packed >> 24) & 0xFF;
        assert_eq!(r, 255);
        assert_eq!(g, 0);
        assert_eq!(b, 128);
        assert_eq!(a, 255);
    }

    #[test]
    fn pack_color_clamps_out_of_range_values() {
        let packed = super::pack_color([2.0, -1.0, 0.5]);
        let r = packed & 0xFF;
        let g = (packed >> 8) & 0xFF;
        assert_eq!(r, 255); // clamped from 2.0
        assert_eq!(g, 0); // clamped from -1.0
    }

    // -----------------------------------------------------------------------
    // aspect_ratio helper
    // -----------------------------------------------------------------------

    #[test]
    fn aspect_ratio_normal() {
        assert!((super::aspect_ratio(1920, 1080) - (1920.0 / 1080.0)).abs() < 0.001);
    }

    #[test]
    fn aspect_ratio_zero_clamped_to_one() {
        // aspect_ratio(0, 0) => max(0,1) / max(0,1) = 1.0
        assert!((super::aspect_ratio(0, 0) - 1.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // PackedBlockMeta edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn packed_block_meta_zero_values() {
        let meta = PackedBlockMeta::new(0, 0, 0, 0, 0);
        assert_eq!(meta.material_id(), 0);
        assert_eq!(meta.block_light(), 0);
        assert_eq!(meta.sky_light(), 0);
        assert_eq!(meta.flags(), 0);
        assert_eq!(meta.ambient_occlusion(), 0);
    }

    #[test]
    fn packed_block_meta_max_material_id() {
        let meta = PackedBlockMeta::new(4095, 0, 0, 0, 0);
        assert_eq!(meta.material_id(), 4095);
    }

    #[test]
    fn packed_block_meta_max_light_levels() {
        let meta = PackedBlockMeta::new(0, 15, 15, 0, 0);
        assert_eq!(meta.block_light(), 15);
        assert_eq!(meta.sky_light(), 15);
    }

    // -----------------------------------------------------------------------
    // FrameTimings edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn frame_timings_empty_returns_zero() {
        let timings = FrameTimings::new(4);
        assert_eq!(timings.average_ms(), 0.0);
        assert_eq!(timings.fps(), 0.0);
        assert_eq!(timings.worst_ms(), 0.0);
    }

    #[test]
    fn frame_timings_wraps_around() {
        let mut timings = FrameTimings::new(2);
        timings.push(10.0);
        timings.push(20.0);
        timings.push(30.0); // overwrites first slot
        assert!(timings.filled);
        assert!((timings.average_ms() - 25.0).abs() < f32::EPSILON); // (30+20)/2
    }

    #[test]
    fn frame_timings_capacity_clamped_to_one() {
        let timings = FrameTimings::new(0);
        assert_eq!(timings.capacity, 1);
    }

    // -----------------------------------------------------------------------
    // TextureArrayConfig edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn texture_array_mip_levels_256x256() {
        let config = TextureArrayConfig {
            layer_width: 256,
            layer_height: 256,
            ..TextureArrayConfig::default()
        };
        assert_eq!(config.mip_levels(), 9); // log2(256) + 1
    }

    #[test]
    fn texture_array_mip_levels_asymmetric() {
        let config = TextureArrayConfig {
            layer_width: 32,
            layer_height: 64,
            ..TextureArrayConfig::default()
        };
        // max(32,64)=64, log2(64)+1 = 7
        assert_eq!(config.mip_levels(), 7);
    }

    // -----------------------------------------------------------------------
    // IndirectDrawCommand and ChunkDrawSlot Pod safety
    // -----------------------------------------------------------------------

    #[test]
    fn indirect_draw_command_is_pod() {
        let cmd = super::IndirectDrawCommand {
            index_count: 36,
            instance_count: 1,
            first_index: 0,
            base_vertex: 0,
            first_instance: 0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&cmd);
        assert_eq!(bytes.len(), size_of::<super::IndirectDrawCommand>());
    }

    #[test]
    fn chunk_draw_slot_is_pod() {
        let slot = super::ChunkDrawSlot {
            chunk_x: 1,
            chunk_y: 2,
            chunk_z: 3,
            index_count: 36,
            first_index: 0,
            base_vertex: 0,
            _padding: [0; 2],
        };
        let bytes: &[u8] = bytemuck::bytes_of(&slot);
        assert_eq!(bytes.len(), size_of::<super::ChunkDrawSlot>());
    }

    // -----------------------------------------------------------------------
    // ShadowCascadeUniform Pod safety
    // -----------------------------------------------------------------------

    #[test]
    fn shadow_cascade_uniform_is_pod() {
        let uniform = super::ShadowCascadeUniform {
            light_view_projection: [[0.0; 4]; 4],
            split_near: 0.1,
            split_far: 50.0,
            texel_size: 1.0 / 2048.0,
            _padding: 0.0,
        };
        let bytes: &[u8] = bytemuck::bytes_of(&uniform);
        assert_eq!(bytes.len(), size_of::<super::ShadowCascadeUniform>());
    }

    // -----------------------------------------------------------------------
    // PostProcessStack::disabled()
    // -----------------------------------------------------------------------

    #[test]
    fn post_process_stack_disabled() {
        let stack = PostProcessStack::disabled();
        assert!(!stack.enabled);
        assert!(stack.effects.is_empty());
    }

    // -----------------------------------------------------------------------
    // RenderBackendPlan defaults
    // -----------------------------------------------------------------------

    #[test]
    fn render_backend_plan_defaults() {
        let plan = super::RenderBackendPlan::default();
        assert_eq!(plan.primary_backend, "wgpu-vulkan");
        assert!(!plan.target_passes.is_empty());
    }

    // -----------------------------------------------------------------------
    // MeshVertex layout
    // -----------------------------------------------------------------------

    #[test]
    fn mesh_vertex_layout_has_three_attributes() {
        let layout = super::MeshVertex::layout();
        assert_eq!(layout.attributes.len(), 3);
        assert_eq!(layout.array_stride, size_of::<super::MeshVertex>() as u64);
    }

    // -----------------------------------------------------------------------
    // MaterialDefinition additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn material_definition_translucent() {
        let mat = MaterialDefinition::opaque("glass", [0.9, 0.9, 0.9]);
        assert!(!mat.is_translucent());
    }

    #[test]
    fn material_definition_lab_pbr_values() {
        let mat = MaterialDefinition::emissive("lava", [1.0, 0.3, 0.0], 5.0);
        let lab = mat.lab_pbr.unwrap();
        assert_eq!(lab.emission, 5.0);
        assert_eq!(lab.porosity, 0.0);
        assert_eq!(lab.subsurface_scattering, 0.0);
    }

    // -----------------------------------------------------------------------
    // PassScheduler label checks
    // -----------------------------------------------------------------------

    #[test]
    fn standalone_scheduler_pass_has_label() {
        let scheduler = PassScheduler::standalone_voxel();
        assert!(!scheduler.passes()[0].label.is_empty());
    }

    #[test]
    fn deferred_scheduler_labels_are_unique() {
        let scheduler = PassScheduler::deferred();
        let labels: Vec<&str> = scheduler.passes().iter().map(|p| p.label).collect();
        for (i, l) in labels.iter().enumerate() {
            for (j, r) in labels.iter().enumerate() {
                if i != j {
                    assert_ne!(l, r, "duplicate label");
                }
            }
        }
    }
}
