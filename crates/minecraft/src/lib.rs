use ferridian_core::RenderBackendPlan;
use ferridian_shader::ShaderPipelineConfig;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CameraInput {
    pub eye: [f32; 3],
    pub target: [f32; 3],
    pub fov_y_radians: f32,
}

impl Default for CameraInput {
    fn default() -> Self {
        Self {
            eye: [8.0, 10.0, 8.0],
            target: [0.0, 0.0, 0.0],
            fov_y_radians: 55.0_f32.to_radians(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkInput {
    pub coord: ChunkCoord,
    pub block_count: u32,
    pub visible: bool,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameInput {
    pub delta_seconds: f32,
    pub time_seconds: f32,
    pub camera: CameraInput,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MinecraftFramePacket {
    pub frame: FrameInput,
    pub visible_chunks: Vec<ChunkInput>,
}

pub struct MinecraftAdapter {
    pub backend: RenderBackendPlan,
    pub shaders: ShaderPipelineConfig,
}

impl MinecraftAdapter {
    pub fn planned() -> Self {
        Self {
            backend: RenderBackendPlan::default(),
            shaders: ShaderPipelineConfig::default(),
        }
    }

    pub fn prepare_frame(
        &self,
        frame: FrameInput,
        visible_chunks: Vec<ChunkInput>,
    ) -> MinecraftFramePacket {
        MinecraftFramePacket {
            frame,
            visible_chunks,
        }
    }
}

// ---------------------------------------------------------------------------
// Iris compatibility layer — maps Iris shader-pack pass stages onto Ferridian's
// internal render graph without leaking OpenGL assumptions into the core.
// ---------------------------------------------------------------------------

/// Classic Iris/OptiFine shader pack pass stages, kept as data so the core
/// renderer never sees OpenGL-specific names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrisPassStage {
    Shadow,
    GBuffersTerrain,
    GBuffersEntities,
    GBuffersBlock,
    GBuffersWater,
    GBuffersSkyBasic,
    GBuffersSkyTextured,
    GBuffersWeather,
    GBuffersHand,
    Deferred,
    Composite,
    Final,
}

impl IrisPassStage {
    /// Maps an Iris stage to the closest Ferridian render pass kind.
    pub fn to_ferridian_pass(self) -> ferridian_core::RenderPassKind {
        match self {
            Self::Shadow => ferridian_core::RenderPassKind::ShadowMap,
            Self::GBuffersTerrain
            | Self::GBuffersBlock
            | Self::GBuffersSkyBasic
            | Self::GBuffersSkyTextured => ferridian_core::RenderPassKind::GBufferFill,
            Self::GBuffersEntities | Self::GBuffersHand => {
                ferridian_core::RenderPassKind::GBufferFill
            }
            Self::GBuffersWater | Self::GBuffersWeather => {
                ferridian_core::RenderPassKind::Translucent
            }
            Self::Deferred => ferridian_core::RenderPassKind::DeferredLighting,
            Self::Composite | Self::Final => ferridian_core::RenderPassKind::DeferredLighting,
        }
    }

    /// All stages in the classic Iris pass order.
    pub fn all_stages() -> &'static [Self] {
        &[
            Self::Shadow,
            Self::GBuffersTerrain,
            Self::GBuffersEntities,
            Self::GBuffersBlock,
            Self::GBuffersWater,
            Self::GBuffersSkyBasic,
            Self::GBuffersSkyTextured,
            Self::GBuffersWeather,
            Self::GBuffersHand,
            Self::Deferred,
            Self::Composite,
            Self::Final,
        ]
    }
}

/// Describes how a shader pack interacts with the renderer, covering both
/// legacy Iris packs and future Aperture-style explicit pipelines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShaderPackModel {
    /// Classic Iris/OptiFine: the pack defines shader programs per pass stage.
    IrisLegacy,
    /// Future Aperture: the pack defines an explicit pipeline description.
    ApertureExplicit,
}

/// Adapter between a shader-pack's resource, camera, and timing inputs and
/// the internal Ferridian renderer. This keeps Iris-specific assumptions
/// contained and replaceable when Aperture stabilises.
#[derive(Debug, Clone)]
pub struct ShaderPackAdapter {
    pub model: ShaderPackModel,
    pub active_stages: Vec<IrisPassStage>,
}

impl ShaderPackAdapter {
    pub fn iris_default() -> Self {
        Self {
            model: ShaderPackModel::IrisLegacy,
            active_stages: IrisPassStage::all_stages().to_vec(),
        }
    }

    pub fn aperture_stub() -> Self {
        Self {
            model: ShaderPackModel::ApertureExplicit,
            active_stages: Vec::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// RenderPipelineProvider — abstraction so standalone, fabric-iris, and future
// aperture entry paths can drive the same renderer core.
// ---------------------------------------------------------------------------

/// The entry-point classification for how the renderer was bootstrapped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RenderEntryPath {
    Standalone,
    FabricIris,
    Aperture,
}

/// Configuration provided by whatever entry path bootstrapped the renderer.
#[derive(Debug, Clone)]
pub struct RenderPipelineProvider {
    pub entry_path: RenderEntryPath,
    pub shader_pack: Option<ShaderPackAdapter>,
    pub backend: RenderBackendPlan,
}

impl RenderPipelineProvider {
    pub fn standalone() -> Self {
        Self {
            entry_path: RenderEntryPath::Standalone,
            shader_pack: None,
            backend: RenderBackendPlan::default(),
        }
    }

    pub fn fabric_iris() -> Self {
        Self {
            entry_path: RenderEntryPath::FabricIris,
            shader_pack: Some(ShaderPackAdapter::iris_default()),
            backend: RenderBackendPlan::default(),
        }
    }

    pub fn aperture() -> Self {
        Self {
            entry_path: RenderEntryPath::Aperture,
            shader_pack: Some(ShaderPackAdapter::aperture_stub()),
            backend: RenderBackendPlan::default(),
        }
    }
}

// ---------------------------------------------------------------------------
// Fabric bootstrap path — connects the Fabric mod entry point to the shared
// renderer core without forking renderer ownership into Java.
// ---------------------------------------------------------------------------

/// State machine for the Fabric bootstrap sequence.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FabricBootstrapState {
    /// Native library not yet loaded.
    Uninitialized,
    /// Native library loaded, awaiting first world render.
    NativeLoaded,
    /// Renderer is active and processing frames.
    RendererActive,
    /// Error state — native library failed to load.
    Failed,
}

/// Tracks the Fabric bootstrap lifecycle and connects it to the pipeline
/// provider so the shared renderer core can be reached from Java.
#[derive(Debug, Clone)]
pub struct FabricBootstrapPath {
    pub state: FabricBootstrapState,
    pub pipeline: RenderPipelineProvider,
    pub dimensions: (u32, u32),
}

impl FabricBootstrapPath {
    pub fn new() -> Self {
        Self {
            state: FabricBootstrapState::Uninitialized,
            pipeline: RenderPipelineProvider::fabric_iris(),
            dimensions: (0, 0),
        }
    }

    pub fn on_native_loaded(&mut self) {
        if self.state == FabricBootstrapState::Uninitialized {
            self.state = FabricBootstrapState::NativeLoaded;
        }
    }

    pub fn on_native_load_failed(&mut self) {
        self.state = FabricBootstrapState::Failed;
    }

    pub fn on_renderer_init(&mut self, width: u32, height: u32) {
        if self.state == FabricBootstrapState::NativeLoaded {
            self.dimensions = (width, height);
            self.state = FabricBootstrapState::RendererActive;
        }
    }

    pub fn on_world_unload(&mut self) {
        if self.state == FabricBootstrapState::RendererActive {
            self.state = FabricBootstrapState::NativeLoaded;
        }
    }

    pub fn is_ready(&self) -> bool {
        self.state == FabricBootstrapState::RendererActive
    }
}

impl Default for FabricBootstrapPath {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// End-to-end Fabric→Iris→Rust prototype — the full path where Fabric boots
// the mod, Iris-facing hooks collect frame inputs, and Rust executes the
// renderer through the existing bridge API.
// ---------------------------------------------------------------------------

/// Represents a single frame flowing through the full Fabric→Iris→Rust path.
/// This is the prototype for the e2e integration: Java collects inputs,
/// serializes them, and Rust consumes them to drive the renderer.
#[derive(Debug, Clone)]
pub struct FabricIrisFrame {
    pub camera: CameraInput,
    pub active_pass: ferridian_core::RenderPassKind,
    pub iris_stages_executed: Vec<IrisPassStage>,
    pub delta_seconds: f32,
    pub time_seconds: f32,
}

impl FabricIrisFrame {
    /// Creates a frame from a Fabric lifecycle snapshot and an Iris pass assignment.
    pub fn from_iris_snapshot(
        camera: CameraInput,
        time: f32,
        delta: f32,
        stages: &[IrisPassStage],
    ) -> Self {
        let primary_pass = stages
            .first()
            .map(|s| s.to_ferridian_pass())
            .unwrap_or(ferridian_core::RenderPassKind::OpaqueTerrain);

        Self {
            camera,
            active_pass: primary_pass,
            iris_stages_executed: stages.to_vec(),
            delta_seconds: delta,
            time_seconds: time,
        }
    }

    /// Converts this frame into the FrameInput expected by the Minecraft adapter.
    pub fn to_frame_input(&self) -> FrameInput {
        FrameInput {
            delta_seconds: self.delta_seconds,
            time_seconds: self.time_seconds,
            camera: self.camera,
        }
    }
}

/// Drives the full e2e path: bootstrap → frame collection → renderer execution.
/// This function validates that all pieces connect end-to-end.
pub fn drive_fabric_iris_frame(
    bootstrap: &FabricBootstrapPath,
    frame: &FabricIrisFrame,
    chunks: Vec<ChunkInput>,
) -> Option<MinecraftFramePacket> {
    if !bootstrap.is_ready() {
        return None;
    }

    let adapter = MinecraftAdapter::planned();
    let packet = adapter.prepare_frame(frame.to_frame_input(), chunks);
    Some(packet)
}

// ---------------------------------------------------------------------------
// FRAPI-compatible model and material handling — types that allow Sodium-era
// Fabric mods (using the Fabric Rendering API) to remain compatible.
// ---------------------------------------------------------------------------

/// A block model quad in FRAPI's convention: four vertices with per-vertex
/// color, UV, normal, and a face direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrapiQuad {
    pub vertices: [[f32; 3]; 4],
    pub uvs: [[f32; 2]; 4],
    pub color: u32,
    pub face: FrapiFace,
    pub material_id: u16,
}

/// The six possible face directions matching FRAPI/Minecraft block faces.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FrapiFace {
    Down,
    Up,
    North,
    South,
    West,
    East,
}

/// FRAPI material render layer, controlling how geometry is sorted and blended.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum FrapiRenderLayer {
    #[default]
    Solid,
    Cutout,
    CutoutMipped,
    Translucent,
}

/// A FRAPI-compatible material definition that bridges between Fabric mod
/// material conventions and Ferridian's internal PBR material system.
#[derive(Debug, Clone)]
pub struct FrapiMaterial {
    pub name: String,
    pub render_layer: FrapiRenderLayer,
    pub disable_ao: bool,
    pub disable_diffuse: bool,
    pub emissive: bool,
}

impl FrapiMaterial {
    pub fn solid(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            render_layer: FrapiRenderLayer::Solid,
            disable_ao: false,
            disable_diffuse: false,
            emissive: false,
        }
    }

    pub fn translucent(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            render_layer: FrapiRenderLayer::Translucent,
            disable_ao: false,
            disable_diffuse: false,
            emissive: false,
        }
    }

    /// Convert to the internal Ferridian material system.
    pub fn to_ferridian_material(&self) -> ferridian_core::MaterialDefinition {
        if self.emissive {
            ferridian_core::MaterialDefinition::emissive(&self.name, [1.0, 1.0, 1.0], 1.0)
        } else {
            ferridian_core::MaterialDefinition::opaque(&self.name, [1.0, 1.0, 1.0])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CameraInput, ChunkCoord, ChunkInput, FabricBootstrapPath, FabricBootstrapState,
        FabricIrisFrame, FrameInput, FrapiMaterial, FrapiRenderLayer, IrisPassStage,
        MinecraftAdapter, RenderEntryPath, RenderPipelineProvider, ShaderPackModel,
    };

    #[test]
    fn adapter_prepares_frame_packets() {
        let adapter = MinecraftAdapter::planned();
        let packet = adapter.prepare_frame(
            FrameInput {
                delta_seconds: 1.0 / 60.0,
                time_seconds: 4.0,
                camera: CameraInput::default(),
            },
            vec![ChunkInput {
                coord: ChunkCoord { x: 0, y: 0, z: 0 },
                block_count: 4096,
                visible: true,
            }],
        );

        assert_eq!(packet.visible_chunks.len(), 1);
        assert!(packet.visible_chunks[0].visible);
    }

    #[test]
    fn iris_stages_map_to_ferridian_passes() {
        let gbuffer = IrisPassStage::GBuffersTerrain.to_ferridian_pass();
        assert_eq!(gbuffer, ferridian_core::RenderPassKind::GBufferFill);

        let translucent = IrisPassStage::GBuffersWater.to_ferridian_pass();
        assert_eq!(translucent, ferridian_core::RenderPassKind::Translucent);

        let deferred = IrisPassStage::Deferred.to_ferridian_pass();
        assert_eq!(deferred, ferridian_core::RenderPassKind::DeferredLighting);
    }

    #[test]
    fn all_iris_stages_are_representable() {
        for stage in IrisPassStage::all_stages() {
            let _pass = stage.to_ferridian_pass();
        }
    }

    #[test]
    fn render_pipeline_provider_has_three_entry_paths() {
        let standalone = RenderPipelineProvider::standalone();
        assert_eq!(standalone.entry_path, RenderEntryPath::Standalone);
        assert!(standalone.shader_pack.is_none());

        let iris = RenderPipelineProvider::fabric_iris();
        assert_eq!(iris.entry_path, RenderEntryPath::FabricIris);
        assert_eq!(
            iris.shader_pack.as_ref().unwrap().model,
            ShaderPackModel::IrisLegacy
        );

        let aperture = RenderPipelineProvider::aperture();
        assert_eq!(aperture.entry_path, RenderEntryPath::Aperture);
        assert_eq!(
            aperture.shader_pack.as_ref().unwrap().model,
            ShaderPackModel::ApertureExplicit
        );
    }

    #[test]
    fn frapi_solid_material_defaults() {
        let mat = FrapiMaterial::solid("stone");
        assert_eq!(mat.render_layer, FrapiRenderLayer::Solid);
        assert!(!mat.disable_ao);
        assert!(!mat.emissive);
    }

    #[test]
    fn frapi_material_converts_to_ferridian() {
        let mat = FrapiMaterial::solid("stone");
        let ferridian_mat = mat.to_ferridian_material();
        assert!(!ferridian_mat.is_emissive());

        let emissive = FrapiMaterial {
            emissive: true,
            ..FrapiMaterial::solid("glowstone")
        };
        let ferridian_emissive = emissive.to_ferridian_material();
        assert!(ferridian_emissive.is_emissive());
    }

    #[test]
    fn frapi_translucent_layer() {
        let mat = FrapiMaterial::translucent("water");
        assert_eq!(mat.render_layer, FrapiRenderLayer::Translucent);
    }

    #[test]
    fn fabric_bootstrap_lifecycle() {
        let mut path = FabricBootstrapPath::new();
        assert_eq!(path.state, FabricBootstrapState::Uninitialized);
        assert!(!path.is_ready());

        path.on_native_loaded();
        assert_eq!(path.state, FabricBootstrapState::NativeLoaded);
        assert!(!path.is_ready());

        path.on_renderer_init(1920, 1080);
        assert_eq!(path.state, FabricBootstrapState::RendererActive);
        assert!(path.is_ready());
        assert_eq!(path.dimensions, (1920, 1080));

        path.on_world_unload();
        assert_eq!(path.state, FabricBootstrapState::NativeLoaded);
        assert!(!path.is_ready());
    }

    #[test]
    fn fabric_bootstrap_handles_failed_load() {
        let mut path = FabricBootstrapPath::new();
        path.on_native_load_failed();
        assert_eq!(path.state, FabricBootstrapState::Failed);

        // Renderer init should be ignored in failed state.
        path.on_renderer_init(1920, 1080);
        assert_eq!(path.state, FabricBootstrapState::Failed);
    }

    #[test]
    fn e2e_fabric_iris_frame_drives_renderer() {
        // Simulate the full path: bootstrap → collect Iris frame → drive renderer.
        let mut bootstrap = FabricBootstrapPath::new();
        bootstrap.on_native_loaded();
        bootstrap.on_renderer_init(1920, 1080);
        assert!(bootstrap.is_ready());

        let stages = vec![
            IrisPassStage::Shadow,
            IrisPassStage::GBuffersTerrain,
            IrisPassStage::Deferred,
            IrisPassStage::Composite,
            IrisPassStage::Final,
        ];
        let frame =
            FabricIrisFrame::from_iris_snapshot(CameraInput::default(), 120.0, 1.0 / 60.0, &stages);

        assert_eq!(frame.active_pass, ferridian_core::RenderPassKind::ShadowMap);
        assert_eq!(frame.iris_stages_executed.len(), 5);

        let chunks = vec![ChunkInput {
            coord: ChunkCoord { x: 0, y: 4, z: 0 },
            block_count: 4096,
            visible: true,
        }];

        let packet = super::drive_fabric_iris_frame(&bootstrap, &frame, chunks);
        assert!(packet.is_some());
        assert_eq!(packet.unwrap().visible_chunks.len(), 1);
    }

    #[test]
    fn e2e_returns_none_when_bootstrap_not_ready() {
        let bootstrap = FabricBootstrapPath::new();
        let frame = FabricIrisFrame::from_iris_snapshot(
            CameraInput::default(),
            0.0,
            0.016,
            &[IrisPassStage::GBuffersTerrain],
        );
        let result = super::drive_fabric_iris_frame(&bootstrap, &frame, vec![]);
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // CameraInput tests
    // -----------------------------------------------------------------------

    #[test]
    fn camera_input_default_values() {
        let cam = CameraInput::default();
        assert_eq!(cam.eye, [8.0, 10.0, 8.0]);
        assert_eq!(cam.target, [0.0, 0.0, 0.0]);
        assert!(cam.fov_y_radians > 0.0);
        assert!(cam.fov_y_radians < std::f32::consts::PI);
    }

    // -----------------------------------------------------------------------
    // ChunkCoord tests
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_coord_equality() {
        let a = ChunkCoord { x: 1, y: 2, z: 3 };
        let b = ChunkCoord { x: 1, y: 2, z: 3 };
        let c = ChunkCoord { x: 4, y: 5, z: 6 };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn chunk_coord_negative_values() {
        let coord = ChunkCoord {
            x: -5,
            y: -10,
            z: -3,
        };
        assert_eq!(coord.x, -5);
        assert_eq!(coord.y, -10);
        assert_eq!(coord.z, -3);
    }

    // -----------------------------------------------------------------------
    // ChunkInput tests
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_input_construction() {
        let input = ChunkInput {
            coord: ChunkCoord { x: 0, y: 0, z: 0 },
            block_count: 0,
            visible: false,
        };
        assert_eq!(input.block_count, 0);
        assert!(!input.visible);
    }

    // -----------------------------------------------------------------------
    // FrameInput tests
    // -----------------------------------------------------------------------

    #[test]
    fn frame_input_with_custom_camera() {
        let frame = FrameInput {
            delta_seconds: 0.016,
            time_seconds: 100.0,
            camera: CameraInput {
                eye: [0.0, 0.0, 0.0],
                target: [1.0, 0.0, 0.0],
                fov_y_radians: 90.0_f32.to_radians(),
            },
        };
        assert!((frame.delta_seconds - 0.016).abs() < f32::EPSILON);
        assert_eq!(frame.camera.eye, [0.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // IrisPassStage comprehensive tests
    // -----------------------------------------------------------------------

    #[test]
    fn iris_shadow_maps_to_shadow_map() {
        assert_eq!(
            IrisPassStage::Shadow.to_ferridian_pass(),
            ferridian_core::RenderPassKind::ShadowMap
        );
    }

    #[test]
    fn iris_composite_and_final_map_to_deferred() {
        assert_eq!(
            IrisPassStage::Composite.to_ferridian_pass(),
            ferridian_core::RenderPassKind::DeferredLighting
        );
        assert_eq!(
            IrisPassStage::Final.to_ferridian_pass(),
            ferridian_core::RenderPassKind::DeferredLighting
        );
    }

    #[test]
    fn iris_all_stages_count() {
        assert_eq!(IrisPassStage::all_stages().len(), 12);
    }

    // -----------------------------------------------------------------------
    // ShaderPackAdapter tests
    // -----------------------------------------------------------------------

    #[test]
    fn shader_pack_adapter_iris_default_has_all_stages() {
        let adapter = super::ShaderPackAdapter::iris_default();
        assert_eq!(adapter.model, ShaderPackModel::IrisLegacy);
        assert_eq!(adapter.active_stages.len(), 12);
    }

    #[test]
    fn shader_pack_adapter_aperture_stub_has_no_stages() {
        let adapter = super::ShaderPackAdapter::aperture_stub();
        assert_eq!(adapter.model, ShaderPackModel::ApertureExplicit);
        assert!(adapter.active_stages.is_empty());
    }

    // -----------------------------------------------------------------------
    // FrapiFace tests
    // -----------------------------------------------------------------------

    #[test]
    fn frapi_face_all_six_variants_exist() {
        let faces = [
            super::FrapiFace::Down,
            super::FrapiFace::Up,
            super::FrapiFace::North,
            super::FrapiFace::South,
            super::FrapiFace::West,
            super::FrapiFace::East,
        ];
        assert_eq!(faces.len(), 6);
        // Each is distinct
        for (i, a) in faces.iter().enumerate() {
            for (j, b) in faces.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b);
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // FrapiRenderLayer tests
    // -----------------------------------------------------------------------

    #[test]
    fn frapi_render_layer_default_is_solid() {
        assert_eq!(FrapiRenderLayer::default(), FrapiRenderLayer::Solid);
    }

    #[test]
    fn frapi_render_layer_cutout_variants() {
        assert_ne!(FrapiRenderLayer::Cutout, FrapiRenderLayer::CutoutMipped);
        assert_ne!(FrapiRenderLayer::Cutout, FrapiRenderLayer::Solid);
    }

    // -----------------------------------------------------------------------
    // FrapiQuad test
    // -----------------------------------------------------------------------

    #[test]
    fn frapi_quad_construction() {
        let quad = super::FrapiQuad {
            vertices: [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            uvs: [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
            color: 0xFFFFFFFF,
            face: super::FrapiFace::North,
            material_id: 42,
        };
        assert_eq!(quad.face, super::FrapiFace::North);
        assert_eq!(quad.material_id, 42);
    }

    // -----------------------------------------------------------------------
    // FrapiMaterial additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn frapi_material_disable_ao() {
        let mat = super::FrapiMaterial {
            disable_ao: true,
            ..super::FrapiMaterial::solid("custom")
        };
        assert!(mat.disable_ao);
        assert!(!mat.disable_diffuse);
    }

    #[test]
    fn frapi_material_emissive_converts_correctly() {
        let mat = super::FrapiMaterial {
            emissive: true,
            ..super::FrapiMaterial::solid("torch")
        };
        let ferridian_mat = mat.to_ferridian_material();
        assert!(ferridian_mat.is_emissive());
    }

    // -----------------------------------------------------------------------
    // FabricIrisFrame additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn fabric_iris_frame_to_frame_input() {
        let frame = FabricIrisFrame::from_iris_snapshot(
            CameraInput::default(),
            42.0,
            0.016,
            &[IrisPassStage::GBuffersTerrain],
        );
        let input = frame.to_frame_input();
        assert!((input.time_seconds - 42.0).abs() < f32::EPSILON);
        assert!((input.delta_seconds - 0.016).abs() < f32::EPSILON);
        assert_eq!(input.camera.eye, CameraInput::default().eye);
    }

    #[test]
    fn fabric_iris_frame_empty_stages_defaults_to_opaque() {
        let frame = FabricIrisFrame::from_iris_snapshot(CameraInput::default(), 0.0, 0.016, &[]);
        assert_eq!(
            frame.active_pass,
            ferridian_core::RenderPassKind::OpaqueTerrain
        );
        assert!(frame.iris_stages_executed.is_empty());
    }

    // -----------------------------------------------------------------------
    // FabricBootstrapPath additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn fabric_bootstrap_default_matches_new() {
        let a = FabricBootstrapPath::new();
        let b = FabricBootstrapPath::default();
        assert_eq!(a.state, b.state);
        assert_eq!(a.dimensions, b.dimensions);
    }

    #[test]
    fn fabric_bootstrap_skips_to_active_not_from_failed() {
        let mut path = FabricBootstrapPath::new();
        // try to init without loading native first => should stay uninitialized
        path.on_renderer_init(800, 600);
        assert_eq!(path.state, FabricBootstrapState::Uninitialized);
    }

    #[test]
    fn fabric_bootstrap_world_unload_from_non_active_is_noop() {
        let mut path = FabricBootstrapPath::new();
        path.on_native_loaded();
        path.on_world_unload(); // should not transition since not active
        assert_eq!(path.state, FabricBootstrapState::NativeLoaded);
    }

    // -----------------------------------------------------------------------
    // MinecraftFramePacket tests
    // -----------------------------------------------------------------------

    #[test]
    fn minecraft_frame_packet_fields() {
        let packet = super::MinecraftFramePacket {
            frame: FrameInput {
                delta_seconds: 0.016,
                time_seconds: 10.0,
                camera: CameraInput::default(),
            },
            visible_chunks: vec![],
        };
        assert!(packet.visible_chunks.is_empty());
        assert!((packet.frame.time_seconds - 10.0).abs() < f32::EPSILON);
    }

    // -----------------------------------------------------------------------
    // RenderPipelineProvider tests
    // -----------------------------------------------------------------------

    #[test]
    fn render_pipeline_provider_standalone_has_no_shader_pack() {
        let p = RenderPipelineProvider::standalone();
        assert!(p.shader_pack.is_none());
    }

    #[test]
    fn render_pipeline_provider_fabric_iris_has_iris_pack() {
        let p = RenderPipelineProvider::fabric_iris();
        let pack = p.shader_pack.as_ref().unwrap();
        assert_eq!(pack.model, ShaderPackModel::IrisLegacy);
        assert!(!pack.active_stages.is_empty());
    }

    // -----------------------------------------------------------------------
    // RenderEntryPath exhaustive
    // -----------------------------------------------------------------------

    #[test]
    fn render_entry_path_all_variants() {
        let paths = [
            RenderEntryPath::Standalone,
            RenderEntryPath::FabricIris,
            RenderEntryPath::Aperture,
        ];
        assert_eq!(paths.len(), 3);
    }
}
