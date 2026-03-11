use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum ShaderDialect {
    Glsl,
    Wgsl,
    SpirV,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum GlslStage {
    Vertex,
    Fragment,
    Compute,
}

impl GlslStage {
    fn to_naga(self) -> naga::ShaderStage {
        match self {
            Self::Vertex => naga::ShaderStage::Vertex,
            Self::Fragment => naga::ShaderStage::Fragment,
            Self::Compute => naga::ShaderStage::Compute,
        }
    }
}

pub type ShaderDefines = HashMap<String, String>;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShaderPipelineConfig {
    pub dialect: ShaderDialect,
    pub supports_hot_reload: bool,
    pub needs_spirv_translation: bool,
}

impl Default for ShaderPipelineConfig {
    fn default() -> Self {
        Self {
            dialect: ShaderDialect::Wgsl,
            supports_hot_reload: true,
            needs_spirv_translation: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ShaderPipelineMetadata {
    pub label: String,
    pub vertex_entry: String,
    pub fragment_entry: String,
    pub requires_depth: bool,
}

impl ShaderPipelineMetadata {
    pub fn standalone_voxel() -> Self {
        Self {
            label: "Ferridian Standalone Voxel Shader".to_string(),
            vertex_entry: "vs_main".to_string(),
            fragment_entry: "fs_main".to_string(),
            requires_depth: true,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ShaderAsset {
    path: PathBuf,
    source: String,
    modified_at: Option<SystemTime>,
    pub config: ShaderPipelineConfig,
    pub metadata: ShaderPipelineMetadata,
    pub glsl_stage: Option<GlslStage>,
    pub defines: ShaderDefines,
}

impl ShaderAsset {
    pub fn load_workspace_wgsl(
        relative_path: impl AsRef<Path>,
        metadata: ShaderPipelineMetadata,
    ) -> Result<Self> {
        let path = workspace_root().join(relative_path.as_ref());
        Self::load_absolute(path, ShaderPipelineConfig::default(), metadata, None)
    }

    pub fn load_workspace_glsl(
        relative_path: impl AsRef<Path>,
        metadata: ShaderPipelineMetadata,
        stage: GlslStage,
    ) -> Result<Self> {
        let path = workspace_root().join(relative_path.as_ref());
        let config = ShaderPipelineConfig {
            dialect: ShaderDialect::Glsl,
            supports_hot_reload: true,
            needs_spirv_translation: true,
        };
        Self::load_absolute(path, config, metadata, Some(stage))
    }

    pub fn create_module(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
        match self.config.dialect {
            ShaderDialect::Wgsl => device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(&self.metadata.label),
                source: wgpu::ShaderSource::Wgsl(self.source.clone().into()),
            }),
            ShaderDialect::Glsl => {
                let stage = self.glsl_stage.unwrap_or(GlslStage::Vertex).to_naga();
                device.create_shader_module(wgpu::ShaderModuleDescriptor {
                    label: Some(&self.metadata.label),
                    source: wgpu::ShaderSource::Glsl {
                        shader: self.source.clone().into(),
                        stage,
                        defines: Default::default(),
                    },
                })
            }
            ShaderDialect::SpirV => {
                panic!("SpirV dialect requires create_module_spirv with raw bytes");
            }
        }
    }

    pub fn reload_if_changed(&mut self) -> Result<bool> {
        let metadata = fs::metadata(&self.path)
            .with_context(|| format!("failed to stat shader file {}", self.path.display()))?;
        let modified_at = metadata.modified().ok();
        if modified_at == self.modified_at {
            return Ok(false);
        }

        let source = fs::read_to_string(&self.path)
            .with_context(|| format!("failed to read shader file {}", self.path.display()))?;
        self.source = source;
        self.modified_at = modified_at;
        Ok(true)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn source(&self) -> &str {
        &self.source
    }

    pub fn permutation_key(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.source.hash(&mut hasher);
        self.config.dialect.hash(&mut hasher);
        let mut sorted_defines: Vec<_> = self.defines.iter().collect();
        sorted_defines.sort_by_key(|(k, _)| (*k).clone());
        for (key, value) in sorted_defines {
            key.hash(&mut hasher);
            value.hash(&mut hasher);
        }
        hasher.finish()
    }

    fn load_absolute(
        path: PathBuf,
        config: ShaderPipelineConfig,
        metadata: ShaderPipelineMetadata,
        glsl_stage: Option<GlslStage>,
    ) -> Result<Self> {
        let file_metadata = fs::metadata(&path)
            .with_context(|| format!("failed to stat shader file {}", path.display()))?;
        let source = fs::read_to_string(&path)
            .with_context(|| format!("failed to read shader file {}", path.display()))?;

        Ok(Self {
            path,
            source,
            modified_at: file_metadata.modified().ok(),
            config,
            metadata,
            glsl_stage,
            defines: ShaderDefines::new(),
        })
    }
}

/// Validates WGSL source using naga's front-end parser and validator.
pub fn validate_wgsl(source: &str) -> Result<naga::valid::ModuleInfo> {
    let module = naga::front::wgsl::parse_str(source)
        .map_err(|err| anyhow::anyhow!("WGSL parse error: {err}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|err| anyhow::anyhow!("WGSL validation error: {err}"))?;
    Ok(info)
}

/// Validates GLSL source using naga's GLSL front-end parser and validator.
pub fn validate_glsl(source: &str, stage: GlslStage) -> Result<naga::valid::ModuleInfo> {
    let mut parser = naga::front::glsl::Frontend::default();
    let options = naga::front::glsl::Options {
        stage: stage.to_naga(),
        defines: Default::default(),
    };
    let module = parser
        .parse(&options, source)
        .map_err(|errors| anyhow::anyhow!("GLSL parse errors: {errors:?}"))?;
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|err| anyhow::anyhow!("GLSL validation error: {err}"))?;
    Ok(info)
}

// ---------------------------------------------------------------------------
// Modular WGSL composition with imports and defines.
// ---------------------------------------------------------------------------

/// A simple WGSL preprocessor that handles `#import` directives and
/// `#define` / `#ifdef` / `#ifndef` / `#endif` blocks at the text level,
/// then feeds the result through the naga validator to produce a Module.
#[derive(Debug, Default)]
pub struct ShaderComposer {
    modules: HashMap<String, String>,
}

impl ShaderComposer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a WGSL source module that can be imported by name.
    pub fn register_module(&mut self, name: &str, source: &str) {
        self.modules.insert(name.to_string(), source.to_string());
    }

    /// Compose a final WGSL module, resolving `#import <name>` directives and
    /// evaluating `#ifdef` / `#ifndef` / `#else` / `#endif` blocks.
    pub fn compose(&self, source: &str, defines: &HashMap<String, String>) -> Result<naga::Module> {
        let resolved = self.resolve_source(source, defines, &mut Vec::new())?;
        let module = naga::front::wgsl::parse_str(&resolved)
            .map_err(|err| anyhow::anyhow!("WGSL parse error during compose: {err}"))?;
        let _info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .map_err(|err| anyhow::anyhow!("WGSL validation error during compose: {err}"))?;
        Ok(module)
    }

    fn resolve_source(
        &self,
        source: &str,
        defines: &HashMap<String, String>,
        import_stack: &mut Vec<String>,
    ) -> Result<String> {
        let mut output = String::new();
        let mut skip_depth: usize = 0;
        let mut branch_stack: Vec<bool> = Vec::new();

        for line in source.lines() {
            let trimmed = line.trim();

            if let Some(name) = trimmed.strip_prefix("#import ") {
                let name = name.trim();
                if skip_depth == 0 {
                    anyhow::ensure!(
                        !import_stack.contains(&name.to_string()),
                        "circular import detected: {name}"
                    );
                    let module_src = self
                        .modules
                        .get(name)
                        .ok_or_else(|| anyhow::anyhow!("unknown module: {name}"))?;
                    import_stack.push(name.to_string());
                    let resolved = self.resolve_source(module_src, defines, import_stack)?;
                    import_stack.pop();
                    output.push_str(&resolved);
                    output.push('\n');
                }
            } else if let Some(key) = trimmed.strip_prefix("#ifdef ") {
                let active = defines.contains_key(key.trim());
                branch_stack.push(active);
                if !active {
                    skip_depth += 1;
                }
            } else if let Some(key) = trimmed.strip_prefix("#ifndef ") {
                let active = !defines.contains_key(key.trim());
                branch_stack.push(active);
                if !active {
                    skip_depth += 1;
                }
            } else if trimmed == "#else" {
                if let Some(was_active) = branch_stack.last_mut() {
                    if *was_active {
                        skip_depth += 1;
                    } else {
                        skip_depth = skip_depth.saturating_sub(1);
                    }
                    *was_active = !*was_active;
                }
            } else if trimmed == "#endif" {
                if let Some(was_active) = branch_stack.pop() {
                    if !was_active {
                        skip_depth = skip_depth.saturating_sub(1);
                    }
                }
            } else if skip_depth == 0 {
                output.push_str(line);
                output.push('\n');
            }
        }
        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Shader binding layout validator — catches binding drift at compile/test time
// ---------------------------------------------------------------------------

/// Describes expected binding layout for type-safe validation against WGSL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpectedBinding {
    pub group: u32,
    pub binding: u32,
    pub name: String,
}

/// Extracts `@group(N) @binding(M)` declarations from WGSL source text.
pub fn extract_wgsl_bindings(source: &str) -> Vec<ExpectedBinding> {
    let mut bindings = Vec::new();
    let mut group: Option<u32> = None;
    let mut binding: Option<u32> = None;

    for line in source.lines() {
        let trimmed = line.trim();
        // Parse @group(N) and @binding(M) on the same or adjacent lines.
        for token in trimmed.split_whitespace() {
            if let Some(inner) = token
                .strip_prefix("@group(")
                .and_then(|s| s.strip_suffix(')'))
            {
                group = inner.parse().ok();
            }
            if let Some(inner) = token
                .strip_prefix("@binding(")
                .and_then(|s| s.strip_suffix(')'))
            {
                binding = inner.parse().ok();
            }
        }
        if let (Some(g), Some(b)) = (group, binding) {
            // Extract the variable name from `var<...> name` or `var name`.
            let name = trimmed
                .split("var")
                .nth(1)
                .and_then(|rest| {
                    let rest = rest.trim();
                    let rest = if rest.starts_with('<') {
                        rest.split('>').nth(1).unwrap_or(rest).trim()
                    } else {
                        rest
                    };
                    rest.split([':', ' ', ';'])
                        .next()
                        .map(|s| s.trim().to_string())
                })
                .unwrap_or_default();
            if !name.is_empty() {
                bindings.push(ExpectedBinding {
                    group: g,
                    binding: b,
                    name,
                });
            }
            group = None;
            binding = None;
        }
    }
    bindings
}

/// Validates that the bindings found in WGSL source match the expected layout.
pub fn validate_binding_layout(
    source: &str,
    expected: &[ExpectedBinding],
) -> std::result::Result<(), Vec<String>> {
    let actual = extract_wgsl_bindings(source);
    let mut errors = Vec::new();

    for exp in expected {
        if !actual
            .iter()
            .any(|a| a.group == exp.group && a.binding == exp.binding)
        {
            errors.push(format!(
                "missing binding: group={} binding={} ({})",
                exp.group, exp.binding, exp.name
            ));
        }
    }

    for act in &actual {
        if !expected
            .iter()
            .any(|e| e.group == act.group && e.binding == act.binding)
        {
            errors.push(format!(
                "unexpected binding: group={} binding={} ({})",
                act.group, act.binding, act.name
            ));
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

/// A cache for compiled shader permutations, keyed by source hash and defines.
#[derive(Debug, Default)]
pub struct ShaderPermutationCache {
    entries: HashMap<u64, CachedPermutation>,
}

#[derive(Debug, Clone)]
pub struct CachedPermutation {
    pub key: u64,
    pub label: String,
    pub dialect: ShaderDialect,
}

impl ShaderPermutationCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get(&self, key: u64) -> Option<&CachedPermutation> {
        self.entries.get(&key)
    }

    pub fn insert(&mut self, asset: &ShaderAsset) -> u64 {
        let key = asset.permutation_key();
        self.entries.insert(
            key,
            CachedPermutation {
                key,
                label: asset.metadata.label.clone(),
                dialect: asset.config.dialect,
            },
        );
        key
    }

    pub fn invalidate(&mut self, key: u64) -> bool {
        self.entries.remove(&key).is_some()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
    }
}

// ---------------------------------------------------------------------------
// SPIR-V module loading — loads pre-compiled Rust-GPU SPIR-V modules
// ---------------------------------------------------------------------------

/// A pre-compiled SPIR-V shader module produced by `cargo xtask build-shaders`.
#[derive(Debug, Clone)]
pub struct SpirvModule {
    pub label: String,
    pub spirv_bytes: Vec<u8>,
}

impl SpirvModule {
    /// Load a SPIR-V module from raw bytes (e.g. from `include_bytes!`).
    pub fn from_bytes(label: impl Into<String>, bytes: &[u8]) -> Result<Self> {
        anyhow::ensure!(
            bytes.len() >= 4 && bytes.len() % 4 == 0,
            "SPIR-V data must be 4-byte aligned and non-empty"
        );
        // Validate SPIR-V magic number (0x07230203)
        let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
        anyhow::ensure!(
            magic == 0x07230203,
            "invalid SPIR-V magic number: {magic:#010x}"
        );
        Ok(Self {
            label: label.into(),
            spirv_bytes: bytes.to_vec(),
        })
    }

    /// Load a SPIR-V module from a file on disk.
    pub fn load_file(label: impl Into<String>, path: &Path) -> Result<Self> {
        let bytes = fs::read(path)
            .with_context(|| format!("failed to read SPIR-V file: {}", path.display()))?;
        Self::from_bytes(label, &bytes)
    }

    /// Load the default Ferridian SPIR-V module from `shaders/spirv/`.
    pub fn load_default() -> Result<Self> {
        let path = workspace_root()
            .join("shaders")
            .join("spirv")
            .join("ferridian_shader_gpu.spv");
        Self::load_file("ferridian-shader-gpu", &path)
    }

    /// Create a wgpu shader module from this SPIR-V data.
    ///
    /// # Safety
    /// The caller must ensure the SPIR-V bytecode is valid and trusted.
    /// wgpu will validate the SPIR-V through naga, but malformed modules
    /// could still cause driver issues on some backends.
    pub unsafe fn create_module(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
        let spirv: &[u32] = bytemuck::cast_slice(&self.spirv_bytes);
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&self.label),
            source: wgpu::ShaderSource::SpirV(std::borrow::Cow::Borrowed(spirv)),
        })
    }

    pub fn word_count(&self) -> usize {
        self.spirv_bytes.len() / 4
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

// ---------------------------------------------------------------------------
// Shader pack loading — reads Iris/OptiFine .zip or folder-based shader packs,
// parses shaders.properties, and loads GLSL programs per pipeline stage.
// ---------------------------------------------------------------------------

/// Iris/OptiFine shader stage file names mapped to their pipeline function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum IrisShaderStage {
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

impl IrisShaderStage {
    pub fn file_prefix(self) -> &'static str {
        match self {
            Self::Shadow => "shadow",
            Self::GBuffersTerrain => "gbuffers_terrain",
            Self::GBuffersEntities => "gbuffers_entities",
            Self::GBuffersBlock => "gbuffers_block",
            Self::GBuffersWater => "gbuffers_water",
            Self::GBuffersSkyBasic => "gbuffers_skybasic",
            Self::GBuffersSkyTextured => "gbuffers_skytextured",
            Self::GBuffersWeather => "gbuffers_weather",
            Self::GBuffersHand => "gbuffers_hand",
            Self::Deferred => "deferred",
            Self::Composite => "composite",
            Self::Final => "final",
        }
    }

    pub fn all() -> &'static [Self] {
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

/// A loaded GLSL shader program for a single Iris pipeline stage.
#[derive(Debug, Clone)]
pub struct IrisShaderProgram {
    pub stage: IrisShaderStage,
    pub vertex_source: Option<String>,
    pub fragment_source: Option<String>,
    pub compute_source: Option<String>,
}

impl IrisShaderProgram {
    pub fn validate(&self) -> Result<()> {
        if let Some(ref src) = self.vertex_source {
            validate_glsl(src, GlslStage::Vertex)
                .with_context(|| format!("{} vertex", self.stage.file_prefix()))?;
        }
        if let Some(ref src) = self.fragment_source {
            validate_glsl(src, GlslStage::Fragment)
                .with_context(|| format!("{} fragment", self.stage.file_prefix()))?;
        }
        if let Some(ref src) = self.compute_source {
            validate_glsl(src, GlslStage::Compute)
                .with_context(|| format!("{} compute", self.stage.file_prefix()))?;
        }
        Ok(())
    }

    pub fn has_any_source(&self) -> bool {
        self.vertex_source.is_some()
            || self.fragment_source.is_some()
            || self.compute_source.is_some()
    }
}

/// Parsed `shaders.properties` from an Iris shader pack.
#[derive(Debug, Clone, Default)]
pub struct ShaderPackProperties {
    pub entries: HashMap<String, String>,
}

impl ShaderPackProperties {
    pub fn parse(content: &str) -> Self {
        let mut entries = HashMap::new();
        for line in content.lines() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            if let Some((key, value)) = trimmed.split_once('=') {
                entries.insert(key.trim().to_string(), value.trim().to_string());
            }
        }
        Self { entries }
    }

    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(|s| s.as_str())
    }

    pub fn shadow_map_resolution(&self) -> u32 {
        self.get("shadow.resolution")
            .and_then(|s| s.parse().ok())
            .unwrap_or(2048)
    }

    pub fn shadow_distance(&self) -> f32 {
        self.get("shadowDistance")
            .and_then(|s| s.parse().ok())
            .unwrap_or(128.0)
    }

    pub fn old_lighting(&self) -> bool {
        self.get("oldLighting")
            .map(|s| s == "true")
            .unwrap_or(false)
    }

    pub fn normals_enabled(&self) -> bool {
        self.get("texture.normal")
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }

    pub fn specular_enabled(&self) -> bool {
        self.get("texture.specular")
            .map(|s| !s.is_empty())
            .unwrap_or(false)
    }
}

/// A loaded Iris shader pack with all programs and properties.
#[derive(Debug, Clone)]
pub struct ShaderPack {
    pub name: String,
    pub properties: ShaderPackProperties,
    pub programs: Vec<IrisShaderProgram>,
}

impl ShaderPack {
    pub fn program(&self, stage: IrisShaderStage) -> Option<&IrisShaderProgram> {
        self.programs.iter().find(|p| p.stage == stage)
    }

    pub fn active_stages(&self) -> Vec<IrisShaderStage> {
        self.programs
            .iter()
            .filter(|p| p.has_any_source())
            .map(|p| p.stage)
            .collect()
    }

    pub fn validate_all(&self) -> Result<()> {
        for program in &self.programs {
            if program.has_any_source() {
                program.validate()?;
            }
        }
        Ok(())
    }
}

/// Loads shader packs from zip files or directories.
pub struct ShaderPackLoader;

impl ShaderPackLoader {
    /// Load a shader pack from a directory on disk.
    pub fn load_directory(path: &Path) -> Result<ShaderPack> {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let shaders_dir = if path.join("shaders").exists() {
            path.join("shaders")
        } else {
            path.to_path_buf()
        };

        let properties = Self::load_properties_from_dir(&shaders_dir)?;
        let programs = Self::load_programs_from_dir(&shaders_dir)?;

        Ok(ShaderPack {
            name,
            properties,
            programs,
        })
    }

    /// Load a shader pack from a .zip file.
    pub fn load_zip(path: &Path) -> Result<ShaderPack> {
        let name = path
            .file_stem()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        let file = fs::File::open(path)
            .with_context(|| format!("failed to open shader pack: {}", path.display()))?;
        let mut archive = zip::ZipArchive::new(file)
            .with_context(|| format!("invalid zip file: {}", path.display()))?;

        // Detect the shaders/ prefix inside the zip
        let prefix = Self::detect_zip_prefix(&mut archive);

        let properties = Self::load_properties_from_zip(&mut archive, &prefix)?;
        let programs = Self::load_programs_from_zip(&mut archive, &prefix)?;

        Ok(ShaderPack {
            name,
            properties,
            programs,
        })
    }

    /// Auto-detect whether the path is a directory or zip and load accordingly.
    pub fn load(path: &Path) -> Result<ShaderPack> {
        if path.is_dir() {
            Self::load_directory(path)
        } else {
            Self::load_zip(path)
        }
    }

    fn load_properties_from_dir(shaders_dir: &Path) -> Result<ShaderPackProperties> {
        let props_path = shaders_dir.join("shaders.properties");
        if props_path.exists() {
            let content = fs::read_to_string(&props_path)
                .with_context(|| format!("failed to read {}", props_path.display()))?;
            Ok(ShaderPackProperties::parse(&content))
        } else {
            Ok(ShaderPackProperties::default())
        }
    }

    fn load_programs_from_dir(shaders_dir: &Path) -> Result<Vec<IrisShaderProgram>> {
        let mut programs = Vec::new();
        for &stage in IrisShaderStage::all() {
            let prefix = stage.file_prefix();
            let vsh = shaders_dir.join(format!("{prefix}.vsh"));
            let fsh = shaders_dir.join(format!("{prefix}.fsh"));
            let csh = shaders_dir.join(format!("{prefix}.csh"));

            let vertex_source = if vsh.exists() {
                Some(fs::read_to_string(&vsh).with_context(|| vsh.display().to_string())?)
            } else {
                None
            };
            let fragment_source = if fsh.exists() {
                Some(fs::read_to_string(&fsh).with_context(|| fsh.display().to_string())?)
            } else {
                None
            };
            let compute_source = if csh.exists() {
                Some(fs::read_to_string(&csh).with_context(|| csh.display().to_string())?)
            } else {
                None
            };

            let program = IrisShaderProgram {
                stage,
                vertex_source,
                fragment_source,
                compute_source,
            };
            if program.has_any_source() {
                programs.push(program);
            }
        }
        Ok(programs)
    }

    fn detect_zip_prefix(archive: &mut zip::ZipArchive<fs::File>) -> String {
        for i in 0..archive.len() {
            if let Ok(entry) = archive.by_index_raw(i) {
                let name = entry.name();
                if name.ends_with("shaders.properties") {
                    if let Some(prefix) = name.strip_suffix("shaders.properties") {
                        return prefix.to_string();
                    }
                }
            }
        }
        // Try common prefixes
        for i in 0..archive.len() {
            if let Ok(entry) = archive.by_index_raw(i) {
                let name = entry.name();
                if name.contains("/shaders/") {
                    if let Some(idx) = name.find("/shaders/") {
                        return format!("{}/shaders/", &name[..idx]);
                    }
                }
                if name.starts_with("shaders/") {
                    return "shaders/".to_string();
                }
            }
        }
        String::new()
    }

    fn load_properties_from_zip(
        archive: &mut zip::ZipArchive<fs::File>,
        prefix: &str,
    ) -> Result<ShaderPackProperties> {
        let props_name = format!("{prefix}shaders.properties");
        match archive.by_name(&props_name) {
            Ok(mut entry) => {
                let mut content = String::new();
                std::io::Read::read_to_string(&mut entry, &mut content)?;
                Ok(ShaderPackProperties::parse(&content))
            }
            Err(_) => Ok(ShaderPackProperties::default()),
        }
    }

    fn load_programs_from_zip(
        archive: &mut zip::ZipArchive<fs::File>,
        prefix: &str,
    ) -> Result<Vec<IrisShaderProgram>> {
        let mut programs = Vec::new();
        for &stage in IrisShaderStage::all() {
            let file_prefix = stage.file_prefix();
            let vsh_name = format!("{prefix}{file_prefix}.vsh");
            let fsh_name = format!("{prefix}{file_prefix}.fsh");
            let csh_name = format!("{prefix}{file_prefix}.csh");

            let vertex_source = Self::read_zip_entry(archive, &vsh_name)?;
            let fragment_source = Self::read_zip_entry(archive, &fsh_name)?;
            let compute_source = Self::read_zip_entry(archive, &csh_name)?;

            let program = IrisShaderProgram {
                stage,
                vertex_source,
                fragment_source,
                compute_source,
            };
            if program.has_any_source() {
                programs.push(program);
            }
        }
        Ok(programs)
    }

    fn read_zip_entry(
        archive: &mut zip::ZipArchive<fs::File>,
        name: &str,
    ) -> Result<Option<String>> {
        match archive.by_name(name) {
            Ok(mut entry) => {
                let mut content = String::new();
                std::io::Read::read_to_string(&mut entry, &mut content)?;
                Ok(Some(content))
            }
            Err(zip::result::ZipError::FileNotFound) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ShaderAsset, ShaderPermutationCache, ShaderPipelineConfig, ShaderPipelineMetadata,
        validate_wgsl,
    };
    use std::fs;
    use std::path::PathBuf;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn loads_workspace_shader_from_disk() {
        let shader = ShaderAsset::load_workspace_wgsl(
            "shaders/wgsl/standalone.wgsl",
            ShaderPipelineMetadata::standalone_voxel(),
        )
        .expect("workspace shader should load");

        assert!(shader.path().ends_with("shaders/wgsl/standalone.wgsl"));
    }

    #[test]
    fn reload_if_changed_refreshes_shader_source() {
        let path = unique_test_shader_path();
        fs::write(&path, "// initial\n").expect("should create test shader");

        let mut shader = ShaderAsset::load_absolute(
            path.clone(),
            ShaderPipelineConfig::default(),
            ShaderPipelineMetadata::standalone_voxel(),
            None,
        )
        .expect("test shader should load");

        fs::write(&path, "// updated\n").expect("should update shader source");
        shader.modified_at = None;

        let did_reload = shader
            .reload_if_changed()
            .expect("reload should succeed after source change");

        assert!(did_reload);
        assert!(shader.source().contains("updated"));

        let _ = fs::remove_file(path);
    }

    #[test]
    fn validates_wgsl_source_with_naga() {
        let shader = ShaderAsset::load_workspace_wgsl(
            "shaders/wgsl/standalone.wgsl",
            ShaderPipelineMetadata::standalone_voxel(),
        )
        .expect("should load");

        let info = validate_wgsl(shader.source());
        assert!(info.is_ok(), "standalone shader should validate: {info:?}");
    }

    #[test]
    fn rejects_invalid_wgsl() {
        let result = validate_wgsl("fn broken(x: nonexistent_type) {}");
        assert!(result.is_err());
    }

    #[test]
    fn permutation_cache_inserts_and_retrieves() {
        let mut cache = ShaderPermutationCache::new();
        assert!(cache.is_empty());

        let shader = ShaderAsset::load_workspace_wgsl(
            "shaders/wgsl/standalone.wgsl",
            ShaderPipelineMetadata::standalone_voxel(),
        )
        .expect("should load");

        let key = cache.insert(&shader);
        assert_eq!(cache.len(), 1);
        assert!(cache.get(key).is_some());
        assert!(cache.invalidate(key));
        assert!(cache.is_empty());
    }

    #[test]
    fn permutation_key_changes_with_source() {
        let path = unique_test_shader_path();
        fs::write(&path, "// version A\n").expect("create");

        let shader_a = ShaderAsset::load_absolute(
            path.clone(),
            ShaderPipelineConfig::default(),
            ShaderPipelineMetadata::standalone_voxel(),
            None,
        )
        .expect("load a");

        fs::write(&path, "// version B\n").expect("update");
        let shader_b = ShaderAsset::load_absolute(
            path.clone(),
            ShaderPipelineConfig::default(),
            ShaderPipelineMetadata::standalone_voxel(),
            None,
        )
        .expect("load b");

        assert_ne!(shader_a.permutation_key(), shader_b.permutation_key());

        let _ = fs::remove_file(path);
    }

    fn unique_test_shader_path() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock should be monotonic enough for tests")
            .as_nanos();
        std::env::temp_dir().join(format!("ferridian-shader-test-{unique}.wgsl"))
    }

    #[test]
    fn shader_composer_creates_module() {
        let composer = super::ShaderComposer::new();
        let source = r#"
            @vertex
            fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
                return vec4<f32>(0.0, 0.0, 0.0, 1.0);
            }
        "#;
        let module = composer.compose(source, &Default::default());
        assert!(module.is_ok());
    }

    #[test]
    fn shader_composer_resolves_imports() {
        let mut composer = super::ShaderComposer::new();
        composer.register_module("utils", "fn helper() -> f32 { return 1.0; }");
        let source = "#import utils\n@vertex\nfn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {\n    return vec4<f32>(helper(), 0.0, 0.0, 1.0);\n}";
        let module = composer.compose(source, &Default::default());
        assert!(module.is_ok());
    }

    #[test]
    fn shader_composer_ifdef() {
        let composer = super::ShaderComposer::new();
        let source = "\
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var y: f32 = 0.0;
#ifdef USE_OFFSET
    y = 1.0;
#endif
    return vec4<f32>(0.0, y, 0.0, 1.0);
}";
        // Without define — should compile with y = 0.0
        let module = composer.compose(source, &Default::default());
        assert!(module.is_ok());

        // With define — should compile with y = 1.0
        let mut defs = std::collections::HashMap::new();
        defs.insert("USE_OFFSET".to_string(), "1".to_string());
        let module = composer.compose(source, &defs);
        assert!(module.is_ok());
    }

    #[test]
    fn extract_wgsl_bindings_finds_group_binding() {
        let source = r#"
@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(0) @binding(1) var<storage, read> chunks: array<ChunkData>;
@group(1) @binding(0) var tex: texture_2d<f32>;
"#;
        let bindings = super::extract_wgsl_bindings(source);
        assert_eq!(bindings.len(), 3);
        assert_eq!(bindings[0].group, 0);
        assert_eq!(bindings[0].binding, 0);
        assert_eq!(bindings[0].name, "camera");
        assert_eq!(bindings[1].group, 0);
        assert_eq!(bindings[1].binding, 1);
        assert_eq!(bindings[1].name, "chunks");
        assert_eq!(bindings[2].group, 1);
        assert_eq!(bindings[2].binding, 0);
        assert_eq!(bindings[2].name, "tex");
    }

    #[test]
    fn validate_binding_layout_catches_drift() {
        let source = "@group(0) @binding(0) var<uniform> camera: CameraUniform;\n";
        let expected = vec![
            super::ExpectedBinding {
                group: 0,
                binding: 0,
                name: "camera".into(),
            },
            super::ExpectedBinding {
                group: 0,
                binding: 1,
                name: "missing".into(),
            },
        ];
        let result = super::validate_binding_layout(source, &expected);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.contains("missing binding")));
    }

    // -----------------------------------------------------------------------
    // GLSL validation tests
    // -----------------------------------------------------------------------

    #[test]
    fn validate_glsl_vertex_shader() {
        let source = r#"
            #version 450
            layout(location = 0) in vec3 a_pos;
            void main() {
                gl_Position = vec4(a_pos, 1.0);
            }
        "#;
        let result = super::validate_glsl(source, super::GlslStage::Vertex);
        assert!(
            result.is_ok(),
            "valid GLSL vertex shader should validate: {result:?}"
        );
    }

    #[test]
    fn validate_glsl_fragment_shader() {
        let source = r#"
            #version 450
            layout(location = 0) out vec4 frag_color;
            void main() {
                frag_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        "#;
        let result = super::validate_glsl(source, super::GlslStage::Fragment);
        assert!(
            result.is_ok(),
            "valid GLSL fragment shader should validate: {result:?}"
        );
    }

    #[test]
    fn validate_glsl_rejects_invalid_source() {
        let result = super::validate_glsl(
            "#version 450\nvoid main() { nonexistent_func(); }",
            super::GlslStage::Vertex,
        );
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // GlslStage mapping
    // -----------------------------------------------------------------------

    #[test]
    fn glsl_stage_to_naga_mapping() {
        assert_eq!(
            super::GlslStage::Vertex.to_naga(),
            naga::ShaderStage::Vertex
        );
        assert_eq!(
            super::GlslStage::Fragment.to_naga(),
            naga::ShaderStage::Fragment
        );
        assert_eq!(
            super::GlslStage::Compute.to_naga(),
            naga::ShaderStage::Compute
        );
    }

    // -----------------------------------------------------------------------
    // ShaderPipelineConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn shader_pipeline_config_defaults() {
        let config = super::ShaderPipelineConfig::default();
        assert_eq!(config.dialect, super::ShaderDialect::Wgsl);
        assert!(config.supports_hot_reload);
        assert!(config.needs_spirv_translation);
    }

    // -----------------------------------------------------------------------
    // ShaderPipelineMetadata
    // -----------------------------------------------------------------------

    #[test]
    fn standalone_voxel_metadata_values() {
        let meta = ShaderPipelineMetadata::standalone_voxel();
        assert_eq!(meta.vertex_entry, "vs_main");
        assert_eq!(meta.fragment_entry, "fs_main");
        assert!(meta.requires_depth);
    }

    // -----------------------------------------------------------------------
    // ShaderAsset path and source accessors
    // -----------------------------------------------------------------------

    #[test]
    fn shader_asset_path_and_source_accessors() {
        let shader = ShaderAsset::load_workspace_wgsl(
            "shaders/wgsl/standalone.wgsl",
            ShaderPipelineMetadata::standalone_voxel(),
        )
        .expect("should load");
        assert!(!shader.source().is_empty());
        assert!(shader.path().exists());
    }

    // -----------------------------------------------------------------------
    // ShaderComposer advanced tests
    // -----------------------------------------------------------------------

    #[test]
    fn shader_composer_circular_import_detected() {
        let mut composer = super::ShaderComposer::new();
        composer.register_module("a", "#import b\nfn a_fn() -> f32 { return 1.0; }");
        composer.register_module("b", "#import a\nfn b_fn() -> f32 { return 2.0; }");
        let result = composer.compose("#import a", &Default::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("circular"),
            "expected circular import error, got: {msg}"
        );
    }

    #[test]
    fn shader_composer_unknown_module_error() {
        let composer = super::ShaderComposer::new();
        let result = composer.compose("#import nonexistent", &Default::default());
        assert!(result.is_err());
        let msg = result.unwrap_err().to_string();
        assert!(
            msg.contains("unknown module"),
            "expected unknown module error, got: {msg}"
        );
    }

    #[test]
    fn shader_composer_ifndef_includes_when_not_defined() {
        let composer = super::ShaderComposer::new();
        let source = "\
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var y: f32 = 0.0;
#ifndef DISABLED_FEATURE
    y = 1.0;
#endif
    return vec4<f32>(0.0, y, 0.0, 1.0);
}";
        // DISABLED_FEATURE is NOT defined => y = 1.0 should be included
        let module = composer.compose(source, &Default::default());
        assert!(module.is_ok());
    }

    #[test]
    fn shader_composer_ifndef_excludes_when_defined() {
        let composer = super::ShaderComposer::new();
        let source = "\
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var y: f32 = 0.0;
#ifndef USE_OFFSET
    y = 1.0;
#endif
    return vec4<f32>(0.0, y, 0.0, 1.0);
}";
        let mut defs = std::collections::HashMap::new();
        defs.insert("USE_OFFSET".to_string(), "1".to_string());
        // USE_OFFSET IS defined => y = 1.0 block should be skipped
        let module = composer.compose(source, &defs);
        assert!(module.is_ok());
    }

    #[test]
    fn shader_composer_else_branch() {
        let composer = super::ShaderComposer::new();
        let source = "\
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> @builtin(position) vec4<f32> {
    var y: f32 = 0.0;
#ifdef HAS_FEATURE
    y = 1.0;
#else
    y = 2.0;
#endif
    return vec4<f32>(0.0, y, 0.0, 1.0);
}";
        // Without define: should use else branch (y = 2.0)
        let module = composer.compose(source, &Default::default());
        assert!(module.is_ok());

        // With define: should use ifdef branch (y = 1.0)
        let mut defs = std::collections::HashMap::new();
        defs.insert("HAS_FEATURE".to_string(), "1".to_string());
        let module = composer.compose(source, &defs);
        assert!(module.is_ok());
    }

    // -----------------------------------------------------------------------
    // ShaderPermutationCache additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn permutation_cache_clear() {
        let mut cache = super::ShaderPermutationCache::new();
        let shader = ShaderAsset::load_workspace_wgsl(
            "shaders/wgsl/standalone.wgsl",
            ShaderPipelineMetadata::standalone_voxel(),
        )
        .expect("should load");
        cache.insert(&shader);
        assert!(!cache.is_empty());
        cache.clear();
        assert!(cache.is_empty());
        assert_eq!(cache.len(), 0);
    }

    #[test]
    fn permutation_key_changes_with_defines() {
        let path = unique_test_shader_path();
        fs::write(&path, "// same source\n").expect("create");

        let mut shader_a = ShaderAsset::load_absolute(
            path.clone(),
            ShaderPipelineConfig::default(),
            ShaderPipelineMetadata::standalone_voxel(),
            None,
        )
        .expect("load");

        let key_a = shader_a.permutation_key();
        shader_a
            .defines
            .insert("FEATURE_X".to_string(), "1".to_string());
        let key_b = shader_a.permutation_key();

        assert_ne!(key_a, key_b, "defines should affect permutation key");

        let _ = fs::remove_file(path);
    }

    // -----------------------------------------------------------------------
    // validate_binding_layout success case
    // -----------------------------------------------------------------------

    #[test]
    fn validate_binding_layout_passes_when_matching() {
        let source = "@group(0) @binding(0) var<uniform> camera: CameraUniform;\n@group(0) @binding(1) var<storage, read> data: array<f32>;\n";
        let expected = vec![
            super::ExpectedBinding {
                group: 0,
                binding: 0,
                name: "camera".into(),
            },
            super::ExpectedBinding {
                group: 0,
                binding: 1,
                name: "data".into(),
            },
        ];
        let result = super::validate_binding_layout(source, &expected);
        assert!(result.is_ok());
    }

    #[test]
    fn validate_binding_layout_detects_unexpected_bindings() {
        let source = "@group(0) @binding(0) var<uniform> camera: CameraUniform;\n@group(0) @binding(1) var<storage, read> extra: array<f32>;\n";
        let expected = vec![super::ExpectedBinding {
            group: 0,
            binding: 0,
            name: "camera".into(),
        }];
        let result = super::validate_binding_layout(source, &expected);
        assert!(result.is_err());
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|e| e.contains("unexpected binding")));
    }

    // -----------------------------------------------------------------------
    // Shader pack loading tests
    // -----------------------------------------------------------------------

    #[test]
    fn shader_pack_properties_parses_key_value() {
        let content = "shadow.resolution=4096\nshadowDistance=256.0\n# comment\noldLighting=true\n";
        let props = super::ShaderPackProperties::parse(content);
        assert_eq!(props.shadow_map_resolution(), 4096);
        assert_eq!(props.shadow_distance(), 256.0);
        assert!(props.old_lighting());
    }

    #[test]
    fn shader_pack_properties_defaults() {
        let props = super::ShaderPackProperties::default();
        assert_eq!(props.shadow_map_resolution(), 2048);
        assert_eq!(props.shadow_distance(), 128.0);
        assert!(!props.old_lighting());
    }

    #[test]
    fn iris_shader_stage_file_prefixes() {
        assert_eq!(super::IrisShaderStage::Shadow.file_prefix(), "shadow");
        assert_eq!(
            super::IrisShaderStage::GBuffersTerrain.file_prefix(),
            "gbuffers_terrain"
        );
        assert_eq!(super::IrisShaderStage::Final.file_prefix(), "final");
    }

    #[test]
    fn iris_shader_stage_all_has_12_entries() {
        assert_eq!(super::IrisShaderStage::all().len(), 12);
    }

    #[test]
    fn shader_pack_loads_from_directory() {
        let dir = std::env::temp_dir().join("ferridian-test-shaderpack");
        let shaders_dir = dir.join("shaders");
        let _ = fs::create_dir_all(&shaders_dir);

        // Write a minimal shader pack
        fs::write(
            shaders_dir.join("shaders.properties"),
            "shadow.resolution=1024\n",
        )
        .unwrap();
        fs::write(
            shaders_dir.join("gbuffers_terrain.vsh"),
            "#version 150\nvoid main() { gl_Position = vec4(0.0); }\n",
        )
        .unwrap();
        fs::write(
            shaders_dir.join("gbuffers_terrain.fsh"),
            "#version 150\nout vec4 fragColor;\nvoid main() { fragColor = vec4(1.0); }\n",
        )
        .unwrap();

        let pack = super::ShaderPackLoader::load_directory(&dir).unwrap();
        assert_eq!(pack.properties.shadow_map_resolution(), 1024);
        assert_eq!(pack.active_stages().len(), 1);
        assert_eq!(
            pack.active_stages()[0],
            super::IrisShaderStage::GBuffersTerrain
        );
        assert!(
            pack.program(super::IrisShaderStage::GBuffersTerrain)
                .unwrap()
                .vertex_source
                .is_some()
        );

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn shader_pack_auto_detect_shaders_subdir() {
        let dir = std::env::temp_dir().join("ferridian-test-shaderpack-autodetect");
        let shaders_dir = dir.join("shaders");
        let _ = fs::create_dir_all(&shaders_dir);

        fs::write(
            shaders_dir.join("shadow.vsh"),
            "#version 150\nvoid main() { gl_Position = vec4(0.0); }\n",
        )
        .unwrap();

        let pack = super::ShaderPackLoader::load(&dir).unwrap();
        assert_eq!(pack.active_stages().len(), 1);
        assert_eq!(pack.active_stages()[0], super::IrisShaderStage::Shadow);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn iris_program_validate_checks_glsl() {
        let program = super::IrisShaderProgram {
            stage: super::IrisShaderStage::GBuffersTerrain,
            vertex_source: Some(
                "#version 450\nvoid main() { gl_Position = vec4(0.0); }\n".to_string(),
            ),
            fragment_source: None,
            compute_source: None,
        };
        assert!(program.validate().is_ok());
    }

    #[test]
    fn iris_program_has_any_source() {
        let empty = super::IrisShaderProgram {
            stage: super::IrisShaderStage::Shadow,
            vertex_source: None,
            fragment_source: None,
            compute_source: None,
        };
        assert!(!empty.has_any_source());

        let with_vert = super::IrisShaderProgram {
            vertex_source: Some("test".into()),
            ..empty
        };
        assert!(with_vert.has_any_source());
    }

    // -----------------------------------------------------------------------
    // Validate new WGSL deferred shaders
    // -----------------------------------------------------------------------

    #[test]
    fn validates_gbuffer_fill_shader() {
        let root = super::workspace_root();
        let source = fs::read_to_string(root.join("shaders/wgsl/gbuffer_fill.wgsl"))
            .expect("should read gbuffer_fill.wgsl");
        let result = validate_wgsl(&source);
        assert!(
            result.is_ok(),
            "gbuffer_fill.wgsl should validate: {result:?}"
        );
    }

    #[test]
    fn validates_deferred_lighting_shader() {
        let root = super::workspace_root();
        let source = fs::read_to_string(root.join("shaders/wgsl/deferred_lighting.wgsl"))
            .expect("should read deferred_lighting.wgsl");
        let result = validate_wgsl(&source);
        assert!(
            result.is_ok(),
            "deferred_lighting.wgsl should validate: {result:?}"
        );
    }

    #[test]
    fn validates_shadow_depth_shader() {
        let root = super::workspace_root();
        let source = fs::read_to_string(root.join("shaders/wgsl/shadow_depth.wgsl"))
            .expect("should read shadow_depth.wgsl");
        let result = validate_wgsl(&source);
        assert!(
            result.is_ok(),
            "shadow_depth.wgsl should validate: {result:?}"
        );
    }

    #[test]
    fn validates_translucent_shader() {
        let root = super::workspace_root();
        let source = fs::read_to_string(root.join("shaders/wgsl/translucent.wgsl"))
            .expect("should read translucent.wgsl");
        let result = validate_wgsl(&source);
        assert!(
            result.is_ok(),
            "translucent.wgsl should validate: {result:?}"
        );
    }
}
