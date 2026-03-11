use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ShaderDialect {
    Glsl,
    Wgsl,
}

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
}

impl ShaderAsset {
    pub fn load_workspace_wgsl(
        relative_path: impl AsRef<Path>,
        metadata: ShaderPipelineMetadata,
    ) -> Result<Self> {
        let path = workspace_root().join(relative_path.as_ref());
        Self::load_absolute(path, ShaderPipelineConfig::default(), metadata)
    }

    pub fn create_module(&self, device: &wgpu::Device) -> wgpu::ShaderModule {
        device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&self.metadata.label),
            source: wgpu::ShaderSource::Wgsl(self.source.clone().into()),
        })
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

    fn load_absolute(
        path: PathBuf,
        config: ShaderPipelineConfig,
        metadata: ShaderPipelineMetadata,
    ) -> Result<Self> {
        match config.dialect {
            ShaderDialect::Wgsl => {}
            ShaderDialect::Glsl => bail!("GLSL loading is planned but not implemented yet"),
        }

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
        })
    }
}

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .expect("workspace root")
        .to_path_buf()
}

#[cfg(test)]
mod tests {
    use super::{ShaderAsset, ShaderPipelineMetadata};

    #[test]
    fn loads_workspace_shader_from_disk() {
        let shader = ShaderAsset::load_workspace_wgsl(
            "shaders/wgsl/standalone.wgsl",
            ShaderPipelineMetadata::standalone_voxel(),
        )
        .expect("workspace shader should load");

        assert!(shader.path().ends_with("shaders/wgsl/standalone.wgsl"));
    }
}
