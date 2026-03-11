use serde::{Deserialize, Serialize};

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
