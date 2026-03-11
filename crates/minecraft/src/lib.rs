use ferridian_core::RenderBackendPlan;
use ferridian_shader::ShaderPipelineConfig;

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
}
