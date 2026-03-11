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

#[cfg(test)]
mod tests {
    use super::{CameraInput, ChunkCoord, ChunkInput, FrameInput, MinecraftAdapter};

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
}
