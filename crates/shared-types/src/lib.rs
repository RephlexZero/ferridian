//! Shared types between CPU (ferridian-core) and GPU (ferridian-shader-gpu).
//!
//! Every type here must be `#[repr(C)]`, `Pod`, and `Zeroable` so the same
//! binary layout is used on both sides of the pipeline with zero translation.
//! When compiled for `spirv`, this crate operates in `no_std` mode.

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]

use bytemuck::{Pod, Zeroable};

// ---------------------------------------------------------------------------
// Uniforms — laid out identically on CPU and GPU
// ---------------------------------------------------------------------------

/// Per-frame scene uniforms. Updated once per frame on the CPU side,
/// read by every shader pass on the GPU side.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SceneUniforms {
    pub view_projection: [[f32; 4]; 4],
    pub inv_view_projection: [[f32; 4]; 4],
    pub camera_position: [f32; 4],
    pub sun_direction: [f32; 4],
    pub sun_color: [f32; 4],
    pub ambient_color: [f32; 4],
    pub screen_size: [f32; 2],
    pub time: f32,
    pub _pad0: f32,
}

/// Shadow pass uniforms.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct ShadowUniforms {
    pub light_view_projection: [[f32; 4]; 4],
}

// ---------------------------------------------------------------------------
// Vertex formats
// ---------------------------------------------------------------------------

/// Packed terrain vertex — 8 bytes total.
///
/// `data[0]` bit layout:
///   [4:0]   position X (0..16)
///   [13:5]  position Y (0..256)
///   [18:14] position Z (0..16)
///   [21:19] normal index (0..5)
///   [23:22] ambient occlusion (0..3)
///   [31:24] block light + sky light (4 bits each)
///
/// `data[1]` bit layout:
///   [7:0]   texture / material index
///   [31:8]  reserved
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PackedVertex {
    pub data: [u32; 2],
}

impl PackedVertex {
    /// Pack a terrain vertex into 8 bytes.
    #[inline]
    pub fn new(pos: [u32; 3], normal_idx: u32, ao: u32, light: u32, tex_id: u32) -> Self {
        let d0 = (pos[0] & 0x1F)
            | ((pos[1] & 0x1FF) << 5)
            | ((pos[2] & 0x1F) << 14)
            | ((normal_idx & 0x7) << 19)
            | ((ao & 0x3) << 22)
            | ((light & 0xFF) << 24);
        let d1 = tex_id & 0xFF;
        Self { data: [d0, d1] }
    }

    /// Extract local X position (0..16).
    #[inline]
    pub fn x(&self) -> u32 {
        self.data[0] & 0x1F
    }

    /// Extract local Y position (0..256).
    #[inline]
    pub fn y(&self) -> u32 {
        (self.data[0] >> 5) & 0x1FF
    }

    /// Extract local Z position (0..16).
    #[inline]
    pub fn z(&self) -> u32 {
        (self.data[0] >> 14) & 0x1F
    }

    /// Extract face normal index (0..5).
    #[inline]
    pub fn normal_index(&self) -> u32 {
        (self.data[0] >> 19) & 0x7
    }

    /// Extract AO value (0..3).
    #[inline]
    pub fn ao(&self) -> u32 {
        (self.data[0] >> 22) & 0x3
    }

    /// Extract light value (block + sky, packed).
    #[inline]
    pub fn light(&self) -> u32 {
        (self.data[0] >> 24) & 0xFF
    }

    /// Extract texture / material index.
    #[inline]
    pub fn texture_id(&self) -> u32 {
        self.data[1] & 0xFF
    }
}

// ---------------------------------------------------------------------------
// Material data
// ---------------------------------------------------------------------------

/// Material lookup entry stored in a GPU buffer, indexed by texture ID.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct MaterialEntry {
    pub albedo: [f32; 4],
    pub roughness: f32,
    pub metallic: f32,
    pub emission: f32,
    pub _pad: f32,
}

// ---------------------------------------------------------------------------
// Indirect draw
// ---------------------------------------------------------------------------

/// GPU indirect draw command, matching `wgpu::DrawIndexedIndirect`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct DrawIndexedIndirect {
    pub index_count: u32,
    pub instance_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub first_instance: u32,
}

// ---------------------------------------------------------------------------
// Constants shared between CPU configuration and GPU shaders
// ---------------------------------------------------------------------------

pub const CHUNK_SIZE: u32 = 16;
pub const CHUNK_HEIGHT: u32 = 256;
pub const MAX_MATERIALS: u32 = 256;
pub const SHADOW_MAP_SIZE: u32 = 2048;
pub const NORMAL_FACE_POS_Y: u32 = 0;
pub const NORMAL_FACE_NEG_Y: u32 = 1;
pub const NORMAL_FACE_POS_X: u32 = 2;
pub const NORMAL_FACE_NEG_X: u32 = 3;
pub const NORMAL_FACE_POS_Z: u32 = 4;
pub const NORMAL_FACE_NEG_Z: u32 = 5;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packed_vertex_roundtrip() {
        let v = PackedVertex::new([15, 200, 12], 3, 2, 0xAB, 42);
        assert_eq!(v.x(), 15);
        assert_eq!(v.y(), 200);
        assert_eq!(v.z(), 12);
        assert_eq!(v.normal_index(), 3);
        assert_eq!(v.ao(), 2);
        assert_eq!(v.light(), 0xAB);
        assert_eq!(v.texture_id(), 42);
    }

    #[test]
    fn packed_vertex_zero() {
        let v = PackedVertex::new([0, 0, 0], 0, 0, 0, 0);
        assert_eq!(v.x(), 0);
        assert_eq!(v.y(), 0);
        assert_eq!(v.z(), 0);
        assert_eq!(v.normal_index(), 0);
        assert_eq!(v.ao(), 0);
        assert_eq!(v.texture_id(), 0);
    }

    #[test]
    fn packed_vertex_max_values() {
        let v = PackedVertex::new([15, 255, 15], 5, 3, 0xFF, 255);
        assert_eq!(v.x(), 15);
        assert_eq!(v.y(), 255);
        assert_eq!(v.z(), 15);
        assert_eq!(v.normal_index(), 5);
        assert_eq!(v.ao(), 3);
        assert_eq!(v.light(), 0xFF);
        assert_eq!(v.texture_id(), 255);
    }

    #[test]
    fn scene_uniforms_size() {
        assert_eq!(
            core::mem::size_of::<SceneUniforms>(),
            4 * 4 * 4  // view_projection
            + 4 * 4 * 4 // inv_view_projection
            + 4 * 4     // camera_position
            + 4 * 4     // sun_direction
            + 4 * 4     // sun_color
            + 4 * 4     // ambient_color
            + 4 * 2     // screen_size
            + 4          // time
            + 4, // _pad0
        );
    }

    #[test]
    fn material_entry_size() {
        // 8 floats = 32 bytes, nicely aligned
        assert_eq!(core::mem::size_of::<MaterialEntry>(), 32);
    }

    #[test]
    fn draw_indexed_indirect_size() {
        // Must match wgpu's expected layout (20 bytes)
        assert_eq!(core::mem::size_of::<DrawIndexedIndirect>(), 20);
    }
}
