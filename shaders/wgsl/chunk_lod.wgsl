// Chunk LOD selection compute shader.
// For each chunk slot, computes the distance to the camera and selects an
// appropriate level of detail.  The LOD level is written to the chunk slot's
// first padding field so downstream culling and rendering can use it.
//
// LOD 0 = full detail, LOD 1 = half, LOD 2 = quarter, etc.
// LOD distances are controlled via the uniform.

struct LodUniforms {
    camera_position: vec4<f32>,
    // LOD distance thresholds: lod_distances[i] = max distance for LOD i.
    // Chunks beyond lod_distances[3] are culled entirely.
    lod_distances: vec4<f32>,
    chunk_count: u32,
    max_lod: u32,
    _pad0: u32,
    _pad1: u32,
};

struct ChunkDrawSlot {
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    lod_level: u32,
    flags: u32,
};

@group(0) @binding(0) var<uniform> uniforms: LodUniforms;
@group(0) @binding(1) var<storage, read_write> chunk_slots: array<ChunkDrawSlot>;

const CHUNK_SIZE: f32 = 16.0;

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot_index = id.x;
    if slot_index >= uniforms.chunk_count {
        return;
    }

    let slot = chunk_slots[slot_index];
    if slot.index_count == 0u {
        return;
    }

    // Compute chunk center
    let chunk_min = vec3<f32>(f32(slot.chunk_x), f32(slot.chunk_y), f32(slot.chunk_z)) * CHUNK_SIZE;
    let chunk_center = chunk_min + vec3<f32>(CHUNK_SIZE * 0.5);

    // Distance from camera to chunk center
    let offset = chunk_center - uniforms.camera_position.xyz;
    let dist = length(offset);

    // Select LOD based on distance thresholds
    var lod = uniforms.max_lod;
    if dist < uniforms.lod_distances.x {
        lod = 0u;
    } else if dist < uniforms.lod_distances.y {
        lod = 1u;
    } else if dist < uniforms.lod_distances.z {
        lod = 2u;
    } else if dist < uniforms.lod_distances.w {
        lod = 3u;
    }
    // Chunks beyond max distance get max_lod which the cull pass can skip if desired.

    chunk_slots[slot_index].lod_level = lod;
}
