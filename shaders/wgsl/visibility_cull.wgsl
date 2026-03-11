// Compute shader for GPU-driven frustum and occlusion culling.
// Reads ChunkDrawSlot entries, tests each chunk's AABB against the view frustum
// and optionally the Hi-Z depth pyramid, then writes surviving chunks as
// IndirectDrawCommand entries into an indirect draw buffer.
//
// Dispatch: ceil(chunk_count / 64) workgroups of 64 threads each.

struct CullUniforms {
    view_projection: mat4x4<f32>,
    // Six frustum planes (xyz = normal, w = distance) packed as rows.
    frustum_plane_0: vec4<f32>,
    frustum_plane_1: vec4<f32>,
    frustum_plane_2: vec4<f32>,
    frustum_plane_3: vec4<f32>,
    frustum_plane_4: vec4<f32>,
    frustum_plane_5: vec4<f32>,
    chunk_count: u32,
    enable_occlusion: u32,
    hiz_width: f32,
    hiz_height: f32,
};

struct ChunkDrawSlot {
    chunk_x: i32,
    chunk_y: i32,
    chunk_z: i32,
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    _pad0: u32,
    _pad1: u32,
};

struct IndirectDrawCommand {
    index_count: u32,
    instance_count: u32,
    first_index: u32,
    base_vertex: i32,
    first_instance: u32,
};

// Atomic counter for the number of visible draw commands.
struct DrawCount {
    count: atomic<u32>,
};

@group(0) @binding(0) var<uniform> uniforms: CullUniforms;
@group(0) @binding(1) var<storage, read> chunk_slots: array<ChunkDrawSlot>;
@group(0) @binding(2) var<storage, read_write> indirect_draws: array<IndirectDrawCommand>;
@group(0) @binding(3) var<storage, read_write> draw_count: DrawCount;

@group(1) @binding(0) var hiz_texture: texture_2d<f32>;
@group(1) @binding(1) var hiz_sampler: sampler;

const CHUNK_SIZE: f32 = 16.0;

// Test an AABB against a single frustum plane.
// Returns true if the AABB is entirely behind the plane (culled).
fn aabb_behind_plane(aabb_min: vec3<f32>, aabb_max: vec3<f32>, plane: vec4<f32>) -> bool {
    // Compute the AABB vertex most in the direction of the plane normal.
    let p = vec3<f32>(
        select(aabb_min.x, aabb_max.x, plane.x >= 0.0),
        select(aabb_min.y, aabb_max.y, plane.y >= 0.0),
        select(aabb_min.z, aabb_max.z, plane.z >= 0.0),
    );
    return dot(plane.xyz, p) + plane.w < 0.0;
}

// Frustum test: returns true if the AABB should be culled.
fn frustum_cull(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    if aabb_behind_plane(aabb_min, aabb_max, uniforms.frustum_plane_0) { return true; }
    if aabb_behind_plane(aabb_min, aabb_max, uniforms.frustum_plane_1) { return true; }
    if aabb_behind_plane(aabb_min, aabb_max, uniforms.frustum_plane_2) { return true; }
    if aabb_behind_plane(aabb_min, aabb_max, uniforms.frustum_plane_3) { return true; }
    if aabb_behind_plane(aabb_min, aabb_max, uniforms.frustum_plane_4) { return true; }
    if aabb_behind_plane(aabb_min, aabb_max, uniforms.frustum_plane_5) { return true; }
    return false;
}

// Hi-Z occlusion test: returns true if the AABB is occluded.
fn hiz_occlusion_cull(aabb_min: vec3<f32>, aabb_max: vec3<f32>) -> bool {
    // Project all 8 AABB corners to clip space, then compute screen AABB.
    var screen_min = vec2<f32>(1.0, 1.0);
    var screen_max = vec2<f32>(-1.0, -1.0);
    var max_depth: f32 = 0.0;

    for (var i = 0u; i < 8u; i++) {
        let corner = vec3<f32>(
            select(aabb_min.x, aabb_max.x, (i & 1u) != 0u),
            select(aabb_min.y, aabb_max.y, (i & 2u) != 0u),
            select(aabb_min.z, aabb_max.z, (i & 4u) != 0u),
        );
        let clip = uniforms.view_projection * vec4<f32>(corner, 1.0);
        if clip.w <= 0.0 {
            // Behind the camera — conservatively mark as visible
            return false;
        }
        let ndc = clip.xyz / clip.w;
        screen_min = min(screen_min, ndc.xy);
        screen_max = max(screen_max, ndc.xy);
        max_depth = max(max_depth, ndc.z);
    }

    // Convert NDC [-1, 1] to UV [0, 1]
    let uv_min = screen_min * 0.5 + 0.5;
    let uv_max = screen_max * 0.5 + 0.5;

    // Pick the mip level based on the screen-space extent
    let extent = (uv_max - uv_min) * vec2<f32>(uniforms.hiz_width, uniforms.hiz_height);
    let mip_level = ceil(log2(max(extent.x, extent.y)));
    let mip = u32(clamp(mip_level, 0.0, 10.0));

    // Sample the Hi-Z pyramid at the AABB center
    let center_uv = (uv_min + uv_max) * 0.5;
    let hiz_depth = textureSampleLevel(hiz_texture, hiz_sampler, center_uv, f32(mip)).x;

    // If the chunk's closest depth is farther than the Hi-Z depth, it's occluded.
    return max_depth > hiz_depth;
}

@compute @workgroup_size(64, 1, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let slot_index = id.x;
    if slot_index >= uniforms.chunk_count {
        return;
    }

    let slot = chunk_slots[slot_index];
    // Skip empty slots
    if slot.index_count == 0u {
        return;
    }

    let aabb_min = vec3<f32>(f32(slot.chunk_x), f32(slot.chunk_y), f32(slot.chunk_z)) * CHUNK_SIZE;
    let aabb_max = aabb_min + vec3<f32>(CHUNK_SIZE, CHUNK_SIZE, CHUNK_SIZE);

    // Frustum cull
    if frustum_cull(aabb_min, aabb_max) {
        return;
    }

    // Optional Hi-Z occlusion cull
    if uniforms.enable_occlusion != 0u {
        if hiz_occlusion_cull(aabb_min, aabb_max) {
            return;
        }
    }

    // This chunk is visible — append an indirect draw command.
    let out_index = atomicAdd(&draw_count.count, 1u);
    indirect_draws[out_index].index_count = slot.index_count;
    indirect_draws[out_index].instance_count = 1u;
    indirect_draws[out_index].first_index = slot.first_index;
    indirect_draws[out_index].base_vertex = slot.base_vertex;
    indirect_draws[out_index].first_instance = 0u;
}
