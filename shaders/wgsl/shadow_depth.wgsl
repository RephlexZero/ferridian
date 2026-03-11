// Shadow depth pass — renders geometry from light perspective, depth-only output.

struct ShadowUniforms {
    light_view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
};

@group(0) @binding(0)
var<uniform> shadow: ShadowUniforms;

struct VertexInput {
    @location(0) packed_position: u32,
    @location(1) packed_normal: u32,
    @location(2) packed_color: u32,
};

fn unpack_position(packed: u32) -> vec3<f32> {
    let x = f32(packed & 31u);
    let y = f32((packed >> 5u) & 31u);
    let z = f32((packed >> 10u) & 31u);
    return vec3<f32>(x, y, z);
}

@vertex
fn vs_main(input: VertexInput) -> @builtin(position) vec4<f32> {
    let position = unpack_position(input.packed_position);
    let world_position = shadow.model * vec4<f32>(position, 1.0);
    return shadow.light_view_projection * world_position;
}
