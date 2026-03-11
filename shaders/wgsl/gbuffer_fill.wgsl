// G-buffer fill pass — writes albedo, normal (octahedral), roughness, and
// metallic to MRT.  Feature-matched with Rust-GPU gbuffer_fill entry points.
// Uses octahedral normal encoding so deferred_lighting can decode with the
// same decode_normal_octahedral function.

struct SceneUniform {
    view_projection: mat4x4<f32>,
    model: mat4x4<f32>,
    light_direction: vec4<f32>,
    tint_and_time: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> uniforms: SceneUniform;

struct VertexInput {
    @location(0) packed_position: u32,
    @location(1) packed_normal: u32,
    @location(2) packed_color: u32,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_normal: vec3<f32>,
    @location(1) color: vec3<f32>,
    @location(2) world_position: vec3<f32>,
};

struct GBufferOutput {
    @location(0) albedo: vec4<f32>,
    @location(1) normal_material: vec4<f32>,
    @location(2) material: vec4<f32>,
};

fn unpack_position(packed: u32) -> vec3<f32> {
    let x = f32(packed & 31u);
    let y = f32((packed >> 5u) & 31u);
    let z = f32((packed >> 10u) & 31u);
    return vec3<f32>(x, y, z);
}

fn unpack_normal(index: u32) -> vec3<f32> {
    switch index {
        case 0u: { return vec3<f32>(0.0, 0.0, 1.0); }
        case 1u: { return vec3<f32>(0.0, 0.0, -1.0); }
        case 2u: { return vec3<f32>(1.0, 0.0, 0.0); }
        case 3u: { return vec3<f32>(-1.0, 0.0, 0.0); }
        case 4u: { return vec3<f32>(0.0, 1.0, 0.0); }
        default: { return vec3<f32>(0.0, -1.0, 0.0); }
    }
}

fn unpack_color(packed: u32) -> vec3<f32> {
    let r = f32(packed & 255u) / 255.0;
    let g = f32((packed >> 8u) & 255u) / 255.0;
    let b = f32((packed >> 16u) & 255u) / 255.0;
    return vec3<f32>(r, g, b);
}

// Octahedral normal encoding: vec3 (unit) → vec2 in [0, 1].
fn encode_normal_octahedral(n: vec3<f32>) -> vec2<f32> {
    let s = abs(n.x) + abs(n.y) + abs(n.z);
    var ox = n.x / s;
    var oz = n.z / s;
    if n.y < 0.0 {
        let sx = sign(ox);
        let sz = sign(oz);
        ox = (1.0 - abs(oz)) * sx;
        oz = (1.0 - abs(ox)) * sz;
    }
    return vec2<f32>(ox * 0.5 + 0.5, oz * 0.5 + 0.5);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let position = unpack_position(input.packed_position);
    let normal = unpack_normal(input.packed_normal);
    let color = unpack_color(input.packed_color);
    let world_position = uniforms.model * vec4<f32>(position, 1.0);
    let world_normal = normalize((uniforms.model * vec4<f32>(normal, 0.0)).xyz);

    var output: VertexOutput;
    output.clip_position = uniforms.view_projection * world_position;
    output.world_normal = world_normal;
    output.color = color;
    output.world_position = world_position.xyz;
    return output;
}

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    var out: GBufferOutput;
    out.albedo = vec4<f32>(input.color * uniforms.tint_and_time.xyz, 1.0);
    // Encode normal + roughness/metallic into a single render target
    let enc = encode_normal_octahedral(input.world_normal);
    out.normal_material = vec4<f32>(enc.x, enc.y, 0.8, 0.0); // roughness=0.8, metallic=0.0
    // Additional material channel (reserved for future use)
    out.material = vec4<f32>(0.8, 0.0, 0.0, 1.0);
    return out;
}
