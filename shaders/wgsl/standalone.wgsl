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

fn unpack_position(packed: u32) -> vec3<f32> {
    let x = f32(packed & 31u);
    let y = f32((packed >> 5u) & 31u);
    let z = f32((packed >> 10u) & 31u);
    return vec3<f32>(x, y, z);
}

fn unpack_normal(index: u32) -> vec3<f32> {
    switch index {
        case 0u: {
            return vec3<f32>(0.0, 0.0, 1.0);
        }
        case 1u: {
            return vec3<f32>(0.0, 0.0, -1.0);
        }
        case 2u: {
            return vec3<f32>(1.0, 0.0, 0.0);
        }
        case 3u: {
            return vec3<f32>(-1.0, 0.0, 0.0);
        }
        case 4u: {
            return vec3<f32>(0.0, 1.0, 0.0);
        }
        default: {
            return vec3<f32>(0.0, -1.0, 0.0);
        }
    }
}

fn unpack_color(packed: u32) -> vec3<f32> {
    let r = f32(packed & 255u) / 255.0;
    let g = f32((packed >> 8u) & 255u) / 255.0;
    let b = f32((packed >> 16u) & 255u) / 255.0;
    return vec3<f32>(r, g, b);
}

// Reconstruct world-space position from depth buffer + inverse view-projection.
// Used by the deferred lighting pass to evaluate lights without storing
// explicit world positions in the G-buffer.
fn reconstruct_position_from_depth(
    uv: vec2<f32>,
    depth: f32,
    inv_view_projection: mat4x4<f32>,
) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world_h = inv_view_projection * ndc;
    return world_h.xyz / world_h.w;
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
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let light_direction = normalize(uniforms.light_direction.xyz);
    let lambert = max(dot(input.world_normal, light_direction), 0.18);
    let pulse = 0.88 + 0.12 * sin(uniforms.tint_and_time.w * 1.6 + input.world_position.y * 2.0);
    let shaded_color = input.color * uniforms.tint_and_time.xyz * lambert * pulse;
    return vec4<f32>(shaded_color, 1.0);
}