// Translucent forward pass — renders alpha-blended geometry (water, glass,
// particles).  Feature-matched with the Rust-GPU SPIR-V translucent entry
// points: wave displacement on vertex, Blinn-Phong specular^64, ACES tonemap.

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

fn aces_tonemap(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let cc = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let position = unpack_position(input.packed_position);
    let normal = unpack_normal(input.packed_normal);
    let color = unpack_color(input.packed_color);

    // Wave displacement (matches Rust-GPU translucent_vs)
    let time = uniforms.tint_and_time.w;
    let wave_y = position.y
        + sin(position.x * 0.8 + time * 2.0) * 0.05
        + cos(position.z * 0.6 + time * 1.5) * 0.03;

    let displaced = vec3<f32>(position.x, wave_y, position.z);
    let world_position = uniforms.model * vec4<f32>(displaced, 1.0);
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
    let water_color = vec3<f32>(0.1, 0.3, 0.5);
    let sun_dir = normalize(uniforms.light_direction.xyz);
    let normal = normalize(input.world_normal);

    // Diffuse
    let n_dot_l = max(dot(normal, sun_dir), 0.0);

    // Blinn-Phong specular (^64 to match Rust-GPU)
    // Approximate camera position from tint_and_time — use light as fallback
    let view_dir = normalize(-input.world_position);
    let reflect_dir = reflect(-sun_dir, normal);
    var spec = max(dot(view_dir, reflect_dir), 0.0);
    // pow(spec, 64) via repeated squaring
    spec = spec * spec; // ^2
    spec = spec * spec; // ^4
    spec = spec * spec; // ^8
    spec = spec * spec; // ^16
    spec = spec * spec; // ^32
    spec = spec * spec; // ^64

    let color = water_color * (0.3 + 0.7 * n_dot_l) + vec3<f32>(spec * 0.5);
    let mapped = aces_tonemap(color);

    return vec4<f32>(mapped, 0.55);
}
