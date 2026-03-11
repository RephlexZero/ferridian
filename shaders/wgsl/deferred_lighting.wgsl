// Deferred lighting pass — fullscreen triangle reads G-buffer, applies
// Cook-Torrance PBR lighting with ACES tonemapping.  Feature-matched with
// the Rust-GPU SPIR-V deferred_lighting entry points.

const PI: f32 = 3.14159265359;

struct LightingUniforms {
    light_direction: vec4<f32>,
    light_color: vec4<f32>,
    camera_position: vec4<f32>,
    inv_view_projection: mat4x4<f32>,
    shadow_view_projection: mat4x4<f32>,
    ambient_color: vec4<f32>,
};

@group(0) @binding(0) var albedo_tex: texture_2d<f32>;
@group(0) @binding(1) var normal_tex: texture_2d<f32>;
@group(0) @binding(2) var material_tex: texture_2d<f32>;
@group(0) @binding(3) var depth_tex: texture_depth_2d;
@group(0) @binding(4) var shadow_tex: texture_depth_2d;
@group(0) @binding(5) var tex_sampler: sampler;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;

@group(1) @binding(0) var<uniform> lighting: LightingUniforms;

struct VertexOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// Fullscreen triangle without vertex buffer — 3 vertices cover the screen.
@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VertexOutput {
    var out: VertexOutput;
    let x = f32(i32(idx) / 2) * 4.0 - 1.0;
    let y = f32(i32(idx) % 2) * 4.0 - 1.0;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    out.uv = vec2<f32>((x + 1.0) * 0.5, 1.0 - (y + 1.0) * 0.5);
    return out;
}

// ---------------------------------------------------------------------------
// PBR helpers — matches Rust-GPU lighting module
// ---------------------------------------------------------------------------

fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
    let a = roughness * roughness;
    let a2 = a * a;
    let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
    return a2 / (PI * denom * denom + 0.0001);
}

fn geometry_schlick_ggx(n_dot: f32, k: f32) -> f32 {
    return n_dot / (n_dot * (1.0 - k) + k);
}

fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
    let r = roughness + 1.0;
    let k = (r * r) / 8.0;
    return geometry_schlick_ggx(n_dot_v, k) * geometry_schlick_ggx(n_dot_l, k);
}

fn fresnel_schlick(cos_theta: f32, f0: vec3<f32>) -> vec3<f32> {
    let t = clamp(1.0 - cos_theta, 0.0, 1.0);
    let t2 = t * t;
    let t5 = t2 * t2 * t;
    return f0 + (vec3<f32>(1.0) - f0) * t5;
}

fn cook_torrance_brdf(
    n: vec3<f32>,
    v: vec3<f32>,
    l: vec3<f32>,
    albedo: vec3<f32>,
    metallic: f32,
    roughness: f32,
) -> vec3<f32> {
    let h = normalize(v + l);
    let n_dot_v = max(dot(n, v), 0.0001);
    let n_dot_l = max(dot(n, l), 0.0);
    let n_dot_h = max(dot(n, h), 0.0);
    let h_dot_v = max(dot(h, v), 0.0);

    let f0 = mix(vec3<f32>(0.04), albedo, metallic);

    let d = distribution_ggx(n_dot_h, roughness);
    let g = geometry_smith(n_dot_v, n_dot_l, roughness);
    let f = fresnel_schlick(h_dot_v, f0);

    let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
    let spec_scale = d * g / denominator;

    let k_d = (vec3<f32>(1.0) - f) * (1.0 - metallic);

    return (k_d * albedo / PI + f * spec_scale) * n_dot_l;
}

fn aces_tonemap(c: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let cc = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((c * (a * c + b)) / (c * (cc * c + d) + e), vec3<f32>(0.0), vec3<f32>(1.0));
}

// ---------------------------------------------------------------------------
// Normal encoding — octahedral, matching Rust-GPU
// ---------------------------------------------------------------------------

fn decode_normal_octahedral(enc: vec2<f32>) -> vec3<f32> {
    let ox = enc.x * 2.0 - 1.0;
    let oz = enc.y * 2.0 - 1.0;
    let ny = 1.0 - abs(ox) - abs(oz);
    var nx = ox;
    var nz = oz;
    if ny < 0.0 {
        nx = (1.0 - abs(oz)) * sign(ox);
        nz = (1.0 - abs(ox)) * sign(oz);
    }
    return normalize(vec3<f32>(nx, ny, nz));
}

// ---------------------------------------------------------------------------
// Position reconstruction and shadow mapping
// ---------------------------------------------------------------------------

fn reconstruct_position(uv: vec2<f32>, depth: f32, inv_vp: mat4x4<f32>) -> vec3<f32> {
    let ndc = vec4<f32>(uv * 2.0 - 1.0, depth, 1.0);
    let world_h = inv_vp * ndc;
    return world_h.xyz / world_h.w;
}

fn pcf_shadow(shadow_coord: vec3<f32>, bias: f32) -> f32 {
    let depth = shadow_coord.z - bias;
    let uv = shadow_coord.xy;

    // 4-tap PCF
    let texel = 1.0 / 2048.0;
    var shadow = 0.0;
    shadow += textureSampleCompare(shadow_tex, shadow_sampler, uv + vec2<f32>(-texel, -texel), depth);
    shadow += textureSampleCompare(shadow_tex, shadow_sampler, uv + vec2<f32>( texel, -texel), depth);
    shadow += textureSampleCompare(shadow_tex, shadow_sampler, uv + vec2<f32>(-texel,  texel), depth);
    shadow += textureSampleCompare(shadow_tex, shadow_sampler, uv + vec2<f32>( texel,  texel), depth);
    return shadow * 0.25;
}

// ---------------------------------------------------------------------------
// Fragment
// ---------------------------------------------------------------------------

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let uv = input.uv;
    let tex_size = textureDimensions(albedo_tex);
    let coord = vec2<i32>(vec2<f32>(tex_size) * uv);

    let albedo = textureLoad(albedo_tex, coord, 0);
    let normal_raw = textureLoad(normal_tex, coord, 0);
    let material = textureLoad(material_tex, coord, 0);
    let depth = textureLoad(depth_tex, coord, 0);

    // Discard sky fragments (depth == 1.0)
    if depth >= 1.0 {
        return vec4<f32>(0.04, 0.08, 0.11, 1.0);
    }

    // Decode normal from octahedral encoding
    let normal = decode_normal_octahedral(normal_raw.xy);

    // Material: r=roughness, g=metallic (matches gbuffer_fill output)
    let roughness = normal_raw.z;
    let metallic = normal_raw.w;

    // Reconstruct world-space position from depth
    let world_pos = reconstruct_position(uv, depth, lighting.inv_view_projection);

    // Directional light
    let light_dir = normalize(lighting.light_direction.xyz);
    let sun_color = lighting.light_color.rgb;

    // Shadow mapping
    let shadow_pos = lighting.shadow_view_projection * vec4<f32>(world_pos, 1.0);
    let shadow_ndc = shadow_pos.xyz / shadow_pos.w;
    let shadow_uv = vec3<f32>(
        shadow_ndc.x * 0.5 + 0.5,
        1.0 - (shadow_ndc.y * 0.5 + 0.5),
        shadow_ndc.z,
    );
    var shadow_factor = 1.0;
    if shadow_uv.x >= 0.0 && shadow_uv.x <= 1.0 && shadow_uv.y >= 0.0 && shadow_uv.y <= 1.0 {
        shadow_factor = pcf_shadow(shadow_uv, 0.002);
    }

    // Cook-Torrance PBR BRDF
    let view_dir = normalize(lighting.camera_position.xyz - world_pos);
    let brdf = cook_torrance_brdf(normal, view_dir, light_dir, albedo.rgb, metallic, roughness);

    // Combine
    let lit = lighting.ambient_color.rgb * albedo.rgb + brdf * sun_color * shadow_factor;

    // ACES tonemapping
    let mapped = aces_tonemap(lit);

    return vec4<f32>(mapped, 1.0);
}
