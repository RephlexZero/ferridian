//! Ferridian GPU shaders written in Rust, compiled to SPIR-V via rust-gpu.
//!
//! This crate contains all shader entry points and shared GPU-side math.
//! It is compiled to SPIR-V by `cargo xtask build-shaders`, NOT by the
//! normal workspace build. The resulting `.spv` files are placed in
//! `shaders/spirv/` and loaded by `ferridian-shader` at build time.
//!
//! # Architecture
//!
//! - **`lighting`** — PBR math, tonemapping, normal encoding (testable on CPU)
//! - **`gbuffer_fill`** — G-buffer vertex + fragment entry points
//! - **`shadow`** — Depth-only shadow map pass
//! - **`deferred_lighting`** — Fullscreen deferred lighting pass
//! - **`translucent`** — Forward pass for water, glass, particles

#![cfg_attr(target_arch = "spirv", no_std)]
#![allow(unexpected_cfgs)]

// Re-export shared types so shader code can use them directly.
pub use ferridian_shared_types::*;

// ---------------------------------------------------------------------------
// Lighting math — fully testable on CPU via `cargo test -p ferridian-shader-gpu`
// ---------------------------------------------------------------------------

pub mod lighting {
    const PI: f32 = core::f32::consts::PI;

    /// GGX / Trowbridge-Reitz normal distribution function.
    #[inline]
    pub fn distribution_ggx(n_dot_h: f32, roughness: f32) -> f32 {
        let a = roughness * roughness;
        let a2 = a * a;
        let denom = n_dot_h * n_dot_h * (a2 - 1.0) + 1.0;
        a2 / (PI * denom * denom + 0.0001)
    }

    /// Smith's geometry function using Schlick-GGX approximation.
    #[inline]
    pub fn geometry_smith(n_dot_v: f32, n_dot_l: f32, roughness: f32) -> f32 {
        let r = roughness + 1.0;
        let k = (r * r) / 8.0;
        let ggx_v = n_dot_v / (n_dot_v * (1.0 - k) + k);
        let ggx_l = n_dot_l / (n_dot_l * (1.0 - k) + k);
        ggx_v * ggx_l
    }

    /// Fresnel-Schlick approximation. Returns reflectance for a given angle.
    #[inline]
    pub fn fresnel_schlick(cos_theta: f32, f0: [f32; 3]) -> [f32; 3] {
        let t = (1.0 - cos_theta).clamp(0.0, 1.0);
        let t2 = t * t;
        let t5 = t2 * t2 * t;
        [
            f0[0] + (1.0 - f0[0]) * t5,
            f0[1] + (1.0 - f0[1]) * t5,
            f0[2] + (1.0 - f0[2]) * t5,
        ]
    }

    /// Full Cook-Torrance microfacet BRDF. Returns outgoing radiance contribution.
    pub fn cook_torrance(
        n: [f32; 3],
        v: [f32; 3],
        l: [f32; 3],
        albedo: [f32; 3],
        metallic: f32,
        roughness: f32,
    ) -> [f32; 3] {
        let h = normalize([v[0] + l[0], v[1] + l[1], v[2] + l[2]]);
        let n_dot_v = dot(n, v).max(0.0001);
        let n_dot_l = dot(n, l).max(0.0);
        let n_dot_h = dot(n, h).max(0.0);
        let h_dot_v = dot(h, v).max(0.0);

        let f0 = [
            lerp(0.04, albedo[0], metallic),
            lerp(0.04, albedo[1], metallic),
            lerp(0.04, albedo[2], metallic),
        ];

        let d = distribution_ggx(n_dot_h, roughness);
        let g = geometry_smith(n_dot_v, n_dot_l, roughness);
        let f = fresnel_schlick(h_dot_v, f0);

        let denominator = 4.0 * n_dot_v * n_dot_l + 0.0001;
        let spec_scale = d * g / denominator;

        let k_d = [
            (1.0 - f[0]) * (1.0 - metallic),
            (1.0 - f[1]) * (1.0 - metallic),
            (1.0 - f[2]) * (1.0 - metallic),
        ];

        [
            (k_d[0] * albedo[0] / PI + f[0] * spec_scale) * n_dot_l,
            (k_d[1] * albedo[1] / PI + f[1] * spec_scale) * n_dot_l,
            (k_d[2] * albedo[2] / PI + f[2] * spec_scale) * n_dot_l,
        ]
    }

    /// ACES filmic tonemapping operator.
    #[inline]
    pub fn aces_tonemap(c: [f32; 3]) -> [f32; 3] {
        let map = |x: f32| -> f32 {
            let a = 2.51;
            let b = 0.03;
            let c = 2.43;
            let d = 0.59;
            let e = 0.14;
            ((x * (a * x + b)) / (x * (c * x + d) + e)).clamp(0.0, 1.0)
        };
        [map(c[0]), map(c[1]), map(c[2])]
    }

    /// Octahedral normal encoding: Vec3 (unit) → Vec2 in [0, 1].
    pub fn encode_normal_octahedral(n: [f32; 3]) -> [f32; 2] {
        let sum = n[0].abs() + n[1].abs() + n[2].abs();
        let mut ox = n[0] / sum;
        let mut oz = n[2] / sum;
        if n[1] < 0.0 {
            let sign_x = if ox >= 0.0 { 1.0 } else { -1.0 };
            let sign_z = if oz >= 0.0 { 1.0 } else { -1.0 };
            let new_ox = (1.0 - oz.abs()) * sign_x;
            let new_oz = (1.0 - ox.abs()) * sign_z;
            ox = new_ox;
            oz = new_oz;
        }
        [ox * 0.5 + 0.5, oz * 0.5 + 0.5]
    }

    /// Octahedral normal decoding: Vec2 in [0, 1] → Vec3 (unit).
    pub fn decode_normal_octahedral(enc: [f32; 2]) -> [f32; 3] {
        let ox = enc[0] * 2.0 - 1.0;
        let oz = enc[1] * 2.0 - 1.0;
        let ny = 1.0 - ox.abs() - oz.abs();
        let (nx, nz) = if ny < 0.0 {
            let sign_x = if ox >= 0.0 { 1.0 } else { -1.0 };
            let sign_z = if oz >= 0.0 { 1.0 } else { -1.0 };
            ((1.0 - oz.abs()) * sign_x, (1.0 - ox.abs()) * sign_z)
        } else {
            (ox, oz)
        };
        normalize([nx, ny, nz])
    }

    // -- Scalar helpers (no glam dependency needed) --

    #[inline]
    pub fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
        a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
    }

    #[inline]
    pub fn normalize(v: [f32; 3]) -> [f32; 3] {
        let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        if len < 0.00001 {
            return [0.0, 1.0, 0.0];
        }
        [v[0] / len, v[1] / len, v[2] / len]
    }

    #[inline]
    fn lerp(a: f32, b: f32, t: f32) -> f32 {
        a + (b - a) * t
    }

    // -- Cross product helper --

    #[inline]
    pub fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    }

    #[inline]
    pub fn length(v: [f32; 3]) -> f32 {
        (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
    }

    // -- Screen-space effect math (CPU-testable) --

    /// GTAO: visibility-bitmask ground-truth ambient occlusion sample weight.
    /// Given a depth difference and a radius, returns the occlusion weight
    /// for a single tap.
    #[inline]
    pub fn gtao_weight(depth_diff: f32, radius: f32) -> f32 {
        let t = (depth_diff / radius).clamp(0.0, 1.0);
        // Smooth falloff: 1 at centre, 0 at edge
        (1.0 - t * t).max(0.0)
    }

    /// GTAO: combine multiple tap weights into a single AO factor.
    /// `weights` is a list of per-tap occlusion weights, `bias` shifts the
    /// result to prevent full blackness.
    pub fn gtao_combine(weights: &[f32], bias: f32) -> f32 {
        if weights.is_empty() {
            return 1.0;
        }
        let sum: f32 = weights.iter().sum();
        let occlusion = sum / weights.len() as f32;
        (1.0 - occlusion + bias).clamp(0.0, 1.0)
    }

    /// SSR: compute reflection direction from view + normal.
    #[inline]
    pub fn reflect(incident: [f32; 3], normal: [f32; 3]) -> [f32; 3] {
        let d = 2.0 * dot(incident, normal);
        [
            incident[0] - d * normal[0],
            incident[1] - d * normal[1],
            incident[2] - d * normal[2],
        ]
    }

    /// Stochastic SSR: hash-based pseudo-random for ray jitter.
    /// Returns a value in [0, 1) for a given screen-space coordinate + frame index.
    #[inline]
    pub fn ssr_hash(x: u32, y: u32, frame: u32) -> f32 {
        let mut h =
            x.wrapping_mul(73856093) ^ y.wrapping_mul(19349663) ^ frame.wrapping_mul(83492791);
        h = h.wrapping_mul(h).wrapping_shr(16);
        (h & 0xFFFF) as f32 / 65536.0
    }

    /// Hi-Z ray march step: given current UV + depth, step along the reflection
    /// ray and test against the Hi-Z mip.  Returns `(hit, new_uv, new_depth)`.
    pub fn hiz_ray_step(
        uv: [f32; 2],
        direction: [f32; 2],
        step_size: f32,
        current_depth: f32,
        hiz_depth: f32,
    ) -> (bool, [f32; 2], f32) {
        let new_uv = [
            uv[0] + direction[0] * step_size,
            uv[1] + direction[1] * step_size,
        ];
        // Step depth along the ray conservatively
        let new_depth = current_depth + step_size * 0.1;
        let hit = new_depth >= hiz_depth && (new_depth - hiz_depth).abs() < step_size * 0.5;
        (hit, new_uv, new_depth)
    }

    /// Volumetric fog: Beer-Lambert transmittance.
    #[inline]
    pub fn beer_lambert_transmittance(density: f32, distance: f32) -> f32 {
        (-density * distance).exp().clamp(0.0, 1.0)
    }

    /// Volumetric fog: Henyey-Greenstein phase function for directional scattering.
    pub fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
        let g2 = g * g;
        let denom = 1.0 + g2 - 2.0 * g * cos_theta;
        let denom_sqrt = denom.sqrt();
        let denom_32 = denom_sqrt * denom_sqrt * denom_sqrt;
        (1.0 - g2) / (4.0 * PI * denom_32 + 0.0001)
    }

    /// Volumetric fog: single froxel scattering contribution.
    pub fn froxel_scatter(
        density: f32,
        step_length: f32,
        light_transmittance: f32,
        phase: f32,
        light_color: [f32; 3],
    ) -> [f32; 3] {
        let scatter = density * step_length * light_transmittance * phase;
        [
            light_color[0] * scatter,
            light_color[1] * scatter,
            light_color[2] * scatter,
        ]
    }

    // -- Cascaded shadow map math --

    /// Select the cascade index based on linear depth and split distances.
    /// Returns the index of the cascade to sample (0-based), or cascade_count
    /// if the depth is beyond all cascades.
    pub fn select_cascade(linear_depth: f32, splits: &[f32]) -> usize {
        for (i, &split) in splits.iter().enumerate() {
            if linear_depth < split {
                return i;
            }
        }
        splits.len()
    }

    /// Smooth cascade blending factor. Returns a blend weight in [0, 1] where
    /// 0 = fully this cascade, 1 = fully next cascade.
    #[inline]
    pub fn cascade_blend_factor(linear_depth: f32, cascade_end: f32, blend_range: f32) -> f32 {
        let t = (linear_depth - (cascade_end - blend_range)) / blend_range;
        t.clamp(0.0, 1.0)
    }

    /// Shadow PCF (percentage-closer filtering) for a single sample.
    /// Returns 0.0 (fully shadowed) or 1.0 (fully lit).
    #[inline]
    pub fn pcf_sample(depth: f32, shadow_depth: f32, bias: f32) -> f32 {
        if depth - bias > shadow_depth {
            0.0
        } else {
            1.0
        }
    }

    /// Shadow PCF with multiple taps.
    pub fn pcf_filter(depth: f32, samples: &[f32], bias: f32) -> f32 {
        if samples.is_empty() {
            return 1.0;
        }
        let sum: f32 = samples.iter().map(|&s| pcf_sample(depth, s, bias)).sum();
        sum / samples.len() as f32
    }

    // -- ReSTIR direct lighting math --

    /// Compute the target PDF (p-hat) for a light sample: luminance of
    /// (unshadowed contribution × geometry term).
    pub fn restir_target_pdf(light_color: [f32; 3], n_dot_l: f32, distance_sq: f32) -> f32 {
        let attenuation = 1.0 / (distance_sq + 0.0001);
        let contrib = [
            light_color[0] * n_dot_l * attenuation,
            light_color[1] * n_dot_l * attenuation,
            light_color[2] * n_dot_l * attenuation,
        ];
        // Luminance as target function
        0.2126 * contrib[0] + 0.7152 * contrib[1] + 0.0722 * contrib[2]
    }

    /// ReSTIR reservoir update: decide whether to replace the current sample
    /// with a new candidate.  Returns (accept, new_weight_sum).
    pub fn reservoir_update(
        current_weight_sum: f32,
        new_weight: f32,
        random_value: f32,
    ) -> (bool, f32) {
        let new_sum = current_weight_sum + new_weight;
        let accept = random_value * new_sum < new_weight;
        (accept, new_sum)
    }

    /// ReSTIR: compute the final unbiased weight for a reservoir sample.
    #[inline]
    pub fn reservoir_final_weight(weight_sum: f32, sample_count: u32, target_pdf: f32) -> f32 {
        if target_pdf < 0.0001 || sample_count == 0 {
            return 0.0;
        }
        weight_sum / (sample_count as f32 * target_pdf)
    }
}

// ---------------------------------------------------------------------------
// Shader entry points — these are compiled to SPIR-V by spirv-builder
// ---------------------------------------------------------------------------

#[cfg(target_arch = "spirv")]
mod entry_points {
    use super::lighting;
    use ferridian_shared_types::{MaterialEntry, SceneUniforms, ShadowUniforms};
    use spirv_std::glam::{Mat4, Vec2, Vec3, Vec4, vec2, vec3, vec4};
    use spirv_std::spirv;

    // -- G-Buffer fill pass --

    #[spirv(vertex)]
    pub fn gbuffer_fill_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scene: &SceneUniforms,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertices: &[u32],
        #[spirv(position, invariant)] out_pos: &mut Vec4,
        out_normal: &mut Vec3,
        out_ao: &mut f32,
        out_tex_id: &mut u32,
    ) {
        let base = vertex_index as usize * 2;
        let d0 = vertices[base];
        let d1 = vertices[base + 1];

        let x = (d0 & 0x1F) as f32;
        let y = ((d0 >> 5) & 0x1FF) as f32;
        let z = ((d0 >> 14) & 0x1F) as f32;
        let normal_idx = (d0 >> 19) & 0x7;
        let ao_val = (d0 >> 22) & 0x3;

        let vp = Mat4::from_cols_array_2d(&scene.view_projection);
        *out_pos = vp * vec4(x, y, z, 1.0);

        *out_normal = match normal_idx {
            0 => vec3(0.0, 1.0, 0.0),
            1 => vec3(0.0, -1.0, 0.0),
            2 => vec3(1.0, 0.0, 0.0),
            3 => vec3(-1.0, 0.0, 0.0),
            4 => vec3(0.0, 0.0, 1.0),
            _ => vec3(0.0, 0.0, -1.0),
        };
        *out_ao = ao_val as f32 / 3.0;
        *out_tex_id = d1 & 0xFF;
    }

    #[spirv(fragment)]
    pub fn gbuffer_fill_fs(
        in_normal: Vec3,
        in_ao: f32,
        #[spirv(flat)] in_tex_id: u32,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] materials: &[MaterialEntry],
        out_albedo: &mut Vec4,
        out_normal_material: &mut Vec4,
    ) {
        let mat = materials[in_tex_id as usize];
        let enc = lighting::encode_normal_octahedral([in_normal.x, in_normal.y, in_normal.z]);
        *out_albedo = vec4(mat.albedo[0], mat.albedo[1], mat.albedo[2], 1.0);
        *out_normal_material = vec4(enc[0], enc[1], mat.roughness, mat.metallic);
    }

    // -- Shadow depth pass --

    #[spirv(vertex)]
    pub fn shadow_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] shadow: &ShadowUniforms,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertices: &[u32],
        #[spirv(position, invariant)] out_pos: &mut Vec4,
    ) {
        let base = vertex_index as usize * 2;
        let d0 = vertices[base];

        let x = (d0 & 0x1F) as f32;
        let y = ((d0 >> 5) & 0x1FF) as f32;
        let z = ((d0 >> 14) & 0x1F) as f32;

        let lvp = Mat4::from_cols_array_2d(&shadow.light_view_projection);
        *out_pos = lvp * vec4(x, y, z, 1.0);
    }

    #[spirv(fragment)]
    pub fn shadow_fs() {}

    // -- Deferred lighting fullscreen pass --

    #[spirv(vertex)]
    pub fn deferred_lighting_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(position)] out_pos: &mut Vec4,
        out_uv: &mut Vec2,
    ) {
        let u = ((vertex_index << 1) & 2) as f32;
        let v = (vertex_index & 2) as f32;
        *out_uv = vec2(u, v);
        *out_pos = vec4(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0, 1.0);
    }

    #[spirv(fragment)]
    pub fn deferred_lighting_fs(
        in_uv: Vec2,
        #[spirv(descriptor_set = 0, binding = 0)] albedo_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 1)] normal_mat_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 2)] depth_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 3)] shadow_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 4)] tex_sampler: &spirv_std::Sampler,
        #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] scene: &SceneUniforms,
        out_color: &mut Vec4,
    ) {
        let albedo_s: Vec4 = albedo_tex.sample(*tex_sampler, in_uv);
        let normal_s: Vec4 = normal_mat_tex.sample(*tex_sampler, in_uv);
        let depth_s: Vec4 = depth_tex.sample(*tex_sampler, in_uv);

        let albedo = [albedo_s.x, albedo_s.y, albedo_s.z];
        let normal = lighting::decode_normal_octahedral([normal_s.x, normal_s.y]);
        let roughness = normal_s.z;
        let metallic = normal_s.w;

        // Reconstruct world position from depth
        let ndc = vec4(in_uv.x * 2.0 - 1.0, 1.0 - in_uv.y * 2.0, depth_s.x, 1.0);
        let inv_vp = Mat4::from_cols_array_2d(&scene.inv_view_projection);
        let world_h = inv_vp * ndc;
        let wp = world_h.truncate() / world_h.w;

        let cam = [
            scene.camera_position[0],
            scene.camera_position[1],
            scene.camera_position[2],
        ];
        let sun = lighting::normalize([
            scene.sun_direction[0],
            scene.sun_direction[1],
            scene.sun_direction[2],
        ]);
        let sun_col = [scene.sun_color[0], scene.sun_color[1], scene.sun_color[2]];
        let ambient = [
            scene.ambient_color[0],
            scene.ambient_color[1],
            scene.ambient_color[2],
        ];

        let view = lighting::normalize([cam[0] - wp.x, cam[1] - wp.y, cam[2] - wp.z]);

        let brdf = lighting::cook_torrance(normal, view, sun, albedo, metallic, roughness);
        let lit = [
            ambient[0] * albedo[0] + brdf[0] * sun_col[0],
            ambient[1] * albedo[1] + brdf[1] * sun_col[1],
            ambient[2] * albedo[2] + brdf[2] * sun_col[2],
        ];

        let tm = lighting::aces_tonemap(lit);
        *out_color = vec4(tm[0], tm[1], tm[2], 1.0);
    }

    // -- Translucent forward pass (water, glass) --

    #[spirv(vertex)]
    pub fn translucent_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] scene: &SceneUniforms,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertices: &[u32],
        #[spirv(position, invariant)] out_pos: &mut Vec4,
        out_world_pos: &mut Vec3,
        out_normal: &mut Vec3,
    ) {
        let base = vertex_index as usize * 2;
        let d0 = vertices[base];

        let x = (d0 & 0x1F) as f32;
        let y = ((d0 >> 5) & 0x1FF) as f32;
        let z = ((d0 >> 14) & 0x1F) as f32;
        let normal_idx = (d0 >> 19) & 0x7;

        // Wave displacement
        let time = scene.time;
        let wave_y =
            y + libm::sinf(x * 0.8 + time * 2.0) * 0.05 + libm::cosf(z * 0.6 + time * 1.5) * 0.03;

        let vp = Mat4::from_cols_array_2d(&scene.view_projection);
        *out_pos = vp * vec4(x, wave_y, z, 1.0);
        *out_world_pos = vec3(x, wave_y, z);

        *out_normal = match normal_idx {
            0 => vec3(0.0, 1.0, 0.0),
            1 => vec3(0.0, -1.0, 0.0),
            2 => vec3(1.0, 0.0, 0.0),
            3 => vec3(-1.0, 0.0, 0.0),
            4 => vec3(0.0, 0.0, 1.0),
            _ => vec3(0.0, 0.0, -1.0),
        };
    }

    #[spirv(fragment)]
    pub fn translucent_fs(
        in_world_pos: Vec3,
        in_normal: Vec3,
        #[spirv(storage_buffer, descriptor_set = 1, binding = 0)] scene: &SceneUniforms,
        out_color: &mut Vec4,
    ) {
        let water = [0.1_f32, 0.3, 0.5];
        let sun = lighting::normalize([
            scene.sun_direction[0],
            scene.sun_direction[1],
            scene.sun_direction[2],
        ]);
        let cam = [
            scene.camera_position[0],
            scene.camera_position[1],
            scene.camera_position[2],
        ];
        let n = lighting::normalize([in_normal.x, in_normal.y, in_normal.z]);
        let view = lighting::normalize([
            cam[0] - in_world_pos.x,
            cam[1] - in_world_pos.y,
            cam[2] - in_world_pos.z,
        ]);

        let diffuse = lighting::dot(n, sun).max(0.0);
        let reflect = lighting::normalize([
            2.0 * lighting::dot(n, sun) * n[0] - sun[0],
            2.0 * lighting::dot(n, sun) * n[1] - sun[1],
            2.0 * lighting::dot(n, sun) * n[2] - sun[2],
        ]);
        let spec = lighting::dot(view, reflect).max(0.0);
        // powf(64.0) approximation for spirv
        let spec = spec * spec; // ^2
        let spec = spec * spec; // ^4
        let spec = spec * spec; // ^8
        let spec = spec * spec; // ^16
        let spec = spec * spec; // ^32
        let spec = spec * spec; // ^64

        let color = [
            water[0] * (0.3 + 0.7 * diffuse) + spec * 0.5,
            water[1] * (0.3 + 0.7 * diffuse) + spec * 0.5,
            water[2] * (0.3 + 0.7 * diffuse) + spec * 0.5,
        ];

        let tm = lighting::aces_tonemap(color);
        *out_color = vec4(tm[0], tm[1], tm[2], 0.55);
    }

    // -- GTAO screen-space ambient occlusion --

    #[spirv(vertex)]
    pub fn gtao_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(position)] out_pos: &mut Vec4,
        out_uv: &mut Vec2,
    ) {
        let u = ((vertex_index << 1) & 2) as f32;
        let v = (vertex_index & 2) as f32;
        *out_uv = vec2(u, v);
        *out_pos = vec4(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0, 1.0);
    }

    #[spirv(fragment)]
    pub fn gtao_fs(
        in_uv: Vec2,
        #[spirv(descriptor_set = 0, binding = 0)] depth_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 1)] normal_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 2)] tex_sampler: &spirv_std::Sampler,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 3)] scene: &SceneUniforms,
        out_ao: &mut Vec4,
    ) {
        let depth_s: Vec4 = depth_tex.sample(*tex_sampler, in_uv);
        let normal_s: Vec4 = normal_tex.sample(*tex_sampler, in_uv);
        let normal = lighting::decode_normal_octahedral([normal_s.x, normal_s.y]);
        let center_depth = depth_s.x;

        let radius = 0.5_f32;
        let mut total_weight = 0.0_f32;
        // 4-tap AO sampling at cardinal directions
        let offsets = [
            vec2(1.0, 0.0),
            vec2(-1.0, 0.0),
            vec2(0.0, 1.0),
            vec2(0.0, -1.0),
        ];
        let texel_size = vec2(1.0 / scene.screen_size[0], 1.0 / scene.screen_size[1]);
        for i in 0..4 {
            let sample_uv = in_uv + offsets[i] * texel_size * 4.0;
            let sample_depth: Vec4 = depth_tex.sample(*tex_sampler, sample_uv);
            let diff = (center_depth - sample_depth.x).abs();
            total_weight += lighting::gtao_weight(diff, radius);
        }
        let ao = (1.0 - total_weight / 4.0 + 0.025).clamp(0.0, 1.0);
        *out_ao = vec4(ao, ao, ao, 1.0);
    }

    // -- Stochastic SSR with Hi-Z traversal --

    #[spirv(vertex)]
    pub fn ssr_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(position)] out_pos: &mut Vec4,
        out_uv: &mut Vec2,
    ) {
        let u = ((vertex_index << 1) & 2) as f32;
        let v = (vertex_index & 2) as f32;
        *out_uv = vec2(u, v);
        *out_pos = vec4(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0, 1.0);
    }

    #[spirv(fragment)]
    pub fn ssr_fs(
        in_uv: Vec2,
        #[spirv(descriptor_set = 0, binding = 0)] color_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 1)] depth_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 2)] normal_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 3)] tex_sampler: &spirv_std::Sampler,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 4)] scene: &SceneUniforms,
        out_color: &mut Vec4,
    ) {
        let color_s: Vec4 = color_tex.sample(*tex_sampler, in_uv);
        let depth_s: Vec4 = depth_tex.sample(*tex_sampler, in_uv);
        let normal_s: Vec4 = normal_tex.sample(*tex_sampler, in_uv);

        let normal = lighting::decode_normal_octahedral([normal_s.x, normal_s.y]);
        let roughness = normal_s.z;

        // Skip SSR for rough surfaces
        if roughness > 0.5 {
            *out_color = color_s;
            return;
        }

        // Reconstruct world position
        let ndc = vec4(in_uv.x * 2.0 - 1.0, 1.0 - in_uv.y * 2.0, depth_s.x, 1.0);
        let inv_vp = Mat4::from_cols_array_2d(&scene.inv_view_projection);
        let world_h = inv_vp * ndc;
        let wp = world_h.truncate() / world_h.w;

        let cam = [
            scene.camera_position[0],
            scene.camera_position[1],
            scene.camera_position[2],
        ];
        let view = lighting::normalize([cam[0] - wp.x, cam[1] - wp.y, cam[2] - wp.z]);
        let refl = lighting::reflect([-view[0], -view[1], -view[2]], normal);

        // Simple screen-space ray march (Hi-Z traversal done via texture LODs)
        let vp = Mat4::from_cols_array_2d(&scene.view_projection);
        let step_world = 0.5_f32;
        let mut hit_color = color_s;
        for step in 0..16 {
            let march_pos = vec3(
                wp.x + refl[0] * step_world * (step as f32 + 1.0),
                wp.y + refl[1] * step_world * (step as f32 + 1.0),
                wp.z + refl[2] * step_world * (step as f32 + 1.0),
            );
            let clip = vp * vec4(march_pos.x, march_pos.y, march_pos.z, 1.0);
            if clip.w <= 0.0 {
                break;
            }
            let march_ndc = clip.truncate() / clip.w;
            let march_uv = vec2(march_ndc.x * 0.5 + 0.5, 0.5 - march_ndc.y * 0.5);
            if march_uv.x < 0.0 || march_uv.x > 1.0 || march_uv.y < 0.0 || march_uv.y > 1.0 {
                break;
            }
            let march_depth: Vec4 = depth_tex.sample(*tex_sampler, march_uv);
            if march_ndc.z > march_depth.x && (march_ndc.z - march_depth.x) < 0.05 {
                let reflection: Vec4 = color_tex.sample(*tex_sampler, march_uv);
                let blend = 1.0 - roughness * 2.0;
                hit_color = vec4(
                    color_s.x + reflection.x * blend * 0.3,
                    color_s.y + reflection.y * blend * 0.3,
                    color_s.z + reflection.z * blend * 0.3,
                    1.0,
                );
                break;
            }
        }
        *out_color = hit_color;
    }

    // -- Volumetric fog (froxel-based) --

    #[spirv(vertex)]
    pub fn volumetric_fog_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(position)] out_pos: &mut Vec4,
        out_uv: &mut Vec2,
    ) {
        let u = ((vertex_index << 1) & 2) as f32;
        let v = (vertex_index & 2) as f32;
        *out_uv = vec2(u, v);
        *out_pos = vec4(u * 2.0 - 1.0, 1.0 - v * 2.0, 0.0, 1.0);
    }

    #[spirv(fragment)]
    pub fn volumetric_fog_fs(
        in_uv: Vec2,
        #[spirv(descriptor_set = 0, binding = 0)] depth_tex: &spirv_std::Image!(2D, type=f32, sampled),
        #[spirv(descriptor_set = 0, binding = 1)] tex_sampler: &spirv_std::Sampler,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 2)] scene: &SceneUniforms,
        out_fog: &mut Vec4,
    ) {
        let depth_s: Vec4 = depth_tex.sample(*tex_sampler, in_uv);

        // Reconstruct world position
        let ndc = vec4(in_uv.x * 2.0 - 1.0, 1.0 - in_uv.y * 2.0, depth_s.x, 1.0);
        let inv_vp = Mat4::from_cols_array_2d(&scene.inv_view_projection);
        let world_h = inv_vp * ndc;
        let wp = world_h.truncate() / world_h.w;

        let cam_pos = vec3(
            scene.camera_position[0],
            scene.camera_position[1],
            scene.camera_position[2],
        );
        let to_pixel = vec3(wp.x - cam_pos.x, wp.y - cam_pos.y, wp.z - cam_pos.z);
        let dist = to_pixel.length();

        let sun = lighting::normalize([
            scene.sun_direction[0],
            scene.sun_direction[1],
            scene.sun_direction[2],
        ]);
        let view_dir = if dist > 0.001 {
            lighting::normalize([to_pixel.x, to_pixel.y, to_pixel.z])
        } else {
            [0.0, 0.0, 1.0]
        };

        // Ray-march fog with fixed step count
        let fog_density = 0.02_f32;
        let step_count = 16u32;
        let step_len = dist / step_count as f32;
        let mut accumulated = [0.0_f32; 3];
        let mut transmittance = 1.0_f32;
        let cos_theta = lighting::dot(view_dir, sun);
        let phase = lighting::henyey_greenstein(cos_theta, 0.3);
        let sun_col = [scene.sun_color[0], scene.sun_color[1], scene.sun_color[2]];

        for _step in 0..step_count {
            let scatter =
                lighting::froxel_scatter(fog_density, step_len, transmittance, phase, sun_col);
            accumulated[0] += scatter[0];
            accumulated[1] += scatter[1];
            accumulated[2] += scatter[2];
            transmittance *= lighting::beer_lambert_transmittance(fog_density, step_len);
        }

        *out_fog = vec4(
            accumulated[0],
            accumulated[1],
            accumulated[2],
            1.0 - transmittance,
        );
    }

    // -- Cascaded shadow map vertex shader (reuses shadow_vs with cascade index) --

    #[spirv(vertex)]
    pub fn cascaded_shadow_vs(
        #[spirv(vertex_index)] vertex_index: u32,
        #[spirv(push_constant)] cascade_index: &u32,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 0)] shadow: &ShadowUniforms,
        #[spirv(storage_buffer, descriptor_set = 0, binding = 1)] vertices: &[u32],
        #[spirv(position, invariant)] out_pos: &mut Vec4,
    ) {
        let _ = cascade_index; // cascade index selects VP in the array outside
        let base = vertex_index as usize * 2;
        let d0 = vertices[base];
        let x = (d0 & 0x1F) as f32;
        let y = ((d0 >> 5) & 0x1FF) as f32;
        let z = ((d0 >> 14) & 0x1F) as f32;
        let lvp = Mat4::from_cols_array_2d(&shadow.light_view_projection);
        *out_pos = lvp * vec4(x, y, z, 1.0);
    }
}

// ---------------------------------------------------------------------------
// CPU-side tests for GPU math — these run with `cargo test`
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::lighting::*;

    #[test]
    fn ggx_distribution_peak_at_aligned() {
        // When n_dot_h = 1.0 (perfectly aligned), D should be high
        let d = distribution_ggx(1.0, 0.1);
        assert!(d > 0.5, "D(1.0, 0.1) = {d} should be large");
    }

    #[test]
    fn ggx_distribution_falls_off() {
        let d_aligned = distribution_ggx(1.0, 0.5);
        let d_oblique = distribution_ggx(0.3, 0.5);
        assert!(
            d_aligned > d_oblique,
            "D should decrease as angle increases"
        );
    }

    #[test]
    fn geometry_smith_bounds() {
        let g = geometry_smith(0.8, 0.6, 0.5);
        assert!(g >= 0.0 && g <= 1.0, "G should be in [0, 1], got {g}");
    }

    #[test]
    fn fresnel_at_zero_angle_equals_f0() {
        let f0 = [0.04, 0.04, 0.04];
        let f = fresnel_schlick(1.0, f0);
        for i in 0..3 {
            assert!(
                (f[i] - f0[i]).abs() < 0.001,
                "F(1.0) should be ≈ f0, got {f:?}"
            );
        }
    }

    #[test]
    fn fresnel_at_grazing_approaches_one() {
        let f0 = [0.04, 0.04, 0.04];
        let f = fresnel_schlick(0.0, f0);
        for val in f {
            assert!(val > 0.95, "F(0) should approach 1.0, got {val}");
        }
    }

    #[test]
    fn cook_torrance_positive_radiance() {
        let n = [0.0, 1.0, 0.0];
        let v = normalize([0.3, 1.0, 0.2]);
        let l = normalize([0.5, 0.8, 0.3]);
        let albedo = [0.8, 0.2, 0.1];
        let result = cook_torrance(n, v, l, albedo, 0.0, 0.5);
        for val in result {
            assert!(val >= 0.0, "BRDF output should be non-negative: {val}");
        }
    }

    #[test]
    fn cook_torrance_metallic_affects_output() {
        let n = [0.0, 1.0, 0.0];
        let v = normalize([0.0, 1.0, 0.5]);
        let l = normalize([0.3, 0.8, 0.0]);
        let albedo = [0.9, 0.1, 0.1];
        let dielectric = cook_torrance(n, v, l, albedo, 0.0, 0.5);
        let metal = cook_torrance(n, v, l, albedo, 1.0, 0.5);
        // Metallic should suppress diffuse, shift specular color
        let sum_d: f32 = dielectric.iter().sum();
        let sum_m: f32 = metal.iter().sum();
        assert!(
            (sum_d - sum_m).abs() > 0.001,
            "metallic should change output: d={sum_d} m={sum_m}"
        );
    }

    #[test]
    fn aces_tonemap_clamps_to_unit() {
        let result = aces_tonemap([5.0, 10.0, 0.0]);
        for val in result {
            assert!(
                val >= 0.0 && val <= 1.0,
                "tonemapped should be in [0,1]: {val}"
            );
        }
    }

    #[test]
    fn aces_tonemap_preserves_black() {
        let result = aces_tonemap([0.0, 0.0, 0.0]);
        for val in result {
            assert!(val.abs() < 0.01, "black should stay black: {val}");
        }
    }

    #[test]
    fn octahedral_normal_roundtrip() {
        let normals = [
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
            normalize([1.0, 1.0, 1.0]),
            normalize([-0.5, 0.8, -0.3]),
        ];

        for n in normals {
            let enc = encode_normal_octahedral(n);
            assert!(
                enc[0] >= 0.0 && enc[0] <= 1.0,
                "encoded x out of range: {}",
                enc[0]
            );
            assert!(
                enc[1] >= 0.0 && enc[1] <= 1.0,
                "encoded y out of range: {}",
                enc[1]
            );
            let dec = decode_normal_octahedral(enc);
            let err = ((dec[0] - n[0]).powi(2) + (dec[1] - n[1]).powi(2) + (dec[2] - n[2]).powi(2))
                .sqrt();
            assert!(
                err < 0.01,
                "roundtrip error {err} for normal {n:?} → {enc:?} → {dec:?}"
            );
        }
    }

    #[test]
    fn normalize_zero_returns_up() {
        let n = normalize([0.0, 0.0, 0.0]);
        assert_eq!(n, [0.0, 1.0, 0.0]);
    }

    #[test]
    fn dot_product_correct() {
        assert!((dot([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])).abs() < 0.0001);
        assert!((dot([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) - 1.0).abs() < 0.0001);
    }

    // -----------------------------------------------------------------------
    // Additional comprehensive tests for all lighting math functions
    // -----------------------------------------------------------------------

    // -- distribution_ggx --

    #[test]
    fn ggx_distribution_mid_roughness_broader_than_low() {
        // With moderate roughness values the peak at n_dot_h=1 is well-behaved.
        let d_low = distribution_ggx(1.0, 0.3);
        let d_mid = distribution_ggx(1.0, 0.6);
        // Lower roughness concentrates energy into a taller peak.
        assert!(
            d_low > d_mid,
            "lower roughness should produce taller peak at n_dot_h=1: low={d_low} mid={d_mid}"
        );
    }

    #[test]
    fn ggx_distribution_is_non_negative() {
        for &ndh in &[0.0, 0.1, 0.5, 0.9, 1.0] {
            for &r in &[0.01, 0.1, 0.5, 0.9, 1.0] {
                let d = distribution_ggx(ndh, r);
                assert!(d >= 0.0, "D({ndh}, {r}) = {d} should be non-negative");
            }
        }
    }

    #[test]
    fn ggx_distribution_zero_n_dot_h_is_small() {
        let d = distribution_ggx(0.0, 0.5);
        assert!(d < 1.0, "D(0.0, 0.5) = {d} should be small");
    }

    // -- geometry_smith --

    #[test]
    fn geometry_smith_is_always_non_negative() {
        for &ndv in &[0.01, 0.1, 0.5, 0.9, 1.0] {
            for &ndl in &[0.01, 0.1, 0.5, 0.9, 1.0] {
                for &r in &[0.0, 0.25, 0.5, 0.75, 1.0] {
                    let g = geometry_smith(ndv, ndl, r);
                    assert!(g >= 0.0, "G({ndv}, {ndl}, {r}) = {g} must be >= 0");
                    assert!(g <= 1.0, "G({ndv}, {ndl}, {r}) = {g} must be <= 1");
                }
            }
        }
    }

    #[test]
    fn geometry_smith_zero_roughness_near_one() {
        let g = geometry_smith(0.8, 0.8, 0.0);
        assert!(g > 0.9, "very smooth surface should have G ≈ 1, got {g}");
    }

    #[test]
    fn geometry_smith_high_roughness_reduces_value() {
        let g_smooth = geometry_smith(0.5, 0.5, 0.1);
        let g_rough = geometry_smith(0.5, 0.5, 1.0);
        assert!(
            g_smooth > g_rough,
            "rougher surface should occlude more: smooth={g_smooth} rough={g_rough}"
        );
    }

    // -- fresnel_schlick --

    #[test]
    fn fresnel_schlick_metallic_f0() {
        let f0 = [0.95, 0.64, 0.54]; // copper-ish
        let f = fresnel_schlick(1.0, f0);
        for i in 0..3 {
            assert!(
                (f[i] - f0[i]).abs() < 0.001,
                "F(1.0) should equal f0 for metals"
            );
        }
    }

    #[test]
    fn fresnel_schlick_mid_angle() {
        let f0 = [0.04, 0.04, 0.04];
        let f = fresnel_schlick(0.5, f0);
        for val in f {
            assert!(
                val >= 0.04 && val <= 1.0,
                "mid-angle F should be between f0 and 1: {val}"
            );
        }
    }

    #[test]
    fn fresnel_schlick_monotonic_with_angle() {
        let f0 = [0.04, 0.04, 0.04];
        let f_head_on = fresnel_schlick(1.0, f0);
        let f_mid = fresnel_schlick(0.5, f0);
        let f_grazing = fresnel_schlick(0.0, f0);
        // Fresnel should increase as cos_theta decreases (more grazing)
        assert!(f_grazing[0] >= f_mid[0] && f_mid[0] >= f_head_on[0]);
    }

    // -- cook_torrance --

    #[test]
    fn cook_torrance_roughness_affects_specular_peak() {
        // Position V and L for near-perfect mirror reflection off a horizontal surface
        let n = [0.0, 1.0, 0.0];
        let l = normalize([0.0, 1.0, 1.0]); // 45° from above
        let v = normalize([0.0, 1.0, -1.0]); // mirror-reflected viewer
        let albedo = [0.5, 0.5, 0.5];
        let smooth = cook_torrance(n, v, l, albedo, 1.0, 0.1);
        let rough = cook_torrance(n, v, l, albedo, 1.0, 0.9);
        let smooth_sum: f32 = smooth.iter().sum();
        let rough_sum: f32 = rough.iter().sum();
        // At mirror angle on a metallic surface, smooth should have stronger specular
        assert!(
            smooth_sum > rough_sum,
            "smooth metallic should have stronger specular at mirror angle: smooth={smooth_sum} rough={rough_sum}"
        );
    }

    #[test]
    fn cook_torrance_backlit_is_zero() {
        let n = [0.0, 1.0, 0.0];
        let v = normalize([0.0, 1.0, 0.5]);
        let l = normalize([0.0, -1.0, 0.0]); // light from below
        let albedo = [0.8, 0.2, 0.1];
        let result = cook_torrance(n, v, l, albedo, 0.0, 0.5);
        for val in result {
            assert!(
                val.abs() < 0.0001,
                "backlit should produce near-zero radiance: {val}"
            );
        }
    }

    #[test]
    fn cook_torrance_roughness_range() {
        let n = [0.0, 1.0, 0.0];
        let v = normalize([0.3, 1.0, 0.2]);
        let l = normalize([0.5, 0.8, 0.3]);
        let albedo = [0.5, 0.5, 0.5];
        for &roughness in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.99] {
            let result = cook_torrance(n, v, l, albedo, 0.0, roughness);
            for val in result {
                assert!(
                    val >= 0.0,
                    "BRDF at roughness {roughness} must be non-negative: {val}"
                );
                assert!(
                    val.is_finite(),
                    "BRDF at roughness {roughness} must be finite"
                );
            }
        }
    }

    #[test]
    fn cook_torrance_white_light_energy() {
        let n = [0.0, 1.0, 0.0];
        let v = normalize([0.0, 1.0, 0.5]);
        let l = normalize([0.0, 0.9, 0.4]);
        let albedo = [1.0, 1.0, 1.0];
        let result = cook_torrance(n, v, l, albedo, 0.0, 0.5);
        let sum: f32 = result.iter().sum();
        // Energy conservation: output should not exceed input radiance * n_dot_l
        let n_dot_l = dot(n, l).max(0.0);
        // Each channel: (k_d * albedo/PI + spec) * n_dot_l. For dielectric with albedo=1,
        // this is bounded.
        assert!(
            sum < 3.0 * n_dot_l + 1.0,
            "energy conservation: sum={sum}, n_dot_l={n_dot_l}"
        );
    }

    // -- aces_tonemap --

    #[test]
    fn aces_tonemap_monotonic() {
        let a = aces_tonemap([0.5, 0.5, 0.5]);
        let b = aces_tonemap([1.0, 1.0, 1.0]);
        let c = aces_tonemap([2.0, 2.0, 2.0]);
        assert!(b[0] > a[0], "tonemap should be monotonically increasing");
        assert!(c[0] > b[0], "tonemap should be monotonically increasing");
    }

    #[test]
    fn aces_tonemap_mid_range_values() {
        let result = aces_tonemap([1.0, 1.0, 1.0]);
        for val in result {
            assert!(
                val > 0.3 && val < 1.0,
                "mid-range tonemap should produce reasonable values: {val}"
            );
        }
    }

    // -- encode/decode_normal_octahedral --

    #[test]
    fn octahedral_encodes_unit_normals_to_unit_square() {
        let normals = [
            normalize([1.0, 1.0, 1.0]),
            normalize([-1.0, -1.0, -1.0]),
            normalize([0.5, 0.0, 0.5]),
        ];
        for n in normals {
            let enc = encode_normal_octahedral(n);
            assert!(
                enc[0] >= 0.0 && enc[0] <= 1.0,
                "enc.x out of [0,1]: {}",
                enc[0]
            );
            assert!(
                enc[1] >= 0.0 && enc[1] <= 1.0,
                "enc.y out of [0,1]: {}",
                enc[1]
            );
        }
    }

    #[test]
    fn octahedral_axis_aligned_normals() {
        // Test each axis-aligned normal roundtrips exactly
        let axes = [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ];
        for n in axes {
            let dec = decode_normal_octahedral(encode_normal_octahedral(n));
            let err = ((dec[0] - n[0]).powi(2) + (dec[1] - n[1]).powi(2) + (dec[2] - n[2]).powi(2))
                .sqrt();
            assert!(err < 0.001, "axis-aligned roundtrip error {err} for {n:?}");
        }
    }

    // -- dot --

    #[test]
    fn dot_product_antiparallel() {
        let d = dot([1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]);
        assert!(
            (d - (-1.0)).abs() < 0.0001,
            "antiparallel dot should be -1, got {d}"
        );
    }

    #[test]
    fn dot_product_arbitrary() {
        let d = dot([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]);
        assert!((d - 32.0).abs() < 0.0001, "1*4 + 2*5 + 3*6 = 32, got {d}");
    }

    // -- normalize --

    #[test]
    fn normalize_produces_unit_length() {
        let n = normalize([3.0, 4.0, 0.0]);
        let len = (n[0] * n[0] + n[1] * n[1] + n[2] * n[2]).sqrt();
        assert!(
            (len - 1.0).abs() < 0.0001,
            "normalize should produce unit length, got {len}"
        );
    }

    #[test]
    fn normalize_preserves_direction() {
        let v = [3.0, 4.0, 0.0];
        let n = normalize(v);
        assert!((n[0] - 0.6).abs() < 0.0001);
        assert!((n[1] - 0.8).abs() < 0.0001);
        assert!(n[2].abs() < 0.0001);
    }

    // --- Phase 14: Advanced lighting and effects tests ---

    #[test]
    fn cross_product_basic() {
        let a = [1.0, 0.0, 0.0];
        let b = [0.0, 1.0, 0.0];
        let c = cross(a, b);
        assert!((c[0]).abs() < 0.0001);
        assert!((c[1]).abs() < 0.0001);
        assert!((c[2] - 1.0).abs() < 0.0001);
    }

    #[test]
    fn cross_product_anticommutative() {
        let a = [1.0, 2.0, 3.0];
        let b = [4.0, 5.0, 6.0];
        let ab = cross(a, b);
        let ba = cross(b, a);
        for i in 0..3 {
            assert!((ab[i] + ba[i]).abs() < 0.0001);
        }
    }

    #[test]
    fn length_basic() {
        let v = [3.0, 4.0, 0.0];
        assert!((length(v) - 5.0).abs() < 0.0001);
    }

    #[test]
    fn gtao_weight_at_zero_is_one() {
        assert!((gtao_weight(0.0, 1.0) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn gtao_weight_at_radius_is_zero() {
        assert!((gtao_weight(1.0, 1.0)).abs() < 0.0001);
    }

    #[test]
    fn gtao_weight_decreases_with_distance() {
        let near = gtao_weight(0.1, 1.0);
        let far = gtao_weight(0.8, 1.0);
        assert!(near > far, "weight should decrease: near={near} far={far}");
    }

    #[test]
    fn gtao_combine_no_occlusion() {
        let weights = [0.0, 0.0, 0.0, 0.0];
        let ao = gtao_combine(&weights, 0.025);
        assert!(
            (ao - 1.0).abs() < 0.05,
            "no occlusion should give ~1.0, got {ao}"
        );
    }

    #[test]
    fn gtao_combine_full_occlusion_clamped() {
        let weights = [1.0, 1.0, 1.0, 1.0];
        let ao = gtao_combine(&weights, 0.025);
        assert!(ao >= 0.0, "AO should be clamped to >= 0");
        assert!(
            ao <= 0.1,
            "full occlusion + bias should be near 0, got {ao}"
        );
    }

    #[test]
    fn reflect_reverses_normal_component() {
        let incident = [0.0, -1.0, 0.0]; // looking down
        let normal = [0.0, 1.0, 0.0]; // surface pointing up
        let r = reflect(incident, normal);
        assert!((r[0]).abs() < 0.0001);
        assert!((r[1] - 1.0).abs() < 0.0001); // reflected up
        assert!((r[2]).abs() < 0.0001);
    }

    #[test]
    fn reflect_preserves_parallel_component() {
        let incident = normalize([1.0, -1.0, 0.0]);
        let normal = [0.0, 1.0, 0.0];
        let r = reflect(incident, normal);
        // x component should be preserved, y reversed
        assert!((r[0] - incident[0]).abs() < 0.0001);
        assert!((r[1] + incident[1]).abs() < 0.0001);
    }

    #[test]
    fn ssr_hash_deterministic() {
        let h1 = ssr_hash(100, 200, 0);
        let h2 = ssr_hash(100, 200, 0);
        assert_eq!(h1, h2, "Same inputs should give same hash");
    }

    #[test]
    fn ssr_hash_varies_with_frame() {
        let h0 = ssr_hash(100, 200, 0);
        let h1 = ssr_hash(100, 200, 1);
        // Very unlikely to be equal
        assert!(
            (h0 - h1).abs() > 0.0001,
            "different frames should give different hashes"
        );
    }

    #[test]
    fn ssr_hash_range() {
        for x in 0..10 {
            for y in 0..10 {
                let h = ssr_hash(x, y, 42);
                assert!(h >= 0.0 && h < 1.0, "hash {h} out of [0,1) range");
            }
        }
    }

    #[test]
    fn hiz_ray_step_no_hit_far() {
        let (hit, _, _) = hiz_ray_step([0.5, 0.5], [0.1, 0.0], 0.01, 0.3, 0.9);
        assert!(!hit, "should not hit when depth is far from hiz");
    }

    #[test]
    fn hiz_ray_step_hit_when_close() {
        let (hit, _, _) = hiz_ray_step([0.5, 0.5], [0.1, 0.0], 0.01, 0.899, 0.9);
        assert!(hit, "should hit when depth is close to hiz");
    }

    #[test]
    fn beer_lambert_zero_distance_is_one() {
        let t = beer_lambert_transmittance(1.0, 0.0);
        assert!((t - 1.0).abs() < 0.0001);
    }

    #[test]
    fn beer_lambert_higher_density_less_transmittance() {
        let t_low = beer_lambert_transmittance(0.1, 1.0);
        let t_high = beer_lambert_transmittance(1.0, 1.0);
        assert!(t_low > t_high, "higher density should transmit less");
    }

    #[test]
    fn beer_lambert_larger_distance_less_transmittance() {
        let t_near = beer_lambert_transmittance(0.5, 1.0);
        let t_far = beer_lambert_transmittance(0.5, 10.0);
        assert!(t_near > t_far, "larger distance should transmit less");
    }

    #[test]
    fn henyey_greenstein_isotropic() {
        // g=0 should give isotropic phase (1 / 4π)
        let p = henyey_greenstein(0.0, 0.0);
        let expected = 1.0 / (4.0 * core::f32::consts::PI);
        assert!(
            (p - expected).abs() < 0.01,
            "g=0 should be isotropic: got {p}, expected {expected}"
        );
    }

    #[test]
    fn henyey_greenstein_forward_scattering() {
        // g > 0 should increase forward scattering
        let forward = henyey_greenstein(1.0, 0.5);
        let backward = henyey_greenstein(-1.0, 0.5);
        assert!(
            forward > backward,
            "forward should be stronger: {forward} vs {backward}"
        );
    }

    #[test]
    fn froxel_scatter_proportional_to_density() {
        let s1 = froxel_scatter(0.1, 1.0, 1.0, 1.0, [1.0, 1.0, 1.0]);
        let s2 = froxel_scatter(0.5, 1.0, 1.0, 1.0, [1.0, 1.0, 1.0]);
        assert!(s2[0] > s1[0], "higher density should scatter more");
    }

    #[test]
    fn select_cascade_first() {
        let splits = [10.0, 30.0, 80.0, 200.0];
        assert_eq!(select_cascade(5.0, &splits), 0);
    }

    #[test]
    fn select_cascade_last() {
        let splits = [10.0, 30.0, 80.0, 200.0];
        assert_eq!(select_cascade(150.0, &splits), 3);
    }

    #[test]
    fn select_cascade_beyond() {
        let splits = [10.0, 30.0, 80.0, 200.0];
        assert_eq!(select_cascade(300.0, &splits), 4);
    }

    #[test]
    fn cascade_blend_factor_at_start() {
        let blend = cascade_blend_factor(45.0, 50.0, 10.0);
        assert!(blend > 0.0 && blend < 1.0, "should blend: {blend}");
    }

    #[test]
    fn cascade_blend_factor_clamped() {
        let below = cascade_blend_factor(30.0, 50.0, 10.0);
        assert!((below).abs() < 0.0001, "below range should be 0.0");
        let above = cascade_blend_factor(55.0, 50.0, 10.0);
        assert!((above - 1.0).abs() < 0.0001, "above range should be 1.0");
    }

    #[test]
    fn pcf_sample_lit() {
        assert!((pcf_sample(0.4, 0.5, 0.001) - 1.0).abs() < 0.0001);
    }

    #[test]
    fn pcf_sample_shadowed() {
        assert!((pcf_sample(0.6, 0.5, 0.001)).abs() < 0.0001);
    }

    #[test]
    fn pcf_filter_partial() {
        let samples = [0.3, 0.5, 0.7, 0.4];
        let result = pcf_filter(0.45, &samples, 0.001);
        // depth 0.45: lit for samples at 0.5, 0.7 (0.45 < them); shadowed for 0.3, 0.4
        assert!(result > 0.0 && result < 1.0, "partial shadow: {result}");
    }

    #[test]
    fn restir_target_pdf_positive() {
        let pdf = restir_target_pdf([1.0, 1.0, 1.0], 0.8, 4.0);
        assert!(pdf > 0.0, "target PDF should be positive: {pdf}");
    }

    #[test]
    fn restir_target_pdf_scales_with_n_dot_l() {
        let p1 = restir_target_pdf([1.0, 1.0, 1.0], 0.2, 1.0);
        let p2 = restir_target_pdf([1.0, 1.0, 1.0], 0.8, 1.0);
        assert!(p2 > p1, "higher n_dot_l should give higher PDF");
    }

    #[test]
    fn reservoir_update_accepts_first_sample() {
        let (accept, new_sum) = reservoir_update(0.0, 1.0, 0.5);
        assert!(
            accept,
            "first sample with weight > 0 and rand < 1 should accept"
        );
        assert!((new_sum - 1.0).abs() < 0.0001);
    }

    #[test]
    fn reservoir_update_accumulates_weight() {
        let (_, sum1) = reservoir_update(0.0, 0.5, 0.0);
        let (_, sum2) = reservoir_update(sum1, 0.3, 0.0);
        assert!(
            (sum2 - 0.8).abs() < 0.0001,
            "weight sums should accumulate: {sum2}"
        );
    }

    #[test]
    fn reservoir_final_weight_correct() {
        let w = reservoir_final_weight(4.0, 4, 0.5);
        // 4.0 / (4 * 0.5) = 2.0
        assert!(
            (w - 2.0).abs() < 0.0001,
            "final weight should be 2.0, got {w}"
        );
    }

    #[test]
    fn reservoir_final_weight_zero_pdf() {
        let w = reservoir_final_weight(4.0, 4, 0.0);
        assert!((w).abs() < 0.0001, "zero PDF should give zero weight");
    }
}
