// Deferred-style lighting over the intercepted frame: reconstruct a G-buffer
// (view-space position from depth, face normal from position derivatives),
// then relight with a sun + ambient split modulated by the shadows pass.
// A true material G-buffer arrives with the layer's render-target taps.

#include "fullscreen.slang"
#include "common.slang"

[[vk::binding(0, 0)]]
Sampler2D game_color;
[[vk::binding(1, 0)]]
Sampler2D game_depth;
[[vk::binding(2, 0)]]
Sampler2D shadow_mask;

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    float3 albedo = game_color.Sample(input.uv).rgb;
    float3 position = view_position(input.uv, game_depth.Sample(input.uv).r);
    // Derivative normals are exactly what a screen-space pass can know
    // without material taps: flat per-quad, but consistent.
    float3 normal = normalize(cross(ddy(position), ddx(position)));
    float shadow = shadow_mask.Sample(input.uv).r;

    float sun = max(dot(normal, camera.sun_direction), 0.0) * shadow;
    float3 lit = albedo * (0.25 + 0.75 * sun);
    return float4(lit, 1.0);
}
