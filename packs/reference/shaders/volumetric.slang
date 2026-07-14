// Single-scatter volumetric light, v0: march the view ray through a
// constant-density fog, accumulating sunlight where the shadow term says the
// sun reaches. Output is (inscattered rgb, transmittance a) so composite can
// attenuate-then-add. Height falloff and froxel scattering land once
// world-space transforms are available from the contract.

#include "common.slang"

[[vk::binding(0, 0)]]
Sampler2D game_depth;
[[vk::binding(1, 0)]]
Sampler2D shadow_mask;

static const uint FOG_STEPS = 16;
static const float FOG_DENSITY = 0.02;
static const float3 SUN_COLOR = float3(1.0, 0.9, 0.7);

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    float end = linearize_depth(game_depth.Sample(input.uv).r);
    // Screen-space proxy for sun reach along the ray; per-step shadow-map
    // taps replace this when real shadow maps are captured.
    float shadow = shadow_mask.Sample(input.uv).r;

    float stepLength = end / float(FOG_STEPS);
    float transmittance = 1.0;
    float inscattered = 0.0;
    for (uint i = 0; i < FOG_STEPS; ++i)
    {
        float absorb = exp(-FOG_DENSITY * stepLength);
        inscattered += transmittance * (1.0 - absorb) * shadow;
        transmittance *= absorb;
    }
    return float4(SUN_COLOR * inscattered, transmittance);
}
