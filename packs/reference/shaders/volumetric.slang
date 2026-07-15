// Single-scatter volumetric light, v0: march the view ray through a
// constant-density fog, accumulating sunlight where the shadow term says the
// sun reaches. Output is (inscattered rgb, transmittance a) so composite can
// attenuate-then-add. Height falloff and froxel scattering land once
// world-space transforms are available from the contract.
//
// A compute pass — the first one: fog needs no rasterization, and marching
// is exactly the kind of work later GI passes will do in compute. Sampling
// uses SampleLevel (compute has no implicit derivatives), which at mip
// level 0 over unmipped inputs matches the fragment version texel for texel.

#include "common.slang"

[[vk::binding(0, 0)]]
Sampler2D game_depth;
[[vk::binding(1, 0)]]
Sampler2D shadow_mask;
// The declared format must match the executor's intermediate images
// (R16G16B16A16_SFLOAT). Without it the SPIR-V needs the optional
// shaderStorageImage{Read,Write}WithoutFormat features — which a real game's
// device does not enable, so pipeline creation fails under the game's own
// validation (observed live: Minecraft 26.2 with --vulkanValidation true).
[[vk::binding(2, 0)]]
[format("rgba16f")]
RWTexture2D<float4> fog;

static const uint FOG_STEPS = 16;
static const float FOG_DENSITY = 0.02;
static const float3 SUN_COLOR = float3(1.0, 0.9, 0.7);

[shader("compute")]
[numthreads(8, 8, 1)]
void cs_main(uint3 tid: SV_DispatchThreadID)
{
    uint width, height;
    fog.GetDimensions(width, height);
    if (tid.x >= width || tid.y >= height)
        return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(width, height);

    float end = linearize_depth(game_depth.SampleLevel(uv, 0).r);
    // Screen-space proxy for sun reach along the ray; per-step shadow-map
    // taps replace this when real shadow maps are captured.
    float shadow = shadow_mask.SampleLevel(uv, 0).r;

    float stepLength = end / float(FOG_STEPS);
    float transmittance = 1.0;
    float inscattered = 0.0;
    for (uint i = 0; i < FOG_STEPS; ++i)
    {
        float absorb = exp(-FOG_DENSITY * stepLength);
        inscattered += transmittance * (1.0 - absorb) * shadow;
        transmittance *= absorb;
    }
    fog[tid.xy] = float4(SUN_COLOR * inscattered, transmittance);
}
