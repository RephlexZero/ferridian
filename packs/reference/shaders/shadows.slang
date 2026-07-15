// Screen-space contact shadows: march from each pixel toward the sun and
// darken where nearer geometry occludes the ray. This produces the shadow
// term the deferred and volumetric passes consume; real cascaded shadow maps
// arrive once the layer taps the game's own shadow pass.

#include "fullscreen.slang"
#include "common.slang"

[[vk::binding(0, 0)]]
Sampler2D game_depth;

static const uint MARCH_STEPS = 12;
// March length in uv units and the view-space band in which a nearer sample
// counts as an occluder rather than unrelated geometry.
static const float MARCH_LENGTH = 0.08;
static const float THICKNESS = 0.5;

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    float3 origin = view_position(input.uv, game_depth.Sample(input.uv).r);

    // Screen-space direction toward the sun; the epsilon keeps a sun straight
    // overhead from degenerating into a zero-length march.
    float2 stepUv = normalize(float2(camera.sun_direction.x, -camera.sun_direction.z) + 1e-4)
        * (MARCH_LENGTH / float(MARCH_STEPS));
    float rise = camera.sun_direction.y * THICKNESS;

    float visibility = 1.0;
    for (uint i = 1; i <= MARCH_STEPS; ++i)
    {
        float2 uv = input.uv + stepUv * float(i);
        if (any(uv < 0.0) || any(uv > 1.0))
            break;
        float sampleZ = linearize_depth(game_depth.Sample(uv).r);
        float expectedZ = origin.z - rise * float(i);
        if (sampleZ < expectedZ - 0.02 && expectedZ - sampleZ < THICKNESS)
        {
            visibility = 0.0;
            break;
        }
    }
    return float4(visibility.xxx, 1.0);
}
