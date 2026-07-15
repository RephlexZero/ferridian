// Final combine: the lit scene attenuated by fog transmittance, plus the
// fog's inscattered light, through a placeholder Reinhard tonemap. Real
// grading (exposure, purkinje, LUTs) arrives with the showcase.

#include "fullscreen.slang"
#include "common.slang"

[[vk::binding(0, 0)]]
Sampler2D lit;
[[vk::binding(1, 0)]]
Sampler2D fog;

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    float4 scatter = fog.Sample(input.uv);
    float3 color = lit.Sample(input.uv).rgb * scatter.a + scatter.rgb;
    color = color / (color + 1.0);
    return float4(color, 1.0);
}
