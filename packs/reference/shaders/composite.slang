// Fullscreen composite: samples the captured game color and applies a
// placeholder tonemap. The simplest possible pass that exercises the whole
// packc pipeline (Slang -> SPIR-V -> reflection -> artifact).

struct VsOut
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOut vs_main(uint vertexId: SV_VertexID)
{
    // Fullscreen triangle from the vertex id; no vertex buffer needed.
    VsOut output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}

[[vk::binding(0, 0)]]
Sampler2D game_color;

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    float3 color = game_color.Sample(input.uv).rgb;
    // Placeholder Reinhard tonemap; real grading arrives with the showcase.
    color = color / (color + 1.0);
    return float4(color, 1.0);
}
