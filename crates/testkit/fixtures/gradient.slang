// Golden-harness fixture: a fullscreen UV gradient with zero bindings, so the
// whole slangc -> SPIR-V -> pipeline -> readback path is exercised with no
// descriptor machinery in the way. Deterministic per-pixel math only.

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

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    float3 color = float3(input.uv, 0.25 + 0.5 * input.uv.x * input.uv.y);
    return float4(color, 1.0);
}
