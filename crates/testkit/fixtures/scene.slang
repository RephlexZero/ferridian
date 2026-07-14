// A game-shaped synthetic scene for layer↔executor tests: color is a
// warm-left / cool-right albedo split, depth is two plateaus written through
// SV_Depth — near geometry in the top half, the far plane in the bottom.
// Hardware depth is hyperbolic under the reference pack's placeholder camera
// (NEAR 0.05, FAR 512), so a smooth ramp would collapse almost the whole far
// field into one quantization step; plateaus are what actually exercise the
// fog march. Mirrors the synthetic fixtures in testkit's pack_exec test
// (depth 77/255 ≈ 0.302, and the far plane), so the two tests assert the
// same physics through different seams.

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

struct FsOut
{
    float4 color : SV_Target;
    float depth : SV_Depth;
};

[shader("fragment")]
FsOut fs_main(VsOut input)
{
    FsOut output;
    float3 albedo = input.uv.x < 0.5
        ? float3(200.0, 140.0, 80.0) / 255.0
        : float3(80.0, 120.0, 200.0) / 255.0;
    output.color = float4(albedo, 1.0);
    output.depth = input.uv.y < 0.5 ? 77.0 / 255.0 : 1.0;
    return output;
}
