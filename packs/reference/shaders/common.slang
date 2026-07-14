// Shared scaffolding for every reference-pack pass: the fullscreen-triangle
// vertex stage and the placeholder camera model.
//
// Included (not imported) so each pass module stays self-contained — packc
// compiles one module per pass, and every graphics module carries its own
// vertex stage.

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

// Placeholder camera model until the contract exposes the game's real
// matrices and sun state: a fixed sun, fixed clip planes, and a unit-focal
// projection. Every use is view-space-consistent, so swapping in the real
// uniforms later changes constants, not structure.
static const float3 SUN_DIRECTION = normalize(float3(0.35, 0.9, 0.2));
static const float NEAR_PLANE = 0.05;
static const float FAR_PLANE = 512.0;

// Standard [0,1] hardware depth to view-space distance.
float linearize_depth(float depth)
{
    return NEAR_PLANE * FAR_PLANE / (FAR_PLANE - depth * (FAR_PLANE - NEAR_PLANE));
}

// View-space position under the placeholder projection.
float3 view_position(float2 uv, float depth)
{
    float z = linearize_depth(depth);
    float2 ndc = uv * 2.0 - 1.0;
    return float3(ndc * z, z);
}
