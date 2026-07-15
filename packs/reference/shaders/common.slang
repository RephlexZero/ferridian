// Shared scaffolding for every reference-pack pass: the contract camera
// block and depth helpers. The fullscreen-triangle vertex stage lives in
// fullscreen.slang, included by graphics passes only — compute passes
// include this file alone so their module carries a single entry point.
//
// Included (not imported) so each pass module stays self-contained — packc
// compiles one module per pass.

// The contract camera block (ferridian_contract::CameraUniforms, std140):
// member names, order, and offsets are the contract, and the engine wires
// the binding by its variable name — `camera`, storage class Uniform. The
// slot sits above every pass's sampler range so the shared declaration can
// never collide with a per-pass binding. Until the shim publishes real
// per-frame values the engine uploads the placeholder (the same fixed sun
// and Minecraft-shaped clip planes these used to hard-code as constants).
struct Camera
{
    float3 sun_direction;
    float time_seconds;
    float near_plane;
    float far_plane;
};

[[vk::binding(7, 0)]]
ConstantBuffer<Camera> camera;

// Standard [0,1] hardware depth to view-space distance.
float linearize_depth(float depth)
{
    return camera.near_plane * camera.far_plane
        / (camera.far_plane - depth * (camera.far_plane - camera.near_plane));
}

// View-space position under a unit-focal projection; the real projection
// arrives with the shim's matrices.
float3 view_position(float2 uv, float depth)
{
    float z = linearize_depth(depth);
    float2 ndc = uv * 2.0 - 1.0;
    return float3(ndc * z, z);
}
