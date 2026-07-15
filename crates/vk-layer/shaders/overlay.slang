// The layer's composite-proof overlay (M2 walking skeleton): a solid magenta
// corner triangle drawn *into the game's own render pass* after its draws.
// Deliberately unmistakable against any plausible game output; zero bindings
// so no descriptor state can clash with the host application's.
//
// The committed `overlay.vertex.spv`/`overlay.fragment.spv` next to this file
// are what the layer embeds (a layer inside a game process has no slangc,
// and a module mixing both entry points is exactly what GPU-assisted
// validation can't instrument); `committed_overlay_spv_matches_pinned_slangc_output`
// keeps them in sync.

struct VsOut
{
    float4 position : SV_Position;
};

[shader("vertex")]
VsOut vs_main(uint vertexId: SV_VertexID)
{
    // A right triangle tucked into the top-left corner of clip space.
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(-0.55, -1.0),
        float2(-1.0, -0.55),
    };
    VsOut output;
    output.position = float4(positions[vertexId], 0.0, 1.0);
    return output;
}

[shader("fragment")]
float4 fs_main() : SV_Target
{
    return float4(1.0, 0.0, 1.0, 1.0);
}
