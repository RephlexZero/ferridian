// Hi-Z depth pyramid downscale compute shader.
// Takes one mip level and produces the next by taking the max (farthest)
// depth from each 2x2 neighbourhood.  Dispatch one workgroup per 8x8 output
// texels.
//
// Bind group 0:
//   binding 0: source mip (texture_2d<f32>)
//   binding 1: output mip (texture_storage_2d<r32float, write>)
//   binding 2: uniform { src_width, src_height, dst_width, dst_height }

struct HizDownscaleUniforms {
    src_width: f32,
    src_height: f32,
    dst_width: f32,
    dst_height: f32,
};

@group(0) @binding(0) var src_mip: texture_2d<f32>;
@group(0) @binding(1) var dst_mip: texture_storage_2d<r32float, write>;
@group(0) @binding(2) var<uniform> uniforms: HizDownscaleUniforms;

@compute @workgroup_size(8, 8, 1)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let dst_coord = vec2<i32>(i32(id.x), i32(id.y));
    if f32(dst_coord.x) >= uniforms.dst_width || f32(dst_coord.y) >= uniforms.dst_height {
        return;
    }

    let src_coord = dst_coord * 2;

    // Sample 2x2 neighbourhood from the source mip, taking the max (farthest)
    // depth.  This is conservative — if any of the 4 texels are far, the
    // downscaled texel reports far, so we never incorrectly cull geometry that
    // sits behind a nearer occluder at full resolution.
    let d00 = textureLoad(src_mip, src_coord + vec2<i32>(0, 0), 0).x;
    let d10 = textureLoad(src_mip, src_coord + vec2<i32>(1, 0), 0).x;
    let d01 = textureLoad(src_mip, src_coord + vec2<i32>(0, 1), 0).x;
    let d11 = textureLoad(src_mip, src_coord + vec2<i32>(1, 1), 0).x;

    let max_depth = max(max(d00, d10), max(d01, d11));

    textureStore(dst_mip, dst_coord, vec4<f32>(max_depth, 0.0, 0.0, 0.0));
}
