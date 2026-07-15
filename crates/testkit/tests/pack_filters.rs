//! Per-binding sampler filters end to end: a fixture pack upscales a 2×2
//! input to a 4×4 frame twice — once through a `filters`-declared linear
//! binding, once through the nearest default — and packs both reads into one
//! output pixel. The expected values differ per channel, so the assertions
//! fail if the filter table doesn't reach the right sampler (both channels
//! blocky, or both interpolated), not just if sampling breaks outright.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use ash::vk;
use ferridian_engine::exec::{Filter, plan_execution};
use ferridian_engine::pack::load_pack;
use ferridian_pack_compiler::{SlangCompiler, compile_pack, write_artifact};
use ferridian_testkit::{
    RgbaImage, TestGpu, create_target, read_back, require_gpu, upload_texture,
};
use ferridian_vk_rt::{CameraUniforms, OutputTarget, PackExecutor};

/// Frame extent; the game_color fixture is 2×2, so sampling it across the
/// frame magnifies 2× — where linear and nearest visibly diverge.
const SIZE: u32 = 4;

const MANIFEST: &str = r#"
[pack]
name = "filter-fixture"
version = "0.1.0"

[[pass]]
name = "smooth"
kind = "graphics"
shader = "shaders/smooth.slang"
inputs = ["game_color"]
outputs = ["soft"]
filters = { game_color = "linear" }

[[pass]]
name = "merge"
kind = "graphics"
shader = "shaders/merge.slang"
inputs = ["soft", "game_color"]
outputs = ["swapchain"]
"#;

const FULLSCREEN: &str = r#"
struct VsOut
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOut vs_main(uint vertexId: SV_VertexID)
{
    VsOut output;
    float2 uv = float2((vertexId << 1) & 2, vertexId & 2);
    output.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    output.uv = uv;
    return output;
}
"#;

const SMOOTH: &str = r#"
#include "fullscreen.slang"

[[vk::binding(0, 0)]]
Sampler2D game_color;

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    return game_color.Sample(input.uv);
}
"#;

/// Red = the linear upscale (via `soft`), green = the same source read
/// through this pass's own nearest-default binding.
const MERGE: &str = r#"
#include "fullscreen.slang"

[[vk::binding(0, 0)]]
Sampler2D soft;
[[vk::binding(1, 0)]]
Sampler2D game_color;

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    return float4(soft.Sample(input.uv).r, game_color.Sample(input.uv).r, 0.0, 1.0);
}
"#;

fn write_fixture_pack() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferridian-filter-src-{}", std::process::id()));
    let shaders = dir.join("shaders");
    fs::create_dir_all(&shaders).expect("create fixture pack dir");
    fs::write(dir.join("pack.toml"), MANIFEST).expect("write manifest");
    fs::write(shaders.join("fullscreen.slang"), FULLSCREEN).expect("write fullscreen stage");
    fs::write(shaders.join("smooth.slang"), SMOOTH).expect("write smooth shader");
    fs::write(shaders.join("merge.slang"), MERGE).expect("write merge shader");
    dir
}

/// 2×2: black left column, white right column (identical rows).
fn two_tone_2x2() -> RgbaImage {
    let mut image = RgbaImage::new(2, 2);
    for (index, pixel) in image.pixels.chunks_exact_mut(4).enumerate() {
        let value = if index % 2 == 0 { 0 } else { 255 };
        pixel.copy_from_slice(&[value, value, value, 255]);
    }
    image
}

#[test]
fn declared_filters_reach_their_bindings() {
    require_gpu!();
    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");
    let src = write_fixture_pack();
    let artifact = compile_pack(&src, &compiler).expect("filter fixture pack compiles");
    let out = std::env::temp_dir().join(format!("ferridian-filter-out-{}", std::process::id()));
    write_artifact(&artifact, &out).expect("write artifact");

    let pack = load_pack(&out, 1).expect("artifact loads");
    let plan = plan_execution(&pack).expect("filter pack wires");
    assert_eq!(plan.passes[0].bindings[0].filter, Filter::Linear);
    assert_eq!(
        plan.passes[1]
            .bindings
            .iter()
            .map(|binding| binding.filter)
            .collect::<Vec<_>>(),
        vec![Filter::Nearest, Filter::Nearest],
        "unlisted inputs default to nearest"
    );

    let mut gpu = TestGpu::new();
    let runtime = gpu.runtime();
    let mut game_color = upload_texture(runtime, &two_tone_2x2());
    let mut target = create_target(
        runtime,
        SIZE,
        SIZE,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );
    let ctx = runtime.exec_context();
    let external_inputs = BTreeMap::from([("game_color".to_owned(), game_color.view)]);
    let output = OutputTarget {
        view: target.view,
        format: vk::Format::R8G8B8A8_UNORM,
        final_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    };
    let mut executor = PackExecutor::new(
        &ctx,
        &plan,
        &pack.modules,
        vk::Extent2D {
            width: SIZE,
            height: SIZE,
        },
        &external_inputs,
        &CameraUniforms::placeholder(),
        &output,
    )
    .expect("filter plan instantiates on the device");
    executor.execute(&ctx).expect("filter pack executes");
    let frame = read_back(runtime, target.image, SIZE, SIZE);

    // Texel-space coordinates at the 4 sample columns are -0.25, 0.25, 0.75,
    // 1.25 over the 2-texel source: linear blends 0.75/0.25 (then 0.25/0.75)
    // between black and white, nearest snaps. All weights are exact in
    // hardware fixed point; the ±1 window only covers UNORM8 rounding of
    // 63.75/191.25.
    let expected_linear = [0i16, 64, 191, 255];
    let expected_nearest = [0u8, 0, 255, 255];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let offset = ((y * SIZE + x) * 4) as usize;
            let [r, g, b, a]: [u8; 4] = frame.pixels[offset..offset + 4]
                .try_into()
                .expect("4-byte pixel");
            assert!(
                (r as i16 - expected_linear[x as usize]).abs() <= 1,
                "linear channel at ({x},{y}): expected ~{}, got {r}",
                expected_linear[x as usize]
            );
            assert_eq!(
                g, expected_nearest[x as usize],
                "nearest channel at ({x},{y})"
            );
            assert_eq!((b, a), (0, 255), "constant channels at ({x},{y})");
        }
    }

    // SAFETY: execute() fence-waits and read_back fence-waits its own
    // submission — no work references these objects.
    unsafe { executor.destroy(gpu.device()) };
    game_color.destroy(gpu.device());
    target.destroy(gpu.device());
    gpu.assert_no_validation_errors();

    fs::remove_dir_all(&src).ok();
    fs::remove_dir_all(&out).ok();
}
