//! Multiple render targets end to end: a purpose-built two-pass pack whose
//! first pass rasterizes once into *two* color attachments with
//! distinguishable per-target math, so the exact-value assertions fail if
//! the executor maps manifest outputs to the wrong attachment locations —
//! not just if it drops one. Compiled by the pinned slangc at test time,
//! loaded as an untrusted artifact, validation silent.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use ash::vk;
use ferridian_engine::exec::plan_execution;
use ferridian_engine::pack::load_pack;
use ferridian_pack_compiler::{SlangCompiler, compile_pack, write_artifact};
use ferridian_testkit::{
    RgbaImage, TestGpu, create_target, read_back, require_gpu, upload_texture,
};
use ferridian_vk_rt::{CameraUniforms, OutputTarget, PackExecutor};

const SIZE: u32 = 64;

const MANIFEST: &str = r#"
[pack]
name = "mrt-fixture"
version = "0.1.0"

[[pass]]
name = "split"
kind = "graphics"
shader = "shaders/split.slang"
inputs = ["game_color"]
outputs = ["halved", "inverted"]

[[pass]]
name = "merge"
kind = "graphics"
shader = "shaders/merge.slang"
inputs = ["halved", "inverted"]
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

/// SV_Target0/1 land on locations 0/1 — manifest order, per the wiring rules.
const SPLIT: &str = r#"
#include "fullscreen.slang"

[[vk::binding(0, 0)]]
Sampler2D game_color;

struct Split
{
    float4 halved : SV_Target0;
    float4 inverted : SV_Target1;
};

[shader("fragment")]
Split fs_main(VsOut input)
{
    float3 color = game_color.Sample(input.uv).rgb;
    Split output;
    output.halved = float4(color * 0.5, 1.0);
    output.inverted = float4(1.0 - color, 1.0);
    return output;
}
"#;

const MERGE: &str = r#"
#include "fullscreen.slang"

[[vk::binding(0, 0)]]
Sampler2D halved;
[[vk::binding(1, 0)]]
Sampler2D inverted;

[shader("fragment")]
float4 fs_main(VsOut input) : SV_Target
{
    float3 color = halved.Sample(input.uv).rgb * 0.5 + inverted.Sample(input.uv).rgb;
    return float4(color, 1.0);
}
"#;

fn write_fixture_pack() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferridian-mrt-src-{}", std::process::id()));
    let shaders = dir.join("shaders");
    fs::create_dir_all(&shaders).expect("create fixture pack dir");
    fs::write(dir.join("pack.toml"), MANIFEST).expect("write manifest");
    fs::write(shaders.join("fullscreen.slang"), FULLSCREEN).expect("write fullscreen stage");
    fs::write(shaders.join("split.slang"), SPLIT).expect("write split shader");
    fs::write(shaders.join("merge.slang"), MERGE).expect("write merge shader");
    dir
}

fn solid_magenta() -> RgbaImage {
    let mut image = RgbaImage::new(SIZE, SIZE);
    for pixel in image.pixels.chunks_exact_mut(4) {
        pixel.copy_from_slice(&[255, 0, 255, 255]);
    }
    image
}

#[test]
fn one_pass_writes_two_targets_and_both_arrive_intact() {
    require_gpu!();
    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");
    let src = write_fixture_pack();
    let artifact = compile_pack(&src, &compiler).expect("MRT fixture pack compiles");
    let out = std::env::temp_dir().join(format!("ferridian-mrt-out-{}", std::process::id()));
    write_artifact(&artifact, &out).expect("write artifact");

    let pack = load_pack(&out, 1).expect("artifact loads");
    let plan = plan_execution(&pack).expect("MRT pack wires");
    assert_eq!(
        plan.passes[0]
            .outputs
            .iter()
            .map(|resource| resource.0.as_str())
            .collect::<Vec<_>>(),
        vec!["halved", "inverted"],
        "the split pass carries both outputs in manifest order"
    );
    assert_eq!(
        plan.intermediates
            .iter()
            .map(|resource| resource.0.as_str())
            .collect::<Vec<_>>(),
        vec!["halved", "inverted"]
    );

    let mut gpu = TestGpu::new();
    let runtime = gpu.runtime();
    let mut game_color = upload_texture(runtime, &solid_magenta());
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
    .expect("MRT plan instantiates on the device");
    executor.execute(&ctx).expect("MRT pack executes");
    let frame = read_back(runtime, target.image, SIZE, SIZE);

    // magenta (1, 0, 1): halved = (.5, 0, .5), inverted = (0, 1, 0), so the
    // merge is (.25, 1, .25) — every value exact in RGBA16F, and 63.75
    // rounds to 64 in UNORM8. A *swapped* attachment order would instead
    // produce (.5, .5, .5): these constants prove placement, not just
    // survival.
    let expected = [64u8, 255, 64, 255];
    assert!(
        frame.pixels.chunks_exact(4).all(|pixel| pixel == expected),
        "every pixel must be {expected:?}, got {:?} at the first mismatch",
        frame
            .pixels
            .chunks_exact(4)
            .find(|pixel| *pixel != expected)
    );

    // SAFETY: execute() fence-waits and read_back fence-waits its own
    // submission — no work references these objects.
    unsafe { executor.destroy(gpu.device()) };
    game_color.destroy(gpu.device());
    target.destroy(gpu.device());
    gpu.assert_no_validation_errors();

    fs::remove_dir_all(&src).ok();
    fs::remove_dir_all(&out).ok();
}
