//! M4 seam end to end: the real reference pack — compiled by the pinned
//! slangc, loaded as an untrusted artifact, wired by name (binding ↔
//! resource), instantiated as images/descriptors/pipelines — *executes* on a
//! real device: four scheduled passes over synthetic game inputs, pixels out,
//! validation silent. No seam is mocked.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use ash::vk;
use ferridian_engine::exec::{StagePlan, plan_execution};
use ferridian_engine::pack::load_pack;
use ferridian_pack_compiler::{SlangCompiler, compile_pack, write_artifact};
use ferridian_testkit::{
    RgbaImage, TestGpu, create_target, read_back, require_gpu, upload_texture,
};
use ferridian_vk_rt::{CameraUniforms, OutputTarget, PackExecutor};

const SIZE: u32 = 128;

fn reference_pack_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/reference")
}

/// Two-tone albedo: warm left half, cool right half.
fn synthetic_game_color() -> RgbaImage {
    let mut image = RgbaImage::new(SIZE, SIZE);
    for y in 0..SIZE {
        for x in 0..SIZE {
            let offset = ((y * SIZE + x) * 4) as usize;
            let rgb: [u8; 3] = if x < SIZE / 2 {
                [200, 140, 80]
            } else {
                [80, 120, 200]
            };
            image.pixels[offset..offset + 3].copy_from_slice(&rgb);
            image.pixels[offset + 3] = 255;
        }
    }
    image
}

/// Depth plateaus: near geometry in the top half, the far plane in the
/// bottom half. Hardware depth is hyperbolic — under the placeholder camera
/// (NEAR 0.05, FAR 512) an 8-bit ramp collapses almost the whole far field
/// into its last quantization step, so a smooth u8 ramp cannot exercise the
/// fog march. Two plateaus can: ~0.07 view units vs the full 512.
fn synthetic_game_depth() -> RgbaImage {
    let mut image = RgbaImage::new(SIZE, SIZE);
    for y in 0..SIZE {
        let depth = if y < SIZE / 2 { 77 } else { 255 };
        for x in 0..SIZE {
            let offset = ((y * SIZE + x) * 4) as usize;
            image.pixels[offset..offset + 4].copy_from_slice(&[depth, depth, depth, 255]);
        }
    }
    image
}

fn mean_luma(image: &RgbaImage, rows: std::ops::Range<u32>) -> f64 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for y in rows {
        for x in 0..image.width {
            let offset = ((y * image.width + x) * 4) as usize;
            sum += image.pixels[offset] as u64
                + image.pixels[offset + 1] as u64
                + image.pixels[offset + 2] as u64;
            count += 3;
        }
    }
    sum as f64 / count as f64
}

#[test]
fn reference_pack_executes_end_to_end_on_a_real_device() {
    require_gpu!();
    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");
    let artifact = compile_pack(&reference_pack_dir(), &compiler).expect("reference pack compiles");
    let out = std::env::temp_dir().join(format!("ferridian-pack-exec-{}", std::process::id()));
    write_artifact(&artifact, &out).expect("write artifact");

    // Loaded exactly as the watcher would load it: untrusted artifact bytes.
    let pack = load_pack(&out, 1).expect("artifact loads");
    let plan = plan_execution(&pack).expect("reference pack wires by name");
    assert_eq!(plan.passes.len(), 4);
    assert_eq!(
        plan.passes.last().map(|pass| pass.name.as_str()),
        Some("composite"),
        "the swapchain writer is scheduled last"
    );
    assert_eq!(
        plan.external_inputs
            .iter()
            .map(|resource| resource.0.as_str())
            .collect::<Vec<_>>(),
        vec!["game_color", "game_depth"]
    );
    assert_eq!(
        plan.intermediates
            .iter()
            .map(|resource| resource.0.as_str())
            .collect::<Vec<_>>(),
        vec!["shadow_mask", "lit", "fog"]
    );
    // Camera-consuming passes wire the contract block; composite never
    // references it, and slangc strips the unused declaration.
    assert_eq!(
        plan.passes
            .iter()
            .map(|pass| (pass.name.as_str(), pass.camera_binding))
            .collect::<Vec<_>>(),
        vec![
            ("shadows", Some(7)),
            ("deferred", Some(7)),
            ("volumetric", Some(7)),
            ("composite", None),
        ]
    );
    // Volumetric is the pack's first compute pass: dispatched over the
    // frame, writing `fog` as a storage image instead of an attachment.
    assert_eq!(
        plan.passes[2].stage,
        StagePlan::Compute {
            entry: "cs_main".to_owned(),
            workgroup_size: [8, 8, 1],
        }
    );
    assert_eq!(plan.passes[2].output_binding, Some(2));

    let mut gpu = TestGpu::new();
    let runtime = gpu.runtime();
    let mut game_color = upload_texture(runtime, &synthetic_game_color());
    let mut game_depth = upload_texture(runtime, &synthetic_game_depth());
    let mut target = create_target(
        runtime,
        SIZE,
        SIZE,
        vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_SRC,
    );

    let ctx = runtime.exec_context();
    let external_inputs = BTreeMap::from([
        ("game_color".to_owned(), game_color.view),
        ("game_depth".to_owned(), game_depth.view),
    ]);
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
    .expect("plan instantiates on the device");

    executor.execute(&ctx).expect("pack executes");
    let first = read_back(runtime, target.image, SIZE, SIZE);
    if let Some(dir) = std::env::var_os("FERRIDIAN_EXEC_DUMP") {
        first
            .write_png(&Path::new(&dir).join("pack_exec_output.png"))
            .expect("dump output");
    }
    // An executor is a per-reload object, not a per-frame one: the same
    // pipelines must replay deterministically.
    executor.execute(&ctx).expect("pack re-executes");
    let second = read_back(runtime, target.image, SIZE, SIZE);
    assert_eq!(first, second, "two executions produce identical frames");

    // The camera block is live, not baked: pull the far plane in from 512
    // to 8 view units and the far plateau's fog march collapses — the
    // fog-dominated bottom half must change dramatically. (The sun direction
    // would be a poor probe here: the fixture's plateaus face the camera, so
    // the deferred sun term is already zero.)
    // SAFETY: execute() fence-waits and read_back fence-waits its own
    // submission, so nothing in flight is reading the buffer.
    unsafe {
        executor
            .update_camera(
                gpu.device(),
                &CameraUniforms {
                    far_plane: 8.0,
                    ..CameraUniforms::placeholder()
                },
            )
            .expect("camera updates");
    }
    executor
        .execute(&ctx)
        .expect("pack executes with a short far plane");
    let short_range = read_back(runtime, target.image, SIZE, SIZE);
    let foggy_far = mean_luma(&first, SIZE / 2..SIZE);
    let short_far = mean_luma(&short_range, SIZE / 2..SIZE);
    assert!(
        (foggy_far - short_far).abs() > 25.0,
        "an 8-unit far plane must collapse the fog march \
         (far half {foggy_far:.1} vs {short_far:.1})"
    );

    // And restoring the placeholder restores the exact frame.
    // SAFETY: as above — the sunless execution was fence-waited.
    unsafe {
        executor
            .update_camera(gpu.device(), &CameraUniforms::placeholder())
            .expect("camera restores");
    }
    executor.execute(&ctx).expect("pack re-executes restored");
    let restored = read_back(runtime, target.image, SIZE, SIZE);
    assert_eq!(
        first, restored,
        "restoring the camera reproduces the frame exactly"
    );

    // The composite writes alpha = 1 everywhere; anything else means a pass
    // didn't cover the frame.
    assert!(
        first.pixels.chunks_exact(4).all(|px| px[3] == 255),
        "alpha must be 1.0 across the frame"
    );
    // The albedo split must survive lighting where geometry is near (the
    // top half — the fogged bottom half converges to the sun color).
    let (mut left, mut right) = (0u64, 0u64);
    for y in 0..SIZE / 2 {
        for x in 0..SIZE {
            let offset = ((y * SIZE + x) * 4) as usize;
            let red = first.pixels[offset] as u64;
            if x < SIZE / 2 {
                left += red;
            } else {
                right += red;
            }
        }
    }
    assert!(
        left > right,
        "warm half should stay redder than cool half (left {left} vs right {right})"
    );
    // The far plateau must come back fog-dominated: transmittance ~0 kills
    // the albedo and inscattered sunlight takes over, so the two plateaus
    // can't resemble each other.
    let near = mean_luma(&first, 0..8);
    let far = mean_luma(&first, SIZE - 8..SIZE);
    assert!(
        (far - near).abs() > 25.0,
        "volumetric fog should separate near ({near:.1}) from far ({far:.1}) plateaus"
    );

    // The pixel checks above are necessary but not sufficient — pin the
    // exact frame so any unintended shift in any of the four passes fails
    // the build, not just a code reviewer's eyeball.
    ferridian_testkit::assert_matches_golden("reference_pack_composite", &first);

    // SAFETY: execute() fence-waits, and the readback submissions above were
    // fence-waited by read_back — no work references these objects.
    unsafe { executor.destroy(gpu.device()) };
    game_color.destroy(gpu.device());
    game_depth.destroy(gpu.device());
    target.destroy(gpu.device());
    gpu.assert_no_validation_errors();

    std::fs::remove_dir_all(&out).ok();
}
