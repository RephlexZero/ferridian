//! The M4 seam, end to end on the real Vulkan loader: the layer loads the
//! *real* reference pack (compiled by the pinned slangc, published as an
//! untrusted artifact), and at the end of the intercepted world-final pass
//! taps the application's own color+depth attachments, runs the pack's four
//! passes over them, and writes the composite back into the application's
//! color attachment — which the app then reads back, none the wiser.
//!
//! Assertions are physical: the lit albedo split survives in the near half,
//! the far half comes back fog-dominated, unclassified and non-trigger
//! passes are untouched, and replays are deterministic.
//!
//! nextest runs each test in its own process, so setting env vars here is
//! safe. The staging helpers mirror `loaded.rs` (test binaries don't share
//! modules).

use std::path::PathBuf;

use ferridian_contract::{Contract, GamePassKind};
use ferridian_pack_compiler::{SlangCompiler, compile_pack, write_artifact};
use ferridian_testkit::{RenderSpec, RgbaImage, ShaderSpec, require_gpu};
use ferridian_vk_rt::{RuntimeOptions, VkRuntime};

const SIZE: u32 = 128;

/// The unhashed cdylib cargo links into `target/<profile>/` when building
/// this crate's lib target.
fn layer_dylib_path() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let deps_dir = exe.parent().expect("test executable lives in deps/");
    let name = if cfg!(windows) {
        "ferridian_vk_layer.dll"
    } else if cfg!(target_os = "macos") {
        "libferridian_vk_layer.dylib"
    } else {
        "libferridian_vk_layer.so"
    };
    let candidates = [
        deps_dir.join(name),
        deps_dir.parent().expect("deps/ has a parent").join(name),
    ];
    candidates
        .iter()
        .find(|path| path.is_file())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "layer cdylib not found at {} — did the lib target build?",
                candidates[0].display()
            )
        })
}

/// Stage the committed manifest with its library_path rewritten to the
/// freshly built cdylib, and point `VK_LAYER_PATH` at it (preserving every
/// entry of any existing multi-path value — VVL has to stay findable).
fn stage_layer() -> PathBuf {
    let dylib = layer_dylib_path();
    let manifest_dir =
        std::env::temp_dir().join(format!("ferridian-composited-{}", std::process::id()));
    std::fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    let mut manifest: serde_json::Value =
        serde_json::from_str(include_str!("../manifest/VkLayer_FERRIDIAN_overlay.json"))
            .expect("committed manifest is JSON");
    manifest["layer"]["library_path"] =
        serde_json::Value::from(dylib.to_str().expect("dylib path is valid UTF-8"));
    std::fs::write(
        manifest_dir.join("VkLayer_FERRIDIAN_overlay.json"),
        manifest.to_string(),
    )
    .expect("write manifest");
    let system_layers = std::env::var("VK_LAYER_PATH")
        .unwrap_or_else(|_| "/usr/share/vulkan/explicit_layer.d".to_owned());
    let layer_path = std::env::join_paths(
        std::iter::once(manifest_dir.clone()).chain(std::env::split_paths(&system_layers)),
    )
    .expect("join VK_LAYER_PATH entries");
    // SAFETY: nextest gives this test its own process; nothing else is
    // reading the environment concurrently.
    unsafe { std::env::set_var("VK_LAYER_PATH", layer_path) };
    manifest_dir
}

/// Compile and publish the reference pack, and point `FERRIDIAN_PACK` at the
/// artifact — the layer picks it up at device creation.
fn stage_reference_pack() -> PathBuf {
    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");
    let pack_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packs/reference");
    let artifact = compile_pack(&pack_dir, &compiler).expect("reference pack compiles");
    let out =
        std::env::temp_dir().join(format!("ferridian-composited-pack-{}", std::process::id()));
    write_artifact(&artifact, &out).expect("write artifact");
    // SAFETY: as in stage_layer — this process owns its environment.
    unsafe { std::env::set_var(ferridian_vk_layer::PACK_ENV, &out) };
    out
}

fn boot_layered_runtime() -> VkRuntime {
    let layer_name = ferridian_vk_layer::LAYER_NAME
        .to_str()
        .expect("layer name is ASCII")
        .to_owned();
    VkRuntime::new(&RuntimeOptions {
        app_name: "ferridian-composited-test".to_owned(),
        enable_validation: true,
        prefer_software_device: true,
        extra_layers: vec![layer_name],
    })
    .expect("boot lavapipe with the Ferridian layer + validation enabled")
}

/// Render the synthetic game scene (color split + depth plateaus) through
/// the layered chain, optionally inside a labelled debug group.
fn render_scene(runtime: &VkRuntime, spirv: &[u32], pass_label: Option<&str>) -> RgbaImage {
    ferridian_testkit::render_offscreen_with_depth(
        runtime,
        &RenderSpec {
            width: SIZE,
            height: SIZE,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            vertex_count: 3,
            shader: ShaderSpec {
                spirv,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
            },
            pass_label,
        },
    )
}

fn pixel(image: &RgbaImage, x: u32, y: u32) -> [u8; 4] {
    let offset = ((y * SIZE + x) * 4) as usize;
    image.pixels[offset..offset + 4]
        .try_into()
        .expect("4 bytes per pixel")
}

fn mean_luma(image: &RgbaImage, rows: std::ops::Range<u32>) -> f64 {
    let mut sum = 0u64;
    let mut count = 0u64;
    for y in rows {
        for x in 0..SIZE {
            let px = pixel(image, x, y);
            sum += px[0] as u64 + px[1] as u64 + px[2] as u64;
            count += 3;
        }
    }
    sum as f64 / count as f64
}

fn assert_close(actual: [u8; 4], expected: [u8; 4], what: &str) {
    let close = actual
        .iter()
        .zip(&expected)
        .all(|(a, e)| a.abs_diff(*e) <= 1);
    assert!(close, "{what}: got {actual:?}, expected ~{expected:?}");
}

/// The raw scene as the fixture paints it (no pack, no overlay).
fn assert_untouched_scene(image: &RgbaImage, what: &str) {
    assert_close(
        pixel(image, SIZE / 4, SIZE / 4),
        [200, 140, 80, 255],
        &format!("{what}: warm left albedo"),
    );
    assert_close(
        pixel(image, 3 * SIZE / 4, SIZE / 4),
        [80, 120, 200, 255],
        &format!("{what}: cool right albedo"),
    );
    assert_close(
        pixel(image, SIZE / 4, 3 * SIZE / 4),
        [200, 140, 80, 255],
        &format!("{what}: bottom half carries raw albedo, not fog"),
    );
}

#[test]
fn layer_runs_the_reference_pack_over_the_intercepted_frame() {
    require_gpu!();
    let manifest_dir = stage_layer();
    let pack_out = stage_reference_pack();

    let contract = Contract::current();
    let anchor = |kind: GamePassKind| {
        contract
            .passes
            .iter()
            .find(|pass| pass.kind == kind)
            .expect("contract tracks the kind")
            .game_anchor
            .clone()
    };
    let translucent = anchor(GamePassKind::Translucent);
    let terrain = anchor(GamePassKind::Terrain);

    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/scene.slang");
    let scene_spirv = compiler
        .compile_to_spirv(&fixture, "scene")
        .expect("compile scene fixture");

    let runtime = boot_layered_runtime();

    // The world-final pass triggers the pack. Twice: the app destroys and
    // recreates its attachments between frames (as a swapchain resize
    // would), so the second run exercises invalidation + rebuild.
    let composited = render_scene(&runtime, &scene_spirv, Some(&translucent));
    let composited_again = render_scene(&runtime, &scene_spirv, Some(&translucent));
    // A classified pass that is not the world-final one must not trigger —
    // and with a pack armed, the embedded overlay stays out of it too.
    let terrain_pass = render_scene(&runtime, &scene_spirv, Some(&terrain));
    // An Unknown pass is forwarded untouched, always.
    let unlabeled = render_scene(&runtime, &scene_spirv, None);

    assert_eq!(
        composited, composited_again,
        "rebuild after attachment turnover must reproduce the frame exactly"
    );
    assert_untouched_scene(&terrain_pass, "terrain-classified (non-trigger) pass");
    assert_untouched_scene(&unlabeled, "unclassified pass");

    // The pack really ran: full-frame alpha (the composite covered every
    // pixel), and not the raw scene anymore.
    assert!(
        composited.pixels.chunks_exact(4).all(|px| px[3] == 255),
        "composite must cover the frame with alpha 1.0"
    );
    assert_ne!(
        pixel(&composited, SIZE / 4, SIZE / 4),
        pixel(&unlabeled, SIZE / 4, SIZE / 4),
        "the near half must be lit, not raw albedo"
    );

    // Lighting: the albedo split survives in the near (top) half.
    let (mut left, mut right) = (0u64, 0u64);
    for y in 0..SIZE / 2 {
        for x in 0..SIZE {
            let red = pixel(&composited, x, y)[0] as u64;
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

    // Fog: the far (bottom) plateau is transformed away from the near one.
    let near = mean_luma(&composited, 0..8);
    let far = mean_luma(&composited, SIZE - 8..SIZE);
    assert!(
        (far - near).abs() > 25.0,
        "volumetric fog should separate near ({near:.1}) from far ({far:.1}) plateaus"
    );

    // No embedded overlay anywhere: the pack replaces it.
    const OVERLAY_MAGENTA: [u8; 4] = [255, 0, 255, 255];
    assert_ne!(
        pixel(&composited, 2, 2),
        OVERLAY_MAGENTA,
        "a loaded pack must replace the embedded overlay, not stack on it"
    );

    let messages = runtime.validation_messages();
    assert!(
        messages.is_empty(),
        "validation reported {} message(s) with the pack compositing:\n{}",
        messages.len(),
        messages.join("\n")
    );

    drop(runtime);
    std::fs::remove_dir_all(manifest_dir).ok();
    std::fs::remove_dir_all(pack_out).ok();
}
