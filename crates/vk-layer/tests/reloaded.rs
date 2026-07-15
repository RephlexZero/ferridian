//! Layer-side hot reload, end to end on the real Vulkan loader: the layer
//! composites generation 1 of the reference pack, an "author edit" publishes
//! generation 2 exactly like `packc serve` (artifact files, then the atomic
//! generation bump), and the *running* layer swaps compositors between
//! frames — the next composite comes out visibly transformed. A corrupt
//! generation 3 then proves a broken publish keeps the previous pack
//! running instead of taking compositing (or the game) down.
//!
//! nextest runs each test in its own process, so setting env vars here is
//! safe. The staging helpers mirror `composited.rs` (test binaries don't
//! share modules).

use std::fs;
use std::path::PathBuf;

use ferridian_contract::{Contract, GamePassKind};
use ferridian_pack_compiler::{SlangCompiler, compile_pack, write_artifact, write_generation};
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
        std::env::temp_dir().join(format!("ferridian-reloaded-{}", std::process::id()));
    fs::create_dir_all(&manifest_dir).expect("create manifest dir");
    let mut manifest: serde_json::Value =
        serde_json::from_str(include_str!("../manifest/VkLayer_FERRIDIAN_overlay.json"))
            .expect("committed manifest is JSON");
    manifest["layer"]["library_path"] =
        serde_json::Value::from(dylib.to_str().expect("dylib path is valid UTF-8"));
    fs::write(
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

fn reference_pack_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../packs/reference")
}

/// Copy the reference pack's sources and invert the composite's final color
/// — the smallest author edit with a frame-wide, arithmetically checkable
/// effect (v2 = 1 − v1, channel by channel).
fn edited_pack_dir() -> PathBuf {
    let src = reference_pack_dir();
    let dst = std::env::temp_dir().join(format!("ferridian-reloaded-src-{}", std::process::id()));
    let shaders = dst.join("shaders");
    fs::create_dir_all(&shaders).expect("create edited pack dir");
    fs::copy(src.join("pack.toml"), dst.join("pack.toml")).expect("copy pack.toml");
    for entry in fs::read_dir(src.join("shaders")).expect("list reference shaders") {
        let entry = entry.expect("read dir entry");
        fs::copy(entry.path(), shaders.join(entry.file_name())).expect("copy shader");
    }
    let composite_path = shaders.join("composite.slang");
    let composite = fs::read_to_string(&composite_path).expect("read composite shader");
    let before = "return float4(color, 1.0);";
    assert!(
        composite.contains(before),
        "composite.slang no longer ends with the expected return — update this test's edit"
    );
    fs::write(
        &composite_path,
        composite.replace(before, "return float4(1.0 - color, 1.0);"),
    )
    .expect("write edited composite shader");
    dst
}

fn boot_layered_runtime() -> VkRuntime {
    let layer_name = ferridian_vk_layer::LAYER_NAME
        .to_str()
        .expect("layer name is ASCII")
        .to_owned();
    VkRuntime::new(&RuntimeOptions {
        app_name: "ferridian-reloaded-test".to_owned(),
        enable_validation: true,
        prefer_software_device: true,
        extra_layers: vec![layer_name],
    })
    .expect("boot lavapipe with the Ferridian layer + validation enabled")
}

/// Render the synthetic game scene (color split + depth plateaus) through
/// the layered chain inside the world-final debug group.
fn render_frame(runtime: &VkRuntime, spirv: &[u32], pass_label: &str) -> RgbaImage {
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
            pass_label: Some(pass_label),
        },
    )
}

#[test]
fn layer_hot_swaps_the_pack_when_a_new_generation_is_published() {
    require_gpu!();
    let manifest_dir = stage_layer();

    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");

    // Generation 1: the pristine reference pack, published like packc serve.
    let artifact = compile_pack(&reference_pack_dir(), &compiler).expect("reference pack compiles");
    let out = std::env::temp_dir().join(format!("ferridian-reloaded-pack-{}", std::process::id()));
    write_artifact(&artifact, &out).expect("write artifact");
    write_generation(&out, 1).expect("publish generation 1");
    // SAFETY: as in stage_layer — this process owns its environment.
    unsafe { std::env::set_var(ferridian_vk_layer::PACK_ENV, &out) };

    let translucent = Contract::current()
        .passes
        .iter()
        .find(|pass| pass.kind == GamePassKind::Translucent)
        .expect("contract tracks the translucent pass")
        .game_anchor
        .clone();
    let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/scene.slang");
    let scene_spirv = compiler
        .compile_to_spirv(&fixture, "scene")
        .expect("compile scene fixture");

    let runtime = boot_layered_runtime();
    let v1 = render_frame(&runtime, &scene_spirv, &translucent);

    // The author edit: generation 2 inverts the composite. Files first, then
    // the atomic generation bump — the exact packc-serve handshake.
    let edited = edited_pack_dir();
    let artifact = compile_pack(&edited, &compiler).expect("edited pack compiles");
    write_artifact(&artifact, &out).expect("write edited artifact");
    write_generation(&out, 2).expect("publish generation 2");

    let v2 = render_frame(&runtime, &scene_spirv, &translucent);
    let v2_again = render_frame(&runtime, &scene_spirv, &translucent);
    assert_eq!(
        v2, v2_again,
        "the swapped-in pack must be deterministic across frames"
    );

    // Every pixel of v2 must be the inversion of v1 (±1 for UNORM rounding:
    // round(255·(1−c)) and 255−round(255·c) can differ by one).
    let mismatch = v1
        .pixels
        .chunks_exact(4)
        .zip(v2.pixels.chunks_exact(4))
        .position(|(a, b)| {
            a[..3]
                .iter()
                .zip(&b[..3])
                .any(|(&one, &two)| (255 - one).abs_diff(two) > 1)
                || b[3] != 255
        });
    assert_eq!(
        mismatch, None,
        "generation 2 must invert generation 1 frame-wide (first bad pixel index: {mismatch:?})"
    );

    // Generation 3 is corrupt on disk: the reload must fail loudly in the
    // logs but keep generation 2 compositing — a broken publish never takes
    // the running pack (or the game) down.
    fs::write(out.join("composite.spv"), b"not spir-v").expect("corrupt module");
    write_generation(&out, 3).expect("publish generation 3");
    let v3 = render_frame(&runtime, &scene_spirv, &translucent);
    assert_eq!(
        v3, v2,
        "a reload that fails to load must keep the previous compositor running"
    );

    let messages = runtime.validation_messages();
    assert!(
        messages.is_empty(),
        "validation reported {} message(s) across the hot swap:\n{}",
        messages.len(),
        messages.join("\n")
    );

    drop(runtime);
    fs::remove_dir_all(manifest_dir).ok();
    fs::remove_dir_all(out).ok();
    fs::remove_dir_all(edited).ok();
}
