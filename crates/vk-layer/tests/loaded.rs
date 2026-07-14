//! M2, proven end to end on the real Vulkan loader: the built layer cdylib
//! is discovered via a staged manifest + `VK_LAYER_PATH`, chains
//! instance/device creation down to lavapipe underneath the Khronos
//! validation layer with zero validation messages, intercepts
//! `vkCmdBeginRenderPass` and the debug-utils label commands, classifies
//! passes against the contract's anchors, and keeps per-object dispatch
//! state correct across multiple concurrent instances/devices.
//!
//! nextest runs each test in its own process, so setting env vars here is
//! safe — and the layer's process-global counters start at zero per test.

use std::path::PathBuf;

use ferridian_contract::{Contract, GamePassKind};
use ferridian_engine::frame::kind_index;
use ferridian_testkit::{RenderSpec, ShaderSpec, require_gpu};
use ferridian_vk_rt::{RuntimeOptions, VkRuntime};

/// The unhashed cdylib cargo links into `target/<profile>/` when building
/// this crate's lib target (which it must, for this test to exist).
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
    // The unhashed cdylib in deps/ is the freshly built one; the copy one
    // level up is a hardlink cargo does not always refresh for test builds.
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

/// Stage the *committed* manifest (the packaging artifact) with its
/// library_path rewritten to the freshly built cdylib, and point
/// `VK_LAYER_PATH` at it. Returns (manifest dir, cdylib path).
fn stage_layer() -> (PathBuf, PathBuf) {
    let dylib = layer_dylib_path();
    let manifest_dir = std::env::temp_dir().join(format!("ferridian-layer-{}", std::process::id()));
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
    // VK_LAYER_PATH *replaces* the loader's explicit-layer search path, so it
    // must also cover wherever the validation layer lives (any existing
    // setting, else the standard system dir the CI image installs VVL into).
    // The existing setting may itself already be a multi-entry PATH-style
    // list, so split it back into individual paths before rejoining — passing
    // it through as one opaque entry makes join_paths reject the embedded
    // separator.
    let system_layers = std::env::var("VK_LAYER_PATH")
        .unwrap_or_else(|_| "/usr/share/vulkan/explicit_layer.d".to_owned());
    let layer_path = std::env::join_paths(
        std::iter::once(manifest_dir.clone()).chain(std::env::split_paths(&system_layers)),
    )
    .expect("join VK_LAYER_PATH entries");
    // SAFETY: nextest gives this test its own process; nothing else is
    // reading the environment concurrently.
    unsafe { std::env::set_var("VK_LAYER_PATH", layer_path) };
    (manifest_dir, dylib)
}

fn boot_layered_runtime() -> VkRuntime {
    let layer_name = ferridian_vk_layer::LAYER_NAME
        .to_str()
        .expect("layer name is ASCII")
        .to_owned();
    VkRuntime::new(&RuntimeOptions {
        app_name: "ferridian-layer-test".to_owned(),
        enable_validation: true,
        prefer_software_device: true,
        extra_layers: vec![layer_name],
    })
    .expect("boot lavapipe with the Ferridian layer + validation enabled")
}

/// Drive a real draw through the layered dispatch chain and return the
/// readback.
fn render_gradient(runtime: &VkRuntime, pass_label: Option<&str>) -> ferridian_testkit::RgbaImage {
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/gradient.slang");
    let compiler = ferridian_pack_compiler::SlangCompiler::from_environment()
        .expect("GPU tests need slangc: the container sets FERRIDIAN_SLANGC");
    let spirv = compiler
        .compile_to_spirv(&fixture, "gradient")
        .expect("compile gradient fixture");
    let image = ferridian_testkit::render_offscreen(
        runtime,
        &RenderSpec {
            width: 64,
            height: 64,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            vertex_count: 3,
            shader: ShaderSpec {
                spirv: &spirv,
                vertex_entry: "vs_main",
                fragment_entry: "fs_main",
            },
            pass_label,
        },
    );
    assert_eq!(image.pixels.len(), 64 * 64 * 4);
    image
}

/// One RGBA pixel out of a 64×64 readback.
fn pixel(image: &ferridian_testkit::RgbaImage, x: usize, y: usize) -> [u8; 4] {
    let offset = (y * 64 + x) * 4;
    image.pixels[offset..offset + 4]
        .try_into()
        .expect("4 bytes per pixel")
}

/// The overlay shader's solid fill — see crates/vk-layer/shaders/overlay.slang.
const OVERLAY_MAGENTA: [u8; 4] = [255, 0, 255, 255];

/// The loader loaded its own copy of the cdylib; dlopen the same file to
/// reach that copy's exported counters (dlopen refcounts, same handle).
struct LayerProbe {
    library: libloading::Library,
}

impl LayerProbe {
    fn new(dylib: &PathBuf) -> LayerProbe {
        // SAFETY: the layer cdylib is already resident and its init is trivial.
        let library = unsafe { libloading::Library::new(dylib) }.expect("dlopen the layer cdylib");
        LayerProbe { library }
    }

    fn render_pass_count(&self) -> u64 {
        // SAFETY: the symbol is defined by this crate with exactly this type;
        // it is a trivial exported getter.
        unsafe {
            let count: libloading::Symbol<'_, unsafe extern "system" fn() -> u64> = self
                .library
                .get(b"ferridian_layer_render_pass_count")
                .expect("layer exports its render pass counter");
            count()
        }
    }

    fn classified_count(&self, kind: GamePassKind) -> u64 {
        let index = u32::try_from(kind_index(kind)).expect("kind index fits u32");
        // SAFETY: as above.
        unsafe {
            let count: libloading::Symbol<'_, unsafe extern "system" fn(u32) -> u64> = self
                .library
                .get(b"ferridian_layer_classified_pass_count")
                .expect("layer exports its classified pass counter");
            count(index)
        }
    }
}

fn assert_validation_clean(runtime: &VkRuntime, context: &str) {
    let messages = runtime.validation_messages();
    assert!(
        messages.is_empty(),
        "validation reported {} message(s) {context}:\n{}",
        messages.len(),
        messages.join("\n")
    );
}

#[test]
fn layer_loads_intercepts_and_classifies_validation_clean() {
    require_gpu!();
    let (manifest_dir, dylib) = stage_layer();

    // Label the draw's render pass with the contract's terrain anchor — the
    // stand-in for Blaze3D's debug group around its terrain pass.
    let contract = Contract::current();
    let terrain_anchor = &contract
        .passes
        .iter()
        .find(|pass| pass.kind == GamePassKind::Terrain)
        .expect("contract tracks a terrain pass")
        .game_anchor;

    let runtime = boot_layered_runtime();
    let classified = render_gradient(&runtime, Some(terrain_anchor));
    let unclassified = render_gradient(&runtime, None);

    let probe = LayerProbe::new(&dylib);
    assert!(
        probe.render_pass_count() >= 2,
        "the layer sits in the dispatch chain but never saw vkCmdBeginRenderPass"
    );
    assert!(
        probe.classified_count(GamePassKind::Terrain) >= 1,
        "the labelled pass was not classified as terrain (label interception broken?)"
    );
    assert_eq!(
        probe.classified_count(GamePassKind::Sky),
        0,
        "no sky-anchored label was ever pushed"
    );

    // Composite over the intercepted frame: the classified pass carries the
    // layer's magenta corner overlay; the unclassified pass is untouched.
    assert_eq!(
        pixel(&classified, 2, 2),
        OVERLAY_MAGENTA,
        "the terrain-classified pass should carry the overlay in its corner"
    );
    assert_ne!(
        pixel(&classified, 60, 60),
        OVERLAY_MAGENTA,
        "the overlay must stay in its corner, not repaint the frame"
    );
    assert_ne!(
        pixel(&unclassified, 2, 2),
        OVERLAY_MAGENTA,
        "an Unknown pass must be forwarded untouched — unknown never means composite"
    );

    assert_validation_clean(&runtime, "with the layer active");
    drop(runtime);
    std::fs::remove_dir_all(manifest_dir).ok();
}

#[test]
fn per_object_dispatch_survives_two_concurrent_runtimes() {
    require_gpu!();
    let (manifest_dir, dylib) = stage_layer();

    // Two full instance+device stacks alive at once: single-slot layer state
    // would cross their down-chain pointers or lose one of them.
    let first = boot_layered_runtime();
    let second = boot_layered_runtime();
    render_gradient(&first, None);
    render_gradient(&second, None);

    // Destroying one stack must tear down only its own dispatch entries…
    assert_validation_clean(&first, "on the first runtime");
    drop(first);

    // …leaving the survivor fully routable.
    render_gradient(&second, None);
    assert_validation_clean(
        &second,
        "on the second runtime after the first was destroyed",
    );

    let probe = LayerProbe::new(&dylib);
    assert!(
        probe.render_pass_count() >= 3,
        "expected all three draws to route through the layer, saw {}",
        probe.render_pass_count()
    );

    drop(second);
    std::fs::remove_dir_all(manifest_dir).ok();
}
