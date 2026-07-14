//! M2 groundwork, proven end to end: the built layer cdylib is discovered by
//! the real Vulkan loader (via a generated manifest + `VK_LAYER_PATH`),
//! chains instance/device creation down to lavapipe underneath the Khronos
//! validation layer with zero validation messages, and actually intercepts
//! `vkCmdBeginRenderPass` (asserted by reading the layer's exported counter
//! across the cdylib boundary).
//!
//! nextest runs each test in its own process, so setting env vars here is
//! safe.

use std::path::PathBuf;

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

#[test]
fn layer_loads_intercepts_and_stays_validation_clean() {
    require_gpu!();

    // Stage the *committed* manifest (the packaging artifact), with its
    // library_path rewritten to the freshly built cdylib.
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
    let system_layers = std::env::var("VK_LAYER_PATH")
        .unwrap_or_else(|_| "/usr/share/vulkan/explicit_layer.d".to_owned());
    let layer_path = std::env::join_paths([manifest_dir.as_path(), system_layers.as_ref()])
        .expect("join VK_LAYER_PATH entries");
    // SAFETY: nextest gives this test its own process; nothing else is
    // reading the environment concurrently.
    unsafe { std::env::set_var("VK_LAYER_PATH", layer_path) };

    let layer_name = ferridian_vk_layer::LAYER_NAME
        .to_str()
        .expect("layer name is ASCII")
        .to_owned();
    let runtime = VkRuntime::new(&RuntimeOptions {
        app_name: "ferridian-layer-test".to_owned(),
        enable_validation: true,
        prefer_software_device: true,
        extra_layers: vec![layer_name],
    })
    .expect("boot lavapipe with the Ferridian layer + validation enabled");

    // Drive a real draw through the layered dispatch chain.
    let fixture =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../testkit/fixtures/gradient.slang");
    let compiler = ferridian_pack_compiler::SlangCompiler::from_environment()
        .expect("GPU tests need slangc: the container sets FERRIDIAN_SLANGC");
    let spirv = compiler
        .compile_to_spirv(&fixture, "gradient")
        .expect("compile gradient fixture");
    let image = ferridian_testkit::render_offscreen(
        &runtime,
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
        },
    );
    assert_eq!(image.pixels.len(), 64 * 64 * 4);

    // The loader loaded its own copy of the cdylib; dlopen the same file to
    // reach that copy's exported counter (dlopen refcounts, same handle).
    // SAFETY: the layer cdylib is already resident and its init is trivial.
    let library = unsafe { libloading::Library::new(&dylib) }.expect("dlopen the layer cdylib");
    // SAFETY: the symbol is defined by this crate with exactly this type.
    let count: libloading::Symbol<'_, unsafe extern "system" fn() -> u64> =
        unsafe { library.get(b"ferridian_layer_render_pass_count") }
            .expect("layer exports its render pass counter");
    // SAFETY: trivial exported getter.
    let passes = unsafe { count() };
    assert!(
        passes >= 1,
        "the layer sits in the dispatch chain but never saw vkCmdBeginRenderPass"
    );

    let messages = runtime.validation_messages();
    assert!(
        messages.is_empty(),
        "validation reported {} message(s) with the layer active:\n{}",
        messages.len(),
        messages.join("\n")
    );

    drop(runtime);
    std::fs::remove_dir_all(manifest_dir).ok();
}
