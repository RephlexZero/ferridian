//! M3 end to end: `packc` compiles the real reference pack with the pinned
//! slangc, publishes it exactly like `packc serve` (artifact files, then the
//! atomic generation bump), the engine's `PackWatcher` picks it up, and every
//! loaded module must be accepted by a real Vulkan device under validation.
//! This is the full author-edit → engine-reload path with no seams mocked.

use std::fs;
use std::path::{Path, PathBuf};

use ash::vk;
use ferridian_engine::pack::PackWatcher;
use ferridian_pack_compiler::{SlangCompiler, compile_pack, write_artifact, write_generation};
use ferridian_testkit::{TestGpu, require_gpu};

fn reference_pack_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../packs/reference")
}

fn scratch_out_dir() -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ferridian-pack-reload-{}", std::process::id()));
    fs::create_dir_all(&dir).expect("create scratch artifact dir");
    dir
}

#[test]
fn reference_pack_round_trips_serve_handshake_onto_a_real_device() {
    require_gpu!();
    let compiler =
        SlangCompiler::from_environment().expect("GPU tests need slangc (FERRIDIAN_SLANGC)");
    let artifact = compile_pack(&reference_pack_dir(), &compiler).expect("reference pack compiles");

    let out = scratch_out_dir();
    write_artifact(&artifact, &out).expect("write artifact");
    write_generation(&out, 1).expect("publish generation");

    let mut watcher = PackWatcher::new(&out);
    let pack = watcher
        .poll()
        .expect("published generation is visible")
        .expect("published artifact loads");
    assert_eq!(pack.generation, 1);
    assert_eq!(
        pack.modules.len(),
        pack.manifest.passes.len(),
        "one SPIR-V module per declared pass"
    );
    assert_eq!(
        pack.execution_order.len(),
        pack.manifest.passes.len(),
        "every pass is schedulable"
    );

    // The point of the exercise: what the watcher hands back must be real
    // shader code a driver accepts, not just well-shaped bytes.
    let mut gpu = TestGpu::new();
    for index in &pack.execution_order {
        let name = &pack.manifest.passes[*index].name;
        let words = &pack.modules[name];
        let info = vk::ShaderModuleCreateInfo::default().code(words);
        // SAFETY: `words` outlives the call; the module is destroyed below on
        // the same device that created it.
        let module = unsafe { gpu.device().create_shader_module(&info, None) }
            .unwrap_or_else(|e| panic!("device rejected reloaded module for pass {name}: {e}"));
        // SAFETY: created just above, never handed anywhere else.
        unsafe { gpu.device().destroy_shader_module(module, None) };
    }
    gpu.assert_no_validation_errors();

    fs::remove_dir_all(&out).ok();
}
