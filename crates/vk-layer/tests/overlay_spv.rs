//! The committed `shaders/overlay.spv` must be exactly what the pinned
//! slangc produces from `shaders/overlay.slang` — the layer embeds the .spv
//! (no compiler exists inside a game process), so drift between the two
//! would silently ship a stale shader.

use std::path::PathBuf;

use ferridian_testkit::require_gpu;

#[test]
fn committed_overlay_spv_matches_pinned_slangc_output() {
    // Gated with the GPU tests: both need the container's pinned toolchain.
    require_gpu!();
    let shaders = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let compiler = ferridian_pack_compiler::SlangCompiler::from_environment()
        .expect("GPU tests need slangc: the container sets FERRIDIAN_SLANGC");
    let words = compiler
        .compile_to_spirv(&shaders.join("overlay.slang"), "overlay")
        .expect("compile overlay shader");
    let compiled: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
    let committed = std::fs::read(shaders.join("overlay.spv")).expect("read committed overlay.spv");
    assert_eq!(
        compiled, committed,
        "shaders/overlay.spv is stale — regenerate in the container:\n  \
         $FERRIDIAN_SLANGC crates/vk-layer/shaders/overlay.slang -target spirv \
         -profile spirv_1_5 -fvk-use-entrypoint-name -o crates/vk-layer/shaders/overlay.spv"
    );
}
