//! The committed `shaders/overlay.{vertex,fragment}.spv` must be exactly
//! what the pinned slangc produces from `shaders/overlay.slang` — the layer
//! embeds the .spv (no compiler exists inside a game process), so drift
//! between the two would silently ship a stale shader. Two modules, never
//! one mixing both entry points — that's exactly the shape GPU-assisted
//! validation can't instrument.

use std::path::PathBuf;

use ferridian_testkit::require_gpu;

#[test]
fn committed_overlay_spv_matches_pinned_slangc_output() {
    // Gated with the GPU tests: both need the container's pinned toolchain.
    require_gpu!();
    let shaders = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
    let compiler = ferridian_pack_compiler::SlangCompiler::from_environment()
        .expect("GPU tests need slangc: the container sets FERRIDIAN_SLANGC");
    let source = shaders.join("overlay.slang");
    for (entry, stage) in [("vs_main", "vertex"), ("fs_main", "fragment")] {
        let words = compiler
            .compile_stage(&source, "overlay", entry, stage)
            .unwrap_or_else(|error| panic!("compile overlay {stage} stage: {error}"));
        let compiled: Vec<u8> = words.iter().flat_map(|word| word.to_le_bytes()).collect();
        let committed_path = shaders.join(format!("overlay.{stage}.spv"));
        let committed = std::fs::read(&committed_path)
            .unwrap_or_else(|error| panic!("read {}: {error}", committed_path.display()));
        assert_eq!(
            compiled, committed,
            "shaders/overlay.{stage}.spv is stale — regenerate in the container:\n  \
             $FERRIDIAN_SLANGC crates/vk-layer/shaders/overlay.slang -target spirv \
             -profile spirv_1_5 -fvk-use-entrypoint-name -entry {entry} -stage {stage} \
             -o crates/vk-layer/shaders/overlay.{stage}.spv"
        );
    }
}
