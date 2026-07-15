//! The golden-image render harness end to end (M1): compile a Slang fixture
//! with the pinned slangc, render it offscreen on lavapipe under full
//! validation, and compare against the LFS baseline.
//!
//! Baselines are only valid against the pinned Mesa container; see
//! `goldens/README.md`. Bless with `mise run golden-bless`.

use std::path::Path;

use ferridian_pack_compiler::SlangCompiler;
use ferridian_testkit::{RenderSpec, ShaderSpec, TestGpu, require_gpu};

/// Compile a fixture's `vs_main`/`fs_main` entries to their own modules —
/// never one module mixing both, which GPU-assisted validation can't
/// instrument.
fn compile_fixture(name: &str) -> (Vec<u32>, Vec<u32>) {
    let source = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("fixtures/{name}.slang"));
    let compiler = SlangCompiler::from_environment()
        .expect("GPU tests need slangc: the container sets FERRIDIAN_SLANGC");
    let vertex = compiler
        .compile_stage(&source, name, "vs_main", "vertex")
        .unwrap_or_else(|error| panic!("compiling fixture {name} vertex stage: {error}"));
    let fragment = compiler
        .compile_stage(&source, name, "fs_main", "fragment")
        .unwrap_or_else(|error| panic!("compiling fixture {name} fragment stage: {error}"));
    (vertex, fragment)
}

#[test]
fn gradient_matches_golden() {
    require_gpu!();
    let mut gpu = TestGpu::new();
    let (vertex_spirv, fragment_spirv) = compile_fixture("gradient");
    let image = ferridian_testkit::render_offscreen(
        gpu.runtime(),
        &RenderSpec {
            width: 256,
            height: 256,
            clear_color: [0.0, 0.0, 0.0, 1.0],
            vertex_count: 3,
            shader: ShaderSpec {
                vertex_spirv: &vertex_spirv,
                vertex_entry: "vs_main",
                fragment_spirv: &fragment_spirv,
                fragment_entry: "fs_main",
            },
            pass_label: None,
        },
    );
    // The render must be validation-clean *before* it can be golden.
    gpu.assert_no_validation_errors();
    ferridian_testkit::assert_matches_golden("gradient", &image);
}

#[test]
fn clear_color_reaches_readback() {
    require_gpu!();
    let mut gpu = TestGpu::new();
    let (vertex_spirv, fragment_spirv) = compile_fixture("gradient");
    // Zero vertices: nothing is drawn, so every pixel is the clear color —
    // a golden-free sanity check that the readback path reports what the GPU
    // actually did (guards against blessing garbage).
    let image = ferridian_testkit::render_offscreen(
        gpu.runtime(),
        &RenderSpec {
            width: 8,
            height: 8,
            clear_color: [1.0, 0.0, 0.0, 1.0],
            vertex_count: 0,
            shader: ShaderSpec {
                vertex_spirv: &vertex_spirv,
                vertex_entry: "vs_main",
                fragment_spirv: &fragment_spirv,
                fragment_entry: "fs_main",
            },
            pass_label: None,
        },
    );
    gpu.assert_no_validation_errors();
    assert!(
        image
            .pixels
            .chunks_exact(4)
            .all(|px| px == [255, 0, 0, 255]),
        "clear color did not survive the readback path"
    );
}
