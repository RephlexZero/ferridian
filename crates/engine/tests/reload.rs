//! The consumer half of hot reload against the real producer: artifacts and
//! generations are published with `ferridian-pack-compiler`'s write side, so
//! this test pins the handshake format from both ends.
//!
//! Miri skips this file (real file I/O); the nightly Miri job covers the pure
//! logic in `src/pack.rs` unit tests instead.
#![cfg(not(miri))]

use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};

use ferridian_engine::pack::{PackWatcher, PassModules, ReloadError};
use ferridian_pack_compiler::write_generation;

/// A fresh scratch artifact directory per test (std-only tempdir).
fn scratch_out_dir() -> PathBuf {
    static COUNTER: AtomicU32 = AtomicU32::new(0);
    let dir = std::env::temp_dir().join(format!(
        "ferridian-reload-test-{}-{}",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::Relaxed)
    ));
    fs::create_dir_all(&dir).expect("create scratch artifact dir");
    dir
}

const MANIFEST: &str = r#"
[pack]
name = "reload-test"
version = "0.1.0"

[[pass]]
name = "lighting"
kind = "compute"
shader = "shaders/lighting.slang"
inputs = ["game_color"]
outputs = ["lit"]

[[pass]]
name = "composite"
kind = "graphics"
shader = "shaders/composite.slang"
inputs = ["lit"]
outputs = ["swapchain"]
"#;

/// The smallest byte string the loader accepts as a module: magic + a
/// plausible header. Loader-level validity only — GPU consumption of real
/// slangc output is proven in `ferridian-testkit`'s reload test.
fn fake_spirv(tag: u32) -> Vec<u8> {
    [0x0723_0203u32, 0x0001_0500, 0, 1, 0, tag]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect()
}

fn publish(out: &std::path::Path, generation: u64, tag: u32) {
    fs::write(out.join("manifest.toml"), MANIFEST).unwrap();
    fs::write(out.join("lighting.compute.spv"), fake_spirv(tag)).unwrap();
    fs::write(out.join("composite.vertex.spv"), fake_spirv(tag + 1)).unwrap();
    fs::write(out.join("composite.fragment.spv"), fake_spirv(tag + 1)).unwrap();
    // The real producer: artifact files first, then the atomic bump.
    write_generation(out, generation).unwrap();
}

/// Pull the tag word back out of a loaded pack's module for assertions.
fn tag_of(modules: &PassModules) -> u32 {
    match modules {
        PassModules::Compute { compute } => compute[5],
        PassModules::Graphics { fragment, .. } => fragment[5],
    }
}

#[test]
fn watcher_delivers_each_generation_exactly_once() {
    let out = scratch_out_dir();
    let mut watcher = PackWatcher::new(&out);

    // Nothing published yet.
    assert!(watcher.poll().is_none());

    publish(&out, 1, 100);
    let pack = watcher.poll().expect("generation 1 visible").unwrap();
    assert_eq!(pack.generation, 1);
    assert_eq!(pack.manifest.pack.name, "reload-test");
    // Execution order re-derived from the graph, and both modules decoded.
    assert_eq!(pack.execution_order, vec![0, 1]);
    assert_eq!(pack.modules.len(), 2);
    assert_eq!(tag_of(&pack.modules["lighting"]), 100);
    assert_eq!(tag_of(&pack.modules["composite"]), 101);

    // Same generation → silent.
    assert!(watcher.poll().is_none());
    assert!(watcher.poll().is_none());

    // A rebuild with changed content arrives as the next generation.
    publish(&out, 2, 200);
    let pack = watcher.poll().expect("generation 2 visible").unwrap();
    assert_eq!(pack.generation, 2);
    assert_eq!(tag_of(&pack.modules["lighting"]), 200);
    assert!(watcher.poll().is_none());

    fs::remove_dir_all(&out).ok();
}

#[test]
fn broken_artifact_reports_once_then_recovers_on_next_generation() {
    let out = scratch_out_dir();
    let mut watcher = PackWatcher::new(&out);

    publish(&out, 1, 0);
    watcher.poll().expect("generation 1").unwrap();

    // A corrupt module: reported exactly once, not retried every poll.
    fs::write(out.join("composite.fragment.spv"), &fake_spirv(0)[..9]).unwrap();
    write_generation(&out, 2).unwrap();
    let error = watcher.poll().expect("failure surfaces").unwrap_err();
    assert!(
        matches!(error, ReloadError::TruncatedSpirv { len: 9, .. }),
        "unexpected error: {error}"
    );
    assert!(watcher.poll().is_none(), "failed generation must not spam");

    // The fix ships as generation 3 and loads cleanly.
    publish(&out, 3, 300);
    let pack = watcher.poll().expect("generation 3").unwrap();
    assert_eq!(pack.generation, 3);
    assert_eq!(tag_of(&pack.modules["composite"]), 301);

    fs::remove_dir_all(&out).ok();
}

#[test]
fn hostile_manifest_in_artifact_cannot_escape_the_artifact_dir() {
    let out = scratch_out_dir();
    // Bypass the producer: a downloaded artifact can contain anything.
    fs::write(
        out.join("manifest.toml"),
        MANIFEST.replace("\"composite\"", "\"../../composite\""),
    )
    .unwrap();
    write_generation(&out, 1).unwrap();

    let mut watcher = PackWatcher::new(&out);
    let error = watcher
        .poll()
        .expect("hostile artifact surfaces")
        .unwrap_err();
    assert!(
        matches!(error, ReloadError::Manifest(_)),
        "path-traversal pass name must die in manifest validation, got: {error}"
    );

    fs::remove_dir_all(&out).ok();
}
