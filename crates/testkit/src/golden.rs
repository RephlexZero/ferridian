//! Golden-image assertions (§5 step 3 of the plan).
//!
//! Baselines live in `goldens/` at the workspace root (git-lfs) and are only
//! meaningful against the pinned Mesa/lavapipe in `ci/mesa.Dockerfile` —
//! bless from inside the container, never from a desktop GPU.
//!
//! - `FERRIDIAN_BLESS=1` rewrites the baseline instead of comparing.
//! - On mismatch, the actual render and a difference heatmap are written to
//!   `target/golden-failures/` (uploaded as CI artifacts) and the test panics
//!   with the diff stats.

use std::path::{Path, PathBuf};

use crate::image::{DiffPolicy, ImageError, RgbaImage, diff_heatmap, diff_images, diff_passes};

/// `goldens/` at the workspace root.
pub fn goldens_dir() -> PathBuf {
    workspace_root().join("goldens")
}

/// Where failing renders + heatmaps land for CI artifact upload.
pub fn failures_dir() -> PathBuf {
    workspace_root().join("target/golden-failures")
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// True when goldens should be (re)written instead of compared.
pub fn bless_enabled() -> bool {
    std::env::var_os("FERRIDIAN_BLESS").is_some_and(|v| v == "1")
}

/// Compare `actual` against `goldens/<name>.png` under the default policy.
#[track_caller]
pub fn assert_matches_golden(name: &str, actual: &RgbaImage) {
    assert_matches_golden_with(name, actual, &DiffPolicy::default());
}

#[track_caller]
pub fn assert_matches_golden_with(name: &str, actual: &RgbaImage, policy: &DiffPolicy) {
    let golden_path = goldens_dir().join(format!("{name}.png"));
    if bless_enabled() {
        actual
            .write_png(&golden_path)
            .unwrap_or_else(|error| panic!("blessing {name}: {error}"));
        eprintln!("blessed golden {name} -> {}", golden_path.display());
        return;
    }
    let expected = match RgbaImage::read_png(&golden_path) {
        Ok(image) => image,
        Err(ImageError::Io { source, .. }) if source.kind() == std::io::ErrorKind::NotFound => {
            panic!(
                "no golden named {name} ({} does not exist).\n\
                 If this is a new fixture, bless it from inside the pinned Mesa container:\n\
                 mise run golden-bless",
                golden_path.display()
            );
        }
        Err(error) => panic!("loading golden {name}: {error}"),
    };
    let diff = diff_images(&expected, actual, policy)
        .unwrap_or_else(|mismatch| panic!("golden {name}: {mismatch}"));
    if diff_passes(&diff, policy) {
        return;
    }
    let failures = failures_dir();
    let actual_path = failures.join(format!("{name}.actual.png"));
    let heatmap_path = failures.join(format!("{name}.heatmap.png"));
    let mut artifact_note = String::new();
    match actual.write_png(&actual_path) {
        Ok(()) => artifact_note.push_str(&format!("\n  actual:  {}", actual_path.display())),
        Err(error) => artifact_note.push_str(&format!("\n  (failed to write actual: {error})")),
    }
    match diff_heatmap(&expected, actual).write_png(&heatmap_path) {
        Ok(()) => artifact_note.push_str(&format!("\n  heatmap: {}", heatmap_path.display())),
        Err(error) => artifact_note.push_str(&format!("\n  (failed to write heatmap: {error})")),
    }
    panic!(
        "render differs from golden {name}: {differing}/{total} pixels beyond \
         delta {delta} (allowed fraction {allowed}), max channel delta {max}, \
         mean {mean:.4}{artifacts}\n\
         If the change is intended, re-bless inside the pinned container: \
         mise run golden-bless",
        differing = diff.differing_pixels,
        total = diff.total_pixels,
        delta = policy.max_channel_delta,
        allowed = policy.max_differing_fraction,
        max = diff.max_channel_delta,
        mean = diff.mean_channel_delta,
        artifacts = artifact_note,
    );
}
