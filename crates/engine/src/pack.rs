//! Consuming compiled pack artifacts, including hot reload (M3).
//!
//! The producer half lives in `ferridian-pack-compiler::serve`: every
//! successful `packc serve` rebuild writes the artifact files and *then*
//! atomically renames a new monotonic value into `<out>/generation`. This
//! module is the consumer: [`PackWatcher::poll`] notices a generation bump,
//! loads the artifact directory (manifest + one SPIR-V module per pass), and
//! re-derives the execution schedule from the dependency graph.
//!
//! Artifacts are untrusted input (packs get downloaded), so loading validates
//! everything it touches: the manifest structurally (via `pack-format`), the
//! graph for cycles/conflicts, and each module for SPIR-V shape. File I/O
//! lives only in [`load_pack`]/[`PackWatcher`]; unit tests stay pure so the
//! crate remains Miri-able (the I/O paths are covered by `tests/reload.rs`,
//! which Miri skips).

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use ferridian_pack_format::{ManifestError, PackManifest, PassKind as ManifestPassKind};

use crate::{GraphError, PassGraph, PassKind, PassNode, ResourceId};

/// First word of every valid SPIR-V module (little-endian on disk).
const SPIRV_MAGIC: u32 = 0x0723_0203;

#[derive(Debug, thiserror::Error)]
pub enum ReloadError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(transparent)]
    Graph(#[from] GraphError),
    #[error("pass {pass}: module is {len} bytes, not a whole number of SPIR-V words")]
    TruncatedSpirv { pass: String, len: usize },
    #[error("pass {pass}: module does not start with the SPIR-V magic number")]
    BadSpirvMagic { pass: String },
}

/// A fully loaded pack artifact, ready to hand to the render graph.
#[derive(Debug)]
pub struct LoadedPack {
    /// The `generation` value this load corresponds to.
    pub generation: u64,
    pub manifest: PackManifest,
    /// Indices into `manifest.passes` in execution order (writers before
    /// readers) — re-derived from the graph, not manifest declaration order.
    pub execution_order: Vec<usize>,
    /// SPIR-V words per pass name.
    pub modules: BTreeMap<String, Vec<u32>>,
}

/// Build the engine pass graph from a validated manifest.
pub fn graph_from_manifest(manifest: &PackManifest) -> PassGraph {
    let mut graph = PassGraph::new();
    for pass in &manifest.passes {
        graph.add_pass(PassNode {
            name: pass.name.clone(),
            kind: match pass.kind {
                ManifestPassKind::Graphics => PassKind::Graphics,
                ManifestPassKind::Compute => PassKind::Compute,
            },
            reads: pass.inputs.iter().cloned().map(ResourceId).collect(),
            writes: pass.outputs.iter().cloned().map(ResourceId).collect(),
        });
    }
    graph
}

/// Decode an on-disk SPIR-V module into words, rejecting anything that is
/// not plausibly SPIR-V before it gets near a driver.
pub fn words_from_spirv_bytes(pass: &str, bytes: &[u8]) -> Result<Vec<u32>, ReloadError> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err(ReloadError::TruncatedSpirv {
            pass: pass.to_owned(),
            len: bytes.len(),
        });
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if words[0] != SPIRV_MAGIC {
        return Err(ReloadError::BadSpirvMagic {
            pass: pass.to_owned(),
        });
    }
    Ok(words)
}

/// Load one artifact directory as published by `packc build`/`serve`:
/// `manifest.toml` plus `<pass>.spv` per declared pass.
pub fn load_pack(out_dir: &Path, generation: u64) -> Result<LoadedPack, ReloadError> {
    let io_err = |path: &Path, source| ReloadError::Io {
        path: path.to_owned(),
        source,
    };
    let manifest_path = out_dir.join("manifest.toml");
    let text = fs::read_to_string(&manifest_path).map_err(|e| io_err(&manifest_path, e))?;
    // `from_toml_str` validates structurally, including that pass names are
    // benign single path components — which is what makes the join below safe
    // on an untrusted artifact.
    let manifest = PackManifest::from_toml_str(&text)?;
    let execution_order = graph_from_manifest(&manifest).execution_order()?;

    let mut modules = BTreeMap::new();
    for pass in &manifest.passes {
        let spv_path = out_dir.join(format!("{}.spv", pass.name));
        let bytes = fs::read(&spv_path).map_err(|e| io_err(&spv_path, e))?;
        modules.insert(
            pass.name.clone(),
            words_from_spirv_bytes(&pass.name, &bytes)?,
        );
    }
    Ok(LoadedPack {
        generation,
        manifest,
        execution_order,
        modules,
    })
}

/// Read the published generation, `None` if never published (or unreadable —
/// both mean "artifact not ready yet"). Format contract: the decimal value
/// written by `ferridian-pack-compiler::serve::write_generation`; the
/// `tests/reload.rs` round-trip pins the two ends together.
fn read_generation(out_dir: &Path) -> Option<u64> {
    fs::read_to_string(out_dir.join("generation"))
        .ok()?
        .trim()
        .parse()
        .ok()
}

/// Polls an artifact directory and reloads when the producer publishes a new
/// generation. One `poll` call per frame (or timer tick) is the intended use.
#[derive(Debug)]
pub struct PackWatcher {
    out_dir: PathBuf,
    last_seen: Option<u64>,
}

impl PackWatcher {
    pub fn new(out_dir: impl Into<PathBuf>) -> PackWatcher {
        PackWatcher {
            out_dir: out_dir.into(),
            last_seen: None,
        }
    }

    /// `None` when nothing new is published; `Some(result)` exactly once per
    /// generation, whether it loaded or failed (a failed generation is not
    /// retried — the producer only ever moves forward, so the fix arrives as
    /// the *next* generation).
    pub fn poll(&mut self) -> Option<Result<LoadedPack, ReloadError>> {
        let generation = read_generation(&self.out_dir)?;
        if Some(generation) == self.last_seen {
            return None;
        }
        let loaded = load_pack(&self.out_dir, generation);
        // Torn-read guard: if the producer republished while we were reading,
        // the files we read may straddle two builds. Discard and let the next
        // poll settle on the newer generation.
        if read_generation(&self.out_dir) != Some(generation) {
            return None;
        }
        self.last_seen = Some(generation);
        match &loaded {
            Ok(pack) => tracing::info!(
                generation,
                passes = pack.manifest.passes.len(),
                "pack reloaded"
            ),
            Err(error) => tracing::warn!(generation, %error, "pack reload failed"),
        }
        Some(loaded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn manifest(passes: &str) -> PackManifest {
        PackManifest::from_toml_str(&format!(
            "[pack]\nname = \"t\"\nversion = \"0.1.0\"\n{passes}"
        ))
        .expect("test manifest is valid")
    }

    #[test]
    fn graph_reorders_manifest_declaration_order() {
        // Declared consumer-first: manifest validation only sees builtins as
        // inputs here, but the graph must still schedule the writer first.
        let manifest = manifest(
            "[[pass]]\nname = \"lighting\"\nkind = \"compute\"\nshader = \"a.slang\"\n\
             inputs = [\"game_color\"]\noutputs = [\"lit\"]\n\
             [[pass]]\nname = \"composite\"\nkind = \"graphics\"\nshader = \"b.slang\"\n\
             inputs = [\"lit\"]\noutputs = [\"swapchain\"]\n",
        );
        let order = graph_from_manifest(&manifest).execution_order().unwrap();
        assert_eq!(order, vec![0, 1]);
        let graph = graph_from_manifest(&manifest);
        assert_eq!(graph.passes()[0].kind, PassKind::Compute);
        assert_eq!(graph.passes()[1].kind, PassKind::Graphics);
    }

    #[test]
    fn spirv_decoding_validates_shape() {
        let good: Vec<u8> = [SPIRV_MAGIC, 0x0001_0500, 0, 1, 0]
            .iter()
            .flat_map(|w: &u32| w.to_le_bytes())
            .collect();
        let words = words_from_spirv_bytes("p", &good).unwrap();
        assert_eq!(words[0], SPIRV_MAGIC);
        assert_eq!(words.len(), 5);

        assert!(matches!(
            words_from_spirv_bytes("p", &good[..7]),
            Err(ReloadError::TruncatedSpirv { len: 7, .. })
        ));
        assert!(matches!(
            words_from_spirv_bytes("p", &[]),
            Err(ReloadError::TruncatedSpirv { len: 0, .. })
        ));
        assert!(matches!(
            words_from_spirv_bytes("p", &[0, 0, 0, 0]),
            Err(ReloadError::BadSpirvMagic { .. })
        ));
    }
}
