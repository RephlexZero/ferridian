//! Pack manifest schema and artifact reflection types.
//!
//! Types only: parsing from strings, structural validation, serde. No file
//! I/O lives here — that keeps the whole crate runnable under Miri and makes
//! the manifest parser a clean fuzz target (`fuzz/fuzz_targets/pack_manifest.rs`),
//! since pack manifests are untrusted input. SPIR-V reflection lives here for
//! the same reason: it is part of the artifact format, both the compiler
//! (producing `reflection.toml`) and the engine (re-reflecting loaded modules
//! to wire descriptors) parse it, and its input is equally untrusted.

pub mod reflection;

pub use reflection::{BindingReflection, EntryPointReflection, ShaderReflection, reflect_spirv};

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Resources the engine provides to every pack; passes may read these without
/// any pack having written them. Mirrors `ferridian_engine::BUILTIN_RESOURCES`.
pub const BUILTIN_RESOURCES: [&str; 3] = ["game_color", "game_depth", "swapchain"];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackManifest {
    pub pack: PackInfo,
    #[serde(default)]
    pub requirements: Requirements,
    #[serde(rename = "pass", default)]
    pub passes: Vec<PassDecl>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PackInfo {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub authors: Vec<String>,
}

/// Capability tier a pack requires (§3.3 of the project plan).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TierRequirement {
    /// Vulkan 1.3 core + portability-subset clean (MoltenVK-safe).
    #[default]
    Baseline,
    /// May use mesh shading / ray query where present.
    Enhanced,
}

#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Requirements {
    #[serde(default)]
    pub tier: TierRequirement,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PassKind {
    Graphics,
    Compute,
}

/// How a pass's sampled input is filtered. Nearest is the default: every
/// same-extent read at texel centers is filter-invariant, and nearest is the
/// only mode universally legal on depth formats (linear filtering of depth
/// is an optional format feature).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Filter {
    #[default]
    Nearest,
    Linear,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PassDecl {
    pub name: String,
    pub kind: PassKind,
    /// Path to the Slang source, relative to the manifest.
    pub shader: String,
    #[serde(default)]
    pub inputs: Vec<String>,
    #[serde(default)]
    pub outputs: Vec<String>,
    /// Sampler filter per input (`filters = { game_color = "linear" }`);
    /// unlisted inputs get [`Filter::Nearest`]. Keys must name declared
    /// inputs of this pass.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub filters: BTreeMap<String, Filter>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ManifestError {
    #[error("manifest is not valid TOML: {0}")]
    Toml(String),
    #[error("pack declares no passes")]
    NoPasses,
    #[error("duplicate pass name: {0}")]
    DuplicatePass(String),
    #[error("pass {pass} input {input:?} is not produced by an earlier pass and is not a builtin")]
    UnknownInput { pass: String, input: String },
    #[error("pass {pass} has an empty shader path")]
    EmptyShaderPath { pass: String },
    #[error(
        "invalid pass name {0:?}: pass names become artifact file names, only [A-Za-z0-9_-] is allowed"
    )]
    InvalidPassName(String),
    #[error("pass {pass} sets a filter for {input:?}, which is not one of its inputs")]
    FilterWithoutInput { pass: String, input: String },
}

impl PackManifest {
    /// Parse and structurally validate a manifest from TOML text.
    pub fn from_toml_str(text: &str) -> Result<PackManifest, ManifestError> {
        let manifest: PackManifest =
            toml::from_str(text).map_err(|e| ManifestError::Toml(e.to_string()))?;
        manifest.validate()?;
        Ok(manifest)
    }

    pub fn to_toml_string(&self) -> String {
        toml::to_string_pretty(self).expect("manifest types always serialize")
    }

    /// Structural checks that need no I/O: pass name uniqueness and that every
    /// input is a builtin or the output of an *earlier* pass (declaration
    /// order is execution order at the manifest level; the engine re-derives
    /// the real schedule from the dependency graph).
    pub fn validate(&self) -> Result<(), ManifestError> {
        if self.passes.is_empty() {
            return Err(ManifestError::NoPasses);
        }
        let mut produced: Vec<&str> = BUILTIN_RESOURCES.to_vec();
        let mut names: Vec<&str> = Vec::new();
        for pass in &self.passes {
            // Pass names name artifact files (`<name>.spv`) and manifests are
            // untrusted, so a name must be a single benign path component —
            // no separators, no "..", no empty string.
            if pass.name.is_empty()
                || !pass
                    .name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
            {
                return Err(ManifestError::InvalidPassName(pass.name.clone()));
            }
            if names.contains(&pass.name.as_str()) {
                return Err(ManifestError::DuplicatePass(pass.name.clone()));
            }
            names.push(&pass.name);
            if pass.shader.trim().is_empty() {
                return Err(ManifestError::EmptyShaderPath {
                    pass: pass.name.clone(),
                });
            }
            for input in &pass.inputs {
                if !produced.contains(&input.as_str()) {
                    return Err(ManifestError::UnknownInput {
                        pass: pass.name.clone(),
                        input: input.clone(),
                    });
                }
            }
            for filtered in pass.filters.keys() {
                if !pass.inputs.contains(filtered) {
                    return Err(ManifestError::FilterWithoutInput {
                        pass: pass.name.clone(),
                        input: filtered.clone(),
                    });
                }
            }
            produced.extend(pass.outputs.iter().map(String::as_str));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINIMAL: &str = r#"
        [pack]
        name = "reference"
        version = "0.1.0"
        authors = ["Ferridian contributors"]

        [requirements]
        tier = "baseline"

        [[pass]]
        name = "composite"
        kind = "graphics"
        shader = "shaders/composite.slang"
        inputs = ["game_color"]
        outputs = ["swapchain"]
    "#;

    #[test]
    fn parses_minimal_manifest() {
        let manifest = PackManifest::from_toml_str(MINIMAL).unwrap();
        assert_eq!(manifest.pack.name, "reference");
        assert_eq!(manifest.requirements.tier, TierRequirement::Baseline);
        assert_eq!(manifest.passes.len(), 1);
    }

    #[test]
    fn round_trips() {
        let manifest = PackManifest::from_toml_str(MINIMAL).unwrap();
        let text = manifest.to_toml_string();
        assert_eq!(PackManifest::from_toml_str(&text).unwrap(), manifest);
    }

    #[test]
    fn rejects_unknown_input() {
        let bad = MINIMAL.replace("game_color", "nonexistent");
        assert!(matches!(
            PackManifest::from_toml_str(&bad),
            Err(ManifestError::UnknownInput { .. })
        ));
    }

    #[test]
    fn rejects_duplicate_pass() {
        let manifest = PackManifest::from_toml_str(MINIMAL).unwrap();
        let mut dup = manifest.clone();
        dup.passes.push(manifest.passes[0].clone());
        assert_eq!(
            dup.validate(),
            Err(ManifestError::DuplicatePass("composite".to_owned()))
        );
    }

    #[test]
    fn rejects_path_traversal_pass_names() {
        for hostile in ["../evil", "a/b", "a\\b", "", ".."] {
            let mut manifest = PackManifest::from_toml_str(MINIMAL).unwrap();
            manifest.passes[0].name = hostile.to_owned();
            assert_eq!(
                manifest.validate(),
                Err(ManifestError::InvalidPassName(hostile.to_owned())),
                "pass name {hostile:?} must be rejected"
            );
        }
    }

    #[test]
    fn parses_filters_and_rejects_ones_naming_no_input() {
        let filtered = MINIMAL.replace(
            "inputs = [\"game_color\"]",
            "inputs = [\"game_color\"]\nfilters = { game_color = \"linear\" }",
        );
        let manifest = PackManifest::from_toml_str(&filtered).unwrap();
        assert_eq!(
            manifest.passes[0].filters.get("game_color"),
            Some(&Filter::Linear)
        );
        // Round-trips through serialization like every other field.
        let text = manifest.to_toml_string();
        assert_eq!(PackManifest::from_toml_str(&text).unwrap(), manifest);

        let dangling = MINIMAL.replace(
            "inputs = [\"game_color\"]",
            "inputs = [\"game_color\"]\nfilters = { game_depth = \"nearest\" }",
        );
        assert_eq!(
            PackManifest::from_toml_str(&dangling),
            Err(ManifestError::FilterWithoutInput {
                pass: "composite".to_owned(),
                input: "game_depth".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_unknown_fields() {
        let bad = format!("{MINIMAL}\n[surprise]\nkey = 1\n");
        assert!(matches!(
            PackManifest::from_toml_str(&bad),
            Err(ManifestError::Toml(_))
        ));
    }

    #[test]
    fn allows_pass_reading_earlier_output() {
        let two = format!(
            "{MINIMAL}\n[[pass]]\nname = \"post\"\nkind = \"graphics\"\nshader = \"shaders/post.slang\"\ninputs = [\"swapchain\"]\noutputs = []\n"
        );
        PackManifest::from_toml_str(&two).unwrap();
    }
}
