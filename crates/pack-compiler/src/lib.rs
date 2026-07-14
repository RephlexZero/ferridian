//! Pack compilation: Slang sources → SPIR-V + reflection, AOT at pack build
//! time. Errors surface at author time; the engine only ever loads a
//! compiled artifact.

mod serve;
mod slang;

// Reflection moved to `pack-format` (the engine re-reflects artifacts at load
// time); re-exported here so compiler-side callers keep one import path.
pub use ferridian_pack_format::{
    BindingReflection, EntryPointReflection, ShaderReflection, reflect_spirv,
};
pub use serve::{SourceSnapshot, read_generation, write_generation};
pub use slang::{SlangCompiler, resolve_slangc};

use std::fs;
use std::path::{Path, PathBuf};

use ferridian_pack_format::{ManifestError, PackManifest};

#[derive(Debug, thiserror::Error)]
pub enum CompileError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    #[error(transparent)]
    Manifest(#[from] ManifestError),
    #[error(
        "slangc not found: install Slang (mise handles this in the devcontainer) or set FERRIDIAN_SLANGC"
    )]
    SlangcNotFound,
    #[error("slangc failed for pass {pass}:\n{stderr}")]
    Slangc { pass: String, stderr: String },
    #[error("pass {pass} produced invalid SPIR-V: {message}")]
    InvalidSpirv { pass: String, message: String },
}

/// One compiled pass: SPIR-V words plus what we learned from them.
#[derive(Debug)]
pub struct CompiledModule {
    pub pass: String,
    pub spirv: Vec<u32>,
    pub reflection: ShaderReflection,
}

/// A fully compiled pack, ready to be written out as the runtime artifact.
#[derive(Debug)]
pub struct PackArtifact {
    pub manifest: PackManifest,
    pub modules: Vec<CompiledModule>,
}

/// Read + validate `pack.toml` from a pack directory. All manifest I/O lives
/// here so `pack-format` stays pure.
pub fn load_manifest(pack_dir: &Path) -> Result<PackManifest, CompileError> {
    let path = pack_dir.join("pack.toml");
    let text = fs::read_to_string(&path).map_err(|source| CompileError::Io {
        path: path.clone(),
        source,
    })?;
    Ok(PackManifest::from_toml_str(&text)?)
}

/// Validate a pack without compiling: manifest structure + shader files exist.
pub fn validate_pack(pack_dir: &Path) -> Result<PackManifest, CompileError> {
    let manifest = load_manifest(pack_dir)?;
    for pass in &manifest.passes {
        let shader = pack_dir.join(&pass.shader);
        if !shader.is_file() {
            return Err(CompileError::Io {
                path: shader,
                source: std::io::Error::new(std::io::ErrorKind::NotFound, "shader source missing"),
            });
        }
    }
    Ok(manifest)
}

/// Compile every pass of a pack to SPIR-V and reflect the results.
pub fn compile_pack(
    pack_dir: &Path,
    compiler: &SlangCompiler,
) -> Result<PackArtifact, CompileError> {
    let manifest = validate_pack(pack_dir)?;
    let mut modules = Vec::with_capacity(manifest.passes.len());
    for pass in &manifest.passes {
        let source = pack_dir.join(&pass.shader);
        let spirv = compiler.compile_to_spirv(&source, &pass.name)?;
        let reflection = reflect_spirv(&spirv).map_err(|message| CompileError::InvalidSpirv {
            pass: pass.name.clone(),
            message,
        })?;
        tracing::info!(pass = %pass.name, entry_points = reflection.entry_points.len(), "compiled");
        modules.push(CompiledModule {
            pass: pass.name.clone(),
            spirv,
            reflection,
        });
    }
    Ok(PackArtifact { manifest, modules })
}

/// Write an artifact directory: `manifest.toml`, one `.spv` per pass, and a
/// `reflection.toml` describing every module. Signing is a follow-up.
pub fn write_artifact(artifact: &PackArtifact, out_dir: &Path) -> Result<(), CompileError> {
    let io_err = |path: &Path, source| CompileError::Io {
        path: path.to_owned(),
        source,
    };
    fs::create_dir_all(out_dir).map_err(|e| io_err(out_dir, e))?;

    let manifest_path = out_dir.join("manifest.toml");
    fs::write(&manifest_path, artifact.manifest.to_toml_string())
        .map_err(|e| io_err(&manifest_path, e))?;

    let mut reflections = String::new();
    for module in &artifact.modules {
        let spv_path = out_dir.join(format!("{}.spv", module.pass));
        let bytes: Vec<u8> = module.spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        fs::write(&spv_path, bytes).map_err(|e| io_err(&spv_path, e))?;

        reflections.push_str(&format!("[{}]\n", toml_key(&module.pass)));
        reflections.push_str(
            &toml::to_string_pretty(&module.reflection).expect("reflection always serializes"),
        );
        reflections.push('\n');
    }
    let reflection_path = out_dir.join("reflection.toml");
    fs::write(&reflection_path, reflections).map_err(|e| io_err(&reflection_path, e))?;
    Ok(())
}

fn toml_key(raw: &str) -> String {
    if raw
        .chars()
        .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        raw.to_owned()
    } else {
        format!("{raw:?}")
    }
}
