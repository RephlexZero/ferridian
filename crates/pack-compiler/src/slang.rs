//! Invocation of the pinned `slangc` binary. The compiler itself is a tool
//! dependency (mise / the CI image), not a crate dependency.

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

use crate::CompileError;

/// Locate `slangc`: explicit path, `FERRIDIAN_SLANGC`, then `$PATH`.
pub fn resolve_slangc(explicit: Option<PathBuf>) -> Result<PathBuf, CompileError> {
    if let Some(path) = explicit {
        return Ok(path);
    }
    if let Some(path) = env::var_os("FERRIDIAN_SLANGC") {
        return Ok(PathBuf::from(path));
    }
    let exe = if cfg!(windows) {
        "slangc.exe"
    } else {
        "slangc"
    };
    env::var_os("PATH")
        .and_then(|paths| {
            env::split_paths(&paths)
                .map(|dir| dir.join(exe))
                .find(|candidate| candidate.is_file())
        })
        .ok_or(CompileError::SlangcNotFound)
}

#[derive(Debug)]
pub struct SlangCompiler {
    slangc: PathBuf,
}

impl SlangCompiler {
    pub fn new(slangc: PathBuf) -> SlangCompiler {
        SlangCompiler { slangc }
    }

    pub fn from_environment() -> Result<SlangCompiler, CompileError> {
        Ok(SlangCompiler::new(resolve_slangc(None)?))
    }

    /// Compile one Slang source file to a SPIR-V module (all entry points).
    pub fn compile_to_spirv(&self, source: &Path, pass: &str) -> Result<Vec<u32>, CompileError> {
        let out = tempfile_path(pass);
        let output = Command::new(&self.slangc)
            .arg(source)
            .args(["-target", "spirv"])
            .args(["-profile", "spirv_1_5"])
            .arg("-fvk-use-entrypoint-name")
            .arg("-o")
            .arg(&out)
            .output()
            .map_err(|source| CompileError::Io {
                path: self.slangc.clone(),
                source,
            })?;
        if !output.status.success() {
            let _ = std::fs::remove_file(&out);
            return Err(CompileError::Slangc {
                pass: pass.to_owned(),
                stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
            });
        }
        let bytes = std::fs::read(&out).map_err(|source| CompileError::Io {
            path: out.clone(),
            source,
        })?;
        let _ = std::fs::remove_file(&out);
        if bytes.len() % 4 != 0 {
            return Err(CompileError::InvalidSpirv {
                pass: pass.to_owned(),
                message: format!("byte length {} is not a multiple of 4", bytes.len()),
            });
        }
        Ok(bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect())
    }
}

fn tempfile_path(pass: &str) -> PathBuf {
    let sanitized: String = pass
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    env::temp_dir().join(format!(
        "ferridian-packc-{}-{}.spv",
        std::process::id(),
        sanitized
    ))
}
