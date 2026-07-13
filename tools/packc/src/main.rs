//! `packc` — the pack author's front door. Build-time validation is the
//! product promise: errors surface here, not at game load time.

use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};
use ferridian_pack_compiler::{SlangCompiler, compile_pack, validate_pack, write_artifact};

#[derive(Parser)]
#[command(name = "packc", about = "Ferridian shader pack compiler", version)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Compile a pack to a runnable artifact (SPIR-V + reflection).
    Build {
        /// Pack directory containing pack.toml.
        pack_dir: PathBuf,
        /// Output directory (default: <pack_dir>/build).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Path to slangc (default: $FERRIDIAN_SLANGC, then $PATH).
        #[arg(long)]
        slangc: Option<PathBuf>,
    },
    /// Validate a pack's manifest and file layout without compiling.
    Validate { pack_dir: PathBuf },
    /// Watch a pack and hot-reload it into a running engine.
    Serve { pack_dir: PathBuf },
}

fn main() -> ExitCode {
    match run() {
        Ok(code) => code,
        Err(error) => {
            eprintln!("error: {error:#}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> anyhow::Result<ExitCode> {
    match Cli::parse().command {
        Command::Build {
            pack_dir,
            out,
            slangc,
        } => {
            let compiler = SlangCompiler::new(ferridian_pack_compiler::resolve_slangc(slangc)?);
            let artifact = compile_pack(&pack_dir, &compiler)
                .with_context(|| format!("building pack {}", pack_dir.display()))?;
            let out = out.unwrap_or_else(|| pack_dir.join("build"));
            write_artifact(&artifact, &out)?;
            println!(
                "built {} ({} pass(es)) -> {}",
                artifact.manifest.pack.name,
                artifact.modules.len(),
                out.display()
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Validate { pack_dir } => {
            let manifest = validate_pack(&pack_dir)
                .with_context(|| format!("validating pack {}", pack_dir.display()))?;
            println!(
                "ok: {} v{} ({} pass(es))",
                manifest.pack.name,
                manifest.pack.version,
                manifest.passes.len()
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Serve { pack_dir } => {
            eprintln!(
                "packc serve is not implemented yet (arrives with hot reload in M3); \
                 got: {}",
                pack_dir.display()
            );
            Ok(ExitCode::from(2))
        }
    }
}
