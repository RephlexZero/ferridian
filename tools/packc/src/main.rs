//! `packc` — the pack author's front door. Build-time validation is the
//! product promise: errors surface here, not at game load time.

use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use anyhow::Context;
use clap::{Parser, Subcommand};
use ferridian_pack_compiler::{
    SlangCompiler, SourceSnapshot, compile_pack, validate_pack, write_artifact, write_generation,
};

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
    /// Watch a pack, rebuild on change, and publish each rebuild by bumping
    /// `<out>/generation` (the hot-reload handshake a running engine polls).
    Serve {
        /// Pack directory containing pack.toml.
        pack_dir: PathBuf,
        /// Output directory (default: <pack_dir>/build).
        #[arg(long)]
        out: Option<PathBuf>,
        /// Path to slangc (default: $FERRIDIAN_SLANGC, then $PATH).
        #[arg(long)]
        slangc: Option<PathBuf>,
    },
}

/// Serve's change-detection latency. A pack is a handful of files; scanning
/// them five times a second is invisible next to a slangc invocation.
const SERVE_POLL: Duration = Duration::from_millis(200);

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
        Command::Serve {
            pack_dir,
            out,
            slangc,
        } => {
            let compiler = SlangCompiler::new(ferridian_pack_compiler::resolve_slangc(slangc)?);
            let out = out.unwrap_or_else(|| pack_dir.join("build"));
            serve(&pack_dir, &out, &compiler)
        }
    }
}

/// Watch → rebuild → publish, forever. A failed rebuild is the loop's normal
/// weather (the author is mid-edit): report it and keep watching — the
/// running engine simply keeps the last good generation.
fn serve(pack_dir: &Path, out: &Path, compiler: &SlangCompiler) -> anyhow::Result<ExitCode> {
    let mut generation: u64 = 0;
    let rebuild = |generation: &mut u64| {
        match compile_pack(pack_dir, compiler).and_then(|artifact| {
            write_artifact(&artifact, out)?;
            Ok(artifact)
        }) {
            Ok(artifact) => {
                *generation += 1;
                write_generation(out, *generation)?;
                println!(
                    "generation {}: {} ({} pass(es)) -> {}",
                    generation,
                    artifact.manifest.pack.name,
                    artifact.modules.len(),
                    out.display()
                );
            }
            Err(error) => eprintln!("build failed (still watching): {error:#}"),
        }
        anyhow::Ok(())
    };

    let mut snapshot = SourceSnapshot::scan(pack_dir, out)
        .with_context(|| format!("watching pack {}", pack_dir.display()))?;
    rebuild(&mut generation)?;
    println!("watching {} (Ctrl-C to stop)", pack_dir.display());
    loop {
        std::thread::sleep(SERVE_POLL);
        let next = SourceSnapshot::scan(pack_dir, out)?;
        let changes = snapshot.changes_since(&next);
        if changes.is_empty() {
            continue;
        }
        // Debounce: rescan until the tree is quiet, so a save spanning
        // multiple files (or a slow write) compiles once, not per event.
        snapshot = next;
        loop {
            std::thread::sleep(SERVE_POLL);
            let settled = SourceSnapshot::scan(pack_dir, out)?;
            if snapshot.changes_since(&settled).is_empty() {
                break;
            }
            snapshot = settled;
        }
        for path in &changes {
            println!(
                "changed: {}",
                path.strip_prefix(pack_dir).unwrap_or(path).display()
            );
        }
        rebuild(&mut generation)?;
    }
}
