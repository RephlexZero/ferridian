use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};
use upstream_watch::{
    EXIT_CHANGES_FOUND, KnownVersions, MOJANG_MANIFEST_URL, SignatureInventory, VersionManifest,
    diff_inventories, find_new_versions, render_diff_report,
};

#[derive(Parser)]
#[command(
    name = "upstream-watch",
    about = "Minecraft upstream breakage detection",
    version
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Check Mojang's version manifest for versions we haven't processed.
    /// Exits 3 when new versions exist.
    Poll {
        #[arg(long, default_value = MOJANG_MANIFEST_URL)]
        manifest_url: String,
        #[arg(long, default_value = "tools/upstream-watch/data/known_versions.toml")]
        known: PathBuf,
        /// Append newly seen versions to the known file.
        #[arg(long)]
        write: bool,
    },
    /// Diff two derived signature inventories. Exits 3 on differences.
    ///
    /// Inventory *extraction* (download jar -> Vineflower -> signatures) runs
    /// in CI only and is not implemented yet (M1); only derived inventories
    /// ever land in the repo.
    Diff { old: PathBuf, new: PathBuf },
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
        Command::Poll {
            manifest_url,
            known,
            write,
        } => {
            let manifest: VersionManifest = ureq::get(&manifest_url)
                .call()
                .with_context(|| format!("fetching {manifest_url}"))?
                .into_json()
                .context("parsing version manifest")?;
            let known_versions: KnownVersions = match fs::read_to_string(&known) {
                Ok(text) => {
                    toml::from_str(&text).with_context(|| format!("parsing {}", known.display()))?
                }
                Err(e) if e.kind() == std::io::ErrorKind::NotFound => KnownVersions::default(),
                Err(e) => return Err(e).with_context(|| format!("reading {}", known.display())),
            };
            let new = find_new_versions(&manifest, &known_versions);
            if new.is_empty() {
                println!(
                    "up to date (latest release {}, snapshot {})",
                    manifest.latest.release, manifest.latest.snapshot
                );
                return Ok(ExitCode::SUCCESS);
            }
            for entry in &new {
                println!("new {}: {}", entry.kind, entry.id);
            }
            if write {
                let mut updated = known_versions;
                updated
                    .versions
                    .extend(new.iter().map(|entry| entry.id.clone()));
                updated.versions.sort();
                if let Some(parent) = known.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(&known, toml::to_string_pretty(&updated)?)
                    .with_context(|| format!("writing {}", known.display()))?;
                println!("recorded {} version(s) in {}", new.len(), known.display());
            }
            Ok(ExitCode::from(EXIT_CHANGES_FOUND))
        }
        Command::Diff { old, new } => {
            let load = |path: &PathBuf| -> anyhow::Result<SignatureInventory> {
                let text = fs::read_to_string(path)
                    .with_context(|| format!("reading {}", path.display()))?;
                toml::from_str(&text).with_context(|| format!("parsing {}", path.display()))
            };
            let diff = diff_inventories(&load(&old)?, &load(&new)?);
            println!("{}", render_diff_report(&diff));
            if diff.is_empty() {
                Ok(ExitCode::SUCCESS)
            } else {
                Ok(ExitCode::from(EXIT_CHANGES_FOUND))
            }
        }
    }
}
