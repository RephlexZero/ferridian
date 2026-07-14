use std::fs;
use std::path::PathBuf;
use std::process::ExitCode;

use anyhow::Context;
use clap::{Parser, Subcommand};
use upstream_watch::{
    EXIT_CHANGES_FOUND, KnownVersions, MOJANG_MANIFEST_URL, SignatureInventory, VersionDetail,
    VersionManifest, diff_inventories, find_new_versions, render_diff_report,
};

use upstream_watch::extract::{TrackedClasses, extract_inventory};

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
    Diff { old: PathBuf, new: PathBuf },
    /// Download a version's (unobfuscated) client jar, verifying its sha1.
    ///
    /// CI-only plumbing: the jar never enters the repo — it exists to have an
    /// inventory extracted from it and is discarded with the runner.
    FetchJar {
        /// Version id from the manifest (e.g. "26.2").
        version: String,
        #[arg(long, default_value = MOJANG_MANIFEST_URL)]
        manifest_url: String,
        #[arg(long)]
        out: PathBuf,
    },
    /// Extract the derived signature inventory of tracked classes from a jar.
    ///
    /// Signatures come straight from the classfiles — nothing is decompiled,
    /// and only derived names/signatures are written out (legal guardrail).
    Extract {
        jar: PathBuf,
        #[arg(long, default_value = "tools/upstream-watch/data/tracked_classes.toml")]
        tracked: PathBuf,
        /// Where to write the inventory TOML.
        #[arg(long)]
        out: PathBuf,
    },
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
        Command::FetchJar {
            version,
            manifest_url,
            out,
        } => {
            let manifest: VersionManifest = ureq::get(&manifest_url)
                .call()
                .with_context(|| format!("fetching {manifest_url}"))?
                .into_json()
                .context("parsing version manifest")?;
            let entry = manifest
                .versions
                .iter()
                .find(|entry| entry.id == version)
                .with_context(|| format!("version {version} not in the manifest"))?;
            let detail: VersionDetail = ureq::get(&entry.url)
                .call()
                .with_context(|| format!("fetching {}", entry.url))?
                .into_json()
                .with_context(|| format!("parsing version detail for {version}"))?;
            let client = &detail.downloads.client;
            println!(
                "downloading client jar for {version} ({} bytes) ...",
                client.size
            );
            let mut reader = ureq::get(&client.url)
                .call()
                .with_context(|| format!("downloading {}", client.url))?
                .into_reader();
            let mut bytes = Vec::with_capacity(client.size as usize);
            std::io::copy(&mut reader, &mut bytes).context("reading jar body")?;
            let mut sha1 = sha1_smol::Sha1::new();
            sha1.update(&bytes);
            let digest = sha1.digest().to_string();
            anyhow::ensure!(
                digest == client.sha1,
                "sha1 mismatch for {version}: expected {}, got {digest}",
                client.sha1
            );
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&out, &bytes).with_context(|| format!("writing {}", out.display()))?;
            println!(
                "wrote {} ({} bytes, sha1 verified)",
                out.display(),
                bytes.len()
            );
            Ok(ExitCode::SUCCESS)
        }
        Command::Extract { jar, tracked, out } => {
            let tracked: TrackedClasses = toml::from_str(
                &fs::read_to_string(&tracked)
                    .with_context(|| format!("reading {}", tracked.display()))?,
            )
            .with_context(|| format!("parsing {}", tracked.display()))?;
            let file =
                fs::File::open(&jar).with_context(|| format!("opening {}", jar.display()))?;
            let extraction = extract_inventory(std::io::BufReader::new(file), &tracked)
                .with_context(|| format!("extracting from {}", jar.display()))?;
            for class in &extraction.missing {
                eprintln!("warning: tracked class not in jar: {class}");
            }
            for (class, reason) in &extraction.unparsable {
                eprintln!("warning: could not parse {class}: {reason}");
            }
            if let Some(parent) = out.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(&out, toml::to_string_pretty(&extraction.inventory)?)
                .with_context(|| format!("writing {}", out.display()))?;
            println!(
                "extracted {} class(es) ({} missing, {} unparsable) -> {}",
                extraction.inventory.classes.len(),
                extraction.missing.len(),
                extraction.unparsable.len(),
                out.display()
            );
            Ok(ExitCode::SUCCESS)
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
