use anyhow::{Context, Result, bail};
use camino::Utf8PathBuf;
use clap::{Parser, Subcommand};
use std::process::Command;

#[derive(Debug, Parser)]
#[command(author, version, about = "Ferridian workspace automation")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    Ci {
        #[arg(long)]
        fast: bool,
        #[arg(long)]
        skip_java: bool,
    },
    Fmt,
    Hooks,
}

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Ci { fast, skip_java } => run_ci(fast, skip_java),
        Commands::Fmt => run_command("cargo", &["fmt", "--all"], &workspace_root()),
        Commands::Hooks => run_command(
            "git",
            &["config", "core.hooksPath", ".githooks"],
            &workspace_root(),
        ),
    }
}

fn run_ci(fast: bool, skip_java: bool) -> Result<()> {
    let root = workspace_root();

    run_command("cargo", &["fmt", "--all", "--check"], &root)?;
    run_command(
        "cargo",
        &[
            "clippy",
            "--workspace",
            "--all-targets",
            "--all-features",
            "--",
            "-D",
            "warnings",
        ],
        &root,
    )?;

    if !fast {
        run_command("cargo", &["test", "--workspace", "--all-features"], &root)?;
    }

    if !skip_java {
        maybe_run_gradle_check(&root)?;
    }

    Ok(())
}

fn maybe_run_gradle_check(root: &Utf8PathBuf) -> Result<()> {
    let java_dir = root.join("java");
    let build_file = java_dir.join("build.gradle.kts");

    if !build_file.exists() {
        return Ok(());
    }

    if !tool_exists("gradle") {
        eprintln!("Skipping Java checks because gradle is not installed.");
        return Ok(());
    }

    run_command("gradle", &["-p", "java", "check"], root)
}

fn run_command(program: &str, args: &[&str], cwd: &Utf8PathBuf) -> Result<()> {
    let status = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .status()
        .with_context(|| format!("failed to spawn {program}"))?;

    if status.success() {
        return Ok(());
    }

    bail!("command failed: {program} {}", args.join(" "))
}

fn tool_exists(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

fn workspace_root() -> Utf8PathBuf {
    Utf8PathBuf::from_path_buf(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .ancestors()
            .nth(2)
            .expect("workspace root")
            .to_path_buf(),
    )
    .expect("workspace root must be valid utf-8")
}
