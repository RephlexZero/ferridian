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
    Release {
        #[arg(long, default_value = "debug")]
        profile: String,
        #[arg(long)]
        skip_java: bool,
    },
    Pgo {
        #[arg(long, default_value = "ferridian-standalone")]
        binary: String,
    },
    /// Package platform-specific JNI native libraries for Java consumption.
    PackageJni {
        #[arg(long, default_value = "release")]
        profile: String,
    },
    /// Cross-compile native artifacts for target platforms.
    CrossBuild {
        /// Target triple (e.g. x86_64-unknown-linux-gnu, aarch64-apple-darwin)
        #[arg(long)]
        target: String,
        #[arg(long, default_value = "release")]
        profile: String,
    },
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
        Commands::Release { profile, skip_java } => run_release(&profile, skip_java),
        Commands::Pgo { binary } => run_pgo(&binary),
        Commands::PackageJni { profile } => run_package_jni(&profile),
        Commands::CrossBuild { target, profile } => run_cross_build(&target, &profile),
    }
}

fn run_release(profile: &str, skip_java: bool) -> Result<()> {
    let root = workspace_root();

    // Build native artifacts
    let cargo_profile = if profile == "release" {
        vec!["build", "--workspace", "--release"]
    } else {
        vec!["build", "--workspace"]
    };
    run_command("cargo", &cargo_profile, &root)?;

    // Build JNI cdylib specifically
    let mut jni_args = vec!["build", "-p", "ferridian-jni-bridge"];
    if profile == "release" {
        jni_args.push("--release");
    }
    run_command("cargo", &jni_args, &root)?;

    if !skip_java && tool_exists("gradle") {
        let java_dir = root.join("java");
        if java_dir.join("build.gradle.kts").exists() {
            run_command("gradle", &["-p", "java", "build"], &root)?;
        }
    }

    let profile_dir = if profile == "release" {
        "release"
    } else {
        "debug"
    };
    let target_dir = root.join("target").join(profile_dir);
    eprintln!("Release artifacts built in {target_dir}");

    Ok(())
}

fn run_pgo(binary: &str) -> Result<()> {
    let root = workspace_root();
    let pgo_dir = root.join("target").join("pgo-profiles");

    eprintln!("Step 1: Building with PGO instrumentation...");
    run_command("cargo", &["build", "--release", "--workspace"], &root)?;
    // Note: actual PGO requires RUSTFLAGS=-Cprofile-generate=<dir>
    // and a training run, then RUSTFLAGS=-Cprofile-use=<merged.profdata>
    // This command documents the workflow steps.
    eprintln!("Step 2: PGO profile directory: {pgo_dir}");
    eprintln!("Step 3: Run the training workload with RUSTFLAGS=-Cprofile-generate={pgo_dir}");
    eprintln!("  cargo build --release -p {binary}");
    eprintln!("  target/release/{binary} (run your training scenario)");
    eprintln!("Step 4: Merge profiles with llvm-profdata merge -o merged.profdata {pgo_dir}");
    eprintln!("Step 5: Build optimized with RUSTFLAGS=-Cprofile-use=merged.profdata");
    Ok(())
}

fn run_package_jni(profile: &str) -> Result<()> {
    let root = workspace_root();
    let profile_dir = if profile == "release" {
        "release"
    } else {
        "debug"
    };

    // Build the JNI cdylib
    let mut args = vec!["build", "-p", "ferridian-jni-bridge"];
    if profile == "release" {
        args.push("--release");
    }
    run_command("cargo", &args, &root)?;

    // Determine the platform-specific library name
    let lib_name = if cfg!(target_os = "linux") {
        "libferridian_jni.so"
    } else if cfg!(target_os = "macos") {
        "libferridian_jni.dylib"
    } else if cfg!(target_os = "windows") {
        "ferridian_jni.dll"
    } else {
        bail!("unsupported OS for JNI packaging");
    };

    let src = root.join("target").join(profile_dir).join(lib_name);
    let dest_dir = root
        .join("java")
        .join("build")
        .join("resources")
        .join("main")
        .join("natives");

    std::fs::create_dir_all(&dest_dir).with_context(|| format!("failed to create {dest_dir}"))?;

    let platform_tag = format!("{}-{}", std::env::consts::OS, std::env::consts::ARCH);
    let dest = dest_dir.join(format!("{platform_tag}-{lib_name}"));
    std::fs::copy(&src, &dest).with_context(|| format!("failed to copy {src} → {dest}"))?;

    eprintln!("Packaged JNI library: {dest}");
    Ok(())
}

fn run_cross_build(target: &str, profile: &str) -> Result<()> {
    let root = workspace_root();

    // Check if the target is installed
    let installed = Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .with_context(|| "failed to list installed targets")?;
    let installed_str = String::from_utf8_lossy(&installed.stdout);
    if !installed_str.lines().any(|l| l.trim() == target) {
        eprintln!("Installing target: {target}");
        run_command("rustup", &["target", "add", target], &root)?;
    }

    let mut args = vec!["build", "--workspace", "--target", target];
    if profile == "release" {
        args.push("--release");
    }
    run_command("cargo", &args, &root)?;

    let profile_dir = if profile == "release" {
        "release"
    } else {
        "debug"
    };
    let target_dir = root.join("target").join(target).join(profile_dir);
    eprintln!("Cross-build artifacts for {target} in {target_dir}");
    Ok(())
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn workspace_root_exists() {
        let root = workspace_root();
        assert!(root.exists(), "workspace root should exist: {root}");
    }

    #[test]
    fn workspace_root_contains_cargo_toml() {
        let root = workspace_root();
        assert!(root.join("Cargo.toml").exists());
    }

    #[test]
    fn tool_exists_finds_cargo() {
        assert!(tool_exists("cargo"));
    }

    #[test]
    fn tool_exists_returns_false_for_missing_tool() {
        assert!(!tool_exists("totally-nonexistent-tool-xyz"));
    }

    #[test]
    fn workspace_root_has_crates_dir() {
        let root = workspace_root();
        assert!(root.join("crates").is_dir());
    }

    #[test]
    fn workspace_root_has_java_dir() {
        let root = workspace_root();
        assert!(root.join("java").is_dir());
    }
}
