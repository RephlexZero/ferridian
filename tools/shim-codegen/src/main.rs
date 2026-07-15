use std::fs;
use std::path::PathBuf;

use anyhow::Context;
use clap::Parser;
use ferridian_contract::Contract;

#[derive(Parser)]
#[command(
    name = "shim-codegen",
    about = "Regenerate the Java contract sources",
    version
)]
struct Cli {
    /// Java source root to emit into.
    #[arg(long, default_value = "shim/core/src/main/generated")]
    out: PathBuf,
}

fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    let contract = Contract::current();
    for (relative, content) in shim_codegen::render_java(&contract) {
        let path = cli.out.join(&relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("creating {}", parent.display()))?;
        }
        fs::write(&path, content).with_context(|| format!("writing {}", path.display()))?;
        println!("wrote {}", path.display());
    }
    Ok(())
}
