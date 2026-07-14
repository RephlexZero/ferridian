//! Jar → derived signature inventory, for the tracked render classes.
//!
//! This is the CI-only half of upstream watch: the jar itself and everything
//! inside it stays on the runner; only the [`SignatureInventory`] (names and
//! signatures) comes out.

use std::collections::BTreeMap;
use std::io::{Read, Seek};

use serde::{Deserialize, Serialize};

use crate::SignatureInventory;
use crate::classfile::{class_signatures, parse_class};

/// Classes whose signatures we track across Minecraft versions, committed as
/// `tools/upstream-watch/data/tracked_classes.toml`. Fully-qualified dotted
/// names, e.g. `net.minecraft.client.renderer.LevelRenderer`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct TrackedClasses {
    pub classes: Vec<String>,
}

/// A single classfile is not going to be bigger than this; anything larger is
/// a hostile or corrupt jar and gets skipped.
const MAX_CLASSFILE_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Debug, thiserror::Error)]
pub enum ExtractError {
    #[error("failed to read jar: {0}")]
    Jar(#[from] zip::result::ZipError),
    #[error("failed to read {name} from jar: {source}")]
    Entry {
        name: String,
        source: std::io::Error,
    },
}

/// The outcome of an extraction: the inventory plus what wasn't found.
/// Missing classes are a *signal* (the diff will flag them as removed), not
/// an error — renames are exactly what upstream watch exists to catch.
#[derive(Debug)]
pub struct Extraction {
    pub inventory: SignatureInventory,
    pub missing: Vec<String>,
    /// Classes that were present but unparsable (with the reason) — should be
    /// empty for real Mojang jars, so these get surfaced loudly.
    pub unparsable: Vec<(String, String)>,
}

/// Extract the signature inventory of `tracked` classes from a jar.
pub fn extract_inventory<R: Read + Seek>(
    jar: R,
    tracked: &TrackedClasses,
) -> Result<Extraction, ExtractError> {
    let mut archive = zip::ZipArchive::new(jar)?;
    let mut classes = BTreeMap::new();
    let mut missing = Vec::new();
    let mut unparsable = Vec::new();
    for class_name in &tracked.classes {
        let entry_name = format!("{}.class", class_name.replace('.', "/"));
        let mut entry = match archive.by_name(&entry_name) {
            Ok(entry) => entry,
            Err(zip::result::ZipError::FileNotFound) => {
                missing.push(class_name.clone());
                continue;
            }
            Err(error) => return Err(error.into()),
        };
        if entry.size() > MAX_CLASSFILE_BYTES {
            unparsable.push((class_name.clone(), "classfile implausibly large".to_owned()));
            continue;
        }
        let mut bytes = Vec::with_capacity(entry.size() as usize);
        entry
            .read_to_end(&mut bytes)
            .map_err(|source| ExtractError::Entry {
                name: entry_name.clone(),
                source,
            })?;
        match parse_class(&bytes) {
            Ok(parsed) => {
                classes.insert(class_name.clone(), class_signatures(&parsed));
            }
            Err(error) => unparsable.push((class_name.clone(), error.to_string())),
        }
    }
    Ok(Extraction {
        inventory: SignatureInventory { classes },
        missing,
        unparsable,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Cursor, Write};

    /// Build an in-memory jar holding the committed fixture classfile (our
    /// own Java source, compiled with the pinned JDK — see
    /// `tests/fixtures/README.md`).
    fn fixture_jar() -> Cursor<Vec<u8>> {
        let class_bytes = include_bytes!("../tests/fixtures/RenderStandin.class");
        let mut jar = zip::ZipWriter::new(Cursor::new(Vec::new()));
        jar.start_file::<_, ()>(
            "io/ferridian/fixture/RenderStandin.class",
            zip::write::FileOptions::default(),
        )
        .unwrap();
        jar.write_all(class_bytes).unwrap();
        jar.finish().unwrap()
    }

    #[test]
    fn extracts_tracked_class_signatures() {
        let tracked = TrackedClasses {
            classes: vec![
                "io.ferridian.fixture.RenderStandin".to_owned(),
                "io.ferridian.fixture.DoesNotExist".to_owned(),
            ],
        };
        let extraction = extract_inventory(fixture_jar(), &tracked).unwrap();
        assert_eq!(
            extraction.missing,
            vec!["io.ferridian.fixture.DoesNotExist"]
        );
        assert!(extraction.unparsable.is_empty());
        let signatures = &extraction.inventory.classes["io.ferridian.fixture.RenderStandin"];
        assert!(
            signatures
                .iter()
                .any(|s| s == "public void renderLevel(java.lang.String, float[], int)"),
            "expected renderLevel signature, got: {signatures:#?}"
        );
        assert!(
            signatures
                .iter()
                .any(|s| s == "private static final long FRAME_BUDGET_NS"),
            "expected field signature, got: {signatures:#?}"
        );
    }
}
