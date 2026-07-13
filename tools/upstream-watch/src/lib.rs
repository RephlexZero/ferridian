//! Upstream breakage detection (§5 of the plan).
//!
//! `poll` watches Mojang's version manifest for unseen versions. `diff`
//! compares two *derived signature inventories* — names/signatures of tracked
//! render classes, extracted in CI from unobfuscated jars. Only inventories
//! are ever committed; decompiled Mojang source never enters the repo
//! (legal guardrail, enforced by review + CI discipline).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

pub const MOJANG_MANIFEST_URL: &str =
    "https://piston-meta.mojang.com/mc/game/version_manifest_v2.json";

/// Exit code the CI workflow interprets as "upstream changed, act on it"
/// (distinct from 1 = tool failure).
pub const EXIT_CHANGES_FOUND: u8 = 3;

/// The subset of Mojang's `version_manifest_v2.json` we care about.
#[derive(Debug, Deserialize)]
pub struct VersionManifest {
    pub latest: LatestVersions,
    pub versions: Vec<VersionEntry>,
}

#[derive(Debug, Deserialize)]
pub struct LatestVersions {
    pub release: String,
    pub snapshot: String,
}

#[derive(Debug, Deserialize)]
pub struct VersionEntry {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
}

/// Versions we've already processed, committed to the repo.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct KnownVersions {
    #[serde(default)]
    pub versions: Vec<String>,
}

/// Manifest ids not yet in the known list, in manifest (newest-first) order.
pub fn find_new_versions<'a>(
    manifest: &'a VersionManifest,
    known: &KnownVersions,
) -> Vec<&'a VersionEntry> {
    manifest
        .versions
        .iter()
        .filter(|entry| !known.versions.iter().any(|id| id == &entry.id))
        .collect()
}

/// A derived signature inventory: tracked class name → sorted member
/// signatures. This is the only artifact of upstream jars that may be
/// committed.
#[derive(Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignatureInventory {
    #[serde(default)]
    pub classes: BTreeMap<String, Vec<String>>,
}

#[derive(Debug, Default, PartialEq, Eq)]
pub struct InventoryDiff {
    pub added_classes: Vec<String>,
    pub removed_classes: Vec<String>,
    /// class → (removed signatures, added signatures)
    pub changed_classes: BTreeMap<String, (Vec<String>, Vec<String>)>,
}

impl InventoryDiff {
    pub fn is_empty(&self) -> bool {
        self.added_classes.is_empty()
            && self.removed_classes.is_empty()
            && self.changed_classes.is_empty()
    }
}

pub fn diff_inventories(old: &SignatureInventory, new: &SignatureInventory) -> InventoryDiff {
    let mut diff = InventoryDiff::default();
    for (class, old_sigs) in &old.classes {
        match new.classes.get(class) {
            None => diff.removed_classes.push(class.clone()),
            Some(new_sigs) if new_sigs != old_sigs => {
                let removed: Vec<String> = old_sigs
                    .iter()
                    .filter(|sig| !new_sigs.contains(sig))
                    .cloned()
                    .collect();
                let added: Vec<String> = new_sigs
                    .iter()
                    .filter(|sig| !old_sigs.contains(sig))
                    .cloned()
                    .collect();
                diff.changed_classes.insert(class.clone(), (removed, added));
            }
            Some(_) => {}
        }
    }
    for class in new.classes.keys() {
        if !old.classes.contains_key(class) {
            diff.added_classes.push(class.clone());
        }
    }
    diff
}

pub fn render_diff_report(diff: &InventoryDiff) -> String {
    if diff.is_empty() {
        return "no signature changes in tracked render classes".to_owned();
    }
    let mut out = String::from("signature inventory changed:\n");
    for class in &diff.added_classes {
        out.push_str(&format!("  + class {class}\n"));
    }
    for class in &diff.removed_classes {
        out.push_str(&format!("  - class {class}\n"));
    }
    for (class, (removed, added)) in &diff.changed_classes {
        out.push_str(&format!("  ~ class {class}\n"));
        for sig in removed {
            out.push_str(&format!("      - {sig}\n"));
        }
        for sig in added {
            out.push_str(&format!("      + {sig}\n"));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn inventory(entries: &[(&str, &[&str])]) -> SignatureInventory {
        SignatureInventory {
            classes: entries
                .iter()
                .map(|(class, sigs)| {
                    (
                        (*class).to_owned(),
                        sigs.iter().map(|s| (*s).to_owned()).collect(),
                    )
                })
                .collect(),
        }
    }

    #[test]
    fn finds_unseen_versions() {
        let manifest = VersionManifest {
            latest: LatestVersions {
                release: "26.2".to_owned(),
                snapshot: "26.3-pre1".to_owned(),
            },
            versions: vec![
                VersionEntry {
                    id: "26.3-pre1".to_owned(),
                    kind: "snapshot".to_owned(),
                },
                VersionEntry {
                    id: "26.2".to_owned(),
                    kind: "release".to_owned(),
                },
            ],
        };
        let known = KnownVersions {
            versions: vec!["26.2".to_owned()],
        };
        let new = find_new_versions(&manifest, &known);
        assert_eq!(new.len(), 1);
        assert_eq!(new[0].id, "26.3-pre1");
    }

    #[test]
    fn identical_inventories_diff_empty() {
        let a = inventory(&[("LevelRenderer", &["void renderLevel(PoseStack)"])]);
        let b = inventory(&[("LevelRenderer", &["void renderLevel(PoseStack)"])]);
        assert!(diff_inventories(&a, &b).is_empty());
    }

    #[test]
    fn detects_signature_change() {
        let old = inventory(&[("LevelRenderer", &["void renderLevel(PoseStack)"])]);
        let new = inventory(&[("LevelRenderer", &["void renderLevel(GpuBufferSlice)"])]);
        let diff = diff_inventories(&old, &new);
        let (removed, added) = &diff.changed_classes["LevelRenderer"];
        assert_eq!(removed, &["void renderLevel(PoseStack)"]);
        assert_eq!(added, &["void renderLevel(GpuBufferSlice)"]);
    }

    #[test]
    fn detects_added_and_removed_classes() {
        let old = inventory(&[("Gone", &[])]);
        let new = inventory(&[("Fresh", &[])]);
        let diff = diff_inventories(&old, &new);
        assert_eq!(diff.added_classes, vec!["Fresh".to_owned()]);
        assert_eq!(diff.removed_classes, vec!["Gone".to_owned()]);
    }
}
