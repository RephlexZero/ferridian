//! Single source of truth for the Java↔Rust contract.
//!
//! The Fabric shim publishes pass metadata (which game pipeline corresponds to
//! terrain, entities, …) to the engine over this versioned contract. The Java
//! side of the contract is *generated* from these definitions by
//! `tools/shim-codegen`; nothing else may define the wire format.
//!
//! The contract is regenerated per Minecraft version — mechanically, from the
//! unobfuscated sources — so version drift is detected, never guessed at.

use serde::{Deserialize, Serialize};

/// Bumped whenever the wire format or the meaning of any field changes.
/// Both sides refuse to talk across a version mismatch.
pub const CONTRACT_VERSION: u32 = 1;

/// Semantic classification of a game render pass, as published by the shim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum GamePassKind {
    Sky,
    Terrain,
    Entities,
    BlockEntities,
    Particles,
    Weather,
    Translucent,
    Hand,
    Gui,
    /// A pass the shim could not classify. The engine must render it
    /// unmodified — unknown never means "drop".
    Unknown,
}

impl GamePassKind {
    /// Every variant, in wire order. Codegen and exhaustiveness tests key off this.
    pub const ALL: [GamePassKind; 10] = [
        GamePassKind::Sky,
        GamePassKind::Terrain,
        GamePassKind::Entities,
        GamePassKind::BlockEntities,
        GamePassKind::Particles,
        GamePassKind::Weather,
        GamePassKind::Translucent,
        GamePassKind::Hand,
        GamePassKind::Gui,
        GamePassKind::Unknown,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            GamePassKind::Sky => "sky",
            GamePassKind::Terrain => "terrain",
            GamePassKind::Entities => "entities",
            GamePassKind::BlockEntities => "block_entities",
            GamePassKind::Particles => "particles",
            GamePassKind::Weather => "weather",
            GamePassKind::Translucent => "translucent",
            GamePassKind::Hand => "hand",
            GamePassKind::Gui => "gui",
            GamePassKind::Unknown => "unknown",
        }
    }
}

/// One pass the shim knows how to identify in the current Minecraft version.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PassDescriptor {
    pub kind: GamePassKind,
    /// The game-side anchor the shim uses to identify this pass (a render
    /// pipeline/stage name in the unobfuscated sources). Regenerated per
    /// Minecraft version by upstream tooling.
    pub game_anchor: String,
}

/// The full contract document exchanged between shim and engine.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Contract {
    pub version: u32,
    /// Minecraft version(s) the `game_anchor`s were derived from.
    pub minecraft_version: String,
    pub passes: Vec<PassDescriptor>,
}

#[derive(Debug, thiserror::Error)]
pub enum ContractError {
    #[error("failed to parse contract: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("failed to serialize contract: {0}")]
    Serialize(#[from] toml::ser::Error),
    #[error("contract version mismatch: ours {ours}, theirs {theirs}")]
    VersionMismatch { ours: u32, theirs: u32 },
}

impl Contract {
    /// The canonical contract for the Minecraft version we currently track.
    ///
    /// `game_anchor` values are placeholders until pass detection lands (M2);
    /// the *shape* of this data is what codegen and the shim build against.
    pub fn current() -> Contract {
        Contract {
            version: CONTRACT_VERSION,
            minecraft_version: "26.2".to_owned(),
            passes: GamePassKind::ALL
                .iter()
                .map(|&kind| PassDescriptor {
                    kind,
                    game_anchor: format!("todo/{}", kind.as_str()),
                })
                .collect(),
        }
    }

    pub fn to_toml(&self) -> Result<String, ContractError> {
        Ok(toml::to_string_pretty(self)?)
    }

    pub fn from_toml(text: &str) -> Result<Contract, ContractError> {
        let contract: Contract = toml::from_str(text)?;
        if contract.version != CONTRACT_VERSION {
            return Err(ContractError::VersionMismatch {
                ours: CONTRACT_VERSION,
                theirs: contract.version,
            });
        }
        Ok(contract)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_toml() {
        let contract = Contract::current();
        let text = contract.to_toml().unwrap();
        assert_eq!(Contract::from_toml(&text).unwrap(), contract);
    }

    #[test]
    fn rejects_version_mismatch() {
        let mut contract = Contract::current();
        contract.version = CONTRACT_VERSION + 1;
        let text = toml::to_string_pretty(&contract).unwrap();
        assert!(matches!(
            Contract::from_toml(&text),
            Err(ContractError::VersionMismatch { .. })
        ));
    }

    #[test]
    fn all_variants_have_unique_wire_names() {
        let mut names: Vec<_> = GamePassKind::ALL.iter().map(|k| k.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), GamePassKind::ALL.len());
    }
}
