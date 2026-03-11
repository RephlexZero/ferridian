use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceMetadata {
    pub name: &'static str,
    pub focus: &'static str,
}

impl WorkspaceMetadata {
    pub const fn new(name: &'static str, focus: &'static str) -> Self {
        Self { name, focus }
    }
}

#[derive(Debug, Error)]
pub enum FerridianError {
    #[error("unsupported renderer backend: {0}")]
    UnsupportedBackend(&'static str),
}
