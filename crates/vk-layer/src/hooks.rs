//! The seam between the boring interception shell and the engine.
//!
//! Everything the layer learns about the game funnels through these
//! functions; `ferridian-engine` never sees loader plumbing, and this module
//! never grows logic. TODO(M2): route into `ferridian_engine` frame
//! orchestration once pass detection exists.

use std::sync::atomic::{AtomicU64, Ordering};

/// Render passes begun through the layer since load. The first observable
/// fact the layer extracts from a host application; pass *classification*
/// (which pass is terrain/sky/…) builds on the same interception point.
static RENDER_PASSES_BEGUN: AtomicU64 = AtomicU64::new(0);

pub(crate) fn instance_created() {
    tracing::debug!("ferridian layer: instance created (pass-through)");
}

pub(crate) fn device_created() {
    tracing::debug!("ferridian layer: device created (pass-through)");
}

pub(crate) fn render_pass_begun() {
    RENDER_PASSES_BEGUN.fetch_add(1, Ordering::Relaxed);
}

pub(crate) fn render_pass_count() -> u64 {
    RENDER_PASSES_BEGUN.load(Ordering::Relaxed)
}
