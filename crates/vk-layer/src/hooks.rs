//! The seam between the boring interception shell and the engine.
//!
//! Everything the layer learns about the game funnels through these
//! functions; `ferridian-engine` never sees loader plumbing, and this module
//! never grows logic. TODO(M2): route into `ferridian_engine` frame
//! orchestration once pass detection exists.

pub(crate) fn instance_created() {
    tracing::debug!("ferridian layer: instance created (pass-through)");
}

pub(crate) fn device_created() {
    tracing::debug!("ferridian layer: device created (pass-through)");
}
