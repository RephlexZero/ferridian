//! The shim → layer transport for per-frame camera state (M3 follow-up).
//!
//! The Fabric shim's `System.load` of this cdylib loads the *same* file the
//! Vulkan loader also loaded as a layer — on Linux/macOS `dlopen` dedups by
//! canonical path, so both "loads" resolve to one mapped image sharing this
//! module's statics. The generated `NativeBridge.publishCamera` (see
//! `tools/shim-codegen`) calls straight into [`publish`] through the JNI
//! exports in `lib.rs`; [`current_camera`] is what
//! [`ferridian_vk_rt::PackCompositor::update_camera`] reads once per
//! composited frame, immediately before recording.
//!
//! No shim has published anything until a game does: [`current_camera`]
//! falls back to [`CameraUniforms::placeholder`], the exact values packs
//! read before this transport existed, so a `packc`-only or non-Fabric host
//! keeps working unchanged.

use std::sync::Mutex;

use ferridian_contract::CameraUniforms;

static CURRENT_CAMERA: Mutex<Option<CameraUniforms>> = Mutex::new(None);

/// The most recently published camera state, or the placeholder if nothing
/// has published one yet.
pub(crate) fn current_camera() -> CameraUniforms {
    CURRENT_CAMERA
        .lock()
        .expect("camera transport lock poisoned")
        .unwrap_or_else(CameraUniforms::placeholder)
}

/// Store a freshly published camera state.
pub(crate) fn publish(camera: CameraUniforms) {
    *CURRENT_CAMERA
        .lock()
        .expect("camera transport lock poisoned") = Some(camera);
}

#[cfg(test)]
mod tests {
    use super::*;

    // One test, one lifecycle: the static is process-global, and `cargo
    // test` runs unit tests of one crate on threads sharing a process, so a
    // second test here could race this one.
    #[test]
    fn falls_back_to_the_placeholder_then_reflects_whatever_was_published() {
        assert_eq!(current_camera(), CameraUniforms::placeholder());
        let camera = CameraUniforms {
            sun_direction: [0.0, 1.0, 0.0],
            time_seconds: 42.0,
            near_plane: 0.1,
            far_plane: 256.0,
        };
        publish(camera);
        assert_eq!(current_camera(), camera);
    }
}
