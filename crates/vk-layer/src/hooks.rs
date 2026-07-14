//! The seam between the boring interception shell and the engine.
//!
//! Everything the layer learns about the game funnels through these
//! functions; `ferridian-engine` never sees loader plumbing, and this module
//! never grows logic — classification lives in [`ferridian_engine::frame`].

use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
use ferridian_contract::{Contract, GamePassKind};
use ferridian_engine::frame::PassObserver;
use ferridian_engine::pack::load_pack;
use ferridian_vk_rt::PackCompositor;

/// Artifact directory to composite over intercepted frames. Read once per
/// device creation; hot reload through the layer is a follow-up (the engine's
/// `PackWatcher` is ready for it, the layer needs a safe swap point first).
pub const PACK_ENV: &str = "FERRIDIAN_PACK";

/// Render passes begun through the layer since load. The first observable
/// fact the layer extracts from a host application.
static RENDER_PASSES_BEGUN: AtomicU64 = AtomicU64::new(0);

/// One observer for the whole process: label state is per-command-buffer on
/// the Vulkan side, so this assumes single-threaded recording (true of the
/// test harness and vanilla's render thread). TODO(M2): per-command-buffer
/// observers when multi-threaded recording matters.
static OBSERVER: Mutex<Option<PassObserver>> = Mutex::new(None);

fn with_observer<T>(observe: impl FnOnce(&mut PassObserver) -> T) -> T {
    let mut guard = OBSERVER.lock().expect("pass observer lock poisoned");
    let observer = guard.get_or_insert_with(|| PassObserver::new(&Contract::current()));
    observe(observer)
}

pub(crate) fn instance_created() {
    tracing::debug!("ferridian layer: instance created");
}

pub(crate) fn instance_destroyed() {
    tracing::debug!("ferridian layer: instance destroyed");
}

pub(crate) fn device_created() {
    tracing::debug!("ferridian layer: device created");
}

/// Load and wire the pack named by [`PACK_ENV`], if any. Every failure is a
/// warning and `None` — a broken pack must never take the game down with it.
pub(crate) fn create_compositor(
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    memory_properties: vk::PhysicalDeviceMemoryProperties,
) -> Option<PackCompositor> {
    let dir = std::env::var_os(PACK_ENV)?;
    let pack = match load_pack(Path::new(&dir), 0) {
        Ok(pack) => pack,
        Err(error) => {
            tracing::warn!(%error, pack = %dir.to_string_lossy(), "pack failed to load; compositing disabled");
            return None;
        }
    };
    match PackCompositor::new(device, queue, queue_family_index, memory_properties, &pack) {
        Ok(compositor) => {
            tracing::info!(
                pack = %dir.to_string_lossy(),
                passes = compositor.pass_count(),
                "pack loaded and wired; compositing armed"
            );
            Some(compositor)
        }
        Err(error) => {
            tracing::warn!(%error, pack = %dir.to_string_lossy(), "pack does not wire; compositing disabled");
            None
        }
    }
}

pub(crate) fn device_destroyed() {
    tracing::debug!("ferridian layer: device destroyed");
}

pub(crate) fn render_pass_begun() -> GamePassKind {
    RENDER_PASSES_BEGUN.fetch_add(1, Ordering::Relaxed);
    let kind = with_observer(|observer| observer.render_pass_begun());
    tracing::trace!(kind = kind.as_str(), "ferridian layer: render pass begun");
    kind
}

pub(crate) fn label_begun(name: &str) {
    with_observer(|observer| observer.label_begun(name));
}

pub(crate) fn label_ended() {
    with_observer(|observer| observer.label_ended());
}

pub(crate) fn render_pass_count() -> u64 {
    RENDER_PASSES_BEGUN.load(Ordering::Relaxed)
}

/// Count of passes classified as `GamePassKind::ALL[kind]`; 0 if out of range.
pub(crate) fn classified_pass_count(kind: u32) -> u64 {
    let Some(&kind) = usize::try_from(kind)
        .ok()
        .and_then(|index| GamePassKind::ALL.get(index))
    else {
        return 0;
    };
    with_observer(|observer| observer.count_for(kind))
}
