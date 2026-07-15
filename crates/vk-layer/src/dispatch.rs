//! Per-object dispatch tables, keyed the way every Khronos sample layer keys
//! them: by *dispatch key*.
//!
//! The loader writes a dispatch-table pointer as the first pointer-sized word
//! of every dispatchable handle it creates, and objects that share a dispatch
//! table share that word — physical devices carry their instance's, command
//! buffers and queues carry their device's. Reading that word therefore maps
//! any dispatchable handle to the state we stored for its instance/device at
//! create time, which is exactly the lookup a layer needs in a command hook
//! where only a `VkCommandBuffer` is in hand.

use std::collections::BTreeMap;
use std::sync::{Mutex, RwLock};

use ash::vk;
use ferridian_engine::pack::PackWatcher;
use ferridian_vk_rt::{PackCompositor, TapRegistry};

use crate::overlay::Overlay;

pub(crate) type PfnDestroyInstance =
    for<'a> unsafe extern "system" fn(vk::Instance, *const vk::AllocationCallbacks<'a>);

pub(crate) type PfnDestroyDevice =
    for<'a> unsafe extern "system" fn(vk::Device, *const vk::AllocationCallbacks<'a>);

pub(crate) type PfnCmdBeginRenderPass = for<'a> unsafe extern "system" fn(
    vk::CommandBuffer,
    *const vk::RenderPassBeginInfo<'a>,
    vk::SubpassContents,
);

pub(crate) type PfnCmdEndRenderPass = unsafe extern "system" fn(vk::CommandBuffer);

pub(crate) type PfnCmdBeginDebugUtilsLabel =
    for<'a> unsafe extern "system" fn(vk::CommandBuffer, *const vk::DebugUtilsLabelEXT<'a>);

pub(crate) type PfnCmdEndDebugUtilsLabel = unsafe extern "system" fn(vk::CommandBuffer);

pub(crate) type PfnCreateImageView = for<'a> unsafe extern "system" fn(
    vk::Device,
    *const vk::ImageViewCreateInfo<'a>,
    *const vk::AllocationCallbacks<'a>,
    *mut vk::ImageView,
) -> vk::Result;

pub(crate) type PfnDestroyImageView = for<'a> unsafe extern "system" fn(
    vk::Device,
    vk::ImageView,
    *const vk::AllocationCallbacks<'a>,
);

pub(crate) type PfnCreateFramebuffer = for<'a> unsafe extern "system" fn(
    vk::Device,
    *const vk::FramebufferCreateInfo<'a>,
    *const vk::AllocationCallbacks<'a>,
    *mut vk::Framebuffer,
) -> vk::Result;

pub(crate) type PfnDestroyFramebuffer = for<'a> unsafe extern "system" fn(
    vk::Device,
    vk::Framebuffer,
    *const vk::AllocationCallbacks<'a>,
);

pub(crate) type PfnCreateRenderPass = for<'a> unsafe extern "system" fn(
    vk::Device,
    *const vk::RenderPassCreateInfo<'a>,
    *const vk::AllocationCallbacks<'a>,
    *mut vk::RenderPass,
) -> vk::Result;

pub(crate) type PfnDestroyRenderPass = for<'a> unsafe extern "system" fn(
    vk::Device,
    vk::RenderPass,
    *const vk::AllocationCallbacks<'a>,
);

/// Down-chain state for one live `VkInstance`.
pub(crate) struct InstanceState {
    pub instance: vk::Instance,
    pub gipa: vk::PFN_vkGetInstanceProcAddr,
    pub destroy_instance: PfnDestroyInstance,
}

/// Down-chain state for one live `VkDevice`. The command pointers are
/// resolved once at device creation; `None` means the chain below us does
/// not provide the function (legal for extension commands — we then observe
/// without forwarding).
pub(crate) struct DeviceState {
    pub gdpa: vk::PFN_vkGetDeviceProcAddr,
    pub destroy_device: PfnDestroyDevice,
    pub cmd_begin_render_pass: Option<PfnCmdBeginRenderPass>,
    pub cmd_end_render_pass: Option<PfnCmdEndRenderPass>,
    pub cmd_begin_debug_utils_label: Option<PfnCmdBeginDebugUtilsLabel>,
    pub cmd_end_debug_utils_label: Option<PfnCmdEndDebugUtilsLabel>,
    pub create_image_view: Option<PfnCreateImageView>,
    pub destroy_image_view: Option<PfnDestroyImageView>,
    pub create_framebuffer: Option<PfnCreateFramebuffer>,
    pub destroy_framebuffer: Option<PfnDestroyFramebuffer>,
    pub create_render_pass: Option<PfnCreateRenderPass>,
    pub destroy_render_pass: Option<PfnDestroyRenderPass>,
    /// The overlay's per-device Vulkan objects; `None` when creation
    /// failed (the embedded overlay then stays off for this device).
    pub overlay: Mutex<Option<Overlay>>,
    /// The application's live views/framebuffers/render passes, fed from the
    /// create/destroy hooks so a begun pass can be resolved to attachments.
    pub taps: Mutex<TapRegistry>,
    /// Present when `FERRIDIAN_PACK` named a loadable, wireable pack at
    /// device creation; replaces the embedded overlay when active.
    pub compositor: Mutex<Option<PackCompositor>>,
    /// Watches the same directory for a newer generation; `None` when
    /// `FERRIDIAN_PACK` was unset at device creation. Rebuilding a
    /// [`PackCompositor`] on reload needs the same device/queue/memory
    /// parameters `compositor` was originally built with, so they're kept
    /// here too rather than only inside the (possibly absent) compositor.
    pub pack_watcher: Mutex<Option<PackWatcher>>,
    pub device: ash::Device,
    pub queue: vk::Queue,
    pub queue_family_index: u32,
    pub memory_properties: vk::PhysicalDeviceMemoryProperties,
}

static INSTANCES: RwLock<BTreeMap<usize, InstanceState>> = RwLock::new(BTreeMap::new());
static DEVICES: RwLock<BTreeMap<usize, DeviceState>> = RwLock::new(BTreeMap::new());

/// Read a dispatchable handle's dispatch key.
///
/// # Safety
/// `raw` must be the raw value of a live *dispatchable* Vulkan handle (its
/// first pointer-sized word is loader-owned and readable).
pub(crate) unsafe fn dispatch_key(raw: u64) -> usize {
    // SAFETY: per this function's contract.
    unsafe { *(raw as usize as *const usize) }
}

pub(crate) fn insert_instance(key: usize, state: InstanceState) {
    INSTANCES
        .write()
        .expect("instance registry poisoned")
        .insert(key, state);
}

pub(crate) fn remove_instance(key: usize) -> Option<InstanceState> {
    INSTANCES
        .write()
        .expect("instance registry poisoned")
        .remove(&key)
}

/// Look up instance state and copy out what the caller needs. A closure keeps
/// the read lock's scope explicit and the state from escaping it.
pub(crate) fn with_instance<T>(key: usize, read: impl FnOnce(&InstanceState) -> T) -> Option<T> {
    INSTANCES
        .read()
        .expect("instance registry poisoned")
        .get(&key)
        .map(read)
}

pub(crate) fn insert_device(key: usize, state: DeviceState) {
    DEVICES
        .write()
        .expect("device registry poisoned")
        .insert(key, state);
}

pub(crate) fn remove_device(key: usize) -> Option<DeviceState> {
    DEVICES
        .write()
        .expect("device registry poisoned")
        .remove(&key)
}

pub(crate) fn with_device<T>(key: usize, read: impl FnOnce(&DeviceState) -> T) -> Option<T> {
    DEVICES
        .read()
        .expect("device registry poisoned")
        .get(&key)
        .map(read)
}

#[cfg(test)]
mod tests {
    use super::*;

    unsafe extern "system" fn fake_gipa(
        _instance: vk::Instance,
        _name: *const std::ffi::c_char,
    ) -> vk::PFN_vkVoidFunction {
        None
    }

    unsafe extern "system" fn fake_destroy_instance(
        _instance: vk::Instance,
        _allocator: *const vk::AllocationCallbacks<'_>,
    ) {
    }

    /// Two fake "handles" whose first word plays the loader's dispatch-table
    /// pointer: distinct objects sharing the word share the key, exactly the
    /// property instance↔physical-device and device↔command-buffer lookups
    /// rely on.
    #[test]
    fn shared_first_word_means_shared_key() {
        let table_a: usize = 0xA11CE;
        let table_b: usize = 0xB0B;
        let instance_like = Box::new(table_a);
        let physical_device_like = Box::new(table_a);
        let other_instance_like = Box::new(table_b);

        // SAFETY: the boxes are live and start with a readable usize.
        let (key_i, key_pd, key_other) = unsafe {
            (
                dispatch_key(&raw const *instance_like as u64),
                dispatch_key(&raw const *physical_device_like as u64),
                dispatch_key(&raw const *other_instance_like as u64),
            )
        };
        assert_eq!(key_i, key_pd);
        assert_ne!(key_i, key_other);
    }

    #[test]
    fn registry_insert_lookup_remove_roundtrip() {
        let key = 0xF00D;
        insert_instance(
            key,
            InstanceState {
                instance: vk::Instance::null(),
                gipa: fake_gipa,
                destroy_instance: fake_destroy_instance,
            },
        );
        assert_eq!(
            with_instance(key, |state| state.instance),
            Some(vk::Instance::null())
        );
        assert!(remove_instance(key).is_some());
        assert!(with_instance(key, |_| ()).is_none());
        assert!(remove_instance(key).is_none());
    }
}
