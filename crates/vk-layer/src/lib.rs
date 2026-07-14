//! The Vulkan layer cdylib: Ferridian's primary interception seam.
//!
//! This crate stays *boring* by design. It implements the Khronos
//! loader↔layer contract (negotiation, proc-addr plumbing, create-info chain
//! advancement) and nothing else; all engine logic lives behind the [`hooks`]
//! seam so tests and Miri reach maximal code in `ferridian-engine`.
//!
//! Current state (M2 in progress): per-object dispatch tables keyed by
//! dispatch key ([`dispatch`]), interception of `vkCmdBeginRenderPass` and
//! the debug-utils label commands (Blaze3D's debug groups — the anchor
//! stream contract-driven pass detection classifies), with everything else
//! forwarded unmodified.

mod dispatch;
mod hooks;
mod loader_interface;
mod overlay;

pub use loader_interface::LAYER_NAME;

use std::ffi::{CStr, c_char};
use std::sync::Mutex;

use ash::vk;
use ash::vk::Handle;
use ferridian_contract::GamePassKind;

use dispatch::{
    DeviceState, InstanceState, PfnCmdBeginDebugUtilsLabel, PfnCmdBeginRenderPass,
    PfnCmdEndDebugUtilsLabel, PfnCmdEndRenderPass, PfnDestroyDevice, PfnDestroyInstance,
    dispatch_key,
};
use loader_interface::{
    LayerFunction, NEGOTIATE_INTERFACE_STRUCT, PfnCreateDevice, PfnCreateInstance,
    VkLayerDeviceCreateInfo, VkLayerInstanceCreateInfo, VkNegotiateLayerInterface,
};
use overlay::{ActivePass, Overlay};

/// Render passes begun through the layer, readable across the cdylib
/// boundary (tests dlopen the layer and assert interception actually
/// happened; the count is Relaxed-accurate, which is all a probe needs).
#[unsafe(no_mangle)]
pub extern "system" fn ferridian_layer_render_pass_count() -> u64 {
    hooks::render_pass_count()
}

/// Render passes classified as the given [`ferridian_contract::GamePassKind`]
/// (indexed per `GamePassKind::ALL` wire order). Out-of-range kinds return 0.
#[unsafe(no_mangle)]
pub extern "system" fn ferridian_layer_classified_pass_count(kind: u32) -> u64 {
    hooks::classified_pass_count(kind)
}

/// Erase a typed function pointer into the loader's `PFN_vkVoidFunction`.
macro_rules! as_void_pfn {
    ($typed:ty, $function:expr) => {
        // SAFETY: transmuting a fn pointer to the erased PFN_vkVoidFunction;
        // the receiver transmutes it back to exactly $typed per the spec.
        Some(unsafe { std::mem::transmute::<$typed, unsafe extern "system" fn()>($function) })
    };
}

/// Entry point named in the layer JSON manifest; the loader calls this first.
///
/// # Safety
/// Called by the Vulkan loader with a pointer to a live negotiation struct.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn vkNegotiateLoaderLayerInterfaceVersion(
    p_version_struct: *mut VkNegotiateLayerInterface,
) -> vk::Result {
    if p_version_struct.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    // SAFETY: non-null, loader-owned struct, valid for the duration of the call.
    let negotiate = unsafe { &mut *p_version_struct };
    if negotiate.s_type != NEGOTIATE_INTERFACE_STRUCT
        || negotiate.loader_layer_interface_version < 2
    {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    negotiate.loader_layer_interface_version = 2;
    negotiate.pfn_get_instance_proc_addr =
        Some(ferridian_get_instance_proc_addr as vk::PFN_vkGetInstanceProcAddr);
    negotiate.pfn_get_device_proc_addr =
        Some(ferridian_get_device_proc_addr as vk::PFN_vkGetDeviceProcAddr);
    negotiate.pfn_get_physical_device_proc_addr = None;
    vk::Result::SUCCESS
}

/// # Safety
/// Standard `vkGetInstanceProcAddr` contract: `p_name` is a NUL-terminated string.
pub unsafe extern "system" fn ferridian_get_instance_proc_addr(
    instance: vk::Instance,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }
    // SAFETY: caller guarantees a NUL-terminated string per the Vulkan spec.
    let name = unsafe { CStr::from_ptr(p_name) };
    match name.to_bytes() {
        b"vkGetInstanceProcAddr" => as_void_pfn!(
            vk::PFN_vkGetInstanceProcAddr,
            ferridian_get_instance_proc_addr
        ),
        b"vkGetDeviceProcAddr" => {
            as_void_pfn!(vk::PFN_vkGetDeviceProcAddr, ferridian_get_device_proc_addr)
        }
        b"vkCreateInstance" => as_void_pfn!(PfnCreateInstance, ferridian_create_instance),
        b"vkDestroyInstance" => as_void_pfn!(PfnDestroyInstance, ferridian_destroy_instance),
        b"vkCreateDevice" => as_void_pfn!(PfnCreateDevice, ferridian_create_device),
        _ => {
            if instance == vk::Instance::null() {
                return None;
            }
            // SAFETY: a non-null instance handle passed to GIPA is live and
            // dispatchable per the Vulkan spec.
            let key = unsafe { dispatch_key(instance.as_raw()) };
            let gipa = dispatch::with_instance(key, |state| state.gipa)?;
            // SAFETY: forwarding the unmodified query down the chain.
            unsafe { gipa(instance, p_name) }
        }
    }
}

/// # Safety
/// Standard `vkGetDeviceProcAddr` contract.
pub unsafe extern "system" fn ferridian_get_device_proc_addr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if p_name.is_null() || device == vk::Device::null() {
        return None;
    }
    // SAFETY: caller guarantees a NUL-terminated string per the Vulkan spec.
    let name = unsafe { CStr::from_ptr(p_name) };
    match name.to_bytes() {
        b"vkGetDeviceProcAddr" => {
            as_void_pfn!(vk::PFN_vkGetDeviceProcAddr, ferridian_get_device_proc_addr)
        }
        b"vkDestroyDevice" => as_void_pfn!(PfnDestroyDevice, ferridian_destroy_device),
        b"vkCmdBeginRenderPass" => {
            as_void_pfn!(PfnCmdBeginRenderPass, ferridian_cmd_begin_render_pass)
        }
        b"vkCmdEndRenderPass" => {
            as_void_pfn!(PfnCmdEndRenderPass, ferridian_cmd_end_render_pass)
        }
        b"vkCmdBeginDebugUtilsLabelEXT" => as_void_pfn!(
            PfnCmdBeginDebugUtilsLabel,
            ferridian_cmd_begin_debug_utils_label
        ),
        b"vkCmdEndDebugUtilsLabelEXT" => as_void_pfn!(
            PfnCmdEndDebugUtilsLabel,
            ferridian_cmd_end_debug_utils_label
        ),
        _ => {
            // SAFETY: a non-null device handle passed to GDPA is live and
            // dispatchable per the Vulkan spec.
            let key = unsafe { dispatch_key(device.as_raw()) };
            let gdpa = dispatch::with_device(key, |state| state.gdpa)?;
            // SAFETY: forwarding the unmodified query down the chain.
            unsafe { gdpa(device, p_name) }
        }
    }
}

/// # Safety
/// Standard `vkCmdBeginRenderPass` contract; called by the dispatch chain
/// only after our GDPA handed this pointer out.
unsafe extern "system" fn ferridian_cmd_begin_render_pass(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo<'_>,
    contents: vk::SubpassContents,
) {
    let kind = hooks::render_pass_begun();
    if !p_render_pass_begin.is_null() {
        // SAFETY: a non-null begin info is live for the duration of the call.
        let begin = unsafe { &*p_render_pass_begin };
        overlay::pass_begun(
            command_buffer,
            ActivePass {
                render_pass: begin.render_pass,
                render_area: begin.render_area,
                // The compositor only ever touches passes the contract
                // classified; Unknown is forwarded untouched.
                composite: kind != GamePassKind::Unknown,
            },
        );
    }
    // SAFETY: command buffers are dispatchable handles carrying their
    // device's dispatch key.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_begin_render_pass).flatten();
    if let Some(next) = next {
        // SAFETY: forwarding the caller's (still valid) arguments down the chain.
        unsafe { next(command_buffer, p_render_pass_begin, contents) };
    }
}

/// # Safety
/// Standard `vkCmdEndRenderPass` contract; called by the dispatch chain
/// only after our GDPA handed this pointer out.
unsafe extern "system" fn ferridian_cmd_end_render_pass(command_buffer: vk::CommandBuffer) {
    let pass = overlay::pass_ending(command_buffer);
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    dispatch::with_device(key, |state| {
        if let Some(pass) = pass.as_ref().filter(|pass| pass.composite) {
            let mut overlay = state.overlay.lock().expect("overlay lock poisoned");
            if let Some(overlay) = overlay.as_mut() {
                // Inject *before* the pass closes, after every app draw in it.
                overlay.composite(command_buffer, pass);
            }
        }
        if let Some(next) = state.cmd_end_render_pass {
            // SAFETY: forwarding the caller's (still valid) argument down the chain.
            unsafe { next(command_buffer) };
        }
    });
}

/// # Safety
/// Standard `vkCmdBeginDebugUtilsLabelEXT` contract; `p_label` points to a
/// live label struct with a NUL-terminated name.
unsafe extern "system" fn ferridian_cmd_begin_debug_utils_label(
    command_buffer: vk::CommandBuffer,
    p_label: *const vk::DebugUtilsLabelEXT<'_>,
) {
    // SAFETY: label struct and its name pointer are valid per the contract.
    let name = unsafe {
        (!p_label.is_null())
            .then(|| (*p_label).p_label_name)
            .filter(|name| !name.is_null())
            .map(|name| CStr::from_ptr(name))
    };
    if let Some(name) = name {
        hooks::label_begun(&name.to_string_lossy());
    }
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_begin_debug_utils_label).flatten();
    if let Some(next) = next {
        // SAFETY: forwarding the caller's (still valid) arguments down the chain.
        unsafe { next(command_buffer, p_label) };
    }
}

/// # Safety
/// Standard `vkCmdEndDebugUtilsLabelEXT` contract.
unsafe extern "system" fn ferridian_cmd_end_debug_utils_label(command_buffer: vk::CommandBuffer) {
    hooks::label_ended();
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_end_debug_utils_label).flatten();
    if let Some(next) = next {
        // SAFETY: forwarding the caller's (still valid) argument down the chain.
        unsafe { next(command_buffer) };
    }
}

/// # Safety
/// Standard `vkCreateInstance` contract, called through the loader with a
/// layer create-info chain in `p_create_info->pNext`.
unsafe extern "system" fn ferridian_create_instance(
    p_create_info: *const vk::InstanceCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_instance: *mut vk::Instance,
) -> vk::Result {
    if p_create_info.is_null() || p_instance.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    // SAFETY: walking the loader-provided pNext chain of a valid create info.
    let link = unsafe {
        loader_interface::find_layer_link::<VkLayerInstanceCreateInfo>(
            (*p_create_info).p_next,
            vk::StructureType::LOADER_INSTANCE_CREATE_INFO,
            LayerFunction::LayerLinkInfo,
        )
    };
    let Some(chain_info) = link else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: chain_info was validated by find_layer_link; the loader owns the
    // link list and expects layers to advance it exactly once.
    let next_gipa = unsafe {
        let layer_info = (*chain_info).u.p_layer_info;
        if layer_info.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let next = (*layer_info).pfn_next_get_instance_proc_addr;
        (*chain_info).u.p_layer_info = (*layer_info).p_next;
        next
    };
    let Some(next_gipa) = next_gipa else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: querying the down-chain vkCreateInstance from the loader.
    let create = unsafe { next_gipa(vk::Instance::null(), c"vkCreateInstance".as_ptr()) };
    let Some(create) = create else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: the loader hands back the down-chain vkCreateInstance here.
    let create =
        unsafe { std::mem::transmute::<unsafe extern "system" fn(), PfnCreateInstance>(create) };
    // SAFETY: forwarding the caller's (still valid) arguments down the chain.
    let result = unsafe { create(p_create_info, p_allocator, p_instance) };
    if result != vk::Result::SUCCESS {
        return result;
    }
    // SAFETY: on success the loader guarantees *p_instance is a live
    // dispatchable handle.
    let (instance, key) = unsafe { (*p_instance, dispatch_key((*p_instance).as_raw())) };
    // SAFETY: querying the down-chain vkDestroyInstance for the new instance.
    let destroy = unsafe { next_gipa(instance, c"vkDestroyInstance".as_ptr()) };
    let Some(destroy) = destroy else {
        // Core function missing below us: the chain is unusable.
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    dispatch::insert_instance(
        key,
        InstanceState {
            instance,
            gipa: next_gipa,
            // SAFETY: the down-chain vkDestroyInstance has exactly this type.
            destroy_instance: unsafe {
                std::mem::transmute::<unsafe extern "system" fn(), PfnDestroyInstance>(destroy)
            },
        },
    );
    hooks::instance_created();
    result
}

/// # Safety
/// Standard `vkDestroyInstance` contract.
unsafe extern "system" fn ferridian_destroy_instance(
    instance: vk::Instance,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if instance == vk::Instance::null() {
        return;
    }
    // SAFETY: a non-null instance passed to vkDestroyInstance is still live.
    let key = unsafe { dispatch_key(instance.as_raw()) };
    let Some(state) = dispatch::remove_instance(key) else {
        // Not ours (or already gone) — nothing to forward to.
        return;
    };
    hooks::instance_destroyed();
    // SAFETY: forwarding the destroy down the chain exactly once.
    unsafe { (state.destroy_instance)(instance, p_allocator) };
}

/// # Safety
/// Standard `vkCreateDevice` contract, called through the loader with a
/// layer create-info chain in `p_create_info->pNext`.
unsafe extern "system" fn ferridian_create_device(
    physical_device: vk::PhysicalDevice,
    p_create_info: *const vk::DeviceCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_device: *mut vk::Device,
) -> vk::Result {
    if p_create_info.is_null() || p_device.is_null() {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    }
    // SAFETY: walking the loader-provided pNext chain of a valid create info.
    let link = unsafe {
        loader_interface::find_layer_link::<VkLayerDeviceCreateInfo>(
            (*p_create_info).p_next,
            vk::StructureType::LOADER_DEVICE_CREATE_INFO,
            LayerFunction::LayerLinkInfo,
        )
    };
    let Some(chain_info) = link else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: as in ferridian_create_instance.
    let (next_gipa, next_gdpa) = unsafe {
        let layer_info = (*chain_info).u.p_layer_info;
        if layer_info.is_null() {
            return vk::Result::ERROR_INITIALIZATION_FAILED;
        }
        let gipa = (*layer_info).pfn_next_get_instance_proc_addr;
        let gdpa = (*layer_info).pfn_next_get_device_proc_addr;
        (*chain_info).u.p_layer_info = (*layer_info).p_next;
        (gipa, gdpa)
    };
    let (Some(next_gipa), Some(next_gdpa)) = (next_gipa, next_gdpa) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // Physical devices carry their instance's dispatch key — that lookup is
    // how a device create finds the instance it belongs to.
    // SAFETY: the physical device is a live dispatchable handle.
    let instance_key = unsafe { dispatch_key(physical_device.as_raw()) };
    let Some(instance) = dispatch::with_instance(instance_key, |state| state.instance) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: querying the down-chain vkCreateDevice from the loader.
    let create = unsafe { next_gipa(instance, c"vkCreateDevice".as_ptr()) };
    let Some(create) = create else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: the loader hands back the down-chain vkCreateDevice here.
    let create =
        unsafe { std::mem::transmute::<unsafe extern "system" fn(), PfnCreateDevice>(create) };
    // SAFETY: forwarding the caller's (still valid) arguments down the chain.
    let result = unsafe { create(physical_device, p_create_info, p_allocator, p_device) };
    if result != vk::Result::SUCCESS {
        return result;
    }
    // SAFETY: on success the loader guarantees *p_device is a live
    // dispatchable handle.
    let (device, key) = unsafe { (*p_device, dispatch_key((*p_device).as_raw())) };
    // SAFETY: resolving down-chain commands for the new device.
    let resolve = |name: &CStr| unsafe { next_gdpa(device, name.as_ptr()) };
    let Some(destroy) = resolve(c"vkDestroyDevice") else {
        // Core function missing below us: the chain is unusable.
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    /// Retype a resolved down-chain pointer. Each caller names the command it
    /// resolved, whose Vulkan-spec type is exactly the target type; extension
    /// commands may legitimately resolve to `None`.
    macro_rules! retype {
        ($pfn:expr, $typed:ty) => {
            // SAFETY: per the macro's contract above.
            $pfn.map(|pfn| unsafe {
                std::mem::transmute::<unsafe extern "system" fn(), $typed>(pfn)
            })
        };
    }
    // SAFETY: the down-chain vkDestroyDevice has exactly this type.
    let destroy_device =
        unsafe { std::mem::transmute::<unsafe extern "system" fn(), PfnDestroyDevice>(destroy) };
    dispatch::insert_device(
        key,
        DeviceState {
            gdpa: next_gdpa,
            destroy_device,
            cmd_begin_render_pass: retype!(resolve(c"vkCmdBeginRenderPass"), PfnCmdBeginRenderPass),
            cmd_end_render_pass: retype!(resolve(c"vkCmdEndRenderPass"), PfnCmdEndRenderPass),
            cmd_begin_debug_utils_label: retype!(
                resolve(c"vkCmdBeginDebugUtilsLabelEXT"),
                PfnCmdBeginDebugUtilsLabel
            ),
            cmd_end_debug_utils_label: retype!(
                resolve(c"vkCmdEndDebugUtilsLabelEXT"),
                PfnCmdEndDebugUtilsLabel
            ),
            overlay: Mutex::new(Overlay::new(next_gdpa, device)),
        },
    );
    hooks::device_created();
    result
}

/// # Safety
/// Standard `vkDestroyDevice` contract.
unsafe extern "system" fn ferridian_destroy_device(
    device: vk::Device,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    if device == vk::Device::null() {
        return;
    }
    // SAFETY: a non-null device passed to vkDestroyDevice is still live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    let Some(state) = dispatch::remove_device(key) else {
        // Not ours (or already gone) — nothing to forward to.
        return;
    };
    // The overlay's child objects must die before their device does (VVL
    // reports them as leaked at vkDestroyDevice otherwise).
    if let Some(overlay) = state
        .overlay
        .lock()
        .expect("overlay lock poisoned")
        .as_mut()
    {
        overlay.destroy();
    }
    hooks::device_destroyed();
    // SAFETY: forwarding the destroy down the chain exactly once.
    unsafe { (state.destroy_device)(device, p_allocator) };
}
