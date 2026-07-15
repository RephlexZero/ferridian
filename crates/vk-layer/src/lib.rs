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

pub use hooks::PACK_ENV;
pub use loader_interface::LAYER_NAME;

use std::ffi::{CStr, c_char, c_void};
use std::sync::{Arc, Mutex};

use ash::vk;
use ash::vk::Handle;
use ferridian_engine::frame::composite_trigger;
use ferridian_vk_rt::{TapRegistry, ViewRecord};

use dispatch::{
    DeviceState, InstanceState, PfnCmdBeginDebugUtilsLabel, PfnCmdBeginRenderPass,
    PfnCmdEndDebugUtilsLabel, PfnCmdEndRenderPass, PfnCreateFramebuffer, PfnCreateImageView,
    PfnCreateRenderPass, PfnDestroyDevice, PfnDestroyFramebuffer, PfnDestroyImageView,
    PfnDestroyInstance, PfnDestroyRenderPass, dispatch_key,
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
        b"vkCreateImageView" => as_void_pfn!(PfnCreateImageView, ferridian_create_image_view),
        b"vkDestroyImageView" => as_void_pfn!(PfnDestroyImageView, ferridian_destroy_image_view),
        b"vkCreateFramebuffer" => as_void_pfn!(PfnCreateFramebuffer, ferridian_create_framebuffer),
        b"vkDestroyFramebuffer" => {
            as_void_pfn!(PfnDestroyFramebuffer, ferridian_destroy_framebuffer)
        }
        b"vkCreateRenderPass" => as_void_pfn!(PfnCreateRenderPass, ferridian_create_render_pass),
        b"vkDestroyRenderPass" => {
            as_void_pfn!(PfnDestroyRenderPass, ferridian_destroy_render_pass)
        }
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
                framebuffer: begin.framebuffer,
                render_area: begin.render_area,
                kind,
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
        let pack_active = state
            .compositor
            .lock()
            .expect("compositor lock poisoned")
            .is_some();
        // The embedded overlay is the no-pack walking skeleton; a loaded
        // pack replaces it outright.
        if !pack_active && let Some(pass) = pass.as_ref().filter(|pass| pass.classified()) {
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
        // The pack runs *after* the world-final pass has closed: its own
        // render passes cannot nest inside the app's.
        if let Some(pass) = pass.as_ref().filter(|pass| composite_trigger(pass.kind)) {
            let resolved = state
                .taps
                .lock()
                .expect("tap registry poisoned")
                .resolve(pass.render_pass, pass.framebuffer);
            // The composite trigger is the one place we already touch the
            // compositor every frame, so it doubles as the reload check: at
            // most one poll per frame, and any swap happens before this
            // frame's own compositing below.
            let mut watcher = state
                .pack_watcher
                .lock()
                .expect("pack watcher lock poisoned");
            if let Some(reloaded) = watcher.as_mut().and_then(|watcher| {
                hooks::poll_reload(
                    watcher,
                    state.device.clone(),
                    state.queue,
                    state.queue_family_index,
                    state.allocator.clone(),
                )
            }) {
                let mut compositor = state.compositor.lock().expect("compositor lock poisoned");
                // SAFETY: `device_wait_idle` only waits on work already
                // submitted in *previous* frames — `command_buffer` is still
                // being recorded and hasn't been submitted yet, so this
                // cannot deadlock against it. Once it returns, nothing
                // GPU-side references the old compositor's objects, which is
                // exactly `destroy`'s safety requirement.
                unsafe {
                    state
                        .device
                        .device_wait_idle()
                        .expect("device wait idle before hot-swapping the pack compositor");
                    if let Some(old) = compositor.as_mut() {
                        old.destroy();
                    }
                }
                *compositor = Some(reloaded);
            }
            drop(watcher);
            let mut compositor = state.compositor.lock().expect("compositor lock poisoned");
            if let (Some(compositor), Some(frame)) = (compositor.as_mut(), resolved) {
                // SAFETY: we are on the app's recording thread, immediately
                // after its render pass ended in this command buffer.
                unsafe { compositor.record_over(command_buffer, &frame) };
            } else if pack_active {
                tracing::debug!("trigger pass not resolvable to attachments; frame left untouched");
            }
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
/// Standard `vkCreateImageView` contract.
unsafe extern "system" fn ferridian_create_image_view(
    device: vk::Device,
    p_create_info: *const vk::ImageViewCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_view: *mut vk::ImageView,
) -> vk::Result {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    let Some(Some(next)) = dispatch::with_device(key, |state| state.create_image_view) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: forwarding the caller's (still valid) arguments down the chain.
    let result = unsafe { next(device, p_create_info, p_allocator, p_view) };
    if result == vk::Result::SUCCESS && !p_create_info.is_null() && !p_view.is_null() {
        // SAFETY: create info and out-pointer are valid per the contract; the
        // create succeeded so *p_view holds the new handle.
        let (info, view) = unsafe { (&*p_create_info, *p_view) };
        dispatch::with_device(key, |state| {
            state
                .taps
                .lock()
                .expect("tap registry poisoned")
                .record_view(
                    view,
                    ViewRecord {
                        image: info.image,
                        format: info.format,
                        aspect: info.subresource_range.aspect_mask,
                    },
                );
        });
    }
    result
}

/// # Safety
/// Standard `vkDestroyImageView` contract.
unsafe extern "system" fn ferridian_destroy_image_view(
    device: vk::Device,
    view: vk::ImageView,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    dispatch::with_device(key, |state| {
        if view != vk::ImageView::null() {
            let mut compositor = state.compositor.lock().expect("compositor lock poisoned");
            if let Some(compositor) = compositor.as_mut() {
                // SAFETY: called from the vkDestroyImageView hook before
                // forwarding, exactly this method's contract.
                unsafe { compositor.invalidate_view(view) };
            }
            state
                .taps
                .lock()
                .expect("tap registry poisoned")
                .forget_view(view);
        }
        if let Some(next) = state.destroy_image_view {
            // SAFETY: forwarding the caller's (still valid) arguments down the chain.
            unsafe { next(device, view, p_allocator) };
        }
    });
}

/// # Safety
/// Standard `vkCreateFramebuffer` contract.
unsafe extern "system" fn ferridian_create_framebuffer(
    device: vk::Device,
    p_create_info: *const vk::FramebufferCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_framebuffer: *mut vk::Framebuffer,
) -> vk::Result {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    let Some(Some(next)) = dispatch::with_device(key, |state| state.create_framebuffer) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: forwarding the caller's (still valid) arguments down the chain.
    let result = unsafe { next(device, p_create_info, p_allocator, p_framebuffer) };
    if result == vk::Result::SUCCESS && !p_create_info.is_null() && !p_framebuffer.is_null() {
        // SAFETY: create info and out-pointer are valid per the contract.
        let (info, framebuffer) = unsafe { (&*p_create_info, *p_framebuffer) };
        // Imageless framebuffers carry no attachment handles to record; the
        // registry then simply won't resolve passes using them.
        if !info.flags.contains(vk::FramebufferCreateFlags::IMAGELESS)
            && (info.attachment_count == 0 || !info.p_attachments.is_null())
        {
            // SAFETY: for a non-imageless framebuffer, p_attachments points
            // at attachment_count views per the Vulkan spec.
            let attachments = unsafe {
                std::slice::from_raw_parts(info.p_attachments, info.attachment_count as usize)
            }
            .to_vec();
            dispatch::with_device(key, |state| {
                state
                    .taps
                    .lock()
                    .expect("tap registry poisoned")
                    .record_framebuffer(
                        framebuffer,
                        attachments,
                        vk::Extent2D {
                            width: info.width,
                            height: info.height,
                        },
                    );
            });
        }
    }
    result
}

/// # Safety
/// Standard `vkDestroyFramebuffer` contract.
unsafe extern "system" fn ferridian_destroy_framebuffer(
    device: vk::Device,
    framebuffer: vk::Framebuffer,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    dispatch::with_device(key, |state| {
        state
            .taps
            .lock()
            .expect("tap registry poisoned")
            .forget_framebuffer(framebuffer);
        if let Some(next) = state.destroy_framebuffer {
            // SAFETY: forwarding the caller's (still valid) arguments down the chain.
            unsafe { next(device, framebuffer, p_allocator) };
        }
    });
}

/// # Safety
/// Standard `vkCreateRenderPass` contract.
unsafe extern "system" fn ferridian_create_render_pass(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_render_pass: *mut vk::RenderPass,
) -> vk::Result {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    let Some(Some(next)) = dispatch::with_device(key, |state| state.create_render_pass) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    // SAFETY: forwarding the caller's (still valid) arguments down the chain.
    let result = unsafe { next(device, p_create_info, p_allocator, p_render_pass) };
    if result == vk::Result::SUCCESS && !p_create_info.is_null() && !p_render_pass.is_null() {
        // SAFETY: create info and out-pointer are valid per the contract.
        let (info, render_pass) = unsafe { (&*p_create_info, *p_render_pass) };
        if info.attachment_count == 0 || !info.p_attachments.is_null() {
            // SAFETY: p_attachments points at attachment_count descriptions
            // per the Vulkan spec.
            let final_layouts: Vec<vk::ImageLayout> = unsafe {
                std::slice::from_raw_parts(info.p_attachments, info.attachment_count as usize)
            }
            .iter()
            .map(|attachment| attachment.final_layout)
            .collect();
            dispatch::with_device(key, |state| {
                state
                    .taps
                    .lock()
                    .expect("tap registry poisoned")
                    .record_render_pass(render_pass, final_layouts);
            });
        }
    }
    result
}

/// # Safety
/// Standard `vkDestroyRenderPass` contract.
unsafe extern "system" fn ferridian_destroy_render_pass(
    device: vk::Device,
    render_pass: vk::RenderPass,
    p_allocator: *const vk::AllocationCallbacks<'_>,
) {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    dispatch::with_device(key, |state| {
        state
            .taps
            .lock()
            .expect("tap registry poisoned")
            .forget_render_pass(render_pass);
        if let Some(next) = state.destroy_render_pass {
            // SAFETY: forwarding the caller's (still valid) arguments down the chain.
            unsafe { next(device, render_pass, p_allocator) };
        }
    });
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

    // The *down-chain* device table both compositors record through — calls
    // on it never re-enter the layer. Missing functions become panicking
    // stubs we only reach by calling an unsupported command (we stick to
    // core 1.0).
    // SAFETY: loading the table for the device that was just created.
    let device_table = unsafe {
        ash::Device::load_with(
            |name| {
                std::mem::transmute::<vk::PFN_vkVoidFunction, *const c_void>(next_gdpa(
                    device,
                    name.as_ptr(),
                ))
            },
            device,
        )
    };

    // What PackExecutor needs from the app's device: the queue family the
    // app renders on (its first requested family — vanilla's graphics
    // queue), and the physical device's memory types.
    // SAFETY: for a successful vkCreateDevice, queue_create_info_count ≥ 1
    // and the array is valid.
    let queue_family_index = unsafe {
        let info = &*p_create_info;
        if info.queue_create_info_count > 0 && !info.p_queue_create_infos.is_null() {
            (*info.p_queue_create_infos).queue_family_index
        } else {
            0
        }
    };
    // SAFETY: queue 0 of a family the app requested exists per the create info.
    let queue = unsafe { device_table.get_device_queue(queue_family_index, 0) };

    // A full ash::Instance table, built the same way device_table was built
    // above, purely so gpu-allocator can query memory/physical-device
    // properties itself rather than the layer hand-rolling that lookup.
    // SAFETY: forwarding proc-addr lookups down the chain, exactly as for
    // device_table above.
    let instance_table = unsafe {
        ash::Instance::load_with(
            |name| {
                // `vkGetDeviceProcAddr` is technically an instance-level
                // command, so ash's loader queries it here too — but this
                // loader/layer stack segfaults answering that query through
                // `next_gipa` mid-vkCreateDevice (observed on lavapipe+VVL).
                // We already have the correct pointer from the device
                // layer-info directly, so short-circuit to it.
                if name.to_bytes() == b"vkGetDeviceProcAddr" {
                    return next_gdpa as *const c_void;
                }
                std::mem::transmute::<vk::PFN_vkVoidFunction, *const c_void>(next_gipa(
                    instance,
                    name.as_ptr(),
                ))
            },
            instance,
        )
    };
    let allocator =
        gpu_allocator::vulkan::Allocator::new(&gpu_allocator::vulkan::AllocatorCreateDesc {
            instance: instance_table,
            device: device_table.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        });
    let allocator = match allocator {
        Ok(allocator) => Some(Arc::new(Mutex::new(allocator))),
        Err(error) => {
            // Pack compositing needs an allocator; everything else the layer
            // does (overlay, taps, vanilla forwarding) does not, so the
            // device is still usable — just never composites a pack.
            tracing::warn!(%error, "gpu-allocator init failed; pack compositing disabled");
            None
        }
    };
    let compositor = hooks::create_compositor(
        device_table.clone(),
        queue,
        queue_family_index,
        allocator.clone(),
    );
    let pack_watcher = hooks::create_watcher();

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
            create_image_view: retype!(resolve(c"vkCreateImageView"), PfnCreateImageView),
            destroy_image_view: retype!(resolve(c"vkDestroyImageView"), PfnDestroyImageView),
            create_framebuffer: retype!(resolve(c"vkCreateFramebuffer"), PfnCreateFramebuffer),
            destroy_framebuffer: retype!(resolve(c"vkDestroyFramebuffer"), PfnDestroyFramebuffer),
            create_render_pass: retype!(resolve(c"vkCreateRenderPass"), PfnCreateRenderPass),
            destroy_render_pass: retype!(resolve(c"vkDestroyRenderPass"), PfnDestroyRenderPass),
            overlay: Mutex::new(Overlay::new(device_table.clone())),
            taps: Mutex::new(TapRegistry::default()),
            compositor: Mutex::new(compositor),
            pack_watcher: Mutex::new(pack_watcher),
            device: device_table,
            queue,
            queue_family_index,
            allocator,
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
    // Both compositors' child objects must die before their device does (VVL
    // reports them as leaked at vkDestroyDevice otherwise).
    if let Some(overlay) = state
        .overlay
        .lock()
        .expect("overlay lock poisoned")
        .as_mut()
    {
        overlay.destroy();
    }
    if let Some(compositor) = state
        .compositor
        .lock()
        .expect("compositor lock poisoned")
        .as_mut()
    {
        // SAFETY: the app must have idled the device per vkDestroyDevice's
        // rules, so nothing in flight references the compositor's objects.
        unsafe { compositor.destroy() };
    }
    hooks::device_destroyed();
    // SAFETY: forwarding the destroy down the chain exactly once.
    unsafe { (state.destroy_device)(device, p_allocator) };
}
