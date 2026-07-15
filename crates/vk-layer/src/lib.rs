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
    PfnCmdBeginRenderPass2, PfnCmdBeginRendering, PfnCmdEndDebugUtilsLabel, PfnCmdEndRenderPass,
    PfnCmdEndRenderPass2, PfnCmdEndRendering, PfnCreateFramebuffer, PfnCreateImage,
    PfnCreateImageView, PfnCreateRenderPass, PfnCreateRenderPass2, PfnDestroyDevice,
    PfnDestroyFramebuffer, PfnDestroyImageView, PfnDestroyInstance, PfnDestroyRenderPass,
    dispatch_key,
};
use loader_interface::{
    LayerFunction, NEGOTIATE_INTERFACE_STRUCT, PfnCreateDevice, PfnCreateInstance,
    VkLayerDeviceCreateInfo, VkLayerInstanceCreateInfo, VkNegotiateLayerInterface,
};
use overlay::{ActivePass, Overlay, PassGeometry};

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
        // `VK_EXT_debug_utils` is an *instance* extension, so its `Cmd*`
        // commands — despite taking a `VkCommandBuffer` — are resolved via
        // `vkGetInstanceProcAddr`, not `vkGetDeviceProcAddr` (confirmed
        // empirically: the loader/ash never queries these by name through
        // our GDPA at all). Intercepting them only in
        // `ferridian_get_device_proc_addr` left classification silently dead
        // whenever this layer sits above `VK_LAYER_KHRONOS_validation` in
        // the chain — the hooks themselves are unchanged, they just also
        // need to be reachable from here.
        b"vkCmdBeginDebugUtilsLabelEXT" => as_void_pfn!(
            PfnCmdBeginDebugUtilsLabel,
            ferridian_cmd_begin_debug_utils_label
        ),
        b"vkCmdEndDebugUtilsLabelEXT" => as_void_pfn!(
            PfnCmdEndDebugUtilsLabel,
            ferridian_cmd_end_debug_utils_label
        ),
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
        b"vkCreateImage" => as_void_pfn!(PfnCreateImage, ferridian_create_image),
        b"vkCreateRenderPass2" | b"vkCreateRenderPass2KHR" => {
            as_void_pfn!(PfnCreateRenderPass2, ferridian_create_render_pass2)
        }
        b"vkCmdBeginRenderPass2" | b"vkCmdBeginRenderPass2KHR" => {
            as_void_pfn!(PfnCmdBeginRenderPass2, ferridian_cmd_begin_render_pass2)
        }
        b"vkCmdEndRenderPass2" | b"vkCmdEndRenderPass2KHR" => {
            as_void_pfn!(PfnCmdEndRenderPass2, ferridian_cmd_end_render_pass2)
        }
        b"vkCmdBeginRendering" | b"vkCmdBeginRenderingKHR" => {
            as_void_pfn!(PfnCmdBeginRendering, ferridian_cmd_begin_rendering)
        }
        b"vkCmdEndRendering" | b"vkCmdEndRenderingKHR" => {
            as_void_pfn!(PfnCmdEndRendering, ferridian_cmd_end_rendering)
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
    // SAFETY: the begin info (when non-null) is live for the duration of the
    // call; see `begin_render_pass_common`.
    unsafe { begin_render_pass_common(command_buffer, p_render_pass_begin) };
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
/// Standard `vkCmdBeginRenderPass2{,KHR}` contract; called by the dispatch
/// chain only after our GDPA handed this pointer out.
unsafe extern "system" fn ferridian_cmd_begin_render_pass2(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo<'_>,
    p_subpass_begin_info: *const vk::SubpassBeginInfo<'_>,
) {
    // SAFETY: as in ferridian_cmd_begin_render_pass — `RenderPass2`'s begin
    // info is the exact same `VkRenderPassBeginInfo` the classic path uses.
    unsafe { begin_render_pass_common(command_buffer, p_render_pass_begin) };
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_begin_render_pass2).flatten();
    if let Some(next) = next {
        // SAFETY: forwarding the caller's (still valid) arguments down the chain.
        unsafe { next(command_buffer, p_render_pass_begin, p_subpass_begin_info) };
    }
}

/// Shared tail of `vkCmdBeginRenderPass`/`vkCmdBeginRenderPass2{,KHR}`:
/// classify against the label stack and register the active pass. Both
/// commands share the identical `VkRenderPassBeginInfo`.
///
/// # Safety
/// `p_render_pass_begin`, when non-null, is live for the duration of the call.
unsafe fn begin_render_pass_common(
    command_buffer: vk::CommandBuffer,
    p_render_pass_begin: *const vk::RenderPassBeginInfo<'_>,
) {
    let kind = hooks::render_pass_begun();
    if !p_render_pass_begin.is_null() {
        // SAFETY: per this function's contract.
        let begin = unsafe { &*p_render_pass_begin };
        overlay::pass_begun(
            command_buffer,
            ActivePass {
                geometry: PassGeometry::RenderPass {
                    render_pass: begin.render_pass,
                    framebuffer: begin.framebuffer,
                },
                render_area: begin.render_area,
                kind,
            },
        );
    }
}

/// # Safety
/// Standard `vkCmdEndRenderPass` contract; called by the dispatch chain
/// only after our GDPA handed this pointer out.
unsafe extern "system" fn ferridian_cmd_end_render_pass(command_buffer: vk::CommandBuffer) {
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_end_render_pass).flatten();
    // SAFETY: the forwarded call's arguments are exactly what the caller
    // passed us, still valid for its duration.
    unsafe {
        end_render_pass_common(command_buffer, move || {
            if let Some(next) = next {
                next(command_buffer);
            }
        });
    }
}

/// # Safety
/// Standard `vkCmdEndRenderPass2{,KHR}` contract; called by the dispatch
/// chain only after our GDPA handed this pointer out.
unsafe extern "system" fn ferridian_cmd_end_render_pass2(
    command_buffer: vk::CommandBuffer,
    p_subpass_end_info: *const vk::SubpassEndInfo<'_>,
) {
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_end_render_pass2).flatten();
    // SAFETY: as in ferridian_cmd_end_render_pass.
    unsafe {
        end_render_pass_common(command_buffer, move || {
            if let Some(next) = next {
                next(command_buffer, p_subpass_end_info);
            }
        });
    }
}

/// Shared tail of every `vkCmdEndRenderPass*`/`vkCmdEndRendering*` hook: pop
/// the active pass, run the embedded-overlay skeleton, forward the real end
/// call via `forward` (which must run at exactly the point the real
/// end-of-pass call belongs — after the overlay's draw, which must land
/// before the pass closes, and before the pack compositor's own, separate
/// render passes), then run the pack compositor/hot-reload if this was the
/// world-final pass.
///
/// # Safety
/// `forward` must perform exactly the down-chain call this hook is standing
/// in for, with the caller's original (still-valid) arguments.
unsafe fn end_render_pass_common(command_buffer: vk::CommandBuffer, forward: impl FnOnce()) {
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
        forward();
        // The pack runs *after* the world-final pass has closed: its own
        // render passes cannot nest inside the app's.
        if let Some(pass) = pass.as_ref().filter(|pass| composite_trigger(pass.kind)) {
            let resolved = pass.resolve(&state.taps.lock().expect("tap registry poisoned"));
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
/// Standard `vkCmdBeginRendering{,KHR}` contract; called by the dispatch
/// chain only after our GDPA handed this pointer out.
unsafe extern "system" fn ferridian_cmd_begin_rendering(
    command_buffer: vk::CommandBuffer,
    p_rendering_info: *const vk::RenderingInfo<'_>,
) {
    let kind = hooks::render_pass_begun();
    if !p_rendering_info.is_null() {
        // SAFETY: a non-null rendering info is live for the duration of the
        // call, and its attachment array (when present) has exactly
        // `color_attachment_count` elements, per the Vulkan spec.
        let resolved =
            unsafe { resolve_dynamic_rendering_frame(command_buffer, &*p_rendering_info) };
        // SAFETY: as above.
        let render_area = unsafe { (*p_rendering_info).render_area };
        overlay::pass_begun(
            command_buffer,
            ActivePass {
                geometry: PassGeometry::Dynamic { resolved },
                render_area,
                kind,
            },
        );
    }
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_begin_rendering).flatten();
    if let Some(next) = next {
        // SAFETY: forwarding the caller's (still valid) arguments down the chain.
        unsafe { next(command_buffer, p_rendering_info) };
    }
}

/// Resolve a dynamic-rendering pass's attachments straight from its
/// `VkRenderingInfo` — there is no framebuffer/render-pass indirection to
/// walk. `None` when there's no color attachment (nothing to composite into)
/// or its view is unknown; a `p_depth_attachment` with an unknown view just
/// degrades to no depth (`TapRegistry::resolve_views`'s contract).
///
/// # Safety
/// `info`'s attachment pointers (when non-null) point at their declared
/// counts, per the Vulkan spec.
unsafe fn resolve_dynamic_rendering_frame(
    command_buffer: vk::CommandBuffer,
    info: &vk::RenderingInfo<'_>,
) -> Option<ferridian_vk_rt::ResolvedFrame> {
    if info.color_attachment_count == 0 || info.p_color_attachments.is_null() {
        return None;
    }
    // SAFETY: per this function's contract.
    let color_attachments = unsafe {
        std::slice::from_raw_parts(
            info.p_color_attachments,
            info.color_attachment_count as usize,
        )
    };
    let color = color_attachments
        .first()
        .filter(|attachment| attachment.image_view != vk::ImageView::null())?;
    let depth = if info.p_depth_attachment.is_null() {
        None
    } else {
        // SAFETY: a non-null p_depth_attachment points at one live struct.
        let attachment = unsafe { &*info.p_depth_attachment };
        (attachment.image_view != vk::ImageView::null()).then_some(attachment)
    };
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    dispatch::with_device(key, |state| {
        state
            .taps
            .lock()
            .expect("tap registry poisoned")
            .resolve_views(
                info.render_area.extent,
                (color.image_view, color.image_layout),
                depth.map(|attachment| (attachment.image_view, attachment.image_layout)),
            )
    })
    .flatten()
}

/// # Safety
/// Standard `vkCmdEndRendering{,KHR}` contract; called by the dispatch chain
/// only after our GDPA handed this pointer out.
unsafe extern "system" fn ferridian_cmd_end_rendering(command_buffer: vk::CommandBuffer) {
    // SAFETY: dispatchable handle; see ferridian_cmd_begin_render_pass.
    let key = unsafe { dispatch_key(command_buffer.as_raw()) };
    let next = dispatch::with_device(key, |state| state.cmd_end_rendering).flatten();
    // SAFETY: as in ferridian_cmd_end_render_pass.
    unsafe {
        end_render_pass_common(command_buffer, move || {
            if let Some(next) = next {
                next(command_buffer);
            }
        });
    }
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
/// Standard `vkCreateRenderPass2{,KHR}` contract. `VkAttachmentDescription2`
/// carries the same `format`/`finalLayout` fields the classic path reads,
/// just behind the `sType`/`pNext` header every `RenderPass2` struct has.
unsafe extern "system" fn ferridian_create_render_pass2(
    device: vk::Device,
    p_create_info: *const vk::RenderPassCreateInfo2<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_render_pass: *mut vk::RenderPass,
) -> vk::Result {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    let Some(Some(next)) = dispatch::with_device(key, |state| state.create_render_pass2) else {
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
/// Standard `vkCreateImage` contract.
unsafe extern "system" fn ferridian_create_image(
    device: vk::Device,
    p_create_info: *const vk::ImageCreateInfo<'_>,
    p_allocator: *const vk::AllocationCallbacks<'_>,
    p_image: *mut vk::Image,
) -> vk::Result {
    // SAFETY: a non-null device passed to a device command is live.
    let key = unsafe { dispatch_key(device.as_raw()) };
    let Some(Some(next)) = dispatch::with_device(key, |state| state.create_image) else {
        return vk::Result::ERROR_INITIALIZATION_FAILED;
    };
    if p_create_info.is_null() {
        // Nothing to inspect; forward untouched.
        // SAFETY: forwarding the caller's (still valid) arguments down the chain.
        return unsafe { next(device, p_create_info, p_allocator, p_image) };
    }
    // SAFETY: a non-null create info is live for the duration of the call.
    let info = unsafe { &*p_create_info };
    let wants_attachment = info.usage.contains(vk::ImageUsageFlags::COLOR_ATTACHMENT)
        || info
            .usage
            .contains(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT);
    let already_transferable = info.usage.contains(vk::ImageUsageFlags::TRANSFER_SRC);
    let patched_usage = info.usage | vk::ImageUsageFlags::TRANSFER_SRC;
    // A real game's attachments (colour targets, depth buffers) rarely carry
    // TRANSFER_SRC themselves — the compositor needs it to copy them out
    // mid-frame. Adding the bit is only safe if the exact resulting image
    // config is still one `vkCreateImage` would accept — checked via the
    // same query the spec provides for exactly this question, not a
    // guessed-at format-feature bit.
    let should_patch = wants_attachment
        && !already_transferable
        && dispatch::with_device(key, |state| {
            // SAFETY: `instance`/`physical_device` are valid for the whole
            // life of the device, which outlives this call.
            unsafe {
                state.instance.get_physical_device_image_format_properties(
                    state.physical_device,
                    info.format,
                    info.image_type,
                    info.tiling,
                    patched_usage,
                    info.flags,
                )
            }
            .is_ok()
        })
        .unwrap_or(false);
    if !should_patch {
        // SAFETY: forwarding the caller's (still valid) arguments down the chain.
        return unsafe { next(device, p_create_info, p_allocator, p_image) };
    }
    let mut patched = *info;
    patched.usage = patched_usage;
    // SAFETY: `patched` is a bitwise copy of `*p_create_info` (a `Copy` type
    // with no owned data — only scalars and borrowed pointers) with only
    // `usage` changed, and outlives this call.
    unsafe { next(device, &patched, p_allocator, p_image) }
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
    // Extension-promoted-to-core commands: try the core name first, fall
    // back to the KHR name — both resolve to the identical down-chain
    // command per the Vulkan spec's promotion rules, so which name the app
    // itself queried us with doesn't need to match which one we ask for.
    let resolve_first = |core: &CStr, khr: &CStr| resolve(core).or_else(|| resolve(khr));
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
            instance: instance_table.clone(),
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
            create_image: retype!(resolve(c"vkCreateImage"), PfnCreateImage),
            create_render_pass2: retype!(
                resolve_first(c"vkCreateRenderPass2", c"vkCreateRenderPass2KHR"),
                PfnCreateRenderPass2
            ),
            cmd_begin_render_pass2: retype!(
                resolve_first(c"vkCmdBeginRenderPass2", c"vkCmdBeginRenderPass2KHR"),
                PfnCmdBeginRenderPass2
            ),
            cmd_end_render_pass2: retype!(
                resolve_first(c"vkCmdEndRenderPass2", c"vkCmdEndRenderPass2KHR"),
                PfnCmdEndRenderPass2
            ),
            cmd_begin_rendering: retype!(
                resolve_first(c"vkCmdBeginRendering", c"vkCmdBeginRenderingKHR"),
                PfnCmdBeginRendering
            ),
            cmd_end_rendering: retype!(
                resolve_first(c"vkCmdEndRendering", c"vkCmdEndRenderingKHR"),
                PfnCmdEndRendering
            ),
            overlay: Mutex::new(Overlay::new(device_table.clone())),
            taps: Mutex::new(TapRegistry::default()),
            compositor: Mutex::new(compositor),
            pack_watcher: Mutex::new(pack_watcher),
            device: device_table,
            queue,
            queue_family_index,
            allocator,
            instance: instance_table,
            physical_device,
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
    let Some(mut state) = dispatch::remove_device(key) else {
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
    // `.take()`, not `.as_mut()`: this must drop the compositor itself (not
    // just tear down its GPU objects) *before* the real device is destroyed
    // below. The compositor holds its own `Arc<Mutex<Allocator>>` clone
    // (alongside `state.allocator`'s), and gpu-allocator's own `Drop` impl
    // frees any block it still owns through its internally-cloned
    // `ash::Device` — running that after `destroy_device` is a
    // use-after-free (this crashed real runs before this fix).
    if let Some(mut compositor) = state
        .compositor
        .lock()
        .expect("compositor lock poisoned")
        .take()
    {
        // SAFETY: the app must have idled the device per vkDestroyDevice's
        // rules, so nothing in flight references the compositor's objects.
        unsafe { compositor.destroy() };
        drop(compositor);
    }
    // Same reasoning: this is the layer's own allocator (distinct from
    // `PackExecutor`/`PackCompositor`'s clone, just freed above) — drop it
    // before the device goes away too.
    drop(state.allocator.take());
    hooks::device_destroyed();
    // SAFETY: forwarding the destroy down the chain exactly once.
    unsafe { (state.destroy_device)(device, p_allocator) };
}
