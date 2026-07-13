//! The Vulkan layer cdylib: Ferridian's primary interception seam.
//!
//! This crate stays *boring* by design. It implements the Khronos
//! loader↔layer contract (negotiation, proc-addr plumbing, create-info chain
//! advancement) and nothing else; all engine logic lives behind the [`hooks`]
//! seam so tests and Miri reach maximal code in `ferridian-engine`.
//!
//! Current state: a correct pass-through layer. It chains `vkCreateInstance`
//! and `vkCreateDevice` down the layer stack and forwards every other call
//! unmodified. Actual frame interception is M2.

mod hooks;
mod loader_interface;

pub use loader_interface::LAYER_NAME;

use std::ffi::{CStr, c_char};
use std::sync::Mutex;

use ash::vk;

use loader_interface::{
    LayerFunction, NEGOTIATE_INTERFACE_STRUCT, PfnCreateDevice, PfnCreateInstance,
    VkLayerDeviceCreateInfo, VkLayerInstanceCreateInfo, VkNegotiateLayerInterface,
};

/// The next-layer proc-addr entry points, captured during create calls.
/// A game process creates one instance/device, so single slots suffice for
/// the skeleton; per-object dispatch tables arrive with real interception.
static NEXT_GIPA: Mutex<Option<vk::PFN_vkGetInstanceProcAddr>> = Mutex::new(None);
static NEXT_GDPA: Mutex<Option<vk::PFN_vkGetDeviceProcAddr>> = Mutex::new(None);
static INSTANCE: Mutex<vk::Instance> = Mutex::new(vk::Instance::null());

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
        b"vkGetInstanceProcAddr" => {
            // SAFETY: transmuting a fn pointer to the erased PFN_vkVoidFunction.
            Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetInstanceProcAddr, unsafe extern "system" fn()>(
                    ferridian_get_instance_proc_addr,
                )
            })
        }
        b"vkGetDeviceProcAddr" => {
            // SAFETY: as above.
            Some(unsafe {
                std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(
                    ferridian_get_device_proc_addr,
                )
            })
        }
        b"vkCreateInstance" => {
            // SAFETY: as above.
            Some(unsafe {
                std::mem::transmute::<PfnCreateInstance, unsafe extern "system" fn()>(
                    ferridian_create_instance,
                )
            })
        }
        b"vkCreateDevice" => {
            // SAFETY: as above.
            Some(unsafe {
                std::mem::transmute::<PfnCreateDevice, unsafe extern "system" fn()>(
                    ferridian_create_device,
                )
            })
        }
        _ => {
            let next = *NEXT_GIPA.lock().expect("layer state lock poisoned");
            // SAFETY: forwarding the unmodified query down the chain.
            next.and_then(|gipa| unsafe { gipa(instance, p_name) })
        }
    }
}

/// # Safety
/// Standard `vkGetDeviceProcAddr` contract.
pub unsafe extern "system" fn ferridian_get_device_proc_addr(
    device: vk::Device,
    p_name: *const c_char,
) -> vk::PFN_vkVoidFunction {
    if p_name.is_null() {
        return None;
    }
    // SAFETY: caller guarantees a NUL-terminated string per the Vulkan spec.
    let name = unsafe { CStr::from_ptr(p_name) };
    if name.to_bytes() == b"vkGetDeviceProcAddr" {
        // SAFETY: transmuting a fn pointer to the erased PFN_vkVoidFunction.
        return Some(unsafe {
            std::mem::transmute::<vk::PFN_vkGetDeviceProcAddr, unsafe extern "system" fn()>(
                ferridian_get_device_proc_addr,
            )
        });
    }
    let next = *NEXT_GDPA.lock().expect("layer state lock poisoned");
    // SAFETY: forwarding the unmodified query down the chain.
    next.and_then(|gdpa| unsafe { gdpa(device, p_name) })
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
    if result == vk::Result::SUCCESS {
        *NEXT_GIPA.lock().expect("layer state lock poisoned") = Some(next_gipa);
        // SAFETY: on success the loader guarantees *p_instance is initialized.
        *INSTANCE.lock().expect("layer state lock poisoned") = unsafe { *p_instance };
        hooks::instance_created();
    }
    result
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
    let instance = *INSTANCE.lock().expect("layer state lock poisoned");
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
    if result == vk::Result::SUCCESS {
        *NEXT_GDPA.lock().expect("layer state lock poisoned") = Some(next_gdpa);
        hooks::device_created();
    }
    result
}
