//! FFI mirror of the parts of `vk_layer.h` that ash does not expose: the
//! loader↔layer negotiation struct and the create-info chain link types.
//!
//! Layouts follow the Khronos loader/layer interface spec
//! (<https://github.com/KhronosGroup/Vulkan-Loader/blob/main/docs/LoaderLayerInterface.md>).

use std::ffi::{CStr, c_void};

use ash::vk;

/// Our layer's canonical name; must match the JSON manifest.
pub const LAYER_NAME: &CStr = c"VK_LAYER_FERRIDIAN_overlay";

/// `LAYER_NEGOTIATE_INTERFACE_STRUCT` from `vk_layer.h`.
pub const NEGOTIATE_INTERFACE_STRUCT: i32 = 1;

/// `PFN_GetPhysicalDeviceProcAddr` — same shape as `vkGetInstanceProcAddr`.
pub type PfnGetPhysicalDeviceProcAddr = vk::PFN_vkGetInstanceProcAddr;

pub type PfnCreateInstance = for<'a, 'b> unsafe extern "system" fn(
    *const vk::InstanceCreateInfo<'a>,
    *const vk::AllocationCallbacks<'b>,
    *mut vk::Instance,
) -> vk::Result;

pub type PfnCreateDevice = for<'a, 'b> unsafe extern "system" fn(
    vk::PhysicalDevice,
    *const vk::DeviceCreateInfo<'a>,
    *const vk::AllocationCallbacks<'b>,
    *mut vk::Device,
) -> vk::Result;

/// `VkNegotiateLayerInterface` from `vk_layer.h`.
#[repr(C)]
pub struct VkNegotiateLayerInterface {
    pub s_type: i32,
    pub p_next: *mut c_void,
    pub loader_layer_interface_version: u32,
    pub pfn_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub pfn_get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
    pub pfn_get_physical_device_proc_addr: Option<PfnGetPhysicalDeviceProcAddr>,
}

/// `VkLayerFunction` from `vk_layer.h`. All variants mirror the C ABI even
/// though we only match on `LayerLinkInfo` today.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
pub enum LayerFunction {
    LayerLinkInfo = 0,
    LoaderDataCallback = 1,
    LoaderLayerCreateDeviceCallback = 2,
    LoaderFeatures = 3,
}

/// `VkLayerInstanceLink` from `vk_layer.h`.
#[repr(C)]
pub struct VkLayerInstanceLink {
    pub p_next: *mut VkLayerInstanceLink,
    pub pfn_next_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub pfn_next_get_physical_device_proc_addr: Option<PfnGetPhysicalDeviceProcAddr>,
}

/// `VkLayerDeviceLink` from `vk_layer.h`.
#[repr(C)]
pub struct VkLayerDeviceLink {
    pub p_next: *mut VkLayerDeviceLink,
    pub pfn_next_get_instance_proc_addr: Option<vk::PFN_vkGetInstanceProcAddr>,
    pub pfn_next_get_device_proc_addr: Option<vk::PFN_vkGetDeviceProcAddr>,
}

/// The `u` union of `VkLayerInstanceCreateInfo`. Only `p_layer_info` is
/// accessed; `_size_match` pads the union to the C size (its largest member
/// is a two-pointer struct used by `VK_LOADER_LAYER_CREATE_DEVICE_CALLBACK`).
#[repr(C)]
pub union LayerCreateInfoUnion<Link> {
    pub p_layer_info: *mut Link,
    pub _size_match: [*const c_void; 2],
}

/// `VkLayerInstanceCreateInfo` from `vk_layer.h`
/// (`sType == VK_STRUCTURE_TYPE_LOADER_INSTANCE_CREATE_INFO`).
#[repr(C)]
pub struct VkLayerInstanceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: LayerFunction,
    pub u: LayerCreateInfoUnion<VkLayerInstanceLink>,
}

/// `VkLayerDeviceCreateInfo` from `vk_layer.h`
/// (`sType == VK_STRUCTURE_TYPE_LOADER_DEVICE_CREATE_INFO`).
#[repr(C)]
pub struct VkLayerDeviceCreateInfo {
    pub s_type: vk::StructureType,
    pub p_next: *const c_void,
    pub function: LayerFunction,
    pub u: LayerCreateInfoUnion<VkLayerDeviceLink>,
}

/// Trait unifying the two chain-info shapes for [`find_layer_link`].
pub trait LayerCreateInfo {
    fn s_type(&self) -> vk::StructureType;
    fn p_next(&self) -> *const c_void;
    fn function(&self) -> LayerFunction;
}

impl LayerCreateInfo for VkLayerInstanceCreateInfo {
    fn s_type(&self) -> vk::StructureType {
        self.s_type
    }
    fn p_next(&self) -> *const c_void {
        self.p_next
    }
    fn function(&self) -> LayerFunction {
        self.function
    }
}

impl LayerCreateInfo for VkLayerDeviceCreateInfo {
    fn s_type(&self) -> vk::StructureType {
        self.s_type
    }
    fn p_next(&self) -> *const c_void {
        self.p_next
    }
    fn function(&self) -> LayerFunction {
        self.function
    }
}

/// Walk a create-info `pNext` chain looking for the loader's layer link
/// struct of the given type/function.
///
/// # Safety
/// `chain` must be the `pNext` pointer of a live `Vk*CreateInfo` whose chain
/// obeys the Vulkan struct-chaining rules (every node starts with
/// `sType`/`pNext`).
pub unsafe fn find_layer_link<T: LayerCreateInfo>(
    chain: *const c_void,
    wanted_s_type: vk::StructureType,
    wanted_function: LayerFunction,
) -> Option<*mut T> {
    let mut cursor = chain as *mut T;
    while !cursor.is_null() {
        // SAFETY: chain nodes are valid per this function's contract; we only
        // read the sType/pNext/function header fields.
        let node = unsafe { &*cursor };
        if node.s_type() == wanted_s_type && node.function() == wanted_function {
            return Some(cursor);
        }
        cursor = node.p_next() as *mut T;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn negotiate_struct_layout_matches_c() {
        // sType(i32, padded to 8) + pNext(8) + version(u32, padded to 8) + 3 fn ptrs.
        assert_eq!(
            size_of::<VkNegotiateLayerInterface>(),
            8 + 8 + 8 + 3 * size_of::<*const c_void>()
        );
    }

    #[test]
    fn union_is_two_pointers_wide() {
        assert_eq!(
            size_of::<LayerCreateInfoUnion<VkLayerInstanceLink>>(),
            2 * size_of::<*const c_void>()
        );
    }
}
