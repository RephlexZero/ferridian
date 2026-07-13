//! Capability tiers (§3.3): pure, data-driven checks so pack requirements can
//! be validated against committed device profiles (including a MoltenVK
//! profile) at pack build time — no live device needed.

use ash::vk;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapabilityTier {
    /// Vulkan 1.3 core, portability-subset clean. The floor every pack may assume.
    Baseline,
    /// Mesh shading and ray query available.
    Enhanced,
}

/// A device described as data. Real devices are queried into this; committed
/// profiles (e.g. `moltenvk.toml`, future work) deserialize into it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceProfile {
    pub name: String,
    pub api_version_major: u32,
    pub api_version_minor: u32,
    pub mesh_shading: bool,
    pub ray_query: bool,
    /// Device only advertises `VK_KHR_portability_subset` conformance.
    pub portability_subset: bool,
}

impl DeviceProfile {
    /// Query a live physical device into a profile.
    pub fn from_physical_device(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> Result<DeviceProfile, vk::Result> {
        // SAFETY: `device` must originate from `instance`, per this fn's contract.
        let properties = unsafe { instance.get_physical_device_properties(device) };
        // SAFETY: same as above.
        let extensions = unsafe { instance.enumerate_device_extension_properties(device)? };
        let has = |wanted: &std::ffi::CStr| {
            extensions.iter().any(|ext| {
                ext.extension_name_as_c_str()
                    .is_ok_and(|name| name == wanted)
            })
        };
        Ok(DeviceProfile {
            name: properties
                .device_name_as_c_str()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_else(|_| "<unnamed device>".to_owned()),
            api_version_major: vk::api_version_major(properties.api_version),
            api_version_minor: vk::api_version_minor(properties.api_version),
            mesh_shading: has(c"VK_EXT_mesh_shader"),
            ray_query: has(c"VK_KHR_ray_query"),
            portability_subset: has(c"VK_KHR_portability_subset"),
        })
    }

    pub fn meets_baseline(&self) -> bool {
        (self.api_version_major, self.api_version_minor) >= (1, 3)
    }
}

pub fn capability_tier(profile: &DeviceProfile) -> CapabilityTier {
    if profile.mesh_shading && profile.ray_query {
        CapabilityTier::Enhanced
    } else {
        CapabilityTier::Baseline
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> DeviceProfile {
        DeviceProfile {
            name: "test".to_owned(),
            api_version_major: 1,
            api_version_minor: 3,
            mesh_shading: false,
            ray_query: false,
            portability_subset: false,
        }
    }

    #[test]
    fn lavapipe_class_device_is_baseline() {
        assert_eq!(capability_tier(&profile()), CapabilityTier::Baseline);
    }

    #[test]
    fn mesh_and_ray_query_reach_enhanced() {
        let full = DeviceProfile {
            mesh_shading: true,
            ray_query: true,
            ..profile()
        };
        assert_eq!(capability_tier(&full), CapabilityTier::Enhanced);
    }

    #[test]
    fn mesh_shading_alone_stays_baseline() {
        let partial = DeviceProfile {
            mesh_shading: true,
            ..profile()
        };
        assert_eq!(capability_tier(&partial), CapabilityTier::Baseline);
    }

    #[test]
    fn moltenvk_style_profile_meets_baseline() {
        let moltenvk = DeviceProfile {
            portability_subset: true,
            ..profile()
        };
        assert!(moltenvk.meets_baseline());
        assert_eq!(capability_tier(&moltenvk), CapabilityTier::Baseline);
    }

    #[test]
    fn vulkan_1_2_fails_baseline() {
        let old = DeviceProfile {
            api_version_minor: 2,
            ..profile()
        };
        assert!(!old.meets_baseline());
    }
}
