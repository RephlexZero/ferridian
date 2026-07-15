//! Vulkan runtime: instance/device bootstrap, queue selection, capability
//! tiers. `ash` lives here (and in `vk-layer`); everything above this crate
//! talks in terms of the runtime's types.

mod capability;
pub mod compositor;
pub mod exec;
pub mod tap;

pub use capability::{CapabilityTier, DeviceProfile, capability_tier};
pub use compositor::{CompositorError, PackCompositor};
pub use exec::{ExecContext, ExecError, OutputTarget, PackExecutor};
/// Re-exported so executor callers get the camera type from the same crate.
pub use ferridian_contract::CameraUniforms;
pub use tap::{ResolvedAttachment, ResolvedFrame, TapRegistry, ViewRecord};

use std::ffi::{CStr, CString, c_char, c_void};
use std::mem::ManuallyDrop;
use std::sync::{Arc, Mutex};

use ash::ext::debug_utils;
use ash::vk;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};

const VALIDATION_LAYER: &CStr = c"VK_LAYER_KHRONOS_validation";

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub app_name: String,
    /// Enable `VK_LAYER_KHRONOS_validation` and collect its messages.
    /// Fails hard if the layer is not installed — a silent fallback would
    /// void the guarantee the testkit exists to provide.
    pub enable_validation: bool,
    /// Prefer a CPU (software) device — lavapipe in the CI container.
    pub prefer_software_device: bool,
    /// Additional explicit layers to enable (e.g. the Ferridian layer under
    /// test, discovered via `VK_LAYER_PATH`). Each must be installed; a
    /// missing layer is an error, not a silent skip.
    pub extra_layers: Vec<String>,
}

impl Default for RuntimeOptions {
    fn default() -> Self {
        RuntimeOptions {
            app_name: "ferridian".to_owned(),
            enable_validation: false,
            prefer_software_device: false,
            extra_layers: Vec::new(),
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RuntimeError {
    #[error("failed to load the Vulkan loader: {0}")]
    Loading(#[from] ash::LoadingError),
    #[error("Vulkan call failed: {0}")]
    Vk(#[from] vk::Result),
    #[error("validation was requested but VK_LAYER_KHRONOS_validation is not installed")]
    ValidationUnavailable,
    #[error("requested layer {0} is not installed (is VK_LAYER_PATH set?)")]
    LayerUnavailable(String),
    #[error("layer name {0:?} contains an interior NUL byte")]
    BadLayerName(String),
    #[error("no Vulkan physical device offers a graphics queue")]
    NoSuitableDevice,
    #[error("app name contains an interior NUL byte")]
    BadAppName,
    #[error(transparent)]
    Allocator(#[from] gpu_allocator::AllocationError),
}

struct DebugMessenger {
    loader: debug_utils::Instance,
    messenger: vk::DebugUtilsMessengerEXT,
}

/// An initialized Vulkan instance + logical device with one graphics queue.
pub struct VkRuntime {
    _entry: ash::Entry,
    instance: ash::Instance,
    debug: Option<DebugMessenger>,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    queue: vk::Queue,
    queue_family_index: u32,
    device_name: String,
    validation_messages: Arc<Mutex<Vec<String>>>,
    /// `ManuallyDrop` so [`Drop for VkRuntime`] can free it before
    /// `destroy_device` — gpu-allocator's own `Drop` frees any remaining
    /// blocks via `vkFreeMemory`, which is invalid on an already-destroyed
    /// device.
    allocator: ManuallyDrop<Arc<Mutex<Allocator>>>,
}

impl VkRuntime {
    pub fn new(options: &RuntimeOptions) -> Result<VkRuntime, RuntimeError> {
        // SAFETY: loading the system Vulkan loader; no Vulkan objects exist yet.
        let entry = unsafe { ash::Entry::load()? };

        let app_name =
            CString::new(options.app_name.clone()).map_err(|_| RuntimeError::BadAppName)?;
        let app_info = vk::ApplicationInfo::default()
            .application_name(&app_name)
            .engine_name(c"ferridian")
            .api_version(vk::API_VERSION_1_3);

        let mut layers: Vec<*const c_char> = Vec::new();
        let mut extensions: Vec<*const c_char> = Vec::new();
        let installed = installed_layers(&entry)?;
        // `ppEnabledLayerNames` order is the call-chain order: the first
        // layer sits closest to the application, the last closest to the
        // driver. `extra_layers` (the Ferridian layer, in every test that
        // enables it) goes *first* so its parameter rewrites (e.g.
        // `vkCreateImage` forcing `TRANSFER_SRC` onto attachment-usage
        // images) are what validation itself sees and checks against —
        // listed after, validation would instead record the *app's*
        // original, unpatched parameters and never observe the fix.
        let extra_layers: Vec<CString> = options
            .extra_layers
            .iter()
            .map(|name| {
                if !installed
                    .iter()
                    .any(|installed_name| installed_name == name)
                {
                    return Err(RuntimeError::LayerUnavailable(name.clone()));
                }
                CString::new(name.clone()).map_err(|_| RuntimeError::BadLayerName(name.clone()))
            })
            .collect::<Result<_, _>>()?;
        layers.extend(extra_layers.iter().map(|name| name.as_ptr()));
        if options.enable_validation {
            if !installed
                .iter()
                .any(|name| name == "VK_LAYER_KHRONOS_validation")
            {
                return Err(RuntimeError::ValidationUnavailable);
            }
            layers.push(VALIDATION_LAYER.as_ptr());
            extensions.push(debug_utils::NAME.as_ptr());
        }

        let validation_messages: Arc<Mutex<Vec<String>>> = Arc::default();

        let mut debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
            .message_severity(
                vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                    | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            )
            .message_type(
                vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                    | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                    | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            )
            .pfn_user_callback(Some(collect_validation_message))
            .user_data(Arc::as_ptr(&validation_messages) as *mut c_void);

        let mut instance_info = vk::InstanceCreateInfo::default()
            .application_info(&app_info)
            .enabled_layer_names(&layers)
            .enabled_extension_names(&extensions);
        if options.enable_validation {
            instance_info = instance_info.push_next(&mut debug_info);
        }

        // SAFETY: create infos and the strings/slices they borrow outlive the call.
        let instance = unsafe { entry.create_instance(&instance_info, None)? };

        let debug = if options.enable_validation {
            let loader = debug_utils::Instance::new(&entry, &instance);
            // SAFETY: `debug_info` borrows `validation_messages`, which lives in
            // the returned VkRuntime; the messenger is destroyed in Drop before
            // the Arc is released.
            let messenger = unsafe {
                loader
                    .create_debug_utils_messenger(&debug_info, None)
                    .inspect_err(|_e| {
                        instance.destroy_instance(None);
                    })?
            };
            Some(DebugMessenger { loader, messenger })
        } else {
            None
        };

        let picked = pick_device(&instance, options.prefer_software_device);
        let (physical_device, queue_family_index, device_name) = match picked {
            Ok(found) => found,
            Err(error) => {
                // SAFETY: instance was created above and nothing else owns it yet.
                unsafe {
                    if let Some(debug) = &debug {
                        debug
                            .loader
                            .destroy_debug_utils_messenger(debug.messenger, None);
                    }
                    instance.destroy_instance(None);
                }
                return Err(error);
            }
        };

        let priorities = [1.0f32];
        let queue_infos = [vk::DeviceQueueCreateInfo::default()
            .queue_family_index(queue_family_index)
            .queue_priorities(&priorities)];
        // slangc-compiled shaders routinely declare the DrawParameters
        // capability (SV_VertexID and friends), which is only legal with the
        // 1.1-core shaderDrawParameters feature enabled — so enable it
        // wherever the device offers it. Dynamic rendering is a 1.3 feature
        // bit, not implied merely by requesting API 1.3 — the layer's
        // `vkCmdBeginRendering` interception needs it actually enabled on
        // devices this runtime creates (testkit's layered tests, in
        // particular) to exercise that path at all.
        let mut supported_11 = vk::PhysicalDeviceVulkan11Features::default();
        let mut supported_13 = vk::PhysicalDeviceVulkan13Features::default();
        let mut supported = vk::PhysicalDeviceFeatures2::default()
            .push_next(&mut supported_11)
            .push_next(&mut supported_13);
        // SAFETY: physical_device comes from this instance; the structs are live.
        unsafe { instance.get_physical_device_features2(physical_device, &mut supported) };
        let mut enabled_11 = vk::PhysicalDeviceVulkan11Features::default()
            .shader_draw_parameters(supported_11.shader_draw_parameters == vk::TRUE);
        let mut enabled_13 = vk::PhysicalDeviceVulkan13Features::default()
            .dynamic_rendering(supported_13.dynamic_rendering == vk::TRUE);
        let device_info = vk::DeviceCreateInfo::default()
            .queue_create_infos(&queue_infos)
            .push_next(&mut enabled_11)
            .push_next(&mut enabled_13);
        // SAFETY: physical_device comes from this instance; create info outlives the call.
        let device = unsafe { instance.create_device(physical_device, &device_info, None)? };
        // SAFETY: the queue family/index were validated during device creation.
        let queue = unsafe { device.get_device_queue(queue_family_index, 0) };

        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.clone(),
            device: device.clone(),
            physical_device,
            debug_settings: gpu_allocator::AllocatorDebugSettings::default(),
            buffer_device_address: false,
            allocation_sizes: gpu_allocator::AllocationSizes::default(),
        });
        let allocator = match allocator {
            Ok(allocator) => allocator,
            Err(error) => {
                // SAFETY: device/debug/instance were all created above and
                // nothing else owns them yet.
                unsafe {
                    device.destroy_device(None);
                    if let Some(debug) = &debug {
                        debug
                            .loader
                            .destroy_debug_utils_messenger(debug.messenger, None);
                    }
                    instance.destroy_instance(None);
                }
                return Err(error.into());
            }
        };

        tracing::info!(device = %device_name, software = options.prefer_software_device, "vulkan runtime ready");

        Ok(VkRuntime {
            _entry: entry,
            instance,
            debug,
            physical_device,
            device,
            queue,
            queue_family_index,
            device_name,
            validation_messages,
            allocator: ManuallyDrop::new(Arc::new(Mutex::new(allocator))),
        })
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn queue(&self) -> vk::Queue {
        self.queue
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    /// This runtime's device-side handles, in the form [`exec::PackExecutor`]
    /// consumes (inside the layer the same struct is built from the
    /// intercepted application's device instead).
    pub fn exec_context(&self) -> ExecContext<'_> {
        ExecContext {
            device: &self.device,
            queue: self.queue,
            queue_family_index: self.queue_family_index,
            allocator: (*self.allocator).clone(),
        }
    }

    /// Messages collected from the validation layer so far (errors + warnings).
    pub fn validation_messages(&self) -> Vec<String> {
        self.validation_messages
            .lock()
            .expect("validation message lock poisoned")
            .clone()
    }
}

impl Drop for VkRuntime {
    fn drop(&mut self) {
        // SAFETY: all handles were created by this runtime and are destroyed
        // exactly once, device before messenger before instance. Waiting
        // idle first means nothing submitted still references memory the
        // allocator is about to free.
        unsafe {
            let _ = self.device.device_wait_idle();
            // Dropped here (not left to the field's own destructor) so it
            // runs while `self.device` is still valid — gpu-allocator's Drop
            // calls vkFreeMemory on any remaining blocks.
            ManuallyDrop::drop(&mut self.allocator);
            self.device.destroy_device(None);
            if let Some(debug) = self.debug.take() {
                debug
                    .loader
                    .destroy_debug_utils_messenger(debug.messenger, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}

fn installed_layers(entry: &ash::Entry) -> Result<Vec<String>, RuntimeError> {
    // SAFETY: plain enumeration; no preconditions beyond a loaded entry.
    let layers = unsafe { entry.enumerate_instance_layer_properties()? };
    Ok(layers
        .iter()
        .filter_map(|layer| {
            layer
                .layer_name_as_c_str()
                .ok()
                .map(|name| name.to_string_lossy().into_owned())
        })
        .collect())
}

fn pick_device(
    instance: &ash::Instance,
    prefer_software: bool,
) -> Result<(vk::PhysicalDevice, u32, String), RuntimeError> {
    // SAFETY: instance is live for the duration of the call.
    let devices = unsafe { instance.enumerate_physical_devices()? };
    let mut best: Option<(u32, vk::PhysicalDevice, u32, String)> = None;
    for device in devices {
        // SAFETY: `device` was just enumerated from this instance.
        let properties = unsafe { instance.get_physical_device_properties(device) };
        // SAFETY: same as above.
        let queue_families =
            unsafe { instance.get_physical_device_queue_family_properties(device) };
        let Some(family_index) = queue_families
            .iter()
            .position(|family| family.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        else {
            continue;
        };
        let score = device_score(properties.device_type, prefer_software);
        let name = properties
            .device_name_as_c_str()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|_| "<unnamed device>".to_owned());
        if best.as_ref().is_none_or(|(top, ..)| score > *top) {
            best = Some((score, device, family_index as u32, name));
        }
    }
    best.map(|(_, device, family, name)| (device, family, name))
        .ok_or(RuntimeError::NoSuitableDevice)
}

fn device_score(device_type: vk::PhysicalDeviceType, prefer_software: bool) -> u32 {
    match (prefer_software, device_type) {
        (true, vk::PhysicalDeviceType::CPU) => 4,
        (true, vk::PhysicalDeviceType::VIRTUAL_GPU) => 3,
        (true, vk::PhysicalDeviceType::INTEGRATED_GPU) => 2,
        (true, vk::PhysicalDeviceType::DISCRETE_GPU) => 1,
        (false, vk::PhysicalDeviceType::DISCRETE_GPU) => 4,
        (false, vk::PhysicalDeviceType::INTEGRATED_GPU) => 3,
        (false, vk::PhysicalDeviceType::VIRTUAL_GPU) => 2,
        (false, vk::PhysicalDeviceType::CPU) => 1,
        _ => 0,
    }
}

unsafe extern "system" fn collect_validation_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    user_data: *mut c_void,
) -> vk::Bool32 {
    if data.is_null() || user_data.is_null() {
        return vk::FALSE;
    }
    // SAFETY: `data` is valid for the duration of the callback per the Vulkan
    // spec; `user_data` is the Mutex<Vec<String>> owned by the VkRuntime that
    // registered this messenger, which outlives the messenger.
    unsafe {
        let message = (*data)
            .message_as_c_str()
            .map(|m| m.to_string_lossy().into_owned())
            .unwrap_or_else(|| "<message missing>".to_owned());
        let sink = &*(user_data as *const Mutex<Vec<String>>);
        if let Ok(mut messages) = sink.lock() {
            messages.push(format!("[{severity:?}] {message}"));
        }
    }
    vk::FALSE
}
