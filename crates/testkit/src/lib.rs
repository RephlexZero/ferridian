//! GPU test harness.
//!
//! `TestGpu` boots a software Vulkan device (lavapipe in the CI container)
//! with the Khronos validation layer enabled, and treats **any validation
//! message as a test failure** — the M0 safety-net guarantee everything else
//! builds on.
//!
//! GPU tests are gated on `FERRIDIAN_GPU_TESTS=1` (set by `mise run gpu-test`
//! and the container CI job) so laptop/CI legs without a Vulkan ICD stay green.

mod golden;
mod image;
mod render;
mod texture;

pub use golden::{
    assert_matches_golden, assert_matches_golden_with, bless_enabled, failures_dir, goldens_dir,
};
pub use image::{
    DiffPolicy, ImageDiff, ImageError, RgbaImage, diff_heatmap, diff_images, diff_passes,
};
pub use render::{RenderSpec, ShaderSpec, render_offscreen};
pub use texture::{GpuImage, create_target, read_back, upload_texture};

use ferridian_vk_rt::{RuntimeOptions, VkRuntime};

/// True when GPU-gated tests should actually run.
pub fn gpu_tests_enabled() -> bool {
    std::env::var_os("FERRIDIAN_GPU_TESTS").is_some_and(|v| v == "1")
}

/// Skip the current test (by returning early) unless GPU tests are enabled.
#[macro_export]
macro_rules! require_gpu {
    () => {
        if !$crate::gpu_tests_enabled() {
            eprintln!("skipped: set FERRIDIAN_GPU_TESTS=1 to run GPU tests");
            return;
        }
    };
}

pub struct TestGpu {
    runtime: VkRuntime,
    checked: bool,
}

impl TestGpu {
    /// Boot a validation-enabled software device. Panics on failure — in the
    /// test container both lavapipe and VVL are guaranteed present, so failure
    /// is a broken environment, not a condition to handle.
    pub fn new() -> TestGpu {
        let runtime = VkRuntime::new(&RuntimeOptions {
            app_name: "ferridian-testkit".to_owned(),
            enable_validation: true,
            prefer_software_device: true,
            extra_layers: Vec::new(),
        })
        .expect("testkit requires a Vulkan ICD (lavapipe) and VK_LAYER_KHRONOS_validation");
        TestGpu {
            runtime,
            checked: false,
        }
    }

    pub fn runtime(&self) -> &VkRuntime {
        &self.runtime
    }

    pub fn device(&self) -> &ash::Device {
        self.runtime.device()
    }

    /// Validation messages collected so far. Use this in tests that
    /// *deliberately* provoke validation errors.
    pub fn take_validation_messages(&mut self) -> Vec<String> {
        self.checked = true;
        self.runtime.validation_messages()
    }

    /// Assert the validation layer stayed silent. Called implicitly on drop,
    /// so simply letting a `TestGpu` fall out of scope enforces the guarantee.
    pub fn assert_no_validation_errors(&mut self) {
        self.checked = true;
        let messages = self.runtime.validation_messages();
        assert!(
            messages.is_empty(),
            "validation layer reported {} message(s):\n{}",
            messages.len(),
            messages.join("\n")
        );
    }
}

impl Default for TestGpu {
    fn default() -> Self {
        TestGpu::new()
    }
}

impl Drop for TestGpu {
    fn drop(&mut self) {
        // Don't double-panic while unwinding, and don't re-check a TestGpu the
        // test already inspected (e.g. deliberate-error tests).
        if !self.checked && !std::thread::panicking() {
            self.assert_no_validation_errors();
        }
    }
}
