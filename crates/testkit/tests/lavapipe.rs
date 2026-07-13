//! M0 exit criteria: the harness boots lavapipe, and a validation error makes
//! a test fail — proven here by provoking one on purpose and asserting it was
//! captured.

use ash::vk;
use ferridian_testkit::{TestGpu, require_gpu};

#[test]
fn boots_software_device_cleanly() {
    require_gpu!();
    let mut gpu = TestGpu::new();

    let fence_info = vk::FenceCreateInfo::default();
    // SAFETY: valid create info; fence is destroyed before the device.
    unsafe {
        let fence = gpu
            .device()
            .create_fence(&fence_info, None)
            .expect("fence creation on a fresh device");
        gpu.device().destroy_fence(fence, None);
    }

    gpu.assert_no_validation_errors();
}

#[test]
fn validation_errors_are_caught() {
    require_gpu!();
    let mut gpu = TestGpu::new();

    // Deliberately invalid: VkBufferCreateInfo with size 0 and no usage flags
    // violates VUID-VkBufferCreateInfo-size-00912 (and friends). The call may
    // or may not "succeed" — validation reports either way.
    let bad_buffer_info = vk::BufferCreateInfo::default();
    // SAFETY: intentionally invalid parameters; validation layers are loaded
    // and this is exactly what they exist to catch.
    unsafe {
        if let Ok(buffer) = gpu.device().create_buffer(&bad_buffer_info, None) {
            gpu.device().destroy_buffer(buffer, None);
        }
    }

    let messages = gpu.take_validation_messages();
    assert!(
        !messages.is_empty(),
        "the validation layer should have flagged a zero-size, zero-usage buffer — \
         the safety net is not armed"
    );
}
