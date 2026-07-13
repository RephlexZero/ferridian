//! The JSON manifest is what the Vulkan loader actually reads; keep it in
//! lockstep with the code's exported names.

const MANIFEST: &str = include_str!("../manifest/VkLayer_FERRIDIAN_overlay.json");

#[test]
fn manifest_matches_exports() {
    let manifest: serde_json::Value = serde_json::from_str(MANIFEST).expect("manifest is JSON");
    let layer = &manifest["layer"];
    assert_eq!(
        layer["name"].as_str().unwrap(),
        ferridian_vk_layer::LAYER_NAME.to_str().unwrap()
    );
    assert_eq!(
        layer["functions"]["vkNegotiateLoaderLayerInterfaceVersion"]
            .as_str()
            .unwrap(),
        "vkNegotiateLoaderLayerInterfaceVersion"
    );
    // Khronos naming rule: VK_LAYER_<VENDOR>_<name>.
    assert!(layer["name"].as_str().unwrap().starts_with("VK_LAYER_"));
}
