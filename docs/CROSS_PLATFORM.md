# Cross-Platform Validation

## Platform Status

| Platform                | Build | Test | Window Creation | Notes |
|-------------------------|-------|------|-----------------|-------|
| Linux x86_64            | ✅    | ✅   | ✅ (standalone) | Primary dev environment |
| macOS aarch64 (Apple)   | 🔧    | 🔧   | 🔧              | Metal backend via wgpu |
| macOS x86_64 (Intel)    | 🔧    | 🔧   | 🔧              | Metal backend via wgpu |
| Windows x86_64          | 🔧    | 🔧   | 🔧              | DX12/Vulkan via wgpu |

Legend: ✅ Verified | 🔧 Expected to work, not yet tested on actual hardware

## Backend Coverage

Ferridian uses `wgpu` which abstracts over platform backends:

- **Vulkan**: Linux, Windows (primary backend)
- **Metal**: macOS, iOS
- **DX12**: Windows (fallback)
- **WebGPU**: Browser (future, not a current target)

The `BackendConfig` provides:
- `adapter_options()` — standard high-performance adapter
- `fallback_adapter_options()` — software/fallback adapter for CI
- `headless_adapter_options()` — no surface, for testing

## Cross-Compilation

Use `cargo run -p xtask -- cross-build` to target other platforms:

```sh
# Linux to macOS
cargo run -p xtask -- cross-build --target aarch64-apple-darwin

# Linux to Windows
cargo run -p xtask -- cross-build --target x86_64-pc-windows-gnu
```

Note: cross-compilation may require platform-specific linkers and SDKs.

## JNI Platform Libraries

Use `cargo run -p xtask -- package-jni` to package the native JNI library
for the current platform. The output is placed in `java/build/resources/main/natives/`
with a platform tag (e.g., `linux-x86_64-libferridian_jni.so`).

## Portable Code Practices

All platform-specific code is isolated:
- SIMD: `CpuFeatures::detect()` with `#[cfg(target_arch)]` guards
- JNI: `ferridian-jni-bridge` crate, `cdylib` target
- Window creation: `winit` (cross-platform), only in standalone example
- GPU: `wgpu` handles all backend selection
