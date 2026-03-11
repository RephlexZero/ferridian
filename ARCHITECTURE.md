# Architecture

Ferridian is intentionally split into thin seams so the renderer, shader compiler, JNI layer, and Java-side integration can evolve independently.

## Rust workspace

- `ferridian-core` owns renderer-facing plans, data flow, and eventually the render graph.
- `ferridian-shader` will own shader translation, preprocessing, reflection, and hot reload.
- `ferridian-minecraft` will adapt Minecraft scene data into renderer-friendly structures.
- `ferridian-jni-bridge` is the shared library boundary for Java integration.
- `ferridian-utils` stays small and dependency-light so shared types do not drag the workspace.
- `xtask` centralizes repo automation so CI and local hooks stay aligned.

## Integration direction

The intended runtime stack is Java shell -> JNI bridge -> Rust renderer. The Java side remains intentionally small: window handles, world state extraction, and lifecycle hooks should live there, while GPU ownership stays on the Rust side.

Before the JNI path is real, runnable slices should land in `examples/standalone` and call into `ferridian-core` rather than duplicating bootstrapping logic in the example itself.

## Performance direction

The renderer should bias toward Vulkan-first wgpu execution, deferred shading, aggressive GPU-side batching, and CPU-side data structures that are prepared for SIMD, multiversioning, and profile-guided optimization.
