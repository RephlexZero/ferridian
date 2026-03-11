# Ferridian

Ferridian is a scaffold for a Rust-first Minecraft shader engine built around a wgpu renderer, a JNI bridge, and a Java integration shell that can evolve alongside Minecraft Java Edition's Vulkan transition.

The repository is dual-licensed under MIT or Apache-2.0.

Core repository docs live under `docs/`:

- `docs/ARCHITECTURE.md`
- `docs/CONTRIBUTING.md`
- `docs/plan.md`
- `docs/comment.md`

## Current shape

- `crates/core`: renderer-facing primitives and backend planning
- `crates/shader`: shader pipeline configuration and future compilation tooling
- `crates/minecraft`: Minecraft-facing adapter layer
- `crates/jni-bridge`: native library surface for Java interop
- `crates/utils`: shared workspace types and error definitions
- `crates/xtask`: repository automation used by CI and git hooks
- `java`: Gradle-based Java shell for the future Fabric-side bridge
- `shaders`: WGSL and GLSL staging areas
- `examples/standalone`: renderer slices that can run before Minecraft integration

## Local workflow

Use the devcontainer for a consistent Rust + Java toolchain.

```bash
cargo run -p xtask -- hooks
cargo run -p xtask -- ci
cargo run -p ferridian-standalone
```

The pre-commit hook runs the same `xtask ci` command as GitHub Actions so local failures match CI failures as closely as possible.

The standalone example is the first runnable renderer slice: it opens a desktop window and exercises the current Rust-side wgpu bootstrap with a camera-driven indexed voxel chunk before depending on Minecraft integration.
