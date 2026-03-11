# Standalone Examples

This directory contains renderer slices that can run without the Minecraft adapter while the core pipeline is still under construction.

## First slice

The initial standalone binary opens a native window, initializes wgpu through `ferridian-core`, loads a WGSL shader from disk, and renders a meshed chunk section with culled interior faces, a compact packed-vertex terrain format, a moving camera, and depth buffering.

During local development, editing [shaders/wgsl/standalone.wgsl](/workspaces/ferridian/shaders/wgsl/standalone.wgsl) now triggers a live pipeline rebuild on the next frame, so shader iteration does not require restarting the standalone app.

```bash
cargo run -p ferridian-standalone
```