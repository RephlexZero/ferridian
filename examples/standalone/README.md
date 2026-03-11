# Standalone Examples

This directory contains renderer slices that can run without the Minecraft adapter while the core pipeline is still under construction.

## First slice

The initial standalone binary opens a native window, initializes wgpu through `ferridian-core`, loads a WGSL shader from disk, and renders a small indexed voxel chunk with a moving camera and depth buffering.

```bash
cargo run -p ferridian-standalone
```