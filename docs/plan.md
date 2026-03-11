# Ferridian Plan

This plan is intentionally execution-oriented. Checked items exist in the repository today. Unchecked items are still planned work.

As much as possible, all work should be test driven.

## Project framing

- [x] Commit to a Rust-first renderer architecture built around `wgpu`
- [x] Commit to a JNI bridge plus a Java integration shell rather than Java-side GPU ownership
- [x] Keep the long-form rationale and ecosystem notes in repo docs under `docs/`
- [ ] Revisit the plan once Mojang or Aperture publishes a stable Vulkan-facing mod/shader interface

## Repository and workflow foundation

- [x] Create a Cargo workspace with dedicated crates for `core`, `shader`, `minecraft`, `jni-bridge`, `utils`, and `xtask`
- [x] Add a Java shell project under `java/`
- [x] Add shader staging directories under `shaders/wgsl` and `shaders/glsl`
- [x] Move narrative repository docs under `docs/`
- [x] Add a devcontainer for Rust plus Java development
- [x] Add CI that runs the shared workspace automation command
- [x] Add versioned git hooks that use the same automation path as CI
- [x] Centralize shared Rust dependencies in `[workspace.dependencies]`
- [ ] Add dual-license files to the repository root
- [ ] Add release packaging and native artifact publishing flow

## Current runnable slice

- [x] Create a reusable Rust-side surface bootstrap in `ferridian-core`
- [x] Add a standalone runnable example outside Minecraft integration
- [x] Render an animated WGSL triangle through the standalone example
- [x] Keep example bootstrapping thin and push renderer ownership into `ferridian-core`
- [ ] Add a camera abstraction usable by both standalone and future Minecraft entry points
- [ ] Replace the triangle slice with reusable mesh and material primitives
- [ ] Add depth buffering and basic transform handling

## Shader pipeline foundation

- [x] Establish WGSL as the first live shader path
- [ ] Add shader module loading from disk instead of `include_str!`
- [ ] Add hot reload support for local standalone development
- [ ] Add shader reflection or manifest metadata for pipeline setup
- [ ] Introduce GLSL ingestion for compatibility and migration tooling
- [ ] Evaluate `naga`, `shaderc`, and `rspirv` responsibilities in code rather than only in docs

## Java and JNI integration

- [x] Create a Java bootstrap shell with tests
- [x] Create a JNI bridge crate stub that compiles cleanly as a `cdylib`
- [ ] Add native library loading from the Java side
- [ ] Expose a minimal end-to-end JNI ping from Java into Rust
- [ ] Define the first stable bridge API for renderer init, resize, and frame execution
- [ ] Move bulk data exchange to direct buffers rather than per-call copies
- [ ] Unify the standalone renderer entry path and JNI renderer entry path

## Minecraft adapter layer

- [x] Reserve a dedicated `ferridian-minecraft` crate for game-facing adaptation
- [ ] Define chunk, camera, and frame input types for Minecraft-facing data
- [ ] Add a Fabric-side loader or hook shell in the Java project
- [ ] Connect the Java side to real Minecraft lifecycle or render hooks
- [ ] Feed Minecraft scene state into Rust without direct renderer ownership on the Java side
- [ ] Prototype chunk upload and per-frame camera updates over JNI

## Renderer milestones

### Phase 1: Immediate next renderer steps

- [x] Clear the swapchain successfully
- [x] Render a simple animated triangle
- [ ] Render indexed geometry with a reusable vertex format
- [ ] Add a camera uniform and basic view/projection control
- [ ] Load one external shader asset through the shader crate
- [ ] Render a simple voxel chunk or cube grid in the standalone example

### Phase 2: Forward rendering baseline

- [ ] Add texture loading and sampler management
- [ ] Add a depth buffer and deterministic resize handling for all attachments
- [ ] Add simple directional lighting
- [ ] Add material definitions that can evolve toward PBR
- [ ] Add a basic render graph or pass scheduler in `ferridian-core`

### Phase 3: Deferred renderer foundation

- [ ] Introduce a G-buffer pass layout
- [ ] Store albedo, normal, and material data in explicit render targets
- [ ] Reconstruct positions from depth in a deferred pass
- [ ] Add a first deferred lighting pass
- [ ] Add translucent pass separation

### Phase 4: Feature parity targets

- [ ] Shadow map pass
- [ ] Shadow filtering with PCF or PCSS
- [ ] SSAO or GTAO
- [ ] Screen-space reflections
- [ ] Volumetric lighting
- [ ] Bloom
- [ ] TAA
- [ ] Tonemapping and color grading
- [ ] Water shading path
- [ ] LabPBR-compatible material interpretation

## Performance roadmap

- [x] Keep the workspace split in a way that allows hot-path experimentation without destabilizing every crate
- [ ] Add criterion or scenario benchmarks for renderer and upload hot paths
- [ ] Track frame time, CPU time, and upload bandwidth regressions in CI or scheduled runs
- [ ] Add an opt-in PGO release flow
- [ ] Introduce runtime CPU feature detection for hot paths
- [ ] Evaluate portable SIMD or other nightly-only experiments in isolated modules or crates
- [ ] Push data-oriented layouts through scene, upload, and pass preparation code
- [ ] Make zero-copy JNI boundaries a hard requirement for chunk and uniform uploads

## Packaging and portability

- [x] Keep the renderer backend Vulkan-first while still using `wgpu` portability
- [ ] Validate the standalone example on Linux, macOS, and Windows with actual window creation
- [ ] Add fallback-adapter or software-renderer coverage where practical
- [ ] Package platform-specific JNI libraries for Java consumption
- [ ] Add cross-compilation or cross-platform release automation for native artifacts

## Learning and research backlog

- [ ] Study `wgpu-mc` for JNI and Minecraft renderer ownership patterns
- [ ] Study Bevy render architecture for render graph and pipeline caching ideas
- [ ] Study Veloren for voxel renderer production patterns
- [ ] Review Complementary or BSL-derived shader structure for compatibility targets
- [ ] Review Mojang snapshot and modding API changes as the Vulkan transition becomes public

## Definition of “first meaningful milestone”

- [x] The repo scaffolds cleanly and validates through shared automation
- [x] There is a standalone runnable renderer slice outside Minecraft
- [ ] The standalone slice renders chunk-like geometry rather than only a demo primitive
- [ ] Java can successfully call into Rust over JNI
- [ ] A single renderer path can be driven either from standalone or from the Java shell

## Definition of “feature complete enough to chase Minecraft integration”

- [ ] Camera, mesh, shader, and texture primitives are reusable from `ferridian-core`
- [ ] The shader crate owns real loading/translation responsibilities
- [ ] The JNI boundary is real, not stubbed
- [ ] The Java shell can load and call the native library
- [ ] The standalone path can render a simple voxel scene with depth and basic lighting

## Definition of “feature complete enough to chase shader-pack parity”

- [ ] Deferred rendering path exists
- [ ] Shadowing exists
- [ ] Material system can express LabPBR-like data
- [ ] Post-processing stack exists
- [ ] Performance measurement exists so visual additions do not happen blind