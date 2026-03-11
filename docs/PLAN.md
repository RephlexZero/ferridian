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
- [x] Add dual-license files to the repository root
- [ ] Add release packaging and native artifact publishing flow

## Current runnable slice

- [x] Create a reusable Rust-side surface bootstrap in `ferridian-core`
- [x] Add a standalone runnable example outside Minecraft integration
- [x] Render an animated WGSL triangle through the standalone example
- [x] Keep example bootstrapping thin and push renderer ownership into `ferridian-core`
- [x] Add a camera abstraction usable by both standalone and future Minecraft entry points
- [x] Replace the triangle slice with reusable mesh and material primitives
- [x] Add depth buffering and basic transform handling

## Shader pipeline foundation

- [x] Establish WGSL as the first live shader path
- [x] Add shader module loading from disk instead of `include_str!`
- [ ] Add hot reload support for local standalone development
- [x] Add shader reflection or manifest metadata for pipeline setup
- [ ] Introduce GLSL ingestion for compatibility and migration tooling
- [ ] Evaluate `naga`, `shaderc`, and `rspirv` responsibilities in code rather than only in docs

## Java and JNI integration

- [x] Create a Java bootstrap shell with tests
- [x] Create a JNI bridge crate stub that compiles cleanly as a `cdylib`
- [x] Add native library loading from the Java side
- [x] Expose a minimal end-to-end JNI ping from Java into Rust
- [x] Define the first stable bridge API for renderer init, resize, and frame execution
- [ ] Move bulk data exchange to direct buffers rather than per-call copies
- [ ] Unify the standalone renderer entry path and JNI renderer entry path

## Minecraft adapter layer

- [x] Reserve a dedicated `ferridian-minecraft` crate for game-facing adaptation
- [x] Define chunk, camera, and frame input types for Minecraft-facing data
- [ ] Add a Fabric-side loader or hook shell in the Java project
- [ ] Connect the Java side to real Minecraft lifecycle or render hooks
- [ ] Feed Minecraft scene state into Rust without direct renderer ownership on the Java side
- [ ] Prototype chunk upload and per-frame camera updates over JNI

## Fabric, Iris, and Aperture integration track

- [ ] Build the first Minecraft runtime path on Fabric rather than a custom launcher path
- [ ] Define a renderer integration seam that can sit behind Iris today without making Iris-specific assumptions pervasive across the codebase
- [ ] Identify the exact Fabric lifecycle, resource reload, and world render hooks needed for first renderer bootstrap
- [ ] Add an Iris compatibility layer for shader-pack-era resource, camera, and frame timing inputs
- [ ] Keep shader-pack parsing, renderer ownership, and GPU scheduling on the Rust side even when Fabric or Iris provides the entry point
- [ ] Treat Iris integration as a transitional adapter and keep the Java bridge boundary narrow enough to swap to Aperture when its public pipeline stabilizes
- [ ] Introduce an internal `RenderPipelineProvider` or similarly named abstraction so `standalone`, `fabric-iris`, and future `aperture` entry paths can drive the same renderer core
- [ ] Capture differences between Fabric plus Iris expectations and anticipated Aperture expectations in a compatibility matrix under `docs/`
- [ ] Prototype one end-to-end path where Fabric boots the mod, Iris-facing hooks collect frame inputs, and Rust executes the renderer through the existing bridge API
- [ ] Revisit and simplify the adapter layer once Aperture publishes stable integration points

## Renderer milestones

### Phase 1: Immediate next renderer steps

- [x] Clear the swapchain successfully
- [x] Render a simple animated triangle
- [x] Render indexed geometry with a reusable vertex format
- [x] Add a camera uniform and basic view/projection control
- [x] Load one external shader asset through the shader crate
- [x] Render a simple voxel chunk or cube grid in the standalone example

### Phase 2: Forward rendering baseline

- [ ] Add texture loading and sampler management
- [x] Add a depth buffer and deterministic resize handling for all attachments
- [x] Add simple directional lighting
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
- [x] The standalone slice renders chunk-like geometry rather than only a demo primitive
- [x] Java can successfully call into Rust over JNI
- [ ] A single renderer path can be driven either from standalone or from the Java shell
- [ ] A Fabric-based bootstrap path can reach the shared renderer core without forking renderer ownership into Java

## Definition of “feature complete enough to chase Minecraft integration”

- [ ] Camera, mesh, shader, and texture primitives are reusable from `ferridian-core`
- [ ] The shader crate owns real loading/translation responsibilities
- [x] The JNI boundary is real, not stubbed
- [x] The Java shell can load and call the native library
- [x] The standalone path can render a simple voxel scene with depth and basic lighting
- [ ] A Fabric plus Iris adapter path exists and is isolated enough to be replaced by Aperture later

## Definition of “feature complete enough to chase shader-pack parity”

- [ ] Deferred rendering path exists
- [ ] Shadowing exists
- [ ] Material system can express LabPBR-like data
- [ ] Post-processing stack exists
- [ ] Performance measurement exists so visual additions do not happen blind