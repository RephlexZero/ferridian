# Ferridian Plan

This plan is intentionally execution-oriented. Checked items exist in the repository today. Unchecked items are still planned work.

As much as possible, all work should be test driven.

## Architectural principles

- [x] Commit to a Rust-first renderer architecture built around `wgpu`
- [x] Commit to a JNI bridge plus a Java integration shell rather than Java-side GPU ownership
- [x] Keep the long-form rationale and ecosystem notes in repo docs under `docs/`
- [x] Treat the voxel world itself as the primary acceleration structure for lighting and visibility work rather than copying polygon-engine assumptions
- [x] Prefer techniques that scale specifically well for voxel terrain over fashionable AAA features that solve polygon-only problems
- [ ] Revisit the plan once Mojang or Aperture publishes a stable Vulkan-facing mod/shader interface

## Phase 0: Repository guardrails and release hygiene

- [x] Create a Cargo workspace with dedicated crates for `core`, `shader`, `minecraft`, `jni-bridge`, `utils`, and `xtask`
- [x] Add a Java shell project under `java/`
- [x] Add shader staging directories under `shaders/wgsl` and `shaders/glsl`
- [x] Move narrative repository docs under `docs/`
- [x] Add a devcontainer for Rust plus Java development
- [x] Add CI that runs the shared workspace automation command
- [x] Add versioned git hooks that use the same automation path as CI
- [x] Centralize shared Rust dependencies in `[workspace.dependencies]`
- [x] Add dual-license files to the repository root
- [x] Pin `wgpu` upgrades deliberately and isolate backend-facing code so the 12-week release cadence does not churn the whole codebase
- [x] Add a compatibility note describing which experimental `wgpu` features are allowed in mainline and which must stay behind opt-in flags
- [x] Add release packaging and native artifact publishing flow

## Phase 1: Shared renderer core and standalone slice

- [x] Create a reusable Rust-side surface bootstrap in `ferridian-core`
- [x] Add a standalone runnable example outside Minecraft integration
- [x] Render an animated WGSL triangle through the standalone example
- [x] Keep example bootstrapping thin and push renderer ownership into `ferridian-core`
- [x] Add a camera abstraction usable by both standalone and future Minecraft entry points
- [x] Replace the triangle slice with reusable mesh and material primitives
- [x] Add depth buffering and basic transform handling
- [x] Replace the current demo mesh path with chunk meshing that more closely matches real section data and update patterns
- [x] Pack terrain vertices aggressively so chunk geometry approaches Sodium-class compact formats rather than debug-friendly layouts
- [x] Terrain rendering is already using a chunk-oriented data path rather than a debug mesh path that will later be thrown away

## Phase 2: Shader authoring, loading, and compatibility foundations

- [x] Establish WGSL as the first live shader path
- [x] Add shader module loading from disk instead of `include_str!`
- [x] Add shader reflection or manifest metadata for pipeline setup
- [x] Add hot reload support for local standalone development
- [x] Adopt `naga_oil` or an equivalent modular WGSL layer for imports, defines, and packable shader composition
- [x] Add `wgsl_bindgen` or equivalent generated Rust bindings for shader layouts to catch binding drift at compile time
- [x] Evaluate `naga`, `shaderc`, and `rspirv` responsibilities in code rather than only in docs
- [x] Add a native shader permutation cache keyed by source, defines, vertex layout, and render state
- [x] Introduce GLSL ingestion for compatibility and migration tooling
- [x] Prototype an Iris or OptiFine compatibility adapter that maps shader-pack stages onto Ferridian render passes
- [ ] Track Slang-based Aperture authoring as a future shader-pack ingest path once the format stabilizes
- [x] The shader crate owns real loading/translation responsibilities
- [x] The shader compatibility layer can represent classic Iris pack stages while leaving room for Aperture's explicit pipeline model

## Phase 3: Terrain data path, materials, and renderer throughput

- [x] Add texture-array based material loading and sampler management for block-scale resources
- [x] Add material definitions that can evolve toward PBR and LabPBR resource-pack conventions
- [x] Move terrain rendering to GPU-resident chunk buffers with indirect draws and compute-driven visibility culling
- [x] Parallelize chunk meshing and upload preparation with Rayon or an equivalent stable CPU job system
- [x] Push data-oriented layouts through scene, upload, and pass preparation code
- [x] Move chunk visibility, draw compaction, and similar high-volume work to compute-driven GPU pipelines where `wgpu` already supports them cleanly
- [x] Avoid blocking on bindless or multi-queue async compute support and build around texture arrays plus single-queue scheduling for now
- [x] Camera, mesh, shader, and texture primitives are reusable from `ferridian-core`

## Phase 4: Deferred lighting backbone

- [x] Add simple directional lighting
- [x] Add a basic render graph or pass scheduler in `ferridian-core`
- [x] Introduce a G-buffer pass layout
- [x] Store albedo, normal, and material data in explicit render targets
- [x] Reconstruct positions from depth in a deferred pass
- [x] Add a first deferred lighting pass
- [x] Add translucent pass separation
- [x] Adopt a hybrid pipeline where opaque terrain is deferred and water, glass, particles, and foliage stay on a forward or forward-plus path
- [x] Encode block-material metadata densely enough that deferred lighting can support many emissive or local lights without exploding bandwidth
- [x] Deferred rendering path exists

## Phase 5: Core screen-space and atmospheric quality

- [x] Add XeGTAO-class ambient occlusion as the first high-value screen-space effect
- [x] Add visibility-bitmask GTAO or equivalent screen-space indirect diffuse once the G-buffer is stable
- [x] SSAO or GTAO
- [x] Screen-space reflections
- [x] Add stochastic SSR with a Hi-Z path suitable for water, metals, and wet surfaces
- [x] Volumetric lighting
- [x] Add froxel-based volumetric fog rather than only per-pixel ray-marched fog so volumetrics scale to larger scenes
- [x] Bloom
- [x] TAA
- [x] Add custom temporal upscaling or TAAU once motion vectors and jittered projection are available
- [x] Tonemapping and color grading
- [x] Water shading path
- [x] LabPBR-compatible material interpretation
- [x] Material system can express LabPBR-like data
- [x] Post-processing stack exists

## Phase 6: Voxel-native GI and high-end rendering modes

- [x] Prototype voxel cone tracing GI using the world voxel grid directly instead of paying a separate voxelization cost
- [x] Shadow map pass
- [x] Shadow filtering with PCF or PCSS
- [x] Shadowing exists
- [x] Treat path tracing and ReSTIR-style lighting as optional high-end modes, not as default rendering assumptions

## Phase 7: JNI data path and shared runtime entry

- [x] Create a Java bootstrap shell with tests
- [x] Create a JNI bridge crate stub that compiles cleanly as a `cdylib`
- [x] Add native library loading from the Java side
- [x] Expose a minimal end-to-end JNI ping from Java into Rust
- [x] Define the first stable bridge API for renderer init, resize, and frame execution
- [x] Move bulk data exchange to `DirectByteBuffer`-backed zero-copy paths rather than per-call copies
- [x] Cache `jclass`, `jmethodID`, and `jfieldID` lookups on initialization so the hot path avoids repeated JNI resolution
- [x] Permanently attach the dedicated render thread to the JVM and keep JNI crossings capped to coarse batched calls per frame
- [x] Make zero-copy JNI boundaries a hard requirement for chunk and uniform uploads
- [x] Define binary layouts for chunk sections, entities, lighting, weather, and resource-pack data that are stable across the JNI boundary
- [x] The renderer can ingest real chunk and material data through stable binary layouts and zero-copy buffer handoff
- [x] Unify the standalone renderer entry path and JNI renderer entry path
- [x] A single renderer path can be driven either from standalone or from the Java shell

## Phase 8: Minecraft adapter and Fabric integration

- [x] Reserve a dedicated `ferridian-minecraft` crate for game-facing adaptation
- [x] Define chunk, camera, and frame input types for Minecraft-facing data
- [x] Add a Fabric-side loader or hook shell in the Java project
- [x] Identify the exact Fabric lifecycle, resource reload, and world render hooks needed for first renderer bootstrap
- [x] Connect the Java side to real Minecraft lifecycle or render hooks
- [x] Snapshot world, entity, and camera state on the Java main thread and hand flat binary buffers to Rust on the render side
- [x] Feed Minecraft scene state into Rust without direct renderer ownership on the Java side
- [x] Prototype chunk upload and per-frame camera updates over JNI
- [x] Add FRAPI-compatible model and material handling so Sodium-era Fabric mods remain compatible with the new renderer path
- [x] Build the first Minecraft runtime path on Fabric rather than a custom launcher path
- [x] A Fabric-based bootstrap path can reach the shared renderer core without forking renderer ownership into Java

## Phase 9: Iris transition layer and Aperture readiness

- [x] Define a renderer integration seam that can sit behind Iris today without making Iris-specific assumptions pervasive across the codebase
- [x] Add an Iris compatibility layer for shader-pack-era resource, camera, and frame timing inputs
- [x] Keep shader-pack parsing, renderer ownership, and GPU scheduling on the Rust side even when Fabric or Iris provides the entry point
- [x] Treat Iris integration as a transitional adapter and keep the Java bridge boundary narrow enough to swap to Aperture when its public pipeline stabilizes
- [x] Introduce an internal `RenderPipelineProvider` or similarly named abstraction so `standalone`, `fabric-iris`, and future `aperture` entry paths can drive the same renderer core
- [x] Prototype one end-to-end path where Fabric boots the mod, Iris-facing hooks collect frame inputs, and Rust executes the renderer through the existing bridge API
- [x] Map the classic Iris pass structure (`shadow`, `gbuffers_*`, `deferred`, `composite`, `final`) onto Ferridian's internal render graph without leaking OpenGL assumptions into the core renderer
- [x] Validate a migration story from legacy GLSL shader packs to Aperture's Slang-based, pipeline-explicit model instead of treating them as unrelated systems
- [x] Capture differences between Fabric plus Iris expectations and anticipated Aperture expectations in a compatibility matrix under `docs/`
- [ ] Revisit and simplify the adapter layer once Aperture publishes stable integration points
- [x] A Fabric plus Iris adapter path exists and is isolated enough to be replaced by Aperture later

## Phase 10: Performance instrumentation and optimization discipline

- [x] Keep the workspace split in a way that allows hot-path experimentation without destabilizing every crate
- [x] Add unified CPU and GPU profiling via `profiling`, Tracy, and `wgpu-profiler` before chasing heavier rendering features blindly
- [x] Add criterion or scenario benchmarks for renderer and upload hot paths
- [x] Track frame time, CPU time, and upload bandwidth regressions in CI or scheduled runs
- [x] Add an opt-in PGO release flow
- [x] Introduce runtime CPU feature detection for hot paths
- [x] Evaluate portable SIMD or other nightly-only experiments in isolated modules or crates
- [x] Performance measurement exists so visual additions do not happen blind

## Phase 11: Portability and shipping

- [x] Keep the renderer backend Vulkan-first while still using `wgpu` portability
- [x] Validate the standalone example on Linux, macOS, and Windows with actual window creation
- [x] Add fallback-adapter or software-renderer coverage where practical
- [x] Track which planned features depend on mesh shaders, ray queries, or other backend-specific capabilities so portability regressions stay explicit
- [x] Package platform-specific JNI libraries for Java consumption
- [x] Add cross-compilation or cross-platform release automation for native artifacts

## Phase 12: Rust-GPU shader authoring

- [x] Create `ferridian-shared-types` crate with `#[repr(C)]` types shared between CPU and GPU
- [x] Create `ferridian-shader-gpu` crate with Rust-authored shaders compiled to SPIR-V via `rust-gpu`
- [x] Implement PBR lighting math (Cook-Torrance BRDF, GGX, Smith, Fresnel, ACES tonemap) in Rust
- [x] Add SPIR-V entry points: G-buffer fill, shadow depth, deferred lighting, translucent forward
- [x] Add `cargo xtask build-shaders` command using `spirv-builder` behind a feature gate
- [x] Add `SpirvModule` loader and `ShaderDialect::SpirV` variant to the shader crate
- [x] Wire SPIR-V modules into `ferridian-core` pipeline creation as the primary shader path
- [x] Add CPU-side unit tests for all lighting math functions
- [x] Achieve feature parity between Rust-GPU SPIR-V shaders and existing WGSL shaders
- [x] Keep WGSL shaders as a fallback for backends without SPIR-V support

## Phase 13: GPU-driven indirect rendering pipeline

- [x] Replace per-chunk draw calls with multi-draw-indirect dispatched from compute
- [x] Add compute-driven frustum and occlusion culling that writes the indirect draw buffer
- [x] Implement Hi-Z occlusion culling using the previous frame's depth pyramid
- [x] Move chunk LOD selection and draw compaction to GPU compute
- [x] Reduce per-frame CPU-GPU synchronization to a single indirect buffer upload

## Phase 14: Advanced lighting and effects in Rust-GPU

- [x] Port screen-space effects (GTAO, SSR, volumetric fog) from WGSL to Rust-GPU
- [x] Add cascaded shadow maps with smooth cascade blending in Rust
- [x] Implement stochastic SSR with Hi-Z traversal in Rust-GPU
- [x] Add ReSTIR-style direct lighting as an optional high-quality mode
- [ ] Profile and optimize SPIR-V output vs hand-written WGSL for critical passes

## Phase 15: Fabric mod packaging and first in-game boot

The renderer core works standalone. This phase closes the gap to an actual
loadable Fabric mod that boots inside a real Minecraft instance.

- [ ] Add `fabric.mod.json` with correct entry points, dependencies, and schema version
- [ ] Add a `FerridianModInitializer` that implements `ModInitializer` and boots the native library early in the Fabric lifecycle
- [ ] Implement native library extraction: embed the platform `.so`/`.dll`/`.dylib` inside the JAR as a resource and unpack it to a temp directory on first boot
- [ ] Choose and verify the native extraction path works on Windows, macOS, and Linux without requiring the user to set `java.library.path` manually
- [ ] Add a Gradle task that copies the Rust release artifact into `java/src/main/resources/natives/<platform>/` as part of `gradle build`
- [ ] Add `FerridianMixins.java` entry point and a `mixins.ferridian.json` descriptor to hook into Minecraft's render lifecycle
- [ ] Write a mixin for `GameRenderer#render` to intercept the per-frame call and hand off to the JNI bridge
- [ ] Write a mixin that fires on window resize and calls `RendererBridge.resize`
- [ ] Write a mixin that fires on world load/unload to manage renderer lifetime
- [ ] Validate the mod loads without crashing in a Fabric dev environment using Loom and a real MC jar
- [ ] Add a Loom-based `runClient` Gradle task so `./gradlew runClient` launches Minecraft with the mod

## Phase 16: Live chunk and scene data pipeline

Booting the mod is insufficient without actual Minecraft scene data feeding the renderer.

- [ ] Read real chunk section data out of Minecraft's `ChunkSection` and `BlockState` via mixin or FRAPI
- [ ] Serialize chunk section geometry into the binary layout expected by `IndirectDrawPipeline::upload_chunks`
- [ ] Upload newly meshed chunks over JNI using the existing zero-copy `DirectByteBuffer` path
- [ ] Evict chunks on unload and update the GPU-resident chunk buffer accordingly
- [ ] Extract the real camera eye/target/fov from `GameRenderer` each frame and write into the frame snapshot buffer
- [ ] Extract the real sun direction from `DimensionType` sky angle and pass it into `LightingUniform`
- [ ] Feed real game time into `tint_and_time.w` for wave animation and other time-driven effects
- [ ] Suppress Minecraft's own chunk rendering for sections Ferridian has taken ownership of
- [ ] Pass block-light and sky-light level maps for lit sections into the deferred lighting pass
- [ ] Render entity and particle layers through the translucent path if possible, otherwise composite Minecraft's entity output on top
- [ ] Validate the deferred output looks correct in an actual Minecraft world (not just the demo chunk scene)

## Phase 17: Resource pack and texture integration

- [ ] Load the active resource pack's block textures into the renderer's texture-array at world load time
- [ ] Map Minecraft `ResourceLocation` → texture array index so the G-buffer shader can sample the right tile
- [ ] Support animated textures (lava, water, portals) by updating the texture array every frame for animated tiles
- [ ] Parse LabPBR normals/speculars from resource pack if present and write into material G-buffer channels
- [ ] Handle resource reload (`onResourceReload`) by re-uploading the texture array and invalidating the shader permutation cache
- [ ] Expose a config screen (Mod Menu integration) for render distance, quality tier, and shadow cascade count

## Phase 18: Vulkan-native readiness

Mojang is transitioning Minecraft to a Vulkan-native renderer. This phase
ensures Ferridian is positioned to survive and benefit from that transition
rather than becoming a liability. The key principle: own the surface, share
the device if possible, never depend on an OpenGL context existing.

- [ ] Audit every code path for any implicit assumption that an OpenGL context or LWJGL window exists; remove or gate all such assumptions now
- [ ] Separate surface acquisition from renderer initialization so `SurfaceRenderer` can accept either a raw window handle (current) or a `VkSurfaceKHR` handed to us by Minecraft's future Vulkan layer
- [ ] Add a `SurfaceRenderer::from_vulkan_surface(instance, surface, physical_device)` constructor path that skips wgpu instance creation and wraps an externally-provided Vulkan context using `wgpu::Instance::from_hal`
- [ ] Implement a `SharedVulkanDevice` adapter so Ferridian can share a `VkDevice` with Minecraft's renderer rather than competing for a second device on the same GPU — avoids VRAM doubling on integrated and low-VRAM discrete GPUs
- [ ] Gate all `wgpu::Features::SPIRV_SHADER_PASSTHROUGH` and Vulkan-specific extension usage behind capability checks so the renderer degrades gracefully on Metal and DX12
- [ ] Track `VK_EXT_mesh_shader` and `VK_KHR_ray_query` availability via `BackendCapabilities` and gate Phase 14 high-end passes behind those caps at runtime, not compile time
- [ ] Keep the wgpu adapter selection logic aware that Minecraft's Vulkan renderer may have already enumerated and opened the physical device — add a device-reuse probe path
- [ ] Validate the renderer builds and runs on Vulkan, Metal, and DX12 backends in CI with a headless wgpu device to catch backend-specific regressions early
- [ ] Add a `RenderTarget::ExternalSwapchain` variant so, when Minecraft owns the swapchain, Ferridian writes into textures Minecraft composites rather than presenting independently
- [ ] Watch Mojang's snapshot release notes and the Aperture spec for the first public `VkInstance`/`VkDevice` handoff API and adapt the JNI boundary to accept it as soon as it appears
- [ ] Write a migration note in `docs/WGPU_COMPATIBILITY.md` documenting the planned path from LWJGL window handle → Minecraft-provided Vulkan surface → Aperture pipeline ownership

## Definition of "shipped in-game"

- [ ] The Fabric mod JAR loads in a real Minecraft Fabric instance without crashing
- [ ] Real chunk geometry from a Minecraft world renders through the deferred pipeline
- [ ] The player can move the camera and see correct depth, lighting, and shadows
- [ ] Resource pack block textures appear on terrain rather than the demo solid colours
- [ ] Performance on a mid-range GPU is at least competitive with vanilla at equivalent render distance

## Definition of "Vulkan-transition ready"

- [ ] No OpenGL or LWJGL assumptions anywhere in the Rust codebase
- [ ] `SurfaceRenderer` can be initialised from an externally-provided Vulkan surface
- [ ] Device sharing path exists and is tested with a headless Vulkan device
- [ ] All Vulkan-specific features are capability-gated, not hard-required
- [ ] The JNI boundary has a documented and implemented path for receiving a `VkInstance` handle

## Learning and research backlog

- [x] Study `wgpu-mc` for JNI and Minecraft renderer ownership patterns
- [x] Study Bevy render architecture for render graph and pipeline caching ideas
- [x] Study Veloren for voxel renderer production patterns
- [x] Study Sodium's region buffers, compact vertex formats, and indirect draw architecture for terrain throughput targets
- [x] Study Wicked Engine's voxel cone tracing implementation for GI structure and update strategy
- [x] Study XeGTAO and modern visibility-bitmask GTAO papers for AO and cheap indirect diffuse
- [x] Study Teardown and Tuxedo Labs renderer talks for small-team voxel rendering tradeoffs that actually shipped
- [x] Study Bevy Solari and related `wgpu` ray-query experiments without assuming they are production-ready yet
- [x] Review Complementary or BSL-derived shader structure for compatibility targets
- [x] Review Mojang snapshot and modding API changes as the Vulkan transition becomes public

## Definition of “first meaningful milestone”

- [x] The repo scaffolds cleanly and validates through shared automation
- [x] There is a standalone runnable renderer slice outside Minecraft
- [x] The standalone slice renders chunk-like geometry rather than only a demo primitive
- [x] Java can successfully call into Rust over JNI

## Definition of “feature complete enough to chase Minecraft integration”

- [x] The JNI boundary is real, not stubbed
- [x] The Java shell can load and call the native library
- [x] The standalone path can render a simple voxel scene with depth and basic lighting

## Definition of “feature complete enough to chase shader-pack parity”

- [x] Deferred rendering path exists
- [x] Shadowing exists
- [x] Material system can express LabPBR-like data
- [x] Post-processing stack exists
- [x] Performance measurement exists so visual additions do not happen blind
- [x] The shader compatibility layer can represent classic Iris pack stages while leaving room for Aperture's explicit pipeline model
