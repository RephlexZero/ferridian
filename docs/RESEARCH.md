# Ferridian: technology landscape and strategic roadmap

**Ferridian's highest-leverage move is not chasing AAA techniques — it's exploiting the structural advantage that a voxel world is already voxelized.** This single fact eliminates the most expensive step in global illumination (scene voxelization), makes GPU-driven rendering natural, and means techniques like voxel cone tracing cost a fraction of what they cost in polygon engines. The Rust/wgpu stack is production-ready for this work: wgpu 28.x ships mesh shaders, experimental ray queries, and cross-platform Vulkan/Metal/DX12; naga handles WGSL↔SPIR-V↔GLSL translation at 4–13× the speed of alternatives; and jni-rs 0.22 provides lifetime-safe JNI with zero-copy `DirectByteBuffer` paths. The Iris/Aperture ecosystem is converging toward a pipeline-agnostic shader format (now Slang-based), which aligns well with a non-OpenGL backend. Teardown — built by one developer on OpenGL 3.3 — proves that algorithmic cleverness on simple APIs outperforms complex APIs with naive algorithms. Ferridian should internalize this lesson.

---

## 1. Executive summary: ten highest-value findings

**Finding 1 — wgpu 28.x is production-ready with critical native extensions.** Mesh shaders are fully supported on Vulkan/Metal/DX12. Experimental ray queries (inline ray tracing) work on Vulkan. Bindless is partially available via `binding_array` but not yet production-quality. The 12-week breaking-release cadence demands version pinning. Async compute (multi-queue) is **not** exposed — the single-queue model means all GPU work is sequential within command encoders.

**Finding 2 — The voxel world IS the acceleration structure.** Unlike polygon engines that spend 2–5ms per frame voxelizing geometry for GI, a Minecraft-like world hands you the voxel grid for free. This makes voxel cone tracing (VCT) uniquely cheap: skip voxelization entirely, inject lighting into cascaded 3D textures, and cone-trace the final gather. Wicked Engine ships this in production at 3–8ms.

**Finding 3 — GPU-resident chunks with indirect draws yield 90% CPU reduction.** Sodium proved this definitively: multi-draw indirect batching, compact **8-byte vertices** (two `u32`s packing position, UVs, face, AO, texture ID, light), and region-based buffer arenas cut CPU overhead by an order of magnitude. This is the single most impactful optimization for any voxel renderer.

**Finding 4 — Iris's Aperture format has moved to Slang and is pipeline-agnostic.** Aperture is not a separate renderer — it's Iris's next-gen shader pack specification. It recently migrated from TypeScript to Microsoft's Slang shading language. Unlike the legacy OptiFine format (which assumes OpenGL semantics), Aperture requires explicit pipeline construction, making it far more compatible with a Vulkan/wgpu backend. This is Ferridian's strategic opening.

**Finding 5 — Radiance cascades are proven in 2D but unproven in 3D.** Alexander Sannikov's technique achieves sub-millisecond 2D GI on GTX 970 with no temporal accumulation. The 3D extension faces a fundamental scaling problem: angular resolution requirements in 3D are 16–64× higher than 2D. For voxel worlds, the "dealbreaker" of scene voxelization doesn't apply, but no production 3D implementation exists. Watch closely; don't bet on it.

**Finding 6 — ReSTIR DI is production-proven and accessible via wgpu's experimental ray queries.** Bevy's Solari plugin demonstrates ReSTIR DI + GI running on wgpu's experimental ray tracing. ReSTIR DI solves the many-light problem (thousands of block lights) with reservoir-based sampling. However, wgpu lacks ray tracing pipeline support, limiting this to inline ray queries only. Shader Execution Reordering (SER), which delivers 20–40% RT speedups, is also blocked on RT pipeline support.

**Finding 7 — jni-rs 0.22 supports zero-copy rendering data transfer.** The critical path is `DirectByteBuffer`: Java allocates direct byte buffers, Rust accesses the native pointer with zero copy. Cache `jmethodID`/`jfieldID` on first lookup, pre-attach render threads permanently, and batch all world updates into 1–2 JNI calls per frame. The wgpu-mc project (533 GitHub stars) validates this exact architecture pattern for a Rust Minecraft renderer.

**Finding 8 — Froxel-based volumetric fog is the best bang-for-buck visual upgrade.** Industry-standard (Frostbite, Unreal, Unity HDRP), 1–3ms cost, and god rays come free. For a voxel world, the voxel grid itself can serve as the density/extinction volume. Minecraft shader packs use simpler per-pixel ray marching, but froxels are more efficient and extensible.

**Finding 9 — GTAO with visibility bitmask delivers AO + indirect diffuse in one pass.** The 2023 visibility bitmask variant of GTAO (Therrien et al.) replaces horizon angles with a 32-bit sector bitmask, computing both ambient occlusion and screen-space indirect lighting simultaneously. XeGTAO (Intel, MIT license) is the reference implementation at ~0.56ms/1080p. This is the highest-value screen-space technique.

**Finding 10 — Teardown's architecture proves the viability of a one-developer voxel renderer.** Dennis Gustafsson shipped a commercially successful voxel game using OpenGL 3.3, fragment-shader ray marching against 3D textures, a 1-bit volumetric shadow map for shadows/AO/reflections, and 8-bit palette materials. No compute shaders, no hardware RT. The new Tuxedo Labs engine (with Gabe Rundlett) moves to Vulkan HW RT + path tracing + DLSS. Both generations validate the approach.

---

## 2. Ranked recommendations

### Adopt now

| Technique | Why | Upside | Complexity | Risk | Dependencies | Fit |
|-----------|-----|--------|------------|------|--------------|-----|
| **GPU-resident chunks + indirect draw** | 90% CPU reduction, proven by Sodium | Transformational frame time improvement | 3–4 weeks | Low — well-documented pattern | wgpu compute, buffer management | Core renderer |
| **Compact vertex packing (8 bytes)** | 40% VRAM reduction, 18× fewer vertices with greedy meshing | Doubles renderable chunk count within memory budget | 1–2 weeks | Very low | Greedy meshing algorithm | Core renderer |
| **Hybrid deferred + forward transparency** | Enables cheap multi-light; natural for voxel worlds with many block emitters | Decouples geometry from lighting; unlocks all post-processing | 4–6 weeks | Low — industry standard | G-buffer setup | Core renderer |
| **naga_oil for modular WGSL** | Shader composition with `#import`, `#define`, hot-reload | Fast iteration, modular shader development | 1 week | Very low | naga_oil crate | Shader subsystem |
| **wgsl_bindgen for type-safe bindings** | Catches binding errors at compile time | Eliminates a class of GPU debugging pain | 3 days | Very low | wgsl_bindgen crate | Shader subsystem |
| **jni-rs 0.22 + DirectByteBuffer** | Zero-copy chunk data transfer, lifetime-safe API | Minimal JNI overhead on hot path | 2 weeks | Low — proven by wgpu-mc | jni-rs 0.22, Java DirectByteBuffer | JNI bridge |
| **XeGTAO ambient occlusion** | MIT-licensed, 0.56ms at 1080p, state-of-the-art quality | Immediate visual quality jump | 1–2 weeks | Very low — reference implementation exists | G-buffer depth + normals | Post-processing |
| **Texture arrays for block materials** | No UV bleeding, per-layer mipmapping, wgpu-native | 256 layers × 16² × 4 bytes = 256KB per array | 1 week | Very low | wgpu texture array support | Material system |
| **profiling crate + tracy + wgpu-profiler** | Unified CPU/GPU profiling | Data-driven optimization from day one | 2 days | Very low | profiling, tracy-client, wgpu-profiler crates | Tooling |
| **Rayon for chunk meshing** | CPU-parallel greedy meshing on background threads | Eliminates meshing lag spikes | 1 week | Very low | rayon crate | Core renderer |

### Prototype next

| Technique | Why | Upside | Complexity | Risk | Dependencies | Fit |
|-----------|-----|--------|------------|------|--------------|-----|
| **Voxel cone tracing GI** | Voxel world = free voxelization; proven in Wicked Engine | High-quality diffuse GI at 3–8ms | 2–3 months | Medium — tuning cascaded 3D textures is nontrivial | 3D texture support, mipmap chain | Shader/lighting |
| **Froxel volumetric fog** | Industry standard, god rays included, 1–3ms | Atmospheric depth, volumetric lighting | 3–4 weeks | Low — well-documented | Compute shaders, 3D textures, shadow maps | Post-processing |
| **Stochastic SSR with Hi-Z** | AMD FidelityFX SSSR as MIT reference | Glossy reflections for water, metals, wet surfaces | 2–3 weeks | Low — reference exists | Hi-Z buffer, G-buffer | Post-processing |
| **LOD system (3-tier)** | Distant Horizons proves 512-chunk distances work | Massive draw distance at constant GPU cost | 4–6 weeks | Medium — transition seams are tricky | Chunk management, octree or simplified meshes | Core renderer |
| **Iris shader pack compatibility layer** | Unlocks entire existing shader pack ecosystem | Huge community adoption leverage | 2–4 months | High — GLSL→WGSL translation, uniform mapping is complex | naga GLSL frontend, uniform compatibility | Shader subsystem |
| **LabPBR material support** | Industry standard for Minecraft PBR resource packs | Enables PBR lighting, parallax, emission | 2–3 weeks | Low — well-specified format | Texture arrays, normal/specular maps | Material system |
| **Custom TAA-based temporal upscaling** | Render at lower res, temporal accumulate, sharpen | 2× or more performance improvement | 2–3 weeks | Medium — ghosting artifacts need careful tuning | Motion vectors, jittered projection | Post-processing |
| **Render graph with resource pooling** | Clean pass management, automatic barrier insertion | Maintainable multi-pass pipeline | 3–4 weeks | Low — Bevy's architecture as reference | wgpu resource management | Core architecture |
| **Visibility bitmask SSGI** | AO + indirect diffuse in one pass, incremental over GTAO | Screen-space GI essentially free on top of GTAO | 1–2 weeks | Low — published technique with code | XeGTAO already implemented | Post-processing |

### Watch closely

| Technique | Why watching | Trigger to adopt | Timeline |
|-----------|-------------|-----------------|----------|
| **Radiance cascades 3D** | Potentially optimal GI for voxel worlds; no voxelization needed | Production-quality 3D implementation demonstrated | 12–18 months |
| **ReSTIR DI via wgpu ray queries** | Solves many-light problem elegantly; Bevy Solari validates wgpu path | wgpu ray query support matures beyond experimental | 6–12 months |
| **Aperture shader format** | Pipeline-agnostic, Slang-based, aligns with non-OpenGL backends | Aperture exits beta, shader packs adopt it | 6–12 months |
| **wgpu bindless resources** | Eliminates texture array workarounds, enables true GPU-driven rendering | WebGPU bindless proposal lands in wgpu | 6–12 months |
| **wgpu async compute (multi-queue)** | Overlapping compute and graphics for free performance | wgpu exposes multiple queues | 12+ months |
| **Shader Execution Reordering** | 20–40% RT performance boost, now multi-vendor Vulkan extension | wgpu adds ray tracing pipeline support | 12–18 months |
| **FSR 2 WGSL port** | MIT-licensed temporal upscaler, better than custom TAA | Community or self port of HLSL compute shaders to WGSL | 3–6 months |
| **Work graphs / device generated commands** | Full GPU-driven pipeline without CPU roundtrips | Vulkan VK_EXT_device_generated_commands matures | 18+ months |

### Avoid for now

| Technique | Why avoid | What to do instead |
|-----------|-----------|-------------------|
| **Full Nanite-style software rasterizer** | 100K+ lines of code, requires AAA team, unnecessary for voxels | Use indirect draws + LOD; voxels don't need Nanite's polygon LOD |
| **Lumen-style multi-fallback GI** | 6–12 person-months, designed for general polygon scenes | Use VCT (cheap for voxels) or SSGI |
| **DLSS/XeSS integration** | Proprietary, vendor-locked, binary SDKs | Custom TAA upscaling or FSR 2 port |
| **Rust code hot-reload** | Fragile with complex state, not production-ready | Shader hot-reload via file watcher + pipeline recreation |
| **std::simd (portable_simd)** | Still nightly-only, stabilization timeline unclear | Use `wide` crate on stable Rust, or `std::arch` intrinsics (safe since Rust 1.87) |
| **Canvas renderer format compatibility** | Tiny ecosystem (56K downloads), incompatible with Sodium/Iris | Focus on Iris/OptiFine format, then Aperture |
| **Full path tracing as default** | Requires RTX hardware, heavy denoising infrastructure | Offer as "Ultra" quality tier only; default to VCT or SSGI |
| **rend3** | Abandoned, last release 4 years ago | Use Bevy render architecture as reference; build custom |

---

## 3. Concrete applications mapped to Ferridian subsystems

### Core renderer (`ferridian-core`)

The core renderer should adopt a **simplified retained render graph** inspired by Bevy's architecture but tailored for voxel rendering. The graph declares passes as nodes with explicit resource dependencies; wgpu handles most barriers automatically, but compute↔render transitions need explicit texture usage declarations.

**G-buffer layout (128 bits/pixel, 4 render targets):** RT0 stores albedo RGB + material flags in RGBA8. RT1 packs octahedral normals (2×16-bit) + smoothness + block light in RG16+BA8. RT2 holds F0/metallic + emission + AO + block ID in RGBA8. Hardware depth uses D32F. Position reconstruction from depth + inverse projection eliminates a render target. This layout supports deferred lighting with hundreds of block light sources at constant cost.

**Chunk pipeline:** Rayon-parallel greedy meshing on background threads produces **8-byte packed vertices** (position in 5+5+5 bits, UVs in 5+5 bits, face direction in 3 bits, AO in 2 bits, texture ID in 10 bits, light in 8 bits). Meshes upload into a pre-allocated **gigabuffer** (256–400MB) with virtual sub-allocation. A compute shader performs frustum culling against chunk AABBs, writes surviving draw commands to an indirect buffer, and a single `draw_indexed_indirect` call per material tier renders everything. This pattern matches Sodium's proven 90% CPU reduction.

**Triple-buffered frame data** separates the game logic thread (JNI interface, world updates) from the render thread (wgpu command encoding, submission). Arc-wrapped `Device` and `Queue` enable safe cross-thread access. A ring buffer handles per-frame uniform uploads.

### Shader subsystem (`ferridian-shader`)

The shader system needs two parallel tracks: a native WGSL pipeline for engine-authored shaders, and a compatibility layer for Iris/OptiFine shader packs.

**Native pipeline:** WGSL authored with naga_oil for `#import`/`#define` composition. wgsl_bindgen generates type-safe Rust bindings at build time. Shader hot-reload uses the `notify` crate watching shader directories, with 100ms debouncing, triggering `ShaderModule` + `RenderPipeline` recreation. Pipeline caching uses a hash of (shader source + defines + vertex format + render state).

**Compatibility layer:** The Iris/OptiFine pipeline has a fixed structure — shadow → shadowcomp → prepare → gbuffers_* → deferred → gbuffers_water → composite → final — with up to 16 color attachments (colortex0–15) and standardized uniforms (gbufferModelView, shadowProjection, sunPosition, etc.). Ferridian's compatibility adapter maps these shader programs to native render graph passes, translates GLSL to SPIR-V via naga's GLSL frontend, and provides the expected uniform set. The `SodiumTransformer` pattern from Iris (remapping `gl_ModelViewMatrix` → `iris_ModelViewMatrix`, `gl_Vertex` → `getVertexPosition()`) is the template for attribute translation.

**Shader permutation management:** Preprocessor defines select variants (`#define SHADOW_QUALITY 2`). Pre-compile common permutations; lazy-compile rare ones. wgpu's internal pipeline cache handles GPU-side caching.

### Minecraft integration (`ferridian-minecraft`)

**Data flow architecture:** Follow wgpu-mc's proven pattern. Mixin hooks disable Blaze3D (Minecraft's OpenGL renderer). The Java side collects world state (block states, entity data, camera, time/weather) into `DirectByteBuffer`s. A single JNI call per frame transfers buffer pointers to Rust. The Rust renderer processes independently, returning only a "frame complete" signal.

**Required data from Minecraft:** Block states (ID + properties) per position, block models (JSON), texture atlases (base + `_n` + `_s` for LabPBR), entity positions/models/animations, camera position/rotation, time of day, weather strength, biome at camera, dimension, fog settings, chunk load/unload events, and block/sky light levels. This data should be serialized into flat binary layouts matching Rust struct representations via `bytemuck`.

**Fabric integration:** Fabric's WorldRenderEvents (BEFORE_ENTITIES, AFTER_ENTITIES, AFTER_TRANSLUCENT) provide hook points. FRAPI (Fabric Rendering API) support is essential for mod compatibility — since Sodium 0.6.0, FRAPI is built-in and Indium is deprecated. Ferridian must implement FRAPI's `RenderMaterial` and `Mesh` APIs to support modded block rendering. The NativeLoader Fabric mod pattern demonstrates native library loading via `System.loadLibrary()`.

**Thread safety:** JNI environment is per-thread. Use `JavaVM::attach_current_thread_permanently()` for the render thread. Snapshot world data on the main thread (Sodium's `WorldDataSnapshots` pattern), hand off to render thread via crossbeam channel. Never call JNI from the render thread during frame submission.

### JNI bridge (`ferridian-jni`)

The bridge should minimize crossings to **2–3 calls per frame**: one to receive the world-state buffer pointer, one to signal frame completion, and optionally one for input events. Cache all `jmethodID`/`jfieldID` values on initialization. Use `GlobalRef` for long-lived Java objects (the Minecraft instance, world reference). Error handling via `jni::errors::Result` with automatic Java exception translation.

**Zero-copy strategy:** Java allocates `ByteBuffer.allocateDirect()` for chunk data, entity lists, and camera state. Rust accesses via `GetDirectBufferAddress` — no copy. For chunk mesh uploads, the Rust side writes directly into the wgpu staging buffer, then submits. The only copy is the GPU upload.

**Alternative to watch:** Java 22+ Foreign Function & Memory (FFM) APIs show better performance than JNI in benchmarks. The KryptonFNP Fabric mod tests this. If Minecraft's Java version advances, FFM could replace JNI.

### Java shell (`ferridian-java`)

The Java shell is a thin Fabric mod: Mixin hooks to disable vanilla rendering, world state extraction, JNI native method declarations, and Fabric event registration. It should expose a configuration GUI (via Fabric's config API or Cloth Config) for quality presets (Low/Medium/High/Ultra mapping to shader quality tiers). The shell handles resource pack loading (extracting textures, models, shader packs) and passes paths to the Rust side.

---

## 4. Architecture proposals by horizon

### Two-week sprint: deferred foundation

Complete the hybrid deferred pipeline. Implement the 4-RT G-buffer layout with the compact encoding described above. Wire up a basic deferred lighting pass handling a directional sun light + ambient. Add depth-only pre-pass for the forward transparency stage (water, glass). Integrate XeGTAO for ambient occlusion — port the MIT-licensed HLSL compute shaders to WGSL. Set up the `profiling` + `tracy` + `wgpu-profiler` stack for data-driven iteration. At the end of two weeks, Ferridian should render opaque voxel terrain with deferred lighting, basic shadows (single cascaded shadow map), and GTAO. This establishes the rendering backbone that every subsequent feature builds on.

### Two-month milestone: competitive visual quality

**Weeks 1–3:** Implement GPU-resident chunk rendering. Pre-allocate a 256MB gigabuffer for mesh data. Build compute shader frustum culling writing indirect draw commands. Switch from per-chunk draw calls to single indirect draws per material tier. Target: >90% CPU time reduction for world rendering.

**Weeks 3–5:** Build the LabPBR material system with texture arrays. Load `_n` (normal) and `_s` (specular) textures per block. Implement the LabPBR decode in the deferred lighting shader: perceptual smoothness → roughness conversion, F0/metal discrimination (0–229 dielectric, 230–254 predefined metals), emission, subsurface scattering flags. Fall back to flat normals + default specular for blocks without PBR textures.

**Weeks 5–7:** Post-processing pipeline. TAA with Halton jitter + YCoCg neighborhood clamping. HDR bloom (downsample chain + Gaussian blur + upsample, ~0.3ms). Tone mapping (start with ACES, evaluate AgX). Froxel-based volumetric fog (compute shader light injection into 160×90×64 froxel grid, front-to-back accumulation). Stochastic SSR with Hi-Z traversal for water and wet surfaces.

**Week 8:** Begin Iris shader pack loading. Parse `shaders.properties`, load GLSL shader programs, compile through naga's GLSL frontend to SPIR-V. Map colortex0–15 to internal render targets. Provide the standard uniform set. Target: basic BSL or Complementary shader pack loads and partially renders.

At the two-month mark, Ferridian should produce visuals competitive with mid-tier Minecraft shader packs (BSL-level), with significantly better frame times due to the Rust/wgpu architecture.

### Six-month milestone: ecosystem-ready release

**Months 3–4:** Complete Iris/OptiFine shader pack compatibility. Handle all gbuffers_* programs, deferred passes, composite passes, shadow mapping with configurable resolution, and the `final` program. Implement the `SodiumTransformer`-style attribute remapping. Test against top shader packs: Complementary Reimagined, BSL, SEUS Renewed, Sildur's Vibrant. This alone makes Ferridian useful to the Minecraft community.

**Month 4:** Three-tier LOD system. Near (0–8 chunks): full greedy-meshed geometry. Medium (8–32 chunks): simplified octree-based meshes with 4× fewer vertices. Far (32–128 chunks): heightmap-based billboards or stripped geometry à la Distant Horizons. Implement chunk streaming with LRU eviction capped to a VRAM budget. Add Hi-Z occlusion culling via compute shader.

**Month 5:** Voxel cone tracing GI. Inject direct lighting into cascaded 3D textures (3 cascades covering near/mid/far). Generate mipmap chains for cone widening. Trace 5–9 cones per pixel in the deferred lighting pass for diffuse GI. Since the world IS voxels, skip voxelization entirely — read block data directly into the 3D textures during chunk updates. Target: 3–5ms for GI at 1080p, configurable cone count for quality scaling.

**Month 6:** Aperture format support. Parse Slang-based pack definitions. Implement the `configureRenderer` / `configurePipeline` interface. Support combination passes, command lists, and explicit pipeline construction. This positions Ferridian as a first-mover for the next generation of Iris shader packs. Also: implement PGO builds (`RUSTFLAGS="-Cprofile-generate"` → profile → `-Cprofile-use"`), fat LTO with `codegen-units = 1`, and `target-cpu=native` for release builds. Target: **10–20% additional performance** from compiler optimizations alone.

---

## 5. Competitive landscape: what transfers vs what's hype

### Proven and directly transferable

**GPU-resident geometry + indirect draws** is the single most impactful technique across the industry. Sodium, Bevy 0.16, Unity 6's GPU Resident Drawer, and Teardown all validate it. For Ferridian, this means pre-allocated gigabuffers, compute-shader culling, and single indirect draw calls per material tier. Implementation is 3–4 weeks and the payoff is immediate.

**Deferred rendering** is universal. Every competitive renderer uses it for opaque geometry. The hybrid approach (deferred opaques + forward transparency) is industry-standard in Unity HDRP, Bevy, and every Minecraft shader pack. For Ferridian, the G-buffer naturally maps to voxel attributes (block ID, face normal, material properties).

**Voxel cone tracing** is the technique where Ferridian has the largest structural advantage. In polygon engines (Wicked Engine, CryEngine), voxelization costs 2–5ms per frame and produces lossy approximations. In a voxel world, the scene IS the voxel grid — voxelization is free and lossless. This means VCT costs roughly half what it costs in polygon engines, while producing higher quality results. **This is Ferridian's competitive moat for GI.**

**Teardown's architectural philosophy** transfers completely. Fragment-shader ray marching against 3D textures, volumetric shadow maps for multi-purpose occlusion, and 8-bit palette materials are all implementable by a small team. Teardown's commercial success on GTX 1070-class hardware proves that software voxel rendering without RT hardware is viable.

### Partially transferable — worth adapting, not copying

**UE5's visibility buffer** concept is transferable in principle: render object IDs + depth first, then shade only visible pixels. For voxels, this eliminates overdraw on complex terrain. But the full Nanite pipeline (software rasterizer, hierarchical cluster LOD, streaming) is 100K+ lines and unnecessary for voxels — voxels already have natural LOD via octree levels.

**Lumen's screen-space probe cache** is a good idea in isolation: place probes in screen space, gather irradiance, interpolate. But Lumen's full system with its SWRT/HWRT fallback paths, mesh distance fields, and radiosity final gather requires AAA-scale engineering. For Ferridian, a simpler probe-based approach (DDGI-style) combined with VCT is more appropriate.

**Unity 6's Render Graph API** validates the frame-graph pattern for managing passes. Bevy's render graph (inspired by Bungie's Destiny renderer) is the closest Rust reference. Ferridian should adopt the pattern (topologically sorted DAG of passes with declared resource dependencies) without copying either engine's full complexity.

### Hype — avoid or defer

**Nanite for voxels** is a category error. Nanite solves automatic LOD for polygon meshes with millions of triangles. Voxel worlds don't have this problem — their geometry is already regular and LOD-able via simple octree decimation. The concept of "smart streaming and GPU culling" transfers; the full Nanite pipeline does not.

**"Real-time path tracing for everyone"** ignores hardware reality. Tuxedo Labs' new engine uses DLSS Ray Reconstruction on RTX hardware; SEUS PTGI achieves software path tracing but at 10–20ms and with aggressive denoising. Path tracing should be offered as an "Ultra" quality tier, never as the default.

**Radiance cascades in 3D** has extraordinary theoretical properties but zero production 3D implementations. The 2D version is proven (sub-millisecond, no temporal accumulation). The 3D extension faces 16–64× scaling from 2D angular resolution to spherical. For voxel worlds, the absence of voxelization cost is a genuine advantage, but this remains a research bet.

**Work graphs and device-generated commands** are not accessible from wgpu today. DX12 work graphs are real (AMD, NVIDIA, Intel all support v1.0), and Vulkan's `VK_EXT_device_generated_commands` is the multi-vendor equivalent. But wgpu abstracts over these backends and doesn't expose either. The workaround — compute dispatch chains with indirect dispatch — covers 90% of GPU-driven use cases.

### The honest comparison

| Engine/Project | Team size | GI approach | Voxel-specific? | Rust? | Status |
|---|---|---|---|---|---|
| **Teardown** (Tuxedo Labs) | 1→~10 | Volumetric shadow map (no true GI) → path tracing (new engine) | Yes | No (C++) | Shipped, commercial success |
| **Veloren** | ~50 contributors | Basic lighting, no GI | Yes | Yes (wgpu) | Playable, active |
| **wgpu-mc/Electrum** | ~5 contributors | Delegates to shader packs | Minecraft | Yes (wgpu + JNI) | WIP, engine "fairly mature" |
| **Blaze4D** | Small team | N/A (early) | Minecraft | Yes (Rust core + JNI) | Early, stale |
| **VulkanMod** | 1–2 devs | Delegates to vanilla/shader packs | Minecraft | No (Java + Vulkan) | Active, 685K downloads |
| **Sodium + Iris** | CaffeineMC + IrisMC | Delegates to shader packs | Minecraft | No (Java + OpenGL) | De facto standard, 100M+ downloads |
| **Bevy** | ~100+ contributors | Forward/deferred (0.16: GPU-driven) | No | Yes (wgpu) | Active, general-purpose |
| **Wicked Engine** | 1 lead + contributors | Voxel cone tracing | No (but VCT works for voxels) | No (C++) | Active, reference quality |

Ferridian's unique positioning: the only project combining Rust/wgpu, JNI-bridged Minecraft integration, AND voxel-optimized GI (VCT with free voxelization). wgpu-mc is the closest competitor but focuses on vanilla rendering parity rather than advanced techniques.

---

## 6. Actions, bets, and traps

### Top 5 immediate actions

**Action 1: Implement GPU-resident chunks with indirect draws.** This is the single highest-ROI change. Allocate a 256MB gigabuffer, implement compute-shader frustum culling, and batch all terrain into one indirect draw per material tier. Every millisecond saved here compounds across all future features. Reference Sodium's architecture and Nick's vertex-pool blog post. Timeline: 3 weeks.

**Action 2: Complete the G-buffer and deferred lighting pipeline.** Use the 4-RT 128-bit/pixel layout. Implement directional sun lighting + a clustered/tiled approach for block lights. Add XeGTAO immediately — it's MIT-licensed, well-documented, and the visual impact is outsized for the implementation cost. Timeline: 2 weeks.

**Action 3: Set up the LabPBR material system with texture arrays.** Create three texture arrays (albedo, normal, specular) with one layer per block face. Decode LabPBR in the deferred lighting shader. This immediately enables Ferridian to render PBR resource packs and differentiates from vanilla Minecraft's flat lighting. Timeline: 2 weeks.

**Action 4: Build the JNI data pipeline with DirectByteBuffer.** Implement zero-copy world state transfer. Define flat binary layouts for chunk data (block IDs as u16 array per 16³ section), entity state (position, rotation, model ID), and camera state. Cache all JNI IDs on load. Test round-trip latency — target <0.1ms per frame for JNI overhead. Timeline: 2 weeks.

**Action 5: Establish shader hot-reload and profiling infrastructure.** Wire `notify` crate to watch shader directories, debounce at 100ms, and recreate affected pipelines. Integrate `profiling` + `tracy` + `wgpu-profiler` for CPU and GPU timing. This investment pays back on every subsequent feature by cutting iteration time. Timeline: 3 days.

### Top 5 speculative bets

**Bet 1: Voxel cone tracing with free voxelization becomes Ferridian's signature feature.** If VCT can run at 3–5ms with the voxel world providing free scene data, Ferridian offers GI quality that polygon renderers can't match at the same cost. The bet is that tuning cascaded 3D texture updates (incremental on chunk changes, not full rebuild) keeps costs bounded.

**Bet 2: Aperture format adoption creates a first-mover advantage.** If Aperture exits beta and shader pack authors adopt it, Ferridian's non-OpenGL backend becomes an advantage rather than a compatibility problem. The Slang-based pipeline specification is inherently more portable than GLSL. The risk: Aperture may stall in development or fail to attract pack authors.

**Bet 3: ReSTIR DI via wgpu ray queries solves the many-light problem elegantly.** wgpu's experimental ray query support is enough for inline ReSTIR DI. If it matures, Ferridian could handle thousands of colored block lights (torches, lava, glowstone, redstone) with physically correct falloff and shadowing. The risk: wgpu's RT support may remain experimental and buggy for years.

**Bet 4: 3D radiance cascades become practical for voxel worlds first.** The technique's biggest objection (scene voxelization cost) doesn't apply to voxel worlds. If Sannikov or the community produces a viable 3D implementation, Ferridian could adopt it for sub-millisecond, temporally stable GI. The risk: the 3D constant factor may be prohibitively large even with free voxelization.

**Bet 5: Rust + wgpu + JNI becomes the standard architecture for next-gen Minecraft mods.** If Ferridian, wgpu-mc, and Blaze4D collectively validate this pattern, the Minecraft modding community may shift toward native renderers for performance-critical work. The risk: Java FFM APIs improve enough that JNI overhead becomes the bottleneck, or the community rejects native dependencies as too complex to distribute.

### Top 5 traps to avoid

**Trap 1: Chasing wgpu API stability.** wgpu ships breaking changes every 12 weeks. Don't build against `main` or chase the latest release during active feature development. Pin your version, update deliberately between milestones, and isolate wgpu calls behind a thin abstraction layer so API changes don't cascade through the codebase.

**Trap 2: Premature shader pack compatibility before the native pipeline works.** The Iris/OptiFine shader format is GLSL-based and assumes OpenGL semantics (gl_ModelViewMatrix, gl_FragData, implicit alpha test). Translating this to WGSL/Vulkan requires extensive attribute remapping, uniform compatibility, and careful framebuffer management. Don't attempt this until the native deferred pipeline is solid — otherwise you'll be debugging two rendering architectures simultaneously.

**Trap 3: Treating bindless as a prerequisite.** wgpu's bindless support (`binding_array`) is partially available but not production-quality. Many projects block on "waiting for bindless." Don't. Use texture arrays (up to 256 layers, 2048×2048 each on most hardware) as a workaround. Texture arrays cover 100% of Minecraft block rendering needs. Bindless is a nice-to-have for entity textures; it's not blocking.

**Trap 4: Over-engineering the render graph.** A production render graph with automatic barrier insertion, resource aliasing, transient resource pooling, and async compute scheduling is 2–3 months of work. Ferridian's pass count is small (shadow, G-buffer, deferred light, transparent, composite, post-process, final). A manually-ordered pass list with explicit resource declarations covers this comfortably. Build the simplest render graph that works; refactor to a DAG only when the pass count exceeds ~15.

**Trap 5: Ignoring mod compatibility in favor of rendering features.** Minecraft modding lives or dies on compatibility. If Ferridian breaks mods that make OpenGL calls, use FRAPI custom models, or depend on Sodium's internal APIs, adoption will be near zero regardless of visual quality. The wgpu-mc project explicitly warns about this: disabling Blaze3D "breaks mods that make direct GL calls." Plan for an FRAPI compatibility layer from the start, and test against top-20 Fabric mods continuously.