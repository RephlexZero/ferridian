# Iris / Aperture Compatibility Matrix

This document captures the differences between Fabric + Iris expectations and
anticipated Aperture expectations so the Ferridian adapter layer can serve both.

## Pass stage mapping

| Iris / OptiFine Stage  | Ferridian Pass Kind    | Aperture Equivalent        |
|------------------------|------------------------|----------------------------|
| `shadow`               | `ShadowMap`            | Explicit shadow pipeline   |
| `gbuffers_terrain`     | `GBufferFill`          | Material pass              |
| `gbuffers_entities`    | `GBufferFill`          | Material pass              |
| `gbuffers_block`       | `GBufferFill`          | Material pass              |
| `gbuffers_water`       | `Translucent`          | Translucent pipeline       |
| `gbuffers_skybasic`    | `GBufferFill`          | Sky pass                   |
| `gbuffers_skytextured` | `GBufferFill`          | Sky pass                   |
| `gbuffers_weather`     | `Translucent`          | Particle / weather pass    |
| `gbuffers_hand`        | `GBufferFill`          | UI overlay                 |
| `deferred`             | `DeferredLighting`     | Lighting pipeline          |
| `composite`            | `DeferredLighting`     | Post-process pipeline      |
| `final`                | `DeferredLighting`     | Final compose              |

## Shader language

| Aspect            | Iris (Legacy)         | Aperture (Planned)         |
|-------------------|-----------------------|----------------------------|
| Primary language  | GLSL (OpenGL dialect) | Slang → SPIR-V / WGSL     |
| Pipeline model    | Implicit (per-stage)  | Explicit pipeline desc     |
| Uniform binding   | OpenGL locations      | Descriptor sets / bindings |
| Entry point       | Fixed names           | Declared in pipeline desc  |

## Integration boundary

| Concern               | Iris path                              | Aperture path                          |
|------------------------|----------------------------------------|----------------------------------------|
| Renderer ownership     | Rust (via `RenderPipelineProvider`)     | Rust (via `RenderPipelineProvider`)     |
| Shader-pack parsing    | Rust-side GLSL ingestion via naga      | Rust-side Slang compilation            |
| Frame input collection | Java hooks → binary buffer → JNI       | Java hooks → binary buffer → JNI       |
| GPU scheduling         | Single queue, Rust-driven              | Single queue, Rust-driven              |
| Pack hot-reload        | File watcher on Rust side              | Pipeline rebuild on Rust side          |

## Migration path

1. **Today**: Iris-style GLSL packs mapped through `IrisPassStage::to_ferridian_pass()`
2. **Transition**: Validate GLSL → WGSL translation via naga for core shaders
3. **Future**: Accept Slang packs via Aperture pipeline descriptors, fall back to GLSL path for legacy packs

## Status

- [x] `IrisPassStage` enum covers all classic stages
- [x] `ShaderPackAdapter` supports both `IrisLegacy` and `ApertureExplicit` models
- [x] `RenderPipelineProvider` abstracts standalone / Fabric-Iris / Aperture entry paths
- [ ] End-to-end GLSL shader pack loading through the Iris adapter
- [ ] Slang compilation path for Aperture packs
