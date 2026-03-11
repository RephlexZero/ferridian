we can implement this # GLSL → Aperture Slang Migration Validation

This document validates the migration story from legacy GLSL shader packs
(used with Iris/OptiFine) to Aperture's Slang-based, pipeline-explicit model.

## Migration Layers

Ferridian treats GLSL and Aperture/Slang as related systems, not unrelated ones.
The migration path flows through three layers:

### Layer 1: GLSL Ingestion (current)

Legacy GLSL shader packs are loaded through naga's GLSL front-end:

1. GLSL source → naga `Module` (via `naga::front::glsl`)
2. Validate against naga's IR validator
3. Map Iris pass stages (`shadow`, `gbuffers_*`, `deferred`, `composite`, `final`)
   onto Ferridian's `RenderPassKind` using `IrisPassStage::to_ferridian_pass()`

This path is implemented in `ferridian-shader` (`validate_glsl`, `load_workspace_glsl`)
and `ferridian-minecraft` (`IrisPassStage`).

### Layer 2: WGSL as Internal Intermediate

All Ferridian-authored shaders use WGSL via the `ShaderComposer`:

- `#import` for modular composition
- `#ifdef`/`#ifndef` for feature toggles per permutation
- naga IR validation ensures correctness
- `ShaderPermutationCache` avoids redundant compilation

When Aperture stabilizes, its Slang output can target WGSL or SPIR-V. Either way,
the naga IR validator and bind group layout checker (`validate_binding_layout`)
ensure correctness.

### Layer 3: Aperture/Slang Ingest (future)

When Aperture publishes a stable shader format:

1. Slang source → SPIR-V or WGSL (via Slang compiler or naga)
2. Feed into the same `ShaderComposer` / `ShaderPermutationCache` pipeline
3. Aperture's pipeline-explicit model maps directly to `RenderPipelineProvider`
   using `RenderEntryPath::Aperture`

The `ShaderPackAdapter` already distinguishes `IrisLegacy` vs `ApertureExplicit`
models, so both can coexist.

## Compatibility Matrix

| Feature               | GLSL (Iris)    | WGSL (Ferridian) | Slang (Aperture) |
|-----------------------|----------------|-------------------|-------------------|
| Pass structure        | Fixed stages   | Render graph      | Pipeline-explicit |
| Shader language       | GLSL 150+      | WGSL              | Slang → SPIR-V   |
| Material model        | LabPBR         | PBR/LabPBR        | PBR (Aperture)    |
| Composition           | `#include`     | `#import`/`#ifdef`| Slang modules     |
| Validation            | Driver-side    | naga              | Slang compiler    |
| Binding layout check  | None           | `validate_binding_layout` | Slang reflection |

## Validation Status

- [x] GLSL ingestion produces valid naga modules
- [x] Iris pass stages map to Ferridian render passes
- [x] ShaderComposer handles imports and conditional compilation
- [x] Binding layout validator catches group/binding drift
- [x] ShaderPackAdapter distinguishes IrisLegacy from ApertureExplicit
- [x] RenderPipelineProvider supports all three entry paths
- [ ] Aperture Slang format not yet published — monitor for updates

## Conclusion

The migration story is validated: GLSL packs work today through naga ingestion
and Iris stage mapping. Aperture packs will flow through the same intermediate
representation and pipeline abstraction when the format stabilizes. The systems
are not unrelated — they share the naga IR, validation, and permutation cache
infrastructure.
