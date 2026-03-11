# Backend-Specific Capability Tracking

This document tracks which planned Ferridian features depend on backend-specific
GPU capabilities so portability regressions remain explicit.

## Feature → Capability Matrix

| Feature                  | Required Capability         | wgpu Support Status       | Fallback                        |
|--------------------------|-----------------------------|---------------------------|---------------------------------|
| Indirect draw            | `INDIRECT_FIRST_INSTANCE`   | Stable (most backends)    | Direct draw calls               |
| Compute visibility cull  | Compute shaders             | Stable                    | CPU-side frustum culling        |
| Texture arrays           | `max_texture_array_layers`  | Stable (≥256 on desktop)  | Texture atlas                   |
| SSAO (half-res)          | Render to half-res targets  | Stable                    | Full resolution                 |
| Hi-Z SSR                 | Mip generation in compute   | Stable                    | Linear ray march                |
| Shadow map cascades      | Depth textures              | Stable                    | None (required)                 |
| Voxel cone tracing       | 3D textures + compute       | Stable                    | None (feature is optional)      |
| Path tracing             | Ray query extension         | Experimental (wgpu)       | Voxel cone tracing              |
| Mesh shaders             | `MESH_SHADER` (Vulkan ext)  | Not in wgpu yet           | Traditional vertex pipeline     |
| Bindless textures        | `PARTIALLY_BOUND_BINDING`   | Experimental              | Texture arrays                  |
| Multi-queue async comp.  | Multiple queue families      | Not in wgpu yet           | Single-queue scheduling         |
| 16-bit storage           | `SHADER_F16`                | Experimental              | 32-bit storage                  |

## Current Strategy

- **Vulkan-first**: All features target Vulkan as primary backend via wgpu.
- **No bindless blocking**: Texture arrays + single queue until wgpu exposes stable bindless.
- **Experimental opt-in**: `BackendConfig.experimental = true` enables experimental wgpu features.
- **Graceful degradation**: Features check `BackendCapabilities.features` and disable themselves when unsupported.
