# WGPU Compatibility Policy

Ferridian stays Vulkan-first in intent, but mainline code should remain portable across the `wgpu` backends that matter for desktop development and shipping.

## Mainline policy

- Mainline renderer code must default to stable `wgpu` APIs only.
- Mainline device creation must keep `ExperimentalFeatures::disabled()` unless a deliberate opt-in path is being exercised.
- Stable `wgpu::Features` may be enabled in mainline only when there is a clear renderer-level need, a documented fallback story, and no requirement to fork ownership into the Java side.
- Backend-facing code should stay concentrated in `ferridian-core` and future renderer internals rather than leaking through the JNI or Minecraft-facing crates.

## Allowed in mainline

- Stable WGSL shader modules and pipeline creation.
- Stable render, depth, texture-array, compute, and indirect-draw features that `wgpu` exposes without experimental toggles.
- Runtime capability probing that selects between stable implementations while keeping a conservative default path available.
- Development tooling such as shader hot reload, reflection, profiling hooks, and pipeline caches as long as they do not require experimental `wgpu` features to run.

## Must stay behind opt-in flags

- Anything requiring `wgpu` experimental features.
- Backend-specific techniques with no portable fallback, including work that depends on mesh shaders, ray queries, or similarly uneven capability coverage.
- Research paths that materially change shader authoring or pipeline ownership assumptions before they are validated in a standalone slice.
- High-risk performance experiments that need unstable APIs, nightly-only Rust, or per-backend code paths that have not yet been isolated.

Opt-in work should live behind explicit Cargo features or isolated example binaries so the default workspace path remains conservative.

## Upgrade discipline

- `wgpu` stays pinned from `[workspace.dependencies]` and upgrades should be intentional, not opportunistic.
- Each `wgpu` version bump should include a compatibility pass over `ferridian-core`, the standalone example, and any JNI-facing surface bootstrap code.
- If an upgrade forces backend churn, isolate the adaptation behind renderer internals before expanding feature work.

## Current status

Today the core renderer requests no stable feature flags and explicitly disables `wgpu` experimental features. New work should preserve that baseline unless a plan item explicitly introduces a gated exception.