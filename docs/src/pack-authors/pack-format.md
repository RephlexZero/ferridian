# Pack Format

The manifest is `pack.toml`. Schema source of truth:
`crates/pack-format` (Rust types with serde).

```toml
[pack]
name = "reference"
version = "0.1.0"
authors = ["you"]

[requirements]
tier = "baseline"        # or "enhanced" (mesh shading + ray query)

[[pass]]
name = "composite"
kind = "graphics"        # or "compute"
shader = "shaders/composite.slang"
inputs = ["game_color"]
outputs = ["swapchain"]
```

Rules enforced at build time:

- Pass names are unique, and use only `[A-Za-z0-9_-]` (they become artifact
  file names).
- Every `input` is a builtin (`game_color`, `game_depth`, `swapchain`) or the
  `output` of an earlier pass.
- Each resource has exactly one writer.
- The pass graph is acyclic.

Wiring rules, enforced when the engine loads the artifact (it re-reflects
every module rather than trusting `reflection.toml`):

- Descriptors bind **by name**: declare each input as
  `[[vk::binding(n, 0)]] Sampler2D <input name>;` — the binding name must
  match a declared `input` of that pass, and every declared `input` must be
  bound (an unused input would put false edges in the pass graph).
- The engine's per-frame camera block may be bound on any free slot as
  `ConstantBuffer<Camera> camera` without declaring it as an input — it is
  engine state, not a graph resource (layout: `crates/contract`, std140).
- Exactly one pass writes `swapchain`; no pass reads it or writes
  `game_color`/`game_depth`.
- A **graphics** pass has exactly one vertex and one fragment entry point
  and writes its single output through the color attachment (a fullscreen
  triangle — see `packs/reference/shaders/fullscreen.slang`).
- A **compute** pass has exactly one compute entry point with a
  `[numthreads(…)]` workgroup size (the engine dispatches
  `ceil(extent / workgroup size)` groups over the frame), binds its single
  output as a `RWTexture2D` named after the declared output, and samples
  inputs with `SampleLevel` (no implicit derivatives in compute). Compute
  cannot write `swapchain` — the game's image has no storage usage; write an
  intermediate and composite it with a graphics pass.
- Still load-time errors until the executor grows them: multiple render
  targets, uniform blocks other than `camera`, storage buffers, and
  descriptor sets other than 0.

TODO: resource formats/sizes, settings/options surface, capability
requirements per pass — designed during M3.
