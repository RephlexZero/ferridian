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
- Exactly one pass writes `swapchain`; no pass reads it or writes
  `game_color`/`game_depth`.
- Executable today: graphics passes with one output, combined image samplers
  on set 0, `vs_main`/`fs_main` entry points. Compute passes, uniform
  buffers, and multiple render targets are load-time errors until the
  executor grows them.

TODO: resource formats/sizes, settings/options surface, capability
requirements per pass — designed during M3.
