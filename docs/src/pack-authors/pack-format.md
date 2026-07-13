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

- Pass names are unique.
- Every `input` is a builtin (`game_color`, `game_depth`, `swapchain`) or the
  `output` of an earlier pass.
- Each resource has exactly one writer.
- The pass graph is acyclic.

TODO: resource formats/sizes, settings/options surface, capability
requirements per pass — designed during M3.
