# Ferridian

A **Vulkan-native shader engine for Minecraft: Java Edition**, built for the
post-OpenGL era. Ferridian intercepts the game's own Vulkan renderer through
a Khronos-ABI layer, pairs it with a thin *generated* Fabric shim, and gives
pack authors a Slang→SPIR-V toolchain that catches errors at build time.

We don't compete for the default-loader spot — we build the flagship tier:
next-gen GI as engine primitives, the best pack-developer experience in the
ecosystem, and robustness as an architecture property. The full thesis,
competitive landscape, and milestones live in [`overhaul.md`](overhaul.md);
pack-author docs live in [`docs/`](docs/) (mdbook).

Dual-licensed MIT or Apache-2.0.

## Layout

| Path | What it is |
|---|---|
| `crates/vk-layer` | Vulkan layer cdylib — the interception seam; deliberately boring |
| `crates/vk-rt` | ash device/queue/alloc runtime, capability tiers |
| `crates/engine` | pass graph + frame orchestration (no Vulkan init; Miri-able) |
| `crates/pack-format` | pack manifest schema — pure serde, fuzzed |
| `crates/pack-compiler` | Slang→SPIR-V compilation + rspirv reflection |
| `crates/contract` | single source of truth: Java↔Rust ABI + pass metadata |
| `crates/testkit` | lavapipe bootstrap; **any validation error fails the test** |
| `tools/packc` | pack-author CLI: build / validate / serve |
| `tools/shim-codegen` | contract → generated Java (`shim/src/main/generated`) |
| `tools/upstream-watch` | Mojang manifest poll + signature-inventory diff |
| `shim/` | Fabric mod shell (~90% generated) |
| `packs/reference` | the marquee pack; dogfoods packc from day one |
| `goldens/` | git-lfs image baselines (harness lands M1) |
| `ci/mesa.Dockerfile` | the one container image: pinned lavapipe, CI + devcontainer |

## Workflow

```bash
mise install          # pinned tools (JDK 21, slang, taplo, nextest, …)
mise run setup        # lefthook hooks + git-lfs
mise run ci           # fmt, clippy -D warnings, taplo, typos, tests
mise run gpu-test     # testkit against a Vulkan ICD (devcontainer: lavapipe)
cargo run -p packc -- build packs/reference
```

The devcontainer builds from `ci/mesa.Dockerfile`, so dev and CI share one
pinned rasteriser; `/dev/dri` is passed through for real-GPU runs.

## Status: M2 in-container scope complete; M3 hot reload closed end-to-end

M0 (walking skeleton) and M1 (safety net) are done: golden-image harness with
blessed lavapipe baselines (`mise run golden-bless`), upstream-watch doing
jar→classfile-signature extraction live-tested against real 26.2/26.3 jars
(it caught Mojang's `blaze3d`→`renderpearl` move in the 26.3 snapshots),
four fuzz targets (which evicted panicky rspirv from the untrusted path),
and a nightly Miri/ASan/sync-validation workflow.

M2, everything provable without a live game: the layer runs under the real
Vulkan loader beneath VVL with per-object dispatch tables (proven with two
concurrent instance+device stacks), intercepts debug-utils labels and
classifies render passes against the contract's anchors, and **composites
its own draw inside the game's render pass** — pixel-verified, and only
into classified passes; Unknown is forwarded untouched. M3's hot reload is
closed end to end: `packc serve` watches, rebuilds, and atomically publishes
artifact generations; `engine::pack::PackWatcher` consumes them (exactly-once
per generation, torn-read guarded, artifacts treated as untrusted input),
proven by a testkit round-trip that puts every reloaded module on a real
device under validation. The reference pack is now the four-pass M3 shape —
screen-space shadows → deferred lighting → volumetric fog → composite. The
Fabric shim builds against live Maven (MC 26.2, loader 0.19.3, Loom 1.17.14,
JDK 25) and CI rejects contract/codegen drift.

Follow-ups tracked toward M2+:

- [ ] vk-layer: injection into a real game process; anchors from Blaze3D's real debug groups (M2)
- [ ] Execute reloaded packs: descriptor wiring (binding name ↔ resource name) + pass scheduling on device (M4)
- [ ] Reference pack: swap the placeholder camera model for real contract uniforms; golden-image the four passes
- [ ] Switch CI gpu job to the immutable GHCR image tag once `container.yml` has pushed one
- [ ] cargo-vet audit seed + release attestations; cargo-semver-checks on publish
- [ ] GPU-assisted validation in nightly — verified 2026-07-14 and **blocked**:
      VVL cannot instrument slangc's mixed-stage modules ("Mixed stage shader
      module not supported"), and its on-disk shader_validation_cache masks
      that error on every run after the first (delete
      `~/.cache/{instrumented_shader,shader_validation}_cache*` to reproduce).
      Needs per-stage module emission in pack-compiler first
- [ ] MoltenVK on GitHub macOS runners — real render or capability-lint only? (open question)
- [ ] Photon port permission outreach (human task, before any port work)
