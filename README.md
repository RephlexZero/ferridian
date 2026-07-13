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

## Status: M0 walking skeleton

Done: workspace + tooling + CI wiring; pass-through Vulkan layer
(negotiation + create-call chaining); testkit proving VVL errors fail tests
on lavapipe; packc building the reference pack end-to-end (verified against
slangc 2026.13); contract→Java codegen; upstream-watch polling + inventory
diffing; mesa image + devcontainer.

Follow-ups tracked toward M1+:

- [ ] Real interception/dispatch in `vk-layer` (M2) and pass detection
- [ ] Golden-image render harness + first LFS baselines (M1)
- [ ] upstream-watch: CI-side decompile → signature inventory extraction (M1)
- [ ] Verify shim Gradle/Loom versions against live Maven; wire `shim-build` into CI
- [ ] Switch CI gpu job to the immutable GHCR image tag once `container.yml` has pushed one
- [ ] cargo-vet audit seed + release attestations; cargo-semver-checks on publish
- [ ] Nightly workflow: slow VVL modes, ASan/LSan on vk-layer, Miri on pure crates
- [ ] MoltenVK on GitHub macOS runners — real render or capability-lint only? (open question)
- [ ] Photon port permission outreach (human task, before any port work)
