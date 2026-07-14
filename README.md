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

## Status: M1 safety net complete, M2 started

M0 (walking skeleton) and M1 (safety net) are done: golden-image harness with
blessed lavapipe baselines (`mise run golden-bless`), upstream-watch doing
jar→classfile-signature extraction live-tested against real 26.2/26.3 jars
(it caught Mojang's `blaze3d`→`renderpearl` move in the 26.3 snapshots),
four fuzz targets (which evicted panicky rspirv from the untrusted path),
and a nightly Miri/ASan/sync-validation workflow. M2 has its first real
interception: the layer loads under the real Vulkan loader beneath VVL and
counts `vkCmdBeginRenderPass`, proven by a GPU-gated end-to-end test.

Follow-ups tracked toward M2+:

- [ ] vk-layer: per-object dispatch tables, then injection into a real game process (M2)
- [ ] Engine composite over the intercepted frame; pass detection via the contract (M2)
- [ ] packc `serve` hot reload (M3)
- [ ] Verify shim Gradle/Loom versions against live Maven; wire `shim-build` into CI
- [ ] Switch CI gpu job to the immutable GHCR image tag once `container.yml` has pushed one
- [ ] cargo-vet audit seed + release attestations; cargo-semver-checks on publish
- [ ] Renovate/Dependabot for Fabric Loader/API + Vulkan-Headers/VVL bumps
- [ ] GPU-assisted validation in nightly once verified against lavapipe
- [ ] MoltenVK on GitHub macOS runners — real render or capability-lint only? (open question)
- [ ] Photon port permission outreach (human task, before any port work)
