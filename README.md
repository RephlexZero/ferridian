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
| `crates/pack-format` | pack manifest schema + SPIR-V reflection — pure, fuzzed |
| `crates/pack-compiler` | Slang→SPIR-V compilation, artifact writing |
| `crates/contract` | single source of truth: Java↔Rust ABI + pass metadata |
| `crates/testkit` | lavapipe bootstrap; **any validation error fails the test** |
| `tools/packc` | pack-author CLI: build / validate / serve |
| `tools/shim-codegen` | contract → generated Java (`shim/core/src/main/generated`) |
| `tools/upstream-watch` | Mojang manifest poll + signature-inventory diff |
| `shim/` | Mod-loader shells (~90% generated): `core` loader-agnostic, `fabric` the Fabric entrypoint |
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

## Status: M3 done, Executor v1 complete — the layer runs loaded packs over intercepted frames

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
device under validation. And loaded packs now **execute**: `engine::exec`
re-reflects each module and wires descriptors by name (shader binding
`game_depth` ↔ manifest resource `game_depth` — mismatches are load-time
errors, not runtime surprises), and `vk-rt`'s `PackExecutor` instantiates the
plan (intermediate images, descriptor sets, one fullscreen pipeline per pass)
and replays the schedule writers-before-readers. The four-pass reference pack
— screen-space shadows → deferred lighting → volumetric fog → composite —
runs end to end on lavapipe under validation: synthetic game frames in,
correctly lit and fogged pixels out, deterministic across replays — and the
composite is pinned as a golden image, so any unintended shift in any pass
fails the build. Packs mix pipeline kinds now: the volumetric pass is a
**compute dispatch** (reflection reads the workgroup size and storage-image
bindings straight from the SPIR-V word stream; the executor barriers the
output between `GENERAL` and sampled around each dispatch), and converting
it from fragment to compute reproduced the blessed golden byte for byte.
**Executor v1 is now complete**: graphics passes can write multiple render
targets (reflection reads `SV_TargetN`'s attachment locations, the planner
holds them against the manifest's declared outputs in order, the executor
builds one attachment/blend-state per target), each input samples through
a per-pass declared filter (`filters = { name = "linear" }`, nearest by
default — the only mode legal on every depth format), and every pack
allocation — intermediates, the camera buffer, compositor taps — comes from
a shared `gpu-allocator` instance per device rather than one dedicated
`vkAllocateMemory` per resource. Packs
now read per-frame state through the contract's `camera` uniform block
(std140, one field table drives the Rust encoder and the generated Java
mirror), and the executor owns the buffer — updates are proven live
on-device, with the placeholder values uploaded until the shim publishes
real ones. And the
layer↔executor seam is closed: with a pack artifact armed (`FERRIDIAN_PACK`),
the layer tracks the app's views/framebuffers/render passes, and at the end
of the world-final classified pass taps its color+depth attachments, runs
the pack over them (`vk-rt::PackCompositor`), and writes the composite back
into the app's own color attachment — later passes (hand, GUI) draw on top
untouched, the embedded overlay is replaced outright, and attachment
turnover (resize-shaped) invalidates and rebuilds cleanly. Proven end to end
on the real loader: the reference pack lights and fogs an intercepted
game-shaped frame, non-trigger and Unknown passes come back byte-identical.
Hot reload reaches through the layer too: a `PackWatcher` over the armed
directory is polled at the composite trigger, and a newly published
generation swaps compositors between frames (proven live — an
inverted-composite generation 2 flips the very next frame; a corrupt
generation 3 leaves it running). The Fabric shim builds against live Maven
(MC 26.2, loader 0.19.3, Loom 1.17.14, JDK 25) and CI rejects
contract/codegen drift.

The layer's attachment coverage now reaches beyond the classic render-pass
path: `vkCreateRenderPass2`/`vkCmdBeginRenderPass2{,KHR}`/
`vkCmdEndRenderPass2{,KHR}` classify and composite exactly like
`vkCreateRenderPass`/`vkCmdBeginRenderPass`/`vkCmdEndRenderPass` (same
`VkRenderPassBeginInfo`, same code); `VK_KHR_dynamic_rendering`
(`vkCmdBeginRendering`/`vkCmdEndRendering`) has no render-pass/framebuffer
object at all, so its attachments are resolved directly from
`VkRenderingInfo` at begin time and carried on the active pass (the embedded
overlay skeleton, which builds pipelines against a `VkRenderPass`, simply
stays out of these passes — a real pack's compositor has no such
limitation). And `vkCreateImage` now forces `TRANSFER_SRC` onto
attachment-usage images the app didn't request it for (checked against
`vkGetPhysicalDeviceImageFormatProperties` before patching), so the
compositor's tap-copy works against an unmodified game's own images.
Proving this real-loader coverage surfaced two genuine bugs along the way:
`VK_EXT_debug_utils`'s `Cmd*` label commands are spec-classified as
instance-level despite taking a command buffer, so GDPA-only interception
left classification silently dead whenever this layer sits above
`VK_LAYER_KHRONOS_validation` in the enabled-layers order; and the layer's
own gpu-allocator instance was held by two `Arc` clones that only dropped
*after* `vkDestroyDevice` forwarded to the real device — a use-after-free,
fixed by explicitly dropping both first (the same `ManuallyDrop` discipline
`VkRuntime` already uses).

GPU-assisted validation is unblocked and wired into nightly: every
`VkShaderModule` the codebase creates — pack artifacts, the layer's embedded
overlay, testkit's own render harness — is now single-stage (`packc`
compiles each entry point to its own SPIR-V module via `slangc -entry
... -stage ...`, discovered from a never-shipped combined probe compile),
since a module mixing vertex and fragment entry points is exactly the shape
VVL's GPU-assisted validation refuses to instrument ("Mixed stage shader
module not supported").

Follow-ups tracked toward M2+:

- [ ] vk-layer: injection into a real game process; anchors from Blaze3D's real debug groups (M2)
- [ ] Shim → layer transport for per-frame camera state (the compositor uploads `CameraUniforms::placeholder()` until then)
- [ ] Switch CI gpu job to the immutable GHCR image tag once `container.yml` has pushed one
- [ ] cargo-vet audit seed + release attestations; cargo-semver-checks on publish
- [ ] MoltenVK on GitHub macOS runners — real render or capability-lint only? (open question)
- [ ] Photon port permission outreach (human task, before any port work)
