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

## Status: M2 closed visually — the reference pack composites on-screen over a live Minecraft's frames, fed by real camera state

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
on-device. **The shim → layer camera transport now exists**: the same
std140 field table also drives a generated `NativeBridge` (JNI native
declarations — one setter, one getter per field), the layer receives
published values into a small process-global store
(`ferridian-vk-layer::camera_transport`), and the compositor reads through
it every frame instead of a hardcoded constant. Proven end to end by a real
JVM — not a stand-in — loading the actual cdylib and calling the actual
generated `NativeBridge.java` via `System.load`, publishing values and
reading them straight back. And real Minecraft-side capture now feeds it:
the shim's `CameraPublisher` polls the live game for sun angle (26.2's
environment-attribute probe), world time, and clip planes, and the layer's
compositor consumes them on real frames (see the visible-composite status
below). And the
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

The shim is now a multi-module Gradle build: `shim/core` holds the generated
contract and stays loader-agnostic, `shim/fabric` is the Fabric entrypoint
(depends on `core`, embeds its classes directly into the jar task — 26.x
ships unobfuscated, so Loom's usual remapJar-hooked include() mechanism has
nothing to attach to). A `neoforge` module is the natural next addition
alongside it. Supply-chain: `cargo-vet` is seeded (`supply-chain/`, every
current dependency audited or exempted) and wired into the `gates` CI job
alongside `cargo-deny`.

Most `crates/contract` pass anchors are real now, not `todo/*` placeholders:
harvested live from Minecraft 26.2's actual `VK_EXT_debug_utils` labels by
booting the real client headlessly in-container (no GPU passthrough needed —
Xvfb + lavapipe; recipe below) and grepping the layer's own debug-utils
interception for the strings Blaze3D really pushes. Classification against a
live game is proven this way (real per-frame
`terrain`/`translucent`/`sky`/`entities`/`gui` counts, not testkit's
synthetic anchors); `block_entities`/`particles`/`weather`/`hand` stay
`todo/*` honestly since nothing exercised them in that session.

**The reference pack now visibly composites over real game frames**
(2026-07-15, all three suspects from the previous session resolved). The
root cause of the invisible composite was none of the pass-order guesses:
the volumetric compute pipeline failed `vkCreateComputePipelines` with
`ERROR_VALIDATION_FAILED_EXT` on the game's device, because slangc emits
`StorageImage{Read,Write}WithoutFormat` capabilities for an unformatted
`RWTexture2D` and a real game — unlike testkit — doesn't enable those
optional features (and Mojang's `--vulkanValidation true`, the same flag
that produces the anchor labels, turns validation errors into hard
failures). Fixed by declaring `[format("rgba16f")]` on the storage image.
Finding it exposed a lying success metric: the layer counted "composites"
even when `record_over` bailed on its disabled path — it now returns whether
it actually recorded, and `ferridian_layer_composite_count` /
the `"pack composite recorded"` info line only count real recordings.
Proof is visual and live: a hot-swapped inverted-composite generation flips
the whole world (HUD/hand/clouds still vanilla on top, exactly the designed
seam), the next generation restores it, and a broken generation (probe
shader whose dead-code elimination unbound `game_color`) is rejected at
wire time while the old compositor keeps running. The shim side is real
too: `FerridianShim` now `System.load`s the layer cdylib
(`FERRIDIAN_LAYER_LIB`) and a `CameraPublisher` daemon thread polls
`Minecraft.getInstance()` at tick rate, publishing real sun direction (from
the 26.2 environment-attribute probe), world-clock time, and clip planes
through the generated `NativeBridge` — confirmed flowing in the live run.

The real 26.2 Vulkan frame order (from per-pass attachment tracing, now a
trace-level layer feature): atlas/lightmap upkeep → sky → `Section layers
for opaque` → `Section layers for translucent` → **Clouds → translucent
entity immediate draws** → `Blit render target` → GUI blur post chain →
GUI — every world/UI pass rendering into the *same* 854×480 color
attachment. So translucent is *not* the last world content (clouds and
translucent entities draw over the composite — acceptable, documented on
`composite_trigger`), and Mojang's "blit" is a fullscreen draw into the
same image, not a copy to another target.

Follow-ups tracked toward M2+:

- [ ] **Minecraft 26.2's Vulkan backend uses reversed-Z** (proven live: a
  hot-swapped depth-visualizing generation renders sky ≈ 0.0, near ground
  bright). The pack's `linearize_depth` (`packs/reference/shaders/common.slang`)
  assumes conventional depth, so screen-space shadows and fog are wrong on
  real frames — dark-ambient-only output even in daylight (night is
  legitimately dark: the sun term goes to zero, and the zombie that killed
  the test player agrees). Fix `linearize_depth` for reversed-Z
  (`near*far / (near + d*(far-near))` for finite reversed), and make
  testkit's synthetic game frames reversed-Z too so the goldens stay
  representative — that re-bless is why this didn't land as a drive-by.
- [ ] The reference pack has no ambient/skylight floor beyond `0.25*albedo`,
  so nighttime is near-black; add a lightmap-aware or sky-ambient term when
  real material taps land.
- [ ] Consider layer-side `vkCreateDevice` feature patching
  (`shaderStorageImage{Read,Write}WithoutFormat` when the physical device
  supports them) as defense-in-depth for arbitrary packs' unformatted
  storage images; today `packc`-built packs should declare formats instead.
- [ ] `shim/neoforge`: a sibling module depending on `shim/core`, via NeoForge's ModDevGradle (plan in overhaul.md §3.4; NeoForge maven is already reachable from CI/devcontainer)
- [ ] Switch CI gpu job to the immutable GHCR image tag once `container.yml` has pushed one

Real-game harness recipe (in-container, no root): `apt-get download` +
`dpkg -x` Xvfb, xdotool, x11-apps (xwd), x11-xkb-utils and xkb-data plus the
handful of libs `ldd` reports missing; Ubuntu's Xvfb hardcodes
`/usr/bin/xkbcomp` and `/var/lib/xkb/` (no root, no user namespaces in the
sandbox), so binary-patch those two strings to same-length `/tmp` paths and
point a wrapper script at the extracted xkbcomp. Run Xvfb with `-xkbdir` at
the extracted keymaps, then `gradle -p shim :fabric:runClient` with
`DISPLAY`, `VK_ICD_FILENAMES=…/lvp_icd.json`, `VK_ADD_LAYER_PATH` +
`VK_INSTANCE_LAYERS` at a dir holding the built cdylib + manifest,
`FERRIDIAN_PACK=packs/reference/build`, `FERRIDIAN_LAYER_LIB` at the same
cdylib (the shim's `System.load` — same .so the loader maps, one set of
process globals), and `FERRIDIAN_LOG` (`ferridian_vk_layer=trace` for the
per-pass attachment map). Loom already passes `--graphicsBackend VULKAN
--vulkanValidation true --quickPlaySingleplayer "New World"` — quickPlay
only *joins* a save of that exact name ("New World" is committed history
from the first session's xdotool run through the create-world screen).
Hot reload against the armed dir: `packc serve packs/reference` — note a
restarted serve renumbers from generation 1, which the in-game watcher has
already consumed, so touch a shader once to publish generation 2 before
expecting a swap. Screenshots: `xwd -root` (a tiny xwd→png converter is
enough; game F2 also works via `xdotool key`).
- [ ] Release attestations; cargo-semver-checks on publish (still blocked: every workspace crate is `publish = false`)
- [ ] MoltenVK on GitHub macOS runners — real render or capability-lint only? (open question)
- [ ] Photon port permission outreach (human task, before any port work)
