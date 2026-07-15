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
on-device. **The shim → layer camera transport now exists**: the same
std140 field table also drives a generated `NativeBridge` (JNI native
declarations — one setter, one getter per field), the layer receives
published values into a small process-global store
(`ferridian-vk-layer::camera_transport`), and the compositor reads through
it every frame instead of a hardcoded constant. Proven end to end by a real
JVM — not a stand-in — loading the actual cdylib and calling the actual
generated `NativeBridge.java` via `System.load`, publishing values and
reading them straight back. What still feeds it the placeholder is real
Minecraft-side capture (sun angle, clip planes) — like the rest of the
shim's game-facing code, that waits on real Vulkan layer injection into a
live game process (the M2 remainder). And the
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
Xvfb + lavapipe; see the follow-up below for the exact recipe) and grepping
the layer's own debug-utils interception for the strings Blaze3D really
pushes. Classification against a live game is proven this way (real
per-frame `terrain`/`translucent`/`sky`/`entities`/`gui` counts, not
testkit's synthetic anchors); `block_entities`/`particles`/`weather`/`hand`
stay `todo/*` honestly since nothing exercised them in that session. The
reference pack loads and wires against the real device with no errors — but
the actual visual composite has **not** been confirmed on screen yet (next
follow-up).

Follow-ups tracked toward M2+:

- [ ] **Get the reference pack actually visible on a real frame.** Anchors/classification are proven against a live game (above); the composite itself isn't. Suspects, roughly in order to check: (1) no positive log confirms `PackCompositor::record_over` actually ran — `end_render_pass_common` in `crates/vk-layer/src/lib.rs` only logs on the *failure* path (`"trigger pass not resolvable to attachments"`), so add a success-path trace too; (2) `composite_trigger` (`crates/engine/src/frame.rs`) assumes `Translucent` is the last world pass before hand/GUI, carried over from the classic OpenGL pipeline order — never checked against Mojang's real Vulkan-mode pass ordering, which may differ; (3) the shadows/deferred passes still read `CameraUniforms::placeholder()`, not real captured camera state (`FerridianShim.onInitializeClient()` in `shim/fabric/src/main/java/io/ferridian/shim/FerridianShim.java` still just logs a line — no `System.load`, no `NativeBridge.publishCamera` call, ever), so even a working composite could look deceptively close to vanilla. To re-run the real-game harness: extract Xvfb (`apt-get install --download-only` + `dpkg -x`, no root needed) and set up a `VK_LAYER_FERRIDIAN_overlay` dir with the built cdylib + manifest; launch via `gradle -p shim :fabric:runClient` with `DISPLAY` pointed at the Xvfb display, `VK_ICD_FILENAMES` at `lvp_icd.json`, `VK_ADD_LAYER_PATH`/`VK_INSTANCE_LAYERS` for the layer, `FERRIDIAN_PACK` at `packs/reference/build`; Loom `programArgs` need `--graphicsBackend VULKAN --vulkanValidation true` (both undocumented, found via the client jar's own bytecode) to get Mojang's real Vulkan backend instead of the OpenGL default; a first-run "Continue" screen blocks everything including `--quickPlaySingleplayer` until dismissed, so extract `xdotool` the same no-root way to click through and into a world; `--quickPlaySingleplayer <name>` only *joins* an existing save of that exact name, it doesn't create one.
- [ ] Real Minecraft-side camera capture (sun angle, clip planes) over the now-existing shim → layer transport — see suspect (3) above; this no longer waits on real-game injection (that part's proven), it's just unwritten
- [ ] `shim/neoforge`: a sibling module depending on `shim/core`, via NeoForge's ModDevGradle (plan in overhaul.md §3.4; NeoForge maven is already reachable from CI/devcontainer)
- [ ] Switch CI gpu job to the immutable GHCR image tag once `container.yml` has pushed one
- [ ] Release attestations; cargo-semver-checks on publish (still blocked: every workspace crate is `publish = false`)
- [ ] MoltenVK on GitHub macOS runners — real render or capability-lint only? (open question)
- [ ] Photon port permission outreach (human task, before any port work)
