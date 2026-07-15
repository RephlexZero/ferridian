# Project Plan — Rust Vulkan Shader Engine for Minecraft: Java Edition

**Working codename:** ✅ **Ferridian** (decided 2026-07-13 — existing name and `io.ferridian` namespace kept; revisit is cheap later)
**Author:** Jake · **Date:** 2026-07-13 · **Status:** Draft v1 · **M0 complete 2026-07-13** (commits `105e36d`…`140b01b`)

---

## 1. Thesis

Mojang is replacing Java Edition's OpenGL renderer with Vulkan (experimental in 26.2, OpenGL removal expected ~2027). This forcibly obsoletes the entire OptiFine-lineage shader ecosystem. The incumbent successor, **Aperture** (Iris team), dropped backwards compatibility — meaning every shader pack author on earth is rewriting from scratch right now, and the switching cost to an alternative platform is at an all-time low.

We do **not** compete for the "default shader loader" position (Aperture wins that on distribution). We build the **flagship/high-end engine**: a Rust-native, Vulkan-native shader platform whose differentiators are:

1. **Modern GPU techniques as engine primitives** — ReSTIR GI, radiance cascades, ray-query paths on capable hardware, temporal upscaling as infrastructure (Ferridian thesis, redeployed).
2. **A materially better pack-developer experience** — Slang-based packs compiled AOT to SPIR-V with full reflection, an LSP, hot reload, and build-time validation (vs. Aperture's TypeScript-config-over-GLSL, validated at load time).
3. **Robustness as an architecture property** — Vulkan layer interception (stable Khronos ABI), a single generated Java↔Rust contract, and automated upstream breakage detection against Mojang's now-unobfuscated jars.

### Why now (the window)

- Minecraft is unobfuscated as of 26.1 (Dec 2025) → automated version-diff tooling is finally viable.
- Native Vulkan shipped experimental in 26.2 (June 2026); renderer internals still churning → everyone's engine is young.
- Aperture's Vulkan rewrite only began ~Jan 2026; private beta reached shader devs in May 2026. Their core is ~6 months old.
- Iris will be discontinued once Vulkan is default. Pack authors are mid-rewrite **now**. Window closes when major packs ship Aperture versions (estimate: 6–12 months).

---

## 2. Competitive landscape (as of July 2026)

| Player | Position | Notes |
|---|---|---|
| **Mojang Vulkan renderer** | Baseline | Experimental in 26.2; Vibrant Visuals will raise vanilla visual floor. Internals unstable until OpenGL removal (~2027). |
| **Sodium** | Perf incumbent | 0.9.x shipped with early Vulkan support (June 2026). Moat = ecosystem compat, not renderer. Don't compete. |
| **Aperture** | Loader incumbent-successor | TypeScript pipeline config (`pack.ts`), no legacy pack support, compute/SSBO/CSM support, GLSL underneath. Private beta with pack authors. Wins "default." |
| **VulkanMod** | Dead end | Being dropped by modpacks in favour of native Vulkan. |
| **Us** | Flagship tier | Next-gen GI + best-in-class pack toolchain + verifiable safety story. |

**Aperture's exposed flanks:** voluntary moat reset (no legacy packs), young Vulkan core, small core team, TS-runtime-in-loop complexity, loader-not-frontier positioning, macOS/MoltenVK portability risk.

**Our hard requirements to matter:** a stunning reference pack (ours) + at least one recognisable ported pack + credible docs, before Aperture's 1.0 + big-pack releases.

---

## 3. Architecture

### 3.1 Interception strategy

- **Primary seam:** Vulkan layer (implicit/explicit) in Rust (`ash`). Stable, Khronos-versioned ABI; survives game updates by construction. ReShade-style.
- **Semantic seam:** thin, loader-agnostic shim publishing pass metadata (which pipeline = terrain/entities/etc.) and per-frame camera state over a versioned contract. Regenerated per MC version — mechanically, from unobfuscated sources. Fabric is the first entrypoint; NeoForge is a planned sibling module, not a rearchitecture (§3.4).
- **Discipline:** the layer cdylib stays *boring*. All logic lives in `engine`/`vk-rt` behind a clean seam so Miri and unit tests reach maximal code.

### 3.2 Pack format

- **Language:** Slang (primary), GLSL 4.6 (secondary), both → SPIR-V AOT via `pack-compiler`.
- **Pipeline definition:** declarative pass graph (TOML/RON manifest), compiled + validated at pack build time by `packc`. Reflection-driven binding; no stringly-typed uniforms.
- **Runtime:** engine consumes a compiled, signed pack artifact. Load-time work ≈ zero; errors surface at author time.
- **Compat posture:** no OptiFine/Iris legacy support (same call Aperture made). Optional future: Aperture-pack import tool if their format stabilises and licensing allows.

### 3.3 Portability floor

- Target Vulkan 1.3 core + `VK_KHR_portability_subset` compliance for the macOS/MoltenVK path (Mojang's stated reason for the migration is keeping macOS alive → portability subset is the de facto conformance target).
- Capability tiers: `baseline` (portability-clean), `enhanced` (mesh shading, ray query where present). Engine validates pack requirements against a committed device profile set (incl. MoltenVK profile) at pack build time.

### 3.4 Mod-loader targets

The layer is the real engine; the shim's only job is the semantic seam (§3.1)
plus, as of the camera transport, publishing per-frame state through a
generated JNI bridge — neither depends on Fabric. That split is now real in
the repo, not just aspirational: `shim/core` holds the generated contract
(`GamePassKind`, `FerridianContract`, `CameraUniforms`) and the generated
`NativeBridge` (one JNI setter/getter pair per contract field, so the wire
order can't drift from `ferridian-contract`'s own field table); it has zero
Fabric or Loom dependency. `shim/fabric` depends on `core`, adds only what
Fabric needs — the Loom toolchain, the Minecraft/Loader coordinates,
`FerridianShim implements ClientModInitializer` — and, since Minecraft 26.x
ships unobfuscated (Loom never wires a `remapJar` here, so its usual
`include()` jar-in-jar mechanism has nothing to attach to), embeds `core`'s
compiled classes directly into its own `jar` task output instead.

Supporting a second loader is adding a sibling module, not rearchitecting:
a planned `shim/neoforge` depends on `shim/core` exactly like `fabric` does,
swaps `net.fabricmc.fabric-loom` for NeoForge's **ModDevGradle** plugin, and
replaces `ClientModInitializer` with NeoForge's `@Mod`-annotated entrypoint
(registered on the mod-bus `FMLClientSetupEvent` instead of Fabric's
client-init callback). Because the shim still does no mixins or
loader-specific game hooks — it's a thin publisher over the contract, not a
gameplay mod — there's no Fabric-specific logic to port; `core`'s generated
sources and native bridge are consumed identically by both. The dependency
NeoForge's toolchain needs (`https://maven.neoforged.net/releases/`) is
reachable from CI/devcontainer today, so the module is a matter of writing
it and wiring a `shim` CI job leg for it (mirroring the existing Fabric
`shim` job), not a new unblock. Not yet started — tracked as a follow-up
(README) rather than scaffolded, since it should land with its own build
verification pass rather than as a drive-by.

---

## 4. Repository

### 4.1 Layout

```
.
├── mise.toml                    # tools + tasks: the one true entrypoint
├── rust-toolchain.toml
├── Cargo.toml                   # workspace.lints, workspace.dependencies, profiles
├── crates/
│   ├── engine/                  # render graph, frame orchestration (no vulkan init)
│   ├── vk-rt/                   # device/queue/sync/alloc runtime (ash, gpu-allocator)
│   ├── vk-layer/                # cdylib interception shell — thin as physics allows
│   ├── pack-format/             # schema types only, serde, zero I/O (Miri-able)
│   ├── pack-compiler/           # slang→SPIR-V, reflection (rspirv), validation
│   ├── contract/                # single source of truth: Java↔Rust ABI + pass metadata
│   └── testkit/                 # lavapipe bootstrap, VVL-errors-are-panics, goldens
├── tools/
│   ├── packc/                   # shader-author CLI: build / validate / serve (hot reload)
│   ├── shim-codegen/            # contract → generated Java
│   └── upstream-watch/          # MC manifest poll → decompile → signature diff
├── shim/                        # Gradle multi-project; ~90% generated
│   ├── core/                    #   loader-agnostic: generated contract + native bridge
│   ├── fabric/                  #   Fabric entrypoint (Loom); depends on core
│   └── neoforge/                #   planned — see §3.4
├── packs/reference/             # the marquee pack; dogfoods packc from day one
├── goldens/                     # git-lfs image baselines
├── ci/mesa.Dockerfile           # pinned lavapipe; also the devcontainer base image
├── .devcontainer/
├── .github/workflows/           # ci / goldens / upstream / fuzz / release
└── docs/                        # mdbook — pack-author docs are product surface
```

### 4.2 Toolchain decisions

| Decision | Choice | Rationale |
|---|---|---|
| Task runner / tool pinning | ✅ **mise** (`mise.toml`) | Pins JDK, lefthook, taplo, slang, etc. cross-OS; task runner included. No crate named `xtask` — logic-bearing tasks are real CLI crates in `tools/`. |
| Rust pinning | ✅ `rust-toolchain.toml` (1.96.0) | With mise, covers ~90% of hermeticity without containers on Windows. |
| Git hooks | ✅ **lefthook**, <5 s budget | fmt check, typos, taplo, commit-lint only. Advisory UX; CI is the enforcement layer. Wired by `mise run setup`. |
| Test runner | ✅ **cargo-nextest** | Parallelism, retries for rare lavapipe flakes, JUnit output. |
| Snapshots | ✅ **insta** | SPIR-V reflection dumps + codegen output as reviewable snapshots. |
| Supply chain | ✅ cargo-deny + **cargo-vet** (seeded `supply-chain/`, every dependency audited or exempted) + ⏳ cargo-auditable + GitHub artifact attestations (SLSA) | We ship a cdylib injected next to people's game — verifiable builds are ethics *and* marketing. *(deny + vet wired and green in `gates`; auditable/attestations remain M1+ follow-ups — see README checklist)* |
| Deps/releases | ⏳ Dependabot (cargo/gradle/actions, grouped weekly) ✅; release-plz pending | *(commit-lint hook in place; `.github/dependabot.yml` activates on push)* |
| Deliberately skipped (for now) | Nix, Bazel, cargo-hakari, OSS-Fuzz enrolment | Bloat at this stage; revisit at scale. |

### 4.3 Containers — final position

Exactly **one** image (`ci/mesa.Dockerfile`): pinned Mesa/lavapipe, pushed to GHCR, bumped deliberately with a baseline re-blessing PR. Reason: golden-image tests are only meaningful against a pinned rasteriser.

- **Linux dev (primary):** devcontainer on that same image → dev/CI parity; lavapipe default; mount `/dev/dri` for real-GPU (trivial on Mesa/AMD — no nvidia-toolkit ceremony). Claude Code sessions run sandboxed inside it.
- **Windows dev:** native, no container. Same `mise run *` commands. Its job: MSVC build + real Windows driver behaviour (untestable in a container anyway). **Never** WSL2-for-Vulkan (Dozen is incomplete and will mislead).
- **CI:** container for render jobs; plain runners otherwise.

> ✅ **Status:** `ci/mesa.Dockerfile` + devcontainer implemented and verified locally (Mesa 25.2.8, VVL 1.3.275, slangc 2026.13, lavapipe boots under VVL). `container.yml` pushes the GHCR pin on Dockerfile change; the CI gpu job builds the image inline until the first pushed tag exists.

---

## 5. CI / safety wiring

**Per-PR (fast-fail order):**
1. ✅ Lint: `fmt`, `clippy -D warnings` (workspace.lints), typos, taplo.
2. ✅ Build+test matrix: `ubuntu-latest`, `windows-latest`, `macos-14` (Apple silicon) — nextest. *(workflow in place; first hosted run pending push of the restructure)*
3. ✅ Golden render job (pinned Mesa container): fixtures on lavapipe, **VVL enabled, any validation error = test failure**, perceptual diff vs LFS baselines. *(2026-07-14: testkit harness — offscreen render, PNG readback, DiffPolicy of one quantum/zero differing pixels, failure heatmaps as CI artifacts; first baseline blessed in-container; `mise run golden-bless` is the re-bless flow. Already caught a real bug: slangc output requires `shaderDrawParameters`.)*
4. macOS leg: MoltenVK if runnable on GH macOS VMs; otherwise SwiftShader + static portability-subset capability lint against committed MoltenVK profile. *(Verify MoltenVK-on-runner early — open question.)*
5. ⏳ Gates: cargo-semver-checks (public crates), cargo-deny ✅, cargo-vet.

**Nightly:** ✅ `nightly.yml` (2026-07-14): VVL sync validation on lavapipe; ASan/LSan on `vk-layer`/`vk-rt`/`engine`; Miri on `pack-format`/`engine`/`contract`. All three legs verified in-container before wiring. **GPU-assisted validation joined 2026-07-15**, its own job (not combined with sync validation — the two VVL modes are independently diagnosable): unblocked by giving every `VkShaderModule` the codebase creates exactly one stage, and verified for real (reproduced the exact "Mixed stage shader module not supported" blocker on the old mixed-stage shape, confirmed it gone on the new one, with a cleared shader-validation cache to rule out the known masking gotcha).

**Weekly:** ✅ cargo-fuzz matrix (`fuzz.yml`): pack manifest parser, SPIR-V reflection, contract decoder, classfile parser. *(2026-07-14: fuzzing immediately paid for itself — rspirv 0.12 panics on malformed SPIR-V, so reflection now walks the word stream itself and rspirv is off the untrusted path; crash inputs are committed regression fixtures.)*

**Upstream watch (6 h cron):**
1. ✅ Poll Mojang `version_manifest_v2.json`. *(live-tested against the real manifest; `upstream-watch.yml` opens/comments issues on exit 3)*
2. ✅ New version → `fetch-jar` (sha1-verified) in CI only. **Vineflower turned out unnecessary**: signatures parse straight out of the classfiles (own minimal bounds-checked parser, no JVM/decompiler in the loop) — structurally stronger legal guardrail, and fuzzable.
3. ✅ Extract derived **signature inventory** of tracked render classes/methods; diff vs committed inventory. *(2026-07-14: 26.2 baseline committed — 18 classes across `blaze3d.vulkan`, `blaze3d.systems`, `client.renderer`. Live diff against 26.3-snapshot-3 correctly caught Mojang moving the GPU stack to `com.mojang.renderpearl` — the exact churn class this exists for.)*
4. ⏳ Diff report lands on the tracking issue ✅ (fetch→extract→diff wired into the workflow); Claude Code Action drafting a shim-regen PR gated on goldens = still to come.
5. ✅ **Legal guardrail:** commit only derived inventories (signatures/hashes) — never decompiled Mojang source in a public repo. *(now enforced by construction: nothing in the pipeline can emit source)*
6. ✅ Dependabot watches cargo, the shim's Gradle deps (Fabric Loader/Loom), and workflow actions — grouped weekly PRs judged by the full safety net. Mesa/VVL stay pinned via the container image by design.

---

## 6. v1 shader pack decision

### Requirements for the v1 port
Prove the engine on a real-world pack; give users a familiar look at launch; exercise the full pipeline (deferred/CSM/volumetrics/compute) without drowning in settings-surface; be legally distributable.

### Candidates

| Pack | Technical fit | Licence | Verdict |
|---|---|---|---|
| **Photon** (sixthsurge) | **Excellent** — modern, clean, well-structured codebase (~98% GLSL); semi-realistic gameplay focus; volumetrics, coloured lighting, SSR, TAA/temporal upscaling; already ships for 26.1.2; moderate settings surface | Source-available **custom licence**: modification OK, but anti-competing-fork clause and no selling derivatives without permission → **port requires written permission** | ✅ **Primary pick — with permission** |
| Complementary (EminGT) | Most popular; but huge settings surface, BSL-derived legacy structure | Custom, permission needed; author is presumably in Aperture's private-beta orbit | Partnership target for later, not v1 |
| BSL (Capt Tatsu) | Legacy architecture | Restrictive custom licence | ❌ |
| Bliss (X0nk, Chocapic13-based) | Popular, decent look | Chocapic13 licence permits modification with credit — most permissive of the majors | ✅ **Fallback** if Photon permission fails |
| SEUS / Kappa / Nostalgia | — | Proprietary | ❌ |

### Recommendation

**Tier 0 (weeks 1–8):** our own minimal reference pack (`packs/reference/`) — built *with* `packc` from day one; grows into the marquee next-gen-GI showcase. This is non-negotiable regardless of the port.

**Tier 1 (v1 port): Photon.** Best technical fit by a distance: it's the most modern mainstream codebase (clean deferred structure, temporal upscaling already a feature, active on 26.x), scoped tightly enough to port in bounded time, and its author (sixthsurge, UK) has a collaborative track record (credits exchanged with Emin/DrDesten/Jessie). **Action: approach sixthsurge for written permission — or better, collaboration — before writing a line of the port.** A first-mover pack author choosing us over/alongside Aperture is worth more than the port itself.

**Fallback:** Bliss under the Chocapic13 licence (permission-light), accepting an older code lineage; or a clean-room "Photon-class" feature-set pack of our own (no licence risk, more work, still valuable).

**Explicitly rejected:** porting BSL/Complementary without permission — legal exposure + community goodwill damage in the exact community we need to court.

---

## 7. Milestones

| # | Milestone | Target | Exit criteria | Status |
|---|---|---|---|---|
| M0 | Walking skeleton | +2 wk | Repo scaffold; `mise run ci` green on 3 OSes; testkit boots lavapipe and fails a test on a VVL error; mesa image + devcontainer live | ✅ **2026-07-13** — scaffold per §4.1; VVL-failure gate proven on lavapipe in the container; ci green locally (3-OS matrix defined, first hosted run pending push) |
| M1 | Safety net complete | +6 wk | Golden harness + baselines; upstream-watch opening issues on real snapshots; fuzz targets running | ✅ **2026-07-14** — golden harness + first blessed baseline (in-container); fetch→extract→diff live-tested against real 26.2/26.3 jars and wired into the issue workflow; four fuzz targets (one real rspirv panic found + fixed). Hosted runs of the workflows pending push |
| M2 | Layer + triangle | +10 wk | vk-layer intercepts 26.x snapshot; engine composites over game frame; pass detection on current renderer | ⏳ in-container scope done 2026-07-14: per-object dispatch tables (dispatch-key keyed, destroy-clean, proven with two concurrent instance+device stacks); debug-utils label interception classifying passes against contract anchors; **composite over the intercepted frame** (embedded overlay drawn inside the app's render pass, pixel-verified, only into classified passes) — all beneath VVL with zero messages. **Attachment-tapping coverage widened 2026-07-15**: `vkCreateRenderPass2`/`vkCmdBeginRenderPass2{,KHR}`/`vkCmdEndRenderPass2{,KHR}` classify and composite exactly like the classic path (same `VkRenderPassBeginInfo`); `VK_KHR_dynamic_rendering` (no render-pass/framebuffer object at all) resolves its attachments directly from `VkRenderingInfo` at begin time; `vkCreateImage` now forces `TRANSFER_SRC` onto attachment-usage images the app didn't request it for (checked against `vkGetPhysicalDeviceImageFormatProperties` first), so the compositor's tap-copy works against an unmodified game's own images rather than requiring the app to ask for it. Proving this surfaced and fixed two real bugs: `VK_EXT_debug_utils`'s `Cmd*` commands resolve via `vkGetInstanceProcAddr` not `vkGetDeviceProcAddr` (GDPA-only interception left classification silently dead depending on layer order), and a genuine use-after-free where the layer's own gpu-allocator instance outlived `vkDestroyDevice` via two un-dropped `Arc` clones. ~~Remaining needs a real game: process injection + live anchors from Blaze3D's actual debug groups~~ → **closed 2026-07-15**: booted the real Minecraft 26.2 client headlessly in-container (Xvfb + lavapipe, no `/dev/dri`; `--graphicsBackend VULKAN --vulkanValidation true` to get Mojang's own Vulkan backend, `xdotool` to get past a first-run dialog blocking everything else) and harvested real `VK_EXT_debug_utils` label strings straight from the layer's own interception — classification against a live game now uses these, proven via real per-frame `terrain`/`translucent`/`sky`/`entities`/`gui` counts, not testkit's synthetic anchors (see README follow-ups). **Not yet closed**: the actual visual composite hasn't been confirmed on a real frame — no positive success log exists for `record_over`, `composite_trigger`'s "translucent is last before hand/GUI" assumption was never checked against Mojang's real pass order, and the shim still never captures real camera state, so even a working composite could look deceptively close to vanilla. That's the concrete next M2 task. |
| M3 | Pack pipeline v0 | +16 wk | `packc` builds Slang pack → SPIR-V artifact ✅; hot reload; reference pack renders (shadows + deferred + one volumetric) | ✅ **2026-07-14** — `packc build/validate/serve` end-to-end on `packs/reference` (atomic generations, live-verified edit→rebuild→recover); `engine::pack::PackWatcher` consumes the handshake, loading artifacts as untrusted input. And the pack *renders*: `engine::exec` wires descriptors by name (binding ↔ resource, re-reflected from the SPIR-V, mismatches rejected at load), `vk-rt::PackExecutor` schedules the four passes on device — proven on lavapipe under VVL, synthetic game frames in, lit+fogged pixels out, deterministic across replays. ~~Caveat carried forward: MRT not yet executable.~~ → **closed 2026-07-15**: reflection reads `SV_TargetN` attachment locations from the SPIR-V word stream, the planner holds a graphics pass's outputs against them in manifest order, and the executor builds one attachment/blend-state per target — proven with a compiled-at-test-time two-target pack whose exact per-channel pixel values fail on swapped attachment order, not just a dropped target; the reference golden is untouched (the single-target path is byte-identical through the generalized code). ~~Compute passes not yet executable~~ → **closed 2026-07-15**: reflection reads workgroup sizes and storage-image bindings from the SPIR-V word stream, the planner wires compute passes (single storage-image output, no swapchain writes), the executor dispatches with `GENERAL`↔sampled barriers — and the reference volumetric pass, converted from fragment to compute, reproduces the blessed composite golden byte for byte through the executor *and* the layer (hot reload included). Executor v1 closed out further 2026-07-15: per-pass sampler filters (`filters = { name = "linear" }`, nearest default) reach the right binding, proven with a fixture pack that upscales through both a linear and a nearest binding into separate channels of one frame; and every pack allocation (intermediates, camera buffer, compositor taps) now comes from a shared `gpu-allocator` instance per device instead of one dedicated `vkAllocateMemory` each — the layer's own instance-table construction hit a real segfault answering a `vkGetDeviceProcAddr` query through the device-creation chain's proc-addr resolver on lavapipe+VVL, fixed by special-casing that one query to the already-resolved device layer-info pointer. ~~Placeholder camera model until contract uniforms exist~~ → **closed 2026-07-14**: the contract's `CameraUniforms` block (std140 on both sides, Java mirror generated from the same field table) is wired by name, uploaded by the executor, and proven live on-device — placeholder *values* remain until the shim has a transport. ~~Shim → layer camera transport doesn't exist~~ → **closed 2026-07-15**: the same std140 field table also renders a generated `NativeBridge` (one JNI setter, one getter per field); the layer stores whatever's published in a small process-global (`ferridian-vk-layer::camera_transport`) and the compositor reads through it every frame instead of a hardcoded constant. Proven by a real JVM — `System.load`-ing the actual cdylib and calling the actual generated `NativeBridge.java`, publishing values and reading them straight back through the real JNI ABI, no seam mocked. Real Minecraft-side capture (sun angle, clip planes) feeding it is a separate, still-open item: it needs the same real-game Vulkan layer injection as the M2 remainder. **Layer seam closed 2026-07-14**: with `FERRIDIAN_PACK` armed, the layer taps the world-final classified pass's color+depth attachments and runs the pack over the *intercepted* frame (replacing the embedded overlay), proven on the real loader in-container — what remains for a live game is process injection + real anchors (the M2 remainder). ~~GPU-assisted validation blocked on mixed-stage shader modules~~ → **closed 2026-07-15**: `packc` now compiles every entry point to its own SPIR-V module (`slangc -entry ... -stage ...`, one `VkShaderModule` per stage everywhere — pack artifacts, the layer's embedded overlay, testkit's own render harness), and the nightly `gpu-assisted-validation` job is wired and green (§5) |
| M4 | Photon port alpha | +24 wk | Permission secured; Photon-on-engine parity screenshots vs Iris/OpenGL reference goldens | not started — **permission email to sixthsurge is the next human action** |
| M5 | Public alpha | Aligned to Mojang's OpenGL-removal messaging (~late 2026/early 2027) | Reference pack showcase + Photon port + docs + verifiable release artifacts | — |

Cadence risk: 26.3/26.4 renderer churn will invalidate pass detection repeatedly until OpenGL removal — upstream-watch (M1) exists precisely to make this a 1-day chore, not a surprise.

---

## 8. Risks & mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Aperture ships 1.0 + big packs before our alpha | Medium-high | Don't race the loader; win the flagship segment; recruit one marquee pack author early (Photon approach) |
| Mojang renderer internals churn until 2027 | Certain | Layer-first architecture; upstream-watch; semantic shim regeneratable per version |
| Vibrant Visuals compresses low-end shader demand | Medium | We target the high end explicitly |
| Photon permission declined | Medium | Bliss fallback / clean-room pack; permission ask costs one email |
| MoltenVK not runnable in GH macOS CI | Unknown | Verify in M0; SwiftShader + capability lint fallback |
| Solo-maintainer bus factor / shipped-state pattern | Known | M0–M1 front-load automation so maintenance is cheap; milestone exit criteria are demos, not research |
| Legal: decompiled source handling | Low if disciplined | Derived-inventory-only rule in CI; no Mojang code in repo |

## 9. Open questions

1. Confirm Vulkan layer injection works cleanly with the 26.2 experimental renderer on all three OS loaders (Windows registry / Linux manifest / macOS-MoltenVK path).
2. MoltenVK on GitHub macOS runners — real render or capability-lint only?
3. Slang maturity for the full pack surface vs GLSL-primary at launch — spike in M3. *(early signal good: slangc 2026.13 compiles the reference composite pass with clean rspirv reflection)*
4. Aperture licence & format stability — monitor for a future import tool.
5. ~~Codename.~~ ✅ Resolved: **Ferridian**.