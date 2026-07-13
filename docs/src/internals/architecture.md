# Architecture

Authoritative plan: [`overhaul.md`](https://github.com/ferridian/ferridian/blob/main/overhaul.md)
at the repo root. Summary of the seams:

| Piece | Crate | Role |
|---|---|---|
| Interception | `crates/vk-layer` (cdylib) | Khronos-ABI Vulkan layer; boring by design |
| Runtime | `crates/vk-rt` | ash device/queue/alloc bootstrap, capability tiers |
| Orchestration | `crates/engine` | pass graph + frame logic; no Vulkan init; Miri-able |
| Pack schema | `crates/pack-format` | serde types, zero I/O, fuzzed |
| Pack build | `crates/pack-compiler` + `tools/packc` | Slang→SPIR-V AOT + reflection |
| Java contract | `crates/contract` → `tools/shim-codegen` → `shim/` | single source of truth, Java side generated |
| Safety net | `crates/testkit` | lavapipe boot, VVL errors = test failures |
| Upstream | `tools/upstream-watch` | Mojang manifest poll + signature-inventory diff |

Invariants:

- The layer cdylib stays thin; logic lives in `engine`/`vk-rt` where tests reach it.
- Nothing outside `contract` defines the Java↔Rust wire format.
- No decompiled Mojang source ever enters the repo — derived inventories only.
- Validation errors are failures, everywhere, always.
