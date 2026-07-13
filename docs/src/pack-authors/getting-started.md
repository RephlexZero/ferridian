# Getting Started

> **Status:** pre-alpha. The toolchain below works today; the engine that
> plays your pack under Minecraft is under construction (M2/M3).

A Ferridian pack is a directory:

```
mypack/
├── pack.toml          # manifest: metadata, requirements, pass graph
└── shaders/
    └── composite.slang
```

Build and validate it:

```bash
packc validate mypack/
packc build mypack/            # -> mypack/build/ (SPIR-V + reflection)
```

Every error a pack can have — unknown pass inputs, missing shaders, shader
compile errors, requirement mismatches — surfaces here, at author time.

The canonical example is
[`packs/reference`](https://github.com/ferridian/ferridian/tree/main/packs/reference)
in the repo, which the engine team dogfoods daily.
