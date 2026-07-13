# The packc CLI

```
packc build <pack_dir> [--out <dir>] [--slangc <path>]
packc validate <pack_dir>
packc serve <pack_dir>        # hot reload — not implemented yet (M3)
```

`build` compiles every pass with `slangc` (Slang → SPIR-V), reflects entry
points and descriptor bindings with rspirv, and writes an artifact directory:

```
build/
├── manifest.toml      # echo of the validated manifest
├── composite.spv
└── reflection.toml    # entry points + bindings per pass
```

`slangc` resolution order: `--slangc`, `$FERRIDIAN_SLANGC`, `$PATH`. In the
repo devcontainer it is preinstalled and pinned.
