# The packc CLI

```
packc build <pack_dir> [--out <dir>] [--slangc <path>]
packc validate <pack_dir>
packc serve <pack_dir> [--out <dir>] [--slangc <path>]
```

`build` compiles every pass with `slangc` (Slang → SPIR-V), reflects entry
points and descriptor bindings with rspirv, and writes an artifact directory:

```
build/
├── manifest.toml      # echo of the validated manifest
├── shadows.spv        # one module per pass
├── composite.spv
├── reflection.toml    # entry points + bindings per pass
└── generation         # monotonic publish counter (written by serve)
```

`serve` is hot reload: it builds once, then watches the pack directory
(200 ms poll) and rebuilds on any change. Every *successful* rebuild
atomically bumps `<out>/generation`; a running engine polls that one file and
reloads when the number changes, so it never sees a half-written artifact.
Compile errors are printed and the last good generation stays published —
fix the file and save again.

`slangc` resolution order: `--slangc`, `$FERRIDIAN_SLANGC`, `$PATH`. In the
repo devcontainer it is preinstalled and pinned.
