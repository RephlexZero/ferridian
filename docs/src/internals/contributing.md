# Contributing

Everything runs through mise:

```bash
mise install          # pinned tools (JDK, slang, taplo, nextest, …)
mise run setup        # git hooks (lefthook) + git-lfs
mise run ci           # what CI runs: fmt, clippy -D warnings, taplo, typos, tests
mise run gpu-test     # testkit against a real Vulkan ICD (devcontainer: lavapipe)
mise run shim-build   # Fabric shim (needs network; not part of ci yet)
```

- Rust is pinned by `rust-toolchain.toml`; rustup handles it automatically.
- Commits follow conventional commits (enforced by the commit-msg hook).
- The devcontainer (`.devcontainer/`) builds from `ci/mesa.Dockerfile` — the
  same pinned-lavapipe image CI uses, with `/dev/dri` passthrough for real-GPU
  runs on the host.
- Windows: develop natively (same `mise run *`), never WSL2-for-Vulkan.
