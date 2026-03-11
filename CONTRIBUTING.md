# Contributing

## Baseline expectations

- Use the devcontainer unless you already have Rust stable, JDK 21, and Gradle available locally.
- Install the versioned git hooks with `cargo run -p xtask -- hooks`.
- Run `cargo run -p xtask -- ci` before opening a pull request.

## Scope for early contributions

- Keep the current edits structural unless the task specifically calls for real rendering behavior.
- Prefer adding new crates or modules behind clean seams rather than reaching across layers.
- Keep Java changes limited to integration shell concerns; renderer ownership belongs in Rust.
- If you add a renderer slice, make it runnable through `examples/standalone` and keep platform bootstrapping reusable from `ferridian-core`.
