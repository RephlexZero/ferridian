# Golden image baselines

Reference renders for the perceptual-diff harness. PNG/EXR files in this
tree are stored in **git-lfs** (see `.gitattributes`); run `mise run setup`
once so your clone has LFS wired.

Workflow:

- Golden tests live in `crates/testkit/tests/goldens.rs` and run with
  `mise run gpu-test` (gated on `FERRIDIAN_GPU_TESTS=1`).
- To add or intentionally change a baseline, run `mise run golden-bless`
  **inside the devcontainer / Mesa image** and commit the PNG.
- The comparison policy (`DiffPolicy` in `ferridian-testkit`) allows one
  quantum per channel and zero differing pixels: lavapipe is pinned, so any
  real difference is a change to be reviewed, not noise to be tolerated.
- On failure the actual render + a difference heatmap land in
  `target/golden-failures/` (uploaded as CI artifacts).

Rules:

- Baselines are only meaningful against the **pinned Mesa/lavapipe** in
  `ci/mesa.Dockerfile`. Never re-bless from a desktop GPU.
- Bumping the Mesa pin and re-blessing baselines happens together, in one
  deliberate PR.
- Any Vulkan validation error during a golden run fails the job before images
  are even compared.
