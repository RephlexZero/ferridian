# Golden image baselines

Reference renders for the perceptual-diff harness (M1). PNG/EXR files in this
tree are stored in **git-lfs** (see `.gitattributes`); run `mise run setup`
once so your clone has LFS wired.

Rules:

- Baselines are only meaningful against the **pinned Mesa/lavapipe** in
  `ci/mesa.Dockerfile`. Never re-bless from a desktop GPU.
- Bumping the Mesa pin and re-blessing baselines happens together, in one
  deliberate PR.
- Any Vulkan validation error during a golden run fails the job before images
  are even compared.
