# Research & Study Notes

This file collects study notes, key findings, and architectural takeaways from
the projects and papers listed in the Ferridian learning backlog.

## wgpu-mc — JNI and Minecraft Renderer Ownership

- **Repository**: https://github.com/nicholasgasior/wgpu-mc
- **Key patterns**: JNI bridge with DirectByteBuffer, Rust-side renderer ownership,
  chunk mesh upload via flat binary buffers.
- **Takeaway**: Keep renderer state entirely in Rust. Java side should only snapshot
  world state and hand it off. Never let Java hold GPU handles.

## Bevy Render Architecture

- **Key ideas**: Render graph with explicit node dependencies, pipeline caching via
  `SpecializedRenderPipeline`, bind group caching, extract-prepare-queue-render phases.
- **Takeaway**: naga_oil for shader composition matches Bevy's approach. Consider
  similar pipeline specialization cache for Ferridian's material permutations.

## Veloren — Voxel Renderer Production Patterns

- **Key patterns**: wgpu-based voxel renderer, greedy meshing, LoD system, dual
  contouring for smooth terrain.
- **Takeaway**: Greedy meshing significantly reduces vertex count for voxel terrain.
  Consider as optimization after basic face-culled meshing is profiled.

## Sodium — Region Buffers and Compact Formats

- **Key patterns**: Region-based chunk buffers, compact 20-byte vertices, indirect
  draw for visible chunks, parallel chunk meshing on CPU.
- **Takeaway**: Our 12-byte MeshVertex format already matches the compact target.
  GPU-resident indirect draw is the next throughput step.

## Wicked Engine — Voxel Cone Tracing GI

- **Key patterns**: Sparse voxel octree, 3D texture cascades, cone tracing with
  multiple cone angles for diffuse/specular.
- **Takeaway**: For Minecraft, the block grid IS the voxel structure — skip the
  expensive voxelization step. Use cascaded 3D textures matching chunk regions.

## XeGTAO — SSAO & Visibility-Bitmask GTAO

- **Papers**: "Practical Real-Time Strategies for Accurate Indirect Occlusion"
  (Jimenez et al.), Intel XeGTAO.
- **Key idea**: Replace sample-based AO with visibility bitmask — cheaper per pixel,
  better quality, and naturally feeds into indirect diffuse.
- **Takeaway**: Implement GTAO with half-resolution option for performance scaling.

## Teardown / Tuxedo Labs — Small-Team Voxel Rendering

- **Key patterns**: Single developer built a voxel path tracer. Uses denoising
  aggressively, keeps shading simple, prioritizes interactive frame times.
- **Takeaway**: Path tracing as optional high-end mode is validated. Denoising
  quality matters more than raw ray count.

## Bevy Solari — wgpu Ray-Query Experiments

- **Status**: Experimental, not production-ready.
- **Takeaway**: Monitor but don't depend on. wgpu ray-query support is still
  evolving. Keep path tracing behind opt-in flag.

## Complementary / BSL Shader Packs

- **Key patterns**: GLSL shader packs with classic Iris pass stages. Heavy use of
  composite passes for post-processing. LabPBR material interpretation.
- **Takeaway**: Our IrisPassStage mapping covers the common structure. GLSL
  ingestion via naga handles the shader translation.

## Mojang / Aperture Snapshot Tracking

- **Status**: No stable Vulkan-facing interface published yet.
- **Takeaway**: Keep the Iris adapter transitional. The RenderPipelineProvider
  abstraction allows swapping to Aperture when it stabilizes.
