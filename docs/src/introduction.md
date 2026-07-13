# Ferridian

Ferridian is a Vulkan-native shader engine for Minecraft: Java Edition,
built for the post-OpenGL era. It is not a shader *loader*: it targets the
flagship tier — modern GPU techniques (ReSTIR GI, radiance cascades,
ray-query paths, temporal upscaling) as engine primitives, with a pack
toolchain that catches your mistakes at build time, not at game load.

Three ideas define it:

1. **Interception at the Vulkan layer.** The engine attaches to the game's
   own Vulkan renderer through the stable Khronos layer ABI, so it survives
   game updates by construction. A thin, generated Fabric shim publishes
   semantic pass metadata over a versioned contract.
2. **Packs are compiled, not interpreted.** You write Slang (or GLSL); `packc`
   compiles it ahead-of-time to SPIR-V with full reflection, validates the
   pass graph, and emits an artifact the engine loads with near-zero work.
3. **Robustness is an architecture property.** Validation-layer errors fail
   CI. Golden images render on a pinned software rasteriser. Mojang snapshots
   are diffed automatically for renderer changes.

These docs are the product surface for pack authors. Start with
[Getting Started](pack-authors/getting-started.md).
