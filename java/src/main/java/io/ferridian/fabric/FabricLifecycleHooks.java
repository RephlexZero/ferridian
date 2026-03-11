package io.ferridian.fabric;

import java.nio.ByteBuffer;

/**
 * Defines the Fabric lifecycle hooks that Ferridian needs to integrate with
 * Minecraft's render and resource reload paths. Implementations bridge the
 * concrete Fabric API calls (which depend on Fabric loader being present)
 * into this renderer-agnostic interface.
 */
public interface FabricLifecycleHooks {

    /**
     * Called when world rendering begins for a new frame. The implementation
     * should snapshot the camera, time, and visible chunk state and package
     * it into a flat binary buffer for handoff to Rust.
     *
     * @param cameraX   camera eye X
     * @param cameraY   camera eye Y
     * @param cameraZ   camera eye Z
     * @param yaw       camera yaw in radians
     * @param pitch     camera pitch in radians
     * @param tickDelta partial tick for interpolation
     * @return a direct ByteBuffer containing the serialized frame input
     */
    ByteBuffer snapshotFrameState(
            double cameraX, double cameraY, double cameraZ,
            float yaw, float pitch, float tickDelta);

    /**
     * Called when resource packs are reloaded. The implementation should
     * invalidate cached shader permutations and material definitions and
     * trigger a reload on the Rust side.
     */
    void onResourceReload();

    /**
     * Called when the world is unloaded (e.g. returning to title screen).
     * The implementation should release GPU resources and reset renderer state.
     */
    void onWorldUnload();
}
