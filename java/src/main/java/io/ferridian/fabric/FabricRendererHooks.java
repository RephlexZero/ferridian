package io.ferridian.fabric;

import io.ferridian.bridge.NativeBridge;
import io.ferridian.bridge.RendererBridge;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Concrete implementation of {@link FabricLifecycleHooks} that connects
 * Fabric lifecycle events to the native Ferridian renderer via JNI.
 *
 * <p>
 * This class is the runtime bridge between Minecraft's Fabric hooks and
 * the Rust-side renderer. It serializes frame state into the binary layout
 * expected by {@code submit_frame_snapshot} on the Rust side.
 */
public final class FabricRendererHooks implements FabricLifecycleHooks {

    /**
     * Binary size of the frame snapshot header (must match Rust
     * FRAME_SNAPSHOT_HEADER_BYTES).
     */
    private static final int FRAME_SNAPSHOT_BYTES = 32;

    private boolean rendererInitialized = false;

    /**
     * Initialize the renderer if it hasn't been started yet.
     * Called lazily on first frame.
     */
    public void ensureRendererReady(int width, int height) {
        if (!rendererInitialized && FerridianModInitializer.isNativeLoaded()) {
            RendererBridge.initRenderer(width, height);
            rendererInitialized = true;
        }
    }

    @Override
    public ByteBuffer snapshotFrameState(
            double cameraX, double cameraY, double cameraZ,
            float yaw, float pitch, float tickDelta) {
        ByteBuffer buf = ByteBuffer.allocateDirect(FRAME_SNAPSHOT_BYTES);
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.putFloat((float) cameraX);
        buf.putFloat((float) cameraY);
        buf.putFloat((float) cameraZ);
        buf.putFloat(yaw);
        buf.putFloat(pitch);
        buf.putFloat(0.0f); // game time placeholder
        buf.putFloat(tickDelta);
        buf.putInt(0); // visible chunk count placeholder
        buf.flip();
        return buf;
    }

    @Override
    public void onResourceReload() {
        if (FerridianModInitializer.isNativeLoaded()) {
            // Future: call native resource reload
            System.out.println("[Ferridian] Resource reload requested");
        }
    }

    @Override
    public void onWorldUnload() {
        if (rendererInitialized && FerridianModInitializer.isNativeLoaded()) {
            System.out.println("[Ferridian] World unloaded, releasing renderer resources");
            rendererInitialized = false;
        }
    }

    /**
     * @return true if the renderer has been initialized.
     */
    public boolean isRendererInitialized() {
        return rendererInitialized;
    }
}
