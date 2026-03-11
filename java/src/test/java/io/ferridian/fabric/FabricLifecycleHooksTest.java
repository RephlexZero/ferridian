package io.ferridian.fabric;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

class FabricLifecycleHooksTest {

    /**
     * Verify FabricRendererHooks implements the FabricLifecycleHooks interface.
     */
    @Test
    void fabricRendererHooksImplementsInterface() {
        FabricLifecycleHooks hooks = new FabricRendererHooks();
        assertNotNull(hooks);
    }

    @Test
    void snapshotFrameStateViaInterface() {
        FabricLifecycleHooks hooks = new FabricRendererHooks();
        ByteBuffer buf = hooks.snapshotFrameState(0.0, 0.0, 0.0, 0.0f, 0.0f, 0.0f);
        assertNotNull(buf);
        assertEquals(32, buf.remaining());
    }

    @Test
    void onResourceReloadDoesNotThrow() {
        FabricLifecycleHooks hooks = new FabricRendererHooks();
        assertDoesNotThrow(hooks::onResourceReload);
    }

    @Test
    void onWorldUnloadDoesNotThrow() {
        FabricLifecycleHooks hooks = new FabricRendererHooks();
        assertDoesNotThrow(hooks::onWorldUnload);
    }

    @Test
    void snapshotWithExtremeValues() {
        FabricLifecycleHooks hooks = new FabricRendererHooks();
        ByteBuffer buf = hooks.snapshotFrameState(
                Double.MAX_VALUE, Double.MIN_VALUE, -1e10,
                Float.MAX_VALUE, Float.MIN_VALUE, 0.001f);
        assertNotNull(buf);
        assertTrue(buf.isDirect());
        assertEquals(32, buf.remaining());
    }
}
