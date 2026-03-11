package io.ferridian.fabric;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

class FabricRendererHooksTest {

    @Test
    void snapshotFrameStateProducesCorrectSize() {
        FabricRendererHooks hooks = new FabricRendererHooks();
        ByteBuffer buf = hooks.snapshotFrameState(1.0, 2.0, 3.0, 0.5f, 0.3f, 0.75f);
        assertNotNull(buf);
        assertTrue(buf.isDirect());
        assertEquals(32, buf.remaining());
    }

    @Test
    void snapshotFrameStateContainsCorrectValues() {
        FabricRendererHooks hooks = new FabricRendererHooks();
        ByteBuffer buf = hooks.snapshotFrameState(10.0, 64.0, -20.0, 1.5f, -0.5f, 0.25f);
        buf.order(ByteOrder.LITTLE_ENDIAN);

        assertEquals(10.0f, buf.getFloat(), 0.001f);
        assertEquals(64.0f, buf.getFloat(), 0.001f);
        assertEquals(-20.0f, buf.getFloat(), 0.001f);
        assertEquals(1.5f, buf.getFloat(), 0.001f);
        assertEquals(-0.5f, buf.getFloat(), 0.001f);
    }

    @Test
    void rendererNotInitializedByDefault() {
        FabricRendererHooks hooks = new FabricRendererHooks();
        assertFalse(hooks.isRendererInitialized());
    }

    @Test
    void onWorldUnloadResetsRenderer() {
        FabricRendererHooks hooks = new FabricRendererHooks();
        hooks.onWorldUnload();
        assertFalse(hooks.isRendererInitialized());
    }
}
