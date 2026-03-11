package io.ferridian.bridge;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class RendererBridgeTest {
    @Test
    void allocatesRendererHandlesAndAcceptsFrames() {
        long handle = RendererBridge.initRenderer(1280, 720);

        assertTrue(handle > 0);

        RendererBridge.resizeRenderer(handle, 1920, 1080);
        RendererBridge.renderFrame(handle, 0.5f);
    }

    @Test
    void uploadsChunkSectionsThroughDirectByteBuffer() {
        long handle = RendererBridge.initRenderer(1280, 720);
        ByteBuffer sectionBuffer = ChunkSectionBuffers.singleColumnSample();

        int nonAirBlocks = RendererBridge.uploadChunkSection(handle, sectionBuffer);

        assertEquals(3, nonAirBlocks);
    }
}