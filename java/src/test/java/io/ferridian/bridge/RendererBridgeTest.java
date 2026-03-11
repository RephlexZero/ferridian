package io.ferridian.bridge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertTrue;

final class RendererBridgeTest {
    @Test
    void allocatesRendererHandlesAndAcceptsFrames() {
        long handle = RendererBridge.initRenderer(1280, 720);

        assertTrue(handle > 0);

        RendererBridge.resizeRenderer(handle, 1920, 1080);
        RendererBridge.renderFrame(handle, 0.5f);
    }
}