package io.ferridian.bridge;

import java.nio.ByteBuffer;

public final class RendererBridge {
    private RendererBridge() {
    }

    public static long initRenderer(int width, int height) {
        NativeBridge.ensureLoaded();
        return initRendererNative(width, height);
    }

    public static void resizeRenderer(long handle, int width, int height) {
        NativeBridge.ensureLoaded();
        resizeRendererNative(handle, width, height);
    }

    public static void renderFrame(long handle, float timeSeconds) {
        NativeBridge.ensureLoaded();
        renderFrameNative(handle, timeSeconds);
    }

    public static int uploadChunkSection(long handle, ByteBuffer sectionBuffer) {
        NativeBridge.ensureLoaded();
        return uploadChunkSectionNative(handle, sectionBuffer);
    }

    private static native long initRendererNative(int width, int height);

    private static native void resizeRendererNative(long handle, int width, int height);

    private static native void renderFrameNative(long handle, float timeSeconds);

    private static native int uploadChunkSectionNative(long handle, ByteBuffer sectionBuffer);
}