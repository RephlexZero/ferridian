package io.ferridian.bridge;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public final class ChunkSectionBuffers {
    public static final int LAYOUT_VERSION = 1;
    public static final int EDGE = 16;
    public static final int VOXEL_COUNT = EDGE * EDGE * EDGE;
    public static final int HEADER_BYTES = 24;
    public static final int PACKET_BYTES = HEADER_BYTES + VOXEL_COUNT;

    private ChunkSectionBuffers() {
    }

    public static ByteBuffer singleColumnSample() {
        ByteBuffer buffer = ByteBuffer.allocateDirect(PACKET_BYTES).order(ByteOrder.LITTLE_ENDIAN);
        buffer.putInt(LAYOUT_VERSION);
        buffer.putInt(0);
        buffer.putInt(0);
        buffer.putInt(0);
        buffer.putInt(3);
        buffer.putInt(VOXEL_COUNT);
        buffer.put((byte) 1);
        buffer.put((byte) 2);
        buffer.put((byte) 3);
        buffer.position(0);
        return buffer;
    }
}