package io.ferridian.bridge;

import org.junit.jupiter.api.Test;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import static org.junit.jupiter.api.Assertions.*;

final class ChunkSectionBuffersTest {

    @Test
    void constantsAreConsistent() {
        assertEquals(16, ChunkSectionBuffers.EDGE);
        assertEquals(16 * 16 * 16, ChunkSectionBuffers.VOXEL_COUNT);
        assertEquals(24, ChunkSectionBuffers.HEADER_BYTES);
        assertEquals(24 + 4096, ChunkSectionBuffers.PACKET_BYTES);
    }

    @Test
    void layoutVersionMatchesRust() {
        assertEquals(1, ChunkSectionBuffers.LAYOUT_VERSION);
    }

    @Test
    void singleColumnSampleIsDirect() {
        ByteBuffer buf = ChunkSectionBuffers.singleColumnSample();
        assertTrue(buf.isDirect());
    }

    @Test
    void singleColumnSampleHasCorrectSize() {
        ByteBuffer buf = ChunkSectionBuffers.singleColumnSample();
        assertEquals(ChunkSectionBuffers.PACKET_BYTES, buf.remaining());
    }

    @Test
    void singleColumnSampleHasCorrectHeader() {
        ByteBuffer buf = ChunkSectionBuffers.singleColumnSample();
        buf.order(ByteOrder.LITTLE_ENDIAN);
        assertEquals(ChunkSectionBuffers.LAYOUT_VERSION, buf.getInt()); // version
        assertEquals(0, buf.getInt()); // section_x
        assertEquals(0, buf.getInt()); // section_y
        assertEquals(0, buf.getInt()); // section_z
        assertEquals(3, buf.getInt()); // non_air_blocks
        assertEquals(ChunkSectionBuffers.VOXEL_COUNT, buf.getInt()); // block_bytes
    }

    @Test
    void singleColumnSampleHasThreeNonAirBlocks() {
        ByteBuffer buf = ChunkSectionBuffers.singleColumnSample();
        buf.order(ByteOrder.LITTLE_ENDIAN);
        buf.position(ChunkSectionBuffers.HEADER_BYTES);
        int nonAir = 0;
        while (buf.hasRemaining()) {
            if (buf.get() != 0) {
                nonAir++;
            }
        }
        assertEquals(3, nonAir);
    }
}
