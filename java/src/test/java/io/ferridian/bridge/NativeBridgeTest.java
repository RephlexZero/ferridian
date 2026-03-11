package io.ferridian.bridge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

final class NativeBridgeTest {
    @Test
    void loadsNativeLibraryAndPingsRust() {
        assertEquals("ferridian_jni", NativeBridge.libraryName());

        String response = NativeBridge.pingChecked();

        assertTrue(NativeBridge.isLoaded());
        assertEquals("ferridian-jni-ready", response);
    }
}