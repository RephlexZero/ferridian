package io.ferridian.fabric;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;

class FerridianModInitializerTest {

    @Test
    void nativeNotLoadedByDefault() {
        FerridianModInitializer.resetForTesting();
        assertFalse(FerridianModInitializer.isNativeLoaded());
    }

    @Test
    void loadNativeLibraryDoesNotThrow() {
        FerridianModInitializer.resetForTesting();
        // Should never throw regardless of whether the library is available.
        assertDoesNotThrow(() -> FerridianModInitializer.loadNativeLibrary());
    }
}
