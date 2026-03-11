package io.ferridian.fabric;

/**
 * Fabric mod initializer for Ferridian. This is the entry point when running
 * as a Fabric mod and is responsible for:
 * <ol>
 * <li>Loading the native Ferridian library</li>
 * <li>Registering lifecycle hooks (resource reload, world render)</li>
 * <li>Bridging Minecraft frame state to the Rust renderer</li>
 * </ol>
 *
 * <p>
 * This class is referenced in {@code fabric.mod.json} as the mod initializer.
 * Until the Fabric API dependency is wired in, it serves as a loader shell
 * that validates the native library and bootstrap path.
 */
public final class FerridianModInitializer {

    private static boolean nativeLoaded = false;

    /**
     * Called by Fabric during mod initialization. Loads the native renderer
     * library and prepares the bridge layer.
     */
    public void onInitialize() {
        loadNativeLibrary();
    }

    /**
     * Attempts to load the Ferridian native library. Fails gracefully so
     * the mod can report a clear error instead of crashing the game with an
     * {@link UnsatisfiedLinkError}.
     */
    static void loadNativeLibrary() {
        if (nativeLoaded) {
            return;
        }
        try {
            System.loadLibrary("ferridian_jni");
            nativeLoaded = true;
        } catch (UnsatisfiedLinkError e) {
            System.err.println("[Ferridian] Failed to load native library: " + e.getMessage());
        }
    }

    /**
     * @return true if the native renderer library is loaded and ready.
     */
    public static boolean isNativeLoaded() {
        return nativeLoaded;
    }

    /**
     * Reset the loaded flag for testing.
     */
    static void resetForTesting() {
        nativeLoaded = false;
    }
}
