package io.ferridian.bridge;

public final class NativeBridge {
  private static final String LIBRARY_NAME = "ferridian_jni";
  private static volatile boolean loaded;

  private NativeBridge() {
  }

  public static String libraryName() {
    return LIBRARY_NAME;
  }

  public static boolean isLoaded() {
    return loaded;
  }

  public static synchronized void ensureLoaded() {
    if (loaded) {
      return;
    }

    System.loadLibrary(LIBRARY_NAME);
    loaded = true;
  }

  public static String pingChecked() {
    ensureLoaded();
    return ping();
  }

  public static native String ping();
}
