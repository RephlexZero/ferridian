package io.ferridian.bridge;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

final class FerridianBootstrapTest {
  @Test
  void reportsRustBridgeBackend() {
    FerridianBootstrap bootstrap = new FerridianBootstrap();

    assertEquals("rust-jni", bootstrap.rendererBackend());
  }
}
