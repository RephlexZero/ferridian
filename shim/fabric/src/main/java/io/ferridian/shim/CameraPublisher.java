package io.ferridian.shim;

import io.ferridian.shim.contract.NativeBridge;
import net.minecraft.client.Camera;
import net.minecraft.client.Minecraft;
import net.minecraft.client.multiplayer.ClientLevel;
import net.minecraft.world.attribute.EnvironmentAttributes;

/**
 * Publishes real per-frame camera state (sun direction, world time, clip
 * planes) to the Vulkan layer through the generated {@link NativeBridge}.
 *
 * <p>The shim deliberately does no mixins or game hooks, and without
 * fabric-api there is no tick event to ride either — so this is a daemon
 * thread polling {@link Minecraft#getInstance()} at tick rate. Everything it
 * reads is a handful of primitive fields; a torn read races only against the
 * next poll 50&nbsp;ms later, and any surprise from reading render state off
 * the render thread is caught and skipped rather than allowed to take the
 * game down.
 */
final class CameraPublisher implements Runnable {
    /** Poll at Minecraft's own tick rate; sun/time/planes move no faster. */
    private static final long POLL_MILLIS = 50;

    /**
     * Vanilla's classic far plane: effective render distance in blocks,
     * times the same 4x margin the OpenGL pipeline's getDepthFar() used.
     */
    private static final float FAR_PLANE_MARGIN = 4.0f;

    private boolean publishedOnce;

    static void start() {
        Thread thread = new Thread(new CameraPublisher(), "ferridian-camera-publisher");
        thread.setDaemon(true);
        thread.start();
    }

    private CameraPublisher() {}

    @Override
    public void run() {
        while (true) {
            try {
                publishCurrentState();
            } catch (Throwable error) {
                // Racing the render thread for a few floats: skip this poll,
                // never take the game down.
            }
            try {
                Thread.sleep(POLL_MILLIS);
            } catch (InterruptedException interrupted) {
                Thread.currentThread().interrupt();
                return;
            }
        }
    }

    private void publishCurrentState() {
        Minecraft minecraft = Minecraft.getInstance();
        if (minecraft == null || minecraft.gameRenderer == null) {
            return;
        }
        ClientLevel level = minecraft.level;
        Camera camera = minecraft.gameRenderer.mainCamera();
        if (level == null || camera == null || !camera.isInitialized()) {
            return;
        }

        // 26.2's renderer reads the sun's sky angle (radians; 0 = noon, the
        // convention the sky rotation has kept since the OpenGL pipeline)
        // through the camera's environment-attribute probe — the same source
        // SkyRenderer.extractRenderState uses.
        float sunAngle = camera.attributeProbe().getValue(EnvironmentAttributes.SUN_ANGLE, 1.0f);
        // The sky rotates around the horizon axis: noon straight up (+y),
        // sunrise/sunset on the east-west (x) axis.
        float sunX = -(float) Math.sin(sunAngle);
        float sunY = (float) Math.cos(sunAngle);

        float timeSeconds = level.getDefaultClockTime() / 20.0f;
        float farPlane =
                minecraft.options.getEffectiveRenderDistance() * 16 * FAR_PLANE_MARGIN;

        NativeBridge.publishCamera(
                sunX, sunY, 0.0f, timeSeconds, Camera.PROJECTION_Z_NEAR, farPlane);

        if (!publishedOnce) {
            publishedOnce = true;
            System.out.println(
                    "[ferridian-shim] first real camera published: sun=(" + sunX + ", " + sunY
                            + ", 0.0) time=" + timeSeconds + "s near=" + Camera.PROJECTION_Z_NEAR
                            + " far=" + farPlane);
        }
    }
}
