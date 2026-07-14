package io.ferridian.fixture;

/**
 * Test fixture for the classfile parser: OUR OWN source (not Mojang's),
 * compiled with the pinned JDK into RenderStandin.class next to this file.
 * Shapes mirror what real render classes look like: instance/static methods,
 * arrays, object params, and fields with assorted modifiers.
 *
 * Regenerate after editing: javac RenderStandin.java  (from this directory)
 */
public abstract class RenderStandin {
    private static final long FRAME_BUDGET_NS = 16_666_667L;
    protected int frameIndex;
    public String[] passNames;

    public void renderLevel(String camera, float[] matrices, int flags) {
        frameIndex += flags;
    }

    public static boolean isBudgetExceeded(long elapsedNs) {
        return elapsedNs > FRAME_BUDGET_NS;
    }

    protected abstract double[] exposureHistogram();
}
