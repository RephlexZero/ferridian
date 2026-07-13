package io.ferridian.shim;

import io.ferridian.shim.contract.FerridianContract;
import net.fabricmc.api.ClientModInitializer;

/**
 * Entry point for the Ferridian shim.
 *
 * <p>The shim's whole job is to publish pass metadata (which game pipeline is
 * terrain, entities, …) to the engine over the versioned contract in
 * {@link io.ferridian.shim.contract}. Everything in that package is generated
 * by {@code tools/shim-codegen}; never edit it by hand.
 *
 * <p>TODO(M2): publish {@link FerridianContract#PASSES} to the engine once the
 * Vulkan layer's transport lands.
 */
public final class FerridianShim implements ClientModInitializer {
    @Override
    public void onInitializeClient() {
        System.out.println(
                "[ferridian-shim] contract v" + FerridianContract.CONTRACT_VERSION
                        + " for Minecraft " + FerridianContract.MINECRAFT_VERSION
                        + " (" + FerridianContract.PASSES.size() + " passes)");
    }
}
