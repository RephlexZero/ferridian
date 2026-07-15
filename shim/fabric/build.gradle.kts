// Fabric entrypoint. Everything loader-agnostic (the generated contract, the
// native bridge into the Vulkan layer) lives in shim/core; this module adds
// only what Fabric itself requires: the loom toolchain, the Minecraft/Loader
// dependency set, and FerridianShim's ClientModInitializer.

plugins {
    java
    id("net.fabricmc.fabric-loom")
}

// So `project(":core").sourceSets` below is safe to read: core's own
// build.gradle.kts must have been evaluated first.
evaluationDependsOn(":core")

group = "io.ferridian"
version = "0.1.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

dependencies {
    minecraft("com.mojang:minecraft:${providers.gradleProperty("minecraft_version").get()}")
    // Minecraft is unobfuscated as of 26.1: no mappings dependency at all
    // (Fabric stopped maintaining third-party mappings from 26.1 onward).
    implementation("net.fabricmc:fabric-loader:${providers.gradleProperty("loader_version").get()}")

    implementation(project(":core"))
}

loom {
    runs {
        named("client") {
            // 26.x's Vulkan backend is opt-in (OpenGL remains the default
            // until removal, ~2027); this is the engine's actual seam, so
            // dev-testing the shim against the OpenGL backend proves nothing.
            programArgs("--graphicsBackend", "VULKAN")
            // Debug-utils labels (what our layer classifies passes against)
            // are gated behind Mojang's own validation flag, not free on
            // every run.
            programArgs("--vulkanValidation", "true")
            // Joins the committed dev save instead of idling at the title
            // screen, so real per-frame world passes (terrain, entities,
            // sky, ...) actually render for the layer to observe. quickPlay
            // only *joins* a save of this exact name — it never creates one;
            // "New World" is the save the first real-game session created
            // (via xdotool through the vanilla create-world screen).
            programArgs("--quickPlaySingleplayer", "New World")
        }
    }
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release.set(25)
}

// Minecraft 26.x ships unobfuscated, so Loom never wires a remapJar task
// here (named == intermediary, nothing to remap) — meaning its usual
// include()/jar-in-jar mechanism, which hooks into remapJar, has nothing to
// attach to. Embed core's classes directly into the plain jar task instead:
// this is what Fabric Loader actually loads at runtime.
tasks.jar {
    from(project(":core").sourceSets.main.get().output)
}
