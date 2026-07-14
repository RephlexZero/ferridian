// The shim is ~90% generated: pass metadata under src/main/generated comes
// from `cargo run -p shim-codegen` (source of truth: crates/contract).
// Keep hand-written code to lifecycle glue; logic belongs engine-side.

plugins {
    java
    id("net.fabricmc.fabric-loom")
}

group = "io.ferridian"
version = "0.1.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(25))
    }
}

sourceSets {
    main {
        java {
            srcDir("src/main/generated")
        }
    }
}

dependencies {
    minecraft("com.mojang:minecraft:${providers.gradleProperty("minecraft_version").get()}")
    // Minecraft is unobfuscated as of 26.1: no mappings dependency at all
    // (Fabric stopped maintaining third-party mappings from 26.1 onward).
    implementation("net.fabricmc:fabric-loader:${providers.gradleProperty("loader_version").get()}")
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release.set(25)
}
