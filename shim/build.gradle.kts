// The shim is ~90% generated: pass metadata under src/main/generated comes
// from `cargo run -p shim-codegen` (source of truth: crates/contract).
// Keep hand-written code to lifecycle glue; logic belongs engine-side.

plugins {
    java
    id("fabric-loom") version (extra["loom_version"] as String)
}

group = "io.ferridian"
version = "0.1.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
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
    minecraft("com.mojang:minecraft:${extra["minecraft_version"]}")
    // Minecraft is unobfuscated as of 26.1; official mappings are the identity choice.
    mappings(loom.officialMojangMappings())
    modImplementation("net.fabricmc:fabric-loader:${extra["loader_version"]}")
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release.set(21)
}
