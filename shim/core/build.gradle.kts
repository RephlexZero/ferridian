// Loader-agnostic Java: the generated contract (src/main/generated) plus the
// native-method bridge into the Vulkan layer's cdylib. No Fabric/Loom
// dependency here — every loader entrypoint module (fabric/, eventually
// neoforge/) depends on this one and adds only what that loader requires.

plugins {
    java
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

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
    options.release.set(25)
}
