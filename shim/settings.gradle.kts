pluginManagement {
    repositories {
        maven("https://maven.fabricmc.net/") { name = "Fabric" }
        mavenCentral()
        gradlePluginPortal()
    }
    plugins {
        // Resolved here because the project-level plugins {} block cannot
        // read gradle.properties through `extra` in the Kotlin DSL.
        id("net.fabricmc.fabric-loom") version providers.gradleProperty("loom_version").get()
    }
}

rootProject.name = "ferridian-shim"
