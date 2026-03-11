plugins {
  java
}

group = "io.ferridian"
version = "0.1.0"

java {
  toolchain {
    languageVersion = JavaLanguageVersion.of(21)
  }
}

repositories {
  mavenCentral()
}

dependencies {
  testImplementation(platform("org.junit:junit-bom:5.12.0"))
  testImplementation("org.junit.jupiter:junit-jupiter")
  testRuntimeOnly("org.junit.platform:junit-platform-launcher")
}

val workspaceRoot = rootDir.parentFile

val buildNativeDebug by tasks.registering(Exec::class) {
  workingDir = workspaceRoot
  commandLine("cargo", "build", "-p", "ferridian-jni-bridge")
}

tasks.test {
  dependsOn(buildNativeDebug)
  useJUnitPlatform()
  systemProperty("java.library.path", workspaceRoot.resolve("target/debug").absolutePath)
}
