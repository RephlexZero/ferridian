//! Proves the shim → layer camera transport end to end: a real JVM, loading
//! the real cdylib, calling the real generated `NativeBridge.java` — no seam
//! mocked. The JVM and this test are separate OS processes, so the only way
//! to observe what the JVM privately wrote into `camera_transport`'s process
//! state is to have the JVM read it back itself (through the same generated
//! getters a real shim would use) and print it; this test spawns `java`
//! against a tiny throwaway harness class and parses stdout.
//!
//! Doesn't need a GPU: `System.load` only resolves the cdylib's own direct
//! link dependencies, and none of the exported symbols here touch Vulkan
//! (ash dlopens `libvulkan.so.1` lazily, from inside actual layer-negotiation
//! calls this test never makes) — so this runs everywhere, GPU or not.

use std::path::PathBuf;
use std::process::Command;

/// The unhashed cdylib cargo links into `target/<profile>/` when building
/// this crate's lib target. Duplicated from `composited.rs`: nextest test
/// binaries don't share modules.
fn layer_dylib_path() -> PathBuf {
    let exe = std::env::current_exe().expect("test executable path");
    let deps_dir = exe.parent().expect("test executable lives in deps/");
    let name = if cfg!(windows) {
        "ferridian_vk_layer.dll"
    } else if cfg!(target_os = "macos") {
        "libferridian_vk_layer.dylib"
    } else {
        "libferridian_vk_layer.so"
    };
    let candidates = [
        deps_dir.join(name),
        deps_dir.parent().expect("deps/ has a parent").join(name),
    ];
    candidates
        .iter()
        .find(|path| path.is_file())
        .cloned()
        .unwrap_or_else(|| {
            panic!(
                "layer cdylib not found at {} — did the lib target build?",
                candidates[0].display()
            )
        })
}

fn generated_native_bridge_source() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../shim/core/src/main/generated/io/ferridian/shim/contract/NativeBridge.java")
}

#[test]
fn native_bridge_round_trips_through_a_real_jvm() {
    let dylib = layer_dylib_path();
    let workdir =
        std::env::temp_dir().join(format!("ferridian-camera-transport-{}", std::process::id()));
    let package_dir = workdir.join("io/ferridian/shim/contract");
    std::fs::create_dir_all(&package_dir).expect("create package dir");
    std::fs::copy(
        generated_native_bridge_source(),
        package_dir.join("NativeBridge.java"),
    )
    .expect("stage the real generated NativeBridge.java");

    // A published sun direction that isn't a unit vector: this test only
    // proves the transport carries values through unchanged, not the
    // contract's placeholder invariants.
    let dylib_path = dylib.to_str().expect("dylib path is valid UTF-8");
    let harness = format!(
        r#"
import io.ferridian.shim.contract.NativeBridge;

public final class CameraTransportProbe {{
    public static void main(String[] args) {{
        System.load("{dylib_path}");
        NativeBridge.publishCamera(0.25f, 0.5f, 0.75f, 12.5f, 0.1f, 256.0f);
        System.out.println(NativeBridge.currentSunDirectionX());
        System.out.println(NativeBridge.currentSunDirectionY());
        System.out.println(NativeBridge.currentSunDirectionZ());
        System.out.println(NativeBridge.currentTimeSeconds());
        System.out.println(NativeBridge.currentNearPlane());
        System.out.println(NativeBridge.currentFarPlane());
    }}
}}
"#
    );
    std::fs::write(workdir.join("CameraTransportProbe.java"), harness).expect("write harness");

    let javac = Command::new("javac")
        .arg("-d")
        .arg(&workdir)
        .arg(package_dir.join("NativeBridge.java"))
        .arg(workdir.join("CameraTransportProbe.java"))
        .output()
        .expect("javac runs");
    assert!(
        javac.status.success(),
        "javac failed to compile the generated NativeBridge + probe harness:\n{}",
        String::from_utf8_lossy(&javac.stderr)
    );

    let run = Command::new("java")
        .arg("-cp")
        .arg(&workdir)
        .arg("CameraTransportProbe")
        .output()
        .expect("java runs");
    assert!(
        run.status.success(),
        "java exited non-zero:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );

    let values: Vec<f32> = String::from_utf8(run.stdout)
        .expect("stdout is UTF-8")
        .lines()
        .map(|line| {
            line.trim()
                .parse()
                .unwrap_or_else(|_| panic!("{line:?} parses as f32"))
        })
        .collect();
    assert_eq!(
        values,
        vec![0.25, 0.5, 0.75, 12.5, 0.1, 256.0],
        "the JVM must read back exactly what it published, through the real JNI ABI"
    );

    std::fs::remove_dir_all(&workdir).ok();
}
