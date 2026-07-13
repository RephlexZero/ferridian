//! Pack manifests are untrusted input: fuzz the parse + validate path.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = ferridian_pack_format::PackManifest::from_toml_str(text);
    }
});
