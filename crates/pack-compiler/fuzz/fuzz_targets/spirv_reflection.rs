//! The engine reflects SPIR-V from pack artifacts — untrusted input if a
//! pack was tampered with post-build. Reflection must reject garbage with an
//! error, never panic or hang.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    let words: Vec<u32> = data
        .chunks_exact(4)
        .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect();
    let _ = ferridian_pack_compiler::reflect_spirv(&words);
});
