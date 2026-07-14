//! The contract document crosses the Java<->Rust boundary; the decoder must
//! reject arbitrary bytes with an error, never panic.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(text) = std::str::from_utf8(data) {
        let _ = ferridian_contract::Contract::from_toml(text);
    }
});
