//! Jars are untrusted input; the classfile parser and signature renderer
//! must reject arbitrary bytes with an error — never panic, hang, or blow
//! the stack.
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    if let Ok(parsed) = upstream_watch::classfile::parse_class(data) {
        let _ = upstream_watch::classfile::class_signatures(&parsed);
    }
});
