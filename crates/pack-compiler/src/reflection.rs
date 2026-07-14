//! SPIR-V reflection: what the engine needs to bind a compiled module without
//! stringly-typed guessing — entry points and descriptor bindings, derived
//! from the module itself.
//!
//! The engine re-reflects pack artifacts at load time, so this input is
//! untrusted (a `.spv` can be tampered with after `packc build`). Fuzzing
//! found that rspirv 0.12's decoder *panics* on several malformed-input
//! shapes (out-of-bounds slices, unknown enum operands), so reflection walks
//! the word stream itself — every read bounds-checked, every failure an
//! `Err` — and uses rspirv only for enum names. The four instructions we
//! need (OpEntryPoint, OpName, OpDecorate, OpVariable) are flat and simple.

use std::collections::BTreeMap;

use rspirv::spirv::{Decoration, ExecutionModel, Op, StorageClass, Word};
use serde::Serialize;

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ShaderReflection {
    pub entry_points: Vec<EntryPointReflection>,
    pub bindings: Vec<BindingReflection>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct EntryPointReflection {
    pub name: String,
    pub stage: String,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct BindingReflection {
    pub set: u32,
    pub binding: u32,
    /// SPIR-V storage class of the bound variable ("Uniform",
    /// "UniformConstant", "StorageBuffer", …).
    pub storage_class: String,
    pub name: Option<String>,
}

const SPIRV_MAGIC: u32 = 0x0723_0203;
const HEADER_WORDS: usize = 5;

/// Parse a SPIR-V module and extract its interface. Returns a human-readable
/// message on malformed input (callers wrap it with pass context).
pub fn reflect_spirv(words: &[u32]) -> Result<ShaderReflection, String> {
    if words.len() < HEADER_WORDS {
        return Err(format!(
            "module has {} word(s); even the SPIR-V header needs {HEADER_WORDS}",
            words.len()
        ));
    }
    if words[0] != SPIRV_MAGIC {
        return Err(format!(
            "bad SPIR-V magic {:#010x} (byte-swapped or not SPIR-V)",
            words[0]
        ));
    }

    let mut names: BTreeMap<Word, String> = BTreeMap::new();
    let mut storage_classes: BTreeMap<Word, u32> = BTreeMap::new();
    let mut sets: BTreeMap<Word, u32> = BTreeMap::new();
    let mut binding_slots: BTreeMap<Word, u32> = BTreeMap::new();
    let mut entry_points = Vec::new();

    let mut offset = HEADER_WORDS;
    while offset < words.len() {
        let word_count = (words[offset] >> 16) as usize;
        let opcode = words[offset] & 0xffff;
        if word_count == 0 {
            return Err(format!("instruction at word {offset} declares zero length"));
        }
        if word_count > words.len() - offset {
            return Err(format!(
                "instruction at word {offset} declares {word_count} words but only {} remain",
                words.len() - offset
            ));
        }
        let operands = &words[offset + 1..offset + word_count];

        if opcode == Op::EntryPoint as u32 {
            // [execution model, entry point id, name string, interface ids…]
            if operands.len() < 3 {
                return Err(format!("OpEntryPoint at word {offset} is too short"));
            }
            let name = decode_string(&operands[2..])
                .ok_or_else(|| format!("OpEntryPoint at word {offset} has an unterminated name"))?;
            entry_points.push(EntryPointReflection {
                name,
                stage: enum_name(ExecutionModel::from_u32(operands[0]), operands[0]),
            });
        } else if opcode == Op::Name as u32 {
            // [target id, name string]
            if operands.len() >= 2
                && let Some(name) = decode_string(&operands[1..])
            {
                names.insert(operands[0], name);
            }
        } else if opcode == Op::Decorate as u32 {
            // [target id, decoration, extra literals…]
            if operands.len() >= 3 {
                if operands[1] == Decoration::DescriptorSet as u32 {
                    sets.insert(operands[0], operands[2]);
                } else if operands[1] == Decoration::Binding as u32 {
                    binding_slots.insert(operands[0], operands[2]);
                }
            }
        } else if opcode == Op::Variable as u32 {
            // [result type, result id, storage class, (initializer)]
            if operands.len() >= 3 {
                storage_classes.insert(operands[1], operands[2]);
            }
        }

        offset += word_count;
    }

    entry_points.sort();
    let mut bindings: Vec<BindingReflection> = binding_slots
        .iter()
        .map(|(&id, &binding)| BindingReflection {
            set: sets.get(&id).copied().unwrap_or(0),
            binding,
            storage_class: storage_classes
                .get(&id)
                .map(|&class| enum_name(StorageClass::from_u32(class), class))
                .unwrap_or_else(|| "Unknown".to_owned()),
            name: names.get(&id).cloned(),
        })
        .collect();
    bindings.sort();

    Ok(ShaderReflection {
        entry_points,
        bindings,
    })
}

/// A SPIR-V literal string: NUL-terminated UTF-8 packed little-endian into
/// words. `None` when the terminator is missing (malformed instruction) —
/// invalid UTF-8 degrades lossily rather than failing, names are advisory.
fn decode_string(operands: &[u32]) -> Option<String> {
    let mut bytes = Vec::with_capacity(operands.len() * 4);
    for word in operands {
        for byte in word.to_le_bytes() {
            if byte == 0 {
                return Some(String::from_utf8_lossy(&bytes).into_owned());
            }
            bytes.push(byte);
        }
    }
    None
}

/// Debug-format a known enum value, or show the raw number for values this
/// rspirv doesn't know (future SPIR-V versions are not parse failures).
fn enum_name<E: std::fmt::Debug>(known: Option<E>, raw: u32) -> String {
    match known {
        Some(value) => format!("{value:?}"),
        None => format!("Unknown({raw})"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_terminated_strings() {
        // "uv" + NUL padding in one word.
        let word = u32::from_le_bytes([b'u', b'v', 0, 0]);
        assert_eq!(decode_string(&[word]).as_deref(), Some("uv"));
    }

    #[test]
    fn unterminated_string_is_none() {
        let word = u32::from_le_bytes([b'a', b'b', b'c', b'd']);
        assert_eq!(decode_string(&[word]), None);
    }
}
