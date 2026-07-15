//! SPIR-V reflection: what the engine needs to bind a compiled module without
//! stringly-typed guessing — entry points and descriptor bindings, derived
//! from the module itself.
//!
//! The engine re-reflects pack artifacts at load time, so this input is
//! untrusted (a `.spv` can be tampered with after `packc build`). Fuzzing
//! found that rspirv 0.12's decoder *panics* on several malformed-input
//! shapes (out-of-bounds slices, unknown enum operands), so reflection walks
//! the word stream itself — every read bounds-checked, every failure an
//! `Err` — and uses rspirv only for enum names. The instructions we need
//! (OpEntryPoint, OpExecutionMode, OpName, OpDecorate, OpVariable, and the
//! type instructions behind storage-image detection) are flat and simple.

use std::collections::BTreeMap;

use rspirv::spirv::{Decoration, ExecutionMode, ExecutionModel, Op, StorageClass, Word};
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
    /// `OpExecutionMode LocalSize` — present on compute entry points, the
    /// executor derives dispatch group counts from it. `None` for graphics
    /// stages (and for `LocalSizeId`, which needs constant resolution this
    /// walker deliberately doesn't do — the planner rejects that as missing).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub workgroup_size: Option<[u32; 3]>,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize)]
pub struct BindingReflection {
    pub set: u32,
    pub binding: u32,
    /// SPIR-V storage class of the bound variable ("Uniform",
    /// "UniformConstant", "StorageBuffer", …).
    pub storage_class: String,
    /// True when the variable is a storage image (`OpTypeImage` with
    /// `Sampled = 2`, e.g. Slang's `RWTexture2D`) — written by shaders, as
    /// opposed to a sampled texture. Both live in `UniformConstant`, so the
    /// storage class alone can't tell them apart.
    pub storage_image: bool,
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
    // (entry point id, name, stage) — workgroup sizes attach after the walk,
    // since OpExecutionMode references the id.
    let mut entry_points: Vec<(Word, String, String)> = Vec::new();
    let mut workgroup_sizes: BTreeMap<Word, [u32; 3]> = BTreeMap::new();
    // Type graph for storage-image detection: variable -> pointer type ->
    // pointee type -> OpTypeImage's Sampled literal.
    let mut variable_types: BTreeMap<Word, Word> = BTreeMap::new();
    let mut pointer_pointees: BTreeMap<Word, Word> = BTreeMap::new();
    let mut image_sampled: BTreeMap<Word, u32> = BTreeMap::new();

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
            entry_points.push((
                operands[1],
                name,
                enum_name(ExecutionModel::from_u32(operands[0]), operands[0]),
            ));
        } else if opcode == Op::ExecutionMode as u32 {
            // [entry point id, mode, literals…] — LocalSize carries x, y, z.
            if operands.len() >= 5 && operands[1] == ExecutionMode::LocalSize as u32 {
                workgroup_sizes.insert(operands[0], [operands[2], operands[3], operands[4]]);
            }
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
                variable_types.insert(operands[1], operands[0]);
            }
        } else if opcode == Op::TypePointer as u32 {
            // [result id, storage class, pointee type id]
            if operands.len() >= 3 {
                pointer_pointees.insert(operands[0], operands[2]);
            }
        } else if opcode == Op::TypeImage as u32 {
            // [result id, sampled type, dim, depth, arrayed, ms, sampled,
            // format, (access qualifier)] — Sampled = 2 means storage image.
            if operands.len() >= 8 {
                image_sampled.insert(operands[0], operands[6]);
            }
        }

        offset += word_count;
    }

    let mut entry_points: Vec<EntryPointReflection> = entry_points
        .into_iter()
        .map(|(id, name, stage)| EntryPointReflection {
            name,
            stage,
            workgroup_size: workgroup_sizes.get(&id).copied(),
        })
        .collect();
    entry_points.sort();
    let is_storage_image = |id: Word| {
        variable_types
            .get(&id)
            .and_then(|pointer| pointer_pointees.get(pointer))
            .and_then(|pointee| image_sampled.get(pointee))
            == Some(&2)
    };
    let mut bindings: Vec<BindingReflection> = binding_slots
        .iter()
        .map(|(&id, &binding)| BindingReflection {
            set: sets.get(&id).copied().unwrap_or(0),
            binding,
            storage_class: storage_classes
                .get(&id)
                .map(|&class| enum_name(StorageClass::from_u32(class), class))
                .unwrap_or_else(|| "Unknown".to_owned()),
            storage_image: is_storage_image(id),
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

    /// Assemble one instruction: length+opcode word, then operands.
    fn inst(opcode: Op, operands: &[u32]) -> Vec<u32> {
        let mut words = vec![(((operands.len() + 1) as u32) << 16) | opcode as u32];
        words.extend_from_slice(operands);
        words
    }

    /// A SPIR-V literal string: NUL-terminated UTF-8 packed into words.
    fn string_words(text: &str) -> Vec<u32> {
        let mut bytes = text.as_bytes().to_vec();
        bytes.push(0);
        while !bytes.len().is_multiple_of(4) {
            bytes.push(0);
        }
        bytes
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("4-byte chunk")))
            .collect()
    }

    fn module(instructions: &[Vec<u32>]) -> Vec<u32> {
        let mut words = vec![SPIRV_MAGIC, 0x0001_0600, 0, 100, 0];
        for instruction in instructions {
            words.extend_from_slice(instruction);
        }
        words
    }

    /// A minimal compute module: `cs_main` with `LocalSize 8 8 1` writing a
    /// storage image (`RWTexture2D`-shaped: OpTypeImage with Sampled = 2)
    /// named "fog" at binding 2, plus a combined-sampler binding at 0.
    fn compute_module() -> Vec<u32> {
        let mut entry = vec![ExecutionModel::GLCompute as u32, 1];
        entry.extend(string_words("cs_main"));
        let mut fog_name = vec![12];
        fog_name.extend(string_words("fog"));
        let mut depth_name = vec![22];
        depth_name.extend(string_words("game_depth"));
        module(&[
            inst(Op::EntryPoint, &entry),
            inst(
                Op::ExecutionMode,
                &[1, ExecutionMode::LocalSize as u32, 8, 8, 1],
            ),
            inst(Op::Name, &fog_name),
            inst(Op::Name, &depth_name),
            inst(Op::Decorate, &[12, Decoration::DescriptorSet as u32, 0]),
            inst(Op::Decorate, &[12, Decoration::Binding as u32, 2]),
            inst(Op::Decorate, &[22, Decoration::DescriptorSet as u32, 0]),
            inst(Op::Decorate, &[22, Decoration::Binding as u32, 0]),
            // %10 = storage image type (Sampled = 2); %11 = pointer; %12 = var.
            inst(Op::TypeImage, &[10, 9, 1, 0, 0, 0, 2, 0]),
            inst(
                Op::TypePointer,
                &[11, StorageClass::UniformConstant as u32, 10],
            ),
            inst(
                Op::Variable,
                &[11, 12, StorageClass::UniformConstant as u32],
            ),
            // %20 = sampled image type (Sampled = 1) wrapped in
            // OpTypeSampledImage %21; %23 = pointer; %22 = var.
            inst(Op::TypeImage, &[20, 9, 1, 0, 0, 0, 1, 0]),
            inst(Op::TypeSampledImage, &[21, 20]),
            inst(
                Op::TypePointer,
                &[23, StorageClass::UniformConstant as u32, 21],
            ),
            inst(
                Op::Variable,
                &[23, 22, StorageClass::UniformConstant as u32],
            ),
        ])
    }

    #[test]
    fn reflects_compute_workgroup_size_and_storage_images() {
        let reflection = reflect_spirv(&compute_module()).expect("synthetic module reflects");
        assert_eq!(
            reflection.entry_points,
            vec![EntryPointReflection {
                name: "cs_main".to_owned(),
                stage: "GLCompute".to_owned(),
                workgroup_size: Some([8, 8, 1]),
            }]
        );
        assert_eq!(
            reflection.bindings,
            vec![
                BindingReflection {
                    set: 0,
                    binding: 0,
                    storage_class: "UniformConstant".to_owned(),
                    storage_image: false,
                    name: Some("game_depth".to_owned()),
                },
                BindingReflection {
                    set: 0,
                    binding: 2,
                    storage_class: "UniformConstant".to_owned(),
                    storage_image: true,
                    name: Some("fog".to_owned()),
                },
            ]
        );
    }

    #[test]
    fn graphics_entry_points_have_no_workgroup_size() {
        let mut entry = vec![ExecutionModel::Fragment as u32, 1];
        entry.extend(string_words("fs_main"));
        let reflection =
            reflect_spirv(&module(&[inst(Op::EntryPoint, &entry)])).expect("module reflects");
        assert_eq!(reflection.entry_points.len(), 1);
        assert_eq!(reflection.entry_points[0].workgroup_size, None);
    }

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
