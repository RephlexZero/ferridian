//! SPIR-V reflection: what the engine needs to bind a compiled module without
//! stringly-typed guessing — entry points and descriptor bindings, derived
//! from the module itself.

use std::collections::BTreeMap;

use rspirv::dr::{Module, Operand};
use rspirv::spirv::{Decoration, Op, StorageClass, Word};
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

/// Parse a SPIR-V module and extract its interface. Returns a human-readable
/// message on malformed input (callers wrap it with pass context).
pub fn reflect_spirv(words: &[u32]) -> Result<ShaderReflection, String> {
    let module = rspirv::dr::load_words(words).map_err(|error| error.to_string())?;
    Ok(reflect_module(&module))
}

fn reflect_module(module: &Module) -> ShaderReflection {
    let mut names: BTreeMap<Word, String> = BTreeMap::new();
    for inst in &module.debug_names {
        if inst.class.opcode == Op::Name
            && let (Some(Operand::IdRef(target)), Some(Operand::LiteralString(name))) =
                (inst.operands.first(), inst.operands.get(1))
        {
            names.insert(*target, name.clone());
        }
    }

    let mut storage_classes: BTreeMap<Word, StorageClass> = BTreeMap::new();
    for inst in &module.types_global_values {
        if inst.class.opcode == Op::Variable
            && let (Some(id), Some(Operand::StorageClass(class))) =
                (inst.result_id, inst.operands.first())
        {
            storage_classes.insert(id, *class);
        }
    }

    let mut sets: BTreeMap<Word, u32> = BTreeMap::new();
    let mut binding_slots: BTreeMap<Word, u32> = BTreeMap::new();
    for inst in &module.annotations {
        if inst.class.opcode != Op::Decorate {
            continue;
        }
        let (
            Some(Operand::IdRef(target)),
            Some(Operand::Decoration(decoration)),
            Some(Operand::LiteralBit32(value)),
        ) = (
            inst.operands.first(),
            inst.operands.get(1),
            inst.operands.get(2),
        )
        else {
            continue;
        };
        match decoration {
            Decoration::DescriptorSet => {
                sets.insert(*target, *value);
            }
            Decoration::Binding => {
                binding_slots.insert(*target, *value);
            }
            _ => {}
        }
    }

    let mut entry_points = Vec::new();
    for inst in &module.entry_points {
        if inst.class.opcode == Op::EntryPoint
            && let (Some(Operand::ExecutionModel(model)), Some(Operand::LiteralString(name))) =
                (inst.operands.first(), inst.operands.get(2))
        {
            entry_points.push(EntryPointReflection {
                name: name.clone(),
                stage: format!("{model:?}"),
            });
        }
    }
    entry_points.sort();

    let mut bindings: Vec<BindingReflection> = binding_slots
        .iter()
        .map(|(&id, &binding)| BindingReflection {
            set: sets.get(&id).copied().unwrap_or(0),
            binding,
            storage_class: storage_classes
                .get(&id)
                .map(|class| format!("{class:?}"))
                .unwrap_or_else(|| "Unknown".to_owned()),
            name: names.get(&id).cloned(),
        })
        .collect();
    bindings.sort();

    ShaderReflection {
        entry_points,
        bindings,
    }
}
