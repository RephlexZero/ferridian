//! Reflection is exercised against a module assembled in-memory with rspirv —
//! no binary fixtures in the repo, and the test runs everywhere (no slangc
//! needed). The snapshot is the reviewable contract of what reflection emits.

use ferridian_pack_compiler::reflect_spirv;
use rspirv::binary::Assemble;
use rspirv::dr::Operand;
use rspirv::spirv;

fn assemble_test_module() -> Vec<u32> {
    let mut b = rspirv::dr::Builder::new();
    b.set_version(1, 5);
    b.capability(spirv::Capability::Shader);
    b.memory_model(spirv::AddressingModel::Logical, spirv::MemoryModel::GLSL450);

    // A uniform block at set 0, binding 1, named like packc output would be.
    let float = b.type_float(32);
    let block = b.type_struct([float]);
    let block_ptr = b.type_pointer(None, spirv::StorageClass::Uniform, block);
    let scene_uniforms = b.variable(block_ptr, None, spirv::StorageClass::Uniform, None);
    b.decorate(
        scene_uniforms,
        spirv::Decoration::DescriptorSet,
        [Operand::LiteralBit32(0)],
    );
    b.decorate(
        scene_uniforms,
        spirv::Decoration::Binding,
        [Operand::LiteralBit32(1)],
    );
    b.name(scene_uniforms, "scene_uniforms");

    let void = b.type_void();
    let void_fn = b.type_function(void, []);
    let main_fn = b
        .begin_function(void, None, spirv::FunctionControl::NONE, void_fn)
        .unwrap();
    b.begin_block(None).unwrap();
    b.ret().unwrap();
    b.end_function().unwrap();
    b.entry_point(spirv::ExecutionModel::Fragment, main_fn, "fs_main", []);
    b.execution_mode(main_fn, spirv::ExecutionMode::OriginUpperLeft, []);

    b.module().assemble()
}

#[test]
fn reflects_entry_points_and_bindings() {
    let words = assemble_test_module();
    let reflection = reflect_spirv(&words).expect("assembled module is valid");
    insta::assert_toml_snapshot!(reflection);
}

#[test]
fn rejects_garbage() {
    assert!(reflect_spirv(&[0xdead_beef, 0x0bad_f00d]).is_err());
}
