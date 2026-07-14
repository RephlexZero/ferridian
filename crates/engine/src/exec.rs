//! Wiring a loaded pack for execution: which resource feeds each shader
//! binding, and what a device executor must create (M4 seam).
//!
//! The correspondence is by *name*: a pass binds its declared inputs as
//! `Sampler2D <input name>` (see `packs/reference`), and the manifest is the
//! contract. A binding naming anything other than a declared input is an
//! error; so is a declared input the shader never binds — it would put false
//! edges in the pass graph. Modules are re-reflected here rather than
//! trusting the artifact's `reflection.toml`, which could disagree with the
//! SPIR-V it sits next to.
//!
//! Pure logic, no Vulkan: the device half is `ferridian-vk-rt`'s executor.

use std::collections::{BTreeMap, BTreeSet};

use ferridian_pack_format::{
    EntryPointReflection, PackManifest, PassKind as ManifestPassKind, ShaderReflection,
    reflect_spirv,
};

use crate::ResourceId;
use crate::pack::LoadedPack;

/// The final render target every executable pack must write.
pub const SWAPCHAIN: &str = "swapchain";

/// Builtins a pack may read but the executor never creates — the engine
/// captures them from the game.
pub const EXTERNAL_INPUTS: [&str; 2] = ["game_color", "game_depth"];

/// The one uniform block packs may bind: the contract's per-frame camera
/// (`ferridian_contract::CameraUniforms`, std140). It is engine state, not a
/// graph resource — passes bind it as `ConstantBuffer<Camera> camera` without
/// declaring it in `inputs`, and the executor owns the buffer.
pub const CAMERA: &str = "camera";

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum WireError {
    #[error("pass {pass}: module failed reflection: {message}")]
    Reflection { pass: String, message: String },
    #[error("pass {pass}: no reflection supplied")]
    MissingReflection { pass: String },
    #[error("pass {pass}: compute passes are not executable yet")]
    ComputeUnsupported { pass: String },
    #[error("pass {pass}: declares {count} outputs; execution supports exactly one")]
    OutputCount { pass: String, count: usize },
    #[error("pass {pass}: writes {resource:?}, a builtin the game owns")]
    WritesReadOnlyBuiltin { pass: String, resource: String },
    #[error("no pass writes {SWAPCHAIN:?}; the pack would render nothing")]
    NoSwapchainWriter,
    #[error("pass {pass}: reads {SWAPCHAIN:?}, which packs can only write")]
    SwapchainRead { pass: String },
    #[error("pass {pass}: expected exactly one {stage} entry point, found {count}")]
    StageEntryPoints {
        pass: String,
        stage: &'static str,
        count: usize,
    },
    #[error("pass {pass}: binding {binding} has no name to wire by")]
    UnnamedBinding { pass: String, binding: u32 },
    #[error("pass {pass}: binding {name:?} uses descriptor set {set}; only set 0 is supported")]
    UnsupportedSet {
        pass: String,
        name: String,
        set: u32,
    },
    #[error(
        "pass {pass}: binding {name:?} is {storage_class}; only combined image samplers (UniformConstant) and the {CAMERA:?} uniform block are wireable yet"
    )]
    UnsupportedStorageClass {
        pass: String,
        name: String,
        storage_class: String,
    },
    #[error(
        "pass {pass}: uniform block {name:?} is not a builtin; only the contract {CAMERA:?} block is wireable yet"
    )]
    UnknownUniformBlock { pass: String, name: String },
    #[error("pass {pass}: the {CAMERA:?} block is bound twice")]
    DuplicateCameraBinding { pass: String },
    #[error("pass {pass}: descriptor slot {binding} is bound twice")]
    DuplicateBindingSlot { pass: String, binding: u32 },
    #[error("pass {pass}: binding {name:?} matches none of the pass's declared inputs")]
    UnknownBindingResource { pass: String, name: String },
    #[error("pass {pass}: declared input {resource:?} is never bound by the shader")]
    UnboundInput { pass: String, resource: String },
}

/// One descriptor of a pass, resolved to the resource that feeds it.
/// Set is always 0 (enforced during planning).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BindingPlan {
    pub binding: u32,
    pub resource: ResourceId,
}

/// Everything the executor needs to run one graphics pass: entry points,
/// wired descriptors, and the single color target it writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassPlan {
    pub name: String,
    pub vertex_entry: String,
    pub fragment_entry: String,
    /// Sorted by binding slot.
    pub bindings: Vec<BindingPlan>,
    /// Descriptor slot of the [`CAMERA`] uniform block, when the pass binds
    /// it. Never collides with a sampler slot in `bindings`.
    pub camera_binding: Option<u32>,
    pub output: ResourceId,
}

/// A wired, executable schedule derived from a [`LoadedPack`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExecutionPlan {
    /// Passes in execution order (writers before readers).
    pub passes: Vec<PassPlan>,
    /// Resources the executor must create, in first-write order. Never
    /// contains builtins; single-writer is already guaranteed by the graph.
    pub intermediates: Vec<ResourceId>,
    /// Builtins the pack samples — the caller must supply these images.
    pub external_inputs: Vec<ResourceId>,
}

/// Reflect every module of a loaded pack and wire it for execution.
pub fn plan_execution(pack: &LoadedPack) -> Result<ExecutionPlan, WireError> {
    let mut reflections = BTreeMap::new();
    for (name, words) in &pack.modules {
        let reflection = reflect_spirv(words).map_err(|message| WireError::Reflection {
            pass: name.clone(),
            message,
        })?;
        reflections.insert(name.clone(), reflection);
    }
    plan_with_reflections(&pack.manifest, &pack.execution_order, &reflections)
}

/// Wire a manifest against per-pass reflections. Split out from
/// [`plan_execution`] so the wiring rules are testable without SPIR-V.
pub fn plan_with_reflections(
    manifest: &PackManifest,
    execution_order: &[usize],
    reflections: &BTreeMap<String, ShaderReflection>,
) -> Result<ExecutionPlan, WireError> {
    let mut passes = Vec::with_capacity(execution_order.len());
    let mut intermediates = Vec::new();
    let mut external_inputs = BTreeSet::new();
    let mut writes_swapchain = false;

    for &index in execution_order {
        let pass = &manifest.passes[index];
        let err_pass = || pass.name.clone();
        if pass.kind != ManifestPassKind::Graphics {
            return Err(WireError::ComputeUnsupported { pass: err_pass() });
        }
        if pass.inputs.iter().any(|input| input == SWAPCHAIN) {
            return Err(WireError::SwapchainRead { pass: err_pass() });
        }
        if pass.outputs.len() != 1 {
            return Err(WireError::OutputCount {
                pass: err_pass(),
                count: pass.outputs.len(),
            });
        }
        let output = pass.outputs[0].as_str();
        if output == SWAPCHAIN {
            writes_swapchain = true;
        } else if EXTERNAL_INPUTS.contains(&output) {
            return Err(WireError::WritesReadOnlyBuiltin {
                pass: err_pass(),
                resource: output.to_owned(),
            });
        } else {
            intermediates.push(ResourceId(output.to_owned()));
        }

        let reflection = reflections
            .get(&pass.name)
            .ok_or_else(|| WireError::MissingReflection { pass: err_pass() })?;
        let vertex_entry = single_stage_entry(reflection, "Vertex", &pass.name)?;
        let fragment_entry = single_stage_entry(reflection, "Fragment", &pass.name)?;

        let mut bindings = Vec::with_capacity(reflection.bindings.len());
        let mut camera_binding = None;
        let mut used_slots = BTreeSet::new();
        let mut bound_inputs = BTreeSet::new();
        for reflected in &reflection.bindings {
            let name = reflected
                .name
                .as_deref()
                .ok_or_else(|| WireError::UnnamedBinding {
                    pass: err_pass(),
                    binding: reflected.binding,
                })?;
            if reflected.set != 0 {
                return Err(WireError::UnsupportedSet {
                    pass: err_pass(),
                    name: name.to_owned(),
                    set: reflected.set,
                });
            }
            match reflected.storage_class.as_str() {
                "UniformConstant" => {}
                // A uniform block: engine state, not a graph resource — only
                // the contract camera exists, and it bypasses the
                // declared-inputs check below.
                "Uniform" => {
                    if name != CAMERA {
                        return Err(WireError::UnknownUniformBlock {
                            pass: err_pass(),
                            name: name.to_owned(),
                        });
                    }
                    if camera_binding.is_some() {
                        return Err(WireError::DuplicateCameraBinding { pass: err_pass() });
                    }
                    if !used_slots.insert(reflected.binding) {
                        return Err(WireError::DuplicateBindingSlot {
                            pass: err_pass(),
                            binding: reflected.binding,
                        });
                    }
                    camera_binding = Some(reflected.binding);
                    continue;
                }
                _ => {
                    return Err(WireError::UnsupportedStorageClass {
                        pass: err_pass(),
                        name: name.to_owned(),
                        storage_class: reflected.storage_class.clone(),
                    });
                }
            }
            if !used_slots.insert(reflected.binding) {
                return Err(WireError::DuplicateBindingSlot {
                    pass: err_pass(),
                    binding: reflected.binding,
                });
            }
            if !pass.inputs.iter().any(|input| input == name) {
                return Err(WireError::UnknownBindingResource {
                    pass: err_pass(),
                    name: name.to_owned(),
                });
            }
            bound_inputs.insert(name.to_owned());
            if EXTERNAL_INPUTS.contains(&name) {
                external_inputs.insert(name.to_owned());
            }
            bindings.push(BindingPlan {
                binding: reflected.binding,
                resource: ResourceId(name.to_owned()),
            });
        }
        for input in &pass.inputs {
            if !bound_inputs.contains(input) {
                return Err(WireError::UnboundInput {
                    pass: err_pass(),
                    resource: input.clone(),
                });
            }
        }
        bindings.sort_by_key(|binding| binding.binding);

        passes.push(PassPlan {
            name: pass.name.clone(),
            vertex_entry,
            fragment_entry,
            bindings,
            camera_binding,
            output: ResourceId(output.to_owned()),
        });
    }

    if !writes_swapchain {
        return Err(WireError::NoSwapchainWriter);
    }
    Ok(ExecutionPlan {
        passes,
        intermediates,
        external_inputs: external_inputs.into_iter().map(ResourceId).collect(),
    })
}

fn single_stage_entry(
    reflection: &ShaderReflection,
    stage: &'static str,
    pass: &str,
) -> Result<String, WireError> {
    let matches: Vec<&EntryPointReflection> = reflection
        .entry_points
        .iter()
        .filter(|entry| entry.stage == stage)
        .collect();
    match matches.as_slice() {
        [only] => Ok(only.name.clone()),
        _ => Err(WireError::StageEntryPoints {
            pass: pass.to_owned(),
            stage,
            count: matches.len(),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferridian_pack_format::{BindingReflection, EntryPointReflection};

    fn manifest(passes: &str) -> PackManifest {
        PackManifest::from_toml_str(&format!(
            "[pack]\nname = \"t\"\nversion = \"0.1.0\"\n{passes}"
        ))
        .expect("test manifest is valid")
    }

    /// A reference-shaped manifest: shadows -> deferred -> composite.
    fn reference_shaped() -> PackManifest {
        manifest(
            "[[pass]]\nname = \"shadows\"\nkind = \"graphics\"\nshader = \"s.slang\"\n\
             inputs = [\"game_depth\"]\noutputs = [\"shadow_mask\"]\n\
             [[pass]]\nname = \"deferred\"\nkind = \"graphics\"\nshader = \"d.slang\"\n\
             inputs = [\"game_color\", \"game_depth\", \"shadow_mask\"]\noutputs = [\"lit\"]\n\
             [[pass]]\nname = \"composite\"\nkind = \"graphics\"\nshader = \"c.slang\"\n\
             inputs = [\"lit\"]\noutputs = [\"swapchain\"]\n",
        )
    }

    fn stages() -> Vec<EntryPointReflection> {
        vec![
            EntryPointReflection {
                name: "vs_main".to_owned(),
                stage: "Vertex".to_owned(),
            },
            EntryPointReflection {
                name: "fs_main".to_owned(),
                stage: "Fragment".to_owned(),
            },
        ]
    }

    fn sampler(set: u32, binding: u32, name: &str) -> BindingReflection {
        BindingReflection {
            set,
            binding,
            storage_class: "UniformConstant".to_owned(),
            name: Some(name.to_owned()),
        }
    }

    fn uniform(binding: u32, name: &str) -> BindingReflection {
        BindingReflection {
            set: 0,
            binding,
            storage_class: "Uniform".to_owned(),
            name: Some(name.to_owned()),
        }
    }

    fn reflection(bindings: Vec<BindingReflection>) -> ShaderReflection {
        ShaderReflection {
            entry_points: stages(),
            bindings,
        }
    }

    fn reference_reflections() -> BTreeMap<String, ShaderReflection> {
        BTreeMap::from([
            (
                "shadows".to_owned(),
                reflection(vec![sampler(0, 0, "game_depth")]),
            ),
            (
                "deferred".to_owned(),
                reflection(vec![
                    sampler(0, 0, "game_color"),
                    sampler(0, 1, "game_depth"),
                    sampler(0, 2, "shadow_mask"),
                ]),
            ),
            (
                "composite".to_owned(),
                reflection(vec![sampler(0, 0, "lit")]),
            ),
        ])
    }

    fn plan(
        manifest: &PackManifest,
        reflections: &BTreeMap<String, ShaderReflection>,
    ) -> Result<ExecutionPlan, WireError> {
        let order: Vec<usize> = (0..manifest.passes.len()).collect();
        plan_with_reflections(manifest, &order, reflections)
    }

    #[test]
    fn wires_a_reference_shaped_pack() {
        let plan = plan(&reference_shaped(), &reference_reflections()).unwrap();
        assert_eq!(plan.passes.len(), 3);
        assert_eq!(
            plan.intermediates,
            vec![ResourceId("shadow_mask".into()), ResourceId("lit".into())]
        );
        assert_eq!(
            plan.external_inputs,
            vec![
                ResourceId("game_color".into()),
                ResourceId("game_depth".into())
            ]
        );
        let deferred = &plan.passes[1];
        assert_eq!(deferred.vertex_entry, "vs_main");
        assert_eq!(deferred.fragment_entry, "fs_main");
        assert_eq!(
            deferred
                .bindings
                .iter()
                .map(|b| (b.binding, b.resource.0.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "game_color"), (1, "game_depth"), (2, "shadow_mask")]
        );
        assert_eq!(plan.passes[2].output, ResourceId("swapchain".into()));
    }

    #[test]
    fn wires_the_camera_block_without_declaring_it_as_an_input() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "shadows".to_owned(),
            reflection(vec![sampler(0, 0, "game_depth"), uniform(7, "camera")]),
        );
        let plan = plan(&reference_shaped(), &reflections).unwrap();
        assert_eq!(plan.passes[0].camera_binding, Some(7));
        // Passes that don't bind it stay camera-free.
        assert_eq!(plan.passes[1].camera_binding, None);
        // The camera never appears as a wired image resource.
        assert!(
            plan.passes[0]
                .bindings
                .iter()
                .all(|binding| binding.resource.0 != CAMERA)
        );
    }

    #[test]
    fn rejects_uniform_blocks_that_are_not_the_camera() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "shadows".to_owned(),
            reflection(vec![
                sampler(0, 0, "game_depth"),
                uniform(7, "scene_uniforms"),
            ]),
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::UnknownUniformBlock {
                pass: "shadows".to_owned(),
                name: "scene_uniforms".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_camera_bound_twice_or_on_a_taken_slot() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "shadows".to_owned(),
            reflection(vec![
                sampler(0, 0, "game_depth"),
                uniform(6, "camera"),
                uniform(7, "camera"),
            ]),
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::DuplicateCameraBinding {
                pass: "shadows".to_owned(),
            })
        );

        let mut reflections = reference_reflections();
        reflections.insert(
            "shadows".to_owned(),
            reflection(vec![sampler(0, 0, "game_depth"), uniform(0, "camera")]),
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::DuplicateBindingSlot {
                pass: "shadows".to_owned(),
                binding: 0,
            })
        );
    }

    #[test]
    fn rejects_binding_names_outside_declared_inputs() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "shadows".to_owned(),
            reflection(vec![sampler(0, 0, "game_color")]),
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::UnknownBindingResource {
                pass: "shadows".to_owned(),
                name: "game_color".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_declared_inputs_the_shader_never_binds() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "deferred".to_owned(),
            reflection(vec![
                sampler(0, 0, "game_color"),
                sampler(0, 2, "shadow_mask"),
            ]),
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::UnboundInput {
                pass: "deferred".to_owned(),
                resource: "game_depth".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_hostile_binding_shapes() {
        let base = reference_shaped();
        for (bad, expected) in [
            (
                BindingReflection {
                    name: None,
                    ..sampler(0, 0, "")
                },
                WireError::UnnamedBinding {
                    pass: "shadows".to_owned(),
                    binding: 0,
                },
            ),
            (
                sampler(1, 0, "game_depth"),
                WireError::UnsupportedSet {
                    pass: "shadows".to_owned(),
                    name: "game_depth".to_owned(),
                    set: 1,
                },
            ),
            (
                BindingReflection {
                    storage_class: "StorageBuffer".to_owned(),
                    ..sampler(0, 0, "game_depth")
                },
                WireError::UnsupportedStorageClass {
                    pass: "shadows".to_owned(),
                    name: "game_depth".to_owned(),
                    storage_class: "StorageBuffer".to_owned(),
                },
            ),
        ] {
            let mut reflections = reference_reflections();
            reflections.insert("shadows".to_owned(), reflection(vec![bad]));
            assert_eq!(plan(&base, &reflections), Err(expected));
        }
    }

    #[test]
    fn rejects_duplicate_descriptor_slots() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "deferred".to_owned(),
            reflection(vec![
                sampler(0, 0, "game_color"),
                sampler(0, 0, "game_depth"),
                sampler(0, 2, "shadow_mask"),
            ]),
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::DuplicateBindingSlot {
                pass: "deferred".to_owned(),
                binding: 0,
            })
        );
    }

    #[test]
    fn rejects_modules_without_both_stages() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "composite".to_owned(),
            ShaderReflection {
                entry_points: vec![EntryPointReflection {
                    name: "vs_main".to_owned(),
                    stage: "Vertex".to_owned(),
                }],
                bindings: vec![sampler(0, 0, "lit")],
            },
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::StageEntryPoints {
                pass: "composite".to_owned(),
                stage: "Fragment",
                count: 0,
            })
        );
    }

    #[test]
    fn rejects_unexecutable_manifest_shapes() {
        // Compute pass.
        let compute = manifest(
            "[[pass]]\nname = \"lighting\"\nkind = \"compute\"\nshader = \"l.slang\"\n\
             inputs = [\"game_color\"]\noutputs = [\"swapchain\"]\n",
        );
        assert_eq!(
            plan(&compute, &BTreeMap::new()),
            Err(WireError::ComputeUnsupported {
                pass: "lighting".to_owned(),
            })
        );

        // Nothing writes the swapchain.
        let headless = manifest(
            "[[pass]]\nname = \"shadows\"\nkind = \"graphics\"\nshader = \"s.slang\"\n\
             inputs = [\"game_depth\"]\noutputs = [\"shadow_mask\"]\n",
        );
        let reflections = BTreeMap::from([(
            "shadows".to_owned(),
            reflection(vec![sampler(0, 0, "game_depth")]),
        )]);
        assert_eq!(
            plan(&headless, &reflections),
            Err(WireError::NoSwapchainWriter)
        );

        // Reading the swapchain.
        let feedback = manifest(
            "[[pass]]\nname = \"echo\"\nkind = \"graphics\"\nshader = \"e.slang\"\n\
             inputs = [\"swapchain\"]\noutputs = [\"swapchain\"]\n",
        );
        assert!(matches!(
            plan(&feedback, &BTreeMap::new()),
            Err(WireError::SwapchainRead { .. })
        ));
    }
}
