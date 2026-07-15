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

/// Re-exported so executors take the filter from the same module as the plan.
pub use ferridian_pack_format::Filter;
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
    #[error(
        "pass {pass}: compute passes cannot write {SWAPCHAIN:?} (the game's image lacks storage usage); write an intermediate and composite it with a graphics pass"
    )]
    ComputeSwapchainWrite { pass: String },
    #[error(
        "pass {pass}: compute entry point declares no workgroup size (OpExecutionMode LocalSize)"
    )]
    MissingWorkgroupSize { pass: String },
    #[error("pass {pass}: never binds its output {resource:?} as a storage image")]
    ComputeOutputUnbound { pass: String, resource: String },
    #[error(
        "pass {pass}: binds storage image {name:?}, but only its output {output:?} is writable"
    )]
    StorageImageNotOutput {
        pass: String,
        name: String,
        output: String,
    },
    #[error("pass {pass}: binds storage image {name:?}; graphics passes write through attachments")]
    StorageImageOutsideCompute { pass: String, name: String },
    #[error("pass {pass}: binds its output as a storage image twice")]
    DuplicateOutputBinding { pass: String },
    #[error("pass {pass}: declares no outputs; every pass must write something")]
    NoOutputs { pass: String },
    #[error("pass {pass}: compute passes write exactly one output, this one declares {count}")]
    ComputeOutputCount { pass: String, count: usize },
    #[error(
        "pass {pass}: writes {SWAPCHAIN:?} alongside other outputs; the game's image must be its pass's only target"
    )]
    SwapchainWithOtherOutputs { pass: String },
    #[error(
        "pass {pass}: declares {declared} output(s) but the fragment entry writes locations {locations:?}; outputs map to locations 0..{declared} in manifest order"
    )]
    FragmentOutputMismatch {
        pass: String,
        declared: usize,
        locations: Vec<u32>,
    },
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
    /// How the executor's sampler filters this input (from the manifest's
    /// per-pass `filters` table; nearest when unlisted).
    pub filter: Filter,
}

/// How a pass runs on the device: a fullscreen-triangle graphics pipeline or
/// a compute dispatch covering the frame.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StagePlan {
    Graphics {
        vertex_entry: String,
        fragment_entry: String,
    },
    Compute {
        entry: String,
        /// `OpExecutionMode LocalSize` — the executor dispatches
        /// `ceil(extent / workgroup_size)` groups.
        workgroup_size: [u32; 3],
    },
}

/// Everything the executor needs to run one pass: entry points, wired
/// descriptors, and the targets it writes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PassPlan {
    pub name: String,
    pub stage: StagePlan,
    /// Sampled inputs, sorted by binding slot.
    pub bindings: Vec<BindingPlan>,
    /// Descriptor slot of the [`CAMERA`] uniform block, when the pass binds
    /// it. Never collides with any other slot of the pass.
    pub camera_binding: Option<u32>,
    /// Compute only: the descriptor slot where the single output is bound as
    /// a storage image (graphics passes write through attachments instead).
    pub output_binding: Option<u32>,
    /// What the pass writes, in manifest order. For a graphics pass, index =
    /// color attachment location (checked against the fragment entry's
    /// reflected output locations); a pass writing [`SWAPCHAIN`] writes
    /// nothing else, and a compute pass has exactly one output.
    pub outputs: Vec<ResourceId>,
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
        let is_compute = pass.kind == ManifestPassKind::Compute;
        if pass.inputs.iter().any(|input| input == SWAPCHAIN) {
            return Err(WireError::SwapchainRead { pass: err_pass() });
        }
        if pass.outputs.is_empty() {
            return Err(WireError::NoOutputs { pass: err_pass() });
        }
        if is_compute && pass.outputs.len() != 1 {
            return Err(WireError::ComputeOutputCount {
                pass: err_pass(),
                count: pass.outputs.len(),
            });
        }
        if pass.outputs.iter().any(|output| output == SWAPCHAIN) && pass.outputs.len() != 1 {
            return Err(WireError::SwapchainWithOtherOutputs { pass: err_pass() });
        }
        for output in &pass.outputs {
            if output == SWAPCHAIN {
                if is_compute {
                    return Err(WireError::ComputeSwapchainWrite { pass: err_pass() });
                }
                writes_swapchain = true;
            } else if EXTERNAL_INPUTS.contains(&output.as_str()) {
                return Err(WireError::WritesReadOnlyBuiltin {
                    pass: err_pass(),
                    resource: output.clone(),
                });
            } else {
                intermediates.push(ResourceId(output.clone()));
            }
        }
        // The compute output (bound as a storage image below); unused for
        // graphics, whose outputs are attachments in manifest order.
        let output = pass.outputs[0].as_str();

        let reflection = reflections
            .get(&pass.name)
            .ok_or_else(|| WireError::MissingReflection { pass: err_pass() })?;
        let stage = if is_compute {
            let entry = single_stage_entry(reflection, "GLCompute", &pass.name)?;
            let workgroup_size = entry
                .workgroup_size
                .filter(|size| size.iter().all(|&n| n > 0))
                .ok_or_else(|| WireError::MissingWorkgroupSize { pass: err_pass() })?;
            StagePlan::Compute {
                entry: entry.name.clone(),
                workgroup_size,
            }
        } else {
            let fragment = single_stage_entry(reflection, "Fragment", &pass.name)?;
            // The fragment entry must write attachment locations 0..N —
            // manifest order is the location assignment, and an output the
            // shader never writes would silently stay black.
            let declared: Vec<u32> = (0..pass.outputs.len() as u32).collect();
            if fragment.output_locations != declared {
                return Err(WireError::FragmentOutputMismatch {
                    pass: err_pass(),
                    declared: pass.outputs.len(),
                    locations: fragment.output_locations.clone(),
                });
            }
            StagePlan::Graphics {
                vertex_entry: single_stage_entry(reflection, "Vertex", &pass.name)?
                    .name
                    .clone(),
                fragment_entry: fragment.name.clone(),
            }
        };

        let mut bindings = Vec::with_capacity(reflection.bindings.len());
        let mut camera_binding = None;
        let mut output_binding = None;
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
                // A storage image: the only writable binding, and only a
                // compute pass's declared output may be one — it bypasses
                // the declared-inputs check below.
                "UniformConstant" if reflected.storage_image => {
                    if !is_compute {
                        return Err(WireError::StorageImageOutsideCompute {
                            pass: err_pass(),
                            name: name.to_owned(),
                        });
                    }
                    if name != output {
                        return Err(WireError::StorageImageNotOutput {
                            pass: err_pass(),
                            name: name.to_owned(),
                            output: output.to_owned(),
                        });
                    }
                    if output_binding.is_some() {
                        return Err(WireError::DuplicateOutputBinding { pass: err_pass() });
                    }
                    if !used_slots.insert(reflected.binding) {
                        return Err(WireError::DuplicateBindingSlot {
                            pass: err_pass(),
                            binding: reflected.binding,
                        });
                    }
                    output_binding = Some(reflected.binding);
                    continue;
                }
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
                filter: pass.filters.get(name).copied().unwrap_or_default(),
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
        if is_compute && output_binding.is_none() {
            return Err(WireError::ComputeOutputUnbound {
                pass: err_pass(),
                resource: output.to_owned(),
            });
        }
        bindings.sort_by_key(|binding| binding.binding);

        passes.push(PassPlan {
            name: pass.name.clone(),
            stage,
            bindings,
            camera_binding,
            output_binding,
            outputs: pass.outputs.iter().cloned().map(ResourceId).collect(),
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

fn single_stage_entry<'a>(
    reflection: &'a ShaderReflection,
    stage: &'static str,
    pass: &str,
) -> Result<&'a EntryPointReflection, WireError> {
    let matches: Vec<&EntryPointReflection> = reflection
        .entry_points
        .iter()
        .filter(|entry| entry.stage == stage)
        .collect();
    match matches.as_slice() {
        [only] => Ok(only),
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

    /// Vertex + fragment entries, the fragment writing `locations`.
    fn stages_writing(locations: &[u32]) -> Vec<EntryPointReflection> {
        vec![
            EntryPointReflection {
                name: "vs_main".to_owned(),
                stage: "Vertex".to_owned(),
                workgroup_size: None,
                output_locations: vec![],
            },
            EntryPointReflection {
                name: "fs_main".to_owned(),
                stage: "Fragment".to_owned(),
                workgroup_size: None,
                output_locations: locations.to_vec(),
            },
        ]
    }

    fn stages() -> Vec<EntryPointReflection> {
        stages_writing(&[0])
    }

    fn compute_stage(workgroup_size: Option<[u32; 3]>) -> Vec<EntryPointReflection> {
        vec![EntryPointReflection {
            name: "cs_main".to_owned(),
            stage: "GLCompute".to_owned(),
            workgroup_size,
            output_locations: vec![],
        }]
    }

    fn sampler(set: u32, binding: u32, name: &str) -> BindingReflection {
        BindingReflection {
            set,
            binding,
            storage_class: "UniformConstant".to_owned(),
            storage_image: false,
            name: Some(name.to_owned()),
        }
    }

    fn storage_image(binding: u32, name: &str) -> BindingReflection {
        BindingReflection {
            storage_image: true,
            ..sampler(0, binding, name)
        }
    }

    fn uniform(binding: u32, name: &str) -> BindingReflection {
        BindingReflection {
            set: 0,
            binding,
            storage_class: "Uniform".to_owned(),
            storage_image: false,
            name: Some(name.to_owned()),
        }
    }

    fn reflection(bindings: Vec<BindingReflection>) -> ShaderReflection {
        ShaderReflection {
            entry_points: stages(),
            bindings,
        }
    }

    fn compute_reflection(bindings: Vec<BindingReflection>) -> ShaderReflection {
        ShaderReflection {
            entry_points: compute_stage(Some([8, 8, 1])),
            bindings,
        }
    }

    /// The reference pack's shape after the volumetric conversion: graphics,
    /// then a compute pass writing an intermediate, then graphics composite.
    fn compute_shaped() -> PackManifest {
        manifest(
            "[[pass]]\nname = \"deferred\"\nkind = \"graphics\"\nshader = \"d.slang\"\n\
             inputs = [\"game_color\"]\noutputs = [\"lit\"]\n\
             [[pass]]\nname = \"volumetric\"\nkind = \"compute\"\nshader = \"v.slang\"\n\
             inputs = [\"game_depth\"]\noutputs = [\"fog\"]\n\
             [[pass]]\nname = \"composite\"\nkind = \"graphics\"\nshader = \"c.slang\"\n\
             inputs = [\"lit\", \"fog\"]\noutputs = [\"swapchain\"]\n",
        )
    }

    fn compute_shaped_reflections() -> BTreeMap<String, ShaderReflection> {
        BTreeMap::from([
            (
                "deferred".to_owned(),
                reflection(vec![sampler(0, 0, "game_color")]),
            ),
            (
                "volumetric".to_owned(),
                compute_reflection(vec![
                    sampler(0, 0, "game_depth"),
                    storage_image(1, "fog"),
                    uniform(7, "camera"),
                ]),
            ),
            (
                "composite".to_owned(),
                reflection(vec![sampler(0, 0, "lit"), sampler(0, 1, "fog")]),
            ),
        ])
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
        assert_eq!(
            deferred.stage,
            StagePlan::Graphics {
                vertex_entry: "vs_main".to_owned(),
                fragment_entry: "fs_main".to_owned(),
            }
        );
        assert_eq!(
            deferred
                .bindings
                .iter()
                .map(|b| (b.binding, b.resource.0.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "game_color"), (1, "game_depth"), (2, "shadow_mask")]
        );
        assert_eq!(plan.passes[2].outputs, vec![ResourceId("swapchain".into())]);
    }

    #[test]
    fn wires_declared_filters_and_defaults_to_nearest() {
        let filtered = manifest(
            "[[pass]]\nname = \"composite\"\nkind = \"graphics\"\nshader = \"c.slang\"\n\
             inputs = [\"game_color\", \"game_depth\"]\noutputs = [\"swapchain\"]\n\
             filters = { game_color = \"linear\" }\n",
        );
        let reflections = BTreeMap::from([(
            "composite".to_owned(),
            reflection(vec![
                sampler(0, 0, "game_color"),
                sampler(0, 1, "game_depth"),
            ]),
        )]);
        let plan = plan(&filtered, &reflections).unwrap();
        assert_eq!(
            plan.passes[0]
                .bindings
                .iter()
                .map(|binding| (binding.resource.0.as_str(), binding.filter))
                .collect::<Vec<_>>(),
            vec![
                ("game_color", Filter::Linear),
                ("game_depth", Filter::Nearest)
            ]
        );
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
                    workgroup_size: None,
                    output_locations: vec![],
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

    /// A gbuffer-shaped pack: one pass writes two targets, the composite
    /// consumes both.
    fn mrt_shaped() -> PackManifest {
        manifest(
            "[[pass]]\nname = \"gbuffer\"\nkind = \"graphics\"\nshader = \"g.slang\"\n\
             inputs = [\"game_color\"]\noutputs = [\"albedo\", \"normal\"]\n\
             [[pass]]\nname = \"composite\"\nkind = \"graphics\"\nshader = \"c.slang\"\n\
             inputs = [\"albedo\", \"normal\"]\noutputs = [\"swapchain\"]\n",
        )
    }

    fn mrt_reflections() -> BTreeMap<String, ShaderReflection> {
        BTreeMap::from([
            (
                "gbuffer".to_owned(),
                ShaderReflection {
                    entry_points: stages_writing(&[0, 1]),
                    bindings: vec![sampler(0, 0, "game_color")],
                },
            ),
            (
                "composite".to_owned(),
                reflection(vec![sampler(0, 0, "albedo"), sampler(0, 1, "normal")]),
            ),
        ])
    }

    #[test]
    fn wires_multiple_render_targets_in_manifest_order() {
        let plan = plan(&mrt_shaped(), &mrt_reflections()).unwrap();
        assert_eq!(
            plan.passes[0].outputs,
            vec![ResourceId("albedo".into()), ResourceId("normal".into())]
        );
        assert_eq!(plan.passes[0].output_binding, None);
        assert_eq!(
            plan.intermediates,
            vec![ResourceId("albedo".into()), ResourceId("normal".into())]
        );
    }

    #[test]
    fn rejects_malformed_output_declarations() {
        // The fragment entry writes fewer locations than the pass declares —
        // "normal" would silently stay black.
        let mut reflections = mrt_reflections();
        reflections.insert(
            "gbuffer".to_owned(),
            reflection(vec![sampler(0, 0, "game_color")]),
        );
        assert_eq!(
            plan(&mrt_shaped(), &reflections),
            Err(WireError::FragmentOutputMismatch {
                pass: "gbuffer".to_owned(),
                declared: 2,
                locations: vec![0],
            })
        );

        // And the reverse: a location the pass never declares an output for.
        let mut reflections = reference_reflections();
        reflections.insert(
            "shadows".to_owned(),
            ShaderReflection {
                entry_points: stages_writing(&[0, 1]),
                bindings: vec![sampler(0, 0, "game_depth")],
            },
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::FragmentOutputMismatch {
                pass: "shadows".to_owned(),
                declared: 1,
                locations: vec![0, 1],
            })
        );

        // No outputs at all.
        let headless = manifest(
            "[[pass]]\nname = \"sink\"\nkind = \"graphics\"\nshader = \"s.slang\"\n\
             inputs = [\"game_color\"]\noutputs = []\n",
        );
        assert_eq!(
            plan(&headless, &BTreeMap::new()),
            Err(WireError::NoOutputs {
                pass: "sink".to_owned(),
            })
        );

        // The swapchain sharing a pass with other outputs.
        let greedy = manifest(
            "[[pass]]\nname = \"final\"\nkind = \"graphics\"\nshader = \"f.slang\"\n\
             inputs = [\"game_color\"]\noutputs = [\"swapchain\", \"extra\"]\n",
        );
        assert_eq!(
            plan(&greedy, &BTreeMap::new()),
            Err(WireError::SwapchainWithOtherOutputs {
                pass: "final".to_owned(),
            })
        );

        // A compute pass declaring two outputs.
        let wide = manifest(
            "[[pass]]\nname = \"split\"\nkind = \"compute\"\nshader = \"s.slang\"\n\
             inputs = [\"game_color\"]\noutputs = [\"a\", \"b\"]\n\
             [[pass]]\nname = \"composite\"\nkind = \"graphics\"\nshader = \"c.slang\"\n\
             inputs = [\"a\", \"b\"]\noutputs = [\"swapchain\"]\n",
        );
        assert_eq!(
            plan(&wide, &BTreeMap::new()),
            Err(WireError::ComputeOutputCount {
                pass: "split".to_owned(),
                count: 2,
            })
        );
    }

    #[test]
    fn wires_a_compute_pass_mid_graph() {
        let plan = plan(&compute_shaped(), &compute_shaped_reflections()).unwrap();
        let volumetric = &plan.passes[1];
        assert_eq!(
            volumetric.stage,
            StagePlan::Compute {
                entry: "cs_main".to_owned(),
                workgroup_size: [8, 8, 1],
            }
        );
        // The storage-image output and the camera are wired by slot, not as
        // sampled inputs.
        assert_eq!(volumetric.output_binding, Some(1));
        assert_eq!(volumetric.camera_binding, Some(7));
        assert_eq!(
            volumetric
                .bindings
                .iter()
                .map(|b| (b.binding, b.resource.0.as_str()))
                .collect::<Vec<_>>(),
            vec![(0, "game_depth")]
        );
        // Graphics passes never carry an output binding.
        assert_eq!(plan.passes[0].output_binding, None);
        assert_eq!(
            plan.external_inputs
                .iter()
                .map(|resource| resource.0.as_str())
                .collect::<Vec<_>>(),
            vec!["game_color", "game_depth"]
        );
    }

    #[test]
    fn rejects_malformed_compute_passes() {
        // Writing the swapchain from compute.
        let to_swapchain = manifest(
            "[[pass]]\nname = \"lighting\"\nkind = \"compute\"\nshader = \"l.slang\"\n\
             inputs = [\"game_color\"]\noutputs = [\"swapchain\"]\n",
        );
        assert_eq!(
            plan(&to_swapchain, &BTreeMap::new()),
            Err(WireError::ComputeSwapchainWrite {
                pass: "lighting".to_owned(),
            })
        );

        // No workgroup size on the compute entry.
        let mut reflections = compute_shaped_reflections();
        reflections.insert(
            "volumetric".to_owned(),
            ShaderReflection {
                entry_points: compute_stage(None),
                bindings: vec![sampler(0, 0, "game_depth"), storage_image(1, "fog")],
            },
        );
        assert_eq!(
            plan(&compute_shaped(), &reflections),
            Err(WireError::MissingWorkgroupSize {
                pass: "volumetric".to_owned(),
            })
        );

        // The output never bound as a storage image.
        let mut reflections = compute_shaped_reflections();
        reflections.insert(
            "volumetric".to_owned(),
            compute_reflection(vec![sampler(0, 0, "game_depth")]),
        );
        assert_eq!(
            plan(&compute_shaped(), &reflections),
            Err(WireError::ComputeOutputUnbound {
                pass: "volumetric".to_owned(),
                resource: "fog".to_owned(),
            })
        );

        // A storage image that is not the declared output.
        let mut reflections = compute_shaped_reflections();
        reflections.insert(
            "volumetric".to_owned(),
            compute_reflection(vec![
                sampler(0, 0, "game_depth"),
                storage_image(1, "fog"),
                storage_image(2, "scratch"),
            ]),
        );
        assert_eq!(
            plan(&compute_shaped(), &reflections),
            Err(WireError::StorageImageNotOutput {
                pass: "volumetric".to_owned(),
                name: "scratch".to_owned(),
                output: "fog".to_owned(),
            })
        );

        // The output bound as a storage image twice.
        let mut reflections = compute_shaped_reflections();
        reflections.insert(
            "volumetric".to_owned(),
            compute_reflection(vec![
                sampler(0, 0, "game_depth"),
                storage_image(1, "fog"),
                storage_image(2, "fog"),
            ]),
        );
        assert_eq!(
            plan(&compute_shaped(), &reflections),
            Err(WireError::DuplicateOutputBinding {
                pass: "volumetric".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_storage_images_in_graphics_passes() {
        let mut reflections = reference_reflections();
        reflections.insert(
            "shadows".to_owned(),
            reflection(vec![
                sampler(0, 0, "game_depth"),
                storage_image(1, "shadow_mask"),
            ]),
        );
        assert_eq!(
            plan(&reference_shaped(), &reflections),
            Err(WireError::StorageImageOutsideCompute {
                pass: "shadows".to_owned(),
                name: "shadow_mask".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_unexecutable_manifest_shapes() {
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
