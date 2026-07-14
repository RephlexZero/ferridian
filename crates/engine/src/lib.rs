//! Frame orchestration and the pack pass graph.
//!
//! This crate is deliberately free of Vulkan initialization: it holds the pure
//! logic (graph construction, scheduling, validation) so unit tests and Miri
//! reach as much of the engine as physics allows. Device work lives in
//! `ferridian-vk-rt`; interception lives in `ferridian-vk-layer`.

pub mod frame;

use std::collections::{BTreeMap, BTreeSet};

/// A resource name flowing between passes ("game_color", "shadow_map", …).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ResourceId(pub String);

impl<T: Into<String>> From<T> for ResourceId {
    fn from(value: T) -> Self {
        ResourceId(value.into())
    }
}

/// Resources the engine itself provides to a pack (captured from the game).
pub const BUILTIN_RESOURCES: [&str; 3] = ["game_color", "game_depth", "swapchain"];

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PassKind {
    Graphics,
    Compute,
    Transfer,
}

/// One node in a pack's pass graph.
#[derive(Debug, Clone)]
pub struct PassNode {
    pub name: String,
    pub kind: PassKind,
    pub reads: Vec<ResourceId>,
    pub writes: Vec<ResourceId>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum GraphError {
    #[error("duplicate pass name: {0}")]
    DuplicatePass(String),
    #[error("pass {pass} reads {resource:?} which no pass writes and is not a builtin")]
    DanglingRead { pass: String, resource: String },
    #[error("resource {resource:?} is written by multiple passes: {first} and {second}")]
    MultipleWriters {
        resource: String,
        first: String,
        second: String,
    },
    #[error("pass graph contains a cycle involving pass {0}")]
    Cycle(String),
}

/// A validated, schedulable pass graph.
#[derive(Debug, Default)]
pub struct PassGraph {
    nodes: Vec<PassNode>,
}

impl PassGraph {
    pub fn new() -> PassGraph {
        PassGraph::default()
    }

    pub fn add_pass(&mut self, node: PassNode) -> &mut Self {
        self.nodes.push(node);
        self
    }

    pub fn passes(&self) -> &[PassNode] {
        &self.nodes
    }

    /// Validate the graph and return pass indices in execution order
    /// (writers before readers, ties broken by insertion order).
    pub fn execution_order(&self) -> Result<Vec<usize>, GraphError> {
        let mut seen = BTreeSet::new();
        for node in &self.nodes {
            if !seen.insert(node.name.as_str()) {
                return Err(GraphError::DuplicatePass(node.name.clone()));
            }
        }

        let mut writer_of: BTreeMap<&str, usize> = BTreeMap::new();
        for (index, node) in self.nodes.iter().enumerate() {
            for written in &node.writes {
                if let Some(&first) = writer_of.get(written.0.as_str()) {
                    return Err(GraphError::MultipleWriters {
                        resource: written.0.clone(),
                        first: self.nodes[first].name.clone(),
                        second: node.name.clone(),
                    });
                }
                writer_of.insert(written.0.as_str(), index);
            }
        }

        let mut dependencies: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (index, node) in self.nodes.iter().enumerate() {
            for read in &node.reads {
                match writer_of.get(read.0.as_str()) {
                    Some(&writer) => dependencies[index].push(writer),
                    None if BUILTIN_RESOURCES.contains(&read.0.as_str()) => {}
                    None => {
                        return Err(GraphError::DanglingRead {
                            pass: node.name.clone(),
                            resource: read.0.clone(),
                        });
                    }
                }
            }
        }

        // Kahn's algorithm, preferring lower insertion index for determinism.
        let mut in_degree: Vec<usize> = dependencies.iter().map(Vec::len).collect();
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); self.nodes.len()];
        for (index, deps) in dependencies.iter().enumerate() {
            for &dep in deps {
                dependents[dep].push(index);
            }
        }
        let mut ready: BTreeSet<usize> = in_degree
            .iter()
            .enumerate()
            .filter(|&(_, &degree)| degree == 0)
            .map(|(index, _)| index)
            .collect();
        let mut order = Vec::with_capacity(self.nodes.len());
        while let Some(&next) = ready.iter().next() {
            ready.remove(&next);
            order.push(next);
            for &dependent in &dependents[next] {
                in_degree[dependent] -= 1;
                if in_degree[dependent] == 0 {
                    ready.insert(dependent);
                }
            }
        }
        if order.len() != self.nodes.len() {
            let stuck = in_degree
                .iter()
                .position(|&degree| degree > 0)
                .expect("cycle implies a node with unresolved dependencies");
            return Err(GraphError::Cycle(self.nodes[stuck].name.clone()));
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pass(name: &str, reads: &[&str], writes: &[&str]) -> PassNode {
        PassNode {
            name: name.to_owned(),
            kind: PassKind::Graphics,
            reads: reads.iter().map(|&r| r.into()).collect(),
            writes: writes.iter().map(|&w| w.into()).collect(),
        }
    }

    #[test]
    fn orders_linear_chain() {
        let mut graph = PassGraph::new();
        graph
            .add_pass(pass("composite", &["lit"], &["swapchain"]))
            .add_pass(pass("lighting", &["game_color"], &["lit"]));
        // "composite" was inserted first but depends on "lighting".
        assert_eq!(graph.execution_order().unwrap(), vec![1, 0]);
    }

    #[test]
    fn orders_diamond_deterministically() {
        let mut graph = PassGraph::new();
        graph
            .add_pass(pass("gbuffer", &["game_color"], &["albedo"]))
            .add_pass(pass("ssao", &["albedo"], &["ao"]))
            .add_pass(pass("ssr", &["albedo"], &["reflections"]))
            .add_pass(pass("composite", &["ao", "reflections"], &["swapchain"]));
        assert_eq!(graph.execution_order().unwrap(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn rejects_dangling_read() {
        let mut graph = PassGraph::new();
        graph.add_pass(pass("composite", &["missing"], &["swapchain"]));
        assert_eq!(
            graph.execution_order(),
            Err(GraphError::DanglingRead {
                pass: "composite".to_owned(),
                resource: "missing".to_owned(),
            })
        );
    }

    #[test]
    fn rejects_cycle() {
        let mut graph = PassGraph::new();
        graph
            .add_pass(pass("a", &["b_out"], &["a_out"]))
            .add_pass(pass("b", &["a_out"], &["b_out"]));
        assert!(matches!(graph.execution_order(), Err(GraphError::Cycle(_))));
    }

    #[test]
    fn rejects_duplicate_pass_names() {
        let mut graph = PassGraph::new();
        graph
            .add_pass(pass("composite", &[], &["a"]))
            .add_pass(pass("composite", &[], &["b"]));
        assert_eq!(
            graph.execution_order(),
            Err(GraphError::DuplicatePass("composite".to_owned()))
        );
    }

    #[test]
    fn rejects_multiple_writers() {
        let mut graph = PassGraph::new();
        graph
            .add_pass(pass("first", &[], &["shared"]))
            .add_pass(pass("second", &[], &["shared"]));
        assert!(matches!(
            graph.execution_order(),
            Err(GraphError::MultipleWriters { .. })
        ));
    }
}
