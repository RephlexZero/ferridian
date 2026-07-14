//! Contract-driven classification of the game's render passes.
//!
//! Blaze3D pushes debug groups around its render passes; through the Vulkan
//! backend those surface as `vkCmdBeginDebugUtilsLabelEXT` labels, which the
//! layer intercepts and feeds here. A [`PassObserver`] matches the innermost
//! active label against the contract's `game_anchor`s to decide *which* game
//! pass a `vkCmdBeginRenderPass` is — the seam the compositor keys off.
//!
//! Pure logic, no Vulkan: unit tests and Miri cover all of it.

use ferridian_contract::{Contract, GamePassKind};

/// Label nesting beyond this depth is tracked as a bare counter instead of
/// stored — a buggy (or hostile) app must not grow our memory unboundedly.
const MAX_LABEL_DEPTH: usize = 64;

/// Classifies render-pass begins against a [`Contract`]'s anchors.
///
/// Label begin/end events are per-command-buffer state on the Vulkan side;
/// one observer assumes the single-threaded recording the current test
/// harness (and vanilla's render thread) does.
#[derive(Debug)]
pub struct PassObserver {
    /// `game_anchor` → kind, resolved once at construction.
    anchors: Vec<(String, GamePassKind)>,
    /// Kinds of the currently open labels, innermost last; `None` for labels
    /// matching no anchor.
    label_stack: Vec<Option<GamePassKind>>,
    /// Open labels beyond [`MAX_LABEL_DEPTH`], counted so ends stay balanced.
    overflow_depth: u64,
    counts: [u64; GamePassKind::ALL.len()],
}

impl PassObserver {
    pub fn new(contract: &Contract) -> PassObserver {
        PassObserver {
            anchors: contract
                .passes
                .iter()
                .map(|pass| (pass.game_anchor.clone(), pass.kind))
                .collect(),
            label_stack: Vec::new(),
            overflow_depth: 0,
            counts: [0; GamePassKind::ALL.len()],
        }
    }

    pub fn label_begun(&mut self, name: &str) {
        if self.label_stack.len() >= MAX_LABEL_DEPTH {
            self.overflow_depth += 1;
            return;
        }
        let kind = self
            .anchors
            .iter()
            .find(|(anchor, _)| anchor == name)
            .map(|&(_, kind)| kind);
        self.label_stack.push(kind);
    }

    pub fn label_ended(&mut self) {
        if self.overflow_depth > 0 {
            self.overflow_depth -= 1;
        } else {
            // An unbalanced end (validation would flag it) is ignored rather
            // than allowed to underflow.
            self.label_stack.pop();
        }
    }

    /// Classify a render-pass begin by the innermost anchor-matching label.
    /// No match means [`GamePassKind::Unknown`] — which the engine renders
    /// unmodified, never drops.
    pub fn render_pass_begun(&mut self) -> GamePassKind {
        let kind = self
            .label_stack
            .iter()
            .rev()
            .find_map(|&kind| kind)
            .unwrap_or(GamePassKind::Unknown);
        self.counts[kind_index(kind)] += 1;
        kind
    }

    pub fn count_for(&self, kind: GamePassKind) -> u64 {
        self.counts[kind_index(kind)]
    }
}

/// A kind's index in [`GamePassKind::ALL`] wire order.
pub fn kind_index(kind: GamePassKind) -> usize {
    GamePassKind::ALL
        .iter()
        .position(|&candidate| candidate == kind)
        .expect("GamePassKind::ALL is exhaustive")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observer() -> PassObserver {
        PassObserver::new(&Contract::current())
    }

    /// `Contract::current()` anchors are `todo/<wire name>` placeholders.
    fn anchor(kind: GamePassKind) -> String {
        format!("todo/{}", kind.as_str())
    }

    #[test]
    fn classifies_by_innermost_matching_label() {
        let mut observer = observer();
        observer.label_begun(&anchor(GamePassKind::Terrain));
        observer.label_begun("chunk section 12/40"); // no anchor: inner detail
        assert_eq!(observer.render_pass_begun(), GamePassKind::Terrain);
        observer.label_begun(&anchor(GamePassKind::Entities));
        assert_eq!(observer.render_pass_begun(), GamePassKind::Entities);
        observer.label_ended();
        assert_eq!(observer.render_pass_begun(), GamePassKind::Terrain);
        assert_eq!(observer.count_for(GamePassKind::Terrain), 2);
        assert_eq!(observer.count_for(GamePassKind::Entities), 1);
    }

    #[test]
    fn unlabelled_passes_are_unknown_not_dropped() {
        let mut observer = observer();
        assert_eq!(observer.render_pass_begun(), GamePassKind::Unknown);
        observer.label_begun("some future pass we have no anchor for");
        assert_eq!(observer.render_pass_begun(), GamePassKind::Unknown);
        assert_eq!(observer.count_for(GamePassKind::Unknown), 2);
    }

    #[test]
    fn unbalanced_ends_do_not_underflow() {
        let mut observer = observer();
        observer.label_ended();
        observer.label_ended();
        observer.label_begun(&anchor(GamePassKind::Sky));
        assert_eq!(observer.render_pass_begun(), GamePassKind::Sky);
    }

    #[test]
    fn hostile_label_depth_stays_bounded_and_balanced() {
        // Anything past MAX_LABEL_DEPTH only exercises the saturating counter,
        // so Miri's interpreter gets a flood that is deep enough to prove the
        // bound without spending an hour on identical iterations.
        let flood = if cfg!(miri) {
            MAX_LABEL_DEPTH + 100
        } else {
            1_000_000
        };
        let mut observer = observer();
        observer.label_begun(&anchor(GamePassKind::Gui));
        for i in 0..flood {
            observer.label_begun(&format!("flood {i}"));
        }
        assert!(observer.label_stack.len() <= MAX_LABEL_DEPTH);
        // The GUI label is still the innermost *matching* one…
        assert_eq!(observer.render_pass_begun(), GamePassKind::Gui);
        for _ in 0..flood {
            observer.label_ended();
        }
        // …and exactly balancing the flood leaves it open.
        assert_eq!(observer.render_pass_begun(), GamePassKind::Gui);
        observer.label_ended();
        assert_eq!(observer.render_pass_begun(), GamePassKind::Unknown);
    }

    #[test]
    fn kind_index_matches_wire_order() {
        assert_eq!(kind_index(GamePassKind::Sky), 0);
        assert_eq!(
            kind_index(GamePassKind::Unknown),
            GamePassKind::ALL.len() - 1
        );
    }
}
