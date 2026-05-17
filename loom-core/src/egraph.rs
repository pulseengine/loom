//! Acyclic E-Graph (ægraph) MVP — v1.0.3 Track 2.
//!
//! This module ships the minimal data structure for a Cranelift-style
//! *acyclic* equality graph (ægraph). Unlike a traditional egg-style
//! e-graph, an ægraph forbids cycles among e-classes: every e-node may
//! only reference e-class ids that strictly precede it in insertion
//! order. This trades the global congruence-closure fixpoint of full
//! equality saturation for a far simpler, per-rewrite-verifiable
//! substrate — which is exactly what LOOM's "provably correct" mission
//! requires.
//!
//! ## Scope of this PR
//!
//! INFRASTRUCTURE ONLY. This module ships:
//!
//! 1. An [`ENode`] sum type covering a small subset of i32 arithmetic
//!    and bitwise ops (enough to demonstrate structural sharing on
//!    typical wasm bodies).
//! 2. An [`EGraph`] that hash-conses e-nodes (so isomorphic terms share
//!    one e-class id) and enforces the acyclic invariant.
//! 3. Conversion helpers between LOOM's [`crate::Instruction`] enum and
//!    e-graph nodes ([`ENode::from_instruction`] / [`EGraph::extract`]).
//!
//! What is intentionally *not* in this PR (future work, see module
//! docs at the bottom):
//!
//! - A `union` / rewrite engine (rewrite rules that merge classes).
//! - A real cost model (we use a node-count proxy in `extract`).
//! - Integration with the optimizer pipeline.
//!
//! ## Soundness invariants
//!
//! - **Acyclicity.** Every [`EClassId`] in `ENode::children` is strictly
//!   less than the id of the e-class containing that node. New e-classes
//!   are appended at the end, so no e-node can reference its own class
//!   or any later class. [`EGraph::add`] enforces this at the API
//!   boundary.
//! - **Structural sharing.** Two e-nodes that are bit-identical
//!   (`Op + children`) hash-cons to the same [`EClassId`]. We do *not*
//!   normalize for algebraic identities (commutativity, identity element,
//!   etc.) — that is the job of a future rewrite engine running on top
//!   of this substrate.
//! - **No external deps.** We use only `std::collections::HashMap` and
//!   roll a tiny inline union-find for the equivalence layer.

use std::collections::HashMap;
use std::fmt;

use crate::Instruction;

/// Opaque identifier for an e-class in the [`EGraph`].
///
/// E-class ids are dense `u32`s allocated in insertion order. The
/// acyclic invariant is enforced by requiring every child id in an
/// [`ENode`] to be strictly less than the id of the class that contains
/// the node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct EClassId(pub u32);

impl fmt::Display for EClassId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{}", self.0)
    }
}

/// Operator discriminator for an e-node.
///
/// This is intentionally a small subset of LOOM's full [`Instruction`]
/// enum — just enough to demonstrate structural sharing on typical wasm
/// arithmetic. The arity of each variant is encoded by the length of
/// the e-node's child vector (validated by [`Op::arity`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Op {
    /// 32-bit constant. Arity 0.
    Const(i32),
    /// 64-bit constant. Arity 0.
    Const64(i64),
    /// `local.get N`. Arity 0.
    LocalGet(u32),
    /// `i32.add`. Arity 2.
    I32Add,
    /// `i32.sub`. Arity 2.
    I32Sub,
    /// `i32.mul`. Arity 2.
    I32Mul,
    /// `i32.and`. Arity 2.
    I32And,
    /// `i32.or`. Arity 2.
    I32Or,
    /// `i32.xor`. Arity 2.
    I32Xor,
    /// `i32.shl`. Arity 2.
    I32Shl,
    /// `i32.shr_s`. Arity 2.
    I32ShrS,
    /// `i32.shr_u`. Arity 2.
    I32ShrU,
    /// `i32.eq`. Arity 2.
    I32Eq,
    /// `i32.eqz`. Arity 1.
    I32Eqz,
}

impl Op {
    /// Expected child count for this operator.
    ///
    /// Used by [`EGraph::add`] to reject malformed e-nodes.
    pub fn arity(&self) -> usize {
        match self {
            Op::Const(_) | Op::Const64(_) | Op::LocalGet(_) => 0,
            Op::I32Eqz => 1,
            Op::I32Add
            | Op::I32Sub
            | Op::I32Mul
            | Op::I32And
            | Op::I32Or
            | Op::I32Xor
            | Op::I32Shl
            | Op::I32ShrS
            | Op::I32ShrU
            | Op::I32Eq => 2,
        }
    }

    /// Convert this operator back to a stack-machine instruction.
    fn to_instruction(self) -> Instruction {
        match self {
            Op::Const(v) => Instruction::I32Const(v),
            Op::Const64(v) => Instruction::I64Const(v),
            Op::LocalGet(idx) => Instruction::LocalGet(idx),
            Op::I32Add => Instruction::I32Add,
            Op::I32Sub => Instruction::I32Sub,
            Op::I32Mul => Instruction::I32Mul,
            Op::I32And => Instruction::I32And,
            Op::I32Or => Instruction::I32Or,
            Op::I32Xor => Instruction::I32Xor,
            Op::I32Shl => Instruction::I32Shl,
            Op::I32ShrS => Instruction::I32ShrS,
            Op::I32ShrU => Instruction::I32ShrU,
            Op::I32Eq => Instruction::I32Eq,
            Op::I32Eqz => Instruction::I32Eqz,
        }
    }
}

/// A single e-node: an operator paired with the e-class ids of its
/// arguments. Two e-nodes that are bit-identical hash-cons to the same
/// [`EClassId`] inside an [`EGraph`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ENode {
    /// The operator at this node.
    pub op: Op,
    /// E-class ids of the arguments, in order. Length must equal
    /// `op.arity()`.
    pub children: Vec<EClassId>,
}

impl ENode {
    /// Construct an e-node directly from an operator + children.
    pub fn new(op: Op, children: Vec<EClassId>) -> Self {
        ENode { op, children }
    }

    /// Convert a LOOM [`Instruction`] (plus pre-resolved child e-class
    /// ids) into an [`ENode`]. Returns `None` for instructions that the
    /// MVP e-graph does not yet model — callers should fall back to the
    /// existing optimizer paths in that case.
    ///
    /// The caller is responsible for popping the appropriate number of
    /// child e-classes off its working stack and passing them here in
    /// **stack order** (i.e. `child_ids[0]` is the deeper / left
    /// operand, `child_ids[1]` is the topmost / right operand for binary
    /// ops). This matches wasm's evaluation order.
    pub fn from_instruction(instr: &Instruction, child_ids: &[EClassId]) -> Option<ENode> {
        let op = match instr {
            Instruction::I32Const(v) => Op::Const(*v),
            Instruction::I64Const(v) => Op::Const64(*v),
            Instruction::LocalGet(idx) => Op::LocalGet(*idx),
            Instruction::I32Add => Op::I32Add,
            Instruction::I32Sub => Op::I32Sub,
            Instruction::I32Mul => Op::I32Mul,
            Instruction::I32And => Op::I32And,
            Instruction::I32Or => Op::I32Or,
            Instruction::I32Xor => Op::I32Xor,
            Instruction::I32Shl => Op::I32Shl,
            Instruction::I32ShrS => Op::I32ShrS,
            Instruction::I32ShrU => Op::I32ShrU,
            Instruction::I32Eq => Op::I32Eq,
            Instruction::I32Eqz => Op::I32Eqz,
            _ => return None,
        };
        if child_ids.len() != op.arity() {
            return None;
        }
        Some(ENode {
            op,
            children: child_ids.to_vec(),
        })
    }
}

/// Acyclic e-graph.
///
/// Each call to [`EGraph::add`] performs hash-consing: an isomorphic
/// e-node already in the graph returns its existing [`EClassId`];
/// otherwise a fresh class is created. There is no `union` API in this
/// MVP — rewriting and equality merging are deferred to a future PR so
/// the substrate can be reviewed in isolation.
#[derive(Debug, Default, Clone)]
pub struct EGraph {
    /// All e-nodes ever inserted, indexed by their [`EClassId`]. For
    /// the MVP we maintain a 1:1 mapping between e-nodes and e-classes
    /// (no class merging), so `nodes[i]` is the canonical representative
    /// of class `EClassId(i)`.
    nodes: Vec<ENode>,
    /// Hash-cons cache: e-node -> the e-class id that first hosted an
    /// isomorphic node. Drives structural sharing.
    cons: HashMap<ENode, EClassId>,
    /// Inline union-find over e-class ids. Currently every class is its
    /// own root; the field exists so a future `union()` PR can plug in
    /// without changing the public API of `add` / `extract`.
    uf: UnionFind,
}

impl EGraph {
    /// Create an empty e-graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of distinct e-classes currently in the graph.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// `true` if the e-graph contains no classes.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look up the e-node that defines a class. Panics if `id` is out
    /// of range — that would indicate the caller fabricated an id we
    /// never handed out, which violates the API contract.
    pub fn node(&self, id: EClassId) -> &ENode {
        &self.nodes[id.0 as usize]
    }

    /// Canonical root of `id` under the union-find. For the MVP every
    /// class is its own root, but callers should still go through this
    /// accessor so a future rewrite engine can transparently introduce
    /// merging.
    pub fn find(&mut self, id: EClassId) -> EClassId {
        self.uf.find(id)
    }

    /// Insert an e-node into the graph.
    ///
    /// Performs the following checks before allocating a class:
    ///
    /// 1. **Arity.** `node.children.len()` must match `node.op.arity()`.
    /// 2. **Bounded children.** Every child id must already exist in the
    ///    graph — i.e., `child.0 < self.len()`. This is what enforces
    ///    acyclicity: a new class is appended at index `self.len()`, so
    ///    no child of the new node can reference itself or any later
    ///    class.
    ///
    /// Returns `Err` if either check fails, or the existing
    /// [`EClassId`] if the node already hash-consed.
    pub fn add(&mut self, node: ENode) -> Result<EClassId, EGraphError> {
        if node.children.len() != node.op.arity() {
            return Err(EGraphError::ArityMismatch {
                op: node.op,
                expected: node.op.arity(),
                actual: node.children.len(),
            });
        }
        let next_id = self.nodes.len() as u32;
        for child in &node.children {
            if child.0 >= next_id {
                // This is the acyclic guard: a child id equal to or
                // greater than the about-to-be-created class would
                // either be self-referential (==) or forward (>),
                // both of which violate the invariant.
                return Err(EGraphError::CycleRejected {
                    new_id: EClassId(next_id),
                    bad_child: *child,
                });
            }
        }

        if let Some(existing) = self.cons.get(&node) {
            return Ok(*existing);
        }

        let id = EClassId(next_id);
        self.nodes.push(node.clone());
        self.cons.insert(node, id);
        self.uf.make_set(id);
        Ok(id)
    }

    /// Linearize the subgraph rooted at `class_id` back into a
    /// stack-machine instruction sequence.
    ///
    /// Extraction is a simple post-order traversal that emits each
    /// child's instructions before its parent operator. This is the
    /// trivial "node-count" cost model — every e-class in the MVP has
    /// exactly one representative node, so there is no choice to make.
    /// When a real rewrite engine lands and classes carry multiple
    /// equivalent nodes, this function will need to pick the
    /// cost-minimal representative; today it just walks the unique
    /// node.
    ///
    /// Returns the emitted instructions in evaluation order (deepest
    /// child first), suitable for direct splicing into a function body.
    pub fn extract(&self, class_id: EClassId) -> Vec<Instruction> {
        let mut out = Vec::new();
        self.extract_into(class_id, &mut out);
        out
    }

    fn extract_into(&self, class_id: EClassId, out: &mut Vec<Instruction>) {
        let node = &self.nodes[class_id.0 as usize];
        for child in &node.children {
            self.extract_into(*child, out);
        }
        out.push(node.op.to_instruction());
    }
}

/// Errors produced by [`EGraph::add`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EGraphError {
    /// The e-node was constructed with the wrong number of children
    /// for its operator.
    ArityMismatch {
        /// The operator whose arity was violated.
        op: Op,
        /// Number of children expected by `op.arity()`.
        expected: usize,
        /// Number of children actually supplied.
        actual: usize,
    },
    /// Inserting this e-node would have created a self-reference or
    /// forward edge, violating the acyclic invariant.
    CycleRejected {
        /// The class id that would have been allocated.
        new_id: EClassId,
        /// The child id that triggered the rejection.
        bad_child: EClassId,
    },
}

impl fmt::Display for EGraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EGraphError::ArityMismatch {
                op,
                expected,
                actual,
            } => write!(
                f,
                "ægraph arity mismatch for {:?}: expected {} children, got {}",
                op, expected, actual
            ),
            EGraphError::CycleRejected { new_id, bad_child } => write!(
                f,
                "ægraph cycle rejected: new class {} cannot reference child {} (would form a cycle)",
                new_id, bad_child
            ),
        }
    }
}

impl std::error::Error for EGraphError {}

/// Minimal inline union-find (path compression + union-by-rank).
///
/// This is intentionally tiny (~30 LOC) to avoid pulling in the
/// `union-find` crate. The MVP doesn't merge classes, but a real
/// rewrite engine in a follow-up PR will, and having the structure in
/// place now lets that PR keep the public surface of [`EGraph`]
/// unchanged.
#[derive(Debug, Default, Clone)]
struct UnionFind {
    parent: Vec<u32>,
    rank: Vec<u8>,
}

impl UnionFind {
    fn make_set(&mut self, id: EClassId) {
        debug_assert_eq!(id.0 as usize, self.parent.len());
        self.parent.push(id.0);
        self.rank.push(0);
    }

    fn find(&mut self, id: EClassId) -> EClassId {
        let mut x = id.0;
        while self.parent[x as usize] != x {
            let p = self.parent[x as usize];
            // Path compression: point x straight at its grandparent.
            self.parent[x as usize] = self.parent[p as usize];
            x = self.parent[x as usize];
        }
        EClassId(x)
    }

    #[allow(dead_code)] // wired for the future rewrite engine
    fn union(&mut self, a: EClassId, b: EClassId) -> EClassId {
        let ra = self.find(a).0;
        let rb = self.find(b).0;
        if ra == rb {
            return EClassId(ra);
        }
        let (small, large) =
            if self.rank[ra as usize] < self.rank[rb as usize] {
                (ra, rb)
            } else {
                (rb, ra)
            };
        self.parent[small as usize] = large;
        if self.rank[small as usize] == self.rank[large as usize] {
            self.rank[large as usize] += 1;
        }
        EClassId(large)
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Hash-consing a literal constant twice must yield the same id.
    #[test]
    fn test_egraph_hash_cons_simple_const() {
        let mut g = EGraph::new();
        let a = g.add(ENode::new(Op::Const(42), vec![])).unwrap();
        let b = g.add(ENode::new(Op::Const(42), vec![])).unwrap();
        assert_eq!(a, b);
        assert_eq!(g.len(), 1);
    }

    /// Two structurally identical `i32.add(1, 2)` trees must share one
    /// e-class and one root id.
    #[test]
    fn test_egraph_hash_cons_arith() {
        let mut g = EGraph::new();
        let c1_a = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let c2_a = g.add(ENode::new(Op::Const(2), vec![])).unwrap();
        let add_a = g.add(ENode::new(Op::I32Add, vec![c1_a, c2_a])).unwrap();

        let c1_b = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let c2_b = g.add(ENode::new(Op::Const(2), vec![])).unwrap();
        let add_b = g.add(ENode::new(Op::I32Add, vec![c1_b, c2_b])).unwrap();

        assert_eq!(c1_a, c1_b);
        assert_eq!(c2_a, c2_b);
        assert_eq!(add_a, add_b);
        // Exactly three distinct classes: Const(1), Const(2), Add.
        assert_eq!(g.len(), 3);
    }

    /// We deliberately do NOT canonicalize commutativity in the MVP.
    /// `add(1, 2)` and `add(2, 1)` are distinct e-classes; the rewrite
    /// engine that lands later will be responsible for unifying them.
    #[test]
    fn test_egraph_distinct_nodes_get_distinct_ids() {
        let mut g = EGraph::new();
        let c1 = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let c2 = g.add(ENode::new(Op::Const(2), vec![])).unwrap();
        let add_12 = g.add(ENode::new(Op::I32Add, vec![c1, c2])).unwrap();
        let add_21 = g.add(ENode::new(Op::I32Add, vec![c2, c1])).unwrap();
        assert_ne!(add_12, add_21);
        assert_eq!(g.len(), 4);
    }

    /// Build an `i32.add(local.get 0, i32.const 7)` tree, extract back
    /// to instructions, and confirm the byte stream round-trips through
    /// LOOM's wasm encoder (i.e., produces a valid function body).
    #[test]
    fn test_egraph_extract_round_trip() {
        use crate::encode::encode_wasm;
        use crate::{Function, FunctionSignature, Instruction, Module, ValueType};

        let mut g = EGraph::new();
        let lg0 = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let c7 = g.add(ENode::new(Op::Const(7), vec![])).unwrap();
        let add = g.add(ENode::new(Op::I32Add, vec![lg0, c7])).unwrap();

        let mut instrs = g.extract(add);
        // Expected post-order: local.get 0, i32.const 7, i32.add.
        assert_eq!(
            instrs,
            vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(7),
                Instruction::I32Add,
            ]
        );

        // Build a minimal module that returns the result so the encoder
        // sees a well-typed body. We need an End to terminate the body.
        instrs.push(Instruction::End);

        let module = Module {
            functions: vec![Function {
                name: Some("addk".to_string()),
                signature: FunctionSignature {
                    params: vec![ValueType::I32],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions: instrs,
            }],
            memories: vec![],
            tables: vec![],
            globals: vec![],
            types: vec![FunctionSignature {
                params: vec![ValueType::I32],
                results: vec![ValueType::I32],
            }],
            exports: vec![],
            imports: vec![],
            data_segments: vec![],
            element_section_bytes: None,
            start_function: None,
            custom_sections: vec![],
            type_section_bytes: None,
            global_section_bytes: None,
        };

        let bytes = encode_wasm(&module).expect("encode should succeed");
        // wasm header: 0x00 0x61 0x73 0x6d  + version 1.
        assert_eq!(&bytes[0..4], b"\0asm");
    }

    /// Direct attempt to fabricate a self-referential e-node must be
    /// rejected by [`EGraph::add`]. We synthesize the would-be class id
    /// (it's just `len()`) and feed it as a child.
    #[test]
    fn test_egraph_acyclic_refuses_cycle() {
        let mut g = EGraph::new();
        // The next class id that would be allocated is `EClassId(0)`.
        // Feed it as a child to a binary op and confirm rejection.
        let fake_self = EClassId(0);
        let other = EClassId(0); // doesn't matter; even one bad id rejects.
        let err = g
            .add(ENode::new(Op::I32Add, vec![fake_self, other]))
            .unwrap_err();
        match err {
            EGraphError::CycleRejected { new_id, bad_child } => {
                assert_eq!(new_id, EClassId(0));
                assert_eq!(bad_child, EClassId(0));
            }
            other => panic!("expected CycleRejected, got {:?}", other),
        }
        // And the graph must not have been mutated.
        assert_eq!(g.len(), 0);
    }

    /// `from_instruction` returns `None` for any op outside the MVP
    /// subset, so upstream callers can cleanly bail out to the existing
    /// optimizer paths.
    #[test]
    fn test_egraph_from_instruction_unknown_op_returns_none() {
        // I32DivS is deliberately not in the MVP subset.
        assert!(ENode::from_instruction(&Instruction::I32DivS, &[]).is_none());
        // Wrong arity should also fail even for a supported op.
        let bogus = ENode::from_instruction(&Instruction::I32Add, &[]);
        assert!(bogus.is_none());
    }

    /// Supported ops convert correctly when given the right children.
    #[test]
    fn test_egraph_from_instruction_supported() {
        let mut g = EGraph::new();
        let a = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let b = g.add(ENode::new(Op::Const(2), vec![])).unwrap();
        let node = ENode::from_instruction(&Instruction::I32Add, &[a, b])
            .expect("I32Add is supported");
        let id = g.add(node).unwrap();
        assert_eq!(g.extract(id).len(), 3); // c1, c2, add
    }
}

// ---------------------------------------------------------------------
// Follow-up work (v1.0.4+)
// ---------------------------------------------------------------------
//
// 1. **Rewrite engine.** Add a `union(a, b)` API plus a small driver
//    that walks the existing ISLE pattern set and merges semantically
//    equal classes. Each rewrite must come with its proof obligation
//    (Z3 or algebraic) per LOOM's correctness mandate.
//
// 2. **Wider op coverage.** Extend [`Op`] to cover i64 arithmetic,
//    comparisons, conversions, and memory ops as the rewrite engine
//    needs them. Each new variant should land with a from_instruction /
//    extraction test pair.
//
// 3. **Real cost model.** Replace the trivial post-order extractor with
//    a true cost-driven selector: per-op latency / size weights, dynamic
//    programming over classes, and tie-breaking that prefers ops the
//    backend can fuse.
//
// 4. **Integration with `canonicalize` / `simplify_with_env`.** Once the
//    rewrite engine lands, the existing pipeline should optionally feed
//    function bodies through the ægraph instead of the term rewriter
//    for the supported op subset, and round-trip back via `extract`.
//    Gate this behind a CLI flag until corpus measurements confirm wins.
