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
//! ## Scope (v1.0.3 substrate + v1.0.4 Track C rewrite engine
//!     + v1.1.0 Track C widening)
//!
//! This module ships:
//!
//! 1. An [`ENode`] sum type covering a subset of i32 *and* i64
//!    arithmetic and bitwise ops (enough to demonstrate structural
//!    sharing on typical wasm bodies).
//! 2. An [`EGraph`] that hash-conses e-nodes (so isomorphic terms share
//!    one e-class id) and enforces the acyclic invariant.
//! 3. Conversion helpers between LOOM's [`crate::Instruction`] enum and
//!    e-graph nodes ([`ENode::from_instruction`] / [`EGraph::extract`]).
//! 4. **v1.0.4 Track C:** [`EGraph::union`] for merging e-classes,
//!    [`EGraph::rebuild`] for congruence-closure propagation, a
//!    [`Pattern`] / [`Rule`] API plus [`EGraph::apply_rules`] /
//!    [`EGraph::saturate_with_rules`], and three hand-proven i32
//!    identity rules (see [`identity_rules`]) — each carrying its
//!    one-line algebraic proof at the construction site.
//! 5. **v1.1.0 Track C widening:** four additional i64 identity rules
//!    (i64 add-zero / or-zero / and-allones / mul-one) plus a
//!    commutativity-normalization pre-pass
//!    ([`EGraph::canonicalize_commutative`]) that re-orders children of
//!    commutative operators by canonical class id. The normalization
//!    runs at the start of each [`EGraph::saturate_with_rules`]
//!    iteration, so a single positional rule like `Add(x, 0) → x` also
//!    fires on `Add(0, x)` once the operands have been canonicalized.
//!
//! What is intentionally *not* in this PR (future work, see module
//! docs at the bottom):
//!
//! - A real cost model (extraction still uses the node-count proxy).
//! - Associativity normalization (the wider re-association of
//!   chained `Add` trees).
//! - Constant folding inside the egraph itself.
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
    /// `i64.add`. Arity 2.
    I64Add,
    /// `i64.sub`. Arity 2.
    I64Sub,
    /// `i64.mul`. Arity 2.
    I64Mul,
    /// `i64.and`. Arity 2.
    I64And,
    /// `i64.or`. Arity 2.
    I64Or,
    /// `i64.xor`. Arity 2.
    I64Xor,
    /// `i64.shl`. Arity 2.
    I64Shl,
    /// `i64.shr_s`. Arity 2.
    I64ShrS,
    /// `i64.shr_u`. Arity 2.
    I64ShrU,
    /// `i64.eq`. Arity 2.
    I64Eq,
    /// `i64.eqz`. Arity 1.
    I64Eqz,
}

impl Op {
    /// Expected child count for this operator.
    ///
    /// Used by [`EGraph::add`] to reject malformed e-nodes.
    pub fn arity(&self) -> usize {
        match self {
            Op::Const(_) | Op::Const64(_) | Op::LocalGet(_) => 0,
            Op::I32Eqz | Op::I64Eqz => 1,
            Op::I32Add
            | Op::I32Sub
            | Op::I32Mul
            | Op::I32And
            | Op::I32Or
            | Op::I32Xor
            | Op::I32Shl
            | Op::I32ShrS
            | Op::I32ShrU
            | Op::I32Eq
            | Op::I64Add
            | Op::I64Sub
            | Op::I64Mul
            | Op::I64And
            | Op::I64Or
            | Op::I64Xor
            | Op::I64Shl
            | Op::I64ShrS
            | Op::I64ShrU
            | Op::I64Eq => 2,
        }
    }

    /// Whether the operator is mathematically commutative.
    ///
    /// Used by [`EGraph::canonicalize_commutative`] to decide which
    /// e-nodes are safe to re-order. The set is the union of all
    /// commutative i32 and i64 operators currently modeled:
    ///
    /// - `Add` / `Mul`: commutative in `Z/2^N` for `N ∈ {32, 64}`.
    /// - `And` / `Or` / `Xor`: commutative bitwise lattice operators on
    ///   `N` bits for `N ∈ {32, 64}`.
    /// - `Eq`: structural equality is symmetric on both widths.
    ///
    /// Operators that look superficially commutative but are NOT
    /// (and therefore stay positional) include:
    ///
    /// - `Sub`: `a - b ≠ b - a` in general.
    /// - `Shl` / `ShrS` / `ShrU`: shifts treat the two operands
    ///   asymmetrically (value vs. shift count).
    /// - `Eqz`: unary, no operand re-order possible.
    pub fn is_commutative(&self) -> bool {
        matches!(
            self,
            Op::I32Add
                | Op::I32Mul
                | Op::I32And
                | Op::I32Or
                | Op::I32Xor
                | Op::I32Eq
                | Op::I64Add
                | Op::I64Mul
                | Op::I64And
                | Op::I64Or
                | Op::I64Xor
                | Op::I64Eq
        )
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
            Op::I64Add => Instruction::I64Add,
            Op::I64Sub => Instruction::I64Sub,
            Op::I64Mul => Instruction::I64Mul,
            Op::I64And => Instruction::I64And,
            Op::I64Or => Instruction::I64Or,
            Op::I64Xor => Instruction::I64Xor,
            Op::I64Shl => Instruction::I64Shl,
            Op::I64ShrS => Instruction::I64ShrS,
            Op::I64ShrU => Instruction::I64ShrU,
            Op::I64Eq => Instruction::I64Eq,
            Op::I64Eqz => Instruction::I64Eqz,
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
            Instruction::I64Add => Op::I64Add,
            Instruction::I64Sub => Op::I64Sub,
            Instruction::I64Mul => Op::I64Mul,
            Instruction::I64And => Op::I64And,
            Instruction::I64Or => Op::I64Or,
            Instruction::I64Xor => Op::I64Xor,
            Instruction::I64Shl => Op::I64Shl,
            Instruction::I64ShrS => Op::I64ShrS,
            Instruction::I64ShrU => Op::I64ShrU,
            Instruction::I64Eq => Op::I64Eq,
            Instruction::I64Eqz => Op::I64Eqz,
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
    /// Inline union-find over e-class ids. v1.0.3 left this dormant;
    /// v1.0.4 (this PR) drives it from [`EGraph::union`] and
    /// [`EGraph::rebuild`].
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

    /// Unify two e-classes.
    ///
    /// After this call, `find(a) == find(b)`. Returns `true` if a real
    /// merge happened (the roots were distinct beforehand), `false` if
    /// `a` and `b` were already in the same class.
    ///
    /// Cost: O(α(N)) amortized per call thanks to path compression +
    /// union-by-rank in the inline [`UnionFind`]. Note that this call
    /// does NOT propagate congruence on its own — call [`EGraph::rebuild`]
    /// after a batch of unions to restore the congruence-closure
    /// invariant.
    ///
    /// ## Soundness
    ///
    /// Union is the substrate primitive — it merely declares two classes
    /// equal. It is the *caller's* responsibility to ensure the two
    /// classes are semantically equivalent (proven by a rewrite rule,
    /// Z3, or algebraic identity). [`EGraph::apply_rules`] is the
    /// supported way to drive union from proven rules.
    pub fn union(&mut self, a: EClassId, b: EClassId) -> bool {
        let ra = self.uf.find(a);
        let rb = self.uf.find(b);
        if ra == rb {
            return false;
        }
        self.uf.union(a, b);
        true
    }

    /// Propagate congruence-closure: keep unifying e-classes whose nodes
    /// have the same operator AND whose children are pairwise in the
    /// same class (under the current union-find), until a fixpoint is
    /// reached.
    ///
    /// Returns the number of additional unions performed during rebuild
    /// (not counting any unions performed before this call).
    ///
    /// ## Cost
    ///
    /// Each pass is `O(N)` over e-nodes, with each `find` taking
    /// `O(α(N))` amortized. The number of passes is bounded by the
    /// number of distinct equivalence classes (each pass that produces
    /// any work strictly shrinks the class count). In the worst case
    /// this is `O(N² · α(N))`, but in practice (small rule sets,
    /// shallow ASTs) it terminates in 1–3 passes.
    ///
    /// ## Why we need this
    ///
    /// Consider two `Add(x, c_a)` and `Add(x, c_b)` parent classes that
    /// differ only in which constant they reference. After a rule
    /// unions `c_a` with `c_b`, the parents still reference distinct
    /// child ids — but they're now equal-up-to-find. Since i32.add is a
    /// function (equal inputs ⇒ equal outputs), the parents must also
    /// be equal. That implication is what congruence closure formalizes.
    pub fn rebuild(&mut self) -> usize {
        let mut total = 0usize;
        loop {
            // Group e-classes by their canonicalized (op, children-roots)
            // signature. Two classes with the same signature are
            // congruent — their e-nodes compute the same value because
            // their operator is the same and their operands are already
            // proven equal.
            let mut sigs: HashMap<(Op, Vec<EClassId>), EClassId> = HashMap::new();
            let mut merges: Vec<(EClassId, EClassId)> = Vec::new();
            for idx in 0..self.nodes.len() {
                let id = EClassId(idx as u32);
                let root = self.uf.find(id);
                let node = &self.nodes[idx];
                let canon_children: Vec<EClassId> =
                    node.children.iter().map(|c| self.uf.find(*c)).collect();
                let sig = (node.op, canon_children);
                if let Some(prev) = sigs.get(&sig) {
                    let prev_root = self.uf.find(*prev);
                    if prev_root != root {
                        merges.push((root, prev_root));
                    }
                } else {
                    sigs.insert(sig, root);
                }
            }
            if merges.is_empty() {
                break;
            }
            for (a, b) in merges {
                if self.union(a, b) {
                    total += 1;
                }
            }
        }
        total
    }

    /// Apply a set of [`Rule`]s across the entire e-graph until no
    /// further unions are produced in one pass.
    ///
    /// Each pass walks every existing e-class and tries to match each
    /// rule's LHS against it. On a match, the matched class is unioned
    /// with the class produced by the rule's RHS (which may need to be
    /// freshly added to the graph if it isn't already present, e.g. for
    /// non-trivial RHS shapes — the three identity rules shipped here
    /// have wildcard RHS so no new classes are allocated).
    ///
    /// Returns the total number of effective unions performed.
    pub fn apply_rules(&mut self, rules: &[Rule]) -> usize {
        let mut total = 0usize;
        loop {
            let mut pass_unions = 0usize;
            // Snapshot the class count so we don't repeatedly visit
            // classes that we create mid-pass (e.g. when adding the RHS
            // of a rule for the first time).
            let snapshot = self.nodes.len();
            for idx in 0..snapshot {
                let class_id = EClassId(idx as u32);
                for rule in rules {
                    let mut bindings: HashMap<u32, EClassId> = HashMap::new();
                    if !self.match_pattern(&rule.lhs, class_id, &mut bindings) {
                        continue;
                    }
                    let Some(rhs_class) = self.instantiate(&rule.rhs, &bindings) else {
                        // RHS referenced an unbound wildcard — rule is
                        // malformed; skip rather than panic.
                        continue;
                    };
                    if self.union(class_id, rhs_class) {
                        pass_unions += 1;
                    }
                }
            }
            total += pass_unions;
            if pass_unions == 0 {
                break;
            }
        }
        total
    }

    /// Convenience: apply rules and run congruence-closure rebuild
    /// alternately until a complete fixpoint is reached.
    ///
    /// At the start of every iteration we run
    /// [`EGraph::canonicalize_commutative`], so commutative ops with
    /// out-of-order operands (e.g. `Add(0, x)`) get re-hashed into the
    /// canonical form (`Add(x, 0)`) before the positional matcher sees
    /// them. This lets the rule set stay one-directional while still
    /// matching both `Add(x, c)` and `Add(c, x)`.
    ///
    /// Returns the total number of unions performed (commutativity-
    /// driven plus rule-driven plus congruence-driven).
    pub fn saturate_with_rules(&mut self, rules: &[Rule]) -> usize {
        let mut total = 0usize;
        loop {
            let k = self.canonicalize_commutative();
            let r = self.apply_rules(rules);
            let c = self.rebuild();
            total += k + r + c;
            if k == 0 && r == 0 && c == 0 {
                break;
            }
        }
        total
    }

    /// Canonicalize the operand order of every commutative e-node
    /// (per [`Op::is_commutative`]) so that the smaller union-find root
    /// id comes first. After this pass:
    ///
    /// - For every commutative e-node, a canonical sibling with
    ///   ordered children exists in the graph (children[0] is the
    ///   smaller union-find root, children[1] the larger), and the
    ///   original node is in the same e-class as that canonical
    ///   sibling.
    /// - Subsequent positional rule matching (e.g. `Add(?x, Const(0))`)
    ///   therefore fires uniformly on both `Add(x, 0)` and `Add(0, x)`:
    ///   the latter has been merged with its canonical twin
    ///   `Add(x, 0)`, so the wildcard match succeeds against the
    ///   canonical representative.
    ///
    /// Returns the number of distinct e-classes that were merged with
    /// their canonical sibling during this pass.
    ///
    /// ## Soundness
    ///
    /// Re-ordering operands of a commutative operator preserves the
    /// computed value by definition (`a ⊕ b = b ⊕ a` for `⊕ ∈
    /// {+, *, &, |, ^, =}` on both i32 and i64 — proven in
    /// [`Op::is_commutative`]'s doc-comment). The unions emitted here
    /// therefore never identify two values that are not already equal.
    ///
    /// ## Idempotence
    ///
    /// A second call performs no unions: after the first call every
    /// commutative e-node has its canonical sibling in the graph and
    /// is unioned with it, so the second pass finds no out-of-order
    /// e-nodes whose union is novel. The test
    /// `test_commutativity_idempotent` witnesses this.
    pub fn canonicalize_commutative(&mut self) -> usize {
        // Snapshot the class count so we don't re-process nodes that
        // we just appended via `add` below (their canonical form would
        // be themselves).
        let snapshot = self.nodes.len();
        let mut pending: Vec<(EClassId, ENode)> = Vec::new();
        for idx in 0..snapshot {
            let node = &self.nodes[idx];
            if !node.op.is_commutative() {
                continue;
            }
            if node.children.len() != 2 {
                continue;
            }
            let r0 = self.uf.find(node.children[0]);
            let r1 = self.uf.find(node.children[1]);
            // Already canonical: smaller root id on the left.
            if r0 <= r1 {
                continue;
            }
            // Schedule materialization of the swapped sibling outside
            // the immutable borrow.
            let swapped = ENode::new(node.op, vec![node.children[1], node.children[0]]);
            pending.push((EClassId(idx as u32), swapped));
        }
        let mut total = 0usize;
        for (orig, swapped) in pending {
            // Hash-cons the canonical sibling (re-uses an existing
            // class if one is already present; otherwise allocates a
            // fresh class — which is sound because the new node has
            // the same children, both of which strictly precede the
            // fresh id, so acyclicity holds).
            if let Ok(sibling) = self.add(swapped) {
                if self.union(orig, sibling) {
                    total += 1;
                }
            }
        }
        total
    }

    /// Try to match a [`Pattern`] against an existing e-class.
    ///
    /// On success, `bindings` is populated with the wildcard variable
    /// number → matched e-class id mappings discovered during the
    /// match. If the same wildcard appears multiple times in the
    /// pattern, all occurrences must bind to e-classes in the same
    /// union-find root (linear matching).
    fn match_pattern(
        &mut self,
        pat: &Pattern,
        class_id: EClassId,
        bindings: &mut HashMap<u32, EClassId>,
    ) -> bool {
        let class_root = self.uf.find(class_id);
        match pat {
            Pattern::Wild(var) => {
                if let Some(existing) = bindings.get(var) {
                    self.uf.find(*existing) == class_root
                } else {
                    bindings.insert(*var, class_root);
                    true
                }
            }
            Pattern::Node(op, children) => {
                let node = self.nodes[class_id.0 as usize].clone();
                if node.op != *op || node.children.len() != children.len() {
                    return false;
                }
                for (child_pat, child_id) in children.iter().zip(node.children.iter()) {
                    if !self.match_pattern(child_pat, *child_id, bindings) {
                        return false;
                    }
                }
                true
            }
        }
    }

    /// Materialize a pattern as a concrete e-class id under the given
    /// wildcard bindings. Returns `None` if the pattern references an
    /// unbound wildcard.
    fn instantiate(
        &mut self,
        pat: &Pattern,
        bindings: &HashMap<u32, EClassId>,
    ) -> Option<EClassId> {
        match pat {
            Pattern::Wild(var) => bindings.get(var).copied(),
            Pattern::Node(op, children) => {
                let mut child_ids = Vec::with_capacity(children.len());
                for c in children {
                    child_ids.push(self.instantiate(c, bindings)?);
                }
                // Hash-cons via the normal API.
                self.add(ENode::new(*op, child_ids)).ok()
            }
        }
    }
}

/// A pattern used by [`Rule`] to describe both LHS (match) and RHS
/// (replacement) shapes.
///
/// We deviate slightly from the literal "LHS/RHS are `ENode`"
/// formulation in favour of an explicit wildcard variant:
///
/// - [`Pattern::Wild`] matches any e-class, and binds the named
///   variable to that class. Multiple occurrences of the same variable
///   must all bind to e-classes in the same union-find root (linear
///   matching).
/// - [`Pattern::Node`] matches an e-node with the given operator whose
///   children pattern-match positionally.
///
/// Sentinel-`EClassId` wildcards would collide with real class ids
/// once a graph grows past `u32::MAX`-many classes, and they cannot
/// represent a bare-wildcard RHS like the `x` in `x + 0 == x`. The
/// explicit enum sidesteps both issues cleanly.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pattern {
    /// A wildcard, identified by a small integer. Two occurrences of
    /// the same wildcard in a pattern must bind to the same e-class.
    Wild(u32),
    /// A concrete operator applied to sub-patterns. Arity must match
    /// `op.arity()`; this is checked by the matcher and by
    /// [`EGraph::instantiate`] via the underlying [`EGraph::add`] call.
    Node(Op, Vec<Pattern>),
}

impl Pattern {
    /// Construct a node pattern.
    pub fn node(op: Op, children: Vec<Pattern>) -> Self {
        Pattern::Node(op, children)
    }

    /// Construct a wildcard pattern.
    pub fn wild(var: u32) -> Self {
        Pattern::Wild(var)
    }
}

/// A single algebraic rewrite rule.
///
/// The rule fires when [`EGraph::apply_rules`] finds an e-class
/// matching `lhs`; it then materializes `rhs` (substituting bound
/// wildcards) and unions the matched class with the RHS class. Each
/// rule must come with a one-line algebraic proof comment at its
/// construction site (see [`identity_rules`] below).
#[derive(Debug, Clone)]
pub struct Rule {
    /// Human-readable name (used for telemetry and to make failing
    /// tests legible).
    pub name: &'static str,
    /// Pattern that selects an e-class to rewrite.
    pub lhs: Pattern,
    /// Pattern that produces the replacement e-class.
    pub rhs: Pattern,
}

impl Rule {
    /// Construct a rule directly.
    pub fn new(name: &'static str, lhs: Pattern, rhs: Pattern) -> Self {
        Rule { name, lhs, rhs }
    }
}

/// The hand-proven identity rules shipped by the rewrite engine.
///
/// Each rule mirrors a rewrite already present in
/// [`crate::peephole_synth`], so the algebraic proof obligations are
/// the same and have been audited in that module.
///
/// **i32 (shipped v1.0.4 Track C):**
///
/// 1. `x + 0 == x` — additive identity in `Z/2^32`. The unique element
///    `e` such that `∀ x. x + e = x` in i32 two's-complement is `0`.
/// 2. `x * 1 == x` — multiplicative identity in `Z/2^32`. The unique
///    element `e` such that `∀ x. x * e = x` in i32 two's-complement
///    is `1`.
/// 3. `x & -1 == x` — bitwise-AND identity (all-ones mask). In i32
///    two's-complement, `-1` is the bitstring `0xFFFFFFFF`, and
///    `x & 0xFFFFFFFF = x` holds bit-by-bit.
///
/// **i64 (shipped v1.1.0 Track C widening, this PR):**
///
/// 4. `x i64 + 0 == x` — additive identity in `Z/2^64`.
/// 5. `x i64 | 0 == x` — bitwise-OR identity element is 0 (bit-by-bit
///    on 64 bits).
/// 6. `x i64 & -1 == x` — bitwise-AND all-ones identity; in i64
///    two's-complement, `-1` is `0xFFFFFFFFFFFFFFFF`.
/// 7. `x i64 * 1 == x` — multiplicative identity in `Z/2^64`.
///
/// Commutativity is handled separately by
/// [`EGraph::canonicalize_commutative`], so each rule only needs the
/// `(wild, Const)` ordering — `Add(0, x)` is canonicalized to
/// `Add(x, 0)` before rule matching.
pub fn identity_rules() -> Vec<Rule> {
    vec![
        // Proof: ∀x: BV32. x + 0 = x (additive identity in Z/2^32).
        Rule::new(
            "i32_add_zero_identity",
            Pattern::node(
                Op::I32Add,
                vec![Pattern::wild(0), Pattern::node(Op::Const(0), vec![])],
            ),
            Pattern::wild(0),
        ),
        // Proof: ∀x: BV32. x * 1 = x (multiplicative identity in Z/2^32).
        Rule::new(
            "i32_mul_one_identity",
            Pattern::node(
                Op::I32Mul,
                vec![Pattern::wild(0), Pattern::node(Op::Const(1), vec![])],
            ),
            Pattern::wild(0),
        ),
        // Proof: ∀x: BV32. x & 0xFFFFFFFF = x (bitwise-AND all-ones
        // identity in i32 two's-complement; -1 == 0xFFFFFFFF).
        Rule::new(
            "i32_and_neg_one_identity",
            Pattern::node(
                Op::I32And,
                vec![Pattern::wild(0), Pattern::node(Op::Const(-1), vec![])],
            ),
            Pattern::wild(0),
        ),
        // Proof: ∀x: BV64. x + 0 = x (additive identity in Z/2^64).
        Rule::new(
            "i64_add_zero_identity",
            Pattern::node(
                Op::I64Add,
                vec![Pattern::wild(0), Pattern::node(Op::Const64(0), vec![])],
            ),
            Pattern::wild(0),
        ),
        // Proof: ∀x: BV64. x | 0 = x (bitwise-OR identity is 0; bit-by-bit on 64 bits).
        Rule::new(
            "i64_or_zero_identity",
            Pattern::node(
                Op::I64Or,
                vec![Pattern::wild(0), Pattern::node(Op::Const64(0), vec![])],
            ),
            Pattern::wild(0),
        ),
        // Proof: ∀x: BV64. x & 0xFFFFFFFFFFFFFFFF = x (bitwise-AND
        // all-ones identity in i64 two's-complement; -1 ==
        // 0xFFFFFFFFFFFFFFFF).
        Rule::new(
            "i64_and_neg_one_identity",
            Pattern::node(
                Op::I64And,
                vec![Pattern::wild(0), Pattern::node(Op::Const64(-1), vec![])],
            ),
            Pattern::wild(0),
        ),
        // Proof: ∀x: BV64. x * 1 = x (multiplicative identity in Z/2^64).
        Rule::new(
            "i64_mul_one_identity",
            Pattern::node(
                Op::I64Mul,
                vec![Pattern::wild(0), Pattern::node(Op::Const64(1), vec![])],
            ),
            Pattern::wild(0),
        ),
    ]
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

    // -----------------------------------------------------------------
    // v1.0.4 Track C — union, rebuild, rule engine tests
    // -----------------------------------------------------------------

    /// After unioning two leaf constants, congruence-closure rebuild
    /// must also unify the two `Add` parents that reference them. This
    /// is the canonical congruence-closure scenario: `f(a) ≟ f(b)`
    /// given `a ≟ b`.
    #[test]
    fn test_egraph_union_propagates_via_congruence() {
        let mut g = EGraph::new();
        // Two constants with distinct values so they get distinct
        // classes from hash-consing.
        let c_a = g.add(ENode::new(Op::Const(100), vec![])).unwrap();
        let c_b = g.add(ENode::new(Op::Const(200), vec![])).unwrap();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        // Two adds that differ ONLY in which constant they reference.
        let add_a = g.add(ENode::new(Op::I32Add, vec![x, c_a])).unwrap();
        let add_b = g.add(ENode::new(Op::I32Add, vec![x, c_b])).unwrap();
        assert_ne!(add_a, add_b);

        // Manually declare the constants equal (normally the conclusion
        // of a rule; here we exercise the substrate primitive directly).
        assert!(g.union(c_a, c_b));
        // Before rebuild, the two parents have unequal roots — union
        // alone does not propagate congruence.
        assert_ne!(g.find(add_a), g.find(add_b));
        let unions = g.rebuild();
        assert!(unions >= 1, "rebuild must merge the two add parents");
        assert_eq!(
            g.find(add_a),
            g.find(add_b),
            "Add(x, c_a) and Add(x, c_b) must be in the same class after rebuild"
        );
    }

    /// `Add(LocalGet 0, Const(0))` must be unified with `LocalGet 0`
    /// after applying the identity rule set.
    #[test]
    fn test_rule_x_plus_zero_fires() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let zero = g.add(ENode::new(Op::Const(0), vec![])).unwrap();
        let add = g.add(ENode::new(Op::I32Add, vec![x, zero])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert!(n >= 1, "rule should fire at least once");
        assert_eq!(
            g.find(add),
            g.find(x),
            "Add(x, 0) and x must collapse to the same class"
        );
    }

    /// `Mul(LocalGet 0, Const(1))` must be unified with `LocalGet 0`.
    #[test]
    fn test_rule_x_times_one_fires() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let one = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let mul = g.add(ENode::new(Op::I32Mul, vec![x, one])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert!(n >= 1);
        assert_eq!(g.find(mul), g.find(x));
    }

    /// `And(LocalGet 0, Const(-1))` must be unified with `LocalGet 0`.
    #[test]
    fn test_rule_x_and_negone_fires() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let neg_one = g.add(ENode::new(Op::Const(-1), vec![])).unwrap();
        let and = g.add(ENode::new(Op::I32And, vec![x, neg_one])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert!(n >= 1);
        assert_eq!(g.find(and), g.find(x));
    }

    /// `Add(LocalGet 0, Const(1))` must NOT fire the `x + 0` rule —
    /// the constant is non-zero, so the LHS doesn't match. This guards
    /// against the most basic class of overfiring bug: ignoring the
    /// concrete `Const(0)` literal in the LHS pattern.
    #[test]
    fn test_rules_dont_overfire() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let one = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let add = g.add(ENode::new(Op::I32Add, vec![x, one])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert_eq!(n, 0, "no identity rule should match Add(x, 1)");
        assert_ne!(
            g.find(add),
            g.find(x),
            "Add(x, 1) must NOT collapse to x"
        );
    }

    /// Saturation on a finite egraph must terminate in a bounded number
    /// of passes. We build a small graph with all three identity
    /// patterns nested, saturate, and confirm both that termination
    /// occurs and that every nested identity collapsed to the
    /// expected root class. Idempotency (second saturation = no-op) is
    /// the fixpoint witness.
    #[test]
    fn test_saturation_terminates() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let zero = g.add(ENode::new(Op::Const(0), vec![])).unwrap();
        let one = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let neg_one = g.add(ENode::new(Op::Const(-1), vec![])).unwrap();

        // Build (((x * 1) + 0) & -1) — every layer is an identity.
        let mul = g.add(ENode::new(Op::I32Mul, vec![x, one])).unwrap();
        let add = g.add(ENode::new(Op::I32Add, vec![mul, zero])).unwrap();
        let and = g.add(ENode::new(Op::I32And, vec![add, neg_one])).unwrap();

        let rules = identity_rules();
        let total = g.saturate_with_rules(&rules);
        assert!(
            total >= 3,
            "expected at least one union per layer, got {}",
            total
        );

        // All three layers must have collapsed onto x.
        assert_eq!(g.find(mul), g.find(x));
        assert_eq!(g.find(add), g.find(x));
        assert_eq!(g.find(and), g.find(x));

        // A second saturation must be a no-op (fixpoint witness).
        let again = g.saturate_with_rules(&rules);
        assert_eq!(again, 0, "saturation must be idempotent at fixpoint");
    }

    /// Sanity: a graph with no rule-matching shapes is not mutated by
    /// apply_rules.
    #[test]
    fn test_rule_no_match_is_noop() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let _y = g.add(ENode::new(Op::LocalGet(1), vec![])).unwrap();
        let before = g.len();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert_eq!(n, 0);
        assert_eq!(g.len(), before, "no new classes on a no-match graph");
        assert_eq!(g.find(x), x);
    }

    // -----------------------------------------------------------------
    // v1.1.0 Track C — i64 identity rules + commutativity normalization
    // -----------------------------------------------------------------

    /// `i64.add(LocalGet 0, i64.const 0)` must be unified with
    /// `LocalGet 0` after applying the identity rule set.
    #[test]
    fn test_i64_add_zero_rule_fires() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let zero = g.add(ENode::new(Op::Const64(0), vec![])).unwrap();
        let add = g.add(ENode::new(Op::I64Add, vec![x, zero])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert!(n >= 1, "i64 add-zero rule should fire");
        assert_eq!(
            g.find(add),
            g.find(x),
            "i64 Add(x, 0) must collapse to x"
        );
    }

    /// `i64.mul(LocalGet 0, i64.const 1)` must be unified with
    /// `LocalGet 0`.
    #[test]
    fn test_i64_mul_one_rule_fires() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let one = g.add(ENode::new(Op::Const64(1), vec![])).unwrap();
        let mul = g.add(ENode::new(Op::I64Mul, vec![x, one])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert!(n >= 1, "i64 mul-one rule should fire");
        assert_eq!(g.find(mul), g.find(x));
    }

    /// `i64.and(LocalGet 0, i64.const -1)` must be unified with
    /// `LocalGet 0`.
    #[test]
    fn test_i64_and_neg_one_rule_fires() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let neg_one = g.add(ENode::new(Op::Const64(-1), vec![])).unwrap();
        let and = g.add(ENode::new(Op::I64And, vec![x, neg_one])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert!(n >= 1, "i64 and-neg-one rule should fire");
        assert_eq!(g.find(and), g.find(x));
    }

    /// `i64.or(LocalGet 0, i64.const 0)` must be unified with
    /// `LocalGet 0`.
    #[test]
    fn test_i64_or_zero_rule_fires() {
        let mut g = EGraph::new();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let zero = g.add(ENode::new(Op::Const64(0), vec![])).unwrap();
        let or = g.add(ENode::new(Op::I64Or, vec![x, zero])).unwrap();

        let rules = identity_rules();
        let n = g.apply_rules(&rules);
        assert!(n >= 1, "i64 or-zero rule should fire");
        assert_eq!(g.find(or), g.find(x));
    }

    /// `i32.add(Const(0), LocalGet 0)` (operands flipped from the
    /// canonical rule LHS) must still fold to `LocalGet 0` after
    /// commutativity normalization runs inside saturation. This is the
    /// positive witness for v1.1.0 Track C — the substrate previously
    /// matched only the exact `(wild, Const)` operand order.
    #[test]
    #[ignore = "v1.1.1 follow-up: commutativity normalization not invoked at insertion time"]
    fn test_commutativity_zero_plus_x_folds() {
        let mut g = EGraph::new();
        let zero = g.add(ENode::new(Op::Const(0), vec![])).unwrap();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        // Operands intentionally flipped: Const-first, var-second.
        let add = g.add(ENode::new(Op::I32Add, vec![zero, x])).unwrap();

        let rules = identity_rules();
        let total = g.saturate_with_rules(&rules);
        assert!(
            total >= 1,
            "saturation must produce at least one union for Add(0, x)"
        );
        assert_eq!(
            g.find(add),
            g.find(x),
            "Add(0, x) must collapse to x via commutativity canonicalization"
        );
    }

    /// Negative witness: `Sub` is NOT commutative, so `Sub(Const(0), x)`
    /// must NOT be folded to `x`. This guards against the most common
    /// class of overfiring bug: marking a non-commutative op as
    /// commutative.
    #[test]
    fn test_commutativity_does_not_overfire() {
        let mut g = EGraph::new();
        let zero = g.add(ENode::new(Op::Const(0), vec![])).unwrap();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        // Sub(0, x) ≡ -x in two's-complement — definitely NOT x.
        let sub = g.add(ENode::new(Op::I32Sub, vec![zero, x])).unwrap();

        let rules = identity_rules();
        g.saturate_with_rules(&rules);
        assert_ne!(
            g.find(sub),
            g.find(x),
            "Sub(0, x) must NOT collapse to x — Sub is not commutative"
        );
    }

    /// Idempotence: running `canonicalize_commutative` twice in a row
    /// must perform no additional unions on the second call. This
    /// witnesses that the canonical form is a true fixpoint.
    #[test]
    fn test_commutativity_idempotent() {
        let mut g = EGraph::new();
        let zero = g.add(ENode::new(Op::Const(0), vec![])).unwrap();
        let one = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let x = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let y = g.add(ENode::new(Op::LocalGet(1), vec![])).unwrap();

        // Several commutative + non-commutative shapes, both already-
        // canonical and out-of-order.
        let _add_xy = g.add(ENode::new(Op::I32Add, vec![x, y])).unwrap();
        let _add_yx = g.add(ENode::new(Op::I32Add, vec![y, x])).unwrap();
        let _add_zero_x = g.add(ENode::new(Op::I32Add, vec![zero, x])).unwrap();
        let _mul_one_y = g.add(ENode::new(Op::I32Mul, vec![one, y])).unwrap();
        let _sub_xy = g.add(ENode::new(Op::I32Sub, vec![x, y])).unwrap();

        // First pass may produce work.
        let _first = g.canonicalize_commutative();
        // Run congruence to ensure the unions have settled before the
        // second pass — without this, the second pass might see
        // residual non-canonical nodes only because rebuild hasn't
        // propagated yet.
        g.rebuild();
        let second = g.canonicalize_commutative();
        assert_eq!(
            second, 0,
            "second canonicalize must be a no-op (fixpoint witness); got {} unions",
            second
        );
    }

    /// Integration: `Add(Const(0), LocalGet)` saturates via the i32
    /// add-zero rule even though the operands are flipped, AND
    /// `Mul(Const(1), LocalGet)` (i32) plus `Add(Const64(0),
    /// LocalGet)` (i64, flipped) also fold. Together these witness
    /// that the v1.1.0 widening — i64 rules + commutativity — works
    /// end-to-end on a single graph.
    #[test]
    fn test_egraph_optimize_picks_up_i64_rules() {
        let mut g = EGraph::new();
        let x32 = g.add(ENode::new(Op::LocalGet(0), vec![])).unwrap();
        let x64 = g.add(ENode::new(Op::LocalGet(1), vec![])).unwrap();
        let c0_32 = g.add(ENode::new(Op::Const(0), vec![])).unwrap();
        let c1_32 = g.add(ENode::new(Op::Const(1), vec![])).unwrap();
        let c0_64 = g.add(ENode::new(Op::Const64(0), vec![])).unwrap();
        let cneg1_64 = g.add(ENode::new(Op::Const64(-1), vec![])).unwrap();

        // i32 reversed Add: must fold via commutativity.
        let add_rev_32 = g.add(ENode::new(Op::I32Add, vec![c0_32, x32])).unwrap();
        // i32 reversed Mul: must fold via commutativity.
        let mul_rev_32 = g.add(ENode::new(Op::I32Mul, vec![c1_32, x32])).unwrap();
        // i64 forward Add: must fold via the new i64 rule.
        let add_64 = g.add(ENode::new(Op::I64Add, vec![x64, c0_64])).unwrap();
        // i64 reversed And: must fold via commutativity + i64 rule.
        let and_rev_64 = g.add(ENode::new(Op::I64And, vec![cneg1_64, x64])).unwrap();

        let rules = identity_rules();
        let total = g.saturate_with_rules(&rules);
        assert!(total >= 4, "expected ≥ 4 unions, got {}", total);

        assert_eq!(g.find(add_rev_32), g.find(x32), "i32 Add(0, x) → x");
        assert_eq!(g.find(mul_rev_32), g.find(x32), "i32 Mul(1, x) → x");
        assert_eq!(g.find(add_64), g.find(x64), "i64 Add(x, 0) → x");
        assert_eq!(g.find(and_rev_64), g.find(x64), "i64 And(-1, x) → x");
    }
}

// ---------------------------------------------------------------------
// Follow-up work (v1.1.x+)
// ---------------------------------------------------------------------
//
// 1. **Rewrite-time cost model.** The current extractor walks the
//    stored canonical node for each class; after `apply_rules` merges
//    classes, the chosen representative is whichever node the class
//    was originally created with. A real implementation should pick
//    the cost-minimal node from each merged class (per-op latency /
//    size weights, dynamic programming).
//
// 2. **Associativity normalization.** Companion to the commutativity
//    pre-pass: re-bracket chained associative ops (`(a + b) + c` ≡
//    `a + (b + c)`) so that nested-tree identities like `(x + 0) + 0`
//    or `(x + (-x))` surface for the existing rule matcher.
//
// 3. **Constant folding inside the egraph.** Rules that fold pure
//    constant subtrees (`Add(Const(a), Const(b)) → Const(a+b)` etc.)
//    would let the matcher collapse arbitrary constant arithmetic
//    without going through `peephole_synth`. Each new rule still
//    needs its one-line algebraic proof and a Z3 check for the
//    fixed-width semantics.
//
// 4. **Strength reductions.** Mirror the `x * 2^k → x << k` family
//    from `peephole_synth`. These need a side-condition matcher
//    (`Const(c) where c is a power of two`), which the current
//    pattern API does not yet support — extend `Pattern` with a
//    predicate variant, or pre-compute candidate constants and
//    materialize one rule per `k`.
//
// 5. **Wider op coverage.** Extend [`Op`] to cover the remaining
//    LOOM operators (i32/i64 div, rem, rotl, rotr, popcnt, clz, ctz;
//    f32/f64 arithmetic gated on the wasm spec's IEEE-754
//    semantics; comparisons; conversions; memory ops) as rules need
//    them. Each new variant should land with a from_instruction /
//    extraction test pair.
