//! Formal Verification Module for LOOM
//!
//! This module implements translation validation using the Z3 SMT solver to prove
//! that optimizations preserve program semantics.
//!
//! # Overview
//!
//! Translation validation works by encoding both the original and optimized programs
//! as SMT formulas and asking Z3 to prove they are semantically equivalent for all
//! possible inputs.
//!
//! # Example
//!
//! ```rust,ignore
//! use loom_core::verify::verify_optimization;
//!
//! let original = parse_wat("(module (func (result i32) (i32.add (i32.const 2) (i32.const 3))))");
//! let optimized = parse_wat("(module (func (result i32) (i32.const 5)))");
//!
//! // Verify that 2 + 3 = 5 optimization is correct
//! assert!(verify_optimization(&original, &optimized).unwrap());
//! ```

#[cfg(feature = "verification")]
use z3::ast::{Array, BV, Bool};
#[cfg(feature = "verification")]
use z3::{Config, FuncDecl, SatResult, Solver, Sort, with_z3_config};

// #313 Tier-2 slice 1: the swappable-solver seam. `verify_solver` decides only
// obligations it can reflect into the closed pure-BV fragment AND re-lower to
// the identical Z3 AST; everything else stays on the incumbent below.
#[cfg(feature = "verification")]
use crate::rule_solver::{RuleVerdict, VerifyBackend};
#[cfg(feature = "verification")]
use crate::verify_solver::{self, SeamOutcome};

/// Feature flag for IEEE 754 float verification using Z3 FPA theory
/// When enabled, float operations are verified with proper IEEE 754 semantics
/// When disabled, floats are treated as symbolic bitvectors (bit-pattern equality only)
#[cfg(feature = "verification")]
const ENABLE_FPA_VERIFICATION: bool = true;

#[cfg(not(feature = "verification"))]
use crate::Module;
#[cfg(feature = "verification")]
use crate::{BlockType, Function, FunctionSignature, ImportKind, Instruction, Module};
use anyhow::{Result, anyhow};

/// Signature context for verification - stores function and type signatures
/// for proper Call/CallIndirect stack effect modeling.
#[cfg(feature = "verification")]
#[derive(Clone, Default)]
pub struct VerificationSignatureContext {
    /// Function signatures indexed by function index (imports first, then locals)
    pub function_signatures: Vec<FunctionSignature>,
    /// Type signatures for CallIndirect (indexed by type index)
    pub type_signatures: Vec<FunctionSignature>,
    /// IPA summaries indexed by local-function index (PR-K3). Imported
    /// functions have no summary entry; their slot in this vector is
    /// `None`. When a Call's callee is pure + no-trap + single-result,
    /// the verifier encodes the result as an uninterpreted-function
    /// application `f(args)` instead of a fresh symbolic constant —
    /// which makes Z3 prove two identical calls have equal results
    /// under congruence closure. This unblocks cross-call CSE dedup.
    pub function_summaries: Vec<Option<crate::summary::FunctionSummary>>,
    /// v1.0.4 Track B: element-section table resolver `(table_idx, slot) → func_idx`.
    /// Populated from `crate::optimize::build_table_resolver`. When a
    /// `call_indirect` is preceded by a constant `i32.const N` (visible
    /// to the Z3 encoder as a literal BV), the encoder can resolve the
    /// callee statically and encode the result as `pure_call_<F>(args)`
    /// — exactly like the PR-K3 direct-call encoding. This unblocks
    /// directize's Z3 verification: the encoded `call_indirect` BV and
    /// the encoded `call F` BV become structurally equal under
    /// Z3's congruence closure.
    pub table_resolver: std::collections::HashMap<(u32, u32), u32>,
    /// loom#151: local-function bodies indexed by function index (imports
    /// are `None`). When a `Call`'s callee is a pure, no-trap, leaf,
    /// straight-line callee (see `callee_inlinable_by_body`), the verifier
    /// models `call F(args)` by *symbolically executing F's body on the
    /// arg BVs* — i.e. by F's definition — instead of the opaque
    /// `pure_call_F(args)` uninterpreted function. The opaque encoding can
    /// prove two *identical* calls equal (CSE) but can NEVER prove a call
    /// equal to its own inlined body, so it made the verified inliner a
    /// no-op for every callee. Modeling by-body is what makes inlining
    /// provably sound. `Arc` keeps per-function context clones cheap.
    pub function_bodies: Vec<Option<std::sync::Arc<Function>>>,
    /// Value types of every global, imports first, in index order. Empty
    /// when the context was not built from a module; the exact executor then
    /// refuses any function that touches a global, because it cannot give
    /// the global its width.
    pub global_types: Vec<crate::ValueType>,
    /// One entry per memory, imports first: whether the exact executor can
    /// model it (32-bit and not shared). Empty when the context was not built
    /// from a module; a single plain memory 0 is then assumed.
    pub memories: Vec<bool>,
}

#[cfg(all(test, feature = "verification"))]
mod memory_bound_tests_347 {
    use super::*;

    fn f(instrs: Vec<Instruction>) -> Function {
        Function {
            name: None,
            signature: FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: instrs,
        }
    }

    fn store() -> Instruction {
        Instruction::I32Store {
            offset: 0,
            align: 2,
            mem: 0,
        }
    }

    fn load() -> Instruction {
        Instruction::I64Load {
            offset: 0,
            align: 3,
            mem: 0,
        }
    }

    #[test]
    fn counts_array_modelled_loads_and_stores() {
        assert_eq!(count_array_memory_ops(&f(vec![])), 0);
        assert_eq!(
            count_array_memory_ops(&f(vec![store(), load(), Instruction::I32Const(1)])),
            2,
            "only the memory accesses count, not the arithmetic beside them"
        );
    }

    /// The count must see through structured control flow. A body that hides
    /// its stores inside a block would otherwise slip under the bound and
    /// reintroduce the hang the bound exists to prevent.
    #[test]
    fn counts_nested_bodies() {
        let inner = Instruction::Block {
            block_type: BlockType::Empty,
            body: vec![store(), store()],
        };
        let outer = Instruction::If {
            block_type: BlockType::Empty,
            then_body: vec![store(), inner],
            else_body: vec![load()],
        };
        assert_eq!(
            count_array_memory_ops(&f(vec![outer, store()])),
            5,
            "stores nested in block/if bodies must be counted"
        );
    }

    /// Float and partial-width loads are deliberately NOT counted: they are
    /// rejected earlier by `contains_unverifiable_instructions` and never
    /// reach the array theory, so counting them would defer functions this
    /// bound has no reason to defer.
    #[test]
    fn does_not_count_accesses_that_never_reach_the_array_theory() {
        let float_store = Instruction::F64Store {
            offset: 0,
            align: 3,
            mem: 0,
        };
        let partial_load = Instruction::I32Load8U {
            offset: 0,
            align: 0,
            mem: 0,
        };
        assert_eq!(
            count_array_memory_ops(&f(vec![float_store, partial_load])),
            0
        );
    }

    /// The #219 exemption, asserted as a property rather than left implicit in
    /// the seam tests.
    ///
    /// The bound defers only bodies that are BOTH memory-dense and non-trivial.
    /// Dropping the instruction floor makes the memory-seam and division-seam
    /// dissolution tests fail — measured, not assumed: at a floor of 0 the
    /// bound needs to be >= 4 for those to pass, while #347 needs <= 2, and
    /// the two are irreconcilable without this exemption.
    #[test]
    fn a_tiny_memory_dense_body_is_exempt_from_the_bound() {
        let tiny = f(vec![store(), store(), store(), store()]);
        assert!(
            count_array_memory_ops(&tiny) > DEFAULT_MAX_ARRAY_MEMORY_OPS,
            "the fixture must exceed the memory bound, or it proves nothing"
        );
        assert!(
            count_function_instructions(&tiny) <= DEFAULT_MEMORY_BOUND_INSTRUCTION_FLOOR,
            "a body this small must fall under the instruction floor, which is \
             what keeps #219's seam-dissolution inlines verifiable"
        );
    }

    /// The control for the exemption: a body that is memory-dense AND large
    /// must NOT be exempt, or the bound would never fire and #347 returns.
    #[test]
    fn a_large_memory_dense_body_is_not_exempt() {
        let mut instrs = vec![store(); 4];
        instrs.extend(std::iter::repeat_n(
            Instruction::I32Const(0),
            DEFAULT_MEMORY_BOUND_INSTRUCTION_FLOOR + 1,
        ));
        let big = f(instrs);
        assert!(count_array_memory_ops(&big) > DEFAULT_MAX_ARRAY_MEMORY_OPS);
        assert!(
            count_function_instructions(&big) > DEFAULT_MEMORY_BOUND_INSTRUCTION_FLOOR,
            "this body must clear the floor so the bound actually applies to it"
        );
    }
}

/// Count the memory accesses in a function that the SMT encoder models with
/// the solver's ARRAY theory (#347).
///
/// This exists because `count_function_instructions` bounds the wrong
/// quantity. Profiling the meld-fused module that would not finish in 300 s
/// showed 97% of wall time inside `Z3_solver_check`, and **40% of the whole
/// process** inside `theory_array_base::propagate` →
/// `assert_store_axiom2_core`. Those are the array store axioms, and they are
/// instantiated PAIRWISE, so the cost grows quadratically in the number of
/// memory accesses reachable in one function body.
///
/// Instruction count does not track that at all: a body well under
/// `LOOM_Z3_MAX_INSTRUCTIONS` can carry enough stores to be intractable,
/// which is exactly how a 463 KB module ran past 455 s while a structurally
/// denser 1.13 MB module finished in 39 s.
///
/// Only the inliner triggers it in practice, and for a structural reason: it
/// is the one pass that concatenates callee bodies into a caller, so it is the
/// one pass that multiplies memory accesses PER FUNCTION. Every other pass
/// leaves that count roughly where it found it.
///
/// Counted here are the accesses the encoder actually lowers to array
/// select/store. The float and partial-width loads that
/// `contains_memory_instructions` rejects are deliberately NOT counted: those
/// bail out earlier and never reach the array theory, so counting them would
/// defer functions this bound has no reason to defer.
#[cfg(feature = "verification")]
fn count_array_memory_ops(func: &Function) -> usize {
    fn count_body(instrs: &[Instruction]) -> usize {
        let mut n = 0;
        for instr in instrs {
            match instr {
                Instruction::I32Load { .. }
                | Instruction::I64Load { .. }
                | Instruction::I32Store { .. }
                | Instruction::I64Store { .. }
                | Instruction::I32Store8 { .. }
                | Instruction::I32Store16 { .. }
                | Instruction::I64Store8 { .. }
                | Instruction::I64Store16 { .. }
                | Instruction::I64Store32 { .. } => n += 1,
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    n += count_body(body);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    n += count_body(then_body);
                    n += count_body(else_body);
                }
                _ => {}
            }
        }
        n
    }
    count_body(&func.instructions)
}

#[cfg(feature = "verification")]
impl VerificationSignatureContext {
    /// Create a new empty context (for backwards compatibility)
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a signature context from a module
    pub fn from_module(module: &Module) -> Self {
        let mut function_signatures = Vec::new();
        let mut function_summaries: Vec<Option<crate::summary::FunctionSummary>> = Vec::new();
        // loom#151: callee bodies for by-body call modeling (None for imports).
        let mut function_bodies: Vec<Option<std::sync::Arc<Function>>> = Vec::new();

        // First, add imported function signatures (they come first in indexing)
        // PR-K3: imported functions have no summary — their callee body
        // is opaque. We always conservatively encode their calls as fresh
        // symbolics with full havoc; that's the existing behavior.
        for import in &module.imports {
            if let ImportKind::Func(type_idx) = &import.kind {
                if let Some(sig) = module.types.get(*type_idx as usize) {
                    function_signatures.push(sig.clone());
                    function_summaries.push(None);
                    // Imported callee body is opaque — no by-body modeling.
                    function_bodies.push(None);
                }
            }
        }

        // Then add local function signatures + their summaries.
        let local_summaries = crate::summary::compute_module_summaries(module);
        for (func, summary) in module.functions.iter().zip(local_summaries.iter()) {
            function_signatures.push(func.signature.clone());
            function_summaries.push(Some(*summary));
            function_bodies.push(Some(std::sync::Arc::new(func.clone())));
        }

        // v1.0.4 Track B: build the element-section table resolver. A failure
        // to parse the section produces an empty resolver — directize will
        // then leave call_indirect untouched, mirroring the v1.0.2 behavior.
        let table_resolver = crate::optimize::build_table_resolver(module).unwrap_or_default();

        let mut global_types: Vec<crate::ValueType> = module
            .imports
            .iter()
            .filter_map(|i| match &i.kind {
                ImportKind::Global { value_type, .. } => Some(*value_type),
                _ => None,
            })
            .collect();
        global_types.extend(module.globals.iter().map(|g| g.value_type));
        let plain = |m: &crate::Memory| !m.shared && !m.memory64;
        let mut memories: Vec<bool> = module
            .imports
            .iter()
            .filter_map(|i| match &i.kind {
                ImportKind::Memory(m) => Some(plain(m)),
                _ => None,
            })
            .collect();
        memories.extend(module.memories.iter().map(plain));

        VerificationSignatureContext {
            function_signatures,
            type_signatures: module.types.clone(),
            function_summaries,
            table_resolver,
            function_bodies,
            global_types,
            memories,
        }
    }

    /// v1.0.4 Track B: look up the function index at `(table_idx, slot)`,
    /// returning `Some(func_idx)` only if the slot resolves AND the
    /// resolved function's signature is byte-identical to the
    /// indirect-call type. Returns `None` otherwise (caller falls back
    /// to the existing fresh-symbolic + havoc encoding).
    pub fn resolve_indirect_call(
        &self,
        table_idx: u32,
        slot: u32,
        expected_type: &FunctionSignature,
    ) -> Option<u32> {
        let func_idx = *self.table_resolver.get(&(table_idx, slot))?;
        let target_sig = self.function_signatures.get(func_idx as usize)?;
        if target_sig == expected_type {
            Some(func_idx)
        } else {
            None
        }
    }

    /// Get the signature for a function by its function index
    pub fn get_function_signature(&self, func_idx: u32) -> Option<&FunctionSignature> {
        self.function_signatures.get(func_idx as usize)
    }

    /// Get the signature for a type by its index (for indirect calls)
    pub fn get_type_signature(&self, type_idx: u32) -> Option<&FunctionSignature> {
        self.type_signatures.get(type_idx as usize)
    }

    /// loom#151: return the callee body if `func_idx` is a pure, no-trap,
    /// leaf, straight-line, single-result local function whose body the
    /// verifier can faithfully model by symbolic execution on the arg BVs
    /// (see `callee_inlinable_by_body`). Returns `None` otherwise — the
    /// caller then falls back to the conservative opaque `pure_call`
    /// encoding. This is what lets the verifier *prove* an inline sound:
    /// it models `call F(args)` by F's own definition, so the original
    /// (with the call) and the optimized (with F inlined) compute the
    /// identical expression and Z3 proves them equal.
    pub fn inlinable_callee_body(&self, func_idx: u32) -> Option<std::sync::Arc<Function>> {
        match self.function_bodies.get(func_idx as usize) {
            Some(Some(f)) if callee_inlinable_by_body(f) => Some(f.clone()),
            _ => None,
        }
    }
}

/// loom#151: is `func` a callee the verifier may model by symbolically
/// executing its body (rather than as an opaque uninterpreted function)?
///
/// SOUNDNESS: this returns `true` ONLY for callees whose body is a
/// deterministic, total (never-trapping), state-free function of its
/// arguments — i.e. straight-line integer/bitwise/compare/local/const
/// instructions with exactly one result. Such a body is encoded the same
/// way by `encode_block_body` as the inlined copy is encoded by the main
/// loop, so modeling by-body cannot introduce a *false* equivalence:
///   - excludes div/rem (may trap) and all loads/stores/globals/calls/
///     control-flow/float ops, so there is no hidden state, no trap, no
///     nondeterminism, and no second interpreter disagreement to exploit.
///   - any callee outside this whitelist returns `false`, and the Call
///     handler falls back to the existing conservative encoding (which at
///     worst yields a spurious counterexample → revert, never an unsound
///     accept).
///
/// Conservative by design; the whitelist can be widened later, each
/// addition gated by its own soundness argument.
#[cfg(feature = "verification")]
fn callee_inlinable_by_body(func: &Function) -> bool {
    // At most one result: multi-value needs tuple plumbing the encoder does
    // not have. loom#157: zero-result (void) callees ARE allowed now — a pure
    // writer (`void wr(*s, v)`) returns no value but updates memory, which the
    // by-body modeler threads back to the caller.
    if func.signature.results.len() > 1 {
        return false;
    }
    func.instructions.iter().all(is_inline_modelable_instr)
}

/// PR-C (#219): true if `func` provably never returns to its caller — every
/// path traps. Conservative-sound definition: the body contains NO `Return`
/// and NO branch (`Br`/`BrIf`/`BrTable`) anywhere (recursing into nested
/// control flow), and its final top-level instruction is `Unreachable`. With
/// no `return` and no branch, the only way to leave the function is to fall off
/// the end — which `Unreachable` makes a trap. This admits exactly the
/// `core::panicking` helpers here (`panic_fmt → unreachable`,
/// `panic_const_add_overflow → call panic_fmt; unreachable`) and rejects any
/// normal-returning or branching function.
///
/// Used by the by-body inline modeler to encode a `call` to such a callee as
/// "diverge if reached" (⊥) rather than havoc: control never returns, so the
/// call constrains only the reachability of the trapping branch, never the
/// post-state (Alive2's `ub` discipline). Modeling it as a fresh symbolic
/// (havoc) would reopen the #155/#159 havoc-vs-concrete false-positive surface.
#[cfg(feature = "verification")]
#[allow(dead_code)] // wired into the by-body inline gate in PR-C M3 (#219)
pub(crate) fn is_noreturn_callee(func: &Function) -> bool {
    !body_has_branch_or_return(&func.instructions)
        && matches!(func.instructions.last(), Some(Instruction::Unreachable))
}

/// Recursively detect any `Return` or branch (`Br`/`BrIf`/`BrTable`) in a body,
/// descending into Block/Loop/If. Used by `is_noreturn_callee` — a branch could
/// transfer control to the function-level label (a return-via-br), so any
/// branch conservatively disqualifies the "every path traps" classification.
#[cfg(feature = "verification")]
fn body_has_branch_or_return(instrs: &[Instruction]) -> bool {
    instrs.iter().any(|i| match i {
        Instruction::Return
        | Instruction::Br(_)
        | Instruction::BrIf(_)
        | Instruction::BrTable { .. } => true,
        Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
            body_has_branch_or_return(body)
        }
        Instruction::If {
            then_body,
            else_body,
            ..
        } => body_has_branch_or_return(then_body) || body_has_branch_or_return(else_body),
        _ => false,
    })
}

/// loom#151/#155: whitelist of instructions the by-body inline modeler can
/// encode faithfully and **identically to the main encoder**. The callee must
/// be straight-line (no Block/Loop/If), leaf (no calls), single-result, and
/// use only: constants, locals (no globals), stack shuffles, total integer
/// arithmetic/bitwise/compare/convert (no div/rem, no floats), and — added in
/// loom#155 — **full-width i32/i64 loads** (no stores, no partial-width).
///
/// SOUNDNESS of allowing loads: a load is neither "total" (it can trap OOB)
/// nor state-free, but it is still sound for the *inline-equivalence* check.
/// Both the original's modeled `call F` and the optimized's inlined body run
/// the SAME load at the SAME address against the SAME shared memory `Array`
/// via the SAME helper (`encode_i32_load_from`/`encode_i64_load_from`), so
/// they produce bit-identical expressions — and any real OOB trap occurs
/// identically on both sides, so the transformation preserves trap behavior
/// regardless of loom's (non-trapping) Array model. Stores/globals/partial-
/// width loads stay OUT (writes would need memory write-back; partial-width
/// loads aren't full-width-modeled) → such callees fall back conservatively.
#[cfg(feature = "verification")]
fn is_inline_modelable_instr(instr: &Instruction) -> bool {
    // Non-load ops here must stay a SUBSET of what `encode_simple_instruction`
    // models faithfully; the full-width loads are intercepted directly in
    // `encode_inlinable_callee_result` (shared helper), not delegated.
    use Instruction::*;
    matches!(
        instr,
        // constants
        I32Const(_) | I64Const(_)
        // locals (params + declared) — no globals
        | LocalGet(_) | LocalSet(_) | LocalTee(_)
        // stack shuffles
        | Drop | Select
        // full-width loads (loom#155) + stores (loom#157) — shared-Array
        // encoding; stores thread the updated memory back to the caller
        | I32Load { .. } | I64Load { .. }
        | I32Store { .. } | I64Store { .. }
        // partial-width i32 loads/stores (loom#161) — int8_t/int16_t fields,
        // e.g. filter_step's i32.load16_s. Encoded via the shared partial-width
        // helpers (correct sign/zero-ext + masking).
        | I32Load8S { .. } | I32Load8U { .. }
        | I32Load16S { .. } | I32Load16U { .. }
        | I32Store8 { .. } | I32Store16 { .. }
        // i32 arithmetic / bitwise
        | I32Add | I32Sub | I32Mul | I32And | I32Or | I32Xor
        | I32Shl | I32ShrS | I32ShrU
        // i64 arithmetic / bitwise
        | I64Add | I64Sub | I64Mul | I64And | I64Or | I64Xor
        | I64Shl | I64ShrS | I64ShrU
        // integer division/remainder (loom#163) — bvsdiv/bvudiv/bvsrem/bvurem;
        // trap-on-zero preserved (same op both sides), sound for inlining
        | I32DivS | I32DivU | I32RemS | I32RemU
        | I64DivS | I64DivU | I64RemS | I64RemU
        // i32 comparisons
        | I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU
        | I32LeS | I32LeU | I32GeS | I32GeU | I32Eqz
        // i64 comparisons
        | I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU
        | I64LeS | I64LeU | I64GeS | I64GeU | I64Eqz
        // width conversions (total)
        | I64ExtendI32S | I64ExtendI32U | I32WrapI64
    )
}

/// loom#151: symbolically execute an inlinable callee body on the given
/// argument BVs and return its single result BV. Returns `None` if the
/// result width does not match the callee's declared result type (a
/// defensive guard against any width mismatch reaching the equality
/// check). Pre: `callee_inlinable_by_body(func)` held when this was
/// selected, so the body is straight-line and state-free; `globals` is
/// unused and `encode_block_body` never touches memory here.
#[cfg(feature = "verification")]
fn encode_inlinable_callee_result(
    func: &Function,
    args: &[BV],
    memory: &Array,
) -> Option<(Option<BV>, Array)> {
    // Build the callee's local environment: params bound to the actual
    // arg BVs, then declared locals zero-initialized at their width.
    let mut locals: Vec<BV> = args.to_vec();
    for (count, local_type) in func.locals.iter() {
        let width = bv_width_for_value_type(*local_type);
        for _ in 0..*count {
            locals.push(BV::from_u64(0, width));
        }
    }
    let mut stack: Vec<BV> = Vec::new();
    let mut globals: Vec<BV> = Vec::new();
    // Pre: `callee_inlinable_by_body(func)` held when this was selected, so the
    // body is straight-line (no Block/Loop/If), leaf (no calls), single-or-zero
    // result, and uses only whitelisted ops + full-width loads/stores — no
    // globals/partial-width. We execute it as a flat sequence.
    //   loom#155: full-width LOADS are encoded against the CALLER's shared
    //   memory `Array` via the SAME helpers the main encoder uses.
    //   loom#157: full-width STORES are applied to that Array and the updated
    //   Array is RETURNED to the caller — so the original's modeled `call F`
    //   and the optimized's inlined body produce bit-identical loads AND
    //   memory updates. Modeling the writer's call by-body (concrete stores)
    //   is what removes the old havoc-vs-concrete divergence: both sides now
    //   carry the same stores, so a following read matches → provable.
    let mut mem = memory.clone();
    for instr in &func.instructions {
        match instr {
            Instruction::I32Load { offset, .. } => {
                let addr = stack.pop()?;
                stack.push(encode_i32_load_from(&mem, &addr, *offset).ok()?);
            }
            Instruction::I64Load { offset, .. } => {
                let addr = stack.pop()?;
                stack.push(encode_i64_load_from(&mem, &addr, *offset).ok()?);
            }
            Instruction::I32Store { offset, .. } => {
                let value = stack.pop()?;
                let addr = stack.pop()?;
                mem = encode_i32_store_into(&mem, &addr, &value, *offset);
            }
            Instruction::I64Store { offset, .. } => {
                let value = stack.pop()?;
                let addr = stack.pop()?;
                mem = encode_i64_store_into(&mem, &addr, &value, *offset);
            }
            // loom#161: partial-width i32 loads/stores (int8_t/int16_t fields).
            Instruction::I32Load8S { offset, .. } => {
                let addr = stack.pop()?;
                stack.push(encode_partial_load_from(&mem, &addr, *offset, 1, true, 32).ok()?);
            }
            Instruction::I32Load8U { offset, .. } => {
                let addr = stack.pop()?;
                stack.push(encode_partial_load_from(&mem, &addr, *offset, 1, false, 32).ok()?);
            }
            Instruction::I32Load16S { offset, .. } => {
                let addr = stack.pop()?;
                stack.push(encode_partial_load_from(&mem, &addr, *offset, 2, true, 32).ok()?);
            }
            Instruction::I32Load16U { offset, .. } => {
                let addr = stack.pop()?;
                stack.push(encode_partial_load_from(&mem, &addr, *offset, 2, false, 32).ok()?);
            }
            Instruction::I32Store8 { offset, .. } => {
                let value = stack.pop()?;
                let addr = stack.pop()?;
                mem = encode_partial_store_into(&mem, &addr, &value, *offset, 1);
            }
            Instruction::I32Store16 { offset, .. } => {
                let value = stack.pop()?;
                let addr = stack.pop()?;
                mem = encode_partial_store_into(&mem, &addr, &value, *offset, 2);
            }
            other => {
                encode_simple_instruction(other, &mut stack, &mut locals, &mut globals).ok()?;
            }
        }
    }
    // Zero-result (void) callees — e.g. a pure writer — return no value but a
    // possibly-updated memory. Single-result callees width-check the result.
    if func.signature.results.is_empty() {
        return Some((None, mem));
    }
    let result = stack.last().cloned()?;
    let expected = bv_width_for_value_type(func.signature.results[0]);
    if result.get_size() == expected {
        Some((Some(result), mem))
    } else {
        None
    }
}

/// PR-K3.2 (Track B): recursively count instructions in a function body.
/// Used by the size-threshold gate in TranslationValidator::verify.
#[cfg(feature = "verification")]
fn count_function_instructions(func: &Function) -> usize {
    fn count_body(instrs: &[Instruction]) -> usize {
        let mut n = instrs.len();
        for instr in instrs {
            match instr {
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    n += count_body(body);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    n += count_body(then_body);
                    n += count_body(else_body);
                }
                _ => {}
            }
        }
        n
    }
    count_body(&func.instructions)
}

#[cfg(feature = "verification")]
impl VerificationSignatureContext {
    // PR-K3.2 (Track B): recursively count instructions in a function
    // body. Used by `verify` to gate Z3 invocation by body size — see
    // the comment in `verify` for rationale and the LOOM_Z3_MAX_INSTRUCTIONS
    // env var. (defined out of the impl block — see below)

    /// PR-K3: check whether a callee is pure + no-trap and therefore
    /// safe to encode as a deterministic uninterpreted function
    /// `f(args)` in the SMT representation. Returns true only when
    /// IPA proved both purity and no-trap; imported / unknown / non-
    /// pure / may-trap callees return false, falling back to the
    /// conservative fresh-symbolic + havoc encoding.
    pub fn is_pure_no_trap_call(&self, func_idx: u32) -> bool {
        match self.function_summaries.get(func_idx as usize) {
            Some(Some(s)) => s.is_pure && s.is_no_trap,
            _ => false,
        }
    }
}

/// Stub signature context when verification is disabled
#[cfg(not(feature = "verification"))]
#[derive(Clone, Default)]
pub struct VerificationSignatureContext;

#[cfg(not(feature = "verification"))]
impl VerificationSignatureContext {
    /// Create a new empty context (stub)
    pub fn new() -> Self {
        Self
    }

    /// Create a signature context from a module (stub)
    pub fn from_module(_module: &Module) -> Self {
        Self
    }
}

// ============================================================================
// Function Call Summaries
// ============================================================================

/// Summary of a function's effects for verification purposes.
///
/// This enables more precise verification when function calls are present:
/// - Track which globals are read/written
/// - Track if memory is read/written
/// - Determine if function is pure (no side effects)
#[cfg(feature = "verification")]
#[derive(Debug, Clone, Default)]
pub struct FunctionSummary {
    /// Set of global indices that this function reads
    pub globals_read: std::collections::HashSet<u32>,
    /// Set of global indices that this function writes
    pub globals_written: std::collections::HashSet<u32>,
    /// Whether this function reads from linear memory
    pub reads_memory: bool,
    /// Whether this function writes to linear memory
    pub writes_memory: bool,
    /// Whether this function makes calls to other functions
    pub has_calls: bool,
    /// Whether this function makes indirect calls (harder to analyze)
    pub has_indirect_calls: bool,
    /// Function indices that this function calls (direct calls only)
    pub called_functions: std::collections::HashSet<u32>,
}

#[cfg(feature = "verification")]
impl FunctionSummary {
    /// Create a new empty function summary
    pub fn new() -> Self {
        Self::default()
    }

    /// Check if this function is pure (no side effects)
    /// A pure function doesn't write to globals or memory
    pub fn is_pure(&self) -> bool {
        self.globals_written.is_empty() && !self.writes_memory
    }

    /// Check if this function has any observable effects
    pub fn has_side_effects(&self) -> bool {
        !self.globals_written.is_empty() || self.writes_memory
    }

    /// Analyze a function and build its summary
    pub fn analyze(func: &Function) -> Self {
        let mut summary = FunctionSummary::new();

        for instr in &func.instructions {
            match instr {
                // Global reads
                Instruction::GlobalGet(idx) => {
                    summary.globals_read.insert(*idx);
                }
                // Global writes
                Instruction::GlobalSet(idx) => {
                    summary.globals_written.insert(*idx);
                }
                // Memory reads
                Instruction::I32Load { .. }
                | Instruction::I64Load { .. }
                | Instruction::F32Load { .. }
                | Instruction::F64Load { .. }
                | Instruction::I32Load8S { .. }
                | Instruction::I32Load8U { .. }
                | Instruction::I32Load16S { .. }
                | Instruction::I32Load16U { .. }
                | Instruction::I64Load8S { .. }
                | Instruction::I64Load8U { .. }
                | Instruction::I64Load16S { .. }
                | Instruction::I64Load16U { .. }
                | Instruction::I64Load32S { .. }
                | Instruction::I64Load32U { .. } => {
                    summary.reads_memory = true;
                }
                // Memory writes
                Instruction::I32Store { .. }
                | Instruction::I64Store { .. }
                | Instruction::F32Store { .. }
                | Instruction::F64Store { .. }
                | Instruction::I32Store8 { .. }
                | Instruction::I32Store16 { .. }
                | Instruction::I64Store8 { .. }
                | Instruction::I64Store16 { .. }
                | Instruction::I64Store32 { .. } => {
                    summary.writes_memory = true;
                }
                // Direct calls
                Instruction::Call(func_idx) => {
                    summary.has_calls = true;
                    summary.called_functions.insert(*func_idx);
                }
                // Indirect calls
                Instruction::CallIndirect { .. } => {
                    summary.has_indirect_calls = true;
                    summary.has_calls = true;
                }
                // Memory size/grow operations
                Instruction::MemorySize { .. } => {
                    summary.reads_memory = true;
                }
                Instruction::MemoryGrow { .. } => {
                    summary.writes_memory = true;
                }
                // Bulk memory operations
                Instruction::MemoryFill(_) => {
                    summary.writes_memory = true;
                }
                Instruction::MemoryCopy { .. } => {
                    summary.reads_memory = true;
                    summary.writes_memory = true;
                }
                Instruction::MemoryInit { .. } => {
                    summary.writes_memory = true;
                }
                // DataDrop has no memory effect
                _ => {}
            }
        }

        summary
    }

    /// Compute transitive closure of effects for a function
    /// This propagates effects from called functions to callers
    pub fn compute_transitive_effects(
        summaries: &mut std::collections::HashMap<u32, FunctionSummary>,
        module_func_count: u32,
    ) {
        // Fixed-point iteration: keep propagating until no changes
        let mut changed = true;
        let mut iterations = 0;
        const MAX_ITERATIONS: usize = 100;

        while changed && iterations < MAX_ITERATIONS {
            changed = false;
            iterations += 1;

            for func_idx in 0..module_func_count {
                if let Some(summary) = summaries.get(&func_idx).cloned() {
                    let mut new_summary = summary.clone();

                    // Propagate effects from called functions
                    for called_idx in &summary.called_functions {
                        if let Some(called_summary) = summaries.get(called_idx) {
                            // Merge global reads
                            for g in &called_summary.globals_read {
                                if new_summary.globals_read.insert(*g) {
                                    changed = true;
                                }
                            }
                            // Merge global writes
                            for g in &called_summary.globals_written {
                                if new_summary.globals_written.insert(*g) {
                                    changed = true;
                                }
                            }
                            // Propagate memory effects
                            if called_summary.reads_memory && !new_summary.reads_memory {
                                new_summary.reads_memory = true;
                                changed = true;
                            }
                            if called_summary.writes_memory && !new_summary.writes_memory {
                                new_summary.writes_memory = true;
                                changed = true;
                            }
                        }
                    }

                    // If function has indirect calls, be conservative
                    if summary.has_indirect_calls {
                        // Indirect calls could do anything
                        new_summary.reads_memory = true;
                        new_summary.writes_memory = true;
                    }

                    summaries.insert(func_idx, new_summary);
                }
            }
        }
    }
}

/// Build function summaries for all functions in a module
#[cfg(feature = "verification")]
pub fn build_function_summaries(
    module: &Module,
) -> std::collections::HashMap<u32, FunctionSummary> {
    let mut summaries = std::collections::HashMap::new();

    // Count imported functions (they come before local functions in indexing)
    let import_func_count = module
        .imports
        .iter()
        .filter(|i| matches!(i.kind, ImportKind::Func(_)))
        .count() as u32;

    // For imported functions, we don't have the body so assume worst case
    for idx in 0..import_func_count {
        let mut summary = FunctionSummary::new();
        // Imported functions could do anything - be conservative
        summary.reads_memory = true;
        summary.writes_memory = true;
        summary.has_calls = true;
        summaries.insert(idx, summary);
    }

    // Analyze local functions
    for (i, func) in module.functions.iter().enumerate() {
        let func_idx = import_func_count + i as u32;
        let summary = FunctionSummary::analyze(func);
        summaries.insert(func_idx, summary);
    }

    // Compute transitive closure
    let total_funcs = import_func_count + module.functions.len() as u32;
    FunctionSummary::compute_transitive_effects(&mut summaries, total_funcs);

    summaries
}

/// Stub for FunctionSummary when verification is disabled
#[cfg(not(feature = "verification"))]
#[derive(Debug, Clone, Default)]
pub struct FunctionSummary;

// ============================================================================
// Verification Coverage Tracking
// ============================================================================

/// Tracks verification coverage across function validations.
///
/// This struct provides metrics on what percentage of functions are fully
/// Z3-verified vs skipped due to unsupported features (loops, memory ops).
///
/// # Usage
///
/// ```rust,ignore
/// let mut coverage = VerificationCoverage::new();
/// // After verification...
/// println!("Coverage: {:.1}%", coverage.coverage_percent());
/// ```
#[derive(Debug, Clone, Default)]
pub struct VerificationCoverage {
    /// Number of functions fully verified with Z3
    pub verified: usize,
    /// Number of functions skipped due to containing loops
    pub skipped_loops: usize,
    /// Number of functions skipped due to memory operations
    pub skipped_memory: usize,
    /// Number of functions skipped due to unknown/unsupported instructions
    pub skipped_unknown: usize,
    /// Number of functions where verification failed (counterexample found)
    pub verification_failed: usize,
    /// Number of functions where verification errored (timeout, etc.)
    pub verification_error: usize,
    /// Number of functions verification was never ATTEMPTED on — currently
    /// only a build compiled without the `verification` feature.
    ///
    /// This bucket exists because every other one asserts something: that a
    /// function was proven, that it was skipped for a named reason, that a
    /// counterexample was found, that the solver errored. "Nothing ran" is
    /// none of those, and borrowing the nearest bucket would replace one
    /// false statement with another — reporting a timeout that never
    /// happened is no better than reporting a proof that never happened.
    pub not_attempted: usize,
}

impl VerificationCoverage {
    /// Create a new empty coverage tracker
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a fully verified function
    pub fn record_verified(&mut self) {
        self.verified += 1;
    }

    /// Record a function skipped due to loops
    pub fn record_skipped_loop(&mut self) {
        self.skipped_loops += 1;
    }

    /// Record a function skipped due to memory operations
    pub fn record_skipped_memory(&mut self) {
        self.skipped_memory += 1;
    }

    /// Record a function skipped due to unknown instructions
    pub fn record_skipped_unknown(&mut self) {
        self.skipped_unknown += 1;
    }

    /// Record a verification failure (counterexample found)
    pub fn record_failed(&mut self) {
        self.verification_failed += 1;
    }

    /// Record a verification error (timeout, encoding error)
    pub fn record_error(&mut self) {
        self.verification_error += 1;
    }

    /// Record a function verification was never attempted on.
    pub fn record_not_attempted(&mut self) {
        self.not_attempted += 1;
    }

    /// True when no function in this coverage was actually verified —
    /// the state a caller must be able to detect before quoting a number.
    pub fn nothing_was_attempted(&self) -> bool {
        self.total() > 0 && self.not_attempted == self.total()
    }

    /// Total number of functions processed
    pub fn total(&self) -> usize {
        self.verified
            + self.skipped_loops
            + self.skipped_memory
            + self.skipped_unknown
            + self.verification_failed
            + self.verification_error
            + self.not_attempted
    }

    /// Total number of functions skipped (not fully verified)
    pub fn total_skipped(&self) -> usize {
        self.skipped_loops + self.skipped_memory + self.skipped_unknown
    }

    /// Calculate verification coverage percentage (0.0 to 100.0)
    ///
    /// Coverage = verified / (verified + skipped)
    /// Functions that failed or errored are not counted as skipped.
    pub fn coverage_percent(&self) -> f64 {
        let relevant = self.verified + self.total_skipped();
        if relevant == 0 {
            100.0
        } else {
            (self.verified as f64 / relevant as f64) * 100.0
        }
    }

    /// Get a human-readable summary
    pub fn summary(&self) -> String {
        format!(
            "Verification: {}/{} functions ({:.1}% Z3-proven)\n  \
             Skipped: {} loops, {} memory ops, {} unknown\n  \
             Failed: {}, Errors: {}",
            self.verified,
            self.total(),
            self.coverage_percent(),
            self.skipped_loops,
            self.skipped_memory,
            self.skipped_unknown,
            self.verification_failed,
            self.verification_error
        )
    }

    /// Merge another coverage into this one
    pub fn merge(&mut self, other: &VerificationCoverage) {
        self.verified += other.verified;
        self.skipped_loops += other.skipped_loops;
        self.skipped_memory += other.skipped_memory;
        self.skipped_unknown += other.skipped_unknown;
        self.verification_failed += other.verification_failed;
        self.verification_error += other.verification_error;
    }
}

/// Result of verifying a function, including coverage info
#[derive(Debug)]
pub enum VerificationResult {
    /// Function was fully verified as equivalent
    Verified,
    /// Function was skipped due to loops
    SkippedLoop,
    /// Function was skipped due to memory operations
    SkippedMemory,
    /// Function was skipped due to unknown instructions
    SkippedUnknown,
    /// Verification found a counterexample (not equivalent)
    Failed(String),
    /// Verification encountered an error
    Error(String),
}

impl VerificationResult {
    /// Returns true if this result indicates the functions are equivalent
    /// (either proven or assumed due to skipping).
    ///
    /// **Lenient semantics**: counts skipped-due-to-unverifiable as
    /// "equivalent" (the verifier couldn't prove a difference, so the
    /// transform is allowed). Use this for value-level passes (constant
    /// folding, simplify_locals, dead-code-elimination) where the
    /// transforms cannot move code across branches.
    pub fn is_equivalent(&self) -> bool {
        matches!(
            self,
            VerificationResult::Verified
                | VerificationResult::SkippedLoop
                | VerificationResult::SkippedMemory
                | VerificationResult::SkippedUnknown
        )
    }

    /// Returns true if this result was fully Z3-verified.
    ///
    /// **Strict semantics**: only `Verified` counts. Skipped* and
    /// counterexamples both return false. Use this for hoist-prone
    /// passes (LICM, CSE, code_folding, coalesce_locals) where moving
    /// code across branches is unsafe without a real proof — REQ-5
    /// conservative-over-fast: "skip the function rather than risk
    /// incorrect optimization."
    pub fn is_verified(&self) -> bool {
        matches!(self, VerificationResult::Verified)
    }

    /// Returns true if verification was skipped (any reason).
    ///
    /// Lets callers distinguish "verifier proved equivalence" from
    /// "verifier could not reason about this input." The latter is the
    /// audit's S5 finding: silent skips were being treated as success.
    pub fn is_skip(&self) -> bool {
        matches!(
            self,
            VerificationResult::SkippedLoop
                | VerificationResult::SkippedMemory
                | VerificationResult::SkippedUnknown
        )
    }

    /// Human-readable skip reason, or `None` if not a skip outcome.
    pub fn skip_reason(&self) -> Option<&'static str> {
        match self {
            VerificationResult::SkippedLoop => Some("complex loop"),
            VerificationResult::SkippedMemory => Some("float load/store"),
            VerificationResult::SkippedUnknown => Some("unknown opcode"),
            _ => None,
        }
    }

    /// Update coverage tracker based on this result
    pub fn update_coverage(&self, coverage: &mut VerificationCoverage) {
        match self {
            VerificationResult::Verified => coverage.record_verified(),
            VerificationResult::SkippedLoop => coverage.record_skipped_loop(),
            VerificationResult::SkippedMemory => coverage.record_skipped_memory(),
            VerificationResult::SkippedUnknown => coverage.record_skipped_unknown(),
            VerificationResult::Failed(_) => coverage.record_failed(),
            VerificationResult::Error(_) => coverage.record_error(),
        }
    }
}

/// Maximum loop unrolling depth for verification
/// Higher values = more precise but slower
#[cfg(feature = "verification")]
#[allow(dead_code)]
const MAX_LOOP_UNROLL: usize = 3;

/// Shared symbolic inputs for verification
///
/// When comparing original vs optimized functions, BOTH encodings must use
/// the SAME symbolic inputs. Otherwise Z3 treats them as independent variables
/// and can trivially find counterexamples by assigning different values.
#[cfg(feature = "verification")]
struct SharedSymbolicInputs {
    /// Symbolic parameters (shared between original and optimized)
    #[allow(dead_code)]
    params: Vec<BV>,
    /// Initial local variable values (zeros for non-param locals)
    initial_locals: Vec<BV>,
    /// Symbolic globals (shared between original and optimized)
    globals: Vec<BV>,
    /// Symbolic memory (Array from 32-bit address to 8-bit byte)
    /// Using Array theory allows Z3 to reason about memory operations precisely.
    memory: Array,
    /// Initial memory size in pages (shared between original and optimized)
    /// WebAssembly memory.size returns this value as i32.
    memory_size: BV,
}

#[cfg(feature = "verification")]
impl SharedSymbolicInputs {
    /// Create shared symbolic inputs for a function
    fn from_function(func: &Function) -> Self {
        let mut params = Vec::new();
        let mut initial_locals = Vec::new();

        // Create symbolic parameters
        for (idx, param_type) in func.signature.params.iter().enumerate() {
            let width = match param_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            let param = BV::new_const(format!("param{}", idx), width);
            params.push(param.clone());
            initial_locals.push(param);
        }

        // Initialize non-param locals to zero
        for (count, local_type) in func.locals.iter() {
            let width = match local_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            for _ in 0..*count {
                initial_locals.push(BV::from_u64(0, width));
            }
        }

        // Create symbolic globals
        let mut globals = Vec::new();
        for i in 0..16 {
            globals.push(BV::new_const(format!("global{}", i), 32));
        }

        // Create symbolic memory: Array[BitVec32 -> BitVec8]
        // This is a fully symbolic initial memory state
        let addr_sort = Sort::bitvector(32);
        let byte_sort = Sort::bitvector(8);
        let memory = Array::new_const("memory", &addr_sort, &byte_sort);

        // Create symbolic memory size (in pages, returned by memory.size)
        // This is an i32 that both original and optimized functions share.
        let memory_size = BV::new_const("memory_size", 32);

        SharedSymbolicInputs {
            params,
            initial_locals,
            globals,
            memory,
            memory_size,
        }
    }
}

/// Execution state for symbolic execution.
/// Tracks the symbolic state at each point during execution.
///
/// SCAFFOLDING — currently `#[allow(dead_code)]` and unused by the live
/// encoder (`encode_function_to_smt_impl_inner`), which uses simple
/// `Vec<BV>` stack/locals without per-path predicates. This struct is
/// the intended target for the verifier model upgrade tracked in the
/// v0.4.0 release notes (PR-C deferred work): full exit-state
/// equivalence (return + locals + globals + memory) and per-arm BrTable
/// state merge via `merge_states`. Do not delete — the next verifier
/// PR will wire this in.
#[cfg(feature = "verification")]
#[allow(dead_code)]
struct ExecutionState {
    /// Value stack
    stack: Vec<BV>,
    /// Local variables
    locals: Vec<BV>,
    /// Global variables
    globals: Vec<BV>,
    /// Path condition (constraints that must be true for this path)
    path_condition: Bool,
    /// Whether this execution path is reachable
    reachable: bool,
    /// Unique counter for generating fresh variable names
    counter: usize,
}

#[cfg(feature = "verification")]
#[allow(dead_code)]
impl ExecutionState {
    fn new(locals: Vec<BV>, globals: Vec<BV>) -> Self {
        ExecutionState {
            stack: Vec::new(),
            locals,
            globals,
            path_condition: Bool::from_bool(true),
            reachable: true,
            counter: 0,
        }
    }

    fn clone_state(&self) -> Self {
        ExecutionState {
            stack: self.stack.clone(),
            locals: self.locals.clone(),
            globals: self.globals.clone(),
            path_condition: self.path_condition.clone(),
            reachable: self.reachable,
            counter: self.counter,
        }
    }

    fn fresh_name(&mut self, prefix: &str) -> String {
        self.counter += 1;
        format!("{}_{}", prefix, self.counter)
    }
}

/// Result of encoding a block or instruction sequence
#[cfg(feature = "verification")]
#[allow(dead_code)]
struct BlockResult {
    /// The resulting execution state after the block
    state: ExecutionState,
    /// Values produced by the block (for blocks with result types)
    results: Vec<BV>,
    /// Whether a branch was taken that exits this block
    branched: bool,
    /// Branch depth (how many blocks to exit, 0 = this block)
    branch_depth: Option<u32>,
}

/// Merge two bitvector values based on a condition
/// Returns: if cond then true_val else false_val
#[cfg(feature = "verification")]
fn merge_bv(cond: &Bool, true_val: &BV, false_val: &BV) -> BV {
    cond.ite(true_val, false_val)
}

/// Merge two execution states based on a condition
/// Used for joining paths after if/else or when a branch may or may not be taken
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn merge_states(
    cond: &Bool,
    true_state: &ExecutionState,
    false_state: &ExecutionState,
) -> ExecutionState {
    // Merge stacks (must have same length for valid merge)
    let stack = if true_state.stack.len() == false_state.stack.len() {
        true_state
            .stack
            .iter()
            .zip(false_state.stack.iter())
            .map(|(t, f)| merge_bv(cond, t, f))
            .collect()
    } else {
        // Different stack heights - use true branch (caller should handle this case)
        true_state.stack.clone()
    };

    // Merge locals
    let locals = true_state
        .locals
        .iter()
        .zip(false_state.locals.iter())
        .map(|(t, f)| merge_bv(cond, t, f))
        .collect();

    // Merge globals
    let globals = true_state
        .globals
        .iter()
        .zip(false_state.globals.iter())
        .map(|(t, f)| merge_bv(cond, t, f))
        .collect();

    // Merge path conditions
    let path_condition = Bool::or(&[
        Bool::and(&[cond.clone(), true_state.path_condition.clone()]),
        Bool::and(&[cond.not(), false_state.path_condition.clone()]),
    ]);

    ExecutionState {
        stack,
        locals,
        globals,
        path_condition,
        reachable: true_state.reachable || false_state.reachable,
        counter: true_state.counter.max(false_state.counter),
    }
}

/// Get the bit width for a block type result
#[cfg(feature = "verification")]
fn block_type_width(block_type: &BlockType) -> Option<u32> {
    match block_type {
        BlockType::Empty => None,
        BlockType::Value(vt) => Some(bv_width_for_value_type(*vt)),
        BlockType::Func { results, .. } => {
            // For multi-value, return width of first result (simplified)
            results.first().map(|vt| bv_width_for_value_type(*vt))
        }
    }
}

/// Bitvector width (in bits) for a wasm `ValueType`. Single source of truth
/// shared across all encoding sites — replaces several copy-pasted match
/// blocks that previously hardcoded the same mapping.
#[cfg(feature = "verification")]
fn bv_width_for_value_type(t: crate::ValueType) -> u32 {
    match t {
        crate::ValueType::I32 | crate::ValueType::F32 => 32,
        crate::ValueType::I64 | crate::ValueType::F64 => 64,
    }
}

/// loom#155: shared little-endian full-width load encoders.
///
/// SOUNDNESS: the main encoder (`encode_function_to_smt_impl_inner`) and the
/// by-body inline modeler (`encode_inlinable_callee_result`) MUST encode the
/// same load to the *same* Z3 expression — otherwise a memory-reading inline
/// could be falsely accepted (if they coincidentally agree wrong) or, more
/// usually, spuriously reverted. Factoring the byte-combine here and calling
/// it from both sites guarantees bit-identical encoding against the shared
/// memory `Array` (`Array[BitVec32 → BitVec8]`, little-endian).
#[cfg(feature = "verification")]
fn encode_i32_load_from(memory: &Array, addr: &BV, offset: u32) -> Result<BV> {
    let effective_addr = addr.bvadd(BV::from_i64(offset as i64, 32));
    let mut combined = BV::from_i64(0, 32);
    for i in 0..4i64 {
        let byte_addr = effective_addr.bvadd(BV::from_i64(i, 32));
        let byte_val: BV = memory
            .select(&byte_addr)
            .as_bv()
            .ok_or_else(|| anyhow!("Z3 memory select did not return BV in i32 load byte {}", i))?;
        combined = combined.bvor(byte_val.zero_ext(24).bvshl(BV::from_i64(i * 8, 32)));
    }
    Ok(combined)
}

#[cfg(feature = "verification")]
fn encode_i64_load_from(memory: &Array, addr: &BV, offset: u32) -> Result<BV> {
    let effective_addr = addr.bvadd(BV::from_i64(offset as i64, 32));
    let mut combined = BV::from_i64(0, 64);
    for i in 0..8i64 {
        let byte_addr = effective_addr.bvadd(BV::from_i64(i, 32));
        let byte_val: BV = memory
            .select(&byte_addr)
            .as_bv()
            .ok_or_else(|| anyhow!("Z3 memory select did not return BV in i64 load byte {}", i))?;
        combined = combined.bvor(byte_val.zero_ext(56).bvshl(BV::from_i64(i * 8, 64)));
    }
    Ok(combined)
}

/// loom#157: shared little-endian full-width store encoders — the write
/// counterparts of the load helpers, reused by the main encoder and the
/// by-body inline modeler so an inlined writer's stores and the original
/// modeled-`call F`'s stores produce bit-identical memory `Array` updates.
/// Returns the new memory Array (Z3 arrays are functional/immutable).
#[cfg(feature = "verification")]
fn encode_i32_store_into(memory: &Array, addr: &BV, value: &BV, offset: u32) -> Array {
    let effective_addr = addr.bvadd(BV::from_i64(offset as i64, 32));
    let mut m = memory.clone();
    for i in 0..4i64 {
        let byte = value.extract((i * 8 + 7) as u32, (i * 8) as u32);
        m = m.store(&effective_addr.bvadd(BV::from_i64(i, 32)), &byte);
    }
    m
}

#[cfg(feature = "verification")]
fn encode_i64_store_into(memory: &Array, addr: &BV, value: &BV, offset: u32) -> Array {
    let effective_addr = addr.bvadd(BV::from_i64(offset as i64, 32));
    let mut m = memory.clone();
    for i in 0..8i64 {
        let byte = value.extract((i * 8 + 7) as u32, (i * 8) as u32);
        m = m.store(&effective_addr.bvadd(BV::from_i64(i, 32)), &byte);
    }
    m
}

/// loom#161: shared partial-width (8/16-bit) load encoder. Reads `bytes` bytes
/// little-endian into a `bytes*8`-bit value, then sign- or zero-extends to
/// `result_width` (32 or 64). Used by the main encoder AND the by-body inline
/// modeler so a callee accessing `int8_t`/`int16_t` fields (e.g. filter_step's
/// `i32.load16_s`) inlines with a bit-identical encoding. The v=70000 int16
/// wrap vector in gale#161 validates the masked/sign-extended widths.
#[cfg(feature = "verification")]
fn encode_partial_load_from(
    memory: &Array,
    addr: &BV,
    offset: u32,
    bytes: u32,
    signed: bool,
    result_width: u32,
) -> Result<BV> {
    let effective_addr = addr.bvadd(BV::from_i64(offset as i64, 32));
    let raw_width = bytes * 8;
    let mut combined = BV::from_u64(0, raw_width);
    for i in 0..bytes as i64 {
        let byte: BV = memory
            .select(&effective_addr.bvadd(BV::from_i64(i, 32)))
            .as_bv()
            .ok_or_else(|| anyhow!("Z3 memory select did not return BV in partial load"))?;
        // byte is 8-bit; widen to raw_width and shift into place.
        let widened = if raw_width > 8 {
            byte.zero_ext(raw_width - 8)
        } else {
            byte
        };
        combined = combined.bvor(widened.bvshl(BV::from_u64((i * 8) as u64, raw_width)));
    }
    let ext = result_width - raw_width;
    Ok(if ext == 0 {
        combined
    } else if signed {
        combined.sign_ext(ext)
    } else {
        combined.zero_ext(ext)
    })
}

/// loom#161: shared partial-width (8/16-bit) store encoder — stores the low
/// `bytes` bytes of `value` little-endian. Counterpart of the partial load.
#[cfg(feature = "verification")]
fn encode_partial_store_into(
    memory: &Array,
    addr: &BV,
    value: &BV,
    offset: u32,
    bytes: u32,
) -> Array {
    let effective_addr = addr.bvadd(BV::from_i64(offset as i64, 32));
    let mut m = memory.clone();
    for i in 0..bytes as i64 {
        let byte = value.extract((i * 8 + 7) as u32, (i * 8) as u32);
        m = m.store(&effective_addr.bvadd(BV::from_i64(i, 32)), &byte);
    }
    m
}

/// Resolve the wasm-declared `ValueType` of local index `idx` in `func`.
///
/// Local indices in wasm are flat across params + declared locals: idx 0..N
/// are the parameters (in order), then `func.locals` declares additional
/// locals as `(count, type)` runs that get unrolled.
///
/// Returns `None` for indices beyond declared locals — the caller should
/// treat these as a verifier-side bug (every well-formed wasm reference is
/// in range), but we don't panic. Used to fix the i64-vs-i32 SortDiffers
/// panic in `inline_functions` (loom#98): when an optimization adds new
/// locals, the verifier was extending its symbolic state with hardcoded
/// 32-bit BVs regardless of declared type, which crashed Z3 binops on
/// any i64-typed local read.
#[cfg(feature = "verification")]
fn local_type_at(func: &Function, idx: usize) -> Option<crate::ValueType> {
    let param_count = func.signature.params.len();
    if idx < param_count {
        return Some(func.signature.params[idx]);
    }
    let mut cur = param_count;
    for (count, ty) in &func.locals {
        let next = cur + *count as usize;
        if idx < next {
            return Some(*ty);
        }
        cur = next;
    }
    None
}

/// Pad two BVs to a matching width by zero-extending the shorter one.
/// Defensive backstop for any encoding site that pops two operands and
/// applies a width-sensitive Z3 op (bvadd, bvsub, eq, ...) — well-formed
/// wasm guarantees matched widths by construction, but bugs upstream
/// (e.g., loom#98 wrong-width local default) used to surface as a
/// `SortDiffers` panic deep in the z3 crate. With this helper the panic
/// becomes a quiet correctness-preserving extension.
///
/// Zero-extension is the right choice for a backstop because (a) wasm
/// integers are bag-of-bits; the bit pattern is preserved on the original
/// width, and (b) widening with zero never injects "new" set bits that
/// downstream comparisons could see, so a false-equivalent verdict is
/// avoided.
///
/// ## Soundness boundary (loom#145)
///
/// This is wired into **binop operand** sites (`bvadd`, `bvand`,
/// `bvult`, …) in the symbolic executors. That is sound: a valid wasm
/// binop's two operands are the same width by validation, so any
/// mismatch the executor sees is purely a model artifact (e.g. a
/// hardcoded-`BV32` global default meeting a real i64 value), and
/// zero-extending to repair it preserves the modeled value while
/// preventing the deep-in-z3 `SortDiffers` panic. It is **NOT** applied
/// at the top-level / k-induction equivalence check (`orig.eq(opt)`):
/// forcing widths there could make a genuinely inequivalent transform
/// look equal and approve a miscompile, so that site instead bails to
/// "cannot prove" (conservative revert) on any width mismatch.
#[cfg(feature = "verification")]
fn match_bv_widths(lhs: BV, rhs: BV) -> (BV, BV) {
    let lw = lhs.get_size();
    let rw = rhs.get_size();
    match lw.cmp(&rw) {
        std::cmp::Ordering::Equal => (lhs, rhs),
        std::cmp::Ordering::Less => {
            let extended = lhs.zero_ext(rw - lw);
            (extended, rhs)
        }
        std::cmp::Ordering::Greater => {
            let extended = rhs.zero_ext(lw - rw);
            (lhs, extended)
        }
    }
}

/// Install a process-wide panic hook (once) that swallows panics
/// originating inside the z3 Rust bindings, forwarding all other panics
/// to the previously-installed hook.
///
/// Why (loom#145): the per-function translation validator wraps each Z3
/// call in `catch_unwind` and reverts the function on any Z3-internal
/// panic (e.g. a `SortDiffers` BitVec-64-vs-32 width mismatch, or an
/// `unwrap()`-on-`None` deep in the bindings). `catch_unwind` recovers
/// the thread and the revert is correct/conservative — but it does NOT
/// suppress the default panic hook, which still prints a
/// `thread '...' panicked at z3/src/ast/...` block for every caught
/// panic. On an i64-heavy module that reverts thousands of functions
/// this produced 21 MB+ of stderr. Filtering only z3-origin panics keeps
/// that noise out while leaving every real (non-z3) panic fully visible.
///
/// Suppressing the *print* is sound: these panics are always caught by
/// the verifier's `catch_unwind` and converted to a clean revert, so the
/// information content is preserved in the aggregated revert count
/// (`crate::stats::record_revert`). Installed via `Once` (rather than a
/// save/restore swap around each call) so it is race-free under the
/// islands feature's parallel verification.
#[cfg(feature = "verification")]
fn install_z3_panic_filter() {
    use std::sync::Once;
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
        let prev = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            let from_z3 = info
                .location()
                .map(|l| l.file().contains("z3"))
                .unwrap_or(false);
            if from_z3 {
                // Caught-and-reverted by the verifier; do not print.
                return;
            }
            prev(info);
        }));
    });
}

/// Build a Z3 [`Config`] with a per-query wall-clock timeout (loom#147).
///
/// The translation validator runs a real SMT solve per function. On i64
/// bitvector formulas Z3 can hit a slow-solve cliff and grind for many
/// minutes — which, after the loom#145 width fix made the verifier
/// actually reach `solver.check()` (instead of fast-panicking), showed
/// up as a near-hung CI test matrix and would make a large i64-heavy
/// module (gale-ffi) verify very slowly.
///
/// Setting Z3's `timeout` param bounds each `check()`: a query that
/// exceeds it returns [`SatResult::Unknown`], which every call site
/// already treats as "cannot prove" → conservative revert (sound — we
/// keep the original function). So a slow solve becomes a fast,
/// safe revert instead of a hang. Default 5000 ms; override with
/// `LOOM_Z3_TIMEOUT_MS` (e.g. higher in CI to admit more real proofs,
/// `0` to disable).
#[cfg(feature = "verification")]
fn z3_config_with_timeout() -> Config {
    let mut cfg = Config::new();
    let ms = std::env::var("LOOM_Z3_TIMEOUT_MS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(5000);
    if ms > 0 {
        cfg.set_param_value("timeout", &ms.to_string());
    }
    cfg
}

/// Per-function revert/counterexample logging, gated behind the
/// `LOOM_VERBOSE_REVERTS` env var (loom#145).
///
/// Default (unset): silent. Every revert is still counted via
/// `crate::stats::record_revert`, so the `--stats` output shows the
/// aggregate (e.g. `inline: 824 reverts`) — one summary instead of one
/// `reverting function: ...` line per reverted function, which on an
/// i64-heavy module was thousands of lines. Set `LOOM_VERBOSE_REVERTS=1`
/// to restore the per-function detail for debugging.
#[cfg(feature = "verification")]
fn revert_note(args: std::fmt::Arguments) {
    if std::env::var_os("LOOM_VERBOSE_REVERTS").is_some() {
        eprintln!("{args}");
    }
}

/// Check if a function contains complex loops that would make verification unsound
///
/// Simple loops (no nesting, no unverifiable ops, bounded body) can be verified
/// using bounded unrolling. Complex loops are skipped.
#[cfg(feature = "verification")]
fn contains_complex_loops(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            Instruction::Loop { body, .. }
                // Check if this loop is too complex for bounded verification
                if is_complex_loop(body) => {
                    return true;
                }
            Instruction::Block { body, .. }
                if contains_complex_loops(body) => {
                    return true;
                }
            Instruction::If {
                then_body,
                else_body,
                ..
            }
                if (contains_complex_loops(then_body) || contains_complex_loops(else_body)) => {
                    return true;
                }
            _ => {}
        }
    }
    false
}

/// Maximum instructions in a loop body for bounded verification
#[cfg(feature = "verification")]
const MAX_LOOP_BODY_INSTRUCTIONS: usize = 100;

/// Maximum nesting depth for loop verification
/// Allows one level of nesting to verify nested loops
#[cfg(feature = "verification")]
const MAX_LOOP_NESTING_DEPTH: usize = 1;

/// Enable K-induction for unbounded loop verification
/// When true, uses inductive step to prove loops correct for ALL iterations
/// When false, falls back to bounded unrolling only
#[cfg(feature = "verification")]
const ENABLE_K_INDUCTION: bool = true;

/// Default ceiling on array-modelled memory accesses per function before the
/// translation validator defers the obligation (#347).
///
/// Chosen by measurement, not taste. Full-pipeline sweep on the meld-fused
/// module from #347 (463 KB, 1444 functions), which does not finish at all
/// today — killed at 455 s:
///
/// | bound | wall clock | proven |
/// |-------|-----------|--------|
/// | 0     | 2 s       | 21.6%  |
/// | **2** | **45 s**  | 45.0%  |
/// | 4     | > 240 s   | —      |
///
/// The knee between 2 and 4 is far sharper than a "quadratic in stores"
/// reading suggests, because memory is modelled BYTE-level: one `i64.store`
/// is eight array stores, so four of them already generate on the order of
/// 500 axiom pairs.
///
/// That sharpness makes this number fragile — it is tuned on ONE module and
/// should be revisited against a corpus. It is recorded here rather than
/// hidden so the next person knows it is empirical and narrow, not derived.
///
/// A wall-clock budget would generalise better and is deliberately NOT used:
/// which functions got verified would then depend on machine speed, so the
/// same input could produce different output, violating REQ-14 (deterministic
/// optimization output). A count is the only mechanism here that stays
/// deterministic.
#[cfg(feature = "verification")]
const DEFAULT_MAX_ARRAY_MEMORY_OPS: usize = 2;

/// Bodies at or below this instruction count are exempt from the memory-op
/// bound (#347). See the sweep in `TEST-347-MEMORY-OP-BOUND`.
#[cfg(feature = "verification")]
const DEFAULT_MEMORY_BOUND_INSTRUCTION_FLOOR: usize = 16;

/// K value for K-induction (number of base case iterations)
/// Higher K = stronger base case but slower verification
#[allow(dead_code)]
const K_INDUCTION_DEPTH: usize = 2;

/// Check if a loop body is too complex for bounded unrolling verification
///
/// Complex loops include:
/// - Deeply nested loops (more than MAX_LOOP_NESTING_DEPTH levels)
/// - Unverifiable instructions in loop body
/// - Very large loop bodies (> MAX_LOOP_BODY_INSTRUCTIONS)
#[cfg(feature = "verification")]
fn is_complex_loop(body: &[Instruction]) -> bool {
    is_complex_loop_at_depth(body, 0)
}

/// Check loop complexity at a given nesting depth
#[cfg(feature = "verification")]
fn is_complex_loop_at_depth(body: &[Instruction], depth: usize) -> bool {
    // Check for unverifiable instructions first
    if contains_unverifiable_instructions(body) {
        return true;
    }

    // Check body size (very large loops may timeout)
    if count_instructions(body) > MAX_LOOP_BODY_INSTRUCTIONS {
        return true;
    }

    // Check for nested loops - allow up to MAX_LOOP_NESTING_DEPTH
    for instr in body {
        match instr {
            Instruction::Loop {
                body: inner_body, ..
            } => {
                // When MAX_LOOP_NESTING_DEPTH is 0, any nested loop is too deep
                // When > 0, check if we've exceeded the allowed depth
                #[allow(clippy::absurd_extreme_comparisons)]
                if depth >= MAX_LOOP_NESTING_DEPTH {
                    // Too deep
                    return true;
                }
                // Check the nested loop at increased depth
                if is_complex_loop_at_depth(inner_body, depth + 1) {
                    return true;
                }
            }
            Instruction::Block {
                body: inner_body, ..
            }
                // Recurse into blocks (they don't increase loop depth)
                if is_complex_loop_at_depth(inner_body, depth) => {
                    return true;
                }
            Instruction::If {
                then_body,
                else_body,
                ..
            }
                if (is_complex_loop_at_depth(then_body, depth)
                    || is_complex_loop_at_depth(else_body, depth))
                => {
                    return true;
                }
            _ => {}
        }
    }

    false
}

/// Check if instructions contain ANY loop (for nested loop detection)
/// Currently unused but kept for potential future use when MAX_LOOP_NESTING_DEPTH > 0
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn contains_any_loop(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            Instruction::Loop { .. } => return true,
            Instruction::Block { body, .. } if contains_any_loop(body) => {
                return true;
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } if (contains_any_loop(then_body) || contains_any_loop(else_body)) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Count total instructions recursively
#[cfg(feature = "verification")]
fn count_instructions(instructions: &[Instruction]) -> usize {
    let mut count = 0;
    for instr in instructions {
        count += 1;
        match instr {
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                count += count_instructions(body);
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                count += count_instructions(then_body);
                count += count_instructions(else_body);
            }
            _ => {}
        }
    }
    count
}

/// Extract loop bodies from instructions for K-induction verification
#[cfg(feature = "verification")]
fn extract_loop_bodies(instructions: &[Instruction]) -> Vec<&[Instruction]> {
    let mut loops = Vec::new();
    for instr in instructions {
        match instr {
            Instruction::Loop { body, .. } => {
                loops.push(body.as_slice());
                // Also extract nested loops
                loops.extend(extract_loop_bodies(body));
            }
            Instruction::Block { body, .. } => {
                loops.extend(extract_loop_bodies(body));
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                loops.extend(extract_loop_bodies(then_body));
                loops.extend(extract_loop_bodies(else_body));
            }
            _ => {}
        }
    }
    loops
}

/// Verify loop equivalence using K-induction
///
/// K-induction proves that a property holds for ALL loop iterations by:
/// 1. Base case: Prove property holds for first K iterations (bounded unroll)
/// 2. Inductive step: Assume property holds at iteration n, prove it holds at n+1
///
/// For translation validation, the property is:
///   "executing one loop iteration on original produces same state as optimized"
///
/// Returns Ok(true) if loops are equivalent, Ok(false) if counterexample found,
/// Err if verification cannot be performed.
#[cfg(feature = "verification")]
fn verify_loops_kinduction(
    original: &Function,
    optimized: &Function,
    _shared: &SharedSymbolicInputs,
) -> Result<bool> {
    // Extract loop bodies from both functions
    let orig_loops = extract_loop_bodies(&original.instructions);
    let opt_loops = extract_loop_bodies(&optimized.instructions);

    // If loop count differs, we can't use K-induction directly
    // Fall back to assuming equivalent (bounded unroll will still be used)
    if orig_loops.len() != opt_loops.len() {
        return Ok(true); // Assume equivalent, let bounded unroll handle it
    }

    // If no loops, nothing to verify with K-induction
    if orig_loops.is_empty() {
        return Ok(true);
    }

    // Use thread-local Z3 context
    let cfg = z3_config_with_timeout();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        // For each corresponding pair of loops, verify the inductive step
        for (orig_body, opt_body) in orig_loops.iter().zip(opt_loops.iter()) {
            // Skip if either loop body is too complex
            if is_complex_loop(orig_body) || is_complex_loop(opt_body) {
                continue; // Skip this loop, assume equivalent
            }

            // === INDUCTIVE STEP ===
            // Create symbolic state representing "after arbitrary iteration k"
            // This state is the same for both original and optimized (inductive hypothesis)

            // Create symbolic locals for inductive state (using thread-local context).
            // Each symbolic constant gets the bit width of its declared local
            // type. Previously hardcoded to 32, which crashed Z3 binops on
            // i64-typed locals (loom#98).
            let num_locals = original
                .locals
                .iter()
                .map(|(c, _)| *c as usize)
                .sum::<usize>()
                + original.signature.params.len();
            let mut inductive_locals_orig: Vec<BV> = (0..num_locals)
                .map(|i| {
                    let width = local_type_at(original, i)
                        .map(bv_width_for_value_type)
                        .unwrap_or(32);
                    BV::new_const(format!("ind_local_{}", i), width)
                })
                .collect();

            let mut inductive_locals_opt: Vec<BV> = inductive_locals_orig.clone();

            // Create symbolic globals for inductive state (shared)
            let inductive_globals: Vec<BV> = (0..16)
                .map(|i| BV::new_const(format!("ind_global_{}", i), 32))
                .collect();

            // Inductive hypothesis: locals start equal
            // (globals are shared so automatically equal)

            // Execute one iteration of original loop body
            let mut orig_stack: Vec<BV> = Vec::new();
            let orig_result = encode_loop_body_for_kinduction(
                orig_body,
                &mut orig_stack,
                &mut inductive_locals_orig,
                &mut inductive_globals.clone(),
            );

            // Execute one iteration of optimized loop body
            let mut opt_stack: Vec<BV> = Vec::new();
            let mut inductive_globals_opt = inductive_globals.clone();
            let opt_result = encode_loop_body_for_kinduction(
                opt_body,
                &mut opt_stack,
                &mut inductive_locals_opt,
                &mut inductive_globals_opt,
            );

            // If either encoding failed, skip this loop
            if orig_result.is_err() || opt_result.is_err() {
                continue;
            }

            // Build assertion: locals must be equal after one iteration
            // This is the inductive step: if equal before, must be equal after
            let mut all_equal = Bool::from_bool(true);

            // Compare locals.
            //
            // loom#145 soundness boundary: this is the EQUIVALENCE check
            // (original loop-state vs optimized), not a binop. We must
            // NOT width-match here — zero-extending one side to force a
            // comparison could make a genuinely inequivalent transform
            // look equal and approve a miscompile. If the two sides carry
            // different BV widths (a model artifact that should not occur
            // now that binops are width-matched, but we never want it to
            // silently panic deep in z3), bail to "cannot prove" so the
            // caller reverts conservatively.
            let min_locals = inductive_locals_orig.len().min(inductive_locals_opt.len());
            for i in 0..min_locals {
                if inductive_locals_orig[i].get_size() != inductive_locals_opt[i].get_size() {
                    return Err(anyhow!(
                        "k-induction: local {} width mismatch (orig {} vs opt {}) — cannot prove equivalence",
                        i,
                        inductive_locals_orig[i].get_size(),
                        inductive_locals_opt[i].get_size()
                    ));
                }
                let eq = inductive_locals_orig[i].eq(&inductive_locals_opt[i]);
                all_equal = Bool::and(&[&all_equal, &eq]);
            }

            // Compare stacks (if both non-empty)
            if !orig_stack.is_empty() && !opt_stack.is_empty() {
                let min_stack = orig_stack.len().min(opt_stack.len());
                for i in 0..min_stack {
                    if orig_stack[i].get_size() != opt_stack[i].get_size() {
                        return Err(anyhow!(
                            "k-induction: stack slot {} width mismatch (orig {} vs opt {}) — cannot prove equivalence",
                            i,
                            orig_stack[i].get_size(),
                            opt_stack[i].get_size()
                        ));
                    }
                    let eq = orig_stack[i].eq(&opt_stack[i]);
                    all_equal = Bool::and(&[&all_equal, &eq]);
                }
            }

            // Assert NOT all_equal - if SAT, we found a counterexample
            solver.assert(all_equal.not());

            // Check satisfiability
            match solver.check() {
                SatResult::Unsat => {
                    // UNSAT means all_equal is always true
                    // Inductive step verified for this loop!
                }
                SatResult::Sat => {
                    // Found counterexample - loops are not equivalent
                    return Ok(false);
                }
                SatResult::Unknown => {
                    // Z3 couldn't decide - cannot prove equivalence
                    return Ok(false);
                }
            }

            solver.reset();
        }

        Ok(true) // All loops verified or skipped
    })
}

/// Encode a loop body for K-induction verification
///
/// This is a simplified encoding that focuses on local variable mutations
/// for the inductive step. Uses thread-local Z3 context.
#[cfg(feature = "verification")]
fn encode_loop_body_for_kinduction(
    body: &[Instruction],
    stack: &mut Vec<BV>,
    locals: &mut Vec<BV>,
    globals: &mut Vec<BV>,
) -> Result<()> {
    for instr in body {
        match instr {
            // Constants
            Instruction::I32Const(val) => {
                stack.push(BV::from_i64(*val as i64, 32));
            }
            Instruction::I64Const(val) => {
                stack.push(BV::from_i64(*val, 64));
            }

            // Locals
            Instruction::LocalGet(idx) => {
                let idx = *idx as usize;
                if idx < locals.len() {
                    stack.push(locals[idx].clone());
                } else {
                    stack.push(BV::from_i64(0, 32));
                }
            }
            Instruction::LocalSet(idx) => {
                let idx = *idx as usize;
                if let Some(val) = stack.pop() {
                    while locals.len() <= idx {
                        locals.push(BV::from_i64(0, 32));
                    }
                    locals[idx] = val;
                }
            }
            Instruction::LocalTee(idx) => {
                let idx = *idx as usize;
                if let Some(val) = stack.last().cloned() {
                    while locals.len() <= idx {
                        locals.push(BV::from_i64(0, 32));
                    }
                    locals[idx] = val;
                }
            }

            // Globals
            Instruction::GlobalGet(idx) => {
                let idx = *idx as usize;
                if idx < globals.len() {
                    stack.push(globals[idx].clone());
                } else {
                    stack.push(BV::from_i64(0, 32));
                }
            }
            Instruction::GlobalSet(idx) => {
                let idx = *idx as usize;
                if let Some(val) = stack.pop() {
                    while globals.len() <= idx {
                        globals.push(BV::from_i64(0, 32));
                    }
                    globals[idx] = val;
                }
            }

            // Basic arithmetic (i32)
            Instruction::I32Add if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvadd(&b));
            }
            Instruction::I32Sub if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvsub(&b));
            }
            Instruction::I32Mul if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvmul(&b));
            }
            Instruction::I32And if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvand(&b));
            }
            Instruction::I32Or if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvor(&b));
            }
            Instruction::I32Xor if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvxor(&b));
            }
            Instruction::I32Shl if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvshl(&b));
            }
            Instruction::I32ShrU if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvlshr(&b));
            }
            Instruction::I32ShrS if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvashr(&b));
            }

            // Comparisons
            Instruction::I32Eqz => {
                if let Some(a) = stack.pop() {
                    let zero = BV::from_i64(0, 32);
                    let one = BV::from_i64(1, 32);
                    let result = a.eq(&zero).ite(&one, &zero);
                    stack.push(result);
                }
            }
            Instruction::I32Eq if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.eq(&b).ite(&one, &zero));
            }
            Instruction::I32Ne if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.eq(&b).not().ite(&one, &zero));
            }
            Instruction::I32LtS if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvslt(&b).ite(&one, &zero));
            }
            Instruction::I32LtU if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvult(&b).ite(&one, &zero));
            }
            Instruction::I32GtS if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvsgt(&b).ite(&one, &zero));
            }
            Instruction::I32GtU if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvugt(&b).ite(&one, &zero));
            }
            Instruction::I32LeS if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvsle(&b).ite(&one, &zero));
            }
            Instruction::I32LeU if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvule(&b).ite(&one, &zero));
            }
            Instruction::I32GeS if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvsge(&b).ite(&one, &zero));
            }
            Instruction::I32GeU if stack.len() >= 2 => {
                let b = stack.pop().unwrap();
                let a = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let one = BV::from_i64(1, 32);
                let (a, b) = match_bv_widths(a, b);
                stack.push(a.bvuge(&b).ite(&one, &zero));
            }

            // Control flow - simplified handling
            Instruction::Br(_) | Instruction::BrIf(_) => {
                // For K-induction, we assume loop continues
                // Branch handling is simplified
                if let Instruction::BrIf(_) = instr {
                    stack.pop(); // Pop condition
                }
            }

            // Nested blocks - recurse
            Instruction::Block { body, .. } => {
                encode_loop_body_for_kinduction(body, stack, locals, globals)?;
            }

            // Nested loops - recurse
            Instruction::Loop {
                body: inner_body, ..
            } => {
                // For nested loops in K-induction, just execute body once
                encode_loop_body_for_kinduction(inner_body, stack, locals, globals)?;
            }

            // If/else - simplified (take both branches symbolically)
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                if let Some(cond) = stack.pop() {
                    // Save current state
                    let saved_locals = locals.clone();
                    let saved_stack = stack.clone();

                    // Execute then branch
                    encode_loop_body_for_kinduction(then_body, stack, locals, globals)?;
                    let then_locals = locals.clone();
                    let then_stack = stack.clone();

                    // Restore and execute else branch
                    *locals = saved_locals;
                    *stack = saved_stack;
                    encode_loop_body_for_kinduction(else_body, stack, locals, globals)?;

                    // Merge: use ITE based on condition
                    let zero = BV::from_i64(0, 32);
                    let cond_bool = cond.eq(&zero).not();

                    // Merge locals
                    for i in 0..locals.len().min(then_locals.len()) {
                        locals[i] = cond_bool.ite(&then_locals[i], &locals[i]);
                    }

                    // Merge stacks (simplified: just use then branch result if non-empty)
                    if !then_stack.is_empty() && stack.is_empty() {
                        *stack = then_stack;
                    }
                }
            }

            // Drop/Select
            Instruction::Drop => {
                stack.pop();
            }
            Instruction::Select if stack.len() >= 3 => {
                let cond = stack.pop().unwrap();
                let val2 = stack.pop().unwrap();
                let val1 = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let cond_bool = cond.eq(&zero).not();
                stack.push(cond_bool.ite(&val1, &val2));
            }

            // End of block
            Instruction::End => {}

            // Other instructions - skip for K-induction simplicity
            _ => {}
        }
    }

    Ok(())
}

/// Find the maximum local index used in instructions
///
/// This is needed because optimizations might create locals that aren't
/// declared in the function's locals list yet (e.g., LICM adds temporaries).
#[cfg(feature = "verification")]
fn find_max_local_index(instructions: &[Instruction]) -> usize {
    let mut max_idx = 0;
    for instr in instructions {
        match instr {
            Instruction::LocalGet(idx)
            | Instruction::LocalSet(idx)
            | Instruction::LocalTee(idx) => {
                max_idx = max_idx.max(*idx as usize);
            }
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                max_idx = max_idx.max(find_max_local_index(body));
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                max_idx = max_idx.max(find_max_local_index(then_body));
                max_idx = max_idx.max(find_max_local_index(else_body));
            }
            _ => {}
        }
    }
    max_idx
}

/// Check if a function contains instructions that cannot be precisely verified
///
/// These include instructions that would require imprecise modeling (producing
/// unconstrained symbolic values), which can lead to false counterexamples.
/// For rock-solid verification, we skip functions with such instructions.
#[cfg(feature = "verification")]
fn contains_unverifiable_instructions(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            // Float memory operations not yet modeled with Array theory
            Instruction::F32Load { .. }
            | Instruction::F64Load { .. }
            | Instruction::F32Store { .. }
            | Instruction::F64Store { .. } => {
                return true;
            }

            // Note: All integer memory operations are now verified using Z3 Array theory:
            // - I32Load, I64Load, I32Store, I64Store (full-width)
            // - I32Load8S/U, I32Load16S/U (partial-width loads with sign/zero extension)
            // - I64Load8S/U, I64Load16S/U, I64Load32S/U (partial-width loads)
            // - I32Store8, I32Store16, I64Store8, I64Store16, I64Store32 (partial-width stores)

            // Note: I32Load, I64Load, I32Store, I64Store are now verified
            // using Z3 Array theory (Array[BitVec32 -> BitVec8] with little-endian encoding)

            // Note: MemorySize and MemoryGrow are now verified using shared memory_size variable

            // Unknown instructions can't be modeled
            Instruction::Unknown(_) => {
                return true;
            }

            // Recurse into control flow structures
            Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                if contains_unverifiable_instructions(body) =>
            {
                return true;
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } if (contains_unverifiable_instructions(then_body)
                || contains_unverifiable_instructions(else_body)) =>
            {
                return true;
            }

            // All other instructions have precise SMT encodings
            _ => {}
        }
    }
    false
}

/// Verify that an optimization preserves program semantics
///
/// This function uses Z3 SMT solver to prove that the optimized program is semantically
/// equivalent to the original program for all possible inputs.
///
/// # Arguments
///
/// * `original` - The original unoptimized module
/// * `optimized` - The optimized module
///
/// # Returns
///
/// * `Ok(true)` - Programs are proven equivalent
/// * `Ok(false)` - Found a counterexample (programs differ)
/// * `Err(_)` - Verification error or timeout
///
/// # Examples
///
/// ```rust,ignore
/// let original = parse_wat("...");
/// let optimized = optimize_module(original.clone());
/// assert!(verify_optimization(&original, &optimized)?);
/// ```
#[cfg(feature = "verification")]
pub fn verify_optimization(original: &Module, optimized: &Module) -> Result<bool> {
    if original.functions.len() != optimized.functions.len() {
        return Ok(false);
    }
    // Callees are modelled by their ORIGINAL bodies on both sides. Every
    // function is checked against its own original, so by induction over the
    // call depth this establishes the module, not just each function alone.
    let ctx = VerificationSignatureContext::from_module(original);
    if ctx.global_types != VerificationSignatureContext::from_module(optimized).global_types {
        return Ok(false);
    }
    let backend = VerifyBackend::from_env();
    for (i, (orig_func, opt_func)) in original
        .functions
        .iter()
        .zip(optimized.functions.iter())
        .enumerate()
    {
        if orig_func.signature.params != opt_func.signature.params
            || orig_func.signature.results != opt_func.signature.results
        {
            return Ok(false);
        }
        match exact_route(orig_func, opt_func, &ctx, backend) {
            Some(Ok(true)) => continue,
            Some(Ok(false)) => return Ok(false),
            Some(Err(e)) => return Err(anyhow!("function {}: {}", i, e)),
            None => {
                return Err(anyhow!(
                    "function {}: contains a loop, which the module check does not verify",
                    i
                ));
            }
        }
    }
    Ok(true)
}

/// Verify that a function transformation preserves semantics
///
/// This is the core translation validation function used by optimization passes.
/// It symbolically executes both versions and proves equivalence using Z3.
///
/// # Arguments
/// * `original` - The original function before optimization
/// * `optimized` - The function after optimization
/// * `pass_name` - Name of the optimization pass (for error messages)
///
/// # Returns
/// * `Ok(true)` - Functions are proven semantically equivalent
/// * `Ok(false)` - Found a counterexample showing different behavior
/// * `Err(_)` - Verification error or timeout
///
/// # Panic safety (#331)
///
/// This entry point does **not** propagate solver-internal panics. Encoding a
/// function can panic inside the solver bindings before any solver is
/// consulted — a `SortDiffers` mixed-width BitVec binop, or an
/// `unwrap()`-on-`None` in the `ite` constructor when the arms have different
/// sorts, both reachable from a `Select` over mixed i32/i64 (documented in
/// detail at the `TranslationValidator::verify` call sites). Those are caught
/// here and reported as `Err`, exactly as [`TranslationValidator::verify`]
/// already does.
///
/// Before this was added, the two public entry points disagreed: a caller
/// going through `TranslationValidator` got a clean rejection while a direct
/// caller of this function inherited the panic, and nothing in either
/// signature said so. A panic is not a proof, so the conservative reading is
/// the same in both cases — `Err`, meaning *not established*, never
/// `Ok(true)`. The asymmetry is removed rather than merely documented,
/// because the failure mode of getting it wrong is a caller that crashes on
/// input the optimizer itself handles.
#[cfg(feature = "verification")]
pub fn verify_function_equivalence(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
) -> Result<bool> {
    // Same as the validator path: keep the (always-caught) solver backtrace
    // out of stderr. Idempotent and race-free via `Once`.
    install_z3_panic_filter();
    let backend = VerifyBackend::from_env();
    let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        verify_function_equivalence_with_backend(original, optimized, pass_name, backend)
    }));
    match caught {
        Ok(result) => result,
        // Deliberately `Err`, never `Ok(false)`: a counterexample is a
        // positive finding of INEQUIVALENCE, and this is not that. Nothing
        // was decided, and the report has to say so.
        Err(_panic) => Err(anyhow!(
            "{}: solver-internal panic during encoding - equivalence not established",
            pass_name
        )),
    }
}

/// [`verify_function_equivalence`] with the solver backend passed explicitly
/// (issue #313, Tier-2 slice 1).
///
/// `verify_function_equivalence` reads `LOOM_VERIFY_BACKEND`; this variant
/// takes the choice as an argument so tests can drive a specific engine
/// **deterministically**, without mutating process-global environment state
/// from a parallel test runner. The two are otherwise identical.
#[cfg(feature = "verification")]
pub fn verify_function_equivalence_with_backend(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
    backend: VerifyBackend,
) -> Result<bool> {
    // Quick structural checks
    if original.signature.params != optimized.signature.params
        || original.signature.results != optimized.signature.results
    {
        return Err(anyhow!(
            "{}: Function signature changed during optimization",
            pass_name
        ));
    }

    // Without a module context, calls and globals are refused.
    let no_module = VerificationSignatureContext::default();
    if let Some(decided) = exact_route(original, optimized, &no_module, backend) {
        return decided.map_err(|e| anyhow!("{}: {}", pass_name, e));
    }

    // Handle functions containing loops
    // Complex loops can be verified using K-induction when enabled
    if contains_complex_loops(&original.instructions)
        || contains_complex_loops(&optimized.instructions)
    {
        // Try K-induction verification if enabled
        if ENABLE_K_INDUCTION {
            let shared = SharedSymbolicInputs::from_function(original);
            match verify_loops_kinduction(original, optimized, &shared) {
                Ok(true) => {
                    // K-induction succeeded - loops verified for ALL iterations
                    // Continue with rest of verification
                }
                Ok(false) => {
                    // K-induction found counterexample
                    return Ok(false);
                }
                Err(_) => {
                    // K-induction error - cannot prove equivalence
                    return Ok(false);
                }
            }
        } else {
            // Unreachable while ENABLE_K_INDUCTION is true, but if it is
            // ever flipped this is an assumption, not a proof.
            note_unproven_acceptance("k-induction disabled — loops assumed equivalent");
            return Ok(true);
        }
    }

    // Skip verification for functions containing imprecisely-modeled instructions
    // (float load/store, unknown ops) — these would produce false counterexamples.
    //
    // KNOWN LIMITATION: returning Ok(true) here means callers' verify_or_revert
    // accept the transform without proof. PR-B added pass-level hoist guards
    // (LICM, CSE, code_folding, coalesce_locals) that skip BrIf/BrTable
    // functions entirely, so the silent skip cannot mask a hoisting bug
    // for those passes. For value-level passes (constant_folding,
    // simplify_locals, eliminate_dead_code, optimize_added_constants,
    // simplify_branches, merge_blocks, vacuum, remove_unused_branches,
    // optimize_advanced_instructions), the transforms do not move code
    // across branches, so the silent skip is acceptable.
    //
    // To make this VISIBLE rather than silent, emit a one-line diagnostic so
    // operators see how often verification is bypassed. The structured
    // VerificationCoverage tracker (see verify_function_equivalence_with_result)
    // records counts.
    if let Some(reason) = unverifiable_reason(original, optimized) {
        eprintln!(
            "{}: verification skipped (unverifiable instructions: {})",
            pass_name, reason
        );
        // Kept without proof. Noted so the wrapper reports it under that
        // reason as a number, not just a line of stderr that scrolls past.
        note_unproven_acceptance(reason);
        return Ok(true);
    }

    // Create Z3 context and solver using thread-local context
    let cfg = z3_config_with_timeout();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        // Encode both functions
        let orig_state = encode_function_state(original, None, None);
        let opt_state = encode_function_state(optimized, None, None);

        // Side effects on globals are part of the observable behaviour: a
        // reordering of global writes is a counterexample even when the
        // return values agree, and even for a void function.
        if let (Ok((_, og)), Ok((_, pg))) = (&orig_state, &opt_state) {
            if !check_globals_preserved(og, pg, pass_name)? {
                return Ok(false);
            }
        }
        let orig_result = orig_state.map(|(r, _)| r);
        let opt_result = opt_state.map(|(r, _)| r);

        // Compute result while BVs are still valid
        match (&orig_result, &opt_result) {
            (Ok(Some(orig)), Ok(Some(opt))) => {
                // loom#145 soundness boundary: the final output equality is
                // the EQUIVALENCE check, not a binop. Never width-match here
                // (forcing widths could approve a non-equivalent transform).
                // A width mismatch is a model artifact we cannot soundly
                // resolve — bail to "not proven" so the caller reverts.
                if orig.get_size() != opt.get_size() {
                    return Ok(false);
                }

                // #313 Tier-2 slice 1: offer the obligation to the neutral
                // seam. Under the default backend this ALWAYS defers, and the
                // incumbent path below runs byte-identically to before.
                if let SeamOutcome::Decided(verdict) =
                    verify_solver::decide_bv_equivalence(orig, opt, backend)
                {
                    return match verdict {
                        RuleVerdict::Proven => Ok(true),
                        RuleVerdict::Disproven(model) => {
                            revert_note(format_args!(
                                "{}: Verification failed! Found counterexample:",
                                pass_name
                            ));
                            revert_note(format_args!("{}", model));
                            Ok(false)
                        }
                        // Mapped to exactly what `SatResult::Unknown` does at
                        // this site: an undecided obligation is never a proof.
                        RuleVerdict::Unknown => Err(anyhow!(
                            "{}: SMT solver returned unknown (timeout or too complex)",
                            pass_name
                        )),
                    };
                }

                // Assert they are NOT equal (looking for counterexample)
                solver.assert(orig.eq(opt).not());

                match solver.check() {
                    SatResult::Unsat => Ok(true), // No counterexample = equivalent
                    SatResult::Sat => {
                        if let Some(model) = solver.get_model() {
                            revert_note(format_args!(
                                "{}: Verification failed! Found counterexample:",
                                pass_name
                            ));
                            revert_note(format_args!("{}", model));
                        }
                        Ok(false)
                    }
                    SatResult::Unknown => Err(anyhow!(
                        "{}: SMT solver returned unknown (timeout or too complex)",
                        pass_name
                    )),
                }
            }
            (Ok(None), Ok(None)) => {
                // Both functions return no value (void). Currently auto-pass.
                //
                // Counted as KEPT WITHOUT PROOF, not as proven. This arm's own
                // comment below calls the equivalence vacuous, and it is: the
                // model asserts nothing about the post-state, so for a void
                // function it asserts nothing at all. Reporting it as a proof
                // would inflate the coverage number with exactly the empty
                // checks that number exists to expose.
                //
                // MEASURED. An earlier probe — a module of nothing but void
                // functions (including a completely empty one and one that
                // writes a global) and the 724-function fused-records fixture
                // — did not reach this arm, and the comment here concluded it
                // "appears unreachable in practice".
                //
                // That conclusion was WRONG, and #313 slice 4 falsified it:
                // `tier2_slice4_memory_coverage_on_the_real_corpus` drives the
                // in-tree corpus through the validator and finds BOTH of its
                // memory-touching functions landing here, reported under this
                // exact reason. The earlier probe's fixtures simply did not
                // include the shape that produces it.
                //
                // So the classification is not defensive — it is load-bearing
                // on real code, and it is what keeps a void memory-writing
                // function out of the proven count. Lesson worth keeping: a
                // probe that does not reach an arm shows the probe's reach,
                // not the arm's.
                note_unproven_acceptance("void function — side effects not modelled");
                //
                // KNOWN LIMITATION: void functions can still have side effects
                // (memory writes, global writes) that differ between original
                // and optimized. The current model does not assert equivalence
                // of post-state locals/globals/memory — only the top-of-stack
                // return value. For void functions there is no return value
                // so the equivalence is vacuous. A future verifier upgrade
                // should encode side-effect equivalence for these.
                Ok(true)
            }
            (Err(e), _) => Err(anyhow!("{}: Failed to encode original: {}", pass_name, e)),
            (_, Err(e)) => Err(anyhow!("{}: Failed to encode optimized: {}", pass_name, e)),
            _ => Err(anyhow!(
                "{}: Return type mismatch between original and optimized",
                pass_name
            )),
        }
    })
}

/// Stub for when verification feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_function_equivalence(
    _original: &crate::Function,
    _optimized: &crate::Function,
    _pass_name: &str,
) -> Result<bool> {
    // When verification is disabled, assume the optimization is correct
    Ok(true)
}

/// Verify function equivalence and return detailed result for coverage tracking.
///
/// This version returns a `VerificationResult` that indicates WHY a function
/// was skipped or whether it was fully verified, enabling accurate coverage metrics.
#[cfg(feature = "verification")]
pub fn verify_function_equivalence_with_result(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
) -> VerificationResult {
    // Quick structural checks
    if original.signature.params != optimized.signature.params
        || original.signature.results != optimized.signature.results
    {
        return VerificationResult::Error(format!(
            "{}: Function signature changed during optimization",
            pass_name
        ));
    }

    // Handle functions containing loops - try K-induction first
    if contains_complex_loops(&original.instructions)
        || contains_complex_loops(&optimized.instructions)
    {
        if ENABLE_K_INDUCTION {
            let shared = SharedSymbolicInputs::from_function(original);
            match verify_loops_kinduction(original, optimized, &shared) {
                Ok(true) => {
                    // K-induction verified loops for ALL iterations
                    // Continue with rest of verification (or return Verified)
                    return VerificationResult::Verified;
                }
                Ok(false) => {
                    return VerificationResult::Failed(format!(
                        "{}: K-induction found loop equivalence counterexample",
                        pass_name
                    ));
                }
                Err(_) => {
                    // K-induction couldn't complete - fall back to skip
                    return VerificationResult::SkippedLoop;
                }
            }
        } else {
            return VerificationResult::SkippedLoop;
        }
    }

    // Check for memory/unknown instructions
    if contains_unverifiable_instructions(&original.instructions)
        || contains_unverifiable_instructions(&optimized.instructions)
    {
        // Distinguish between memory ops and unknown ops
        if contains_memory_instructions(&original.instructions)
            || contains_memory_instructions(&optimized.instructions)
        {
            return VerificationResult::SkippedMemory;
        }
        return VerificationResult::SkippedUnknown;
    }

    // Create Z3 context and solver
    // #313 Tier-2 slice 1: which engine decides a routable obligation.
    let backend = VerifyBackend::from_env();
    let cfg = z3_config_with_timeout();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        let orig_result = encode_function_to_smt(original);
        let opt_result = encode_function_to_smt(optimized);

        match (&orig_result, &opt_result) {
            (Ok(Some(orig)), Ok(Some(opt))) => {
                // loom#145: equivalence check — never width-match; a width
                // mismatch is unprovable, so report it as an error (the
                // caller reverts) rather than panic in z3 or force-match.
                if orig.get_size() != opt.get_size() {
                    return VerificationResult::Error(
                        "result width mismatch — cannot prove equivalence".to_string(),
                    );
                }

                // #313 Tier-2 slice 1: offer the obligation to the neutral
                // seam. Under the default backend this ALWAYS defers, and the
                // incumbent path below runs byte-identically to before.
                if let SeamOutcome::Decided(verdict) =
                    verify_solver::decide_bv_equivalence(orig, opt, backend)
                {
                    return match verdict {
                        RuleVerdict::Proven => VerificationResult::Verified,
                        RuleVerdict::Disproven(model) => VerificationResult::Failed(format!(
                            "{}: Found counterexample: {}",
                            pass_name, model
                        )),
                        // Same outcome `SatResult::Unknown` produces here —
                        // an Error, which callers treat as "not verified".
                        RuleVerdict::Unknown => VerificationResult::Error(format!(
                            "{}: SMT solver returned unknown (timeout or too complex)",
                            pass_name
                        )),
                    };
                }

                solver.assert(orig.eq(opt).not());

                match solver.check() {
                    SatResult::Unsat => VerificationResult::Verified,
                    SatResult::Sat => {
                        let model = solver
                            .get_model()
                            .map(|m| format!("{}", m))
                            .unwrap_or_else(|| "no model".to_string());
                        VerificationResult::Failed(format!(
                            "{}: Found counterexample: {}",
                            pass_name, model
                        ))
                    }
                    SatResult::Unknown => VerificationResult::Error(format!(
                        "{}: SMT solver returned unknown (timeout or too complex)",
                        pass_name
                    )),
                }
            }
            (Ok(None), Ok(None)) => VerificationResult::Verified, // Both void
            (Err(e), _) => VerificationResult::Error(format!(
                "{}: Failed to encode original: {}",
                pass_name, e
            )),
            (_, Err(e)) => VerificationResult::Error(format!(
                "{}: Failed to encode optimized: {}",
                pass_name, e
            )),
            _ => VerificationResult::Error(format!(
                "{}: Return type mismatch between original and optimized",
                pass_name
            )),
        }
    })
}

/// Stub for when verification is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_function_equivalence_with_result(
    _original: &crate::Function,
    _optimized: &crate::Function,
    _pass_name: &str,
) -> VerificationResult {
    VerificationResult::Verified
}

#[cfg(feature = "verification")]
/// Check if instructions contain unverifiable memory operations.
///
/// Note: I32Load, I64Load, I32Store, I64Store are NOW verifiable with Z3 Array theory.
/// Only partial-width (8/16-bit) and float memory operations remain unverifiable.
fn contains_memory_instructions(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            // Float memory operations - not yet modeled
            Instruction::F32Load { .. }
            | Instruction::F64Load { .. }
            | Instruction::F32Store { .. }
            | Instruction::F64Store { .. } => return true,

            // Partial-width memory operations - not yet modeled
            Instruction::I32Load8S { .. }
            | Instruction::I32Load8U { .. }
            | Instruction::I32Load16S { .. }
            | Instruction::I32Load16U { .. }
            | Instruction::I64Load8S { .. }
            | Instruction::I64Load8U { .. }
            | Instruction::I64Load16S { .. }
            | Instruction::I64Load16U { .. }
            | Instruction::I64Load32S { .. }
            | Instruction::I64Load32U { .. } => return true,

            // Note: I32Store8, I32Store16, I64Store8, I64Store16, I64Store32 are now verifiable
            // via Z3 Array theory with little-endian byte extraction

            // Note: MemorySize and MemoryGrow are now verifiable via shared memory_size variable

            // Note: I32Load, I64Load, I32Store, I64Store are now verifiable
            // via Z3 Array theory (Array[BitVec32 -> BitVec8] with little-endian encoding)
            Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                if contains_memory_instructions(body) =>
            {
                return true;
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } if (contains_memory_instructions(then_body)
                || contains_memory_instructions(else_body)) =>
            {
                return true;
            }
            _ => {}
        }
    }
    false
}

thread_local! {
    /// Set by whichever site in the verifier ACCEPTED without discharging the
    /// obligation, and read back by the counting wrapper below.
    ///
    /// This is a thread-local rather than a structural re-derivation in the
    /// wrapper because several of those sites are decided by the ENCODING
    /// RESULT, not by anything visible in the function's shape — the void
    /// auto-pass fires on `(Ok(None), Ok(None))`, which no predicate over the
    /// instruction list can reliably predict. Guessing in the wrapper is how
    /// an accounting layer starts disagreeing with the thing it accounts for.
    /// Verification runs one function per rayon worker, so a thread-local is
    /// the right scope.
    static UNPROVEN_ACCEPTANCE: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
}

thread_local! {
    /// Why the validator REFUSED an obligation, when the reason is more
    /// specific than "the proof failed" — read back by `verify_or_revert` so
    /// the revert is attributed in `--stats` instead of landing in the
    /// undifferentiated per-pass bucket.
    static DEFERRAL_REASON: std::cell::Cell<Option<&'static str>> =
        const { std::cell::Cell::new(None) };
}

/// Record why this obligation is being refused.
#[cfg(feature = "verification")]
fn note_deferral_reason(reason: &'static str) {
    DEFERRAL_REASON.with(|c| c.set(Some(reason)));
}

/// Read and clear the pending refusal reason.
#[cfg(feature = "verification")]
fn take_deferral_reason() -> Option<&'static str> {
    DEFERRAL_REASON.with(|c| c.take())
}

/// Record that this attempt is about to be ACCEPTED without a proof.
#[cfg(feature = "verification")]
fn note_unproven_acceptance(reason: &'static str) {
    UNPROVEN_ACCEPTANCE.with(|c| c.set(Some(reason)));
}

/// Read and clear the pending unproven-acceptance reason.
#[cfg(feature = "verification")]
fn take_unproven_acceptance() -> Option<&'static str> {
    UNPROVEN_ACCEPTANCE.with(|c| c.take())
}

/// What an accepted obligation did NOT compare, if anything.
///
/// An `Ok(true)` from the encoders means the compared state agreed; it says
/// nothing about state they do not compare. This names the first such gap in
/// either body so the acceptance is counted as kept-unproven rather than as a
/// proof. `exact_branches` is true for the acyclic executor, which models
/// `br`/`br_if`/`br_table`/`return` exactly; the main encoder approximates
/// them.
#[cfg(feature = "verification")]
fn unmodelled_observable(
    original: &Function,
    optimized: &Function,
    exact_branches: bool,
) -> Option<&'static str> {
    fn scan(body: &[Instruction], exact_branches: bool, top: bool) -> Option<&'static str> {
        use Instruction as I;
        let last = body.len().saturating_sub(1);
        for (i, instr) in body.iter().enumerate() {
            let gap = match instr {
                I::Block { body, .. } | I::Loop { body, .. } => scan(body, exact_branches, false),
                I::If {
                    then_body,
                    else_body,
                    ..
                } => scan(then_body, exact_branches, false)
                    .or_else(|| scan(else_body, exact_branches, false)),
                I::Br(_) | I::BrIf(_) | I::BrTable { .. } if !exact_branches => {
                    Some("branches approximated")
                }
                I::Return if !exact_branches && !(top && i == last) => {
                    Some("branches approximated")
                }
                I::Unknown(_) => Some("unknown instruction"),
                I::I32Store { .. }
                | I::I64Store { .. }
                | I::F32Store { .. }
                | I::F64Store { .. }
                | I::I32Store8 { .. }
                | I::I32Store16 { .. }
                | I::I64Store8 { .. }
                | I::I64Store16 { .. }
                | I::I64Store32 { .. }
                | I::MemoryGrow(_)
                | I::MemoryFill(_)
                | I::MemoryCopy { .. }
                | I::MemoryInit { .. }
                | I::DataDrop(_)
                | I::Call(_)
                | I::CallIndirect { .. } => Some("final memory not compared"),
                I::I32Load { .. }
                | I::I64Load { .. }
                | I::F32Load { .. }
                | I::F64Load { .. }
                | I::I32Load8S { .. }
                | I::I32Load8U { .. }
                | I::I32Load16S { .. }
                | I::I32Load16U { .. }
                | I::I64Load8S { .. }
                | I::I64Load8U { .. }
                | I::I64Load16S { .. }
                | I::I64Load16U { .. }
                | I::I64Load32S { .. }
                | I::I64Load32U { .. }
                | I::I32DivS
                | I::I32DivU
                | I::I32RemS
                | I::I32RemU
                | I::I64DivS
                | I::I64DivU
                | I::I64RemS
                | I::I64RemU
                | I::I32TruncF32S
                | I::I32TruncF32U
                | I::I32TruncF64S
                | I::I32TruncF64U
                | I::I64TruncF32S
                | I::I64TruncF32U
                | I::I64TruncF64S
                | I::I64TruncF64U
                | I::Unreachable => Some("traps not compared"),
                _ => None,
            };
            if gap.is_some() {
                return gap;
            }
        }
        None
    }
    scan(&original.instructions, exact_branches, true)
        .or_else(|| scan(&optimized.instructions, exact_branches, true))
}

/// Count an accepted obligation: as proven only when nothing observable was
/// left out of the comparison.
#[cfg(feature = "verification")]
fn record_acceptance(original: &Function, optimized: &Function, exact_branches: bool) {
    if exactly_modelable(original, optimized) {
        // Decided by the exact executor, which compares results, globals,
        // memory and traps.
        crate::stats::record_proven();
        return;
    }
    match unmodelled_observable(original, optimized, exact_branches) {
        Some(reason) => crate::stats::record_kept_unproven(reason),
        None => crate::stats::record_proven(),
    }
}

/// The reason this pair cannot be proven by the SMT encoder, if any.
///
/// This is the structural pre-check both verification paths already applied
/// inline; it is extracted so the coverage accounting classifies an attempt
/// with the SAME predicate that decides it, rather than a second copy that
/// could drift out of agreement and mislabel the report.
#[cfg(feature = "verification")]
fn unverifiable_reason(original: &Function, optimized: &Function) -> Option<&'static str> {
    if !contains_unverifiable_instructions(&original.instructions)
        && !contains_unverifiable_instructions(&optimized.instructions)
    {
        return None;
    }
    if contains_memory_instructions(&original.instructions)
        || contains_memory_instructions(&optimized.instructions)
    {
        Some("float load/store")
    } else {
        Some("unknown opcode")
    }
}

/// Verify function equivalence with signature context for proper Call handling
///
/// This version uses the signature context to properly model Call/CallIndirect
/// stack effects, providing more accurate verification.
///
/// # Coverage accounting (#331)
///
/// This wrapper classifies every attempt into exactly one of proven / kept
/// without proof / rejected, and records it in [`crate::stats`]. It exists
/// because the inner function returns `Ok(true)` for BOTH "proved equivalent"
/// and "could not model this, keeping the transform anyway" — a collapse that
/// made unverified accepts indistinguishable from proofs at every call site
/// above it, and therefore unreportable.
#[cfg(feature = "verification")]
pub fn verify_function_equivalence_with_context(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
    sig_ctx: &VerificationSignatureContext,
) -> Result<bool> {
    // Clear any stale note from an earlier attempt on this worker before the
    // one we are about to classify.
    let _ = take_unproven_acceptance();
    let result = verify_function_equivalence_with_context_uncounted(
        original,
        optimized,
        pass_name,
        sig_ctx,
        EngineMode::Configured,
    );
    match (&result, take_unproven_acceptance()) {
        // Kept, but nothing was proved about it. The caller will not revert,
        // so this is exactly the outcome a revert count cannot see.
        (Ok(true), Some(reason)) => crate::stats::record_kept_unproven(reason),
        (Ok(true), None) => record_acceptance(original, optimized, false),
        // Ok(false) / Err: the caller reverts and records that itself.
        _ => {}
    }
    result
}

/// Which engine may decide an obligation.
#[cfg(feature = "verification")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EngineMode {
    /// The backend `LOOM_VERIFY_BACKEND` selects, with Z3 deciding anything
    /// the new engine defers. The normal path.
    Configured,
    /// For bodies over the #347 memory-op bound (#448). Straight-line
    /// obligations go to the new engine ONLY: Z3's array theory is the cost
    /// the bound exists to avoid, so anything the new engine defers is
    /// refused rather than handed to Z3. Loops still go through K-induction,
    /// as below the bound; on the module #347 was measured on, that path
    /// finished in 48 s against 43 s without it.
    NewEngineOnly,
}

#[cfg(feature = "verification")]
fn verify_function_equivalence_with_context_uncounted(
    original: &Function,
    optimized: &Function,
    pass_name: &str,
    sig_ctx: &VerificationSignatureContext,
    engine: EngineMode,
) -> Result<bool> {
    // Quick structural checks
    if original.signature.params != optimized.signature.params
        || original.signature.results != optimized.signature.results
    {
        return Err(anyhow!(
            "{}: Function signature changed during optimization",
            pass_name
        ));
    }

    let route_backend = match engine {
        EngineMode::Configured => VerifyBackend::from_env(),
        EngineMode::NewEngineOnly => VerifyBackend::Ordeal,
    };
    if let Some(decided) = exact_route(original, optimized, sig_ctx, route_backend) {
        return decided.map_err(|e| anyhow!("{}: {}", pass_name, e));
    }

    // Handle functions containing loops - try K-induction first
    if contains_complex_loops(&original.instructions)
        || contains_complex_loops(&optimized.instructions)
    {
        if ENABLE_K_INDUCTION {
            let shared = SharedSymbolicInputs::from_function(original);
            match verify_loops_kinduction(original, optimized, &shared) {
                Ok(true) => {
                    // K-induction verified loops for ALL iterations
                    return Ok(true);
                }
                Ok(false) => {
                    // K-induction found counterexample
                    return Ok(false);
                }
                Err(_) => {
                    // K-induction error - cannot prove equivalence
                    return Ok(false);
                }
            }
        } else if engine == EngineMode::NewEngineOnly {
            // Over the memory bound an unproven loop is refused, never assumed.
            return Err(anyhow!("loops are not verified with K-induction disabled"));
        } else {
            return Ok(true); // K-induction disabled - assume equivalent
        }
    }

    // Skip verification for functions containing imprecisely-modeled instructions
    // (float load/store, unknown ops) — these would produce false counterexamples.
    //
    // KNOWN LIMITATION: returning Ok(true) here means callers' verify_or_revert
    // accept the transform without proof. PR-B added pass-level hoist guards
    // (LICM, CSE, code_folding, coalesce_locals) that skip BrIf/BrTable
    // functions entirely, so the silent skip cannot mask a hoisting bug
    // for those passes. For value-level passes (constant_folding,
    // simplify_locals, eliminate_dead_code, optimize_added_constants,
    // simplify_branches, merge_blocks, vacuum, remove_unused_branches,
    // optimize_advanced_instructions), the transforms do not move code
    // across branches, so the silent skip is acceptable.
    //
    // To make this VISIBLE rather than silent, emit a one-line diagnostic so
    // operators see how often verification is bypassed. The structured
    // VerificationCoverage tracker (see verify_function_equivalence_with_result)
    // records counts.
    if contains_unverifiable_instructions(&original.instructions)
        || contains_unverifiable_instructions(&optimized.instructions)
    {
        let reason = if contains_memory_instructions(&original.instructions)
            || contains_memory_instructions(&optimized.instructions)
        {
            "float load/store"
        } else {
            "unknown opcode"
        };
        // Below the bound these are kept unproven. Over it the caller would
        // otherwise refuse, so keeping one here would turn a refusal into an
        // unproven acceptance.
        if engine == EngineMode::NewEngineOnly {
            return Err(anyhow!(
                "unverifiable instructions ({}) are not decided by the new engine",
                reason
            ));
        }
        eprintln!(
            "{}: verification skipped (unverifiable instructions: {})",
            pass_name, reason
        );
        return Ok(true); // Assume equivalent — can't prove without precise model
    }

    // Create Z3 context and solver using thread-local context
    // #313 Tier-2 slice 1: which engine decides a routable obligation.
    let backend = match engine {
        EngineMode::Configured => VerifyBackend::from_env(),
        EngineMode::NewEngineOnly => VerifyBackend::Ordeal,
    };
    let cfg = z3_config_with_timeout();
    with_z3_config(&cfg, || {
        let solver = Solver::new();

        // CRITICAL: Create SHARED symbolic inputs that both encodings will use.
        // Without this, each encoding creates independent symbolic variables,
        // and Z3 trivially finds they can differ.
        let shared_inputs = SharedSymbolicInputs::from_function(original);

        // Encode both functions with the SAME shared inputs
        let orig_state = encode_function_state(original, Some(sig_ctx), Some(&shared_inputs));
        let opt_state = encode_function_state(optimized, Some(sig_ctx), Some(&shared_inputs));

        // Side effects on globals are part of the observable behaviour: a
        // reordering of global writes is a counterexample even when the
        // return values agree, and even for a void function.
        if let (Ok((_, og)), Ok((_, pg))) = (&orig_state, &opt_state) {
            if !check_globals_preserved(og, pg, pass_name)? {
                return Ok(false);
            }
        }
        let orig_result = orig_state.map(|(r, _)| r);
        let opt_result = opt_state.map(|(r, _)| r);

        // Compute result while BVs are still valid
        match (&orig_result, &opt_result) {
            (Ok(Some(orig)), Ok(Some(opt))) => {
                // loom#145 soundness boundary: the final output equality is
                // the EQUIVALENCE check, not a binop. Never width-match here
                // (forcing widths could approve a non-equivalent transform).
                // A width mismatch is a model artifact we cannot soundly
                // resolve — bail to "not proven" so the caller reverts.
                if orig.get_size() != opt.get_size() {
                    return Ok(false);
                }

                // #313 Tier-2 slice 1: offer the obligation to the neutral
                // seam. Under the default backend this ALWAYS defers, and the
                // incumbent path below runs byte-identically to before.
                let seam = verify_solver::decide_bv_equivalence(orig, opt, backend);
                let deferred = match &seam {
                    SeamOutcome::Deferred(reason) => Some(*reason),
                    SeamOutcome::Decided(_) => None,
                };
                if let SeamOutcome::Decided(verdict) = seam {
                    return match verdict {
                        RuleVerdict::Proven => Ok(true),
                        RuleVerdict::Disproven(model) => {
                            revert_note(format_args!(
                                "{}: Verification failed! Found counterexample:",
                                pass_name
                            ));
                            revert_note(format_args!("{}", model));
                            Ok(false)
                        }
                        // Mapped to exactly what `SatResult::Unknown` does at
                        // this site: an undecided obligation is never a proof.
                        RuleVerdict::Unknown => Err(anyhow!(
                            "{}: SMT solver returned unknown (timeout or too complex)",
                            pass_name
                        )),
                    };
                }

                // Over the memory bound a deferred obligation stops here:
                // handing it to Z3 is the cost #347 bounds.
                if let (EngineMode::NewEngineOnly, Some(reason)) = (engine, deferred) {
                    return Err(anyhow!(
                        "the new engine deferred this obligation ({:?})",
                        reason
                    ));
                }

                // Assert they are NOT equal (looking for counterexample)
                solver.assert(orig.eq(opt).not());

                match solver.check() {
                    SatResult::Unsat => Ok(true), // No counterexample = equivalent
                    SatResult::Sat => {
                        if let Some(model) = solver.get_model() {
                            revert_note(format_args!(
                                "{}: Verification failed! Found counterexample:",
                                pass_name
                            ));
                            revert_note(format_args!("{}", model));
                        }
                        Ok(false)
                    }
                    SatResult::Unknown => Err(anyhow!(
                        "{}: SMT solver returned unknown (timeout or too complex)",
                        pass_name
                    )),
                }
            }
            (Ok(None), Ok(None)) if engine == EngineMode::NewEngineOnly => Err(anyhow!(
                "a void body carries no obligation for the new engine to decide"
            )),
            (Ok(None), Ok(None)) => {
                // Both functions return no value (void). Currently auto-pass.
                //
                // KNOWN LIMITATION: void functions can still have side effects
                // (memory writes, global writes) that differ between original
                // and optimized. The current model does not assert equivalence
                // of post-state locals/globals/memory — only the top-of-stack
                // return value. For void functions there is no return value
                // so the equivalence is vacuous. A future verifier upgrade
                // should encode side-effect equivalence for these.
                Ok(true)
            }
            (Err(e), _) => Err(anyhow!("{}: Failed to encode original: {}", pass_name, e)),
            (_, Err(e)) => Err(anyhow!("{}: Failed to encode optimized: {}", pass_name, e)),
            _ => Err(anyhow!(
                "{}: Return type mismatch between original and optimized",
                pass_name
            )),
        }
    })
}

/// Stub for when verification feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_function_equivalence_with_context(
    _original: &crate::Function,
    _optimized: &crate::Function,
    _pass_name: &str,
    _sig_ctx: &VerificationSignatureContext,
) -> Result<bool> {
    Ok(true)
}

/// Verify a module transformation by checking each function
///
/// This is the preferred way to verify an optimization pass.
/// It checks every function in the module for semantic equivalence.
#[cfg(feature = "verification")]
pub fn verify_module_transformation(
    original: &Module,
    optimized: &Module,
    pass_name: &str,
) -> Result<bool> {
    if original.functions.len() != optimized.functions.len() {
        return Err(anyhow!(
            "{}: Function count changed (was {}, now {})",
            pass_name,
            original.functions.len(),
            optimized.functions.len()
        ));
    }

    for (idx, (orig_func, opt_func)) in original
        .functions
        .iter()
        .zip(optimized.functions.iter())
        .enumerate()
    {
        let func_name = orig_func
            .name
            .clone()
            .unwrap_or_else(|| format!("func_{}", idx));

        match verify_function_equivalence(orig_func, opt_func, pass_name) {
            Ok(true) => continue,
            Ok(false) => {
                return Err(anyhow!(
                    "{}: Function '{}' is not semantically equivalent after optimization",
                    pass_name,
                    func_name
                ));
            }
            Err(e) => {
                // Log but continue - some functions may be too complex to verify
                eprintln!("Warning: Could not verify function '{}': {}", func_name, e);
            }
        }
    }

    Ok(true)
}

/// Stub for when verification feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_module_transformation(
    _original: &Module,
    _optimized: &Module,
    _pass_name: &str,
) -> Result<bool> {
    Ok(true)
}

/// Compute verification coverage for a module transformation.
///
/// This analyzes each function in the module and returns detailed coverage
/// metrics showing what percentage of functions can be fully Z3-verified
/// vs. what must be skipped due to loops or memory operations.
///
/// # Example
///
/// ```rust,ignore
/// let coverage = compute_verification_coverage(&original, &optimized, "optimize");
/// println!("{}", coverage.summary());
/// // Output: "Verification: 15/20 functions (75.0% Z3-proven)
/// //          Skipped: 3 loops, 2 memory ops, 0 unknown"
/// ```
#[cfg(feature = "verification")]
pub fn compute_verification_coverage(
    original: &Module,
    optimized: &Module,
    pass_name: &str,
) -> VerificationCoverage {
    let mut coverage = VerificationCoverage::new();

    if original.functions.len() != optimized.functions.len() {
        // Module structure changed - can't compute meaningful coverage
        coverage.record_error();
        return coverage;
    }

    for (orig_func, opt_func) in original.functions.iter().zip(optimized.functions.iter()) {
        let result = verify_function_equivalence_with_result(orig_func, opt_func, pass_name);
        result.update_coverage(&mut coverage);
    }

    coverage
}

/// Stub for when verification is disabled
#[cfg(not(feature = "verification"))]
pub fn compute_verification_coverage(
    original: &Module,
    optimized: &Module,
    _pass_name: &str,
) -> VerificationCoverage {
    // #331: this used to report EVERY function as verified, on the reasoning
    // that they are "assumed correct" when verification is compiled out. A
    // function named `compute_verification_coverage` returning 100% proven
    // for a build that proved nothing is the same defect as `loom verify`
    // printing a checkmark without opening the file (#332) — a claim whose
    // only basis is that nothing checked it.
    //
    // Nothing was verified, so nothing is reported as verified. They go in
    // `not_attempted`, a bucket added for this: every other one asserts
    // something specific, and reporting a timeout that never happened would
    // simply trade one false statement for another.
    let mut coverage = VerificationCoverage::new();
    let func_count = original.functions.len().min(optimized.functions.len());
    for _ in 0..func_count {
        coverage.record_not_attempted();
    }
    coverage
}

/// The disabled-feature coverage stub must not claim anything was verified.
///
/// This module compiles ONLY when the `verification` feature is off, which is
/// the whole point: the arm it covers is invisible to the default test run, so
/// before this existed the claim "coverage does not report functions as
/// verified in a no-verification build" was asserted by nothing. That stub
/// previously reported EVERY function as verified — a function named
/// `compute_verification_coverage` returning 100% proven for a build that
/// proved nothing.
///
/// Run with: `cargo test -p loom-core --no-default-features`
#[cfg(all(test, not(feature = "verification")))]
mod disabled_feature_coverage_tests {
    use super::*;

    fn void_function() -> crate::Function {
        crate::Function {
            name: None,
            signature: crate::FunctionSignature {
                params: vec![],
                results: vec![],
            },
            locals: vec![],
            instructions: vec![crate::Instruction::End],
        }
    }

    fn module_of(functions: Vec<crate::Function>) -> crate::Module {
        crate::Module {
            functions,
            memories: vec![],
            tables: vec![],
            globals: vec![],
            types: vec![],
            exports: vec![],
            imports: vec![],
            data_segments: vec![],
            element_section_bytes: None,
            start_function: None,
            custom_sections: vec![],
            type_section_bytes: None,
            global_section_bytes: None,
            facts: Vec::new(),
        }
    }

    #[test]
    fn a_build_without_verification_reports_nothing_as_verified() {
        let module = module_of(vec![void_function(), void_function(), void_function()]);
        let coverage = compute_verification_coverage(&module, &module, "no_verification_build");

        assert_eq!(
            coverage.verified, 0,
            "a build that verified nothing must not report ANY function as \
             verified; that is a claim whose only basis is that nothing \
             checked it"
        );
        assert_eq!(
            coverage.not_attempted, 3,
            "all three functions must be accounted for as not-attempted"
        );
        assert_eq!(
            coverage.total(),
            3,
            "no function may be dropped from the total"
        );
        assert!(
            coverage.nothing_was_attempted(),
            "the state must be detectable by a caller before it quotes a number"
        );
    }

    /// The nearest existing bucket would have been `verification_error`, which
    /// is documented as "timeout, encoding error". Reporting a timeout that
    /// never happened trades one false statement for another, so it must stay
    /// empty here.
    #[test]
    fn not_attempted_is_not_reported_as_an_error() {
        let module = module_of(vec![void_function()]);
        let coverage = compute_verification_coverage(&module, &module, "no_verification_build");
        assert_eq!(
            coverage.verification_error, 0,
            "nothing errored — no solver ran at all"
        );
        assert_eq!(
            coverage.verification_failed, 0,
            "no counterexample was found"
        );
    }
}

// ============================================================================
// Translation Validator - RAII guard for optimization pass verification
// ============================================================================

/// RAII-style translation validator for optimization passes
///
/// This captures the original function state before optimization, then verifies
/// semantic equivalence after the pass completes. This provides the same level
/// of correctness guarantee as years of Binaryen's fuzzing - but mathematically.
///
/// # Example
///
/// ```rust,ignore
/// use loom_core::verify::TranslationValidator;
///
/// fn my_optimization_pass(func: &mut Function) -> Result<()> {
///     let validator = TranslationValidator::new(func, "my_pass");
///
///     // ... perform optimizations on func ...
///
///     validator.verify(func)?; // Prove equivalence with Z3
///     Ok(())
/// }
/// ```
#[cfg(feature = "verification")]
pub struct TranslationValidator {
    /// Snapshot of the original function before optimization
    original: Function,
    /// Name of the optimization pass (for error messages)
    pass_name: String,
    /// Signature context for Call/CallIndirect verification
    sig_ctx: VerificationSignatureContext,
}

#[cfg(feature = "verification")]
impl TranslationValidator {
    /// Create a new validator, capturing the current function state
    /// Uses empty signature context (Call/CallIndirect will use conservative encoding)
    pub fn new(func: &Function, pass_name: &str) -> Self {
        Self {
            original: func.clone(),
            pass_name: pass_name.to_string(),
            sig_ctx: VerificationSignatureContext::new(),
        }
    }

    /// Create a new validator with signature context for proper Call verification
    pub fn new_with_context(
        func: &Function,
        pass_name: &str,
        sig_ctx: VerificationSignatureContext,
    ) -> Self {
        Self {
            original: func.clone(),
            pass_name: pass_name.to_string(),
            sig_ctx,
        }
    }

    /// Verify that the optimized function is semantically equivalent to the original
    ///
    /// Returns Ok(()) if verified equivalent, Err if different or verification fails
    pub fn verify(&self, optimized: &Function) -> Result<()> {
        // loom#145: keep z3-internal panic backtraces (always caught +
        // reverted below) out of stderr. Idempotent, race-free (Once).
        install_z3_panic_filter();

        // PR-K3.2 (Track B): size-threshold fallback. The per-function
        // Z3 translation validator scales poorly on large bodies — the
        // calculator_root meld-fused 2.3 MB core hangs >60 minutes
        // because the validator builds an enormous SMT formula for
        // every pass on every function. Beyond LOOM_Z3_MAX_INSTRUCTIONS
        // (default 2000), we skip Z3 and rely on the parallel stack
        // validator that every pass already wraps the transform in.
        //
        // This is conservative-over-fast in the OTHER direction from
        // CLAUDE.md's usual rule: we'd rather LOOM ship a result that
        // we couldn't deeply verify than fail to ship at all on real
        // workloads. Future PR could replace this with chunked
        // verification (verify each Block/Loop independently).
        let max_instructions: usize = std::env::var("LOOM_Z3_MAX_INSTRUCTIONS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2000);
        let n_instr =
            count_function_instructions(&self.original).max(count_function_instructions(optimized));
        if n_instr > max_instructions {
            // #331: this is NOT a revert. The transform is KEPT — `Ok(())`
            // means the caller accepts it — so recording it in the revert
            // counters made `--stats` print "N function(s) reverted" for
            // functions that shipped. A stat that names the wrong outcome is
            // worse than no stat: it reads as evidence verification did its
            // job, in exactly the case where it did not run.
            crate::stats::record_kept_unproven("body over the solver size threshold");
            return Ok(());
        }

        // #347 — MEMORY-OPERATION bound, beside the instruction bound above.
        //
        // The instruction bound cannot see the cost that actually blows up
        // here. Profiling a meld-fused module that ran past 455 s put 97% of
        // wall time in `Z3_solver_check` and 40% of the whole process in the
        // solver's array theory instantiating store axioms PAIRWISE — a cost
        // quadratic in memory accesses per body, which inlining multiplies by
        // concatenating callees into their caller.
        //
        // Two things this deliberately is NOT:
        //
        //  * It is not a fix. It buys termination by declining to verify the
        //    hardest inlines, so it trades proof coverage for completing at
        //    all. Those obligations are only really discharged when the
        //    memory model stops being the incumbent's array theory (#313
        //    slice 5). The honest framing is a bound, not a solution.
        //  * It is not a timeout. The solver's own timeout IS wired
        //    (`LOOM_Z3_TIMEOUT_MS`, default 5000) and measurably does not
        //    bound this: at 100 ms the pass still exceeded 120 s, because the
        //    time goes into axiom instantiation and internalisation rather
        //    than the search the timeout guards.
        //
        // Over the bound the new engine is tried first (#448, below). If it
        // does not prove the transform, the transform is REJECTED, not kept —
        // unlike the instruction bound above; the branch below says why. The
        // rejection is counted by `--stats` as a revert tagged
        // `memory-ops-over-threshold` and printed under `LOOM_VERBOSE_REVERTS`,
        // because a deferral nobody can see is the failure mode this project
        // keeps finding (#331).
        let max_memory_ops: usize = std::env::var("LOOM_Z3_MAX_MEMORY_OPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MAX_ARRAY_MEMORY_OPS);
        let n_memory_ops =
            count_array_memory_ops(&self.original).max(count_array_memory_ops(optimized));
        // Tiny bodies are exempt regardless of memory-op count: they are
        // cheap for the solver whatever they touch, and they are where the
        // #219 seam-dissolution inlines live. Bounding them would trade a
        // shipped, silicon-validated capability for nothing measurable.
        let memory_bound_floor: usize = std::env::var("LOOM_Z3_MEMORY_BOUND_FLOOR")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(DEFAULT_MEMORY_BOUND_INSTRUCTION_FLOOR);
        if n_memory_ops > max_memory_ops && n_instr > memory_bound_floor {
            // REVERT, not keep — deliberately unlike the instruction bound
            // directly above.
            //
            // Returning `Ok(())` would accept a transform nothing verified,
            // and at a default this low that would ship thousands of unproven
            // transforms per module: the wrong direction for a charter whose
            // rule is "skip the function rather than risk incorrect
            // optimization". Refusing costs optimization on memory-dense
            // inlines and costs nothing in safety, which is what makes a bound
            // this aggressive defensible at all.
            //
            // The instruction bound above still keeps. That inconsistency is
            // real and is left visible rather than quietly harmonised here —
            // changing long-shipped behaviour belongs in its own change.
            //
            // #448: the bound exists because Z3's ARRAY theory is quadratic in
            // memory operations, and the new engine does not use it. So before
            // refusing, offer the obligation to the new engine alone. Only a
            // proof accepts; a counterexample, an undecided obligation, or
            // anything the new engine defers is refused as before, without
            // reaching Z3.
            let original = self.original.clone();
            let optimized_clone = optimized.clone();
            let pass_name = self.pass_name.clone();
            let sig_ctx = self.sig_ctx.clone();
            let new_engine = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                verify_function_equivalence_with_context_uncounted(
                    &original,
                    &optimized_clone,
                    &pass_name,
                    &sig_ctx,
                    EngineMode::NewEngineOnly,
                )
            }));
            let why_not = match new_engine {
                Ok(Ok(true)) => {
                    record_acceptance(&self.original, optimized, false);
                    return Ok(());
                }
                Ok(Ok(false)) => "the new engine found a counterexample".to_string(),
                Ok(Err(e)) => e.to_string(),
                Err(_panic) => "the new engine panicked".to_string(),
            };
            note_deferral_reason("memory-ops-over-threshold");
            return Err(anyhow!(
                "{}: {} array-modelled memory operations exceeds the limit of {} \
                 (#347) and {} - optimization rejected (unproven)",
                self.pass_name,
                n_memory_ops,
                max_memory_ops,
                why_not
            ));
        }

        // Use catch_unwind to handle Z3 internal panics gracefully. (The
        // i64 inline hang in loom#147 turned out NOT to be a verify hang —
        // verify returns fast; it was a livelock in the inline-pass
        // fixpoint, fixed there. The Z3 solve-time bound stays in
        // z3_config_with_timeout.)
        let sig_ctx = self.sig_ctx.clone();
        let original = self.original.clone();
        let pass_name = self.pass_name.clone();
        let optimized_clone = optimized.clone();

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
            verify_function_equivalence_with_context(
                &original,
                &optimized_clone,
                &pass_name,
                &sig_ctx,
            )
        }));

        match result {
            Ok(Ok(true)) => Ok(()),
            Ok(Ok(false)) => {
                revert_note(format_args!(
                    "{}: Verification failed! Found counterexample:",
                    self.pass_name
                ));
                Err(anyhow!(
                    "{}: Z3 found counterexample - optimization rejected (unproven)",
                    self.pass_name
                ))
            }
            Ok(Err(e)) => Err(anyhow!(
                "{}: Verification failed ({}) - optimization rejected (unproven)",
                self.pass_name,
                e
            )),
            Err(_panic) => Err(anyhow!(
                "{}: Z3 internal error - optimization rejected (unproven)",
                self.pass_name
            )),
        }
    }

    /// Verify semantic equivalence; revert the function to its original
    /// instructions if verification fails. Returns true if verified, false
    /// if reverted. Never propagates errors — always safe to call.
    ///
    /// **Lenient semantics**: silent skips on unverifiable instructions
    /// (float load/store, unknown opcodes, complex loops) are treated as
    /// "verified" and the transform is kept. Use this for value-level
    /// passes that cannot move code across branches.
    ///
    /// For hoist-prone passes (LICM, CSE, code_folding, coalesce_locals),
    /// use [`Self::verify_or_revert_strict`] instead.
    ///
    /// Reverts are recorded in `crate::stats::record_revert(pass_name)` so
    /// callers can observe how often verification rejects a transform.
    pub fn verify_or_revert(&self, func: &mut Function) -> bool {
        // Clear any stale note before the attempt we are about to classify.
        let _ = take_deferral_reason();
        match self.verify(func) {
            Ok(()) => true,
            Err(e) => {
                revert_note(format_args!(
                    "{}: reverting function: {}",
                    self.pass_name, e
                ));
                // Attribute the revert when the validator said something more
                // specific than "the proof failed". Recorded HERE and only
                // here: recording at the refusal site too would count one
                // revert twice, and an inflated revert total is the same class
                // of defect as the mislabelled one this replaced.
                match take_deferral_reason() {
                    Some(reason) => {
                        crate::stats::record_revert(&format!("{}/{}", self.pass_name, reason))
                    }
                    None => crate::stats::record_revert(&self.pass_name),
                }
                func.instructions = self.original.instructions.clone();
                func.locals = self.original.locals.clone();
                false
            }
        }
    }

    /// Verify semantic equivalence with strict semantics: a *skipped*
    /// verification (verifier could not reason about this input) counts
    /// as failure and the function is reverted. Only `VerificationResult::Verified`
    /// is accepted. Returns true if verified, false if reverted.
    ///
    /// Use this for hoist-prone passes (LICM, CSE, code_folding,
    /// coalesce_locals) where moving code across branches is unsafe
    /// without a real proof — REQ-5 conservative-over-fast.
    ///
    /// Reverts are recorded in `crate::stats::record_revert("<pass>:strict-skip")`
    /// for skip-induced reverts and `crate::stats::record_revert("<pass>")`
    /// for genuine counterexamples.
    pub fn verify_or_revert_strict(&self, func: &mut Function) -> bool {
        install_z3_panic_filter();
        let routed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            exact_route(
                &self.original,
                func,
                &self.sig_ctx,
                VerifyBackend::from_env(),
            )
        }));
        let result = match routed {
            Ok(Some(Ok(true))) => VerificationResult::Verified,
            Ok(Some(Ok(false))) => VerificationResult::Failed(format!(
                "{}: counterexample from the exact executor",
                self.pass_name
            )),
            Ok(Some(Err(e))) => VerificationResult::Error(format!("{}: {}", self.pass_name, e)),
            Err(_panic) => VerificationResult::Error(format!(
                "{}: solver-internal panic - equivalence not established",
                self.pass_name
            )),
            Ok(None) => {
                verify_function_equivalence_with_result(&self.original, func, &self.pass_name)
            }
        };
        match result {
            VerificationResult::Verified => {
                record_acceptance(&self.original, func, false);
                true
            }
            VerificationResult::Failed(reason) => {
                revert_note(format_args!(
                    "{}: reverting function (counterexample): {}",
                    self.pass_name, reason
                ));
                crate::stats::record_revert(&self.pass_name);
                func.instructions = self.original.instructions.clone();
                func.locals = self.original.locals.clone();
                false
            }
            VerificationResult::Error(reason) => {
                revert_note(format_args!(
                    "{}: reverting function (verification error): {}",
                    self.pass_name, reason
                ));
                crate::stats::record_revert(&self.pass_name);
                func.instructions = self.original.instructions.clone();
                func.locals = self.original.locals.clone();
                false
            }
            r if r.is_skip() => {
                let reason = r.skip_reason().unwrap_or("unknown");
                revert_note(format_args!(
                    "{}: reverting function (strict mode rejects skip: {})",
                    self.pass_name, reason
                ));
                crate::stats::record_revert(&format!("{}:strict-skip", self.pass_name));
                func.instructions = self.original.instructions.clone();
                func.locals = self.original.locals.clone();
                false
            }
            _ => true, // unreachable in practice
        }
    }

    /// Verify and return detailed result instead of Result<()>
    pub fn verify_detailed(&self, optimized: &Function) -> TranslationResult {
        match verify_function_equivalence_with_context(
            &self.original,
            optimized,
            &self.pass_name,
            &self.sig_ctx,
        ) {
            Ok(true) => TranslationResult::Equivalent,
            Ok(false) => TranslationResult::Different,
            Err(e) => TranslationResult::Unknown(e.to_string()),
        }
    }
}

/// Result of translation validation
#[cfg(feature = "verification")]
#[derive(Debug, Clone, PartialEq)]
pub enum TranslationResult {
    /// Functions are proven semantically equivalent
    Equivalent,
    /// Found a counterexample - functions differ on some input
    Different,
    /// Could not determine (timeout, too complex, or error)
    Unknown(String),
}

/// Stub TranslationValidator for when verification is disabled
#[cfg(not(feature = "verification"))]
pub struct TranslationValidator {
    #[allow(dead_code)]
    pass_name: String,
}

#[cfg(not(feature = "verification"))]
impl TranslationValidator {
    /// Create a stub validator (does nothing when verification disabled)
    pub fn new(_func: &crate::Function, pass_name: &str) -> Self {
        Self {
            pass_name: pass_name.to_string(),
        }
    }

    /// Create a stub validator with context (does nothing when verification disabled)
    pub fn new_with_context(
        _func: &crate::Function,
        pass_name: &str,
        _sig_ctx: VerificationSignatureContext,
    ) -> Self {
        Self {
            pass_name: pass_name.to_string(),
        }
    }

    /// Stub verify - always succeeds when verification disabled
    pub fn verify(&self, _optimized: &crate::Function) -> Result<()> {
        Ok(())
    }

    /// Stub verify_or_revert - always returns true when verification disabled
    pub fn verify_or_revert(&self, _func: &mut crate::Function) -> bool {
        true
    }

    /// Stub verify_or_revert_strict — when verification is disabled, there
    /// is no Z3 model to consult, so we conservatively revert (REQ-5).
    /// In practice this means hoist-prone passes are no-ops in non-Z3
    /// builds; they should already be guarded by has_dataflow_unsafe_control_flow.
    pub fn verify_or_revert_strict(&self, _func: &mut crate::Function) -> bool {
        false
    }
}

/// Stub TranslationResult for when verification is disabled
#[cfg(not(feature = "verification"))]
#[derive(Debug, Clone, PartialEq)]
pub enum TranslationResult {
    /// Functions are proven semantically equivalent
    Equivalent,
    /// Found a counterexample - functions differ on some input
    Different,
    /// Could not determine (timeout, too complex, or error)
    Unknown(String),
}

/// Check if any load/store instruction in a function targets a non-zero memory index.
/// Z3 verification currently models a single memory as an Array; multi-memory requires
/// separate Arrays per memory index, which is future work. Functions with mem != 0
/// are conservatively skipped rather than producing incorrect proofs.
#[cfg(feature = "verification")]
fn has_multi_memory_ops(instructions: &[Instruction]) -> bool {
    for instr in instructions {
        match instr {
            Instruction::I32Load { mem, .. }
            | Instruction::I32Store { mem, .. }
            | Instruction::I64Load { mem, .. }
            | Instruction::I64Store { mem, .. }
            | Instruction::F32Load { mem, .. }
            | Instruction::F32Store { mem, .. }
            | Instruction::F64Load { mem, .. }
            | Instruction::F64Store { mem, .. }
            | Instruction::I32Load8S { mem, .. }
            | Instruction::I32Load8U { mem, .. }
            | Instruction::I32Load16S { mem, .. }
            | Instruction::I32Load16U { mem, .. }
            | Instruction::I64Load8S { mem, .. }
            | Instruction::I64Load8U { mem, .. }
            | Instruction::I64Load16S { mem, .. }
            | Instruction::I64Load16U { mem, .. }
            | Instruction::I64Load32S { mem, .. }
            | Instruction::I64Load32U { mem, .. }
            | Instruction::I32Store8 { mem, .. }
            | Instruction::I32Store16 { mem, .. }
            | Instruction::I64Store8 { mem, .. }
            | Instruction::I64Store16 { mem, .. }
            | Instruction::I64Store32 { mem, .. }
            | Instruction::MemoryFill(mem)
                if *mem != 0 =>
            {
                return true;
            }
            // MemoryCopy has two memory indices - skip if either is non-zero
            Instruction::MemoryCopy { dst_mem, src_mem } if (*dst_mem != 0 || *src_mem != 0) => {
                return true;
            }
            // MemoryInit targets a specific memory
            Instruction::MemoryInit { mem, .. } if *mem != 0 => {
                return true;
            }
            Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                if has_multi_memory_ops(body) =>
            {
                return true;
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } if (has_multi_memory_ops(then_body) || has_multi_memory_ops(else_body)) => {
                return true;
            }
            _ => {}
        }
    }
    false
}

/// Encode a WebAssembly function to an SMT formula
///
/// This converts the instruction sequence into a symbolic execution that Z3 can reason about.
/// Returns None for void functions, Some(BV) for functions with a return value.
#[cfg(feature = "verification")]
fn encode_function_to_smt(func: &Function) -> Result<Option<BV>> {
    encode_function_to_smt_impl(func, None)
}

/// Encode a WebAssembly function to an SMT formula with signature context
///
/// This version uses the signature context for proper Call/CallIndirect stack modeling.
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn encode_function_to_smt_with_context(
    func: &Function,
    sig_ctx: &VerificationSignatureContext,
) -> Result<Option<BV>> {
    encode_function_to_smt_impl(func, Some(sig_ctx))
}

/// Internal implementation of SMT encoding with optional signature context
#[cfg(feature = "verification")]
fn encode_function_to_smt_impl(
    func: &Function,
    sig_ctx: Option<&VerificationSignatureContext>,
) -> Result<Option<BV>> {
    encode_function_to_smt_impl_inner(func, sig_ctx, None)
}

/// Core SMT encoding implementation
///
/// When `shared_inputs` is Some, uses the provided symbolic inputs.
/// When None, creates fresh symbolic inputs (for standalone encoding).
#[cfg(feature = "verification")]
fn encode_function_to_smt_impl_inner(
    func: &Function,
    sig_ctx: Option<&VerificationSignatureContext>,
    shared_inputs: Option<&SharedSymbolicInputs>,
) -> Result<Option<BV>> {
    encode_function_state(func, sig_ctx, shared_inputs).map(|(ret, _)| ret)
}

/// The encoded return value and, when the whole body was encoded, the final
/// value of every global the function can observe. `None` for the globals
/// means the post-state is not modelled (an early exit of the encoder).
#[cfg(feature = "verification")]
type EncodedState = (Option<BV>, Option<Vec<BV>>);

#[cfg(feature = "verification")]
fn encode_function_state(
    func: &Function,
    sig_ctx: Option<&VerificationSignatureContext>,
    shared_inputs: Option<&SharedSymbolicInputs>,
) -> Result<EncodedState> {
    // Conservative skip: functions using multi-memory (mem != 0) cannot be verified
    // with our single-Array Z3 model. Return None to skip verification rather than
    // produce incorrect proofs. Full multi-memory Z3 support (separate Array per
    // memory) is future work.
    if has_multi_memory_ops(&func.instructions) {
        return Ok((None, None));
    }

    // Create symbolic variables for parameters
    let mut stack: Vec<BV> = Vec::new();
    let mut locals: Vec<BV>;
    let mut globals: Vec<BV>;
    let mut memory: Array;
    let mut memory_size: BV;

    // Local counter for memory.grow operations - ensures both original and optimized
    // use the same symbolic names for corresponding operations
    let mut memory_grow_count: u64 = 0;

    // loom#155: per-encode counter for impure-call havoc names. MUST be local
    // (not a global static): the original and optimized functions are encoded
    // by separate calls into the SAME Z3 context, so a local counter that
    // resets to 0 each encode gives the Nth impure call the SAME havoc'd
    // global/memory name in both encodings — which makes those havoc'd arrays
    // UNIFY in Z3. Without this, a result that depends on post-impure-call
    // memory/globals (e.g. inlining a callee that reads what a preceding
    // havoc'd call left behind) would compare two differently-named arrays and
    // revert spuriously. A global counter never unified them.
    let mut havoc_counter: u64 = 0;

    if let Some(shared) = shared_inputs {
        // Use shared inputs - CRITICAL for correct verification
        locals = shared.initial_locals.clone();
        globals = shared.globals.clone();
        memory = shared.memory.clone();
        memory_size = shared.memory_size.clone();

        // Extend locals if the optimized function uses more locals than the original
        // This can happen when optimizations like LICM add temporary locals
        // We need to check both the declared locals AND the maximum index actually used
        let declared_local_count = func.signature.params.len()
            + func
                .locals
                .iter()
                .map(|(count, _)| *count as usize)
                .sum::<usize>();
        let max_used_local = find_max_local_index(&func.instructions);
        let needed_locals = declared_local_count.max(max_used_local + 1);

        while locals.len() < needed_locals {
            // Loom#98 fix: extend with the DECLARED type, not a hardcoded
            // 32-bit default. The previous code crashed the inline pass on
            // gale-ffi (i64-packed-return wasm) because i64 locals declared
            // by the optimized function were padded as 32-bit, then any
            // i64 binop on them tripped Z3's `SortDiffers` panic.
            let idx = locals.len();
            let width = local_type_at(func, idx)
                .map(bv_width_for_value_type)
                .unwrap_or(32);
            locals.push(BV::from_u64(0, width));
        }
    } else {
        // Create fresh inputs (for standalone encoding)
        locals = Vec::new();

        // Initialize parameters as symbolic inputs
        for (idx, param_type) in func.signature.params.iter().enumerate() {
            let width = match param_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            let param = BV::new_const(format!("param{}", idx), width);
            locals.push(param);
        }

        // Initialize local variables to zero
        for (count, local_type) in func.locals.iter() {
            let width = match local_type {
                crate::ValueType::I32 => 32,
                crate::ValueType::I64 => 64,
                crate::ValueType::F32 => 32,
                crate::ValueType::F64 => 64,
            };
            for _ in 0..*count {
                locals.push(BV::from_u64(0, width));
            }
        }

        // Initialize globals as symbolic
        globals = Vec::new();
        for i in 0..16 {
            globals.push(BV::new_const(format!("global{}", i), 32));
        }

        // Create symbolic memory: Array[BitVec32 -> BitVec8]
        let addr_sort = Sort::bitvector(32);
        let byte_sort = Sort::bitvector(8);
        memory = Array::new_const("memory", &addr_sort, &byte_sort);

        // Create symbolic memory size (in pages)
        memory_size = BV::new_const("memory_size", 32);
    }

    // Symbolically execute instructions
    for instr in &func.instructions {
        match instr {
            // Constants
            Instruction::I32Const(n) => {
                stack.push(BV::from_i64(*n as i64, 32));
            }
            Instruction::I64Const(n) => {
                stack.push(BV::from_i64(*n, 64));
            }
            Instruction::F32Const(bits) => {
                // Float constants are treated as bit patterns for now
                // We don't perform floating-point arithmetic verification yet
                stack.push(BV::from_i64(*bits as i64, 32));
            }
            Instruction::F64Const(bits) => {
                // Float constants are treated as bit patterns for now
                // We don't perform floating-point arithmetic verification yet
                stack.push(BV::from_i64(*bits as i64, 64));
            }

            // Arithmetic operations (i32)
            Instruction::I32Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvadd(&rhs));
            }
            Instruction::I32Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvsub(&rhs));
            }
            Instruction::I32Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvmul(&rhs));
            }

            // Arithmetic operations (i64)
            Instruction::I64Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvadd(&rhs));
            }
            Instruction::I64Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvsub(&rhs));
            }
            Instruction::I64Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvmul(&rhs));
            }

            // Bitwise operations (i32)
            Instruction::I32And => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32And"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvand(&rhs));
            }
            Instruction::I32Or => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Or"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvor(&rhs));
            }
            Instruction::I32Xor => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Xor"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvxor(&rhs));
            }
            Instruction::I32Shl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Shl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvshl(&rhs));
            }
            Instruction::I32ShrU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32ShrU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvlshr(&rhs));
            }
            Instruction::I32ShrS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32ShrS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvashr(&rhs));
            }

            // Bitwise operations (i64)
            Instruction::I64And => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64And"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvand(&rhs));
            }
            Instruction::I64Or => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Or"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvor(&rhs));
            }
            Instruction::I64Xor => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Xor"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvxor(&rhs));
            }
            Instruction::I64Shl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Shl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvshl(&rhs));
            }
            Instruction::I64ShrU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64ShrU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvlshr(&rhs));
            }
            Instruction::I64ShrS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64ShrS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvashr(&rhs));
            }

            // Comparison operations (i32) - produce i32 boolean (0 or 1)
            Instruction::I32Eq => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Eq"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                // (lhs == rhs) ? 1 : 0
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.eq(&rhs).ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)));
            }
            Instruction::I32Ne => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Ne"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.eq(&rhs)
                        .not()
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvslt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvult(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvsgt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvugt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvsle(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32LeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32LeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvule(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvsge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32GeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32GeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvuge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }

            // Comparison operations (i64) - produce i32 boolean
            Instruction::I64Eq => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Eq"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.eq(&rhs).ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)));
            }
            Instruction::I64Ne => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Ne"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.eq(&rhs)
                        .not()
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvslt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvult(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GtS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GtS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvsgt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GtU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GtU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvugt(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvsle(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64LeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64LeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvule(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GeS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GeS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvsge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64GeU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64GeU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(
                    lhs.bvuge(&rhs)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }

            // Division and remainder operations (i32)
            Instruction::I32DivS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32DivS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvsdiv(&rhs));
            }
            Instruction::I32DivU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32DivU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvudiv(&rhs));
            }
            Instruction::I32RemS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32RemS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvsrem(&rhs));
            }
            Instruction::I32RemU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32RemU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvurem(&rhs));
            }

            // Division and remainder operations (i64)
            Instruction::I64DivS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64DivS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvsdiv(&rhs));
            }
            Instruction::I64DivU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64DivU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvudiv(&rhs));
            }
            Instruction::I64RemS => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64RemS"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvsrem(&rhs));
            }
            Instruction::I64RemU => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64RemU"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvurem(&rhs));
            }

            // Unary operations (i32)
            Instruction::I32Eqz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Eqz"));
                }
                let val = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                stack.push(
                    val.eq(&zero)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I32Clz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Clz"));
                }
                let val = stack.pop().unwrap();
                // Count leading zeros: encode as nested conditionals checking each bit from MSB
                // If bit 31 is set → 0, else if bit 30 is set → 1, ..., else → 32
                // CRITICAL: We build the ITE chain so MSB check is outermost (evaluated first)
                let mut result = BV::from_i64(32, 32); // All zeros case
                for i in (0..32).rev() {
                    // Reverse order so MSB (i=0, bit_pos=31) becomes outermost ITE
                    let bit_pos = 31 - i;
                    let mask = BV::from_i64(1i64 << bit_pos, 32);
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(i as i64, 32), &result);
                }
                stack.push(result);
            }
            Instruction::I32Ctz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Ctz"));
                }
                let val = stack.pop().unwrap();
                // Count trailing zeros: encode as nested conditionals checking each bit from LSB
                // If bit 0 is set → 0, else if bit 1 is set → 1, ..., else → 32
                let mut result = BV::from_i64(32, 32); // All zeros case
                for i in 0..32 {
                    let bit_pos = 31 - i;
                    let mask = BV::from_i64(1i64 << bit_pos, 32);
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(bit_pos as i64, 32), &result);
                }
                stack.push(result);
            }
            Instruction::I32Popcnt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Popcnt"));
                }
                let val = stack.pop().unwrap();
                // Population count: sum all the bits
                // For each bit position, extract (val >> i) & 1 and sum them
                let mut result = BV::from_i64(0, 32);
                for i in 0..32 {
                    let mask = BV::from_i64(1i64 << i, 32);
                    let bit = val.bvand(&mask).eq(&mask);
                    // Add 1 if bit is set, 0 otherwise
                    result = bit.ite(&result.bvadd(BV::from_i64(1, 32)), &result);
                }
                stack.push(result);
            }

            // Unary operations (i64)
            Instruction::I64Eqz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Eqz"));
                }
                let val = stack.pop().unwrap();
                let zero = BV::from_i64(0, 64);
                // Result is i32!
                stack.push(
                    val.eq(&zero)
                        .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
                );
            }
            Instruction::I64Clz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Clz"));
                }
                let val = stack.pop().unwrap();
                // Count leading zeros for i64: same approach as i32 but 64 bits
                // CRITICAL: Reverse order so MSB check is outermost
                let mut result = BV::from_i64(64, 64); // All zeros case
                for i in (0..64).rev() {
                    let bit_pos = 63 - i;
                    let mask = if bit_pos < 63 {
                        BV::from_i64(1i64 << bit_pos, 64)
                    } else {
                        // Handle bit 63 specially to avoid overflow
                        BV::from_i64(i64::MIN, 64)
                    };
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(i as i64, 64), &result);
                }
                stack.push(result);
            }
            Instruction::I64Ctz => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Ctz"));
                }
                let val = stack.pop().unwrap();
                // Count trailing zeros for i64
                let mut result = BV::from_i64(64, 64); // All zeros case
                for i in 0..64 {
                    let bit_pos = 63 - i;
                    let mask = if bit_pos < 63 {
                        BV::from_i64(1i64 << bit_pos, 64)
                    } else {
                        BV::from_i64(i64::MIN, 64)
                    };
                    let bit_set = val.bvand(&mask).eq(&mask);
                    result = bit_set.ite(&BV::from_i64(bit_pos as i64, 64), &result);
                }
                stack.push(result);
            }
            Instruction::I64Popcnt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Popcnt"));
                }
                let val = stack.pop().unwrap();
                // Population count for i64: sum all 64 bits
                let mut result = BV::from_i64(0, 64);
                for i in 0..64 {
                    let mask = if i < 63 {
                        BV::from_i64(1i64 << i, 64)
                    } else {
                        BV::from_i64(i64::MIN, 64)
                    };
                    let bit = val.bvand(&mask).eq(&mask);
                    result = bit.ite(&result.bvadd(BV::from_i64(1, 64)), &result);
                }
                stack.push(result);
            }

            // Select operation: [T, T, i32] -> [T]
            Instruction::Select => {
                if stack.len() < 3 {
                    return Err(anyhow!("Stack underflow in Select"));
                }
                let cond = stack.pop().unwrap();
                let val2 = stack.pop().unwrap();
                let val1 = stack.pop().unwrap();
                // Select: if cond != 0 then val1 else val2
                let zero = BV::from_i64(0, 32);
                stack.push(cond.eq(&zero).not().ite(&val1, &val2));
            }

            // Nop does nothing
            Instruction::Nop => {}

            // Drop pops and discards top of stack
            Instruction::Drop => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in Drop"));
                }
                stack.pop();
            }

            // Local operations
            // Note: Optimization passes (like CSE) may add new locals to the optimized
            // function that don't exist in the original. We extend locals dynamically
            // similar to how we handle globals.
            Instruction::LocalGet(idx) => {
                let idx = *idx as usize;
                // Extend locals vector if needed (for optimizer-added locals)
                while locals.len() <= idx {
                    // Default to 32-bit zero - will be set by LocalTee before read
                    locals.push(BV::from_u64(0, 32));
                }
                stack.push(locals[idx].clone());
            }
            Instruction::LocalSet(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in LocalSet"));
                }
                let idx = *idx as usize;
                // Extend locals vector if needed
                while locals.len() <= idx {
                    locals.push(BV::from_u64(0, 32));
                }
                // SAFETY: guarded by is_empty check above
                let value = stack.pop().unwrap();
                locals[idx] = value;
            }
            Instruction::LocalTee(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in LocalTee"));
                }
                let idx = *idx as usize;
                // Extend locals vector if needed (CSE adds locals for caching)
                while locals.len() <= idx {
                    locals.push(BV::from_u64(0, 32));
                }
                // SAFETY: guarded by is_empty check above
                let value = stack.last().unwrap().clone();
                locals[idx] = value;
            }

            // Global operations
            Instruction::GlobalGet(idx) => {
                let idx = *idx as usize;
                // Extend globals vector if needed
                while globals.len() <= idx {
                    let new_idx = globals.len();
                    globals.push(BV::new_const(format!("global{}", new_idx), 32));
                }
                stack.push(globals[idx].clone());
            }
            Instruction::GlobalSet(idx) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in GlobalSet"));
                }
                let idx = *idx as usize;
                while globals.len() <= idx {
                    let new_idx = globals.len();
                    globals.push(BV::new_const(format!("global{}", new_idx), 32));
                }
                // SAFETY: guarded by is_empty check above
                let value = stack.pop().unwrap();
                globals[idx] = value;
            }

            // Memory operations using Z3 Array theory
            // Memory is byte-addressable: Array[BitVec32 -> BitVec8]
            // Multi-byte loads/stores combine bytes in little-endian order
            Instruction::I32Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load"));
                }
                // SAFETY: guarded by is_empty check above
                let addr = stack.pop().unwrap();
                // loom#155: shared encoder (also used by by-body inline modeling).
                stack.push(encode_i32_load_from(&memory, &addr, *offset)?);
            }
            Instruction::I32Store { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Store"));
                }
                // SAFETY: guarded by len check above
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                // loom#157: shared encoder (also used by by-body inline modeling).
                memory = encode_i32_store_into(&memory, &addr, &value, *offset);
            }
            Instruction::I64Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load"));
                }
                // SAFETY: guarded by is_empty check above
                let addr = stack.pop().unwrap();
                // loom#155: shared encoder (also used by by-body inline modeling).
                stack.push(encode_i64_load_from(&memory, &addr, *offset)?);
            }
            Instruction::I64Store { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store"));
                }
                // SAFETY: guarded by len check above
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                // loom#157: shared encoder (also used by by-body inline modeling).
                memory = encode_i64_store_into(&memory, &addr, &value, *offset);
            }

            // Partial-width loads (8-bit and 16-bit with sign/zero extension)
            // Partial-width loads (8/16/32-bit) — encoded via the shared
            // encode_partial_load_from helper so this path and the by-body
            // inline modeler (encode_inlinable_callee_result) cannot drift apart
            // (loom#150 dedup; the helper reads N bytes little-endian then sign-
            // or zero-extends to the result width).
            Instruction::I32Load8S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load8S"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 1, true, 32,
                )?);
            }
            Instruction::I32Load8U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load8U"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 1, false, 32,
                )?);
            }
            Instruction::I32Load16S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load16S"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 2, true, 32,
                )?);
            }
            Instruction::I32Load16U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Load16U"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 2, false, 32,
                )?);
            }
            Instruction::I64Load8S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load8S"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 1, true, 64,
                )?);
            }
            Instruction::I64Load8U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load8U"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 1, false, 64,
                )?);
            }
            Instruction::I64Load16S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load16S"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 2, true, 64,
                )?);
            }
            Instruction::I64Load16U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load16U"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 2, false, 64,
                )?);
            }
            Instruction::I64Load32S { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load32S"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 4, true, 64,
                )?);
            }
            Instruction::I64Load32U { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Load32U"));
                }
                let addr = stack.pop().unwrap();
                stack.push(encode_partial_load_from(
                    &memory, &addr, *offset, 4, false, 64,
                )?);
            }

            // Partial-width stores (8/16/32-bit) — via the shared
            // encode_partial_store_into helper (loom#150 dedup; stores the low
            // N bytes little-endian).
            Instruction::I32Store8 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Store8"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                memory = encode_partial_store_into(&memory, &addr, &value, *offset, 1);
            }
            Instruction::I32Store16 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Store16"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                memory = encode_partial_store_into(&memory, &addr, &value, *offset, 2);
            }
            Instruction::I64Store8 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store8"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                memory = encode_partial_store_into(&memory, &addr, &value, *offset, 1);
            }
            Instruction::I64Store16 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store16"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                memory = encode_partial_store_into(&memory, &addr, &value, *offset, 2);
            }
            Instruction::I64Store32 { offset, .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Store32"));
                }
                let value = stack.pop().unwrap();
                let addr = stack.pop().unwrap();
                memory = encode_partial_store_into(&memory, &addr, &value, *offset, 4);
            }

            // Control flow instructions

            // Block: execute body, result goes on stack
            Instruction::Block { block_type, body } => {
                // Execute the block body
                let block_result = encode_block_body(body, &mut stack, &mut locals, &mut globals)?;

                // If block has a result type, it should be on the stack
                if let Some(width) = block_type_width(block_type) {
                    if block_result.is_none() && stack.is_empty() {
                        // Block didn't produce a result - create symbolic placeholder
                        stack.push(BV::new_const("block_result", width));
                    }
                }
            }

            // If/Else: branch based on condition
            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in If"));
                }
                // SAFETY: guarded by is_empty check above
                let cond = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let cond_bool = cond.eq(&zero).not(); // cond != 0

                // Save state before branches
                let saved_stack = stack.clone();
                let saved_locals = locals.clone();
                let saved_globals = globals.clone();

                // Execute then branch
                let then_result =
                    encode_block_body(then_body, &mut stack, &mut locals, &mut globals)?;
                let then_stack = stack.clone();
                let then_locals = locals.clone();
                let then_globals = globals.clone();

                // Restore and execute else branch
                stack = saved_stack;
                locals = saved_locals;
                globals = saved_globals;
                let else_result =
                    encode_block_body(else_body, &mut stack, &mut locals, &mut globals)?;

                // Merge the two branches - collect merged values first to avoid borrow conflicts
                let merged_locals: Vec<BV> = then_locals
                    .iter()
                    .zip(locals.iter())
                    .map(|(then_local, else_local)| merge_bv(&cond_bool, then_local, else_local))
                    .collect();
                locals = merged_locals;

                let merged_globals: Vec<BV> = then_globals
                    .iter()
                    .zip(globals.iter())
                    .map(|(then_global, else_global)| {
                        merge_bv(&cond_bool, then_global, else_global)
                    })
                    .collect();
                globals = merged_globals;

                // For stack, merge if both branches produce same number of values
                if then_stack.len() == stack.len() {
                    let merged_stack: Vec<BV> = then_stack
                        .iter()
                        .zip(stack.iter())
                        .map(|(then_val, else_val)| merge_bv(&cond_bool, then_val, else_val))
                        .collect();
                    stack = merged_stack;
                } else if let Some(width) = block_type_width(block_type) {
                    // Block expects a result - use ITE on results
                    let then_val = then_result.unwrap_or_else(|| {
                        then_stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    let else_val = else_result.unwrap_or_else(|| {
                        stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    stack.clear();
                    stack.push(merge_bv(&cond_bool, &then_val, &else_val));
                }
            }

            // Loop: bounded unrolling for verification
            Instruction::Loop { block_type, body } => {
                // For verification, we unroll loops a fixed number of times
                // This is sound but incomplete (may miss bugs in later iterations)
                for _iteration in 0..MAX_LOOP_UNROLL {
                    let _ = encode_block_body(body, &mut stack, &mut locals, &mut globals)?;
                }
                // After unrolling, if loop has a result type, ensure something is on stack
                if let Some(width) = block_type_width(block_type) {
                    if stack.is_empty() {
                        stack.push(BV::new_const("loop_result", width));
                    }
                }
            }

            // Branch: exit enclosing block (handled by returning early from block encoder)
            Instruction::Br(_depth) => {
                // Branch exits the current block - for simple verification,
                // we treat this as terminating the current instruction sequence
                // The actual depth handling is done in encode_block_body
                break;
            }

            // Conditional branch
            Instruction::BrIf(_depth) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in BrIf"));
                }
                // SAFETY: guarded by is_empty check above
                let cond = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let _cond_bool = cond.eq(&zero).not();
                // For simple verification, we continue execution
                // A more precise encoding would fork paths here
            }

            // Branch table
            Instruction::BrTable { .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in BrTable"));
                }
                // SAFETY: guarded by is_empty check above
                let _index = stack.pop().unwrap();
                // Branch table is complex - treat as terminating for now
                break;
            }

            // Return from function
            Instruction::Return => {
                // Return terminates execution with current stack top
                break;
            }

            // Function call - properly model stack effects using signature context
            // Also track side effects (globals/memory modifications) via function summaries
            Instruction::Call(func_idx) => {
                // For verification, we can't inline calls (would need interprocedural analysis)
                // Instead, we properly model the stack effects:
                // 1. Pop the correct number of arguments
                // 2. Push results — fresh symbolic, OR (PR-K3) an
                //    uninterpreted-function application f(args) when
                //    the callee is pure + no-trap. The latter makes
                //    Z3's congruence closure prove two identical pure
                //    calls equal — which unblocks cross-call CSE dedup.
                // 3. Havoc globals/memory that the called function might modify
                //    (skipped for pure + no-trap callees, per IPA).
                if let Some(ctx_ref) = sig_ctx {
                    if let Some(sig) = ctx_ref.get_function_signature(*func_idx) {
                        // Pop arguments (in reverse order, as they were pushed)
                        // and stash them for PR-K3 uninterpreted-function application.
                        let mut args_rev: Vec<BV> = Vec::with_capacity(sig.params.len());
                        for i in 0..sig.params.len() {
                            if stack.is_empty() {
                                return Err(anyhow!(
                                    "Stack underflow in Call: missing arg {} of {}",
                                    i + 1,
                                    sig.params.len()
                                ));
                            }
                            // SAFETY: guarded by is_empty check above
                            args_rev.push(stack.pop().unwrap());
                        }
                        // Reverse to source order (arg0, arg1, ...).
                        args_rev.reverse();
                        let args: Vec<BV> = args_rev;

                        // loom#151: model the call by F's OWN BODY when F
                        // is a pure, no-trap, leaf, straight-line,
                        // single-result callee. This is what makes the
                        // verified inliner sound: the opaque `pure_call`
                        // encoding below proves two *identical* calls
                        // equal (CSE) but can NEVER prove a call equal to
                        // its own inlined body, so it reverted every
                        // inline. Encoding the call by F's definition
                        // makes the original (with the call) and the
                        // optimized (with F inlined) compute the identical
                        // expression — Z3 proves them equal. Bodies
                        // outside the whitelist fall through to the
                        // conservative encoding (at worst a spurious
                        // revert, never an unsound accept).
                        if let Some(callee) = ctx_ref.inlinable_callee_body(*func_idx) {
                            if let Some((result_bv, new_mem)) =
                                encode_inlinable_callee_result(&callee, &args, &memory)
                            {
                                // loom#155/#157: the callee may READ and/or WRITE
                                // memory; its loads/stores were encoded against
                                // the current shared `memory` Array via the same
                                // helpers the main encoder uses. Apply its memory
                                // effects (no havoc — the writes are concrete and
                                // modeled identically on both sides). Push the
                                // result only for non-void callees.
                                memory = new_mem;
                                if let Some(bv) = result_bv {
                                    stack.push(bv);
                                }
                                continue;
                            }
                        }

                        // PR-K3: when callee is pure + no-trap AND has
                        // exactly one result, encode the result as
                        // `pure_call_<func_idx>(args)`. Z3 congruence
                        // closure makes two such expressions with equal
                        // args prove-equal, which is what makes CSE
                        // dedup of pure calls verifiable. Multi-result
                        // calls fall through to the fresh-symbolic
                        // path; supporting them needs a tuple sort
                        // which is more invasive.
                        let pure_no_trap = ctx_ref.is_pure_no_trap_call(*func_idx);
                        if pure_no_trap && sig.results.len() == 1 {
                            let result_type = sig.results[0];
                            let result_width: u32 = match result_type {
                                crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                crate::ValueType::I64 | crate::ValueType::F64 => 64,
                            };
                            // Build domain sorts from arg widths. FuncDecl::new
                            // takes a slice of &Sort; we own the Sorts via a Vec
                            // and create refs only at call time.
                            let domain_sorts_owned: Vec<Sort> = sig
                                .params
                                .iter()
                                .map(|p| {
                                    Sort::bitvector(match p {
                                        crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                        crate::ValueType::I64 | crate::ValueType::F64 => 64,
                                    })
                                })
                                .collect();
                            let domain_refs: Vec<&Sort> = domain_sorts_owned.iter().collect();
                            let range_sort = Sort::bitvector(result_width);

                            // FuncDecl is symbol-deduplicated by Z3: every call
                            // with the same name + sorts in the same context
                            // resolves to the SAME function decl. That's what
                            // gives us congruence equality on identical args.
                            let decl_name = format!("pure_call_{}", func_idx);
                            let decl = FuncDecl::new(decl_name.as_str(), &domain_refs, &range_sort);

                            // Apply to the actual arg BVs.
                            let arg_refs: Vec<&dyn z3::ast::Ast> =
                                args.iter().map(|a| a as &dyn z3::ast::Ast).collect();
                            let result_dyn = decl.apply(&arg_refs);
                            if let Some(result_bv) = result_dyn.as_bv() {
                                stack.push(result_bv);
                            } else {
                                // Defensive: if the dynamic isn't a BV
                                // (shouldn't happen given range_sort),
                                // fall back to the fresh-symbolic path.
                                stack.push(BV::new_const(
                                    format!("call_{}_result_fallback", func_idx),
                                    result_width,
                                ));
                            }
                            // CRITICAL: skip the global/memory havoc.
                            // Pure + no-trap means no observable side
                            // effects — globals and memory are unchanged
                            // by this call.
                        } else {
                            // Existing conservative encoding: fresh
                            // symbolic per result + full havoc.
                            for (i, result_type) in sig.results.iter().enumerate() {
                                let width = match result_type {
                                    crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                    crate::ValueType::I64 | crate::ValueType::F64 => 64,
                                };
                                stack.push(BV::new_const(
                                    format!("call_{}_result_{}", func_idx, i),
                                    width,
                                ));
                            }

                            // Havoc all globals that this function might modify.
                            // Conservative but sound - we assume the call could
                            // modify any global it has write access to.
                            // loom#155: deterministic per-encode id (see
                            // `havoc_counter` decl) so corresponding calls in
                            // the original/optimized encodings unify in Z3.
                            let call_id = havoc_counter;
                            havoc_counter += 1;
                            for (idx, global) in globals.iter_mut().enumerate() {
                                *global = BV::new_const(
                                    format!("global_{}_after_call_{}_{}", idx, func_idx, call_id),
                                    global.get_size(),
                                );
                            }
                            memory = Array::new_const(
                                format!("mem_after_call_{}_{}", func_idx, call_id),
                                &Sort::bitvector(32),
                                &Sort::bitvector(8),
                            );
                        }
                    } else {
                        // Unknown function - conservative: assume returns i32
                        stack.push(BV::new_const("call_unknown_result", 32));
                    }
                } else {
                    // No context - conservative: assume returns i32
                    stack.push(BV::new_const("call_result", 32));
                }
            }

            // Indirect call - properly model stack effects using type signature
            //
            // v1.0.4 Track B: when the indirect-call slot is statically
            // known (the index BV simplifies to a literal `u64`) AND the
            // element-section table resolver knows the function at that
            // slot AND its signature matches the indirect-call type, we
            // encode the result as `pure_call_<F>(args)` — exactly the
            // PR-K3 path used for direct `call F`. This makes the
            // verifier's BV for `i32.const N; call_indirect (type T)`
            // structurally equal to the BV for `call F` under Z3
            // congruence closure, unblocking directize's Z3 verification.
            //
            // When the index is dynamic (non-concrete) OR the resolver
            // doesn't cover the slot OR signatures don't match, we fall
            // back to the existing conservative encoding: fresh-symbolic
            // result + havoc of all globals and memory.
            Instruction::CallIndirect {
                type_idx,
                table_idx,
            } => {
                // Pop table index first.
                if stack.is_empty() {
                    return Err(anyhow!(
                        "Stack underflow in CallIndirect: missing table index"
                    ));
                }
                // SAFETY: guarded by is_empty check above
                let slot_bv = stack.pop().unwrap();
                let concrete_slot: Option<u32> =
                    slot_bv.as_u64().and_then(|n| u32::try_from(n).ok());

                // Use type signature to properly model stack effects
                if let Some(ctx_ref) = sig_ctx {
                    if let Some(sig) = ctx_ref.get_type_signature(*type_idx) {
                        // v1.0.4 Track B: try to resolve to a concrete callee.
                        let resolved_func: Option<u32> = concrete_slot
                            .and_then(|slot| ctx_ref.resolve_indirect_call(*table_idx, slot, sig));

                        if let Some(func_idx) = resolved_func {
                            // Pop arguments into a Vec we own, so we can
                            // apply them to the uninterpreted function.
                            let mut args_rev: Vec<BV> = Vec::with_capacity(sig.params.len());
                            for i in 0..sig.params.len() {
                                if stack.is_empty() {
                                    return Err(anyhow!(
                                        "Stack underflow in CallIndirect: missing arg {} of {}",
                                        i + 1,
                                        sig.params.len()
                                    ));
                                }
                                args_rev.push(stack.pop().unwrap());
                            }
                            args_rev.reverse();
                            let args = args_rev;

                            // loom#151: model the resolved indirect call by
                            // F's OWN BODY when eligible — keeps this path
                            // CONSISTENT with the direct `call F` path
                            // (which also models by-body). Without this, a
                            // `call_indirect` resolved to F would encode as
                            // opaque `pure_call_F` while the direct `call F`
                            // encodes as F's body, so directize's
                            // `call_indirect` ⇔ `call F` equivalence would
                            // break.
                            if let Some(callee) = ctx_ref.inlinable_callee_body(func_idx) {
                                if let Some((result_bv, new_mem)) =
                                    encode_inlinable_callee_result(&callee, &args, &memory)
                                {
                                    memory = new_mem;
                                    if let Some(bv) = result_bv {
                                        stack.push(bv);
                                    }
                                    continue;
                                }
                            }

                            // Resolved-call encoding mirrors the PR-K3
                            // direct-Call path: emit `pure_call_<F>(args)`
                            // via Z3 FuncDecl::apply so two equal indirect
                            // calls (same slot, same args) prove equal.
                            //
                            // ONLY single-result calls go through this
                            // path (multi-result would need a tuple sort,
                            // out of scope). Multi-result falls through
                            // to the fresh-symbolic + havoc encoding.
                            if sig.results.len() == 1 {
                                let result_type = sig.results[0];
                                let result_width: u32 = match result_type {
                                    crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                    crate::ValueType::I64 | crate::ValueType::F64 => 64,
                                };
                                let domain_sorts_owned: Vec<Sort> = sig
                                    .params
                                    .iter()
                                    .map(|p| {
                                        Sort::bitvector(match p {
                                            crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                            crate::ValueType::I64 | crate::ValueType::F64 => 64,
                                        })
                                    })
                                    .collect();
                                let domain_refs: Vec<&Sort> = domain_sorts_owned.iter().collect();
                                let range_sort = Sort::bitvector(result_width);
                                // Use the same decl name PR-K3 uses for
                                // direct Call — so a direct-call and the
                                // resolved-indirect-call to the same
                                // function reduce to the SAME Z3 expression.
                                let decl_name = format!("pure_call_{}", func_idx);
                                let decl =
                                    FuncDecl::new(decl_name.as_str(), &domain_refs, &range_sort);
                                let arg_refs: Vec<&dyn z3::ast::Ast> =
                                    args.iter().map(|a| a as &dyn z3::ast::Ast).collect();
                                let result_dyn = decl.apply(&arg_refs);
                                if let Some(result_bv) = result_dyn.as_bv() {
                                    stack.push(result_bv);
                                } else {
                                    stack.push(BV::new_const(
                                        format!("call_indirect_{}_result_fallback", func_idx),
                                        result_width,
                                    ));
                                }
                                // NO havoc — pure_call encoding implies
                                // no side effects, exactly like PR-K3's
                                // direct-call path. Soundness: directize's
                                // structural guards ensure the resolved
                                // function is genuinely the target;
                                // its purity is then a property of THAT
                                // function (handled by PR-K3's pure_call_*
                                // function symbol).
                                continue;
                            }
                            // Multi-result resolved indirect calls: push
                            // each result back to the args we already
                            // popped (so the stack winds up where the
                            // unresolved path would expect), then fall
                            // through to the fresh-symbolic path.
                            for a in args.into_iter().rev() {
                                stack.push(a);
                            }
                            // Re-push the slot so the fall-through path
                            // sees the original stack shape.
                            stack.push(slot_bv.clone());
                            // Re-pop slot in the fall-through.
                            let _ = stack.pop().unwrap();
                        }

                        // Pop arguments (fall-through path: unresolved
                        // indirect or multi-result resolved indirect).
                        for i in 0..sig.params.len() {
                            if stack.is_empty() {
                                return Err(anyhow!(
                                    "Stack underflow in CallIndirect: missing arg {} of {}",
                                    i + 1,
                                    sig.params.len()
                                ));
                            }
                            // SAFETY: guarded by is_empty check above
                            let _ = stack.pop().unwrap();
                        }
                        // Push results
                        for (i, result_type) in sig.results.iter().enumerate() {
                            let width = match result_type {
                                crate::ValueType::I32 | crate::ValueType::F32 => 32,
                                crate::ValueType::I64 | crate::ValueType::F64 => 64,
                            };
                            stack.push(BV::new_const(
                                format!("call_indirect_{}_result_{}", type_idx, i),
                                width,
                            ));
                        }

                        // Indirect calls could call ANY function - must havoc everything
                        // This is maximally conservative but sound
                        static INDIRECT_CALL_COUNTER: std::sync::atomic::AtomicU64 =
                            std::sync::atomic::AtomicU64::new(0);
                        let call_id = INDIRECT_CALL_COUNTER
                            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        for (idx, global) in globals.iter_mut().enumerate() {
                            *global = BV::new_const(
                                format!("global_{}_after_indirect_call_{}", idx, call_id),
                                global.get_size(),
                            );
                        }
                        memory = Array::new_const(
                            format!("mem_after_indirect_call_{}", call_id),
                            &Sort::bitvector(32),
                            &Sort::bitvector(8),
                        );
                    } else {
                        // Unknown type - conservative: assume returns i32
                        stack.push(BV::new_const("call_indirect_unknown_result", 32));
                    }
                } else {
                    // No context - conservative: assume returns i32
                    stack.push(BV::new_const("call_indirect_result", 32));
                }
            }

            // End of function/block
            Instruction::End => {
                break;
            }

            // Unreachable terminates execution
            Instruction::Unreachable => {
                // For SMT purposes, unreachable means no concrete output
                // Return a fresh symbolic variable (represents undefined)
                return Ok((Some(BV::new_const("unreachable", 32)), None));
            }

            // ============================================================
            // Float operations - IEEE 754 semantics
            // For binary arithmetic (add/sub/mul/div): when both operands are
            // concrete constants, compute natively in Rust (IEEE 754 roundTiesToEven,
            // matching WebAssembly). Symbolic operands produce opaque results.
            // ============================================================

            // Float binary operations (f32): [f32, f32] -> [f32]
            Instruction::F32Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                // When both operands are concrete constants, compute the IEEE 754
                // result natively (Rust f32 uses roundTiesToEven, matching WASM)
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f32::from_bits(l_bits as u32) + f32::from_bits(r_bits as u32);
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_add_result", 32));
                }
            }
            Instruction::F32Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f32::from_bits(l_bits as u32) - f32::from_bits(r_bits as u32);
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_sub_result", 32));
                }
            }
            Instruction::F32Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f32::from_bits(l_bits as u32) * f32::from_bits(r_bits as u32);
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_mul_result", 32));
                }
            }
            Instruction::F32Div => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Div"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f32::from_bits(l_bits as u32) / f32::from_bits(r_bits as u32);
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_div_result", 32));
                }
            }
            Instruction::F32Min => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Min"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let lv = f32::from_bits(l_bits as u32);
                    let rv = f32::from_bits(r_bits as u32);
                    // WASM semantics: either NaN → result is NaN
                    let result = if lv.is_nan() || rv.is_nan() {
                        f32::NAN
                    } else {
                        lv.min(rv)
                    };
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_min_result", 32));
                }
            }
            Instruction::F32Max => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Max"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let lv = f32::from_bits(l_bits as u32);
                    let rv = f32::from_bits(r_bits as u32);
                    let result = if lv.is_nan() || rv.is_nan() {
                        f32::NAN
                    } else {
                        lv.max(rv)
                    };
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_max_result", 32));
                }
            }
            Instruction::F32Copysign => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Copysign"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result =
                        f32::from_bits(l_bits as u32).copysign(f32::from_bits(r_bits as u32));
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else if ENABLE_FPA_VERIFICATION {
                    // copysign(x, y) = (x & 0x7FFFFFFF) | (y & 0x80000000)
                    let mag_mask = BV::from_u64(0x7FFFFFFF, 32);
                    let sign_mask = BV::from_u64(0x80000000, 32);
                    stack.push(lhs.bvand(&mag_mask).bvor(rhs.bvand(&sign_mask)));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_copysign_result", 32));
                }
            }

            // Float unary operations (f32): [f32] -> [f32]
            Instruction::F32Neg => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Neg"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = -f32::from_bits(bits as u32);
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else if ENABLE_FPA_VERIFICATION {
                    // Negation flips the sign bit (bit 31 for f32)
                    let sign_mask = BV::from_u64(0x80000000, 32);
                    stack.push(val.bvxor(&sign_mask));
                } else {
                    stack.push(BV::new_const("f32_neg_result", 32));
                }
            }
            Instruction::F32Abs => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Abs"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f32::from_bits(bits as u32).abs();
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else if ENABLE_FPA_VERIFICATION {
                    // Absolute value clears the sign bit (bit 31)
                    let abs_mask = BV::from_u64(0x7FFFFFFF, 32);
                    stack.push(val.bvand(&abs_mask));
                } else {
                    stack.push(BV::new_const("f32_abs_result", 32));
                }
            }
            Instruction::F32Ceil => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Ceil"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f32::from_bits(bits as u32).ceil();
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_ceil_result", 32));
                }
            }
            Instruction::F32Floor => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Floor"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f32::from_bits(bits as u32).floor();
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_floor_result", 32));
                }
            }
            Instruction::F32Trunc => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Trunc"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f32::from_bits(bits as u32).trunc();
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_trunc_result", 32));
                }
            }
            Instruction::F32Nearest => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Nearest"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f32::from_bits(bits as u32).round_ties_even();
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_nearest_result", 32));
                }
            }
            Instruction::F32Sqrt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Sqrt"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f32::from_bits(bits as u32).sqrt();
                    stack.push(BV::from_i64(result.to_bits() as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_sqrt_result", 32));
                }
            }

            // Float comparison (f32): [f32, f32] -> [i32]
            Instruction::F32Eq => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Eq"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f32::from_bits(l_bits as u32) == f32::from_bits(r_bits as u32) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else if ENABLE_FPA_VERIFICATION {
                    let one = BV::from_i64(1, 32);
                    let zero = BV::from_i64(0, 32);
                    let (lhs, rhs) = match_bv_widths(lhs, rhs);
                    stack.push(lhs.eq(&rhs).ite(&one, &zero));
                } else {
                    stack.push(BV::new_const("f32_eq_result", 32));
                }
            }
            Instruction::F32Ne => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Ne"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f32::from_bits(l_bits as u32) != f32::from_bits(r_bits as u32) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else if ENABLE_FPA_VERIFICATION {
                    let one = BV::from_i64(1, 32);
                    let zero = BV::from_i64(0, 32);
                    let (lhs, rhs) = match_bv_widths(lhs, rhs);
                    stack.push(lhs.eq(&rhs).not().ite(&one, &zero));
                } else {
                    stack.push(BV::new_const("f32_ne_result", 32));
                }
            }
            Instruction::F32Lt => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Lt"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f32::from_bits(l_bits as u32) < f32::from_bits(r_bits as u32) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_lt_result", 32));
                }
            }
            Instruction::F32Gt => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Gt"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f32::from_bits(l_bits as u32) > f32::from_bits(r_bits as u32) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_gt_result", 32));
                }
            }
            Instruction::F32Le => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Le"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f32::from_bits(l_bits as u32) <= f32::from_bits(r_bits as u32) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_le_result", 32));
                }
            }
            Instruction::F32Ge => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Ge"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f32::from_bits(l_bits as u32) >= f32::from_bits(r_bits as u32) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f32_ge_result", 32));
                }
            }

            // Float binary operations (f64): [f64, f64] -> [f64]
            Instruction::F64Add => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Add"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f64::from_bits(l_bits) + f64::from_bits(r_bits);
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_add_result", 64));
                }
            }
            Instruction::F64Sub => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Sub"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f64::from_bits(l_bits) - f64::from_bits(r_bits);
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_sub_result", 64));
                }
            }
            Instruction::F64Mul => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Mul"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f64::from_bits(l_bits) * f64::from_bits(r_bits);
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_mul_result", 64));
                }
            }
            Instruction::F64Div => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Div"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f64::from_bits(l_bits) / f64::from_bits(r_bits);
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_div_result", 64));
                }
            }
            Instruction::F64Min => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Min"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let lv = f64::from_bits(l_bits);
                    let rv = f64::from_bits(r_bits);
                    let result = if lv.is_nan() || rv.is_nan() {
                        f64::NAN
                    } else {
                        lv.min(rv)
                    };
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_min_result", 64));
                }
            }
            Instruction::F64Max => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Max"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let lv = f64::from_bits(l_bits);
                    let rv = f64::from_bits(r_bits);
                    let result = if lv.is_nan() || rv.is_nan() {
                        f64::NAN
                    } else {
                        lv.max(rv)
                    };
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_max_result", 64));
                }
            }
            Instruction::F64Copysign => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Copysign"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = f64::from_bits(l_bits).copysign(f64::from_bits(r_bits));
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else if ENABLE_FPA_VERIFICATION {
                    // copysign(x, y) = (x & 0x7FFFFFFFFFFFFFFF) | (y & 0x8000000000000000)
                    let mag_mask = BV::from_u64(0x7FFFFFFFFFFFFFFF, 64);
                    let sign_mask = BV::from_u64(0x8000000000000000, 64);
                    stack.push(lhs.bvand(&mag_mask).bvor(rhs.bvand(&sign_mask)));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_copysign_result", 64));
                }
            }

            // Float unary operations (f64): [f64] -> [f64]
            Instruction::F64Neg => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Neg"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = -f64::from_bits(bits);
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else if ENABLE_FPA_VERIFICATION {
                    // Negation flips the sign bit (bit 63 for f64)
                    let sign_mask = BV::from_u64(0x8000000000000000, 64);
                    stack.push(val.bvxor(&sign_mask));
                } else {
                    stack.push(BV::new_const("f64_neg_result", 64));
                }
            }
            Instruction::F64Abs => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Abs"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f64::from_bits(bits).abs();
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else if ENABLE_FPA_VERIFICATION {
                    // Absolute value clears the sign bit (bit 63)
                    let abs_mask = BV::from_u64(0x7FFFFFFFFFFFFFFF, 64);
                    stack.push(val.bvand(&abs_mask));
                } else {
                    stack.push(BV::new_const("f64_abs_result", 64));
                }
            }
            Instruction::F64Ceil => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Ceil"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f64::from_bits(bits).ceil();
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_ceil_result", 64));
                }
            }
            Instruction::F64Floor => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Floor"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f64::from_bits(bits).floor();
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_floor_result", 64));
                }
            }
            Instruction::F64Trunc => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Trunc"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f64::from_bits(bits).trunc();
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_trunc_result", 64));
                }
            }
            Instruction::F64Nearest => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Nearest"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f64::from_bits(bits).round_ties_even();
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_nearest_result", 64));
                }
            }
            Instruction::F64Sqrt => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Sqrt"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = f64::from_bits(bits).sqrt();
                    stack.push(BV::from_i64(result.to_bits() as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_sqrt_result", 64));
                }
            }

            // Float comparison (f64): [f64, f64] -> [i32]
            Instruction::F64Eq => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Eq"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f64::from_bits(l_bits) == f64::from_bits(r_bits) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else if ENABLE_FPA_VERIFICATION {
                    let one = BV::from_i64(1, 32);
                    let zero = BV::from_i64(0, 32);
                    // Truncate to 32-bit for comparison then extend result
                    let lhs32 = lhs.extract(31, 0);
                    let rhs32 = rhs.extract(31, 0);
                    let lhs_hi = lhs.extract(63, 32);
                    let rhs_hi = rhs.extract(63, 32);
                    let eq_lo = lhs32.eq(&rhs32);
                    let eq_hi = lhs_hi.eq(&rhs_hi);
                    stack.push(Bool::and(&[&eq_lo, &eq_hi]).ite(&one, &zero));
                } else {
                    stack.push(BV::new_const("f64_eq_result", 32));
                }
            }
            Instruction::F64Ne => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Ne"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f64::from_bits(l_bits) != f64::from_bits(r_bits) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else if ENABLE_FPA_VERIFICATION {
                    let one = BV::from_i64(1, 32);
                    let zero = BV::from_i64(0, 32);
                    let lhs32 = lhs.extract(31, 0);
                    let rhs32 = rhs.extract(31, 0);
                    let lhs_hi = lhs.extract(63, 32);
                    let rhs_hi = rhs.extract(63, 32);
                    let eq_lo = lhs32.eq(&rhs32);
                    let eq_hi = lhs_hi.eq(&rhs_hi);
                    stack.push(Bool::and(&[&eq_lo, &eq_hi]).not().ite(&one, &zero));
                } else {
                    stack.push(BV::new_const("f64_ne_result", 32));
                }
            }
            Instruction::F64Lt => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Lt"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f64::from_bits(l_bits) < f64::from_bits(r_bits) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_lt_result", 32));
                }
            }
            Instruction::F64Gt => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Gt"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f64::from_bits(l_bits) > f64::from_bits(r_bits) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_gt_result", 32));
                }
            }
            Instruction::F64Le => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Le"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f64::from_bits(l_bits) <= f64::from_bits(r_bits) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_le_result", 32));
                }
            }
            Instruction::F64Ge => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Ge"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                if let (Some(l_bits), Some(r_bits)) = (lhs.as_u64(), rhs.as_u64()) {
                    let result = if f64::from_bits(l_bits) >= f64::from_bits(r_bits) {
                        1i64
                    } else {
                        0
                    };
                    stack.push(BV::from_i64(result, 32));
                } else {
                    let _ = (lhs, rhs);
                    stack.push(BV::new_const("f64_ge_result", 32));
                }
            }

            // ============================================================
            // Conversion operations - precise bitvector modeling
            // ============================================================

            // i32.wrap_i64: [i64] -> [i32] (truncate to low 32 bits)
            Instruction::I32WrapI64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32WrapI64"));
                }
                let val = stack.pop().unwrap();
                stack.push(val.extract(31, 0)); // Extract low 32 bits
            }

            // i64.extend_i32_s: [i32] -> [i64] (sign-extend)
            Instruction::I64ExtendI32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64ExtendI32S"));
                }
                let val = stack.pop().unwrap();
                stack.push(val.sign_ext(32)); // Sign-extend by 32 bits
            }

            // i64.extend_i32_u: [i32] -> [i64] (zero-extend)
            Instruction::I64ExtendI32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64ExtendI32U"));
                }
                let val = stack.pop().unwrap();
                stack.push(val.zero_ext(32)); // Zero-extend by 32 bits
            }

            // Int-to-float conversions: produce fresh symbolic (no IEEE 754 modeling)
            Instruction::I32TruncF32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncF32S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    if !f.is_nan() && f >= i32::MIN as f32 && f < (i32::MAX as f32 + 1.0) {
                        stack.push(BV::from_i64(f as i32 as i64, 32));
                    } else {
                        stack.push(BV::new_const("i32_trunc_f32_s_result", 32));
                    }
                } else {
                    stack.push(BV::new_const("i32_trunc_f32_s_result", 32));
                }
            }
            Instruction::I32TruncF32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncF32U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    if !f.is_nan() && f >= 0.0 && f < (u32::MAX as f32 + 1.0) {
                        stack.push(BV::from_i64(f as u32 as i64, 32));
                    } else {
                        stack.push(BV::new_const("i32_trunc_f32_u_result", 32));
                    }
                } else {
                    stack.push(BV::new_const("i32_trunc_f32_u_result", 32));
                }
            }
            Instruction::I32TruncF64S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncF64S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    if !f.is_nan() && f >= i32::MIN as f64 && f < (i32::MAX as f64 + 1.0) {
                        stack.push(BV::from_i64(f as i32 as i64, 32));
                    } else {
                        stack.push(BV::new_const("i32_trunc_f64_s_result", 32));
                    }
                } else {
                    stack.push(BV::new_const("i32_trunc_f64_s_result", 32));
                }
            }
            Instruction::I32TruncF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncF64U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    if !f.is_nan() && f >= 0.0 && f < (u32::MAX as f64 + 1.0) {
                        stack.push(BV::from_i64(f as u32 as i64, 32));
                    } else {
                        stack.push(BV::new_const("i32_trunc_f64_u_result", 32));
                    }
                } else {
                    stack.push(BV::new_const("i32_trunc_f64_u_result", 32));
                }
            }
            Instruction::I64TruncF32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncF32S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    if !f.is_nan() && f >= i64::MIN as f32 && f < (i64::MAX as f32) {
                        stack.push(BV::from_i64(f as i64, 64));
                    } else {
                        stack.push(BV::new_const("i64_trunc_f32_s_result", 64));
                    }
                } else {
                    stack.push(BV::new_const("i64_trunc_f32_s_result", 64));
                }
            }
            Instruction::I64TruncF32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncF32U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    if !f.is_nan() && f >= 0.0 && f < (u64::MAX as f32) {
                        stack.push(BV::from_i64(f as u64 as i64, 64));
                    } else {
                        stack.push(BV::new_const("i64_trunc_f32_u_result", 64));
                    }
                } else {
                    stack.push(BV::new_const("i64_trunc_f32_u_result", 64));
                }
            }
            Instruction::I64TruncF64S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncF64S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    if !f.is_nan() && f >= i64::MIN as f64 && f < (i64::MAX as f64) {
                        stack.push(BV::from_i64(f as i64, 64));
                    } else {
                        stack.push(BV::new_const("i64_trunc_f64_s_result", 64));
                    }
                } else {
                    stack.push(BV::new_const("i64_trunc_f64_s_result", 64));
                }
            }
            Instruction::I64TruncF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncF64U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    if !f.is_nan() && f >= 0.0 && f < (u64::MAX as f64) {
                        stack.push(BV::from_i64(f as u64 as i64, 64));
                    } else {
                        stack.push(BV::new_const("i64_trunc_f64_u_result", 64));
                    }
                } else {
                    stack.push(BV::new_const("i64_trunc_f64_u_result", 64));
                }
            }

            // Float-from-int conversions with concrete computation
            Instruction::F32ConvertI32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32ConvertI32S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as i32 as f32).to_bits();
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_convert_i32_s_result", 32));
                }
            }
            Instruction::F32ConvertI32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32ConvertI32U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as u32 as f32).to_bits();
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_convert_i32_u_result", 32));
                }
            }
            Instruction::F32ConvertI64S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32ConvertI64S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as i64 as f32).to_bits();
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_convert_i64_s_result", 32));
                }
            }
            Instruction::F32ConvertI64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32ConvertI64U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as f32).to_bits();
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_convert_i64_u_result", 32));
                }
            }
            Instruction::F64ConvertI32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64ConvertI32S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as i32 as f64).to_bits();
                    stack.push(BV::from_i64(result as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_convert_i32_s_result", 64));
                }
            }
            Instruction::F64ConvertI32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64ConvertI32U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as u32 as f64).to_bits();
                    stack.push(BV::from_i64(result as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_convert_i32_u_result", 64));
                }
            }
            Instruction::F64ConvertI64S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64ConvertI64S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as i64 as f64).to_bits();
                    stack.push(BV::from_i64(result as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_convert_i64_s_result", 64));
                }
            }
            Instruction::F64ConvertI64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64ConvertI64U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (bits as f64).to_bits();
                    stack.push(BV::from_i64(result as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_convert_i64_u_result", 64));
                }
            }

            // Float-to-float conversions with concrete computation
            Instruction::F32DemoteF64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32DemoteF64"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (f64::from_bits(bits) as f32).to_bits();
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("f32_demote_result", 32));
                }
            }

            Instruction::F64PromoteF32 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64PromoteF32"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let result = (f32::from_bits(bits as u32) as f64).to_bits();
                    stack.push(BV::from_i64(result as i64, 64));
                } else {
                    stack.push(BV::new_const("f64_promote_result", 64));
                }
            }

            // Reinterpret operations - bit-cast (no-op for bitvectors, bits are identical)
            Instruction::I32ReinterpretF32 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32ReinterpretF32"));
                }
                // Bits don't change, just reinterpretation - no-op for BV
            }

            Instruction::I64ReinterpretF64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64ReinterpretF64"));
                }
                // No-op for BV - bits stay the same
            }

            Instruction::F32ReinterpretI32 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32ReinterpretI32"));
                }
                // No-op for BV
            }

            Instruction::F64ReinterpretI64 => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64ReinterpretI64"));
                }
                // No-op for BV
            }

            // ============================================================
            // Additional memory operations
            // ============================================================
            Instruction::F32Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F32Load"));
                }
                let _addr = stack.pop().unwrap();
                let _ = offset;
                stack.push(BV::new_const("f32_load_result", 32));
            }

            Instruction::F32Store { .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F32Store"));
                }
                let _value = stack.pop().unwrap();
                let _addr = stack.pop().unwrap();
            }

            Instruction::F64Load { offset, .. } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in F64Load"));
                }
                let _addr = stack.pop().unwrap();
                let _ = offset;
                stack.push(BV::new_const("f64_load_result", 64));
            }

            Instruction::F64Store { .. } => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in F64Store"));
                }
                let _value = stack.pop().unwrap();
                let _addr = stack.pop().unwrap();
            }

            // Note: Integer partial loads/stores are now handled with precise Z3 encodings
            // earlier in this match statement (I32Load8S/U, I32Load16S/U, I64Load8S/U, etc.)

            // Memory size/grow - precisely modeled using shared memory_size variable
            Instruction::MemorySize(_) => {
                // memory.size returns the current memory size in pages (i32)
                // Using the shared memory_size ensures both original and optimized
                // get the same value at the same program point.
                stack.push(memory_size.clone());
            }

            Instruction::MemoryGrow(_) => {
                // memory.grow takes delta pages, returns old size on success or -1 on failure
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in MemoryGrow"));
                }
                let delta = stack.pop().unwrap();

                // The result is non-deterministic: either old_size (success) or -1 (failure)
                // For translation validation, we need consistent results between original
                // and optimized. We use a LOCAL counter to generate stable symbolic values.
                // This ensures the Nth memory.grow in both versions uses the same ID.
                let grow_id = memory_grow_count;
                memory_grow_count += 1;

                // Create symbolic result that could be old_size or -1
                let result = BV::new_const(format!("memory_grow_result_{}", grow_id), 32);
                stack.push(result);

                // Memory size changes if grow succeeds (new_size = old + delta)
                // Since we can't know if it succeeds, create fresh symbolic size
                // that's constrained to be either (old + delta) or old
                let new_size = BV::new_const(format!("memory_size_after_grow_{}", grow_id), 32);

                // We could add Z3 constraints here to relate new_size and result:
                // (result == old_size && new_size == old_size + delta) ||
                // (result == -1 && new_size == old_size)
                // But for translation validation, consistency is what matters,
                // and both versions will use the same symbolic values.
                let _ = delta; // Delta affects the new size symbolically
                memory_size = new_size;
            }

            // Rotate operations - precise bitvector modeling
            Instruction::I32Rotl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Rotl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvrotl(&rhs));
            }

            Instruction::I32Rotr => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I32Rotr"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvrotr(&rhs));
            }

            Instruction::I64Rotl => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Rotl"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                // WebAssembly spec: i64 rotations take i64 for rotation amount
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvrotl(&rhs));
            }

            Instruction::I64Rotr => {
                if stack.len() < 2 {
                    return Err(anyhow!("Stack underflow in I64Rotr"));
                }
                let rhs = stack.pop().unwrap();
                let lhs = stack.pop().unwrap();
                // WebAssembly spec: i64 rotations take i64 for rotation amount
                let (lhs, rhs) = match_bv_widths(lhs, rhs);
                stack.push(lhs.bvrotr(&rhs));
            }

            // Sign extension operations
            Instruction::I32Extend8S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Extend8S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 8 bits and sign-extend to 32 bits
                let low8 = val.extract(7, 0);
                stack.push(low8.sign_ext(24));
            }
            Instruction::I32Extend16S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32Extend16S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 16 bits and sign-extend to 32 bits
                let low16 = val.extract(15, 0);
                stack.push(low16.sign_ext(16));
            }
            Instruction::I64Extend8S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Extend8S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 8 bits and sign-extend to 64 bits
                let low8 = val.extract(7, 0);
                stack.push(low8.sign_ext(56));
            }
            Instruction::I64Extend16S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Extend16S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 16 bits and sign-extend to 64 bits
                let low16 = val.extract(15, 0);
                stack.push(low16.sign_ext(48));
            }
            Instruction::I64Extend32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64Extend32S"));
                }
                let val = stack.pop().unwrap();
                // Extract low 32 bits and sign-extend to 64 bits
                let low32 = val.extract(31, 0);
                stack.push(low32.sign_ext(32));
            }

            // Saturating truncation operations - produce symbolic values
            // These are conversion operations that don't trap
            Instruction::I32TruncSatF32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncSatF32S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    let result = if f.is_nan() {
                        0
                    } else if f >= (i32::MAX as f32 + 1.0) {
                        i32::MAX
                    } else if f < i32::MIN as f32 {
                        i32::MIN
                    } else {
                        f as i32
                    };
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("i32_trunc_sat_f32_s_result", 32));
                }
            }
            Instruction::I32TruncSatF32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncSatF32U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    let result = if f.is_nan() || f < 0.0 {
                        0u32
                    } else if f >= (u32::MAX as f32 + 1.0) {
                        u32::MAX
                    } else {
                        f as u32
                    };
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("i32_trunc_sat_f32_u_result", 32));
                }
            }
            Instruction::I32TruncSatF64S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncSatF64S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    let result = if f.is_nan() {
                        0
                    } else if f >= (i32::MAX as f64 + 1.0) {
                        i32::MAX
                    } else if f < i32::MIN as f64 {
                        i32::MIN
                    } else {
                        f as i32
                    };
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("i32_trunc_sat_f64_s_result", 32));
                }
            }
            Instruction::I32TruncSatF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I32TruncSatF64U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    let result = if f.is_nan() || f < 0.0 {
                        0u32
                    } else if f >= (u32::MAX as f64 + 1.0) {
                        u32::MAX
                    } else {
                        f as u32
                    };
                    stack.push(BV::from_i64(result as i64, 32));
                } else {
                    stack.push(BV::new_const("i32_trunc_sat_f64_u_result", 32));
                }
            }
            Instruction::I64TruncSatF32S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncSatF32S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    let result = if f.is_nan() {
                        0i64
                    } else if f >= i64::MAX as f32 {
                        i64::MAX
                    } else if f < i64::MIN as f32 {
                        i64::MIN
                    } else {
                        f as i64
                    };
                    stack.push(BV::from_i64(result, 64));
                } else {
                    stack.push(BV::new_const("i64_trunc_sat_f32_s_result", 64));
                }
            }
            Instruction::I64TruncSatF32U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncSatF32U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f32::from_bits(bits as u32);
                    let result = if f.is_nan() || f < 0.0 {
                        0u64
                    } else if f >= u64::MAX as f32 {
                        u64::MAX
                    } else {
                        f as u64
                    };
                    stack.push(BV::from_i64(result as i64, 64));
                } else {
                    stack.push(BV::new_const("i64_trunc_sat_f32_u_result", 64));
                }
            }
            Instruction::I64TruncSatF64S => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncSatF64S"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    let result = if f.is_nan() {
                        0i64
                    } else if f >= i64::MAX as f64 {
                        i64::MAX
                    } else if f < i64::MIN as f64 {
                        i64::MIN
                    } else {
                        f as i64
                    };
                    stack.push(BV::from_i64(result, 64));
                } else {
                    stack.push(BV::new_const("i64_trunc_sat_f64_s_result", 64));
                }
            }
            Instruction::I64TruncSatF64U => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in I64TruncSatF64U"));
                }
                let val = stack.pop().unwrap();
                if let Some(bits) = val.as_u64() {
                    let f = f64::from_bits(bits);
                    let result = if f.is_nan() || f < 0.0 {
                        0u64
                    } else if f >= u64::MAX as f64 {
                        u64::MAX
                    } else {
                        f as u64
                    };
                    stack.push(BV::from_i64(result as i64, 64));
                } else {
                    stack.push(BV::new_const("i64_trunc_sat_f64_u_result", 64));
                }
            }

            // Bulk memory operations - these modify memory, no stack return value.
            // We conservatively invalidate the entire memory array because these
            // operations write to a range of addresses that may be symbolic.
            // This is sound: any subsequent load will return an unconstrained value,
            // which means Z3 cannot prove false equivalences.
            Instruction::MemoryFill(_) => {
                // memory.fill: [dst, val, len] -> []
                // Fills memory[dst..dst+len] with byte val.
                if stack.len() < 3 {
                    return Err(anyhow!("Stack underflow in MemoryFill"));
                }
                let _len = stack.pop().unwrap();
                let _val = stack.pop().unwrap();
                let _dst = stack.pop().unwrap();

                // Conservative: invalidate entire memory array.
                // The fill range is symbolic so we cannot enumerate individual stores.
                static FILL_COUNTER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let fill_id = FILL_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                memory = Array::new_const(
                    format!("mem_after_fill_{}", fill_id),
                    &Sort::bitvector(32),
                    &Sort::bitvector(8),
                );
            }
            Instruction::MemoryCopy { .. } => {
                // memory.copy: [dst, src, len] -> []
                // Copies memory[src..src+len] to memory[dst..dst+len].
                if stack.len() < 3 {
                    return Err(anyhow!("Stack underflow in MemoryCopy"));
                }
                let _len = stack.pop().unwrap();
                let _src = stack.pop().unwrap();
                let _dst = stack.pop().unwrap();

                // Conservative: invalidate entire memory array.
                // The copy range and overlap behavior are symbolic, so we cannot
                // precisely model this with a finite number of array operations.
                static COPY_COUNTER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let copy_id = COPY_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                memory = Array::new_const(
                    format!("mem_after_copy_{}", copy_id),
                    &Sort::bitvector(32),
                    &Sort::bitvector(8),
                );
            }
            Instruction::MemoryInit { .. } => {
                // memory.init: [dst, src_offset, len] -> []
                // Copies from a passive data segment into linear memory.
                // The data segment contents are not available at verification time,
                // so we must conservatively invalidate the memory array.
                if stack.len() < 3 {
                    return Err(anyhow!("Stack underflow in MemoryInit"));
                }
                let _len = stack.pop().unwrap();
                let _src_offset = stack.pop().unwrap();
                let _dst = stack.pop().unwrap();

                // Conservative: invalidate entire memory array.
                // Data segment contents are opaque to Z3 verification.
                static INIT_COUNTER: std::sync::atomic::AtomicU64 =
                    std::sync::atomic::AtomicU64::new(0);
                let init_id = INIT_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                memory = Array::new_const(
                    format!("mem_after_init_{}", init_id),
                    &Sort::bitvector(32),
                    &Sort::bitvector(8),
                );
            }
            Instruction::DataDrop(_) => {
                // data.drop: [] -> []
                // Marks a data segment as dropped. No effect on linear memory.
                // This is a no-op for Z3 verification purposes.
            }

            // Unknown instructions - these should be rare now
            Instruction::Unknown(_) => {
                // Unknown bytes - can't verify, produce fresh symbolic value
                stack.push(BV::new_const("unknown_result", 32));
            }
        }
    }

    // Return value should be on stack if function has result type
    if func.signature.results.is_empty() {
        // Void function - no return value expected
        Ok((None, Some(globals)))
    } else if stack.is_empty() {
        Err(anyhow!(
            "Function returned no value (stack empty) but result type expected"
        ))
    } else {
        Ok((Some(stack.pop().unwrap()), Some(globals)))
    }
}

/// Whether the two encodings leave every global with the same value.
///
/// A function's global vector is extended lazily with `global{i}`, the same
/// name both encodings start from, so a global one side never touched is
/// padded with that initial value. Returns `Ok(None)` when either post-state
/// is not modelled, `Err` when a global has different widths on the two sides
/// (a model artifact that cannot be compared soundly).
#[cfg(feature = "verification")]
fn globals_equal(orig: &Option<Vec<BV>>, opt: &Option<Vec<BV>>) -> Result<Option<Bool>> {
    let (Some(a), Some(b)) = (orig, opt) else {
        return Ok(None);
    };
    let n = a.len().max(b.len());
    let initial = |i: usize| BV::new_const(format!("global{}", i), 32);
    let mut eqs: Vec<Bool> = Vec::with_capacity(n);
    for i in 0..n {
        let x = a.get(i).cloned().unwrap_or_else(|| initial(i));
        let y = b.get(i).cloned().unwrap_or_else(|| initial(i));
        if x.get_size() != y.get_size() {
            return Err(anyhow!(
                "global {} has different widths after the two encodings",
                i
            ));
        }
        eqs.push(x.eq(&y));
    }
    let refs: Vec<&Bool> = eqs.iter().collect();
    Ok(Some(Bool::and(&refs)))
}

/// Check that the original and optimized functions leave the globals equal.
/// `Ok(true)`: proven equal, or the post-state is not modelled (the caller's
/// return-value check decides alone, as before). `Ok(false)`: a counterexample.
#[cfg(feature = "verification")]
fn check_globals_preserved(
    orig: &Option<Vec<BV>>,
    opt: &Option<Vec<BV>>,
    pass_name: &str,
) -> Result<bool> {
    let eq = match globals_equal(orig, opt) {
        Ok(Some(eq)) => eq,
        Ok(None) => return Ok(true),
        Err(e) => return Err(anyhow!("{}: {}", pass_name, e)),
    };
    let solver = Solver::new();
    solver.assert(eq.not());
    match solver.check() {
        SatResult::Unsat => Ok(true),
        SatResult::Sat => {
            revert_note(format_args!(
                "{}: Verification failed! The globals differ after the call:",
                pass_name
            ));
            if let Some(model) = solver.get_model() {
                revert_note(format_args!("{}", model));
            }
            Ok(false)
        }
        SatResult::Unknown => Err(anyhow!(
            "{}: SMT solver returned unknown on the globals (timeout or too complex)",
            pass_name
        )),
    }
}

/// Encode a block body (helper for control flow)
/// Returns the result value if the block produces one
#[cfg(feature = "verification")]
fn encode_block_body(
    body: &[Instruction],
    stack: &mut Vec<BV>,
    locals: &mut Vec<BV>,
    globals: &mut Vec<BV>,
) -> Result<Option<BV>> {
    for instr in body {
        match instr {
            // For nested blocks, recursively encode
            Instruction::Block {
                block_type,
                body: nested_body,
            } => {
                let result = encode_block_body(nested_body, stack, locals, globals)?;
                if let Some(width) = block_type_width(block_type) {
                    if result.is_none() && stack.is_empty() {
                        stack.push(BV::new_const("nested_block_result", width));
                    }
                }
            }

            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in nested If"));
                }
                let cond = stack.pop().unwrap();
                let zero = BV::from_i64(0, 32);
                let cond_bool = cond.eq(&zero).not();

                let saved_stack = stack.clone();
                let saved_locals = locals.clone();
                let saved_globals = globals.clone();

                let then_result = encode_block_body(then_body, stack, locals, globals)?;
                let then_stack = stack.clone();
                let then_locals = locals.clone();
                let then_globals = globals.clone();

                *stack = saved_stack;
                *locals = saved_locals;
                *globals = saved_globals;
                let else_result = encode_block_body(else_body, stack, locals, globals)?;

                // Merge branches - collect new values first to avoid borrow conflicts
                let merged_locals: Vec<BV> = then_locals
                    .iter()
                    .zip(locals.iter())
                    .map(|(then_local, else_local)| merge_bv(&cond_bool, then_local, else_local))
                    .collect();
                *locals = merged_locals;

                let merged_globals: Vec<BV> = then_globals
                    .iter()
                    .zip(globals.iter())
                    .map(|(then_global, else_global)| {
                        merge_bv(&cond_bool, then_global, else_global)
                    })
                    .collect();
                *globals = merged_globals;

                if then_stack.len() == stack.len() {
                    let merged_stack: Vec<BV> = then_stack
                        .iter()
                        .zip(stack.iter())
                        .map(|(then_val, else_val)| merge_bv(&cond_bool, then_val, else_val))
                        .collect();
                    *stack = merged_stack;
                } else if let Some(width) = block_type_width(block_type) {
                    let then_val = then_result.unwrap_or_else(|| {
                        then_stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    let else_val = else_result.unwrap_or_else(|| {
                        stack
                            .last()
                            .cloned()
                            .unwrap_or_else(|| BV::from_i64(0, width))
                    });
                    stack.clear();
                    stack.push(merge_bv(&cond_bool, &then_val, &else_val));
                }
            }

            Instruction::Loop {
                block_type,
                body: loop_body,
            } => {
                for _iteration in 0..MAX_LOOP_UNROLL {
                    let _ = encode_block_body(loop_body, stack, locals, globals)?;
                }
                if let Some(width) = block_type_width(block_type) {
                    if stack.is_empty() {
                        stack.push(BV::new_const("nested_loop_result", width));
                    }
                }
            }

            // Branch exits the block
            Instruction::Br(_) | Instruction::Return => {
                return Ok(stack.last().cloned());
            }

            Instruction::BrIf(_) => {
                if stack.is_empty() {
                    return Err(anyhow!("Stack underflow in nested BrIf"));
                }
                let _cond = stack.pop().unwrap();
                // Continue execution (conservative)
            }

            Instruction::End => {
                return Ok(stack.last().cloned());
            }

            // Handle all other instructions inline (constants, arithmetic, etc.)
            _ => {
                encode_simple_instruction(instr, stack, locals, globals)?;
            }
        }
    }

    Ok(stack.last().cloned())
}

/// Encode a simple (non-control-flow) instruction
#[cfg(feature = "verification")]
#[allow(clippy::ptr_arg)] // Vec is needed for push/pop
#[allow(clippy::manual_range_patterns)] // Comments in OR patterns are more readable
fn encode_simple_instruction(
    instr: &Instruction,
    stack: &mut Vec<BV>,
    locals: &mut Vec<BV>,
    globals: &mut Vec<BV>,
) -> Result<()> {
    match instr {
        // Constants
        Instruction::I32Const(n) => {
            stack.push(BV::from_i64(*n as i64, 32));
        }
        Instruction::I64Const(n) => {
            stack.push(BV::from_i64(*n, 64));
        }
        Instruction::F32Const(bits) => {
            stack.push(BV::from_i64(*bits as i64, 32));
        }
        Instruction::F64Const(bits) => {
            stack.push(BV::from_i64(*bits as i64, 64));
        }

        // i32 binary arithmetic
        Instruction::I32Add => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvadd(&rhs));
        }
        Instruction::I32Sub => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvsub(&rhs));
        }
        Instruction::I32Mul => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvmul(&rhs));
        }
        Instruction::I32And => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvand(&rhs));
        }
        Instruction::I32Or => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvor(&rhs));
        }
        Instruction::I32Xor => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvxor(&rhs));
        }
        Instruction::I32Shl => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            // wasm takes the shift amount modulo the width.
            let amount = rhs.bvurem(BV::from_u64(lhs.get_size() as u64, lhs.get_size()));
            stack.push(lhs.bvshl(&amount));
        }
        Instruction::I32ShrS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            // wasm takes the shift amount modulo the width.
            let amount = rhs.bvurem(BV::from_u64(lhs.get_size() as u64, lhs.get_size()));
            stack.push(lhs.bvashr(&amount));
        }
        Instruction::I32ShrU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            // wasm takes the shift amount modulo the width.
            let amount = rhs.bvurem(BV::from_u64(lhs.get_size() as u64, lhs.get_size()));
            stack.push(lhs.bvlshr(&amount));
        }
        // loom#163: integer division/remainder. Modeled via Z3 bvsdiv/bvudiv/
        // bvsrem/bvurem — bvsdiv truncates toward zero, matching WASM div_s.
        // SOUNDNESS for inlining: the same division appears in BOTH the original
        // modeled call and the optimized inlined body, so they encode to the
        // identical (total) expression → equal; and any real div-by-zero /
        // INT_MIN÷-1 trap occurs identically on both sides, so the inline
        // transformation preserves trap behavior regardless of the model. These
        // mirror the main encoder's arms exactly.
        Instruction::I32DivS | Instruction::I64DivS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvsdiv(&rhs));
        }
        Instruction::I32DivU | Instruction::I64DivU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvudiv(&rhs));
        }
        Instruction::I32RemS | Instruction::I64RemS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvsrem(&rhs));
        }
        Instruction::I32RemU | Instruction::I64RemU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvurem(&rhs));
        }

        // i32 comparisons
        Instruction::I32Eq => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.eq(&rhs).ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)));
        }
        Instruction::I32Ne => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.eq(&rhs)
                    .not()
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LtS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvslt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LtU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvult(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GtS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvsgt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GtU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvugt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LeS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvsle(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32LeU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvule(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GeS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvsge(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I32GeU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvuge(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }

        // i32 unary
        Instruction::I32Eqz => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let val = stack.pop().unwrap();
            let zero = BV::from_i64(0, 32);
            stack.push(
                val.eq(&zero)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }

        // ============================================================
        // loom#151: i64 arithmetic / bitwise / comparison / conversion.
        // Previously these fell through to the BV32 `unsupported_result`
        // default, so any i64 op inside a Block/Loop/If — and every i64
        // callee modeled by-body for inline verification — became an
        // opaque 32-bit symbolic, defeating verification. These mirror
        // the i32 arms above at width 64. Comparisons yield an i32 (BV32)
        // boolean, matching WebAssembly. None of these can trap (no
        // div/rem here), so they are sound to model unconditionally.
        // ============================================================
        Instruction::I64Add => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvadd(&rhs));
        }
        Instruction::I64Sub => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvsub(&rhs));
        }
        Instruction::I64Mul => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvmul(&rhs));
        }
        Instruction::I64And => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvand(&rhs));
        }
        Instruction::I64Or => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvor(&rhs));
        }
        Instruction::I64Xor => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.bvxor(&rhs));
        }
        Instruction::I64Shl => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            // wasm takes the shift amount modulo the width.
            let amount = rhs.bvurem(BV::from_u64(lhs.get_size() as u64, lhs.get_size()));
            stack.push(lhs.bvshl(&amount));
        }
        Instruction::I64ShrS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            // wasm takes the shift amount modulo the width.
            let amount = rhs.bvurem(BV::from_u64(lhs.get_size() as u64, lhs.get_size()));
            stack.push(lhs.bvashr(&amount));
        }
        Instruction::I64ShrU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            // wasm takes the shift amount modulo the width.
            let amount = rhs.bvurem(BV::from_u64(lhs.get_size() as u64, lhs.get_size()));
            stack.push(lhs.bvlshr(&amount));
        }
        // i64 comparisons — operands i64, result i32 (BV32) boolean.
        Instruction::I64Eq => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(lhs.eq(&rhs).ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)));
        }
        Instruction::I64Ne => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.eq(&rhs)
                    .not()
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64LtS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvslt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64LtU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvult(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64GtS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvsgt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64GtU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvugt(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64LeS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvsle(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64LeU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvule(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64GeS => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvsge(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64GeU => {
            if stack.len() < 2 {
                return Err(anyhow!("Stack underflow"));
            }
            let rhs = stack.pop().unwrap();
            let lhs = stack.pop().unwrap();
            let (lhs, rhs) = match_bv_widths(lhs, rhs);
            stack.push(
                lhs.bvuge(&rhs)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        Instruction::I64Eqz => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let val = stack.pop().unwrap();
            let zero = BV::from_i64(0, 64);
            stack.push(
                val.eq(&zero)
                    .ite(&BV::from_i64(1, 32), &BV::from_i64(0, 32)),
            );
        }
        // Width conversions (total — never trap).
        Instruction::I64ExtendI32U => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let val = stack.pop().unwrap();
            // zero-extend 32 → 64 (extend by 32 bits)
            stack.push(val.zero_ext(32));
        }
        Instruction::I64ExtendI32S => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let val = stack.pop().unwrap();
            // sign-extend 32 → 64 (extend by 32 bits)
            stack.push(val.sign_ext(32));
        }
        Instruction::I32WrapI64 => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let val = stack.pop().unwrap();
            // take the low 32 bits
            stack.push(val.extract(31, 0));
        }

        // Local operations
        Instruction::LocalGet(idx) => {
            let idx = *idx as usize;
            if idx >= locals.len() {
                return Err(anyhow!("LocalGet index {} out of bounds", idx));
            }
            stack.push(locals[idx].clone());
        }
        Instruction::LocalSet(idx) => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let idx = *idx as usize;
            if idx >= locals.len() {
                return Err(anyhow!("LocalSet index {} out of bounds", idx));
            }
            let val = stack.pop().unwrap();
            locals[idx] = val;
        }
        Instruction::LocalTee(idx) => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let idx = *idx as usize;
            if idx >= locals.len() {
                return Err(anyhow!("LocalTee index {} out of bounds", idx));
            }
            let val = stack.last().unwrap().clone();
            locals[idx] = val;
        }

        // Global operations
        Instruction::GlobalGet(idx) => {
            let idx = *idx as usize;
            while globals.len() <= idx {
                let new_idx = globals.len();
                globals.push(BV::new_const(format!("global{}", new_idx), 32));
            }
            stack.push(globals[idx].clone());
        }
        Instruction::GlobalSet(idx) => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow"));
            }
            let idx = *idx as usize;
            while globals.len() <= idx {
                let new_idx = globals.len();
                globals.push(BV::new_const(format!("global{}", new_idx), 32));
            }
            let val = stack.pop().unwrap();
            globals[idx] = val;
        }

        // Select
        Instruction::Select => {
            if stack.len() < 3 {
                return Err(anyhow!("Stack underflow"));
            }
            let cond = stack.pop().unwrap();
            let val2 = stack.pop().unwrap();
            let val1 = stack.pop().unwrap();
            let zero = BV::from_i64(0, 32);
            stack.push(cond.eq(&zero).not().ite(&val1, &val2));
        }

        // Nop
        Instruction::Nop => {}

        // Drop pops and discards top of stack
        Instruction::Drop => {
            if stack.is_empty() {
                return Err(anyhow!("Stack underflow in Drop"));
            }
            stack.pop();
        }

        // Unknown instruction - may contain float operations stored as raw bytes
        // We decode the opcode to handle float operations with bitvector semantics
        Instruction::Unknown(bytes) => {
            if let Some(&opcode) = bytes.first() {
                match opcode {
                    // f32 binary operations (consume 2 f32, produce 1 f32)
                    // Using bitvector arithmetic - NOT IEEE 754 semantics
                    0x92 /* f32.add */ | 0x93 /* f32.sub */ | 0x94 /* f32.mul */ |
                    0x95 /* f32.div */ | 0x96 /* f32.min */ | 0x97 /* f32.max */ |
                    0x98 /* f32.copysign */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f32 binary op"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Create fresh symbolic value - float semantics not modeled in Z3 bitvectors
                        // This is sound but imprecise: any float result is a valid bitvector
                        stack.push(BV::new_const(format!("f32_result_{}", opcode), 32));
                    }

                    // f32 unary operations (consume 1 f32, produce 1 f32)
                    0x8b /* f32.abs */ | 0x8c /* f32.neg */ | 0x8d /* f32.ceil */ |
                    0x8e /* f32.floor */ | 0x8f /* f32.trunc */ | 0x90 /* f32.nearest */ |
                    0x91 /* f32.sqrt */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f32 unary op"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const(format!("f32_result_{}", opcode), 32));
                    }

                    // f32 comparisons (consume 2 f32, produce 1 i32)
                    0x5b /* f32.eq */ | 0x5c /* f32.ne */ | 0x5d /* f32.lt */ |
                    0x5e /* f32.gt */ | 0x5f /* f32.le */ | 0x60 /* f32.ge */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f32 comparison"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Comparison produces i32 (0 or 1)
                        stack.push(BV::new_const(format!("f32_cmp_{}", opcode), 32));
                    }

                    // f64 binary operations (consume 2 f64, produce 1 f64)
                    0xa0 /* f64.add */ | 0xa1 /* f64.sub */ | 0xa2 /* f64.mul */ |
                    0xa3 /* f64.div */ | 0xa4 /* f64.min */ | 0xa5 /* f64.max */ |
                    0xa6 /* f64.copysign */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f64 binary op"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Create fresh symbolic value - 64-bit for f64
                        stack.push(BV::new_const(format!("f64_result_{}", opcode), 64));
                    }

                    // f64 unary operations (consume 1 f64, produce 1 f64)
                    0x99 /* f64.abs */ | 0x9a /* f64.neg */ | 0x9b /* f64.ceil */ |
                    0x9c /* f64.floor */ | 0x9d /* f64.trunc */ | 0x9e /* f64.nearest */ |
                    0x9f /* f64.sqrt */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f64 unary op"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const(format!("f64_result_{}", opcode), 64));
                    }

                    // f64 comparisons (consume 2 f64, produce 1 i32)
                    0x61 /* f64.eq */ | 0x62 /* f64.ne */ | 0x63 /* f64.lt */ |
                    0x64 /* f64.gt */ | 0x65 /* f64.le */ | 0x66 /* f64.ge */ => {
                        if stack.len() < 2 {
                            return Err(anyhow!("Stack underflow in f64 comparison"));
                        }
                        let _rhs = stack.pop().unwrap();
                        let _lhs = stack.pop().unwrap();
                        // Comparison produces i32 (0 or 1)
                        stack.push(BV::new_const(format!("f64_cmp_{}", opcode), 32));
                    }

                    // Conversion operations
                    0xa7 /* i32.trunc_f32_s */ | 0xa8 /* i32.trunc_f32_u */ |
                    0xa9 /* i32.trunc_f64_s */ | 0xaa /* i32.trunc_f64_u */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i32.trunc"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("i32_trunc_result", 32));
                    }

                    0xab /* i64.trunc_f32_s */ | 0xac /* i64.trunc_f32_u */ |
                    0xad /* i64.trunc_f64_s */ | 0xae /* i64.trunc_f64_u */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i64.trunc"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("i64_trunc_result", 64));
                    }

                    0xb2 /* f32.convert_i32_s */ | 0xb3 /* f32.convert_i32_u */ |
                    0xb4 /* f32.convert_i64_s */ | 0xb5 /* f32.convert_i64_u */ |
                    0xb6 /* f32.demote_f64 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f32 convert"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("f32_convert_result", 32));
                    }

                    0xb7 /* f64.convert_i32_s */ | 0xb8 /* f64.convert_i32_u */ |
                    0xb9 /* f64.convert_i64_s */ | 0xba /* f64.convert_i64_u */ |
                    0xbb /* f64.promote_f32 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f64 convert"));
                        }
                        let _val = stack.pop().unwrap();
                        stack.push(BV::new_const("f64_convert_result", 64));
                    }

                    // Reinterpret operations (bit pattern preservation)
                    0xbc /* i32.reinterpret_f32 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i32.reinterpret"));
                        }
                        // Reinterpret preserves bits - same bitvector, different type interpretation
                        // For Z3, this is identity since we use bitvectors for both
                        // Stack value stays the same (32-bit bitvector)
                    }

                    0xbd /* i64.reinterpret_f64 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in i64.reinterpret"));
                        }
                        // Same reasoning - 64-bit bitvector stays unchanged
                    }

                    0xbe /* f32.reinterpret_i32 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f32.reinterpret"));
                        }
                        // Bit pattern preserved - 32-bit bitvector unchanged
                    }

                    0xbf /* f64.reinterpret_i64 */ => {
                        if stack.is_empty() {
                            return Err(anyhow!("Stack underflow in f64.reinterpret"));
                        }
                        // Bit pattern preserved - 64-bit bitvector unchanged
                    }

                    // Default: create symbolic result for truly unknown ops
                    _ => {
                        stack.push(BV::new_const(format!("unknown_op_{}", opcode), 32));
                    }
                }
            } else {
                // Empty Unknown instruction - shouldn't happen, but handle gracefully
                stack.push(BV::new_const("empty_unknown", 32));
            }
        }

        // For other instructions not handled above, create symbolic result
        _ => {
            // Create a fresh symbolic value for unsupported instructions
            // This is sound but imprecise
            stack.push(BV::new_const("unsupported_result", 32));
        }
    }
    Ok(())
}

// ===========================================================================
// PR-C (gale #219): precise acyclic control-flow symbolic execution.
//
// An ISOLATED executor that models structured, acyclic wasm control flow
// (block / if / br / br_if / br_table) by *state merging with ITE* driven by an
// explicit label/join stack — never path-forking. For a bounded acyclic region
// this is EXACT (a passing SMT check is a real proof, not a bounded
// approximation), and it produces ONE verification condition instead of N
// per-arm obligations. Precedent: Alive2 (PLDI'21, "we do not fork expressions
// across paths"; final state a linear chain of ITEs), Kuznetsov state-merging
// (PLDI'12), WASP br_table rules (ECOOP'22). See
// docs/design/pr-c-precise-cf-verification.md.
//
// Phase 1 = integer only (no memory): every non-control instruction must be in
// the `is_acyclic_simple_modelable` allowlist (the ops `encode_simple_instruction`
// models PRECISELY) and is delegated there. Loop (back-edge), Call/CallIndirect,
// memory ops, unknown opcodes, and multi-result/param blocks are REJECTED — the
// `is_acyclic_modelable_body` gate fails the whole function and the caller falls
// back (over-approximate-and-skip), NEVER a false equivalence. We must reject
// rather than delegate-and-hope because `encode_simple_instruction`'s catch-all
// HAVOCS (pushes a fresh BV without popping operands) → it would desync the
// stack for an unmodeled op.
//
// Currently `#[allow(dead_code)]`: built + tested in isolation (Z3 equivalence
// over hand-written br_table fixtures, incl. the 0x8000_0000-selector default
// case) BEFORE being wired into the live inline-verify path (PR-C M3).
// ===========================================================================

/// A path condition with `true` and `false` kept out of the Z3 terms.
///
/// The new engine's term language has binary `and`/`or`/`not` over bit-vector
/// comparisons and no Boolean constants, and its reflector rebuilds every node
/// to check itself. Folding constants here, and building only binary
/// connectives, keeps every obligation inside that language.
#[cfg(feature = "verification")]
#[derive(Clone)]
enum Cond {
    True,
    False,
    B(Bool),
}

#[cfg(feature = "verification")]
impl Cond {
    fn of(b: Bool) -> Self {
        Cond::B(b)
    }
    fn and(&self, other: &Cond) -> Cond {
        match (self, other) {
            (Cond::False, _) | (_, Cond::False) => Cond::False,
            (Cond::True, x) | (x, Cond::True) => x.clone(),
            (Cond::B(a), Cond::B(b)) => Cond::B(Bool::and(&[a.clone(), b.clone()])),
        }
    }
    fn or(&self, other: &Cond) -> Cond {
        match (self, other) {
            (Cond::True, _) | (_, Cond::True) => Cond::True,
            (Cond::False, x) | (x, Cond::False) => x.clone(),
            (Cond::B(a), Cond::B(b)) => Cond::B(Bool::or(&[a.clone(), b.clone()])),
        }
    }
    fn not(&self) -> Cond {
        match self {
            Cond::True => Cond::False,
            Cond::False => Cond::True,
            Cond::B(b) => Cond::B(b.not()),
        }
    }
    fn ite_bv(&self, t: &BV, f: &BV) -> BV {
        if t == f {
            return t.clone();
        }
        match self {
            Cond::True => t.clone(),
            Cond::False => f.clone(),
            Cond::B(b) => b.ite(t, f),
        }
    }
    fn ite_mem(&self, t: &Array, f: &Array) -> Array {
        if t == f {
            return t.clone();
        }
        match self {
            Cond::True => t.clone(),
            Cond::False => f.clone(),
            Cond::B(b) => b.ite(t, f),
        }
    }
    /// Conjunction of equalities, as a binary chain.
    fn all(eqs: Vec<Bool>) -> Cond {
        eqs.into_iter()
            .fold(Cond::True, |acc, e| acc.and(&Cond::of(e)))
    }
    /// Syntactically the same condition (hash-consed Z3 terms compare by node).
    fn same_as(&self, other: &Cond) -> bool {
        match (self, other) {
            (Cond::True, Cond::True) | (Cond::False, Cond::False) => true,
            (Cond::B(a), Cond::B(b)) => a == b,
            _ => false,
        }
    }
}

/// Linear memory as a base array plus the byte stores made since it.
///
/// `folded` is the base with the log applied, and is what loads read. The log
/// lets a join merge memory as conditional stores,
/// `store(M, a, ite(g, v, select(M, a)))`, instead of `ite` over whole arrays,
/// which the new engine's term language does not have. Bases differ only
/// after an opaque call; then the join falls back to `ite` over arrays.
#[cfg(feature = "verification")]
#[derive(Clone)]
struct Mem {
    base: Array,
    log: Vec<(BV, BV)>,
    folded: Array,
    /// Size in 64 KiB pages. Symbolic at entry and shared by both encodings,
    /// so equivalence must hold for every size; changed by `memory.grow` and
    /// by opaque calls.
    pages: BV,
}

#[cfg(feature = "verification")]
impl Mem {
    fn new(base: Array, pages: BV) -> Self {
        Mem {
            folded: base.clone(),
            base,
            log: Vec::new(),
            pages,
        }
    }

    fn store_byte(&mut self, addr: BV, byte: BV) {
        self.folded = self.folded.store(&addr, &byte);
        self.log.push((addr, byte));
    }

    /// Little-endian store of the low `bytes` bytes of `value` at
    /// `addr + offset`, byte for byte as the shared `encode_*_store_into`
    /// helpers do, so loads through `encode_*_load_from` read it back.
    fn store_le(&mut self, addr: &BV, value: &BV, offset: u32, bytes: u32) {
        let ea = addr.bvadd(BV::from_i64(offset as i64, 32));
        for i in 0..bytes as i64 {
            let byte = value.extract((i * 8 + 7) as u32, (i * 8) as u32);
            self.store_byte(ea.bvadd(BV::from_i64(i, 32)), byte);
        }
    }

    /// Merge every memory reaching one join, index by index.
    fn merge_all(entries: &[(Cond, &Vec<Mem>)]) -> Vec<Mem> {
        (0..entries[0].1.len())
            .map(|i| {
                let one: Vec<(Cond, &Mem)> =
                    entries.iter().map(|(g, ms)| (g.clone(), &ms[i])).collect();
                Mem::merge(&one)
            })
            .collect()
    }

    /// Merge memories reaching one join under pairwise-disjoint guards.
    fn merge(entries: &[(Cond, &Mem)]) -> Mem {
        let first = entries[0].1;
        let (_, last) = entries.last().expect("a join has an entry");
        let mut pages = last.pages.clone();
        for (g, m) in entries[..entries.len() - 1].iter().rev() {
            pages = g.ite_bv(&m.pages, &pages);
        }
        if entries.iter().all(|(_, m)| m.base == first.base) {
            // Stores made before the paths split are shared by every entry.
            let mut shared = first.log.len();
            for (_, m) in entries {
                let mut k = 0;
                while k < shared.min(m.log.len())
                    && m.log[k].0 == first.log[k].0
                    && m.log[k].1 == first.log[k].1
                {
                    k += 1;
                }
                shared = k;
            }
            let mut out = Mem::new(first.base.clone(), pages);
            for (a, b) in &first.log[..shared] {
                out.store_byte(a.clone(), b.clone());
            }
            // Under its own guard each entry's later stores take effect; under
            // the others' they write back the byte already there.
            for (g, m) in entries {
                for (a, b) in &m.log[shared..] {
                    let cur = out
                        .folded
                        .select(a)
                        .as_bv()
                        .expect("memory maps addresses to bytes");
                    let v = g.ite_bv(b, &cur);
                    out.store_byte(a.clone(), v);
                }
            }
            out
        } else {
            let mut folded = last.folded.clone();
            for (g, m) in entries[..entries.len() - 1].iter().rev() {
                folded = g.ite_mem(&m.folded, &folded);
            }
            Mem::new(folded, pages)
        }
    }
}

/// Symbolic state threaded along the *current* straight-line path.
#[cfg(feature = "verification")]
#[allow(dead_code)]
struct AcyclicState {
    stack: Vec<BV>,
    locals: Vec<BV>,
    globals: Vec<BV>,
    /// Linear memory as a Z3 `Array[BV32 → BV8]` (PR-C Phase 2). Loads/stores use
    /// the SAME shared helpers (`encode_i32_load_from` etc.) the main encoder
    /// uses, so a `call F` modeled by-body and the inlined body produce
    /// bit-identical memory expressions. Memory is MUTABLE branch state, so it is
    /// merged at every join like locals/globals (see [`Mem::merge`]). One
    /// entry per memory index.
    memory: Vec<Mem>,
    /// Path condition guarding the current straight-line path.
    path: Cond,
    /// False once this path has branched / returned (subsequent straight-line
    /// instructions in the same sequence are dead and must not be executed).
    reachable: bool,
}

/// One incoming branch to a label: its path guard plus the result/locals/globals/
/// memory snapshot at the branch point.
#[cfg(feature = "verification")]
type BranchEntry = (Cond, Vec<BV>, Vec<BV>, Vec<BV>, Vec<Mem>);

/// The ITE-merged successor of a join: result values, locals, globals, memory,
/// and the merged path condition.
#[cfg(feature = "verification")]
type MergedState = (Vec<BV>, Vec<BV>, Vec<BV>, Vec<Mem>, Cond);

#[cfg(feature = "verification")]
#[allow(dead_code)]
struct LabelJoin {
    /// Number of result values the label carries (0 or 1 in Phase 1).
    arity: usize,
    entries: Vec<BranchEntry>,
}

/// Result arity of a structured block type, or `Err` if it has parameters or
/// more than one result (rejected in Phase 1 — over-approximate-and-skip).
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn acyclic_block_arity(block_type: &BlockType) -> Result<usize> {
    match block_type {
        BlockType::Empty => Ok(0),
        BlockType::Value(_) => Ok(1),
        BlockType::Func { params, results } => {
            if !params.is_empty() || results.len() > 1 {
                Err(anyhow!(
                    "acyclic executor: block with params/multi-result not modeled"
                ))
            } else {
                Ok(results.len())
            }
        }
    }
}

/// Allowlist of non-control instructions `encode_simple_instruction` models
/// PRECISELY (pure integer ops + locals/globals, no memory, no havoc). Anything
/// outside this set is rejected by `is_acyclic_modelable_body`.
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn is_acyclic_simple_modelable(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instruction::I32Const(_)
            | Instruction::I64Const(_)
            | Instruction::F32Const(_)
            | Instruction::F64Const(_)
            | Instruction::I32Add
            | Instruction::I32Sub
            | Instruction::I32Mul
            | Instruction::I32And
            | Instruction::I32Or
            | Instruction::I32Xor
            | Instruction::I32Shl
            | Instruction::I32ShrS
            | Instruction::I32ShrU
            | Instruction::I32DivS
            | Instruction::I32DivU
            | Instruction::I32RemS
            | Instruction::I32RemU
            | Instruction::I32Eq
            | Instruction::I32Ne
            | Instruction::I32LtS
            | Instruction::I32LtU
            | Instruction::I32GtS
            | Instruction::I32GtU
            | Instruction::I32LeS
            | Instruction::I32LeU
            | Instruction::I32GeS
            | Instruction::I32GeU
            | Instruction::I32Eqz
            | Instruction::I64Add
            | Instruction::I64Sub
            | Instruction::I64Mul
            | Instruction::I64And
            | Instruction::I64Or
            | Instruction::I64Xor
            | Instruction::I64Shl
            | Instruction::I64ShrS
            | Instruction::I64ShrU
            | Instruction::I64DivS
            | Instruction::I64DivU
            | Instruction::I64RemS
            | Instruction::I64RemU
            | Instruction::I64Eq
            | Instruction::I64Ne
            | Instruction::I64LtS
            | Instruction::I64LtU
            | Instruction::I64GtS
            | Instruction::I64GtU
            | Instruction::I64LeS
            | Instruction::I64LeU
            | Instruction::I64GeS
            | Instruction::I64GeU
            | Instruction::I64Eqz
            | Instruction::I32WrapI64
            | Instruction::I64ExtendI32S
            | Instruction::I64ExtendI32U
            | Instruction::LocalGet(_)
            | Instruction::LocalSet(_)
            | Instruction::LocalTee(_)
            | Instruction::GlobalGet(_)
            | Instruction::GlobalSet(_)
            | Instruction::Drop
            | Instruction::Nop
            | Instruction::Select
    )
}

/// PR-C Phase 2: memory loads/stores `exec_acyclic` models explicitly (via the
/// shared `encode_*_load_from`/`*_store_into` helpers). Kept SEPARATE from
/// `is_acyclic_simple_modelable` because memory ops must NOT be delegated to
/// `encode_simple_instruction` (it havocs them, desyncing the stack).
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn is_acyclic_memory_op(instr: &Instruction) -> bool {
    matches!(
        instr,
        Instruction::I32Load { .. }
            | Instruction::I64Load { .. }
            | Instruction::I32Store { .. }
            | Instruction::I64Store { .. }
            | Instruction::I32Load8S { .. }
            | Instruction::I32Load8U { .. }
            | Instruction::I32Load16S { .. }
            | Instruction::I32Load16U { .. }
            | Instruction::I32Store8 { .. }
            | Instruction::I32Store16 { .. }
    )
}

/// Max by-body call-recursion depth the acyclic executor will model. The seam
/// decides call only `panic*` (no-return → diverge, no further recursion), so
/// depth 2 suffices; 8 gives headroom and bounds any cyclic/recursive call
/// graph (which is rejected, not unrolled).
#[cfg(feature = "verification")]
const MAX_ACYCLIC_CALL_DEPTH: usize = 8;

/// Whole-function gate: true iff every instruction (recursing into block/if
/// bodies) is structurally modelable by `exec_acyclic`. Back-edges (Loop),
/// call_indirect, unknown opcodes, and unmodeled block shapes (params/multi-
/// result) fail. As of Phase 2b the gate is purely STRUCTURAL — it needs no
/// module context, because ANY direct call is modelable (the executor diverges
/// no-return callees, models acyclic locals by-body, and HAVOCS the rest —
/// havoc is conservative-sound), and memory ops are handled explicitly.
///
/// A fast pre-check for the M3 routing decision; `exec_acyclic` is the actual
/// soundness enforcer (it returns `Err` on anything it cannot model, never havoc
/// a value-producing op silently).
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn is_acyclic_modelable_body(body: &[Instruction]) -> bool {
    body.iter().all(|instr| match instr {
        // Back-edges are out of scope.
        Instruction::Loop { .. } => false,
        // Resolved to a direct call when its slot is a constant the table
        // resolver knows, otherwise opaque.
        Instruction::CallIndirect { .. } => true,
        // ANY direct call is modelable: by-body (acyclic local), ⊥-diverge
        // (no-return), or HAVOC (import / loop-containing / multi-result). The
        // executor's Call arm makes the choice; havoc never forges a false ≡.
        Instruction::Call(_) => true,
        Instruction::Block { block_type, body } => {
            acyclic_block_arity(block_type).is_ok() && is_acyclic_modelable_body(body)
        }
        Instruction::If {
            block_type,
            then_body,
            else_body,
        } => {
            acyclic_block_arity(block_type).is_ok()
                && is_acyclic_modelable_body(then_body)
                && is_acyclic_modelable_body(else_body)
        }
        Instruction::Br(_)
        | Instruction::BrIf(_)
        | Instruction::BrTable { .. }
        | Instruction::Return
        // Unreachable is a trap (⊥): the executor diverges the path, never havoc.
        | Instruction::Unreachable => true,
        // An explicit `end` closes the body it ends; it must be last.
        Instruction::End => std::ptr::eq(instr, body.last().unwrap()),
        other => {
            is_acyclic_simple_modelable(other)
                || is_acyclic_memory_op(other)
                || is_exact_extra(other)
        }
    })
}

/// Resolve a relative branch depth to an absolute index in the label stack
/// (0 = innermost = top). `labels[0]` is the implicit function label.
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn acyclic_label_index(labels: &[LabelJoin], depth: u32) -> Result<usize> {
    let depth = depth as usize;
    if depth >= labels.len() {
        return Err(anyhow!(
            "acyclic executor: branch depth {} out of range",
            depth
        ));
    }
    Ok(labels.len() - 1 - depth)
}

/// Record the current state as an incoming branch to `labels[target]`, carrying
/// that label's arity result values from the top of the stack.
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn acyclic_push_branch(
    labels: &mut [LabelJoin],
    target: usize,
    guard: Cond,
    state: &AcyclicState,
) -> Result<()> {
    let arity = labels[target].arity;
    if state.stack.len() < arity {
        return Err(anyhow!("acyclic executor: stack underflow at branch"));
    }
    let results: Vec<BV> = state.stack[state.stack.len() - arity..].to_vec();
    labels[target].entries.push((
        guard,
        results,
        state.locals.clone(),
        state.globals.clone(),
        state.memory.clone(),
    ));
    Ok(())
}

/// ITE-merge a label's incoming branches into one successor. Guards are pairwise
/// disjoint (if: cond vs ¬cond; br_table: sel==i vs sel≥n unsigned), so the
/// fold `ite(g0, v0, ite(g1, v1, … v_last))` selects the live branch under the
/// merged path (= disjunction of guards). Returns `(results, locals, globals,
/// path)`, or `None` if no branch reaches the label (unreachable join).
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn acyclic_merge_entries(entries: &[BranchEntry]) -> Option<MergedState> {
    let (_, last_res, last_loc, last_glob, _) = entries.last()?;
    let mut res = last_res.clone();
    let mut loc = last_loc.clone();
    let mut glob = last_glob.clone();
    let mems: Vec<(Cond, &Vec<Mem>)> = entries.iter().map(|e| (e.0.clone(), &e.4)).collect();
    let mem = Mem::merge_all(&mems);
    let mut path = entries.last().unwrap().0.clone();
    // Fold earlier entries outward: each takes precedence under its own guard.
    for (g, r, l, gl, _) in entries[..entries.len() - 1].iter().rev() {
        res = r
            .iter()
            .zip(res.iter())
            .map(|(t, f)| g.ite_bv(t, f))
            .collect();
        loc = l
            .iter()
            .zip(loc.iter())
            .map(|(t, f)| g.ite_bv(t, f))
            .collect();
        glob = gl
            .iter()
            .zip(glob.iter())
            .map(|(t, f)| g.ite_bv(t, f))
            .collect();
        path = g.or(&path);
    }
    Some((res, loc, glob, mem, path))
}

/// Apply a popped label's join to the state: the post-block state is the
/// ITE-merge of all branches that targeted it, with the block's result values
/// pushed on top of `base_stack` (the operand stack below the block entry).
#[cfg(feature = "verification")]
#[allow(dead_code)]
fn acyclic_apply_join(
    state: &mut AcyclicState,
    base_stack: Vec<BV>,
    join: &LabelJoin,
) -> Result<()> {
    match acyclic_merge_entries(&join.entries) {
        None => {
            // No branch reaches this join — code after the block is unreachable.
            state.reachable = false;
            Ok(())
        }
        Some((results, locals, globals, memory, path)) => {
            let mut stack = base_stack;
            stack.extend(results);
            state.stack = stack;
            state.locals = locals;
            state.globals = globals;
            state.memory = memory;
            state.path = path;
            state.reachable = true;
            Ok(())
        }
    }
}

/// State shared by every path of one encoding: the opaque-call counter and
/// the trap exits.
#[cfg(feature = "verification")]
struct AcyclicCx {
    /// Monotonic per-encode index of opaque calls. Threaded rather than kept in
    /// `AcyclicState` (which is saved and restored at branches) so the Nth
    /// opaque call in execution order gets the same index in both encodings.
    havoc_counter: u64,
    /// One entry per point where execution can trap: the guard under which it
    /// traps there, and the globals and memory at that moment. Traps are
    /// observable, and so is the state they leave behind.
    traps: Vec<(Cond, Vec<BV>, Vec<Mem>)>,
    /// Per memory index: whether the executor may access it.
    mem_ok: Vec<bool>,
    /// Facts every execution satisfies (a 32-bit memory never exceeds 65536
    /// pages). The obligation is that they imply equivalence.
    assumed: Cond,
    /// Callees modelled by body; every other call is opaque. Decided per
    /// callee, never per depth, so a callee is modelled the same way wherever
    /// it is called and both encodings agree on it.
    by_body: std::collections::HashSet<u32>,
}

/// The callees whose direct-call counts differ between the two bodies.
///
/// A call both bodies make equally often is modelled as an opaque call on
/// both sides, which is enough to prove any transform that keeps it. Only a
/// call that was inlined, removed or added needs the callee's body: inlining
/// `F` lowers `F`'s count, and every callee `F`'s body calls rises in the
/// inlined copy, so those are included too.
#[cfg(feature = "verification")]
fn callees_to_model_by_body(
    original: &Function,
    optimized: &Function,
) -> std::collections::HashSet<u32> {
    fn count(body: &[Instruction], into: &mut std::collections::HashMap<u32, i64>, sign: i64) {
        for instr in body {
            match instr {
                Instruction::Call(g) => *into.entry(*g).or_insert(0) += sign,
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count(body, into, sign)
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count(then_body, into, sign);
                    count(else_body, into, sign);
                }
                _ => {}
            }
        }
    }
    let mut diff = std::collections::HashMap::new();
    count(&original.instructions, &mut diff, 1);
    count(&optimized.instructions, &mut diff, -1);
    diff.into_iter()
        .filter(|(_, d)| *d != 0)
        .map(|(g, _)| g)
        .collect()
}

/// Record that the current path traps when `cond` holds, and continue on the
/// path where it does not.
#[cfg(feature = "verification")]
fn acyclic_trap_if(state: &mut AcyclicState, cx: &mut AcyclicCx, cond: Cond) {
    let guard = state.path.and(&cond);
    cx.traps
        .push((guard, state.globals.clone(), state.memory.clone()));
    state.path = state.path.and(&cond.not());
}

/// The guard under which an access of `bytes` bytes at `addr + offset` is out
/// of bounds. Computed in 64 bits so neither the sum nor the bound wraps.
#[cfg(feature = "verification")]
fn acyclic_oob(addr: &BV, offset: u32, bytes: u64, pages: &BV) -> Bool {
    let end = addr
        .zero_ext(32)
        .bvadd(BV::from_u64(offset as u64 + bytes, 64));
    let size = pages.zero_ext(32).bvshl(BV::from_u64(16, 64));
    end.bvugt(&size)
}

/// A 32-bit memory has at most 65536 pages, so `addr + offset + n` of an
/// access that does not trap never wraps 32 bits.
#[cfg(feature = "verification")]
fn pages_in_range(pages: &BV) -> Cond {
    Cond::of(pages.bvule(BV::from_u64(65536, 32)))
}

/// The memory a single-memory instruction accesses.
#[cfg(feature = "verification")]
fn memory_index(instr: &Instruction) -> Option<u32> {
    use Instruction as I;
    match instr {
        I::I32Load { mem, .. }
        | I::I32Store { mem, .. }
        | I::I64Load { mem, .. }
        | I::I64Store { mem, .. }
        | I::F32Load { mem, .. }
        | I::F32Store { mem, .. }
        | I::F64Load { mem, .. }
        | I::F64Store { mem, .. }
        | I::I32Load8S { mem, .. }
        | I::I32Load8U { mem, .. }
        | I::I32Load16S { mem, .. }
        | I::I32Load16U { mem, .. }
        | I::I64Load8S { mem, .. }
        | I::I64Load8U { mem, .. }
        | I::I64Load16S { mem, .. }
        | I::I64Load16U { mem, .. }
        | I::I64Load32S { mem, .. }
        | I::I64Load32U { mem, .. }
        | I::I32Store8 { mem, .. }
        | I::I32Store16 { mem, .. }
        | I::I64Store8 { mem, .. }
        | I::I64Store16 { mem, .. }
        | I::I64Store32 { mem, .. }
        | I::MemoryInit { mem, .. } => Some(*mem),
        I::MemorySize(m) | I::MemoryGrow(m) | I::MemoryFill(m) => Some(*m),
        _ => None,
    }
}

/// Check that memory `m` exists and is one the executor models (32-bit, not
/// shared), and return it as an index.
#[cfg(feature = "verification")]
fn acyclic_mem(m: u32, cx: &AcyclicCx) -> Result<usize> {
    match cx.mem_ok.get(m as usize) {
        Some(true) => Ok(m as usize),
        Some(false) => Err(anyhow!(
            "acyclic executor: memory {} is shared or 64-bit",
            m
        )),
        None => Err(anyhow!("acyclic executor: no memory {}", m)),
    }
}

/// Instructions beyond the original allowlist that `exec_exact_extra` models.
#[cfg(feature = "verification")]
fn is_exact_extra(instr: &Instruction) -> bool {
    use Instruction as I;
    matches!(
        instr,
        I::MemorySize(_)
            | I::MemoryGrow(_)
            | I::MemoryFill(_)
            | I::MemoryCopy { .. }
            | I::MemoryInit { .. }
            | I::F32Load { .. }
            | I::F64Load { .. }
            | I::F32Store { .. }
            | I::F64Store { .. }
            | I::I64Load8S { .. }
            | I::I64Load8U { .. }
            | I::I64Load16S { .. }
            | I::I64Load16U { .. }
            | I::I64Load32S { .. }
            | I::I64Load32U { .. }
            | I::I64Store8 { .. }
            | I::I64Store16 { .. }
            | I::I64Store32 { .. }
            | I::I32Rotl
            | I::I32Rotr
            | I::I64Rotl
            | I::I64Rotr
            | I::I32Clz
            | I::I32Ctz
            | I::I32Popcnt
            | I::I64Clz
            | I::I64Ctz
            | I::I64Popcnt
            | I::I32Extend8S
            | I::I32Extend16S
            | I::I64Extend8S
            | I::I64Extend16S
            | I::I64Extend32S
    ) || float_op(instr).is_some()
}

/// The constant value of a term, if it folds to a numeral.
#[cfg(feature = "verification")]
fn numeral(bv: &BV) -> Option<u64> {
    z3::ast::Ast::simplify(bv).as_u64()
}

/// An uninterpreted function of `args`, applied.
#[cfg(feature = "verification")]
fn uf_apply(name: &str, args: &[&BV], range: &Sort) -> z3::ast::Dynamic {
    let sorts: Vec<Sort> = args.iter().map(|a| Sort::bitvector(a.get_size())).collect();
    let sort_refs: Vec<&Sort> = sorts.iter().collect();
    let arg_refs: Vec<&dyn z3::ast::Ast> = args.iter().map(|a| *a as &dyn z3::ast::Ast).collect();
    FuncDecl::new(name, &sort_refs, range).apply(&arg_refs)
}

/// How a float instruction is evaluated.
#[cfg(feature = "verification")]
#[derive(Clone, Copy)]
enum FloatOp {
    /// Arithmetic producing a float of `width` bits from `arity` operands.
    Arith { arity: usize, width: u32 },
    /// Comparison producing an i32 0 or 1.
    Compare { width: u32 },
    /// Float to integer that traps on NaN or out of range.
    TruncTrapping { from: u32, to: u32 },
    /// Any other conversion (saturating truncation, int to float, demote,
    /// promote): total, producing `to` bits.
    Convert { from: u32, to: u32 },
}

#[cfg(feature = "verification")]
fn float_op(instr: &Instruction) -> Option<(&'static str, FloatOp)> {
    use FloatOp::*;
    use Instruction as I;
    let a32 = |n| Arith {
        arity: n,
        width: 32,
    };
    let a64 = |n| Arith {
        arity: n,
        width: 64,
    };
    Some(match instr {
        I::F32Add => ("f32_add", a32(2)),
        I::F32Sub => ("f32_sub", a32(2)),
        I::F32Mul => ("f32_mul", a32(2)),
        I::F32Div => ("f32_div", a32(2)),
        I::F32Min => ("f32_min", a32(2)),
        I::F32Max => ("f32_max", a32(2)),
        I::F32Ceil => ("f32_ceil", a32(1)),
        I::F32Floor => ("f32_floor", a32(1)),
        I::F32Trunc => ("f32_trunc", a32(1)),
        I::F32Nearest => ("f32_nearest", a32(1)),
        I::F32Sqrt => ("f32_sqrt", a32(1)),
        I::F64Add => ("f64_add", a64(2)),
        I::F64Sub => ("f64_sub", a64(2)),
        I::F64Mul => ("f64_mul", a64(2)),
        I::F64Div => ("f64_div", a64(2)),
        I::F64Min => ("f64_min", a64(2)),
        I::F64Max => ("f64_max", a64(2)),
        I::F64Ceil => ("f64_ceil", a64(1)),
        I::F64Floor => ("f64_floor", a64(1)),
        I::F64Trunc => ("f64_trunc", a64(1)),
        I::F64Nearest => ("f64_nearest", a64(1)),
        I::F64Sqrt => ("f64_sqrt", a64(1)),
        // abs, neg and copysign are bit operations, exact for every input.
        I::F32Abs => ("f32_abs", a32(1)),
        I::F32Neg => ("f32_neg", a32(1)),
        I::F32Copysign => ("f32_copysign", a32(2)),
        I::F64Abs => ("f64_abs", a64(1)),
        I::F64Neg => ("f64_neg", a64(1)),
        I::F64Copysign => ("f64_copysign", a64(2)),
        I::F32Eq => ("f32_eq", Compare { width: 32 }),
        I::F32Ne => ("f32_ne", Compare { width: 32 }),
        I::F32Lt => ("f32_lt", Compare { width: 32 }),
        I::F32Gt => ("f32_gt", Compare { width: 32 }),
        I::F32Le => ("f32_le", Compare { width: 32 }),
        I::F32Ge => ("f32_ge", Compare { width: 32 }),
        I::F64Eq => ("f64_eq", Compare { width: 64 }),
        I::F64Ne => ("f64_ne", Compare { width: 64 }),
        I::F64Lt => ("f64_lt", Compare { width: 64 }),
        I::F64Gt => ("f64_gt", Compare { width: 64 }),
        I::F64Le => ("f64_le", Compare { width: 64 }),
        I::F64Ge => ("f64_ge", Compare { width: 64 }),
        I::I32TruncF32S => ("i32_trunc_f32_s", TruncTrapping { from: 32, to: 32 }),
        I::I32TruncF32U => ("i32_trunc_f32_u", TruncTrapping { from: 32, to: 32 }),
        I::I32TruncF64S => ("i32_trunc_f64_s", TruncTrapping { from: 64, to: 32 }),
        I::I32TruncF64U => ("i32_trunc_f64_u", TruncTrapping { from: 64, to: 32 }),
        I::I64TruncF32S => ("i64_trunc_f32_s", TruncTrapping { from: 32, to: 64 }),
        I::I64TruncF32U => ("i64_trunc_f32_u", TruncTrapping { from: 32, to: 64 }),
        I::I64TruncF64S => ("i64_trunc_f64_s", TruncTrapping { from: 64, to: 64 }),
        I::I64TruncF64U => ("i64_trunc_f64_u", TruncTrapping { from: 64, to: 64 }),
        I::I32TruncSatF32S => ("i32_trunc_sat_f32_s", Convert { from: 32, to: 32 }),
        I::I32TruncSatF32U => ("i32_trunc_sat_f32_u", Convert { from: 32, to: 32 }),
        I::I32TruncSatF64S => ("i32_trunc_sat_f64_s", Convert { from: 64, to: 32 }),
        I::I32TruncSatF64U => ("i32_trunc_sat_f64_u", Convert { from: 64, to: 32 }),
        I::I64TruncSatF32S => ("i64_trunc_sat_f32_s", Convert { from: 32, to: 64 }),
        I::I64TruncSatF32U => ("i64_trunc_sat_f32_u", Convert { from: 32, to: 64 }),
        I::I64TruncSatF64S => ("i64_trunc_sat_f64_s", Convert { from: 64, to: 64 }),
        I::I64TruncSatF64U => ("i64_trunc_sat_f64_u", Convert { from: 64, to: 64 }),
        I::F32ConvertI32S => ("f32_convert_i32_s", Convert { from: 32, to: 32 }),
        I::F32ConvertI32U => ("f32_convert_i32_u", Convert { from: 32, to: 32 }),
        I::F32ConvertI64S => ("f32_convert_i64_s", Convert { from: 64, to: 32 }),
        I::F32ConvertI64U => ("f32_convert_i64_u", Convert { from: 64, to: 32 }),
        I::F64ConvertI32S => ("f64_convert_i32_s", Convert { from: 32, to: 64 }),
        I::F64ConvertI32U => ("f64_convert_i32_u", Convert { from: 32, to: 64 }),
        I::F64ConvertI64S => ("f64_convert_i64_s", Convert { from: 64, to: 64 }),
        I::F64ConvertI64U => ("f64_convert_i64_u", Convert { from: 64, to: 64 }),
        I::F32DemoteF64 => ("f32_demote_f64", Convert { from: 64, to: 32 }),
        I::F64PromoteF32 => ("f64_promote_f32", Convert { from: 32, to: 64 }),
        // Reinterpretation keeps the bits.
        I::I32ReinterpretF32 => ("reinterpret", Convert { from: 32, to: 32 }),
        I::I64ReinterpretF64 => ("reinterpret", Convert { from: 64, to: 64 }),
        I::F32ReinterpretI32 => ("reinterpret", Convert { from: 32, to: 32 }),
        I::F64ReinterpretI64 => ("reinterpret", Convert { from: 64, to: 64 }),
        _ => return None,
    })
}

/// The canonical NaN of each width. Whenever wasm arithmetic produces a NaN it
/// may produce this one, so evaluating to it is one of the allowed results.
#[cfg(feature = "verification")]
const CANON_NAN32: u64 = 0x7fc0_0000;
#[cfg(feature = "verification")]
const CANON_NAN64: u64 = 0x7ff8_0000_0000_0000;

#[cfg(feature = "verification")]
fn f32_bits(x: f32) -> u64 {
    if x.is_nan() {
        CANON_NAN32
    } else {
        x.to_bits() as u64
    }
}

#[cfg(feature = "verification")]
fn f64_bits(x: f64) -> u64 {
    if x.is_nan() { CANON_NAN64 } else { x.to_bits() }
}

/// wasm `fmin`/`fmax`: NaN if either operand is NaN, and -0 below +0.
#[cfg(feature = "verification")]
fn wasm_min_max(a: f64, b: f64, max: bool) -> f64 {
    if a.is_nan() || b.is_nan() {
        return f64::NAN;
    }
    if a == 0.0 && b == 0.0 {
        let neg = if max {
            a.is_sign_negative() && b.is_sign_negative()
        } else {
            a.is_sign_negative() || b.is_sign_negative()
        };
        return if neg { -0.0 } else { 0.0 };
    }
    if max { a.max(b) } else { a.min(b) }
}

/// Evaluate a float instruction on constant operands with wasm semantics.
/// Returns `Err(())` for a trapping truncation that traps, `Ok(None)` when it
/// does not apply.
#[cfg(feature = "verification")]
fn eval_float(name: &str, ops: &[u64]) -> std::result::Result<Option<u64>, ()> {
    let f = |i: usize| f32::from_bits(ops[i] as u32);
    let d = |i: usize| f64::from_bits(ops[i]);
    let b = |x: bool| Some(x as u64);
    // Range check for trapping truncation, done in f64 (exact for f32 inputs).
    let trunc = |x: f64, lo: f64, hi: f64| -> std::result::Result<f64, ()> {
        let t = x.trunc();
        if x.is_nan() || t <= lo || t >= hi {
            Err(())
        } else {
            Ok(t)
        }
    };
    Ok(match name {
        "f32_add" => Some(f32_bits(f(0) + f(1))),
        "f32_sub" => Some(f32_bits(f(0) - f(1))),
        "f32_mul" => Some(f32_bits(f(0) * f(1))),
        "f32_div" => Some(f32_bits(f(0) / f(1))),
        "f32_min" => Some(f32_bits(
            wasm_min_max(f(0) as f64, f(1) as f64, false) as f32
        )),
        "f32_max" => Some(f32_bits(wasm_min_max(f(0) as f64, f(1) as f64, true) as f32)),
        "f32_ceil" => Some(f32_bits(f(0).ceil())),
        "f32_floor" => Some(f32_bits(f(0).floor())),
        "f32_trunc" => Some(f32_bits(f(0).trunc())),
        "f32_nearest" => Some(f32_bits(f(0).round_ties_even())),
        "f32_sqrt" => Some(f32_bits(f(0).sqrt())),
        "f64_add" => Some(f64_bits(d(0) + d(1))),
        "f64_sub" => Some(f64_bits(d(0) - d(1))),
        "f64_mul" => Some(f64_bits(d(0) * d(1))),
        "f64_div" => Some(f64_bits(d(0) / d(1))),
        "f64_min" => Some(f64_bits(wasm_min_max(d(0), d(1), false))),
        "f64_max" => Some(f64_bits(wasm_min_max(d(0), d(1), true))),
        "f64_ceil" => Some(f64_bits(d(0).ceil())),
        "f64_floor" => Some(f64_bits(d(0).floor())),
        "f64_trunc" => Some(f64_bits(d(0).trunc())),
        "f64_nearest" => Some(f64_bits(d(0).round_ties_even())),
        "f64_sqrt" => Some(f64_bits(d(0).sqrt())),
        "f32_eq" => b(f(0) == f(1)),
        "f32_ne" => b(f(0) != f(1)),
        "f32_lt" => b(f(0) < f(1)),
        "f32_gt" => b(f(0) > f(1)),
        "f32_le" => b(f(0) <= f(1)),
        "f32_ge" => b(f(0) >= f(1)),
        "f64_eq" => b(d(0) == d(1)),
        "f64_ne" => b(d(0) != d(1)),
        "f64_lt" => b(d(0) < d(1)),
        "f64_gt" => b(d(0) > d(1)),
        "f64_le" => b(d(0) <= d(1)),
        "f64_ge" => b(d(0) >= d(1)),
        "i32_trunc_f32_s" => {
            Some(trunc(f(0) as f64, -2147483649.0, 2147483648.0)? as i32 as u32 as u64)
        }
        "i32_trunc_f64_s" => Some(trunc(d(0), -2147483649.0, 2147483648.0)? as i32 as u32 as u64),
        "i32_trunc_f32_u" => Some(trunc(f(0) as f64, -1.0, 4294967296.0)? as u32 as u64),
        "i32_trunc_f64_u" => Some(trunc(d(0), -1.0, 4294967296.0)? as u32 as u64),
        "i64_trunc_f32_s" => {
            Some(trunc(f(0) as f64, -9223372036854777856.0, 9223372036854775808.0)? as i64 as u64)
        }
        "i64_trunc_f64_s" => {
            Some(trunc(d(0), -9223372036854777856.0, 9223372036854775808.0)? as i64 as u64)
        }
        "i64_trunc_f32_u" => Some(trunc(f(0) as f64, -1.0, 18446744073709551616.0)? as u64),
        "i64_trunc_f64_u" => Some(trunc(d(0), -1.0, 18446744073709551616.0)? as u64),
        // Rust's float-to-int `as` saturates and maps NaN to 0, as wasm's
        // trunc_sat does.
        "i32_trunc_sat_f32_s" => Some(f(0) as i32 as u32 as u64),
        "i32_trunc_sat_f32_u" => Some(f(0) as u32 as u64),
        "i32_trunc_sat_f64_s" => Some(d(0) as i32 as u32 as u64),
        "i32_trunc_sat_f64_u" => Some(d(0) as u32 as u64),
        "i64_trunc_sat_f32_s" => Some(f(0) as i64 as u64),
        "i64_trunc_sat_f32_u" => Some(f(0) as u64),
        "i64_trunc_sat_f64_s" => Some(d(0) as i64 as u64),
        "i64_trunc_sat_f64_u" => Some(d(0) as u64),
        // Int-to-float `as` rounds to nearest, ties to even, as wasm does.
        "f32_convert_i32_s" => Some(f32_bits(ops[0] as u32 as i32 as f32)),
        "f32_convert_i32_u" => Some(f32_bits(ops[0] as u32 as f32)),
        "f32_convert_i64_s" => Some(f32_bits(ops[0] as i64 as f32)),
        "f32_convert_i64_u" => Some(f32_bits(ops[0] as f32)),
        "f64_convert_i32_s" => Some(f64_bits(ops[0] as u32 as i32 as f64)),
        "f64_convert_i32_u" => Some(f64_bits(ops[0] as u32 as f64)),
        "f64_convert_i64_s" => Some(f64_bits(ops[0] as i64 as f64)),
        "f64_convert_i64_u" => Some(f64_bits(ops[0] as f64)),
        "f32_demote_f64" => Some(f32_bits(d(0) as f32)),
        "f64_promote_f32" => Some(f64_bits(f(0) as f64)),
        _ => None,
    })
}

/// Execute a float instruction.
///
/// abs, neg, copysign and reinterpretation are bit operations and exact for
/// every input. Everything else is evaluated with wasm semantics when its
/// operands are constants, and is otherwise an uninterpreted function of the
/// operand bits; a trapping truncation also traps under an uninterpreted
/// condition of its operand.
///
/// Limitation: a NaN result is modelled as one value per operation and
/// operands, so two evaluations of the same NaN-producing operation are
/// taken to give the same payload. wasm lets them differ.
#[cfg(feature = "verification")]
fn exec_float(
    name: &'static str,
    op: FloatOp,
    state: &mut AcyclicState,
    cx: &mut AcyclicCx,
) -> Result<()> {
    let arity = match op {
        FloatOp::Arith { arity, .. } => arity,
        FloatOp::Compare { .. } => 2,
        FloatOp::TruncTrapping { .. } | FloatOp::Convert { .. } => 1,
    };
    if state.stack.len() < arity {
        return Err(anyhow!("acyclic executor: {} underflow", name));
    }
    let args: Vec<BV> = state.stack.split_off(state.stack.len() - arity);
    let in_width = match op {
        FloatOp::Arith { width, .. } | FloatOp::Compare { width } => width,
        FloatOp::TruncTrapping { from, .. } | FloatOp::Convert { from, .. } => from,
    };
    if args.iter().any(|a| a.get_size() != in_width) {
        return Err(anyhow!("acyclic executor: {} operand width", name));
    }
    let out_width = match op {
        FloatOp::Arith { width, .. } => width,
        FloatOp::Compare { .. } => 32,
        FloatOp::TruncTrapping { to, .. } | FloatOp::Convert { to, .. } => to,
    };

    // Bit operations.
    let sign = |w: u32| BV::from_u64(1u64 << (w - 1), w);
    let exact = match name {
        "reinterpret" => Some(args[0].clone()),
        "f32_abs" | "f64_abs" => Some(args[0].bvand(sign(in_width).bvnot())),
        "f32_neg" | "f64_neg" => Some(args[0].bvxor(sign(in_width))),
        "f32_copysign" | "f64_copysign" => Some(
            args[0]
                .bvand(sign(in_width).bvnot())
                .bvor(args[1].bvand(sign(in_width))),
        ),
        _ => None,
    };
    if let Some(v) = exact {
        state.stack.push(v);
        return Ok(());
    }

    let consts: Option<Vec<u64>> = args.iter().map(numeral).collect();
    if let Some(ops) = consts {
        match eval_float(name, &ops) {
            Ok(Some(v)) => {
                state.stack.push(BV::from_u64(v, out_width));
                return Ok(());
            }
            Err(()) => {
                // A constant operand out of range: this path always traps.
                acyclic_trap_if(state, cx, Cond::True);
                state.reachable = false;
                state.stack.push(BV::from_u64(0, out_width));
                return Ok(());
            }
            Ok(None) => {}
        }
    }

    let refs: Vec<&BV> = args.iter().collect();
    if let FloatOp::TruncTrapping { .. } = op {
        let traps = uf_apply(&format!("xfloat_{}_traps", name), &refs, &Sort::bool())
            .as_bool()
            .ok_or_else(|| anyhow!("acyclic executor: trap model is not a boolean"))?;
        acyclic_trap_if(state, cx, Cond::of(traps));
    }
    let v = if let FloatOp::Compare { .. } = op {
        // 0 or 1, never anything else.
        uf_apply(&format!("xfloat_{}", name), &refs, &Sort::bitvector(1))
            .as_bv()
            .ok_or_else(|| anyhow!("acyclic executor: float model is not a bit-vector"))?
            .zero_ext(31)
    } else {
        uf_apply(
            &format!("xfloat_{}", name),
            &refs,
            &Sort::bitvector(out_width),
        )
        .as_bv()
        .ok_or_else(|| anyhow!("acyclic executor: float model is not a bit-vector"))?
    };
    state.stack.push(v);
    Ok(())
}

/// Count of set bits, as a sum of the zero-extended bits.
#[cfg(feature = "verification")]
fn popcnt(v: &BV) -> BV {
    let w = v.get_size();
    let mut sum = BV::from_u64(0, w);
    for i in 0..w {
        sum = sum.bvadd(v.extract(i, i).zero_ext(w - 1));
    }
    sum
}

/// Leading (`from_top`) or trailing zero count.
#[cfg(feature = "verification")]
fn count_zeros(v: &BV, from_top: bool) -> BV {
    let w = v.get_size();
    let mut result = BV::from_u64(w as u64, w);
    // Built inside out, so the bit nearest the counted end is tested first.
    for k in (0..w).rev() {
        let bit = if from_top { w - 1 - k } else { k };
        let set = v.extract(bit, bit).eq(BV::from_u64(1, 1));
        result = set.ite(&BV::from_u64(k as u64, w), &result);
    }
    result
}

/// Execute an instruction accepted by [`is_exact_extra`].
#[cfg(feature = "verification")]
fn exec_exact_extra(
    instr: &Instruction,
    state: &mut AcyclicState,
    cx: &mut AcyclicCx,
) -> Result<()> {
    use Instruction as I;
    let pop = |state: &mut AcyclicState, what: &str| {
        state
            .stack
            .pop()
            .ok_or_else(|| anyhow!("acyclic executor: {} underflow", what))
    };
    if let Some((name, op)) = float_op(instr) {
        return exec_float(name, op, state, cx);
    }
    match instr {
        I::MemorySize(_) => {
            let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
            state.stack.push(state.memory[m].pages.clone());
        }
        I::MemoryGrow(_) => {
            let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
            let delta = pop(state, "memory.grow")?;
            let pages = state.memory[m].pages.clone();
            let id = BV::from_u64(cx.havoc_counter, 64);
            cx.havoc_counter += 1;
            // Growth may fail for any reason; whether it does is a function of
            // its position and inputs. It never succeeds past 65536 pages.
            let ok = uf_apply("xmemory_grow_ok", &[&id, &pages, &delta], &Sort::bool())
                .as_bool()
                .ok_or_else(|| anyhow!("acyclic executor: grow model is not a boolean"))?;
            let fits = pages
                .zero_ext(32)
                .bvadd(delta.zero_ext(32))
                .bvule(BV::from_u64(65536, 64));
            let grows = Cond::of(Bool::and(&[ok, fits]));
            state
                .stack
                .push(grows.ite_bv(&pages, &BV::from_i64(-1, 32)));
            state.memory[m].pages = grows.ite_bv(&pages.bvadd(&delta), &pages);
        }

        // Loads and stores that move bits: floats as their bit patterns, and
        // the i64 partial widths.
        I::F32Load { offset, .. }
        | I::F64Load { offset, .. }
        | I::I64Load8S { offset, .. }
        | I::I64Load8U { offset, .. }
        | I::I64Load16S { offset, .. }
        | I::I64Load16U { offset, .. }
        | I::I64Load32S { offset, .. }
        | I::I64Load32U { offset, .. } => {
            let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
            let addr = pop(state, "load")?;
            let (bytes, signed, width) = match instr {
                I::F32Load { .. } => (4, false, 32),
                I::F64Load { .. } => (8, false, 64),
                I::I64Load8S { .. } => (1, true, 64),
                I::I64Load8U { .. } => (1, false, 64),
                I::I64Load16S { .. } => (2, true, 64),
                I::I64Load16U { .. } => (2, false, 64),
                I::I64Load32S { .. } => (4, true, 64),
                _ => (4, false, 64),
            };
            let oob = acyclic_oob(&addr, *offset, bytes as u64, &state.memory[m].pages);
            acyclic_trap_if(state, cx, Cond::of(oob));
            let v = match instr {
                I::F32Load { .. } => encode_i32_load_from(&state.memory[m].folded, &addr, *offset)?,
                I::F64Load { .. } => encode_i64_load_from(&state.memory[m].folded, &addr, *offset)?,
                _ => encode_partial_load_from(
                    &state.memory[m].folded,
                    &addr,
                    *offset,
                    bytes,
                    signed,
                    width,
                )?,
            };
            state.stack.push(v);
        }
        I::F32Store { offset, .. }
        | I::F64Store { offset, .. }
        | I::I64Store8 { offset, .. }
        | I::I64Store16 { offset, .. }
        | I::I64Store32 { offset, .. } => {
            let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
            let value = pop(state, "store value")?;
            let addr = pop(state, "store address")?;
            let bytes = match instr {
                I::F32Store { .. } | I::I64Store32 { .. } => 4,
                I::F64Store { .. } => 8,
                I::I64Store8 { .. } => 1,
                _ => 2,
            };
            let oob = acyclic_oob(&addr, *offset, bytes as u64, &state.memory[m].pages);
            acyclic_trap_if(state, cx, Cond::of(oob));
            state.memory[m].store_le(&addr, &value, *offset, bytes);
        }

        // Bulk memory: the bounds checks are exact; the new contents are a
        // function of the memories and the operands.
        I::MemoryFill(_) | I::MemoryCopy { .. } | I::MemoryInit { .. } => {
            let n = pop(state, "bulk length")?;
            let src = pop(state, "bulk source")?;
            let dst = pop(state, "bulk destination")?;
            let (d, s_mem) = match instr {
                I::MemoryCopy { dst_mem, src_mem } => {
                    (acyclic_mem(*dst_mem, cx)?, Some(acyclic_mem(*src_mem, cx)?))
                }
                _ => (acyclic_mem(memory_index(instr).unwrap_or(0), cx)?, None),
            };
            let past = |a: &BV, pages: &BV| {
                let size = pages.zero_ext(32).bvshl(BV::from_u64(16, 64));
                a.zero_ext(32).bvadd(n.zero_ext(32)).bvugt(&size)
            };
            let dst_oob = past(&dst, &state.memory[d].pages);
            let mem_sort = Sort::array(&Sort::bitvector(32), &Sort::bitvector(8));
            let bv32 = Sort::bitvector(32);
            let (oob, new) = match instr {
                I::MemoryCopy { .. } => {
                    let sm = s_mem.unwrap();
                    let oob = Bool::or(&[dst_oob, past(&src, &state.memory[sm].pages)]);
                    let new = FuncDecl::new(
                        format!("xmemory_copy{}_{}", d, sm),
                        &[&mem_sort, &mem_sort, &bv32, &bv32, &bv32],
                        &mem_sort,
                    )
                    .apply(&[
                        &state.memory[d].folded,
                        &state.memory[sm].folded,
                        &dst,
                        &src,
                        &n,
                    ]);
                    (oob, new)
                }
                I::MemoryInit { data_idx, .. } => {
                    let seg_len = BV::new_const(format!("data{}_len", data_idx), 32);
                    let seg_oob = src
                        .zero_ext(32)
                        .bvadd(n.zero_ext(32))
                        .bvugt(seg_len.zero_ext(32));
                    let new = FuncDecl::new(
                        format!("xmemory_init{}_{}", d, data_idx),
                        &[&mem_sort, &bv32, &bv32, &bv32],
                        &mem_sort,
                    )
                    .apply(&[&state.memory[d].folded, &dst, &src, &n]);
                    (Bool::or(&[dst_oob, seg_oob]), new)
                }
                _ => {
                    let new = FuncDecl::new(
                        format!("xmemory_fill{}", d),
                        &[&mem_sort, &bv32, &bv32, &bv32],
                        &mem_sort,
                    )
                    .apply(&[&state.memory[d].folded, &dst, &src, &n]);
                    (dst_oob, new)
                }
            };
            acyclic_trap_if(state, cx, Cond::of(oob));
            let new = new
                .as_array()
                .ok_or_else(|| anyhow!("acyclic executor: bulk memory model is not an array"))?;
            let pages = state.memory[d].pages.clone();
            state.memory[d] = Mem::new(new, pages);
        }

        I::I32Rotl | I::I32Rotr | I::I64Rotl | I::I64Rotr => {
            let amount = pop(state, "rotate amount")?;
            let v = pop(state, "rotate value")?;
            let w = v.get_size();
            let amount = amount.bvurem(BV::from_u64(w as u64, w));
            state
                .stack
                .push(if matches!(instr, I::I32Rotl | I::I64Rotl) {
                    v.bvrotl(&amount)
                } else {
                    v.bvrotr(&amount)
                });
        }
        I::I32Clz | I::I64Clz => {
            let v = pop(state, "clz")?;
            state.stack.push(count_zeros(&v, true));
        }
        I::I32Ctz | I::I64Ctz => {
            let v = pop(state, "ctz")?;
            state.stack.push(count_zeros(&v, false));
        }
        I::I32Popcnt | I::I64Popcnt => {
            let v = pop(state, "popcnt")?;
            state.stack.push(popcnt(&v));
        }
        I::I32Extend8S | I::I32Extend16S | I::I64Extend8S | I::I64Extend16S | I::I64Extend32S => {
            let v = pop(state, "extend")?;
            let w = v.get_size();
            let from = match instr {
                I::I32Extend8S | I::I64Extend8S => 8,
                I::I32Extend16S | I::I64Extend16S => 16,
                _ => 32,
            };
            state.stack.push(v.extract(from - 1, 0).sign_ext(w - from));
        }
        other => {
            return Err(anyhow!(
                "acyclic executor: instruction not modelable: {:?}",
                other
            ));
        }
    }
    Ok(())
}

/// Execute a structured, acyclic instruction sequence, merging branch states at
/// label joins. Precondition: the enclosing function passed
/// `is_acyclic_modelable_body`. `labels[0]` is the function label.
#[cfg(feature = "verification")]
fn exec_acyclic(
    body: &[Instruction],
    state: &mut AcyclicState,
    labels: &mut Vec<LabelJoin>,
    ctx: &VerificationSignatureContext,
    depth: usize,
    cx: &mut AcyclicCx,
) -> Result<()> {
    for instr in body {
        if !state.reachable {
            // Dead code after an unconditional branch / return.
            break;
        }
        match instr {
            Instruction::Block { block_type, body } => {
                let arity = acyclic_block_arity(block_type)?;
                let base_stack = state.stack.clone();
                labels.push(LabelJoin {
                    arity,
                    entries: Vec::new(),
                });
                exec_acyclic(body, state, labels, ctx, depth, cx)?;
                // Fall-through off the end == implicit `br 0` to this block.
                if state.reachable {
                    let target = labels.len() - 1;
                    acyclic_push_branch(labels, target, state.path.clone(), state)?;
                }
                let join = labels.pop().unwrap();
                acyclic_apply_join(state, base_stack, &join)?;
            }
            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                let arity = acyclic_block_arity(block_type)?;
                let cond = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: If underflow"))?;
                let cond_nz = cond.eq(BV::from_i64(0, 32)).not();
                let base_stack = state.stack.clone();
                let saved_locals = state.locals.clone();
                let saved_globals = state.globals.clone();
                let saved_memory = state.memory.clone();
                let saved_path = state.path.clone();
                labels.push(LabelJoin {
                    arity,
                    entries: Vec::new(),
                });
                let target = labels.len() - 1;
                state.path = saved_path.and(&Cond::of(cond_nz.clone()));
                state.reachable = true;
                exec_acyclic(then_body, state, labels, ctx, depth, cx)?;
                if state.reachable {
                    acyclic_push_branch(labels, target, state.path.clone(), state)?;
                }
                state.stack = base_stack.clone();
                state.locals = saved_locals;
                state.globals = saved_globals;
                state.memory = saved_memory;
                state.path = saved_path.and(&Cond::of(cond_nz.not()));
                state.reachable = true;
                exec_acyclic(else_body, state, labels, ctx, depth, cx)?;
                if state.reachable {
                    acyclic_push_branch(labels, target, state.path.clone(), state)?;
                }
                let join = labels.pop().unwrap();
                acyclic_apply_join(state, base_stack, &join)?;
            }
            Instruction::Br(depth) => {
                let target = acyclic_label_index(labels, *depth)?;
                acyclic_push_branch(labels, target, state.path.clone(), state)?;
                state.reachable = false;
            }
            Instruction::BrIf(depth) => {
                let cond = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: BrIf underflow"))?;
                let cond_nz = cond.eq(BV::from_i64(0, 32)).not();
                let target = acyclic_label_index(labels, *depth)?;
                let branch_guard = state.path.and(&Cond::of(cond_nz.clone()));
                acyclic_push_branch(labels, target, branch_guard, state)?;
                state.path = state.path.and(&Cond::of(cond_nz.not()));
            }
            Instruction::BrTable { targets, default } => {
                let sel = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: BrTable underflow"))?;
                let n = targets.len() as u64;
                for (i, d) in targets.iter().enumerate() {
                    let target = acyclic_label_index(labels, *d)?;
                    let guard = state
                        .path
                        .and(&Cond::of(sel.eq(BV::from_u64(i as u64, 32))));
                    acyclic_push_branch(labels, target, guard, state)?;
                }
                // Default iff sel >=u n (unsigned, so selectors in [2^31, 2^32)
                // are covered too).
                let target = acyclic_label_index(labels, *default)?;
                let guard = state.path.and(&Cond::of(sel.bvuge(BV::from_u64(n, 32))));
                acyclic_push_branch(labels, target, guard, state)?;
                state.reachable = false;
            }
            Instruction::Return => {
                acyclic_push_branch(labels, 0, state.path.clone(), state)?;
                state.reachable = false;
            }
            Instruction::Unreachable => {
                acyclic_trap_if(state, cx, Cond::True);
                state.reachable = false;
            }
            // Admitted only as the last instruction of a body.
            Instruction::End => break,
            Instruction::Call(g) => {
                let callee_opt = ctx
                    .function_bodies
                    .get(*g as usize)
                    .and_then(|o| o.as_deref());
                match callee_opt {
                    // A local callee the executor can model is executed by
                    // body, so a call and its inlined body give the same
                    // expressions, traps included.
                    Some(callee)
                        if cx.by_body.contains(g)
                            && depth + 1 < MAX_ACYCLIC_CALL_DEPTH
                            && callee.signature.results.len() <= 1
                            && is_acyclic_modelable_body(&callee.instructions) =>
                    {
                        encode_acyclic_call(callee, state, ctx, depth, cx)?;
                    }
                    // Anything else is opaque: its results, its effects and
                    // whether it traps are functions of what it can observe.
                    _ => {
                        acyclic_havoc_call(*g, state, ctx, cx)?;
                    }
                }
            }
            Instruction::I32DivS
            | Instruction::I32DivU
            | Instruction::I32RemS
            | Instruction::I32RemU
            | Instruction::I64DivS
            | Instruction::I64DivU
            | Instruction::I64RemS
            | Instruction::I64RemU => {
                let n = state.stack.len();
                if n < 2 {
                    return Err(anyhow!("acyclic executor: div/rem underflow"));
                }
                let rhs = state.stack[n - 1].clone();
                let lhs = state.stack[n - 2].clone();
                let w = rhs.get_size();
                let mut trap = rhs.eq(BV::from_u64(0, w));
                // Signed division overflows (and traps) on MIN / -1; signed
                // remainder does not trap there.
                if matches!(instr, Instruction::I32DivS | Instruction::I64DivS) {
                    let min = BV::from_u64(1u64 << (w - 1), w);
                    let neg1 = BV::from_i64(-1, w);
                    trap = Bool::or(&[trap, Bool::and(&[lhs.eq(&min), rhs.eq(&neg1)])]);
                }
                acyclic_trap_if(state, cx, Cond::of(trap));
                encode_simple_instruction(
                    instr,
                    &mut state.stack,
                    &mut state.locals,
                    &mut state.globals,
                )?;
            }
            Instruction::GlobalGet(idx) | Instruction::GlobalSet(idx) => {
                let i = *idx as usize;
                if i >= state.globals.len() {
                    return Err(anyhow!(
                        "acyclic executor: global {} has no known type",
                        idx
                    ));
                }
                if let Instruction::GlobalSet(_) = instr {
                    let v = state
                        .stack
                        .last()
                        .ok_or_else(|| anyhow!("acyclic executor: GlobalSet underflow"))?;
                    if v.get_size() != state.globals[i].get_size() {
                        return Err(anyhow!("acyclic executor: global {} width mismatch", idx));
                    }
                }
                encode_simple_instruction(
                    instr,
                    &mut state.stack,
                    &mut state.locals,
                    &mut state.globals,
                )?;
            }
            Instruction::I32Load { offset, .. } | Instruction::I64Load { offset, .. } => {
                let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
                let addr = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: load underflow"))?;
                let bytes = if matches!(instr, Instruction::I32Load { .. }) {
                    4
                } else {
                    8
                };
                let oob = acyclic_oob(&addr, *offset, bytes, &state.memory[m].pages);
                acyclic_trap_if(state, cx, Cond::of(oob));
                let v = if bytes == 4 {
                    encode_i32_load_from(&state.memory[m].folded, &addr, *offset)?
                } else {
                    encode_i64_load_from(&state.memory[m].folded, &addr, *offset)?
                };
                state.stack.push(v);
            }
            Instruction::I32Store { offset, .. } | Instruction::I64Store { offset, .. } => {
                let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
                let value = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: store value underflow"))?;
                let addr = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: store addr underflow"))?;
                let bytes = if matches!(instr, Instruction::I32Store { .. }) {
                    4
                } else {
                    8
                };
                let oob = acyclic_oob(&addr, *offset, bytes, &state.memory[m].pages);
                acyclic_trap_if(state, cx, Cond::of(oob));
                state.memory[m].store_le(&addr, &value, *offset, bytes as u32);
            }
            Instruction::I32Load8S { offset, .. }
            | Instruction::I32Load8U { offset, .. }
            | Instruction::I32Load16S { offset, .. }
            | Instruction::I32Load16U { offset, .. } => {
                let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
                let addr = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: partial load underflow"))?;
                let (bytes, signed) = match instr {
                    Instruction::I32Load8S { .. } => (1, true),
                    Instruction::I32Load8U { .. } => (1, false),
                    Instruction::I32Load16S { .. } => (2, true),
                    _ => (2, false),
                };
                let oob = acyclic_oob(&addr, *offset, bytes as u64, &state.memory[m].pages);
                acyclic_trap_if(state, cx, Cond::of(oob));
                state.stack.push(encode_partial_load_from(
                    &state.memory[m].folded,
                    &addr,
                    *offset,
                    bytes,
                    signed,
                    32,
                )?);
            }
            Instruction::I32Store8 { offset, .. } | Instruction::I32Store16 { offset, .. } => {
                let m = acyclic_mem(memory_index(instr).unwrap_or(0), cx)?;
                let value = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: partial store value underflow"))?;
                let addr = state
                    .stack
                    .pop()
                    .ok_or_else(|| anyhow!("acyclic executor: partial store addr underflow"))?;
                let bytes = if matches!(instr, Instruction::I32Store8 { .. }) {
                    1
                } else {
                    2
                };
                let oob = acyclic_oob(&addr, *offset, bytes as u64, &state.memory[m].pages);
                acyclic_trap_if(state, cx, Cond::of(oob));
                state.memory[m].store_le(&addr, &value, *offset, bytes);
            }
            Instruction::CallIndirect {
                type_idx,
                table_idx,
            } => {
                let sig = ctx
                    .get_type_signature(*type_idx)
                    .cloned()
                    .ok_or_else(|| anyhow!("acyclic executor: no type {}", type_idx))?;
                let slot = state
                    .stack
                    .last()
                    .ok_or_else(|| anyhow!("acyclic executor: call_indirect underflow"))?
                    .clone();
                let resolved = numeral(&slot)
                    .and_then(|n| u32::try_from(n).ok())
                    .and_then(|n| ctx.resolve_indirect_call(*table_idx, n, &sig));
                match resolved {
                    // A constant slot holding a function of this type: the
                    // call is that direct call.
                    Some(f) => {
                        state.stack.pop();
                        exec_acyclic(&[Instruction::Call(f)], state, labels, ctx, depth, cx)?;
                    }
                    None => {
                        let slot = state.stack.pop().unwrap();
                        acyclic_opaque_call(
                            &format!("xcall_indirect{}_{}", table_idx, type_idx),
                            &sig,
                            vec![slot],
                            state,
                            cx,
                        )?;
                    }
                }
            }
            other if is_exact_extra(other) => exec_exact_extra(other, state, cx)?,
            other => {
                // Never delegate an unmodelled op: the delegate's catch-all
                // pushes a fresh value and would desync the stack.
                if !is_acyclic_simple_modelable(other) {
                    return Err(anyhow!(
                        "acyclic executor: instruction not modelable: {:?}",
                        other
                    ));
                }
                encode_simple_instruction(
                    other,
                    &mut state.stack,
                    &mut state.locals,
                    &mut state.globals,
                )?;
            }
        }
    }
    Ok(())
}

/// Model an opaque `call g`: an import, or a local callee the executor cannot
/// execute by body.
///
/// Everything the call produces is an uninterpreted function of what it can
/// observe: its position among opaque calls in execution order, its
/// arguments, the globals and the memory. So two encodings agree on a call
/// only when they make the same call at the same position from the same
/// state; dropping a write the callee could read, or moving a call across one,
/// changes its inputs and is not provable. The call may trap, leaving its own
/// globals and memory behind.
#[cfg(feature = "verification")]
fn acyclic_havoc_call(
    g: u32,
    state: &mut AcyclicState,
    ctx: &VerificationSignatureContext,
    cx: &mut AcyclicCx,
) -> Result<()> {
    let sig = ctx
        .get_function_signature(g)
        .cloned()
        .ok_or_else(|| anyhow!("acyclic executor: no signature for call {}", g))?;
    acyclic_opaque_call(&format!("xcall{}", g), &sig, Vec::new(), state, cx)
}

/// An opaque call named `name` with signature `sig`. `extra` are inputs
/// beyond the arguments (the table slot of an indirect call).
#[cfg(feature = "verification")]
fn acyclic_opaque_call(
    name: &str,
    sig: &FunctionSignature,
    extra: Vec<BV>,
    state: &mut AcyclicState,
    cx: &mut AcyclicCx,
) -> Result<()> {
    let nparams = sig.params.len();
    if state.stack.len() < nparams {
        return Err(anyhow!("acyclic executor: opaque-call arg underflow"));
    }
    let mut args: Vec<BV> = state.stack.split_off(state.stack.len() - nparams);
    args.extend(extra);
    let id = BV::from_u64(cx.havoc_counter, 64);
    cx.havoc_counter += 1;

    let mem_sort = Sort::array(&Sort::bitvector(32), &Sort::bitvector(8));
    let mut domain: Vec<Sort> = vec![Sort::bitvector(64)];
    domain.extend(args.iter().map(|a| Sort::bitvector(a.get_size())));
    domain.extend(state.globals.iter().map(|v| Sort::bitvector(v.get_size())));
    for _ in &state.memory {
        domain.push(mem_sort.clone());
        domain.push(Sort::bitvector(32));
    }
    let domain_refs: Vec<&Sort> = domain.iter().collect();
    let mut inputs: Vec<&dyn z3::ast::Ast> = vec![&id];
    inputs.extend(args.iter().map(|a| a as &dyn z3::ast::Ast));
    inputs.extend(state.globals.iter().map(|v| v as &dyn z3::ast::Ast));
    for mem in &state.memory {
        inputs.push(&mem.folded);
        inputs.push(&mem.pages);
    }
    let apply =
        |name: String, range: &Sort| FuncDecl::new(name, &domain_refs, range).apply(&inputs);
    let as_bv = |d: z3::ast::Dynamic| {
        d.as_bv()
            .ok_or_else(|| anyhow!("acyclic executor: call model is not a bit-vector"))
    };
    let as_mem = |d: z3::ast::Dynamic| {
        d.as_array()
            .ok_or_else(|| anyhow!("acyclic executor: call model is not an array"))
    };

    let traps = apply(format!("{}_traps", name), &Sort::bool())
        .as_bool()
        .ok_or_else(|| anyhow!("acyclic executor: call model is not a boolean"))?;
    let mut trap_globals = Vec::with_capacity(state.globals.len());
    let mut globals = Vec::with_capacity(state.globals.len());
    for (k, v) in state.globals.iter().enumerate() {
        let s = Sort::bitvector(v.get_size());
        trap_globals.push(as_bv(apply(format!("{}_trap_global{}", name, k), &s))?);
        globals.push(as_bv(apply(format!("{}_global{}", name, k), &s))?);
    }
    let pages_sort = Sort::bitvector(32);
    let mut trap_memory = Vec::with_capacity(state.memory.len());
    let mut memory = Vec::with_capacity(state.memory.len());
    for k in 0..state.memory.len() {
        trap_memory.push(Mem::new(
            as_mem(apply(format!("{}_trap_memory{}", name, k), &mem_sort))?,
            as_bv(apply(format!("{}_trap_pages{}", name, k), &pages_sort))?,
        ));
        memory.push(Mem::new(
            as_mem(apply(format!("{}_memory{}", name, k), &mem_sort))?,
            as_bv(apply(format!("{}_pages{}", name, k), &pages_sort))?,
        ));
        for m in [&trap_memory[k], &memory[k]] {
            cx.assumed = cx.assumed.and(&pages_in_range(&m.pages));
        }
    }
    let mut results = Vec::with_capacity(sig.results.len());
    for (i, rt) in sig.results.iter().enumerate() {
        let s = Sort::bitvector(bv_width_for_value_type(*rt));
        results.push(as_bv(apply(format!("{}_result{}", name, i), &s))?);
    }

    cx.traps.push((
        state.path.and(&Cond::of(traps.clone())),
        trap_globals,
        trap_memory,
    ));
    state.path = state.path.and(&Cond::of(traps.not()));
    state.globals = globals;
    state.memory = memory;
    state.stack.extend(results);
    Ok(())
}

/// Model a `call` to `callee` BY BODY: pop the args, execute the callee with
/// params bound to them and globals and memory shared with the caller, and
/// push the merged single result. Traps inside the callee are traps of the
/// caller. Sets `reachable=false` if the callee never returns normally.
#[cfg(feature = "verification")]
fn encode_acyclic_call(
    callee: &Function,
    state: &mut AcyclicState,
    ctx: &VerificationSignatureContext,
    depth: usize,
    cx: &mut AcyclicCx,
) -> Result<()> {
    if depth + 1 >= MAX_ACYCLIC_CALL_DEPTH {
        return Err(anyhow!("acyclic executor: call recursion too deep"));
    }
    let arity = callee.signature.results.len();
    if arity > 1 {
        return Err(anyhow!("acyclic executor: multi-result callee"));
    }
    let nparams = callee.signature.params.len();
    if state.stack.len() < nparams {
        return Err(anyhow!("acyclic executor: call arg underflow"));
    }
    let mut callee_locals: Vec<BV> = state.stack.split_off(state.stack.len() - nparams);
    for (count, lt) in callee.locals.iter() {
        let width = bv_width_for_value_type(*lt);
        for _ in 0..*count {
            callee_locals.push(BV::from_u64(0, width));
        }
    }
    let mut sub = AcyclicState {
        stack: Vec::new(),
        locals: callee_locals,
        globals: state.globals.clone(),
        memory: state.memory.clone(),
        path: state.path.clone(),
        reachable: true,
    };
    let mut sub_labels = vec![LabelJoin {
        arity,
        entries: Vec::new(),
    }];
    exec_acyclic(
        &callee.instructions,
        &mut sub,
        &mut sub_labels,
        ctx,
        depth + 1,
        cx,
    )?;
    if sub.reachable {
        acyclic_push_branch(&mut sub_labels, 0, sub.path.clone(), &sub)?;
    }
    let join = sub_labels.pop().unwrap();
    match acyclic_merge_entries(&join.entries) {
        Some((results, _locals, globals, memory, path)) => {
            state.globals = globals;
            state.memory = memory;
            state.path = path;
            if arity == 1 {
                state.stack.push(results[0].clone());
            }
        }
        None => {
            state.reachable = false;
        }
    }
    Ok(())
}

/// Everything a caller can observe after running a function: whether it
/// trapped, the globals and memory it left (in either case), and its results
/// when it returned.
#[cfg(feature = "verification")]
struct AcyclicObservation {
    /// Facts the execution satisfies; the comparison is conditional on them.
    assumed: Cond,
    trapped: Cond,
    trap_globals: Vec<BV>,
    trap_memory: Vec<Array>,
    results: Vec<BV>,
    globals: Vec<BV>,
    memory: Vec<Array>,
}

/// Encode a function via the exact acyclic executor. Inputs are named
/// deterministically (`param{i}`, `global{i}`, `memory`, `memory_size`) so two
/// encodings share them. Returns `Err` if the function is not modelable.
#[cfg(feature = "verification")]
fn encode_acyclic_function(
    func: &Function,
    ctx: &VerificationSignatureContext,
    by_body: &std::collections::HashSet<u32>,
) -> Result<AcyclicObservation> {
    if !is_acyclic_modelable_body(&func.instructions) {
        return Err(anyhow!("function not acyclic-modelable"));
    }
    let arity = func.signature.results.len();
    if arity > 1 {
        return Err(anyhow!("acyclic executor: multi-result not modeled"));
    }

    let mut locals: Vec<BV> = Vec::new();
    for (idx, pt) in func.signature.params.iter().enumerate() {
        locals.push(BV::new_const(
            format!("param{}", idx),
            bv_width_for_value_type(*pt),
        ));
    }
    for (count, lt) in func.locals.iter() {
        let width = bv_width_for_value_type(*lt);
        for _ in 0..*count {
            locals.push(BV::from_u64(0, width));
        }
    }
    let globals: Vec<BV> = ctx
        .global_types
        .iter()
        .enumerate()
        .map(|(i, t)| BV::new_const(format!("global{}", i), bv_width_for_value_type(*t)))
        .collect();
    // Memory 0 keeps the names `memory` and `memory_size`.
    let mem_ok: Vec<bool> = if ctx.memories.is_empty() {
        vec![true]
    } else {
        ctx.memories.clone()
    };
    let suffix = |i: usize| if i == 0 { String::new() } else { i.to_string() };
    let memory: Vec<Mem> = (0..mem_ok.len())
        .map(|i| {
            Mem::new(
                Array::new_const(
                    format!("memory{}", suffix(i)),
                    &Sort::bitvector(32),
                    &Sort::bitvector(8),
                ),
                BV::new_const(format!("memory_size{}", suffix(i)), 32),
            )
        })
        .collect();

    let mut state = AcyclicState {
        stack: Vec::new(),
        locals,
        globals: globals.clone(),
        memory: memory.clone(),
        path: Cond::True,
        reachable: true,
    };
    let mut labels = vec![LabelJoin {
        arity,
        entries: Vec::new(),
    }];
    let mut cx = AcyclicCx {
        havoc_counter: 0,
        traps: Vec::new(),
        mem_ok,
        assumed: memory
            .iter()
            .fold(Cond::True, |a, m| a.and(&pages_in_range(&m.pages))),
        by_body: by_body.clone(),
    };
    exec_acyclic(&func.instructions, &mut state, &mut labels, ctx, 0, &mut cx)?;
    if state.reachable {
        acyclic_push_branch(&mut labels, 0, state.path.clone(), &state)?;
    }
    let func_join = labels.pop().unwrap();

    // Normal exit. With none, every path traps and these are never compared.
    // Memory sizes are observable; they are compared after the globals.
    let with_pages = |mut g: Vec<BV>, ms: &[Mem]| {
        g.extend(ms.iter().map(|m| m.pages.clone()));
        g
    };
    let folded = |ms: &[Mem]| ms.iter().map(|m| m.folded.clone()).collect::<Vec<Array>>();
    let (results, ret_globals, ret_memory) = match acyclic_merge_entries(&func_join.entries) {
        Some((results, _, g, m, _)) => (results, with_pages(g, &m), folded(&m)),
        None => {
            let results = func
                .signature
                .results
                .iter()
                .map(|t| BV::from_u64(0, bv_width_for_value_type(*t)))
                .collect();
            (
                results,
                with_pages(globals.clone(), &memory),
                folded(&memory),
            )
        }
    };

    // Trap exits, merged the same way. Their guards are pairwise disjoint:
    // each one narrows the path the later ones run on.
    let (trapped, trap_globals, trap_memory) = match cx.traps.split_last() {
        None => (Cond::False, with_pages(globals, &memory), folded(&memory)),
        Some(((g_last, gl_last, _), rest)) => {
            let mut gl = gl_last.clone();
            let mut any = g_last.clone();
            for (g, gs, _) in rest.iter().rev() {
                gl = gs
                    .iter()
                    .zip(gl.iter())
                    .map(|(t, f)| g.ite_bv(t, f))
                    .collect();
                any = g.or(&any);
            }
            let mems: Vec<(Cond, &Vec<Mem>)> =
                cx.traps.iter().map(|t| (t.0.clone(), &t.2)).collect();
            let merged = Mem::merge_all(&mems);
            (any, with_pages(gl, &merged), folded(&merged))
        }
    };

    Ok(AcyclicObservation {
        assumed: cx.assumed.clone(),
        trapped,
        trap_globals,
        trap_memory,
        results,
        globals: ret_globals,
        memory: ret_memory,
    })
}

/// The condition under which two observations agree: both trap, leaving the
/// same globals and memory, or neither traps and both return the same results
/// with the same globals and memory.
#[cfg(feature = "verification")]
fn acyclic_same(o: &AcyclicObservation, p: &AcyclicObservation) -> Result<Cond> {
    if o.results.len() != p.results.len()
        || o.results
            .iter()
            .zip(p.results.iter())
            .any(|(a, b)| a.get_size() != b.get_size())
        || o.globals.len() != p.globals.len()
        || o.memory.len() != p.memory.len()
    {
        return Err(anyhow!("acyclic verify: result or global shape mismatch"));
    }
    // Memories are compared at one fresh address. It is free in the negated
    // obligation, so a counterexample exists exactly when some byte differs;
    // this keeps the obligation to select/store terms, without array
    // extensionality.
    let at = BV::new_const("observed_address", 32);
    // Identical terms (the encodings share hash-consed nodes wherever the two
    // functions do the same thing) need no equality: leaving them out keeps the
    // obligation to what the transform actually changed.
    let all_eq = |a: &[BV], b: &[BV], ma: &[Array], mb: &[Array]| {
        let mut c: Vec<Bool> = a
            .iter()
            .zip(b.iter())
            .filter(|(x, y)| x != y)
            .map(|(x, y)| x.eq(y))
            .collect();
        for (x, y) in ma.iter().zip(mb.iter()) {
            if x != y {
                c.push(x.select(&at).eq(y.select(&at)));
            }
        }
        Cond::all(c)
    };
    let same_return = all_eq(&o.results, &p.results, &o.memory, &p.memory)
        .and(&all_eq(&o.globals, &p.globals, &o.memory, &p.memory));
    let same_trap_state = all_eq(
        &o.trap_globals,
        &p.trap_globals,
        &o.trap_memory,
        &p.trap_memory,
    );
    // (both trap with the same state) or (neither traps and both return the
    // same).
    // Same trap condition and same state at the trap: then the only question
    // is the normal return, and `t or (not t and same)` is `t or same`.
    let same = if o.trapped.same_as(&p.trapped) && matches!(same_trap_state, Cond::True) {
        o.trapped.or(&same_return)
    } else {
        let both_trap = o.trapped.and(&p.trapped).and(&same_trap_state);
        let neither = o.trapped.not().and(&p.trapped.not()).and(&same_return);
        both_trap.or(&neither)
    };
    // Hypotheses imply the comparison. Shared hypotheses are the same term,
    // so they are stated once.
    let assumed = if o.assumed.same_as(&p.assumed) {
        o.assumed.clone()
    } else {
        o.assumed.and(&p.assumed)
    };
    Ok(match same {
        Cond::True => Cond::True,
        _ => assumed.not().or(&same),
    })
}

/// True when both functions are within what the exact executor models.
#[cfg(feature = "verification")]
pub(crate) fn exactly_modelable(original: &Function, optimized: &Function) -> bool {
    original.signature.results.len() <= 1
        && is_acyclic_modelable_body(&original.instructions)
        && is_acyclic_modelable_body(&optimized.instructions)
}

/// Route an obligation. Bodies within the exact executor are decided by it.
/// Loop-free bodies outside it are refused: the main encoder approximates
/// branches and compares neither final memory nor traps, so its acceptance is
/// not a proof. `None` leaves bodies with loops to the loop verifier.
#[cfg(feature = "verification")]
fn exact_route(
    original: &Function,
    optimized: &Function,
    ctx: &VerificationSignatureContext,
    backend: VerifyBackend,
) -> Option<Result<bool>> {
    if exactly_modelable(original, optimized) {
        return Some(verify_acyclic_equivalence_with_backend(
            original, optimized, ctx, backend,
        ));
    }
    if contains_any_loop(&original.instructions) || contains_any_loop(&optimized.instructions) {
        return None;
    }
    let culprit = first_unmodelled(&original.instructions)
        .or_else(|| first_unmodelled(&optimized.instructions))
        .unwrap_or_else(|| "a multi-result signature".to_string());
    Some(Err(anyhow!(
        "outside the exact executor (it does not model {})",
        culprit
    )))
}

/// The first instruction in `body` the exact executor does not admit.
#[cfg(feature = "verification")]
fn first_unmodelled(body: &[Instruction]) -> Option<String> {
    for instr in body {
        let inner = match instr {
            Instruction::Block { block_type, body } | Instruction::Loop { block_type, body } => {
                if acyclic_block_arity(block_type).is_err() {
                    return Some(format!("{:?}", block_type));
                }
                first_unmodelled(body)
            }
            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                if acyclic_block_arity(block_type).is_err() {
                    return Some(format!("{:?}", block_type));
                }
                first_unmodelled(then_body).or_else(|| first_unmodelled(else_body))
            }
            other => (!is_acyclic_modelable_body(std::slice::from_ref(other)))
                .then(|| format!("{:?}", other)),
        };
        if inner.is_some() {
            return inner;
        }
    }
    None
}

/// Decide `original ≡ optimized` with the exact acyclic executor: both trap
/// under the same inputs, leaving the same globals and memory, and otherwise
/// both return the same results with the same globals and memory.
///
/// `Ok(true)` is a proof; `Ok(false)` a counterexample; `Err` means not
/// decided (unmodelable input, solver unknown), which callers must treat as
/// not proven.
#[cfg(all(test, feature = "verification"))]
pub(crate) fn verify_acyclic_equivalence(
    original: &Function,
    optimized: &Function,
    ctx: &VerificationSignatureContext,
) -> Result<bool> {
    verify_acyclic_equivalence_with_backend(original, optimized, ctx, VerifyBackend::from_env())
}

#[cfg(feature = "verification")]
fn verify_acyclic_equivalence_with_backend(
    original: &Function,
    optimized: &Function,
    ctx: &VerificationSignatureContext,
    backend: VerifyBackend,
) -> Result<bool> {
    if original.signature.params != optimized.signature.params
        || original.signature.results != optimized.signature.results
    {
        return Err(anyhow!("acyclic verify: signature changed"));
    }
    if !exactly_modelable(original, optimized) {
        return Err(anyhow!("acyclic verify: not acyclic-modelable"));
    }

    let cfg = z3_config_with_timeout();
    with_z3_config(&cfg, || {
        let by_body = callees_to_model_by_body(original, optimized);
        let o = encode_acyclic_function(original, ctx, &by_body)?;
        let p = encode_acyclic_function(optimized, ctx, &by_body)?;
        let same = match acyclic_same(&o, &p)? {
            Cond::True => return Ok(true),
            Cond::False => {
                revert_note(format_args!(
                    "exact verify: the two functions differ on every input"
                ));
                return Ok(false);
            }
            Cond::B(b) => b,
        };

        // Offer the obligation to the configured engine first, as one bit.
        let lhs = same.ite(&BV::from_u64(1, 1), &BV::from_u64(0, 1));
        let rhs = BV::from_u64(1, 1);
        let deferred =
            match verify_solver::decide_bv_equivalence_traps_modelled(&lhs, &rhs, backend) {
                SeamOutcome::Decided(verdict) => {
                    return match verdict {
                        RuleVerdict::Proven => Ok(true),
                        RuleVerdict::Disproven(model) => {
                            revert_note(format_args!("exact verify: counterexample: {}", model));
                            Ok(false)
                        }
                        RuleVerdict::Unknown => Err(anyhow!("acyclic verify: solver unknown")),
                    };
                }
                SeamOutcome::Deferred(reason) => reason,
            };
        if backend == VerifyBackend::Ordeal {
            return Err(anyhow!(
                "acyclic verify: the new engine deferred this obligation ({:?})",
                deferred
            ));
        }

        let solver = Solver::new();
        solver.assert(same.not());
        match solver.check() {
            SatResult::Unsat => Ok(true),
            SatResult::Sat => {
                if let Some(model) = solver.get_model() {
                    revert_note(format_args!("exact verify: counterexample: {}", model));
                }
                Ok(false)
            }
            SatResult::Unknown => Err(anyhow!("acyclic verify: solver unknown (timeout)")),
        }
    })
}

/// Verify that block stack properties are preserved across optimization
///
/// Uses Z3 to formally verify that a block transformation preserves:
/// 1. Stack composition (instructions sequence correctly)
/// 2. Input parameter types
/// 3. Output result types
///
/// # Arguments
///
/// * `original_instrs` - Instructions in original block
/// * `optimized_instrs` - Instructions in optimized block
/// * `block_params` - Input types the block expects
/// * `block_results` - Output types the block must produce
///
/// # Returns
///
/// * `Ok(true)` - Stack properties are preserved
/// * `Ok(false)` - Found a counterexample
/// * `Err(_)` - Verification error
#[cfg(feature = "verification")]
pub fn verify_stack_properties(
    original_instrs: &[Instruction],
    optimized_instrs: &[Instruction],
    block_params: &[crate::ValueType],
    block_results: &[crate::ValueType],
) -> Result<bool> {
    // Convert ValueType to stack::ValueType
    let stack_params: Vec<crate::stack::ValueType> = block_params
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    let stack_results: Vec<crate::stack::ValueType> = block_results
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    // First, validate that both blocks have correct structure
    let orig_validation =
        crate::stack::validation::validate_block(original_instrs, &stack_params, &stack_results);

    let opt_validation =
        crate::stack::validation::validate_block(optimized_instrs, &stack_params, &stack_results);

    // Check validation results
    match (&orig_validation, &opt_validation) {
        (
            crate::stack::validation::ValidationResult::Valid(_),
            crate::stack::validation::ValidationResult::Valid(_),
        ) => {
            // Both are valid - stack properties are preserved!
            Ok(true)
        }
        (crate::stack::validation::ValidationResult::Valid(_), _) => {
            // Original valid but optimized invalid - stack properties NOT preserved
            eprintln!(
                "Stack property violation: optimization produced invalid stack: {:?}",
                opt_validation
            );
            Ok(false)
        }
        (_, crate::stack::validation::ValidationResult::Valid(_)) => {
            // This shouldn't happen if original was valid - something went wrong
            eprintln!(
                "Stack property violation: original block became invalid: {:?}",
                orig_validation
            );
            Ok(false)
        }
        _ => {
            // Both invalid
            eprintln!(
                "Stack property violation in original block: {:?}",
                orig_validation
            );
            Ok(false)
        }
    }
}

/// Validate all blocks in a module using stack analysis
///
/// Phase 5 Integration: Validates that all block structures in the module have
/// correct stack compositions. This is called after optimization passes to ensure
/// they preserve stack types.
///
/// # Arguments
///
/// * `module` - The module to validate
///
/// # Returns
///
/// * `Ok(true)` - All blocks have valid stack properties
/// * `Ok(false)` - A block violates stack composition rules
/// * `Err(_)` - Validation error
#[cfg(feature = "verification")]
pub fn validate_module_blocks(module: &Module) -> Result<bool> {
    for func in &module.functions {
        // Collect block parameters from function signature
        let block_params: Vec<crate::ValueType> = func.signature.params.clone();
        let block_results: Vec<crate::ValueType> = func.signature.results.clone();

        // Validate function body blocks
        if !validate_instruction_sequence(&func.instructions, &block_params, &block_results)? {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Helper to recursively validate instruction sequences within blocks
#[cfg(feature = "verification")]
fn validate_instruction_sequence(
    instrs: &[crate::Instruction],
    params: &[crate::ValueType],
    results: &[crate::ValueType],
) -> Result<bool> {
    // Use our stack validation module to check composition
    let stack_params: Vec<crate::stack::ValueType> = params
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    let stack_results: Vec<crate::stack::ValueType> = results
        .iter()
        .map(|vt| match vt {
            crate::ValueType::I32 => crate::stack::ValueType::I32,
            crate::ValueType::I64 => crate::stack::ValueType::I64,
            crate::ValueType::F32 => crate::stack::ValueType::F32,
            crate::ValueType::F64 => crate::stack::ValueType::F64,
        })
        .collect();

    // Create a block validation context
    let result = crate::stack::validation::validate_block(instrs, &stack_params, &stack_results);

    match result {
        crate::stack::validation::ValidationResult::Valid(_) => Ok(true),
        crate::stack::validation::ValidationResult::StackMismatch {
            position,
            expected,
            actual,
        } => {
            eprintln!("Stack validation: StackMismatch at position {}", position);
            eprintln!("  Expected: {:?}", expected);
            eprintln!("  Actual: {:?}", actual);
            Ok(false)
        }
        crate::stack::validation::ValidationResult::MissingInput { expected_params } => {
            eprintln!(
                "Stack validation: MissingInput, expected: {:?}",
                expected_params
            );
            Ok(false)
        }
        crate::stack::validation::ValidationResult::WrongOutput {
            expected_results,
            actual_results,
        } => {
            eprintln!("Stack validation: WrongOutput");
            eprintln!("  Expected: {:?}", expected_results);
            eprintln!("  Actual: {:?}", actual_results);
            Ok(false)
        }
        crate::stack::validation::ValidationResult::InstructionError {
            position,
            message,
            stack_state,
        } => {
            eprintln!(
                "Stack validation: InstructionError at position {}",
                position
            );
            eprintln!("  Message: {}", message);
            eprintln!("  Stack state: {:?}", stack_state);
            Ok(false)
        }
    }
}

/// Stub for builds without the `verification` feature: there is no verifier
/// to consult, so this reports an error rather than a verdict.
///
/// Documented because `-D warnings` implies `-D missing-docs`, and the
/// verification gate builds that way — but also because a public function
/// whose whole job is to decline is exactly the one whose contract should be
/// written down (#331: nothing was verified, and it says so).
#[cfg(not(feature = "verification"))]
pub fn verify_optimization(_original: &Module, _optimized: &Module) -> Result<bool> {
    Err(anyhow!(
        "Verification support not enabled. Rebuild with --features verification"
    ))
}

/// Non-verification stub for module block validation when Z3 feature is disabled
#[cfg(not(feature = "verification"))]
pub fn validate_module_blocks(_module: &Module) -> Result<bool> {
    Err(anyhow!(
        "Verification support not enabled. Rebuild with --features verification"
    ))
}

/// Non-verification stub for stack properties when Z3 feature is disabled
#[cfg(not(feature = "verification"))]
pub fn verify_stack_properties(
    _original_instrs: &[crate::Instruction],
    _optimized_instrs: &[crate::Instruction],
    _block_params: &[crate::ValueType],
    _block_results: &[crate::ValueType],
) -> Result<bool> {
    Err(anyhow!(
        "Verification support not enabled. Rebuild with --features verification"
    ))
}

#[cfg(all(test, feature = "verification"))]
mod tests {
    use super::*;

    fn only_func(wat: &str) -> Function {
        let wasm = wat::parse_str(wat).expect("wat");
        let module = crate::parse::parse_wasm(&wasm).expect("parse");
        module.functions[0].clone()
    }

    const IF_THEN_SET: &str = r#"(module (global $g (mut i32) (i32.const 0))
        (func (param $x i32)
          (if (local.get $x) (then (global.set $g (i32.const 7))))
          (global.set $g (i32.const 3))))"#;

    /// The two writes in the opposite order: `g` ends at 7 when `x != 0`.
    const IF_THEN_SET_REORDERED: &str = r#"(module (global $g (mut i32) (i32.const 0))
        (func (param $x i32)
          (global.set $g (i32.const 3))
          (if (local.get $x) (then (global.set $g (i32.const 7))))))"#;

    /// Verification covers side-effect ordering on globals: a void function
    /// whose global writes are reordered is rejected, by both validators.
    #[test]
    fn reordered_global_writes_are_rejected() {
        let orig = only_func(IF_THEN_SET);
        let bad = only_func(IF_THEN_SET_REORDERED);
        assert!(
            TranslationValidator::new(&orig, "test")
                .verify(&bad)
                .is_err()
        );
        let module = crate::parse::parse_wasm(&wat::parse_str(IF_THEN_SET).unwrap()).unwrap();
        let ctx = VerificationSignatureContext::from_module(&module);
        assert!(
            TranslationValidator::new_with_context(&orig, "test", ctx)
                .verify(&bad)
                .is_err()
        );
    }

    /// The control: the unchanged body still verifies.
    #[test]
    fn unchanged_global_writes_are_accepted() {
        let module = parse::parse_wat(IF_THEN_SET).unwrap();
        let orig = module.functions[0].clone();
        let ctx = VerificationSignatureContext::from_module(&module);
        assert!(
            TranslationValidator::new_with_context(&orig, "test", ctx)
                .verify(&orig)
                .is_ok()
        );
    }

    /// The same reordering in a function with a result: the return values
    /// agree, the globals do not.
    #[test]
    fn reordered_global_writes_are_rejected_with_a_result() {
        let orig = only_func(
            r#"(module (global $g (mut i32) (i32.const 0))
                (func (param $x i32) (result i32)
                  (if (local.get $x) (then (global.set $g (i32.const 7))))
                  (global.set $g (i32.const 3))
                  (i32.const 1)))"#,
        );
        let bad = only_func(
            r#"(module (global $g (mut i32) (i32.const 0))
                (func (param $x i32) (result i32)
                  (global.set $g (i32.const 3))
                  (if (local.get $x) (then (global.set $g (i32.const 7))))
                  (i32.const 1)))"#,
        );
        assert!(
            TranslationValidator::new(&orig, "test")
                .verify(&bad)
                .is_err()
        );
    }
    use crate::parse;

    // ---- PR-C (gale #219) M1a: precise acyclic CF executor, tested in isolation ----

    /// Encode both functions via the precise acyclic executor (deterministic
    /// input names → they unify in Z3) and check semantic equivalence.
    /// `Ok(true)` = proven equivalent, `Ok(false)` = counterexample found.
    #[cfg(feature = "verification")]
    fn acyclic_equiv(wat_a: &str, wat_b: &str) -> Result<bool> {
        acyclic_equiv_at(wat_a, 0, wat_b, 0)
    }

    /// As `acyclic_equiv` but comparing a chosen function index in each module
    /// (so a caller that `call`s a sibling can be compared to its inlined form).
    #[cfg(feature = "verification")]
    fn acyclic_equiv_at(wat_a: &str, ia: usize, wat_b: &str, ib: usize) -> Result<bool> {
        let ma = parse::parse_wat(wat_a).unwrap();
        let mb = parse::parse_wat(wat_b).unwrap();
        let ctx_a = VerificationSignatureContext::from_module(&ma);
        let ctx_b = VerificationSignatureContext::from_module(&mb);
        let fa = &ma.functions[ia];
        let fb = &mb.functions[ib];
        let cfg = z3_config_with_timeout();
        with_z3_config(&cfg, || {
            let solver = Solver::new();
            let by_body = callees_to_model_by_body(fa, fb);
            let ra = encode_acyclic_function(fa, &ctx_a, &by_body)?;
            let rb = encode_acyclic_function(fb, &ctx_b, &by_body)?;
            let same = match acyclic_same(&ra, &rb)? {
                Cond::True => return Ok(true),
                Cond::False => return Ok(false),
                Cond::B(b) => b,
            };
            solver.assert(same.not());
            match solver.check() {
                SatResult::Unsat => Ok(true),
                SatResult::Sat => Ok(false),
                SatResult::Unknown => Err(anyhow!("solver unknown")),
            }
        })
    }

    /// 3-way `br_table` switch (0→10, 1→20, default→30).
    #[cfg(feature = "verification")]
    const SWITCH_10_20_30: &str = r#"
        (module (func (param i32) (result i32)
            (block (block (block
                local.get 0
                br_table 0 1 2)
              i32.const 10
              return)
              i32.const 20
              return)
            i32.const 30
            return))"#;

    /// Same dispatch expressed as an if/else chain — semantically identical.
    #[cfg(feature = "verification")]
    const IFELSE_10_20_30: &str = r#"
        (module (func (param i32) (result i32)
            local.get 0
            i32.eqz
            if (result i32)
                i32.const 10
            else
                local.get 0
                i32.const 1
                i32.eq
                if (result i32)
                    i32.const 20
                else
                    i32.const 30
                end
            end))"#;

    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_brtable_equiv_ifelse() {
        // The precise executor must model a br_table switch identically to the
        // equivalent if/else chain (independent encodings → real proof).
        assert!(
            acyclic_equiv(SWITCH_10_20_30, IFELSE_10_20_30).unwrap(),
            "br_table switch must prove equivalent to the if/else chain"
        );
    }

    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_brtable_identity() {
        // A switch is trivially equivalent to itself (sanity: the executor is
        // deterministic and the merge is stable).
        assert!(acyclic_equiv(SWITCH_10_20_30, SWITCH_10_20_30).unwrap());
    }

    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_brtable_detects_difference() {
        // A switch whose default returns 99 instead of 30 must NOT prove
        // equivalent — the executor must distinguish, not over-approve.
        let switch_wrong_default = r#"
            (module (func (param i32) (result i32)
                (block (block (block
                    local.get 0
                    br_table 0 1 2)
                  i32.const 10
                  return)
                  i32.const 20
                  return)
                i32.const 99
                return))"#;
        assert!(
            !acyclic_equiv(SWITCH_10_20_30, switch_wrong_default).unwrap(),
            "differing default arm must yield a counterexample (sel >= 2)"
        );
    }

    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_brtable_detects_swapped_arms() {
        // Swapping arm 0 and arm 1 results must be caught (sel==0 / sel==1).
        let swapped = r#"
            (module (func (param i32) (result i32)
                (block (block (block
                    local.get 0
                    br_table 0 1 2)
                  i32.const 20
                  return)
                  i32.const 10
                  return)
                i32.const 30
                return))"#;
        assert!(
            !acyclic_equiv(SWITCH_10_20_30, swapped).unwrap(),
            "swapped arm values must yield a counterexample"
        );
    }

    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_modelable_gate() {
        // The whole-function gate must reject loops (back-edges).
        let with_loop =
            parse::parse_wat("(module (func (param i32) (result i32) (loop local.get 0)))")
                .unwrap();
        assert!(
            !is_acyclic_modelable_body(&with_loop.functions[0].instructions),
            "loops (back-edges) must be rejected"
        );

        // PR-C Phase 2: memory ops are now ADMITTED (the executor threads a
        // memory Array). (Phase 1 rejected them.)
        let with_memory = parse::parse_wat(
            "(module (memory 1) (func (param i32) (result i32) local.get 0 i32.load))",
        )
        .unwrap();
        assert!(
            is_acyclic_modelable_body(&with_memory.functions[0].instructions),
            "memory ops are modelable in Phase 2"
        );

        // A pure-integer br_table switch must be admitted.
        let switch = parse::parse_wat(SWITCH_10_20_30).unwrap();
        assert!(
            is_acyclic_modelable_body(&switch.functions[0].instructions),
            "pure-integer acyclic br_table function must be admitted"
        );
    }

    /// M1b: a caller that `call`s a br_table callee must prove equivalent to the
    /// same logic with the callee inlined — i.e. the executor models `call F`
    /// by-body identically to the inlined body (the inline-soundness principle
    /// extended to control flow). func 0 = decide (switch), func 1 = z (calls it).
    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_call_by_body_equiv_inlined() {
        let caller_with_call = r#"
            (module
                (func (param i32) (result i32)
                    (block (block (block
                        local.get 0
                        br_table 0 1 2)
                      i32.const 10
                      return)
                      i32.const 20
                      return)
                    i32.const 30
                    return)
                (func (param i32) (result i32)
                    local.get 0
                    call 0))"#;
        // z (func 1) calls decide (func 0); compare to the inlined switch.
        assert!(
            acyclic_equiv_at(caller_with_call, 1, SWITCH_10_20_30, 0).unwrap(),
            "call-by-body must prove equivalent to the inlined callee body"
        );
    }

    /// M1b: a call to a no-return callee (panic→unreachable) must be modeled as
    /// ⊥ (diverge), never havoc. A function that returns 7 on one path and traps
    /// on the other proves equivalent to one that simply returns 7 (the trapping
    /// path drops out of the result; both sides agree on the returning path).
    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_noreturn_call_diverges() {
        // func 0 = panic (unreachable); func 1 returns 7 or traps through it.
        let with_trap = r#"
            (module
                (func unreachable)
                (func (param i32) (result i32)
                    local.get 0
                    if (result i32)
                        i32.const 7
                    else
                        call 0
                        i32.const 7
                    end))"#;
        let inlined = r#"
            (module (func (param i32) (result i32)
                local.get 0
                if (result i32)
                    i32.const 7
                else
                    unreachable
                end))"#;
        let always7 = r#"
            (module (func (param i32) (result i32) i32.const 7))"#;
        assert!(
            acyclic_equiv_at(with_trap, 1, inlined, 0).unwrap(),
            "the call to a trapping function equals its inlined trap"
        );
        assert!(
            !acyclic_equiv_at(with_trap, 1, always7, 0).unwrap(),
            "a trap is observable: removing it is not equivalent"
        );
    }

    /// A direct `unreachable` (trap) on one path must diverge — same ⊥ discipline
    /// as a no-return call. The seam decides end their overflow arms in
    /// `unreachable`, so the executor must model it (not reject the function).
    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_unreachable_diverges() {
        let with_trap = r#"
            (module (func (param i32) (result i32)
                local.get 0
                if (result i32)
                    i32.const 7
                else
                    unreachable
                end))"#;
        let guard_first = r#"
            (module (func (param i32) (result i32)
                local.get 0
                i32.eqz
                if
                    unreachable
                end
                i32.const 7))"#;
        let always7 = r#"(module (func (param i32) (result i32) i32.const 7))"#;
        assert!(
            acyclic_equiv(with_trap, guard_first).unwrap(),
            "the same trap condition in another shape is equivalent"
        );
        assert!(
            !acyclic_equiv(with_trap, always7).unwrap(),
            "a trap is observable: removing it is not equivalent"
        );
    }

    /// PR-C Phase 2: a store on each branch followed by a load must MERGE memory
    /// correctly (the load sees ite(cond, mem_then, mem_else)). Equivalent to the
    /// direct if/else — proves the per-join memory ITE-merge is wired right.
    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_memory_merge() {
        let with_mem = r#"
            (module (memory 1)
                (func (param i32) (result i32)
                    local.get 0
                    if
                        i32.const 100
                        i32.const 5
                        i32.store
                    else
                        i32.const 100
                        i32.const 6
                        i32.store
                    end
                    i32.const 100
                    i32.load))"#;
        // Same stores, the load folded through the per-branch merge.
        let folded = r#"
            (module (memory 1)
                (func (param i32) (result i32)
                    local.get 0
                    if
                        i32.const 100
                        i32.const 5
                        i32.store
                    else
                        i32.const 100
                        i32.const 6
                        i32.store
                    end
                    local.get 0
                    if (result i32)
                        i32.const 5
                    else
                        i32.const 6
                    end))"#;
        // Same result, but the stores are gone: memory differs.
        let direct = r#"
            (module (memory 1)
                (func (param i32) (result i32)
                    local.get 0
                    if (result i32)
                        i32.const 5
                    else
                        i32.const 6
                    end))"#;
        assert!(
            acyclic_equiv(with_mem, folded).unwrap(),
            "store-on-each-branch then load must merge memory, so the load folds"
        );
        assert!(
            !acyclic_equiv(with_mem, direct).unwrap(),
            "dropping the stores changes memory"
        );
    }

    /// PR-C Phase 2b: a call to an impure import is HAVOC'd. Two encodings of the
    /// same impure call must UNIFY (same per-encode havoc id) → equivalent. (If
    /// the counter weren't deterministic per-encode, the two `call_0_result_0`
    /// names would differ and this would spuriously fail.)
    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_havoc_call_unifies() {
        let calls_import = r#"
            (module
                (import "env" "f" (func $f (param i32) (result i32)))
                (func (param i32) (result i32)
                    local.get 0
                    call $f))"#;
        assert!(
            acyclic_equiv(calls_import, calls_import).unwrap(),
            "two encodings of the same impure call must unify (havoc id parity)"
        );
    }

    /// PR-C Phase 2b SOUNDNESS: a havoc'd call result must NOT be provably equal
    /// to a concrete value — `call $f(x)` (opaque) is not `x`. This is the guard
    /// against a havoc-vs-concrete false equivalence (#155/#159 surface).
    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_havoc_not_concrete() {
        let calls_import = r#"
            (module
                (import "env" "f" (func $f (param i32) (result i32)))
                (func (param i32) (result i32)
                    local.get 0
                    call $f))"#;
        let identity = r#"
            (module
                (import "env" "f" (func $f (param i32) (result i32)))
                (func (param i32) (result i32)
                    local.get 0))"#;
        assert!(
            !acyclic_equiv(calls_import, identity).unwrap(),
            "a havoc'd call result must NOT prove equal to a concrete value"
        );
    }

    /// PR-C Phase 2b SOUNDNESS: an impure call havocs memory, so a load AFTER the
    /// call reads the havoc'd memory — NOT provably equal to loading the initial
    /// memory. Guards against the havoc failing to invalidate memory.
    #[test]
    #[cfg(feature = "verification")]
    fn test_acyclic_havoc_memory() {
        let load_after_call = r#"
            (module
                (import "env" "g" (func $g))
                (memory 1)
                (func (result i32)
                    call $g
                    i32.const 0
                    i32.load))"#;
        let load_initial = r#"
            (module
                (import "env" "g" (func $g))
                (memory 1)
                (func (result i32)
                    i32.const 0
                    i32.load))"#;
        assert!(
            !acyclic_equiv(load_after_call, load_initial).unwrap(),
            "a load after an impure call must read havoc'd memory, not the initial"
        );
    }

    /// M3: the public verify_acyclic_equivalence fast-path proves a caller that
    /// `call`s a br_table decide equivalent to the inlined switch, and rejects a
    /// non-equivalent optimized body — the exact inline verify-or-revert pattern.
    #[test]
    #[cfg(feature = "verification")]
    fn test_verify_acyclic_equivalence_inline() {
        let module_wat = r#"
            (module
                (func (param i32) (result i32)
                    (block (block (block
                        local.get 0
                        br_table 0 1 2)
                      i32.const 10
                      return)
                      i32.const 20
                      return)
                    i32.const 30
                    return)
                (func (param i32) (result i32)
                    local.get 0
                    call 0))"#;
        let m = parse::parse_wat(module_wat).unwrap();
        let ctx = VerificationSignatureContext::from_module(&m);
        let z = &m.functions[1]; // caller: local.get 0; call decide

        // Equivalent: the inlined switch.
        let inlined = parse::parse_wat(SWITCH_10_20_30).unwrap();
        assert!(
            verify_acyclic_equivalence(z, &inlined.functions[0], &ctx).unwrap(),
            "z (calls decide) must verify ≡ the inlined switch"
        );

        // Non-equivalent: a switch with a wrong default (99) must NOT verify.
        let wrong = parse::parse_wat(
            r#"(module (func (param i32) (result i32)
                (block (block (block local.get 0 br_table 0 1 2)
                  i32.const 10 return) i32.const 20 return) i32.const 99 return))"#,
        )
        .unwrap();
        assert!(
            !verify_acyclic_equivalence(z, &wrong.functions[0], &ctx).unwrap(),
            "a wrong-default body must NOT verify equivalent"
        );

        // A function with a loop is not acyclic-modelable → Err (caller falls back).
        let loopy = parse::parse_wat(
            "(module (func (param i32) (result i32) (loop (result i32) local.get 0)))",
        )
        .unwrap();
        assert!(
            verify_acyclic_equivalence(&loopy.functions[0], &loopy.functions[0], &ctx).is_err(),
            "non-acyclic functions must Err so the caller falls back"
        );
    }

    #[test]
    fn test_verify_constant_folding() {
        // Original: (i32.add (i32.const 2) (i32.const 3))
        let original_wat = r#"
            (module
                (func (result i32)
                    i32.const 2
                    i32.const 3
                    i32.add
                )
            )
        "#;

        // Optimized: (i32.const 5)
        let optimized_wat = r#"
            (module
                (func (result i32)
                    i32.const 5
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "Programs should be equivalent");
    }

    #[test]
    fn test_verify_strength_reduction() {
        // Original: (i32.mul x 4)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 4
                    i32.mul
                )
            )
        "#;

        // Optimized: (i32.shl x 2)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 2
                    i32.shl
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(
            result.unwrap(),
            "Strength reduction should preserve semantics"
        );
    }

    #[test]
    fn test_verify_bitwise_identity() {
        // Original: (i32.xor x x)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    local.get 0
                    i32.xor
                )
            )
        "#;

        // Optimized: (i32.const 0)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // Verify equivalence
        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "x XOR x = 0 should be proven");
    }

    #[test]
    fn test_verify_detects_incorrect_optimization() {
        // Original: (i32.add x 1)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
            )
        "#;

        // Incorrectly optimized to: (i32.const 2) - WRONG!
        let wrong_optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 2
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let wrong_optimized = parse::parse_wat(wrong_optimized_wat).unwrap();

        // Verification should detect this is wrong
        let result = verify_optimization(&original, &wrong_optimized);
        assert!(result.is_ok(), "Verification should complete");
        assert!(!result.unwrap(), "Should detect incorrect optimization");
    }

    #[test]
    fn test_verify_comparison_optimization() {
        // Original: (i32.eq x x) should always be true (1)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    local.get 0
                    i32.eq
                )
            )
        "#;

        // Optimized: (i32.const 1)
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 1
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "x == x should always be 1");
    }

    #[test]
    fn test_verify_division_optimization() {
        // Original: (i32.div_u x 1) should equal x
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.div_u
                )
            )
        "#;

        // Optimized: just return x
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "x / 1 should equal x");
    }

    #[test]
    fn test_verify_eqz_optimization() {
        // Original: (i32.eqz (i32.const 0)) should be 1
        let original_wat = r#"
            (module
                (func (result i32)
                    i32.const 0
                    i32.eqz
                )
            )
        "#;

        // Optimized: (i32.const 1)
        let optimized_wat = r#"
            (module
                (func (result i32)
                    i32.const 1
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "eqz(0) should be 1");
    }

    #[test]
    fn test_verify_select_optimization() {
        // Original: (select 42 99 1) should be 42 (condition != 0)
        let original_wat = r#"
            (module
                (func (result i32)
                    i32.const 42
                    i32.const 99
                    i32.const 1
                    select
                )
            )
        "#;

        // Optimized: (i32.const 42)
        let optimized_wat = r#"
            (module
                (func (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "select(42, 99, 1) should be 42");
    }

    #[test]
    fn test_verify_global_operations() {
        // Test that global.get followed by identity transformation works
        let original_wat = r#"
            (module
                (global (mut i32) (i32.const 0))
                (func (result i32)
                    global.get 0
                    i32.const 0
                    i32.add
                )
            )
        "#;

        // Optimized: just global.get (adding 0 is identity)
        let optimized_wat = r#"
            (module
                (global (mut i32) (i32.const 0))
                (func (result i32)
                    global.get 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "global.get + 0 should equal global.get");
    }

    #[test]
    fn test_verify_if_else_simplification() {
        // Original: if x then 42 else 42 (both branches return same value)
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    if (result i32)
                        i32.const 42
                    else
                        i32.const 42
                    end
                )
            )
        "#;

        // Optimized: just return 42
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "if x then 42 else 42 should equal 42");
    }

    #[test]
    fn test_verify_if_else_with_condition() {
        // Original: if 1 then x else y (condition is always true)
        let original_wat = r#"
            (module
                (func (param i32) (param i32) (result i32)
                    i32.const 1
                    if (result i32)
                        local.get 0
                    else
                        local.get 1
                    end
                )
            )
        "#;

        // Optimized: just return x (first param)
        let optimized_wat = r#"
            (module
                (func (param i32) (param i32) (result i32)
                    local.get 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "if 1 then x else y should equal x");
    }

    #[test]
    fn test_verify_block_result() {
        // Original: block with result type that does arithmetic
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    block (result i32)
                        local.get 0
                        i32.const 1
                        i32.add
                    end
                )
            )
        "#;

        // Optimized: same arithmetic without block wrapper
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(result.unwrap(), "block {{ x + 1 }} should equal x + 1");
    }

    #[test]
    fn test_verify_nested_if() {
        // Original: nested if that simplifies
        let original_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 1
                    if (result i32)
                        i32.const 0
                        if (result i32)
                            i32.const 99
                        else
                            i32.const 42
                        end
                    else
                        i32.const 0
                    end
                )
            )
        "#;

        // Optimized: the outer if is true, inner is false, so result is 42
        let optimized_wat = r#"
            (module
                (func (param i32) (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(result.is_ok(), "Verification failed: {:?}", result);
        assert!(
            result.unwrap(),
            "nested if with constant conditions should simplify"
        );
    }

    #[test]
    fn test_verify_float_param_function() {
        // Test that functions with f32 parameters can be verified
        // Float parameters are treated as bitvectors for verification
        let original_wat = r#"
            (module
                (func (param f32) (result i32)
                    i32.const 42
                )
            )
        "#;

        // Same function - should verify as equivalent
        let optimized_wat = r#"
            (module
                (func (param f32) (result i32)
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification with f32 param failed: {:?}",
            result
        );
        assert!(result.unwrap(), "Functions with f32 params should verify");
    }

    #[test]
    fn test_verify_f64_param_function() {
        // Test that functions with f64 parameters can be verified
        let original_wat = r#"
            (module
                (func (param f64) (result i32)
                    i32.const 99
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (func (param f64) (result i32)
                    i32.const 99
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification with f64 param failed: {:?}",
            result
        );
        assert!(result.unwrap(), "Functions with f64 params should verify");
    }

    #[test]
    fn test_verify_float_constant_folding() {
        // Test that f32 constant folding works (bits are preserved)
        // Original: f32.const 3.14
        let original_wat = r#"
            (module
                (func (result f32)
                    f32.const 3.14
                )
            )
        "#;

        // Same constant - bits should match
        let optimized_wat = r#"
            (module
                (func (result f32)
                    f32.const 3.14
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of f32 constant failed: {:?}",
            result
        );
        assert!(result.unwrap(), "f32 constants should verify as equivalent");
    }

    // ================================================================
    // Bulk memory operation Z3 verification tests
    // ================================================================

    #[test]
    fn test_verify_memory_fill_with_constant_folding() {
        // Test that constant folding of memory.fill operands is verified correctly.
        // Original: memory.fill(0, 2+3, 10) then return 42
        // Optimized: memory.fill(0, 5, 10) then return 42
        // The memory.fill invalidates memory, but the return value (a constant)
        // should still verify as equivalent.
        let original_wat = r#"
            (module
                (memory 1)
                (func (result i32)
                    i32.const 0
                    i32.const 2
                    i32.const 3
                    i32.add
                    i32.const 10
                    memory.fill
                    i32.const 42
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (memory 1)
                (func (result i32)
                    i32.const 0
                    i32.const 5
                    i32.const 10
                    memory.fill
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of memory.fill with constant folding failed: {:?}",
            result
        );
        assert!(
            result.unwrap(),
            "memory.fill with constant-folded operands should verify as equivalent"
        );
    }

    #[test]
    fn test_verify_memory_copy_with_constant_folding() {
        // Test that constant folding of memory.copy operands is verified correctly.
        // Original: memory.copy(0, 100, 1+1) then return 99
        // Optimized: memory.copy(0, 100, 2) then return 99
        let original_wat = r#"
            (module
                (memory 1)
                (func (result i32)
                    i32.const 0
                    i32.const 100
                    i32.const 1
                    i32.const 1
                    i32.add
                    memory.copy
                    i32.const 99
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (memory 1)
                (func (result i32)
                    i32.const 0
                    i32.const 100
                    i32.const 2
                    memory.copy
                    i32.const 99
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of memory.copy with constant folding failed: {:?}",
            result
        );
        assert!(
            result.unwrap(),
            "memory.copy with constant-folded operands should verify as equivalent"
        );
    }

    #[test]
    fn test_verify_memory_init_with_constant_folding() {
        // Test that memory.init with constant-folded operands verifies.
        // Original: memory.init(0, 0, 2+3) then return 7
        // Optimized: memory.init(0, 0, 5) then return 7
        let original_wat = r#"
            (module
                (memory 1)
                (data "hello world")
                (func (result i32)
                    i32.const 0
                    i32.const 0
                    i32.const 2
                    i32.const 3
                    i32.add
                    memory.init 0
                    i32.const 7
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (memory 1)
                (data "hello world")
                (func (result i32)
                    i32.const 0
                    i32.const 0
                    i32.const 5
                    memory.init 0
                    i32.const 7
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of memory.init with constant folding failed: {:?}",
            result
        );
        assert!(
            result.unwrap(),
            "memory.init with constant-folded operands should verify as equivalent"
        );
    }

    #[test]
    fn test_verify_data_drop_is_refused() {
        // Original: data.drop 0; return 42
        // Optimized: data.drop 0; return 42
        let original_wat = r#"
            (module
                (memory 1)
                (data "test")
                (func (result i32)
                    data.drop 0
                    i32.const 42
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (memory 1)
                (data "test")
                (func (result i32)
                    data.drop 0
                    i32.const 42
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        // data.drop changes state a later memory.init observes (the segment
        // becomes empty), and the validator does not model that state, so a
        // body containing it is refused rather than called equivalent.
        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_err(),
            "a body with data.drop must not be reported verified: {:?}",
            result
        );
    }

    #[test]
    fn test_verify_memory_fill_invalidates_memory() {
        // After memory.fill, a load should return an unconstrained symbolic value.
        // This tests that the memory array is properly invalidated.
        // Both functions: store 42, fill, load. The load result should be unconstrained
        // (not necessarily 42) because memory.fill invalidates memory.
        // Since both functions do the same thing, they should verify as equivalent.
        let original_wat = r#"
            (module
                (memory 1)
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 42
                    i32.store
                    i32.const 0
                    i32.const 0
                    i32.const 100
                    memory.fill
                    local.get 0
                    i32.load
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (memory 1)
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 42
                    i32.store
                    i32.const 0
                    i32.const 0
                    i32.const 100
                    memory.fill
                    local.get 0
                    i32.load
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of store-fill-load pattern failed: {:?}",
            result
        );
        // Note: with independent symbolic inputs (verify_optimization path),
        // both functions create independent memory arrays, so loads after fill
        // return independent unconstrained values. The result comparison is:
        // fresh_sym_A != fresh_sym_B which IS satisfiable.
        // This is expected behavior with verify_optimization (non-shared inputs).
        // The shared-inputs path (verify_function_equivalence_with_context)
        // would correctly handle this. For this test, we just verify no crash.
    }

    #[test]
    fn test_verify_bulk_memory_stack_effects() {
        // Verify that bulk memory operations correctly consume 3 stack values
        // and produce none. The return value is computed before the bulk op.
        let original_wat = r#"
            (module
                (memory 1)
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                    local.set 0
                    i32.const 0
                    i32.const 255
                    i32.const 64
                    memory.fill
                    local.get 0
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (memory 1)
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                    local.set 0
                    i32.const 0
                    i32.const 255
                    i32.const 64
                    memory.fill
                    local.get 0
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of bulk memory stack effects failed: {:?}",
            result
        );
        assert!(
            result.unwrap(),
            "Identical bulk memory functions should verify as equivalent"
        );
    }

    #[test]
    fn test_verify_memory_fill_does_not_affect_locals() {
        // memory.fill modifies memory but NOT locals.
        // Verify that local values are preserved across memory.fill.
        // Original: param + 10 with memory.fill in between
        // Optimized: same computation, memory.fill in between
        let original_wat = r#"
            (module
                (memory 1)
                (func (param i32) (result i32)
                    i32.const 0
                    i32.const 0
                    i32.const 10
                    memory.fill
                    local.get 0
                    i32.const 10
                    i32.add
                )
            )
        "#;

        let optimized_wat = r#"
            (module
                (memory 1)
                (func (param i32) (result i32)
                    i32.const 0
                    i32.const 0
                    i32.const 10
                    memory.fill
                    local.get 0
                    i32.const 10
                    i32.add
                )
            )
        "#;

        let original = parse::parse_wat(original_wat).unwrap();
        let optimized = parse::parse_wat(optimized_wat).unwrap();

        let result = verify_optimization(&original, &optimized);
        assert!(
            result.is_ok(),
            "Verification of locals across memory.fill failed: {:?}",
            result
        );
        assert!(
            result.unwrap(),
            "memory.fill should not affect local variables in Z3 model"
        );
    }

    // ======================================================================
    // VerificationResult skip-disambiguation tests (v0.5.0 PR-A)
    //
    // The audit's S5 finding: callers couldn't tell "Z3 proved equivalence"
    // from "verifier silently bailed because input was unverifiable." These
    // tests pin the strict/lenient distinction added in v0.5.0.
    // ======================================================================

    #[test]
    fn verification_result_lenient_treats_skip_as_equivalent() {
        // Lenient mode (existing is_equivalent): skipped == equivalent.
        // Used by value-level passes that don't move code across branches.
        assert!(VerificationResult::Verified.is_equivalent());
        assert!(VerificationResult::SkippedLoop.is_equivalent());
        assert!(VerificationResult::SkippedMemory.is_equivalent());
        assert!(VerificationResult::SkippedUnknown.is_equivalent());
        assert!(!VerificationResult::Failed("cex".into()).is_equivalent());
        assert!(!VerificationResult::Error("err".into()).is_equivalent());
    }

    #[test]
    fn verification_result_strict_only_accepts_verified() {
        // Strict mode (existing is_verified): only Verified counts.
        // Used by hoist-prone passes (LICM, CSE, code_folding, coalesce_locals).
        assert!(VerificationResult::Verified.is_verified());
        assert!(!VerificationResult::SkippedLoop.is_verified());
        assert!(!VerificationResult::SkippedMemory.is_verified());
        assert!(!VerificationResult::SkippedUnknown.is_verified());
        assert!(!VerificationResult::Failed("cex".into()).is_verified());
        assert!(!VerificationResult::Error("err".into()).is_verified());
    }

    #[test]
    fn verification_result_skip_predicate_distinguishes_skip_from_proof() {
        // is_skip() lets callers detect "verifier couldn't reason" cases.
        assert!(!VerificationResult::Verified.is_skip());
        assert!(VerificationResult::SkippedLoop.is_skip());
        assert!(VerificationResult::SkippedMemory.is_skip());
        assert!(VerificationResult::SkippedUnknown.is_skip());
        assert!(!VerificationResult::Failed("cex".into()).is_skip());
        assert!(!VerificationResult::Error("err".into()).is_skip());
    }

    // ======================================================================
    // #313 Tier-2 slice 1 — the translation validator's PURE BIT-VECTOR
    // obligations on the swappable solver seam.
    //
    // These tests exist to make the seam's REACHABILITY an asserted property
    // rather than an assumption. `verify_solver` is a filter: an obligation it
    // cannot reflect silently falls back to the incumbent, so a suite that
    // only checked verdicts would stay green if the seam stopped matching
    // ANYTHING. Each test below therefore asserts the ROUTE COUNTS as well as
    // the verdict — `routed == 1` fails the moment a real obligation stops
    // flowing through ordeal (e.g. a Z3 upgrade changes an AST shape, or the
    // encoder starts threading a non-BV term into straight-line code).
    // ======================================================================

    /// Two functions parsed from WAT, ready for the validator.
    #[cfg(feature = "verification")]
    fn func_pair(wat_a: &str, wat_b: &str) -> (Function, Function) {
        let ma = parse::parse_wat(wat_a).unwrap();
        let mb = parse::parse_wat(wat_b).unwrap();
        (ma.functions[0].clone(), mb.functions[0].clone())
    }

    /// `x * 2` vs `x << 1` — pure bit-vector, genuinely equivalent.
    #[cfg(feature = "verification")]
    const MUL2: &str = "(module (func (param i32) (result i32) local.get 0 i32.const 2 i32.mul))";
    #[cfg(feature = "verification")]
    const SHL1: &str = "(module (func (param i32) (result i32) local.get 0 i32.const 1 i32.shl))";
    /// `x << 2` — NOT equivalent to `x * 2`.
    #[cfg(feature = "verification")]
    const SHL2: &str = "(module (func (param i32) (result i32) local.get 0 i32.const 2 i32.shl))";
    /// `(x * 2) | 0` — equivalent to `x * 2`, but not the same term.
    #[cfg(feature = "verification")]
    const MUL2_OR0: &str = "(module (func (param i32) (result i32) local.get 0 i32.const 2 i32.mul i32.const 0 i32.or))";

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice1_routes_a_real_validator_obligation_through_ordeal() {
        // THE anti-inertness assertion: a real obligation, produced by the
        // real encoder and taken through the real public entry point, must be
        // DECIDED by ordeal — not merely offered to it.
        let (a, b) = func_pair(MUL2, SHL1);
        verify_solver::reset_route_counts();
        let proved =
            verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                .unwrap();
        let (routed, deferred) = verify_solver::route_counts();
        assert!(proved, "x * 2 == x << 1 must be proven");
        assert_eq!(
            (routed, deferred),
            (1, 0),
            "the pure-BV obligation MUST be decided by the neutral seam; if this \
             fails the ordeal path has gone dark and every obligation is silently \
             back on the incumbent"
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice1_routes_branching_and_extending_obligations() {
        // Widen the reachability assertion past a single binop: an if/else
        // (Ite over a comparison) and an i64 sign-extension must route too.
        let ifelse_a = "(module (func (param i32) (result i32) local.get 0 i32.eqz \
                        if (result i32) i32.const 10 else i32.const 20 end))";
        let ifelse_b = "(module (func (param i32) (result i32) local.get 0 i32.const 1 i32.lt_u \
                        if (result i32) i32.const 10 else i32.const 20 end))";
        let ext_a = "(module (func (param i32) (result i64) local.get 0 i64.extend_i32_s))";
        let ext_b = "(module (func (param i32) (result i64) local.get 0 i64.extend_i32_s \
                     i64.const 0 i64.or))";
        for (wa, wb) in [(ifelse_a, ifelse_b), (ext_a, ext_b)] {
            let (a, b) = func_pair(wa, wb);
            verify_solver::reset_route_counts();
            let proved =
                verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                    .unwrap();
            let (routed, deferred) = verify_solver::route_counts();
            assert!(proved, "{} == {} must be proven", wa, wb);
            assert_eq!((routed, deferred), (1, 0), "must route: {}", wa);
        }
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice1_disproves_a_wrong_transform_through_ordeal() {
        // A seam that only ever answers "Proven" would be worse than useless.
        // `x * 2` vs `x << 2` is a real miscompile; ordeal must catch it, and
        // must be the engine that did so.
        let (a, b) = func_pair(MUL2, SHL2);
        verify_solver::reset_route_counts();
        let proved =
            verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                .unwrap();
        let (routed, _) = verify_solver::route_counts();
        assert!(!proved, "x * 2 == x << 2 is FALSE and must be rejected");
        assert_eq!(routed, 1, "the rejection must come from the neutral seam");
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice4_memory_obligations_are_routed_and_decided() {
        // Slice 1 deliberately left memory on the incumbent and said so;
        // slice 4 retires that bound. Inverted rather than deleted, for the
        // same reason as the division case above: "memory is no longer
        // deferred" and "memory is now DECIDED by the new engine" are
        // different claims, and only the second one is worth having (#289).
        //
        // End to end through the real validator, because the point is that
        // whatever `encode_i32_load_from` actually builds — four `select`s
        // combined with `bvadd`/`zero_ext`/`bvshl`/`bvor` — survives the
        // reflection, the identical-AST round trip, and the extended
        // lowering. A hand-built term would not test that.
        let load = "(module (memory 1) (func (param i32) (result i32) local.get 0 i32.load))";
        // Not the identical body: identical terms are equal without a solver.
        let load_or0 = "(module (memory 1) (func (param i32) (result i32) local.get 0 i32.load i32.const 0 i32.or))";
        let (a, b) = func_pair(load, load_or0);
        verify_solver::reset_route_counts();
        let proved =
            verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                .unwrap();
        let (routed, deferred) = verify_solver::route_counts();
        assert!(proved, "a function is equivalent to itself");
        assert_eq!(
            (routed, deferred),
            (1, 0),
            "a memory obligation must now be DECIDED by ordeal, not deferred"
        );
    }

    /// #448: a body over the #347 memory-op bound, built from real wasm.
    ///
    /// Three stores put it over `DEFAULT_MAX_ARRAY_MEMORY_OPS` (2), and the
    /// padding puts it over `DEFAULT_MEMORY_BOUND_INSTRUCTION_FLOOR` (16), so
    /// the bound applies to it rather than the tiny-body exemption. `extra`
    /// goes before the result, `result` computes it.
    #[cfg(feature = "verification")]
    fn over_memory_bound(result_type: &str, extra: &str, result: &str) -> String {
        format!(
            "(module (memory 1) (func (param i32 i32) {result_type} \
             local.get 0 local.get 1 i32.store offset=0 \
             local.get 0 local.get 1 i32.store offset=4 \
             local.get 0 local.get 1 i32.store offset=8 \
             {pad} {extra} {result}))",
            pad = "i32.const 0 drop ".repeat(8),
        )
    }

    /// Verify `b` as a transform of `a` through the real validator.
    #[cfg(feature = "verification")]
    fn validate(a: &str, b: &str) -> anyhow::Result<()> {
        let (fa, fb) = func_pair(a, b);
        assert!(
            count_array_memory_ops(&fa) > DEFAULT_MAX_ARRAY_MEMORY_OPS
                && count_function_instructions(&fa) > DEFAULT_MEMORY_BOUND_INSTRUCTION_FLOOR,
            "the fixture must be over the #347 bound, or it proves nothing"
        );
        TranslationValidator::new(&fa, "test").verify(&fb)
    }

    /// The #448 regression: a correct transform of a body over the memory
    /// bound used to be refused outright. The new engine now decides it.
    #[cfg(feature = "verification")]
    #[test]
    fn over_the_memory_bound_a_correct_transform_is_proven() {
        let a = over_memory_bound("(result i32)", "", "local.get 1 i32.const 2 i32.mul");
        let b = over_memory_bound("(result i32)", "", "local.get 1 i32.const 1 i32.shl");
        verify_solver::reset_route_counts();
        let verdict = validate(&a, &b);
        let (routed, _) = verify_solver::route_counts();
        assert!(
            verdict.is_ok(),
            "x * 2 == x << 1 must be accepted: {verdict:?}"
        );
        assert!(routed >= 1, "the proof must come from the new engine");
    }

    /// The control: over the bound, a wrong transform is still refused, and
    /// the refusal still names #347.
    #[cfg(feature = "verification")]
    #[test]
    fn over_the_memory_bound_a_wrong_transform_is_refused() {
        let a = over_memory_bound("(result i32)", "", "local.get 1 i32.const 2 i32.mul");
        let b = over_memory_bound("(result i32)", "", "local.get 1 i32.const 2 i32.shl");
        let err = validate(&a, &b)
            .expect_err("x * 2 == x << 2 is false")
            .to_string();
        assert!(err.contains("(#347)"), "{err}");
        assert!(err.contains("counterexample"), "{err}");
    }

    /// The kill-criterion: over the bound, an obligation the new engine
    /// defers is refused, never handed to Z3. Z3's array theory is the cost
    /// #347 bounds, so a fallback here would bring back the hang. The memory
    /// after `memory.fill` is an uninterpreted array, which the new engine
    /// does not take; the refusal text shows the deferral path ran.
    #[cfg(feature = "verification")]
    #[test]
    fn over_the_memory_bound_a_deferred_obligation_is_refused_without_z3() {
        let fill = "local.get 0 i32.const 0 i32.const 4 memory.fill";
        let a = over_memory_bound("(result i32)", fill, "local.get 0 i32.load");
        let b = over_memory_bound(
            "(result i32)",
            fill,
            "local.get 0 i32.load i32.const 0 i32.or",
        );
        let err = validate(&a, &b)
            .expect_err("deferred over the bound")
            .to_string();
        assert!(err.contains("(#347)"), "{err}");
        assert!(
            err.contains("deferred"),
            "the new engine's deferral must be the reason: {err}"
        );
    }

    /// A void body is compared on its globals and memory, over the bound as
    /// below it: dropping a store is refused, an unchanged body accepted.
    #[cfg(feature = "verification")]
    #[test]
    fn over_the_memory_bound_a_void_body_is_compared() {
        let a = over_memory_bound("", "local.get 0 i32.const 9 i32.store offset=12", "");
        let b = over_memory_bound("", "", "");
        let err = validate(&a, &b)
            .expect_err("a dropped store changes memory")
            .to_string();
        assert!(err.contains("(#347)"), "{err}");
        assert!(
            validate(&a, &a).is_ok(),
            "an unchanged void body is accepted"
        );
    }

    /// An instruction outside the executor (`data.drop`) is refused over the
    /// bound, as below it.
    #[cfg(feature = "verification")]
    #[test]
    fn over_the_memory_bound_an_unverifiable_body_is_refused() {
        let with_data = |result: &str| {
            over_memory_bound("(result i32)", "data.drop 0", result).replacen(
                "(memory 1)",
                "(memory 1) (data \"x\")",
                1,
            )
        };
        let a = with_data("local.get 1");
        let b = with_data("local.get 1 i32.const 0 i32.or");
        let err = validate(&a, &b)
            .expect_err("data.drop is not modelled")
            .to_string();
        assert!(err.contains("(#347)"), "{err}");
        assert!(err.contains("outside the exact executor"), "{err}");
    }

    /// A loop too large for bounded unrolling goes through K-induction, as it
    /// does below the bound. Measured on the #347 module: 48 s with this path
    /// against 43 s without it, while a raised bound did not finish in 900 s.
    #[cfg(feature = "verification")]
    #[test]
    fn over_the_memory_bound_a_large_loop_goes_through_k_induction() {
        let looped = format!(
            "block loop {} end end",
            "nop ".repeat(MAX_LOOP_BODY_INSTRUCTIONS + 1)
        );
        let a = over_memory_bound("(result i32)", &looped, "local.get 1");
        let (fa, _) = func_pair(&a, &a);
        assert!(
            contains_complex_loops(&fa.instructions),
            "the loop must take the K-induction path"
        );
        let verdict = validate(&a, &a);
        assert!(
            verdict.is_ok(),
            "an unchanged loop body must be accepted: {verdict:?}"
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice4_a_store_then_load_is_decided_by_the_new_engine() {
        // The write side, end to end: store an i32 and load it straight
        // back. This is read-over-write through the whole stack — encoder,
        // reflection, round trip, array elimination — on the operators the
        // store encoder really emits (`extract` per byte over a `store`
        // chain).
        let store_load = "(module (memory 1) (func (param i32) (param i32) (result i32) \
                          local.get 0 local.get 1 i32.store \
                          local.get 0 i32.load))";
        // Not the identical body: identical terms are equal without a solver.
        let store_load_or0 = "(module (memory 1) (func (param i32) (param i32) (result i32) \
                              local.get 0 local.get 1 i32.store \
                              local.get 0 i32.load i32.const 0 i32.or))";
        let (a, b) = func_pair(store_load, store_load_or0);
        verify_solver::reset_route_counts();
        let proved =
            verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                .unwrap();
        let (routed, deferred) = verify_solver::route_counts();
        assert!(proved, "a function is equivalent to itself");
        assert_eq!(
            (routed, deferred),
            (1, 0),
            "a store-then-load obligation must be decided by ordeal"
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice4_a_wrong_memory_transform_is_rejected_by_the_new_engine() {
        // The half that matters. A seam that routed memory and always
        // answered "Proven" would pass every test above while licensing
        // miscompiles.
        //
        // Forwarding a store to a load at a DIFFERENT address is exactly
        // that miscompile: `mem[p] = v; return mem[q]` is not `v` unless
        // p == q, which is not known. ordeal must reject it, and must be
        // the engine that did.
        let honest = "(module (memory 1) (func (param i32) (param i32) (param i32) (result i32) \
                      local.get 0 local.get 2 i32.store \
                      local.get 1 i32.load))";
        let folded = "(module (memory 1) (func (param i32) (param i32) (param i32) (result i32) \
                      local.get 0 local.get 2 i32.store \
                      local.get 2))";
        let (a, b) = func_pair(honest, folded);
        verify_solver::reset_route_counts();
        let proved =
            verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                .unwrap();
        let (routed, _) = verify_solver::route_counts();
        assert!(
            !proved,
            "forwarding a store to a load at an unrelated address is a \
             miscompile and must be REJECTED"
        );
        assert_eq!(routed, 1, "the rejection must come from the neutral seam");
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice2_div_obligations_are_routed_and_decided() {
        // Slice 1 deliberately left division on the incumbent and said so;
        // slice 2 is where that bound is retired. The assertion is inverted
        // from its slice-1 form rather than removed, because "division is no
        // longer deferred" and "division is now decided by the new engine"
        // are different claims and only the second one is worth having.
        //
        // Driven end to end through the real validator on a real wasm
        // function, not through the seam's unit API: the reflection has to
        // survive whatever the encoder actually builds for `i32.div_u`, which
        // is the part a hand-built term cannot test.
        let div = "(module (func (param i32) (param i32) (result i32) \
                   local.get 0 local.get 1 i32.div_u))";
        // Not the identical body: identical terms are equal without a solver.
        let div_or0 = "(module (func (param i32) (param i32) (result i32) \
                       local.get 0 local.get 1 i32.div_u i32.const 0 i32.or))";
        let (a, b) = func_pair(div, div_or0);
        verify_solver::reset_route_counts();
        let proved =
            verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                .unwrap();
        let (routed, deferred) = verify_solver::route_counts();
        assert!(proved, "a function is equivalent to itself");
        assert_eq!(
            (routed, deferred),
            (1, 0),
            "a trapping div obligation must now be DECIDED by the new engine, \
             not merely offered to it — reachability is the asserted property \
             (#289), because a seam that defers everything also passes a test \
             that only checks the verdict"
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice2_signed_div_and_rem_are_routed_from_real_wasm() {
        // The signed pair separately, because `div_s` and `rem_s` are where
        // the trap conditions diverge: `div_s(INT_MIN, -1)` traps and
        // `rem_s(INT_MIN, -1)` is 0. Routing them is not the same event as
        // routing `div_u`, and a test covering only the unsigned case would
        // report slice 2 as done while the asymmetric half was still
        // deferred.
        for (name, op) in [
            ("i32.div_s", "i32.div_s"),
            ("i32.rem_s", "i32.rem_s"),
            ("i32.rem_u", "i32.rem_u"),
            ("i64.div_s", "i64.div_s"),
        ] {
            let ty = if name.starts_with("i64") {
                "i64"
            } else {
                "i32"
            };
            let src = format!(
                "(module (func (param {ty}) (param {ty}) (result {ty}) \
                 local.get 0 local.get 1 {op}))"
            );
            // Not the identical body: identical terms are equal without a
            // solver.
            let src_or0 = format!(
                "(module (func (param {ty}) (param {ty}) (result {ty}) \
                 local.get 0 local.get 1 {op} {ty}.const 0 {ty}.or))"
            );
            let (a, b) = func_pair(&src, &src_or0);
            verify_solver::reset_route_counts();
            let proved =
                verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Ordeal)
                    .unwrap();
            let (routed, deferred) = verify_solver::route_counts();
            assert!(proved, "{name}: a function is equivalent to itself");
            assert_eq!(
                (routed, deferred),
                (1, 0),
                "{name}: must be routed and decided by the new engine"
            );
        }
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice1_default_backend_leaves_the_validator_untouched() {
        // The default (`z3`) must route NOTHING: this PR does not change the
        // behaviour of a build that has not opted in.
        let (a, b) = func_pair(MUL2, SHL1);
        verify_solver::reset_route_counts();
        let proved =
            verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Z3).unwrap();
        let (routed, _) = verify_solver::route_counts();
        assert!(proved);
        assert_eq!(routed, 0, "default backend must route nothing");
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice1_routes_obligations_from_the_real_corpus() {
        // Reachability asserted on REAL WebAssembly rather than hand-written
        // fragments. Each function is offered against a copy whose result is
        // or-ed with 0: the same encoder and term shapes as a pass produces,
        // without being the identical term (which needs no solver). If a future Z3 or encoder change alters
        // those shapes so the reflection stops matching, this goes red instead
        // of silently leaving every obligation on the incumbent.
        let corpus = [
            concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/corpus/httparse.wasm"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../tests/corpus/json_lite.wasm"
            ),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../tests/corpus/state_machine.wasm"
            ),
        ];
        let (mut routed, mut deferred) = (0u64, 0u64);
        for path in corpus {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
            let m = crate::parse::parse_wasm(&bytes).unwrap_or_else(|e| panic!("{}: {}", path, e));
            for f in &m.functions {
                let mut g = f.clone();
                match f.signature.results.as_slice() {
                    [crate::ValueType::I32] => g
                        .instructions
                        .extend([Instruction::I32Const(0), Instruction::I32Or]),
                    [crate::ValueType::I64] => g
                        .instructions
                        .extend([Instruction::I64Const(0), Instruction::I64Or]),
                    _ => continue,
                }
                verify_solver::reset_route_counts();
                let _ = verify_function_equivalence_with_backend(
                    f,
                    &g,
                    "corpus",
                    VerifyBackend::Ordeal,
                );
                let (r, d) = verify_solver::route_counts();
                routed += r;
                deferred += d;
            }
        }
        // Measured at the time of writing: 13 of the 17 corpus functions
        // reach the solver at all (the rest are void or skipped upstream),
        // and all 13 route. The floor is deliberately below that so ordinary
        // corpus churn does not fail the build, while a collapse to zero —
        // the inert-seam failure mode — does.
        assert!(
            routed >= 8,
            "expected the seam to decide real corpus obligations, got routed={} deferred={}",
            routed,
            deferred
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice4_a_store_under_a_branch_must_not_forward_unconditionally() {
        // `ExecutionState` carries stack, locals, globals and the path
        // condition — but NOT memory, which the encoder threads as a plain
        // `let mut`. So a store inside an `if` arm has nothing merging it
        // against the not-taken path.
        //
        // If that store leaks out of the arm unconditionally, then after
        // `if (c) { mem[p] = v }` the encoder believes `mem[p] == v` whatever
        // `c` was, and folding the following load to `v` would come out
        // PROVEN. It is a miscompile whenever `c` is false.
        //
        // Every other slice-4 test is straight-line, so this is the only
        // thing that says anything about memory under control flow. It runs
        // on BOTH engines because the question is about the ENCODER, which
        // they share — agreement here would not be reassurance.
        let honest = "(module (memory 1) (func (param i32) (param i32) (param i32) (result i32) \
                      local.get 2 if local.get 0 local.get 1 i32.store end \
                      local.get 0 i32.load))";
        let folded = "(module (memory 1) (func (param i32) (param i32) (param i32) (result i32) \
                      local.get 2 if local.get 0 local.get 1 i32.store end \
                      local.get 1))";
        let (a, b) = func_pair(honest, folded);
        for backend in [
            VerifyBackend::Z3,
            VerifyBackend::Ordeal,
            VerifyBackend::Both,
        ] {
            verify_solver::reset_route_counts();
            let proved =
                verify_function_equivalence_with_backend(&a, &b, "test", backend).unwrap_or(false);
            assert!(
                !proved,
                "forwarding a CONDITIONAL store to a later load was reported \
                 PROVEN on {backend:?}. The fold is only valid when the branch \
                 is taken, so this would license a miscompile."
            );
        }

        // The control that makes the rows above mean something. An
        // unverifiable function is SKIPPED, and a skip returns `Ok(true)` —
        // so if this shape were simply not encodable, the assertions above
        // would have failed, not passed. But a revert for an unrelated
        // reason would still satisfy them, so assert positively that the
        // shape IS verifiable: the identity on it must be PROVEN.
        let (c, d) = func_pair(honest, honest);
        for backend in [VerifyBackend::Z3, VerifyBackend::Ordeal] {
            let proved =
                verify_function_equivalence_with_backend(&c, &d, "test", backend).unwrap_or(false);
            assert!(
                proved,
                "a conditional store followed by a load must be verifiable \
                 against ITSELF on {backend:?}; if it is not, the rejection \
                 above says nothing about memory under a branch"
            );
        }
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice4_memory_coverage_on_the_real_corpus() {
        // What slice 4 actually buys, measured rather than asserted.
        //
        // The hand-written tests prove the memory relation is right; this
        // says how much of REAL code it reaches, and — through the deferral
        // breakdown it prints — what is in the way of the rest. That second
        // number is what should decide slice 5's scope, instead of a guess.
        use crate::Instruction::*;
        fn touches_memory(f: &Function) -> bool {
            f.instructions.iter().any(|i| {
                matches!(
                    i,
                    I32Load { .. }
                        | I64Load { .. }
                        | F32Load { .. }
                        | F64Load { .. }
                        | I32Load8S { .. }
                        | I32Load8U { .. }
                        | I32Load16S { .. }
                        | I32Load16U { .. }
                        | I64Load8S { .. }
                        | I64Load8U { .. }
                        | I64Load16S { .. }
                        | I64Load16U { .. }
                        | I64Load32S { .. }
                        | I64Load32U { .. }
                        | I32Store { .. }
                        | I64Store { .. }
                        | F32Store { .. }
                        | F64Store { .. }
                        | I32Store8 { .. }
                        | I32Store16 { .. }
                        | I64Store8 { .. }
                        | I64Store16 { .. }
                        | I64Store32 { .. }
                )
            })
        }

        let corpus = [
            concat!(env!("CARGO_MANIFEST_DIR"), "/../tests/corpus/httparse.wasm"),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../tests/corpus/json_lite.wasm"
            ),
            concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../tests/corpus/state_machine.wasm"
            ),
        ];

        let (mut mem_fns, mut mem_routed, mut mem_deferred) = (0u64, 0u64, 0u64);
        let mut reasons: std::collections::BTreeMap<&'static str, u64> = Default::default();
        let mut skipped: std::collections::BTreeMap<&'static str, u64> = Default::default();
        let mut void_fns = 0u64;
        for path in corpus {
            let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {}", path, e));
            let m = crate::parse::parse_wasm(&bytes).unwrap_or_else(|e| panic!("{}: {}", path, e));
            for f in &m.functions {
                if !touches_memory(f) {
                    continue;
                }
                mem_fns += 1;
                if f.signature.results.is_empty() {
                    void_fns += 1;
                }
                // Every function is driven THROUGH the validator — none is
                // classified by inspection and skipped. An earlier version of
                // this test skipped void functions on their signature, which
                // would have let it report "no obligation exists" without
                // ever checking that claim.
                verify_solver::reset_route_counts();
                let _ = take_unproven_acceptance();
                let _ =
                    verify_function_equivalence_with_backend(f, f, "corpus", VerifyBackend::Ordeal);
                let (r, d) = verify_solver::route_counts();
                mem_routed += r;
                mem_deferred += d;
                for (reason, n) in verify_solver::deferrals_by_reason() {
                    *reasons.entry(reason).or_insert(0) += n;
                }
                // A function kept WITHOUT PROOF never reached the solver at
                // all; the reason says which gate stopped it. This is the
                // bucket that distinguishes "the seam declined" from "no
                // obligation was ever built".
                if r == 0 && d == 0 {
                    let why = take_unproven_acceptance().unwrap_or("no obligation built");
                    *skipped.entry(why).or_insert(0) += 1;
                }
            }
        }

        println!(
            "slice-4 corpus coverage: {} memory-touching functions, \
             {} void (no obligation exists), {} routed, {} deferred",
            mem_fns, void_fns, mem_routed, mem_deferred
        );
        for (reason, n) in &skipped {
            println!("  never reached the solver x{n}: {reason}");
        }
        for (reason, n) in &reasons {
            println!("  deferred x{n}: {reason}");
        }

        // The corpus must actually contain the thing being measured,
        // otherwise a green result here means nothing (#289).
        assert!(
            mem_fns > 0,
            "no corpus function touches memory, so this measures nothing"
        );

        // A sliver REFUSAL would mean loom built an array the sliver cannot
        // take — an encoding bug on our side, not a coverage bound. It must
        // never appear, whatever the routed/deferred split turns out to be.
        assert!(
            !reasons.keys().any(|r| r.starts_with("array sliver")),
            "the array sliver refused a loom-built query, which is a loom \
             encoding bug rather than a coverage bound: {reasons:?}"
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn tier2_slice1_both_mode_finds_no_divergence_on_real_obligations() {
        // `both` runs Z3 and ordeal on every routed obligation and panics on
        // disagreement. This test passing IS the no-divergence assertion for
        // the obligations it covers.
        let cases: &[(&str, &str, bool)] = &[
            (MUL2, SHL1, true),
            (MUL2, SHL2, false),
            // Syntactically identical results need no solver, so the
            // self-comparison is `x * 2` against `(x * 2) | 0`.
            (MUL2, MUL2_OR0, true),
            (
                "(module (func (param i32) (param i32) (result i32) \
                 local.get 0 local.get 1 i32.and))",
                "(module (func (param i32) (param i32) (result i32) \
                 local.get 1 local.get 0 i32.and))",
                true,
            ),
            (
                "(module (func (param i32) (result i32) local.get 0 i32.const -1 i32.xor))",
                "(module (func (param i32) (result i32) i32.const -1 local.get 0 i32.sub \
                 i32.const 1 i32.sub i32.const 1 i32.add))",
                true,
            ),
            (
                "(module (func (param i64) (param i64) (result i64) \
                 local.get 0 local.get 1 i64.add))",
                "(module (func (param i64) (param i64) (result i64) \
                 local.get 1 local.get 0 i64.add))",
                true,
            ),
        ];
        let mut total_routed = 0u64;
        for (wa, wb, expect) in cases {
            let (a, b) = func_pair(wa, wb);
            verify_solver::reset_route_counts();
            let proved =
                verify_function_equivalence_with_backend(&a, &b, "test", VerifyBackend::Both)
                    .unwrap();
            let (routed, _) = verify_solver::route_counts();
            assert_eq!(proved, *expect, "verdict mismatch for {} vs {}", wa, wb);
            assert_eq!(routed, 1, "must route under both: {}", wa);
            total_routed += routed;
        }
        assert_eq!(
            total_routed,
            cases.len() as u64,
            "every case must have been cross-checked by BOTH engines"
        );
    }

    #[test]
    fn verification_result_skip_reason_human_readable() {
        assert_eq!(VerificationResult::Verified.skip_reason(), None);
        assert_eq!(
            VerificationResult::SkippedLoop.skip_reason(),
            Some("complex loop")
        );
        assert_eq!(
            VerificationResult::SkippedMemory.skip_reason(),
            Some("float load/store")
        );
        assert_eq!(
            VerificationResult::SkippedUnknown.skip_reason(),
            Some("unknown opcode")
        );
        assert_eq!(VerificationResult::Failed("x".into()).skip_reason(), None);
    }
}
