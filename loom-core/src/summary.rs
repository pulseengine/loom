//! Function-summary interprocedural analysis (IPA).
//!
//! Computes per-function summaries so downstream passes can reason across
//! `Call` boundaries: `vacuum` folds `Call f; Drop` and CSE deduplicates
//! identical calls when the summary allows it.
//!
//! ## Definitions
//!
//! - **Pure** (`is_pure`): the function writes no state. It contains no
//!   store, `global.set`, `memory.grow`/`fill`/`copy`/`init`, `data.drop`,
//!   `call_indirect`, unknown instruction or import call, and every direct
//!   callee is pure. Reading memory or globals is allowed; see `reads_state`.
//!
//! - **Reads state** (`reads_state`): the result may depend on something
//!   other than the arguments: a load, `global.get`, `memory.size`, a
//!   memory instruction, `call_indirect`, an unknown instruction, an import
//!   call, or a callee that reads state. Two calls with equal arguments
//!   return equal results only when the callee is pure and does not read
//!   state.
//!
//! - **Completes normally** (`is_no_trap`): every call returns. The function
//!   contains no instruction that can trap (`unreachable`, a load or store,
//!   integer `div`/`rem`, non-saturating float-to-int truncation, a memory
//!   instruction, `call_indirect`, an unknown instruction), no `loop` (which
//!   may not terminate), no import call, and every direct callee completes
//!   normally. A function on a call cycle never completes normally by this
//!   definition: unbounded recursion exhausts the call stack, which traps.
//!
//! ## Fixpoints
//!
//! `is_pure` and `reads_state` are greatest/least fixpoints that start from
//! each function's own instructions and propagate along call edges until
//! nothing changes. `is_no_trap` is a least fixpoint: a function is proven
//! to complete normally only once all of its callees are, so a call cycle is
//! never proven.
//!
//! ## Indexing
//!
//! `Call(idx)` indexes the function space, where imported functions come
//! first. [`compute_module_summaries`] returns one entry per *local*
//! function; [`compute_call_summaries`] returns one entry per function-space
//! index (imports get [`FunctionSummary::OPAQUE`]) and is the one to index
//! with a `Call` operand.

use crate::{Function, ImportKind, Instruction, Module, ValueType};

/// Per-function summary computed by the IPA.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FunctionSummary {
    /// The function writes no state (see the module docs).
    pub is_pure: bool,
    /// Every call of the function returns: no trap, no loop, no recursion.
    pub is_no_trap: bool,
    /// The result may depend on memory, globals or the host, not only on
    /// the arguments.
    pub reads_state: bool,
}

impl FunctionSummary {
    /// The summary of a function nothing is known about (an import, or a
    /// call index out of range).
    pub const OPAQUE: FunctionSummary = FunctionSummary {
        is_pure: false,
        is_no_trap: false,
        reads_state: true,
    };

    /// Removing a call whose result is dropped preserves behavior.
    pub fn is_removable(&self) -> bool {
        self.is_pure && self.is_no_trap
    }

    /// Two calls with equal arguments return equal results and either can
    /// be removed: the call is a function of its arguments.
    pub fn is_deterministic(&self) -> bool {
        self.is_pure && self.is_no_trap && !self.reads_state
    }
}

/// Number of imported functions, which precede local functions in the
/// function index space.
pub fn num_imported_funcs(module: &Module) -> usize {
    module
        .imports
        .iter()
        .filter(|i| matches!(i.kind, ImportKind::Func(_)))
        .count()
}

/// Summaries indexed by function-space index: one
/// [`FunctionSummary::OPAQUE`] per imported function, then one entry per
/// local function. Index this with a `Call` operand.
pub fn compute_call_summaries(module: &Module) -> Vec<FunctionSummary> {
    let mut out = vec![FunctionSummary::OPAQUE; num_imported_funcs(module)];
    out.extend(compute_module_summaries(module));
    out
}

/// `(params, results)` for every function-space index, imports first, so it
/// lines up with [`compute_call_summaries`]. An import whose type index is
/// out of range gets `None`.
pub fn call_signatures(module: &Module) -> Vec<Option<(Vec<ValueType>, Vec<ValueType>)>> {
    let mut out: Vec<Option<(Vec<ValueType>, Vec<ValueType>)>> = module
        .imports
        .iter()
        .filter_map(|i| match i.kind {
            ImportKind::Func(t) => Some(
                module
                    .types
                    .get(t as usize)
                    .map(|sig| (sig.params.clone(), sig.results.clone())),
            ),
            _ => None,
        })
        .collect();
    out.extend(
        module
            .functions
            .iter()
            .map(|f| Some((f.signature.params.clone(), f.signature.results.clone()))),
    );
    out
}

/// Summaries for every local function. Index `i` matches
/// `module.functions[i]`, not the function-space index; see
/// [`compute_call_summaries`].
pub fn compute_module_summaries(module: &Module) -> Vec<FunctionSummary> {
    let n = module.functions.len();
    let ni = num_imported_funcs(module);

    // Step 1: each function's own instructions. A call to an import or to
    // an out-of-range index is opaque and is folded in here; calls to local
    // functions become edges (local indices) for step 2.
    let mut own: Vec<FunctionSummary> = Vec::with_capacity(n);
    let mut edges: Vec<Vec<usize>> = Vec::with_capacity(n);
    for func in &module.functions {
        let (mut s, calls) = scan_function(func);
        let mut local = Vec::new();
        for idx in calls {
            match (idx as usize).checked_sub(ni) {
                Some(l) if l < n => local.push(l),
                _ => {
                    s.is_pure = false;
                    s.is_no_trap = false;
                    s.reads_state = true;
                }
            }
        }
        own.push(s);
        edges.push(local);
    }

    // Step 2a: purity and state reads propagate from callees until nothing
    // changes. Each step only flips is_pure true→false or reads_state
    // false→true, so this terminates.
    let mut is_pure: Vec<bool> = own.iter().map(|s| s.is_pure).collect();
    let mut reads_state: Vec<bool> = own.iter().map(|s| s.reads_state).collect();
    loop {
        let mut changed = false;
        for i in 0..n {
            for &c in &edges[i] {
                if is_pure[i] && !is_pure[c] {
                    is_pure[i] = false;
                    changed = true;
                }
                if !reads_state[i] && reads_state[c] {
                    reads_state[i] = true;
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
    }

    // Step 2b: completes-normally is a LEAST fixpoint. Start with nothing
    // proven; a function is proven once its own instructions are clean and
    // every callee is already proven. A call cycle is never proven, because
    // no member of it can be proven before the others.
    let mut is_no_trap = vec![false; n];
    loop {
        let mut changed = false;
        for i in 0..n {
            if !is_no_trap[i] && own[i].is_no_trap && edges[i].iter().all(|&c| is_no_trap[c]) {
                is_no_trap[i] = true;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    (0..n)
        .map(|i| FunctionSummary {
            is_pure: is_pure[i],
            is_no_trap: is_no_trap[i],
            reads_state: reads_state[i],
        })
        .collect()
}

/// Scan a function's instruction tree. Returns its own summary (callees
/// not yet considered) and the function-space indices it calls directly.
fn scan_function(func: &Function) -> (FunctionSummary, Vec<u32>) {
    let mut s = FunctionSummary {
        is_pure: true,
        is_no_trap: true,
        reads_state: false,
    };
    let mut callees: Vec<u32> = Vec::new();
    scan_body(&func.instructions, &mut s, &mut callees);
    (s, callees)
}

/// The match is exhaustive on purpose: a new `Instruction` variant does not
/// compile until someone decides how it affects the summary.
fn scan_body(instructions: &[Instruction], s: &mut FunctionSummary, callees: &mut Vec<u32>) {
    use Instruction as I;
    for instr in instructions {
        match instr {
            I::Call(idx) => callees.push(*idx),

            // Nothing is known: writes, reads, traps.
            I::CallIndirect { .. } | I::Unknown(..) => {
                s.is_pure = false;
                s.is_no_trap = false;
                s.reads_state = true;
            }

            I::Unreachable => {
                s.is_pure = false;
                s.is_no_trap = false;
            }

            // Stores write memory and trap out of bounds.
            I::I32Store { .. }
            | I::I64Store { .. }
            | I::F32Store { .. }
            | I::F64Store { .. }
            | I::I32Store8 { .. }
            | I::I32Store16 { .. }
            | I::I64Store8 { .. }
            | I::I64Store16 { .. }
            | I::I64Store32 { .. } => {
                s.is_pure = false;
                s.is_no_trap = false;
            }

            // Bulk memory and memory.grow write memory (or segment state),
            // depend on the current size, and can trap.
            I::MemoryGrow(..)
            | I::MemoryFill(..)
            | I::MemoryCopy { .. }
            | I::MemoryInit { .. }
            | I::DataDrop(..) => {
                s.is_pure = false;
                s.is_no_trap = false;
                s.reads_state = true;
            }

            // Loads read memory and trap out of bounds.
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
            | I::I64Load32U { .. } => {
                s.is_no_trap = false;
                s.reads_state = true;
            }

            I::GlobalGet(..) | I::MemorySize(..) => s.reads_state = true,
            I::GlobalSet(..) => s.is_pure = false,

            // Trap on a zero divisor (and signed overflow for div_s).
            I::I32DivS
            | I::I32DivU
            | I::I32RemS
            | I::I32RemU
            | I::I64DivS
            | I::I64DivU
            | I::I64RemS
            | I::I64RemU => s.is_no_trap = false,

            // Trap on NaN or an out-of-range value.
            I::I32TruncF32S
            | I::I32TruncF32U
            | I::I32TruncF64S
            | I::I32TruncF64U
            | I::I64TruncF32S
            | I::I64TruncF32U
            | I::I64TruncF64S
            | I::I64TruncF64U => s.is_no_trap = false,

            // A loop may not terminate.
            I::Loop { body, .. } => {
                s.is_no_trap = false;
                scan_body(body, s, callees);
            }
            I::Block { body, .. } => scan_body(body, s, callees),
            I::If {
                then_body,
                else_body,
                ..
            } => {
                scan_body(then_body, s, callees);
                scan_body(else_body, s, callees);
            }

            // Locals, constants, branches and total numeric operations:
            // no write, no state read, no trap.
            I::LocalGet(..)
            | I::LocalSet(..)
            | I::LocalTee(..)
            | I::I32Const(..)
            | I::I64Const(..)
            | I::F32Const(..)
            | I::F64Const(..)
            | I::Br(..)
            | I::BrIf(..)
            | I::BrTable { .. }
            | I::Return
            | I::Nop
            | I::Drop
            | I::End
            | I::Select
            | I::I32Add
            | I::I32Sub
            | I::I32Mul
            | I::I32And
            | I::I32Or
            | I::I32Xor
            | I::I32Shl
            | I::I32ShrS
            | I::I32ShrU
            | I::I32Rotl
            | I::I32Rotr
            | I::I64Add
            | I::I64Sub
            | I::I64Mul
            | I::I64And
            | I::I64Or
            | I::I64Xor
            | I::I64Shl
            | I::I64ShrS
            | I::I64ShrU
            | I::I64Rotl
            | I::I64Rotr
            | I::I32Eq
            | I::I32Ne
            | I::I32LtS
            | I::I32LtU
            | I::I32GtS
            | I::I32GtU
            | I::I32LeS
            | I::I32LeU
            | I::I32GeS
            | I::I32GeU
            | I::I64Eq
            | I::I64Ne
            | I::I64LtS
            | I::I64LtU
            | I::I64GtS
            | I::I64GtU
            | I::I64LeS
            | I::I64LeU
            | I::I64GeS
            | I::I64GeU
            | I::I32Eqz
            | I::I32Clz
            | I::I32Ctz
            | I::I32Popcnt
            | I::I64Eqz
            | I::I64Clz
            | I::I64Ctz
            | I::I64Popcnt
            | I::F32Add
            | I::F32Sub
            | I::F32Mul
            | I::F32Div
            | I::F32Min
            | I::F32Max
            | I::F32Copysign
            | I::F32Abs
            | I::F32Neg
            | I::F32Ceil
            | I::F32Floor
            | I::F32Trunc
            | I::F32Nearest
            | I::F32Sqrt
            | I::F32Eq
            | I::F32Ne
            | I::F32Lt
            | I::F32Gt
            | I::F32Le
            | I::F32Ge
            | I::F64Add
            | I::F64Sub
            | I::F64Mul
            | I::F64Div
            | I::F64Min
            | I::F64Max
            | I::F64Copysign
            | I::F64Abs
            | I::F64Neg
            | I::F64Ceil
            | I::F64Floor
            | I::F64Trunc
            | I::F64Nearest
            | I::F64Sqrt
            | I::F64Eq
            | I::F64Ne
            | I::F64Lt
            | I::F64Gt
            | I::F64Le
            | I::F64Ge
            | I::I32TruncSatF32S
            | I::I32TruncSatF32U
            | I::I32TruncSatF64S
            | I::I32TruncSatF64U
            | I::I64TruncSatF32S
            | I::I64TruncSatF32U
            | I::I64TruncSatF64S
            | I::I64TruncSatF64U
            | I::F32ConvertI32S
            | I::F32ConvertI32U
            | I::F32ConvertI64S
            | I::F32ConvertI64U
            | I::F64ConvertI32S
            | I::F64ConvertI32U
            | I::F64ConvertI64S
            | I::F64ConvertI64U
            | I::F32DemoteF64
            | I::F64PromoteF32
            | I::I32ReinterpretF32
            | I::I64ReinterpretF64
            | I::F32ReinterpretI32
            | I::F64ReinterpretI64
            | I::I64ExtendI32S
            | I::I64ExtendI32U
            | I::I32WrapI64
            | I::I32Extend8S
            | I::I32Extend16S
            | I::I64Extend8S
            | I::I64Extend16S
            | I::I64Extend32S => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parse;

    fn analyze_wat(wat: &str) -> Vec<FunctionSummary> {
        let module = parse::parse_wat(wat).expect("parse");
        compute_module_summaries(&module)
    }

    #[test]
    fn test_pure_arithmetic_function_is_pure_and_no_trap() {
        let wat = r#"(module
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure, "arithmetic-only function must be pure");
        assert!(s[0].is_no_trap, "arithmetic-only function must be no-trap");
    }

    #[test]
    fn test_store_makes_function_impure() {
        let wat = r#"(module
            (memory 1)
            (func (param i32 i32)
                local.get 0
                local.get 1
                i32.store
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "i32.store must mark function impure");
        assert!(!s[0].is_no_trap, "i32.store must mark function may-trap");
    }

    #[test]
    fn test_load_is_pure_but_may_trap() {
        let wat = r#"(module
            (memory 1)
            (func (param i32) (result i32)
                local.get 0
                i32.load
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure, "i32.load is pure (no observable write)");
        assert!(!s[0].is_no_trap, "i32.load can fault on bad address");
    }

    #[test]
    fn test_divide_is_pure_but_may_trap() {
        let wat = r#"(module
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.div_s
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure);
        assert!(!s[0].is_no_trap, "i32.div_s traps on divisor 0");
    }

    #[test]
    fn test_global_set_makes_function_impure() {
        let wat = r#"(module
            (global $g (mut i32) (i32.const 0))
            (func (param i32)
                local.get 0
                global.set $g
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "global.set must mark function impure");
    }

    #[test]
    fn test_pure_caller_of_pure_callee_is_pure() {
        // Two-function module: $callee is pure, $caller calls $callee.
        let wat = r#"(module
            (func $callee (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $caller (param i32) (result i32)
                local.get 0
                call $callee
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure && s[0].is_no_trap, "callee is pure + no-trap");
        assert!(
            s[1].is_pure && s[1].is_no_trap,
            "caller of pure+no-trap callee must inherit"
        );
    }

    #[test]
    fn test_impure_propagates_through_call() {
        // Caller is intrinsically pure but calls an impure callee.
        let wat = r#"(module
            (memory 1)
            (func $impure (param i32)
                local.get 0
                i32.const 1
                i32.store
            )
            (func $caller (param i32)
                local.get 0
                call $impure
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "callee performs store, must be impure");
        assert!(
            !s[1].is_pure,
            "caller of impure callee must propagate impurity"
        );
    }

    #[test]
    fn test_call_indirect_marks_caller_impure_and_may_trap() {
        let wat = r#"(module
            (table 1 funcref)
            (type $sig (func (param i32) (result i32)))
            (func (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call_indirect (type $sig)
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(
            !s[0].is_pure && !s[0].is_no_trap,
            "call_indirect can't resolve target — conservative"
        );
    }

    #[test]
    fn test_mutual_recursion_converges() {
        // Two functions calling each other; both pure intrinsically.
        let wat = r#"(module
            (func $a (param i32) (result i32)
                local.get 0
                i32.eqz
                if (result i32)
                    i32.const 0
                else
                    local.get 0
                    i32.const 1
                    i32.sub
                    call $b
                end
            )
            (func $b (param i32) (result i32)
                local.get 0
                call $a
            )
        )"#;
        let s = analyze_wat(wat);
        // Both are pure, but neither completes normally on every input:
        // $a(0x7fffffff) recurses 2^31 deep and exhausts the call stack.
        assert!(s[0].is_pure && !s[0].is_no_trap);
        assert!(s[1].is_pure && !s[1].is_no_trap);
    }

    #[test]
    fn test_self_recursion_may_trap() {
        // Unconditional self-recursion traps with stack exhaustion; removing
        // `call $f; drop` would remove that trap.
        let wat = r#"(module
            (func $f (result i32) call $f)
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure);
        assert!(!s[0].is_no_trap, "a call cycle never completes normally");
    }

    #[test]
    fn test_call_to_import_is_opaque() {
        // Local function 0 is pure, and import 0 shares its call index. A
        // caller of the import must not inherit local function 0's summary.
        let wat = r#"(module
            (import "env" "tick" (func $tick (result i32)))
            (func $seven (result i32) i32.const 7)
            (func $calls_import (result i32) call $tick)
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure && s[0].is_no_trap && !s[0].reads_state);
        assert!(!s[1].is_pure, "an import call may write host state");
        assert!(!s[1].is_no_trap, "an import call may trap");
        assert!(s[1].reads_state, "an import result depends on the host");
    }

    #[test]
    fn test_callee_index_skips_imports() {
        // With one import, `call 2` is local function 1 ($store), which is
        // impure. Reading local slot 2 instead would see $pure2.
        let wat = r#"(module
            (import "env" "tick" (func $tick (result i32)))
            (memory 1)
            (func $caller (result i32) call $store)
            (func $store (result i32) i32.const 0 i32.const 1 i32.store i32.const 0)
            (func $pure2 (result i32) i32.const 2)
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[1].is_pure);
        assert!(s[2].is_pure);
        assert!(!s[0].is_pure, "caller of the impure $store is impure");
    }

    #[test]
    fn test_call_summaries_are_function_space_indexed() {
        let wat = r#"(module
            (import "env" "tick" (func $tick (result i32)))
            (func $seven (result i32) i32.const 7)
        )"#;
        let module = crate::parse::parse_wat(wat).expect("parse");
        let cs = compute_call_summaries(&module);
        assert_eq!(cs.len(), 2);
        assert_eq!(cs[0], FunctionSummary::OPAQUE);
        assert!(cs[1].is_deterministic());
        let sigs = call_signatures(&module);
        assert_eq!(sigs.len(), 2);
        assert!(sigs.iter().all(|s| s.is_some()));
    }

    #[test]
    fn test_float_to_int_truncation_may_trap() {
        let wat = r#"(module
            (func $t (param f32) (result i32) local.get 0 i32.trunc_f32_s)
            (func $sat (param f32) (result i32) local.get 0 i32.trunc_sat_f32_s)
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_no_trap, "i32.trunc_f32_s traps on NaN");
        assert!(s[1].is_no_trap, "the saturating form does not trap");
    }

    #[test]
    fn test_loop_may_not_complete() {
        let wat = r#"(module
            (func $spin (result i32) (loop $l br $l) i32.const 0)
        )"#;
        let s = analyze_wat(wat);
        assert!(s[0].is_pure);
        assert!(!s[0].is_no_trap, "a loop may not terminate");
    }

    #[test]
    fn test_state_reads_are_not_deterministic() {
        let wat = r#"(module
            (global $g (mut i32) (i32.const 0))
            (memory 1)
            (func $rd (result i32) global.get $g)
            (func $sz (result i32) memory.size)
            (func $via (result i32) call $rd)
            (func $args (param i32) (result i32) local.get 0 i32.const 1 i32.add)
        )"#;
        let s = analyze_wat(wat);
        for (i, name) in [(0, "global.get"), (1, "memory.size"), (2, "callee reading")] {
            assert!(s[i].is_removable(), "{name}: removable when dropped");
            assert!(
                !s[i].is_deterministic(),
                "{name}: not a function of its args"
            );
        }
        assert!(s[3].is_deterministic());
    }

    #[test]
    fn test_recursion_with_impure_self() {
        // Function that recursively calls itself AND has an impure op.
        // The self-recursion shouldn't mask the intrinsic impurity.
        let wat = r#"(module
            (memory 1)
            (func $self_call (param i32)
                local.get 0
                i32.const 1
                i32.store
                local.get 0
                call $self_call
            )
        )"#;
        let s = analyze_wat(wat);
        assert!(!s[0].is_pure, "self-recursion does not save an impure body");
    }
}
