//! #231 proof-carrying facts (P1) — the fact **SOURCE**.
//!
//! v1.3.0 shipped the `wsc.facts` EMITTER (wire format, drop-safety, the
//! `wsc.*` trust boundary) but nothing populated [`Module::facts`], so the
//! section never shipped a single premise. This module is the source: it
//! derives value-range facts that loom can justify **structurally**, attaches
//! them to loom [`Value`] terms in an [`OptimizationEnv`] (the value-keyed
//! carrier — never an instruction index), and resolves each one to the
//! producing operator in the FINAL operator sequence.
//!
//! # What makes a fact emittable here
//!
//! Every derivation below is a property of the value's own TERM SHAPE — it
//! does not depend on the program point, on local/memory state, or on any
//! path condition. That is what makes value-keying sound: the same term has
//! the same range at every occurrence, so a fact can never be "stale" for the
//! value it names. Anything that would need a program-point argument (a local
//! that was just assigned a constant, a loop-carried bound, a branch guard) is
//! deliberately NOT derived — loom does not guess a range.
//!
//! Sources wired in P1 (each a one-line soundness argument):
//!
//! 1. **Constants.** `i32.const c` / `i64.const c` evaluates to exactly `c`,
//!    so its range is `[c, c]`. This is the source that carries loom's own
//!    constant folding across the boundary: the fact is derived from the FINAL
//!    body, so a value the folder rewrote to `i32.const 5` reports `[5, 5]`
//!    at the operator the encoder actually emits.
//! 2. **Masks.** For a non-negative constant `K`, the set bits of `x & K` are
//!    a subset of the set bits of `K`, hence `0 <= (x & K) <= K` — for ANY
//!    `x`, signed or unsigned, and with no assumption about `x` whatsoever.
//!    `K >= 0` also makes the sign bit of the result provably clear, so the
//!    signed and unsigned readings coincide. This is the shape that makes a
//!    downstream array-bounds check elidable.
//! 3. **Booleans.** Every WebAssembly integer comparison and `eqz` yields
//!    exactly `0` or `1` (core spec: the relops return `i32` `0`/`1`), so the
//!    range is `[0, 1]`.
//!
//! # What is NOT wired, and why
//!
//! * **The #240 mid-end premise map.** `OptimizationEnv::value_max` /
//!   `assume_max` have no non-test callers: the optimizer never derives a
//!   bound of its own today, so there is nothing to harvest. Harvesting an
//!   always-empty map would be theatre, so it is not wired.
//! * **Arithmetic propagation** (`(x & 0xff) + (y & 0xff)` etc.). True over
//!   the integers, but wasm arithmetic wraps; a sound version needs an
//!   overflow argument per operator. Not derived — see charter: an unproven
//!   optimization is a bug waiting to happen.
//! * **Negative ranges.** A derived range whose `lo` is negative is DROPPED
//!   rather than emitted. The consumer's reading of a negative bound for a
//!   value it may treat as unsigned is not pinned down by the frozen wire
//!   format we can see from this repo, and `FactSet::unsigned_max` already
//!   applies exactly this conservatism internally. Facts we cannot state
//!   unambiguously are not stated.
//!
//! # Operator keying (the mis-keying footgun)
//!
//! `ModuleFact::value_id` is the index of the producing operator in the final
//! operator sequence — the sequence a consumer decodes out of the binary.
//! loom's `Function::instructions` is a NESTED representation: the encoder
//! flattens `Block`/`Loop`/`If` bodies (a `Block` becomes `block` + body +
//! `end`), and an `Instruction::End` in the list encodes to NO operator at
//! all. So an `instructions` index equals the emitted operator ordinal only
//! while every preceding entry encodes to exactly one operator.
//!
//! The walk therefore models a whitelist of operators that each encode 1:1,
//! and **stops at the first instruction outside it** — including all
//! structured control flow. Facts are only produced for the straight-line
//! prefix, where `index == ordinal` holds by construction. This is
//! conservative: it costs coverage inside control flow (see the PR notes),
//! never correctness.
//!
//! The collector runs on the FINAL module, immediately before encode, so a
//! fact never outlives the pass that could renumber it. A value deleted by a
//! later pass is simply never derived; nothing is re-pointed.

use crate::{Function, ImportKind, Instruction, Module, ModuleFact, ModuleFactKind};
use loom_shared::{
    Imm32, Imm64, OptimizationEnv, Value, ValueData, global_get, iadd32, iadd64, iand32, iand64,
    iconst32, iconst64, ieq32, ieq64, ieqz32, ieqz64, iges32, iges64, igeu32, igeu64, igts32,
    igts64, igtu32, igtu64, iles32, iles64, ileu32, ileu64, ilts32, ilts64, iltu32, iltu64, imul32,
    imul64, ine32, ine64, ior32, ior64, ishl32, ishl64, ishrs32, ishrs64, ishru32, ishru64, isub32,
    isub64, ixor32, ixor64, local_get, local_tee,
};

/// #231 fact SOURCE entry point: derive every value-range fact loom can
/// justify for `module`, resolved to the FINAL operator sequence.
///
/// Call this on the fully-optimized module immediately before
/// [`crate::encode::encode_wasm_with_facts`] — the resolution is only valid
/// against the body the encoder is about to emit. The result is deterministic
/// (functions in index order, operators in body order; no hash-map iteration
/// feeds the output order), which REQ-14 requires.
///
/// With the facts opt-in off this is never called, so the default output is
/// byte-identical to a build without a fact source.
pub fn collect_module_facts(module: &Module) -> Vec<ModuleFact> {
    // `func_index` is the FULL wasm function index: imported functions occupy
    // the low indices, local functions follow (the space the emitter and the
    // consumer both decode).
    let imported_funcs = module
        .imports
        .iter()
        .filter(|i| matches!(i.kind, ImportKind::Func(_)))
        .count() as u32;

    let mut facts = Vec::new();
    for (local_idx, func) in module.functions.iter().enumerate() {
        let func_index = imported_funcs + local_idx as u32;
        collect_function_facts(func, func_index, &mut facts);
    }
    facts
}

/// Derive the facts of a single function into `out`.
///
/// Walks the straight-line prefix of the body with an exact abstract stack.
/// Every modeled instruction encodes to exactly one operator, so the
/// `instructions` index of a push IS the operator ordinal of that push. The
/// first unmodeled instruction ends the walk (both the ordinal correspondence
/// and the stack model would become assumptions past it).
fn collect_function_facts(func: &Function, func_index: u32, out: &mut Vec<ModuleFact>) {
    // The value-keyed carrier: facts live on the Value, exactly as #231's
    // design requires. `candidates` records the resolution (value → producing
    // operator index) in deterministic walk order.
    let mut env = OptimizationEnv::new();
    // The abstract operand stack: one entry per live stack slot, holding the
    // term that produced it.
    let mut stack: Vec<Value> = Vec::new();
    let mut candidates: Vec<(u32, Value)> = Vec::new();

    for (idx, instr) in func.instructions.iter().enumerate() {
        let idx = idx as u32;

        // `None` => this instruction consumed operands but pushed nothing.
        // `break` => an unmodeled instruction (or a broken stack model): the
        // walk ends here and everything derived BEFORE it still stands.
        let produced: Option<Value> = match instr {
            // --- nullary producers (1 operator, push 1) ---
            Instruction::I32Const(c) => Some(iconst32(Imm32::from(*c))),
            Instruction::I64Const(c) => Some(iconst64(Imm64::from(*c))),
            Instruction::LocalGet(i) => Some(local_get(*i)),
            Instruction::GlobalGet(i) => Some(global_get(*i)),

            // --- pure consumers (1 operator, push 0) ---
            Instruction::Drop | Instruction::LocalSet(_) | Instruction::GlobalSet(_) => {
                if stack.pop().is_none() {
                    break; // stack model broke — stop rather than guess
                }
                None
            }

            // --- local.tee: pops and pushes back the SAME value ---
            Instruction::LocalTee(i) => {
                let Some(v) = stack.pop() else { break };
                Some(local_tee(*i, v))
            }

            // --- unary (pop 1, push 1) ---
            Instruction::I32Eqz => {
                let Some(v) = stack.pop() else { break };
                Some(ieqz32(v))
            }
            Instruction::I64Eqz => {
                let Some(v) = stack.pop() else { break };
                Some(ieqz64(v))
            }

            // --- binary (pop 2, push 1) ---
            _ => match binary_ctor(instr) {
                Some(ctor) => {
                    let Some(rhs) = stack.pop() else { break };
                    let Some(lhs) = stack.pop() else { break };
                    Some(ctor(lhs, rhs))
                }
                // Everything else — structured control flow, branches, calls,
                // loads/stores, floats, `End`, `Unknown` — ends the walk. Past
                // an unmodeled operator neither the operator ordinal nor the
                // stack contents are known, and loom does not guess.
                None => break,
            },
        };

        if let Some(value) = produced {
            // Structural derivation → record on the VALUE (the carrier), then
            // remember which operator produced it so the fact can be resolved
            // to a `value_id` below.
            if let Some((lo, hi)) = derive_range(&value) {
                if lo >= 0 {
                    env.assume_range(value.clone(), lo, hi);
                }
            }
            candidates.push((idx, value.clone()));
            stack.push(value);
        }
    }

    // Resolution: value → final operator index. A candidate with no recorded
    // premise contributes nothing (silence, not a guess).
    for (idx, value) in candidates {
        let Some(fact_set) = env.facts.get(&value) else {
            continue;
        };
        let Some((lo, hi)) = fact_set.value_range else {
            continue;
        };
        // Emit-time re-check of the non-negative discipline: a range that dips
        // negative is not stated (see module docs).
        if lo < 0 || lo > hi {
            continue;
        }
        out.push(ModuleFact {
            func_index,
            value_id: idx,
            kind: ModuleFactKind::ValueRange { lo, hi },
        });
    }
}

/// The binary (pop 2, push 1) instructions the walk models, mapped to their
/// term constructor. Every entry encodes to exactly ONE operator and has an
/// unconditional 2-in/1-out stack effect — the two properties the walk relies
/// on. The list is integer arithmetic, bitwise, shifts and comparisons; the
/// trapping (div/rem) and float operators are left out, which only ends the
/// walk earlier.
fn binary_ctor(instr: &Instruction) -> Option<fn(Value, Value) -> Value> {
    let ctor: fn(Value, Value) -> Value = match instr {
        Instruction::I32Add => iadd32,
        Instruction::I32Sub => isub32,
        Instruction::I32Mul => imul32,
        Instruction::I32And => iand32,
        Instruction::I32Or => ior32,
        Instruction::I32Xor => ixor32,
        Instruction::I32Shl => ishl32,
        Instruction::I32ShrS => ishrs32,
        Instruction::I32ShrU => ishru32,
        Instruction::I64Add => iadd64,
        Instruction::I64Sub => isub64,
        Instruction::I64Mul => imul64,
        Instruction::I64And => iand64,
        Instruction::I64Or => ior64,
        Instruction::I64Xor => ixor64,
        Instruction::I64Shl => ishl64,
        Instruction::I64ShrS => ishrs64,
        Instruction::I64ShrU => ishru64,
        Instruction::I32Eq => ieq32,
        Instruction::I32Ne => ine32,
        Instruction::I32LtS => ilts32,
        Instruction::I32LtU => iltu32,
        Instruction::I32GtS => igts32,
        Instruction::I32GtU => igtu32,
        Instruction::I32LeS => iles32,
        Instruction::I32LeU => ileu32,
        Instruction::I32GeS => iges32,
        Instruction::I32GeU => igeu32,
        Instruction::I64Eq => ieq64,
        Instruction::I64Ne => ine64,
        Instruction::I64LtS => ilts64,
        Instruction::I64LtU => iltu64,
        Instruction::I64GtS => igts64,
        Instruction::I64GtU => igtu64,
        Instruction::I64LeS => iles64,
        Instruction::I64LeU => ileu64,
        Instruction::I64GeS => iges64,
        Instruction::I64GeU => igeu64,
        _ => return None,
    };
    Some(ctor)
}

/// STRUCTURAL range derivation: the inclusive signed range of `v`, justified
/// by the shape of `v` alone. `None` means "loom cannot justify a range" —
/// which is emitted as silence, never as a guess.
///
/// Each arm's justification:
/// * `i32.const c` / `i64.const c` — the value IS `c`.
/// * `x & K` for a non-negative constant `K` (either operand; `and` is
///   commutative) — the result's set bits are a subset of `K`'s, so
///   `0 <= result <= K` for every `x`. With `K >= 0` the sign bit is clear,
///   so the signed and unsigned readings agree.
/// * relops / `eqz` — the WebAssembly integer comparisons produce `i32` `0`
///   or `1` by definition.
/// * `local.tee` — pushes back exactly the value it stored, so it inherits
///   that value's range.
fn derive_range(v: &Value) -> Option<(i64, i64)> {
    match v.data() {
        ValueData::I32Const { val } => {
            let c = val.0 as i64;
            Some((c, c))
        }
        ValueData::I64Const { val } => {
            let c = val.0;
            Some((c, c))
        }
        ValueData::I32And { lhs, rhs } => and_mask_bound(const_i32(lhs), const_i32(rhs)),
        ValueData::I64And { lhs, rhs } => and_mask_bound(const_i64(lhs), const_i64(rhs)),
        // Comparisons and eqz: 0 or 1, always.
        ValueData::I32Eq { .. }
        | ValueData::I32Ne { .. }
        | ValueData::I32LtS { .. }
        | ValueData::I32LtU { .. }
        | ValueData::I32GtS { .. }
        | ValueData::I32GtU { .. }
        | ValueData::I32LeS { .. }
        | ValueData::I32LeU { .. }
        | ValueData::I32GeS { .. }
        | ValueData::I32GeU { .. }
        | ValueData::I64Eq { .. }
        | ValueData::I64Ne { .. }
        | ValueData::I64LtS { .. }
        | ValueData::I64LtU { .. }
        | ValueData::I64GtS { .. }
        | ValueData::I64GtU { .. }
        | ValueData::I64LeS { .. }
        | ValueData::I64LeU { .. }
        | ValueData::I64GeS { .. }
        | ValueData::I64GeU { .. }
        | ValueData::I32Eqz { .. }
        | ValueData::I64Eqz { .. } => Some((0, 1)),
        ValueData::LocalTee { val, .. } => derive_range(val),
        _ => None,
    }
}

/// `[0, K]` for the tightest non-negative constant mask operand, or `None`
/// when neither operand is a non-negative constant (a negative mask leaves the
/// sign bit free — no non-negative bound is derivable, so nothing is claimed).
fn and_mask_bound(lhs: Option<i64>, rhs: Option<i64>) -> Option<(i64, i64)> {
    let bound = [lhs, rhs].into_iter().flatten().filter(|k| *k >= 0).min()?;
    Some((0, bound))
}

fn const_i32(v: &Value) -> Option<i64> {
    match v.data() {
        ValueData::I32Const { val } => Some(val.0 as i64),
        _ => None,
    }
}

fn const_i64(v: &Value) -> Option<i64> {
    match v.data() {
        ValueData::I64Const { val } => Some(val.0),
        _ => None,
    }
}

#[cfg(test)]
mod wsc_facts_source_tests {
    use super::*;
    use crate::{BlockType, FunctionSignature, Import, ValueType, encode};

    fn sig(params: Vec<ValueType>, results: Vec<ValueType>) -> FunctionSignature {
        FunctionSignature { params, results }
    }

    fn func(signature: FunctionSignature, instructions: Vec<Instruction>) -> Function {
        Function {
            name: None,
            signature,
            locals: vec![],
            instructions,
        }
    }

    /// A single-local-function module. `func_index` of that function is 0
    /// (no imports), which the tests assert against.
    fn module_with(func: Function) -> Module {
        Module {
            functions: vec![func],
            memories: vec![],
            tables: vec![],
            globals: vec![],
            types: vec![sig(vec![], vec![])],
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

    /// Decode `wasm` and return the flat operator sequence of local function
    /// `local_idx` — the sequence a CONSUMER sees, which is what `value_id`
    /// must index into. This is the authoritative check on operator keying:
    /// it never consults loom's own instruction list.
    fn decoded_operators(wasm: &[u8], local_idx: usize) -> Vec<String> {
        let mut bodies = Vec::new();
        let parser = wasmparser::Parser::new(0);
        for payload in parser.parse_all(wasm) {
            if let Ok(wasmparser::Payload::CodeSectionEntry(body)) = payload {
                let mut ops = Vec::new();
                let mut reader = body.get_operators_reader().expect("operators reader");
                while !reader.eof() {
                    let op = reader.read().expect("operator");
                    ops.push(format!("{:?}", op));
                }
                bodies.push(ops);
            }
        }
        bodies
            .into_iter()
            .nth(local_idx)
            .expect("function body present")
    }

    /// Every fact the module carries must name an operator whose decoded
    /// opcode is the one we claim produced the value. Mis-keying is the
    /// failure this guards.
    fn assert_facts_key_expected_operators(module: &Module, expected: &[(u32, &str)]) {
        let wasm = encode::encode_wasm_with_facts(module, true).expect("encode with facts");
        wasmparser::validate(&wasm).expect("facts-carrying module must be valid wasm");
        for fact in &module.facts {
            let ops = decoded_operators(&wasm, fact.func_index as usize);
            let op = ops
                .get(fact.value_id as usize)
                .unwrap_or_else(|| panic!("value_id {} names no operator", fact.value_id));
            let want = expected
                .iter()
                .find(|(id, _)| *id == fact.value_id)
                .unwrap_or_else(|| panic!("unexpected fact at value_id {}", fact.value_id))
                .1;
            assert!(
                op.starts_with(want),
                "fact at value_id {} claims a {} but the emitted operator is {}",
                fact.value_id,
                want,
                op
            );
        }
    }

    /// SOURCE 2 (masks): `x & 0xFF` is in `[0, 255]` for any `x`, and the
    /// fact must reach `wsc.facts` keyed to the `i32.and` operator.
    #[test]
    fn mask_yields_zero_to_mask_range_fact() {
        let mut module = module_with(func(
            sig(vec![ValueType::I32], vec![ValueType::I32]),
            vec![
                Instruction::LocalGet(0),   // 0
                Instruction::I32Const(255), // 1
                Instruction::I32And,        // 2  <- the masked value
            ],
        ));
        module.facts = collect_module_facts(&module);

        let mask_fact = module
            .facts
            .iter()
            .find(|f| f.kind == ModuleFactKind::ValueRange { lo: 0, hi: 255 })
            .expect("the masked value must carry a [0, 255] fact");
        assert_eq!(mask_fact.func_index, 0);
        assert_eq!(
            mask_fact.value_id, 2,
            "the fact must name the i32.and operator, not its operands"
        );
        // No fact claims a range for the unconstrained `local.get 0`.
        assert!(
            !module.facts.iter().any(|f| f.value_id == 0),
            "an unconstrained local.get must carry no fact"
        );

        // ...and it survives into the section, keyed to the real operator.
        assert_facts_key_expected_operators(&module, &[(1, "I32Const"), (2, "I32And")]);
        let wasm = encode::encode_wasm_with_facts(&module, true).expect("encode");
        let payload = extract_wsc_facts(&wasm).expect("wsc.facts section present");
        // kind=0x01, func=0, value_id=2, body_len=3, lo=0 (00), hi=255 (ff 01)
        assert!(
            payload
                .windows(6)
                .any(|w| w == [0x01, 0x00, 0x02, 0x03, 0x00, 0xff]),
            "schema-v1 encoding of the [0,255] fact must appear in the payload: {:02x?}",
            payload
        );
    }

    fn extract_wsc_facts(wasm: &[u8]) -> Option<Vec<u8>> {
        let parser = wasmparser::Parser::new(0);
        for payload in parser.parse_all(wasm) {
            if let Ok(wasmparser::Payload::CustomSection(reader)) = payload {
                if reader.name() == encode::WSC_FACTS_SECTION {
                    return Some(reader.data().to_vec());
                }
            }
        }
        None
    }

    /// SOURCE 1 (constants): a value loom's constant folder rewrote to
    /// `i32.const 5` must carry `[5, 5]`, keyed to the FOLDED operator index
    /// (0), not to the pre-folding index of the `i32.add` (2).
    #[test]
    fn constant_folded_value_yields_point_range_fact() {
        let mut module = module_with(func(
            sig(vec![], vec![ValueType::I32]),
            vec![
                Instruction::I32Const(2),
                Instruction::I32Const(3),
                Instruction::I32Add,
            ],
        ));
        crate::optimize::constant_folding(&mut module).expect("constant folding");
        assert_eq!(
            module.functions[0].instructions,
            vec![Instruction::I32Const(5)],
            "precondition: the folder must have produced a single i32.const 5"
        );

        module.facts = collect_module_facts(&module);
        assert_eq!(
            module.facts,
            vec![ModuleFact {
                func_index: 0,
                value_id: 0,
                kind: ModuleFactKind::ValueRange { lo: 5, hi: 5 },
            }],
            "the folded constant must carry [5,5] at its FINAL operator index"
        );
        assert_facts_key_expected_operators(&module, &[(0, "I32Const")]);
    }

    /// SOUNDNESS GUARD: values loom cannot bound get NO fact — silence, not a
    /// guess. Covered here: an unconstrained `local.get`, a sum of two
    /// unknowns, a NEGATIVE mask (the sign bit is free), a negative constant
    /// (whose unsigned reading at the boundary is not pinned down), and a
    /// shift by an unknown amount.
    #[test]
    fn unjustifiable_values_yield_no_fact() {
        let mut module = module_with(func(
            sig(vec![ValueType::I32, ValueType::I32], vec![]),
            vec![
                Instruction::LocalGet(0),    // 0  unknown
                Instruction::LocalGet(1),    // 1  unknown
                Instruction::I32Add,         // 2  sum of unknowns
                Instruction::I32Const(-256), // 3  negative constant
                Instruction::I32And,         // 4  mask with the sign bit SET
                Instruction::LocalGet(0),    // 5
                Instruction::LocalGet(1),    // 6
                Instruction::I32ShrU,        // 7  shift by an unknown amount
                Instruction::Drop,           // 8
                Instruction::Drop,           // 9
            ],
        ));
        module.facts = collect_module_facts(&module);
        assert!(
            module.facts.is_empty(),
            "no value here has a justifiable range; got {:?}",
            module.facts
        );
    }

    /// ORDINAL SAFETY: inside a structured body the `instructions` index is
    /// NOT the emitted operator ordinal (the encoder flattens `block`/`end`),
    /// so the walk stops at the first structured operator and derives nothing
    /// past it. The prefix before the block still yields its facts.
    #[test]
    fn walk_stops_at_structured_control_flow() {
        let mut module = module_with(func(
            sig(vec![ValueType::I32], vec![]),
            vec![
                Instruction::I32Const(7), // 0  prefix: a fact is derivable here
                Instruction::Drop,        // 1
                Instruction::Block {
                    block_type: BlockType::Empty,
                    body: vec![
                        Instruction::LocalGet(0),
                        Instruction::I32Const(255),
                        Instruction::I32And,
                        Instruction::Drop,
                    ],
                },
                Instruction::I32Const(9), // after the block: ordinal != index
                Instruction::Drop,
            ],
        ));
        module.facts = collect_module_facts(&module);
        assert_eq!(
            module.facts,
            vec![ModuleFact {
                func_index: 0,
                value_id: 0,
                kind: ModuleFactKind::ValueRange { lo: 7, hi: 7 },
            }],
            "only the straight-line prefix may produce facts"
        );
        // And the surviving fact really does name that operator in the binary.
        assert_facts_key_expected_operators(&module, &[(0, "I32Const")]);
    }

    /// RENUMBERING SAFETY: a fact-bearing value deleted by a later pass must
    /// produce NO fact, and the facts that DO survive must be re-keyed to the
    /// post-pass operator indices — never left pointing at the pre-pass ones
    /// (which now name different operators, or none at all). The collector
    /// runs on the FINAL body, so this is structural rather than a repair.
    #[test]
    fn value_deleted_by_a_later_pass_is_dropped_not_miskeyed() {
        let mut module = module_with(func(
            sig(vec![ValueType::I32], vec![ValueType::I32]),
            vec![
                Instruction::I32Const(255), // 0  fact [255,255] — about to die
                Instruction::Drop,          // 1
                Instruction::LocalGet(0),   // 2
                Instruction::I32Const(63),  // 3  fact [63,63]
                Instruction::I32And,        // 4  fact [0,63]
            ],
        ));
        // Facts as they stand BEFORE the deleting pass.
        let before = collect_module_facts(&module);
        assert_eq!(
            before
                .iter()
                .map(|f| (f.value_id, f.kind.clone()))
                .collect::<Vec<_>>(),
            vec![
                (0, ModuleFactKind::ValueRange { lo: 255, hi: 255 }),
                (3, ModuleFactKind::ValueRange { lo: 63, hi: 63 }),
                (4, ModuleFactKind::ValueRange { lo: 0, hi: 63 }),
            ],
            "precondition: three facts, keyed to the PRE-pass indices"
        );

        // A later pass deletes the dead constant, renumbering everything after
        // it by two.
        crate::optimize::eliminate_dead_code(&mut module).expect("DCE");
        assert_eq!(
            module.functions[0].instructions,
            vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(63),
                Instruction::I32And
            ],
            "precondition: the pass deleted the fact-bearing const+drop"
        );

        // Re-resolved against the FINAL body.
        module.facts = collect_module_facts(&module);
        assert!(
            !module
                .facts
                .iter()
                .any(|f| f.kind == ModuleFactKind::ValueRange { lo: 255, hi: 255 }),
            "the deleted value's fact must be dropped, not re-pointed: {:?}",
            module.facts
        );
        assert_eq!(
            module.facts,
            vec![
                ModuleFact {
                    func_index: 0,
                    value_id: 1,
                    kind: ModuleFactKind::ValueRange { lo: 63, hi: 63 },
                },
                ModuleFact {
                    func_index: 0,
                    value_id: 2,
                    kind: ModuleFactKind::ValueRange { lo: 0, hi: 63 },
                },
            ],
            "surviving facts must be re-keyed to the post-pass operator indices"
        );
        // The old value_id 4 does not even exist any more; the mask now sits at
        // 2 and the binary agrees.
        assert_facts_key_expected_operators(&module, &[(1, "I32Const"), (2, "I32And")]);
    }

    /// Imports shift the FULL function index; a fact must name the function
    /// the consumer will resolve. Also exercises the boolean source.
    #[test]
    fn facts_use_the_full_function_index_and_bool_range() {
        let mut module = module_with(func(
            sig(vec![ValueType::I32], vec![ValueType::I32]),
            vec![
                Instruction::LocalGet(0), // 0
                Instruction::I32Eqz,      // 1  -> [0,1]
            ],
        ));
        module.imports.push(Import {
            module: "env".to_string(),
            name: "a".to_string(),
            kind: ImportKind::Func(0),
        });

        module.facts = collect_module_facts(&module);
        assert_eq!(
            module.facts,
            vec![ModuleFact {
                func_index: 1,
                value_id: 1,
                kind: ModuleFactKind::ValueRange { lo: 0, hi: 1 },
            }],
            "eqz is 0-or-1 and the local function sits at FULL index 1"
        );
    }

    /// REQ-14: the fact set is deterministic — same module in, identical
    /// `Vec<ModuleFact>` out (no hash-map iteration order leaks into it).
    #[test]
    fn fact_collection_is_deterministic() {
        let module = module_with(func(
            sig(vec![ValueType::I32], vec![]),
            vec![
                Instruction::LocalGet(0),
                Instruction::I32Const(63),
                Instruction::I32And,
                Instruction::LocalGet(0),
                Instruction::I32Const(15),
                Instruction::I32And,
                Instruction::I32LtU,
                Instruction::Drop,
            ],
        ));
        let a = collect_module_facts(&module);
        let b = collect_module_facts(&module);
        assert_eq!(a, b, "fact collection must be deterministic");
        assert_eq!(
            a,
            vec![
                ModuleFact {
                    func_index: 0,
                    value_id: 1,
                    kind: ModuleFactKind::ValueRange { lo: 63, hi: 63 },
                },
                ModuleFact {
                    func_index: 0,
                    value_id: 2,
                    kind: ModuleFactKind::ValueRange { lo: 0, hi: 63 },
                },
                ModuleFact {
                    func_index: 0,
                    value_id: 4,
                    kind: ModuleFactKind::ValueRange { lo: 15, hi: 15 },
                },
                ModuleFact {
                    func_index: 0,
                    value_id: 5,
                    kind: ModuleFactKind::ValueRange { lo: 0, hi: 15 },
                },
                ModuleFact {
                    func_index: 0,
                    value_id: 6,
                    kind: ModuleFactKind::ValueRange { lo: 0, hi: 1 },
                },
            ],
            "facts are emitted in operator order"
        );
    }

    /// A module with NO derivable fact must not gain a `wsc.facts` section —
    /// the source cannot break the facts-absent byte-identity guarantee.
    #[test]
    fn no_derivable_fact_means_no_section() {
        let mut module = module_with(func(
            sig(vec![ValueType::I32], vec![]),
            vec![Instruction::LocalGet(0), Instruction::Drop],
        ));
        module.facts = collect_module_facts(&module);
        assert!(
            module.facts.is_empty(),
            "the SOURCE must derive nothing here; got {:?}",
            module.facts
        );
        let with_facts = encode::encode_wasm_with_facts(&module, true).expect("encode");
        let default = encode::encode_wasm(&module).expect("encode");
        assert!(
            extract_wsc_facts(&with_facts).is_none(),
            "a module the source could not bound must carry no wsc.facts section"
        );
        assert_eq!(
            with_facts, default,
            "a module with no derivable fact must encode byte-identically"
        );
    }

    /// THE ORDINAL CLAIM, locked against the encoder. The walk keys facts by
    /// the `instructions` index, which is only the emitted operator ordinal
    /// because every MODELED instruction encodes to exactly one operator. If a
    /// future encoder change makes one of them emit zero or two operators
    /// (as `End` and `Block` already do — which is why they are not modeled),
    /// every fact after it in that body would be mis-keyed. This test encodes
    /// a body per modeled operator group and asserts the DECODED operator
    /// count is `instructions.len()` plus the single appended function `end`.
    #[test]
    fn every_modeled_instruction_encodes_one_to_one() {
        use Instruction::*;
        let i32_binaries = [
            I32Add, I32Sub, I32Mul, I32And, I32Or, I32Xor, I32Shl, I32ShrS, I32ShrU, I32Eq, I32Ne,
            I32LtS, I32LtU, I32GtS, I32GtU, I32LeS, I32LeU, I32GeS, I32GeU,
        ];
        let i64_binaries = [
            I64Add, I64Sub, I64Mul, I64And, I64Or, I64Xor, I64Shl, I64ShrS, I64ShrU,
        ];
        // i64 relops consume two i64s and produce an i32.
        let i64_relops = [
            I64Eq, I64Ne, I64LtS, I64LtU, I64GtS, I64GtU, I64LeS, I64LeU, I64GeS, I64GeU,
        ];

        let mut cases: Vec<Vec<Instruction>> = Vec::new();
        for op in i32_binaries {
            cases.push(vec![I32Const(1), I32Const(2), op, Drop]);
        }
        for op in i64_binaries.into_iter().chain(i64_relops) {
            cases.push(vec![I64Const(1), I64Const(2), op, Drop]);
        }
        cases.push(vec![I32Const(1), I32Eqz, Drop]);
        cases.push(vec![I64Const(1), I64Eqz, Drop]);
        // local.get / local.set / local.tee over the i32 parameter.
        cases.push(vec![
            I32Const(1),
            LocalSet(0),
            LocalGet(0),
            LocalTee(0),
            Drop,
        ]);
        // global.get / global.set over a mutable i32 global.
        cases.push(vec![I32Const(1), GlobalSet(0), GlobalGet(0), Drop]);

        for body in cases {
            let n = body.len();
            let mut module = module_with(func(sig(vec![ValueType::I32], vec![]), body.clone()));
            module.globals.push(crate::Global {
                value_type: ValueType::I32,
                mutable: true,
                init: vec![I32Const(0)],
            });
            let wasm = encode::encode_wasm(&module).expect("encode");
            wasmparser::validate(&wasm).expect("case must be valid wasm");
            let ops = decoded_operators(&wasm, 0);
            assert_eq!(
                ops.len(),
                n + 1,
                "body {:?} must encode to exactly one operator each (plus the \
                 appended function end); decoded {:?}",
                body,
                ops
            );
        }
    }
}
