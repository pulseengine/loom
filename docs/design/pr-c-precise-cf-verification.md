# PR-C: Precise acyclic control-flow symbolic execution in the Z3 verifier

Status: **design, for review** (gale #219). Prerequisite for inlining the 5
`br_table`-dispatching decides (sem_give 860, mutex_unlock 472, pipe_write,
pipe_read, msgq_put) and for removing the existing `BrIf`/`BrTable` skips in
LICM/CSE/code_folding/coalesce_locals.

## Problem

loom's live verifier (`verify.rs`) does **not** soundly model multi-way control
flow. In both `encode_function_to_smt_impl_inner` and `encode_block_body`:

- `BrIf(_depth)` pops the condition and **continues** ("a more precise encoding
  would fork paths here").
- `BrTable { .. }` pops the index and **`break`s** ("treat as terminating for
  now").
- `Br(_depth)` / `Return` `break`; branch **depth is ignored** (no label stack).

So a function whose result depends on a `br_table` dispatch is modeled as if the
dispatch didn't happen. Reusing this to prove an inline equivalent would let Z3
pick any arm for any selector → a **false** equivalence → an unsound inline that
loom's structural CI cannot catch (the #196/#220 failure class). This is why
LICM/CSE/etc. currently **skip** `BrIf`/`BrTable` functions entirely.

The precise machinery already exists as **dead scaffolding**, flagged in-source
as the deferred upgrade: `struct ExecutionState { stack, locals, globals,
path_condition: Bool, reachable }`, `merge_states(cond, t, f)`, and
`BlockResult.branch_depth`. `merge_bv(cond, a, b) = cond.ite(a, b)` is live (used
for `If`). PR-C wires the scaffolding in.

## Chosen approach (with precedent)

**State-merging-with-ITE driven by an explicit label/continuation stack — never
path-forking.** For a bounded, acyclic (loops=0 on the critical path),
single-normal-exit region this is **exact** (a passing SMT check is a real
proof, not a bounded approximation), and it produces **one** verification
condition instead of N per-arm obligations.

Precedent (all verified against primary sources):

- **Alive2** (Lopes, Lee, Hur, Liu, Regehr, *Bounded Translation Validation for
  LLVM*, PLDI 2021) — the closest production precedent. "We do **not** fork
  expressions across paths in the CFG"; phi/branch merges become "a single SMT
  expression per register"; the final state is "a linear chain of `ite`
  expressions"; an explicit **`ub`** flag makes ill-defined paths refine rather
  than vanish. `switch` is the N-way generalization of its 2-way branch merge.
- **Kuznetsov, Kinder, Bucur, Candea**, *Efficient State Merging in Symbolic
  Execution*, PLDI 2012 — the exact merge rule `(ℓ, pc'∨pc'', λv. ite(pc',
  s'[v], s''[v]))`; "does not over-approximate, no false positives."
- **WASP** (Marques et al., *Concolic Execution for WebAssembly*, ECOOP 2022) —
  the literal wasm rules: `br_table` arm `k≤n` → `Cont(br j_{k+1})` with path
  condition `ŝ = k`; default `k≥n` → `Cont(br jn)` with `ŝ ≥ n`. The selector
  constraint is conjoined onto every arm; the target is resolved by label index.
- **Barnett & Leino**, *Weakest-Precondition of Unstructured Programs*, PASTE
  2005 — per-block predicates keep the VC **linear** in paths (a wide `br_table`
  doesn't blow up); acyclic ⇒ topological, no recursion.
- **WebAssembly Core Spec** + Haas et al. (PLDI 2017) — wasm is structured with
  a label stack; `br N` targets the N-th enclosing label (0 = innermost);
  `br_table` takes the default when the operand is out of bounds.

Per-path forking is also sound but is the wrong fit: exponential over a wide
`br_table`, N separate obligations.

## The model

Carry a single symbolic `ExecutionState`:
`{ stack: Vec<BV>, locals: Vec<BV>, globals: Vec<BV>, memory: Array,
   path_condition: Bool, trapped: Bool }` (`trapped` = Alive2's `ub`/⊥).

A **label/continuation stack**: each entry records `{ arity, join: JoinAccumulator }`.
`block`/`if`/`loop` push a label; the entry's `join` accumulates the merged state
of every branch that targets it.

- `block tf … end`, `if … end`: push a label whose continuation is the code
  **after** `end` (forward join).
- `br N`: resolve the N-th label from the top; **merge** the current state into
  that label's `join` under `path_condition` (carry `arity` result values); mark
  the current path terminated.
- `br_if N`: split — branch path `pc ∧ (cond≠0)` merges into the target join;
  fall-through path `pc ∧ (cond=0)` continues.
- `br_table j0..jn-1 (default jd)` (per WASP): for each arm `i`, merge under
  `pc ∧ (sel = i)` into label `ji`'s join; default under `pc ∧ (sel ≥ n)` into
  `jd`'s join. Guards **partition** the selector domain (totality from the
  default complement).
- At each `end` / the function exit (the arms' post-dominator), the label's
  `join` is the **ITE-merge** of all incoming branches: `path_condition` =
  disjunction; each value = `ite(guard, …)` (reusing `merge_bv` / `merge_states`).
- `unreachable`, and a `Call` to a no-return callee (`is_noreturn_callee`: no
  `Return`, no `Br*` to the function label, every path ends in `Unreachable`),
  set `trapped` (⊥) — **never havoc**. They constrain reachability only; on a
  trapped path the return value is don't-care and the path drops out of the
  result ITE.

**Crucially**: this *one* executor models **both** the original `call decide`
(by-body) and the inlined body, so they yield the **same** merged expression →
equivalence is provable (the #155 by-body principle, extended to CF).

## Soundness — and the pitfalls we explicitly guard

1. **Exact for acyclic** — no fixpoint, no loop invariants; finite single pass;
   merge only at structured-block ends + the single exit (post-dominators).
2. **Selector constraint preserved** — every `br_table` arm carries `sel = i`,
   default carries `sel ≥ n`; guards partition the domain. (Avoids the
   "terminate" false-equivalence.)
3. **Fall-through resolved via the label stack** to the correct join (a `br` to
   an outer label skips intervening joins).
4. **⊥, not havoc, for trap/unreachable/no-return arms** — they refine, never
   forge a matching value (the #155/#159 + Alive2 `ub` discipline).
5. **Branch arity recorded** — `br N` carries the label's result values; the
   merged stack shape stays correct.
6. **Over-approximate-and-skip** anything unmodeled (LOOM's "skip rather than
   risk"): any instruction/shape outside the precise model → the callee is not
   by-body-modelable → opaque fallback (no inline), never a false ≡.

## Scope / staging

- **Loops stay out** — still unrolled (`MAX_LOOP_UNROLL`) / k-induction / skipped
  as today. PR-C is acyclic-only. A back-edge ⇒ not by-body-modelable.
- **Phase 1 — integer acyclic CF (no memory):** activate `ExecutionState` +
  label stack + `merge_states`; implement `Br`/`BrIf`/`BrTable`/`Block`/`If`
  precisely in `encode_block_body`; add `is_noreturn_callee` + diverge-on-trap;
  extend `is_inline_modelable_instr`/`callee_inlinable_by_body` to admit acyclic
  CF + no-return calls; route the by-body modeler through the precise executor.
  **Unblocks sem_give (860)** — pure integer + br_table, no memory.
- **Phase 2 — thread `memory: Array` through the CF executor** (today
  `encode_block_body` has no memory param). Unblocks **mutex_unlock (472)** +
  **pipe_write/read** (sret stores through the shadow frame).

## Test strategy (the soundness gate)

- **Adversarial verify-or-revert** (the hard gate): deliberately-wrong inlined
  bodies MUST produce a Z3 counterexample → revert — wrong arm selected, dropped
  selector guard, dropped trap branch, mis-merged join, wrong branch depth.
- **Differential**: correct inline of a `br_table` fixture proves; a near-miss
  reverts. Fixtures ordered per gale: msgq_put (trap-only) → sem_give
  (`block`+`br_table`+`panic→unreachable`) → mutex_unlock (most blocks, sret).
- **Unit**: `is_noreturn_callee`, the label-stack resolution, `merge_states`.
- **Regression**: full `cargo test` + dogfooding (loom optimizing itself)
  unchanged; re-validate the LICM/CSE/etc. paths once their `br_table` skips are
  removed (a bonus coverage win, but it widens what they verify).
- **e2e + silicon**: `loom optimize` on `repro-219/sem.loom.wasm` dissolves (no
  `call $..._decide`, no i64 pack/unpack in `z_impl_k_sem_give`); then gale's
  G474RE re-flash (sem_give 860→, mutex_unlock 472→) is the kill-criterion.

## Risk

Substantial, soundness-critical (touches the core verifier). De-risked by: the
acyclic restriction (the only source of unsoundness/incompleteness — loops — is
excluded), the existing `ExecutionState`/`merge_states` scaffolding, the
Alive2/WASP precedent, and the adversarial-revert + silicon gates. Staged so
Phase 1 (sem_give) lands and validates before Phase 2 (memory). WIP branch, no
merge until adversarial tests pass + gale silicon-validates.

## References

- Lopes, Lee, Hur, Liu, Regehr. *Alive2: Bounded Translation Validation for
  LLVM.* PLDI 2021. https://users.cs.utah.edu/~regehr/alive2-pldi21.pdf
- Kuznetsov, Kinder, Bucur, Candea. *Efficient State Merging in Symbolic
  Execution.* PLDI 2012. https://dslab.epfl.ch/pubs/stateMerging.pdf
- Avgerinos, Rebert, Cha, Brumley. *Enhancing Symbolic Execution with
  Veritesting.* ICSE 2014. https://softsec.kaist.ac.kr/~sangkilc/papers/avgerinos-icse14.pdf
- Marques, Fragoso Santos, Santos, Adão. *Concolic Execution for WebAssembly
  (WASP).* ECOOP 2022. https://drops.dagstuhl.de/opus/volltexte/2022/16239/pdf/LIPIcs-ECOOP-2022-11.pdf
- Barnett & Leino. *Weakest-Precondition of Unstructured Programs.* PASTE 2005.
  https://www.microsoft.com/en-us/research/wp-content/uploads/2005/01/krml157.pdf
- Haas et al. *Bringing the Web up to Speed with WebAssembly.* PLDI 2017.
- Van Hattum et al. *Lightweight, Modular Verification for Wasm-to-Native
  Instruction Selection (Crocus/veri-isle).* ASPLOS 2024.
