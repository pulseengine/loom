# wasm-opt → LOOM gap analysis (kernel-scheduler workloads)

Date: 2026-04-29
Target: LOOM v0.5.0 vs Binaryen wasm-opt (v116) on gale-style code
(small dispatchers, br_table, bit-manip, state machines, early-exit guards).
Source: gale v0.4.0 measurement report
(`/Users/r/git/pulseengine/loom/docs/research/gale-v0.4.0/measurement-report.md`).
LOOM pipeline confirmed by reading
`/Users/r/git/pulseengine/loom/loom-core/src/lib.rs` (lines 6713-6829, 7689-7770)
and `/Users/r/git/pulseengine/loom/docs/design/{simplify-locals,cse}.md`.

## Executive summary

The +6.3% code-section regression on gale is dominated by two effects:

1. **LOOM's CSE inserts `local.tee/local.get` pairs around 1-byte constants**
   (`-22`, `-16`, `-1`), turning a 2-byte sequence into a 4-byte sequence
   plus a function-header `(local i32)` declaration. This is exactly
   backwards on a workload made of small dispatchers full of `-EINVAL` /
   `-EBUSY` / `K_FOREVER` constants.
2. **LOOM's `simplify_locals` skips every function with early-exit
   control flow** (`has_dataflow_unsafe_control_flow` is true for any
   `BrIf`, `BrTable`, or non-tail `Return`/`Br`). Kernel scheduler code
   *is* early-exit code: every function on the gale measurement that
   could have benefited (`gale_k_sem_give_decide`, `func 0`, `func 13`,
   `func 21`, `func 24`) was skipped. The pass had zero effect on the
   workload by construction.

So the highest-payoff work is **not** to add a brand-new pass — it is
to make the two existing passes pay off on early-exit code, plus add
two cheap shape-canonicalization passes wasm-opt has and LOOM
doesn't (`reorder-locals`, `RedundantSetElimination` proper).

## Ranked picks

### 1. `RedundantSetElimination` (RSE) — proper, with liveness across branches

**What it does.** Removes a `local.set` whose value is overwritten on
every path before any read, even when the path goes through a
branch/return. Binaryen's `RedundantSetElimination` (`--rse`) is a
flow-sensitive forward dataflow that propagates the *set value lattice*
through the CFG; a set is redundant iff for every successor path the
local is overwritten before being read, OR the function exits without
the local being read.

**Why it pays on kernel-scheduler code.** This is exactly the
"dead-store elimination of init-only locals" the gale report calls out.
The pattern in gale is:

```
i32.const -22
local.set 3        ;; sentinel init
... if (cond) { ...; local.set 3; ... } else { ...; local.set 3; ... }
local.get 3
return
```

Both branches write before the read, so the initial sentinel set is
dead. wasm-opt removes it; LOOM's current `simplify_locals` skips the
whole function because `BrIf`/`Return` is present. The init-only-local
pattern is *universal* in Zephyr-style FFI — every function returns an
errno-style `int` and starts by initializing it to a default.

**Why LOOM's current `simplify_locals` cannot do this.**
`has_dataflow_unsafe_control_flow` (lib.rs:6713) bails on any non-tail
`Return`/`Br`/`BrIf`/`BrTable`. The eligibility check is correct for
the *substitution* operations that pass also performs (equivalence
propagation, get/set forwarding) but it is far too coarse for
straight-up dead-set elimination, which is sound under liveness alone.

**Concrete LOOM change.** Split `simplify_locals` into two sub-passes:

- `simplify_locals_pure` — current code, gated by
  `has_dataflow_unsafe_control_flow` (keep the safety net).
- `dead_local_set_elimination` — new, runs liveness analysis on the
  CFG and removes any `local.set` whose value is not live-out. Runs
  unconditionally (no early-exit gate). Replace the set with `drop`
  so the stack stays balanced; the existing DCE pass already collapses
  `i32.const N; drop` to nothing (lib.rs:7109).

**Estimated complexity.** ~600 LOC for live-range computation over
LOOM's instruction stream + ~200 LOC plumbing + ~150 LOC tests.
1.0–1.5 weeks. The CFG already exists for `coalesce_locals`
(lib.rs:9824) — reuse that liveness builder. The hardest part is the
join at if/block/loop merge: take liveness as the union of successor
live-in sets. Loops require iteration-to-fixedpoint over back-edges.

**Risk / verifiability.** Low. The transformation is a textbook
liveness-driven DSE and has an obvious Z3 encoding: for every removed
set, prove `∀paths from set to any subsequent use of the local, ∃ a
preceding set on the same path`. LOOM's `TranslationValidator` already
does pre/post equivalence; for a set→drop rewrite, the post-state
differs only in one local's value, and that local must be unused on
all paths (the very property we proved). Easy Rocq lemma:
`liveness_join_over_approximates ∧ removed_local_not_in_live_out ⇒
post_module ≅ pre_module`. No new SMT theory needed.

### 2. `simplify-locals` "sink/tee" mode — usable on early-exit code

**What it does.** Binaryen's `SimplifyLocals` does *more* than
substitution. Its biggest payoff is the **sinker**: when a value is
computed and stored to a local that has exactly one read, sink the
expression into the read site (deleting both the set and the get).
This is what turns
```
local.tee 1; ...;  local.get 1; i32.add
```
into
```
... ; (sunk_expr); i32.add
```
when the local has one writer + one reader and the reader dominates
no-other-uses. Combined with sinking into block/if/loop result types,
it's the pass that "moves a value through a control-flow merge so it
no longer needs a local at all." The current Binaryen pass is
~1500 lines because of the merge-point bookkeeping.

**Why it pays on kernel-scheduler code.** Dispatcher functions are
chains of `if (errno) goto exit; ... result = compute();` and the
sinker collapses the trailing `result = compute(); local.get result;
return` into a tail `return compute()`. Every gale function ends with
this pattern.

**Why LOOM doesn't have it.** LOOM's `simplify_locals` is a
substitution-only iteration; the design doc (`simplify-locals.md`
lines 14-86) doesn't list sinking. And again: the early-exit gate
disables the whole function.

**Estimated complexity.** ~1200–1500 LOC. This is the most
LOC-expensive pick on this list. Needs single-use/single-def
analysis, dominance, and a "can this expression be moved past these
intervening instructions" effect check (loads vs stores, calls,
local-aliasing). 3–4 weeks done carefully.

**Risk / verifiability.** Moderate. The motion check is the
tricky bit: sinking past a store is unsafe if the value depends on
memory. Use LOOM's existing effects framework
(`crate::verify` already encodes "may-load / may-store / may-call"
at the instruction level — the CSE eligibility logic uses it).
Z3 verifier can pre/post compare for any sunk expression, but
validating *all* possible sinks per-function is expensive; the
practical rule is to gate by "expression has no may-store-affecting
load" and reject anything else. Rocq proof for the gate condition
is one moderate-difficulty lemma (~200 lines).

### 3. `reorder-locals` — slot renumbering for compactness

**What it does.** After all other passes, sort locals by use-count
(descending). Index 0..N-1 frequently-used locals get short LEB128
encodings; rarely-used ones get pushed to high indices. Also drops
locals with zero uses (a free DCE).

**Why it pays on kernel-scheduler code.** The gale report
(section 4.3) calls this out specifically: wasm-opt renames a
one-shot `local 3` to `local 1` and drops the now-empty slot, saving
a local-decl byte. In small dispatcher functions, the function header
is a non-trivial fraction of the bytes — every gale function has
4–8 locals declared, and removing one saves 2 bytes per declaration
group. Across 25 functions that compounds.

**Estimated complexity.** ~250 LOC. Trivial: count `LocalGet`/`LocalSet`/
`LocalTee` per index, build a permutation `old → new`, rewrite. The
only subtleties are (a) parameters keep their indices (params occupy
the low slots by spec), (b) update local-decl-runs (locals are
encoded as run-length groups by type — recompute the groups after
renaming). 0.5 weeks including tests.

**Risk / verifiability.** Trivial. Pure renaming is a bijection; the
post-module is observably identical (proof: structural induction over
instructions, every `Local{Get,Set,Tee}(i)` becomes
`Local{Get,Set,Tee}(σ(i))` and `σ` is a bijection over local indices,
QED). Rocq proof is ~50 lines.

### 4. Compare-operand canonicalization (`lt_u` ↔ `gt_u` flip + tee target reuse)

**What it does.** Recognize the pattern
```
A; local.tee X; B; local.get X; cmp_op
```
where the live value through `B` is `X`. If `cmp_op` is one of the
order comparisons (`lt_u`, `lt_s`, `le_u`, `le_s`, `gt_u`, `gt_s`,
`ge_u`, `ge_s`), swap the tee target with one of the existing
locals declared in the function and flip the comparison. Goal: keep
the live value in the same physical register slot through the
intervening instructions (saves a wasm-runtime register move; on
some engines also saves a stack-spill byte in the encoding when the
new slot has a smaller LEB).

**Why it pays on kernel-scheduler code.** Section 4.2 of the gale
report identifies this exact pattern in the wopt diff. State-machine
transitions are full of "compare current state to threshold, branch"
sequences, and every one of them goes through a tee/get pair around
a comparison. wasm-opt does this in `OptimizeInstructions`; LOOM has
no equivalent.

**Estimated complexity.** ~400 LOC. Pure peephole on a 5-instruction
window with a simple slot-reuse heuristic (prefer slot whose other
uses are temporally adjacent — minimize live range). 1 week
including a use-density heuristic.

**Risk / verifiability.** Easy. The flip
`(a < b) ≡ (b > a)` for unsigned is a one-line Z3 fact and a
two-line Rocq lemma. The slot reuse is just a renaming, same as
pick 3. Combined proof is ~80 lines of Rocq.

### 5. Constant-CSE suppression heuristic (FIX existing CSE pass)

**What it does.** This is *not* adding a pass — it is gating an
existing one. The current LOOM CSE pass (lib.rs CSE phase) replaces
two adjacent `i32.const -22` pushes with `i32.const -22; local.tee N;
local.get N`. This is a regression for any constant whose LEB128
encoding is ≤ 2 bytes (i.e. anything in `-64..=63` for `i32.const`),
because the `local.tee N + local.get N` replacement is 2 + 2 = 4
bytes, plus the function-header `(local i32)` declaration.

**The rule.** Decline to CSE an expression whose post-encoding cost
is `≤ cost_of(local.tee + local.get + local_decl_amortized)`. For
typical functions with ≥1 existing local of the right type, the
break-even is ~3 LEB bytes for the original expression. Below that,
do not CSE.

**Why it pays on kernel-scheduler code.** Gale's section 4.4 traces
the +6.3% almost entirely to constant-CSE on `-22` / `-16` / `-1` /
`0`. Adding this single 20-line gate likely flips the sign of the
LOOM-vs-baseline delta on this whole class of workload.

**Estimated complexity.** ~50 LOC for the cost model + gate;
~100 LOC for tests across constant width tiers. 1–2 days. By far
the highest payoff-per-LOC item on this list.

**Risk / verifiability.** Zero — this is *removing* an unsound /
unprofitable rewrite, not adding one. The existing CSE proof
obligation is unchanged; we just don't fire on certain inputs.
No new Rocq work.

### 6. `directize` — `call_indirect` → `call`

**What it does.** When a `call_indirect` site can be proven to
reach exactly one table entry (because the function index on the
stack came from `i32.const N` with no intervening table mutation,
or because the table is immutable and indexable by a known constant),
rewrite to a direct `call $target`. Direct calls are cheaper to
encode (no type index, no table index, no runtime type-check), and
they expose the callee body to the inliner.

**Why it pays on kernel-scheduler code.** Wasm-component fused
modules — which is exactly what gale produces — generate
`call_indirect` for every cross-module dispatch. After
`fused_optimizer.rs` devirtualizes adapters, a residue of
"resolved-but-still-indirect" call sites remains. Directize gets the
last 5–10% of indirect calls on Zephyr-style FFI shims.

**Estimated complexity.** ~700 LOC. Need: table contents tracking
(LOOM has the table model), constant-stack-tracking dataflow over
the indirect-call site's index operand, and the rewrite. 2 weeks.

**Risk / verifiability.** Moderate. Soundness depends on table
immutability *and* on the index operand being a constant on every
incoming path. The first is a module-level invariant
(no `table.set`, no `table.fill`, no exported table). The second
is a per-call-site dataflow query that LOOM's verifier can encode
to Z3 (the index expression's symbolic value is a singleton set).
Rocq lemma: "if table T is immutable and index expression is
provably constant N, then `call_indirect (type_of T[N])` ≡
`call T[N]`." ~150 lines.

### 7. `merge-locals` — pre-coalesce same-value merging

**What it does.** Binaryen's `MergeLocals` runs *before*
`CoalesceLocals` and merges two locals that hold the same value at
some program point — even if their lifetimes overlap — by rewriting
all reads of one to reads of the other. This collapses pure copies
and pre-improves the interference graph that CoalesceLocals will
then color.

**Why it pays on kernel-scheduler code.** Dispatcher functions
load a struct field, copy it into a "current value" local, and
operate on that. The struct-field local and the working local are
copies of each other for the entire function body. MergeLocals
collapses them; LOOM's existing `coalesce_locals` cannot — it sees
the two locals as live simultaneously and refuses to merge their
slots.

**Estimated complexity.** ~500 LOC. Requires a value-numbering
pass over locals (which LOOM has fragmentary infrastructure for in
the CSE pass). 1.5 weeks.

**Risk / verifiability.** Moderate. The proof obligation per merge
is "for every read of L2, the value of L1 equals the value of L2 at
that program point." Discharge with Z3 forward-symbolic execution
or with a syntactic copy-chain analysis (cheaper, less complete).
Rocq: ~200 lines for the syntactic version. The flow-sensitive
version is harder but can be deferred.

## Specific transformations from the task

### Dead-store elimination of init-only locals

**wasm-opt's algorithm** (from Binaryen `RedundantSetElimination` +
`SimplifyLocals` + `Vacuum` working in concert):

1. `RedundantSetElimination` runs a forward dataflow that, at every
   program point, knows for every local "what set is currently
   responsible for the local's value, if any". When a `local.set`
   is reached and the local already holds the value being set
   (e.g. via copy chain), the set is removed.
2. `SimplifyLocals` runs a backward liveness pass: for every set,
   if the local is not live-out of the set (no use on any forward
   path), the set is replaced by a `drop` of the value (if it has
   side effects) or removed entirely (if pure).
3. `Vacuum` cleans up the `(i32.const N) (drop)` pairs left behind.

The init-only-local pattern (`i32.const -22; local.set 3; ... ;
local.set 3 on every path before any read`) is killed by step 2.
The local has *some* later set, but on every forward path from the
init set there is another set before any get — so the init set's
value is not live-out, hence dead.

**What LOOM needs.** Pick #1 above. Specifically:

- A **liveness analyzer** over LOOM's instruction stream. The
  fragments needed are already present (`coalesce_locals` does live
  ranges; the CFG builder is in `verify.rs`). Wire them into a new
  `dead_local_set_elimination` function.
- The analyzer's **join function** must take the union of
  successor live-in sets at every merge, and iterate to fixed
  point over loop back-edges.
- The **transform** is: for every `local.set X` whose value is not
  in `live_out(set_position)`, replace the set with `drop`. Run
  DCE to eat the `i32.const N; drop`.
- The **gate** must NOT be `has_dataflow_unsafe_control_flow` —
  liveness across branches is precisely the case we want to handle.
  Use a much narrower gate: skip only functions with `Unknown` or
  `CallIndirect` instructions where effects can't be modeled
  (consistent with LOOM's other passes).

### Compare-operand canonicalization for register reuse

This is pick #4 above. The full algorithm:

1. Find a window `[A, local.tee X, B, local.get X, cmp_op]` where
   `B` does not modify `X` and `cmp_op ∈ {lt_u, lt_s, le_*, gt_*,
   ge_*, eq, ne}`.
2. For each candidate target slot `Y` already declared in the
   function:
   - Compute the live range cost of replacing `X` with `Y` (does
     `Y` have other uses that conflict?).
   - Compute the encoding-byte delta (can be negative if `Y` has
     a smaller LEB index).
3. If delta is negative or zero AND no live-range conflict, rewrite
   to `[A, local.tee Y, B, local.get Y, flipped_cmp_op]`. The flip
   table:
   - `lt_u ↔ gt_u`, `lt_s ↔ gt_s`
   - `le_u ↔ ge_u`, `le_s ↔ ge_s`
   - `eq` stays `eq`, `ne` stays `ne` (operand order doesn't matter)

The rule applies because `(a CMP b) = (b CMP_FLIPPED a)` for total
orders, and the operand order on the wasm stack is exactly what we
swapped. Pure shape canonicalization, no value semantics changed.

## Recommended LOOM roadmap (kernel-scheduler-optimized)

| Order | Pass | LOC | Wks | Risk | Expected gale code-section delta |
|---|---|---:|---:|---|---|
| 1 | Constant-CSE suppression gate (pick #5) | 50 | 0.3 | None | -3% to -5% (kills regression source) |
| 2 | reorder-locals (pick #3) | 250 | 0.5 | Trivial | -1% |
| 3 | RedundantSetElimination proper (pick #1) | 600 | 1.5 | Low | -2% to -3% |
| 4 | Compare canonicalization (pick #4) | 400 | 1.0 | Easy | -0.5% |
| 5 | merge-locals (pick #7) | 500 | 1.5 | Moderate | -1% |
| 6 | Directize (pick #6) | 700 | 2.0 | Moderate | -1% to -2% |
| 7 | simplify-locals sinking (pick #2) | 1500 | 4.0 | Moderate | -1% |

Cumulative projected effect on gale code section: roughly -10% (vs
current +6.3%), i.e. crossing parity with wasm-opt -O3 and probably
a small win because the formal verifier should let us be more
aggressive than wasm-opt on a few patterns it conservatively rejects.

The first three rows alone (1.3 weeks of work) likely flip the sign
of the gap on this entire workload class. Picks 1 and 5 should be
the immediate priority — pick 1 unlocks the gate that disables every
other simplify-locals optimization on early-exit code, and pick 5 is
a 50-LOC fix to a known regression.

## Sources

- [Binaryen wasm-opt manpage (Debian)](https://manpages.debian.org/testing/binaryen/wasm-opt.1.en.html)
- [Binaryen pass.cpp (pass registry)](https://github.com/WebAssembly/binaryen/blob/main/src/passes/pass.cpp)
- [Binaryen SimplifyLocals.cpp](https://github.com/WebAssembly/binaryen/blob/main/src/passes/SimplifyLocals.cpp)
- [Binaryen Vacuum.cpp](https://github.com/WebAssembly/binaryen/blob/main/src/passes/Vacuum.cpp)
- [Binaryen Optimizer Cookbook (wiki)](https://github.com/WebAssembly/binaryen/wiki/Optimizer-Cookbook)
- [wasm-opt: The WebAssembly Optimizer (DeepWiki)](https://deepwiki.com/WebAssembly/binaryen/4.1-wasm-opt:-the-webassembly-optimizer)
- [Compiling to and optimizing Wasm with Binaryen (web.dev)](https://web.dev/articles/binaryen)
- [RedundantSetElimination nondeterminism issue #4524](https://github.com/WebAssembly/binaryen/issues/4524)
- gale measurement report (local): `/Users/r/git/pulseengine/loom/docs/research/gale-v0.4.0/measurement-report.md`
- LOOM pipeline source (local): `/Users/r/git/pulseengine/loom/loom-core/src/lib.rs` (lines 6713-6829, 7689-7770, 8042-8200)
- LOOM design docs (local): `/Users/r/git/pulseengine/loom/docs/design/{simplify-locals,cse,dce,vacuum,branch-simplification}.md`
