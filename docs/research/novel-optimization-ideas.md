# LOOM Novel Optimization Research

Brainstorm of optimization ideas that LOOM could pursue *because* it has formal verification (Z3 translation validation + Rocq mechanized rewrites) and source-side knowledge (Verus annotations, Rust ownership, kernel-scheduler invariants), where wasm-opt cannot or will not.

Target workload context: Verus-verified Rust kernel-scheduler primitives (à la `pulseengine/gale`). Tight cycle budgets, branch-heavy dispatch, bit-manipulation, predictable control flow.

Each idea carries an honesty rating (1–10) for plausible soundness in LOOM's framework: 10 = highly confident, 1 = vibes only.

---

## 1. Verus-Witnessed Range Narrowing ("WitnessProp")

When the source crate is Verus-verified, every function carries `requires`/`ensures` clauses giving exact value ranges, monotonicity, and cross-argument relations. LOOM ingests these as a sidecar (compiled to SMT-LIB or Rocq lemmas) and uses them as *trusted lemmas* during pass-local abstract interpretation. wasm-opt has only what it can re-derive from wasm itself — usually nothing useful after Rust monomorphization. LOOM gets the *semantic type*: `i32 ∈ [0, 63]`, `i32 == popcount(other_i32)`, etc. Unlocks bounds-check elimination, branch folding, narrowing to `i8`, table-size shrinkage.

**Example.** `ready_queue_pick(ready_mask: i64) ensures result < 64`. Caller had `if pick(m) >= 64 { panic }`; LOOM with witness drops the `>=64` branch entirely.

**Why LOOM-specific.** wasm-opt has no way to *trust* an external assertion. LOOM encodes Verus `ensures` as a Z3 axiom: if optimized ≡ original *under that axiom*, and the axiom is discharged by Verus, soundness composes.

**Sketch.** Sidecar `function-id → list of (precondition, postcondition)` in SMT-LIB, exported by a `verus-export` plugin. New IR pass `witness-prop` extends the existing range-lattice with axioms looked up by callee id. Validator obligation: prove `pre ∧ original ≡ optimized` *given* the axioms.

**Honesty: 9.** The most obvious LOOM-shaped win. Risk is keeping the witness sidecar in sync — but Verus already emits this; rivet/Bazel can hash-bind it.

---

## 2. Effect-Polarity Hoisting ("Polarity")

Rust's borrow checker plus Verus annotations let LOOM assign each function a *polarity* over a small fixed set of effects: `reads_mem`, `writes_mem`, `reads_global`, `writes_global`, `traps`, `unwinds`, `atomic`. With monotone polarity, LOOM does far more aggressive code motion than wasm-opt's GUFA: hoist pure leaf calls out of loops, sink writes past reads-of-disjoint-region, reorder atomic-free sections around acquire fences. wasm-opt is limited to syntactic purity; LOOM gets *semantic* purity over disjoint regions.

**Example.** Loop body that calls a pure `bitset_count` and does memory writes that don't alias the bitset's inputs — LOOM hoists the call, wasm-opt can't.

**Why LOOM-specific.** wasm-opt has no concept of *disjoint memory regions*. LOOM consumes Rust's region info and discharges aliasing via Z3.

**Sketch.** Annotate functions with effect tuples in IR; default conservative. Derive via a `cargo-wasm-meta` plugin walking MIR. Pass: classical LICM with effect-aware safety check; validator emits noninterference SMT obligation per hoist.

**Honesty: 8.** Aliasing is hard. Soundness depends on the MIR-to-effects exporter — but Verus's region tokens make it tractable.

---

## 3. Lock-Discipline Folding ("LockFold")

Kernel-scheduler code is full of `lock; read field; check invariant; unlock`. With Verus tracking lock ownership as a linear resource (`tracked Mutex::lock`), LOOM knows which instructions are inside a critical section, which globals the section owns, and that no concurrent writer can fire. LOOM can (a) coalesce adjacent critical sections wasm-opt must keep separate, (b) sink non-mutated reads out of the section, (c) replace double-checked-locking with a single check.

**Example.** Two adjacent `(call $lock $m) (load) (call $unlock $m)` blocks collapse to one critical section when the gap is provably yield-free and `$m` uncontended on this CPU's path.

**Why LOOM-specific.** wasm-opt can't see that `$lock`/`$unlock` are matched, has no linear-resource discipline, no preemption model. wasm-opt must treat every `call` as a memory barrier worst-case.

**Sketch.** Verus lock tokens lower to a wasm custom section `loom.locks` listing `(lock_call, unlock_call, mutex_id)` triples. IR-level lock-region inference; merge adjacent regions iff (i) same mutex, (ii) no preemption point between. Validator: per-merge, prove observational equivalence under interleaved-execution model.

**Honesty: 6.** Concurrency reasoning is where soundness goes to die. Gate behind "single-core dispatch only" mode initially.

---

## 4. Bitmask-Shape Specialization ("BitShape")

Ready-queue bitmaps in kernel-scheduler code carry shape invariants: priority bands densely packed, often only the low 8 or 16 bits ever non-zero, `popcount` bounded by NCPUS×4. With Verus proof that "at most 32 threads can be ready," LOOM specializes an `i64` bitmap to `i32` (or `i16` plus one overflow word) along the hot path, replaces generic `ctz`/`popcnt` with branch-free 8-bit lookup-table versions that are *provably equivalent on the restricted domain*.

**Example.** `(local.get $rm) i64.ctz` becomes `(local.get $rm) i32.wrap_i64 i32.ctz` when high 32 bits are provably zero — validator discharges the equivalence.

**Why LOOM-specific.** wasm-opt has no way to know the shape invariant. LOOM has a *proof* the high half is zero — it's always-correct, not "usually faster."

**Sketch.** New abstract domain: bit-pattern lattice (per-position: 0, 1, ⊤). Seeded from Verus invariants and per-block constant prop. Rewrite catalogue: 64→32 narrowing, ctz/popcnt/clz lookup-table versions, conditional-move replacements when popcount ≤ k. Each rewrite has a Rocq lemma; pass-local Z3 query proves the shape precondition holds at the rewrite site.

**Honesty: 8.** Concrete, locally checkable, small proofs. Direct hit on the kernel-scheduler hot path.

---

## 5. Trap-Free-Region Coalescing ("TrapFree")

A wasm bounds-check or `unreachable` traps. wasm-opt preserves trap behavior conservatively because *removing a trap is observationally distinguishable*. LOOM, with Verus proofs that a region never traps, marks it "trap-free" and applies optimizations unsound under arbitrary trap semantics: reorder across formerly-trapping ops, fuse a guard with its guarded op, eliminate impossible guards, move loads ahead of branches.

**Example.** `if (i < 64) { load(base + i) } else { unreachable }` with Verus `requires i < 64` becomes plain `load(base + i)`.

**Why LOOM-specific.** Trap-elimination is the textbook thing wasm-opt *will not do* — the wasm spec is precise about trap observability. Removing traps requires proof they cannot fire — exactly what Verus gives LOOM.

**Sketch.** Per-instruction trap-possibility flag, propagated via abstract interpretation seeded by Verus pre/post. Trap-free regions first-class in IR. New rewrites legal only inside trap-free regions: speculative load hoist, branch-merging, dead-`unreachable` elimination. Validator: every rewrite emits "no path traps" obligation, discharged via the witness axiom set (idea #1).

**Honesty: 9.** The canonical "wasm-opt won't do it, LOOM can" optimization. Composes with #1.

---

## 6. Dispatch-Table Devirtualization ("DispatchPin")

Kernel-scheduler dispatch is often `call_indirect` against a function-pointer table indexed by syscall number / scheduler class / thread state. The valid-index set is small, often known statically from a Rust enum + match. LOOM reads the source-side exhaustive match and emits a proof-checked branch tree of direct calls, each with exact callee identity, enabling aggressive inlining and inter-procedural specialization. wasm-opt rarely has the closed set after Rust codegen via wasm function tables.

**Example.** `dispatch[class](thread)` with `class ∈ {Realtime, Normal, Idle}` becomes a 3-way if-chain (or balanced `br_table`) of direct calls — each inlinable, specializable.

**Why LOOM-specific.** wasm-opt would need the closed set, the table-index → function mapping, and proof the variant tag invariant holds. Verus + monomorphic Rust enums give LOOM all three.

**Sketch.** Cargo plugin emits per-call-site enum-discriminant ranges + table indices. IR pass converts `call_indirect` with closed discriminant into if-chain or `br_table` of direct calls. Validator: prove for each branch that callee identity matches the table entry under the discriminant assumption.

**Honesty: 8.** Standard devirt, but the closed-world piece is the LOOM-special bit.

---

## 7. Yield-Point-Aware Reordering ("YieldFence")

Preemptive kernels have a *fixed, source-known* set of yield points. All non-yield code is, from the scheduler's view, a single atomic straight-line block — the scheduler cannot run between two non-yielding instructions on the same CPU. LOOM treats the IR as "fenced segments separated by yield points" and performs aggressive within-segment reordering (including across calls whose transitive closure is yield-free). wasm-opt treats every call as a potential observation boundary.

**Example.** Inside a critical section with `$compute_priority` proved yield-free, LOOM may reorder/interleave the store and the call freely within the fenced segment.

**Why LOOM-specific.** Yield points are a *kernel design decision*, not a wasm artifact. wasm-opt sees a `call` and assumes the worst.

**Sketch.** Kernel emits `loom.yield-points` custom section listing functions whose call may yield. Compute transitive yield-closure once; mark every IR call as `yield_safe` or `yield_unsafe`. Within a maximal yield-safe span, run a richer scheduler. Validator: per reorder, prove single-thread observational equivalence (existing Z3 machinery suffices).

**Honesty: 7.** Conceptually clean. Risk is the trusted yield manifest; mitigation: derive mechanically from `#[yields]` annotations.

---

## 8. Verified Loop-Invariant Lookup-Table Materialization ("LUTify")

Many kernel primitives compute small pure functions over small finite domains: `priority_to_quantum(p ∈ 0..32)`, `cpu_affinity_lookup(t ∈ 0..NCPUS)`, `sleep_class(state ∈ enum)`. With Verus proving the function pure and total on the finite domain, LOOM pre-computes the entire input → output table at compile time, replaces the call with a `i32.load` from a `data` segment, and *proves the table is correct by symbolic exhaustion*.

**Example.** `(call $priority_to_quantum (local.get $p))` (~30 instr arithmetic) becomes `(i32.load offset=$lut (i32.shl (local.get $p) (i32.const 2)))`. A single load, predictable cache behavior.

**Why LOOM-specific.** Three things needed: (a) prove the function is pure (#2), (b) prove input is in a finite domain (#1), (c) prove the materialized table is bit-identical to the function on every input. The third is *exhaustive symbolic equivalence* — LOOM's bread and butter.

**Sketch.** Trigger: pure + finite-domain function whose body is below an instr threshold. Synthesize the table by abstract interpretation or by enumerating the domain in the validator. Replace call site with indexed load; emit data segment. Validator obligation: ∀ input ∈ domain, table[input] = original(input).

**Honesty: 9.** Mechanical, local, low-risk. The key insight: "compile-time evaluation" is sound only if the function is provably pure and total — both delivered by Verus.

---

## 9. Failure-Path Outlining ("ColdSplit")

Verified kernel code has many "this can't happen, but defense in depth" branches: `if x.is_none() { panic!(...) }` where Verus proved `x.is_some()`. wasm-opt either keeps these (soundness) or, with flags, blindly removes them (unsoundness if proof was wrong). LOOM does something cleverer: keep the panic branch but *outline* it to a cold section, replace the inline check with one conditional jump, and annotate the outlined block as "unreachable under stated invariants" so the engine can place it on a non-resident page.

**Example.** Combined with #5: if Verus proved `$opt` is some, LOOM removes the check entirely. Without proof: LOOM still outlines, hot path zero-cost.

**Why LOOM-specific.** wasm-opt could in principle outline. The *interesting* version — provably-cold outlining that lets the engine de-prioritize page residency — requires proof that the cold path's preconditions are unsatisfiable on hot inputs. That's a verifier query.

**Sketch.** Identify defense-in-depth panics via source attribution. Outline to a separate function or wasm custom section with cold marker. Validator: emit obligation that under stated function preconditions, the cold block is unreachable; if it discharges, mark "provably cold," else "heuristically cold." Engine integration via `loom.coldness` custom section.

**Honesty: 7.** Outlining itself is easy; the *provably cold* annotation depends on engine cooperation.

---

## 10. Cross-Pass Witness Recycling ("WitnessCache")

A *meta* optimization. LOOM's translation validator does N SMT queries per pass, often re-deriving the same facts ("bit 31 is zero," "this pointer is non-null"). With a content-addressed cache of *proven witness facts* keyed on a canonical hash of the relevant IR slice, LOOM amortizes verification cost across passes *and across builds*. Downstream passes consult this cache to enable optimizations that would otherwise be too expensive to verify on the fly. Verified optimization *composes* — the more passes run, the richer the witness pool, the more optimizations become tractable.

**Example.** Pass A proves "after this block, `$x` ∈ [0,7]." Pass B, running later, reads the proven fact in O(1) and uses it to narrow `i32` to `i8` without re-running range analysis.

**Why LOOM-specific.** Requires a verifier and a verified rewrite catalog. wasm-opt has neither. It's the verification infrastructure turned into a first-class optimization-enabling resource.

**Sketch.** Witness format: `(ir-slice-hash, fact-smt, proof-blob)`. Bazel-friendly cache (content-addressed, fits naturally with rivet). Each pass publishes facts it proves; each pass consults the cache before re-deriving. Validator: a cached fact is trusted iff the proof-blob re-checks (cheaply) against the current IR slice's hash.

**Honesty: 8.** Engineering, not theory — but it changes which *other* optimizations are economically viable.

---

## Top 3 by `impact * tractability`

1. **Trap-Free-Region Coalescing (#5).** Massive impact on kernel-scheduler hot paths (every Verus-eliminated bounds check is a deleted branch), conceptually simple, validation obligation composes with existing translation validation. Most likely to *visibly differentiate* LOOM from wasm-opt in benchmarks.

2. **Verus-Witnessed Range Narrowing (#1).** The *foundation* that #5, #4, #8, and #10 all build on. Shipping #1 ships half of the others as a side effect. High tractability — Verus already emits the data; LOOM "just" has to consume it.

3. **Bitmask-Shape Specialization (#4).** Direct hit on the kernel-scheduler workload (ready-queue bitmaps, popcnt-bounded sets), local rewrites with small Rocq lemmas, immediately measurable wins on dispatch hot paths. The most demoable.

**Honorable mention:** #10 (WitnessCache) is the meta-multiplier. It doesn't ship a single optimization but it makes everything else cheaper to verify, which over the project's lifetime is probably worth more than any individual rewrite.

---

## Source

Generated by a research-mode agent on 2026-05-01 as part of LOOM v0.5.0 planning. Inputs: v0.4.0 audit findings, kernel-scheduler workload context (Verus-verified Rust à la `pulseengine/gale`), comparison against wasm-opt and Cranelift.
