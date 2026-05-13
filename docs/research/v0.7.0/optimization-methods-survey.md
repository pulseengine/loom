# Compiler-optimization methods survey for LOOM v0.7.0+

Date: 2026-05-03
Author: read-only research scan
Scope: methods beyond the wasm-opt parity work already tracked in
`docs/research/gale-v0.5.0/wasm-opt-gap-analysis.md`. v0.6.0 shipped
CSE cost-gating, dead-locals, full backward-liveness DSE, and a
const+drop vacuum peephole; this survey looks at the next layer.

The hard constraint that filters every entry below: LOOM is
**proof-first**. Every optimization must be Z3 translation-validated
or be conservative enough that the verifier rejects it on mismatch
and reverts. Methods that have no soundness story are flagged as
unfit even if the literature loves them.

---

## Verdict — invest in these five for v0.7.0

| Rank | Family | One-sentence justification |
|---|---|---|
| 1 | **Acyclic egraph mid-end (ægraph-style)** | LOOM already uses ISLE for rewriting; an aegraph upgrades term-rewriting to a global GVN+LICM+rematerialization data structure with proven productionizability in Cranelift, and the *acyclic* variant preserves verifiability that full equality saturation loses. |
| 2 | **Souper-style verified peephole synthesis** | Mines new rules from real corpus traces and discharges each candidate via Z3 — natively matches LOOM's "no rule without proof" philosophy and would auto-populate the rule database that today is hand-written. |
| 3 | **Function-summary IPA (purity / no-trap / constant-return)** | Wasm has clean call boundaries and LOOM already skips functions it can't model; small summaries unlock CSE/DCE/inline decisions today blocked by `Call` instructions, with trivial soundness (the summary is itself a Z3-checkable assertion). |
| 4 | **Component-Model-aware specialization** | Meld fuses components, LOOM optimizes after; canonical-ABI adapters and resource liftings leave a stable, recognizable residue (lift/lower/canon adapters around fixed-shape values) that no upstream optimizer targets. |
| 5 | **Verification-aware shape canonicalization** | A class of rewrites whose only purpose is to *expand Z3-provable territory* (e.g. `BrIf` → `Select` lowering, normalizing tee/get pairs before CSE) — multiplies the impact of every other pass without changing observable semantics. |

Picks 1 and 2 are the big strategic moves. Picks 3-5 are tactical and
each delivers within a single release. Skipped from the top tier:
egglog/equality-saturation proper (verifier cost too high — see §1),
polyhedral (gale's bounded loops don't need a polytope solver — see §3),
PGO (LOOM is offline — see §4), ML pass-ordering (breaks proof story —
see §12).

---

## 1. Egraph-based / rewrite-driven

**Mechanism.** Represent a program as an e-graph: each e-class is an
equivalence set of e-nodes (terms). Rewrite rules add new e-nodes to
existing classes (`union(a, rewrite(a))`); a cost-aware extractor
picks one term per class at the end. Cranelift's **ægraph** is the
production-relevant variant: it is acyclic, applies rules greedily at
node creation rather than to saturation, and gets GVN+LICM+
rematerialization "for free" during translation in/out
([Fallin 2026](https://cfallin.org/blog/2026/04/09/aegraph/),
[bytecodealliance/rfcs#27](https://github.com/bytecodealliance/rfcs/blob/main/accepted/cranelift-egraph.md)).

**Fit with proof-first.** Yes for ægraphs; *partial* for full
equality saturation. The ægraph variant retains a one-step-rewrite
trace for every union, which is exactly what a translation validator
needs: each rule fired is one proof obligation, identical to LOOM's
current ISLE model. Full saturation (egg / egglog) explodes the trace
and complicates the per-rewrite proof, and it can produce extractions
that no single rule chain explains.

**Where it'd help.** Three high-value wins: (a) global value numbering
across basic blocks (current LOOM CSE is block-local); (b)
rematerialization decisions for the constant-CSE cost gate already
shipped in v0.6.0 (extractor cost model is the natural home);
(c) gale's `match (state, event)` FSM tables (source-pattern-analysis
§7) get a unified representation where the discriminant equivalences
are already canonicalized.

**Effort.** L. The data structure itself is ~2000 LOC (egg-style
hashcons + union-find + acyclic level annotation). Rule porting is
the dominant cost: every ISLE rule needs an extraction-cost
annotation. 6-10 weeks.

**Concrete next step.** Prototype an aegraph wrapping LOOM's existing
`Value` term IR, drive it from a tiny subset of ISLE rules (the
constant-folding family), and benchmark extraction-time vs current
ISLE on the corpus. Don't port everything — prove the round-trip
works on simple rules first. Reference impls: Cranelift's
`cranelift_egraph` crate, Steuwer's
[slotted-egraphs](https://github.com/memoryleak47/slotted-egraphs/)
(handles binders, useful if LOOM later needs to reason about
`local.get` as bound variable).

## 2. SSA-form mid-end transforms

**Mechanism.** Convert stack-IR to SSA, run GVN / SCCP / sparse CCP,
convert back. The SSA → wasm-stack direction is *lossy*: toolchains
emit Relooper-style structured CFG and the wasm runtime then has to
recover the original CFG
([troubles.md](http://troubles.md/why-do-we-need-the-relooper-algorithm-again/)).
V8 moved *away* from Sea-of-Nodes back to a CFG IR in March 2025 — a
vote against SSA-roundtrip in the wasm ecosystem.

**Fit with proof-first.** Partial. SSA conversion is verified
transformation territory but out-of-SSA on a stack machine introduces
register-allocation choices that LOOM's verifier would need to track.

**Where it'd help.** Marginal over ægraph — the ægraph subsumes GVN
and sparse CCP. Effort: XL (3+ months).

**Concrete next step.** *Don't*. Adopt ægraph (§1) instead.

## 3. Polyhedral / loop-nest

**Mechanism.** Represent loop iteration spaces as polytopes, transform
via ILP-based schedulers (Pluto, isl). Strong on dense linear algebra.
[ACM TACO 2024 survey](https://dl.acm.org/doi/10.1145/3674735).

**Fit with proof-first.** Partial — polytope solvers are large
unverified dependencies.

**Where it'd help.** Gale's `decreases`-bounded loops (24 sites) are
*not* polyhedral candidates — fixed-trip 16/32/64-iteration loops
with no nested affine indexing. Effort: XL for polyhedral, S for the
gale-shaped subset.

**Concrete next step.** Skip polyhedral. Build a bounded-trip-count
unroller that recognizes `local.set i; (loop ... br_if; i = i+1)`
with a provably-bounded constant entry value, monotonic decrement,
zero comparison. Unroll when N ≤ 8. ~400 LOC, Z3 obligation
"unrolled body ≡ N applications of loop body."

## 4. Profile-guided / sample-driven

**Mechanism.** Runtime samples / edge counts feed cost-model inputs.
SPE-driven PGO recently hit >99% hotspot fidelity at <5% overhead
([arXiv:2507.16649](https://arxiv.org/abs/2507.16649)).

**Fit with proof-first.** Yes for *cost decisions*; the transforms
themselves are still verified normally.

**Where it'd help.** Inlining cutoff, branch ordering, dispatch-table
speculation. Effort: M ingest + L wasmtime-side harness LOOM doesn't
control.

**Concrete next step.** **Defer.** LOOM is offline; build static
branch heuristics first (errno-style negative-constant compares
predict "error path cold") as a no-runtime substitute for the gale
workload.

## 5. Equality saturation with cost models

Same family as §1 but with full saturation. **Skip** for proof-first
reasons.
[Slotted E-Graphs (PLDI 2025)](https://steuwer.info/files/publications/2025/PLDI-Slotted-E-Graphs.pdf)
and [egglog (PLDI 2025
tutorial)](https://pldi25.sigplan.org/details/pldi-2025-tutorials/4/Unlocking-Optimizations-with-egglog-Equality-Saturation-Meets-Datalog)
are literature to watch *only* if LOOM later needs higher-order
rewrites over component-model resource adapters with binders.

## 6. Superoptimization (Souper-style)

**Mechanism.** Enumerate short instruction sequences, ask an SMT
solver for equivalent shorter forms, harvest positive answers as
rules. [Souper](https://github.com/google/souper) does this for
LLVM IR; [Cabrera Arteaga
2020](https://arxiv.org/pdf/2002.10213) ports the pipeline to wasm
via Souper-over-LLVM (8/12 Rosetta programs shrunk). Recent:
[Minotaur (OOPSLA 2024)](https://users.cs.utah.edu/~regehr/minotaur-oopsla24.pdf)
for SIMD; [PrediPrune
(2025)](https://arxiv.org/pdf/2509.16497) cuts verification overhead
via ML-pruned candidate filtering.

**Fit with proof-first.** **Native.** Every harvested rule is
already SMT-discharged; LOOM only needs to re-verify each candidate
in its own Z3 encoding before adoption.

**Where it'd help.** Rule-database growth. v0.6.0's hand-written
ISLE rules cover a few hundred patterns; Souper-style mining could
10× that on the gale + httparse + nom_numbers corpus. The bit-manip
family (cpu_mask / event / ipi, source-pattern-analysis §6) is
*exactly* what Souper excels at.

**Effort.** M. ~1500 LOC for the Souper-LOOM bridge (candidate
extraction, Souper-IR serialization, results ingest, re-verify, ISLE
emission), 4-5 weeks. Minotaur is the closest reference.

**Concrete next step.** 2-week spike: vendor Souper as external
tool, run offline on v0.6.0 ISLE rules, harvest into
`loom-isle/synthesized.isle`, gate by Z3 re-verification. Even 10
net-new rules is a win.

## 7. Peephole synthesis (LLM-assisted / specification-mining)

**Mechanism.** Either (a) LLM proposes peephole candidates, SMT
verifies (LPO / Lampo —
[arXiv:2508.16125](https://arxiv.org/abs/2508.16125)), or (b)
Hydra-style program-synthesis generalizes a small example to a rule
family ([Hydra OOPSLA
2024](https://users.cs.utah.edu/~regehr/generalization-oopsla24.pdf)).

**Fit with proof-first.** **Native** (same story as §6 — Alive2 /
Z3 is the trust root, the LLM is just a proposal source). LPO found
28 LLVM bugs in 11 months, all SMT-checked.

**Where it'd help.** Same surface as §6 but with *generalization*: a
single concrete hoist-guard counterexample from gale becomes a rule
family that applies to all default-then-override patterns
(source-pattern-analysis §2).

**Effort.** M. The LLM bridge is the easy part; the prompt
engineering and dedup pipeline is what eats time.

**Concrete next step.** Bundle this with §6 — same infrastructure
(SMT verifier, ISLE emitter, candidate dedup). Add LLM proposals as
a second source feeding the same verification pipeline.

## 8. Memory-layout / alias analysis

**Mechanism.** Heap2Local-style escape analysis: prove an allocation
doesn't escape, replace with locals
([Binaryen Heap2Local](https://github.com/WebAssembly/binaryen/blob/main/src/passes/Heap2Local.cpp),
active 2026). Structure-of-arrays, alias sets via Andersen.

**Fit with proof-first.** Yes — escape proofs are bounded
reachability, Z3-checkable.

**Where it'd help.** Gale uses `[Option<Thread>; 64]` arrays heavily.
Pre-GC wasm has no `struct.new` target, but the analysis transfers
to linear-memory bump-allocations emitted by every gale function
returning a `Result`. Effort: M (~1500 LOC).

**Concrete next step.** Target `(memory.grow N) ... store ... load`
patterns where the address provably traces back to the grow result
and doesn't escape. Measure gale residue first — useful only if
LLVM leaves non-trivial bump-alloc behind.

## 9. Function-summary-driven IPA

**Mechanism.** For each function compute a small summary
(`{ pure: bool, no_trap: bool, constant_return: Option<Value>,
side_effects: BitSet<Memory|Global|Table>, may_throw: bool }`).
Callers use the summary to make local decisions: pure no-trap
functions are CSE-eligible across calls; constant-return functions
fold at the call site; no-side-effect functions are eligible for
dead-call elimination. Direct extension of LOOM's existing
`has_unsupported_isle_instructions` gate, just *more* than just
"unknown."

**Fit with proof-first.** **Native and easy.** Each summary field is
a single Z3 query computed during function-level analysis. Summaries
*are* the proof obligations; callers' soundness follows from the
summary's correctness.

**Where it'd help.** This is the single highest-leverage change for
the gale-style FFI workload. Every `sched_*` function
(source-pattern-analysis §1) is pure and has a closed-form return on
specific inputs. Today LOOM's CSE, DCE, and vacuum all bail on `Call`
instructions because they can't model effects; a purity summary
unlocks all three across function boundaries.

**Effort.** M. ~800 LOC for the summary computation + propagation
fixedpoint + per-pass query API. 2-3 weeks. The fixedpoint is
straightforward because wasm has no recursion through indirect calls
once tables are immutable (Cranelift uses this same property).

**Concrete next step.** Add a `FunctionSummary` struct to
`loom-shared` with `{ pure, no_trap, constant_return,
reads_memory, writes_memory, reads_global, writes_global }`.
Compute via a single forward pass over each function's instructions,
combining recursively over called functions to fixedpoint. Wire into
the existing CSE eligibility check first (smallest valuable PR);
DCE-across-calls follows.

## 10. Control-flow specific

**Mechanism.** Small targeted CFG transforms: tail-call promotion
(`call; return` → `return_call`), branch fusion (cascaded `br_if`
with same target), jump threading, `br_table` densification, and
dispatch-table speculation (the `directize` pass already in the
wasm-opt gap analysis).

**Fit with proof-first.** Yes. Each is a textbook CFG rewrite with
straightforward Z3 encoding. Wasm 3.0 made `return_call` baseline in
2025 ([web.dev](https://web.dev/blog/wasmgc-wasm-tail-call-optimizations-baseline)).

**Where it'd help.** Gale's tail-call / dispatch-only `match` pattern
(source-pattern-analysis §4) — 12 sites ending in `call $f; return`.
Each promotion saves a stack frame.

**Effort.** Per-transform S. Bundle the four into one release:
tail-call promotion (300 LOC), branch fusion (200), jump threading
(400), `br_table` densification (500). ~1400 LOC, 3 weeks.

**Concrete next step.** Tail-call promotion first — smallest delta,
trivial encoder support, one-line Rocq lemma.

## 11. Component-Model-aware

**Mechanism.** Wasm components compose via the Canonical ABI:
cross-component calls go through lift/lower adapters that copy and
re-encode across linear memory
([CanonicalABI spec](https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md)).
After meld fuses components, adapters remain as identifiable
patterns: `lower_string` → `memory.copy` → `lift_string` round-trips
on fixed-shape values are no-ops; dead `realloc` calls are removable;
non-escaping resource handle pairs can be unboxed. WASI 0.3 (Aug
2025) adds zero-copy buffer forwarding — more shape patterns.

**Fit with proof-first.** Yes. Adapter patterns are structural
rewrites with canonical-ABI semantics as spec; no new Z3 theory
needed.

**Where it'd help.** LOOM's *moat* — no upstream optimizer sees
post-fusion components, because wasm-opt works on core wasm and
doesn't know about adapters. Even small wins compound across every
component-model call. Effort: M (~1000 LOC, 3-4 weeks); hardest
part is enumerating ~30 canon built-ins.

**Concrete next step.** `lift_string ∘ lower_string` identity
collapse, gated on "intermediate buffer has no other readers."
Single PR isolates component-model awareness without committing to
the whole adapter catalog.

## 12. ML-driven

**Mechanism.** RL / supervised models for inlining, pass ordering,
cost-model parameters. Canonical refs:
[MLGO](https://arxiv.org/pdf/2101.04808),
[Liang pass-ordering](https://proceedings.mlr.press/v202/liang23f/liang23f.pdf),
[Next 700 ML-enabled optimizations
(CC 2024)](https://dl.acm.org/doi/10.1145/3640537.3641580).

**Fit with proof-first.** **Unfit as policy.** Fit only as a *meta*
tool: a learned model that *picks among verified rewrites* is
acceptable; a model that *invents rewrites* is not. Boundary: is the
output Z3-checked before commit.

**Where it'd help.** Cost-gate tuning only (constant-CSE break-even,
inlining cutoff). Effort: L for any real deployment.

**Concrete next step.** **Defer.** Spend the budget on §9 — purity
summaries unlock more performance than any learned cost model will,
at a fraction of the operational cost.

## 13. Verification-aware optimization

**Mechanism.** Rewrites whose goal is to *expand the verifier's
reasoning surface*, not the optimization surface directly. Example:
`BrIf` with symmetric result construction in both arms is hard to
reason across; lowering to `Select` (a single linear expression)
makes every downstream pass more powerful. Same idea: canonicalize
`local.tee X; expr; local.get X` to a uniform shape before CSE so
CSE sees more congruent windows.
[Crocus ASPLOS 2024](https://cfallin.org/pubs/asplos2024_veri_isle.pdf)
is the inspiration.

**Fit with proof-first.** **Native.** Each rewrite is a
semantics-preserving canonicalization with a one-line Z3 proof; the
gain is multiplicative on every other pass.

**Where it'd help.** v0.5.0/v0.6.0 follow-ups: sharper CSE cost-gate
decisions, more DSE hits when `BrIf` arms collapse to `Select`
first, more vacuum fuel from constant-folding flowing through
canonicalized branches. Effort: S per rewrite, M (~1000 LOC) for
~10 canonicalizers.

**Concrete next step.** `BrIf` with identical-shape both arms (same
result type, side-effect-free) → `Select`. Single ISLE rule, ~50
LOC, immediately measurable on gale's `sched_*` `if errno return
...; else continue ...` spine.

---

## What v0.7.0 actually looks like, ranked by ROI

Top-5 from the verdict, scheduled by dependency:

1. **§9 Function summaries** — 2-3 weeks, unlocks cross-call CSE /
   DCE / vacuum for every later pass. Do first.
2. **§13 Verification-aware canonicalization** — 2 weeks, multiplies
   §9's impact and v0.6.0's already-shipped passes.
3. **§10 Control-flow bundle** — 3 weeks, all small, all independently
   verifiable, directly relevant to gale's dispatch patterns.
4. **§6/§7 Souper + LLM peephole synthesis** (combined infra) — 4-5
   weeks. Once running, rules accumulate continuously.
5. **§11 Component-model adapter specialization** — 3-4 weeks, LOOM's
   unique value-add vs wasm-opt.

Picks 1 and 11 are LOOM's strategic moat (no upstream tool offers
them). Picks 2-4 are parity-and-beyond work.

The **ægraph mid-end (verdict-1, §1)** is omitted from this schedule
because it's the v0.8.0 effort — once §9 + §13 prove that summary-
and-shape work makes individual passes more powerful, the ægraph is
the natural unification step. Doing it before §9 + §13 means porting
passes that are then going to change shape.

---

## Sources

### E-graphs and equality saturation
- [The acyclic e-graph: Cranelift's mid-end optimizer (Fallin, 2026)](https://cfallin.org/blog/2026/04/09/aegraph/)
- [Cranelift egraph RFC](https://github.com/bytecodealliance/rfcs/blob/main/accepted/cranelift-egraph.md)
- [ægraphs: Acyclic E-graphs (EGRAPHS 2023)](https://pldi23.sigplan.org/details/egraphs-2023-papers/2/-graphs-Acyclic-E-graphs-for-Efficient-Optimization-in-a-Production-Compiler)
- [egg: Fast and Extensible Equality Saturation (POPL 2021)](https://dl.acm.org/doi/pdf/10.1145/3434304)
- [Slotted E-Graphs (PLDI 2025)](https://steuwer.info/files/publications/2025/PLDI-Slotted-E-Graphs.pdf)
- [DialEgg: Dialect-Agnostic MLIR Optimizer (CGO 2025)](https://dl.acm.org/doi/abs/10.1145/3696443.3708957)
- [Unlocking Optimizations with egglog (PLDI 2025 tutorial)](https://pldi25.sigplan.org/details/pldi-2025-tutorials/4/Unlocking-Optimizations-with-egglog-Equality-Saturation-Meets-Datalog)

### Superoptimization and verified peephole
- [Souper (Google)](https://github.com/google/souper)
- [Superoptimization of WebAssembly Bytecode (Cabrera Arteaga, 2020)](https://arxiv.org/pdf/2002.10213)
- [Minotaur: SIMD-Oriented Synthesizing Superoptimizer (OOPSLA 2024)](https://users.cs.utah.edu/~regehr/minotaur-oopsla24.pdf)
- [PrediPrune: Reducing Verification Overhead in Souper (2025)](https://arxiv.org/pdf/2509.16497)
- [Hydra: Generalizing Peephole Optimizations with Program Synthesis (OOPSLA 2024)](https://users.cs.utah.edu/~regehr/generalization-oopsla24.pdf)
- [LPO: Discovering Missed Peephole Optimizations with Large Language Models (ASPLOS 2026)](https://doi.org/10.1145/3779212.3790184)
- [Lampo / arXiv:2508.16125](https://arxiv.org/abs/2508.16125)
- [Practical Verification of Peephole Optimizations with Alive (CACM 2018)](https://web.ist.utl.pt/nuno.lopes/pubs/alive-cacm18.pdf)
- [AliveInLean (CAV 2019)](https://link.springer.com/chapter/10.1007/978-3-030-25543-5_25)

### Cranelift verification and ISLE
- [Crocus: Lightweight Modular Verification for WebAssembly-to-Native (ASPLOS 2024)](https://cfallin.org/pubs/asplos2024_veri_isle.pdf)
- [wasmtime/cranelift/isle/veri](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/isle/veri/README.md)

### SSA / stack machine
- [WebAssembly Troubles part 2: Relooper (Fransham)](http://troubles.md/why-do-we-need-the-relooper-algorithm-again/)
- [V8 announcement moving away from Sea-of-Nodes (2025, via lobste.rs thread)](https://lobste.rs/s/h8hlp7/wasm_is_not_quite_stack_machine)

### Memory / GC / Heap2Local
- [Binaryen Heap2Local source](https://github.com/WebAssembly/binaryen/blob/main/src/passes/Heap2Local.cpp)
- [GC Optimization Guidebook (Binaryen wiki)](https://github.com/WebAssembly/binaryen/wiki/GC-Optimization-Guidebook)
- [V8 wasm-gc-porting blog](https://v8.dev/blog/wasm-gc-porting)

### Profile-guided
- [From Profiling to Optimization: Unveiling PGO (arXiv:2507.16649)](https://arxiv.org/abs/2507.16649)
- [FOSDEM 2025: PGO in LLVM](https://archive.fosdem.org/2025/schedule/event/fosdem-2025-4109-profile-guided-optimization-pgo-in-llvm-current-challenges-from-the-adopter-perspective/)

### Component Model
- [Canonical ABI spec](https://github.com/WebAssembly/component-model/blob/main/design/mvp/CanonicalABI.md)
- [WASI 0.3 native async preview (Aug 2025)](https://progosling.com/en/dev-digest/2025-08/wasi-0-3-native-async-aug-2025)

### Tail calls / control flow
- [WebAssembly tail-call proposal](https://github.com/WebAssembly/tail-call/blob/main/proposals/tail-call/Overview.md)
- [WasmGC and Wasm tail calls baseline (web.dev, 2025)](https://web.dev/blog/wasmgc-wasm-tail-call-optimizations-baseline)
- [WebAssembly 3.0 release notes (2025)](https://www.x-cmd.com/blog/250924/)

### ML-driven
- [MLGO (arXiv:2101.04808)](https://arxiv.org/pdf/2101.04808)
- [Learning Compiler Pass Orders (ICML 2023)](https://proceedings.mlr.press/v202/liang23f/liang23f.pdf)
- [The Next 700 ML-Enabled Compiler Optimizations (CC 2024)](https://dl.acm.org/doi/10.1145/3640537.3641580)

### Loop optimization
- [A Survey of General-purpose Polyhedral Compilers (ACM TACO 2024)](https://dl.acm.org/doi/10.1145/3674735)
- [WCET-Aware Loop Unrolling](https://www.tuhh.de/es/esd/research/wcc/optimizations/loop-unrolling)

### Internal references (read-only)
- `/Users/r/git/pulseengine/loom/docs/research/gale-v0.5.0/wasm-opt-gap-analysis.md`
- `/Users/r/git/pulseengine/loom/docs/research/gale-v0.5.0/source-pattern-analysis.md`
- `/Users/r/git/pulseengine/loom/docs/research/gale-v0.4.0/measurement-report.md`
- `/Users/r/git/pulseengine/loom/docs/research/mutation-perf-inference-eval.md`
