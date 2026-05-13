# Algorithmic / solver-based optimization feasibility for LOOM

Date: 2026-05-03
Author: read-only feasibility scan
Scope: ranks six families of "let a solver find faster code"
against LOOM's proof-first constraint. Builds on
`docs/research/v0.7.0/optimization-methods-survey.md` and adds
the families not already covered there, treats the literal "ask
Z3 to simplify" question explicitly, and ranks for v0.7.0/v0.8.0.

---

## Verdict (top of file for skimming)

**Adopt Souper-style verified peephole synthesis (family 4) for
v0.7.0.** It is the only family that natively matches LOOM's
proof-first philosophy (every harvested rule is SMT-discharged
before adoption), reuses the existing Z3 verifier and ISLE
pipeline, has a 4-5 week budget that fits one release, and
directly hits the gale bit-mask family (`cpu_mask`, `event`,
`ipi`) where identities like `(m & (m-1)) == 0` and `a & !a == 0`
are exactly what enumerative SMT search excels at. Family 5
(arithmetic identities) falls out of family 4 for free. Family 3
(idiom recognition: popcount, memcpy, strlen) is a strong v0.8.0
follow-up. Families 1 (raw superoptimization), 2 (equality
saturation), and 6 (algorithmic-complexity recognition) are not
recommended for v0.7.0.

Asking Z3 directly to simplify a sub-expression (the literal
"algorithmic algorithm solver" reading) is **viable but limited**
— see §"Direct Z3 simplification". Souper does exactly this,
bounded; LOOM should adopt Souper's shape rather than wire
`Z3_simplify` directly.

---

## The six families, ranked by LOOM-ROI

### 1. Superoptimization (raw enumeration)  — defer

**Mechanism.** Enumerate all instruction sequences up to length k,
ask SMT to prove equivalence. Cabrera Arteaga 2020
([arXiv:2002.10213](https://arxiv.org/pdf/2002.10213)) shrunk 8/12
Rosetta wasm programs.

**Fit:** native — every accepted rewrite is SMT-validated.

**LOOM-side cost:** XL. Wasm-stack enumerator (typed BFS plus
stack-discipline pruning) ~2000 LOC; orchestration (queue / dedup /
re-verify / ISLE emit) ~1000-1500 LOC. 6-8 weeks before any rule
lands.

**Payoff on gale:** indirect. Most enumerated rewrites are
micro-shuffles around `local.get`/`local.tee`. Real gale wins are
*structural* (FSM const-fold, dead-store, bounded unroll), not
micro-peephole.

**Risk:** combinatorial blowup; Z3 queue exhaustion. PrediPrune
([arXiv:2509.16497](https://arxiv.org/pdf/2509.16497)) mitigates
but adds operational complexity.

**Recommendation:** don't build a standalone enumerator. Reuse
Souper (family 4). **Defer to v0.9.0+.**

### 2. Equality saturation  — defer

Covered in `optimization-methods-survey.md` §1, §5. Decision unchanged:
ægraph (acyclic) is the v0.8.0 strategic move; full saturation is
unfit. Not re-evaluated here because the proof-trace complications
are the same regardless of how the question is framed.

### 3. Algorithm / idiom recognition  — v0.8.0 candidate

**Mechanism.** Pattern-match a loop body against a fixed catalog of
idioms (popcount, strlen, memcpy, memset, ctz, bswap) and replace
with a dedicated wasm instruction or host import. LLVM's
[`LoopIdiomRecognize`](https://llvm.org/doxygen/LoopIdiomRecognize_8cpp_source.html)
is the reference (strided load/store → `memcpy`,
`NclPopcountRecognize` for Kernighan popcount).

**Fit with proof-first:** yes. Each idiom is one whole-loop
equivalence theorem, provable once. Catalog is small (≤10).
Verus bit-vector lemmas already in gale align with the proofs
needed.

**LOOM-side cost:** M. ~600-1000 LOC per idiom. First three
(popcount, memcpy-loop, memset-loop) for ~1800 LOC, 3-4 weeks.
Proof obligation is one-shot per idiom, not per-application.

**Payoff on gale:** moderate-to-high. Three concrete sites:
- `compute_ipi_mask` (`src/ipi.rs:109-141`) — popcount-style mask
  in an unrolled bounded loop. Direct popcount-instr win.
- `runq_shift_left` (`src/sched.rs:80-105`) — tagged
  `#[verifier::external_body]`, the LLVM-loopidiom memmove case.
- `cpu_mask` power-of-two test (`src/cpu_mask.rs:81-86`) —
  `mask != 0 && (mask & (mask - 1)) == 0` canonicalization.

**Risk:** low. Idioms are conservative; on shape miss, skip.

**Recommendation:** ship in v0.8.0 after family-4 infrastructure
is mature. Shares verification infrastructure with family 4 but is
operationally independent.

### 4. SMT-driven peephole synthesis (Souper-style)  — **v0.7.0 PICK**

**Mechanism.** For each window of instructions, extract a
predicate-aware SMT formula; enumerate shorter candidates; ask Z3
whether each is equivalent under the path-condition prefix.
Harvest equivalent shorter forms as new rules.
[Souper](https://github.com/google/souper) does this for LLVM IR;
LPO 2026 ([ASPLOS](https://doi.org/10.1145/3779212.3790184))
swaps the candidate source for LLM proposals on the same
SMT-validation backend.

**Fit with proof-first:** **native**. Every harvested rule arrives
with a Z3 proof certificate. LOOM only needs to re-verify in its
own encoding before committing — the same machinery
`verify_rules.rs` (2172 LOC, already present) uses for
hand-written ISLE rules. Zero new trust surface.

**LOOM-side cost:** M. ~1500 LOC, 4-5 weeks:
- Window extractor over `Instruction` slices → candidate format (~400)
- Path-condition tracker (reuses `stack.rs` shape data, ~300)
- Candidate enumerator over a small bitvector alphabet
  (const, local.get, add/sub/and/or/xor/shl/shr — ~400)
- Z3 re-verification harness extending `verify_rules.rs` (~200)
- ISLE emitter to `loom-isle/synthesized.isle` gated on re-verify (~200)

The enumerator is the only novel work; everything else plumbs into
existing infrastructure (`verify.rs`, `verify_rules.rs`,
`summary.rs`, the ISLE rule loader).

**Payoff on gale:** **high** on bit-manip. Source-pattern §6
targets:
- `(events | new_events) & events == events`  (post-monotonic)
- `value & !value == 0u32`  (set/clear roundtrip)
- `(events | new_events) | new_events == events | new_events`  (idempotent)
- `(current | enable) & !disable`  (mask composition)

These are literal Souper-shape inputs — short bitvector
expressions over ≤4 free variables, Z3-solvable in <100ms each.
Cabrera Arteaga 2020 shrank 8/12 Rosetta wasm programs on a
generic corpus; gale's bit-vector density is higher, so hit rate
should be at least as good.

**Strategic upside.** The rule database becomes *self-growing*.
v0.6.0's hand-written ISLE rules cover a few hundred patterns;
Souper mining on the gale + httparse + nom_numbers corpus
plausibly 5-10×s that within a release cycle, all under proof.
Only family on this list where the optimizer learns from its
corpus.

**Risk.** Z3 load bounded (small candidates, ≤3s timeout, offline
queue). Re-verification at adoption is the trust anchor — even
if Souper's solver were wrong, LOOM's verifier catches it.
Synthesized rules are *additional* ISLE rewrites, fired by the
same engine, subject to the existing CSE/vacuum cost gates. No
proof-load explosion: synthesis is offline; only adoption-time
re-verification runs on the build path.

**v0.7.0 plan:** week 1 vendor Souper + write IR exporter;
week 2 harvest on gale + httparse fixtures; week 3 re-verify in
LOOM's encoding (expect ~10% discard on theory/bitwidth quirks);
week 4 ISLE emission + pipeline integration; week 5 differential
testing + corpus dogfooding.

### 5. Constraint-solving for arithmetic identities  — **subsumed by 4**

**Mechanism.** Ask Z3 whether two expressions are equivalent
(e.g. `x*8` vs `x<<3`); keep the cheaper.

**Fit:** native. **Cost:** ~0 additional LOC if family 4 ships —
the family-4 enumerator already covers `*2^k → shl`, `/2^k →
shr_u` (signed div correctly rejected on rounding difference),
and standard bitwise identities. Hand-coded const-fold rules
already exist in `loom-isle/rules/constant_folding.isle`.

**Payoff:** delivered for free by family 4. Without family 4, a
hand-coded catalog of 20-30 identities is 200-400 LOC / 1 week
but loses the self-growth property.

**Recommendation:** treat as a deliverable of family 4. Do not
schedule independently.

### 6. Algorithmic-complexity recognition  — **not applicable**

Detect that an inner loop has worse asymptotic complexity than
necessary (O(n²) where O(n) exists) and swap algorithm. Would
require whole-program matching against a library of algorithmic
schemata, each with a non-trivial equivalence proof — an open
research problem (program synthesis, not peephole). No production
compiler does this. **Skip permanently.**

---

## Direct Z3 simplification — the literal question

Can LOOM ask Z3 to simplify a sub-expression and replace the
original if shorter? **Yes, mechanically.** Z3 exposes
`Z3_simplify` (rewrite engine) and the `ctx-solver-simplify`
tactic (full SMT to discharge sub-formula truth values in context)
([Z3 tactics summary](https://microsoft.github.io/z3guide/docs/strategies/summary/),
[Z3 issue #424](https://github.com/Z3Prover/z3/issues/424)).

**Naïve approach:** `Instruction` window → Z3 BV expression →
`simplify()` → decode back → compare sizes → keep shorter.

**Why it is the wrong shape for LOOM:**

1. **Z3's `simplify` is not optimization-aware.** It canonicalizes
   (constant folding, BV normal form, boolean simplification);
   it does not search for shorter forms, and the canonical output
   may be *larger* than the input
   ([Z3 issue #6694](https://github.com/Z3Prover/z3/issues/6694)).
2. **Decoding Z3 BV expressions back to wasm is lossy.** Z3 has
   no notion of `local.tee` vs `local.set + local.get`, operand
   stack, or pre-existing CSE bindings — the round-trip would un-do
   v0.6.0's CSE cost-gating.
3. **`ctx-solver-simplify` is heavyweight** — full SMT per
   sub-formula, seconds per window. Not viable in a build-path
   pipeline.

**Has anyone done this?** Yes — Souper. Souper's value-add over
"call Z3 simplify" is that it (a) bounds search with an explicit
candidate enumerator, (b) uses SMT only to *verify*, not to
*propose*, and (c) is offline so solver cost doesn't block
compilation. This is precisely why family 4 (Souper-shaped
pipeline) is the recommendation rather than wiring `Z3_simplify`
directly.

---

## ROI ranking

| Rank | Family | When | LOC | Weeks | Risk | Gale payoff |
|---|---|---|---|---|---|---|
| 1 | **4. SMT-driven peephole synthesis (Souper-shaped)** | v0.7.0 | ~1500 | 4-5 | low | high (bit-mask family) |
| 2 | 5. Arithmetic identities | folds into 4 | 0 add'l | 0 | low | medium (subsumed) |
| 3 | 3. Idiom recognition (popcount/memcpy) | v0.8.0 | ~1800 | 3-4 | low | medium-high (3 sites) |
| 4 | 1. Raw superoptimization | v0.9.0+ | 3000+ | 6-8 | medium | low-medium |
| 5 | 2. Equality saturation | v0.8.0 (ægraph only) | ~2000 | 6-10 | medium | strategic, not numeric |
| 6 | 6. Algorithmic-complexity recognition | n/a | n/a | n/a | n/a | n/a |

Picks 1+2 fit cleanly in v0.7.0 with the picks already scheduled in
`optimization-methods-survey.md` (function summaries §9 already
shipped as `loom-core/src/summary.rs` 433 LOC; canonicalization §13;
control-flow bundle §10). Pick 3 sequences naturally into v0.8.0
alongside the ægraph effort.

---

## Why family 4 wins, summarized

- Only family that ships an extensible rule database — every other
  caps at hand-written rules.
- Zero new trust surface — adoption-time re-verification is
  structurally identical to what `verify_rules.rs` already does.
- Targets gale's densest optimization-relevant pattern
  (bit-manip §6) — orthogonal to other v0.7.0 picks.
- Coexists with the rest of the v0.7.0 backlog — synthesized
  rules feed the same engine as canonicalization (§13) and
  summary (§9). Synthesis is offline; only adoption-time
  re-verify runs on the build path. No proof-load explosion.

---

## Sources

- [Souper: A superoptimizer for LLVM IR](https://github.com/google/souper)
- [Superoptimization of WebAssembly Bytecode (Cabrera Arteaga, 2020)](https://arxiv.org/pdf/2002.10213)
- [PrediPrune: Reducing Verification Overhead in Souper (2025)](https://arxiv.org/pdf/2509.16497)
- [LPO: Discovering Missed Peephole Optimizations with LLMs (ASPLOS 2026)](https://doi.org/10.1145/3779212.3790184)
- [Z3 Tactics Summary — simplify, ctx-simplify, ctx-solver-simplify](https://microsoft.github.io/z3guide/docs/strategies/summary/)
- [Z3 Issue #424 — Documentation on Z3_simplify()](https://github.com/Z3Prover/z3/issues/424)
- [Z3 Issue #6694 — Questions on simplification](https://github.com/Z3Prover/z3/issues/6694)
- [LLVM LoopIdiomRecognize.cpp (source)](https://llvm.org/doxygen/LoopIdiomRecognize_8cpp_source.html)
- [Programming Z3 (Bjørner, Stanford notes)](https://theory.stanford.edu/~nikolaj/programmingz3.html)

### Internal references
- `/Users/r/git/pulseengine/loom/docs/research/v0.7.0/optimization-methods-survey.md` (companion survey — picks 1, 2, 5 from there)
- `/Users/r/git/pulseengine/loom/docs/research/gale-v0.5.0/source-pattern-analysis.md` §6 (bit-manip targets)
- `/Users/r/git/pulseengine/loom/loom-core/src/verify_rules.rs` (2172 LOC, existing rule-verification harness — extension point for Souper re-verification)
- `/Users/r/git/pulseengine/loom/loom-core/src/summary.rs` (433 LOC, function-summary IPA shipped in v0.6.x — pattern for adding new analyses)
- `/Users/r/git/pulseengine/loom/loom-shared/isle/rules/` (where synthesized rules would land as `synthesized.isle`)
