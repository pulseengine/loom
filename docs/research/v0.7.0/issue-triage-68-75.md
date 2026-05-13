# Issue Triage #68–#75 — Post v0.6.0 Planning

**Verdict (top 3 to pursue for v0.7.0):**
1. **#70 — Optimize meld-fused async callback adapters** (concrete, leverages existing fused-optimizer infra; direct value to meld/gale)
2. **#68 — Cross-component optimization passes (Tier 1 only)** (force-multiplier; subset already exists, scope expansion is incremental)
3. **#71 — Island-model parallel optimization** (medium effort, large UX win on multi-core devs, preserves Z3 story)

**Recommend closing #69** — it is a *merged PR* (`feat: rivet integration, ISLE fix (#55)…`, 2026-04-12), not an issue. **Recommend deferring #72, #73, #74, #75** with notes below; treat #74 as research-shaped, not implementation-shaped.

Note: #69 in the brief turned out to be a merged PR record (`state: MERGED`), not an open issue. Confirmed via `gh issue view 69 --json state`.

---

## Per-issue analysis

### #68 — Cross-component optimization passes for meld-fused modules
**Created** 2026-04-09. Broad RFC-style issue listing 14 optimization ideas in 4 tiers: adapter elimination, whole-program analysis, memory opt, advanced LTO (escape analysis, devirt, loop fusion, partial eval).
**Status vs v0.6.0:** Partially shipped. `loom-core/src/fused_optimizer.rs:117` already implements memory-import dedup, same-memory adapter collapse, adapter devirtualization, function-type dedup, dead-function elim, import dedup (six of #68's items). Tier-1.1 (scalar adapter inlining) and Tier-2.2 (function body deduplication) are **not** shipped.
**Effort:** Issue is XL as written; Tier-1 subset is M (~600–900 LOC + tests + Rocq stub). Tier-3/4 each individually L–XL.
**Strategic value:** High force-multiplier. Every meld-fused workload (gale included) gets compounding benefit. Tier-1.1 specifically unlocks #70's prerequisites.
**Risk:** Tier 1 is low-risk (we have prior art in `fused_optimizer.rs`). Tier 4 items (escape analysis, partial evaluation) put the proof-first invariant under heavy pressure and should not be attempted before v0.8+.
**Recommendation:** Pursue **only Tier 1 + 2.2** in v0.7. Split into a child issue.

### #70 — Optimize meld-fused async callback adapter patterns
**Created** 2026-04-12. Concrete 6-pass plan for stripping the P3 async wrapper down to a direct call: inline `[async-lift]` → devirtualize `call_indirect` through the element segment → inline callback → DCE the loop → forward task.return shim → DCE the start_task setup.
**Status vs v0.6.0:** Not shipped. Current inliner (`loom-core/src/lib.rs:11827–11907`) only handles small/single-call-site direct calls — no element-segment devirt (`CallIndirect` is in the IR at `lib.rs:807, 1439, 2745` but skipped by the optimizer per the docstring at `lib.rs:6849`).
**Effort:** M–L. Indirect-call devirt via element-segment analysis is ~300 LOC, the rest is composition over existing passes. ~3 weeks including Z3 verification of the devirt step.
**Strategic value:** Direct win for any meld P3 output. Currently leaves ~10 instructions of overhead per async-completing call. Cascades into #68 (the devirt + inlining framework is shared).
**Risk:** Medium. Element-segment-based devirtualization needs a careful proof — table-mutation must be ruled out. The proof shape is familiar (similar to indirect-call elimination in Wasmtime's wasm-opt).
**Recommendation:** Pursue. Sequence after a thin slice of #68 (adapter inlining infra).

### #71 — Island-model parallel optimization
**Created** 2026-04-12. Run N optimizer pipelines in parallel with varying configurations (pass order, inline threshold), Z3-verify each, pick the smallest verified result.
**Status vs v0.6.0:** No parallelism in the workspace today (`grep rayon|par_iter|thread::spawn` in `loom-core/src/lib.rs` returns empty; no `rayon` dep in `Cargo.toml`).
**Effort:** M. ~400 LOC plumbing + `rayon` dependency + CLI flag. Each island reuses `optimize_module` unchanged.
**Strategic value:** Directly addresses the v0.6.0 gale regression discovery loop — different pass orderings would have surfaced the CSE-cost issue automatically (see `afc9318 fix(cse): cost-aware dedup gate`). Helps every workload, not just fused ones.
**Risk:** Low — each island is independent and each result must independently pass the existing Z3 + stack-validation gates at `lib.rs:6843–6873`. No new correctness surface.
**Recommendation:** Pursue. Lowest correctness risk of the three picks.

### #72 — Archetype-based SoA IR storage
**Created** 2026-04-12. Replace AoS `Vec<Instruction>` with per-opcode-class SoA arrays for cache-friendly passes.
**Status:** Not shipped. The current IR (`lib.rs:65, 89, 807, 796` etc., `Instruction` enum is AoS) is woven through every pass in a 16k-line file.
**Effort:** **XL**. Realistically a multi-quarter rewrite. Touches every pass, every encoder/decoder branch, all stack-validation, all Z3 lowering.
**Strategic value:** Speculative 2–5x speedup on opt passes. Loom is not currently bottlenecked on optimizer wall-time — gale's pain was *correctness/output-size* (#afc9318), not speed.
**Risk:** Extremely high — touches the proof boundary at every pass. Would invalidate the StackSignature.v / FusedOptimization.v proofs.
**Recommendation:** **Defer to v1.0+.** Not justified by current bottlenecks.

### #73 — Composable typed transformations (Kamodo pattern)
**Created** 2026-04-12. Encode pass pre/post-conditions as Rust marker traits to forbid invalid pass orderings at compile time.
**Status:** Not shipped. Current pipeline is hardcoded at `lib.rs:6791–6829`.
**Effort:** S–M. ~200 LOC wrapper around existing passes; can be incremental.
**Strategic value:** Low without #71 — typed orderings are most useful when *searching* pass orderings, which is exactly what #71 does. As a standalone, this is documentation that could be a comment.
**Risk:** Low.
**Recommendation:** **Land only as part of #71's island infrastructure**, not standalone. Otherwise close as YAGNI.

### #74 — Batched compilation for multi-component builds
**Created** 2026-04-12. Share ISLE rules + Z3 context across N parallel component optimizations.
**Status:** Not shipped. ISLE rules are already statically linked, so the "amortize loading" benefit is overstated. Z3 contexts (`verify.rs`) are already created per-verification.
**Effort:** M.
**Strategic value:** Low until someone reports a multi-component build being slow. No current user signal.
**Risk:** Sharing a `z3::Context` across threads is a known footgun (Z3's C API is not thread-safe per context). Would require careful design.
**Recommendation:** **Defer.** Re-evaluate after #71 lands — most of its value overlaps.

### #75 — Optimize P3 async callback trampolines + stream buffer access
**Created** 2026-04-12. Branch-table conversion of event-type dispatch, stream-buffer bounds-check elision, dead event-handler DCE, backpressure constant prop.
**Status vs v0.6.0:** Not shipped. `BrTable` is in the IR (`lib.rs:796, 1424, 2733`) so the lowering target exists.
**Effort:** M (~500 LOC) but **80% overlap with #70**. The "convert chained `if (event == K)` into `br_table`" is a peephole; everything else is the same callback-inlining machinery #70 needs.
**Strategic value:** Subset of #70.
**Recommendation:** **Fold into #70 as a sub-task.** Close #75 as duplicate-of-#70 after merging the description into a tracking issue.

---

## Cross-cutting notes

- **Superseded by v0.6.0:** None outright. But several #68 items (memory-import dedup, adapter devirt, dead-function elim) are already shipped — the issue body should be updated to reflect this before further planning.
- **Dependencies:**
  - #70 depends on a thin slice of #68 (adapter-inlining infra) — do #68-Tier-1 first.
  - #75 depends on / is subsumed by #70 — close as duplicate.
  - #73 depends on #71 to be valuable.
  - #74 partially supersedes itself if #71 lands.
- **Research-shaped (not implementation-shaped):** #72 (IR redesign — needs prototype + benchmarks first), #68-Tier-4 (escape analysis / partial eval — needs proof-theory groundwork).
- **Proof-first compatibility:** #70, #71, #68-Tier-1 all preserve the existing Z3 + Rocq invariants (`loom-core/src/verify.rs`, `proofs/simplify/FusedOptimization.v`). #72 and #68-Tier-4 break it.

## Suggested v0.7.0 scope

1. #68 sub-issue: adapter inlining for scalar params + function-body dedup (M, ~3 weeks)
2. #70: async callback adapter collapse, building on (1) (M, ~3 weeks)
3. #71: island-model with 4 default configs + `--islands N` CLI flag (M, ~2 weeks)
4. Close #69 (merged PR, not an issue), #75 (duplicate of #70), defer #72/#73/#74.
