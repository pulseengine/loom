**Headline: of 11 open issues, recommend CLOSE 4, KEEP 4, DEFER 3.** Three Rocq issues (#45, #47, #50) are already substantially shipped — `proofs/` is in tree, `rust_verified/stack_signature_proofs.v` carries 23 `Qed`s with 0 `Admitted` for the compose-associativity property, and `verify_rules.rs` already parses ISLE rules and runs Z3 obligations on them (the Crocus-shaped pass #50 asks for). The big shape change vs. the v0.7.0 triage is that v0.8.0's PR-M (`specialize_adapters`) plus v1.0.2's PR-C (`directize` element-segment devirtualization) have collapsed the prerequisites for #70 — devirt + adapter specialization are no longer the blockers, full callback inlining is. Surprises: (1) #50 was never claimed as done in any release note even though the implementation has been live since v0.4.x, (2) #45/#47 graduated when nobody filed a "closes" PR, (3) #71 (island-model) remains the single highest-leverage open issue because none of the v0.7→v1.0 work touched parallel pipeline exploration.

# v1.0.3 Issue Triage and Roadmap

_Generated 2026-05-16. Decisions reflect the v1.0.2 codebase state (commit 0fd91fa)._

## Headline

Of 11 open issues, I recommend **4 CLOSE**, **4 KEEP**, **3 DEFER**. The three older Rocq-verification issues (#45, #47, #50) have been silently fulfilled by code that landed across v0.4–v1.0 and were never linked back to their issues; close them with citations. The Component-Model cluster (#68/#70/#74/#75) has accreted significant fused-side infrastructure (`fused_optimizer.rs`, `specialize_adapters`, `directize`), but the actual cross-component inlining + async-callback folding work proposed in #70 is genuinely not done — keep #70 and a Tier-1 slice of #68. #71 (island-model parallel optimization) is still the cheapest big UX/quality win on the table.

## Verdicts

| # | Title | Verdict | Rationale (one line) |
|---|---|---|---|
| #45 | Add Rocq formal verification support using rules_rocq_rust | CLOSE | Foundation shipped: `proofs/` tree exists with `BUILD.bazel`, `rust_verified/`, `simplify/`, `stack/`, `codec/`, `isle/`; rules_rocq_rust integration is live. |
| #47 | Prove StackSignature::compose associativity in Rocq | CLOSE | `proofs/rust_verified/stack_signature_proofs.v` has 23 `Qed`s and 0 `Admitted`s; associativity is among them. |
| #48 | Prove parser/encoder round-trip identity | KEEP | `proofs/codec/Roundtrip.v` exists but is one `Admitted` theorem against axiomatized parse/encode; real proof unwritten. |
| #50 | ISLE rule verification (Crocus-style) | CLOSE | `loom-core/src/verify_rules.rs:1640` ships `parse_isle_rules` + 45 `verify_*` Z3 obligation drivers — Crocus-shaped verification is the day-1 design of that module. |
| #68 | Cross-component optimization passes for meld-fused modules | KEEP | Roughly 50% of the issue body is shipped (memory-import dedup, adapter devirt, dead-fn elim, type dedup, `specialize_adapters`, `directize`); Tier-1.1 scalar adapter inlining + Tier-2.2 function-body dedup are not. |
| #70 | Optimize meld-fused async callback adapter patterns | KEEP | Inlining + `directize` are in tree, but the full 6-pass collapse (inline → devirt → inline callback → kill loop → forward shim → kill start_task) is not wired end-to-end and has no test fixture. |
| #71 | Island-model parallel optimization | KEEP | Still no parallel pipeline; no `rayon`/thread-spawning in `loom-core`. Highest-leverage open item by ROI. |
| #72 | Archetype-based SoA IR storage | DEFER | XL rewrite of a 39k-LOC tree for a speculative 2-5× pass-time speedup; LOOM is not pass-time-bound. |
| #73 | Composable typed transformations (Kamodo pattern) | DEFER | Standalone value low; pairs with #71. Land as part of #71 islands or close. |
| #74 | Batched compilation for multi-component builds | DEFER | No user signal of multi-component build slowness; overlaps with #71 on the parallelism axis. |
| #75 | Optimize P3 async callback trampolines + stream buffer access | CLOSE | Duplicate-of-#70; the v0.7.0 triage flagged this and the recommendation still stands. |

## CLOSE recommendations

**#45 — Add Rocq formal verification support using rules_rocq_rust** — Resolved: `proofs/BUILD.bazel` plus per-domain subdirs (`rust_verified/`, `simplify/`, `stack/`, `codec/`, `isle/`) are in tree, with both Rust-translated proofs (`proofs/rust_verified/`) and hand-authored optimization-correctness proofs (`proofs/simplify/FusedOptimization.v`, etc.). `TEST-ROCQ-PROOFS` in `safety/requirements/verification.yaml` runs the suite as part of the verification gate.

**#47 — Prove StackSignature::compose associativity in Rocq** — Resolved: `proofs/rust_verified/stack_signature_proofs.v` contains the full associativity theorem along with empty-identity, decidability, subtyping reflexivity/transitivity, and stack-effect additivity. The file has 23 `Qed`s and 0 `Admitted`s.

**#50 — ISLE rule verification (Crocus-style)** — Resolved: `loom-core/src/verify_rules.rs` is exactly this — `parse_isle_rules` (line 1640) lifts the rule text, `parse_single_rule` extracts LHS/RHS + operation + bitwidth, and the file ships 45 `verify_*` functions that lower each rule shape to a Z3 obligation and discharge it. The build-time-vs-test-time distinction the issue calls out is satisfied: the verification suite is wired into CI via `TEST-Z3-VERIFICATION-CORE` in `safety/requirements/verification.yaml`.

**#75 — Optimize P3 async callback trampolines + stream buffer access patterns** — Duplicate of #70: every concrete optimization listed (event-dispatch branch-table conversion, bounds-check elision, dead handler DCE, backpressure constant prop) is either a sub-task of the #70 callback-inlining pipeline or trivially follows once #70's inlining stage runs. Folding the description into #70 as sub-tasks is the right move.

## Roadmap entries (KEEP issues)

### #68 — Cross-component optimization passes (Tier-1.1 scalar adapter inlining + Tier-2.2 function-body dedup)

**Why now.** v0.8.0's `specialize_adapters` and v1.0.2's `directize` both prove that the fused-module pass surface is correctness-tractable when scoped narrowly. Tier-1.1 (scalar adapter inlining for cross-memory copies that lower to register passing) is the next adapter-shaped delta on the same pass surface, and it is the prerequisite for #70's "inline `[async-lift]` entry" step. Tier-2.2 (function-body dedup across components that linked the same wit-bindgen runtime / wasi-libc) is a pure code-size win and complements `eliminate_dead_functions` already in `fused_optimizer.rs`. Both are mechanical extensions of the existing module — neither needs a new analysis framework, and both fit the `verify_or_revert` safety model.

**How.** New pass `inline_scalar_adapters` in `loom-core/src/fused_optimizer.rs` (slot after `devirtualize_adapters`, before `eliminate_dead_functions`). Reuse the `AdapterInfo` detection from `detect_adapters` (line 1123), filter to adapters whose param + result lists are all scalar (i32/i64/f32/f64) and whose body is exactly load(callee-mem) → load(caller-mem) → call → store(caller-mem) → store(callee-mem). Replace those bodies with a direct stack-passing call; the encoder pass already handles cross-memory `Call` with no copies. For #2.2, hash function bodies (instruction-sequence + signature) using a SipHash-keyed walk, group equal hashes, then redirect calls in `rewrite_calls` (line 1226). Tests: extend `loom-core/src/fused_optimizer.rs` test module with a 2-component fixture exercising both. Gate: `cargo test --release fused_optimizer::tests`, plus the corpus harness must show no regression on calc_component / simple_component.

**Effort.** **M**, ~700–900 LOC across `fused_optimizer.rs` + 2 tests. **Dependencies.** None hard; nice-to-have prerequisite for #70's full pipeline.

### #48 — Prove parser/encoder round-trip identity

**Why now.** The file `proofs/codec/Roundtrip.v` is one of three remaining Rocq stubs (the others are `proofs/stack/StackSignature.v` and `proofs/isle/TermBijection.v` — note these are the *placeholder* duplicates, the *real* proofs live in `proofs/rust_verified/` and `proofs/stack/`). Closing the round-trip proof would be the first end-to-end proof that LOOM's textual Wasm encoder/decoder is total and injective. It complements REQ-13 (valid wasm output) and REQ-14 (deterministic output) in `safety/requirements/`. Doing this in v1.0.3+ would also let us retire the `Roundtrip.v` placeholder and prevent the next round of triage from re-flagging it.

**How.** The pragmatic path is not to translate the full 18k-LOC `loom-core/src/lib.rs` parser/encoder, but to scope to a verifiable subset of `Module` (functions, types, instructions in the ISLE-tracked subset) and prove round-trip there. Concretely: define `Inductive ScopedModule` in `proofs/codec/Roundtrip.v` covering exactly the instructions whose conversion to/from terms is already proven in `proofs/rust_verified/isle_conversion_proofs.v` (line 23, currently `Admitted`); then build a Rocq model of the LEB128 + section encoder for that subset. Discharge with `Coq.Strings.Byte` arithmetic. Gate: `bazel test //proofs:codec_test`.

**Effort.** **L**, ~1500 LOC of Rocq plus ~200 LOC of subset-extraction tooling on the Rust side. **Dependencies.** Should land *after* `proofs/rust_verified/isle_conversion_proofs.v` is closed (currently the bijection there is also `Admitted`) — otherwise the round-trip proof depends on an unproven lemma. Treat as a 2-PR sequence: first remove `Admitted` from the term-bijection proofs, then write Roundtrip.

### #70 — Optimize meld-fused async callback adapter patterns

**Why now.** This is the single highest-impact concrete codegen win we have not yet captured. The v1.0.2 measurement note ("New infrastructure is correct and tested; byte wins compound once the corpus grows") is honest about the bottleneck: we have the inlining substrate (`inline_functions` at `lib.rs:12946`), we have devirt (`directize` at `lib.rs:7754`), we have adapter specialization (`specialize_adapters` at `component_optimizer.rs:745`), but we have not yet composed them on the P3 async pattern. Every meld-fused async component pays the ~10-instruction overhead per sync-completing call today. With the directize verifier-bypass model accepted in v1.0.2, the table-driven dispatch part of this pass is no longer blocked on Z3 teaching.

**How.** Six-pass chain following the issue's structure, all in `loom-core/src/component_optimizer.rs` as a new `optimize_async_callback_adapters` pass slotted between `specialize_adapters` (Phase 3) and the generic optimizer entry. (1) Detect the meld P3 adapter shape: function imports `[async-lift]`, body calls it with constant args, then unpacks an i32 EXIT code. (2) Inline `[async-lift]` via the existing inliner (no new code needed once the function is marked inlinable). (3) After inline, the body contains a `call_indirect` through a constant element-segment slot — `directize` already folds this. (4) Constant-propagate the EXIT discriminant via the existing `constant_folding` pass; mark the WAIT/YIELD branches as `Unreachable`; rerun `eliminate_dead_code`. (5) Forward `task.return` shim: implement as a peephole in `simplify_locals` that pattern-matches `global.set $shim; global.get $shim` and erases both (only fires when the global has a single writer in the function, dominated check via the existing locals tracker). (6) Dead-store eliminate the `start_task` waitable-set init — already covered by `dead-stores` post-DCE. Test fixture: hand-build a 30-instruction meld-shaped P3 async adapter in `loom-core/src/lib.rs` test module; gate is byte-count after pass ≤ 8 instructions. Z3 verification: each constituent pass is already individually Z3-gated; the composite needs no new proof.

**Effort.** **M–L**, ~600 LOC of new code (mostly pattern detection + the global forwarder peephole) plus a meld-shaped test fixture. **Dependencies.** Easier after #68 Tier-1.1 lands (the adapter-inlining infra removes one of the steps), but not blocked by it — the existing `inline_functions` is sufficient for step 2.

### #71 — Island-model parallel optimization

**Why now.** This is the cheapest big quality win left. The v0.6.0→v0.7.0 gale CSE-cost regression (commit `afc9318`) is exactly the failure mode islands would have surfaced automatically — different orderings would have produced different sizes, the smaller verified result would have been picked, and the regression would never have shipped. The infrastructure cost is tiny (~400 LOC, one rayon dep), risk is zero on the proof axis (each island still passes the existing Z3 + stack validation gates independently), and the UX benefit on multi-core dev machines is immediate. It also serves as the test bed for #73 (typed pass composition) without forcing #73 to ship as a standalone.

**How.** New module `loom-core/src/islands.rs`, exposing `optimize_module_islands(module, configs: &[IslandConfig]) -> Result<Module>`. Each `IslandConfig` is `{ pass_order: Vec<PassId>, inline_size_threshold: usize, cse_cost_gate: bool, ... }`; ship 4 default configs (the current pipeline as baseline, plus inline-late / cse-early / aggressive-inline). Use `rayon::scope` to run them concurrently; each island clones the input module. Selection: filter to islands whose result passes both Z3 verification and stack validation (the existing gates at `lib.rs:6843–6873`), pick `min_by_key(|r| encoded_size(r))`. CLI: `--islands N` (default 1 to preserve current behavior), `--islands-config <file>` for advanced users. Tests: fuzz harness in `loom-testing/` that runs all 4 islands on the test corpus and asserts the winner is no larger than any single island; gate `cargo test --release islands`.

**Effort.** **M**, ~400–500 LOC + `rayon = "1"` dep + CLI plumbing. **Dependencies.** None. #73 can be layered on top as a follow-up if the trait-based config proves valuable; if not, #73 closes as YAGNI.

## Deferred (with reason)

**#72 — Archetype-based SoA IR storage.** XL rewrite of an 18k-LOC `Instruction`-enum-centric IR for a speculative 2–5× pass-time speedup. LOOM is not pass-time-bound today (`measure_corpus.sh` runs end-to-end in under a minute on the full corpus); user-visible pain has been correctness and output-size, not wall-time. Touching every pass would also invalidate the `proofs/rust_verified/stack_signature_proofs.v` lemmas that depend on the current shape. Re-evaluate if a profile ever shows pass-iteration cost dominating; until then, the cost-benefit doesn't pencil.

**#73 — Composable typed transformations (Kamodo pattern).** Cleanly typed pass pre/post-conditions are only operationally valuable when something *uses* the types to enumerate valid orderings — that something is #71. As a standalone refactor it's documentation-via-types that the existing `Phase N:` comments at `lib.rs:6806–6851` already provide. Recommend folding into #71's `IslandConfig` design (the per-island pass list naturally wants typed compatibility checking) or closing as YAGNI.

**#74 — Batched compilation for multi-component builds.** No user has reported multi-component build slowness, ISLE rules are already statically linked (so "amortize rule loading" is a non-claim), and the cross-component-sharing benefit overlaps almost entirely with #71's parallel island infrastructure. The known Z3 footgun (`z3::Context` is not thread-safe across the C API in the way the issue body assumes) makes this a higher-risk version of the same parallel-execution story #71 covers. Defer until either #71 ships and a user complains about per-component-startup overhead, or someone reports a slow multi-component build.

## Summary tally

- **CLOSE**: 4 (#45, #47, #50, #75)
- **KEEP**: 4 (#48, #68, #70, #71)
- **DEFER**: 3 (#72, #73, #74)

Suggested v1.0.3..v1.5 sequencing:

1. v1.0.3 (next): close #45/#47/#50/#75 with comments citing the artifacts above. Ship #71 (island-model) — lowest correctness risk, biggest UX delta, paves the way for safe pass-order experimentation.
2. v1.0.4–v1.1: ship #68 Tier-1.1 + Tier-2.2 (`inline_scalar_adapters`, function-body dedup) as the prerequisite slice for #70.
3. v1.1–v1.2: ship #70's six-pass async callback collapse on top of (1) and (2). At this point the strategic-moat story (component-model adapter wins) is end-to-end real.
4. v1.2–v1.5: take #48 if there is appetite for closing the last Rocq `Admitted` lemmas. Otherwise drop to v2.0 backlog.
