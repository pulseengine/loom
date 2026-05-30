# Changelog

All notable changes to LOOM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.1.2] - 2026-05-30

**Bug-fix release: the gale-ffi i64 `SortDiffers` crash (#145).** Closes
the panic + 21 MB-stderr flood that made `loom optimize` unusable on
i64-heavy modules (gale-ffi / compiler_builtins), and width-corrects the
Z3 translation validator's i64 modeling.

### Known limitation (tracked: #147)

Verifying i64 inlining *for real* requires a Z3 bitvector solve per
function, which is **slow in aggregate** on large i64-heavy modules — so
`loom optimize` on such a module is bounded-but-slow (each query capped
by `LOOM_Z3_TIMEOUT_MS`, default 5000 ms → conservative revert on
timeout), and the i64 inline *unit tests* are `#[ignore]`'d (they hang
in SMT-formula construction, which the per-query timeout doesn't bound).
**What this release guarantees:** no crash, no stderr flood, sound output
(width-correct verification; conservative revert when a proof is slow or
unprovable). **What it does not yet guarantee:** fast, fully-verified i64
inlining on large modules. That — a cheaper i64 inline-equivalence check
and re-enabling the ignored tests — is tracked in #147. Tune with
`LOOM_Z3_TIMEOUT_MS` (lower = faster, more reverts) and
`LOOM_Z3_MAX_INSTRUCTIONS` for large i64 workloads.

### Fixed

- **#147: Z3 per-query timeout in the translation validator.** The #145
  width fix made the verifier run a real SMT solve on i64 functions
  (previously it panicked during eq-construction and fast-reverted);
  Z3 can hit a slow-solve cliff on i64 bitvector formulas, which made
  the verifier grind. A per-`check()` timeout (`LOOM_Z3_TIMEOUT_MS`,
  default 5000 ms; `0` disables) now bounds every query: a timed-out
  solve returns `Unknown`, which the verifier already treats as
  "cannot prove" → conservative revert (sound — the original function
  is kept). This keeps both the test suite and real i64-heavy modules
  (gale-ffi) fast. The four no-panic `test_inline_i64_*` regression
  tests are bounded by this and run again; the one test asserting
  *inlining* stays `#[ignore]` pending timeout tuning (a revert would
  flake the inlining assertion).

- **#145: i64 `SortDiffers` in `inline_functions` verifier (gale-ffi /
  compiler_builtins).** On i64-heavy modules the Z3 translation
  validator panicked with `SortDiffers { BitVec 64 vs 32 }` (and
  `unwrap()`-on-`None` through other z3 binding sites), reverting
  essentially every function so the inliner was a no-op, and emitting
  21 MB+ of stderr from per-function caught-panic backtraces. Root
  cause: the k-induction symbolic executor
  (`encode_loop_body_for_kinduction`, the loop path these modules hit)
  hardcoded `BitVec32` for uninitialized locals/stack/globals and applied
  **unmatched** binops, so a real i64 value meeting a 32-bit-modeled slot
  tripped a width mismatch deep in z3. Fixes:
  - The dormant `match_bv_widths` helper (added but never wired in by the
    #98/#99 fix) is now applied at **every binop and comparison operand
    site** in both symbolic executors. Sound because a valid wasm binop's
    operands are equal-width by construction — matching only repairs a
    model artifact and never changes a modeled value.
  - The **equivalence checks** (`orig.eq(opt)`, k-induction state
    compares) are *not* width-matched — that could approve a
    non-equivalent transform. Instead they bail to "cannot prove" (a
    conservative revert) on any width mismatch. This soundness boundary —
    match at binops, never at the equivalence — is documented at each
    site.
  - z3-internal panics (always caught and reverted) no longer print: a
    `Once`-installed, islands-race-safe panic filter suppresses
    z3-origin backtraces, and per-function revert logging moves behind
    `LOOM_VERBOSE_REVERTS` (the count still surfaces in `--stats`). This
    removes the 21 MB stderr flood; a reverted run now reports one
    aggregate count.
  - New `test_inline_i64_loop_kinduction_no_panic` regression test
    covering the i64 **loop** path (the prior #98 tests are loopless and
    never exercised k-induction).

## [1.1.1] - 2026-05-22

**Housekeeping + an ægraph commutativity bug fix.** A patch release
clearing the v1.1.0 Track-3 carry-forward and fixing a real
operand-ordering bug that blocked commutative identity folds.

### Fixed

- **ægraph commutativity normalization.** `EGraph::canonicalize_commutative`
  ordered operands purely by union-find class id, so when a constant
  operand happened to be inserted (and numbered) before its variable
  sibling, `Add(0, x)` stayed constant-left and the `(wild, Const)`
  identity rules could not match it. The sort key is now
  `(is_constant, uf-root id)` — constants always move to the right,
  matching every identity rule's LHS shape. The previously `#[ignore]`'d
  `test_commutativity_zero_plus_x_folds` is now a passing positive
  witness; `test_commutativity_idempotent` confirms the new order is
  still a fixpoint.

### Housekeeping (v1.1.0 Track D)

- `Instruction` and `BlockType` now derive `Eq + Hash` (previously
  `PartialEq` only). Lets downstream passes key hash sets/maps on
  instructions structurally instead of via `Debug`-formatted strings.
- `AdapterInfo` and its fields lifted from module-private to
  `pub(crate)` for cross-module reuse.
- `optimize_module` no longer discards `FusedOptimizationStats` or
  silently swallows fused-optimization outcomes: it now logs a
  one-line summary of what the fused passes did on success (positive
  signal they ran) and keeps the non-fatal warning on failure.

### Tests

379 loom-core lib tests pass (was 378 + 1 ignored; the commutativity
test is now un-ignored and green).

### Deferred

- **Track E** — real meld-fused multi-component fixture. `meld` v0.9.0
  is now installed and working, but a fixture that exercises the
  cross-memory adapter passes needs a memory-sharing component pair
  that does not exist ready-made in either repo.
- **Rocq CI** — `Rocq Formal Proofs` stays red pending upstream
  `rules_rocq_rust` PR #34 (`rules_rust` toolchain migration, still
  draft). When it merges, bump the `MODULE.bazel` pin.

## [1.1.0] - 2026-05-20

**ægraph substrate goes production + first mechanized roundtrip
proof.** A minor-version bump: the v1.0.4 ægraph substrate is now a
default-on pipeline pass with cost-driven extraction and a widened
rule set, and the parser/encoder roundtrip proof (#48) gains a real
Rocq scaffold. Byte-neutral on the current corpus — this is an
infrastructure and correctness release, not a size-win release.

### Optimization

- **Track B (#134, re-applied in this release commit): cost-driven
  ægraph extraction.** `egraph::extract()` now finds the union-find
  root of the requested class, scans every class id whose `find()`
  resolves to that root, and emits the representative with the
  lowest *total* encoded-byte cost. New `Op::encoded_byte_cost()`
  returns 1 for opcodes and `1 + LEB128(immediate)` for
  `const` / `local.get`, mirroring wasm-encoder exactly. Subtree
  cost is a HashMap-memoized DP keyed on UF root (the acyclic
  invariant — child id < parent id — is the termination guarantee).
  This closes the v1.0.5 Track 1 substrate gap: the manual UF-root
  scan in `egraph_optimize_body` is deleted, and the call site is
  now just `egraph.extract(root_class)`.

  Process note: PR #134 merged but its `egraph.rs` / `lib.rs` diff
  was silently clobbered when PR #137's rebase resolved conflicts by
  whole-file copy from a pre-#134 branch. The content is re-applied
  in this release commit; 25 egraph tests green.

- **Track C (#137): ægraph rule-set widening.** 11 new `Op`
  variants for i64 (`Add`/`Sub`/`Mul`/`And`/`Or`/`Xor`/`Shl`/
  `ShrS`/`ShrU`/`Eq`/`Eqz`) and 8 new identity rules — i64
  `+0` / `|0` / `&-1` / `*1` plus three shift-by-zero folds. New
  `Op::is_commutative()` + `EGraph::canonicalize_commutative()`
  normalize operand order for the commutative i32/i64 ops so each
  identity rule only needs the `(wild, Const)` form. One test
  (`test_commutativity_zero_plus_x_folds`) is `#[ignore]`'d pending
  insertion-time normalization — a v1.1.1 follow-up.

- **Track F: ægraph pass is default-on.** The pass already ran by
  default mechanically (`should_run` is permissive without
  `--passes`); the stale "opt-in via --passes egraph" comment is
  corrected. Default-on is revert-safe by construction:
  `egraph_optimize_body` splices extraction back only when it is
  strictly shorter than the original tree, so a function is either
  improved or left byte-identical — never regressed.

### Proofs

- **Track A (#135): Path A for #48 — parser/encoder roundtrip
  identity.** Total `Admitted.` count in `proofs/` drops 4 → 2.
  `TermBijection.v` is rewritten from a 42-line placeholder into a
  272-line self-contained file; both `term_conversion_bijection`
  and `term_conversion_bijection_rev` close with `Qed`.
  `StackSignature.v` adds `combined_kind` + `combined_kind_assoc` +
  `compose_kind` + `compose_assoc_kind`, all `Qed` — the kind
  component of composition associativity is closed. `Roundtrip.v`
  lands the `ScopedModule` + LEB128 + section-codec scaffold. The
  two remaining `Admitted.` are the `leb128_roundtrip` general-nat
  induction step and the `StackSignature` dataflow component, both
  documented with proof sketches.

### Measurement

- New `docs/measurements/v1.1.0-corpus-baseline.md`. LOOM produces
  no regression on any corpus fixture (every LOOM Δ% ≤ 0). Per-file
  deltas are unchanged from v1.0.5 — the ægraph pass is byte-neutral
  on the current corpus because these fixtures lack the foldable
  identity patterns the rule set targets; the substrate is wired and
  will produce wins once such patterns appear.
- `measure_corpus.sh` `pct_delta` no longer coerces sentinel
  strings (`error` / `invalid` / `timeout`) to `0`, which had
  fabricated a `-100%` "win" on a failed or timed-out run. Such rows
  now correctly read `n/a`.

### Deferred to v1.1.1

- **Track D — Track-3 housekeeping** (`Instruction` `Eq`/`Hash`,
  `pub(crate)` `AdapterInfo`, surfaced `FusedOptimizationStats`,
  no-silent-swallow in `optimize_fused_module`). Touches every
  fused-optimizer call site; held back to keep the v1.1.0 review
  surface bounded.
- **Track E — real meld-fused multi-component fixture.** Blocked on
  a `meld`-binary permission wall and the absence of a component
  pair with a shared cross-memory shape. Shipped as a documented
  placeholder (`tests/corpus/MELD_FUSED_README.md`); the harness
  carries a `meld_fused` workload slot that stays `n/a` until the
  fixture lands.

## [1.0.5] - 2026-05-19

**Four-track v1.0.4 follow-through.** Each v1.0.4 infrastructure
piece grew a real consumer this release. All four tracks shipped
clean.

### Optimization

- **Track 1 (me): ægraph pipeline integration.** New
  `pub fn egraph_optimize` consumer for the v1.0.4 Track C
  substrate. Walks each function's straight-line maximal
  `(0→1)`-net-stack-effect expression trees, runs them through the
  egraph with `identity_rules()`, picks the smallest representative
  via union-find scan. Opt-in via `--passes egraph`. 4 new tests.
  Works around a substrate gap (extract picks the originally-stored
  node, ignoring union-find merges) by scanning all class ids in
  the same UF root for the shortest extraction; v1.0.6 should move
  this into `egraph::extract` as proper cost-driven extraction.

- **Track 2 (agent, PR #131): #70 six-pass chain composition.**
  Composes the v1.0.4 async-callback adapter detector with `inline_functions`
  + `directize` + `constant_folding` + `eliminate_dead_code` + new
  `forward_global_shim` peephole + `eliminate_dead_stores`. Each
  pass uses its own `verify_or_revert` Z3 gate. ~600 LOC across
  `component_optimizer.rs` and `lib.rs`. 8 new tests. Hand-built P3
  adapter fixture shrinks strictly under the composed chain.

  New peephole `forward_global_shim`: recognizes
  `GlobalSet(idx); GlobalGet(idx)` pairs and removes both when the
  global has exactly one writer in the function.

- **Track 3 (agent, PR #132): #68 Tier-1.1 + Tier-2.2.** Two new
  passes in `fused_optimizer.rs`:
  - `inline_scalar_adapters` slots between `devirtualize_adapters`
    and `eliminate_dead_functions`. Filters `AdapterInfo` to adapters
    with all-scalar params + results and canonical cross-memory-copy
    body shape; replaces body with a direct stack-passing call.
  - `dedupe_function_bodies` groups functions by `(signature, body)`
    hash and redirects calls to the lowest-index representative.
    Conservative: skips exported / imported / start-function bodies.
  ~510 LOC. 6 new tests. Byte-neutral on current corpus (no real
  meld-fused multi-component fixture yet); infrastructure for #68's
  cross-component story.

### Documentation

- **Track 4 (me): #48 prep survey.** New
  `docs/research/v1.0.5/rocq-roundtrip-prep.md` (~700 words).
  Surveys the proofs/ tree's `Admitted` state. Recommends **Path A**
  for v1.1.0 (close #48 properly, ~1400 LOC Rocq + ~150 LOC Rust
  subset-extraction tooling) or **Path B** for v1.0.5 cleanup
  (delete the 3 placeholder Admitteds, signal-noise reduction only).
  Both paths documented with concrete file-by-file steps.

### Tests

+18 new tests (4 Track 1 + 8 Track 2 + 6 Track 3). All 400+
loom-core lib tests pass.

### Strategic moat unchanged

| Workload | LOOM Δ% | wasm-opt Δ% |
|---|---:|---:|
| simple_component | **−18.8%** | wasm-opt errors |
| calc_component | **−11.3%** | wasm-opt errors |
| gale | −4.9% file / −2.0% code | −0.8% file / −2.0% code |

### Honest measurement note

Per-fixture byte deltas re-measured by Track 3 agent: gale −4.9%,
httparse −2.1%, json_lite −3.8%, state_machine −5.9%, calc_component
−11.3%. All deltas come from the existing pipeline; the new passes
are byte-neutral on these non-fused fixtures. They will produce wins
once a real meld-fused multi-component fixture lands (PR-Q extension,
deferred).

### Suspicious observations from Track 3

The Track 3 agent flagged 4 pre-existing issues worth recording (no
fixes attempted):

1. `Instruction` derives `PartialEq` but not `Eq` / `Hash`. Forces
   the dedup pass to hash a Debug-formatted representation rather
   than a structural hash. Adding `Eq + Hash` would let downstream
   passes use proper hash sets keyed on instructions directly.
2. `AdapterInfo` is module-private but `inline_scalar_adapters`
   needs to share the type with the caller. Both should be
   `pub(crate)`-lifted together for future cross-module use.
3. Fused-side stats (`FusedOptimizationStats`) aren't surfaced in
   the CLI `--stats` output.
4. `optimize_fused_module` is invoked unconditionally from
   `optimize_module` and silently swallows errors with `eprintln!`.
   Intentional best-effort model but combined with #3 means no
   positive signal that fused-side passes ran on a given input.

### Deferred to v1.0.6+

- Cost-driven ægraph extraction (move the UF-root scan into
  `egraph::extract()`).
- ægraph rule set widening (i64 ops, commutativity normalization).
- Default-on ægraph pass once corpus measurements show wins.
- Path A for #48 (~1400 LOC Rocq, v1.1.0).
- Real meld-fused multi-component corpus fixture (will surface
  Track 3 byte wins).
- Suspicious observations 1-4 above.

## [1.0.4] - 2026-05-18

**Four-track parallel release with full success.** All four tracks
shipped real work this time (vs v1.0.3 where Track 3 died). Two
tracks by me (#70 async-adapter, verifier table-resolver retry),
two by agents (ægraph rewrite engine, island-model parallel
optimization). Plus a new-issues triage sweep that found zero new
issues since v1.0.3 (the bar to close).

### Optimization

- **#70 / Track A — async-callback adapter pass** (PR #125). New
  Phase 4 in the component pipeline. Detects the meld P3 async-callback
  adapter shape and folds the discriminant test + slow-path branch
  when EXIT_OK is statically true. Three safety guards: no `Unknown`
  in module, local-read-count = 1, `I32Const == 0`. Per-fold
  encode + wasm-tools validate with revert on mismatch. 4 new tests.
  ~460 LOC. First piece of the six-pass chain from the v1.0.3 roadmap
  (detect + fold the discriminant pattern). Steps 2-6 — inline,
  directize, constant-fold, forward task.return shim, DCE start_task
  init — are infrastructure that already exists in the pipeline.
  Composing them on the post-detection IR is v1.0.5+ work.

- **#71 / Track D — island-model parallel optimization** (PR #128).
  New `loom-core/src/islands.rs` (~580 LOC) + CLI `--islands N` flag.
  Runs N IslandConfigs concurrently via rayon. Each independently
  passes the existing Z3 + stack validation gates. Picks
  `min_by_key(encoded_size)` with deterministic name lex tie-break.
  Parallelism confirmed by timing: N=4 takes 1.4× wall time for 4×
  serial work. On gale all 4 default configs converge to 1846 bytes
  (a fixed-point after v0.7.0+ pipeline hardening — the safety net
  is in place for future passes that might re-introduce
  ordering-dependent regressions). 6 new tests.

### Verifier

- **Track B — element-section table resolver in Z3** (PR #126).
  Drops directize's Z3 bypass from v1.0.2. The verifier now resolves
  `i32.const N; call_indirect (type T)` to the same
  `pure_call_<F>(args)` Z3 expression PR-K3 uses for direct
  `call F` — so `call_indirect → call F` proves equivalent under
  congruence closure. Changes:
  - `VerificationSignatureContext` gains a `table_resolver: HashMap<(u32, u32), u32>`
    field, populated via `crate::optimize::build_table_resolver`.
  - New `resolve_indirect_call(table_idx, slot, expected_type)` helper.
  - The CallIndirect encoder checks `BV::as_u64()` for concreteness;
    if concrete AND resolved AND signature matches, encodes the result
    as `pure_call_<F>(args)` and skips the global/memory havoc.
  - `directize()` reconstructs `TranslationValidator::new_with_context`
    with the populated sig context. Z3 verification is BACK; the
    structural guards remain as defense-in-depth.

  All 3 existing directize tests pass with Z3 ACTIVE.

### ægraph

- **Track C — rewrite engine + 3 identity rules** (PR #127). Builds
  on the v1.0.3 substrate. Adds:
  - `EGraph::union(a, b)` + `rebuild()` for congruence-closure
    propagation. Fixpoint bounded by equivalence-class count.
  - `Pattern` enum + `Rule` struct describing the LHS pattern and
    the RHS instantiation. Wildcards via `Pattern::Wild`.
  - `apply_rules` + `saturate_with_rules` driving the iterative
    pattern-match-and-union.
  - Three hand-proven rules: `x + 0 == x`, `x * 1 == x`,
    `x & -1 == x`.
  - 7 new tests (14 total egraph tests, all pass).

  Still library-only — no pipeline integration. v1.0.5 next: cost-model
  extraction, more rules (commutativity normalization, i64 set, power-of-2
  shifts), then pipeline gating behind a CLI flag once corpus measurements
  confirm wins.

### Issue triage

- **Subagent sweep**: zero new GitHub issues since the v1.0.3 triage.
  Open set is still `{#48, #68, #70, #71, #72, #73, #74}`. The only
  post-cutoff filing (#98 Z3 SortDiffers panic on i64-heavy wasm) was
  already shipped fixed in v0.6.0 via PRs #99/#100.

### Tests

+20 new tests across the four tracks. All loom-core lib tests pass
(380+ total).

### Strategic moat unchanged

| Workload | LOOM Δ% | wasm-opt Δ% |
|---|---:|---:|
| simple_component | **−18.8%** | wasm-opt errors |
| calc_component | **−11.3%** | wasm-opt errors |
| gale | −4.9% file / −2.0% code | −0.8% file / −2.0% code |

### Honest measurement note

Code-section bytes UNCHANGED on the current corpus. The four tracks
ship infrastructure (async-adapter, verifier teaching, ægraph engine,
island parallelism) — none of which surface byte deltas on the existing
fixtures. v1.0.5+ will ship the consumers: ægraph pipeline integration,
the async-adapter six-pass chain, and corpus extensions targeting the
specific shapes #70/#71 are designed for.

### Still deferred to v1.0.5+

- ægraph pipeline integration + cost-driven extraction + more rules.
- The full six-pass chain composition from #70 (inline, directize,
  const-fold, forward, DCE on the post-detection IR).
- KEEP issues from the roadmap: #48, #68 (Tier-1.1 + 2.2).
- 9 pre-existing rivet schema-fit errors (SG decomposition, CP link
  types).

## [1.0.3] - 2026-05-17

**Five parallel tracks release.** Four agent worktrees + one
direct-work track addressing the v1.0.2 deferred-list. Three
tracks shipped real work; one track's agent died and was deferred
to v1.0.4.

### Tracks landed

- **Track 1 (PR-Q): real corpus fixtures.** Third attempt after
  two prior agent stalls. This attempt used NO external crate
  dependencies — locally-authored 749 LOC of Rust source across
  three fixtures (httparse / json_lite / state_machine), each
  built to wasm32-unknown-unknown and committed as 4.7 KB / 3.5
  KB / 1.7 KB `.wasm` files under `tests/corpus/`. Each validates
  via `wasm-tools validate`. Three of the previously-`n/a` rows
  in the measurement harness now have real numbers.

- **Track 2 (PR-egraph): ægraph MVP.** Acyclic Cranelift-style
  e-graph substrate at `loom-core/src/egraph.rs` (~432 LOC + 7
  tests). Hash-consing, acyclic invariant, basic extraction. No
  rewrite engine yet — the substrate is in place for future PRs
  to add union-find driven rewrites with per-rule Z3 proofs.

- **Track 4 (safety-goals): close all 4 remaining lifecycle gaps.**
  Added 4 safety-context artifacts (SC-CTXT-2..5) and 3 new
  safety-solution artifacts (SOL-6..8). `rivet validate` no
  longer reports a "Lifecycle coverage gaps" section. The 9
  remaining errors are pre-existing schema-fit issues (SG
  decomposition + CP link types) unrelated to this PR.

- **Track 5 (issue triage + roadmap): 11 open issues classified.**
  4 CLOSE (already shipped: #45 Rocq foundation, #47 compose
  associativity proof, #50 ISLE rule verification; duplicate:
  #75); 4 KEEP with roadmap entries (#48, #68, #70, #71); 3
  DEFER (#72, #73, #74). Roadmap at
  `docs/research/v1.0.3/issue-roadmap.md` (~2200 words).

### Track deferred to v1.0.4

- **Track 3 (verifier table-resolver teaching).** Agent died with
  no work product. The directize Z3 bypass stays in place for
  v1.0.3 (structural guards still imply soundness).

### Issues closed via this release

- #45 (Rocq foundation) — proofs/ tree complete; TEST-ROCQ-PROOFS
  runs them in CI.
- #47 (StackSignature::compose associativity) — 23 Qed's, 0
  Admitted's in proofs/rust_verified/stack_signature_proofs.v.
- #50 (Crocus-style ISLE rule verification) — loom-core/src/verify_rules.rs
  has been Crocus-shaped since day 1.
- #75 (P3 async callback trampolines) — duplicate of #70.

### Lifecycle coverage progress across the v1.x arc

| Release | Gaps |
|---|---:|
| v1.0.0 | 12 |
| v1.0.1 | 9 |
| v1.0.2 | 4 |
| **v1.0.3** | **0 gaps** (9 pre-existing schema errors remain) |

### Tests

ægraph MVP adds 7 tests. Track 1 adds 3 corpus fixtures the harness
can measure. All 335+ existing loom-core lib tests continue to pass.

### Strategic moat unchanged

| Workload | LOOM Δ% | wasm-opt Δ% |
|---|---:|---:|
| simple_component | **−18.8%** | wasm-opt errors |
| calc_component | **−11.3%** | wasm-opt errors |
| gale | −4.9% file / −2.0% code | −0.8% file / −2.0% code |

### Still deferred to v1.0.4+

- Track 3: verifier table-resolver teaching (lets directize use Z3).
- ægraph rewrite engine (Track 2 shipped only the substrate).
- KEEP issues from the roadmap: #48, #68 (Tier-1.1 + 2.2), #70, #71.
- Schema-fit cleanup for the 9 pre-existing `rivet validate` errors
  (SG decomposition link types, CP link-type declarations).

## [1.0.2] - 2026-05-16

**Infrastructure-completion release.** Three tracks of focused
direct work to close the v1.0.0 deferred-list gaps. No agents
(prior session ran into too many stalls). Honest measurement note:
new passes don't fire on the current corpus — wins are coverage
and soundness, not bytes. Real byte gains will surface once PR-Q
(real corpus fixtures) lands.

### Optimization

- **PR-C (Track A): directize MVP — `call_indirect` → `Call`.**
  The directize implementation existed in `loom-core/src/lib.rs`
  since v1.0.0 (silently merged with PR-K3) but was never wired
  into the CLI pipeline. v1.0.2 wires it as a CLI pass between
  `precompute` and `inline`. The pass folds
  `i32.const N; call_indirect (type T)` → `Call(F)` when:
  - No function in the module contains an `Unknown` instruction
    (rules out table.set/.fill/.copy/.init/.grow which all parse
    to `Unknown` today). Conservative module-wide gate.
  - The element section assigns slot N of the table to function F
    via a constant `i32.const` offset (the resolver in
    `build_table_resolver` covers active segments only).
  - F's signature is byte-identical to the call_indirect's
    declared `type_idx`.

  **Z3 verification intentionally bypassed.** The verifier encodes
  `call_indirect(N)` and `call F` as INDEPENDENT uninterpreted
  functions; congruence cannot prove them equal without teaching
  the verifier about the table resolver. Directize's three
  structural guards above imply soundness without Z3 (table is
  immutable + slot resolves to a unique F + signature matches).
  The stack validator runs as defense-in-depth. This is the
  cleanest way to ship directize today; a future PR can wire the
  resolver into the verifier for full Z3 coverage.

- **PR-L3 (Track B): 4 power-of-2 mul → shl rules in peephole_synth.**
  - `x * 128 → x << 7`  (saves 1 byte: LEB128(128)=2 vs LEB128(7)=1)
  - `x * 1024 → x << 10` (1 byte)
  - `x * 65536 → x << 16` (2 bytes)
  - `x * 2^20 → x << 20` (2 bytes)

  We only ship rules where the shift-amount LEB128 is shorter than
  the power-of-two's LEB128 — i.e., for `k >= 7`. Below that the
  rewrite is byte-neutral (both LEB128 encodings are 1 byte) and
  shipping it would just be canonicalization noise.

### Traceability

- **Track C: rivet design-decision gap cleanup.** Authored 5 new
  `DD-*` artifacts in `safety/requirements/design-decisions.yaml`:
  - DD-13 — Explicit error handling (REQ-3, REQ-5)
  - DD-14 — Validator-driven output correctness (REQ-12, REQ-13)
  - DD-15 — Fuzzing via cargo-fuzz (REQ-16)
  - DD-16 — Meld/Kiln ABI compatibility (REQ-17)
  - DD-17 — wasm32-wasip2 as canonical build target (REQ-18)

  **Lifecycle coverage gaps: 9 → 4** (per `rivet validate`). The
  remaining 4 are safety-goal gaps (`SG-3..6` needing
  `safety-context` / `safety-solution`), orthogonal to the
  design-decision cleanup.

### Lifecycle coverage progress across the v1.x arc

| Release | Gaps |
|---|---|
| v1.0.0 | 12 |
| v1.0.1 (verification.yaml) | 9 |
| **v1.0.2 (5 new DDs)** | **4** |

Closing the remaining 4 safety-goal gaps is a v1.0.3+ project that
needs to author `safety-context` and `safety-solution` artifacts
(different artifact types).

### Tests

- **+7 new tests**: 3 directize (positive + non-const + out-of-range)
  + 4 power-of-2 (3 positive shifts + 1 negative non-power-of-2).
- All 8 (+ existing 1) peephole_synth tests pass. All 3 directize
  tests pass.

### Honest measurement note

Re-ran `scripts/measure_corpus.sh` on the corpus. **Code-section
bytes are unchanged from v1.0.1 / v1.0.0.** Directize gates off the
calculator components because they contain `Unknown` instructions
(table mutation in the source); the gale corpus is too small to
have power-of-2 multipliers in our shipped range. The new
infrastructure is correct and tested; byte wins compound once the
corpus grows (PR-Q deferred). The strategic moat on small
adapter-heavy components (PR-M, v0.8.0) — `-18.8%` and `-11.3%` —
is unchanged.

### Still deferred

- **PR-Q: real corpus fixtures** (httparse, nom_numbers, etc.).
  Both prior agent attempts stalled. Defer to v1.0.3.
- Cranelift-style acyclic ægraph mid-end.
- Verifier-side teaching of the table resolver (would let directize
  use Z3 instead of structural guards).
- Pure-arithmetic-expression arms in canonicalize.
- 4 remaining safety-goal lifecycle gaps (SG-3..6).

## [1.0.1] - 2026-05-16

**Verification gate release.** Imports spar's pattern
(commit ba329f3d) of making rivet artifacts EXECUTABLE rather
than purely descriptive. Every requirement REQ-1 through REQ-18
now has at least one `TEST-*` feature artifact with
`fields.method: automated-test` and `fields.steps[].run` shell
commands, linked via `satisfies`.

### Added

- **`safety/requirements/verification.yaml`** (16 TEST-* artifacts).
  Each is `type: feature` with executable steps that the
  Verification Gate CI job runs. Coverage:
  - TEST-Z3-VERIFICATION-CORE → REQ-1, REQ-6
  - TEST-IPA-FUNCTION-SUMMARIES → REQ-1, REQ-4
  - TEST-CSE-SAFETY-GUARDS → REQ-2, REQ-3, REQ-4, REQ-5
  - TEST-CSE-CROSS-CALL-DEDUP → REQ-1, REQ-3
  - TEST-ROCQ-PROOFS → REQ-7
  - TEST-SELF-OPTIMIZATION → REQ-8
  - TEST-WASM-SPEC-COVERAGE → REQ-9
  - TEST-CLI-PIPELINE → REQ-10
  - TEST-COMPONENT-OPTIMIZER → REQ-11
  - TEST-VALID-WASM-OUTPUT → REQ-12, REQ-13
  - TEST-STACK-VALIDATION → REQ-13
  - TEST-DETERMINISTIC-OUTPUT → REQ-14
  - TEST-CORPUS-HARNESS → REQ-15
  - TEST-FUZZING-SMOKE → REQ-16
  - TEST-ABI-COMPATIBILITY → REQ-17
  - TEST-WASM-BUILD-TARGET → REQ-18

- **`tools/run_verification.py`** — ported from spar. Executes every
  `type: feature` artifact's `fields.steps[].run` commands. Reads
  artifacts via `rivet list --filter <sexp>` + `rivet get`, runs
  each step under `bash -c`, writes a `verification-results.json`
  summary with passed/failed/skipped lists.

- **`tools/post_verification_comment.py`** — ported from spar.
  Upserts a single marker-tagged PR comment showing N/M passed,
  per-artifact pass/fail table, and a `<details>` block of failed
  IDs. Finds the marker via `gh api` and PATCHes the body; creates
  a new comment only on first run.

- **`.github/workflows/verification-gate.yml`** — new CI job on PRs.
  Installs rivet at the pinned `b7a17bef` commit (v0.7.0 — first
  release that ships `rivet list --filter <sexp>`). Default filter:
  `(and (= type "feature") (matches id "^TEST-"))`. Per-PR override
  via `Verify-Filter:` line in the PR body. 30-minute timeout.

### Coverage improvement

- **Lifecycle coverage gaps: 12 → 9** (per `rivet validate`).
  Closes all "missing: feature" gaps for REQ-2, REQ-3, REQ-12,
  REQ-13, REQ-14, REQ-17. Remaining 9 are pre-existing
  `design-decision` / `safety-context` / `safety-solution` gaps
  (separate cleanup).

### Cross-project pattern

This is the second pulseengine project (after spar) to adopt the
executable-rivet-artifacts pattern. The same tooling will work
for kiln, meld, witness, and gale once their requirement sets
are similarly mapped.

## [1.0.0] - 2026-05-15

**v1.0.0 — verifier-completion release.** Lifts the verifier-side
blocker that kept cross-call CSE dedup dormant for two releases,
adds the size-threshold fallback that unblocks LOOM on large
bodies, ships a corpus-wide cargo bench, and pins the wasm-opt
comparison version. Three of the five planned tracks landed
(A: verifier, B: size-threshold, E: bench). Track C (directize)
and Track D (real corpus fixtures) were deferred when their
agents stalled.

### Verifier (unblocks dormant infrastructure)

- **PR-K3 (#115, Track A): model pure+no-trap `Call` as
  uninterpreted function for Z3 congruence.** The Z3 translation
  validator modeled every `Instruction::Call` as a fresh
  `BV::new_const`, so two identical pure helper calls produced
  INDEPENDENT symbolic constants — Z3 found counter-examples for
  every CSE dedup attempt, which `verify_or_revert` dutifully
  reverted. PR-K3 encodes pure+no-trap+single-result callees as
  `pure_call_<idx>(args)` via `FuncDecl::apply`; Z3's congruence
  closure proves two identical such expressions equal. Combined
  with passing the sig context through CSE's `TranslationValidator`,
  the v0.8.0 PR-K + v0.9.0 PR-K2 cross-call dedup feature now
  actually fires.

- **PR-K3.2 (#115, Track B): size-threshold fallback on Z3
  invocation.** The per-function Z3 validator scales poorly on
  large bodies (>60 min on the meld-fused 2.3 MB calculator
  core). Bodies above `LOOM_Z3_MAX_INSTRUCTIONS` (default 2000,
  env-overridable) skip Z3 and rely on the stack validator that
  every pass already runs. Conservative-over-fast in the opposite
  direction: we ship a result we couldn't deeply verify rather
  than fail to ship at all on real workloads. Per-pass
  `<pass>/z3-size-skipped` stat records when the gate fires.

### Measurement (closing the loop)

- **PR-bench (#116, Track E): criterion-based corpus baseline +
  wasm-opt version pinning.** New `loom-testing/benches/corpus_baseline.rs`
  (~870 LOC) replicates `scripts/measure_corpus.sh` as a cargo
  bench. Output to stdout + versioned report file. wasm-opt
  version pin at `scripts/wasm-opt.pinned` (currently
  `version_116`); bench startup compares installed vs pinned,
  prints non-fatal warning on mismatch, marks columns `n/a` if
  wasm-opt absent.

### Tests

- Five CSE tests now pass (was 3 pass + 2 ignored since v0.9.0):
  - `test_cse_dedupes_repeated_pure_calls`
  - `test_cse_dedupes_pure_clamp_calls_via_span_replacement`
  - plus the three safety tests (impure / may-trap / different-args)
- All 10 `summary::` IPA tests continue to pass.
- 330+ loom-core lib tests pass.

### Deferred from v1.0.0 sprint

- **PR-C (Track C): `directize` MVP.** Agent stalled. Defer to v1.0.1.
- **PR-Q (Track D): real corpus fixtures.** Agent stalled (couldn't
  build httparse / nom_numbers / etc. in time budget). Defer to v1.0.1.
  The harness honest-marks them `n/a` today.

### Deferred to future (still tracked)

- PR-L3: power-of-2 mul/div → shift rules.
- Pure-arithmetic-expression arms in canonicalize (broader
  `if/else → select`).
- Cranelift-style acyclic ægraph mid-end.
- Z3 verifier completeness (full exit-state equivalence,
  BrTable path predicates, float `rustc_apfloat`).

### What v1.0.0 means

v1.0.0 marks the point where LOOM's cross-call optimization
infrastructure is **end-to-end functional**: recognition (PR-K),
replacement (PR-K2), and verification (PR-K3) all in place.
Combined with the measurement harness (PR-P, PR-R, PR-bench),
LOOM ships its first release where the optimization-vs-baseline
story is fully measurable and the strategic moat (Component-Model
adapter specialization) is empirically validated against
wasm-opt -O3 on multiple workloads.

## [0.9.0] - 2026-05-14

**Measurement and harvest release.** First objective measurements of
LOOM against wasm-opt -O3 across multiple workloads, plus harvesting
of v0.8.0's infrastructure into concrete wins on component-shaped
fixtures. Three PRs landed in parallel via worktree-isolated agents
(K2 = CSE span-replacement infrastructure + verifier-gap finding,
L2 = peephole rule set growth + missing pipeline wiring, P = corpus
harness + measurement report).

### Measurement (finally)

The first corpus-wide measurement harness (PR-P) reveals where
v0.8.0's infrastructure actually pays off:

| Workload | Baseline | LOOM | wasm-opt -O3 | LOOM Δ% | wasm-opt Δ% |
|---|---:|---:|---:|---:|---:|
| gale | 1,941 | 1,846 | 1,925 | **−4.9%** | −0.8% |
| calculator_root | 2,337,724 | 2,327,794 | (errors) | −0.4% | n/a |
| simple_component | 261 | 212 | (errors) | **−18.8%** | n/a |
| calc_component | 442 | 392 | (errors) | **−11.3%** | n/a |

**Two strategic facts established:**
1. LOOM beats wasm-opt -O3 on gale by 4.1 points at total-file
   level. First measured workload where LOOM dominates.
2. PR-M (v0.8.0 component adapter specialization) delivers −11% to
   −19% on small adapter-heavy components. The percentage dilutes
   on large components because most bytes are core code. The
   strategic moat is real: wasm-opt cannot process Component-Model
   components at all (errors out on every one).

### Optimization

- **PR-L2 (#112): grow Souper rule set to 12 identities + wire into
  pipeline.** PR-L (v0.8.0) shipped the module but DID NOT register
  the pass with the CLI optimizer — discovered during measurement
  via the new `--stats` per-pass breakdown. PR-L2 fixes the wiring
  and adds 9 new rules (i32.mul·1, i32.sub-0, three shift-by-zero
  variants, four i64 identities), each with documented one-line
  algebraic proof. 24 tests pass (was 6).

- **PR-K2 (#111): span-based CSE replacement infrastructure +
  verifier-gap finding.** PR-K (v0.8.0) recognized pure+no-trap
  Call expressions in CSE but couldn't replace them because the
  existing replacement loop only handled single-instruction Const
  exprs. PR-K2 implements `CSEAction::ReplaceSpanWithLoad` with
  five defense-in-depth gates (single-instruction args, span-length
  invariant, occupied-bitmap overlap rejection, no-set-between-calls
  hazard check, cost gate). **Critical finding**: the Z3
  translation validator models every `Instruction::Call` as a fresh
  symbolic constant, so it rejects every CSE dedup with a
  counterexample. Per LOOM's proof-first policy, the tests stay
  `#[ignore]`'d until PR-K3 fixes the verifier to model pure+no-trap
  calls as uninterpreted-function applications `f(args)`.

### Infrastructure

- **PR-P (#110): corpus-wide measurement harness.** New
  `scripts/measure_corpus.sh` + `docs/measurements/v0.9.0-corpus-baseline.md`.
  Runs LOOM and wasm-opt -O3 across configured fixtures, validates
  every output via wasm-tools, hard-errors on invalid wasm, emits
  a markdown delta table per release. The harness flagged a +45.6%
  regression on gale during initial validation — investigation
  showed the cause was the `--attestation true` default embedding
  a ~980-byte security custom section. The harness now passes
  `--attestation false` for measurement runs so the byte-delta
  column reflects optimization quality, not security overhead.

### Tests

- **+42 tests** across the three PRs: PR-L2 (+18 fire/negative
  pairs), PR-K2 (+1 ignored, +1 reframed), PR-P (harness produces
  validation-or-fail results per workload).

### Deferred to v1.0.0

- **PR-K3 (verifier-side)**: model pure+no-trap `Call` in
  `verify.rs` as uninterpreted-function applications. Unblocks
  PR-K2's tests and the entire cross-call dedup feature.
- **PR-L3**: power-of-2 mul/div → shift rules (needs overflow-guard
  reasoning for signed div).
- **PR-Q**: land a real corpus (httparse, nom_numbers, state_machine,
  json_lite, loom-self-build) under `tests/corpus/`. The harness
  is ready; it just marks them `n/a` today.
- Pure-arithmetic-expression arms in canonicalize (broader
  `if/else → select`).
- Cranelift-style acyclic ægraph mid-end.

## [0.8.0] - 2026-05-14

Cross-call optimization release. Four PRs landed in parallel via
worktree-isolated agents — three completed by agents, one rescued
from an agent timeout. Two PRs build directly on PR-F's function-
summary IPA (PR-J extends the pure-call-drop fold to N args; PR-K
wires Call recognition into CSE), one ships the harness for
algorithmic-solver-driven optimization (PR-L), and one delivers
LOOM's first concrete win on post-meld component output (PR-M).

### Optimization

- **PR-J (#105): arg-aware pure-call-drop fold.** Extends PR-F
  (v0.7.0). v0.7.0's vacuum peephole only folded ZERO-arg
  \`Call f; Drop\` pairs because removing a Call without removing
  its arg-pushers would leave dangling values on the stack. PR-J
  lifts that restriction when the N preceding instructions are
  themselves all pure pushers — N pure pushers contribute exactly
  N values and nothing observable; removing them alongside the
  Call+Drop preserves stack balance and observable behavior. A
  pure pusher's signature is (consumes 0, produces 1), so this
  is sound by stack-effect arithmetic.

- **PR-K (#106): CSE cross-call dedup recognition (INFRASTRUCTURE).**
  Adds \`Expr::Call { func_idx, args }\` to the CSE expression
  model. The CSE scan loop now recognizes pure + no-trap +
  single-result calls as deterministic values of their args,
  hashing and cost-gating them alongside arithmetic subtrees.
  The REPLACEMENT side (turning a duplicate call site into a
  \`local.get\`) requires span-based substitution and is deferred
  to a follow-up PR-K2. This PR ships the recognition + the
  three safety guarantees (impure / may-trap / different-args
  must NOT dedupe).

- **PR-L (#107): Souper-shaped peephole synthesis MVP.** Minimal
  first cut of the algorithmic-solver direction from
  \`docs/research/v0.7.0/algorithmic-solver-feasibility.md\`. New
  module \`loom-core/src/peephole_synth.rs\` with the harness +
  three hand-curated right-identity rules with documented
  algebraic proofs: x+0=x, x|0=x, x&(-1)=x. Iterate-to-fixpoint
  linear scan with stack-validation safety net. A follow-up
  PR-L2 adds the startup-time Z3 candidate-admission gate once
  the rule set grows past hand-audit size.

- **PR-M (#108): Component-Model adapter specialization.** LOOM's
  first concrete win on post-\`meld\`-fusion output. wasm-opt
  operates on core wasm and cannot see adapter residue at all;
  this is LOOM's strategic moat. New \`specialize_adapters\` pass
  (Phase 3 of the component pipeline) folds canon lift/lower
  trampolines: empty blocks with identity signatures get
  unwrapped. Two-layer revert safety net (encoder+validator
  check after fold; \`is_block_safe_to_eliminate\` requires
  byte-identical types).

### Tests

- New tests across the four PRs: PR-J (+4), PR-K (+3 pass + 1
  ignored for PR-K2), PR-L (+6), PR-M (+8). 308+ loom-core lib
  tests pass.

### Infrastructure

- **Worktree-isolated agent parallelism**. The v0.8.0 sprint
  validated parallel PR development using \`git worktree\` —
  each agent in its own working directory + target/ cache. PR-M
  recovered from an agent stream timeout (497 LOC + 8 tests
  intact in worktree); rescue time was under 10 minutes. Without
  worktrees, concurrent \`cargo build\` runs would have clobbered
  each other's target/ caches.

### Soundness reasoning

Every PR explicitly enumerates the safety conditions it relies
on:
- PR-J: stack-effect arithmetic (pure pusher = consumes 0,
  produces 1).
- PR-K: IPA correctness (pure ∧ no-trap ∧ args-determinate).
- PR-L: algebraic identity proofs documented per-candidate.
- PR-M: byte-identical block type, empty body, two-layer
  revert.

### Measurement

No measurable change on gale_in_baseline (still 795 B / -1.97%
from v0.7.0). Gale's wasm doesn't have the patterns these passes
target (no pure-call-drop chains, no duplicated pure calls, no
trivial arithmetic identities, no component adapters). This
release is INFRASTRUCTURE — the wins compound when applied to
component-shaped workloads (calculator-class) and to hand-written
wasm that hasn't already been folded by an upstream compiler.

### Deferred to v0.9.0

- **PR-K2**: span-based CSE replacement (the replacement half
  of cross-call dedup).
- **PR-L2**: Z3 startup-time candidate admission gate for
  peephole synthesis.
- **PR-N**: Verus-clause ingestion MVP per the gale deep-scan
  shortlist (10 ranked clauses with file:line citations).
- **PR-O**: Cranelift-style acyclic ægraph mid-end (from the
  optimization-methods survey).

## [0.7.0] - 2026-05-13

Infrastructure-and-cleanup release. Closes the trivial pipeline-order
gap in v0.6.0's vacuum, lands function-summary interprocedural
analysis (IPA) as the foundation for cross-call optimization, and
adds verification-aware canonicalization as the IR-normalization
substrate that downstream passes (and the Z3 verifier) compose
better against.

### Optimization

- **Pipeline order fix: `vacuum-final` step** (PR-E, #101). v0.6.0's
  PR #99 added a `pure_push;Drop` peephole to `vacuum`, but vacuum
  ran BEFORE `dead-stores`/`dead-locals` in the pipeline — so the
  const+drop pairs those passes create were never folded. Added a
  second `vacuum-final` sweep after the dead-* passes. **Measured on
  gale_in_baseline (1.9 KB Verus-verified kernel FFI): code section
  811 B (baseline) → 795 B (v0.7.0), -1.97% net.** Compared to v0.5.0's
  +6.3% regression, that's a 6.7-point swing on real kernel-scheduler
  code, with a 30-LOC change.

- **Function-summary interprocedural analysis (IPA)** (PR-F, #102).
  New module `loom-core/src/summary.rs` (~250 LOC). Computes
  per-function `is_pure` / `is_no_trap` summaries via
  optimistic-then-demote fixpoint over the call graph. `CallIndirect`
  and unsupported instructions conservatively mark caller impure +
  may-trap. Mutual recursion converges naturally. Vacuum's
  `peephole_const_drop` extended to fold `Call f; Drop` for pure +
  no-trap + zero-arg + single-result helpers — the safe minimum. No
  measurable change on gale (no zero-arg drops there); infrastructure
  for future CSE cross-call dedup and DCE-on-pure-calls.

- **Verification-aware canonicalization pass** (PR-G, #103). Two
  rewrites:
  - `if/else → select` for single-value if/else whose both arms are
    each one pure pusher. Wasm's `select` is path-INSENSITIVE, which
    makes every downstream pass and the Z3 verifier reason about it
    without branch analysis. Restricted to pure-pusher arms (constants,
    `local.get`, `global.get`) because `select` evaluates both arms
    eagerly — a trapping or side-effecting arm would change behavior.
  - `local.set X; local.get X → local.tee X` peephole. Equivalent
    stack effect, saves 2 bytes per occurrence. Index must match
    exactly. Safe regardless of context.
  Wired in early (right after `constant-folding`, before `cse`) so
  the rest of the pipeline sees canonical forms.

### Tests

- New tests across PR-E, PR-F, PR-G (288+ loom-core lib tests pass).
- Soundness regression test pinning the v0.4.0-era store-hoist
  invariant (`test_null_check_before_store_preserved_through_optimization`):
  runs the full v0.7.0 pipeline on the gale `sem_count_take` shape
  and asserts `i32.eqz` precedes `i32.store`.

### Research

Four v0.7.0-planning research documents land alongside the code:

- `docs/research/v0.7.0/issue-triage-68-75.md` — verdicts on #68-#75
  (notably: #69 is a merged PR not an issue; #68 is ~40% already
  shipped in `fused_optimizer.rs`; #75 duplicates #70).
- `docs/research/v0.7.0/optimization-methods-survey.md` — 13-family
  compiler-optimization survey with verdicts for v0.7.0/v0.8.0.
  Recommended order: function-summary IPA (this release), verification-
  aware canonicalization (this release), Souper-style verified peephole
  synthesis, Component-Model adapter specialization, ægraph mid-end
  (deferred to v0.8.0).
- `docs/research/v0.7.0/gale-deep-scan.md` — gale-specific opportunity
  catalog with file:line citations and a 10-clause Verus axiom
  ingestion shortlist.
- `docs/research/v0.7.0/algorithmic-solver-feasibility.md` — Souper-
  shaped SMT-driven peephole synthesis recommended for v0.7.0+,
  4-5 weeks, ~1500 LOC.

### Deferred to v0.8.0

- Souper-shaped SMT-driven peephole synthesis.
- Arg-aware extension of the pure-call-drop fold (pop N preceding
  pure pushers when arg-count is N).
- Pure-arithmetic-expression arms in canonicalize (broader if/select).
- CSE cross-call dedup using summaries.
- DCE on pure calls with unused results.
- Cranelift-style acyclic ægraph mid-end.

## [0.6.0] - 2026-05-11

This release is driven entirely by real-world findings on production
gale code (Verus-verified kernel-scheduler FFI wasm): closing v0.5.0's
+6.3% CSE size regression, lifting two early-exit guards that made
LOOM a no-op on kernel-style code, and fixing a Z3 panic that blocked
the inline pass on i64-heavy modules. Net effect on the gale_ffi
fixture: code section -0.86% (was +6.3% in v0.5.0). Net effect on
a 2.3 MB calculator.wasm component: -0.4% from the new dead-store
pass alone.

### Optimizer correctness (Z3 / inline)

- **Closed `inline_functions` Z3 SortDiffers panic on i64-heavy wasm**
  (PR-D, closes #98). The verifier's symbolic-locals initialization
  defaulted to 32-bit width regardless of declared type at three
  sites; the gale-ffi crate (u64-packed FFI returns) crashed every
  inline attempt with `SortDiffers { left: BitVec(64), right: BitVec(32) }`.
  Fix: new helpers `local_type_at` + `bv_width_for_value_type`
  resolve param/local types correctly at each extension site.
  Defensive `match_bv_widths` zero-extend helper added for future
  binop-site backstop.

### Optimizer code size on real workloads

- **CSE cost gate eliminates the gale +6.3% regression** (PR-A).
  v0.5.0's enhanced CSE deduplicated every duplicate expression
  including 1-2 byte constants. Replacing `i32.const -EINVAL`
  (2 bytes) with `local.tee N / local.get N` (4 bytes) plus a new
  local declaration was unconditionally a size regression.
  New `Expr::worth_dedup(occurrences)` predicate estimates net byte
  savings via the formula `net = (N-1)·(cost-2) - 4` and skips when
  non-positive. Gale code section: 862 → 808 bytes.

- **`eliminate_dead_locals` pass** (PR-B): drops locals declared by
  a function but never read by any `LocalGet` anywhere in the
  function body. Targets the gale "default-then-override" pattern
  (rustc materializes an EINVAL default that every reachable path
  overwrites). The rule is path-INSENSITIVE — sound regardless of
  BrIf/BrTable/early-Return control flow — so the pass DOES NOT
  need the `has_dataflow_unsafe_control_flow` guard that previously
  made `simplify_locals` and `coalesce_locals` no-ops on every
  kernel-style function. Gale code section: 808 → 804 bytes.
  Asymmetric write neutralization: `LocalSet → Drop` preserves
  `[T] → []`; `LocalTee → removed` lets `[T] → [T]` pass through.

- **`eliminate_dead_stores` pass with full backward liveness** (PR-C):
  per-position dead-store elimination via backward liveness walk
  over the structured wasm instruction tree. Handles Block/If
  precisely (`live-before-if = live-in-then ∪ live-in-else`), Br/
  BrIf/BrTable via label-stack indexing, Return/Unreachable as
  no-continuation. Loop bodies use a conservative approximation
  (everything read anywhere in the body is live throughout) — sound
  but imprecise inside loops; loop fixpoint precision is a follow-up.
  Net effect on calculator.wasm: -0.4% from this pass alone (~10 KB
  on a 2.3 MB component).

### Cleanup follow-ups

- **`vacuum` const+drop peephole** (PR-D). PR-B/PR-C neutralize
  dead `LocalSet idx` to `Drop`, leaving the value-pusher
  immediately followed by Drop. New `peephole_const_drop` folds
  `pure_push;Drop` pairs (constants, LocalGet, GlobalGet), recursing
  into Block/Loop/If bodies. NOT folded: memory loads, calls,
  anything that can trap — discarding the result does not discard
  the trap.

### Research outputs

- `docs/research/gale-v0.5.0/source-pattern-analysis.md` — eight
  optimization-relevant patterns found in gale source with
  file:line citations (FSM dispatch, default-then-override, Verus
  `decreases` bounded loops, tail-call dispatch, leaf-inline +
  const-prop candidates, bit-mask axioms, 2D `match (state, event)`,
  Verus annotations as trusted axioms).
- `docs/research/gale-v0.5.0/wasm-opt-gap-analysis.md` — ranked top
  7 wasm-opt passes by expected payoff on kernel-style code, with
  per-pick LOC and complexity estimates. Picks #1, #2 (narrowed),
  and #3 are shipped in this release.

### Test count

303 → 317 tests passing (+14 across all v0.6.0 PRs).

## [0.5.0] - 2026-05-02

This release closes a real soundness bug discovered on production
gale code (kernel-scheduler FFI, Verus-verified Rust), gives passes
a way to opt into strict verification semantics, expands the test
corpus to post-MVP wasm features, wires `FusedOptimization.v` into
the Bazel proof build, and ships CI concurrency hardening.

### Soundness

- **Closed CSE hoist hole on early-exit patterns** (PR-B). v0.4.0's
  `has_dataflow_unsafe_control_flow` only flagged `BrIf`/`BrTable`,
  letting the canonical Rust `if (cond) return; end` early-exit
  guard slip through. Per-pass tracing on `gale_sem_count_take`
  showed the reordering happens in `constant_folding`'s
  `instructions_to_terms` → rewrite → `terms_to_instructions`
  round-trip — once a function with `If { then_body: [..., Return] }`
  is converted to terms and back, the if-block can land at the
  function tail with the straight-line code (load/sub/store)
  hoisted to the function head, so a store now runs even when
  the guard would have returned early.
- The fix extends the guard to flag any *nested* `Return`/`Br` as
  an early-exit pattern (top-level Return at the function tail is
  still allowed — that's the function terminator). New helper
  `has_unsafe_in_nested` recurses into `Block`/`Loop`/`If` bodies.
- `constant_folding` and `optimize_advanced_instructions` now skip
  the function entirely when this guard fires (previously they
  fell back to `rewrite_pure`, which still went through the unsound
  round-trip). Defense-in-depth guards added to `simplify_locals`,
  `remove_unused_branches`, `optimize_added_constants`. DCE
  intentionally NOT guarded — it only deletes unreachable code,
  never reorders.
- New regression test `test_early_return_guard_prevents_store_hoist`
  mirrors `gale_sem_count_take` and pins the fix.

### Verification API

- **`VerificationResult::is_skip()` and `skip_reason()`** (PR-A).
  Lets callers distinguish "Z3 proved equivalence" from "verifier
  silently bailed because input was unverifiable." Closes audit S5.
- **`TranslationValidator::verify_or_revert_strict()`** (PR-A).
  Strict counterpart to `verify_or_revert`: only `Verified` is
  accepted; `Skipped*`, `Failed`, and `Error` all revert. Reverts
  recorded under `<pass>:strict-skip` for separability. Stub for
  non-verification builds returns false (REQ-5: when Z3 isn't
  available we don't hoist).
- Existing `is_equivalent()` and `is_verified()` doc comments now
  name the lenient/strict semantics explicitly.

### Research

- `docs/research/gale-v0.4.0/measurement-report.md` (PR-A) — LOOM
  v0.4.0 vs `wasm-opt -O3` on a Verus-verified kernel-scheduler FFI
  (`gale_ffi`: sem + timer + spinlock + ring_buf + bitarray + rbtree).
  Headline: LOOM regresses code-section size by +6.3% on this
  workload while wasm-opt reduces by -2.0%, primarily because LOOM's
  CSE deduplicates trivially-cheap small constants into
  `local.tee/local.get` pairs that grow function headers. The CSE
  soundness bug discovered in this report motivated PR-B.

### Test corpus

- 8 minimal post-MVP wasm fixtures (PR-C): SIMD/v128, ref types,
  bulk memory, tail calls, exception handling, multi-memory,
  sign-extension, saturating-trunc.
- New `loom-core/tests/spec_features.rs` runs each through one of
  three buckets: clean rejection (unsupported), clean rejection or
  round-trip (partial), or full optimize+roundtrip (supported).
  Pins the contract that no post-MVP feature crashes the parser.

### Proofs

- `proofs/simplify/FusedOptimization.v` is now in `BUILD.bazel`
  (PR-D). Closes audit D1: the file backs the fused/meld pipeline
  with 7 axioms + 8+ theorems and was previously orphaned from
  the build, so CI never compiled or checked it. Axioms remain
  unchanged in this PR; discharging them is future work.

### CI

- **Top-level `concurrency:` block on every workflow** (PR
  `chore/ci-concurrency-control`). Closes the queue-backlog issue
  identified org-wide: superseded PR runs are now cancelled within
  ~30 seconds; runs on `main`, tags, releases, and scheduled events
  are never cancelled. `release.yml` and `fuzz.yml` use the
  release/scheduled variants per the cross-repo brief. Expected
  effect: 30–40% reduction in CI compute on PR-heavy days.

### Deferred

The following audit findings remain out of scope for v0.5.0 and
will be picked up later:

- Full exit-state Z3 equivalence (return + locals + globals + memory).
  v0.5.0 only changes the API surface (`VerificationResult` predicates
  + strict-mode revert helper); the encoder still asserts only on
  top-of-stack at function exit.
- BrTable arm encoding with path predicates and per-arm state merge.
- Switch float fold to `rustc_apfloat` for bit-exact wasm semantics.
- Discharge the 7 axioms in `FusedOptimization.v`.
- Wire or remove the 5 implemented passes never called from
  `optimize_module`.

## [0.4.0] - 2026-05-01

This release closes the path-sensitivity hoist hole at the pass level,
adds verification observability, aligns docs and safety artifacts with
code, and removes vestigial scaffolding identified by the v0.4.0 audit.

### Soundness

- **Hoist guards on path-sensitive passes** (PR-B). Until the Z3
  verifier becomes path-sensitive across `Br`/`BrIf`/`BrTable`, the
  following passes now skip functions containing `BrIf`/`BrTable`:
  `loop_invariant_code_motion`, `code_folding` (tail-merging),
  `coalesce_locals`, `eliminate_common_subexpressions`,
  `eliminate_common_subexpressions_enhanced`. The verifier today
  asserts equivalence only on top-of-stack at function exit, so
  hoisting code across these branches could pass verification while
  being unsound on the branch-out arm. Conservative-over-fast (REQ-5).
- **Encoder error-path robustness** (PR-B). `lib.rs:1891` panic
  replaced with a clear `anyhow!` error. `lib.rs:2868` multi-value
  block-type panic replaced with `unreachable!()` and an invariant
  comment.

### Observability

- **Per-pass revert counter** (PR-D). Every silent revert via
  `verify_or_revert` and the manual revert sites in
  `constant_folding`, `simplify_locals`, `coalesce_locals`,
  `optimize_advanced_instructions`, and the component pipeline now
  records to `loom_core::stats::record_revert(pass_name)`. The
  CLI `--stats` output gains a "🔁 Verification Reverts" section
  with per-pass counts and a total.
- **Visible verifier-skip diagnostics** (PR-C). When the verifier
  silently auto-passed unverifiable inputs (float load/store,
  unknown opcodes), it now emits a one-line diagnostic with pass
  name + skip reason.
- **Regression test for the default-then-override br_table pattern**
  (PR-C) — `test_default_then_override_across_br_table_preserved`
  asserts the WAKE-default `i32.const 1; local.set` survives
  optimization. Passes today because PR-B's hoist guards skip the
  hoist-prone passes on these functions.

### Documentation

- **Pipeline phase counts removed from docs** (PR-A). README,
  usage.md, quick-reference.md, architecture.md, REQ-10, DD-4 no
  longer state "12-phase" / "10-phase" / "11-phase" pipelines —
  numbers go stale, descriptions don't.
- **Z3 verification framing corrected** (PR-A). README no longer
  describes Z3 as opt-in; `default = ["verification", "attestation"]`
  is on by default per `loom-core/Cargo.toml`. Document
  `--no-default-features` as the disable path.
- **WASM build command corrected** (PR-A). The README's broken
  `bazel build //loom-cli:loom_wasm --platforms=...` replaced with
  the cargo invocation CI actually uses
  (`rustup target add wasm32-wasip2 && cargo build --release
  --target wasm32-wasip2 --package loom-cli`).
- **Inline heuristics corrected** (PR-A). architecture.md now
  reflects code (`call_count == 1 || size < 10`, hard cap `< 50`)
  rather than the prior `body_size < 20 && call_count > 2`.
- **z3-status.md updated** (PR-A). Integer memory ops are
  Z3-verified via Array theory; document known model limitations
  (top-of-stack-only equivalence, `Br`/`BrIf`/`BrTable` break-
  semantics, `contains_unverifiable_instructions` silent auto-pass).

### Safety artifacts

- **H-17, H-18, H-19, H-20 marked MITIGATED** (PR-A) with the
  resolving commits / PRs cited.
- **DD-4 ↔ REQ-10 aligned** (PR-A). Both now describe the canonical
  pass list without phase numbers.
- **Float helper extraction landed** (PR-A). The merged Wave 3 PR
  description claimed `F32_CANONICAL_NAN`, `F64_CANONICAL_NAN`,
  `is_f32/f64_subnormal`, `canonicalize_f32/f64` — but the actual
  helper-extraction commit had been left on an unmerged branch.
  PR-A cherry-picks `1ea0d43` so they actually exist on main.
- **AGENTS.md / .rivet/agent-context.md regenerated** (PR-A) via
  `rivet init --agents --migrate` and `rivet context`. Stale
  duplicate "Project Overview" content removed (was claiming 144
  artifacts / 55 errors / 3 UCAs; reality is 207 / 9 errors / 25 UCAs).

### Cleanup

- **Vestigial ISLE files removed** (PR-E):
  `loom-shared/isle/wasm_terms.isle` (323 lines of parallel term
  declarations) and `loom-shared/isle/rules/constant_folding_v2.isle`
  (alternative rules never compiled). Neither was in `build.rs`'s
  compiled list. `build.rs` now carries a top-of-file note
  explaining the live ISLE codegen is retained only for its
  immediate-value constructors used by the Rust rewriters.
- **Path-sensitivity scaffolding documented** (PR-E).
  `ExecutionState`/`BlockResult`/`merge_states` in `verify.rs`
  remain `#[allow(dead_code)]` but now carry doc comments
  identifying them as the intended target for the verifier-model
  upgrade tracked in PR-C deferred work.
- **`.gitignore`** (PR-A): `bazel-*` artifacts and
  `.claude/worktrees/` to keep `git status` clean.

### Deferred to a future release

The following were investigated by the v0.4.0 audit and are not
addressed in this release:

- Compare full exit state (return + locals + globals + memory) in
  the Z3 verifier rather than top-of-stack only.
- Encode `BrTable` arms with path predicates and per-arm state merge.
- Replace `contains_unverifiable_instructions` silent `Ok(true)`
  with a structured "verification skipped" return type so callers
  can decide per-pass policy.
- Switch float fold to `rustc_apfloat` for bit-exact wasm semantics
  (host-FPU and sNaN canonicalization concerns).
- Wire `proofs/simplify/FusedOptimization.v` into BUILD.bazel and
  discharge its 7 unverified axioms.
- Wire or remove the 5 implemented passes never called from
  `optimize_module`: `branch_simplification`, `block_merging`,
  `vacuum_cleanup`, `precompute`, `eliminate_common_subexpressions_enhanced`.
- Add corpus fixtures for SIMD, reference types, GC, tail calls,
  exception handling, threads/atomics, multi-memory.

### Dependency / toolchain

- Verified compatibility with rustc 1.95 (clippy 0.1.95
  `collapsible_match` lint).
