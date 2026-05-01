# Changelog

All notable changes to LOOM will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
