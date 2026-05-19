# #48 Parser/Encoder Round-Trip Identity — Survey + Plan

_v1.0.5 Track 4 deliverable. Read-only survey; no Rocq code written here._

## Headline

The v1.0.3 roadmap classified #48 as **L (~1500 LOC Rocq + ~200 LOC subset-extraction tooling on the Rust side)**, blocked by **closing existing `Admitted` lemmas first**. This survey confirms that classification: 4 `Admitted` lemmas remain across 3 `.v` files; 3 of those 4 are in the **placeholder** files (Roundtrip, TermBijection, StackSignature), not the real proof files (which are 100% closed). The real prerequisite work is replacing the placeholder files' Axiom-then-Admit pattern with the existing proven content from `proofs/rust_verified/`.

## Current proof state

| File | Status | Notes |
|---|---|---|
| `proofs/rust_verified/isle_conversion_proofs.v` | **23 Qed, 0 Admitted** | The REAL term-bijection proof. Fully closed. |
| `proofs/rust_verified/stack_signature_proofs.v` | **23 Qed, 0 Admitted** | The REAL associativity proof. Fully closed. (Per v1.0.3 issue #47 closure.) |
| `proofs/rust_verified/value_types_proofs.v` | closed | Foundational types, no Admitteds. |
| `proofs/rust_verified/value_types_translated_proofs.v` | closed | Type-translation invariants. |
| `proofs/simplify/*.v` (5 files) | closed | Identity, StrengthReduction, ConstantFolding, FusedOptimization (7 axioms — flagged in CHANGELOG v0.5.0 deferred-items), Bitwise. |
| `proofs/semantics/{Term,Wasm}Semantics.v` | closed | Semantic models. |
| `proofs/Correctness.v` | closed | Top-level wiring. |
| **`proofs/codec/Roundtrip.v`** | **1 Admitted** (35 LOC) | **PLACEHOLDER**: types are stubs (`EmptyModule`), parse/encode are `Axiom`-ed. Nothing real. |
| **`proofs/isle/TermBijection.v`** | **2 Admitted** (42 LOC) | **PLACEHOLDER**: types are stubs (3-instr enum), conversions are `Axiom`-ed. Real proof lives at `proofs/rust_verified/isle_conversion_proofs.v`. |
| **`proofs/stack/StackSignature.v`** | **1 Admitted** (283 LOC) | Real content for partial-composition associativity, ONE remaining Admit on the "extensive case analysis" branch. The exact-match equivalent is fully proven in `stack_signature_proofs.v` Theorem 13. |

## Two paths forward

### Path A — close #48 properly (the L-effort one)

Steps:

1. **Replace `proofs/isle/TermBijection.v`** with a thin re-export of `proofs/rust_verified/isle_conversion_proofs.v`. The placeholder file's Instruction stub (`I32Const | I32Add | End`) is a strict subset of the real proof's instruction set; we just need a `Require Import` and a derived theorem statement. Closes 2 Admitteds, ~30 LOC churn.

2. **Discharge the `StackSignature.v` partial-composition lemma** by case-splitting on `length results_a` vs `length params_b`. The existing `stack_signature_proofs.v` Theorem 13 provides the exact-match base case; the partial-match cases extend it by induction on the unmatched suffix. Estimated ~200 LOC Rocq. Closes 1 Admitted.

3. **Author the real `Roundtrip.v`** scoped to the ISLE-tracked subset of `Module`:
   - Define `Inductive ScopedModule` mirroring `crate::Module` minus features outside the ISLE op set (no SIMD/ref types/etc.).
   - Build a Rocq model of LEB128 + section encoder for that subset using `Coq.Strings.Byte`.
   - Prove `parse_subset (encode_subset m) = Some m` by induction on `m`'s structure.
   - Estimated ~1200 LOC Rocq + ~150 LOC Rust-side subset-extraction tooling (a `to_scoped_module` translator with a property test gating that every existing fixture's `Module` round-trips through it).

4. **Wire into Bazel** at `proofs/BUILD.bazel`. The existing TEST-ROCQ-PROOFS verification artifact (in `safety/requirements/verification.yaml`) will pick it up automatically.

**Total**: ~1400 LOC Rocq + ~150 LOC Rust. **L-effort**, 2-3 weeks of focused Rocq work.

### Path B — close the placeholder Admitteds without doing #48 (the S-effort one)

If we just want `rivet validate` to stop reporting Admitteds in the placeholder files (signal-cleanup, not REQ-12 closure):

1. **Delete `proofs/codec/Roundtrip.v` entirely**, or make it a doc-only `Roundtrip.md` cross-referencing the eventual real proof location.
2. **Delete `proofs/isle/TermBijection.v`**, or make it a thin `Require Import` re-export of `isle_conversion_proofs`.
3. **Mark the `StackSignature.v` partial-composition lemma `Admitted`** with an explicit `Notation` referencing the exact-match `stack_signature_proofs.v` Theorem 13 as the operational substitute.

This DOES NOT close #48 — it just removes the 3 placeholder Admitteds from CI signal. The roadmap entry for #48 explicitly notes: "should land *after* `proofs/rust_verified/isle_conversion_proofs.v` is closed" — and it is. So Path A is now unblocked.

## Recommendation

**Defer Path A to v1.1.0** (the L-effort work needs a focused 2-week Rocq sprint, not a track in a parallel-multi-track release). **Do Path B in v1.0.5 cleanup** if it fits — but it's signal-noise reduction, not byte-mover-class progress. If neither, leave #48 as "KEEP, deferred to v1.1.0" with this survey as the planning record.

## Bazel test invocation (for future reference)

```
bazel test //proofs:roundtrip_test
```

After Path A lands, this should pass with no `Admitted` anywhere in `proofs/codec/`. Bazel's `rules_rocq_rust` runs `coqc` and fails the test on Admitteds; the existing TEST-ROCQ-PROOFS verification artifact wraps this in CI.

## Files referenced

- `proofs/codec/Roundtrip.v` (placeholder, target of Path A step 3)
- `proofs/isle/TermBijection.v` (placeholder, target of Path A step 1)
- `proofs/stack/StackSignature.v` (1 Admitted, target of Path A step 2)
- `proofs/rust_verified/isle_conversion_proofs.v` (the real bijection proof — already closed)
- `proofs/rust_verified/stack_signature_proofs.v` (the real associativity proof — already closed)
- `safety/requirements/verification.yaml` (TEST-ROCQ-PROOFS artifact)
