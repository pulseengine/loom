# WarpL (arXiv 2604.13693) — Evaluation for PulseEngine CI

**Verdict: Adopt later.** WarpL solves a problem PulseEngine does not yet have — diagnosing the *root cause* of an already-observed runtime perf regression *inside the JIT* — not detecting that a regression occurred. As a CI signal it is the wrong shape: it costs ~3 hours per case after you already know there is a slowdown, and the bug-class it isolates (Wasmtime/Cranelift instruction-lowering pathologies) is upstream of loom/meld output. Defer until we have (a) a stable wasmtime-based perf baseline in CI and (b) a confirmed regression class plausibly attributable to a loom/meld output pattern.

## 1. Paper summary

WarpL (Zeng et al., ICSE '26, arXiv:2604.13693v1, 15 Apr 2026) is a **mutation-based root-cause localizer** for Wasm-runtime performance bugs, not a regression detector. Given a bug-inducing Wasm module already known to be slow on runtime R_buggy:

- **Mutation**. Fine-grained, type-aware, single-instruction edits (Table 1): operand-instruction substitution (`t.const`/`local.get`/`global.get`/`t.load` interchange within the same type), operator-instruction substitution (replace `t.op` with another op of the same `optype`), operator-instruction deletion (delete op + its operand-producers, replace stack effect with a const). Control-flow instructions are explicitly excluded. ~500 mutants generated per case, deduped to ~350.
- **Functionally-similar selection**. Each mutant is run on R_buggy and on an **oracle runtime** R_oracle (a second Wasm engine known not to exhibit the bug). Score = `α·perfDiffScore + β·funcSimScore` where `perfDiffScore` rewards a large execution-time gap on R_buggy and `funcSimScore` rewards near-identical execution time on R_oracle (Algorithm 1). Mutants whose behavior is preserved on R_oracle but whose slowdown disappears on R_buggy are the candidates.
- **Slow-code isolation**. Dump JIT machine code from R_buggy for original vs selected mutant. Diff via Longest Common Subsequence at the x86-64/aarch64 opcode level. The differing instruction window is the "slow code" report.
- **Perf signature**. Single scalar: wall-clock execution time on each runtime. Plus a machine-instruction count from the JIT dump. No syscall histograms, no perf-event traces, no cycle counters.
- **Eval**. 12 issues across Wasmtime / Wasmer / WasmEdge; localized 10/12, found 6 previously-unknown Wasmtime bugs. **Wall time: <3 h per case for WarpL, plus ~11 h per case for `wasm-reduce` pre-shrinking.** Tool is ~400 LOC C++ (Binaryen LibTooling) + ~500 LOC Python/shell. Open source: https://github.com/BZTesting/WarpL.

**Key assumptions / limitations** (Section 6): (1) you already know which module is slow — WarpL needs a labeled bug-inducing input; (2) you need a second Wasm runtime that does *not* trigger the issue, to act as oracle; (3) the bug must live in JIT lowering or IR optimization, not in host I/O or libc (the two failures #7973 and #7745 were both I/O-bound, outside JIT scope); (4) `wasm-reduce` dominates wall time and is the bottleneck the authors flag for future work.

## 2. PulseEngine corpus survey

Confirmed in `/Users/r/git/pulseengine/loom/tests/corpus/`:

| Fixture | Path | Role |
|---|---|---|
| httparse | `tests/corpus/httparse.wasm` | HTTP/1.x header parser (witness real-app fixture) |
| nom_numbers | `tests/corpus/nom_numbers.wasm` | nom parser-combinator numeric primitives |
| state_machine | `tests/corpus/state_machine.wasm` | finite-state-machine kernel (kiln test) |
| json_lite | `tests/corpus/json_lite.wasm` | minimal JSON tokenizer |
| loom (self) | `tests/corpus/loom.wasm` | LOOM compiled to Wasm — dogfood target |

Sibling projects `/Users/r/git/pulseengine/witness` and `/Users/r/git/pulseengine/meld` exist on disk but were outside the read-permission scope of this session; the loom-side mirror at `tests/corpus/` already holds the witness-derived fixtures the user named, so the five above are representative without cross-repo access. `loom-testing/Cargo.toml` already depends on `wasmtime = "17.0"` — the runtime WarpL primarily targets.

## 3. Feasibility

- **On `loom optimize` output**: Yes, mechanically. WarpL is a black-box harness over a Wasm binary; loom's optimized output is a valid module. But WarpL's value is conditional on a *known* slowdown on a runtime. Loom currently validates via diff vs wasm-opt (size, semantics), not runtime wall-clock — so WarpL has no trigger.
- **On `meld fuse` output**: Same answer, with extra friction. Component-Model fused outputs would need `wasm-reduce` adapted for components (today it operates on core modules) — non-trivial.
- **Runtime to capture signatures**: WarpL needs **two** runtimes (buggy + oracle). Wasmtime is already a dep; adding WasmEdge or Wasmer to CI is feasible (both have static binaries) but doubles container size and the wall-clock baseline must be stable enough that a 1.5×–8× gap (the gaps WarpL reported) is detectable above noise — hard on shared GitHub runners.
- **Integration cost**. WarpL itself: ~900 LOC upstream + a Rust harness wrapping it (~300 LOC) + a perf-baseline DB (artifact-store JSON) for the trigger. Per-case CI cost is **~14 h** dominated by `wasm-reduce` — incompatible with PR-blocking CI; only viable as a nightly/manual job on a self-hosted runner.

## 4. Recommendation

**Adopt later.** WarpL is a high-quality localizer but the wrong end of the pipeline for CI: it presumes regression detection has already happened, takes hours per case, and targets bugs in the Wasm runtime's JIT — bugs we cannot fix from loom/meld even if WarpL finds them. The right CI signal for "loom emitted a Wasm whose perf changed" is a cheap wall-clock benchmark on the five fixtures above with a noise-aware threshold; WarpL becomes useful only *after* such a benchmark fires and we want to know whether the cause is in loom's transformation or in Wasmtime's lowering of it.

## 5. Implementation sketch (when triggered)

1. Add a nightly `cargo bench` harness in `loom-testing/` that runs each of the 5 fixtures through `loom optimize` then through wasmtime, recording wall-clock and machine-instruction counts (from `wasmtime compile --emit-clif`) against a committed baseline JSON.
2. Define a regression trigger: ≥1.3× wall-clock slowdown vs baseline, reproducible across 3 runs on a self-hosted runner — only then enqueue WarpL.
3. Vendor WarpL (BZTesting/WarpL) as a submodule under `tools/warpl/`; build it in a separate Docker image with WasmEdge as oracle runtime.
4. Write a Rust wrapper (`loom-testing/src/bin/warpl_localize.rs`) that takes the regressed fixture, runs `wasm-reduce` with a wall-clock-preserving predicate, then invokes WarpL and posts the slow-code report as a GitHub issue with the JIT-diff snippet.
5. Decide per-issue whether the root cause is a loom transformation (fix in loom) or a Wasmtime lowering bug (file upstream) — WarpL's report distinguishes these because the differing mutant is at Wasm-instruction granularity.
