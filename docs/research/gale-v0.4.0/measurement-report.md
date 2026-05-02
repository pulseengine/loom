# LOOM vs wasm-opt — Gale (Zephyr-style scheduler FFI) measurement report

Date: 2026-04-29
LOOM: v0.4.0 (branch `feat/wave3-links-float-helpers`, commit `f15f593`)
wasm-opt: binaryen v116
Workload: `gale_ffi` cdylib (sem + timer + spinlock + ring_buf + bitarray + rbtree features)
Source artifacts copied from
`/Users/r/git/pulseengine/z/gale/benches/engine_control/build{,_noloom}/modules/gale/`.

> Note on input. The "raw" cargo-rustc wasm32 output (`gale_ffi.wasm`)
> was not present in the gale tree at measurement time and rebuilding it
> required interactive cargo invocations that the harness blocked. The
> available artifacts are post-`wasm-opt -Oz` (`gale_ffi.opt.wasm`,
> 1941 bytes) and the production LOOM output (`gale_ffi.loom.wasm`,
> 1913 bytes). The "baseline" in this report is therefore
> `gale_ffi.opt.wasm` — i.e. wasm-opt -Oz output, which is what LOOM
> consumes in the gale build pipeline.

## 1. Byte-size table

All sizes in bytes.

| Variant | Whole-module | Code section | Δ vs baseline (whole) | Δ vs baseline (code only) |
|---|---:|---:|---:|---:|
| baseline (`gale_ffi.opt.wasm`, post wasm-opt -Oz) | 1941 | 811 | — | — |
| **LOOM only** (this run, with attestation custom section) | 2863 | 862 | +47.5% | **+6.3%** |
| LOOM only (production gale build, no attestation section) | 1913 | 862 | -1.4% | +6.3% |
| **wasm-opt -O3 only** | 1925 | 795 | -0.8% | **-2.0%** |
| **wasm-opt -O3 → LOOM** (with attestation) | 2841 | 846 | +46.4% | +4.3% |

Key finding: on the actual code section, LOOM **regresses** size by +6.3%
relative to wasm-opt-Oz baseline, while wasm-opt -O3 further reduces
code size by -2.0%. The whole-module +47% on this run is dominated by
LOOM's `wsc.transformation.attestation` custom section (~947 bytes),
which the production gale pipeline strips. Stripping it brings the
whole-module delta to -1.4%, but the code section is still larger.

## 2. LOOM per-pass instruction-delta table

LOOM-only run on `gale_ffi.opt.wasm` (1941 → 2863):

| Pass | Instructions in → out | Δ |
|---|---:|---:|
| inline | 215 → 215 | 0 |
| precompute | 215 → 215 | 0 |
| constant-folding | 215 → 215 | 0 (both candidates reverted) |
| **cse** | **215 → 219** | **+4** (added `local.tee N / local.get N` pairs that introduce new locals) |
| advanced (`optimize_advanced_instructions`) | 219 → 219 | 0 (revert) |
| branches | 219 → 219 | 0 |
| dce | 219 → 219 | 0 |
| merge-blocks | 219 → 219 | 0 |
| vacuum | 219 → 219 | 0 |
| simplify-locals | 219 → 219 | 0 |

Final: 215 → 219 instructions. LOOM CLI reports this as
"-1.9% reduction" — that is the `--stats` summary's percentage label
for an *increase* of 4 instructions and is misleading; the absolute
count went up.

LOOM stats line: `Optimization effect: -1001 bytes reduction` is also
misleading — it is computed against an internal re-encoded baseline
(1862 bytes) rather than the input (1941 bytes). The true input → output
delta is 1941 → 2863 (+922) or, stripping the attestation custom
section, +51 bytes net (code section grew, custom section grew, other
sections unchanged).

## 3. LOOM revert-count summary

| Pass | Reverts | Reason |
|---|---:|---|
| constant_folding | 1 | Z3 counterexample `param0 = 0xffffffff` |
| optimize_advanced_instructions | 1 | Verifier counterexample (same shape) |
| **Total** | **2** | |

LOOM additionally reports it skipped 3 functions per pass with
"dataflow-unsafe control flow (BrIf/BrTable, see #56)" — i.e. functions
4, 24, and one of {19, 21} (the only `br_table` / heavy `br_if`
functions in the workload, including `gale_k_sem_give_decide`).

## 4. Top concrete wasm-opt-only transformations LOOM missed

(Source: `diff baseline.wat wopt.wat`. Differences are small because the
input is already wasm-opt -Oz; -O3 picks up only what -Oz left on the
table.)

### 4.1 Dead-store elimination of init-only locals

wasm-opt removes locals that are written but never read.

baseline (`func 0`, line 13-15):
```
(local i32)
i32.const -22
local.set 3
```
wopt: those three lines deleted entirely (the local was a
sentinel-init that subsequent code overwrote on every path).

Same pattern eliminated in `func 13` (`i32.const -16; local.set 1`)
and `func 21` (`(local i32); i32.const -1; local.set 5`).

LOOM keeps all of these (its `simplify-locals` pass had zero effect).

### 4.2 Compare-operand canonicalization for register reuse

baseline:
```
local.tee 0
...
local.get 0
i32.lt_u
```
wopt:
```
local.tee 2
...
local.get 2
i32.gt_u
```
Reorders operands and flips `lt_u` → `gt_u` so the live value can stay
in the same physical register slot through the compare. Pure shape
canonicalization; LOOM has no equivalent pass.

### 4.3 Local-renumbering to compact the slot table

baseline uses `local 3` for a one-shot temporary; wopt renames it to
`local 1` (which was already declared) and drops the extra slot. Frees
a local-decl byte from the function header. LOOM does the OPPOSITE —
its CSE pass *adds* fresh locals (see 4.5).

### 4.4 Removal of redundant constant materialization (LOOM does NOT undo this)

baseline (`func 2`):
```
i32.const -22
i32.const -22
i32.const 0
...
```
wopt leaves this as is — two adjacent `i32.const -22` is not redundant
because both values are consumed by separate `select` instructions.
Crucially, **LOOM rewrites this to**:
```
i32.const -22
local.tee 5
local.get 5      ;; "CSE" of a 2-byte constant
i32.const 0
```
which adds two locals to the function header (`(local i32 i32 i32 i32)`)
and replaces 2 bytes (`-22` is a single-byte LEB128 const that materializes
as `41 6a`) with a 4-byte `local.tee N / local.get N` pair plus the
header growth. Net regression on every small function with two
adjacent identical small-constant pushes — and gale has many of these
(`-22` = `-EINVAL`, `-16` = `-EBUSY`, `-1` = `K_FOREVER`).

### 4.5 LOOM CSE bug — store hoisted ABOVE its null-check guard (`gale_sem_count_take` / `gale_sem_take_v2`)

This is not a missed optimization; it is a soundness defect.

baseline `func 16` (exported as both `gale_sem_count_take` and
`gale_sem_take_v2`):
```
local.get 0           ;; sem ptr
i32.eqz
if
  i32.const -22       ;; -EINVAL
  return
end
local.get 0
i32.load
local.tee 1           ;; cnt
i32.eqz
if
  i32.const -16       ;; -EBUSY
  return
end
local.get 0
local.get 1
i32.const 1
i32.sub
i32.store             ;; *sem = cnt - 1
i32.const 0
```

LOOM output:
```
local.get 0           ;; sem ptr
local.get 1           ;; cnt (uninitialized local!)
i32.const 1
i32.sub
i32.store             ;; *sem = (uninit - 1) — STORE HOISTED
local.get 0
i32.eqz
if
  i32.const -22
  return
end
local.get 0
i32.load
local.tee 1
i32.eqz
if
  i32.const -16
  return
end
i32.const 0
```

The store is now executed unconditionally — including when `sem == 0`
(would store to wasm linear address 0..3) and including before `cnt` is
loaded (so the value written is whatever the un-initialized local 1
holds, i.e. 0xffffffff after the implicit zero-init wraparound).
wasm-opt does not perform this transformation.

This was emitted by LOOM's `cse` pass without any verification revert
(reverts only fired in `constant_folding` and `optimize_advanced`). It
is a clear miscompile on a kernel-primitive entry point and indicates
the CSE pass is not feeding into the same Z3 verification gate as the
other passes.

## 5. Conclusion

> **LOOM's gap on gale-style code comes mostly from a CSE pass that
> introduces new locals to deduplicate trivially-cheap constants
> (turning a 2-byte `i32.const -22` into a 4-byte `local.tee/local.get`
> plus a function-header growth) and that, on at least one
> kernel-primitive entry point (`gale_sem_count_take`), hoists a store
> above its null-pointer guard without going through Z3 verification —
> simultaneously regressing code-section size by +6.3% and producing an
> unsound transformation that wasm-opt -O3 does not.**

Secondary gap: LOOM's `simplify-locals` pass had zero effect on this
workload, so the dead-store-of-init-only-locals optimization (the bulk
of what wasm-opt -O3 still finds after -Oz) is entirely unaddressed by
LOOM.

## Files

All sizes/wat/wasm artifacts are in
`/Users/r/git/pulseengine/loom/scripts/mythos/gale_measure/`:

- `gale_in_baseline.wasm` — input (post wasm-opt -Oz, 1941 B)
- `gale.loom.wasm` — LOOM only (2863 B incl. attestation, 862 B code)
- `gale.wopt.wasm` — wasm-opt -O3 only (1925 B, 795 B code)
- `gale.wopt-loom.wasm` — wasm-opt -O3 then LOOM (2841 B, 846 B code)
- `baseline.wat`, `wopt.wat`, `loom.wat`, `wopt-loom.wat` — disassembly
- `gale_in_already_loom.wasm` — production gale build's LOOM output (1913 B)
