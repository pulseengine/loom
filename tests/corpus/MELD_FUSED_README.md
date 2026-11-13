# tests/corpus/meld_fused.wasm — placeholder (not yet built)

> **Status: MISSING.** `scripts/measure_corpus.sh` lists `meld_fused` in its
> `WORKLOADS` array; the row reports `n/a` until this fixture is committed.

## Why this fixture is needed

The fused-module passes in `loom-core/src/fused_optimizer.rs` target shapes
that a `meld fuse` of several components produces. The existing component
fixtures (`loom-core/tests/component_fixtures/simple.component.wasm`,
`calc.component.wasm`) are independent of each other, so fusing them yields
none of these shapes and the passes measure byte-neutral.

| Pass                       | Target shape                                                   |
|----------------------------|----------------------------------------------------------------|
| `devirtualize_adapters`    | trivial adapter: `local.get 0..n-1; call target`               |
| `dedupe_function_bodies`   | two functions with identical `(signature, locals, body)`       |
| `eliminate_dead_functions` | functions unreachable from exports, start and element segments |
| `deduplicate_imports`      | two imports with the same module, name and type                |

## What to build

`tests/corpus/meld_fused.wasm`: a single core module (post-fusion) that

1. validates with `wasm-tools validate`;
2. contains at least one trivial adapter that callers reach;
3. contains two functions with identical `(signature, locals, body)`.

## How to build it

Two independent components do not work: `meld fuse` either rejects them or
produces no adapters worth removing. Use a pair where one component imports
an interface the other exports:

1. Write a small WIT interface with one function, e.g.
   `add: func(a: s32, b: s32) -> s32`.
2. Build a producer component that exports it (Rust with `cargo component`
   or `wit-bindgen`).
3. Build a consumer component that imports it and re-exports a wrapper.
4. Fuse and check:

```bash
meld fuse <producer>.component.wasm <consumer>.component.wasm \
    -o tests/corpus/meld_fused.wasm
wasm-tools validate tests/corpus/meld_fused.wasm
bash scripts/measure_corpus.sh   # the meld_fused row must not be n/a
```

Build the components locally; do not pull them from crates.io.

## Verification once the fixture lands

`loom optimize tests/corpus/meld_fused.wasm -o meld_fused.loom.wasm` must
report a non-zero count for at least one of the passes above under `--stats`,
and the optimized module must behave as the input does on its exports.
