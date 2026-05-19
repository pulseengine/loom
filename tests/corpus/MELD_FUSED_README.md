# tests/corpus/meld_fused.wasm — placeholder (not yet built)

> **Status: MISSING.** This fixture is referenced by `scripts/measure_corpus.sh`
> but has not yet been produced. This README documents exactly what is
> required so the next operator can land it without guesswork.

## Why this fixture is needed

LOOM v1.0.5 Track 3 (commit `27c0a01`, PR #132) added two passes that target
the shapes a real `meld fuse` of multiple components produces:

| Pass                          | Target shape                                                         |
|-------------------------------|----------------------------------------------------------------------|
| `inline_scalar_adapters`      | cross-memory scalar copy: `load(memA); load(memB); call; store(memB); store(memA)` |
| `dedupe_function_bodies`      | two functions with identical `(signature, locals, body)`             |

The existing single-component fixtures
(`loom-core/tests/component_fixtures/simple.component.wasm` and
`calc.component.wasm`) produce a single fused core when run through
`meld fuse`, but neither imports from the other, so the resulting fused
core contains **no cross-memory adapter** — exactly the shape Tier-1.1
targets. Track 3 therefore fired byte-neutral on the v1.0.5 measurement
corpus.

A **multi-component** meld-fused fixture is what Track 3 needs to demonstrate
its byte wins. The harness already supports it; only the fixture is missing.

## What to build

`tests/corpus/meld_fused.wasm` — a single **core module** (post-fusion)
that:

1. Validates via `wasm-tools validate`.
2. Contains at least one **cross-memory scalar adapter** (the
   `inline_scalar_adapters` target shape).
3. Optionally, contains two functions with identical
   `(signature, locals, body)` (the `dedupe_function_bodies` target).

## How to build it

### Inputs that won't work (and why)

The two existing component fixtures
(`loom-core/tests/component_fixtures/simple.component.wasm`,
`calc.component.wasm`) are **independent**: neither imports the other. They
both export only pure scalar `(i32, i32) -> i32` functions and import
nothing. `meld fuse` on two independent components either rejects the
input or produces trivial pass-through adapters with a single memory —
no cross-memory shape, no dedup target.

### Inputs that will work

A pair of components where one **imports** an interface the other
**exports**, and both perform some non-trivial body. Easiest path:

1. Author a tiny `wit` interface (one function, e.g.
   `add: func(a: s32, b: s32) -> s32`).
2. Build a producer component (Rust + `cargo-component` or `wit-bindgen`)
   that exports it.
3. Build a consumer component that imports it and re-exports a wrapped
   version.
4. `meld fuse producer.component.wasm consumer.component.wasm
   -o tests/corpus/meld_fused.wasm` (the consumer's memory and the
   producer's memory are separate → cross-memory adapter shape).

> **Constraint:** v1.0.5 Track E (this work) forbids pulling components
> from crates.io. Build local fixtures or revive the v1.0.0 PR-Q fixture
> set (some of those crates may already be in
> `tests/corpus/rust-examples/`).

### Toolchain

- `meld` v0.x (`/Users/r/.cargo/bin/meld` per the v0.9.0 measurement notes).
- `cargo component` or `wit-bindgen` for building components from Rust.
- `wasm-tools` for validation and inspection.

### Commands

```bash
# 1. Build the two components (toolchain-specific).
# 2. Fuse them into a core module.
meld fuse \
    <producer>.component.wasm \
    <consumer>.component.wasm \
    -o tests/corpus/meld_fused.wasm

# 3. Validate.
wasm-tools validate tests/corpus/meld_fused.wasm

# 4. Confirm the cross-memory adapter shape.
wasm-tools print tests/corpus/meld_fused.wasm | grep -A 20 "adapter\|i32.load\|i32.store"

# 5. Run the harness; the new row must be non-`n/a`.
bash scripts/measure_corpus.sh
```

## Harness wiring

`scripts/measure_corpus.sh` has been updated on this branch to include
`meld_fused` in the `WORKLOADS` array. The row reports `n/a` until the
fixture is produced and committed.

## Honest stop-report from this branch

This branch (`release/v1.1.0-pr-meld-fixture`) attempted to produce the
fixture but was unable to invoke `meld` due to an environment permission
restriction on the agent that authored it (Bash invocations of `meld`
were denied). The agent's choice was either to (a) ship a fake/stub wasm
that validates but doesn't contain the target shape, or (b) ship this
placeholder doc + harness wiring so the next operator can complete the
work in minutes. **Option (b) was chosen per
`CLAUDE.md`: "A correct optimizer that handles 50% of cases is infinitely
better than a fast optimizer that corrupts 1% of cases."** Shipping a
fake fixture would silently change what Track 3 measures against —
exactly the kind of "good enough" shortcut LOOM's mission forbids.

## Verification once the fixture lands

Run `loom optimize tests/corpus/meld_fused.wasm -o /tmp/meld_fused.loom.wasm`
both **with** and **without** Track 3 passes
(`inline_scalar_adapters` / `dedupe_function_bodies`) and confirm the
byte delta is strictly negative with the passes enabled. The
expected effect on a fixture with one cross-memory adapter and one
dedup target is **non-zero**, breaking the byte-neutral measurement
recorded in `docs/measurements/v0.9.0-corpus-baseline.md`.
