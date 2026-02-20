<div align="center">

# Loom

<sup>Formally verified WebAssembly optimizer</sup>

&nbsp;

![Rust](https://img.shields.io/badge/Rust-CE422B?style=flat-square&logo=rust&logoColor=white&labelColor=1a1b27)
![WebAssembly](https://img.shields.io/badge/WebAssembly-654FF0?style=flat-square&logo=webassembly&logoColor=white&labelColor=1a1b27)
![Formally Verified](https://img.shields.io/badge/Formally_Verified-00C853?style=flat-square&logoColor=white&labelColor=1a1b27)
![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue?style=flat-square&labelColor=1a1b27)

&nbsp;

<h6>
  <a href="https://github.com/pulseengine/meld">Meld</a>
  &middot;
  <a href="https://github.com/pulseengine/loom">Loom</a>
  &middot;
  <a href="https://github.com/pulseengine/synth">Synth</a>
  &middot;
  <a href="https://github.com/pulseengine/kiln">Kiln</a>
  &middot;
  <a href="https://github.com/pulseengine/sigil">Sigil</a>
</h6>

</div>

&nbsp;

Meld fuses. Loom weaves. Synth transpiles. Kiln fires. Sigil seals.

Twelve-pass WebAssembly optimization pipeline built on Cranelift's ISLE pattern-matching engine. Constant folding, strength reduction, CSE, inlining, dead code elimination — each pass proven correct through Z3 SMT translation validation. Includes a fused mode purpose-built for Meld output.

Loom consistently achieves 80-95% binary size reduction with 10-30 microsecond optimization times. The entire pipeline is pure Rust with minimal dependencies.

## Quick Start

```bash
# Build from source
git clone https://github.com/pulseengine/loom
cd loom
cargo build --release

# Optimize a WebAssembly module
loom optimize input.wasm -o output.wasm

# With statistics
loom optimize input.wasm -o output.wasm --stats

# With Z3 verification
loom optimize input.wasm -o output.wasm --verify
```

## Architecture

- **loom-core** — Optimization pipeline implementation
- **loom-shared** — Core ISLE definitions and WebAssembly IR (stable API)
- **loom-cli** — Command-line interface
- **loom-testing** — Differential testing framework

The 12-phase pipeline: Precompute, ISLE Constant Folding, Strength Reduction, CSE, Function Inlining, ISLE (Post-inline), Code Folding, LICM, Branch Simplification, Dead Code Elimination, Block Merging, Vacuum and Simplify Locals.

See [docs/architecture.md](docs/architecture.md) for the full pipeline design.

## Features

- **Constant folding** — Compile-time evaluation with cross-function propagation via inlining
- **Strength reduction** — Replace expensive ops with cheaper equivalents (`x * 8` becomes `x << 3`)
- **Common subexpression elimination** — Deduplicate redundant computations
- **Function inlining** — Inline small functions to expose cross-function optimizations
- **Dead code elimination** — Remove unreachable code and unused locals
- **Loop-invariant code motion** — Hoist invariant expressions out of loops
- **Component Model support** — Modern WebAssembly with wasm32-wasip2 build target
- **Stateful dataflow analysis** — Track locals and memory state across passes
- **Idempotent passes** — Safe to run multiple times without degradation

## Formal Verification

Loom supports two verification modes:

**Property-Based (Always Available)**
```bash
loom optimize input.wasm -o output.wasm --verify
```
Fast idempotence checks and constant folding validation with ~5ms overhead.

**Z3 SMT Formal Proof (Optional)**
```bash
cargo build --release --features verification
loom optimize input.wasm -o output.wasm --verify
```
Proves mathematically that optimizations preserve program semantics via translation validation. See [docs/guides/formal-verification.md](docs/guides/formal-verification.md) for details.

> [!NOTE]
> **Cross-cutting verification** &mdash; Rocq mechanized proofs, Kani bounded model checking, Z3 SMT verification, and Verus Rust verification are used across the PulseEngine toolchain. Sigil attestation chains bind it all together.

## Documentation

- [Usage Guide](docs/guides/usage.md) — Complete CLI reference and best practices
- [Quick Reference](docs/guides/quick-reference.md) — Cheat sheet for common tasks
- [Architecture](docs/architecture.md) — Deep dive into the 12-phase pipeline
- [Formal Verification](docs/guides/formal-verification.md) — Z3 SMT verification internals
- [WASM Build](docs/guides/wasm-build.md) — Building Loom to WebAssembly
- [Fused Component Optimization](docs/design/fused-component-optimization.md) — Meld-Loom pipeline
- [Contributing](CONTRIBUTING.md) — How to contribute

## Building

```bash
cargo build --release                            # Standard build
cargo build --release --features verification    # With Z3 verification
cargo test                                       # Run tests
cargo bench                                      # Run benchmarks
bazel build //loom-cli:loom_wasm --platforms=@rules_rust//rust/platform:wasm  # WASM build
```

## License

Apache-2.0

---

<div align="center">

<sub>Part of <a href="https://github.com/pulseengine">PulseEngine</a> &mdash; formally verified WebAssembly toolchain for safety-critical systems</sub>

</div>
