# LOOM - Lowering Optimizer with Optimized Matching

[![Status](https://img.shields.io/badge/status-proof%20of%20concept-yellow)](https://github.com/pulseengine/loom)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

A WebAssembly optimizer built with ISLE (Instruction Selection and Lowering Engine) from Cranelift, featuring declarative optimization rules and stateful dataflow analysis.

## Features

- **Declarative Optimization**: Write rules as pattern matches using ISLE DSL
- **Stateful Analysis**: Track local variables and memory through dataflow
- **Component Model Support**: Optimize modern WebAssembly components
- **Memory Safety**: Pure Rust implementation

## Quick Start

```bash
# Build
cargo build --release

# Optimize WebAssembly
./target/release/loom optimize input.wasm -o output.wasm

# Run tests
cargo test
```

## Current Capabilities

### Expression Optimization (40+ operations)
- Constant folding for arithmetic, bitwise, and comparison operations
- Algebraic simplifications (identity, zero, cancellation)
- Integer operations: add, sub, mul, div, rem, and, or, xor, shl, shr
- Comparisons: eq, ne, lt, le, gt, ge (signed and unsigned)

### Statement-Level Optimization
- **Local variable dataflow**: Constant propagation through local.get/set/tee
- **Memory optimization**: Redundant load elimination and store-to-load forwarding
- **Module metadata preservation**: Maintains memory sections, globals, and locals

## Architecture

```
WebAssembly → wasmparser → ISLE Terms → Optimization → wasm-encoder → Optimized WASM
                                            ↓
                                    Dataflow Analysis
                                  (locals, memory state)
```

## Example

**Input**:
```wat
(func $add (result i32)
  i32.const 10
  i32.const 32
  i32.add
)
```

**Optimized Output**:
```wat
(func $add (result i32)
  i32.const 42
)
```

## Implementation Highlights

### Novel Contribution: Stateful Term Rewriting

LOOM extends ISLE's pure functional term rewriting with stateful dataflow analysis:

```rust
pub struct OptimizationEnv {
    pub locals: HashMap<u32, Value>,
    pub memory: HashMap<MemoryLocation, Value>,
}

pub fn simplify_with_env(val: Value, env: &mut OptimizationEnv) -> Value {
    // Track assignments and propagate constants
}
```

This enables optimizations that require tracking program state across multiple instructions.

## Testing

```bash
# Unit tests
cargo test

# Optimize a benchmark
./target/release/loom optimize tests/fixtures/bench_memory.wat -o /tmp/optimized.wasm

# Validate output
wasm-tools validate /tmp/optimized.wasm
```

## Project Structure

```
loom/
├── loom-core/        # Parser, encoder, optimizer
├── loom-isle/        # ISLE term definitions and rules
├── loom-cli/         # Command-line interface
└── tests/            # Test fixtures and benchmarks
```

## Documentation

- [Requirements](docs/source/requirements/index.rst) - Comprehensive requirements using sphinx-needs
- [ISLE Language Reference](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/isle/docs/language-reference.md)
- [WebAssembly Specification](https://webassembly.github.io/spec/)

Build Sphinx documentation:
```bash
cd docs
pip install -r requirements.txt
make html
open build/html/index.html
```

## Related Projects

- [Binaryen](https://github.com/WebAssembly/binaryen) - Reference WebAssembly optimizer
- [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift) - Source of ISLE DSL
- [WRT](https://github.com/pulseengine/wrt) - Component Model runtime

## License

Apache License 2.0

## Contributing

This is a proof-of-concept project. Contributions and feedback are welcome!

1. Check the [requirements](docs/source/requirements/index.rst) for areas to work on
2. All code changes should pass `cargo test` and `cargo clippy`
3. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
