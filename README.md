# LOOM - Formally Verified WebAssembly Optimizer

[![Status](https://img.shields.io/badge/status-active%20development-brightgreen)](https://github.com/pulseengine/loom)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-20%2F20%20passing-success)](tests/)

A high-performance WebAssembly optimizer with formal verification support. LOOM combines expression-level optimizations with Z3 SMT-based verification to ensure correctness.

## ✨ Features

### 🚀 Expression-Level Optimization Pipeline
- **Constant folding** - Compile-time evaluation of expressions
- **Strength reduction** - Replace expensive ops with cheaper ones (`x * 8` → `x << 3`)
- **Function inlining** - Inline small functions to expose cross-function optimizations
- **Stateful dataflow analysis** - Track locals and memory state across optimizations
- **Idempotent passes** - Safe to run multiple times without degradation

### 🔬 Formal Verification
- **Z3 SMT-based verification** proves optimization correctness via translation validation
- **Property-based testing** ensures idempotence and validity
- **Counterexample generation** for debugging failed optimizations
- Optional verification feature (build with `--features verification`)

### ⚡ Performance
- **Ultra-fast**: 10-30 µs optimization time for most modules
- **Excellent compression**: 80-95% binary size reduction
- **Instruction optimization**: 0-40% instruction count reduction (varies by code)
- **Lightweight**: Minimal dependencies, pure Rust implementation

### 🎯 Advanced Features
- Component Model support for modern WebAssembly
- wasm32-wasip2 build target support
- Comprehensive benchmarking with Criterion
- Full WAT and WASM format support

## 🏛️ Architecture

Loom is built with a modular architecture featuring a shared foundation:

- **loom-shared**: Core ISLE definitions and WebAssembly IR (stable API)
- **loom-core**: Optimization pipeline implementation
- **loom-cli**: Command-line interface and tooling
- **loom-testing**: Differential testing framework

The `loom-shared` crate provides a stable, reusable foundation that other WebAssembly tools can build upon. It contains:
- ISLE term definitions for all WebAssembly instructions
- Module IR (Module, Function, Instruction types)
- WASM parsing and encoding utilities
- Z3 verification infrastructure (optional)

This architecture enables both rapid prototyping in Loom and potential use in safety-critical applications through derived tools.

📖 See [LOOM_SYNTH_ARCHITECTURE.md](LOOM_SYNTH_ARCHITECTURE.md) for detailed architecture documentation.

## 📦 Quick Start

### Installation

```bash
# Build from source
git clone https://github.com/pulseengine/loom
cd loom
cargo build --release

# Binary at target/release/loom
./target/release/loom --help
```

### Basic Usage

```bash
# Optimize WebAssembly module
loom optimize input.wasm -o output.wasm

# Show detailed statistics
loom optimize input.wat -o output.wasm --stats

# Run verification checks
loom optimize input.wasm -o output.wasm --verify

# Output as WAT text format
loom optimize input.wasm -o output.wat --wat
```

### Example Output

```
🔧 LOOM Optimizer v0.1.0
Input: input.wasm
✓ Parsed in 234µs
⚡ Optimizing...
✓ Optimized in 0 ms

📊 Optimization Statistics
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Instructions: 24 → 20 (16.7% reduction)
Binary size:  797 → 92 bytes (88.5% reduction)
Constant folds: 3
Optimization time: 0 ms
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Optimization complete!
```

## 🏗️ Core Optimization Passes

| Pass | Status | Example | Impact |
|------|--------|---------|--------|
| **Constant Folding** | ✅ | `10 + 20` → `30` | Enables other opts |
| **Strength Reduction** | ✅ | `x * 8` → `x << 3` | 2-3x speedup |
| **Function Inlining** | ✅ | Inline small functions | Exposes optimizations |
| **Local Optimization** | ✅ | Dead local removal | Reduces overhead |

**Roadmap**: See [Issue #23](https://github.com/pulseengine/loom/issues/23) for planned optimizations (DCE, control flow, CSE, LICM) and timeline to wasm-opt feature parity.

## 📊 Benchmark Results

### Performance (Criterion benchmarks)

```
Constant Folding:        8-11 µs
Strength Reduction:      10 µs
CSE:                     9-14 µs
Function Inlining:       16-18 µs
Full Pipeline:           19-28 µs
Parser:                  6.8 µs
Encoder:                 183 ns (!)
```

### Real-World Results

| Fixture | Instructions | Binary Size | Constant Folds |
|---------|-------------|-------------|----------------|
| bench_bitops | 24 → 20 (16.7%) | 88.5% reduction | 0 |
| test_input | 9 → 7 (22.2%) | 81.6% reduction | 1 |
| fibonacci | 12 → 12 (0%) | 92.6% reduction | 0 |
| quicksort | Complex | 92.5% reduction | 0 |
| game_logic | Complex | 92.5% reduction | 0 |

**Key Insight**: Binary size reductions are consistently excellent (80-93%), while instruction count improvements vary by code complexity.

## 🔬 Formal Verification

LOOM supports two verification modes:

### 1. Property-Based (Always Available)
```bash
loom optimize input.wasm -o output.wasm --verify
```
- Fast idempotence checks
- Constant folding validation
- ~5ms overhead

### 2. Z3 SMT Formal Proof (Optional)
```bash
# Install Z3
brew install z3  # macOS
sudo apt install z3  # Linux

# Build with verification
cargo build --release --features verification

# Verify with formal proof
./target/release/loom optimize input.wasm -o output.wasm --verify
```

**Output:**
```
🔬 Running Z3 SMT verification...
  ✅ Z3 verification passed: optimizations are semantically equivalent
  📊 Verification coverage: 42 verified, 3 skipped (93.3%)
```

Z3 verification proves mathematically that optimizations preserve program semantics via translation validation. The coverage report shows how many functions were fully verified vs. skipped (due to unsupported patterns like complex loops). See `docs/FORMAL_VERIFICATION_GUIDE.md` for details.

## 💡 Examples

### Example 1: Strength Reduction

**Input:**
```wasm
(module
  (func $optimize_me (param $x i32) (result i32)
    local.get $x
    i32.const 8
    i32.mul
  )
)
```

**After Optimization:**
```wasm
(module
  (func $optimize_me (param $x i32) (result i32)
    local.get $x
    i32.const 3
    i32.shl  ;; 2-3x faster than multiply!
  )
)
```

### Example 2: Constant Folding

**Input:**
```wasm
(func $calculate (result i32)
  i32.const 10
  i32.const 20
  i32.add
  i32.const 5
  i32.mul
)
```

**After Optimization:**
```wasm
(func $calculate (result i32)
  i32.const 150  ;; Computed at compile-time
)
```

### Example 3: CSE

**Input:**
```wasm
(func $duplicate (param $x i32) (result i32)
  local.get $x
  i32.const 4
  i32.mul
  local.get $x
  i32.const 4
  i32.mul  ;; Duplicate computation!
  i32.add
)
```

**After Optimization:**
```wasm
(func $duplicate (param $x i32) (result i32)
  local.get $x
  i32.const 4
  i32.mul
  local.tee $temp
  local.get $temp  ;; Reuse cached result
  i32.add
)
```

## 🏛️ Architecture

```
                    ┌──────────────────┐
                    │  WebAssembly     │
                    │  (WAT or WASM)   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   wasmparser     │
                    │  Parse to AST    │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │  ISLE Terms      │
                    │  (IR)            │
                    └────────┬─────────┘
                             │
                             ▼
    ┌────────────────────────────────────────────┐
    │     12-Phase Optimization Pipeline         │
    │  ┌──────────────────────────────────────┐  │
    │  │ 1. Precompute                        │  │
    │  │ 2. ISLE Constant Folding             │  │
    │  │ 3. Strength Reduction                │  │
    │  │ 4. Common Subexpression Elimination  │  │
    │  │ 5. Function Inlining                 │  │
    │  │ 6. ISLE (Post-inline)                │  │
    │  │ 7. Code Folding                      │  │
    │  │ 8. Loop-Invariant Code Motion        │  │
    │  │ 9. Branch Simplification             │  │
    │  │ 10. Dead Code Elimination            │  │
    │  │ 11. Block Merging                    │  │
    │  │ 12. Vacuum & Simplify Locals         │  │
    │  └──────────────────────────────────────┘  │
    │              │                              │
    │              ▼                              │
    │  ┌──────────────────────────────────────┐  │
    │  │   Dataflow Analysis                  │  │
    │  │   (locals, memory state tracking)    │  │
    │  └──────────────────────────────────────┘  │
    └────────────────────────────────────────────┘
                             │
                             ▼
              ┌──────────────────────────┐
              │   Optional: Z3 SMT       │
              │   Verification           │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │   wasm-encoder           │
              │   Encode to binary       │
              └────────────┬─────────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │  Optimized WebAssembly   │
              └──────────────────────────┘
```

## 📚 Documentation

### User Guides
- **[Usage Guide](docs/USAGE_GUIDE.md)** - Complete CLI reference, examples, and best practices
- **[Quick Reference](docs/QUICK_REFERENCE.md)** - Cheat sheet for common tasks
- **[Performance Comparison](docs/PERFORMANCE_COMPARISON.md)** - LOOM vs wasm-opt benchmarks and analysis

### Technical Documentation
- **[Architecture](docs/ARCHITECTURE.md)** - Deep dive into the 12-phase pipeline and implementation
- **[Formal Verification Guide](docs/FORMAL_VERIFICATION_GUIDE.md)** - Z3 SMT verification internals
- **[WASM Build Guide](docs/WASM_BUILD.md)** - Building LOOM to WebAssembly (wasm32-wasip2)
- **[Implementation Details](docs/IMPLEMENTATION_ACHIEVEMENTS.md)** - Technical implementation notes

### For Contributors
- **[Contributing Guide](CONTRIBUTING.md)** - How to contribute to LOOM
- **[Design Documents](docs/)** - Individual optimization pass designs (CSE, DCE, LICM, etc.)

## 🧪 Testing

```bash
# Run all tests
cargo test

# Run optimization-specific tests
cargo test --test optimization_tests

# Run benchmarks
cargo bench

# Test on real fixtures
./target/release/loom optimize tests/fixtures/quicksort.wat -o /tmp/out.wasm --stats --verify
```

**Test Status:**
- ✅ 20/20 optimization tests passing (100%)
- ✅ 54/57 unit tests passing (95%)
- ✅ All benchmarks complete successfully

## 📁 Project Structure

```
loom/
├── loom-core/                # Core optimizer implementation
│   ├── src/lib.rs            # 12-phase pipeline, optimizations
│   ├── src/verify.rs         # Z3 verification module
│   ├── tests/                # Unit and integration tests
│   └── benches/              # Criterion performance benchmarks
├── loom-isle/                # ISLE term definitions and rules
│   ├── isle/                 # ISLE DSL files
│   └── src/lib.rs            # Rust integration
├── loom-cli/                 # Command-line interface
│   ├── src/main.rs           # CLI implementation
│   └── BUILD.bazel           # Bazel build rules
├── tests/fixtures/           # Real-world test cases
│   ├── fibonacci.wat
│   ├── quicksort.wat
│   ├── matrix_multiply.wat
│   └── ...
└── docs/                     # Comprehensive documentation
    ├── USAGE_GUIDE.md
    ├── QUICK_REFERENCE.md
    └── ...
```

## 🔧 Building

### Standard Build
```bash
cargo build --release
```

### With Z3 Verification
```bash
cargo build --release --features verification
```

### WASM Build (wasm32-wasip2)
```bash
# Using Cargo
cargo build --release --target wasm32-wasip2

# Using Bazel
bazel build //loom-cli:loom_wasm --platforms=@rules_rust//rust/platform:wasm
```

See [WASM_BUILD.md](docs/WASM_BUILD.md) for details.

## 🚀 Use Cases

### 1. Production Deployment
```bash
# Optimize and verify before deployment
loom optimize app.wasm -o app.optimized.wasm --verify --stats
```

### 2. CI/CD Integration
```yaml
- name: Optimize WebAssembly
  run: |
    loom optimize dist/*.wasm -o dist/*.wasm --stats --verify
```

### 3. Development Workflow
```bash
# Optimize during build
cargo build --target wasm32-unknown-unknown
loom optimize target/wasm32-unknown-unknown/release/app.wasm -o dist/app.wasm
```

### 4. Performance Analysis
```bash
# Compare before/after
ls -lh original.wasm optimized.wasm
loom optimize original.wasm -o optimized.wasm --stats
```

## 🎯 Optimization Patterns

### Strength Reduction
- `x * 2^n` → `x << n` (2-3x faster)
- `x / 2^n` → `x >> n` (2-3x faster)
- `x % 2^n` → `x & (2^n - 1)` (2-3x faster)

### Algebraic Simplification
- `x | 0` → `x`
- `x & -1` → `x`
- `x ^ 0` → `x`
- `x + 0` → `x`
- `x * 1` → `x`

### Constant Folding
- Compile-time evaluation of all constant expressions
- Propagation through local variables
- Cross-function constant propagation via inlining

## 🤝 Contributing

Contributions welcome! This project is under active development.

1. Check existing [issues](https://github.com/pulseengine/loom/issues)
2. Run tests: `cargo test && cargo clippy`
3. Follow Rust conventions
4. Add tests for new features
5. Update documentation

## 📜 License

Apache License 2.0

## 🙏 Acknowledgments

- **ISLE DSL** from [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift)
- **Z3 SMT Solver** from [Microsoft Research](https://github.com/Z3Prover/z3)
- **wasmparser & wasm-encoder** from [Bytecode Alliance](https://github.com/bytecodealliance)

## 🔗 Related Projects

- [Binaryen](https://github.com/WebAssembly/binaryen) - Reference WebAssembly optimizer (C++)
- [wasm-opt](https://github.com/WebAssembly/binaryen) - Industry-standard optimizer
- [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift) - High-performance code generator
- [WRT](https://github.com/pulseengine/wrt) - WebAssembly Component Model runtime

## 📈 Roadmap

**Current Status (v0.1.0):**
- ✅ 12-phase optimization pipeline
- ✅ Z3 formal verification
- ✅ Comprehensive benchmarking
- ✅ Component Model support
- ✅ wasm32-wasip2 build target

**Coming Soon:**
- 🚧 More aggressive LICM (arithmetic operations, global reads)
- 🚧 Profile-guided optimization
- 🚧 SIMD-specific optimizations
- 🚧 Inter-procedural analysis
- 🚧 Custom optimization passes

---

**Built with ❤️ by PulseEngine**
