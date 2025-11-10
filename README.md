# LOOM - Lowering Optimizer with Optimized Matching

[![Status](https://img.shields.io/badge/status-proof%20of%20concept-yellow)](https://github.com/pulseengine/loom)
[![Verification](https://img.shields.io/badge/verification-SMT%20based-blue)](https://dl.acm.org/doi/10.1145/3617232.3624862)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

**A formally verified WebAssembly optimizer using ISLE with SMT-based proof system**

LOOM is a proof-of-concept WebAssembly optimizer that combines:
- **ISLE DSL** from Cranelift for declarative optimization rules
- **Crocus** SMT solver integration for formal verification
- **Binaryen compatibility** for testing and benchmarking
- **Component Model** support for modern WebAssembly

## Why LOOM?

Traditional optimizers like wasm-opt are implemented in imperative C++ with manual pattern matching. LOOM takes a different approach:

1. **Declarative** - Write optimization rules as pattern matches, not imperative code
2. **Verified** - Every rule can be formally proven correct using SMT solvers
3. **Rust** - Memory-safe implementation with modern tooling
4. **Proof-Ready** - Built from the ground up for formal verification

## Project Status

🚧 **Proof of Concept** - This project is in early development.

See [Requirements Documentation](docs/source/requirements/index.rst) for detailed status.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    WebAssembly Module                   │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                 wasmparser (Parse)                       │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                LOOM IR (ISLE Terms)                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           Optimization Pipeline (ISLE Rules)             │
│  ┌───────────────────────────────────────────────────┐  │
│  │ - Dead Code Elimination                           │  │
│  │ - Constant Folding                                │  │
│  │ - Inlining                                        │  │
│  │ - Local Optimizations                             │  │
│  │ - Memory Optimizations                            │  │
│  │ - ... (100+ optimization passes)                  │  │
│  └───────────────────────────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│         Crocus Verification (SMT Solver)                 │
│         ✓ Proves each rule sound                         │
│         ✗ Generates counterexamples                      │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              wasm-encoder (Emit)                         │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│                Optimized WebAssembly                     │
└─────────────────────────────────────────────────────────┘
```

## Key Features

### Formal Verification

Every optimization rule can be verified:

```isle
;; Constant folding for addition
(spec (iadd x y)
    (provide (= result (bvadd x y))))

(rule (lower (iadd (iconst x) (iconst y)))
      (iconst (bvadd x y)))
```

Crocus automatically verifies this rule across all bitwidths (8, 16, 32, 64) and generates counterexamples if unsound.

### Declarative Optimization Rules

Instead of imperative C++:

```cpp
// Traditional approach
if (node->type == IADD) {
  if (node->left->type == ICONST && node->right->type == ICONST) {
    return create_const(node->left->value + node->right->value);
  }
}
```

Write declarative ISLE:

```isle
;; LOOM approach
(rule (lower (iadd (iconst x) (iconst y)))
      (iconst (iadd_imm x y)))
```

### Binaryen Test Suite Integration

LOOM can run Binaryen's extensive test suite to ensure compatibility and correctness.

### Component Model Support

Optimize modern WebAssembly components with WIT interfaces and canonical ABI.

## Requirements

See [docs/source/requirements/](docs/source/requirements/) for comprehensive requirements including:

- [Core Infrastructure](docs/source/requirements/core.rst) - ISLE integration, parsers, pipeline
- [Verification](docs/source/requirements/verification.rst) - SMT-based formal verification
- [Testing](docs/source/requirements/testing.rst) - Test harness and Binaryen integration
- [Optimizations](docs/source/requirements/index.rst#requirements-by-priority) - 100+ optimization passes
- [Component Model](docs/source/requirements/component_model.rst) - Component-level optimizations

## Documentation

Full documentation is built with Sphinx and sphinx-needs:

```bash
cd docs
pip install -r requirements.txt
make html
open build/html/index.html
```

## Comparison with wasm-opt

| Feature | wasm-opt (Binaryen) | LOOM |
|---------|---------------------|------|
| Language | C++ | Rust |
| Rules | Imperative | Declarative (ISLE) |
| Verification | Manual testing | Formal (SMT-based) |
| Passes | 134 | 100+ (planned) |
| Component Model | Partial | Full support (planned) |
| Status | Production | Proof of Concept |

## Related Projects

- [Binaryen](https://github.com/WebAssembly/binaryen) - The reference WebAssembly optimizer
- [Cranelift](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift) - Source of ISLE DSL
- [WRT](https://github.com/pulseengine/wrt) - Component Model runtime

## References

- [ISLE Language Reference](https://github.com/bytecodealliance/wasmtime/blob/main/cranelift/isle/docs/language-reference.md)
- [Crocus Verification Tool](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift/isle/veri)
- [ASPLOS 2024 Paper](https://dl.acm.org/doi/10.1145/3617232.3624862) - "Lightweight, Modular Verification for WebAssembly-to-Native Instruction Selection"

## License

Apache License 2.0

## Contributing

This is a proof-of-concept project. Contributions and feedback are welcome!

1. Check the [requirements](docs/source/requirements/index.rst) for areas to work on
2. See planned optimizations and their priorities
3. All optimization rules should include formal specifications
4. Run verification tests before submitting

## Contact

PulseEngine - https://github.com/pulseengine
