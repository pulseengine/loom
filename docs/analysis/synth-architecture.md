# Loom Shared Architecture

## Executive Summary

Loom implements a modular architecture with a shared foundation (`loom-shared`) that can be used by multiple WebAssembly optimization tools. This design:

- Establishes a single source of truth for optimization semantics
- Enables reuse of core ISLE definitions across different tool variants
- Provides stable API for building safety-critical optimization tools
- Supports both open-source development and commercial applications

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                      loom-shared                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  ISLE Optimization Definitions (wasm_terms.isle)    │   │
│  │  - Value term system (I32Add, I64Mul, etc.)        │   │
│  │  - Type definitions (ValueType, BlockType)         │   │
│  │  - Optimization semantics (pattern matching)       │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  Core IR & Utilities (Rust):                               │
│  - Module/Function/Instruction types                       │
│  - WASM parsing & encoding                                 │
│  - Term conversion (Instruction ↔ ISLE Value)              │
│  - (Optional) Z3 verification infrastructure               │
└─────────────────────────────────────────────────────────────┘
         ▲                                    ▲
         │                                    │
         │                                    │
┌────────┴──────────┐              ┌─────────┴──────────┐
│   Loom            │              │  Other Tools       │
│   (Open Source)   │              │  (Various)         │
│   Apache 2.0      │              │                    │
├───────────────────┤              ├────────────────────┤
│ loom-core:        │              │ Can build custom   │
│ - 12-phase        │              │ optimization tools │
│   pipeline        │              │ on loom-shared     │
│ - Z3 verification │              │ foundation         │
│ - Rapid           │              │                    │
│   prototyping     │              │                    │
│                   │              │                    │
│ loom-cli:         │              │                    │
│ - Developer tool  │              │                    │
│ - Benchmarking    │              │                    │
│ - Debugging       │              │                    │
│                   │              │                    │
│ loom-testing:     │              │                    │
│ - Differential    │              │                    │
│   testing         │              │                    │
└───────────────────┘              └────────────────────┘
```

## Component Breakdown

### loom-shared (Foundation Layer)

**Purpose**: Single source of truth for optimization semantics that can be shared across multiple tools.

**Contents**:

#### 1. ISLE Definitions (`isle/wasm_terms.isle`)
```isle
;; Core type system
(type Imm32 (primitive Imm32))
(type Imm64 (primitive Imm64))
(type ValueType (enum I32 I64 F32 F64))
(type BlockType (enum Empty I32Result I64Result))

;; Value term representation
(type ValueData (enum
    (I32Const (val Imm32))
    (I32Add (lhs Value) (rhs Value))
    (I32Sub (lhs Value) (rhs Value))
    ;; ... 60+ instruction variants
))

;; Optimization entry point
(decl simplify (Value) Value)
;; Pattern matching rules defined here
```

**Key Properties**:
- Language-agnostic optimization semantics
- Composable term rewriting rules
- Extensible for new optimizations
- Direct 1:1 mapping to WebAssembly spec

#### 2. Core IR (`src/lib.rs`)
```rust
pub struct Module { functions, memories, globals, types, exports }
pub struct Function { name, signature, locals, instructions }
pub enum Instruction { I32Const, I32Add, Block, Loop, If, ... }
pub enum ValueType { I32, I64, F32, F64 }
pub enum BlockType { Empty, Value(ValueType), Func {...} }
```

**Design Principles**:
- Faithful WebAssembly representation
- Preserves all metadata for reconstruction
- Supports both binary and text format round-tripping

#### 3. Generated ISLE Context (build.rs + generated code)
```rust
// Generated by cranelift-isle compiler
pub struct IsleContext { ... }
impl IsleContext {
    pub fn iconst32(&mut self, val: Imm32) -> Value;
    pub fn iadd32(&mut self, lhs: Value, rhs: Value) -> Value;
    pub fn simplify(&mut self, val: Value) -> Value;
    // ... all ISLE constructors/extractors
}
```

**Versioning**: Follows Semantic Versioning. Breaking changes require major version bump.

**License**: Apache 2.0 (permissive for both open-source and commercial use)

---

### Loom (Open Source Reference Implementation)

**Target Use Cases**:
- Academic research
- Embedded systems development
- Rapid prototyping of new optimizations
- Community-driven bug discovery
- Reference implementation for other tools

**What's in Loom**:

#### loom-core
```rust
// 12-phase optimization pipeline
pub mod optimize {
    pub fn precompute(module: &mut Module) -> Result<()>;
    pub fn constant_folding(module: &mut Module) -> Result<()>;
    pub fn optimize_advanced_instructions(module: &mut Module) -> Result<()>;
    pub fn eliminate_common_subexpressions(module: &mut Module) -> Result<()>;
    pub fn simplify_branches(module: &mut Module) -> Result<()>;
    pub fn eliminate_dead_code(module: &mut Module) -> Result<()>;
    pub fn merge_blocks(module: &mut Module) -> Result<()>;
    pub fn vacuum(module: &mut Module) -> Result<()>;
    pub fn simplify_locals(module: &mut Module) -> Result<()>;
    // ... additional phases
}
```

**Verification Strategy**:
- Z3 SMT-based property checking
- Differential testing against wasm-opt
- Property-based testing with proptest
- Extensive test suite

**Development Model**:
- Fast iteration cycles
- Community contributions welcomed
- Public issue tracker
- Experimental features enabled by default

#### loom-cli
```bash
loom optimize input.wasm -o output.wasm
loom verify input.wasm output.wasm  # Z3 equivalence checking
loom benchmark input.wasm           # Performance metrics
```

---

## Dependency Graph

```
loom-shared  (Apache 2.0, public)
    ↓
    ├──→ loom-isle     (Apache 2.0, re-exports loom-shared)
    │        ↓
    ├──→ loom-core     (Apache 2.0, 12-phase pipeline)
    │        ↓
    ├──→ loom-cli      (Apache 2.0, developer tool)
    │
    └──→ other tools can depend on loom-shared
```

---

## API Stability Guarantees

### loom-shared (SemVer 1.0+)

**Stable API**:
```rust
// Module IR - stable since 1.0
pub struct Module { ... }
pub struct Function { ... }
pub enum Instruction { ... }

// ISLE types - stable since 1.0
pub struct Value(Box<ValueData>);
pub enum ValueData { ... }

// Parsing/encoding - stable since 1.0
pub fn parse_wasm(bytes: &[u8]) -> Result<Module>;
pub fn encode_wasm(module: &Module) -> Result<Vec<u8>>;
```

**Versioning Policy**:
- Major version (X.0.0): Breaking changes to IR or ISLE types
- Minor version (1.X.0): New instructions, new optimizations (backward compatible)
- Patch version (1.0.X): Bug fixes only

**Deprecation Policy**:
- 6 month deprecation period for any breaking change
- Deprecated APIs marked with `#[deprecated]` attribute
- Migration guide provided in CHANGELOG

### loom-core (Unstable API)

```rust
// Optimization pipeline - NO stability guarantees
pub mod optimize { ... }
```

**Rationale**: Loom is a research platform. Optimization passes may be reordered, removed, or replaced without notice.

---

## Testing Strategy

### Loom Test Suite

```bash
# Unit tests (1500+ tests)
cargo test --workspace

# Property-based testing
cargo test --features proptest

# Z3 verification (optional, requires z3)
cargo test --features verification

# Differential testing
cd loom-testing
cargo run --bin differential -- corpus/

# Benchmarks
cargo bench
```

**Coverage Target**: 90%+ line coverage for loom-shared

### CI Pipeline

```yaml
# .github/workflows/test.yml
name: Test Loom

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - name: Build loom-shared
        run: cargo build -p loom-shared

      - name: Run all tests
        run: cargo test --workspace

      - name: Z3 verification (if available)
        run: cargo test --features verification

      - name: Check no breaking changes
        run: cargo semver-checks check-release -p loom-shared
```

---

## Workflow: Adding a New Optimization

### Phase 1: Implementation (Week 1)

```rust
// loom-shared/isle/wasm_terms.isle
;; New optimization: (x * 0) → 0
(rule (simplify (I32Mul lhs (I32Const 0))) (I32Const 0))
(rule (simplify (I32Mul (I32Const 0) rhs)) (I32Const 0))
```

**Actions**:
- Add ISLE rule to loom-shared
- Write property tests
- Z3 verification checks
- Benchmark on real-world WASM

**Validation**:
```bash
cargo test -p loom-shared  # Unit tests
loom verify test.wasm      # Z3 SMT checking
loom benchmark corpus/     # Performance regression
```

### Phase 2: Review & Merge (Week 2)

**Review Process**:
1. Submit pull request
2. Automated CI tests
3. Code review by maintainers
4. Integration testing
5. Merge to main

### Phase 3: Release

**Versioning**:
- Bug fix → Patch version (1.0.X)
- New optimization → Minor version (1.X.0)
- Breaking change → Major version (X.0.0)

---

## Using loom-shared in Other Tools

### Example: Building a Custom Optimizer

```toml
# my-wasm-tool/Cargo.toml
[dependencies]
loom-shared = "1.0"
```

```rust
// my-wasm-tool/src/main.rs
use loom_shared::{parse_wasm, encode_wasm, IsleContext, Module};

fn my_custom_optimizer(wasm: &[u8]) -> Vec<u8> {
    // Parse using loom-shared
    let mut module = parse_wasm(wasm).unwrap();

    // Apply custom transformations
    my_optimization(&mut module);

    // Use ISLE for constant folding
    let mut isle = IsleContext::new();
    for func in &mut module.functions {
        constant_fold_with_isle(func, &mut isle);
    }

    // Encode back to WASM
    encode_wasm(&module).unwrap()
}
```

**Benefits**:
- Reuse battle-tested parsing/encoding
- Access to ISLE optimization framework
- Automatic updates via Cargo

---

## Licensing Strategy

### loom-shared: Apache 2.0

**Rationale**:
- Permissive for commercial use
- No copyleft restrictions
- Patent grant included
- Encourages adoption

### All Loom Components: Apache 2.0

All Loom crates are open-source under Apache 2.0.

---

## Community Strategy

### Open Community

**Contribution Model**:
- GitHub-based development
- Public RFC process for major changes
- Community-maintained optimization catalog
- Academic partnerships encouraged

**Communication Channels**:
- GitHub Discussions
- GitHub Issues
- Documentation and examples

---

## Roadmap

### Current: Foundation Establishment
- ✅ Create loom-shared crate
- ✅ Migrate ISLE definitions
- ✅ Update dependency graph
- 🔄 CI validation pipeline
- 📅 Documentation and examples

### Future: Stabilization
- Freeze loom-shared 1.0 API
- Comprehensive test suite
- Z3 verification for all ISLE rules
- Performance optimization

### Long-term: Ecosystem Growth
- Community-contributed optimizations
- Additional tool integrations
- Research collaborations
- Extended WebAssembly features

---

## Frequently Asked Questions

### Why split loom-shared from loom-core?

**Answer**: This separation allows other tools to reuse the core ISLE definitions and IR types without pulling in Loom's specific optimization pipeline. It establishes a stable foundation that multiple tools can depend on.

### Can I use loom-shared in a commercial product?

**Answer**: Yes! Apache 2.0 license allows commercial use without restrictions. You can build commercial tools on loom-shared without contributing back.

### How do I contribute to Loom?

**Answer**:
1. Fork the repository
2. Add your optimization to `loom-shared/isle/wasm_terms.isle`
3. Write tests
4. Submit a pull request
5. Review by maintainers

### Why ISLE instead of pure Rust?

**Answer**: ISLE provides:
- Declarative pattern matching (easier to verify)
- Automatic optimization rule composition
- Formal semantics
- Less boilerplate than manual Rust matching

### How often does loom-shared break backward compatibility?

**Answer**: Target is <1 major version per year. We maintain 6-month deprecation cycles for breaking changes.

---

## Conclusion

The Loom shared architecture provides:

1. **Modularity**: Clean separation between shared foundation and specific implementations
2. **Reusability**: Other tools can build on loom-shared
3. **Stability**: Semantic versioning with clear API guarantees
4. **Community**: Open-source development model
5. **Quality**: Battle-tested through extensive verification

---

## References

- [WebAssembly Specification](https://webassembly.github.io/spec/)
- [ISLE Language Documentation](https://github.com/bytecodealliance/wasmtime/tree/main/cranelift/isle)
- [Cranelift Compiler](https://cranelift.dev/)
- [Z3 SMT Solver](https://github.com/Z3Prover/z3)

---

**Document Version**: 1.0
**Last Updated**: 2025-11-20
**Authors**: PulseEngine Team
**Status**: Active
