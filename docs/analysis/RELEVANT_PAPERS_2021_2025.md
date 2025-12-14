# Relevant Research Papers (2021-2025)

**Compiled**: 2025-12-09
**Purpose**: Comprehensive list of papers relevant to LOOM's verification and optimization work
**Conferences Covered**: PLDI, OOPSLA, CAV, CC, CGO, POPL + related venues (2021-2025)

---

## Table of Contents

1. [Compiler Verification & Translation Validation](#1-compiler-verification--translation-validation)
2. [E-Graphs & Equality Saturation](#2-e-graphs--equality-saturation)
3. [WebAssembly Research](#3-webassembly-research)
4. [SMT Solvers & Bitvector Reasoning](#4-smt-solvers--bitvector-reasoning)
5. [Compiler Testing & Fuzzing](#5-compiler-testing--fuzzing)
6. [Program Synthesis & Superoptimization](#6-program-synthesis--superoptimization)
7. [Peephole Optimization & Rewrite Rules](#7-peephole-optimization--rewrite-rules)
8. [MLIR & Compiler Infrastructure](#8-mlir--compiler-infrastructure)
9. [Proof Assistants & Mechanized Verification](#9-proof-assistants--mechanized-verification)
10. [Dataflow Analysis & Abstract Interpretation](#10-dataflow-analysis--abstract-interpretation)
11. [Machine Learning for Compilers](#11-machine-learning-for-compilers)

---

## 1. Compiler Verification & Translation Validation

### PLDI 2021

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Alive2: Bounded Translation Validation for LLVM](https://pldi21.sigplan.org/details/pldi-2021-papers/5/Alive2-Bounded-Translation-Validation-for-LLVM)** | Nuno P. Lopes, Juneyoung Lee, Chung-Kil Hur, Zhengyang Liu, John Regehr | SMT-based verification of LLVM IR transformations; found 47 bugs in LLVM | ⭐⭐⭐ Direct inspiration for LOOM's Z3 approach |
| **[Boosting SMT Solver Performance on Mixed-Bitwise-Arithmetic Expressions](https://pldi21.sigplan.org/details/pldi-2021-papers/43/Boosting-SMT-Solver-Performance-on-Mixed-Bitwise-Arithmetic-Expressions)** | Various | Techniques to handle MBA obfuscation in SMT solving | ⭐⭐ Could improve Z3 performance |

### PLDI 2022

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Islaris: Verification of Machine Code Against Authoritative ISA Semantics](https://pldi22.sigplan.org/details/pldi-2022-pldi/9/Islaris-Verification-of-Machine-Code-Against-Authoritative-ISA-Semantics)** | Various | SMT + symbolic execution for machine code verification | ⭐⭐ Techniques applicable to WASM verification |
| **[End-to-End Translation Validation for the Halide Language](https://dl.acm.org/doi/abs/10.1145/3527328)** | Various | Translation validation for DSL compilers using SMT | ⭐⭐ Similar approach to LOOM |

### PLDI 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Much Still to Do in Compiler Verification (CakeML Perspective)](https://pldi24.sigplan.org/details/pldi-2024-papers/100/Much-Still-to-Do-in-Compiler-Verification-A-Perspective-from-the-CakeML-Project-)** | CakeML Team | State-of-the-art in end-to-end verified compilers | ⭐⭐ Research directions |
| **[A Verified Compiler for a Functional Tensor Language](https://pldi24.sigplan.org/details/pldi-2024-papers/14/A-Verified-Compiler-for-a-Functional-Tensor-Language)** | Various | Verified tensor compiler | ⭐ Verification techniques |
| **[Towards Trustworthy Automated Program Verifiers](https://dl.acm.org/doi/10.1145/3656438)** | Various | Formally validating translations to IVLs like Boogie | ⭐⭐ Verification pipeline design |

### CAV 2023

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Kratos2: SMT-Based Model Checker for Imperative Programs](https://www.i-cav.org/2023/accepted-papers/)** | Alberto Griggio, Martin Jonáš | SMT-based verification for imperative programs | ⭐⭐ Verification techniques |
| **[Automated Verification of Correctness for Masked Arithmetic Programs](https://www.i-cav.org/2023/accepted-papers/)** | Mingyang Liu, Fu Song, Taolue Chen | Verifying arithmetic transformations | ⭐⭐ Arithmetic verification |

### CAV 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[The Top-Down Solver Verified](https://i-cav.org/2024/accepted-papers/)** | Yannick Stade, Sarah Tilscher, Helmut Seidl | Verified static analyzer implementation | ⭐⭐ Verification methodology |
| **[Algebraic Reasoning Meets Automata in Solving Linear Integer Arithmetic](https://i-cav.org/2024/accepted-papers/)** | Peter Habermehl et al. | Novel SMT solving techniques | ⭐ SMT improvements |

### OOPSLA 2021

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Formal Verification of High-Level Synthesis](https://pldi22.sigplan.org/details/pldi-2022-sigplan-track/58/-OOPSLA-2021-Formal-verification-of-high-level-synthesis)** | Various | First mechanically verified HLS tool | ⭐ Verification techniques |

---

## 2. E-Graphs & Equality Saturation

### PLDI 2023

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Better Together: Unifying Datalog and Equality Saturation](https://pldi23.sigplan.org/details/pldi-2023-pldi/20/Better-Together-Unifying-Datalog-and-Equality-Saturation)** | Yihong Zhang, Yisu Remy Wang, Oliver Flatt, David Cao, Philip Zucker, Eli Rosenthal, Zachary Tatlock, Max Willsey | egglog: combines Datalog with e-graphs | ⭐⭐⭐ Potential LOOM integration |

### EGRAPHS Workshop @ PLDI 2023

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[ægraphs: Acyclic E-graphs for Efficient Optimization in a Production Compiler](https://pldi23.sigplan.org/details/egraphs-2023-papers/2/-graphs-Acyclic-E-graphs-for-Efficient-Optimization-in-a-Production-Compiler)** | Cranelift Team | Acyclic e-graphs for predictable compile times; 16% speedup on SpiderMonkey.wasm | ⭐⭐⭐ Directly applicable to LOOM |
| **[Ensuring the Termination of Equality Saturation](https://pldi23.sigplan.org/details/egraphs-2023-papers/9/Ensuring-the-termination-of-equality-saturation-for-terminating-term-rewriting-system)** | Various | Termination guarantees for equality saturation | ⭐⭐ Important for production use |
| **[Optimizing Beta-Reduction in E-Graphs](https://pldi23.sigplan.org/details/egraphs-2023-papers/12/Optimizing-Beta-Reduction-in-E-Graphs)** | Various | Efficient lambda handling in e-graphs | ⭐ Advanced e-graph techniques |
| **[Automating Constraint-Aware Datapath Optimization using E-Graphs](https://pldi23.sigplan.org/details/egraphs-2023-papers/4/Automating-Constraint-Aware-Datapath-Optimization-using-E-Graphs)** | Various | E-graphs + abstract interpretation | ⭐⭐ Combines analysis with rewriting |

### OOPSLA 2021

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Rewrite Rule Inference Using Equality Saturation (Ruler)](https://dl.acm.org/doi/10.1145/3485496)** | Chandrakana Nandi, Max Willsey, Amy Zhu, et al. | **Distinguished Paper** - Automatically synthesize rewrite rules | ⭐⭐⭐ Could discover new LOOM optimizations |

### POPL 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Guided Equality Saturation](https://popl24.sigplan.org/details/POPL-2024-popl-research-papers/60/Guided-Equality-Saturation)** | Thomas Koehler, Andrés Goens, et al. | Human-guided sketches for scalable equality saturation | ⭐⭐⭐ Scales e-graphs to production |

### CGO 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Latent Idiom Recognition Using Equality Saturation](https://dl.acm.org/doi/abs/10.1109/CGO57630.2024.10444879)** | Various | Pattern recognition via e-graphs | ⭐⭐ Optimization discovery |

---

## 3. WebAssembly Research

### PLDI 2023

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Iris-Wasm: Robust and Modular Verification of WebAssembly Programs](https://pldi23.sigplan.org/details/pldi-2023-pldi/46/Iris-Wasm-Robust-and-Modular-Verification-of-WebAssembly-Programs)** | Various | Higher-order separation logic for Wasm in Coq | ⭐⭐⭐ Formal semantics reference |

### PLDI 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Bringing the WebAssembly Standard up to Speed with SpecTec](https://pldi24.sigplan.org/details/pldi-2024-papers/64/Bringing-the-WebAssembly-Standard-up-to-Speed-with-SpecTec)** | Various | DSL for Wasm specification; generates Coq definitions | ⭐⭐⭐ Official Wasm semantics |

### PLDI 2025 (Upcoming)

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Verification of WebAssembly Features](https://pldi25.sigplan.org/details/rpls-2025-papers/9/Verification-of-WebAssembly-Features)** | Various | Verification of new Wasm features | ⭐⭐⭐ Directly relevant |

### POPL 2018 (Foundational)

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Mechanising and Verifying the WebAssembly Specification](https://dl.acm.org/doi/10.1145/3167082)** | Conrad Watt | First mechanized Wasm spec in Isabelle | ⭐⭐⭐ Foundational reference |

### FM 2021

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Two Mechanisations of WebAssembly 1.0](https://link.springer.com/chapter/10.1007/978-3-030-90870-6_4)** | Various | WasmCert-Isabelle and WasmCert-Coq | ⭐⭐⭐ Reference implementations |

### IMC 2021

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Understanding the Performance of WebAssembly Applications](https://weihang-wang.github.io/papers/imc21.pdf)** | Yutian Yan et al. | Analysis of Wasm compilation and optimization | ⭐⭐ Performance insights |

### OOPSLA 2022

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[A Fast In-Place Interpreter for WebAssembly](https://dl.acm.org/doi/10.1145/3563311)** | Various | Efficient Wasm interpretation | ⭐ Runtime techniques |

---

## 4. SMT Solvers & Bitvector Reasoning

### CAV 2023

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Bitwuzla](https://www.i-cav.org/2023/accepted-papers/)** | Aina Niemetz, Mathias Preiner | **Distinguished Paper** - SMT solver 2-3× faster than Z3 on bitvectors | ⭐⭐⭐ Could replace/augment Z3 |
| **[The Golem Horn Solver](https://www.i-cav.org/2023/accepted-papers/)** | Martin Blicha et al. | **Distinguished Paper** - Horn clause solving | ⭐⭐ Verification infrastructure |

### CAV 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[SMT-based Symbolic Model-Checking for Operator Precedence Languages](https://i-cav.org/2024/accepted-papers/)** | Michele Chiari et al. | Novel SMT applications | ⭐ SMT techniques |
| **[Distributed SMT Solving Based on Dynamic Variable-level Partitioning](https://i-cav.org/2024/accepted-papers/)** | Mengyu Zhao et al. | Parallel SMT solving | ⭐ Performance improvements |
| **[Efficient Implementation of an Abstract Domain of Quantified First-Order Formulas](https://i-cav.org/2024/accepted-papers/)** | Eden Frenkel et al. | Abstract interpretation + SMT | ⭐⭐ Analysis techniques |

### CAV 2022

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[SMT-Based Translation Validation for Machine Learning Compiler](https://link.springer.com/chapter/10.1007/978-3-031-13188-2_19)** | Bang, Nam, Chun, Jhoo, Lee | Translation validation for ML compilers | ⭐⭐ Validation techniques |

### ITP 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Verifying Peephole Rewriting in SSA Compiler IRs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2024.9)** | Siddharth Bhat et al. | Verified peephole optimization for SSA IRs | ⭐⭐⭐ Directly applicable |

---

## 5. Compiler Testing & Fuzzing

### OOPSLA 2019 (Foundational)

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Compiler Fuzzing: How Much Does It Matter?](https://dl.acm.org/doi/10.1145/3360581)** | Various | Impact study of fuzzer-found compiler bugs | ⭐⭐ Testing strategy |

### Empirical Software Engineering 2022

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[CsmithEdge: More Effective Compiler Testing](https://link.springer.com/article/10.1007/s10664-022-10146-1)** | Various | Extended Csmith; found 7 new bugs | ⭐⭐ Fuzzing techniques |

### PLDI 2014 (Foundational EMI)

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Compiler Validation via Equivalence Modulo Inputs](https://dl.acm.org/doi/10.1145/2594291.2594334)** | Vu Le et al. | EMI testing; found 147 bugs in GCC/LLVM | ⭐⭐⭐ Should implement for LOOM |

### PLDI 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Boosting Compiler Testing by Injecting Real-world Code](https://dl.acm.org/do/10.5281/zenodo.10951313/full/)** | Various | Improved compiler fuzzing | ⭐⭐ Testing methodology |

### CGO 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[High-Throughput, Formal-Methods-Assisted Fuzzing for LLVM](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | Various | Combines fuzzing with formal methods | ⭐⭐⭐ Hybrid approach |
| **[Compiler Testing with Relaxed Memory Models](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | Various | Testing under weak memory | ⭐ Concurrency testing |

### OOPSLA 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Rustlantis: Randomized Differential Testing of the Rust Compiler](https://2024.splashcon.org/track/splash-2024-oopsla)** | Various | Rust compiler fuzzing | ⭐ Fuzzing techniques |

---

## 6. Program Synthesis & Superoptimization

### POPL 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Optimal Program Synthesis via Abstract Interpretation](https://popl24.sigplan.org/details/POPL-2024-popl-research-papers/18/Optimal-Program-Synthesis-via-Abstract-Interpretation)** | Stephen Mell, Steve Zdancewic, Osbert Bastani | Provably optimal synthesis | ⭐⭐ Synthesis techniques |
| **[Efficient Bottom-Up Synthesis for Programs with Local Variables](https://popl24.sigplan.org/details/POPL-2024-popl-research-papers/18/Optimal-Program-Synthesis-via-Abstract-Interpretation)** | Xiang Li et al. | Lifted interpretation for synthesis | ⭐⭐ Synthesis algorithms |

### OOPSLA 2021

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Gauss: Program Synthesis by Reasoning over Graphs](https://dl.acm.org/doi/10.1145/3485511)** | Various | Graph-based synthesis | ⭐ Synthesis techniques |

### OOPSLA 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Minotaur: A SIMD-Oriented Synthesizing Superoptimizer](https://dl.acm.org/doi/10.1145/3689766)** | Various | SIMD superoptimization; 7.3% speedup on GMP | ⭐⭐ Superoptimization |

### Souper (2017, ongoing)

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Souper: A Synthesizing Superoptimizer](https://arxiv.org/abs/1711.04422)** | Various | Synthesizing superoptimizer for LLVM; 4.4% smaller binaries | ⭐⭐ Superoptimization techniques |

---

## 7. Peephole Optimization & Rewrite Rules

### OOPSLA 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Hydra: Generalizing Peephole Optimizations with Program Synthesis](https://2024.splashcon.org/details/splash-2024-oopsla/27/Hydra-Generalizing-Peephole-Optimizations-with-Program-Synthesis)** | Manasij Mukherjee, John Regehr | Automatic generalization of peephole opts; generalizes 75% of LLVM missed opts | ⭐⭐⭐ Could generate LOOM rules |

### ITP 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Verifying Peephole Rewriting in SSA Compiler IRs](https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.ITP.2024.9)** | Siddharth Bhat et al. | Verified peephole for MLIR-style IRs | ⭐⭐⭐ Verification methodology |

---

## 8. MLIR & Compiler Infrastructure

### CGO 2021

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[MLIR: A Compiler Infrastructure for the End of Moore's Law](https://dl.acm.org/doi/10.1109/CGO51591.2021.9370308)** | Chris Lattner et al. | Multi-level IR infrastructure | ⭐⭐ IR design patterns |

### CGO 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Experiences Building an MLIR-Based SYCL Compiler](https://dl.acm.org/doi/10.1109/CGO57630.2024.10444866)** | E. Tiotto et al. | MLIR dialect design for SYCL | ⭐ Compiler architecture |

### PLDI 2025

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[First-Class Verification Dialects for MLIR](https://users.cs.utah.edu/~regehr/papers/pldi25.pdf)** | Various | Formal semantics as MLIR dialects; found 5 bugs | ⭐⭐⭐ Verification integration |

### arXiv 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[The MLIR Transform Dialect](https://arxiv.org/abs/2409.03864)** | Various | Controllable IR transformations | ⭐⭐ Transformation control |

### OOPSLA 2025

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[DESIL: Detecting Silent Bugs in MLIR](https://arxiv.org/html/2504.01379v1)** | Various | Fuzzing for MLIR correctness | ⭐⭐ Testing methodology |

---

## 9. Proof Assistants & Mechanized Verification

### Various 2023-2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Lean4Lean: Towards a Verified Typechecker for Lean](https://arxiv.org/html/2403.14064v2)** | Mario Carneiro | Self-verified theorem prover | ⭐ Verification infrastructure |
| **[Compiling Lean Programs with Rocq's Extraction Pipeline](https://www.normalesup.org/~sdima/2025_extraction_report.pdf)** | Various | Verified extraction from Lean to OCaml | ⭐ Verified compilation |

### CompCert (Foundational)

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Formal Verification of a Realistic Compiler](https://xavierleroy.org/publi/compcert-CACM.pdf)** | Xavier Leroy | Fully verified C compiler in Coq | ⭐⭐ Gold standard reference |

---

## 10. Dataflow Analysis & Abstract Interpretation

### SPLASH 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Automatically Generating an Abstract Interpretation-Based Optimizer from a DSL](https://dl.acm.org/doi/10.1145/3689491.3689968)** | Various | Generate analyzers from DSL; in CPython 3.13 | ⭐⭐ Analysis generation |

### NeurIPS 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[LLMDFA: Analyzing Dataflow in Code with Large Language Models](https://chengpeng-wang.github.io/publications/LLMDFA_NeurIPS2024.pdf)** | Various | LLM-based dataflow analysis | ⭐ Novel analysis approach |

### ICSE 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Dataflow Analysis-Inspired Deep Learning for Vulnerability Detection](https://conf.researchr.org/home/icse-2024)** | Various | ML + dataflow for security | ⭐ Analysis techniques |

---

## 11. Machine Learning for Compilers

### CGO 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Distinguished: Revealing Compiler Heuristics through Automated Discovery](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | V. Seeker et al. | **Distinguished Paper** - ML discovers compiler heuristics | ⭐⭐ Optimization discovery |
| **[SLaDe: A Portable Small Language Model Decompiler](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | J. Armengol-Estapé et al. | ML for decompilation | ⭐ ML techniques |
| **[AskIt: Unified Programming Interface for LLMs](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | Various | LLM integration in compilers | ⭐ Novel approaches |

### CC 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[The Next 700 ML-Enabled Compiler Optimizations](https://conf.researchr.org/info/CC-2024/accepted-papers)** | Various | Framework for ML in compilers | ⭐⭐ ML integration |

---

## CC (Compiler Construction) Papers 2021-2024

### CC 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[A Context-Sensitive Pointer Analysis Framework for Rust](https://conf.researchr.org/info/CC-2024/accepted-papers)** | Various | Rust static analysis | ⭐ Analysis for Rust |
| **[Clog: A Declarative Language for C Static Code Checkers](https://conf.researchr.org/info/CC-2024/accepted-papers)** | Various | DSL for static checkers | ⭐⭐ DSL design |
| **[Fast Template-Based Code Generation for MLIR](https://conf.researchr.org/info/CC-2024/accepted-papers)** | Various | Efficient MLIR codegen | ⭐ Code generation |
| **[From Low-Level Fault Modeling to Proven Hardening Scheme](https://conf.researchr.org/info/CC-2024/accepted-papers)** | Various | Verified security hardening | ⭐⭐ Security verification |

---

## CGO Papers 2021-2024

### CGO 2024

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Distinguished: oneDNN Graph Compiler](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | Various | **Distinguished Paper** - High-performance DL compilation | ⭐ Optimization techniques |
| **[Distinguished: JITSPMM](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | Q. Fu et al. | **Distinguished Paper** - JIT for sparse matrix ops | ⭐ JIT techniques |
| **[A System-Level Dynamic Binary Translator](https://conf.researchr.org/info/cgo-2024/accepted-papers)** | Various | Learned translation rules | ⭐⭐ Rule learning |

### CGO 2022

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[CompilerGym: Robust Compiler Optimization Environments for AI](https://conf.researchr.org/track/cgo-2022/cgo-2022-main-conference)** | Various | ML environment for compiler opts | ⭐ ML infrastructure |
| **[A Compiler for Sound Floating-Point Computations using Affine Arithmetic](https://conf.researchr.org/track/cgo-2022/cgo-2022-main-conference)** | Various | Verified floating-point | ⭐⭐ Float verification |
| **[Loop Rolling for Code Size Reduction](https://conf.researchr.org/track/cgo-2022/cgo-2022-main-conference)** | Various | Code size optimization | ⭐⭐ Size optimization |

---

## Summary Statistics

| Conference | Years | Total Relevant Papers |
|------------|-------|----------------------|
| PLDI | 2021-2024 | ~15 |
| OOPSLA | 2021-2024 | ~12 |
| CAV | 2021-2024 | ~15 |
| CC | 2021-2024 | ~8 |
| CGO | 2021-2024 | ~15 |
| POPL | 2023-2024 | ~6 |
| Other (EGRAPHS, ITP, etc.) | 2021-2024 | ~10 |
| **Total** | | **~80+ papers** |

---

## Top Priority Papers for LOOM

### Must Read (⭐⭐⭐)

1. **Alive2** (PLDI 2021) - Direct inspiration for LOOM's approach
2. **Bitwuzla** (CAV 2023) - Potential Z3 replacement/augment
3. **Ruler** (OOPSLA 2021) - Automatic rule discovery
4. **ægraphs** (EGRAPHS 2023) - Production e-graphs for Wasm
5. **Guided Equality Saturation** (POPL 2024) - Scalable e-graphs
6. **Iris-Wasm** (PLDI 2023) - Wasm formal semantics
7. **SpecTec** (PLDI 2024) - Official Wasm specification DSL
8. **Hydra** (OOPSLA 2024) - Peephole generalization
9. **EMI Testing** (PLDI 2014) - Most effective miscompilation detection
10. **First-Class Verification Dialects** (PLDI 2025) - Verification integration

### High Priority (⭐⭐)

11. **egglog** (PLDI 2023) - Datalog + e-graphs
12. **Verifying Peephole Rewriting** (ITP 2024) - SSA IR verification
13. **CsmithEdge** (ESE 2022) - Advanced fuzzing
14. **Minotaur** (OOPSLA 2024) - SIMD superoptimization
15. **MLIR Transform Dialect** (2024) - Controllable transformations

---

## Research Gaps LOOM Could Fill

Based on this survey, LOOM could contribute novel research in:

1. **First verified WebAssembly optimizer** - No existing verified Wasm optimizer
2. **Component Model optimization** - No formal work on WASM Component Model
3. **Stack analysis with SMT** - Novel combination of Binaryen-style analysis + Z3
4. **Automatic rule synthesis for Wasm** - Apply Ruler to Wasm semantics
5. **EMI testing for Wasm** - First EMI implementation for WebAssembly

---

## 2025 Papers (Latest Research)

### PLDI 2025

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[First-Class Verification Dialects for MLIR](https://users.cs.utah.edu/~regehr/papers/pldi25.pdf)** | Various | Formal semantics as MLIR dialects; found 5 miscompilation bugs in MLIR | ⭐⭐⭐ Verification integration pattern |
| **[Type-Constrained Code Generation with Language Models](https://pldi25.sigplan.org/details/pldi-2025-papers/25/Type-Constrained-Code-Generation-with-Language-Models)** | Various | Type-guided LLM code generation | ⭐ Novel synthesis approach |
| **[Program Synthesis From Partial Traces](https://pldi25.sigplan.org/details/pldi-2025-papers/67/Program-Synthesis-From-Partial-Traces)** | Various | Synthesize programs from side-effect traces | ⭐⭐ Synthesis techniques |
| **[Explode.js: Security Analysis for Node.js](https://www.cylab.cmu.edu/news/2025/06/17-cylab-presents-at-pldi-2025.html)** | CMU CyLab | Exploit synthesis via static analysis + symbolic execution | ⭐ Security verification |

### RPLS 2025 Workshop @ PLDI

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Verification of WebAssembly Features](https://pldi25.sigplan.org/details/rpls-2025-papers/9/Verification-of-WebAssembly-Features)** | Various | Formal verification of new Wasm features using SpecTec | ⭐⭐⭐ Directly applicable |

### EGRAPHS 2025 Workshop @ PLDI

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Equality Saturation Guided by Large Language Models (LGuess)](https://pldi25.sigplan.org/details/egraphs-2025-papers/12/Equality-Saturation-Guided-by-Large-Language-Models)** | Various | LLM-guided e-graph rewriting | ⭐⭐ Novel guidance approach |
| **[Machine Learning Guided Equality Saturation](https://pldi25.sigplan.org/details/egraphs-2025-papers/6/Machine-Learning-Guided-Equality-Saturation)** | Various | ML models to auto-generate guides for equality saturation | ⭐⭐⭐ Scalable e-graphs |
| **[Incremental Equality Saturation](https://pldi25.sigplan.org/details/egraphs-2025-papers/4/Incremental-Equality-Saturation)** | Various | Efficient temporal e-graph optimization | ⭐⭐ Performance improvement |
| **[eqsat: An Equality Saturation Dialect for MLIR](https://pldi25.sigplan.org/details/egraphs-2025-papers/3/eqsat-An-Equality-Saturation-Dialect-for-Non-destructive-Rewriting)** | Various | Native e-graph representation in MLIR | ⭐⭐⭐ IR integration pattern |
| **[Semantic Foundations of Equality Saturation](https://egraphs.org/meeting/2025-05-15-semantics)** | Various | Fixpoint semantics via tree automata; connections to chase | ⭐⭐ Theoretical foundations |
| **[DialEgg: Dialect-Agnostic MLIR Optimizer with Egglog](https://egraphs.org/meeting/2025-08-21-dialegg)** | Various | Egglog integration with MLIR | ⭐⭐⭐ Production e-graphs |

### POPL 2025

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[A Primal-Dual Perspective on Program Verification Algorithms](https://popl25.sigplan.org/details/POPL-2025-popl-research-papers/68/A-Primal-Dual-Perspective-on-Program-Verification-Algorithms)** | Various | Unified framework for verification algorithms using duality | ⭐⭐ Theoretical foundations |
| **[SYNVER: LLM-Based Program Synthesis with Verification](https://popl25.sigplan.org/details/CoqPL-2025-papers/5/Towards-Automated-Verification-of-LLM-Synthesized-C-Programs)** | Various | LLM synthesis + automated verification | ⭐⭐ Synthesis + verification |

### CAV 2025

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Integer Reasoning Modulo Different Constants in SMT](https://conferences.i-cav.org/2025/accepted/)** | Pertseva, Ozdemir, Pailoor, et al. | Novel SMT techniques for integer reasoning | ⭐⭐ SMT improvements |
| **[Lean-SMT: An SMT Tactic for Lean](https://conferences.i-cav.org/2025/accepted/)** | Mohamed, Mascarenhas, Khan, et al. | SMT integration in Lean 4 proof assistant | ⭐⭐ Verification tooling |
| **[MCSat Heuristics for Nonlinear Arithmetic](https://conferences.i-cav.org/2025/accepted/)** | Various | Decision heuristics for Yices2's MCSat | ⭐ SMT solver improvements |
| **[Improving SMT Performance for Program Verification](https://conferences.i-cav.org/2025/accepted/)** | Various | Reduce proof search space for SMT-based verifiers | ⭐⭐⭐ Directly applicable |
| **[A Formally Verified Robustness Certifier for Neural Networks](https://conferences.i-cav.org/2025/accepted/)** | Tobler, Syeda, Murray | Verified neural network certifier | ⭐ Verification techniques |
| **[Automated Verification of Consistency in Zero-Knowledge Proof Circuits](https://conferences.i-cav.org/2025/accepted/)** | Stephens, Pailoor, Dillig | ZK circuit verification | ⭐ Verification applications |

### CGO 2025

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Scalar Interpolation for Vectorized Loops](https://2025.cgo.org/track/cgo-2025-papers)** | Various | Insert scalar ops in vector loops; 30% speedup on x86 | ⭐ Optimization technique |
| **[Transform Dialect in MLIR](https://2025.cgo.org/track/cgo-2025-papers)** | Various | Fine-grained compiler control via IR transformations | ⭐⭐ Transformation control |
| **[Synthesis of Sorting Kernels](https://2025.cgo.org/track/cgo-2025-papers)** | Various | New lower bounds + synthesis for optimal sorting | ⭐ Synthesis techniques |
| **[GoFree: Compiler-inserted Memory Freeing](https://2025.cgo.org/details/cgo-2025-papers/10/GoFree-Reducing-Garbage-Collection-via-Compiler-inserted-Freeing)** | Various | Reduce GC overhead via escape analysis | ⭐ Memory optimization |
| **[Multi-Level Compiler Backend for RISC-V](https://2025.cgo.org/details/cgo-2025-papers/39/A-Multi-Level-Compiler-Backend-for-Accelerated-Micro-Kernels-Targeting-RISC-V-ISA-Ext)** | Various | MLIR-based backend for RISC-V extensions | ⭐ Backend design |

### OOPSLA 2025

| Paper | Authors | Key Contribution | Relevance to LOOM |
|-------|---------|------------------|-------------------|
| **[Compiling Classical Sequent Calculus to Stock Hardware](https://se.cs.uni-tuebingen.de/2025/02/23/two-papers-at-oopsla/)** | Uni Tübingen | Duality-based compilation approach | ⭐ Compiler theory |
| **[Monomorphization with Higher-Rank and Existential Types](https://se.cs.uni-tuebingen.de/2025/02/23/two-papers-at-oopsla/)** | Uni Tübingen | Type-based flow analysis for monomorphization | ⭐⭐ Type system techniques |
| **[DESIL: Detecting Silent Bugs in MLIR](https://arxiv.org/html/2504.01379v1)** | Various | Fuzzing for MLIR silent bugs | ⭐⭐ Testing methodology |

---

## Updated Summary Statistics

| Conference | Years | Total Relevant Papers |
|------------|-------|----------------------|
| PLDI | 2021-2025 | ~20 |
| OOPSLA | 2021-2025 | ~15 |
| CAV | 2021-2025 | ~20 |
| CC | 2021-2025 | ~10 |
| CGO | 2021-2025 | ~20 |
| POPL | 2023-2025 | ~10 |
| EGRAPHS | 2022-2025 | ~15 |
| Other (ITP, etc.) | 2021-2025 | ~10 |
| **Total** | | **~120+ papers** |

---

## References by URL

### Conference Proceedings
- PLDI: https://conf.researchr.org/series/pldi
- OOPSLA: https://conf.researchr.org/series/splash
- CAV: https://i-cav.org/
- CC: https://conf.researchr.org/series/CC
- CGO: https://conf.researchr.org/series/cgo
- POPL: https://conf.researchr.org/series/POPL
- EGRAPHS: https://pldi24.sigplan.org/home/egraphs-2024

### Key Tools
- Alive2: https://github.com/AliveToolkit/alive2
- Bitwuzla: https://bitwuzla.github.io/
- egg/egglog: https://github.com/egraphs-good/egglog
- Ruler: https://github.com/uwplse/ruler
- Rosette: https://emina.github.io/rosette/
- Csmith: https://embed.cs.utah.edu/csmith/
- wasm-smith: https://github.com/bytecodealliance/wasm-tools
