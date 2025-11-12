========================================
Research and Verification Related Work
========================================

This document summarizes academic research and conference papers (2020-2025) related to
formal verification of compiler optimizations, instruction selection, term rewriting systems,
and WebAssembly compilation. This research informs LOOM's verification strategy and validates
the approach of using ISLE with SMT-based verification.

Focus Period: 2020-2025 (emphasis on 2024-2025)

Key Research Papers
===================

ASPLOS 2024: Crocus - Lightweight Modular Verification
-------------------------------------------------------

.. req:: Implement Crocus-Style Verification
   :id: REQ_RESEARCH_001
   :status: planned
   :priority: Critical
   :category: Research
   :links: REQ_VERIFY_001

   Adopt techniques from "Lightweight, Modular Verification for WebAssembly-to-Native
   Instruction Selection" (ASPLOS 2024).

   **Paper Details:**

   - **Title:** Lightweight, Modular Verification for WebAssembly-to-Native Instruction Selection
   - **Venue:** ASPLOS 2024
   - **Tool:** Crocus (SMT-based ISLE verification)
   - **Artifact:** https://github.com/avanhatt/asplos24-ae-crocus
   - **DOI:** https://dl.acm.org/doi/10.1145/3617232.3624862

   **Key Contributions:**

   - SMT solver-based verification (Z3) for ISLE rules
   - Verified 98 rules with 377 type instantiations in Cranelift aarch64 backend
   - Found 7 real bugs including CVEs and sandbox escapes:

     1. x86-64 addressing mode vulnerability (9.9/10 severity)
     2. aarch64 unsigned divide CVE (sign/zero extension)
     3. aarch64 count-leading-sign bug
     4. x86-64 addressing mode defect
     5. Negated constant rules (aarch64)
     6. Imprecise constant semantics (undefined behavior)
     7. Mid-end optimization issues

   **Technical Approach:**

   - Bitvector modeling of WebAssembly values (8, 16, 32, 64, 128-bit)
   - Type instantiation across bitwidths
   - Automatic counterexample generation
   - Memory operation modeling (load_effect, store_effect)
   - Specification annotations (spec, require, provide)

   **Relevance to LOOM:**

   LOOM should adopt Crocus's verification methodology for proving correctness of
   optimization rules. This provides high confidence that transformations preserve
   semantics and don't introduce security vulnerabilities.

   **Implementation Notes:**

   - Use crocus from cranelift/isle/veri
   - Integrate into LOOM build system
   - Verify each optimization pass
   - Generate counterexamples for failed rules
   - Track verification coverage metrics

OOPSLA/PACMPL 2025: Arrival - Scaling Instruction Selection Verification
-------------------------------------------------------------------------

.. req:: Research Arrival Verification Approach
   :id: REQ_RESEARCH_002
   :status: planned
   :priority: High
   :category: Research
   :links: REQ_VERIFY_001

   Study "Scaling Instruction-Selection Verification against Authoritative ISA Semantics"
   for production compiler verification.

   **Paper Details:**

   - **Title:** Scaling Instruction-Selection Verification against Authoritative ISA Semantics
   - **Venue:** Proceedings of the ACM on Programming Languages (PACMPL), 2025
   - **Tool:** Arrival (instruction-selection verifier for Cranelift)
   - **DOI:** https://dl.acm.org/doi/10.1145/3764383

   **Key Contributions:**

   - End-to-end, high-assurance verification for production Wasm-to-native compiler
   - Reduces developer effort for verification
   - Addresses sandbox security guarantees
   - Scales to realistic industrial compiler complexity

   **Technical Challenges Addressed:**

   - Prior verification struggled with industrial compiler scale
   - Instruction selection errors can undermine sandbox guarantees
   - Correctness critical for untrusted code execution

   **Relevance to LOOM:**

   Arrival demonstrates that formal verification can scale to production compilers.
   LOOM should investigate Arrival's techniques for verifying optimization passes
   at scale while maintaining developer productivity.

   **Implementation Notes:**

   - Study Arrival's verification architecture
   - Compare with Crocus approach
   - Identify applicable techniques for optimization verification
   - Consider adoption for LOOM's verification pipeline

POPL 2025: Progressful Interpreters for WebAssembly
----------------------------------------------------

.. req:: Study WasmCert Mechanization Approaches
   :id: REQ_RESEARCH_003
   :status: planned
   :priority: Medium
   :category: Research

   Review "Progressful Interpreters for Efficient WebAssembly Mechanisation" for
   mechanized specification techniques.

   **Paper Details:**

   - **Title:** Progressful Interpreters for Efficient WebAssembly Mechanisation
   - **Venue:** POPL 2025
   - **Tool:** WasmCert-Coq (extended to WebAssembly 2.0)
   - **DOI:** https://dl.acm.org/doi/10.1145/3704858

   **Key Contributions:**

   - Mechanized WebAssembly specification in Coq
   - Progressful interpreter using dependent types
   - Certified soundness and progress properties
   - Performance optimizations fully erasable when extracted

   **Technical Approach:**

   - Uses Coq theorem prover
   - Dependent types for certified properties
   - WebAssembly 2.0 feature coverage
   - Extractable to efficient implementations

   **Relevance to LOOM:**

   Understanding mechanized semantics of WebAssembly provides foundation for
   proving LOOM's transformations preserve WebAssembly semantics. The progressful
   interpreter approach may inform LOOM's validation testing.

   **Implementation Notes:**

   - Review WasmCert-Coq mechanization
   - Consider Coq for high-assurance proofs
   - Study dependent type approaches
   - Investigate extraction to Rust

CPP 2025: CertiCoq-Wasm - Verified Compilation
-----------------------------------------------

.. req:: Study Verified Compilation Techniques
   :id: REQ_RESEARCH_004
   :status: planned
   :priority: Low
   :category: Research

   Investigate verified compilation pipeline techniques from CertiCoq-Wasm.

   **Paper Details:**

   - **Title:** CertiCoq-Wasm: A Verified WebAssembly Backend for CertiCoq
   - **Venue:** CPP 2025 (14th ACM SIGPLAN International Conference on Certified Programs and Proofs)
   - **DOI:** https://dl.acm.org/doi/10.1145/3703595.3705879

   **Key Contributions:**

   - Verified WebAssembly code generation from Coq
   - Works from minimal lambda calculus in ANF
   - Efficient implementation of primitive operations
   - Identified corner case leading to unsoundness

   **Technical Approach:**

   - ANF (Administrative Normal Form) as IR
   - Coq's primitive integers to WebAssembly instructions
   - Correctness proofs for compilation passes
   - Bug discovery through verification

   **Relevance to LOOM:**

   CertiCoq-Wasm demonstrates end-to-end verified compilation. While LOOM focuses
   on optimization rather than compilation, the techniques for proving correctness
   of transformations are applicable.

Equality Saturation and E-Graphs
=================================

PLDI 2024/2025: E-Graphs for Compiler Optimization
---------------------------------------------------

.. req:: Investigate E-Graph Integration
   :id: REQ_RESEARCH_005
   :status: planned
   :priority: Medium
   :category: Research

   Research equality saturation and e-graphs as alternative to pure term rewriting.

   **Relevant Work:**

   - EGRAPHS 2024 Workshop (PLDI 2024)
   - EGRAPHS 2025 Workshop (PLDI 2025)
   - Multiple papers on equality saturation applications

   **Key Papers (2024-2025):**

   1. **"Optimizing Tensor Computation Graphs with Equality Saturation and Monte Carlo Tree Search"**

      - Venue: PACT 2024 (International Conference on Parallel Architectures and Compilation Techniques)
      - Date: October 2024

   2. **"Equality Saturation Guided by Large Language Models"**

      - Venue: EGRAPHS 2025 (PLDI 2025 Workshop)
      - Tool: LGuess (LLM-guided e-graph rewriting)

   3. **"Contextual Equality Saturation"**

      - Venue: SAS 2025 (SPLASH 2025)
      - Approach: E-graphs with contextual reasoning for conditional branches

   4. **"Equality Saturation for Optimizing High-Level Julia IR"**

      - Date: February 2025 (arXiv)

   5. **"Algorithm-Aware Hardware Optimization using E-Graph Rewriting"**

      - Venue: EGRAPHS 2024 (PLDI 2024)
      - Tool: SEER (Super-optimization Exploration using E-graph Rewriting)

   **Technical Concepts:**

   - **Equality Saturation:** Explores exponentially large space of equivalent programs
   - **E-Graphs:** Efficiently represent multiple optimized versions simultaneously
   - **Phase-Ordering Problem:** E-graphs can saturate to represent all optimization orders
   - **Extraction:** Select optimal program from e-graph representation

   **Applications Found:**

   - Compilers
   - Floating point accuracy
   - Test generation
   - Computational fabrication
   - Automatic vectorization
   - Deep learning compute graphs
   - Symbolic computation

   **Relevance to LOOM:**

   E-graphs complement ISLE's term rewriting by solving the phase-ordering problem.
   LOOM could integrate e-graph techniques to explore optimization combinations
   more exhaustively than sequential passes.

   **Implementation Notes:**

   - Study egg library (Rust e-graphs)
   - Compare ISLE term rewriting vs e-graph saturation
   - Consider hybrid approach
   - Evaluate performance vs optimization quality tradeoffs

Term Rewriting and Confluence
==============================

Automated Strategy Invention (2024)
------------------------------------

.. req:: Study Automated Confluence Proving
   :id: REQ_RESEARCH_006
   :status: planned
   :priority: Low
   :category: Research

   Investigate automated techniques for proving rewrite rule confluence.

   **Paper Details:**

   - **Title:** Automated Strategy Invention for Confluence of Term Rewrite Systems
   - **Date:** November 2024 (arXiv:2411.06409)
   - **Approach:** AI techniques for confluence proving

   **Key Contributions:**

   - CSI confluence prover with AI-invented strategies
   - Outperforms human-designed strategies
   - Automatic proof search for term rewriting systems

   **Relevance to LOOM:**

   Confluence is critical for LOOM's optimization rules. If rules are confluent,
   the order of application doesn't matter (modulo termination). AI-assisted
   confluence proving could verify that LOOM's rule sets are well-behaved.

   **Implementation Notes:**

   - Review confluence checking for ISLE rules
   - Identify non-confluent rule interactions
   - Consider automated confluence proving tools
   - Document rule ordering constraints

Instruction Selection Synthesis
================================

FMCAD 2024: Synthesizing Rewrite Rules
---------------------------------------

.. req:: Explore Rule Synthesis Techniques
   :id: REQ_RESEARCH_007
   :status: planned
   :priority: Medium
   :category: Research

   Study automated synthesis of optimization rules.

   **Paper Details:**

   - **Title:** Efficiently Synthesizing Lowest Cost Rewrite Rules for Instruction Selection
   - **Venue:** FMCAD 2024 (Prague, Czech Republic, October 14-18, 2024)
   - **Authors:** Ross Daly, Caleb Donovick, Caleb Terrill, Jackson Melchert,
     Priyanka Raina, Clark W. Barrett, Pat Hanrahan
   - **Session:** SMT Solving and Applications

   **Key Contributions:**

   - Automatically synthesize rewrite rules for instruction selection
   - Optimize for lowest cost
   - SMT-based synthesis approach

   **Relevance to LOOM:**

   Instead of manually writing all optimization rules, LOOM could use synthesis
   techniques to discover new optimization patterns. This could complement manual
   rule writing with automated discovery.

   **Implementation Notes:**

   - Study synthesis techniques for ISLE rules
   - Identify optimization opportunities via synthesis
   - Verify synthesized rules with Crocus
   - Consider cost models for rule selection

WebAssembly Security and Verification
======================================

VeriWasm: Software Fault Isolation Verification
------------------------------------------------

.. req:: Study SFI Verification for Sandboxing
   :id: REQ_RESEARCH_008
   :status: planned
   :priority: Medium
   :category: Research

   Review VeriWasm's approach to verifying memory isolation.

   **Tool Details:**

   - **Name:** VeriWasm
   - **Purpose:** Static offline verifier for native x86-64 WebAssembly binaries
   - **Venue:** NDSS 2021 (with ongoing development)
   - **Repository:** https://github.com/PLSysSec/veriwasm

   **Key Contributions:**

   - Mechanized soundness proofs
   - Verified abstract interpretation passes
   - Post-compilation sandbox verification
   - x86-64 binary analysis

   **Technical Approach:**

   - Static analysis of compiled binaries
   - Verified abstract interpretation
   - Memory isolation guarantees
   - Proven soundness

   **Relevance to LOOM:**

   While LOOM optimizes WebAssembly before native compilation, understanding
   post-compilation verification helps ensure optimizations don't undermine
   security properties. LOOM's output should be amenable to VeriWasm verification.

   **Implementation Notes:**

   - Ensure LOOM preserves security-relevant properties
   - Test LOOM output with VeriWasm
   - Document security invariants
   - Verify sandbox integrity through optimization

ArXiv 2024: WebAssembly Security and Compilation
-------------------------------------------------

.. req:: Review WebAssembly Security Research
   :id: REQ_RESEARCH_009
   :status: planned
   :priority: Low
   :category: Research

   Survey recent security research on WebAssembly compilation.

   **Relevant Papers (2024):**

   1. **"WebAssembly and Security: a review"** (arXiv:2407.12297, July 2024)

      - Mechanized verification using Isabelle
      - Verified type-checker and interpreter
      - Automatic binary transformation for fuzzing

   2. **"Cage: Hardware-Accelerated Safe WebAssembly"** (arXiv:2408.11456, Aug/Dec 2024)

      - LLVM-based compiler toolchain
      - Modified wasi-libc with custom allocator
      - Spatial and temporal memory safety
      - Hardware acceleration

   3. **"Research on WebAssembly Runtimes: A Survey"** (arXiv:2404.12621, April 2024)

      - JIT vs AOT compilation
      - IR optimization techniques
      - Register allocation
      - Native code generation

   4. **"WAMI: Compilation to WebAssembly through MLIR"** (arXiv:2506.16048, 2024)

      - Alternative to LLVM-based compilation
      - Multi-Level Intermediate Representation
      - Compilation strategies

   **Relevance to LOOM:**

   Security research informs optimization safety. LOOM must ensure transformations
   don't introduce vulnerabilities or weaken sandbox guarantees.

Additional Formal Methods Research
===================================

CAV 2024: Computer-Aided Verification
--------------------------------------

.. req:: Review CAV 2024 Relevant Papers
   :id: REQ_RESEARCH_010
   :status: planned
   :priority: Low
   :category: Research

   Study CAV 2024 papers on compiler verification and SMT solvers.

   **Conference Details:**

   - **Venue:** Montreal, QC, Canada
   - **Date:** July 24-27, 2024

   **Relevant Papers:**

   1. **"End-to-End Mechanized Proof of a JIT-Accelerated eBPF Virtual Machine for IoT"**

      - Authors: Shenghao Yuan, Frédéric Besson, Jean-Pierre Talpin
      - Focus: JIT compiler verification

   2. **"The Top-Down Solver Verified: Building Confidence in Static Analyzers"**

      - Authors: Yannick Stade, Sarah Tilscher, Helmut Seidl
      - Focus: Static analysis verification

   3. **SMT Solver Advances:**

      - Z3: An Efficient SMT Solver
      - cvc5: A Versatile and Industrial-Strength SMT Solver (2022)
      - MCSat Yices2: Improved nonlinear arithmetic solving

   **Relevance to LOOM:**

   CAV papers demonstrate state-of-the-art in formal verification. LOOM should
   leverage latest SMT solver capabilities and verification techniques.

Related Language Semantics Work
================================

SpecTec and WebAssembly Specification
--------------------------------------

.. req:: Monitor WebAssembly Specification Evolution
   :id: REQ_RESEARCH_011
   :status: planned
   :priority: Medium
   :category: Research

   Track WebAssembly formal specification and SpecTec developments.

   **Recent Developments (2024-2025):**

   - SpecTec adopted for WebAssembly specification (March 2025)
   - WasmCert mechanization covering Wasm 2.0
   - Porting proofs to SpecTec-based specification
   - Soundness proofs for Wasm 1.0 subset complete

   **Tools:**

   - SpecTec: Specification technology for formal language semantics
   - WasmCert: Mechanized WebAssembly soundness proofs
   - Isabelle/Coq mechanizations

   **Relevance to LOOM:**

   LOOM's optimizations must preserve WebAssembly semantics as defined by the
   formal specification. Tracking spec evolution ensures LOOM stays current
   with WebAssembly features.

   **Implementation Notes:**

   - Follow W3C WebAssembly specification updates
   - Review SpecTec formal semantics
   - Align LOOM semantics with official spec
   - Test against spec test suite

Research Summary and Recommendations
=====================================

.. req:: Research-Informed Verification Strategy
   :id: REQ_RESEARCH_012
   :status: planned
   :priority: Critical
   :category: Research
   :links: REQ_VERIFY_001, REQ_RESEARCH_001, REQ_RESEARCH_002

   Develop verification strategy informed by academic research (2020-2025).

   **Key Findings from Research:**

   1. **SMT-Based Verification Works:** Crocus (ASPLOS 2024) successfully verified
      production compiler rules and found real CVEs.

   2. **Scales to Production:** Arrival (2025) demonstrates end-to-end verification
      can work for industrial compilers like Cranelift.

   3. **E-Graphs Complement Term Rewriting:** Equality saturation addresses
      phase-ordering problems that sequential term rewriting struggles with.

   4. **Mechanized Semantics Available:** WasmCert and related work provide
      mechanized WebAssembly semantics suitable for correctness proofs.

   5. **Security-Critical:** Multiple CVEs and sandbox escapes found through
      verification highlight importance of formal methods.

   **Recommended Verification Approach for LOOM:**

   **Tier 1 (Critical - Must Have):**

   - Adopt Crocus-style SMT verification for all optimization rules
   - Verify rules across all bitwidths (8, 16, 32, 64, 128)
   - Generate counterexamples for debugging
   - Track verification coverage metrics
   - Block unverified rules from production

   **Tier 2 (High Priority - Should Have):**

   - Investigate Arrival's scalability techniques
   - Integrate e-graph equality saturation for phase-ordering
   - Mechanize key correctness properties
   - Test with VeriWasm for sandbox preservation
   - Align with WebAssembly formal specification

   **Tier 3 (Medium Priority - Nice to Have):**

   - Explore rule synthesis for optimization discovery
   - Automated confluence checking
   - Full Coq/Isabelle mechanization
   - Property-based testing with verified interpreter

   **Implementation Roadmap:**

   1. **Phase 1:** Integrate Crocus for rule verification (REQ_RESEARCH_001)
   2. **Phase 2:** Study and adapt Arrival techniques (REQ_RESEARCH_002)
   3. **Phase 3:** Experiment with e-graph integration (REQ_RESEARCH_005)
   4. **Phase 4:** Pursue mechanized correctness proofs (REQ_RESEARCH_003)

   **Success Metrics:**

   - 100% of optimization rules formally verified
   - Zero known unsound transformations
   - Verification time < 5 minutes per rule
   - Zero security vulnerabilities introduced
   - Compatible with VeriWasm and similar tools

Conference and Workshop Tracking
=================================

.. req:: Monitor Relevant Conferences
   :id: REQ_RESEARCH_013
   :status: planned
   :priority: Low
   :category: Research

   Track ongoing research at key conferences and workshops.

   **Annual Conferences to Monitor:**

   **Top-Tier Programming Languages:**

   - POPL (Principles of Programming Languages) - January
   - PLDI (Programming Language Design and Implementation) - June
   - OOPSLA (Object-Oriented Programming, Systems, Languages & Applications) - October
   - ICFP (International Conference on Functional Programming) - September

   **Verification and Formal Methods:**

   - CAV (Computer-Aided Verification) - July
   - FMCAD (Formal Methods in Computer-Aided Design) - October
   - FM (Formal Methods) - September (biennial)
   - TACAS (Tools and Algorithms for the Construction and Analysis of Systems) - April

   **Systems and Compilers:**

   - CGO (Code Generation and Optimization) - March
   - ASPLOS (Architectural Support for Programming Languages and Operating Systems) - March
   - PACT (Parallel Architectures and Compilation Techniques) - October

   **Workshops:**

   - EGRAPHS (E-Graphs workshop at PLDI)
   - WAW (WebAssembly Workshop at POPL)
   - CPP (Certified Programs and Proofs, co-located with POPL)
   - ARRAY (Array Programming at PLDI)

   **Implementation Notes:**

   - Subscribe to conference notifications
   - Review accepted papers lists
   - Download relevant artifacts
   - Track tool developments
   - Engage with research community

Bibliography
============

This section lists all papers referenced above in chronological order.

2020-2023
---------

- **ISLE Language Design:**
  Chris Fallin, "Cranelift's Instruction Selector DSL, ISLE: Term-Rewriting Made Practical"
  Blog post, January 2023
  https://cfallin.org/blog/2023/01/20/cranelift-isle/

- **VeriISLE Technical Report:**
  Monica Pardeshi, "VeriISLE: Verifying Instruction Selection in Cranelift"
  CMU-CS-23-126, 2023
  http://reports-archive.adm.cs.cmu.edu/anon/home/ftp/usr/anon/2023/CMU-CS-23-126.pdf

2024
----

- **Crocus (ASPLOS 2024):**
  "Lightweight, Modular Verification for WebAssembly-to-Native Instruction Selection"
  ASPLOS 2024
  https://dl.acm.org/doi/10.1145/3617232.3624862
  Artifact: https://github.com/avanhatt/asplos24-ae-crocus

- **Tensor E-Graphs (PACT 2024):**
  "Optimizing Tensor Computation Graphs with Equality Saturation and Monte Carlo Tree Search"
  PACT 2024, October 2024
  https://dl.acm.org/doi/10.1145/3656019.3689611
  arXiv:2410.05534

- **Instruction Selection Synthesis (FMCAD 2024):**
  Ross Daly et al., "Efficiently Synthesizing Lowest Cost Rewrite Rules for Instruction Selection"
  FMCAD 2024, Prague, October 2024

- **WebAssembly Security Survey (arXiv 2024):**
  Gaetano Perrone, Simon Pietro Romano, "WebAssembly and Security: a review"
  arXiv:2407.12297, July 2024

- **Cage (arXiv 2024):**
  "Cage: Hardware-Accelerated Safe WebAssembly"
  arXiv:2408.11456, August 2024 (updated December 2024)

- **Wasm Runtimes Survey (arXiv 2024):**
  "Research on WebAssembly Runtimes: A Survey"
  arXiv:2404.12621, April 2024

- **Confluence Proving (arXiv 2024):**
  "Automated Strategy Invention for Confluence of Term Rewrite Systems"
  arXiv:2411.06409, November 2024

2025
----

- **Arrival (PACMPL 2025):**
  "Scaling Instruction-Selection Verification against Authoritative ISA Semantics"
  Proceedings of the ACM on Programming Languages, 2025
  https://dl.acm.org/doi/10.1145/3764383

- **WasmCert-Coq (POPL 2025):**
  "Progressful Interpreters for Efficient WebAssembly Mechanisation"
  POPL 2025
  https://dl.acm.org/doi/10.1145/3704858

- **CertiCoq-Wasm (CPP 2025):**
  "CertiCoq-Wasm: A Verified WebAssembly Backend for CertiCoq"
  CPP 2025
  https://dl.acm.org/doi/10.1145/3703595.3705879

- **LLM-Guided E-Graphs (EGRAPHS 2025):**
  "Equality Saturation Guided by Large Language Models"
  EGRAPHS 2025 (PLDI 2025 Workshop)

- **Contextual E-Graphs (SAS 2025):**
  "Contextual Equality Saturation"
  SAS 2025 (SPLASH 2025)

- **Julia E-Graphs (arXiv 2025):**
  "Equality Saturation for Optimizing High-Level Julia IR"
  arXiv:2502.17075, February 2025

- **SpecTec Adoption:**
  "SpecTec has been adopted"
  WebAssembly.org news, March 2025
  https://webassembly.org/news/2025-03-27-spectec/

Research Artifacts and Code
============================

.. req:: Collect Research Artifacts
   :id: REQ_RESEARCH_014
   :status: planned
   :priority: Medium
   :category: Research

   Collect and study artifacts from relevant research papers.

   **Available Artifacts:**

   - **Crocus:** https://github.com/avanhatt/asplos24-ae-crocus
   - **VeriWasm:** https://github.com/PLSysSec/veriwasm
   - **Cranelift/ISLE:** https://github.com/bytecodealliance/wasmtime/tree/main/cranelift/isle
   - **egg (E-Graphs):** https://github.com/egraphs-good/egg
   - **WasmCert:** Part of WebAssembly specification work

   **Implementation Notes:**

   - Clone and build artifacts
   - Run verification examples
   - Study implementation techniques
   - Adapt code for LOOM
   - Contribute improvements upstream

Related Work Not Yet Reviewed
==============================

.. req:: Future Research Review
   :id: REQ_RESEARCH_015
   :status: planned
   :priority: Low
   :category: Research

   Track additional research areas for future investigation.

   **Topics to Explore:**

   - Alive2 (LLVM optimization verification)
   - CompCert (verified C compiler)
   - CakeML (verified ML compiler)
   - SeL4 (verified OS with verified compiler)
   - Vale (verified assembly language)
   - Dafny (verification-aware programming)
   - F* (verification-oriented functional language)
   - Translation validation techniques
   - Symbolic execution for compiler testing
   - Differential testing methodologies

   **Relevance:**

   These projects demonstrate successful verification at various levels.
   Techniques may be adaptable to LOOM's optimization verification needs.
