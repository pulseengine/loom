============
Verification
============

These requirements define the formal verification capabilities of LOOM using SMT-based
techniques from Crocus.

SMT-Based Verification
======================

.. req:: Crocus SMT Verification
   :id: REQ_VERIFY_001
   :status: planned
   :priority: Critical
   :category: Verification

   Integrate Crocus SMT-based verification tool for proving optimization rule soundness.

   Crocus provides:

   - SMT solver-based verification (Z3)
   - Automatic counterexample generation
   - Bitv vector modeling of WebAssembly values
   - Support for verifying individual rules

   **Implementation Notes:**

   - Use crocus from cranelift/isle/veri
   - Integrate with LOOM build system
   - Create verification test suite
   - Document verification process

   **References:**

   - ASPLOS 2024 paper: "Lightweight, Modular Verification for WebAssembly-to-Native Instruction Selection"
   - https://dl.acm.org/doi/10.1145/3617232.3624862

.. req:: Specification Annotations
   :id: REQ_VERIFY_002
   :status: planned
   :priority: Critical
   :category: Verification
   :links: REQ_VERIFY_001

   Provide formal specifications (spec, require, provide) for all optimization rules.

   Every ISLE term used in optimization rules must have:

   - ``spec`` block with formal semantics
   - ``require`` block for preconditions
   - ``provide`` block for postconditions

   **Example:**

   .. code-block:: isle

      (spec (iadd x y)
          (provide (= result (bvadd x y))))

      (rule (lower (iadd (iconst x) (iconst y)))
            (iconst (iadd_const x y)))

   **Implementation Notes:**

   - Annotate all WebAssembly instructions
   - Provide specs for helper terms
   - Document specification language
   - Create specification templates

.. req:: Bitvector Modeling
   :id: REQ_VERIFY_003
   :status: planned
   :priority: High
   :category: Verification
   :links: REQ_VERIFY_001

   Model WebAssembly values as SMT bitvectors for verification.

   Support the following widths:

   - 8-bit (i8)
   - 16-bit (i16)
   - 32-bit (i32, f32)
   - 64-bit (i64, f64)
   - 128-bit (v128)

   **Implementation Notes:**

   - Use SMT-LIB bitvector theory
   - Model floats as bitvectors
   - Support all WebAssembly numeric operations
   - Handle type conversions

.. req:: Type Instantiation
   :id: REQ_VERIFY_004
   :status: planned
   :priority: High
   :category: Verification
   :links: REQ_VERIFY_001

   Support type instantiation for verification across different bitwidths.

   Using ``instantiate`` and ``form`` directives:

   .. code-block:: isle

      (form
        bv_binary_8_to_64
        ((args (bv  8) (bv  8)) (ret (bv  8)) (canon (bv  8)))
        ((args (bv 16) (bv 16)) (ret (bv 16)) (canon (bv 16)))
        ((args (bv 32) (bv 32)) (ret (bv 32)) (canon (bv 32)))
        ((args (bv 64) (bv 64)) (ret (bv 64)) (canon (bv 64)))
      )

      (instantiate iadd bv_binary_8_to_64)

   **Implementation Notes:**

   - Define type forms for all instruction patterns
   - Instantiate rules for relevant widths
   - Verify each instantiation separately

.. req:: Counterexample Generation
   :id: REQ_VERIFY_005
   :status: planned
   :priority: High
   :category: Verification
   :links: REQ_VERIFY_001

   Generate detailed counterexamples when optimization rules are unsound.

   Counterexamples should show:

   - Input values that trigger unsoundness
   - Expected output
   - Actual output
   - Bitwidth being tested
   - Failed SMT condition

   **Example Output:**

   .. code-block:: text

      Verification failed for iadd_wrong, width 8
      Counterexample summary
      (lower (has_type (fits_in_64 [ty|8]) (iadd [x|#x01] [y|#x00])))
      =>
      (output_reg (alu_wrong [ty|8] [x|#x01] [y|#x00]))

      #x01|0b00000001 =>
      #x02|0b00000010

      Failed condition:
      (= ((_ extract 7 0) lower__13) ((_ extract 7 0) output_reg__16))

   **Implementation Notes:**

   - Format counterexamples clearly
   - Show both hex and binary representations
   - Highlight the failing condition
   - Provide enough context for debugging

.. req:: Memory Operation Verification
   :id: REQ_VERIFY_006
   :status: planned
   :priority: High
   :category: Verification
   :links: REQ_VERIFY_001

   Verify correctness of load/store optimizations with memory models.

   Use Crocus memory operations:

   - ``load_effect`` for modeling loads
   - ``store_effect`` for modeling stores
   - Flags and address modeling

   **Constraints:**

   - Only 1 load_effect per rule side
   - Only 1 store_effect per rule side
   - Memory effects must match on both sides

   **Example:**

   .. code-block:: isle

      (spec (load addr)
          (provide (= result (load_effect #x0000 64 addr))))

   **Implementation Notes:**

   - Model WebAssembly linear memory
   - Support different load/store widths
   - Handle alignment requirements
   - Verify memory ordering

.. req:: Control Flow Verification
   :id: REQ_VERIFY_007
   :status: planned
   :priority: High
   :category: Verification
   :links: REQ_VERIFY_001

   Verify correctness of control flow transformations.

   Ensure that:

   - Branch conditions are preserved
   - Block types match
   - Unreachability is maintained
   - Loop structures are equivalent

   **Implementation Notes:**

   - Model control flow in SMT
   - Use ``if`` (ite) and ``switch`` operations
   - Verify branch targeting
   - Check stack height consistency

.. req:: Verification Test Suite
   :id: REQ_VERIFY_008
   :status: planned
   :priority: Medium
   :category: Verification
   :links: REQ_VERIFY_001

   Create comprehensive test suite for verification.

   Test suite should include:

   - Positive tests (correct rules)
   - Negative tests (intentionally wrong rules)
   - Edge cases (overflow, underflow)
   - All bitwidths
   - Memory operations
   - Control flow

   **Implementation Notes:**

   - Use Rust tests with Crocus
   - Automated CI verification
   - Regression tests for found bugs
   - Performance benchmarks

.. req:: Verification Documentation
   :id: REQ_VERIFY_009
   :status: planned
   :priority: Medium
   :category: Verification

   Provide comprehensive documentation for verification process.

   Documentation should cover:

   - How to write specifications
   - SMT-LIB operations reference
   - Common patterns and idioms
   - Troubleshooting failed verifications
   - Performance tuning

   **Implementation Notes:**

   - Tutorial-style documentation
   - Example specifications
   - Best practices
   - FAQ section

.. req:: Incremental Verification
   :id: REQ_VERIFY_010
   :status: planned
   :priority: Low
   :category: Verification
   :links: REQ_VERIFY_001

   Support incremental verification of changed rules only.

   Cache verification results to:

   - Speed up development iteration
   - Only re-verify changed rules
   - Track verification status per rule

   **Implementation Notes:**

   - Hash rule definitions
   - Store verification results
   - Invalidate on changes
   - Parallel verification

.. req:: Verification Metrics
   :id: REQ_VERIFY_011
   :status: planned
   :priority: Low
   :category: Verification

   Track and report verification coverage metrics.

   Metrics should include:

   - Number of rules verified
   - Number of rules with specs
   - Coverage by category
   - Verification time per rule
   - Failed verifications

   **Implementation Notes:**

   - Generate metrics report
   - CI integration
   - Track over time
   - Visualize in documentation
