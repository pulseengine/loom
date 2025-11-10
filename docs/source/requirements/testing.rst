===============================
Testing and Binaryen Integration
===============================

These requirements define how LOOM integrates with Binaryen's test infrastructure
and provides its own testing capabilities.

Binaryen Test Suite Integration
================================

.. req:: Binaryen Test Suite Compatibility
   :id: REQ_TEST_001
   :status: planned
   :priority: Critical
   :category: Testing

   Support running Binaryen's existing test suite against LOOM optimizations.

   LOOM should be able to:

   - Run lit tests from Binaryen
   - Match expected output formats
   - Support all test directives
   - Report failures clearly

   **Benefits:**

   - Leverage extensive existing test suite
   - Ensure compatibility with wasm-opt behavior
   - Find regressions quickly
   - Build confidence in correctness

   **Implementation Notes:**

   - Clone Binaryen test suite as submodule
   - Create adapter layer for test harness
   - Support FILECHECK-style assertions
   - Integrate with CI

.. req:: Wasm Binary I/O
   :id: REQ_TEST_002
   :status: planned
   :priority: Critical
   :category: Testing
   :links: REQ_TEST_001

   Parse and emit WebAssembly binary format compatible with Binaryen tests.

   Requirements:

   - Read .wasm and .wat files
   - Write optimized .wasm files
   - Preserve module structure
   - Handle all WebAssembly proposals

   **Implementation Notes:**

   - Use wasmparser for parsing
   - Use wasm-encoder for emission
   - Support wat2wasm conversion
   - Handle custom sections

.. req:: Test Harness Adapter
   :id: REQ_TEST_003
   :status: planned
   :priority: High
   :category: Testing
   :links: REQ_TEST_001

   Create adapter layer to run Binaryen lit tests against LOOM.

   The adapter should:

   - Parse lit test files
   - Extract RUN commands
   - Execute LOOM with appropriate flags
   - Compare output with expected results
   - Handle CHECK directives

   **Example lit test:**

   .. code-block:: llvm

      ;; RUN: loom %s -O3 -o %t.wasm
      ;; RUN: loom-dis %t.wasm | filecheck %s

      ;; CHECK: (i32.const 42)

   **Implementation Notes:**

   - Implement filecheck-compatible tool
   - Support all lit test directives
   - Provide detailed failure reports
   - Support test filtering

.. req:: Fuzzing Integration
   :id: REQ_TEST_004
   :status: planned
   :priority: Medium
   :category: Testing

   Support fuzzing with Binaryen's fuzzing infrastructure.

   Fuzzing should:

   - Generate random WebAssembly modules
   - Apply optimizations
   - Check for crashes
   - Verify semantic equivalence
   - Find edge cases

   **Implementation Notes:**

   - Use wasm-smith for fuzzing
   - Integrate with cargo-fuzz
   - Run fuzzing in CI
   - Track found bugs

.. req:: Validation Testing
   :id: REQ_TEST_005
   :status: planned
   :priority: High
   :category: Testing

   Validate all transformed modules maintain WebAssembly validity.

   Validation should check:

   - Type system correctness
   - Stack discipline
   - Reference validity
   - Module structure
   - All proposals supported

   **Implementation Notes:**

   - Use wasmparser validation
   - Run after every transformation
   - Provide detailed error messages
   - Test with invalid inputs

.. req:: Execution Testing
   :id: REQ_TEST_006
   :status: planned
   :priority: High
   :category: Testing

   Test semantic equivalence by executing before/after optimization.

   Execution testing should:

   - Run modules in interpreter
   - Compare outputs
   - Test with various inputs
   - Handle traps correctly
   - Support all instructions

   **Implementation Notes:**

   - Use wasmtime or wasmi
   - Generate test inputs
   - Compare execution traces
   - Handle non-determinism

LOOM-Specific Testing
======================

.. req:: Unit Tests for ISLE Rules
   :id: REQ_TEST_007
   :status: planned
   :priority: High
   :category: Testing

   Provide unit tests for individual ISLE optimization rules.

   Each rule should have tests covering:

   - Happy path (rule applies)
   - Edge cases (boundary conditions)
   - Negative cases (rule doesn't apply)
   - Type variations

   **Implementation Notes:**

   - Use Rust #[test] framework
   - Test generated Rust code
   - Parameterize tests
   - Clear test names

.. req:: Integration Tests
   :id: REQ_TEST_008
   :status: planned
   :priority: High
   :category: Testing

   Test full optimization pipeline on realistic modules.

   Integration tests should:

   - Use real-world WebAssembly modules
   - Test optimization combinations
   - Measure optimization impact
   - Verify correctness

   **Test Sources:**

   - Emscripten-generated modules
   - Rust wasm32 output
   - Hand-written test cases
   - Binaryen examples

   **Implementation Notes:**

   - Collect test corpus
   - Automate test runs
   - Track performance metrics
   - Compare with wasm-opt

.. req:: Regression Tests
   :id: REQ_TEST_009
   :status: planned
   :priority: Medium
   :category: Testing

   Maintain regression test suite for fixed bugs.

   When bugs are found:

   - Add minimal reproduction
   - Document expected behavior
   - Ensure fix is tested
   - Prevent recurrence

   **Implementation Notes:**

   - tests/regression/ directory
   - Issue tracking integration
   - Automated CI checks

.. req:: Performance Benchmarks
   :id: REQ_TEST_010
   :status: planned
   :priority: Medium
   :category: Testing

   Benchmark optimization performance and output quality.

   Metrics to track:

   - Optimization time
   - Module size reduction
   - Execution speed improvement
   - Memory usage

   **Comparison:**

   - LOOM vs wasm-opt
   - Different optimization levels
   - Individual pass impact

   **Implementation Notes:**

   - Use criterion.rs for benchmarks
   - Track results over time
   - Visualize trends
   - CI performance testing

.. req:: Test Coverage Metrics
   :id: REQ_TEST_011
   :status: planned
   :priority: Low
   :category: Testing

   Track and report test coverage for LOOM code.

   Coverage metrics:

   - Line coverage
   - Branch coverage
   - Rule coverage (% of rules tested)
   - Instruction coverage (% of wasm instructions)

   **Implementation Notes:**

   - Use tarpaulin or llvm-cov
   - Generate coverage reports
   - Track in CI
   - Set minimum thresholds

.. req:: Differential Testing
   :id: REQ_TEST_012
   :status: planned
   :priority: Medium
   :category: Testing
   :links: REQ_TEST_001

   Compare LOOM output against wasm-opt on same inputs.

   Differential testing should:

   - Run both optimizers on same module
   - Compare outputs
   - Flag differences
   - Analyze which is better

   **Use Cases:**

   - Find optimization opportunities
   - Detect correctness issues
   - Benchmark against reference
   - Learn from wasm-opt

   **Implementation Notes:**

   - Automated comparison tool
   - Large test corpus
   - Semantic equivalence checking
   - Size/performance comparison

.. req:: Property-Based Testing
   :id: REQ_TEST_013
   :status: planned
   :priority: Low
   :category: Testing

   Use property-based testing for optimization invariants.

   Properties to test:

   - Optimized module validates
   - Execution is equivalent
   - Size doesn't increase (for size opts)
   - Type preservation

   **Implementation Notes:**

   - Use proptest or quickcheck
   - Generate random modules
   - Check properties
   - Shrink failing cases

.. req:: Continuous Integration
   :id: REQ_TEST_014
   :status: planned
   :priority: High
   :category: Testing

   Set up CI pipeline for automated testing.

   CI should run:

   - Unit tests
   - Integration tests
   - Binaryen test suite
   - Fuzzing (limited time)
   - Verification tests
   - Benchmarks

   **Platforms:**

   - Linux (primary)
   - macOS
   - Windows (if applicable)

   **Implementation Notes:**

   - GitHub Actions
   - Matrix builds
   - Artifact storage
   - Failure notifications
