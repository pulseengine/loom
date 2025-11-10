==================================
Inlining and Function Optimizations
==================================

.. req:: Function Inlining
   :id: REQ_OPT_INLINE_001
   :status: planned
   :priority: High
   :category: Optimization
   :binaryen_pass: Inlining.cpp

   Inline function calls based on size and frequency heuristics.

.. req:: Directize
   :id: REQ_OPT_INLINE_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: Directize.cpp

   Convert indirect calls to direct calls where possible.

.. req:: Duplicate Function Elimination
   :id: REQ_OPT_INLINE_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: DuplicateFunctionElimination.cpp

   Merge identical functions.
