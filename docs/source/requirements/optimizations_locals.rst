==============================
Local Variable Optimizations
==============================

.. req:: Simplify Locals
   :id: REQ_OPT_LOCAL_001
   :status: planned
   :priority: High
   :category: Optimization
   :binaryen_pass: SimplifyLocals.cpp

   Optimize local variable usage by sinking sets and coalescing.

.. req:: Coalesce Locals
   :id: REQ_OPT_LOCAL_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: CoalesceLocals.cpp

   Merge local variables with non-overlapping lifetimes.

.. req:: Merge Locals
   :id: REQ_OPT_LOCAL_003
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: MergeLocals.cpp

   Merge locals that always have the same value.

.. req:: Reorder Locals
   :id: REQ_OPT_LOCAL_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: ReorderLocals.cpp

   Reorder local declarations by usage frequency.

.. req:: Local CSE
   :id: REQ_OPT_LOCAL_005
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: LocalCSE.cpp

   Common subexpression elimination for local operations.

.. req:: SSAify
   :id: REQ_OPT_LOCAL_006
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: SSAify.cpp

   Convert to Static Single Assignment form.
