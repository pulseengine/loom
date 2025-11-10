===============
Global Optimizations
===============

.. req:: Simplify Globals
   :id: REQ_OPT_GLOBAL_001
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: SimplifyGlobals.cpp

   Optimize global variable usage.

.. req:: Global Refining
   :id: REQ_OPT_GLOBAL_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: GlobalRefining.cpp

   Refine global types to more specific types.

.. req:: Global Type Optimization
   :id: REQ_OPT_GLOBAL_003
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: GlobalTypeOptimization.cpp

   Optimize global types for GC.

.. req:: Set Globals
   :id: REQ_OPT_GLOBAL_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: SetGlobals.cpp

   Move mutable globals to immutable where possible.
