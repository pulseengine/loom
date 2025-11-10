====================
Memory Optimizations
====================

.. req:: Heap2Local
   :id: REQ_OPT_MEM_001
   :status: planned
   :priority: High
   :category: Optimization
   :binaryen_pass: Heap2Local.cpp

   Convert heap allocations to local variables (stack promotion).

.. req:: Heap Store Optimization
   :id: REQ_OPT_MEM_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: HeapStoreOptimization.cpp

   Optimize redundant heap stores.

.. req:: Memory Packing
   :id: REQ_OPT_MEM_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: MemoryPacking.cpp

   Optimize memory layout for better packing.

.. req:: Avoid Reinterprets
   :id: REQ_OPT_MEM_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: AvoidReinterprets.cpp

   Avoid reinterpret operations via more loads.

.. req:: Pick Load Signs
   :id: REQ_OPT_MEM_005
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: PickLoadSigns.cpp

   Choose optimal sign extension for loads.
