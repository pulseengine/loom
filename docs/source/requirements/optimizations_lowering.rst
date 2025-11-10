========================
Lowering Transformations
========================

.. req:: I64 to I32 Lowering
   :id: REQ_OPT_LOWER_001
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: I64ToI32Lowering.cpp

   Lower 64-bit integers to 32-bit pairs for 32-bit targets.

.. req:: Memory64 Lowering
   :id: REQ_OPT_LOWER_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: Memory64Lowering.cpp

   Lower memory64 operations.

.. req:: Multi-Memory Lowering
   :id: REQ_OPT_LOWER_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: MultiMemoryLowering.cpp

   Lower multiple memories to single memory.

.. req:: Alignment Lowering
   :id: REQ_OPT_LOWER_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: AlignmentLowering.cpp

   Lower unaligned loads/stores to aligned ones.

.. req:: Sign Extension Lowering
   :id: REQ_OPT_LOWER_005
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: SignExtLowering.cpp

   Lower sign extension operations.
