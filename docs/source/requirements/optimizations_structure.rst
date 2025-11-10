==========================
Code Structure Optimizations
==========================

.. req:: Code Folding
   :id: REQ_OPT_STRUCT_001
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: CodeFolding.cpp

   Merge identical code blocks.

.. req:: Code Pushing
   :id: REQ_OPT_STRUCT_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: CodePushing.cpp

   Push code later in execution to minimize work on early exits.

.. req:: Merge Blocks
   :id: REQ_OPT_STRUCT_003
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: MergeBlocks.cpp

   Merge consecutive blocks.

.. req:: Flatten
   :id: REQ_OPT_STRUCT_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: Flatten.cpp

   Flatten nested block structures.

.. req:: ReReloop
   :id: REQ_OPT_STRUCT_005
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: ReReloop.cpp

   Optimize control flow by re-analyzing loop structure.
