=======================
GC and Struct Optimizations
=======================

.. req:: Global Struct Inference
   :id: REQ_OPT_GC_001
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: GlobalStructInference.cpp

   Infer struct types for globals.

.. req:: Tuple Optimization
   :id: REQ_OPT_GC_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: TupleOptimization.cpp

   Optimize tuple operations.

.. req:: Monomorphize
   :id: REQ_OPT_GC_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: Monomorphize.cpp

   Monomorphize polymorphic code via type specialization.
