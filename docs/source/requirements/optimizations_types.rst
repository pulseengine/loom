=================
Type Optimizations
=================

.. req:: Type Refining
   :id: REQ_OPT_TYPE_001
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: TypeRefining.cpp

   Refine types to more specific subtypes.

.. req:: Type Generalizing
   :id: REQ_OPT_TYPE_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: TypeGeneralizing.cpp

   Generalize types where beneficial.

.. req:: Type Merging
   :id: REQ_OPT_TYPE_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: TypeMerging.cpp

   Merge similar types in rec groups.

.. req:: Abstract Type Refining
   :id: REQ_OPT_TYPE_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: AbstractTypeRefining.cpp

   Refine abstract types based on usage.

.. req:: Signature Refining
   :id: REQ_OPT_TYPE_005
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: SignatureRefining.cpp

   Refine function signatures.

.. req:: Signature Pruning
   :id: REQ_OPT_TYPE_006
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: SignaturePruning.cpp

   Remove unused parameters from function signatures.
