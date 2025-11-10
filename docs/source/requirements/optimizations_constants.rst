================================
Constant Folding and Propagation  
================================

.. req:: Precompute
   :id: REQ_OPT_CONST_001
   :status: planned
   :priority: High
   :category: Optimization
   :binaryen_pass: Precompute.cpp

   Compute constant expressions at compile time.

.. req:: Constant Field Propagation
   :id: REQ_OPT_CONST_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: ConstantFieldPropagation.cpp

   Propagate constant values through struct fields.

.. req:: Optimize Added Constants
   :id: REQ_OPT_CONST_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: OptimizeAddedConstants.cpp

   Optimize patterns involving constant additions.
