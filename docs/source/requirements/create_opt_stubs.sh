#!/bin/bash

# Constants
cat > optimizations_constants.rst << 'EOF'
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
EOF

# Inlining
cat > optimizations_inlining.rst << 'EOF'
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
EOF

# Create stubs for remaining categories
for category in locals memory structure instructions globals types loops dataflow gc lowering js special debug reordering minification eh misc; do
  cat > "optimizations_${category}.rst" << STUBEOF
$(echo ${category^} | sed 's/_/ /g') Optimizations
================================================

.. req:: ${category^} Optimization Placeholder
   :id: REQ_OPT_${category^^}_001
   :status: planned
   :priority: Medium
   :category: Optimization

   Optimization requirements for ${category} will be detailed here.
STUBEOF
done

echo "Created all optimization stub files"
