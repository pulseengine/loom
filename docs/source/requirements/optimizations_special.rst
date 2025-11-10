==========================
Special Purpose Optimizations
==========================

.. req:: Asyncify
   :id: REQ_OPT_SPECIAL_001
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: Asyncify.cpp

   Transform code for async/await style execution.

.. req:: Poppify
   :id: REQ_OPT_SPECIAL_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: Poppify.cpp

   Convert to stack machine form.

.. req:: Once Reduction
   :id: REQ_OPT_SPECIAL_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: OnceReduction.cpp

   Reduce once blocks to simpler forms.
