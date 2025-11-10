===========================
Miscellaneous Optimizations
===========================

.. req:: DeNaN
   :id: REQ_OPT_MISC_001
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: DeNaN.cpp

   Remove NaN values where possible.

.. req:: DeAlign
   :id: REQ_OPT_MISC_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: DeAlign.cpp

   Remove unnecessary alignment hints.

.. req:: Duplicate Import Elimination
   :id: REQ_OPT_MISC_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: DuplicateImportElimination.cpp

   Remove duplicate imports.

.. req:: Remove Imports
   :id: REQ_OPT_MISC_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: RemoveImports.cpp

   Remove unused imports.

.. req:: Minimize Rec Groups
   :id: REQ_OPT_MISC_005
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: MinimizeRecGroups.cpp

   Minimize recursive type groups.
