#!/usr/bin/env python3

files = {
    "optimizations_memory.rst": """====================
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
""",
    "optimizations_structure.rst": """==========================
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
""",
    "optimizations_instructions.rst": """===========================
Instruction-Level Optimizations
===========================

.. req:: Optimize Instructions
   :id: REQ_OPT_INSTR_001
   :status: planned
   :priority: High
   :category: Optimization
   :binaryen_pass: OptimizeInstructions.cpp

   Apply peephole optimizations to instruction sequences.

.. req:: Optimize Casts
   :id: REQ_OPT_INSTR_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: OptimizeCasts.cpp

   Optimize type cast operations.

.. req:: Redundant Set Elimination
   :id: REQ_OPT_INSTR_003
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: RedundantSetElimination.cpp

   Remove redundant local.set operations.

.. req:: Untee
   :id: REQ_OPT_INSTR_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: Untee.cpp

   Remove local.tee operations where beneficial.
""",
    "optimizations_globals.rst": """===============
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
""",
    "optimizations_types.rst": """=================
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
""",
    "optimizations_loops.rst": """==================
Loop Optimizations
==================

.. req:: Loop Invariant Code Motion
   :id: REQ_OPT_LOOP_001
   :status: planned
   :priority: High
   :category: Optimization
   :binaryen_pass: LoopInvariantCodeMotion.cpp

   Move loop-invariant code out of loops (LICM).
""",
    "optimizations_dataflow.rst": """========================
Data Flow Optimizations
========================

.. req:: Data Flow Optimization
   :id: REQ_OPT_FLOW_001
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: DataFlowOpts.cpp

   General data flow analysis and optimization.

.. req:: GUFA (Global Unified Flow Analysis)
   :id: REQ_OPT_FLOW_002
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: GUFA.cpp

   Perform interprocedural flow analysis.

.. req:: Global Effects Analysis
   :id: REQ_OPT_FLOW_003
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: GlobalEffects.cpp

   Analyze global side effects of functions.
""",
    "optimizations_gc.rst": """=======================
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
""",
    "optimizations_lowering.rst": """========================
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
""",
    "optimizations_js.rst": """===================================
JavaScript Interop Optimizations
===================================

.. req:: Optimize for JS
   :id: REQ_OPT_JS_001
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: OptimizeForJS.cpp

   Optimize WebAssembly for JavaScript interop.

.. req:: Legalize JS Interface
   :id: REQ_OPT_JS_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: LegalizeJSInterface.cpp

   Legalize types for JavaScript interface.

.. req:: Generate Dynamic Calls
   :id: REQ_OPT_JS_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: GenerateDynCalls.cpp

   Generate dynamic call wrappers.

.. req:: Post-Emscripten
   :id: REQ_OPT_JS_004
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: PostEmscripten.cpp

   Optimizations specific to Emscripten output.

.. req:: JSPI (JavaScript Promise Integration)
   :id: REQ_OPT_JS_005
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: JSPI.cpp

   Optimize JavaScript Promise Integration for async/await.
""",
    "optimizations_special.rst": """==========================
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
""",
    "optimizations_debug.rst": """========================
Debugging and Metadata
========================

.. req:: Debug Location Propagation
   :id: REQ_OPT_DEBUG_001
   :status: planned
   :priority: Medium
   :category: Optimization
   :binaryen_pass: DebugLocationPropagation.cpp

   Propagate source location debug information through optimizations.

.. req:: DWARF Processing
   :id: REQ_OPT_DEBUG_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: DWARF.cpp

   Process DWARF debug information.

.. req:: Strip Debug Info
   :id: REQ_OPT_DEBUG_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: Strip.cpp

   Remove debug information from modules.
""",
    "optimizations_reordering.rst": """==========================
Reordering Optimizations
==========================

.. req:: Reorder Functions
   :id: REQ_OPT_REORDER_001
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: ReorderFunctions.cpp

   Reorder functions for better cache locality.

.. req:: Reorder Globals
   :id: REQ_OPT_REORDER_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: ReorderGlobals.cpp

   Reorder global declarations.

.. req:: Reorder Types
   :id: REQ_OPT_REORDER_003
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: ReorderTypes.cpp

   Reorder type declarations.
""",
    "optimizations_minification.rst": """==============
Minification
==============

.. req:: Minify Imports and Exports
   :id: REQ_OPT_MINIFY_001
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: MinifyImportsAndExports.cpp

   Shorten import and export names.

.. req:: Remove Unused Names
   :id: REQ_OPT_MINIFY_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: RemoveUnusedNames.cpp

   Remove unnecessary names from the name section.
""",
    "optimizations_eh.rst": """====================
Exception Handling
====================

.. req:: Translate Exception Handling
   :id: REQ_OPT_EH_001
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: TranslateEH.cpp

   Transform exception handling constructs.

.. req:: Strip Exception Handling
   :id: REQ_OPT_EH_002
   :status: planned
   :priority: Low
   :category: Optimization
   :binaryen_pass: StripEH.cpp

   Remove exception handling support.
""",
    "optimizations_misc.rst": """===========================
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
"""
}

for filename, content in files.items():
    with open(filename, 'w') as f:
        f.write(content)
    print(f"Created {filename}")

print("\nAll optimization requirement files created successfully!")
