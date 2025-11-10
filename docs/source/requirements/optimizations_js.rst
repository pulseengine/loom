===================================
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
