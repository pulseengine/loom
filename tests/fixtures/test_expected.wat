;; LOOM Expected Output - After Constant Folding
;; Phase 2: WebAssembly I/O Test
;;
;; Optimized version with constant folding applied
;; (i32.const 10) + (i32.const 32) → (i32.const 42)

(module
  (func $add_constants (result i32)
    i32.const 42
  )
)
