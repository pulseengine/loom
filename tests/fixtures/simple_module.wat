;; LOOM Simple Module - Minimal Valid WebAssembly
;; Phase 2: Round-trip test fixture

(module
  (func $get_answer (result i32)
    i32.const 42
  )
)
