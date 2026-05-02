;; Multi-memory minimal fixture (post-MVP wasm feature, partial LOOM support)
;; Two memories with explicit memarg `mem` index on loads/stores.
(module
  (memory $m0 1)
  (memory $m1 1)
  (func (export "mm_test") (result i32)
    ;; store 42 at offset 0 of memory $m1
    i32.const 0
    i32.const 42
    i32.store (memory $m1)
    ;; load from memory $m1
    i32.const 0
    i32.load (memory $m1)
    ;; store 7 at offset 0 of memory $m0
    i32.const 0
    i32.const 7
    i32.store (memory $m0)
    ;; load from memory $m0 and add
    i32.const 0
    i32.load (memory $m0)
    i32.add
  )
)
