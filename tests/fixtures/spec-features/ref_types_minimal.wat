;; Reference types minimal fixture (post-MVP wasm feature)
;; Exercises ref.null, ref.func, table.get on a funcref table.
(module
  (table $t 2 funcref)
  (func $callee (result i32)
    i32.const 7
  )
  (elem (i32.const 0) $callee)
  (func (export "ref_test") (result funcref)
    ;; ref.null funcref
    ref.null func
    drop
    ;; table.get
    i32.const 0
    table.get $t
    drop
    ;; ref.func returning a funcref
    ref.func $callee
  )
)
