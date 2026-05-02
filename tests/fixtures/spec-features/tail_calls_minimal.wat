;; Tail calls minimal fixture (post-MVP wasm feature)
;; LOOM does not currently support tail calls; parser must reject cleanly.
(module
  (type $sig (func (param i32) (result i32)))
  (table $t 1 funcref)
  (elem (i32.const 0) $direct)
  (func $direct (param i32) (result i32)
    local.get 0
    i32.const 1
    i32.add
  )
  (func $tail_direct (param i32) (result i32)
    local.get 0
    return_call $direct
  )
  (func $tail_indirect (param i32) (result i32)
    local.get 0
    i32.const 0
    return_call_indirect (type $sig)
  )
  (export "tail_direct" (func $tail_direct))
  (export "tail_indirect" (func $tail_indirect))
)
