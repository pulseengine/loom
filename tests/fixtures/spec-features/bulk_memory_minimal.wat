;; Bulk memory ops minimal fixture (post-MVP wasm feature, partial LOOM support)
;; Exercises memory.copy, memory.fill, memory.init, data.drop.
(module
  (memory 1)
  (data $d "hello world")
  (func (export "bulk_test")
    ;; memory.fill: dst=0, val=0, len=4
    i32.const 0
    i32.const 0
    i32.const 4
    memory.fill
    ;; memory.copy: dst=8, src=0, len=4
    i32.const 8
    i32.const 0
    i32.const 4
    memory.copy
    ;; memory.init from data segment $d: dst=16, src=0, len=11
    i32.const 16
    i32.const 0
    i32.const 11
    memory.init $d
    ;; data.drop: drop the data segment
    data.drop $d
  )
)
