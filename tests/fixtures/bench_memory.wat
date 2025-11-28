(module
  (memory 1)

  ;; Phase 13: Memory Redundancy Elimination Benchmark

  ;; Redundant load elimination
  (func $redundant_load (result i32)
    ;; Load from address 100 twice - second load is redundant
    i32.const 100
    i32.load
    i32.const 100
    i32.load          ;; Should be eliminated!
    i32.add
  )

  ;; Store-to-load forwarding
  (func $store_load_forward (result i32)
    ;; Store 42 to address 200, then load it
    i32.const 42
    i32.const 200
    i32.store
    i32.const 200
    i32.load          ;; Should forward the stored value 42!
  )

  ;; Multiple loads from same location
  (func $multiple_loads (result i32)
    i32.const 300
    i32.load
    i32.const 300
    i32.load
    i32.add
    i32.const 300
    i32.load          ;; Third load also redundant!
    i32.add
  )

  ;; Store then store (dead store)
  (func $dead_store (result i32)
    i32.const 10
    i32.const 400
    i32.store         ;; Dead! Overwritten below
    i32.const 20
    i32.const 400
    i32.store
    i32.const 400
    i32.load
  )

  ;; With offset
  (func $with_offset (result i32)
    i32.const 500
    i32.load offset=4
    i32.const 500
    i32.load offset=4 ;; Same address+offset, redundant!
    i32.add
  )

  ;; Constant address computation
  (func $computed_addr (result i32)
    i32.const 100
    i32.const 50
    i32.add           ;; = 150
    i32.load
    i32.const 100
    i32.const 50
    i32.add           ;; = 150 again
    i32.load          ;; Redundant!
    i32.add
  )
)
