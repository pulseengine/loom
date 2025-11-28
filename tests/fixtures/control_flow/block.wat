(module
  ;; Simple block test
  (func $test_block (result i32)
    (block (result i32)
      i32.const 42
    )
  )

  ;; Block with multiple instructions
  (func $test_block_multi (result i32)
    (block (result i32)
      i32.const 10
      i32.const 20
      i32.add
      i32.const 12
      i32.add
    )
  )

  ;; Nested blocks
  (func $test_nested_blocks (result i32)
    (block (result i32)
      (block (result i32)
        i32.const 5
        i32.const 10
        i32.add
      )
      i32.const 3
      i32.add
    )
  )
)
