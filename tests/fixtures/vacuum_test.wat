(module
  ;; Test 1: Remove nops
  (func $remove_nops (result i32)
    (nop)
    (i32.const 42)
    (nop)
  )

  ;; Test 2: Unwrap trivial block
  (func $unwrap_trivial_block (result i32)
    (block (result i32)
      (i32.const 42)
    )
  )

  ;; Test 3: Unwrap nested trivial blocks
  (func $unwrap_nested_blocks (result i32)
    (block (result i32)
      (block (result i32)
        (i32.const 42)
      )
    )
  )

  ;; Test 4: Keep block with multiple instructions
  (func $keep_complex_block (result i32)
    (block (result i32)
      (i32.const 10)
      (i32.const 32)
      (i32.add)
    )
  )

  ;; Test 5: Remove empty loop
  (func $remove_empty_loop
    (loop)
  )

  ;; Test 6: Unwrap trivial empty block
  (func $unwrap_empty_block
    (block)
  )

  ;; Test 7: Mixed - some unwrappable, some not
  (func $mixed_patterns (result i32)
    (nop)
    (block (result i32)
      (i32.const 10)
    )
    (nop)
    (block (result i32)
      (i32.const 32)
      (i32.add)
    )
  )
)
