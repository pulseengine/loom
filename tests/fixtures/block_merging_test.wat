(module
  ;; Test 1: Simple nested blocks - should merge
  (func $simple_nested (result i32)
    (block (result i32)
      (block (result i32)
        (i32.const 42)
      )
    )
  )

  ;; Test 2: Triple nested blocks - should fully merge
  (func $triple_nested (result i32)
    (block (result i32)
      (block (result i32)
        (block (result i32)
          (i32.const 42)
        )
      )
    )
  )

  ;; Test 3: Block with content before inner block
  (func $block_with_prefix (result i32)
    (block (result i32)
      (i32.const 10)
      (block (result i32)
        (i32.const 32)
        (i32.add)
      )
    )
  )

  ;; Test 4: Nested empty blocks
  (func $nested_empty
    (block
      (block
        (nop)
      )
    )
  )

  ;; Test 5: Block within if statement
  (func $block_in_if (param $x i32) (result i32)
    (if (result i32) (local.get $x)
      (then
        (block (result i32)
          (block (result i32)
            (i32.const 42)
          )
        )
      )
      (else
        (i32.const 99)
      )
    )
  )

  ;; Test 6: Block within loop
  (func $block_in_loop (result i32)
    (loop (result i32)
      (block (result i32)
        (block (result i32)
          (i32.const 42)
        )
      )
    )
  )

  ;; Test 7: Multiple nested blocks at different levels
  (func $complex_nesting (result i32)
    (block (result i32)
      (i32.const 5)
      (block (result i32)
        (i32.const 10)
        (i32.add)
        (block (result i32)
          (i32.const 3)
          (i32.mul)
        )
      )
    )
  )
)
