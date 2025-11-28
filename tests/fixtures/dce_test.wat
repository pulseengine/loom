(module
  ;; Test 1: Unreachable code after return
  (func $unreachable_after_return (result i32)
    (return (i32.const 42))
    (i32.const 99)
    (drop)
    (i32.const 100)
  )

  ;; Test 2: Unreachable code after unconditional branch
  (func $unreachable_after_br (result i32)
    (block $exit (result i32)
      (br $exit (i32.const 42))
      (i32.const 99)
      (i32.const 100)
      (i32.add)
    )
  )

  ;; Test 3: Nested blocks with dead code
  (func $nested_dead (result i32)
    (block (result i32)
      (block (result i32)
        (return (i32.const 10))
        (i32.const 20)
      )
      (i32.const 30)
    )
  )

  ;; Test 4: Code after unreachable instruction
  (func $after_unreachable
    (unreachable)
    (i32.const 42)
    (drop)
  )

  ;; Test 5: Live code should be preserved
  (func $live_code (param $x i32) (result i32)
    (local.get $x)
    (i32.const 10)
    (i32.add)
  )
)
