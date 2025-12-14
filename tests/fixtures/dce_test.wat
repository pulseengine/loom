(module
  ;; Test 1: Unreachable code after return
  (func $unreachable_after_return (export "unreachable_after_return") (result i32)
    (return (i32.const 42))
    (i32.const 99)  ;; Dead code - never reached
  )

  ;; Test 2: Unreachable code after unconditional branch
  (func $unreachable_after_br (export "unreachable_after_br") (result i32)
    (block $exit (result i32)
      (br $exit (i32.const 42))
      (i32.const 99)  ;; Dead code
    )
  )

  ;; Test 3: Code after unreachable instruction
  (func $after_unreachable (export "after_unreachable")
    (unreachable)
    ;; Everything after unreachable is dead
  )

  ;; Test 4: Live code should be preserved
  (func $live_code (export "live_code") (param $x i32) (result i32)
    (local.get $x)
    (i32.const 10)
    (i32.add)
  )

  ;; Test 5: Dead branch in if
  (func $dead_if_branch (export "dead_if_branch") (result i32)
    (if (result i32) (i32.const 1)
      (then (i32.const 42))
      (else (i32.const 99))  ;; Dead - condition always true
    )
  )

  ;; Test 6: Multiple returns
  (func $multi_return (export "multi_return") (param $x i32) (result i32)
    (if (result i32) (local.get $x)
      (then
        (return (i32.const 1))
        (i32.const 999)  ;; Dead
      )
      (else
        (i32.const 0)
      )
    )
  )
)
