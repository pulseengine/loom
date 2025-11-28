(module
  ;; Simple branch
  (func $test_br (result i32)
    (block $outer (result i32)
      (i32.const 1)
      (br $outer)
      (i32.const 2)
    )
  )

  ;; Conditional branch
  (func $test_br_if (param $x i32) (result i32)
    (block $exit (result i32)
      (i32.const 42)
      (br_if $exit (i32.eqz (local.get $x)))
      (i32.const 10)
      (i32.add)
    )
  )

  ;; Branch table (switch)
  (func $test_br_table (param $x i32) (result i32)
    (block $case0 (result i32)
      (block $case1 (result i32)
        (block $case2 (result i32)
          (block $default (result i32)
            (local.get $x)
            (br_table $case0 $case1 $case2 $default)
          )
          ;; default case
          (i32.const 99)
          (return)
        )
        ;; case 2
        (i32.const 2)
        (return)
      )
      ;; case 1
      (i32.const 1)
      (return)
    )
    ;; case 0
    (i32.const 0)
  )

  ;; Return statement
  (func $test_return (param $x i32) (result i32)
    (if (i32.eqz (local.get $x))
      (then
        (i32.const 0)
        (return)
      )
    )
    (i32.const 1)
  )
)
