(module
  ;; Simple if-then
  (func $test_if_then (param $x i32) (result i32)
    (if (result i32) (i32.eqz (local.get $x))
      (then
        (i32.const 42)
      )
      (else
        (i32.const 0)
      )
    )
  )

  ;; If without else
  (func $test_if_no_else (param $x i32) (result i32)
    (local $result i32)
    (local.set $result (i32.const 0))
    (if (i32.gt_s (local.get $x) (i32.const 10))
      (then
        (local.set $result (i32.const 1))
      )
    )
    (local.get $result)
  )

  ;; Nested if
  (func $test_nested_if (param $x i32) (param $y i32) (result i32)
    (if (result i32) (i32.gt_s (local.get $x) (i32.const 0))
      (then
        (if (result i32) (i32.gt_s (local.get $y) (i32.const 0))
          (then
            (i32.const 1)
          )
          (else
            (i32.const 2)
          )
        )
      )
      (else
        (i32.const 3)
      )
    )
  )

  ;; Complex if-else with expressions
  (func $test_complex_if (param $a i32) (param $b i32) (result i32)
    (if (result i32) (i32.lt_s (local.get $a) (local.get $b))
      (then
        (i32.add (local.get $a) (local.get $b))
      )
      (else
        (i32.sub (local.get $a) (local.get $b))
      )
    )
  )
)
