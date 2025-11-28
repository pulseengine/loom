(module
  ;; Test 1: br_if with constant true - should become unconditional br
  (func $test_br_if_always_taken (result i32)
    (block $exit (result i32)
      (i32.const 42)
      (i32.const 1)
      (br_if $exit)
      (i32.const 99)
    )
  )

  ;; Test 2: br_if with constant false - should be removed
  (func $test_br_if_never_taken (result i32)
    (block $exit (result i32)
      (i32.const 0)
      (br_if $exit)
      (i32.const 42)
    )
  )

  ;; Test 3: if with constant true condition - should take then branch
  (func $test_if_constant_true (result i32)
    (if (result i32) (i32.const 1)
      (then (i32.const 42))
      (else (i32.const 99))
    )
  )

  ;; Test 4: if with constant false condition - should take else branch
  (func $test_if_constant_false (result i32)
    (if (result i32) (i32.const 0)
      (then (i32.const 99))
      (else (i32.const 42))
    )
  )

  ;; Test 5: if with identical arms - should drop condition and keep one arm
  (func $test_if_identical_arms (param $x i32) (result i32)
    (if (result i32) (local.get $x)
      (then (i32.const 42))
      (else (i32.const 42))
    )
  )

  ;; Test 6: nested ifs with constant conditions
  (func $test_nested_constant_ifs (result i32)
    (if (result i32) (i32.const 1)
      (then
        (if (result i32) (i32.const 0)
          (then (i32.const 10))
          (else (i32.const 20))
        )
      )
      (else (i32.const 30))
    )
  )

  ;; Test 7: Nop removal
  (func $test_nop_removal (result i32)
    (nop)
    (i32.const 42)
    (nop)
  )

  ;; Test 8: Complex nested case with multiple optimizations
  (func $test_complex (result i32)
    (block $outer (result i32)
      (if (result i32) (i32.const 1)
        (then
          (block $inner (result i32)
            (i32.const 0)
            (br_if $inner)
            (i32.const 42)
          )
        )
        (else (i32.const 99))
      )
    )
  )
)
