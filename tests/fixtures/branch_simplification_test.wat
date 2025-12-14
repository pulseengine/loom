(module
  ;; Test 1: if with constant true condition - should take then branch
  (func $test_if_constant_true (export "test_if_constant_true") (result i32)
    (if (result i32) (i32.const 1)
      (then (i32.const 42))
      (else (i32.const 99))
    )
  )

  ;; Test 2: if with constant false condition - should take else branch
  (func $test_if_constant_false (export "test_if_constant_false") (result i32)
    (if (result i32) (i32.const 0)
      (then (i32.const 99))
      (else (i32.const 42))
    )
  )

  ;; Test 3: if with identical arms - should drop condition and keep one arm
  (func $test_if_identical_arms (export "test_if_identical_arms") (param $x i32) (result i32)
    (if (result i32) (local.get $x)
      (then (i32.const 42))
      (else (i32.const 42))
    )
  )

  ;; Test 4: nested ifs with constant conditions
  (func $test_nested_constant_ifs (export "test_nested_constant_ifs") (result i32)
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

  ;; Test 5: Nop removal
  (func $test_nop_removal (export "test_nop_removal") (result i32)
    (nop)
    (i32.const 42)
    (nop)
  )

  ;; Test 6: select with constant condition
  (func $test_select_constant (export "test_select_constant") (result i32)
    (select
      (i32.const 42)
      (i32.const 99)
      (i32.const 1)
    )
  )

  ;; Test 7: Simple block with constant
  (func $test_simple_block (export "test_simple_block") (result i32)
    (block (result i32)
      (i32.const 42)
    )
  )

  ;; Test 8: Nested blocks
  (func $test_nested_blocks (export "test_nested_blocks") (result i32)
    (block (result i32)
      (block (result i32)
        (i32.const 42)
      )
    )
  )
)
