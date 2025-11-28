(module
  ;; Test i32.sub constant folding
  (func $test_sub (result i32)
    i32.const 100
    i32.const 42
    i32.sub
  )

  ;; Test i32.mul constant folding
  (func $test_mul (result i32)
    i32.const 6
    i32.const 7
    i32.mul
  )

  ;; Test algebraic: x + 0 = x
  (func $test_add_zero (result i32)
    i32.const 42
    i32.const 0
    i32.add
  )

  ;; Test algebraic: x * 1 = x
  (func $test_mul_one (result i32)
    i32.const 99
    i32.const 1
    i32.mul
  )

  ;; Test algebraic: x * 0 = 0
  (func $test_mul_zero (result i32)
    i32.const 999
    i32.const 0
    i32.mul
  )

  ;; Test i64.add constant folding
  (func $test_i64_add (result i64)
    i64.const 1000
    i64.const 2000
    i64.add
  )

  ;; Test i64.mul constant folding
  (func $test_i64_mul (result i64)
    i64.const 10
    i64.const 20
    i64.mul
  )
)
