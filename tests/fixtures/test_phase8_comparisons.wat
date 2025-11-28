(module
  ;; Test i32.eq constant folding
  (func $test_eq_true (result i32)
    i32.const 42
    i32.const 42
    i32.eq
  )

  (func $test_eq_false (result i32)
    i32.const 10
    i32.const 20
    i32.eq
  )

  ;; Test i32.ne constant folding
  (func $test_ne (result i32)
    i32.const 5
    i32.const 10
    i32.ne
  )

  ;; Test i32.lt_s (signed less than)
  (func $test_lt_s (result i32)
    i32.const -5
    i32.const 10
    i32.lt_s
  )

  ;; Test i32.lt_u (unsigned less than)
  (func $test_lt_u (result i32)
    i32.const 5
    i32.const 10
    i32.lt_u
  )

  ;; Test i32.gt_s (signed greater than)
  (func $test_gt_s (result i32)
    i32.const 20
    i32.const 10
    i32.gt_s
  )

  ;; Test i32.le_s (signed less than or equal)
  (func $test_le_s (result i32)
    i32.const 10
    i32.const 10
    i32.le_s
  )

  ;; Test i32.ge_u (unsigned greater than or equal)
  (func $test_ge_u (result i32)
    i32.const 15
    i32.const 10
    i32.ge_u
  )

  ;; Test i64.eq constant folding
  (func $test_i64_eq (result i32)
    i64.const 1000
    i64.const 1000
    i64.eq
  )

  ;; Test i64.ne constant folding
  (func $test_i64_ne (result i32)
    i64.const 100
    i64.const 200
    i64.ne
  )

  ;; Test i64.lt_s (signed less than)
  (func $test_i64_lt_s (result i32)
    i64.const -1000
    i64.const 500
    i64.lt_s
  )

  ;; Test i64.gt_u (unsigned greater than)
  (func $test_i64_gt_u (result i32)
    i64.const 5000
    i64.const 1000
    i64.gt_u
  )
)
