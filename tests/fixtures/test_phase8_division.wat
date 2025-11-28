(module
  ;; Test i32.div_s constant folding
  (func $test_div_s (result i32)
    i32.const 20
    i32.const 4
    i32.div_s
  )

  ;; Test i32.div_u constant folding
  (func $test_div_u (result i32)
    i32.const 100
    i32.const 10
    i32.div_u
  )

  ;; Test i32.rem_s constant folding
  (func $test_rem_s (result i32)
    i32.const 17
    i32.const 5
    i32.rem_s
  )

  ;; Test i32.rem_u constant folding
  (func $test_rem_u (result i32)
    i32.const 23
    i32.const 7
    i32.rem_u
  )

  ;; Test i64.div_s constant folding
  (func $test_i64_div_s (result i64)
    i64.const 1000
    i64.const 8
    i64.div_s
  )

  ;; Test i64.div_u constant folding
  (func $test_i64_div_u (result i64)
    i64.const 500
    i64.const 25
    i64.div_u
  )

  ;; Test i64.rem_s constant folding
  (func $test_i64_rem_s (result i64)
    i64.const 100
    i64.const 7
    i64.rem_s
  )

  ;; Test i64.rem_u constant folding
  (func $test_i64_rem_u (result i64)
    i64.const 50
    i64.const 6
    i64.rem_u
  )
)
