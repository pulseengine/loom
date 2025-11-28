(module
  ;; Test i32.and constant folding
  (func $test_and (result i32)
    i32.const 0xFF
    i32.const 0x0F
    i32.and
  )

  ;; Test i32.or constant folding
  (func $test_or (result i32)
    i32.const 0xF0
    i32.const 0x0F
    i32.or
  )

  ;; Test i32.xor constant folding
  (func $test_xor (result i32)
    i32.const 0xFF
    i32.const 0x0F
    i32.xor
  )

  ;; Test i32.shl constant folding
  (func $test_shl (result i32)
    i32.const 8
    i32.const 2
    i32.shl
  )

  ;; Test i32.shr_s constant folding (arithmetic shift)
  (func $test_shr_s (result i32)
    i32.const -16
    i32.const 2
    i32.shr_s
  )

  ;; Test i32.shr_u constant folding (logical shift)
  (func $test_shr_u (result i32)
    i32.const 16
    i32.const 2
    i32.shr_u
  )

  ;; Test algebraic: x & 0 = 0
  (func $test_and_zero (result i32)
    i32.const 999
    i32.const 0
    i32.and
  )

  ;; Test algebraic: x | 0 = x
  (func $test_or_zero (result i32)
    i32.const 42
    i32.const 0
    i32.or
  )

  ;; Test algebraic: x ^ 0 = x
  (func $test_xor_zero (result i32)
    i32.const 123
    i32.const 0
    i32.xor
  )

  ;; Test algebraic: x << 0 = x
  (func $test_shl_zero (result i32)
    i32.const 99
    i32.const 0
    i32.shl
  )

  ;; Test i64.and constant folding
  (func $test_i64_and (result i64)
    i64.const 0xFFFF
    i64.const 0x00FF
    i64.and
  )

  ;; Test i64.shl constant folding
  (func $test_i64_shl (result i64)
    i64.const 10
    i64.const 3
    i64.shl
  )
)
