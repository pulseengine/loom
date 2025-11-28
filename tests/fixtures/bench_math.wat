(module
  ;; Mathematical Operations Benchmark
  ;; Demonstrates optimization of arithmetic and division operations

  ;; Polynomial evaluation: 2x^3 + 3x^2 + 4x + 5 at x=3
  ;; Result: 2*27 + 3*9 + 4*3 + 5 = 54 + 27 + 12 + 5 = 98
  (func $polynomial_eval (result i32)
    ;; 2 * 3^3
    i32.const 2
    i32.const 3
    i32.const 3
    i32.mul
    i32.const 3
    i32.mul
    i32.mul
    ;; + 3 * 3^2
    i32.const 3
    i32.const 3
    i32.const 3
    i32.mul
    i32.mul
    i32.add
    ;; + 4 * 3
    i32.const 4
    i32.const 3
    i32.mul
    i32.add
    ;; + 5
    i32.const 5
    i32.add
  )

  ;; Computing factorial with constants: 5! = 120
  (func $factorial_5 (result i32)
    i32.const 1
    i32.const 2
    i32.mul
    i32.const 3
    i32.mul
    i32.const 4
    i32.mul
    i32.const 5
    i32.mul
  )

  ;; Modular arithmetic: (a * b) % m
  ;; (17 * 23) % 100 = 391 % 100 = 91
  (func $modular_mult (result i32)
    i32.const 17
    i32.const 23
    i32.mul
    i32.const 100
    i32.rem_u
  )

  ;; Integer division with remainder check
  ;; 1234 / 56 = 22 remainder 2
  (func $div_with_remainder (result i32)
    ;; Quotient
    i32.const 1234
    i32.const 56
    i32.div_u
    ;; Multiply back
    i32.const 56
    i32.mul
    ;; Add remainder
    i32.const 1234
    i32.const 56
    i32.rem_u
    i32.add
    ;; Should equal original (1234)
  )

  ;; Euclidean algorithm step for GCD
  ;; GCD(48, 18): 48 % 18 = 12
  (func $gcd_step (result i32)
    i32.const 48
    i32.const 18
    i32.rem_u
  )

  ;; Power computation: 2^10 = 1024
  (func $power_of_two (result i32)
    i32.const 2
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
    i32.const 2
    i32.mul
  )

  ;; Fixed-point arithmetic: multiply two Q16.16 numbers
  ;; (2.5 * 0x10000) * (3.25 * 0x10000) / 0x10000
  ;; = 163840 * 212992 / 65536 = 532480
  (func $fixed_point_mult (result i32)
    i32.const 163840  ;; 2.5 in Q16.16
    i32.const 212992  ;; 3.25 in Q16.16
    i32.mul
    i32.const 65536
    i32.div_u
  )

  ;; Average of two numbers without overflow
  ;; (a & b) + ((a ^ b) >> 1)
  ;; For 1000 and 2000: average = 1500
  (func $safe_average (result i32)
    ;; a & b
    i32.const 1000
    i32.const 2000
    i32.and
    ;; (a ^ b) >> 1
    i32.const 1000
    i32.const 2000
    i32.xor
    i32.const 1
    i32.shr_u
    ;; Add them
    i32.add
  )

  ;; Integer square root approximation
  ;; sqrt(100) ≈ 10
  ;; Using Newton's method: one iteration
  ;; x1 = (x0 + n/x0) / 2 with x0 = 10, n = 100
  ;; x1 = (10 + 100/10) / 2 = 10
  (func $sqrt_approx (result i32)
    i32.const 10
    i32.const 100
    i32.const 10
    i32.div_u
    i32.add
    i32.const 2
    i32.div_u
  )

  ;; Percentage calculation: 15% of 200 = 30
  (func $percentage (result i32)
    i32.const 200
    i32.const 15
    i32.mul
    i32.const 100
    i32.div_u
  )

  ;; 64-bit multiplication: large number multiplication
  ;; 1000000 * 2000000 = 2000000000000
  (func $large_mult (result i64)
    i64.const 1000000
    i64.const 2000000
    i64.mul
  )

  ;; 64-bit division: microseconds to seconds
  ;; 5000000 us / 1000000 = 5 seconds
  (func $microseconds_to_seconds (result i64)
    i64.const 5000000
    i64.const 1000000
    i64.div_u
  )

  ;; Signed division with negative numbers
  ;; -100 / 4 = -25
  (func $signed_div_negative (result i32)
    i32.const -100
    i32.const 4
    i32.div_s
  )

  ;; Signed remainder with negative dividend
  ;; -17 % 5 = -2 (sign follows dividend)
  (func $signed_rem_negative (result i32)
    i32.const -17
    i32.const 5
    i32.rem_s
  )

  ;; Mixed operations: (a * b) / c - d
  ;; (12 * 15) / 6 - 10 = 180 / 6 - 10 = 30 - 10 = 20
  (func $mixed_ops (result i32)
    i32.const 12
    i32.const 15
    i32.mul
    i32.const 6
    i32.div_u
    i32.const 10
    i32.sub
  )

  ;; Compute checksum: sum bytes and take modulo
  ;; (1 + 2 + 3 + 4 + 5) % 256 = 15
  (func $checksum (result i32)
    i32.const 1
    i32.const 2
    i32.add
    i32.const 3
    i32.add
    i32.const 4
    i32.add
    i32.const 5
    i32.add
    i32.const 256
    i32.rem_u
  )
)
