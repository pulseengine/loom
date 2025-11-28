(module
  ;; Comparison Operations Benchmark
  ;; Demonstrates optimization of comparison chains and boolean logic

  ;; Range check: 5 <= x < 10 where x = 7 (should be true)
  (func $in_range (result i32)
    ;; x >= 5
    i32.const 7
    i32.const 5
    i32.ge_u
    ;; x < 10
    i32.const 7
    i32.const 10
    i32.lt_u
    ;; Both must be true (1 & 1 = 1)
    i32.and
  )

  ;; Bounds checking: ensure index is valid
  ;; 0 <= index < length, where index=3, length=10
  (func $bounds_check (result i32)
    ;; index >= 0 (always true for unsigned, but let's check signed)
    i32.const 3
    i32.const 0
    i32.ge_s
    ;; index < length
    i32.const 3
    i32.const 10
    i32.lt_s
    i32.and
  )

  ;; Three-way comparison: spaceship operator <=>
  ;; Returns: -1 if a < b, 0 if a == b, 1 if a > b
  ;; Compare 10 and 20 (should return -1)
  (func $three_way_compare (result i32)
    ;; if a < b, return -1
    i32.const 10
    i32.const 20
    i32.lt_s
    ;; if true (1), multiply by -1 to get -1; else 0
    i32.const -1
    i32.mul
    ;; Add (a > b ? 1 : 0)
    i32.const 10
    i32.const 20
    i32.gt_s
    i32.add
  )

  ;; Check if character is uppercase letter (A-Z)
  ;; 'G' (71) should be in range
  (func $is_uppercase (result i32)
    i32.const 71  ;; 'G'
    ;; >= 'A' (65)
    i32.const 65
    i32.ge_u
    i32.const 71
    ;; <= 'Z' (90)
    i32.const 90
    i32.le_u
    i32.and
  )

  ;; Check if number is even: (x % 2) == 0
  ;; 42 should be even (true)
  (func $is_even (result i32)
    i32.const 42
    i32.const 2
    i32.rem_u
    i32.const 0
    i32.eq
  )

  ;; Check if number is odd: (x % 2) != 0
  ;; 43 should be odd (true)
  (func $is_odd (result i32)
    i32.const 43
    i32.const 2
    i32.rem_u
    i32.const 0
    i32.ne
  )

  ;; Sign test: check if negative
  ;; -42 < 0 should be true
  (func $is_negative (result i32)
    i32.const -42
    i32.const 0
    i32.lt_s
  )

  ;; Check if x is a power of 2
  ;; (x != 0) && ((x & (x - 1)) == 0)
  ;; Test with 16 (should be true)
  (func $is_power_of_two_check (result i32)
    ;; x != 0
    i32.const 16
    i32.const 0
    i32.ne
    ;; (x & (x - 1)) == 0
    i32.const 16
    i32.const 16
    i32.const 1
    i32.sub
    i32.and
    i32.const 0
    i32.eq
    ;; Both conditions
    i32.and
  )

  ;; Maximum of two numbers using comparison
  ;; max(42, 17) = 42
  ;; Branchless: (a > b) ? a : b = a * (a > b) + b * (a <= b)
  (func $max_branchless (result i32)
    ;; Check a > b
    i32.const 42
    i32.const 17
    i32.gt_s
    ;; Multiply condition by a
    i32.const 42
    i32.mul
    ;; Check a <= b
    i32.const 42
    i32.const 17
    i32.le_s
    ;; Multiply condition by b
    i32.const 17
    i32.mul
    ;; Add results
    i32.add
  )

  ;; Minimum of two numbers
  ;; min(42, 17) = 17
  (func $min_branchless (result i32)
    ;; Check a < b
    i32.const 42
    i32.const 17
    i32.lt_s
    ;; Multiply condition by a
    i32.const 42
    i32.mul
    ;; Check a >= b
    i32.const 42
    i32.const 17
    i32.ge_s
    ;; Multiply condition by b
    i32.const 17
    i32.mul
    ;; Add results
    i32.add
  )

  ;; Clamp value to range [min, max]
  ;; Simplified: just check if 25 is in range [0, 100]
  (func $clamp_in_range (result i32)
    ;; Since 25 is already in range, just verify
    ;; Check: value >= min
    i32.const 25
    i32.const 0
    i32.ge_s
    ;; Check: value <= max
    i32.const 25
    i32.const 100
    i32.le_s
    ;; Both must be true for value to be returned as-is
    i32.and
    ;; If in range (1), return value (25); else this is just validation
    i32.const 25
    i32.mul
  )

  ;; Equality chain: check if (5 == 5) == 1
  ;; 5 == 5 produces 1, then 1 == 1 produces 1 (true)
  (func $equality_chain (result i32)
    ;; First compare 5 == 5
    i32.const 5
    i32.const 5
    i32.eq
    ;; Result (1) compared to 1
    i32.const 1
    i32.eq
  )

  ;; Inequality check: a != b
  ;; 10 != 20 should be true
  (func $not_equal (result i32)
    i32.const 10
    i32.const 20
    i32.ne
  )

  ;; Complex boolean: (a && b) || (c && d)
  ;; (true && false) || (true && true) = false || true = true
  (func $complex_boolean (result i32)
    ;; a && b
    i32.const 1
    i32.const 0
    i32.and
    ;; c && d
    i32.const 1
    i32.const 1
    i32.and
    ;; OR them
    i32.or
  )

  ;; 64-bit comparison: large number check
  ;; 9999999999 > 1000000000 should be true
  (func $large_compare (result i32)
    i64.const 9999999999
    i64.const 1000000000
    i64.gt_u
  )

  ;; Signed vs unsigned: -1 as unsigned is very large
  ;; -1 < 0 (signed) = true
  ;; -1 < 0 (unsigned) = false (0xFFFFFFFF is large)
  (func $signed_comparison (result i32)
    i32.const -1
    i32.const 0
    i32.lt_s
  )

  (func $unsigned_comparison (result i32)
    i32.const -1
    i32.const 0
    i32.lt_u
  )

  ;; Alphabetic comparison: 'A' < 'Z'
  (func $char_compare (result i32)
    i32.const 65  ;; 'A'
    i32.const 90  ;; 'Z'
    i32.lt_u
  )

  ;; Zero check: x == 0
  (func $is_zero (result i32)
    i32.const 0
    i32.const 0
    i32.eq
  )

  ;; Non-zero check: x != 0
  (func $is_nonzero (result i32)
    i32.const 42
    i32.const 0
    i32.ne
  )
)
