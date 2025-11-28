(module
  ;; Phase 10: Unary Operations and Select Benchmark
  ;; Tests i32.eqz, i64.eqz, i32.clz, i32.ctz, i32.popcnt, i64.clz, i64.ctz, i64.popcnt, select

  ;; ========================================
  ;; i32.eqz - Test if zero (16,563 uses in component)
  ;; ========================================

  (func $is_zero (result i32)
    i32.const 0
    i32.eqz
  )

  (func $is_not_zero (result i32)
    i32.const 42
    i32.eqz
  )

  (func $combined_zero_check (result i32)
    ;; Check if (100 - 100) is zero
    i32.const 100
    i32.const 100
    i32.sub
    i32.eqz
  )

  ;; ========================================
  ;; i64.eqz - Test if zero for 64-bit
  ;; ========================================

  (func $is_zero_64 (result i32)
    i64.const 0
    i64.eqz
  )

  (func $is_not_zero_64 (result i32)
    i64.const 999999999999
    i64.eqz
  )

  ;; ========================================
  ;; i32.clz - Count leading zeros (324 uses)
  ;; ========================================

  (func $clz_zero (result i32)
    i32.const 0
    i32.clz
  )

  (func $clz_one (result i32)
    i32.const 1
    i32.clz
  )

  (func $clz_max (result i32)
    i32.const 0xFFFFFFFF
    i32.clz
  )

  (func $clz_power_of_two (result i32)
    i32.const 256      ;; 0x100, has 24 leading zeros
    i32.clz
  )

  ;; ========================================
  ;; i32.ctz - Count trailing zeros
  ;; ========================================

  (func $ctz_zero (result i32)
    i32.const 0
    i32.ctz
  )

  (func $ctz_one (result i32)
    i32.const 1
    i32.ctz
  )

  (func $ctz_power_of_two (result i32)
    i32.const 256      ;; 0x100, has 8 trailing zeros
    i32.ctz
  )

  ;; ========================================
  ;; i32.popcnt - Count set bits
  ;; ========================================

  (func $popcnt_zero (result i32)
    i32.const 0
    i32.popcnt
  )

  (func $popcnt_one (result i32)
    i32.const 1
    i32.popcnt
  )

  (func $popcnt_all_bits (result i32)
    i32.const 0xFFFFFFFF
    i32.popcnt
  )

  (func $popcnt_nibble (result i32)
    i32.const 0x0F     ;; 4 bits set
    i32.popcnt
  )

  ;; ========================================
  ;; i64 unary operations
  ;; ========================================

  (func $clz_64 (result i64)
    i64.const 256
    i64.clz
  )

  (func $ctz_64 (result i64)
    i64.const 256
    i64.ctz
  )

  (func $popcnt_64 (result i64)
    i64.const 0xFF
    i64.popcnt
  )

  ;; ========================================
  ;; select - Conditional value selection (1,680 uses)
  ;; ========================================

  (func $select_true (result i32)
    i32.const 100      ;; true value
    i32.const 200      ;; false value
    i32.const 1        ;; condition (non-zero = true)
    select
  )

  (func $select_false (result i32)
    i32.const 100      ;; true value
    i32.const 200      ;; false value
    i32.const 0        ;; condition (zero = false)
    select
  )

  (func $select_computed_cond (result i32)
    i32.const 42       ;; true value
    i32.const 99       ;; false value
    i32.const 10
    i32.const 10
    i32.eq             ;; condition: 10 == 10 = 1 (true)
    select
  )

  ;; ========================================
  ;; Complex combinations
  ;; ========================================

  (func $is_power_of_two (result i32)
    ;; A number is a power of 2 if it has exactly 1 bit set
    ;; This can be checked with popcnt == 1
    i32.const 256
    i32.popcnt
    i32.const 1
    i32.eq
  )

  (func $conditional_bit_count (result i32)
    ;; If 16 has zero trailing zeros, return 100, else return the actual count
    i32.const 100      ;; true value
    i32.const 16
    i32.ctz           ;; false value: actual count (4)
    i32.const 16
    i32.ctz
    i32.eqz           ;; condition: is count zero?
    select
  )

  (func $optimized_branch_elimination (result i32)
    ;; This should optimize to constant 999
    ;; because (5 == 5) is always true
    i32.const 999      ;; true value
    i32.const 111      ;; false value
    i32.const 5
    i32.const 5
    i32.eq
    select
  )
)
