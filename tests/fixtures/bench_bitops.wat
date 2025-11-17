(module
  (func $test_bitwise (param $x i32) (param $y i32) (result i32)
    (local $a i32)
    (local $b i32)

    ;; Bitwise operations
    (local.set $a (i32.and (local.get $x) (local.get $y)))
    (local.set $b (i32.or (local.get $x) (local.get $y)))

    (i32.xor (local.get $a) (local.get $b))
  )

  (func $test_shifts (param $val i32) (result i32)
    (local $shifted i32)

    ;; Shift operations
    (local.set $shifted (i32.shl (local.get $val) (i32.const 2)))
    (i32.shr_u (local.get $shifted) (i32.const 1))
  )

  (func $test_bitwise_constants (result i32)
    ;; Test constant folding for bitwise ops
    (i32.and (i32.const 255) (i32.const 15))
  )

  (func $test_xor_same (param $x i32) (result i32)
    ;; x ^ x should become 0
    (i32.xor (local.get $x) (local.get $x))
  )
)