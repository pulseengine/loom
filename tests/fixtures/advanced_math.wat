;; Advanced math operations with multiple optimization opportunities
;; Demonstrates: strength reduction, constant folding, CSE, algebraic simplification

(module
  ;; Complex calculation with many optimization opportunities
  (func $calculate_score (export "calculate_score")
    (param $base i32)
    (param $multiplier i32)
    (param $bonus i32)
    (result i32)
    (local $temp1 i32)
    (local $temp2 i32)

    ;; Multiply by 8 (strength reduction: * 8 → << 3)
    (local.set $temp1
      (i32.mul (local.get $base) (i32.const 8))
    )

    ;; Divide by 4 (strength reduction: / 4 → >> 2)
    (local.set $temp1
      (i32.div_u (local.get $temp1) (i32.const 4))
    )

    ;; Same calculation repeated (CSE opportunity)
    (local.set $temp2
      (i32.mul (local.get $multiplier) (i32.const 16))
    )

    ;; Duplicate of previous (CSE opportunity)
    (local.set $temp2
      (i32.add
        (local.get $temp2)
        (i32.mul (local.get $multiplier) (i32.const 16))
      )
    )

    ;; Modulo by 32 (strength reduction: % 32 → & 31)
    (local.set $temp2
      (i32.rem_u (local.get $temp2) (i32.const 32))
    )

    ;; Constant folding opportunity
    (local.set $temp1
      (i32.add
        (local.get $temp1)
        (i32.add
          (i32.const 100)
          (i32.const 50)
        )
      )
    )

    ;; Bitwise identity operations (algebraic simplification)
    (local.set $temp1
      (i32.or (local.get $temp1) (i32.const 0))  ;; x | 0 → x
    )

    (local.set $temp1
      (i32.and (local.get $temp1) (i32.const -1))  ;; x & -1 → x
    )

    (local.set $temp1
      (i32.xor (local.get $temp1) (i32.const 0))  ;; x ^ 0 → x
    )

    ;; Final result
    (i32.add
      (i32.add (local.get $temp1) (local.get $temp2))
      (local.get $bonus)
    )
  )

  ;; Power of 2 operations (all strength reduction opportunities)
  (func $power_of_two_ops (export "power_of_two_ops")
    (param $x i32)
    (result i32)
    (local $result i32)

    ;; Multiply by powers of 2
    (local.set $result (i32.mul (local.get $x) (i32.const 2)))   ;; → shl 1
    (local.set $result (i32.mul (local.get $x) (i32.const 4)))   ;; → shl 2
    (local.set $result (i32.mul (local.get $x) (i32.const 16)))  ;; → shl 4
    (local.set $result (i32.mul (local.get $x) (i32.const 256))) ;; → shl 8

    ;; Divide by powers of 2
    (local.set $result (i32.div_u (local.get $x) (i32.const 2)))   ;; → shr 1
    (local.set $result (i32.div_u (local.get $x) (i32.const 8)))   ;; → shr 3
    (local.set $result (i32.div_u (local.get $x) (i32.const 64)))  ;; → shr 6
    (local.set $result (i32.div_u (local.get $x) (i32.const 128))) ;; → shr 7

    ;; Modulo by powers of 2
    (local.set $result (i32.rem_u (local.get $x) (i32.const 4)))   ;; → and 3
    (local.set $result (i32.rem_u (local.get $x) (i32.const 8)))   ;; → and 7
    (local.set $result (i32.rem_u (local.get $x) (i32.const 16)))  ;; → and 15
    (local.set $result (i32.rem_u (local.get $x) (i32.const 256))) ;; → and 255

    (local.get $result)
  )

  ;; Nested constant expressions (deep constant folding)
  (func $nested_constants (export "nested_constants")
    (result i32)

    ;; Should fold to single constant
    (i32.add
      (i32.mul
        (i32.add (i32.const 10) (i32.const 20))
        (i32.const 2)
      )
      (i32.div_u
        (i32.sub (i32.const 100) (i32.const 40))
        (i32.const 4)
      )
    )
    ;; = (30 * 2) + (60 / 4) = 60 + 15 = 75
  )

  ;; Repeated subexpressions (CSE opportunities)
  (func $common_subexpressions (export "common_subexpressions")
    (param $a i32)
    (param $b i32)
    (result i32)
    (local $result i32)

    ;; Calculate (a * 7) + (b * 3) three times
    (local.set $result
      (i32.add
        (i32.mul (local.get $a) (i32.const 7))
        (i32.mul (local.get $b) (i32.const 3))
      )
    )

    ;; Same expression again (should be CSE'd)
    (local.set $result
      (i32.add
        (local.get $result)
        (i32.add
          (i32.mul (local.get $a) (i32.const 7))
          (i32.mul (local.get $b) (i32.const 3))
        )
      )
    )

    ;; And again!
    (local.set $result
      (i32.add
        (local.get $result)
        (i32.add
          (i32.mul (local.get $a) (i32.const 7))
          (i32.mul (local.get $b) (i32.const 3))
        )
      )
    )

    (local.get $result)
  )

  ;; Mixed operations showcasing all optimizations
  (func $showcase (export "showcase")
    (param $value i32)
    (result i32)
    (local $temp i32)

    ;; Strength reduction + constant folding + algebraic simplification
    (local.set $temp
      (i32.mul
        (i32.add
          (local.get $value)
          (i32.const 0)  ;; x + 0 → x
        )
        (i32.const 8)  ;; * 8 → << 3
      )
    )

    ;; More optimizations
    (i32.add
      (i32.rem_u (local.get $temp) (i32.const 64))  ;; % 64 → & 63
      (i32.mul
        (i32.const 5)
        (i32.const 7)  ;; Constant folding: 5 * 7 = 35
      )
    )
  )
)
