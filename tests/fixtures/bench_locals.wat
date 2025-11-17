(module
  (func $test_locals (param $x i32) (param $y i32) (result i32)
    (local $tmp1 i32)
    (local $tmp2 i32)
    (local $tmp3 i32)

    ;; Multiple local operations
    (local.set $tmp1 (local.get $x))
    (local.set $tmp2 (local.get $y))
    (local.set $tmp3 (i32.add (local.get $tmp1) (local.get $tmp2)))

    ;; Use tmp3
    (i32.mul (local.get $tmp3) (i32.const 2))
  )

  (func $test_tee (param $a i32) (result i32)
    (local $b i32)
    (local $c i32)

    ;; Test local.tee
    (local.set $b (local.tee $c (i32.add (local.get $a) (i32.const 1))))
    (i32.add (local.get $b) (local.get $c))
  )

  (func $test_constants (result i32)
    (local $x i32)
    (local $y i32)

    ;; Duplicate constants that CSE should optimize
    (local.set $x (i32.const 42))
    (local.set $y (i32.const 42))
    (i32.add (local.get $x) (local.get $y))
  )
)