(module
  (func $add_constants (result i32)
    i32.const 10
    i32.const 32
    i32.add
  )

  (func $multiply (param $x i32) (result i32)
    local.get $x
    i32.const 2
    i32.mul
  )

  (func $test_locals (result i32)
    (local $temp i32)
    i32.const 42
    local.set $temp
    local.get $temp
  )
)