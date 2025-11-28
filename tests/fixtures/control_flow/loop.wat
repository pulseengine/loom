(module
  ;; Simple loop test
  (func $test_loop (param $n i32) (result i32)
    (local $i i32)
    (local $sum i32)
    (local.set $i (i32.const 0))
    (local.set $sum (i32.const 0))
    (block $exit
      (loop $continue
        ;; Add i to sum
        (local.set $sum
          (i32.add (local.get $sum) (local.get $i))
        )
        ;; Increment i
        (local.set $i
          (i32.add (local.get $i) (i32.const 1))
        )
        ;; Continue if i < n
        (br_if $continue
          (i32.lt_u (local.get $i) (local.get $n))
        )
      )
    )
    (local.get $sum)
  )

  ;; Simple loop with break
  (func $test_loop_with_break (result i32)
    (local $i i32)
    (block $exit
      (loop $continue
        (local.set $i
          (i32.add (local.get $i) (i32.const 1))
        )
        ;; Break if i >= 10
        (br_if $exit
          (i32.ge_u (local.get $i) (i32.const 10))
        )
        (br $continue)
      )
    )
    (local.get $i)
  )
)
