;; Fibonacci number calculation
;; Classic recursive implementation with opportunities for optimization

(module
  ;; Recursive fibonacci function
  (func $fib (export "fib") (param $n i32) (result i32)
    (local $temp i32)

    ;; Base cases
    (if (result i32)
      (i32.le_u (local.get $n) (i32.const 1))
      (then
        ;; fib(0) = 0, fib(1) = 1
        (local.get $n)
      )
      (else
        ;; fib(n) = fib(n-1) + fib(n-2)
        (i32.add
          ;; fib(n-1)
          (call $fib
            (i32.sub (local.get $n) (i32.const 1))
          )
          ;; fib(n-2)
          (call $fib
            (i32.sub (local.get $n) (i32.const 2))
          )
        )
      )
    )
  )

  ;; Iterative fibonacci (more efficient)
  (func $fib_iterative (export "fib_iterative") (param $n i32) (result i32)
    (local $a i32)
    (local $b i32)
    (local $temp i32)
    (local $i i32)

    ;; Initialize
    (local.set $a (i32.const 0))
    (local.set $b (i32.const 1))
    (local.set $i (i32.const 0))

    ;; Loop n times
    (block $break
      (loop $continue
        ;; Break if i >= n
        (br_if $break
          (i32.ge_u (local.get $i) (local.get $n))
        )

        ;; temp = a
        (local.set $temp (local.get $a))

        ;; a = b
        (local.set $a (local.get $b))

        ;; b = temp + b
        (local.set $b
          (i32.add (local.get $temp) (local.get $b))
        )

        ;; i++
        (local.set $i
          (i32.add (local.get $i) (i32.const 1))
        )

        (br $continue)
      )
    )

    (local.get $a)
  )
)
