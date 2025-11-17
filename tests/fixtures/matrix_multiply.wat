;; Matrix multiplication example
;; Shows loop optimizations and CSE opportunities

(module
  (memory 1)

  ;; Multiply two 2x2 matrices
  ;; Input: matrices A and B at offsets 0 and 16
  ;; Output: result matrix C at offset 32
  (func $matrix_multiply_2x2 (export "matmul_2x2")
    (local $i i32)
    (local $j i32)
    (local $k i32)
    (local $sum i32)
    (local $a_offset i32)
    (local $b_offset i32)
    (local $c_offset i32)

    ;; For i = 0 to 1
    (local.set $i (i32.const 0))
    (block $i_break
      (loop $i_continue
        (br_if $i_break (i32.ge_u (local.get $i) (i32.const 2)))

        ;; For j = 0 to 1
        (local.set $j (i32.const 0))
        (block $j_break
          (loop $j_continue
            (br_if $j_break (i32.ge_u (local.get $j) (i32.const 2)))

            ;; sum = 0
            (local.set $sum (i32.const 0))

            ;; For k = 0 to 1
            (local.set $k (i32.const 0))
            (block $k_break
              (loop $k_continue
                (br_if $k_break (i32.ge_u (local.get $k) (i32.const 2)))

                ;; a_offset = i * 2 + k (can be strength-reduced)
                (local.set $a_offset
                  (i32.add
                    (i32.mul (local.get $i) (i32.const 2))
                    (local.get $k)
                  )
                )

                ;; b_offset = k * 2 + j
                (local.set $b_offset
                  (i32.add
                    (i32.mul (local.get $k) (i32.const 2))
                    (local.get $j)
                  )
                )

                ;; sum += A[a_offset] * B[b_offset]
                (local.set $sum
                  (i32.add
                    (local.get $sum)
                    (i32.mul
                      (i32.load (i32.shl (local.get $a_offset) (i32.const 2)))
                      (i32.load (i32.add (i32.const 16) (i32.shl (local.get $b_offset) (i32.const 2))))
                    )
                  )
                )

                ;; k++
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $k_continue)
              )
            )

            ;; C[i*2+j] = sum
            (local.set $c_offset
              (i32.add
                (i32.mul (local.get $i) (i32.const 2))
                (local.get $j)
              )
            )
            (i32.store
              (i32.add (i32.const 32) (i32.shl (local.get $c_offset) (i32.const 2)))
              (local.get $sum)
            )

            ;; j++
            (local.set $j (i32.add (local.get $j) (i32.const 1)))
            (br $j_continue)
          )
        )

        ;; i++
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $i_continue)
      )
    )
  )
)
