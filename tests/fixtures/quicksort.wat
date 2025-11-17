;; Quicksort implementation
;; Complex control flow with recursion

(module
  (memory 1)

  ;; Swap two elements in array
  (func $swap (param $arr_offset i32) (param $i i32) (param $j i32)
    (local $temp i32)
    (local $i_addr i32)
    (local $j_addr i32)

    ;; Calculate addresses
    ;; i_addr = arr_offset + i * 4  (strength reduction opportunity)
    (local.set $i_addr
      (i32.add
        (local.get $arr_offset)
        (i32.mul (local.get $i) (i32.const 4))
      )
    )

    ;; j_addr = arr_offset + j * 4
    (local.set $j_addr
      (i32.add
        (local.get $arr_offset)
        (i32.mul (local.get $j) (i32.const 4))
      )
    )

    ;; temp = arr[i]
    (local.set $temp (i32.load (local.get $i_addr)))

    ;; arr[i] = arr[j]
    (i32.store
      (local.get $i_addr)
      (i32.load (local.get $j_addr))
    )

    ;; arr[j] = temp
    (i32.store
      (local.get $j_addr)
      (local.get $temp)
    )
  )

  ;; Partition function for quicksort
  (func $partition (param $arr_offset i32) (param $low i32) (param $high i32) (result i32)
    (local $pivot i32)
    (local $i i32)
    (local $j i32)
    (local $pivot_addr i32)
    (local $j_addr i32)

    ;; Get pivot value (last element)
    ;; pivot_addr = arr_offset + high * 4
    (local.set $pivot_addr
      (i32.add
        (local.get $arr_offset)
        (i32.mul (local.get $high) (i32.const 4))
      )
    )
    (local.set $pivot (i32.load (local.get $pivot_addr)))

    ;; i = low - 1
    (local.set $i (i32.sub (local.get $low) (i32.const 1)))

    ;; for j = low to high-1
    (local.set $j (local.get $low))
    (block $break
      (loop $continue
        ;; Break if j >= high
        (br_if $break (i32.ge_u (local.get $j) (local.get $high)))

        ;; if arr[j] < pivot
        (local.set $j_addr
          (i32.add
            (local.get $arr_offset)
            (i32.mul (local.get $j) (i32.const 4))
          )
        )

        (if (i32.lt_s (i32.load (local.get $j_addr)) (local.get $pivot))
          (then
            ;; i++
            (local.set $i (i32.add (local.get $i) (i32.const 1)))

            ;; swap(arr, i, j)
            (call $swap (local.get $arr_offset) (local.get $i) (local.get $j))
          )
        )

        ;; j++
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $continue)
      )
    )

    ;; swap(arr, i+1, high)
    (call $swap
      (local.get $arr_offset)
      (i32.add (local.get $i) (i32.const 1))
      (local.get $high)
    )

    ;; return i + 1
    (i32.add (local.get $i) (i32.const 1))
  )

  ;; Quicksort recursive function
  (func $quicksort (export "quicksort") (param $arr_offset i32) (param $low i32) (param $high i32)
    (local $pivot_index i32)

    ;; if low < high
    (if (i32.lt_s (local.get $low) (local.get $high))
      (then
        ;; pivot_index = partition(arr, low, high)
        (local.set $pivot_index
          (call $partition (local.get $arr_offset) (local.get $low) (local.get $high))
        )

        ;; quicksort(arr, low, pivot_index - 1)
        (call $quicksort
          (local.get $arr_offset)
          (local.get $low)
          (i32.sub (local.get $pivot_index) (i32.const 1))
        )

        ;; quicksort(arr, pivot_index + 1, high)
        (call $quicksort
          (local.get $arr_offset)
          (i32.add (local.get $pivot_index) (i32.const 1))
          (local.get $high)
        )
      )
    )
  )
)
