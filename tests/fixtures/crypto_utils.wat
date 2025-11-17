;; Cryptographic utilities with bit manipulation
;; Demonstrates: strength reduction, bitwise optimizations, loop opportunities

(module
  (memory 1)

  ;; Rotate left 32-bit (common in crypto)
  (func $rotl32 (export "rotl32")
    (param $value i32)
    (param $shift i32)
    (result i32)

    ;; rotl(x, n) = (x << n) | (x >> (32 - n))
    (i32.or
      (i32.shl
        (local.get $value)
        (local.get $shift)
      )
      (i32.shr_u
        (local.get $value)
        (i32.sub
          (i32.const 32)
          (local.get $shift)
        )
      )
    )
  )

  ;; XOR cipher (simple encryption)
  (func $xor_cipher (export "xor_cipher")
    (param $offset i32)
    (param $length i32)
    (param $key i32)

    (local $i i32)
    (local $addr i32)
    (local $value i32)

    ;; Loop through memory
    (local.set $i (i32.const 0))
    (block $break
      (loop $continue
        ;; Break if i >= length
        (br_if $break (i32.ge_u (local.get $i) (local.get $length)))

        ;; Calculate address (strength reduction opportunity: i * 4)
        (local.set $addr
          (i32.add
            (local.get $offset)
            (i32.mul (local.get $i) (i32.const 4))
          )
        )

        ;; Load, XOR with key, store
        (local.set $value (i32.load (local.get $addr)))
        (local.set $value (i32.xor (local.get $value) (local.get $key)))
        (i32.store (local.get $addr) (local.get $value))

        ;; i++
        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)
      )
    )
  )

  ;; Bit counting (population count)
  (func $popcount (export "popcount")
    (param $x i32)
    (result i32)
    (local $count i32)

    (local.set $count (i32.const 0))

    (block $break
      (loop $continue
        ;; Break if x == 0
        (br_if $break (i32.eqz (local.get $x)))

        ;; count += x & 1
        (local.set $count
          (i32.add
            (local.get $count)
            (i32.and (local.get $x) (i32.const 1))
          )
        )

        ;; x >>= 1
        (local.set $x (i32.shr_u (local.get $x) (i32.const 1)))

        (br $continue)
      )
    )

    (local.get $count)
  )

  ;; Hash function (simple FNV-1a style)
  (func $hash (export "hash")
    (param $data i32)
    (param $length i32)
    (result i32)
    (local $hash i32)
    (local $i i32)
    (local $byte i32)

    ;; FNV offset basis (constant)
    (local.set $hash (i32.const 2166136261))

    ;; Loop through bytes
    (local.set $i (i32.const 0))
    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $i) (local.get $length)))

        ;; Load byte
        (local.set $byte (i32.load8_u (i32.add (local.get $data) (local.get $i))))

        ;; XOR with byte
        (local.set $hash (i32.xor (local.get $hash) (local.get $byte)))

        ;; Multiply by FNV prime (strength reduction: * 16777619)
        ;; Note: Not a power of 2, but can still be optimized
        (local.set $hash (i32.mul (local.get $hash) (i32.const 16777619)))

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)
      )
    )

    (local.get $hash)
  )

  ;; Bit reversal (32-bit)
  (func $reverse_bits (export "reverse_bits")
    (param $x i32)
    (result i32)
    (local $result i32)
    (local $i i32)

    (local.set $result (i32.const 0))
    (local.set $i (i32.const 0))

    (block $break
      (loop $continue
        (br_if $break (i32.ge_u (local.get $i) (i32.const 32)))

        ;; result = (result << 1) | (x & 1)
        (local.set $result
          (i32.or
            (i32.shl (local.get $result) (i32.const 1))
            (i32.and (local.get $x) (i32.const 1))
          )
        )

        ;; x >>= 1
        (local.set $x (i32.shr_u (local.get $x) (i32.const 1)))

        (local.set $i (i32.add (local.get $i) (i32.const 1)))
        (br $continue)
      )
    )

    (local.get $result)
  )

  ;; Parity check (even/odd number of 1 bits)
  (func $parity (export "parity")
    (param $x i32)
    (result i32)
    (local $result i32)

    (local.set $result (i32.const 0))

    (block $break
      (loop $continue
        (br_if $break (i32.eqz (local.get $x)))

        ;; result ^= 1
        (local.set $result (i32.xor (local.get $result) (i32.const 1)))

        ;; x &= x - 1 (clear lowest set bit)
        (local.set $x
          (i32.and
            (local.get $x)
            (i32.sub (local.get $x) (i32.const 1))
          )
        )

        (br $continue)
      )
    )

    (local.get $result)
  )

  ;; Mix bits (used in hash functions)
  (func $mix_bits (export "mix_bits")
    (param $x i32)
    (result i32)

    ;; Multiple bitwise operations that can be optimized
    (local.set $x (i32.xor (local.get $x) (i32.shr_u (local.get $x) (i32.const 16))))
    (local.set $x (i32.mul (local.get $x) (i32.const 2147483629)))  ;; Large prime
    (local.set $x (i32.xor (local.get $x) (i32.shr_u (local.get $x) (i32.const 15))))
    (local.set $x (i32.mul (local.get $x) (i32.const 8)))  ;; Strength reduction: * 8 → << 3
    (local.set $x (i32.xor (local.get $x) (i32.shr_u (local.get $x) (i32.const 16))))

    ;; Identity operations (algebraic simplification)
    (local.set $x (i32.or (local.get $x) (i32.const 0)))   ;; x | 0 → x
    (local.set $x (i32.and (local.get $x) (i32.const -1))) ;; x & -1 → x

    (local.get $x)
  )

  ;; Check if power of 2
  (func $is_power_of_two (export "is_power_of_two")
    (param $x i32)
    (result i32)

    ;; x != 0 && (x & (x - 1)) == 0
    (i32.and
      (i32.ne (local.get $x) (i32.const 0))
      (i32.eqz
        (i32.and
          (local.get $x)
          (i32.sub (local.get $x) (i32.const 1))
        )
      )
    )
  )

  ;; Align to power of 2 boundary
  (func $align (export "align")
    (param $value i32)
    (param $alignment i32)
    (result i32)

    ;; Assumes alignment is power of 2
    ;; result = (value + alignment - 1) & ~(alignment - 1)
    (i32.and
      (i32.add
        (local.get $value)
        (i32.sub (local.get $alignment) (i32.const 1))
      )
      (i32.xor
        (i32.sub (local.get $alignment) (i32.const 1))
        (i32.const -1)
      )
    )
  )
)
