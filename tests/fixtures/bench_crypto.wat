(module
  ;; FNV-1a Hash Function Benchmark
  ;; This demonstrates optimization of a real cryptographic hash function
  ;; Uses: xor, mul, and constant folding

  ;; FNV-1a 32-bit parameters (constants for testing)
  ;; FNV_offset_basis = 2166136261 (0x811c9dc5)
  ;; FNV_prime = 16777619 (0x01000193)

  ;; Hash a single byte with FNV-1a
  ;; hash' = (hash xor byte) * prime
  (func $fnv1a_single_byte (result i32)
    ;; Start with offset basis
    i32.const 2166136261
    ;; XOR with byte value (65 = 'A')
    i32.const 65
    i32.xor
    ;; Multiply by FNV prime
    i32.const 16777619
    i32.mul
  )

  ;; Hash two bytes sequentially
  (func $fnv1a_two_bytes (result i32)
    ;; Hash first byte
    i32.const 2166136261
    i32.const 65  ;; 'A'
    i32.xor
    i32.const 16777619
    i32.mul
    ;; Hash second byte
    i32.const 66  ;; 'B'
    i32.xor
    i32.const 16777619
    i32.mul
  )

  ;; Hash four bytes (simulating "TEST")
  (func $fnv1a_four_bytes (result i32)
    ;; T
    i32.const 2166136261
    i32.const 84
    i32.xor
    i32.const 16777619
    i32.mul
    ;; E
    i32.const 69
    i32.xor
    i32.const 16777619
    i32.mul
    ;; S
    i32.const 83
    i32.xor
    i32.const 16777619
    i32.mul
    ;; T
    i32.const 84
    i32.xor
    i32.const 16777619
    i32.mul
  )

  ;; Constant propagation test: pre-computed partial hash
  (func $fnv1a_precomputed (result i32)
    ;; This should fold: (2166136261 xor 65) * 16777619
    i32.const 2166136261
    i32.const 65
    i32.xor
    i32.const 16777619
    i32.mul
    ;; Then XOR with next byte
    i32.const 66
    i32.xor
    ;; Should fold again
    i32.const 16777619
    i32.mul
  )

  ;; 64-bit FNV-1a variant
  ;; FNV_offset_basis = 14695981039346656037 (0xcbf29ce484222325)
  ;; FNV_prime = 1099511628211 (0x100000001b3)
  (func $fnv1a_64bit (result i64)
    i64.const 14695981039346656037
    i64.const 65  ;; 'A'
    i64.xor
    i64.const 1099511628211
    i64.mul
  )

  ;; Combining multiple hash operations
  (func $fnv1a_combined (result i32)
    ;; Compute hash of 'AB'
    i32.const 2166136261
    i32.const 65
    i32.xor
    i32.const 16777619
    i32.mul
    i32.const 66
    i32.xor
    i32.const 16777619
    i32.mul

    ;; XOR with hash of 'CD'
    i32.const 2166136261
    i32.const 67
    i32.xor
    i32.const 16777619
    i32.mul
    i32.const 68
    i32.xor
    i32.const 16777619
    i32.mul

    i32.xor
  )

  ;; Test algebraic simplification: x XOR 0 = x
  (func $hash_identity (result i32)
    i32.const 2166136261
    i32.const 0
    i32.xor
    i32.const 16777619
    i32.mul
  )

  ;; Test shift-based mixing (common in hash functions)
  (func $hash_mix (result i32)
    i32.const 12345
    ;; Mix bits with shift and xor
    i32.const 12345
    i32.const 16
    i32.shr_u
    i32.xor
    ;; Multiply
    i32.const 16777619
    i32.mul
  )
)
