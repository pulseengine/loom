(module
  ;; Test 1: Redundant copy removal
  ;; Pattern: local.get followed by local.set, dst never used
  (func $redundant_copy (result i32)
    (local $0 i32)
    (local $1 i32)

    ;; Set $0
    (local.set $0 (i32.const 42))

    ;; Redundant copy: $1 = $0, but $1 never used
    (local.get $0)
    (local.set $1)

    ;; Just return $0
    (local.get $0)
  )

  ;; Test 2: Equivalence canonicalization
  ;; Pattern: All uses of $1 should become uses of $0
  (func $equivalence_canon (result i32)
    (local $0 i32)
    (local $1 i32)

    ;; Set $0
    (local.set $0 (i32.const 100))

    ;; Create equivalence: $1 ≡ $0
    (local.get $0)
    (local.set $1)

    ;; These uses of $1 should become uses of $0
    (local.get $1)
    (local.get $1)
    (i32.add)
  )

  ;; Test 3: Equivalence chain
  ;; Pattern: $1 ≡ $0, $2 ≡ $1, so $2 should use $0
  (func $equivalence_chain (result i32)
    (local $0 i32)
    (local $1 i32)
    (local $2 i32)

    (local.set $0 (i32.const 10))

    ;; $1 ≡ $0
    (local.get $0)
    (local.set $1)

    ;; $2 ≡ $1 ≡ $0
    (local.get $1)
    (local.set $2)

    ;; Should all become $0
    (local.get $2)
  )

  ;; Test 4: Multiple redundant copies
  (func $multiple_copies (result i32)
    (local $0 i32)
    (local $1 i32)
    (local $2 i32)
    (local $3 i32)

    (local.set $0 (i32.const 5))

    ;; Redundant: $1 never used
    (local.get $0)
    (local.set $1)

    ;; Redundant: $2 never used
    (local.get $0)
    (local.set $2)

    ;; Redundant: $3 never used
    (local.get $0)
    (local.set $3)

    (local.get $0)
  )

  ;; Test 5: Nested blocks with locals
  (func $nested_blocks (result i32)
    (local $0 i32)
    (local $1 i32)

    (local.set $0 (i32.const 20))

    (block
      ;; Equivalence inside block
      (local.get $0)
      (local.set $1)

      ;; Use should be canonicalized
      (local.get $1)
      (drop)
    )

    (local.get $0)
  )

  ;; Test 6: Used copy (should NOT be removed)
  (func $used_copy (result i32)
    (local $0 i32)
    (local $1 i32)

    (local.set $0 (i32.const 15))

    ;; Copy to $1 (will be used)
    (local.get $0)
    (local.set $1)

    ;; Both are used
    (local.get $0)
    (local.get $1)
    (i32.add)
  )
)
