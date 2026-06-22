;; #219 — seam SROA / scalar-forwarding fixture.
;;
;; Models the gale C<->Rust "decide" seam: `decide` returns two i32 scalars
;; {action, new_count} PACKED into a single i64 (the u64-return ABI), and the
;; shim immediately UNPACKS them back to scalars. After `--passes inline`,
;; `decide` is inlined into `shim`, so on one value we get:
;;
;;   pack:   i64.extend_i32_u ; i64.shl ; i64.or       (build the u64)
;;   carry:  local.set $t / local.get $t               (the dead i64 carrier)
;;   unpack: i64.and ; i64.shr_u ; i32.wrap_i64         (tear it back apart)
;;
;; The u64 is constructed from two i32s and immediately decomposed — textbook
;; SROA. The kill-criterion (gale #219): after the seam-SROA pass the dissolved
;; body has NO i64 pack/unpack and the i64 carrier local is gone; the result is
;; equivalent to operating on `action`/`new_count` directly.
;;
;; Layout: action in bits [0..8), new_count in bits [8..40).
(module
  ;; decide: pack {action, new_count} -> u64
  ;;   t = (extend_u(new_count) << 8) | (extend_u(action) & 0xff)
  (func $decide (param $action i32) (param $new_count i32) (result i64)
    local.get $new_count
    i64.extend_i32_u
    i64.const 8
    i64.shl
    local.get $action
    i64.extend_i32_u
    i64.const 0xff
    i64.and
    i64.or)

  ;; shim: call decide, then unpack both scalars back and combine.
  ;; Post-inline this is the pure pack/unpack round-trip on $t.
  ;; Semantically equivalent to: (action & 0xff) + new_count
  (func (export "run") (param $action i32) (param $new_count i32) (result i32)
    (local $t i64)
    local.get $action
    local.get $new_count
    call $decide
    local.set $t
    ;; unpack action = (t & 0xff)
    local.get $t
    i64.const 0xff
    i64.and
    i32.wrap_i64
    ;; unpack new_count = (t >> 8)
    local.get $t
    i64.const 8
    i64.shr_u
    i32.wrap_i64
    i32.add))
