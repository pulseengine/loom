;; SIMD/v128 minimal fixture (post-MVP wasm feature)
;; LOOM does not currently support SIMD; parser must reject cleanly (no panic).
(module
  (func (export "simd_test") (result v128)
    (v128.const i8x16 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
    (v128.const i8x16 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1)
    i8x16.add
    (v128.const i8x16 1 0 1 0 1 0 1 0 1 0 1 0 1 0 1 0)
    v128.bitselect
  )
)
