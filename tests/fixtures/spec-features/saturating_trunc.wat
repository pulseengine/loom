;; Non-trapping float-to-int (saturating truncation), LOOM-supported.
;; Should parse, optimize, and round-trip cleanly.
(module
  (func (export "i32_trunc_sat_f32_s") (param f32) (result i32)
    local.get 0
    i32.trunc_sat_f32_s
  )
  (func (export "i32_trunc_sat_f32_u") (param f32) (result i32)
    local.get 0
    i32.trunc_sat_f32_u
  )
  (func (export "i32_trunc_sat_f64_s") (param f64) (result i32)
    local.get 0
    i32.trunc_sat_f64_s
  )
  (func (export "i32_trunc_sat_f64_u") (param f64) (result i32)
    local.get 0
    i32.trunc_sat_f64_u
  )
  (func (export "i64_trunc_sat_f32_s") (param f32) (result i64)
    local.get 0
    i64.trunc_sat_f32_s
  )
  (func (export "i64_trunc_sat_f64_u") (param f64) (result i64)
    local.get 0
    i64.trunc_sat_f64_u
  )
)
