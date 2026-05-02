;; Sign-extension operators (wasm 1.1 / standardized feature, LOOM-supported).
;; Should parse, optimize, and round-trip cleanly.
(module
  (func (export "ext8_s") (param i32) (result i32)
    local.get 0
    i32.extend8_s
  )
  (func (export "ext16_s") (param i32) (result i32)
    local.get 0
    i32.extend16_s
  )
  (func (export "i64_ext8_s") (param i64) (result i64)
    local.get 0
    i64.extend8_s
  )
  (func (export "i64_ext16_s") (param i64) (result i64)
    local.get 0
    i64.extend16_s
  )
  (func (export "i64_ext32_s") (param i64) (result i64)
    local.get 0
    i64.extend32_s
  )
)
