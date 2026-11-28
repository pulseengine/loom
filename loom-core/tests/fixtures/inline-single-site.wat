;; #555 fixture: $mix has one call site and is not exported. Inlining moves its
;; body into "run". Without dce-functions afterwards (`--passes inline`) both
;; copies remain and the module grows; the full pipeline removes $mix.
;; Rebuild: wasm-tools parse inline-single-site.wat -o inline-single-site.wasm
(module
  (func $mix (param $x i32) (result i32)
    local.get $x
    i32.const 7
    i32.mul
    local.get $x
    i32.const 3
    i32.shr_u
    i32.xor
    i32.const 11
    i32.add)
  (func (export "run") (param $a i32) (result i32)
    local.get $a
    call $mix))
