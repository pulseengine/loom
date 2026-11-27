;; #554 fixture: the exported $mix stays and is inlined at four call sites,
;; so `loom optimize --stats` grows both instructions and bytes.
;; Rebuild: wasm-tools parse inline-grows.wat -o inline-grows.wasm
(module
  (func $mix (export "mix") (param $x i32) (result i32)
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
    call $mix
    call $mix
    call $mix
    call $mix))
