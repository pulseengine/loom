;; #196: $f has a non-final function type, so call_b traps; loom must leave the module unchanged.
(module
  (type $a (sub (func (result i32))))
  (type $b (func (result i32)))
  (table 1 funcref)
  (elem (i32.const 0) func $f)
  (func $f (type $a) i32.const 42)
  (func (export "call_b") (result i32) i32.const 0 call_indirect (type $b))
  (func (export "call_a") (result i32) i32.const 0 call_indirect (type $a)))
