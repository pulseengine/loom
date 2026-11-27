;; #196: a later segment puts ref.null in slot 0, so slot0 traps; directize must not fold it.
(module
  (type $bin (func (param i32 i32) (result i32)))
  (table $t 2 funcref)
  (elem (i32.const 0) func $add)
  (elem (table $t) (i32.const 0) funcref (ref.null func))
  (func $add (type $bin) local.get 0 local.get 1 i32.add)
  (func (export "slot0") (param i32 i32) (result i32)
    local.get 0 local.get 1 i32.const 0 call_indirect $t (type $bin)))
