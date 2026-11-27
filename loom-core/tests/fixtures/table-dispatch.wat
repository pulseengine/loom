;; #196: call_indirect through a function table (duplicate types, a function reached only
;; through the table, a duplicate body, an empty slot and a signature-mismatch trap).
(module
  (type $bin (func (param i32 i32) (result i32)))
  (type $bin2 (func (param i32 i32) (result i32)))
  (type $un (func (param i32) (result i32)))
  (table $t 8 funcref)
  (elem (i32.const 0) func $add $sub $mul $only_in_table $add_dup $neg $helper_user)
  (func $helper (param i32) (result i32) local.get 0 i32.const 3 i32.mul i32.const 1 i32.add)
  (func $add (type $bin) local.get 0 local.get 1 i32.add)
  (func $sub (type $bin) local.get 0 local.get 1 i32.sub)
  (func $mul (type $bin2) local.get 0 local.get 1 i32.mul)
  (func $only_in_table (type $bin) local.get 0 i32.const 0x55 i32.xor local.get 1 i32.rotl)
  (func $add_dup (type $bin) local.get 0 local.get 1 i32.add)
  (func $neg (type $un) i32.const 0 local.get 0 i32.sub)
  (func $helper_user (type $bin) local.get 0 call $helper local.get 1 i32.add)
  (func $never_called (result i32) i32.const 99)
  (func (export "dispatch") (param $sel i32) (param $a i32) (param $b i32) (result i32)
    local.get $a local.get $b local.get $sel i32.const 7 i32.rem_u
    call_indirect $t (type $bin))
  (func (export "dispatch_un") (param $sel i32) (param $a i32) (result i32)
    local.get $a local.get $sel i32.const 7 i32.rem_u
    call_indirect $t (type $un))
  (func (export "run") (result i32)
    (local $i i32) (local $acc i32)
    i32.const 17 local.set $acc
    block $done
      loop $l
        local.get $i i32.const 7 i32.ge_u br_if $done
        local.get $i i32.const 5 i32.ne
        if
          local.get $acc local.get $i i32.const 13 i32.add local.get $i
          local.get $i call_indirect $t (type $bin)
          i32.const 31 i32.mul i32.add local.set $acc
        end
        local.get $i i32.const 1 i32.add local.set $i
        br $l
      end
    end
    local.get $acc i32.const 5 i32.const 5 call_indirect $t (type $un) i32.add)
  (func (export "empty_slot") (result i32) i32.const 1 i32.const 2 i32.const 7 call_indirect $t (type $bin))
  (func (export "direct") (param i32) (result i32) local.get 0 call $helper call $neg))
