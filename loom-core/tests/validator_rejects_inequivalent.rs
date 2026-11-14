//! The translation validator must reject every pair of functions that behave
//! differently, and keep accepting pairs that do not.
//!
//! Each `DIFFERENT` pair is run in wasmtime on its input first, and the test
//! fails if the two do not differ there in return values, trap, exported
//! globals or exported memory, so a fixture cannot pass by being equivalent.
//! The pairs cover branches carrying values (`br_if`, `br_table`), loads and
//! stores that alias, global writes under branches and around calls, memory
//! writes in functions with and without results, and traps that are deleted
//! or moved past a side effect.
#![cfg(feature = "verification")]

use loom_core::{parse, verify::TranslationValidator, verify::VerificationSignatureContext};
use wasmtime::{Engine, Extern, Instance, Module as WtModule, Store, Val, ValType};

/// (name, original, rewritten, arguments that tell them apart)
const DIFFERENT: &[(&str, &str, &str, &[&str])] = &[
    (
        "memory_01",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32)
i32.const 16
local.get $a
i32.const 255
i32.and
i32.store16
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32)
i32.const 16
local.get $a
i32.store16
)
)"###,
        &["4660"],
    ),
    (
        "memory_02",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $x i32) (param $y i32)
i32.const 0
local.get $x
i32.store
i32.const 0
local.get $y
i32.store16
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $x i32) (param $y i32)
i32.const 0
local.get $y
i32.store16
)
)"###,
        &["305419896", "0"],
    ),
    (
        "memory_03",
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i32)
    i32.const 0
    local.get $x
    i32.store
    i32.const 0
    i32.const 0
    i32.store8
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i32)
    i32.const 0
    i32.const 0
    i32.store8
  )
)"###,
        &["305419896"],
    ),
    (
        "memory_04",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32) (param $v i64)
local.get $a
i32.const 8
i32.add
i32.const 255
i32.and
local.get $v
i64.store
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32) (param $v i64)
local.get $a
i32.const 255
i32.and
local.get $v
i64.store offset=8
)
)"###,
        &["250", "72623859790382856"],
    ),
    (
        "memory_05",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i64)
i32.const 0
local.get $a
i64.store8
i32.const 4
i32.const 0
i64.load8_s
i64.store
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i64)
i32.const 0
local.get $a
i64.store8
i32.const 4
local.get $a
i64.const 255
i64.and
i64.store
)
)"###,
        &["200"],
    ),
    (
        "memory_06",
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i32)
    i32.const 32
    local.get $x
    i64.extend_i32_s
    i64.store
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i32)
    i32.const 32
    local.get $x
    i32.extend8_s
    i64.extend_i32_u
    i64.store
  )
)"###,
        &["128"],
    ),
    (
        "trap_01",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  local.get $a
  f32.reinterpret_i32
  i32.trunc_f32_s
  i32.const 0
  i32.mul
  local.get $b
  i32.add
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  local.get $b
 )
)"###,
        &["1325400064", "5"],
    ),
    (
        "trap_02",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i64) (i64.const 0))
 (func $f (export "f") (param $a i64) (param $b i64) (result i64)
  local.get $a
  local.get $b
  i64.div_s
  drop
  local.get $a
  i64.const 1
  i64.add
 ))"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i64) (i64.const 0))
 (func $f (export "f") (param $a i64) (param $b i64) (result i64)
  local.get $a
  i64.const 1
  i64.add
 ))"###,
        &["5", "0"],
    ),
    (
        "trap_03",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  local.get $a
  local.get $b
  i32.div_u
  drop
  local.get $a
  i32.const 1
  i32.add
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  local.get $a
  i32.const 1
  i32.add
 )
)"###,
        &["5", "0"],
    ),
    (
        "trap_04",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $p i32) (result i32)
  local.get $p
  i32.load offset=16
  drop
  local.get $p
  i32.const 3
  i32.store8
  local.get $p
  i32.load8_u
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $p i32) (result i32)
  local.get $p
  i32.const 3
  i32.store8
  local.get $p
  i32.load8_u
 )
)"###,
        &["65530"],
    ),
    (
        "trap_05",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $a i64) (result i32)
local.get $a
i64.eqz
if
unreachable
end
i32.const 8
local.get $a
i64.store
local.get $a
i32.wrap_i64
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $a i64) (result i32)
i32.const 8
local.get $a
i64.store
local.get $a
i32.wrap_i64
)
)"###,
        &["0"],
    ),
    (
        "trap_06",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (result i64)
  local.get $a
  i64.extend_i32_u
  i64.const 3
  i64.mul
  local.get $a
  i32.const 65535
  i32.and
  i64.load offset=8
  i32.const 1
  select
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (result i64)
  local.get $a
  i64.extend_i32_u
  i64.const 3
  i64.mul
 )
)"###,
        &["65530"],
    ),
    (
        "global_01",
        r###"(module
(memory (export "memory") 1)
(global $g1 (export "g1") (mut i64) (i64.const 0))
(func $f (export "f") (param $a i32) (param $b i32) (result i32)
local.get $a
i64.extend_i32_s
global.set $g1
local.get $a
local.get $b
i32.const 1
i32.or
i32.div_s
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g1 (export "g1") (mut i64) (i64.const 0))
(func $f (export "f") (param $a i32) (param $b i32) (result i32)
local.get $a
local.get $b
i32.const 1
i32.or
i32.div_s
local.get $a
i64.extend_i32_s
global.set $g1
)
)"###,
        &["-2147483648", "-1"],
    ),
    (
        "trap_07",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  (local $t i32)
  local.get $a
  local.get $b
  i32.div_s
  local.set $t
  i32.const 7
  global.set $g0
  local.get $t
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  i32.const 7
  global.set $g0
  local.get $a
  local.get $b
  i32.div_s
 )
)"###,
        &["-2147483648", "-1"],
    ),
    (
        "global_02",
        r###"(module
  (memory (export "memory") 1)
  (global $g0 (export "g0") (mut i32) (i32.const 0))
  (global $h0 (export "h0") (mut i64) (i64.const 0))
  (func $f (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    global.set $g0
    local.get $a
    local.get $b
    i32.div_s
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g0 (export "g0") (mut i32) (i32.const 0))
  (global $h0 (export "h0") (mut i64) (i64.const 0))
  (func $f (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.div_s
    local.get $a
    global.set $g0
  )
)"###,
        &["7", "0"],
    ),
    (
        "trap_08",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32) (local $q i32)
  local.get $a
  local.get $b
  i32.div_u
  local.set $q
  i32.const 0
  local.get $a
  i32.store
  i32.const 7
  global.set $g0
  local.get $q
 ))"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32) (local $q i32)
  i32.const 0
  local.get $a
  i32.store
  i32.const 7
  global.set $g0
  local.get $a
  local.get $b
  i32.div_u
 ))"###,
        &["42", "0"],
    ),
    (
        "trap_09",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i64) (i64.const 0))
(func $f (export "f") (param $x i64) (param $y i64) (result i64)
(local $t i64)
local.get $x
global.set $g0
local.get $x
local.get $y
i64.rem_u
local.set $t
i32.const 40
local.get $t
i64.store8 offset=1
local.get $t
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i64) (i64.const 0))
(func $f (export "f") (param $x i64) (param $y i64) (result i64)
(local $t i64)
local.get $x
local.get $y
i64.rem_u
local.set $t
local.get $x
global.set $g0
i32.const 40
local.get $t
i64.store8 offset=1
local.get $t
)
)"###,
        &["5", "0"],
    ),
    (
        "branch_01",
        r###"(module
 (memory (export "memory") 1)
 (global $g (export "g") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  block (result i32)
  i32.const 1
  local.get $a
  local.get $b
  i32.eq
  br_if 0
  drop
  global.get $g
  i32.const 1
  i32.add
  global.set $g
  i32.const 2
  end
 ))"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g (export "g") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  global.get $g
  i32.const 1
  i32.add
  global.set $g
  block (result i32)
  i32.const 1
  local.get $a
  local.get $b
  i32.eq
  br_if 0
  drop
  i32.const 2
  end
 ))"###,
        &["4", "4"],
    ),
    (
        "branch_02",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $x i32) (result i32)
block
block
block
local.get $x
br_table 0 1 2
end
i32.const 10
return
end
i32.const 20
return
end
i32.const 30
))"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $x i32) (result i32)
block
block
block
local.get $x
br_table 0 2 2
end
i32.const 10
return
end
i32.const 20
return
end
i32.const 30
))"###,
        &["1"],
    ),
    (
        "branch_03",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $p i32) (param $v i32) (result i32)
i32.const 0
local.get $v
i32.store
local.get $p
i32.const 4
i32.lt_u
if
local.get $p
i32.const 0
i32.store8
end
i32.const 0
i32.load
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $p i32) (param $v i32) (result i32)
i32.const 0
local.get $v
i32.store
local.get $p
i32.const 4
i32.lt_u
if
local.get $p
i32.const 0
i32.store8
end
local.get $v
)
)"###,
        &["1", "16909060"],
    ),
    (
        "branch_04",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $x i32) (result i32)
block (result i32)
i32.const 255
local.get $x
i32.const 255
i32.gt_s
br_if 0
drop
i32.const 0
local.get $x
i32.const 0
i32.lt_s
br_if 0
drop
local.get $x
end
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $x i32) (result i32)
block (result i32)
i32.const 255
local.get $x
i32.const 255
i32.ge_u
br_if 0
drop
i32.const 0
local.get $x
i32.const 0
i32.lt_s
br_if 0
drop
local.get $x
end
)
)"###,
        &["-1"],
    ),
    (
        "branch_05",
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i32) (result i64)
    block
    block
    block
    local.get $x
    br_table 0 1 2
    end
    i64.const 10
    return
    end
    i64.const 20
    return
    end
    i64.const 30
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i32) (result i64)
    block
    block
    block
    local.get $x
    br_table 0 2 2
    end
    i64.const 10
    return
    end
    i64.const 20
    return
    end
    i64.const 30
  )
)"###,
        &["1"],
    ),
    (
        "alias_01",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $p i32) (param $q i32) (result i32)
block $done
local.get $p
i32.const 60000
i32.gt_u
br_if $done
local.get $p
i32.const -1
i32.store
local.get $q
i32.const 0
i32.store16
end
local.get $p
i32.load
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $p i32) (param $q i32) (result i32)
block $done
local.get $p
i32.const 60000
i32.gt_u
br_if $done
local.get $q
i32.const 0
i32.store16
local.get $p
i32.const -1
i32.store
end
local.get $p
i32.load
)
)"###,
        &["10", "12"],
    ),
    (
        "alias_02",
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $a i32) (param $b i32) (param $c i32) (result i32)
  local.get $a
  i32.const -1
  i32.store
  local.get $c
  if
  local.get $b
  i32.const 0
  i32.store8 offset=3
  end
  local.get $a
  i32.load
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $a i32) (param $b i32) (param $c i32) (result i32)
  local.get $c
  if
  local.get $b
  i32.const 0
  i32.store8 offset=3
  end
  local.get $a
  i32.const -1
  i32.store
  local.get $a
  i32.load
 )
)"###,
        &["12", "12", "1"],
    ),
    (
        "branch_06",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32) (param $b i32) (result i32)
block (result i32)
i32.const -1
local.get $b
br_if 0
drop
local.get $a
local.get $b
i32.rem_s
local.get $a
i32.const 0
i32.lt_s
br_if 0
i32.const 100
i32.add
end
))"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32) (param $b i32) (result i32)
block (result i32)
i32.const -1
local.get $b
i32.eqz
br_if 0
drop
local.get $a
local.get $b
i32.rem_s
local.get $a
i32.const 0
i32.le_s
br_if 0
i32.const 100
i32.add
end
))"###,
        &["0", "7"],
    ),
    (
        "alias_03",
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $a i32) (param $b i32) (param $c i32) (result i32)
    local.get $a
    local.get $c
    i32.store
    local.get $c
    if
    local.get $b
    i32.const 0
    i32.store
    end
    local.get $a
    i32.load
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $a i32) (param $b i32) (param $c i32) (result i32)
    local.get $a
    local.get $c
    i32.store
    local.get $c
    if
    local.get $b
    i32.const 0
    i32.store
    end
    local.get $c
  )
)"###,
        &["200", "200", "5"],
    ),
    (
        "call_01",
        r###"(module
  (memory (export "memory") 1)
  (global $g0 (export "g0") (mut i32) (i32.const 0))
  (func $f (export "f") (param $a i32) (result i32)
    local.get $a
    global.set $g0
    call $get
  )
  (func $get (result i32)
    global.get $g0
    i32.const 10
    i32.mul
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g0 (export "g0") (mut i32) (i32.const 0))
  (func $f (export "f") (param $a i32) (result i32)
    call $get
    local.get $a
    global.set $g0
  )
  (func $get (result i32)
    global.get $g0
    i32.const 10
    i32.mul
  )
)"###,
        &["4"],
    ),
    (
        "branch_07",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $x i32) (result i32)
local.get $x
global.set $g0
block
block
block
local.get $x
br_table 0 1 2 1
end
i32.const 10
return
end
i32.const 20
return
end
i32.const 30
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $x i32) (result i32)
local.get $x
global.set $g0
block
block
block
local.get $x
br_table 0 1 2 2
end
i32.const 10
return
end
i32.const 20
return
end
i32.const 30
)
)"###,
        &["-1"],
    ),
    (
        "branch_08",
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i64) (param $y i32) (result i32)
    block (result i32)
    i32.const -1
    local.get $x
    i64.const 0
    i64.lt_s
    br_if 0
    drop
    local.get $x
    i32.wrap_i64
    local.get $y
    i32.add
    end
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $x i64) (param $y i32) (result i32)
    block (result i32)
    i32.const -1
    local.get $x
    i64.const 0
    i64.le_s
    br_if 0
    drop
    local.get $x
    i32.wrap_i64
    local.get $y
    i32.add
    end
  )
)"###,
        &["0", "5"],
    ),
    (
        "memory_07",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32) (param $b i32) (result i32)
i32.const 32
local.get $a
i32.store
i32.const 32
local.get $b
i32.store
local.get $a
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $a i32) (param $b i32) (result i32)
i32.const 32
local.get $a
i32.store
local.get $b
if
i32.const 32
local.get $b
i32.store
end
local.get $a
)
)"###,
        &["5", "0"],
    ),
    (
        "call_02",
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $p i32) (param $v i64) (result i32)
  local.get $p
  local.get $v
  call $put
  drop
  local.get $p
  i32.const 1
  i32.add
 )
 (func $put (param $p i32) (param $v i64) (result i64)
  local.get $p
  local.get $v
  i64.store
  local.get $v
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $p i32) (param $v i64) (result i32)
  local.get $p
  i32.const 1
  i32.add
 )
 (func $put (param $p i32) (param $v i64) (result i64)
  local.get $p
  local.get $v
  i64.store
  local.get $v
 )
)"###,
        &["8", "1311768467463790320"],
    ),
    (
        "branch_09",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $p i32) (result i32)
block
local.get $p
i32.const 0
i32.lt_s
br_if 0
i32.const 16
local.get $p
i32.store
global.get $g0
i32.const 1
i32.add
global.set $g0
end
global.get $g0
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $p i32) (result i32)
i32.const 16
local.get $p
i32.store
block
local.get $p
i32.const 0
i32.lt_s
br_if 0
global.get $g0
i32.const 1
i32.add
global.set $g0
end
global.get $g0
)
)"###,
        &["-5"],
    ),
    (
        "branch_10",
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  block $b2
  block $b1
  block $b0
  local.get $a
  br_table $b0 $b1 $b2
  end
  end
  i32.const 0
  local.get $b
  i32.store8 offset=5
  end
  i32.const 0
 ))"###,
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $a i32) (param $b i32) (result i32)
  block $b2
  block $b1
  block $b0
  local.get $a
  br_table $b0 $b1 $b1
  end
  end
  i32.const 0
  local.get $b
  i32.store8 offset=5
  end
  i32.const 0
 ))"###,
        &["2", "4660"],
    ),
    (
        "local_01",
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $p i32) (param $v i32) (param $k i32) (result i32)
  local.get $p
  local.get $v
  i32.store
  local.get $k
  if
  local.get $p
  i32.const 4
  i32.add
  local.set $p
  end
  local.get $p
  local.get $v
  i32.const 1
  i32.add
  i32.store offset=8
  local.get $p
  local.get $v
  i32.add
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (func $f (export "f") (param $p i32) (param $v i32) (param $k i32) (result i32)
  local.get $p
  local.get $v
  i32.store
  local.get $p
  local.get $k
  if
  local.get $p
  i32.const 4
  i32.add
  local.set $p
  end
  local.get $v
  i32.const 1
  i32.add
  i32.store offset=8
  local.get $p
  local.get $v
  i32.add
 )
)"###,
        &["16", "5", "1"],
    ),
    (
        "call_03",
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    call $log
    local.get $b
    i32.const 0
    i32.ge_s
    if (result i32)
      local.get $b
      i32.const 1
      i32.add
    else
      local.get $b
      i32.const 255
      i32.and
      if (result i32)
        i32.const 1
      else
        i32.const 2
      end
    end
  )
  (func $log (param $x i32)
    i32.const 32
    i32.const 32
    i32.load
    local.get $x
    i32.xor
    i32.store
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $b
    i32.const 0
    i32.ge_s
    if (result i32)
      local.get $a
      call $log
      local.get $b
      i32.const 1
      i32.add
    else
      local.get $b
      i32.const 255
      i32.and
      if (result i32)
        i32.const 1
      else
        i32.const 2
      end
    end
  )
  (func $log (param $x i32)
    i32.const 32
    i32.const 32
    i32.load
    local.get $x
    i32.xor
    i32.store
  )
)"###,
        &["7", "-256"],
    ),
    (
        "memory_08",
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $p i32) (param $v i32)
local.get $p
i32.const 4
i32.add
local.get $v
i32.store
)
)"###,
        r###"(module
(memory (export "memory") 1)
(func $f (export "f") (param $p i32) (param $v i32)
local.get $p
local.get $v
i32.store offset=4
)
)"###,
        &["-4", "7"],
    ),
    (
        "memory_09",
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $a i32) (param $v i32)
    block
    local.get $v
    br_if 0
    local.get $a
    i32.const 4
    i32.add
    local.get $v
    i32.store8
    end
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (func $f (export "f") (param $a i32) (param $v i32)
    block
    local.get $v
    i32.eqz
    br_if 0
    local.get $a
    local.get $v
    i32.store8 offset=4
    end
  )
)"###,
        &["-4", "7"],
    ),
    (
        "trap_10",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $p i32) (param $v i32)
local.get $p
local.get $v
i32.store8
local.get $p
i64.load offset=8
drop
local.get $p
local.get $v
i32.store16 offset=2
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $p i32) (param $v i32)
local.get $p
local.get $v
i32.store8
local.get $p
local.get $v
i32.store16 offset=2
)
)"###,
        &["65528", "258"],
    ),
    (
        "trap_11",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32)
  local.get $a
  i32.const -1
  i32.div_s
  local.get $b
  i32.add
  global.set $g0
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32)
  i32.const 0
  local.get $a
  i32.sub
  local.get $b
  i32.add
  global.set $g0
 )
)"###,
        &["-2147483648", "3"],
    ),
    (
        "trap_12",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $a i32) (param $b i32)
i32.const 0
local.get $a
i32.store offset=4
local.get $a
local.get $b
i32.div_u
drop
i32.const 0
local.get $b
i32.store16 offset=10
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(func $f (export "f") (param $a i32) (param $b i32)
i32.const 0
local.get $a
i32.store offset=4
i32.const 0
local.get $b
i32.store16 offset=10
)
)"###,
        &["7", "0"],
    ),
    (
        "trap_13",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32)
  local.get $a
  i32.const 10
  i32.gt_s
  if
  local.get $a
  i32.const 100
  i32.gt_u
  if
  unreachable
  end
  i32.const 0
  local.get $a
  i32.store
  end
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32)
  local.get $a
  i32.const 10
  i32.gt_s
  if
  i32.const 0
  local.get $a
  i32.store
  end
 )
)"###,
        &["200"],
    ),
    (
        "alias_04",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $p i32) (param $q i32) (param $c i32)
  local.get $c
  if
  local.get $p
  i32.const 200
  i32.store
  end
  local.get $q
  i32.load
  i32.extend8_s
  global.set $g0
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $p i32) (param $q i32) (param $c i32)
  local.get $q
  i32.load
  i32.extend8_s
  global.set $g0
  local.get $c
  if
  local.get $p
  i32.const 200
  i32.store
  end
 )
)"###,
        &["64", "64", "1"],
    ),
    (
        "call_04",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (global $g1 (export "g1") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32)
  local.get $a
  global.set $g0
  call $snap
  local.get $b
  global.set $g0
 )
 (func $snap
  global.get $g0
  global.set $g1
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (global $g1 (export "g1") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32)
  call $snap
  local.get $b
  global.set $g0
 )
 (func $snap
  global.get $g0
  global.set $g1
 )
)"###,
        &["5", "9"],
    ),
    (
        "global_03",
        r###"(module
  (memory (export "memory") 1)
  (global $g0 (export "g0") (mut i32) (i32.const 0))
  (global $g1 (export "g1") (mut i32) (i32.const 0))
  (func $f (export "f") (param $a i32) (param $b i32)
    block
    local.get $a
    i32.eqz
    br_if 0
    local.get $b
    global.set $g0
    end
    local.get $b
    i32.const 1
    i32.add
    global.set $g1
  )
)"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g0 (export "g0") (mut i32) (i32.const 0))
  (global $g1 (export "g1") (mut i32) (i32.const 0))
  (func $f (export "f") (param $a i32) (param $b i32)
    local.get $b
    global.set $g0
    local.get $b
    i32.const 1
    i32.add
    global.set $g1
  )
)"###,
        &["0", "77"],
    ),
    (
        "branch_11",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(global $g1 (export "g1") (mut i32) (i32.const 0))
(func $f (export "f") (param $a i32) (param $b i32)
block
local.get $a
i32.eqz
br_if 0
global.get $g0
local.get $b
i32.add
global.set $g0
end
global.get $g1
i32.const 1
i32.add
global.set $g1
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(global $g1 (export "g1") (mut i32) (i32.const 0))
(func $f (export "f") (param $a i32) (param $b i32)
global.get $g0
local.get $b
i32.add
global.set $g0
block
local.get $a
br_if 0
end
global.get $g1
i32.const 1
i32.add
global.set $g1
)
)"###,
        &["0", "5"],
    ),
    (
        "branch_12",
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(global $g1 (export "g1") (mut i32) (i32.const 0))
(func $f (export "f") (param $n i32)
block
local.get $n
i32.const 100
i32.gt_u
br_if 0
global.get $g0
local.get $n
i32.add
global.set $g0
end
global.get $g1
i32.const 1
i32.add
global.set $g1
)
)"###,
        r###"(module
(memory (export "memory") 1)
(global $g0 (export "g0") (mut i32) (i32.const 0))
(global $g1 (export "g1") (mut i32) (i32.const 0))
(func $f (export "f") (param $n i32)
block
local.get $n
i32.const 100
i32.ge_u
br_if 0
global.get $g0
local.get $n
i32.add
global.set $g0
end
global.get $g1
i32.const 1
i32.add
global.set $g1
)
)"###,
        &["100"],
    ),
    (
        "alias_05",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (global $g1 (export "g1") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (param $c i32)
  block
  local.get $a
  i32.eqz
  br_if 0
  local.get $a
  local.get $b
  i32.store8
  end
  local.get $c
  i32.load8_u
  global.set $g0
  local.get $a
  global.set $g1
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (global $g1 (export "g1") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (param $c i32)
  local.get $c
  i32.load8_u
  global.set $g0
  block
  local.get $a
  br_if 0
  local.get $a
  local.get $b
  i32.store8
  end
  local.get $a
  global.set $g1
 )
)"###,
        &["33", "200", "33"],
    ),
    (
        "trap_14",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (param $p i32)
  local.get $a
  local.get $b
  i32.div_u
  global.set $g0
  local.get $p
  local.get $a
  i32.store
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32) (param $p i32)
  local.get $p
  local.get $a
  i32.store
  local.get $a
  local.get $b
  i32.div_u
  global.set $g0
 )
)"###,
        &["7", "0", "65536"],
    ),
    (
        "call_05",
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32)
  block
  local.get $a
  br_if 0
  local.get $b
  local.get $a
  i32.store
  local.get $a
  call $check
  global.set $g0
  end
 )
 (func $check (param $x i32) (result i32)
  local.get $x
  i32.const 100
  i32.gt_u
  if
  unreachable
  end
  local.get $x
  i32.const 1
  i32.add
 )
)"###,
        r###"(module
 (memory (export "memory") 1)
 (global $g0 (export "g0") (mut i32) (i32.const 0))
 (func $f (export "f") (param $a i32) (param $b i32)
  block
  local.get $a
  i32.eqz
  br_if 0
  local.get $a
  call $check
  global.set $g0
  local.get $b
  local.get $a
  i32.store
  end
 )
 (func $check (param $x i32) (result i32)
  local.get $x
  i32.const 100
  i32.gt_u
  if
  unreachable
  end
  local.get $x
  i32.const 1
  i32.add
 )
)"###,
        &["200", "70000"],
    ),
    (
        "memory_void_swapped_stores",
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32)
    i32.const 0
    i32.const 1
    i32.store
    i32.const 0
    i32.const 2
    i32.store
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32)
    i32.const 0
    i32.const 2
    i32.store
    i32.const 0
    i32.const 1
    i32.store
))"###,
        &["0"],
    ),
    (
        "global_if_then_store",
        r###"(module (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $x i32)
    (if (local.get $x) (then (global.set $g (i32.const 7))))
    (global.set $g (i32.const 3))))"###,
        r###"(module (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $x i32)
    (global.set $g (i32.const 3))
    (if (local.get $x) (then (global.set $g (i32.const 7))))))"###,
        &["1"],
    ),
    (
        "arith_add_vs_sub",
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.sub
))"###,
        &["3", "5"],
    ),
];

/// (original, rewritten, arguments): equivalent pairs.
const SAME: &[(&str, &str, &[&str])] = &[
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 332
    i32.add
    local.get $b
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 332
    i32.add
    local.get $b
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 971
    i32.mul
    local.get $b
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 971
    i32.mul
    local.get $b
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 155
    i32.and
    local.get $b
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 155
    i32.and
    local.get $b
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 405
    i32.or
    local.get $b
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 405
    i32.or
    local.get $b
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 667
    i32.xor
    local.get $b
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 667
    i32.xor
    local.get $b
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 50
    i32.add
    local.get $b
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 50
    i32.add
    local.get $b
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 75
    i32.mul
    local.get $b
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 75
    i32.mul
    local.get $b
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 841
    i32.and
    local.get $b
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 841
    i32.and
    local.get $b
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 549
    i32.or
    local.get $b
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 549
    i32.or
    local.get $b
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 97
    i32.xor
    local.get $b
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 97
    i32.xor
    local.get $b
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 375
    i32.add
    local.get $b
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 375
    i32.add
    local.get $b
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 597
    i32.mul
    local.get $b
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 597
    i32.mul
    local.get $b
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 60
    i32.and
    local.get $b
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 60
    i32.and
    local.get $b
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 932
    i32.or
    local.get $b
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 932
    i32.or
    local.get $b
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 520
    i32.xor
    local.get $b
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 520
    i32.xor
    local.get $b
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 220
    i32.add
    local.get $b
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 220
    i32.add
    local.get $b
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 39
    i32.mul
    local.get $b
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 39
    i32.mul
    local.get $b
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 89
    i32.and
    local.get $b
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 89
    i32.and
    local.get $b
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 445
    i32.or
    local.get $b
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 445
    i32.or
    local.get $b
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 429
    i32.xor
    local.get $b
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 429
    i32.xor
    local.get $b
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 72
    i32.add
    local.get $b
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 72
    i32.add
    local.get $b
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 247
    i32.mul
    local.get $b
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 247
    i32.mul
    local.get $b
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 93
    i32.and
    local.get $b
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 93
    i32.and
    local.get $b
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 565
    i32.or
    local.get $b
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 565
    i32.or
    local.get $b
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 435
    i32.xor
    local.get $b
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    i32.const 435
    i32.xor
    local.get $b
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
    i32.const 61
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 61
    local.get $b
    local.get $a
    i32.add
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.mul
    i32.const 847
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 847
    local.get $b
    local.get $a
    i32.mul
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.and
    i32.const 580
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 580
    local.get $b
    local.get $a
    i32.and
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.or
    i32.const 127
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 127
    local.get $b
    local.get $a
    i32.or
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.xor
    i32.const 971
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 971
    local.get $b
    local.get $a
    i32.xor
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
    i32.const 229
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 229
    local.get $b
    local.get $a
    i32.add
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.mul
    i32.const 646
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 646
    local.get $b
    local.get $a
    i32.mul
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.and
    i32.const 643
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 643
    local.get $b
    local.get $a
    i32.and
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.or
    i32.const 597
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 597
    local.get $b
    local.get $a
    i32.or
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.xor
    i32.const 971
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 971
    local.get $b
    local.get $a
    i32.xor
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
    i32.const 64
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 64
    local.get $b
    local.get $a
    i32.add
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.mul
    i32.const 591
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 591
    local.get $b
    local.get $a
    i32.mul
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.and
    i32.const 600
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 600
    local.get $b
    local.get $a
    i32.and
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.or
    i32.const 407
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 407
    local.get $b
    local.get $a
    i32.or
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.xor
    i32.const 51
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 51
    local.get $b
    local.get $a
    i32.xor
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
    i32.const 1000
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 1000
    local.get $b
    local.get $a
    i32.add
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.mul
    i32.const 227
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 227
    local.get $b
    local.get $a
    i32.mul
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.and
    i32.const 48
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 48
    local.get $b
    local.get $a
    i32.and
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.or
    i32.const 571
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 571
    local.get $b
    local.get $a
    i32.or
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.xor
    i32.const 880
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 880
    local.get $b
    local.get $a
    i32.xor
    i32.xor
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.add
    i32.const 137
    i32.add
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 137
    local.get $b
    local.get $a
    i32.add
    i32.add
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.mul
    i32.const 297
    i32.mul
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 297
    local.get $b
    local.get $a
    i32.mul
    i32.mul
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.and
    i32.const 430
    i32.and
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 430
    local.get $b
    local.get $a
    i32.and
    i32.and
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.or
    i32.const 148
    i32.or
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 148
    local.get $b
    local.get $a
    i32.or
    i32.or
))"###,
        &["3", "5"],
    ),
    (
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    local.get $a
    local.get $b
    i32.xor
    i32.const 554
    i32.xor
))"###,
        r###"(module
  (memory (export "memory") 1)
  (global $g (export "g") (mut i32) (i32.const 0))
  (func (export "f") (param $a i32) (param $b i32) (result i32)
    i32.const 554
    local.get $b
    local.get $a
    i32.xor
    i32.xor
))"###,
        &["3", "5"],
    ),
];

/// What a caller can observe after calling `f`.
#[derive(Debug, PartialEq)]
struct Observed {
    outcome: Result<Vec<String>, String>,
    globals: Vec<(String, String)>,
    memory: Vec<u8>,
}

fn observe(wat: &str, args: &[&str]) -> Observed {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wat).expect("fixture compiles");
    let mut store = Store::new(&engine, ());
    let instance = Instance::new(&mut store, &module, &[]).expect("instantiate");
    let f = instance.get_func(&mut store, "f").expect("export f");
    let ty = f.ty(&store);
    let params: Vec<Val> = ty
        .params()
        .zip(args)
        .map(|(t, a)| match t {
            ValType::I32 => Val::I32(a.parse::<i64>().unwrap() as i32),
            ValType::I64 => Val::I64(a.parse::<i64>().unwrap()),
            ValType::F32 => Val::F32(a.parse::<f32>().unwrap().to_bits()),
            ValType::F64 => Val::F64(a.parse::<f64>().unwrap().to_bits()),
            other => panic!("unsupported parameter type {other:?}"),
        })
        .collect();
    let mut results: Vec<Val> = ty.results().map(|_| Val::I32(0)).collect();
    let outcome = f
        .call(&mut store, &params, &mut results)
        .map(|()| results.iter().map(|v| format!("{v:?}")).collect())
        .map_err(|e| format!("{:?}", e.downcast_ref::<wasmtime::Trap>()));
    let mut globals = Vec::new();
    let mut memory = Vec::new();
    let exports: Vec<(String, Extern)> = instance
        .exports(&mut store)
        .map(|e| (e.name().to_string(), e.into_extern()))
        .collect();
    for (name, ext) in exports {
        match ext {
            Extern::Global(g) => globals.push((name, format!("{:?}", g.get(&mut store)))),
            Extern::Memory(m) => memory = m.data(&store).to_vec(),
            _ => {}
        }
    }
    Observed {
        outcome,
        globals,
        memory,
    }
}

/// The exported `f`, its module, and a validator context for that module.
fn function_f(wat: &str) -> (loom_core::Function, VerificationSignatureContext) {
    let module = parse::parse_wat(wat).expect("parse");
    let index = module
        .exports
        .iter()
        .find(|e| e.name == "f")
        .and_then(|e| match e.kind {
            loom_core::ExportKind::Func(i) => Some(i as usize),
            _ => None,
        })
        .expect("export f");
    let imported = module
        .imports
        .iter()
        .filter(|i| matches!(i.kind, loom_core::ImportKind::Func(_)))
        .count();
    let ctx = VerificationSignatureContext::from_module(&module);
    (module.functions[index - imported].clone(), ctx)
}

/// Whether each validator entry point accepts `new` as a rewrite of `orig`.
fn accepted(orig: &str, new: &str) -> (bool, bool) {
    let (a, ctx) = function_f(orig);
    let (b, _) = function_f(new);
    let with_module = TranslationValidator::new_with_context(&a, "test", ctx)
        .verify(&b)
        .is_ok();
    let without = TranslationValidator::new(&a, "test").verify(&b).is_ok();
    (with_module, without)
}

#[test]
fn different_pairs_differ_in_wasmtime_and_are_rejected() {
    let mut wrong = Vec::new();
    for (name, orig, new, args) in DIFFERENT {
        assert_ne!(
            observe(orig, args),
            observe(new, args),
            "{name}: the fixture pair does not differ on {args:?}"
        );
        let (with_module, without) = accepted(orig, new);
        if with_module || without {
            wrong.push(format!(
                "{name}: accepted (with module context: {with_module}, without: {without})"
            ));
        }
    }
    assert!(
        wrong.is_empty(),
        "accepted {} of {}:\n{}",
        wrong.len(),
        DIFFERENT.len(),
        wrong.join("\n")
    );
}

#[test]
fn equivalent_pairs_are_accepted() {
    let mut wrong = Vec::new();
    for (i, (orig, new, args)) in SAME.iter().enumerate() {
        assert_eq!(observe(orig, args), observe(new, args), "pair {i} differs");
        let (with_module, _) = accepted(orig, new);
        if !with_module {
            wrong.push(i);
        }
    }
    assert!(wrong.is_empty(), "rejected equivalent pairs {wrong:?}");
}
