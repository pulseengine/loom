//! A module's `(start)` function survives the full optimization pipeline and
//! runs exactly once.
//!
//! Components built by rustc 1.100 carry a wit-component fixup core module
//! whose `(start)` calls the main module's `_initialize`. loom skips those
//! modules today (#196), so the start survives by pass-through. This test
//! makes the core pipeline rewrite a module that has a start function, then
//! checks with wasmtime that the output still declares the start and that
//! instantiating it runs the start once. The planted control drops the start
//! from the optimized module and must fail the same check.

use loom_core::{encode, optimize, parse};
use wasmtime::{Engine, Instance, Module as WtModule, Store};

/// `$init` is the start function. It adds `(2 + 3) - 4 = 1` to a global, so
/// a start that runs once leaves `count() == 1`, a dropped start leaves 0 and
/// a start that runs twice leaves 2. Both bodies hold constant arithmetic the
/// pipeline folds, so the pipeline rewrites the module rather than copying it.
const WAT: &str = r#"
(module
  (global $count (mut i32) (i32.const 0))
  (func $init
    global.get $count
    i32.const 2
    i32.const 3
    i32.add
    i32.const 4
    i32.sub
    i32.add
    global.set $count)
  (func (export "count") (result i32)
    global.get $count
    i32.const 0
    i32.add)
  (start $init))
"#;

fn optimized(wasm: &[u8]) -> Vec<u8> {
    let mut module = parse::parse_wasm(wasm).expect("parse");
    optimize::optimize_module(&mut module).expect("optimize");
    encode::encode_wasm(&module).expect("encode")
}

/// Instantiate `wasm` in wasmtime and read `count()` twice.
fn count_after_instantiation(wasm: &[u8]) -> Result<(i32, i32), String> {
    let engine = Engine::default();
    let module = WtModule::new(&engine, wasm).map_err(|e| format!("load: {e}"))?;
    let mut store = Store::new(&engine, ());
    let instance =
        Instance::new(&mut store, &module, &[]).map_err(|e| format!("instantiate: {e}"))?;
    let count = instance
        .get_typed_func::<(), i32>(&mut store, "count")
        .map_err(|e| format!("export count: {e}"))?;
    let first = count
        .call(&mut store, ())
        .map_err(|e| format!("call: {e}"))?;
    let second = count
        .call(&mut store, ())
        .map_err(|e| format!("call: {e}"))?;
    Ok((first, second))
}

/// The property: `output` declares the same start function as `input`, and
/// instantiating it runs the start exactly once.
fn start_kept_and_runs_once(input: &[u8], output: &[u8]) -> Result<(), String> {
    let want = parse::parse_wasm(input)
        .map_err(|e| format!("parse input: {e}"))?
        .start_function;
    let got = parse::parse_wasm(output)
        .map_err(|e| format!("parse output: {e}"))?
        .start_function;
    if want.is_none() {
        return Err("input has no start function; the test proves nothing".into());
    }
    if got != want {
        return Err(format!("start function {want:?} became {got:?}"));
    }
    match count_after_instantiation(output)? {
        (1, 1) => Ok(()),
        (n, m) => Err(format!(
            "start ran {n} times (count() read {n}, then {m}); expected once"
        )),
    }
}

#[test]
fn pipeline_rewrites_the_module_and_keeps_its_start() {
    let input = wat::parse_str(WAT).expect("wat");
    let output = optimized(&input);

    // Not pass-through: the pipeline changed at least one function body.
    let before = parse::parse_wasm(&input).unwrap();
    let after = parse::parse_wasm(&output).unwrap();
    let sizes = |m: &loom_core::Module| {
        m.functions
            .iter()
            .map(|f| f.instructions.len())
            .collect::<Vec<_>>()
    };
    assert_ne!(
        sizes(&before),
        sizes(&after),
        "the pipeline left every function body unchanged, so this test would only show pass-through"
    );

    assert_eq!(count_after_instantiation(&input), Ok((1, 1)), "input");
    start_kept_and_runs_once(&input, &output).unwrap();
}

#[test]
fn control_a_pass_that_drops_the_start_fails() {
    let input = wat::parse_str(WAT).expect("wat");
    let mut module = parse::parse_wasm(&optimized(&input)).unwrap();
    module.start_function = None;
    let dropped = encode::encode_wasm(&module).unwrap();

    let err = start_kept_and_runs_once(&input, &dropped).unwrap_err();
    assert!(err.contains("start function"), "{err}");
}

#[test]
fn control_a_start_that_runs_twice_fails() {
    // A module whose instantiation leaves count() at 2 stands in for a pass
    // that duplicates the start's effect.
    let input = wat::parse_str(WAT).expect("wat");
    let twice = wat::parse_str(WAT.replace("i32.const 4", "i32.const 3")).expect("wat");
    let err = start_kept_and_runs_once(&input, &twice).unwrap_err();
    assert!(err.contains("start ran 2 times"), "{err}");
}
