//! Execution oracle shared by the `fuzz_differential` target and its tests.
//!
//! [`compare_modules`] runs an original and an optimized module under wasmtime
//! and reports whether their observable behaviour agrees: return values,
//! trap-vs-no-trap, and the final contents of exported memories and globals.
//!
//! Every execution is bounded by fuel, memory limits and wasmtime's native
//! stack limit, so infinite loops and runaway recursion cannot hang the
//! fuzzer. A run that hits one of those limits is *inconclusive* (an optimizer
//! may legitimately use less fuel or stack) and is never reported as a
//! divergence.

use wasmtime::{
    Config, Engine, Extern, ExternType, Instance, Module, Store, StoreLimits, Trap, Val, ValType,
};

/// Fuel granted to instantiation and to each exported-function call.
const FUEL: u64 = 200_000;
/// Cap on linear memory so `memory.grow` loops cannot exhaust the host.
const MAX_MEMORY_BYTES: usize = 16 * 1024 * 1024;
/// Cap on exported functions exercised per module.
const MAX_EXPORTS: usize = 32;

/// Result of comparing two modules.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every compared call and the final state agreed. `calls` is the number
    /// of conclusive function calls that were compared (0 means nothing ran).
    Equivalent { calls: usize },
    /// The comparison could not be made (does not compile, uses an
    /// unsupported feature, hit a resource bound, ...).
    Skipped(&'static str),
    /// The two modules are observably different: a bug in the optimizer.
    Mismatch(String),
}

#[derive(Debug, PartialEq)]
enum Obs {
    I32(i32),
    I64(i64),
    /// f32 bits, with every NaN canonicalised (payload is nondeterministic)
    F32(u32),
    F64(u64),
}

#[derive(Debug, PartialEq)]
enum Outcome {
    Returned(Vec<Obs>),
    Trapped,
    /// Fuel, stack or memory bound hit: no conclusion can be drawn.
    Inconclusive,
}

struct Side {
    store: Store<StoreLimits>,
    instance: Instance,
}

fn observe(v: &Val) -> Option<Obs> {
    Some(match v {
        Val::I32(x) => Obs::I32(*x),
        Val::I64(x) => Obs::I64(*x),
        Val::F32(b) => {
            let f = f32::from_bits(*b);
            Obs::F32(if f.is_nan() { 0x7fc0_0000 } else { *b })
        }
        Val::F64(b) => {
            let f = f64::from_bits(*b);
            Obs::F64(if f.is_nan() {
                0x7ff8_0000_0000_0000
            } else {
                *b
            })
        }
        _ => return None,
    })
}

fn classify(err: &wasmtime::Error) -> Outcome {
    match err.downcast_ref::<Trap>() {
        Some(Trap::OutOfFuel) | Some(Trap::StackOverflow) => Outcome::Inconclusive,
        Some(_) => Outcome::Trapped,
        // Not a wasm trap (resource limit, host error): no conclusion.
        None => Outcome::Inconclusive,
    }
}

fn instantiate(engine: &Engine, module: &Module) -> Result<Side, Outcome> {
    let mut store = Store::new(
        engine,
        wasmtime::StoreLimitsBuilder::new()
            .memory_size(MAX_MEMORY_BYTES)
            .trap_on_grow_failure(false)
            .build(),
    );
    store.limiter(|l| l);
    store.set_fuel(FUEL).map_err(|_| Outcome::Inconclusive)?;
    match Instance::new(&mut store, module, &[]) {
        Ok(instance) => Ok(Side { store, instance }),
        Err(e) => Err(classify(&e)),
    }
}

fn call(side: &mut Side, name: &str, args: &[Val], n_results: usize) -> Outcome {
    let Some(Extern::Func(f)) = side.instance.get_export(&mut side.store, name) else {
        return Outcome::Inconclusive;
    };
    if side.store.set_fuel(FUEL).is_err() {
        return Outcome::Inconclusive;
    }
    let mut results = vec![Val::I32(0); n_results];
    match f.call(&mut side.store, args, &mut results) {
        Ok(()) => Outcome::Returned(results.iter().filter_map(observe).collect()),
        Err(e) => classify(&e),
    }
}

/// splitmix64: tiny deterministic generator for argument values.
fn next(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Deterministic argument sets for a signature: all-zero, all-one, and two
/// seeded sets. `None` if a parameter type is not numeric.
fn arg_sets(params: &[ValType], seed: u64) -> Option<Vec<Vec<Val>>> {
    let mut sets = Vec::new();
    let mut state = seed;
    for round in 0..4u8 {
        let mut vals = Vec::with_capacity(params.len());
        for p in params {
            let r = match round {
                0 => 0,
                1 => 1,
                _ => next(&mut state),
            };
            vals.push(match p {
                ValType::I32 => Val::I32(r as i32),
                ValType::I64 => Val::I64(r as i64),
                ValType::F32 => Val::F32(f32::from_bits(r as u32 & 0x7f7f_ffff).to_bits()),
                ValType::F64 => Val::F64(r & 0x7fef_ffff_ffff_ffff),
                _ => return None,
            });
        }
        sets.push(vals);
        if params.is_empty() {
            break;
        }
    }
    Some(sets)
}

/// Compare `original` against `optimized` (both wasm binaries) by executing
/// their exports under wasmtime. `seed` drives argument generation.
pub fn compare_modules(original: &[u8], optimized: &[u8], seed: u64) -> Verdict {
    let mut config = Config::new();
    config.consume_fuel(true);
    let Ok(engine) = Engine::new(&config) else {
        return Verdict::Skipped("engine");
    };
    let Ok(orig_mod) = Module::new(&engine, original) else {
        return Verdict::Skipped("original does not compile");
    };
    let opt_mod = match Module::new(&engine, optimized) {
        Ok(m) => m,
        Err(e) => return Verdict::Mismatch(format!("optimized module rejected by wasmtime: {e}")),
    };

    // Instantiation (runs the start function): compare trap-vs-no-trap.
    let (mut o, mut p) = match (
        instantiate(&engine, &orig_mod),
        instantiate(&engine, &opt_mod),
    ) {
        (Ok(o), Ok(p)) => (o, p),
        (Err(Outcome::Trapped), Err(Outcome::Trapped)) => {
            return Verdict::Skipped("both trap at instantiation")
        }
        (Err(Outcome::Trapped), Ok(_)) => {
            return Verdict::Mismatch("original traps at instantiation, optimized does not".into())
        }
        (Ok(_), Err(Outcome::Trapped)) => {
            return Verdict::Mismatch("optimized traps at instantiation, original does not".into())
        }
        _ => return Verdict::Skipped("instantiation inconclusive"),
    };

    // Exported functions of the original; each must survive optimization.
    let mut funcs = Vec::new();
    for e in orig_mod.exports() {
        if let ExternType::Func(ty) = e.ty() {
            funcs.push((e.name().to_string(), ty));
        }
    }
    funcs.truncate(MAX_EXPORTS);

    let mut calls = 0usize;
    // Once any call is inconclusive the two stores may legitimately differ,
    // so later calls and the final-state check are no longer comparable.
    let mut state_comparable = true;
    for (name, ty) in &funcs {
        if p.instance.get_export(&mut p.store, name).is_none() {
            return Verdict::Mismatch(format!("export `{name}` missing from optimized module"));
        }
        let params: Vec<ValType> = ty.params().collect();
        let n_results = ty.results().len();
        let Some(sets) = arg_sets(&params, seed) else {
            continue;
        };
        for args in sets {
            let a = call(&mut o, name, &args, n_results);
            let b = call(&mut p, name, &args, n_results);
            match (&a, &b) {
                (Outcome::Inconclusive, _) | (_, Outcome::Inconclusive) => {
                    state_comparable = false;
                }
                _ if a == b => calls += 1,
                _ => {
                    return Verdict::Mismatch(format!(
                        "export `{name}` args {args:?}: original {a:?}, optimized {b:?}"
                    ))
                }
            }
        }
    }

    if state_comparable {
        if let Some(m) = compare_state(&mut o, &mut p, &orig_mod) {
            return Verdict::Mismatch(m);
        }
    }
    Verdict::Equivalent { calls }
}

/// Compare exported memories (bytes) and globals (values) after the calls.
fn compare_state(o: &mut Side, p: &mut Side, orig_mod: &Module) -> Option<String> {
    for e in orig_mod.exports() {
        let name = e.name();
        match e.ty() {
            ExternType::Memory(_) => {
                let a = o.instance.get_memory(&mut o.store, name)?;
                let Some(b) = p.instance.get_memory(&mut p.store, name) else {
                    return Some(format!(
                        "memory export `{name}` missing from optimized module"
                    ));
                };
                if a.data(&o.store) != b.data(&p.store) {
                    return Some(format!("memory export `{name}` differs after execution"));
                }
            }
            ExternType::Global(_) => {
                let a = o.instance.get_global(&mut o.store, name)?;
                let Some(b) = p.instance.get_global(&mut p.store, name) else {
                    return Some(format!(
                        "global export `{name}` missing from optimized module"
                    ));
                };
                let (va, vb) = (a.get(&mut o.store), b.get(&mut p.store));
                if observe(&va) != observe(&vb) {
                    return Some(format!(
                        "global export `{name}`: original {va:?}, optimized {vb:?}"
                    ));
                }
            }
            _ => {}
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    fn wasm(src: &str) -> Vec<u8> {
        wat::parse_str(src).expect("valid wat")
    }

    const RET1: &str = r#"(module (func (export "f") (result i32) i32.const 1))"#;

    #[test]
    fn identical_modules_are_equivalent() {
        let w = wasm(RET1);
        assert_eq!(compare_modules(&w, &w, 7), Verdict::Equivalent { calls: 1 });
    }

    /// Negative control: a hand-mutated "optimized" module returning a
    /// different constant must be reported as a mismatch.
    #[test]
    fn different_constant_is_a_mismatch() {
        let orig = wasm(RET1);
        let bad = wasm(r#"(module (func (export "f") (result i32) i32.const 2))"#);
        assert!(matches!(
            compare_modules(&orig, &bad, 7),
            Verdict::Mismatch(_)
        ));
    }

    #[test]
    fn trap_vs_no_trap_is_a_mismatch() {
        let orig = wasm(r#"(module (func (export "f") (result i32) unreachable))"#);
        let bad = wasm(RET1);
        assert!(matches!(
            compare_modules(&orig, &bad, 7),
            Verdict::Mismatch(_)
        ));
        assert!(matches!(
            compare_modules(&bad, &orig, 7),
            Verdict::Mismatch(_)
        ));
    }

    #[test]
    fn argument_dependent_difference_is_caught() {
        let orig = wasm(r#"(module (func (export "f") (param i32) (result i32) local.get 0))"#);
        let bad = wasm(
            r#"(module (func (export "f") (param i32) (result i32)
                 local.get 0 i32.const 1 i32.add))"#,
        );
        assert!(matches!(
            compare_modules(&orig, &bad, 7),
            Verdict::Mismatch(_)
        ));
    }

    #[test]
    fn memory_difference_is_caught() {
        let mk = |v: i32| {
            wasm(&format!(
                r#"(module (memory (export "m") 1)
                    (func (export "f") i32.const 0 i32.const {v} i32.store))"#
            ))
        };
        assert!(matches!(
            compare_modules(&mk(1), &mk(2), 7),
            Verdict::Mismatch(_)
        ));
        assert_eq!(
            compare_modules(&mk(1), &mk(1), 7),
            Verdict::Equivalent { calls: 1 }
        );
    }

    #[test]
    fn global_difference_is_caught() {
        let mk = |v: i32| {
            wasm(&format!(
                r#"(module (global (export "g") i32 (i32.const {v})))"#
            ))
        };
        assert!(matches!(
            compare_modules(&mk(1), &mk(2), 7),
            Verdict::Mismatch(_)
        ));
    }

    #[test]
    fn missing_export_is_a_mismatch() {
        let orig = wasm(RET1);
        let bad = wasm(r#"(module (func (result i32) i32.const 1))"#);
        assert!(matches!(
            compare_modules(&orig, &bad, 7),
            Verdict::Mismatch(_)
        ));
    }

    #[test]
    fn infinite_loop_is_bounded_not_a_hang_or_mismatch() {
        let w = wasm(r#"(module (func (export "f") (loop br 0)))"#);
        assert_eq!(compare_modules(&w, &w, 7), Verdict::Equivalent { calls: 0 });
    }

    #[test]
    fn nan_payloads_are_not_a_divergence() {
        let a = wasm(r#"(module (func (export "f") (result f32) f32.const nan:0x200000))"#);
        let b = wasm(r#"(module (func (export "f") (result f32) f32.const nan))"#);
        assert_eq!(compare_modules(&a, &b, 7), Verdict::Equivalent { calls: 1 });
    }
}
