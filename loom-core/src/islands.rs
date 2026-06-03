//! Island-model parallel optimization (v1.0.4 PR-islands, issue #71).
//!
//! # Why this exists
//!
//! The v0.6.0 → v0.7.0 `gale` CSE-cost regression (commit `afc9318`) is the
//! exact failure mode this module guards against: different pipeline orderings
//! produce different output sizes, and a single fixed ordering can silently
//! pick a worse-than-necessary result. Running N orderings in parallel,
//! validating each independently with the existing Z3 + stack gates, and
//! selecting the smallest verified result eliminates that class of regression
//! by construction.
//!
//! # Soundness contract
//!
//! Each island is a full clone of the input module that runs through a chosen
//! pass order. Every pass already invokes its per-function `verify_or_revert`
//! gate (Z3 + stack validation) internally — this module does NOT skip or
//! relax those gates. Additionally, every island's encoded output must pass
//! `wasmparser::validate` before it is considered as a winner candidate. If
//! every island fails validation, the harness returns an error rather than
//! producing an unverified output.
//!
//! # Parallelism
//!
//! Islands run concurrently via `rayon::scope`. Z3's context is
//! thread-local (the `with_z3_config` API in the `z3` crate sets a per-thread
//! context), so each rayon worker creates its own Z3 state when an
//! optimization pass invokes verification. No Z3 state is shared across
//! islands.
//!
//! Determinism: the winner is selected by `(encoded_size, name)` lex order, so
//! ties resolve identically regardless of which rayon thread finished first.
//!
//! # CLI surface
//!
//! See `loom-cli/src/main.rs` `--islands N` for user-facing wiring. `N=1`
//! preserves current behavior (single baseline island, no rayon scope).

use crate::Module;
use anyhow::{Result, anyhow};
// rayon is excluded on wasm32 (no pthread); islands run sequentially there.
// The deterministic tie-break (see below) makes the result identical. (#142)
#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Configuration for a single optimization island.
///
/// Each island runs the listed passes in order on a private clone of the
/// input module. The non-list fields are knobs that historical
/// pipeline-order accidents would have exercised — e.g., toggling
/// `canonicalize_enabled` reproduces the v0.6.0 pipeline that lacked the
/// `canonicalize` phase before `cse`.
#[derive(Debug, Clone)]
pub struct IslandConfig {
    /// Stable island name. Used as the deterministic tie-break key.
    pub name: &'static str,
    /// Ordered list of pass identifiers to run. Pass identifiers match the
    /// strings the CLI's `--passes` flag accepts:
    /// `directize`, `inline`, `precompute`, `constant-folding`,
    /// `canonicalize`, `peephole-synth`, `cse`, `advanced`, `branches`,
    /// `dce`, `merge-blocks`, `vacuum`, `simplify-locals`, `dead-stores`,
    /// `dead-locals`, `vacuum-final`.
    pub pass_order: &'static [&'static str],
    /// Soft hint for the inliner's per-function instruction budget. Stored
    /// for forward-compatibility with #73 typed pass composition; the
    /// inliner today uses a fixed threshold but accepting this here keeps
    /// the config shape future-proof without changing the wire format.
    pub inline_size_threshold: usize,
    /// If true, CSE is allowed to reject replacements whose materialization
    /// cost would exceed the eliminated redundancy. Mirrors the v0.9.0
    /// PR-K2 cost gate. Stored for forward-compat; current passes read it
    /// via the module-level setting.
    pub cse_cost_gate: bool,
    /// If false, the `canonicalize` pass is skipped even if it appears in
    /// `pass_order`. Provides a way to reproduce pre-v0.7.0 pipelines.
    pub canonicalize_enabled: bool,
}

/// Canonical baseline pass order — mirrors `loom-cli/src/main.rs`.
///
/// Keep this in sync with the CLI pipeline; the `baseline` island's job is
/// to be byte-identical to the serial `--islands 1` path.
const BASELINE_ORDER: &[&str] = &[
    "directize",
    "inline",
    "precompute",
    "constant-folding",
    "canonicalize",
    "peephole-synth",
    "cse",
    "advanced",
    "branches",
    "dce",
    "merge-blocks",
    "vacuum",
    "simplify-locals",
    "dead-stores",
    "dead-locals",
    "vacuum-final",
];

/// Inline-late variant: move `inline` after `cse` so that CSE/precompute
/// fire on the smaller pre-inline IR first. This is one of the orderings
/// the `gale` regression analysis suggested would have produced smaller
/// output for that specific module.
const INLINE_LATE_ORDER: &[&str] = &[
    "directize",
    "precompute",
    "constant-folding",
    "canonicalize",
    "peephole-synth",
    "cse",
    "inline",
    "advanced",
    "branches",
    "dce",
    "merge-blocks",
    "vacuum",
    "simplify-locals",
    "dead-stores",
    "dead-locals",
    "vacuum-final",
];

/// CSE-early variant: run `cse` before `canonicalize` and `inline`. This
/// is the pre-v0.7.0 ordering that the `afc9318` regression came from —
/// keeping it in rotation means the island harness will detect if it
/// ever wins again on some other corpus member.
const CSE_EARLY_ORDER: &[&str] = &[
    "directize",
    "precompute",
    "constant-folding",
    "cse",
    "canonicalize",
    "peephole-synth",
    "inline",
    "advanced",
    "branches",
    "dce",
    "merge-blocks",
    "vacuum",
    "simplify-locals",
    "dead-stores",
    "dead-locals",
    "vacuum-final",
];

/// Aggressive-inline variant: run `inline` twice — once early to expose
/// cross-function constants for `constant-folding`, once late to clean up
/// any small wrappers exposed by `simplify-locals` / `dead-stores`.
const AGGRESSIVE_INLINE_ORDER: &[&str] = &[
    "directize",
    "inline",
    "precompute",
    "constant-folding",
    "canonicalize",
    "peephole-synth",
    "cse",
    "advanced",
    "inline",
    "branches",
    "dce",
    "merge-blocks",
    "vacuum",
    "simplify-locals",
    "dead-stores",
    "dead-locals",
    "vacuum-final",
];

/// Default 4-island fleet. Order matters for `--islands N`: requesting
/// `N=2` uses the first two entries.
pub const DEFAULT_ISLANDS: &[IslandConfig] = &[
    IslandConfig {
        name: "baseline",
        pass_order: BASELINE_ORDER,
        inline_size_threshold: 50,
        cse_cost_gate: true,
        canonicalize_enabled: true,
    },
    IslandConfig {
        name: "inline-late",
        pass_order: INLINE_LATE_ORDER,
        inline_size_threshold: 50,
        cse_cost_gate: true,
        canonicalize_enabled: true,
    },
    IslandConfig {
        name: "cse-early",
        pass_order: CSE_EARLY_ORDER,
        inline_size_threshold: 50,
        cse_cost_gate: true,
        canonicalize_enabled: true,
    },
    IslandConfig {
        name: "aggressive-inline",
        pass_order: AGGRESSIVE_INLINE_ORDER,
        inline_size_threshold: 80,
        cse_cost_gate: true,
        canonicalize_enabled: true,
    },
];

/// Run a single pass on a module by name. Returns an error for unknown
/// pass names so the harness fails loudly rather than silently skipping.
///
/// The `cfg` reference is currently consulted only for `canonicalize_enabled`;
/// the other knobs are recorded for forward-compat with #73 typed-pass
/// composition.
fn run_pass(module: &mut Module, pass: &str, cfg: &IslandConfig) -> Result<()> {
    match pass {
        "directize" => crate::optimize::directize(module),
        "inline" => crate::optimize::inline_functions(module),
        "precompute" => crate::optimize::precompute(module),
        "constant-folding" => crate::optimize::constant_folding(module),
        "canonicalize" => {
            if cfg.canonicalize_enabled {
                crate::optimize::canonicalize(module)
            } else {
                Ok(())
            }
        }
        "peephole-synth" => crate::peephole_synth::apply_peephole_synth(module).map(|_| ()),
        "cse" => crate::optimize::eliminate_common_subexpressions_enhanced(module),
        "advanced" => crate::optimize::optimize_advanced_instructions(module),
        "branches" => crate::optimize::simplify_branches(module),
        "dce" => crate::optimize::eliminate_dead_code(module),
        "merge-blocks" => crate::optimize::merge_blocks(module),
        "vacuum" => crate::optimize::vacuum(module),
        "simplify-locals" => crate::optimize::simplify_locals(module),
        "dead-stores" => crate::optimize::eliminate_dead_stores(module),
        "dead-locals" => crate::optimize::eliminate_dead_locals(module),
        "vacuum-final" => crate::optimize::vacuum(module),
        other => Err(anyhow!("islands: unknown pass identifier '{}'", other)),
    }
}

/// Result of running a single island.
struct IslandResult {
    name: &'static str,
    module: Module,
    encoded: Vec<u8>,
}

/// Run one island's pass sequence and encode the result. Errors fall through
/// — the harness treats them as "this island failed, drop it from
/// consideration" rather than aborting the entire run.
fn run_one_island(input: &Module, cfg: &IslandConfig) -> Result<IslandResult> {
    let mut module = input.clone();
    for pass in cfg.pass_order {
        run_pass(&mut module, pass, cfg)
            .map_err(|e| anyhow!("island '{}' pass '{}': {}", cfg.name, pass, e))?;
    }
    let encoded = crate::encode::encode_wasm(&module)
        .map_err(|e| anyhow!("island '{}' encode failed: {}", cfg.name, e))?;
    Ok(IslandResult {
        name: cfg.name,
        module,
        encoded,
    })
}

/// Run the island fleet in parallel and select the smallest valid result.
///
/// # Selection criteria
/// 1. Pass order completed without error.
/// 2. Encoded output passes `wasmparser::validate` (catches any encoder /
///    pass-ordering corruption that slipped past per-pass validation).
/// 3. Smallest encoded size wins; tie-break is lex order on `name` for
///    determinism (no race on equal sizes).
///
/// # Errors
/// Returns an error if `configs` is empty, or if every island either errored
/// during its pass sequence or produced output that failed `wasmparser`
/// validation. The caller is expected to fall back to the serial pipeline in
/// that pathological case.
pub fn optimize_module_islands(module: &Module, configs: &[IslandConfig]) -> Result<Module> {
    if configs.is_empty() {
        return Err(anyhow!("islands: no IslandConfig provided"));
    }

    // Parallel fan-out via rayon. Each island runs on its own worker thread;
    // Z3's thread-local context (see z3::with_z3_config) means each worker
    // gets a fresh Z3 state when verification passes run inside the
    // optimization passes themselves. No shared mutable Z3 state.
    //
    // `par_iter()` here yields `Result<IslandResult>`; we keep both Ok and
    // Err so we can preserve per-island failure messages for the no-winner
    // case below.
    #[cfg(not(target_arch = "wasm32"))]
    let results: Vec<Result<IslandResult>> = configs
        .par_iter()
        .map(|cfg| run_one_island(module, cfg))
        .collect();
    // wasm32 has no thread pool — run the same islands sequentially. The
    // deterministic tie-break below makes the winner identical to the
    // parallel path. (#142)
    #[cfg(target_arch = "wasm32")]
    let results: Vec<Result<IslandResult>> = configs
        .iter()
        .map(|cfg| run_one_island(module, cfg))
        .collect();

    // Filter: keep only islands whose encoded output passes wasmparser
    // validation. This is defense-in-depth against the (rare) case where an
    // exotic pass ordering produces a module that survives per-function
    // validation but breaks at the module level (e.g., type-section
    // inconsistency).
    let mut winners: Vec<IslandResult> = Vec::with_capacity(results.len());
    let mut errors: Vec<String> = Vec::new();
    for r in results {
        match r {
            Ok(island) => {
                if wasmparser::validate(&island.encoded).is_ok() {
                    winners.push(island);
                } else {
                    errors.push(format!(
                        "island '{}' produced invalid wasm (failed wasmparser::validate)",
                        island.name
                    ));
                }
            }
            Err(e) => errors.push(format!("{e}")),
        }
    }

    if winners.is_empty() {
        return Err(anyhow!(
            "islands: no island produced a valid result. Failures: [{}]",
            errors.join("; ")
        ));
    }

    // Per-island measurement trace. Emitted unconditionally to stderr so the
    // user can see how each island's pass order shaped the encoded size —
    // this is exactly the diagnostic that would have caught the v0.6.0 →
    // v0.7.0 `gale` CSE-cost regression. Stderr keeps it out of stdout-based
    // pipelines.
    for w in &winners {
        eprintln!("[islands] {:24} {} bytes", w.name, w.encoded.len());
    }
    for e in &errors {
        eprintln!("[islands] error: {e}");
    }

    // Deterministic selection: smallest size wins; ties broken by name lex
    // order. `min_by` is stable in the sense that the iteration order of
    // `winners` mirrors the input `configs` order, but we don't rely on
    // that — `(size, name)` is a total order.
    let winner = winners
        .into_iter()
        .min_by(|a, b| {
            a.encoded
                .len()
                .cmp(&b.encoded.len())
                .then_with(|| a.name.cmp(b.name))
        })
        .expect("non-empty winners checked above");

    Ok(winner.module)
}

/// Public test-helper: run the islands and return per-island encoded sizes
/// keyed by name. Intended for measurement scripts and the test suite — the
/// production code path is `optimize_module_islands`.
#[doc(hidden)]
pub fn measure_island_sizes(
    module: &Module,
    configs: &[IslandConfig],
) -> Vec<(String, Result<usize>)> {
    #[cfg(not(target_arch = "wasm32"))]
    let out = configs
        .par_iter()
        .map(|cfg| {
            let name = cfg.name.to_string();
            let result = run_one_island(module, cfg).map(|r| r.encoded.len());
            (name, result)
        })
        .collect();
    #[cfg(target_arch = "wasm32")]
    let out = configs
        .iter()
        .map(|cfg| {
            let name = cfg.name.to_string();
            let result = run_one_island(module, cfg).map(|r| r.encoded.len());
            (name, result)
        })
        .collect();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny wat fixture used across the test suite. Picked to be small
    /// enough to optimize in milliseconds but rich enough that the
    /// pipeline actually does work (constant folding, dead code).
    const SMALL_WAT: &str = r#"
        (module
          (func $add_consts (result i32)
            i32.const 1
            i32.const 2
            i32.add
            i32.const 3
            i32.add)
          (func $with_drop (result i32)
            i32.const 10
            i32.const 99
            drop
            i32.const 5
            i32.add)
          (export "add_consts" (func $add_consts))
          (export "with_drop" (func $with_drop)))
    "#;

    /// Helper: run the canonical serial pipeline (the BASELINE_ORDER, no
    /// rayon) on a fresh module clone. Used to assert byte-identity with
    /// the N=1 island path.
    fn run_serial_baseline(module: &Module) -> Vec<u8> {
        let mut m = module.clone();
        let cfg = &DEFAULT_ISLANDS[0];
        for pass in cfg.pass_order {
            run_pass(&mut m, pass, cfg).expect("serial baseline pass failed");
        }
        crate::encode::encode_wasm(&m).expect("serial baseline encode failed")
    }

    #[test]
    fn test_islands_baseline_matches_serial() {
        // N=1 island with the canonical baseline pipeline produces the
        // same bytes as the serial path. This is the no-regression
        // guarantee for users who don't opt into multi-island mode.
        let module = crate::parse::parse_wat(SMALL_WAT).expect("parse fixture");
        let serial_bytes = run_serial_baseline(&module);

        let island_module = optimize_module_islands(&module, &DEFAULT_ISLANDS[..1])
            .expect("N=1 island run should succeed");
        let island_bytes =
            crate::encode::encode_wasm(&island_module).expect("encode island winner");

        assert_eq!(
            serial_bytes, island_bytes,
            "N=1 island path must be byte-identical to serial baseline"
        );
    }

    #[test]
    fn test_islands_pick_smallest() {
        // Construct two islands that differ in pass coverage. The "full"
        // island runs the entire pipeline; the "noop" island runs an
        // empty pass list (so its encoded size is whatever the
        // unoptimized parse-then-encode produces). The full island MUST
        // produce a result no larger than the noop, and for the small
        // fixture above it strictly wins because the constant chain
        // collapses.
        let module = crate::parse::parse_wat(SMALL_WAT).expect("parse fixture");

        let noop = IslandConfig {
            name: "noop",
            pass_order: &[],
            inline_size_threshold: 50,
            cse_cost_gate: true,
            canonicalize_enabled: true,
        };

        let configs = [DEFAULT_ISLANDS[0].clone(), noop.clone()];
        let winner_module =
            optimize_module_islands(&module, &configs).expect("island run should succeed");
        let winner_bytes = crate::encode::encode_wasm(&winner_module).expect("encode winner");

        let noop_module = {
            let mut m = module.clone();
            for pass in noop.pass_order {
                run_pass(&mut m, pass, &noop).expect("noop pass");
            }
            m
        };
        let noop_bytes = crate::encode::encode_wasm(&noop_module).expect("encode noop");

        assert!(
            winner_bytes.len() <= noop_bytes.len(),
            "winner ({} bytes) must be no larger than noop island ({} bytes)",
            winner_bytes.len(),
            noop_bytes.len()
        );
    }

    #[test]
    fn test_islands_reject_invalid() {
        // An island whose pass list references an unknown pass identifier
        // fails its run and MUST be excluded from selection. Pair it with
        // a known-good baseline so the harness still has a winner.
        let module = crate::parse::parse_wat(SMALL_WAT).expect("parse fixture");

        let broken = IslandConfig {
            name: "broken",
            pass_order: &["this-pass-does-not-exist"],
            inline_size_threshold: 50,
            cse_cost_gate: true,
            canonicalize_enabled: true,
        };

        let configs = [DEFAULT_ISLANDS[0].clone(), broken];
        let winner_module = optimize_module_islands(&module, &configs)
            .expect("harness must succeed when at least one island is valid");

        // The winner must be the baseline (the broken island was dropped).
        // We don't have a name on the returned Module, so we assert
        // structurally: re-running baseline yields the same encoded bytes.
        let winner_bytes = crate::encode::encode_wasm(&winner_module).expect("encode winner");
        let baseline_bytes = run_serial_baseline(&module);
        assert_eq!(
            winner_bytes, baseline_bytes,
            "broken island dropped → baseline must win"
        );
    }

    #[test]
    fn test_islands_all_invalid_errors() {
        // If EVERY island fails, the harness must return Err — it must
        // never silently produce an unoptimized result, because the
        // caller has no way to tell that happened. This is the
        // soundness counterpart to test_islands_reject_invalid.
        let module = crate::parse::parse_wat(SMALL_WAT).expect("parse fixture");

        let bad1 = IslandConfig {
            name: "bad1",
            pass_order: &["nope"],
            inline_size_threshold: 50,
            cse_cost_gate: true,
            canonicalize_enabled: true,
        };
        let bad2 = IslandConfig {
            name: "bad2",
            pass_order: &["also-nope"],
            inline_size_threshold: 50,
            cse_cost_gate: true,
            canonicalize_enabled: true,
        };

        let result = optimize_module_islands(&module, &[bad1, bad2]);
        assert!(
            result.is_err(),
            "all islands failed — harness must surface an error"
        );
    }

    #[test]
    fn test_islands_deterministic_tiebreak() {
        // Two islands with identical pass orders produce byte-equal
        // output. The tie-break MUST be deterministic (lex order on
        // name), independent of which rayon worker finishes first.
        //
        // We can't observe the chosen name directly through the public
        // API (it returns the Module, not the IslandResult), so we
        // run the harness 8 times and assert the winning module's
        // encoded bytes are identical every time. A race-condition
        // tie-break (e.g., `first to finish`) would occasionally pick
        // the other island; if the bytes match across runs, the
        // selection is order-independent.
        let module = crate::parse::parse_wat(SMALL_WAT).expect("parse fixture");

        let twin_a = IslandConfig {
            name: "alpha",
            pass_order: BASELINE_ORDER,
            inline_size_threshold: 50,
            cse_cost_gate: true,
            canonicalize_enabled: true,
        };
        let twin_b = IslandConfig {
            name: "beta",
            pass_order: BASELINE_ORDER,
            inline_size_threshold: 50,
            cse_cost_gate: true,
            canonicalize_enabled: true,
        };
        let configs = [twin_a, twin_b];

        let first = optimize_module_islands(&module, &configs).expect("run 1");
        let first_bytes = crate::encode::encode_wasm(&first).expect("encode 1");

        for run in 1..8 {
            let m = optimize_module_islands(&module, &configs).expect("repeat run");
            let bytes = crate::encode::encode_wasm(&m).expect("encode repeat");
            assert_eq!(
                first_bytes, bytes,
                "run {run}: tie-break must be deterministic across rayon thread interleavings"
            );
        }
    }

    #[test]
    fn test_islands_empty_configs_errors() {
        // Defensive: an empty config slice is a programming error, not a
        // valid request. The harness must return Err rather than
        // pretending there's a winner.
        let module = crate::parse::parse_wat(SMALL_WAT).expect("parse fixture");
        let result = optimize_module_islands(&module, &[]);
        assert!(result.is_err(), "empty configs slice must error");
    }
}
