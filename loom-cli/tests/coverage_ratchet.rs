//! Coverage ratchet over the repository's 30 baseline inputs (v1.6.0 plan,
//! Bet 1 item 5; TEST-O3-COVERAGE-RATCHET).
//!
//! For every input, `loom optimize --stats-json` is compared with the
//! committed baseline `safety/coverage-ratchet.json`. The ratchet fails when,
//! for any input:
//!
//! - `kept_unproven` rises,
//! - `proof + solver_verdict` falls (attempts decided by a solver: ordeal with
//!   a re-checked certificate, or Z3),
//! - the output grows by any number of bytes,
//! - `attempts` falls,
//!
//! or when an input is missing from either side. A rise in `solver_verdict`
//! alone does not fail it while Z3 is the default engine. Improvements pass;
//! commit a new baseline to lock them in.
//!
//! The live comparison runs 30 optimizations and the counts are not yet shown
//! to be equal across machines (plan item 3), so it is `#[ignore]`d:
//!
//! ```text
//! cargo test --release -p loom-cli --test coverage_ratchet -- --ignored
//! ```
//!
//! `LOOM_RATCHET_WRITE=<file>` writes the measured values as a new baseline
//! instead of comparing. The comparison itself (`violations`) is tested on
//! every run by planted regressions, one per condition.

use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

const BASELINE: &str = "safety/coverage-ratchet.json";
const SCHEMA: &str = "loom-coverage-ratchet/1";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Measure {
    attempts: u64,
    solver_proven: u64,
    kept_unproven: u64,
    output_bytes: u64,
}

type Measures = BTreeMap<String, Measure>;

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("loom-cli has a parent directory")
        .to_path_buf()
}

/// The 30 inputs, as repository-relative paths, sorted.
fn inputs(root: &Path) -> Vec<String> {
    fn list(dir: &Path, ext: &str, root: &Path, out: &mut Vec<String>) {
        let entries =
            std::fs::read_dir(dir).unwrap_or_else(|e| panic!("cannot read {}: {e}", dir.display()));
        for entry in entries.flatten() {
            let p = entry.path();
            if p.is_file() && p.extension().and_then(|e| e.to_str()) == Some(ext) {
                out.push(rel(&p, root));
            }
        }
    }
    let mut v = Vec::new();
    list(&root.join("tests/fixtures"), "wat", root, &mut v);
    list(&root.join("tests/corpus"), "wasm", root, &mut v);
    for f in [
        "loom-core/tests/fixtures/issue254-records-fused.wasm",
        "loom-core/tests/fixtures/repro-219/sem.loom.wasm",
    ] {
        assert!(root.join(f).is_file(), "missing input {f}");
        v.push(f.to_string());
    }
    v.sort();
    v
}

fn rel(p: &Path, root: &Path) -> String {
    p.strip_prefix(root)
        .expect("under the repository root")
        .to_str()
        .expect("UTF-8 path")
        .replace('\\', "/")
}

fn num(v: &Value, what: &str) -> u64 {
    v.as_u64()
        .unwrap_or_else(|| panic!("{what}: expected a number, got {v}"))
}

/// Run `loom optimize` on one input and read the four measures.
fn measure(root: &Path, input: &str, scratch: &Path) -> Measure {
    let out = scratch.join("out.wasm");
    let stats = scratch.join("stats.json");
    let _ = std::fs::remove_file(&out);
    let _ = std::fs::remove_file(&stats);
    let run = Command::new(env!("CARGO_BIN_EXE_loom"))
        .current_dir(root)
        .args(["optimize", input, "-o"])
        .arg(&out)
        .args(["--attestation", "false", "--stats-json"])
        .arg(&stats)
        .env_remove("LOOM_VERIFY_BACKEND")
        .output()
        .expect("run the loom binary");
    assert!(
        run.status.success(),
        "loom optimize {input} failed:\n{}",
        String::from_utf8_lossy(&run.stderr)
    );
    let j: Value =
        serde_json::from_str(&std::fs::read_to_string(&stats).expect("--stats-json wrote no file"))
            .expect("--stats-json is valid JSON");
    assert_eq!(
        j["attempts"]["consistent"], true,
        "{input}: inconsistent stats"
    );
    let by_engine = &j["outcomes"]["proven_by_engine"];
    let engine = |k: &str| by_engine.get(k).map_or(0, |v| num(v, k));
    Measure {
        attempts: num(&j["attempts"]["entered"], "attempts.entered"),
        solver_proven: engine("ordeal") + engine("z3"),
        kept_unproven: num(&j["outcomes"]["kept_unproven"], "kept_unproven"),
        output_bytes: std::fs::metadata(&out).expect("output written").len(),
    }
}

fn to_json(m: &Measures) -> Value {
    let mut inputs = Map::new();
    for (k, v) in m {
        inputs.insert(
            k.clone(),
            json!({
                "attempts": v.attempts,
                "proof_plus_solver_verdict": v.solver_proven,
                "kept_unproven": v.kept_unproven,
                "output_bytes": v.output_bytes,
            }),
        );
    }
    json!({ "schema": SCHEMA, "inputs": inputs })
}

fn from_json(j: &Value) -> Measures {
    assert_eq!(j["schema"], SCHEMA, "baseline schema");
    j["inputs"]
        .as_object()
        .expect("baseline inputs object")
        .iter()
        .map(|(k, v)| {
            (
                k.clone(),
                Measure {
                    attempts: num(&v["attempts"], "attempts"),
                    solver_proven: num(
                        &v["proof_plus_solver_verdict"],
                        "proof_plus_solver_verdict",
                    ),
                    kept_unproven: num(&v["kept_unproven"], "kept_unproven"),
                    output_bytes: num(&v["output_bytes"], "output_bytes"),
                },
            )
        })
        .collect()
}

/// Every way `current` is worse than `baseline`, one line each, naming the
/// condition. Empty means the ratchet holds.
fn violations(baseline: &Measures, current: &Measures) -> Vec<String> {
    let mut v = Vec::new();
    for (input, b) in baseline {
        let Some(c) = current.get(input) else {
            v.push(format!("{input}: input missing from this run"));
            continue;
        };
        if c.kept_unproven > b.kept_unproven {
            v.push(format!(
                "{input}: kept_unproven rose {} -> {}",
                b.kept_unproven, c.kept_unproven
            ));
        }
        if c.solver_proven < b.solver_proven {
            v.push(format!(
                "{input}: proof + solver_verdict fell {} -> {}",
                b.solver_proven, c.solver_proven
            ));
        }
        if c.output_bytes > b.output_bytes {
            v.push(format!(
                "{input}: output bytes grew {} -> {}",
                b.output_bytes, c.output_bytes
            ));
        }
        if c.attempts < b.attempts {
            v.push(format!(
                "{input}: attempts fell {} -> {}",
                b.attempts, c.attempts
            ));
        }
    }
    for input in current.keys().filter(|k| !baseline.contains_key(*k)) {
        v.push(format!("{input}: input missing from the baseline"));
    }
    v
}

#[test]
#[ignore = "runs 30 optimizations; see the module docs"]
fn coverage_does_not_regress_against_the_committed_baseline() {
    let root = repo_root();
    let scratch = std::env::temp_dir().join(format!("loom-ratchet-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).expect("scratch dir");
    let current: Measures = inputs(&root)
        .into_iter()
        .map(|i| {
            let m = measure(&root, &i, &scratch);
            (i, m)
        })
        .collect();
    let _ = std::fs::remove_dir_all(&scratch);
    let total = |f: fn(&Measure) -> u64| current.values().map(f).sum::<u64>();
    println!(
        "ratchet: {} inputs, {} attempts, {} proof + solver_verdict, {} kept_unproven, {} output bytes",
        current.len(),
        total(|m| m.attempts),
        total(|m| m.solver_proven),
        total(|m| m.kept_unproven),
        total(|m| m.output_bytes)
    );

    if let Some(path) = std::env::var_os("LOOM_RATCHET_WRITE") {
        let text = serde_json::to_string_pretty(&to_json(&current)).unwrap() + "\n";
        std::fs::write(&path, text).expect("write the new baseline");
        println!("ratchet: wrote {}", PathBuf::from(path).display());
        return;
    }

    let baseline = from_json(
        &serde_json::from_str(
            &std::fs::read_to_string(root.join(BASELINE)).expect("read the baseline"),
        )
        .expect("baseline is valid JSON"),
    );
    let v = violations(&baseline, &current);
    assert!(v.is_empty(), "coverage ratchet failed:\n{}", v.join("\n"));
}

fn base() -> Measures {
    let mut m = Measures::new();
    m.insert(
        "a.wat".into(),
        Measure {
            attempts: 10,
            solver_proven: 4,
            kept_unproven: 1,
            output_bytes: 100,
        },
    );
    m
}

fn changed(f: impl FnOnce(&mut Measure)) -> Measures {
    let mut m = base();
    f(m.get_mut("a.wat").unwrap());
    m
}

fn assert_one(v: Vec<String>, name: &str) {
    assert_eq!(v.len(), 1, "expected one violation naming {name:?}: {v:?}");
    assert!(v[0].contains(name), "expected {name:?} in {v:?}");
}

#[test]
fn control_kept_unproven_rising_fails() {
    assert_one(
        violations(&base(), &changed(|m| m.kept_unproven += 1)),
        "kept_unproven rose",
    );
}

#[test]
fn control_solver_proven_falling_fails() {
    assert_one(
        violations(&base(), &changed(|m| m.solver_proven -= 1)),
        "proof + solver_verdict fell",
    );
}

#[test]
fn control_output_growing_fails() {
    assert_one(
        violations(&base(), &changed(|m| m.output_bytes += 1)),
        "output bytes grew",
    );
}

#[test]
fn control_attempts_falling_fails() {
    assert_one(
        violations(&base(), &changed(|m| m.attempts -= 1)),
        "attempts fell",
    );
}

#[test]
fn control_missing_inputs_fail() {
    assert_one(
        violations(&base(), &Measures::new()),
        "missing from this run",
    );
    assert_one(
        violations(&Measures::new(), &base()),
        "missing from the baseline",
    );
}

#[test]
fn improvements_and_more_solver_verdicts_pass() {
    let better = changed(|m| {
        m.kept_unproven -= 1;
        m.solver_proven += 3;
        m.output_bytes -= 10;
        m.attempts += 5;
    });
    assert!(violations(&base(), &better).is_empty());
    assert!(violations(&base(), &base()).is_empty());
}

#[test]
fn baseline_round_trips_through_json() {
    let m = base();
    assert_eq!(from_json(&to_json(&m)), m);
}
