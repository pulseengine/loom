//! `loom optimize --stats-json` (TEST-O3-STATS-JSON-ENGINE).
//!
//! The JSON reports every validator attempt with its outcome and the engine
//! that decided it. These tests drive the real binary, each in its own
//! process, so the process-wide counters cannot interfere between tests.

use serde_json::Value;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(rel: &str) -> PathBuf {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join(rel);
    assert!(p.exists(), "missing fixture: {}", p.display());
    p
}

/// Run `loom optimize <input> --stats --stats-json` with `backend` (or the
/// default when `None`) and return the parsed JSON and stdout.
fn run(input: &Path, backend: Option<&str>, tag: &str) -> (Value, String) {
    let dir = std::env::temp_dir().join(format!("loom-stats-json-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let out_wasm = dir.join("out.wasm");
    let json_path = dir.join("stats.json");
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_loom"));
    cmd.args([
        "optimize",
        input.to_str().unwrap(),
        "-o",
        out_wasm.to_str().unwrap(),
        "--stats",
        "--stats-json",
        json_path.to_str().unwrap(),
    ]);
    cmd.env_remove("LOOM_VERIFY_BACKEND");
    if let Some(b) = backend {
        cmd.env("LOOM_VERIFY_BACKEND", b);
    }
    let out = cmd.output().expect("failed to run the loom binary");
    assert!(
        out.status.success(),
        "loom optimize failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let text = std::fs::read_to_string(&json_path).expect("--stats-json wrote no file");
    let json: Value = serde_json::from_str(&text).expect("--stats-json is not valid JSON");
    let _ = std::fs::remove_dir_all(&dir);
    (json, String::from_utf8_lossy(&out.stdout).into_owned())
}

fn u(v: &Value) -> u64 {
    v.as_u64()
        .unwrap_or_else(|| panic!("expected a number, got {v}"))
}

/// Every entered attempt reaches exactly one outcome, the outcome counts add
/// up to the records, and the JSON agrees with the `--stats` text.
///
/// `attempts.entered` is counted when the validator is entered, not summed
/// from the outcome buckets, so this equality can fail: it fails when an
/// attempt ends without an outcome or a revert is recorded outside any
/// attempt.
#[test]
fn every_attempt_reaches_one_outcome_and_matches_the_text_report() {
    let (j, stdout) = run(&fixture("tests/fixtures/test_phase7.wat"), None, "phase7");
    assert_eq!(j["schema"], "loom-stats/1");

    let a = &j["attempts"];
    let o = &j["outcomes"];
    let records = j["records"].as_array().expect("records array");
    assert!(u(&a["entered"]) > 0, "no attempt was entered: {j}");
    assert_eq!(u(&a["unclassified"]), 0, "attempts without an outcome: {j}");
    assert_eq!(
        u(&a["outcomes_without_entry"]),
        0,
        "outcomes without an attempt: {j}"
    );
    assert_eq!(u(&a["entered"]), u(&a["classified"]));
    assert_eq!(u(&a["classified"]), records.len() as u64);
    assert_eq!(a["consistent"], true);
    assert_eq!(
        u(&o["proven"]) + u(&o["kept_unproven"]) + u(&o["reverted"]),
        records.len() as u64
    );

    // The JSON and the text report count the same proven attempts.
    let proven = u(&o["proven"]);
    assert_eq!(proven, u(&j["text_counters"]["proven"]));
    assert!(
        stdout.contains(&format!("Proven:  {proven}/")),
        "the text report does not show {proven} proven attempts:\n{stdout}"
    );

    // Engine counts partition the proven attempts.
    let by_engine: u64 = o["proven_by_engine"]
        .as_object()
        .expect("proven_by_engine object")
        .values()
        .map(u)
        .sum();
    assert_eq!(by_engine, proven);
}

/// With Z3 deciding, no accepted attempt is reported as certified; with
/// ordeal deciding, every solver-decided attempt is ordeal's and certified.
/// The fixture has attempts a solver must decide (measured: 7 under Z3).
#[cfg(feature = "verification")]
#[test]
fn the_deciding_engine_follows_the_selected_backend() {
    let input = fixture("tests/fixtures/test_phase7.wat");

    let (z3, _) = run(&input, Some("z3"), "z3");
    let e = &z3["outcomes"]["proven_by_engine"];
    assert!(
        e["z3"].as_u64().unwrap_or(0) > 0,
        "no attempt was attributed to z3 under LOOM_VERIFY_BACKEND=z3: {e}"
    );
    assert_eq!(e["ordeal"].as_u64().unwrap_or(0), 0, "{e}");
    assert_eq!(u(&z3["outcomes"]["proven_certified"]), 0);

    let (ord, _) = run(&input, Some("ordeal"), "ordeal");
    let e = &ord["outcomes"]["proven_by_engine"];
    let ordeal = e["ordeal"].as_u64().unwrap_or(0);
    assert!(
        ordeal > 0,
        "no attempt was attributed to ordeal under LOOM_VERIFY_BACKEND=ordeal: {e}"
    );
    assert_eq!(e["z3"].as_u64().unwrap_or(0), 0, "{e}");
    assert_eq!(u(&ord["outcomes"]["proven_certified"]), ordeal);
}

/// A pass that changes nothing is reported as `unchanged`, not as a solver
/// proof. Measured on json_lite at main c717ce6: all 45 accepted attempts
/// compare a function with itself.
#[test]
fn unchanged_functions_are_not_reported_as_solver_proofs() {
    let (j, _) = run(&fixture("tests/corpus/json_lite.wasm"), None, "json_lite");
    let e = &j["outcomes"]["proven_by_engine"];
    let unchanged = e["unchanged"].as_u64().unwrap_or(0);
    assert!(unchanged > 0, "no unchanged attempt reported: {e}");
    let records = j["records"].as_array().expect("records array");
    let n = records
        .iter()
        .filter(|r| r["outcome"] == "proven" && r["engine"] == "unchanged")
        .count() as u64;
    assert_eq!(n, unchanged);
}

/// The report never carries a value it cannot vouch for as free of host
/// details: a non-numeric `LOOM_*` value is shown as `<set>`.
#[test]
fn budget_values_that_are_not_numbers_are_redacted() {
    let input = fixture("tests/fixtures/simple_module.wat");
    let dir = std::env::temp_dir().join(format!("loom-stats-json-redact-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let json_path = dir.join("stats.json");
    let out = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args([
            "optimize",
            input.to_str().unwrap(),
            "-o",
            dir.join("out.wasm").to_str().unwrap(),
            "--stats-json",
            json_path.to_str().unwrap(),
        ])
        .env("LOOM_TEST_PATH_LIKE", "/some/host/path")
        .env("LOOM_Z3_TIMEOUT_MS", "12345")
        .output()
        .expect("failed to run the loom binary");
    assert!(
        out.status.success(),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let j: Value = serde_json::from_str(&std::fs::read_to_string(&json_path).unwrap()).unwrap();
    let _ = std::fs::remove_dir_all(&dir);
    assert_eq!(j["budgets_set"]["LOOM_TEST_PATH_LIKE"], "<set>");
    assert_eq!(j["budgets_set"]["LOOM_Z3_TIMEOUT_MS"], "12345");
    assert!(!j.to_string().contains("/some/host/path"));
}

/// A revert a pass decides by its own check (here the size guard of
/// forward_carrier_locals) is reported apart from validator outcomes, so the
/// attempt accounting stays consistent. Measured at a725eaa: both inputs
/// reported an outcome without an attempt, because the size-guard revert was
/// counted as a validator rejection.
#[test]
fn pass_check_reverts_are_not_validator_outcomes() {
    for (input, tag) in [
        ("tests/fixtures/bench_locals.wat", "bench_locals"),
        ("tests/fixtures/quicksort.wat", "quicksort"),
    ] {
        let (j, stdout) = run(&fixture(input), None, tag);
        let a = &j["attempts"];
        assert_eq!(u(&a["outcomes_without_entry"]), 0, "{input}: {a}");
        assert_eq!(a["consistent"], true, "{input}: {a}");
        let p = &j["pass_check_reverts_by_key"];
        assert!(
            p["forward_carrier_locals_no_shrink"].as_u64().unwrap_or(0) > 0,
            "{input}: the size-guard revert this test depends on did not happen: {p}"
        );
        assert!(
            stdout.contains("Reverted by a pass check"),
            "{input}: the text report does not list the pass-check revert:\n{stdout}"
        );
        // The validator's revert count and the text report agree.
        assert_eq!(
            u(&j["outcomes"]["reverted"]),
            u(&j["text_counters"]["reverted"]),
            "{input}"
        );
    }
}
