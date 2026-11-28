//! Output-size gate (#555): `loom optimize` does not write an output larger
//! than its input unless `--allow-growth` is given.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("loom-555-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Runs `loom optimize <input> -o <dir>/out.wasm --stats-json <dir>/stats.json
/// [extra]` and returns (stdout, output bytes, size_gate record).
fn optimize(input: &Path, dir: &Path, extra: &[&str]) -> (String, Vec<u8>, serde_json::Value) {
    let out_wasm = dir.join("out.wasm");
    let stats = dir.join("stats.json");
    let mut args = vec![
        "optimize".to_string(),
        input.to_str().unwrap().to_string(),
        "-o".to_string(),
        out_wasm.to_str().unwrap().to_string(),
        "--stats-json".to_string(),
        stats.to_str().unwrap().to_string(),
    ];
    args.extend(extra.iter().map(|s| s.to_string()));
    let out = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args(&args)
        .output()
        .expect("failed to run the loom binary");
    assert!(
        out.status.success(),
        "loom optimize failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let doc: serde_json::Value =
        serde_json::from_slice(&std::fs::read(&stats).expect("read stats json"))
            .expect("parse stats json");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        std::fs::read(&out_wasm).expect("read output"),
        doc["size_gate"].clone(),
    )
}

/// inline-single-site.wasm (85 bytes) grows to 97 bytes under `--passes inline`:
/// the single-site $mix is inlined and, without dce-functions, also kept.
const GROWS: &str = "../loom-core/tests/fixtures/inline-single-site.wasm";
const INLINE_ONLY: [&str; 2] = ["--passes", "inline"];

#[test]
fn a_module_that_grows_is_written_unchanged_and_the_gate_says_so() {
    let input = fixture(GROWS);
    let (stdout, output, gate) = optimize(&input, &scratch("grows"), &INLINE_ONLY);

    assert_eq!(
        output,
        std::fs::read(&input).expect("read fixture"),
        "a larger optimized module was written"
    );
    assert!(
        stdout.contains("Output-size gate") && stdout.contains("writing the input unchanged"),
        "{stdout}"
    );
    assert_eq!(gate["written"], "input", "{gate}");
    assert_eq!(gate["allow_growth"], false, "{gate}");
    assert!(
        gate["optimized_bytes"].as_u64() > gate["input_bytes"].as_u64(),
        "{gate}"
    );
}

#[test]
fn allow_growth_keeps_the_larger_output() {
    let input = fixture(GROWS);
    let input_len = std::fs::metadata(&input).expect("stat fixture").len() as usize;
    let (_, output, gate) = optimize(
        &input,
        &scratch("allow"),
        &["--passes", "inline", "--allow-growth"],
    );

    assert!(output.len() > input_len, "output {} bytes", output.len());
    assert_eq!(gate["written"], "optimized", "{gate}");
    assert_eq!(gate["allow_growth"], true, "{gate}");
}

/// Control: a module that shrinks is written optimized, with no gate message.
#[test]
fn a_module_that_shrinks_is_written_optimized() {
    let input = fixture("../tests/corpus/state_machine.wasm");
    let input_bytes = std::fs::read(&input).expect("read fixture");
    let (stdout, output, gate) = optimize(&input, &scratch("shrinks"), &[]);

    assert!(
        output.len() < input_bytes.len(),
        "output {} bytes",
        output.len()
    );
    assert!(!stdout.contains("Output-size gate"), "{stdout}");
    assert_eq!(gate["written"], "optimized", "{gate}");
}

/// inline-grows.wasm: the exported $mix has four call sites. Inlining it copies
/// the body four times and keeps the original, 97 -> 145 bytes.
const MULTI_SITE: &str = "../loom-core/tests/fixtures/inline-grows.wasm";

#[test]
fn the_inline_budget_keeps_a_multi_site_callee_out_by_default() {
    // Budget 0%: no single-site inline pays for copying $mix, so it stays a
    // call and the module does not grow, before the size gate is consulted.
    let input = fixture(MULTI_SITE);
    let (stdout, output, gate) = optimize(&input, &scratch("budget"), &[]);

    assert!(!stdout.contains("Output-size gate"), "{stdout}");
    assert_eq!(gate["written"], "optimized", "{gate}");
    assert!(
        gate["optimized_bytes"].as_u64() <= gate["input_bytes"].as_u64(),
        "{gate}"
    );
    assert!(
        output.len() <= std::fs::metadata(&input).expect("stat").len() as usize,
        "output {} bytes",
        output.len()
    );
}

/// Control: with --allow-growth there is no budget, and $mix is inlined as
/// before #555.
#[test]
fn allow_growth_lifts_the_inline_budget() {
    let input = fixture(MULTI_SITE);
    let input_len = std::fs::metadata(&input).expect("stat fixture").len() as usize;
    let (_, output, gate) = optimize(&input, &scratch("unbudgeted"), &["--allow-growth"]);

    assert!(output.len() > input_len, "output {} bytes", output.len());
    assert_eq!(gate["written"], "optimized", "{gate}");
}
