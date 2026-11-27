//! `loom optimize` leaves a core module whose element segments reference
//! functions unoptimized, as `optimize_module` and the component optimizer
//! do (#196, #345). Structural checks and per-function translation validation
//! cannot see a scrambled or stale function-pointer table, so no unproven
//! table rewrite may reach the output.

use std::path::{Path, PathBuf};
use std::process::Command;

mod common;
use common::without_element_section;

const REASON: &str = "function-referencing element table, pending a behavioral gate (#196)";

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../loom-core/tests/fixtures/issue254-records-fused.wasm")
}

/// Runs `loom optimize <input> -o <dir>/out.wasm --stats-json <dir>/stats.json`
/// and returns (stdout, output bytes, parsed stats).
fn optimize(input: &Path, dir: &Path) -> (String, Vec<u8>, serde_json::Value) {
    let out_wasm = dir.join("out.wasm");
    let stats = dir.join("stats.json");
    let out = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args([
            "optimize",
            input.to_str().unwrap(),
            "-o",
            out_wasm.to_str().unwrap(),
            "--stats-json",
            stats.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run the loom binary");
    assert!(
        out.status.success(),
        "loom optimize failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let doc = serde_json::from_slice(&std::fs::read(&stats).expect("read stats json"))
        .expect("parse stats json");
    (
        String::from_utf8_lossy(&out.stdout).into_owned(),
        std::fs::read(&out_wasm).expect("read output"),
        doc,
    )
}

fn scratch(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("loom-345-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn a_module_with_a_function_table_comes_out_byte_identical_and_says_why() {
    let input = fixture();
    let dir = scratch("table");
    let (stdout, output, stats) = optimize(&input, &dir);

    assert_eq!(
        output,
        std::fs::read(&input).expect("read fixture"),
        "the CLI rewrote a module whose element segments reference functions (#345)"
    );
    assert!(
        stdout.contains(REASON),
        "the skip must be reported on stdout. Output:\n{stdout}"
    );
    assert_eq!(stats["module_skipped"], REASON, "stats: {stats}");
    assert_eq!(
        stats["attempts"]["entered"], 0,
        "no pass may run on a skipped module. stats: {stats}"
    );
}

/// Control: the same module without its element section is optimized, and
/// `module_skipped` is null. Without this, the test above passes for a CLI
/// that never optimizes anything.
#[test]
fn the_same_module_without_its_element_section_is_optimized() {
    let dir = scratch("control");
    let input = dir.join("no-elements.wasm");
    let bytes = without_element_section(&std::fs::read(fixture()).expect("read fixture"));
    std::fs::write(&input, &bytes).expect("write input");
    let (_, output, stats) = optimize(&input, &dir);

    assert_ne!(output, bytes, "the element-free module was not optimized");
    assert!(stats["module_skipped"].is_null(), "stats: {stats}");
    assert!(
        stats["attempts"]["entered"].as_u64().unwrap_or(0) > 0,
        "stats: {stats}"
    );
}
