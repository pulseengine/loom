//! How `loom optimize` treats a core module whose element segments reference
//! functions (#196, #345).
//!
//! Such a module is optimized when its tables stay sound: function renumbering
//! rewrites the element section, and `directize` folds only slots fixed for the
//! whole run. The exception is a module whose GC type identity loom's encoder
//! does not preserve: re-declaring a function under a structurally equal but
//! distinct type would turn a `call_indirect` signature trap into a call. That
//! module is written out unchanged, and the reason is reported.

use std::path::{Path, PathBuf};
use std::process::Command;

const REASON: &str =
    "function-referencing element table with GC type identity loom does not preserve (#196)";

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../loom-core/tests/fixtures")
        .join(name)
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
    let dir = std::env::temp_dir().join(format!("loom-196-{name}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

#[test]
fn a_table_module_with_a_non_final_function_type_comes_out_byte_identical() {
    // `$f` is declared with a non-final `(sub (func (result i32)))`, so
    // `call_indirect (type $b)` on it traps. Re-declared under the final `$b`,
    // the call would return 42.
    let input = fixture("table-sub-type.wasm");
    let dir = scratch("sub-type");
    let (stdout, output, stats) = optimize(&input, &dir);

    assert_eq!(
        output,
        std::fs::read(&input).expect("read fixture"),
        "the CLI rewrote a table module whose type identity it does not preserve"
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

/// Control: a 724-function meld-fused module with function tables and MVP
/// types is optimized, and `module_skipped` is null. Before #196 it was
/// skipped like the module above.
#[test]
fn a_table_module_with_mvp_types_is_optimized() {
    let input = fixture("issue254-records-fused.wasm");
    let dir = scratch("mvp");
    let (_, output, stats) = optimize(&input, &dir);

    assert_ne!(
        output,
        std::fs::read(&input).expect("read fixture"),
        "the table module was not optimized"
    );
    assert!(stats["module_skipped"].is_null(), "stats: {stats}");
    assert!(
        stats["attempts"]["entered"].as_u64().unwrap_or(0) > 0,
        "stats: {stats}"
    );
}
