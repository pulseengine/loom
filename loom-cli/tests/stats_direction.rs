//! `loom optimize --stats` names the direction of every change (#554).
//!
//! Growth used to print as a negative reduction ("-17.9% reduction",
//! "-650461 bytes reduction", and "-34 instructions" for a pass that added
//! 34), so a module that got larger read as if it got smaller.

use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join(rel)
}

/// Runs `loom optimize <input> -o <tmp>/out.wasm --stats` and returns stdout.
fn stats_stdout(input: &Path, name: &str) -> String {
    let out_wasm = std::env::temp_dir().join(format!(
        "loom-stats-direction-{}-{}.wasm",
        name,
        std::process::id()
    ));
    let out = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args([
            "optimize",
            input.to_str().unwrap(),
            "-o",
            out_wasm.to_str().unwrap(),
            "--stats",
        ])
        .output()
        .expect("failed to run the loom binary");
    let _ = std::fs::remove_file(&out_wasm);
    assert!(
        out.status.success(),
        "loom optimize failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8(out.stdout).unwrap()
}

fn line<'a>(stdout: &'a str, prefix: &str) -> &'a str {
    stdout
        .lines()
        .map(str::trim_start)
        .find(|l| l.starts_with(prefix))
        .unwrap_or_else(|| panic!("no line starting with {prefix:?} in:\n{stdout}"))
}

/// A negative number directly before "% reduction" or " bytes reduction".
fn has_negative_reduction(s: &str) -> bool {
    s.match_indices('-').any(|(i, _)| {
        let rest = &s[i + 1..];
        let digits = rest
            .find(|c: char| !(c.is_ascii_digit() || c == '.'))
            .unwrap_or(rest.len());
        digits > 0 && {
            let tail = &rest[digits..];
            tail.starts_with("% reduction") || tail.starts_with(" bytes reduction")
        }
    })
}

#[test]
fn growth_is_reported_as_growth() {
    // The exported $mix is inlined at four call sites and kept, so both the
    // instruction count and the binary grow.
    let stdout = stats_stdout(
        &fixture("../loom-core/tests/fixtures/inline-grows.wasm"),
        "grows",
    );

    let instr = line(&stdout, "Instructions:");
    assert!(instr.ends_with("% growth)"), "{instr}");
    assert!(instr.contains("(+"), "{instr}");

    let bytes = line(&stdout, "Binary size:");
    assert!(bytes.ends_with("% growth)"), "{bytes}");

    let effect = line(&stdout, "Optimization effect:");
    assert!(effect.ends_with("bytes growth"), "{effect}");
    assert!(effect.contains(": +"), "{effect}");

    let inline = line(&stdout, "inline ");
    assert!(
        inline.contains("(+") && inline.ends_with("instructions)"),
        "{inline}"
    );

    assert!(
        !has_negative_reduction(&stdout),
        "growth printed as a negative reduction:\n{stdout}"
    );
}

#[test]
fn reduction_is_still_reported_as_reduction() {
    // Control: state_machine.wasm shrinks on re-encoding (1655 -> 1558 bytes
    // with loom 1.5.4), so the size line must say reduction, without a sign.
    let stdout = stats_stdout(&fixture("../tests/corpus/state_machine.wasm"), "shrinks");
    let bytes = line(&stdout, "Binary size:");
    assert!(bytes.ends_with("% reduction)"), "{bytes}");
    assert!(!bytes.contains("(-") && !bytes.contains("(+"), "{bytes}");
    assert!(!has_negative_reduction(&stdout), "{stdout}");
}

#[test]
fn the_negative_reduction_detector_fires() {
    // The exact lines #554 measured on loom 1.5.4.
    assert!(has_negative_reduction(
        "Instructions: 154043 → 181652 (-17.9% reduction)"
    ));
    assert!(has_negative_reduction(
        "  Optimization effect: -650461 bytes reduction"
    ));
    assert!(!has_negative_reduction(
        "Binary size:  1655 → 1558 bytes (5.9% reduction)"
    ));
    assert!(!has_negative_reduction(
        "Re-encoding effect: 1558 bytes (-97)"
    ));
}
