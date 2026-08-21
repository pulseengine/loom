//! `loom optimize --stats` must report what verification actually
//! ESTABLISHED, not only what it caught (#331 Gap 1).
//!
//! These tests drive the real binary over real modules, because the property
//! at stake is what reaches the operator — not what the counters hold.

use std::path::Path;
use std::process::Command;

/// Two things were wrong before this.
///
/// First, nothing surfaced a **denominator**. `--stats` printed revert counts,
/// which answer "how often did verification catch something?" but cannot
/// answer "how much was verified at all?" — and the outcome that matters most
/// is invisible to a revert count by construction: a transform KEPT without a
/// proof does not revert, so it never appears. A `VerificationCoverage`
/// tracker with the full taxonomy had existed in `verify.rs` since before this
/// change, documented with the exact summary line it was meant to produce, and
/// had **zero non-test callers**.
///
/// Second, and worse, the solver size-threshold bypass recorded itself with
/// `record_revert("<pass>/z3-size-skipped")` and then returned acceptance. So
/// `--stats` printed "N function(s) reverted" for functions that **shipped**.
/// A stat naming the wrong outcome is worse than no stat: it reads as evidence
/// verification did its job, in precisely the case where it did not run.
///
/// This drives the real binary over a real 724-function module, so it pins the
/// user-visible reporting surface rather than the arithmetic behind it.
#[test]
fn coverage_is_reported_with_a_denominator_and_size_skips_are_not_reverts() {
    let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../loom-core/tests/fixtures/issue254-records-fused.wasm");
    assert!(fixture.exists(), "missing fixture: {}", fixture.display());

    let out_dir = std::env::temp_dir().join(format!("loom-331-{}", std::process::id()));
    std::fs::create_dir_all(&out_dir).expect("create scratch dir");
    let out_wasm = out_dir.join("optimized.wasm");

    let out = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args([
            "optimize",
            fixture.to_str().unwrap(),
            "-o",
            out_wasm.to_str().unwrap(),
            "--stats",
        ])
        .output()
        .expect("failed to run the loom binary");
    assert!(
        out.status.success(),
        "optimizing the fixture failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains("Verification Coverage"),
        "`--stats` printed no verification coverage at all, so a user still \
         cannot tell how much of their module was actually proven. Output:\n{stdout}"
    );

    // The denominator. "Proven: N/M attempts" — a bare count of proofs with
    // nothing to divide it by is the gap this closes.
    let proven_line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("Proven:"))
        .unwrap_or_else(|| panic!("no `Proven:` line in --stats output:\n{stdout}"));
    assert!(
        proven_line.contains('/') && proven_line.contains("attempts"),
        "the proven count must carry its denominator, got: {proven_line:?}"
    );

    // The mislabel: functions the size threshold bypassed were KEPT, not
    // reverted, and must no longer be counted among reverts.
    assert!(
        !stdout.contains("z3-size-skipped"),
        "the solver size-threshold bypass is still reported in the revert \
         block. Those transforms shipped; calling them reverts tells the \
         operator verification rejected code that in fact went out \
         unverified. Output:\n{stdout}"
    );

    // Reachability (#289). This fixture is large enough to cross the solver
    // size threshold, so the kept-without-proof bucket must actually be
    // exercised — otherwise every assertion above would still pass for an
    // implementation that never records the bucket, which is exactly the
    // failure mode that left the original tracker unread for so long.
    assert!(
        stdout.contains("Kept WITHOUT proof"),
        "no kept-without-proof attempts were reported. This module crosses \
         the solver size threshold, so that bucket must be reachable; its \
         absence means the reason is being dropped rather than surfaced. \
         Output:\n{stdout}"
    );
    assert!(
        stdout.contains("body over the solver size threshold"),
        "the kept-without-proof attempts must be attributed BY REASON — an \
         unattributed count cannot be acted on, because the operator cannot \
         tell whether to widen the encoder or raise a threshold. Output:\n{stdout}"
    );

    // And an incomplete run has to say so in words, not only in numbers a
    // reader has to subtract for themselves.
    assert!(
        stdout.contains("Not every transform in this output carries a proof"),
        "a run that kept transforms without proof must say so plainly. \
         Output:\n{stdout}"
    );
}

/// The positive control for the warning above: a module that verifies
/// completely must NOT be told it has unproven transforms. Without this, the
/// previous test is satisfied by a build that prints the warning
/// unconditionally — a caveat that is always shown carries no information and
/// trains the reader to skip it.
#[test]
fn a_fully_proven_run_reports_no_unproven_transforms() {
    let dir = std::env::temp_dir().join(format!("loom-331-clean-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let input = dir.join("add.wasm");
    // (module (func (export "add") (param i32 i32) (result i32)
    //   local.get 0 local.get 1 i32.add))
    const ADD_WASM: &[u8] = &[
        0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x07, 0x01, 0x60, 0x02, 0x7f, 0x7f,
        0x01, 0x7f, 0x03, 0x02, 0x01, 0x00, 0x07, 0x07, 0x01, 0x03, 0x61, 0x64, 0x64, 0x00, 0x00,
        0x0a, 0x09, 0x01, 0x07, 0x00, 0x20, 0x00, 0x20, 0x01, 0x6a, 0x0b,
    ];
    std::fs::write(&input, ADD_WASM).expect("write the input module");
    let output = dir.join("add.opt.wasm");

    let out = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args([
            "optimize",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--stats",
        ])
        .output()
        .expect("failed to run the loom binary");
    assert!(
        out.status.success(),
        "optimizing a trivial module failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    let stdout = String::from_utf8_lossy(&out.stdout);

    assert!(
        stdout.contains("Verification Coverage"),
        "coverage must be reported for a clean run too, not only when \
         something went unproven. Output:\n{stdout}"
    );
    assert!(
        !stdout.contains("Not every transform in this output carries a proof"),
        "a fully proven run must not carry the unproven-transform warning; a \
         caveat shown unconditionally carries no information. Output:\n{stdout}"
    );
    assert!(
        !stdout.contains("Kept WITHOUT proof"),
        "a fully proven run must not report kept-without-proof attempts. \
         Output:\n{stdout}"
    );
}
