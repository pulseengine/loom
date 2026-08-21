//! `loom verify` must never report success (#332).
//!
//! The subcommand shipped in v1.2.0 printed `✓ LOOM Verification`, never
//! opened its argument, and **exited 0** — for a wasm module passed where an
//! ISLE file was expected, for a file full of garbage, for an empty file, and
//! for a path that did not exist. Anything wiring it into CI would have got a
//! check that was green forever over nothing.
//!
//! The property these tests pin is therefore not "the errors are nice". It is
//! that **no input produces exit 0**, including the input that is completely
//! well-formed — because a well-formed ISLE file still gets verified by
//! nothing. Every negative control from the report is here as its own case, so
//! a future change that re-introduces a success path for any one of them fails
//! a named test rather than quietly restoring the vacuous gate.
//!
//! `CARGO_BIN_EXE_loom` is set by cargo for integration tests, so this runs the
//! real binary without pulling in a test-harness dependency.

use std::path::Path;
use std::process::{Command, Output};

fn loom_verify(arg: &str) -> Output {
    Command::new(env!("CARGO_BIN_EXE_loom"))
        .args(["verify", arg])
        .output()
        .expect("failed to run the loom binary")
}

/// A scratch file with the given name and contents, inside cargo's own
/// per-run temp area so parallel test binaries cannot collide.
fn scratch(name: &str, contents: &[u8]) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("loom-332-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    let path = dir.join(name);
    std::fs::write(&path, contents).expect("write scratch file");
    path
}

fn assert_no_success(out: &Output, what: &str) {
    assert_ne!(
        out.status.code(),
        Some(0),
        "`loom verify` exited 0 for {what} — nothing was verified, so success \
         must not be reportable (#332). stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        out.stdout.is_empty(),
        "`loom verify` wrote to stdout for {what}; the placeholder must keep \
         stdout empty so a caller redirecting it gets nothing that reads as a \
         result. stdout: {:?}",
        String::from_utf8_lossy(&out.stdout)
    );
}

#[test]
fn a_nonexistent_path_is_an_error() {
    let out = loom_verify("/nonexistent/definitely/not/here.isle");
    assert_no_success(&out, "a path that does not exist");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("no such file"),
        "expected a missing-file error, got: {err}"
    );
}

#[test]
fn a_wasm_module_where_an_isle_file_was_expected_is_an_error() {
    // The exact case from the report: a real, existing, perfectly valid file
    // of entirely the wrong kind.
    let path = scratch("add.wasm", b"\0asm\x01\0\0\0");
    let out = loom_verify(path.to_str().unwrap());
    assert_no_success(&out, "a .wasm passed where an ISLE file was expected");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("expected an ISLE rule file"),
        "expected a wrong-file-type error, got: {err}"
    );
}

#[test]
fn a_directory_is_an_error() {
    let dir = std::env::temp_dir().join(format!("loom-332-dir-{}.isle", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create dir");
    let out = loom_verify(dir.to_str().unwrap());
    assert_no_success(&out, "a directory");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("not a regular file"),
        "expected a not-a-file error, got: {err}"
    );
    let _ = std::fs::remove_dir(&dir);
}

/// The load-bearing case. The other tests could all be satisfied by a
/// validator that rejects bad input and then reports success for good input —
/// which is the vacuous gate wearing a coat. A *valid* ISLE file must also
/// fail, because it is verified by nothing.
#[test]
fn a_well_formed_isle_file_still_does_not_report_success() {
    let path = scratch("rules.isle", b"(rule (simplify (iadd x (iconst 0))) x)\n");
    let out = loom_verify(path.to_str().unwrap());
    assert_no_success(&out, "a well-formed .isle file");
    let err = String::from_utf8_lossy(&out.stderr);
    assert!(
        err.contains("nothing was verified"),
        "a validated file must still say plainly that nothing was verified; \
         got: {err}"
    );
    assert!(
        !err.contains('✓'),
        "the checkmark must not appear while there is nothing to check; got: {err}"
    );
}

/// Garbage and empty files were two of the four negative controls in the
/// report, and neither is distinguishable from the valid case today — which is
/// itself the point: the exit code is the same because the amount verified is
/// the same, namely none.
#[test]
fn garbage_and_empty_isle_files_do_not_report_success() {
    for (name, body) in [
        ("junk.isle", &b"this is not ISLE at all {{{"[..]),
        ("empty.isle", &b""[..]),
    ] {
        let path = scratch(name, body);
        let out = loom_verify(path.to_str().unwrap());
        assert_no_success(&out, name);
    }
}

/// `verify`'s exit code must stay distinct from the code an ordinary
/// optimization failure produces, so a caller can tell "this build cannot
/// verify anything" from "verification ran and rejected something" once Phase
/// 5 makes the second outcome possible.
#[test]
fn the_unimplemented_exit_code_is_distinct_from_a_plain_failure() {
    let path = scratch("codes.isle", b"(rule)\n");
    let verify = loom_verify(path.to_str().unwrap());
    assert_eq!(
        verify.status.code(),
        Some(2),
        "expected the dedicated unimplemented code 2"
    );

    // A real optimization failure, for contrast: a file that is not wasm.
    let not_wasm = scratch("not-a-module.wasm", b"definitely not wasm");
    let optimize = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args(["optimize", not_wasm.to_str().unwrap()])
        .output()
        .expect("failed to run the loom binary");
    assert_ne!(
        optimize.status.code(),
        Some(0),
        "optimizing a non-wasm file must fail"
    );
    assert_ne!(
        optimize.status.code(),
        verify.status.code(),
        "the unimplemented-verify code must not collide with the code an \
         ordinary failure produces, or the distinction it exists to draw is \
         unobservable"
    );
}

/// The docs must not advertise a form of the command that does not exist. The
/// two references in `docs/analysis/synth-architecture.md` used a two-wasm
/// form the CLI has never accepted; fixing the exit code while a doc still
/// recommends the command would leave half the defect in place.
#[test]
fn no_doc_advertises_the_two_argument_wasm_form() {
    let doc = Path::new(env!("CARGO_MANIFEST_DIR")).join("../docs/analysis/synth-architecture.md");
    let text = std::fs::read_to_string(&doc).expect("read the architecture doc");
    for line in text.lines() {
        let trimmed = line.trim_start();
        // Only command lines, not the prose or the notes that explain why the
        // command is unavailable.
        if trimmed.starts_with("loom verify") {
            panic!(
                "{} still advertises `loom verify` as a runnable command: {:?}",
                doc.display(),
                line
            );
        }
    }
}
