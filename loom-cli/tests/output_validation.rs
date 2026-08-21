//! `loom optimize` must never write a module that does not validate (#346).
//!
//! Two defects met here. The encoder dropped the **data count section** while
//! keeping the `memory.init` / `data.drop` instructions that require it — the
//! spec makes that section mandatory precisely so those instructions can be
//! validated without scanning the data section, so the output was structurally
//! invalid rather than merely suboptimal. And nothing checked: the CLI wrote
//! whatever the encoder produced and printed `✅ Optimization complete!` over
//! it, exit 0.
//!
//! Silent invalid output is the worst shape this can take. The next tool in
//! the chain reports the failure against *its own* input, so the blame lands
//! downstream of the tool that actually broke the module.
//!
//! `loom_core::optimize::optimize_module` has carried an output-validation
//! backstop since #257 — described there as the systemic guarantee that loom
//! can NEVER emit structurally invalid wasm. The CLI does not go through that
//! function (#345), so it inherited none of it.
//!
//! These tests drive the real binary, because the property is about what lands
//! on disk.

use std::process::Command;

/// The #346 reproduction, assembled from:
///
/// ```wat
/// (module
///   (memory 1)
///   (data $d "hello")
///   (func (export "init") (param i32)
///     local.get 0  i32.const 0  i32.const 5
///     memory.init $d
///     data.drop $d))
/// ```
///
/// 79 bytes, valid on input, and the smallest thing that exercises both
/// instructions that make the data count section mandatory.
const MEMORY_INIT_WASM: &[u8] = &[
    0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, 0x01, 0x05, 0x01, 0x60, 0x01, 0x7f, 0x00, 0x03,
    0x02, 0x01, 0x00, 0x05, 0x03, 0x01, 0x00, 0x01, 0x07, 0x08, 0x01, 0x04, 0x69, 0x6e, 0x69, 0x74,
    0x00, 0x00, 0x0c, 0x01, 0x01, 0x0a, 0x11, 0x01, 0x0f, 0x00, 0x20, 0x00, 0x41, 0x00, 0x41, 0x05,
    0xfc, 0x08, 0x00, 0x00, 0xfc, 0x09, 0x00, 0x0b, 0x0b, 0x08, 0x01, 0x01, 0x05, 0x68, 0x65, 0x6c,
    0x6c, 0x6f, 0x00, 0x0b, 0x04, 0x6e, 0x61, 0x6d, 0x65, 0x09, 0x04, 0x01, 0x00, 0x01, 0x64,
];

fn scratch_dir(tag: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("loom-346-{tag}-{}", std::process::id()));
    std::fs::create_dir_all(&dir).expect("create scratch dir");
    dir
}

/// Section id 12 is the data count section. Walking the section headers is
/// enough — every section after the 8-byte preamble is `[id][uleb size]`.
fn has_data_count_section(bytes: &[u8]) -> bool {
    let mut i = 8; // magic + version
    while i < bytes.len() {
        let id = bytes[i];
        i += 1;
        // uleb128 section size
        let mut size: usize = 0;
        let mut shift = 0;
        loop {
            if i >= bytes.len() {
                return false;
            }
            let b = bytes[i];
            i += 1;
            size |= ((b & 0x7f) as usize) << shift;
            if b & 0x80 == 0 {
                break;
            }
            shift += 7;
        }
        if id == 12 {
            return true;
        }
        i += size;
    }
    false
}

/// The end-to-end property: a module using `memory.init` must come out valid.
///
/// Before the fix this wrote a module rejected with "data count section
/// required" while exiting 0.
#[test]
fn a_module_using_memory_init_is_emitted_valid() {
    let dir = scratch_dir("init");
    let input = dir.join("in.wasm");
    let output = dir.join("out.wasm");
    std::fs::write(&input, MEMORY_INIT_WASM).expect("write input");

    // The input itself must be valid, or the test proves nothing about output.
    assert!(
        loom_core::encode::validate_output_bytes(MEMORY_INIT_WASM).is_ok(),
        "the fixture is not valid wasm; the test cannot attribute an invalid \
         OUTPUT to the optimizer"
    );

    let out = Command::new(env!("CARGO_BIN_EXE_loom"))
        .args([
            "optimize",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ])
        .output()
        .expect("failed to run the loom binary");

    assert!(
        out.status.success(),
        "optimizing a valid memory.init module failed:\nstdout: {}\nstderr: {}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );

    let emitted = std::fs::read(&output).expect("read the emitted module");
    if let Err(e) = loom_core::encode::validate_output_bytes(&emitted) {
        panic!(
            "loom emitted a module that does not validate: {e}\n\
             (this is #346: the data count section is mandatory whenever \
             memory.init or data.drop appear)"
        );
    }

    // And specifically the section that was missing. Validation alone would
    // also pass if some future change removed the instructions entirely, so
    // assert the section is there rather than only that nothing complained.
    assert!(
        has_data_count_section(&emitted),
        "the emitted module validates but carries no data count section — \
         check whether the memory.init instructions survived at all"
    );
}

/// The systemic half, and the one that matters beyond this instance: the CLI
/// must not write output it has not validated.
///
/// Rather than asserting on the encoder (already covered above), this asserts
/// the property that makes the NEXT encoder bug loud instead of silent — that
/// success on stdout implies the artifact on disk validates. Driven over every
/// wasm fixture in the repo that the binary accepts.
#[test]
fn every_successful_optimize_leaves_a_valid_module_on_disk() {
    let fixtures =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../loom-core/tests/fixtures");
    let dir = scratch_dir("sweep");

    let mut checked = 0usize;
    let entries = match std::fs::read_dir(&fixtures) {
        Ok(e) => e,
        Err(_) => return, // fixtures directory absent in some checkouts
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|e| e.to_str()) != Some("wasm") {
            continue;
        }
        // Only modules that were valid to begin with can hold loom responsible
        // for an invalid result.
        let input_bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(_) => continue,
        };
        if loom_core::encode::validate_output_bytes(&input_bytes).is_err() {
            continue;
        }

        let output = dir.join(format!(
            "{}.opt.wasm",
            path.file_stem().unwrap().to_string_lossy()
        ));
        let out = Command::new(env!("CARGO_BIN_EXE_loom"))
            .args([
                "optimize",
                path.to_str().unwrap(),
                "-o",
                output.to_str().unwrap(),
                // Keep the sweep quick: the full pipeline is exercised by the
                // test above, and #347 makes `inline` unusably slow on some
                // real modules.
                "--passes",
                "dce,vacuum,constant-folding",
            ])
            .output()
            .expect("failed to run the loom binary");

        if !out.status.success() {
            // A refusal is the CORRECT behaviour now — it means the gate
            // fired. What must never happen is success over invalid output.
            continue;
        }
        checked += 1;
        let emitted = std::fs::read(&output).expect("read the emitted module");
        if let Err(e) = loom_core::encode::validate_output_bytes(&emitted) {
            panic!(
                "loom reported SUCCESS but wrote an invalid module for {}: {e}\n\
                 Exit 0 must imply the artifact validates (#346).",
                path.display()
            );
        }
    }

    assert!(
        checked > 0,
        "the sweep validated no modules at all, so it asserts nothing — check \
         that the fixtures directory is present and that the binary accepts them"
    );
}
