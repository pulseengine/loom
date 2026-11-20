//! Dogfooding (REQ-8): LOOM optimizes its own WebAssembly build, and the
//! optimized LOOM produces the same results as the original.
//!
//! 1. The native `loom` binary optimizes `loom.wasm` (loom-cli built for
//!    wasm32-wasip2) into `loom.opt.wasm`.
//! 2. Both are run under wasmtime on every input file in the repository's
//!    test trees (`tests/`, `loom-core/tests/`, `fuzz/seeds/`), each as
//!    `loom optimize <input> -o <output> --attestation false`.
//! 3. For each input, the exit status, whether an output file was written,
//!    and the output bytes must be identical between the two runs.
//!
//! The test fails on any difference, on 0 inputs, and on an input set where
//! the original LOOM succeeded on none (two runs that both fail everything
//! would agree and prove nothing). It prints the counts.
//!
//! It needs a wasm32-wasip2 build and a wasmtime binary, so it is
//! `#[ignore]`d in a plain `cargo test` and run explicitly by the rivet
//! artifact TEST-SELF-OPTIMIZATION:
//!
//! ```text
//! LOOM_DOGFOOD_WASM=<loom.wasm> LOOM_DOGFOOD_WASMTIME=<wasmtime> \
//!   cargo test --release -p loom-cli --test dogfood -- --ignored
//! ```
//!
//! Missing variables are a failure, not a skip.
//!
//! Control: with `LOOM_DOGFOOD_CONTROL_FLIP=1`, one byte of every output
//! written by the optimized LOOM is flipped before the comparison, and the
//! test must fail.

use std::path::{Path, PathBuf};
use std::process::Command;

fn required_env(name: &str) -> PathBuf {
    let value = std::env::var_os(name).unwrap_or_else(|| {
        panic!("{name} is not set; the dogfood test does not skip (see the module docs)")
    });
    let path = PathBuf::from(value);
    assert!(path.is_file(), "{name}={} is not a file", path.display());
    path
}

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("loom-cli has a parent directory")
        .to_path_buf()
}

/// Every `.wasm` and `.wat` file under the given directories, as paths
/// relative to the repository root, sorted.
fn inputs(root: &Path, dirs: &[&str]) -> Vec<PathBuf> {
    fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                walk(&path, out);
            } else if matches!(
                path.extension().and_then(|e| e.to_str()),
                Some("wasm") | Some("wat")
            ) {
                out.push(path);
            }
        }
    }
    let mut found = Vec::new();
    for dir in dirs {
        walk(&root.join(dir), &mut found);
    }
    let mut rel: Vec<PathBuf> = found
        .into_iter()
        .map(|p| p.strip_prefix(root).expect("under root").to_path_buf())
        .collect();
    rel.sort();
    rel
}

#[derive(Debug, PartialEq, Eq)]
struct RunResult {
    status: Option<i32>,
    output: Option<Vec<u8>>,
}

fn run_under_wasmtime(
    wasmtime: &Path,
    component: &Path,
    root: &Path,
    input: &Path,
    out_dir: &Path,
) -> RunResult {
    let out_file = out_dir.join("out.wasm");
    let _ = std::fs::remove_file(&out_file);
    let guest_input = format!("/in/{}", input.display());
    let status = Command::new(wasmtime)
        .arg("run")
        .arg("--dir")
        .arg(format!("{}::/in", root.display()))
        .arg("--dir")
        .arg(format!("{}::/out", out_dir.display()))
        .arg(component)
        .args(["optimize", &guest_input, "-o", "/out/out.wasm"])
        .args(["--attestation", "false"])
        .output()
        .unwrap_or_else(|e| panic!("cannot run {}: {e}", wasmtime.display()))
        .status;
    RunResult {
        status: status.code(),
        output: std::fs::read(&out_file).ok(),
    }
}

/// Byte length of each core module directly inside a component, in order.
/// The order is the one the component optimizer numbers modules by.
fn top_level_core_module_sizes(component: &[u8]) -> Vec<u64> {
    use wasmparser::{Parser, Payload};
    let mut sizes = Vec::new();
    let mut depth = 0usize;
    for payload in Parser::new(0).parse_all(component) {
        match payload.expect("loom.wasm parses as a component") {
            Payload::ModuleSection {
                unchecked_range, ..
            } => {
                if depth == 0 {
                    sizes.push(unchecked_range.end - unchecked_range.start);
                }
                depth += 1;
            }
            Payload::ComponentSection { .. } => depth += 1,
            Payload::End(_) => depth = depth.saturating_sub(1),
            _ => {}
        }
    }
    sizes
}

/// Indices of the core modules the component optimizer reports as
/// "Module N: Optimized successfully".
fn optimized_module_indices(report: &str) -> Vec<usize> {
    report
        .lines()
        .filter_map(|line| {
            let rest = line.split("Module ").nth(1)?;
            let (idx, tail) = rest.split_once(':')?;
            tail.trim_start()
                .starts_with("Optimized successfully")
                .then(|| idx.trim().parse().ok())
                .flatten()
        })
        .collect()
}

struct SelfOptimization {
    work: PathBuf,
    original: PathBuf,
    optimized: PathBuf,
    module_sizes: Vec<u64>,
    optimized_modules: Vec<usize>,
}

impl SelfOptimization {
    /// Index of the largest core module: LOOM's own code.
    fn main_module(&self) -> usize {
        (0..self.module_sizes.len())
            .max_by_key(|&i| self.module_sizes[i])
            .expect("loom.wasm has at least one core module")
    }
}

/// The native LOOM optimizes `LOOM_DOGFOOD_WASM`; prints what it changed.
fn self_optimize(tag: &str) -> SelfOptimization {
    let original = required_env("LOOM_DOGFOOD_WASM");
    let work = std::env::temp_dir().join(format!("loom-dogfood-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&work);
    std::fs::create_dir_all(&work).expect("temp dir");
    let optimized = work.join("loom.opt.wasm");
    let opt = Command::new(env!("CARGO_BIN_EXE_loom"))
        .arg("optimize")
        .arg(&original)
        .arg("-o")
        .arg(&optimized)
        .args(["--attestation", "false"])
        .output()
        .expect("run native loom");
    let stderr = String::from_utf8_lossy(&opt.stderr).into_owned();
    assert!(
        opt.status.success(),
        "native loom failed to optimize {}:\n{stderr}",
        original.display()
    );
    let original_bytes = std::fs::read(&original).expect("read loom.wasm");
    let optimized_bytes = std::fs::read(&optimized).expect("read loom.opt.wasm");
    let run = SelfOptimization {
        module_sizes: top_level_core_module_sizes(&original_bytes),
        optimized_modules: optimized_module_indices(&stderr),
        work,
        original,
        optimized,
    };
    let main = run.main_module();
    println!(
        "loom.wasm {} bytes -> loom.opt.wasm {} bytes",
        original_bytes.len(),
        optimized_bytes.len()
    );
    println!(
        "{} of {} core modules optimized {:?}; main module {} ({} bytes) {}",
        run.optimized_modules.len(),
        run.module_sizes.len(),
        run.optimized_modules,
        main,
        run.module_sizes[main],
        if run.optimized_modules.contains(&main) {
            "optimized"
        } else {
            "NOT optimized"
        }
    );
    run
}

#[test]
#[ignore = "needs LOOM_DOGFOOD_WASM and LOOM_DOGFOOD_WASMTIME; run by TEST-SELF-OPTIMIZATION"]
fn optimized_loom_produces_the_same_results_as_loom() {
    let wasmtime = required_env("LOOM_DOGFOOD_WASMTIME");
    let flip = std::env::var("LOOM_DOGFOOD_CONTROL_FLIP").as_deref() == Ok("1");
    let SelfOptimization {
        work,
        original,
        optimized,
        ..
    } = self_optimize("compare");

    let root = repo_root();
    let files = inputs(&root, &["tests", "loom-core/tests", "fuzz/seeds"]);
    assert!(!files.is_empty(), "0 inputs found under {}", root.display());

    let out_a = work.join("a");
    let out_b = work.join("b");
    std::fs::create_dir_all(&out_a).unwrap();
    std::fs::create_dir_all(&out_b).unwrap();

    let mut differences = Vec::new();
    let mut original_succeeded = 0usize;
    for input in &files {
        let a = run_under_wasmtime(&wasmtime, &original, &root, input, &out_a);
        let mut b = run_under_wasmtime(&wasmtime, &optimized, &root, input, &out_b);
        if flip {
            if let Some(bytes) = b.output.as_mut() {
                if let Some(first) = bytes.first_mut() {
                    *first ^= 0xff;
                }
            }
        }
        if a.status == Some(0) && a.output.is_some() {
            original_succeeded += 1;
        }
        if a != b {
            differences.push(format!(
                "{}: original status {:?}, output {:?} bytes; optimized status {:?}, output {:?} bytes",
                input.display(),
                a.status,
                a.output.as_ref().map(Vec::len),
                b.status,
                b.output.as_ref().map(Vec::len)
            ));
        }
    }

    println!(
        "dogfood: {} inputs, {} optimized successfully by the original LOOM, {} differences",
        files.len(),
        original_succeeded,
        differences.len()
    );
    assert!(
        original_succeeded > 0,
        "the original LOOM succeeded on none of {} inputs; agreement would prove nothing",
        files.len()
    );
    let _ = std::fs::remove_dir_all(&work);
    assert!(
        differences.is_empty(),
        "the optimized LOOM differs from the original on {} of {} inputs:\n{}",
        differences.len(),
        files.len(),
        differences.join("\n")
    );
}

/// TEST-SELF-OPTIMIZATION-OWN-CODE: LOOM optimizes the core module that holds
/// its own code, not only the small ones around it. Fails while that module
/// is kept as it was (today under #196), so it cannot pass at 1 of 3.
#[test]
#[ignore = "needs LOOM_DOGFOOD_WASM; run by TEST-SELF-OPTIMIZATION-OWN-CODE"]
fn loom_optimizes_its_own_code_module() {
    let run = self_optimize("own-code");
    let main = run.main_module();
    let _ = std::fs::remove_dir_all(&run.work);
    assert!(
        run.optimized_modules.contains(&main),
        "the main core module ({main}, {} bytes) was not optimized; {} of {} core modules were ({:?})",
        run.module_sizes[main],
        run.optimized_modules.len(),
        run.module_sizes.len(),
        run.optimized_modules
    );
}

#[test]
fn module_report_parsing() {
    let report = "\
⚠  Module 0: Skipped (function-referencing element table, pending a behavioral gate (#196)); original bytes kept
✓ Module 1: Optimized successfully
⚠  Module 2: Skipped (x); original bytes kept
";
    assert_eq!(optimized_module_indices(report), vec![1]);
    assert_eq!(
        optimized_module_indices("✓ Module 0: Optimized successfully\n"),
        vec![0]
    );
    assert!(optimized_module_indices("").is_empty());
}
