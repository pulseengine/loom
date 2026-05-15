//! Criterion-driven corpus baseline harness.
//!
//! This is the cargo-bench equivalent of `scripts/measure_corpus.sh`. It runs
//! the same comparison matrix per fixture:
//!
//!   baseline -> LOOM         (file & code-section bytes)
//!   baseline -> wasm-opt -O3 (if wasm-opt installed)
//!   wasm-opt -> LOOM         (the "wasm-opt then LOOM" pipeline)
//!   meld fuse -> wasm-opt    (component-only)
//!   meld fuse -> LOOM        (component-only)
//!
//! and emits both:
//!
//!   - a markdown table to stdout (so `cargo bench` output is grep-able), AND
//!   - a versioned report file at
//!         docs/measurements/v<workspace-version>-corpus-baseline.md
//!     that mirrors the bash harness's report format.
//!
//! The criterion measurement layer wraps each fixture's LOOM run in a
//! `bench_function`, so timings land in `target/criterion/`. The custom
//! byte-count and percent-delta metrics are emitted in the markdown table
//! (criterion 0.5 doesn't natively support custom metrics).
//!
//! Pre-flight: a wasm-opt version pin is checked at startup
//! (scripts/wasm-opt.pinned). A mismatch is non-fatal but printed prominently.
//!
//! Concrete entry points:
//!   cargo bench -p loom-testing --bench corpus_baseline
//!   cargo bench -p loom-testing --bench corpus_baseline -- --test   (smoke)
//!
//! Tooling resolution:
//!   - LOOM:       $LOOM env, else <repo>/target/release/loom
//!   - wasm-opt:   $WASM_OPT env, else wasm-opt on $PATH (optional)
//!   - wasm-tools: $WASM_TOOLS env, else wasm-tools on $PATH (mandatory)
//!   - meld:       $MELD env, else meld on $PATH (optional, components only)
//!
//! Per-tool invocations time out after $PER_RUN_TIMEOUT seconds (default 300).

use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::sync::OnceLock;
use std::time::Duration;

use criterion::{Criterion, criterion_group, criterion_main};

// ---------------------------------------------------------------------------
// Workload catalogue. Kept in lock-step with scripts/measure_corpus.sh.
// Format mirrors the shell script: (display_name, repo-relative path, note).
// ---------------------------------------------------------------------------
const WORKLOADS: &[(&str, &str, &str)] = &[
    ("gale",             "scripts/mythos/gale_measure/gale_in_baseline.wasm",      "kernel-FFI fixture"),
    ("httparse",         "tests/corpus/httparse.wasm",                              "HTTP parser"),
    ("nom_numbers",      "tests/corpus/nom_numbers.wasm",                           "parser-combinator primitives"),
    ("state_machine",    "tests/corpus/state_machine.wasm",                         "FSM kernel"),
    ("json_lite",        "tests/corpus/json_lite.wasm",                             "minimal JSON tokenizer"),
    ("loom",             "tests/corpus/loom.wasm",                                  "LOOM self-build (dogfood target)"),
    ("calculator",       "tests/calculator.wasm",                                   "component-shaped fixture"),
    ("calculator_root",  "calculator.wasm",                                         "2.3 MB component (root, large)"),
    ("simple_component", "loom-core/tests/component_fixtures/simple.component.wasm","tiny component (adapter-heavy)"),
    ("calc_component",   "loom-core/tests/component_fixtures/calc.component.wasm",  "small component (adapter-heavy)"),
];

// ---------------------------------------------------------------------------
// Measurement record. One per fixture. `n/a` is represented as `None`.
// ---------------------------------------------------------------------------
#[derive(Debug, Clone)]
struct Row {
    name: String,
    note: String,
    #[allow(dead_code)]
    missing: bool,
    base_bytes: Option<u64>,
    loom_bytes: Option<u64>,
    wopt_bytes: Option<u64>,
    wopt_loom_bytes: Option<u64>,
    base_code: Option<u64>,
    loom_code: Option<u64>,
    wopt_code: Option<u64>,
    // Component-meld-baseline columns (only set for component fixtures when
    // `meld` is available).
    meld_base: Option<u64>,
    meld_wopt: Option<u64>,
    meld_loom: Option<u64>,
}

impl Row {
    fn missing(name: &str, note: &str) -> Self {
        Self {
            name: name.into(),
            note: note.into(),
            missing: true,
            base_bytes: None,
            loom_bytes: None,
            wopt_bytes: None,
            wopt_loom_bytes: None,
            base_code: None,
            loom_code: None,
            wopt_code: None,
            meld_base: None,
            meld_wopt: None,
            meld_loom: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Resolved tool paths and environment.
// ---------------------------------------------------------------------------
struct BenchEnv {
    repo_root: PathBuf,
    tmp_dir: PathBuf,
    loom: PathBuf,
    wasm_tools: String,
    wasm_opt: Option<String>,
    meld: Option<String>,
    per_run_timeout_secs: u64,
    loom_version: String,
    wasm_opt_version: Option<String>,
    wasm_tools_version: String,
    pin_status: PinStatus,
}

#[derive(Debug, Clone)]
enum PinStatus {
    Match { version: String },
    Mismatch { pinned: String, installed: String },
    NoWasmOpt { pinned: String },
    PinUnreadable,
}

fn repo_root() -> PathBuf {
    // CARGO_MANIFEST_DIR is set at compile time by cargo for benches; this
    // points at loom-testing/. The repo root is its parent.
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("loom-testing must have a parent dir")
        .to_path_buf()
}

fn resolve_env() -> BenchEnv {
    let repo_root = repo_root();
    let tmp_dir = PathBuf::from(
        env::var("TMP_DIR").unwrap_or_else(|_| "/tmp/loom-measure-corpus".into()),
    );
    let _ = fs::create_dir_all(&tmp_dir);

    let loom = env::var("LOOM")
        .map(PathBuf::from)
        .unwrap_or_else(|_| repo_root.join("target/release/loom"));

    let wasm_tools = env::var("WASM_TOOLS").unwrap_or_else(|_| "wasm-tools".into());
    let wasm_opt_name = env::var("WASM_OPT").unwrap_or_else(|_| "wasm-opt".into());
    let meld_name = env::var("MELD").unwrap_or_else(|_| "meld".into());

    let wasm_opt = if tool_exists(&wasm_opt_name) { Some(wasm_opt_name) } else { None };
    let meld = if tool_exists(&meld_name) { Some(meld_name) } else { None };

    let per_run_timeout_secs = env::var("PER_RUN_TIMEOUT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(300u64);

    let loom_version = first_line(&Command::new(&loom).arg("--version").output().ok())
        .unwrap_or_else(|| "unknown".into());
    let wasm_tools_version = first_line(
        &Command::new(&wasm_tools).arg("--version").output().ok(),
    )
    .unwrap_or_else(|| "unknown".into());
    let wasm_opt_version = wasm_opt.as_ref().and_then(|name| {
        first_line(&Command::new(name).arg("--version").output().ok())
    });

    let pin_status = check_wasm_opt_pin(&repo_root, wasm_opt_version.as_deref());

    BenchEnv {
        repo_root,
        tmp_dir,
        loom,
        wasm_tools,
        wasm_opt,
        meld,
        per_run_timeout_secs,
        loom_version,
        wasm_opt_version,
        wasm_tools_version,
        pin_status,
    }
}

fn tool_exists(name: &str) -> bool {
    // `<tool> --version` is the most portable existence probe; falling back
    // to inspecting `$PATH` directly would miss tools installed via aliases.
    Command::new(name)
        .arg("--version")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

fn first_line(out: &Option<std::process::Output>) -> Option<String> {
    out.as_ref().and_then(|o| {
        if !o.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&o.stdout);
        s.lines().next().map(|l| l.trim().to_string())
    })
}

// ---------------------------------------------------------------------------
// wasm-opt pin file. Format (see scripts/wasm-opt.pinned): one line,
// `version_NNN`, with `#`-prefixed comments allowed.
// ---------------------------------------------------------------------------
fn read_pin(repo_root: &Path) -> Option<String> {
    let path = repo_root.join("scripts/wasm-opt.pinned");
    let text = fs::read_to_string(&path).ok()?;
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        return Some(trimmed.to_string());
    }
    None
}

fn parse_version_token(s: &str) -> Option<String> {
    // wasm-opt --version emits e.g. `wasm-opt version 116 (version_116)`.
    // Prefer the parenthesised form.
    if let Some(start) = s.find("(version_") {
        let after = &s[start + 1..];
        if let Some(end) = after.find(')') {
            return Some(after[..end].to_string());
        }
    }
    // Fall back to a bare `version_NNN` token.
    for word in s.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
        if let Some(rest) = word.strip_prefix("version_") {
            if !rest.is_empty() && rest.chars().all(|c| c.is_ascii_digit()) {
                return Some(word.to_string());
            }
        }
    }
    // Fall back to `version N` -> `version_N`.
    let lower = s.to_lowercase();
    if let Some(idx) = lower.find("version ") {
        let tail = &s[idx + "version ".len()..];
        let num: String = tail.chars().take_while(|c| c.is_ascii_digit()).collect();
        if !num.is_empty() {
            return Some(format!("version_{num}"));
        }
    }
    None
}

fn check_wasm_opt_pin(repo_root: &Path, wasm_opt_version_raw: Option<&str>) -> PinStatus {
    let Some(pinned) = read_pin(repo_root) else {
        return PinStatus::PinUnreadable;
    };
    let Some(raw) = wasm_opt_version_raw else {
        return PinStatus::NoWasmOpt { pinned };
    };
    match parse_version_token(raw) {
        Some(installed) if installed == pinned => PinStatus::Match { version: installed },
        Some(installed) => PinStatus::Mismatch { pinned, installed },
        None => PinStatus::Mismatch { pinned, installed: raw.to_string() },
    }
}

fn print_pin_banner(env: &BenchEnv) {
    match &env.pin_status {
        PinStatus::Match { version } => {
            eprintln!("[bench] wasm-opt pin OK: {version}");
        }
        PinStatus::Mismatch { pinned, installed } => {
            eprintln!(
                "[bench] WARNING: wasm-opt version mismatch (installed={installed}, pinned={pinned})"
            );
            eprintln!("[bench]   To match the pin: cargo install wasm-opt --locked --version <X>");
            eprintln!("[bench]   To bump the pin:  edit scripts/wasm-opt.pinned");
        }
        PinStatus::NoWasmOpt { pinned } => {
            eprintln!(
                "[bench] wasm-opt not installed; pin = {pinned}; wasm-opt columns will be n/a"
            );
        }
        PinStatus::PinUnreadable => {
            eprintln!("[bench] WARNING: scripts/wasm-opt.pinned missing or malformed");
        }
    }
}

// ---------------------------------------------------------------------------
// Tool runners. All wrap their child in a configurable timeout.
// ---------------------------------------------------------------------------
fn run_with_timeout(
    cmd: &mut Command,
    timeout: Duration,
) -> std::io::Result<std::process::ExitStatus> {
    cmd.stdout(Stdio::null()).stderr(Stdio::null());
    let mut child = cmd.spawn()?;
    let start = std::time::Instant::now();
    loop {
        match child.try_wait()? {
            Some(status) => return Ok(status),
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::TimedOut,
                        "timed out",
                    ));
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }
}

fn file_size(p: &Path) -> Option<u64> {
    fs::metadata(p).ok().map(|m| m.len())
}

fn is_component(path: &Path) -> bool {
    let Ok(mut f) = fs::File::open(path) else { return false };
    use std::io::Read;
    let mut buf = [0u8; 8];
    if f.read(&mut buf).ok() != Some(8) {
        return false;
    }
    // Core module:    \0 a s m  01 00 00 00
    // Component:      \0 a s m  0d 00 01 00
    &buf[0..4] == b"\0asm" && buf[4] == 0x0d
}

/// Run `wasm-tools objdump <path>` and sum every `code` section's byte count.
/// This matches the shell harness exactly.
fn code_section_bytes(env: &BenchEnv, path: &Path) -> Option<u64> {
    if !path.exists() {
        return None;
    }
    let out = Command::new(&env.wasm_tools)
        .arg("objdump")
        .arg(path)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let mut sum: u64 = 0;
    let mut any = false;
    for line in text.lines() {
        // objdump rows look like:
        //   code        | 0x3ae - 0x6d9 |  811 bytes | 28 count
        let cols: Vec<&str> = line.split('|').collect();
        if cols.len() < 3 {
            continue;
        }
        if cols[0].trim() != "code" {
            continue;
        }
        let third = cols[2].trim();
        let n_str: String = third
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        if let Ok(n) = n_str.parse::<u64>() {
            sum = sum.saturating_add(n);
            any = true;
        }
    }
    if any { Some(sum) } else { None }
}

fn validate_wasm(env: &BenchEnv, path: &Path) -> bool {
    Command::new(&env.wasm_tools)
        .arg("validate")
        .arg(path)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
}

/// Run `loom optimize <input> --attestation false -o <output>`.
fn run_loom(env: &BenchEnv, input: &Path, output: &Path) -> bool {
    if !env.loom.exists() {
        return false;
    }
    let mut cmd = Command::new(&env.loom);
    cmd.arg("optimize")
        .arg(input)
        .arg("--attestation")
        .arg("false")
        .arg("-o")
        .arg(output);
    run_with_timeout(&mut cmd, Duration::from_secs(env.per_run_timeout_secs))
        .map(|s| s.success())
        .unwrap_or(false)
        && validate_wasm(env, output)
}

/// Run `wasm-opt -O3 <input> -o <output>`. Returns false if wasm-opt is absent.
fn run_wasm_opt(env: &BenchEnv, input: &Path, output: &Path) -> bool {
    let Some(name) = env.wasm_opt.as_ref() else { return false };
    let mut cmd = Command::new(name);
    cmd.arg("-O3").arg(input).arg("-o").arg(output);
    run_with_timeout(&mut cmd, Duration::from_secs(env.per_run_timeout_secs))
        .map(|s| s.success())
        .unwrap_or(false)
        && validate_wasm(env, output)
}

/// Run `meld fuse <component> -o <core> --no-attestation`.
fn run_meld(env: &BenchEnv, input: &Path, output: &Path) -> bool {
    let Some(name) = env.meld.as_ref() else { return false };
    let mut cmd = Command::new(name);
    cmd.arg("fuse")
        .arg(input)
        .arg("-o")
        .arg(output)
        .arg("--no-attestation");
    run_with_timeout(&mut cmd, Duration::from_secs(env.per_run_timeout_secs))
        .map(|s| s.success())
        .unwrap_or(false)
        && validate_wasm(env, output)
}

// ---------------------------------------------------------------------------
// Per-fixture measurement. This is invoked once per fixture (criterion calls
// the closure several times for timing; we use a Mutex<Vec<Option<Row>>> to
// record results exactly once and short-circuit subsequent iterations).
// ---------------------------------------------------------------------------
fn measure_fixture(env: &BenchEnv, name: &str, rel_path: &str, note: &str) -> Row {
    let fixture = env.repo_root.join(rel_path);
    if !fixture.exists() {
        return Row::missing(name, note);
    }
    let Some(base_bytes) = file_size(&fixture) else { return Row::missing(name, note) };
    if base_bytes == 0 {
        return Row::missing(name, note);
    }
    let base_code = code_section_bytes(env, &fixture);

    let loom_out = env.tmp_dir.join(format!("{name}.loom.wasm"));
    let wopt_out = env.tmp_dir.join(format!("{name}.wopt.wasm"));
    let wopt_loom_out = env.tmp_dir.join(format!("{name}.wopt-loom.wasm"));

    // ---- LOOM pass ---------------------------------------------------------
    let (loom_bytes, loom_code) = if run_loom(env, &fixture, &loom_out) {
        (file_size(&loom_out), code_section_bytes(env, &loom_out))
    } else {
        (None, None)
    };

    // ---- wasm-opt pass -----------------------------------------------------
    let (wopt_bytes, wopt_code) = if run_wasm_opt(env, &fixture, &wopt_out) {
        (file_size(&wopt_out), code_section_bytes(env, &wopt_out))
    } else {
        (None, None)
    };

    // ---- wasm-opt -> LOOM --------------------------------------------------
    let wopt_loom_bytes = if wopt_bytes.is_some() && run_loom(env, &wopt_out, &wopt_loom_out) {
        file_size(&wopt_loom_out)
    } else {
        None
    };

    // ---- Component-only: meld -> wasm-opt and meld -> LOOM -----------------
    let mut meld_base = None;
    let mut meld_wopt = None;
    let mut meld_loom = None;
    if is_component(&fixture) && env.meld.is_some() {
        let meld_out = env.tmp_dir.join(format!("{name}.melded.wasm"));
        if run_meld(env, &fixture, &meld_out) {
            meld_base = file_size(&meld_out);
            if env.wasm_opt.is_some() {
                let mwo = env.tmp_dir.join(format!("{name}.melded.wopt.wasm"));
                if run_wasm_opt(env, &meld_out, &mwo) {
                    meld_wopt = file_size(&mwo);
                }
            }
            let mlo = env.tmp_dir.join(format!("{name}.melded.loom.wasm"));
            if run_loom(env, &meld_out, &mlo) {
                meld_loom = file_size(&mlo);
            }
        }
    }

    Row {
        name: name.into(),
        note: note.into(),
        missing: false,
        base_bytes: Some(base_bytes),
        loom_bytes,
        wopt_bytes,
        wopt_loom_bytes,
        base_code,
        loom_code,
        wopt_code,
        meld_base,
        meld_wopt,
        meld_loom,
    }
}

// ---------------------------------------------------------------------------
// Markdown emission.
// ---------------------------------------------------------------------------
fn pct_delta(new: Option<u64>, base: Option<u64>) -> String {
    match (new, base) {
        (Some(n), Some(b)) if b > 0 => {
            let d = (n as f64 - b as f64) * 100.0 / b as f64;
            format!("{d:+.1}")
        }
        _ => "n/a".into(),
    }
}

fn cell_u64(v: Option<u64>) -> String {
    v.map(|n| n.to_string()).unwrap_or_else(|| "n/a".into())
}

fn render_markdown(env: &BenchEnv, rows: &[Row]) -> String {
    use std::fmt::Write;

    let timestamp = utc_now_iso8601();
    let mut s = String::new();
    let workspace_version = env!("CARGO_PKG_VERSION");
    writeln!(
        s,
        "# v{workspace_version} Corpus Baseline -- LOOM vs wasm-opt -O3"
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "_Generated by `cargo bench -p loom-testing --bench corpus_baseline` at `{timestamp}`._"
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(s, "- LOOM version: `{}`", env.loom_version).unwrap();
    match &env.wasm_opt_version {
        Some(v) => writeln!(s, "- wasm-opt: `{v}` (used)").unwrap(),
        None => writeln!(s, "- wasm-opt: NOT INSTALLED (wasm-opt columns marked n/a)").unwrap(),
    }
    writeln!(s, "- wasm-tools: `{}`", env.wasm_tools_version).unwrap();
    match &env.pin_status {
        PinStatus::Match { version } => {
            writeln!(s, "- wasm-opt pin: `{version}` (match)").unwrap()
        }
        PinStatus::Mismatch { pinned, installed } => writeln!(
            s,
            "- wasm-opt pin: **MISMATCH** (installed `{installed}` vs pinned `{pinned}`)"
        )
        .unwrap(),
        PinStatus::NoWasmOpt { pinned } => {
            writeln!(s, "- wasm-opt pin: `{pinned}` (wasm-opt not installed)").unwrap()
        }
        PinStatus::PinUnreadable => {
            writeln!(s, "- wasm-opt pin: _file missing or malformed_").unwrap()
        }
    }
    writeln!(s).unwrap();

    // File-size table.
    writeln!(
        s,
        "## Results -- file size (total bytes incl. all sections)"
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "| Workload | Baseline | LOOM | wasm-opt -O3 | wasm-opt -> LOOM | LOOM \u{0394}% | wasm-opt \u{0394}% | Note |"
    )
    .unwrap();
    writeln!(s, "|---|---:|---:|---:|---:|---:|---:|---|").unwrap();
    for r in rows {
        writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} | {} |",
            r.name,
            cell_u64(r.base_bytes),
            cell_u64(r.loom_bytes),
            cell_u64(r.wopt_bytes),
            cell_u64(r.wopt_loom_bytes),
            pct_delta(r.loom_bytes, r.base_bytes),
            pct_delta(r.wopt_bytes, r.base_bytes),
            r.note,
        )
        .unwrap();
    }
    writeln!(s).unwrap();

    // Code-section table.
    writeln!(s, "## Results -- code section only (optimizer-relevant)").unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "| Workload | Baseline (code) | LOOM (code) | wasm-opt (code) | LOOM code \u{0394}% | wasm-opt code \u{0394}% | Note |"
    )
    .unwrap();
    writeln!(s, "|---|---:|---:|---:|---:|---:|---|").unwrap();
    for r in rows {
        writeln!(
            s,
            "| {} | {} | {} | {} | {} | {} | {} |",
            r.name,
            cell_u64(r.base_code),
            cell_u64(r.loom_code),
            cell_u64(r.wopt_code),
            pct_delta(r.loom_code, r.base_code),
            pct_delta(r.wopt_code, r.base_code),
            r.note,
        )
        .unwrap();
    }
    writeln!(s).unwrap();

    // Meld table.
    let meld_rows: Vec<&Row> = rows.iter().filter(|r| r.meld_base.is_some()).collect();
    if !meld_rows.is_empty() {
        writeln!(s, "## Components via meld (fused-core baseline)").unwrap();
        writeln!(s).unwrap();
        writeln!(
            s,
            "| Workload | meld baseline | wasm-opt -O3 | LOOM | wasm-opt \u{0394}% | LOOM \u{0394}% | Note |"
        )
        .unwrap();
        writeln!(s, "|---|---:|---:|---:|---:|---:|---|").unwrap();
        for r in meld_rows {
            writeln!(
                s,
                "| {} | {} | {} | {} | {} | {} | {} |",
                r.name,
                cell_u64(r.meld_base),
                cell_u64(r.meld_wopt),
                cell_u64(r.meld_loom),
                pct_delta(r.meld_wopt, r.meld_base),
                pct_delta(r.meld_loom, r.meld_base),
                r.note,
            )
            .unwrap();
        }
        writeln!(s).unwrap();
    }

    writeln!(s, "## Methodology").unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "For each workload (fixture path relative to repo root):"
    )
    .unwrap();
    writeln!(
        s,
        "1. Record baseline byte count via `fs::metadata` and code-section size via `wasm-tools objdump`."
    )
    .unwrap();
    writeln!(
        s,
        "2. Run `loom optimize <fixture> --attestation false -o <name>.loom.wasm`."
    )
    .unwrap();
    writeln!(
        s,
        "3. Run `wasm-opt -O3 <fixture> -o <name>.wopt.wasm` (skipped if wasm-opt unavailable)."
    )
    .unwrap();
    writeln!(s, "4. Re-run LOOM on the wasm-opt output.").unwrap();
    writeln!(
        s,
        "5. For Component-Model fixtures, run `meld fuse` to obtain a fused core module; then run wasm-opt and LOOM on it. The meld output is its own baseline."
    )
    .unwrap();
    writeln!(
        s,
        "6. Validate every output via `wasm-tools validate`. Failures are reported as missing columns."
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(
        s,
        "Tool versions are pinned via `scripts/wasm-opt.pinned`; mismatches surface in the header above."
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(s, "## Reproducing").unwrap();
    writeln!(s).unwrap();
    writeln!(s, "```bash").unwrap();
    writeln!(s, "# Build LOOM first (Z3 verification enabled)").unwrap();
    writeln!(s, "Z3_SYS_Z3_HEADER=/opt/homebrew/include/z3.h \\").unwrap();
    writeln!(
        s,
        "  LIBRARY_PATH=/opt/homebrew/lib cargo build --release"
    )
    .unwrap();
    writeln!(s).unwrap();
    writeln!(s, "# Run the criterion harness").unwrap();
    writeln!(s, "cargo bench -p loom-testing --bench corpus_baseline").unwrap();
    writeln!(s, "```").unwrap();
    s
}

/// Minimal UTC `YYYY-MM-DDTHH:MM:SSZ` without pulling in chrono.
fn utc_now_iso8601() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (y, mo, d, h, mi, se) = epoch_to_ymdhms(secs);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{mi:02}:{se:02}Z")
}

fn epoch_to_ymdhms(secs: u64) -> (i64, u32, u32, u32, u32, u32) {
    // Howard Hinnant's civil_from_days algorithm.
    let days = (secs / 86_400) as i64;
    let rem = secs % 86_400;
    let h = (rem / 3600) as u32;
    let mi = ((rem % 3600) / 60) as u32;
    let se = (rem % 60) as u32;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = z - era * 146_097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365; // [0, 399]
    let mut y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32; // [1, 31]
    let mo = (if mp < 10 { mp + 3 } else { mp - 9 }) as u32; // [1, 12]
    if mo <= 2 {
        y += 1;
    }
    (y, mo, d, h, mi, se)
}

// ---------------------------------------------------------------------------
// Criterion glue. We collect rows in a Mutex<Vec<Option<Row>>>; each fixture
// has a bench that records timing for the LOOM pass and (the first time it's
// invoked) populates its row. Markdown is rendered via a libc atexit hook
// registered the first time the bench group is built.
// ---------------------------------------------------------------------------
static ROWS: OnceLock<Mutex<Vec<Option<Row>>>> = OnceLock::new();
static ENV_CELL: OnceLock<BenchEnv> = OnceLock::new();

fn rows() -> &'static Mutex<Vec<Option<Row>>> {
    ROWS.get_or_init(|| Mutex::new(vec![None; WORKLOADS.len()]))
}

fn shared_env() -> &'static BenchEnv {
    ENV_CELL.get_or_init(|| {
        let e = resolve_env();
        print_pin_banner(&e);
        e
    })
}

fn emit_report() {
    let env = shared_env();
    let guard = rows().lock().unwrap();
    let collected: Vec<Row> = guard
        .iter()
        .enumerate()
        .map(|(idx, row)| {
            row.clone().unwrap_or_else(|| {
                let (name, _, note) = WORKLOADS[idx];
                Row::missing(name, note)
            })
        })
        .collect();
    drop(guard);
    let md = render_markdown(env, &collected);

    println!("\n========== corpus_baseline markdown report ==========");
    println!("{md}");
    println!("==========       end markdown report       ==========\n");

    // Write to a `-criterion.md` sidecar so we never clobber the
    // shell-harness-generated `v<X>-corpus-baseline.md`. The sidecar is
    // overwritten every bench run -- if you want history, copy it to a
    // versioned name before re-running.
    let workspace_version = env!("CARGO_PKG_VERSION");
    let report_path = env
        .repo_root
        .join("docs/measurements")
        .join(format!("v{workspace_version}-corpus-baseline-criterion.md"));
    if let Some(parent) = report_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    match fs::write(&report_path, md) {
        Ok(_) => eprintln!("[bench] wrote report to {}", report_path.display()),
        Err(e) => eprintln!(
            "[bench] WARNING: failed to write report to {}: {e}",
            report_path.display()
        ),
    }
}

fn corpus_baseline(c: &mut Criterion) {
    // Register the report emitter exactly once via libc atexit. Rust's
    // `static` drop semantics don't fire for non-Drop statics, and
    // criterion_main returns cleanly, so we register a C-level hook.
    static GUARD_INSTALLED: OnceLock<()> = OnceLock::new();
    GUARD_INSTALLED.get_or_init(|| {
        extern "C" fn run_report() {
            emit_report();
        }
        libc_atexit(run_report);
    });

    let env = shared_env();
    if !env.loom.exists() {
        eprintln!(
            "[bench] LOOM binary not found at {}; build with `cargo build --release` first",
            env.loom.display()
        );
        eprintln!("[bench] proceeding anyway: all LOOM columns will be marked n/a");
    }

    let mut group = c.benchmark_group("corpus_baseline");
    // Keep the timing budget modest -- we care more about correctness of the
    // measurement matrix than about Criterion's confidence interval here.
    group.sample_size(10);
    group.measurement_time(Duration::from_secs(5));
    group.warm_up_time(Duration::from_millis(500));

    for (idx, &(name, path, note)) in WORKLOADS.iter().enumerate() {
        group.bench_function(name, |b| {
            b.iter(|| {
                // Record the row on the first invocation; subsequent criterion
                // iterations short-circuit (we don't want to re-run a
                // multi-minute LOOM pass per sample).
                let mut guard = rows().lock().unwrap();
                if guard[idx].is_none() {
                    let row = measure_fixture(env, name, path, note);
                    guard[idx] = Some(row);
                }
                criterion::black_box(idx)
            });
        });
    }

    group.finish();
}

// Minimal libc atexit binding without an extra crate dep. The C signature is
// `int atexit(void (*fn)(void))`; we ignore the return value (per POSIX, it
// fails only on resource exhaustion, in which case we're already in trouble).
unsafe extern "C" {
    fn atexit(cb: extern "C" fn()) -> i32;
}
fn libc_atexit(cb: extern "C" fn()) {
    unsafe {
        let _ = atexit(cb);
    }
}

criterion_group!(benches, corpus_baseline);
criterion_main!(benches);
