//! The validator trust ledger (TEST-O3-VALIDATOR-TRUST-LEDGER).
//!
//! `safety/VALIDATOR_TRUST_LEDGER.txt` lists every way the translation
//! validator accepts a transform without proving it, and every `LOOM_*`
//! variable that changes what it decides. Two of its sections are derived
//! from the code and checked here:
//!
//! - `kept-unproven:` one line per [`KeptReason`] variant. The enum is the
//!   only argument `record_kept_unproven` takes, so the compiler, not a text
//!   search, enumerates these.
//! - `env:` one line per `"LOOM_*"` name read in `loom-core/src` or
//!   `loom-cli/src`. An `env::var` call whose name is not a string literal
//!   fails the check, because its name cannot be listed.
//!
//! `structural:` lines are reviewed, not derived. A line leaves the ledger
//! only as a `removed:` line that links the change removing the escape; with
//! `LOOM_LEDGER_BASE` naming the base branch's copy, a line that disappears
//! without one fails (CI sets it on pull requests).

use loom_core::stats::KeptReason;
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

const LEDGER: &str = "safety/VALIDATOR_TRUST_LEDGER.txt";
const SCANNED: [&str; 2] = ["loom-core/src", "loom-cli/src"];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..")
}

/// One parsed ledger line.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum Entry {
    Kept(String),
    Env(String),
    Structural(String),
    /// The removed entry, written as it was (`kind: name`), and its link.
    Removed(String, String),
}

impl Entry {
    /// `kind: name`, the identity a removal refers to.
    fn key(&self) -> Option<String> {
        match self {
            Entry::Kept(n) => Some(format!("kept-unproven: {n}")),
            Entry::Env(n) => Some(format!("env: {n}")),
            Entry::Structural(n) => Some(format!("structural: {n}")),
            Entry::Removed(..) => None,
        }
    }
}

/// Parse the ledger. Every non-comment line is `kind: name | text`.
fn parse_ledger(text: &str) -> Result<Vec<Entry>, Vec<String>> {
    let mut out = Vec::new();
    let mut errors = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let Some((kind, rest)) = line.split_once(':') else {
            errors.push(format!("line {}: no `kind:` prefix: {line}", i + 1));
            continue;
        };
        let (name, text) = match rest.split_once(" | ") {
            Some((n, t)) => (n.trim().to_string(), t.trim().to_string()),
            None => (rest.trim().to_string(), String::new()),
        };
        if name.is_empty() || text.is_empty() {
            errors.push(format!("line {}: needs `name | text`: {line}", i + 1));
            continue;
        }
        out.push(match kind.trim() {
            "kept-unproven" => Entry::Kept(name),
            "env" => Entry::Env(name),
            "structural" => Entry::Structural(name),
            "removed" => Entry::Removed(name, text),
            other => {
                errors.push(format!("line {}: unknown kind `{other}`", i + 1));
                continue;
            }
        });
    }
    if errors.is_empty() {
        Ok(out)
    } else {
        Err(errors)
    }
}

/// A removal must point at the change that removed the escape.
fn is_link(text: &str) -> bool {
    let t = text.trim();
    let pr = t.strip_prefix('#').is_some_and(|n| {
        let n = n.split_whitespace().next().unwrap_or("");
        !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())
    });
    pr || t.starts_with("https://github.com/pulseengine/loom/pull/")
        || t.starts_with("https://github.com/pulseengine/loom/commit/")
}

/// The `"LOOM_*"` names a Rust source reads, and the `env::var` calls whose
/// name is not a literal (reported as `file:line`).
fn env_names(path: &str, src: &str) -> (BTreeSet<String>, Vec<String>) {
    let mut names = BTreeSet::new();
    let mut computed = Vec::new();
    let lines: Vec<&str> = src.lines().collect();
    for (i, raw) in lines.iter().enumerate() {
        let line = raw.trim_start();
        if line.starts_with("//") {
            continue;
        }
        names.extend(loom_literals(line));
        for call in ["env::var(", "env::var_os("] {
            if let Some(at) = line.find(call) {
                let after = &line[at + call.len()..];
                // rustfmt may put the argument on the next line.
                let arg = if after.trim().is_empty() {
                    lines.get(i + 1).map(|l| l.trim()).unwrap_or("")
                } else {
                    after.trim()
                };
                if !arg.starts_with('"') {
                    computed.push(format!("{path}:{}", i + 1));
                }
            }
        }
    }
    (names, computed)
}

/// Every `"LOOM_X"` literal (at least one character after `LOOM_`, all of
/// `A-Z0-9_`) on a line.
fn loom_literals(line: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = line;
    while let Some(at) = rest.find("\"LOOM_") {
        let body = &rest[at + 1..];
        let end = body
            .find(|c: char| !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_'))
            .unwrap_or(body.len());
        if body[end..].starts_with('"') && end > "LOOM_".len() {
            out.push(body[..end].to_string());
        }
        rest = &body[end..];
    }
    out
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap_or_else(|e| panic!("{}: {e}", dir.display())) {
        let p = e.unwrap().path();
        if p.is_dir() {
            rust_files(&p, out);
        } else if p.extension().is_some_and(|x| x == "rs") {
            out.push(p);
        }
    }
}

/// What the code says the ledger must list.
struct Derived {
    kept: BTreeSet<String>,
    env: BTreeSet<String>,
    computed: Vec<String>,
}

fn derive_from_code() -> Derived {
    let kept = KeptReason::ALL
        .iter()
        .map(|r| r.as_str().to_string())
        .collect();
    let mut env = BTreeSet::new();
    let mut computed = Vec::new();
    for dir in SCANNED {
        let mut files = Vec::new();
        rust_files(&repo().join(dir), &mut files);
        assert!(!files.is_empty(), "no Rust sources under {dir}");
        for f in files {
            let src = std::fs::read_to_string(&f).unwrap();
            let rel = f.strip_prefix(repo()).unwrap_or(&f).display().to_string();
            let (n, c) = env_names(&rel, &src);
            env.extend(n);
            computed.extend(c);
        }
    }
    Derived {
        kept,
        env,
        computed,
    }
}

/// Every rule that needs only the ledger and the code. Returns the errors.
fn check(ledger: &[Entry], code: &Derived) -> Vec<String> {
    let mut errors = Vec::new();
    for c in &code.computed {
        errors.push(format!(
            "{c}: env::var with a computed name; the ledger cannot list it"
        ));
    }
    let listed = |f: fn(&Entry) -> Option<&String>| -> BTreeSet<String> {
        ledger.iter().filter_map(f).cloned().collect()
    };
    let kept = listed(|e| match e {
        Entry::Kept(n) => Some(n),
        _ => None,
    });
    let env = listed(|e| match e {
        Entry::Env(n) => Some(n),
        _ => None,
    });
    for (kind, in_code, in_ledger) in [
        ("kept-unproven", &code.kept, &kept),
        ("env", &code.env, &env),
    ] {
        for n in in_code.difference(in_ledger) {
            errors.push(format!(
                "not in {LEDGER}: `{kind}: {n}` (add it with a reason)"
            ));
        }
        for n in in_ledger.difference(in_code) {
            errors.push(format!(
                "no longer in the code: `{kind}: {n}` (replace it with \
                 `removed: {kind}: {n} | <link to the change>`)"
            ));
        }
    }
    let live: BTreeSet<String> = ledger.iter().filter_map(Entry::key).collect();
    for e in ledger {
        if let Entry::Removed(what, link) = e {
            if !is_link(link) {
                errors.push(format!(
                    "`removed: {what}` has no link to the change (#N or a loom pull/commit URL): {link}"
                ));
            }
            if live.contains(what) {
                errors.push(format!("`{what}` is listed both live and removed"));
            }
        }
    }
    errors
}

/// A line of the base ledger may leave only as a linked `removed:` line.
fn check_against_base(base: &[Entry], head: &[Entry]) -> Vec<String> {
    let head_live: BTreeSet<String> = head.iter().filter_map(Entry::key).collect();
    let head_removed: BTreeSet<&String> = head
        .iter()
        .filter_map(|e| match e {
            Entry::Removed(what, _) => Some(what),
            _ => None,
        })
        .collect();
    let mut errors = Vec::new();
    for key in base.iter().filter_map(Entry::key) {
        if !head_live.contains(&key) && !head_removed.contains(&key) {
            errors.push(format!(
                "`{key}` was deleted from {LEDGER} without a `removed: {key} | <link>` line"
            ));
        }
    }
    for e in base {
        if let Entry::Removed(what, _) = e {
            if !head_removed.contains(what) {
                errors.push(format!("the `removed: {what}` line was deleted"));
            }
        }
    }
    errors
}

fn read_ledger() -> Vec<Entry> {
    let text =
        std::fs::read_to_string(repo().join(LEDGER)).unwrap_or_else(|e| panic!("{LEDGER}: {e}"));
    parse_ledger(&text).unwrap_or_else(|e| panic!("{LEDGER} does not parse:\n{}", e.join("\n")))
}

#[test]
fn the_ledger_lists_every_trust_escape_in_the_code() {
    let ledger = read_ledger();
    let code = derive_from_code();
    // The derivation must find something, or an empty ledger would pass.
    assert_eq!(code.kept.len(), KeptReason::ALL.len());
    assert!(
        !code.env.is_empty(),
        "no LOOM_* read found under {SCANNED:?}"
    );
    let errors = check(&ledger, &code);
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

/// On pull requests CI writes the base branch's ledger to a file and names it
/// in `LOOM_LEDGER_BASE`; a line may only leave as a linked removal.
#[test]
fn ledger_lines_leave_only_as_linked_removals() {
    let Some(base_path) = std::env::var_os("LOOM_LEDGER_BASE") else {
        println!("LOOM_LEDGER_BASE unset: base comparison not run");
        return;
    };
    let text = std::fs::read_to_string(&base_path)
        .unwrap_or_else(|e| panic!("LOOM_LEDGER_BASE={}: {e}", base_path.to_string_lossy()));
    if text.trim().is_empty() {
        println!("the base branch has no ledger yet: base comparison not run");
        return;
    }
    let base = parse_ledger(&text).unwrap_or_else(|e| panic!("base ledger: {}", e.join("\n")));
    let errors = check_against_base(&base, &read_ledger());
    assert!(errors.is_empty(), "{}", errors.join("\n"));
}

/// `ALL` and the labels agree: no variant is left out of `ALL` (which the
/// ledger check iterates) and no two share a label.
#[test]
fn every_kept_reason_has_a_distinct_label() {
    let labels: BTreeSet<&str> = KeptReason::ALL.iter().map(|r| r.as_str()).collect();
    assert_eq!(labels.len(), KeptReason::ALL.len());
    // A variant missing from ALL would compile; this match would not.
    for r in KeptReason::ALL {
        match r {
            KeptReason::OverSizeThreshold
            | KeptReason::FloatLoadStore
            | KeptReason::UnknownOpcode
            | KeptReason::BranchesApproximated
            | KeptReason::UnknownInstruction
            | KeptReason::FinalMemoryNotCompared
            | KeptReason::TrapsNotCompared
            | KeptReason::VoidFunction
            | KeptReason::KInductionDisabled => {}
        }
    }
    let distinct: BTreeSet<KeptReason> = KeptReason::ALL.into_iter().collect();
    assert_eq!(distinct.len(), 9, "ALL lists a variant twice or misses one");
}

// ---- controls: each planted violation must fail with a named message ----

fn real() -> (Vec<Entry>, Derived) {
    (read_ledger(), derive_from_code())
}

#[test]
fn control_a_planted_kept_reason_fails_naming_it() {
    let (ledger, mut code) = real();
    code.kept.insert("planted reason".into());
    let e = check(&ledger, &code);
    assert!(
        e.iter()
            .any(|m| m.contains("not in") && m.contains("kept-unproven: planted reason")),
        "{e:?}"
    );
}

#[test]
fn control_a_planted_env_read_fails_naming_it() {
    let (ledger, mut code) = real();
    let (names, computed) = env_names(
        "planted.rs",
        "let x = std::env::var(\"LOOM_PLANTED_BUDGET\").ok();\n",
    );
    assert!(computed.is_empty());
    code.env.extend(names);
    let e = check(&ledger, &code);
    assert!(
        e.iter()
            .any(|m| m.contains("not in") && m.contains("env: LOOM_PLANTED_BUDGET")),
        "{e:?}"
    );
}

#[test]
fn control_a_computed_env_name_fails_naming_the_line() {
    let (ledger, mut code) = real();
    let src = "fn f(k: &str) {\n    let _ = std::env::var(k);\n    let _ = std::env::var_os(\n        name,\n    );\n}\n";
    let (_, computed) = env_names("planted.rs", src);
    assert_eq!(computed, ["planted.rs:2", "planted.rs:3"]);
    code.computed.extend(computed);
    let e = check(&ledger, &code);
    assert!(
        e.iter()
            .any(|m| m.starts_with("planted.rs:2: env::var with a computed name")),
        "{e:?}"
    );
}

#[test]
fn control_a_stale_entry_fails() {
    let (mut ledger, code) = real();
    ledger.push(Entry::Env("LOOM_GONE".into()));
    let e = check(&ledger, &code);
    assert!(
        e.iter()
            .any(|m| m.contains("no longer in the code") && m.contains("env: LOOM_GONE")),
        "{e:?}"
    );
}

#[test]
fn control_a_removal_without_a_link_fails() {
    let (mut ledger, code) = real();
    ledger.push(Entry::Removed("env: LOOM_GONE".into(), "fixed it".into()));
    let e = check(&ledger, &code);
    assert!(e.iter().any(|m| m.contains("has no link")), "{e:?}");
    // The same removal with a link passes.
    let (mut ledger, code) = real();
    ledger.push(Entry::Removed("env: LOOM_GONE".into(), "#999".into()));
    assert!(check(&ledger, &code).is_empty());
}

#[test]
fn control_a_deletion_against_the_base_fails() {
    let head = read_ledger();
    let mut base = head.clone();
    base.push(Entry::Structural("planted-escape".into()));
    let e = check_against_base(&base, &head);
    assert_eq!(
        e,
        [format!(
            "`structural: planted-escape` was deleted from {LEDGER} without a \
             `removed: structural: planted-escape | <link>` line"
        )]
    );
    let mut head2 = head.clone();
    head2.push(Entry::Removed(
        "structural: planted-escape".into(),
        "#999".into(),
    ));
    assert!(check_against_base(&base, &head2).is_empty());
    // An unchanged ledger passes against itself.
    assert!(check_against_base(&head, &head).is_empty());
}

#[test]
fn literal_scanner_ignores_comments_and_prefix_filters() {
    let src = "// std::env::var(\"LOOM_IN_COMMENT\")\n\
               /// `\"LOOM_IN_DOC\"`\n\
               let v = std::env::vars().filter(|(k, _)| k.starts_with(\"LOOM_\"));\n";
    let (names, computed) = env_names("x.rs", src);
    assert!(names.is_empty(), "{names:?}");
    assert!(computed.is_empty(), "{computed:?}");
}
