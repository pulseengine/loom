//! Optimization observability counters.
//!
//! When a pass attempts a transformation that the Z3 verifier rejects,
//! the per-function `verify_or_revert` mechanism reverts the function
//! to its pre-pass instructions and locals. Without instrumentation
//! these reverts are silent — the build reports "success" even when
//! every function in a module reverted.
//!
//! This module records revert counts by pass name so callers (CLI
//! `--stats`, CI gates, tests) can observe how often verification
//! actually catches and reverts a transform.
//!
//! Counters are global and process-wide; tests and CLI invocations
//! that want isolated counts should call `take_revert_summary()`
//! between runs.

use std::collections::BTreeMap;
use std::sync::{Mutex, OnceLock};

fn counters() -> &'static Mutex<BTreeMap<String, u64>> {
    static COUNTERS: OnceLock<Mutex<BTreeMap<String, u64>>> = OnceLock::new();
    COUNTERS.get_or_init(|| Mutex::new(BTreeMap::new()))
}

/// Increment the revert counter for the named pass.
pub fn record_revert(pass_name: &str) {
    {
        let mut map = counters().lock().expect("revert counter mutex poisoned");
        *map.entry(pass_name.to_string()).or_insert(0) += 1;
    }
    finish(|_| Outcome::Reverted(pass_name.to_string()));
}

/// Read a snapshot of the current revert counters.
///
/// Returns a sorted map (pass name → revert count). Counters are not
/// reset; use `take_revert_summary` for read-and-reset semantics.
pub fn revert_summary() -> BTreeMap<String, u64> {
    counters()
        .lock()
        .expect("revert counter mutex poisoned")
        .clone()
}

/// Read the current revert counters and reset them to zero.
///
/// Useful for tests that want to assert on revert counts without
/// being affected by other tests in the same process.
pub fn take_revert_summary() -> BTreeMap<String, u64> {
    let mut map = counters().lock().expect("revert counter mutex poisoned");
    let snapshot = map.clone();
    map.clear();
    snapshot
}

/// Total reverts across all passes.
pub fn total_reverts() -> u64 {
    counters()
        .lock()
        .expect("revert counter mutex poisoned")
        .values()
        .sum()
}

/// Verification coverage: what the translation validator actually
/// ESTABLISHED, as opposed to what it was asked about.
///
/// # Why this exists separately from the revert counters
///
/// The revert counters answer "how often did verification catch something?".
/// They cannot answer "how much was verified at all?", because they have no
/// denominator — and the interesting failure is not a caught miscompile, it
/// is a transform that was KEPT without a proof. Those are invisible to a
/// revert count by construction: nothing reverted.
///
/// loom keeps such transforms deliberately and for stated reasons (a float
/// load/store the encoder models imprecisely, an unknown opcode, a body over
/// the solver size threshold). That is a defensible engineering position. It
/// stops being defensible the moment the tool reports success without saying
/// it happened — a "no silent failures" charter (REQ-3) is not satisfied by a
/// failure that is merely not an error.
///
/// So every attempt lands in exactly one of three buckets: **proven**,
/// **kept without proof** (with the reason), or **reverted**.
///
/// # What the denominator counts
///
/// One entry per (function, pass) attempt — NOT per function. A module of 10
/// functions run through 15 passes produces up to 150 attempts, and with
/// `--islands N` the same function is verified once per pipeline. The
/// reporting surface says so rather than implying a function count.
fn coverage() -> &'static Mutex<Coverage> {
    static COVERAGE: OnceLock<Mutex<Coverage>> = OnceLock::new();
    COVERAGE.get_or_init(|| Mutex::new(Coverage::default()))
}

#[derive(Debug, Default, Clone)]
struct Coverage {
    proven: u64,
    kept_unproven: BTreeMap<String, u64>,
}

/// A snapshot of verification coverage for a run.
#[derive(Debug, Default, Clone)]
pub struct CoverageSummary {
    /// Attempts the validator discharged with an actual proof.
    pub proven: u64,
    /// Attempts whose transform was KEPT although nothing was proven,
    /// keyed by the stated reason. These are not reverts; the code shipped.
    pub kept_unproven: BTreeMap<String, u64>,
    /// Attempts the validator rejected, so the pass reverted the function.
    /// Taken from the revert counters, which is where reverts are recorded.
    pub reverted: u64,
}

impl CoverageSummary {
    /// Total verification attempts — the denominator.
    pub fn attempts(&self) -> u64 {
        self.proven + self.total_kept_unproven() + self.reverted
    }

    /// Attempts kept without a proof, across all reasons.
    pub fn total_kept_unproven(&self) -> u64 {
        self.kept_unproven.values().sum()
    }

    /// Proven share of all attempts, or `None` when nothing was attempted —
    /// deliberately not `100.0`, because a run that verified nothing has not
    /// achieved full coverage, and a percentage would say it had.
    pub fn proven_percentage(&self) -> Option<f64> {
        let attempts = self.attempts();
        if attempts == 0 {
            return None;
        }
        Some(self.proven as f64 * 100.0 / attempts as f64)
    }

    /// True when every attempt was discharged by an actual proof.
    pub fn is_fully_proven(&self) -> bool {
        self.attempts() > 0 && self.proven == self.attempts()
    }
}

/// Record an attempt the validator discharged with a real proof.
pub fn record_proven() {
    {
        let mut c = coverage().lock().expect("coverage mutex poisoned");
        c.proven += 1;
    }
    finish(Outcome::Proven);
}

/// Record an attempt whose transform was KEPT although nothing was proven.
///
/// `reason` must name the specific reason the obligation could not be
/// discharged ("float load/store", "unknown opcode", "body over the solver
/// size threshold"), because an unattributed count cannot be acted on: the
/// operator needs to know whether to widen the encoder, raise a threshold, or
/// stop trusting the result.
pub fn record_kept_unproven(reason: &str) {
    {
        let mut c = coverage().lock().expect("coverage mutex poisoned");
        *c.kept_unproven.entry(reason.to_string()).or_insert(0) += 1;
    }
    finish(|_| Outcome::KeptUnproven(reason.to_string()));
}

/// Read a snapshot of verification coverage. Counters are not reset.
pub fn coverage_summary() -> CoverageSummary {
    let c = coverage().lock().expect("coverage mutex poisoned").clone();
    CoverageSummary {
        proven: c.proven,
        kept_unproven: c.kept_unproven,
        reverted: total_reverts(),
    }
}

/// Read verification coverage and reset it, so a test can assert on its own
/// counts without being affected by others in the same process.
pub fn take_coverage_summary() -> CoverageSummary {
    let mut c = coverage().lock().expect("coverage mutex poisoned");
    let snapshot = c.clone();
    *c = Coverage::default();
    drop(c);
    CoverageSummary {
        proven: snapshot.proven,
        kept_unproven: snapshot.kept_unproven,
        reverted: total_reverts(),
    }
}

// ============================================================================
// Per-attempt accounting (TEST-O3-STATS-JSON-ENGINE)
// ============================================================================
//
// `CoverageSummary` computes its denominator as the sum of its buckets, so it
// cannot notice an attempt that landed in no bucket. This ledger counts an
// attempt when the validator is ENTERED, independently of how it ends, and
// keeps one record per attempt with the deciding engine. The two disagree
// exactly when the instrumentation has a gap:
//
// - `unclassified`: an attempt was entered and a later one began on the same
//   thread before any outcome was recorded for it;
// - `outcomes_without_entry`: an outcome was recorded with no attempt open,
//   for example a structural revert that never reached the validator.
//
// The deciding engine is taken from what actually ran during the attempt:
// ordeal decisions at the solver seam and Z3 `check()` calls in the
// validator. A Z3 `Unsat` carries no certificate; an ordeal `Unsat` carries
// one that loom re-checks.

/// Which engine decided an accepted attempt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Engine {
    /// Every solver decision in the attempt came from ordeal.
    Ordeal,
    /// Every solver decision in the attempt came from Z3.
    Z3,
    /// Both engines decided parts of the attempt.
    Mixed,
    /// The pass left the function byte-for-byte unchanged.
    Unchanged,
    /// Accepted without any solver call, for example when the exact executor
    /// built the same term for both sides.
    NoSolver,
}

impl Engine {
    /// Stable label used in reports.
    pub fn label(self) -> &'static str {
        match self {
            Engine::Ordeal => "ordeal",
            Engine::Z3 => "z3",
            Engine::Mixed => "mixed",
            Engine::Unchanged => "unchanged",
            Engine::NoSolver => "no_solver",
        }
    }

    /// True when every decision in the attempt carries a re-checked
    /// certificate.
    pub fn is_certified(self) -> bool {
        matches!(self, Engine::Ordeal)
    }
}

/// How one attempt ended.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    /// Accepted with a proof, decided by the given engine(s).
    Proven(Engine),
    /// Accepted although nothing was proven, for the stated reason.
    KeptUnproven(String),
    /// Rejected; the function was reverted. Carries the revert key.
    Reverted(String),
}

/// One validator attempt: one function after one pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AttemptRecord {
    /// Pass that produced the candidate.
    pub pass: String,
    /// Function name from the name section, when the module has one.
    pub function: Option<String>,
    /// How the attempt ended.
    pub outcome: Outcome,
}

/// Snapshot of the per-attempt ledger.
#[derive(Debug, Default, Clone)]
pub struct AttemptAccounting {
    /// Attempts counted when the validator was entered.
    pub entered: u64,
    /// One record per attempt that reached an outcome.
    pub records: Vec<AttemptRecord>,
    /// Attempts entered that never reached an outcome.
    pub unclassified: u64,
    /// Outcomes recorded while no attempt was open.
    pub outcomes_without_entry: u64,
}

impl AttemptAccounting {
    /// Accepted attempts by deciding engine.
    pub fn proven_by_engine(&self) -> BTreeMap<Engine, u64> {
        let mut m = BTreeMap::new();
        for r in &self.records {
            if let Outcome::Proven(e) = r.outcome {
                *m.entry(e).or_insert(0) += 1;
            }
        }
        m
    }

    /// Accepted attempts whose every decision carries a re-checked
    /// certificate.
    pub fn proven_certified(&self) -> u64 {
        self.proven_by_engine()
            .iter()
            .filter(|(e, _)| e.is_certified())
            .map(|(_, n)| n)
            .sum()
    }

    /// Records with each outcome kind: (proven, kept without proof, reverted).
    pub fn outcome_counts(&self) -> (u64, u64, u64) {
        let mut c = (0, 0, 0);
        for r in &self.records {
            match r.outcome {
                Outcome::Proven(_) => c.0 += 1,
                Outcome::KeptUnproven(_) => c.1 += 1,
                Outcome::Reverted(_) => c.2 += 1,
            }
        }
        c
    }

    /// True when every entered attempt reached exactly one outcome and no
    /// outcome arrived without an attempt.
    pub fn is_consistent(&self) -> bool {
        self.unclassified == 0
            && self.outcomes_without_entry == 0
            && self.records.len() as u64 == self.entered
    }
}

#[derive(Debug)]
struct Pending {
    pass: String,
    function: Option<String>,
    ordeal: u32,
    z3: u32,
    unchanged: bool,
}

thread_local! {
    static PENDING: std::cell::RefCell<Option<Pending>> = const { std::cell::RefCell::new(None) };
}

fn accounting() -> &'static Mutex<AttemptAccounting> {
    static ACCOUNTING: OnceLock<Mutex<AttemptAccounting>> = OnceLock::new();
    ACCOUNTING.get_or_init(|| Mutex::new(AttemptAccounting::default()))
}

/// Open a validator attempt on this thread. An attempt still open from an
/// earlier call is counted as unclassified.
pub fn begin_attempt(pass: &str, function: Option<&str>) {
    let stale = PENDING.with(|p| {
        p.borrow_mut().replace(Pending {
            pass: pass.to_string(),
            function: function.map(str::to_string),
            ordeal: 0,
            z3: 0,
            unchanged: false,
        })
    });
    let mut a = accounting().lock().expect("accounting mutex poisoned");
    a.entered += 1;
    if stale.is_some() {
        a.unclassified += 1;
    }
}

/// Open an attempt only if none is open on this thread. For entry points
/// that can be reached both directly and from inside an open attempt.
pub fn ensure_attempt(pass: &str, function: Option<&str>) {
    if PENDING.with(|p| p.borrow().is_none()) {
        begin_attempt(pass, function);
    }
}

/// Note that Z3 decided an obligation inside the open attempt.
pub fn note_z3_check() {
    PENDING.with(|p| {
        if let Some(p) = p.borrow_mut().as_mut() {
            p.z3 += 1;
        }
    });
}

/// Note that the open attempt compares a function with itself: the pass
/// changed nothing.
pub fn note_unchanged() {
    PENDING.with(|p| {
        if let Some(p) = p.borrow_mut().as_mut() {
            p.unchanged = true;
        }
    });
}

/// Note that ordeal decided an obligation inside the open attempt.
pub fn note_ordeal_decision() {
    PENDING.with(|p| {
        if let Some(p) = p.borrow_mut().as_mut() {
            p.ordeal += 1;
        }
    });
}

fn engine_of(p: &Pending) -> Engine {
    if p.unchanged && p.ordeal == 0 && p.z3 == 0 {
        return Engine::Unchanged;
    }
    match (p.ordeal > 0, p.z3 > 0) {
        (true, false) => Engine::Ordeal,
        (false, true) => Engine::Z3,
        (true, true) => Engine::Mixed,
        (false, false) => Engine::NoSolver,
    }
}

/// Close the open attempt with `outcome`; `proven` outcomes take their engine
/// from what ran during the attempt.
fn finish(make: impl FnOnce(Engine) -> Outcome) {
    let pending = PENDING.with(|p| p.borrow_mut().take());
    let mut a = accounting().lock().expect("accounting mutex poisoned");
    match pending {
        Some(p) => {
            let outcome = make(engine_of(&p));
            a.records.push(AttemptRecord {
                pass: p.pass,
                function: p.function,
                outcome,
            });
        }
        None => a.outcomes_without_entry += 1,
    }
}

/// Read a snapshot of the per-attempt ledger. An attempt still open on the
/// calling thread is reported as unclassified in the snapshot.
pub fn attempt_accounting() -> AttemptAccounting {
    let open = PENDING.with(|p| p.borrow().is_some());
    let mut a = accounting()
        .lock()
        .expect("accounting mutex poisoned")
        .clone();
    if open {
        a.unclassified += 1;
    }
    a
}

/// Read the per-attempt ledger and reset it, closing any attempt open on the
/// calling thread.
pub fn take_attempt_accounting() -> AttemptAccounting {
    let open = PENDING.with(|p| p.borrow_mut().take().is_some());
    let mut guard = accounting().lock().expect("accounting mutex poisoned");
    let mut a = std::mem::take(&mut *guard);
    if open {
        a.unclassified += 1;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Serialises the tests that depend on the process-global revert map.
    ///
    /// A unique pass name is not enough: `take_resets` clears the WHOLE map,
    /// so run on a parallel test thread it can erase `record_and_read`'s
    /// entries between its writes and its read, which then sees `None`.
    /// Measured on main: 1 failure in 400 runs of this module.
    static REVERT_MAP: Mutex<()> = Mutex::new(());

    fn lock_revert_map() -> std::sync::MutexGuard<'static, ()> {
        // A panicking test must not cascade into the others as poison.
        REVERT_MAP.lock().unwrap_or_else(|e| e.into_inner())
    }

    #[test]
    fn record_and_read() {
        let _guard = lock_revert_map();
        let pass = "stats_test_record_and_read";
        record_revert(pass);
        record_revert(pass);
        record_revert(pass);

        let summary = revert_summary();
        assert_eq!(summary.get(pass).copied(), Some(3));
    }

    /// The three buckets must stay distinct, and the denominator must be
    /// their sum. Before #331 there were only revert counters, so a transform
    /// kept without a proof landed in no bucket at all — or, for the solver
    /// size threshold, in the WRONG one, printing "reverted" for code that
    /// shipped.
    #[test]
    fn the_three_outcomes_are_accounted_separately() {
        let mut kept = BTreeMap::new();
        kept.insert("float load/store".to_string(), 2);
        kept.insert("body over the solver size threshold".to_string(), 1);
        let c = CoverageSummary {
            proven: 7,
            kept_unproven: kept,
            reverted: 4,
        };

        assert_eq!(c.total_kept_unproven(), 3);
        assert_eq!(c.attempts(), 14, "denominator must be the sum of all three");
        assert_eq!(c.proven_percentage(), Some(50.0));
        assert!(
            !c.is_fully_proven(),
            "a run with kept-unproven attempts is not fully proven"
        );
    }

    /// A run that attempted nothing must not report full coverage.
    ///
    /// `0/0` is the shape that invites a `100%` — it divides out to "no
    /// failures" — and that is precisely the vacuous claim this reporting
    /// exists to prevent. `None` forces the caller to say "0/0 attempts"
    /// rather than "100% proven".
    #[test]
    fn nothing_attempted_is_not_full_coverage() {
        let empty = CoverageSummary::default();
        assert_eq!(empty.attempts(), 0);
        assert_eq!(
            empty.proven_percentage(),
            None,
            "a run that verified nothing must not be reportable as a percentage"
        );
        assert!(
            !empty.is_fully_proven(),
            "verifying nothing is not verifying everything"
        );
    }

    /// A run in which every attempt was proven is the only case that may
    /// claim full coverage — the positive control for the test above.
    #[test]
    fn all_proven_is_full_coverage() {
        let c = CoverageSummary {
            proven: 5,
            ..Default::default()
        };
        assert!(c.is_fully_proven());
        assert_eq!(c.proven_percentage(), Some(100.0));
    }

    fn pending(ordeal: u32, z3: u32, unchanged: bool) -> Pending {
        Pending {
            pass: "p".into(),
            function: None,
            ordeal,
            z3,
            unchanged,
        }
    }

    /// The deciding engine is read from what ran: a Z3 call anywhere in the
    /// attempt makes it uncertified, and "unchanged" applies only when no
    /// solver ran at all.
    #[test]
    fn engine_attribution_reflects_what_ran() {
        assert_eq!(engine_of(&pending(2, 0, false)), Engine::Ordeal);
        assert_eq!(engine_of(&pending(0, 1, false)), Engine::Z3);
        assert_eq!(engine_of(&pending(1, 1, false)), Engine::Mixed);
        assert_eq!(engine_of(&pending(0, 0, false)), Engine::NoSolver);
        assert_eq!(engine_of(&pending(0, 0, true)), Engine::Unchanged);
        assert_eq!(engine_of(&pending(0, 1, true)), Engine::Z3);
        assert!(Engine::Ordeal.is_certified());
        for e in [
            Engine::Z3,
            Engine::Mixed,
            Engine::NoSolver,
            Engine::Unchanged,
        ] {
            assert!(!e.is_certified(), "{e:?} must not count as certified");
        }
    }

    /// The per-attempt ledger is consistent only when entries and records
    /// agree; each kind of gap breaks it.
    #[test]
    fn attempt_accounting_detects_gaps() {
        let rec = |o: Outcome| AttemptRecord {
            pass: "p".into(),
            function: None,
            outcome: o,
        };
        let ok = AttemptAccounting {
            entered: 3,
            records: vec![
                rec(Outcome::Proven(Engine::Ordeal)),
                rec(Outcome::KeptUnproven("r".into())),
                rec(Outcome::Reverted("p".into())),
            ],
            unclassified: 0,
            outcomes_without_entry: 0,
        };
        assert!(ok.is_consistent());
        assert_eq!(ok.outcome_counts(), (1, 1, 1));
        assert_eq!(ok.proven_certified(), 1);

        let mut more_entered = ok.clone();
        more_entered.entered = 4;
        assert!(!more_entered.is_consistent(), "an entry without a record");

        let mut unclassified = ok.clone();
        unclassified.unclassified = 1;
        assert!(!unclassified.is_consistent());

        let mut orphan = ok.clone();
        orphan.outcomes_without_entry = 1;
        assert!(!orphan.is_consistent());
    }

    #[test]
    fn take_resets() {
        let _guard = lock_revert_map();
        let pass = "stats_test_take_resets";
        record_revert(pass);
        let _ = take_revert_summary();
        let summary = revert_summary();
        assert!(
            !summary.contains_key(pass),
            "take_revert_summary should reset counters; pass still present"
        );
    }
}
