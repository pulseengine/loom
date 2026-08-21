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
    let mut map = counters().lock().expect("revert counter mutex poisoned");
    *map.entry(pass_name.to_string()).or_insert(0) += 1;
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
    let mut c = coverage().lock().expect("coverage mutex poisoned");
    c.proven += 1;
}

/// Record an attempt whose transform was KEPT although nothing was proven.
///
/// `reason` must name the specific reason the obligation could not be
/// discharged ("float load/store", "unknown opcode", "body over the solver
/// size threshold"), because an unattributed count cannot be acted on: the
/// operator needs to know whether to widen the encoder, raise a threshold, or
/// stop trusting the result.
pub fn record_kept_unproven(reason: &str) {
    let mut c = coverage().lock().expect("coverage mutex poisoned");
    *c.kept_unproven.entry(reason.to_string()).or_insert(0) += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_read() {
        // Use a unique pass name so we do not race with other tests
        // that may run concurrently and share the global counter.
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

    #[test]
    fn take_resets() {
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
