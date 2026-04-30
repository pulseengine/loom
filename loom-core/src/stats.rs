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
