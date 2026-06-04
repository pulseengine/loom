//! Ingest scry's published invariant contract (`scry-invariants/v1`) — loom#144.
//!
//! scry (the analyzer) proves abstract invariants about a Wasm module and
//! publishes them as a stable, versioned JSON document (schema id
//! `https://pulseengine.eu/scry-invariants/v1`, vendored at
//! `loom-core/contracts/scry-invariants-v1.schema.json`). This module is the
//! analyzer→optimizer half of the loop: loom ingests that JSON to strengthen
//! transformation preconditions from PROVEN facts.
//!
//! # What it unlocks (each gated on a matching, `sound` invariant)
//!
//! | scry invariant                                  | loom transform              |
//! |-------------------------------------------------|-----------------------------|
//! | singleton `i32-interval` (`lo == hi`) on a value| constant-fold → `i32.const` |
//! | `region-pointer` with offset within `memory`    | elide the load bounds check |
//! | singleton **sound** call-edge target set        | devirtualize `call_indirect`|
//!
//! # Soundness contract (defense-in-depth)
//!
//! 1. **Validate before trusting.** Input is parsed with `serde(deny_unknown_fields)`
//!    mirroring the schema's `additionalProperties: false` + required-field
//!    sets, and the `schema` URI is checked against [`SCHEMA_ID`]. An input that
//!    does not match the contract shape is rejected — loom never acts on an
//!    unvalidated invariant. (Full JSON-Schema validation — `pattern`/range —
//!    is a planned hardening; the structural + URI gates plus the Z3 backstop
//!    below are the soundness gates today.)
//! 2. **Z3 translation-validation remains the backstop.** Every
//!    invariant-dependent transform still runs through loom's existing
//!    `verify_or_revert` equivalence check. Even if scry asserted something
//!    false and loom acted on it, the pre/post equivalence check fails and the
//!    transform is reverted — a wrong invariant becomes a (loud) revert, never
//!    a silent miscompile. scry's soundness and loom's translation-validation
//!    are independent gates.
//!
//! This module is ingestion + indexing + precondition queries only; wiring the
//! three transforms onto these preconditions is done in the passes (each behind
//! the Z3 gate), tracked under #144.

use anyhow::{Result, anyhow, bail};
use serde::Deserialize;
use std::collections::HashMap;

/// The contract id loom consumes. An input whose `invariants.schema` differs is
/// rejected — a different id means a different (unvetted) contract version.
pub const SCHEMA_ID: &str = "https://pulseengine.eu/scry-invariants/v1";

/// scry's `analysis-result` — the root document.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AnalysisResult {
    /// Per-program-point abstract values.
    pub invariants: InvariantBundle,
    /// Resolved call graph (FEAT-006) — drives devirtualization.
    #[serde(rename = "call-graph")]
    pub call_graph: Vec<CallEdge>,
    /// Per-function summaries (FEAT-007).
    #[serde(rename = "function-summaries")]
    pub function_summaries: Vec<FunctionSummary>,
    /// Fields loom does not consume but the contract may carry. Accepted
    /// (so a conformant document validates) and ignored.
    #[serde(default)]
    pub diagnostics: Option<serde_json::Value>,
    #[serde(default)]
    pub provenance: Option<serde_json::Value>,
    #[serde(rename = "taint-findings", default)]
    pub taint_findings: Option<serde_json::Value>,
}

/// `invariant-bundle`: per-program-point abstract values (FEAT-001/FEAT-005).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InvariantBundle {
    /// JSON-schema URI this document conforms to (must be [`SCHEMA_ID`]).
    pub schema: String,
    /// SHA-256 of the analyzed module bytes, lowercase hex — lets loom confirm
    /// the invariants describe the module it is actually optimizing.
    #[serde(rename = "module-sha256")]
    pub module_sha256: String,
    /// Per-program-point invariants. Order is not significant.
    pub points: Vec<ProgramPoint>,
}

/// `program-point`: invariants at one `(func-index, pc)`.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProgramPoint {
    #[serde(rename = "func-index")]
    pub func_index: u32,
    /// Program counter (byte offset within the function body).
    pub pc: u32,
    pub locals: Vec<LocalInvariant>,
}

/// `local-invariant`: abstract value for one local.
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct LocalInvariant {
    #[serde(rename = "local-index")]
    pub local_index: u32,
    pub value: AbstractValue,
}

/// `abstract-value`: a `kind`-tagged union (exactly one case applies).
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", deny_unknown_fields)]
pub enum AbstractValue {
    #[serde(rename = "i32-interval")]
    I32Interval { interval: Interval },
    #[serde(rename = "i64-interval")]
    I64Interval { interval: Interval },
    #[serde(rename = "region-pointer")]
    RegionPointer { region: Region },
    #[serde(rename = "unknown")]
    Unknown,
}

/// `interval`: inclusive integer range `[lo, hi]` over s64.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Interval {
    pub lo: i64,
    pub hi: i64,
}

/// `region-pointer-payload`: a region id plus a byte-offset interval into it.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Region {
    #[serde(rename = "region-id")]
    pub region_id: u32,
    pub offset: Interval,
}

/// `call-edge`: one resolved call site (FEAT-006).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CallEdge {
    #[serde(rename = "caller-func")]
    pub caller_func: u32,
    pub pc: u32,
    /// `true` for `call_indirect`, `false` for a direct call.
    pub indirect: bool,
    /// Resolved target function indices. A singleton **sound** set permits
    /// devirtualization; empty means provably unreachable.
    #[serde(rename = "resolved-targets")]
    pub resolved_targets: Vec<u32>,
    pub soundness: Soundness,
}

/// `soundness-tag`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Deserialize)]
pub enum Soundness {
    /// A sound (possibly over-approximating) cover of every reachable target.
    #[serde(rename = "sound")]
    Sound,
    /// Conservative fallback for constructs the analyzer cannot resolve soundly.
    #[serde(rename = "unsound-fallback")]
    UnsoundFallback,
}

/// `function-summary`: compositional per-function summary (FEAT-007).
#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FunctionSummary {
    #[serde(rename = "func-index")]
    pub func_index: u32,
    #[serde(rename = "param-count")]
    pub param_count: u32,
    #[serde(rename = "result-summary")]
    pub result_summary: Vec<AbstractValue>,
    #[serde(rename = "context-sensitive")]
    pub context_sensitive: bool,
    pub recursive: bool,
}

impl AbstractValue {
    /// Constant-fold precondition: a singleton 32-bit interval (`lo == hi`)
    /// whose value fits in i32. Returns the proven constant, or `None`.
    pub fn as_singleton_i32(&self) -> Option<i32> {
        match self {
            AbstractValue::I32Interval { interval } if interval.lo == interval.hi => {
                i32::try_from(interval.lo).ok()
            }
            _ => None,
        }
    }

    /// Constant-fold precondition for i64: a singleton 64-bit interval.
    pub fn as_singleton_i64(&self) -> Option<i64> {
        match self {
            AbstractValue::I64Interval { interval } if interval.lo == interval.hi => {
                Some(interval.lo)
            }
            _ => None,
        }
    }

    /// Bounds-check-elision precondition: the value is a region pointer.
    /// The in-region check (offset interval within the region's size) is
    /// applied by the transform, which has the memory-size context.
    pub fn as_region_pointer(&self) -> Option<&Region> {
        match self {
            AbstractValue::RegionPointer { region } => Some(region),
            _ => None,
        }
    }
}

impl CallEdge {
    /// Devirtualization precondition: an indirect call whose resolved target
    /// set is a sound singleton. Returns the unique target, or `None`.
    pub fn devirt_target(&self) -> Option<u32> {
        if self.indirect && self.soundness == Soundness::Sound && self.resolved_targets.len() == 1 {
            Some(self.resolved_targets[0])
        } else {
            None
        }
    }
}

/// Parse and validate a scry invariant document.
///
/// Validation gates (see module docs): the structural shape is enforced by
/// `serde(deny_unknown_fields)` + required fields (mirroring the schema's
/// `additionalProperties: false`), and the `invariants.schema` URI must equal
/// [`SCHEMA_ID`]. Returns an error — never a partially-trusted result — on any
/// mismatch.
pub fn parse_and_validate(json: &str) -> Result<AnalysisResult> {
    let result: AnalysisResult = serde_json::from_str(json)
        .map_err(|e| anyhow!("scry invariant document does not match the v1 contract: {e}"))?;
    if result.invariants.schema != SCHEMA_ID {
        bail!(
            "scry invariant document declares schema {:?}, expected {:?} — refusing to act on an \
             unvetted contract version",
            result.invariants.schema,
            SCHEMA_ID
        );
    }
    Ok(result)
}

/// Query index over a validated [`AnalysisResult`], keyed by `(func-index, pc)`
/// so a pass can look up the invariants at the instruction it is rewriting in
/// O(1). Borrows the result; build it once per module.
pub struct InvariantIndex<'a> {
    /// `(func-index, pc)` → that point's local invariants.
    points: HashMap<(u32, u32), &'a [LocalInvariant]>,
    /// `(caller-func, pc)` → the resolved call edge at that site.
    call_edges: HashMap<(u32, u32), &'a CallEdge>,
}

impl<'a> InvariantIndex<'a> {
    /// Build the index. Later duplicate `(func, pc)` keys overwrite earlier
    /// ones (the contract says point order is not significant; scry emits one
    /// entry per point).
    pub fn build(result: &'a AnalysisResult) -> Self {
        let mut points = HashMap::new();
        for p in &result.invariants.points {
            points.insert((p.func_index, p.pc), p.locals.as_slice());
        }
        let mut call_edges = HashMap::new();
        for e in &result.call_graph {
            call_edges.insert((e.caller_func, e.pc), e);
        }
        Self { points, call_edges }
    }

    /// The abstract value proven for `local` at `(func, pc)`, if any.
    pub fn local_value(&self, func: u32, pc: u32, local: u32) -> Option<&AbstractValue> {
        self.points
            .get(&(func, pc))?
            .iter()
            .find(|li| li.local_index == local)
            .map(|li| &li.value)
    }

    /// The resolved call edge at `(func, pc)`, if any.
    pub fn call_edge(&self, func: u32, pc: u32) -> Option<&CallEdge> {
        self.call_edges.get(&(func, pc)).copied()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal but schema-conformant document exercising all three
    /// transform preconditions.
    fn sample() -> String {
        r#"{
          "invariants": {
            "schema": "https://pulseengine.eu/scry-invariants/v1",
            "module-sha256": "0000000000000000000000000000000000000000000000000000000000000000",
            "points": [
              { "func-index": 2, "pc": 17, "locals": [
                  { "local-index": 0, "value": { "kind": "i32-interval", "interval": { "lo": 42, "hi": 42 } } },
                  { "local-index": 1, "value": { "kind": "i32-interval", "interval": { "lo": 0, "hi": 99 } } },
                  { "local-index": 2, "value": { "kind": "region-pointer", "region": { "region-id": 0, "offset": { "lo": 0, "hi": 15 } } } }
              ] }
            ]
          },
          "call-graph": [
            { "caller-func": 2, "pc": 30, "indirect": true, "resolved-targets": [7], "soundness": "sound" },
            { "caller-func": 2, "pc": 40, "indirect": true, "resolved-targets": [8, 9], "soundness": "sound" },
            { "caller-func": 2, "pc": 50, "indirect": true, "resolved-targets": [11], "soundness": "unsound-fallback" }
          ],
          "function-summaries": []
        }"#
        .to_string()
    }

    #[test]
    fn parses_and_indexes_a_valid_bundle() {
        let r = parse_and_validate(&sample()).expect("valid bundle must parse");
        let idx = InvariantIndex::build(&r);
        // Singleton i32 interval → constant-fold precondition holds for local 0.
        let v0 = idx.local_value(2, 17, 0).unwrap();
        assert_eq!(v0.as_singleton_i32(), Some(42));
        // Non-singleton interval → no constant.
        let v1 = idx.local_value(2, 17, 1).unwrap();
        assert_eq!(v1.as_singleton_i32(), None);
        // Region pointer → bounds-elision precondition surfaces the region.
        let v2 = idx.local_value(2, 17, 2).unwrap();
        assert_eq!(v2.as_region_pointer().map(|r| r.region_id), Some(0));
        // Missing point.
        assert!(idx.local_value(2, 99, 0).is_none());
    }

    #[test]
    fn devirtualization_precondition() {
        let r = parse_and_validate(&sample()).unwrap();
        let idx = InvariantIndex::build(&r);
        // Sound singleton → devirtualizable.
        assert_eq!(idx.call_edge(2, 30).unwrap().devirt_target(), Some(7));
        // Sound but two targets → not devirtualizable.
        assert_eq!(idx.call_edge(2, 40).unwrap().devirt_target(), None);
        // Singleton but unsound fallback → MUST NOT devirtualize.
        assert_eq!(idx.call_edge(2, 50).unwrap().devirt_target(), None);
    }

    #[test]
    fn rejects_wrong_schema_id() {
        let bad = sample().replace("scry-invariants/v1", "scry-invariants/v2");
        assert!(parse_and_validate(&bad).is_err());
    }

    #[test]
    fn rejects_unknown_field() {
        // additionalProperties: false — an extra field is a contract violation.
        let bad = sample().replace(r#""pc": 17,"#, r#""pc": 17, "bogus-field": 1,"#);
        assert!(parse_and_validate(&bad).is_err());
    }

    #[test]
    fn rejects_unknown_abstract_value_kind() {
        let bad = sample().replace(r#""kind": "i32-interval""#, r#""kind": "i128-interval""#);
        assert!(parse_and_validate(&bad).is_err());
    }
}
