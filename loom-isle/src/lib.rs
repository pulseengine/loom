//! LOOM ISLE Definitions
//!
//! This crate re-exports ISLE term definitions from loom-shared for backward compatibility.
//! The canonical ISLE definitions are now in loom-shared, which serves as the shared foundation
//! for both Loom (open-source, ASIL A/B) and Synth (commercial, ASIL D).
//!
//! All ISLE types and functions are re-exported from loom-shared.

#![allow(dead_code)]
#![allow(unused_variables)]

// Re-export all types from loom-shared
pub use loom_shared::*;
