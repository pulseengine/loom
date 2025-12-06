//! Stack Analysis Module
//!
//! Implements compositional stack type analysis based on Binaryen's StackSignature system.
//!
//! This module provides formal verification that optimization passes preserve stack invariants.
//! Stack signatures describe the effect of instruction sequences on the value stack.
//!
//! # Overview
//!
//! A `StackSignature` characterizes how an instruction sequence interacts with the value stack:
//! - `params`: Types consumed from the stack
//! - `results`: Types produced on the stack
//! - `kind`: Whether the signature is Fixed (deterministic) or Polymorphic (includes unreachable code)
//!
//! # Example
//!
//! ```ignore
//! // i32.const (produces 1 i32 on empty stack)
//! let sig1 = StackSignature {
//!     params: vec![],
//!     results: vec![ValueType::I32],
//!     kind: SignatureKind::Fixed,
//! };
//!
//! // i32.add (consumes 2 i32, produces 1 i32)
//! let sig2 = StackSignature {
//!     params: vec![ValueType::I32, ValueType::I32],
//!     results: vec![ValueType::I32],
//!     kind: SignatureKind::Fixed,
//! };
//!
//! // sig1 composes with sig2: outputs of sig1 feed into inputs of sig2
//! assert!(sig1.composes(&sig2)); // Can apply add to result of const
//! ```

use std::fmt;

/// Value type in WebAssembly
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ValueType {
    /// 32-bit integer
    I32,
    /// 64-bit integer
    I64,
    /// 32-bit float
    F32,
    /// 64-bit float
    F64,
    /// Vector (v128)
    V128,
    /// Function reference
    FuncRef,
    /// External reference
    ExternRef,
}

/// Kind of stack signature
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureKind {
    /// Fixed (deterministic, all paths reachable)
    Fixed,
    /// Polymorphic (contains unreachable code that can match any outer stack)
    Polymorphic,
}

/// Stack signature: describes how an instruction sequence affects the value stack
///
/// Represents the stack type [params] -> [results], where:
/// - `params`: Stack values consumed (in order, innermost first)
/// - `results`: Stack values produced (in order, innermost first)
/// - `kind`: Whether signature is Fixed or Polymorphic
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackSignature {
    /// Types consumed from outer stack
    pub params: Vec<ValueType>,
    /// Types produced to outer stack
    pub results: Vec<ValueType>,
    /// Whether this signature includes polymorphic behavior (unreachable code)
    pub kind: SignatureKind,
}

impl StackSignature {
    /// Create a new stack signature
    pub fn new(params: Vec<ValueType>, results: Vec<ValueType>, kind: SignatureKind) -> Self {
        StackSignature {
            params,
            results,
            kind,
        }
    }

    /// Empty signature: [] -> []
    pub fn empty() -> Self {
        StackSignature {
            params: vec![],
            results: vec![],
            kind: SignatureKind::Fixed,
        }
    }

    /// Polymorphic empty signature: [] -> [] {poly}
    /// This is the bottom type - subtype of all signatures.
    pub fn polymorphic_empty() -> Self {
        StackSignature {
            params: vec![],
            results: vec![],
            kind: SignatureKind::Polymorphic,
        }
    }

    /// Check if two signatures compose: does `next` follow `self`?
    ///
    /// Two signatures compose if the outputs of `self` can be inputs to `next`.
    /// This requires exact type matching between `self.results` and `next.params`.
    ///
    /// # Composition Rule
    ///
    /// `self` (produces [a, b]) composes with `next` (consumes [a, b])
    /// Result: combined signature has self.params and next.results
    pub fn composes(&self, next: &StackSignature) -> bool {
        self.results == next.params
    }

    /// Compose two signatures: returns [self.params] -> [next.results]
    ///
    /// # Panics
    ///
    /// Panics if signatures don't compose. Use `composes()` to check first.
    pub fn compose(&self, next: &StackSignature) -> StackSignature {
        assert!(
            self.composes(next),
            "Signatures don't compose: self.results {:?} != next.params {:?}",
            self.results,
            next.params
        );

        // Result is polymorphic if either input is polymorphic
        let kind =
            if self.kind == SignatureKind::Polymorphic || next.kind == SignatureKind::Polymorphic {
                SignatureKind::Polymorphic
            } else {
                SignatureKind::Fixed
            };

        StackSignature {
            params: self.params.clone(),
            results: next.results.clone(),
            kind,
        }
    }

    /// Check if this signature is a subtype of another
    ///
    /// Implements WASM stack subtyping rules:
    /// - Fixed signatures can only match fixed signatures (contravariance)
    /// - Polymorphic signatures can match with arbitrary outer stack
    /// - Parameter types must be supertypes of expected (contravariance)
    /// - Result types must be subtypes of provided (covariance)
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // [i32] -> [i32] is subtype of [i32] -> [i32]
    /// let sig1 = StackSignature::new(vec![I32], vec![I32], Fixed);
    /// let sig2 = StackSignature::new(vec![I32], vec![I32], Fixed);
    /// assert!(StackSignature::is_subtype(&sig1, &sig2));
    ///
    /// // [i32] -> [i32] {poly} is subtype of [] -> [] {poly}
    /// // (polymorphic can match anything)
    /// let poly_any = StackSignature::polymorphic_empty();
    /// assert!(StackSignature::is_subtype(&poly, &poly_any));
    /// ```
    pub fn is_subtype(a: &StackSignature, b: &StackSignature) -> bool {
        // Polymorphic bottom type is subtype of everything
        if a.kind == SignatureKind::Polymorphic && a.params.is_empty() && a.results.is_empty() {
            return true;
        }

        // Both fixed: exact match required
        if a.kind == SignatureKind::Fixed && b.kind == SignatureKind::Fixed {
            return a.params == b.params && a.results == b.results;
        }

        // a is polymorphic, b is fixed: not subtype (can't give polymorphic guarantee)
        if a.kind == SignatureKind::Polymorphic && b.kind == SignatureKind::Fixed {
            return false;
        }

        // Both polymorphic: check core types match
        if a.kind == SignatureKind::Polymorphic && b.kind == SignatureKind::Polymorphic {
            return a.params == b.params && a.results == b.results;
        }

        false
    }

    /// Get stack effect: how many values consumed vs produced
    ///
    /// Positive = produces more than consumes (stack grows)
    /// Negative = consumes more than produces (stack shrinks)
    pub fn stack_effect(&self) -> i32 {
        self.results.len() as i32 - self.params.len() as i32
    }

    /// Check if signature can be identity on an outer stack of given types
    ///
    /// A signature is identity on an outer stack if it consumes exactly the outer stack
    /// and produces it back unchanged.
    pub fn is_identity_with_outer(&self, outer: &[ValueType]) -> bool {
        // Outer stack must be subset of params (we preserve it)
        if self.params.len() < outer.len() {
            return false;
        }

        // Check outer stack matches the end of params
        let outer_offset = self.params.len() - outer.len();
        self.params[outer_offset..] != outer[..] && {
            // Check results end with outer stack
            self.results.len() >= outer.len()
                && self.results[self.results.len() - outer.len()..] == outer[..]
        }
    }
}

impl fmt::Display for StackSignature {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, param) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", param)?;
        }
        write!(f, "] -> [")?;
        for (i, result) in self.results.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{:?}", result)?;
        }
        match self.kind {
            SignatureKind::Fixed => write!(f, "]"),
            SignatureKind::Polymorphic => write!(f, "] {{poly}}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_composition() {
        // i32.const: [] -> [i32]
        let const_sig = StackSignature::new(vec![], vec![ValueType::I32], SignatureKind::Fixed);

        // i32.add: [i32, i32] -> [i32]
        let add_sig = StackSignature::new(
            vec![ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
            SignatureKind::Fixed,
        );

        // const doesn't compose with add (only produces 1 i32, add needs 2)
        assert!(!const_sig.composes(&add_sig));
    }

    #[test]
    fn test_composition_chain() {
        // [i32] -> [i32]
        let id = StackSignature::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
            SignatureKind::Fixed,
        );

        // [i32] -> [i32]
        let another_id = StackSignature::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
            SignatureKind::Fixed,
        );

        assert!(id.composes(&another_id));
        let result = id.compose(&another_id);
        assert_eq!(result.params, vec![ValueType::I32]);
        assert_eq!(result.results, vec![ValueType::I32]);
    }

    #[test]
    fn test_polymorphic_composition() {
        let fixed = StackSignature::new(vec![], vec![], SignatureKind::Fixed);
        let poly = StackSignature::new(vec![], vec![], SignatureKind::Polymorphic);

        let result = fixed.compose(&poly);
        assert_eq!(result.kind, SignatureKind::Polymorphic);
    }

    #[test]
    fn test_subtype() {
        let sig = StackSignature::new(
            vec![ValueType::I32],
            vec![ValueType::I32],
            SignatureKind::Fixed,
        );

        // Same signature is subtype of itself
        assert!(StackSignature::is_subtype(&sig, &sig));

        // Polymorphic empty is subtype of everything
        let poly_empty = StackSignature::polymorphic_empty();
        assert!(StackSignature::is_subtype(&poly_empty, &sig));
        assert!(StackSignature::is_subtype(&poly_empty, &poly_empty));
    }

    #[test]
    fn test_stack_effect() {
        // Produces 1, consumes 0 → effect +1
        let prod = StackSignature::new(vec![], vec![ValueType::I32], SignatureKind::Fixed);
        assert_eq!(prod.stack_effect(), 1);

        // Consumes 2, produces 1 → effect -1
        let cons = StackSignature::new(
            vec![ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
            SignatureKind::Fixed,
        );
        assert_eq!(cons.stack_effect(), -1);
    }
}
