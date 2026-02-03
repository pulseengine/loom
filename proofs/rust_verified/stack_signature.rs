/// Stack Signature Core - Subset for Formal Verification
///
/// This module contains the core stack signature types and composition logic
/// from loom-core/src/stack.rs, simplified for translation to Rocq.
///
/// Key properties to verify:
/// 1. compose is associative: (a.compose(b)).compose(c) = a.compose(b.compose(c))
/// 2. empty is identity: empty.compose(a) = a = a.compose(empty)
/// 3. composes is transitive under compose
/// 4. is_subtype is reflexive and transitive

/// Value type in WebAssembly
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    I32,
    I64,
    F32,
    F64,
}

/// Kind of stack signature
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureKind {
    /// Fixed (deterministic, all paths reachable)
    Fixed,
    /// Polymorphic (contains unreachable code)
    Polymorphic,
}

/// Stack signature: [params] -> [results]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackSignature {
    pub params: Vec<ValueType>,
    pub results: Vec<ValueType>,
    pub kind: SignatureKind,
}

impl StackSignature {
    /// Create a new stack signature
    pub fn new(params: Vec<ValueType>, results: Vec<ValueType>, kind: SignatureKind) -> Self {
        StackSignature { params, results, kind }
    }

    /// Empty signature: [] -> []
    pub fn empty() -> Self {
        StackSignature {
            params: Vec::new(),
            results: Vec::new(),
            kind: SignatureKind::Fixed,
        }
    }

    /// Polymorphic empty: [] -> [] {poly}
    pub fn polymorphic_empty() -> Self {
        StackSignature {
            params: Vec::new(),
            results: Vec::new(),
            kind: SignatureKind::Polymorphic,
        }
    }

    /// Check if two signatures compose: does `next` follow `self`?
    pub fn composes(&self, next: &StackSignature) -> bool {
        self.results == next.params
    }

    /// Get stack effect: positive = grows, negative = shrinks
    pub fn stack_effect(&self) -> i32 {
        self.results.len() as i32 - self.params.len() as i32
    }

    /// Check if signature is the identity (no-op)
    pub fn is_identity(&self) -> bool {
        self.params == self.results && self.kind == SignatureKind::Fixed
    }

    /// Check if this is the polymorphic bottom type
    pub fn is_polymorphic_bottom(&self) -> bool {
        self.params.is_empty() && self.results.is_empty()
            && self.kind == SignatureKind::Polymorphic
    }
}

/// Compose two signatures: [self.params] -> [next.results]
/// Returns None if signatures don't compose
pub fn compose(a: &StackSignature, b: &StackSignature) -> Option<StackSignature> {
    if !a.composes(b) {
        return None;
    }

    let kind = if a.kind == SignatureKind::Polymorphic || b.kind == SignatureKind::Polymorphic {
        SignatureKind::Polymorphic
    } else {
        SignatureKind::Fixed
    };

    Some(StackSignature {
        params: a.params.clone(),
        results: b.results.clone(),
        kind,
    })
}

/// Check if `a` is a subtype of `b`
pub fn is_subtype(a: &StackSignature, b: &StackSignature) -> bool {
    // Polymorphic bottom is subtype of everything
    if a.is_polymorphic_bottom() {
        return true;
    }

    // Both fixed: exact match required
    if a.kind == SignatureKind::Fixed && b.kind == SignatureKind::Fixed {
        return a.params == b.params && a.results == b.results;
    }

    // Polymorphic can't be subtype of fixed
    if a.kind == SignatureKind::Polymorphic && b.kind == SignatureKind::Fixed {
        return false;
    }

    // Both polymorphic: check types match
    if a.kind == SignatureKind::Polymorphic && b.kind == SignatureKind::Polymorphic {
        return a.params == b.params && a.results == b.results;
    }

    // Fixed is subtype of polymorphic with same types
    if a.kind == SignatureKind::Fixed && b.kind == SignatureKind::Polymorphic {
        return a.params == b.params && a.results == b.results;
    }

    false
}

/// Example instruction signatures for common WASM instructions
pub mod instructions {
    use super::*;

    /// i32.const: [] -> [i32]
    pub fn i32_const() -> StackSignature {
        StackSignature::new(vec![], vec![ValueType::I32], SignatureKind::Fixed)
    }

    /// i32.add: [i32, i32] -> [i32]
    pub fn i32_add() -> StackSignature {
        StackSignature::new(
            vec![ValueType::I32, ValueType::I32],
            vec![ValueType::I32],
            SignatureKind::Fixed,
        )
    }

    /// drop: [T] -> [] (simplified as i32 for this example)
    pub fn drop_i32() -> StackSignature {
        StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Fixed)
    }

    /// nop: [] -> []
    pub fn nop() -> StackSignature {
        StackSignature::empty()
    }

    /// unreachable: [] -> [] {poly}
    pub fn unreachable() -> StackSignature {
        StackSignature::polymorphic_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compose_identity() {
        let sig = StackSignature::new(
            vec![ValueType::I32],
            vec![ValueType::I64],
            SignatureKind::Fixed,
        );
        let empty = StackSignature::empty();

        // empty.compose(sig) should fail (results don't match params)
        // But sig with matching results/params works
        let id = StackSignature::new(
            vec![ValueType::I64],
            vec![ValueType::I64],
            SignatureKind::Fixed,
        );

        let composed = compose(&sig, &id);
        assert!(composed.is_some());
        let result = composed.unwrap();
        assert_eq!(result.params, vec![ValueType::I32]);
        assert_eq!(result.results, vec![ValueType::I64]);
    }

    #[test]
    fn test_subtype_reflexive() {
        let sig = StackSignature::new(
            vec![ValueType::I32],
            vec![ValueType::I64],
            SignatureKind::Fixed,
        );
        assert!(is_subtype(&sig, &sig));
    }
}
