/// WebAssembly value types - core to LOOM's type system
///
/// This module defines the fundamental value types that LOOM operates on.
/// By translating this to Rocq, we can prove properties about type safety.

/// WebAssembly value types (from loom-core/src/stack.rs)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValueType {
    I32,
    I64,
    F32,
    F64,
}

impl ValueType {
    /// Returns the size of the value type in bytes
    pub fn size_bytes(&self) -> u32 {
        match self {
            ValueType::I32 => 4,
            ValueType::I64 => 8,
            ValueType::F32 => 4,
            ValueType::F64 => 8,
        }
    }

    /// Returns true if this is an integer type
    pub fn is_integer(&self) -> bool {
        match self {
            ValueType::I32 | ValueType::I64 => true,
            ValueType::F32 | ValueType::F64 => false,
        }
    }

    /// Returns true if this is a floating-point type
    pub fn is_float(&self) -> bool {
        !self.is_integer()
    }

    /// Returns true if this type has the same representation as another
    pub fn same_repr(&self, other: &ValueType) -> bool {
        self.size_bytes() == other.size_bytes()
    }
}

/// Stack signature representing inputs and outputs of an instruction
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StackSignature {
    pub inputs: Vec<ValueType>,
    pub outputs: Vec<ValueType>,
}

impl StackSignature {
    /// Creates a new empty signature
    pub fn empty() -> Self {
        StackSignature {
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Creates a signature that consumes inputs and produces outputs
    pub fn new(inputs: Vec<ValueType>, outputs: Vec<ValueType>) -> Self {
        StackSignature { inputs, outputs }
    }

    /// Returns the net stack effect (positive = pushes, negative = pops)
    pub fn net_effect(&self) -> i32 {
        self.outputs.len() as i32 - self.inputs.len() as i32
    }
}
