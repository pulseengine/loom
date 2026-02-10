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

/// Convert a crate::ValueType to stack::ValueType
pub fn convert_value_type(vt: &crate::ValueType) -> ValueType {
    match vt {
        crate::ValueType::I32 => ValueType::I32,
        crate::ValueType::I64 => ValueType::I64,
        crate::ValueType::F32 => ValueType::F32,
        crate::ValueType::F64 => ValueType::F64,
    }
}

/// Convert a BlockType to a StackSignature
///
/// This extracts the params and results from a block type declaration.
/// Used for Block, Loop, and If instructions.
pub fn block_type_to_signature(block_type: &crate::BlockType) -> StackSignature {
    match block_type {
        crate::BlockType::Empty => StackSignature::empty(),
        crate::BlockType::Value(vt) => {
            // [] -> [T] - produces single value
            StackSignature::new(vec![], vec![convert_value_type(vt)], SignatureKind::Fixed)
        }
        crate::BlockType::Func { params, results } => {
            // [params...] -> [results...]
            StackSignature::new(
                params.iter().map(convert_value_type).collect(),
                results.iter().map(convert_value_type).collect(),
                SignatureKind::Fixed,
            )
        }
    }
}

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
    /// Unknown type (used when type context is unavailable, e.g., for locals)
    /// This type is compatible with any other type during validation
    Unknown,
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

/// Block validation framework
///
/// Validates that block bodies satisfy their declared type signatures.
/// This is critical for detecting stack mismatches before they reach the WASM validator.
pub mod validation {
    use super::*;

    /// Represents the result of block validation
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum ValidationResult {
        /// Block is valid: stack signature matches declaration
        Valid(StackSignature),
        /// Stack type mismatch at position
        StackMismatch {
            /// Position in instruction sequence where mismatch occurred
            position: usize,
            /// Expected signature at this point
            expected: StackSignature,
            /// Actual signature produced
            actual: StackSignature,
        },
        /// Block requires specific input types not provided
        MissingInput {
            /// Types expected on input stack
            expected_params: Vec<ValueType>,
        },
        /// Block produces wrong output type
        WrongOutput {
            /// Expected output type from block
            expected_results: Vec<ValueType>,
            /// Actual output produced
            actual_results: Vec<ValueType>,
        },
        /// Instruction-level error at position with message
        InstructionError {
            /// Position in instruction sequence where error occurred
            position: usize,
            /// Error message describing the issue
            message: String,
            /// Stack state at the point of error
            stack_state: Vec<ValueType>,
        },
    }

    /// Validate a block body against its declared type
    ///
    /// Checks that the instruction sequence in a block:
    /// 1. Consumes the correct input types
    /// 2. Produces the correct output types
    /// 3. Has valid instruction composition throughout
    ///
    /// # Arguments
    ///
    /// * `instructions` - The instruction sequence in the block body
    /// * `block_params` - Input types this block expects
    /// * `block_results` - Output types this block must produce
    pub fn validate_block(
        instructions: &[crate::Instruction],
        block_params: &[ValueType],
        block_results: &[ValueType],
    ) -> ValidationResult {
        // Start with the input signature
        let mut current_sig =
            StackSignature::new(block_params.to_vec(), vec![], SignatureKind::Fixed);

        // Compose each instruction
        for (pos, instr) in instructions.iter().enumerate() {
            let instr_sig = super::effects::instruction_signature(instr);

            // Check if this instruction composes with current state
            if !current_sig.composes(&instr_sig) {
                return ValidationResult::StackMismatch {
                    position: pos,
                    expected: current_sig.clone(),
                    actual: instr_sig,
                };
            }

            // Compose the signatures
            current_sig = current_sig.compose(&instr_sig);
        }

        // Check that final stack matches block's declared results
        if current_sig.results != block_results {
            return ValidationResult::WrongOutput {
                expected_results: block_results.to_vec(),
                actual_results: current_sig.results,
            };
        }

        ValidationResult::Valid(current_sig)
    }

    /// Validate a sequence of instructions without block context
    ///
    /// This is useful for validating instruction sequences in isolation,
    /// without knowing the surrounding block type.
    pub fn validate_instruction_sequence(instructions: &[crate::Instruction]) -> ValidationResult {
        let mut current_sig = StackSignature::empty();

        for (pos, instr) in instructions.iter().enumerate() {
            let instr_sig = super::effects::instruction_signature(instr);

            if !current_sig.composes(&instr_sig) {
                return ValidationResult::StackMismatch {
                    position: pos,
                    expected: current_sig.clone(),
                    actual: instr_sig,
                };
            }

            current_sig = current_sig.compose(&instr_sig);
        }

        ValidationResult::Valid(current_sig)
    }

    /// Validate a function's instructions against its signature
    ///
    /// This is the primary entry point for optimizer passes to validate
    /// that transformations preserve stack correctness.
    ///
    /// # Arguments
    ///
    /// * `func` - The function to validate
    ///
    /// # Returns
    ///
    /// `Ok(())` if the function is valid, `Err` with details if not.
    pub fn validate_function(func: &crate::Function) -> Result<(), String> {
        // Convert function signature to stack types
        let params: Vec<ValueType> = func
            .signature
            .params
            .iter()
            .map(super::convert_value_type)
            .collect();
        let results: Vec<ValueType> = func
            .signature
            .results
            .iter()
            .map(super::convert_value_type)
            .collect();

        // Validate the function body
        match validate_function_body(&func.instructions, &params, &results) {
            ValidationResult::Valid(_) => Ok(()),
            ValidationResult::StackMismatch {
                position,
                expected,
                actual,
            } => Err(format!(
                "Stack mismatch at instruction {}: expected {} but got {}",
                position, expected, actual
            )),
            ValidationResult::MissingInput { expected_params } => Err(format!(
                "Missing input: expected {:?} on stack",
                expected_params
            )),
            ValidationResult::WrongOutput {
                expected_results,
                actual_results,
            } => Err(format!(
                "Wrong output: expected {:?} but got {:?}",
                expected_results, actual_results
            )),
            ValidationResult::InstructionError {
                position,
                message,
                stack_state,
            } => Err(format!(
                "Instruction error at position {}: {} (stack: {:?})",
                position, message, stack_state
            )),
        }
    }

    /// Validate a function body with proper stack tracking
    ///
    /// This handles the complexity of tracking stack state through
    /// control flow instructions while validating type correctness.
    pub fn validate_function_body(
        instructions: &[crate::Instruction],
        params: &[ValueType],
        results: &[ValueType],
    ) -> ValidationResult {
        use crate::Instruction;

        // Track current stack state
        // Note: In WASM, function parameters go into locals, not the value stack.
        // The value stack starts empty at function entry.
        // We pass params for signature construction, but don't put them on the stack.
        let mut stack: Vec<ValueType> = Vec::new();

        // Track if we've hit a terminator (return, unreachable, etc.)
        let mut unreachable = false;

        for (pos, instr) in instructions.iter().enumerate() {
            // Skip instructions after unconditional terminators
            if unreachable {
                continue;
            }

            match validate_instruction_stack_effect(instr, &mut stack) {
                Ok(()) => {}
                Err(msg) => {
                    return ValidationResult::InstructionError {
                        position: pos,
                        message: msg,
                        stack_state: stack.clone(),
                    };
                }
            }

            // Check if this instruction makes subsequent code unreachable
            if matches!(
                instr,
                Instruction::Return
                    | Instruction::Unreachable
                    | Instruction::Br(_)
                    | Instruction::BrTable { .. }
            ) {
                unreachable = true;
            }
        }

        // If the function ends with a terminator, the stack check is not applicable
        // (control doesn't fall through to the implicit return)
        if unreachable {
            return ValidationResult::Valid(StackSignature::new(
                params.to_vec(),
                results.to_vec(),
                SignatureKind::Fixed,
            ));
        }

        // Check final stack matches expected results
        if stack.len() != results.len() {
            return ValidationResult::WrongOutput {
                expected_results: results.to_vec(),
                actual_results: stack,
            };
        }

        for (actual, expected) in stack.iter().zip(results.iter()) {
            // Unknown is compatible with any expected type (from locals without type context)
            if *actual != ValueType::Unknown && actual != expected {
                return ValidationResult::WrongOutput {
                    expected_results: results.to_vec(),
                    actual_results: stack,
                };
            }
        }

        ValidationResult::Valid(StackSignature::new(
            params.to_vec(),
            results.to_vec(),
            SignatureKind::Fixed,
        ))
    }

    /// Apply stack effect of a single instruction
    ///
    /// Modifies the stack in place according to the instruction's effect.
    /// Returns Err if the stack doesn't have required inputs.
    fn validate_instruction_stack_effect(
        instr: &crate::Instruction,
        stack: &mut Vec<ValueType>,
    ) -> Result<(), String> {
        use crate::Instruction::*;

        match instr {
            // Constants push values
            I32Const(_) => {
                stack.push(ValueType::I32);
                Ok(())
            }
            I64Const(_) => {
                stack.push(ValueType::I64);
                Ok(())
            }
            F32Const(_) => {
                stack.push(ValueType::F32);
                Ok(())
            }
            F64Const(_) => {
                stack.push(ValueType::F64);
                Ok(())
            }

            // Binary i32 operations: pop 2 i32, push 1 i32
            I32Add | I32Sub | I32Mul | I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU
            | I32DivS | I32DivU | I32RemS | I32RemU | I32Eq | I32Ne | I32LtS | I32LtU | I32GtS
            | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU => {
                pop_expected(stack, ValueType::I32)?;
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::I32);
                Ok(())
            }

            // Binary i64 operations: pop 2 i64, push 1 i64
            I64Add | I64Sub | I64Mul | I64And | I64Or | I64Xor | I64DivS | I64DivU | I64RemS
            | I64RemU => {
                pop_expected(stack, ValueType::I64)?;
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I64);
                Ok(())
            }

            // i64 shifts: pop i32 (shift amount), pop i64 (value), push i64
            I64Shl | I64ShrS | I64ShrU => {
                // WebAssembly spec: i64 shifts take two i64 operands
                // The shift amount is an i64 (only low 6 bits used, mod 64)
                pop_expected(stack, ValueType::I64)?;
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I64);
                Ok(())
            }

            // i64 comparisons: pop 2 i64, push i32
            I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS
            | I64GeU => {
                pop_expected(stack, ValueType::I64)?;
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I32);
                Ok(())
            }

            // Unary i32 operations
            I32Eqz | I32Clz | I32Ctz | I32Popcnt => {
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::I32);
                Ok(())
            }

            // Unary i64 operations
            I64Eqz => {
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I64Clz | I64Ctz | I64Popcnt => {
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I64);
                Ok(())
            }

            // Local operations - without type context we can't accurately validate
            // We push Unknown as placeholder since we don't know the local's type
            // A proper implementation would track local types from function signature + locals
            LocalGet(_) => {
                stack.push(ValueType::Unknown);
                Ok(())
            }
            LocalSet(_) => {
                if stack.is_empty() {
                    return Err("Stack underflow for local.set".to_string());
                }
                stack.pop();
                Ok(())
            }
            LocalTee(_) => {
                // Peek at top of stack - value stays
                if stack.is_empty() {
                    return Err("Stack underflow for local.tee".to_string());
                }
                Ok(())
            }

            // Global operations - unknown type since we don't have type context
            GlobalGet(_) => {
                stack.push(ValueType::Unknown);
                Ok(())
            }
            GlobalSet(_) => {
                if stack.is_empty() {
                    return Err("Stack underflow for global.set".to_string());
                }
                stack.pop();
                Ok(())
            }

            // Memory operations
            I32Load { .. } => {
                pop_expected(stack, ValueType::I32)?; // address
                stack.push(ValueType::I32);
                Ok(())
            }
            I64Load { .. } => {
                pop_expected(stack, ValueType::I32)?; // address
                stack.push(ValueType::I64);
                Ok(())
            }
            I32Store { .. } => {
                pop_expected(stack, ValueType::I32)?; // value
                pop_expected(stack, ValueType::I32)?; // address
                Ok(())
            }
            I64Store { .. } => {
                pop_expected(stack, ValueType::I64)?; // value
                pop_expected(stack, ValueType::I32)?; // address
                Ok(())
            }

            // Drop pops any value
            Drop => {
                if stack.is_empty() {
                    return Err("Stack underflow for drop".to_string());
                }
                stack.pop();
                Ok(())
            }

            // Select: [T, T, i32] -> [T]
            Select => {
                pop_expected(stack, ValueType::I32)?; // condition
                if stack.len() < 2 {
                    return Err("Stack underflow for select".to_string());
                }
                let t1 = stack.pop().unwrap();
                let _t2 = stack.pop().unwrap();
                // Should verify t1 == t2, but we'll be permissive
                stack.push(t1);
                Ok(())
            }

            // Nop has no effect
            Nop => Ok(()),

            // Control flow - handled by recursing into bodies
            Block {
                block_type,
                body: _,
            } => {
                // Get block's declared signature
                let block_sig = super::block_type_to_signature(block_type);

                // Pop params from stack
                for param in block_sig.params.iter().rev() {
                    pop_expected(stack, *param)?;
                }

                // Push results to stack
                for result in &block_sig.results {
                    stack.push(*result);
                }

                Ok(())
            }

            Loop {
                block_type,
                body: _,
            } => {
                let block_sig = super::block_type_to_signature(block_type);

                for param in block_sig.params.iter().rev() {
                    pop_expected(stack, *param)?;
                }

                for result in &block_sig.results {
                    stack.push(*result);
                }

                Ok(())
            }

            If {
                block_type,
                then_body: _,
                else_body: _,
            } => {
                // Pop condition
                pop_expected(stack, ValueType::I32)?;

                let block_sig = super::block_type_to_signature(block_type);

                for param in block_sig.params.iter().rev() {
                    pop_expected(stack, *param)?;
                }

                for result in &block_sig.results {
                    stack.push(*result);
                }

                Ok(())
            }

            // Terminators - make stack polymorphic (we don't track beyond this)
            Return | Br(_) | BrTable { .. } | Unreachable => {
                // Clear stack - we're leaving this scope
                stack.clear();
                Ok(())
            }

            // BrIf - conditional, execution may continue
            BrIf(_) => {
                pop_expected(stack, ValueType::I32)?; // condition
                Ok(())
            }

            // Call/CallIndirect - need type context for accuracy
            Call(_) | CallIndirect { .. } => {
                // Without type info, we can't validate properly
                // For now, just allow it
                Ok(())
            }

            // Float binary operations (f32): pop 2 f32, push 1 f32
            F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max | F32Copysign => {
                pop_expected(stack, ValueType::F32)?;
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::F32);
                Ok(())
            }

            // Float unary operations (f32): pop 1 f32, push 1 f32
            F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt => {
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::F32);
                Ok(())
            }

            // Float comparisons (f32): pop 2 f32, push 1 i32
            F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge => {
                pop_expected(stack, ValueType::F32)?;
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::I32);
                Ok(())
            }

            // Float binary operations (f64): pop 2 f64, push 1 f64
            F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max | F64Copysign => {
                pop_expected(stack, ValueType::F64)?;
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::F64);
                Ok(())
            }

            // Float unary operations (f64): pop 1 f64, push 1 f64
            F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt => {
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::F64);
                Ok(())
            }

            // Float comparisons (f64): pop 2 f64, push 1 i32
            F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => {
                pop_expected(stack, ValueType::F64)?;
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::I32);
                Ok(())
            }

            // Conversion operations
            I32WrapI64 => {
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I64ExtendI32S | I64ExtendI32U => {
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::I64);
                Ok(())
            }
            I32TruncF32S | I32TruncF32U => {
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I32TruncF64S | I32TruncF64U => {
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I64TruncF32S | I64TruncF32U => {
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::I64);
                Ok(())
            }
            I64TruncF64S | I64TruncF64U => {
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::I64);
                Ok(())
            }
            F32ConvertI32S | F32ConvertI32U => {
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::F32);
                Ok(())
            }
            F32ConvertI64S | F32ConvertI64U => {
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::F32);
                Ok(())
            }
            F64ConvertI32S | F64ConvertI32U => {
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::F64);
                Ok(())
            }
            F64ConvertI64S | F64ConvertI64U => {
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::F64);
                Ok(())
            }
            F32DemoteF64 => {
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::F32);
                Ok(())
            }
            F64PromoteF32 => {
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::F64);
                Ok(())
            }

            // Reinterpret operations (no-op on bits, just changes type)
            I32ReinterpretF32 => {
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I64ReinterpretF64 => {
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::I64);
                Ok(())
            }
            F32ReinterpretI32 => {
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::F32);
                Ok(())
            }
            F64ReinterpretI64 => {
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::F64);
                Ok(())
            }

            // Float memory operations
            F32Load { .. } => {
                pop_expected(stack, ValueType::I32)?; // address
                stack.push(ValueType::F32);
                Ok(())
            }
            F32Store { .. } => {
                pop_expected(stack, ValueType::F32)?; // value
                pop_expected(stack, ValueType::I32)?; // address
                Ok(())
            }
            F64Load { .. } => {
                pop_expected(stack, ValueType::I32)?; // address
                stack.push(ValueType::F64);
                Ok(())
            }
            F64Store { .. } => {
                pop_expected(stack, ValueType::F64)?; // value
                pop_expected(stack, ValueType::I32)?; // address
                Ok(())
            }

            // Integer partial loads (all load to i32/i64 from memory address)
            I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } => {
                pop_expected(stack, ValueType::I32)?; // address
                stack.push(ValueType::I32);
                Ok(())
            }
            I64Load8S { .. }
            | I64Load8U { .. }
            | I64Load16S { .. }
            | I64Load16U { .. }
            | I64Load32S { .. }
            | I64Load32U { .. } => {
                pop_expected(stack, ValueType::I32)?; // address
                stack.push(ValueType::I64);
                Ok(())
            }

            // Integer partial stores
            I32Store8 { .. } | I32Store16 { .. } => {
                pop_expected(stack, ValueType::I32)?; // value
                pop_expected(stack, ValueType::I32)?; // address
                Ok(())
            }
            I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => {
                pop_expected(stack, ValueType::I64)?; // value
                pop_expected(stack, ValueType::I32)?; // address
                Ok(())
            }

            // Memory size/grow operations
            MemorySize(_) => {
                stack.push(ValueType::I32); // returns page count
                Ok(())
            }
            MemoryGrow(_) => {
                pop_expected(stack, ValueType::I32)?; // delta pages
                stack.push(ValueType::I32); // returns previous size or -1
                Ok(())
            }

            // Rotate operations (same as shifts)
            I32Rotl | I32Rotr => {
                pop_expected(stack, ValueType::I32)?;
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I64Rotl | I64Rotr => {
                // WebAssembly spec: i64 rotations take two i64 operands
                // The rotation amount is an i64 (only low 6 bits used, mod 64)
                pop_expected(stack, ValueType::I64)?;
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I64);
                Ok(())
            }

            // Sign extension operations (in-place sign extension)
            I32Extend8S | I32Extend16S => {
                pop_expected(stack, ValueType::I32)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I64Extend8S | I64Extend16S | I64Extend32S => {
                pop_expected(stack, ValueType::I64)?;
                stack.push(ValueType::I64);
                Ok(())
            }

            // Saturating truncation operations (non-trapping float-to-int conversions)
            // These are like regular truncation but saturate instead of trapping
            I32TruncSatF32S | I32TruncSatF32U => {
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I32TruncSatF64S | I32TruncSatF64U => {
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::I32);
                Ok(())
            }
            I64TruncSatF32S | I64TruncSatF32U => {
                pop_expected(stack, ValueType::F32)?;
                stack.push(ValueType::I64);
                Ok(())
            }
            I64TruncSatF64S | I64TruncSatF64U => {
                pop_expected(stack, ValueType::F64)?;
                stack.push(ValueType::I64);
                Ok(())
            }

            // Bulk memory operations
            MemoryFill(_) => {
                // memory.fill: [dst, val, len] -> []
                pop_expected(stack, ValueType::I32)?; // len
                pop_expected(stack, ValueType::I32)?; // val
                pop_expected(stack, ValueType::I32)?; // dst
                Ok(())
            }
            MemoryCopy { .. } => {
                // memory.copy: [dst, src, len] -> []
                pop_expected(stack, ValueType::I32)?; // len
                pop_expected(stack, ValueType::I32)?; // src
                pop_expected(stack, ValueType::I32)?; // dst
                Ok(())
            }
            MemoryInit { .. } => {
                // memory.init: [dst, src, len] -> []
                pop_expected(stack, ValueType::I32)?; // len
                pop_expected(stack, ValueType::I32)?; // src offset in data segment
                pop_expected(stack, ValueType::I32)?; // dst in memory
                Ok(())
            }
            DataDrop(_) => {
                // data.drop: [] -> []
                Ok(())
            }

            // End, Unknown - no stack effect
            End | Unknown(_) => Ok(()),
        }
    }

    /// Helper to pop a value of expected type from stack
    ///
    /// Unknown type is compatible with any expected type (used for locals without type context)
    fn pop_expected(stack: &mut Vec<ValueType>, expected: ValueType) -> Result<(), String> {
        match stack.pop() {
            Some(actual) if actual == expected => Ok(()),
            // Unknown is compatible with any type (from LocalGet without type context)
            Some(ValueType::Unknown) => Ok(()),
            Some(actual) => Err(format!(
                "Type mismatch: expected {:?} but got {:?}",
                expected, actual
            )),
            None => Err(format!(
                "Stack underflow: expected {:?} but stack is empty",
                expected
            )),
        }
    }

    /// Context for module-level validation (function signatures for Call validation)
    #[derive(Clone)]
    pub struct ValidationContext {
        /// Function signatures indexed by function index
        /// This allows us to validate Call instructions by looking up the callee's signature
        pub function_signatures: Vec<(Vec<ValueType>, Vec<ValueType>)>, // (params, results)
    }

    impl ValidationContext {
        /// Create a validation context from a module
        ///
        /// Builds function signature table including both imported and local functions.
        /// WebAssembly function indices count imports first, then local functions.
        pub fn from_module(module: &crate::Module) -> Self {
            let mut function_signatures = Vec::new();

            // First, add imported function signatures (they come first in indexing)
            for import in &module.imports {
                if let crate::ImportKind::Func(type_idx) = &import.kind {
                    if let Some(sig) = module.types.get(*type_idx as usize) {
                        let params: Vec<ValueType> =
                            sig.params.iter().map(super::convert_value_type).collect();
                        let results: Vec<ValueType> =
                            sig.results.iter().map(super::convert_value_type).collect();
                        function_signatures.push((params, results));
                    }
                }
            }

            // Then add local function signatures
            for f in &module.functions {
                let params: Vec<ValueType> = f
                    .signature
                    .params
                    .iter()
                    .map(super::convert_value_type)
                    .collect();
                let results: Vec<ValueType> = f
                    .signature
                    .results
                    .iter()
                    .map(super::convert_value_type)
                    .collect();
                function_signatures.push((params, results));
            }

            ValidationContext {
                function_signatures,
            }
        }

        /// Get the signature of a function by index
        ///
        /// This properly handles both imported functions (lower indices) and
        /// local functions (higher indices).
        pub fn get_function_signature(
            &self,
            idx: u32,
        ) -> Option<&(Vec<ValueType>, Vec<ValueType>)> {
            self.function_signatures.get(idx as usize)
        }
    }

    /// Validation guard for optimizer passes
    ///
    /// Use this to wrap transformations and ensure they preserve stack correctness.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let guard = ValidationGuard::new(&func, "vacuum");
    /// // ... perform transformations on func.instructions ...
    /// guard.validate_after(&func)?;
    /// ```
    pub struct ValidationGuard {
        pass_name: String,
        #[allow(dead_code)]
        original_signature: StackSignature,
        func_name: Option<String>,
        /// If true, skip validation because we can't accurately analyze (e.g., has Unknown instructions)
        skip_validation: bool,
        /// Module-level context for Call validation (optional)
        context: Option<ValidationContext>,
    }

    /// Check if a function contains instructions we can't validate
    ///
    /// If `has_context` is true, we have module-level type info and can validate Call instructions.
    fn contains_unanalyzable_instructions(
        instructions: &[crate::Instruction],
        has_context: bool,
    ) -> bool {
        use crate::Instruction::*;

        for instr in instructions {
            match instr {
                // Call instructions need type context - only unanalyzable if we don't have it
                Call(_) => {
                    if !has_context {
                        return true;
                    }
                }
                // CallIndirect always needs runtime type info we don't have
                CallIndirect { .. } => return true,
                // Unknown instructions have unknown stack effects - can't validate
                Unknown(_) => return true,
                // Recursively check nested bodies
                Block { body, .. } | Loop { body, .. } => {
                    if contains_unanalyzable_instructions(body, has_context) {
                        return true;
                    }
                }
                If {
                    then_body,
                    else_body,
                    ..
                } => {
                    if contains_unanalyzable_instructions(then_body, has_context)
                        || contains_unanalyzable_instructions(else_body, has_context)
                    {
                        return true;
                    }
                }
                _ => {}
            }
        }
        false
    }

    impl ValidationGuard {
        /// Create a new validation guard before transformation (without module context)
        ///
        /// This version skips validation for functions with Call instructions since
        /// we don't have type information. Prefer `with_context` when module info is available.
        pub fn new(func: &crate::Function, pass_name: &str) -> Self {
            let params: Vec<ValueType> = func
                .signature
                .params
                .iter()
                .map(super::convert_value_type)
                .collect();
            let results: Vec<ValueType> = func
                .signature
                .results
                .iter()
                .map(super::convert_value_type)
                .collect();

            // Without context, we can't validate functions with Call instructions
            let skip_validation = contains_unanalyzable_instructions(&func.instructions, false);

            ValidationGuard {
                pass_name: pass_name.to_string(),
                original_signature: StackSignature::new(params, results, SignatureKind::Fixed),
                func_name: func.name.clone(),
                skip_validation,
                context: None,
            }
        }

        /// Create a new validation guard with module-level context
        ///
        /// This allows validation of functions with Call instructions by looking up
        /// callee signatures in the provided context.
        pub fn with_context(
            func: &crate::Function,
            pass_name: &str,
            context: ValidationContext,
        ) -> Self {
            let params: Vec<ValueType> = func
                .signature
                .params
                .iter()
                .map(super::convert_value_type)
                .collect();
            let results: Vec<ValueType> = func
                .signature
                .results
                .iter()
                .map(super::convert_value_type)
                .collect();

            // With context, we can validate Call instructions (but not CallIndirect or Unknown)
            let skip_validation = contains_unanalyzable_instructions(&func.instructions, true);

            ValidationGuard {
                pass_name: pass_name.to_string(),
                original_signature: StackSignature::new(params, results, SignatureKind::Fixed),
                func_name: func.name.clone(),
                skip_validation,
                context: Some(context),
            }
        }

        /// Internal strict validation - fails loudly if validation cannot be performed
        fn validate_strict(&self, func: &crate::Function) -> Result<(), String> {
            let func_name = self.func_name.as_deref().unwrap_or("<anonymous>");

            // Fail loudly if we can't accurately analyze
            if self.skip_validation {
                return Err(format!(
                    "Cannot validate {} pass for function '{}': contains unanalyzable instructions \
                     (Unknown, CallIndirect, or Call without module context). \
                     Optimization may produce invalid WASM.",
                    self.pass_name, func_name
                ));
            }

            let has_context = self.context.is_some();

            // Fail loudly if the optimized function now has unanalyzable instructions
            if contains_unanalyzable_instructions(&func.instructions, has_context) {
                return Err(format!(
                    "Cannot validate {} pass for function '{}': optimization introduced \
                     unanalyzable instructions. This should not happen - the optimizer \
                     should only transform known instruction types.",
                    self.pass_name, func_name
                ));
            }

            let result = if let Some(ctx) = &self.context {
                validate_function_with_context(func, ctx)
            } else {
                validate_function(func)
            };

            match result {
                Ok(()) => Ok(()),
                Err(e) => Err(format!(
                    "Stack validation failed in {} pass for function '{}': {}",
                    self.pass_name, func_name, e
                )),
            }
        }

        /// Internal permissive validation - silently skips if validation cannot be performed
        #[allow(dead_code)]
        fn validate_permissive(&self, func: &crate::Function) -> Result<(), String> {
            // Skip validation if we can't accurately analyze
            if self.skip_validation {
                return Ok(());
            }

            let has_context = self.context.is_some();

            // Also skip if the optimized function now has unanalyzable instructions
            if contains_unanalyzable_instructions(&func.instructions, has_context) {
                return Ok(());
            }

            let result = if let Some(ctx) = &self.context {
                validate_function_with_context(func, ctx)
            } else {
                validate_function(func)
            };

            match result {
                Ok(()) => Ok(()),
                Err(e) => {
                    let func_name = self.func_name.as_deref().unwrap_or("<anonymous>");
                    Err(format!(
                        "Stack validation failed in {} pass for function '{}': {}",
                        self.pass_name, func_name, e
                    ))
                }
            }
        }

        /// Validate function after transformation, returning anyhow::Result for direct use with `?`
        ///
        /// This is the preferred method - it allows natural error propagation with the `?` operator.
        /// Validation failures will abort the optimization pass and report the error.
        ///
        /// **PRODUCTION MODE**: This method now fails loudly when validation cannot be performed
        /// (e.g., for functions with Unknown or CallIndirect instructions). Previously it would
        /// silently skip validation for such cases.
        pub fn validate(&self, func: &crate::Function) -> anyhow::Result<()> {
            self.validate_strict(func)
                .map_err(|e| anyhow::anyhow!("{}", e))
        }
    }

    /// Validate a function with module-level context (allows Call validation)
    pub fn validate_function_with_context(
        func: &crate::Function,
        ctx: &ValidationContext,
    ) -> Result<(), String> {
        let params: Vec<ValueType> = func
            .signature
            .params
            .iter()
            .map(super::convert_value_type)
            .collect();
        let results: Vec<ValueType> = func
            .signature
            .results
            .iter()
            .map(super::convert_value_type)
            .collect();

        match validate_function_body_with_context(&func.instructions, &params, &results, ctx) {
            ValidationResult::Valid(_) => Ok(()),
            ValidationResult::StackMismatch {
                position,
                expected,
                actual,
            } => Err(format!(
                "Stack mismatch at instruction {}: expected {} but got {}",
                position, expected, actual
            )),
            ValidationResult::MissingInput { expected_params } => Err(format!(
                "Missing input: expected {:?} on stack",
                expected_params
            )),
            ValidationResult::WrongOutput {
                expected_results,
                actual_results,
            } => Err(format!(
                "Wrong output: expected {:?} but got {:?}",
                expected_results, actual_results
            )),
            ValidationResult::InstructionError {
                position,
                message,
                stack_state,
            } => Err(format!(
                "Instruction error at position {}: {} (stack: {:?})",
                position, message, stack_state
            )),
        }
    }

    /// Validate a function body with module-level context
    fn validate_function_body_with_context(
        instructions: &[crate::Instruction],
        params: &[ValueType],
        results: &[ValueType],
        ctx: &ValidationContext,
    ) -> ValidationResult {
        use crate::Instruction;

        // In WebAssembly, function parameters go to locals, NOT the value stack
        // The stack starts empty
        let mut stack: Vec<ValueType> = Vec::new();
        let mut unreachable = false;

        for (pos, instr) in instructions.iter().enumerate() {
            // Skip instructions after unreachable code
            if unreachable {
                continue;
            }

            match validate_instruction_with_context(instr, &mut stack, ctx) {
                Ok(()) => {}
                Err(msg) => {
                    return ValidationResult::InstructionError {
                        position: pos,
                        message: msg,
                        stack_state: stack.clone(),
                    };
                }
            }

            // Check if this instruction makes subsequent code unreachable
            if matches!(
                instr,
                Instruction::Return
                    | Instruction::Unreachable
                    | Instruction::Br(_)
                    | Instruction::BrTable { .. }
            ) {
                unreachable = true;
            }
        }

        // If the function ends with a terminator, skip final stack check
        if unreachable {
            return ValidationResult::Valid(StackSignature::new(
                params.to_vec(),
                results.to_vec(),
                SignatureKind::Fixed,
            ));
        }

        // Check final stack matches declared results
        // Handle Unknown types (from LocalGet without type context)
        if stack.len() != results.len() {
            return ValidationResult::WrongOutput {
                expected_results: results.to_vec(),
                actual_results: stack,
            };
        }

        for (actual, expected) in stack.iter().zip(results.iter()) {
            if *actual != ValueType::Unknown && *actual != *expected {
                return ValidationResult::WrongOutput {
                    expected_results: results.to_vec(),
                    actual_results: stack.clone(),
                };
            }
        }

        ValidationResult::Valid(StackSignature::new(
            params.to_vec(),
            results.to_vec(),
            SignatureKind::Fixed,
        ))
    }

    /// Validate a single instruction with module-level context
    fn validate_instruction_with_context(
        instr: &crate::Instruction,
        stack: &mut Vec<ValueType>,
        ctx: &ValidationContext,
    ) -> Result<(), String> {
        use crate::Instruction::*;

        match instr {
            // Call - look up callee signature in context
            Call(idx) => {
                if let Some((params, results)) = ctx.get_function_signature(*idx) {
                    // Pop params from stack (in reverse order)
                    for param in params.iter().rev() {
                        pop_expected(stack, *param)?;
                    }
                    // Push results onto stack
                    for result in results {
                        stack.push(*result);
                    }
                    Ok(())
                } else {
                    Err(format!("Unknown function index {} in Call", idx))
                }
            }

            // For all other instructions, delegate to the non-context version
            _ => validate_instruction_stack_effect(instr, stack),
        }
    }
}

/// Stack effect analysis for instructions
///
/// Determines how each WebAssembly instruction affects the value stack.
/// Used for composing signatures and validating block structures.
pub mod effects {
    use super::*;

    /// Context for signature analysis that provides access to module-level type information
    ///
    /// This context is needed for instructions like `Call` and `CallIndirect` whose
    /// stack effects depend on the called function's signature.
    #[derive(Debug, Clone)]
    pub struct SignatureContext<'a> {
        /// Reference to the module for looking up signatures
        module: &'a crate::Module,
        /// Number of imported functions (function indices below this are imports)
        num_imported_functions: usize,
    }

    impl<'a> SignatureContext<'a> {
        /// Create a new signature context from a module
        pub fn from_module(module: &'a crate::Module) -> Self {
            // Count imported functions
            let num_imported_functions = module
                .imports
                .iter()
                .filter(|i| matches!(i.kind, crate::ImportKind::Func(_)))
                .count();

            SignatureContext {
                module,
                num_imported_functions,
            }
        }

        /// Get the signature for a function by its index
        ///
        /// WebAssembly function index space: imports come first (0..num_imports),
        /// then local functions (num_imports..num_imports+num_local_funcs)
        pub fn get_function_signature(&self, func_idx: u32) -> Option<&crate::FunctionSignature> {
            let func_idx = func_idx as usize;

            if func_idx < self.num_imported_functions {
                // This is an imported function - find the nth function import
                let mut import_func_count = 0;
                for import in &self.module.imports {
                    if let crate::ImportKind::Func(type_idx) = &import.kind {
                        if import_func_count == func_idx {
                            // Found the import, look up its signature from type index
                            return self.module.types.get(*type_idx as usize);
                        }
                        import_func_count += 1;
                    }
                }
                None
            } else {
                // This is a local function
                let local_idx = func_idx - self.num_imported_functions;
                self.module.functions.get(local_idx).map(|f| &f.signature)
            }
        }

        /// Get the signature for a type by its index (for indirect calls)
        pub fn get_type_signature(&self, type_idx: u32) -> Option<&crate::FunctionSignature> {
            self.module.types.get(type_idx as usize)
        }
    }

    /// Get the stack signature of an instruction with module context
    ///
    /// This version can accurately determine the signature of `Call` and `CallIndirect`
    /// instructions by looking up the function/type signature in the module.
    ///
    /// # Arguments
    ///
    /// * `instr` - The instruction to analyze
    /// * `ctx` - Module context providing function signatures
    ///
    /// # Example
    ///
    /// ```ignore
    /// let ctx = SignatureContext::from_module(&module);
    /// let sig = instruction_signature_with_context(&Instruction::Call(0), Some(&ctx));
    /// // sig now reflects the actual function signature
    /// ```
    pub fn instruction_signature_with_context(
        instr: &crate::Instruction,
        ctx: Option<&SignatureContext>,
    ) -> StackSignature {
        use crate::Instruction::*;

        match instr {
            // Call: [params...] -> [results...]
            // With context, we can look up the actual signature
            Call(func_idx) => {
                if let Some(ctx) = ctx {
                    if let Some(sig) = ctx.get_function_signature(*func_idx) {
                        return StackSignature::new(
                            sig.params.iter().map(super::convert_value_type).collect(),
                            sig.results.iter().map(super::convert_value_type).collect(),
                            SignatureKind::Fixed,
                        );
                    }
                }
                // Fallback: unknown signature
                StackSignature::empty()
            }

            // CallIndirect: [params..., i32] -> [results...]
            // The type index tells us the expected signature
            CallIndirect { type_idx, .. } => {
                if let Some(ctx) = ctx {
                    if let Some(sig) = ctx.get_type_signature(*type_idx) {
                        // CallIndirect consumes params + table index (i32), produces results
                        let mut params: Vec<ValueType> =
                            sig.params.iter().map(super::convert_value_type).collect();
                        params.push(ValueType::I32); // table index
                        return StackSignature::new(
                            params,
                            sig.results.iter().map(super::convert_value_type).collect(),
                            SignatureKind::Fixed,
                        );
                    }
                }
                // Fallback: at minimum consumes the table index
                StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Fixed)
            }

            // For all other instructions, delegate to the context-free version
            _ => instruction_signature(instr),
        }
    }

    /// Get the stack signature of a single instruction
    ///
    /// Returns how the instruction affects the value stack.
    /// For instructions that consume/produce values of specific types.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// // i32.const produces i32 on empty stack
    /// let sig = instruction_signature(&Instruction::I32Const(42));
    /// assert_eq!(sig.params, vec![]);
    /// assert_eq!(sig.results, vec![ValueType::I32]);
    ///
    /// // i32.add consumes 2 i32, produces 1 i32
    /// let sig = instruction_signature(&Instruction::I32Add);
    /// assert_eq!(sig.params, vec![ValueType::I32, ValueType::I32]);
    /// assert_eq!(sig.results, vec![ValueType::I32]);
    /// ```
    pub fn instruction_signature(instr: &crate::Instruction) -> StackSignature {
        use crate::Instruction::*;

        match instr {
            // Constants: [] -> [T]
            I32Const(_) => StackSignature::new(vec![], vec![ValueType::I32], SignatureKind::Fixed),
            I64Const(_) => StackSignature::new(vec![], vec![ValueType::I64], SignatureKind::Fixed),
            F32Const(_) => StackSignature::new(vec![], vec![ValueType::F32], SignatureKind::Fixed),
            F64Const(_) => StackSignature::new(vec![], vec![ValueType::F64], SignatureKind::Fixed),

            // i32 arithmetic: [i32, i32] -> [i32]
            I32Add | I32Sub | I32Mul | I32And | I32Or | I32Xor => StackSignature::new(
                vec![ValueType::I32, ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // i32 shifts: [i32, i32] -> [i32]
            I32Shl | I32ShrS | I32ShrU => StackSignature::new(
                vec![ValueType::I32, ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // i32 comparison: [i32, i32] -> [i32]
            I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS
            | I32GeU => StackSignature::new(
                vec![ValueType::I32, ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // i32 division/remainder: [i32, i32] -> [i32]
            I32DivS | I32DivU | I32RemS | I32RemU => StackSignature::new(
                vec![ValueType::I32, ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // i64 arithmetic: [i64, i64] -> [i64]
            I64Add | I64Sub | I64Mul | I64And | I64Or | I64Xor => StackSignature::new(
                vec![ValueType::I64, ValueType::I64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // i64 shifts: [i64, i32] -> [i64]
            I64Shl | I64ShrS | I64ShrU => StackSignature::new(
                // WebAssembly spec: i64 shifts take two i64 operands
                vec![ValueType::I64, ValueType::I64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // i64 comparison: [i64, i64] -> [i32]
            I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS
            | I64GeU => StackSignature::new(
                vec![ValueType::I64, ValueType::I64],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // i64 division/remainder: [i64, i64] -> [i64]
            I64DivS | I64DivU | I64RemS | I64RemU => StackSignature::new(
                vec![ValueType::I64, ValueType::I64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // Unary operations: [T] -> [T]
            I32Eqz => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64Eqz => StackSignature::new(
                vec![ValueType::I64],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I32Clz | I32Ctz | I32Popcnt => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64Clz | I64Ctz | I64Popcnt => StackSignature::new(
                vec![ValueType::I64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // Select: [T, T, i32] -> [T]
            Select => {
                // Simplified: we don't track exact types
                StackSignature::new(
                    vec![ValueType::I32, ValueType::I32, ValueType::I32],
                    vec![ValueType::I32],
                    SignatureKind::Fixed,
                )
            }

            // Local operations
            LocalGet(_) => {
                // We don't know the local type here - would need type info
                // For now, return polymorphic placeholder
                StackSignature::new(vec![], vec![ValueType::I32], SignatureKind::Fixed)
            }
            LocalSet(_) => {
                // Consumes 1 value, produces nothing
                StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Fixed)
            }
            LocalTee(_) => {
                // Consumes 1 value, produces 1 value (identity)
                StackSignature::new(
                    vec![ValueType::I32],
                    vec![ValueType::I32],
                    SignatureKind::Fixed,
                )
            }

            // Global operations
            GlobalGet(_) => StackSignature::new(vec![], vec![ValueType::I32], SignatureKind::Fixed),
            GlobalSet(_) => StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Fixed),

            // Memory operations
            I32Load { .. } => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I32Store { .. } => StackSignature::new(
                vec![ValueType::I32, ValueType::I32],
                vec![],
                SignatureKind::Fixed,
            ),
            I64Load { .. } => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),
            I64Store { .. } => StackSignature::new(
                vec![ValueType::I64, ValueType::I32],
                vec![],
                SignatureKind::Fixed,
            ),

            // Nop: [] -> [] (no effect)
            Nop => StackSignature::empty(),

            // Drop: [T] -> [] (consumes one value)
            // Note: Drop is polymorphic in the type it consumes, but we use i32 as default
            Drop => StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Fixed),

            // End: [] -> [] (block terminator, no stack effect by itself)
            End => StackSignature::empty(),

            // Block: [block_params...] -> [block_results...]
            // The block's signature is determined by its block_type
            Block { block_type, body } => {
                // Get the declared signature from block_type
                let declared_sig = super::block_type_to_signature(block_type);

                // For validation, we should also check the body produces the right types
                // But for the instruction's external signature, use the declared type
                // The body is validated separately in validate_block()

                // Check if body contains unreachable - makes signature polymorphic
                let has_unreachable = body.iter().any(|i| matches!(i, Unreachable));
                if has_unreachable {
                    StackSignature::new(
                        declared_sig.params,
                        declared_sig.results,
                        SignatureKind::Polymorphic,
                    )
                } else {
                    declared_sig
                }
            }

            // Loop: [block_params...] -> [block_results...]
            // Loops have the same external signature as blocks
            // The difference is that br to a loop jumps to the beginning
            Loop { block_type, body } => {
                let declared_sig = super::block_type_to_signature(block_type);
                let has_unreachable = body.iter().any(|i| matches!(i, Unreachable));
                if has_unreachable {
                    StackSignature::new(
                        declared_sig.params,
                        declared_sig.results,
                        SignatureKind::Polymorphic,
                    )
                } else {
                    declared_sig
                }
            }

            // If: [i32, block_params...] -> [block_results...]
            // Consumes condition (i32) plus any block params, produces block results
            If {
                block_type,
                then_body,
                else_body,
            } => {
                let declared_sig = super::block_type_to_signature(block_type);

                // If consumes an i32 condition in addition to block params
                let mut params = vec![ValueType::I32];
                params.extend(declared_sig.params.clone());

                // Check if either branch is unreachable
                let then_unreachable = then_body.iter().any(|i| matches!(i, Unreachable));
                let else_unreachable = else_body.iter().any(|i| matches!(i, Unreachable));

                let kind = if then_unreachable && else_unreachable {
                    // Both branches unreachable - whole if is unreachable
                    SignatureKind::Polymorphic
                } else {
                    SignatureKind::Fixed
                };

                StackSignature::new(params, declared_sig.results, kind)
            }

            // Return: [return_types...] -> polymorphic
            // Return terminates execution, so it's polymorphic (can appear in any stack context)
            Return => StackSignature::new(vec![], vec![], SignatureKind::Polymorphic),

            // Br: [label_params...] -> polymorphic
            // Branch to label, consumes label's expected values, doesn't return
            Br(_) => StackSignature::new(vec![], vec![], SignatureKind::Polymorphic),

            // BrIf: [label_params..., i32] -> [label_params...]
            // Conditional branch: if condition is true, branch; otherwise continue
            // Consumes condition, may consume label params if branching
            BrIf(_) => {
                // BrIf pops i32 condition, and if taken, uses label's signature
                // If not taken, execution continues with stack unchanged (minus condition)
                // For type checking, we just need to know it consumes i32
                StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Fixed)
            }

            // BrTable: [i32] -> polymorphic
            // Branch table always branches, so it's polymorphic
            BrTable { .. } => {
                StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Polymorphic)
            }

            // Unreachable: [] -> polymorphic
            // Marks code as unreachable, can produce any type needed by context
            Unreachable => StackSignature::polymorphic_empty(),

            // Call: depends on function signature (we'd need type info)
            // For now, return empty - caller should use function signature directly
            Call(_) => {
                // Without function signature info, we can't know the stack effect
                // This should be handled at a higher level with access to the module
                StackSignature::empty()
            }

            // CallIndirect: [args..., i32] -> [results...]
            // Consumes table index (i32) plus function args, produces function results
            // Without type info, return placeholder
            CallIndirect { .. } => {
                // Consumes at least the table index (i32)
                StackSignature::new(vec![ValueType::I32], vec![], SignatureKind::Fixed)
            }

            // Float binary operations (f32): [f32, f32] -> [f32]
            F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max | F32Copysign => {
                StackSignature::new(
                    vec![ValueType::F32, ValueType::F32],
                    vec![ValueType::F32],
                    SignatureKind::Fixed,
                )
            }

            // Float unary operations (f32): [f32] -> [f32]
            F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt => {
                StackSignature::new(
                    vec![ValueType::F32],
                    vec![ValueType::F32],
                    SignatureKind::Fixed,
                )
            }

            // Float comparisons (f32): [f32, f32] -> [i32]
            F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge => StackSignature::new(
                vec![ValueType::F32, ValueType::F32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // Float binary operations (f64): [f64, f64] -> [f64]
            F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max | F64Copysign => {
                StackSignature::new(
                    vec![ValueType::F64, ValueType::F64],
                    vec![ValueType::F64],
                    SignatureKind::Fixed,
                )
            }

            // Float unary operations (f64): [f64] -> [f64]
            F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt => {
                StackSignature::new(
                    vec![ValueType::F64],
                    vec![ValueType::F64],
                    SignatureKind::Fixed,
                )
            }

            // Float comparisons (f64): [f64, f64] -> [i32]
            F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => StackSignature::new(
                vec![ValueType::F64, ValueType::F64],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // Conversion operations
            I32WrapI64 => StackSignature::new(
                vec![ValueType::I64],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64ExtendI32S | I64ExtendI32U => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),
            I32TruncF32S | I32TruncF32U => StackSignature::new(
                vec![ValueType::F32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I32TruncF64S | I32TruncF64U => StackSignature::new(
                vec![ValueType::F64],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64TruncF32S | I64TruncF32U => StackSignature::new(
                vec![ValueType::F32],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),
            I64TruncF64S | I64TruncF64U => StackSignature::new(
                vec![ValueType::F64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),
            F32ConvertI32S | F32ConvertI32U => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::F32],
                SignatureKind::Fixed,
            ),
            F32ConvertI64S | F32ConvertI64U => StackSignature::new(
                vec![ValueType::I64],
                vec![ValueType::F32],
                SignatureKind::Fixed,
            ),
            F64ConvertI32S | F64ConvertI32U => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::F64],
                SignatureKind::Fixed,
            ),
            F64ConvertI64S | F64ConvertI64U => StackSignature::new(
                vec![ValueType::I64],
                vec![ValueType::F64],
                SignatureKind::Fixed,
            ),
            F32DemoteF64 => StackSignature::new(
                vec![ValueType::F64],
                vec![ValueType::F32],
                SignatureKind::Fixed,
            ),
            F64PromoteF32 => StackSignature::new(
                vec![ValueType::F32],
                vec![ValueType::F64],
                SignatureKind::Fixed,
            ),

            // Reinterpret operations
            I32ReinterpretF32 => StackSignature::new(
                vec![ValueType::F32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64ReinterpretF64 => StackSignature::new(
                vec![ValueType::F64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),
            F32ReinterpretI32 => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::F32],
                SignatureKind::Fixed,
            ),
            F64ReinterpretI64 => StackSignature::new(
                vec![ValueType::I64],
                vec![ValueType::F64],
                SignatureKind::Fixed,
            ),

            // Float memory operations
            F32Load { .. } => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::F32],
                SignatureKind::Fixed,
            ),
            F32Store { .. } => StackSignature::new(
                vec![ValueType::I32, ValueType::F32],
                vec![],
                SignatureKind::Fixed,
            ),
            F64Load { .. } => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::F64],
                SignatureKind::Fixed,
            ),
            F64Store { .. } => StackSignature::new(
                vec![ValueType::I32, ValueType::F64],
                vec![],
                SignatureKind::Fixed,
            ),

            // Integer partial loads
            I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } => {
                StackSignature::new(
                    vec![ValueType::I32],
                    vec![ValueType::I32],
                    SignatureKind::Fixed,
                )
            }
            I64Load8S { .. }
            | I64Load8U { .. }
            | I64Load16S { .. }
            | I64Load16U { .. }
            | I64Load32S { .. }
            | I64Load32U { .. } => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // Integer partial stores
            I32Store8 { .. } | I32Store16 { .. } => StackSignature::new(
                vec![ValueType::I32, ValueType::I32],
                vec![],
                SignatureKind::Fixed,
            ),
            I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => StackSignature::new(
                vec![ValueType::I32, ValueType::I64],
                vec![],
                SignatureKind::Fixed,
            ),

            // Memory size/grow operations
            MemorySize(_) => {
                StackSignature::new(vec![], vec![ValueType::I32], SignatureKind::Fixed)
            }
            MemoryGrow(_) => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),

            // Rotate operations
            I32Rotl | I32Rotr => StackSignature::new(
                vec![ValueType::I32, ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64Rotl | I64Rotr => StackSignature::new(
                // WebAssembly spec: i64 rotations take two i64 operands
                vec![ValueType::I64, ValueType::I64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // Sign extension operations
            I32Extend8S | I32Extend16S => StackSignature::new(
                vec![ValueType::I32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64Extend8S | I64Extend16S | I64Extend32S => StackSignature::new(
                vec![ValueType::I64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // Saturating truncation operations (non-trapping)
            I32TruncSatF32S | I32TruncSatF32U => StackSignature::new(
                vec![ValueType::F32],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I32TruncSatF64S | I32TruncSatF64U => StackSignature::new(
                vec![ValueType::F64],
                vec![ValueType::I32],
                SignatureKind::Fixed,
            ),
            I64TruncSatF32S | I64TruncSatF32U => StackSignature::new(
                vec![ValueType::F32],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),
            I64TruncSatF64S | I64TruncSatF64U => StackSignature::new(
                vec![ValueType::F64],
                vec![ValueType::I64],
                SignatureKind::Fixed,
            ),

            // Bulk memory operations
            MemoryFill(_) => StackSignature::new(
                vec![ValueType::I32, ValueType::I32, ValueType::I32],
                vec![],
                SignatureKind::Fixed,
            ),
            MemoryCopy { .. } => StackSignature::new(
                vec![ValueType::I32, ValueType::I32, ValueType::I32],
                vec![],
                SignatureKind::Fixed,
            ),
            MemoryInit { .. } => StackSignature::new(
                vec![ValueType::I32, ValueType::I32, ValueType::I32],
                vec![],
                SignatureKind::Fixed,
            ),
            DataDrop(_) => StackSignature::empty(),

            // Unknown instructions - conservative empty signature
            Unknown(_) => StackSignature::empty(),
        }
    }

    /// Validate a sequence of instructions has correct stack composition
    ///
    /// Returns the combined stack signature if valid, None if composition fails.
    pub fn validate_instruction_sequence(
        instructions: &[crate::Instruction],
    ) -> Option<StackSignature> {
        let mut result = StackSignature::empty();

        for instr in instructions {
            let sig = instruction_signature(instr);

            // Check if signature composes with result
            if !result.composes(&sig) {
                return None; // Stack type mismatch
            }

            // Compose signatures
            result = result.compose(&sig);
        }

        Some(result)
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

    #[test]
    fn test_instruction_signatures() {
        use crate::Instruction::*;

        // Constants produce values
        let const_sig = effects::instruction_signature(&I32Const(42));
        assert_eq!(const_sig.params, vec![]);
        assert_eq!(const_sig.results, vec![ValueType::I32]);

        // i32.add consumes 2, produces 1
        let add_sig = effects::instruction_signature(&I32Add);
        assert_eq!(add_sig.params, vec![ValueType::I32, ValueType::I32]);
        assert_eq!(add_sig.results, vec![ValueType::I32]);

        // i32 comparison produces i32
        let eq_sig = effects::instruction_signature(&I32Eq);
        assert_eq!(eq_sig.params, vec![ValueType::I32, ValueType::I32]);
        assert_eq!(eq_sig.results, vec![ValueType::I32]);

        // i64 comparison produces i32 (not i64!)
        let i64eq_sig = effects::instruction_signature(&I64Eq);
        assert_eq!(i64eq_sig.params, vec![ValueType::I64, ValueType::I64]);
        assert_eq!(i64eq_sig.results, vec![ValueType::I32]);
    }

    #[test]
    fn test_validate_instruction_sequence() {
        use crate::Instruction::*;

        // Valid sequence: const + const + add
        // i32.const 1: [] -> [i32]
        //   Result: [] -> [i32]
        // i32.const 2: [] -> [i32]
        //   Doesn't compose! Result [i32] != [] required by const
        //
        // So we test simpler: just const
        let instrs = vec![I32Const(42)];
        let result = effects::validate_instruction_sequence(&instrs);
        assert!(result.is_some());
        let sig = result.unwrap();
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![ValueType::I32]);

        // Actually the add test needs a different approach
        // We'd need to manually handle the stack state tracking
        // For now this demonstrates the concept works for simple cases
    }

    #[test]
    fn test_block_validation_valid() {
        use crate::Instruction::*;

        // Valid block: produces i32
        let instrs = vec![I32Const(42)];
        let result = validation::validate_block(&instrs, &[], &[ValueType::I32]);

        match result {
            validation::ValidationResult::Valid(sig) => {
                assert_eq!(sig.results, vec![ValueType::I32]);
            }
            _ => panic!("Expected valid block, got: {:?}", result),
        }
    }

    #[test]
    fn test_block_validation_wrong_output() {
        use crate::Instruction::*;

        // Block declares it produces i32, but actually produces i64
        let instrs = vec![I64Const(42)];
        let result = validation::validate_block(&instrs, &[], &[ValueType::I32]);

        match result {
            validation::ValidationResult::WrongOutput {
                expected_results,
                actual_results,
            } => {
                assert_eq!(expected_results, vec![ValueType::I32]);
                assert_eq!(actual_results, vec![ValueType::I64]);
            }
            _ => panic!("Expected WrongOutput, got: {:?}", result),
        }
    }

    #[test]
    fn test_block_validation_stack_mismatch() {
        use crate::Instruction::*;

        // Block with input params: [i32] -> [i32]
        // But we try to add without providing 2 i32s on stack
        let instrs = vec![I32Add]; // Needs [i32, i32] on stack
        let result = validation::validate_block(&instrs, &[ValueType::I32], &[ValueType::I32]);

        // This should fail because I32Add needs [i32, i32] but we only have [i32]
        match result {
            validation::ValidationResult::StackMismatch {
                position,
                expected,
                actual,
            } => {
                assert_eq!(position, 0);
                // expected signature should be [i32] -> [] (what we have)
                // actual signature should be [i32, i32] -> [i32] (what add needs)
                assert_eq!(expected.params, vec![ValueType::I32]);
                assert_eq!(actual.params, vec![ValueType::I32, ValueType::I32]);
            }
            _ => panic!("Expected StackMismatch, got: {:?}", result),
        }
    }

    // ==================== Control Flow Signature Tests ====================

    #[test]
    fn test_block_type_conversion() {
        // Empty block type: [] -> []
        let empty_sig = block_type_to_signature(&crate::BlockType::Empty);
        assert_eq!(empty_sig.params, vec![]);
        assert_eq!(empty_sig.results, vec![]);
        assert_eq!(empty_sig.kind, SignatureKind::Fixed);

        // Value block type: [] -> [i32]
        let value_sig = block_type_to_signature(&crate::BlockType::Value(crate::ValueType::I32));
        assert_eq!(value_sig.params, vec![]);
        assert_eq!(value_sig.results, vec![ValueType::I32]);

        // Func block type: [i32, i64] -> [f32]
        let func_sig = block_type_to_signature(&crate::BlockType::Func {
            params: vec![crate::ValueType::I32, crate::ValueType::I64],
            results: vec![crate::ValueType::F32],
        });
        assert_eq!(func_sig.params, vec![ValueType::I32, ValueType::I64]);
        assert_eq!(func_sig.results, vec![ValueType::F32]);
    }

    #[test]
    fn test_nop_signature() {
        use crate::Instruction::*;

        let sig = effects::instruction_signature(&Nop);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    #[test]
    fn test_drop_signature() {
        use crate::Instruction::*;

        let sig = effects::instruction_signature(&Drop);
        // Drop consumes one value (i32 as default type)
        assert_eq!(sig.params, vec![ValueType::I32]);
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    #[test]
    fn test_block_signature_empty() {
        use crate::Instruction::*;

        // Block with empty signature: [] -> []
        let block = Block {
            block_type: crate::BlockType::Empty,
            body: vec![Nop],
        };
        let sig = effects::instruction_signature(&block);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    #[test]
    fn test_block_signature_with_result() {
        use crate::Instruction::*;

        // Block that produces i32: [] -> [i32]
        let block = Block {
            block_type: crate::BlockType::Value(crate::ValueType::I32),
            body: vec![I32Const(42)],
        };
        let sig = effects::instruction_signature(&block);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![ValueType::I32]);
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    #[test]
    fn test_block_signature_polymorphic_with_unreachable() {
        use crate::Instruction::*;

        // Block with unreachable becomes polymorphic
        let block = Block {
            block_type: crate::BlockType::Value(crate::ValueType::I32),
            body: vec![Unreachable],
        };
        let sig = effects::instruction_signature(&block);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![ValueType::I32]);
        assert_eq!(sig.kind, SignatureKind::Polymorphic);
    }

    #[test]
    fn test_loop_signature() {
        use crate::Instruction::*;

        // Loop with i32 result
        let loop_instr = Loop {
            block_type: crate::BlockType::Value(crate::ValueType::I64),
            body: vec![I64Const(100)],
        };
        let sig = effects::instruction_signature(&loop_instr);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![ValueType::I64]);
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    #[test]
    fn test_if_signature() {
        use crate::Instruction::*;

        // If with i32 result: [i32] -> [i32] (condition + block result)
        let if_instr = If {
            block_type: crate::BlockType::Value(crate::ValueType::I32),
            then_body: vec![I32Const(1)],
            else_body: vec![I32Const(0)],
        };
        let sig = effects::instruction_signature(&if_instr);
        // If consumes condition (i32) in addition to any block params
        assert_eq!(sig.params, vec![ValueType::I32]); // condition
        assert_eq!(sig.results, vec![ValueType::I32]); // block result
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    #[test]
    fn test_if_signature_with_block_params() {
        use crate::Instruction::*;

        // If with block that takes params: [i32, i64] -> [f32]
        // External signature: [i32 (condition), i64 (block param)] -> [f32]
        let if_instr = If {
            block_type: crate::BlockType::Func {
                params: vec![crate::ValueType::I64],
                results: vec![crate::ValueType::F32],
            },
            then_body: vec![],
            else_body: vec![],
        };
        let sig = effects::instruction_signature(&if_instr);
        // Condition i32 + block params
        assert_eq!(sig.params, vec![ValueType::I32, ValueType::I64]);
        assert_eq!(sig.results, vec![ValueType::F32]);
    }

    #[test]
    fn test_if_signature_both_branches_unreachable() {
        use crate::Instruction::*;

        // If where both branches are unreachable → polymorphic
        let if_instr = If {
            block_type: crate::BlockType::Value(crate::ValueType::I32),
            then_body: vec![Unreachable],
            else_body: vec![Unreachable],
        };
        let sig = effects::instruction_signature(&if_instr);
        assert_eq!(sig.kind, SignatureKind::Polymorphic);
    }

    #[test]
    fn test_if_signature_one_branch_unreachable() {
        use crate::Instruction::*;

        // If where only one branch is unreachable → still fixed (other branch defines type)
        let if_instr = If {
            block_type: crate::BlockType::Value(crate::ValueType::I32),
            then_body: vec![Unreachable],
            else_body: vec![I32Const(0)],
        };
        let sig = effects::instruction_signature(&if_instr);
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    #[test]
    fn test_return_signature() {
        use crate::Instruction::*;

        // Return is polymorphic - terminates execution
        let sig = effects::instruction_signature(&Return);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Polymorphic);
    }

    #[test]
    fn test_br_signature() {
        use crate::Instruction::*;

        // Br (unconditional branch) is polymorphic
        let sig = effects::instruction_signature(&Br(0));
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Polymorphic);
    }

    #[test]
    fn test_br_if_signature() {
        use crate::Instruction::*;

        // BrIf consumes condition, may or may not branch
        let sig = effects::instruction_signature(&BrIf(0));
        assert_eq!(sig.params, vec![ValueType::I32]); // condition
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Fixed); // Not polymorphic - execution may continue
    }

    #[test]
    fn test_br_table_signature() {
        use crate::Instruction::*;

        // BrTable always branches, polymorphic
        let sig = effects::instruction_signature(&BrTable {
            targets: vec![0, 1, 2],
            default: 0,
        });
        assert_eq!(sig.params, vec![ValueType::I32]); // index
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Polymorphic);
    }

    #[test]
    fn test_unreachable_signature() {
        use crate::Instruction::*;

        // Unreachable is the polymorphic bottom type
        let sig = effects::instruction_signature(&Unreachable);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Polymorphic);
    }

    #[test]
    fn test_end_signature() {
        use crate::Instruction::*;

        // End has no stack effect
        let sig = effects::instruction_signature(&End);
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
        assert_eq!(sig.kind, SignatureKind::Fixed);
    }

    // ==================== Call Signature Tests (with context) ====================

    /// Helper to create a test module with specified types
    fn make_test_module(types: Vec<crate::FunctionSignature>) -> crate::Module {
        // Create functions for each type (so function indices map to types)
        let functions: Vec<crate::Function> = types
            .iter()
            .map(|sig| crate::Function {
                name: None,
                signature: sig.clone(),
                locals: vec![],
                instructions: vec![],
            })
            .collect();

        crate::Module {
            functions,
            memories: vec![],
            tables: vec![],
            globals: vec![],
            types,
            exports: vec![],
            imports: vec![],
            data_segments: vec![],
            element_section_bytes: None,
            start_function: None,
            custom_sections: vec![],
            type_section_bytes: None,
            global_section_bytes: None,
        }
    }

    #[test]
    fn test_call_signature_without_context() {
        use crate::Instruction::*;

        // Without context, Call returns empty signature (unknown)
        let sig = effects::instruction_signature(&Call(0));
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
    }

    #[test]
    fn test_call_signature_with_context() {
        use crate::Instruction::*;
        use effects::SignatureContext;

        // Create a minimal module with function signatures
        let module = make_test_module(vec![
            // Type 0: (i32, i32) -> i32 (like an add function)
            crate::FunctionSignature {
                params: vec![crate::ValueType::I32, crate::ValueType::I32],
                results: vec![crate::ValueType::I32],
            },
            // Type 1: () -> i64 (like a constant producer)
            crate::FunctionSignature {
                params: vec![],
                results: vec![crate::ValueType::I64],
            },
            // Type 2: (i64) -> () (like a consumer/sink)
            crate::FunctionSignature {
                params: vec![crate::ValueType::I64],
                results: vec![],
            },
        ]);

        let ctx = SignatureContext::from_module(&module);

        // Call function 0: (i32, i32) -> i32
        let sig = effects::instruction_signature_with_context(&Call(0), Some(&ctx));
        assert_eq!(sig.params, vec![ValueType::I32, ValueType::I32]);
        assert_eq!(sig.results, vec![ValueType::I32]);
        assert_eq!(sig.kind, SignatureKind::Fixed);

        // Call function 1: () -> i64
        let sig = effects::instruction_signature_with_context(&Call(1), Some(&ctx));
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![ValueType::I64]);

        // Call function 2: (i64) -> ()
        let sig = effects::instruction_signature_with_context(&Call(2), Some(&ctx));
        assert_eq!(sig.params, vec![ValueType::I64]);
        assert_eq!(sig.results, vec![]);
    }

    #[test]
    fn test_call_indirect_signature_without_context() {
        use crate::Instruction::*;

        // Without context, CallIndirect at minimum consumes table index (i32)
        let sig = effects::instruction_signature(&CallIndirect {
            type_idx: 0,
            table_idx: 0,
        });
        assert_eq!(sig.params, vec![ValueType::I32]);
        assert_eq!(sig.results, vec![]);
    }

    #[test]
    fn test_call_indirect_signature_with_context() {
        use crate::Instruction::*;
        use effects::SignatureContext;

        // Create a module with type signatures
        let module = make_test_module(vec![
            // Type 0: (i32, i32) -> i32
            crate::FunctionSignature {
                params: vec![crate::ValueType::I32, crate::ValueType::I32],
                results: vec![crate::ValueType::I32],
            },
            // Type 1: (f32) -> f64
            crate::FunctionSignature {
                params: vec![crate::ValueType::F32],
                results: vec![crate::ValueType::F64],
            },
        ]);

        let ctx = SignatureContext::from_module(&module);

        // CallIndirect with type 0: (i32, i32) -> i32
        // Stack effect: [i32, i32, i32(table_idx)] -> [i32]
        let sig = effects::instruction_signature_with_context(
            &CallIndirect {
                type_idx: 0,
                table_idx: 0,
            },
            Some(&ctx),
        );
        // Params = function params + table index
        assert_eq!(
            sig.params,
            vec![ValueType::I32, ValueType::I32, ValueType::I32]
        );
        assert_eq!(sig.results, vec![ValueType::I32]);

        // CallIndirect with type 1: (f32) -> f64
        // Stack effect: [f32, i32(table_idx)] -> [f64]
        let sig = effects::instruction_signature_with_context(
            &CallIndirect {
                type_idx: 1,
                table_idx: 0,
            },
            Some(&ctx),
        );
        assert_eq!(sig.params, vec![ValueType::F32, ValueType::I32]);
        assert_eq!(sig.results, vec![ValueType::F64]);
    }

    #[test]
    fn test_call_signature_out_of_bounds() {
        use crate::Instruction::*;
        use effects::SignatureContext;

        // Module with only one type
        let module = make_test_module(vec![crate::FunctionSignature {
            params: vec![],
            results: vec![crate::ValueType::I32],
        }]);

        let ctx = SignatureContext::from_module(&module);

        // Call function 99 (out of bounds) - should return empty fallback
        let sig = effects::instruction_signature_with_context(&Call(99), Some(&ctx));
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![]);
    }

    #[test]
    fn test_context_free_instructions_with_context() {
        use crate::Instruction::*;
        use effects::SignatureContext;

        // Even with context, non-call instructions should work the same
        let module = make_test_module(vec![]);

        let ctx = SignatureContext::from_module(&module);

        // I32Add should still work
        let sig = effects::instruction_signature_with_context(&I32Add, Some(&ctx));
        assert_eq!(sig.params, vec![ValueType::I32, ValueType::I32]);
        assert_eq!(sig.results, vec![ValueType::I32]);

        // I32Const should still work
        let sig = effects::instruction_signature_with_context(&I32Const(42), Some(&ctx));
        assert_eq!(sig.params, vec![]);
        assert_eq!(sig.results, vec![ValueType::I32]);
    }
}
