/// ISLE Term Conversion - Simplified Subset for Formal Verification
///
/// This module demonstrates the bijection between Instructions and ISLE Terms
/// for a representative subset of WebAssembly instructions.
///
/// Key property to verify:
///   terms_to_instructions(instructions_to_terms(instrs)) = Ok(instrs)
///
/// The full conversion in loom-core/src/lib.rs handles ~100 instruction types.
/// This subset focuses on the core patterns: constants and binary operations.

/// Simplified instruction set (subset of full WebAssembly)
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Instruction {
    I32Const(i32),
    I64Const(i64),
    I32Add,
    I32Sub,
    I32Mul,
    I64Add,
    I64Sub,
    I64Mul,
    Drop,
    Nop,
}

/// ISLE Term representation
/// Terms form a tree structure where operations reference their operands
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Term {
    /// i32 constant: (iconst32 val)
    I32Const(i32),
    /// i64 constant: (iconst64 val)
    I64Const(i64),
    /// i32 addition: (iadd32 lhs rhs)
    I32Add(Box<Term>, Box<Term>),
    /// i32 subtraction: (isub32 lhs rhs)
    I32Sub(Box<Term>, Box<Term>),
    /// i32 multiplication: (imul32 lhs rhs)
    I32Mul(Box<Term>, Box<Term>),
    /// i64 addition: (iadd64 lhs rhs)
    I64Add(Box<Term>, Box<Term>),
    /// i64 subtraction: (isub64 lhs rhs)
    I64Sub(Box<Term>, Box<Term>),
    /// i64 multiplication: (imul64 lhs rhs)
    I64Mul(Box<Term>, Box<Term>),
    /// Drop value (side effect)
    Drop(Box<Term>),
    /// Nop (no operation)
    Nop,
}

/// Conversion error
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConversionError {
    StackUnderflow,
    InvalidTerm,
}

/// Convert instructions to terms using a simulated stack
pub fn instructions_to_terms(instructions: &[Instruction]) -> Result<Vec<Term>, ConversionError> {
    let mut stack: Vec<Term> = Vec::new();
    let mut side_effects: Vec<Term> = Vec::new();

    for instr in instructions {
        match instr {
            Instruction::I32Const(val) => {
                stack.push(Term::I32Const(*val));
            }
            Instruction::I64Const(val) => {
                stack.push(Term::I64Const(*val));
            }
            Instruction::I32Add => {
                let rhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                let lhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                stack.push(Term::I32Add(Box::new(lhs), Box::new(rhs)));
            }
            Instruction::I32Sub => {
                let rhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                let lhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                stack.push(Term::I32Sub(Box::new(lhs), Box::new(rhs)));
            }
            Instruction::I32Mul => {
                let rhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                let lhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                stack.push(Term::I32Mul(Box::new(lhs), Box::new(rhs)));
            }
            Instruction::I64Add => {
                let rhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                let lhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                stack.push(Term::I64Add(Box::new(lhs), Box::new(rhs)));
            }
            Instruction::I64Sub => {
                let rhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                let lhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                stack.push(Term::I64Sub(Box::new(lhs), Box::new(rhs)));
            }
            Instruction::I64Mul => {
                let rhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                let lhs = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                stack.push(Term::I64Mul(Box::new(lhs), Box::new(rhs)));
            }
            Instruction::Drop => {
                let val = stack.pop().ok_or(ConversionError::StackUnderflow)?;
                side_effects.push(Term::Drop(Box::new(val)));
            }
            Instruction::Nop => {
                // Nop produces no value and has no side effect
            }
        }
    }

    // Combine side effects and remaining stack values
    let mut result = side_effects;
    result.extend(stack);
    Ok(result)
}

/// Convert a single term to instructions (depth-first traversal)
fn term_to_instructions_recursive(term: &Term, instructions: &mut Vec<Instruction>) {
    match term {
        Term::I32Const(val) => {
            instructions.push(Instruction::I32Const(*val));
        }
        Term::I64Const(val) => {
            instructions.push(Instruction::I64Const(*val));
        }
        Term::I32Add(lhs, rhs) => {
            term_to_instructions_recursive(lhs, instructions);
            term_to_instructions_recursive(rhs, instructions);
            instructions.push(Instruction::I32Add);
        }
        Term::I32Sub(lhs, rhs) => {
            term_to_instructions_recursive(lhs, instructions);
            term_to_instructions_recursive(rhs, instructions);
            instructions.push(Instruction::I32Sub);
        }
        Term::I32Mul(lhs, rhs) => {
            term_to_instructions_recursive(lhs, instructions);
            term_to_instructions_recursive(rhs, instructions);
            instructions.push(Instruction::I32Mul);
        }
        Term::I64Add(lhs, rhs) => {
            term_to_instructions_recursive(lhs, instructions);
            term_to_instructions_recursive(rhs, instructions);
            instructions.push(Instruction::I64Add);
        }
        Term::I64Sub(lhs, rhs) => {
            term_to_instructions_recursive(lhs, instructions);
            term_to_instructions_recursive(rhs, instructions);
            instructions.push(Instruction::I64Sub);
        }
        Term::I64Mul(lhs, rhs) => {
            term_to_instructions_recursive(lhs, instructions);
            term_to_instructions_recursive(rhs, instructions);
            instructions.push(Instruction::I64Mul);
        }
        Term::Drop(inner) => {
            term_to_instructions_recursive(inner, instructions);
            instructions.push(Instruction::Drop);
        }
        Term::Nop => {
            instructions.push(Instruction::Nop);
        }
    }
}

/// Convert terms back to instructions
pub fn terms_to_instructions(terms: &[Term]) -> Vec<Instruction> {
    let mut instructions = Vec::new();
    for term in terms {
        term_to_instructions_recursive(term, &mut instructions);
    }
    instructions
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_const() {
        let instrs = vec![Instruction::I32Const(42)];
        let terms = instructions_to_terms(&instrs).unwrap();
        let back = terms_to_instructions(&terms);
        assert_eq!(instrs, back);
    }

    #[test]
    fn test_roundtrip_add() {
        // i32.const 1; i32.const 2; i32.add
        let instrs = vec![
            Instruction::I32Const(1),
            Instruction::I32Const(2),
            Instruction::I32Add,
        ];
        let terms = instructions_to_terms(&instrs).unwrap();
        let back = terms_to_instructions(&terms);
        assert_eq!(instrs, back);
    }

    #[test]
    fn test_roundtrip_complex() {
        // (1 + 2) * (3 - 4)
        let instrs = vec![
            Instruction::I32Const(1),
            Instruction::I32Const(2),
            Instruction::I32Add,
            Instruction::I32Const(3),
            Instruction::I32Const(4),
            Instruction::I32Sub,
            Instruction::I32Mul,
        ];
        let terms = instructions_to_terms(&instrs).unwrap();
        let back = terms_to_instructions(&terms);
        assert_eq!(instrs, back);
    }
}
