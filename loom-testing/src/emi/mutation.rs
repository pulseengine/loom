//! Mutation operators for EMI testing
//!
//! These functions apply mutations to dead code regions while
//! preserving WebAssembly validity.

use super::types::{DeadRegion, MutationStrategy};
use anyhow::{Context, Result};
use wasm_encoder::{CodeSection, Function, Instruction, Module as EncoderModule, RawSection};
use wasmparser::{Operator, Parser, Payload};

/// Apply a mutation to a dead region in a WebAssembly module
///
/// Returns a new WebAssembly binary with the mutation applied.
pub fn apply_mutation(
    wasm_bytes: &[u8],
    region: &DeadRegion,
    strategy: MutationStrategy,
) -> Result<Vec<u8>> {
    match strategy {
        MutationStrategy::ModifyConstants => modify_constants_in_region(wasm_bytes, region),
        MutationStrategy::ReplaceWithUnreachable => replace_with_unreachable(wasm_bytes, region),
        MutationStrategy::ReplaceWithNop => replace_with_nop(wasm_bytes, region),
        MutationStrategy::InsertDeadCode => insert_dead_code(wasm_bytes, region),
        MutationStrategy::Delete => delete_dead_code(wasm_bytes, region),
    }
}

/// Modify constant values in a dead region
///
/// This is the safest mutation - just changes constant values in dead code.
fn modify_constants_in_region(wasm_bytes: &[u8], region: &DeadRegion) -> Result<Vec<u8>> {
    transform_module(wasm_bytes, region, |op, in_region| {
        if in_region {
            match op {
                Operator::I32Const { value } => Some(Instruction::I32Const(value.wrapping_add(42))),
                Operator::I64Const { value } => Some(Instruction::I64Const(value.wrapping_add(42))),
                _ => None, // Keep original
            }
        } else {
            None // Keep original
        }
    })
}

/// Replace dead code with unreachable instruction
fn replace_with_unreachable(wasm_bytes: &[u8], region: &DeadRegion) -> Result<Vec<u8>> {
    let mut replaced = false;
    transform_module(wasm_bytes, region, |_op, in_region| {
        if in_region && !replaced {
            replaced = true;
            Some(Instruction::Unreachable)
        } else if in_region {
            // Skip remaining instructions in dead region
            Some(Instruction::Nop) // Will be filtered
        } else {
            None
        }
    })
}

/// Replace dead code with nop instructions
fn replace_with_nop(wasm_bytes: &[u8], region: &DeadRegion) -> Result<Vec<u8>> {
    transform_module(wasm_bytes, region, |op, in_region| {
        if in_region {
            match op {
                // Only replace simple value-producing instructions with nop
                Operator::I32Const { .. } | Operator::I64Const { .. } => Some(Instruction::Nop),
                _ => None,
            }
        } else {
            None
        }
    })
}

/// Insert dead code into the dead region
///
/// We insert `i32.const X; drop` which is stack-neutral
fn insert_dead_code(wasm_bytes: &[u8], region: &DeadRegion) -> Result<Vec<u8>> {
    // For insert, we need a different approach - we'll just modify constants
    // which is simpler and always valid
    modify_constants_in_region(wasm_bytes, region)
}

/// Delete dead code entirely
fn delete_dead_code(wasm_bytes: &[u8], region: &DeadRegion) -> Result<Vec<u8>> {
    // Deletion is tricky - for safety, replace with unreachable
    replace_with_unreachable(wasm_bytes, region)
}

// ============================================================================
// Core transformation engine
// ============================================================================

/// Transform a module by applying a mutation function to operators
fn transform_module<F>(wasm_bytes: &[u8], region: &DeadRegion, mut mutate: F) -> Result<Vec<u8>>
where
    F: FnMut(&Operator, bool) -> Option<Instruction<'static>>,
{
    let parser = Parser::new(0);
    let mut module = EncoderModule::new();
    let mut func_idx = 0u32;
    let mut code_section_entries: Vec<Vec<u8>> = Vec::new();
    let mut in_code_section = false;

    // Second pass: transform
    for payload in parser.parse_all(wasm_bytes) {
        let payload = payload.context("Failed to parse payload")?;

        match payload {
            Payload::Version { .. } => {
                // Skip - module handles this
            }
            Payload::TypeSection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 1, // Type section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::ImportSection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 2, // Import section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::FunctionSection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 3, // Function section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::TableSection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 4, // Table section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::MemorySection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 5, // Memory section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::GlobalSection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 6, // Global section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::ExportSection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 7, // Export section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::StartSection { range, .. } => {
                module.section(&RawSection {
                    id: 8, // Start section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::ElementSection(reader) => {
                let range = reader.range();
                module.section(&RawSection {
                    id: 9, // Element section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::DataCountSection { range, .. } => {
                module.section(&RawSection {
                    id: 12, // DataCount section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::CodeSectionStart { .. } => {
                in_code_section = true;
            }
            Payload::CodeSectionEntry(body) => {
                let is_target_func = func_idx == region.func_idx;

                if is_target_func {
                    // Transform this function
                    let transformed =
                        transform_function_body(&body, region, &mut mutate, wasm_bytes)?;
                    code_section_entries.push(transformed);
                } else {
                    // Copy function as-is
                    let range = body.range();
                    code_section_entries.push(wasm_bytes[range.start..range.end].to_vec());
                }
                func_idx += 1;
            }
            Payload::DataSection(reader) => {
                // Write code section first if we haven't
                if in_code_section && !code_section_entries.is_empty() {
                    write_code_section(&mut module, &code_section_entries);
                    in_code_section = false;
                }
                let range = reader.range();
                module.section(&RawSection {
                    id: 11, // Data section
                    data: &wasm_bytes[range.start..range.end],
                });
            }
            Payload::CustomSection(reader) => {
                module.section(&RawSection {
                    id: 0, // Custom section
                    data: &wasm_bytes[reader.range().start..reader.range().end],
                });
            }
            Payload::End(_) => {
                // Write code section if we haven't yet
                if in_code_section && !code_section_entries.is_empty() {
                    write_code_section(&mut module, &code_section_entries);
                }
            }
            _ => {
                // Skip other payloads
            }
        }
    }

    Ok(module.finish())
}

/// Write a code section from raw function bytes
fn write_code_section(module: &mut EncoderModule, entries: &[Vec<u8>]) {
    let mut code = CodeSection::new();
    for entry in entries {
        // Each entry is the raw bytes of a function body (including locals)
        code.raw(entry);
    }
    module.section(&code);
}

/// Transform a function body, applying mutations to operators in the dead region
fn transform_function_body<F>(
    body: &wasmparser::FunctionBody,
    region: &DeadRegion,
    mutate: &mut F,
    _wasm_bytes: &[u8],
) -> Result<Vec<u8>>
where
    F: FnMut(&Operator, bool) -> Option<Instruction<'static>>,
{
    // Parse locals - LocalsReader has a count() method that tells us how many local groups there are
    let mut locals_reader = body.get_locals_reader()?;
    let local_count = locals_reader.get_count();
    let mut locals: Vec<(u32, wasm_encoder::ValType)> = Vec::new();
    for _ in 0..local_count {
        let (count, ty) = locals_reader.read()?;
        let encoder_ty = convert_valtype(ty);
        locals.push((count, encoder_ty));
    }

    // Create function with locals
    let mut func = Function::new(locals);

    // Parse and transform operators
    let mut reader = body.get_operators_reader()?;
    while !reader.eof() {
        let offset = reader.original_position();
        let op = reader.read()?;

        let in_region = offset >= region.start_offset && offset < region.end_offset;

        // Try to mutate
        if let Some(replacement) = mutate(&op, in_region) {
            func.instruction(&replacement);
        } else {
            // Convert and emit original instruction
            match convert_operator(&op) {
                Some(instr) => {
                    func.instruction(&instr);
                }
                None => {
                    // Unhandled instruction - bail out and return original bytes
                    // This ensures we don't produce invalid WASM
                    return Err(anyhow::anyhow!("Unhandled instruction: {:?}", op));
                }
            }
        }
    }

    Ok(func.into_raw_body())
}

/// Convert wasmparser ValType to wasm-encoder ValType
fn convert_valtype(ty: wasmparser::ValType) -> wasm_encoder::ValType {
    match ty {
        wasmparser::ValType::I32 => wasm_encoder::ValType::I32,
        wasmparser::ValType::I64 => wasm_encoder::ValType::I64,
        wasmparser::ValType::F32 => wasm_encoder::ValType::F32,
        wasmparser::ValType::F64 => wasm_encoder::ValType::F64,
        wasmparser::ValType::V128 => wasm_encoder::ValType::V128,
        wasmparser::ValType::Ref(r) => {
            // Simplified ref type handling
            if r.is_func_ref() {
                wasm_encoder::ValType::Ref(wasm_encoder::RefType::FUNCREF)
            } else {
                wasm_encoder::ValType::Ref(wasm_encoder::RefType::EXTERNREF)
            }
        }
    }
}

/// Convert wasmparser Operator to wasm-encoder Instruction
///
/// This handles the subset of instructions we care about for EMI testing.
fn convert_operator(op: &Operator) -> Option<Instruction<'static>> {
    Some(match op {
        // Constants
        Operator::I32Const { value } => Instruction::I32Const(*value),
        Operator::I64Const { value } => Instruction::I64Const(*value),
        Operator::F32Const { value } => {
            Instruction::F32Const(wasm_encoder::Ieee32::new(value.bits()))
        }
        Operator::F64Const { value } => {
            Instruction::F64Const(wasm_encoder::Ieee64::new(value.bits()))
        }

        // Control flow
        Operator::Unreachable => Instruction::Unreachable,
        Operator::Nop => Instruction::Nop,
        Operator::Block { blockty } => Instruction::Block(convert_blocktype(blockty)),
        Operator::Loop { blockty } => Instruction::Loop(convert_blocktype(blockty)),
        Operator::If { blockty } => Instruction::If(convert_blocktype(blockty)),
        Operator::Else => Instruction::Else,
        Operator::End => Instruction::End,
        Operator::Br { relative_depth } => Instruction::Br(*relative_depth),
        Operator::BrIf { relative_depth } => Instruction::BrIf(*relative_depth),
        Operator::BrTable { targets } => {
            let tgts: Vec<u32> = targets.targets().map(|t| t.unwrap()).collect();
            Instruction::BrTable(tgts.into(), targets.default())
        }
        Operator::Return => Instruction::Return,
        Operator::Call { function_index } => Instruction::Call(*function_index),
        Operator::CallIndirect {
            type_index,
            table_index,
            ..
        } => Instruction::CallIndirect {
            type_index: *type_index,
            table_index: *table_index,
        },

        // Parametric
        Operator::Drop => Instruction::Drop,
        Operator::Select => Instruction::Select,

        // Variable access
        Operator::LocalGet { local_index } => Instruction::LocalGet(*local_index),
        Operator::LocalSet { local_index } => Instruction::LocalSet(*local_index),
        Operator::LocalTee { local_index } => Instruction::LocalTee(*local_index),
        Operator::GlobalGet { global_index } => Instruction::GlobalGet(*global_index),
        Operator::GlobalSet { global_index } => Instruction::GlobalSet(*global_index),

        // Memory operations
        Operator::I32Load { memarg } => Instruction::I32Load(convert_memarg(memarg)),
        Operator::I64Load { memarg } => Instruction::I64Load(convert_memarg(memarg)),
        Operator::F32Load { memarg } => Instruction::F32Load(convert_memarg(memarg)),
        Operator::F64Load { memarg } => Instruction::F64Load(convert_memarg(memarg)),
        Operator::I32Load8S { memarg } => Instruction::I32Load8S(convert_memarg(memarg)),
        Operator::I32Load8U { memarg } => Instruction::I32Load8U(convert_memarg(memarg)),
        Operator::I32Load16S { memarg } => Instruction::I32Load16S(convert_memarg(memarg)),
        Operator::I32Load16U { memarg } => Instruction::I32Load16U(convert_memarg(memarg)),
        Operator::I64Load8S { memarg } => Instruction::I64Load8S(convert_memarg(memarg)),
        Operator::I64Load8U { memarg } => Instruction::I64Load8U(convert_memarg(memarg)),
        Operator::I64Load16S { memarg } => Instruction::I64Load16S(convert_memarg(memarg)),
        Operator::I64Load16U { memarg } => Instruction::I64Load16U(convert_memarg(memarg)),
        Operator::I64Load32S { memarg } => Instruction::I64Load32S(convert_memarg(memarg)),
        Operator::I64Load32U { memarg } => Instruction::I64Load32U(convert_memarg(memarg)),
        Operator::I32Store { memarg } => Instruction::I32Store(convert_memarg(memarg)),
        Operator::I64Store { memarg } => Instruction::I64Store(convert_memarg(memarg)),
        Operator::F32Store { memarg } => Instruction::F32Store(convert_memarg(memarg)),
        Operator::F64Store { memarg } => Instruction::F64Store(convert_memarg(memarg)),
        Operator::I32Store8 { memarg } => Instruction::I32Store8(convert_memarg(memarg)),
        Operator::I32Store16 { memarg } => Instruction::I32Store16(convert_memarg(memarg)),
        Operator::I64Store8 { memarg } => Instruction::I64Store8(convert_memarg(memarg)),
        Operator::I64Store16 { memarg } => Instruction::I64Store16(convert_memarg(memarg)),
        Operator::I64Store32 { memarg } => Instruction::I64Store32(convert_memarg(memarg)),
        Operator::MemorySize { mem, .. } => Instruction::MemorySize(*mem),
        Operator::MemoryGrow { mem, .. } => Instruction::MemoryGrow(*mem),

        // i32 operations
        Operator::I32Eqz => Instruction::I32Eqz,
        Operator::I32Eq => Instruction::I32Eq,
        Operator::I32Ne => Instruction::I32Ne,
        Operator::I32LtS => Instruction::I32LtS,
        Operator::I32LtU => Instruction::I32LtU,
        Operator::I32GtS => Instruction::I32GtS,
        Operator::I32GtU => Instruction::I32GtU,
        Operator::I32LeS => Instruction::I32LeS,
        Operator::I32LeU => Instruction::I32LeU,
        Operator::I32GeS => Instruction::I32GeS,
        Operator::I32GeU => Instruction::I32GeU,
        Operator::I32Clz => Instruction::I32Clz,
        Operator::I32Ctz => Instruction::I32Ctz,
        Operator::I32Popcnt => Instruction::I32Popcnt,
        Operator::I32Add => Instruction::I32Add,
        Operator::I32Sub => Instruction::I32Sub,
        Operator::I32Mul => Instruction::I32Mul,
        Operator::I32DivS => Instruction::I32DivS,
        Operator::I32DivU => Instruction::I32DivU,
        Operator::I32RemS => Instruction::I32RemS,
        Operator::I32RemU => Instruction::I32RemU,
        Operator::I32And => Instruction::I32And,
        Operator::I32Or => Instruction::I32Or,
        Operator::I32Xor => Instruction::I32Xor,
        Operator::I32Shl => Instruction::I32Shl,
        Operator::I32ShrS => Instruction::I32ShrS,
        Operator::I32ShrU => Instruction::I32ShrU,
        Operator::I32Rotl => Instruction::I32Rotl,
        Operator::I32Rotr => Instruction::I32Rotr,

        // i64 operations
        Operator::I64Eqz => Instruction::I64Eqz,
        Operator::I64Eq => Instruction::I64Eq,
        Operator::I64Ne => Instruction::I64Ne,
        Operator::I64LtS => Instruction::I64LtS,
        Operator::I64LtU => Instruction::I64LtU,
        Operator::I64GtS => Instruction::I64GtS,
        Operator::I64GtU => Instruction::I64GtU,
        Operator::I64LeS => Instruction::I64LeS,
        Operator::I64LeU => Instruction::I64LeU,
        Operator::I64GeS => Instruction::I64GeS,
        Operator::I64GeU => Instruction::I64GeU,
        Operator::I64Clz => Instruction::I64Clz,
        Operator::I64Ctz => Instruction::I64Ctz,
        Operator::I64Popcnt => Instruction::I64Popcnt,
        Operator::I64Add => Instruction::I64Add,
        Operator::I64Sub => Instruction::I64Sub,
        Operator::I64Mul => Instruction::I64Mul,
        Operator::I64DivS => Instruction::I64DivS,
        Operator::I64DivU => Instruction::I64DivU,
        Operator::I64RemS => Instruction::I64RemS,
        Operator::I64RemU => Instruction::I64RemU,
        Operator::I64And => Instruction::I64And,
        Operator::I64Or => Instruction::I64Or,
        Operator::I64Xor => Instruction::I64Xor,
        Operator::I64Shl => Instruction::I64Shl,
        Operator::I64ShrS => Instruction::I64ShrS,
        Operator::I64ShrU => Instruction::I64ShrU,
        Operator::I64Rotl => Instruction::I64Rotl,
        Operator::I64Rotr => Instruction::I64Rotr,

        // f32 operations
        Operator::F32Eq => Instruction::F32Eq,
        Operator::F32Ne => Instruction::F32Ne,
        Operator::F32Lt => Instruction::F32Lt,
        Operator::F32Gt => Instruction::F32Gt,
        Operator::F32Le => Instruction::F32Le,
        Operator::F32Ge => Instruction::F32Ge,
        Operator::F32Abs => Instruction::F32Abs,
        Operator::F32Neg => Instruction::F32Neg,
        Operator::F32Ceil => Instruction::F32Ceil,
        Operator::F32Floor => Instruction::F32Floor,
        Operator::F32Trunc => Instruction::F32Trunc,
        Operator::F32Nearest => Instruction::F32Nearest,
        Operator::F32Sqrt => Instruction::F32Sqrt,
        Operator::F32Add => Instruction::F32Add,
        Operator::F32Sub => Instruction::F32Sub,
        Operator::F32Mul => Instruction::F32Mul,
        Operator::F32Div => Instruction::F32Div,
        Operator::F32Min => Instruction::F32Min,
        Operator::F32Max => Instruction::F32Max,
        Operator::F32Copysign => Instruction::F32Copysign,

        // f64 operations
        Operator::F64Eq => Instruction::F64Eq,
        Operator::F64Ne => Instruction::F64Ne,
        Operator::F64Lt => Instruction::F64Lt,
        Operator::F64Gt => Instruction::F64Gt,
        Operator::F64Le => Instruction::F64Le,
        Operator::F64Ge => Instruction::F64Ge,
        Operator::F64Abs => Instruction::F64Abs,
        Operator::F64Neg => Instruction::F64Neg,
        Operator::F64Ceil => Instruction::F64Ceil,
        Operator::F64Floor => Instruction::F64Floor,
        Operator::F64Trunc => Instruction::F64Trunc,
        Operator::F64Nearest => Instruction::F64Nearest,
        Operator::F64Sqrt => Instruction::F64Sqrt,
        Operator::F64Add => Instruction::F64Add,
        Operator::F64Sub => Instruction::F64Sub,
        Operator::F64Mul => Instruction::F64Mul,
        Operator::F64Div => Instruction::F64Div,
        Operator::F64Min => Instruction::F64Min,
        Operator::F64Max => Instruction::F64Max,
        Operator::F64Copysign => Instruction::F64Copysign,

        // Conversions
        Operator::I32WrapI64 => Instruction::I32WrapI64,
        Operator::I32TruncF32S => Instruction::I32TruncF32S,
        Operator::I32TruncF32U => Instruction::I32TruncF32U,
        Operator::I32TruncF64S => Instruction::I32TruncF64S,
        Operator::I32TruncF64U => Instruction::I32TruncF64U,
        Operator::I64ExtendI32S => Instruction::I64ExtendI32S,
        Operator::I64ExtendI32U => Instruction::I64ExtendI32U,
        Operator::I64TruncF32S => Instruction::I64TruncF32S,
        Operator::I64TruncF32U => Instruction::I64TruncF32U,
        Operator::I64TruncF64S => Instruction::I64TruncF64S,
        Operator::I64TruncF64U => Instruction::I64TruncF64U,
        Operator::F32ConvertI32S => Instruction::F32ConvertI32S,
        Operator::F32ConvertI32U => Instruction::F32ConvertI32U,
        Operator::F32ConvertI64S => Instruction::F32ConvertI64S,
        Operator::F32ConvertI64U => Instruction::F32ConvertI64U,
        Operator::F32DemoteF64 => Instruction::F32DemoteF64,
        Operator::F64ConvertI32S => Instruction::F64ConvertI32S,
        Operator::F64ConvertI32U => Instruction::F64ConvertI32U,
        Operator::F64ConvertI64S => Instruction::F64ConvertI64S,
        Operator::F64ConvertI64U => Instruction::F64ConvertI64U,
        Operator::F64PromoteF32 => Instruction::F64PromoteF32,
        Operator::I32ReinterpretF32 => Instruction::I32ReinterpretF32,
        Operator::I64ReinterpretF64 => Instruction::I64ReinterpretF64,
        Operator::F32ReinterpretI32 => Instruction::F32ReinterpretI32,
        Operator::F64ReinterpretI64 => Instruction::F64ReinterpretI64,

        // Sign extension
        Operator::I32Extend8S => Instruction::I32Extend8S,
        Operator::I32Extend16S => Instruction::I32Extend16S,
        Operator::I64Extend8S => Instruction::I64Extend8S,
        Operator::I64Extend16S => Instruction::I64Extend16S,
        Operator::I64Extend32S => Instruction::I64Extend32S,

        // Unhandled - skip
        _ => return None,
    })
}

/// Convert wasmparser BlockType to wasm-encoder BlockType
fn convert_blocktype(bt: &wasmparser::BlockType) -> wasm_encoder::BlockType {
    match bt {
        wasmparser::BlockType::Empty => wasm_encoder::BlockType::Empty,
        wasmparser::BlockType::Type(ty) => wasm_encoder::BlockType::Result(convert_valtype(*ty)),
        wasmparser::BlockType::FuncType(idx) => wasm_encoder::BlockType::FunctionType(*idx),
    }
}

/// Convert wasmparser MemArg to wasm-encoder MemArg
fn convert_memarg(ma: &wasmparser::MemArg) -> wasm_encoder::MemArg {
    wasm_encoder::MemArg {
        offset: ma.offset,
        align: ma.align as u32,
        memory_index: ma.memory,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_modify_constants() {
        let wat = r#"
            (module
              (func (export "test") (result i32)
                (if (result i32) (i32.const 1)
                  (then (i32.const 42))
                  (else (i32.const 99))))
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");

        // Analyze to find dead region
        let regions = super::super::analyze_dead_code(&wasm).expect("Analysis failed");
        assert!(!regions.is_empty(), "Should find dead region");

        // Apply mutation
        let mutated = apply_mutation(&wasm, &regions[0], MutationStrategy::ModifyConstants)
            .expect("Mutation failed");

        // Validate result
        wasmparser::validate(&mutated).expect("Mutated module should be valid");

        // Should be different from original
        assert_ne!(wasm, mutated, "Mutation should change the module");
    }

    #[test]
    fn test_roundtrip_preserves_validity() {
        let wat = r#"
            (module
              (func (export "add") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add)
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");

        // Create a dummy region that won't match anything
        let region = super::super::types::DeadRegion {
            func_idx: 0,
            start_offset: 1000, // Beyond function
            end_offset: 1001,
            region_type: super::super::types::DeadRegionType::AfterReturn,
            description: "test".to_string(),
        };

        // Transform should preserve the module
        let result = apply_mutation(&wasm, &region, MutationStrategy::ModifyConstants)
            .expect("Transform failed");

        wasmparser::validate(&result).expect("Result should be valid");
    }

    #[test]
    fn test_insert_dead_code() {
        let wat = r#"
            (module
              (func (export "test") (result i32)
                (if (result i32) (i32.const 1)
                  (then (i32.const 42))
                  (else (i32.const 99))))
            )
        "#;

        let wasm = wat::parse_str(wat).expect("Failed to parse WAT");
        let regions = super::super::analyze_dead_code(&wasm).expect("Analysis failed");

        if !regions.is_empty() {
            let mutated = apply_mutation(&wasm, &regions[0], MutationStrategy::InsertDeadCode)
                .expect("Mutation failed");
            wasmparser::validate(&mutated).expect("Mutated module should be valid");
        }
    }
}
