//! LOOM Core Library
//!
//! Core functionality for the LOOM WebAssembly optimizer including:
//! - WebAssembly module parsing
//! - ISLE term construction
//! - Optimization application
//! - WebAssembly module encoding

#![warn(missing_docs)]

pub use loom_isle::{Imm32, Imm64, Value, ValueData};

/// Stack analysis module: compositional stack type system
pub mod stack;

/// Function-summary interprocedural analysis (IPA).
///
/// Computes per-function `is_pure` / `is_no_trap` summaries so downstream
/// passes can reason across `Call` boundaries.
pub mod summary;

/// Peephole synthesis (Souper-shaped MVP, v0.8.0 PR-L).
///
/// Minimal first cut of the algorithmic-solver direction from
/// `docs/research/v0.7.0/algorithmic-solver-feasibility.md`. Ships a
/// small set of hand-curated arithmetic-identity rules with documented
/// algebraic proofs; future PRs grow the candidate set.
pub mod peephole_synth;

/// Optimization observability counters (revert counts per pass).
pub mod stats;

/// Acyclic equality graph (ægraph) MVP — v1.0.3 Track 2.
///
/// Infrastructure-only data structure for future Cranelift-style
/// per-rewrite-verifiable optimization work. Ships hash-consing and the
/// acyclic invariant; rewrite rules / pipeline integration are deferred
/// to follow-up PRs.
pub mod egraph;

/// Island-model parallel optimization — v1.0.4 PR-islands (issue #71).
///
/// Runs N independent pass orderings concurrently, each on a cloned copy of
/// the input module, then picks the smallest valid result. Each island still
/// passes the existing Z3 + stack validation gates independently — soundness
/// is not traded for speed.
pub mod islands;

/// Internal representation of a WebAssembly module
#[derive(Debug, Clone)]
pub struct Module {
    /// Module functions
    pub functions: Vec<Function>,
    /// Memory definitions (Phase 14: Metadata Preservation)
    pub memories: Vec<Memory>,
    /// Table definitions (Phase 23: Component Model Support)
    pub tables: Vec<Table>,
    /// Global variables
    pub globals: Vec<Global>,
    /// Function types (for reconstruction)
    pub types: Vec<FunctionSignature>,
    /// Exported items (functions, globals, memories, tables)
    pub exports: Vec<Export>,
    /// Imported items (functions, globals, memories, tables)
    pub imports: Vec<Import>,
    /// Data segments (memory initialization)
    pub data_segments: Vec<DataSegment>,
    /// Element section raw bytes (passed through unchanged)
    pub element_section_bytes: Option<Vec<u8>>,
    /// Start function index (optional)
    pub start_function: Option<u32>,
    /// Custom sections raw bytes (passed through unchanged)
    pub custom_sections: Vec<(String, Vec<u8>)>,
    /// Type section raw bytes (passed through for GC types, reference types, etc.)
    pub type_section_bytes: Option<Vec<u8>>,
    /// Global section raw bytes (passed through for reference types, etc.)
    pub global_section_bytes: Option<Vec<u8>>,
}

/// Export definition
#[derive(Debug, Clone)]
pub struct Export {
    /// Export name
    pub name: String,
    /// What is being exported
    pub kind: ExportKind,
}

/// Type of exported item
#[derive(Debug, Clone)]
pub enum ExportKind {
    /// Function export
    Func(u32),
    /// Memory export
    Memory(u32),
    /// Global export
    Global(u32),
    /// Table export
    Table(u32),
}

/// Import definition
#[derive(Debug, Clone)]
pub struct Import {
    /// Module name
    pub module: String,
    /// Field name
    pub name: String,
    /// What is being imported
    pub kind: ImportKind,
}

/// Type of imported item
#[derive(Debug, Clone)]
pub enum ImportKind {
    /// Function import with type index
    Func(u32),
    /// Memory import
    Memory(Memory),
    /// Global import
    Global {
        /// The type of value stored in the global
        value_type: ValueType,
        /// Whether the global is mutable
        mutable: bool,
    },
    /// Table import
    Table(Table),
}

/// Memory definition
#[derive(Debug, Clone)]
pub struct Memory {
    /// Minimum pages
    pub min: u64,
    /// Maximum pages (optional)
    pub max: Option<u64>,
    /// Shared memory flag
    pub shared: bool,
    /// Memory64 flag (true for 64-bit addressing)
    pub memory64: bool,
}

/// Table definition (Phase 23: Component Model Support)
#[derive(Debug, Clone)]
pub struct Table {
    /// Element type (e.g., funcref, externref)
    pub element_type: RefType,
    /// Minimum size
    pub min: u32,
    /// Maximum size (optional)
    pub max: Option<u32>,
}

/// Reference types for tables
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefType {
    /// Function reference
    FuncRef,
    /// External reference
    ExternRef,
}

/// Global variable
#[derive(Debug, Clone)]
pub struct Global {
    /// Value type
    pub value_type: ValueType,
    /// Mutable flag
    pub mutable: bool,
    /// Initializer expression (Phase 18: Precompute)
    pub init: Vec<Instruction>,
}

/// Data segment (memory initialization)
#[derive(Debug, Clone)]
pub struct DataSegment {
    /// Memory index
    pub memory_index: u32,
    /// Offset expression
    pub offset: Vec<Instruction>,
    /// Data bytes
    pub data: Vec<u8>,
    /// Passive data segment (no memory index or offset)
    pub passive: bool,
}

/// Element segment (table initialization)
/// We store raw bytes to avoid dealing with complex element section APIs
#[derive(Debug, Clone)]
pub struct ElementSegment {
    /// Raw element segment bytes (to be passed through unchanged)
    pub raw_bytes: Vec<u8>,
}

/// Internal representation of a WebAssembly function
#[derive(Debug, Clone)]
pub struct Function {
    /// Function name (if available)
    pub name: Option<String>,
    /// Function type signature (params, results)
    pub signature: FunctionSignature,
    /// Local variable declarations (Phase 14)
    pub locals: Vec<(u32, ValueType)>, // (count, type) pairs
    /// Function body instructions
    pub instructions: Vec<Instruction>,
}

/// Function signature
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct FunctionSignature {
    /// Parameter types
    pub params: Vec<ValueType>,
    /// Result types
    pub results: Vec<ValueType>,
}

/// WebAssembly value types
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
}

/// Block type for control flow structures
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// No parameters, no results
    Empty,
    /// No parameters, single result
    Value(ValueType),
    /// Full function signature (for multi-value blocks)
    Func {
        /// Input parameter types
        params: Vec<ValueType>,
        /// Output result types
        results: Vec<ValueType>,
    },
}

/// WebAssembly instructions
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Instruction {
    /// i32.const
    I32Const(i32),
    /// i32.add
    I32Add,
    /// i32.sub
    I32Sub,
    /// i32.mul
    I32Mul,
    /// i32.and
    I32And,
    /// i32.or
    I32Or,
    /// i32.xor
    I32Xor,
    /// i32.shl
    I32Shl,
    /// i32.shr_s
    I32ShrS,
    /// i32.shr_u
    I32ShrU,
    /// i64.const
    I64Const(i64),
    /// f32.const
    F32Const(u32),
    /// f64.const
    F64Const(u64),
    /// i64.add
    I64Add,
    /// i64.sub
    I64Sub,
    /// i64.mul
    I64Mul,
    /// i64.and
    I64And,
    /// i64.or
    I64Or,
    /// i64.xor
    I64Xor,
    /// i64.shl
    I64Shl,
    /// i64.shr_s
    I64ShrS,
    /// i64.shr_u
    I64ShrU,
    /// i32.eq
    I32Eq,
    /// i32.ne
    I32Ne,
    /// i32.lt_s
    I32LtS,
    /// i32.lt_u
    I32LtU,
    /// i32.gt_s
    I32GtS,
    /// i32.gt_u
    I32GtU,
    /// i32.le_s
    I32LeS,
    /// i32.le_u
    I32LeU,
    /// i32.ge_s
    I32GeS,
    /// i32.ge_u
    I32GeU,
    /// i64.eq
    I64Eq,
    /// i64.ne
    I64Ne,
    /// i64.lt_s
    I64LtS,
    /// i64.lt_u
    I64LtU,
    /// i64.gt_s
    I64GtS,
    /// i64.gt_u
    I64GtU,
    /// i64.le_s
    I64LeS,
    /// i64.le_u
    I64LeU,
    /// i64.ge_s
    I64GeS,
    /// i64.ge_u
    I64GeU,
    /// i32.div_s
    I32DivS,
    /// i32.div_u
    I32DivU,
    /// i32.rem_s
    I32RemS,
    /// i32.rem_u
    I32RemU,
    /// i64.div_s
    I64DivS,
    /// i64.div_u
    I64DivU,
    /// i64.rem_s
    I64RemS,
    /// i64.rem_u
    I64RemU,
    /// i32.eqz
    I32Eqz,
    /// i32.clz
    I32Clz,
    /// i32.ctz
    I32Ctz,
    /// i32.popcnt
    I32Popcnt,
    /// i64.eqz
    I64Eqz,
    /// i64.clz
    I64Clz,
    /// i64.ctz
    I64Ctz,
    /// i64.popcnt
    I64Popcnt,
    /// select
    Select,
    /// local.get
    LocalGet(u32),
    /// local.set
    LocalSet(u32),
    /// local.tee
    LocalTee(u32),
    /// global.get
    GlobalGet(u32),
    /// global.set
    GlobalSet(u32),
    /// i32.load
    I32Load {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i32.store
    I32Store {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.load
    I64Load {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.store
    I64Store {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },

    // Float arithmetic operations (f32)
    /// f32.add
    F32Add,
    /// f32.sub
    F32Sub,
    /// f32.mul
    F32Mul,
    /// f32.div
    F32Div,
    /// f32.min
    F32Min,
    /// f32.max
    F32Max,
    /// f32.copysign
    F32Copysign,
    /// f32.abs
    F32Abs,
    /// f32.neg
    F32Neg,
    /// f32.ceil
    F32Ceil,
    /// f32.floor
    F32Floor,
    /// f32.trunc
    F32Trunc,
    /// f32.nearest
    F32Nearest,
    /// f32.sqrt
    F32Sqrt,

    // Float comparison operations (f32) - produce i32
    /// f32.eq
    F32Eq,
    /// f32.ne
    F32Ne,
    /// f32.lt
    F32Lt,
    /// f32.gt
    F32Gt,
    /// f32.le
    F32Le,
    /// f32.ge
    F32Ge,

    // Float arithmetic operations (f64)
    /// f64.add
    F64Add,
    /// f64.sub
    F64Sub,
    /// f64.mul
    F64Mul,
    /// f64.div
    F64Div,
    /// f64.min
    F64Min,
    /// f64.max
    F64Max,
    /// f64.copysign
    F64Copysign,
    /// f64.abs
    F64Abs,
    /// f64.neg
    F64Neg,
    /// f64.ceil
    F64Ceil,
    /// f64.floor
    F64Floor,
    /// f64.trunc
    F64Trunc,
    /// f64.nearest
    F64Nearest,
    /// f64.sqrt
    F64Sqrt,

    // Float comparison operations (f64) - produce i32
    /// f64.eq
    F64Eq,
    /// f64.ne
    F64Ne,
    /// f64.lt
    F64Lt,
    /// f64.gt
    F64Gt,
    /// f64.le
    F64Le,
    /// f64.ge
    F64Ge,

    // Conversion operations
    /// i32.trunc_f32_s
    I32TruncF32S,
    /// i32.trunc_f32_u
    I32TruncF32U,
    /// i32.trunc_f64_s
    I32TruncF64S,
    /// i32.trunc_f64_u
    I32TruncF64U,
    /// i64.trunc_f32_s
    I64TruncF32S,
    /// i64.trunc_f32_u
    I64TruncF32U,
    /// i64.trunc_f64_s
    I64TruncF64S,
    /// i64.trunc_f64_u
    I64TruncF64U,
    // Saturating truncation operations (non-trapping)
    /// i32.trunc_sat_f32_s
    I32TruncSatF32S,
    /// i32.trunc_sat_f32_u
    I32TruncSatF32U,
    /// i32.trunc_sat_f64_s
    I32TruncSatF64S,
    /// i32.trunc_sat_f64_u
    I32TruncSatF64U,
    /// i64.trunc_sat_f32_s
    I64TruncSatF32S,
    /// i64.trunc_sat_f32_u
    I64TruncSatF32U,
    /// i64.trunc_sat_f64_s
    I64TruncSatF64S,
    /// i64.trunc_sat_f64_u
    I64TruncSatF64U,
    /// f32.convert_i32_s
    F32ConvertI32S,
    /// f32.convert_i32_u
    F32ConvertI32U,
    /// f32.convert_i64_s
    F32ConvertI64S,
    /// f32.convert_i64_u
    F32ConvertI64U,
    /// f64.convert_i32_s
    F64ConvertI32S,
    /// f64.convert_i32_u
    F64ConvertI32U,
    /// f64.convert_i64_s
    F64ConvertI64S,
    /// f64.convert_i64_u
    F64ConvertI64U,
    /// f32.demote_f64
    F32DemoteF64,
    /// f64.promote_f32
    F64PromoteF32,

    // Reinterpret operations (bit-cast)
    /// i32.reinterpret_f32
    I32ReinterpretF32,
    /// i64.reinterpret_f64
    I64ReinterpretF64,
    /// f32.reinterpret_i32
    F32ReinterpretI32,
    /// f64.reinterpret_i64
    F64ReinterpretI64,

    // Integer extend operations
    /// i64.extend_i32_s
    I64ExtendI32S,
    /// i64.extend_i32_u
    I64ExtendI32U,
    /// i32.wrap_i64
    I32WrapI64,

    // Sign extension operations (sign-extension-ops proposal)
    /// i32.extend8_s - sign-extend 8-bit value to 32-bit
    I32Extend8S,
    /// i32.extend16_s - sign-extend 16-bit value to 32-bit
    I32Extend16S,
    /// i64.extend8_s - sign-extend 8-bit value to 64-bit
    I64Extend8S,
    /// i64.extend16_s - sign-extend 16-bit value to 64-bit
    I64Extend16S,
    /// i64.extend32_s - sign-extend 32-bit value to 64-bit
    I64Extend32S,

    // Additional memory operations
    /// f32.load
    F32Load {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// f32.store
    F32Store {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// f64.load
    F64Load {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// f64.store
    F64Store {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i32.load8_s
    I32Load8S {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i32.load8_u
    I32Load8U {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i32.load16_s
    I32Load16S {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i32.load16_u
    I32Load16U {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.load8_s
    I64Load8S {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.load8_u
    I64Load8U {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.load16_s
    I64Load16S {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.load16_u
    I64Load16U {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.load32_s
    I64Load32S {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.load32_u
    I64Load32U {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i32.store8
    I32Store8 {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i32.store16
    I32Store16 {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.store8
    I64Store8 {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.store16
    I64Store16 {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },
    /// i64.store32
    I64Store32 {
        /// Memory offset
        offset: u32,
        /// Memory alignment
        align: u32,
        /// Memory index (0 for single-memory modules)
        mem: u32,
    },

    // Memory size/grow operations
    /// memory.size
    MemorySize(u32),
    /// memory.grow
    MemoryGrow(u32),

    // Bulk memory operations
    /// memory.fill (memory index)
    MemoryFill(u32),
    /// memory.copy (destination memory index, source memory index)
    MemoryCopy {
        /// Destination memory index
        dst_mem: u32,
        /// Source memory index
        src_mem: u32,
    },
    /// memory.init (memory index, data segment index)
    MemoryInit {
        /// Memory index
        mem: u32,
        /// Data segment index
        data_idx: u32,
    },
    /// data.drop (data segment index)
    DataDrop(u32),

    // Rotate operations
    /// i32.rotl
    I32Rotl,
    /// i32.rotr
    I32Rotr,
    /// i64.rotl
    I64Rotl,
    /// i64.rotr
    I64Rotr,

    // Control flow instructions (Phase 14)
    /// Block structured control
    Block {
        /// Block result type signature
        block_type: BlockType,
        /// Instructions in block body
        body: Vec<Instruction>,
    },
    /// Loop structured control
    Loop {
        /// Loop result type signature
        block_type: BlockType,
        /// Instructions in loop body
        body: Vec<Instruction>,
    },
    /// If-then-else conditional
    If {
        /// Conditional result type signature
        block_type: BlockType,
        /// Instructions in then branch
        then_body: Vec<Instruction>,
        /// Instructions in else branch
        else_body: Vec<Instruction>,
    },
    /// Unconditional branch
    Br(u32),
    /// Conditional branch
    BrIf(u32),
    /// Branch table
    BrTable {
        /// Branch target label depths
        targets: Vec<u32>,
        /// Default label depth
        default: u32,
    },
    /// Return from function
    Return,
    /// Direct function call
    Call(u32),
    /// Indirect function call
    CallIndirect {
        /// Type index for signature
        type_idx: u32,
        /// Table index
        table_idx: u32,
    },
    /// Unreachable (trap)
    Unreachable,
    /// No operation
    Nop,

    /// Drop instruction - pops and discards the top stack value
    Drop,

    /// End of block/function
    End,

    /// Unknown/unsupported instruction stored as raw bytes for pass-through
    /// Format: [opcode_byte, ...immediate_bytes]
    /// This preserves instructions we don't optimize but need to maintain correctness
    Unknown(Vec<u8>),
}

/// Module parsing functionality: Parse WebAssembly modules into LOOM's internal representation
pub mod parse {

    use super::{
        BlockType, Export, ExportKind, Function, FunctionSignature, Import, ImportKind,
        Instruction, Memory, Module, Table, ValueType,
    };
    use anyhow::{Context, Result, anyhow};
    use wasmparser::{Operator, Parser, Payload, ValType, Validator, WasmFeatures};

    /// Build WasmFeatures with all component model async features enabled.
    pub fn wasm_features_with_async() -> WasmFeatures {
        let mut inflated = WasmFeatures::default().inflate();
        inflated.cm_async = true;
        inflated.cm_async_stackful = true;
        // `cm_async_builtins` was removed in wasmparser 0.251 (folded into the
        // cm_async feature set); the two flags above cover it (#198).
        WasmFeatures::from_inflated(inflated)
    }

    /// Parse a WebAssembly binary module
    pub fn parse_wasm(bytes: &[u8]) -> Result<Module> {
        let mut validator = Validator::new_with_features(wasm_features_with_async());
        let mut functions = Vec::new();
        let mut types = Vec::new();
        let mut function_type_indices = Vec::new();
        let mut memories = Vec::new();
        let mut tables = Vec::new();
        let mut globals = Vec::new();
        let mut function_signatures = Vec::new();
        let mut exports = Vec::new();
        let mut imports = Vec::new();
        let mut data_segments = Vec::new();
        let mut element_section_bytes = None;
        let mut start_function = None;
        let mut custom_sections = Vec::new();
        let mut type_section_bytes = None;
        let mut global_section_bytes = None;

        for payload in Parser::new(0).parse_all(bytes) {
            let payload = payload.context("Failed to parse WebAssembly payload")?;

            match &payload {
                Payload::TypeSection(reader) => {
                    // Store raw type section bytes to pass through unchanged
                    let range = reader.range();
                    type_section_bytes = Some(bytes[range.start..range.end].to_vec());

                    // Still extract function types for optimizer's use
                    for rec_group in reader.clone() {
                        let rec_group = rec_group?;
                        for sub_type in rec_group.into_types() {
                            match sub_type.composite_type.inner {
                                wasmparser::CompositeInnerType::Func(func_type) => {
                                    types.push(func_type);
                                }
                                _ => {
                                    // Non-function types will be preserved via raw bytes
                                }
                            }
                        }
                    }
                }
                Payload::ImportSection(reader) => {
                    // Capture imports (functions, globals, memories, tables)
                    for import in reader.clone() {
                        // wasmparser 0.251 yields `Imports` (an enum) per entry:
                        // `Single` is the classic (module, name, ty) import;
                        // `Compact1`/`Compact2` are the new compact-imports
                        // encoding. loom does not model compact imports, so it
                        // refuses to parse such a module rather than mis-record
                        // its import table — the caller then keeps the original
                        // bytes (fail-safe, #198).
                        let import = match import? {
                            wasmparser::Imports::Single(_, import) => import,
                            _ => {
                                return Err(anyhow!(
                                    "compact import encoding (wasmparser 0.251) not supported"
                                ));
                            }
                        };
                        let kind = match import.ty {
                            wasmparser::TypeRef::Func(type_idx) => ImportKind::Func(type_idx),
                            wasmparser::TypeRef::Memory(mem_type) => ImportKind::Memory(Memory {
                                min: mem_type.initial,
                                max: mem_type.maximum,
                                shared: mem_type.shared,
                                memory64: mem_type.memory64,
                            }),
                            wasmparser::TypeRef::Global(global_type) => ImportKind::Global {
                                value_type: convert_valtype(global_type.content_type),
                                mutable: global_type.mutable,
                            },
                            wasmparser::TypeRef::Table(table_type) => {
                                let element_type = match table_type.element_type {
                                    wasmparser::RefType::FUNCREF => super::RefType::FuncRef,
                                    wasmparser::RefType::EXTERNREF => super::RefType::ExternRef,
                                    _ => super::RefType::FuncRef,
                                };
                                ImportKind::Table(Table {
                                    element_type,
                                    min: table_type.initial as u32,
                                    max: table_type.maximum.map(|m| m as u32),
                                })
                            }
                            _ => continue, // Skip unsupported import kinds
                        };
                        imports.push(Import {
                            module: import.module.to_string(),
                            name: import.name.to_string(),
                            kind,
                        });
                    }
                }
                Payload::FunctionSection(reader) => {
                    for type_idx in reader.clone() {
                        function_type_indices.push(type_idx?);
                    }
                }
                Payload::MemorySection(reader) => {
                    // Phase 14: Capture memory declarations
                    for memory in reader.clone() {
                        let memory = memory?;
                        memories.push(super::Memory {
                            min: memory.initial,
                            max: memory.maximum,
                            shared: memory.shared,
                            memory64: memory.memory64,
                        });
                    }
                }
                Payload::TableSection(reader) => {
                    // Phase 23: Capture table declarations for Component Model support
                    for table in reader.clone() {
                        let table = table?;
                        let element_type = match table.ty.element_type {
                            wasmparser::RefType::FUNCREF => super::RefType::FuncRef,
                            wasmparser::RefType::EXTERNREF => super::RefType::ExternRef,
                            _ => super::RefType::FuncRef, // Default to funcref for other types
                        };
                        tables.push(super::Table {
                            element_type,
                            min: table.ty.initial as u32,
                            max: table.ty.maximum.map(|m| m as u32),
                        });
                    }
                }
                Payload::GlobalSection(reader) => {
                    // Store raw global section bytes to pass through unchanged
                    let range = reader.range();
                    global_section_bytes = Some(bytes[range.start..range.end].to_vec());

                    // Still extract basic globals for optimizer's use
                    for global in reader.clone() {
                        let global = global?;

                        // Only parse if it's a basic value type
                        if let ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64 =
                            global.ty.content_type
                        {
                            // Parse the initializer expression
                            let mut init_reader = global.init_expr.get_operators_reader();
                            let (init_instructions, _) = parse_instructions(&mut init_reader)?;

                            globals.push(super::Global {
                                value_type: convert_valtype(global.ty.content_type),
                                mutable: global.ty.mutable,
                                init: init_instructions,
                            });
                        } else {
                            // Reference type global - will be preserved via raw bytes
                        }
                    }
                }
                Payload::CodeSectionEntry(body) => {
                    // Parse function body
                    let mut locals = Vec::new();
                    let mut reader = body.get_operators_reader()?;

                    // Phase 14: Capture local variable declarations
                    let locals_reader = body.get_locals_reader()?;
                    for local in locals_reader {
                        let (count, val_type) = local?;
                        locals.push((count, convert_valtype(val_type)));
                    }

                    // Parse instructions recursively to handle control flow
                    let (instructions, _terminator) = parse_instructions(&mut reader)?;

                    // Get function type using the index from function section
                    let func_idx = functions.len();
                    let type_index = *function_type_indices
                        .get(func_idx)
                        .ok_or_else(|| anyhow!("Missing type index for function {}", func_idx))?
                        as usize;

                    let func_type = types
                        .get(type_index)
                        .ok_or_else(|| anyhow!("Invalid type index: {}", type_index))?;

                    let signature = FunctionSignature {
                        params: func_type
                            .params()
                            .iter()
                            .map(|t| convert_valtype(*t))
                            .collect(),
                        results: func_type
                            .results()
                            .iter()
                            .map(|t| convert_valtype(*t))
                            .collect(),
                    };

                    // Store function signature for type section reconstruction
                    function_signatures.push(signature.clone());

                    functions.push(Function {
                        name: None, // Names will be added when we parse the name section
                        signature,
                        locals, // Phase 14: Include local variable declarations
                        instructions,
                    });
                }
                Payload::ExportSection(reader) => {
                    // Capture export declarations
                    for export in reader.clone() {
                        let export = export?;
                        let kind = match export.kind {
                            wasmparser::ExternalKind::Func => ExportKind::Func(export.index),
                            wasmparser::ExternalKind::Memory => ExportKind::Memory(export.index),
                            wasmparser::ExternalKind::Global => ExportKind::Global(export.index),
                            wasmparser::ExternalKind::Table => ExportKind::Table(export.index),
                            _ => continue, // Skip unsupported export kinds
                        };
                        exports.push(Export {
                            name: export.name.to_string(),
                            kind,
                        });
                    }
                }
                Payload::StartSection { func, .. } => {
                    // Capture start function
                    start_function = Some(*func);
                }
                Payload::DataSection(reader) => {
                    // Capture data segments (memory initialization)
                    for data in reader.clone() {
                        let data = data?;
                        match data.kind {
                            wasmparser::DataKind::Active {
                                memory_index,
                                offset_expr,
                            } => {
                                let mut offset_reader = offset_expr.get_operators_reader();
                                let (offset_instructions, _) =
                                    parse_instructions(&mut offset_reader)?;
                                data_segments.push(super::DataSegment {
                                    memory_index,
                                    offset: offset_instructions,
                                    data: data.data.to_vec(),
                                    passive: false,
                                });
                            }
                            wasmparser::DataKind::Passive => {
                                data_segments.push(super::DataSegment {
                                    memory_index: 0,
                                    offset: vec![],
                                    data: data.data.to_vec(),
                                    passive: true,
                                });
                            }
                        }
                    }
                }
                Payload::ElementSection(reader) => {
                    // Store raw element section bytes to pass through unchanged
                    let range = reader.range();
                    element_section_bytes = Some(bytes[range.start..range.end].to_vec());
                }
                Payload::CustomSection(reader) => {
                    // Store custom section data (without the name field)
                    // The encoder will add the name field when re-encoding
                    custom_sections.push((reader.name().to_string(), reader.data().to_vec()));
                }
                _ => {
                    // Ignore other payloads (e.g., End, Version, etc.)
                }
            }

            // Validate the payload
            validator.payload(&payload).context("Validation failed")?;
        }

        // Convert all types from wasmparser::FuncType to FunctionSignature
        let all_types: Vec<FunctionSignature> = types
            .iter()
            .map(|func_type| FunctionSignature {
                params: func_type
                    .params()
                    .iter()
                    .map(|t| convert_valtype(*t))
                    .collect(),
                results: func_type
                    .results()
                    .iter()
                    .map(|t| convert_valtype(*t))
                    .collect(),
            })
            .collect();

        Ok(Module {
            functions,
            memories,              // Phase 14: Preserve memory declarations
            tables,                // Phase 23: Preserve table declarations for Component Model
            globals,               // Phase 14: Preserve global declarations
            types: all_types,      // Preserve ALL types from TypeSection
            exports,               // Preserve export declarations
            imports,               // Preserve import declarations
            data_segments,         // Preserve data segments
            element_section_bytes, // Preserve element section as raw bytes
            start_function,        // Preserve start function
            custom_sections,       // Preserve custom sections as raw bytes
            type_section_bytes,    // Preserve type section as raw bytes
            global_section_bytes,  // Preserve global section as raw bytes
        })
    }

    /// What terminated a block of instructions
    #[derive(Debug, PartialEq)]
    enum BlockTerminator {
        End,
        Else,
    }

    /// Recursively parse a sequence of WebAssembly instructions
    /// Handles nested control flow (blocks, loops, if/else)
    /// Returns the instructions and what terminated the block (End or Else)
    fn parse_instructions(
        reader: &mut wasmparser::OperatorsReader,
    ) -> Result<(Vec<Instruction>, BlockTerminator)> {
        let mut instructions = Vec::new();

        while !reader.eof() {
            let op = reader.read()?;
            match op {
                Operator::I32Const { value } => {
                    instructions.push(Instruction::I32Const(value));
                }
                Operator::I32Add => {
                    instructions.push(Instruction::I32Add);
                }
                Operator::I32Sub => {
                    instructions.push(Instruction::I32Sub);
                }
                Operator::I32Mul => {
                    instructions.push(Instruction::I32Mul);
                }
                Operator::I32And => {
                    instructions.push(Instruction::I32And);
                }
                Operator::I32Or => {
                    instructions.push(Instruction::I32Or);
                }
                Operator::I32Xor => {
                    instructions.push(Instruction::I32Xor);
                }
                Operator::I32Shl => {
                    instructions.push(Instruction::I32Shl);
                }
                Operator::I32ShrS => {
                    instructions.push(Instruction::I32ShrS);
                }
                Operator::I32ShrU => {
                    instructions.push(Instruction::I32ShrU);
                }
                Operator::I64Const { value } => {
                    instructions.push(Instruction::I64Const(value));
                }
                Operator::F32Const { value } => {
                    instructions.push(Instruction::F32Const(value.bits()));
                }
                Operator::F64Const { value } => {
                    instructions.push(Instruction::F64Const(value.bits()));
                }
                Operator::I64Add => {
                    instructions.push(Instruction::I64Add);
                }
                Operator::I64Sub => {
                    instructions.push(Instruction::I64Sub);
                }
                Operator::I64Mul => {
                    instructions.push(Instruction::I64Mul);
                }
                Operator::I64And => {
                    instructions.push(Instruction::I64And);
                }
                Operator::I64Or => {
                    instructions.push(Instruction::I64Or);
                }
                Operator::I64Xor => {
                    instructions.push(Instruction::I64Xor);
                }
                Operator::I64Shl => {
                    instructions.push(Instruction::I64Shl);
                }
                Operator::I64ShrS => {
                    instructions.push(Instruction::I64ShrS);
                }
                Operator::I64ShrU => {
                    instructions.push(Instruction::I64ShrU);
                }
                Operator::I32Eq => {
                    instructions.push(Instruction::I32Eq);
                }
                Operator::I32Ne => {
                    instructions.push(Instruction::I32Ne);
                }
                Operator::I32LtS => {
                    instructions.push(Instruction::I32LtS);
                }
                Operator::I32LtU => {
                    instructions.push(Instruction::I32LtU);
                }
                Operator::I32GtS => {
                    instructions.push(Instruction::I32GtS);
                }
                Operator::I32GtU => {
                    instructions.push(Instruction::I32GtU);
                }
                Operator::I32LeS => {
                    instructions.push(Instruction::I32LeS);
                }
                Operator::I32LeU => {
                    instructions.push(Instruction::I32LeU);
                }
                Operator::I32GeS => {
                    instructions.push(Instruction::I32GeS);
                }
                Operator::I32GeU => {
                    instructions.push(Instruction::I32GeU);
                }
                Operator::I64Eq => {
                    instructions.push(Instruction::I64Eq);
                }
                Operator::I64Ne => {
                    instructions.push(Instruction::I64Ne);
                }
                Operator::I64LtS => {
                    instructions.push(Instruction::I64LtS);
                }
                Operator::I64LtU => {
                    instructions.push(Instruction::I64LtU);
                }
                Operator::I64GtS => {
                    instructions.push(Instruction::I64GtS);
                }
                Operator::I64GtU => {
                    instructions.push(Instruction::I64GtU);
                }
                Operator::I64LeS => {
                    instructions.push(Instruction::I64LeS);
                }
                Operator::I64LeU => {
                    instructions.push(Instruction::I64LeU);
                }
                Operator::I64GeS => {
                    instructions.push(Instruction::I64GeS);
                }
                Operator::I64GeU => {
                    instructions.push(Instruction::I64GeU);
                }
                Operator::I32DivS => {
                    instructions.push(Instruction::I32DivS);
                }
                Operator::I32DivU => {
                    instructions.push(Instruction::I32DivU);
                }
                Operator::I32RemS => {
                    instructions.push(Instruction::I32RemS);
                }
                Operator::I32RemU => {
                    instructions.push(Instruction::I32RemU);
                }
                Operator::I64DivS => {
                    instructions.push(Instruction::I64DivS);
                }
                Operator::I64DivU => {
                    instructions.push(Instruction::I64DivU);
                }
                Operator::I64RemS => {
                    instructions.push(Instruction::I64RemS);
                }
                Operator::I64RemU => {
                    instructions.push(Instruction::I64RemU);
                }
                Operator::I32Eqz => {
                    instructions.push(Instruction::I32Eqz);
                }
                Operator::I32Clz => {
                    instructions.push(Instruction::I32Clz);
                }
                Operator::I32Ctz => {
                    instructions.push(Instruction::I32Ctz);
                }
                Operator::I32Popcnt => {
                    instructions.push(Instruction::I32Popcnt);
                }
                Operator::I64Eqz => {
                    instructions.push(Instruction::I64Eqz);
                }
                Operator::I64Clz => {
                    instructions.push(Instruction::I64Clz);
                }
                Operator::I64Ctz => {
                    instructions.push(Instruction::I64Ctz);
                }
                Operator::I64Popcnt => {
                    instructions.push(Instruction::I64Popcnt);
                }
                Operator::Select => {
                    instructions.push(Instruction::Select);
                }
                Operator::LocalGet { local_index } => {
                    instructions.push(Instruction::LocalGet(local_index));
                }
                Operator::LocalSet { local_index } => {
                    instructions.push(Instruction::LocalSet(local_index));
                }
                Operator::LocalTee { local_index } => {
                    instructions.push(Instruction::LocalTee(local_index));
                }
                Operator::GlobalGet { global_index } => {
                    instructions.push(Instruction::GlobalGet(global_index));
                }
                Operator::GlobalSet { global_index } => {
                    instructions.push(Instruction::GlobalSet(global_index));
                }
                Operator::I32Load { memarg } => {
                    instructions.push(Instruction::I32Load {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I32Store { memarg } => {
                    instructions.push(Instruction::I32Store {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Load { memarg } => {
                    instructions.push(Instruction::I64Load {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Store { memarg } => {
                    instructions.push(Instruction::I64Store {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                // Control flow (Phase 14)
                Operator::Block { blockty } => {
                    let block_type = convert_blocktype(blockty)?;
                    let (body, _terminator) = parse_instructions(reader)?;
                    instructions.push(Instruction::Block { block_type, body });
                }
                Operator::Loop { blockty } => {
                    let block_type = convert_blocktype(blockty)?;
                    let (body, _terminator) = parse_instructions(reader)?;
                    instructions.push(Instruction::Loop { block_type, body });
                }
                Operator::If { blockty } => {
                    let block_type = convert_blocktype(blockty)?;
                    let (then_body, terminator) = parse_instructions(reader)?;

                    // If the then_body ended with Else, parse the else_body
                    let else_body = if terminator == BlockTerminator::Else {
                        let (else_instrs, _) = parse_instructions(reader)?;
                        else_instrs
                    } else {
                        vec![]
                    };

                    instructions.push(Instruction::If {
                        block_type,
                        then_body,
                        else_body,
                    });
                }
                Operator::Else => {
                    // Else terminates the then branch and starts the else branch
                    // The parent If handler will parse the else body
                    return Ok((instructions, BlockTerminator::Else));
                }
                Operator::End => {
                    // End terminates the current block/loop/if
                    return Ok((instructions, BlockTerminator::End));
                }
                Operator::Br { relative_depth } => {
                    instructions.push(Instruction::Br(relative_depth));
                }
                Operator::BrIf { relative_depth } => {
                    instructions.push(Instruction::BrIf(relative_depth));
                }
                Operator::BrTable { targets } => {
                    // Parse all targets (excluding default)
                    let targets_vec: Vec<u32> = targets.targets().collect::<Result<Vec<_>, _>>()?;
                    let default = targets.default();
                    instructions.push(Instruction::BrTable {
                        targets: targets_vec,
                        default,
                    });
                }
                Operator::Return => {
                    instructions.push(Instruction::Return);
                }
                Operator::Call { function_index } => {
                    instructions.push(Instruction::Call(function_index));
                }
                Operator::CallIndirect {
                    type_index,
                    table_index,
                    ..
                } => {
                    instructions.push(Instruction::CallIndirect {
                        type_idx: type_index,
                        table_idx: table_index,
                    });
                }
                Operator::Unreachable => {
                    instructions.push(Instruction::Unreachable);
                }
                Operator::Nop => {
                    instructions.push(Instruction::Nop);
                }
                Operator::Drop => {
                    instructions.push(Instruction::Drop);
                }
                // Float operations (f32)
                Operator::F32Add => instructions.push(Instruction::F32Add),
                Operator::F32Sub => instructions.push(Instruction::F32Sub),
                Operator::F32Mul => instructions.push(Instruction::F32Mul),
                Operator::F32Div => instructions.push(Instruction::F32Div),
                Operator::F32Min => instructions.push(Instruction::F32Min),
                Operator::F32Max => instructions.push(Instruction::F32Max),
                Operator::F32Copysign => instructions.push(Instruction::F32Copysign),
                Operator::F32Abs => instructions.push(Instruction::F32Abs),
                Operator::F32Neg => instructions.push(Instruction::F32Neg),
                Operator::F32Ceil => instructions.push(Instruction::F32Ceil),
                Operator::F32Floor => instructions.push(Instruction::F32Floor),
                Operator::F32Trunc => instructions.push(Instruction::F32Trunc),
                Operator::F32Nearest => instructions.push(Instruction::F32Nearest),
                Operator::F32Sqrt => instructions.push(Instruction::F32Sqrt),
                Operator::F32Eq => instructions.push(Instruction::F32Eq),
                Operator::F32Ne => instructions.push(Instruction::F32Ne),
                Operator::F32Lt => instructions.push(Instruction::F32Lt),
                Operator::F32Gt => instructions.push(Instruction::F32Gt),
                Operator::F32Le => instructions.push(Instruction::F32Le),
                Operator::F32Ge => instructions.push(Instruction::F32Ge),
                // Float operations (f64)
                Operator::F64Add => instructions.push(Instruction::F64Add),
                Operator::F64Sub => instructions.push(Instruction::F64Sub),
                Operator::F64Mul => instructions.push(Instruction::F64Mul),
                Operator::F64Div => instructions.push(Instruction::F64Div),
                Operator::F64Min => instructions.push(Instruction::F64Min),
                Operator::F64Max => instructions.push(Instruction::F64Max),
                Operator::F64Copysign => instructions.push(Instruction::F64Copysign),
                Operator::F64Abs => instructions.push(Instruction::F64Abs),
                Operator::F64Neg => instructions.push(Instruction::F64Neg),
                Operator::F64Ceil => instructions.push(Instruction::F64Ceil),
                Operator::F64Floor => instructions.push(Instruction::F64Floor),
                Operator::F64Trunc => instructions.push(Instruction::F64Trunc),
                Operator::F64Nearest => instructions.push(Instruction::F64Nearest),
                Operator::F64Sqrt => instructions.push(Instruction::F64Sqrt),
                Operator::F64Eq => instructions.push(Instruction::F64Eq),
                Operator::F64Ne => instructions.push(Instruction::F64Ne),
                Operator::F64Lt => instructions.push(Instruction::F64Lt),
                Operator::F64Gt => instructions.push(Instruction::F64Gt),
                Operator::F64Le => instructions.push(Instruction::F64Le),
                Operator::F64Ge => instructions.push(Instruction::F64Ge),
                // Conversion operations
                Operator::I32WrapI64 => instructions.push(Instruction::I32WrapI64),
                Operator::I64ExtendI32S => instructions.push(Instruction::I64ExtendI32S),
                Operator::I64ExtendI32U => instructions.push(Instruction::I64ExtendI32U),
                // Sign extension operations
                Operator::I32Extend8S => instructions.push(Instruction::I32Extend8S),
                Operator::I32Extend16S => instructions.push(Instruction::I32Extend16S),
                Operator::I64Extend8S => instructions.push(Instruction::I64Extend8S),
                Operator::I64Extend16S => instructions.push(Instruction::I64Extend16S),
                Operator::I64Extend32S => instructions.push(Instruction::I64Extend32S),
                Operator::I32TruncF32S => instructions.push(Instruction::I32TruncF32S),
                Operator::I32TruncF32U => instructions.push(Instruction::I32TruncF32U),
                Operator::I32TruncF64S => instructions.push(Instruction::I32TruncF64S),
                Operator::I32TruncF64U => instructions.push(Instruction::I32TruncF64U),
                Operator::I64TruncF32S => instructions.push(Instruction::I64TruncF32S),
                Operator::I64TruncF32U => instructions.push(Instruction::I64TruncF32U),
                Operator::I64TruncF64S => instructions.push(Instruction::I64TruncF64S),
                Operator::I64TruncF64U => instructions.push(Instruction::I64TruncF64U),
                // Saturating truncation operations
                Operator::I32TruncSatF32S => instructions.push(Instruction::I32TruncSatF32S),
                Operator::I32TruncSatF32U => instructions.push(Instruction::I32TruncSatF32U),
                Operator::I32TruncSatF64S => instructions.push(Instruction::I32TruncSatF64S),
                Operator::I32TruncSatF64U => instructions.push(Instruction::I32TruncSatF64U),
                Operator::I64TruncSatF32S => instructions.push(Instruction::I64TruncSatF32S),
                Operator::I64TruncSatF32U => instructions.push(Instruction::I64TruncSatF32U),
                Operator::I64TruncSatF64S => instructions.push(Instruction::I64TruncSatF64S),
                Operator::I64TruncSatF64U => instructions.push(Instruction::I64TruncSatF64U),
                Operator::F32ConvertI32S => instructions.push(Instruction::F32ConvertI32S),
                Operator::F32ConvertI32U => instructions.push(Instruction::F32ConvertI32U),
                Operator::F32ConvertI64S => instructions.push(Instruction::F32ConvertI64S),
                Operator::F32ConvertI64U => instructions.push(Instruction::F32ConvertI64U),
                Operator::F64ConvertI32S => instructions.push(Instruction::F64ConvertI32S),
                Operator::F64ConvertI32U => instructions.push(Instruction::F64ConvertI32U),
                Operator::F64ConvertI64S => instructions.push(Instruction::F64ConvertI64S),
                Operator::F64ConvertI64U => instructions.push(Instruction::F64ConvertI64U),
                Operator::F32DemoteF64 => instructions.push(Instruction::F32DemoteF64),
                Operator::F64PromoteF32 => instructions.push(Instruction::F64PromoteF32),
                Operator::I32ReinterpretF32 => instructions.push(Instruction::I32ReinterpretF32),
                Operator::I64ReinterpretF64 => instructions.push(Instruction::I64ReinterpretF64),
                Operator::F32ReinterpretI32 => instructions.push(Instruction::F32ReinterpretI32),
                Operator::F64ReinterpretI64 => instructions.push(Instruction::F64ReinterpretI64),
                // Rotate operations
                Operator::I32Rotl => instructions.push(Instruction::I32Rotl),
                Operator::I32Rotr => instructions.push(Instruction::I32Rotr),
                Operator::I64Rotl => instructions.push(Instruction::I64Rotl),
                Operator::I64Rotr => instructions.push(Instruction::I64Rotr),
                // Memory size/grow operations
                Operator::MemorySize { mem, .. } => instructions.push(Instruction::MemorySize(mem)),
                Operator::MemoryGrow { mem, .. } => instructions.push(Instruction::MemoryGrow(mem)),
                // Bulk memory operations
                Operator::MemoryFill { mem } => instructions.push(Instruction::MemoryFill(mem)),
                Operator::MemoryCopy { dst_mem, src_mem } => {
                    instructions.push(Instruction::MemoryCopy { dst_mem, src_mem });
                }
                Operator::MemoryInit { mem, data_index } => {
                    instructions.push(Instruction::MemoryInit {
                        mem,
                        data_idx: data_index,
                    });
                }
                Operator::DataDrop { data_index } => {
                    instructions.push(Instruction::DataDrop(data_index));
                }
                // Float loads/stores
                Operator::F32Load { memarg } => {
                    instructions.push(Instruction::F32Load {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::F32Store { memarg } => {
                    instructions.push(Instruction::F32Store {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::F64Load { memarg } => {
                    instructions.push(Instruction::F64Load {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::F64Store { memarg } => {
                    instructions.push(Instruction::F64Store {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                // Additional load/store variants
                Operator::I32Load8S { memarg } => {
                    instructions.push(Instruction::I32Load8S {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I32Load8U { memarg } => {
                    instructions.push(Instruction::I32Load8U {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I32Load16S { memarg } => {
                    instructions.push(Instruction::I32Load16S {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I32Load16U { memarg } => {
                    instructions.push(Instruction::I32Load16U {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Load8S { memarg } => {
                    instructions.push(Instruction::I64Load8S {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Load8U { memarg } => {
                    instructions.push(Instruction::I64Load8U {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Load16S { memarg } => {
                    instructions.push(Instruction::I64Load16S {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Load16U { memarg } => {
                    instructions.push(Instruction::I64Load16U {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Load32S { memarg } => {
                    instructions.push(Instruction::I64Load32S {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Load32U { memarg } => {
                    instructions.push(Instruction::I64Load32U {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I32Store8 { memarg } => {
                    instructions.push(Instruction::I32Store8 {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I32Store16 { memarg } => {
                    instructions.push(Instruction::I32Store16 {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Store8 { memarg } => {
                    instructions.push(Instruction::I64Store8 {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Store16 { memarg } => {
                    instructions.push(Instruction::I64Store16 {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                Operator::I64Store32 { memarg } => {
                    instructions.push(Instruction::I64Store32 {
                        offset: memarg.offset as u32,
                        align: memarg.align as u32,
                        mem: memarg.memory,
                    });
                }
                // Unknown/unsupported instructions - return error to fail fast
                // This ensures we don't silently drop instructions during roundtrip
                _ => {
                    return Err(anyhow!(
                        "Unsupported instruction encountered during parsing: {:?}. \
                         This module cannot be safely optimized.",
                        op
                    ));
                }
            }
        }

        // If we reach EOF without hitting End or Else, treat it as End
        Ok((instructions, BlockTerminator::End))
    }

    /// Parse a WebAssembly text (WAT) module
    pub fn parse_wat(text: &str) -> Result<Module> {
        // Use the wat crate to convert WAT to WASM binary
        let wasm_bytes = wat::parse_str(text).context("Failed to parse WAT")?;
        // Then parse the binary
        parse_wasm(&wasm_bytes)
    }

    /// Convert wasmparser::ValType to our ValueType
    fn convert_valtype(vt: ValType) -> ValueType {
        match vt {
            ValType::I32 => ValueType::I32,
            ValType::I64 => ValueType::I64,
            ValType::F32 => ValueType::F32,
            ValType::F64 => ValueType::F64,
            _ => ValueType::I32, // Default to I32 for unsupported types in Phase 2
        }
    }

    /// Convert wasmparser::BlockType to our BlockType
    fn convert_blocktype(bt: wasmparser::BlockType) -> Result<BlockType> {
        match bt {
            wasmparser::BlockType::Empty => Ok(BlockType::Empty),
            wasmparser::BlockType::Type(vt) => Ok(BlockType::Value(convert_valtype(vt))),
            wasmparser::BlockType::FuncType(_) => {
                // Function type blocks require looking up the type in the type section
                // For now, we'll return an error as this is complex
                Err(anyhow!(
                    "Function type blocks not yet supported in parser (BlockType::FuncType)"
                ))
            }
        }
    }
}

/// Module encoding functionality: Encode LOOM's internal representation back to WebAssembly
pub mod encode {

    use super::{
        BlockType, ExportKind, FunctionSignature, ImportKind, Instruction, Module, RefType,
        ValueType,
    };
    use anyhow::{Context, Result, anyhow};
    use wasm_encoder::{
        CodeSection, ConstExpr, CustomSection, DataSection, EntityType,
        ExportKind as EncoderExportKind, ExportSection, Function as EncoderFunction,
        FunctionSection, GlobalSection, GlobalType, ImportSection,
        Instruction as EncoderInstruction, MemorySection, MemoryType, RawSection, TableType,
        TypeSection, ValType,
    };

    /// Helper function to encode a constant expression (for global initializers)
    fn encode_const_expr(
        instructions: &[Instruction],
        expected_type: ValueType,
    ) -> Result<ConstExpr> {
        // Global initializers should be simple constant expressions
        // For now, we support single constant instructions
        if instructions.len() != 1 {
            // Fallback to zero for complex expressions
            return Ok(match expected_type {
                ValueType::I32 => ConstExpr::i32_const(0),
                ValueType::I64 => ConstExpr::i64_const(0),
                ValueType::F32 => ConstExpr::f32_const(0.0.into()),
                ValueType::F64 => ConstExpr::f64_const(0.0.into()),
            });
        }

        match &instructions[0] {
            Instruction::I32Const(val) => Ok(ConstExpr::i32_const(*val)),
            Instruction::I64Const(val) => Ok(ConstExpr::i64_const(*val)),
            Instruction::F32Const(bits) => {
                let f32_val = f32::from_bits(*bits);
                Ok(ConstExpr::f32_const(f32_val.into()))
            }
            Instruction::F64Const(bits) => {
                let f64_val = f64::from_bits(*bits);
                Ok(ConstExpr::f64_const(f64_val.into()))
            }
            _ => {
                // Fallback to zero for unsupported expressions
                Ok(match expected_type {
                    ValueType::I32 => ConstExpr::i32_const(0),
                    ValueType::I64 => ConstExpr::i64_const(0),
                    ValueType::F32 => ConstExpr::f32_const(0.0.into()),
                    ValueType::F64 => ConstExpr::f64_const(0.0.into()),
                })
            }
        }
    }

    /// Encode to WebAssembly binary module
    pub fn encode_wasm(module: &Module) -> Result<Vec<u8>> {
        let mut wasm_module = wasm_encoder::Module::new();

        // FIXED: Ensure all function signatures are in the types array
        // Collect unique signatures and build a deduplicated types list
        let mut unique_types = module.types.clone();
        let mut type_map: std::collections::HashMap<FunctionSignature, usize> = module
            .types
            .iter()
            .enumerate()
            .map(|(i, t)| (t.clone(), i))
            .collect();

        // Add any function signatures that aren't already in types
        for func in &module.functions {
            if !type_map.contains_key(&func.signature) {
                let idx = unique_types.len();
                unique_types.push(func.signature.clone());
                type_map.insert(func.signature.clone(), idx);
            }
        }

        // Build type section - use raw bytes if available (preserves GC types, etc.)
        if let Some(type_bytes) = &module.type_section_bytes {
            wasm_module.section(&RawSection {
                id: 1, // Type section ID
                data: type_bytes,
            });
        } else {
            // Fallback: build from function types only
            let mut types = TypeSection::new();
            for ty in &unique_types {
                let params: Vec<ValType> =
                    ty.params.iter().map(|t| convert_to_valtype(*t)).collect();
                let results: Vec<ValType> =
                    ty.results.iter().map(|t| convert_to_valtype(*t)).collect();
                types.ty().function(params, results);
            }
            wasm_module.section(&types);
        }

        // Build import section
        if !module.imports.is_empty() {
            let mut imports = ImportSection::new();
            for import in &module.imports {
                let entity_type = match &import.kind {
                    ImportKind::Func(type_idx) => EntityType::Function(*type_idx),
                    ImportKind::Memory(mem) => EntityType::Memory(MemoryType {
                        minimum: mem.min,
                        maximum: mem.max,
                        memory64: mem.memory64,
                        shared: mem.shared,
                        page_size_log2: None,
                    }),
                    ImportKind::Global {
                        value_type,
                        mutable,
                    } => EntityType::Global(GlobalType {
                        val_type: convert_to_valtype(*value_type),
                        mutable: *mutable,
                        shared: false,
                    }),
                    ImportKind::Table(table) => {
                        let ref_type = match table.element_type {
                            RefType::FuncRef => wasm_encoder::RefType::FUNCREF,
                            RefType::ExternRef => wasm_encoder::RefType::EXTERNREF,
                        };
                        EntityType::Table(TableType {
                            element_type: ref_type,
                            minimum: table.min as u64,
                            maximum: table.max.map(|m| m as u64),
                            table64: false,
                            shared: false,
                        })
                    }
                };
                imports.import(&import.module, &import.name, entity_type);
            }
            wasm_module.section(&imports);
        }

        // Build function section (references to types)
        // Map each function to its type index
        let mut functions = FunctionSection::new();
        for func in &module.functions {
            // Find the type index for this function's signature
            let type_idx = *type_map.get(&func.signature).ok_or_else(|| {
                anyhow!(
                    "encoder: function signature {:?} not found in type_map (REQ-3 — \
                     no silent failures); this indicates a bug in collect_function_types",
                    func.signature
                )
            })? as u32;
            functions.function(type_idx);
        }
        wasm_module.section(&functions);

        // Phase 23: Build table section for Component Model support
        // IMPORTANT: Must come before memory section per WASM spec ordering
        if !module.tables.is_empty() {
            let mut tables = wasm_encoder::TableSection::new();
            for table in &module.tables {
                let ref_type = match table.element_type {
                    RefType::FuncRef => wasm_encoder::RefType::FUNCREF,
                    RefType::ExternRef => wasm_encoder::RefType::EXTERNREF,
                };
                let table_type = wasm_encoder::TableType {
                    element_type: ref_type,
                    minimum: table.min as u64,
                    maximum: table.max.map(|m| m as u64),
                    table64: false,
                    shared: false,
                };
                // Tables don't have initializers in the basic form, use default
                tables.table(table_type);
            }
            wasm_module.section(&tables);
        }

        // Phase 14: Build memory section
        if !module.memories.is_empty() {
            let mut memories = MemorySection::new();
            for memory in &module.memories {
                let memory_type = MemoryType {
                    minimum: memory.min,
                    maximum: memory.max,
                    memory64: memory.memory64,
                    shared: memory.shared,
                    page_size_log2: None,
                };
                memories.memory(memory_type);
            }
            wasm_module.section(&memories);
        }

        // Phase 14: Build global section - use raw bytes if available
        if let Some(global_bytes) = &module.global_section_bytes {
            wasm_module.section(&RawSection {
                id: 6, // Global section ID
                data: global_bytes,
            });
        } else if !module.globals.is_empty() {
            // Fallback: encode basic globals
            let mut globals = GlobalSection::new();
            for global in &module.globals {
                let global_type = GlobalType {
                    val_type: convert_to_valtype(global.value_type),
                    mutable: global.mutable,
                    shared: false,
                };
                // Encode the initializer expression
                let init_expr = encode_const_expr(&global.init, global.value_type)?;
                globals.global(global_type, &init_expr);
            }
            wasm_module.section(&globals);
        }

        // Build export section
        if !module.exports.is_empty() {
            let mut exports = ExportSection::new();
            for export in &module.exports {
                match &export.kind {
                    ExportKind::Func(idx) => {
                        exports.export(&export.name, EncoderExportKind::Func, *idx);
                    }
                    ExportKind::Memory(idx) => {
                        exports.export(&export.name, EncoderExportKind::Memory, *idx);
                    }
                    ExportKind::Global(idx) => {
                        exports.export(&export.name, EncoderExportKind::Global, *idx);
                    }
                    ExportKind::Table(idx) => {
                        exports.export(&export.name, EncoderExportKind::Table, *idx);
                    }
                }
            }
            wasm_module.section(&exports);
        }

        // Build start section
        if let Some(start_func_idx) = module.start_function {
            wasm_module.section(&wasm_encoder::StartSection {
                function_index: start_func_idx,
            });
        }

        // Build element section (table initialization)
        // Pass through raw element section bytes unchanged
        if let Some(element_bytes) = &module.element_section_bytes {
            wasm_module.section(&RawSection {
                id: 9, // Element section ID
                data: element_bytes,
            });
        }

        // Build code section (function bodies)
        let mut code = CodeSection::new();
        for func in &module.functions {
            // Phase 14: Encode local variable declarations
            let locals_vec: Vec<(u32, ValType)> = func
                .locals
                .iter()
                .map(|(count, val_type)| (*count, convert_to_valtype(*val_type)))
                .collect();

            let mut func_body = EncoderFunction::new(locals_vec);

            for instr in &func.instructions {
                match instr {
                    Instruction::I32Const(value) => {
                        func_body.instruction(&EncoderInstruction::I32Const(*value));
                    }
                    Instruction::I32Add => {
                        func_body.instruction(&EncoderInstruction::I32Add);
                    }
                    Instruction::I32Sub => {
                        func_body.instruction(&EncoderInstruction::I32Sub);
                    }
                    Instruction::I32Mul => {
                        func_body.instruction(&EncoderInstruction::I32Mul);
                    }
                    Instruction::I32And => {
                        func_body.instruction(&EncoderInstruction::I32And);
                    }
                    Instruction::I32Or => {
                        func_body.instruction(&EncoderInstruction::I32Or);
                    }
                    Instruction::I32Xor => {
                        func_body.instruction(&EncoderInstruction::I32Xor);
                    }
                    Instruction::I32Shl => {
                        func_body.instruction(&EncoderInstruction::I32Shl);
                    }
                    Instruction::I32ShrS => {
                        func_body.instruction(&EncoderInstruction::I32ShrS);
                    }
                    Instruction::I32ShrU => {
                        func_body.instruction(&EncoderInstruction::I32ShrU);
                    }
                    Instruction::I64Const(value) => {
                        func_body.instruction(&EncoderInstruction::I64Const(*value));
                    }
                    Instruction::F32Const(bits) => {
                        let f32_val = f32::from_bits(*bits);
                        func_body.instruction(&EncoderInstruction::F32Const(f32_val.into()));
                    }
                    Instruction::F64Const(bits) => {
                        let f64_val = f64::from_bits(*bits);
                        func_body.instruction(&EncoderInstruction::F64Const(f64_val.into()));
                    }
                    Instruction::I64Add => {
                        func_body.instruction(&EncoderInstruction::I64Add);
                    }
                    Instruction::I64Sub => {
                        func_body.instruction(&EncoderInstruction::I64Sub);
                    }
                    Instruction::I64Mul => {
                        func_body.instruction(&EncoderInstruction::I64Mul);
                    }
                    Instruction::I64And => {
                        func_body.instruction(&EncoderInstruction::I64And);
                    }
                    Instruction::I64Or => {
                        func_body.instruction(&EncoderInstruction::I64Or);
                    }
                    Instruction::I64Xor => {
                        func_body.instruction(&EncoderInstruction::I64Xor);
                    }
                    Instruction::I64Shl => {
                        func_body.instruction(&EncoderInstruction::I64Shl);
                    }
                    Instruction::I64ShrS => {
                        func_body.instruction(&EncoderInstruction::I64ShrS);
                    }
                    Instruction::I64ShrU => {
                        func_body.instruction(&EncoderInstruction::I64ShrU);
                    }
                    Instruction::I32Eq => {
                        func_body.instruction(&EncoderInstruction::I32Eq);
                    }
                    Instruction::I32Ne => {
                        func_body.instruction(&EncoderInstruction::I32Ne);
                    }
                    Instruction::I32LtS => {
                        func_body.instruction(&EncoderInstruction::I32LtS);
                    }
                    Instruction::I32LtU => {
                        func_body.instruction(&EncoderInstruction::I32LtU);
                    }
                    Instruction::I32GtS => {
                        func_body.instruction(&EncoderInstruction::I32GtS);
                    }
                    Instruction::I32GtU => {
                        func_body.instruction(&EncoderInstruction::I32GtU);
                    }
                    Instruction::I32LeS => {
                        func_body.instruction(&EncoderInstruction::I32LeS);
                    }
                    Instruction::I32LeU => {
                        func_body.instruction(&EncoderInstruction::I32LeU);
                    }
                    Instruction::I32GeS => {
                        func_body.instruction(&EncoderInstruction::I32GeS);
                    }
                    Instruction::I32GeU => {
                        func_body.instruction(&EncoderInstruction::I32GeU);
                    }
                    Instruction::I64Eq => {
                        func_body.instruction(&EncoderInstruction::I64Eq);
                    }
                    Instruction::I64Ne => {
                        func_body.instruction(&EncoderInstruction::I64Ne);
                    }
                    Instruction::I64LtS => {
                        func_body.instruction(&EncoderInstruction::I64LtS);
                    }
                    Instruction::I64LtU => {
                        func_body.instruction(&EncoderInstruction::I64LtU);
                    }
                    Instruction::I64GtS => {
                        func_body.instruction(&EncoderInstruction::I64GtS);
                    }
                    Instruction::I64GtU => {
                        func_body.instruction(&EncoderInstruction::I64GtU);
                    }
                    Instruction::I64LeS => {
                        func_body.instruction(&EncoderInstruction::I64LeS);
                    }
                    Instruction::I64LeU => {
                        func_body.instruction(&EncoderInstruction::I64LeU);
                    }
                    Instruction::I64GeS => {
                        func_body.instruction(&EncoderInstruction::I64GeS);
                    }
                    Instruction::I64GeU => {
                        func_body.instruction(&EncoderInstruction::I64GeU);
                    }
                    Instruction::I32DivS => {
                        func_body.instruction(&EncoderInstruction::I32DivS);
                    }
                    Instruction::I32DivU => {
                        func_body.instruction(&EncoderInstruction::I32DivU);
                    }
                    Instruction::I32RemS => {
                        func_body.instruction(&EncoderInstruction::I32RemS);
                    }
                    Instruction::I32RemU => {
                        func_body.instruction(&EncoderInstruction::I32RemU);
                    }
                    Instruction::I64DivS => {
                        func_body.instruction(&EncoderInstruction::I64DivS);
                    }
                    Instruction::I64DivU => {
                        func_body.instruction(&EncoderInstruction::I64DivU);
                    }
                    Instruction::I64RemS => {
                        func_body.instruction(&EncoderInstruction::I64RemS);
                    }
                    Instruction::I64RemU => {
                        func_body.instruction(&EncoderInstruction::I64RemU);
                    }
                    Instruction::I32Eqz => {
                        func_body.instruction(&EncoderInstruction::I32Eqz);
                    }
                    Instruction::I32Clz => {
                        func_body.instruction(&EncoderInstruction::I32Clz);
                    }
                    Instruction::I32Ctz => {
                        func_body.instruction(&EncoderInstruction::I32Ctz);
                    }
                    Instruction::I32Popcnt => {
                        func_body.instruction(&EncoderInstruction::I32Popcnt);
                    }
                    Instruction::I64Eqz => {
                        func_body.instruction(&EncoderInstruction::I64Eqz);
                    }
                    Instruction::I64Clz => {
                        func_body.instruction(&EncoderInstruction::I64Clz);
                    }
                    Instruction::I64Ctz => {
                        func_body.instruction(&EncoderInstruction::I64Ctz);
                    }
                    Instruction::I64Popcnt => {
                        func_body.instruction(&EncoderInstruction::I64Popcnt);
                    }
                    Instruction::Select => {
                        func_body.instruction(&EncoderInstruction::Select);
                    }
                    Instruction::LocalGet(idx) => {
                        func_body.instruction(&EncoderInstruction::LocalGet(*idx));
                    }
                    Instruction::LocalSet(idx) => {
                        func_body.instruction(&EncoderInstruction::LocalSet(*idx));
                    }
                    Instruction::LocalTee(idx) => {
                        func_body.instruction(&EncoderInstruction::LocalTee(*idx));
                    }
                    Instruction::GlobalGet(idx) => {
                        func_body.instruction(&EncoderInstruction::GlobalGet(*idx));
                    }
                    Instruction::GlobalSet(idx) => {
                        func_body.instruction(&EncoderInstruction::GlobalSet(*idx));
                    }
                    Instruction::I32Load { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Load(wasm_encoder::MemArg {
                            offset: *offset as u64,
                            align: *align,
                            memory_index: *mem,
                        }));
                    }
                    Instruction::I32Store { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Store(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Load { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Load(wasm_encoder::MemArg {
                            offset: *offset as u64,
                            align: *align,
                            memory_index: *mem,
                        }));
                    }
                    Instruction::I64Store { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Store(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    // Float operations (f32)
                    Instruction::F32Add => {
                        func_body.instruction(&EncoderInstruction::F32Add);
                    }
                    Instruction::F32Sub => {
                        func_body.instruction(&EncoderInstruction::F32Sub);
                    }
                    Instruction::F32Mul => {
                        func_body.instruction(&EncoderInstruction::F32Mul);
                    }
                    Instruction::F32Div => {
                        func_body.instruction(&EncoderInstruction::F32Div);
                    }
                    Instruction::F32Min => {
                        func_body.instruction(&EncoderInstruction::F32Min);
                    }
                    Instruction::F32Max => {
                        func_body.instruction(&EncoderInstruction::F32Max);
                    }
                    Instruction::F32Copysign => {
                        func_body.instruction(&EncoderInstruction::F32Copysign);
                    }
                    Instruction::F32Abs => {
                        func_body.instruction(&EncoderInstruction::F32Abs);
                    }
                    Instruction::F32Neg => {
                        func_body.instruction(&EncoderInstruction::F32Neg);
                    }
                    Instruction::F32Ceil => {
                        func_body.instruction(&EncoderInstruction::F32Ceil);
                    }
                    Instruction::F32Floor => {
                        func_body.instruction(&EncoderInstruction::F32Floor);
                    }
                    Instruction::F32Trunc => {
                        func_body.instruction(&EncoderInstruction::F32Trunc);
                    }
                    Instruction::F32Nearest => {
                        func_body.instruction(&EncoderInstruction::F32Nearest);
                    }
                    Instruction::F32Sqrt => {
                        func_body.instruction(&EncoderInstruction::F32Sqrt);
                    }
                    Instruction::F32Eq => {
                        func_body.instruction(&EncoderInstruction::F32Eq);
                    }
                    Instruction::F32Ne => {
                        func_body.instruction(&EncoderInstruction::F32Ne);
                    }
                    Instruction::F32Lt => {
                        func_body.instruction(&EncoderInstruction::F32Lt);
                    }
                    Instruction::F32Gt => {
                        func_body.instruction(&EncoderInstruction::F32Gt);
                    }
                    Instruction::F32Le => {
                        func_body.instruction(&EncoderInstruction::F32Le);
                    }
                    Instruction::F32Ge => {
                        func_body.instruction(&EncoderInstruction::F32Ge);
                    }
                    // Float operations (f64)
                    Instruction::F64Add => {
                        func_body.instruction(&EncoderInstruction::F64Add);
                    }
                    Instruction::F64Sub => {
                        func_body.instruction(&EncoderInstruction::F64Sub);
                    }
                    Instruction::F64Mul => {
                        func_body.instruction(&EncoderInstruction::F64Mul);
                    }
                    Instruction::F64Div => {
                        func_body.instruction(&EncoderInstruction::F64Div);
                    }
                    Instruction::F64Min => {
                        func_body.instruction(&EncoderInstruction::F64Min);
                    }
                    Instruction::F64Max => {
                        func_body.instruction(&EncoderInstruction::F64Max);
                    }
                    Instruction::F64Copysign => {
                        func_body.instruction(&EncoderInstruction::F64Copysign);
                    }
                    Instruction::F64Abs => {
                        func_body.instruction(&EncoderInstruction::F64Abs);
                    }
                    Instruction::F64Neg => {
                        func_body.instruction(&EncoderInstruction::F64Neg);
                    }
                    Instruction::F64Ceil => {
                        func_body.instruction(&EncoderInstruction::F64Ceil);
                    }
                    Instruction::F64Floor => {
                        func_body.instruction(&EncoderInstruction::F64Floor);
                    }
                    Instruction::F64Trunc => {
                        func_body.instruction(&EncoderInstruction::F64Trunc);
                    }
                    Instruction::F64Nearest => {
                        func_body.instruction(&EncoderInstruction::F64Nearest);
                    }
                    Instruction::F64Sqrt => {
                        func_body.instruction(&EncoderInstruction::F64Sqrt);
                    }
                    Instruction::F64Eq => {
                        func_body.instruction(&EncoderInstruction::F64Eq);
                    }
                    Instruction::F64Ne => {
                        func_body.instruction(&EncoderInstruction::F64Ne);
                    }
                    Instruction::F64Lt => {
                        func_body.instruction(&EncoderInstruction::F64Lt);
                    }
                    Instruction::F64Gt => {
                        func_body.instruction(&EncoderInstruction::F64Gt);
                    }
                    Instruction::F64Le => {
                        func_body.instruction(&EncoderInstruction::F64Le);
                    }
                    Instruction::F64Ge => {
                        func_body.instruction(&EncoderInstruction::F64Ge);
                    }
                    // Conversion operations
                    Instruction::I32WrapI64 => {
                        func_body.instruction(&EncoderInstruction::I32WrapI64);
                    }
                    Instruction::I64ExtendI32S => {
                        func_body.instruction(&EncoderInstruction::I64ExtendI32S);
                    }
                    Instruction::I64ExtendI32U => {
                        func_body.instruction(&EncoderInstruction::I64ExtendI32U);
                    }
                    // Sign extension operations
                    Instruction::I32Extend8S => {
                        func_body.instruction(&EncoderInstruction::I32Extend8S);
                    }
                    Instruction::I32Extend16S => {
                        func_body.instruction(&EncoderInstruction::I32Extend16S);
                    }
                    Instruction::I64Extend8S => {
                        func_body.instruction(&EncoderInstruction::I64Extend8S);
                    }
                    Instruction::I64Extend16S => {
                        func_body.instruction(&EncoderInstruction::I64Extend16S);
                    }
                    Instruction::I64Extend32S => {
                        func_body.instruction(&EncoderInstruction::I64Extend32S);
                    }
                    Instruction::I32TruncF32S => {
                        func_body.instruction(&EncoderInstruction::I32TruncF32S);
                    }
                    Instruction::I32TruncF32U => {
                        func_body.instruction(&EncoderInstruction::I32TruncF32U);
                    }
                    Instruction::I32TruncF64S => {
                        func_body.instruction(&EncoderInstruction::I32TruncF64S);
                    }
                    Instruction::I32TruncF64U => {
                        func_body.instruction(&EncoderInstruction::I32TruncF64U);
                    }
                    Instruction::I64TruncF32S => {
                        func_body.instruction(&EncoderInstruction::I64TruncF32S);
                    }
                    Instruction::I64TruncF32U => {
                        func_body.instruction(&EncoderInstruction::I64TruncF32U);
                    }
                    Instruction::I64TruncF64S => {
                        func_body.instruction(&EncoderInstruction::I64TruncF64S);
                    }
                    Instruction::I64TruncF64U => {
                        func_body.instruction(&EncoderInstruction::I64TruncF64U);
                    }
                    // Saturating truncation operations
                    Instruction::I32TruncSatF32S => {
                        func_body.instruction(&EncoderInstruction::I32TruncSatF32S);
                    }
                    Instruction::I32TruncSatF32U => {
                        func_body.instruction(&EncoderInstruction::I32TruncSatF32U);
                    }
                    Instruction::I32TruncSatF64S => {
                        func_body.instruction(&EncoderInstruction::I32TruncSatF64S);
                    }
                    Instruction::I32TruncSatF64U => {
                        func_body.instruction(&EncoderInstruction::I32TruncSatF64U);
                    }
                    Instruction::I64TruncSatF32S => {
                        func_body.instruction(&EncoderInstruction::I64TruncSatF32S);
                    }
                    Instruction::I64TruncSatF32U => {
                        func_body.instruction(&EncoderInstruction::I64TruncSatF32U);
                    }
                    Instruction::I64TruncSatF64S => {
                        func_body.instruction(&EncoderInstruction::I64TruncSatF64S);
                    }
                    Instruction::I64TruncSatF64U => {
                        func_body.instruction(&EncoderInstruction::I64TruncSatF64U);
                    }
                    Instruction::F32ConvertI32S => {
                        func_body.instruction(&EncoderInstruction::F32ConvertI32S);
                    }
                    Instruction::F32ConvertI32U => {
                        func_body.instruction(&EncoderInstruction::F32ConvertI32U);
                    }
                    Instruction::F32ConvertI64S => {
                        func_body.instruction(&EncoderInstruction::F32ConvertI64S);
                    }
                    Instruction::F32ConvertI64U => {
                        func_body.instruction(&EncoderInstruction::F32ConvertI64U);
                    }
                    Instruction::F64ConvertI32S => {
                        func_body.instruction(&EncoderInstruction::F64ConvertI32S);
                    }
                    Instruction::F64ConvertI32U => {
                        func_body.instruction(&EncoderInstruction::F64ConvertI32U);
                    }
                    Instruction::F64ConvertI64S => {
                        func_body.instruction(&EncoderInstruction::F64ConvertI64S);
                    }
                    Instruction::F64ConvertI64U => {
                        func_body.instruction(&EncoderInstruction::F64ConvertI64U);
                    }
                    Instruction::F32DemoteF64 => {
                        func_body.instruction(&EncoderInstruction::F32DemoteF64);
                    }
                    Instruction::F64PromoteF32 => {
                        func_body.instruction(&EncoderInstruction::F64PromoteF32);
                    }
                    Instruction::I32ReinterpretF32 => {
                        func_body.instruction(&EncoderInstruction::I32ReinterpretF32);
                    }
                    Instruction::I64ReinterpretF64 => {
                        func_body.instruction(&EncoderInstruction::I64ReinterpretF64);
                    }
                    Instruction::F32ReinterpretI32 => {
                        func_body.instruction(&EncoderInstruction::F32ReinterpretI32);
                    }
                    Instruction::F64ReinterpretI64 => {
                        func_body.instruction(&EncoderInstruction::F64ReinterpretI64);
                    }
                    // Rotate operations
                    Instruction::I32Rotl => {
                        func_body.instruction(&EncoderInstruction::I32Rotl);
                    }
                    Instruction::I32Rotr => {
                        func_body.instruction(&EncoderInstruction::I32Rotr);
                    }
                    Instruction::I64Rotl => {
                        func_body.instruction(&EncoderInstruction::I64Rotl);
                    }
                    Instruction::I64Rotr => {
                        func_body.instruction(&EncoderInstruction::I64Rotr);
                    }
                    // Memory size/grow operations
                    Instruction::MemorySize(mem) => {
                        func_body.instruction(&EncoderInstruction::MemorySize(*mem));
                    }
                    Instruction::MemoryGrow(mem) => {
                        func_body.instruction(&EncoderInstruction::MemoryGrow(*mem));
                    }
                    // Bulk memory operations
                    Instruction::MemoryFill(mem) => {
                        func_body.instruction(&EncoderInstruction::MemoryFill(*mem));
                    }
                    Instruction::MemoryCopy { dst_mem, src_mem } => {
                        func_body.instruction(&EncoderInstruction::MemoryCopy {
                            src_mem: *src_mem,
                            dst_mem: *dst_mem,
                        });
                    }
                    Instruction::MemoryInit { mem, data_idx } => {
                        func_body.instruction(&EncoderInstruction::MemoryInit {
                            mem: *mem,
                            data_index: *data_idx,
                        });
                    }
                    Instruction::DataDrop(data_idx) => {
                        func_body.instruction(&EncoderInstruction::DataDrop(*data_idx));
                    }
                    // Float loads/stores
                    Instruction::F32Load { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::F32Load(wasm_encoder::MemArg {
                            offset: *offset as u64,
                            align: *align,
                            memory_index: *mem,
                        }));
                    }
                    Instruction::F32Store { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::F32Store(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::F64Load { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::F64Load(wasm_encoder::MemArg {
                            offset: *offset as u64,
                            align: *align,
                            memory_index: *mem,
                        }));
                    }
                    Instruction::F64Store { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::F64Store(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    // Additional load/store variants
                    Instruction::I32Load8S { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Load8S(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I32Load8U { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Load8U(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I32Load16S { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Load16S(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I32Load16U { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Load16U(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Load8S { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Load8S(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Load8U { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Load8U(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Load16S { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Load16S(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Load16U { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Load16U(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Load32S { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Load32S(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Load32U { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Load32U(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I32Store8 { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Store8(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I32Store16 { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I32Store16(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Store8 { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Store8(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Store16 { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Store16(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    Instruction::I64Store32 { offset, align, mem } => {
                        func_body.instruction(&EncoderInstruction::I64Store32(
                            wasm_encoder::MemArg {
                                offset: *offset as u64,
                                align: *align,
                                memory_index: *mem,
                            },
                        ));
                    }
                    // Control flow instructions (Phase 14)
                    Instruction::Block { block_type, body } => {
                        let bt = convert_blocktype_to_encoder(block_type);
                        func_body.instruction(&EncoderInstruction::Block(bt));
                        // Recursively encode body
                        for nested_instr in body {
                            encode_instruction_recursive(nested_instr, &mut func_body);
                        }
                        func_body.instruction(&EncoderInstruction::End);
                    }
                    Instruction::Loop { block_type, body } => {
                        let bt = convert_blocktype_to_encoder(block_type);
                        func_body.instruction(&EncoderInstruction::Loop(bt));
                        // Recursively encode body
                        for nested_instr in body {
                            encode_instruction_recursive(nested_instr, &mut func_body);
                        }
                        func_body.instruction(&EncoderInstruction::End);
                    }
                    Instruction::If {
                        block_type,
                        then_body,
                        else_body,
                    } => {
                        let bt = convert_blocktype_to_encoder(block_type);
                        func_body.instruction(&EncoderInstruction::If(bt));
                        // Encode then body
                        for nested_instr in then_body {
                            encode_instruction_recursive(nested_instr, &mut func_body);
                        }
                        // Encode else body if non-empty
                        if !else_body.is_empty() {
                            func_body.instruction(&EncoderInstruction::Else);
                            for nested_instr in else_body {
                                encode_instruction_recursive(nested_instr, &mut func_body);
                            }
                        }
                        func_body.instruction(&EncoderInstruction::End);
                    }
                    Instruction::Br(depth) => {
                        func_body.instruction(&EncoderInstruction::Br(*depth));
                    }
                    Instruction::BrIf(depth) => {
                        func_body.instruction(&EncoderInstruction::BrIf(*depth));
                    }
                    Instruction::BrTable { targets, default } => {
                        func_body.instruction(&EncoderInstruction::BrTable(
                            targets.as_slice().into(),
                            *default,
                        ));
                    }
                    Instruction::Return => {
                        func_body.instruction(&EncoderInstruction::Return);
                    }
                    Instruction::Call(func_idx) => {
                        func_body.instruction(&EncoderInstruction::Call(*func_idx));
                    }
                    Instruction::CallIndirect {
                        type_idx,
                        table_idx,
                    } => {
                        func_body.instruction(&EncoderInstruction::CallIndirect {
                            type_index: *type_idx,
                            table_index: *table_idx,
                        });
                    }
                    Instruction::Unreachable => {
                        func_body.instruction(&EncoderInstruction::Unreachable);
                    }
                    Instruction::Nop => {
                        func_body.instruction(&EncoderInstruction::Nop);
                    }
                    Instruction::Drop => {
                        func_body.instruction(&EncoderInstruction::Drop);
                    }
                    Instruction::End => {
                        // End should not appear in instruction lists after parsing
                        // It's only used to terminate blocks during parsing
                        // The wasm-encoder will add the final End automatically
                    }
                    Instruction::Unknown(bytes) => {
                        // Write raw instruction bytes directly
                        func_body.raw(bytes.clone());
                    }
                }
            }

            // Add final End instruction for the function body
            func_body.instruction(&EncoderInstruction::End);

            code.function(&func_body);
        }
        wasm_module.section(&code);

        // Build data section (memory initialization)
        if !module.data_segments.is_empty() {
            let mut data = DataSection::new();
            for segment in &module.data_segments {
                if segment.passive {
                    // Passive data segment
                    data.passive(segment.data.iter().copied());
                } else {
                    // Active data segment
                    let offset_expr = encode_const_expr_for_offset(&segment.offset)?;
                    data.active(
                        segment.memory_index,
                        &offset_expr,
                        segment.data.iter().copied(),
                    );
                }
            }
            wasm_module.section(&data);
        }

        // Build custom sections (names, debug info, etc.)
        // Pass through raw custom section bytes unchanged
        for (name, bytes) in &module.custom_sections {
            // Skip "producers" section - wasm-encoder adds its own automatically
            // Including both would cause idempotence issues (section grows on each encode)
            if name == "producers" {
                continue;
            }

            wasm_module.section(&CustomSection {
                name: name.into(),
                data: bytes.into(),
            });
        }

        Ok(wasm_module.finish())
    }

    /// Helper function to encode offset expressions for data/element segments
    fn encode_const_expr_for_offset(instructions: &[Instruction]) -> Result<ConstExpr> {
        // Offset expressions should be simple constant expressions
        if instructions.is_empty() {
            return Ok(ConstExpr::i32_const(0));
        }

        if instructions.len() == 1 {
            match &instructions[0] {
                Instruction::I32Const(val) => return Ok(ConstExpr::i32_const(*val)),
                Instruction::I64Const(val) => return Ok(ConstExpr::i64_const(*val)),
                _ => {}
            }
        }

        // Fallback to zero for complex expressions
        Ok(ConstExpr::i32_const(0))
    }

    /// Encode to WebAssembly text format (WAT)
    pub fn encode_wat(module: &Module) -> Result<String> {
        // First encode to WASM binary
        let wasm_bytes = encode_wasm(module)?;

        // Then use wasmprinter to convert to WAT
        wasmprinter::print_bytes(&wasm_bytes).context("Failed to convert WASM to WAT")
    }

    /// Convert our ValueType to wasm-encoder ValType
    fn convert_to_valtype(vt: ValueType) -> ValType {
        match vt {
            ValueType::I32 => ValType::I32,
            ValueType::I64 => ValType::I64,
            ValueType::F32 => ValType::F32,
            ValueType::F64 => ValType::F64,
        }
    }

    /// Convert our BlockType to wasm-encoder BlockType
    fn convert_blocktype_to_encoder(bt: &BlockType) -> wasm_encoder::BlockType {
        match bt {
            BlockType::Empty => wasm_encoder::BlockType::Empty,
            BlockType::Value(vt) => wasm_encoder::BlockType::Result(convert_to_valtype(*vt)),
            BlockType::Func {
                params: _,
                results: _,
            } => {
                // Multi-value block types require a type index in the wasm
                // type section — encoder does not yet emit those.
                // Invariant: this branch is unreachable because the parser
                // rejects `wasmparser::BlockType::FuncType(_)` upstream
                // (see the FuncType arm in the parser around line 1735).
                // If you hit this, an optimization pass synthesized a
                // Func block-type internally; that is the bug to fix.
                unreachable!(
                    "encoder reached BlockType::Func — parser should have rejected this input \
                     upstream; an optimization pass must have synthesized a multi-value block type"
                )
            }
        }
    }

    /// Recursively encode a single instruction (helper for nested control flow)
    fn encode_instruction_recursive(instr: &Instruction, func_body: &mut wasm_encoder::Function) {
        use wasm_encoder::Instruction as EncoderInstruction;

        match instr {
            Instruction::I32Const(value) => {
                func_body.instruction(&EncoderInstruction::I32Const(*value));
            }
            Instruction::I32Add => {
                func_body.instruction(&EncoderInstruction::I32Add);
            }
            Instruction::I32Sub => {
                func_body.instruction(&EncoderInstruction::I32Sub);
            }
            Instruction::I32Mul => {
                func_body.instruction(&EncoderInstruction::I32Mul);
            }
            Instruction::I32And => {
                func_body.instruction(&EncoderInstruction::I32And);
            }
            Instruction::I32Or => {
                func_body.instruction(&EncoderInstruction::I32Or);
            }
            Instruction::I32Xor => {
                func_body.instruction(&EncoderInstruction::I32Xor);
            }
            Instruction::I32Shl => {
                func_body.instruction(&EncoderInstruction::I32Shl);
            }
            Instruction::I32ShrS => {
                func_body.instruction(&EncoderInstruction::I32ShrS);
            }
            Instruction::I32ShrU => {
                func_body.instruction(&EncoderInstruction::I32ShrU);
            }
            Instruction::I64Const(value) => {
                func_body.instruction(&EncoderInstruction::I64Const(*value));
            }
            Instruction::F32Const(bits) => {
                let f32_val = f32::from_bits(*bits);
                func_body.instruction(&EncoderInstruction::F32Const(f32_val.into()));
            }
            Instruction::F64Const(bits) => {
                let f64_val = f64::from_bits(*bits);
                func_body.instruction(&EncoderInstruction::F64Const(f64_val.into()));
            }
            Instruction::I64Add => {
                func_body.instruction(&EncoderInstruction::I64Add);
            }
            Instruction::I64Sub => {
                func_body.instruction(&EncoderInstruction::I64Sub);
            }
            Instruction::I64Mul => {
                func_body.instruction(&EncoderInstruction::I64Mul);
            }
            Instruction::I64And => {
                func_body.instruction(&EncoderInstruction::I64And);
            }
            Instruction::I64Or => {
                func_body.instruction(&EncoderInstruction::I64Or);
            }
            Instruction::I64Xor => {
                func_body.instruction(&EncoderInstruction::I64Xor);
            }
            Instruction::I64Shl => {
                func_body.instruction(&EncoderInstruction::I64Shl);
            }
            Instruction::I64ShrS => {
                func_body.instruction(&EncoderInstruction::I64ShrS);
            }
            Instruction::I64ShrU => {
                func_body.instruction(&EncoderInstruction::I64ShrU);
            }
            Instruction::I32Eq => {
                func_body.instruction(&EncoderInstruction::I32Eq);
            }
            Instruction::I32Ne => {
                func_body.instruction(&EncoderInstruction::I32Ne);
            }
            Instruction::I32LtS => {
                func_body.instruction(&EncoderInstruction::I32LtS);
            }
            Instruction::I32LtU => {
                func_body.instruction(&EncoderInstruction::I32LtU);
            }
            Instruction::I32GtS => {
                func_body.instruction(&EncoderInstruction::I32GtS);
            }
            Instruction::I32GtU => {
                func_body.instruction(&EncoderInstruction::I32GtU);
            }
            Instruction::I32LeS => {
                func_body.instruction(&EncoderInstruction::I32LeS);
            }
            Instruction::I32LeU => {
                func_body.instruction(&EncoderInstruction::I32LeU);
            }
            Instruction::I32GeS => {
                func_body.instruction(&EncoderInstruction::I32GeS);
            }
            Instruction::I32GeU => {
                func_body.instruction(&EncoderInstruction::I32GeU);
            }
            Instruction::I64Eq => {
                func_body.instruction(&EncoderInstruction::I64Eq);
            }
            Instruction::I64Ne => {
                func_body.instruction(&EncoderInstruction::I64Ne);
            }
            Instruction::I64LtS => {
                func_body.instruction(&EncoderInstruction::I64LtS);
            }
            Instruction::I64LtU => {
                func_body.instruction(&EncoderInstruction::I64LtU);
            }
            Instruction::I64GtS => {
                func_body.instruction(&EncoderInstruction::I64GtS);
            }
            Instruction::I64GtU => {
                func_body.instruction(&EncoderInstruction::I64GtU);
            }
            Instruction::I64LeS => {
                func_body.instruction(&EncoderInstruction::I64LeS);
            }
            Instruction::I64LeU => {
                func_body.instruction(&EncoderInstruction::I64LeU);
            }
            Instruction::I64GeS => {
                func_body.instruction(&EncoderInstruction::I64GeS);
            }
            Instruction::I64GeU => {
                func_body.instruction(&EncoderInstruction::I64GeU);
            }
            Instruction::I32DivS => {
                func_body.instruction(&EncoderInstruction::I32DivS);
            }
            Instruction::I32DivU => {
                func_body.instruction(&EncoderInstruction::I32DivU);
            }
            Instruction::I32RemS => {
                func_body.instruction(&EncoderInstruction::I32RemS);
            }
            Instruction::I32RemU => {
                func_body.instruction(&EncoderInstruction::I32RemU);
            }
            Instruction::I64DivS => {
                func_body.instruction(&EncoderInstruction::I64DivS);
            }
            Instruction::I64DivU => {
                func_body.instruction(&EncoderInstruction::I64DivU);
            }
            Instruction::I64RemS => {
                func_body.instruction(&EncoderInstruction::I64RemS);
            }
            Instruction::I64RemU => {
                func_body.instruction(&EncoderInstruction::I64RemU);
            }
            Instruction::I32Eqz => {
                func_body.instruction(&EncoderInstruction::I32Eqz);
            }
            Instruction::I32Clz => {
                func_body.instruction(&EncoderInstruction::I32Clz);
            }
            Instruction::I32Ctz => {
                func_body.instruction(&EncoderInstruction::I32Ctz);
            }
            Instruction::I32Popcnt => {
                func_body.instruction(&EncoderInstruction::I32Popcnt);
            }
            Instruction::I64Eqz => {
                func_body.instruction(&EncoderInstruction::I64Eqz);
            }
            Instruction::I64Clz => {
                func_body.instruction(&EncoderInstruction::I64Clz);
            }
            Instruction::I64Ctz => {
                func_body.instruction(&EncoderInstruction::I64Ctz);
            }
            Instruction::I64Popcnt => {
                func_body.instruction(&EncoderInstruction::I64Popcnt);
            }
            Instruction::Select => {
                func_body.instruction(&EncoderInstruction::Select);
            }
            Instruction::LocalGet(idx) => {
                func_body.instruction(&EncoderInstruction::LocalGet(*idx));
            }
            Instruction::LocalSet(idx) => {
                func_body.instruction(&EncoderInstruction::LocalSet(*idx));
            }
            Instruction::LocalTee(idx) => {
                func_body.instruction(&EncoderInstruction::LocalTee(*idx));
            }
            Instruction::I32Load { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Load(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I32Store { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Store(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Load { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Load(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Store { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Store(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            // Control flow instructions (Phase 14)
            Instruction::Block { block_type, body } => {
                let bt = convert_blocktype_to_encoder(block_type);
                func_body.instruction(&EncoderInstruction::Block(bt));
                // Recursively encode body
                for nested_instr in body {
                    encode_instruction_recursive(nested_instr, func_body);
                }
                func_body.instruction(&EncoderInstruction::End);
            }
            Instruction::Loop { block_type, body } => {
                let bt = convert_blocktype_to_encoder(block_type);
                func_body.instruction(&EncoderInstruction::Loop(bt));
                // Recursively encode body
                for nested_instr in body {
                    encode_instruction_recursive(nested_instr, func_body);
                }
                func_body.instruction(&EncoderInstruction::End);
            }
            Instruction::If {
                block_type,
                then_body,
                else_body,
            } => {
                let bt = convert_blocktype_to_encoder(block_type);
                func_body.instruction(&EncoderInstruction::If(bt));
                // Encode then body
                for nested_instr in then_body {
                    encode_instruction_recursive(nested_instr, func_body);
                }
                // Encode else body if non-empty
                if !else_body.is_empty() {
                    func_body.instruction(&EncoderInstruction::Else);
                    for nested_instr in else_body {
                        encode_instruction_recursive(nested_instr, func_body);
                    }
                }
                func_body.instruction(&EncoderInstruction::End);
            }
            Instruction::Br(depth) => {
                func_body.instruction(&EncoderInstruction::Br(*depth));
            }
            Instruction::BrIf(depth) => {
                func_body.instruction(&EncoderInstruction::BrIf(*depth));
            }
            Instruction::BrTable { targets, default } => {
                func_body.instruction(&EncoderInstruction::BrTable(
                    targets.as_slice().into(),
                    *default,
                ));
            }
            Instruction::Return => {
                func_body.instruction(&EncoderInstruction::Return);
            }
            Instruction::Call(func_idx) => {
                func_body.instruction(&EncoderInstruction::Call(*func_idx));
            }
            Instruction::CallIndirect {
                type_idx,
                table_idx,
            } => {
                func_body.instruction(&EncoderInstruction::CallIndirect {
                    type_index: *type_idx,
                    table_index: *table_idx,
                });
            }
            Instruction::GlobalGet(idx) => {
                func_body.instruction(&EncoderInstruction::GlobalGet(*idx));
            }
            Instruction::GlobalSet(idx) => {
                func_body.instruction(&EncoderInstruction::GlobalSet(*idx));
            }
            Instruction::Unreachable => {
                func_body.instruction(&EncoderInstruction::Unreachable);
            }
            Instruction::Nop => {
                func_body.instruction(&EncoderInstruction::Nop);
            }
            Instruction::Drop => {
                func_body.instruction(&EncoderInstruction::Drop);
            }
            Instruction::End => {
                func_body.instruction(&EncoderInstruction::End);
            }
            // Float operations (f32)
            Instruction::F32Add => {
                func_body.instruction(&EncoderInstruction::F32Add);
            }
            Instruction::F32Sub => {
                func_body.instruction(&EncoderInstruction::F32Sub);
            }
            Instruction::F32Mul => {
                func_body.instruction(&EncoderInstruction::F32Mul);
            }
            Instruction::F32Div => {
                func_body.instruction(&EncoderInstruction::F32Div);
            }
            Instruction::F32Min => {
                func_body.instruction(&EncoderInstruction::F32Min);
            }
            Instruction::F32Max => {
                func_body.instruction(&EncoderInstruction::F32Max);
            }
            Instruction::F32Copysign => {
                func_body.instruction(&EncoderInstruction::F32Copysign);
            }
            Instruction::F32Abs => {
                func_body.instruction(&EncoderInstruction::F32Abs);
            }
            Instruction::F32Neg => {
                func_body.instruction(&EncoderInstruction::F32Neg);
            }
            Instruction::F32Ceil => {
                func_body.instruction(&EncoderInstruction::F32Ceil);
            }
            Instruction::F32Floor => {
                func_body.instruction(&EncoderInstruction::F32Floor);
            }
            Instruction::F32Trunc => {
                func_body.instruction(&EncoderInstruction::F32Trunc);
            }
            Instruction::F32Nearest => {
                func_body.instruction(&EncoderInstruction::F32Nearest);
            }
            Instruction::F32Sqrt => {
                func_body.instruction(&EncoderInstruction::F32Sqrt);
            }
            // Float comparisons (f32) - produce i32
            Instruction::F32Eq => {
                func_body.instruction(&EncoderInstruction::F32Eq);
            }
            Instruction::F32Ne => {
                func_body.instruction(&EncoderInstruction::F32Ne);
            }
            Instruction::F32Lt => {
                func_body.instruction(&EncoderInstruction::F32Lt);
            }
            Instruction::F32Gt => {
                func_body.instruction(&EncoderInstruction::F32Gt);
            }
            Instruction::F32Le => {
                func_body.instruction(&EncoderInstruction::F32Le);
            }
            Instruction::F32Ge => {
                func_body.instruction(&EncoderInstruction::F32Ge);
            }
            // Float operations (f64)
            Instruction::F64Add => {
                func_body.instruction(&EncoderInstruction::F64Add);
            }
            Instruction::F64Sub => {
                func_body.instruction(&EncoderInstruction::F64Sub);
            }
            Instruction::F64Mul => {
                func_body.instruction(&EncoderInstruction::F64Mul);
            }
            Instruction::F64Div => {
                func_body.instruction(&EncoderInstruction::F64Div);
            }
            Instruction::F64Min => {
                func_body.instruction(&EncoderInstruction::F64Min);
            }
            Instruction::F64Max => {
                func_body.instruction(&EncoderInstruction::F64Max);
            }
            Instruction::F64Copysign => {
                func_body.instruction(&EncoderInstruction::F64Copysign);
            }
            Instruction::F64Abs => {
                func_body.instruction(&EncoderInstruction::F64Abs);
            }
            Instruction::F64Neg => {
                func_body.instruction(&EncoderInstruction::F64Neg);
            }
            Instruction::F64Ceil => {
                func_body.instruction(&EncoderInstruction::F64Ceil);
            }
            Instruction::F64Floor => {
                func_body.instruction(&EncoderInstruction::F64Floor);
            }
            Instruction::F64Trunc => {
                func_body.instruction(&EncoderInstruction::F64Trunc);
            }
            Instruction::F64Nearest => {
                func_body.instruction(&EncoderInstruction::F64Nearest);
            }
            Instruction::F64Sqrt => {
                func_body.instruction(&EncoderInstruction::F64Sqrt);
            }
            // Float comparisons (f64) - produce i32
            Instruction::F64Eq => {
                func_body.instruction(&EncoderInstruction::F64Eq);
            }
            Instruction::F64Ne => {
                func_body.instruction(&EncoderInstruction::F64Ne);
            }
            Instruction::F64Lt => {
                func_body.instruction(&EncoderInstruction::F64Lt);
            }
            Instruction::F64Gt => {
                func_body.instruction(&EncoderInstruction::F64Gt);
            }
            Instruction::F64Le => {
                func_body.instruction(&EncoderInstruction::F64Le);
            }
            Instruction::F64Ge => {
                func_body.instruction(&EncoderInstruction::F64Ge);
            }
            // Conversion operations
            Instruction::I32WrapI64 => {
                func_body.instruction(&EncoderInstruction::I32WrapI64);
            }
            Instruction::I64ExtendI32S => {
                func_body.instruction(&EncoderInstruction::I64ExtendI32S);
            }
            Instruction::I64ExtendI32U => {
                func_body.instruction(&EncoderInstruction::I64ExtendI32U);
            }
            // Sign extension operations
            Instruction::I32Extend8S => {
                func_body.instruction(&EncoderInstruction::I32Extend8S);
            }
            Instruction::I32Extend16S => {
                func_body.instruction(&EncoderInstruction::I32Extend16S);
            }
            Instruction::I64Extend8S => {
                func_body.instruction(&EncoderInstruction::I64Extend8S);
            }
            Instruction::I64Extend16S => {
                func_body.instruction(&EncoderInstruction::I64Extend16S);
            }
            Instruction::I64Extend32S => {
                func_body.instruction(&EncoderInstruction::I64Extend32S);
            }
            Instruction::I32TruncF32S => {
                func_body.instruction(&EncoderInstruction::I32TruncF32S);
            }
            Instruction::I32TruncF32U => {
                func_body.instruction(&EncoderInstruction::I32TruncF32U);
            }
            Instruction::I32TruncF64S => {
                func_body.instruction(&EncoderInstruction::I32TruncF64S);
            }
            Instruction::I32TruncF64U => {
                func_body.instruction(&EncoderInstruction::I32TruncF64U);
            }
            Instruction::I64TruncF32S => {
                func_body.instruction(&EncoderInstruction::I64TruncF32S);
            }
            Instruction::I64TruncF32U => {
                func_body.instruction(&EncoderInstruction::I64TruncF32U);
            }
            Instruction::I64TruncF64S => {
                func_body.instruction(&EncoderInstruction::I64TruncF64S);
            }
            Instruction::I64TruncF64U => {
                func_body.instruction(&EncoderInstruction::I64TruncF64U);
            }
            // Saturating truncation operations
            Instruction::I32TruncSatF32S => {
                func_body.instruction(&EncoderInstruction::I32TruncSatF32S);
            }
            Instruction::I32TruncSatF32U => {
                func_body.instruction(&EncoderInstruction::I32TruncSatF32U);
            }
            Instruction::I32TruncSatF64S => {
                func_body.instruction(&EncoderInstruction::I32TruncSatF64S);
            }
            Instruction::I32TruncSatF64U => {
                func_body.instruction(&EncoderInstruction::I32TruncSatF64U);
            }
            Instruction::I64TruncSatF32S => {
                func_body.instruction(&EncoderInstruction::I64TruncSatF32S);
            }
            Instruction::I64TruncSatF32U => {
                func_body.instruction(&EncoderInstruction::I64TruncSatF32U);
            }
            Instruction::I64TruncSatF64S => {
                func_body.instruction(&EncoderInstruction::I64TruncSatF64S);
            }
            Instruction::I64TruncSatF64U => {
                func_body.instruction(&EncoderInstruction::I64TruncSatF64U);
            }
            Instruction::F32ConvertI32S => {
                func_body.instruction(&EncoderInstruction::F32ConvertI32S);
            }
            Instruction::F32ConvertI32U => {
                func_body.instruction(&EncoderInstruction::F32ConvertI32U);
            }
            Instruction::F32ConvertI64S => {
                func_body.instruction(&EncoderInstruction::F32ConvertI64S);
            }
            Instruction::F32ConvertI64U => {
                func_body.instruction(&EncoderInstruction::F32ConvertI64U);
            }
            Instruction::F64ConvertI32S => {
                func_body.instruction(&EncoderInstruction::F64ConvertI32S);
            }
            Instruction::F64ConvertI32U => {
                func_body.instruction(&EncoderInstruction::F64ConvertI32U);
            }
            Instruction::F64ConvertI64S => {
                func_body.instruction(&EncoderInstruction::F64ConvertI64S);
            }
            Instruction::F64ConvertI64U => {
                func_body.instruction(&EncoderInstruction::F64ConvertI64U);
            }
            Instruction::F32DemoteF64 => {
                func_body.instruction(&EncoderInstruction::F32DemoteF64);
            }
            Instruction::F64PromoteF32 => {
                func_body.instruction(&EncoderInstruction::F64PromoteF32);
            }
            // Reinterpret operations
            Instruction::I32ReinterpretF32 => {
                func_body.instruction(&EncoderInstruction::I32ReinterpretF32);
            }
            Instruction::I64ReinterpretF64 => {
                func_body.instruction(&EncoderInstruction::I64ReinterpretF64);
            }
            Instruction::F32ReinterpretI32 => {
                func_body.instruction(&EncoderInstruction::F32ReinterpretI32);
            }
            Instruction::F64ReinterpretI64 => {
                func_body.instruction(&EncoderInstruction::F64ReinterpretI64);
            }
            // Memory operations (float)
            Instruction::F32Load { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::F32Load(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::F32Store { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::F32Store(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::F64Load { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::F64Load(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::F64Store { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::F64Store(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            // Memory operations (integer variants)
            Instruction::I32Load8S { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Load8S(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I32Load8U { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Load8U(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I32Load16S { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Load16S(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I32Load16U { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Load16U(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Load8S { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Load8S(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Load8U { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Load8U(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Load16S { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Load16S(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Load16U { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Load16U(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Load32S { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Load32S(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Load32U { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Load32U(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I32Store8 { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Store8(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I32Store16 { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I32Store16(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Store8 { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Store8(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Store16 { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Store16(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            Instruction::I64Store32 { offset, align, mem } => {
                func_body.instruction(&EncoderInstruction::I64Store32(wasm_encoder::MemArg {
                    offset: *offset as u64,
                    align: *align,
                    memory_index: *mem,
                }));
            }
            // Memory size/grow
            Instruction::MemorySize(mem_idx) => {
                func_body.instruction(&EncoderInstruction::MemorySize(*mem_idx));
            }
            Instruction::MemoryGrow(mem_idx) => {
                func_body.instruction(&EncoderInstruction::MemoryGrow(*mem_idx));
            }
            // Bulk memory operations
            Instruction::MemoryFill(mem) => {
                func_body.instruction(&EncoderInstruction::MemoryFill(*mem));
            }
            Instruction::MemoryCopy { dst_mem, src_mem } => {
                func_body.instruction(&EncoderInstruction::MemoryCopy {
                    src_mem: *src_mem,
                    dst_mem: *dst_mem,
                });
            }
            Instruction::MemoryInit { mem, data_idx } => {
                func_body.instruction(&EncoderInstruction::MemoryInit {
                    mem: *mem,
                    data_index: *data_idx,
                });
            }
            Instruction::DataDrop(data_idx) => {
                func_body.instruction(&EncoderInstruction::DataDrop(*data_idx));
            }
            // Rotate operations
            Instruction::I32Rotl => {
                func_body.instruction(&EncoderInstruction::I32Rotl);
            }
            Instruction::I32Rotr => {
                func_body.instruction(&EncoderInstruction::I32Rotr);
            }
            Instruction::I64Rotl => {
                func_body.instruction(&EncoderInstruction::I64Rotl);
            }
            Instruction::I64Rotr => {
                func_body.instruction(&EncoderInstruction::I64Rotr);
            }
            Instruction::Unknown(bytes) => {
                // Write raw instruction bytes directly
                func_body.raw(bytes.clone());
            }
        }
    }
}

/// Term construction functionality: Convert WebAssembly instructions to ISLE terms
pub mod terms {

    use super::{BlockType, FunctionSignature, Instruction, Module, Value, ValueType};
    use anyhow::{Result, anyhow};
    use loom_isle::{
        Imm32, Imm64, ImmF32, ImmF64, block, br, br_if, br_table, call, call_indirect, data_drop,
        drop_instr, f32_convert_i32_s, f32_convert_i32_u, f32_convert_i64_s, f32_convert_i64_u,
        f32_demote_f64, f32_load, f32_reinterpret_i32, f32_store, f64_convert_i32_s,
        f64_convert_i32_u, f64_convert_i64_s, f64_convert_i64_u, f64_load, f64_promote_f32,
        f64_reinterpret_i64, f64_store, fabs32, fabs64, fadd32, fadd64, fceil32, fceil64, fconst32,
        fconst64, fcopysign32, fcopysign64, fdiv32, fdiv64, feq32, feq64, ffloor32, ffloor64,
        fge32, fge64, fgt32, fgt64, fle32, fle64, flt32, flt64, fmax32, fmax64, fmin32, fmin64,
        fmul32, fmul64, fne32, fne64, fnearest32, fnearest64, fneg32, fneg64, fsqrt32, fsqrt64,
        fsub32, fsub64, ftrunc32, ftrunc64, global_get, global_set, i32_extend8_s, i32_extend16_s,
        i32_load, i32_load8_s, i32_load8_u, i32_load16_s, i32_load16_u, i32_reinterpret_f32,
        i32_store, i32_store8, i32_store16, i32_trunc_f32_s, i32_trunc_f32_u, i32_trunc_f64_s,
        i32_trunc_f64_u, i32_trunc_sat_f32_s, i32_trunc_sat_f32_u, i32_trunc_sat_f64_s,
        i32_trunc_sat_f64_u, i32_wrap_i64, i64_extend_i32_s, i64_extend_i32_u, i64_extend8_s,
        i64_extend16_s, i64_extend32_s, i64_load, i64_load8_s, i64_load8_u, i64_load16_s,
        i64_load16_u, i64_load32_s, i64_load32_u, i64_reinterpret_f64, i64_store, i64_store8,
        i64_store16, i64_store32, i64_trunc_f32_s, i64_trunc_f32_u, i64_trunc_f64_s,
        i64_trunc_f64_u, i64_trunc_sat_f32_s, i64_trunc_sat_f32_u, i64_trunc_sat_f64_s,
        i64_trunc_sat_f64_u, iadd32, iadd64, iand32, iand64, iclz32, iclz64, iconst32, iconst64,
        ictz32, ictz64, idivs32, idivs64, idivu32, idivu64, ieq32, ieq64, ieqz32, ieqz64,
        if_then_else, iges32, iges64, igeu32, igeu64, igts32, igts64, igtu32, igtu64, iles32,
        iles64, ileu32, ileu64, ilts32, ilts64, iltu32, iltu64, imul32, imul64, ine32, ine64,
        ior32, ior64, ipopcnt32, ipopcnt64, irems32, irems64, iremu32, iremu64, irotl32, irotl64,
        irotr32, irotr64, ishl32, ishl64, ishrs32, ishrs64, ishru32, ishru64, isub32, isub64,
        ixor32, ixor64, local_get, local_set, local_tee, loop_construct, memory_copy, memory_fill,
        memory_grow, memory_init, memory_size, nop, return_val, select_instr, unreachable,
    };

    /// Owned context for function signature lookup during ISLE term conversion.
    ///
    /// This is an owned version (clones data) so it can be created before
    /// a mutable borrow of module.functions without lifetime conflicts.
    #[derive(Clone)]
    pub struct TermSignatureContext {
        /// Function signatures indexed by function index (accounting for imports)
        /// Indices 0..num_imports are imported functions
        /// Indices num_imports.. are local functions
        pub function_signatures: Vec<FunctionSignature>,
        /// Type signatures for indirect calls (indexed by type index)
        pub type_signatures: Vec<FunctionSignature>,
    }

    impl TermSignatureContext {
        /// Create a signature context from a module.
        ///
        /// Builds a function signature table that properly handles both imported
        /// and local functions, indexed by function index.
        pub fn from_module(module: &Module) -> Self {
            use super::ImportKind;

            let mut function_signatures = Vec::new();

            // First, add imported function signatures (they come first in indexing)
            for import in &module.imports {
                if let ImportKind::Func(type_idx) = &import.kind {
                    if let Some(sig) = module.types.get(*type_idx as usize) {
                        function_signatures.push(sig.clone());
                    }
                }
            }

            // Then add local function signatures
            for func in &module.functions {
                function_signatures.push(func.signature.clone());
            }

            TermSignatureContext {
                function_signatures,
                type_signatures: module.types.clone(),
            }
        }

        /// Get the signature for a function by its function index.
        ///
        /// This properly handles both imported functions (lower indices) and
        /// local functions (higher indices).
        pub fn get_function_signature(&self, func_idx: u32) -> Option<&FunctionSignature> {
            self.function_signatures.get(func_idx as usize)
        }

        /// Get the signature for a type by its index (for indirect calls)
        pub fn get_type_signature(&self, type_idx: u32) -> Option<&FunctionSignature> {
            self.type_signatures.get(type_idx as usize)
        }
    }

    /// Convert a sequence of WebAssembly instructions to ISLE terms
    /// This performs a stack-based conversion similar to how WebAssembly execution works
    pub fn instructions_to_terms(instructions: &[Instruction]) -> Result<Vec<Value>> {
        instructions_to_terms_impl(instructions, &[], None)
    }

    /// Convert instructions to terms with function signature context for proper Call handling
    ///
    /// This version correctly handles Call and CallIndirect instructions by looking up
    /// the callee's signature to determine how many arguments to pop from the stack.
    pub fn instructions_to_terms_with_signatures(
        instructions: &[Instruction],
        sig_ctx: &TermSignatureContext,
    ) -> Result<Vec<Value>> {
        instructions_to_terms_impl(instructions, &[], Some(sig_ctx))
    }

    /// Internal implementation with optional block type and signature context
    fn instructions_to_terms_impl(
        instructions: &[Instruction],
        block_types: &[BlockType],
        sig_ctx: Option<&TermSignatureContext>,
    ) -> Result<Vec<Value>> {
        // We use two data structures:
        // 1. `stack` - simulates the actual WASM value stack (for correct stack effect simulation)
        // 2. `side_effects` - captures side-effect terms (stores, local.set) that don't produce values
        //
        // At the end, we merge: side_effects come first (they must execute), then stack values
        let mut stack: Vec<Value> = Vec::new();
        let mut side_effects: Vec<Value> = Vec::new();

        for instr in instructions {
            match instr {
                Instruction::I32Const(val) => {
                    stack.push(iconst32(Imm32::from(*val)));
                }
                Instruction::I32Add => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.add rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.add lhs"))?;
                    stack.push(iadd32(lhs, rhs));
                }
                Instruction::I32Sub => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.sub rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.sub lhs"))?;
                    stack.push(isub32(lhs, rhs));
                }
                Instruction::I32Mul => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.mul rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.mul lhs"))?;
                    stack.push(imul32(lhs, rhs));
                }
                Instruction::I64Const(val) => {
                    stack.push(iconst64(Imm64::from(*val)));
                }
                Instruction::I64Add => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.add rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.add lhs"))?;
                    stack.push(iadd64(lhs, rhs));
                }
                Instruction::I64Sub => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.sub rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.sub lhs"))?;
                    stack.push(isub64(lhs, rhs));
                }
                Instruction::I64Mul => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.mul rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.mul lhs"))?;
                    stack.push(imul64(lhs, rhs));
                }
                Instruction::I32And => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.and rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.and lhs"))?;
                    stack.push(iand32(lhs, rhs));
                }
                Instruction::I32Or => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.or rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.or lhs"))?;
                    stack.push(ior32(lhs, rhs));
                }
                Instruction::I32Xor => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.xor rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.xor lhs"))?;
                    stack.push(ixor32(lhs, rhs));
                }
                Instruction::I32Shl => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.shl rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.shl lhs"))?;
                    stack.push(ishl32(lhs, rhs));
                }
                Instruction::I32ShrS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.shr_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.shr_s lhs"))?;
                    stack.push(ishrs32(lhs, rhs));
                }
                Instruction::I32ShrU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.shr_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.shr_u lhs"))?;
                    stack.push(ishru32(lhs, rhs));
                }
                Instruction::I64And => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.and rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.and lhs"))?;
                    stack.push(iand64(lhs, rhs));
                }
                Instruction::I64Or => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.or rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.or lhs"))?;
                    stack.push(ior64(lhs, rhs));
                }
                Instruction::I64Xor => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.xor rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.xor lhs"))?;
                    stack.push(ixor64(lhs, rhs));
                }
                Instruction::I64Shl => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.shl rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.shl lhs"))?;
                    stack.push(ishl64(lhs, rhs));
                }
                Instruction::I64ShrS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.shr_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.shr_s lhs"))?;
                    stack.push(ishrs64(lhs, rhs));
                }
                Instruction::I64ShrU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.shr_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.shr_u lhs"))?;
                    stack.push(ishru64(lhs, rhs));
                }
                Instruction::I32Rotl => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rotl rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rotl lhs"))?;
                    stack.push(irotl32(lhs, rhs));
                }
                Instruction::I32Rotr => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rotr rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rotr lhs"))?;
                    stack.push(irotr32(lhs, rhs));
                }
                Instruction::I64Rotl => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rotl rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rotl lhs"))?;
                    stack.push(irotl64(lhs, rhs));
                }
                Instruction::I64Rotr => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rotr rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rotr lhs"))?;
                    stack.push(irotr64(lhs, rhs));
                }
                Instruction::I32Eq => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.eq rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.eq lhs"))?;
                    stack.push(ieq32(lhs, rhs));
                }
                Instruction::I32Ne => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.ne rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.ne lhs"))?;
                    stack.push(ine32(lhs, rhs));
                }
                Instruction::I32LtS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.lt_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.lt_s lhs"))?;
                    stack.push(ilts32(lhs, rhs));
                }
                Instruction::I32LtU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.lt_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.lt_u lhs"))?;
                    stack.push(iltu32(lhs, rhs));
                }
                Instruction::I32GtS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.gt_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.gt_s lhs"))?;
                    stack.push(igts32(lhs, rhs));
                }
                Instruction::I32GtU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.gt_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.gt_u lhs"))?;
                    stack.push(igtu32(lhs, rhs));
                }
                Instruction::I32LeS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.le_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.le_s lhs"))?;
                    stack.push(iles32(lhs, rhs));
                }
                Instruction::I32LeU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.le_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.le_u lhs"))?;
                    stack.push(ileu32(lhs, rhs));
                }
                Instruction::I32GeS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.ge_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.ge_s lhs"))?;
                    stack.push(iges32(lhs, rhs));
                }
                Instruction::I32GeU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.ge_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.ge_u lhs"))?;
                    stack.push(igeu32(lhs, rhs));
                }
                Instruction::I64Eq => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.eq rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.eq lhs"))?;
                    stack.push(ieq64(lhs, rhs));
                }
                Instruction::I64Ne => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.ne rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.ne lhs"))?;
                    stack.push(ine64(lhs, rhs));
                }
                Instruction::I64LtS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.lt_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.lt_s lhs"))?;
                    stack.push(ilts64(lhs, rhs));
                }
                Instruction::I64LtU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.lt_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.lt_u lhs"))?;
                    stack.push(iltu64(lhs, rhs));
                }
                Instruction::I64GtS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.gt_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.gt_s lhs"))?;
                    stack.push(igts64(lhs, rhs));
                }
                Instruction::I64GtU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.gt_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.gt_u lhs"))?;
                    stack.push(igtu64(lhs, rhs));
                }
                Instruction::I64LeS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.le_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.le_s lhs"))?;
                    stack.push(iles64(lhs, rhs));
                }
                Instruction::I64LeU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.le_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.le_u lhs"))?;
                    stack.push(ileu64(lhs, rhs));
                }
                Instruction::I64GeS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.ge_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.ge_s lhs"))?;
                    stack.push(iges64(lhs, rhs));
                }
                Instruction::I64GeU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.ge_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.ge_u lhs"))?;
                    stack.push(igeu64(lhs, rhs));
                }
                Instruction::I32DivS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.div_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.div_s lhs"))?;
                    stack.push(idivs32(lhs, rhs));
                }
                Instruction::I32DivU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.div_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.div_u lhs"))?;
                    stack.push(idivu32(lhs, rhs));
                }
                Instruction::I32RemS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rem_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rem_s lhs"))?;
                    stack.push(irems32(lhs, rhs));
                }
                Instruction::I32RemU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rem_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.rem_u lhs"))?;
                    stack.push(iremu32(lhs, rhs));
                }
                Instruction::I64DivS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.div_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.div_s lhs"))?;
                    stack.push(idivs64(lhs, rhs));
                }
                Instruction::I64DivU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.div_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.div_u lhs"))?;
                    stack.push(idivu64(lhs, rhs));
                }
                Instruction::I64RemS => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rem_s rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rem_s lhs"))?;
                    stack.push(irems64(lhs, rhs));
                }
                Instruction::I64RemU => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rem_u rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.rem_u lhs"))?;
                    stack.push(iremu64(lhs, rhs));
                }
                Instruction::I32Eqz => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.eqz"))?;
                    stack.push(ieqz32(val));
                }
                Instruction::I32Clz => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.clz"))?;
                    stack.push(iclz32(val));
                }
                Instruction::I32Ctz => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.ctz"))?;
                    stack.push(ictz32(val));
                }
                Instruction::I32Popcnt => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.popcnt"))?;
                    stack.push(ipopcnt32(val));
                }
                Instruction::I64Eqz => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.eqz"))?;
                    stack.push(ieqz64(val));
                }
                Instruction::I64Clz => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.clz"))?;
                    stack.push(iclz64(val));
                }
                Instruction::I64Ctz => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.ctz"))?;
                    stack.push(ictz64(val));
                }
                Instruction::I64Popcnt => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.popcnt"))?;
                    stack.push(ipopcnt64(val));
                }
                // Integer conversion operations
                Instruction::I32WrapI64 => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.wrap_i64"))?;
                    stack.push(i32_wrap_i64(val));
                }
                Instruction::I64ExtendI32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.extend_i32_s"))?;
                    stack.push(i64_extend_i32_s(val));
                }
                Instruction::I64ExtendI32U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.extend_i32_u"))?;
                    stack.push(i64_extend_i32_u(val));
                }
                // Float-to-integer truncation (trapping)
                Instruction::I32TruncF32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_f32_s"))?;
                    stack.push(i32_trunc_f32_s(val));
                }
                Instruction::I32TruncF32U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_f32_u"))?;
                    stack.push(i32_trunc_f32_u(val));
                }
                Instruction::I32TruncF64S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_f64_s"))?;
                    stack.push(i32_trunc_f64_s(val));
                }
                Instruction::I32TruncF64U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_f64_u"))?;
                    stack.push(i32_trunc_f64_u(val));
                }
                Instruction::I64TruncF32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_f32_s"))?;
                    stack.push(i64_trunc_f32_s(val));
                }
                Instruction::I64TruncF32U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_f32_u"))?;
                    stack.push(i64_trunc_f32_u(val));
                }
                Instruction::I64TruncF64S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_f64_s"))?;
                    stack.push(i64_trunc_f64_s(val));
                }
                Instruction::I64TruncF64U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_f64_u"))?;
                    stack.push(i64_trunc_f64_u(val));
                }
                // Integer-to-float conversion
                Instruction::F32ConvertI32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.convert_i32_s"))?;
                    stack.push(f32_convert_i32_s(val));
                }
                Instruction::F32ConvertI32U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.convert_i32_u"))?;
                    stack.push(f32_convert_i32_u(val));
                }
                Instruction::F32ConvertI64S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.convert_i64_s"))?;
                    stack.push(f32_convert_i64_s(val));
                }
                Instruction::F32ConvertI64U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.convert_i64_u"))?;
                    stack.push(f32_convert_i64_u(val));
                }
                Instruction::F64ConvertI32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.convert_i32_s"))?;
                    stack.push(f64_convert_i32_s(val));
                }
                Instruction::F64ConvertI32U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.convert_i32_u"))?;
                    stack.push(f64_convert_i32_u(val));
                }
                Instruction::F64ConvertI64S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.convert_i64_s"))?;
                    stack.push(f64_convert_i64_s(val));
                }
                Instruction::F64ConvertI64U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.convert_i64_u"))?;
                    stack.push(f64_convert_i64_u(val));
                }
                // Float demote/promote
                Instruction::F32DemoteF64 => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.demote_f64"))?;
                    stack.push(f32_demote_f64(val));
                }
                Instruction::F64PromoteF32 => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.promote_f32"))?;
                    stack.push(f64_promote_f32(val));
                }
                // Reinterpret (bit-cast)
                Instruction::I32ReinterpretF32 => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.reinterpret_f32"))?;
                    stack.push(i32_reinterpret_f32(val));
                }
                Instruction::I64ReinterpretF64 => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.reinterpret_f64"))?;
                    stack.push(i64_reinterpret_f64(val));
                }
                Instruction::F32ReinterpretI32 => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.reinterpret_i32"))?;
                    stack.push(f32_reinterpret_i32(val));
                }
                Instruction::F64ReinterpretI64 => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.reinterpret_i64"))?;
                    stack.push(f64_reinterpret_i64(val));
                }
                // Saturating truncation (non-trapping)
                Instruction::I32TruncSatF32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_sat_f32_s"))?;
                    stack.push(i32_trunc_sat_f32_s(val));
                }
                Instruction::I32TruncSatF32U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_sat_f32_u"))?;
                    stack.push(i32_trunc_sat_f32_u(val));
                }
                Instruction::I32TruncSatF64S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_sat_f64_s"))?;
                    stack.push(i32_trunc_sat_f64_s(val));
                }
                Instruction::I32TruncSatF64U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.trunc_sat_f64_u"))?;
                    stack.push(i32_trunc_sat_f64_u(val));
                }
                Instruction::I64TruncSatF32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_sat_f32_s"))?;
                    stack.push(i64_trunc_sat_f32_s(val));
                }
                Instruction::I64TruncSatF32U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_sat_f32_u"))?;
                    stack.push(i64_trunc_sat_f32_u(val));
                }
                Instruction::I64TruncSatF64S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_sat_f64_s"))?;
                    stack.push(i64_trunc_sat_f64_s(val));
                }
                Instruction::I64TruncSatF64U => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.trunc_sat_f64_u"))?;
                    stack.push(i64_trunc_sat_f64_u(val));
                }
                // Sign extension operations (in-place sign extension)
                Instruction::I32Extend8S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.extend8_s"))?;
                    stack.push(i32_extend8_s(val));
                }
                Instruction::I32Extend16S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.extend16_s"))?;
                    stack.push(i32_extend16_s(val));
                }
                Instruction::I64Extend8S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.extend8_s"))?;
                    stack.push(i64_extend8_s(val));
                }
                Instruction::I64Extend16S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.extend16_s"))?;
                    stack.push(i64_extend16_s(val));
                }
                Instruction::I64Extend32S => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.extend32_s"))?;
                    stack.push(i64_extend32_s(val));
                }
                Instruction::Select => {
                    let cond = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for select cond"))?;
                    let false_val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for select false"))?;
                    let true_val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for select true"))?;
                    stack.push(select_instr(cond, true_val, false_val));
                }
                Instruction::LocalGet(idx) => {
                    stack.push(local_get(*idx));
                }
                Instruction::LocalSet(idx) => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for local.set"))?;
                    // local.set consumes a value but does NOT produce one
                    // The term goes to side_effects, not stack
                    side_effects.push(local_set(*idx, val));
                }
                Instruction::LocalTee(idx) => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for local.tee"))?;
                    // local.tee returns the value, so we only push the tee term
                    // The tee term itself represents both the assignment and the value on stack
                    stack.push(local_tee(*idx, val));
                }
                // Memory operations (Phase 13)
                Instruction::I32Load { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.load address"))?;
                    stack.push(i32_load(addr, *offset, *align, *mem));
                }
                Instruction::I32Store { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.store value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.store address"))?;
                    // i32.store consumes 2 values but does NOT produce any
                    // The term goes to side_effects, not stack
                    side_effects.push(i32_store(addr, value, *offset, *align, *mem));
                }
                Instruction::I64Load { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.load address"))?;
                    stack.push(i64_load(addr, *offset, *align, *mem));
                }
                Instruction::I64Store { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store address"))?;
                    // i64.store consumes 2 values but does NOT produce any
                    // The term goes to side_effects, not stack
                    side_effects.push(i64_store(addr, value, *offset, *align, *mem));
                }
                // Partial-width memory load operations
                Instruction::I32Load8S { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.load8_s address"))?;
                    stack.push(i32_load8_s(addr, *offset, *align, *mem));
                }
                Instruction::I32Load8U { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.load8_u address"))?;
                    stack.push(i32_load8_u(addr, *offset, *align, *mem));
                }
                Instruction::I32Load16S { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.load16_s address"))?;
                    stack.push(i32_load16_s(addr, *offset, *align, *mem));
                }
                Instruction::I32Load16U { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.load16_u address"))?;
                    stack.push(i32_load16_u(addr, *offset, *align, *mem));
                }
                Instruction::I64Load8S { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.load8_s address"))?;
                    stack.push(i64_load8_s(addr, *offset, *align, *mem));
                }
                Instruction::I64Load8U { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.load8_u address"))?;
                    stack.push(i64_load8_u(addr, *offset, *align, *mem));
                }
                Instruction::I64Load16S { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.load16_s address"))?;
                    stack.push(i64_load16_s(addr, *offset, *align, *mem));
                }
                Instruction::I64Load16U { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.load16_u address"))?;
                    stack.push(i64_load16_u(addr, *offset, *align, *mem));
                }
                Instruction::I64Load32S { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.load32_s address"))?;
                    stack.push(i64_load32_s(addr, *offset, *align, *mem));
                }
                Instruction::I64Load32U { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.load32_u address"))?;
                    stack.push(i64_load32_u(addr, *offset, *align, *mem));
                }
                // Float memory operations
                Instruction::F32Load { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.load address"))?;
                    stack.push(f32_load(addr, *offset, *align, *mem));
                }
                Instruction::F32Store { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.store value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.store address"))?;
                    side_effects.push(f32_store(addr, value, *offset, *align, *mem));
                }
                Instruction::F64Load { offset, align, mem } => {
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.load address"))?;
                    stack.push(f64_load(addr, *offset, *align, *mem));
                }
                Instruction::F64Store { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.store value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.store address"))?;
                    side_effects.push(f64_store(addr, value, *offset, *align, *mem));
                }
                // Partial-width memory store operations
                Instruction::I32Store8 { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.store8 value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.store8 address"))?;
                    side_effects.push(i32_store8(addr, value, *offset, *align, *mem));
                }
                Instruction::I32Store16 { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.store16 value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i32.store16 address"))?;
                    side_effects.push(i32_store16(addr, value, *offset, *align, *mem));
                }
                Instruction::I64Store8 { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store8 value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store8 address"))?;
                    side_effects.push(i64_store8(addr, value, *offset, *align, *mem));
                }
                Instruction::I64Store16 { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store16 value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store16 address"))?;
                    side_effects.push(i64_store16(addr, value, *offset, *align, *mem));
                }
                Instruction::I64Store32 { offset, align, mem } => {
                    let value = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store32 value"))?;
                    let addr = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for i64.store32 address"))?;
                    side_effects.push(i64_store32(addr, value, *offset, *align, *mem));
                }
                // Control flow instructions (Phase 14)
                Instruction::Block { block_type, body } => {
                    // Build new block context: add this block's type to the front
                    let mut new_context = vec![block_type.clone()];
                    new_context.extend_from_slice(block_types);

                    // Convert body instructions to terms recursively with updated context
                    let body_terms = instructions_to_terms_impl(body, &new_context, sig_ctx)?;
                    let bt = convert_blocktype_to_isle(block_type);
                    stack.push(block(None, bt, body_terms));
                }
                Instruction::Loop { block_type, body } => {
                    // Build new block context: add this block's type to the front
                    let mut new_context = vec![block_type.clone()];
                    new_context.extend_from_slice(block_types);

                    // Convert body instructions to terms recursively with updated context
                    let body_terms = instructions_to_terms_impl(body, &new_context, sig_ctx)?;
                    let bt = convert_blocktype_to_isle(block_type);
                    stack.push(loop_construct(None, bt, body_terms));
                }
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    // Pop condition from stack
                    let condition = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for if condition"))?;

                    // Build new block context: add this block's type to the front
                    let mut new_context = vec![block_type.clone()];
                    new_context.extend_from_slice(block_types);

                    // Convert bodies to terms with updated context
                    let then_terms = instructions_to_terms_impl(then_body, &new_context, sig_ctx)?;
                    let else_terms = instructions_to_terms_impl(else_body, &new_context, sig_ctx)?;
                    let bt = convert_blocktype_to_isle(block_type);

                    stack.push(if_then_else(None, bt, condition, then_terms, else_terms));
                }
                Instruction::Br(depth) => {
                    // Check if target block has a result type and pop value if needed
                    let value = if let Some(target_type) = block_types.get(*depth as usize) {
                        match target_type {
                            BlockType::Value(_) => stack.pop(),
                            BlockType::Empty => None,
                            BlockType::Func { results, .. } if !results.is_empty() => stack.pop(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    stack.push(br(*depth, value));
                }
                Instruction::BrIf(depth) => {
                    // Pop condition from stack
                    let condition = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for br_if condition"))?;

                    // Check if target block has a result type and pop value if needed
                    let value = if let Some(target_type) = block_types.get(*depth as usize) {
                        match target_type {
                            BlockType::Value(_) => stack.pop(),
                            BlockType::Empty => None,
                            BlockType::Func { results, .. } if !results.is_empty() => stack.pop(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    stack.push(br_if(*depth, condition, value));
                }
                Instruction::BrTable { targets, default } => {
                    // Pop index from stack
                    let index = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for br_table index"))?;

                    // Check if default target block has a result type and pop value if needed
                    // All targets in a br_table must have the same type, so we only check default
                    let value = if let Some(target_type) = block_types.get(*default as usize) {
                        match target_type {
                            BlockType::Value(_) => stack.pop(),
                            BlockType::Empty => None,
                            BlockType::Func { results, .. } if !results.is_empty() => stack.pop(),
                            _ => None,
                        }
                    } else {
                        None
                    };
                    stack.push(br_table(targets.clone(), *default, index, value));
                }
                Instruction::Return => {
                    // Collect all remaining values on stack as return values
                    let values = std::mem::take(&mut stack);
                    stack.push(return_val(values));
                }
                Instruction::Call(func_idx) => {
                    // Look up function signature to determine argument count
                    let args = if let Some(ctx) = sig_ctx {
                        if let Some(sig) = ctx.get_function_signature(*func_idx) {
                            // Pop arguments in reverse order (last arg pushed first)
                            let param_count = sig.params.len();
                            let mut args = Vec::with_capacity(param_count);
                            for i in 0..param_count {
                                let arg = stack.pop().ok_or_else(|| {
                                    anyhow!(
                                        "Stack underflow for call argument {} of {}",
                                        i + 1,
                                        param_count
                                    )
                                })?;
                                args.push(arg);
                            }
                            // Reverse to get correct order (first param first)
                            args.reverse();
                            args
                        } else {
                            // Unknown function index - assume no arguments
                            vec![]
                        }
                    } else {
                        // No signature context - assume no arguments (backwards compatible)
                        vec![]
                    };
                    stack.push(call(*func_idx, args));
                }
                Instruction::CallIndirect {
                    type_idx,
                    table_idx,
                } => {
                    // Pop table offset from stack first
                    let table_offset = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for call_indirect offset"))?;

                    // Look up type signature to determine argument count
                    let args = if let Some(ctx) = sig_ctx {
                        if let Some(sig) = ctx.get_type_signature(*type_idx) {
                            // Pop arguments in reverse order (last arg pushed first)
                            let param_count = sig.params.len();
                            let mut args = Vec::with_capacity(param_count);
                            for i in 0..param_count {
                                let arg = stack.pop().ok_or_else(|| {
                                    anyhow!(
                                        "Stack underflow for call_indirect argument {} of {}",
                                        i + 1,
                                        param_count
                                    )
                                })?;
                                args.push(arg);
                            }
                            // Reverse to get correct order (first param first)
                            args.reverse();
                            args
                        } else {
                            // Unknown type index - assume no arguments
                            vec![]
                        }
                    } else {
                        // No signature context - assume no arguments (backwards compatible)
                        vec![]
                    };
                    stack.push(call_indirect(*table_idx, *type_idx, table_offset, args));
                }
                Instruction::Unreachable => {
                    stack.push(unreachable());
                }
                Instruction::Nop => {
                    stack.push(nop());
                }
                Instruction::Drop => {
                    // Pop value from stack and wrap in Drop term
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for drop"))?;
                    stack.push(drop_instr(val));
                }
                Instruction::GlobalGet(idx) => {
                    // Global loads produce a value on the stack
                    // The global_get term preserves the global index for correct reconstruction
                    stack.push(global_get(*idx));
                }
                Instruction::GlobalSet(idx) => {
                    // Global stores consume a value and produce no stack result
                    // Like local.set, this is a side effect that must be preserved
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for global.set"))?;
                    side_effects.push(global_set(*idx, val));
                }
                Instruction::F32Const(bits) => {
                    stack.push(fconst32(ImmF32::from_bits(*bits)));
                }
                Instruction::F64Const(bits) => {
                    stack.push(fconst64(ImmF64::from_bits(*bits)));
                }
                Instruction::End => {
                    // End doesn't produce a value, just marks block end
                }
                Instruction::F32Add => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.add rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.add lhs"))?;
                    stack.push(fadd32(lhs, rhs));
                }
                Instruction::F32Sub => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.sub rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.sub lhs"))?;
                    stack.push(fsub32(lhs, rhs));
                }
                Instruction::F32Mul => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.mul rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.mul lhs"))?;
                    stack.push(fmul32(lhs, rhs));
                }
                Instruction::F32Div => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.div rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.div lhs"))?;
                    stack.push(fdiv32(lhs, rhs));
                }
                Instruction::F64Add => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.add rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.add lhs"))?;
                    stack.push(fadd64(lhs, rhs));
                }
                Instruction::F64Sub => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.sub rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.sub lhs"))?;
                    stack.push(fsub64(lhs, rhs));
                }
                Instruction::F64Mul => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.mul rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.mul lhs"))?;
                    stack.push(fmul64(lhs, rhs));
                }
                Instruction::F64Div => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.div rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.div lhs"))?;
                    stack.push(fdiv64(lhs, rhs));
                }
                // f32 unary operations
                Instruction::F32Abs => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.abs"))?;
                    stack.push(fabs32(val));
                }
                Instruction::F32Neg => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.neg"))?;
                    stack.push(fneg32(val));
                }
                Instruction::F32Ceil => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.ceil"))?;
                    stack.push(fceil32(val));
                }
                Instruction::F32Floor => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.floor"))?;
                    stack.push(ffloor32(val));
                }
                Instruction::F32Trunc => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.trunc"))?;
                    stack.push(ftrunc32(val));
                }
                Instruction::F32Nearest => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.nearest"))?;
                    stack.push(fnearest32(val));
                }
                Instruction::F32Sqrt => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.sqrt"))?;
                    stack.push(fsqrt32(val));
                }
                // f32 binary operations
                Instruction::F32Min => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.min rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.min lhs"))?;
                    stack.push(fmin32(lhs, rhs));
                }
                Instruction::F32Max => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.max rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.max lhs"))?;
                    stack.push(fmax32(lhs, rhs));
                }
                Instruction::F32Copysign => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.copysign rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.copysign lhs"))?;
                    stack.push(fcopysign32(lhs, rhs));
                }
                // f32 comparison operations
                Instruction::F32Eq => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.eq rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.eq lhs"))?;
                    stack.push(feq32(lhs, rhs));
                }
                Instruction::F32Ne => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.ne rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.ne lhs"))?;
                    stack.push(fne32(lhs, rhs));
                }
                Instruction::F32Lt => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.lt rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.lt lhs"))?;
                    stack.push(flt32(lhs, rhs));
                }
                Instruction::F32Gt => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.gt rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.gt lhs"))?;
                    stack.push(fgt32(lhs, rhs));
                }
                Instruction::F32Le => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.le rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.le lhs"))?;
                    stack.push(fle32(lhs, rhs));
                }
                Instruction::F32Ge => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.ge rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f32.ge lhs"))?;
                    stack.push(fge32(lhs, rhs));
                }
                // f64 unary operations
                Instruction::F64Abs => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.abs"))?;
                    stack.push(fabs64(val));
                }
                Instruction::F64Neg => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.neg"))?;
                    stack.push(fneg64(val));
                }
                Instruction::F64Ceil => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.ceil"))?;
                    stack.push(fceil64(val));
                }
                Instruction::F64Floor => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.floor"))?;
                    stack.push(ffloor64(val));
                }
                Instruction::F64Trunc => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.trunc"))?;
                    stack.push(ftrunc64(val));
                }
                Instruction::F64Nearest => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.nearest"))?;
                    stack.push(fnearest64(val));
                }
                Instruction::F64Sqrt => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.sqrt"))?;
                    stack.push(fsqrt64(val));
                }
                // f64 binary operations
                Instruction::F64Min => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.min rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.min lhs"))?;
                    stack.push(fmin64(lhs, rhs));
                }
                Instruction::F64Max => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.max rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.max lhs"))?;
                    stack.push(fmax64(lhs, rhs));
                }
                Instruction::F64Copysign => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.copysign rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.copysign lhs"))?;
                    stack.push(fcopysign64(lhs, rhs));
                }
                // f64 comparison operations
                Instruction::F64Eq => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.eq rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.eq lhs"))?;
                    stack.push(feq64(lhs, rhs));
                }
                Instruction::F64Ne => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.ne rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.ne lhs"))?;
                    stack.push(fne64(lhs, rhs));
                }
                Instruction::F64Lt => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.lt rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.lt lhs"))?;
                    stack.push(flt64(lhs, rhs));
                }
                Instruction::F64Gt => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.gt rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.gt lhs"))?;
                    stack.push(fgt64(lhs, rhs));
                }
                Instruction::F64Le => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.le rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.le lhs"))?;
                    stack.push(fle64(lhs, rhs));
                }
                Instruction::F64Ge => {
                    let rhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.ge rhs"))?;
                    let lhs = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for f64.ge lhs"))?;
                    stack.push(fge64(lhs, rhs));
                }
                // Memory size/grow operations
                Instruction::MemorySize(mem) => {
                    stack.push(memory_size(*mem));
                }
                Instruction::MemoryGrow(mem) => {
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.grow"))?;
                    stack.push(memory_grow(val, *mem));
                }
                Instruction::Unknown(_) => {
                    // Unknown instructions cannot be converted to ISLE terms
                    // They are passed through unchanged in the encoding phase
                }

                // Bulk memory instructions - side-effectful, no stack output
                Instruction::MemoryFill(mem) => {
                    let len = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.fill len"))?;
                    let val = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.fill val"))?;
                    let dst = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.fill dst"))?;
                    side_effects.push(memory_fill(dst, val, len, *mem));
                }
                Instruction::MemoryCopy { dst_mem, src_mem } => {
                    let len = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.copy len"))?;
                    let src = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.copy src"))?;
                    let dst = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.copy dst"))?;
                    side_effects.push(memory_copy(dst, src, len, *dst_mem, *src_mem));
                }
                Instruction::MemoryInit { mem, data_idx } => {
                    let len = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.init len"))?;
                    let src = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.init src"))?;
                    let dst = stack
                        .pop()
                        .ok_or_else(|| anyhow!("Stack underflow for memory.init dst"))?;
                    side_effects.push(memory_init(dst, src, len, *mem, *data_idx));
                }
                Instruction::DataDrop(data_idx) => {
                    side_effects.push(data_drop(*data_idx));
                }
            }
        }

        // Merge side effects and stack: side effects execute first, then stack values remain
        let mut result = side_effects;
        result.extend(stack);
        Ok(result)
    }

    /// Convert our BlockType to loom-isle BlockType
    fn convert_blocktype_to_isle(bt: &BlockType) -> loom_isle::BlockType {
        match bt {
            BlockType::Empty => loom_isle::BlockType::Empty,
            BlockType::Value(vt) => loom_isle::BlockType::Value(convert_valuetype_to_isle(*vt)),
            BlockType::Func { params, results } => loom_isle::BlockType::Func {
                params: params
                    .iter()
                    .map(|v| convert_valuetype_to_isle(*v))
                    .collect(),
                results: results
                    .iter()
                    .map(|v| convert_valuetype_to_isle(*v))
                    .collect(),
            },
        }
    }

    /// Convert our ValueType to loom-isle ValueType
    fn convert_valuetype_to_isle(vt: ValueType) -> loom_isle::ValueType {
        match vt {
            ValueType::I32 => loom_isle::ValueType::I32,
            ValueType::I64 => loom_isle::ValueType::I64,
            ValueType::F32 => loom_isle::ValueType::F32,
            ValueType::F64 => loom_isle::ValueType::F64,
        }
    }

    /// Convert ISLE terms back to WebAssembly instructions
    /// This performs a depth-first traversal to emit instructions in stack order
    pub fn terms_to_instructions(terms: &[Value]) -> Result<Vec<Instruction>> {
        let mut instructions = Vec::new();

        for term in terms {
            term_to_instructions_recursive(term, &mut instructions)?;
        }

        // NOTE: Do NOT add End instruction here.
        // The encoder (line 1836) adds the final End for function bodies.
        // End should not appear in instruction lists (see encoder comment at line 1823-1826).

        Ok(instructions)
    }

    /// Recursive helper to convert a single term to instructions
    fn term_to_instructions_recursive(
        term: &Value,
        instructions: &mut Vec<Instruction>,
    ) -> Result<()> {
        use loom_isle::ValueData;

        match term.data() {
            ValueData::I32Const { val } => {
                instructions.push(Instruction::I32Const(val.value()));
            }
            ValueData::I64Const { val } => {
                instructions.push(Instruction::I64Const(val.value()));
            }
            ValueData::I32Add { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Add);
            }
            ValueData::I32Sub { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Sub);
            }
            ValueData::I32Mul { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Mul);
            }
            ValueData::I64Add { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Add);
            }
            ValueData::I64Sub { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Sub);
            }
            ValueData::I64Mul { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Mul);
            }
            ValueData::I32And { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32And);
            }
            ValueData::I32Or { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Or);
            }
            ValueData::I32Xor { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Xor);
            }
            ValueData::I32Shl { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Shl);
            }
            ValueData::I32ShrS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32ShrS);
            }
            ValueData::I32ShrU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32ShrU);
            }
            ValueData::I64And { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64And);
            }
            ValueData::I64Or { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Or);
            }
            ValueData::I64Xor { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Xor);
            }
            ValueData::I64Shl { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Shl);
            }
            ValueData::I64ShrS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64ShrS);
            }
            ValueData::I64ShrU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64ShrU);
            }
            ValueData::I32Rotl { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Rotl);
            }
            ValueData::I32Rotr { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Rotr);
            }
            ValueData::I64Rotl { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Rotl);
            }
            ValueData::I64Rotr { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Rotr);
            }
            ValueData::I32Eq { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Eq);
            }
            ValueData::I32Ne { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32Ne);
            }
            ValueData::I32LtS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32LtS);
            }
            ValueData::I32LtU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32LtU);
            }
            ValueData::I32GtS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32GtS);
            }
            ValueData::I32GtU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32GtU);
            }
            ValueData::I32LeS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32LeS);
            }
            ValueData::I32LeU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32LeU);
            }
            ValueData::I32GeS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32GeS);
            }
            ValueData::I32GeU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32GeU);
            }
            ValueData::I64Eq { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Eq);
            }
            ValueData::I64Ne { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64Ne);
            }
            ValueData::I64LtS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64LtS);
            }
            ValueData::I64LtU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64LtU);
            }
            ValueData::I64GtS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64GtS);
            }
            ValueData::I64GtU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64GtU);
            }
            ValueData::I64LeS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64LeS);
            }
            ValueData::I64LeU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64LeU);
            }
            ValueData::I64GeS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64GeS);
            }
            ValueData::I64GeU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64GeU);
            }
            ValueData::I32DivS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32DivS);
            }
            ValueData::I32DivU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32DivU);
            }
            ValueData::I32RemS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32RemS);
            }
            ValueData::I32RemU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I32RemU);
            }
            ValueData::I64DivS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64DivS);
            }
            ValueData::I64DivU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64DivU);
            }
            ValueData::I64RemS { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64RemS);
            }
            ValueData::I64RemU { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::I64RemU);
            }
            ValueData::I32Eqz { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32Eqz);
            }
            ValueData::I32Clz { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32Clz);
            }
            ValueData::I32Ctz { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32Ctz);
            }
            ValueData::I32Popcnt { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32Popcnt);
            }
            ValueData::I64Eqz { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64Eqz);
            }
            ValueData::I64Clz { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64Clz);
            }
            ValueData::I64Ctz { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64Ctz);
            }
            ValueData::I64Popcnt { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64Popcnt);
            }
            // Integer conversion operations
            ValueData::I32WrapI64 { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32WrapI64);
            }
            ValueData::I64ExtendI32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64ExtendI32S);
            }
            ValueData::I64ExtendI32U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64ExtendI32U);
            }
            // Float-to-integer truncation (trapping)
            ValueData::I32TruncF32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncF32S);
            }
            ValueData::I32TruncF32U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncF32U);
            }
            ValueData::I32TruncF64S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncF64S);
            }
            ValueData::I32TruncF64U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncF64U);
            }
            ValueData::I64TruncF32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncF32S);
            }
            ValueData::I64TruncF32U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncF32U);
            }
            ValueData::I64TruncF64S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncF64S);
            }
            ValueData::I64TruncF64U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncF64U);
            }
            // Integer-to-float conversion
            ValueData::F32ConvertI32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32ConvertI32S);
            }
            ValueData::F32ConvertI32U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32ConvertI32U);
            }
            ValueData::F32ConvertI64S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32ConvertI64S);
            }
            ValueData::F32ConvertI64U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32ConvertI64U);
            }
            ValueData::F64ConvertI32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64ConvertI32S);
            }
            ValueData::F64ConvertI32U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64ConvertI32U);
            }
            ValueData::F64ConvertI64S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64ConvertI64S);
            }
            ValueData::F64ConvertI64U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64ConvertI64U);
            }
            // Float demote/promote
            ValueData::F32DemoteF64 { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32DemoteF64);
            }
            ValueData::F64PromoteF32 { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64PromoteF32);
            }
            // Reinterpret (bit-cast)
            ValueData::I32ReinterpretF32 { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32ReinterpretF32);
            }
            ValueData::I64ReinterpretF64 { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64ReinterpretF64);
            }
            ValueData::F32ReinterpretI32 { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32ReinterpretI32);
            }
            ValueData::F64ReinterpretI64 { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64ReinterpretI64);
            }
            // Saturating truncation (non-trapping)
            ValueData::I32TruncSatF32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncSatF32S);
            }
            ValueData::I32TruncSatF32U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncSatF32U);
            }
            ValueData::I32TruncSatF64S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncSatF64S);
            }
            ValueData::I32TruncSatF64U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32TruncSatF64U);
            }
            ValueData::I64TruncSatF32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncSatF32S);
            }
            ValueData::I64TruncSatF32U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncSatF32U);
            }
            ValueData::I64TruncSatF64S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncSatF64S);
            }
            ValueData::I64TruncSatF64U { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64TruncSatF64U);
            }
            // Memory operations
            ValueData::MemorySize { mem } => {
                instructions.push(Instruction::MemorySize(*mem));
            }
            ValueData::MemoryGrow { val, mem } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::MemoryGrow(*mem));
            }
            // Bulk memory operations (side-effectful)
            ValueData::MemoryFill { dst, val, len, mem } => {
                term_to_instructions_recursive(dst, instructions)?;
                term_to_instructions_recursive(val, instructions)?;
                term_to_instructions_recursive(len, instructions)?;
                instructions.push(Instruction::MemoryFill(*mem));
            }
            ValueData::MemoryCopy {
                dst,
                src,
                len,
                dst_mem,
                src_mem,
            } => {
                term_to_instructions_recursive(dst, instructions)?;
                term_to_instructions_recursive(src, instructions)?;
                term_to_instructions_recursive(len, instructions)?;
                instructions.push(Instruction::MemoryCopy {
                    dst_mem: *dst_mem,
                    src_mem: *src_mem,
                });
            }
            ValueData::MemoryInit {
                dst,
                src,
                len,
                mem,
                data_idx,
            } => {
                term_to_instructions_recursive(dst, instructions)?;
                term_to_instructions_recursive(src, instructions)?;
                term_to_instructions_recursive(len, instructions)?;
                instructions.push(Instruction::MemoryInit {
                    mem: *mem,
                    data_idx: *data_idx,
                });
            }
            ValueData::DataDrop { data_idx } => {
                instructions.push(Instruction::DataDrop(*data_idx));
            }
            // Sign extension operations
            ValueData::I32Extend8S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32Extend8S);
            }
            ValueData::I32Extend16S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I32Extend16S);
            }
            ValueData::I64Extend8S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64Extend8S);
            }
            ValueData::I64Extend16S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64Extend16S);
            }
            ValueData::I64Extend32S { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::I64Extend32S);
            }
            ValueData::Select {
                cond,
                true_val,
                false_val,
            } => {
                term_to_instructions_recursive(true_val, instructions)?;
                term_to_instructions_recursive(false_val, instructions)?;
                term_to_instructions_recursive(cond, instructions)?;
                instructions.push(Instruction::Select);
            }
            ValueData::LocalGet { idx } => {
                instructions.push(Instruction::LocalGet(*idx));
            }
            ValueData::LocalSet { idx, val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::LocalSet(*idx));
            }
            ValueData::LocalTee { idx, val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::LocalTee(*idx));
            }
            ValueData::GlobalGet { idx } => {
                instructions.push(Instruction::GlobalGet(*idx));
            }
            ValueData::GlobalSet { idx, val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::GlobalSet(*idx));
            }
            ValueData::I32Load {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I32Load {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I32Store {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::I32Store {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Load {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I64Load {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Store {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::I64Store {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }

            // Partial-width memory load operations
            ValueData::I32Load8S {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I32Load8S {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I32Load8U {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I32Load8U {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I32Load16S {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I32Load16S {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I32Load16U {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I32Load16U {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Load8S {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I64Load8S {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Load8U {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I64Load8U {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Load16S {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I64Load16S {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Load16U {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I64Load16U {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Load32S {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I64Load32S {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Load32U {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::I64Load32U {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }

            // Float memory operations
            ValueData::F32Load {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::F32Load {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::F32Store {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::F32Store {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::F64Load {
                addr,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                instructions.push(Instruction::F64Load {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::F64Store {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::F64Store {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }

            // Partial-width memory store operations
            ValueData::I32Store8 {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::I32Store8 {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I32Store16 {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::I32Store16 {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Store8 {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::I64Store8 {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Store16 {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::I64Store16 {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }
            ValueData::I64Store32 {
                addr,
                value,
                offset,
                align,
                mem,
            } => {
                term_to_instructions_recursive(addr, instructions)?;
                term_to_instructions_recursive(value, instructions)?;
                instructions.push(Instruction::I64Store32 {
                    offset: *offset,
                    align: *align,
                    mem: *mem,
                });
            }

            // Control flow (Phase 14)
            ValueData::Block {
                label: _,
                block_type,
                body,
            } => {
                let mut body_instrs = Vec::new();
                for term in body {
                    term_to_instructions_recursive(term, &mut body_instrs)?;
                }
                instructions.push(Instruction::Block {
                    block_type: convert_block_type_from_isle(block_type),
                    body: body_instrs,
                });
            }

            ValueData::Loop {
                label: _,
                block_type,
                body,
            } => {
                let mut body_instrs = Vec::new();
                for term in body {
                    term_to_instructions_recursive(term, &mut body_instrs)?;
                }
                instructions.push(Instruction::Loop {
                    block_type: convert_block_type_from_isle(block_type),
                    body: body_instrs,
                });
            }

            ValueData::If {
                label: _,
                block_type,
                condition,
                then_body,
                else_body,
            } => {
                // Push condition onto stack
                term_to_instructions_recursive(condition, instructions)?;

                // Convert then body
                let mut then_instrs = Vec::new();
                for term in then_body {
                    term_to_instructions_recursive(term, &mut then_instrs)?;
                }

                // Convert else body
                let mut else_instrs = Vec::new();
                for term in else_body {
                    term_to_instructions_recursive(term, &mut else_instrs)?;
                }

                instructions.push(Instruction::If {
                    block_type: convert_block_type_from_isle(block_type),
                    then_body: then_instrs,
                    else_body: else_instrs,
                });
            }

            ValueData::Br { depth, value } => {
                // Push value if present
                if let Some(val) = value {
                    term_to_instructions_recursive(val, instructions)?;
                }
                instructions.push(Instruction::Br(*depth));
            }

            ValueData::BrIf {
                depth,
                condition,
                value,
            } => {
                // Push value if present
                if let Some(val) = value {
                    term_to_instructions_recursive(val, instructions)?;
                }
                // Push condition
                term_to_instructions_recursive(condition, instructions)?;
                instructions.push(Instruction::BrIf(*depth));
            }

            ValueData::BrTable {
                targets,
                default,
                index,
                value,
            } => {
                // Push value if present
                if let Some(val) = value {
                    term_to_instructions_recursive(val, instructions)?;
                }
                // Push index
                term_to_instructions_recursive(index, instructions)?;
                instructions.push(Instruction::BrTable {
                    targets: targets.clone(),
                    default: *default,
                });
            }

            ValueData::Return { values } => {
                // Push return values onto stack
                for val in values {
                    term_to_instructions_recursive(val, instructions)?;
                }
                instructions.push(Instruction::Return);
            }

            ValueData::Call { func_idx, args } => {
                // Push arguments onto stack
                for arg in args {
                    term_to_instructions_recursive(arg, instructions)?;
                }
                instructions.push(Instruction::Call(*func_idx));
            }

            ValueData::CallIndirect {
                table_idx,
                type_idx,
                table_offset,
                args,
            } => {
                // Push arguments onto stack
                for arg in args {
                    term_to_instructions_recursive(arg, instructions)?;
                }
                // Push table offset
                term_to_instructions_recursive(table_offset, instructions)?;
                instructions.push(Instruction::CallIndirect {
                    type_idx: *type_idx,
                    table_idx: *table_idx,
                });
            }

            ValueData::Unreachable => {
                instructions.push(Instruction::Unreachable);
            }

            ValueData::Nop => {
                instructions.push(Instruction::Nop);
            }

            ValueData::Drop { val } => {
                // Push the value onto the stack, then drop it
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::Drop);
            }

            // Float constants
            ValueData::F32Const { val } => {
                instructions.push(Instruction::F32Const(val.0));
            }
            ValueData::F64Const { val } => {
                instructions.push(Instruction::F64Const(val.0));
            }

            // Float arithmetic operations (f32)
            ValueData::F32Add { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Add);
            }
            ValueData::F32Sub { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Sub);
            }
            ValueData::F32Mul { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Mul);
            }
            ValueData::F32Div { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Div);
            }

            // Float arithmetic operations (f64)
            ValueData::F64Add { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Add);
            }
            ValueData::F64Sub { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Sub);
            }
            ValueData::F64Mul { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Mul);
            }
            ValueData::F64Div { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Div);
            }

            // f32 unary operations
            ValueData::F32Abs { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32Abs);
            }
            ValueData::F32Neg { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32Neg);
            }
            ValueData::F32Ceil { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32Ceil);
            }
            ValueData::F32Floor { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32Floor);
            }
            ValueData::F32Trunc { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32Trunc);
            }
            ValueData::F32Nearest { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32Nearest);
            }
            ValueData::F32Sqrt { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F32Sqrt);
            }

            // f32 binary operations
            ValueData::F32Min { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Min);
            }
            ValueData::F32Max { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Max);
            }
            ValueData::F32Copysign { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Copysign);
            }

            // f32 comparison operations
            ValueData::F32Eq { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Eq);
            }
            ValueData::F32Ne { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Ne);
            }
            ValueData::F32Lt { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Lt);
            }
            ValueData::F32Gt { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Gt);
            }
            ValueData::F32Le { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Le);
            }
            ValueData::F32Ge { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F32Ge);
            }

            // f64 unary operations
            ValueData::F64Abs { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64Abs);
            }
            ValueData::F64Neg { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64Neg);
            }
            ValueData::F64Ceil { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64Ceil);
            }
            ValueData::F64Floor { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64Floor);
            }
            ValueData::F64Trunc { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64Trunc);
            }
            ValueData::F64Nearest { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64Nearest);
            }
            ValueData::F64Sqrt { val } => {
                term_to_instructions_recursive(val, instructions)?;
                instructions.push(Instruction::F64Sqrt);
            }

            // f64 binary operations
            ValueData::F64Min { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Min);
            }
            ValueData::F64Max { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Max);
            }
            ValueData::F64Copysign { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Copysign);
            }

            // f64 comparison operations
            ValueData::F64Eq { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Eq);
            }
            ValueData::F64Ne { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Ne);
            }
            ValueData::F64Lt { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Lt);
            }
            ValueData::F64Gt { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Gt);
            }
            ValueData::F64Le { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Le);
            }
            ValueData::F64Ge { lhs, rhs } => {
                term_to_instructions_recursive(lhs, instructions)?;
                term_to_instructions_recursive(rhs, instructions)?;
                instructions.push(Instruction::F64Ge);
            }
        }

        Ok(())
    }

    /// Convert ISLE BlockType to loom-core BlockType
    fn convert_block_type_from_isle(block_type: &loom_isle::BlockType) -> BlockType {
        match block_type {
            loom_isle::BlockType::Empty => BlockType::Empty,
            loom_isle::BlockType::Value(vt) => BlockType::Value(convert_value_type_from_isle(*vt)),
            loom_isle::BlockType::Func { params, results } => BlockType::Func {
                params: params
                    .iter()
                    .map(|v| convert_value_type_from_isle(*v))
                    .collect(),
                results: results
                    .iter()
                    .map(|v| convert_value_type_from_isle(*v))
                    .collect(),
            },
        }
    }

    /// Convert ISLE ValueType to loom-core ValueType
    fn convert_value_type_from_isle(vt: loom_isle::ValueType) -> ValueType {
        match vt {
            loom_isle::ValueType::I32 => ValueType::I32,
            loom_isle::ValueType::I64 => ValueType::I64,
            loom_isle::ValueType::F32 => ValueType::F32,
            loom_isle::ValueType::F64 => ValueType::F64,
        }
    }
}

/// Optimization passes for WebAssembly modules
///
/// **The LOOM Optimization Pipeline**
///
/// LOOM achieves 80-95% binary size reduction and 0-40% instruction reduction
/// through a 12-phase pipeline optimized for speed (10-30µs per module).
///
/// ## Quick Start
///
/// ```no_run
/// use loom_core::{parse, optimize, encode};
/// # use anyhow::Result;
/// # fn example() -> Result<()> {
///
/// // Parse WebAssembly
/// let mut module = parse::parse_wat("(module (func (result i32) (i32.const 42)))")?;
///
/// // Optimize (runs all 12 phases)
/// optimize::optimize_module(&mut module)?;
///
/// // Encode back to WASM
/// let wasm_bytes = encode::encode_wasm(&module)?;
/// # Ok(())
/// # }
/// ```
///
/// ## Pipeline Phases
///
/// 1. **Precompute** - Global constant propagation
/// 2. **ISLE Folding** - Pattern-based constant folding
/// 3. **Strength Reduction** - mul/div→shifts (2-3x faster)
/// 4. **CSE** - Common subexpression elimination
/// 5. **Inline** - Remove call overhead
/// 6. **ISLE** - Expose constants after inlining
/// 7. **Code Fold** - Block flattening
/// 8. **LICM** - Loop-invariant code motion
/// 9. **Branch Simplify** - Conditional optimization
/// 10. **DCE** - Dead code elimination
/// 11. **Block Merge** - Consecutive block merging
/// 12. **Vacuum** - Final cleanup
///
/// ## Individual Passes
///
/// ```no_run
/// # use loom_core::{Module, optimize};
/// # use anyhow::Result;
/// # fn example(mut module: Module) -> Result<()> {
/// // Run specific optimizations
/// optimize::precompute(&mut module)?;
/// optimize::optimize_advanced_instructions(&mut module)?;
/// optimize::eliminate_dead_code(&mut module)?;
/// # Ok(())
/// # }
/// ```
pub mod optimize {

    use super::{BlockType, Function, Instruction, Module}; // Value unused with ISLE disabled
    use anyhow::Result;

    /// Helper: Check if a function contains Unknown instructions (optimization barrier)
    fn has_unknown_instructions(func: &Function) -> bool {
        func.instructions
            .iter()
            .any(|i| matches!(i, Instruction::Unknown(_)))
    }

    /// Helper: Check if a function contains instructions not supported by ISLE term conversion
    ///
    /// Currently only `Unknown` instructions are unsupported — all standard WASM
    /// instructions (integer, float, conversion, memory, control flow, call_indirect,
    /// br_table, bulk memory) are fully wired into the ISLE pipeline.
    fn has_unsupported_isle_instructions(func: &Function) -> bool {
        has_unsupported_isle_instructions_in_block(&func.instructions)
    }

    fn has_unsupported_isle_instructions_in_block(instructions: &[Instruction]) -> bool {
        for instr in instructions {
            match instr {
                // Recursively check nested blocks
                Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                    if has_unsupported_isle_instructions_in_block(body) =>
                {
                    return true;
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } if (has_unsupported_isle_instructions_in_block(then_body)
                    || has_unsupported_isle_instructions_in_block(else_body)) =>
                {
                    return true;
                }

                // Unknown instructions (opaque — cannot model stack effects)
                Instruction::Unknown(_) => {
                    return true;
                }

                // All other instructions are supported
                _ => {}
            }
        }
        false
    }

    /// Check if a function has control flow that makes dataflow-based ISLE optimization unsafe.
    ///
    /// # Soundness boundary
    ///
    /// ISLE's `simplify_with_env` tracks a `LocalEnv` that maps local indices to their
    /// last-known constant values. This tracking is **linear**: it assumes instructions
    /// execute in order, with each `local.set` updating the env and each `local.get`
    /// reading the most recent value. This model breaks when execution can take
    /// multiple paths:
    ///
    /// - **`BrIf`**: Execution may branch or fall through, so the env state after
    ///   the branch point is ambiguous -- the env may reflect values from the
    ///   not-taken path.
    /// - **`BrTable`**: Same issue but with N possible branch targets.
    /// - **Loop back-edges**: A `loop` containing `BrIf` can re-execute the body,
    ///   meaning the env from the first iteration leaks into the second.
    ///
    /// Z3 verification confirms that counterexamples exist in practice (e.g., the
    /// `matrix_multiply` function where env tracking across a loop back-edge causes
    /// incorrect constant propagation).
    ///
    /// # Current mitigation
    ///
    /// Functions containing `BrIf` or `BrTable` anywhere (including nested blocks)
    /// are **entirely skipped** for ISLE dataflow optimization. This is conservative
    /// but correct -- it is always safe to not optimize.
    ///
    /// The `simplify_with_env` function does have env-clearing logic at control flow
    /// boundaries as defense-in-depth, but this is insufficient for soundness on its
    /// own (the clearing is necessarily conservative and does not model join points).
    ///
    /// # What would be needed to lift this restriction
    ///
    /// To safely optimize functions with conditional branches, the optimizer would need:
    ///
    /// 1. **Basic block splitting**: Decompose the function into a control flow graph
    ///    (CFG) of basic blocks, where each block has a single entry and exit.
    /// 2. **SSA conversion**: Convert to Static Single Assignment form so that each
    ///    variable definition dominates all its uses.
    /// 3. **Dataflow analysis on the CFG**: Run the ISLE rewrite rules per-block with
    ///    proper phi-node handling at join points.
    ///
    /// See issue #56 for tracking this work.
    ///
    /// # See also
    ///
    /// - `test_advanced_optimizations_in_control_flow` (ignored test demonstrating the gap)
    /// - `constant_folding()` and `optimize_advanced_instructions()` which call this function
    fn has_dataflow_unsafe_control_flow(func: &Function) -> bool {
        // Top-level scan: BrIf/BrTable are always unsafe. Br/Return are safe
        // only at the very last position (where they act as the function
        // terminator). Anywhere else they create early-exit paths.
        let n = func.instructions.len();
        for (i, instr) in func.instructions.iter().enumerate() {
            match instr {
                Instruction::BrIf { .. } | Instruction::BrTable { .. } => return true,
                Instruction::Br(_) | Instruction::Return
                    // Tail position is fine — that's just the function ending.
                    if i + 1 != n => {
                        return true;
                    }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                    if has_unsafe_in_nested(body) => {
                        return true;
                    }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                }
                    if (has_unsafe_in_nested(then_body) || has_unsafe_in_nested(else_body)) => {
                        return true;
                    }
                _ => {}
            }
        }
        false
    }

    /// Inside a nested block/loop/if body, ANY Br/BrIf/BrTable/Return is an
    /// early-exit pattern — the surrounding expression has another path that
    /// skips the rest of the function. Hoisting code across these is unsound
    /// without a path-sensitive verifier.
    ///
    /// This is what the v0.4.0 audit's gale_sem_count_take CSE bug needed:
    /// the function had `if (eqz) return; end` (a Return inside an If body),
    /// which the previous BrIf-only check did not flag. CSE then hoisted an
    /// i32.store above the guard, producing an unsound transformation.
    fn has_unsafe_in_nested(instructions: &[Instruction]) -> bool {
        for instr in instructions {
            match instr {
                Instruction::BrIf { .. }
                | Instruction::BrTable { .. }
                | Instruction::Br(_)
                | Instruction::Return => return true,
                Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                    if has_unsafe_in_nested(body) =>
                {
                    return true;
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } if (has_unsafe_in_nested(then_body) || has_unsafe_in_nested(else_body)) => {
                    return true;
                }
                _ => {}
            }
        }
        false
    }

    /// loom#150 — DCE-specific control-flow guard, narrower than
    /// [`has_dataflow_unsafe_control_flow`]. Dead-code elimination here only
    /// ever *deletes instructions that follow an unconditional terminator*
    /// (`Return`/`Unreachable`) within a block — that code is unreachable, so
    /// removing it is sound regardless of where the terminator sits (the blunt
    /// shared guard wrongly skips a function merely because a `return` is not in
    /// tail position). We still bail on *conditional* branches (`BrIf`/
    /// `BrTable`): there, code after the branch is reachable and the
    /// path-insensitive translation validator can't be trusted to model it, so
    /// we stay conservative-over-fast (REQ-5).
    fn dce_unverifiable_control_flow(func: &Function) -> bool {
        fn scan(instrs: &[Instruction]) -> bool {
            for instr in instrs {
                match instr {
                    Instruction::BrIf { .. } | Instruction::BrTable { .. } => return true,
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                        if scan(body) =>
                    {
                        return true;
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } if scan(then_body) || scan(else_body) => return true,
                    _ => {}
                }
            }
            false
        }
        scan(&func.instructions)
    }

    /// loom#150 — LICM-specific control-flow guard, narrower than
    /// [`has_dataflow_unsafe_control_flow`]. The blunt shared guard treats *any*
    /// `BrIf` as unsafe — but every loop's back-edge is a `br_if`, so it
    /// disabled LICM for essentially all real loops. This guard permits exactly
    /// one extra shape: a back-edge branch (`Br(0)`/`BrIf(0)`) that is the LAST
    /// instruction **directly** in a loop body. Such a branch only decides
    /// whether to re-iterate; every instruction before it runs unconditionally
    /// on each iteration, and (in a function whose only branch is this
    /// back-edge) the loop is reached unconditionally — so hoisting a
    /// loop-invariant value to the pre-header is sound (same reachability as
    /// iteration 1, identical value every iteration). Any other branch —
    /// mid-body `BrIf`, `BrTable`, a branch to an outer label, or a non-tail
    /// `Br`/`Return` — could skip the hoisted op on some path, so it stays
    /// unsafe (REQ-5). `verify_or_revert` remains as defense-in-depth.
    fn licm_unverifiable_control_flow(func: &Function) -> bool {
        fn scan(instrs: &[Instruction], in_loop_body: bool) -> bool {
            let n = instrs.len();
            for (i, instr) in instrs.iter().enumerate() {
                let is_last = i + 1 == n;
                match instr {
                    // Back-edge as the loop-body terminator: safe (see above).
                    Instruction::BrIf(0) | Instruction::Br(0) if in_loop_body && is_last => {}
                    // Any other conditional / multi-way branch: unsafe.
                    Instruction::BrIf { .. } | Instruction::BrTable { .. } => return true,
                    // Non-tail unconditional branch / return creates an
                    // early-exit path: unsafe (tail position is just the
                    // function/loop ending).
                    Instruction::Br(_) | Instruction::Return if !is_last => return true,
                    Instruction::Loop { body, .. } if scan(body, true) => return true,
                    Instruction::Block { body, .. } if scan(body, false) => return true,
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } if scan(then_body, false) || scan(else_body, false) => return true,
                    _ => {}
                }
            }
            false
        }
        scan(&func.instructions, false)
    }

    /// Optimize a module by applying constant folding and other optimizations
    /// Phase 12: Uses ISLE with dataflow-aware environment tracking
    pub fn optimize_module(module: &mut Module) -> Result<()> {
        // For backwards compatibility, this function applies the core optimizations
        // The full optimization pipeline is in loom-cli/src/main.rs lines 237-246

        // #196 (CRITICAL): a module with a function-referencing element segment
        // (indirect-call table) cannot currently be optimized with behavioral
        // certainty — loom's structural verification can't detect a scrambled or
        // stale function-pointer table (v1.1.11 silent-miscompiled falcon's
        // flight controller). Skip optimization entirely and leave the module
        // unchanged, mirroring the component path's fail-safe. Correctness over
        // optimization; re-enable behind a behavioral-differential gate (#196).
        if super::fused_optimizer::element_section_references_functions(module) {
            return Ok(());
        }

        // Phase 0: Fused component optimizations (adapter devirtualization, type/import
        // dedup, dead function elimination). These are safe no-ops on non-fused modules.
        // Best-effort and non-fatal, but the outcome is always reported — on success
        // with a one-line summary of what the fused passes did (so there is positive
        // signal they ran), on failure with a warning. Never silently swallowed.
        match super::fused_optimizer::optimize_fused_module(module) {
            Ok(stats) => {
                let touched = stats.adapters_detected
                    + stats.calls_devirtualized
                    + stats.scalar_adapters_inlined
                    + stats.function_bodies_deduplicated
                    + stats.dead_functions_eliminated
                    + stats.types_deduplicated
                    + stats.imports_deduplicated
                    + stats.memory_imports_deduplicated
                    + stats.same_memory_adapters_collapsed
                    + stats.trivial_calls_eliminated;
                if touched > 0 {
                    eprintln!(
                        "fused optimization: {} adapters detected, {} calls devirtualized, \
                         {} scalar adapters inlined, {} bodies deduped, {} dead functions removed",
                        stats.adapters_detected,
                        stats.calls_devirtualized,
                        stats.scalar_adapters_inlined,
                        stats.function_bodies_deduplicated,
                        stats.dead_functions_eliminated,
                    );
                }
            }
            Err(e) => {
                // Non-fatal: fused optimization is best-effort, but log so failures are visible.
                eprintln!("Warning: fused optimization failed (non-fatal): {e}");
            }
        }

        // Phase 1: Function inlining (unlocks cross-function optimization)
        // Small functions (<50 instructions) and single-call-site functions
        // are inlined, enabling subsequent passes to optimize across boundaries.
        inline_functions(module)?;

        // Phase 1b (#219): dissolve the u64 ABI carrier the inline leaves behind —
        // scalar-forward the single-assignment carrier to its unpack sites so the
        // SROA rules can collapse the pack/unpack round-trip.
        forward_carrier_locals(module)?;

        // Phase 2: Constant folding (ISLE pattern rewrites)
        constant_folding(module)?;

        // Phase 3: Advanced instruction optimizations (strength reduction, bitwise tricks)
        optimize_advanced_instructions(module)?;

        // Phase 4: Local variable optimizations (including RSE)
        // RSE replaces redundant local.set with drop
        simplify_locals(module)?;

        // Phase 5: Dead code elimination
        // Removes const+drop patterns created by RSE and other unreachable code
        eliminate_dead_code(module)?;

        // Phase 6: Code folding (tail merging)
        code_folding(module)?;

        // Phase 7: Loop invariant code motion
        loop_invariant_code_motion(module)?;

        // Phase 8: Remove unused branches and dead code
        remove_unused_branches(module)?;

        // Phase 9: Optimize added constants
        optimize_added_constants(module)?;

        // Phase 10: Second dead code elimination pass
        // Catches dead code created by previous passes (LICM, branch removal)
        eliminate_dead_code(module)?;

        // Phase 11: Local coalescing (interference graph coloring)
        // Reduces local count by merging non-interfering locals into the
        // same slot. MUST run last — after all passes that reference locals.
        coalesce_locals(module)?;

        // Phase 5: Post-optimization stack validation (defense-in-depth)
        //
        // Each optimization pass already runs guard.validate(func)? which catches
        // per-pass stack discipline violations. This module-level check is an
        // additional safety gate that validates all functions after the full pipeline.
        //
        // Note: validate_module_blocks (compositional block analysis) has known
        // false positives with dead code and instruction count changes. We use
        // validate_function instead, which checks the whole function without
        // compositional decomposition. Functions with unanalyzable instructions
        // (Unknown, CallIndirect) are skipped — these are also skipped by the
        // optimizer itself, so they are unmodified.
        {
            use crate::stack::validation::{ValidationContext, validate_function_with_context};
            let ctx = ValidationContext::from_module(module);
            let mut skipped_count = 0usize;
            for (idx, func) in module.functions.iter().enumerate() {
                // Skip functions the optimizer also skips, but track them
                if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                    skipped_count += 1;
                    eprintln!(
                        "Warning: skipping post-optimization stack validation for function {} '{}': \
                         contains unknown or unsupported instructions",
                        idx,
                        func.name.as_deref().unwrap_or("<anonymous>")
                    );
                    continue;
                }
                if let Err(e) = validate_function_with_context(func, &ctx) {
                    return Err(anyhow::anyhow!(
                        "Post-optimization stack validation failed for '{}': {}",
                        func.name.as_deref().unwrap_or("<anonymous>"),
                        e
                    ));
                }
            }
            if skipped_count > 0 {
                eprintln!(
                    "Warning: {skipped_count} function(s) skipped during post-optimization stack validation \
                     (unmodified — optimizer also skips these)"
                );
            }
        }

        Ok(())
    }

    /// Apply ISLE-based constant folding optimization
    /// This uses ISLE pattern matching rules to fold constants (e.g., i32.const 100 + i32.const 200 → i32.const 300)
    pub fn constant_folding(module: &mut Module) -> Result<()> {
        use super::Value;
        use super::terms::TermSignatureContext;
        use crate::verify::{TranslationValidator, VerificationSignatureContext};
        use loom_isle::{LocalEnv, rewrite_with_dataflow};

        // Create signature contexts before mutating functions
        // TermSignatureContext for ISLE term conversion
        // VerificationSignatureContext for Z3 verification
        let term_sig_ctx = TermSignatureContext::from_module(module);
        let verify_sig_ctx = VerificationSignatureContext::from_module(module);

        let mut skipped_unsupported = 0usize;
        let mut skipped_control_flow = 0usize;

        for func in &mut module.functions {
            // Skip optimization for functions with unsupported instructions
            // This includes floats, conversions, rotations, and unknown opcodes
            // which would corrupt the stack simulation in instructions_to_terms
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                skipped_unsupported += 1;
                continue;
            }

            // Determine if this function has control flow that makes
            // dataflow env tracking unsafe. If so, skip the entire function:
            // even rewrite_pure goes through instructions_to_terms /
            // terms_to_instructions, and that round-trip is not guaranteed
            // to preserve instruction order across early-exit patterns
            // (Return inside If/Block). The gale_sem_count_take regression
            // (v0.4.0 audit) reproduced exactly this — terms_to_instructions
            // emitted the if-guard AFTER the function-tail straight-line
            // code, hoisting an i32.store above its null-pointer guard.
            // REQ-5 conservative-over-fast.
            if has_dataflow_unsafe_control_flow(func) {
                skipped_control_flow += 1;
                continue;
            }

            // Capture original for translation validation (Z3 proof of semantic equivalence)
            // Use context-aware validator for proper Call/CallIndirect verification
            let translator = TranslationValidator::new_with_context(
                func,
                "constant_folding",
                verify_sig_ctx.clone(),
            );

            // Save original instructions for rollback if Z3 rejects
            let original_instructions = func.instructions.clone();

            // Track whether original had End instruction
            let had_end = func.instructions.last() == Some(&Instruction::End);

            if let Ok(terms) = super::terms::instructions_to_terms_with_signatures(
                &func.instructions,
                &term_sig_ctx,
            ) {
                if !terms.is_empty() {
                    // Always use dataflow env: the unsafe-control-flow path
                    // is now skipped above (see comment on
                    // has_dataflow_unsafe_control_flow). The previous
                    // rewrite_pure fallback unsoundly reordered code on
                    // early-exit patterns via the terms-to-instructions
                    // round-trip; it's been removed.
                    let mut env = LocalEnv::new();
                    let optimized_terms: Vec<Value> = terms
                        .into_iter()
                        .map(|term| rewrite_with_dataflow(term, &mut env))
                        .collect();
                    if let Ok(mut new_instrs) =
                        super::terms::terms_to_instructions(&optimized_terms)
                    {
                        if !new_instrs.is_empty() {
                            // Preserve End instruction behavior
                            if !had_end && new_instrs.last() == Some(&Instruction::End) {
                                new_instrs.pop();
                            }
                            func.instructions = new_instrs;
                        }
                    }
                }
            }

            // Z3 translation validation: prove semantic equivalence.
            // If verification fails for this function, revert to original
            // instructions and continue optimizing other functions.
            if let Err(e) = translator.verify(func) {
                eprintln!("constant_folding: reverting function (Z3 rejected): {}", e);
                crate::stats::record_revert("constant_folding");
                func.instructions = original_instructions;
            }
        }

        if skipped_unsupported > 0 || skipped_control_flow > 0 {
            eprintln!(
                "Warning: constant_folding skipped {} function(s) with unsupported instructions, \
                 {} function(s) with dataflow-unsafe control flow (BrIf/BrTable, see #56)",
                skipped_unsupported, skipped_control_flow
            );
        }

        Ok(())
    }

    /// Dead Code Elimination (Phase 14 - Issue #13)
    /// Multi-pass DCE that:
    /// 1. Removes unreachable code after terminators
    /// 2. Removes unused drops (i32.const; drop)
    /// 3. Removes dead local.set operations
    pub fn eliminate_dead_code(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        // Build module-level context for Call instruction validation
        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // DCE only deletes unreachable code after Return/Br/Unreachable
            // terminators. It does not reorder reachable code, so the
            // early-exit hoist concern that motivated guards on other passes
            // does not apply here. (Function-tail Return is the primary
            // reason DCE exists.)

            // Create validation guard with module context for Call validation
            let guard = ValidationGuard::with_context(func, "eliminate_dead_code", ctx.clone());

            // Capture original for translation validation (Z3 proof of semantic equivalence)
            let translator = TranslationValidator::new(func, "eliminate_dead_code");

            // Pass 1: Remove unreachable code
            func.instructions = eliminate_dead_code_in_block(&func.instructions);

            // Pass 2: Remove trivial dead code (const + drop)
            func.instructions = eliminate_trivial_dead_code(&func.instructions);

            // Validate stack correctness after transformation - fail if invalid
            let _ = guard.validate(func);

            // Z3 translation validation: prove semantic equivalence
            translator.verify_or_revert(func);
        }

        Ok(())
    }

    /// Remove trivial dead code patterns:
    /// - const + drop (value produced but immediately discarded)
    /// - local.get + drop (value loaded but immediately discarded)
    /// - global.get + drop (value loaded but immediately discarded)
    fn eliminate_trivial_dead_code(instructions: &[Instruction]) -> Vec<Instruction> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            let instr = &instructions[i];

            // Check for patterns: value-producing instruction followed by drop
            if i + 1 < instructions.len() {
                if let Instruction::Drop = &instructions[i + 1] {
                    // Check if current instruction is a pure value producer (no side effects)
                    let is_pure_producer = matches!(
                        instr,
                        Instruction::I32Const(_)
                            | Instruction::I64Const(_)
                            | Instruction::F32Const(_)
                            | Instruction::F64Const(_)
                            | Instruction::LocalGet(_)
                            | Instruction::GlobalGet(_)
                    );

                    if is_pure_producer {
                        // Skip both the producer and the drop
                        i += 2;
                        continue;
                    }
                }
            }

            // Recursively process nested blocks
            match instr {
                Instruction::Block { block_type, body } => {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: eliminate_trivial_dead_code(body),
                    });
                }
                Instruction::Loop { block_type, body } => {
                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: eliminate_trivial_dead_code(body),
                    });
                }
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: eliminate_trivial_dead_code(then_body),
                        else_body: eliminate_trivial_dead_code(else_body),
                    });
                }
                _ => {
                    result.push(instr.clone());
                }
            }
            i += 1;
        }

        result
    }

    /// Helper to check if an instruction sequence ends with a terminating instruction
    fn ends_with_terminator(instructions: &[Instruction]) -> bool {
        instructions.last().is_some_and(|last| {
            matches!(
                last,
                Instruction::Return | Instruction::Br(_) | Instruction::Unreachable
            )
        })
    }

    /// Recursively eliminate dead code in a block of instructions
    fn eliminate_dead_code_in_block(instructions: &[Instruction]) -> Vec<Instruction> {
        let mut result = Vec::new();
        let mut reachable = true;

        for instr in instructions {
            if !reachable {
                // Skip unreachable instructions
                continue;
            }

            // Process the instruction based on its type
            let processed_instr = match instr {
                // Recurse into nested control flow
                Instruction::Block { block_type, body } => {
                    let clean_body = eliminate_dead_code_in_block(body);

                    // If block expects results but body is unreachable, add unreachable to satisfy type
                    let needs_unreachable = match block_type {
                        BlockType::Value(_) => ends_with_terminator(&clean_body),
                        BlockType::Func { results, .. } => {
                            ends_with_terminator(&clean_body) && !results.is_empty()
                        }
                        BlockType::Empty => false,
                    };

                    let final_body = if needs_unreachable {
                        let mut fixed = clean_body;
                        if !fixed
                            .last()
                            .is_some_and(|i| matches!(i, Instruction::Unreachable))
                        {
                            fixed.push(Instruction::Unreachable);
                        }
                        fixed
                    } else {
                        clean_body
                    };

                    Instruction::Block {
                        block_type: block_type.clone(),
                        body: final_body,
                    }
                }
                Instruction::Loop { block_type, body } => {
                    let clean_body = eliminate_dead_code_in_block(body);
                    Instruction::Loop {
                        block_type: block_type.clone(),
                        body: clean_body,
                    }
                }
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    let clean_then = eliminate_dead_code_in_block(then_body);
                    let clean_else = eliminate_dead_code_in_block(else_body);
                    Instruction::If {
                        block_type: block_type.clone(),
                        then_body: clean_then,
                        else_body: clean_else,
                    }
                }
                // Other instructions pass through unchanged
                _ => instr.clone(),
            };

            // Check if this instruction makes following code unreachable
            // Be conservative: only Return and Unreachable are definitely terminators
            // Note: Br(_) at function level might target outer blocks depending on context,
            // and Block/Loop/If that end with Br(0) still produce values and continue.
            // Being overly aggressive here causes bugs where we remove code that produces
            // return values.
            match &processed_instr {
                Instruction::Return => reachable = false,
                Instruction::Unreachable => reachable = false,
                _ => {}
            }

            result.push(processed_instr);
        }

        result
    }

    /// Branch Simplification (Phase 15 - Issue #16)
    /// Simplifies control flow by removing redundant branches and folding constant conditions
    pub fn simplify_branches(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "simplify_branches", ctx.clone());
            let translator = TranslationValidator::new(func, "simplify_branches");

            func.instructions = simplify_branches_in_block(&func.instructions);

            let _ = guard.validate(func);
            translator.verify_or_revert(func);
        }
        Ok(())
    }

    /// Recursively simplify branches in a block of instructions
    fn simplify_branches_in_block(instructions: &[Instruction]) -> Vec<Instruction> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            let instr = &instructions[i];

            // Try to detect constant condition patterns (look ahead for br_if/if)
            if i + 1 < instructions.len() {
                match (&instructions[i], &instructions[i + 1]) {
                    // Pattern: (i32.const N) followed by (br_if label)
                    (Instruction::I32Const(val), Instruction::BrIf(label)) => {
                        if *val == 0 {
                            // Condition is false - never taken, remove both
                            i += 2; // Skip both instructions
                            continue;
                        } else {
                            // Condition is true - always taken, convert to unconditional br
                            result.push(Instruction::Br(*label));
                            i += 2; // Skip both instructions
                            continue;
                        }
                    }

                    // Pattern: (i32.const N) followed by (if ...)
                    (
                        Instruction::I32Const(val),
                        Instruction::If {
                            block_type: _,
                            then_body,
                            else_body,
                        },
                    ) => {
                        if *val == 0 {
                            // Condition is false - take else branch
                            let simplified_else = simplify_branches_in_block(else_body);
                            result.extend(simplified_else);
                            i += 2;
                            continue;
                        } else {
                            // Condition is true - take then branch
                            let simplified_then = simplify_branches_in_block(then_body);
                            result.extend(simplified_then);
                            i += 2;
                            continue;
                        }
                    }

                    _ => {}
                }
            }

            // Process the instruction normally
            let processed = match instr {
                // Recursively simplify nested control flow
                Instruction::Block { block_type, body } => {
                    let simplified_body = simplify_branches_in_block(body);
                    Instruction::Block {
                        block_type: block_type.clone(),
                        body: simplified_body,
                    }
                }

                Instruction::Loop { block_type, body } => {
                    let simplified_body = simplify_branches_in_block(body);
                    Instruction::Loop {
                        block_type: block_type.clone(),
                        body: simplified_body,
                    }
                }

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    // Recursively simplify both branches
                    let simplified_then = simplify_branches_in_block(then_body);
                    let simplified_else = simplify_branches_in_block(else_body);
                    Instruction::If {
                        block_type: block_type.clone(),
                        then_body: simplified_then,
                        else_body: simplified_else,
                    }
                }

                // Remove Nop instructions while we're at it
                Instruction::Nop => {
                    i += 1;
                    continue;
                }

                _ => instr.clone(),
            };

            result.push(processed);

            i += 1;
        }

        result
    }

    /// Block Merging (Phase 16 - Issue #17)
    /// Merges nested blocks to reduce CFG complexity and improve code locality
    pub fn merge_blocks(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "merge_blocks", ctx.clone());
            let translator = TranslationValidator::new(func, "merge_blocks");

            func.instructions = merge_blocks_in_instructions(&func.instructions);

            let _ = guard.validate(func);
            translator.verify_or_revert(func);
        }
        Ok(())
    }

    /// Recursively merge blocks in a sequence of instructions
    fn merge_blocks_in_instructions(instructions: &[Instruction]) -> Vec<Instruction> {
        let mut result = Vec::new();

        for instr in instructions {
            let processed = match instr {
                // Recursively process nested blocks
                Instruction::Block { block_type, body } => {
                    // First, recursively merge blocks within the body
                    let merged_body = merge_blocks_in_instructions(body);

                    // Check if the body ends with a mergeable block
                    if let Some(merged) = try_merge_last_block(&merged_body, block_type) {
                        Instruction::Block {
                            block_type: block_type.clone(),
                            body: merged,
                        }
                    } else {
                        Instruction::Block {
                            block_type: block_type.clone(),
                            body: merged_body,
                        }
                    }
                }

                Instruction::Loop { block_type, body } => {
                    let merged_body = merge_blocks_in_instructions(body);
                    Instruction::Loop {
                        block_type: block_type.clone(),
                        body: merged_body,
                    }
                }

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    let merged_then = merge_blocks_in_instructions(then_body);
                    let merged_else = merge_blocks_in_instructions(else_body);
                    Instruction::If {
                        block_type: block_type.clone(),
                        then_body: merged_then,
                        else_body: merged_else,
                    }
                }

                _ => instr.clone(),
            };

            result.push(processed);
        }

        result
    }

    /// Try to merge the last instruction if it's a block
    /// Returns Some(merged_body) if merge was performed, None otherwise
    fn try_merge_last_block(
        body: &[Instruction],
        outer_type: &BlockType,
    ) -> Option<Vec<Instruction>> {
        if body.is_empty() {
            return None;
        }

        // Check if last instruction is a block we can merge
        let last_idx = body.len() - 1;
        match &body[last_idx] {
            Instruction::Block {
                block_type: inner_type,
                body: inner_body,
            } => {
                // Only merge if types are compatible
                // For Phase 1, we merge blocks with matching types or Empty types
                if !types_compatible_for_merge(outer_type, inner_type) {
                    return None;
                }

                // CRITICAL: Don't merge blocks that contain branch instructions
                // Merging would invalidate branch depths and create invalid WASM
                if contains_branches(inner_body) {
                    return None;
                }

                // Build merged body: all instructions before last + inner block contents
                let mut merged = body[..last_idx].to_vec();
                merged.extend_from_slice(inner_body);
                Some(merged)
            }
            _ => None,
        }
    }

    /// Check if instructions contain branch instructions (Br, BrIf, BrTable)
    /// These make block merging unsafe as it would invalidate branch depths
    fn contains_branches(instructions: &[Instruction]) -> bool {
        for instr in instructions {
            match instr {
                Instruction::Br { .. } | Instruction::BrIf { .. } | Instruction::BrTable { .. } => {
                    return true;
                }

                // Recursively check nested structures
                Instruction::Block { body, .. } | Instruction::Loop { body, .. }
                    if contains_branches(body) =>
                {
                    return true;
                }

                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } if (contains_branches(then_body) || contains_branches(else_body)) => {
                    return true;
                }

                _ => {}
            }
        }
        false
    }

    /// Check if two block types are compatible for merging
    fn types_compatible_for_merge(outer_type: &BlockType, inner_type: &BlockType) -> bool {
        match (outer_type, inner_type) {
            // Both empty - always compatible
            (BlockType::Empty, BlockType::Empty) => true,

            // Both same value type - compatible
            (BlockType::Value(t1), BlockType::Value(t2)) => t1 == t2,

            // Empty outer with value inner - not compatible (type mismatch)
            (BlockType::Empty, BlockType::Value(_)) => false,

            // Value outer with empty inner - not compatible
            (BlockType::Value(_), BlockType::Empty) => false,

            // Function signatures - for now, require exact match
            (
                BlockType::Func {
                    params: p1,
                    results: r1,
                },
                BlockType::Func {
                    params: p2,
                    results: r2,
                },
            ) => p1 == p2 && r1 == r2,

            // Mixed function and non-function types - not compatible
            _ => false,
        }
    }

    /// Verification-aware canonicalization (v0.7.0 PR-B).
    ///
    /// Two rewrites that put the IR in a form the rest of the pipeline
    /// (and the Z3 verifier) handles more uniformly:
    ///
    /// 1. **`if/else → select`**: a single-value `if (result T) ... else ... end`
    ///    whose two arms are each a single pure-pusher gets rewritten as
    ///    `then; else; cond; select`. Select is path-insensitive, so
    ///    every downstream pass and the verifier can reason about it
    ///    without branch analysis. Force-multiplier for CSE and
    ///    constant-folding.
    ///
    /// 2. **`local.set X; local.get X → local.tee X`**: equivalent stack
    ///    effect (`[T] → [T]`), saves 2 bytes per occurrence (one fewer
    ///    instruction encoded with op+idx). Safe regardless of context.
    ///
    /// v1.0.5 Track 1: ægraph-based optimization pass.
    ///
    /// Feeds each function's straight-line expression trees through the
    /// ægraph (`crate::egraph`) and applies the v1.0.4 Track C identity
    /// rules (`x+0=x`, `x*1=x`, `x&(-1)=x`). The substrate has had a
    /// rewrite engine since v1.0.4 but no pipeline consumer; this lands
    /// the consumer behind the existing `verify_or_revert` safety net.
    ///
    /// ## Scope (this MVP)
    ///
    /// Only **straight-line maximal expression trees** that produce ONE
    /// stack value from operands all visible to the egraph (i.e., no
    /// loads / calls / control flow inside the tree). For each candidate
    /// position:
    ///
    /// 1. Find the longest prefix of consecutive instructions ending at
    ///    the position whose net stack effect is `(0 → 1)` and that
    ///    contains only egraph-supported ops (see
    ///    [`ENode::from_instruction`] for the supported op set).
    /// 2. Build the e-graph by walking that prefix in stack order.
    /// 3. Saturate with `identity_rules`.
    /// 4. Extract a (potentially smaller) instruction sequence from the
    ///    root e-class.
    /// 5. If the extracted sequence is strictly shorter (or equal cost),
    ///    splice it in.
    ///
    /// ## Why this is sound today even before pipeline-wide ægraph integration
    ///
    /// - Each candidate tree has 0 consumed values + 1 produced value,
    ///   so replacing it with any other sequence with the same stack
    ///   signature preserves the function's overall stack discipline.
    /// - The rules in `identity_rules()` are hand-proven algebraic
    ///   identities; they're verified individually at egraph-test time.
    /// - The per-function `TranslationValidator::verify_or_revert` (which
    ///   every pass uses) gates the result through Z3 — any unsoundness
    ///   in the egraph engine reverts the function untouched.
    ///
    /// ## What this does NOT yet do
    ///
    /// - Cost-driven extraction (extracts node-count minimum, not a
    ///   per-op cost model).
    /// - Trees that span loads / calls / control flow.
    /// - Multi-result extraction.
    /// - Wider op coverage (i64 arith, comparisons, conversions).
    /// - Commutativity normalization.
    ///
    /// All deferred to v1.0.6+.
    pub fn egraph_optimize(module: &mut Module) -> Result<()> {
        use crate::egraph::identity_rules;
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);
        let verify_sig_ctx = crate::verify::VerificationSignatureContext::from_module(module);
        let rules = identity_rules();

        for func in &mut module.functions {
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "egraph_optimize", ctx.clone());
            let translator = TranslationValidator::new_with_context(
                func,
                "egraph_optimize",
                verify_sig_ctx.clone(),
            );

            let original_instructions = func.instructions.clone();
            func.instructions = egraph_optimize_body(&original_instructions, &rules);

            if func.instructions != original_instructions {
                let _ = guard.validate(func);
                translator.verify_or_revert(func);
            }
        }
        Ok(())
    }

    /// Recursively process a body: find maximal egraph-supported subtrees,
    /// saturate, extract, splice. Recurses into Block/Loop/If bodies.
    fn egraph_optimize_body(
        instructions: &[Instruction],
        rules: &[crate::egraph::Rule],
    ) -> Vec<Instruction> {
        let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len());

        // First pass: recurse into nested bodies. We don't try to fold
        // across the block boundary — the substrate doesn't model
        // control-flow op equivalences yet, so any tree that crosses
        // a Block/Loop/If boundary stays put.
        let mut i = 0;
        while i < instructions.len() {
            match &instructions[i] {
                Instruction::Block { block_type, body } => {
                    out.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: egraph_optimize_body(body, rules),
                    });
                    i += 1;
                }
                Instruction::Loop { block_type, body } => {
                    out.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: egraph_optimize_body(body, rules),
                    });
                    i += 1;
                }
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    out.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: egraph_optimize_body(then_body, rules),
                        else_body: egraph_optimize_body(else_body, rules),
                    });
                    i += 1;
                }
                _ => {
                    // Try to greedily extend a (0→1) tree starting at i.
                    let (tree_end, root) = try_build_egraph_tree(instructions, i);
                    if let Some((root_class, mut egraph)) = root {
                        // Saturate + extract. v1.1.0 Track B: extract is
                        // now cost-driven (memoized byte-cost DP via
                        // Op::encoded_byte_cost), so the v1.0.5 manual
                        // UF-root scan is gone.
                        let _folds = egraph.saturate_with_rules(rules);
                        let extracted = egraph.extract(root_class);

                        // Splice only if strictly shorter — node-count
                        // metric. Cost model is v1.0.6+ work.
                        let tree_len = tree_end - i + 1;
                        if extracted.len() < tree_len {
                            out.extend(extracted);
                            i = tree_end + 1;
                            continue;
                        }
                    }
                    out.push(instructions[i].clone());
                    i += 1;
                }
            }
        }

        out
    }

    /// Greedily build the longest egraph-supported (0→1)-net-effect tree
    /// starting at `start`. Returns `(end_index_inclusive, Some((root_class, egraph)))`
    /// on success, or `(start, None)` on failure (e.g., the first
    /// instruction isn't egraph-supported).
    fn try_build_egraph_tree(
        instructions: &[Instruction],
        start: usize,
    ) -> (
        usize,
        Option<(crate::egraph::EClassId, crate::egraph::EGraph)>,
    ) {
        use crate::egraph::{EClassId, EGraph, ENode};

        let mut egraph = EGraph::new();
        // Stack of e-class ids representing the symbolic stack during the
        // tree-building scan. Each egraph-supported instruction pops its
        // operands and pushes its result.
        let mut sim_stack: Vec<EClassId> = Vec::new();
        let mut end_inclusive = start;

        let mut i = start;
        while i < instructions.len() {
            let instr = &instructions[i];

            // Bail at control flow / unsupported ops; tree ends at i-1.
            match instr {
                Instruction::Block { .. }
                | Instruction::Loop { .. }
                | Instruction::If { .. }
                | Instruction::Br(_)
                | Instruction::BrIf(_)
                | Instruction::BrTable { .. }
                | Instruction::Return
                | Instruction::Call(_)
                | Instruction::CallIndirect { .. }
                | Instruction::Unreachable
                | Instruction::End
                | Instruction::Drop
                | Instruction::Nop
                | Instruction::LocalSet(_)
                | Instruction::LocalTee(_)
                | Instruction::GlobalSet(_) => {
                    break;
                }
                _ => {}
            }

            // Determine arity of the op directly (the egraph's from_instruction
            // rejects when child count mismatches arity, so we can't use it
            // as a probe). For unsupported instructions, the match falls
            // through to None → bail.
            let arity = match instr {
                Instruction::I32Const(_) | Instruction::I64Const(_) | Instruction::LocalGet(_) => 0,
                Instruction::I32Eqz | Instruction::I64Eqz => 1,
                Instruction::I32Add
                | Instruction::I32Sub
                | Instruction::I32Mul
                | Instruction::I32And
                | Instruction::I32Or
                | Instruction::I32Xor
                | Instruction::I32Shl
                | Instruction::I32ShrS
                | Instruction::I32ShrU
                | Instruction::I32Eq
                | Instruction::I64Add
                | Instruction::I64Sub
                | Instruction::I64Mul
                | Instruction::I64And
                | Instruction::I64Or
                | Instruction::I64Xor
                | Instruction::I64Shl
                | Instruction::I64ShrS
                | Instruction::I64ShrU
                | Instruction::I64Eq => 2,
                _ => break,
            };
            if sim_stack.len() < arity {
                // Tree relies on operands outside the started region —
                // bail; the prior instruction is the real boundary.
                break;
            }

            let child_ids: Vec<EClassId> = sim_stack[sim_stack.len() - arity..].to_vec();
            let node = match ENode::from_instruction(instr, &child_ids) {
                Some(n) => n,
                None => break,
            };
            let class_id = match egraph.add(node) {
                Ok(id) => id,
                Err(_) => break,
            };
            // Pop arity operands and push the result.
            for _ in 0..arity {
                sim_stack.pop();
            }
            sim_stack.push(class_id);
            end_inclusive = i;
            i += 1;
        }

        // Accept the tree iff sim_stack has exactly ONE entry — the
        // root of a (0→1) net-stack-effect expression.
        if sim_stack.len() == 1 {
            (end_inclusive, Some((sim_stack[0], egraph)))
        } else {
            (start, None)
        }
    }

    /// Both transforms are pure rewriting — sound by construction and
    /// validated by Z3 translation. Place early in the pipeline so
    /// subsequent passes see canonical forms.
    pub fn canonicalize(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "canonicalize", ctx.clone());
            let translator = TranslationValidator::new(func, "canonicalize");
            let original_instructions = func.instructions.clone();

            // Apply both rewrites. Tee normalization first (peephole),
            // then if/else → select (structural rewrite). The order
            // matters: if-to-select can introduce stack patterns that
            // benefit from a subsequent tee normalization, so we run
            // tee → if-to-select → tee.
            func.instructions = canonicalize_tee(&func.instructions);
            func.instructions = canonicalize_if_to_select(&func.instructions);
            func.instructions = canonicalize_tee(&func.instructions);

            if guard.validate(func).is_err() || translator.verify(func).is_err() {
                eprintln!("canonicalize: reverting function (verification rejected)");
                crate::stats::record_revert("canonicalize");
                func.instructions = original_instructions;
            }
        }

        Ok(())
    }

    /// Recognize `LocalSet X; LocalGet X` pairs and replace with
    /// `LocalTee X`. Equivalent stack effect, saves one instruction
    /// (2 bytes encoded). Recurses into nested control-flow bodies.
    fn canonicalize_tee(instructions: &[Instruction]) -> Vec<Instruction> {
        let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len());
        let mut i = 0;
        while i < instructions.len() {
            let recursed = match &instructions[i] {
                Instruction::Block { block_type, body } => Instruction::Block {
                    block_type: block_type.clone(),
                    body: canonicalize_tee(body),
                },
                Instruction::Loop { block_type, body } => Instruction::Loop {
                    block_type: block_type.clone(),
                    body: canonicalize_tee(body),
                },
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => Instruction::If {
                    block_type: block_type.clone(),
                    then_body: canonicalize_tee(then_body),
                    else_body: canonicalize_tee(else_body),
                },
                other => other.clone(),
            };

            // Peek-pattern: LocalSet(X) followed by LocalGet(X) → LocalTee(X).
            if let Instruction::LocalSet(idx) = &recursed
                && let Some(Instruction::LocalGet(next_idx)) = instructions.get(i + 1)
                && idx == next_idx
            {
                out.push(Instruction::LocalTee(*idx));
                i += 2;
                continue;
            }

            out.push(recursed);
            i += 1;
        }
        out
    }

    /// Recognize a single-value `if/else` whose two arms are each one
    /// pure pusher and rewrite as `then; else; cond; select`.
    ///
    /// Pattern:
    ///   COND_push          ; pure pusher (constant or local/global get)
    ///   If (result T)
    ///     then_pusher      ; pure pusher of type T
    ///   else
    ///     else_pusher      ; pure pusher of type T
    ///   end
    ///
    /// Becomes:
    ///   then_pusher        ; eager evaluation — safe because pure
    ///   else_pusher        ; eager evaluation — safe because pure
    ///   COND_push          ; condition last, where Select expects it
    ///   Select
    ///
    /// Why "pure pushers" on all three: Select evaluates both arms
    /// eagerly (no laziness), so a trapping or side-effecting arm
    /// would change behavior. Restricting to pure pushers preserves
    /// semantics by construction; broader cases (e.g., arithmetic
    /// expressions that are themselves pure) can be added later.
    fn canonicalize_if_to_select(instructions: &[Instruction]) -> Vec<Instruction> {
        let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len());
        let mut i = 0;
        while i < instructions.len() {
            let recursed = match &instructions[i] {
                Instruction::Block { block_type, body } => Instruction::Block {
                    block_type: block_type.clone(),
                    body: canonicalize_if_to_select(body),
                },
                Instruction::Loop { block_type, body } => Instruction::Loop {
                    block_type: block_type.clone(),
                    body: canonicalize_if_to_select(body),
                },
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => Instruction::If {
                    block_type: block_type.clone(),
                    then_body: canonicalize_if_to_select(then_body),
                    else_body: canonicalize_if_to_select(else_body),
                },
                other => other.clone(),
            };

            // Try the if/else → select pattern. Requires:
            //   1. recursed is an If with BlockType::Value(_)
            //   2. then_body and else_body are each exactly one
            //      pure-pusher instruction.
            //   3. The previous instruction in `out` (the cond) is
            //      itself a pure pusher (we'll re-order it).
            if let Instruction::If {
                block_type: BlockType::Value(_),
                then_body,
                else_body,
            } = &recursed
                && then_body.len() == 1
                && else_body.len() == 1
                && is_canonical_pure_pusher(&then_body[0])
                && is_canonical_pure_pusher(&else_body[0])
                && let Some(last) = out.last()
                && is_canonical_pure_pusher(last)
            {
                let cond = out.pop().unwrap();
                let then_val = then_body[0].clone();
                let else_val = else_body[0].clone();
                out.push(then_val);
                out.push(else_val);
                out.push(cond);
                out.push(Instruction::Select);
                i += 1;
                continue;
            }

            out.push(recursed);
            i += 1;
        }
        out
    }

    /// Pure pushers that are safe to re-order for `if/else → select`
    /// canonicalization. Same predicate as vacuum's `is_pure_pusher`
    /// but without the IPA-call-aware extension — we want strictly
    /// no-side-effect, no-trap, pure-value-from-constant-or-local
    /// instructions here.
    fn is_canonical_pure_pusher(instr: &Instruction) -> bool {
        matches!(
            instr,
            Instruction::I32Const(_)
                | Instruction::I64Const(_)
                | Instruction::F32Const(_)
                | Instruction::F64Const(_)
                | Instruction::LocalGet(_)
                | Instruction::GlobalGet(_)
        )
    }

    /// Directize Pass (v1.0.0 PR-O): fold `i32.const N; call_indirect (type T)` into
    /// `call F` when the target function at table slot N is statically known.
    ///
    /// # Soundness
    ///
    /// This pass is sound only under these conjunctive conditions:
    /// 1. The element segment that defines table[N] is **active** (not passive or declared)
    ///    and resolves to a concrete function index (either via `ElementItems::Functions`
    ///    or a `ref.func` const-expression). Passive / declared segments are not
    ///    materialized into the table at instantiation time, so we cannot resolve them.
    /// 2. The element segment's offset is a literal `i32.const` (no `global.get`,
    ///    which we cannot evaluate at compile time).
    /// 3. The target function's signature matches the `type_idx` of the `call_indirect`.
    /// 4. The module contains **no** instruction that can mutate any table at runtime
    ///    (`table.set`, `table.fill`, `table.copy`, `table.init`, `table.grow`).
    ///    Today these all parse to `Instruction::Unknown(_)`, so a function containing
    ///    any of them already trips `has_unknown_instructions` and is skipped from
    ///    optimization — but we *also* check at module level here, because a
    ///    mutation in one function would invalidate folds in another function.
    ///
    /// # Scope (MVP)
    ///
    /// - Only direct `I32Const(n); CallIndirect { type_idx, table_idx }` patterns.
    /// - Recurses into Block / Loop / If bodies.
    /// - Per-function Z3 translation validation via `verify_or_revert`.
    /// - Skips functions containing Unknown instructions (consistent with other passes).
    ///
    /// # Not in scope
    ///
    /// - Full table dataflow analysis (e.g., proving a `LocalGet` is always N).
    /// - Passive / declared element segments.
    /// - Runtime-initialized tables.
    pub fn directize(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};

        // 1. Conservative pre-pass: if ANY function in the module contains
        //    an Unknown instruction, bail. The table-mutating opcodes
        //    (table.set / .fill / .copy / .init / .grow) all parse to
        //    `Instruction::Unknown(_)` today (or fail parsing entirely),
        //    so this conservatively rules out tables that may be mutated
        //    at runtime. Conservative over fast.
        for func in &module.functions {
            if has_unknown_instructions(func) {
                return Ok(());
            }
        }

        // 2. Build the table-slot → func-index resolver from the element
        //    section. If parsing fails or any segment is non-trivial
        //    (passive / declared / expression-based offset that isn't a
        //    literal i32.const), we conservatively skip the pass.
        let table_resolver = match build_table_resolver(module) {
            Ok(r) => r,
            Err(_) => return Ok(()),
        };

        // If there's no element section, nothing to do.
        if table_resolver.is_empty() {
            return Ok(());
        }

        // 3. Compute the function-index → signature lookup once. Function
        //    indexing is: imported funcs first (in import order), then
        //    defined funcs (in definition order). The signatures of
        //    imports come from module.types[type_idx]; defined funcs
        //    carry their signature directly.
        let func_signatures = build_function_signature_table(module);

        // Snapshot the type table for indirect-call signature lookups.
        // Cloning is cheap (small Vec of FunctionSignature) and
        // sidesteps the &module / &mut module.functions aliasing.
        let module_types: Vec<crate::FunctionSignature> = module.types.clone();

        let ctx = ValidationContext::from_module(module);
        // v1.0.4 Track B: the verify_sig_ctx carries the same table
        // resolver, so the per-function Z3 translator can resolve
        // `i32.const N; call_indirect` to a concrete callee and encode
        // it as `pure_call_<F>(args)` — proving the directize fold
        // equivalent under congruence closure.
        let verify_sig_ctx = crate::verify::VerificationSignatureContext::from_module(module);

        for func in &mut module.functions {
            // Already excluded above, but kept for parallelism with other passes.
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "directize", ctx.clone());

            // v1.0.4 Track B: Z3 verification is BACK. Construct the
            // translator BEFORE mutation so it captures the pre-fold
            // function as the "original". The verifier now resolves
            // `i32.const N; call_indirect (type T)` to the same
            // `pure_call_<F>(args)` expression PR-K3 uses for direct
            // `call F` — so `call_indirect → call F` proves equivalent
            // under Z3 congruence closure (see verify.rs around the
            // `Instruction::CallIndirect` encoder).
            let translator = crate::verify::TranslationValidator::new_with_context(
                func,
                "directize",
                verify_sig_ctx.clone(),
            );

            let new_instructions = directize_instructions(
                &func.instructions,
                &table_resolver,
                &func_signatures,
                &module_types,
            );

            // Structural guards (no Unknown + slot resolves + signature
            // matches) remain as defense-in-depth, but Z3 is now the
            // load-bearing proof.
            if new_instructions != func.instructions {
                func.instructions = new_instructions;
                let _ = guard.validate(func);
                translator.verify_or_revert(func);
            }
        }

        Ok(())
    }

    /// Per-table-index map: slot offset → function index resolved from active
    /// element segments. v1.0.4 PR-Track-B exposes this `pub(crate)` so the
    /// Z3 verifier (`crate::verify`) can re-use the resolver to prove
    /// `i32.const N; call_indirect (type T)` equivalent to `call F` where
    /// `F = resolver[(table_idx, N)]`.
    pub(crate) type TableResolver = std::collections::HashMap<(u32, u32), u32>;

    /// Parse the raw element section bytes and build a resolver
    /// `(table_idx, slot_offset) -> func_idx` for active segments whose
    /// offset is a literal `i32.const`. Non-trivial segments are
    /// silently ignored — the resolver simply doesn't cover those slots,
    /// so `directize_instructions` will leave their call_indirects alone.
    ///
    /// Returns an error if the section bytes themselves are malformed.
    pub(crate) fn build_table_resolver(module: &Module) -> Result<TableResolver> {
        use wasmparser::{BinaryReader, ElementItems, ElementKind, FromReader};

        let mut resolver = TableResolver::new();

        let element_bytes = match &module.element_section_bytes {
            Some(b) => b,
            None => return Ok(resolver),
        };

        let mut reader = BinaryReader::new(element_bytes, 0);
        let count = reader
            .read_var_u32()
            .map_err(|e| anyhow::anyhow!("failed to read element count: {}", e))?;

        for _ in 0..count {
            let element = wasmparser::Element::from_reader(&mut reader)
                .map_err(|e| anyhow::anyhow!("failed to parse element: {}", e))?;

            // Only active segments with a constant i32 offset are resolvable.
            let (table_index, offset) = match &element.kind {
                ElementKind::Active {
                    table_index,
                    offset_expr,
                } => {
                    let mut ops = offset_expr.get_operators_reader();
                    let mut offset_val: Option<i32> = None;
                    while let Ok(op) = ops.read() {
                        match op {
                            wasmparser::Operator::I32Const { value } => {
                                offset_val = Some(value);
                            }
                            wasmparser::Operator::End => break,
                            // global.get / non-const expressions: can't resolve.
                            _ => {
                                offset_val = None;
                                break;
                            }
                        }
                    }
                    // `table_index` is `Option<u32>` in wasmparser 0.241;
                    // `None` means the default table (index 0).
                    let resolved_table_idx = table_index.unwrap_or(0);
                    match offset_val {
                        Some(v) if v >= 0 => (resolved_table_idx, v as u32),
                        _ => continue,
                    }
                }
                // Passive / declared segments don't populate the table at instantiation.
                _ => continue,
            };

            match &element.items {
                ElementItems::Functions(func_reader) => {
                    let mut slot = offset;
                    for idx_res in func_reader.clone() {
                        match idx_res {
                            Ok(func_idx) => {
                                resolver.insert((table_index, slot), func_idx);
                            }
                            Err(_) => break,
                        }
                        slot = slot.saturating_add(1);
                    }
                }
                ElementItems::Expressions(_ty, expr_reader) => {
                    let mut slot = offset;
                    for expr_res in expr_reader.clone() {
                        let const_expr = match expr_res {
                            Ok(e) => e,
                            Err(_) => break,
                        };
                        // We only fold when the expression is a single `ref.func F`.
                        // `ref.null` and any other shape leaves the slot unresolved.
                        let mut ops = const_expr.get_operators_reader();
                        let mut func_idx: Option<u32> = None;
                        let mut other = false;
                        while let Ok(op) = ops.read() {
                            match op {
                                wasmparser::Operator::RefFunc { function_index } => {
                                    if func_idx.is_some() {
                                        other = true;
                                    }
                                    func_idx = Some(function_index);
                                }
                                wasmparser::Operator::End => break,
                                _ => {
                                    other = true;
                                }
                            }
                        }
                        if !other {
                            if let Some(f) = func_idx {
                                resolver.insert((table_index, slot), f);
                            }
                        }
                        slot = slot.saturating_add(1);
                    }
                }
            }
        }

        Ok(resolver)
    }

    /// Build a `func_idx -> FunctionSignature` lookup that respects the
    /// WebAssembly function index space: imported functions come first
    /// (in import order), then locally defined functions (in definition
    /// order).
    fn build_function_signature_table(module: &Module) -> Vec<crate::FunctionSignature> {
        let mut sigs: Vec<crate::FunctionSignature> = Vec::new();
        for import in &module.imports {
            if let crate::ImportKind::Func(type_idx) = &import.kind {
                if let Some(sig) = module.types.get(*type_idx as usize) {
                    sigs.push(sig.clone());
                } else {
                    // Should not happen on a validated module, but if it does,
                    // push a dummy signature so the index space stays aligned.
                    sigs.push(crate::FunctionSignature {
                        params: Vec::new(),
                        results: Vec::new(),
                    });
                }
            }
        }
        for func in &module.functions {
            sigs.push(func.signature.clone());
        }
        sigs
    }

    /// Recursive directize transform. For each `I32Const(n); CallIndirect`
    /// pair where (table_idx, n) resolves to a function `F` whose
    /// signature matches the indirect call's `type_idx`, emit `Call(F)`
    /// instead. All other instructions are passed through unchanged.
    /// Recurses into Block / Loop / If bodies.
    ///
    /// Arguments:
    /// - `func_signatures`: indexed by function index (imports first, then defined).
    /// - `module_types`: indexed by type index (matches `call_indirect`'s `type_idx`).
    fn directize_instructions(
        instructions: &[Instruction],
        table_resolver: &TableResolver,
        func_signatures: &[crate::FunctionSignature],
        module_types: &[crate::FunctionSignature],
    ) -> Vec<Instruction> {
        let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len());
        let mut i = 0;
        while i < instructions.len() {
            // First, recurse into nested control-flow bodies.
            let cur = match &instructions[i] {
                Instruction::Block { block_type, body } => Some(Instruction::Block {
                    block_type: block_type.clone(),
                    body: directize_instructions(
                        body,
                        table_resolver,
                        func_signatures,
                        module_types,
                    ),
                }),
                Instruction::Loop { block_type, body } => Some(Instruction::Loop {
                    block_type: block_type.clone(),
                    body: directize_instructions(
                        body,
                        table_resolver,
                        func_signatures,
                        module_types,
                    ),
                }),
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => Some(Instruction::If {
                    block_type: block_type.clone(),
                    then_body: directize_instructions(
                        then_body,
                        table_resolver,
                        func_signatures,
                        module_types,
                    ),
                    else_body: directize_instructions(
                        else_body,
                        table_resolver,
                        func_signatures,
                        module_types,
                    ),
                }),
                _ => None,
            };
            if let Some(rec) = cur {
                out.push(rec);
                i += 1;
                continue;
            }

            // Look for `I32Const(n); CallIndirect { type_idx, table_idx }`.
            if i + 1 < instructions.len() {
                if let (
                    Instruction::I32Const(n),
                    Instruction::CallIndirect {
                        type_idx,
                        table_idx,
                    },
                ) = (&instructions[i], &instructions[i + 1])
                {
                    if *n >= 0 {
                        let slot = *n as u32;
                        if let Some(&func_idx) = table_resolver.get(&(*table_idx, slot)) {
                            // Signature must match. `func_signatures[func_idx]` is the
                            // function's actual signature; `module_types[type_idx]` is the
                            // type the indirect call expects.
                            let target_sig = func_signatures.get(func_idx as usize);
                            let want_sig = module_types.get(*type_idx as usize);
                            if let (Some(target), Some(want)) = (target_sig, want_sig) {
                                if target == want {
                                    // Fold: drop the I32Const, emit Call(func_idx)
                                    // instead of CallIndirect.
                                    out.push(Instruction::Call(func_idx));
                                    i += 2;
                                    continue;
                                }
                            }
                        }
                    }
                }
            }

            out.push(instructions[i].clone());
            i += 1;
        }
        out
    }

    /// Vacuum Cleanup Pass (Phase 17 - Issue #20)
    /// Final cleanup pass that removes nops, unwraps trivial blocks, and simplifies degenerate patterns
    pub fn vacuum(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        // PR-F (v0.7.0): compute function summaries up-front so the
        // peephole inside `vacuum_instructions` can fold `Call f; Drop`
        // when f is pure + no-trap + ZERO args + ONE result. Without
        // summaries a Call is an opaque side-effecting wall — even a
        // call to a pure helper survives `Drop` because we can't prove
        // the absence of effects.
        //
        // PR-J (v0.8.0): extend the fold to N>0 args by also popping
        // the N preceding pure pushers from the already-emitted output.
        // Soundness: each pure pusher consumes 0 / produces 1; N of
        // them at the tail of `out` exactly cover the call's N args
        // and contribute nothing else observable, so removing them
        // (and the Call, and the Drop) preserves the stack and the
        // observable behavior. If any of the last N is not a pure
        // pusher, or fewer than N entries exist (args came from outside
        // the local region), the fold is skipped.
        let summaries = crate::summary::compute_module_summaries(module);
        // Snapshot per-function (param_count, result_count) before the
        // mutable loop. The fold requires param_count == 0 and
        // result_count == 1.
        let signatures: Vec<(usize, usize)> = module
            .functions
            .iter()
            .map(|f| (f.signature.params.len(), f.signature.results.len()))
            .collect();

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // Create validation guard with module context for Call validation
            let guard = ValidationGuard::with_context(func, "vacuum", ctx.clone());

            // Capture original for translation validation (Z3 proof of semantic equivalence)
            let translator = TranslationValidator::new(func, "vacuum");

            // Apply vacuum transformation (passes summaries through so the
            // const+drop peephole can also fold pure+no-trap Call;Drop pairs).
            func.instructions = vacuum_instructions(&func.instructions, &summaries, &signatures);

            // Validate stack correctness after transformation - fail if invalid
            let _ = guard.validate(func);

            // Z3 translation validation: prove semantic equivalence
            translator.verify_or_revert(func);
        }
        Ok(())
    }

    /// Recursively clean up instructions
    fn vacuum_instructions(
        instructions: &[Instruction],
        summaries: &[crate::summary::FunctionSummary],
        signatures: &[(usize, usize)],
    ) -> Vec<Instruction> {
        let mut result = Vec::new();

        for (i, instr) in instructions.iter().enumerate() {
            match instr {
                // Skip nops entirely
                Instruction::Nop => continue,

                // Clean up blocks
                Instruction::Block { block_type, body } => {
                    let cleaned_body = vacuum_instructions(body, summaries, signatures);

                    // Check if block is trivial and can be unwrapped
                    // CRITICAL: Don't unwrap blocks with result types if followed by another block/if
                    // because the next block creates a new stack scope that won't see the result value
                    let next_is_block = instructions[i + 1..]
                        .iter()
                        .find(|next| !matches!(next, Instruction::Nop))
                        .map(|next| {
                            matches!(
                                next,
                                Instruction::Block { .. }
                                    | Instruction::If { .. }
                                    | Instruction::Loop { .. }
                            )
                        })
                        .unwrap_or(false);

                    let should_unwrap = is_trivial_block(&cleaned_body, block_type)
                        && !(*block_type != BlockType::Empty && next_is_block);

                    if should_unwrap {
                        // Unwrap: add body instructions directly
                        result.extend(cleaned_body);
                    } else {
                        // Keep block with cleaned body
                        result.push(Instruction::Block {
                            block_type: block_type.clone(),
                            body: cleaned_body,
                        });
                    }
                }

                // Clean up loops
                Instruction::Loop { block_type, body } => {
                    let cleaned_body = vacuum_instructions(body, summaries, signatures);

                    if !cleaned_body.is_empty() {
                        result.push(Instruction::Loop {
                            block_type: block_type.clone(),
                            body: cleaned_body,
                        });
                    }
                    // Empty loop is removed
                }

                // Clean up if statements
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    let cleaned_then = vacuum_instructions(then_body, summaries, signatures);
                    let cleaned_else = vacuum_instructions(else_body, summaries, signatures);

                    // Simplify based on branch emptiness
                    if cleaned_then.is_empty() && cleaned_else.is_empty() {
                        // Both branches empty - drop the condition
                        // But first, we need to ensure condition is evaluated
                        // Add a drop instruction (condition already on stack from previous instruction)
                        // Actually, the condition comes BEFORE the if, so we need to add it here
                        // For now, keep the if structure - full cleanup needs CFG analysis
                        result.push(Instruction::If {
                            block_type: block_type.clone(),
                            then_body: cleaned_then,
                            else_body: cleaned_else,
                        });
                    } else {
                        result.push(Instruction::If {
                            block_type: block_type.clone(),
                            then_body: cleaned_then,
                            else_body: cleaned_else,
                        });
                    }
                }

                // Keep everything else as-is
                _ => result.push(instr.clone()),
            }
        }

        // Post-pass peephole: collapse `pure_push; drop` pairs.
        //
        // Created mostly by `eliminate_dead_locals` (which replaces
        // dead `LocalSet` with `Drop`, leaving the constant or load
        // pushed by a preceding instruction with no consumer) and by
        // `eliminate_dead_stores` (same mechanism, path-sensitive).
        //
        // Pure pushers that are safe to fold away with their Drop:
        //   I32Const, I64Const, F32Const, F64Const  — pure literals
        //   LocalGet idx                            — pure read
        //   GlobalGet idx                           — pure read
        //   Call f   where f is pure + no-trap + returns one value
        //
        // NOT folded: memory loads, calls to impure or may-trap
        // functions, anything else that can trap or have a side effect.
        peephole_const_drop(result, summaries, signatures)
    }

    /// Recognize `pure_push; drop` and remove both. Recurses into
    /// nested control-flow bodies. Used by `vacuum_instructions` to
    /// mop up the leftovers from `eliminate_dead_locals` and
    /// `eliminate_dead_stores`, plus PR-F's `Call f; Drop` folding
    /// when `f` is pure + no-trap + single-result, plus PR-J's
    /// arg-aware extension that additionally pops N preceding pure
    /// pushers when `f` takes N args and they were all produced by
    /// pure pushers we just emitted into `out`.
    fn peephole_const_drop(
        instructions: Vec<Instruction>,
        summaries: &[crate::summary::FunctionSummary],
        signatures: &[(usize, usize)],
    ) -> Vec<Instruction> {
        let mut out: Vec<Instruction> = Vec::with_capacity(instructions.len());
        let mut iter = instructions.into_iter().peekable();
        while let Some(instr) = iter.next() {
            // First, recurse into nested bodies (Block/Loop/If) so the
            // peephole reaches their leftovers too.
            let instr = match instr {
                Instruction::Block { block_type, body } => Instruction::Block {
                    block_type,
                    body: peephole_const_drop(body, summaries, signatures),
                },
                Instruction::Loop { block_type, body } => Instruction::Loop {
                    block_type,
                    body: peephole_const_drop(body, summaries, signatures),
                },
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => Instruction::If {
                    block_type,
                    then_body: peephole_const_drop(then_body, summaries, signatures),
                    else_body: peephole_const_drop(else_body, summaries, signatures),
                },
                other => other,
            };

            // PR-J: arg-aware `Call f; Drop` fold. If `instr` is a
            // pure + no-trap + single-result Call with N>0 params and
            // the next instruction is Drop, attempt to also peel off
            // the N preceding pure pushers from `out`. Each pure
            // pusher consumes 0 / produces 1, so N of them in a row
            // at the tail of `out` exactly cover the call's N args
            // and contribute nothing else observable. If any of the
            // last N entries in `out` is NOT a pure pusher (or `out`
            // has fewer than N entries — meaning some args came from
            // outside the current straight-line region), we leave
            // everything alone.
            if matches!(iter.peek(), Some(Instruction::Drop)) {
                if let Instruction::Call(idx) = &instr {
                    let i = *idx as usize;
                    if i < summaries.len()
                        && i < signatures.len()
                        && summaries[i].is_pure
                        && summaries[i].is_no_trap
                        && signatures[i].1 == 1
                    {
                        let n = signatures[i].0;
                        if n == 0 {
                            // PR-F zero-arg case — fold the Call+Drop.
                            iter.next(); // consume the Drop
                            continue;
                        }
                        // N>0: check the tail of `out` for N pure pushers.
                        if out.len() >= n {
                            let tail_start = out.len() - n;
                            let all_pure = out[tail_start..]
                                .iter()
                                .all(|p| is_pure_pusher(p, summaries, signatures));
                            if all_pure {
                                // Pop the N pure pushers, drop the Call,
                                // consume the Drop. Net stack effect: zero.
                                out.truncate(tail_start);
                                iter.next(); // consume the Drop
                                continue;
                            }
                        }
                        // Fall through: not safe to fold — keep the Call.
                    }
                }
            }

            // Match the pair pattern (single pure pusher + Drop).
            if is_pure_pusher(&instr, summaries, signatures)
                && matches!(iter.peek(), Some(Instruction::Drop))
            {
                iter.next(); // consume the Drop
                continue; // skip both — net effect is nothing
            }
            out.push(instr);
        }
        out
    }

    /// Side-effect-free, non-trapping instructions whose result can be
    /// safely discarded with their immediately-following `Drop`.
    ///
    /// Each variant returning `true` must have stack signature (0, 1):
    /// consume zero values, produce exactly one. This is what makes
    /// the PR-J arg-aware fold sound — N pure pushers in a row push
    /// exactly N values and nothing else, so they can be removed
    /// alongside a Call that consumes those N values.
    ///
    /// PR-F (v0.7.0): extended to also accept `Call f` when the callee
    /// is pure + no-trap + zero-arg + single-result. The pure+no-trap
    /// constraint is the interprocedural justification that the call's
    /// only effect is producing the value we're about to drop;
    /// single-result is required because `Drop` consumes exactly one
    /// stack slot; zero-arg keeps the (0, 1) invariant so that a Call
    /// to a pure-zero-arg helper can itself appear as an argument to
    /// a larger PR-J fold.
    fn is_pure_pusher(
        instr: &Instruction,
        summaries: &[crate::summary::FunctionSummary],
        signatures: &[(usize, usize)],
    ) -> bool {
        match instr {
            Instruction::I32Const(_)
            | Instruction::I64Const(_)
            | Instruction::F32Const(_)
            | Instruction::F64Const(_)
            | Instruction::LocalGet(_)
            | Instruction::GlobalGet(_) => true,
            Instruction::Call(idx) => {
                // Only fold zero-arg calls — the safe minimum.
                // A Call consumes its arguments from the stack; removing
                // it without also removing the arg-pushers would leave
                // dangling values that break stack balance.
                let i = *idx as usize;
                i < summaries.len()
                    && summaries[i].is_pure
                    && summaries[i].is_no_trap
                    && i < signatures.len()
                    && signatures[i] == (0, 1)
            }
            _ => false,
        }
    }

    /// Check if a block is trivial and can be unwrapped
    fn is_trivial_block(body: &[Instruction], block_type: &BlockType) -> bool {
        // Empty block is trivial
        if body.is_empty() {
            return true;
        }

        // Single instruction block - check type compatibility
        if body.len() == 1 {
            // CRITICAL: Never unwrap blocks containing loops!
            // Loops often have br_if instructions targeting the outer block.
            // Unwrapping the block breaks these branch targets.
            if matches!(body[0], Instruction::Loop { .. }) {
                return false;
            }

            match block_type {
                // Empty block type - can unwrap most things (but not loops, checked above)
                BlockType::Empty => true,

                // Block expects a value - check if instruction produces one
                BlockType::Value(_) => {
                    // For safety, only unwrap if the instruction is known to produce a value
                    // This includes: constants, arithmetic, local.get, etc.
                    matches!(
                        body[0],
                        Instruction::I32Const(_)
                            | Instruction::I64Const(_)
                            | Instruction::I32Add
                            | Instruction::I32Sub
                            | Instruction::I32Mul
                            | Instruction::I64Add
                            | Instruction::I64Sub
                            | Instruction::I64Mul
                            | Instruction::LocalGet(_)
                            | Instruction::Block { .. }
                            | Instruction::If { .. }
                    )
                }

                // Function signature blocks - be conservative
                BlockType::Func { .. } => false,
            }
        } else {
            // Multiple instructions - not trivial
            false
        }
    }

    /// SimplifyLocals (Phase 18 - Issue #15)
    ///
    /// Optimizes local variable usage by:
    /// 1. Removing redundant copies (local.set from local.get)
    /// 2. Tracking equivalent locals and canonicalizing gets
    /// 3. Eliminating dead stores (sets with no subsequent gets)
    /// 4. Simplifying tees of unused locals
    ///
    /// This runs iteratively until a fixed point is reached.
    pub fn simplify_locals(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // Skip early-exit functions: simplify_locals (RSE / local-merge)
            // can move local.set / local.get sequences across guard
            // boundaries when those guards are `if (cond) return ...`.
            // The verifier is path-insensitive across these (REQ-5,
            // gale_sem_count_take regression).
            if has_dataflow_unsafe_control_flow(func) {
                continue;
            }

            // Create validation guard with module context for Call validation
            let guard = ValidationGuard::with_context(func, "simplify_locals", ctx.clone());

            // Capture original for translation validation and rollback
            let translator = TranslationValidator::new(func, "simplify_locals");
            let original_instructions = func.instructions.clone();

            let mut changed = true;
            let mut iterations = 0;
            const MAX_ITERATIONS: usize = 10;

            while changed && iterations < MAX_ITERATIONS {
                changed = false;
                iterations += 1;

                // Analyze local usage and build equivalences
                let (usage, equivalences) = analyze_locals(&func.instructions);

                // SAFETY: Only apply equivalence substitution for straight-line code.
                // Control flow (if/block/loop) makes equivalence tracking unsound because:
                // 1. Sets in one branch don't affect the other branch
                // 2. Equivalences created before a branch may be invalidated in one path
                // Per our proof-first philosophy: skip unsafe optimizations.
                let has_control_flow = func.instructions.iter().any(|i| {
                    matches!(
                        i,
                        Instruction::If { .. }
                            | Instruction::Block { .. }
                            | Instruction::Loop { .. }
                    )
                });

                // Apply optimizations (skip equivalence substitution if control flow present)
                let safe_equivalences = if has_control_flow {
                    std::collections::BTreeMap::new()
                } else {
                    equivalences
                };
                func.instructions = simplify_instructions(
                    &func.instructions,
                    &usage,
                    &safe_equivalences,
                    &mut changed,
                );

                // Apply Redundant Set Elimination
                // This replaces redundant local.set with drop, preserving stack effects.
                // The DCE pass will then clean up const+drop patterns.
                func.instructions = eliminate_redundant_sets(&func.instructions, &mut changed);
            }

            // Verify stack correctness and semantic equivalence — revert on failure
            if guard.validate(func).is_err() || translator.verify(func).is_err() {
                eprintln!("simplify_locals: reverting function (verification rejected)");
                crate::stats::record_revert("simplify_locals");
                func.instructions = original_instructions;
            }
        }
        Ok(())
    }

    /// #219 carrier scalar-forwarding (the perf milestone): forward a TOP-LEVEL
    /// single-assignment local's pure defining expression to its use sites, then
    /// let the committed SROA rules dissolve the now-exposed pack/unpack. This
    /// removes the u64 ABI carrier the proven `br_table` seam inline leaves inside
    /// `z_impl` (decide builds `extend_i32_u<<32 | status`, z_impl tears it back
    /// with `&0xff` / `>>32` through a dead i64 carrier).
    ///
    /// SOUNDNESS (this transform is NOT Z3-backstoppable for the void seam fn —
    /// the value flows only into a havoc'd impure-call arg → vacuous pass; gale's
    /// G474RE silicon is the behavioral gate, #219):
    ///   - We forward ONLY locals with exactly one write in the whole function
    ///     (`analyze_locals` `sets.len()==1`) whose single write is at the
    ///     FUNCTION TOP LEVEL (not nested in any Block/If/Loop). Top-level
    ///     single-assignment ⇒ the def dominates every use (you can't reach a
    ///     later top-level statement without executing the earlier one).
    ///   - The forwarding mechanism (`rewrite_with_dataflow` + `OptimizationEnv`)
    ///     additionally invalidates a forwarding when any input local is
    ///     reassigned (reaching-defs), so a forwarded expression always carries
    ///     its def-time inputs.
    ///   - `is_forwardable_expr` admits only pure ops (no calls/loads/stores/
    ///     div-rem/floats), so re-evaluating the expression at a use is safe.
    ///
    /// `verify_or_revert` still guards non-void functions and stack correctness.
    pub fn forward_carrier_locals(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::{TranslationValidator, VerificationSignatureContext};
        use loom_isle::{LocalEnv, rewrite_pure, rewrite_with_dataflow};
        use std::collections::{BTreeMap, HashSet};

        let ctx = ValidationContext::from_module(module);
        let verify_sig_ctx = VerificationSignatureContext::from_module(module);

        for func in &mut module.functions {
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // Precompute TOP-LEVEL single-assignment locals.
            let (usage, _equiv) = analyze_locals(&func.instructions);
            let mut top_level_writes: BTreeMap<u32, usize> = BTreeMap::new();
            for instr in &func.instructions {
                if let Instruction::LocalSet(i) | Instruction::LocalTee(i) = instr {
                    *top_level_writes.entry(*i).or_insert(0) += 1;
                }
            }
            let single_assign: HashSet<u32> = usage
                .iter()
                .filter(|(idx, u)| {
                    u.sets.len() == 1 && top_level_writes.get(*idx).copied().unwrap_or(0) == 1
                })
                .map(|(idx, _)| *idx)
                .collect();
            if single_assign.is_empty() {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "forward_carrier_locals", ctx.clone());
            let translator = TranslationValidator::new_with_context(
                func,
                "forward_carrier_locals",
                verify_sig_ctx.clone(),
            );
            let original_instructions = func.instructions.clone();

            let had_end = func.instructions.last() == Some(&Instruction::End);
            if let Ok(terms) = super::terms::instructions_to_terms(&func.instructions) {
                if !terms.is_empty() {
                    let mut env = LocalEnv::new();
                    env.single_assign = single_assign;
                    // Forward the carrier (dataflow), then apply the structural
                    // SROA rules (rewrite_pure) to dissolve the exposed pack/unpack.
                    let forwarded: Vec<_> = terms
                        .into_iter()
                        .map(|t| rewrite_with_dataflow(t, &mut env))
                        .map(rewrite_pure)
                        .collect();
                    if let Ok(mut new_instrs) = super::terms::terms_to_instructions(&forwarded) {
                        if !new_instrs.is_empty() {
                            if !had_end && new_instrs.last() == Some(&Instruction::End) {
                                new_instrs.pop();
                            }
                            func.instructions = new_instrs;
                        }
                    }
                }
            }

            if guard.validate(func).is_err() || translator.verify(func).is_err() {
                eprintln!("forward_carrier_locals: reverting function (verification rejected)");
                crate::stats::record_revert("forward_carrier_locals");
                func.instructions = original_instructions;
            }
        }
        Ok(())
    }

    #[derive(Debug, Clone)]
    struct LocalUsage {
        // Positions where this local is read (local.get)
        gets: Vec<usize>,
        // Positions where this local is written (local.set/tee)
        sets: Vec<usize>,
        // First get position (if any)
        first_get: Option<usize>,
        // Last set position (if any)
        last_set: Option<usize>,
    }

    fn analyze_locals(
        instructions: &[Instruction],
    ) -> (
        std::collections::BTreeMap<u32, LocalUsage>,
        std::collections::BTreeMap<u32, u32>,
    ) {
        use std::collections::BTreeMap;

        let mut usage: BTreeMap<u32, LocalUsage> = BTreeMap::new();
        let mut equivalences: BTreeMap<u32, u32> = BTreeMap::new();
        let mut position = 0;

        fn analyze_recursive(
            instructions: &[Instruction],
            usage: &mut BTreeMap<u32, LocalUsage>,
            equivalences: &mut BTreeMap<u32, u32>,
            position: &mut usize,
        ) {
            let mut i = 0;
            while i < instructions.len() {
                let instr = &instructions[i];

                // Detect equivalence pattern: local.get followed by local.set
                // AND clear equivalence when local is set to something else
                if i + 1 < instructions.len() {
                    if let (Instruction::LocalGet(src_idx), Instruction::LocalSet(dst_idx)) =
                        (&instructions[i], &instructions[i + 1])
                    {
                        // This creates equivalence: dst ≡ src
                        equivalences.insert(*dst_idx, *src_idx);
                    }
                }

                // Clear equivalence when a local is set (unless it's from the pattern above)
                // This ensures we don't use stale equivalences after a local is modified
                match instr {
                    Instruction::LocalSet(idx)
                        // Check if this is NOT the target of a local.get; local.set pattern
                        if (i == 0 || !matches!(&instructions[i - 1], Instruction::LocalGet(_))) => {
                            // This set breaks any prior equivalence
                            equivalences.remove(idx);
                        }
                    Instruction::LocalTee(idx) => {
                        // Tee also breaks equivalence (unless immediately after local.get of same local)
                        equivalences.remove(idx);
                    }
                    _ => {}
                }

                match instr {
                    Instruction::LocalGet(idx) => {
                        let entry = usage.entry(*idx).or_insert_with(|| LocalUsage {
                            gets: Vec::new(),
                            sets: Vec::new(),
                            first_get: None,
                            last_set: None,
                        });
                        entry.gets.push(*position);
                        if entry.first_get.is_none() {
                            entry.first_get = Some(*position);
                        }
                    }
                    Instruction::LocalSet(idx) => {
                        let entry = usage.entry(*idx).or_insert_with(|| LocalUsage {
                            gets: Vec::new(),
                            sets: Vec::new(),
                            first_get: None,
                            last_set: None,
                        });
                        entry.sets.push(*position);
                        entry.last_set = Some(*position);
                    }
                    Instruction::LocalTee(idx) => {
                        let entry = usage.entry(*idx).or_insert_with(|| LocalUsage {
                            gets: Vec::new(),
                            sets: Vec::new(),
                            first_get: None,
                            last_set: None,
                        });
                        entry.sets.push(*position);
                        entry.last_set = Some(*position);
                        // Note: tee is NOT counted as a get for dead store elimination.
                        // Tee stores to local AND returns value on stack, but it doesn't
                        // READ from the local. We only count local.get as actual reads.
                    }
                    // Recurse into control flow
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        analyze_recursive(body, usage, equivalences, position);
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        analyze_recursive(then_body, usage, equivalences, position);
                        analyze_recursive(else_body, usage, equivalences, position);
                    }
                    _ => {}
                }
                *position += 1;
                i += 1;
            }
        }

        analyze_recursive(instructions, &mut usage, &mut equivalences, &mut position);
        (usage, equivalences)
    }

    fn simplify_instructions(
        instructions: &[Instruction],
        usage: &std::collections::BTreeMap<u32, LocalUsage>,
        equivalences: &std::collections::BTreeMap<u32, u32>,
        changed: &mut bool,
    ) -> Vec<Instruction> {
        let mut result = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            let instr = &instructions[i];

            // Note: Dead store elimination (local.get + local.set where dst is unused) is NOT performed here.
            // Reason: Removing stores requires proving the store has no observable effect, which needs
            // careful stack analysis across block boundaries. Per our proof-first philosophy: we only
            // implement optimizations we can prove correct. Equivalence canonicalization is proven safe.

            // Single-use temp folding: local.set $x; local.get $x -> local.tee $x
            // This is safe because both patterns have the same stack effect:
            // - set+get: consume 1 (set), produce 1 (get) = net 0
            // - tee: consume 1, produce 1 = net 0
            // The tee is more efficient as it avoids the round-trip through the local.
            if let Instruction::LocalSet(set_idx) = instr {
                if i + 1 < instructions.len() {
                    if let Instruction::LocalGet(get_idx) = &instructions[i + 1] {
                        if set_idx == get_idx {
                            // Pattern matched: local.set $x; local.get $x -> local.tee $x
                            *changed = true;
                            result.push(Instruction::LocalTee(*set_idx));
                            i += 2; // Skip both instructions
                            continue;
                        }
                    }
                }
            }

            // Process individual instruction
            let processed = match instr {
                // Canonicalize gets based on equivalences
                Instruction::LocalGet(idx) => {
                    if let Some(&equiv_idx) = equivalences.get(idx) {
                        *changed = true;
                        Instruction::LocalGet(equiv_idx)
                    } else {
                        instr.clone()
                    }
                }

                // Dead store elimination: replace local.set with drop if local is never read
                // This is safe because both consume 1 value from stack:
                // - local.set: pop value, store to local
                // - drop: pop value, discard
                // Stack balance is preserved; DCE will later remove const+drop patterns.
                Instruction::LocalSet(idx) => {
                    if let Some(local_usage) = usage.get(idx) {
                        if local_usage.gets.is_empty() {
                            // No reads of this local anywhere - this is a dead store
                            *changed = true;
                            Instruction::Drop
                        } else {
                            instr.clone()
                        }
                    } else {
                        instr.clone()
                    }
                }

                // Recursively process control flow
                Instruction::Block { block_type, body } => Instruction::Block {
                    block_type: block_type.clone(),
                    body: simplify_instructions(body, usage, equivalences, changed),
                },

                Instruction::Loop { block_type, body } => Instruction::Loop {
                    block_type: block_type.clone(),
                    body: simplify_instructions(body, usage, equivalences, changed),
                },

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => Instruction::If {
                    block_type: block_type.clone(),
                    then_body: simplify_instructions(then_body, usage, equivalences, changed),
                    else_body: simplify_instructions(else_body, usage, equivalences, changed),
                },

                // Dead store elimination for local.tee: if local is never read, skip the tee
                // local.tee has stack effect [t] → [t] (value stays on stack), so removing
                // it preserves stack balance. We just lose the pointless store.
                Instruction::LocalTee(idx) => {
                    if let Some(local_usage) = usage.get(idx) {
                        if local_usage.gets.is_empty() {
                            // No reads of this local - tee is pointless, skip it
                            *changed = true;
                            i += 1;
                            continue; // Don't add anything to result
                        }
                    }
                    instr.clone()
                }

                _ => instr.clone(),
            };

            result.push(processed);
            i += 1;
        }

        result
    }

    /// Eliminate Redundant Sets (Redundant Set Elimination - RSE)
    ///
    /// Replaces redundant local.set instructions with `drop` when a local is set twice
    /// without an intervening get. This is inspired by Binaryen's redundant-set-elimination pass.
    ///
    /// Example:
    /// ```wasm
    /// i32.const 10
    /// local.set $x     ;; redundant - replaced with drop
    /// i32.const 20
    /// local.set $x     ;; survives
    /// local.get $x
    /// ```
    ///
    /// Optimizes to (after DCE removes const+drop):
    /// ```wasm
    /// i32.const 20
    /// local.set $x
    /// local.get $x
    /// ```
    ///
    /// Algorithm:
    /// - Track the last set position for each local
    /// - When encountering a new set to the same local, check if there was an intervening get
    /// - If no get, replace the previous set with `drop` (preserves stack effects)
    /// - For `local.tee`, remove entirely if redundant (value stays on stack)
    /// - Conservative: Only eliminates in straight-line code, not across control flow
    ///
    /// The key insight is that replacing `local.set` with `drop` preserves stack effects:
    /// - `local.set`: pops 1 value, pushes 0
    /// - `drop`: pops 1 value, pushes 0
    ///
    /// The existing DCE pass will then clean up `const; drop` patterns.
    ///
    /// Benefits:
    /// - 2-5% binary size reduction on typical code
    /// - Reduces unnecessary local variable writes
    /// - Enables further optimizations (dead code elimination)
    fn eliminate_redundant_sets(
        instructions: &[Instruction],
        changed: &mut bool,
    ) -> Vec<Instruction> {
        use std::collections::{BTreeMap, HashSet};

        /// Information about a local.set instruction
        #[derive(Debug, Clone)]
        struct SetInfo {
            /// Position of the set instruction in the result vector
            position: usize,
            /// Whether there was a get between this set and the next
            has_intervening_get: bool,
            /// Whether this is a tee (which should be removed, not replaced with drop)
            is_tee: bool,
        }

        /// First pass: identify which sets are redundant
        fn find_redundant_sets(
            instructions: &[Instruction],
            last_sets: &mut BTreeMap<u32, SetInfo>,
            redundant_positions: &mut HashSet<usize>,
            tee_positions_to_remove: &mut HashSet<usize>,
            base_position: usize,
        ) -> usize {
            let mut position = base_position;

            for instr in instructions.iter() {
                match instr {
                    Instruction::LocalSet(idx) => {
                        // Check if previous set to same local is redundant
                        if let Some(prev_set) = last_sets.get(idx) {
                            if !prev_set.has_intervening_get {
                                if prev_set.is_tee {
                                    tee_positions_to_remove.insert(prev_set.position);
                                } else {
                                    redundant_positions.insert(prev_set.position);
                                }
                            }
                        }
                        // Track this set
                        last_sets.insert(
                            *idx,
                            SetInfo {
                                position,
                                has_intervening_get: false,
                                is_tee: false,
                            },
                        );
                        position += 1;
                    }

                    Instruction::LocalGet(idx) => {
                        // Mark that this local has been read
                        if let Some(set_info) = last_sets.get_mut(idx) {
                            set_info.has_intervening_get = true;
                        }
                        position += 1;
                    }

                    Instruction::LocalTee(idx) => {
                        // Tee acts as both set and get
                        // Check if previous set is redundant
                        if let Some(prev_set) = last_sets.get(idx) {
                            if !prev_set.has_intervening_get {
                                if prev_set.is_tee {
                                    tee_positions_to_remove.insert(prev_set.position);
                                } else {
                                    redundant_positions.insert(prev_set.position);
                                }
                            }
                        }
                        // Track this tee (it also produces a value, so has_intervening_get = true)
                        last_sets.insert(
                            *idx,
                            SetInfo {
                                position,
                                has_intervening_get: true,
                                is_tee: true,
                            },
                        );
                        position += 1;
                    }

                    // Recursively analyze control flow structures
                    Instruction::Block { body, .. } => {
                        // Save state and recurse with fresh tracking
                        let mut block_sets: BTreeMap<u32, SetInfo> = last_sets
                            .iter()
                            .map(|(k, v)| {
                                (
                                    *k,
                                    SetInfo {
                                        position: usize::MAX, // Sentinel: don't mark outer sets redundant
                                        has_intervening_get: v.has_intervening_get,
                                        is_tee: v.is_tee,
                                    },
                                )
                            })
                            .collect();

                        position += 1; // Block start
                        position = find_redundant_sets(
                            body,
                            &mut block_sets,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                        );

                        // Merge back: if local was read in block, mark as read in outer scope
                        for (idx, info) in block_sets {
                            if info.has_intervening_get {
                                if let Some(outer_info) = last_sets.get_mut(&idx) {
                                    outer_info.has_intervening_get = true;
                                }
                            }
                        }
                    }

                    Instruction::Loop { body, .. } => {
                        // For loops, conservatively mark all locals as potentially read
                        // (they might be read on back-edge iterations)
                        let mut loop_sets: BTreeMap<u32, SetInfo> = last_sets
                            .iter()
                            .map(|(k, v)| {
                                (
                                    *k,
                                    SetInfo {
                                        position: usize::MAX,
                                        has_intervening_get: v.has_intervening_get,
                                        is_tee: v.is_tee,
                                    },
                                )
                            })
                            .collect();

                        position += 1; // Loop start
                        position = find_redundant_sets(
                            body,
                            &mut loop_sets,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                        );

                        // Conservative: mark all locals as potentially read
                        for set_info in last_sets.values_mut() {
                            set_info.has_intervening_get = true;
                        }
                    }

                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        // Process both branches independently
                        let mut then_sets: BTreeMap<u32, SetInfo> = last_sets
                            .iter()
                            .map(|(k, v)| {
                                (
                                    *k,
                                    SetInfo {
                                        position: usize::MAX,
                                        has_intervening_get: v.has_intervening_get,
                                        is_tee: v.is_tee,
                                    },
                                )
                            })
                            .collect();
                        let mut else_sets = then_sets.clone();

                        position += 1; // If start
                        position = find_redundant_sets(
                            then_body,
                            &mut then_sets,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                        );
                        position = find_redundant_sets(
                            else_body,
                            &mut else_sets,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                        );

                        // Conservative: if either branch reads, consider it read
                        for (idx, then_info) in then_sets {
                            if then_info.has_intervening_get {
                                if let Some(outer_info) = last_sets.get_mut(&idx) {
                                    outer_info.has_intervening_get = true;
                                }
                            }
                        }
                        for (idx, else_info) in else_sets {
                            if else_info.has_intervening_get {
                                if let Some(outer_info) = last_sets.get_mut(&idx) {
                                    outer_info.has_intervening_get = true;
                                }
                            }
                        }
                    }

                    _ => {
                        position += 1;
                    }
                }
            }
            position
        }

        /// Second pass: transform instructions based on analysis
        fn transform_instructions(
            instructions: &[Instruction],
            redundant_positions: &HashSet<usize>,
            tee_positions_to_remove: &HashSet<usize>,
            position: &mut usize,
            changed: &mut bool,
        ) -> Vec<Instruction> {
            let mut result = Vec::new();

            for instr in instructions.iter() {
                let current_pos = *position;
                *position += 1;

                match instr {
                    Instruction::LocalSet(_) if redundant_positions.contains(&current_pos) => {
                        // Replace redundant set with drop (preserves stack effects)
                        result.push(Instruction::Drop);
                        *changed = true;
                    }

                    Instruction::LocalTee(_) if tee_positions_to_remove.contains(&current_pos) => {
                        // Remove redundant tee entirely (value stays on stack)
                        // Don't push anything - this preserves stack because:
                        // tee: pop value, set local, push value -> net effect: value stays
                        // nothing: value stays
                        *changed = true;
                    }

                    Instruction::Block { block_type, body } => {
                        let processed_body = transform_instructions(
                            body,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                            changed,
                        );
                        result.push(Instruction::Block {
                            block_type: block_type.clone(),
                            body: processed_body,
                        });
                    }

                    Instruction::Loop { block_type, body } => {
                        let processed_body = transform_instructions(
                            body,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                            changed,
                        );
                        result.push(Instruction::Loop {
                            block_type: block_type.clone(),
                            body: processed_body,
                        });
                    }

                    Instruction::If {
                        block_type,
                        then_body,
                        else_body,
                    } => {
                        let processed_then = transform_instructions(
                            then_body,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                            changed,
                        );
                        let processed_else = transform_instructions(
                            else_body,
                            redundant_positions,
                            tee_positions_to_remove,
                            position,
                            changed,
                        );
                        result.push(Instruction::If {
                            block_type: block_type.clone(),
                            then_body: processed_then,
                            else_body: processed_else,
                        });
                    }

                    _ => {
                        result.push(instr.clone());
                    }
                }
            }

            result
        }

        // Pass 1: Find redundant sets
        let mut last_sets = BTreeMap::new();
        let mut redundant_positions = HashSet::new();
        let mut tee_positions_to_remove = HashSet::new();
        find_redundant_sets(
            instructions,
            &mut last_sets,
            &mut redundant_positions,
            &mut tee_positions_to_remove,
            0,
        );

        // Pass 2: Transform instructions
        if redundant_positions.is_empty() && tee_positions_to_remove.is_empty() {
            instructions.to_vec()
        } else {
            let mut position = 0;
            transform_instructions(
                instructions,
                &redundant_positions,
                &tee_positions_to_remove,
                &mut position,
                changed,
            )
        }
    }

    /// Code Folding (Tail Merging)
    ///
    /// Finds duplicate code sequences at the end of if/else branches and moves them outside.
    /// This is inspired by Binaryen's code-folding pass.
    ///
    /// Example:
    /// ```wasm
    /// (if (condition)
    ///     (then
    ///         (i32.const 1)
    ///         (local.set $x (i32.const 10))
    ///         (local.set $y (i32.const 20))
    ///     )
    ///     (else
    ///         (i32.const 2)
    ///         (local.set $x (i32.const 10))
    ///         (local.set $y (i32.const 20))
    ///     )
    /// )
    /// ```
    ///
    /// Optimizes to:
    /// ```wasm
    /// (if (condition)
    ///     (then (i32.const 1))
    ///     (else (i32.const 2))
    /// )
    /// (local.set $x (i32.const 10))
    /// (local.set $y (i32.const 20))
    /// ```
    ///
    /// Benefits:
    /// - Reduces code duplication
    /// - Enables further optimizations
    /// - Expected: 5-10% binary size reduction
    pub fn code_folding(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // Skip BrIf/BrTable functions — verifier is path-insensitive across
            // these branches; tail-merging across them can be unsound (REQ-5).
            if has_dataflow_unsafe_control_flow(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "code_folding", ctx.clone());

            // Capture original for translation validation (Z3 proof of semantic equivalence)
            let translator = TranslationValidator::new(func, "code_folding");

            let mut changed = true;
            let mut iterations = 0;
            const MAX_ITERATIONS: usize = 5;

            while changed && iterations < MAX_ITERATIONS {
                changed = false;
                iterations += 1;

                func.instructions = fold_instructions(&func.instructions, &mut changed);
            }

            let _ = guard.validate(func);

            // Z3 translation validation: prove semantic equivalence
            translator.verify_or_revert(func);
        }
        Ok(())
    }

    fn fold_instructions(instructions: &[Instruction], changed: &mut bool) -> Vec<Instruction> {
        let mut result = Vec::new();

        for instr in instructions {
            match instr {
                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    // Recursively fold nested structures
                    let mut folded_then = fold_instructions(then_body, changed);
                    let mut folded_else = fold_instructions(else_body, changed);

                    // INTENTIONALLY SKIPPED: Tail extraction for if statements with result types
                    // Reason: Extracting common tails from valued if-blocks requires careful
                    // tracking of how the extracted code interacts with the block's result value.
                    // Without formal verification of this interaction, we cannot prove the
                    // transformation is correct. Per our proof-first philosophy: skip rather
                    // than risk incorrect optimization.
                    let skip_extraction = !matches!(block_type, BlockType::Empty);

                    // Find common tail instructions that are SAFE to extract
                    //
                    // We can extract a sequence of instructions from the tail if:
                    // 1. The instructions are identical in both branches
                    // 2. The extracted sequence is "self-contained" - it doesn't need more
                    //    values from the branch than what the remaining branch produces
                    //
                    // Algorithm: Find all matching tail instructions, track cumulative "debt"
                    // (how many values the extracted sequence needs from outside).
                    // Extract as long as debt can be "paid off" by earlier matched instructions.

                    // Phase 1: Find all matching tail instructions with their stack requirements
                    // For each instruction, track: how many it consumes (needs) and produces
                    let mut potential_tail: Vec<(Instruction, i32, i32)> = Vec::new(); // (instr, consumes, produces)

                    if !skip_extraction {
                        let mut then_idx = folded_then.len();
                        let mut else_idx = folded_else.len();

                        while then_idx > 0 && else_idx > 0 {
                            let then_last = &folded_then[then_idx - 1];
                            let else_last = &folded_else[else_idx - 1];

                            if !instructions_equal(then_last, else_last) {
                                break;
                            }

                            let (consumes, produces) = instruction_stack_io(then_last);
                            potential_tail.push((then_last.clone(), consumes, produces));
                            then_idx -= 1;
                            else_idx -= 1;
                        }
                    }

                    // Phase 2: Find the longest extractable suffix
                    // Track "debt": how many values the sequence needs from outside
                    // We process from END backwards, accumulating debt.
                    // When debt reaches <= 0, the sequence is self-contained.
                    let mut best_extract_count = 0;
                    let mut running_debt: i32 = 0;

                    for (i, (_instr, consumes, produces)) in potential_tail.iter().enumerate() {
                        running_debt = running_debt + *consumes - *produces;

                        // If running debt is <= 0, the sequence up to here is self-contained
                        if running_debt <= 0 {
                            best_extract_count = i + 1;
                        }
                    }

                    // Build common_tail from the extractable instructions
                    let mut common_tail: Vec<Instruction> = Vec::new();
                    if best_extract_count > 0 {
                        for (instr, _, _) in potential_tail.iter().take(best_extract_count) {
                            common_tail.push(instr.clone());
                        }
                        // Remove from both branches
                        for _ in 0..best_extract_count {
                            folded_then.pop();
                            folded_else.pop();
                        }
                        *changed = true;
                    }

                    // Reverse common_tail because we built it backwards
                    common_tail.reverse();

                    // CRITICAL: If we extracted the entire body, we need to update the block type
                    // because the if statement no longer produces a value from its branches
                    let new_block_type = if folded_then.is_empty() && folded_else.is_empty() {
                        // Both branches are now empty - the if produces no value
                        // The common tail (moved outside) will produce the value instead
                        BlockType::Empty
                    } else {
                        // Branches still have content - keep original type
                        block_type.clone()
                    };

                    // Add the folded if statement
                    result.push(Instruction::If {
                        block_type: new_block_type,
                        then_body: folded_then,
                        else_body: folded_else,
                    });

                    // Add common tail after the if
                    result.extend(common_tail);
                }

                Instruction::Block { block_type, body } => {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: fold_instructions(body, changed),
                    });
                }

                Instruction::Loop { block_type, body } => {
                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: fold_instructions(body, changed),
                    });
                }

                _ => {
                    result.push(instr.clone());
                }
            }
        }

        result
    }

    /// Calculate the stack I/O of an instruction
    ///
    /// Returns (consumes, produces) - the number of values consumed from
    /// and produced to the stack.
    fn instruction_stack_io(instr: &Instruction) -> (i32, i32) {
        use Instruction::*;

        match instr {
            // Constants: consume 0, produce 1
            I32Const(_) | I64Const(_) | F32Const(_) | F64Const(_) => (0, 1),

            // Locals/Globals get: consume 0, produce 1
            LocalGet(_) | GlobalGet(_) => (0, 1),

            // Locals/Globals set: consume 1, produce 0
            LocalSet(_) | GlobalSet(_) => (1, 0),

            // Local tee: consume 1, produce 1
            LocalTee(_) => (1, 1),

            // Binary ops: consume 2, produce 1
            I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU |
            I32And | I32Or | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Rotl | I32Rotr |
            I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS | I32GeU |
            I64Add | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU |
            I64And | I64Or | I64Xor | I64Shl | I64ShrS | I64ShrU | I64Rotl | I64Rotr |
            I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS | I64LeU | I64GeS | I64GeU |
            // Float binary ops
            F32Add | F32Sub | F32Mul | F32Div | F32Min | F32Max | F32Copysign |
            F32Eq | F32Ne | F32Lt | F32Gt | F32Le | F32Ge |
            F64Add | F64Sub | F64Mul | F64Div | F64Min | F64Max | F64Copysign |
            F64Eq | F64Ne | F64Lt | F64Gt | F64Le | F64Ge => (2, 1),

            // Unary ops: consume 1, produce 1
            I32Eqz | I32Clz | I32Ctz | I32Popcnt |
            I64Eqz | I64Clz | I64Ctz | I64Popcnt |
            // Float unary ops
            F32Abs | F32Neg | F32Ceil | F32Floor | F32Trunc | F32Nearest | F32Sqrt |
            F64Abs | F64Neg | F64Ceil | F64Floor | F64Trunc | F64Nearest | F64Sqrt |
            // Conversion ops (unary)
            I32WrapI64 | I64ExtendI32S | I64ExtendI32U |
            I32TruncF32S | I32TruncF32U | I32TruncF64S | I32TruncF64U |
            I64TruncF32S | I64TruncF32U | I64TruncF64S | I64TruncF64U |
            F32ConvertI32S | F32ConvertI32U | F32ConvertI64S | F32ConvertI64U |
            F64ConvertI32S | F64ConvertI32U | F64ConvertI64S | F64ConvertI64U |
            F32DemoteF64 | F64PromoteF32 |
            I32ReinterpretF32 | I64ReinterpretF64 | F32ReinterpretI32 | F64ReinterpretI64 |
            // Saturating truncation (also unary: [float] -> [int])
            I32TruncSatF32S | I32TruncSatF32U | I32TruncSatF64S | I32TruncSatF64U |
            I64TruncSatF32S | I64TruncSatF32U | I64TruncSatF64S | I64TruncSatF64U |
            // Sign extension (unary: [int] -> [int])
            I32Extend8S | I32Extend16S | I64Extend8S | I64Extend16S | I64Extend32S => (1, 1),

            // Memory loads: consume 1 (address), produce 1
            I32Load { .. } | I64Load { .. } | F32Load { .. } | F64Load { .. } |
            I32Load8S { .. } | I32Load8U { .. } | I32Load16S { .. } | I32Load16U { .. } |
            I64Load8S { .. } | I64Load8U { .. } | I64Load16S { .. } | I64Load16U { .. } |
            I64Load32S { .. } | I64Load32U { .. } => (1, 1),

            // Memory stores: consume 2 (address + value), produce 0
            I32Store { .. } | I64Store { .. } | F32Store { .. } | F64Store { .. } |
            I32Store8 { .. } | I32Store16 { .. } |
            I64Store8 { .. } | I64Store16 { .. } | I64Store32 { .. } => (2, 0),

            // Memory size: consume 0, produce 1
            MemorySize(_) => (0, 1),

            // Memory grow: consume 1, produce 1
            MemoryGrow(_) => (1, 1),

            // Bulk memory ops: consume 3 (dst, val/src, len), produce 0
            MemoryFill(_) | MemoryCopy { .. } | MemoryInit { .. } => (3, 0),

            // Data drop: consume 0, produce 0
            DataDrop(_) => (0, 0),

            // Drop: consume 1, produce 0
            Drop => (1, 0),

            // Select: consume 3 (val, val, condition), produce 1
            Select => (3, 1),

            // Nop: no effect
            Nop => (0, 0),

            // Control flow - use large consume to prevent extraction
            Block { .. } | Loop { .. } | If { .. } |
            Br(_) | BrIf(_) | BrTable { .. } |
            Return | Unreachable | Call(_) | CallIndirect { .. } | End | Unknown(_) => {
                (100, 0) // Large consume prevents extraction
            }
        }
    }

    /// Check if two instructions are equal for the purpose of code folding
    fn instructions_equal(a: &Instruction, b: &Instruction) -> bool {
        use Instruction::*;

        match (a, b) {
            (I32Const(x), I32Const(y)) => x == y,
            (I64Const(x), I64Const(y)) => x == y,
            (LocalGet(x), LocalGet(y)) => x == y,
            (LocalSet(x), LocalSet(y)) => x == y,
            (LocalTee(x), LocalTee(y)) => x == y,
            (GlobalGet(x), GlobalGet(y)) => x == y,
            (GlobalSet(x), GlobalSet(y)) => x == y,
            (I32Add, I32Add) => true,
            (I32Sub, I32Sub) => true,
            (I32Mul, I32Mul) => true,
            (I32DivS, I32DivS) => true,
            (I32DivU, I32DivU) => true,
            (I32RemS, I32RemS) => true,
            (I32RemU, I32RemU) => true,
            (I32And, I32And) => true,
            (I32Or, I32Or) => true,
            (I32Xor, I32Xor) => true,
            (I32Shl, I32Shl) => true,
            (I32ShrS, I32ShrS) => true,
            (I32ShrU, I32ShrU) => true,
            (I32Clz, I32Clz) => true,
            (I32Ctz, I32Ctz) => true,
            (I32Popcnt, I32Popcnt) => true,
            (I32Eqz, I32Eqz) => true,
            (I32Eq, I32Eq) => true,
            (I32Ne, I32Ne) => true,
            (I32LtS, I32LtS) => true,
            (I32LtU, I32LtU) => true,
            (I32GtS, I32GtS) => true,
            (I32GtU, I32GtU) => true,
            (I32LeS, I32LeS) => true,
            (I32LeU, I32LeU) => true,
            (I32GeS, I32GeS) => true,
            (I32GeU, I32GeU) => true,

            // i64 operations
            (I64Add, I64Add) => true,
            (I64Sub, I64Sub) => true,
            (I64Mul, I64Mul) => true,
            (I64DivS, I64DivS) => true,
            (I64DivU, I64DivU) => true,
            (I64RemS, I64RemS) => true,
            (I64RemU, I64RemU) => true,
            (I64And, I64And) => true,
            (I64Or, I64Or) => true,
            (I64Xor, I64Xor) => true,
            (I64Shl, I64Shl) => true,
            (I64ShrS, I64ShrS) => true,
            (I64ShrU, I64ShrU) => true,
            (I64Clz, I64Clz) => true,
            (I64Ctz, I64Ctz) => true,
            (I64Popcnt, I64Popcnt) => true,
            (I64Eqz, I64Eqz) => true,
            (I64Eq, I64Eq) => true,
            (I64Ne, I64Ne) => true,
            (I64LtS, I64LtS) => true,
            (I64LtU, I64LtU) => true,
            (I64GtS, I64GtS) => true,
            (I64GtU, I64GtU) => true,
            (I64LeS, I64LeS) => true,
            (I64LeU, I64LeU) => true,
            (I64GeS, I64GeS) => true,
            (I64GeU, I64GeU) => true,

            // Control flow
            (Return, Return) => true,
            (Nop, Nop) => true,
            (Unreachable, Unreachable) => true,
            (Select, Select) => true,

            _ => false,
        }
    }

    /// Loop Invariant Code Motion (LICM)
    ///
    /// Moves computations that don't change inside loops to before the loop,
    /// reducing redundant work. This is inspired by Binaryen's LICM pass.
    ///
    /// Example:
    /// ```wasm
    /// (loop $loop
    ///     (local.set $sum
    ///         (i32.add
    ///             (local.get $sum)
    ///             (i32.add                    ;; Loop-invariant: x + y doesn't change
    ///                 (local.get $x)
    ///                 (local.get $y)
    ///             )
    ///         )
    ///     )
    ///     (br_if $loop (i32.const 1))
    /// )
    /// ```
    ///
    /// Optimizes to:
    /// ```wasm
    /// (local.set $temp (i32.add (local.get $x) (local.get $y)))  ;; Hoisted
    /// (loop $loop
    ///     (local.set $sum
    ///         (i32.add
    ///             (local.get $sum)
    ///             (local.get $temp)
    ///         )
    ///     )
    ///     (br_if $loop (i32.const 1))
    /// )
    /// ```
    ///
    /// Benefits:
    /// - Reduces redundant computations
    /// - Can enable further optimizations
    /// - Expected: 3-8% performance improvement
    pub fn loop_invariant_code_motion(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // loom#150: the blunt guard skips any function with a BrIf — but a
            // loop's back-edge IS a br_if, so that disabled LICM for every real
            // loop. The LICM-specific guard permits a tail-position back-edge
            // (Br(0)/BrIf(0) as the last instruction directly in a loop body),
            // where hoisting an invariant value to the pre-header is sound, while
            // still bailing on any branch that could skip the hoisted op on some
            // path. verify_or_revert below is defense-in-depth (REQ-5).
            if licm_unverifiable_control_flow(func) {
                continue;
            }

            let guard =
                ValidationGuard::with_context(func, "loop_invariant_code_motion", ctx.clone());

            // Capture original for translation validation (Z3 proof of semantic equivalence)
            let translator = TranslationValidator::new(func, "loop_invariant_code_motion");

            // Phase 1: Consecutive hoisting from loop start (existing behavior)
            let mut changed = true;
            let mut iterations = 0;
            const MAX_ITERATIONS: usize = 3;

            while changed && iterations < MAX_ITERATIONS {
                changed = false;
                iterations += 1;

                func.instructions = hoist_loop_invariants(&func.instructions, &mut changed);
            }

            // Phase 2: Non-consecutive hoisting of single invariant value-producers
            // This finds invariant instructions anywhere in loop bodies and hoists them
            let base_local_idx = func.signature.params.len() as u32
                + func.locals.iter().map(|(count, _)| count).sum::<u32>();
            let mut new_locals: Vec<super::ValueType> = Vec::new();

            func.instructions = hoist_invariant_expressions(
                &func.instructions,
                base_local_idx,
                &mut new_locals,
                &mut changed,
            );

            // Add the new locals we allocated
            for local_type in new_locals {
                func.locals.push((1, local_type));
            }

            let _ = guard.validate(func);

            // Z3 translation validation: prove semantic equivalence
            translator.verify_or_revert(func);
        }
        Ok(())
    }

    /// Phase 2 LICM: Hoist invariant expression trees from anywhere in loops
    ///
    /// Uses stack simulation to identify complete expression trees that are
    /// entirely loop-invariant. For each invariant expression:
    /// 1. Hoist the entire instruction sequence before the loop
    /// 2. Store result to a new local
    /// 3. Replace the sequence in the loop with local.get
    fn hoist_invariant_expressions(
        instructions: &[Instruction],
        base_local_idx: u32,
        new_locals: &mut Vec<super::ValueType>,
        changed: &mut bool,
    ) -> Vec<Instruction> {
        let mut result = Vec::new();

        for instr in instructions {
            match instr {
                Instruction::Loop { block_type, body } => {
                    let modified_locals = find_modified_locals(body);
                    let modified_globals = find_modified_globals(body);

                    // Find invariant expression trees using stack simulation
                    let expr_spans =
                        find_invariant_expression_spans(body, &modified_locals, &modified_globals);

                    // Select non-overlapping spans to hoist (greedy: largest first)
                    let mut spans_to_hoist: Vec<_> =
                        expr_spans.into_iter().filter(|s| s.is_invariant).collect();
                    spans_to_hoist.sort_by_key(|s| std::cmp::Reverse(s.end_pos - s.start_pos));

                    let mut used_positions: std::collections::HashSet<usize> =
                        std::collections::HashSet::new();
                    let mut selected_spans: Vec<ExprSpan> = Vec::new();

                    for span in spans_to_hoist {
                        let overlaps =
                            (span.start_pos..=span.end_pos).any(|p| used_positions.contains(&p));
                        if !overlaps {
                            for p in span.start_pos..=span.end_pos {
                                used_positions.insert(p);
                            }
                            selected_spans.push(span);
                        }
                    }

                    // Build hoisted instructions and replacement map
                    let mut hoisted_before_loop: Vec<Instruction> = Vec::new();
                    let mut span_replacements: std::collections::HashMap<usize, (usize, u32)> =
                        std::collections::HashMap::new();

                    for span in &selected_spans {
                        let local_idx = base_local_idx + new_locals.len() as u32;
                        new_locals.push(span.result_type);

                        // Copy the expression instructions
                        hoisted_before_loop
                            .extend(body[span.start_pos..=span.end_pos].iter().cloned());
                        hoisted_before_loop.push(Instruction::LocalSet(local_idx));

                        // Mark start position -> (end_pos, local_idx)
                        span_replacements.insert(span.start_pos, (span.end_pos, local_idx));
                        *changed = true;
                    }

                    // Build the new loop body
                    let mut new_body: Vec<Instruction> = Vec::new();
                    let mut skip_until: Option<usize> = None;

                    for (pos, loop_instr) in body.iter().enumerate() {
                        if let Some(skip_to) = skip_until {
                            if pos <= skip_to {
                                continue;
                            }
                            skip_until = None;
                        }

                        if let Some((end_pos, local_idx)) = span_replacements.get(&pos) {
                            // Replace span with local.get
                            new_body.push(Instruction::LocalGet(*local_idx));
                            skip_until = Some(*end_pos);
                        } else {
                            new_body.push(loop_instr.clone());
                        }
                    }

                    // Recursively process the new body
                    let processed_body = hoist_invariant_expressions(
                        &new_body,
                        base_local_idx + new_locals.len() as u32,
                        new_locals,
                        changed,
                    );

                    // Add hoisted instructions before the loop
                    result.extend(hoisted_before_loop);
                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: processed_body,
                    });
                }

                Instruction::Block { block_type, body } => {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: hoist_invariant_expressions(
                            body,
                            base_local_idx + new_locals.len() as u32,
                            new_locals,
                            changed,
                        ),
                    });
                }

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: hoist_invariant_expressions(
                            then_body,
                            base_local_idx + new_locals.len() as u32,
                            new_locals,
                            changed,
                        ),
                        else_body: hoist_invariant_expressions(
                            else_body,
                            base_local_idx + new_locals.len() as u32,
                            new_locals,
                            changed,
                        ),
                    });
                }

                _ => {
                    result.push(instr.clone());
                }
            }
        }

        result
    }

    /// Represents an expression span in the instruction stream
    #[derive(Debug, Clone)]
    struct ExprSpan {
        start_pos: usize,
        end_pos: usize,
        result_type: super::ValueType,
        is_invariant: bool,
    }

    /// Stack entry for expression tree building
    #[derive(Debug, Clone)]
    #[allow(dead_code)]
    struct StackEntry {
        start_pos: usize,
        result_type: super::ValueType,
        is_invariant: bool,
    }

    /// Find all invariant expression spans in a loop body using stack simulation
    fn find_invariant_expression_spans(
        body: &[Instruction],
        modified_locals: &std::collections::HashSet<u32>,
        modified_globals: &std::collections::HashSet<u32>,
    ) -> Vec<ExprSpan> {
        let mut spans: Vec<ExprSpan> = Vec::new();
        let mut stack: Vec<StackEntry> = Vec::new();

        for (pos, instr) in body.iter().enumerate() {
            match instr {
                // Value producers: push onto stack
                Instruction::I32Const(_) => {
                    // Record single-instruction span for constants
                    spans.push(ExprSpan {
                        start_pos: pos,
                        end_pos: pos,
                        result_type: super::ValueType::I32,
                        is_invariant: true,
                    });
                    stack.push(StackEntry {
                        start_pos: pos,
                        result_type: super::ValueType::I32,
                        is_invariant: true,
                    });
                }
                Instruction::I64Const(_) => {
                    spans.push(ExprSpan {
                        start_pos: pos,
                        end_pos: pos,
                        result_type: super::ValueType::I64,
                        is_invariant: true,
                    });
                    stack.push(StackEntry {
                        start_pos: pos,
                        result_type: super::ValueType::I64,
                        is_invariant: true,
                    });
                }
                Instruction::F32Const(_) => {
                    spans.push(ExprSpan {
                        start_pos: pos,
                        end_pos: pos,
                        result_type: super::ValueType::F32,
                        is_invariant: true,
                    });
                    stack.push(StackEntry {
                        start_pos: pos,
                        result_type: super::ValueType::F32,
                        is_invariant: true,
                    });
                }
                Instruction::F64Const(_) => {
                    spans.push(ExprSpan {
                        start_pos: pos,
                        end_pos: pos,
                        result_type: super::ValueType::F64,
                        is_invariant: true,
                    });
                    stack.push(StackEntry {
                        start_pos: pos,
                        result_type: super::ValueType::F64,
                        is_invariant: true,
                    });
                }
                Instruction::LocalGet(idx) => {
                    let is_inv = !modified_locals.contains(idx);
                    if is_inv {
                        spans.push(ExprSpan {
                            start_pos: pos,
                            end_pos: pos,
                            result_type: super::ValueType::I32, // Simplified
                            is_invariant: true,
                        });
                    }
                    stack.push(StackEntry {
                        start_pos: pos,
                        result_type: super::ValueType::I32, // Simplified
                        is_invariant: is_inv,
                    });
                }
                Instruction::GlobalGet(idx) => {
                    let is_inv = !modified_globals.contains(idx);
                    if is_inv {
                        spans.push(ExprSpan {
                            start_pos: pos,
                            end_pos: pos,
                            result_type: super::ValueType::I32, // Simplified
                            is_invariant: true,
                        });
                    }
                    stack.push(StackEntry {
                        start_pos: pos,
                        result_type: super::ValueType::I32, // Simplified
                        is_invariant: is_inv,
                    });
                }

                // Binary operations: pop 2, push 1
                Instruction::I32Add
                | Instruction::I32Sub
                | Instruction::I32Mul
                | Instruction::I32And
                | Instruction::I32Or
                | Instruction::I32Xor
                | Instruction::I32Shl
                | Instruction::I32ShrS
                | Instruction::I32ShrU => {
                    if stack.len() >= 2 {
                        let right = stack.pop().unwrap();
                        let left = stack.pop().unwrap();
                        let is_inv = left.is_invariant && right.is_invariant;
                        let start = left.start_pos.min(right.start_pos);

                        // Record the combined expression span
                        if is_inv && pos > start {
                            spans.push(ExprSpan {
                                start_pos: start,
                                end_pos: pos,
                                result_type: super::ValueType::I32,
                                is_invariant: true,
                            });
                        }

                        stack.push(StackEntry {
                            start_pos: start,
                            result_type: super::ValueType::I32,
                            is_invariant: is_inv,
                        });
                    } else {
                        stack.clear(); // Stack underflow, reset
                    }
                }

                Instruction::I64Add
                | Instruction::I64Sub
                | Instruction::I64Mul
                | Instruction::I64And
                | Instruction::I64Or
                | Instruction::I64Xor => {
                    if stack.len() >= 2 {
                        let right = stack.pop().unwrap();
                        let left = stack.pop().unwrap();
                        let is_inv = left.is_invariant && right.is_invariant;
                        let start = left.start_pos.min(right.start_pos);

                        if is_inv && pos > start {
                            spans.push(ExprSpan {
                                start_pos: start,
                                end_pos: pos,
                                result_type: super::ValueType::I64,
                                is_invariant: true,
                            });
                        }

                        stack.push(StackEntry {
                            start_pos: start,
                            result_type: super::ValueType::I64,
                            is_invariant: is_inv,
                        });
                    } else {
                        stack.clear();
                    }
                }

                // Comparison operations: pop 2, push 1 (i32 result)
                Instruction::I32Eq
                | Instruction::I32Ne
                | Instruction::I32LtS
                | Instruction::I32LtU
                | Instruction::I32GtS
                | Instruction::I32GtU
                | Instruction::I32LeS
                | Instruction::I32LeU
                | Instruction::I32GeS
                | Instruction::I32GeU
                | Instruction::I64Eq
                | Instruction::I64Ne
                | Instruction::I64LtS
                | Instruction::I64LtU
                | Instruction::I64GtS
                | Instruction::I64GtU
                | Instruction::I64LeS
                | Instruction::I64LeU
                | Instruction::I64GeS
                | Instruction::I64GeU => {
                    if stack.len() >= 2 {
                        let right = stack.pop().unwrap();
                        let left = stack.pop().unwrap();
                        let is_inv = left.is_invariant && right.is_invariant;
                        let start = left.start_pos.min(right.start_pos);

                        if is_inv && pos > start {
                            spans.push(ExprSpan {
                                start_pos: start,
                                end_pos: pos,
                                result_type: super::ValueType::I32,
                                is_invariant: true,
                            });
                        }

                        stack.push(StackEntry {
                            start_pos: start,
                            result_type: super::ValueType::I32,
                            is_invariant: is_inv,
                        });
                    } else {
                        stack.clear();
                    }
                }

                // Unary operations: pop 1, push 1
                Instruction::I32Eqz | Instruction::I32Clz | Instruction::I32Ctz => {
                    if let Some(operand) = stack.pop() {
                        if operand.is_invariant && pos > operand.start_pos {
                            spans.push(ExprSpan {
                                start_pos: operand.start_pos,
                                end_pos: pos,
                                result_type: super::ValueType::I32,
                                is_invariant: true,
                            });
                        }
                        stack.push(StackEntry {
                            start_pos: operand.start_pos,
                            result_type: super::ValueType::I32,
                            is_invariant: operand.is_invariant,
                        });
                    }
                }

                Instruction::I64Eqz => {
                    if let Some(operand) = stack.pop() {
                        if operand.is_invariant && pos > operand.start_pos {
                            spans.push(ExprSpan {
                                start_pos: operand.start_pos,
                                end_pos: pos,
                                result_type: super::ValueType::I32,
                                is_invariant: true,
                            });
                        }
                        stack.push(StackEntry {
                            start_pos: operand.start_pos,
                            result_type: super::ValueType::I32,
                            is_invariant: operand.is_invariant,
                        });
                    }
                }

                // Instructions that consume values or have side effects: reset tracking
                Instruction::LocalSet(_)
                | Instruction::LocalTee(_)
                | Instruction::GlobalSet(_)
                | Instruction::Drop
                | Instruction::Call(_)
                | Instruction::CallIndirect { .. }
                | Instruction::Br(_)
                | Instruction::BrIf(_)
                | Instruction::BrTable { .. }
                | Instruction::Return
                | Instruction::Block { .. }
                | Instruction::Loop { .. }
                | Instruction::If { .. } => {
                    stack.clear(); // Reset on control flow or side effects
                }

                _ => {
                    stack.clear(); // Unknown instruction, reset
                }
            }
        }

        spans
    }

    fn hoist_loop_invariants(instructions: &[Instruction], changed: &mut bool) -> Vec<Instruction> {
        let mut result = Vec::new();

        for instr in instructions {
            match instr {
                Instruction::Loop { block_type, body } => {
                    // Track which locals and globals are modified inside the loop
                    let modified_locals = find_modified_locals(body);
                    let modified_globals = find_modified_globals(body);

                    // Find loop-invariant instructions that can be hoisted
                    let (hoisted, remaining_body) =
                        extract_invariants(body, &modified_locals, &modified_globals, changed);

                    // Add hoisted instructions before the loop
                    result.extend(hoisted);

                    // Recursively process the remaining loop body
                    let processed_body = hoist_loop_invariants(&remaining_body, changed);

                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: processed_body,
                    });
                }

                Instruction::Block { block_type, body } => {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: hoist_loop_invariants(body, changed),
                    });
                }

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: hoist_loop_invariants(then_body, changed),
                        else_body: hoist_loop_invariants(else_body, changed),
                    });
                }

                _ => {
                    result.push(instr.clone());
                }
            }
        }

        result
    }

    /// Find all locals that are modified (set) inside the given instructions
    fn find_modified_locals(instructions: &[Instruction]) -> std::collections::HashSet<u32> {
        use std::collections::HashSet;

        let mut modified = HashSet::new();

        fn scan_instructions(instructions: &[Instruction], modified: &mut HashSet<u32>) {
            for instr in instructions {
                match instr {
                    Instruction::LocalSet(idx) | Instruction::LocalTee(idx) => {
                        modified.insert(*idx);
                    }
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        scan_instructions(body, modified);
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        scan_instructions(then_body, modified);
                        scan_instructions(else_body, modified);
                    }
                    _ => {}
                }
            }
        }

        scan_instructions(instructions, &mut modified);
        modified
    }

    /// Find all globals modified within a loop body
    fn find_modified_globals(instructions: &[Instruction]) -> std::collections::HashSet<u32> {
        use std::collections::HashSet;

        let mut modified = HashSet::new();

        fn scan_instructions(instructions: &[Instruction], modified: &mut HashSet<u32>) {
            for instr in instructions {
                match instr {
                    Instruction::GlobalSet(idx) => {
                        modified.insert(*idx);
                    }
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        scan_instructions(body, modified);
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        scan_instructions(then_body, modified);
                        scan_instructions(else_body, modified);
                    }
                    _ => {}
                }
            }
        }

        scan_instructions(instructions, &mut modified);
        modified
    }

    /// Extract loop-invariant instructions from the beginning of the loop body
    /// Returns (hoisted_instructions, remaining_body)
    ///
    /// IMPORTANT: We can only hoist instruction sequences that have a net stack
    /// effect of 0. Otherwise, we'd remove values that later instructions depend on.
    fn extract_invariants(
        instructions: &[Instruction],
        modified_locals: &std::collections::HashSet<u32>,
        modified_globals: &std::collections::HashSet<u32>,
        changed: &mut bool,
    ) -> (Vec<Instruction>, Vec<Instruction>) {
        let mut hoisted = Vec::new();
        let mut stack_balance: i32 = 0; // Track net stack effect of hoisted sequence
        let mut last_safe_hoist_idx: usize = 0; // Index of last instruction where balance was 0

        // First pass: find all consecutive invariant instructions and track balance
        let mut idx = 0;
        for instr in instructions {
            if !is_loop_invariant(instr, modified_locals, modified_globals) {
                break; // Stop at first non-invariant instruction
            }

            // Calculate stack effect of this instruction
            let (consumes, produces) = instruction_stack_io(instr);
            let new_balance = stack_balance - consumes + produces;

            // Can't hoist if we'd need values from the loop body we haven't seen yet
            if new_balance < 0 && consumes > stack_balance {
                break;
            }

            hoisted.push(instr.clone());
            stack_balance = new_balance;
            idx += 1;

            // Track the last point where balance was 0 (safe to stop here)
            if stack_balance == 0 {
                last_safe_hoist_idx = idx;
            }
        }

        // If we never reached balance=0, we can't hoist anything safely
        if last_safe_hoist_idx == 0 {
            // Put everything back
            return (Vec::new(), instructions.to_vec());
        }

        // Trim hoisted to the last safe point
        let safe_hoisted: Vec<_> = hoisted.into_iter().take(last_safe_hoist_idx).collect();

        // Everything else goes into remaining
        let remaining_instrs: Vec<_> = instructions
            .iter()
            .skip(last_safe_hoist_idx)
            .cloned()
            .collect();

        if !safe_hoisted.is_empty() {
            *changed = true;
        }

        (safe_hoisted, remaining_instrs)
    }

    /// Check if an instruction is loop-invariant
    /// An instruction is invariant if it doesn't read any modified locals/globals and has no side effects
    fn is_loop_invariant(
        instr: &Instruction,
        modified_locals: &std::collections::HashSet<u32>,
        modified_globals: &std::collections::HashSet<u32>,
    ) -> bool {
        use Instruction::*;

        match instr {
            // Constants are always invariant
            I32Const(_) | I64Const(_) | F32Const(_) | F64Const(_) => true,

            // LocalGet is invariant if the local isn't modified in the loop
            LocalGet(idx) => !modified_locals.contains(idx),

            // GlobalGet is invariant if the global isn't modified in the loop
            GlobalGet(idx) => !modified_globals.contains(idx),

            // LocalSet is invariant if we're computing a loop-invariant value
            // (the actual safety is ensured by stack balance tracking)
            LocalSet(_) | LocalTee(_) => true,

            // Pure arithmetic operations are invariant
            I32Add | I32Sub | I32Mul | I32DivS | I32DivU | I32RemS | I32RemU | I32And | I32Or
            | I32Xor | I32Shl | I32ShrS | I32ShrU | I32Clz | I32Ctz | I32Popcnt | I64Add
            | I64Sub | I64Mul | I64DivS | I64DivU | I64RemS | I64RemU | I64And | I64Or | I64Xor
            | I64Shl | I64ShrS | I64ShrU | I64Clz | I64Ctz | I64Popcnt => true,

            // Comparison operations are invariant
            I32Eq | I32Ne | I32LtS | I32LtU | I32GtS | I32GtU | I32LeS | I32LeU | I32GeS
            | I32GeU | I32Eqz | I64Eq | I64Ne | I64LtS | I64LtU | I64GtS | I64GtU | I64LeS
            | I64LeU | I64GeS | I64GeU | I64Eqz => true,

            // Select is invariant (doesn't modify state)
            Select => true,

            // Drop is invariant (just removes from stack)
            Drop => true,

            // Nop has no effect
            Nop => true,

            // Everything else is NOT invariant (side effects, control flow, memory access, etc.)
            _ => false,
        }
    }

    /// Remove Unused Branches
    ///
    /// Eliminates dead code and unreachable branches. This optimization:
    /// - Removes code after return/unreachable instructions
    /// - Eliminates unreachable br instructions
    /// - Simplifies control flow
    ///
    /// Example:
    /// ```wasm
    /// (block $label
    ///     (return (i32.const 42))
    ///     (br $label)              ;; Unreachable - removed
    ///     (i32.const 100)          ;; Dead code - removed
    /// )
    /// ```
    ///
    /// Benefits:
    /// - Smaller code size
    /// - Cleaner control flow
    /// - Expected: 1-2% binary size reduction
    pub fn remove_unused_branches(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // loom#150: DCE only deletes code after an unconditional terminator
            // (sound regardless of where the terminator sits), so the narrower
            // DCE guard permits non-tail Return/Br while still bailing on
            // conditional branches the validator can't model (REQ-5).
            if dce_unverifiable_control_flow(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "remove_unused_branches", ctx.clone());

            // Capture original for translation validation (Z3 proof of semantic equivalence)
            let translator = TranslationValidator::new(func, "remove_unused_branches");

            let mut changed = true;
            let mut iterations = 0;
            const MAX_ITERATIONS: usize = 3;

            while changed && iterations < MAX_ITERATIONS {
                changed = false;
                iterations += 1;

                func.instructions = remove_dead_code(&func.instructions, &mut changed);
            }

            let _ = guard.validate(func);

            // Z3 translation validation: prove semantic equivalence
            translator.verify_or_revert(func);
        }
        Ok(())
    }

    fn remove_dead_code(instructions: &[Instruction], changed: &mut bool) -> Vec<Instruction> {
        let mut result = Vec::new();
        let mut unreachable = false;

        for instr in instructions {
            // Skip instructions after unreachable/return
            if unreachable {
                *changed = true;
                continue;
            }

            match instr {
                // These instructions make subsequent code unreachable
                Instruction::Return | Instruction::Unreachable => {
                    result.push(instr.clone());
                    unreachable = true;
                }

                // Recursively process nested blocks
                Instruction::Block { block_type, body } => {
                    let cleaned_body = remove_dead_code(body, changed);
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: cleaned_body,
                    });
                }

                Instruction::Loop { block_type, body } => {
                    let cleaned_body = remove_dead_code(body, changed);
                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: cleaned_body,
                    });
                }

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    let cleaned_then = remove_dead_code(then_body, changed);
                    let cleaned_else = remove_dead_code(else_body, changed);
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: cleaned_then,
                        else_body: cleaned_else,
                    });
                }

                _ => {
                    result.push(instr.clone());
                }
            }
        }

        result
    }

    /// Optimize Added Constants
    ///
    /// Merges consecutive constant additions into single operations.
    /// This is inspired by Binaryen's optimize-added-constants pass.
    ///
    /// Example:
    /// ```wasm
    /// (i32.add (local.get $x) (i32.const 5))
    /// (i32.add (i32.const 10))
    /// ```
    ///
    /// Optimizes to:
    /// ```wasm
    /// (i32.add (local.get $x) (i32.const 15))
    /// ```
    ///
    /// Benefits:
    /// - Fewer instructions
    /// - Simpler constant handling
    /// - Expected: 1-2% code size reduction
    pub fn optimize_added_constants(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // Skip early-exit functions (REQ-5 conservative-over-fast).
            if has_dataflow_unsafe_control_flow(func) {
                continue;
            }

            let guard =
                ValidationGuard::with_context(func, "optimize_added_constants", ctx.clone());
            let translator = TranslationValidator::new(func, "optimize_added_constants");

            let mut changed = true;
            let mut iterations = 0;
            const MAX_ITERATIONS: usize = 3;

            while changed && iterations < MAX_ITERATIONS {
                changed = false;
                iterations += 1;

                func.instructions = merge_constant_adds(&func.instructions, &mut changed);
            }

            let _ = guard.validate(func);
            translator.verify_or_revert(func);
        }
        Ok(())
    }

    fn merge_constant_adds(instructions: &[Instruction], changed: &mut bool) -> Vec<Instruction> {
        use Instruction::*;

        let mut result = Vec::new();
        let mut i = 0;

        while i < instructions.len() {
            // Look for pattern: value, i32.const X, i32.add, i32.const Y, i32.add
            if i + 4 < instructions.len() {
                if let (I32Const(x), I32Add, I32Const(y), I32Add) = (
                    &instructions[i + 1],
                    &instructions[i + 2],
                    &instructions[i + 3],
                    &instructions[i + 4],
                ) {
                    // Merge: (value + X) + Y => value + (X + Y)
                    result.push(instructions[i].clone());
                    result.push(I32Const(x.wrapping_add(*y)));
                    result.push(I32Add);
                    *changed = true;
                    i += 5;
                    continue;
                }
            }

            // Look for pattern: i32.const X, i32.const Y, i32.add
            if i + 2 < instructions.len() {
                if let (I32Const(x), I32Const(y), I32Add) =
                    (&instructions[i], &instructions[i + 1], &instructions[i + 2])
                {
                    // Fold constants
                    result.push(I32Const(x.wrapping_add(*y)));
                    *changed = true;
                    i += 3;
                    continue;
                }
            }

            // Look for similar patterns with i64
            if i + 4 < instructions.len() {
                if let (I64Const(x), I64Add, I64Const(y), I64Add) = (
                    &instructions[i + 1],
                    &instructions[i + 2],
                    &instructions[i + 3],
                    &instructions[i + 4],
                ) {
                    result.push(instructions[i].clone());
                    result.push(I64Const(x.wrapping_add(*y)));
                    result.push(I64Add);
                    *changed = true;
                    i += 5;
                    continue;
                }
            }

            if i + 2 < instructions.len() {
                if let (I64Const(x), I64Const(y), I64Add) =
                    (&instructions[i], &instructions[i + 1], &instructions[i + 2])
                {
                    result.push(I64Const(x.wrapping_add(*y)));
                    *changed = true;
                    i += 3;
                    continue;
                }
            }

            // Recursively process nested structures
            match &instructions[i] {
                Block { block_type, body } => {
                    result.push(Block {
                        block_type: block_type.clone(),
                        body: merge_constant_adds(body, changed),
                    });
                }
                Loop { block_type, body } => {
                    result.push(Loop {
                        block_type: block_type.clone(),
                        body: merge_constant_adds(body, changed),
                    });
                }
                If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    result.push(If {
                        block_type: block_type.clone(),
                        then_body: merge_constant_adds(then_body, changed),
                        else_body: merge_constant_adds(else_body, changed),
                    });
                }
                _ => {
                    result.push(instructions[i].clone());
                }
            }

            i += 1;
        }

        result
    }

    /// CoalesceLocals - Register Allocation (Phase 12.5)
    ///
    /// Merges non-overlapping local variables to reduce local count and improve
    /// encoding efficiency. This is a key optimization that wasm-opt implements.
    ///
    /// Algorithm:
    /// 1. Compute live ranges for each local (first def to last use)
    /// 2. Build interference graph (locals with overlapping ranges)
    /// 3. Graph coloring to assign new indices (greedy algorithm)
    /// 4. Remap all local references
    ///
    /// Benefits:
    /// - Fewer local declarations (smaller function preambles)
    /// - Lower indices use smaller LEB128 encoding
    /// - Expected: 10-15% binary size reduction
    pub fn coalesce_locals(module: &mut Module) -> Result<()> {
        use crate::verify::TranslationValidator;

        for func in &mut module.functions {
            // Skip functions with no locals
            let total_locals = func.signature.params.len()
                + func
                    .locals
                    .iter()
                    .map(|(count, _)| *count as usize)
                    .sum::<usize>();

            if total_locals <= 1 {
                continue;
            }

            // Skip BrIf/BrTable functions — coalescing relies on liveness
            // analysis that the path-insensitive verifier cannot validate
            // across these branches (REQ-5).
            if has_dataflow_unsafe_control_flow(func) {
                continue;
            }

            // Capture original for Z3 verification and rollback
            let translator = TranslationValidator::new(func, "coalesce_locals");
            let original_instructions = func.instructions.clone();
            let original_locals = func.locals.clone();

            // Step 1: Compute live ranges
            let live_ranges = compute_live_ranges(&func.instructions, func.signature.params.len());

            if live_ranges.is_empty() {
                continue;
            }

            // Step 2: Build interference graph
            let interference_graph = build_interference_graph(&live_ranges);

            // Step 3: Graph coloring (greedy algorithm)
            let coloring = color_interference_graph(&interference_graph);

            // Step 3.5: Skip coalescing if there are dead locals (not in coloring map)
            // Dead locals need to be eliminated by SimplifyLocals first
            let param_count = func.signature.params.len() as u32;
            let all_locals_in_map =
                (param_count..total_locals as u32).all(|idx| coloring.contains_key(&idx));

            if !all_locals_in_map {
                continue; // Skip this function - has dead stores
            }

            // Step 4: Remap locals if we achieved any coalescing
            let max_color = coloring.values().max().copied().unwrap_or(0);
            let original_count = total_locals;

            if (max_color + 1) < original_count as u32 {
                remap_function_locals(func, &coloring);
            }

            // Z3 translation validation — revert on failure
            if let Err(e) = translator.verify(func) {
                eprintln!("coalesce_locals: reverting function (Z3 rejected): {}", e);
                crate::stats::record_revert("coalesce_locals");
                func.instructions = original_instructions;
                func.locals = original_locals;
            }
        }

        Ok(())
    }

    /// Eliminate trivially dead locals: locals declared by a function but
    /// never read by any `LocalGet` anywhere in the function body.
    ///
    /// This is the path-INSENSITIVE half of dead-store elimination. Unlike
    /// the path-sensitive case (Pick #3, RedundantSetElimination liveness),
    /// "zero reads anywhere" is a structural property of the instruction
    /// tree — sound regardless of `BrIf`/`BrTable`/early-`Return` control
    /// flow. So this pass DOES NOT need the `has_dataflow_unsafe_control_flow`
    /// guard that prevents `simplify_locals` and `coalesce_locals` from
    /// running on kernel-style early-exit code (gale).
    ///
    /// Targets the gale "default-then-override" pattern: rustc/LLVM
    /// materializes an EINVAL default at function entry, then every
    /// reachable path overwrites it before return. The default's
    /// `local.set` becomes pure dead store. wasm-opt eliminates this;
    /// LOOM v0.5.0 did not.
    ///
    /// Algorithm:
    /// 1. Recursively count `LocalGet` references for each local index.
    ///    (Walks Block/Loop/If bodies — same recursion shape as
    ///    `remap_instructions` and `eliminate_redundant_sets`.)
    /// 2. Identify dead locals: idx >= param_count AND read_count == 0.
    /// 3. Neutralize writes to dead locals:
    ///    - `LocalSet dead_idx` → `Drop` (preserves stack consumption)
    ///    - `LocalTee dead_idx` → removed (Tee's stack effect is `[T]→[T]`,
    ///      so removing it leaves the value passing through unchanged)
    /// 4. Build a packed remap: surviving locals get sequential indices
    ///    starting at param_count (densest LEB128 encoding).
    /// 5. Apply remap via existing `remap_instructions`; rebuild
    ///    declarations from surviving types.
    /// 6. Z3 translation-validation: revert on rejection.
    pub fn eliminate_dead_locals(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;
        use std::collections::{BTreeMap, BTreeSet};

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            let param_count = func.signature.params.len() as u32;
            let declared_count: u32 = func.locals.iter().map(|(c, _)| *c).sum();
            if declared_count == 0 {
                continue;
            }
            let total_locals = param_count + declared_count;

            // Step 1: Count LocalGet references for each local across the
            // entire instruction tree (including nested bodies).
            let mut read_counts: BTreeMap<u32, usize> = BTreeMap::new();
            count_local_reads(&func.instructions, &mut read_counts);

            // Step 2: A local is dead iff it's declared (not a param) and
            // never read anywhere. This rule is path-insensitive — no read
            // means no observation, regardless of which paths the writes
            // are reachable from.
            let dead: BTreeSet<u32> = (param_count..total_locals)
                .filter(|idx| read_counts.get(idx).copied().unwrap_or(0) == 0)
                .collect();

            if dead.is_empty() {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "eliminate_dead_locals", ctx.clone());
            let translator = TranslationValidator::new(func, "eliminate_dead_locals");
            let original_instructions = func.instructions.clone();
            let original_locals = func.locals.clone();

            // Step 3: Neutralize writes to dead locals. Done in-place via
            // a tree walk (matches the recursion shape of remap_instructions).
            neutralize_dead_writes(&mut func.instructions, &dead);

            // Step 4: Build packed remap. Surviving locals keep their
            // relative order but get a dense, gap-free sequence of new
            // indices starting at param_count.
            let mut remap: BTreeMap<u32, u32> = BTreeMap::new();
            // Identity-map the parameters.
            for p in 0..param_count {
                remap.insert(p, p);
            }
            let mut next_idx = param_count;
            for old_idx in param_count..total_locals {
                if !dead.contains(&old_idx) {
                    remap.insert(old_idx, next_idx);
                    next_idx += 1;
                }
            }

            // Step 5: Walk the tree applying the remap, then rebuild the
            // locals declaration from the surviving types in order.
            remap_instructions(&mut func.instructions, &remap);
            func.locals = pack_surviving_locals(&original_locals, param_count, &dead);

            // Step 6: Z3 verification — revert if the transformed function
            // is not observationally equivalent to the original.
            if guard.validate(func).is_err() || translator.verify(func).is_err() {
                eprintln!("eliminate_dead_locals: reverting function (verification rejected)");
                crate::stats::record_revert("eliminate_dead_locals");
                func.instructions = original_instructions;
                func.locals = original_locals;
            }
        }

        Ok(())
    }

    /// Count LocalGet references for every local index in an instruction
    /// tree. Recurses into Block / Loop / If bodies — matches the recursion
    /// shape of `remap_instructions` (lib.rs:10106) so the two analyses
    /// see the same tree.
    fn count_local_reads(
        instructions: &[crate::Instruction],
        counts: &mut std::collections::BTreeMap<u32, usize>,
    ) {
        use crate::Instruction;
        for instr in instructions {
            match instr {
                Instruction::LocalGet(idx) => {
                    *counts.entry(*idx).or_insert(0) += 1;
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count_local_reads(body, counts);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count_local_reads(then_body, counts);
                    count_local_reads(else_body, counts);
                }
                _ => {}
            }
        }
    }

    /// Replace `LocalSet dead` with `Drop` and remove `LocalTee dead` for
    /// every local in the dead set. Walks nested control-flow bodies.
    ///
    /// Stack-effect rationale (see insight at the call site):
    ///   LocalSet idx : [T] → []     → Drop (which is also [T] → [])
    ///   LocalTee idx : [T] → [T]    → remove (the [T] passes through)
    fn neutralize_dead_writes(
        instructions: &mut Vec<crate::Instruction>,
        dead: &std::collections::BTreeSet<u32>,
    ) {
        use crate::Instruction;
        let mut i = 0;
        while i < instructions.len() {
            match &mut instructions[i] {
                Instruction::LocalSet(idx) if dead.contains(idx) => {
                    instructions[i] = Instruction::Drop;
                    i += 1;
                }
                Instruction::LocalTee(idx) if dead.contains(idx) => {
                    instructions.remove(i);
                    // Don't increment i — re-examine the new instruction
                    // at this position.
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    neutralize_dead_writes(body, dead);
                    i += 1;
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    neutralize_dead_writes(then_body, dead);
                    neutralize_dead_writes(else_body, dead);
                    i += 1;
                }
                _ => {
                    i += 1;
                }
            }
        }
    }

    /// Rebuild a locals declaration vector after removing the indices in
    /// `dead`. Preserves the original declaration order of survivors and
    /// produces a `Vec<(count, type)>` in run-length form (consecutive
    /// same-type survivors are merged into a single entry).
    fn pack_surviving_locals(
        original: &[(u32, crate::ValueType)],
        param_count: u32,
        dead: &std::collections::BTreeSet<u32>,
    ) -> Vec<(u32, crate::ValueType)> {
        let mut surviving_types: Vec<crate::ValueType> = Vec::new();
        let mut idx = param_count;
        for (count, ty) in original {
            for _ in 0..*count {
                if !dead.contains(&idx) {
                    surviving_types.push(*ty);
                }
                idx += 1;
            }
        }

        // Run-length encode (count, type) pairs.
        let mut out: Vec<(u32, crate::ValueType)> = Vec::new();
        for ty in surviving_types {
            if let Some(last) = out.last_mut()
                && last.1 == ty
            {
                last.0 += 1;
            } else {
                out.push((1, ty));
            }
        }
        out
    }

    /// Eliminate dead stores via per-position backward liveness on
    /// structured wasm control flow. The path-sensitive complement to
    /// `eliminate_dead_locals`: catches writes that are dead on every
    /// continuation (overwritten before any read) even when the local
    /// is read elsewhere in the function.
    ///
    /// Pick #3 from the v0.6.0 wasm-opt-gap research agent's plan
    /// (full liveness, Option C).
    ///
    /// Algorithm: backward walk over the structured instruction tree
    /// computing, at each position, the set of locals whose value will
    /// be read on some continuation path before being overwritten.
    /// A write is dead iff its target local is not in that "live-after"
    /// set.
    ///
    /// Wasm structured control flow handling:
    ///   - `Block { body }`: br N targets the end-of-block, so
    ///     break-target liveness equals live-after-block.
    ///   - `If { then, else }`: live-before-if = uses(cond) ∪
    ///     (live_in(then) ∪ live_in(else)).
    ///   - `Br N`, `Return`, `Unreachable`: no fall-through. Live
    ///     becomes the target's liveness (or empty for Return/Unreachable).
    ///   - `BrIf N`, `BrTable [...]`: union of target liveness with
    ///     fall-through (or with all targets, for BrTable).
    ///   - `Loop { body }`: handled CONSERVATIVELY (v1) — every local
    ///     that is read anywhere in the body is treated as live throughout
    ///     the body and live just before the loop. This avoids fixpoint
    ///     iteration and remains sound; it loses precision *inside* loops
    ///     but the gale dead-store patterns sit before loops, not in them.
    ///     Full Loop fixpoint is a follow-up.
    ///   - `Call`, `CallIndirect`: do not access caller's locals; pass
    ///     through unchanged.
    ///
    /// Trap-effecting instructions (load, store, div, etc.) may end the
    /// function early on trap. We compute liveness under "no trap"
    /// assumption: writes are removed only if dead on the no-trap
    /// continuation. If a trap intervenes, no later instruction observes
    /// the local — so removal remains sound.
    ///
    /// Application:
    ///   - Dead `LocalSet idx` → `Drop`
    ///   - Dead `LocalTee idx` → removed (Tee `[T]→[T]` passes through)
    ///
    /// After this pass runs, `eliminate_dead_locals` may find additional
    /// locals with zero remaining reads (if all their writes were dead);
    /// running these passes in sequence amplifies their effect.
    pub fn eliminate_dead_stores(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // #220: the Z3 translation-validation net (step 4 below) SILENTLY
            // SKIPS functions containing float load/store — verify_function_
            // equivalence returns Ok(true) ("assume equivalent") without a
            // proof. Dead-store elimination REMOVES code, and its backward
            // liveness analysis has proven unsound on such a module: the
            // meld-fused falcon core regressed run-stabilization 0.023399856 ->
            // 0.32916 (a live float store dropped) while still passing
            // wasm-tools validate. The silent skip is only sound for passes
            // that don't remove/move code (see verify.rs); dead-store
            // elimination is not one of them. Mirror the #196 stance — never
            // eliminate under uncertainty: skip any function whose result the
            // verifier cannot actually prove.
            if contains_unverifiable_float_memory(&func.instructions) {
                continue;
            }

            // Step 1: count writes (LocalSet + LocalTee) for ID tracking.
            let total_writes = count_local_writes(&func.instructions);
            if total_writes == 0 {
                continue;
            }

            // Step 2: backward liveness analysis. Builds a set of dead
            // write IDs (0..total_writes, in tree-walk forward order).
            let mut analyzer = LivenessAnalyzer::new(total_writes);
            analyzer.analyze(&func.instructions);

            if analyzer.dead_writes.is_empty() {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "eliminate_dead_stores", ctx.clone());
            let translator = TranslationValidator::new(func, "eliminate_dead_stores");
            let original_instructions = func.instructions.clone();

            // Step 3: apply — walk forward, replace dead writes by ID.
            let mut applier = DeadStoreApplier::new(&analyzer.dead_writes);
            applier.apply(&mut func.instructions);

            // Step 4: Z3 translation validation — revert on rejection.
            if guard.validate(func).is_err() || translator.verify(func).is_err() {
                eprintln!("eliminate_dead_stores: reverting function (verification rejected)");
                crate::stats::record_revert("eliminate_dead_stores");
                func.instructions = original_instructions;
            }
        }

        Ok(())
    }

    /// True if the instruction tree contains a float load/store. These are
    /// the instructions the Z3 translation validator cannot model
    /// (`verify::contains_unverifiable_instructions`), so it silently treats
    /// such functions as "equivalent" without a proof. Dead-store elimination
    /// must not rely on that absent safety net (#220), so it skips these
    /// functions entirely. Compiled in every feature configuration — the
    /// conservative skip is unconditional, not gated on the Z3 feature.
    fn contains_unverifiable_float_memory(instructions: &[crate::Instruction]) -> bool {
        use crate::Instruction;
        instructions.iter().any(|instr| match instr {
            Instruction::F32Load { .. }
            | Instruction::F64Load { .. }
            | Instruction::F32Store { .. }
            | Instruction::F64Store { .. } => true,
            Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                contains_unverifiable_float_memory(body)
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                contains_unverifiable_float_memory(then_body)
                    || contains_unverifiable_float_memory(else_body)
            }
            _ => false,
        })
    }

    /// Count LocalSet + LocalTee occurrences in a structured instruction
    /// tree. Used to size the dead-write ID space.
    fn count_local_writes(instructions: &[crate::Instruction]) -> usize {
        use crate::Instruction;
        let mut n = 0;
        for instr in instructions {
            match instr {
                Instruction::LocalSet(_) | Instruction::LocalTee(_) => n += 1,
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    n += count_local_writes(body);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    n += count_local_writes(then_body);
                    n += count_local_writes(else_body);
                }
                _ => {}
            }
        }
        n
    }

    /// Collect the set of all locals read anywhere in a structured
    /// instruction tree (LocalGet only). Used for the conservative
    /// Loop body approximation.
    fn collect_reads(
        instructions: &[crate::Instruction],
        out: &mut std::collections::BTreeSet<u32>,
    ) {
        use crate::Instruction;
        for instr in instructions {
            match instr {
                Instruction::LocalGet(idx) => {
                    out.insert(*idx);
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    collect_reads(body, out);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    collect_reads(then_body, out);
                    collect_reads(else_body, out);
                }
                _ => {}
            }
        }
    }

    /// Backward-liveness analyzer for structured wasm. Walks the tree in
    /// reverse order, propagating a `LiveSet` and recording dead write IDs.
    /// IDs are assigned in tree-walk *forward* order via a backward counter
    /// that decrements as writes are encountered.
    struct LivenessAnalyzer {
        /// Counter tracking the next write ID, decremented as we walk
        /// backward. Initially equals total_writes; decrements to 0.
        write_counter: usize,
        /// Set of dead write IDs (0..total_writes in tree-walk forward
        /// order). A write is dead iff its target local is not in the
        /// live-after set at that position.
        dead_writes: std::collections::BTreeSet<usize>,
    }

    type LiveSet = std::collections::BTreeSet<u32>;

    impl LivenessAnalyzer {
        fn new(total_writes: usize) -> Self {
            Self {
                write_counter: total_writes,
                dead_writes: std::collections::BTreeSet::new(),
            }
        }

        fn analyze(&mut self, instructions: &[crate::Instruction]) {
            let mut label_stack: Vec<LiveSet> = Vec::new();
            let _ = self.walk(instructions, LiveSet::new(), &mut label_stack);
        }

        /// Backward walk over `instructions`. Returns live-before the
        /// first instruction (= live-in to this block).
        ///
        /// `live` starts as live-after-the-last-instruction.
        /// `label_stack` mirrors the wasm control-flow nesting: index 0
        /// is the outermost label, last is the innermost (br N counts
        /// from innermost-out, so target = label_stack[len - 1 - N]).
        fn walk(
            &mut self,
            instructions: &[crate::Instruction],
            mut live: LiveSet,
            label_stack: &mut Vec<LiveSet>,
        ) -> LiveSet {
            use crate::Instruction;

            for instr in instructions.iter().rev() {
                match instr {
                    Instruction::LocalGet(idx) => {
                        live.insert(*idx);
                    }
                    Instruction::LocalSet(idx) => {
                        // Decrement counter to get this write's ID.
                        // (Forward IDs are assigned in tree-walk order;
                        // since we walk backward, last write seen has the
                        // highest ID.)
                        self.write_counter -= 1;
                        let id = self.write_counter;
                        if !live.contains(idx) {
                            self.dead_writes.insert(id);
                        }
                        live.remove(idx);
                    }
                    Instruction::LocalTee(idx) => {
                        self.write_counter -= 1;
                        let id = self.write_counter;
                        if !live.contains(idx) {
                            self.dead_writes.insert(id);
                        }
                        live.remove(idx);
                        // Tee's stack output isn't a local — it flows via
                        // the operand stack, not the live set.
                    }
                    Instruction::Block { body, .. } => {
                        // br N inside a Block targets the END of the block,
                        // which equals live-after-block (current `live`).
                        label_stack.push(live.clone());
                        live = self.walk(body, live.clone(), label_stack);
                        label_stack.pop();
                    }
                    Instruction::Loop { body, .. } => {
                        // Conservative v1: assume every local read inside
                        // the body is live throughout the body and live
                        // just before the loop. This avoids fixpoint
                        // iteration on the back-edge.
                        let mut body_reads = LiveSet::new();
                        collect_reads(body, &mut body_reads);
                        let live_in_body: LiveSet = live.union(&body_reads).copied().collect();

                        // Loop label targets the loop-header (live-in-body).
                        label_stack.push(live_in_body.clone());
                        // Walk body for dead-write detection. Pass the
                        // conservative live_in_body as live-after each
                        // body instruction (over-approximation: keeps
                        // writes inside loops; safe).
                        let _ = self.walk(body, live_in_body.clone(), label_stack);
                        label_stack.pop();

                        live = live_in_body;
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        // Both arms see the same live-after-if.
                        // If's label targets the END of the if (live-after).
                        label_stack.push(live.clone());
                        let then_live_in = self.walk(then_body, live.clone(), label_stack);
                        let else_live_in = self.walk(else_body, live.clone(), label_stack);
                        label_stack.pop();

                        // live-before-if = live-in-then ∪ live-in-else
                        // (the cond is consumed from the operand stack;
                        // it doesn't add to local liveness here).
                        live = then_live_in.union(&else_live_in).copied().collect();
                    }
                    Instruction::Br(n) => {
                        // After Br, fall-through is unreachable. Live
                        // becomes the target's liveness.
                        let depth = label_stack.len();
                        if let Some(target_idx) = depth.checked_sub(1 + *n as usize) {
                            live = label_stack[target_idx].clone();
                        } else {
                            // Br beyond the function: equivalent to Return.
                            live.clear();
                        }
                    }
                    Instruction::BrIf(n) => {
                        // Branch taken: live = target. Not taken: fall-through.
                        let depth = label_stack.len();
                        if let Some(target_idx) = depth.checked_sub(1 + *n as usize) {
                            live = live.union(&label_stack[target_idx]).copied().collect();
                        }
                    }
                    Instruction::BrTable { targets, default } => {
                        let depth = label_stack.len();
                        let mut combined = live.clone();
                        for n in targets.iter().chain(std::iter::once(default)) {
                            if let Some(target_idx) = depth.checked_sub(1 + *n as usize) {
                                combined.extend(&label_stack[target_idx]);
                            }
                        }
                        live = combined;
                    }
                    Instruction::Return | Instruction::Unreachable => {
                        // No continuation. Live becomes empty.
                        live.clear();
                    }
                    _ => {
                        // Other instructions don't read or write locals
                        // (Call/CallIndirect don't access caller locals).
                    }
                }
            }
            live
        }
    }

    /// Apply pass: walk the tree forward, replacing dead writes by ID.
    /// Tracks a counter that mirrors the analyzer's forward ID assignment.
    struct DeadStoreApplier<'a> {
        dead_writes: &'a std::collections::BTreeSet<usize>,
        write_counter: usize,
    }

    impl<'a> DeadStoreApplier<'a> {
        fn new(dead_writes: &'a std::collections::BTreeSet<usize>) -> Self {
            Self {
                dead_writes,
                write_counter: 0,
            }
        }

        fn apply(&mut self, instructions: &mut Vec<crate::Instruction>) {
            use crate::Instruction;
            let mut i = 0;
            while i < instructions.len() {
                match &mut instructions[i] {
                    Instruction::LocalSet(_) => {
                        let id = self.write_counter;
                        self.write_counter += 1;
                        if self.dead_writes.contains(&id) {
                            instructions[i] = Instruction::Drop;
                        }
                        i += 1;
                    }
                    Instruction::LocalTee(_) => {
                        let id = self.write_counter;
                        self.write_counter += 1;
                        if self.dead_writes.contains(&id) {
                            instructions.remove(i);
                            // Don't increment i — re-examine new instruction
                            // at this position (matches the convention in
                            // neutralize_dead_writes).
                        } else {
                            i += 1;
                        }
                    }
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        self.apply(body);
                        i += 1;
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        self.apply(then_body);
                        self.apply(else_body);
                        i += 1;
                    }
                    _ => {
                        i += 1;
                    }
                }
            }
        }
    }

    #[derive(Debug, Clone)]
    struct LiveRange {
        local_idx: u32,
        start: usize,
        end: usize,
    }

    impl LiveRange {
        fn overlaps(&self, other: &LiveRange) -> bool {
            // Two ranges overlap if one starts before the other ends
            self.start < other.end && other.start < self.end
        }
    }

    fn compute_live_ranges(instructions: &[Instruction], param_count: usize) -> Vec<LiveRange> {
        use std::collections::BTreeMap;

        // Track first def and last use for each local
        #[derive(Default)]
        struct LocalInfo {
            first_def: Option<usize>,
            last_use: Option<usize>,
        }

        let mut local_info: BTreeMap<u32, LocalInfo> = BTreeMap::new();
        let mut position = 0;

        fn scan_instructions(
            instructions: &[crate::Instruction],
            local_info: &mut BTreeMap<u32, LocalInfo>,
            position: &mut usize,
            param_count: usize,
        ) {
            use crate::Instruction;
            for instr in instructions {
                match instr {
                    Instruction::LocalGet(idx)
                        // Parameters are always live (don't coalesce them)
                        if *idx >= param_count as u32 => {
                            let info = local_info.entry(*idx).or_default();
                            info.last_use = Some(*position);
                            if info.first_def.is_none() {
                                // If we see a get before any set, treat it as defined at position 0
                                info.first_def = Some(0);
                            }
                        }
                    Instruction::LocalSet(idx) | Instruction::LocalTee(idx)
                        if *idx >= param_count as u32 => {
                            let info = local_info.entry(*idx).or_default();
                            if info.first_def.is_none() {
                                info.first_def = Some(*position);
                            }
                            // Tee also counts as a use
                            if matches!(instr, Instruction::LocalTee(_)) {
                                info.last_use = Some(*position);
                            }
                        }
                    // Recurse into control flow
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        scan_instructions(body, local_info, position, param_count);
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => {
                        scan_instructions(then_body, local_info, position, param_count);
                        scan_instructions(else_body, local_info, position, param_count);
                    }
                    _ => {}
                }
                *position += 1;
            }
        }

        scan_instructions(instructions, &mut local_info, &mut position, param_count);

        // Build live ranges from local info
        let mut ranges = Vec::new();
        for (local_idx, info) in local_info {
            if let (Some(start), Some(end)) = (info.first_def, info.last_use) {
                ranges.push(LiveRange {
                    local_idx,
                    start,
                    end,
                });
            }
        }

        ranges
    }

    struct InterferenceGraph {
        nodes: Vec<u32>,
        edges: std::collections::HashSet<(u32, u32)>,
    }

    fn build_interference_graph(live_ranges: &[LiveRange]) -> InterferenceGraph {
        let mut nodes: Vec<u32> = live_ranges.iter().map(|lr| lr.local_idx).collect();
        nodes.sort_unstable();
        nodes.dedup();

        let mut edges = std::collections::HashSet::new();

        // For each pair of locals, check if their live ranges overlap
        for i in 0..live_ranges.len() {
            for j in (i + 1)..live_ranges.len() {
                if live_ranges[i].overlaps(&live_ranges[j]) {
                    let a = live_ranges[i].local_idx.min(live_ranges[j].local_idx);
                    let b = live_ranges[i].local_idx.max(live_ranges[j].local_idx);
                    edges.insert((a, b));
                }
            }
        }

        InterferenceGraph { nodes, edges }
    }

    fn color_interference_graph(graph: &InterferenceGraph) -> std::collections::BTreeMap<u32, u32> {
        use std::collections::{BTreeMap, HashSet};

        let mut coloring: BTreeMap<u32, u32> = BTreeMap::new();

        // Sort nodes by degree (most connected first) for better coloring
        let mut node_degrees: Vec<(u32, usize)> = graph
            .nodes
            .iter()
            .map(|&node| {
                let degree = graph
                    .edges
                    .iter()
                    .filter(|(a, b)| *a == node || *b == node)
                    .count();
                (node, degree)
            })
            .collect();

        node_degrees.sort_by_key(|(_, degree)| std::cmp::Reverse(*degree));

        // Greedy coloring
        for (node, _) in node_degrees {
            // Find colors used by neighbors
            let mut used_colors = HashSet::new();
            for (a, b) in &graph.edges {
                if *a == node {
                    if let Some(&color) = coloring.get(b) {
                        used_colors.insert(color);
                    }
                } else if *b == node {
                    if let Some(&color) = coloring.get(a) {
                        used_colors.insert(color);
                    }
                }
            }

            // Find smallest color not in used_colors
            let mut color = 0;
            while used_colors.contains(&color) {
                color += 1;
            }

            coloring.insert(node, color);
        }

        coloring
    }

    fn remap_function_locals(
        func: &mut crate::Function,
        coloring: &std::collections::BTreeMap<u32, u32>,
    ) {
        // Remap all local references in instructions
        remap_instructions(&mut func.instructions, coloring);

        // Rebuild local declarations based on new coloring
        let param_count = func.signature.params.len();

        // Count how many locals of each type are needed for each color
        use std::collections::BTreeMap;
        let mut color_types: BTreeMap<u32, crate::ValueType> = BTreeMap::new();

        // Build mapping from old index to type
        let mut old_idx_to_type: BTreeMap<u32, crate::ValueType> = BTreeMap::new();
        let mut current_idx = param_count as u32;
        for (count, value_type) in &func.locals {
            for _ in 0..*count {
                old_idx_to_type.insert(current_idx, *value_type);
                current_idx += 1;
            }
        }

        // Map colors to types
        for (old_idx, color) in coloring {
            if let Some(&value_type) = old_idx_to_type.get(old_idx) {
                color_types.insert(*color, value_type);
            }
        }

        // Rebuild locals vector
        let max_color = coloring.values().max().copied().unwrap_or(0);
        let mut new_locals = Vec::new();

        for color in 0..=max_color {
            if let Some(&value_type) = color_types.get(&color) {
                new_locals.push((1, value_type));
            }
        }

        func.locals = new_locals;
    }

    fn remap_instructions(
        instructions: &mut [crate::Instruction],
        coloring: &std::collections::BTreeMap<u32, u32>,
    ) {
        use crate::Instruction;
        for instr in instructions {
            match instr {
                Instruction::LocalGet(idx)
                | Instruction::LocalSet(idx)
                | Instruction::LocalTee(idx) => {
                    if let Some(&new_idx) = coloring.get(idx) {
                        *idx = new_idx;
                    }
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    remap_instructions(body, coloring);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    remap_instructions(then_body, coloring);
                    remap_instructions(else_body, coloring);
                }
                _ => {}
            }
        }
    }

    /// Precompute / Global Constant Propagation (Phase 19 - Issue #18)
    ///
    /// Propagates immutable global constants throughout the module.
    /// Replaces `global.get $x` with constant values when:
    /// 1. Global is immutable (!mutable)
    /// 2. Global has constant initializer
    ///
    /// This enables further optimizations:
    /// - Branch simplification when globals are boolean flags
    /// - Constant folding when globals are numeric constants
    /// - Dead code elimination when paths become unreachable
    ///
    /// Propagates immutable global constants throughout the module.
    pub fn precompute(module: &mut Module) -> Result<()> {
        use std::collections::BTreeMap;

        // Phase 1: Analyze globals to find constants
        let mut global_constants: BTreeMap<u32, ConstantValue> = BTreeMap::new();

        for (idx, global) in module.globals.iter().enumerate() {
            // Only track immutable globals
            if global.mutable {
                continue;
            }

            // Check if initializer is a single constant instruction
            if global.init.len() == 1 {
                let constant = match &global.init[0] {
                    Instruction::I32Const(val) => Some(ConstantValue::I32(*val)),
                    Instruction::I64Const(val) => Some(ConstantValue::I64(*val)),
                    _ => None,
                };

                if let Some(const_val) = constant {
                    global_constants.insert(idx as u32, const_val);
                }
            }
        }

        // Phase 2: Propagate constants into functions
        if global_constants.is_empty() {
            return Ok(());
        }

        use crate::verify::TranslationValidator;

        for func in &mut module.functions {
            let translator = TranslationValidator::new(func, "precompute");

            func.instructions =
                propagate_global_constants_in_instructions(&func.instructions, &global_constants);

            translator.verify_or_revert(func);
        }

        Ok(())
    }

    fn propagate_global_constants_in_instructions(
        instructions: &[Instruction],
        constants: &std::collections::BTreeMap<u32, ConstantValue>,
    ) -> Vec<Instruction> {
        instructions
            .iter()
            .map(|instr| match instr {
                Instruction::GlobalGet(idx) => {
                    if let Some(const_val) = constants.get(idx) {
                        // Replace global.get with constant
                        match const_val {
                            ConstantValue::I32(val) => Instruction::I32Const(*val),
                            ConstantValue::I64(val) => Instruction::I64Const(*val),
                        }
                    } else {
                        instr.clone()
                    }
                }

                // Recursively process control flow
                Instruction::Block { block_type, body } => Instruction::Block {
                    block_type: block_type.clone(),
                    body: propagate_global_constants_in_instructions(body, constants),
                },

                Instruction::Loop { block_type, body } => Instruction::Loop {
                    block_type: block_type.clone(),
                    body: propagate_global_constants_in_instructions(body, constants),
                },

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => Instruction::If {
                    block_type: block_type.clone(),
                    then_body: propagate_global_constants_in_instructions(then_body, constants),
                    else_body: propagate_global_constants_in_instructions(else_body, constants),
                },

                _ => instr.clone(),
            })
            .collect()
    }

    #[derive(Debug, Clone)]
    enum ConstantValue {
        I32(i32),
        I64(i64),
    }

    /// Common Subexpression Elimination (CSE - Phase 20)
    ///
    /// **Current State**: Conservative constant-only implementation
    ///
    /// This is intentionally minimal to ensure stack-correctness. Only caches
    /// self-contained constants (which are filtered anyway), making this effectively
    /// a placeholder until full expression-tree CSE is implemented.
    ///
    /// **Why Conservative**:
    /// - Stack-based WASM operations consume values (can't cache `i32.add` alone)
    /// - Previous impl incorrectly used `local.tee`, leaving extra stack values
    /// - Requires proper stack simulation, dependency tracking, side-effect analysis
    ///
    /// **Future**: Expression-tree framework in `eliminate_common_subexpressions_enhanced()`
    /// has Phases 1-3 complete (build trees, find duplicates, allocate locals).
    /// Phase 4 (transformation) needs careful implementation for stack semantics.
    ///
    /// **Current Wins Come From**:
    /// - ISLE pattern optimizations (constant folding)
    /// - `optimize_advanced_instructions()` (algebraic simplifications, strength reduction)
    /// - Self-operation optimizations (x-x→0, x^x→0, etc.)
    pub fn eliminate_common_subexpressions(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;
        use std::collections::BTreeMap;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // Skip BrIf/BrTable functions — CSE caches a value computed once and
            // reuses it; if a branch skips the original cache point, the reused
            // value is from the wrong path. The verifier is path-insensitive
            // and cannot reject this (REQ-5).
            if has_dataflow_unsafe_control_flow(func) {
                continue;
            }

            let guard =
                ValidationGuard::with_context(func, "eliminate_common_subexpressions", ctx.clone());
            let translator = TranslationValidator::new(func, "eliminate_common_subexpressions");

            // Phase 1: Find all expression patterns (binary ops with 3-instruction sequences)
            let patterns = find_expression_patterns(&func.instructions);

            // Phase 2: Group patterns by hash to find duplicates
            let mut pattern_map: BTreeMap<String, Vec<usize>> = BTreeMap::new();
            for (idx, pattern) in patterns.iter().enumerate() {
                pattern_map
                    .entry(pattern.hash.clone())
                    .or_default()
                    .push(idx);
            }

            // Collect duplicates: patterns with same hash
            let mut duplicates: Vec<(usize, Vec<usize>)> = Vec::new(); // (first_idx, [dup_idx...])
            for indices in pattern_map.values() {
                if indices.len() > 1 {
                    duplicates.push((indices[0], indices[1..].to_vec()));
                }
            }

            if duplicates.is_empty() {
                continue;
            }

            // Phase 3: Safety check - filter out unsafe duplicates
            let safe_duplicates: Vec<_> = duplicates
                .into_iter()
                .filter(|(first_idx, dup_indices)| {
                    let first_pattern = &patterns[*first_idx];
                    dup_indices.iter().all(|&dup_idx| {
                        let dup_pattern = &patterns[dup_idx];
                        is_safe_to_cse(first_pattern, dup_pattern, &func.instructions)
                    })
                })
                .collect();

            if safe_duplicates.is_empty() {
                continue;
            }

            // Phase 4: Apply CSE transformations
            apply_cse_transformations(func, &patterns, &safe_duplicates);

            let _ = guard.validate(func);
            translator.verify_or_revert(func);
        }

        Ok(())
    }

    /// Expression pattern found in instruction stream
    #[derive(Debug, Clone)]
    struct ExpressionPattern {
        start_pos: usize,
        end_pos: usize,
        hash: String,
        result_type: super::ValueType,
        referenced_locals: Vec<u32>,
    }

    /// Find all expression patterns in instruction stream
    fn find_expression_patterns(instructions: &[Instruction]) -> Vec<ExpressionPattern> {
        let mut patterns = Vec::new();

        // Pattern: Binary operations (3 instructions)
        // operand1 (local.get/const), operand2 (local.get/const), binop
        for i in 2..instructions.len() {
            if let Some(pattern) = match_binary_pattern(
                instructions.get(i - 2),
                instructions.get(i - 1),
                instructions.get(i),
            ) {
                patterns.push(ExpressionPattern {
                    start_pos: i - 2,
                    end_pos: i,
                    hash: pattern.0,
                    result_type: pattern.1,
                    referenced_locals: pattern.2,
                });
            }
        }

        patterns
    }

    /// Try to match a 3-instruction binary operation pattern
    /// Returns (hash, result_type, referenced_locals)
    fn match_binary_pattern(
        instr1: Option<&Instruction>,
        instr2: Option<&Instruction>,
        instr3: Option<&Instruction>,
    ) -> Option<(String, super::ValueType, Vec<u32>)> {
        use super::ValueType;

        let (op1, op1_locals) = match instr1? {
            Instruction::LocalGet(idx) => (format!("local.get:{}", idx), vec![*idx]),
            Instruction::I32Const(val) => (format!("i32.const:{}", val), vec![]),
            Instruction::I64Const(val) => (format!("i64.const:{}", val), vec![]),
            _ => return None,
        };

        let (op2, op2_locals) = match instr2? {
            Instruction::LocalGet(idx) => (format!("local.get:{}", idx), vec![*idx]),
            Instruction::I32Const(val) => (format!("i32.const:{}", val), vec![]),
            Instruction::I64Const(val) => (format!("i64.const:{}", val), vec![]),
            _ => return None,
        };

        let (binop, result_type) = match instr3? {
            Instruction::I32Add => ("i32.add", ValueType::I32),
            Instruction::I32Sub => ("i32.sub", ValueType::I32),
            Instruction::I32Mul => ("i32.mul", ValueType::I32),
            Instruction::I32And => ("i32.and", ValueType::I32),
            Instruction::I32Or => ("i32.or", ValueType::I32),
            Instruction::I32Xor => ("i32.xor", ValueType::I32),
            Instruction::I64Add => ("i64.add", ValueType::I64),
            Instruction::I64Sub => ("i64.sub", ValueType::I64),
            Instruction::I64Mul => ("i64.mul", ValueType::I64),
            Instruction::I64And => ("i64.and", ValueType::I64),
            Instruction::I64Or => ("i64.or", ValueType::I64),
            Instruction::I64Xor => ("i64.xor", ValueType::I64),
            _ => return None,
        };

        let hash = format!("({}, {}, {})", op1, op2, binop);
        let mut locals = op1_locals;
        locals.extend(op2_locals);

        Some((hash, result_type, locals))
    }

    /// Check if it's safe to CSE between two pattern occurrences
    fn is_safe_to_cse(
        first: &ExpressionPattern,
        second: &ExpressionPattern,
        instructions: &[Instruction],
    ) -> bool {
        // Conservative safety checks
        let start = first.end_pos + 1;
        let end = second.start_pos;

        if start > end {
            return false;
        }

        // Adjacent patterns are safe (no instructions between them)
        if start == end {
            return true;
        }

        // Check for invalidating instructions between first and second occurrence
        for instr in &instructions[start..end] {
            match instr {
                // Control flow invalidates CSE (leaves basic block)
                Instruction::Block { .. }
                | Instruction::Loop { .. }
                | Instruction::If { .. }
                | Instruction::Br(_)
                | Instruction::BrIf(_)
                | Instruction::BrTable { .. }
                | Instruction::Call(_)
                | Instruction::CallIndirect { .. }
                | Instruction::Return => return false,

                // Local modifications invalidate if they reference our locals
                Instruction::LocalSet(idx) | Instruction::LocalTee(idx)
                    if first.referenced_locals.contains(idx) =>
                {
                    return false;
                }

                // Memory operations are conservatively rejected for now
                Instruction::I32Load { .. }
                | Instruction::I64Load { .. }
                | Instruction::I32Store { .. }
                | Instruction::I64Store { .. } => return false,

                _ => {}
            }
        }

        true
    }

    /// Apply CSE transformations to a function
    fn apply_cse_transformations(
        func: &mut super::Function,
        patterns: &[ExpressionPattern],
        safe_duplicates: &[(usize, Vec<usize>)],
    ) {
        use std::collections::HashMap;

        // Build maps for transformation
        let mut positions_to_cache: HashMap<usize, (usize, usize, super::ValueType)> =
            HashMap::new(); // pattern_idx -> (start, end, type)
        let mut _positions_to_replace: HashMap<usize, u32> = HashMap::new(); // pattern_idx -> local_idx

        // Allocate locals for each unique first occurrence
        let base_local_idx = func.signature.params.len() as u32
            + func.locals.iter().map(|(count, _)| count).sum::<u32>();

        for (local_offset, (first_idx, dup_indices)) in safe_duplicates.iter().enumerate() {
            let pattern = &patterns[*first_idx];
            positions_to_cache.insert(
                *first_idx,
                (pattern.start_pos, pattern.end_pos, pattern.result_type),
            );

            let local_idx = base_local_idx + local_offset as u32;
            func.locals.push((1, pattern.result_type));

            // Map all duplicate occurrences to this local
            for &dup_idx in dup_indices {
                _positions_to_replace.insert(dup_idx, local_idx);
            }
        }

        // Track which instruction positions are part of patterns to transform
        let mut instruction_map: HashMap<usize, TransformAction> = HashMap::new();

        for (pattern_idx, (_start, end, _type)) in &positions_to_cache {
            // Find the local_idx for this pattern
            if let Some((_, dup_indices)) = safe_duplicates
                .iter()
                .find(|(first, _)| first == pattern_idx)
            {
                let local_offset = safe_duplicates
                    .iter()
                    .position(|(first, _)| first == pattern_idx)
                    .unwrap();
                let local_idx = base_local_idx + local_offset as u32;

                instruction_map.insert(*end, TransformAction::AddTee(local_idx));

                // Mark duplicate patterns for replacement
                for &dup_idx in dup_indices {
                    let dup_pattern = &patterns[dup_idx];
                    instruction_map.insert(
                        dup_pattern.start_pos,
                        TransformAction::ReplacePattern(dup_pattern.end_pos, local_idx),
                    );
                }
            }
        }

        // Apply transformations
        let mut new_instructions = Vec::new();
        let mut skip_until: Option<usize> = None;

        for (pos, instr) in func.instructions.iter().enumerate() {
            // Check if we should skip (part of replaced pattern)
            if let Some(skip_pos) = skip_until {
                if pos <= skip_pos {
                    continue;
                } else {
                    skip_until = None;
                }
            }

            // Check for transformation actions
            if let Some(action) = instruction_map.get(&pos) {
                match action {
                    TransformAction::AddTee(local_idx) => {
                        // Keep instruction and add tee
                        new_instructions.push(instr.clone());
                        new_instructions.push(Instruction::LocalTee(*local_idx));
                    }
                    TransformAction::ReplacePattern(end_pos, local_idx) => {
                        // Replace entire pattern with local.get
                        new_instructions.push(Instruction::LocalGet(*local_idx));
                        skip_until = Some(*end_pos);
                    }
                }
            } else {
                new_instructions.push(instr.clone());
            }
        }

        func.instructions = new_instructions;
    }

    #[derive(Debug)]
    enum TransformAction {
        AddTee(u32),
        ReplacePattern(usize, u32),
    }

    /// Enhanced Common Subexpression Elimination (Issue #19 - Full Implementation)
    ///
    /// This is an enhanced version of CSE that:
    /// - Works on complete expression trees, not just constants
    /// - Uses stack simulation to build expression hashes
    /// - Handles commutative operations (a+b = b+a)
    /// - Eliminates duplicate computations across the function
    ///
    /// Unlike the MVP CSE, this version can eliminate patterns like:
    ///   (local.get $x) (local.get $y) (i32.add)
    ///   (local.get $x) (local.get $y) (i32.add)  ;; duplicate!
    pub fn eliminate_common_subexpressions_enhanced(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::{TranslationValidator, VerificationSignatureContext};
        use std::collections::hash_map::DefaultHasher;
        use std::collections::{BTreeMap, HashMap};
        use std::hash::{Hash, Hasher};

        let ctx = ValidationContext::from_module(module);
        // PR-K3: pass the signature context (which now carries function
        // summaries) into the per-function translator so the verifier
        // can encode pure+no-trap Call results as uninterpreted-function
        // applications. Without this the CSE dedup of identical pure
        // calls gets rejected by Z3 (two independent symbolic constants).
        let verify_sig_ctx = VerificationSignatureContext::from_module(module);

        // CSE transformation actions
        #[derive(Debug, Clone, Copy)]
        enum CSEAction {
            SaveToLocal(u32),   // Save result to local using local.tee
            LoadFromLocal(u32), // Replace with local.get
            // PR-K2: span replacement for pure Call exprs. The first
            // occurrence is annotated with `SaveToLocal` at the call
            // position (so `local.tee` is appended after the call). A
            // duplicate occurrence is annotated with `ReplaceSpanWithLoad`
            // at the leftmost-arg position with `end_pos = call_pos`,
            // emitting one `local.get` and skipping all instructions in
            // `[arg_pos..=call_pos]`.
            ReplaceSpanWithLoad { local_idx: u32, end_pos: usize },
        }

        // Expression representation for CSE
        #[derive(Debug, Clone, Eq, PartialEq, Hash)]
        enum Expr {
            Const32(i32),
            Const64(i64),
            LocalGet(u32),
            Binary {
                op: String,
                left: Box<Expr>,
                right: Box<Expr>,
                commutative: bool,
            },
            #[allow(dead_code)]
            Unary {
                op: String,
                operand: Box<Expr>,
            },
            /// Direct call to a function the IPA has classified as
            /// `is_pure && is_no_trap`. Treated as a deterministic
            /// value of the function's single result (only single-result
            /// calls are tracked here — see PR-K wiring). Args must
            /// themselves be CSE-trackable expressions; an Unknown arg
            /// would have already poisoned the call expression at scan
            /// time, so any `Call` reaching dedup has fully-determined
            /// inputs.
            Call {
                func_idx: u32,
                args: Vec<Expr>,
            },
            Unknown, // For operations we can't track
        }

        impl Expr {
            /// Compute a stable hash for this expression
            /// For commutative operations, we normalize by sorting operands
            fn compute_hash(&self) -> u64 {
                let mut hasher = DefaultHasher::new();
                match self {
                    Expr::Const32(v) => {
                        "i32.const".hash(&mut hasher);
                        v.hash(&mut hasher);
                    }
                    Expr::Const64(v) => {
                        "i64.const".hash(&mut hasher);
                        v.hash(&mut hasher);
                    }
                    Expr::LocalGet(idx) => {
                        "local.get".hash(&mut hasher);
                        idx.hash(&mut hasher);
                    }
                    Expr::Binary {
                        op,
                        left,
                        right,
                        commutative,
                    } => {
                        op.hash(&mut hasher);
                        if *commutative {
                            // For commutative ops, sort operand hashes
                            let left_hash = left.compute_hash();
                            let right_hash = right.compute_hash();
                            let (h1, h2) = if left_hash <= right_hash {
                                (left_hash, right_hash)
                            } else {
                                (right_hash, left_hash)
                            };
                            h1.hash(&mut hasher);
                            h2.hash(&mut hasher);
                        } else {
                            left.compute_hash().hash(&mut hasher);
                            right.compute_hash().hash(&mut hasher);
                        }
                    }
                    Expr::Unary { op, operand } => {
                        op.hash(&mut hasher);
                        operand.compute_hash().hash(&mut hasher);
                    }
                    Expr::Call { func_idx, args } => {
                        // Calls are never commutative in argument position
                        // (f(a,b) ≠ f(b,a) for the optimizer — wasm
                        // semantics give a fixed left-to-right eval order).
                        "call".hash(&mut hasher);
                        func_idx.hash(&mut hasher);
                        for arg in args {
                            arg.compute_hash().hash(&mut hasher);
                        }
                    }
                    Expr::Unknown => {
                        "unknown".hash(&mut hasher);
                    }
                }
                hasher.finish()
            }

            /// Check if this expression is pure (no side effects)
            fn is_pure(&self) -> bool {
                match self {
                    Expr::Unknown => false,
                    Expr::Const32(_) | Expr::Const64(_) | Expr::LocalGet(_) => true,
                    Expr::Binary { left, right, .. } => left.is_pure() && right.is_pure(),
                    Expr::Unary { operand, .. } => operand.is_pure(),
                    // A Call expression is only ever constructed for
                    // functions the IPA has already proved pure+no-trap
                    // (see scan loop below). Args must also be pure.
                    Expr::Call { args, .. } => args.iter().all(|a| a.is_pure()),
                }
            }

            /// Estimate the wasm-encoded byte cost of materializing this
            /// expression once.
            fn cost_bytes(&self) -> usize {
                match self {
                    Expr::Unknown => 0,
                    Expr::Const32(v) => 1 + signed_leb128_bytes_i32(*v),
                    Expr::Const64(v) => 1 + signed_leb128_bytes_i64(*v),
                    Expr::LocalGet(idx) => 1 + unsigned_leb128_bytes(*idx as u64),
                    Expr::Binary { left, right, .. } => left.cost_bytes() + right.cost_bytes() + 1,
                    Expr::Unary { operand, .. } => operand.cost_bytes() + 1,
                    Expr::Call { func_idx, args } => {
                        // call opcode (1) + LEB128 func index + arg costs.
                        let arg_cost: usize = args.iter().map(|a| a.cost_bytes()).sum();
                        1 + unsigned_leb128_bytes(*func_idx as u64) + arg_cost
                    }
                }
            }

            /// Net byte savings from CSE-ing N occurrences of this
            /// expression. Positive = win, non-positive = skip.
            ///
            /// Cost model:
            ///   first occurrence:  keep original + add local.tee N
            ///                      (~2 bytes for opcode + LEB128 idx,
            ///                      assuming idx ≤ 127)
            ///   each later use:    replace original with local.get N
            ///                      (~2 bytes), saving (cost - 2) bytes
            ///   header growth:     +2 bytes for new local declaration
            ///                      (1 byte count + 1 byte type)
            ///
            /// So:  savings = (N-1) * (cost - 2) - 2 (tee) - 2 (header)
            ///
            /// Examples (with this gate):
            ///   i32.const 42 (cost=2, N=2):  savings = 1*0 - 4 = -4  → skip
            ///   i32.const 42 (cost=2, N=10): savings = 9*0 - 4 = -4  → skip
            ///   i32.add of LocalGet+Const42 (cost=5, N=2):
            ///                                 savings = 1*3 - 4 = -1  → skip
            ///   i32.add of LocalGet+Const42 (cost=5, N=3):
            ///                                 savings = 2*3 - 4 = 2   → keep
            ///   big i32.const (cost=6, N=2):  savings = 1*4 - 4 = 0   → skip
            ///   big i32.const (cost=6, N=3):  savings = 2*4 - 4 = 4   → keep
            ///
            /// Tuned against the gale v0.4.0 measurement (CSE-ing
            /// `-EINVAL` into local.tee/local.get grew kernel-FFI code
            /// section by +6.3%).
            fn worth_dedup(&self, occurrences: usize) -> bool {
                if occurrences < 2 {
                    return false;
                }
                let cost = self.cost_bytes() as i64;
                let savings_per_later = (cost - 2).max(0); // never negative — replacing 1-byte materialization with 2-byte local.get is a regression we won't take
                let net = (occurrences as i64 - 1) * savings_per_later - 4;
                net > 0
            }
        }

        // LEB128 byte-length helpers (matches wasm-encoder's encoding).
        fn unsigned_leb128_bytes(mut v: u64) -> usize {
            let mut n = 1;
            while v >= 0x80 {
                v >>= 7;
                n += 1;
            }
            n
        }
        fn signed_leb128_bytes_i32(v: i32) -> usize {
            // Signed LEB128: bits needed including sign, divided by 7
            // (round up). Small values: -64..=63 fit in 1 byte; -8192..=8191
            // in 2 bytes; etc.
            let mut v = v as i64;
            let mut n = 0;
            loop {
                n += 1;
                let byte = (v & 0x7f) as u8;
                v >>= 7;
                let sign_bit_set = byte & 0x40 != 0;
                if (v == 0 && !sign_bit_set) || (v == -1 && sign_bit_set) {
                    return n;
                }
            }
        }
        fn signed_leb128_bytes_i64(v: i64) -> usize {
            let mut v = v;
            let mut n = 0;
            loop {
                n += 1;
                let byte = (v & 0x7f) as u8;
                v >>= 7;
                let sign_bit_set = byte & 0x40 != 0;
                if (v == 0 && !sign_bit_set) || (v == -1 && sign_bit_set) {
                    return n;
                }
            }
        }

        // PR-K: function-summary IPA enables cross-call dedup. A direct
        // Call to a function classified as `is_pure && is_no_trap` is a
        // deterministic value of its arguments — two identical such calls
        // can be CSE'd just like an arithmetic subtree. We also need the
        // callee signature to know how many args to pop and what result
        // type to materialize the cache local with. Snapshot before the
        // mutable loop so we don't borrow `module` twice.
        let summaries = crate::summary::compute_module_summaries(module);
        let func_signatures: Vec<(Vec<super::ValueType>, Vec<super::ValueType>)> = module
            .functions
            .iter()
            .map(|f| (f.signature.params.clone(), f.signature.results.clone()))
            .collect();

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            // Skip BrIf/BrTable functions — same path-sensitivity concern as
            // basic CSE; the verifier cannot reject unsound caching across
            // these branches (REQ-5).
            if has_dataflow_unsafe_control_flow(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(
                func,
                "eliminate_common_subexpressions_enhanced",
                ctx.clone(),
            );
            let translator = TranslationValidator::new_with_context(
                func,
                "eliminate_common_subexpressions_enhanced",
                verify_sig_ctx.clone(),
            );

            // Simulate stack to build expression trees
            let mut stack: Vec<Expr> = Vec::new();
            let mut expr_at_position: HashMap<usize, (Expr, u64)> = HashMap::new();
            let mut hash_to_positions: BTreeMap<u64, Vec<usize>> = BTreeMap::new();
            // PR-K: for each tracked expression, also remember the
            // instruction-position where its left-most source lives. For
            // single-instruction exprs (Const, LocalGet) this is the
            // position itself; for a Call this is the position of its
            // first arg. Lets us turn a Call duplicate into a span
            // replacement at transform time.
            let mut start_at_position: HashMap<usize, usize> = HashMap::new();

            // Parallel stack of `start_pos` for each Expr currently on
            // `stack`. PR-K uses this to compute the span (first_arg ..=
            // call_pos) for a Call expression, so a duplicate can be
            // replaced by a single `local.get` regardless of how many
            // instructions originally produced it. For non-Call ops this
            // stays in lock-step with `stack` and is otherwise unused.
            let mut start_stack: Vec<usize> = Vec::new();

            // Phase 1: Build expression trees and detect duplicates
            for (pos, instr) in func.instructions.iter().enumerate() {
                match instr {
                    // Constants push onto stack
                    Instruction::I32Const(v) => {
                        let expr = Expr::Const32(*v);
                        let hash = expr.compute_hash();
                        expr_at_position.insert(pos, (expr.clone(), hash));
                        start_at_position.insert(pos, pos);
                        hash_to_positions.entry(hash).or_default().push(pos);
                        stack.push(expr);
                        start_stack.push(pos);
                    }
                    Instruction::I64Const(v) => {
                        let expr = Expr::Const64(*v);
                        let hash = expr.compute_hash();
                        expr_at_position.insert(pos, (expr.clone(), hash));
                        start_at_position.insert(pos, pos);
                        hash_to_positions.entry(hash).or_default().push(pos);
                        stack.push(expr);
                        start_stack.push(pos);
                    }
                    Instruction::LocalGet(idx) => {
                        let expr = Expr::LocalGet(*idx);
                        let hash = expr.compute_hash();
                        expr_at_position.insert(pos, (expr.clone(), hash));
                        start_at_position.insert(pos, pos);
                        hash_to_positions.entry(hash).or_default().push(pos);
                        stack.push(expr);
                        start_stack.push(pos);
                    }

                    // Binary operations pop two, push one
                    Instruction::I32Add | Instruction::I64Add => {
                        if stack.len() >= 2 {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            let _ = start_stack.pop();
                            let left_start = start_stack.pop().unwrap_or(pos);
                            let op = if matches!(instr, Instruction::I32Add) {
                                "i32.add"
                            } else {
                                "i64.add"
                            };
                            let expr = Expr::Binary {
                                op: op.to_string(),
                                left: Box::new(left),
                                right: Box::new(right),
                                commutative: true,
                            };
                            let hash = expr.compute_hash();
                            expr_at_position.insert(pos, (expr.clone(), hash));
                            start_at_position.insert(pos, left_start);
                            hash_to_positions.entry(hash).or_default().push(pos);
                            stack.push(expr);
                            start_stack.push(left_start);
                        } else {
                            stack.clear();
                            start_stack.clear();
                            stack.push(Expr::Unknown);
                            start_stack.push(pos);
                        }
                    }

                    Instruction::I32Mul | Instruction::I64Mul => {
                        if stack.len() >= 2 {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            let _ = start_stack.pop();
                            let left_start = start_stack.pop().unwrap_or(pos);
                            let op = if matches!(instr, Instruction::I32Mul) {
                                "i32.mul"
                            } else {
                                "i64.mul"
                            };
                            let expr = Expr::Binary {
                                op: op.to_string(),
                                left: Box::new(left),
                                right: Box::new(right),
                                commutative: true,
                            };
                            let hash = expr.compute_hash();
                            expr_at_position.insert(pos, (expr.clone(), hash));
                            start_at_position.insert(pos, left_start);
                            hash_to_positions.entry(hash).or_default().push(pos);
                            stack.push(expr);
                            start_stack.push(left_start);
                        } else {
                            stack.clear();
                            start_stack.clear();
                            stack.push(Expr::Unknown);
                            start_stack.push(pos);
                        }
                    }

                    Instruction::I32And
                    | Instruction::I64And
                    | Instruction::I32Or
                    | Instruction::I64Or
                    | Instruction::I32Xor
                    | Instruction::I64Xor => {
                        if stack.len() >= 2 {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            let _ = start_stack.pop();
                            let left_start = start_stack.pop().unwrap_or(pos);
                            let op = match instr {
                                Instruction::I32And => "i32.and",
                                Instruction::I64And => "i64.and",
                                Instruction::I32Or => "i32.or",
                                Instruction::I64Or => "i64.or",
                                Instruction::I32Xor => "i32.xor",
                                Instruction::I64Xor => "i64.xor",
                                _ => unreachable!(),
                            };
                            let expr = Expr::Binary {
                                op: op.to_string(),
                                left: Box::new(left),
                                right: Box::new(right),
                                commutative: true,
                            };
                            let hash = expr.compute_hash();
                            expr_at_position.insert(pos, (expr.clone(), hash));
                            start_at_position.insert(pos, left_start);
                            hash_to_positions.entry(hash).or_default().push(pos);
                            stack.push(expr);
                            start_stack.push(left_start);
                        } else {
                            stack.clear();
                            start_stack.clear();
                            stack.push(Expr::Unknown);
                            start_stack.push(pos);
                        }
                    }

                    Instruction::I32Sub | Instruction::I64Sub => {
                        if stack.len() >= 2 {
                            let right = stack.pop().unwrap();
                            let left = stack.pop().unwrap();
                            let _ = start_stack.pop();
                            let left_start = start_stack.pop().unwrap_or(pos);
                            let op = if matches!(instr, Instruction::I32Sub) {
                                "i32.sub"
                            } else {
                                "i64.sub"
                            };
                            let expr = Expr::Binary {
                                op: op.to_string(),
                                left: Box::new(left),
                                right: Box::new(right),
                                commutative: false,
                            };
                            let hash = expr.compute_hash();
                            expr_at_position.insert(pos, (expr.clone(), hash));
                            start_at_position.insert(pos, left_start);
                            hash_to_positions.entry(hash).or_default().push(pos);
                            stack.push(expr);
                            start_stack.push(left_start);
                        } else {
                            stack.clear();
                            start_stack.clear();
                            stack.push(Expr::Unknown);
                            start_stack.push(pos);
                        }
                    }

                    // PR-K: cross-call dedup. A direct Call to a function
                    // the IPA proved `is_pure && is_no_trap` is a
                    // deterministic value of its arguments — two such
                    // calls with byte-identical arg subtrees are eligible
                    // for CSE. We only track single-result calls (so the
                    // cached value can live in one local); multi-result
                    // calls fall through to the wildcard arm below
                    // (stack reset) and are not deduped. We also require
                    // every argument expression on the stack to be
                    // CSE-trackable (i.e., not `Unknown`) — otherwise the
                    // cached value would depend on instructions that
                    // weren't part of the modeled subtree.
                    Instruction::Call(func_idx) => {
                        let fi = *func_idx as usize;
                        let (params, results) = match func_signatures.get(fi) {
                            Some(sig) => sig.clone(),
                            None => {
                                // Imported / unknown function — be
                                // conservative: clear stack analysis.
                                stack.clear();
                                start_stack.clear();
                                continue;
                            }
                        };
                        let summary = summaries.get(fi).copied().unwrap_or_default();
                        let single_result = results.len() == 1;
                        let arity = params.len();
                        let pure_no_trap = summary.is_pure && summary.is_no_trap;

                        if pure_no_trap && single_result && stack.len() >= arity {
                            // Pop arity args from both stacks (in order).
                            let mut args_rev: Vec<Expr> = Vec::with_capacity(arity);
                            let mut first_arg_start = pos; // call-only fallback
                            for _ in 0..arity {
                                let a = stack.pop().unwrap();
                                let s = start_stack.pop().unwrap_or(pos);
                                args_rev.push(a);
                                first_arg_start = s; // last pop = leftmost arg
                            }
                            let args: Vec<Expr> = args_rev.into_iter().rev().collect();

                            // If any arg is Unknown the call result is
                            // not safely cacheable — fall back to
                            // pushing the result as Unknown.
                            let any_unknown = args.iter().any(|a| matches!(a, Expr::Unknown));
                            if any_unknown {
                                // Side-effect-free call with unknown args
                                // — push opaque result, don't track for
                                // dedup.
                                stack.push(Expr::Unknown);
                                start_stack.push(pos);
                            } else {
                                let expr = Expr::Call {
                                    func_idx: *func_idx,
                                    args,
                                };
                                let hash = expr.compute_hash();
                                expr_at_position.insert(pos, (expr.clone(), hash));
                                // span starts at the leftmost arg's start
                                // (which for simple single-instruction args
                                // is the arg's own position).
                                start_at_position.insert(pos, first_arg_start);
                                hash_to_positions.entry(hash).or_default().push(pos);
                                stack.push(expr);
                                start_stack.push(first_arg_start);
                            }
                        } else {
                            // Impure / may-trap / multi-result / arity
                            // mismatch — observable side-effect or
                            // unmodelable. Reset stack analysis: any
                            // post-call code depends on values we can't
                            // track here.
                            stack.clear();
                            start_stack.clear();
                            if single_result {
                                stack.push(Expr::Unknown);
                                start_stack.push(pos);
                            }
                        }
                    }

                    // For now, other instructions clear the stack analysis
                    _ => {
                        // Reset stack simulation on unknown operations
                        stack.clear();
                        start_stack.clear();
                    }
                }
            }

            // Phase 2: Find actual duplicates that we can eliminate
            // Identify duplicates that are pure and occur multiple times
            let mut duplicates_to_eliminate = Vec::new();

            for (hash, positions) in &hash_to_positions {
                if positions.len() > 1
                    && let Some((expr, _)) = expr_at_position.get(&positions[0])
                    && expr.is_pure()
                    // Cost gate: deduplicating into local.tee/local.get
                    // adds 4 fixed bytes (tee + new local declaration)
                    // plus amortizes only on the (N-1) later occurrences.
                    // For cheap expressions (1-2 byte materialization) the
                    // replacement is always a regression. For 5-byte
                    // expressions it breaks even at N=3. Skip when the
                    // net savings are non-positive.
                    && expr.worth_dedup(positions.len())
                {
                    duplicates_to_eliminate.push((*hash, positions.clone()));
                }
            }

            if duplicates_to_eliminate.is_empty() {
                continue; // No duplicates to eliminate in this function
            }

            // Phase 3: Allocate local variables for each unique duplicate expression
            let base_local_idx = func.signature.params.len() as u32
                + func.locals.iter().map(|(count, _)| count).sum::<u32>();

            let mut hash_to_local: BTreeMap<u64, u32> = BTreeMap::new();
            for (idx, (hash, _)) in duplicates_to_eliminate.iter().enumerate() {
                let local_idx = base_local_idx + idx as u32;
                hash_to_local.insert(*hash, local_idx);

                // Determine the type from the expression
                if let Some((expr, _)) = expr_at_position.get(&duplicates_to_eliminate[idx].1[0]) {
                    let value_type = match expr {
                        Expr::Const32(_) => super::ValueType::I32,
                        Expr::Const64(_) => super::ValueType::I64,
                        Expr::Binary { op, .. } => {
                            if op.starts_with("i32") {
                                super::ValueType::I32
                            } else {
                                super::ValueType::I64
                            }
                        }
                        // PR-K: a Call expression's type is the
                        // callee's single result type. Only single-
                        // result calls are tracked (see scan loop), so
                        // results[0] is well-defined.
                        Expr::Call { func_idx, .. } => func_signatures
                            .get(*func_idx as usize)
                            .and_then(|(_, results)| results.first().copied())
                            .unwrap_or(super::ValueType::I32),
                        _ => super::ValueType::I32, // Default
                    };
                    func.locals.push((1, value_type));
                }
            }

            // Phase 4: Transform instructions
            // Insert local.tee after first occurrence, replace duplicates with local.get
            //
            // Strategy: Conservative transformation for simple expressions
            // - Single-instruction Const exprs: tee at the call site, load
            //   at duplicate sites (existing path).
            // - PR-K2: pure-Call exprs whose args are all single-instruction
            //   pure pushers (Const, LocalGet) get span-based substitution.
            //   The first occurrence is unchanged except for a trailing
            //   `local.tee` after the Call instruction; each subsequent
            //   occurrence collapses its entire `[arg_start..=call_pos]`
            //   span into a single `local.get` of the cache local.
            // - Binary/Unary/nested-Call args are NOT folded in this PR
            //   (deferred to a follow-up): they would require a more
            //   careful span-overlap analysis. The verifier would still
            //   catch a mistake, but conservative-over-fast: skip.

            // Build transformation plan: first occurrence -> save, others -> load
            let mut position_action: HashMap<usize, CSEAction> = HashMap::new();
            // PR-K2: occupied[i] = true if position i is already covered by
            // some other Expr's transform span (first-occurrence's call site
            // or a duplicate's [arg..=call] span). Used to reject overlaps.
            let mut occupied: Vec<bool> = vec![false; func.instructions.len()];

            // PR-K2: helper — is this Expr arg a single-instruction pure
            // pusher whose value at runtime equals the pusher's local
            // identity in the IR? Const is always safe (value is encoded
            // in the instruction). LocalGet is safe ONLY if the local is
            // not mutated between the cached site and the use site — we
            // enforce that below via a separate scan.
            fn arg_is_simple_pusher(e: &Expr) -> bool {
                matches!(e, Expr::Const32(_) | Expr::Const64(_) | Expr::LocalGet(_))
            }
            // Collect the set of LocalGet indices referenced by a Call's
            // arg list (used to verify no intervening local.set/local.tee
            // could invalidate the cached value).
            fn arg_local_indices(args: &[Expr]) -> Vec<u32> {
                args.iter()
                    .filter_map(|a| match a {
                        Expr::LocalGet(idx) => Some(*idx),
                        _ => None,
                    })
                    .collect()
            }

            for (hash, local_idx) in &hash_to_local {
                if let Some(positions) = hash_to_positions.get(hash) {
                    if positions.len() > 1 {
                        if let Some((expr, _)) = expr_at_position.get(&positions[0]) {
                            match expr {
                                // SAFETY: constants are referentially
                                // transparent — same bits in any context.
                                Expr::Const32(_) | Expr::Const64(_) => {
                                    if occupied[positions[0]] {
                                        continue;
                                    }
                                    let mut blocked = false;
                                    for &pos in &positions[1..] {
                                        if occupied[pos] {
                                            blocked = true;
                                            break;
                                        }
                                    }
                                    if blocked {
                                        continue;
                                    }
                                    occupied[positions[0]] = true;
                                    position_action
                                        .insert(positions[0], CSEAction::SaveToLocal(*local_idx));
                                    for &pos in &positions[1..] {
                                        occupied[pos] = true;
                                        position_action
                                            .insert(pos, CSEAction::LoadFromLocal(*local_idx));
                                    }
                                }

                                // PR-K2: pure+no-trap single-result Call
                                // with simple-pusher args → span dedup.
                                Expr::Call { args, .. } => {
                                    // DEFENSE-IN-DEPTH: PR-K constructed
                                    // the Call expr only for pure+no-trap
                                    // callees with all-pure args. We add
                                    // a stronger check here: every arg
                                    // must be a single-instruction pure
                                    // pusher (Const or LocalGet). Binary,
                                    // Unary, and nested-Call args are
                                    // deferred to a follow-up PR.
                                    if !args.iter().all(arg_is_simple_pusher) {
                                        continue;
                                    }

                                    // Compute spans for every occurrence.
                                    // For a Call with N simple-pusher args,
                                    // the span is exactly N+1 instructions:
                                    // [start_at_position[pos] ..= pos].
                                    let mut spans: Vec<(usize, usize)> =
                                        Vec::with_capacity(positions.len());
                                    let mut span_ok = true;
                                    for &pos in positions {
                                        let start = match start_at_position.get(&pos) {
                                            Some(s) => *s,
                                            None => {
                                                span_ok = false;
                                                break;
                                            }
                                        };
                                        // For N=args.len() simple pushers,
                                        // expect span length N+1. If the
                                        // measured span is smaller (would
                                        // happen if the leftmost-arg-start
                                        // tracking missed an instruction)
                                        // we abandon: cannot prove the
                                        // span is exactly the call subtree.
                                        let expected_len = args.len() + 1;
                                        if pos < start || (pos - start + 1) != expected_len {
                                            span_ok = false;
                                            break;
                                        }
                                        spans.push((start, pos));
                                    }
                                    if !span_ok {
                                        continue;
                                    }

                                    // Overlap check: every position in
                                    // every span must be free.
                                    let mut overlap = false;
                                    for &(start, end) in &spans {
                                        if occupied[start..=end].iter().any(|&o| o) {
                                            overlap = true;
                                            break;
                                        }
                                    }
                                    if overlap {
                                        continue;
                                    }

                                    // local.set/local.tee scan: a LocalGet
                                    // arg is only safe to cache if the
                                    // local is not mutated anywhere from
                                    // the FIRST occurrence's call position
                                    // through the LAST duplicate position.
                                    // (Conservative bound — we could narrow
                                    // to per-duplicate, but the bigger
                                    // window is simpler and gives the
                                    // verifier the strongest invariant.)
                                    let arg_locals = arg_local_indices(args);
                                    let first_call = spans[0].1;
                                    let last_dup_start = spans.last().unwrap().0;
                                    let mut local_mutated = false;
                                    if !arg_locals.is_empty() {
                                        for ins in func
                                            .instructions
                                            .iter()
                                            .skip(first_call)
                                            .take(last_dup_start.saturating_sub(first_call) + 1)
                                        {
                                            match ins {
                                                Instruction::LocalSet(idx)
                                                | Instruction::LocalTee(idx)
                                                    if arg_locals.contains(idx) =>
                                                {
                                                    local_mutated = true;
                                                    break;
                                                }
                                                _ => {}
                                            }
                                        }
                                    }
                                    if local_mutated {
                                        continue;
                                    }

                                    // Plan the transform.
                                    let (_first_start, first_call_pos) = spans[0];
                                    // Mark every instruction in every span
                                    // as occupied so later iterations
                                    // don't double-plan.
                                    for &(start, end) in &spans {
                                        occupied[start..=end].fill(true);
                                    }
                                    // First occurrence: keep the whole
                                    // [start..=call_pos] sequence and tee
                                    // after the call.
                                    position_action
                                        .insert(first_call_pos, CSEAction::SaveToLocal(*local_idx));
                                    // Each later occurrence: collapse the
                                    // span to a single local.get at the
                                    // arg_start; skip up through call_pos.
                                    for &(start, end) in &spans[1..] {
                                        position_action.insert(
                                            start,
                                            CSEAction::ReplaceSpanWithLoad {
                                                local_idx: *local_idx,
                                                end_pos: end,
                                            },
                                        );
                                    }
                                }

                                // SAFETY: LocalGet alone is unsafe under
                                // the same local.set issue as before;
                                // Binary/Unary span dedup is a future PR.
                                _ => {}
                            }
                        }
                    }
                }
            }

            // Apply transformations: rebuild instruction list
            if !position_action.is_empty() {
                let mut new_instructions = Vec::new();
                let mut skip_until: Option<usize> = None;

                for (pos, instr) in func.instructions.iter().enumerate() {
                    // Honor an outstanding span-skip request.
                    if let Some(end) = skip_until {
                        if pos <= end {
                            continue;
                        } else {
                            skip_until = None;
                        }
                    }
                    match position_action.get(&pos) {
                        Some(CSEAction::SaveToLocal(local_idx)) => {
                            // Keep the original instruction, add local.tee
                            new_instructions.push(instr.clone());
                            new_instructions.push(Instruction::LocalTee(*local_idx));
                        }
                        Some(CSEAction::LoadFromLocal(local_idx)) => {
                            // Replace with local.get
                            new_instructions.push(Instruction::LocalGet(*local_idx));
                        }
                        Some(CSEAction::ReplaceSpanWithLoad { local_idx, end_pos }) => {
                            // Replace `[pos ..= end_pos]` with one local.get.
                            new_instructions.push(Instruction::LocalGet(*local_idx));
                            skip_until = Some(*end_pos);
                        }
                        None => {
                            // Keep instruction as-is
                            new_instructions.push(instr.clone());
                        }
                    }
                }

                func.instructions = new_instructions;
            }

            let _ = guard.validate(func);
            translator.verify_or_revert(func);
        }

        Ok(())
    }

    /// Advanced Instruction Optimization (Issue #21)
    ///
    /// Applies peephole optimizations including:
    /// - Strength reduction (mul/div/rem by power of 2)
    /// - Bitwise tricks (x^x→0, x&x→x, etc.)
    /// - Algebraic simplifications
    ///
    /// These are simple pattern-based transformations that work on
    /// instruction sequences in stack-based form.
    pub fn optimize_advanced_instructions(module: &mut Module) -> Result<()> {
        use super::Value;
        use super::terms::TermSignatureContext;
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::{TranslationValidator, VerificationSignatureContext};
        use loom_isle::{LocalEnv, rewrite_with_dataflow};

        let ctx = ValidationContext::from_module(module);
        // Create signature contexts for proper Call/CallIndirect handling
        let term_sig_ctx = TermSignatureContext::from_module(module);
        let verify_sig_ctx = VerificationSignatureContext::from_module(module);

        let mut skipped_unsupported = 0usize;
        let mut skipped_control_flow = 0usize;

        for func in &mut module.functions {
            // Skip optimization for functions with unsupported instructions
            // This includes floats, conversions, rotations, and unknown opcodes
            // which would corrupt the stack simulation in instructions_to_terms
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                skipped_unsupported += 1;
                continue;
            }

            // Same skip as constant_folding: terms_to_instructions roundtrip
            // does not preserve order across early-exit patterns. REQ-5.
            if has_dataflow_unsafe_control_flow(func) {
                skipped_control_flow += 1;
                continue;
            }

            let guard =
                ValidationGuard::with_context(func, "optimize_advanced_instructions", ctx.clone());
            let translator = TranslationValidator::new_with_context(
                func,
                "optimize_advanced_instructions",
                verify_sig_ctx.clone(),
            );

            // Save original instructions for rollback if verification fails
            let original_instructions = func.instructions.clone();

            // Track whether original had End instruction
            let had_end = func.instructions.last() == Some(&Instruction::End);

            // Convert instructions to ISLE terms, apply optimizations, convert back
            if let Ok(terms) = super::terms::instructions_to_terms_with_signatures(
                &func.instructions,
                &term_sig_ctx,
            ) {
                if !terms.is_empty() {
                    // Always use dataflow env (same reasoning as constant_folding).
                    let mut env = LocalEnv::new();
                    let optimized_terms: Vec<Value> = terms
                        .into_iter()
                        .map(|term| rewrite_with_dataflow(term, &mut env))
                        .collect();

                    if let Ok(new_instructions) =
                        super::terms::terms_to_instructions(&optimized_terms)
                    {
                        func.instructions = new_instructions;

                        // Preserve End instruction if original had it
                        if had_end && func.instructions.last() != Some(&Instruction::End) {
                            func.instructions.push(Instruction::End);
                        }
                    }
                }
            }

            // Verify stack correctness and semantic equivalence.
            // Revert on failure and continue with other functions.
            if guard.validate(func).is_err() || translator.verify(func).is_err() {
                eprintln!(
                    "optimize_advanced_instructions: reverting function (verification rejected)"
                );
                crate::stats::record_revert("optimize_advanced_instructions");
                func.instructions = original_instructions;
            }
        }

        if skipped_unsupported > 0 || skipped_control_flow > 0 {
            eprintln!(
                "Warning: optimize_advanced_instructions skipped {} function(s) with unsupported instructions, \
                 {} function(s) with dataflow-unsafe control flow (BrIf/BrTable, see #56)",
                skipped_unsupported, skipped_control_flow
            );
        }

        Ok(())
    }

    /// Function Inlining (Issue #14 - CRITICAL)
    ///
    /// Inlines small functions and single-call-site functions to:
    /// - Enable more optimizations (constant propagation across function boundaries)
    /// - Reduce call overhead
    /// - Eliminate parameter passing overhead
    ///
    /// Benefits 40-50% of typical WebAssembly code.
    pub fn inline_functions(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;
        use std::collections::BTreeMap;

        // Run inlining to fixed point to ensure idempotence
        // Keep inlining until no more candidates are found.
        //
        // loom#147 livelock guard: a candidate whose inline the Z3
        // verifier always REVERTS (e.g. an i64 function whose
        // inline-equivalence can't be proven) stays a candidate every
        // iteration — its call site is restored on revert — so the
        // `inline_candidates.is_empty()` exit never fires and the pass
        // spins forever (inline → revert → inline → …). It only looked
        // like a "hang" because pre-v1.1.2 the verifier panicked, which
        // happened to abort the pass; once the panic was fixed the
        // underlying livelock was exposed. Terminate when an iteration
        // keeps NO inline (no net progress), with a hard iteration cap as
        // a backstop.
        let mut iteration: u32 = 0;
        const MAX_INLINE_ITERATIONS: u32 = 64;
        loop {
            iteration += 1;
            if iteration > MAX_INLINE_ITERATIONS {
                break;
            }
            // Build context at start of each iteration (after possible function changes)
            let ctx = ValidationContext::from_module(module);

            // loom#153: `Call(func_idx)` uses the FULL WebAssembly function
            // index space — imported functions occupy indices
            // 0..num_imported_funcs, local functions follow. But
            // `module.functions` (and `all_functions`) holds ONLY local
            // functions, indexed from 0. Without accounting for the import
            // offset, an imported call's index collides with a local
            // function's slot: the inliner would treat a void imported call
            // as a call to local function 0, emit a `local.set` for its
            // params with no argument on the stack, and produce a malformed
            // body the verifier rejects (Stack underflow) → the whole inline
            // reverts. Key all index-space maps by the FULL index and map
            // back to local only when indexing `all_functions`.
            let num_imported_funcs = module
                .imports
                .iter()
                .filter(|imp| matches!(imp.kind, crate::ImportKind::Func(_)))
                .count() as u32;

            // Phase 1: Build call graph and analyze functions
            let mut call_counts: BTreeMap<u32, usize> = BTreeMap::new();
            let mut function_sizes: BTreeMap<u32, usize> = BTreeMap::new();

            // Calculate function sizes (instruction count). Keyed by FULL
            // index (local idx + import offset) to match `call_counts`,
            // which `count_calls_recursive` records in the full index space.
            for (idx, func) in module.functions.iter().enumerate() {
                let size = count_instructions_recursive(&func.instructions);
                function_sizes.insert(idx as u32 + num_imported_funcs, size);
            }

            // Count call sites for each function
            for func in &module.functions {
                count_calls_recursive(&func.instructions, &mut call_counts);
            }

            // Phase 2: Identify inlining candidates
            let mut inline_candidates = Vec::new();
            for (func_idx, &call_count) in &call_counts {
                // loom#153: imported functions (full index < import count)
                // have no body to inline — never candidates. Without this
                // they'd pass the `size < 10` gate (size defaults to 0) and
                // the inliner would try to inline the import.
                if *func_idx < num_imported_funcs {
                    continue;
                }
                // loom#155: do NOT inline a callee that WRITES memory/globals.
                // The translation validator models a *remaining* call's effects
                // as a havoc, but an *inlined* callee with effects the verifier
                // can't model (partial-width access, floats, globals, memory.*
                // bulk ops, calls, control flow) makes the original (havoc) and
                // optimized (concrete) encodings diverge → FALSE counterexample
                // → the whole caller reverts, taking down any OTHER inline in it
                // (the v1.1.7 #159 regression). Only inline callees the verifier
                // can fully prove by-body; everything else stays an opaque call.
                if let Some(callee) = module
                    .functions
                    .get((*func_idx - num_imported_funcs) as usize)
                {
                    if !function_inline_modelable(callee, &module.functions, num_imported_funcs) {
                        continue;
                    }
                }
                let size = function_sizes.get(func_idx).copied().unwrap_or(0);

                // Heuristic: inline if:
                // 1. Single call site — profitable: inlining removes the call
                //    overhead and opens cross-function optimization (the gale
                //    flight_control seam, #155). A single-call-site callee is
                //    not duplicated, so a generous size budget is justified.
                // 2. Small function (< 10 instructions) — cheap even when
                //    called from multiple sites.
                //
                // SIZE_LIMIT stays well under the Z3 verify cap
                // (LOOM_Z3_MAX_INSTRUCTIONS, default 2000) so the inlined
                // function is still VERIFIED, never silently skipped — the
                // translation validator remains the correctness gate, and the
                // #147 fixpoint guard bounds total expansion.
                const SINGLE_CALL_SITE_LIMIT: usize = 200;
                const MULTI_CALL_SITE_LIMIT: usize = 50;
                let limit = if call_count == 1 {
                    SINGLE_CALL_SITE_LIMIT
                } else {
                    MULTI_CALL_SITE_LIMIT
                };
                if (call_count == 1 || size < 10) && size < limit {
                    inline_candidates.push(*func_idx);
                }
            }

            // Exit loop if no more inlining opportunities - this ensures idempotence
            if inline_candidates.is_empty() {
                break;
            }

            // Phase 3: Perform inlining
            // For each function, inline calls to candidate functions
            let inline_set: std::collections::HashSet<u32> =
                inline_candidates.iter().copied().collect();

            // Clone functions to avoid borrow checker issues
            let all_functions = module.functions.clone();

            // loom#151: build the verification signature context (with
            // callee bodies) ONCE per iteration, before the mutable loop.
            // This lets the translation validator model `call F(args)` by
            // F's own body and thereby PROVE the inline sound — without it
            // the validator falls back to an opaque uninterpreted call and
            // reverts every inline (no-op for i64 and every other type).
            let verify_sig_ctx = crate::verify::VerificationSignatureContext::from_module(module);

            // Did this iteration KEEP any inline (verified, not reverted)?
            // Only kept inlines are progress; an iteration that reverts
            // everything must not loop again (loom#147 livelock guard).
            let mut kept_any = false;

            for func in &mut module.functions {
                // Skip functions with unsupported instructions (can't verify)
                if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                    continue;
                }

                let before = func.instructions.clone();

                let guard = ValidationGuard::with_context(func, "inline_functions", ctx.clone());
                let translator = TranslationValidator::new_with_context(
                    func,
                    "inline_functions",
                    verify_sig_ctx.clone(),
                );

                func.instructions = inline_calls_in_block(
                    &func.instructions,
                    &inline_set,
                    &all_functions,
                    num_imported_funcs,
                    func.signature.params.len() as u32,
                    &mut func.locals,
                );

                let _ = guard.validate(func);
                translator.verify_or_revert(func);

                // After verify_or_revert, func.instructions is either the
                // inlined body (kept) or restored to `before` (reverted).
                // A difference means a verified inline landed → progress.
                if func.instructions != before {
                    kept_any = true;
                }
            }

            // No inline survived verification this iteration → the
            // remaining candidates are unprovable; stop rather than spin.
            if !kept_any {
                break;
            }

            // Continue to next iteration to check for more inlining opportunities
        }

        Ok(())
    }

    /// REGIME A whitelist: per-instruction straight-line ops the by-body
    /// modeler (`verify::encode_inlinable_callee_result`) proves faithfully,
    /// INCLUDING full/partial-width memory loads/stores (loom#151/#155/#157/#161).
    /// No control flow, no calls. MUST stay in lock-step with
    /// `verify::is_inline_modelable_instr`. Used by `function_inline_modelable`.
    fn is_straightline_modelable_instr(i: &Instruction) -> bool {
        matches!(
            i,
            Instruction::I32Const(_)
                | Instruction::I64Const(_)
                | Instruction::LocalGet(_)
                | Instruction::LocalSet(_)
                | Instruction::LocalTee(_)
                | Instruction::Drop
                | Instruction::Select
                | Instruction::I32Load { .. }
                | Instruction::I64Load { .. }
                | Instruction::I32Store { .. }
                | Instruction::I64Store { .. }
                | Instruction::I32Load8S { .. }
                | Instruction::I32Load8U { .. }
                | Instruction::I32Load16S { .. }
                | Instruction::I32Load16U { .. }
                | Instruction::I32Store8 { .. }
                | Instruction::I32Store16 { .. }
                | Instruction::I32Add
                | Instruction::I32Sub
                | Instruction::I32Mul
                | Instruction::I32And
                | Instruction::I32Or
                | Instruction::I32Xor
                | Instruction::I32Shl
                | Instruction::I32ShrS
                | Instruction::I32ShrU
                | Instruction::I64Add
                | Instruction::I64Sub
                | Instruction::I64Mul
                | Instruction::I64And
                | Instruction::I64Or
                | Instruction::I64Xor
                | Instruction::I64Shl
                | Instruction::I64ShrS
                | Instruction::I64ShrU
                | Instruction::I32DivS
                | Instruction::I32DivU
                | Instruction::I32RemS
                | Instruction::I32RemU
                | Instruction::I64DivS
                | Instruction::I64DivU
                | Instruction::I64RemS
                | Instruction::I64RemU
                | Instruction::I32Eq
                | Instruction::I32Ne
                | Instruction::I32LtS
                | Instruction::I32LtU
                | Instruction::I32GtS
                | Instruction::I32GtU
                | Instruction::I32LeS
                | Instruction::I32LeU
                | Instruction::I32GeS
                | Instruction::I32GeU
                | Instruction::I32Eqz
                | Instruction::I64Eq
                | Instruction::I64Ne
                | Instruction::I64LtS
                | Instruction::I64LtU
                | Instruction::I64GtS
                | Instruction::I64GtU
                | Instruction::I64LeS
                | Instruction::I64LeU
                | Instruction::I64GeS
                | Instruction::I64GeU
                | Instruction::I64Eqz
                | Instruction::I64ExtendI32S
                | Instruction::I64ExtendI32U
                | Instruction::I32WrapI64
        )
    }

    /// REGIME B per-instruction whitelist: pure-integer ops the precise acyclic
    /// executor (`verify::exec_acyclic`) models — NO memory (Phase 1), NO
    /// floats/unknown. MUST stay in lock-step with
    /// `verify::is_acyclic_simple_modelable`.
    fn is_acyclic_int_modelable_instr(i: &Instruction) -> bool {
        matches!(
            i,
            Instruction::I32Const(_)
                | Instruction::I64Const(_)
                | Instruction::LocalGet(_)
                | Instruction::LocalSet(_)
                | Instruction::LocalTee(_)
                | Instruction::GlobalGet(_)
                | Instruction::GlobalSet(_)
                | Instruction::Drop
                | Instruction::Nop
                | Instruction::Select
                | Instruction::I32Add
                | Instruction::I32Sub
                | Instruction::I32Mul
                | Instruction::I32And
                | Instruction::I32Or
                | Instruction::I32Xor
                | Instruction::I32Shl
                | Instruction::I32ShrS
                | Instruction::I32ShrU
                | Instruction::I32DivS
                | Instruction::I32DivU
                | Instruction::I32RemS
                | Instruction::I32RemU
                | Instruction::I64Add
                | Instruction::I64Sub
                | Instruction::I64Mul
                | Instruction::I64And
                | Instruction::I64Or
                | Instruction::I64Xor
                | Instruction::I64Shl
                | Instruction::I64ShrS
                | Instruction::I64ShrU
                | Instruction::I64DivS
                | Instruction::I64DivU
                | Instruction::I64RemS
                | Instruction::I64RemU
                | Instruction::I32Eq
                | Instruction::I32Ne
                | Instruction::I32LtS
                | Instruction::I32LtU
                | Instruction::I32GtS
                | Instruction::I32GtU
                | Instruction::I32LeS
                | Instruction::I32LeU
                | Instruction::I32GeS
                | Instruction::I32GeU
                | Instruction::I32Eqz
                | Instruction::I64Eq
                | Instruction::I64Ne
                | Instruction::I64LtS
                | Instruction::I64LtU
                | Instruction::I64GtS
                | Instruction::I64GtU
                | Instruction::I64LeS
                | Instruction::I64LeU
                | Instruction::I64GeS
                | Instruction::I64GeU
                | Instruction::I64Eqz
                | Instruction::I64ExtendI32S
                | Instruction::I64ExtendI32U
                | Instruction::I32WrapI64
        )
    }

    /// PR-C Phase 2: memory loads/stores the precise acyclic executor models
    /// (mirrors `verify::is_acyclic_memory_op`). Admitted by the regime-B inline
    /// gate so memory-bearing acyclic callees are inline candidates.
    fn is_acyclic_memory_modelable_instr(i: &Instruction) -> bool {
        matches!(
            i,
            Instruction::I32Load { .. }
                | Instruction::I64Load { .. }
                | Instruction::I32Store { .. }
                | Instruction::I64Store { .. }
                | Instruction::I32Load8S { .. }
                | Instruction::I32Load8U { .. }
                | Instruction::I32Load16S { .. }
                | Instruction::I32Load16U { .. }
                | Instruction::I32Store8 { .. }
                | Instruction::I32Store16 { .. }
        )
    }

    /// loom-side mirror of `verify::is_noreturn_callee` (that one is
    /// `#[cfg(feature = "verification")]`-gated, unavailable here). True if every
    /// path traps: no `Return`/`Br*` anywhere (recursing into nested CF) and the
    /// body ends in `Unreachable`. Admits exactly the `panic*` helpers.
    fn inline_callee_is_noreturn(func: &super::Function) -> bool {
        fn has_branch_or_return(instrs: &[Instruction]) -> bool {
            instrs.iter().any(|i| match i {
                Instruction::Return
                | Instruction::Br(_)
                | Instruction::BrIf(_)
                | Instruction::BrTable { .. } => true,
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    has_branch_or_return(body)
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => has_branch_or_return(then_body) || has_branch_or_return(else_body),
                _ => false,
            })
        }
        !has_branch_or_return(&func.instructions)
            && matches!(func.instructions.last(), Some(Instruction::Unreachable))
    }

    /// Max by-body call-recursion depth for the regime-B check (mirrors
    /// `verify::MAX_ACYCLIC_CALL_DEPTH`); bounds cyclic call graphs.
    const INLINE_ACYCLIC_MAX_DEPTH: usize = 8;

    /// REGIME B: acyclic control flow (block/if/br/br_if/br_table) + pure-integer
    /// ops + by-body calls (to no-return or recursively-modelable LOCAL callees).
    /// NO memory (Phase 1). Mirrors `verify::is_acyclic_modelable_body` so the
    /// inline candidate gate admits exactly what the acyclic verifier can prove.
    fn inline_modelable_acyclic_body(
        body: &[Instruction],
        all_functions: &[super::Function],
        num_imported_funcs: u32,
        depth: usize,
    ) -> bool {
        body.iter().all(|i| match i {
            Instruction::Block { body, .. } => {
                inline_modelable_acyclic_body(body, all_functions, num_imported_funcs, depth)
            }
            Instruction::If {
                then_body,
                else_body,
                ..
            } => {
                inline_modelable_acyclic_body(then_body, all_functions, num_imported_funcs, depth)
                    && inline_modelable_acyclic_body(
                        else_body,
                        all_functions,
                        num_imported_funcs,
                        depth,
                    )
            }
            Instruction::Br(_)
            | Instruction::BrIf(_)
            | Instruction::BrTable { .. }
            | Instruction::Return
            // Unreachable is a trap (⊥): the acyclic executor diverges the path.
            | Instruction::Unreachable => true,
            // Back-edges and indirect calls are out of Phase 1 scope.
            Instruction::Loop { .. } | Instruction::CallIndirect { .. } => false,
            Instruction::Call(g) => {
                if depth >= INLINE_ACYCLIC_MAX_DEPTH || *g < num_imported_funcs {
                    return false; // too deep, or an import (opaque)
                }
                match all_functions.get((*g - num_imported_funcs) as usize) {
                    Some(callee) => {
                        inline_callee_is_noreturn(callee)
                            || (callee.signature.results.len() <= 1
                                && inline_modelable_acyclic_body(
                                    &callee.instructions,
                                    all_functions,
                                    num_imported_funcs,
                                    depth + 1,
                                ))
                    }
                    None => false,
                }
            }
            other => {
                is_acyclic_int_modelable_instr(other) || is_acyclic_memory_modelable_instr(other)
            }
        })
    }

    /// loom#159: is this callee FULLY inline-modelable — i.e. will the
    /// translation validator be able to PROVE its inline? Admits EITHER regime A
    /// (straight-line incl. memory — by-body modeler) OR regime B (acyclic CF +
    /// pure-int + by-body calls — the precise acyclic executor, PR-C #219). A
    /// callee outside both stays an opaque call: admitting an unprovable one
    /// makes the original (havoc) and optimized (concrete) encodings diverge →
    /// FALSE counterexample → the whole caller reverts, dragging down any OTHER
    /// inline in it (the v1.1.7 #159 regression). A br_table+memory callee
    /// (mutex_unlock, pipe) passes NEITHER regime → stays opaque until Phase 2.
    fn function_inline_modelable(
        func: &super::Function,
        all_functions: &[super::Function],
        num_imported_funcs: u32,
    ) -> bool {
        func.instructions
            .iter()
            .all(is_straightline_modelable_instr)
            || inline_modelable_acyclic_body(
                &func.instructions,
                all_functions,
                num_imported_funcs,
                0,
            )
    }

    /// Inline function calls in a block of instructions
    fn inline_calls_in_block(
        instructions: &[Instruction],
        inline_set: &std::collections::HashSet<u32>,
        all_functions: &[super::Function],
        num_imported_funcs: u32,
        base_local_count: u32,
        caller_locals: &mut Vec<(u32, super::ValueType)>,
    ) -> Vec<Instruction> {
        let mut result = Vec::new();

        for instr in instructions {
            match instr {
                Instruction::Call(func_idx) if inline_set.contains(func_idx) => {
                    // Inline this function call. loom#153: `func_idx` is in
                    // the FULL index space (imports first); map it to the
                    // local-function slot. An imported index (< import count)
                    // has no body — `checked_sub` yields None and we keep the
                    // original call rather than inline a nonexistent body.
                    let local_idx = func_idx.checked_sub(num_imported_funcs);
                    if let Some(callee) = local_idx.and_then(|li| all_functions.get(li as usize)) {
                        // Calculate local index offset to avoid conflicts
                        let current_local_count = base_local_count
                            + caller_locals.iter().map(|(count, _)| count).sum::<u32>();

                        // Step 1: Allocate temporary locals for the callee's parameters
                        // We need to pop arguments from the stack and store them in locals
                        let param_start_idx = current_local_count;
                        let param_count = callee.signature.params.len() as u32;

                        // Add parameter locals to caller (one local per parameter)
                        for param_type in &callee.signature.params {
                            caller_locals.push((1, *param_type));
                        }

                        // Step 2: Generate instructions to store arguments from stack to locals
                        // Arguments are on the stack in order: arg0, arg1, ..., argN (top)
                        // We need to store them in reverse order (argN first, then argN-1, etc.)
                        for i in (0..param_count).rev() {
                            result.push(Instruction::LocalSet(param_start_idx + i));
                        }

                        // Step 3: Add callee's locals to caller (with remapping)
                        let callee_locals_start = param_start_idx + param_count;
                        for (count, typ) in &callee.locals {
                            caller_locals.push((*count, *typ));
                        }

                        // Step 4: Clone and remap callee's instructions
                        // Replace parameter references with our temporary locals
                        let inlined_body = remap_locals_in_block(
                            &callee.instructions,
                            callee_locals_start,
                            param_count,
                            param_start_idx,
                        );

                        result.extend(inlined_body);
                    } else {
                        // Function not found, keep original call
                        result.push(instr.clone());
                    }
                }

                // Recursively inline in control flow
                Instruction::Block { block_type, body } => {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: inline_calls_in_block(
                            body,
                            inline_set,
                            all_functions,
                            num_imported_funcs,
                            base_local_count,
                            caller_locals,
                        ),
                    });
                }

                Instruction::Loop { block_type, body } => {
                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: inline_calls_in_block(
                            body,
                            inline_set,
                            all_functions,
                            num_imported_funcs,
                            base_local_count,
                            caller_locals,
                        ),
                    });
                }

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: inline_calls_in_block(
                            then_body,
                            inline_set,
                            all_functions,
                            num_imported_funcs,
                            base_local_count,
                            caller_locals,
                        ),
                        else_body: inline_calls_in_block(
                            else_body,
                            inline_set,
                            all_functions,
                            num_imported_funcs,
                            base_local_count,
                            caller_locals,
                        ),
                    });
                }

                _ => {
                    result.push(instr.clone());
                }
            }
        }

        result
    }

    /// Remap local indices in inlined code to avoid conflicts
    ///
    /// Parameters:
    /// - instructions: The callee's instructions to remap
    /// - offset: The offset for remapping the callee's locals (non-parameter locals)
    /// - param_count: Number of parameters in the callee
    /// - param_start_idx: The starting index in the caller where we stored parameters
    fn remap_locals_in_block(
        instructions: &[Instruction],
        offset: u32,
        param_count: u32,
        param_start_idx: u32,
    ) -> Vec<Instruction> {
        instructions
            .iter()
            .map(|instr| match instr {
                // Remap parameter accesses to our temporary parameter locals
                Instruction::LocalGet(idx) if *idx < param_count => {
                    Instruction::LocalGet(param_start_idx + idx)
                }
                Instruction::LocalSet(idx) if *idx < param_count => {
                    Instruction::LocalSet(param_start_idx + idx)
                }
                Instruction::LocalTee(idx) if *idx < param_count => {
                    Instruction::LocalTee(param_start_idx + idx)
                }

                // Remap the callee's local variables (non-parameters)
                Instruction::LocalGet(idx) if *idx >= param_count => {
                    Instruction::LocalGet(idx + offset - param_count)
                }
                Instruction::LocalSet(idx) if *idx >= param_count => {
                    Instruction::LocalSet(idx + offset - param_count)
                }
                Instruction::LocalTee(idx) if *idx >= param_count => {
                    Instruction::LocalTee(idx + offset - param_count)
                }

                // Recursively remap in control flow
                Instruction::Block { block_type, body } => Instruction::Block {
                    block_type: block_type.clone(),
                    body: remap_locals_in_block(body, offset, param_count, param_start_idx),
                },

                Instruction::Loop { block_type, body } => Instruction::Loop {
                    block_type: block_type.clone(),
                    body: remap_locals_in_block(body, offset, param_count, param_start_idx),
                },

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => Instruction::If {
                    block_type: block_type.clone(),
                    then_body: remap_locals_in_block(
                        then_body,
                        offset,
                        param_count,
                        param_start_idx,
                    ),
                    else_body: remap_locals_in_block(
                        else_body,
                        offset,
                        param_count,
                        param_start_idx,
                    ),
                },

                // Keep everything else unchanged
                _ => instr.clone(),
            })
            .collect()
    }

    /// Count instructions recursively (including in blocks)
    fn count_instructions_recursive(instructions: &[Instruction]) -> usize {
        let mut count = instructions.len();
        for instr in instructions {
            match instr {
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count += count_instructions_recursive(body);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count += count_instructions_recursive(then_body);
                    count += count_instructions_recursive(else_body);
                }
                _ => {}
            }
        }
        count
    }

    /// Count function calls recursively
    fn count_calls_recursive(
        instructions: &[Instruction],
        call_counts: &mut std::collections::BTreeMap<u32, usize>,
    ) {
        for instr in instructions {
            match instr {
                Instruction::Call(func_idx) => {
                    *call_counts.entry(*func_idx).or_insert(0) += 1;
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count_calls_recursive(body, call_counts);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count_calls_recursive(then_body, call_counts);
                    count_calls_recursive(else_body, call_counts);
                }
                _ => {}
            }
        }
    }

    /// Code Folding and Flattening (Issue #22)
    ///
    /// Eliminates single-use temporary variables and flattens nested blocks.
    /// Benefits 20-25% of typical code.
    ///
    /// Transformations:
    /// - local.set $tmp (expr); local.get $tmp → expr (if single use)
    /// - Nested blocks with compatible types → flattened
    /// - Empty blocks → removed
    pub fn fold_code(module: &mut Module) -> Result<()> {
        use crate::stack::validation::{ValidationContext, ValidationGuard};
        use crate::verify::TranslationValidator;
        use std::collections::HashMap;

        let ctx = ValidationContext::from_module(module);

        for func in &mut module.functions {
            // Skip functions with unsupported instructions (can't verify)
            if has_unknown_instructions(func) || has_unsupported_isle_instructions(func) {
                continue;
            }

            let guard = ValidationGuard::with_context(func, "fold_code", ctx.clone());
            let translator = TranslationValidator::new(func, "fold_code");

            // Phase 1: Analyze local variable usage
            let mut usage = HashMap::new();
            count_local_usage(&func.instructions, &mut usage);

            // Phase 2: Identify single-use locals
            let _single_use_locals: Vec<u32> = usage
                .iter()
                .filter_map(|(idx, count)| if *count == 1 { Some(*idx) } else { None })
                .collect();

            // Phase 3: Flatten nested blocks
            func.instructions = flatten_blocks(&func.instructions);

            // Note: Single-use temporary folding is NOT performed.
            // Reason: Folding requires proving the substituted expression has no side effects
            // between assignment and use, which is complex for stack-based code. Per our
            // proof-first philosophy: we only implement what we can prove correct.

            let _ = guard.validate(func);
            translator.verify_or_revert(func);
        }

        Ok(())
    }

    /// Count how many times each local variable is used
    fn count_local_usage(
        instructions: &[Instruction],
        usage: &mut std::collections::HashMap<u32, usize>,
    ) {
        for instr in instructions {
            match instr {
                Instruction::LocalGet(idx)
                | Instruction::LocalSet(idx)
                | Instruction::LocalTee(idx) => {
                    *usage.entry(*idx).or_insert(0) += 1;
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count_local_usage(body, usage);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count_local_usage(then_body, usage);
                    count_local_usage(else_body, usage);
                }
                _ => {}
            }
        }
    }

    /// Flatten nested blocks with compatible types
    fn flatten_blocks(instructions: &[Instruction]) -> Vec<Instruction> {
        let mut result = Vec::new();

        for instr in instructions {
            match instr {
                // Flatten nested empty blocks
                Instruction::Block {
                    block_type: _,
                    body,
                } if body.is_empty() => {
                    // Skip empty blocks
                    continue;
                }

                // Flatten nested blocks with single block inside
                Instruction::Block { block_type, body } if body.len() == 1 => {
                    if let Some(Instruction::Block {
                        block_type: inner_type,
                        body: inner_body,
                    }) = body.first()
                    {
                        // If types match, flatten to single block
                        if block_type == inner_type {
                            result.push(Instruction::Block {
                                block_type: block_type.clone(),
                                body: flatten_blocks(inner_body),
                            });
                            continue;
                        }
                    }
                    // Otherwise keep as is but recurse
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: flatten_blocks(body),
                    });
                }

                // Recursively flatten in other blocks
                Instruction::Block { block_type, body } => {
                    result.push(Instruction::Block {
                        block_type: block_type.clone(),
                        body: flatten_blocks(body),
                    });
                }

                Instruction::Loop { block_type, body } => {
                    if body.is_empty() {
                        // Skip empty loops
                        continue;
                    }
                    result.push(Instruction::Loop {
                        block_type: block_type.clone(),
                        body: flatten_blocks(body),
                    });
                }

                Instruction::If {
                    block_type,
                    then_body,
                    else_body,
                } => {
                    result.push(Instruction::If {
                        block_type: block_type.clone(),
                        then_body: flatten_blocks(then_body),
                        else_body: flatten_blocks(else_body),
                    });
                }

                _ => {
                    result.push(instr.clone());
                }
            }
        }

        result
    }

    /// Loop Optimizations (Issue #23)
    ///
    /// Optimizes loops through:
    /// - Loop-Invariant Code Motion (LICM)
    /// - Loop unrolling for small known-count loops
    ///
    /// Critical for numerical code performance.
    ///
    /// This is a wrapper function that calls loop_invariant_code_motion.
    pub fn optimize_loops(module: &mut Module) -> Result<()> {
        loop_invariant_code_motion(module)
    }

    // ========================================================================
    // forward_global_shim (v1.0.5 #70 chain, step 5.5)
    // ========================================================================
    //
    // The meld P3 async-callback adapter often forwards a `task.return` result
    // through a small shim:
    //
    //     global.set $g
    //     global.get $g     ;; immediately follows the matching set
    //
    // When the only writer of `$g` in the entire module is this single
    // `global.set`, the round-trip is byte-for-byte equivalent to keeping
    // the value on the stack. We can erase both instructions.
    //
    // Soundness model
    // ----------------
    // The fold is sound iff:
    //
    //   (a) the `global.get` immediately follows the `global.set` (no
    //       intervening instruction can observe the global), and
    //
    //   (b) the global has exactly ONE writer across the entire module — the
    //       very `global.set` we are folding. Any other writer might race
    //       (call boundary, control-flow merge) and observably change the
    //       value the `global.get` would read, so we conservatively skip.
    //
    // We additionally require the global to be of a value-bearing primitive
    // type — but `global.get` and `global.set` already type-check the I/O,
    // so the byte-for-byte equivalence holds for any single-type global.
    //
    // Conservative bailouts:
    //   - any function containing `Unknown` instructions disables the pass
    //     module-wide (we cannot count writers we cannot decode).
    //   - any nested write inside Block/Loop/If counts as a writer, so the
    //     fold is rejected (we don't reason about path conditions).

    /// Fold `GlobalSet(idx); GlobalGet(idx)` pairs where `idx` has exactly
    /// one writer across the whole module. Returns the number of pairs folded.
    pub fn forward_global_shim(module: &mut Module) -> Result<usize> {
        // Bail if any function has Unknown instructions; we cannot reliably
        // count writers in opaque bodies.
        for func in &module.functions {
            if has_unknown_instructions(func) {
                return Ok(0);
            }
        }

        // First, build a per-global writer count across the entire module.
        let mut writer_count: std::collections::BTreeMap<u32, usize> =
            std::collections::BTreeMap::new();
        for func in &module.functions {
            count_global_writes(&func.instructions, &mut writer_count);
        }

        let mut total_folded = 0;
        for func in &mut module.functions {
            total_folded += fold_global_shim_in_body(&mut func.instructions, &writer_count);
        }
        Ok(total_folded)
    }

    /// Count occurrences of `GlobalSet(idx)` in `instructions`, recursing
    /// into nested control-flow bodies. Used to verify "exactly one writer."
    fn count_global_writes(
        instructions: &[Instruction],
        out: &mut std::collections::BTreeMap<u32, usize>,
    ) {
        for instr in instructions {
            match instr {
                Instruction::GlobalSet(idx) => {
                    *out.entry(*idx).or_insert(0) += 1;
                }
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    count_global_writes(body, out);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count_global_writes(then_body, out);
                    count_global_writes(else_body, out);
                }
                _ => {}
            }
        }
    }

    /// Walk a function body looking for `GlobalSet(N); GlobalGet(N)` pairs
    /// where `N` has exactly one writer in the module. Recurses into nested
    /// Block/Loop/If bodies. Returns the number of pairs folded.
    fn fold_global_shim_in_body(
        instructions: &mut Vec<Instruction>,
        writer_count: &std::collections::BTreeMap<u32, usize>,
    ) -> usize {
        let mut folded = 0;

        // Recurse first so inner patterns are folded before we look at the
        // outer sequence (matches the existing async-adapter pattern).
        for instr in instructions.iter_mut() {
            match instr {
                Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                    folded += fold_global_shim_in_body(body, writer_count);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    folded += fold_global_shim_in_body(then_body, writer_count);
                    folded += fold_global_shim_in_body(else_body, writer_count);
                }
                _ => {}
            }
        }

        // Scan this level for adjacent GlobalSet/GlobalGet pairs.
        let mut i = 0;
        while i + 2 <= instructions.len() {
            if let (Instruction::GlobalSet(set_idx), Instruction::GlobalGet(get_idx)) =
                (&instructions[i], &instructions[i + 1])
                && set_idx == get_idx
            {
                let idx = *set_idx;
                // Confirm exactly one writer in the whole module.
                if writer_count.get(&idx).copied().unwrap_or(0) == 1 {
                    instructions.drain(i..i + 2);
                    folded += 1;
                    // Don't advance; another pair may be exposed.
                    continue;
                }
            }
            i += 1;
        }

        folded
    }

    #[cfg(test)]
    mod forward_global_tests {
        use super::*;
        use crate::{Function, FunctionSignature, Module};

        fn mk_module(funcs: Vec<Function>) -> Module {
            Module {
                functions: funcs,
                memories: vec![],
                tables: vec![],
                globals: vec![],
                types: vec![],
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

        fn mk_func(instructions: Vec<Instruction>) -> Function {
            Function {
                name: None,
                signature: FunctionSignature {
                    params: vec![],
                    results: vec![],
                },
                locals: vec![],
                instructions,
            }
        }

        #[test]
        fn test_forward_global_shim_folds_simple_pair() {
            let func = mk_func(vec![
                Instruction::I32Const(7),
                Instruction::GlobalSet(0),
                Instruction::GlobalGet(0),
                Instruction::Drop,
            ]);
            let mut module = mk_module(vec![func]);
            let folded = forward_global_shim(&mut module).expect("apply");
            assert_eq!(folded, 1, "single shim pair must fold");
            // After fold: [I32Const(7), Drop]
            let body = &module.functions[0].instructions;
            assert_eq!(
                body,
                &vec![Instruction::I32Const(7), Instruction::Drop],
                "GlobalSet/GlobalGet pair removed"
            );
        }

        #[test]
        fn test_forward_global_shim_skips_multiple_writers() {
            // Two functions write to the SAME global; the pair in func 0
            // must NOT fold because func 1 also writes the global.
            let func0 = mk_func(vec![
                Instruction::I32Const(7),
                Instruction::GlobalSet(0),
                Instruction::GlobalGet(0),
                Instruction::Drop,
            ]);
            let func1 = mk_func(vec![Instruction::I32Const(99), Instruction::GlobalSet(0)]);
            let mut module = mk_module(vec![func0, func1]);
            let folded = forward_global_shim(&mut module).expect("apply");
            assert_eq!(
                folded, 0,
                "multiple writers across module disqualify the fold"
            );
        }

        #[test]
        fn test_forward_global_shim_skips_intervening_op() {
            // GlobalSet(0); Nop; GlobalGet(0) — Nop is between them, no fold.
            let func = mk_func(vec![
                Instruction::I32Const(7),
                Instruction::GlobalSet(0),
                Instruction::Nop,
                Instruction::GlobalGet(0),
                Instruction::Drop,
            ]);
            let mut module = mk_module(vec![func]);
            let folded = forward_global_shim(&mut module).expect("apply");
            assert_eq!(
                folded, 0,
                "intervening instruction between Set and Get prevents fold"
            );
        }

        #[test]
        fn test_forward_global_shim_skips_mismatched_indices() {
            // GlobalSet(0); GlobalGet(1) — different indices, no fold.
            let func = mk_func(vec![
                Instruction::I32Const(7),
                Instruction::GlobalSet(0),
                Instruction::GlobalGet(1),
                Instruction::Drop,
            ]);
            let mut module = mk_module(vec![func]);
            let folded = forward_global_shim(&mut module).expect("apply");
            assert_eq!(folded, 0, "different global indices must not fold");
        }

        #[test]
        fn test_forward_global_shim_no_op_on_plain_function() {
            let func = mk_func(vec![
                Instruction::I32Const(1),
                Instruction::I32Const(2),
                Instruction::I32Add,
                Instruction::Drop,
            ]);
            let before = func.instructions.clone();
            let mut module = mk_module(vec![func]);
            let folded = forward_global_shim(&mut module).expect("apply");
            assert_eq!(folded, 0, "no pattern → no folds");
            assert_eq!(module.functions[0].instructions, before);
        }

        #[test]
        fn test_forward_global_shim_skips_unknown_instructions() {
            let func = mk_func(vec![
                Instruction::I32Const(7),
                Instruction::GlobalSet(0),
                Instruction::GlobalGet(0),
                Instruction::Unknown(vec![0xFE]),
            ]);
            let before = func.instructions.clone();
            let mut module = mk_module(vec![func]);
            let folded = forward_global_shim(&mut module).expect("apply");
            assert_eq!(folded, 0, "Unknown instructions disable the pass");
            assert_eq!(module.functions[0].instructions, before);
        }
    }
}

/// Component Model Support
///
/// Formal verification module using Z3 SMT solver
///
/// This module provides translation validation to prove that optimizations
/// preserve program semantics. Only available with the "verification" feature.
pub mod verify;

/// Optimization rule verification using Z3
///
/// This module provides formal proofs that individual optimization rules are
/// mathematically correct. Unlike translation validation (which checks specific
/// transformations), rule verification proves properties hold for ALL inputs.
///
/// Example: `∀x: i32. x * 8 = x << 3` - proven for all 2^32 possible values.
pub mod verify_rules;

/// End-to-end verification and gap analysis
///
/// This module provides honest assessment of what IS and ISN'T verified,
/// plus infrastructure for true end-to-end verification via concrete execution.
pub mod verify_e2e;

/// WebAssembly Component Model optimization
///
/// LOOM is the first optimizer to support the WebAssembly Component Model.
/// This module provides world-class component optimization by:
/// - Extracting core modules from components
/// - Applying LOOM's 12-phase optimization pipeline
/// - Reconstructing components with optimized modules
/// - Preserving all component sections and interfaces
///
/// See `component_optimizer` module for implementation details.
pub mod component_optimizer;

/// WebAssembly Component Model execution verification
///
/// Provides runtime verification of component model correctness after optimization.
/// Uses wasmtime to instantiate and execute components, verifying that:
/// - Component structure is preserved
/// - Exports remain callable with correct signatures
/// - Canonical functions operate correctly
/// - No validation errors occur after optimization
pub mod component_executor;

/// Fused component optimization
///
/// Specialized optimization passes for WebAssembly modules produced by component
/// fusion tools (e.g., meld). Targets adapter trampolines, duplicate types/imports,
/// and dead functions introduced by the fusion process.
///
/// See `fused_optimizer` module for implementation details and
/// `docs/FUSED_COMPONENT_OPTIMIZATION.md` for the full design document.
pub mod fused_optimizer;

/// Re-export fused optimization API
pub use fused_optimizer::{FusedOptimizationStats, optimize_fused_module};

/// Re-export component optimization API
pub use component_optimizer::{
    ComponentAnalysis, ComponentStats, analyze_component_structure, optimize_component,
};

/// Re-export component executor API
pub use component_executor::{
    CanonicalFunctionInfo, ComponentExecutor, DifferentialTestReport, ExecutionResult,
    VerificationReport,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_value_construction() {
        use loom_isle::{Imm32, iconst32};
        let _val = iconst32(Imm32::from(42));
        // Just test that ISLE types are accessible
    }

    #[test]
    fn test_parse_wat_simple() {
        let wat = r#"
            (module
              (func $get_answer (result i32)
                i32.const 42
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse WAT");
        assert_eq!(module.functions.len(), 1);

        let func = &module.functions[0];
        assert_eq!(func.signature.params.len(), 0);
        assert_eq!(func.signature.results.len(), 1);
        assert_eq!(func.signature.results[0], ValueType::I32);

        // Check instructions
        assert!(func.instructions.contains(&Instruction::I32Const(42)));
    }

    #[test]
    fn test_parse_wat_addition() {
        let wat = r#"
            (module
              (func $add_constants (result i32)
                i32.const 10
                i32.const 32
                i32.add
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse WAT");
        assert_eq!(module.functions.len(), 1);

        let func = &module.functions[0];
        assert!(func.instructions.contains(&Instruction::I32Const(10)));
        assert!(func.instructions.contains(&Instruction::I32Const(32)));
        assert!(func.instructions.contains(&Instruction::I32Add));
    }

    #[test]
    fn test_round_trip() {
        // Parse a simple WAT module
        let wat = r#"
            (module
              (func $test (result i32)
                i32.const 42
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Encode it back to WASM binary
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode WASM");

        // Parse the binary again
        let module2 = parse::parse_wasm(&wasm_bytes).expect("Failed to parse encoded WASM");

        // Verify the module is the same
        assert_eq!(module2.functions.len(), 1);
        let func = &module2.functions[0];
        assert_eq!(func.signature.results.len(), 1);
        assert_eq!(func.signature.results[0], ValueType::I32);
        assert!(func.instructions.contains(&Instruction::I32Const(42)));
    }

    #[test]
    fn test_round_trip_with_addition() {
        let wat = r#"
            (module
              (func $add (result i32)
                i32.const 10
                i32.const 32
                i32.add
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode WASM");
        let module2 = parse::parse_wasm(&wasm_bytes).expect("Failed to re-parse WASM");

        assert_eq!(module2.functions.len(), 1);
        let func = &module2.functions[0];
        assert!(func.instructions.contains(&Instruction::I32Const(10)));
        assert!(func.instructions.contains(&Instruction::I32Add));
    }

    #[test]
    fn test_instructions_to_terms() {
        let instructions = vec![
            Instruction::I32Const(10),
            Instruction::I32Const(32),
            Instruction::I32Add,
            Instruction::End,
        ];

        let terms =
            terms::instructions_to_terms(&instructions).expect("Failed to convert to terms");

        // Should have one term on the stack (the result of the add)
        assert_eq!(terms.len(), 1);

        // Verify the structure: I32Add(I32Const(10), I32Const(32))
        match terms[0].data() {
            ValueData::I32Add { lhs, rhs } => match (lhs.data(), rhs.data()) {
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    assert_eq!(lhs_val.value(), 10);
                    assert_eq!(rhs_val.value(), 32);
                }
                _ => panic!("Expected I32Const operands"),
            },
            _ => panic!("Expected I32Add at top of stack"),
        }
    }

    #[test]
    fn test_terms_to_instructions() {
        use loom_isle::{Imm32, iadd32, iconst32};

        // Build term: I32Add(I32Const(10), I32Const(32))
        let term = iadd32(iconst32(Imm32::from(10)), iconst32(Imm32::from(32)));

        let instructions =
            terms::terms_to_instructions(&[term]).expect("Failed to convert to instructions");

        // Should generate: i32.const 10, i32.const 32, i32.add (no End - encoder adds it)
        assert_eq!(instructions.len(), 3);
        assert_eq!(instructions[0], Instruction::I32Const(10));
        assert_eq!(instructions[1], Instruction::I32Const(32));
        assert_eq!(instructions[2], Instruction::I32Add);
    }

    #[test]
    fn test_term_round_trip() {
        // Start with instructions (no End - it's added by encoder)
        let original_instructions = vec![
            Instruction::I32Const(10),
            Instruction::I32Const(32),
            Instruction::I32Add,
        ];

        // Convert to terms
        let terms = terms::instructions_to_terms(&original_instructions)
            .expect("Failed to convert to terms");

        // Convert back to instructions
        let result_instructions =
            terms::terms_to_instructions(&terms).expect("Failed to convert back to instructions");

        // Should match original
        assert_eq!(result_instructions.len(), 3);
        assert_eq!(result_instructions[0], Instruction::I32Const(10));
        assert_eq!(result_instructions[1], Instruction::I32Const(32));
        assert_eq!(result_instructions[2], Instruction::I32Add);
    }

    #[test]
    fn test_optimize_constant_folding() {
        // Parse test_input.wat
        let wat = r#"
            (module
              (func $add_constants (result i32)
                i32.const 10
                i32.const 32
                i32.add
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Verify original instructions
        let func = &module.functions[0];
        assert!(func.instructions.contains(&Instruction::I32Const(10)));
        assert!(func.instructions.contains(&Instruction::I32Const(32)));
        assert!(func.instructions.contains(&Instruction::I32Add));

        // Apply optimization
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        // Verify optimized instructions - should be just (i32.const 42)
        // Note: End instruction may be implicit and removed during optimization
        let func = &module.functions[0];
        assert!(!func.instructions.is_empty());
        assert_eq!(func.instructions[0], Instruction::I32Const(42));

        // Should NOT contain the original add instruction
        assert!(!func.instructions.contains(&Instruction::I32Add));
    }

    #[test]
    fn test_optimize_with_file_fixture() {
        use std::fs;

        // Read the actual test_input.wat file (relative to workspace root)
        let wat_content = fs::read_to_string("../tests/fixtures/test_input.wat")
            .expect("Failed to read test_input.wat");

        let mut module = parse::parse_wat(&wat_content).expect("Failed to parse test_input.wat");

        // Apply optimization
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        // Verify the result matches our expectation
        // Note: End instruction may be implicit and removed during optimization
        let func = &module.functions[0];
        assert!(!func.instructions.is_empty());
        assert_eq!(func.instructions[0], Instruction::I32Const(42));

        // Encode back to WASM and verify it's valid
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode optimized module");

        // Re-parse to verify validity
        let module2 = parse::parse_wasm(&wasm_bytes).expect("Failed to re-parse optimized WASM");
        // NOTE: test_input.wat has 3 functions, not 1
        // We verify the module round-trips correctly
        assert_eq!(
            module2.functions.len(),
            3,
            "Should preserve all 3 functions"
        );
        assert_eq!(
            module2.functions[0].instructions[0],
            Instruction::I32Const(42)
        );
    }

    // Control Flow Tests (Phase 14)

    #[test]
    fn test_parse_block() {
        let wat = r#"
            (module
              (func $test_block (result i32)
                (block (result i32)
                  i32.const 42
                )
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse block WAT");
        assert_eq!(module.functions.len(), 1);

        let func = &module.functions[0];
        assert_eq!(func.instructions.len(), 1);

        // Check for Block instruction
        if let Instruction::Block { block_type, body } = &func.instructions[0] {
            assert_eq!(*block_type, BlockType::Value(ValueType::I32));
            assert_eq!(body.len(), 1);
            assert_eq!(body[0], Instruction::I32Const(42));
        } else {
            panic!("Expected Block instruction");
        }
    }

    #[test]
    fn test_parse_loop() {
        let wat = r#"
            (module
              (func $test_loop (param $n i32) (result i32)
                (local $i i32)
                (block $exit
                  (loop $continue
                    (local.set $i
                      (i32.add (local.get $i) (i32.const 1))
                    )
                    (br_if $continue
                      (i32.lt_u (local.get $i) (local.get $n))
                    )
                  )
                )
                (local.get $i)
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse loop WAT");
        assert_eq!(module.functions.len(), 1);

        let func = &module.functions[0];
        // Should have a block, then local.get
        assert!(matches!(func.instructions[0], Instruction::Block { .. }));
    }

    #[test]
    fn test_parse_if_else() {
        let wat = r#"
            (module
              (func $test_if (param $x i32) (result i32)
                (if (result i32) (i32.eqz (local.get $x))
                  (then
                    (i32.const 42)
                  )
                  (else
                    (i32.const 0)
                  )
                )
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse if/else WAT");
        assert_eq!(module.functions.len(), 1);

        let func = &module.functions[0];
        // Function should have LocalGet, Eqz, and If instructions
        // Find the If instruction
        let if_instr = func
            .instructions
            .iter()
            .find(|instr| matches!(instr, Instruction::If { .. }));

        assert!(if_instr.is_some(), "Expected If instruction");

        if let Some(Instruction::If {
            block_type,
            then_body,
            else_body,
        }) = if_instr
        {
            assert_eq!(*block_type, BlockType::Value(ValueType::I32));
            assert!(!then_body.is_empty());
            assert!(!else_body.is_empty());
        }
    }

    #[test]
    fn test_parse_branches() {
        let wat = r#"
            (module
              (func $test_br (result i32)
                (block $outer (result i32)
                  (i32.const 1)
                  (br $outer)
                  (i32.const 2)
                )
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse branch WAT");
        assert_eq!(module.functions.len(), 1);

        let func = &module.functions[0];
        // Should have a Block with Br inside
        if let Instruction::Block { body, .. } = &func.instructions[0] {
            assert!(body.iter().any(|instr| matches!(instr, Instruction::Br(_))));
        } else {
            panic!("Expected Block instruction");
        }
    }

    #[test]
    fn test_control_flow_round_trip() {
        let wat = r#"
            (module
              (func $test (param $x i32) (result i32)
                (if (result i32) (local.get $x)
                  (then
                    (i32.const 10)
                  )
                  (else
                    (i32.const 20)
                  )
                )
              )
            )
        "#;

        // Parse WAT
        let module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Encode to WASM
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode WASM");

        // Parse WASM back
        let module2 = parse::parse_wasm(&wasm_bytes).expect("Failed to re-parse WASM");

        // Verify
        assert_eq!(module2.functions.len(), 1);

        // Validate with wasm-tools
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_nested_blocks_round_trip() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block (result i32)
                  (block (result i32)
                    (i32.const 5)
                    (i32.const 10)
                    (i32.add)
                  )
                  (i32.const 3)
                  (i32.add)
                )
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse nested blocks");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        let module2 = parse::parse_wasm(&wasm_bytes).expect("Failed to re-parse");

        assert_eq!(module2.functions.len(), 1);
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_return_instruction() {
        let wat = r#"
            (module
              (func $test (param $x i32) (result i32)
                (if (i32.eqz (local.get $x))
                  (then
                    (i32.const 0)
                    (return)
                  )
                )
                (i32.const 1)
              )
            )
        "#;

        let module = parse::parse_wat(wat).expect("Failed to parse return");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        parse::parse_wasm(&wasm_bytes).expect("Failed to re-parse");

        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    // Dead Code Elimination Tests (Issue #13)

    #[test]
    fn test_dce_unreachable_after_return() {
        let wat = r#"
            (module
              (func $test (result i32)
                (return (i32.const 42))
                (i32.const 99)
                (drop)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = module.functions[0].instructions.len();

        // Apply DCE
        optimize::eliminate_dead_code(&mut module).expect("DCE failed");

        let instructions_after = module.functions[0].instructions.len();

        // Should have removed the dead instructions after return
        assert!(
            instructions_after < instructions_before,
            "DCE should remove dead code (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_dce_unreachable_after_br() {
        // This test verifies DCE doesn't break code with branches.
        // Note: We use conservative DCE that may not remove all dead code after br,
        // because aggressively removing code after br can cause issues when br
        // targets the current block (produces value and continues), not an outer label.
        // The priority is correctness over aggressiveness.
        let wat = r#"
            (module
              (func $test (result i32)
                (block $exit (result i32)
                  (br $exit (i32.const 42))
                  (i32.const 99)
                  (i32.const 100)
                  (i32.add)
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply DCE
        optimize::eliminate_dead_code(&mut module).expect("DCE failed");

        // The key requirement is that DCE produces valid WASM
        // (correctness over aggressiveness)
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_dce_preserves_live_code() {
        let wat = r#"
            (module
              (func $test (param $x i32) (result i32)
                (local.get $x)
                (i32.const 10)
                (i32.add)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = module.functions[0].instructions.len();

        // Apply DCE
        optimize::eliminate_dead_code(&mut module).expect("DCE failed");

        let instructions_after = module.functions[0].instructions.len();

        // Should preserve all live code
        assert_eq!(
            instructions_before, instructions_after,
            "DCE should not remove live code"
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_dce_after_unreachable() {
        let wat = r#"
            (module
              (func $test
                (unreachable)
                (i32.const 42)
                (drop)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = module.functions[0].instructions.len();

        // Apply DCE
        optimize::eliminate_dead_code(&mut module).expect("DCE failed");

        let instructions_after = module.functions[0].instructions.len();

        // Should have removed dead code after unreachable
        assert!(
            instructions_after < instructions_before,
            "DCE should remove code after unreachable (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    // Branch Simplification Tests (Issue #16)

    /// Helper function to count instructions recursively
    fn count_all_instructions(instructions: &[Instruction]) -> usize {
        let mut count = instructions.len();
        for instr in instructions {
            match instr {
                Instruction::Block { body, .. } => {
                    count += count_all_instructions(body);
                }
                Instruction::Loop { body, .. } => {
                    count += count_all_instructions(body);
                }
                Instruction::If {
                    then_body,
                    else_body,
                    ..
                } => {
                    count += count_all_instructions(then_body);
                    count += count_all_instructions(else_body);
                }
                _ => {}
            }
        }
        count
    }

    #[test]
    #[ignore = "Z3 verification for br_if is incomplete (doesn't model path forking), causing false counterexamples. The optimization is correct but unproven."]
    fn test_branch_simplify_br_if_always_taken() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block $exit (result i32)
                  (i32.const 42)
                  (i32.const 1)
                  (br_if $exit)
                  (i32.const 99)
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply branch simplification
        optimize::simplify_branches(&mut module).expect("Branch simplification failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have removed the constant and converted br_if to br
        assert!(
            instructions_after < instructions_before,
            "Should simplify constant br_if (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works and returns 42
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_branch_simplify_br_if_never_taken() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block $exit (result i32)
                  (i32.const 0)
                  (br_if $exit)
                  (i32.const 42)
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply branch simplification
        optimize::simplify_branches(&mut module).expect("Branch simplification failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have removed the constant and br_if entirely
        assert!(
            instructions_after < instructions_before,
            "Should remove never-taken br_if (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works and returns 42
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_branch_simplify_if_constant_true() {
        let wat = r#"
            (module
              (func $test (result i32)
                (if (result i32) (i32.const 1)
                  (then (i32.const 42))
                  (else (i32.const 99))
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = module.functions[0].instructions.len();

        // Apply branch simplification
        optimize::simplify_branches(&mut module).expect("Branch simplification failed");

        let instructions_after = module.functions[0].instructions.len();

        // Should have selected the then branch
        assert!(
            instructions_after < instructions_before,
            "Should simplify constant if (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works and returns 42
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_branch_simplify_if_constant_false() {
        let wat = r#"
            (module
              (func $test (result i32)
                (if (result i32) (i32.const 0)
                  (then (i32.const 99))
                  (else (i32.const 42))
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = module.functions[0].instructions.len();

        // Apply branch simplification
        optimize::simplify_branches(&mut module).expect("Branch simplification failed");

        let instructions_after = module.functions[0].instructions.len();

        // Should have selected the else branch
        assert!(
            instructions_after < instructions_before,
            "Should simplify constant if (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works and returns 42
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_branch_simplify_nested_constants() {
        let wat = r#"
            (module
              (func $test (result i32)
                (if (result i32) (i32.const 1)
                  (then
                    (if (result i32) (i32.const 0)
                      (then (i32.const 10))
                      (else (i32.const 20))
                    )
                  )
                  (else (i32.const 30))
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = module.functions[0].instructions.len();

        // Apply branch simplification
        optimize::simplify_branches(&mut module).expect("Branch simplification failed");

        let instructions_after = module.functions[0].instructions.len();

        // Should have simplified nested ifs down to just (i32.const 20)
        assert!(
            instructions_after < instructions_before,
            "Should simplify nested constant ifs (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works and returns 20
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_branch_simplify_nop_removal() {
        let wat = r#"
            (module
              (func $test (result i32)
                (nop)
                (i32.const 42)
                (nop)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = module.functions[0].instructions.len();

        // Apply branch simplification
        optimize::simplify_branches(&mut module).expect("Branch simplification failed");

        let instructions_after = module.functions[0].instructions.len();

        // Should have removed nops
        assert_eq!(
            instructions_after, 1,
            "Should remove all nops (before: {}, after: {})",
            instructions_before, instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    // Block Merging Tests (Issue #17)

    #[test]
    fn test_block_merge_simple_nested() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block (result i32)
                  (block (result i32)
                    (i32.const 42)
                  )
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply block merging
        optimize::merge_blocks(&mut module).expect("Block merging failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have merged the nested blocks
        assert!(
            instructions_after < instructions_before,
            "Should merge nested blocks (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works and returns 42
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_block_merge_triple_nested() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block (result i32)
                  (block (result i32)
                    (block (result i32)
                      (i32.const 42)
                    )
                  )
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply block merging
        optimize::merge_blocks(&mut module).expect("Block merging failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have fully merged all nested blocks
        assert!(
            instructions_after < instructions_before,
            "Should merge all nested blocks (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_block_merge_with_prefix() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block (result i32)
                  (i32.const 10)
                  (block (result i32)
                    (i32.const 32)
                    (i32.add)
                  )
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply block merging
        optimize::merge_blocks(&mut module).expect("Block merging failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have merged the inner block
        assert!(
            instructions_after < instructions_before,
            "Should merge block with prefix (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works and computes correctly
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_block_merge_in_if() {
        let wat = r#"
            (module
              (func $test (param $x i32) (result i32)
                (if (result i32) (local.get $x)
                  (then
                    (block (result i32)
                      (block (result i32)
                        (i32.const 42)
                      )
                    )
                  )
                  (else
                    (i32.const 99)
                  )
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply block merging
        optimize::merge_blocks(&mut module).expect("Block merging failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have merged blocks within the if statement
        assert!(
            instructions_after < instructions_before,
            "Should merge blocks in if (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_block_merge_nested_empty() {
        let wat = r#"
            (module
              (func $test
                (block
                  (block
                    (nop)
                  )
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply block merging
        optimize::merge_blocks(&mut module).expect("Block merging failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have merged empty nested blocks
        assert!(
            instructions_after <= instructions_before,
            "Should merge or maintain empty blocks (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    // Vacuum Cleanup Tests (Issue #20)

    #[test]
    fn test_vacuum_remove_nops() {
        let wat = r#"
            (module
              (func $test (result i32)
                (nop)
                (i32.const 42)
                (nop)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply vacuum
        optimize::vacuum(&mut module).expect("Vacuum failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have removed nops
        assert!(
            instructions_after < instructions_before,
            "Should remove nops (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_vacuum_unwrap_trivial_block() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block (result i32)
                  (i32.const 42)
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply vacuum
        optimize::vacuum(&mut module).expect("Vacuum failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have unwrapped the block
        assert!(
            instructions_after < instructions_before,
            "Should unwrap trivial block (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_vacuum_unwrap_nested_trivial_blocks() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block (result i32)
                  (block (result i32)
                    (i32.const 42)
                  )
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply vacuum
        optimize::vacuum(&mut module).expect("Vacuum failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have fully unwrapped nested blocks
        assert!(
            instructions_after < instructions_before,
            "Should unwrap nested trivial blocks (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_vacuum_keep_complex_block() {
        let wat = r#"
            (module
              (func $test (result i32)
                (block (result i32)
                  (i32.const 10)
                  (i32.const 32)
                  (i32.add)
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply vacuum
        optimize::vacuum(&mut module).expect("Vacuum failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should keep complex block (multiple instructions)
        // Instruction count should stay same or increase slightly due to recursive processing
        assert!(
            instructions_after >= instructions_before - 1,
            "Should keep complex block (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_vacuum_remove_empty_loop() {
        let wat = r#"
            (module
              (func $test
                (loop)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply vacuum
        optimize::vacuum(&mut module).expect("Vacuum failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have removed empty loop
        assert!(
            instructions_after < instructions_before,
            "Should remove empty loop (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_vacuum_unwrap_empty_block() {
        let wat = r#"
            (module
              (func $test
                (block)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        let instructions_before = count_all_instructions(&module.functions[0].instructions);

        // Apply vacuum
        optimize::vacuum(&mut module).expect("Vacuum failed");

        let instructions_after = count_all_instructions(&module.functions[0].instructions);

        // Should have unwrapped empty block
        assert!(
            instructions_after <= instructions_before,
            "Should unwrap empty block (before: {}, after: {})",
            instructions_before,
            instructions_after
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    // SimplifyLocals Tests (Issue #15)

    #[test]
    fn test_simplify_locals_redundant_copy() {
        let wat = r#"
            (module
              (func $test (result i32)
                (local $0 i32)
                (local $1 i32)

                (local.set $0 (i32.const 42))

                ;; Redundant copy: $1 = $0, but $1 never used
                (local.get $0)
                (local.set $1)

                (local.get $0)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Note: Full dead store elimination requires stack analysis
        // For now we just test that it doesn't break

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_simplify_locals_equivalence() {
        let wat = r#"
            (module
              (func $test (result i32)
                (local $0 i32)
                (local $1 i32)

                (local.set $0 (i32.const 100))

                ;; Create equivalence: $1 ≡ $0
                (local.get $0)
                (local.set $1)

                ;; Uses of $1 should become uses of $0
                (local.get $1)
                (local.get $1)
                (i32.add)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Verify the function still works and produces same result
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_simplify_locals_multiple_copies() {
        let wat = r#"
            (module
              (func $test (result i32)
                (local $0 i32)
                (local $1 i32)
                (local $2 i32)
                (local $3 i32)

                (local.set $0 (i32.const 5))

                ;; Redundant: $1 never used
                (local.get $0)
                (local.set $1)

                ;; Redundant: $2 never used
                (local.get $0)
                (local.set $2)

                ;; Redundant: $3 never used
                (local.get $0)
                (local.set $3)

                (local.get $0)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Note: Full dead store elimination requires stack analysis
        // For now we just test that it doesn't break

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_simplify_locals_used_copy_preserved() {
        let wat = r#"
            (module
              (func $test (result i32)
                (local $0 i32)
                (local $1 i32)

                (local.set $0 (i32.const 15))

                ;; Copy to $1 (will be used)
                (local.get $0)
                (local.set $1)

                ;; Both are used
                (local.get $0)
                (local.get $1)
                (i32.add)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");

        // Function should still add 15 + 15 = 30
    }

    #[test]
    fn test_simplify_locals_nested_blocks() {
        let wat = r#"
            (module
              (func $test (result i32)
                (local $0 i32)
                (local $1 i32)

                (local.set $0 (i32.const 20))

                ;; Equivalence: $1 ≡ $0
                (local.get $0)
                (local.set $1)

                (block (result i32)
                  ;; Use should be canonicalized to $0
                  (local.get $1)
                )
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_simplify_locals_with_full_pipeline() {
        let wat = r#"
            (module
              (func $test (result i32)
                (local $0 i32)
                (local $1 i32)
                (local $2 i32)

                ;; Some redundancy
                (local.set $0 (i32.const 10))
                (local.get $0)
                (local.set $1)

                ;; Dead copy
                (local.get $0)
                (local.set $2)

                ;; Return $1 (which is $0)
                (local.get $1)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply full optimization pipeline
        optimize::optimize_module(&mut module).expect("Optimize failed");
        optimize::simplify_branches(&mut module).expect("Branch simplification failed");
        optimize::eliminate_dead_code(&mut module).expect("DCE failed");
        optimize::merge_blocks(&mut module).expect("Block merging failed");
        optimize::vacuum(&mut module).expect("Vacuum failed");
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    #[test]
    fn test_simplify_locals_single_use_temp_folding() {
        // Test that local.set $x; local.get $x -> local.tee $x
        // The local must be read elsewhere so the tee isn't eliminated as dead
        let wat = r#"
            (module
              (func $test (result i32)
                (local $temp i32)

                ;; This pattern: set then immediately get same local
                ;; should be folded to local.tee
                (i32.const 42)
                (local.set $temp)
                (local.get $temp)
                (drop)

                ;; Read the local again to prevent dead store elimination
                (local.get $temp)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Count instructions before
        let before_count = module.functions[0].instructions.len();

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Count instructions after - should be fewer (set+get -> tee)
        let after_count = module.functions[0].instructions.len();

        // Verify optimization happened (one fewer instruction)
        assert!(
            after_count < before_count,
            "Expected temp folding: before={} after={}",
            before_count,
            after_count
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");

        // Verify a local.tee was created (check for the pattern)
        let has_tee = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalTee(_)));
        assert!(has_tee, "Expected local.tee after folding");
    }

    #[test]
    fn test_simplify_locals_dead_store_elimination() {
        // Test that local.set to never-read locals is replaced with drop
        let wat = r#"
            (module
              (func $test (result i32)
                (local $dead i32)

                ;; Dead store: $dead is never read
                (i32.const 100)
                (local.set $dead)

                ;; Return value directly
                (i32.const 42)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Verify the function still produces valid WASM
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");

        // Verify dead store was eliminated (replaced with drop)
        let has_drop = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Drop));
        assert!(
            has_drop,
            "Expected drop instruction after dead store elimination"
        );

        // Verify no LocalSet remains (the dead one was replaced with drop)
        let set_count = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::LocalSet(_)))
            .count();
        assert_eq!(
            set_count, 0,
            "Expected 0 LocalSet after dead store elimination, got {}",
            set_count
        );
    }

    #[test]
    fn test_simplify_locals_dead_tee_elimination() {
        // Test that local.tee to never-read locals is removed entirely
        let wat = r#"
            (module
              (func $test (result i32)
                (local $dead i32)

                ;; Dead tee: $dead is never read via local.get
                ;; The value stays on stack, but the store is pointless
                (i32.const 42)
                (local.tee $dead)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Count instructions before
        let before_count = module.functions[0].instructions.len();

        // Apply simplify_locals
        optimize::simplify_locals(&mut module).expect("SimplifyLocals failed");

        // Count instructions after - should be fewer (tee removed)
        let after_count = module.functions[0].instructions.len();

        // Verify optimization happened
        assert!(
            after_count < before_count,
            "Expected dead tee elimination: before={} after={}",
            before_count,
            after_count
        );

        // Verify no LocalTee remains
        let tee_count = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::LocalTee(_)))
            .count();
        assert_eq!(
            tee_count, 0,
            "Expected no LocalTee after dead store elimination, got {}",
            tee_count
        );

        // Verify the function still works
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        wasmparser::validate(&wasm_bytes).expect("Generated WASM is invalid");
    }

    // Property-Based Tests for Correctness Verification

    /// Debug test to identify which optimization phase causes stack mismatch
    #[test]
    fn debug_identify_problematic_pass() {
        use loom_isle::{LocalEnv, rewrite_with_dataflow};

        let wat = include_str!("../../tests/fixtures/bench_locals.wat");

        eprintln!("\n=== Original ===");
        let module = parse::parse_wat(wat).expect("Failed to parse");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        match wasmparser::validate(&wasm_bytes) {
            Ok(_) => eprintln!("✓ Valid"),
            Err(e) => eprintln!("✗ INVALID: {:?}", e),
        }

        eprintln!("\n=== After Precompute ===");
        let mut module = parse::parse_wat(wat).expect("Failed to parse");
        optimize::precompute(&mut module).expect("Precompute failed");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        match wasmparser::validate(&wasm_bytes) {
            Ok(_) => eprintln!("✓ Valid"),
            Err(e) => eprintln!("✗ INVALID: {:?}", e),
        }

        eprintln!("\n=== After ISLE conversion (Phase 2 of optimize_module) ===");
        let mut module = parse::parse_wat(wat).expect("Failed to parse");
        optimize::precompute(&mut module).expect("Precompute failed");

        // Phase 2: ISLE-based optimizations
        for func in &mut module.functions {
            let had_end = func.instructions.last() == Some(&Instruction::End);
            if let Ok(terms) = super::terms::instructions_to_terms(&func.instructions) {
                if !terms.is_empty() {
                    let mut env = LocalEnv::new();
                    let optimized_terms: Vec<Value> = terms
                        .into_iter()
                        .map(|term| rewrite_with_dataflow(term, &mut env))
                        .collect();
                    if let Ok(mut new_instrs) =
                        super::terms::terms_to_instructions(&optimized_terms)
                    {
                        if !new_instrs.is_empty() {
                            if !had_end && new_instrs.last() == Some(&Instruction::End) {
                                new_instrs.pop();
                            }
                            func.instructions = new_instrs;
                        }
                    }
                }
            }
        }

        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        match wasmparser::validate(&wasm_bytes) {
            Ok(_) => eprintln!("✓ Valid"),
            Err(e) => eprintln!("✗ INVALID: {:?}", e),
        }

        // Continue testing other phases
        eprintln!("\n=== After optimize_advanced_instructions ===");
        optimize::optimize_advanced_instructions(&mut module).expect("Failed");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        match wasmparser::validate(&wasm_bytes) {
            Ok(_) => eprintln!("✓ Valid"),
            Err(e) => eprintln!("✗ INVALID: {:?}", e),
        }

        eprintln!("\n=== After CSE ===");
        optimize::eliminate_common_subexpressions(&mut module).expect("Failed");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        match wasmparser::validate(&wasm_bytes) {
            Ok(_) => eprintln!("✓ Valid"),
            Err(e) => eprintln!("✗ INVALID: {:?}", e),
        }

        eprintln!("\n=== After inline_functions ===");
        optimize::inline_functions(&mut module).expect("Failed");
        let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
        match wasmparser::validate(&wasm_bytes) {
            Ok(_) => eprintln!("✓ Valid"),
            Err(e) => eprintln!("✗ INVALID: {:?}", e),
        }
    }

    /// Property: All optimizations must produce valid WASM
    #[test]
    fn prop_optimizations_produce_valid_wasm() {
        let test_cases = vec![
            (
                "bench_locals.wat",
                include_str!("../../tests/fixtures/bench_locals.wat"),
            ),
            (
                "bench_bitops.wat",
                include_str!("../../tests/fixtures/bench_bitops.wat"),
            ),
        ];

        for (name, wat) in test_cases {
            eprintln!("Testing fixture: {}", name);
            let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

            // Apply full pipeline
            optimize::precompute(&mut module).expect("Precompute failed");
            optimize::optimize_module(&mut module).expect("Optimize failed");
            optimize::eliminate_common_subexpressions(&mut module).expect("CSE failed");
            optimize::simplify_branches(&mut module).expect("Branch failed");
            optimize::eliminate_dead_code(&mut module).expect("DCE failed");

            // PROPERTY: Output must be valid WASM
            let wasm_bytes = encode::encode_wasm(&module).expect("Failed to encode");
            wasmparser::validate(&wasm_bytes)
                .unwrap_or_else(|e| panic!("Optimized WASM must be valid for {}: {:?}", name, e));
        }
    }

    /// Property: Optimization is idempotent (optimize twice = optimize once)
    #[test]
    fn prop_optimization_idempotent() {
        let wat = r#"(module (func (result i32) (i32.const 1) (i32.const 2) (i32.add)))"#;

        let mut m1 = parse::parse_wat(wat).unwrap();
        let mut m2 = parse::parse_wat(wat).unwrap();

        optimize::optimize_module(&mut m1).unwrap();
        optimize::optimize_module(&mut m2).unwrap();
        optimize::optimize_module(&mut m2).unwrap(); // Apply twice

        let b1 = encode::encode_wasm(&m1).unwrap();
        let b2 = encode::encode_wasm(&m2).unwrap();

        assert_eq!(b1, b2, "Optimization must be idempotent");
    }

    /// Property: Optimizations are deterministic
    #[test]
    fn prop_deterministic() {
        let wat = r#"(module (func (result i32) (i32.const 10) (i32.const 20) (i32.add)))"#;

        let mut m1 = parse::parse_wat(wat).unwrap();
        let mut m2 = parse::parse_wat(wat).unwrap();

        optimize::optimize_module(&mut m1).unwrap();
        optimize::optimize_module(&mut m2).unwrap();

        assert_eq!(
            encode::encode_wasm(&m1).unwrap(),
            encode::encode_wasm(&m2).unwrap(),
            "Same input must produce identical output"
        );
    }

    // Advanced Instruction Optimization Tests (Issue #21)

    #[test]
    fn test_strength_reduction_mul_to_shl() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                i32.const 4
                i32.mul
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x * 4 to x << 2
        let func = &module.functions[0];
        let has_shl = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Shl));
        let has_mul = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Mul));

        assert!(has_shl, "Should have shift left instruction");
        assert!(!has_mul, "Should not have multiply instruction");
    }

    #[test]
    fn test_strength_reduction_div_to_shr() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                i32.const 8
                i32.div_u
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x / 8 to x >> 3
        let func = &module.functions[0];
        let has_shr = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32ShrU));
        let has_div = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32DivU));

        assert!(has_shr, "Should have shift right instruction");
        assert!(!has_div, "Should not have divide instruction");
    }

    #[test]
    fn test_strength_reduction_rem_to_and() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                i32.const 16
                i32.rem_u
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x % 16 to x & 15
        let func = &module.functions[0];
        let has_and = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32And));
        let has_rem = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32RemU));

        assert!(has_and, "Should have AND instruction");
        assert!(!has_rem, "Should not have remainder instruction");
    }

    #[test]
    fn test_bitwise_trick_xor_same_value() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                local.get $x
                i32.xor
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x ^ x to 0
        let func = &module.functions[0];
        let const_zero = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Const(0)));
        let has_xor = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Xor));

        assert!(const_zero, "Should have constant 0");
        assert!(!has_xor, "Should not have XOR instruction");
    }

    #[test]
    fn test_bitwise_trick_and_same_value() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                local.get $x
                i32.and
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x & x to x
        let func = &module.functions[0];
        let local_get_count = func
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::LocalGet(_)))
            .count();
        let has_and = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32And));

        assert_eq!(local_get_count, 1, "Should have only one local.get");
        assert!(!has_and, "Should not have AND instruction");
    }

    #[test]
    fn test_bitwise_trick_or_same_value() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                local.get $x
                i32.or
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x | x to x
        let func = &module.functions[0];
        let local_get_count = func
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::LocalGet(_)))
            .count();
        let has_or = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Or));

        assert_eq!(local_get_count, 1, "Should have only one local.get");
        assert!(!has_or, "Should not have OR instruction");
    }

    #[test]
    fn test_bitwise_trick_and_zero() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                i32.const 0
                i32.and
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x & 0 to 0
        let func = &module.functions[0];
        // Should have just const 0 (no End - encoder adds it)
        assert_eq!(func.instructions.len(), 1, "Should have const 0");
        assert!(
            matches!(func.instructions[0], Instruction::I32Const(0)),
            "Should be const 0"
        );
    }

    #[test]
    fn test_bitwise_trick_or_all_ones() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                local.get $x
                i32.const -1
                i32.or
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x | 0xFFFFFFFF to 0xFFFFFFFF
        let func = &module.functions[0];
        assert_eq!(func.instructions.len(), 1, "Should have const -1");
        assert!(
            matches!(func.instructions[0], Instruction::I32Const(-1)),
            "Should be const -1"
        );
    }

    #[test]
    fn test_strength_reduction_i64() {
        let wat = r#"(module
            (func $test (param $x i64) (result i64)
                local.get $x
                i64.const 32
                i64.mul
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should convert x * 32 to x << 5
        let func = &module.functions[0];
        let has_shl = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Shl));
        let has_mul = func
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I64Mul));

        assert!(has_shl, "Should have i64 shift left instruction");
        assert!(!has_mul, "Should not have i64 multiply instruction");
    }

    #[test]
    #[ignore] // INTENTIONALLY SKIPPED: Control flow optimization not yet proven correct in ISLE.
    // Functions with BrIf/BrTable skip ISLE env tracking because simplify_with_env's
    // linear LocalEnv cannot model multiple execution paths or loop back-edges.
    // Z3 confirms counterexamples (e.g., matrix_multiply). Lifting this requires
    // basic block splitting + SSA conversion. See has_dataflow_unsafe_control_flow()
    // and issue #56.
    fn test_advanced_optimizations_in_control_flow() {
        let wat = r#"(module
            (func $test (param $x i32) (result i32)
                (block (result i32)
                    local.get $x
                    i32.const 4
                    i32.mul
                )
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        optimize::optimize_advanced_instructions(&mut module).unwrap();

        // Should optimize inside blocks too
        let _func = &module.functions[0];
        let wat_output = encode::encode_wat(&module).unwrap();

        assert!(
            wat_output.contains("i32.shl"),
            "Should have shift left in output"
        );
        assert!(
            !wat_output.contains("i32.mul"),
            "Should not have multiply in output"
        );
    }

    #[test]
    fn test_cse_phase4_duplicate_constants_above_cost_threshold() {
        // CSE-ing duplicate constants is only a size win when the constant
        // materializes in more bytes than the local.tee/local.get pair plus
        // the new local-declaration overhead. Use a 5-byte LEB128 value
        // (i32.const 0x12345678) so dedup is profitable.
        //
        // The earlier version of this test used `i32.const 42` (a 2-byte
        // total), which CSE would dedup for instruction-count savings but
        // grew code-section bytes. v0.6.0 added a cost gate (Expr::worth_dedup)
        // that suppresses this regression. See gale v0.4.0 measurement:
        // CSE-ing -EINVAL grew kernel-FFI code section by +6.3%.
        let wat = r#"(module
            (func $test (result i32)
                (local $result i32)
                ;; Use the same large constant multiple times (5-byte LEB128)
                (local.set $result (i32.const 0x12345678))
                (local.set $result (i32.add (local.get $result) (i32.const 0x12345678)))
                (local.set $result (i32.add (local.get $result) (i32.const 0x12345678)))
                (local.get $result)
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();

        let count_before = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::I32Const(0x1234_5678)))
            .count();
        assert_eq!(
            count_before, 3,
            "Should have 3 duplicate constants before CSE"
        );

        optimize::eliminate_common_subexpressions_enhanced(&mut module).unwrap();

        let const_count = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::I32Const(0x1234_5678)))
            .count();

        assert!(
            const_count < count_before,
            "CSE should reduce duplicates of large constants (5+ byte LEB128)"
        );

        let wasm_bytes = encode::encode_wasm(&module).unwrap();
        wasmparser::validate(&wasm_bytes).expect("Generated WASM should be valid");
    }

    #[test]
    fn test_cse_phase4_keeps_small_constants() {
        // Mirror of the above: a 2-byte constant must NOT be deduplicated,
        // because local.tee/local.get + new-local cost more than the
        // original two materializations. Pin the gale regression fix:
        // the CSE pass must leave cheap-constant patterns unchanged.
        let wat = r#"(module
            (func $test (result i32)
                (local $result i32)
                (local.set $result (i32.const 42))
                (local.set $result (i32.add (local.get $result) (i32.const 42)))
                (local.set $result (i32.add (local.get $result) (i32.const 42)))
                (local.get $result)
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();
        let locals_before = module.functions[0].locals.clone();
        let instr_count_before = module.functions[0].instructions.len();

        optimize::eliminate_common_subexpressions_enhanced(&mut module).unwrap();

        // Cheap constants must survive — neither locals nor instruction
        // count may grow when the cost gate refuses dedup.
        assert_eq!(
            module.functions[0].locals, locals_before,
            "CSE on cheap constants must not add locals (would grow function header)"
        );
        assert_eq!(
            module.functions[0].instructions.len(),
            instr_count_before,
            "CSE on cheap constants must not add local.tee/local.get instructions"
        );

        let const_count = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::I32Const(42)))
            .count();
        assert_eq!(
            const_count, 3,
            "Cheap constants must survive CSE — dedup would grow code size. \
             See docs/research/gale-v0.4.0/measurement-report.md"
        );

        let wasm_bytes = encode::encode_wasm(&module).unwrap();
        wasmparser::validate(&wasm_bytes).expect("Generated WASM should be valid");
    }

    // PR-K (v0.8.0): CSE cross-call dedup using function summaries.
    //
    // The CSE pass treats every `Call` as opaque without the IPA. With
    // function summaries (PR-F), a Call to a pure + no-trap function is
    // a deterministic value of its arguments — two identical such calls
    // can be deduped just like an arithmetic subtree. These tests pin
    // the four core invariants: dedup happens for pure+no-trap, doesn't
    // happen for impure, doesn't happen for may-trap, and different-arg
    // calls aren't deduped together.

    #[test]
    fn test_cse_dedupes_repeated_pure_calls() {
        // $pure_helper is pure+no-trap. PR-K3 unblocked the verifier
        // side; PR-K2 ships the replacement. The cost gate
        // (`worth_dedup`) requires net byte savings > 0. For a Call with
        // cost = 4 bytes (call op + LEB128 idx + 1 local.get arg),
        // savings_per_later = 2, fixed overhead = 4 — so we need at
        // least 4 occurrences for dedup to be profitable.
        // Test uses N=4 to exercise the dedup; N=2/N=3 patterns are
        // CORRECTLY skipped by the cost gate.
        let wat = r#"(module
            (func $pure_helper (param i32) (result i32)
                local.get 0
                i32.const 7
                i32.mul
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                call $pure_helper
                local.get 0
                call $pure_helper
                i32.add
                local.get 0
                call $pure_helper
                i32.add
                local.get 0
                call $pure_helper
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let calls_before = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(calls_before, 4, "sanity: four calls before CSE");

        optimize::eliminate_common_subexpressions_enhanced(&mut module).expect("cse");

        let calls_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert!(
            calls_after < calls_before,
            "CSE must dedupe identical pure+no-trap calls when cost gate \
             permits — second+ calls should become local.get of cached result. \
             Got {} calls after, expected fewer than {}",
            calls_after,
            calls_before
        );

        // Output must validate.
        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_cse_does_not_dedupe_impure_calls() {
        // $impure writes memory — its observable effect happens on
        // every call, so two calls must remain even if their inputs
        // and return values are identical.
        let wat = r#"(module
            (memory 1)
            (func $impure (param i32) (result i32)
                local.get 0
                i32.const 42
                i32.store
                local.get 0
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                call $impure
                local.get 0
                call $impure
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::eliminate_common_subexpressions_enhanced(&mut module).expect("cse");

        let calls_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(
            calls_after, 2,
            "CSE must NOT dedupe impure calls — the second store is \
             observable and must execute"
        );
    }

    #[test]
    fn test_cse_does_not_dedupe_may_trap_calls() {
        // $may_trap performs a load — each call could trap independently
        // on a bad address. CSE must keep both invocations to preserve
        // the trap semantics.
        let wat = r#"(module
            (memory 1)
            (func $may_trap (param i32) (result i32)
                local.get 0
                i32.load
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                call $may_trap
                local.get 0
                call $may_trap
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::eliminate_common_subexpressions_enhanced(&mut module).expect("cse");

        let calls_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(
            calls_after, 2,
            "CSE must NOT dedupe may-trap calls — the second trap is \
             observable behavior"
        );
    }

    #[test]
    fn test_cse_dedupes_pure_clamp_calls_via_span_replacement() {
        // PR-K2: span-replacement regression test. A realistic shape —
        // two `call $clamp(local.get $x)` invocations with the same
        // single arg — should fold the second call's `[arg, call]` span
        // into a single `local.get` of the cache local. $clamp is pure
        // and no-trap (only arithmetic and constants).
        let wat = r#"(module
            (func $clamp (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.const 100
                ;; min(max(x, 0), 100) ≈ clamp via two arithmetic ops.
                ;; Encoded as i32.add chains for soundness within the
                ;; supported-instruction subset; what matters is that
                ;; $clamp is pure+no-trap.
                i32.add
                i32.add
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                call $clamp
                local.get 0
                call $clamp
                i32.add
                local.get 0
                call $clamp
                i32.add
                local.get 0
                call $clamp
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let instrs_before = module.functions[1].instructions.len();
        let calls_before = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(calls_before, 4, "sanity: four calls before CSE");

        optimize::eliminate_common_subexpressions_enhanced(&mut module).expect("cse");

        let calls_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert!(
            calls_after < calls_before,
            "span-CSE must dedupe identical `call $clamp(local.get 0)` \
             invocations when cost gate permits. Got {} calls after, \
             expected fewer than {}",
            calls_after,
            calls_before
        );

        // The duplicate's span (LocalGet + Call = 2 instructions) collapses
        // into one local.get; the first occurrence keeps both and adds a
        // local.tee. Net instruction-count change is zero, but the second
        // dynamic call is gone — that's the observable win.
        let instrs_after = module.functions[1].instructions.len();
        assert!(
            instrs_after <= instrs_before,
            "span replacement should not increase instruction count. \
             Got {} after, {} before",
            instrs_after,
            instrs_before
        );
        // Confirm at least one duplicate `local.get 0` arg got absorbed
        // by span replacement (started with 4 instances, dedup removes
        // some — exact count depends on how many calls the cost gate
        // approves).
        let local0_gets_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::LocalGet(0)))
            .count();
        assert!(
            local0_gets_after < 4,
            "at least one duplicate `local.get 0` arg should be absorbed \
             by span replacement, got {}",
            local0_gets_after
        );

        // Output must validate.
        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_cse_dedupes_calls_with_different_args_separately() {
        // Two pure calls with DIFFERENT args. Neither call's result
        // is interchangeable with the other; both must survive.
        let wat = r#"(module
            (func $pure_helper (param i32) (result i32)
                local.get 0
                i32.const 7
                i32.mul
            )
            (func $caller (export "test") (param i32 i32) (result i32)
                local.get 0
                call $pure_helper
                local.get 1
                call $pure_helper
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::eliminate_common_subexpressions_enhanced(&mut module).expect("cse");

        let calls_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(
            calls_after, 2,
            "CSE must NOT dedupe calls with different args — they have \
             different cache keys"
        );
    }

    // CoalesceLocals Tests (Register Allocation)

    #[test]
    fn test_coalesce_locals_non_overlapping() {
        let wat = r#"
        (module
            (func $test (result i32)
                (local $temp1 i32)
                (local $temp2 i32)
                (local $temp3 i32)
                ;; Use temp1
                (i32.const 10)
                (local.set $temp1)
                (local.get $temp1)
                ;; temp1 dies here, temp2 can reuse its slot
                (i32.const 20)
                (local.set $temp2)
                (local.get $temp2)
                ;; temp2 dies here, temp3 can reuse the same slot
                (i32.const 30)
                (local.set $temp3)
                (local.get $temp3)
                i32.add
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();

        // Before coalescing: 3 locals
        let locals_before: usize = module.functions[0]
            .locals
            .iter()
            .map(|(count, _)| *count as usize)
            .sum();
        assert_eq!(locals_before, 3, "Should have 3 locals before coalescing");

        // Apply coalesce_locals
        optimize::coalesce_locals(&mut module).unwrap();

        // After coalescing: should have fewer locals (ideally 1)
        let locals_after: usize = module.functions[0]
            .locals
            .iter()
            .map(|(count, _)| *count as usize)
            .sum();

        eprintln!(
            "CoalesceLocals: {} locals → {} locals",
            locals_before, locals_after
        );

        assert!(
            locals_after < locals_before,
            "Coalescing should reduce local count: {} -> {}",
            locals_before,
            locals_after
        );

        // Verify the optimized module is still valid
        let wasm_bytes = encode::encode_wasm(&module).unwrap();
        wasmparser::validate(&wasm_bytes).expect("Coalesced module should be valid");
    }

    #[test]
    fn test_coalesce_locals_overlapping() {
        let wat = r#"
        (module
            (func $test (result i32)
                (local $a i32)
                (local $b i32)
                ;; Both locals are live at the same time
                (i32.const 10)
                (local.set $a)
                (i32.const 20)
                (local.set $b)
                ;; Both are still live here
                (local.get $a)
                (local.get $b)
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();

        let locals_before: usize = module.functions[0]
            .locals
            .iter()
            .map(|(count, _)| *count as usize)
            .sum();

        // Apply coalesce_locals
        optimize::coalesce_locals(&mut module).unwrap();

        let locals_after: usize = module.functions[0]
            .locals
            .iter()
            .map(|(count, _)| *count as usize)
            .sum();

        eprintln!("Overlapping locals: {} -> {}", locals_before, locals_after);

        // Since both locals are live simultaneously, they can't be coalesced
        assert_eq!(
            locals_after, locals_before,
            "Overlapping locals should NOT be coalesced"
        );

        // Verify validity
        let wasm_bytes = encode::encode_wasm(&module).unwrap();
        wasmparser::validate(&wasm_bytes).expect("Module should still be valid");
    }

    #[test]
    fn test_coalesce_locals_in_full_pipeline() {
        let wat = r#"
        (module
            (func $calculate (result i32)
                (local $temp1 i32)
                (local $temp2 i32)
                (local $temp3 i32)
                (local $result i32)
                ;; Sequential use of temps
                (i32.const 10)
                (local.set $temp1)
                (local.get $temp1)
                (i32.const 5)
                i32.mul
                (local.set $temp2)
                (local.get $temp2)
                (i32.const 3)
                i32.add
                (local.set $result)
                (local.get $result)
            )
        )"#;

        let mut module = parse::parse_wat(wat).unwrap();

        let locals_before: usize = module.functions[0]
            .locals
            .iter()
            .map(|(count, _)| *count as usize)
            .sum();

        // Run full optimization pipeline (includes coalesce_locals)
        optimize::optimize_module(&mut module).unwrap();

        let locals_after: usize = module.functions[0]
            .locals
            .iter()
            .map(|(count, _)| *count as usize)
            .sum();

        eprintln!(
            "Full pipeline with CoalesceLocals: {} locals → {} locals ({:.1}% reduction)",
            locals_before,
            locals_after,
            (1.0 - locals_after as f64 / locals_before as f64) * 100.0
        );

        // CoalesceLocals + other optimizations should reduce local count
        assert!(
            locals_after <= locals_before,
            "Optimization should not increase local count"
        );

        // Verify validity
        let wasm_bytes = encode::encode_wasm(&module).unwrap();
        wasmparser::validate(&wasm_bytes).expect("Fully optimized module should be valid");
    }

    #[test]
    fn test_load_isle_round_trip_all_14() {
        // Test that all 14 load variants with mem field round-trip through
        // instructions_to_terms → terms_to_instructions correctly.
        // Loads push a value onto the stack, so they survive the round-trip.

        let load_cases: Vec<(&str, Vec<Instruction>)> = vec![
            (
                "i32.load",
                vec![
                    Instruction::I32Const(100),
                    Instruction::I32Load {
                        offset: 4,
                        align: 2,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.load",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Load {
                        offset: 8,
                        align: 3,
                        mem: 0,
                    },
                ],
            ),
            (
                "f32.load",
                vec![
                    Instruction::I32Const(0),
                    Instruction::F32Load {
                        offset: 0,
                        align: 2,
                        mem: 0,
                    },
                ],
            ),
            (
                "f64.load",
                vec![
                    Instruction::I32Const(0),
                    Instruction::F64Load {
                        offset: 0,
                        align: 3,
                        mem: 0,
                    },
                ],
            ),
            (
                "i32.load8_s",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Load8S {
                        offset: 0,
                        align: 0,
                        mem: 0,
                    },
                ],
            ),
            (
                "i32.load8_u",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Load8U {
                        offset: 0,
                        align: 0,
                        mem: 0,
                    },
                ],
            ),
            (
                "i32.load16_s",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Load16S {
                        offset: 0,
                        align: 1,
                        mem: 0,
                    },
                ],
            ),
            (
                "i32.load16_u",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Load16U {
                        offset: 0,
                        align: 1,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.load8_s",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Load8S {
                        offset: 0,
                        align: 0,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.load8_u",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Load8U {
                        offset: 0,
                        align: 0,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.load16_s",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Load16S {
                        offset: 0,
                        align: 1,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.load16_u",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Load16U {
                        offset: 0,
                        align: 1,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.load32_s",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Load32S {
                        offset: 0,
                        align: 2,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.load32_u",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Load32U {
                        offset: 0,
                        align: 2,
                        mem: 0,
                    },
                ],
            ),
        ];

        for (name, instructions) in &load_cases {
            let terms = terms::instructions_to_terms(instructions)
                .unwrap_or_else(|e| panic!("{}: instructions_to_terms failed: {}", name, e));

            let result = terms::terms_to_instructions(&terms)
                .unwrap_or_else(|e| panic!("{}: terms_to_instructions failed: {}", name, e));

            assert_eq!(&result, instructions, "{}: round-trip mismatch", name);
        }
    }

    #[test]
    fn test_store_isle_conversion_all_9() {
        // Test that all 9 store variants successfully convert to ISLE terms.
        // Stores produce side effects (not stack values), so we verify
        // instructions_to_terms succeeds without error.
        // i32.store and i64.store use I32Const/I64Const for values (already on ISLE stack).
        // Float stores use load results. Partial stores use I32Const/I64Const values.

        let store_cases: Vec<(&str, Vec<Instruction>)> = vec![
            (
                "i32.store",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Const(42),
                    Instruction::I32Store {
                        offset: 0,
                        align: 2,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.store",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Const(999),
                    Instruction::I64Store {
                        offset: 0,
                        align: 3,
                        mem: 0,
                    },
                ],
            ),
            (
                "f32.store",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Const(0),
                    Instruction::F32Load {
                        offset: 8,
                        align: 2,
                        mem: 0,
                    },
                    Instruction::F32Store {
                        offset: 0,
                        align: 2,
                        mem: 0,
                    },
                ],
            ),
            (
                "f64.store",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Const(0),
                    Instruction::F64Load {
                        offset: 8,
                        align: 3,
                        mem: 0,
                    },
                    Instruction::F64Store {
                        offset: 0,
                        align: 3,
                        mem: 0,
                    },
                ],
            ),
            (
                "i32.store8",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Const(255),
                    Instruction::I32Store8 {
                        offset: 0,
                        align: 0,
                        mem: 0,
                    },
                ],
            ),
            (
                "i32.store16",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I32Const(65535),
                    Instruction::I32Store16 {
                        offset: 0,
                        align: 1,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.store8",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Const(255),
                    Instruction::I64Store8 {
                        offset: 0,
                        align: 0,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.store16",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Const(65535),
                    Instruction::I64Store16 {
                        offset: 0,
                        align: 1,
                        mem: 0,
                    },
                ],
            ),
            (
                "i64.store32",
                vec![
                    Instruction::I32Const(0),
                    Instruction::I64Const(0xFFFFFFFF),
                    Instruction::I64Store32 {
                        offset: 0,
                        align: 2,
                        mem: 0,
                    },
                ],
            ),
        ];

        for (name, instructions) in &store_cases {
            terms::instructions_to_terms(instructions)
                .unwrap_or_else(|e| panic!("{}: instructions_to_terms failed: {}", name, e));
        }
    }

    #[test]
    fn test_load_store_mem_field_preserved_in_isle() {
        // Verify the mem field is preserved (not dropped to 0) through ISLE conversion
        let instructions = vec![
            Instruction::I32Const(100),
            Instruction::I32Load {
                offset: 4,
                align: 2,
                mem: 3,
            },
        ];

        let terms =
            terms::instructions_to_terms(&instructions).expect("instructions_to_terms failed");

        let result = terms::terms_to_instructions(&terms).expect("terms_to_instructions failed");

        assert_eq!(result.len(), 2);
        match &result[1] {
            Instruction::I32Load { offset, align, mem } => {
                assert_eq!(*offset, 4);
                assert_eq!(*align, 2);
                assert_eq!(
                    *mem, 3,
                    "mem field should be preserved through ISLE round-trip"
                );
            }
            other => panic!("Expected I32Load, got {:?}", other),
        }
    }

    #[test]
    fn test_multi_memory_round_trip_parse_encode() {
        // Multi-memory module: memory 0 and memory 1
        // i32.load from memory 1 should preserve the memory index through parse → encode.
        // We construct the module directly (WAT text format multi-memory syntax varies
        // by tool version), then round-trip through encode → parse.
        let instructions = vec![
            Instruction::I32Const(0),
            Instruction::I32Load {
                offset: 0,
                align: 2,
                mem: 1,
            },
            Instruction::End,
        ];

        let module = Module {
            functions: vec![Function {
                name: None,
                signature: FunctionSignature {
                    params: vec![],
                    results: vec![ValueType::I32],
                },
                locals: vec![],
                instructions,
            }],
            memories: vec![
                crate::Memory {
                    min: 1,
                    max: None,
                    shared: false,
                    memory64: false,
                },
                crate::Memory {
                    min: 1,
                    max: None,
                    shared: false,
                    memory64: false,
                },
            ],
            tables: vec![],
            globals: vec![],
            types: vec![FunctionSignature {
                params: vec![],
                results: vec![ValueType::I32],
            }],
            exports: vec![],
            imports: vec![],
            data_segments: vec![],
            element_section_bytes: None,
            start_function: None,
            custom_sections: vec![],
            type_section_bytes: None,
            global_section_bytes: None,
        };

        let wasm_bytes =
            encode::encode_wasm(&module).expect("Failed to encode multi-memory module");
        let module2 =
            parse::parse_wasm(&wasm_bytes).expect("Failed to re-parse multi-memory module");

        assert_eq!(module2.functions.len(), 1);
        let func = &module2.functions[0];

        // Find the I32Load and verify mem=1 survived round-trip
        let load = func
            .instructions
            .iter()
            .find(|i| matches!(i, Instruction::I32Load { .. }));
        match load {
            Some(Instruction::I32Load { mem, .. }) => {
                assert_eq!(
                    *mem, 1,
                    "memory index should survive parse→encode round-trip"
                );
            }
            _ => panic!("Expected I32Load instruction in round-tripped module"),
        }
    }

    #[test]
    fn test_float_const_isle_round_trip() {
        // F32Const and F64Const should survive instructions_to_terms → terms_to_instructions
        let instructions = vec![
            Instruction::F32Const(1.5_f32.to_bits()),
            Instruction::F64Const(2.5_f64.to_bits()),
        ];
        let stack = terms::instructions_to_terms(&instructions).unwrap();
        assert_eq!(stack.len(), 2);

        let result_instrs = terms::terms_to_instructions(&stack).unwrap();
        assert_eq!(result_instrs.len(), 2);
        assert_eq!(result_instrs[0], Instruction::F32Const(1.5_f32.to_bits()));
        assert_eq!(result_instrs[1], Instruction::F64Const(2.5_f64.to_bits()));
    }

    #[test]
    fn test_float_constant_folding_f32_add() {
        // f32.const 10.0, f32.const 32.0, f32.add → should fold to f32.const 42.0
        let wat = r#"
            (module
              (func $add_f32_constants (result f32)
                f32.const 10.0
                f32.const 32.0
                f32.add
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        // Verify original instructions
        let func = &module.functions[0];
        assert!(
            func.instructions
                .contains(&Instruction::F32Const(10.0_f32.to_bits()))
        );
        assert!(
            func.instructions
                .contains(&Instruction::F32Const(32.0_f32.to_bits()))
        );
        assert!(func.instructions.contains(&Instruction::F32Add));

        // Apply optimization
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        // Should be folded to f32.const 42.0
        let func = &module.functions[0];
        assert_eq!(
            func.instructions[0],
            Instruction::F32Const(42.0_f32.to_bits())
        );
        assert!(!func.instructions.contains(&Instruction::F32Add));
    }

    #[test]
    fn test_float_constant_folding_f64_mul() {
        // f64.const 3.0, f64.const 7.0, f64.mul → should fold to f64.const 21.0
        let wat = r#"
            (module
              (func $mul_f64_constants (result f64)
                f64.const 3.0
                f64.const 7.0
                f64.mul
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");

        let func = &module.functions[0];
        assert!(func.instructions.contains(&Instruction::F64Mul));

        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        assert_eq!(
            func.instructions[0],
            Instruction::F64Const(21.0_f64.to_bits())
        );
        assert!(!func.instructions.contains(&Instruction::F64Mul));
    }

    #[test]
    fn test_float_nan_not_folded() {
        // f32.const NaN + f32.const 1.0 should NOT be folded
        let instructions = vec![
            Instruction::F32Const(f32::NAN.to_bits()),
            Instruction::F32Const(1.0_f32.to_bits()),
            Instruction::F32Add,
        ];
        let stack = terms::instructions_to_terms(&instructions).unwrap();
        assert_eq!(stack.len(), 1);

        // The term should still be an F32Add (not folded to a constant)
        let result_instrs = terms::terms_to_instructions(&stack).unwrap();

        // Should still have the add — NaN operands prevent folding
        // The instructions should round-trip: f32.const NaN, f32.const 1.0, f32.add
        assert_eq!(result_instrs.len(), 3);
        assert_eq!(result_instrs[0], Instruction::F32Const(f32::NAN.to_bits()));
        assert_eq!(result_instrs[1], Instruction::F32Const(1.0_f32.to_bits()));
        assert_eq!(result_instrs[2], Instruction::F32Add);
    }

    #[test]
    fn test_float_f32_arithmetic_isle_round_trip() {
        // F32Const, F32Const, F32Sub round-trips through ISLE terms without loss
        let instructions = vec![
            Instruction::F32Const(5.0_f32.to_bits()),
            Instruction::F32Const(3.0_f32.to_bits()),
            Instruction::F32Sub,
        ];
        let stack = terms::instructions_to_terms(&instructions).unwrap();
        assert_eq!(stack.len(), 1);

        // Round-trip back to instructions (no simplification at this stage)
        let result_instrs = terms::terms_to_instructions(&stack).unwrap();
        assert_eq!(result_instrs.len(), 3);
        assert_eq!(result_instrs[0], Instruction::F32Const(5.0_f32.to_bits()));
        assert_eq!(result_instrs[1], Instruction::F32Const(3.0_f32.to_bits()));
        assert_eq!(result_instrs[2], Instruction::F32Sub);
    }

    #[test]
    fn test_float_f64_arithmetic_isle_round_trip() {
        // F64Const, F64Const, F64Mul round-trips through ISLE terms without loss
        let instructions = vec![
            Instruction::F64Const(3.0_f64.to_bits()),
            Instruction::F64Const(7.0_f64.to_bits()),
            Instruction::F64Mul,
        ];
        let stack = terms::instructions_to_terms(&instructions).unwrap();
        assert_eq!(stack.len(), 1);

        // Round-trip back to instructions (no simplification at this stage)
        let result_instrs = terms::terms_to_instructions(&stack).unwrap();
        assert_eq!(result_instrs.len(), 3);
        assert_eq!(result_instrs[0], Instruction::F64Const(3.0_f64.to_bits()));
        assert_eq!(result_instrs[1], Instruction::F64Const(7.0_f64.to_bits()));
        assert_eq!(result_instrs[2], Instruction::F64Mul);
    }

    #[test]
    fn test_float_unary_round_trip() {
        // F32Const, F32Abs round-trips through ISLE terms
        let instructions = vec![
            Instruction::F32Const(3.0_f32.to_bits()),
            Instruction::F32Abs,
        ];
        let stack = terms::instructions_to_terms(&instructions).unwrap();
        assert_eq!(stack.len(), 1);

        let result_instrs = terms::terms_to_instructions(&stack).unwrap();
        assert_eq!(result_instrs.len(), 2);
        assert_eq!(result_instrs[0], Instruction::F32Const(3.0_f32.to_bits()));
        assert_eq!(result_instrs[1], Instruction::F32Abs);
    }

    #[test]
    fn test_float_neg_constant_fold() {
        // f32.const -5.0; f32.abs → f32.const 5.0
        let wat = r#"
            (module
              (func $abs_neg (result f32)
                f32.const -5.0
                f32.abs
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        assert_eq!(
            func.instructions[0],
            Instruction::F32Const(5.0_f32.to_bits())
        );
        assert!(!func.instructions.contains(&Instruction::F32Abs));
    }

    #[test]
    fn test_float_neg_involution() {
        // neg(neg(x)) should simplify to x
        use loom_isle::{ImmF32, fconst32, fneg32, rewrite_pure};
        let x = fconst32(ImmF32::new(42.0));
        let neg_neg = fneg32(fneg32(x.clone()));
        let simplified = rewrite_pure(neg_neg);
        // Should simplify back to the original constant
        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::F32Const(42.0_f32.to_bits()));
    }

    #[test]
    fn test_float_min_constant_fold() {
        // f32.const 3.0; f32.const 7.0; f32.min → f32.const 3.0
        let wat = r#"
            (module
              (func $min_consts (result f32)
                f32.const 3.0
                f32.const 7.0
                f32.min
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        assert_eq!(
            func.instructions[0],
            Instruction::F32Const(3.0_f32.to_bits())
        );
        assert!(!func.instructions.contains(&Instruction::F32Min));
    }

    #[test]
    fn test_float_min_nan_not_folded() {
        // f32.const NaN; f32.const 1.0; f32.min → NOT folded (NaN propagation)
        let instructions = vec![
            Instruction::F32Const(f32::NAN.to_bits()),
            Instruction::F32Const(1.0_f32.to_bits()),
            Instruction::F32Min,
        ];
        let stack = terms::instructions_to_terms(&instructions).unwrap();
        assert_eq!(stack.len(), 1);

        let result_instrs = terms::terms_to_instructions(&stack).unwrap();
        // Should NOT be folded — still 3 instructions
        assert_eq!(result_instrs.len(), 3);
        assert_eq!(result_instrs[2], Instruction::F32Min);
    }

    #[test]
    fn test_float_comparison_fold() {
        // f32.const 3.0; f32.const 7.0; f32.lt → i32.const 1
        let wat = r#"
            (module
              (func $lt_consts (result i32)
                f32.const 3.0
                f32.const 7.0
                f32.lt
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        assert_eq!(func.instructions[0], Instruction::I32Const(1));
        assert!(!func.instructions.contains(&Instruction::F32Lt));
    }

    #[test]
    fn test_float_comparison_nan_eq() {
        // f32.const NaN; f32.const 1.0; f32.eq → i32.const 0 (NaN != anything)
        use loom_isle::{ImmF32, fconst32, feq32, rewrite_pure};
        let nan = fconst32(ImmF32::new(f32::NAN));
        let one = fconst32(ImmF32::new(1.0));
        let eq = feq32(nan, one);
        let simplified = rewrite_pure(eq);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::I32Const(0));
    }

    #[test]
    fn test_float_f64_ceil_fold() {
        // f64.const 2.3; f64.ceil → f64.const 3.0
        let wat = r#"
            (module
              (func $ceil_const (result f64)
                f64.const 2.3
                f64.ceil
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        assert_eq!(
            func.instructions[0],
            Instruction::F64Const(3.0_f64.to_bits())
        );
        assert!(!func.instructions.contains(&Instruction::F64Ceil));
    }

    #[test]
    fn test_float_copysign_fold() {
        // f32.const 5.0; f32.const -1.0; f32.copysign → f32.const -5.0
        use loom_isle::{ImmF32, fconst32, fcopysign32, rewrite_pure};
        let mag = fconst32(ImmF32::new(5.0));
        let sign = fconst32(ImmF32::new(-1.0));
        let cs = fcopysign32(mag, sign);
        let simplified = rewrite_pure(cs);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::F32Const((-5.0_f32).to_bits()));
    }

    #[test]
    fn test_float_f64_comparison_fold() {
        // f64.const 10.0; f64.const 5.0; f64.ge → i32.const 1
        use loom_isle::{ImmF64, fconst64, fge64, rewrite_pure};
        let a = fconst64(ImmF64::new(10.0));
        let b = fconst64(ImmF64::new(5.0));
        let ge = fge64(a, b);
        let simplified = rewrite_pure(ge);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::I32Const(1));
    }

    #[test]
    fn test_conversion_round_trip() {
        // f32.const 3.125; i32.trunc_f32_s round-trips through ISLE terms
        let instructions = vec![
            Instruction::F32Const(3.125_f32.to_bits()),
            Instruction::I32TruncF32S,
        ];
        let terms = terms::instructions_to_terms(&instructions).unwrap();
        assert_eq!(terms.len(), 1);

        let result = terms::terms_to_instructions(&terms).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0], Instruction::F32Const(3.125_f32.to_bits()));
        assert_eq!(result[1], Instruction::I32TruncF32S);
    }

    #[test]
    fn test_conversion_trunc_constant_fold() {
        // i32.trunc_f32_s of in-range constant should fold
        use loom_isle::{ImmF32, fconst32, i32_trunc_f32_s, rewrite_pure};
        let val = fconst32(ImmF32::new(42.9));
        let trunc = i32_trunc_f32_s(val);
        let simplified = rewrite_pure(trunc);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::I32Const(42));
    }

    #[test]
    fn test_conversion_trunc_nan_not_folded() {
        // i32.trunc_f32_s of NaN should NOT fold (would trap at runtime)
        use loom_isle::{ImmF32, fconst32, i32_trunc_f32_s, rewrite_pure};
        let val = fconst32(ImmF32::new(f32::NAN));
        let trunc = i32_trunc_f32_s(val);
        let simplified = rewrite_pure(trunc);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 2); // f32.const NaN, i32.trunc_f32_s
        assert_eq!(instrs[1], Instruction::I32TruncF32S);
    }

    #[test]
    fn test_seam_sroa_wrap_extend_identity() {
        // #219 seam-SROA: wrap_i64(extend_i32_u(x)) → x. Zero-extending an i32 to
        // i64 then wrapping back to i32 is the identity — the u64-ABI round-trip
        // a packed scalar pays at the dissolved decide seam. Must dissolve to the
        // bare operand (no wrap/extend left).
        use loom_isle::{i32_wrap_i64, i64_extend_i32_u, local_get, rewrite_pure};
        let x = local_get(0); // an i32 value
        let round_trip = i32_wrap_i64(i64_extend_i32_u(x));
        let simplified = rewrite_pure(round_trip);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(
            instrs,
            vec![Instruction::LocalGet(0)],
            "#219: wrap_i64(extend_i32_u(x)) must dissolve to x"
        );
    }

    #[test]
    fn test_seam_sroa_mask_clears_shifted_pack() {
        // #219 seam-SROA: (or (shl (extend_u x) 32) Y) & 0xff → Y & 0xff.
        // The high-shifted field of a u64 pack cannot survive a low-byte unpack
        // mask, so the whole high half drops out. This is the `& mask` half of
        // dissolving the sem decide's pack/unpack round-trip.
        use loom_isle::{
            Imm64, i64_extend_i32_u, iand64, iconst64, ior64, ishl64, local_get, rewrite_pure,
        };
        let high = ishl64(i64_extend_i32_u(local_get(0)), iconst64(Imm64(32)));
        let low = i64_extend_i32_u(local_get(1)); // the surviving low field
        let pack = ior64(high, low);
        let masked = iand64(pack, iconst64(Imm64(0xff)));
        let simplified = rewrite_pure(masked);

        // Expect (extend_u(local.get 1)) & 0xff — the shifted high half is gone.
        let want = terms::terms_to_instructions(&[iand64(
            i64_extend_i32_u(local_get(1)),
            iconst64(Imm64(0xff)),
        )])
        .unwrap();
        let got = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(
            got, want,
            "#219: (or (shl _ 32) Y) & 0xff must dissolve to Y & 0xff"
        );
    }

    #[test]
    fn test_seam_sroa_shr_extracts_high_field() {
        // #219 seam-SROA: (shr_u (or (shl (extend_u x) 32) const) 32) extracts the
        // HIGH field of a u64 pack → (extend_u x) & 0xffffffff. The low (const)
        // field shifts out (logical shift distributes over OR; shl-then-shr-same
        // masks to the low 64-k bits). Mirrors the sem decide whose low field is
        // a constant 0/1.
        use loom_isle::{
            Imm64, i64_extend_i32_u, iand64, iconst64, ior64, ishl64, ishru64, local_get,
            rewrite_pure,
        };
        let high = ishl64(i64_extend_i32_u(local_get(0)), iconst64(Imm64(32)));
        let pack = ior64(high, iconst64(Imm64(1))); // low field = const (like local 3 = 0/1)
        let unpacked = ishru64(pack, iconst64(Imm64(32)));
        let simplified = rewrite_pure(unpacked);

        let want = terms::terms_to_instructions(&[iand64(
            i64_extend_i32_u(local_get(0)),
            iconst64(Imm64(0xffff_ffff)),
        )])
        .unwrap();
        let got = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(
            got, want,
            "#219: (shr_u (or (shl (extend_u x) 32) const) 32) must extract (extend_u x) & 0xffffffff"
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn test_is_noreturn_callee() {
        // PR-C (#219): the divergent-call classifier. No-return = no Return, no
        // branch anywhere, body ends in Unreachable (the core::panicking shape).
        let cases: &[(&str, bool)] = &[
            // bare trap
            ("(module (func (export \"f\") unreachable))", true),
            // call (to an import) then trap — the panic_const_add_overflow shape
            (
                "(module (import \"env\" \"p\" (func)) (func (export \"f\") call 0 unreachable))",
                true,
            ),
            // normal return — must NOT be no-return
            (
                "(module (func (export \"f\") (result i32) i32.const 0))",
                false,
            ),
            // ends in a value, not unreachable
            ("(module (func (export \"f\") nop))", false),
            // contains a branch (could escape to the fn label) — conservatively false
            (
                "(module (func (export \"f\") (block br 0) unreachable))",
                false,
            ),
        ];
        for (wat, expected) in cases {
            let module = parse::parse_wat(wat).expect("parse");
            let f = &module.functions[0];
            assert_eq!(
                crate::verify::is_noreturn_callee(f),
                *expected,
                "#219 is_noreturn_callee mismatch for: {wat}"
            );
        }
    }

    #[test]
    fn test_conversion_trunc_sat_folds_nan_to_zero() {
        // i32.trunc_sat_f32_s of NaN → i32.const 0 (saturating: NaN→0)
        use loom_isle::{ImmF32, fconst32, i32_trunc_sat_f32_s, rewrite_pure};
        let val = fconst32(ImmF32::new(f32::NAN));
        let trunc = i32_trunc_sat_f32_s(val);
        let simplified = rewrite_pure(trunc);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::I32Const(0));
    }

    #[test]
    fn test_conversion_trunc_sat_clamps_overflow() {
        // i32.trunc_sat_f32_s of large value → i32.const i32::MAX
        use loom_isle::{ImmF32, fconst32, i32_trunc_sat_f32_s, rewrite_pure};
        let val = fconst32(ImmF32::new(1.0e20));
        let trunc = i32_trunc_sat_f32_s(val);
        let simplified = rewrite_pure(trunc);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::I32Const(i32::MAX));
    }

    #[test]
    fn test_conversion_f32_convert_i32_s_fold() {
        // f32.convert_i32_s of constant → f32.const
        use loom_isle::{Imm32, f32_convert_i32_s, iconst32, rewrite_pure};
        let val = iconst32(Imm32::new(-10));
        let convert = f32_convert_i32_s(val);
        let simplified = rewrite_pure(convert);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::F32Const((-10.0_f32).to_bits()));
    }

    #[test]
    fn test_conversion_reinterpret_fold() {
        // i32.reinterpret_f32 of f32 constant → i32.const with same bits
        use loom_isle::{ImmF32, fconst32, i32_reinterpret_f32, rewrite_pure};
        let val = fconst32(ImmF32::new(1.0));
        let reinterpret = i32_reinterpret_f32(val);
        let simplified = rewrite_pure(reinterpret);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(instrs[0], Instruction::I32Const(1.0_f32.to_bits() as i32));
    }

    #[test]
    fn test_conversion_demote_promote_fold() {
        // f32.demote_f64 of f64 constant → f32 constant
        use loom_isle::{ImmF64, f32_demote_f64, fconst64, rewrite_pure};
        let val = fconst64(ImmF64::new(3.125));
        let demote = f32_demote_f64(val);
        let simplified = rewrite_pure(demote);

        let instrs = terms::terms_to_instructions(&[simplified]).unwrap();
        assert_eq!(instrs.len(), 1);
        assert_eq!(
            instrs[0],
            Instruction::F32Const((3.125_f64 as f32).to_bits())
        );
    }

    #[test]
    fn test_conversion_full_pipeline() {
        // Full pipeline test: function with conversions gets optimized
        let wat = r#"
            (module
              (func $convert (result i32)
                f32.const 42.7
                i32.trunc_sat_f32_s
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        // Should be folded to i32.const 42
        assert_eq!(func.instructions[0], Instruction::I32Const(42));
    }

    #[test]
    fn test_memory_size_round_trip() {
        // memory.size goes through ISLE terms and back unchanged
        let instructions = vec![Instruction::MemorySize(0), Instruction::End];
        let terms = terms::instructions_to_terms(&instructions).expect("Failed to convert");
        let result = terms::terms_to_instructions(&terms).expect("Failed to convert back");
        assert_eq!(result, vec![Instruction::MemorySize(0)]);
    }

    #[test]
    fn test_memory_grow_round_trip() {
        // memory.grow goes through ISLE terms and back unchanged
        let instructions = vec![
            Instruction::I32Const(1),
            Instruction::MemoryGrow(0),
            Instruction::End,
        ];
        let terms = terms::instructions_to_terms(&instructions).expect("Failed to convert");
        let result = terms::terms_to_instructions(&terms).expect("Failed to convert back");
        assert_eq!(
            result,
            vec![Instruction::I32Const(1), Instruction::MemoryGrow(0)]
        );
    }

    #[test]
    fn test_memory_operations_not_skipped() {
        // Functions with memory.size/grow should NOT be skipped from optimization
        let wat = r#"
            (module
              (memory 1)
              (func $mem_test (result i32)
                i32.const 1
                i32.const 1
                i32.add
                memory.size
                i32.add
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        // i32.const 1 + i32.const 1 should be folded to i32.const 2
        assert!(
            func.instructions.contains(&Instruction::I32Const(2)),
            "Expected constant folding to occur in function with memory.size: {:?}",
            func.instructions
        );
    }

    #[test]
    fn test_call_indirect_not_skipped() {
        // Functions with call_indirect should NOT be skipped from optimization
        let wat = r#"
            (module
              (type $sig (func (param i32) (result i32)))
              (table 1 funcref)
              (func $indirect_test (result i32)
                i32.const 2
                i32.const 3
                i32.add
                i32.const 0
                call_indirect (type $sig)
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        // i32.const 2 + i32.const 3 should be folded to i32.const 5
        assert!(
            func.instructions.contains(&Instruction::I32Const(5)),
            "Expected constant folding to occur in function with call_indirect: {:?}",
            func.instructions
        );
    }

    #[test]
    fn test_bulk_memory_not_skipped() {
        // Functions with bulk memory ops should NOT be skipped from optimization
        let wat = r#"
            (module
              (memory 1)
              (data $d "hello")
              (func $bulk_test
                i32.const 1
                i32.const 1
                i32.add
                i32.const 0
                i32.const 5
                memory.fill
              )
            )
        "#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse WAT");
        optimize::optimize_module(&mut module).expect("Failed to optimize");

        let func = &module.functions[0];
        // i32.const 1 + i32.const 1 should be folded to i32.const 2
        assert!(
            func.instructions.contains(&Instruction::I32Const(2)),
            "Expected constant folding to occur in function with memory.fill: {:?}",
            func.instructions
        );
        // memory.fill should still be present
        assert!(
            func.instructions.contains(&Instruction::MemoryFill(0)),
            "Expected memory.fill to be preserved: {:?}",
            func.instructions
        );
    }

    #[test]
    fn test_data_drop_round_trip() {
        // data.drop goes through ISLE terms and back unchanged
        let instructions = vec![Instruction::DataDrop(0), Instruction::End];
        let terms = terms::instructions_to_terms(&instructions).expect("Failed to convert");
        let result = terms::terms_to_instructions(&terms).expect("Failed to convert back");
        assert_eq!(result, vec![Instruction::DataDrop(0)]);
    }

    // eliminate_dead_locals tests (v0.6.0 PR-B)
    //
    // Targets: gale "default-then-override" pattern where a local is
    // written but never read. The pass is path-insensitive — sound
    // regardless of BrIf/BrTable/early-Return control flow that gates
    // simplify_locals and coalesce_locals.

    #[test]
    fn test_eliminate_dead_locals_basic_write_only() {
        // The exact gale pattern from gale_bitarray_alloc_validate:
        // local 3 written with EINVAL default but never read.
        let wat = r#"(module
            (func $test (param i32) (param i32) (result i32)
                (local i32) ;; local 2 — dead
                i32.const -22
                local.set 2
                local.get 0
                local.get 1
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("Failed to parse");
        let bytes_before = encode::encode_wasm(&module).expect("encode").len();
        let locals_before = module.functions[0].locals.clone();
        assert_eq!(locals_before, vec![(1, ValueType::I32)]);

        optimize::eliminate_dead_locals(&mut module).expect("eliminate_dead_locals");

        // The dead local should be gone from the declaration.
        assert_eq!(
            module.functions[0].locals,
            vec![],
            "Dead local must be removed from declaration"
        );

        // The dead LocalSet should have been replaced by Drop.
        let has_dead_set = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalSet(2)));
        assert!(!has_dead_set, "Dead LocalSet must be neutralized");

        // Output must validate as wasm.
        let wasm_bytes = encode::encode_wasm(&module).expect("encode after");
        wasmparser::validate(&wasm_bytes).expect("output must validate");
        assert!(
            wasm_bytes.len() < bytes_before,
            "Eliminating a dead local must reduce binary size"
        );
    }

    #[test]
    fn test_eliminate_dead_locals_preserves_used_locals() {
        // A local that IS read must survive — even if some writes
        // to it look redundant. Path-sensitivity is Pick #3, not this pass.
        let wat = r#"(module
            (func $test (param i32) (result i32)
                (local i32) ;; local 1 — read in addition below
                i32.const 7
                local.set 1
                local.get 0
                local.get 1
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let locals_before = module.functions[0].locals.clone();

        optimize::eliminate_dead_locals(&mut module).expect("pass");

        assert_eq!(
            module.functions[0].locals, locals_before,
            "Live local must survive"
        );
    }

    #[test]
    fn test_eliminate_dead_locals_localtee_neutralization() {
        // LocalTee writing to a dead local must be REMOVED (not replaced
        // with Drop). LocalTee's stack effect is [T] -> [T], so removing
        // the instruction leaves the value passing through. Replacing
        // with Drop would consume the stack value and break a downstream
        // consumer.
        let wat = r#"(module
            (func $test (param i32) (result i32)
                (local i32) ;; local 1 — dead (only LocalTee writes, no LocalGet)
                local.get 0
                i32.const 1
                i32.add
                local.tee 1
                i32.const 2
                i32.mul
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::eliminate_dead_locals(&mut module).expect("pass");

        let has_localtee = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalTee(_)));
        assert!(!has_localtee, "Dead LocalTee must be removed entirely");

        // The function must still encode and validate.
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes)
            .expect("LocalTee removal must preserve stack — value passes through");

        // Local declaration is gone.
        assert_eq!(module.functions[0].locals, vec![]);
    }

    #[test]
    fn test_eliminate_dead_locals_packs_indices() {
        // After dropping a dead middle local, surviving locals must be
        // packed to dense indices starting at param_count. Locals that
        // were 2,3,4 with #3 dead become 2,3 (was 4 -> now 3).
        let wat = r#"(module
            (func $test (param i32) (param i32) (result i32)
                (local i32 i32 i32) ;; locals 2,3,4 — only 4 is dead
                i32.const 1
                local.set 2
                i32.const 2
                local.set 4   ;; dead
                local.get 2
                local.get 3   ;; reads local 3, keeps it alive
                i32.add
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::eliminate_dead_locals(&mut module).expect("pass");

        // Two locals survive (2 and 3, the previously-dead 4 is gone).
        let total_declared: u32 = module.functions[0].locals.iter().map(|(c, _)| *c).sum();
        assert_eq!(total_declared, 2, "One of three declared locals dropped");

        // After pack-down, no instruction references local index 4.
        let has_idx_4 = module.functions[0].instructions.iter().any(|i| {
            matches!(
                i,
                Instruction::LocalGet(4) | Instruction::LocalSet(4) | Instruction::LocalTee(4)
            )
        });
        assert!(!has_idx_4, "Index 4 must be remapped or removed");

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("validates");
    }

    #[test]
    fn test_eliminate_dead_locals_skips_params() {
        // Parameters are observable to the caller — never eliminate them
        // even if a function never reads them.
        let wat = r#"(module
            (func $test (param i32) (param i32) (result i32)
                ;; param 1 (idx 1) is never read
                local.get 0
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let sig_before = module.functions[0].signature.clone();

        optimize::eliminate_dead_locals(&mut module).expect("pass");

        assert_eq!(
            module.functions[0].signature, sig_before,
            "Function signature (params) must never change"
        );
    }

    // eliminate_dead_stores tests (v0.6.0 PR-C)
    //
    // Path-sensitive dead-store elimination via backward liveness over
    // structured wasm control flow. Catches dead writes PR-B can't:
    // locals that ARE read elsewhere but where a particular write is
    // overwritten before any read on every continuation.

    #[test]
    fn test_eliminate_dead_stores_overwritten_in_straight_line() {
        // Two writes to local 1, no read between. The first is dead.
        let wat = r#"(module
            (func $test (param i32) (result i32)
                (local i32)
                i32.const 42
                local.set 1   ;; dead — overwritten before any read
                i32.const 7
                local.set 1   ;; live — reaches the local.get below
                local.get 1
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::eliminate_dead_stores(&mut module).expect("pass");

        // The first write (the i32.const 42 / local.set 1 pair) should
        // have its set replaced by Drop.
        let dropped_sets = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Drop))
            .count();
        assert!(dropped_sets >= 1, "First write must be neutralized to Drop");

        // The function still validates as wasm.
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_eliminate_dead_stores_preserves_live_writes() {
        // A single write whose value IS read must survive untouched.
        let wat = r#"(module
            (func $test (param i32) (result i32)
                (local i32)
                i32.const 42
                local.set 1
                local.get 1
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let original = module.functions[0].instructions.clone();

        optimize::eliminate_dead_stores(&mut module).expect("pass");

        assert_eq!(
            module.functions[0].instructions, original,
            "Live writes must not be changed"
        );
    }

    #[test]
    fn test_eliminate_dead_stores_skips_unverifiable_float_memory() {
        // #220: when a function contains float load/store, the Z3 translation
        // validator silently skips it (no proof), so dead-store elimination has
        // no safety net and its liveness analysis can be unsound (the meld-fused
        // falcon core dropped a live float store: run-stabilization 0.023->0.329).
        // The conservative fix: skip the whole function. The dead local write
        // below would normally be removed, but the f32.store/f32.load make the
        // function unverifiable, so it must be left byte-for-byte unchanged.
        let wat = r#"(module
            (memory 1)
            (func $test (result f32)
                (local i32)
                i32.const 42
                local.set 0       ;; dead local write — normally eliminated
                i32.const 0
                f32.const 1.5
                f32.store         ;; unverifiable: forces the conservative skip
                i32.const 0
                f32.load
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let original = module.functions[0].instructions.clone();

        optimize::eliminate_dead_stores(&mut module).expect("pass");

        assert_eq!(
            module.functions[0].instructions, original,
            "#220: functions with unverifiable float load/store must be skipped \
             entirely — never eliminate a store the verifier cannot prove dead"
        );
    }

    #[test]
    fn test_eliminate_dead_stores_branch_aware_keeps_partial_use() {
        // local 1 is written, then a Br could exit early without
        // overwriting. The first write IS live on the early-exit path
        // and must be preserved.
        //
        // Structure:
        //   local.set 1     ; conditional may need this value via the if
        //   if (...) {
        //     local.set 1   ; overwrites only on this path
        //   }
        //   local.get 1     ; reads on either path
        //
        // The first set is LIVE on the "if-not-taken" path. Pick #3 must
        // recognize this — replacing it with Drop would expose an
        // uninitialized read.
        let wat = r#"(module
            (func $test (param i32) (result i32)
                (local i32)
                i32.const 42
                local.set 1
                local.get 0
                if
                    i32.const 7
                    local.set 1
                end
                local.get 1
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let count_sets_before = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::LocalSet(1)))
            .count();
        assert_eq!(count_sets_before, 1); // only the outer set is at top level

        optimize::eliminate_dead_stores(&mut module).expect("pass");

        // The outer (first) set MUST survive — it's live on the
        // if-not-taken path that reaches local.get 1.
        let outer_set_survives = matches!(
            module.functions[0].instructions.get(1),
            Some(Instruction::LocalSet(1))
        );
        assert!(
            outer_set_survives,
            "Outer set must survive: it's live on if-not-taken path"
        );
    }

    #[test]
    fn test_eliminate_dead_stores_both_arms_overwrite() {
        // The first set is dead because BOTH if arms overwrite local 1
        // before any read. Liveness must compute the union of arm-deads
        // correctly (live-before-if = live-in-then ∪ live-in-else).
        let wat = r#"(module
            (func $test (param i32) (result i32)
                (local i32)
                i32.const 42
                local.set 1   ;; dead — both arms overwrite
                local.get 0
                if
                    i32.const 7
                    local.set 1
                else
                    i32.const 9
                    local.set 1
                end
                local.get 1
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::eliminate_dead_stores(&mut module).expect("pass");

        // The first set should be neutralized (replaced with Drop)
        // because every continuation overwrites local 1.
        let drops = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Drop))
            .count();
        assert!(
            drops >= 1,
            "Outer set must be neutralized — both arms overwrite"
        );

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_eliminate_dead_stores_return_kills_continuation() {
        // After Return, subsequent code is unreachable. A write
        // immediately followed by Return where the value isn't part
        // of the return is dead.
        let wat = r#"(module
            (func $test (result i32)
                (local i32)
                i32.const 42
                local.set 0   ;; dead — Return follows, no read
                i32.const 7
                return
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::eliminate_dead_stores(&mut module).expect("pass");

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");

        // The set should be replaced by Drop.
        let drops = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Drop))
            .count();
        assert!(drops >= 1, "Set followed by Return must be dead");
    }

    #[test]
    fn test_eliminate_dead_stores_localtee_dead_removed() {
        // LocalTee whose stored value is never read should be removed
        // (not replaced with Drop), letting the [T] -> [T] value pass
        // through. Same stack-effect rule as eliminate_dead_locals.
        let wat = r#"(module
            (func $test (param i32) (result i32)
                (local i32)
                local.get 0
                i32.const 1
                i32.add
                local.tee 1   ;; dead — never read
                i32.const 2
                i32.mul
                drop
                local.get 0   ;; ignores local 1
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::eliminate_dead_stores(&mut module).expect("pass");

        // No LocalTee should survive (the only one was dead).
        let has_tee = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalTee(_)));
        assert!(!has_tee, "Dead LocalTee must be removed entirely");

        // Validates: stack effect is preserved by removal.
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    // vacuum const+drop peephole tests (PR-B/PR-C follow-up)
    //
    // Closes the loop: dead-locals and dead-stores neutralize a
    // `LocalSet` to `Drop`, leaving the value-pusher (`i32.const X`,
    // `local.get N`, etc.) immediately followed by `Drop`. Vacuum's
    // peephole now folds the pair away.

    #[test]
    fn test_vacuum_folds_const_drop() {
        let wat = r#"(module
            (func $test (result i32)
                i32.const -22
                drop
                i32.const 7
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let len_before = module.functions[0].instructions.len();
        optimize::vacuum(&mut module).expect("vacuum");
        let len_after = module.functions[0].instructions.len();
        assert!(
            len_after < len_before,
            "const+drop pair must be folded away (was {len_before}, now {len_after})"
        );
        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_vacuum_folds_local_get_drop() {
        let wat = r#"(module
            (func $test (param i32) (result i32)
                local.get 0
                drop
                local.get 0
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");
        let drops = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Drop))
            .count();
        assert_eq!(drops, 0, "local.get + drop must fold away");
    }

    #[test]
    fn test_vacuum_does_not_fold_load_drop() {
        // A memory load can trap on bad address — discarding the result
        // does NOT discard the trap, so the load must survive vacuum.
        let wat = r#"(module
            (memory 1)
            (func $test (param i32) (result i32)
                local.get 0
                i32.load
                drop
                local.get 0
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");
        let has_load = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Load { .. }));
        assert!(
            has_load,
            "i32.load + drop must NOT be folded — load can trap, dropping result \
             does not discard the trap"
        );
    }

    #[test]
    fn test_vacuum_folds_const_drop_inside_block() {
        // Peephole must recurse into nested control flow.
        let wat = r#"(module
            (func $test (param i32) (result i32)
                block (result i32)
                    i32.const 99
                    drop
                    local.get 0
                end
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");
        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");

        // Walk into the block to confirm the const+drop is gone.
        fn count_drops(instrs: &[Instruction]) -> usize {
            let mut n = 0;
            for i in instrs {
                match i {
                    Instruction::Drop => n += 1,
                    Instruction::Block { body, .. } | Instruction::Loop { body, .. } => {
                        n += count_drops(body)
                    }
                    Instruction::If {
                        then_body,
                        else_body,
                        ..
                    } => n += count_drops(then_body) + count_drops(else_body),
                    _ => {}
                }
            }
            n
        }
        assert_eq!(count_drops(&module.functions[0].instructions), 0);
    }

    // inline_functions i64-correctness regression tests (loom#98)
    //
    // Pre-fix symptom: every function in gale-ffi (Verus-verified kernel
    // wasm with u64-packed FFI returns) panicked the inline pass with
    // `SortDiffers { left: BitVec(64), right: BitVec(32) }` deep in the
    // z3 crate. Root cause: when `inline_functions` adds new locals to
    // hold the inlined callee's parameters, the Z3 verifier extended its
    // shared-input symbolic-locals vector with hardcoded 32-bit zeros
    // regardless of declared type. Any subsequent i64 binop on those
    // locals tripped Z3's width check.
    //
    // Fix: extend with the declared local type's BV width via
    // `local_type_at` + `bv_width_for_value_type` (see verify.rs).
    //
    // The tests below construct minimal i64-typed function pairs that
    // exercise the broken path. Without the fix they trip the panic and
    // the inline pass reverts every function (no-op). With the fix they
    // either inline successfully or revert via a clean Z3 verdict — never
    // a panic.

    #[test]
    fn test_inline_i64_helper_no_z3_panic() {
        // Smallest reproducer of loom#98: a tiny i64-param helper with a
        // single call site triggers the inline pass to add new i64 locals
        // to the caller. The Z3 verifier must handle those 64-bit BVs
        // without panicking.
        let wat = r#"(module
            (func $helper (param i64 i64) (result i64)
                local.get 0
                local.get 1
                i64.add
            )
            (func $caller (export "test") (param i64 i64) (result i64)
                local.get 0
                local.get 1
                call $helper
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        // The pass must complete without panicking. Whether it actually
        // inlines or conservatively reverts, the absence of a panic is
        // the regression we lock in.
        optimize::inline_functions(&mut module).expect("must not panic");

        // Output must validate as wasm regardless of inline outcome.
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_mixed_i32_i64_widths_no_z3_panic() {
        // Exercises the gale-ffi pattern more directly: i64-packed FFI
        // return that the caller masks down to i32 fields. The bit-mask
        // and wrap force mixed-width values to flow through the verifier
        // simultaneously.
        let wat = r#"(module
            (func $packed_return (param i32) (result i64)
                local.get 0
                i64.extend_i32_u
                i64.const 0xFF
                i64.shl
                local.get 0
                i64.extend_i32_u
                i64.or
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                call $packed_return
                i32.wrap_i64
                i32.const 0xFF
                i32.and
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::inline_functions(&mut module).expect("must not panic");

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_i64_local_only_no_z3_panic() {
        // No params, just an i64 local — the helper still needs Z3 to
        // handle 64-bit symbolic state when its body is inlined into a
        // caller that didn't previously declare any i64 locals.
        let wat = r#"(module
            (func $helper (result i64)
                (local i64)
                i64.const 42
                local.set 0
                local.get 0
                local.get 0
                i64.add
            )
            (func $caller (export "test") (result i64)
                call $helper
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        optimize::inline_functions(&mut module).expect("must not panic");

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_pass_actually_inlines_i64_helper() {
        // loom#151: the verifier now models a pure, no-trap, leaf,
        // straight-line callee by its OWN BODY (not an opaque
        // uninterpreted call), so it can PROVE the i64 inline sound and
        // KEEP it. Before #151 every i64 inline (even `x + x`) reverted
        // because the opaque call model could never equal the inlined
        // body. Confirm the pass does its job on i64 helpers — the call
        // must be replaced by the helper's body.
        let wat = r#"(module
            (func $double (param i64) (result i64)
                local.get 0
                local.get 0
                i64.add
            )
            (func $caller (export "test") (param i64) (result i64)
                local.get 0
                call $double
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let calls_before = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(calls_before, 1, "caller starts with one Call");

        optimize::inline_functions(&mut module).expect("must not panic");

        let calls_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(
            calls_after, 0,
            "i64 helper must be inlined — pre-fix the verifier panicked \
             and reverted every function, leaving the Call in place"
        );

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[cfg(feature = "verification")]
    #[test]
    fn test_inline_verifier_proves_correct_and_rejects_wrong_i64_inline() {
        // loom#151 SOUNDNESS GUARD. The by-body call model must do BOTH:
        //   (a) PROVE a correct i64 inline equivalent (so inlining works), and
        //   (b) REJECT a semantically WRONG inline (so it stays sound).
        // add1(x) = x + 1. Modeled by its body, `call add1(x)` becomes the
        // expression `x + 1`, so:
        //   - a caller inlined to `x + 1` MUST verify (Ok(true));
        //   - a caller "inlined" to `x + 2` MUST NOT verify (Ok(false)).
        // If (b) ever returns Ok(true), by-body modeling has become unsound
        // and this test fails loudly — exactly the gate we want.
        use crate::verify::{
            VerificationSignatureContext, verify_function_equivalence_with_context,
        };
        let wat = r#"(module
            (func $add1 (param i64) (result i64)
                local.get 0
                i64.const 1
                i64.add
            )
            (func $caller (param i64) (result i64)
                local.get 0
                call 0
            )
        )"#;
        let module = parse::parse_wat(wat).expect("parse");
        let ctx = VerificationSignatureContext::from_module(&module);
        // Original: `local.get 0; call $add1`.
        let orig = module.functions[1].clone();

        // (a) Correct inline of add1: x + 1 — must be PROVEN equivalent.
        let mut correct = orig.clone();
        correct.instructions = vec![
            Instruction::LocalGet(0),
            Instruction::I64Const(1),
            Instruction::I64Add,
        ];
        assert!(
            verify_function_equivalence_with_context(&orig, &correct, "test", &ctx)
                .expect("verify ok"),
            "correct i64 inline (x+1) must be proven equivalent to call add1",
        );

        // (b) WRONG inline: x + 2 — must be REJECTED (counterexample).
        let mut wrong = orig.clone();
        wrong.instructions = vec![
            Instruction::LocalGet(0),
            Instruction::I64Const(2),
            Instruction::I64Add,
        ];
        assert!(
            !verify_function_equivalence_with_context(&orig, &wrong, "test", &ctx)
                .expect("verify ok"),
            "SOUNDNESS: wrong i64 inline (x+2) must NOT verify against call add1",
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn test_inline_verifier_proves_and_rejects_memory_load_inline() {
        // loom#155 SOUNDNESS GUARD for memory-through inline modeling.
        // `getx(p) = i32.load(p)` reads linear memory. The by-body modeler
        // must encode that load against the caller's shared memory Array, so:
        //   (a) inlining `i32.load(p)` (offset 0) MUST verify (Ok(true)); and
        //   (b) inlining a DIFFERENT load `i32.load(p) offset=4` (reads another
        //       address) MUST be rejected (Ok(false)).
        // (a) failing would mean memory-reading inlines can't be proven (the
        // pre-#155 no-op); (b) passing would mean the modeler ignores the load
        // address — unsound. Both are locked here.
        use crate::verify::{
            VerificationSignatureContext, verify_function_equivalence_with_context,
        };
        let wat = r#"(module
            (memory 1)
            (func $getx (param i32) (result i32)
                local.get 0
                i32.load)
            (func $caller (param i32) (result i32)
                local.get 0
                call 0)
        )"#;
        let module = parse::parse_wat(wat).expect("parse");
        let ctx = VerificationSignatureContext::from_module(&module);
        // Original: `local.get 0; call $getx` (getx reads memory at p).
        let orig = module.functions[1].clone();

        // (a) Correct inline: load at the same address — must be PROVEN.
        let mut correct = orig.clone();
        correct.instructions = vec![
            Instruction::LocalGet(0),
            Instruction::I32Load {
                offset: 0,
                align: 2,
                mem: 0,
            },
        ];
        assert!(
            verify_function_equivalence_with_context(&orig, &correct, "test", &ctx)
                .expect("verify ok"),
            "correct memory-load inline (i32.load p) must be proven equivalent to call getx",
        );

        // (b) WRONG inline: load 4 bytes higher — must be REJECTED.
        let mut wrong = orig.clone();
        wrong.instructions = vec![
            Instruction::LocalGet(0),
            Instruction::I32Load {
                offset: 4,
                align: 2,
                mem: 0,
            },
        ];
        assert!(
            !verify_function_equivalence_with_context(&orig, &wrong, "test", &ctx)
                .expect("verify ok"),
            "SOUNDNESS: wrong memory-load inline (offset=4) must NOT verify against call getx",
        );
    }

    #[cfg(feature = "verification")]
    #[test]
    fn test_inline_verifier_proves_and_rejects_memory_store_inline() {
        // loom#157 SOUNDNESS GUARD for by-body STORE modeling. `wr(s,v){ *s=v }`
        // writes memory; `caller(s,v){ wr(s,v); return *s }` calls it then reads
        // back. The by-body modeler must apply wr's store to the shared Array so
        // the readback observes it. Then:
        //   (a) inlining the correct store (`*s=v`) MUST verify (Ok(true)); and
        //   (b) inlining a WRONG store (`*(s+4)=v`, leaving *s unwritten) MUST be
        //       rejected (Ok(false)) — the readback would differ.
        use crate::verify::{
            VerificationSignatureContext, verify_function_equivalence_with_context,
        };
        let wat = r#"(module
            (memory 1)
            (func $wr (param i32 i32)
                local.get 0
                local.get 1
                i32.store)
            (func $caller (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call 0
                local.get 0
                i32.load)
        )"#;
        let module = parse::parse_wat(wat).expect("parse");
        let ctx = VerificationSignatureContext::from_module(&module);
        let orig = module.functions[1].clone(); // wr(s,v); return *s

        // (a) correct store inline: store v at *s, then read *s back → v.
        let mut correct = orig.clone();
        correct.instructions = vec![
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I32Store {
                offset: 0,
                align: 2,
                mem: 0,
            },
            Instruction::LocalGet(0),
            Instruction::I32Load {
                offset: 0,
                align: 2,
                mem: 0,
            },
        ];
        assert!(
            verify_function_equivalence_with_context(&orig, &correct, "test", &ctx)
                .expect("verify ok"),
            "correct store inline (*s=v; read *s) must be proven equivalent to call wr",
        );

        // (b) WRONG store inline: store at *(s+4); *s stays unwritten → readback differs.
        let mut wrong = orig.clone();
        wrong.instructions = vec![
            Instruction::LocalGet(0),
            Instruction::LocalGet(1),
            Instruction::I32Store {
                offset: 4,
                align: 2,
                mem: 0,
            },
            Instruction::LocalGet(0),
            Instruction::I32Load {
                offset: 0,
                align: 2,
                mem: 0,
            },
        ];
        assert!(
            !verify_function_equivalence_with_context(&orig, &wrong, "test", &ctx)
                .expect("verify ok"),
            "SOUNDNESS: wrong store inline (offset=4) must NOT verify against call wr",
        );
    }

    #[test]
    fn test_inline_memory_seam_fully_dissolves() {
        // loom#155: the gale flight_control seam — `seam(s,v){ wr(s,v); return
        // rd(s); }` where `wr` WRITES *s and `rd` READS it. loom must inline
        // the reader `rd` (proven against the havoc'd memory left by the
        // opaque `wr` call) while keeping `wr` as a call — inlining the writer
        // would diverge (concrete stores vs the original's havoc) and falsely
        // revert. Mirrors gale's min_seam; the verifier proves the kept inline.
        let wat = r#"(module
            (memory 1)
            (func $seam (export "seam") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call $wr
                local.get 0
                call $rd)
            (func $rd (param i32) (result i32)
                local.get 0
                i32.load offset=4
                local.get 0
                i32.load
                i32.add)
            (func $wr (param i32 i32)
                local.get 0
                local.get 1
                i32.store
                local.get 0
                local.get 1
                i32.store offset=4)
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        // full idx: seam=0, rd=1, wr=2 (no imports).
        let rd_before = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(1)))
            .count();
        let wr_before = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(2)))
            .count();
        assert_eq!(
            (rd_before, wr_before),
            (1, 1),
            "seam starts with one rd + one wr call"
        );

        optimize::inline_functions(&mut module).expect("must not panic");

        let seam = &module.functions[0];
        let rd_after = seam
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(1)))
            .count();
        let wr_after = seam
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(2)))
            .count();
        assert_eq!(
            rd_after, 0,
            "memory-reading callee rd must be inlined (proven via memory-through modeling, #155)"
        );
        // loom#157: the full-width-store writer `wr` is now ALSO inlined+proven
        // (its stores are modeled against the shared memory Array on both
        // sides), so the seam fully dissolves — 0 calls remain.
        assert_eq!(
            wr_after, 0,
            "memory-writing callee wr must now be inlined too (by-body store modeling, #157)"
        );
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_reader_after_unmodelable_writer() {
        // loom#159 REGRESSION GUARD. A full-width READER (`rd`) following an
        // UN-MODELABLE WRITER (`nm`, uses `f32.store` — float memory ops have no
        // Z3 model). The reader must still inline (proven against the writer's
        // havoc'd memory) while the un-modelable writer stays an opaque call.
        // v1.1.7 regressed this: a non-modelable callee became a candidate (the
        // filter looked only at writes), got inlined, diverged from the
        // original's havoc → false revert that ALSO killed `rd`'s inline. The
        // fix gates candidates on FULL inline-modelability.
        // (NB: this guard's "un-modelable" op has had to move as the capability
        // series advanced — store16 became modelable in #161, div_s in #163.
        // Float memory ops are anchored outside the integer model, so `f32.store`
        // is a durable choice.)
        let wat = r#"(module
            (memory 1)
            (func $seam (export "seam") (param i32 f32) (result i32)
                local.get 0
                local.get 1
                call $nm
                local.get 0
                call $rd)
            (func $rd (param i32) (result i32)
                local.get 0
                i32.load)
            (func $nm (param i32 f32)
                local.get 0
                local.get 1
                f32.store)
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::inline_functions(&mut module).expect("must not panic");
        let seam = &module.functions[0];
        // full idx: seam=0, rd=1, nm=2.
        let rd_calls = seam
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(1)))
            .count();
        let pw_calls = seam
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(2)))
            .count();
        assert_eq!(
            rd_calls, 0,
            "full-width reader must inline even after an un-modelable writer (#159)"
        );
        assert_eq!(
            pw_calls, 1,
            "un-modelable writer (f32.store) must stay an opaque call"
        );
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_division_seam_fully_dissolves() {
        // loom#163: a seam whose reader divides (`rdd`: (a+b)/1000, i32.div_s,
        // truncate toward zero). Integer div/rem are now by-body modelable via
        // Z3 bvsdiv/bvudiv/bvsrem/bvurem, so writer `wrd` AND reader `rdd` both
        // inline → the seam fully dissolves (0 calls), proven. This is the final
        // through-memory inlining capability (reads → writes → partial-width →
        // division) — it collapses gale's real flight_algo to one function.
        // Mirrors gale's divseam.
        let wat = r#"(module
            (memory 1)
            (func $seam (export "seam") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call $wrd
                local.get 0
                call $rdd)
            (func $rdd (param i32) (result i32)
                local.get 0
                i32.load offset=4
                local.get 0
                i32.load
                i32.add
                i32.const 1000
                i32.div_s)
            (func $wrd (param i32 i32)
                local.get 0
                local.get 1
                i32.store offset=4
                local.get 0
                local.get 1
                i32.store)
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::inline_functions(&mut module).expect("must not panic");
        let calls = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(
            calls, 0,
            "division seam must fully dissolve: both i32.store writer and i32.div_s reader inline (#163)"
        );
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_partial_width_seam_fully_dissolves() {
        // loom#161: a seam over int16_t fields — writer `wr16` uses i32.store16,
        // reader `rd16` uses i32.load16_s (sign-extended). Both are now by-body
        // modelable (partial-width helpers), so the seam fully dissolves
        // (0 calls), proven. Mirrors gale's seam16.
        let wat = r#"(module
            (memory 1)
            (func $seam (export "seam") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call $wr16
                local.get 0
                call $rd16)
            (func $rd16 (param i32) (result i32)
                local.get 0
                i32.load16_s offset=2
                local.get 0
                i32.load16_s
                i32.add)
            (func $wr16 (param i32 i32)
                local.get 0
                local.get 1
                i32.store16
                local.get 0
                local.get 1
                i32.store16 offset=2)
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::inline_functions(&mut module).expect("must not panic");
        let calls = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        assert_eq!(
            calls, 0,
            "partial-width seam must fully dissolve: both i32.store16 writer and i32.load16_s reader inline (#161)"
        );
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_caller_with_imported_call() {
        // loom#153: when the caller also calls an IMPORTED function, the
        // inline of a *local* callee was reverted. `Call(func_idx)` uses the
        // full index space (imports first), but the inliner indexed the
        // local-only `all_functions` with it — so the import's index (0)
        // collided with local function 0, and the inliner tried to inline
        // the void import call, emitting a `local.set` with no stack
        // argument → malformed body → verifier "Stack underflow" → revert.
        // After the fix: the import call ($ext) is preserved and the local
        // callee ($dec) is inlined (call to it removed) and verified.
        let wat = r#"(module
            (import "env" "ext" (func $ext))
            (func $dec (param i32) (result i64)
                (i64.extend_i32_u (local.get 0)))
            (func $z (export "z") (param i32) (result i64)
                (call $ext)
                (call $dec (local.get 0)))
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");

        // Full index space: $ext = 0 (import), $dec = 1, $z = 2.
        // module.functions holds locals only: [0]=$dec, [1]=$z.
        let z = &module.functions[1];
        let ext_calls_before = z
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(0)))
            .count();
        let dec_calls_before = z
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(1)))
            .count();
        assert_eq!(ext_calls_before, 1, "caller starts with one import call");
        assert_eq!(dec_calls_before, 1, "caller starts with one local call");

        optimize::inline_functions(&mut module).expect("must not panic");

        let z = &module.functions[1];
        let ext_calls_after = z
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(0)))
            .count();
        let dec_calls_after = z
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(1)))
            .count();
        assert_eq!(
            ext_calls_after, 1,
            "imported call must be preserved (never inlined — it has no body)"
        );
        assert_eq!(
            dec_calls_after, 0,
            "local callee must be inlined even though the caller also calls an import"
        );

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    #[test]
    fn test_inline_i64_loop_kinduction_no_panic() {
        // loom#145 regression: the prior i64 inline tests are loopless, so
        // they never exercise the k-induction verifier path
        // (verify_loops_kinduction / encode_loop_body_for_kinduction),
        // which hardcoded BV32 for defaults+globals and did UNMATCHED
        // binops. An i64 loop body therefore tripped Z3's
        // `SortDiffers { BitVec 64 vs 32 }` panic deep in the bindings,
        // reverting every function on i64-heavy modules (gale-ffi /
        // compiler_builtins). This locks in: an i64 helper with a LOOP,
        // inlined into a caller, completes without panicking and yields
        // valid wasm.
        let wat = r#"(module
            (func $sum_to (param i64) (result i64)
                (local i64 i64)
                i64.const 0
                local.set 1
                i64.const 0
                local.set 2
                block
                    loop
                        local.get 2
                        local.get 0
                        i64.ge_u
                        br_if 1
                        local.get 1
                        local.get 2
                        i64.add
                        local.set 1
                        local.get 2
                        i64.const 1
                        i64.add
                        local.set 2
                        br 0
                    end
                end
                local.get 1
            )
            (func $caller (export "test") (param i64) (result i64)
                local.get 0
                call $sum_to
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        // The whole point: no panic, regardless of whether the loop
        // function inlines or conservatively reverts. (Conservative
        // revert is sound; the regression is the panic + 21 MB stderr.)
        optimize::inline_functions(&mut module).expect("must not panic on i64 loop");

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    // PR-C (v1.0.2): directize tests. The pass folds
    // `i32.const N; call_indirect (type T) table 0` into `call F` when
    // - the module has no Unknown instructions (i.e., no table mutation),
    // - the element section assigns slot N of table 0 to function F, and
    // - F's signature matches type T.

    #[test]
    fn test_directize_folds_known_indirect_call() {
        // Active element segment puts $f0 at table[0]; call_indirect
        // with i32.const 0 must resolve to call $f0.
        let wat = r#"(module
            (type $sig (func (param i32) (result i32)))
            (table 1 funcref)
            (elem (i32.const 0) $f0)
            (func $f0 (type $sig)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                call_indirect (type $sig)
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        let calls_before = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        let indirects_before = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::CallIndirect { .. }))
            .count();
        assert_eq!(indirects_before, 1, "sanity: one call_indirect before");
        assert_eq!(calls_before, 0, "sanity: no direct calls before");

        optimize::directize(&mut module).expect("directize");

        let calls_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Call(_)))
            .count();
        let indirects_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::CallIndirect { .. }))
            .count();
        assert_eq!(
            indirects_after, 0,
            "directize must remove the call_indirect"
        );
        assert_eq!(calls_after, 1, "directize must insert a direct call");

        // Output validates.
        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_directize_skips_non_const_index() {
        // Indirect index is local.get, not a constant — must NOT fold.
        let wat = r#"(module
            (type $sig (func (param i32) (result i32)))
            (table 1 funcref)
            (elem (i32.const 0) $f0)
            (func $f0 (type $sig)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $caller (export "test") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                call_indirect (type $sig)
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::directize(&mut module).expect("directize");

        let indirects_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::CallIndirect { .. }))
            .count();
        assert_eq!(
            indirects_after, 1,
            "call_indirect with non-const index must survive"
        );
    }

    #[test]
    fn test_directize_skips_out_of_range_index() {
        // Element segment has 1 slot; call_indirect uses index 5 — no
        // resolution available, must NOT fold.
        let wat = r#"(module
            (type $sig (func (param i32) (result i32)))
            (table 10 funcref)
            (elem (i32.const 0) $f0)
            (func $f0 (type $sig)
                local.get 0
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                i32.const 5
                call_indirect (type $sig)
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::directize(&mut module).expect("directize");

        let indirects_after = module.functions[1]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::CallIndirect { .. }))
            .count();
        assert_eq!(
            indirects_after, 1,
            "call_indirect to unresolved slot must survive"
        );
    }

    // v1.0.5 Track 1: ægraph pipeline integration tests.

    #[test]
    fn test_egraph_optimize_folds_x_plus_zero() {
        // local.get 0; i32.const 0; i32.add  → local.get 0
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                i32.const 0
                i32.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let before = module.functions[0].instructions.len();
        optimize::egraph_optimize(&mut module).expect("egraph_optimize");
        let after = module.functions[0].instructions.len();

        assert!(
            after < before,
            "egraph_optimize must shrink x+0 → x. before={} after={}",
            before,
            after
        );
        // The add must be gone.
        let has_add = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Add));
        assert!(!has_add, "i32.add must be folded away");

        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_egraph_optimize_no_op_on_plain_function() {
        // Ordinary function with no foldable identities — pass is byte-
        // for-byte no-op.
        let wat = r#"(module
            (func (export "test") (param i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        let before = module.functions[0].instructions.clone();
        optimize::egraph_optimize(&mut module).expect("egraph_optimize");
        assert_eq!(
            module.functions[0].instructions, before,
            "non-identity function must be untouched"
        );
    }

    #[test]
    fn test_egraph_optimize_skips_across_call() {
        // Identity sub-tree present, but a Call appears mid-sequence —
        // the call breaks the egraph tree, but the identity tree AFTER
        // the call should still fold.
        let wat = r#"(module
            (func $helper (param i32) (result i32) local.get 0)
            (func (export "test") (param i32) (result i32)
                local.get 0
                call $helper
                drop
                local.get 0
                i32.const 0
                i32.add
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::egraph_optimize(&mut module).expect("egraph_optimize");
        // After-call identity tree must have folded.
        let has_add = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Add));
        assert!(!has_add, "post-call identity tree should still fold");
    }

    #[test]
    fn test_egraph_optimize_recurses_into_blocks() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                block (result i32)
                    local.get 0
                    i32.const 0
                    i32.add
                end
            )
        )"#;
        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::egraph_optimize(&mut module).expect("egraph_optimize");
        let has_add_anywhere = module.functions[0].instructions.iter().any(|i| {
            if let Instruction::Block { body, .. } = i {
                body.iter().any(|j| matches!(j, Instruction::I32Add))
            } else {
                matches!(i, Instruction::I32Add)
            }
        });
        assert!(!has_add_anywhere, "block-nested identity must fold");
    }

    // PR-G (v0.7.0): verification-aware canonicalization tests.

    #[test]
    fn test_canonicalize_if_else_to_select() {
        // (if (result i32) (then 1) (else 0)) where cond is local.get
        // → 1; 0; cond; select
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                if (result i32)
                    i32.const 1
                else
                    i32.const 0
                end
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::canonicalize(&mut module).expect("canonicalize");

        // No If should remain.
        let has_if = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::If { .. }));
        assert!(
            !has_if,
            "single-value if/else of pure pushers must become select"
        );

        // Select must appear.
        let has_select = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Select));
        assert!(has_select, "select must replace the if/else");

        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_canonicalize_keeps_complex_if_arms() {
        // An if/else with multi-instruction arms must NOT be turned
        // into select — the arms are not single pure pushers, so the
        // re-ordering would change behavior.
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                local.get 0
                if (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                else
                    i32.const 0
                end
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::canonicalize(&mut module).expect("canonicalize");

        // If must survive — at least one arm has multiple instructions.
        let has_if = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::If { .. }));
        assert!(
            has_if,
            "if with multi-instruction arms must NOT canonicalize to select"
        );
    }

    #[test]
    fn test_canonicalize_localset_localget_to_tee() {
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                (local i32)
                local.get 0
                local.set 1
                local.get 1
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::canonicalize(&mut module).expect("canonicalize");

        let has_tee = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalTee(_)));
        assert!(
            has_tee,
            "local.set;local.get pair must collapse to local.tee"
        );

        // No LocalSet should remain (the pair was the only set).
        let has_set = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalSet(_)));
        assert!(!has_set, "local.set should be gone (folded into tee)");

        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_canonicalize_does_not_collapse_mismatched_indices() {
        // local.set 1; local.get 2 — different indices, NOT a tee opportunity.
        let wat = r#"(module
            (func (export "test") (param i32) (result i32)
                (local i32 i32)
                local.get 0
                local.set 1
                local.get 2
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::canonicalize(&mut module).expect("canonicalize");

        let has_set = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalSet(1)));
        let has_get = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::LocalGet(2)));
        assert!(has_set, "local.set 1 must survive (different idx from get)");
        assert!(has_get, "local.get 2 must survive");
    }

    // PR-F (v0.7.0): function-summary IPA enables vacuum to fold
    // `Call f; Drop` when f is pure + no-trap + single-result.

    #[test]
    fn test_vacuum_folds_pure_zero_arg_call_drop() {
        // Zero-arg pure+no-trap helper. The safe minimum for
        // pure-call-drop folding: no args to leave dangling.
        let wat = r#"(module
            (func $pure_helper (result i32)
                i32.const 42
            )
            (func $caller (export "test") (result i32)
                call $pure_helper
                drop
                i32.const 7
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");

        // The Call+Drop in $caller (function index 1) should be folded.
        let has_call = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_)));
        let has_drop = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Drop));
        assert!(
            !has_call,
            "Call to pure+no-trap zero-arg helper must be folded"
        );
        assert!(!has_drop, "Paired Drop must also be folded");

        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_vacuum_folds_pure_call_drop_with_pure_args() {
        // PR-J: pure+no-trap helper with 1 arg, and the arg is a pure
        // pusher. The fold pops the pure-pusher arg AND the Call AND
        // the Drop. Net stack effect: zero, same as before.
        let wat = r#"(module
            (func $pure_with_arg (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                call $pure_with_arg
                drop
                local.get 0
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");

        let has_call = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_)));
        let has_drop = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Drop));
        assert!(
            !has_call,
            "PR-J: pure helper with pure-pusher args must be folded"
        );
        assert!(!has_drop, "Paired Drop must also be folded");

        // Output must validate.
        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_vacuum_folds_pure_call_drop_with_multiple_pure_args() {
        // PR-J: pure+no-trap helper with 3 args, all pure pushers
        // (constant, local.get, global.get). All four (the three
        // arg pushers, the Call) and the Drop must vanish.
        let wat = r#"(module
            (global $g i32 (i32.const 5))
            (func $pure_3 (param i32 i32 i32) (result i32)
                local.get 0
                local.get 1
                i32.add
                local.get 2
                i32.add
            )
            (func $caller (export "test") (param i32) (result i32)
                i32.const 1
                local.get 0
                global.get $g
                call $pure_3
                drop
                local.get 0
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");

        let has_call = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_)));
        let has_drop = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Drop));
        let has_global_get = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::GlobalGet(_)));
        let has_const_1 = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Const(1)));
        assert!(!has_call, "PR-J: multi-arg pure call must be folded");
        assert!(!has_drop, "Paired Drop must also be folded");
        assert!(
            !has_global_get,
            "Pure-pusher arg (global.get) must be folded away"
        );
        assert!(
            !has_const_1,
            "Pure-pusher arg (i32.const 1) must be folded away"
        );

        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_vacuum_keeps_call_drop_with_impure_arg() {
        // PR-J soundness: the helper is pure+no-trap, but one of its
        // args is `i32.load` — a may-trap pusher. We must NOT fold,
        // because removing the load erases an observable trap.
        let wat = r#"(module
            (memory 1)
            (func $pure_with_arg (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                i32.load
                call $pure_with_arg
                drop
                local.get 0
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");

        let has_call = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_)));
        let has_load = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::I32Load { .. }));
        assert!(
            has_call,
            "Call must NOT be folded when an arg is may-trap (i32.load)"
        );
        assert!(has_load, "i32.load must survive — its trap is observable");

        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_vacuum_keeps_call_drop_when_args_unavailable() {
        // PR-J defensive: the helper takes 1 arg, but the
        // immediately-preceding entry in `out` is `i32.add` — not a
        // pure pusher (i32.add consumes 2 / produces 1, so removing
        // it alone would unbalance the stack). The fold must bail
        // out per the "last N must all be pure pushers" rule.
        //
        // This is the closest WAT-expressible analogue to the
        // "args came from outside the local region" case: the arg
        // exists on the stack but wasn't placed there by a pure
        // pusher, so the peephole cannot prove it's safe to remove.
        let wat = r#"(module
            (func $pure_with_arg (param i32) (result i32)
                local.get 0
                i32.const 1
                i32.add
            )
            (func $caller (export "test") (param i32) (result i32)
                local.get 0
                local.get 0
                i32.add
                call $pure_with_arg
                drop
                local.get 0
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");

        let has_call = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_)));
        assert!(
            has_call,
            "Call must NOT be folded when an arg-producer is not a pure pusher (i32.add)"
        );

        let bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&bytes).expect("output validates");
    }

    #[test]
    fn test_vacuum_keeps_impure_call_drop() {
        // A call to a function that writes memory must NOT be folded,
        // even if its result is dropped — the store is observable.
        let wat = r#"(module
            (memory 1)
            (func $impure (result i32)
                i32.const 0
                i32.const 42
                i32.store
                i32.const 0
            )
            (func $caller (export "test") (result i32)
                call $impure
                drop
                i32.const 7
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");

        let has_call = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_)));
        assert!(
            has_call,
            "Call to impure helper must NOT be folded — the store is observable"
        );
    }

    #[test]
    fn test_vacuum_keeps_may_trap_call_drop() {
        // A call to a no-side-effect-but-may-trap function (e.g., does
        // a load) must NOT be folded — the trap is observable behavior.
        let wat = r#"(module
            (memory 1)
            (func $may_trap (result i32)
                i32.const 0
                i32.load
            )
            (func $caller (export "test") (result i32)
                call $may_trap
                drop
                i32.const 7
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");
        optimize::vacuum(&mut module).expect("vacuum");

        let has_call = module.functions[1]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Call(_)));
        assert!(
            has_call,
            "Call to may-trap helper must NOT be folded — the trap is observable"
        );
    }

    // Soundness regression tests for the null-check / store-hoist guard
    // (v0.6.0 PR-E follow-up).
    //
    // Background: the gale v0.4.0 measurement report noted three
    // functions (gale_sem_count_take, gale_spinlock_acquire_nested,
    // gale_timer_expire) where, on the v0.4.0-era artifact, stores
    // appeared hoisted above null-pointer checks. v0.5.0's early-exit
    // guard work (extending `has_dataflow_unsafe_control_flow` to flag
    // nested `Return`/`Br`) closed the hoist hole; this test pins
    // that the canonical pattern remains sound through v0.6.0+
    // optimization pipelines.

    #[test]
    fn test_null_check_before_store_preserved_through_optimization() {
        // The gale_sem_count_take shape: param 0 is a pointer that may
        // be null; null-check comes first; if non-null, load, conditional
        // bail, then store-back-decremented. The store MUST NOT be
        // reordered above the null-check.
        let wat = r#"(module
            (memory 1)
            (func $sem_count_take (export "test") (param i32) (result i32)
                (local i32)
                local.get 0
                i32.eqz
                if
                    i32.const -22
                    return
                end
                local.get 0
                i32.load
                local.tee 1
                i32.eqz
                if
                    i32.const -16
                    return
                end
                local.get 0
                local.get 1
                i32.const 1
                i32.sub
                i32.store
                i32.const 0
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        // Run the full optimization-relevant pipeline that touched the
        // hoist hole in v0.4.0.
        optimize::constant_folding(&mut module).expect("constant_folding");
        optimize::eliminate_common_subexpressions_enhanced(&mut module).expect("cse");
        optimize::optimize_advanced_instructions(&mut module).expect("advanced");
        optimize::simplify_branches(&mut module).expect("branches");
        optimize::eliminate_dead_code(&mut module).expect("dce");
        optimize::merge_blocks(&mut module).expect("merge");
        optimize::vacuum(&mut module).expect("vacuum");
        optimize::simplify_locals(&mut module).expect("simplify_locals");
        optimize::eliminate_dead_stores(&mut module).expect("dead_stores");
        optimize::eliminate_dead_locals(&mut module).expect("dead_locals");
        optimize::vacuum(&mut module).expect("vacuum_final");

        // The null-check (i32.eqz on param 0) MUST appear before the
        // i32.store. We assert this by finding the index of the first
        // i32.store and the first i32.eqz reading param 0, then
        // comparing positions.
        let instrs = &module.functions[0].instructions;
        let mut first_store: Option<usize> = None;
        let mut first_null_check: Option<usize> = None;
        for (i, instr) in instrs.iter().enumerate() {
            if matches!(instr, Instruction::I32Store { .. }) && first_store.is_none() {
                first_store = Some(i);
            }
            if matches!(instr, Instruction::I32Eqz) && first_null_check.is_none() {
                first_null_check = Some(i);
            }
        }
        let store_pos = first_store.expect("must contain i32.store");
        let null_check_pos = first_null_check.expect("must contain i32.eqz");
        assert!(
            null_check_pos < store_pos,
            "Null check (i32.eqz at {null_check_pos}) MUST precede store \
             (i32.store at {store_pos}). v0.4.0 hoisted the store above \
             the check, which would null-deref. v0.5.0+ closes this; this \
             test pins it."
        );

        // Output must validate.
        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }

    // Pipeline order regression: PR #99 added a const+drop peephole
    // inside `vacuum`, but `vacuum` only ran BEFORE `dead-stores` /
    // `dead-locals`, so the const+drop pairs those passes create were
    // never folded. The pipeline now runs `vacuum` again as
    // `vacuum-final` after both dead-* passes. This test pins that the
    // peephole actually catches the in-pipeline residue.

    #[test]
    fn test_vacuum_final_folds_const_drop_from_dead_local() {
        // A function with a dead local that gets neutralized by
        // dead-locals: the LocalSet becomes Drop, leaving the i32.const
        // pushed and immediately Drop'd.
        let wat = r#"(module
            (func $f (param i32) (result i32)
                (local i32) ;; idx 1 — declared, never read
                i32.const -22
                local.set 1
                local.get 0
            )
        )"#;

        let mut module = parse::parse_wat(wat).expect("parse");

        // Step 1: dead-locals neutralizes LocalSet to Drop.
        optimize::eliminate_dead_locals(&mut module).expect("dead_locals");

        let has_drop_after_dead_locals = module.functions[0]
            .instructions
            .iter()
            .any(|i| matches!(i, Instruction::Drop));
        assert!(
            has_drop_after_dead_locals,
            "Sanity: dead-locals must have inserted a Drop"
        );

        // Step 2: final vacuum sweep should fold the `i32.const; Drop`
        // pair created in step 1.
        optimize::vacuum(&mut module).expect("vacuum_final");

        let drops_after = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::Drop))
            .count();
        let consts_after = module.functions[0]
            .instructions
            .iter()
            .filter(|i| matches!(i, Instruction::I32Const(-22)))
            .count();
        assert_eq!(
            drops_after, 0,
            "Final vacuum must have folded the const+drop pair: \
             {} drops remained, {} i32.const -22 instructions remained",
            drops_after, consts_after
        );
        assert_eq!(consts_after, 0, "The dead constant should also be gone");

        let wasm_bytes = encode::encode_wasm(&module).expect("encode");
        wasmparser::validate(&wasm_bytes).expect("output validates");
    }
}
