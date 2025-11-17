//! LOOM ISLE Definitions
//!
//! This crate contains ISLE (Instruction Selection/Lowering Expressions) term definitions
//! for WebAssembly optimization rules. The ISLE compiler generates Rust code from .isle files
//! during the build process.

#![allow(dead_code)]
#![allow(unused_variables)]

/// WebAssembly value types
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ValueType {
    I32,
    I64,
    F32,
    F64,
}

/// Block type for control flow structures
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum BlockType {
    /// No parameters, no results
    Empty,
    /// No parameters, single result
    Value(ValueType),
    /// Full function signature (for multi-value blocks)
    Func {
        params: Vec<ValueType>,
        results: Vec<ValueType>,
    },
}

/// Primitive type for 32-bit immediates
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Imm32(pub i32);

impl From<i32> for Imm32 {
    fn from(val: i32) -> Self {
        Imm32(val)
    }
}

impl From<Imm32> for i32 {
    fn from(imm: Imm32) -> Self {
        imm.0
    }
}

impl Imm32 {
    /// Create a new Imm32 from an i32
    pub fn new(val: i32) -> Self {
        Imm32(val)
    }

    /// Get the raw value
    pub fn value(&self) -> i32 {
        self.0
    }
}

/// Primitive type for 64-bit immediates
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Imm64(pub i64);

impl From<i64> for Imm64 {
    fn from(val: i64) -> Self {
        Imm64(val)
    }
}

impl From<Imm64> for i64 {
    fn from(imm: Imm64) -> Self {
        imm.0
    }
}

impl Imm64 {
    /// Create a new Imm64 from an i64
    pub fn new(val: i64) -> Self {
        Imm64(val)
    }

    /// Get the raw value
    pub fn value(&self) -> i64 {
        self.0
    }
}

/// Optional string for block/loop labels
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct OptionString(pub Option<String>);

impl OptionString {
    pub fn none() -> Self {
        OptionString(None)
    }

    pub fn some(s: String) -> Self {
        OptionString(Some(s))
    }
}

/// List of instructions (placeholder for control flow bodies)
/// In the actual implementation, this would reference the instruction vec
/// For now, we use an empty placeholder since control flow optimization
/// is handled in Rust passes rather than ISLE
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct InstructionList(pub Vec<u8>);

impl InstructionList {
    pub fn empty() -> Self {
        InstructionList(Vec::new())
    }
}

/// Value is a boxed pointer to ValueData
/// This allows recursive term structures
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Value(Box<ValueData>);

impl Value {
    /// Get a reference to the inner ValueData
    pub fn data(&self) -> &ValueData {
        &self.0
    }
}

/// ValueData represents the actual WebAssembly value/expression
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ValueData {
    /// i32.const N
    I32Const {
        val: Imm32,
    },
    /// i32.add lhs rhs
    I32Add {
        lhs: Value,
        rhs: Value,
    },
    /// i32.sub lhs rhs
    I32Sub {
        lhs: Value,
        rhs: Value,
    },
    /// i32.mul lhs rhs
    I32Mul {
        lhs: Value,
        rhs: Value,
    },
    /// i64.const N
    I64Const {
        val: Imm64,
    },
    /// i64.add lhs rhs
    I64Add {
        lhs: Value,
        rhs: Value,
    },
    /// i64.sub lhs rhs
    I64Sub {
        lhs: Value,
        rhs: Value,
    },
    /// i64.mul lhs rhs
    I64Mul {
        lhs: Value,
        rhs: Value,
    },

    /// Bitwise operations (i32)
    I32And {
        lhs: Value,
        rhs: Value,
    },
    I32Or {
        lhs: Value,
        rhs: Value,
    },
    I32Xor {
        lhs: Value,
        rhs: Value,
    },
    I32Shl {
        lhs: Value,
        rhs: Value,
    },
    I32ShrS {
        lhs: Value,
        rhs: Value,
    },
    I32ShrU {
        lhs: Value,
        rhs: Value,
    },

    /// Bitwise operations (i64)
    I64And {
        lhs: Value,
        rhs: Value,
    },
    I64Or {
        lhs: Value,
        rhs: Value,
    },
    I64Xor {
        lhs: Value,
        rhs: Value,
    },
    I64Shl {
        lhs: Value,
        rhs: Value,
    },
    I64ShrS {
        lhs: Value,
        rhs: Value,
    },
    I64ShrU {
        lhs: Value,
        rhs: Value,
    },

    /// Comparison operations (i32) - return i32 (0 or 1)
    I32Eq {
        lhs: Value,
        rhs: Value,
    },
    I32Ne {
        lhs: Value,
        rhs: Value,
    },
    I32LtS {
        lhs: Value,
        rhs: Value,
    },
    I32LtU {
        lhs: Value,
        rhs: Value,
    },
    I32GtS {
        lhs: Value,
        rhs: Value,
    },
    I32GtU {
        lhs: Value,
        rhs: Value,
    },
    I32LeS {
        lhs: Value,
        rhs: Value,
    },
    I32LeU {
        lhs: Value,
        rhs: Value,
    },
    I32GeS {
        lhs: Value,
        rhs: Value,
    },
    I32GeU {
        lhs: Value,
        rhs: Value,
    },

    /// Comparison operations (i64) - return i32 (0 or 1)
    I64Eq {
        lhs: Value,
        rhs: Value,
    },
    I64Ne {
        lhs: Value,
        rhs: Value,
    },
    I64LtS {
        lhs: Value,
        rhs: Value,
    },
    I64LtU {
        lhs: Value,
        rhs: Value,
    },
    I64GtS {
        lhs: Value,
        rhs: Value,
    },
    I64GtU {
        lhs: Value,
        rhs: Value,
    },
    I64LeS {
        lhs: Value,
        rhs: Value,
    },
    I64LeU {
        lhs: Value,
        rhs: Value,
    },
    I64GeS {
        lhs: Value,
        rhs: Value,
    },
    I64GeU {
        lhs: Value,
        rhs: Value,
    },

    /// Division and remainder operations (i32)
    I32DivS {
        lhs: Value,
        rhs: Value,
    },
    I32DivU {
        lhs: Value,
        rhs: Value,
    },
    I32RemS {
        lhs: Value,
        rhs: Value,
    },
    I32RemU {
        lhs: Value,
        rhs: Value,
    },

    /// Division and remainder operations (i64)
    I64DivS {
        lhs: Value,
        rhs: Value,
    },
    I64DivU {
        lhs: Value,
        rhs: Value,
    },
    I64RemS {
        lhs: Value,
        rhs: Value,
    },
    I64RemU {
        lhs: Value,
        rhs: Value,
    },

    /// Unary operations (i32)
    I32Eqz {
        val: Value,
    },
    I32Clz {
        val: Value,
    },
    I32Ctz {
        val: Value,
    },
    I32Popcnt {
        val: Value,
    },

    /// Unary operations (i64) - note i64.eqz returns i32 like comparisons
    I64Eqz {
        val: Value,
    },
    I64Clz {
        val: Value,
    },
    I64Ctz {
        val: Value,
    },
    I64Popcnt {
        val: Value,
    },

    /// Select instruction - (select cond true_val false_val)
    Select {
        cond: Value,
        true_val: Value,
        false_val: Value,
    },

    /// Local variable operations (Phase 12)
    LocalGet {
        idx: u32,
    },
    LocalSet {
        idx: u32,
        val: Value,
    },
    LocalTee {
        idx: u32,
        val: Value,
    },

    /// Memory operations (Phase 13 - Memory Optimization)
    I32Load {
        addr: Value,
        offset: u32,
        align: u32,
    },
    I32Store {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
    },
    I64Load {
        addr: Value,
        offset: u32,
        align: u32,
    },
    I64Store {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
    },

    // ========================================================================
    // Control Flow Operations (Phase 14 - Control Flow Representation)
    // ========================================================================
    /// Block: structured control that can be branched to
    /// Branches to this label jump past the end (forward)
    Block {
        /// Optional label for debugging
        label: Option<String>,
        /// Block type (input/output signature)
        block_type: BlockType,
        /// Body instructions (sequence)
        body: Vec<Value>,
    },

    /// Loop: structured control where branches restart
    /// Branches to this label jump to the start (backward)
    Loop {
        label: Option<String>,
        block_type: BlockType,
        body: Vec<Value>,
    },

    /// If-then-else conditional
    /// Pops i32 condition, executes then or else branch
    If {
        label: Option<String>,
        block_type: BlockType,
        condition: Value,
        then_body: Vec<Value>,
        else_body: Vec<Value>, // empty Vec for if without else
    },

    /// Unconditional branch to label
    /// Jumps to target, unwinds stack to block entry
    Br {
        /// Relative label depth (0 = innermost)
        depth: u32,
        /// Value to leave on stack (if block expects result)
        value: Option<Box<Value>>,
    },

    /// Conditional branch
    /// Pops i32 condition, if non-zero branches
    BrIf {
        depth: u32,
        condition: Value,
        value: Option<Box<Value>>,
    },

    /// Branch table (switch/case)
    /// Pops i32 index, branches to targets[index] or default
    BrTable {
        /// List of target label depths
        targets: Vec<u32>,
        /// Default label depth
        default: u32,
        /// Index to select target
        index: Value,
        /// Value to pass (if blocks expect results)
        value: Option<Box<Value>>,
    },

    /// Return from function
    /// Returns from function with values matching function signature
    Return {
        /// Return values
        values: Vec<Value>,
    },

    /// Function call (direct)
    /// Calls function by index with arguments
    Call {
        /// Function index
        func_idx: u32,
        /// Arguments
        args: Vec<Value>,
    },

    /// Function call (indirect through table)
    /// Dynamically calls function from table with type checking
    CallIndirect {
        /// Table index
        table_idx: u32,
        /// Type index (for signature checking)
        type_idx: u32,
        /// Table offset (which function in table)
        table_offset: Value,
        /// Arguments
        args: Vec<Value>,
    },

    /// Unreachable - traps execution
    Unreachable,

    /// Nop - no operation
    Nop,
}

// Include the ISLE-generated code in a module so `super::*` works
#[allow(clippy::all)]
#[allow(unused_imports)]
mod generated {
    use super::*;
    include!(concat!(env!("OUT_DIR"), "/isle_generated.rs"));
}

// Re-export generated items (if any beyond what we've manually defined)
pub use generated::*;

// ============================================================================
// Constructor implementations for ISLE extern constructors
// ============================================================================

/// Construct an i32.const value
pub fn iconst32(val: Imm32) -> Value {
    Value(Box::new(ValueData::I32Const { val }))
}

/// Construct an i32.add operation
pub fn iadd32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Add { lhs, rhs }))
}

/// Construct an i32.sub operation
pub fn isub32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Sub { lhs, rhs }))
}

/// Construct an i32.mul operation
pub fn imul32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Mul { lhs, rhs }))
}

/// Construct an i64.const value
pub fn iconst64(val: Imm64) -> Value {
    Value(Box::new(ValueData::I64Const { val }))
}

/// Construct an i64.add operation
pub fn iadd64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Add { lhs, rhs }))
}

/// Construct an i64.sub operation
pub fn isub64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Sub { lhs, rhs }))
}

/// Construct an i64.mul operation
pub fn imul64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Mul { lhs, rhs }))
}

// Bitwise operation constructors (i32)

/// Construct an i32.and operation
pub fn iand32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32And { lhs, rhs }))
}

/// Construct an i32.or operation
pub fn ior32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Or { lhs, rhs }))
}

/// Construct an i32.xor operation
pub fn ixor32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Xor { lhs, rhs }))
}

/// Construct an i32.shl operation
pub fn ishl32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Shl { lhs, rhs }))
}

/// Construct an i32.shr_s operation (arithmetic/signed right shift)
pub fn ishrs32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32ShrS { lhs, rhs }))
}

/// Construct an i32.shr_u operation (logical/unsigned right shift)
pub fn ishru32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32ShrU { lhs, rhs }))
}

// Bitwise operation constructors (i64)

/// Construct an i64.and operation
pub fn iand64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64And { lhs, rhs }))
}

/// Construct an i64.or operation
pub fn ior64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Or { lhs, rhs }))
}

/// Construct an i64.xor operation
pub fn ixor64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Xor { lhs, rhs }))
}

/// Construct an i64.shl operation
pub fn ishl64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Shl { lhs, rhs }))
}

/// Construct an i64.shr_s operation (arithmetic/signed right shift)
pub fn ishrs64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64ShrS { lhs, rhs }))
}

/// Construct an i64.shr_u operation (logical/unsigned right shift)
pub fn ishru64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64ShrU { lhs, rhs }))
}

// Comparison operation constructors (i32) - return i32 (0 or 1)

/// Construct an i32.eq operation
pub fn ieq32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Eq { lhs, rhs }))
}

/// Construct an i32.ne operation
pub fn ine32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Ne { lhs, rhs }))
}

/// Construct an i32.lt_s operation (signed less than)
pub fn ilts32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32LtS { lhs, rhs }))
}

/// Construct an i32.lt_u operation (unsigned less than)
pub fn iltu32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32LtU { lhs, rhs }))
}

/// Construct an i32.gt_s operation (signed greater than)
pub fn igts32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32GtS { lhs, rhs }))
}

/// Construct an i32.gt_u operation (unsigned greater than)
pub fn igtu32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32GtU { lhs, rhs }))
}

/// Construct an i32.le_s operation (signed less than or equal)
pub fn iles32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32LeS { lhs, rhs }))
}

/// Construct an i32.le_u operation (unsigned less than or equal)
pub fn ileu32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32LeU { lhs, rhs }))
}

/// Construct an i32.ge_s operation (signed greater than or equal)
pub fn iges32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32GeS { lhs, rhs }))
}

/// Construct an i32.ge_u operation (unsigned greater than or equal)
pub fn igeu32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32GeU { lhs, rhs }))
}

// Comparison operation constructors (i64) - return i32 (0 or 1)

/// Construct an i64.eq operation
pub fn ieq64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Eq { lhs, rhs }))
}

/// Construct an i64.ne operation
pub fn ine64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Ne { lhs, rhs }))
}

/// Construct an i64.lt_s operation (signed less than)
pub fn ilts64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64LtS { lhs, rhs }))
}

/// Construct an i64.lt_u operation (unsigned less than)
pub fn iltu64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64LtU { lhs, rhs }))
}

/// Construct an i64.gt_s operation (signed greater than)
pub fn igts64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64GtS { lhs, rhs }))
}

/// Construct an i64.gt_u operation (unsigned greater than)
pub fn igtu64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64GtU { lhs, rhs }))
}

/// Construct an i64.le_s operation (signed less than or equal)
pub fn iles64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64LeS { lhs, rhs }))
}

/// Construct an i64.le_u operation (unsigned less than or equal)
pub fn ileu64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64LeU { lhs, rhs }))
}

/// Construct an i64.ge_s operation (signed greater than or equal)
pub fn iges64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64GeS { lhs, rhs }))
}

/// Construct an i64.ge_u operation (unsigned greater than or equal)
pub fn igeu64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64GeU { lhs, rhs }))
}

// Division and remainder operation constructors (i32)

/// Construct an i32.div_s operation (signed division)
pub fn idivs32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32DivS { lhs, rhs }))
}

/// Construct an i32.div_u operation (unsigned division)
pub fn idivu32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32DivU { lhs, rhs }))
}

/// Construct an i32.rem_s operation (signed remainder)
pub fn irems32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32RemS { lhs, rhs }))
}

/// Construct an i32.rem_u operation (unsigned remainder)
pub fn iremu32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32RemU { lhs, rhs }))
}

// Division and remainder operation constructors (i64)

/// Construct an i64.div_s operation (signed division)
pub fn idivs64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64DivS { lhs, rhs }))
}

/// Construct an i64.div_u operation (unsigned division)
pub fn idivu64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64DivU { lhs, rhs }))
}

/// Construct an i64.rem_s operation (signed remainder)
pub fn irems64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64RemS { lhs, rhs }))
}

/// Construct an i64.rem_u operation (unsigned remainder)
pub fn iremu64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64RemU { lhs, rhs }))
}

// Unary operation constructors (i32)

/// Construct an i32.eqz operation (test if zero)
pub fn ieqz32(val: Value) -> Value {
    Value(Box::new(ValueData::I32Eqz { val }))
}

/// Construct an i32.clz operation (count leading zeros)
pub fn iclz32(val: Value) -> Value {
    Value(Box::new(ValueData::I32Clz { val }))
}

/// Construct an i32.ctz operation (count trailing zeros)
pub fn ictz32(val: Value) -> Value {
    Value(Box::new(ValueData::I32Ctz { val }))
}

/// Construct an i32.popcnt operation (count set bits)
pub fn ipopcnt32(val: Value) -> Value {
    Value(Box::new(ValueData::I32Popcnt { val }))
}

// Unary operation constructors (i64)

/// Construct an i64.eqz operation (test if zero, returns i32)
pub fn ieqz64(val: Value) -> Value {
    Value(Box::new(ValueData::I64Eqz { val }))
}

/// Construct an i64.clz operation (count leading zeros)
pub fn iclz64(val: Value) -> Value {
    Value(Box::new(ValueData::I64Clz { val }))
}

/// Construct an i64.ctz operation (count trailing zeros)
pub fn ictz64(val: Value) -> Value {
    Value(Box::new(ValueData::I64Ctz { val }))
}

/// Construct an i64.popcnt operation (count set bits)
pub fn ipopcnt64(val: Value) -> Value {
    Value(Box::new(ValueData::I64Popcnt { val }))
}

/// Construct a select instruction (select cond true_val false_val)
pub fn select_instr(cond: Value, true_val: Value, false_val: Value) -> Value {
    Value(Box::new(ValueData::Select {
        cond,
        true_val,
        false_val,
    }))
}

/// Construct a local.get operation
pub fn local_get(idx: u32) -> Value {
    Value(Box::new(ValueData::LocalGet { idx }))
}

/// Construct a local.set operation
pub fn local_set(idx: u32, val: Value) -> Value {
    Value(Box::new(ValueData::LocalSet { idx, val }))
}

/// Construct a local.tee operation
pub fn local_tee(idx: u32, val: Value) -> Value {
    Value(Box::new(ValueData::LocalTee { idx, val }))
}

/// Construct an i32.load operation
pub fn i32_load(addr: Value, offset: u32, align: u32) -> Value {
    Value(Box::new(ValueData::I32Load {
        addr,
        offset,
        align,
    }))
}

/// Construct an i32.store operation
pub fn i32_store(addr: Value, value: Value, offset: u32, align: u32) -> Value {
    Value(Box::new(ValueData::I32Store {
        addr,
        value,
        offset,
        align,
    }))
}

/// Construct an i64.load operation
pub fn i64_load(addr: Value, offset: u32, align: u32) -> Value {
    Value(Box::new(ValueData::I64Load {
        addr,
        offset,
        align,
    }))
}

/// Construct an i64.store operation
pub fn i64_store(addr: Value, value: Value, offset: u32, align: u32) -> Value {
    Value(Box::new(ValueData::I64Store {
        addr,
        value,
        offset,
        align,
    }))
}

// ============================================================================
// Control Flow Constructors (Phase 14)
// ============================================================================

/// Construct a block
pub fn block(label: Option<String>, block_type: BlockType, body: Vec<Value>) -> Value {
    Value(Box::new(ValueData::Block {
        label,
        block_type,
        body,
    }))
}

/// Construct a loop
pub fn loop_construct(label: Option<String>, block_type: BlockType, body: Vec<Value>) -> Value {
    Value(Box::new(ValueData::Loop {
        label,
        block_type,
        body,
    }))
}

/// Construct an if-then-else
pub fn if_then_else(
    label: Option<String>,
    block_type: BlockType,
    condition: Value,
    then_body: Vec<Value>,
    else_body: Vec<Value>,
) -> Value {
    Value(Box::new(ValueData::If {
        label,
        block_type,
        condition,
        then_body,
        else_body,
    }))
}

/// Construct an unconditional branch
pub fn br(depth: u32, value: Option<Value>) -> Value {
    Value(Box::new(ValueData::Br {
        depth,
        value: value.map(Box::new),
    }))
}

/// Construct a conditional branch
pub fn br_if(depth: u32, condition: Value, value: Option<Value>) -> Value {
    Value(Box::new(ValueData::BrIf {
        depth,
        condition,
        value: value.map(Box::new),
    }))
}

/// Construct a branch table
pub fn br_table(targets: Vec<u32>, default: u32, index: Value, value: Option<Value>) -> Value {
    Value(Box::new(ValueData::BrTable {
        targets,
        default,
        index,
        value: value.map(Box::new),
    }))
}

/// Construct a return
pub fn return_val(values: Vec<Value>) -> Value {
    Value(Box::new(ValueData::Return { values }))
}

/// Construct a direct function call
pub fn call(func_idx: u32, args: Vec<Value>) -> Value {
    Value(Box::new(ValueData::Call { func_idx, args }))
}

/// Construct an indirect function call
pub fn call_indirect(
    table_idx: u32,
    type_idx: u32,
    table_offset: Value,
    args: Vec<Value>,
) -> Value {
    Value(Box::new(ValueData::CallIndirect {
        table_idx,
        type_idx,
        table_offset,
        args,
    }))
}

/// Construct an unreachable instruction
pub fn unreachable() -> Value {
    Value(Box::new(ValueData::Unreachable))
}

/// Construct a nop instruction
pub fn nop() -> Value {
    Value(Box::new(ValueData::Nop))
}

/// Extract ValueData from Value (for ISLE pattern matching)
pub fn value_data(v: &Value) -> Option<ValueData> {
    Some((*v.0).clone())
}

// ============================================================================
// ISLE Control Flow Constructor Wrappers (Issue #12)
// ============================================================================
//
// These wrappers adapt between ISLE's type system (which uses primitive types
// like OptionString and InstructionList) and Rust's native types.
// Since control flow optimization is primarily handled in Rust passes rather
// than ISLE rules, these create placeholder structures.

/// Block constructor for ISLE
pub fn block_instr(label_opt: OptionString, block_type: BlockType, body: InstructionList) -> Value {
    // Convert OptionString to Option<String>
    let label = label_opt.0;
    // For now, create empty body since control flow optimization is in Rust
    Value(Box::new(ValueData::Block {
        label,
        block_type,
        body: Vec::new(), // Placeholder - actual bodies handled in Rust
    }))
}

/// Loop constructor for ISLE
pub fn loop_instr(label_opt: OptionString, block_type: BlockType, body: InstructionList) -> Value {
    let label = label_opt.0;
    Value(Box::new(ValueData::Loop {
        label,
        block_type,
        body: Vec::new(),
    }))
}

/// If constructor for ISLE
pub fn if_instr(cond: Value, block_type: BlockType, then_body: InstructionList, else_body: InstructionList) -> Value {
    let label = None; // ISLE version doesn't include label
    Value(Box::new(ValueData::If {
        label,
        block_type,
        condition: cond,
        then_body: Vec::new(),
        else_body: Vec::new(),
    }))
}

/// Branch constructor for ISLE
pub fn br_instr(depth: u32) -> Value {
    Value(Box::new(ValueData::Br {
        depth,
        value: None,
    }))
}

/// Conditional branch constructor for ISLE
pub fn br_if_instr(cond: Value, depth: u32) -> Value {
    Value(Box::new(ValueData::BrIf {
        depth,
        condition: cond,
        value: None,
    }))
}

/// Call constructor for ISLE
pub fn call_instr(func_idx: u32) -> Value {
    Value(Box::new(ValueData::Call {
        func_idx,
        args: Vec::new(),
    }))
}

/// Return constructor for ISLE
pub fn return_instr() -> Value {
    Value(Box::new(ValueData::Return {
        values: Vec::new(),
    }))
}

/// BlockType::Empty constructor
pub fn block_type_empty() -> BlockType {
    BlockType::Empty
}

/// BlockType::I32Result constructor
pub fn block_type_i32() -> BlockType {
    BlockType::Value(ValueType::I32)
}

/// BlockType::I64Result constructor
pub fn block_type_i64() -> BlockType {
    BlockType::Value(ValueType::I64)
}

// ============================================================================
// Helper Functions for Optimization Rules
// ============================================================================

/// Add two Imm32 values with wrapping overflow semantics (matching WebAssembly i32.add)
pub fn imm32_add(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0.wrapping_add(rhs.0))
}

/// Subtract two Imm32 values with wrapping overflow semantics (matching WebAssembly i32.sub)
pub fn imm32_sub(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0.wrapping_sub(rhs.0))
}

/// Multiply two Imm32 values with wrapping overflow semantics (matching WebAssembly i32.mul)
pub fn imm32_mul(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0.wrapping_mul(rhs.0))
}

/// Add two Imm64 values with wrapping overflow semantics (matching WebAssembly i64.add)
pub fn imm64_add(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0.wrapping_add(rhs.0))
}

/// Subtract two Imm64 values with wrapping overflow semantics (matching WebAssembly i64.sub)
pub fn imm64_sub(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0.wrapping_sub(rhs.0))
}

/// Multiply two Imm64 values with wrapping overflow semantics (matching WebAssembly i64.mul)
pub fn imm64_mul(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0.wrapping_mul(rhs.0))
}

// Bitwise helper functions (i32)

/// Bitwise AND for 32-bit immediates
pub fn imm32_and(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0 & rhs.0)
}

/// Bitwise OR for 32-bit immediates
pub fn imm32_or(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0 | rhs.0)
}

/// Bitwise XOR for 32-bit immediates
pub fn imm32_xor(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0 ^ rhs.0)
}

/// Shift left for 32-bit immediates (WebAssembly masks shift amount to 0-31)
pub fn imm32_shl(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0.wrapping_shl((rhs.0 & 0x1F) as u32))
}

/// Arithmetic (signed) shift right for 32-bit immediates
pub fn imm32_shr_s(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(lhs.0.wrapping_shr((rhs.0 & 0x1F) as u32))
}

/// Logical (unsigned) shift right for 32-bit immediates
pub fn imm32_shr_u(lhs: Imm32, rhs: Imm32) -> Imm32 {
    Imm32(((lhs.0 as u32).wrapping_shr((rhs.0 & 0x1F) as u32)) as i32)
}

// Bitwise helper functions (i64)

/// Bitwise AND for 64-bit immediates
pub fn imm64_and(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0 & rhs.0)
}

/// Bitwise OR for 64-bit immediates
pub fn imm64_or(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0 | rhs.0)
}

/// Bitwise XOR for 64-bit immediates
pub fn imm64_xor(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0 ^ rhs.0)
}

/// Shift left for 64-bit immediates (WebAssembly masks shift amount to 0-63)
pub fn imm64_shl(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0.wrapping_shl((rhs.0 & 0x3F) as u32))
}

/// Arithmetic (signed) shift right for 64-bit immediates
pub fn imm64_shr_s(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(lhs.0.wrapping_shr((rhs.0 & 0x3F) as u32))
}

/// Logical (unsigned) shift right for 64-bit immediates
pub fn imm64_shr_u(lhs: Imm64, rhs: Imm64) -> Imm64 {
    Imm64(((lhs.0 as u64).wrapping_shr((rhs.0 & 0x3F) as u32)) as i64)
}

/// Memory location identifier: (base_address, offset)
/// We track memory as (const_addr + offset) pairs
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MemoryLocation {
    /// Base address (if known constant)
    base: Option<i32>,
    /// Static offset
    offset: u32,
}

/// Environment for dataflow analysis
pub struct OptimizationEnv {
    /// Local variable constants
    pub locals: std::collections::HashMap<u32, Value>,
    /// Memory state: location → stored value
    pub memory: std::collections::HashMap<MemoryLocation, Value>,
}

impl Default for OptimizationEnv {
    fn default() -> Self {
        Self::new()
    }
}

impl OptimizationEnv {
    pub fn new() -> Self {
        OptimizationEnv {
            locals: std::collections::HashMap::new(),
            memory: std::collections::HashMap::new(),
        }
    }

    /// Invalidate all memory state (conservative, on unknown stores/calls)
    pub fn invalidate_memory(&mut self) {
        self.memory.clear();
    }
}

/// Legacy type alias for compatibility
pub type LocalEnv = OptimizationEnv;

/// Simplify/optimize a Value term with dataflow-aware optimization
/// Phase 13: Extended with memory redundancy elimination
pub fn simplify_with_env(val: Value, env: &mut OptimizationEnv) -> Value {
    match val.data() {
        // Local variable operations
        ValueData::LocalSet { idx, val: set_val } => {
            let simplified_val = simplify_with_env(set_val.clone(), env);

            // Track this assignment in our environment
            if matches!(
                simplified_val.data(),
                ValueData::I32Const { .. } | ValueData::I64Const { .. }
            ) {
                env.locals.insert(*idx, simplified_val.clone());
            } else {
                env.locals.remove(idx);
            }

            local_set(*idx, simplified_val)
        }

        ValueData::LocalGet { idx } => {
            // Look up in environment - dataflow analysis!
            if let Some(known_val) = env.locals.get(idx) {
                known_val.clone()
            } else {
                local_get(*idx)
            }
        }

        ValueData::LocalTee { idx, val: tee_val } => {
            let simplified_val = simplify_with_env(tee_val.clone(), env);

            if matches!(
                simplified_val.data(),
                ValueData::I32Const { .. } | ValueData::I64Const { .. }
            ) {
                env.locals.insert(*idx, simplified_val.clone());
            } else {
                env.locals.remove(idx);
            }

            local_tee(*idx, simplified_val)
        }

        // Memory operations - Phase 13: Memory Redundancy Elimination!
        ValueData::I32Load {
            addr,
            offset,
            align,
        } => {
            let simplified_addr = simplify_with_env(addr.clone(), env);

            // Try to extract memory location
            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                };

                // Redundant load elimination: check if we know this value!
                if let Some(known_value) = env.memory.get(&mem_loc) {
                    // OPTIMIZATION: Return known value instead of loading!
                    return known_value.clone();
                }
            }

            i32_load(simplified_addr, *offset, *align)
        }

        ValueData::I32Store {
            addr,
            value,
            offset,
            align,
        } => {
            let simplified_addr = simplify_with_env(addr.clone(), env);
            let simplified_value = simplify_with_env(value.clone(), env);

            // Track this store in memory state
            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                };

                // Store the value in our memory tracking
                if matches!(simplified_value.data(), ValueData::I32Const { .. }) {
                    env.memory.insert(mem_loc, simplified_value.clone());
                }
            } else {
                // Unknown address - invalidate all memory conservatively
                env.invalidate_memory();
            }

            i32_store(simplified_addr, simplified_value, *offset, *align)
        }

        ValueData::I64Load {
            addr,
            offset,
            align,
        } => {
            let simplified_addr = simplify_with_env(addr.clone(), env);

            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                };

                if let Some(known_value) = env.memory.get(&mem_loc) {
                    return known_value.clone();
                }
            }

            i64_load(simplified_addr, *offset, *align)
        }

        ValueData::I64Store {
            addr,
            value,
            offset,
            align,
        } => {
            let simplified_addr = simplify_with_env(addr.clone(), env);
            let simplified_value = simplify_with_env(value.clone(), env);

            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                };

                if matches!(simplified_value.data(), ValueData::I64Const { .. }) {
                    env.memory.insert(mem_loc, simplified_value.clone());
                }
            } else {
                env.invalidate_memory();
            }

            i64_store(simplified_addr, simplified_value, *offset, *align)
        }

        // All other optimizations follow...
        _ => simplify_stateless(val),
    }
}

/// Original stateless simplification (for compatibility)
pub fn simplify(val: Value) -> Value {
    let mut env = LocalEnv::new();
    simplify_with_env(val, &mut env)
}

/// Check if two values are structurally equal
/// This is used for optimizations like x ^ x = 0, x & x = x, x | x = x
fn are_values_equal(lhs: &Value, rhs: &Value) -> bool {
    match (lhs.data(), rhs.data()) {
        // Constants are equal if their values match
        (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => l.value() == r.value(),
        (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => l.value() == r.value(),

        // LocalGet is equal if same index
        (ValueData::LocalGet { idx: l }, ValueData::LocalGet { idx: r }) => l == r,

        // Binary operations are equal if operation and operands match
        (ValueData::I32Add { lhs: l1, rhs: r1 }, ValueData::I32Add { lhs: l2, rhs: r2 }) => {
            are_values_equal(l1, l2) && are_values_equal(r1, r2)
        }
        (ValueData::I32Sub { lhs: l1, rhs: r1 }, ValueData::I32Sub { lhs: l2, rhs: r2 }) => {
            are_values_equal(l1, l2) && are_values_equal(r1, r2)
        }
        (ValueData::I32Mul { lhs: l1, rhs: r1 }, ValueData::I32Mul { lhs: l2, rhs: r2 }) => {
            are_values_equal(l1, l2) && are_values_equal(r1, r2)
        }

        // For other cases, conservatively return false
        // We could expand this for more cases, but these cover the common patterns
        _ => false,
    }
}

/// Stateless simplification (expression-level only)
fn simplify_stateless(val: Value) -> Value {
    match val.data() {
        // i32.add optimizations
        ValueData::I32Add { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding: (i32.add (i32.const A) (i32.const B)) → (i32.const (A+B))
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_add(*lhs_val, *rhs_val))
                }
                // Algebraic: x + 0 = x
                (_, ValueData::I32Const { val }) if val.value() == 0 => lhs_simplified,
                // Algebraic: 0 + x = x
                (ValueData::I32Const { val }, _) if val.value() == 0 => rhs_simplified,
                _ => iadd32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.sub optimizations
        ValueData::I32Sub { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x - x = 0 pattern (self-subtraction)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst32(Imm32(0));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding: (i32.sub (i32.const A) (i32.const B)) → (i32.const (A-B))
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_sub(*lhs_val, *rhs_val))
                }
                // Algebraic: x - 0 = x
                (_, ValueData::I32Const { val }) if val.value() == 0 => lhs_simplified,
                _ => isub32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.mul optimizations
        ValueData::I32Mul { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding: (i32.mul (i32.const A) (i32.const B)) → (i32.const (A*B))
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_mul(*lhs_val, *rhs_val))
                }
                // Algebraic: x * 0 = 0
                (_, ValueData::I32Const { val }) if val.value() == 0 => iconst32(Imm32(0)),
                // Algebraic: 0 * x = 0
                (ValueData::I32Const { val }, _) if val.value() == 0 => iconst32(Imm32(0)),
                // Algebraic: x * 1 = x
                (_, ValueData::I32Const { val }) if val.value() == 1 => lhs_simplified,
                // Algebraic: 1 * x = x
                (ValueData::I32Const { val }, _) if val.value() == 1 => rhs_simplified,
                _ => imul32(lhs_simplified, rhs_simplified),
            }
        }

        // i64.add optimizations
        ValueData::I64Add { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_add(*lhs_val, *rhs_val))
                }
                (_, ValueData::I64Const { val }) if val.value() == 0 => lhs_simplified,
                (ValueData::I64Const { val }, _) if val.value() == 0 => rhs_simplified,
                _ => iadd64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.sub optimizations
        ValueData::I64Sub { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x - x = 0 pattern (self-subtraction)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst64(Imm64(0));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_sub(*lhs_val, *rhs_val))
                }
                (_, ValueData::I64Const { val }) if val.value() == 0 => lhs_simplified,
                _ => isub64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.mul optimizations
        ValueData::I64Mul { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_mul(*lhs_val, *rhs_val))
                }
                (_, ValueData::I64Const { val }) if val.value() == 0 => iconst64(Imm64(0)),
                (ValueData::I64Const { val }, _) if val.value() == 0 => iconst64(Imm64(0)),
                (_, ValueData::I64Const { val }) if val.value() == 1 => lhs_simplified,
                (ValueData::I64Const { val }, _) if val.value() == 1 => rhs_simplified,
                _ => imul64(lhs_simplified, rhs_simplified),
            }
        }

        // i32.and optimizations
        ValueData::I32And { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x & x = x pattern (self-AND)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return lhs_simplified;
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_and(*lhs_val, *rhs_val))
                }
                // Algebraic: x & 0 = 0
                (_, ValueData::I32Const { val }) if val.value() == 0 => iconst32(Imm32(0)),
                (ValueData::I32Const { val }, _) if val.value() == 0 => iconst32(Imm32(0)),
                // Algebraic: x & -1 = x (all bits set)
                (_, ValueData::I32Const { val }) if val.value() == -1 => lhs_simplified,
                (ValueData::I32Const { val }, _) if val.value() == -1 => rhs_simplified,
                _ => iand32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.or optimizations
        ValueData::I32Or { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x | x = x pattern (self-OR)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return lhs_simplified;
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_or(*lhs_val, *rhs_val))
                }
                // Algebraic: x | 0 = x
                (_, ValueData::I32Const { val }) if val.value() == 0 => lhs_simplified,
                (ValueData::I32Const { val }, _) if val.value() == 0 => rhs_simplified,
                // Algebraic: x | -1 = -1 (all bits set)
                (_, ValueData::I32Const { val }) if val.value() == -1 => iconst32(Imm32(-1)),
                (ValueData::I32Const { val }, _) if val.value() == -1 => iconst32(Imm32(-1)),
                _ => ior32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.xor optimizations
        ValueData::I32Xor { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x ^ x = 0 pattern (self-XOR)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst32(Imm32(0));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_xor(*lhs_val, *rhs_val))
                }
                // Algebraic: x ^ 0 = x
                (_, ValueData::I32Const { val }) if val.value() == 0 => lhs_simplified,
                (ValueData::I32Const { val }, _) if val.value() == 0 => rhs_simplified,
                _ => ixor32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.shl optimizations
        ValueData::I32Shl { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_shl(*lhs_val, *rhs_val))
                }
                // Algebraic: x << 0 = x
                (_, ValueData::I32Const { val }) if (val.value() & 0x1F) == 0 => lhs_simplified,
                _ => ishl32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.shr_s optimizations
        ValueData::I32ShrS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_shr_s(*lhs_val, *rhs_val))
                }
                // Algebraic: x >> 0 = x
                (_, ValueData::I32Const { val }) if (val.value() & 0x1F) == 0 => lhs_simplified,
                _ => ishrs32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.shr_u optimizations
        ValueData::I32ShrU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_shr_u(*lhs_val, *rhs_val))
                }
                // Algebraic: x >> 0 = x
                (_, ValueData::I32Const { val }) if (val.value() & 0x1F) == 0 => lhs_simplified,
                _ => ishru32(lhs_simplified, rhs_simplified),
            }
        }

        // i64.and optimizations
        ValueData::I64And { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x & x = x pattern (self-AND)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return lhs_simplified;
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_and(*lhs_val, *rhs_val))
                }
                // Algebraic: x & 0 = 0
                (_, ValueData::I64Const { val }) if val.value() == 0 => iconst64(Imm64(0)),
                (ValueData::I64Const { val }, _) if val.value() == 0 => iconst64(Imm64(0)),
                // Algebraic: x & -1 = x (all bits set)
                (_, ValueData::I64Const { val }) if val.value() == -1 => lhs_simplified,
                (ValueData::I64Const { val }, _) if val.value() == -1 => rhs_simplified,
                _ => iand64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.or optimizations
        ValueData::I64Or { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x | x = x pattern (self-OR)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return lhs_simplified;
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_or(*lhs_val, *rhs_val))
                }
                // Algebraic: x | 0 = x
                (_, ValueData::I64Const { val }) if val.value() == 0 => lhs_simplified,
                (ValueData::I64Const { val }, _) if val.value() == 0 => rhs_simplified,
                // Algebraic: x | -1 = -1 (all bits set)
                (_, ValueData::I64Const { val }) if val.value() == -1 => iconst64(Imm64(-1)),
                (ValueData::I64Const { val }, _) if val.value() == -1 => iconst64(Imm64(-1)),
                _ => ior64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.xor optimizations
        ValueData::I64Xor { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x ^ x = 0 pattern (self-XOR)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst64(Imm64(0));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_xor(*lhs_val, *rhs_val))
                }
                // Algebraic: x ^ 0 = x
                (_, ValueData::I64Const { val }) if val.value() == 0 => lhs_simplified,
                (ValueData::I64Const { val }, _) if val.value() == 0 => rhs_simplified,
                _ => ixor64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.shl optimizations
        ValueData::I64Shl { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_shl(*lhs_val, *rhs_val))
                }
                // Algebraic: x << 0 = x
                (_, ValueData::I64Const { val }) if (val.value() & 0x3F) == 0 => lhs_simplified,
                _ => ishl64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.shr_s optimizations
        ValueData::I64ShrS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_shr_s(*lhs_val, *rhs_val))
                }
                // Algebraic: x >> 0 = x
                (_, ValueData::I64Const { val }) if (val.value() & 0x3F) == 0 => lhs_simplified,
                _ => ishrs64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.shr_u optimizations
        ValueData::I64ShrU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_shr_u(*lhs_val, *rhs_val))
                }
                // Algebraic: x >> 0 = x
                (_, ValueData::I64Const { val }) if (val.value() & 0x3F) == 0 => lhs_simplified,
                _ => ishru64(lhs_simplified, rhs_simplified),
            }
        }

        // Comparison optimizations (i32)
        ValueData::I32Eq { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x == x = 1 pattern (self-equality always true)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst32(Imm32(1));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() == r.value() { 1 } else { 0 }))
                }
                _ => ieq32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32Ne { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x != x = 0 pattern (self-inequality always false)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst32(Imm32(0));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() != r.value() { 1 } else { 0 }))
                }
                _ => ine32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32LtS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() < r.value() { 1 } else { 0 }))
                }
                _ => ilts32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32LtU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u32) < (r.value() as u32) {
                        1
                    } else {
                        0
                    }))
                }
                _ => iltu32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32GtS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() > r.value() { 1 } else { 0 }))
                }
                _ => igts32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32GtU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u32) > (r.value() as u32) {
                        1
                    } else {
                        0
                    }))
                }
                _ => igtu32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32LeS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() <= r.value() { 1 } else { 0 }))
                }
                _ => iles32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32LeU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u32) <= (r.value() as u32) {
                        1
                    } else {
                        0
                    }))
                }
                _ => ileu32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32GeS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() >= r.value() { 1 } else { 0 }))
                }
                _ => iges32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32GeU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u32) >= (r.value() as u32) {
                        1
                    } else {
                        0
                    }))
                }
                _ => igeu32(lhs_simplified, rhs_simplified),
            }
        }

        // Comparison optimizations (i64)
        ValueData::I64Eq { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x == x = 1 pattern (self-equality always true)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst32(Imm32(1));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() == r.value() { 1 } else { 0 }))
                }
                _ => ieq64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64Ne { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());

            // Check for x != x = 0 pattern (self-inequality always false)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst32(Imm32(0));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() != r.value() { 1 } else { 0 }))
                }
                _ => ine64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64LtS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() < r.value() { 1 } else { 0 }))
                }
                _ => ilts64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64LtU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u64) < (r.value() as u64) {
                        1
                    } else {
                        0
                    }))
                }
                _ => iltu64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64GtS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() > r.value() { 1 } else { 0 }))
                }
                _ => igts64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64GtU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u64) > (r.value() as u64) {
                        1
                    } else {
                        0
                    }))
                }
                _ => igtu64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64LeS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() <= r.value() { 1 } else { 0 }))
                }
                _ => iles64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64LeU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u64) <= (r.value() as u64) {
                        1
                    } else {
                        0
                    }))
                }
                _ => ileu64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64GeS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() >= r.value() { 1 } else { 0 }))
                }
                _ => iges64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64GeU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if (l.value() as u64) >= (r.value() as u64) {
                        1
                    } else {
                        0
                    }))
                }
                _ => igeu64(lhs_simplified, rhs_simplified),
            }
        }

        // Division and remainder optimizations (i32) - constant folding only, avoid division by zero
        ValueData::I32DivS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 =>
                {
                    // Signed division with wrapping for overflow case (INT_MIN / -1)
                    iconst32(Imm32(l.value().wrapping_div(r.value())))
                }
                _ => idivs32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32DivU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst32(Imm32(((l.value() as u32) / (r.value() as u32)) as i32))
                }
                _ => idivu32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32RemS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst32(Imm32(l.value().wrapping_rem(r.value())))
                }
                _ => irems32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32RemU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst32(Imm32(((l.value() as u32) % (r.value() as u32)) as i32))
                }
                _ => iremu32(lhs_simplified, rhs_simplified),
            }
        }

        // Division and remainder optimizations (i64) - constant folding only, avoid division by zero
        ValueData::I64DivS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst64(Imm64(l.value().wrapping_div(r.value())))
                }
                _ => idivs64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64DivU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst64(Imm64(((l.value() as u64) / (r.value() as u64)) as i64))
                }
                _ => idivu64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64RemS { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst64(Imm64(l.value().wrapping_rem(r.value())))
                }
                _ => irems64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64RemU { lhs, rhs } => {
            let lhs_simplified = simplify(lhs.clone());
            let rhs_simplified = simplify(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst64(Imm64(((l.value() as u64) % (r.value() as u64)) as i64))
                }
                _ => iremu64(lhs_simplified, rhs_simplified),
            }
        }

        // Unary operations (i32) optimizations
        ValueData::I32Eqz { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.eqz (i32.const N)) → (i32.const (N == 0 ? 1 : 0))
                ValueData::I32Const { val: v } => {
                    iconst32(Imm32(if v.value() == 0 { 1 } else { 0 }))
                }
                _ => ieqz32(val_simplified),
            }
        }

        ValueData::I32Clz { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.clz (i32.const N)) → (i32.const count_leading_zeros(N))
                ValueData::I32Const { val: v } => iconst32(Imm32(v.value().leading_zeros() as i32)),
                _ => iclz32(val_simplified),
            }
        }

        ValueData::I32Ctz { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.ctz (i32.const N)) → (i32.const count_trailing_zeros(N))
                ValueData::I32Const { val: v } => {
                    iconst32(Imm32(v.value().trailing_zeros() as i32))
                }
                _ => ictz32(val_simplified),
            }
        }

        ValueData::I32Popcnt { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.popcnt (i32.const N)) → (i32.const count_ones(N))
                ValueData::I32Const { val: v } => iconst32(Imm32(v.value().count_ones() as i32)),
                _ => ipopcnt32(val_simplified),
            }
        }

        // Unary operations (i64) optimizations
        ValueData::I64Eqz { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.eqz (i64.const N)) → (i32.const (N == 0 ? 1 : 0))
                ValueData::I64Const { val: v } => {
                    iconst32(Imm32(if v.value() == 0 { 1 } else { 0 }))
                }
                _ => ieqz64(val_simplified),
            }
        }

        ValueData::I64Clz { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.clz (i64.const N)) → (i64.const count_leading_zeros(N))
                ValueData::I64Const { val: v } => iconst64(Imm64(v.value().leading_zeros() as i64)),
                _ => iclz64(val_simplified),
            }
        }

        ValueData::I64Ctz { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.ctz (i64.const N)) → (i64.const count_trailing_zeros(N))
                ValueData::I64Const { val: v } => {
                    iconst64(Imm64(v.value().trailing_zeros() as i64))
                }
                _ => ictz64(val_simplified),
            }
        }

        ValueData::I64Popcnt { val } => {
            let val_simplified = simplify(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.popcnt (i64.const N)) → (i64.const count_ones(N))
                ValueData::I64Const { val: v } => iconst64(Imm64(v.value().count_ones() as i64)),
                _ => ipopcnt64(val_simplified),
            }
        }

        // Select instruction optimization
        ValueData::Select {
            cond,
            true_val,
            false_val,
        } => {
            let cond_simplified = simplify(cond.clone());
            let true_simplified = simplify(true_val.clone());
            let false_simplified = simplify(false_val.clone());

            match cond_simplified.data() {
                // Constant folding: (select (i32.const 0) true false) → false
                ValueData::I32Const { val } if val.value() == 0 => false_simplified,
                // Constant folding: (select (i32.const N) true false) → true (N != 0)
                ValueData::I32Const { .. } => true_simplified,
                _ => select_instr(cond_simplified, true_simplified, false_simplified),
            }
        }

        // Constants are already in simplest form
        _ => val,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_iconst32() {
        let val = iconst32(Imm32::from(42));
        match val.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 42);
            }
            _ => panic!("Expected I32Const"),
        }
    }

    #[test]
    fn test_iconst64() {
        let val = iconst64(Imm64::from(42));
        match val.data() {
            ValueData::I64Const { val } => {
                assert_eq!(val.value(), 42);
            }
            _ => panic!("Expected I64Const"),
        }
    }

    #[test]
    fn test_iadd32() {
        let lhs = iconst32(Imm32::from(10));
        let rhs = iconst32(Imm32::from(32));
        let add = iadd32(lhs, rhs);

        match add.data() {
            ValueData::I32Add { lhs, rhs } => match (lhs.data(), rhs.data()) {
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    assert_eq!(lhs_val.value(), 10);
                    assert_eq!(rhs_val.value(), 32);
                }
                _ => panic!("Expected I32Const operands"),
            },
            _ => panic!("Expected I32Add"),
        }
    }

    #[test]
    fn test_nested_expr() {
        // Test (i32.add (i32.const 5) (i32.add (i32.const 10) (i32.const 20)))
        let inner_add = iadd32(iconst32(Imm32::from(10)), iconst32(Imm32::from(20)));
        let outer_add = iadd32(iconst32(Imm32::from(5)), inner_add);

        match outer_add.data() {
            ValueData::I32Add { lhs, rhs } => {
                // lhs should be I32Const(5)
                match lhs.data() {
                    ValueData::I32Const { val } => assert_eq!(val.value(), 5),
                    _ => panic!("Expected I32Const for lhs"),
                }
                // rhs should be I32Add
                match rhs.data() {
                    ValueData::I32Add { .. } => {
                        // Success - nested structure verified
                    }
                    _ => panic!("Expected I32Add for rhs"),
                }
            }
            _ => panic!("Expected I32Add"),
        }
    }

    #[test]
    fn test_imm32_add() {
        let a = Imm32::from(10);
        let b = Imm32::from(32);
        let result = imm32_add(a, b);
        assert_eq!(result.value(), 42);
    }

    #[test]
    fn test_imm32_add_overflow() {
        // Test wrapping behavior
        let a = Imm32::from(i32::MAX);
        let b = Imm32::from(1);
        let result = imm32_add(a, b);
        assert_eq!(result.value(), i32::MIN); // Wraps around
    }

    #[test]
    fn test_simplify_constant_folding() {
        // Test: (i32.add (i32.const 10) (i32.const 32)) → (i32.const 42)
        let term = iadd32(iconst32(Imm32::from(10)), iconst32(Imm32::from(32)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 42);
            }
            _ => panic!("Expected constant folding to produce I32Const(42)"),
        }
    }

    #[test]
    fn test_simplify_nested_folding() {
        // Test: (i32.add (i32.const 5) (i32.add (i32.const 10) (i32.const 20)))
        // Should simplify to: (i32.add (i32.const 5) (i32.const 30))
        // Then to: (i32.const 35)
        let inner_add = iadd32(iconst32(Imm32::from(10)), iconst32(Imm32::from(20)));
        let outer_add = iadd32(iconst32(Imm32::from(5)), inner_add);
        let simplified = simplify(outer_add);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 35);
            }
            _ => panic!("Expected constant folding to produce I32Const(35)"),
        }
    }

    #[test]
    fn test_simplify_no_folding() {
        // If we had variables, this wouldn't fold
        // For now, test that constants are unchanged
        let term = iconst32(Imm32::from(42));
        let simplified = simplify(term.clone());

        assert_eq!(simplified, term);
    }

    #[test]
    fn test_simplify_overflow() {
        // Test overflow wrapping in constant folding
        let term = iadd32(iconst32(Imm32::from(i32::MAX)), iconst32(Imm32::from(1)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), i32::MIN);
            }
            _ => panic!("Expected constant folding with overflow"),
        }
    }

    // ========================================================================
    // Phase 7 Tests: Additional Operations and Algebraic Simplifications
    // ========================================================================

    #[test]
    fn test_i32_sub_constant_folding() {
        // Test: (i32.sub (i32.const 100) (i32.const 42)) → (i32.const 58)
        let term = isub32(iconst32(Imm32::from(100)), iconst32(Imm32::from(42)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 58);
            }
            _ => panic!("Expected I32Const(58)"),
        }
    }

    #[test]
    fn test_i32_mul_constant_folding() {
        // Test: (i32.mul (i32.const 6) (i32.const 7)) → (i32.const 42)
        let term = imul32(iconst32(Imm32::from(6)), iconst32(Imm32::from(7)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 42);
            }
            _ => panic!("Expected I32Const(42)"),
        }
    }

    #[test]
    fn test_algebraic_add_zero() {
        // Test: (i32.add (i32.const 42) (i32.const 0)) → (i32.const 42)
        let term = iadd32(iconst32(Imm32::from(42)), iconst32(Imm32::from(0)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 42);
            }
            _ => panic!("Expected algebraic simplification: x + 0 = x"),
        }
    }

    #[test]
    fn test_algebraic_sub_zero() {
        // Test: (i32.sub (i32.const 99) (i32.const 0)) → (i32.const 99)
        let term = isub32(iconst32(Imm32::from(99)), iconst32(Imm32::from(0)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 99);
            }
            _ => panic!("Expected algebraic simplification: x - 0 = x"),
        }
    }

    #[test]
    fn test_algebraic_mul_zero() {
        // Test: (i32.mul (i32.const 999) (i32.const 0)) → (i32.const 0)
        let term = imul32(iconst32(Imm32::from(999)), iconst32(Imm32::from(0)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 0);
            }
            _ => panic!("Expected algebraic simplification: x * 0 = 0"),
        }
    }

    #[test]
    fn test_algebraic_mul_one() {
        // Test: (i32.mul (i32.const 123) (i32.const 1)) → (i32.const 123)
        let term = imul32(iconst32(Imm32::from(123)), iconst32(Imm32::from(1)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 123);
            }
            _ => panic!("Expected algebraic simplification: x * 1 = x"),
        }
    }

    #[test]
    fn test_i64_add_constant_folding() {
        // Test: (i64.add (i64.const 1000) (i64.const 2000)) → (i64.const 3000)
        let term = iadd64(iconst64(Imm64::from(1000)), iconst64(Imm64::from(2000)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I64Const { val } => {
                assert_eq!(val.value(), 3000);
            }
            _ => panic!("Expected I64Const(3000)"),
        }
    }

    #[test]
    fn test_i64_sub_constant_folding() {
        // Test: (i64.sub (i64.const 500) (i64.const 300)) → (i64.const 200)
        let term = isub64(iconst64(Imm64::from(500)), iconst64(Imm64::from(300)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I64Const { val } => {
                assert_eq!(val.value(), 200);
            }
            _ => panic!("Expected I64Const(200)"),
        }
    }

    #[test]
    fn test_i64_mul_constant_folding() {
        // Test: (i64.mul (i64.const 10) (i64.const 20)) → (i64.const 200)
        let term = imul64(iconst64(Imm64::from(10)), iconst64(Imm64::from(20)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I64Const { val } => {
                assert_eq!(val.value(), 200);
            }
            _ => panic!("Expected I64Const(200)"),
        }
    }

    #[test]
    fn test_i32_sub_overflow() {
        // Test: (i32.sub (i32.const i32::MIN) (i32.const 1)) wraps
        let term = isub32(iconst32(Imm32::from(i32::MIN)), iconst32(Imm32::from(1)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), i32::MAX);
            }
            _ => panic!("Expected overflow wrapping"),
        }
    }

    #[test]
    fn test_i32_mul_overflow() {
        // Test: (i32.mul (i32.const i32::MAX) (i32.const 2)) wraps
        let term = imul32(iconst32(Imm32::from(i32::MAX)), iconst32(Imm32::from(2)));
        let simplified = simplify(term);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), -2); // i32::MAX * 2 wraps to -2
            }
            _ => panic!("Expected overflow wrapping"),
        }
    }

    #[test]
    fn test_nested_operations() {
        // Test: ((5 * 2) + 10) → (10 + 10) → 20
        let mul = imul32(iconst32(Imm32::from(5)), iconst32(Imm32::from(2)));
        let add = iadd32(mul, iconst32(Imm32::from(10)));
        let simplified = simplify(add);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 20);
            }
            _ => panic!("Expected nested constant folding"),
        }
    }
}
