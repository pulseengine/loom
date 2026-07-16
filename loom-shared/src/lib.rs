//! LOOM ISLE Definitions
//!
//! This crate contains ISLE (Instruction Selection/Lowering Expressions) term definitions
//! for WebAssembly optimization rules. The ISLE compiler generates Rust code from .isle files
//! during the build process.

#![allow(dead_code)]
#![allow(unused_variables)]

// cranelift-isle 0.132.1's generated code emits `alloc::` paths; bring the
// alloc crate into scope so the included ISLE output resolves them (#198).
extern crate alloc;

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

/// Primitive type for 32-bit float immediates
/// Stored as u32 bits to ensure Hash/Eq work correctly
#[derive(Clone, Copy, Debug)]
pub struct ImmF32(pub u32);

impl From<f32> for ImmF32 {
    fn from(val: f32) -> Self {
        ImmF32(val.to_bits())
    }
}

impl From<ImmF32> for f32 {
    fn from(imm: ImmF32) -> Self {
        f32::from_bits(imm.0)
    }
}

impl ImmF32 {
    /// Create a new ImmF32 from an f32
    pub fn new(val: f32) -> Self {
        ImmF32(val.to_bits())
    }

    /// Create a new ImmF32 from raw bits
    pub fn from_bits(bits: u32) -> Self {
        ImmF32(bits)
    }

    /// Get the float value
    pub fn value(&self) -> f32 {
        f32::from_bits(self.0)
    }

    /// Get the raw bits
    pub fn bits(&self) -> u32 {
        self.0
    }
}

impl PartialEq for ImmF32 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ImmF32 {}

impl std::hash::Hash for ImmF32 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

/// Primitive type for 64-bit float immediates
/// Stored as u64 bits to ensure Hash/Eq work correctly
#[derive(Clone, Copy, Debug)]
pub struct ImmF64(pub u64);

impl From<f64> for ImmF64 {
    fn from(val: f64) -> Self {
        ImmF64(val.to_bits())
    }
}

impl From<ImmF64> for f64 {
    fn from(imm: ImmF64) -> Self {
        f64::from_bits(imm.0)
    }
}

impl ImmF64 {
    /// Create a new ImmF64 from an f64
    pub fn new(val: f64) -> Self {
        ImmF64(val.to_bits())
    }

    /// Create a new ImmF64 from raw bits
    pub fn from_bits(bits: u64) -> Self {
        ImmF64(bits)
    }

    /// Get the float value
    pub fn value(&self) -> f64 {
        f64::from_bits(self.0)
    }

    /// Get the raw bits
    pub fn bits(&self) -> u64 {
        self.0
    }
}

impl PartialEq for ImmF64 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for ImmF64 {}

impl std::hash::Hash for ImmF64 {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.hash(state);
    }
}

// ====================================================================
// Float NaN/Subnormal Helpers
// ====================================================================
// These helpers centralize WebAssembly float constant-folding guards so
// that every arithmetic fold site uses exactly the same logic.

/// Canonical NaN for f32 (WebAssembly spec: 0x7fc00000)
pub const F32_CANONICAL_NAN: u32 = 0x7fc00000;

/// Canonical NaN for f64 (WebAssembly spec: 0x7ff8000000000000)
pub const F64_CANONICAL_NAN: u64 = 0x7ff8000000000000;

/// Returns `true` if the f32 is subnormal (non-zero with biased exponent 0).
/// We skip constant folding for subnormal results because the host FPU may
/// flush subnormals to zero (e.g. ARM FTZ), while WebAssembly requires
/// IEEE 754 gradual underflow.
pub fn is_f32_subnormal(val: f32) -> bool {
    val != 0.0 && val.is_subnormal()
}

/// Returns `true` if the f64 is subnormal (non-zero with biased exponent 0).
/// See [`is_f32_subnormal`] for rationale.
pub fn is_f64_subnormal(val: f64) -> bool {
    val != 0.0 && val.is_subnormal()
}

/// Canonicalize an f32 result for constant folding.
/// If the result is NaN, returns the WebAssembly canonical NaN bits.
/// Otherwise returns the result's bit representation.
pub fn canonicalize_f32(result: f32) -> u32 {
    if result.is_nan() {
        F32_CANONICAL_NAN
    } else {
        result.to_bits()
    }
}

/// Canonicalize an f64 result for constant folding.
/// If the result is NaN, returns the WebAssembly canonical NaN bits.
/// Otherwise returns the result's bit representation.
pub fn canonicalize_f64(result: f64) -> u64 {
    if result.is_nan() {
        F64_CANONICAL_NAN
    } else {
        result.to_bits()
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

    /// Rotation operations (i32)
    I32Rotl {
        lhs: Value,
        rhs: Value,
    },
    I32Rotr {
        lhs: Value,
        rhs: Value,
    },

    /// Rotation operations (i64)
    I64Rotl {
        lhs: Value,
        rhs: Value,
    },
    I64Rotr {
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

    /// Global variable operations
    GlobalGet {
        idx: u32,
    },
    GlobalSet {
        idx: u32,
        val: Value,
    },

    /// Memory operations (Phase 13 - Memory Optimization)
    I32Load {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    I32Store {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    I64Load {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    I64Store {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },

    // ========================================================================
    // Float Memory Operations
    // ========================================================================
    /// f32.load - Load 32-bit float from memory
    F32Load {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// f32.store - Store 32-bit float to memory
    F32Store {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// f64.load - Load 64-bit float from memory
    F64Load {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// f64.store - Store 64-bit float to memory
    F64Store {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },

    // ========================================================================
    // Partial-Width Memory Load Operations
    // ========================================================================
    /// i32.load8_s - Load 8 bits and sign-extend to i32
    I32Load8S {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i32.load8_u - Load 8 bits and zero-extend to i32
    I32Load8U {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i32.load16_s - Load 16 bits and sign-extend to i32
    I32Load16S {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i32.load16_u - Load 16 bits and zero-extend to i32
    I32Load16U {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.load8_s - Load 8 bits and sign-extend to i64
    I64Load8S {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.load8_u - Load 8 bits and zero-extend to i64
    I64Load8U {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.load16_s - Load 16 bits and sign-extend to i64
    I64Load16S {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.load16_u - Load 16 bits and zero-extend to i64
    I64Load16U {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.load32_s - Load 32 bits and sign-extend to i64
    I64Load32S {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.load32_u - Load 32 bits and zero-extend to i64
    I64Load32U {
        addr: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },

    // ========================================================================
    // Partial-Width Memory Store Operations
    // ========================================================================
    /// i32.store8 - Store low 8 bits of i32
    I32Store8 {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i32.store16 - Store low 16 bits of i32
    I32Store16 {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.store8 - Store low 8 bits of i64
    I64Store8 {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.store16 - Store low 16 bits of i64
    I64Store16 {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
    },
    /// i64.store32 - Store low 32 bits of i64
    I64Store32 {
        addr: Value,
        value: Value,
        offset: u32,
        align: u32,
        mem: u32,
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

    /// Drop - discards the top stack value
    Drop {
        val: Value,
    },

    /// Integer conversion operations
    /// i32.wrap_i64 - truncates i64 to i32 (keeps low 32 bits)
    I32WrapI64 {
        val: Value,
    },
    /// i64.extend_i32_s - sign-extends i32 to i64
    I64ExtendI32S {
        val: Value,
    },
    /// i64.extend_i32_u - zero-extends i32 to i64
    I64ExtendI32U {
        val: Value,
    },

    // ========================================================================
    // Float-to-Integer Truncation Operations (trapping)
    // ========================================================================
    I32TruncF32S {
        val: Value,
    },
    I32TruncF32U {
        val: Value,
    },
    I32TruncF64S {
        val: Value,
    },
    I32TruncF64U {
        val: Value,
    },
    I64TruncF32S {
        val: Value,
    },
    I64TruncF32U {
        val: Value,
    },
    I64TruncF64S {
        val: Value,
    },
    I64TruncF64U {
        val: Value,
    },

    // ========================================================================
    // Integer-to-Float Conversion Operations
    // ========================================================================
    F32ConvertI32S {
        val: Value,
    },
    F32ConvertI32U {
        val: Value,
    },
    F32ConvertI64S {
        val: Value,
    },
    F32ConvertI64U {
        val: Value,
    },
    F64ConvertI32S {
        val: Value,
    },
    F64ConvertI32U {
        val: Value,
    },
    F64ConvertI64S {
        val: Value,
    },
    F64ConvertI64U {
        val: Value,
    },

    // ========================================================================
    // Float Demote/Promote Operations
    // ========================================================================
    F32DemoteF64 {
        val: Value,
    },
    F64PromoteF32 {
        val: Value,
    },

    // ========================================================================
    // Reinterpret (bit-cast) Operations
    // ========================================================================
    I32ReinterpretF32 {
        val: Value,
    },
    I64ReinterpretF64 {
        val: Value,
    },
    F32ReinterpretI32 {
        val: Value,
    },
    F64ReinterpretI64 {
        val: Value,
    },

    // ========================================================================
    // Saturating Float-to-Integer Truncation Operations (non-trapping)
    // ========================================================================
    I32TruncSatF32S {
        val: Value,
    },
    I32TruncSatF32U {
        val: Value,
    },
    I32TruncSatF64S {
        val: Value,
    },
    I32TruncSatF64U {
        val: Value,
    },
    I64TruncSatF32S {
        val: Value,
    },
    I64TruncSatF32U {
        val: Value,
    },
    I64TruncSatF64S {
        val: Value,
    },
    I64TruncSatF64U {
        val: Value,
    },

    // ========================================================================
    // Memory Size/Grow Operations
    // ========================================================================
    /// memory.size - returns current memory size in pages
    MemorySize {
        mem: u32,
    },
    /// memory.grow - grows memory by delta pages, returns previous size or -1
    MemoryGrow {
        val: Value,
        mem: u32,
    },

    // ========================================================================
    // Bulk Memory Operations (side-effectful, no stack output)
    // ========================================================================
    /// memory.fill - fill memory region with a byte value
    MemoryFill {
        dst: Value,
        val: Value,
        len: Value,
        mem: u32,
    },
    /// memory.copy - copy memory region from src to dst
    MemoryCopy {
        dst: Value,
        src: Value,
        len: Value,
        dst_mem: u32,
        src_mem: u32,
    },
    /// memory.init - initialize memory from a data segment
    MemoryInit {
        dst: Value,
        src: Value,
        len: Value,
        mem: u32,
        data_idx: u32,
    },
    /// data.drop - drop a data segment (no stack operands)
    DataDrop {
        data_idx: u32,
    },

    // ========================================================================
    // Sign Extension Operations (in-place sign extension)
    // ========================================================================
    /// i32.extend8_s - sign-extend low 8 bits to 32 bits
    I32Extend8S {
        val: Value,
    },
    /// i32.extend16_s - sign-extend low 16 bits to 32 bits
    I32Extend16S {
        val: Value,
    },
    /// i64.extend8_s - sign-extend low 8 bits to 64 bits
    I64Extend8S {
        val: Value,
    },
    /// i64.extend16_s - sign-extend low 16 bits to 64 bits
    I64Extend16S {
        val: Value,
    },
    /// i64.extend32_s - sign-extend low 32 bits to 64 bits
    I64Extend32S {
        val: Value,
    },

    // ========================================================================
    // Floating-Point Operations
    // ========================================================================
    /// f32.const N
    F32Const {
        val: ImmF32,
    },
    /// f64.const N
    F64Const {
        val: ImmF64,
    },

    /// f32.add lhs rhs
    F32Add {
        lhs: Value,
        rhs: Value,
    },
    /// f32.sub lhs rhs
    F32Sub {
        lhs: Value,
        rhs: Value,
    },
    /// f32.mul lhs rhs
    F32Mul {
        lhs: Value,
        rhs: Value,
    },
    /// f32.div lhs rhs
    F32Div {
        lhs: Value,
        rhs: Value,
    },

    /// f64.add lhs rhs
    F64Add {
        lhs: Value,
        rhs: Value,
    },
    /// f64.sub lhs rhs
    F64Sub {
        lhs: Value,
        rhs: Value,
    },
    /// f64.mul lhs rhs
    F64Mul {
        lhs: Value,
        rhs: Value,
    },
    /// f64.div lhs rhs
    F64Div {
        lhs: Value,
        rhs: Value,
    },
    // f32 unary operations
    /// f32.abs val
    F32Abs {
        val: Value,
    },
    /// f32.neg val
    F32Neg {
        val: Value,
    },
    /// f32.ceil val
    F32Ceil {
        val: Value,
    },
    /// f32.floor val
    F32Floor {
        val: Value,
    },
    /// f32.trunc val
    F32Trunc {
        val: Value,
    },
    /// f32.nearest val
    F32Nearest {
        val: Value,
    },
    /// f32.sqrt val
    F32Sqrt {
        val: Value,
    },
    // f32 binary operations
    /// f32.min lhs rhs
    F32Min {
        lhs: Value,
        rhs: Value,
    },
    /// f32.max lhs rhs
    F32Max {
        lhs: Value,
        rhs: Value,
    },
    /// f32.copysign lhs rhs
    F32Copysign {
        lhs: Value,
        rhs: Value,
    },
    // f32 comparison operations (produce i32)
    /// f32.eq lhs rhs
    F32Eq {
        lhs: Value,
        rhs: Value,
    },
    /// f32.ne lhs rhs
    F32Ne {
        lhs: Value,
        rhs: Value,
    },
    /// f32.lt lhs rhs
    F32Lt {
        lhs: Value,
        rhs: Value,
    },
    /// f32.gt lhs rhs
    F32Gt {
        lhs: Value,
        rhs: Value,
    },
    /// f32.le lhs rhs
    F32Le {
        lhs: Value,
        rhs: Value,
    },
    /// f32.ge lhs rhs
    F32Ge {
        lhs: Value,
        rhs: Value,
    },
    // f64 unary operations
    /// f64.abs val
    F64Abs {
        val: Value,
    },
    /// f64.neg val
    F64Neg {
        val: Value,
    },
    /// f64.ceil val
    F64Ceil {
        val: Value,
    },
    /// f64.floor val
    F64Floor {
        val: Value,
    },
    /// f64.trunc val
    F64Trunc {
        val: Value,
    },
    /// f64.nearest val
    F64Nearest {
        val: Value,
    },
    /// f64.sqrt val
    F64Sqrt {
        val: Value,
    },
    // f64 binary operations
    /// f64.min lhs rhs
    F64Min {
        lhs: Value,
        rhs: Value,
    },
    /// f64.max lhs rhs
    F64Max {
        lhs: Value,
        rhs: Value,
    },
    /// f64.copysign lhs rhs
    F64Copysign {
        lhs: Value,
        rhs: Value,
    },
    // f64 comparison operations (produce i32)
    /// f64.eq lhs rhs
    F64Eq {
        lhs: Value,
        rhs: Value,
    },
    /// f64.ne lhs rhs
    F64Ne {
        lhs: Value,
        rhs: Value,
    },
    /// f64.lt lhs rhs
    F64Lt {
        lhs: Value,
        rhs: Value,
    },
    /// f64.gt lhs rhs
    F64Gt {
        lhs: Value,
        rhs: Value,
    },
    /// f64.le lhs rhs
    F64Le {
        lhs: Value,
        rhs: Value,
    },
    /// f64.ge lhs rhs
    F64Ge {
        lhs: Value,
        rhs: Value,
    },
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

/// Extract i32.const value (extractor for pattern matching)
pub fn iconst32_extract(val: &Value) -> Option<Imm32> {
    match val.0.as_ref() {
        ValueData::I32Const { val } => Some(*val),
        _ => None,
    }
}

/// Construct an i32.add operation
pub fn iadd32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Add { lhs, rhs }))
}

/// Extract i32.add operands (extractor for pattern matching)
pub fn iadd32_extract(val: &Value) -> Option<(Value, Value)> {
    match val.0.as_ref() {
        ValueData::I32Add { lhs, rhs } => Some((lhs.clone(), rhs.clone())),
        _ => None,
    }
}

/// Construct an i32.sub operation
pub fn isub32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Sub { lhs, rhs }))
}

/// Extract i32.sub operands (extractor for pattern matching)
pub fn isub32_extract(val: &Value) -> Option<(Value, Value)> {
    match val.0.as_ref() {
        ValueData::I32Sub { lhs, rhs } => Some((lhs.clone(), rhs.clone())),
        _ => None,
    }
}

/// Construct an i32.mul operation
pub fn imul32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Mul { lhs, rhs }))
}

/// Extract i32.mul operands (extractor for pattern matching)
pub fn imul32_extract(val: &Value) -> Option<(Value, Value)> {
    match val.0.as_ref() {
        ValueData::I32Mul { lhs, rhs } => Some((lhs.clone(), rhs.clone())),
        _ => None,
    }
}

/// Construct an i64.const value
pub fn iconst64(val: Imm64) -> Value {
    Value(Box::new(ValueData::I64Const { val }))
}

/// Extract i64.const value (extractor for pattern matching)
pub fn iconst64_extract(val: &Value) -> Option<Imm64> {
    match val.0.as_ref() {
        ValueData::I64Const { val } => Some(*val),
        _ => None,
    }
}

/// Construct an i64.add operation
pub fn iadd64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Add { lhs, rhs }))
}

/// Extract i64.add operands (extractor for pattern matching)
pub fn iadd64_extract(val: &Value) -> Option<(Value, Value)> {
    match val.0.as_ref() {
        ValueData::I64Add { lhs, rhs } => Some((lhs.clone(), rhs.clone())),
        _ => None,
    }
}

/// Construct an i64.sub operation
pub fn isub64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Sub { lhs, rhs }))
}

/// Extract i64.sub operands (extractor for pattern matching)
pub fn isub64_extract(val: &Value) -> Option<(Value, Value)> {
    match val.0.as_ref() {
        ValueData::I64Sub { lhs, rhs } => Some((lhs.clone(), rhs.clone())),
        _ => None,
    }
}

/// Construct an i64.mul operation
pub fn imul64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Mul { lhs, rhs }))
}

/// Extract i64.mul operands (extractor for pattern matching)
pub fn imul64_extract(val: &Value) -> Option<(Value, Value)> {
    match val.0.as_ref() {
        ValueData::I64Mul { lhs, rhs } => Some((lhs.clone(), rhs.clone())),
        _ => None,
    }
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

/// Construct an i32.rotl operation (rotate left)
pub fn irotl32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Rotl { lhs, rhs }))
}

/// Construct an i32.rotr operation (rotate right)
pub fn irotr32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I32Rotr { lhs, rhs }))
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

/// Construct an i64.rotl operation (rotate left)
pub fn irotl64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Rotl { lhs, rhs }))
}

/// Construct an i64.rotr operation (rotate right)
pub fn irotr64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::I64Rotr { lhs, rhs }))
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

/// Construct a global.get operation
pub fn global_get(idx: u32) -> Value {
    Value(Box::new(ValueData::GlobalGet { idx }))
}

/// Construct a global.set operation
pub fn global_set(idx: u32, val: Value) -> Value {
    Value(Box::new(ValueData::GlobalSet { idx, val }))
}

/// Construct an i32.load operation
pub fn i32_load(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Load {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i32.store operation
pub fn i32_store(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Store {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.load operation
pub fn i64_load(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Load {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.store operation
pub fn i64_store(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Store {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

/// Construct an f32.load operation
pub fn f32_load(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::F32Load {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an f32.store operation
pub fn f32_store(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::F32Store {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

/// Construct an f64.load operation
pub fn f64_load(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::F64Load {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an f64.store operation
pub fn f64_store(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::F64Store {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

// ============================================================================
// Partial-Width Memory Load Constructors
// ============================================================================

/// Construct an i32.load8_s operation (load 8 bits, sign-extend to i32)
pub fn i32_load8_s(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Load8S {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i32.load8_u operation (load 8 bits, zero-extend to i32)
pub fn i32_load8_u(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Load8U {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i32.load16_s operation (load 16 bits, sign-extend to i32)
pub fn i32_load16_s(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Load16S {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i32.load16_u operation (load 16 bits, zero-extend to i32)
pub fn i32_load16_u(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Load16U {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.load8_s operation (load 8 bits, sign-extend to i64)
pub fn i64_load8_s(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Load8S {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.load8_u operation (load 8 bits, zero-extend to i64)
pub fn i64_load8_u(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Load8U {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.load16_s operation (load 16 bits, sign-extend to i64)
pub fn i64_load16_s(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Load16S {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.load16_u operation (load 16 bits, zero-extend to i64)
pub fn i64_load16_u(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Load16U {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.load32_s operation (load 32 bits, sign-extend to i64)
pub fn i64_load32_s(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Load32S {
        addr,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.load32_u operation (load 32 bits, zero-extend to i64)
pub fn i64_load32_u(addr: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Load32U {
        addr,
        offset,
        align,
        mem,
    }))
}

// ============================================================================
// Partial-Width Memory Store Constructors
// ============================================================================

/// Construct an i32.store8 operation (store low 8 bits of i32)
pub fn i32_store8(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Store8 {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

/// Construct an i32.store16 operation (store low 16 bits of i32)
pub fn i32_store16(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I32Store16 {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.store8 operation (store low 8 bits of i64)
pub fn i64_store8(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Store8 {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.store16 operation (store low 16 bits of i64)
pub fn i64_store16(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Store16 {
        addr,
        value,
        offset,
        align,
        mem,
    }))
}

/// Construct an i64.store32 operation (store low 32 bits of i64)
pub fn i64_store32(addr: Value, value: Value, offset: u32, align: u32, mem: u32) -> Value {
    Value(Box::new(ValueData::I64Store32 {
        addr,
        value,
        offset,
        align,
        mem,
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
pub fn if_instr(
    cond: Value,
    block_type: BlockType,
    then_body: InstructionList,
    else_body: InstructionList,
) -> Value {
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
    Value(Box::new(ValueData::Br { depth, value: None }))
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
    Value(Box::new(ValueData::Return { values: Vec::new() }))
}

/// Drop constructor for ISLE - discards a value from stack
pub fn drop_instr(val: Value) -> Value {
    Value(Box::new(ValueData::Drop { val }))
}

/// i32.wrap_i64 constructor - truncates i64 to i32 (keeps low 32 bits)
pub fn i32_wrap_i64(val: Value) -> Value {
    Value(Box::new(ValueData::I32WrapI64 { val }))
}

/// i64.extend_i32_s constructor - sign-extends i32 to i64
pub fn i64_extend_i32_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64ExtendI32S { val }))
}

/// i64.extend_i32_u constructor - zero-extends i32 to i64
pub fn i64_extend_i32_u(val: Value) -> Value {
    Value(Box::new(ValueData::I64ExtendI32U { val }))
}

/// Extractor for i32.wrap_i64 — used by ISLE rules to pattern-match the term
/// on a rule LHS (#219 seam-SROA). Returns the inner operand.
pub fn i32_wrap_i64_extract(val: &Value) -> Option<Value> {
    match val.0.as_ref() {
        ValueData::I32WrapI64 { val } => Some(val.clone()),
        _ => None,
    }
}

/// Extractor for i64.extend_i32_u — ISLE rule LHS matching (#219 seam-SROA).
pub fn i64_extend_i32_u_extract(val: &Value) -> Option<Value> {
    match val.0.as_ref() {
        ValueData::I64ExtendI32U { val } => Some(val.clone()),
        _ => None,
    }
}

// ============================================================================
// Float-to-Integer Truncation Constructors (trapping)
// ============================================================================
pub fn i32_trunc_f32_s(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncF32S { val }))
}
pub fn i32_trunc_f32_u(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncF32U { val }))
}
pub fn i32_trunc_f64_s(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncF64S { val }))
}
pub fn i32_trunc_f64_u(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncF64U { val }))
}
pub fn i64_trunc_f32_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncF32S { val }))
}
pub fn i64_trunc_f32_u(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncF32U { val }))
}
pub fn i64_trunc_f64_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncF64S { val }))
}
pub fn i64_trunc_f64_u(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncF64U { val }))
}

// ============================================================================
// Integer-to-Float Conversion Constructors
// ============================================================================
pub fn f32_convert_i32_s(val: Value) -> Value {
    Value(Box::new(ValueData::F32ConvertI32S { val }))
}
pub fn f32_convert_i32_u(val: Value) -> Value {
    Value(Box::new(ValueData::F32ConvertI32U { val }))
}
pub fn f32_convert_i64_s(val: Value) -> Value {
    Value(Box::new(ValueData::F32ConvertI64S { val }))
}
pub fn f32_convert_i64_u(val: Value) -> Value {
    Value(Box::new(ValueData::F32ConvertI64U { val }))
}
pub fn f64_convert_i32_s(val: Value) -> Value {
    Value(Box::new(ValueData::F64ConvertI32S { val }))
}
pub fn f64_convert_i32_u(val: Value) -> Value {
    Value(Box::new(ValueData::F64ConvertI32U { val }))
}
pub fn f64_convert_i64_s(val: Value) -> Value {
    Value(Box::new(ValueData::F64ConvertI64S { val }))
}
pub fn f64_convert_i64_u(val: Value) -> Value {
    Value(Box::new(ValueData::F64ConvertI64U { val }))
}

// ============================================================================
// Float Demote/Promote Constructors
// ============================================================================
pub fn f32_demote_f64(val: Value) -> Value {
    Value(Box::new(ValueData::F32DemoteF64 { val }))
}
pub fn f64_promote_f32(val: Value) -> Value {
    Value(Box::new(ValueData::F64PromoteF32 { val }))
}

// ============================================================================
// Reinterpret (bit-cast) Constructors
// ============================================================================
pub fn i32_reinterpret_f32(val: Value) -> Value {
    Value(Box::new(ValueData::I32ReinterpretF32 { val }))
}
pub fn i64_reinterpret_f64(val: Value) -> Value {
    Value(Box::new(ValueData::I64ReinterpretF64 { val }))
}
pub fn f32_reinterpret_i32(val: Value) -> Value {
    Value(Box::new(ValueData::F32ReinterpretI32 { val }))
}
pub fn f64_reinterpret_i64(val: Value) -> Value {
    Value(Box::new(ValueData::F64ReinterpretI64 { val }))
}

// ============================================================================
// Saturating Float-to-Integer Truncation Constructors (non-trapping)
// ============================================================================
pub fn i32_trunc_sat_f32_s(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncSatF32S { val }))
}
pub fn i32_trunc_sat_f32_u(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncSatF32U { val }))
}
pub fn i32_trunc_sat_f64_s(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncSatF64S { val }))
}
pub fn i32_trunc_sat_f64_u(val: Value) -> Value {
    Value(Box::new(ValueData::I32TruncSatF64U { val }))
}
pub fn i64_trunc_sat_f32_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncSatF32S { val }))
}
pub fn i64_trunc_sat_f32_u(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncSatF32U { val }))
}
pub fn i64_trunc_sat_f64_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncSatF64S { val }))
}
pub fn i64_trunc_sat_f64_u(val: Value) -> Value {
    Value(Box::new(ValueData::I64TruncSatF64U { val }))
}

// ============================================================================
// Memory Size/Grow Constructors
// ============================================================================
pub fn memory_size(mem: u32) -> Value {
    Value(Box::new(ValueData::MemorySize { mem }))
}
pub fn memory_grow(val: Value, mem: u32) -> Value {
    Value(Box::new(ValueData::MemoryGrow { val, mem }))
}

// ============================================================================
// Bulk Memory Constructors
// ============================================================================
pub fn memory_fill(dst: Value, val: Value, len: Value, mem: u32) -> Value {
    Value(Box::new(ValueData::MemoryFill { dst, val, len, mem }))
}
pub fn memory_copy(dst: Value, src: Value, len: Value, dst_mem: u32, src_mem: u32) -> Value {
    Value(Box::new(ValueData::MemoryCopy {
        dst,
        src,
        len,
        dst_mem,
        src_mem,
    }))
}
pub fn memory_init(dst: Value, src: Value, len: Value, mem: u32, data_idx: u32) -> Value {
    Value(Box::new(ValueData::MemoryInit {
        dst,
        src,
        len,
        mem,
        data_idx,
    }))
}
pub fn data_drop(data_idx: u32) -> Value {
    Value(Box::new(ValueData::DataDrop { data_idx }))
}

// ============================================================================
// Sign Extension Constructors
// ============================================================================

/// i32.extend8_s constructor - sign-extend low 8 bits to 32 bits
pub fn i32_extend8_s(val: Value) -> Value {
    Value(Box::new(ValueData::I32Extend8S { val }))
}

/// i32.extend16_s constructor - sign-extend low 16 bits to 32 bits
pub fn i32_extend16_s(val: Value) -> Value {
    Value(Box::new(ValueData::I32Extend16S { val }))
}

/// i64.extend8_s constructor - sign-extend low 8 bits to 64 bits
pub fn i64_extend8_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64Extend8S { val }))
}

/// i64.extend16_s constructor - sign-extend low 16 bits to 64 bits
pub fn i64_extend16_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64Extend16S { val }))
}

/// i64.extend32_s constructor - sign-extend low 32 bits to 64 bits
pub fn i64_extend32_s(val: Value) -> Value {
    Value(Box::new(ValueData::I64Extend32S { val }))
}

// ============================================================================
// Floating-Point Constructors
// ============================================================================

/// Construct an f32.const value
pub fn fconst32(val: ImmF32) -> Value {
    Value(Box::new(ValueData::F32Const { val }))
}

/// Construct an f64.const value
pub fn fconst64(val: ImmF64) -> Value {
    Value(Box::new(ValueData::F64Const { val }))
}

/// Construct an f32.add operation
pub fn fadd32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Add { lhs, rhs }))
}

/// Construct an f32.sub operation
pub fn fsub32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Sub { lhs, rhs }))
}

/// Construct an f32.mul operation
pub fn fmul32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Mul { lhs, rhs }))
}

/// Construct an f32.div operation
pub fn fdiv32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Div { lhs, rhs }))
}

/// Construct an f64.add operation
pub fn fadd64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Add { lhs, rhs }))
}

/// Construct an f64.sub operation
pub fn fsub64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Sub { lhs, rhs }))
}

/// Construct an f64.mul operation
pub fn fmul64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Mul { lhs, rhs }))
}

/// Construct an f64.div operation
pub fn fdiv64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Div { lhs, rhs }))
}

// f32 unary operation constructors
/// Construct an f32.abs operation
pub fn fabs32(val: Value) -> Value {
    Value(Box::new(ValueData::F32Abs { val }))
}
/// Construct an f32.neg operation
pub fn fneg32(val: Value) -> Value {
    Value(Box::new(ValueData::F32Neg { val }))
}
/// Construct an f32.ceil operation
pub fn fceil32(val: Value) -> Value {
    Value(Box::new(ValueData::F32Ceil { val }))
}
/// Construct an f32.floor operation
pub fn ffloor32(val: Value) -> Value {
    Value(Box::new(ValueData::F32Floor { val }))
}
/// Construct an f32.trunc operation
pub fn ftrunc32(val: Value) -> Value {
    Value(Box::new(ValueData::F32Trunc { val }))
}
/// Construct an f32.nearest operation
pub fn fnearest32(val: Value) -> Value {
    Value(Box::new(ValueData::F32Nearest { val }))
}
/// Construct an f32.sqrt operation
pub fn fsqrt32(val: Value) -> Value {
    Value(Box::new(ValueData::F32Sqrt { val }))
}

// f32 binary operation constructors
/// Construct an f32.min operation
pub fn fmin32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Min { lhs, rhs }))
}
/// Construct an f32.max operation
pub fn fmax32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Max { lhs, rhs }))
}
/// Construct an f32.copysign operation
pub fn fcopysign32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Copysign { lhs, rhs }))
}

// f32 comparison operation constructors
/// Construct an f32.eq operation
pub fn feq32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Eq { lhs, rhs }))
}
/// Construct an f32.ne operation
pub fn fne32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Ne { lhs, rhs }))
}
/// Construct an f32.lt operation
pub fn flt32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Lt { lhs, rhs }))
}
/// Construct an f32.gt operation
pub fn fgt32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Gt { lhs, rhs }))
}
/// Construct an f32.le operation
pub fn fle32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Le { lhs, rhs }))
}
/// Construct an f32.ge operation
pub fn fge32(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F32Ge { lhs, rhs }))
}

// f64 unary operation constructors
/// Construct an f64.abs operation
pub fn fabs64(val: Value) -> Value {
    Value(Box::new(ValueData::F64Abs { val }))
}
/// Construct an f64.neg operation
pub fn fneg64(val: Value) -> Value {
    Value(Box::new(ValueData::F64Neg { val }))
}
/// Construct an f64.ceil operation
pub fn fceil64(val: Value) -> Value {
    Value(Box::new(ValueData::F64Ceil { val }))
}
/// Construct an f64.floor operation
pub fn ffloor64(val: Value) -> Value {
    Value(Box::new(ValueData::F64Floor { val }))
}
/// Construct an f64.trunc operation
pub fn ftrunc64(val: Value) -> Value {
    Value(Box::new(ValueData::F64Trunc { val }))
}
/// Construct an f64.nearest operation
pub fn fnearest64(val: Value) -> Value {
    Value(Box::new(ValueData::F64Nearest { val }))
}
/// Construct an f64.sqrt operation
pub fn fsqrt64(val: Value) -> Value {
    Value(Box::new(ValueData::F64Sqrt { val }))
}

// f64 binary operation constructors
/// Construct an f64.min operation
pub fn fmin64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Min { lhs, rhs }))
}
/// Construct an f64.max operation
pub fn fmax64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Max { lhs, rhs }))
}
/// Construct an f64.copysign operation
pub fn fcopysign64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Copysign { lhs, rhs }))
}

// f64 comparison operation constructors
/// Construct an f64.eq operation
pub fn feq64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Eq { lhs, rhs }))
}
/// Construct an f64.ne operation
pub fn fne64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Ne { lhs, rhs }))
}
/// Construct an f64.lt operation
pub fn flt64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Lt { lhs, rhs }))
}
/// Construct an f64.gt operation
pub fn fgt64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Gt { lhs, rhs }))
}
/// Construct an f64.le operation
pub fn fle64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Le { lhs, rhs }))
}
/// Construct an f64.ge operation
pub fn fge64(lhs: Value, rhs: Value) -> Value {
    Value(Box::new(ValueData::F64Ge { lhs, rhs }))
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

// ============================================================================
// Helper Functions for Strength Reduction Optimizations
// ============================================================================

/// Check if a 32-bit immediate is a power of 2 (used as ISLE extractor)
/// Returns Some(value) if power of 2, None otherwise
pub fn is_power_of_two_i32(val: Imm32) -> Option<Imm32> {
    let n = val.0;
    if n > 0 && (n & (n - 1)) == 0 {
        Some(val)
    } else {
        None
    }
}

/// Check if a 64-bit immediate is a power of 2 (used as ISLE extractor)
/// Returns Some(value) if power of 2, None otherwise
pub fn is_power_of_two_i64(val: Imm64) -> Option<Imm64> {
    let n = val.0;
    if n > 0 && (n & (n - 1)) == 0 {
        Some(val)
    } else {
        None
    }
}

/// Get log2 of a power-of-2 immediate (32-bit)
/// Assumes the input is a power of 2
pub fn log2_i32(val: Imm32) -> Imm32 {
    let mut n = val.0;
    let mut log = 0;
    while n > 1 {
        n >>= 1;
        log += 1;
    }
    Imm32(log)
}

/// Get log2 of a power-of-2 immediate (64-bit)
/// Assumes the input is a power of 2
pub fn log2_i64(val: Imm64) -> Imm64 {
    let mut n = val.0;
    let mut log = 0;
    while n > 1 {
        n >>= 1;
        log += 1;
    }
    Imm64(log)
}

/// Subtract 1 from an immediate (for computing masks in x % power_of_2 optimization)
pub fn imm32_sub_1(val: Imm32) -> Imm32 {
    Imm32(val.0.wrapping_sub(1))
}

/// Subtract 1 from an immediate (for computing masks in x % power_of_2 optimization)
pub fn imm64_sub_1(val: Imm64) -> Imm64 {
    Imm64(val.0.wrapping_sub(1))
}

/// Memory location identifier: (base_address, offset)
/// We track memory as (const_addr + offset) pairs
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub struct MemoryLocation {
    /// Base address (if known constant)
    base: Option<i32>,
    /// Static offset
    offset: u32,
    /// Memory index (a load from memory 0 at offset X != memory 1 at offset X)
    mem: u32,
}

/// #231 proof-carrying facts (P1): a set of premises attached to a single
/// loom [`Value`] (term). Facts are carried on the **Value**, never on an
/// operator index — a `Value` is stable across renumbering/DCE, an index is
/// not. This is the load-bearing subtlety of the fact model: the same `Value`
/// keeps its facts through any value-preserving rewrite, and a pass that
/// cannot re-establish a fact simply drops it (a dropped fact forgoes an
/// optimization; it never miscompiles).
///
/// P1 carries exactly one fact kind — a signed value-range `[lo, hi]`
/// (inclusive) — which is a strict superset of the existing unsigned
/// [`OptimizationEnv::value_max`] premise. `value_max`/`fits_below_bit` remain
/// available as an unsigned VIEW over the range so #240's range-gated rewrites
/// are untouched.
#[derive(Clone, Debug, PartialEq, Eq, Default)]
pub struct FactSet {
    /// Signed value-range premise: the value ∈ `[lo, hi]` (inclusive both
    /// ends), interpreted as a signed i32/i64. `None` until established.
    pub value_range: Option<(i64, i64)>,
}

impl FactSet {
    /// A `FactSet` carrying a single signed value-range fact.
    pub fn range(lo: i64, hi: i64) -> Self {
        FactSet {
            value_range: Some((lo, hi)),
        }
    }

    /// The unsigned inclusive upper bound this range implies, if any. Only a
    /// range whose whole span is non-negative (`lo >= 0`) yields a sound
    /// unsigned `value_max` view: a negative `lo` means the value could be a
    /// large unsigned quantity (two's-complement), so no unsigned upper bound
    /// is derivable and this returns `None` (conservative — the #240 gate then
    /// declines to fire, never misfires).
    pub fn unsigned_max(&self) -> Option<u64> {
        match self.value_range {
            Some((lo, hi)) if lo >= 0 && hi >= 0 => Some(hi as u64),
            _ => None,
        }
    }
}

/// Environment for dataflow analysis
#[derive(Clone)]
pub struct OptimizationEnv {
    /// Local variable constants
    pub locals: std::collections::HashMap<u32, Value>,
    /// Memory state: location → stored value
    pub memory: std::collections::HashMap<MemoryLocation, Value>,
    /// #219 carrier scalar-forwarding: indices of SINGLE-ASSIGNMENT locals
    /// (exactly one write in the whole function, precomputed by the caller) whose
    /// pure defining expression may be forwarded to its uses. Empty by default →
    /// forwarding OFF, behavior identical to plain dataflow.
    pub single_assign: std::collections::HashSet<u32>,
    /// #219: captured defining expression for each `single_assign` local, recorded
    /// at its (single) `local.set`/`local.tee`. SURVIVES control-flow `clear()`
    /// (sound: a single-assignment local has one value on every path that reaches
    /// a use). INVALIDATED when any input local of the expression is reassigned
    /// (reaching-defs guard), so a forwarded expression always carries its
    /// def-time inputs. `local.get` of such a local forwards to this expression.
    pub pinned: std::collections::HashMap<u32, Value>,
    /// #240 Tier-2 range-premise hook: an INCLUSIVE unsigned upper bound on the
    /// value of an expression (`(value as unsigned) <= bound`).
    ///
    /// This is the minimal IR premise representation the range-gated algebraic
    /// rules consult. A premise is a *fact* — `assume value <= bound` — that must
    /// be discharged by a machine-checked source (a Verus/Rocq value-range proof
    /// on the dissolved function). Loom does NOT invent these bounds: the map is
    /// empty unless a fact-source populates it, so with no premises the gated
    /// rules never fire and behavior is byte-identical to plain dataflow.
    ///
    /// DEFERRED (#240 part 2 / #231): wiring the real fact-source that fills this
    /// map from synth/gale-emitted proof-carrying annotations. Tests inject
    /// bounds directly via [`Self::assume_max`].
    pub value_max: std::collections::HashMap<Value, u64>,
    /// #231 proof-carrying facts (P1): the value-attached [`FactSet`] carrier,
    /// keyed by the loom [`Value`] the fact constrains. This is the superset of
    /// `value_max` — the unsigned bound in `value_max` is kept as a VIEW
    /// maintained by [`Self::assume_range`], so #240's range-gated rewrites are
    /// untouched. Empty unless a fact-source (deferred: meld cross-component
    /// metadata) or a test injects one via [`Self::assume_range`].
    pub facts: std::collections::HashMap<Value, FactSet>,
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
            single_assign: std::collections::HashSet::new(),
            pinned: std::collections::HashMap::new(),
            value_max: std::collections::HashMap::new(),
            facts: std::collections::HashMap::new(),
        }
    }

    /// Invalidate all memory state (conservative, on unknown stores/calls)
    pub fn invalidate_memory(&mut self) {
        self.memory.clear();
    }

    /// #240 Tier-2: record a machine-checked value-range premise `expr <= bound`
    /// (bound is an inclusive unsigned upper bound). Intended to be called by the
    /// fact-source wiring (deferred); tests call it directly to inject a premise.
    pub fn assume_max(&mut self, expr: Value, bound: u64) {
        // Keep the tightest bound if one already exists.
        self.value_max
            .entry(expr.clone())
            .and_modify(|b| *b = (*b).min(bound))
            .or_insert(bound);
        // Reflect into the FactSet superset so both views agree. `assume_max`
        // asserts `0 <= expr <= bound` (unsigned), i.e. the signed range
        // `[0, bound]` whenever `bound` fits in an i64 (always true for any
        // bound < 2^63; for larger bounds the exact signed range is not
        // representable and we leave the range untouched — the unsigned
        // `value_max` view still serves #240).
        if bound <= i64::MAX as u64 {
            self.assume_range(expr, 0, bound as i64);
        }
    }

    /// #231 proof-carrying facts (P1): record a machine-checked SIGNED
    /// value-range premise `lo <= expr <= hi` (inclusive) on `expr`. This is
    /// the superset premise; it also maintains the unsigned [`Self::value_max`]
    /// VIEW so #240's range-gated rewrites keep working unchanged.
    ///
    /// The value-attached invariant: the fact is keyed by the [`Value`] itself.
    /// A value-preserving rewrite that reproduces the same `Value` inherits the
    /// fact for free; a pass that cannot re-establish it simply never copies it
    /// to the new value (drop-safe — see [`crate`] module docs on facts).
    ///
    /// Intended to be called by the fact-source wiring (deferred: meld
    /// cross-component metadata). Tests call it directly to inject a premise.
    pub fn assume_range(&mut self, expr: Value, lo: i64, hi: i64) {
        if lo > hi {
            // An empty/contradictory range names nothing usable; ignore it
            // rather than record an unsatisfiable premise.
            return;
        }
        // Intersect with any existing range (tightest wins — both premises
        // hold simultaneously).
        let entry = self.facts.entry(expr.clone()).or_default();
        entry.value_range = Some(match entry.value_range {
            Some((elo, ehi)) => (elo.max(lo), ehi.min(hi)),
            None => (lo, hi),
        });
        // Maintain the unsigned `value_max` VIEW.
        if let Some(m) = entry.unsigned_max() {
            self.value_max
                .entry(expr)
                .and_modify(|b| *b = (*b).min(m))
                .or_insert(m);
        }
    }

    /// #240 Tier-2: does a recorded premise prove every set bit of `expr` lies
    /// strictly below bit `k` — i.e. `(expr as unsigned) < 2^k`? This is the
    /// side condition that makes `(expr << (W-?)) >>u ...` masks a no-op and,
    /// specifically, licenses `(expr << k) >>u k -> expr` for a 32-bit value:
    /// there the surviving mask is `2^(32-k)-1`, a no-op exactly when
    /// `expr < 2^(32-k)`. Returns false (conservative) when no premise is known.
    pub fn fits_below_bit(&self, expr: &Value, k: u32) -> bool {
        if k >= 64 {
            return true;
        }
        match self.value_max.get(expr) {
            Some(&bound) => bound < (1u64 << k),
            None => false,
        }
    }
}

/// Legacy type alias for compatibility
pub type LocalEnv = OptimizationEnv;

/// #219 carrier scalar-forwarding: is `v` a side-effect-free, re-evaluatable
/// value expression safe to FORWARD to a single-assignment local's use sites?
/// Whitelist of total pure ops (constants, locals, globals, integer
/// arithmetic/bitwise/shift/rotate/compare, width converts, select); recurses
/// into operands. Calls, ANY load/store, div/rem (trap), floats, blocks/branches
/// → `false`. Bounded to the listed variants (NOT a full 187-variant walk):
/// unknown/impure → `false` (conservative, sound).
#[allow(clippy::match_like_matches_macro)]
fn is_forwardable_expr(v: &Value) -> bool {
    match v.data() {
        ValueData::I32Const { .. }
        | ValueData::I64Const { .. }
        | ValueData::LocalGet { .. }
        | ValueData::GlobalGet { .. } => true,
        ValueData::I32WrapI64 { val }
        | ValueData::I64ExtendI32S { val }
        | ValueData::I64ExtendI32U { val }
        | ValueData::I32Eqz { val }
        | ValueData::I64Eqz { val } => is_forwardable_expr(val),
        ValueData::I32Add { lhs, rhs }
        | ValueData::I32Sub { lhs, rhs }
        | ValueData::I32Mul { lhs, rhs }
        | ValueData::I32And { lhs, rhs }
        | ValueData::I32Or { lhs, rhs }
        | ValueData::I32Xor { lhs, rhs }
        | ValueData::I32Shl { lhs, rhs }
        | ValueData::I32ShrS { lhs, rhs }
        | ValueData::I32ShrU { lhs, rhs }
        | ValueData::I32Rotl { lhs, rhs }
        | ValueData::I32Rotr { lhs, rhs }
        | ValueData::I32Eq { lhs, rhs }
        | ValueData::I32Ne { lhs, rhs }
        | ValueData::I32LtS { lhs, rhs }
        | ValueData::I32LtU { lhs, rhs }
        | ValueData::I32GtS { lhs, rhs }
        | ValueData::I32GtU { lhs, rhs }
        | ValueData::I32LeS { lhs, rhs }
        | ValueData::I32LeU { lhs, rhs }
        | ValueData::I32GeS { lhs, rhs }
        | ValueData::I32GeU { lhs, rhs }
        | ValueData::I64Add { lhs, rhs }
        | ValueData::I64Sub { lhs, rhs }
        | ValueData::I64Mul { lhs, rhs }
        | ValueData::I64And { lhs, rhs }
        | ValueData::I64Or { lhs, rhs }
        | ValueData::I64Xor { lhs, rhs }
        | ValueData::I64Shl { lhs, rhs }
        | ValueData::I64ShrS { lhs, rhs }
        | ValueData::I64ShrU { lhs, rhs }
        | ValueData::I64Rotl { lhs, rhs }
        | ValueData::I64Rotr { lhs, rhs }
        | ValueData::I64Eq { lhs, rhs }
        | ValueData::I64Ne { lhs, rhs }
        | ValueData::I64LtS { lhs, rhs }
        | ValueData::I64LtU { lhs, rhs }
        | ValueData::I64GtS { lhs, rhs }
        | ValueData::I64GtU { lhs, rhs }
        | ValueData::I64LeS { lhs, rhs }
        | ValueData::I64LeU { lhs, rhs }
        | ValueData::I64GeS { lhs, rhs }
        | ValueData::I64GeU { lhs, rhs } => is_forwardable_expr(lhs) && is_forwardable_expr(rhs),
        ValueData::Select {
            cond,
            true_val,
            false_val,
        } => {
            is_forwardable_expr(cond)
                && is_forwardable_expr(true_val)
                && is_forwardable_expr(false_val)
        }
        _ => false,
    }
}

/// #278: is `v` provably TRAP-FREE, so that DISCARDING it (dropping the operand
/// entirely) preserves observable semantics? A trapping operand — an
/// out-of-bounds memory load/store, an integer div/rem (traps on 0 / INT_MIN÷-1),
/// an `unreachable`, a call (callees can trap), a trapping float→int truncate,
/// etc. — must NOT be discarded, or the optimized module would return a value
/// where the original TRAPS.
///
/// Zero-annihilation / absorption identities (`x*0→0`, `0*x→0`, `x&0→0`,
/// `x|-1→-1`, a `select`'s untaken arm, …) throw away a whole operand subtree.
/// LOOM's value-equivalence / Z3 checking does not model traps (#273/#274/#276),
/// so these folds are only sound when the discarded operand cannot trap. This is
/// exactly the `is_forwardable_expr` whitelist: constants, locals, globals, and
/// total pure integer arithmetic / bitwise / shift / rotate / compare /
/// width-convert / select over trap-free operands. Loads/stores, div/rem, calls,
/// unreachable, floats, and every unknown/impure variant are NOT on the whitelist
/// and are conservatively treated as MAY-trap (→ do not discard).
fn is_no_trap_expr(v: &Value) -> bool {
    is_forwardable_expr(v)
}

/// #219: does `v` (a forwardable expr — same whitelist as `is_forwardable_expr`)
/// reference `local.get local_idx`? Used to INVALIDATE a pinned forwarding when
/// one of its input locals is reassigned (reaching-defs guard), so a forwarded
/// expression never picks up a later redefinition of an input.
fn expr_references_local(v: &Value, local_idx: u32) -> bool {
    match v.data() {
        ValueData::LocalGet { idx } => *idx == local_idx,
        ValueData::I32Const { .. } | ValueData::I64Const { .. } | ValueData::GlobalGet { .. } => {
            false
        }
        ValueData::I32WrapI64 { val }
        | ValueData::I64ExtendI32S { val }
        | ValueData::I64ExtendI32U { val }
        | ValueData::I32Eqz { val }
        | ValueData::I64Eqz { val } => expr_references_local(val, local_idx),
        ValueData::I32Add { lhs, rhs }
        | ValueData::I32Sub { lhs, rhs }
        | ValueData::I32Mul { lhs, rhs }
        | ValueData::I32And { lhs, rhs }
        | ValueData::I32Or { lhs, rhs }
        | ValueData::I32Xor { lhs, rhs }
        | ValueData::I32Shl { lhs, rhs }
        | ValueData::I32ShrS { lhs, rhs }
        | ValueData::I32ShrU { lhs, rhs }
        | ValueData::I32Rotl { lhs, rhs }
        | ValueData::I32Rotr { lhs, rhs }
        | ValueData::I32Eq { lhs, rhs }
        | ValueData::I32Ne { lhs, rhs }
        | ValueData::I32LtS { lhs, rhs }
        | ValueData::I32LtU { lhs, rhs }
        | ValueData::I32GtS { lhs, rhs }
        | ValueData::I32GtU { lhs, rhs }
        | ValueData::I32LeS { lhs, rhs }
        | ValueData::I32LeU { lhs, rhs }
        | ValueData::I32GeS { lhs, rhs }
        | ValueData::I32GeU { lhs, rhs }
        | ValueData::I64Add { lhs, rhs }
        | ValueData::I64Sub { lhs, rhs }
        | ValueData::I64Mul { lhs, rhs }
        | ValueData::I64And { lhs, rhs }
        | ValueData::I64Or { lhs, rhs }
        | ValueData::I64Xor { lhs, rhs }
        | ValueData::I64Shl { lhs, rhs }
        | ValueData::I64ShrS { lhs, rhs }
        | ValueData::I64ShrU { lhs, rhs }
        | ValueData::I64Rotl { lhs, rhs }
        | ValueData::I64Rotr { lhs, rhs }
        | ValueData::I64Eq { lhs, rhs }
        | ValueData::I64Ne { lhs, rhs }
        | ValueData::I64LtS { lhs, rhs }
        | ValueData::I64LtU { lhs, rhs }
        | ValueData::I64GtS { lhs, rhs }
        | ValueData::I64GtU { lhs, rhs }
        | ValueData::I64LeS { lhs, rhs }
        | ValueData::I64LeU { lhs, rhs }
        | ValueData::I64GeS { lhs, rhs }
        | ValueData::I64GeU { lhs, rhs } => {
            expr_references_local(lhs, local_idx) || expr_references_local(rhs, local_idx)
        }
        ValueData::Select {
            cond,
            true_val,
            false_val,
        } => {
            expr_references_local(cond, local_idx)
                || expr_references_local(true_val, local_idx)
                || expr_references_local(false_val, local_idx)
        }
        // A pinned expr only ever holds is_forwardable_expr shapes; anything else
        // conservatively "might reference" → invalidate.
        _ => true,
    }
}

/// #219: a local `idx` was just (re)assigned — drop any pinned forwarding whose
/// expression references it, so a forwarded expression never captures a stale or
/// post-redefinition input value.
fn invalidate_pins_referencing(env: &mut OptimizationEnv, idx: u32) {
    env.pinned
        .retain(|_, expr| !expr_references_local(expr, idx));
}

/// Dataflow-aware ISLE rewrite — tracks local variables and memory state.
///
/// Applies all pure structural rewrites (constant folding, algebraic
/// identities, strength reduction) PLUS:
/// - Local variable constant propagation (LocalSet/Get/Tee tracking)
/// - Memory redundancy elimination (load-after-store forwarding)
/// - Environment invalidation at control flow boundaries
///
/// Only safe for straight-line code or with env clearing at join points.
/// For functions with BrIf/BrTable, use `rewrite_pure` instead.
///
/// #219: when `env.single_assign` is non-empty, also forwards those
/// single-assignment locals' pure defining expressions (carrier scalar-
/// forwarding), with a reaching-defs invalidation guard.
pub fn rewrite_with_dataflow(val: Value, env: &mut OptimizationEnv) -> Value {
    match val.data() {
        // Local variable operations
        ValueData::LocalSet { idx, val: set_val } => {
            let simplified_val = rewrite_with_dataflow(set_val.clone(), env);

            // Track this assignment in our environment
            if matches!(
                simplified_val.data(),
                ValueData::I32Const { .. } | ValueData::I64Const { .. }
            ) {
                env.locals.insert(*idx, simplified_val.clone());
            } else {
                env.locals.remove(idx);
            }

            // #219 reaching-defs guard: reassigning `idx` invalidates any pinned
            // forwarding that references it (so a forwarded expr never captures a
            // post-redefinition input). Then, if `idx` is the single-assignment
            // carrier with a forwardable RHS, pin its defining expression.
            invalidate_pins_referencing(env, *idx);
            if env.single_assign.contains(idx) && is_forwardable_expr(&simplified_val) {
                env.pinned.insert(*idx, simplified_val.clone());
            }

            local_set(*idx, simplified_val)
        }

        ValueData::LocalGet { idx } => {
            // Look up in environment - dataflow analysis!
            if let Some(known_val) = env.locals.get(idx) {
                known_val.clone()
            } else if let Some(pinned_val) = env.pinned.get(idx) {
                // #219 carrier scalar-forwarding: forward the single-assignment
                // local's defining expression to this use, exposing pack/unpack.
                pinned_val.clone()
            } else {
                local_get(*idx)
            }
        }

        ValueData::LocalTee { idx, val: tee_val } => {
            let simplified_val = rewrite_with_dataflow(tee_val.clone(), env);

            if matches!(
                simplified_val.data(),
                ValueData::I32Const { .. } | ValueData::I64Const { .. }
            ) {
                env.locals.insert(*idx, simplified_val.clone());
            } else {
                env.locals.remove(idx);
            }

            // #219 (see LocalSet): invalidate pins referencing `idx`, then pin the
            // single-assignment carrier. local.tee both stores AND leaves the value.
            invalidate_pins_referencing(env, *idx);
            if env.single_assign.contains(idx) && is_forwardable_expr(&simplified_val) {
                env.pinned.insert(*idx, simplified_val.clone());
            }

            local_tee(*idx, simplified_val)
        }

        // Memory operations - Phase 13: Memory Redundancy Elimination!
        ValueData::I32Load {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);

            // Try to extract memory location
            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                    mem: *mem,
                };

                // Redundant load elimination: check if we know this value!
                if let Some(known_value) = env.memory.get(&mem_loc) {
                    // OPTIMIZATION: Return known value instead of loading!
                    return known_value.clone();
                }
            }

            i32_load(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I32Store {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);

            // Track this store in memory state
            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                    mem: *mem,
                };

                // Store the value in our memory tracking
                if matches!(simplified_value.data(), ValueData::I32Const { .. }) {
                    env.memory.insert(mem_loc, simplified_value.clone());
                }
            } else {
                // Unknown address - invalidate all memory conservatively
                env.invalidate_memory();
            }

            i32_store(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        ValueData::I64Load {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);

            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                    mem: *mem,
                };

                if let Some(known_value) = env.memory.get(&mem_loc) {
                    return known_value.clone();
                }
            }

            i64_load(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I64Store {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);

            if let ValueData::I32Const { val: addr_val } = simplified_addr.data() {
                let mem_loc = MemoryLocation {
                    base: Some(addr_val.value()),
                    offset: *offset,
                    mem: *mem,
                };

                if matches!(simplified_value.data(), ValueData::I64Const { .. }) {
                    env.memory.insert(mem_loc, simplified_value.clone());
                }
            } else {
                env.invalidate_memory();
            }

            i64_store(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        // Float memory operations - simplify address, no memory tracking
        ValueData::F32Load {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            f32_load(simplified_addr, *offset, *align, *mem)
        }

        ValueData::F32Store {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);
            // Unknown store type - invalidate conservatively
            env.invalidate_memory();
            f32_store(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        ValueData::F64Load {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            f64_load(simplified_addr, *offset, *align, *mem)
        }

        ValueData::F64Store {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);
            env.invalidate_memory();
            f64_store(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        // Partial-width memory load operations
        // These simplify their address but don't participate in memory tracking
        // because they load different widths than what might have been stored
        ValueData::I32Load8S {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i32_load8_s(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I32Load8U {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i32_load8_u(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I32Load16S {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i32_load16_s(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I32Load16U {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i32_load16_u(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I64Load8S {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i64_load8_s(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I64Load8U {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i64_load8_u(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I64Load16S {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i64_load16_s(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I64Load16U {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i64_load16_u(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I64Load32S {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i64_load32_s(simplified_addr, *offset, *align, *mem)
        }

        ValueData::I64Load32U {
            addr,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            i64_load32_u(simplified_addr, *offset, *align, *mem)
        }

        // Partial-width memory store operations - simplify address and value,
        // invalidate memory conservatively (different width than full stores)
        ValueData::I32Store8 {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);
            env.invalidate_memory();
            i32_store8(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        ValueData::I32Store16 {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);
            env.invalidate_memory();
            i32_store16(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        ValueData::I64Store8 {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);
            env.invalidate_memory();
            i64_store8(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        ValueData::I64Store16 {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);
            env.invalidate_memory();
            i64_store16(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        ValueData::I64Store32 {
            addr,
            value,
            offset,
            align,
            mem,
        } => {
            let simplified_addr = rewrite_with_dataflow(addr.clone(), env);
            let simplified_value = rewrite_with_dataflow(value.clone(), env);
            env.invalidate_memory();
            i64_store32(simplified_addr, simplified_value, *offset, *align, *mem)
        }

        // Structured control flow: recursively optimize bodies with appropriate
        // env save/restore semantics. This is the key enabler for optimizing
        // functions with BrIf/BrTable — we descend into each body with the
        // correct env state rather than skipping the entire function.
        ValueData::Block {
            label,
            block_type,
            body,
        } => {
            // Clear env at block entry. A block can be the target of a br from
            // an inner loop, meaning it can be reached with different local values
            // than the linear env tracks. Conservative: start fresh.
            env.locals.clear();
            env.invalidate_memory();
            let optimized_body: Vec<Value> = body
                .iter()
                .map(|term| rewrite_with_dataflow(term.clone(), env))
                .collect();
            // After block: clear env. A br/br_if inside can exit early.
            env.locals.clear();
            env.invalidate_memory();
            Value(Box::new(ValueData::Block {
                label: label.clone(),
                block_type: block_type.clone(),
                body: optimized_body,
            }))
        }

        ValueData::Loop {
            label,
            block_type,
            body,
        } => {
            // Clear env at loop entry — a loop can be re-entered from a
            // back-edge (br to loop label). Values set in a previous iteration
            // are unknown at re-entry. This is the critical fix for the
            // Z3-identified unsoundness.
            env.locals.clear();
            env.invalidate_memory();
            let optimized_body: Vec<Value> = body
                .iter()
                .map(|term| rewrite_with_dataflow(term.clone(), env))
                .collect();
            // After loop: clear env. We don't know which iteration's values
            // the locals hold at loop exit.
            env.locals.clear();
            env.invalidate_memory();
            Value(Box::new(ValueData::Loop {
                label: label.clone(),
                block_type: block_type.clone(),
                body: optimized_body,
            }))
        }

        ValueData::If {
            label,
            block_type,
            condition,
            then_body,
            else_body,
        } => {
            // Simplify condition with current env
            let simplified_cond = rewrite_with_dataflow(condition.clone(), env);
            // Fork env for each branch
            let mut then_env = env.clone();
            let mut else_env = env.clone();
            let optimized_then: Vec<Value> = then_body
                .iter()
                .map(|term| rewrite_with_dataflow(term.clone(), &mut then_env))
                .collect();
            let optimized_else: Vec<Value> = else_body
                .iter()
                .map(|term| rewrite_with_dataflow(term.clone(), &mut else_env))
                .collect();
            // After if: clear env. We don't know which branch was taken.
            env.locals.clear();
            env.invalidate_memory();
            // #219 reaching-defs across the fork: keep a pin only if it SURVIVED
            // in BOTH branches. A branch that reassigned one of the pin's input
            // locals dropped it (via invalidate_pins_referencing during that
            // branch), so the intersection drops any pin whose inputs could have
            // changed on either path — sound regardless of which branch ran.
            env.pinned
                .retain(|k, _| then_env.pinned.contains_key(k) && else_env.pinned.contains_key(k));
            Value(Box::new(ValueData::If {
                label: label.clone(),
                block_type: block_type.clone(),
                condition: simplified_cond,
                then_body: optimized_then,
                else_body: optimized_else,
            }))
        }

        // Unconditional branch and return: code after these is dead.
        // Clear env so dead code doesn't pollute tracked state.
        ValueData::Br { depth, value } => {
            let simplified_val = value
                .as_ref()
                .map(|v| Box::new(rewrite_with_dataflow(v.as_ref().clone(), env)));
            env.locals.clear();
            env.invalidate_memory();
            Value(Box::new(ValueData::Br {
                depth: *depth,
                value: simplified_val,
            }))
        }

        ValueData::Return { values } => {
            let simplified_vals: Vec<Value> = values
                .iter()
                .map(|v| rewrite_with_dataflow(v.clone(), env))
                .collect();
            env.locals.clear();
            env.invalidate_memory();
            Value(Box::new(ValueData::Return {
                values: simplified_vals,
            }))
        }

        // Control flow that creates multiple execution paths: clear tracked state.
        // After a BrIf, execution may continue (branch not taken) or jump away
        // (branch taken). We cannot assume locals set before BrIf are still valid
        // because the branch target may have different assignments.
        // After a BrTable, any of the targets may be taken.
        // After a Call/CallIndirect, the callee may modify globals and memory,
        // and we cannot track its effects on our dataflow state.
        ValueData::BrIf {
            depth,
            condition,
            value,
        } => {
            let simplified_cond = rewrite_with_dataflow(condition.clone(), env);
            let simplified_val = value
                .as_ref()
                .map(|v| rewrite_with_dataflow(v.as_ref().clone(), env));
            // Invalidate all tracked state at conditional branch point
            env.locals.clear();
            env.invalidate_memory();
            br_if(*depth, simplified_cond, simplified_val)
        }

        ValueData::BrTable {
            targets,
            default,
            index,
            value,
        } => {
            let simplified_index = rewrite_with_dataflow(index.clone(), env);
            let simplified_val = value
                .as_ref()
                .map(|v| Box::new(rewrite_with_dataflow(v.as_ref().clone(), env)));
            // Invalidate all tracked state at multi-way branch
            env.locals.clear();
            env.invalidate_memory();
            Value(Box::new(ValueData::BrTable {
                targets: targets.clone(),
                default: *default,
                index: simplified_index,
                value: simplified_val,
            }))
        }

        ValueData::Call { func_idx, args } => {
            let simplified_args: Vec<Value> = args
                .iter()
                .map(|a| rewrite_with_dataflow(a.clone(), env))
                .collect();
            // Calls may have arbitrary side effects — invalidate all state
            env.locals.clear();
            env.invalidate_memory();
            Value(Box::new(ValueData::Call {
                func_idx: *func_idx,
                args: simplified_args,
            }))
        }

        ValueData::CallIndirect {
            type_idx,
            table_idx,
            table_offset,
            args,
        } => {
            let simplified_offset = rewrite_with_dataflow(table_offset.clone(), env);
            let simplified_args: Vec<Value> = args
                .iter()
                .map(|a| rewrite_with_dataflow(a.clone(), env))
                .collect();
            // Indirect calls have unknown side effects — invalidate all state
            env.locals.clear();
            env.invalidate_memory();
            Value(Box::new(ValueData::CallIndirect {
                type_idx: *type_idx,
                table_idx: *table_idx,
                table_offset: simplified_offset,
                args: simplified_args,
            }))
        }

        // #240 Tier-2 range-gated flagship: (x << k) >>u k -> x when a premise
        // proves x < 2^(32-k). Env-aware because it consults `env.value_max`;
        // the unconditional version (mask to 2^(32-k)-1) lives in
        // rewrite_pure_impl and still fires when no premise is present.
        //
        // Soundness (Z3, UNDER the premise): (x<<k)>>u k == x & (2^(32-k)-1) for
        // 0<k<32; when x < 2^(32-k) every set bit of x is below bit (32-k) so
        // the mask is the identity and the whole expression equals x. The
        // motivating shape `256*(ch-1024) >> 8` (== `(x<<8)>>u8`, x < 2^24)
        // folds to `x` — LLVM's fold, now available to loom BUT only with the
        // machine-checked bound.
        ValueData::I32ShrU { lhs, rhs } => {
            let lhs_s = rewrite_with_dataflow(lhs.clone(), env);
            let rhs_s = rewrite_with_dataflow(rhs.clone(), env);
            if let (
                ValueData::I32Shl {
                    lhs: inner,
                    rhs: shl_amt,
                },
                ValueData::I32Const { val: k },
            ) = (lhs_s.data(), rhs_s.data())
            {
                if i32_shr_undoes_shl(shl_amt, k) {
                    let kk = (k.value() as u32) & 0x1F;
                    // surviving mask is 2^(32-kk)-1; the premise must prove
                    // inner < 2^(32-kk) for the mask to be a no-op.
                    if env.fits_below_bit(inner, 32 - kk) {
                        return inner.clone();
                    }
                }
            }
            // No premise (or shape mismatch): reconstruct and apply the
            // unconditional structural rules (incl. the masking fold).
            rewrite_pure_impl(ishru32(lhs_s, rhs_s))
        }

        // All other optimizations follow...
        _ => rewrite_pure_impl(val),
    }
}

/// Pure structural ISLE rewrites — no dataflow state, safe for all control flow.
///
/// Applies: constant folding, algebraic identities, strength reduction.
/// Does NOT propagate local variable values or track memory state.
/// Use `rewrite_with_dataflow` for straight-line code where local/memory
/// propagation is safe.
pub fn rewrite_pure(val: Value) -> Value {
    rewrite_pure_impl(val)
}

/// Backward-compatible alias for `rewrite_pure`.
#[doc(hidden)]
pub fn simplify(val: Value) -> Value {
    rewrite_pure(val)
}

/// Backward-compatible alias for `rewrite_with_dataflow`.
#[doc(hidden)]
pub fn simplify_with_env(val: Value, env: &mut OptimizationEnv) -> Value {
    rewrite_with_dataflow(val, env)
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

/// #219 seam-SROA helper: true if `op` is `(i64.shl _ (i64.const k))` with
/// `0 < k < 64` and every set bit of `mask` is below k (`(mask as u64) >> k ==
/// 0`). In that case `(op & mask) == 0` for ALL shift inputs — the shifted
/// value only occupies bits [k,64), which the mask zeroes. k is restricted to
/// (0,64) so the wasm shift-amount-mod-64 wrap can't change the effective
/// shift; k>=64 is left to the verifier rather than reasoned about here.
fn i64_shl_cleared_by_mask(op: &Value, mask: &Imm64) -> bool {
    if let ValueData::I64Shl { rhs, .. } = op.data() {
        if let ValueData::I64Const { val: k } = rhs.data() {
            let k = k.value();
            if k > 0 && k < 64 {
                return (mask.value() as u64) >> (k as u64) == 0;
            }
        }
    }
    false
}

/// #219 seam-SROA helper: true if `shl_amt` is the constant `k` and `0 < k <
/// 64` — i.e. an `(i64.shl _ k)` whose amount equals the enclosing `shr_u`'s.
/// `(Z << k) >> k == Z & (u64::MAX >> k)` exactly when k is in (0,64) (no wasm
/// shift-mod-64 wrap).
fn i64_shr_undoes_shl(shl_amt: &Value, shr_amt: &Imm64) -> bool {
    if let ValueData::I64Const { val: k2 } = shl_amt.data() {
        let k = shr_amt.value();
        return k == k2.value() && k > 0 && k < 64;
    }
    false
}

/// #240 algebraic mid-end helper: true if `shl_amt` is a constant `k2` equal to
/// the enclosing `shr_u`'s amount `k`, with `0 < k < 32`. In that case
/// `(x << k) >>u k == x & (2^(32-k)-1)` exactly (no wasm shift-mod-32 wrap). The
/// i32 mirror of `i64_shr_undoes_shl`.
fn i32_shr_undoes_shl(shl_amt: &Value, shr_amt: &Imm32) -> bool {
    if let ValueData::I32Const { val: k2 } = shl_amt.data() {
        let k = shr_amt.value() & 0x1F;
        return k == (k2.value() & 0x1F) && k > 0 && k < 32;
    }
    false
}

/// #240 helper: the effective i32 shift amount `k & 0x1F` if `v` is an
/// `(i32.const k)`, else `None`. Used to fuse double shifts.
fn i32_shift_amount(v: &Value) -> Option<u32> {
    if let ValueData::I32Const { val } = v.data() {
        Some((val.value() as u32) & 0x1F)
    } else {
        None
    }
}

/// #240 helper: true if `a` is `(i32.const)` and `(a&0x1F)+(b&0x1F) < 32`. The
/// double-shift collapse `(x<<a)<<b → x<<(a+b)` (and the shr_u analogue) is only
/// sound while the fused amount stays below the wasm shift-mod-32 wrap.
fn i32_shift_sum_lt_width(a: &Value, b: &Imm32) -> bool {
    match i32_shift_amount(a) {
        Some(av) => av + ((b.value() as u32) & 0x1F) < 32,
        None => false,
    }
}

/// #240 helper: effective i64 shift amount `k & 0x3F` for an `(i64.const k)`.
fn i64_shift_amount(v: &Value) -> Option<u64> {
    if let ValueData::I64Const { val } = v.data() {
        Some((val.value() as u64) & 0x3F)
    } else {
        None
    }
}

/// #240 helper: true if `a` is `(i64.const)` and `(a&0x3F)+(b&0x3F) < 64`.
fn i64_shift_sum_lt_width(a: &Value, b: &Imm64) -> bool {
    match i64_shift_amount(a) {
        Some(av) => av + ((b.value() as u64) & 0x3F) < 64,
        None => false,
    }
}

/// #219 seam-SROA helper: true if `op` is `(i64.shl _ (i64.const amt))` with
/// `0 < amt < 64`. Used to target the shr_u-over-or distribution at the pack
/// shape (`(or (shl Z k) B) >> k`) so it only fires where it dissolves.
fn i64_is_shl_by(op: &Value, amt: i64) -> bool {
    if amt <= 0 || amt >= 64 {
        return false;
    }
    if let ValueData::I64Shl { rhs, .. } = op.data() {
        if let ValueData::I64Const { val: k } = rhs.data() {
            return k.value() == amt;
        }
    }
    false
}

/// Stateless simplification (expression-level only)
fn rewrite_pure_impl(val: Value) -> Value {
    match val.data() {
        // i32.add optimizations
        ValueData::I32Add { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
                // Algebraic: 0 - (0 - x) = x (double negation)
                (
                    ValueData::I32Const { val: l },
                    ValueData::I32Sub {
                        lhs: inner_l,
                        rhs: inner_r,
                    },
                ) if l.value() == 0 => {
                    if let ValueData::I32Const { val: inner_l_val } = inner_l.data() {
                        if inner_l_val.value() == 0 {
                            return inner_r.clone();
                        }
                    }
                    isub32(lhs_simplified, rhs_simplified)
                }
                _ => isub32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.mul optimizations
        ValueData::I32Mul { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding: (i32.mul (i32.const A) (i32.const B)) → (i32.const (A*B))
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_mul(*lhs_val, *rhs_val))
                }
                // Algebraic: x * 0 = 0 — only when the DISCARDED `x` cannot trap
                // (#278: a trapping load/div in x must still fault).
                (_, ValueData::I32Const { val })
                    if val.value() == 0 && is_no_trap_expr(&lhs_simplified) =>
                {
                    iconst32(Imm32(0))
                }
                // Algebraic: 0 * x = 0 — only when the DISCARDED `x` cannot trap.
                (ValueData::I32Const { val }, _)
                    if val.value() == 0 && is_no_trap_expr(&rhs_simplified) =>
                {
                    iconst32(Imm32(0))
                }
                // Algebraic: x * 1 = x
                (_, ValueData::I32Const { val }) if val.value() == 1 => lhs_simplified,
                // Algebraic: 1 * x = x
                (ValueData::I32Const { val }, _) if val.value() == 1 => rhs_simplified,
                // Algebraic: x * -1 = 0 - x (negate)
                (_, ValueData::I32Const { val }) if val.value() == -1 => {
                    isub32(iconst32(Imm32(0)), lhs_simplified)
                }
                // Algebraic: -1 * x = 0 - x (negate)
                (ValueData::I32Const { val }, _) if val.value() == -1 => {
                    isub32(iconst32(Imm32(0)), rhs_simplified)
                }
                // Strength reduction: x * power_of_2 → x << log2(power_of_2)
                (_, ValueData::I32Const { val: rhs_val })
                    if is_power_of_two_i32(*rhs_val).is_some() =>
                {
                    let shift_amount = log2_i32(*rhs_val);
                    ishl32(lhs_simplified, iconst32(shift_amount))
                }
                // Strength reduction: power_of_2 * x → x << log2(power_of_2)
                (ValueData::I32Const { val: lhs_val }, _)
                    if is_power_of_two_i32(*lhs_val).is_some() =>
                {
                    let shift_amount = log2_i32(*lhs_val);
                    ishl32(rhs_simplified, iconst32(shift_amount))
                }
                _ => imul32(lhs_simplified, rhs_simplified),
            }
        }

        // i64.add optimizations
        ValueData::I64Add { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            // Check for x - x = 0 pattern (self-subtraction)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return iconst64(Imm64(0));
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_sub(*lhs_val, *rhs_val))
                }
                (_, ValueData::I64Const { val }) if val.value() == 0 => lhs_simplified,
                // Algebraic: 0 - (0 - x) = x (double negation)
                (
                    ValueData::I64Const { val: l },
                    ValueData::I64Sub {
                        lhs: inner_l,
                        rhs: inner_r,
                    },
                ) if l.value() == 0 => {
                    if let ValueData::I64Const { val: inner_l_val } = inner_l.data() {
                        if inner_l_val.value() == 0 {
                            return inner_r.clone();
                        }
                    }
                    isub64(lhs_simplified, rhs_simplified)
                }
                _ => isub64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.mul optimizations
        ValueData::I64Mul { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_mul(*lhs_val, *rhs_val))
                }
                // Algebraic: x * 0 = 0 / 0 * x = 0 — only when the DISCARDED
                // operand cannot trap (#278).
                (_, ValueData::I64Const { val })
                    if val.value() == 0 && is_no_trap_expr(&lhs_simplified) =>
                {
                    iconst64(Imm64(0))
                }
                (ValueData::I64Const { val }, _)
                    if val.value() == 0 && is_no_trap_expr(&rhs_simplified) =>
                {
                    iconst64(Imm64(0))
                }
                (_, ValueData::I64Const { val }) if val.value() == 1 => lhs_simplified,
                (ValueData::I64Const { val }, _) if val.value() == 1 => rhs_simplified,
                // Algebraic: x * -1 = 0 - x (negate)
                (_, ValueData::I64Const { val }) if val.value() == -1 => {
                    isub64(iconst64(Imm64(0)), lhs_simplified)
                }
                // Algebraic: -1 * x = 0 - x (negate)
                (ValueData::I64Const { val }, _) if val.value() == -1 => {
                    isub64(iconst64(Imm64(0)), rhs_simplified)
                }
                // Strength reduction: x * power_of_2 → x << log2(power_of_2)
                // WebAssembly spec: i64.shl takes i64 for the shift amount
                (_, ValueData::I64Const { val: rhs_val })
                    if is_power_of_two_i64(*rhs_val).is_some() =>
                {
                    let shift_amount = log2_i64(*rhs_val);
                    ishl64(lhs_simplified, iconst64(shift_amount))
                }
                // Strength reduction: power_of_2 * x → x << log2(power_of_2)
                // WebAssembly spec: i64.shl takes i64 for the shift amount
                (ValueData::I64Const { val: lhs_val }, _)
                    if is_power_of_two_i64(*lhs_val).is_some() =>
                {
                    let shift_amount = log2_i64(*lhs_val);
                    ishl64(rhs_simplified, iconst64(shift_amount))
                }
                _ => imul64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.div_u optimizations (unsigned division)
        ValueData::I64DivU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Strength reduction: x / power_of_2 → x >> log2(power_of_2)
                // WebAssembly spec: i64.shr_u takes i64 for the shift amount
                (_, ValueData::I64Const { val: rhs_val })
                    if is_power_of_two_i64(*rhs_val).is_some() =>
                {
                    let shift_amount = log2_i64(*rhs_val);
                    ishru64(lhs_simplified, iconst64(shift_amount))
                }
                // Algebraic: x / 1 = x
                (_, ValueData::I64Const { val }) if val.value() == 1 => lhs_simplified,
                _ => idivu64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.rem_u optimizations (unsigned remainder/modulo)
        ValueData::I64RemU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Strength reduction: x % power_of_2 → x & (power_of_2 - 1)
                (_, ValueData::I64Const { val: rhs_val })
                    if is_power_of_two_i64(*rhs_val).is_some() =>
                {
                    let mask = imm64_sub_1(*rhs_val);
                    iand64(lhs_simplified, iconst64(mask))
                }
                _ => iremu64(lhs_simplified, rhs_simplified),
            }
        }

        // i32.and optimizations
        ValueData::I32And { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            // Check for x & x = x pattern (self-AND)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return lhs_simplified;
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_and(*lhs_val, *rhs_val))
                }
                // Algebraic: x & 0 = 0 — only when the DISCARDED operand cannot
                // trap (#278).
                (_, ValueData::I32Const { val })
                    if val.value() == 0 && is_no_trap_expr(&lhs_simplified) =>
                {
                    iconst32(Imm32(0))
                }
                (ValueData::I32Const { val }, _)
                    if val.value() == 0 && is_no_trap_expr(&rhs_simplified) =>
                {
                    iconst32(Imm32(0))
                }
                // Algebraic: x & -1 = x (all bits set)
                (_, ValueData::I32Const { val }) if val.value() == -1 => lhs_simplified,
                (ValueData::I32Const { val }, _) if val.value() == -1 => rhs_simplified,

                // #240: (x & c1) & c2 → x & (c1 & c2). AND is associative and its
                // constant operand collapses, so two masks fuse into one.
                // Unconditionally sound. Recurse so the fused mask can trip the
                // -1/0 identities above (e.g. redundant re-mask after a narrowing).
                (ValueData::I32And { lhs: x, rhs: c1 }, ValueData::I32Const { val: c2 }) => {
                    if let ValueData::I32Const { val: c1v } = c1.data() {
                        rewrite_pure(iand32(x.clone(), iconst32(imm32_and(*c1v, *c2))))
                    } else {
                        iand32(lhs_simplified, rhs_simplified)
                    }
                }
                _ => iand32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.or optimizations
        ValueData::I32Or { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
                // Algebraic: x | -1 = -1 (all bits set) — only when the DISCARDED
                // operand cannot trap (#278).
                (_, ValueData::I32Const { val })
                    if val.value() == -1 && is_no_trap_expr(&lhs_simplified) =>
                {
                    iconst32(Imm32(-1))
                }
                (ValueData::I32Const { val }, _)
                    if val.value() == -1 && is_no_trap_expr(&rhs_simplified) =>
                {
                    iconst32(Imm32(-1))
                }
                _ => ior32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.xor optimizations
        ValueData::I32Xor { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_shl(*lhs_val, *rhs_val))
                }
                // Algebraic: x << 0 = x
                (_, ValueData::I32Const { val }) if (val.value() & 0x1F) == 0 => lhs_simplified,

                // #240: (x << a) << b → x << (a+b) when a+b < 32. Left shift by a
                // then b equals a single shift by a+b because wrap32 commutes with
                // <<; gated to a+b<32 so the fused amount stays below the wasm
                // shift-mod-32 wrap (for a+b>=32 the single shl would use a
                // smaller effective amount and differ).
                (ValueData::I32Shl { lhs: z, rhs: a }, ValueData::I32Const { val: b })
                    if i32_shift_sum_lt_width(a, b) =>
                {
                    let sum = i32_shift_amount(a).unwrap() + ((b.value() as u32) & 0x1F);
                    rewrite_pure(ishl32(z.clone(), iconst32(Imm32(sum as i32))))
                }
                _ => ishl32(lhs_simplified, rhs_simplified),
            }
        }

        // i32.shr_s optimizations
        ValueData::I32ShrS { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    iconst32(imm32_shr_u(*lhs_val, *rhs_val))
                }
                // Algebraic: x >> 0 = x
                (_, ValueData::I32Const { val }) if (val.value() & 0x1F) == 0 => lhs_simplified,

                // #240 algebraic mid-end: (x << k) >>u k → x & (0xFFFFFFFF >>u k).
                // Shifting an i32 left by k clears the low k bits and pushes x's
                // low (32-k) bits up to [k,32); the matching logical right shift
                // brings them back to [0,32-k), which is exactly x masked to its
                // low (32-k) bits. Unconditionally sound for 0<k<32 (Z3:
                // (x<<k)>>u k == x & (2^(32-k)-1)); the k restriction avoids the
                // wasm shift-amount-mod-32 wrap. This folds synth's literal
                // `256*(x-c) >> 8` shape (== `(x'<<8) >>u 8`) to a single mask,
                // and when the caller already knows x < 2^(32-k) the mask is a
                // no-op the Tier-2 range gate removes entirely.
                (ValueData::I32Shl { lhs: z, rhs: shamt }, ValueData::I32Const { val: k })
                    if i32_shr_undoes_shl(shamt, k) =>
                {
                    let kk = (k.value() as u32) & 0x1F;
                    let mask = (u32::MAX >> kk) as i32;
                    rewrite_pure(iand32(z.clone(), iconst32(Imm32(mask))))
                }
                // #240: (x >>u a) >>u b → x >>u (a+b) when a+b < 32. Logical
                // right shifts compose exactly (no value wrap); gated to a+b<32
                // so the fused amount doesn't cross the wasm shift-mod-32 wrap.
                (ValueData::I32ShrU { lhs: z, rhs: a }, ValueData::I32Const { val: b })
                    if i32_shift_sum_lt_width(a, b) =>
                {
                    let sum = i32_shift_amount(a).unwrap() + ((b.value() as u32) & 0x1F);
                    rewrite_pure(ishru32(z.clone(), iconst32(Imm32(sum as i32))))
                }
                _ => ishru32(lhs_simplified, rhs_simplified),
            }
        }

        // i64.and optimizations
        ValueData::I64And { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            // Check for x & x = x pattern (self-AND)
            if are_values_equal(&lhs_simplified, &rhs_simplified) {
                return lhs_simplified;
            }

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_and(*lhs_val, *rhs_val))
                }
                // Algebraic: x & 0 = 0 — only when the DISCARDED operand cannot
                // trap (#278).
                (_, ValueData::I64Const { val })
                    if val.value() == 0 && is_no_trap_expr(&lhs_simplified) =>
                {
                    iconst64(Imm64(0))
                }
                (ValueData::I64Const { val }, _)
                    if val.value() == 0 && is_no_trap_expr(&rhs_simplified) =>
                {
                    iconst64(Imm64(0))
                }
                // Algebraic: x & -1 = x (all bits set)
                (_, ValueData::I64Const { val }) if val.value() == -1 => lhs_simplified,
                (ValueData::I64Const { val }, _) if val.value() == -1 => rhs_simplified,

                // #219 seam-SROA: (shl Z k) & M → 0 when M's set bits are all
                // below k — the left shift zeroes bits [0,k), the mask zeroes
                // bits [k,64), so nothing survives. Unconditionally sound for
                // any Z (Z3: (Z<<k) & M == 0 when M >> k == 0). Dissolves the
                // high half of a u64 pack under a low-byte unpack mask.
                // #278: the whole `(shl Z k)` operand is DISCARDED, so only fire
                // when Z (and thus the shl) cannot trap.
                (ValueData::I64Shl { .. }, ValueData::I64Const { val: m })
                    if i64_shl_cleared_by_mask(&lhs_simplified, m)
                        && is_no_trap_expr(&lhs_simplified) =>
                {
                    iconst64(Imm64(0))
                }
                (ValueData::I64Const { val: m }, ValueData::I64Shl { .. })
                    if i64_shl_cleared_by_mask(&rhs_simplified, m)
                        && is_no_trap_expr(&rhs_simplified) =>
                {
                    iconst64(Imm64(0))
                }
                // #219 seam-SROA: extend_i32_u(x) & M → extend_i32_u(x) when
                // M's low 32 bits are all set. Zero-extending an i32 yields a
                // value in [0, 2^32), so a mask covering bits [0,32) preserves
                // it (bits [32,64) of the extend are already 0). Z3: (zext32 x)
                // & M == (zext32 x) when (M & 0xffffffff) == 0xffffffff.
                (ValueData::I64ExtendI32U { .. }, ValueData::I64Const { val: m })
                    if (m.value() as u64) & 0xffff_ffff == 0xffff_ffff =>
                {
                    lhs_simplified
                }
                (ValueData::I64Const { val: m }, ValueData::I64ExtendI32U { .. })
                    if (m.value() as u64) & 0xffff_ffff == 0xffff_ffff =>
                {
                    rhs_simplified
                }
                // #219 seam-SROA: (or A B) & M → (survivor & M) when one OR
                // operand is a left shift the mask clears. Recurse so the
                // survivor (and a both-shifted case) simplifies further.
                // #278: the CLEARED operand is DISCARDED, so only fire when it
                // cannot trap.
                (ValueData::I64Or { lhs: a, rhs: b }, ValueData::I64Const { val: m })
                    if (i64_shl_cleared_by_mask(a, m) && is_no_trap_expr(a))
                        || (i64_shl_cleared_by_mask(b, m) && is_no_trap_expr(b)) =>
                {
                    let cleared_a = i64_shl_cleared_by_mask(a, m) && is_no_trap_expr(a);
                    let survivor = if cleared_a { b } else { a };
                    rewrite_pure(iand64(survivor.clone(), iconst64(*m)))
                }
                (ValueData::I64Const { val: m }, ValueData::I64Or { lhs: a, rhs: b })
                    if (i64_shl_cleared_by_mask(a, m) && is_no_trap_expr(a))
                        || (i64_shl_cleared_by_mask(b, m) && is_no_trap_expr(b)) =>
                {
                    let cleared_a = i64_shl_cleared_by_mask(a, m) && is_no_trap_expr(a);
                    let survivor = if cleared_a { b } else { a };
                    rewrite_pure(iand64(survivor.clone(), iconst64(*m)))
                }
                // #240: (x & c1) & c2 → x & (c1 & c2) (see i32 note). Associative
                // AND with a constant fold; recurse for the -1/0 identities.
                (ValueData::I64And { lhs: x, rhs: c1 }, ValueData::I64Const { val: c2 }) => {
                    if let ValueData::I64Const { val: c1v } = c1.data() {
                        rewrite_pure(iand64(x.clone(), iconst64(imm64_and(*c1v, *c2))))
                    } else {
                        iand64(lhs_simplified, rhs_simplified)
                    }
                }
                _ => iand64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.or optimizations
        ValueData::I64Or { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
                // Algebraic: x | -1 = -1 (all bits set) — only when the DISCARDED
                // operand cannot trap (#278).
                (_, ValueData::I64Const { val })
                    if val.value() == -1 && is_no_trap_expr(&lhs_simplified) =>
                {
                    iconst64(Imm64(-1))
                }
                (ValueData::I64Const { val }, _)
                    if val.value() == -1 && is_no_trap_expr(&rhs_simplified) =>
                {
                    iconst64(Imm64(-1))
                }
                _ => ior64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.xor optimizations
        ValueData::I64Xor { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_shl(*lhs_val, *rhs_val))
                }
                // Algebraic: x << 0 = x
                (_, ValueData::I64Const { val }) if (val.value() & 0x3F) == 0 => lhs_simplified,

                // #240: (x << a) << b → x << (a+b) when a+b < 64 (see i32 note).
                (ValueData::I64Shl { lhs: z, rhs: a }, ValueData::I64Const { val: b })
                    if i64_shift_sum_lt_width(a, b) =>
                {
                    let sum = i64_shift_amount(a).unwrap() + ((b.value() as u64) & 0x3F);
                    rewrite_pure(ishl64(z.clone(), iconst64(Imm64(sum as i64))))
                }
                _ => ishl64(lhs_simplified, rhs_simplified),
            }
        }

        // i64.shr_s optimizations
        ValueData::I64ShrS { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    iconst64(imm64_shr_u(*lhs_val, *rhs_val))
                }
                // Algebraic: x >> 0 = x
                (_, ValueData::I64Const { val }) if (val.value() & 0x3F) == 0 => lhs_simplified,

                // #219 seam-SROA: (shr_u (shl Z k) k) → Z & (low 64-k bits).
                // Shifting left by k then logically right by the same k clears
                // the high k bits and keeps the low 64-k. Unconditionally sound
                // for 0<k<64 (Z3: (Z<<k)>>k == Z & (2^(64-k)-1)). Dissolves the
                // high half of a u64 pack under a >>k unpack.
                (ValueData::I64Shl { lhs: z, rhs: shamt }, ValueData::I64Const { val: k })
                    if i64_shr_undoes_shl(shamt, k) =>
                {
                    let mask = (u64::MAX >> (k.value() as u64)) as i64;
                    rewrite_pure(iand64(z.clone(), iconst64(Imm64(mask))))
                }
                // #219 seam-SROA: (shr_u (or P Q) k) → (or (P>>k) (Q>>k)) when an
                // OR operand is (shl _ k) — logical right shift distributes over
                // bitwise OR (sound), and recursing lets the shl side collapse
                // via the rule above. Targeted to the pack shape (matching shl
                // present) so it never bloats unrelated `(or _ _) >> k`.
                (ValueData::I64Or { lhs: p, rhs: q }, ValueData::I64Const { val: k })
                    if i64_is_shl_by(p, k.value()) || i64_is_shl_by(q, k.value()) =>
                {
                    rewrite_pure(ior64(
                        ishru64(p.clone(), iconst64(*k)),
                        ishru64(q.clone(), iconst64(*k)),
                    ))
                }
                // #240: (x >>u a) >>u b → x >>u (a+b) when a+b < 64 (see i32 note).
                (ValueData::I64ShrU { lhs: z, rhs: a }, ValueData::I64Const { val: b })
                    if i64_shift_sum_lt_width(a, b) =>
                {
                    let sum = i64_shift_amount(a).unwrap() + ((b.value() as u64) & 0x3F);
                    rewrite_pure(ishru64(z.clone(), iconst64(Imm64(sum as i64))))
                }
                _ => ishru64(lhs_simplified, rhs_simplified),
            }
        }

        // Rotation optimizations (i32)
        ValueData::I32Rotl { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    let lhs_u = lhs_val.value() as u32;
                    let rhs_u = (rhs_val.value() as u32) & 0x1F; // Rotation amount mod 32
                    let result = lhs_u.rotate_left(rhs_u);
                    iconst32(Imm32(result as i32))
                }
                // Algebraic: x rotl 0 = x
                (_, ValueData::I32Const { val }) if (val.value() & 0x1F) == 0 => lhs_simplified,
                _ => irotl32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32Rotr { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I32Const { val: lhs_val }, ValueData::I32Const { val: rhs_val }) => {
                    let lhs_u = lhs_val.value() as u32;
                    let rhs_u = (rhs_val.value() as u32) & 0x1F; // Rotation amount mod 32
                    let result = lhs_u.rotate_right(rhs_u);
                    iconst32(Imm32(result as i32))
                }
                // Algebraic: x rotr 0 = x
                (_, ValueData::I32Const { val }) if (val.value() & 0x1F) == 0 => lhs_simplified,
                _ => irotr32(lhs_simplified, rhs_simplified),
            }
        }

        // Rotation optimizations (i64)
        ValueData::I64Rotl { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    let lhs_u = lhs_val.value() as u64;
                    let rhs_u = (rhs_val.value() as u64) & 0x3F; // Rotation amount mod 64
                    let result = lhs_u.rotate_left(rhs_u as u32);
                    iconst64(Imm64(result as i64))
                }
                // Algebraic: x rotl 0 = x
                (_, ValueData::I64Const { val }) if (val.value() & 0x3F) == 0 => lhs_simplified,
                _ => irotl64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64Rotr { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding
                (ValueData::I64Const { val: lhs_val }, ValueData::I64Const { val: rhs_val }) => {
                    let lhs_u = lhs_val.value() as u64;
                    let rhs_u = (rhs_val.value() as u64) & 0x3F; // Rotation amount mod 64
                    let result = lhs_u.rotate_right(rhs_u as u32);
                    iconst64(Imm64(result as i64))
                }
                // Algebraic: x rotr 0 = x
                (_, ValueData::I64Const { val }) if (val.value() & 0x3F) == 0 => lhs_simplified,
                _ => irotr64(lhs_simplified, rhs_simplified),
            }
        }

        // Comparison optimizations (i32)
        ValueData::I32Eq { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() < r.value() { 1 } else { 0 }))
                }
                _ => ilts32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32LtU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() > r.value() { 1 } else { 0 }))
                }
                _ => igts32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32GtU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() <= r.value() { 1 } else { 0 }))
                }
                _ => iles32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32LeU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r }) => {
                    iconst32(Imm32(if l.value() >= r.value() { 1 } else { 0 }))
                }
                _ => iges32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32GeU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() < r.value() { 1 } else { 0 }))
                }
                _ => ilts64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64LtU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() > r.value() { 1 } else { 0 }))
                }
                _ => igts64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64GtU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() <= r.value() { 1 } else { 0 }))
                }
                _ => iles64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64LeU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r }) => {
                    iconst32(Imm32(if l.value() >= r.value() { 1 } else { 0 }))
                }
                _ => iges64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64GeU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
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
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                // #273: only fold when it CANNOT trap. div_s traps on divisor 0
                // (guarded) AND on signed overflow INT_MIN / -1 — folding that to a
                // constant would silently drop the mandatory `integer overflow` trap,
                // so we DO NOT fold it; the op stays and traps correctly at runtime.
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 && !(l.value() == i32::MIN && r.value() == -1) =>
                {
                    iconst32(Imm32(l.value() / r.value()))
                }
                _ => idivs32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32DivU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding: const / const
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst32(Imm32(((l.value() as u32) / (r.value() as u32)) as i32))
                }
                // Strength reduction: x / power_of_2 → x >> log2(power_of_2)
                (_, ValueData::I32Const { val: rhs_val })
                    if is_power_of_two_i32(*rhs_val).is_some() =>
                {
                    let shift_amount = log2_i32(*rhs_val);
                    ishru32(lhs_simplified, iconst32(shift_amount))
                }
                // Algebraic: x / 1 = x
                (_, ValueData::I32Const { val }) if val.value() == 1 => lhs_simplified,
                _ => idivu32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32RemS { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                // #273: rem_s traps on divisor 0 (guarded). INT_MIN % -1 does NOT
                // trap in wasm (result 0), but we still avoid `wrapping_rem`'s panic
                // path and keep the fold sound for that case explicitly.
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 =>
                {
                    let folded = if l.value() == i32::MIN && r.value() == -1 {
                        0
                    } else {
                        l.value() % r.value()
                    };
                    iconst32(Imm32(folded))
                }
                _ => irems32(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I32RemU { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding: const % const
                (ValueData::I32Const { val: l }, ValueData::I32Const { val: r })
                    if r.value() != 0 =>
                {
                    iconst32(Imm32(((l.value() as u32) % (r.value() as u32)) as i32))
                }
                // Strength reduction: x % power_of_2 → x & (power_of_2 - 1)
                (_, ValueData::I32Const { val: rhs_val })
                    if is_power_of_two_i32(*rhs_val).is_some() =>
                {
                    let mask = imm32_sub_1(*rhs_val);
                    iand32(lhs_simplified, iconst32(mask))
                }
                _ => iremu32(lhs_simplified, rhs_simplified),
            }
        }

        // Division and remainder optimizations (i64) - constant folding only, avoid division by zero
        ValueData::I64DivS { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                // #273: only fold when it CANNOT trap. div_s traps on divisor 0
                // (guarded) AND on signed overflow INT_MIN / -1 — folding that to a
                // constant would silently drop the mandatory `integer overflow` trap,
                // so we DO NOT fold it; the op stays and traps correctly at runtime.
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r })
                    if r.value() != 0 && !(l.value() == i64::MIN && r.value() == -1) =>
                {
                    iconst64(Imm64(l.value() / r.value()))
                }
                _ => idivs64(lhs_simplified, rhs_simplified),
            }
        }

        ValueData::I64RemS { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                // #273: rem_s traps on divisor 0 (guarded). INT_MIN % -1 does NOT
                // trap in wasm (result 0), but we still avoid `wrapping_rem`'s panic
                // path and keep the fold sound for that case explicitly.
                (ValueData::I64Const { val: l }, ValueData::I64Const { val: r })
                    if r.value() != 0 =>
                {
                    let folded = if l.value() == i64::MIN && r.value() == -1 {
                        0
                    } else {
                        l.value() % r.value()
                    };
                    iconst64(Imm64(folded))
                }
                _ => irems64(lhs_simplified, rhs_simplified),
            }
        }

        // Unary operations (i32) optimizations
        ValueData::I32Eqz { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.eqz (i32.const N)) → (i32.const (N == 0 ? 1 : 0))
                ValueData::I32Const { val: v } => {
                    iconst32(Imm32(if v.value() == 0 { 1 } else { 0 }))
                }
                _ => ieqz32(val_simplified),
            }
        }

        ValueData::I32Clz { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.clz (i32.const N)) → (i32.const count_leading_zeros(N))
                ValueData::I32Const { val: v } => iconst32(Imm32(v.value().leading_zeros() as i32)),
                _ => iclz32(val_simplified),
            }
        }

        ValueData::I32Ctz { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.ctz (i32.const N)) → (i32.const count_trailing_zeros(N))
                ValueData::I32Const { val: v } => {
                    iconst32(Imm32(v.value().trailing_zeros() as i32))
                }
                _ => ictz32(val_simplified),
            }
        }

        ValueData::I32Popcnt { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.popcnt (i32.const N)) → (i32.const count_ones(N))
                ValueData::I32Const { val: v } => iconst32(Imm32(v.value().count_ones() as i32)),
                _ => ipopcnt32(val_simplified),
            }
        }

        // Unary operations (i64) optimizations
        ValueData::I64Eqz { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.eqz (i64.const N)) → (i32.const (N == 0 ? 1 : 0))
                ValueData::I64Const { val: v } => {
                    iconst32(Imm32(if v.value() == 0 { 1 } else { 0 }))
                }
                _ => ieqz64(val_simplified),
            }
        }

        ValueData::I64Clz { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.clz (i64.const N)) → (i64.const count_leading_zeros(N))
                ValueData::I64Const { val: v } => iconst64(Imm64(v.value().leading_zeros() as i64)),
                _ => iclz64(val_simplified),
            }
        }

        ValueData::I64Ctz { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.ctz (i64.const N)) → (i64.const count_trailing_zeros(N))
                ValueData::I64Const { val: v } => {
                    iconst64(Imm64(v.value().trailing_zeros() as i64))
                }
                _ => ictz64(val_simplified),
            }
        }

        ValueData::I64Popcnt { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.popcnt (i64.const N)) → (i64.const count_ones(N))
                ValueData::I64Const { val: v } => iconst64(Imm64(v.value().count_ones() as i64)),
                _ => ipopcnt64(val_simplified),
            }
        }

        // Integer width conversions (#219 seam-SROA). The live rewrite for the
        // decide seam's u64 pack/unpack round-trip. Z3-validated per pass.
        ValueData::I32WrapI64 { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // #219 seam-SROA: wrap_i64(extend_i32_u(x)) = x. zero-extending an
                // i32 to i64 then taking the low 32 bits is the identity on i32 —
                // unconditionally sound (Z3: extract(31,0, zero_ext(32,x)) = x).
                ValueData::I64ExtendI32U { val: inner } => inner.clone(),
                // Constant folding: wrap_i64(i64.const N) → i32.const (low 32 bits).
                ValueData::I64Const { val: v } => iconst32(Imm32(v.value() as i32)),
                _ => i32_wrap_i64(val_simplified),
            }
        }

        ValueData::I64ExtendI32U { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: extend_i32_u(i32.const N) → i64.const (zero-extended).
                ValueData::I32Const { val: v } => iconst64(Imm64((v.value() as u32 as u64) as i64)),
                _ => i64_extend_i32_u(val_simplified),
            }
        }

        // Select instruction optimization
        ValueData::Select {
            cond,
            true_val,
            false_val,
        } => {
            let cond_simplified = rewrite_pure(cond.clone());
            let true_simplified = rewrite_pure(true_val.clone());
            let false_simplified = rewrite_pure(false_val.clone());

            // Algebraic: select(c, x, x) → x (both branches same).
            // #278: WASM `select` evaluates ALL three operands, so the DISCARDED
            // condition `c` must be trap-free to drop it.
            if are_values_equal(&true_simplified, &false_simplified)
                && is_no_trap_expr(&cond_simplified)
            {
                return true_simplified;
            }

            match cond_simplified.data() {
                // Constant folding: (select (i32.const 0) true false) → false.
                // #278: the untaken `true` arm is DISCARDED but WASM still
                // evaluates it, so it must be trap-free to drop it.
                ValueData::I32Const { val }
                    if val.value() == 0 && is_no_trap_expr(&true_simplified) =>
                {
                    false_simplified
                }
                // Constant folding: (select (i32.const N) true false) → true
                // (N != 0). #278: the untaken `false` arm is DISCARDED but still
                // evaluated by WASM, so it must be trap-free to drop it.
                ValueData::I32Const { .. } if is_no_trap_expr(&false_simplified) => true_simplified,
                _ => select_instr(cond_simplified, true_simplified, false_simplified),
            }
        }

        // Sign extension optimizations (i32)
        ValueData::I32Extend8S { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.extend8_s (i32.const N)) → (i32.const sign_extend_8(N))
                ValueData::I32Const { val: v } => {
                    // Sign-extend low 8 bits to 32 bits
                    let low8 = v.value() as i8;
                    iconst32(Imm32(low8 as i32))
                }
                _ => i32_extend8_s(val_simplified),
            }
        }

        ValueData::I32Extend16S { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i32.extend16_s (i32.const N)) → (i32.const sign_extend_16(N))
                ValueData::I32Const { val: v } => {
                    // Sign-extend low 16 bits to 32 bits
                    let low16 = v.value() as i16;
                    iconst32(Imm32(low16 as i32))
                }
                _ => i32_extend16_s(val_simplified),
            }
        }

        // Sign extension optimizations (i64)
        ValueData::I64Extend8S { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.extend8_s (i64.const N)) → (i64.const sign_extend_8(N))
                ValueData::I64Const { val: v } => {
                    // Sign-extend low 8 bits to 64 bits
                    let low8 = v.value() as i8;
                    iconst64(Imm64(low8 as i64))
                }
                _ => i64_extend8_s(val_simplified),
            }
        }

        ValueData::I64Extend16S { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.extend16_s (i64.const N)) → (i64.const sign_extend_16(N))
                ValueData::I64Const { val: v } => {
                    // Sign-extend low 16 bits to 64 bits
                    let low16 = v.value() as i16;
                    iconst64(Imm64(low16 as i64))
                }
                _ => i64_extend16_s(val_simplified),
            }
        }

        ValueData::I64Extend32S { val } => {
            let val_simplified = rewrite_pure(val.clone());
            match val_simplified.data() {
                // Constant folding: (i64.extend32_s (i64.const N)) → (i64.const sign_extend_32(N))
                ValueData::I64Const { val: v } => {
                    // Sign-extend low 32 bits to 64 bits
                    let low32 = v.value() as i32;
                    iconst64(Imm64(low32 as i64))
                }
                _ => i64_extend32_s(val_simplified),
            }
        }

        // ====================================================================
        // Floating-Point Optimizations
        // ====================================================================
        // IMPORTANT: Float optimizations must be careful with NaN semantics.
        // - NaN propagates through arithmetic operations
        // - NaN != NaN (reflexivity does not hold for equality)
        // - Operations with NaN inputs produce NaN outputs
        // - We only fold constants when BOTH operands are not NaN
        // - When the result IS NaN (e.g., 0.0/0.0, inf-inf), we canonicalize
        //   to WebAssembly canonical NaN (f32: 0x7fc00000, f64: 0x7ff8000000000000)
        // - When the result is subnormal, we conservatively skip folding because
        //   host FPU may flush subnormals to zero (ARM FTZ), while WebAssembly
        //   requires IEEE 754 gradual underflow
        // - x + 0.0 = x and x * 1.0 = x hold for all x including NaN
        // - Division by zero produces infinity (not a trap in WebAssembly)

        // f32.add optimizations
        ValueData::F32Add { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                // Constant folding when neither operand is NaN
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv + rv;
                        if result.is_nan() {
                            fconst32(ImmF32(F32_CANONICAL_NAN))
                        } else if is_f32_subnormal(result) {
                            fadd32(lhs_simplified, rhs_simplified)
                        } else {
                            fconst32(ImmF32::new(result))
                        }
                    } else {
                        fadd32(lhs_simplified, rhs_simplified)
                    }
                }
                // Algebraic: x + 0.0 = x (holds for all x including NaN)
                (_, ValueData::F32Const { val })
                    if val.value() == 0.0 && !val.value().is_sign_negative() =>
                {
                    lhs_simplified
                }
                (ValueData::F32Const { val }, _)
                    if val.value() == 0.0 && !val.value().is_sign_negative() =>
                {
                    rhs_simplified
                }
                _ => fadd32(lhs_simplified, rhs_simplified),
            }
        }

        // f32.sub optimizations
        ValueData::F32Sub { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv - rv;
                        if result.is_nan() {
                            fconst32(ImmF32(F32_CANONICAL_NAN))
                        } else if is_f32_subnormal(result) {
                            fsub32(lhs_simplified, rhs_simplified)
                        } else {
                            fconst32(ImmF32::new(result))
                        }
                    } else {
                        fsub32(lhs_simplified, rhs_simplified)
                    }
                }
                // Algebraic: x - 0.0 = x
                (_, ValueData::F32Const { val })
                    if val.value() == 0.0 && !val.value().is_sign_negative() =>
                {
                    lhs_simplified
                }
                _ => fsub32(lhs_simplified, rhs_simplified),
            }
        }

        // f32.mul optimizations
        ValueData::F32Mul { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv * rv;
                        if result.is_nan() {
                            fconst32(ImmF32(F32_CANONICAL_NAN))
                        } else if is_f32_subnormal(result) {
                            fmul32(lhs_simplified, rhs_simplified)
                        } else {
                            fconst32(ImmF32::new(result))
                        }
                    } else {
                        fmul32(lhs_simplified, rhs_simplified)
                    }
                }
                // Algebraic: x * 1.0 = x (holds for all x including NaN)
                (_, ValueData::F32Const { val }) if val.value() == 1.0 => lhs_simplified,
                (ValueData::F32Const { val }, _) if val.value() == 1.0 => rhs_simplified,
                _ => fmul32(lhs_simplified, rhs_simplified),
            }
        }

        // f32.div optimizations
        ValueData::F32Div { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv / rv;
                        if result.is_nan() {
                            fconst32(ImmF32(F32_CANONICAL_NAN))
                        } else if is_f32_subnormal(result) {
                            fdiv32(lhs_simplified, rhs_simplified)
                        } else {
                            fconst32(ImmF32::new(result))
                        }
                    } else {
                        fdiv32(lhs_simplified, rhs_simplified)
                    }
                }
                // Algebraic: x / 1.0 = x
                (_, ValueData::F32Const { val }) if val.value() == 1.0 => lhs_simplified,
                _ => fdiv32(lhs_simplified, rhs_simplified),
            }
        }

        // f64.add optimizations
        ValueData::F64Add { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv + rv;
                        if result.is_nan() {
                            fconst64(ImmF64(F64_CANONICAL_NAN))
                        } else if is_f64_subnormal(result) {
                            fadd64(lhs_simplified, rhs_simplified)
                        } else {
                            fconst64(ImmF64::new(result))
                        }
                    } else {
                        fadd64(lhs_simplified, rhs_simplified)
                    }
                }
                (_, ValueData::F64Const { val })
                    if val.value() == 0.0 && !val.value().is_sign_negative() =>
                {
                    lhs_simplified
                }
                (ValueData::F64Const { val }, _)
                    if val.value() == 0.0 && !val.value().is_sign_negative() =>
                {
                    rhs_simplified
                }
                _ => fadd64(lhs_simplified, rhs_simplified),
            }
        }

        // f64.sub optimizations
        ValueData::F64Sub { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv - rv;
                        if result.is_nan() {
                            fconst64(ImmF64(F64_CANONICAL_NAN))
                        } else if is_f64_subnormal(result) {
                            fsub64(lhs_simplified, rhs_simplified)
                        } else {
                            fconst64(ImmF64::new(result))
                        }
                    } else {
                        fsub64(lhs_simplified, rhs_simplified)
                    }
                }
                (_, ValueData::F64Const { val })
                    if val.value() == 0.0 && !val.value().is_sign_negative() =>
                {
                    lhs_simplified
                }
                _ => fsub64(lhs_simplified, rhs_simplified),
            }
        }

        // f64.mul optimizations
        ValueData::F64Mul { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv * rv;
                        if result.is_nan() {
                            fconst64(ImmF64(F64_CANONICAL_NAN))
                        } else if is_f64_subnormal(result) {
                            fmul64(lhs_simplified, rhs_simplified)
                        } else {
                            fconst64(ImmF64::new(result))
                        }
                    } else {
                        fmul64(lhs_simplified, rhs_simplified)
                    }
                }
                (_, ValueData::F64Const { val }) if val.value() == 1.0 => lhs_simplified,
                (ValueData::F64Const { val }, _) if val.value() == 1.0 => rhs_simplified,
                _ => fmul64(lhs_simplified, rhs_simplified),
            }
        }

        // f64.div optimizations
        ValueData::F64Div { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());

            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    let lv = l.value();
                    let rv = r.value();
                    if !lv.is_nan() && !rv.is_nan() {
                        let result = lv / rv;
                        if result.is_nan() {
                            fconst64(ImmF64(F64_CANONICAL_NAN))
                        } else if is_f64_subnormal(result) {
                            fdiv64(lhs_simplified, rhs_simplified)
                        } else {
                            fconst64(ImmF64::new(result))
                        }
                    } else {
                        fdiv64(lhs_simplified, rhs_simplified)
                    }
                }
                (_, ValueData::F64Const { val }) if val.value() == 1.0 => lhs_simplified,
                _ => fdiv64(lhs_simplified, rhs_simplified),
            }
        }

        // f32.abs optimizations — always safe (clears sign bit, well-defined for NaN)
        ValueData::F32Abs { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F32Const { val } => fconst32(ImmF32::new(val.value().abs())),
                ValueData::F32Abs { .. } => simplified,
                ValueData::F32Neg { val: inner2 } => fabs32(rewrite_pure(inner2.clone())),
                _ => fabs32(simplified),
            }
        }

        // f32.neg optimizations — always safe (flips sign bit, well-defined for NaN)
        ValueData::F32Neg { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F32Const { val } => fconst32(ImmF32::new(-val.value())),
                ValueData::F32Neg { val: inner2 } => rewrite_pure(inner2.clone()),
                _ => fneg32(simplified),
            }
        }

        // f32.ceil optimizations
        ValueData::F32Ceil { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F32Const { val } if !val.value().is_nan() => {
                    fconst32(ImmF32::new(val.value().ceil()))
                }
                _ => fceil32(simplified),
            }
        }

        // f32.floor optimizations
        ValueData::F32Floor { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F32Const { val } if !val.value().is_nan() => {
                    fconst32(ImmF32::new(val.value().floor()))
                }
                _ => ffloor32(simplified),
            }
        }

        // f32.trunc optimizations
        ValueData::F32Trunc { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F32Const { val } if !val.value().is_nan() => {
                    fconst32(ImmF32::new(val.value().trunc()))
                }
                _ => ftrunc32(simplified),
            }
        }

        // f32.nearest optimizations (round ties to even)
        ValueData::F32Nearest { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F32Const { val } if !val.value().is_nan() => {
                    fconst32(ImmF32::new(val.value().round_ties_even()))
                }
                _ => fnearest32(simplified),
            }
        }

        // f32.sqrt optimizations
        ValueData::F32Sqrt { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F32Const { val } if !val.value().is_nan() => {
                    fconst32(ImmF32::new(val.value().sqrt()))
                }
                _ => fsqrt32(simplified),
            }
        }

        // f32.min optimizations — NaN propagation: fold only when both non-NaN
        ValueData::F32Min { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r })
                    if !l.value().is_nan() && !r.value().is_nan() =>
                {
                    fconst32(ImmF32::new(l.value().min(r.value())))
                }
                _ => fmin32(lhs_simplified, rhs_simplified),
            }
        }

        // f32.max optimizations — NaN propagation: fold only when both non-NaN
        ValueData::F32Max { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r })
                    if !l.value().is_nan() && !r.value().is_nan() =>
                {
                    fconst32(ImmF32::new(l.value().max(r.value())))
                }
                _ => fmax32(lhs_simplified, rhs_simplified),
            }
        }

        // f32.copysign optimizations — always safe (pure bit manipulation)
        ValueData::F32Copysign { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    fconst32(ImmF32::new(l.value().copysign(r.value())))
                }
                _ => fcopysign32(lhs_simplified, rhs_simplified),
            }
        }

        // f32 comparison optimizations — Rust IEEE 754 semantics match WASM
        ValueData::F32Eq { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() == r.value() { 1 } else { 0 }))
                }
                _ => feq32(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F32Ne { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() != r.value() { 1 } else { 0 }))
                }
                _ => fne32(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F32Lt { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() < r.value() { 1 } else { 0 }))
                }
                _ => flt32(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F32Gt { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() > r.value() { 1 } else { 0 }))
                }
                _ => fgt32(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F32Le { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() <= r.value() { 1 } else { 0 }))
                }
                _ => fle32(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F32Ge { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F32Const { val: l }, ValueData::F32Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() >= r.value() { 1 } else { 0 }))
                }
                _ => fge32(lhs_simplified, rhs_simplified),
            }
        }

        // f64.abs optimizations — always safe
        ValueData::F64Abs { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F64Const { val } => fconst64(ImmF64::new(val.value().abs())),
                ValueData::F64Abs { .. } => simplified,
                ValueData::F64Neg { val: inner2 } => fabs64(rewrite_pure(inner2.clone())),
                _ => fabs64(simplified),
            }
        }

        // f64.neg optimizations — always safe
        ValueData::F64Neg { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F64Const { val } => fconst64(ImmF64::new(-val.value())),
                ValueData::F64Neg { val: inner2 } => rewrite_pure(inner2.clone()),
                _ => fneg64(simplified),
            }
        }

        // f64.ceil optimizations
        ValueData::F64Ceil { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F64Const { val } if !val.value().is_nan() => {
                    fconst64(ImmF64::new(val.value().ceil()))
                }
                _ => fceil64(simplified),
            }
        }

        // f64.floor optimizations
        ValueData::F64Floor { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F64Const { val } if !val.value().is_nan() => {
                    fconst64(ImmF64::new(val.value().floor()))
                }
                _ => ffloor64(simplified),
            }
        }

        // f64.trunc optimizations
        ValueData::F64Trunc { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F64Const { val } if !val.value().is_nan() => {
                    fconst64(ImmF64::new(val.value().trunc()))
                }
                _ => ftrunc64(simplified),
            }
        }

        // f64.nearest optimizations (round ties to even)
        ValueData::F64Nearest { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F64Const { val } if !val.value().is_nan() => {
                    fconst64(ImmF64::new(val.value().round_ties_even()))
                }
                _ => fnearest64(simplified),
            }
        }

        // f64.sqrt optimizations
        ValueData::F64Sqrt { val: inner } => {
            let simplified = rewrite_pure(inner.clone());
            match simplified.data() {
                ValueData::F64Const { val } if !val.value().is_nan() => {
                    fconst64(ImmF64::new(val.value().sqrt()))
                }
                _ => fsqrt64(simplified),
            }
        }

        // f64.min optimizations — NaN propagation: fold only when both non-NaN
        ValueData::F64Min { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r })
                    if !l.value().is_nan() && !r.value().is_nan() =>
                {
                    fconst64(ImmF64::new(l.value().min(r.value())))
                }
                _ => fmin64(lhs_simplified, rhs_simplified),
            }
        }

        // f64.max optimizations — NaN propagation: fold only when both non-NaN
        ValueData::F64Max { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r })
                    if !l.value().is_nan() && !r.value().is_nan() =>
                {
                    fconst64(ImmF64::new(l.value().max(r.value())))
                }
                _ => fmax64(lhs_simplified, rhs_simplified),
            }
        }

        // f64.copysign optimizations — always safe
        ValueData::F64Copysign { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    fconst64(ImmF64::new(l.value().copysign(r.value())))
                }
                _ => fcopysign64(lhs_simplified, rhs_simplified),
            }
        }

        // f64 comparison optimizations — Rust IEEE 754 semantics match WASM
        ValueData::F64Eq { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() == r.value() { 1 } else { 0 }))
                }
                _ => feq64(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F64Ne { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() != r.value() { 1 } else { 0 }))
                }
                _ => fne64(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F64Lt { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() < r.value() { 1 } else { 0 }))
                }
                _ => flt64(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F64Gt { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() > r.value() { 1 } else { 0 }))
                }
                _ => fgt64(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F64Le { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() <= r.value() { 1 } else { 0 }))
                }
                _ => fle64(lhs_simplified, rhs_simplified),
            }
        }
        ValueData::F64Ge { lhs, rhs } => {
            let lhs_simplified = rewrite_pure(lhs.clone());
            let rhs_simplified = rewrite_pure(rhs.clone());
            match (lhs_simplified.data(), rhs_simplified.data()) {
                (ValueData::F64Const { val: l }, ValueData::F64Const { val: r }) => {
                    iconst32(Imm32::new(if l.value() >= r.value() { 1 } else { 0 }))
                }
                _ => fge64(lhs_simplified, rhs_simplified),
            }
        }

        // ====================================================================
        // Float-to-Integer Truncation (trapping) — only fold when in-range
        // WASM traps on NaN or out-of-range; we cannot represent traps, so
        // we only fold when the conversion would succeed.
        // ====================================================================
        ValueData::I32TruncF32S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= i32::MIN as f32 && f < (i32::MAX as f32 + 1.0) {
                    iconst32(Imm32::new(f as i32))
                } else {
                    i32_trunc_f32_s(v)
                }
            } else {
                i32_trunc_f32_s(v)
            }
        }
        ValueData::I32TruncF32U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= 0.0 && f < (u32::MAX as f32 + 1.0) {
                    iconst32(Imm32::new(f as u32 as i32))
                } else {
                    i32_trunc_f32_u(v)
                }
            } else {
                i32_trunc_f32_u(v)
            }
        }
        ValueData::I32TruncF64S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= i32::MIN as f64 && f < (i32::MAX as f64 + 1.0) {
                    iconst32(Imm32::new(f as i32))
                } else {
                    i32_trunc_f64_s(v)
                }
            } else {
                i32_trunc_f64_s(v)
            }
        }
        ValueData::I32TruncF64U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= 0.0 && f < (u32::MAX as f64 + 1.0) {
                    iconst32(Imm32::new(f as u32 as i32))
                } else {
                    i32_trunc_f64_u(v)
                }
            } else {
                i32_trunc_f64_u(v)
            }
        }
        ValueData::I64TruncF32S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= i64::MIN as f32 && f < (i64::MAX as f32) {
                    iconst64(Imm64::new(f as i64))
                } else {
                    i64_trunc_f32_s(v)
                }
            } else {
                i64_trunc_f32_s(v)
            }
        }
        ValueData::I64TruncF32U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= 0.0 && f < (u64::MAX as f32) {
                    iconst64(Imm64::new(f as u64 as i64))
                } else {
                    i64_trunc_f32_u(v)
                }
            } else {
                i64_trunc_f32_u(v)
            }
        }
        ValueData::I64TruncF64S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= i64::MIN as f64 && f < (i64::MAX as f64) {
                    iconst64(Imm64::new(f as i64))
                } else {
                    i64_trunc_f64_s(v)
                }
            } else {
                i64_trunc_f64_s(v)
            }
        }
        ValueData::I64TruncF64U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                if !f.is_nan() && f >= 0.0 && f < (u64::MAX as f64) {
                    iconst64(Imm64::new(f as u64 as i64))
                } else {
                    i64_trunc_f64_u(v)
                }
            } else {
                i64_trunc_f64_u(v)
            }
        }

        // ====================================================================
        // Integer-to-Float Conversion — always safe to fold
        // ====================================================================
        ValueData::F32ConvertI32S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I32Const { val: c } = v.data() {
                fconst32(ImmF32::new(c.value() as f32))
            } else {
                f32_convert_i32_s(v)
            }
        }
        ValueData::F32ConvertI32U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I32Const { val: c } = v.data() {
                fconst32(ImmF32::new(c.value() as u32 as f32))
            } else {
                f32_convert_i32_u(v)
            }
        }
        ValueData::F32ConvertI64S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I64Const { val: c } = v.data() {
                fconst32(ImmF32::new(c.value() as f32))
            } else {
                f32_convert_i64_s(v)
            }
        }
        ValueData::F32ConvertI64U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I64Const { val: c } = v.data() {
                fconst32(ImmF32::new(c.value() as u64 as f32))
            } else {
                f32_convert_i64_u(v)
            }
        }
        ValueData::F64ConvertI32S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I32Const { val: c } = v.data() {
                fconst64(ImmF64::new(c.value() as f64))
            } else {
                f64_convert_i32_s(v)
            }
        }
        ValueData::F64ConvertI32U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I32Const { val: c } = v.data() {
                fconst64(ImmF64::new(c.value() as u32 as f64))
            } else {
                f64_convert_i32_u(v)
            }
        }
        ValueData::F64ConvertI64S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I64Const { val: c } = v.data() {
                fconst64(ImmF64::new(c.value() as f64))
            } else {
                f64_convert_i64_s(v)
            }
        }
        ValueData::F64ConvertI64U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I64Const { val: c } = v.data() {
                fconst64(ImmF64::new(c.value() as u64 as f64))
            } else {
                f64_convert_i64_u(v)
            }
        }

        // ====================================================================
        // Float Demote/Promote — always safe to fold
        // ====================================================================
        ValueData::F32DemoteF64 { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                fconst32(ImmF32::new(c.value() as f32))
            } else {
                f32_demote_f64(v)
            }
        }
        ValueData::F64PromoteF32 { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                fconst64(ImmF64::new(c.value() as f64))
            } else {
                f64_promote_f32(v)
            }
        }

        // ====================================================================
        // Reinterpret (bit-cast) — always safe, pure bit reinterpretation
        // ====================================================================
        ValueData::I32ReinterpretF32 { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                iconst32(Imm32::new(c.value().to_bits() as i32))
            } else {
                i32_reinterpret_f32(v)
            }
        }
        ValueData::I64ReinterpretF64 { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                iconst64(Imm64::new(c.value().to_bits() as i64))
            } else {
                i64_reinterpret_f64(v)
            }
        }
        ValueData::F32ReinterpretI32 { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I32Const { val: c } = v.data() {
                fconst32(ImmF32::new(f32::from_bits(c.value() as u32)))
            } else {
                f32_reinterpret_i32(v)
            }
        }
        ValueData::F64ReinterpretI64 { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::I64Const { val: c } = v.data() {
                fconst64(ImmF64::new(f64::from_bits(c.value() as u64)))
            } else {
                f64_reinterpret_i64(v)
            }
        }

        // ====================================================================
        // Saturating Truncation — always safe to fold (NaN→0, clamp on overflow)
        // ====================================================================
        ValueData::I32TruncSatF32S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() {
                    0
                } else if f >= (i32::MAX as f32 + 1.0) {
                    i32::MAX
                } else if f < i32::MIN as f32 {
                    i32::MIN
                } else {
                    f as i32
                };
                iconst32(Imm32::new(result))
            } else {
                i32_trunc_sat_f32_s(v)
            }
        }
        ValueData::I32TruncSatF32U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() || f < 0.0 {
                    0u32
                } else if f >= (u32::MAX as f32 + 1.0) {
                    u32::MAX
                } else {
                    f as u32
                };
                iconst32(Imm32::new(result as i32))
            } else {
                i32_trunc_sat_f32_u(v)
            }
        }
        ValueData::I32TruncSatF64S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() {
                    0
                } else if f >= (i32::MAX as f64 + 1.0) {
                    i32::MAX
                } else if f < i32::MIN as f64 {
                    i32::MIN
                } else {
                    f as i32
                };
                iconst32(Imm32::new(result))
            } else {
                i32_trunc_sat_f64_s(v)
            }
        }
        ValueData::I32TruncSatF64U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() || f < 0.0 {
                    0u32
                } else if f >= (u32::MAX as f64 + 1.0) {
                    u32::MAX
                } else {
                    f as u32
                };
                iconst32(Imm32::new(result as i32))
            } else {
                i32_trunc_sat_f64_u(v)
            }
        }
        ValueData::I64TruncSatF32S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() {
                    0i64
                } else if f >= i64::MAX as f32 {
                    i64::MAX
                } else if f < i64::MIN as f32 {
                    i64::MIN
                } else {
                    f as i64
                };
                iconst64(Imm64::new(result))
            } else {
                i64_trunc_sat_f32_s(v)
            }
        }
        ValueData::I64TruncSatF32U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F32Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() || f < 0.0 {
                    0u64
                } else if f >= u64::MAX as f32 {
                    u64::MAX
                } else {
                    f as u64
                };
                iconst64(Imm64::new(result as i64))
            } else {
                i64_trunc_sat_f32_u(v)
            }
        }
        ValueData::I64TruncSatF64S { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() {
                    0i64
                } else if f >= i64::MAX as f64 {
                    i64::MAX
                } else if f < i64::MIN as f64 {
                    i64::MIN
                } else {
                    f as i64
                };
                iconst64(Imm64::new(result))
            } else {
                i64_trunc_sat_f64_s(v)
            }
        }
        ValueData::I64TruncSatF64U { val } => {
            let v = rewrite_pure(val.clone());
            if let ValueData::F64Const { val: c } = v.data() {
                let f = c.value();
                let result = if f.is_nan() || f < 0.0 {
                    0u64
                } else if f >= u64::MAX as f64 {
                    u64::MAX
                } else {
                    f as u64
                };
                iconst64(Imm64::new(result as i64))
            } else {
                i64_trunc_sat_f64_u(v)
            }
        }

        // ====================================================================
        // Memory Size/Grow — side-effectful, cannot fold, simplify children
        // ====================================================================
        ValueData::MemorySize { .. } => val, // no children to simplify
        ValueData::MemoryGrow { val: inner, mem } => {
            let v = rewrite_pure(inner.clone());
            memory_grow(v, *mem)
        }

        // ====================================================================
        // Bulk Memory — side-effectful, cannot fold, simplify children
        // ====================================================================
        ValueData::MemoryFill {
            dst,
            val: v,
            len,
            mem,
        } => memory_fill(
            rewrite_pure(dst.clone()),
            rewrite_pure(v.clone()),
            rewrite_pure(len.clone()),
            *mem,
        ),
        ValueData::MemoryCopy {
            dst,
            src,
            len,
            dst_mem,
            src_mem,
        } => memory_copy(
            rewrite_pure(dst.clone()),
            rewrite_pure(src.clone()),
            rewrite_pure(len.clone()),
            *dst_mem,
            *src_mem,
        ),
        ValueData::MemoryInit {
            dst,
            src,
            len,
            mem,
            data_idx,
        } => memory_init(
            rewrite_pure(dst.clone()),
            rewrite_pure(src.clone()),
            rewrite_pure(len.clone()),
            *mem,
            *data_idx,
        ),
        ValueData::DataDrop { .. } => val, // no children to simplify

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(outer_add);

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
        let simplified = rewrite_pure(term.clone());

        assert_eq!(simplified, term);
    }

    #[test]
    fn test_simplify_overflow() {
        // Test overflow wrapping in constant folding
        let term = iadd32(iconst32(Imm32::from(i32::MAX)), iconst32(Imm32::from(1)));
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(term);

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
        let simplified = rewrite_pure(add);

        match simplified.data() {
            ValueData::I32Const { val } => {
                assert_eq!(val.value(), 20);
            }
            _ => panic!("Expected nested constant folding"),
        }
    }

    #[test]
    fn test_mul_negative_one() {
        // Test: (i32.mul x -1) → (i32.sub 0 x)
        let x = local_get(0);
        let term = imul32(x, iconst32(Imm32::from(-1)));
        let simplified = rewrite_pure(term);

        // Should be (0 - local.get 0)
        match simplified.data() {
            ValueData::I32Sub { lhs, rhs } => {
                match lhs.data() {
                    ValueData::I32Const { val } => assert_eq!(val.value(), 0),
                    _ => panic!("Expected i32.const 0"),
                }
                match rhs.data() {
                    ValueData::LocalGet { idx } => assert_eq!(*idx, 0),
                    _ => panic!("Expected local.get 0"),
                }
            }
            _ => panic!("Expected (i32.sub 0 x) for x * -1"),
        }
    }

    #[test]
    fn test_double_negation_i32() {
        // Test: (i32.sub 0 (i32.sub 0 x)) → x
        let x = local_get(0);
        let neg_x = isub32(iconst32(Imm32::from(0)), x);
        let double_neg = isub32(iconst32(Imm32::from(0)), neg_x);
        let simplified = rewrite_pure(double_neg);

        match simplified.data() {
            ValueData::LocalGet { idx } => assert_eq!(*idx, 0),
            _ => panic!("Expected local.get 0 for double negation"),
        }
    }

    #[test]
    fn test_select_same_branches() {
        // Test: (select cond x x) → x
        let cond = local_get(0);
        let x = iconst32(Imm32::from(42));
        let term = select_instr(cond, x.clone(), x);
        let simplified = rewrite_pure(term);

        match simplified.data() {
            ValueData::I32Const { val } => assert_eq!(val.value(), 42),
            _ => panic!("Expected i32.const 42 for select with same branches"),
        }
    }

    #[test]
    fn test_i64_mul_negative_one() {
        // Test: (i64.mul x -1) → (i64.sub 0 x)
        let x = local_get(0);
        let term = imul64(x, iconst64(Imm64::from(-1)));
        let simplified = rewrite_pure(term);

        // Note: local_get produces i32 conceptually, but the mul expects i64
        // The test verifies the structure of the transformation
        match simplified.data() {
            ValueData::I64Sub { lhs, rhs } => {
                match lhs.data() {
                    ValueData::I64Const { val } => assert_eq!(val.value(), 0),
                    _ => panic!("Expected i64.const 0"),
                }
                match rhs.data() {
                    ValueData::LocalGet { idx } => assert_eq!(*idx, 0),
                    _ => panic!("Expected local.get 0"),
                }
            }
            _ => panic!("Expected (i64.sub 0 x) for x * -1"),
        }
    }

    #[test]
    fn test_cross_memory_store_load_not_redundant() {
        // A store to memory 0 at (addr=0, offset=4) followed by a load from memory 1
        // at the same address should NOT be treated as redundant — they are different memories.
        let mut env = OptimizationEnv::default();

        // Store i32.const 42 to memory 0 at address 0+4
        let store = i32_store(
            iconst32(Imm32::from(0)),
            iconst32(Imm32::from(42)),
            4, // offset
            2, // align
            0, // mem = 0
        );
        let _ = rewrite_with_dataflow(store, &mut env);

        // Load from memory 1 at the same address (0+4)
        let load = i32_load(
            iconst32(Imm32::from(0)),
            4, // offset
            2, // align
            1, // mem = 1 — different memory!
        );
        let result = rewrite_with_dataflow(load, &mut env);

        // The load should NOT be simplified to i32.const 42 — it's a different memory
        match result.data() {
            ValueData::I32Const { val } if val.value() == 42 => {
                panic!("Cross-memory load should NOT be eliminated as redundant");
            }
            ValueData::I32Load { mem, .. } => {
                assert_eq!(*mem, 1, "Load should still reference memory 1");
            }
            _ => panic!("Expected I32Load, got {:?}", result.data()),
        }
    }

    #[test]
    fn test_same_memory_store_load_is_redundant() {
        // Sanity check: a store to memory 0 followed by a load from memory 0
        // at the same address SHOULD be eliminated.
        let mut env = OptimizationEnv::default();

        let store = i32_store(
            iconst32(Imm32::from(0)),
            iconst32(Imm32::from(42)),
            4, // offset
            2, // align
            0, // mem = 0
        );
        let _ = rewrite_with_dataflow(store, &mut env);

        let load = i32_load(
            iconst32(Imm32::from(0)),
            4, // offset
            2, // align
            0, // mem = 0 — same memory
        );
        let result = rewrite_with_dataflow(load, &mut env);

        // The load SHOULD be simplified to i32.const 42
        match result.data() {
            ValueData::I32Const { val } => {
                assert_eq!(
                    val.value(),
                    42,
                    "Same-memory load should return stored value"
                );
            }
            _ => panic!(
                "Expected redundant load to be eliminated, got {:?}",
                result.data()
            ),
        }
    }

    // ========================================================================
    // #240 algebraic mid-end — Tier 1 (unconditional, Z3-verified) rules
    // ========================================================================

    /// (x << 8) >>u 8 -> x & 0x00FF_FFFF  (unconditional masking fold).
    /// This is the structural form of synth's `256*(x-c) >> 8` shape.
    #[test]
    fn test_240_i32_shl_shru_masks() {
        let x = local_get(0);
        let expr = ishru32(
            ishl32(x.clone(), iconst32(Imm32::from(8))),
            iconst32(Imm32::from(8)),
        );
        let out = rewrite_pure(expr);
        match out.data() {
            ValueData::I32And { lhs, rhs } => {
                assert_eq!(lhs.data(), x.data(), "masked operand should be x");
                match rhs.data() {
                    ValueData::I32Const { val } => {
                        assert_eq!(val.value() as u32, 0x00FF_FFFF, "mask = 2^(32-8)-1")
                    }
                    _ => panic!("expected const mask, got {:?}", rhs.data()),
                }
            }
            other => panic!("expected (x & 0xFFFFFF), got {:?}", other),
        }
    }

    /// (x << a) << b -> x << (a+b) when a+b < 32.
    #[test]
    fn test_240_i32_double_shl_collapses() {
        let x = local_get(0);
        let expr = ishl32(
            ishl32(x.clone(), iconst32(Imm32::from(3))),
            iconst32(Imm32::from(4)),
        );
        let out = rewrite_pure(expr);
        match out.data() {
            ValueData::I32Shl { lhs, rhs } => {
                assert_eq!(lhs.data(), x.data());
                match rhs.data() {
                    ValueData::I32Const { val } => assert_eq!(val.value(), 7),
                    _ => panic!("expected const 7"),
                }
            }
            other => panic!("expected x << 7, got {:?}", other),
        }
    }

    /// (x << a) << b does NOT collapse when a+b >= 32 (wasm shift-mod-32 wrap).
    #[test]
    fn test_240_i32_double_shl_no_collapse_across_width() {
        let x = local_get(0);
        let expr = ishl32(
            ishl32(x.clone(), iconst32(Imm32::from(20))),
            iconst32(Imm32::from(20)),
        );
        let out = rewrite_pure(expr);
        // Must remain a nested shift (not fused to << 40 which would mask to << 8).
        match out.data() {
            ValueData::I32Shl { lhs, .. } => {
                assert!(matches!(lhs.data(), ValueData::I32Shl { .. }));
            }
            other => panic!("expected preserved nested shift, got {:?}", other),
        }
    }

    /// (x >>u a) >>u b -> x >>u (a+b) when a+b < 32.
    #[test]
    fn test_240_i32_double_shru_collapses() {
        let x = local_get(0);
        let expr = ishru32(
            ishru32(x.clone(), iconst32(Imm32::from(5))),
            iconst32(Imm32::from(6)),
        );
        let out = rewrite_pure(expr);
        match out.data() {
            ValueData::I32ShrU { lhs, rhs } => {
                assert_eq!(lhs.data(), x.data());
                match rhs.data() {
                    ValueData::I32Const { val } => assert_eq!(val.value(), 11),
                    _ => panic!("expected const 11"),
                }
            }
            other => panic!("expected x >>u 11, got {:?}", other),
        }
    }

    /// (x & c1) & c2 -> x & (c1 & c2); and a redundant re-mask folds to identity.
    #[test]
    fn test_240_i32_and_mask_collapses() {
        let x = local_get(0);
        let expr = iand32(
            iand32(x.clone(), iconst32(Imm32::from(0x00FF_00FF))),
            iconst32(Imm32::from(0x0000_FFFF)),
        );
        let out = rewrite_pure(expr);
        match out.data() {
            ValueData::I32And { lhs, rhs } => {
                assert_eq!(lhs.data(), x.data());
                match rhs.data() {
                    ValueData::I32Const { val } => {
                        assert_eq!(val.value() as u32, 0x0000_00FF)
                    }
                    _ => panic!("expected fused mask"),
                }
            }
            other => panic!("expected x & 0xFF, got {:?}", other),
        }
    }

    /// i64: (x << a) << b -> x << (a+b) when a+b < 64.
    #[test]
    fn test_240_i64_double_shl_collapses() {
        let x = local_get(0);
        let expr = ishl64(
            ishl64(x.clone(), iconst64(Imm64::from(10))),
            iconst64(Imm64::from(20)),
        );
        let out = rewrite_pure(expr);
        match out.data() {
            ValueData::I64Shl { rhs, .. } => match rhs.data() {
                ValueData::I64Const { val } => assert_eq!(val.value(), 30),
                _ => panic!("expected const 30"),
            },
            other => panic!("expected x << 30, got {:?}", other),
        }
    }

    // ========================================================================
    // #240 Tier-2 range-premise hook — GATED flagship (x<<k)>>u k -> x
    // ========================================================================

    /// WITHOUT a premise: falls back to the unconditional mask (no `-> x`).
    #[test]
    fn test_240_gated_no_premise_keeps_mask() {
        let mut env = OptimizationEnv::new();
        let x = local_get(0);
        let expr = ishru32(
            ishl32(x.clone(), iconst32(Imm32::from(8))),
            iconst32(Imm32::from(8)),
        );
        let out = rewrite_with_dataflow(expr, &mut env);
        // No premise -> must NOT be bare x; must stay the safe masked form.
        assert!(
            matches!(out.data(), ValueData::I32And { .. }),
            "without a premise the fold must keep the mask, got {:?}",
            out.data()
        );
    }

    /// WITH a premise (x < 2^24): (x << 8) >>u 8 -> x. The flagship fold.
    #[test]
    fn test_240_gated_with_premise_folds_to_x() {
        let mut env = OptimizationEnv::new();
        let x = local_get(0);
        // Machine-checked fact injected directly (deferred fact-source stands in):
        // x <= 2^24 - 1  ==>  x < 2^24  ==>  high 8 bits are zero.
        env.assume_max(x.clone(), (1u64 << 24) - 1);
        let expr = ishru32(
            ishl32(x.clone(), iconst32(Imm32::from(8))),
            iconst32(Imm32::from(8)),
        );
        let out = rewrite_with_dataflow(expr, &mut env);
        assert_eq!(
            out.data(),
            x.data(),
            "with x < 2^24 the (x<<8)>>u8 round-trip must fold to x"
        );
    }

    /// A too-weak premise (x < 2^25, but 8 high bits needed) must NOT fold to x.
    #[test]
    fn test_240_gated_weak_premise_does_not_fold() {
        let mut env = OptimizationEnv::new();
        let x = local_get(0);
        env.assume_max(x.clone(), (1u64 << 25) - 1); // only proves top 7 bits zero
        let expr = ishru32(
            ishl32(x.clone(), iconst32(Imm32::from(8))),
            iconst32(Imm32::from(8)),
        );
        let out = rewrite_with_dataflow(expr, &mut env);
        assert!(
            !matches!(out.data(), ValueData::LocalGet { .. }),
            "an insufficient bound must not license the fold, got {:?}",
            out.data()
        );
    }

    // -------------------------------------------------------------------------
    // #278: zero-annihilation / absorption folds must NOT discard a TRAPPING
    // operand. A discarded out-of-bounds load / div-by-zero would silently
    // eliminate a mandatory WASM trap, returning a value where the original
    // module traps. Below, an OOB `i32.load`/`i64.load` and a `div` stand in for
    // "may-trap"; a `local.get` stands in for "provably trap-free".
    // -------------------------------------------------------------------------

    /// A load anywhere in the tree that stands for an OOB access.
    fn tree_contains_load(v: &Value) -> bool {
        match v.data() {
            ValueData::I32Load { .. }
            | ValueData::I64Load { .. }
            | ValueData::I32DivS { .. }
            | ValueData::I32DivU { .. }
            | ValueData::I64DivS { .. }
            | ValueData::I64DivU { .. } => true,
            ValueData::I32Mul { lhs, rhs }
            | ValueData::I32And { lhs, rhs }
            | ValueData::I32Or { lhs, rhs }
            | ValueData::I64Mul { lhs, rhs }
            | ValueData::I64And { lhs, rhs }
            | ValueData::I64Or { lhs, rhs }
            | ValueData::I64Shl { lhs, rhs } => tree_contains_load(lhs) || tree_contains_load(rhs),
            ValueData::Select {
                cond,
                true_val,
                false_val,
            } => {
                tree_contains_load(cond)
                    || tree_contains_load(true_val)
                    || tree_contains_load(false_val)
            }
            _ => false,
        }
    }

    fn oob_i32_load() -> Value {
        // i32.load at a huge constant address — OOB in any small memory.
        i32_load(iconst32(Imm32::from(1_000_000)), 0, 2, 0)
    }
    fn oob_i64_load() -> Value {
        i64_load(iconst64(Imm64::from(1_000_000)), 0, 3, 0)
    }
    fn trapping_div() -> Value {
        // (i32.div_u (i32.const 1) (i32.const 0)) — div-by-zero trap.
        idivu32(iconst32(Imm32::from(1)), iconst32(Imm32::from(0)))
    }

    #[test]
    fn test_278_i32_mul_by_zero_keeps_trapping_load() {
        // (i32.mul (i32.load OOB) (i32.const 0)) must NOT fold to 0.
        let a = rewrite_pure(imul32(oob_i32_load(), iconst32(Imm32::from(0))));
        assert!(
            tree_contains_load(&a),
            "x*0 dropped a trapping load: {:?}",
            a.data()
        );
        // Reversed operand order: (i32.mul (i32.const 0) (i32.load OOB)).
        let b = rewrite_pure(imul32(iconst32(Imm32::from(0)), oob_i32_load()));
        assert!(
            tree_contains_load(&b),
            "0*x dropped a trapping load: {:?}",
            b.data()
        );
    }

    #[test]
    fn test_278_i32_mul_by_zero_keeps_trapping_div() {
        let a = rewrite_pure(imul32(trapping_div(), iconst32(Imm32::from(0))));
        assert!(
            tree_contains_load(&a),
            "x*0 dropped a trapping div: {:?}",
            a.data()
        );
    }

    #[test]
    fn test_278_i64_mul_by_zero_keeps_trapping_load() {
        let a = rewrite_pure(imul64(oob_i64_load(), iconst64(Imm64::from(0))));
        assert!(
            tree_contains_load(&a),
            "i64 x*0 dropped a trapping load: {:?}",
            a.data()
        );
        let b = rewrite_pure(imul64(iconst64(Imm64::from(0)), oob_i64_load()));
        assert!(
            tree_contains_load(&b),
            "i64 0*x dropped a trapping load: {:?}",
            b.data()
        );
    }

    #[test]
    fn test_278_i32_and_zero_keeps_trapping_load() {
        let a = rewrite_pure(iand32(oob_i32_load(), iconst32(Imm32::from(0))));
        assert!(
            tree_contains_load(&a),
            "x&0 dropped a trapping load: {:?}",
            a.data()
        );
        let b = rewrite_pure(iand32(iconst32(Imm32::from(0)), oob_i32_load()));
        assert!(
            tree_contains_load(&b),
            "0&x dropped a trapping load: {:?}",
            b.data()
        );
    }

    #[test]
    fn test_278_i64_and_zero_keeps_trapping_load() {
        let a = rewrite_pure(iand64(oob_i64_load(), iconst64(Imm64::from(0))));
        assert!(
            tree_contains_load(&a),
            "i64 x&0 dropped a trapping load: {:?}",
            a.data()
        );
        let b = rewrite_pure(iand64(iconst64(Imm64::from(0)), oob_i64_load()));
        assert!(
            tree_contains_load(&b),
            "i64 0&x dropped a trapping load: {:?}",
            b.data()
        );
    }

    #[test]
    fn test_278_i32_or_allones_keeps_trapping_load() {
        // (i32.or (i32.load OOB) (i32.const -1)) must NOT fold to -1.
        let a = rewrite_pure(ior32(oob_i32_load(), iconst32(Imm32::from(-1))));
        assert!(
            tree_contains_load(&a),
            "x|-1 dropped a trapping load: {:?}",
            a.data()
        );
        let b = rewrite_pure(ior32(iconst32(Imm32::from(-1)), oob_i32_load()));
        assert!(
            tree_contains_load(&b),
            "-1|x dropped a trapping load: {:?}",
            b.data()
        );
    }

    #[test]
    fn test_278_i64_or_allones_keeps_trapping_load() {
        let a = rewrite_pure(ior64(oob_i64_load(), iconst64(Imm64::from(-1))));
        assert!(
            tree_contains_load(&a),
            "i64 x|-1 dropped a trapping load: {:?}",
            a.data()
        );
        let b = rewrite_pure(ior64(iconst64(Imm64::from(-1)), oob_i64_load()));
        assert!(
            tree_contains_load(&b),
            "i64 -1|x dropped a trapping load: {:?}",
            b.data()
        );
    }

    #[test]
    fn test_278_select_const_cond_keeps_trapping_untaken_arm() {
        // select(0, trap, safe) → safe would DROP the trapping `true` arm.
        let a = rewrite_pure(select_instr(
            iconst32(Imm32::from(0)),
            oob_i32_load(),
            iconst32(Imm32::from(7)),
        ));
        assert!(
            tree_contains_load(&a),
            "select(0,..) dropped trapping true arm: {:?}",
            a.data()
        );
        // select(1, safe, trap) → safe would DROP the trapping `false` arm.
        let b = rewrite_pure(select_instr(
            iconst32(Imm32::from(1)),
            iconst32(Imm32::from(7)),
            oob_i32_load(),
        ));
        assert!(
            tree_contains_load(&b),
            "select(1,..) dropped trapping false arm: {:?}",
            b.data()
        );
    }

    #[test]
    fn test_278_select_same_arms_keeps_trapping_cond() {
        // select(trap_cond, x, x) → x would DROP the trapping condition.
        let out = rewrite_pure(select_instr(
            oob_i32_load(),
            iconst32(Imm32::from(5)),
            iconst32(Imm32::from(5)),
        ));
        assert!(
            tree_contains_load(&out),
            "select(c,x,x) dropped trapping cond: {:?}",
            out.data()
        );
    }

    // ---- No over-suppression: trap-FREE operands STILL fold ------------------

    #[test]
    fn test_278_trap_free_mul_by_zero_still_folds() {
        // (local.get 0) * 0 → 0 (local.get cannot trap).
        let out = rewrite_pure(imul32(local_get(0), iconst32(Imm32::from(0))));
        assert!(
            matches!(out.data(), ValueData::I32Const { val } if val.value() == 0),
            "trap-free x*0 must still fold to 0, got {:?}",
            out.data()
        );
        let out2 = rewrite_pure(imul32(iconst32(Imm32::from(0)), local_get(0)));
        assert!(
            matches!(out2.data(), ValueData::I32Const { val } if val.value() == 0),
            "trap-free 0*x must still fold to 0, got {:?}",
            out2.data()
        );
    }

    #[test]
    fn test_278_trap_free_and_zero_still_folds() {
        let out = rewrite_pure(iand32(local_get(0), iconst32(Imm32::from(0))));
        assert!(
            matches!(out.data(), ValueData::I32Const { val } if val.value() == 0),
            "trap-free x&0 must still fold to 0, got {:?}",
            out.data()
        );
    }

    #[test]
    fn test_278_trap_free_or_allones_still_folds() {
        let out = rewrite_pure(ior32(local_get(0), iconst32(Imm32::from(-1))));
        assert!(
            matches!(out.data(), ValueData::I32Const { val } if val.value() == -1),
            "trap-free x|-1 must still fold to -1, got {:?}",
            out.data()
        );
    }

    #[test]
    fn test_278_trap_free_i64_still_folds() {
        let m = rewrite_pure(imul64(local_get(0), iconst64(Imm64::from(0))));
        assert!(
            matches!(m.data(), ValueData::I64Const { val } if val.value() == 0),
            "trap-free i64 x*0 must still fold, got {:?}",
            m.data()
        );
        let a = rewrite_pure(iand64(local_get(0), iconst64(Imm64::from(0))));
        assert!(
            matches!(a.data(), ValueData::I64Const { val } if val.value() == 0),
            "trap-free i64 x&0 must still fold, got {:?}",
            a.data()
        );
    }

    #[test]
    fn test_278_trap_free_select_still_folds() {
        // Both arms trap-free constants; select(0,7,9) → 9, select(1,7,9) → 7.
        let f = rewrite_pure(select_instr(
            iconst32(Imm32::from(0)),
            iconst32(Imm32::from(7)),
            iconst32(Imm32::from(9)),
        ));
        assert!(
            matches!(f.data(), ValueData::I32Const { val } if val.value() == 9),
            "trap-free select(0,..) must fold to false arm, got {:?}",
            f.data()
        );
        let t = rewrite_pure(select_instr(
            iconst32(Imm32::from(1)),
            iconst32(Imm32::from(7)),
            iconst32(Imm32::from(9)),
        ));
        assert!(
            matches!(t.data(), ValueData::I32Const { val } if val.value() == 7),
            "trap-free select(1,..) must fold to true arm, got {:?}",
            t.data()
        );
    }
}
