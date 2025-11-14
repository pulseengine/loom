# Function Inlining Design

## Overview

Function inlining replaces function calls with the function body when beneficial. This:
- **Eliminates call overhead** (critical for small helper functions)
- **Exposes optimization opportunities** across former call boundaries
- **Reduces code size** for single-call-site functions
- **Improves runtime performance** by enabling further optimizations

## Key Concepts

### When to Inline

Based on Binaryen heuristics:

1. **Always Inline** (size ≤ threshold, e.g., 10 instructions)
   - Tiny functions always benefit from inlining
   - Call overhead > function body size

2. **Single Call Site** (refs == 1 && size ≤ threshold, e.g., 50 instructions)
   - Eliminates the function entirely after inlining
   - Net code size reduction

3. **Trivial Instructions** (body is single instruction)
   - `(func $add (param i32 i32) (result i32) (i32.add (local.get 0) (local.get 1)))`
   - Always worth inlining - removes call overhead

4. **Never Inline**
   - Recursive functions (direct or mutual recursion)
   - Very large functions (size > threshold)
   - Functions with multiple call sites AND large bodies

### Example

```wasm
;; Before
(func $double (param $x i32) (result i32)
  (local.get $x)
  (local.get $x)
  (i32.add)
)

(func $test (result i32)
  (i32.const 5)
  (call $double)    ;; Single call site
)

;; After inlining
(func $test (result i32)
  (local $0 i32)    ;; Parameter from $double
  (local.set $0 (i32.const 5))
  (local.get $0)
  (local.get $0)
  (i32.add)
)
;; $double can now be removed (dead function elimination)
```

## Algorithm

### Phase 1: Analysis

```
For each function:
1. Count call sites (how many times it's called)
2. Measure size (instruction count)
3. Detect recursion (direct or mutual)
4. Compute inlineability score

Build call graph to detect recursion:
- Direct: function calls itself
- Mutual: A calls B, B calls A
```

### Phase 2: Inlining

```
For each function (in topological order from leaves):
  For each call instruction:
    If should_inline(callee):
      1. Remap parameters (args → callee params)
      2. Remap locals (renumber to avoid conflicts)
      3. Remap returns (convert to branch or result)
      4. Inline body at call site
      5. Mark function for potential removal
```

### Phase 3: Cleanup

```
- Remove functions with zero call sites
- Re-optimize inlined code (constant folding, DCE)
- Iterate if more inlining opportunities exist
```

## Implementation Details

### Parameter Substitution

When inlining a function, parameters become local variables initialized with call arguments:

```wasm
;; Callee
(func $f (param $p0 i32) (param $p1 i32) (result i32)
  (local.get $p0)
  (local.get $p1)
  (i32.add)
)

;; Call site
(i32.const 10)
(i32.const 20)
(call $f)

;; After inlining (simplified)
(local $inlined_p0 i32)
(local $inlined_p1 i32)
(local.set $inlined_p1 (i32.const 20))  ;; Params in reverse order (stack!)
(local.set $inlined_p0 (i32.const 10))
(local.get $inlined_p0)
(local.get $inlined_p1)
(i32.add)
```

**Note**: WebAssembly stack is LIFO, so arguments must be popped in reverse order!

### Local Variable Remapping

Callee locals must be renumbered to avoid conflicts with caller locals:

```wasm
;; Caller has locals 0, 1
;; Callee has params 0, 1 and local 2

;; After inlining, callee locals become:
;; - param 0 → local 2 (next available in caller)
;; - param 1 → local 3
;; - local 2 → local 4
```

### Return Handling

#### Simple Case: Single Return at End
```wasm
;; Callee
(func $f (result i32)
  (i32.const 42)
)

;; Inlined: Just replace call with body
(i32.const 42)
```

#### Complex Case: Multiple Returns
```wasm
;; Callee with early return
(func $f (param $x i32) (result i32)
  (local.get $x)
  (i32.eqz)
  (if (result i32)
    (then (i32.const 0) (return))  ;; Early return
    (else (nop))
  )
  (i32.const 1)
)

;; Inlined: Use block with branches
(block $inline_exit (result i32)
  (local.get $inlined_x)
  (i32.eqz)
  (if (result i32)
    (then (i32.const 0) (br $inline_exit))  ;; return → br
    (else (nop))
  )
  (i32.const 1)
)
```

### Recursion Detection

Build call graph and detect cycles:

```rust
struct CallGraph {
    calls: HashMap<FuncIdx, Vec<FuncIdx>>,
}

fn detect_recursion(graph: &CallGraph, func: FuncIdx) -> bool {
    let mut visited = HashSet::new();
    let mut stack = vec![func];

    while let Some(current) = stack.pop() {
        if current == func && !visited.is_empty() {
            return true; // Cycle detected
        }
        if visited.insert(current) {
            if let Some(callees) = graph.calls.get(&current) {
                stack.extend(callees);
            }
        }
    }
    false
}
```

## Heuristics (Based on Binaryen)

```rust
struct InliningConfig {
    always_inline_max_size: usize,        // 10 instructions
    one_caller_inline_max_size: usize,    // 50 instructions
    flexible_inline_max_size: usize,      // 100 instructions
    allow_functions_with_loops: bool,     // false (conservative)
}

fn should_inline(
    func: &Function,
    call_sites: usize,
    config: &InliningConfig
) -> bool {
    // Never inline recursive functions
    if is_recursive(func) {
        return false;
    }

    let size = measure_size(func);

    // Always inline tiny functions
    if size <= config.always_inline_max_size {
        return true;
    }

    // Inline single-call-site functions (code size win)
    if call_sites == 1 && size <= config.one_caller_inline_max_size {
        return true;
    }

    // Don't inline large multi-call functions
    if call_sites > 1 && size > config.flexible_inline_max_size {
        return false;
    }

    // Conservative: don't inline functions with loops
    if has_loops(func) && !config.allow_functions_with_loops {
        return false;
    }

    true
}
```

## Implementation Strategy

### Phase 1: Simple Inlining (This Implementation)

Focus on easy cases that always win:
1. **Leaf functions** (don't call other functions)
2. **Single call site** functions
3. **Trivial functions** (1-2 instructions)
4. **No recursion**
5. **Simple return handling** (single return at end)

### Phase 2: Advanced Inlining (Future)

1. Multi-return handling with block wrapping
2. Inline functions that call other functions (bottom-up)
3. Partial inlining (extract cold paths)
4. Profile-guided inlining

## Test Cases

### 1. Simple Leaf Function
```wasm
(func $add (param i32 i32) (result i32)
  (i32.add (local.get 0) (local.get 1))
)
(func $test (result i32)
  (call $add (i32.const 1) (i32.const 2))
)
;; Should inline to: (i32.add (i32.const 1) (i32.const 2))
;; Then constant fold to: (i32.const 3)
```

### 2. Single Call Site
```wasm
(func $helper (param i32) (result i32)
  (local.get 0)
  (i32.const 10)
  (i32.mul)
)
(func $test (result i32)
  (call $helper (i32.const 5))
)
;; Should inline (single call site)
```

### 3. Multiple Call Sites - Don't Inline Large
```wasm
(func $compute (param i32) (result i32)
  ;; Large function body (20+ instructions)
  ...
)
(func $test1 (result i32)
  (call $compute (i32.const 1))
)
(func $test2 (result i32)
  (call $compute (i32.const 2))
)
;; Should NOT inline (multiple calls, large body)
```

### 4. Recursion - Don't Inline
```wasm
(func $factorial (param i32) (result i32)
  (local.get 0)
  (i32.eqz)
  (if (result i32)
    (then (i32.const 1))
    (else
      (local.get 0)
      (i32.const 1)
      (i32.sub)
      (call $factorial)  ;; Recursive!
      (local.get 0)
      (i32.mul)
    )
  )
)
;; Should NOT inline (recursive)
```

### 5. Local Remapping
```wasm
(func $callee (param $x i32) (result i32)
  (local $temp i32)
  (local.set $temp (i32.const 1))
  (i32.add (local.get $x) (local.get $temp))
)
(func $caller (result i32)
  (local $a i32)
  (local.set $a (i32.const 5))
  (call $callee (local.get $a))
)
;; After inlining:
;; (func $caller (result i32)
;;   (local $a i32)
;;   (local $inlined_x i32)      ;; Remapped param
;;   (local $inlined_temp i32)   ;; Remapped local
;;   (local.set $a (i32.const 5))
;;   (local.set $inlined_x (local.get $a))
;;   (local.set $inlined_temp (i32.const 1))
;;   (i32.add (local.get $inlined_x) (local.get $inlined_temp))
;; )
```

## Integration

**Pipeline Position**: After SimplifyLocals, before final optimization round

```
Constant Folding → Branch Simplification → DCE → Block Merging →
Vacuum → SimplifyLocals → Inlining → [Repeat optimizations]
```

**Rationale**:
- Inlining exposes new optimization opportunities
- Should re-run constant folding, DCE, etc. after inlining
- May enable more inlining in next iteration

## Performance Impact

Based on WebAssembly benchmarks:
- **40-50% of code** benefits from inlining
- Particularly effective for:
  - Accessor/getter functions
  - Small math helpers
  - Wrapper functions
  - Single-use helpers

## Success Metrics

- Single-call-site functions inlined
- Tiny functions (<10 instructions) inlined
- No recursive functions inlined
- All inlined code produces valid WASM
- Code size reduction for single-call cases
- Enables further optimization (constant propagation across calls)
