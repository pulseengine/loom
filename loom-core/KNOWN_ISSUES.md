# Known Optimizer Correctness Issues

## Status Summary
- **Total test fixtures**: 25
- **Passing validation**: 25/25 (100%)
- **Known failing**: 0/25 (0%)

All fixture tests now pass with Z3 formal verification!

## Fixed Issues

### ✅ Dead Code Elimination (DCE) - FIXED
- **Commit**: 5228555
- **Issue**: DCE was producing invalid WASM when removing code after terminators
- **Root Cause**:
  - Blocks with result types but unreachable bodies left no value on stack
  - Blocks containing terminators didn't mark following code as unreachable
- **Solution**: Added unreachable instruction injection and proper reachability tracking

### ✅ Enhanced CSE LocalTee Index Bug - FIXED
- **Issue**: Z3 verification failed with "LocalTee index out of bounds"
- **Root Cause**:
  - CSE pass adds new locals to the optimized function
  - TranslationValidator's shared inputs used original function's local count
  - When verifying optimized function, new local indices exceeded shared inputs size
- **Solution**:
  1. Extended locals vector dynamically in `verify.rs` (lines 1680-1712)
  2. Only CSE constants, not LocalGet (which can change between uses)

### ✅ SimplifyLocals Equivalence Bug - FIXED
- **Issue**: Z3 found counterexample for functions with control flow
- **Root Cause**:
  - Equivalence tracking (`dst = src`) didn't account for control flow
  - Sets in one branch of if/block didn't clear equivalences
  - Led to incorrect substitution of `local.get $dst` with `local.get $src`
- **Solution**:
  1. Clear equivalences when locals are set outside of copy patterns
  2. Skip equivalence substitution entirely for functions with control flow
  - Per proof-first philosophy: skip unsafe optimizations

## Architecture Notes

### Z3 Translation Validation
Every optimization pass uses `TranslationValidator` to prove semantic equivalence:
1. Capture original function before optimization
2. Apply optimization
3. Z3 proves `∀ inputs: original(inputs) = optimized(inputs)`

This catches bugs that would be missed by traditional testing.

### Stack Analysis
LOOM now has comprehensive stack validation:
- `ValidationGuard` validates stack correctness after each pass
- `TranslationValidator` proves semantic equivalence with Z3
- Both must pass for an optimization to be accepted

## References

- WASM Spec: https://webassembly.org/specs/core/exec/instructions.html
- Z3 SMT Solver: https://github.com/Z3Prover/z3
- Translation Validation: https://en.wikipedia.org/wiki/Translation_validation
