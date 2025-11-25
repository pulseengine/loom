# Fuzzing Infrastructure for LOOM

## Overview

LOOM uses property-based fuzzing with `wasm-smith` to automatically generate random valid WebAssembly modules and test that LOOM's parser, encoder, and optimizer handle them correctly.

## Why Fuzzing?

Fuzzing helps catch:
- **Parse+encode roundtrip bugs** (like Issue #30)
- **Optimization correctness issues**
- **Edge cases** in complex WASM modules
- **Crashes and panics** on unusual inputs

The fuzzer found parse+encode bugs within the first 5 test cases!

## Running Fuzz Tests

### Deterministic Tests (Always Run)

These tests use known, simple WASM modules and always pass:

```bash
cargo test -p loom-core --test fuzz_tests
```

### Property-Based Fuzz Tests (Currently Failing)

These generate random WASM and test properties. Currently marked `#[ignore]` due to known bugs:

```bash
# Run parse+encode roundtrip fuzzing
cargo test -p loom-core --test fuzz_tests prop_parse_encode -- --ignored

# Run optimization idempotence fuzzing
cargo test -p loom-core --test fuzz_tests prop_optimization -- --ignored

# Run all property tests
cargo test -p loom-core --test fuzz_tests -- --ignored
```

## Properties Tested

### 1. Parse+Encode Roundtrip Validity

**Property:** `parse(encode(parse(x))) should produce valid WASM`

**What it catches:**
- Encoder bugs that corrupt the module structure
- Parser bugs that lose information
- IR representation issues

**Status:** ❌ **Currently failing** - reveals Issue #30 bugs

**Example failure:**
```
Parse+encode roundtrip produced invalid WASM
Minimal failing input: [0, 97, 115, 109, 1, 0, 0, 0, ...]
```

### 2. Optimization Idempotence

**Property:** `optimize(optimize(x)) = optimize(x)`

**What it catches:**
- Optimizations that aren't stable
- Passes that modify already-optimized code
- Non-deterministic transformations

**Status:** ⏸️ Depends on roundtrip working

### 3. Optimization Validity Preservation

**Property:** `valid(x) → valid(optimize(x))`

**What it catches:**
- Optimizations that break WASM validity
- Stack discipline violations
- Type errors introduced by transformations

**Status:** ⏸️ Depends on roundtrip working

## Configuration

Fuzzing parameters are in `loom-core/tests/fuzz_tests.rs`:

```rust
proptest! {
    #![proptest_config(ProptestConfig {
        cases: 10,  // Number of random test cases
        failure_persistence: None,
        ..ProptestConfig::default()
    })]
```

**wasm-smith configuration:**

```rust
let mut config = wasm_smith::Config::default();
config.min_funcs = 1;
config.max_funcs = 5;
config.max_instructions = 50;
config.max_memories = 1;
```

Adjust these to:
- **Increase coverage:** More functions, instructions, features
- **Find simpler bugs:** Fewer functions, simpler modules
- **Speed up testing:** Fewer cases

## Roadmap

### Short Term
- ✅ Add fuzzing infrastructure
- ✅ Create property tests
- ✅ Document fuzzing approach
- ⏸️ Fix parse+encode roundtrip bugs (Issue #30)

### Medium Term
- [ ] Increase wasm-smith config complexity (reference types, SIMD)
- [ ] Add semantic equivalence testing with wasmtime
- [ ] Integrate with CI for continuous fuzzing
- [ ] Add coverage-guided fuzzing with cargo-fuzz

### Long Term
- [ ] Differential fuzzing against wasm-opt
- [ ] Minimize failing test cases automatically
- [ ] Add performance regression fuzzing
- [ ] Fuzz individual optimization passes

## Tips for Debugging Failures

### 1. Minimize the Failing Case

When a property test fails, proptest shows the minimal input:

```
minimal failing input: wasm_bytes = [0, 97, 115, 109, ...]
```

Save this to a file and test directly:

```bash
echo "00 61 73 6d 01 00 00 00 ..." | xxd -r -p > failing.wasm
./target/release/loom optimize failing.wasm -o out.wasm
wasm-tools validate out.wasm
```

### 2. Compare Original vs Optimized

```bash
wasm-tools print failing.wasm > original.wat
wasm-tools print out.wasm > optimized.wat
diff -u original.wat optimized.wat
```

### 3. Test Individual Passes

```bash
./target/release/loom optimize failing.wasm -o out.wasm --passes vacuum
./target/release/loom optimize failing.wasm -o out.wasm --passes inline
# etc.
```

### 4. Use wasmparser Error Messages

The validation error often points to the exact issue:

```
error: func 2 failed to validate
Caused by:
    type mismatch: expected i32 but nothing on stack (at offset 0xb541)
```

## Integration with CI

Currently fuzz tests are **not run in CI** because they expose known bugs. Once Issue #30 is fixed:

1. Remove `#[ignore]` attributes
2. Add to CI workflow:

```yaml
- name: Run fuzz tests
  run: cargo test -p loom-core --test fuzz_tests --release
```

3. Consider adding nightly fuzzing job with more cases

## References

- [wasm-smith documentation](https://docs.rs/wasm-smith/)
- [proptest book](https://proptest-rs.github.io/proptest/)
- [Issue #30: Self-optimization bug](https://github.com/pulseengine/loom/issues/30)
- [cargo-fuzz](https://rust-fuzz.github.io/book/cargo-fuzz.html) - for future coverage-guided fuzzing
