# LOOM Comprehensive Testing Framework

**Goal:** Build the most robust WebAssembly optimizer testing infrastructure in existence

---

## Overview

This document describes the **immediate, actionable** testing framework for LOOM. Unlike the high-level roadmap, this focuses on **what to build today**.

---

## Phase 1: Differential Testing (START HERE)

### Why This Matters

Research (DITWO, ISSTA 2023) found **1,293 missed optimizations** in wasm-opt using differential testing. We can use the same approach to:
1. Validate LOOM's correctness
2. Find cases where LOOM beats wasm-opt
3. Identify gaps where wasm-opt is better
4. Build confidence in production use

### Implementation

#### Step 1: Create Testing Infrastructure

```rust
// loom-testing/src/lib.rs
use anyhow::Result;
use std::process::Command;

pub struct DifferentialTester {
    loom_binary: String,
    wasm_opt_binary: String,
}

impl DifferentialTester {
    pub fn new() -> Result<Self> {
        // Find binaries
        let loom = which::which("loom")?;
        let wasm_opt = which::which("wasm-opt")?;

        Ok(Self {
            loom_binary: loom.to_string_lossy().to_string(),
            wasm_opt_binary: wasm_opt.to_string_lossy().to_string(),
        })
    }

    pub fn test(&self, input_wasm: &[u8]) -> Result<TestResult> {
        // Optimize with LOOM
        let loom_output = self.run_loom(input_wasm)?;

        // Optimize with wasm-opt
        let wasm_opt_output = self.run_wasm_opt(input_wasm)?;

        // Compare
        TestResult::compare(input_wasm, &loom_output, &wasm_opt_output)
    }

    fn run_loom(&self, input: &[u8]) -> Result<Vec<u8>> {
        // Write input to temp file
        let temp_in = tempfile::NamedTempFile::new()?;
        std::fs::write(temp_in.path(), input)?;

        let temp_out = tempfile::NamedTempFile::new()?;

        // Run loom
        let output = Command::new(&self.loom_binary)
            .arg("optimize")
            .arg(temp_in.path())
            .arg("-o")
            .arg(temp_out.path())
            .output()?;

        if !output.status.success() {
            anyhow::bail!("LOOM failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        Ok(std::fs::read(temp_out.path())?)
    }

    fn run_wasm_opt(&self, input: &[u8]) -> Result<Vec<u8>> {
        let temp_in = tempfile::NamedTempFile::new()?;
        std::fs::write(temp_in.path(), input)?;

        let temp_out = tempfile::NamedTempFile::new()?;

        // Run wasm-opt with -O3 (highest optimization)
        let output = Command::new(&self.wasm_opt_binary)
            .arg(temp_in.path())
            .arg("-O3")
            .arg("-o")
            .arg(temp_out.path())
            .output()?;

        if !output.status.success() {
            anyhow::bail!("wasm-opt failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        Ok(std::fs::read(temp_out.path())?)
    }
}

#[derive(Debug)]
pub struct TestResult {
    pub input_size: usize,
    pub loom_size: usize,
    pub wasm_opt_size: usize,

    pub loom_valid: bool,
    pub wasm_opt_valid: bool,

    pub semantically_equivalent: Option<bool>, // None if can't execute
}

impl TestResult {
    pub fn compare(input: &[u8], loom: &[u8], wasm_opt: &[u8]) -> Result<Self> {
        let input_size = input.len();
        let loom_size = loom.len();
        let wasm_opt_size = wasm_opt.len();

        // Validate both outputs
        let loom_valid = wasmparser::validate(loom).is_ok();
        let wasm_opt_valid = wasmparser::validate(wasm_opt).is_ok();

        // TODO: Add semantic equivalence checking via wasmtime execution
        let semantically_equivalent = None;

        Ok(TestResult {
            input_size,
            loom_size,
            wasm_opt_size,
            loom_valid,
            wasm_opt_valid,
            semantically_equivalent,
        })
    }

    pub fn loom_wins(&self) -> bool {
        self.loom_valid && self.loom_size < self.wasm_opt_size
    }

    pub fn wasm_opt_wins(&self) -> bool {
        self.wasm_opt_valid && self.wasm_opt_size < self.loom_size
    }

    pub fn tie(&self) -> bool {
        self.loom_valid && self.wasm_opt_valid && self.loom_size == self.wasm_opt_size
    }

    pub fn winner(&self) -> &'static str {
        if self.loom_wins() {
            "LOOM"
        } else if self.wasm_opt_wins() {
            "wasm-opt"
        } else if self.tie() {
            "TIE"
        } else {
            "ERROR"
        }
    }
}
```

#### Step 2: Create Test Runner

```rust
// loom-testing/src/bin/differential.rs
use loom_testing::DifferentialTester;
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let tester = DifferentialTester::new()?;

    // Get all test WASM files
    let test_dir = PathBuf::from("tests/corpus");
    let wasm_files: Vec<_> = glob::glob(&format!("{}/**/*.wasm", test_dir.display()))?
        .collect::<Result<Vec<_>, _>>()?;

    println!("Testing {} WASM files...\n", wasm_files.len());

    let mut loom_wins = 0;
    let mut wasm_opt_wins = 0;
    let mut ties = 0;
    let mut errors = 0;

    for (i, path) in wasm_files.iter().enumerate() {
        print!("[{}/{}] Testing {}...", i + 1, wasm_files.len(), path.display());

        let input = std::fs::read(path)?;
        let result = tester.test(&input)?;

        match result.winner() {
            "LOOM" => {
                loom_wins += 1;
                println!(" ✅ LOOM wins ({} vs {} bytes)", result.loom_size, result.wasm_opt_size);
            }
            "wasm-opt" => {
                wasm_opt_wins += 1;
                println!(" ⚠️  wasm-opt wins ({} vs {} bytes)", result.wasm_opt_size, result.loom_size);
                // TODO: Analyze why wasm-opt won
            }
            "TIE" => {
                ties += 1;
                println!(" 🤝 Tie ({} bytes)", result.loom_size);
            }
            "ERROR" => {
                errors += 1;
                println!(" ❌ Error (LOOM valid: {}, wasm-opt valid: {})",
                    result.loom_valid, result.wasm_opt_valid);
            }
            _ => unreachable!(),
        }
    }

    // Summary
    println!("\n📊 Differential Testing Summary");
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");
    println!("Total tests:     {}", wasm_files.len());
    println!("LOOM wins:       {} ({:.1}%)", loom_wins, 100.0 * loom_wins as f64 / wasm_files.len() as f64);
    println!("wasm-opt wins:   {} ({:.1}%)", wasm_opt_wins, 100.0 * wasm_opt_wins as f64 / wasm_files.len() as f64);
    println!("Ties:            {} ({:.1}%)", ties, 100.0 * ties as f64 / wasm_files.len() as f64);
    println!("Errors:          {}", errors);
    println!("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━");

    let success_rate = (loom_wins + ties) as f64 / wasm_files.len() as f64 * 100.0;
    println!("\n🎯 LOOM success rate: {:.1}%", success_rate);

    Ok(())
}
```

#### Step 3: Collect Test Corpus

Create `scripts/collect_corpus.sh`:

```bash
#!/bin/bash
# Collect real-world WASM files for testing

set -e

CORPUS_DIR="tests/corpus"
mkdir -p "$CORPUS_DIR"

echo "📦 Collecting WebAssembly test corpus..."

# 1. Download from WebAssembly benchmarks repo
echo "Downloading official benchmarks..."
git clone --depth 1 https://github.com/WebAssembly/benchmarks "$CORPUS_DIR/official-benchmarks" || true

# 2. Download wasm-score benchmarks
echo "Downloading wasm-score..."
git clone --depth 1 https://github.com/bytecodealliance/wasm-score "$CORPUS_DIR/wasm-score" || true

# 3. Build Rust examples to WASM
echo "Building Rust examples..."
cargo new --lib "$CORPUS_DIR/rust-examples"
cd "$CORPUS_DIR/rust-examples"
cat > Cargo.toml <<EOF
[package]
name = "rust-examples"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
EOF

cat > src/lib.rs <<EOF
#[no_mangle]
pub extern "C" fn fibonacci(n: i32) -> i32 {
    if n <= 1 { n } else { fibonacci(n - 1) + fibonacci(n - 2) }
}

#[no_mangle]
pub extern "C" fn factorial(n: i32) -> i32 {
    if n <= 1 { 1 } else { n * factorial(n - 1) }
}

#[no_mangle]
pub extern "C" fn sum_array(arr: &[i32]) -> i32 {
    arr.iter().sum()
}
EOF

cargo build --target wasm32-unknown-unknown --release
cp target/wasm32-unknown-unknown/release/rust_examples.wasm "$CORPUS_DIR/"
cd -

# 4. Copy LOOM's own test fixtures
echo "Copying LOOM fixtures..."
cp -r tests/fixtures/*.wasm "$CORPUS_DIR/" 2>/dev/null || true

# 5. Download popular WASM projects (if available)
# Example: Emscripten, AssemblyScript outputs

echo "✅ Corpus collection complete!"
echo "Total WASM files: $(find "$CORPUS_DIR" -name '*.wasm' | wc -l)"
```

---

## Phase 2: Fuzzing (NEXT PRIORITY)

### Setup cargo-fuzz

```bash
cargo install cargo-fuzz
cargo fuzz init
```

### Create Fuzz Target

```rust
// fuzz/fuzz_targets/optimize.rs
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Skip if not valid WASM
    if wasmparser::validate(data).is_err() {
        return;
    }

    // Try to optimize
    if let Ok(optimized) = loom_core::optimize(data) {
        // CRITICAL: Optimized output MUST be valid
        assert!(
            wasmparser::validate(&optimized).is_ok(),
            "LOOM produced invalid WASM!"
        );

        // Size should not increase
        assert!(
            optimized.len() <= data.len(),
            "Optimization increased size: {} -> {}",
            data.len(),
            optimized.len()
        );

        // Idempotence: optimizing again should be no-op
        if let Ok(optimized2) = loom_core::optimize(&optimized) {
            assert_eq!(
                optimized, optimized2,
                "Optimization is not idempotent!"
            );
        }
    }
});
```

### Advanced: Semantic Fuzzing

```rust
// fuzz/fuzz_targets/semantic.rs
#![no_main]

use libfuzzer_sys::fuzz_target;
use wasmtime::*;

fuzz_target!(|data: &[u8]| {
    if wasmparser::validate(data).is_err() {
        return;
    }

    let optimized = match loom_core::optimize(data) {
        Ok(opt) => opt,
        Err(_) => return,
    };

    // Execute both and compare results
    let engine = Engine::default();

    let original_module = match Module::new(&engine, data) {
        Ok(m) => m,
        Err(_) => return,
    };

    let optimized_module = match Module::new(&engine, &optimized) {
        Ok(m) => m,
        Err(_) => return,
    };

    // TODO: Call exported functions with random inputs and compare outputs
    // This proves semantic equivalence
});
```

### Run Fuzzing

```bash
# Fuzz for 1 hour
cargo fuzz run optimize -- -max_total_time=3600

# Fuzz overnight (8 hours)
cargo fuzz run optimize -- -max_total_time=28800

# Continuous fuzzing in CI
cargo fuzz run optimize -- -max_total_time=600  # 10 min per PR
```

---

## Phase 3: Real-World Benchmarks

### Benchmark Structure

```
benchmarks/
├── real-world/
│   ├── pdf-rendering/       # Nutrient benchmark
│   ├── crypto/              # libsodium
│   ├── image-processing/    # image manipulation
│   ├── games/               # Unity, Godot WASM
│   └── scientific/          # numeric computation
├── synthetic/
│   ├── constant-heavy/      # Test constant folding
│   ├── loop-heavy/          # Test LICM
│   └── call-heavy/          # Test inlining
└── regression/
    └── ...                  # Previous bugs
```

### Benchmark Runner

```rust
// loom-benchmarks/src/main.rs
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use std::fs;

fn benchmark_real_world(c: &mut Criterion) {
    let mut group = c.benchmark_group("real-world");

    for entry in glob::glob("benchmarks/real-world/**/*.wasm").unwrap() {
        let path = entry.unwrap();
        let name = path.file_stem().unwrap().to_string_lossy();
        let wasm = fs::read(&path).unwrap();

        group.bench_with_input(
            BenchmarkId::new("loom", &*name),
            &wasm,
            |b, wasm| {
                b.iter(|| {
                    loom_core::optimize(black_box(wasm)).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, benchmark_real_world);
criterion_main!(benches);
```

### Comparison Script

```bash
#!/bin/bash
# scripts/benchmark_comparison.sh

OUTPUT="benchmarks/results/$(date +%Y%m%d_%H%M%S).md"
mkdir -p "$(dirname "$OUTPUT")"

echo "# LOOM vs wasm-opt Benchmark Results" > "$OUTPUT"
echo "**Date:** $(date)" >> "$OUTPUT"
echo "" >> "$OUTPUT"
echo "| Module | Input Size | LOOM Size | wasm-opt Size | Winner | LOOM Time | wasm-opt Time |" >> "$OUTPUT"
echo "|--------|-----------|-----------|---------------|--------|-----------|---------------|" >> "$OUTPUT"

for wasm in benchmarks/real-world/**/*.wasm; do
    name=$(basename "$wasm")
    input_size=$(wc -c < "$wasm")

    # LOOM
    loom_start=$(date +%s%N)
    loom optimize "$wasm" -o /tmp/loom.wasm 2>/dev/null
    loom_end=$(date +%s%N)
    loom_size=$(wc -c < /tmp/loom.wasm)
    loom_time=$(echo "scale=2; ($loom_end - $loom_start) / 1000000" | bc)

    # wasm-opt
    wopt_start=$(date +%s%N)
    wasm-opt "$wasm" -O3 -o /tmp/wasm-opt.wasm 2>/dev/null
    wopt_end=$(date +%s%N)
    wopt_size=$(wc -c < /tmp/wasm-opt.wasm)
    wopt_time=$(echo "scale=2; ($wopt_end - $wopt_start) / 1000000" | bc)

    # Winner
    if [ "$loom_size" -lt "$wopt_size" ]; then
        winner="✅ LOOM"
    elif [ "$loom_size" -gt "$wopt_size" ]; then
        winner="⚠️ wasm-opt"
    else
        winner="🤝 Tie"
    fi

    echo "| $name | $input_size | $loom_size | $wopt_size | $winner | ${loom_time}ms | ${wopt_time}ms |" >> "$OUTPUT"
done

echo "" >> "$OUTPUT"
echo "**Summary:**" >> "$OUTPUT"
# TODO: Calculate win rate, average sizes, etc.

cat "$OUTPUT"
```

---

## Phase 4: Component Model Testing

Since LOOM is the ONLY Component Model optimizer, we need extensive testing:

### Component Test Generator

```rust
// loom-testing/src/component_generator.rs

pub fn generate_test_component(complexity: ComponentComplexity) -> Vec<u8> {
    use wasm_encoder::*;

    let mut component = Component::new();

    // Add core module
    let core_module = generate_core_module(complexity.module_size);
    component.section(&RawSection {
        id: ComponentSectionId::CoreModule.into(),
        data: &core_module,
    });

    // Add component types
    for _ in 0..complexity.type_count {
        // TODO: Generate component types
    }

    // Add exports
    for i in 0..complexity.export_count {
        // TODO: Generate exports
    }

    component.finish()
}

pub enum ComponentComplexity {
    Simple {
        module_size: usize,
        type_count: usize,
        export_count: usize,
    },
    Complex {
        module_count: usize,
        type_count: usize,
        import_count: usize,
        export_count: usize,
        instance_count: usize,
    },
}
```

### Component Fuzzing

```rust
// fuzz/fuzz_targets/component.rs
#![no_main]

use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    // Check if it's a component
    if !is_component(data) || wasmparser::validate(data).is_err() {
        return;
    }

    // Try to optimize component
    if let Ok((optimized, stats)) = loom_core::optimize_component(data) {
        // Must be valid
        assert!(wasmparser::validate(&optimized).is_ok());

        // Must still be a component
        assert!(is_component(&optimized));

        // Size should decrease or stay same
        assert!(optimized.len() <= data.len());

        // Module count should match
        // TODO: Verify structure preservation
    }
});

fn is_component(bytes: &[u8]) -> bool {
    bytes.len() >= 8
        && &bytes[0..4] == b"\0asm"
        && bytes[4] == 0x0d
        && bytes[6] == 0x01
}
```

---

## Phase 5: CI/CD Integration

### GitHub Actions Workflow

```yaml
# .github/workflows/comprehensive-test.yml
name: Comprehensive Testing

on: [push, pull_request]

jobs:
  differential-test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install wasm-opt
        run: |
          wget https://github.com/WebAssembly/binaryen/releases/latest/download/binaryen-linux-x64.tar.gz
          tar xzf binaryen-linux-x64.tar.gz
          sudo cp binaryen-*/bin/wasm-opt /usr/local/bin/

      - name: Build LOOM
        run: cargo build --release

      - name: Collect test corpus
        run: bash scripts/collect_corpus.sh

      - name: Run differential tests
        run: cargo run --bin differential

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: differential-results
          path: results/

  fuzzing:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Install cargo-fuzz
        run: cargo install cargo-fuzz

      - name: Fuzz for 10 minutes
        run: cargo fuzz run optimize -- -max_total_time=600

      - name: Check for crashes
        run: |
          if [ -d fuzz/artifacts/optimize ]; then
            echo "❌ Fuzzing found crashes!"
            exit 1
          fi

  benchmark:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Run benchmarks
        run: cargo bench

      - name: Compare with baseline
        run: |
          # TODO: Compare with previous results
          # Fail if performance regressed

  regression:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Build LOOM
        run: cargo build --release

      - name: Run regression tests
        run: cargo test --test regression

      - name: Verify golden files unchanged
        run: git diff --exit-code tests/golden/
```

---

## Phase 6: Validation & Proof

### Property Tests

```rust
// loom-core/tests/properties.rs

use proptest::prelude::*;

proptest! {
    #[test]
    fn prop_optimization_preserves_validity(wasm in valid_wasm_strategy()) {
        let optimized = loom_core::optimize(&wasm)?;
        assert!(wasmparser::validate(&optimized).is_ok());
    }

    #[test]
    fn prop_optimization_is_idempotent(wasm in valid_wasm_strategy()) {
        let opt1 = loom_core::optimize(&wasm)?;
        let opt2 = loom_core::optimize(&opt1)?;
        assert_eq!(opt1, opt2);
    }

    #[test]
    fn prop_size_decreases_or_equal(wasm in valid_wasm_strategy()) {
        let optimized = loom_core::optimize(&wasm)?;
        assert!(optimized.len() <= wasm.len());
    }
}

fn valid_wasm_strategy() -> impl Strategy<Value = Vec<u8>> {
    // TODO: Generate random valid WASM
    // For now, use fixtures
    prop::sample::select(vec![
        include_bytes!("../tests/fixtures/fibonacci.wasm").to_vec(),
        // ... more fixtures
    ])
}
```

---

## Quick Start Checklist

### Week 1: Foundation
- [ ] Install wasm-opt: `brew install binaryen` (macOS) or download from GitHub
- [ ] Create `loom-testing` crate: `cargo new --lib loom-testing`
- [ ] Implement `DifferentialTester` struct (copy from above)
- [ ] Create `scripts/collect_corpus.sh` (copy from above)
- [ ] Run corpus collection: `bash scripts/collect_corpus.sh`
- [ ] Implement differential test runner
- [ ] **Run first differential test!**

### Week 2: Fuzzing
- [ ] Install cargo-fuzz: `cargo install cargo-fuzz`
- [ ] Initialize fuzzing: `cargo fuzz init`
- [ ] Create `optimize` fuzz target (copy from above)
- [ ] Run fuzz for 1 hour: `cargo fuzz run optimize -- -max_total_time=3600`
- [ ] Fix any crashes found
- [ ] Create `component` fuzz target
- [ ] Run component fuzzing

### Week 3: Benchmarks
- [ ] Create `benchmarks/` directory structure
- [ ] Download WasmScore: `git clone https://github.com/bytecodealliance/wasm-score`
- [ ] Implement benchmark runner (copy from above)
- [ ] Create comparison script (copy from above)
- [ ] **Generate first benchmark report!**

### Week 4: CI Integration
- [ ] Create `.github/workflows/comprehensive-test.yml` (copy from above)
- [ ] Test workflow locally with `act`
- [ ] Push and verify all tests pass
- [ ] Set up automatic benchmark comparisons

---

## Success Metrics

After completing this framework, you should have:

1. ✅ **Differential testing** against wasm-opt on 100+ modules
2. ✅ **Fuzzing** running continuously (0 crashes target)
3. ✅ **Benchmarks** showing LOOM's performance vs wasm-opt
4. ✅ **CI integration** blocking bad PRs automatically
5. ✅ **Proof of correctness** via property testing

**Outcome:** Confidence to claim "LOOM is production-ready with proven correctness"

---

## Troubleshooting

### wasm-opt not found
```bash
# macOS
brew install binaryen

# Linux
wget https://github.com/WebAssembly/binaryen/releases/latest/download/binaryen-linux-x64.tar.gz
tar xzf binaryen-linux-x64.tar.gz
sudo cp binaryen-*/bin/wasm-opt /usr/local/bin/
```

### Fuzzing crashes immediately
- Check that `loom_core::optimize()` handles errors gracefully
- Add error handling in fuzz target
- Use `fuzz_target!(|data: &[u8]| { let _ = loom_core::optimize(data); });`

### Not enough test corpus
- Use wasm-smith to generate synthetic tests
- Download more real-world projects
- Build examples from Rust, C, AssemblyScript

---

## Next Steps After This Framework

Once differential testing + fuzzing + benchmarks are working:

1. **Analyze gaps** - Where does wasm-opt beat LOOM?
2. **Implement missing optimizations** - See OPTIMIZATION_ROADMAP.md
3. **Publish results** - Blog post, paper, benchmark dashboard
4. **Get adoption** - Promote LOOM to WebAssembly community

This framework is the **foundation for proving LOOM is best-in-class**.
