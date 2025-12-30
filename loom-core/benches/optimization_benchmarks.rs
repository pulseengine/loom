//! Performance benchmarks for LOOM optimizations
//!
//! Run with: cargo bench

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use loom_core::{optimize, parse};

/// Benchmark constant folding performance
fn bench_constant_folding(c: &mut Criterion) {
    let mut group = c.benchmark_group("constant_folding");

    let inputs = vec![
        (
            "simple",
            r#"
            (module
                (func (result i32)
                    i32.const 10
                    i32.const 20
                    i32.add
                )
            )
        "#,
        ),
        (
            "nested",
            r#"
            (module
                (func (result i32)
                    i32.const 5
                    i32.const 10
                    i32.add
                    i32.const 2
                    i32.mul
                    i32.const 3
                    i32.add
                )
            )
        "#,
        ),
        (
            "complex",
            r#"
            (module
                (func (result i32)
                    i32.const 100
                    i32.const 50
                    i32.add
                    i32.const 2
                    i32.mul
                    i32.const 10
                    i32.sub
                    i32.const 5
                    i32.div_u
                )
            )
        "#,
        ),
    ];

    for (name, wat) in inputs {
        group.bench_with_input(BenchmarkId::from_parameter(name), wat, |b, wat| {
            b.iter(|| {
                let mut module = parse::parse_wat(wat).unwrap();
                optimize::optimize_module(black_box(&mut module)).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark strength reduction performance
fn bench_strength_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("strength_reduction");

    let inputs = vec![
        (
            "mul_by_4",
            r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 4
                    i32.mul
                )
            )
        "#,
        ),
        (
            "mul_by_8",
            r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 8
                    i32.mul
                )
            )
        "#,
        ),
        (
            "div_by_16",
            r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 16
                    i32.div_u
                )
            )
        "#,
        ),
        (
            "rem_by_32",
            r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 32
                    i32.rem_u
                )
            )
        "#,
        ),
    ];

    for (name, wat) in inputs {
        group.bench_with_input(BenchmarkId::from_parameter(name), wat, |b, wat| {
            b.iter(|| {
                let mut module = parse::parse_wat(wat).unwrap();
                optimize::optimize_module(black_box(&mut module)).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark CSE performance
fn bench_cse(c: &mut Criterion) {
    let mut group = c.benchmark_group("cse");

    let inputs = vec![
        (
            "duplicate_const",
            r#"
            (module
                (func (result i32)
                    i32.const 42
                    i32.const 42
                    i32.add
                    i32.const 42
                    i32.add
                )
            )
        "#,
        ),
        (
            "duplicate_computation",
            r#"
            (module
                (func (param i32) (result i32)
                    local.get 0
                    i32.const 10
                    i32.add
                    local.get 0
                    i32.const 10
                    i32.add
                    i32.add
                )
            )
        "#,
        ),
    ];

    for (name, wat) in inputs {
        group.bench_with_input(BenchmarkId::from_parameter(name), wat, |b, wat| {
            b.iter(|| {
                let mut module = parse::parse_wat(wat).unwrap();
                optimize::optimize_module(black_box(&mut module)).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark function inlining performance
fn bench_function_inlining(c: &mut Criterion) {
    let mut group = c.benchmark_group("function_inlining");

    let inputs = vec![
        (
            "simple_inline",
            r#"
            (module
                (func $helper (param i32) (result i32)
                    local.get 0
                    i32.const 1
                    i32.add
                )
                (func $main (param i32) (result i32)
                    local.get 0
                    call $helper
                )
            )
        "#,
        ),
        (
            "multiple_calls",
            r#"
            (module
                (func $helper (param i32) (result i32)
                    local.get 0
                    i32.const 2
                    i32.mul
                )
                (func $main (param i32) (result i32)
                    local.get 0
                    call $helper
                    local.get 0
                    call $helper
                    i32.add
                )
            )
        "#,
        ),
    ];

    for (name, wat) in inputs {
        group.bench_with_input(BenchmarkId::from_parameter(name), wat, |b, wat| {
            b.iter(|| {
                let mut module = parse::parse_wat(wat).unwrap();
                optimize::optimize_module(black_box(&mut module)).unwrap()
            });
        });
    }

    group.finish();
}

/// Benchmark full optimization pipeline on realistic code
fn bench_full_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("full_pipeline");

    // Realistic example: simple math function
    let math_example = r#"
        (module
            (func $compute (param $x i32) (param $y i32) (result i32)
                (local $temp i32)
                ;; Compute (x * 4 + y * 8) / 16
                local.get $x
                i32.const 4
                i32.mul
                local.get $y
                i32.const 8
                i32.mul
                i32.add
                i32.const 16
                i32.div_u
            )
        )
    "#;

    // Realistic example: loop with invariant code
    let loop_example = r#"
        (module
            (func $loop_sum (param $n i32) (result i32)
                (local $i i32)
                (local $sum i32)
                (local $factor i32)

                i32.const 0
                local.set $i
                i32.const 0
                local.set $sum

                (loop $continue
                    ;; Invariant: compute factor (should be hoisted)
                    i32.const 10
                    i32.const 20
                    i32.add
                    local.set $factor

                    ;; Add factor to sum
                    local.get $sum
                    local.get $factor
                    i32.add
                    local.set $sum

                    ;; Increment counter
                    local.get $i
                    i32.const 1
                    i32.add
                    local.tee $i

                    ;; Continue if i < n
                    local.get $n
                    i32.lt_u
                    br_if $continue
                )

                local.get $sum
            )
        )
    "#;

    group.bench_function("math_function", |b| {
        b.iter(|| {
            let mut module = parse::parse_wat(math_example).unwrap();
            optimize::optimize_module(black_box(&mut module)).unwrap()
        });
    });

    group.bench_function("loop_with_licm", |b| {
        b.iter(|| {
            let mut module = parse::parse_wat(loop_example).unwrap();
            optimize::optimize_module(black_box(&mut module)).unwrap()
        });
    });

    group.finish();
}

/// Benchmark parser performance (baseline comparison)
fn bench_parser(c: &mut Criterion) {
    let mut group = c.benchmark_group("parser");

    let simple = r#"
        (module
            (func (result i32)
                i32.const 42
            )
        )
    "#;

    group.bench_function("parse_simple", |b| {
        b.iter(|| parse::parse_wat(black_box(simple)).unwrap());
    });

    group.finish();
}

/// Benchmark encoder performance (baseline comparison)
fn bench_encoder(c: &mut Criterion) {
    let mut group = c.benchmark_group("encoder");

    let simple = r#"
        (module
            (func (result i32)
                i32.const 42
            )
        )
    "#;

    let module = parse::parse_wat(simple).unwrap();

    group.bench_function("encode_simple", |b| {
        b.iter(|| loom_core::encode::encode_wasm(black_box(&module)).unwrap());
    });

    group.finish();
}

/// Benchmark optimization idempotence
fn bench_idempotence(c: &mut Criterion) {
    let mut group = c.benchmark_group("idempotence");

    let complex = r#"
        (module
            (func (param i32) (result i32)
                local.get 0
                i32.const 4
                i32.mul
                i32.const 0
                i32.add
                local.get 0
                i32.const 4
                i32.mul
                i32.add
            )
        )
    "#;

    // First optimization pass
    let mut module1 = parse::parse_wat(complex).unwrap();
    optimize::optimize_module(&mut module1).unwrap();

    group.bench_function("second_pass", |b| {
        b.iter(|| {
            let mut module = module1.clone();
            optimize::optimize_module(black_box(&mut module)).unwrap()
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_constant_folding,
    bench_strength_reduction,
    bench_cse,
    bench_function_inlining,
    bench_full_pipeline,
    bench_parser,
    bench_encoder,
    bench_idempotence,
);

criterion_main!(benches);
