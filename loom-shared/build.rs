use std::env;
use std::path::PathBuf;

use cranelift_isle as isle;

fn main() {
    // Get the output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Define ISLE source files
    let isle_files = vec![PathBuf::from("isle/wasm_terms.isle")];

    // Tell cargo to rerun if any ISLE file changes
    for file in &isle_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    // Check all files exist
    for file in &isle_files {
        if !file.exists() {
            panic!("ISLE file {} does not exist", file.display());
        }
    }

    // Configure code generation options
    let options = isle::codegen::CodegenOptions {
        exclude_global_allow_pragmas: true,
    };

    // Compile ISLE files to Rust code
    let generated_code =
        isle::compile::from_files(isle_files.iter(), &options).expect("Failed to compile ISLE");

    // Write to output directory
    let output_file = out_dir.join("isle_generated.rs");
    std::fs::write(&output_file, &generated_code).expect("Failed to write generated ISLE code");

    println!(
        "cargo:warning=Generated ISLE code written to {}",
        output_file.display()
    );
}
