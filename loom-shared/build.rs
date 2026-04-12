use std::env;
use std::path::PathBuf;

use cranelift_isle as isle;

fn main() {
    // Get the output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    // Define ISLE source files in order
    // Order matters: types -> constructors -> rules
    let isle_files = vec![
        PathBuf::from("isle/types.isle"),
        PathBuf::from("isle/constructors.isle"),
        PathBuf::from("isle/rules/constant_folding.isle"),
        PathBuf::from("isle/rules/default.isle"),
    ];

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

    // Post-process generated code to fix ownership issues.
    //
    // The ISLE compiler generates code that passes Value by value to extractor
    // functions. Since our Value type wraps Box<ValueData> and is not Copy,
    // this causes "use of moved value" errors when multiple extractors are tried
    // on the same value. We fix this by:
    //
    // 1. Changing extractor trait signatures to take &Value instead of Value
    // 2. Changing extractor call sites to pass references instead of owned values
    //
    // This is safe because extractors only inspect the value (read-only)
    // and return owned copies of inner data.
    let generated_code = fix_extractor_ownership(&generated_code);

    // Write to output directory
    let output_file = out_dir.join("isle_generated.rs");
    std::fs::write(&output_file, &generated_code).expect("Failed to write generated ISLE code");

    println!(
        "cargo:warning=Generated ISLE code written to {}",
        output_file.display()
    );
}

/// Post-process ISLE-generated Rust code to fix Value ownership issues.
///
/// The ISLE compiler assumes all primitive types are Copy, but our Value type
/// wraps Box<ValueData> and cannot be Copy. This function rewrites extractor
/// signatures and call sites to use references instead of owned values.
fn fix_extractor_ownership(code: &str) -> String {
    let mut result = code.to_string();

    // List of all extractor functions that take Value and need to take &Value
    let extractors = [
        "iconst32_extract",
        "iadd32_extract",
        "isub32_extract",
        "imul32_extract",
        "iconst64_extract",
        "iadd64_extract",
        "isub64_extract",
        "imul64_extract",
    ];

    for extractor in &extractors {
        // Fix trait declarations: change "arg0: Value)" to "arg0: &Value)"
        // Pattern: "fn xxx_extract(&mut self, arg0: Value)"
        let owned_sig = format!("fn {}(&mut self, arg0: Value)", extractor);
        let ref_sig = format!("fn {}(&mut self, arg0: &Value)", extractor);
        result = result.replace(&owned_sig, &ref_sig);

        // Fix call sites: change "C::xxx_extract(ctx, v)" to "C::xxx_extract(ctx, &v)"
        // We need to handle various variable names (arg0, v2.0, v2.1, etc.)
        // The pattern is "C::xxx_extract(ctx, " followed by a variable
        let call_prefix = format!("C::{}(ctx, ", extractor);
        // Replace all occurrences of the call pattern
        let mut new_result = String::new();
        let mut remaining = result.as_str();
        while let Some(pos) = remaining.find(&call_prefix) {
            new_result.push_str(&remaining[..pos]);
            new_result.push_str(&call_prefix);
            remaining = &remaining[pos + call_prefix.len()..];

            // Find the closing parenthesis for this call
            if let Some(paren_pos) = remaining.find(')') {
                let arg = &remaining[..paren_pos];
                // Only add & if the argument doesn't already start with &
                if !arg.starts_with('&') {
                    new_result.push('&');
                }
                new_result.push_str(arg);
                new_result.push(')');
                remaining = &remaining[paren_pos + 1..];
            }
        }
        new_result.push_str(remaining);
        result = new_result;
    }

    result
}
