==============
Build Systems
==============

These requirements define how LOOM should be built natively using Cargo and Bazel,
including support for compiling to WASM-WASIP2 as a build target.

Native Build with Cargo
========================

.. req:: Cargo Workspace Configuration
   :id: REQ_BUILD_001
   :status: planned
   :priority: Critical
   :category: Build

   Configure Cargo workspace for building LOOM as a native Rust application.

   The Cargo workspace should:

   - Define workspace members for all crates
   - Configure dependencies (wasmparser, wasm-encoder, ISLE)
   - Set up build.rs for ISLE compilation
   - Support cargo build, test, bench commands
   - Enable workspace-level optimization settings

   **Implementation Notes:**

   - Root Cargo.toml with [workspace] configuration
   - Member crates: loom-core, loom-cli, loom-isle, loom-verify
   - Workspace dependencies for version consistency
   - Profile configurations (dev, release, bench)

.. req:: ISLE Build Integration
   :id: REQ_BUILD_002
   :status: planned
   :priority: Critical
   :category: Build
   :links: REQ_CORE_003

   Integrate ISLE compiler into Cargo build process via build.rs.

   Build script should:

   - Detect .isle files in src/
   - Run ISLE compiler to generate Rust code
   - Output to OUT_DIR for inclusion
   - Trigger rebuild on .isle file changes
   - Support incremental compilation

   **Example build.rs:**

   .. code-block:: rust

      use std::path::PathBuf;

      fn main() {
          let isle_files = vec![
              "src/wasm_terms.isle",
              "src/optimizations.isle",
          ];

          for file in &isle_files {
              println!("cargo:rerun-if-changed={}", file);
          }

          let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

          // Run ISLE compiler
          isle::compile(&isle_files, &out_dir).unwrap();
      }

   **Implementation Notes:**

   - Add isle-compiler as build-dependency
   - Generate one Rust module per .isle file
   - Include generated code with include!() macro
   - Cache compilation artifacts

.. req:: Native Binary Targets
   :id: REQ_BUILD_003
   :status: planned
   :priority: High
   :category: Build

   Support building native binaries for multiple platforms.

   Target platforms:

   - x86_64-unknown-linux-gnu
   - x86_64-unknown-linux-musl (static)
   - x86_64-apple-darwin
   - aarch64-apple-darwin
   - x86_64-pc-windows-msvc
   - aarch64-unknown-linux-gnu

   **Deliverables:**

   - loom CLI tool (bin/loom)
   - loom library (libloom.a, libloom.so)
   - C API headers (optional)

   **Implementation Notes:**

   - Use cargo build --release --target <triple>
   - Configure for static linking where possible
   - Strip debug symbols in release builds
   - Test on each platform in CI

.. req:: Cargo Feature Flags
   :id: REQ_BUILD_004
   :status: planned
   :priority: Medium
   :category: Build

   Define Cargo feature flags for optional functionality.

   Feature flags:

   - ``verification`` - Include Crocus SMT verification
   - ``component-model`` - Component Model support
   - ``simd`` - SIMD optimization passes
   - ``gc`` - GC proposal support
   - ``threads`` - Threads proposal support
   - ``eh`` - Exception handling support
   - ``cli`` - Build CLI tool

   **Example:**

   .. code-block:: toml

      [features]
      default = ["cli"]
      verification = ["dep:crocus", "dep:z3"]
      component-model = ["dep:wit-parser", "dep:wit-component"]
      simd = []
      gc = []
      threads = []
      eh = []
      cli = ["dep:clap"]
      full = ["verification", "component-model", "simd", "gc", "threads", "eh"]

   **Implementation Notes:**

   - Keep core library minimal by default
   - Optional features for heavy dependencies
   - Document feature requirements
   - Test feature combinations in CI

Bazel Build System
==================

.. req:: Bazel Workspace Configuration
   :id: REQ_BUILD_005
   :status: planned
   :priority: Critical
   :category: Build

   Configure Bazel workspace for building LOOM with Bazel.

   The workspace should:

   - Define WORKSPACE with Rust rules
   - Configure rules_rust for Rust compilation
   - Set up rules_wasm_component for WASM targets
   - Define external dependencies
   - Support hermetic builds

   **Example WORKSPACE:**

   .. code-block:: python

      workspace(name = "loom")

      load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

      # Rust rules
      http_archive(
          name = "rules_rust",
          sha256 = "...",
          urls = ["..."],
      )

      load("@rules_rust//rust:repositories.bzl", "rust_repositories")
      rust_repositories(edition = "2021")

      # WASM component rules
      http_archive(
          name = "rules_wasm_component",
          sha256 = "...",
          urls = ["..."],
      )

      load("@rules_wasm_component//wasm:repositories.bzl", "wasm_component_repositories")
      wasm_component_repositories()

   **Implementation Notes:**

   - Pin all external dependencies
   - Use workspace rules for monorepo layout
   - Configure toolchain registration
   - Enable remote caching

.. req:: Bazel Build Targets
   :id: REQ_BUILD_006
   :status: planned
   :priority: Critical
   :category: Build
   :links: REQ_BUILD_005

   Define Bazel BUILD targets for all LOOM components.

   Target structure:

   - ``//loom-core:loom_core`` - Core library
   - ``//loom-cli:loom`` - CLI binary
   - ``//loom-isle:isle_generated`` - Generated ISLE code
   - ``//loom-verify:loom_verify`` - Verification library
   - ``//tests:all`` - All tests

   **Example BUILD file:**

   .. code-block:: python

      load("@rules_rust//rust:defs.bzl", "rust_library", "rust_binary", "rust_test")

      rust_library(
          name = "loom_core",
          srcs = glob(["src/**/*.rs"]) + [":isle_generated"],
          deps = [
              "@crates//:wasmparser",
              "@crates//:wasm-encoder",
          ],
          visibility = ["//visibility:public"],
      )

      genrule(
          name = "isle_generated",
          srcs = glob(["src/**/*.isle"]),
          outs = ["isle_gen.rs"],
          cmd = "$(location //tools:isle-compiler) $(SRCS) > $@",
          tools = ["//tools:isle-compiler"],
      )

      rust_binary(
          name = "loom",
          srcs = ["src/main.rs"],
          deps = [":loom_core"],
      )

   **Implementation Notes:**

   - Use genrule for ISLE compilation
   - Glob patterns for source files
   - Explicit dependency declarations
   - Test targets for each library

.. req:: ISLE Compilation in Bazel
   :id: REQ_BUILD_007
   :status: planned
   :priority: High
   :category: Build
   :links: REQ_BUILD_006, REQ_CORE_003

   Integrate ISLE compiler as a Bazel build action.

   The integration should:

   - Build ISLE compiler as tool
   - Run compiler in genrule
   - Track .isle file dependencies
   - Cache generated Rust code
   - Support incremental builds

   **Example:**

   .. code-block:: python

      load("//bazel:isle.bzl", "isle_library")

      isle_library(
          name = "wasm_optimizations",
          srcs = [
              "src/wasm_terms.isle",
              "src/optimizations/dce.isle",
              "src/optimizations/constants.isle",
          ],
          visibility = ["//visibility:public"],
      )

   **Implementation Notes:**

   - Create custom Starlark rule for ISLE
   - Use aspects for dependency tracking
   - Generate single .rs file per isle_library
   - Support incremental compilation

WASM-WASIP2 Compilation
=======================

.. req:: WASM-WASIP2 Build Target
   :id: REQ_BUILD_008
   :status: planned
   :priority: Critical
   :category: Build
   :links: REQ_BUILD_005

   Support compiling LOOM to WASM-WASIP2 using Bazel and rules_wasm_component.

   The WASM build should:

   - Target wasm32-wasip2 (WASI Preview 2)
   - Compile to .wasm component
   - Support WIT interface definitions
   - Work with component model tooling
   - Provide WASI-compatible binary

   **Target triple:** ``wasm32-wasip2``

   **Expected outputs:**

   - loom.wasm - Core WebAssembly module
   - loom.component.wasm - Component Model component
   - loom.wit - WIT interface definition

   **Implementation Notes:**

   - Use Rust wasm32-wasip2 target
   - Compile with component model support
   - Link against wasi-libc
   - Optimize for size

.. req:: rules_wasm_component Integration
   :id: REQ_BUILD_009
   :status: planned
   :priority: Critical
   :category: Build
   :links: REQ_BUILD_008

   Integrate rules_wasm_component Bazel rules for WASM compilation.

   Use rules_wasm_component to:

   - Compile Rust to wasm32-wasip2
   - Generate WIT interfaces
   - Create component bindings
   - Link multiple components
   - Validate component structure

   **Example BUILD:**

   .. code-block:: python

      load("@rules_wasm_component//wasm:defs.bzl",
           "wasm_rust_binary", "wasm_component")

      wasm_rust_binary(
          name = "loom_wasm",
          srcs = ["src/main.rs"],
          target = "wasm32-wasip2",
          deps = [
              "//loom-core",
              "@crates//:wasmparser",
          ],
      )

      wasm_component(
          name = "loom_component",
          binary = ":loom_wasm",
          wit = "loom.wit",
          world = "loom",
      )

   **Implementation Notes:**

   - Load rules from @rules_wasm_component
   - Define WIT interfaces for API
   - Use wasm_component for component creation
   - Test with wasmtime

.. req:: WIT Interface Definition
   :id: REQ_BUILD_010
   :status: planned
   :priority: High
   :category: Build
   :links: REQ_BUILD_008, REQ_BUILD_009

   Define WIT (WebAssembly Interface Types) interface for LOOM.

   The WIT interface should expose:

   - Module parsing and validation
   - Optimization pipeline execution
   - Individual optimization passes
   - Configuration options
   - Error handling

   **Example loom.wit:**

   .. code-block:: wit

      package loom:optimizer@0.1.0;

      interface optimize {
          record optimization-config {
              level: u8,
              passes: list<string>,
              verify: bool,
          }

          enum optimization-error {
              parse-error,
              validation-error,
              optimization-error,
          }

          optimize: func(
              input: list<u8>,
              config: optimization-config
          ) -> result<list<u8>, optimization-error>;

          validate: func(input: list<u8>) -> result<_, optimization-error>;
      }

      world loom {
          export optimize;
      }

   **Implementation Notes:**

   - Define minimal, stable API
   - Version the interface
   - Document all functions
   - Support component composition

.. req:: WASM-WASIP2 Feature Support
   :id: REQ_BUILD_011
   :status: planned
   :priority: High
   :category: Build
   :links: REQ_BUILD_008

   Ensure LOOM functionality works correctly when compiled to WASM-WASIP2.

   Requirements:

   - File I/O via WASI filesystem interface
   - Memory management within WASM limits
   - No native thread dependencies
   - Async I/O support (if needed)
   - Error handling via Result types

   **Limitations to handle:**

   - No native process spawning
   - Limited memory (can be paged)
   - No direct syscalls
   - WASI-only I/O

   **Implementation Notes:**

   - Use conditional compilation (#[cfg(target_arch = "wasm32")])
   - Abstract I/O behind traits
   - Test WASM builds regularly
   - Document WASM-specific limitations

.. req:: WASM Optimization for Size
   :id: REQ_BUILD_012
   :status: planned
   :priority: Medium
   :category: Build
   :links: REQ_BUILD_008

   Optimize WASM binary size for distribution.

   Size optimization techniques:

   - Compile with opt-level = "z"
   - Strip debug symbols
   - Run wasm-opt on output
   - Use LTO (Link Time Optimization)
   - Minimize standard library usage

   **Target size:** < 2MB for core LOOM component

   **Build flags:**

   .. code-block:: toml

      [profile.release-wasm]
      inherits = "release"
      opt-level = "z"
      lto = true
      codegen-units = 1
      strip = true
      panic = "abort"

   **Implementation Notes:**

   - Create separate release-wasm profile
   - Post-process with wasm-opt -Oz
   - Measure binary size in CI
   - Track size regressions

Build Testing and CI
====================

.. req:: Multi-Target Build Testing
   :id: REQ_BUILD_013
   :status: planned
   :priority: High
   :category: Build

   Test that LOOM builds successfully for all target platforms.

   CI should test:

   - Native Linux (x86_64, aarch64)
   - Native macOS (x86_64, aarch64)
   - Native Windows (x86_64)
   - WASM-WASIP2 (wasm32-wasip2)

   Both build systems:

   - Cargo builds
   - Bazel builds

   **Implementation Notes:**

   - Matrix builds in GitHub Actions
   - Cache build artifacts
   - Test both debug and release
   - Verify binary execution

.. req:: Build System Parity
   :id: REQ_BUILD_014
   :status: planned
   :priority: High
   :category: Build
   :links: REQ_BUILD_005

   Ensure Cargo and Bazel builds produce equivalent outputs.

   Verify that:

   - Same source files are compiled
   - Same dependencies are used
   - Generated code is identical
   - Tests pass with both builds
   - Performance is comparable

   **Testing approach:**

   - Build with both systems
   - Compare binary behavior
   - Run full test suite
   - Benchmark performance
   - Check for Cargo-specific or Bazel-specific bugs

   **Implementation Notes:**

   - Maintain parallel build configs
   - Document differences if any
   - CI tests both systems
   - Keep dependencies in sync

.. req:: Hermetic Builds
   :id: REQ_BUILD_015
   :status: planned
   :priority: Medium
   :category: Build
   :links: REQ_BUILD_005

   Ensure builds are hermetic and reproducible.

   Hermetic build requirements:

   - No network access during build
   - All dependencies pre-fetched
   - Deterministic output
   - No system dependencies
   - Reproducible on any machine

   **For Bazel:**

   - All deps in WORKSPACE
   - No repository_ctx.download()
   - Sandbox execution
   - Remote caching

   **For Cargo:**

   - Cargo.lock pinning
   - Vendored dependencies
   - Offline builds (--offline)

   **Implementation Notes:**

   - Test offline builds
   - Vendor dependencies
   - Use content-addressed storage
   - Verify reproducibility

.. req:: Cross-Compilation Support
   :id: REQ_BUILD_016
   :status: planned
   :priority: Medium
   :category: Build
   :links: REQ_BUILD_003

   Support cross-compilation for all target platforms.

   Cross-compilation scenarios:

   - Linux → Windows (via MinGW)
   - macOS → Linux (via cross)
   - x86_64 → aarch64
   - Native → WASM-WASIP2

   **Tools:**

   - cross-rs for Cargo
   - Bazel toolchain registration
   - QEMU for testing

   **Implementation Notes:**

   - Pre-built cross-compilation toolchains
   - Document cross-compilation setup
   - Test cross-compiled binaries
   - CI cross-compilation checks

.. req:: Build Performance
   :id: REQ_BUILD_017
   :status: planned
   :priority: Low
   :category: Build

   Optimize build performance for development velocity.

   Performance targets:

   - Full clean build: < 5 minutes
   - Incremental rebuild: < 10 seconds
   - Test run: < 2 minutes
   - ISLE regeneration: < 5 seconds

   **Optimization techniques:**

   - Parallel compilation (-j)
   - Incremental compilation
   - sccache or ccache
   - Bazel remote caching
   - Minimal dependencies

   **Implementation Notes:**

   - Profile build times
   - Identify bottlenecks
   - Optimize critical paths
   - Use build caching
   - Measure in CI

.. req:: Development Container
   :id: REQ_BUILD_018
   :status: planned
   :priority: Low
   :category: Build

   Provide development container with all build tools.

   Container should include:

   - Rust toolchain (stable + nightly)
   - Cargo and Bazel
   - WASM tooling (wasm32-wasip2 target)
   - ISLE compiler
   - Test dependencies
   - Documentation tools

   **Outputs:**

   - Dockerfile
   - devcontainer.json (VS Code)
   - docker-compose.yml

   **Implementation Notes:**

   - Based on rust:latest
   - Multi-stage build
   - Volume mounts for source
   - Published to ghcr.io

Documentation and Tooling
=========================

.. req:: Build Documentation
   :id: REQ_BUILD_019
   :status: planned
   :priority: High
   :category: Build

   Document build system setup and usage.

   Documentation should cover:

   - Prerequisites installation
   - Cargo build instructions
   - Bazel build instructions
   - WASM-WASIP2 compilation
   - Cross-compilation guide
   - Troubleshooting common issues

   **Locations:**

   - README.md (quick start)
   - docs/development/building.rst (comprehensive)
   - BUILDING.md (detailed)

   **Implementation Notes:**

   - Step-by-step instructions
   - Platform-specific notes
   - Example commands
   - Links to tool documentation

.. req:: Build Scripts and Automation
   :id: REQ_BUILD_020
   :status: planned
   :priority: Medium
   :category: Build

   Provide scripts to automate common build tasks.

   Scripts should include:

   - ``scripts/build.sh`` - Build all targets
   - ``scripts/build-wasm.sh`` - Build WASM only
   - ``scripts/test.sh`` - Run all tests
   - ``scripts/release.sh`` - Create release artifacts
   - ``scripts/verify.sh`` - Run verification

   **Features:**

   - Support both Cargo and Bazel
   - Parallel builds
   - Error handling
   - Colored output
   - Progress indicators

   **Implementation Notes:**

   - POSIX shell scripts (portable)
   - Clear error messages
   - Verbose and quiet modes
   - CI-friendly output
