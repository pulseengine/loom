# Building LOOM for WebAssembly (wasm32-wasip2)

This document describes how to build LOOM for WebAssembly using the wasm32-wasip2 target.

## Prerequisites

1. **Rust toolchain with wasm32-wasip2 support**:
   ```bash
   rustup target add wasm32-wasip2
   ```

2. **Bazel** (for Bazel builds):
   - Bazel 6.0 or later
   - The wasm32-wasip2 toolchain is automatically registered via `extra_target_triples`

## Building with Cargo

### Development Build

```bash
cargo build --target wasm32-wasip2
```

### Release Build (optimized for size)

```bash
cargo build --target wasm32-wasip2 --profile release-wasm
```

The `release-wasm` profile is configured for minimal WASM binary size:
- `opt-level = "z"` - Optimize for size
- `lto = true` - Enable link-time optimization
- `codegen-units = 1` - Single codegen unit for better optimization
- `strip = true` - Strip debug symbols
- `panic = "abort"` - Smaller panic handler

## Building with Bazel

### Build LOOM for WASM

```bash
# Build the WASM binary
bazel build //loom-cli:loom_wasm --platforms=@rules_rust//rust/platform:wasm

# Output will be in: bazel-bin/loom-cli/loom_wasm.wasm
```

### Build with custom platform

For more control over the WASM build, you can define a custom platform:

```python
# platforms/BUILD.bazel
platform(
    name = "wasm32_wasip2",
    constraint_values = [
        "@platforms//cpu:wasm32",
        "@platforms//os:wasi",
    ],
)
```

Then build with:
```bash
bazel build //loom-cli:loom_wasm --platforms=//platforms:wasm32_wasip2
```

## Deployment

### WebAssembly Component Model

LOOM is designed to work with the WebAssembly Component Model. To create a component:

```bash
# Install wasm-tools
cargo install wasm-tools

# Build LOOM as WASM
cargo build --target wasm32-wasip2 --release

# Create a component (if using WIT interface)
wasm-tools component new \
  target/wasm32-wasip2/release/loom.wasm \
  -o loom.component.wasm
```

### Running LOOM WASM

You can run LOOM compiled to WASM using a WASI runtime:

```bash
# Using wasmtime
wasmtime target/wasm32-wasip2/release/loom.wasm optimize input.wasm -o output.wasm

# Using wasmer
wasmer run target/wasm32-wasip2/release/loom.wasm -- optimize input.wasm -o output.wasm
```

## Size Optimization

The `release-wasm` profile produces binaries optimized for size. Additional optimizations:

### Using wasm-opt

```bash
# Install Binaryen tools
npm install -g binaryen

# Optimize the WASM binary
wasm-opt -Oz target/wasm32-wasip2/release/loom.wasm -o loom.opt.wasm

# With aggressive optimizations
wasm-opt -Oz --enable-bulk-memory --enable-simd \
  target/wasm32-wasip2/release/loom.wasm \
  -o loom.opt.wasm
```

### Using wasm-snip

Remove unused functions:

```bash
cargo install wasm-snip

wasm-snip target/wasm32-wasip2/release/loom.wasm \
  -o loom.snipped.wasm \
  --snip-rust-panicking-code
```

## Troubleshooting

### Missing wasm32-wasip2 target

If you get an error about missing wasm32-wasip2 target:

```bash
rustup target add wasm32-wasip2
```

### Bazel build errors

If Bazel fails to find the WASM toolchain:

```bash
# Re-sync Bazel's Rust toolchains
bazel sync --configure
```

### ISLE build script issues

The ISLE code generation happens during build. For WASM builds, ensure:
- ISLE compiler runs on the host platform (not WASM)
- Generated code is included in the build

## Performance

WASM builds of LOOM can be run in:
- Server-side WASI runtimes (Wasmtime, Wasmer, WasmEdge)
- Browser environments (with WASI polyfill)
- Edge computing platforms (Fastly Compute@Edge, Cloudflare Workers)

Expected performance:
- ~2-3x slower than native for I/O-bound operations
- ~1.5-2x slower for CPU-bound optimizations
- Smaller deployment footprint (~1-2MB compressed)

## Future Work

- **Component Model**: Full WIT interface for LOOM as a component
- **WASI Preview 2**: Migrate to newer WASI standard
- **JavaScript Bindings**: NPM package for browser use
- **Edge Runtime**: Optimized builds for edge platforms
