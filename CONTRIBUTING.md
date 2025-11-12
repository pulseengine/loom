# Contributing to LOOM

Thank you for your interest in contributing to LOOM!

## Development Setup

### Prerequisites

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install pre-commit
pip install pre-commit

# Install wasm-tools (for testing)
cargo install wasm-tools
```

### Setup

```bash
# Clone the repository
git clone https://github.com/pulseengine/loom.git
cd loom

# Install pre-commit hooks
pre-commit install

# Build and test
cargo build
cargo test
```

## Pre-commit Hooks

This project uses pre-commit hooks to ensure code quality. The hooks will:

- Check formatting with `cargo fmt`
- Run lints with `cargo clippy`
- Run tests with `cargo test`
- Validate YAML and TOML files
- Check for trailing whitespace

### Running Hooks Manually

```bash
# Run all hooks
pre-commit run --all-files

# Run specific hook
pre-commit run cargo-fmt --all-files
```

### Skipping Hooks (Not Recommended)

```bash
git commit --no-verify
```

## Code Quality Standards

### Formatting

All code must be formatted with `rustfmt`:

```bash
cargo fmt --all
```

### Linting

Code must pass `clippy` with no warnings:

```bash
cargo clippy --all-targets --all-features -- -D warnings
```

### Testing

All tests must pass:

```bash
cargo test --all
```

### WebAssembly Validation

Optimized output must be valid WebAssembly:

```bash
./target/release/loom optimize input.wat -o output.wasm
wasm-tools validate output.wasm
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`
2. **Make your changes** following the code quality standards
3. **Add tests** for new functionality
4. **Update documentation** if needed
5. **Run pre-commit hooks**: `pre-commit run --all-files`
6. **Commit your changes** with a descriptive message
7. **Push to your fork** and create a pull request

### Commit Message Guidelines

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Reference issues and pull requests when relevant
- Keep first line under 72 characters

Examples:
```
Add memory redundancy elimination

Implement local variable constant propagation

Fix clippy warnings in loom-isle (#42)
```

## CI/CD Pipeline

All pull requests must pass CI checks:

- ✅ Format check (`cargo fmt --check`)
- ✅ Clippy lints (`cargo clippy -D warnings`)
- ✅ Tests on Ubuntu, macOS, Windows
- ✅ Release build
- ✅ WebAssembly validation

## Project Structure

```
loom/
├── loom-core/          # Core library (parser, encoder, optimizer)
├── loom-isle/          # ISLE term definitions and rules
├── loom-cli/           # Command-line interface
├── tests/fixtures/     # Test WebAssembly files
└── docs/               # Sphinx documentation
```

## Adding New Optimizations

1. **Define ISLE terms** in `loom-isle/isle/wasm_terms.isle`
2. **Implement constructors** in `loom-isle/src/lib.rs`
3. **Add optimization logic** to `simplify_with_env` or `simplify_stateless`
4. **Update parser** in `loom-core/src/lib.rs` (instruction_to_term conversion)
5. **Update encoder** in `loom-core/src/lib.rs` (term_to_instruction conversion)
6. **Add tests** in `loom-core/src/lib.rs` or create fixtures
7. **Document** the optimization in requirements

## Testing Guidelines

### Unit Tests

Place tests in the same file as the code:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constant_folding() {
        // Test code
    }
}
```

### Integration Tests

Create test fixtures in `tests/fixtures/`:

```wat
(module
  (func $test (result i32)
    i32.const 10
    i32.const 32
    i32.add
  )
)
```

Then test with:
```bash
./target/release/loom optimize tests/fixtures/test.wat -o /tmp/test.wasm
wasm-tools validate /tmp/test.wasm
```

## Documentation

- **Code comments**: Document non-obvious logic
- **Rustdoc**: Use `///` for public APIs
- **Sphinx docs**: Update `docs/source/requirements/` for major features

Build documentation:
```bash
cd docs
pip install -r requirements.txt
make html
```

## Getting Help

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For questions and ideas
- **Pull Requests**: Review open PRs for examples

## Code of Conduct

Be respectful, inclusive, and professional in all interactions.

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
