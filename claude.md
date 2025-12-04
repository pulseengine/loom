# Claude Code Guidelines for LOOM

## Commit Messages

Use conventional commits with concise formatting:
- Type: feat, fix, refactor, docs, test, perf, chore
- Scope: optional, e.g. (lib), (verify), (tests)
- Subject: lowercase, present tense, max 50 chars
- Body: only if needed, max 72 chars per line
- NO Unicode/emojis
- NO "Generated with Claude Code" footer
- NO "Co-Authored-By" lines
- Keep messages focused on actual changes, not verbose documentation

Example:
```
feat(lib): add F32/F64 constant support

Parser, encoder, and verification for floating-point constants.
Adds 10 test cases covering parsing, roundtrip, and optimization.
```

## Code Style

- Follow Rust conventions and existing LOOM patterns
- Add documentation only for public APIs
- Avoid over-engineering: solve the current problem, not hypothetical ones
- Use type-level proofs via Rust's type system
- Verification via Z3 for optimizer correctness

## Testing

- All changes must pass existing tests
- Add tests for new functionality
- Run full suite before committing: `cargo test`
- Focus on semantics preservation

## PR Guidelines

- Keep changes focused and reviewable
- Reference GitHub issues when applicable
- Verify formal properties via Z3 if optimization-related
- Test empirically with wasmtime/wasm-smith
