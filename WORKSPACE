workspace(name = "loom")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Rust rules
http_archive(
    name = "rules_rust",
    integrity = "sha256-3Ch+PsqAsp1cyV4mHK4nPu3xr0oAqWrpN+I0U02tskw=",
    urls = ["https://github.com/bazelbuild/rules_rust/releases/download/0.67.0/rules_rust-0.67.0.tar.gz"],
)

load("@rules_rust//rust:repositories.bzl", "rules_rust_dependencies", "rust_register_toolchains")

rules_rust_dependencies()

rust_register_toolchains(
    edition = "2021",
    versions = ["1.75.0"],
    extra_target_triples = [
        "wasm32-wasip2",
    ],
)

load("@rules_rust//crate_universe:repositories.bzl", "crate_universe_dependencies")

crate_universe_dependencies()

load("@rules_rust//crate_universe:defs.bzl", "crate", "crates_repository")

# External Rust dependencies
crates_repository(
    name = "crates",
    cargo_lockfile = "//:Cargo.lock",
    lockfile = "//:Cargo.Bazel.lock",
    manifests = [
        "//:Cargo.toml",
    ],
)

load("@crates//:defs.bzl", "crate_repositories")

crate_repositories()

# WASM component rules (for Phase 8)
# Commented out for now since rules_wasm_component setup requires additional configuration
# http_archive(
#     name = "rules_wasm_component",
#     # URL and sha256 to be added when we start Phase 8
# )
