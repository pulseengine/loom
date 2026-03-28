{
  description = "LOOM - provably correct WebAssembly optimizer";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.11";

    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, rust-overlay, flake-utils }:
    flake-utils.lib.eachSystem [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ] (system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs { inherit system overlays; };

        # Rust stable with wasm32-wasip2 target
        rustToolchain = pkgs.rust-bin.stable."1.93.1".default.override {
          extensions = [
            "rust-src"
            "rust-analyzer"
            "clippy"
            "rustfmt"
          ];
          targets = [ "wasm32-wasip2" ];
        };

        # Z3 paths for the z3-sys crate build
        z3Header = "${pkgs.z3.dev}/include/z3.h";
        z3LibPath = "${pkgs.z3.lib}/lib";
      in
      {
        devShells.default = pkgs.mkShell {
          buildInputs = [
            rustToolchain

            # Z3 SMT solver (for --features verification)
            pkgs.z3

            # WebAssembly tooling
            pkgs.wasmtime
            pkgs.binaryen

            pkgs.pkg-config
          ] ++ pkgs.lib.optionals pkgs.stdenv.isDarwin [
            pkgs.darwin.apple_sdk.frameworks.Security
            pkgs.darwin.apple_sdk.frameworks.SystemConfiguration
            pkgs.libiconv
          ];

          env = {
            Z3_SYS_Z3_HEADER = z3Header;
            LIBRARY_PATH = z3LibPath;
            LD_LIBRARY_PATH = z3LibPath;
            RUST_SRC_PATH = "${rustToolchain}/lib/rustlib/src/rust/library";
          };

          shellHook = ''
            echo "loom dev shell"
            echo "  rust:     $(rustc --version)"
            echo "  z3:       ${pkgs.z3.version}"
            echo "  wasmtime: $(wasmtime --version 2>/dev/null)"
          '';
        };
      }
    );
}
