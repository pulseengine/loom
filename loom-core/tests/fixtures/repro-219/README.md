# #219 seam-SROA repro

`sem.loom.wasm` — gale's authentic post-loom-inline dissolved `z_impl_k_sem_give`
(the C↔Rust k_sem_give decide seam). Provided on issue #219; **sha256
`f81da42d…`, 5254 B** (verified on decode). From
`gale-smart-data/benches/engine_control/silicon/wasm-testbed/repro-loom-seam-sroa/`.

It still contains the u64 pack/unpack round-trip the seam-SROA pass must dissolve:

- **pack** (decide builds the u64): `i64.extend_i32_u; i64.shl; i64.or`
- **unpack** (shim tears it back to scalars): `i64.and`, `i64.shr_u`, `i32.wrap_i64`
- a dead `i64` carrier local in between

## Kill-criterion (gale, on-silicon G474RE)

- structural: dissolved body has **no i64 pack/unpack**, carrier local gone,
  ARM body 83 → ~55–60 insns.
- silicon: sem 860 → toward 471 (LLVM-LTO); mutex 472. gale re-flashes.

## Plan (design confirmed on the issue — proof-carrying)

1. wasm-local mem2reg: promote the single-assignment non-escaping i64 carrier so
   the pack expression reaches the unpack sites.
2. SROA / pack-unpack algebraic forwarding via ISLE rewrites (preferred: add
   `i64.extend_i32_u` / `i32.wrap_i64` ISLE terms), each with a **Z3 proof**
   discharged before the rule ships: `(extend_u(a) | (extend_u(b)<<k)) & mask → a`,
   `(...)>>k → b`, `wrap_i64(extend_i32_u(x)) → x`. Then DCE drops the carrier+pack.
3. const dedup/hoist.

See also the synthetic `../seam_sroa_decide.wat`.
