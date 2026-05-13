# Gale deep-scan — optimization opportunities beyond the v0.5.0 pattern analysis

Date: 2026-05-03
Source: `/Users/r/git/pulseengine/z/gale/` (read-only)
Loom: `/Users/r/git/pulseengine/loom/` (read-only)
Prior art: `docs/research/gale-v0.5.0/source-pattern-analysis.md` (8 patterns),
`docs/research/gale-v0.5.0/wasm-opt-gap-analysis.md` (7 picks),
`docs/research/gale-v0.4.0/measurement-report.md` (baseline +6.3 % code regression).

Method: byte-level diff of the two real wasm artifacts in
`benches/engine_control/build/modules/gale/` (opt.wasm =
wasm-opt-Oz baseline, loom.wasm = v0.4.0-era LOOM output),
cross-referenced against `ffi/src/lib.rs` (12 304 LoC),
`ffi/src/coarse.rs` (586 LoC), and the verified kernel modules.
`wasm-opt` and `cargo build` were unavailable in this sandbox;
deltas below are projected from byte counts, not re-measured.

What this report adds: eight new opportunities the source-pattern
doc missed, per-primitive notes, ten ranked Verus clauses for
LOOM axiom ingestion.

---

## 1. Opportunity catalogue (new, with file:line + LOOM emit)

### A. Stale const-drop residue from v0.6.0 dead-store pass
**Wasm:** `gale_ffi.loom.wasm` func 0 line 49-50 (`i32.const -22 ;
drop`), func 19 line 393-394 (`i32.const -16 ; drop`), func 24
line 458-459 (`i32.const -1 ; drop`). **Loom:**
`loom-core/src/lib.rs:7650` `peephole_const_drop` runs inside
`vacuum`, but the artifact was produced pre-v0.6.0
(`eliminate_dead_locals`/`eliminate_dead_stores` at lib.rs:9996,
10228 ship in main). **Action:** confirm pipeline order is
`eliminate_dead_locals → eliminate_dead_stores → vacuum`, then
add a second vacuum sweep *after* the dead-store passes — or
fold `peephole_const_drop` into `DeadStoreApplier` (lib.rs:10486).
**Emit:** `i32.const N ; drop` disappears; the orphaned `(local
i32)` decl also disappears once the write is gone. **Payoff:**
-6 to -9 bytes here; v0.4.0 measurement flagged this as the
canonical regression source.

### B. Hoist-past-trap is unsafe in the v0.4.0-era LOOM artifact
**Wasm:** `gale_ffi.loom.wasm` func 16 line 322-344 (sem_count_take),
func 21 line 412-447 (spinlock_acquire_nested), func 25 line
487-513 (timer_expire). LOOM has hoisted store instructions
(`local.get 0 ; local.get 1 ; i32.const 1 ; i32.sub ; i32.store` in
func 16) *above* the null-pointer check. Source
(`ffi/src/lib.rs:348-369`) does null-check then deref; hoisting
the store past the null check would null-deref on a null pointer.
**Action:** whichever pass produced this hoist (likely a
`simplify_locals`-class motion at lib.rs:7689-7770) must treat
`local.get $p ; i32.eqz ; if ... return` as a *trap-gating*
edge, not a normal merge point. Add a "crosses null-checked
pointer deref" rejection rule. **Emit:** writes stay in the
success arm. **Payoff:** correctness; unblocks LOOM on the v2
coarse API (`coarse.rs:90-138`) which is universally
null-check-then-deref.

> Caveat: this artifact predates v0.6.0; the bug may already be fixed.
> If so, add a regression-suite entry against this byte sequence.

### C. `Result<u32, i32>` returned via i64 pack/unpack — degenerate when matched in-function
**Source:** `ring_buf.rs:129,190,237,346` (init/put/get/peek_at),
`sched.rs:649-787` (six lifecycle fns returning
`Result<SchedThreadState, i32>`). **Pattern:** Rust packs
`Result<u32,i32>` into an i64 (tag low, payload high) via
`i64.extend_i32_u ; i64.const 32 ; i64.shl ; ... ; i64.or` — see
`gale_ffi.opt.wasm:117-161` (func 4 give_decide). **LOOM emit:**
recognise pack + matching unpack
(`i64.const 32 ; i64.shr_u ; i32.wrap_i64`) and fuse when they
appear in the same function. Pack-then-unpack of a freshly-built
`Result` (e.g. `Ok(claim)` immediately matched in `claim_decide`)
is a pure peephole: ~10 instructions → 0. **Payoff:** ~9 bytes in
func 4 alone; module-wide ≈ 40-60 bytes across the 10 callers.

### D. `checked_add` on a value with a Verus-trusted upper bound
**Source:** `ffi/src/lib.rs:8841-8854` (bitarray_alloc_validate),
`8875-8888` (bitarray_region_check) — both use
`offset.checked_add(alloc_nbits) match { Some(end) if end <=
num_bits => OK, _ => EINVAL }`. `checked_add` lowers to
add+overflow-branch producing `Option<u32>`; but the guard
`alloc_nbits <= num_bits` (line 8846) plus `num_bits < u32::MAX`
makes the overflow branch unreachable. **LOOM emit:** ingest
clause #4 below, replace `checked_add` with raw `i32.add` +
range check. **Payoff:** 5-8 bytes per site (×2), drops one
`Option`-tag local.

### E. Ring-buffer `next_idx` runtime branch where capacity is const-propagatable
**Where (source):** `ring_buf.rs:161-173 next_idx` (`if idx + 1 <
self.capacity { idx + 1 } else { 0 }`). When `RingBuf::new(cap)`
is called with a const, the invariant `self.capacity == cap` is a
trusted Verus fact; if `cap` is a power of two, `(idx + 1) % cap`
folds to `(idx + 1) & (cap - 1)`. (The pure form
`bundle_index`'s `bit / 32`, `bit % 32` is *already* folded —
`gale_ffi.opt.wasm` func 1 lines 75-86 — and LOOM preserves
it; no new opportunity there.) **Payoff:** 3 bytes per site, large
multiplicative if any user instantiates with const cap.

### F. Trailing `select` chain over a 3-state enum is a candidate for `directize`-style fold
**Where (wasm):** `gale_ffi.opt.wasm` func 0 lines 56-73, func 2
lines 88-110, func 8 lines 200-210, func 11 lines 259-268,
func 13 lines 274-289, func 15 lines 297-307. Every one of these
ends in a chain like:
```
i32.const -22 ; i32.const 0 ; <pred1> ; select ; <pred2> ; select
```
**Pattern.** The source is a 3-way decision over two booleans
(`if a { -22 } else if b { 0 } else { -22 }`) which Rust lowers
to two stacked `select` operations. The first `select` returns
`-22` or `0`; the second picks between that result and `-22`
again. **LOOM emit.** Recognize `select c1 (κ, x) ; select c2
(., κ)` where κ is a known constant repeated in both inputs and
fold to `select (c1 ∨ c2, κ, x)` — single select, one i32.or,
saves one select (1 byte). **Payoff:** -1 byte per site, 6 sites
visible in this module alone. Pure peephole, no axioms needed.

### G. Conditional-store-then-no-op → unconditional add-with-predicate
**Source:** `coarse.rs:101-105 gale_sem_give_v2` codes
`if s.count < s.limit { s.count += 1; }`. The pure-arithmetic
variant `sem_count_give` (sem.rs:285-296) is already folded by
LLVM to `count + (count < limit)` (`gale_ffi.opt.wasm:290-296`,
elegant 5-instr). But the **memory-store-in-arm** variant in
coarse.rs preserves the branch — wasm-opt's
`OptimizeInstructions` folds it, LOOM does not. **LOOM emit:**
recognise `if c { *p = $p + 1; }` ≡ `*p = $p + (c as i32)` when
`c` is a pure boolean on the same address's value. **Payoff:**
4-6 bytes × ≥ 6 saturating counters (sem give v2, futex,
ring_buf put_n/get_n, stack push, event post) ≈ 25 bytes.

### H. Dead default-i64-store inside a block (i64 cousin of source-pattern #2)
**Wasm:** `gale_ffi.loom.wasm` func 4 lines 120-164
(`gale_k_sem_give_decide`): `i64.const 1 ; local.set 3 ; block
@1 { i64.const 0 ; local.set 3 ; block @2 ... }`. The default
`i64.const 1 ; local.set 3` is dead — both block paths
overwrite local 3 before any read. wasm-opt-Oz eliminates it; LOOM
preserves both stores. **Action:** `eliminate_dead_stores`
(lib.rs:10228) should catch this once Block-entry join handles
"store dominated by store-with-same-target before any read." The
doc-comment at lib.rs:10206 admits Loop handling is conservative;
the same conservatism appears to bleed into Block here.
**Payoff:** 3 bytes per site; wasm-opt-Oz precedent confirms safety.

---

## 2. Per-primitive notes (file:line + transformation)

### `sched.rs` (scheduler primitives)
- `sched.rs:271-278 prio_cmp`: returns `i64`; called once from
  `next_up_smp` line 430. **Inline → fold i64 subtraction to direct
  field compare.** `is_higher_than` (`priority.rs:62-70`) is the
  cleaner version — when the caller only checks the sign,
  `i64::is_negative` ≡ `lhs < rhs` (i32 unsigned compare here because
  priority < MAX_PRIORITY = 32). Drop the i64.
- `sched.rs:285-300 should_preempt`: three call sites, every one with
  at least one boolean compile-time-constant (callers know whether
  `chosen.is_metairq` is statically false on the non-MetaIRQ path).
  **Specialise** the three constant-arg variants — body reduces to
  `swap_ok || !current_is_cooperative` (one boolean expr) in the
  common case.
- `sched.rs:649-787` (six lifecycle funcs): every one is `match state {
  S1 => Ok(T1), _ => Err(EINVAL) }`. **CSE the failure arm.** All six
  end with the same `Err(EINVAL)` constructor — synthesise a single
  shared error helper, all six call it tail-call-style.

### `wait_queue.rs` (blocking primitives)
- `wait_queue.rs:292-332 pend`: insertion search; invariant
  `0 <= i <= self.len ≤ MAX_WAITERS = 64` (line 297-298) means
  trip count is statically bounded. **Unroll-with-threshold +
  SIMD compare** on the 64-slot array.
- `wait_queue.rs:339-365` shift-right loop: pure memmove; invariant
  on line 354 (`entries[j..len+1) = old entries[j-1..len)`) is
  exactly the post-condition LOOM needs to **swap the loop for
  bulk-memory `memory.copy`** (target features include
  `bulk-memory` per artifact line 526).

### `event.rs` (event bitmaps)
- `event.rs:80-101 post`, `103-119 set`, `144-153 clear`: each
  body is a pure bit-vector formula with a matching `by
  (bit_vector)` proof one line below. **Ingest each `by
  (bit_vector)` assert as an ISLE rewrite rule** (lines 98, 213,
  222, 242, 250, 258). Round-trips like `set_masked(x, m);
  clear(m)` collapse to `set_masked(0, m); ...`.

### `futex.rs` (futex wait/wake)
- `futex.rs:290-344 wake`: loop invariant `woken == i` (line 327)
  means `woken` is a redundant copy of the iteration variable.
  **Coalesce `woken` into the loop counter** — every post-loop
  read of `woken` becomes a read of `count` (the loop bound) or a
  match-arm boolean.
- `futex.rs:288` ensures `result.woken <= MAX_WAITERS = 64`
  (a 7-bit value). Pass this bound to callers; eliminate
  downstream range checks.

### `cpu_mask.rs` (CPU affinity bitmaps)
- `cpu_mask.rs:104-143 cpu_mask_mod`: three early-exit arms each
  reconstruct `CpuMaskResult { mask: current_mask, error: EINVAL
  }`. **Struct-CSE** (not scalar) — wasm-opt misses this; LOOM's
  cost-gated CSE (v0.5.0 hot-fix) should catch it once the cost
  model recognises struct constructors.
- `cpu_mask.rs:152-186 cpu_pin_compute`: the 32-case `by
  (bit_vector)` disjunction (lines 173-181) is a precomputed
  switch table. When `cpu_pin_compute(c)` feeds
  `validate_pin_mask` (line 81), the conjunction `mask != 0 &&
  (mask & (mask-1)) == 0` collapses to `mask != 0` (provably true
  for any power of two).

### `ring_buf.rs` (lock-free FIFO)
- `ring_buf.rs:590-641 claim_decide`: nested if-else, pure
  arithmetic. **Two stacked `select` ops** replace both
  branches; Verus ensures (lines 606, 608) prove overflow-free.
- `ring_buf.rs:691-696 size_get_decide`: one `i32.sub` wrapping
  helper called only from `gale_ring_buf_size_get` (func 12 lines
  269-273). **Inline-and-drop**; the FFI shim is already the
  same single sub.

### `sem.rs` (counting semaphores)
- `sem.rs:94-109 take_decide` and `70-86 give_decide`: 3-arm
  enums returned by-value, every caller destructures immediately
  (`ffi/src/lib.rs:469-493`). **Inline + match-fusion** collapses
  the enum-construct + match into two `select`s.
- `sem.rs:265-296 give`: see Opportunity G for the
  saturating-add-with-store fold.

### `atomic.rs` (load/store/CAS wrappers)
- `atomic.rs:218-239 cas`: trusted axioms #6 and #7 below collapse
  CAS to `i32.eq + select` for callers with known expected/current.
- `atomic.rs:271-290 inc`/`dec`: trivial wrappers around `add(1)`/
  `sub(1)`. Known inline candidates; v0.6.0's i64 fix now also
  enables `AtomicU64` if/when gale uses one.

### `priority.rs` (priority comparison)
- `priority.rs:62-81 is_higher_than`/`is_higher_or_equal`:
  one-line wrappers around `<`/`<=`. Each is called from ≥ 10
  sites in `sched.rs`/`wait_queue.rs`/`mutex.rs`. LOOM rejects
  the inline today because `value < MAX_PRIORITY` is a Verus
  precondition, not a code guard. **Ingest clause #2 below;
  unblock 20+ call sites.**

---

## 3. Verus-clause ingestion candidates (top 10)

LOOM can treat verified `requires`/`ensures` as trusted axioms.
The clauses below are concrete, narrow, and each unlocks a specific
LOOM rewrite. Sorted by payoff × likelihood-of-firing.

| # | file:line | Clause text (paraphrased) | LOOM transformation it unlocks |
|---|---|---|---|
| 1 | `ring_buf.rs:651` | `result == (size <= head.wrapping_sub(tail))` | Inline `finish_decide` into FFI wrapper, drop `i32` boolean materialisation. |
| 2 | `priority.rs:27` | `self.value < MAX_PRIORITY` (= 32) | Drop range-check on every `Priority::get` site (~20). |
| 3 | `priority.rs:67` | `is_higher_than: result == (self.value < other.value)` | Inline two-line wrapper → unsigned `i32.lt_u`. |
| 4 | `ffi/src/lib.rs:8851` | `alloc_nbits <= num_bits` (in `Some(end) if end <= num_bits` guard) | Replace `checked_add` with raw `i32.add` + bound check (#D above). |
| 5 | `event.rs:213` | `(events \| new_events) & events == events` (by bit_vector) | Eliminate redundant `events.set & old_check` sequences in poll/futex callers. |
| 6 | `atomic.rs:223-226` | `old.val == expected ==> success && self.val == new_value` | Specialise CAS success path: collapse `if cas(x,y) { ... }` to direct store when `x` provably matches. |
| 7 | `atomic.rs:228-231` | `old.val != expected ==> !success && self.val == old.val` | Eliminate dead "rollback" arm after CAS failure (no memory change to propagate). |
| 8 | `futex.rs:288` | `result.woken <= MAX_WAITERS = 64` | Replace `u32` bound checks with constant `64` in callers; enables `u8` packing in `WakeResult.woken`. |
| 9 | `cpu_mask.rs:115` | `result.error == OK ==> result.mask == ((current\|enable) & !disable)` | Mask-formula CSE across the three EINVAL early-exits (#cpu_mask.rs note). |
| 10 | `sem.rs:266-269` | `wait_q.len_spec() == 0 && count < limit ==> count' == count + 1` | Drop the `if count != limit` branch in `gale_sem_give_v2` (`coarse.rs:101`): provable from caller-side state. |

Selection rationale: each row gives LOOM (a) a Z3-discharged fact
(no re-proof required), (b) a syntactic call-site pattern that the
verifier emits today, and (c) a payoff visible in the wasm bytes.
Rows 1, 2, 3, 4 fire in every FFI artifact gale produces; rows 5-10
fire only when the corresponding primitive is enabled in cargo
features.

---

## 4. Top 5 picks for the v0.7.0 sprint

Ranked by `(payoff bytes × call-site frequency) / engineering weeks`.
Compared to the v0.5.0 wasm-opt-gap list which focused on
adding-passes-LOOM-doesn't-have, this list focuses on
**fixing-passes-LOOM-has-but-misuses**, plus two cheap new wins.

1. **Second vacuum sweep after dead-store passes (Opportunity A).**
   Estimated: 30 LOC, 0.2 weeks, Trivial risk. Payoff: closes a known
   gap in the v0.6.0 dead-store work — three confirmed sites in the
   gale artifact alone. Owner already knows the codebase
   (`peephole_const_drop` at lib.rs:7650). **Do this first.**

2. **Trap-aware hoist guard (Opportunity B).** Estimated: 200 LOC,
   1.0 week, Moderate risk (correctness-critical). Payoff:
   correctness on the v2 coarse API + every future null-check-then-
   deref pattern. Without this, LOOM cannot be enabled on
   `coarse.rs` at all. **Do this second — it gates the whole v2 API
   on/off.**

3. **Trailing-select chain collapse (Opportunity F).** Estimated:
   100 LOC, 0.5 weeks, Easy. Pure peephole, ISLE-friendly, zero
   Verus dependency. Payoff small per-site (-1 byte) but visible in
   6 of 27 module functions = -6 bytes baseline. **Do this third —
   trivial and bench-friendly.**

4. **Result-pack/unpack peephole (Opportunity C).** Estimated:
   400 LOC, 1.5 weeks, Easy. Recognise the 5-instruction
   `i64.extend ; i64.shl ; ... ; i64.or` pack and its inverse;
   fuse pack-then-unpack inside one function. Payoff: ~50 bytes
   module-wide. **Do this fourth — biggest scalar win that doesn't
   need Verus.**

5. **Verus-clause ingestion MVP, clauses #1-4 from the table.**
   Estimated: 800 LOC (clause parser + axiom table + ISLE binding),
   3 weeks, Moderate. Payoff: unblocks ~20 call sites in priority/
   bitarray helpers; this is also the strategic move toward LOOM's
   "trusted-axiom-driven optimization" mission. **Do this fifth —
   slow but high-value research lever.**

Picks 6-8 deferred (cumulative effect <0.5 % on gale code section):
Opportunity D (`checked_add` reduction — already partially folded
by LLVM/wasm-opt), Opportunity E (RingBuf cap specialisation —
narrow, depends on call-site constants), Opportunity G (saturating-
store-in-arm — overlap with the simplify-locals sinker already
planned in `wasm-opt-gap-analysis.md` pick #2).

Cumulative projection (picks 1-5): -3 to -5 % code-section delta vs.
current LOOM output on `gale_ffi.opt.wasm`. Combined with the v0.5.0
hot-fix already merged (`fix(cse): cost-aware dedup gate`,
commit afc9318), that closes the +6.3 % regression and likely puts
LOOM at parity with `wasm-opt -O3` on this workload, with picks 1
and 2 also paying off on every kernel-FFI-shaped workload outside
gale.

---

## Sources

- Gale source: `/Users/r/git/pulseengine/z/gale/src/` and `.../ffi/src/`
- Gale wasm artifacts: `.../benches/engine_control/build/modules/gale/{gale_ffi.opt.wasm, gale_ffi.loom.wasm}` (1941 / 1913 bytes)
- LOOM pipeline: `loom-core/src/lib.rs` lines 6713 (eligibility), 7506 (vacuum), 7650 (peephole-const-drop), 9996 (eliminate_dead_locals), 10228 (eliminate_dead_stores), 10486 (DeadStoreApplier)
- Prior pattern catalogue: `docs/research/gale-v0.5.0/source-pattern-analysis.md`
- Prior pass-gap catalogue: `docs/research/gale-v0.5.0/wasm-opt-gap-analysis.md`
- Baseline measurement: `docs/research/gale-v0.4.0/measurement-report.md`

Wasm tooling notes: `wasm-opt -O3` could not be invoked in this
sandbox (permission-denied on the binaryen subprocess). Re-running
wasm-opt on the current loom-built artifact would tighten the
numeric estimates in section 4 — flagged for the v0.7.0 sprint
kickoff.
