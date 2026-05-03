# Gale Pattern Analysis for LOOM Optimization Research

**Source tree:** `/Users/r/git/pulseengine/z/gale/src/` (Verus-verified Rust
model of Zephyr's kernel-scheduler primitives — sem / mutex / sched /
wait_queue / timeout / event / futex / cpu_mask / atomic).

**Method:** read-only inspection of the largest scheduler-relevant files
(`sched.rs` 916 LoC, `mutex.rs` 560, `sem.rs` 539, `wait_queue.rs` 478,
`timeout.rs` 709, `event.rs`, `futex.rs`, `cpu_mask.rs`, `priority.rs`,
`poll.rs`, `atomic.rs`, `ipi.rs`).

**Module-wide quick facts**

| metric | count | comment |
|---|---|---|
| `requires` clauses | 627 | trusted preconditions |
| `ensures` clauses | 980 | post-conditions LOOM can ingest |
| `decreases` clauses | 24 | bounded-loop ranking functions |
| `match` statements (8 hot files) | 29 | many are 7-arm enum dispatches |

The repository is *unusually* dense in (a) closed-set enum dispatches, (b)
`Result<…, EINVAL>` early-return guards, and (c) bounded for-loops over
fixed-size arrays (`[Option<Thread>; 64]`, `MAX_CPUS = 16`,
`MAX_PARTITIONS`, `MAX_POLL_EVENTS`). Every property below is grounded in
that structure.

---

## 1. Closed-set state-machine dispatch (br_table candidate)

**Pattern.** A 7-arm `match` on a small `#[repr(u8)]`-style enum where every
arm is either `Ok(NewState)` or `Err(EINVAL)`. The discriminant is dense
(0..6), so rustc/LLVM emits a `br_table` (jump table). Many of these
functions are then composed into outer state machines.

**Examples (file:line).**
- `src/sched.rs:649-657` `sched_suspend` — 7 arms, 6 of them `Err(EINVAL)`.
- `src/sched.rs:669-677` `sched_resume` — 7 arms, only `Suspended` succeeds.
- `src/sched.rs:721-729` `sched_sleep` — only `Running` succeeds.
- `src/sched.rs:740-748` `sched_wakeup` — only `Sleeping` succeeds.
- `src/sched.rs:759-767` `sched_pend` — only `Running` succeeds.
- `src/sched.rs:779-787` `sched_unpend` — only `Pending` succeeds.

**Two-axis pair-match (state×event).**
- `src/sched.rs:530-543` `is_valid_transition` — `match (from, to)` over
  `(ThreadState, ThreadState)` (4×4 grid).
- `src/sched.rs:592-630` `sched_is_valid_transition` — `match (from, to)`
  over `(SchedThreadState, SchedThreadState)` (7×7 = 49 cases, encoded as
  ~25 explicit arms + wildcard).

**Why it matters for LOOM.**
1. *Constant-folding through dispatch.* A caller invoking `sched_resume`
   on a known-`Suspended` value should fold to `Ok(Ready)` — single move.
   Today this requires an inter-procedural enum-discriminant tracker.
2. *Switch-table → perfect-hash.* Where >50 % of arms are
   `Err(EINVAL)`, LOOM can rewrite to `if state == OnlyOk { Ok(...) } else
   { Err(EINVAL) }` — kills the indirect jump entirely. Branch-prediction
   wins on the sparse-success pattern that dominates here.
3. *FSM equivalence proofs.* The pair-match form (`(from, to)`) is the
   canonical example LOOM/ISLE rule synthesis was designed for. Every
   `_ => false` is a refutable rule the optimizer can prove sound from the
   `ensures` clause directly above it (e.g. `from === Dead ==> !result`).

**Expected payoff.** Concrete: 10-30 % cycle reduction on `sched_*`
hot paths (one less jump-table indirect, dead code eliminated). Strategic:
this is the cleanest source-level evidence I've seen for the
"trusted-axiom-driven branch elimination" idea in
`docs/research/novel-optimization-ideas.md`.

---

## 2. Default-then-override (the v0.4.0/v0.5.0 hoist-guard pattern)

**Pattern.** `let mut x = DEFAULT; ...; if cond { x = OVERRIDE }; x`.
LOOM v0.4/v0.5 added hoist guards specifically for this idiom because
LLVM frequently fails to forward-propagate when the override is conditional
on `Option::is_some()` or a state-machine match.

**Examples.**

1. **`src/sched.rs:404-444` `next_up_smp` — the canonical case.**
   ```
   let mut thread: Option<Thread> = runq_best;          // default
   if let Some(mirqp) = cpu_state.metairq_preempted { ... thread = Some(mirqp); ... }
   let candidate = match thread { Some(t) => t, None => cpu_state.idle_thread };
   let mut chosen = candidate;                           // default
   if current_is_active { ... chosen = current; ... }
   if !should_preempt(...) { chosen = current; }
   ```
   Two stacked default-then-override variables (`thread`, `chosen`) feeding
   into a single returned `SmpSchedOutcome`. Exactly the shape that
   motivated the hoist guards.

2. **`src/sched.rs:183-208` `RunQueue::add` insertion search.**
   ```
   let mut insert_pos: u32 = self.len;   // default = "append at end"
   let mut i: u32 = 0;
   let mut found: bool = false;
   while ... { if thr_pri < entry_pri { insert_pos = i; found = true; } ... }
   ```
   `insert_pos` defaults to `self.len`, overridden when the loop finds a
   spot. The companion `WaitQueue::pend` at `src/wait_queue.rs:292-332`
   is byte-for-byte the same pattern.

3. **`src/cpu_mask.rs:104-143` `cpu_mask_mod`.**
   ```
   if is_running { return CpuMaskResult { mask: current_mask, error: EINVAL }; }
   let new_mask: u32 = (current_mask | enable) & !disable;
   if new_mask == 0 { return CpuMaskResult { mask: current_mask, error: EINVAL }; }
   if pin_only && (new_mask & (new_mask - 1)) != 0 { return CpuMaskResult { mask: current_mask, error: EINVAL }; }
   CpuMaskResult { mask: new_mask, error: OK }
   ```
   The `mask: current_mask` field is "default", `error: OK/EINVAL` is the
   override. LLVM partly handles this via SROA but the `Result`-shaped
   struct often forces a stack slot.

4. **`src/futex.rs:290-344` `wake`.**
   `let mut woken: u32 = 0; ... match thread { Some(t) => { woken = 1 } None => {} }`
   — classic counter default + match-arm increment.

5. **`src/ipi.rs:109-141` `compute_ipi_mask`.**
   `let mut mask: u32 = 0u32; ... if cond { mask = mask | bit; }` — bitmap
   build via OR-into-default.

**Why it matters.** This is *the* pattern LOOM already targets; gale gives
LOOM a clean, dense, real-world benchmark suite. Every example above has
a Verus `ensures` clause that pins the final value as a function of inputs
— LOOM can use those clauses to prove the override is unconditional in
specific call sites and collapse the assignment.

---

## 3. Verus-verified bounded loops (full-unroll / k-induction targets)

24 `decreases` clauses across the kernel. All ranking functions are
`MAX_CONST - i` or `len - i` with `MAX` as a `pub const u32`.

**Examples.**

| file:line | bound | iterations |
|---|---|---|
| `src/wait_queue.rs:206`, `319`, `359`, `437` | `MAX_WAITERS = 64` | ≤ 64 |
| `src/sched.rs:197` | `MAX_RUNQ_SIZE = 64` | ≤ 64 |
| `src/poll.rs:558,577,598` | `MAX_POLL_EVENTS` | ≤ 32 (typical) |
| `src/mem_domain.rs:202,306,387,460,478` | `MAX_PARTITIONS` | ≤ 16 |
| `src/device_init.rs:329` | `dev.num_deps - i` | ≤ small |
| `src/ipi.rs:121` | `num_cpus - idx`, `MAX_CPUS = 16` | ≤ 16 |
| `src/futex.rs:329` | `count - i`, `count ≤ MAX_WAITERS = 64` | ≤ 64 |

**Why it matters for LOOM.**
- *Full unroll candidates.* Every loop with `decreases MAX_CPUS - i` and
  `MAX_CPUS = 16` is a guaranteed-bounded, fixed-trip loop. LOOM can fully
  unroll without runtime check (the `requires` clause already proved the
  bound) — converting a loop body into a sequence of 16 SIMD-friendly ops.
- *k-induction-friendly.* Loops carrying invariants like
  `forall|k: int| 0 <= k < i ==> entries[k].is_some()` (e.g.
  `src/wait_queue.rs:194-200`) are perfect input to bounded SMT-based
  loop verification. The Verus invariants *are* the k-induction
  hypothesis.
- *Auto-vectorization.* `runq_shift_left` / `runq_shift_and_insert`
  (`src/sched.rs:80-105`, marked `#[verifier::external_body]`) are pure
  array shifts on `[Option<Thread>; 64]` — LOOM can replace with
  `memmove` / SIMD shuffle once it ingests the bound.

**Expected payoff.** Big — 2-5× on the wait-queue / runqueue ops if
unrolled + vectorized. These are the absolute hottest paths in any
RTOS kernel.

---

## 4. Tail-call / dispatch-only `match`

**Pattern.** `match enum { V1 => f(), V2 => g(), V3 => h(), ... }` where
every arm is a single function call with the same signature. LLVM should
emit jump-threaded tail calls but often fails when arms construct
different `Ok(...)` variants.

**Examples.**

- `src/sched.rs:649-657`, `669-677`, `721-729`, `740-748`, `759-767`,
  `779-787` — every `sched_*` lifecycle function has the form
  `match state { S1 => Ok(T1), _ => Err(EINVAL) }` where each arm is a
  single-expression constructor. Six functions, six identical shapes —
  the classic case for *cross-function rule synthesis* (LOOM ingests one
  rule, applies to all six).

- `src/event.rs:296-336` `wait_decide` — three-way dispatch
  (`Matched` / `Pend` / `Timeout`), each arm builds a `WaitDecideResult`
  struct. The struct constructions share the `matched_events: 0` tail —
  LOOM's struct-field constant-folding should collapse two of three arms.

- `src/sched.rs:318-321`, `421-424`, `510-513` — three sites with the
  *exact same* idiom `match opt { Some(t) => t, None => fallback }`.
  Identical AST → ideal for caching one optimization recipe.

**Payoff.** Smaller per-site (a few cycles) but accumulates because these
helpers are called from every kernel object operation.

---

## 5. Cross-function leaf-inlining + constant propagation

**Tiny leaf functions called from one site.**

- `src/sched.rs:271-278` `prio_cmp(a,b) -> i64` — three-line wrapper around
  `i64` subtraction, called only from `next_up_smp`
  (`src/sched.rs:430`) where one operand is `current.priority`. Inline +
  fold = direct `priority` field comparison.

- `src/sched.rs:285-300` `should_preempt` — pure function of three bools,
  called twice in `next_up_smp` and once in `update_cache`. Three call
  sites, each with at least one boolean known at the call site (e.g.
  `chosen.is_metairq` is often statically `false` for non-MetaIRQ paths).
  Specialize → branch elimination.

- `src/sched.rs:483-494` `update_metairq_preempt` — called once. Inline.

- `src/priority.rs:62-70` `is_higher_than` — `self.value < other.value`
  wrapped in a method. Called from many sites; the `requires
  self.inv()` clause guarantees the field read is in-bounds — LOOM can
  drop it.

- `src/atomic.rs:271-279`/`282-290` `inc`/`dec` — wrappers around
  `add(1)`/`sub(1)`. Two-line shims; *every* increment in the kernel goes
  through them.

**Constant-argument propagation.**

- `compute_ipi_mask` (`src/ipi.rs:88-142`) is called with
  `MAX_CPUS = 16` in production. The loop bound `num_cpus <= max_cpus <=
  MAX_CPUS <= 32` means the inner `1u32 << idx` shift is always in range —
  LOOM can specialize the function for `max_cpus = 16` and unroll.

**Payoff.** Each individually small, cumulatively large because these
helpers sit on every fast path.

---

## 6. Bit-manipulation / mask arithmetic

**Pattern.** Sequences of `&`/`|`/`!`/`<<` that are provably equivalent to
simpler forms once their precondition is known.

**Examples.**

- `src/cpu_mask.rs:66-68` spec: `(current | enable) & !disable`. The
  module proves a power-of-two property
  `(m & (m-1)) == 0` (line 83) — LOOM can use this as an
  axiom whenever a mask flows from `cpu_pin_compute`.

- `src/event.rs:144-152` `clear`: `events &= !clear_events`. Combined
  with `post`'s `events |= new_events` (line 82-ish) the module gives
  *complete* monotonicity / idempotence proofs (`events.rs:208-258`).
  Every Verus `by (bit_vector)` proof in this file is an axiom LOOM can
  ingest:
  - `(events | new_events) & events == events` (post-monotonic)
  - `value & !value == 0u32` (set/clear roundtrip)
  - `(events | new_events) | new_events == events | new_events` (idempotent)

- `src/cpu_mask.rs:81-86` `validate_pin_mask`: `mask != 0 && (mask &
  (mask - 1)) == 0` — power-of-two test, hardware popcount candidate.

- `src/ipi.rs:127-134` mask construction inside an unrolled loop —
  `mask |= bit; bit = 1 << idx;` is exactly the pattern that compiles to
  PEXT/PDEP on x86 and `bit-build` on ARMv8.2+.

**Payoff.** ISA-specific (popcount, PDEP, single-cycle bit-set) — but
LOOM's strategic goal of *ingesting verified bit-vector axioms as
optimization rewrite rules* lines up perfectly. Each `assert(...) by
(bit_vector)` in gale is, modulo translation, an ISLE/eqsat rewrite
rule.

---

## 7. State-machine `match (state, event)` (LOOM ISLE rule-synth target)

The premier example in the codebase:

- `src/sched.rs:592-630` `sched_is_valid_transition(from, to)` — pure
  `match (from, to) { ... }` over `(SchedThreadState, SchedThreadState)`,
  7 states each. The `ensures` clause `from === Dead ==> !result` (line
  590) is a *trusted axiom* LOOM can use: any caller path that reaches
  this with statically-known `from = Dead` simplifies to `false` and
  prunes the entire downstream branch.

- `src/sched.rs:530-543` `is_valid_transition` — same shape, smaller FSM
  (4 states).

- `src/event.rs:289-336` `wait_decide(events, desired, wait_type,
  is_no_wait)` — effectively `match (wait_type, condition_met,
  is_no_wait)` collapsed into nested `if`s. The `ensures` clauses
  (lines 304-310) prove `matched_events == 0` on the non-Matched paths —
  LOOM can hoist that constant out.

**Why it matters.** LOOM's Wave 2 ISLE rule verification work was about
precisely this kind of `match (s, e)` table. Gale is the test bed.

---

## 8. Verus annotations as ingestable axioms

`requires` (627), `ensures` (980), and `decreases` (24) clauses are
machine-checked by Verus + Z3. LOOM can ingest them as *trusted* axioms
without re-proving anything.

**Highest-value annotations (sample).**

| file:line | clause | what LOOM gains |
|---|---|---|
| `src/sched.rs:312-316` | `runq_best.is_none() ==> result === Idle` etc. | branch elimination on `next_up` calls |
| `src/sched.rs:589-590` | `from === Dead ==> !result` | dead-code on FSM paths |
| `src/sem.rs:265-273` | `wait_q.len_spec() == 0 && count < limit ==> count' == count + 1` | scalar replacement for the `count+1` path |
| `src/timeout.rs:316-339` | full `Ok((new_tick, fired))` pre/post | timeout-fast-path inlining |
| `src/cpu_mask.rs:111-122` | `error == OK ==> mask == ((current\|enable) & !disable)` | mask-arith folding |
| `src/event.rs:213,220,242,250` | bit-vector lemmas (`by (bit_vector)`) | direct rewrite-rule ingestion |
| `src/futex.rs:280-288` | `result.woken <= MAX_WAITERS` | bound propagation into callers |
| `src/atomic.rs:222-231` | CAS success/failure semantics | lock-free fast-path specialization |

This is the largest, most concentrated source of verified pre/post-
conditions I'm aware of in the LOOM benchmark space — strictly more
machine-readable than C-level Frama-C ACSL in upstream Zephyr.

**Payoff.** Strategic, not numeric: gale lets LOOM measure what
fraction of axioms actually trigger optimizations, and how often
LLVM-only baselines miss the same opportunities. This is the
primary research lever.

---

## Summary table

| # | Pattern | Sites (approx) | Payoff |
|---|---|---|---|
| 1 | Closed-set FSM `match` (br_table → const-fold) | ~10 in sched.rs alone | high |
| 2 | Default-then-override (LOOM v0.4/v0.5 hoist) | ~7 hot sites | high (already targeted) |
| 3 | Verus-bounded loops (full-unroll + SIMD) | 24 `decreases` | very high on wait/run queues |
| 4 | Tail-call dispatch `match` | ~12 | medium |
| 5 | Leaf-inline + const-prop | ~8 leaf fns | medium-high cumulative |
| 6 | Bit-mask axiom ingestion | events/cpu_mask/ipi | ISA-dependent, strategic |
| 7 | `match (state, event)` FSM | 2 large + 1 medium | high (LOOM ISLE target) |
| 8 | Verus pre/post as trusted axioms | 1607 clauses | strategic — primary lever |

## Recommended next steps for LOOM

1. **Pick `sched.rs` lines 530-787 as the canary benchmark.** Six near-
   identical `match`-on-enum FSM functions, all with `ensures` clauses,
   all called from FFI (`pub`). It's the densest source of patterns 1, 7,
   and 8 simultaneously.
2. **Wire `decreases` clauses into LOOM's loop-unrolling cost model.**
   Twenty-four loops with statically-known bounds tied to `pub const`s —
   minimal effort, large payoff.
3. **Treat `assert(...) by (bit_vector)` as ground truth** for the
   bit-mask rewrite-rule database (pattern 6). The Verus bit-vector solver
   has already discharged them.
4. **Use gale's compile output as the regression suite for hoist guards
   (pattern 2)** — `next_up_smp` alone exercises three layered defaults.

---

*Generated 2026-04-29 from gale source tree at
`/Users/r/git/pulseengine/z/gale/`. Read-only inspection; no files
modified.*
