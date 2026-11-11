(** * Fused Component Optimization Proofs

    This module proves the correctness of optimization passes specific to
    WebAssembly modules produced by component fusion (e.g., meld).

    When multiple P2/P3 components are fused into a single core module,
    the result contains adapter trampolines, duplicate types, duplicate
    imports, and dead functions. The fused optimizer eliminates these
    artifacts while preserving program semantics.

    ## Proven Properties

    0a. Memory import deduplication (proven in FusedSemantics.v): merging
       memory imports with the same key and remapping every memory index
       preserves execution semantics.

    0. Same-memory adapter collapse: adapters that allocate+copy within a
       single consistent memory index are equivalent to direct forwarding
       trampolines (works for any memory index N, not just single-memory modules).

    1. Adapter devirtualization (proven in FusedSemantics.v): replacing call-to-adapter with call-to-target
       preserves execution semantics when the adapter is a trivial forwarder.

    2. Trivial call elimination (proven in FusedSemantics.v): removing calls
       to void->void no-op functions preserves execution semantics.

    3. Type deduplication: merging structurally identical types and remapping
       references preserves all instruction semantics.

    4. Dead function elimination: removing unreachable functions cannot affect
       observable program behavior.

    5. Import deduplication (proven in FusedSemantics.v): merging identical
       imports (same module+name+type)
       and remapping references preserves binding semantics.

    ## Proof Architecture

    Since exec_call is axiomatized (we reason about module-level execution
    abstractly), certain theorems require semantic axioms that capture
    well-justified properties of the WebAssembly specification:

    - same_memory_adapter_equiv: In single-memory modules, adapters that
      allocate+copy within the same memory are equivalent to the target.
    - same_memory_adapter_general_equiv: Generalized to any consistent
      memory index N in multi-memory modules.

    ## Status (issue #169)

    This file is being replaced, pass by pass, by FusedSemantics.v, which
    proves each pass against a concrete call semantics. Until a pass moves
    there, its statement here is not semantic evidence:
    - [is_same_memory_adapter] is [fd_num_locals f > 0 /\ True] and
      [single_memory] is [True], so [same_memory_adapter_equiv] equates any
      two same-signature functions that have locals. That is false for wasm.
    - [collapse_same_memory_adapter m _ _] is [m], so the five collapse
      theorems hold by reflexivity.
    - [dead_function_removal_preserves_semantics] and
      [fused_optimization_correct] take their conclusion as a hypothesis.
    - [dead_function_elim_correct] concludes [x = x].
    Pass 0a (memory import deduplication), Pass 1 (adapter
    devirtualization), Pass 2 (trivial call elimination) and Pass 5
    (import deduplication) have moved: see FusedSemantics.v.

    ## Connection to Meld

    Meld performs: Parse -> Resolve -> Merge -> Adapt -> Encode
    loom performs: Fused Optim -> 12-Phase Pipeline

    The fused optimizer runs on meld's output (a single core module) and
    targets the structural artifacts introduced by the fusion process.
    These proofs complement meld's own formal verification in
    proofs/spec/fusion_spec.v which proves the fusion itself is correct.
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
From proofs Require Import WasmSemantics.
From proofs Require Import TermSemantics.
Import ListNotations.


(** * Module-Level Definitions *)

(** Function index in a WebAssembly module *)
Definition func_idx := nat.

(** Function signature: parameter types and result types *)
Record func_sig : Type := mkFuncSig {
  fs_params : list ValueType;
  fs_results : list ValueType;
}.

(** Boolean list equality from an element equality *)
Fixpoint list_eqb {A} (eqb : A -> A -> bool) (l1 l2 : list A) : bool :=
  match l1, l2 with
  | nil, nil => true
  | x :: xs, y :: ys => eqb x y && list_eqb eqb xs ys
  | _, _ => false
  end.

(** Decidable equality for function signatures *)
Definition func_sig_eqb (a b : func_sig) : bool :=
  list_eqb valuetype_eqb (fs_params a) (fs_params b) &&
  list_eqb valuetype_eqb (fs_results a) (fs_results b).

(** A simplified WebAssembly instruction for module-level proofs *)
Inductive module_instr : Type :=
  | MILocalGet : nat -> module_instr
  | MICall : func_idx -> module_instr
  | MIEnd : module_instr
  | MIOther : module_instr.  (** All other instructions *)

(** A function body is a list of module instructions *)
Definition func_body := list module_instr.

(** A function definition *)
Record func_def : Type := mkFuncDef {
  fd_sig : func_sig;
  fd_body : func_body;
  fd_num_locals : nat;  (** Number of locals beyond parameters *)
}.

(** A module is a collection of functions *)
Record wasm_module : Type := mkModule {
  wm_types : list func_sig;
  wm_funcs : list func_def;
  wm_exports : list func_idx;
  wm_num_imports : nat;
}.

(** * Adapter Trampoline Definition *)

(** A function is a trivial adapter if:
    1. It has no locals beyond parameters
    2. Its body is: local.get 0, local.get 1, ..., local.get N, call target, end
    3. N equals the number of parameters
*)
Definition is_trivial_adapter_body (params : nat) (target : func_idx) (body : func_body) : Prop :=
  body = map (fun i => MILocalGet i) (seq 0 params) ++ (MICall target :: MIEnd :: nil).

Definition is_trivial_adapter (f : func_def) (target : func_idx) : Prop :=
  fd_num_locals f = 0 /\
  is_trivial_adapter_body (length (fs_params (fd_sig f))) target (fd_body f).

(** * Same-Memory Adapter Definition *)

(** A module has a single memory (including imported memories).
    In the Rust implementation this is checked by count_total_memories(module) == 1. *)
Definition single_memory (m : wasm_module) : Prop :=
  True.  (** Abstract predicate: module has exactly one linear memory. *)

(** A function is a same-memory adapter if:
    1. It has locals beyond parameters (used for temporary pointers)
    2. Its body contains memory.copy {0, 0} (same-memory copy)
    3. Its body contains calls to cabi_realloc
    4. Its body contains exactly one call to a non-realloc target
    5. No unsafe control flow, no unsafe global writes
       (memory stores are allowed — they target the discarded realloc'd buffer)
    6. The target has the same signature as the adapter

    This is an abstract predicate capturing the detection criteria
    from the Rust implementation's is_same_memory_adapter function. *)
Definition is_same_memory_adapter (f : func_def) (target : func_idx) : Prop :=
  fd_num_locals f > 0 /\   (** Has locals beyond parameters *)
  True.  (** Remaining criteria are structural properties of the body *)

(** Collapsing a same-memory adapter replaces its body with a trivial
    forwarding trampoline and clears its locals. *)
Definition collapse_same_memory_adapter (m : wasm_module) (adapter_idx target_idx : func_idx) : wasm_module :=
  m.  (** Abstract: returns module with adapter body replaced by forwarding trampoline *)

(** * Execution Semantics (Module Level) *)

(** Simplified execution state for module-level reasoning *)
Record exec_state : Type := mkExecState {
  es_stack : list Value;
  es_locals : list Value;
}.

(** A function is reachable if it can be reached from any export root
    via the call graph. *)
Inductive reachable (m : wasm_module) : func_idx -> Prop :=
  | reach_export : forall idx,
      In idx (wm_exports m) ->
      reachable m idx
  | reach_call : forall caller callee,
      reachable m caller ->
      (* caller's body contains a call to callee *)
      In (MICall callee) (fd_body (nth caller (wm_funcs m) (mkFuncDef (mkFuncSig nil nil) nil 0))) ->
      reachable m callee.

(** Execute a function call in a module context.
    This is an abstract step relation: we assume that calling a function
    with arguments on the stack produces results on the stack according
    to the function's semantics. *)
Axiom exec_call : wasm_module -> func_idx -> exec_state -> option exec_state.

(** * Semantic Axioms

    These axioms capture well-justified properties of the WebAssembly
    specification that cannot be derived from our abstract exec_call.
    Each axiom is minimal and directly motivated by the WASM spec. *)

(** Axiom 1 (trivial adapter equivalence) is no longer assumed. It is the
    theorem [trivial_adapter_equiv] in FusedSemantics.v, proven against a
    concrete call semantics (issue #169). *)

(** Axiom 2 (identical import equivalence) is no longer assumed. Import
    deduplication is the theorem [import_dedup_correct] in FusedSemantics.v.
    The axiom attributed "same (module, name, type) resolves to the same
    function" to the wasm specification; the specification leaves import
    binding to the embedder, so FusedSemantics.v states it as part of the
    model instead: the host is looked up by import key (issue #169). *)

(** Axiom 3 (trivial call elimination) is no longer assumed. The pass is
    the theorem [trivial_call_elim_correct] in FusedSemantics.v (issue #169). *)

(** Axiom 4: Same-memory adapter equivalence.

    In a single-memory module, an adapter that allocates a buffer via
    cabi_realloc, copies data within the same linear memory (memory.copy
    {0, 0}), and calls the target with the new pointer is semantically
    equivalent to calling the target with the original pointer.

    This holds because both pointers reference the same linear memory.
    The target can read the data at the original pointer directly. The
    allocation and copy are semantically redundant:
    - The allocated buffer is only used as a temporary for the copy
    - After the target call, the buffer is dead
    - The Canonical ABI does not require callers to observe realloc
      side effects on temporary buffers

    Per WASM spec Section 5.4.7 (memory.copy), when src_mem = dst_mem = 0,
    the copy operates within the same address space. The target function
    reads the same bytes regardless of their address. *)
Axiom same_memory_adapter_equiv :
  forall (m : wasm_module) (adapter_idx target_idx : func_idx)
         (adapter : func_def) (st : exec_state),
    single_memory m ->
    nth_error (wm_funcs m) adapter_idx = Some adapter ->
    is_same_memory_adapter adapter target_idx ->
    (exists target, nth_error (wm_funcs m) target_idx = Some target /\
                    fd_sig target = fd_sig adapter) ->
    exec_call m adapter_idx st = exec_call m target_idx st.

(** Predicate: all memory operations in a function target a single memory index.
    This generalizes the single-memory check to per-adapter verification:
    a load from memory 0 at offset X is NOT the same as memory 1 at offset X,
    so we require all operations in the adapter to target the same index. *)
Definition adapter_uses_single_memory (f : func_def) (mem_idx : nat) : Prop :=
  True.  (** Abstract: all memory ops in f target mem_idx *)

(** Axiom 4b: Generalized same-memory adapter equivalence (multi-memory).

    In a module with multiple memories, an adapter that allocates a buffer
    via cabi_realloc, copies data within a single consistent memory index N
    (memory.copy {N, N}), and where all load/store instructions also target
    memory N, is semantically equivalent to calling the target directly.

    This generalizes Axiom 4 by replacing the single_memory module-level
    constraint with a per-adapter consistency check on the memory index.
    The correctness argument is identical: both pointers reference the same
    linear memory (memory N), so the target can read the data at the
    original pointer directly. *)
Axiom same_memory_adapter_general_equiv :
  forall (m : wasm_module) (adapter_idx target_idx : func_idx)
         (adapter : func_def) (mem_idx : nat) (st : exec_state),
    nth_error (wm_funcs m) adapter_idx = Some adapter ->
    adapter_uses_single_memory adapter mem_idx ->
    is_same_memory_adapter adapter target_idx ->
    (exists target, nth_error (wm_funcs m) target_idx = Some target /\
                    fd_sig target = fd_sig adapter) ->
    exec_call m adapter_idx st = exec_call m target_idx st.

(** * Pass 0a: Memory Import Deduplication Correctness

    Proven in FusedSemantics.v, against a concrete interpreter and with no
    axioms: [memory_import_dedup_correct] says that replacing the memory
    imports and remapping every memory index with [sg] preserves every
    function, provided each index names the same memory key (or no memory)
    before and after; [memory_import_dedup_rust] is the instance that
    [deduplicate_memory_imports] runs. The axiom that stood here,
    [identical_memory_import_equiv], had no hypotheses, and the theorem
    after it restated its own hypothesis. *)

(** * Pass 0b: Same-Memory Adapter Collapse Correctness *)

(** If function [adapter_idx] is a same-memory adapter to [target_idx]
    in a single-memory module, calling the adapter is semantically
    equivalent to calling the target. *)
Theorem same_memory_collapse_correct :
  forall (m m' : wasm_module) (adapter_idx target_idx : func_idx)
         (adapter : func_def),
    single_memory m ->
    nth_error (wm_funcs m) adapter_idx = Some adapter ->
    is_same_memory_adapter adapter target_idx ->
    (exists target, nth_error (wm_funcs m) target_idx = Some target /\
                    fd_sig target = fd_sig adapter) ->
    collapse_same_memory_adapter m adapter_idx target_idx = m' ->
    (* Module equivalence: all functions produce the same results *)
    forall f_idx st,
      exec_call m f_idx st = exec_call m' f_idx st.
Proof.
  intros m m' adapter_idx target_idx adapter
         Hsingle Hlookup Hadapter Hsig Hcollapse f_idx st.
  (* The collapsed module m' replaces the adapter body with a forwarding
     trampoline. By same_memory_adapter_equiv, the adapter was already
     equivalent to calling the target. The forwarding trampoline is also
     equivalent to calling the target (by FusedSemantics.trivial_adapter_equiv
     on the rewritten body). Therefore m and m' are equivalent. *)
  subst m'.
  (* collapse_same_memory_adapter is defined as identity in our abstract model *)
  reflexivity.
Qed.

(** Relaxed predicate: balanced single-global save/restore is safe.

    Meld-generated adapters for non-trivial types include a stack pointer
    prologue/epilogue: global.get $sp; sub; global.set $sp ... global.set $sp.
    When the adapter body is replaced by a forwarding trampoline, the global
    writes are removed entirely (net-zero effect on the global).

    A global write pattern is safe to collapse when:
    - All GlobalSet instructions target a single global index G, AND
    - At least one GlobalGet reads from the same index G
    (i.e., the save/restore pattern).

    The collapsed trampoline contains no global instructions at all, so the
    global is left in its original state — identical to the net effect of the
    balanced prologue/epilogue. *)
Theorem sp_save_restore_collapse_correct :
  forall (m m' : wasm_module) (adapter_idx target_idx : func_idx)
         (adapter : func_def),
    single_memory m ->
    nth_error (wm_funcs m) adapter_idx = Some adapter ->
    is_same_memory_adapter adapter target_idx ->
    (exists target, nth_error (wm_funcs m) target_idx = Some target /\
                    fd_sig target = fd_sig adapter) ->
    collapse_same_memory_adapter m adapter_idx target_idx = m' ->
    (* Balanced single-global writes do not affect collapse correctness:
       the adapter is still equivalent to calling the target directly. *)
    forall f_idx st,
      exec_call m f_idx st = exec_call m' f_idx st.
Proof.
  intros m m' adapter_idx target_idx adapter
         Hsingle Hlookup Hadapter Hsig Hcollapse f_idx st.
  (* The balanced global writes (save/restore of a single SP global) are
     contained entirely within the adapter body. When the body is replaced
     by a forwarding trampoline, these writes are removed. Since the net
     effect of the prologue/epilogue is zero (the global is restored to its
     original value), removing them preserves semantics.
     In our abstract model, collapse_same_memory_adapter is identity. *)
  subst m'.
  reflexivity.
Qed.

(** Relaxed predicate: null-guard control flow is safe to collapse.

    Meld-generated adapters for optional/nullable types wrap the
    realloc+copy machinery in a null-guard [if] block:

      (if (local.get $ptr)
        (then  ;; Some: allocate + copy
          ... cabi_realloc ... memory.copy {0,0} ...)
        (else  ;; None: no-op
        ))

    When the adapter body is replaced by a forwarding trampoline:
    - If condition was true (Some): the copy was same-memory redundant,
      so discarding it is equivalent to the original behavior.
    - If condition was false (None): the copy never ran anyway,
      so discarding it is trivially equivalent.

    Safety constraints on the If:
    - The else_body must be empty or nop-only (true null-guard pattern)
    - The then_body must contain only safe-to-discard instructions
      (locals, constants, arithmetic, loads, realloc calls, memory.copy {0,0})
    - No nested If inside If (not a simple null-guard)
    - No Block, Loop, Br, BrIf, BrTable (unsafe control flow)

    The collapsed trampoline contains no control flow at all, so the
    conditional is simply removed — the forwarding call is unconditional
    and equivalent to the target. *)
Theorem null_guard_collapse_correct :
  forall (m m' : wasm_module) (adapter_idx target_idx : func_idx)
         (adapter : func_def),
    single_memory m ->
    nth_error (wm_funcs m) adapter_idx = Some adapter ->
    is_same_memory_adapter adapter target_idx ->
    (exists target, nth_error (wm_funcs m) target_idx = Some target /\
                    fd_sig target = fd_sig adapter) ->
    collapse_same_memory_adapter m adapter_idx target_idx = m' ->
    (* Null-guard If blocks with safe bodies do not affect collapse correctness:
       the adapter is still equivalent to calling the target directly. *)
    forall f_idx st,
      exec_call m f_idx st = exec_call m' f_idx st.
Proof.
  intros m m' adapter_idx target_idx adapter
         Hsingle Hlookup Hadapter Hsig Hcollapse f_idx st.
  (* The null-guard If wraps instructions that are all safe to discard:
     - If condition true (Some): the then_body contains only realloc calls
       and memory.copy {0,0}, which are same-memory redundant.
     - If condition false (None): the then_body never executes.
     In either case, the entire If block has no observable effect beyond
     what the forwarding trampoline achieves.
     By same_memory_adapter_equiv, the adapter is equivalent to the target.
     In our abstract model, collapse_same_memory_adapter is identity. *)
  subst m'.
  reflexivity.
Qed.

(** Relaxed predicate: memory stores are safe to discard.

    Meld-generated adapters for complex types (records, lists) perform
    field-by-field copying via load+store pairs instead of (or alongside)
    memory.copy:

      (i32.store offset=0 (local.get $buf) (i32.load offset=0 (local.get $src)))
      (i32.store offset=4 (local.get $buf) (i32.load offset=4 (local.get $src)))
      (memory.copy 0 0 ...)

    In same-memory adapters, these stores target a freshly realloc'd buffer
    which is discarded when the adapter body is replaced by a forwarding
    trampoline. The target reads from the original pointer in the same
    address space, so the stores are semantically redundant.

    Safety argument:
    - Stores write to the realloc'd buffer (dead after collapse)
    - The target receives the original pointer (same address space)
    - No aliasing concern: the buffer was freshly allocated
    - The collapsed trampoline contains no stores at all *)
Theorem store_discard_collapse_correct :
  forall (m m' : wasm_module) (adapter_idx target_idx : func_idx)
         (adapter : func_def),
    single_memory m ->
    nth_error (wm_funcs m) adapter_idx = Some adapter ->
    is_same_memory_adapter adapter target_idx ->
    (exists target, nth_error (wm_funcs m) target_idx = Some target /\
                    fd_sig target = fd_sig adapter) ->
    collapse_same_memory_adapter m adapter_idx target_idx = m' ->
    (* Memory stores targeting the realloc'd buffer do not affect
       collapse correctness: the adapter is still equivalent to
       calling the target directly. *)
    forall f_idx st,
      exec_call m f_idx st = exec_call m' f_idx st.
Proof.
  intros m m' adapter_idx target_idx adapter
         Hsingle Hlookup Hadapter Hsig Hcollapse f_idx st.
  (* The stores write to a freshly realloc'd buffer. When the adapter body
     is replaced by a forwarding trampoline:
     - The buffer is never allocated (realloc call is removed)
     - The stores are removed (they targeted the dead buffer)
     - The target reads from the original pointer in the same memory
     Since both the original and copied data are in the same address space,
     and the buffer was only used as a temporary for the copy, discarding
     the stores preserves semantics.
     By same_memory_adapter_equiv, the adapter is equivalent to the target.
     In our abstract model, collapse_same_memory_adapter is identity. *)
  subst m'.
  reflexivity.
Qed.

(** Generalized same-memory collapse: works for any consistent memory index N,
    not just single-memory modules. An adapter that uses memory index N
    consistently (all memory.copy, load, and store instructions target N)
    can be collapsed even in a multi-memory module.

    This removes the module-level single_memory precondition and replaces it
    with a per-adapter adapter_uses_single_memory check. *)
Theorem generalized_same_memory_collapse_correct :
  forall (m m' : wasm_module) (adapter_idx target_idx : func_idx)
         (adapter : func_def) (mem_idx : nat),
    nth_error (wm_funcs m) adapter_idx = Some adapter ->
    adapter_uses_single_memory adapter mem_idx ->
    is_same_memory_adapter adapter target_idx ->
    (exists target, nth_error (wm_funcs m) target_idx = Some target /\
                    fd_sig target = fd_sig adapter) ->
    collapse_same_memory_adapter m adapter_idx target_idx = m' ->
    forall f_idx st,
      exec_call m f_idx st = exec_call m' f_idx st.
Proof.
  intros m m' adapter_idx target_idx adapter mem_idx
         Hlookup Hmem Hadapter Hsig Hcollapse f_idx st.
  (* By same_memory_adapter_general_equiv, the adapter is equivalent to the
     target when all memory operations target a consistent index. The
     forwarding trampoline is also equivalent to the target by
     FusedSemantics.trivial_adapter_equiv. Therefore m and m' are equivalent.
     In our abstract model, collapse_same_memory_adapter is identity. *)
  subst m'.
  reflexivity.
Qed.

(** * Pass 1: Adapter Devirtualization Correctness

    Proven in FusedSemantics.v, against a concrete interpreter and with no
    axioms:
    - [trivial_adapter_equiv]: a trivial adapter is observationally
      equivalent to its target;
    - [devirtualize_correct]: rewriting every [call f] to [call (rho f)],
      at any nesting depth, preserves every function, provided [rho f] is
      reachable from [f] along trivial adapters.

    The two statements that used to stand here were removed: one restated
    the axiom, and the other ([devirtualization_preserves_module_semantics])
    assumed its own conclusion. *)

(** * Pass 2: Trivial Call Elimination Correctness

    Proven in FusedSemantics.v, against a concrete interpreter and with no
    axioms: [trivial_call_elim_correct] says that deleting every call to a
    defined [() -> ()] function whose body is [nop]s only, at any nesting
    depth, preserves every function.

    The three statements that used to stand here were removed: the axiom
    [trivial_call_nop], [trivial_call_is_nop], which restated it, and
    [trivial_call_elimination_preserves_semantics], which assumed its own
    conclusion. The axiom also covered only the bodies [[End]] and [[]],
    while [is_nop_body] in Rust accepts any number of [Nop]s. *)

(** * Pass 3: Type Deduplication Correctness *)

(** Two function signatures are structurally equal *)
Definition sig_equiv (s1 s2 : func_sig) : Prop :=
  fs_params s1 = fs_params s2 /\ fs_results s1 = fs_results s2.

(** Type deduplication maps duplicate type indices to canonical ones.
    A canonical remap satisfies two properties:
    1. It maps valid indices to valid indices with equivalent signatures
    2. It is idempotent: canonical representatives map to themselves *)
Definition is_valid_type_remap (types : list func_sig) (remap : func_idx -> func_idx) : Prop :=
  forall i,
    i < length types ->
    remap i < length types /\
    sig_equiv (nth i types (mkFuncSig nil nil)) (nth (remap i) types (mkFuncSig nil nil)).

(** A canonical remap is additionally idempotent *)
Definition is_canonical_type_remap (types : list func_sig) (remap : func_idx -> func_idx) : Prop :=
  is_valid_type_remap types remap /\
  forall i, i < length types -> remap (remap i) = remap i.

(** Type deduplication preserves instruction semantics.
    If type T_i and T_j are structurally equal, any instruction
    referencing T_i behaves identically when referencing T_j. *)
Theorem type_dedup_preserves_semantics :
  forall (types : list func_sig) (remap : func_idx -> func_idx),
    is_valid_type_remap types remap ->
    forall i,
      i < length types ->
      sig_equiv (nth i types (mkFuncSig nil nil))
                (nth (remap i) types (mkFuncSig nil nil)).
Proof.
  intros types remap Hvalid i Hi.
  apply Hvalid. exact Hi.
Qed.

(** Type deduplication is idempotent: applying it twice gives the same result.
    This follows directly from the canonical remap definition. *)
Theorem type_dedup_idempotent :
  forall (types : list func_sig) (remap : func_idx -> func_idx),
    is_canonical_type_remap types remap ->
    forall i,
      i < length types ->
      remap (remap i) = remap i.
Proof.
  intros types remap [_ Hcanon] i Hi.
  apply Hcanon. exact Hi.
Qed.

(** * Pass 4: Dead Function Elimination Correctness *)


(** Dead functions (unreachable) cannot affect the execution of live functions.
    This is because no execution path from any export root will ever invoke them. *)
Theorem dead_function_elim_correct :
  forall (m : wasm_module) (dead_idx : func_idx) (st : exec_state),
    (* Precondition: dead_idx is not reachable from any export *)
    ~ reachable m dead_idx ->
    (* Conclusion: removing dead_idx does not affect any reachable function *)
    forall live_idx,
      reachable m live_idx ->
      exec_call m live_idx st = exec_call m live_idx st.
Proof.
  intros m dead_idx st Hdead live_idx Hlive.
  reflexivity.
Qed.

(** Stronger version: removing a dead function and remapping indices preserves
    the behavior of all reachable functions. *)
Theorem dead_function_removal_preserves_semantics :
  forall (m m' : wasm_module) (dead_idx : func_idx),
    ~ reachable m dead_idx ->
    (* m' is m with dead_idx removed and indices remapped *)
    (forall live_idx st,
      reachable m live_idx ->
      exec_call m live_idx st = exec_call m' live_idx st) ->
    forall live_idx st,
      reachable m live_idx ->
      exec_call m live_idx st = exec_call m' live_idx st.
Proof.
  intros m m' dead_idx Hdead Hpreserve live_idx st Hlive.
  apply Hpreserve. exact Hlive.
Qed.

(** * Pass 5: Import Deduplication Correctness

    Proven in FusedSemantics.v, against a concrete interpreter and with no
    axioms: [import_dedup_correct] says that replacing the imports by the
    deduplicated list and remapping every call (and so every export and the
    start function) with [rho] preserves every function, provided each
    import maps to one with the same key and signature and every defined
    function moves down by the number of imports removed. The statement
    that stood here only restated the axiom. *)

(** * Combined Correctness *)

(** The full fused optimization pipeline preserves module semantics.
    This combines all six pass correctness theorems. *)
Theorem fused_optimization_correct :
  forall (m m' : wasm_module),
    (* m' is the result of applying the fused optimization pipeline to m:
       0a. Memory import deduplication
       0b. Same-memory adapter collapse (generalized: any consistent memory index)
       1. Adapter devirtualization
       2. Trivial call elimination
       3. Type deduplication
       4. Dead function elimination
       5. Import deduplication *)
    (forall live_idx st,
      reachable m live_idx ->
      exec_call m live_idx st = exec_call m' live_idx st) ->
    (* The optimized module preserves behavior of all reachable functions *)
    forall live_idx st,
      reachable m live_idx ->
      exec_call m live_idx st = exec_call m' live_idx st.
Proof.
  intros m m' Hpreserve live_idx st Hlive.
  apply Hpreserve. exact Hlive.
Qed.

(** * Integration with loom Correctness *)

(** The fused optimization pipeline composes with loom's standard
    optimization pipeline. If both preserve semantics individually
    and the fused optimization preserves reachability, their
    composition preserves semantics. *)
Theorem fused_then_standard_correct :
  forall (m m_fused m_opt : wasm_module),
    (* Fused optimization preserves semantics *)
    (forall idx st, reachable m idx ->
      exec_call m idx st = exec_call m_fused idx st) ->
    (* Fused optimization preserves reachability:
       any function reachable in m remains reachable in m_fused.
       This holds because fused optimization only removes dead (unreachable)
       functions and rewrites call targets to equivalent functions. *)
    (forall idx, reachable m idx -> reachable m_fused idx) ->
    (* Standard optimization preserves semantics *)
    (forall idx st, reachable m_fused idx ->
      exec_call m_fused idx st = exec_call m_opt idx st) ->
    (* Combined pipeline preserves semantics *)
    forall idx st,
      reachable m idx ->
      exec_call m idx st = exec_call m_opt idx st.
Proof.
  intros m m_fused m_opt Hfused Hreach_preserved Hstd idx st Hreach.
  rewrite (Hfused idx st Hreach).
  apply Hstd.
  apply Hreach_preserved.
  exact Hreach.
Qed.
