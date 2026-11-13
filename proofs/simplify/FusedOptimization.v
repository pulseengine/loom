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

    0b. Same-memory adapter collapse: removed. It is not
       semantics-preserving; see the Pass 0b note below.

    1. Adapter devirtualization (proven in FusedSemantics.v): replacing call-to-adapter with call-to-target
       preserves execution semantics when the adapter is a trivial forwarder.

    2. Trivial call elimination (proven in FusedSemantics.v): removing calls
       to void->void no-op functions preserves execution semantics.

    3. Type deduplication: removed. It had no proof and never changed a
       parsed module; see the Pass 3 note below.

    4. Dead function elimination (proven in FusedSemantics.v): removing the
       defined functions outside a call-closed live set, and renumbering the
       rest, preserves every live function.

    5. Import deduplication (proven in FusedSemantics.v): merging identical
       imports (same module+name+type)
       and remapping references preserves binding semantics.

    ## Proof Architecture

    exec_call is axiomatized: the remaining statements here reason about
    module-level execution abstractly, and no other axiom is assumed.

    ## Status (issue #169)

    This file is being replaced, pass by pass, by FusedSemantics.v, which
    proves each pass against a concrete call semantics. Until a pass moves
    there, its statement here is not semantic evidence. Pass 0a (memory
    import deduplication), Pass 1 (adapter devirtualization), Pass 2
    (trivial call elimination), Pass 4 (dead function elimination) and
    Pass 5 (import deduplication) have moved: see FusedSemantics.v. Two
    passes that loom runs have no proof in either file: Pass 1.1 (scalar
    adapter inlining) and Pass 1.2 (function body deduplication).

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

(** * Pass 0a: Memory Import Deduplication Correctness

    Proven in FusedSemantics.v, against a concrete interpreter and with no
    axioms: [memory_import_dedup_correct] says that replacing the memory
    imports and remapping every memory index with [sg] preserves every
    function, provided each index names the same memory key (or no memory)
    before and after; [memory_import_dedup_rust] is the instance that
    [deduplicate_memory_imports] runs. The axiom that stood here,
    [identical_memory_import_equiv], had no hypotheses, and the theorem
    after it restated its own hypothesis. *)

(** * Pass 0b: Same-Memory Adapter Collapse (removed)

    The axioms [same_memory_adapter_equiv] and
    [same_memory_adapter_general_equiv] stated that an adapter which
    allocates a buffer, copies the caller's bytes into it within one memory
    and calls the target with the new pointer behaves as the target called
    with the original pointer. That is false: the target then works on the
    caller's buffer instead of a private copy. With a target that writes 99
    into its argument, a caller that wrote 7 reads 7 back before the
    collapse and 99 after; with a target that returns its argument, the
    adapter returns the allocated pointer before and the caller's pointer
    after. Both are tests in loom-core/src/fused_optimizer.rs
    ([same_memory_adapter_keeps_*]). The axioms, the five collapse theorems
    that rested on them, and the pass itself have been removed. *)

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

(** * Pass 3: Type Deduplication (removed)

    [deduplicate_function_types] merged structurally identical function
    types and remapped the type indices of imports and [call_indirect]. It
    returned 0 whenever the raw type section was present, which the parser
    sets for every module with a type section, so it never changed a parsed
    module; it ran only on modules built in memory. The two statements that
    stood here, [type_dedup_preserves_semantics] and [type_dedup_idempotent],
    restated their hypotheses, and this model has no type indices. The pass
    has been removed from fused_optimizer.rs (issue #169). *)

(** * Pass 4: Dead Function Elimination Correctness

    Proven in FusedSemantics.v, against a concrete interpreter and with no
    axioms: [dead_function_elim_correct] says that removing the defined
    functions outside a live set that is closed under calls, moving each kept
    function down by the number removed before it and rewriting every call
    through that map, preserves every live function. The two statements that
    used to stand here were removed: [dead_function_elim_correct] concluded
    [exec_call m f st = exec_call m f st], and
    [dead_function_removal_preserves_semantics] assumed its own conclusion. *)

(** * Pass 5: Import Deduplication Correctness

    Proven in FusedSemantics.v, against a concrete interpreter and with no
    axioms: [import_dedup_correct] says that replacing the imports by the
    deduplicated list and remapping every call (and so every export and the
    start function) with [rho] preserves every function, provided each
    import maps to one with the same key and signature and every defined
    function moves down by the number of imports removed. The statement
    that stood here only restated the axiom. *)

(** * Combined Correctness

    No statement about the whole pipeline is made here. The one that stood
    here, [fused_optimization_correct], assumed its own conclusion. Passes
    0a, 1, 2, 4 and 5 are proven in FusedSemantics.v; Passes 1.1 and 1.2
    run and are not proven. *)

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
