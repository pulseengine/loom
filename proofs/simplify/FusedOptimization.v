(** * Fused Component Optimization Proofs

    The fused optimizer (loom-core/src/fused_optimizer.rs) runs on the single
    core module that component fusion (e.g. meld) produces, and removes the
    artifacts fusion leaves behind. Every pass it runs is proven against a
    concrete call semantics in FusedSemantics.v, with no axioms:

    - 0a. Memory import deduplication: [memory_import_dedup_correct],
      instantiated for [deduplicate_memory_imports] by
      [memory_import_dedup_rust].
    - 1. Adapter devirtualization: [trivial_adapter_equiv],
      [devirtualize_correct].
    - 1.2. Function body deduplication: [body_dedup_correct].
    - 2. Trivial call elimination: [trivial_call_elim_correct].
    - 4. Dead function elimination: [dead_function_elim_correct].
    - 5. Import deduplication: [import_dedup_correct]. Import binding is left
      to the embedder by the wasm specification; the model makes it explicit
      by looking the host up by import key.

    The optimizer does not collapse same-memory adapters (allocate, copy the
    caller's bytes, call the target with the copy). Passing the target the
    caller's pointer instead lets the target write into the caller's buffer:
    with a target that writes 99 into its argument, a caller that wrote 7
    reads 99 back. The tests [same_memory_adapter_keeps_*] in
    fused_optimizer.rs pin this.

    This file keeps the abstract module model and [exec_call], used by
    [fused_then_standard_correct], which composes a fused-pipeline statement
    with a standard-pipeline statement. Replacing [exec_call] is tracked in
    issue #169.

    ## Connection to Meld

    Meld performs: Parse -> Resolve -> Merge -> Adapt -> Encode
    loom performs: Fused Optim -> 12-Phase Pipeline

    These proofs complement meld's own formal verification in
    proofs/spec/fusion_spec.v, which proves the fusion itself is correct.
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
