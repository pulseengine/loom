(** * LOOM Optimization Correctness Master Theorem

    This module provides the master correctness theorem for LOOM's
    WebAssembly optimizer, combining all individual optimization proofs.

    The central theorem states:
      For all valid WebAssembly terms t,
        evaluate(simplify(t)) = evaluate(t)

    This guarantees that LOOM's optimizations preserve program semantics.
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
From proofs Require Import WasmSemantics.
From proofs Require Import TermSemantics.
From proofs Require Import ConstantFolding.
From proofs Require Import Identity.
From proofs Require Import Bitwise.
From proofs Require Import StrengthReduction.
Import ListNotations.

Open Scope Z_scope.

(** * Optimization Categories *)

(** LOOM applies optimizations in these categories:
    1. Constant folding: compute operations on constants at compile time
    2. Identity elimination: remove operations that don't change values
    3. Bitwise optimizations: simplify bitwise operations
    4. Strength reduction: replace expensive ops with cheaper equivalents
    5. Comparison simplification: fold constant comparisons
*)

(** * Core Semantic Preservation *)

(** The simplify function preserves the value of every term that
    evaluates. This is the foundational theorem that all other
    correctness depends on.

    It is stated as refinement, not as [eval_term t = eval_term
    (simplify t)]: that equality is false for ill-typed terms that fail
    to evaluate (see [simplify_not_equiv] in TermSemantics). *)
Theorem simplify_correct : forall t v,
  eval_term t = TROk v -> eval_term (simplify t) = TROk v.
Proof.
  exact simplify_refines.
Qed.

(** Equivalently: a term that evaluates is equivalent to its
    simplification. *)
Corollary simplify_preserves_equiv : forall t v,
  eval_term t = TROk v -> term_equiv t (simplify t).
Proof.
  unfold term_equiv. intros t v H.
  rewrite H. symmetry. apply simplify_correct. exact H.
Qed.

(** * Optimization Soundness *)

(** Each optimization rule is sound: if we can apply a rule,
    the result is semantically equivalent to the input. *)

(** ** Constant Folding Soundness *)

(** Constant folding of addition *)
Theorem constant_fold_add_sound : forall a b,
  eval_term (TI32Const (i32_add a b)) =
  eval_term (TI32Add (TI32Const a) (TI32Const b)).
Proof.
  intros a b. const_cases.
Qed.

(** Constant folding of subtraction *)
Theorem constant_fold_sub_sound : forall a b,
  eval_term (TI32Const (i32_sub a b)) =
  eval_term (TI32Sub (TI32Const a) (TI32Const b)).
Proof.
  intros a b. const_cases.
Qed.

(** Constant folding of multiplication *)
Theorem constant_fold_mul_sound : forall a b,
  eval_term (TI32Const (i32_mul a b)) =
  eval_term (TI32Mul (TI32Const a) (TI32Const b)).
Proof.
  intros a b. const_cases.
Qed.

(** ** Identity Elimination Soundness *)

(** Adding zero preserves the value *)
Theorem add_zero_sound : forall t,
  eval_term (TI32Add t (TI32Const 0)) =
  match eval_term t with
  | TROk (VI32 v) => TROk (VI32 (i32_add v 0))
  | _ => TRFail
  end.
Proof.
  intros t. const_cases.
Qed.

(** Since i32_add v 0 = wrap32 v, this is semantically equivalent to just v *)
Corollary add_zero_simplifies : forall v,
  i32_add v 0 = wrap32 v.
Proof.
  apply i32_add_identity_right.
Qed.

(** Multiplying by one preserves the value *)
Theorem mul_one_sound : forall v,
  i32_mul v 1 = wrap32 v.
Proof.
  apply i32_mul_identity_right.
Qed.

(** Multiplying by zero gives zero *)
Theorem mul_zero_sound : forall v,
  i32_mul v 0 = 0.
Proof.
  apply i32_mul_zero_right.
Qed.

(** ** Bitwise Optimization Soundness *)

(** AND with zero gives zero *)
Theorem and_zero_sound : forall v,
  i32_and v 0 = 0.
Proof.
  apply and32_zero_annihilates_right.
Qed.

(** XOR with self gives zero *)
Theorem xor_self_sound : forall v,
  i32_xor v v = 0.
Proof.
  apply xor32_self_cancels.
Qed.

(** AND with self is idempotent *)
Theorem and_self_sound : forall v,
  i32_and v v = wrap32 v.
Proof.
  apply and32_idempotent.
Qed.

(** OR with self is idempotent *)
Theorem or_self_sound : forall v,
  i32_or v v = wrap32 v.
Proof.
  apply or32_idempotent.
Qed.

(** ** Strength Reduction Soundness *)

(** Multiplication by 2 equals left shift by 1 *)
Theorem strength_mul2_sound : forall v,
  i32_mul v 2 = i32_shl v 1.
Proof.
  apply mul_2_is_shl_1.
Qed.

(** Multiplication by 4 equals left shift by 2 *)
Theorem strength_mul4_sound : forall v,
  i32_mul v 4 = i32_shl v 2.
Proof.
  apply mul_4_is_shl_2.
Qed.

(** ** Comparison Optimization Soundness *)

(** Comparing a value to itself always yields 1 (true) for equality *)
Theorem eq_self_sound : forall v,
  i32_eq v v = 1.
Proof.
  apply i32_eq_reflexive.
Qed.

(** Comparing a value to itself always yields 0 (false) for inequality *)
Theorem ne_self_sound : forall v,
  i32_ne v v = 0.
Proof.
  apply i32_ne_reflexive.
Qed.

(** * Multi-Pass Optimization Correctness *)

(** LOOM may apply simplify multiple times. Each pass preserves semantics,
    so multiple passes also preserve semantics. *)

Theorem simplify_idempotent_sound : forall t v,
  eval_term t = TROk v -> eval_term (simplify (simplify t)) = TROk v.
Proof.
  intros t v H.
  apply simplify_correct. apply simplify_correct. exact H.
Qed.

(** More generally, any finite number of simplify passes preserves semantics *)
Fixpoint simplify_n (n : nat) (t : Term) : Term :=
  match n with
  | O => t
  | S n' => simplify (simplify_n n' t)
  end.

Theorem simplify_n_correct : forall n t v,
  eval_term t = TROk v -> eval_term (simplify_n n t) = TROk v.
Proof.
  induction n; intros t v H.
  - (* n = 0 *) exact H.
  - (* n = S n' *)
    simpl. apply simplify_correct. apply IHn. exact H.
Qed.

(** * Composition of Optimizations *)

(** When multiple optimization rules could apply, any valid choice
    preserves semantics. *)

(** Helper: 2^32 is nonzero (for modular arithmetic) *)
Lemma pow2_32_nz : 2 ^ 32 <> 0.
Proof. discriminate. Qed.

(** Example: (x + 0) * 1 can be simplified by either rule first *)
Theorem compose_add0_mul1 : forall x,
  eval_term (TI32Mul (TI32Add (TI32Const x) (TI32Const 0)) (TI32Const 1)) =
  eval_term (TI32Const (wrap32 x)).
Proof.
  intros x.
  simpl.
  f_equal. f_equal.
  (* (x + 0) * 1 = wrap32(x + 0) = wrap32(wrap32 x) = ... *)
  rewrite i32_add_identity_right.
  rewrite i32_mul_identity_right.
  (* wrap32 (wrap32 x) = wrap32 x *)
  unfold wrap32.
  rewrite Z.mod_mod.
  - reflexivity.
  - exact pow2_32_nz.
Qed.

(** * Safety Properties *)

(** Simplification never introduces undefined behavior on well-typed terms *)
Theorem simplify_safe : forall t v,
  eval_term t = TROk v ->
  exists v', eval_term (simplify t) = TROk v'.
Proof.
  intros t v Heval_term.
  exists v.
  apply simplify_correct. exact Heval_term.
Qed.

(** Simplification preserves the type of the result *)
Theorem simplify_preserves_type : forall t v,
  eval_term t = TROk v ->
  eval_term (simplify t) = TROk v.
Proof.
  intros t v Heval_term.
  apply simplify_correct. exact Heval_term.
Qed.

(** * Complete Optimization Correctness *)

(** The master theorem combining all guarantees *)
Theorem loom_optimization_correct :
  forall t v,
    eval_term t = TROk v ->
    (* 1. Semantics are preserved *)
    eval_term (simplify t) = TROk v /\
    (* 2. Multi-pass optimization is correct *)
    (forall n, eval_term (simplify_n n t) = TROk v).
Proof.
  intros t v H.
  split.
  - apply simplify_correct. exact H.
  - intros n. apply simplify_n_correct. exact H.
Qed.

(** * Connection to Instruction Semantics *)

(** The optimized term, when compiled to instructions, produces
    equivalent behavior to the original term's instructions. *)

Theorem compiled_optimization_correct : forall t v s,
  eval_term t = TROk v ->
  exec_instrs (term_to_instrs t) s =
  exec_instrs (term_to_instrs (simplify t)) s.
Proof.
  (* optimization_preserves_execution in TermSemantics, which needs the
     original term to evaluate. *)
  exact optimization_preserves_execution.
Qed.

(** * Future Work *)

(** The following theorems represent important properties to prove
    as LOOM's verification infrastructure matures:

    1. Bidirectional correctness: term_instr_bijection (below) proves
       instrs_to_terms (term_to_instrs t) = Some t. The converse, that
       every instruction sequence instrs_to_terms accepts re-encodes to
       itself, is not yet proven.

    2. Z3 agreement: When Z3 reports two terms equivalent, Rocq proofs
       also establish their equivalence.

    3. End-to-end correctness: Parsing, optimization, and encoding
       together preserve the observable behavior of WASM modules.
*)

(** * Instruction-to-term decoding

    [instrs_to_terms] is the inverse of [term_to_instrs]: it runs the
    instruction sequence as a stack machine over terms. A constant or
    [nop] pushes a leaf; a binary operator pops its right then left
    operand; [eqz] and [drop] pop one operand. Decoding succeeds when the
    sequence leaves exactly one term. *)

Definition dec_bin (mk : Term -> Term -> Term) (stk : list Term)
    : option (list Term) :=
  match stk with
  | r :: l :: rest => Some (mk l r :: rest)
  | _ => None
  end.

Definition dec_un (mk : Term -> Term) (stk : list Term)
    : option (list Term) :=
  match stk with
  | t :: rest => Some (mk t :: rest)
  | _ => None
  end.

Definition dec_step (i : Instruction) (stk : list Term)
    : option (list Term) :=
  match i with
  | Instr_I32Const v => Some (TI32Const v :: stk)
  | Instr_I64Const v => Some (TI64Const v :: stk)
  | Instr_I32Add => dec_bin TI32Add stk
  | Instr_I32Sub => dec_bin TI32Sub stk
  | Instr_I32Mul => dec_bin TI32Mul stk
  | Instr_I64Add => dec_bin TI64Add stk
  | Instr_I64Sub => dec_bin TI64Sub stk
  | Instr_I64Mul => dec_bin TI64Mul stk
  | Instr_I32And => dec_bin TI32And stk
  | Instr_I32Or => dec_bin TI32Or stk
  | Instr_I32Xor => dec_bin TI32Xor stk
  | Instr_I64And => dec_bin TI64And stk
  | Instr_I64Or => dec_bin TI64Or stk
  | Instr_I64Xor => dec_bin TI64Xor stk
  | Instr_I32Shl => dec_bin TI32Shl stk
  | Instr_I32ShrS => dec_bin TI32ShrS stk
  | Instr_I32ShrU => dec_bin TI32ShrU stk
  | Instr_I64Shl => dec_bin TI64Shl stk
  | Instr_I64ShrS => dec_bin TI64ShrS stk
  | Instr_I64ShrU => dec_bin TI64ShrU stk
  | Instr_I32Eq => dec_bin TI32Eq stk
  | Instr_I32Ne => dec_bin TI32Ne stk
  | Instr_I32LtS => dec_bin TI32LtS stk
  | Instr_I32LtU => dec_bin TI32LtU stk
  | Instr_I32GtS => dec_bin TI32GtS stk
  | Instr_I32GtU => dec_bin TI32GtU stk
  | Instr_I32LeS => dec_bin TI32LeS stk
  | Instr_I32LeU => dec_bin TI32LeU stk
  | Instr_I32GeS => dec_bin TI32GeS stk
  | Instr_I32GeU => dec_bin TI32GeU stk
  | Instr_I64Eq => dec_bin TI64Eq stk
  | Instr_I64Ne => dec_bin TI64Ne stk
  | Instr_I64LtS => dec_bin TI64LtS stk
  | Instr_I64LtU => dec_bin TI64LtU stk
  | Instr_I64GtS => dec_bin TI64GtS stk
  | Instr_I64GtU => dec_bin TI64GtU stk
  | Instr_I64LeS => dec_bin TI64LeS stk
  | Instr_I64LeU => dec_bin TI64LeU stk
  | Instr_I64GeS => dec_bin TI64GeS stk
  | Instr_I64GeU => dec_bin TI64GeU stk
  | Instr_I32Eqz => dec_un TI32Eqz stk
  | Instr_I64Eqz => dec_un TI64Eqz stk
  | Instr_Drop => dec_un TDrop stk
  | Instr_Nop => Some (TNop :: stk)
  end.

Fixpoint dec_instrs (is : list Instruction) (stk : list Term)
    : option (list Term) :=
  match is with
  | [] => Some stk
  | i :: rest =>
      match dec_step i stk with
      | Some stk' => dec_instrs rest stk'
      | None => None
      end
  end.

Definition instrs_to_terms (is : list Instruction) : option Term :=
  match dec_instrs is [] with
  | Some [t] => Some t
  | _ => None
  end.

(** Decoding the instructions of [t] pushes [t]. *)
Lemma dec_instrs_term : forall t rest stk,
  dec_instrs (term_to_instrs t ++ rest) stk = dec_instrs rest (t :: stk).
Proof.
  induction t; intros rest stk; simpl;
    rewrite <- ?app_assoc; simpl;
    rewrite ?IHt, ?IHt1; simpl; rewrite ?IHt2; reflexivity.
Qed.

(** Converting a term to instructions and back yields the term. *)
Theorem term_instr_bijection : forall (t : Term),
  instrs_to_terms (term_to_instrs t) = Some t.
Proof.
  intros t. unfold instrs_to_terms.
  rewrite <- (app_nil_r (term_to_instrs t)), dec_instrs_term.
  reflexivity.
Qed.

(** * Summary of Proven Properties *)

(**
   PROVEN (Qed):
   - simplify_correct: if eval_term t = TROk v then
     eval_term (simplify t) = TROk v (refinement; the unconditional
     equality is false, see simplify_not_equiv)
   - All constant folding rules (40+ theorems)
   - All algebraic identity rules (25+ theorems)
   - All bitwise optimization rules (20+ theorems)
   - Strength reduction for power-of-2 cases (15+ theorems)
   - Comparison optimizations (10+ theorems)
   - Multi-pass correctness
   - Type preservation

   - compiled_optimization_correct: instruction-level equivalence,
     for terms that evaluate

   - term_instr_bijection: instrs_to_terms (term_to_instrs t) = Some t
     for every term (term_to_instrs is injective with a computable
     inverse)

   NOT COVERED HERE: the relation between this Term model and LOOM's
   Rust implementation (ISLE rules, parser, encoder). The theorems above
   are about the model.
*)

Close Scope Z_scope.
