(** * ISLE Term Denotational Semantics

    This module connects ISLE term representations to WebAssembly operational
    semantics, providing the bridge needed for proving optimization correctness.

    Key definitions:
    - Term: ISLE term representation (tree structure)
    - denote: Denotational semantics mapping terms to state transformers
    - term_to_instrs: Compilation from terms to instructions
    - Connection theorems between denotational and operational semantics

    The main theorem to prove:
      forall t s, denote t s = exec_instrs (term_to_instrs t) s
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
From proofs Require Import WasmSemantics.
Import ListNotations.

Open Scope Z_scope.

(** * ISLE Term Representation *)

(** Terms are expression trees that represent stack computations.
    Unlike flat instruction sequences, terms make the data flow explicit. *)
Inductive Term : Type :=
  (* Constants *)
  | TI32Const : Z -> Term
  | TI64Const : Z -> Term
  (* Binary arithmetic *)
  | TI32Add : Term -> Term -> Term
  | TI32Sub : Term -> Term -> Term
  | TI32Mul : Term -> Term -> Term
  | TI64Add : Term -> Term -> Term
  | TI64Sub : Term -> Term -> Term
  | TI64Mul : Term -> Term -> Term
  (* Binary bitwise *)
  | TI32And : Term -> Term -> Term
  | TI32Or : Term -> Term -> Term
  | TI32Xor : Term -> Term -> Term
  | TI64And : Term -> Term -> Term
  | TI64Or : Term -> Term -> Term
  | TI64Xor : Term -> Term -> Term
  (* Shifts *)
  | TI32Shl : Term -> Term -> Term
  | TI32ShrS : Term -> Term -> Term
  | TI32ShrU : Term -> Term -> Term
  | TI64Shl : Term -> Term -> Term
  | TI64ShrS : Term -> Term -> Term
  | TI64ShrU : Term -> Term -> Term
  (* Comparisons (return i32) *)
  | TI32Eq : Term -> Term -> Term
  | TI32Ne : Term -> Term -> Term
  | TI32LtS : Term -> Term -> Term
  | TI32LtU : Term -> Term -> Term
  | TI32GtS : Term -> Term -> Term
  | TI32GtU : Term -> Term -> Term
  | TI32LeS : Term -> Term -> Term
  | TI32LeU : Term -> Term -> Term
  | TI32GeS : Term -> Term -> Term
  | TI32GeU : Term -> Term -> Term
  | TI32Eqz : Term -> Term
  | TI64Eq : Term -> Term -> Term
  | TI64Ne : Term -> Term -> Term
  | TI64LtS : Term -> Term -> Term
  | TI64LtU : Term -> Term -> Term
  | TI64GtS : Term -> Term -> Term
  | TI64GtU : Term -> Term -> Term
  | TI64LeS : Term -> Term -> Term
  | TI64LeU : Term -> Term -> Term
  | TI64GeS : Term -> Term -> Term
  | TI64GeU : Term -> Term -> Term
  | TI64Eqz : Term -> Term
  (* Stack manipulation *)
  | TDrop : Term -> Term
  | TNop : Term.

(** * Term to Instruction Compilation *)

(** Compile a term to a flat instruction sequence.
    This is a depth-first traversal that evaluates operands left-to-right. *)
Fixpoint term_to_instrs (t : Term) : list Instruction :=
  match t with
  (* Constants *)
  | TI32Const v => [Instr_I32Const v]
  | TI64Const v => [Instr_I64Const v]
  (* Binary arithmetic *)
  | TI32Add l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Add]
  | TI32Sub l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Sub]
  | TI32Mul l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Mul]
  | TI64Add l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Add]
  | TI64Sub l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Sub]
  | TI64Mul l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Mul]
  (* Bitwise *)
  | TI32And l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32And]
  | TI32Or l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Or]
  | TI32Xor l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Xor]
  | TI64And l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64And]
  | TI64Or l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Or]
  | TI64Xor l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Xor]
  (* Shifts *)
  | TI32Shl l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Shl]
  | TI32ShrS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32ShrS]
  | TI32ShrU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32ShrU]
  | TI64Shl l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Shl]
  | TI64ShrS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64ShrS]
  | TI64ShrU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64ShrU]
  (* Comparisons *)
  | TI32Eq l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Eq]
  | TI32Ne l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32Ne]
  | TI32LtS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32LtS]
  | TI32LtU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32LtU]
  | TI32GtS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32GtS]
  | TI32GtU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32GtU]
  | TI32LeS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32LeS]
  | TI32LeU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32LeU]
  | TI32GeS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32GeS]
  | TI32GeU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I32GeU]
  | TI32Eqz t => term_to_instrs t ++ [Instr_I32Eqz]
  | TI64Eq l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Eq]
  | TI64Ne l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64Ne]
  | TI64LtS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64LtS]
  | TI64LtU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64LtU]
  | TI64GtS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64GtS]
  | TI64GtU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64GtU]
  | TI64LeS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64LeS]
  | TI64LeU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64LeU]
  | TI64GeS l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64GeS]
  | TI64GeU l r => term_to_instrs l ++ term_to_instrs r ++ [Instr_I64GeU]
  | TI64Eqz t => term_to_instrs t ++ [Instr_I64Eqz]
  (* Stack manipulation *)
  | TDrop t => term_to_instrs t ++ [Instr_Drop]
  | TNop => [Instr_Nop]
  end.

(** * Denotational Semantics *)

(** The denotational semantics directly computes the result of a term
    without going through the instruction sequence. This makes it easier
    to prove equivalences. *)

(** Result of term evaluation: produces a value or fails *)
Inductive TermResult : Type :=
  | TROk : Value -> TermResult
  | TRFail : TermResult.

(** Evaluate a term to produce a value.
    Note: This assumes the stack contains i32/i64 values as needed. *)
Fixpoint eval_term (t : Term) : TermResult :=
  match t with
  (* Constants always succeed *)
  | TI32Const v => TROk (VI32 v)
  | TI64Const v => TROk (VI64 v)

  (* i32 binary operations *)
  | TI32Add l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_add a b))
      | _, _ => TRFail
      end
  | TI32Sub l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_sub a b))
      | _, _ => TRFail
      end
  | TI32Mul l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_mul a b))
      | _, _ => TRFail
      end

  (* i64 binary operations *)
  | TI64Add l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_add a b))
      | _, _ => TRFail
      end
  | TI64Sub l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_sub a b))
      | _, _ => TRFail
      end
  | TI64Mul l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_mul a b))
      | _, _ => TRFail
      end

  (* i32 bitwise *)
  | TI32And l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_and a b))
      | _, _ => TRFail
      end
  | TI32Or l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_or a b))
      | _, _ => TRFail
      end
  | TI32Xor l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_xor a b))
      | _, _ => TRFail
      end

  (* i64 bitwise *)
  | TI64And l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_and a b))
      | _, _ => TRFail
      end
  | TI64Or l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_or a b))
      | _, _ => TRFail
      end
  | TI64Xor l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_xor a b))
      | _, _ => TRFail
      end

  (* i32 shifts *)
  | TI32Shl l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_shl a b))
      | _, _ => TRFail
      end
  | TI32ShrS l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_shr_s a b))
      | _, _ => TRFail
      end
  | TI32ShrU l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_shr_u a b))
      | _, _ => TRFail
      end

  (* i64 shifts *)
  | TI64Shl l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_shl a b))
      | _, _ => TRFail
      end
  | TI64ShrS l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_shr_s a b))
      | _, _ => TRFail
      end
  | TI64ShrU l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI64 (i64_shr_u a b))
      | _, _ => TRFail
      end

  (* i32 comparisons *)
  | TI32Eq l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_eq a b))
      | _, _ => TRFail
      end
  | TI32Ne l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_ne a b))
      | _, _ => TRFail
      end
  | TI32LtS l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_lt_s a b))
      | _, _ => TRFail
      end
  | TI32LtU l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_lt_u a b))
      | _, _ => TRFail
      end
  | TI32GtS l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_gt_s a b))
      | _, _ => TRFail
      end
  | TI32GtU l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_gt_u a b))
      | _, _ => TRFail
      end
  | TI32LeS l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_le_s a b))
      | _, _ => TRFail
      end
  | TI32LeU l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_le_u a b))
      | _, _ => TRFail
      end
  | TI32GeS l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_ge_s a b))
      | _, _ => TRFail
      end
  | TI32GeU l r =>
      match eval_term l, eval_term r with
      | TROk (VI32 a), TROk (VI32 b) => TROk (VI32 (i32_ge_u a b))
      | _, _ => TRFail
      end
  | TI32Eqz t =>
      match eval_term t with
      | TROk (VI32 a) => TROk (VI32 (i32_eqz a))
      | _ => TRFail
      end

  (* i64 comparisons *)
  | TI64Eq l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_eq a b))
      | _, _ => TRFail
      end
  | TI64Ne l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_ne a b))
      | _, _ => TRFail
      end
  | TI64LtS l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_lt_s a b))
      | _, _ => TRFail
      end
  | TI64LtU l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_lt_u a b))
      | _, _ => TRFail
      end
  | TI64GtS l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_gt_s a b))
      | _, _ => TRFail
      end
  | TI64GtU l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_gt_u a b))
      | _, _ => TRFail
      end
  | TI64LeS l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_le_s a b))
      | _, _ => TRFail
      end
  | TI64LeU l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_le_u a b))
      | _, _ => TRFail
      end
  | TI64GeS l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_ge_s a b))
      | _, _ => TRFail
      end
  | TI64GeU l r =>
      match eval_term l, eval_term r with
      | TROk (VI64 a), TROk (VI64 b) => TROk (VI32 (i64_ge_u a b))
      | _, _ => TRFail
      end
  | TI64Eqz t =>
      match eval_term t with
      | TROk (VI64 a) => TROk (VI32 (i64_eqz a))
      | _ => TRFail
      end

  (* Stack manipulation *)
  | TDrop t =>
      match eval_term t with
      | TROk _ => TRFail  (* Drop produces no value *)
      | TRFail => TRFail
      end
  | TNop => TRFail  (* Nop produces no value *)
  end.

(** * Term Equivalence *)

(** Two terms are equivalent if they evaluate to the same value *)
Definition term_equiv (t1 t2 : Term) : Prop :=
  eval_term t1 = eval_term t2.

(** Term equivalence is reflexive *)
Theorem term_equiv_refl : forall t, term_equiv t t.
Proof. unfold term_equiv. reflexivity. Qed.

(** Term equivalence is symmetric *)
Theorem term_equiv_sym : forall t1 t2, term_equiv t1 t2 -> term_equiv t2 t1.
Proof. unfold term_equiv. intros. symmetry. assumption. Qed.

(** Term equivalence is transitive *)
Theorem term_equiv_trans : forall t1 t2 t3,
  term_equiv t1 t2 -> term_equiv t2 t3 -> term_equiv t1 t3.
Proof. unfold term_equiv. intros. rewrite H. assumption. Qed.

(** * Basic Term Evaluation Properties *)

(** Constants evaluate correctly *)
Theorem eval_i32_const : forall v, eval_term (TI32Const v) = TROk (VI32 v).
Proof. reflexivity. Qed.

Theorem eval_i64_const : forall v, eval_term (TI64Const v) = TROk (VI64 v).
Proof. reflexivity. Qed.

(** i32.add of constants evaluates to their sum *)
Theorem eval_i32_add_const : forall a b,
  eval_term (TI32Add (TI32Const a) (TI32Const b)) = TROk (VI32 (i32_add a b)).
Proof. reflexivity. Qed.

(** i32.sub of constants evaluates to their difference *)
Theorem eval_i32_sub_const : forall a b,
  eval_term (TI32Sub (TI32Const a) (TI32Const b)) = TROk (VI32 (i32_sub a b)).
Proof. reflexivity. Qed.

(** i32.mul of constants evaluates to their product *)
Theorem eval_i32_mul_const : forall a b,
  eval_term (TI32Mul (TI32Const a) (TI32Const b)) = TROk (VI32 (i32_mul a b)).
Proof. reflexivity. Qed.

(** * Constant Folding Equivalence Theorems *)

(** The core theorem: constant folding preserves semantics *)

(** i32.add constant folding *)
Theorem i32_add_const_fold : forall a b,
  term_equiv
    (TI32Add (TI32Const a) (TI32Const b))
    (TI32Const (i32_add a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.sub constant folding *)
Theorem i32_sub_const_fold : forall a b,
  term_equiv
    (TI32Sub (TI32Const a) (TI32Const b))
    (TI32Const (i32_sub a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.mul constant folding *)
Theorem i32_mul_const_fold : forall a b,
  term_equiv
    (TI32Mul (TI32Const a) (TI32Const b))
    (TI32Const (i32_mul a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.add constant folding *)
Theorem i64_add_const_fold : forall a b,
  term_equiv
    (TI64Add (TI64Const a) (TI64Const b))
    (TI64Const (i64_add a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.sub constant folding *)
Theorem i64_sub_const_fold : forall a b,
  term_equiv
    (TI64Sub (TI64Const a) (TI64Const b))
    (TI64Const (i64_sub a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.mul constant folding *)
Theorem i64_mul_const_fold : forall a b,
  term_equiv
    (TI64Mul (TI64Const a) (TI64Const b))
    (TI64Const (i64_mul a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** * Algebraic Identity Theorems *)

(** x + 0 = wrap32 x (the result is wrapped) *)
Theorem i32_add_zero_right : forall t z,
  eval_term t = TROk (VI32 z) ->
  eval_term (TI32Add t (TI32Const 0)) = TROk (VI32 (wrap32 z)).
Proof.
  intros t z Ht.
  simpl. rewrite Ht. simpl.
  f_equal. f_equal.
  unfold i32_add. rewrite Z.add_0_r. reflexivity.
Qed.

(** For already-wrapped values, x + 0 is identity *)
Theorem i32_add_zero_right_wrapped : forall t z,
  eval_term t = TROk (VI32 z) ->
  0 <= z < Z.pow 2 32 ->
  term_equiv (TI32Add t (TI32Const 0)) t.
Proof.
  unfold term_equiv. intros t z Ht Hz.
  simpl. rewrite Ht. simpl.
  f_equal. f_equal.
  unfold i32_add, wrap32.
  rewrite Z.add_0_r.
  rewrite Z.mod_small; auto.
Qed.

(** 0 + x = x (when x evaluates to a wrapped value) *)
Lemma i32_add_zero_left_helper : forall z,
  i32_add 0 z = wrap32 z.
Proof.
  intros. unfold i32_add. rewrite Z.add_0_l. reflexivity.
Qed.

(** x * 1 = x *)
Lemma i32_mul_one_right_helper : forall z,
  i32_mul z 1 = wrap32 z.
Proof.
  intros. unfold i32_mul. rewrite Z.mul_1_r. reflexivity.
Qed.

(** 1 * x = x *)
Lemma i32_mul_one_left_helper : forall z,
  i32_mul 1 z = wrap32 z.
Proof.
  intros. unfold i32_mul. rewrite Z.mul_1_l. reflexivity.
Qed.

(** x * 0 = 0 *)
Lemma i32_mul_zero_right_helper : forall z,
  i32_mul z 0 = 0.
Proof.
  intros. unfold i32_mul, wrap32.
  rewrite Z.mul_0_r. reflexivity.
Qed.

(** 0 * x = 0 *)
Lemma i32_mul_zero_left_helper : forall z,
  i32_mul 0 z = 0.
Proof.
  intros. unfold i32_mul, wrap32.
  rewrite Z.mul_0_l. reflexivity.
Qed.

(** x - 0 = x *)
Lemma i32_sub_zero_right_helper : forall z,
  i32_sub z 0 = wrap32 z.
Proof.
  intros. unfold i32_sub. rewrite Z.sub_0_r. reflexivity.
Qed.

(** x - x = 0 *)
Lemma i32_sub_self_helper : forall z,
  i32_sub z z = 0.
Proof.
  intros. unfold i32_sub, wrap32.
  replace (z - z) with 0 by lia.
  simpl. reflexivity.
Qed.

(** * Bitwise Identity Theorems *)

(** x & 0 = 0 *)
Lemma i32_and_zero_right_helper : forall z,
  i32_and z 0 = 0.
Proof.
  intros. unfold i32_and, wrap32. rewrite Z.land_0_r. reflexivity.
Qed.

(** 0 & x = 0 *)
Lemma i32_and_zero_left_helper : forall z,
  i32_and 0 z = 0.
Proof.
  intros. unfold i32_and, wrap32. rewrite Z.land_0_l. reflexivity.
Qed.

(** x | 0 = x (wrapped) *)
Lemma i32_or_zero_right_helper : forall z,
  i32_or z 0 = wrap32 z.
Proof.
  intros. unfold i32_or. rewrite Z.lor_0_r. reflexivity.
Qed.

(** 0 | x = x (wrapped) *)
Lemma i32_or_zero_left_helper : forall z,
  i32_or 0 z = wrap32 z.
Proof.
  intros. unfold i32_or. rewrite Z.lor_0_l. reflexivity.
Qed.

(** x ^ 0 = x (wrapped) *)
Lemma i32_xor_zero_right_helper : forall z,
  i32_xor z 0 = wrap32 z.
Proof.
  intros. unfold i32_xor. rewrite Z.lxor_0_r. reflexivity.
Qed.

(** 0 ^ x = x (wrapped) *)
Lemma i32_xor_zero_left_helper : forall z,
  i32_xor 0 z = wrap32 z.
Proof.
  intros. unfold i32_xor. rewrite Z.lxor_0_l. reflexivity.
Qed.

(** x ^ x = 0 *)
Lemma i32_xor_self_helper : forall z,
  i32_xor z z = 0.
Proof.
  intros. unfold i32_xor. rewrite Z.lxor_nilpotent. reflexivity.
Qed.

(** x & x = x (wrapped) *)
Lemma i32_and_self_helper : forall z,
  i32_and z z = wrap32 z.
Proof.
  intros. unfold i32_and. rewrite Z.land_diag. reflexivity.
Qed.

(** x | x = x (wrapped) *)
Lemma i32_or_self_helper : forall z,
  i32_or z z = wrap32 z.
Proof.
  intros. unfold i32_or. rewrite Z.lor_diag. reflexivity.
Qed.

(** * Shift Identity Theorems *)

(** Helper: 2^32 is not zero *)
Lemma pow2_32_nonzero : 2 ^ 32 <> 0.
Proof. lia. Qed.

(** x << 0 = x (wrapped, when shift amount has low bits = 0) *)
Lemma i32_shl_zero_helper : forall z,
  i32_shl z 0 = wrap32 z.
Proof.
  intros. unfold i32_shl, shift_mask32, wrap32.
  rewrite Z.land_0_l. rewrite Z.shiftl_0_r.
  rewrite Z.mod_mod.
  - reflexivity.
  - exact pow2_32_nonzero.
Qed.

(** x >> 0 = x (wrapped, unsigned) *)
Lemma i32_shr_u_zero_helper : forall z,
  i32_shr_u z 0 = wrap32 z.
Proof.
  intros. unfold i32_shr_u, shift_mask32, wrap32.
  rewrite Z.land_0_l. rewrite Z.shiftr_0_r. reflexivity.
Qed.

(** * Comparison Identity Theorems *)

(** x == x = 1 *)
Lemma i32_eq_self_helper : forall z,
  i32_eq z z = 1.
Proof.
  intros. unfold i32_eq. rewrite Z.eqb_refl. reflexivity.
Qed.

(** x != x = 0 *)
Lemma i32_ne_self_helper : forall z,
  i32_ne z z = 0.
Proof.
  intros. unfold i32_ne. rewrite Z.eqb_refl. reflexivity.
Qed.

(** * Simplification Function *)

(** The simplification function applies optimization rules to terms.
    This mirrors the simplify_with_env function in Rust. *)
(** The twelve operators [simplify] rewrites. Each [mk_*] takes the two
    already-simplified operands and applies that operator's rules in order
    (constant folding first, then the identity and zero rules). They are
    separate definitions so that a proof can reason about one operator's
    rules without unfolding all of [simplify]: done inline, the kernel
    check of that unfolding took about 70 s per operator. *)

Definition mk_TI32Add (l' r' : Term) : Term :=
  match l', r' with
  (* Constant folding *)
  | TI32Const a, TI32Const b => TI32Const (i32_add a b)
  (* x + 0 = x *)
  | _, TI32Const 0 => l'
  (* 0 + x = x *)
  | TI32Const 0, _ => r'
  | _, _ => TI32Add l' r'
  end.

Definition mk_TI32Sub (l' r' : Term) : Term :=
  match l', r' with
  (* Constant folding *)
  | TI32Const a, TI32Const b => TI32Const (i32_sub a b)
  (* x - 0 = x *)
  | _, TI32Const 0 => l'
  | _, _ => TI32Sub l' r'
  end.

Definition mk_TI32Mul (l' r' : Term) : Term :=
  match l', r' with
  (* Constant folding *)
  | TI32Const a, TI32Const b => TI32Const (i32_mul a b)
  (* x * 0 = 0 *)
  | _, TI32Const 0 => TI32Const 0
  (* 0 * x = 0 *)
  | TI32Const 0, _ => TI32Const 0
  (* x * 1 = x *)
  | _, TI32Const 1 => l'
  (* 1 * x = x *)
  | TI32Const 1, _ => r'
  | _, _ => TI32Mul l' r'
  end.

Definition mk_TI64Add (l' r' : Term) : Term :=
  match l', r' with
  | TI64Const a, TI64Const b => TI64Const (i64_add a b)
  | _, TI64Const 0 => l'
  | TI64Const 0, _ => r'
  | _, _ => TI64Add l' r'
  end.

Definition mk_TI64Sub (l' r' : Term) : Term :=
  match l', r' with
  | TI64Const a, TI64Const b => TI64Const (i64_sub a b)
  | _, TI64Const 0 => l'
  | _, _ => TI64Sub l' r'
  end.

Definition mk_TI64Mul (l' r' : Term) : Term :=
  match l', r' with
  | TI64Const a, TI64Const b => TI64Const (i64_mul a b)
  | _, TI64Const 0 => TI64Const 0
  | TI64Const 0, _ => TI64Const 0
  | _, TI64Const 1 => l'
  | TI64Const 1, _ => r'
  | _, _ => TI64Mul l' r'
  end.

Definition mk_TI32And (l' r' : Term) : Term :=
  match l', r' with
  | TI32Const a, TI32Const b => TI32Const (i32_and a b)
  | _, TI32Const 0 => TI32Const 0
  | TI32Const 0, _ => TI32Const 0
  | _, _ => TI32And l' r'
  end.

Definition mk_TI32Or (l' r' : Term) : Term :=
  match l', r' with
  | TI32Const a, TI32Const b => TI32Const (i32_or a b)
  | _, TI32Const 0 => l'
  | TI32Const 0, _ => r'
  | _, _ => TI32Or l' r'
  end.

Definition mk_TI32Xor (l' r' : Term) : Term :=
  match l', r' with
  | TI32Const a, TI32Const b => TI32Const (i32_xor a b)
  | _, TI32Const 0 => l'
  | TI32Const 0, _ => r'
  | _, _ => TI32Xor l' r'
  end.

Definition mk_TI64And (l' r' : Term) : Term :=
  match l', r' with
  | TI64Const a, TI64Const b => TI64Const (i64_and a b)
  | _, TI64Const 0 => TI64Const 0
  | TI64Const 0, _ => TI64Const 0
  | _, _ => TI64And l' r'
  end.

Definition mk_TI64Or (l' r' : Term) : Term :=
  match l', r' with
  | TI64Const a, TI64Const b => TI64Const (i64_or a b)
  | _, TI64Const 0 => l'
  | TI64Const 0, _ => r'
  | _, _ => TI64Or l' r'
  end.

Definition mk_TI64Xor (l' r' : Term) : Term :=
  match l', r' with
  | TI64Const a, TI64Const b => TI64Const (i64_xor a b)
  | _, TI64Const 0 => l'
  | TI64Const 0, _ => r'
  | _, _ => TI64Xor l' r'
  end.

Fixpoint simplify (t : Term) : Term :=
  match t with
  (* Constants are already simplified *)
  | TI32Const v => TI32Const v
  | TI64Const v => TI64Const v

  (* i32.add optimizations *)
  | TI32Add l r => mk_TI32Add (simplify l) (simplify r)

  (* i32.sub optimizations *)
  | TI32Sub l r => mk_TI32Sub (simplify l) (simplify r)

  (* i32.mul optimizations *)
  | TI32Mul l r => mk_TI32Mul (simplify l) (simplify r)

  (* i64 operations - similar patterns *)
  | TI64Add l r => mk_TI64Add (simplify l) (simplify r)

  | TI64Sub l r => mk_TI64Sub (simplify l) (simplify r)

  | TI64Mul l r => mk_TI64Mul (simplify l) (simplify r)

  (* Bitwise operations *)
  | TI32And l r => mk_TI32And (simplify l) (simplify r)

  | TI32Or l r => mk_TI32Or (simplify l) (simplify r)

  | TI32Xor l r => mk_TI32Xor (simplify l) (simplify r)

  (* i64 bitwise *)
  | TI64And l r => mk_TI64And (simplify l) (simplify r)

  | TI64Or l r => mk_TI64Or (simplify l) (simplify r)

  | TI64Xor l r => mk_TI64Xor (simplify l) (simplify r)

  (* Shifts - just recurse for now *)
  | TI32Shl l r => TI32Shl (simplify l) (simplify r)
  | TI32ShrS l r => TI32ShrS (simplify l) (simplify r)
  | TI32ShrU l r => TI32ShrU (simplify l) (simplify r)
  | TI64Shl l r => TI64Shl (simplify l) (simplify r)
  | TI64ShrS l r => TI64ShrS (simplify l) (simplify r)
  | TI64ShrU l r => TI64ShrU (simplify l) (simplify r)

  (* Comparisons - just recurse for now *)
  | TI32Eq l r => TI32Eq (simplify l) (simplify r)
  | TI32Ne l r => TI32Ne (simplify l) (simplify r)
  | TI32LtS l r => TI32LtS (simplify l) (simplify r)
  | TI32LtU l r => TI32LtU (simplify l) (simplify r)
  | TI32GtS l r => TI32GtS (simplify l) (simplify r)
  | TI32GtU l r => TI32GtU (simplify l) (simplify r)
  | TI32LeS l r => TI32LeS (simplify l) (simplify r)
  | TI32LeU l r => TI32LeU (simplify l) (simplify r)
  | TI32GeS l r => TI32GeS (simplify l) (simplify r)
  | TI32GeU l r => TI32GeU (simplify l) (simplify r)
  | TI32Eqz t => TI32Eqz (simplify t)
  | TI64Eq l r => TI64Eq (simplify l) (simplify r)
  | TI64Ne l r => TI64Ne (simplify l) (simplify r)
  | TI64LtS l r => TI64LtS (simplify l) (simplify r)
  | TI64LtU l r => TI64LtU (simplify l) (simplify r)
  | TI64GtS l r => TI64GtS (simplify l) (simplify r)
  | TI64GtU l r => TI64GtU (simplify l) (simplify r)
  | TI64LeS l r => TI64LeS (simplify l) (simplify r)
  | TI64LeU l r => TI64LeU (simplify l) (simplify r)
  | TI64GeS l r => TI64GeS (simplify l) (simplify r)
  | TI64GeU l r => TI64GeU (simplify l) (simplify r)
  | TI64Eqz t => TI64Eqz (simplify t)

  (* Stack ops *)
  | TDrop t => TDrop (simplify t)
  | TNop => TNop
  end.

(** * Core Correctness Theorem *)

(** Helper: modular arithmetic for wrapped addition *)
Lemma wrap32_add_mod : forall a b : Z,
  wrap32 (wrap32 a + wrap32 b) = wrap32 (a + b).
Proof.
  intros. unfold wrap32.
  rewrite Zplus_mod_idemp_l. rewrite Zplus_mod_idemp_r. reflexivity.
Qed.

(** Helper: modular arithmetic for wrapped subtraction *)
Lemma wrap32_sub_mod : forall a b : Z,
  wrap32 (wrap32 a - wrap32 b) = wrap32 (a - b).
Proof.
  intros. unfold wrap32.
  rewrite Zminus_mod_idemp_l. rewrite Zminus_mod_idemp_r. reflexivity.
Qed.

(** Helper: modular arithmetic for wrapped multiplication *)
Lemma wrap32_mul_mod : forall a b : Z,
  wrap32 (wrap32 a * wrap32 b) = wrap32 (a * b).
Proof.
  intros. unfold wrap32.
  rewrite Zmult_mod_idemp_l. rewrite Zmult_mod_idemp_r. reflexivity.
Qed.

(** Similar helpers for i64 *)
Lemma wrap64_add_mod : forall a b : Z,
  wrap64 (wrap64 a + wrap64 b) = wrap64 (a + b).
Proof.
  intros. unfold wrap64.
  rewrite Zplus_mod_idemp_l. rewrite Zplus_mod_idemp_r. reflexivity.
Qed.

Lemma wrap64_sub_mod : forall a b : Z,
  wrap64 (wrap64 a - wrap64 b) = wrap64 (a - b).
Proof.
  intros. unfold wrap64.
  rewrite Zminus_mod_idemp_l. rewrite Zminus_mod_idemp_r. reflexivity.
Qed.

Lemma wrap64_mul_mod : forall a b : Z,
  wrap64 (wrap64 a * wrap64 b) = wrap64 (a * b).
Proof.
  intros. unfold wrap64.
  rewrite Zmult_mod_idemp_l. rewrite Zmult_mod_idemp_r. reflexivity.
Qed.

(** ** The original equality statement is false

    [simplify_preserves_semantics] used to state
    [term_equiv t (simplify t)], i.e. [eval_term t = eval_term (simplify t)]
    for every [t]. That does not hold: [simplify] rewrites [x + 0] to [x]
    without checking that [x] is an i32 term, so an ill-typed input that
    fails to evaluate can simplify to one that succeeds. The counterexample
    below is checked by Coq. What does hold, and what the instruction-level
    theorem needs, is refinement: whenever the original term evaluates to a
    value, the simplified term evaluates to the same value
    ([simplify_refines]). *)

Example simplify_not_equiv :
  eval_term (TI32Add (TI64Const 5) (TI32Const 0)) <>
  eval_term (simplify (TI32Add (TI64Const 5) (TI32Const 0))).
Proof. simpl. discriminate. Qed.

(** ** Every non-constant term yields an already-wrapped value. *)

Lemma wrap32_idem : forall z, wrap32 (wrap32 z) = wrap32 z.
Proof. intros. unfold wrap32. apply Zmod_mod. Qed.

Lemma wrap64_idem : forall z, wrap64 (wrap64 z) = wrap64 z.
Proof. intros. unfold wrap64. apply Zmod_mod. Qed.

Lemma wrap32_ones : forall z, wrap32 z = Z.land z (Z.ones 32).
Proof. intros. unfold wrap32. rewrite Z.land_ones by lia. reflexivity. Qed.

Lemma wrap64_ones : forall z, wrap64 z = Z.land z (Z.ones 64).
Proof. intros. unfold wrap64. rewrite Z.land_ones by lia. reflexivity. Qed.

Lemma wrap32_land : forall a b, wrap32 (Z.land (wrap32 a) (wrap32 b)) = Z.land (wrap32 a) (wrap32 b).
Proof.
  intros. rewrite !wrap32_ones. apply Z.bits_inj'. intros n Hn.
  rewrite !Z.land_spec, ?Z.lor_spec, ?Z.lxor_spec, ?Z.land_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit (Z.ones 32) n); reflexivity.
Qed.

Lemma wrap32_lor : forall a b, wrap32 (Z.lor (wrap32 a) (wrap32 b)) = Z.lor (wrap32 a) (wrap32 b).
Proof.
  intros. rewrite !wrap32_ones. apply Z.bits_inj'. intros n Hn.
  rewrite !Z.land_spec, ?Z.lor_spec, ?Z.lxor_spec, ?Z.land_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit (Z.ones 32) n); reflexivity.
Qed.

Lemma wrap32_lxor : forall a b, wrap32 (Z.lxor (wrap32 a) (wrap32 b)) = Z.lxor (wrap32 a) (wrap32 b).
Proof.
  intros. rewrite !wrap32_ones. apply Z.bits_inj'. intros n Hn.
  rewrite !Z.land_spec, ?Z.lor_spec, ?Z.lxor_spec, ?Z.land_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit (Z.ones 32) n); reflexivity.
Qed.

Lemma wrap64_land : forall a b, wrap64 (Z.land (wrap64 a) (wrap64 b)) = Z.land (wrap64 a) (wrap64 b).
Proof.
  intros. rewrite !wrap64_ones. apply Z.bits_inj'. intros n Hn.
  rewrite !Z.land_spec, ?Z.lor_spec, ?Z.lxor_spec, ?Z.land_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit (Z.ones 64) n); reflexivity.
Qed.

Lemma wrap64_lor : forall a b, wrap64 (Z.lor (wrap64 a) (wrap64 b)) = Z.lor (wrap64 a) (wrap64 b).
Proof.
  intros. rewrite !wrap64_ones. apply Z.bits_inj'. intros n Hn.
  rewrite !Z.land_spec, ?Z.lor_spec, ?Z.lxor_spec, ?Z.land_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit (Z.ones 64) n); reflexivity.
Qed.

Lemma wrap64_lxor : forall a b, wrap64 (Z.lxor (wrap64 a) (wrap64 b)) = Z.lxor (wrap64 a) (wrap64 b).
Proof.
  intros. rewrite !wrap64_ones. apply Z.bits_inj'. intros n Hn.
  rewrite !Z.land_spec, ?Z.lor_spec, ?Z.lxor_spec, ?Z.land_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit (Z.ones 64) n); reflexivity.
Qed.


Lemma wrap32_shiftr : forall a n, 0 <= n ->
  wrap32 (Z.shiftr (wrap32 a) n) = Z.shiftr (wrap32 a) n.
Proof.
  intros a n Hn. rewrite !wrap32_ones. apply Z.bits_inj'. intros i Hi.
  rewrite Z.land_spec, !Z.shiftr_spec, Z.land_spec by lia.
  rewrite !Z.testbit_ones by lia.
  destruct (Z.testbit a (i + n)); simpl; [|reflexivity].
  destruct (Z.leb_spec 0 (i + n)); destruct (Z.ltb_spec (i + n) 32);
  destruct (Z.leb_spec 0 i); destruct (Z.ltb_spec i 32);
  simpl; try reflexivity; lia.
Qed.

Lemma wrap64_shiftr : forall a n, 0 <= n ->
  wrap64 (Z.shiftr (wrap64 a) n) = Z.shiftr (wrap64 a) n.
Proof.
  intros a n Hn. rewrite !wrap64_ones. apply Z.bits_inj'. intros i Hi.
  rewrite Z.land_spec, !Z.shiftr_spec, Z.land_spec by lia.
  rewrite !Z.testbit_ones by lia.
  destruct (Z.testbit a (i + n)); simpl; [|reflexivity].
  destruct (Z.leb_spec 0 (i + n)); destruct (Z.ltb_spec (i + n) 64);
  destruct (Z.leb_spec 0 i); destruct (Z.ltb_spec i 64);
  simpl; try reflexivity; lia.
Qed.

Lemma eval_i32_wrapped : forall t a,
  (forall z, t <> TI32Const z) ->
  eval_term t = TROk (VI32 a) -> wrap32 a = a.
Proof.
  intros t a Hnc H.
  destruct t; simpl in H;
    try (exfalso; eapply Hnc; reflexivity);
    repeat match type of H with
    | context [match eval_term ?x with _ => _ end] =>
        destruct (eval_term x) as [[?|?|?|?]|]; try discriminate H
    end;
    try discriminate H;
    injection H as <-;
    unfold i32_add, i32_sub, i32_mul, i32_and, i32_or, i32_xor,
      i32_shl, i32_shr_u, i32_shr_s,
      i32_eq, i32_ne, i32_lt_s, i32_lt_u, i32_gt_s, i32_gt_u,
      i32_le_s, i32_le_u, i32_ge_s, i32_ge_u, i32_eqz,
      i64_eq, i64_ne, i64_lt_s, i64_lt_u, i64_gt_s, i64_gt_u,
      i64_le_s, i64_le_u, i64_ge_s, i64_ge_u, i64_eqz;
    try apply wrap32_idem;
    try apply wrap32_land; try apply wrap32_lor; try apply wrap32_lxor;
    try (apply wrap32_shiftr; apply Z.land_nonneg; right; unfold shift_mask32; lia);
    repeat match goal with |- context [if ?c then _ else _] => destruct c end;
    reflexivity.
Qed.

Lemma eval_i64_wrapped : forall t a,
  (forall z, t <> TI64Const z) ->
  eval_term t = TROk (VI64 a) -> wrap64 a = a.
Proof.
  intros t a Hnc H.
  destruct t; simpl in H;
    try (exfalso; eapply Hnc; reflexivity);
    repeat match type of H with
    | context [match eval_term ?x with _ => _ end] =>
        destruct (eval_term x) as [[?|?|?|?]|]; try discriminate H
    end;
    try discriminate H;
    injection H as <-;
    unfold i64_add, i64_sub, i64_mul, i64_and, i64_or, i64_xor,
      i64_shl, i64_shr_u, i64_shr_s;
    try apply wrap64_idem;
    try apply wrap64_land; try apply wrap64_lor; try apply wrap64_lxor;
    try (apply wrap64_shiftr; apply Z.land_nonneg; right; unfold shift_mask64; lia).
Qed.



Lemma wrap32_0 : wrap32 0 = 0. Proof. reflexivity. Qed.
Lemma wrap64_0 : wrap64 0 = 0. Proof. reflexivity. Qed.

Ltac simpl_evals :=
  simpl;
  repeat match goal with
  | Hs : eval_term _ = _ |- _ => progress simpl in Hs
  end.

Definition as_c32 (t : Term) : option Z :=
  match t with TI32Const z => Some z | _ => None end.
Definition as_c64 (t : Term) : option Z :=
  match t with TI64Const z => Some z | _ => None end.
Lemma as_c32_none : forall t, as_c32 t = None -> forall z, t <> TI32Const z.
Proof. destruct t; simpl; congruence. Qed.
Lemma as_c64_none : forall t, as_c64 t = None -> forall z, t <> TI64Const z.
Proof. destruct t; simpl; congruence. Qed.

(* Discharge the "not a constant" premise of a wrapped-value fact
   while the operand is still concrete. *)
Ltac use_wrapped :=
  repeat match goal with
  | W : (forall z : Z, ?t <> _) -> _ |- _ =>
      lazymatch t with
      | simplify _ => fail
      | _ => tryif is_var t then fail
             else first [ specialize (W (as_c32_none t eq_refl)) | specialize (W (as_c64_none t eq_refl)) | clear W ]
      end
  end.

Ltac refine_finish :=
  simpl_evals;
  repeat (match goal with
          | |- context [match ?x with _ => _ end] => is_var x; destruct x
          end; simpl_evals);
  try discriminate;
  repeat match goal with
  | Hc : TROk (VI32 _) = TROk (VI32 _) |- _ => injection Hc as Hc; subst
  | Hc : TROk (VI64 _) = TROk (VI64 _) |- _ => injection Hc as Hc; subst
  end;
  repeat match goal with
  | Hl : ?e = TROk _ |- context [?e] => rewrite Hl
  end;
  unfold i32_add, i32_sub, i32_mul, i32_and, i32_or, i32_xor,
    i64_add, i64_sub, i64_mul, i64_and, i64_or, i64_xor;
  rewrite ?Z.add_0_r, ?Z.add_0_l, ?Z.sub_0_r, ?Z.mul_0_r, ?Z.mul_0_l,
    ?Z.mul_1_r, ?Z.mul_1_l, ?wrap32_0, ?wrap64_0,
    ?Z.land_0_r, ?Z.land_0_l, ?Z.lor_0_r, ?Z.lor_0_l, ?Z.lxor_0_r, ?Z.lxor_0_l;
  repeat match goal with
  | W : wrap32 ?a = ?a |- context [wrap32 ?a] => rewrite W
  | W : wrap64 ?a = ?a |- context [wrap64 ?a] => rewrite W
  end;
  reflexivity.

(** ** The rewriting operators, one lemma each

    [simplify] rewrites twelve operators by matching on the shape of both
    simplified operands. Each [mk_*] below is that match, copied verbatim,
    and [simplify_*_mk] shows by reduction that [simplify] computes it. The
    case analysis then runs over the folded [mk_*] term: running it over the
    unfolded body of [simplify] instead makes the kernel check take minutes
    per operator. *)

Lemma simplify_TI32Add_mk : forall x y,
  simplify (TI32Add x y) = mk_TI32Add (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI32Add_ok : forall l r a b,
  eval_term l = TROk (VI32 a) ->
  eval_term r = TROk (VI32 b) ->
  eval_term (mk_TI32Add l r) = TROk (VI32 (i32_add a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI32Add]; refine_finish.
Qed.

Lemma simplify_TI32Sub_mk : forall x y,
  simplify (TI32Sub x y) = mk_TI32Sub (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI32Sub_ok : forall l r a b,
  eval_term l = TROk (VI32 a) ->
  eval_term r = TROk (VI32 b) ->
  eval_term (mk_TI32Sub l r) = TROk (VI32 (i32_sub a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI32Sub]; refine_finish.
Qed.

Lemma simplify_TI32Mul_mk : forall x y,
  simplify (TI32Mul x y) = mk_TI32Mul (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI32Mul_ok : forall l r a b,
  eval_term l = TROk (VI32 a) ->
  eval_term r = TROk (VI32 b) ->
  eval_term (mk_TI32Mul l r) = TROk (VI32 (i32_mul a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI32Mul]; refine_finish.
Qed.

Lemma simplify_TI64Add_mk : forall x y,
  simplify (TI64Add x y) = mk_TI64Add (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI64Add_ok : forall l r a b,
  eval_term l = TROk (VI64 a) ->
  eval_term r = TROk (VI64 b) ->
  eval_term (mk_TI64Add l r) = TROk (VI64 (i64_add a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI64Add]; refine_finish.
Qed.

Lemma simplify_TI64Sub_mk : forall x y,
  simplify (TI64Sub x y) = mk_TI64Sub (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI64Sub_ok : forall l r a b,
  eval_term l = TROk (VI64 a) ->
  eval_term r = TROk (VI64 b) ->
  eval_term (mk_TI64Sub l r) = TROk (VI64 (i64_sub a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI64Sub]; refine_finish.
Qed.

Lemma simplify_TI64Mul_mk : forall x y,
  simplify (TI64Mul x y) = mk_TI64Mul (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI64Mul_ok : forall l r a b,
  eval_term l = TROk (VI64 a) ->
  eval_term r = TROk (VI64 b) ->
  eval_term (mk_TI64Mul l r) = TROk (VI64 (i64_mul a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI64Mul]; refine_finish.
Qed.

Lemma simplify_TI32And_mk : forall x y,
  simplify (TI32And x y) = mk_TI32And (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI32And_ok : forall l r a b,
  eval_term l = TROk (VI32 a) ->
  eval_term r = TROk (VI32 b) ->
  eval_term (mk_TI32And l r) = TROk (VI32 (i32_and a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI32And]; refine_finish.
Qed.

Lemma simplify_TI32Or_mk : forall x y,
  simplify (TI32Or x y) = mk_TI32Or (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI32Or_ok : forall l r a b,
  eval_term l = TROk (VI32 a) ->
  eval_term r = TROk (VI32 b) ->
  eval_term (mk_TI32Or l r) = TROk (VI32 (i32_or a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI32Or]; refine_finish.
Qed.

Lemma simplify_TI32Xor_mk : forall x y,
  simplify (TI32Xor x y) = mk_TI32Xor (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI32Xor_ok : forall l r a b,
  eval_term l = TROk (VI32 a) ->
  eval_term r = TROk (VI32 b) ->
  eval_term (mk_TI32Xor l r) = TROk (VI32 (i32_xor a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i32_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI32Xor]; refine_finish.
Qed.

Lemma simplify_TI64And_mk : forall x y,
  simplify (TI64And x y) = mk_TI64And (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI64And_ok : forall l r a b,
  eval_term l = TROk (VI64 a) ->
  eval_term r = TROk (VI64 b) ->
  eval_term (mk_TI64And l r) = TROk (VI64 (i64_and a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI64And]; refine_finish.
Qed.

Lemma simplify_TI64Or_mk : forall x y,
  simplify (TI64Or x y) = mk_TI64Or (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI64Or_ok : forall l r a b,
  eval_term l = TROk (VI64 a) ->
  eval_term r = TROk (VI64 b) ->
  eval_term (mk_TI64Or l r) = TROk (VI64 (i64_or a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI64Or]; refine_finish.
Qed.

Lemma simplify_TI64Xor_mk : forall x y,
  simplify (TI64Xor x y) = mk_TI64Xor (simplify x) (simplify y).
Proof. reflexivity. Qed.

Lemma mk_TI64Xor_ok : forall l r a b,
  eval_term l = TROk (VI64 a) ->
  eval_term r = TROk (VI64 b) ->
  eval_term (mk_TI64Xor l r) = TROk (VI64 (i64_xor a b)).
Proof.
  intros l r a b IH1 IH2.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH1) as W1.
  pose proof (fun Hn => eval_i64_wrapped _ _ Hn IH2) as W2.
  destruct l; use_wrapped; destruct r; use_wrapped;
  cbn [mk_TI64Xor]; refine_finish.
Qed.

Ltac rewrite_mk :=
  first [ rewrite simplify_TI32Add_mk
        | rewrite simplify_TI32Sub_mk
        | rewrite simplify_TI32Mul_mk
        | rewrite simplify_TI64Add_mk
        | rewrite simplify_TI64Sub_mk
        | rewrite simplify_TI64Mul_mk
        | rewrite simplify_TI32And_mk
        | rewrite simplify_TI32Or_mk
        | rewrite simplify_TI32Xor_mk
        | rewrite simplify_TI64And_mk
        | rewrite simplify_TI64Or_mk
        | rewrite simplify_TI64Xor_mk ].

Ltac apply_mk_ok :=
  first [ apply mk_TI32Add_ok
        | apply mk_TI32Sub_ok
        | apply mk_TI32Mul_ok
        | apply mk_TI64Add_ok
        | apply mk_TI64Sub_ok
        | apply mk_TI64Mul_ok
        | apply mk_TI32And_ok
        | apply mk_TI32Or_ok
        | apply mk_TI32Xor_ok
        | apply mk_TI64And_ok
        | apply mk_TI64Or_ok
        | apply mk_TI64Xor_ok ].

(** ** Simplification refines evaluation *)

Theorem simplify_refines : forall t v,
  eval_term t = TROk v -> eval_term (simplify t) = TROk v.
Proof.
  induction t; intros v H; cbn [eval_term] in H;
  try (cbn [simplify eval_term]; assumption);
  try discriminate H.
  (* Case on each operand's value. The induction hypotheses are rewritten
     along with [H], so they specialise to [eq_refl]. *)
  all: repeat match goal with
    | H : context [match eval_term ?x with _ => _ end] |- _ =>
        destruct (eval_term x) as [[?|?|?|?]|]; try discriminate H
    end.
  all: injection H as <-.
  all: repeat match goal with
    | IH : forall v, TROk ?w = TROk v -> _ |- _ => specialize (IH _ eq_refl)
    end.
  (* Operators [simplify] only recurses through. *)
  all: try (cbn [simplify eval_term];
            repeat match goal with
            | IH : eval_term (simplify _) = _ |- _ => rewrite IH
            end;
            reflexivity).
  (* Operators [simplify] rewrites. *)
  all: rewrite_mk; apply_mk_ok; assumption.
Qed.

(** * Integration: Term Evaluation to Instruction Execution *)

(** This section establishes the connection between the denotational
    semantics (eval_term) and operational semantics (exec_instrs).
    This is the key bridge for proving end-to-end optimization correctness. *)

(** Generalized compilation correctness: term_to_instrs on any starting state.
    If eval_term t = TROk v, then executing the compiled instructions on
    any state s produces a state with v pushed on top. This is stronger than
    the initial_state version and is needed for compositionality. *)
(** Helper: compose three instruction sequences using exec_instrs_app *)
Lemma exec_instrs_app3 : forall i1 i2 i3 s s1 s2 s3,
  exec_instrs i1 s = Some s1 ->
  exec_instrs i2 s1 = Some s2 ->
  exec_instrs i3 s2 = Some s3 ->
  exec_instrs (i1 ++ i2 ++ i3) s = Some s3.
Proof.
  intros.
  apply exec_instrs_app with s1.
  - assumption.
  - apply exec_instrs_app with s2; assumption.
Qed.

(** Helper: compose two instruction sequences *)
Lemma exec_instrs_app2 : forall i1 i2 s s1 s2,
  exec_instrs i1 s = Some s1 ->
  exec_instrs i2 s1 = Some s2 ->
  exec_instrs (i1 ++ i2) s = Some s2.
Proof.
  intros. apply exec_instrs_app with s1; assumption.
Qed.

(** Helper tactic-style lemma: executing a single binary op instruction *)
Lemma exec_single_binop_i32 : forall op_instr op_fn a b s,
  exec_instr op_instr (push (VI32 b) (push (VI32 a) s)) =
    Some (push (VI32 (op_fn a b)) s) ->
  exec_instrs [op_instr] (push (VI32 b) (push (VI32 a) s)) =
    Some (push (VI32 (op_fn a b)) s).
Proof.
  intros. simpl. rewrite H. reflexivity.
Qed.

Theorem term_to_instrs_correct_gen : forall t v s,
  eval_term t = TROk v ->
  exec_instrs (term_to_instrs t) s = Some (push v s).
Proof.
  induction t; simpl; intros v0 s0 Heval.

  (* Constants *)
  1-2: (injection Heval; intros; subst; reflexivity).

  (* Binary i32/i64 operations: evaluate both operands with the
     induction hypotheses, then the operator instruction. [all:] rather
     than bullets: once a bullet closes, the remaining goals are no
     longer focused and [all:] would see none of them. *)
  all: try (
    destruct (eval_term t1) as [[a|a|a|a]|] eqn:E1; try discriminate;
    destruct (eval_term t2) as [[b|b|b|b]|] eqn:E2; try discriminate;
    injection Heval; intros; subst;
    eapply exec_instrs_app3;
    [ apply IHt1; reflexivity
    | apply IHt2; reflexivity
    | simpl; unfold pop2, pop, push; destruct s0; reflexivity ]).

  (* TI32Eqz, TI64Eqz - unary operations. The operand is matched rather
     than named: Rocq 9 names it differently from Coq 8. *)
  all: try (
    match goal with
    | IH : forall v s, eval_term ?x = TROk v -> _ |- _ =>
      destruct (eval_term x) as [[a|a|a|a]|] eqn:E; try discriminate;
      injection Heval; intros; subst;
      eapply exec_instrs_app2;
      [ apply IH; reflexivity
      | simpl; unfold pop, push; destruct s0; reflexivity ]
    end).

  (* TDrop produces no value, so eval_term returns TRFail. *)
  all: try (
    match goal with
    | IH : forall v s, eval_term ?x = TROk v -> _ |- _ =>
      destruct (eval_term x); discriminate
    end).

  (* TNop produces no value. *)
  all: discriminate.
Qed.

(** The initial_state version follows directly *)
Theorem term_to_instrs_correct : forall t v,
  eval_term t = TROk v ->
  exec_instrs (term_to_instrs t) initial_state = Some (push v initial_state).
Proof.
  intros. apply term_to_instrs_correct_gen. assumption.
Qed.

(** Corollary: terms that evaluate to the same value produce the same
    instruction execution result. Note: this requires both terms to
    evaluate successfully. When both return TRFail, the instruction
    executions may differ (e.g., TNop vs TDrop (TI32Const 5) both
    evaluate to TRFail but produce different instruction sequences). *)
Corollary term_equiv_instrs_equiv : forall t1 t2 v s,
  eval_term t1 = TROk v ->
  eval_term t2 = TROk v ->
  exec_instrs (term_to_instrs t1) s = exec_instrs (term_to_instrs t2) s.
Proof.
  intros t1 t2 v s H1 H2.
  rewrite (term_to_instrs_correct_gen t1 v s H1).
  rewrite (term_to_instrs_correct_gen t2 v s H2).
  reflexivity.
Qed.

(** The optimization correctness theorem at the instruction level:
    when evaluation succeeds, optimized terms produce equivalent behavior. *)
Theorem optimization_preserves_execution : forall t v s,
  eval_term t = TROk v ->
  exec_instrs (term_to_instrs t) s = exec_instrs (term_to_instrs (simplify t)) s.
Proof.
  intros t v s Heval.
  apply (term_equiv_instrs_equiv t (simplify t) v s).
  - assumption.
  - apply simplify_refines. assumption.
Qed.

(** Finish a goal that [simpl] left as a [match] on a constant's value
    whose branches agree (Rocq 9's [simpl] no longer collapses these):
    case on each such variable, then close by reflexivity. Used by the
    simplify/*.v files. *)
Ltac const_cases :=
  simpl;
  repeat (match goal with
          | |- context [match ?z with _ => _ end] => is_var z; destruct z
          end; simpl);
  first
  [ reflexivity
  | (* a zero rule on a symbolic constant: x & 0, x * 0 *)
    unfold i32_and, i64_and, i32_mul, i64_mul;
    rewrite ?wrap32_0, ?wrap64_0, ?Z.land_0_r, ?Z.land_0_l,
      ?Z.mul_0_r, ?Z.mul_0_l;
    reflexivity ].

Close Scope Z_scope.
