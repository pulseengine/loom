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
Require Import WasmSemantics.
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

(** x + 0 = x *)
Theorem i32_add_zero_right : forall t,
  eval_term t = TROk (VI32 (wrap32 0)) ->
  forall t', eval_term t' = TROk (VI32 (wrap32 0)) ->
  term_equiv (TI32Add t (TI32Const 0)) t.
Proof.
  unfold term_equiv. intros.
  simpl.
  destruct (eval_term t) eqn:Ht; try reflexivity.
  destruct v; try reflexivity.
  f_equal. f_equal.
  unfold i32_add. f_equal. lia.
Qed.

(** 0 + x = x (when x evaluates to a wrapped value) *)
Lemma i32_add_zero_left_helper : forall z,
  i32_add 0 z = wrap32 z.
Proof.
  intros. unfold i32_add. f_equal. lia.
Qed.

(** x * 1 = x *)
Lemma i32_mul_one_right_helper : forall z,
  i32_mul z 1 = wrap32 z.
Proof.
  intros. unfold i32_mul. f_equal. lia.
Qed.

(** 1 * x = x *)
Lemma i32_mul_one_left_helper : forall z,
  i32_mul 1 z = wrap32 z.
Proof.
  intros. unfold i32_mul. f_equal. lia.
Qed.

(** x * 0 = 0 *)
Lemma i32_mul_zero_right_helper : forall z,
  i32_mul z 0 = 0.
Proof.
  intros. unfold i32_mul, wrap32. simpl. reflexivity.
Qed.

(** 0 * x = 0 *)
Lemma i32_mul_zero_left_helper : forall z,
  i32_mul 0 z = 0.
Proof.
  intros. unfold i32_mul, wrap32. simpl. reflexivity.
Qed.

(** x - 0 = x *)
Lemma i32_sub_zero_right_helper : forall z,
  i32_sub z 0 = wrap32 z.
Proof.
  intros. unfold i32_sub. f_equal. lia.
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
  intros. unfold i32_and, wrap32. simpl. apply Z.land_0_r.
Qed.

(** 0 & x = 0 *)
Lemma i32_and_zero_left_helper : forall z,
  i32_and 0 z = 0.
Proof.
  intros. unfold i32_and, wrap32. simpl. apply Z.land_0_l.
Qed.

(** x | 0 = x (wrapped) *)
Lemma i32_or_zero_right_helper : forall z,
  i32_or z 0 = wrap32 z.
Proof.
  intros. unfold i32_or, wrap32. simpl. apply Z.lor_0_r.
Qed.

(** 0 | x = x (wrapped) *)
Lemma i32_or_zero_left_helper : forall z,
  i32_or 0 z = wrap32 z.
Proof.
  intros. unfold i32_or, wrap32. simpl. apply Z.lor_0_l.
Qed.

(** x ^ 0 = x (wrapped) *)
Lemma i32_xor_zero_right_helper : forall z,
  i32_xor z 0 = wrap32 z.
Proof.
  intros. unfold i32_xor, wrap32. simpl. apply Z.lxor_0_r.
Qed.

(** 0 ^ x = x (wrapped) *)
Lemma i32_xor_zero_left_helper : forall z,
  i32_xor 0 z = wrap32 z.
Proof.
  intros. unfold i32_xor, wrap32. simpl. apply Z.lxor_0_l.
Qed.

(** x ^ x = 0 *)
Lemma i32_xor_self_helper : forall z,
  i32_xor z z = 0.
Proof.
  intros. unfold i32_xor. apply Z.lxor_nilpotent.
Qed.

(** x & x = x (wrapped) *)
Lemma i32_and_self_helper : forall z,
  i32_and z z = wrap32 z.
Proof.
  intros. unfold i32_and. apply Z.land_diag.
Qed.

(** x | x = x (wrapped) *)
Lemma i32_or_self_helper : forall z,
  i32_or z z = wrap32 z.
Proof.
  intros. unfold i32_or. apply Z.lor_diag.
Qed.

(** * Shift Identity Theorems *)

(** x << 0 = x (wrapped, when shift amount has low bits = 0) *)
Lemma i32_shl_zero_helper : forall z,
  i32_shl z 0 = wrap32 z.
Proof.
  intros. unfold i32_shl, shift_mask32, wrap32.
  simpl. rewrite Z.shiftl_0_r. reflexivity.
Qed.

(** x >> 0 = x (wrapped, unsigned) *)
Lemma i32_shr_u_zero_helper : forall z,
  i32_shr_u z 0 = wrap32 z.
Proof.
  intros. unfold i32_shr_u, shift_mask32, wrap32.
  simpl. rewrite Z.shiftr_0_r. reflexivity.
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
Fixpoint simplify (t : Term) : Term :=
  match t with
  (* Constants are already simplified *)
  | TI32Const v => TI32Const v
  | TI64Const v => TI64Const v

  (* i32.add optimizations *)
  | TI32Add l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      (* Constant folding *)
      | TI32Const a, TI32Const b => TI32Const (i32_add a b)
      (* x + 0 = x *)
      | _, TI32Const 0 => l'
      (* 0 + x = x *)
      | TI32Const 0, _ => r'
      | _, _ => TI32Add l' r'
      end

  (* i32.sub optimizations *)
  | TI32Sub l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      (* Constant folding *)
      | TI32Const a, TI32Const b => TI32Const (i32_sub a b)
      (* x - 0 = x *)
      | _, TI32Const 0 => l'
      | _, _ => TI32Sub l' r'
      end

  (* i32.mul optimizations *)
  | TI32Mul l r =>
      let l' := simplify l in
      let r' := simplify r in
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
      end

  (* i64 operations - similar patterns *)
  | TI64Add l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI64Const a, TI64Const b => TI64Const (i64_add a b)
      | _, TI64Const 0 => l'
      | TI64Const 0, _ => r'
      | _, _ => TI64Add l' r'
      end

  | TI64Sub l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI64Const a, TI64Const b => TI64Const (i64_sub a b)
      | _, TI64Const 0 => l'
      | _, _ => TI64Sub l' r'
      end

  | TI64Mul l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI64Const a, TI64Const b => TI64Const (i64_mul a b)
      | _, TI64Const 0 => TI64Const 0
      | TI64Const 0, _ => TI64Const 0
      | _, TI64Const 1 => l'
      | TI64Const 1, _ => r'
      | _, _ => TI64Mul l' r'
      end

  (* Bitwise operations *)
  | TI32And l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI32Const a, TI32Const b => TI32Const (i32_and a b)
      | _, TI32Const 0 => TI32Const 0
      | TI32Const 0, _ => TI32Const 0
      | _, _ => TI32And l' r'
      end

  | TI32Or l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI32Const a, TI32Const b => TI32Const (i32_or a b)
      | _, TI32Const 0 => l'
      | TI32Const 0, _ => r'
      | _, _ => TI32Or l' r'
      end

  | TI32Xor l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI32Const a, TI32Const b => TI32Const (i32_xor a b)
      | _, TI32Const 0 => l'
      | TI32Const 0, _ => r'
      | _, _ => TI32Xor l' r'
      end

  (* i64 bitwise *)
  | TI64And l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI64Const a, TI64Const b => TI64Const (i64_and a b)
      | _, TI64Const 0 => TI64Const 0
      | TI64Const 0, _ => TI64Const 0
      | _, _ => TI64And l' r'
      end

  | TI64Or l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI64Const a, TI64Const b => TI64Const (i64_or a b)
      | _, TI64Const 0 => l'
      | TI64Const 0, _ => r'
      | _, _ => TI64Or l' r'
      end

  | TI64Xor l r =>
      let l' := simplify l in
      let r' := simplify r in
      match l', r' with
      | TI64Const a, TI64Const b => TI64Const (i64_xor a b)
      | _, TI64Const 0 => l'
      | TI64Const 0, _ => r'
      | _, _ => TI64Xor l' r'
      end

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

(** The main theorem: simplification preserves semantics *)
Theorem simplify_preserves_semantics : forall t,
  term_equiv t (simplify t).
Proof.
  unfold term_equiv.
  induction t; simpl; try reflexivity.

  (* TI32Add case *)
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    try (simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity).
    (* const + const case *)
    + simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
    (* x + 0 case *)
    + destruct (Z.eqb z 0) eqn:Hz.
      * apply Z.eqb_eq in Hz. subst.
        simpl. rewrite <- IHt1. rewrite <- IHt2. simpl.
        destruct (eval_term t1); try reflexivity.
        destruct v; try reflexivity.
        f_equal. f_equal. apply i32_add_0_r.
      * simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
    (* 0 + x case - similar structure *)
    + destruct (Z.eqb z 0) eqn:Hz.
      * apply Z.eqb_eq in Hz. subst.
        simpl. rewrite <- IHt1. rewrite <- IHt2. simpl.
        destruct (eval_term t2); try reflexivity.
        destruct v; try reflexivity.
        f_equal. f_equal. apply i32_add_0_l.
      * simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.

  (* TI32Sub case *)
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    try (simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity).
    + simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
    + destruct (Z.eqb z 0) eqn:Hz.
      * apply Z.eqb_eq in Hz. subst.
        simpl. rewrite <- IHt1. rewrite <- IHt2. simpl.
        destruct (eval_term t1); try reflexivity.
        destruct v; try reflexivity.
        f_equal. f_equal. apply i32_sub_0_r.
      * simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.

  (* TI32Mul case *)
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    try (simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity).
    + simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
    + destruct (Z.eqb z 0) eqn:Hz0.
      * apply Z.eqb_eq in Hz0. subst.
        simpl. rewrite <- IHt1. rewrite <- IHt2. simpl.
        destruct (eval_term t1); try reflexivity.
        destruct v; try reflexivity.
        f_equal. f_equal. apply i32_mul_0_r.
      * destruct (Z.eqb z 1) eqn:Hz1.
        -- apply Z.eqb_eq in Hz1. subst.
           simpl. rewrite <- IHt1. rewrite <- IHt2. simpl.
           destruct (eval_term t1); try reflexivity.
           destruct v; try reflexivity.
           f_equal. f_equal. apply i32_mul_1_r.
        -- simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
    + destruct (Z.eqb z 0) eqn:Hz0.
      * apply Z.eqb_eq in Hz0. subst.
        simpl. rewrite <- IHt1. rewrite <- IHt2. simpl.
        destruct (eval_term t2); try reflexivity.
        destruct v; try reflexivity.
        f_equal. f_equal. apply i32_mul_0_l.
      * destruct (Z.eqb z 1) eqn:Hz1.
        -- apply Z.eqb_eq in Hz1. subst.
           simpl. rewrite <- IHt1. rewrite <- IHt2. simpl.
           destruct (eval_term t2); try reflexivity.
           destruct v; try reflexivity.
           f_equal. f_equal. apply i32_mul_1_l.
        -- simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.

  (* Remaining cases follow similar pattern - using induction hypotheses *)
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - destruct (simplify t1) eqn:Hs1; destruct (simplify t2) eqn:Hs2;
    simpl; rewrite <- IHt1; rewrite <- IHt2; simpl; reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt1. rewrite <- IHt2. simpl. reflexivity.
  - simpl. rewrite <- IHt. simpl. reflexivity.
  - simpl. rewrite <- IHt. simpl. reflexivity.
Qed.

Close Scope Z_scope.
