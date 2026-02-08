(** * Constant Folding Optimization Proofs

    This module proves the correctness of constant folding optimizations
    in LOOM's WebAssembly optimizer.

    Constant folding transforms expressions like:
      i32.add (i32.const 5) (i32.const 3) -> i32.const 8

    We prove that for every supported operation, folding two constants
    produces the same result as computing the operation at runtime.
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
Require Import WasmSemantics.
Require Import TermSemantics.
Import ListNotations.

Open Scope Z_scope.

(** * i32 Constant Folding *)

(** i32.add constant folding is correct *)
Theorem fold_i32_add_correct : forall a b,
  term_equiv
    (TI32Add (TI32Const a) (TI32Const b))
    (TI32Const (i32_add a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.sub constant folding is correct *)
Theorem fold_i32_sub_correct : forall a b,
  term_equiv
    (TI32Sub (TI32Const a) (TI32Const b))
    (TI32Const (i32_sub a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.mul constant folding is correct *)
Theorem fold_i32_mul_correct : forall a b,
  term_equiv
    (TI32Mul (TI32Const a) (TI32Const b))
    (TI32Const (i32_mul a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.and constant folding is correct *)
Theorem fold_i32_and_correct : forall a b,
  term_equiv
    (TI32And (TI32Const a) (TI32Const b))
    (TI32Const (i32_and a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.or constant folding is correct *)
Theorem fold_i32_or_correct : forall a b,
  term_equiv
    (TI32Or (TI32Const a) (TI32Const b))
    (TI32Const (i32_or a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.xor constant folding is correct *)
Theorem fold_i32_xor_correct : forall a b,
  term_equiv
    (TI32Xor (TI32Const a) (TI32Const b))
    (TI32Const (i32_xor a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.shl constant folding is correct *)
Theorem fold_i32_shl_correct : forall a b,
  term_equiv
    (TI32Shl (TI32Const a) (TI32Const b))
    (TI32Const (i32_shl a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.shr_s constant folding is correct *)
Theorem fold_i32_shr_s_correct : forall a b,
  term_equiv
    (TI32ShrS (TI32Const a) (TI32Const b))
    (TI32Const (i32_shr_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.shr_u constant folding is correct *)
Theorem fold_i32_shr_u_correct : forall a b,
  term_equiv
    (TI32ShrU (TI32Const a) (TI32Const b))
    (TI32Const (i32_shr_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.eq constant folding is correct *)
Theorem fold_i32_eq_correct : forall a b,
  term_equiv
    (TI32Eq (TI32Const a) (TI32Const b))
    (TI32Const (i32_eq a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.ne constant folding is correct *)
Theorem fold_i32_ne_correct : forall a b,
  term_equiv
    (TI32Ne (TI32Const a) (TI32Const b))
    (TI32Const (i32_ne a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.lt_s constant folding is correct *)
Theorem fold_i32_lt_s_correct : forall a b,
  term_equiv
    (TI32LtS (TI32Const a) (TI32Const b))
    (TI32Const (i32_lt_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.lt_u constant folding is correct *)
Theorem fold_i32_lt_u_correct : forall a b,
  term_equiv
    (TI32LtU (TI32Const a) (TI32Const b))
    (TI32Const (i32_lt_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.gt_s constant folding is correct *)
Theorem fold_i32_gt_s_correct : forall a b,
  term_equiv
    (TI32GtS (TI32Const a) (TI32Const b))
    (TI32Const (i32_gt_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.gt_u constant folding is correct *)
Theorem fold_i32_gt_u_correct : forall a b,
  term_equiv
    (TI32GtU (TI32Const a) (TI32Const b))
    (TI32Const (i32_gt_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.le_s constant folding is correct *)
Theorem fold_i32_le_s_correct : forall a b,
  term_equiv
    (TI32LeS (TI32Const a) (TI32Const b))
    (TI32Const (i32_le_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.le_u constant folding is correct *)
Theorem fold_i32_le_u_correct : forall a b,
  term_equiv
    (TI32LeU (TI32Const a) (TI32Const b))
    (TI32Const (i32_le_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.ge_s constant folding is correct *)
Theorem fold_i32_ge_s_correct : forall a b,
  term_equiv
    (TI32GeS (TI32Const a) (TI32Const b))
    (TI32Const (i32_ge_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.ge_u constant folding is correct *)
Theorem fold_i32_ge_u_correct : forall a b,
  term_equiv
    (TI32GeU (TI32Const a) (TI32Const b))
    (TI32Const (i32_ge_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i32.eqz constant folding is correct *)
Theorem fold_i32_eqz_correct : forall a,
  term_equiv
    (TI32Eqz (TI32Const a))
    (TI32Const (i32_eqz a)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** * i64 Constant Folding *)

(** i64.add constant folding is correct *)
Theorem fold_i64_add_correct : forall a b,
  term_equiv
    (TI64Add (TI64Const a) (TI64Const b))
    (TI64Const (i64_add a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.sub constant folding is correct *)
Theorem fold_i64_sub_correct : forall a b,
  term_equiv
    (TI64Sub (TI64Const a) (TI64Const b))
    (TI64Const (i64_sub a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.mul constant folding is correct *)
Theorem fold_i64_mul_correct : forall a b,
  term_equiv
    (TI64Mul (TI64Const a) (TI64Const b))
    (TI64Const (i64_mul a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.and constant folding is correct *)
Theorem fold_i64_and_correct : forall a b,
  term_equiv
    (TI64And (TI64Const a) (TI64Const b))
    (TI64Const (i64_and a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.or constant folding is correct *)
Theorem fold_i64_or_correct : forall a b,
  term_equiv
    (TI64Or (TI64Const a) (TI64Const b))
    (TI64Const (i64_or a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.xor constant folding is correct *)
Theorem fold_i64_xor_correct : forall a b,
  term_equiv
    (TI64Xor (TI64Const a) (TI64Const b))
    (TI64Const (i64_xor a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.shl constant folding is correct *)
Theorem fold_i64_shl_correct : forall a b,
  term_equiv
    (TI64Shl (TI64Const a) (TI64Const b))
    (TI64Const (i64_shl a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.shr_s constant folding is correct *)
Theorem fold_i64_shr_s_correct : forall a b,
  term_equiv
    (TI64ShrS (TI64Const a) (TI64Const b))
    (TI64Const (i64_shr_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.shr_u constant folding is correct *)
Theorem fold_i64_shr_u_correct : forall a b,
  term_equiv
    (TI64ShrU (TI64Const a) (TI64Const b))
    (TI64Const (i64_shr_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.eq constant folding is correct *)
Theorem fold_i64_eq_correct : forall a b,
  term_equiv
    (TI64Eq (TI64Const a) (TI64Const b))
    (TI32Const (i64_eq a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.ne constant folding is correct *)
Theorem fold_i64_ne_correct : forall a b,
  term_equiv
    (TI64Ne (TI64Const a) (TI64Const b))
    (TI32Const (i64_ne a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.lt_s constant folding is correct *)
Theorem fold_i64_lt_s_correct : forall a b,
  term_equiv
    (TI64LtS (TI64Const a) (TI64Const b))
    (TI32Const (i64_lt_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.lt_u constant folding is correct *)
Theorem fold_i64_lt_u_correct : forall a b,
  term_equiv
    (TI64LtU (TI64Const a) (TI64Const b))
    (TI32Const (i64_lt_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.gt_s constant folding is correct *)
Theorem fold_i64_gt_s_correct : forall a b,
  term_equiv
    (TI64GtS (TI64Const a) (TI64Const b))
    (TI32Const (i64_gt_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.gt_u constant folding is correct *)
Theorem fold_i64_gt_u_correct : forall a b,
  term_equiv
    (TI64GtU (TI64Const a) (TI64Const b))
    (TI32Const (i64_gt_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.le_s constant folding is correct *)
Theorem fold_i64_le_s_correct : forall a b,
  term_equiv
    (TI64LeS (TI64Const a) (TI64Const b))
    (TI32Const (i64_le_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.le_u constant folding is correct *)
Theorem fold_i64_le_u_correct : forall a b,
  term_equiv
    (TI64LeU (TI64Const a) (TI64Const b))
    (TI32Const (i64_le_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.ge_s constant folding is correct *)
Theorem fold_i64_ge_s_correct : forall a b,
  term_equiv
    (TI64GeS (TI64Const a) (TI64Const b))
    (TI32Const (i64_ge_s a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.ge_u constant folding is correct *)
Theorem fold_i64_ge_u_correct : forall a b,
  term_equiv
    (TI64GeU (TI64Const a) (TI64Const b))
    (TI32Const (i64_ge_u a b)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** i64.eqz constant folding is correct *)
Theorem fold_i64_eqz_correct : forall a,
  term_equiv
    (TI64Eqz (TI64Const a))
    (TI32Const (i64_eqz a)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** * Nested Constant Folding *)

(** Nested adds can be fully folded *)
Theorem fold_nested_i32_add : forall a b c,
  term_equiv
    (TI32Add (TI32Add (TI32Const a) (TI32Const b)) (TI32Const c))
    (TI32Const (i32_add (i32_add a b) c)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** Mixed operations can be folded *)
Theorem fold_mixed_i32 : forall a b c,
  term_equiv
    (TI32Mul (TI32Add (TI32Const a) (TI32Const b)) (TI32Const c))
    (TI32Const (i32_mul (i32_add a b) c)).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** Complex expression: (a + b) * (c - d) *)
Theorem fold_complex_expr : forall a b c d,
  term_equiv
    (TI32Mul
      (TI32Add (TI32Const a) (TI32Const b))
      (TI32Sub (TI32Const c) (TI32Const d)))
    (TI32Const (i32_mul (i32_add a b) (i32_sub c d))).
Proof.
  unfold term_equiv. intros. simpl. reflexivity.
Qed.

(** * Simplify Constant Folding Integration *)

(** simplify correctly folds i32.add constants *)
Theorem simplify_i32_add_const : forall a b,
  simplify (TI32Add (TI32Const a) (TI32Const b)) = TI32Const (i32_add a b).
Proof.
  intros. simpl. reflexivity.
Qed.

(** simplify correctly folds i32.sub constants *)
Theorem simplify_i32_sub_const : forall a b,
  simplify (TI32Sub (TI32Const a) (TI32Const b)) = TI32Const (i32_sub a b).
Proof.
  intros. simpl. reflexivity.
Qed.

(** simplify correctly folds i32.mul constants *)
Theorem simplify_i32_mul_const : forall a b,
  b <> 0 -> b <> 1 ->
  simplify (TI32Mul (TI32Const a) (TI32Const b)) = TI32Const (i32_mul a b).
Proof.
  intros. simpl.
  destruct (Z.eqb a 0) eqn:Ha0.
  - apply Z.eqb_eq in Ha0. subst. reflexivity.
  - destruct (Z.eqb a 1) eqn:Ha1.
    + apply Z.eqb_eq in Ha1. subst. reflexivity.
    + reflexivity.
Qed.

(** simplify correctly folds i64.add constants *)
Theorem simplify_i64_add_const : forall a b,
  simplify (TI64Add (TI64Const a) (TI64Const b)) = TI64Const (i64_add a b).
Proof.
  intros. simpl. reflexivity.
Qed.

(** simplify correctly folds i64.sub constants *)
Theorem simplify_i64_sub_const : forall a b,
  simplify (TI64Sub (TI64Const a) (TI64Const b)) = TI64Const (i64_sub a b).
Proof.
  intros. simpl. reflexivity.
Qed.

(** * Concrete Examples *)

(** 5 + 3 = 8 *)
Theorem example_5_plus_3 :
  simplify (TI32Add (TI32Const 5) (TI32Const 3)) = TI32Const (i32_add 5 3).
Proof. reflexivity. Qed.

(** 10 - 4 = 6 *)
Theorem example_10_minus_4 :
  simplify (TI32Sub (TI32Const 10) (TI32Const 4)) = TI32Const (i32_sub 10 4).
Proof. reflexivity. Qed.

(** 7 * 6 = 42 *)
Theorem example_7_times_6 :
  simplify (TI32Mul (TI32Const 7) (TI32Const 6)) = TI32Const (i32_mul 7 6).
Proof. reflexivity. Qed.

(** Verify the actual value: 7 * 6 = 42 *)
Lemma verify_7_times_6 : i32_mul 7 6 = 42.
Proof.
  unfold i32_mul, wrap32. simpl. reflexivity.
Qed.

Close Scope Z_scope.
