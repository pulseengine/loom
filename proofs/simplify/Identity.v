(** * Algebraic Identity Optimization Proofs

    This module proves the correctness of algebraic identity optimizations
    in LOOM's WebAssembly optimizer.

    Algebraic identities transform expressions like:
      x + 0 -> x
      x * 1 -> x
      x - 0 -> x
      x & -1 -> x
      x | 0 -> x
      x ^ 0 -> x
      x << 0 -> x

    We prove that these transformations preserve semantics.
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

(** * i32 Additive Identities *)

(** x + 0 = x *)
Theorem i32_add_identity_right : forall x,
  i32_add x 0 = wrap32 x.
Proof.
  intros. unfold i32_add. f_equal. lia.
Qed.

(** 0 + x = x *)
Theorem i32_add_identity_left : forall x,
  i32_add 0 x = wrap32 x.
Proof.
  intros. unfold i32_add. f_equal. lia.
Qed.

(** x - 0 = x *)
Theorem i32_sub_identity_right : forall x,
  i32_sub x 0 = wrap32 x.
Proof.
  intros. unfold i32_sub. f_equal. lia.
Qed.

(** 0 - x = -x (wrapped) *)
Theorem i32_sub_zero_left : forall x,
  i32_sub 0 x = wrap32 (- x).
Proof.
  intros. unfold i32_sub. f_equal. lia.
Qed.

(** x - x = 0 *)
Theorem i32_sub_self_is_zero : forall x,
  i32_sub x x = 0.
Proof.
  intros. unfold i32_sub, wrap32.
  replace (x - x) with 0 by lia.
  simpl. reflexivity.
Qed.

(** * i32 Multiplicative Identities *)

(** x * 1 = x *)
Theorem i32_mul_identity_right : forall x,
  i32_mul x 1 = wrap32 x.
Proof.
  intros. unfold i32_mul. f_equal. lia.
Qed.

(** 1 * x = x *)
Theorem i32_mul_identity_left : forall x,
  i32_mul 1 x = wrap32 x.
Proof.
  intros. unfold i32_mul. f_equal. lia.
Qed.

(** x * 0 = 0 *)
Theorem i32_mul_zero_right : forall x,
  i32_mul x 0 = 0.
Proof.
  intros. unfold i32_mul, wrap32. simpl. reflexivity.
Qed.

(** 0 * x = 0 *)
Theorem i32_mul_zero_left : forall x,
  i32_mul 0 x = 0.
Proof.
  intros. unfold i32_mul, wrap32. simpl. reflexivity.
Qed.

(** * i32 Bitwise Identities *)

(** x & -1 = x (all bits set) *)
Theorem i32_and_all_ones_right : forall x,
  i32_and x (wrap32 (-1)) = wrap32 x.
Proof.
  intros. unfold i32_and.
  (* -1 wrapped to 32 bits is 2^32 - 1 = all ones *)
  (* x & all_ones = x for any x in range *)
  rewrite Z.land_ones.
  - reflexivity.
  - (* 32 >= 0 *) lia.
Qed.

(** -1 & x = x *)
Theorem i32_and_all_ones_left : forall x,
  i32_and (wrap32 (-1)) x = wrap32 x.
Proof.
  intros. unfold i32_and.
  rewrite Z.land_comm.
  rewrite Z.land_ones.
  - reflexivity.
  - lia.
Qed.

(** x & 0 = 0 *)
Theorem i32_and_zero_right : forall x,
  i32_and x 0 = 0.
Proof.
  intros. unfold i32_and, wrap32. simpl.
  apply Z.land_0_r.
Qed.

(** 0 & x = 0 *)
Theorem i32_and_zero_left : forall x,
  i32_and 0 x = 0.
Proof.
  intros. unfold i32_and, wrap32. simpl.
  apply Z.land_0_l.
Qed.

(** x | 0 = x *)
Theorem i32_or_identity_right : forall x,
  i32_or x 0 = wrap32 x.
Proof.
  intros. unfold i32_or, wrap32. simpl.
  apply Z.lor_0_r.
Qed.

(** 0 | x = x *)
Theorem i32_or_identity_left : forall x,
  i32_or 0 x = wrap32 x.
Proof.
  intros. unfold i32_or, wrap32. simpl.
  apply Z.lor_0_l.
Qed.

(** x ^ 0 = x *)
Theorem i32_xor_identity_right : forall x,
  i32_xor x 0 = wrap32 x.
Proof.
  intros. unfold i32_xor, wrap32. simpl.
  apply Z.lxor_0_r.
Qed.

(** 0 ^ x = x *)
Theorem i32_xor_identity_left : forall x,
  i32_xor 0 x = wrap32 x.
Proof.
  intros. unfold i32_xor, wrap32. simpl.
  apply Z.lxor_0_l.
Qed.

(** * i32 Shift Identities *)

(** x << 0 = x *)
Theorem i32_shl_zero : forall x,
  i32_shl x 0 = wrap32 x.
Proof.
  intros. unfold i32_shl, shift_mask32, wrap32.
  simpl. rewrite Z.shiftl_0_r. reflexivity.
Qed.

(** x >> 0 = x (unsigned) *)
Theorem i32_shr_u_zero : forall x,
  i32_shr_u x 0 = wrap32 x.
Proof.
  intros. unfold i32_shr_u, shift_mask32, wrap32.
  simpl. rewrite Z.shiftr_0_r. reflexivity.
Qed.

(** * i64 Additive Identities *)

(** x + 0 = x *)
Theorem i64_add_identity_right : forall x,
  i64_add x 0 = wrap64 x.
Proof.
  intros. unfold i64_add. f_equal. lia.
Qed.

(** 0 + x = x *)
Theorem i64_add_identity_left : forall x,
  i64_add 0 x = wrap64 x.
Proof.
  intros. unfold i64_add. f_equal. lia.
Qed.

(** x - 0 = x *)
Theorem i64_sub_identity_right : forall x,
  i64_sub x 0 = wrap64 x.
Proof.
  intros. unfold i64_sub. f_equal. lia.
Qed.

(** x - x = 0 *)
Theorem i64_sub_self_is_zero : forall x,
  i64_sub x x = 0.
Proof.
  intros. unfold i64_sub, wrap64.
  replace (x - x) with 0 by lia.
  simpl. reflexivity.
Qed.

(** * i64 Multiplicative Identities *)

(** x * 1 = x *)
Theorem i64_mul_identity_right : forall x,
  i64_mul x 1 = wrap64 x.
Proof.
  intros. unfold i64_mul. f_equal. lia.
Qed.

(** 1 * x = x *)
Theorem i64_mul_identity_left : forall x,
  i64_mul 1 x = wrap64 x.
Proof.
  intros. unfold i64_mul. f_equal. lia.
Qed.

(** x * 0 = 0 *)
Theorem i64_mul_zero_right : forall x,
  i64_mul x 0 = 0.
Proof.
  intros. unfold i64_mul, wrap64. simpl. reflexivity.
Qed.

(** 0 * x = 0 *)
Theorem i64_mul_zero_left : forall x,
  i64_mul 0 x = 0.
Proof.
  intros. unfold i64_mul, wrap64. simpl. reflexivity.
Qed.

(** * i64 Bitwise Identities *)

(** x & 0 = 0 *)
Theorem i64_and_zero_right : forall x,
  i64_and x 0 = 0.
Proof.
  intros. unfold i64_and, wrap64. simpl.
  apply Z.land_0_r.
Qed.

(** 0 & x = 0 *)
Theorem i64_and_zero_left : forall x,
  i64_and 0 x = 0.
Proof.
  intros. unfold i64_and, wrap64. simpl.
  apply Z.land_0_l.
Qed.

(** x | 0 = x *)
Theorem i64_or_identity_right : forall x,
  i64_or x 0 = wrap64 x.
Proof.
  intros. unfold i64_or, wrap64. simpl.
  apply Z.lor_0_r.
Qed.

(** 0 | x = x *)
Theorem i64_or_identity_left : forall x,
  i64_or 0 x = wrap64 x.
Proof.
  intros. unfold i64_or, wrap64. simpl.
  apply Z.lor_0_l.
Qed.

(** x ^ 0 = x *)
Theorem i64_xor_identity_right : forall x,
  i64_xor x 0 = wrap64 x.
Proof.
  intros. unfold i64_xor, wrap64. simpl.
  apply Z.lxor_0_r.
Qed.

(** 0 ^ x = x *)
Theorem i64_xor_identity_left : forall x,
  i64_xor 0 x = wrap64 x.
Proof.
  intros. unfold i64_xor, wrap64. simpl.
  apply Z.lxor_0_l.
Qed.

(** * i64 Shift Identities *)

(** x << 0 = x *)
Theorem i64_shl_zero : forall x,
  i64_shl x 0 = wrap64 x.
Proof.
  intros. unfold i64_shl, shift_mask64, wrap64.
  simpl. rewrite Z.shiftl_0_r. reflexivity.
Qed.

(** x >> 0 = x (unsigned) *)
Theorem i64_shr_u_zero : forall x,
  i64_shr_u x 0 = wrap64 x.
Proof.
  intros. unfold i64_shr_u, shift_mask64, wrap64.
  simpl. rewrite Z.shiftr_0_r. reflexivity.
Qed.

(** * Comparison Identities *)

(** x == x = 1 *)
Theorem i32_eq_reflexive : forall x,
  i32_eq x x = 1.
Proof.
  intros. unfold i32_eq.
  rewrite Z.eqb_refl. reflexivity.
Qed.

(** x != x = 0 *)
Theorem i32_ne_reflexive : forall x,
  i32_ne x x = 0.
Proof.
  intros. unfold i32_ne.
  rewrite Z.eqb_refl. reflexivity.
Qed.

(** x == x = 1 (i64) *)
Theorem i64_eq_reflexive : forall x,
  i64_eq x x = 1.
Proof.
  intros. unfold i64_eq.
  rewrite Z.eqb_refl. reflexivity.
Qed.

(** x != x = 0 (i64) *)
Theorem i64_ne_reflexive : forall x,
  i64_ne x x = 0.
Proof.
  intros. unfold i64_ne.
  rewrite Z.eqb_refl. reflexivity.
Qed.

(** * Term-Level Identity Proofs *)

(** These prove that simplify correctly applies identity rules *)

(** simplify removes x + 0 *)
Theorem simplify_add_zero_right : forall t,
  simplify (TI32Add t (TI32Const 0)) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes 0 + x *)
Theorem simplify_add_zero_left : forall t,
  simplify (TI32Add (TI32Const 0) t) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes x - 0 *)
Theorem simplify_sub_zero_right : forall t,
  simplify (TI32Sub t (TI32Const 0)) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes x * 1 *)
Theorem simplify_mul_one_right : forall t,
  simplify (TI32Mul t (TI32Const 1)) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t) eqn:Hs; try reflexivity.
  - simpl. destruct (Z.eqb z 0) eqn:Hz0; try reflexivity.
    destruct (Z.eqb z 1) eqn:Hz1; reflexivity.
Qed.

(** simplify removes 1 * x *)
Theorem simplify_mul_one_left : forall t,
  simplify (TI32Mul (TI32Const 1) t) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify x * 0 = 0 *)
Theorem simplify_mul_zero_right : forall t,
  simplify (TI32Mul t (TI32Const 0)) = TI32Const 0.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify 0 * x = 0 *)
Theorem simplify_mul_zero_left : forall t,
  simplify (TI32Mul (TI32Const 0) t) = TI32Const 0.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes x | 0 *)
Theorem simplify_or_zero_right : forall t,
  simplify (TI32Or t (TI32Const 0)) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes 0 | x *)
Theorem simplify_or_zero_left : forall t,
  simplify (TI32Or (TI32Const 0) t) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes x ^ 0 *)
Theorem simplify_xor_zero_right : forall t,
  simplify (TI32Xor t (TI32Const 0)) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes 0 ^ x *)
Theorem simplify_xor_zero_left : forall t,
  simplify (TI32Xor (TI32Const 0) t) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify x & 0 = 0 *)
Theorem simplify_and_zero_right : forall t,
  simplify (TI32And t (TI32Const 0)) = TI32Const 0.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify 0 & x = 0 *)
Theorem simplify_and_zero_left : forall t,
  simplify (TI32And (TI32Const 0) t) = TI32Const 0.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** * i64 Term-Level Identity Proofs *)

(** simplify removes x + 0 (i64) *)
Theorem simplify_i64_add_zero_right : forall t,
  simplify (TI64Add t (TI64Const 0)) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes 0 + x (i64) *)
Theorem simplify_i64_add_zero_left : forall t,
  simplify (TI64Add (TI64Const 0) t) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify removes x * 1 (i64) *)
Theorem simplify_i64_mul_one_right : forall t,
  simplify (TI64Mul t (TI64Const 1)) = simplify t.
Proof.
  intros. simpl.
  destruct (simplify t) eqn:Hs; try reflexivity.
  - simpl. destruct (Z.eqb z 0) eqn:Hz0; try reflexivity.
    destruct (Z.eqb z 1) eqn:Hz1; reflexivity.
Qed.

(** simplify 0 * x = 0 (i64) *)
Theorem simplify_i64_mul_zero_left : forall t,
  simplify (TI64Mul (TI64Const 0) t) = TI64Const 0.
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

Close Scope Z_scope.
