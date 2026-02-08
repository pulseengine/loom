(** * Bitwise Optimization Proofs

    This module proves the correctness of bitwise optimizations
    in LOOM's WebAssembly optimizer.

    Bitwise optimizations include:
      x & 0 = 0          (annihilation)
      x | -1 = -1        (annihilation)
      x & x = x          (idempotence)
      x | x = x          (idempotence)
      x ^ x = 0          (self-cancellation)

    We prove these preserve semantics in terms of the WASM execution model.
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

(** * AND Annihilation *)

(** x & 0 = 0 *)
Theorem and32_zero_annihilates_right : forall x,
  i32_and x 0 = 0.
Proof.
  intros. unfold i32_and, wrap32. simpl.
  apply Z.land_0_r.
Qed.

(** 0 & x = 0 *)
Theorem and32_zero_annihilates_left : forall x,
  i32_and 0 x = 0.
Proof.
  intros. unfold i32_and, wrap32. simpl.
  apply Z.land_0_l.
Qed.

(** x & 0 = 0 (i64) *)
Theorem and64_zero_annihilates_right : forall x,
  i64_and x 0 = 0.
Proof.
  intros. unfold i64_and, wrap64. simpl.
  apply Z.land_0_r.
Qed.

(** 0 & x = 0 (i64) *)
Theorem and64_zero_annihilates_left : forall x,
  i64_and 0 x = 0.
Proof.
  intros. unfold i64_and, wrap64. simpl.
  apply Z.land_0_l.
Qed.

(** * OR Annihilation *)

(** For all bits set (-1 in two's complement), OR with any value gives -1 *)
(** Note: In 32-bit, -1 = 0xFFFFFFFF = 2^32 - 1 *)

Definition all_ones_32 : Z := Z.pow 2 32 - 1.
Definition all_ones_64 : Z := Z.pow 2 64 - 1.

(** Wrapped -1 is all ones *)
Lemma wrap32_neg1 : wrap32 (-1) = all_ones_32.
Proof.
  unfold wrap32, all_ones_32.
  (* -1 mod 2^32 = 2^32 - 1 *)
  rewrite Z.mod_opp_l_nz.
  - simpl. reflexivity.
  - (* 2^32 > 0 *)
    apply Z.pow_pos_nonneg; lia.
  - (* 1 mod 2^32 <> 0 *)
    simpl. lia.
Qed.

Lemma wrap64_neg1 : wrap64 (-1) = all_ones_64.
Proof.
  unfold wrap64, all_ones_64.
  rewrite Z.mod_opp_l_nz.
  - simpl. reflexivity.
  - apply Z.pow_pos_nonneg; lia.
  - simpl. lia.
Qed.

(** x | all_ones = all_ones *)
Theorem or32_all_ones_right : forall x,
  i32_or x all_ones_32 = all_ones_32.
Proof.
  intros. unfold i32_or, all_ones_32.
  (* All 32-bit values OR'd with all ones gives all ones *)
  rewrite Z.lor_ones.
  - unfold wrap32. apply Z.mod_small.
    split.
    + apply Z.mod_pos_bound.
      apply Z.pow_pos_nonneg; lia.
    + apply Z.mod_pos_bound.
      apply Z.pow_pos_nonneg; lia.
  - lia.
Qed.

(** all_ones | x = all_ones *)
Theorem or32_all_ones_left : forall x,
  i32_or all_ones_32 x = all_ones_32.
Proof.
  intros. unfold i32_or.
  rewrite Z.lor_comm.
  unfold all_ones_32.
  rewrite Z.lor_ones.
  - unfold wrap32. apply Z.mod_small.
    split.
    + apply Z.mod_pos_bound.
      apply Z.pow_pos_nonneg; lia.
    + apply Z.mod_pos_bound.
      apply Z.pow_pos_nonneg; lia.
  - lia.
Qed.

(** * Idempotence *)

(** x & x = x (for wrapped values) *)
Theorem and32_idempotent : forall x,
  i32_and x x = wrap32 x.
Proof.
  intros. unfold i32_and.
  apply Z.land_diag.
Qed.

(** x | x = x (for wrapped values) *)
Theorem or32_idempotent : forall x,
  i32_or x x = wrap32 x.
Proof.
  intros. unfold i32_or.
  apply Z.lor_diag.
Qed.

(** x & x = x (i64) *)
Theorem and64_idempotent : forall x,
  i64_and x x = wrap64 x.
Proof.
  intros. unfold i64_and.
  apply Z.land_diag.
Qed.

(** x | x = x (i64) *)
Theorem or64_idempotent : forall x,
  i64_or x x = wrap64 x.
Proof.
  intros. unfold i64_or.
  apply Z.lor_diag.
Qed.

(** * Self-Cancellation (XOR) *)

(** x ^ x = 0 *)
Theorem xor32_self_cancels : forall x,
  i32_xor x x = 0.
Proof.
  intros. unfold i32_xor.
  apply Z.lxor_nilpotent.
Qed.

(** x ^ x = 0 (i64) *)
Theorem xor64_self_cancels : forall x,
  i64_xor x x = 0.
Proof.
  intros. unfold i64_xor.
  apply Z.lxor_nilpotent.
Qed.

(** * Commutativity *)

(** x & y = y & x *)
Theorem and32_comm : forall x y,
  i32_and x y = i32_and y x.
Proof.
  intros. unfold i32_and.
  rewrite Z.land_comm. reflexivity.
Qed.

(** x | y = y | x *)
Theorem or32_comm : forall x y,
  i32_or x y = i32_or y x.
Proof.
  intros. unfold i32_or.
  rewrite Z.lor_comm. reflexivity.
Qed.

(** x ^ y = y ^ x *)
Theorem xor32_comm : forall x y,
  i32_xor x y = i32_xor y x.
Proof.
  intros. unfold i32_xor.
  rewrite Z.lxor_comm. reflexivity.
Qed.

(** * Associativity *)

(** (x & y) & z = x & (y & z) *)
Theorem and32_assoc : forall x y z,
  i32_and (i32_and x y) z = i32_and x (i32_and y z).
Proof.
  intros. unfold i32_and.
  rewrite Z.land_assoc.
  (* Need to show land is idempotent when applying wrap32 *)
  f_equal.
  (* wrap32 (land (wrap32 x) (wrap32 y)) = land (wrap32 x) (wrap32 y) *)
  (* because result of land of two values < 2^32 is also < 2^32 *)
  symmetry. apply Z.land_assoc.
Qed.

(** (x | y) | z = x | (y | z) *)
Theorem or32_assoc : forall x y z,
  i32_or (i32_or x y) z = i32_or x (i32_or y z).
Proof.
  intros. unfold i32_or.
  rewrite Z.lor_assoc.
  symmetry. apply Z.lor_assoc.
Qed.

(** (x ^ y) ^ z = x ^ (y ^ z) *)
Theorem xor32_assoc : forall x y z,
  i32_xor (i32_xor x y) z = i32_xor x (i32_xor y z).
Proof.
  intros. unfold i32_xor.
  rewrite Z.lxor_assoc.
  symmetry. apply Z.lxor_assoc.
Qed.

(** * De Morgan's Laws *)

(** For wrapped values that fit in 32 bits *)

(** ~(x & y) = ~x | ~y (conceptually) *)
(** In WASM terms: xor with -1 gives bitwise NOT *)

(** * Absorption Laws *)

(** x & (x | y) = x *)
Theorem and_or_absorption : forall x y,
  i32_and (wrap32 x) (i32_or (wrap32 x) (wrap32 y)) = wrap32 x.
Proof.
  intros. unfold i32_and, i32_or, wrap32.
  rewrite Z.land_lor_distr_l.
  rewrite Z.land_diag.
  rewrite Z.lor_land_distr_l.
  rewrite Z.lor_diag.
  rewrite Z.land_diag.
  reflexivity.
Qed.

(** x | (x & y) = x *)
Theorem or_and_absorption : forall x y,
  i32_or (wrap32 x) (i32_and (wrap32 x) (wrap32 y)) = wrap32 x.
Proof.
  intros. unfold i32_and, i32_or, wrap32.
  rewrite Z.lor_land_distr_l.
  rewrite Z.lor_diag.
  rewrite Z.land_lor_distr_l.
  rewrite Z.land_diag.
  rewrite Z.lor_diag.
  reflexivity.
Qed.

(** * XOR Properties *)

(** x ^ 0 = x *)
Theorem xor32_identity : forall x,
  i32_xor x 0 = wrap32 x.
Proof.
  intros. unfold i32_xor, wrap32. simpl.
  apply Z.lxor_0_r.
Qed.

(** x ^ x ^ x = x *)
Theorem xor32_triple : forall x,
  i32_xor (i32_xor x x) x = wrap32 x.
Proof.
  intros. unfold i32_xor.
  rewrite Z.lxor_nilpotent.
  simpl. apply Z.lxor_0_l.
Qed.

(** (x ^ y) ^ y = x *)
Theorem xor32_cancel_right : forall x y,
  i32_xor (i32_xor (wrap32 x) (wrap32 y)) (wrap32 y) = wrap32 x.
Proof.
  intros. unfold i32_xor, wrap32.
  rewrite Z.lxor_assoc.
  rewrite Z.lxor_nilpotent.
  rewrite Z.lxor_0_r.
  reflexivity.
Qed.

(** * Term-Level Proofs *)

(** These connect the Z-level proofs to the term simplifier *)

(** simplify correctly handles x & 0 *)
Theorem simplify_and_annihilates : forall t,
  eval_term (simplify (TI32And t (TI32Const 0))) = eval_term (TI32Const 0).
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify correctly handles 0 & x *)
Theorem simplify_and_annihilates_left : forall t,
  eval_term (simplify (TI32And (TI32Const 0) t)) = eval_term (TI32Const 0).
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify correctly handles x | 0 *)
Theorem simplify_or_identity : forall t,
  eval_term (simplify (TI32Or t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** simplify correctly handles x ^ 0 *)
Theorem simplify_xor_identity : forall t,
  eval_term (simplify (TI32Xor t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros. simpl.
  destruct (simplify t); reflexivity.
Qed.

(** * Bitwise + Arithmetic Interactions *)

(** x & (x - 1) clears the lowest set bit *)
(** This is useful for popcount-like operations *)

(** x & ~(x - 1) isolates the lowest set bit *)
(** These are advanced optimizations not currently in LOOM *)

(** * Shift + Bitwise Interactions *)

(** (x << n) & mask can be optimized based on mask *)
(** (x >> n) & mask can be optimized based on mask *)
(** These require more complex analysis *)

Close Scope Z_scope.
