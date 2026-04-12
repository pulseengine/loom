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
From proofs Require Import WasmSemantics.
From proofs Require Import TermSemantics.
Import ListNotations.

Open Scope Z_scope.

(** * AND Annihilation *)

(** Helper: wrap32 0 = 0 *)
Lemma wrap32_0 : wrap32 0 = 0.
Proof. unfold wrap32. reflexivity. Qed.

(** Helper: wrap64 0 = 0 *)
Lemma wrap64_0 : wrap64 0 = 0.
Proof. unfold wrap64. reflexivity. Qed.

(** x & 0 = 0 *)
Theorem and32_zero_annihilates_right : forall x,
  i32_and x 0 = 0.
Proof.
  intros. unfold i32_and.
  rewrite wrap32_0.
  rewrite Z.land_0_r. reflexivity.
Qed.

(** 0 & x = 0 *)
Theorem and32_zero_annihilates_left : forall x,
  i32_and 0 x = 0.
Proof.
  intros. unfold i32_and.
  rewrite wrap32_0.
  rewrite Z.land_0_l. reflexivity.
Qed.

(** x & 0 = 0 (i64) *)
Theorem and64_zero_annihilates_right : forall x,
  i64_and x 0 = 0.
Proof.
  intros. unfold i64_and.
  rewrite wrap64_0.
  rewrite Z.land_0_r. reflexivity.
Qed.

(** 0 & x = 0 (i64) *)
Theorem and64_zero_annihilates_left : forall x,
  i64_and 0 x = 0.
Proof.
  intros. unfold i64_and.
  rewrite wrap64_0.
  rewrite Z.land_0_l. reflexivity.
Qed.

(** * OR Annihilation *)

(** For all bits set (-1 in two's complement), OR with any value gives -1 *)
(** Note: In 32-bit, -1 = 0xFFFFFFFF = 2^32 - 1 *)

Definition all_ones_32 : Z := Z.pow 2 32 - 1.
Definition all_ones_64 : Z := Z.pow 2 64 - 1.

(** Wrapped -1 is all ones *)
Lemma wrap32_neg1 : wrap32 (-1) = all_ones_32.
Proof.
  (* Direct computation: -1 mod 2^32 = 2^32 - 1 *)
  reflexivity.
Qed.

Lemma wrap64_neg1 : wrap64 (-1) = all_ones_64.
Proof.
  reflexivity.
Qed.

(** Helper: wrap32 of all_ones_32 is all_ones_32 *)
Lemma wrap32_all_ones : wrap32 all_ones_32 = all_ones_32.
Proof.
  unfold wrap32, all_ones_32. reflexivity.
Qed.

(** Helper: wrap64 of all_ones_64 is all_ones_64 *)
Lemma wrap64_all_ones : wrap64 all_ones_64 = all_ones_64.
Proof.
  unfold wrap64, all_ones_64. reflexivity.
Qed.

(** Helper: wrap32 produces values in [0, 2^32) *)
Lemma wrap32_range : forall x, 0 <= wrap32 x < 2 ^ 32.
Proof.
  intros. unfold wrap32. apply Z.mod_pos_bound. lia.
Qed.

(** Helper: Z.lor with all-ones mask via bit-level reasoning.
    For any x in [0, 2^32), x | (2^32 - 1) = 2^32 - 1.
    Proof: 2^32 - 1 = Z.ones 32 has all bits set in [0, 32),
    so OR with any value in that range cannot add higher bits. *)
Lemma lor_all_ones_32 : forall x,
  0 <= x < 2 ^ 32 ->
  Z.lor x (2 ^ 32 - 1) = 2 ^ 32 - 1.
Proof.
  intros x Hx.
  replace (2 ^ 32 - 1) with (Z.ones 32) by reflexivity.
  apply Z.bits_inj. intros n Hn.
  rewrite Z.lor_spec.
  destruct (Z.ltb n 32) eqn:Hlt.
  - (* n < 32: bit n of Z.ones 32 is true, so OR is true *)
    apply Z.ltb_lt in Hlt.
    rewrite Z.ones_spec_low by lia.
    rewrite Bool.orb_true_r. reflexivity.
  - (* n >= 32: bit n of Z.ones 32 is false, bit n of x is false *)
    apply Z.ltb_ge in Hlt.
    rewrite Z.ones_spec_high by lia.
    rewrite Bool.orb_false_r.
    apply Z.bits_above_log2.
    + lia.
    + destruct (Z.eq_dec x 0) as [->|Hne].
      * simpl. lia.
      * apply Z.log2_lt_pow2; lia.
Qed.

(** x | all_ones = all_ones *)
Theorem or32_all_ones_right : forall x,
  i32_or x all_ones_32 = all_ones_32.
Proof.
  intros. unfold i32_or, all_ones_32.
  rewrite wrap32_all_ones.
  apply lor_all_ones_32. apply wrap32_range.
Qed.

(** all_ones | x = all_ones *)
Theorem or32_all_ones_left : forall x,
  i32_or all_ones_32 x = all_ones_32.
Proof.
  intros. rewrite or32_comm. apply or32_all_ones_right.
Qed.

(** * Idempotence *)

(** x & x = x (for wrapped values) *)
Theorem and32_idempotent : forall x,
  i32_and x x = wrap32 x.
Proof.
  intros. unfold i32_and.
  rewrite Z.land_diag. reflexivity.
Qed.

(** x | x = x (for wrapped values) *)
Theorem or32_idempotent : forall x,
  i32_or x x = wrap32 x.
Proof.
  intros. unfold i32_or.
  rewrite Z.lor_diag. reflexivity.
Qed.

(** x & x = x (i64) *)
Theorem and64_idempotent : forall x,
  i64_and x x = wrap64 x.
Proof.
  intros. unfold i64_and.
  rewrite Z.land_diag. reflexivity.
Qed.

(** x | x = x (i64) *)
Theorem or64_idempotent : forall x,
  i64_or x x = wrap64 x.
Proof.
  intros. unfold i64_or.
  rewrite Z.lor_diag. reflexivity.
Qed.

(** * Self-Cancellation (XOR) *)

(** x ^ x = 0 *)
Theorem xor32_self_cancels : forall x,
  i32_xor x x = 0.
Proof.
  intros. unfold i32_xor.
  rewrite Z.lxor_nilpotent. reflexivity.
Qed.

(** x ^ x = 0 (i64) *)
Theorem xor64_self_cancels : forall x,
  i64_xor x x = 0.
Proof.
  intros. unfold i64_xor.
  rewrite Z.lxor_nilpotent. reflexivity.
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

(** Helper: Z.land of values in [0, 2^32) stays in [0, 2^32) *)
Lemma land_range_32 : forall x y,
  0 <= x < 2 ^ 32 -> 0 <= y < 2 ^ 32 ->
  0 <= Z.land x y < 2 ^ 32.
Proof.
  intros x y Hx Hy. split.
  - apply Z.land_nonneg. left. lia.
  - apply Z.log2_lt_cancel.
    apply Z.le_lt_trans with (m := Z.min (Z.log2 x) (Z.log2 y)).
    + apply Z.log2_land; lia.
    + apply Z.min_lt_iff.
      destruct (Z.eq_dec x 0) as [->|Hxne]; [left; simpl; lia|].
      destruct (Z.eq_dec y 0) as [->|Hyne]; [right; simpl; lia|].
      left. apply Z.log2_lt_pow2; lia.
Qed.

(** Helper: wrap32 is identity on values already in range *)
Lemma wrap32_small : forall x, 0 <= x < 2 ^ 32 -> wrap32 x = x.
Proof.
  intros x Hx. unfold wrap32. apply Z.mod_small. assumption.
Qed.

(** Helper: Z.lor of values in [0, 2^32) stays in [0, 2^32) *)
Lemma lor_range_32 : forall x y,
  0 <= x < 2 ^ 32 -> 0 <= y < 2 ^ 32 ->
  0 <= Z.lor x y < 2 ^ 32.
Proof.
  intros x y Hx Hy. split.
  - apply Z.lor_nonneg; lia.
  - destruct (Z.eq_dec x 0) as [->|Hxne].
    + rewrite Z.lor_0_l. lia.
    + destruct (Z.eq_dec y 0) as [->|Hyne].
      * rewrite Z.lor_0_r. lia.
      * apply Z.log2_lt_cancel.
        apply Z.le_lt_trans with (m := Z.max (Z.log2 x) (Z.log2 y)).
        -- apply Z.log2_lor; lia.
        -- apply Z.max_lt_iff. split; apply Z.log2_lt_pow2; lia.
Qed.

(** Helper: Z.lxor of values in [0, 2^32) stays in [0, 2^32) *)
Lemma lxor_range_32 : forall x y,
  0 <= x < 2 ^ 32 -> 0 <= y < 2 ^ 32 ->
  0 <= Z.lxor x y < 2 ^ 32.
Proof.
  intros x y Hx Hy. split.
  - apply Z.lxor_nonneg; lia.
  - destruct (Z.eq_dec x 0) as [->|Hxne].
    + rewrite Z.lxor_0_l. lia.
    + destruct (Z.eq_dec y 0) as [->|Hyne].
      * rewrite Z.lxor_0_r. lia.
      * apply Z.log2_lt_cancel.
        apply Z.le_lt_trans with (m := Z.max (Z.log2 x) (Z.log2 y)).
        -- apply Z.log2_lxor; lia.
        -- apply Z.max_lt_iff. split; apply Z.log2_lt_pow2; lia.
Qed.

(** (x & y) & z = x & (y & z) *)
Theorem and32_assoc : forall x y z,
  i32_and (i32_and x y) z = i32_and x (i32_and y z).
Proof.
  intros. unfold i32_and.
  (* wrap32 (Z.land (wrap32 x) (wrap32 y)) = Z.land ... because result is in range *)
  rewrite (wrap32_small (Z.land (wrap32 x) (wrap32 y)))
    by (apply land_range_32; apply wrap32_range).
  rewrite (wrap32_small (Z.land (wrap32 y) (wrap32 z)))
    by (apply land_range_32; apply wrap32_range).
  rewrite Z.land_assoc. reflexivity.
Qed.

(** (x | y) | z = x | (y | z) *)
Theorem or32_assoc : forall x y z,
  i32_or (i32_or x y) z = i32_or x (i32_or y z).
Proof.
  intros. unfold i32_or.
  rewrite (wrap32_small (Z.lor (wrap32 x) (wrap32 y)))
    by (apply lor_range_32; apply wrap32_range).
  rewrite (wrap32_small (Z.lor (wrap32 y) (wrap32 z)))
    by (apply lor_range_32; apply wrap32_range).
  rewrite Z.lor_assoc. reflexivity.
Qed.

(** (x ^ y) ^ z = x ^ (y ^ z) *)
Theorem xor32_assoc : forall x y z,
  i32_xor (i32_xor x y) z = i32_xor x (i32_xor y z).
Proof.
  intros. unfold i32_xor.
  rewrite (wrap32_small (Z.lxor (wrap32 x) (wrap32 y)))
    by (apply lxor_range_32; apply wrap32_range).
  rewrite (wrap32_small (Z.lxor (wrap32 y) (wrap32 z)))
    by (apply lxor_range_32; apply wrap32_range).
  rewrite Z.lxor_assoc. reflexivity.
Qed.

(** * De Morgan's Laws *)

(** For wrapped values that fit in 32 bits *)

(** ~(x & y) = ~x | ~y (conceptually) *)
(** In WASM terms: xor with -1 gives bitwise NOT *)

(** * Absorption Laws *)

(** Helper: a & (a | b) = a for non-negative integers (absorption) *)
Lemma land_lor_absorb : forall a b,
  0 <= a -> 0 <= b ->
  Z.land a (Z.lor a b) = a.
Proof.
  intros a b Ha Hb.
  apply Z.bits_inj. intros n Hn.
  rewrite Z.land_spec, Z.lor_spec.
  destruct (Z.testbit a n); simpl; reflexivity.
Qed.

(** Helper: a | (a & b) = a for non-negative integers (absorption) *)
Lemma lor_land_absorb : forall a b,
  0 <= a -> 0 <= b ->
  Z.lor a (Z.land a b) = a.
Proof.
  intros a b Ha Hb.
  apply Z.bits_inj. intros n Hn.
  rewrite Z.lor_spec, Z.land_spec.
  destruct (Z.testbit a n); simpl; reflexivity.
Qed.

(** x & (x | y) = x *)
Theorem and_or_absorption : forall x y,
  i32_and (wrap32 x) (i32_or (wrap32 x) (wrap32 y)) = wrap32 x.
Proof.
  intros. unfold i32_and, i32_or.
  rewrite (wrap32_small (wrap32 x)) by (apply wrap32_range).
  rewrite (wrap32_small (Z.lor (wrap32 x) (wrap32 y)))
    by (apply lor_range_32; apply wrap32_range).
  apply land_lor_absorb; destruct (wrap32_range x); lia.
Qed.

(** x | (x & y) = x *)
Theorem or_and_absorption : forall x y,
  i32_or (wrap32 x) (i32_and (wrap32 x) (wrap32 y)) = wrap32 x.
Proof.
  intros. unfold i32_or, i32_and.
  rewrite (wrap32_small (wrap32 x)) by (apply wrap32_range).
  rewrite (wrap32_small (Z.land (wrap32 x) (wrap32 y)))
    by (apply land_range_32; apply wrap32_range).
  apply lor_land_absorb; destruct (wrap32_range x); lia.
Qed.

(** * XOR Properties *)

(** x ^ 0 = x *)
Theorem xor32_identity : forall x,
  i32_xor x 0 = wrap32 x.
Proof.
  intros. unfold i32_xor.
  rewrite Z.lxor_0_r. reflexivity.
Qed.

(** x ^ x ^ x = x *)
Theorem xor32_triple : forall x,
  i32_xor (i32_xor x x) x = wrap32 x.
Proof.
  intros.
  rewrite xor32_self_cancels.
  (* i32_xor 0 x = wrap32 x *)
  unfold i32_xor. rewrite wrap32_0. rewrite Z.lxor_0_l. reflexivity.
Qed.

(** (x ^ y) ^ y = x *)
Theorem xor32_cancel_right : forall x y,
  i32_xor (i32_xor (wrap32 x) (wrap32 y)) (wrap32 y) = wrap32 x.
Proof.
  intros. unfold i32_xor.
  rewrite (wrap32_small (wrap32 x)) by (apply wrap32_range).
  rewrite (wrap32_small (wrap32 y)) by (apply wrap32_range).
  rewrite (wrap32_small (Z.lxor (wrap32 x) (wrap32 y)))
    by (apply lxor_range_32; apply wrap32_range).
  rewrite (wrap32_small (wrap32 y)) by (apply wrap32_range).
  rewrite Z.lxor_assoc.
  rewrite Z.lxor_nilpotent.
  rewrite Z.lxor_0_r. reflexivity.
Qed.

(** * Term-Level Proofs *)

(** These connect the Z-level proofs to the term simplifier *)

(** simplify correctly handles x & 0.
    simplify (TI32And t (TI32Const 0)):
    - simplify (TI32Const 0) = TI32Const 0
    - If simplify t = TI32Const a: constant fold gives TI32Const (i32_and a 0) = TI32Const 0
    - Otherwise: pattern (_, TI32Const 0 => TI32Const 0) applies *)
Theorem simplify_and_annihilates : forall t,
  eval_term (simplify (TI32And t (TI32Const 0))) = eval_term (TI32Const 0).
Proof.
  intros t.
  simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

(** simplify correctly handles 0 & x *)
Theorem simplify_and_annihilates_left : forall t,
  eval_term (simplify (TI32And (TI32Const 0) t)) = eval_term (TI32Const 0).
Proof.
  intros t.
  simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

(** simplify correctly handles x | 0 *)
Theorem simplify_or_identity : forall t,
  eval_term (simplify (TI32Or t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros t.
  simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

(** simplify correctly handles x ^ 0 *)
Theorem simplify_xor_identity : forall t,
  eval_term (simplify (TI32Xor t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros t.
  simpl.
  destruct (simplify t); simpl; reflexivity.
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
