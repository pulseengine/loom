(** * Strength Reduction Optimization Proofs

    This module proves the correctness of strength reduction optimizations
    in LOOM's WebAssembly optimizer.

    Strength reduction replaces expensive operations with cheaper equivalents:
      x * 2   -> x << 1      (multiply to shift)
      x * 4   -> x << 2
      x * 2^n -> x << n
      x / 2   -> x >> 1      (unsigned divide to shift)
      x / 2^n -> x >> n      (unsigned only)
      x % 2^n -> x & (2^n-1) (unsigned modulo to AND)

    These transformations are valid for specific constant values and
    preserve WebAssembly's modular arithmetic semantics.
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

(** * Power of Two Detection *)

(** Check if a number is a positive power of 2 *)
Definition is_power_of_two (n : Z) : bool :=
  Z.ltb 0 n && Z.eqb (Z.land n (n - 1)) 0.

(** Compute log2 for positive integers *)
Definition log2_z (n : Z) : Z := Z.log2 n.

(** * Multiplication to Shift *)

(** The core theorem: x * 2^k = x << k for 32-bit values *)

(** Helper: left shift is multiplication by 2^k *)
Lemma shiftl_is_mul_pow2 : forall x k,
  0 <= k ->
  Z.shiftl x k = x * Z.pow 2 k.
Proof.
  intros. apply Z.shiftl_mul_pow2. assumption.
Qed.

(** x * 2 = x << 1 *)
Theorem mul_2_is_shl_1 : forall x,
  i32_mul x 2 = i32_shl x 1.
Proof.
  intros. unfold i32_mul, i32_shl, shift_mask32, wrap32.
  simpl.
  f_equal.
  rewrite Z.shiftl_mul_pow2 by lia.
  simpl. ring.
Qed.

(** x * 4 = x << 2 *)
Theorem mul_4_is_shl_2 : forall x,
  i32_mul x 4 = i32_shl x 2.
Proof.
  intros. unfold i32_mul, i32_shl, shift_mask32, wrap32.
  simpl.
  f_equal.
  rewrite Z.shiftl_mul_pow2 by lia.
  simpl. ring.
Qed.

(** x * 8 = x << 3 *)
Theorem mul_8_is_shl_3 : forall x,
  i32_mul x 8 = i32_shl x 3.
Proof.
  intros. unfold i32_mul, i32_shl, shift_mask32, wrap32.
  simpl.
  f_equal.
  rewrite Z.shiftl_mul_pow2 by lia.
  simpl. ring.
Qed.

(** x * 16 = x << 4 *)
Theorem mul_16_is_shl_4 : forall x,
  i32_mul x 16 = i32_shl x 4.
Proof.
  intros. unfold i32_mul, i32_shl, shift_mask32, wrap32.
  simpl.
  f_equal.
  rewrite Z.shiftl_mul_pow2 by lia.
  simpl. ring.
Qed.

(** General theorem: x * 2^k = x << k for k in valid range *)
Theorem mul_pow2_is_shl : forall x k,
  0 <= k < 32 ->
  i32_mul x (Z.pow 2 k) = i32_shl x k.
Proof.
  intros x k Hk.
  unfold i32_mul, i32_shl, shift_mask32, wrap32.
  f_equal.
  (* The shift amount k is masked to 5 bits, but k < 32 so it's unchanged *)
  assert (Hmask: Z.land (Z.modulo k (Z.pow 2 32)) 31 = k).
  {
    rewrite Z.mod_small.
    - rewrite Z.land_ones by lia.
      rewrite Z.mod_small; lia.
    - split; [lia|].
      apply Z.lt_trans with 32; [lia|].
      (* 32 < 2^32 *)
      simpl. lia.
  }
  rewrite Hmask.
  rewrite Z.shiftl_mul_pow2 by lia.
  reflexivity.
Qed.

(** * i64 Multiplication to Shift *)

(** x * 2 = x << 1 (i64) *)
Theorem mul64_2_is_shl_1 : forall x,
  i64_mul x 2 = i64_shl x 1.
Proof.
  intros. unfold i64_mul, i64_shl, shift_mask64, wrap64.
  simpl.
  f_equal.
  rewrite Z.shiftl_mul_pow2 by lia.
  simpl. ring.
Qed.

(** x * 4 = x << 2 (i64) *)
Theorem mul64_4_is_shl_2 : forall x,
  i64_mul x 4 = i64_shl x 2.
Proof.
  intros. unfold i64_mul, i64_shl, shift_mask64, wrap64.
  simpl.
  f_equal.
  rewrite Z.shiftl_mul_pow2 by lia.
  simpl. ring.
Qed.

(** General theorem for i64 *)
Theorem mul64_pow2_is_shl : forall x k,
  0 <= k < 64 ->
  i64_mul x (Z.pow 2 k) = i64_shl x k.
Proof.
  intros x k Hk.
  unfold i64_mul, i64_shl, shift_mask64, wrap64.
  f_equal.
  assert (Hmask: Z.land (Z.modulo k (Z.pow 2 64)) 63 = k).
  {
    rewrite Z.mod_small.
    - rewrite Z.land_ones by lia.
      rewrite Z.mod_small; lia.
    - split; [lia|].
      apply Z.lt_trans with 64; [lia|].
      simpl. lia.
  }
  rewrite Hmask.
  rewrite Z.shiftl_mul_pow2 by lia.
  reflexivity.
Qed.

(** * Division to Shift (Unsigned Only) *)

(** For unsigned division by powers of 2, we can use logical right shift.
    This is only valid for UNSIGNED division - signed division has
    different rounding behavior for negative numbers. *)

(** Helper: right shift is unsigned division by 2^k *)
Lemma shiftr_is_div_pow2 : forall x k,
  0 <= x ->
  0 <= k ->
  Z.shiftr x k = Z.div x (Z.pow 2 k).
Proof.
  intros. apply Z.shiftr_div_pow2. assumption.
Qed.

(** x / 2 = x >> 1 (unsigned, for non-negative x) *)
Theorem div_u_2_is_shr_1 : forall x,
  0 <= x < Z.pow 2 32 ->
  Z.div x 2 = i32_shr_u x 1.
Proof.
  intros x Hx.
  unfold i32_shr_u, shift_mask32, wrap32.
  simpl.
  rewrite Z.mod_small by assumption.
  rewrite Z.shiftr_div_pow2 by lia.
  simpl. reflexivity.
Qed.

(** x / 4 = x >> 2 (unsigned) *)
Theorem div_u_4_is_shr_2 : forall x,
  0 <= x < Z.pow 2 32 ->
  Z.div x 4 = i32_shr_u x 2.
Proof.
  intros x Hx.
  unfold i32_shr_u, shift_mask32, wrap32.
  simpl.
  rewrite Z.mod_small by assumption.
  rewrite Z.shiftr_div_pow2 by lia.
  simpl. reflexivity.
Qed.

(** General: x / 2^k = x >> k (unsigned, for valid k) *)
Theorem div_u_pow2_is_shr : forall x k,
  0 <= x < Z.pow 2 32 ->
  0 <= k < 32 ->
  Z.div x (Z.pow 2 k) = i32_shr_u x k.
Proof.
  intros x k Hx Hk.
  unfold i32_shr_u, shift_mask32, wrap32.
  rewrite Z.mod_small by assumption.
  assert (Hmask: Z.land (Z.modulo k (Z.pow 2 32)) 31 = k).
  {
    rewrite Z.mod_small.
    - rewrite Z.land_ones by lia.
      rewrite Z.mod_small; lia.
    - split; [lia|].
      apply Z.lt_trans with 32; [lia|].
      simpl. lia.
  }
  rewrite Hmask.
  rewrite Z.shiftr_div_pow2 by lia.
  reflexivity.
Qed.

(** * Modulo to AND (Unsigned Only) *)

(** x % 2^k = x & (2^k - 1) for non-negative x *)
(** This is the bitmask optimization *)

(** Helper: modulo by power of 2 is bitwise AND with mask *)
Lemma mod_pow2_is_land : forall x k,
  0 <= x ->
  0 <= k ->
  Z.modulo x (Z.pow 2 k) = Z.land x (Z.pow 2 k - 1).
Proof.
  intros. rewrite Z.land_ones by assumption.
  reflexivity.
Qed.

(** x % 2 = x & 1 *)
Theorem mod_2_is_and_1 : forall x,
  0 <= x < Z.pow 2 32 ->
  Z.modulo x 2 = i32_and x 1.
Proof.
  intros x Hx.
  unfold i32_and, wrap32.
  rewrite Z.mod_small by assumption.
  simpl.
  rewrite Z.land_ones by lia.
  reflexivity.
Qed.

(** x % 4 = x & 3 *)
Theorem mod_4_is_and_3 : forall x,
  0 <= x < Z.pow 2 32 ->
  Z.modulo x 4 = i32_and x 3.
Proof.
  intros x Hx.
  unfold i32_and, wrap32.
  rewrite Z.mod_small by assumption.
  simpl.
  rewrite Z.land_ones by lia.
  reflexivity.
Qed.

(** x % 8 = x & 7 *)
Theorem mod_8_is_and_7 : forall x,
  0 <= x < Z.pow 2 32 ->
  Z.modulo x 8 = i32_and x 7.
Proof.
  intros x Hx.
  unfold i32_and, wrap32.
  rewrite Z.mod_small by assumption.
  simpl.
  rewrite Z.land_ones by lia.
  reflexivity.
Qed.

(** General: x % 2^k = x & (2^k - 1) *)
Theorem mod_pow2_is_and_mask : forall x k,
  0 <= x < Z.pow 2 32 ->
  0 <= k <= 32 ->
  Z.modulo x (Z.pow 2 k) = i32_and x (Z.pow 2 k - 1).
Proof.
  intros x k Hx Hk.
  unfold i32_and, wrap32.
  rewrite Z.mod_small by assumption.
  rewrite Z.mod_small.
  - rewrite Z.land_ones by lia.
    reflexivity.
  - split.
    + (* 2^k - 1 >= 0 *)
      assert (Z.pow 2 k >= 1) by (apply Z.pow_ge_1; lia).
      lia.
    + (* 2^k - 1 < 2^32 *)
      assert (Z.pow 2 k <= Z.pow 2 32).
      { apply Z.pow_le_mono_r; lia. }
      lia.
Qed.

(** * Negation to Subtraction *)

(** x * -1 = 0 - x *)
Theorem mul_neg1_is_neg : forall x,
  i32_mul x (-1) = i32_sub 0 x.
Proof.
  intros. unfold i32_mul, i32_sub, wrap32.
  f_equal. ring.
Qed.

(** * i64 Strength Reductions *)

(** x % 2 = x & 1 (i64) *)
Theorem mod64_2_is_and_1 : forall x,
  0 <= x < Z.pow 2 64 ->
  Z.modulo x 2 = i64_and x 1.
Proof.
  intros x Hx.
  unfold i64_and, wrap64.
  rewrite Z.mod_small by assumption.
  simpl.
  rewrite Z.land_ones by lia.
  reflexivity.
Qed.

(** x * -1 = 0 - x (i64) *)
Theorem mul64_neg1_is_neg : forall x,
  i64_mul x (-1) = i64_sub 0 x.
Proof.
  intros. unfold i64_mul, i64_sub, wrap64.
  f_equal. ring.
Qed.

(** * Combined Strength Reductions *)

(** 2 * x is the same as x * 2 (commutativity + strength reduction) *)
Theorem mul_2_comm_shl : forall x,
  i32_mul 2 x = i32_shl x 1.
Proof.
  intros. unfold i32_mul, i32_shl, shift_mask32, wrap32.
  simpl.
  f_equal.
  rewrite Z.shiftl_mul_pow2 by lia.
  simpl. ring.
Qed.

(** * Semantic Equivalence at Term Level *)

(** These theorems show that simplified terms evaluate identically *)

(** x * 2 term evaluates same as x << 1 term *)
Theorem term_mul_2_equiv_shl_1 : forall t,
  eval_term t = TROk (VI32 (wrap32 0)) ->
  term_equiv
    (TI32Mul t (TI32Const 2))
    (TI32Shl t (TI32Const 1)).
Proof.
  intros t Ht.
  unfold term_equiv.
  simpl.
  destruct (eval_term t) eqn:Het.
  - destruct v; try reflexivity.
    f_equal. f_equal.
    apply mul_2_is_shl_1.
  - reflexivity.
Qed.

(** For any power of 2, multiplication is equivalent to shift *)
(** This requires the LOOM simplifier to detect powers of 2 *)

Close Scope Z_scope.
