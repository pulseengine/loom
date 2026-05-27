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
From proofs Require Import semantics.WasmSemantics.
From proofs Require Import semantics.TermSemantics.
Import ListNotations.

Open Scope Z_scope.

(** * i32 Additive Identities *)

(** x + 0 = x *)
Theorem i32_add_identity_right : forall x,
  i32_add x 0 = wrap32 x.
Proof.
  intros. unfold i32_add. rewrite Z.add_0_r. reflexivity.
Qed.

(** 0 + x = x *)
Theorem i32_add_identity_left : forall x,
  i32_add 0 x = wrap32 x.
Proof.
  intros. unfold i32_add. rewrite Z.add_0_l. reflexivity.
Qed.

(** x - 0 = x *)
Theorem i32_sub_identity_right : forall x,
  i32_sub x 0 = wrap32 x.
Proof.
  intros. unfold i32_sub. rewrite Z.sub_0_r. reflexivity.
Qed.

(** 0 - x = -x (wrapped) *)
Theorem i32_sub_zero_left : forall x,
  i32_sub 0 x = wrap32 (- x).
Proof.
  intros. unfold i32_sub. rewrite Z.sub_0_l. reflexivity.
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
  intros. unfold i32_mul. rewrite Z.mul_1_r. reflexivity.
Qed.

(** 1 * x = x *)
Theorem i32_mul_identity_left : forall x,
  i32_mul 1 x = wrap32 x.
Proof.
  intros. unfold i32_mul. rewrite Z.mul_1_l. reflexivity.
Qed.

(** x * 0 = 0 *)
Theorem i32_mul_zero_right : forall x,
  i32_mul x 0 = 0.
Proof.
  intros. unfold i32_mul, wrap32. rewrite Z.mul_0_r. reflexivity.
Qed.

(** 0 * x = 0 *)
Theorem i32_mul_zero_left : forall x,
  i32_mul 0 x = 0.
Proof.
  intros. unfold i32_mul, wrap32. rewrite Z.mul_0_l. reflexivity.
Qed.

(** * i32 Bitwise Identities *)

(** Helper: wrap32(-1) = 2^32 - 1 *)
Lemma wrap32_minus1 : wrap32 (-1) = 2 ^ 32 - 1.
Proof. reflexivity. Qed.

(** Helper: wrap32 is idempotent *)
Lemma wrap32_idempotent : forall x, wrap32 (wrap32 x) = wrap32 x.
Proof.
  intros. unfold wrap32.
  rewrite Z.mod_mod; [reflexivity | discriminate].
Qed.

(** Helper: Z.land with all ones mask *)
Lemma land_all_ones_32 : forall x,
  0 <= x < 2 ^ 32 ->
  Z.land x (2 ^ 32 - 1) = x.
Proof.
  intros x Hx.
  replace (2 ^ 32 - 1) with (Z.ones 32) by reflexivity.
  rewrite Z.land_ones by lia.
  apply Z.mod_small. assumption.
Qed.

(** x & -1 = x (all bits set) *)
Theorem i32_and_all_ones_right : forall x,
  i32_and x (wrap32 (-1)) = wrap32 x.
Proof.
  intros. unfold i32_and.
  rewrite wrap32_idempotent.
  rewrite wrap32_minus1.
  apply land_all_ones_32.
  unfold wrap32. apply Z.mod_pos_bound. lia.
Qed.

(** -1 & x = x *)
Theorem i32_and_all_ones_left : forall x,
  i32_and (wrap32 (-1)) x = wrap32 x.
Proof.
  intros. unfold i32_and.
  rewrite wrap32_idempotent.
  rewrite wrap32_minus1.
  rewrite Z.land_comm.
  apply land_all_ones_32.
  unfold wrap32. apply Z.mod_pos_bound. lia.
Qed.

(** x & 0 = 0 *)
Theorem i32_and_zero_right : forall x,
  i32_and x 0 = 0.
Proof.
  intros. unfold i32_and, wrap32.
  rewrite Z.land_0_r. reflexivity.
Qed.

(** 0 & x = 0 *)
Theorem i32_and_zero_left : forall x,
  i32_and 0 x = 0.
Proof.
  intros. unfold i32_and, wrap32.
  rewrite Z.land_0_l. reflexivity.
Qed.

(** x | 0 = x *)
Theorem i32_or_identity_right : forall x,
  i32_or x 0 = wrap32 x.
Proof.
  intros. unfold i32_or.
  rewrite Z.lor_0_r. reflexivity.
Qed.

(** 0 | x = x *)
Theorem i32_or_identity_left : forall x,
  i32_or 0 x = wrap32 x.
Proof.
  intros. unfold i32_or.
  rewrite Z.lor_0_l. reflexivity.
Qed.

(** x ^ 0 = x *)
Theorem i32_xor_identity_right : forall x,
  i32_xor x 0 = wrap32 x.
Proof.
  intros. unfold i32_xor.
  rewrite Z.lxor_0_r. reflexivity.
Qed.

(** 0 ^ x = x *)
Theorem i32_xor_identity_left : forall x,
  i32_xor 0 x = wrap32 x.
Proof.
  intros. unfold i32_xor.
  rewrite Z.lxor_0_l. reflexivity.
Qed.

(** * i32 Shift Identities *)

(** x << 0 = x *)
Theorem i32_shl_zero : forall x,
  i32_shl x 0 = wrap32 x.
Proof.
  intros. unfold i32_shl, shift_mask32, wrap32.
  rewrite Z.land_0_l. rewrite Z.shiftl_0_r.
  rewrite Z.mod_mod.
  - reflexivity.
  - discriminate.
Qed.

(** x >> 0 = x (unsigned) *)
Theorem i32_shr_u_zero : forall x,
  i32_shr_u x 0 = wrap32 x.
Proof.
  intros. unfold i32_shr_u, shift_mask32, wrap32.
  rewrite Z.land_0_l. rewrite Z.shiftr_0_r. reflexivity.
Qed.

(** * i64 Additive Identities *)

(** x + 0 = x *)
Theorem i64_add_identity_right : forall x,
  i64_add x 0 = wrap64 x.
Proof.
  intros. unfold i64_add. rewrite Z.add_0_r. reflexivity.
Qed.

(** 0 + x = x *)
Theorem i64_add_identity_left : forall x,
  i64_add 0 x = wrap64 x.
Proof.
  intros. unfold i64_add. rewrite Z.add_0_l. reflexivity.
Qed.

(** x - 0 = x *)
Theorem i64_sub_identity_right : forall x,
  i64_sub x 0 = wrap64 x.
Proof.
  intros. unfold i64_sub. rewrite Z.sub_0_r. reflexivity.
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
  intros. unfold i64_mul. rewrite Z.mul_1_r. reflexivity.
Qed.

(** 1 * x = x *)
Theorem i64_mul_identity_left : forall x,
  i64_mul 1 x = wrap64 x.
Proof.
  intros. unfold i64_mul. rewrite Z.mul_1_l. reflexivity.
Qed.

(** x * 0 = 0 *)
Theorem i64_mul_zero_right : forall x,
  i64_mul x 0 = 0.
Proof.
  intros. unfold i64_mul, wrap64. rewrite Z.mul_0_r. reflexivity.
Qed.

(** 0 * x = 0 *)
Theorem i64_mul_zero_left : forall x,
  i64_mul 0 x = 0.
Proof.
  intros. unfold i64_mul, wrap64. rewrite Z.mul_0_l. reflexivity.
Qed.

(** * i64 Bitwise Identities *)

(** x & 0 = 0 *)
Theorem i64_and_zero_right : forall x,
  i64_and x 0 = 0.
Proof.
  intros. unfold i64_and, wrap64.
  rewrite Z.land_0_r. reflexivity.
Qed.

(** 0 & x = 0 *)
Theorem i64_and_zero_left : forall x,
  i64_and 0 x = 0.
Proof.
  intros. unfold i64_and, wrap64.
  rewrite Z.land_0_l. reflexivity.
Qed.

(** x | 0 = x *)
Theorem i64_or_identity_right : forall x,
  i64_or x 0 = wrap64 x.
Proof.
  intros. unfold i64_or.
  rewrite Z.lor_0_r. reflexivity.
Qed.

(** 0 | x = x *)
Theorem i64_or_identity_left : forall x,
  i64_or 0 x = wrap64 x.
Proof.
  intros. unfold i64_or.
  rewrite Z.lor_0_l. reflexivity.
Qed.

(** x ^ 0 = x *)
Theorem i64_xor_identity_right : forall x,
  i64_xor x 0 = wrap64 x.
Proof.
  intros. unfold i64_xor.
  rewrite Z.lxor_0_r. reflexivity.
Qed.

(** 0 ^ x = x *)
Theorem i64_xor_identity_left : forall x,
  i64_xor 0 x = wrap64 x.
Proof.
  intros. unfold i64_xor.
  rewrite Z.lxor_0_l. reflexivity.
Qed.

(** * i64 Shift Identities *)

(** Helper: 2^64 is not zero *)
Lemma pow2_64_nonzero : 2 ^ 64 <> 0.
Proof. discriminate. Qed.

(** x << 0 = x *)
Theorem i64_shl_zero : forall x,
  i64_shl x 0 = wrap64 x.
Proof.
  intros. unfold i64_shl, shift_mask64, wrap64.
  rewrite Z.land_0_l. rewrite Z.shiftl_0_r.
  rewrite Z.mod_mod.
  - reflexivity.
  - exact pow2_64_nonzero.
Qed.

(** x >> 0 = x (unsigned) *)
Theorem i64_shr_u_zero : forall x,
  i64_shr_u x 0 = wrap64 x.
Proof.
  intros. unfold i64_shr_u, shift_mask64, wrap64.
  rewrite Z.land_0_l. rewrite Z.shiftr_0_r. reflexivity.
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

(** * Additional Comparison Identities *)

(** x < x = 0 (signed) *)
Theorem i32_lt_s_irreflexive : forall x,
  i32_lt_s x x = 0.
Proof.
  intros. unfold i32_lt_s.
  rewrite Z.ltb_irrefl. reflexivity.
Qed.

(** x < x = 0 (unsigned) *)
Theorem i32_lt_u_irreflexive : forall x,
  i32_lt_u x x = 0.
Proof.
  intros. unfold i32_lt_u.
  rewrite Z.ltb_irrefl. reflexivity.
Qed.

(** x > x = 0 (signed) *)
Theorem i32_gt_s_irreflexive : forall x,
  i32_gt_s x x = 0.
Proof.
  intros. unfold i32_gt_s.
  rewrite Z.ltb_irrefl. reflexivity.
Qed.

(** x > x = 0 (unsigned) *)
Theorem i32_gt_u_irreflexive : forall x,
  i32_gt_u x x = 0.
Proof.
  intros. unfold i32_gt_u.
  rewrite Z.ltb_irrefl. reflexivity.
Qed.

(** x <= x = 1 (signed) *)
Theorem i32_le_s_reflexive : forall x,
  i32_le_s x x = 1.
Proof.
  intros. unfold i32_le_s.
  rewrite Z.leb_refl. reflexivity.
Qed.

(** x <= x = 1 (unsigned) *)
Theorem i32_le_u_reflexive : forall x,
  i32_le_u x x = 1.
Proof.
  intros. unfold i32_le_u.
  rewrite Z.leb_refl. reflexivity.
Qed.

(** x >= x = 1 (signed) *)
Theorem i32_ge_s_reflexive : forall x,
  i32_ge_s x x = 1.
Proof.
  intros. unfold i32_ge_s.
  rewrite Z.leb_refl. reflexivity.
Qed.

(** x >= x = 1 (unsigned) *)
Theorem i32_ge_u_reflexive : forall x,
  i32_ge_u x x = 1.
Proof.
  intros. unfold i32_ge_u.
  rewrite Z.leb_refl. reflexivity.
Qed.

(** * Well-Formedness Predicate *)

(** A term is well-formed when all i32 constants are in the wrapped range
    [0, 2^32) and all i64 constants are in [0, 2^64).  This is always true
    for terms derived from actual WebAssembly programs, which use fixed-width
    integers.  The predicate is needed because our Coq model uses unbounded
    Z for constants, and the identity optimizations rely on wrap being a
    no-op for values already in range. *)

Fixpoint wf_term (t : Term) : Prop :=
  match t with
  | TI32Const z => 0 <= z < 2 ^ 32
  | TI64Const z => 0 <= z < 2 ^ 64
  | TI32Add l r | TI32Sub l r | TI32Mul l r
  | TI64Add l r | TI64Sub l r | TI64Mul l r
  | TI32And l r | TI32Or l r | TI32Xor l r
  | TI64And l r | TI64Or l r | TI64Xor l r
  | TI32Shl l r | TI32ShrS l r | TI32ShrU l r
  | TI64Shl l r | TI64ShrS l r | TI64ShrU l r
  | TI32Eq l r | TI32Ne l r
  | TI32LtS l r | TI32LtU l r | TI32GtS l r | TI32GtU l r
  | TI32LeS l r | TI32LeU l r | TI32GeS l r | TI32GeU l r
  | TI64Eq l r | TI64Ne l r
  | TI64LtS l r | TI64LtU l r | TI64GtS l r | TI64GtU l r
  | TI64LeS l r | TI64LeU l r | TI64GeS l r | TI64GeU l r
    => wf_term l /\ wf_term r
  | TI32Eqz t | TI64Eqz t | TDrop t => wf_term t
  | TNop => True
  end.

(** Key helper: wrap32 is identity on values already in range *)
Lemma wrap32_small : forall z, 0 <= z < 2 ^ 32 -> wrap32 z = z.
Proof.
  intros z Hz. unfold wrap32. apply Z.mod_small. assumption.
Qed.

(** Key helper: wrap64 is identity on values already in range *)
Lemma wrap64_small : forall z, 0 <= z < 2 ^ 64 -> wrap64 z = z.
Proof.
  intros z Hz. unfold wrap64. apply Z.mod_small. assumption.
Qed.

(** ** Bitwise bounds helpers *)

(** Distribution of Z.land over Z.land (bitwise AND distributes over AND):
    (a & c) & (b & c) = (a & b) & c *)
Lemma Z_land_land_distrib : forall a b c,
  Z.land (Z.land a c) (Z.land b c) = Z.land (Z.land a b) c.
Proof.
  intros. apply Z.bits_inj'. intros n Hn.
  repeat rewrite Z.land_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit c n); reflexivity.
Qed.

(** Distribution of Z.lor over Z.land on the mask:
    (a & c) | (b & c) = (a | b) & c *)
Lemma Z_lor_land_distrib_r : forall a b c,
  Z.lor (Z.land a c) (Z.land b c) = Z.land (Z.lor a b) c.
Proof.
  intros. apply Z.bits_inj'. intros n Hn.
  rewrite Z.lor_spec. repeat rewrite Z.land_spec. rewrite Z.lor_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit c n); reflexivity.
Qed.

(** Distribution of Z.lxor over Z.land on the mask:
    (a & c) ^ (b & c) = (a ^ b) & c *)
Lemma Z_lxor_land_distrib_r : forall a b c,
  Z.lxor (Z.land a c) (Z.land b c) = Z.land (Z.lxor a b) c.
Proof.
  intros. apply Z.bits_inj'. intros n Hn.
  rewrite Z.lxor_spec. repeat rewrite Z.land_spec. rewrite Z.lxor_spec.
  destruct (Z.testbit a n), (Z.testbit b n), (Z.testbit c n); reflexivity.
Qed.

(** Z.land of two n-bit values stays in [0, 2^n) *)
Lemma Z_land_mod_bound : forall a b n,
  0 <= n -> 0 <= Z.land (a mod 2^n) (b mod 2^n) < 2^n.
Proof.
  intros a b n Hn. split.
  - apply Z.land_nonneg. left. apply Z.mod_pos_bound. lia.
  - rewrite <- (Z.land_ones a n) by assumption.
    rewrite <- (Z.land_ones b n) by assumption.
    rewrite Z_land_land_distrib.
    rewrite Z.land_ones by assumption.
    apply Z.mod_pos_bound. lia.
Qed.

(** Z.lor of two n-bit values stays in [0, 2^n) *)
Lemma Z_lor_mod_bound : forall a b n,
  0 <= n -> 0 <= Z.lor (a mod 2^n) (b mod 2^n) < 2^n.
Proof.
  intros a b n Hn. split.
  - apply Z.lor_nonneg. split; apply Z.mod_pos_bound; lia.
  - rewrite <- (Z.land_ones a n) by assumption.
    rewrite <- (Z.land_ones b n) by assumption.
    rewrite Z_lor_land_distrib_r.
    rewrite Z.land_ones by assumption.
    apply Z.mod_pos_bound. lia.
Qed.

(** Z.lxor of two n-bit values stays in [0, 2^n) *)
Lemma Z_lxor_mod_bound : forall a b n,
  0 <= n -> 0 <= Z.lxor (a mod 2^n) (b mod 2^n) < 2^n.
Proof.
  intros a b n Hn. split.
  - apply Z.lxor_nonneg. split; apply Z.mod_pos_bound; lia.
  - rewrite <- (Z.land_ones a n) by assumption.
    rewrite <- (Z.land_ones b n) by assumption.
    rewrite Z_lxor_land_distrib_r.
    rewrite Z.land_ones by assumption.
    apply Z.mod_pos_bound. lia.
Qed.

(** i32 bitwise operations produce values in [0, 2^32) *)
Lemma i32_and_bound : forall a b, 0 <= i32_and a b < 2 ^ 32.
Proof. intros. unfold i32_and. apply Z_land_mod_bound. lia. Qed.

Lemma i32_or_bound : forall a b, 0 <= i32_or a b < 2 ^ 32.
Proof. intros. unfold i32_or. apply Z_lor_mod_bound. lia. Qed.

Lemma i32_xor_bound : forall a b, 0 <= i32_xor a b < 2 ^ 32.
Proof. intros. unfold i32_xor. apply Z_lxor_mod_bound. lia. Qed.

(** i64 bitwise operations produce values in [0, 2^64) *)
Lemma i64_and_bound : forall a b, 0 <= i64_and a b < 2 ^ 64.
Proof. intros. unfold i64_and. apply Z_land_mod_bound. lia. Qed.

Lemma i64_or_bound : forall a b, 0 <= i64_or a b < 2 ^ 64.
Proof. intros. unfold i64_or. apply Z_lor_mod_bound. lia. Qed.

Lemma i64_xor_bound : forall a b, 0 <= i64_xor a b < 2 ^ 64.
Proof. intros. unfold i64_xor. apply Z_lxor_mod_bound. lia. Qed.

(** ** Evaluation produces wrapped values *)

(** Key property: eval_term of a well-formed term always produces values
    in the appropriate wrapped range.  This is because:
    - Constants are in range by wf_term
    - All arithmetic/bitwise/shift operations produce wrapped values
    - All comparison operations return 0 or 1 (in range) *)
Lemma eval_term_i32_bound : forall t z,
  wf_term t ->
  eval_term t = TROk (VI32 z) ->
  0 <= z < 2 ^ 32.
Proof.
  induction t; simpl; intros z' Hwf Heq; try discriminate;
    (* TI32Const: z is in range by wf_term *)
    try (injection Heq; intros; subst; exact Hwf);
    (* Binary i32 ops with wrapping: the result is wrap32 (...) which is in range *)
    try (destruct Hwf as [Hl Hr];
         destruct (eval_term t1) as [[?|?|?|?]|]; try discriminate;
         destruct (eval_term t2) as [[?|?|?|?]|]; try discriminate;
         injection Heq; intros; subst;
         first [ unfold i32_add, wrap32; apply Z.mod_pos_bound; lia
               | unfold i32_sub, wrap32; apply Z.mod_pos_bound; lia
               | unfold i32_mul, wrap32; apply Z.mod_pos_bound; lia
               | unfold i32_shl, wrap32; apply Z.mod_pos_bound; lia
               | unfold i32_shr_s, wrap32; apply Z.mod_pos_bound; lia
               | apply i32_and_bound
               | apply i32_or_bound
               | apply i32_xor_bound
               | unfold i32_eq; destruct (Z.eqb _ _); lia
               | unfold i32_ne; destruct (Z.eqb _ _); lia
               | unfold i32_lt_s; destruct (Z.ltb _ _); lia
               | unfold i32_lt_u; destruct (Z.ltb _ _); lia
               | unfold i32_gt_s; destruct (Z.ltb _ _); lia
               | unfold i32_gt_u; destruct (Z.ltb _ _); lia
               | unfold i32_le_s; destruct (Z.leb _ _); lia
               | unfold i32_le_u; destruct (Z.leb _ _); lia
               | unfold i32_ge_s; destruct (Z.leb _ _); lia
               | unfold i32_ge_u; destruct (Z.leb _ _); lia
               | unfold i64_eq; destruct (Z.eqb _ _); lia
               | unfold i64_ne; destruct (Z.eqb _ _); lia
               | unfold i64_lt_s; destruct (Z.ltb _ _); lia
               | unfold i64_lt_u; destruct (Z.ltb _ _); lia
               | unfold i64_gt_s; destruct (Z.ltb _ _); lia
               | unfold i64_gt_u; destruct (Z.ltb _ _); lia
               | unfold i64_le_s; destruct (Z.leb _ _); lia
               | unfold i64_le_u; destruct (Z.leb _ _); lia
               | unfold i64_ge_s; destruct (Z.leb _ _); lia
               | unfold i64_ge_u; destruct (Z.leb _ _); lia
               ]).
  (* TI32ShrU: result is Z.shiftr which needs special treatment *)
  - destruct Hwf as [Hl Hr].
    destruct (eval_term t1) as [[?|?|?|?]|]; try discriminate.
    destruct (eval_term t2) as [[?|?|?|?]|]; try discriminate.
    injection Heq; intros; subst.
    unfold i32_shr_u, shift_mask32.
    assert (Hwr : 0 <= wrap32 z < 2 ^ 32) by (unfold wrap32; apply Z.mod_pos_bound; lia).
    assert (Hsh : 0 <= Z.land (wrap32 z0) 31) by (apply Z.land_nonneg; left; lia).
    rewrite Z.shiftr_div_pow2 by assumption.
    assert (Hpow : 0 < 2 ^ Z.land (wrap32 z0) 31) by (apply Z.pow_pos_nonneg; lia).
    split.
    + apply Z.div_pos; lia.
    + apply Z.le_lt_trans with (wrap32 z); [| lia].
      apply Z.div_le_upper_bound; [lia |].
      assert (1 <= 2 ^ Z.land (wrap32 z0) 31) by (apply Z.pow_le_mono_r; lia).
      nia.
  (* TI32Eqz: unary, returns 0 or 1 *)
  - destruct (eval_term t) as [[?|?|?|?]|]; try discriminate.
    injection Heq; intros; subst.
    unfold i32_eqz. destruct (Z.eqb _ _); lia.
  (* TI64Eqz: returns VI32 *)
  - destruct (eval_term t) as [[?|?|?|?]|]; try discriminate.
    injection Heq; intros; subst.
    unfold i64_eqz. destruct (Z.eqb _ _); lia.
Qed.

(** Similar for i64 values *)
Lemma eval_term_i64_bound : forall t z,
  wf_term t ->
  eval_term t = TROk (VI64 z) ->
  0 <= z < 2 ^ 64.
Proof.
  induction t; simpl; intros z' Hwf Heq; try discriminate;
    try (injection Heq; intros; subst; exact Hwf);
    try (destruct Hwf as [Hl Hr];
         destruct (eval_term t1) as [[?|?|?|?]|]; try discriminate;
         destruct (eval_term t2) as [[?|?|?|?]|]; try discriminate;
         injection Heq; intros; subst;
         first [ unfold i64_add, wrap64; apply Z.mod_pos_bound; lia
               | unfold i64_sub, wrap64; apply Z.mod_pos_bound; lia
               | unfold i64_mul, wrap64; apply Z.mod_pos_bound; lia
               | unfold i64_shl, wrap64; apply Z.mod_pos_bound; lia
               | unfold i64_shr_s, wrap64; apply Z.mod_pos_bound; lia
               | apply i64_and_bound
               | apply i64_or_bound
               | apply i64_xor_bound
               ]).
  (* TI64ShrU *)
  - destruct Hwf as [Hl Hr].
    destruct (eval_term t1) as [[?|?|?|?]|]; try discriminate.
    destruct (eval_term t2) as [[?|?|?|?]|]; try discriminate.
    injection Heq; intros; subst.
    unfold i64_shr_u, shift_mask64.
    assert (Hwr : 0 <= wrap64 z < 2 ^ 64) by (unfold wrap64; apply Z.mod_pos_bound; lia).
    assert (Hsh : 0 <= Z.land (wrap64 z0) 63) by (apply Z.land_nonneg; left; lia).
    rewrite Z.shiftr_div_pow2 by assumption.
    assert (Hpow : 0 < 2 ^ Z.land (wrap64 z0) 63) by (apply Z.pow_pos_nonneg; lia).
    split.
    + apply Z.div_pos; lia.
    + apply Z.le_lt_trans with (wrap64 z); [| lia].
      apply Z.div_le_upper_bound; [lia |].
      assert (1 <= 2 ^ Z.land (wrap64 z0) 63) by (apply Z.pow_le_mono_r; lia).
      nia.
Qed.

(** ** Bridge: simplify produces in-range constants for well-formed terms *)

(** If simplify t = TI32Const z and wf_term t, then z is in range.
    Proof strategy: simplify_preserves_semantics tells us that
    eval_term t = eval_term (simplify t) = TROk (VI32 z),
    and eval_term_i32_bound gives 0 <= z < 2^32. *)
Lemma simplify_i32const_wf : forall t z,
  wf_term t -> simplify t = TI32Const z -> 0 <= z < 2 ^ 32.
Proof.
  intros t z Hwf Hs.
  apply (eval_term_i32_bound t z Hwf).
  rewrite (simplify_preserves_semantics t).
  rewrite Hs. reflexivity.
Qed.

(** Similar for i64 *)
Lemma simplify_i64const_wf : forall t z,
  wf_term t -> simplify t = TI64Const z -> 0 <= z < 2 ^ 64.
Proof.
  intros t z Hwf Hs.
  apply (eval_term_i64_bound t z Hwf).
  rewrite (simplify_preserves_semantics t).
  rewrite Hs. reflexivity.
Qed.

(** * Term-Level Identity Proofs *)

(** These prove that simplify preserves semantics for identity rules.
    The theorems require that input terms are well-formed (all integer
    constants are in the appropriate wrapped range), which is always true
    for terms derived from actual WebAssembly programs.

    The wf_term precondition is necessary because the simplifier's
    constant folding applies wrapping (e.g., i32_add z 0 = wrap32 z),
    and wrap32 z = z only when z is already in [0, 2^32). *)

(** simplify (x + 0) is semantically equivalent to simplify x *)
Theorem simplify_add_zero_right_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Add t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  (* TI32Const z case: constant folding gives i32_add z 0 = wrap32 z,
     need wrap32 z = z, which holds because z is in range *)
  f_equal. f_equal. rewrite i32_add_identity_right.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (0 + x) is semantically equivalent to simplify x *)
Theorem simplify_add_zero_left_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Add (TI32Const 0) t)) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_add_identity_left.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (x - 0) is semantically equivalent to simplify x *)
Theorem simplify_sub_zero_right_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Sub t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_sub_identity_right.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (x * 1) is semantically equivalent to simplify x *)
Theorem simplify_mul_one_right_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Mul t (TI32Const 1))) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_mul_identity_right.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (1 * x) is semantically equivalent to simplify x *)
Theorem simplify_mul_one_left_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Mul (TI32Const 1) t)) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_mul_identity_left.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (x * 0) evaluates to 0.
    This is provable without wf_term: in all cases the simplifier produces
    TI32Const 0 (or TI32Const (i32_mul a 0) = TI32Const 0). *)
Theorem simplify_mul_zero_right_equiv : forall t,
  eval_term (simplify (TI32Mul t (TI32Const 0))) = TROk (VI32 0).
Proof.
  intros t. simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

(** simplify (0 * x) evaluates to 0 *)
Theorem simplify_mul_zero_left_equiv : forall t,
  eval_term (simplify (TI32Mul (TI32Const 0) t)) = TROk (VI32 0).
Proof.
  intros t. simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

(** simplify (x | 0) is semantically equivalent to simplify x *)
Theorem simplify_or_zero_right_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Or t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_or_identity_right.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (0 | x) is semantically equivalent to simplify x *)
Theorem simplify_or_zero_left_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Or (TI32Const 0) t)) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_or_identity_left.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (x ^ 0) is semantically equivalent to simplify x *)
Theorem simplify_xor_zero_right_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Xor t (TI32Const 0))) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_xor_identity_right.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (0 ^ x) is semantically equivalent to simplify x *)
Theorem simplify_xor_zero_left_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI32Xor (TI32Const 0) t)) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i32_xor_identity_left.
  apply wrap32_small. eapply simplify_i32const_wf; eauto.
Qed.

(** simplify (x & 0) evaluates to 0.
    This is provable without wf_term: in all cases the simplifier produces
    TI32Const 0. *)
Theorem simplify_and_zero_right_equiv : forall t,
  eval_term (simplify (TI32And t (TI32Const 0))) = TROk (VI32 0).
Proof.
  intros t. simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

(** simplify (0 & x) evaluates to 0 *)
Theorem simplify_and_zero_left_equiv : forall t,
  eval_term (simplify (TI32And (TI32Const 0) t)) = TROk (VI32 0).
Proof.
  intros t. simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

(** * i64 Term-Level Identity Proofs *)

(** simplify (x + 0) is semantically equivalent to simplify x (i64) *)
Theorem simplify_i64_add_zero_right_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI64Add t (TI64Const 0))) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i64_add_identity_right.
  apply wrap64_small. eapply simplify_i64const_wf; eauto.
Qed.

(** simplify (0 + x) is semantically equivalent to simplify x (i64) *)
Theorem simplify_i64_add_zero_left_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI64Add (TI64Const 0) t)) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i64_add_identity_left.
  apply wrap64_small. eapply simplify_i64const_wf; eauto.
Qed.

(** simplify (x * 1) is semantically equivalent to simplify x (i64) *)
Theorem simplify_i64_mul_one_right_equiv : forall t,
  wf_term t ->
  eval_term (simplify (TI64Mul t (TI64Const 1))) = eval_term (simplify t).
Proof.
  intros t Hwf. simpl.
  destruct (simplify t) eqn:Hs; simpl; try reflexivity.
  f_equal. f_equal. rewrite i64_mul_identity_right.
  apply wrap64_small. eapply simplify_i64const_wf; eauto.
Qed.

(** simplify (0 * x) evaluates to 0 (i64).
    Provable without wf_term: simplifier always produces TI64Const 0. *)
Theorem simplify_i64_mul_zero_left_equiv : forall t,
  eval_term (simplify (TI64Mul (TI64Const 0) t)) = TROk (VI64 0).
Proof.
  intros t. simpl.
  destruct (simplify t); simpl; reflexivity.
Qed.

Close Scope Z_scope.
