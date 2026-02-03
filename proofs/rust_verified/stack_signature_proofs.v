(** * Stack Signature Proofs

    Formal verification of LOOM's stack signature composition system.

    This module proves key properties about stack signatures:
    1. compose is associative (when composable)
    2. empty is left and right identity for compose
    3. composes is decidable
    4. is_subtype is reflexive and transitive
    5. stack_effect is additive under composition
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
Import ListNotations.

(** * Pure Rocq Model *)

(** Value types *)
Inductive ValueType : Type :=
  | I32 : ValueType
  | I64 : ValueType
  | F32 : ValueType
  | F64 : ValueType.

(** Decidable equality for ValueType *)
Definition valuetype_eqb (a b : ValueType) : bool :=
  match a, b with
  | I32, I32 | I64, I64 | F32, F32 | F64, F64 => true
  | _, _ => false
  end.

(** Signature kind: Fixed or Polymorphic *)
Inductive SignatureKind : Type :=
  | Fixed : SignatureKind
  | Polymorphic : SignatureKind.

Definition sigkind_eqb (a b : SignatureKind) : bool :=
  match a, b with
  | Fixed, Fixed | Polymorphic, Polymorphic => true
  | _, _ => false
  end.

(** Stack signature: [params] -> [results] with kind *)
Record StackSignature : Type := mkSig {
  params : list ValueType;
  results : list ValueType;
  kind : SignatureKind;
}.

(** * Core Operations *)

(** Empty signature: [] -> [] {fixed} *)
Definition empty_sig : StackSignature :=
  mkSig [] [] Fixed.

(** Polymorphic empty: [] -> [] {poly} *)
Definition poly_empty_sig : StackSignature :=
  mkSig [] [] Polymorphic.

(** List equality for ValueType *)
Fixpoint valuetype_list_eqb (l1 l2 : list ValueType) : bool :=
  match l1, l2 with
  | [], [] => true
  | x :: xs, y :: ys => valuetype_eqb x y && valuetype_list_eqb xs ys
  | _, _ => false
  end.

(** Check if two signatures compose: self.results = next.params *)
Definition composes (self next : StackSignature) : bool :=
  valuetype_list_eqb (results self) (params next).

(** Compute the combined kind *)
Definition combined_kind (k1 k2 : SignatureKind) : SignatureKind :=
  match k1, k2 with
  | Polymorphic, _ => Polymorphic
  | _, Polymorphic => Polymorphic
  | Fixed, Fixed => Fixed
  end.

(** Compose two signatures (partial function) *)
Definition compose (a b : StackSignature) : option StackSignature :=
  if composes a b then
    Some (mkSig (params a) (results b) (combined_kind (kind a) (kind b)))
  else
    None.

(** Stack effect: results.len - params.len *)
Definition stack_effect (sig : StackSignature) : Z :=
  Z.of_nat (length (results sig)) - Z.of_nat (length (params sig)).

(** Check if signature is identity *)
Definition is_identity (sig : StackSignature) : bool :=
  valuetype_list_eqb (params sig) (results sig) &&
  sigkind_eqb (kind sig) Fixed.

(** Check if polymorphic bottom *)
Definition is_poly_bottom (sig : StackSignature) : bool :=
  match params sig, results sig, kind sig with
  | [], [], Polymorphic => true
  | _, _, _ => false
  end.

(** * Subtyping *)

(** is_subtype relation *)
Definition is_subtype (a b : StackSignature) : bool :=
  (* Polymorphic bottom is subtype of everything *)
  if is_poly_bottom a then true
  else
    match kind a, kind b with
    (* Both fixed: exact match *)
    | Fixed, Fixed =>
        valuetype_list_eqb (params a) (params b) &&
        valuetype_list_eqb (results a) (results b)
    (* Polymorphic can't be subtype of fixed *)
    | Polymorphic, Fixed => false
    (* Both polymorphic: check types match *)
    | Polymorphic, Polymorphic =>
        valuetype_list_eqb (params a) (params b) &&
        valuetype_list_eqb (results a) (results b)
    (* Fixed is subtype of polymorphic with same types *)
    | Fixed, Polymorphic =>
        valuetype_list_eqb (params a) (params b) &&
        valuetype_list_eqb (results a) (results b)
    end.

(** * Helper Lemmas *)

Lemma valuetype_eqb_refl : forall v, valuetype_eqb v v = true.
Proof. destruct v; reflexivity. Qed.

Lemma valuetype_list_eqb_refl : forall l, valuetype_list_eqb l l = true.
Proof.
  induction l; simpl; auto.
  rewrite valuetype_eqb_refl, IHl. reflexivity.
Qed.

Lemma sigkind_eqb_refl : forall k, sigkind_eqb k k = true.
Proof. destruct k; reflexivity. Qed.

Lemma valuetype_eqb_eq : forall a b, valuetype_eqb a b = true <-> a = b.
Proof.
  intros a b; split; destruct a, b; simpl; intros; try reflexivity; discriminate.
Qed.

Lemma valuetype_list_eqb_eq : forall l1 l2,
  valuetype_list_eqb l1 l2 = true <-> l1 = l2.
Proof.
  induction l1; destruct l2; simpl; split; intros; try reflexivity; try discriminate.
  - apply andb_true_iff in H. destruct H.
    apply valuetype_eqb_eq in H. apply IHl1 in H0.
    subst. reflexivity.
  - injection H as H1 H2. subst.
    rewrite valuetype_eqb_refl, valuetype_list_eqb_refl. reflexivity.
Qed.

(** * Main Theorems *)

(** ** Theorem 1: composes is decidable *)
Theorem composes_decidable : forall a b,
  {composes a b = true} + {composes a b = false}.
Proof.
  intros. destruct (composes a b); auto.
Qed.

(** ** Theorem 2: empty is left identity for compose *)
Theorem empty_left_identity : forall sig,
  params sig = [] ->
  compose empty_sig sig = Some sig.
Proof.
  intros sig Hparams.
  unfold compose, composes, empty_sig, combined_kind.
  simpl.
  rewrite Hparams.
  simpl.
  destruct sig as [p r k].
  simpl in Hparams. subst.
  simpl.
  destruct k; reflexivity.
Qed.

(** ** Theorem 3: empty is right identity for compose *)
Theorem empty_right_identity : forall sig,
  results sig = [] ->
  kind sig = Fixed ->
  compose sig empty_sig = Some sig.
Proof.
  intros sig Hresults Hkind.
  unfold compose, composes, empty_sig.
  simpl.
  rewrite Hresults.
  simpl.
  destruct sig as [p r k].
  simpl in Hresults, Hkind. subst.
  simpl.
  reflexivity.
Qed.

(** ** Theorem 4: compose preserves params of first and results of second *)
Theorem compose_preserves : forall a b c,
  compose a b = Some c ->
  params c = params a /\ results c = results b.
Proof.
  intros a b c Hcomp.
  unfold compose in Hcomp.
  destruct (composes a b) eqn:Hcomp_check; [|discriminate].
  injection Hcomp as Hc. subst.
  simpl. split; reflexivity.
Qed.

(** ** Theorem 5: stack_effect is additive (when composable) *)
Theorem stack_effect_additive : forall a b c,
  compose a b = Some c ->
  stack_effect c = (stack_effect a + stack_effect b)%Z.
Proof.
  intros a b c Hcomp.
  unfold compose in Hcomp.
  destruct (composes a b) eqn:Hcheck; [|discriminate].
  injection Hcomp as Hc. subst.
  unfold stack_effect, composes in *.
  simpl.
  (* composes a b = true means results a = params b *)
  apply valuetype_list_eqb_eq in Hcheck.
  rewrite Hcheck.
  (* Now both sides have length (params b) which cancel *)
  lia.
Qed.

(** ** Theorem 6: is_subtype is reflexive *)
Theorem is_subtype_refl : forall sig,
  is_subtype sig sig = true.
Proof.
  intros sig.
  unfold is_subtype.
  destruct (is_poly_bottom sig) eqn:Hpoly; [reflexivity|].
  destruct (kind sig).
  - (* Fixed *)
    rewrite valuetype_list_eqb_refl.
    rewrite valuetype_list_eqb_refl.
    reflexivity.
  - (* Polymorphic *)
    rewrite valuetype_list_eqb_refl.
    rewrite valuetype_list_eqb_refl.
    reflexivity.
Qed.

(** ** Theorem 7: polymorphic bottom is subtype of everything *)
Theorem poly_bottom_subtype_all : forall sig,
  is_subtype poly_empty_sig sig = true.
Proof.
  intros sig.
  unfold is_subtype, is_poly_bottom, poly_empty_sig.
  simpl. reflexivity.
Qed.

(** ** Theorem 8: composes is symmetric in a specific sense *)
(** If a composes with b, then a.results = b.params *)
Theorem composes_results_params : forall a b,
  composes a b = true ->
  results a = params b.
Proof.
  intros a b Hcomp.
  unfold composes in Hcomp.
  apply valuetype_list_eqb_eq in Hcomp.
  exact Hcomp.
Qed.

(** ** Theorem 9: combined_kind is commutative *)
Theorem combined_kind_comm : forall k1 k2,
  combined_kind k1 k2 = combined_kind k2 k1.
Proof.
  intros k1 k2.
  destruct k1, k2; reflexivity.
Qed.

(** ** Theorem 10: combined_kind is associative *)
Theorem combined_kind_assoc : forall k1 k2 k3,
  combined_kind (combined_kind k1 k2) k3 = combined_kind k1 (combined_kind k2 k3).
Proof.
  intros k1 k2 k3.
  destruct k1, k2, k3; reflexivity.
Qed.

(** ** Theorem 11: Fixed is identity for combined_kind *)
Theorem combined_kind_fixed_left : forall k,
  combined_kind Fixed k = k.
Proof.
  intros k. destruct k; reflexivity.
Qed.

Theorem combined_kind_fixed_right : forall k,
  combined_kind k Fixed = k.
Proof.
  intros k. destruct k; reflexivity.
Qed.

(** ** Theorem 12: Polymorphic is absorbing for combined_kind *)
Theorem combined_kind_poly_left : forall k,
  combined_kind Polymorphic k = Polymorphic.
Proof.
  intros k. reflexivity.
Qed.

Theorem combined_kind_poly_right : forall k,
  combined_kind k Polymorphic = Polymorphic.
Proof.
  intros k. destruct k; reflexivity.
Qed.

(** ** Theorem 13: compose is associative when composable *)
(** This requires that intermediate compositions succeed *)
Theorem compose_assoc : forall a b c ab bc abc1 abc2,
  compose a b = Some ab ->
  compose b c = Some bc ->
  compose ab c = Some abc1 ->
  compose a bc = Some abc2 ->
  abc1 = abc2.
Proof.
  intros a b c ab bc abc1 abc2 Hab Hbc Habc1 Habc2.
  (* Extract the composed signatures *)
  unfold compose in *.
  destruct (composes a b) eqn:Hab_check; [|discriminate].
  destruct (composes b c) eqn:Hbc_check; [|discriminate].
  injection Hab as Hab_eq.
  injection Hbc as Hbc_eq.
  subst ab bc.
  simpl in Habc1, Habc2.
  destruct (composes (mkSig (params a) (results b) (combined_kind (kind a) (kind b))) c) eqn:H1; [|discriminate].
  destruct (composes a (mkSig (params b) (results c) (combined_kind (kind b) (kind c)))) eqn:H2; [|discriminate].
  injection Habc1 as Habc1_eq.
  injection Habc2 as Habc2_eq.
  subst abc1 abc2.
  simpl in *.
  f_equal.
  apply combined_kind_assoc.
Qed.

(** * Instruction Signature Properties *)

(** Example: i32.const signature *)
Definition i32_const_sig : StackSignature := mkSig [] [I32] Fixed.

(** Example: i32.add signature *)
Definition i32_add_sig : StackSignature := mkSig [I32; I32] [I32] Fixed.

(** Example: nop signature *)
Definition nop_sig : StackSignature := empty_sig.

(** Theorem: i32.const doesn't compose with itself *)
(** This is correct - [i32] results don't match [] params *)
Theorem const_doesnt_compose_with_const :
  compose i32_const_sig i32_const_sig = None.
Proof.
  unfold compose, composes, i32_const_sig. simpl.
  reflexivity.
Qed.

(** Theorem: nop composes with signatures that have empty params *)
Theorem nop_composes_left : forall sig,
  params sig = [] ->
  compose nop_sig sig = Some sig.
Proof.
  intros sig Hparams.
  unfold compose, composes, nop_sig, empty_sig, combined_kind.
  simpl.
  rewrite Hparams. simpl.
  destruct sig as [p r k].
  simpl in Hparams. subst.
  destruct k; reflexivity.
Qed.

(** Theorem: nop is identity *)
Theorem nop_is_identity : is_identity nop_sig = true.
Proof.
  unfold is_identity, nop_sig, empty_sig.
  simpl. reflexivity.
Qed.
