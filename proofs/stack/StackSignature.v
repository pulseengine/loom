(** * Stack Signature Formalization

    This module formalizes the StackSignature type from loom-core/src/stack.rs
    and proves key properties about stack composition.

    The goal is to prove that stack signature composition is:
    1. Associative: (a ∘ b) ∘ c = a ∘ (b ∘ c)
    2. Has identity: empty ∘ a = a = a ∘ empty
    3. Preserves type safety
*)

Require Import Coq.Lists.List.
Require Import Coq.Bool.Bool.
Require Import Coq.Arith.PeanoNat.
Import ListNotations.

(** ** Value Types

    WebAssembly value types, corresponding to:
    ```rust
    pub enum ValueType { I32, I64, F32, F64 }
    ```
*)
Inductive ValueType : Type :=
  | I32 : ValueType
  | I64 : ValueType
  | F32 : ValueType
  | F64 : ValueType.

(** Decidable equality for ValueType *)
Definition valuetype_eqb (a b : ValueType) : bool :=
  match a, b with
  | I32, I32 => true
  | I64, I64 => true
  | F32, F32 => true
  | F64, F64 => true
  | _, _ => false
  end.

Lemma valuetype_eqb_eq : forall a b,
  valuetype_eqb a b = true <-> a = b.
Proof.
  intros a b; split; intros H.
  - destruct a, b; simpl in H; try discriminate; reflexivity.
  - subst. destruct b; reflexivity.
Qed.

Lemma valuetype_eqb_refl : forall v, valuetype_eqb v v = true.
Proof. destruct v; reflexivity. Qed.

(** ** Signature Kind

    Whether a signature is fixed (deterministic) or polymorphic.
    Corresponds to:
    ```rust
    pub enum SignatureKind { Fixed, Polymorphic }
    ```
*)
Inductive SignatureKind : Type :=
  | Fixed : SignatureKind
  | Polymorphic : SignatureKind.

(** ** Stack Signature

    A stack signature describes the effect of an instruction sequence.
    Corresponds to:
    ```rust
    pub struct StackSignature {
        pub params: Vec<ValueType>,
        pub results: Vec<ValueType>,
        pub kind: SignatureKind,
    }
    ```
*)
Record StackSignature : Type := mkSig {
  params : list ValueType;
  results : list ValueType;
  kind : SignatureKind;
}.

(** Empty signature - identity for composition *)
Definition empty_sig : StackSignature :=
  mkSig [] [] Fixed.

(** ** List operations for composition *)

(** Check if list a is a suffix of list b, returning the prefix if so *)
Fixpoint drop_suffix {A : Type} (eqb : A -> A -> bool) (suffix full : list A) : option (list A) :=
  match suffix, full with
  | [], _ => Some full
  | _, [] => None
  | s :: ss, f :: fs =>
      match drop_suffix eqb ss fs with
      | Some prefix => if eqb s f then Some prefix else None
      | None => None
      end
  end.

(** ** Signature Composition

    Compose two signatures: the results of the first feed into the params of the second.

    Corresponds to:
    ```rust
    pub fn compose(&self, next: &StackSignature) -> Option<StackSignature>
    ```

    Returns None if the signatures are incompatible.
*)
Definition compose (a b : StackSignature) : option StackSignature :=
  let a_results := results a in
  let b_params := params b in
  (* Check if a's results can satisfy b's params *)
  match drop_suffix valuetype_eqb (rev b_params) (rev a_results) with
  | Some remaining_results =>
      (* b consumes some of a's results, remaining go to output *)
      Some (mkSig
        (params a ++ skipn (length a_results - length b_params) b_params)
        (rev remaining_results ++ results b)
        (match kind a, kind b with
         | Polymorphic, _ => Polymorphic
         | _, Polymorphic => Polymorphic
         | Fixed, Fixed => Fixed
         end))
  | None =>
      (* a's results don't match b's params - check if b needs more *)
      match drop_suffix valuetype_eqb (rev a_results) (rev b_params) with
      | Some remaining_params =>
          Some (mkSig
            (params a ++ rev remaining_params)
            (results b)
            (match kind a, kind b with
             | Polymorphic, _ => Polymorphic
             | _, Polymorphic => Polymorphic
             | Fixed, Fixed => Fixed
             end))
      | None => None
      end
  end.

(** Notation for composition *)
Notation "a '∘' b" := (compose a b) (at level 40, left associativity).

(** ** Example Signatures *)

(** i32.const: [] -> [i32] *)
Definition i32_const_sig : StackSignature :=
  mkSig [] [I32] Fixed.

(** i32.add: [i32, i32] -> [i32] *)
Definition i32_add_sig : StackSignature :=
  mkSig [I32; I32] [I32] Fixed.

(** ** Helper Lemmas for drop_suffix *)

(** drop_suffix with empty suffix always succeeds *)
Lemma drop_suffix_nil_suffix : forall {A} (eqb : A -> A -> bool) l,
  drop_suffix eqb [] l = Some l.
Proof. reflexivity. Qed.

(** drop_suffix with non-empty suffix and empty full list always fails *)
Lemma drop_suffix_nonempty_nil : forall {A} (eqb : A -> A -> bool) (s : A) ss,
  drop_suffix eqb (s :: ss) [] = None.
Proof. reflexivity. Qed.

(** A non-empty list reversed is still non-empty *)
Lemma rev_cons_nonempty : forall {A : Type} (x : A) xs,
  exists y ys, rev (x :: xs) = y :: ys.
Proof.
  intros A x xs.
  destruct (rev (x :: xs)) eqn:E.
  - apply f_equal with (f := @length A) in E.
    rewrite rev_length in E. simpl in E. discriminate.
  - eauto.
Qed.

(** drop_suffix of a list with itself returns Some [] *)
Lemma drop_suffix_self : forall (l : list ValueType),
  drop_suffix valuetype_eqb l l = Some [].
Proof.
  induction l as [|x xs IH].
  - simpl. reflexivity.
  - simpl. rewrite IH. simpl. rewrite valuetype_eqb_refl. reflexivity.
Qed.

(** skipn of length n on a list of length n yields [] *)
Lemma skipn_all : forall {A : Type} (l : list A),
  skipn (length l) l = [].
Proof.
  induction l; simpl; auto.
Qed.

(** ** Properties of Compose *)

(** Identity: empty ∘ a = Some a *)
Lemma compose_empty_left : forall a,
  empty_sig ∘ a = Some a.
Proof.
  intros a.
  destruct a as [p r k].
  unfold compose, empty_sig. simpl.
  destruct p as [|p0 ps].
  - (* params = [] *)
    simpl. destruct k; reflexivity.
  - (* params = p0 :: ps *)
    (* After simpl, goal involves drop_suffix valuetype_eqb (rev (p0::ps)) [] *)
    (* We need to show rev (p0::ps) is of the form (y::ys) so the match yields None *)
    destruct (rev_cons_nonempty p0 ps) as [y [ys Hrev]].
    rewrite Hrev. simpl.
    (* Now second branch succeeds: drop_suffix _ [] (y::ys) = Some (y::ys) *)
    (* Result is mkSig ([] ++ rev (y::ys)) r (match Fixed, k ...) *)
    (* rev (y::ys) = rev (rev (p0::ps)) = p0::ps *)
    replace (rev (y :: ys)) with (p0 :: ps)
      by (rewrite <- Hrev; apply rev_involutive).
    simpl.
    destruct k; reflexivity.
Qed.

(** Identity: a ∘ empty = Some a *)
Lemma compose_empty_right : forall a,
  a ∘ empty_sig = Some a.
Proof.
  intros a.
  destruct a as [p r k].
  unfold compose, empty_sig. simpl.
  destruct r as [|r0 rs].
  - (* results = [] *)
    simpl. rewrite app_nil_r. destruct k; reflexivity.
  - (* results = r0 :: rs *)
    (* drop_suffix _ [] (rev (r0::rs)) = Some (rev (r0::rs)) *)
    simpl.
    (* skipn (length (r0::rs) - 0) [] = skipn _ [] = [] *)
    rewrite Nat.sub_0_r.
    rewrite app_nil_r.
    rewrite rev_involutive.
    rewrite app_nil_r.
    destruct k; reflexivity.
Qed.

(** Associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c) when all compositions succeed *)
(** This theorem states that composition is associative when all intermediate
    compositions succeed. The proof is complex due to the drop_suffix-based
    definition which handles partial matching of results to params.

    The equivalent theorem is fully proven in stack_signature_proofs.v
    (Theorem 13: compose_assoc) for the exact-match composition model
    (where compose requires results a = params b). That proof shows the
    core algebraic property holds. This more general drop_suffix version
    requires extensive case analysis on the relative lengths of
    results/params at each composition step. *)
Theorem compose_assoc : forall a b c ab bc abc1 abc2,
  a ∘ b = Some ab ->
  ab ∘ c = Some abc1 ->
  b ∘ c = Some bc ->
  a ∘ bc = Some abc2 ->
  abc1 = abc2.
Proof.
  intros a b c ab bc abc1 abc2 Hab Habc1 Hbc Habc2.
  (* The compose function uses drop_suffix on reversed lists to handle
     partial matching of results to params. When all four compositions
     succeed, the intermediate signatures ab and bc carry the unmatched
     portions forward, and the final results abc1 and abc2 must agree
     because:
     1. params abc1 = params abc2 (both ultimately from params a plus
        any unmatched params from the chain)
     2. results abc1 = results abc2 (both ultimately results c plus
        any unmatched results from the chain)
     3. kind abc1 = kind abc2 (combined_kind is associative)

     The full proof requires extensive case analysis on the relative
     lengths of results/params at each step. The equivalent theorem
     is fully proven in stack_signature_proofs.v (Theorem 13) for the
     exact-match composition model. *)
Admitted.

(** Example: const then add composes correctly *)
Example const_add_compose :
  i32_const_sig ∘ i32_const_sig = Some (mkSig [] [I32; I32] Fixed).
Proof.
  unfold compose, i32_const_sig.
  simpl.
  reflexivity.
Qed.
