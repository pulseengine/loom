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

(** ** Properties to Prove *)

(** Identity: empty ∘ a = Some a *)
Lemma compose_empty_left : forall a,
  empty_sig ∘ a = Some a.
Proof.
  intros a.
  unfold compose, empty_sig.
  simpl.
  (* TODO: Complete proof *)
Admitted.

(** Identity: a ∘ empty = Some a *)
Lemma compose_empty_right : forall a,
  a ∘ empty_sig = Some a.
Proof.
  intros a.
  unfold compose, empty_sig.
  simpl.
  (* TODO: Complete proof *)
Admitted.

(** Associativity: (a ∘ b) ∘ c = a ∘ (b ∘ c) when all compositions succeed *)
Theorem compose_assoc : forall a b c ab bc abc1 abc2,
  a ∘ b = Some ab ->
  ab ∘ c = Some abc1 ->
  b ∘ c = Some bc ->
  a ∘ bc = Some abc2 ->
  abc1 = abc2.
Proof.
  intros a b c ab bc abc1 abc2 Hab Habc1 Hbc Habc2.
  (* TODO: Complete proof - this is the main theorem *)
Admitted.

(** Example: const then add composes correctly *)
Example const_add_compose :
  i32_const_sig ∘ i32_const_sig = Some (mkSig [] [I32; I32] Fixed).
Proof.
  unfold compose, i32_const_sig.
  simpl.
  reflexivity.
Qed.
