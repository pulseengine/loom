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
From Stdlib Require Import Lia.
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
      (* b takes all its params from the top of a's results: the input is
         exactly a's params, and a's unconsumed results stay below b's. *)
      Some (mkSig
        (params a)
        (rev remaining_results ++ results b)
        (match kind a, kind b with
         | Polymorphic, _ => Polymorphic
         | _, Polymorphic => Polymorphic
         | Fixed, Fixed => Fixed
         end))
  | None =>
      (* b needs more than a provides: a's results are the top of b's
         params, and the rest of b's params sit below a's params. *)
      match drop_suffix valuetype_eqb (rev a_results) (rev b_params) with
      | Some remaining_params =>
          Some (mkSig
            (rev remaining_params ++ params a)
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
    (* rev (y::ys) = rev (rev (p0::ps)) = p0::ps. [simpl] has already
       unfolded rev (y::ys) to rev ys ++ [y], so rewrite with that form. *)
    assert (Hback : rev ys ++ [y] = p0 :: ps).
    { change (rev ys ++ [y]) with (rev (y :: ys)).
      rewrite <- Hrev. apply rev_involutive. }
    rewrite ?app_nil_r, Hback.
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
    simpl. destruct k; reflexivity.
  - (* results = r0 :: rs: b = empty consumes nothing, so all of a's
       results remain. [simpl] may have unfolded rev (rev (r0::rs)) to
       rev (rev rs ++ [r0]); normalise back to r0 :: rs. *)
    simpl.
    rewrite ?app_nil_r, ?rev_unit, ?rev_involutive, ?app_nil_r.
    destruct k; reflexivity.
Qed.

(** ** Kind algebra (mirror of stack_signature_proofs.v Theorems 9-12) *)

Definition combined_kind (k1 k2 : SignatureKind) : SignatureKind :=
  match k1, k2 with
  | Polymorphic, _ => Polymorphic
  | _, Polymorphic => Polymorphic
  | Fixed, Fixed => Fixed
  end.

Lemma combined_kind_assoc : forall k1 k2 k3,
  combined_kind (combined_kind k1 k2) k3
  = combined_kind k1 (combined_kind k2 k3).
Proof. intros [|] [|] [|]; reflexivity. Qed.

(** Every kind-output of [compose] is precisely [combined_kind] of inputs. *)
Lemma compose_kind : forall a b ab,
  a ∘ b = Some ab ->
  kind ab = combined_kind (kind a) (kind b).
Proof.
  intros a b ab Hab.
  unfold compose in Hab.
  destruct (drop_suffix valuetype_eqb (rev (params b)) (rev (results a)))
    as [rem1|] eqn:H1.
  - injection Hab as Hab_eq; subst ab; simpl.
    unfold combined_kind.
    destruct (kind a), (kind b); reflexivity.
  - destruct (drop_suffix valuetype_eqb (rev (results a)) (rev (params b)))
      as [rem2|] eqn:H2; [|discriminate].
    injection Hab as Hab_eq; subst ab; simpl.
    unfold combined_kind.
    destruct (kind a), (kind b); reflexivity.
Qed.

(** ** Associativity (general statement)

    [(a ∘ b) ∘ c = a ∘ (b ∘ c)] when all intermediate compositions succeed.

    Earlier versions of [compose] computed the params of the composite as
    [params a ++ skipn (length a_results - length b_params) b_params] when
    [b]'s params are covered by [a]'s results, and as
    [params a ++ remaining] when [b] needs more than [a] provides. Both are
    wrong for a stack (the first counts [b]'s consumed params twice; the
    second puts [b]'s extra params above [a]'s), and with them [compose_assoc]
    is false: for [a = [] -> [i32]] and [c = [i32] -> []], [(a ∘ a) ∘ c]
    gave [[] -> [i32]] while [a ∘ (a ∘ c)] gave [[i32] -> []].

    The proof is semantic. [run s] is the effect of [s] on a stack, written
    bottom to top: it replaces a top segment equal to [params s] with
    [results s]. [compose_run] shows that composing signatures composes
    their effects; associativity of function composition then gives equal
    effects, and [run_inj] recovers equality of params and results from
    equal effects. LOOM's runtime checker (loom-core/src/stack.rs) uses the
    stricter exact-match operator, which is the special case
    [results a = params b] of this one. *)

(** [drop_suffix] is prefix removal: it succeeds exactly when [full] starts
    with [suffix], returning the rest. *)
Lemma drop_suffix_spec : forall s f p,
  drop_suffix valuetype_eqb s f = Some p <-> f = s ++ p.
Proof.
  induction s as [|x xs IH]; intros f p; simpl.
  - split; intros H; congruence.
  - destruct f as [|y ys].
    + split; intros H; discriminate.
    + destruct (drop_suffix valuetype_eqb xs ys) as [q|] eqn:Hq.
      * destruct (valuetype_eqb x y) eqn:Hxy.
        -- apply valuetype_eqb_eq in Hxy. subst y.
           apply IH in Hq. subst ys.
           split; intros H.
           ++ injection H as <-. reflexivity.
           ++ injection H as H. apply app_inv_head in H. subst. reflexivity.
        -- split; intros H; [discriminate|].
           injection H as <- _. rewrite valuetype_eqb_refl in Hxy. discriminate.
      * split; intros H; [discriminate|].
        injection H as _ H. apply IH in H. congruence.
Qed.

(** Remove [p] from the top of stack [st] (stacks are bottom to top). *)
Definition strip_top (p st : list ValueType) : option (list ValueType) :=
  match drop_suffix valuetype_eqb (rev p) (rev st) with
  | Some q => Some (rev q)
  | None => None
  end.

Lemma strip_top_spec : forall p st x,
  strip_top p st = Some x <-> st = x ++ p.
Proof.
  intros p st x. unfold strip_top.
  destruct (drop_suffix valuetype_eqb (rev p) (rev st)) as [q|] eqn:Hq.
  - apply drop_suffix_spec in Hq.
    split; intros H.
    + injection H as <-.
      rewrite <- (rev_involutive st), Hq, rev_app_distr, rev_involutive.
      reflexivity.
    + f_equal. subst st. rewrite rev_app_distr in Hq.
      apply app_inv_head in Hq. subst q. apply rev_involutive.
  - split; intros H; [discriminate|].
    subst st. rewrite rev_app_distr in Hq.
    assert (drop_suffix valuetype_eqb (rev p) (rev p ++ rev x) = Some (rev x))
      as Hs by (apply drop_suffix_spec; reflexivity).
    congruence.
Qed.

Lemma strip_top_app : forall p x, strip_top p (x ++ p) = Some x.
Proof. intros. apply strip_top_spec. reflexivity. Qed.

(** Effect of a signature on a stack. *)
Definition run (s : StackSignature) (st : list ValueType)
    : option (list ValueType) :=
  match strip_top (params s) st with
  | Some x => Some (x ++ results s)
  | None => None
  end.

Definition bind_stack (o : option (list ValueType))
    (f : list ValueType -> option (list ValueType)) : option (list ValueType) :=
  match o with Some st => f st | None => None end.

Lemma run_app : forall s x, run s (x ++ params s) = Some (x ++ results s).
Proof. intros. unfold run. rewrite strip_top_app. reflexivity. Qed.

Lemma run_none : forall s st,
  (forall x, st <> x ++ params s) -> run s st = None.
Proof.
  intros s st H. unfold run.
  destruct (strip_top (params s) st) as [x|] eqn:E; [|reflexivity].
  apply strip_top_spec in E. exfalso. exact (H x E).
Qed.

(** The two shapes of a successful composition, as list equations. *)
Lemma compose_shape : forall a b ab,
  a ∘ b = Some ab ->
  (exists r, results a = r ++ params b /\
             params ab = params a /\ results ab = r ++ results b) \/
  (exists p, params b = p ++ results a /\
             params ab = p ++ params a /\ results ab = results b).
Proof.
  intros a b ab H. unfold compose in H.
  destruct (drop_suffix valuetype_eqb (rev (params b)) (rev (results a)))
    as [q|] eqn:H1.
  - injection H as <-. left. exists (rev q). simpl. repeat split.
    apply drop_suffix_spec in H1.
    rewrite <- (rev_involutive (results a)), H1, rev_app_distr, rev_involutive.
    reflexivity.
  - destruct (drop_suffix valuetype_eqb (rev (results a)) (rev (params b)))
      as [q|] eqn:H2; [|discriminate].
    injection H as <-. right. exists (rev q). simpl. repeat split.
    apply drop_suffix_spec in H2.
    rewrite <- (rev_involutive (params b)), H2, rev_app_distr, rev_involutive.
    reflexivity.
Qed.

(** Composing signatures composes their effects. *)
Lemma compose_run : forall a b ab,
  a ∘ b = Some ab ->
  forall st, run ab st = bind_stack (run a st) (run b).
Proof.
  intros a b ab H st.
  destruct (compose_shape a b ab H)
    as [[r [Hr [Hp Hres]]] | [p [Hpb [Hp Hres]]]].
  - destruct (strip_top (params a) st) as [x|] eqn:E.
    + apply strip_top_spec in E. subst st.
      rewrite <- Hp at 1. rewrite run_app, run_app. simpl.
      rewrite Hr, app_assoc, run_app, Hres, app_assoc. reflexivity.
    + assert (Hn : forall x, st <> x ++ params a).
      { intros x Hx. apply strip_top_spec in Hx. congruence. }
      rewrite (run_none a st Hn). apply run_none. rewrite Hp. exact Hn.
  - destruct (strip_top (params ab) st) as [x|] eqn:E.
    + apply strip_top_spec in E. subst st.
      rewrite run_app, Hp, app_assoc, run_app. simpl.
      rewrite <- app_assoc, <- Hpb, run_app, Hres. reflexivity.
    + assert (Hn : forall x, st <> x ++ params ab).
      { intros x Hx. apply strip_top_spec in Hx. congruence. }
      rewrite (run_none ab st Hn).
      destruct (strip_top (params a) st) as [y|] eqn:Ey.
      * apply strip_top_spec in Ey. subst st.
        rewrite run_app. simpl. symmetry. apply run_none.
        intros z Hz. rewrite Hpb, app_assoc in Hz.
        apply app_inv_tail in Hz. subst y.
        apply (Hn z). rewrite Hp, app_assoc. reflexivity.
      * assert (Hm : forall x, st <> x ++ params a).
        { intros x Hx. apply strip_top_spec in Hx. congruence. }
        rewrite (run_none a st Hm). reflexivity.
Qed.

(** Equal effects determine params and results. *)
Lemma run_inj : forall s1 s2,
  (forall st, run s1 st = run s2 st) ->
  params s1 = params s2 /\ results s1 = results s2.
Proof.
  intros s1 s2 H.
  assert (H1 := H (params s1)). assert (H2 := H (params s2)).
  rewrite <- (app_nil_l (params s1)) in H1 at 1.
  rewrite <- (app_nil_l (params s2)) in H2 at 2.
  rewrite run_app in H1. rewrite run_app in H2. simpl in H1, H2.
  unfold run in H1, H2.
  destruct (strip_top (params s2) (params s1)) as [x|] eqn:E1;
    [|discriminate].
  destruct (strip_top (params s1) (params s2)) as [y|] eqn:E2;
    [|discriminate].
  apply strip_top_spec in E1. apply strip_top_spec in E2.
  injection H1 as H1. injection H2 as H2.
  assert (Hx : x = []).
  { apply length_zero_iff_nil.
    assert (L1 := f_equal (@length _) E1). assert (L2 := f_equal (@length _) E2).
    rewrite length_app in L1, L2. lia. }
  subst x. simpl in *. split; congruence.
Qed.

(** Kind component of [compose_assoc] — fully closed by [combined_kind_assoc]. *)
Theorem compose_assoc_kind : forall a b c ab bc abc1 abc2,
  a ∘ b = Some ab ->
  ab ∘ c = Some abc1 ->
  b ∘ c = Some bc ->
  a ∘ bc = Some abc2 ->
  kind abc1 = kind abc2.
Proof.
  intros a b c ab bc abc1 abc2 Hab Habc1 Hbc Habc2.
  apply compose_kind in Hab.
  apply compose_kind in Hbc.
  apply compose_kind in Habc1.
  apply compose_kind in Habc2.
  rewrite Habc1, Habc2.
  rewrite Hab, Hbc.
  apply combined_kind_assoc.
Qed.

Theorem compose_assoc : forall a b c ab bc abc1 abc2,
  a ∘ b = Some ab ->
  ab ∘ c = Some abc1 ->
  b ∘ c = Some bc ->
  a ∘ bc = Some abc2 ->
  abc1 = abc2.
Proof.
  intros a b c ab bc abc1 abc2 Hab Habc1 Hbc Habc2.
  assert (Hk := compose_assoc_kind a b c ab bc abc1 abc2 Hab Habc1 Hbc Habc2).
  assert (Hrun : forall st, run abc1 st = run abc2 st).
  { intros st.
    rewrite (compose_run ab c abc1 Habc1), (compose_run a bc abc2 Habc2),
            (compose_run a b ab Hab).
    destruct (run a st) as [st'|]; simpl; [|reflexivity].
    symmetry. apply (compose_run b c bc Hbc). }
  destruct (run_inj abc1 abc2 Hrun) as [Hp Hr].
  destruct abc1, abc2; simpl in *; subst; reflexivity.
Qed.

(** Example: const then add composes correctly *)
Example const_add_compose :
  i32_const_sig ∘ i32_const_sig = Some (mkSig [] [I32; I32] Fixed).
Proof.
  unfold compose, i32_const_sig.
  simpl.
  reflexivity.
Qed.
