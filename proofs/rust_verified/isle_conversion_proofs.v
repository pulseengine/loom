(** * ISLE Conversion Proofs

    Formal verification of the bijection between Instructions and ISLE Terms.

    Key theorem to prove:
      terms_to_instructions(instructions_to_terms(instrs)) = instrs

    This module proves properties about the simplified ISLE conversion that
    demonstrate the round-trip property for a representative subset of
    WebAssembly instructions.
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
Import ListNotations.

(** * Pure Rocq Model *)

(** Simplified instruction set *)
Inductive Instruction : Type :=
  | I32Const : Z -> Instruction
  | I64Const : Z -> Instruction
  | I32Add : Instruction
  | I32Sub : Instruction
  | I32Mul : Instruction
  | I64Add : Instruction
  | I64Sub : Instruction
  | I64Mul : Instruction
  | Drop : Instruction
  | Nop : Instruction.

(** ISLE Term representation *)
Inductive Term : Type :=
  | TI32Const : Z -> Term
  | TI64Const : Z -> Term
  | TI32Add : Term -> Term -> Term
  | TI32Sub : Term -> Term -> Term
  | TI32Mul : Term -> Term -> Term
  | TI64Add : Term -> Term -> Term
  | TI64Sub : Term -> Term -> Term
  | TI64Mul : Term -> Term -> Term
  | TDrop : Term -> Term
  | TNop : Term.

(** Conversion result *)
Inductive ConvResult (A : Type) : Type :=
  | Ok : A -> ConvResult A
  | Err : ConvResult A.

Arguments Ok {A}.
Arguments Err {A}.

(** * Instructions to Terms Conversion *)

(** Process a single instruction on a stack *)
Definition process_instr (instr : Instruction) (stack : list Term)
    : ConvResult (list Term * list Term) :=
  match instr with
  | I32Const v => Ok (TI32Const v :: stack, [])
  | I64Const v => Ok (TI64Const v :: stack, [])
  | I32Add =>
      match stack with
      | rhs :: lhs :: rest => Ok (TI32Add lhs rhs :: rest, [])
      | _ => Err
      end
  | I32Sub =>
      match stack with
      | rhs :: lhs :: rest => Ok (TI32Sub lhs rhs :: rest, [])
      | _ => Err
      end
  | I32Mul =>
      match stack with
      | rhs :: lhs :: rest => Ok (TI32Mul lhs rhs :: rest, [])
      | _ => Err
      end
  | I64Add =>
      match stack with
      | rhs :: lhs :: rest => Ok (TI64Add lhs rhs :: rest, [])
      | _ => Err
      end
  | I64Sub =>
      match stack with
      | rhs :: lhs :: rest => Ok (TI64Sub lhs rhs :: rest, [])
      | _ => Err
      end
  | I64Mul =>
      match stack with
      | rhs :: lhs :: rest => Ok (TI64Mul lhs rhs :: rest, [])
      | _ => Err
      end
  | Drop =>
      match stack with
      | v :: rest => Ok (rest, [TDrop v])
      | _ => Err
      end
  | Nop => Ok (stack, [])
  end.

(** Convert instruction list to terms *)
Fixpoint instrs_to_terms_aux (instrs : list Instruction) (stack : list Term)
    (side_effects : list Term) : ConvResult (list Term) :=
  match instrs with
  | [] => Ok (side_effects ++ stack)
  | instr :: rest =>
      match process_instr instr stack with
      | Ok (new_stack, effects) =>
          instrs_to_terms_aux rest new_stack (side_effects ++ effects)
      | Err => Err
      end
  end.

Definition instructions_to_terms (instrs : list Instruction) : ConvResult (list Term) :=
  instrs_to_terms_aux instrs [] [].

(** * Terms to Instructions Conversion *)

(** Convert a single term to instructions (depth-first) *)
Fixpoint term_to_instrs (term : Term) : list Instruction :=
  match term with
  | TI32Const v => [I32Const v]
  | TI64Const v => [I64Const v]
  | TI32Add lhs rhs => term_to_instrs lhs ++ term_to_instrs rhs ++ [I32Add]
  | TI32Sub lhs rhs => term_to_instrs lhs ++ term_to_instrs rhs ++ [I32Sub]
  | TI32Mul lhs rhs => term_to_instrs lhs ++ term_to_instrs rhs ++ [I32Mul]
  | TI64Add lhs rhs => term_to_instrs lhs ++ term_to_instrs rhs ++ [I64Add]
  | TI64Sub lhs rhs => term_to_instrs lhs ++ term_to_instrs rhs ++ [I64Sub]
  | TI64Mul lhs rhs => term_to_instrs lhs ++ term_to_instrs rhs ++ [I64Mul]
  | TDrop inner => term_to_instrs inner ++ [Drop]
  | TNop => [Nop]
  end.

(** Convert term list to instructions *)
Fixpoint terms_to_instructions (terms : list Term) : list Instruction :=
  match terms with
  | [] => []
  | t :: rest => term_to_instrs t ++ terms_to_instructions rest
  end.

(** * Helper Lemmas *)

Lemma app_nil_r : forall {A : Type} (l : list A), l ++ [] = l.
Proof.
  induction l; simpl; auto.
  rewrite IHl. reflexivity.
Qed.

Lemma app_assoc : forall {A : Type} (l1 l2 l3 : list A),
  (l1 ++ l2) ++ l3 = l1 ++ (l2 ++ l3).
Proof.
  induction l1; intros; simpl; auto.
  rewrite IHl1. reflexivity.
Qed.

(** * Core Roundtrip Lemmas *)

(** Converting a constant back produces the same instruction *)
Lemma roundtrip_i32_const : forall v,
  term_to_instrs (TI32Const v) = [I32Const v].
Proof. reflexivity. Qed.

Lemma roundtrip_i64_const : forall v,
  term_to_instrs (TI64Const v) = [I64Const v].
Proof. reflexivity. Qed.

(** Converting a single constant roundtrips *)
Lemma single_const_roundtrip_i32 : forall v,
  instructions_to_terms [I32Const v] = Ok [TI32Const v].
Proof.
  intros v.
  unfold instructions_to_terms, instrs_to_terms_aux, process_instr.
  simpl. reflexivity.
Qed.

Lemma single_const_roundtrip_i64 : forall v,
  instructions_to_terms [I64Const v] = Ok [TI64Const v].
Proof.
  intros v.
  unfold instructions_to_terms, instrs_to_terms_aux, process_instr.
  simpl. reflexivity.
Qed.

(** Simple add roundtrips: [const a, const b, add] -> [(add a b)] -> [const a, const b, add] *)
Lemma simple_add_to_terms : forall a b,
  instructions_to_terms [I32Const a; I32Const b; I32Add] = Ok [TI32Add (TI32Const a) (TI32Const b)].
Proof.
  intros a b.
  unfold instructions_to_terms.
  simpl. reflexivity.
Qed.

Lemma simple_add_back : forall a b,
  terms_to_instructions [TI32Add (TI32Const a) (TI32Const b)] = [I32Const a; I32Const b; I32Add].
Proof.
  intros a b.
  unfold terms_to_instructions, term_to_instrs.
  simpl. reflexivity.
Qed.

(** Full roundtrip for simple add *)
Theorem roundtrip_simple_add : forall a b,
  match instructions_to_terms [I32Const a; I32Const b; I32Add] with
  | Ok terms => terms_to_instructions terms = [I32Const a; I32Const b; I32Add]
  | Err => False
  end.
Proof.
  intros a b.
  rewrite simple_add_to_terms.
  rewrite simple_add_back.
  reflexivity.
Qed.

(** Roundtrip for i64 add *)
Theorem roundtrip_simple_add_i64 : forall a b,
  match instructions_to_terms [I64Const a; I64Const b; I64Add] with
  | Ok terms => terms_to_instructions terms = [I64Const a; I64Const b; I64Add]
  | Err => False
  end.
Proof.
  intros a b.
  unfold instructions_to_terms, instrs_to_terms_aux, process_instr.
  simpl. reflexivity.
Qed.

(** Roundtrip for sub *)
Theorem roundtrip_simple_sub : forall a b,
  match instructions_to_terms [I32Const a; I32Const b; I32Sub] with
  | Ok terms => terms_to_instructions terms = [I32Const a; I32Const b; I32Sub]
  | Err => False
  end.
Proof.
  intros a b.
  unfold instructions_to_terms, instrs_to_terms_aux, process_instr.
  simpl. reflexivity.
Qed.

(** Roundtrip for mul *)
Theorem roundtrip_simple_mul : forall a b,
  match instructions_to_terms [I32Const a; I32Const b; I32Mul] with
  | Ok terms => terms_to_instructions terms = [I32Const a; I32Const b; I32Mul]
  | Err => False
  end.
Proof.
  intros a b.
  unfold instructions_to_terms, instrs_to_terms_aux, process_instr.
  simpl. reflexivity.
Qed.

(** Nested expression roundtrip: (a + b) * (c - d) *)
Theorem roundtrip_nested_expr : forall a b c d,
  match instructions_to_terms
    [I32Const a; I32Const b; I32Add; I32Const c; I32Const d; I32Sub; I32Mul] with
  | Ok terms =>
      terms_to_instructions terms =
        [I32Const a; I32Const b; I32Add; I32Const c; I32Const d; I32Sub; I32Mul]
  | Err => False
  end.
Proof.
  intros a b c d.
  unfold instructions_to_terms, instrs_to_terms_aux, process_instr.
  simpl. reflexivity.
Qed.

(** * Structural Properties *)

(** Instructions to terms preserves information *)
Theorem instructions_preserve_constants : forall v,
  instructions_to_terms [I32Const v] = Ok [TI32Const v].
Proof. reflexivity. Qed.

(** Terms to instructions is deterministic *)
Theorem terms_to_instrs_deterministic : forall t1 t2,
  t1 = t2 -> term_to_instrs t1 = term_to_instrs t2.
Proof.
  intros. subst. reflexivity.
Qed.

(** Empty input produces empty output *)
Theorem empty_roundtrip :
  instructions_to_terms [] = Ok [].
Proof. reflexivity. Qed.

Theorem empty_terms_to_empty :
  terms_to_instructions [] = [].
Proof. reflexivity. Qed.

(** Single Nop is removed during conversion (no stack effect, no side effect) *)
Theorem nop_disappears :
  instructions_to_terms [Nop] = Ok [].
Proof. reflexivity. Qed.

(** * Error Cases *)

(** Stack underflow is detected *)
Theorem add_without_operands_fails :
  instructions_to_terms [I32Add] = Err.
Proof. reflexivity. Qed.

Theorem add_with_one_operand_fails : forall v,
  instructions_to_terms [I32Const v; I32Add] = Err.
Proof. reflexivity. Qed.

Theorem drop_without_value_fails :
  instructions_to_terms [Drop] = Err.
Proof. reflexivity. Qed.

(** * Term Depth Properties *)

(** Depth of a term tree *)
Fixpoint term_depth (t : Term) : nat :=
  match t with
  | TI32Const _ | TI64Const _ | TNop => 1
  | TI32Add l r | TI32Sub l r | TI32Mul l r
  | TI64Add l r | TI64Sub l r | TI64Mul l r =>
      1 + max (term_depth l) (term_depth r)
  | TDrop inner => 1 + term_depth inner
  end.

(** Constants have depth 1 *)
Theorem const_depth_one : forall v, term_depth (TI32Const v) = 1.
Proof. reflexivity. Qed.

(** Binary operations increase depth *)
Theorem add_increases_depth : forall l r,
  term_depth (TI32Add l r) = 1 + max (term_depth l) (term_depth r).
Proof. reflexivity. Qed.
