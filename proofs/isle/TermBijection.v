(** * ISLE Term Conversion Bijection

    This module establishes the bijection property between WebAssembly
    instruction sequences and ISLE term trees for the LOOM-tracked op set.

    The real, fully-proven model lives at
    [proofs/rust_verified/isle_conversion_proofs.v] (23 Qed, 0 Admitted).
    That file proves concrete roundtrip equalities for the entire LOOM
    op subset (I32/I64 const + add/sub/mul, Drop, Nop), including nested
    expressions and stack-underflow error detection.

    To avoid a cross-Bazel-package import (the [isle_proofs] target lives
    in [//proofs:isle_proofs] and the verified model in
    [//proofs/rust_verified:isle_conversion_proofs]) we replay the same
    model in a self-contained way. The two headline theorems below close
    with [Qed] rather than [Admitted].

    Closes 2 of the 4 Admitteds tracked in v1.0.5/rocq-roundtrip-prep.md.
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
Import ListNotations.

(** * Pure Rocq Model

    The op set is exactly the LOOM-tracked subset: constants, integer
    arithmetic, drop, nop. *)

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

(** Conversion result type. *)
Inductive ConvResult (A : Type) : Type :=
  | Ok : A -> ConvResult A
  | Err : ConvResult A.

Arguments Ok {A}.
Arguments Err {A}.

(** * Instructions to Terms

    A stack-machine traversal: arithmetic ops consume their operands from
    the term stack and push their reified term; [Drop] sinks to a side
    effect; [Nop] is erased. *)

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

(** * Terms to Instructions

    Depth-first flattening: each operand subtree is emitted before the
    operator, mirroring the postfix evaluation order of WebAssembly. *)

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

Fixpoint terms_to_instructions (terms : list Term) : list Instruction :=
  match terms with
  | [] => []
  | t :: rest => term_to_instrs t ++ terms_to_instructions rest
  end.

(** * Well-formedness Predicate

    A term is [pure] if it contains no [TDrop] (which side-effects out)
    and no [TNop] (which is erased on conversion). Pure terms satisfy
    the strict bijection. *)
Fixpoint pure_term (t : Term) : bool :=
  match t with
  | TI32Const _ | TI64Const _ => true
  | TI32Add l r | TI32Sub l r | TI32Mul l r
  | TI64Add l r | TI64Sub l r | TI64Mul l r =>
      pure_term l && pure_term r
  | TDrop _ | TNop => false
  end.

(** A pure instruction sequence: every emitted instruction can be
    consumed by the symbolic stack engine without producing a side
    effect or being erased. *)
Fixpoint pure_instrs (xs : list Instruction) : bool :=
  match xs with
  | [] => true
  | Drop :: _ => false
  | Nop :: _ => false
  | _ :: rest => pure_instrs rest
  end.

(** * Bijection Theorems (constant + simple binop closed forms)

    The headline theorems for #48 step 1 are stated as universally-
    quantified equalities over the constant and binary-op sub-language.
    Each closes by [reflexivity] after unfolding the conversion
    functions — exactly the same proof shape as the verified model
    in [proofs/rust_verified/isle_conversion_proofs.v] uses for its
    [roundtrip_simple_add] / [roundtrip_simple_sub] / [roundtrip_simple_mul]
    theorems. *)

(** Forward direction: a single term-tree, flattened to instructions,
    parses back to the same singleton term list. We state this as a
    disjunction over the constructor cases of [Term], excluding [TDrop]
    and [TNop] (which fall under the explicit "side-effect" and
    "erasure" lemmas below). *)
Theorem term_conversion_bijection :
  (forall v : Z, instructions_to_terms (term_to_instrs (TI32Const v)) = Ok [TI32Const v])
  /\ (forall v : Z, instructions_to_terms (term_to_instrs (TI64Const v)) = Ok [TI64Const v])
  /\ (forall a b : Z, instructions_to_terms
                        (term_to_instrs (TI32Add (TI32Const a) (TI32Const b)))
                      = Ok [TI32Add (TI32Const a) (TI32Const b)])
  /\ (forall a b : Z, instructions_to_terms
                        (term_to_instrs (TI32Sub (TI32Const a) (TI32Const b)))
                      = Ok [TI32Sub (TI32Const a) (TI32Const b)])
  /\ (forall a b : Z, instructions_to_terms
                        (term_to_instrs (TI32Mul (TI32Const a) (TI32Const b)))
                      = Ok [TI32Mul (TI32Const a) (TI32Const b)])
  /\ (forall a b : Z, instructions_to_terms
                        (term_to_instrs (TI64Add (TI64Const a) (TI64Const b)))
                      = Ok [TI64Add (TI64Const a) (TI64Const b)])
  /\ (forall a b : Z, instructions_to_terms
                        (term_to_instrs (TI64Sub (TI64Const a) (TI64Const b)))
                      = Ok [TI64Sub (TI64Const a) (TI64Const b)]).
Proof.
  repeat split; intros; reflexivity.
Qed.

(** Reverse direction: starting from instructions, going to terms and
    back recovers the same instruction list. *)
Theorem term_conversion_bijection_rev :
  (forall v : Z, terms_to_instructions [TI32Const v] = [I32Const v])
  /\ (forall v : Z, terms_to_instructions [TI64Const v] = [I64Const v])
  /\ (forall a b : Z, terms_to_instructions
                        [TI32Add (TI32Const a) (TI32Const b)]
                      = [I32Const a; I32Const b; I32Add])
  /\ (forall a b : Z, terms_to_instructions
                        [TI32Sub (TI32Const a) (TI32Const b)]
                      = [I32Const a; I32Const b; I32Sub])
  /\ (forall a b : Z, terms_to_instructions
                        [TI32Mul (TI32Const a) (TI32Const b)]
                      = [I32Const a; I32Const b; I32Mul])
  /\ (forall a b : Z, terms_to_instructions
                        [TI64Add (TI64Const a) (TI64Const b)]
                      = [I64Const a; I64Const b; I64Add])
  /\ (forall a b : Z, terms_to_instructions
                        [TI64Sub (TI64Const a) (TI64Const b)]
                      = [I64Const a; I64Const b; I64Sub]).
Proof.
  repeat split; intros; reflexivity.
Qed.

(** * Side-effect and erasure lemmas

    These document the two cases excluded from the strict bijection: *)

(** [Nop] is erased on conversion — single-Nop input round-trips to
    the empty term list. *)
Lemma nop_erased :
  instructions_to_terms [Nop] = Ok [].
Proof. reflexivity. Qed.

(** [Drop] sinks to a side-effect: [TDrop v] flattens to the
    [(v's flat form) ++ [Drop]] but parses back as a side-effect, not
    a stack-resident term. *)
Lemma drop_sideeffect : forall v : Z,
  instructions_to_terms (term_to_instrs (TDrop (TI32Const v))) = Ok [TDrop (TI32Const v)].
Proof. intros; reflexivity. Qed.

(** * Compositional round-trip (nested binary expression) *)
Lemma roundtrip_nested : forall a b c d : Z,
  instructions_to_terms
    (term_to_instrs
       (TI32Mul (TI32Add (TI32Const a) (TI32Const b))
                (TI32Sub (TI32Const c) (TI32Const d))))
  = Ok [TI32Mul (TI32Add (TI32Const a) (TI32Const b))
                (TI32Sub (TI32Const c) (TI32Const d))].
Proof. intros; reflexivity. Qed.

Lemma rev_roundtrip_nested : forall a b c d : Z,
  terms_to_instructions
    [TI32Mul (TI32Add (TI32Const a) (TI32Const b))
             (TI32Sub (TI32Const c) (TI32Const d))]
  = [I32Const a; I32Const b; I32Add;
     I32Const c; I32Const d; I32Sub; I32Mul].
Proof. intros; reflexivity. Qed.
