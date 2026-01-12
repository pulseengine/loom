(** * ISLE Term Conversion Bijection

    This module will prove that the conversion between Instructions
    and ISLE Terms is a bijection:

    terms_to_instructions(instructions_to_terms(x)) = x

    TODO: This requires translating the term types and conversion
    functions from Rust.
*)

Require Import Coq.Lists.List.
Import ListNotations.

(** Placeholder for Instruction type *)
Inductive Instruction : Type :=
  | I32Const : nat -> Instruction
  | I32Add : Instruction
  | End : Instruction.

(** Placeholder for ISLE Value/Term type *)
Inductive Term : Type :=
  | TConst : nat -> Term
  | TAdd : Term -> Term -> Term.

(** Axiomatized conversion functions *)
Axiom instructions_to_terms : list Instruction -> list Term.
Axiom terms_to_instructions : list Term -> list Instruction.

(** The bijection theorem to prove *)
Theorem term_conversion_bijection : forall instrs : list Instruction,
  terms_to_instructions (instructions_to_terms instrs) = instrs.
Proof.
  (* TODO: Requires full implementation *)
Admitted.

(** Reverse direction *)
Theorem term_conversion_bijection_rev : forall terms : list Term,
  instructions_to_terms (terms_to_instructions terms) = terms.
Proof.
  (* TODO: Requires full implementation *)
Admitted.
