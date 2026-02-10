(** * WebAssembly Operational Semantics

    This module defines the formal operational semantics for WebAssembly
    instructions, providing the foundation for proving optimization correctness.

    Key definitions:
    - Value: WebAssembly runtime values (i32, i64, f32, f64)
    - State: Execution state (value stack + memory)
    - step: Small-step operational semantics

    The semantics follow the WebAssembly specification:
    https://webassembly.github.io/spec/core/exec/index.html
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
Import ListNotations.

Open Scope Z_scope.

(** * Value Types *)

(** WebAssembly value types *)
Inductive ValueType : Type :=
  | TI32 : ValueType
  | TI64 : ValueType
  | TF32 : ValueType
  | TF64 : ValueType.

(** Decidable equality for ValueType *)
Definition valuetype_eqb (a b : ValueType) : bool :=
  match a, b with
  | TI32, TI32 | TI64, TI64 | TF32, TF32 | TF64, TF64 => true
  | _, _ => false
  end.

Lemma valuetype_eqb_refl : forall v, valuetype_eqb v v = true.
Proof. destruct v; reflexivity. Qed.

Lemma valuetype_eqb_eq : forall a b, valuetype_eqb a b = true <-> a = b.
Proof.
  intros a b; split; destruct a, b; simpl; intros; try reflexivity; discriminate.
Qed.

(** * Runtime Values *)

(** WebAssembly runtime values *)
Inductive Value : Type :=
  | VI32 : Z -> Value        (** 32-bit integer (stored as Z for proofs) *)
  | VI64 : Z -> Value        (** 64-bit integer *)
  | VF32 : Z -> Value        (** 32-bit float (represented as bits) *)
  | VF64 : Z -> Value.       (** 64-bit float (represented as bits) *)

(** Get the type of a value *)
Definition value_type (v : Value) : ValueType :=
  match v with
  | VI32 _ => TI32
  | VI64 _ => TI64
  | VF32 _ => TF32
  | VF64 _ => TF64
  end.

(** Value equality *)
Definition value_eqb (a b : Value) : bool :=
  match a, b with
  | VI32 x, VI32 y => Z.eqb x y
  | VI64 x, VI64 y => Z.eqb x y
  | VF32 x, VF32 y => Z.eqb x y
  | VF64 x, VF64 y => Z.eqb x y
  | _, _ => false
  end.

Lemma value_eqb_refl : forall v, value_eqb v v = true.
Proof.
  destruct v; simpl; apply Z.eqb_refl.
Qed.

(** * Modular Arithmetic for WebAssembly Semantics *)

(** 32-bit wrap: result is in range [0, 2^32) interpreted as signed *)
Definition wrap32 (z : Z) : Z := Z.modulo z (Z.pow 2 32).

(** 64-bit wrap: result is in range [0, 2^64) interpreted as signed *)
Definition wrap64 (z : Z) : Z := Z.modulo z (Z.pow 2 64).

(** Convert unsigned 32-bit to signed interpretation *)
Definition to_signed32 (z : Z) : Z :=
  let wrapped := wrap32 z in
  if Z.ltb wrapped (Z.pow 2 31) then wrapped
  else wrapped - Z.pow 2 32.

(** Convert unsigned 64-bit to signed interpretation *)
Definition to_signed64 (z : Z) : Z :=
  let wrapped := wrap64 z in
  if Z.ltb wrapped (Z.pow 2 63) then wrapped
  else wrapped - Z.pow 2 64.

(** Normalize for comparison - treat as unsigned *)
Definition to_unsigned32 (z : Z) : Z := wrap32 z.
Definition to_unsigned64 (z : Z) : Z := wrap64 z.

(** * Integer Arithmetic Operations *)

(** i32.add with wrapping semantics *)
Definition i32_add (a b : Z) : Z := wrap32 (a + b).

(** i32.sub with wrapping semantics *)
Definition i32_sub (a b : Z) : Z := wrap32 (a - b).

(** i32.mul with wrapping semantics *)
Definition i32_mul (a b : Z) : Z := wrap32 (a * b).

(** i64.add with wrapping semantics *)
Definition i64_add (a b : Z) : Z := wrap64 (a + b).

(** i64.sub with wrapping semantics *)
Definition i64_sub (a b : Z) : Z := wrap64 (a - b).

(** i64.mul with wrapping semantics *)
Definition i64_mul (a b : Z) : Z := wrap64 (a * b).

(** * Bitwise Operations *)

(** i32.and *)
Definition i32_and (a b : Z) : Z := Z.land (wrap32 a) (wrap32 b).

(** i32.or *)
Definition i32_or (a b : Z) : Z := Z.lor (wrap32 a) (wrap32 b).

(** i32.xor *)
Definition i32_xor (a b : Z) : Z := Z.lxor (wrap32 a) (wrap32 b).

(** i64.and *)
Definition i64_and (a b : Z) : Z := Z.land (wrap64 a) (wrap64 b).

(** i64.or *)
Definition i64_or (a b : Z) : Z := Z.lor (wrap64 a) (wrap64 b).

(** i64.xor *)
Definition i64_xor (a b : Z) : Z := Z.lxor (wrap64 a) (wrap64 b).

(** * Shift Operations *)

(** Mask for 32-bit shift amount (only low 5 bits matter) *)
Definition shift_mask32 : Z := 31.

(** Mask for 64-bit shift amount (only low 6 bits matter) *)
Definition shift_mask64 : Z := 63.

(** i32.shl - logical shift left *)
Definition i32_shl (a b : Z) : Z :=
  wrap32 (Z.shiftl (wrap32 a) (Z.land (wrap32 b) shift_mask32)).

(** i32.shr_u - logical shift right (unsigned) *)
Definition i32_shr_u (a b : Z) : Z :=
  Z.shiftr (wrap32 a) (Z.land (wrap32 b) shift_mask32).

(** i32.shr_s - arithmetic shift right (signed) *)
Definition i32_shr_s (a b : Z) : Z :=
  let shift := Z.land (wrap32 b) shift_mask32 in
  let signed_a := to_signed32 a in
  wrap32 (Z.shiftr signed_a shift).

(** i64.shl - logical shift left *)
Definition i64_shl (a b : Z) : Z :=
  wrap64 (Z.shiftl (wrap64 a) (Z.land (wrap64 b) shift_mask64)).

(** i64.shr_u - logical shift right (unsigned) *)
Definition i64_shr_u (a b : Z) : Z :=
  Z.shiftr (wrap64 a) (Z.land (wrap64 b) shift_mask64).

(** i64.shr_s - arithmetic shift right (signed) *)
Definition i64_shr_s (a b : Z) : Z :=
  let shift := Z.land (wrap64 b) shift_mask64 in
  let signed_a := to_signed64 a in
  wrap64 (Z.shiftr signed_a shift).

(** * Comparison Operations *)

(** All comparisons return i32 (0 or 1) *)

(** i32.eq - equality *)
Definition i32_eq (a b : Z) : Z :=
  if Z.eqb (wrap32 a) (wrap32 b) then 1 else 0.

(** i32.ne - inequality *)
Definition i32_ne (a b : Z) : Z :=
  if Z.eqb (wrap32 a) (wrap32 b) then 0 else 1.

(** i32.lt_s - less than (signed) *)
Definition i32_lt_s (a b : Z) : Z :=
  if Z.ltb (to_signed32 a) (to_signed32 b) then 1 else 0.

(** i32.lt_u - less than (unsigned) *)
Definition i32_lt_u (a b : Z) : Z :=
  if Z.ltb (wrap32 a) (wrap32 b) then 1 else 0.

(** i32.gt_s - greater than (signed) *)
Definition i32_gt_s (a b : Z) : Z :=
  if Z.ltb (to_signed32 b) (to_signed32 a) then 1 else 0.

(** i32.gt_u - greater than (unsigned) *)
Definition i32_gt_u (a b : Z) : Z :=
  if Z.ltb (wrap32 b) (wrap32 a) then 1 else 0.

(** i32.le_s - less than or equal (signed) *)
Definition i32_le_s (a b : Z) : Z :=
  if Z.leb (to_signed32 a) (to_signed32 b) then 1 else 0.

(** i32.le_u - less than or equal (unsigned) *)
Definition i32_le_u (a b : Z) : Z :=
  if Z.leb (wrap32 a) (wrap32 b) then 1 else 0.

(** i32.ge_s - greater than or equal (signed) *)
Definition i32_ge_s (a b : Z) : Z :=
  if Z.leb (to_signed32 b) (to_signed32 a) then 1 else 0.

(** i32.ge_u - greater than or equal (unsigned) *)
Definition i32_ge_u (a b : Z) : Z :=
  if Z.leb (wrap32 b) (wrap32 a) then 1 else 0.

(** i32.eqz - equal to zero *)
Definition i32_eqz (a : Z) : Z :=
  if Z.eqb (wrap32 a) 0 then 1 else 0.

(** Similar i64 comparisons *)
Definition i64_eq (a b : Z) : Z :=
  if Z.eqb (wrap64 a) (wrap64 b) then 1 else 0.

Definition i64_ne (a b : Z) : Z :=
  if Z.eqb (wrap64 a) (wrap64 b) then 0 else 1.

Definition i64_lt_s (a b : Z) : Z :=
  if Z.ltb (to_signed64 a) (to_signed64 b) then 1 else 0.

Definition i64_lt_u (a b : Z) : Z :=
  if Z.ltb (wrap64 a) (wrap64 b) then 1 else 0.

Definition i64_gt_s (a b : Z) : Z :=
  if Z.ltb (to_signed64 b) (to_signed64 a) then 1 else 0.

Definition i64_gt_u (a b : Z) : Z :=
  if Z.ltb (wrap64 b) (wrap64 a) then 1 else 0.

Definition i64_le_s (a b : Z) : Z :=
  if Z.leb (to_signed64 a) (to_signed64 b) then 1 else 0.

Definition i64_le_u (a b : Z) : Z :=
  if Z.leb (wrap64 a) (wrap64 b) then 1 else 0.

Definition i64_ge_s (a b : Z) : Z :=
  if Z.leb (to_signed64 b) (to_signed64 a) then 1 else 0.

Definition i64_ge_u (a b : Z) : Z :=
  if Z.leb (wrap64 b) (wrap64 a) then 1 else 0.

Definition i64_eqz (a : Z) : Z :=
  if Z.eqb (wrap64 a) 0 then 1 else 0.

(** * Execution State *)

(** Value stack is a list of values (head = top of stack) *)
Definition Stack := list Value.

(** Memory is modeled as a partial function from addresses to bytes.
    For simplicity, we model 32-bit loads/stores directly. *)
Definition Memory := Z -> option Value.

(** Empty memory: all addresses undefined *)
Definition empty_memory : Memory := fun _ => None.

(** Execution state: stack and memory *)
Record State := mkState {
  stack : Stack;
  memory : Memory;
}.

(** Initial state with empty stack and memory *)
Definition initial_state : State := mkState [] empty_memory.

(** * Stack Operations *)

(** Push a value onto the stack *)
Definition push (v : Value) (s : State) : State :=
  mkState (v :: stack s) (memory s).

(** Pop a value from the stack (returns None if stack is empty) *)
Definition pop (s : State) : option (Value * State) :=
  match stack s with
  | [] => None
  | v :: rest => Some (v, mkState rest (memory s))
  end.

(** Pop two values from the stack *)
Definition pop2 (s : State) : option (Value * Value * State) :=
  match stack s with
  | v2 :: v1 :: rest => Some (v1, v2, mkState rest (memory s))
  | _ => None
  end.

(** * Instructions *)

(** Simplified instruction set for core optimizations *)
Inductive Instruction : Type :=
  (* Constants *)
  | Instr_I32Const : Z -> Instruction
  | Instr_I64Const : Z -> Instruction
  (* Arithmetic *)
  | Instr_I32Add : Instruction
  | Instr_I32Sub : Instruction
  | Instr_I32Mul : Instruction
  | Instr_I64Add : Instruction
  | Instr_I64Sub : Instruction
  | Instr_I64Mul : Instruction
  (* Bitwise *)
  | Instr_I32And : Instruction
  | Instr_I32Or : Instruction
  | Instr_I32Xor : Instruction
  | Instr_I64And : Instruction
  | Instr_I64Or : Instruction
  | Instr_I64Xor : Instruction
  (* Shifts *)
  | Instr_I32Shl : Instruction
  | Instr_I32ShrS : Instruction
  | Instr_I32ShrU : Instruction
  | Instr_I64Shl : Instruction
  | Instr_I64ShrS : Instruction
  | Instr_I64ShrU : Instruction
  (* Comparisons *)
  | Instr_I32Eq : Instruction
  | Instr_I32Ne : Instruction
  | Instr_I32LtS : Instruction
  | Instr_I32LtU : Instruction
  | Instr_I32GtS : Instruction
  | Instr_I32GtU : Instruction
  | Instr_I32LeS : Instruction
  | Instr_I32LeU : Instruction
  | Instr_I32GeS : Instruction
  | Instr_I32GeU : Instruction
  | Instr_I32Eqz : Instruction
  | Instr_I64Eq : Instruction
  | Instr_I64Ne : Instruction
  | Instr_I64LtS : Instruction
  | Instr_I64LtU : Instruction
  | Instr_I64GtS : Instruction
  | Instr_I64GtU : Instruction
  | Instr_I64LeS : Instruction
  | Instr_I64LeU : Instruction
  | Instr_I64GeS : Instruction
  | Instr_I64GeU : Instruction
  | Instr_I64Eqz : Instruction
  (* Stack manipulation *)
  | Instr_Drop : Instruction
  | Instr_Nop : Instruction.

(** * Small-Step Operational Semantics *)

(** Single instruction execution *)
Definition exec_instr (instr : Instruction) (s : State) : option State :=
  match instr with
  (* Constants push onto stack *)
  | Instr_I32Const v => Some (push (VI32 v) s)
  | Instr_I64Const v => Some (push (VI64 v) s)

  (* Binary i32 operations *)
  | Instr_I32Add =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_add a b)) s')
      | _ => None
      end
  | Instr_I32Sub =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_sub a b)) s')
      | _ => None
      end
  | Instr_I32Mul =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_mul a b)) s')
      | _ => None
      end

  (* Binary i64 operations *)
  | Instr_I64Add =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_add a b)) s')
      | _ => None
      end
  | Instr_I64Sub =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_sub a b)) s')
      | _ => None
      end
  | Instr_I64Mul =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_mul a b)) s')
      | _ => None
      end

  (* Bitwise i32 *)
  | Instr_I32And =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_and a b)) s')
      | _ => None
      end
  | Instr_I32Or =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_or a b)) s')
      | _ => None
      end
  | Instr_I32Xor =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_xor a b)) s')
      | _ => None
      end

  (* Bitwise i64 *)
  | Instr_I64And =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_and a b)) s')
      | _ => None
      end
  | Instr_I64Or =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_or a b)) s')
      | _ => None
      end
  | Instr_I64Xor =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_xor a b)) s')
      | _ => None
      end

  (* Shifts i32 *)
  | Instr_I32Shl =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_shl a b)) s')
      | _ => None
      end
  | Instr_I32ShrS =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_shr_s a b)) s')
      | _ => None
      end
  | Instr_I32ShrU =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_shr_u a b)) s')
      | _ => None
      end

  (* Shifts i64 *)
  | Instr_I64Shl =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_shl a b)) s')
      | _ => None
      end
  | Instr_I64ShrS =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_shr_s a b)) s')
      | _ => None
      end
  | Instr_I64ShrU =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI64 (i64_shr_u a b)) s')
      | _ => None
      end

  (* i32 Comparisons *)
  | Instr_I32Eq =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_eq a b)) s')
      | _ => None
      end
  | Instr_I32Ne =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_ne a b)) s')
      | _ => None
      end
  | Instr_I32LtS =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_lt_s a b)) s')
      | _ => None
      end
  | Instr_I32LtU =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_lt_u a b)) s')
      | _ => None
      end
  | Instr_I32GtS =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_gt_s a b)) s')
      | _ => None
      end
  | Instr_I32GtU =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_gt_u a b)) s')
      | _ => None
      end
  | Instr_I32LeS =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_le_s a b)) s')
      | _ => None
      end
  | Instr_I32LeU =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_le_u a b)) s')
      | _ => None
      end
  | Instr_I32GeS =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_ge_s a b)) s')
      | _ => None
      end
  | Instr_I32GeU =>
      match pop2 s with
      | Some (VI32 a, VI32 b, s') => Some (push (VI32 (i32_ge_u a b)) s')
      | _ => None
      end
  | Instr_I32Eqz =>
      match pop s with
      | Some (VI32 a, s') => Some (push (VI32 (i32_eqz a)) s')
      | _ => None
      end

  (* i64 Comparisons *)
  | Instr_I64Eq =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_eq a b)) s')
      | _ => None
      end
  | Instr_I64Ne =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_ne a b)) s')
      | _ => None
      end
  | Instr_I64LtS =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_lt_s a b)) s')
      | _ => None
      end
  | Instr_I64LtU =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_lt_u a b)) s')
      | _ => None
      end
  | Instr_I64GtS =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_gt_s a b)) s')
      | _ => None
      end
  | Instr_I64GtU =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_gt_u a b)) s')
      | _ => None
      end
  | Instr_I64LeS =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_le_s a b)) s')
      | _ => None
      end
  | Instr_I64LeU =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_le_u a b)) s')
      | _ => None
      end
  | Instr_I64GeS =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_ge_s a b)) s')
      | _ => None
      end
  | Instr_I64GeU =>
      match pop2 s with
      | Some (VI64 a, VI64 b, s') => Some (push (VI32 (i64_ge_u a b)) s')
      | _ => None
      end
  | Instr_I64Eqz =>
      match pop s with
      | Some (VI64 a, s') => Some (push (VI32 (i64_eqz a)) s')
      | _ => None
      end

  (* Stack manipulation *)
  | Instr_Drop =>
      match pop s with
      | Some (_, s') => Some s'
      | _ => None
      end
  | Instr_Nop => Some s
  end.

(** Execute a sequence of instructions *)
Fixpoint exec_instrs (instrs : list Instruction) (s : State) : option State :=
  match instrs with
  | [] => Some s
  | i :: rest =>
      match exec_instr i s with
      | Some s' => exec_instrs rest s'
      | None => None
      end
  end.

(** * Basic Semantic Properties *)

(** Nop doesn't change state *)
Theorem nop_identity : forall s, exec_instr Instr_Nop s = Some s.
Proof. reflexivity. Qed.

(** Constant push is always successful *)
Theorem const_i32_succeeds : forall v s,
  exec_instr (Instr_I32Const v) s = Some (push (VI32 v) s).
Proof. reflexivity. Qed.

Theorem const_i64_succeeds : forall v s,
  exec_instr (Instr_I64Const v) s = Some (push (VI64 v) s).
Proof. reflexivity. Qed.

(** Empty instruction sequence is identity *)
Theorem empty_instrs_identity : forall s,
  exec_instrs [] s = Some s.
Proof. reflexivity. Qed.

(** Instruction sequence composition *)
Theorem exec_instrs_app : forall i1 i2 s s' s'',
  exec_instrs i1 s = Some s' ->
  exec_instrs i2 s' = Some s'' ->
  exec_instrs (i1 ++ i2) s = Some s''.
Proof.
  intros i1.
  induction i1 as [|a i1' IH]; intros i2 st st' st'' H1 H2.
  - (* Base case: i1 = [] *)
    simpl in H1. simpl.
    (* H1 : Some st = Some st', so st = st' *)
    replace st' with st in H2 by (inversion H1; reflexivity).
    exact H2.
  - (* Inductive case: i1 = a :: i1' *)
    simpl in H1. simpl.
    destruct (exec_instr a st) as [st_mid|] eqn:Ha.
    + eapply IH. exact H1. exact H2.
    + discriminate.
Qed.

(** * Semantic Equivalence *)

(** Two instruction sequences are equivalent if they produce the same result *)
Definition instrs_equiv (i1 i2 : list Instruction) : Prop :=
  forall s, exec_instrs i1 s = exec_instrs i2 s.

(** Equivalence is reflexive *)
Theorem instrs_equiv_refl : forall i, instrs_equiv i i.
Proof. unfold instrs_equiv. reflexivity. Qed.

(** Equivalence is symmetric *)
Theorem instrs_equiv_sym : forall i1 i2,
  instrs_equiv i1 i2 -> instrs_equiv i2 i1.
Proof. unfold instrs_equiv. intros. symmetry. apply H. Qed.

(** Equivalence is transitive *)
Theorem instrs_equiv_trans : forall i1 i2 i3,
  instrs_equiv i1 i2 -> instrs_equiv i2 i3 -> instrs_equiv i1 i3.
Proof.
  unfold instrs_equiv. intros.
  rewrite H. apply H0.
Qed.

(** Nop can be removed from any sequence *)
Theorem nop_elimination_front : forall instrs,
  instrs_equiv (Instr_Nop :: instrs) instrs.
Proof.
  unfold instrs_equiv. intros. simpl. reflexivity.
Qed.

(** * Basic Constant Folding Semantics *)

(** Adding two constants produces their sum *)
Theorem i32_add_const_const : forall a b s,
  exec_instrs [Instr_I32Const a; Instr_I32Const b; Instr_I32Add] s =
  exec_instrs [Instr_I32Const (i32_add a b)] s.
Proof.
  intros. simpl.
  unfold push, pop2.
  destruct s as [stk mem].
  simpl. reflexivity.
Qed.

Theorem i32_sub_const_const : forall a b s,
  exec_instrs [Instr_I32Const a; Instr_I32Const b; Instr_I32Sub] s =
  exec_instrs [Instr_I32Const (i32_sub a b)] s.
Proof.
  intros. simpl.
  unfold push, pop2.
  destruct s as [stk mem].
  simpl. reflexivity.
Qed.

Theorem i32_mul_const_const : forall a b s,
  exec_instrs [Instr_I32Const a; Instr_I32Const b; Instr_I32Mul] s =
  exec_instrs [Instr_I32Const (i32_mul a b)] s.
Proof.
  intros. simpl.
  unfold push, pop2.
  destruct s as [stk mem].
  simpl. reflexivity.
Qed.

(** 64-bit versions *)
Theorem i64_add_const_const : forall a b s,
  exec_instrs [Instr_I64Const a; Instr_I64Const b; Instr_I64Add] s =
  exec_instrs [Instr_I64Const (i64_add a b)] s.
Proof.
  intros. simpl.
  unfold push, pop2.
  destruct s as [stk mem].
  simpl. reflexivity.
Qed.

Theorem i64_sub_const_const : forall a b s,
  exec_instrs [Instr_I64Const a; Instr_I64Const b; Instr_I64Sub] s =
  exec_instrs [Instr_I64Const (i64_sub a b)] s.
Proof.
  intros. simpl.
  unfold push, pop2.
  destruct s as [stk mem].
  simpl. reflexivity.
Qed.

Theorem i64_mul_const_const : forall a b s,
  exec_instrs [Instr_I64Const a; Instr_I64Const b; Instr_I64Mul] s =
  exec_instrs [Instr_I64Const (i64_mul a b)] s.
Proof.
  intros. simpl.
  unfold push, pop2.
  destruct s as [stk mem].
  simpl. reflexivity.
Qed.

(** * Algebraic Properties of Arithmetic *)

(** i32.add is commutative *)
Theorem i32_add_comm : forall a b, i32_add a b = i32_add b a.
Proof.
  intros. unfold i32_add.
  f_equal. lia.
Qed.

(** i32.add is associative *)
Theorem i32_add_assoc : forall a b c,
  i32_add (i32_add a b) c = i32_add a (i32_add b c).
Proof.
  intros. unfold i32_add, wrap32.
  (* Modular arithmetic associativity *)
  rewrite Zplus_mod_idemp_l.
  rewrite Zplus_mod_idemp_r.
  f_equal. lia.
Qed.

(** i32.add with 0 is identity (right) *)
Theorem i32_add_0_r : forall a, i32_add a 0 = wrap32 a.
Proof.
  intros. unfold i32_add, wrap32.
  rewrite Z.add_0_r. reflexivity.
Qed.

(** i32.add with 0 is identity (left) *)
Theorem i32_add_0_l : forall a, i32_add 0 a = wrap32 a.
Proof.
  intros. unfold i32_add, wrap32.
  rewrite Z.add_0_l. reflexivity.
Qed.

(** i32.mul with 1 is identity *)
Theorem i32_mul_1_r : forall a, i32_mul a 1 = wrap32 a.
Proof.
  intros. unfold i32_mul, wrap32.
  rewrite Z.mul_1_r. reflexivity.
Qed.

Theorem i32_mul_1_l : forall a, i32_mul 1 a = wrap32 a.
Proof.
  intros. unfold i32_mul, wrap32.
  rewrite Z.mul_1_l. reflexivity.
Qed.

(** i32.mul with 0 is 0 *)
Theorem i32_mul_0_r : forall a, i32_mul a 0 = 0.
Proof.
  intros. unfold i32_mul, wrap32.
  rewrite Z.mul_0_r. reflexivity.
Qed.

Theorem i32_mul_0_l : forall a, i32_mul 0 a = 0.
Proof.
  intros. unfold i32_mul, wrap32.
  rewrite Z.mul_0_l. reflexivity.
Qed.

(** i32.sub with 0 is identity *)
Theorem i32_sub_0_r : forall a, i32_sub a 0 = wrap32 a.
Proof.
  intros. unfold i32_sub. f_equal. lia.
Qed.

(** x - x = 0 *)
Theorem i32_sub_self : forall a, i32_sub a a = 0.
Proof.
  intros. unfold i32_sub, wrap32.
  replace (a - a) with 0 by lia.
  simpl. reflexivity.
Qed.

(** * Bitwise Properties *)

(** x & 0 = 0 *)
Theorem i32_and_0_r : forall a, i32_and a 0 = 0.
Proof.
  intros. unfold i32_and, wrap32.
  rewrite Z.land_0_r. reflexivity.
Qed.

Theorem i32_and_0_l : forall a, i32_and 0 a = 0.
Proof.
  intros. unfold i32_and, wrap32.
  rewrite Z.land_0_l. reflexivity.
Qed.

(** x | 0 = x *)
Theorem i32_or_0_r : forall a, i32_or a 0 = wrap32 a.
Proof.
  intros. unfold i32_or, wrap32.
  rewrite Z.lor_0_r. reflexivity.
Qed.

Theorem i32_or_0_l : forall a, i32_or 0 a = wrap32 a.
Proof.
  intros. unfold i32_or, wrap32.
  rewrite Z.lor_0_l. reflexivity.
Qed.

(** x ^ 0 = x *)
Theorem i32_xor_0_r : forall a, i32_xor a 0 = wrap32 a.
Proof.
  intros. unfold i32_xor, wrap32. simpl. apply Z.lxor_0_r.
Qed.

Theorem i32_xor_0_l : forall a, i32_xor 0 a = wrap32 a.
Proof.
  intros. unfold i32_xor, wrap32. simpl. apply Z.lxor_0_l.
Qed.

(** x ^ x = 0 *)
Theorem i32_xor_self : forall a, i32_xor a a = 0.
Proof.
  intros. unfold i32_xor. apply Z.lxor_nilpotent.
Qed.

(** x & x = x (for wrapped values) *)
Theorem i32_and_self : forall a, i32_and a a = wrap32 a.
Proof.
  intros. unfold i32_and. apply Z.land_diag.
Qed.

(** x | x = x (for wrapped values) *)
Theorem i32_or_self : forall a, i32_or a a = wrap32 a.
Proof.
  intros. unfold i32_or. apply Z.lor_diag.
Qed.

(** * Comparison Properties *)

(** x == x is always 1 *)
Theorem i32_eq_refl : forall a, i32_eq a a = 1.
Proof.
  intros. unfold i32_eq.
  rewrite Z.eqb_refl. reflexivity.
Qed.

(** x != x is always 0 *)
Theorem i32_ne_refl : forall a, i32_ne a a = 0.
Proof.
  intros. unfold i32_ne.
  rewrite Z.eqb_refl. reflexivity.
Qed.

Close Scope Z_scope.
