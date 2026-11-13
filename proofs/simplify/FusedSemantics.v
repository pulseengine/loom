(** * Concrete call semantics for the fused-module optimizer

    This file gives the fused optimizer's passes a concrete, executable
    semantics, so that their correctness is a theorem rather than an axiom.
    It replaces, pass by pass, the [exec_call] axiom of FusedOptimization.v
    (issue #169).

    ## What is modelled

    - Values are integers ([Z]).
    - Functions: imports first, then defined functions, as in a wasm index
      space. A defined function has a signature, a count of extra locals
      and a body.
    - Structured control flow: [block], [loop], [if], [br], [br_if],
      [return], and [call]; and [nop].
    - Locals: [local.get] and [local.set].
    - Memory instructions: [IMem op ks] names memories by index; the
      indices resolve, through the module's memory imports, to keys, and
      the effect is the section variable [mprim] applied to those keys.
    - Every other instruction is an [IPrim p], and its effect on the operand
      stack, the locals and the store is given by the section variable
      [prim]. The store (linear memories, globals, tables) is the abstract
      section type [store], and imported functions are the section variable
      [host]. Every theorem below therefore holds for every choice of
      [prim], [mprim], [host] and [store]: no instruction outside the list above,
      and no host behaviour, is assumed.

    ## Stack convention

    The head of a [list Z] is the top of the operand stack. A call to a
    function with [n] parameters takes [firstn n stk]. The deepest of those
    values is parameter 0, so the callee's locals start as
    [rev (firstn n stk)] followed by zeros for the extra locals.

    ## Fuel

    [eval_seq k] is a fuel-indexed interpreter: [None] means the fuel ran
    out, and any other answer is the program's real result (a trap is a
    result). Fuel is spent on every call and every loop iteration. Two
    programs are equivalent when, for each input, one of them reaches a
    result with some fuel exactly when the other reaches the same result
    with some fuel. Divergence is preserved too, since it means that
    neither reaches a result.

    ## Correspondence to loom-core/src/fused_optimizer.rs

    loom's function bodies end in an explicit [End]. Here a body is the
    list before that [End]: [is_trivial_adapter] in Rust checks
    [local.get 0..n-1; call t; end], which is [trivial_adapter] below.
*)

From Stdlib Require Import Bool.
From Stdlib Require Import Arith.
From Stdlib Require Import List.
From Stdlib Require Import ZArith.
From Stdlib Require Import Lia.
Import ListNotations.

(** ** Syntax *)

(** A value type code. Signature equality is all the optimizer compares. *)
Definition vtype := nat.

Record sig : Type := mkSig {
  sig_params : list vtype;
  sig_results : list vtype;
}.

(** Instructions and instruction sequences ([code]) are mutually inductive,
    so that recursion into a nested body is structural. *)
Inductive instr : Type :=
  | ILocalGet : nat -> instr
  | ILocalSet : nat -> instr
  | ICall : nat -> instr
  | IPrim : nat -> instr
  | IBlock : nat -> code -> instr          (** result arity, body *)
  | ILoop : nat -> code -> instr           (** result arity, body *)
  | IIf : nat -> code -> code -> instr
  | IBr : nat -> instr
  | IBrIf : nat -> instr
  | IReturn : instr
  | INop : instr
  | IMem : nat -> list nat -> instr        (** operation, memory indices *)
with code : Type :=
  | CNil : code
  | CCons : instr -> code -> code.

Scheme instr_mut := Induction for instr Sort Prop
  with code_mut := Induction for code Sort Prop.

Record func : Type := mkFunc {
  f_sig : sig;
  f_locals : nat;          (** extra locals beyond the parameters *)
  f_body : code;
}.

(** An imported function: the key it is resolved by and its signature.
    The key stands for loom's [ImportKey] (module name, field name, type
    index). The host is given the key, not the import's position, so two
    imports with the same key are bound to the same host function. This is
    how an embedder that links by name behaves; the wasm specification
    itself leaves the binding to the embedder. *)
Record import : Type := mkImport {
  imp_key : nat;
  imp_sig : sig;
}.

(** A module's memories are imported memories only, each given by the
    key it is resolved by: memory index [k] is the [k]-th entry of
    [m_mems]. As with functions, the store is given the key, so two
    memory imports with the same key are the same memory. A module with a
    defined memory is outside this model; [deduplicate_memory_imports]
    declines such modules. *)
Record module : Type := mkModule {
  m_imports : list import; (** imported functions, indices 0.. *)
  m_funcs : list func;     (** defined functions follow the imports *)
  m_mems : list nat;       (** keys of the imported memories, indices 0.. *)
}.

(** ** Call rewriting

    [rw_code rho c] replaces every [call f] in [c], at any nesting depth,
    by [call (rho f)]. This is what [rewrite_calls] in fused_optimizer.rs
    does: it recurses into block, loop and both arms of if. *)
Fixpoint rw_instr (rho : nat -> nat) (i : instr) : instr :=
  match i with
  | ICall f => ICall (rho f)
  | IBlock ar b => IBlock ar (rw_code rho b)
  | ILoop ar b => ILoop ar (rw_code rho b)
  | IIf ar b1 b2 => IIf ar (rw_code rho b1) (rw_code rho b2)
  | _ => i
  end
with rw_code (rho : nat -> nat) (c : code) : code :=
  match c with
  | CNil => CNil
  | CCons i c' => CCons (rw_instr rho i) (rw_code rho c')
  end.

Lemma rw_code_id : forall c, rw_code (fun f => f) c = c.
Proof.
  apply (code_mut (fun i => rw_instr (fun f => f) i = i)
                  (fun c => rw_code (fun f => f) c = c));
    intros; cbn; congruence.
Qed.

(** ** Trivial adapters

    [gets j n tail] is [local.get j; local.get (j+1); ...; local.get (j+n-1)]
    followed by [tail]. *)
Fixpoint gets (j n : nat) (tail : code) : code :=
  match n with
  | 0 => tail
  | S n' => CCons (ILocalGet j) (gets (S j) n' tail)
  end.

(** A trivial adapter to [t]: no extra locals, and the body is
    [local.get 0; ...; local.get (n-1); call t] where [n] is the number of
    parameters. This is exactly what [is_trivial_adapter] in
    loom-core/src/fused_optimizer.rs accepts (its trailing [End] is the end
    of the list here). *)
Definition trivial_adapter (d : func) (t : nat) : Prop :=
  f_locals d = 0 /\
  f_body d = gets 0 (length (sig_params (f_sig d))) (CCons (ICall t) CNil).

(** [rw_module rho m] applies [rw_code rho] to every defined function's
    body and changes nothing else. This is what [devirtualize_adapters]
    does with the resolved adapter map. *)
Definition rw_func (rho : nat -> nat) (d : func) : func :=
  mkFunc (f_sig d) (f_locals d) (rw_code rho (f_body d)).

Definition rw_module (rho : nat -> nat) (m : module) : module :=
  mkModule (m_imports m) (map (rw_func rho) (m_funcs m)) (m_mems m).

(** ** Semantics *)

(** ** Trivial-call elimination

    [nop_code c]: [c] consists of [nop]s only. With the trailing [End]
    dropped, this is what [is_nop_body] in fused_optimizer.rs accepts
    (every instruction is [Nop] or [End]; the empty body included). *)
Inductive nop_code : code -> Prop :=
  | nop_nil : nop_code CNil
  | nop_cons : forall c, nop_code c -> nop_code (CCons INop c).

(** [trivial_fn m f]: [f] is a defined function of type [() -> ()] whose
    body is [nop]s only. Its number of extra locals is unrestricted, as in
    [eliminate_trivial_calls], which does not look at locals. *)
Definition trivial_fn (m : module) (f : nat) : Prop :=
  length (m_imports m) <= f /\
  exists d, nth_error (m_funcs m) (f - length (m_imports m)) = Some d /\
            sig_params (f_sig d) = [] /\ sig_results (f_sig d) = [] /\
            nop_code (f_body d).

(** [del_code T c] deletes every [call f] with [T f = true], at any
    nesting depth. This is [remove_trivial_calls_from_block]: it recurses
    into block, loop and both arms of if, then drops the calls at the
    current level. [T] plays the role of the [trivial_funcs] set. *)
Fixpoint del_instr (T : nat -> bool) (i : instr) : instr :=
  match i with
  | IBlock ar b => IBlock ar (del_code T b)
  | ILoop ar b => ILoop ar (del_code T b)
  | IIf ar b1 b2 => IIf ar (del_code T b1) (del_code T b2)
  | _ => i
  end
with del_code (T : nat -> bool) (c : code) : code :=
  match c with
  | CNil => CNil
  | CCons i c' =>
      match i with
      | ICall f => if T f then del_code T c' else CCons (ICall f) (del_code T c')
      | _ => CCons (del_instr T i) (del_code T c')
      end
  end.

Definition del_func (T : nat -> bool) (d : func) : func :=
  mkFunc (f_sig d) (f_locals d) (del_code T (f_body d)).

Definition del_module (T : nat -> bool) (m : module) : module :=
  mkModule (m_imports m) (map (del_func T) (m_funcs m)) (m_mems m).

(** ** Import deduplication

    [dedup_module rho imps' m] replaces the import list by [imps'] and
    rewrites every [call f] to [call (rho f)]. [dedup_ok] is what
    [deduplicate_imports] in fused_optimizer.rs establishes: every import
    [i] maps to an import of [imps'] with the same key and signature (the
    first import with that [ImportKey]), and every defined function moves
    down by the number of imports removed. *)
Definition dedup_module (rho : nat -> nat) (imps' : list import) (m : module) : module :=
  mkModule imps' (map (rw_func rho) (m_funcs m)) (m_mems m).

Definition dedup_ok (m : module) (imps' : list import) (rho : nat -> nat) : Prop :=
  (forall i, i < length (m_imports m) ->
     rho i < length imps' /\ nth_error imps' (rho i) = nth_error (m_imports m) i) /\
  (forall f, length (m_imports m) <= f ->
     rho f = f - length (m_imports m) + length imps').

(** ** Memory import deduplication

    [mrw_code sg c] replaces every memory index [k] named by a memory
    instruction in [c], at any nesting depth, by [sg k]. This is
    [remap_memory_indices] in fused_optimizer.rs: it covers every
    instruction of loom's IR that carries a memory index (the 23 loads and
    stores, memory.size, .grow, .fill, .copy with both indices, and .init)
    and recurses into block, loop and both arms of if. *)
Fixpoint mrw_instr (sg : nat -> nat) (i : instr) : instr :=
  match i with
  | IMem p ks => IMem p (map sg ks)
  | IBlock ar b => IBlock ar (mrw_code sg b)
  | ILoop ar b => ILoop ar (mrw_code sg b)
  | IIf ar b1 b2 => IIf ar (mrw_code sg b1) (mrw_code sg b2)
  | _ => i
  end
with mrw_code (sg : nat -> nat) (c : code) : code :=
  match c with
  | CNil => CNil
  | CCons i c' => CCons (mrw_instr sg i) (mrw_code sg c')
  end.

Definition mrw_func (sg : nat -> nat) (d : func) : func :=
  mkFunc (f_sig d) (f_locals d) (mrw_code sg (f_body d)).

(** [mdedup_module sg mems' m] replaces the memory imports by [mems'] and
    remaps every memory index by [sg]. Function indices do not move. *)
Definition mdedup_module (sg : nat -> nat) (mems' : list nat) (m : module) : module :=
  mkModule (m_imports m) (map (mrw_func sg) (m_funcs m)) mems'.

(** [mdedup_ok]: every memory index, in range or not, names the same key
    (or no memory) before and after. *)
Definition mdedup_ok (m : module) (mems' : list nat) (sg : nat -> nat) : Prop :=
  forall k, nth_error mems' (sg k) = nth_error (m_mems m) k.

(** The remap [deduplicate_memory_imports] builds for [n] memory imports:
    indices [1 .. n-1] go to [0], every other index is left alone. *)
Definition rust_mem_remap (n k : nat) : nat :=
  if (0 <? k) && (k <? n) then 0 else k.

(** What [deduplicate_memory_imports] checks is enough: if every memory
    import has the key of the first one, keeping only the first and
    remapping by [rust_mem_remap] satisfies [mdedup_ok]. Rust compares the
    import's module and field name, which is the key, and also the
    limits, sharedness and index type, which the proof does not need. *)
Lemma rust_mem_remap_ok : forall m key rest,
  m_mems m = key :: rest ->
  Forall (fun x => x = key) rest ->
  mdedup_ok m [key] (rust_mem_remap (S (length rest))).
Proof.
  intros m key rest Hm Hall k. rewrite Hm. unfold rust_mem_remap.
  destruct k as [|k]; [reflexivity|].
  replace ((0 <? S k) && (S k <? S (length rest))) with (S k <? S (length rest))
    by reflexivity.
  destruct (S k <? S (length rest)) eqn:E.
  - apply Nat.ltb_lt in E. cbn.
    destruct (nth_error rest k) as [x|] eqn:En.
    + rewrite Forall_forall in Hall. rewrite (Hall x (nth_error_In _ _ En)). reflexivity.
    + apply nth_error_None in En. lia.
  - apply Nat.ltb_ge in E. cbn.
    assert (En : nth_error rest k = None) by (apply nth_error_None; lia).
    rewrite En. destruct k; reflexivity.
Qed.

(** [mem_keys ms ks] resolves each memory index in [ks] to its key, or is
    [None] if one is out of range. *)
Fixpoint mem_keys (ms : list nat) (ks : list nat) : option (list nat) :=
  match ks with
  | [] => Some []
  | k :: ks' =>
      match nth_error ms k, mem_keys ms ks' with
      | Some a, Some r => Some (a :: r)
      | _, _ => None
      end
  end.

(** ** Dead function elimination

    [calls_c P c] holds when every [call f] in [c], at any nesting depth,
    has [P f]. *)
Fixpoint calls_i (P : nat -> Prop) (i : instr) : Prop :=
  match i with
  | ICall f => P f
  | IBlock _ b | ILoop _ b => calls_c P b
  | IIf _ b1 b2 => calls_c P b1 /\ calls_c P b2
  | _ => True
  end
with calls_c (P : nat -> Prop) (c : code) : Prop :=
  match c with
  | CNil => True
  | CCons i rest => calls_i P i /\ calls_c P rest
  end.

Lemma calls_true : forall c, calls_c (fun _ => True) c.
Proof.
  apply (code_mut (fun i => calls_i (fun _ => True) i)
                  (fun c => calls_c (fun _ => True) c));
    intros; cbn; auto.
Qed.

(** [keep_from kf j0 l] keeps the [j]-th element of [l] when [kf (j0 + j)];
    [cnt kf j0 j] counts the kept positions among [j0 .. j0 + j - 1]. *)
Fixpoint keep_from (kf : nat -> bool) (j0 : nat) (l : list func) : list func :=
  match l with
  | [] => []
  | d :: l' => if kf j0 then d :: keep_from kf (S j0) l' else keep_from kf (S j0) l'
  end.

Fixpoint cnt (kf : nat -> bool) (j0 j : nat) : nat :=
  match j with
  | 0 => 0
  | S j' => (if kf j0 then 1 else 0) + cnt kf (S j0) j'
  end.

(** [eliminate_dead_functions] in fused_optimizer.rs. [kf j] says whether
    defined function [j] (index [length (m_imports m) + j]) is kept. Imports
    are kept and keep their indices; a kept defined function moves down by
    the number of removed functions before it ([remap]); every call in a kept
    body is rewritten through that map. *)
Definition dfe_rho (kf : nat -> bool) (ni : nat) (f : nat) : nat :=
  if f <? ni then f else ni + cnt kf 0 (f - ni).

Definition dfe_module (kf : nat -> bool) (m : module) : module :=
  let ni := length (m_imports m) in
  mkModule (m_imports m) (map (rw_func (dfe_rho kf ni)) (keep_from kf 0 (m_funcs m)))
           (m_mems m).

(** The live set: every import, and every kept defined function. *)
Definition live (kf : nat -> bool) (ni : nat) (f : nat) : Prop :=
  f < ni \/ kf (f - ni) = true.

(** What the worklist in [eliminate_dead_functions_inner] establishes: the
    live set is closed under calls. Every call in a kept body names an
    import or a kept function. *)
Definition dfe_ok (kf : nat -> bool) (m : module) : Prop :=
  forall j d, kf j = true -> nth_error (m_funcs m) j = Some d ->
    calls_c (live kf (length (m_imports m))) (f_body d).

Lemma keep_nth : forall kf l j0 j d,
  kf (j0 + j) = true -> nth_error l j = Some d ->
  nth_error (keep_from kf j0 l) (cnt kf j0 j) = Some d.
Proof.
  intros kf l; induction l as [|d0 l IH]; intros j0 j d Hk H;
    [destruct j; discriminate|].
  destruct j as [|j]; cbn in H |- *.
  - rewrite Nat.add_0_r in Hk. rewrite Hk. exact H.
  - replace (j0 + S j) with (S j0 + j) in Hk by lia.
    destruct (kf j0); cbn; apply IH; assumption.
Qed.

Lemma keep_out : forall kf l j0 j, length l <= j ->
  nth_error (keep_from kf j0 l) (cnt kf j0 j) = None.
Proof.
  intros kf l; induction l as [|d0 l IH]; intros j0 j Hj;
    [destruct (cnt kf j0 j); reflexivity|].
  destruct j as [|j]; cbn in Hj; [lia|]. cbn.
  destruct (kf j0); cbn; apply IH; lia.
Qed.

(** ** Function body deduplication

    [dedupe_function_bodies] in fused_optimizer.rs redirects calls from a
    defined function to the lowest-indexed defined function whose
    signature, locals and body are identical, and rewrites every call with
    [rewrite_calls]. [same_fn m f g] is that condition: [g] is [f], or both
    are defined functions with the same record. *)
Definition same_fn (m : module) (f g : nat) : Prop :=
  g = f \/
  (length (m_imports m) <= f /\ length (m_imports m) <= g /\
   nth_error (m_funcs m) (f - length (m_imports m))
   = nth_error (m_funcs m) (g - length (m_imports m))).

Section Semantics.

Variable store : Type.

(** Effect of every instruction not modelled above. [None] is a trap. *)
Variable prim : nat -> list Z -> list Z -> store -> option (list Z * list Z * store).

(** Behaviour of the host function bound to import key [k] on its
    arguments. [None] is a trap. *)
Variable host : nat -> list Z -> store -> option (list Z * store).

(** Effect of a memory instruction (load, store, memory.size, .grow,
    .fill, .copy, .init) given the keys of the memories it names. [None]
    is a trap. *)
Variable mprim : nat -> list nat -> list Z -> list Z -> store
                 -> option (list Z * list Z * store).

(** Result of running an instruction sequence. *)
Inductive res : Type :=
  | RNormal : list Z -> list Z -> store -> res   (** stack, locals, store *)
  | RBr : nat -> list Z -> list Z -> store -> res
  | RRet : list Z -> store -> res
  | RTrap : res.

(** Result of a call. *)
Inductive cres : Type :=
  | CDone : list Z -> store -> cres              (** caller's new stack, store *)
  | CTrap : cres.

Definition replace_nth (l : list Z) (i : nat) (v : Z) : list Z :=
  firstn i l ++ v :: skipn (S i) l.

(** Signature of function index [f], import or defined. *)
Definition sig_of (m : module) (f : nat) : option sig :=
  if f <? length (m_imports m) then option_map imp_sig (nth_error (m_imports m) f)
  else option_map f_sig (nth_error (m_funcs m) (f - length (m_imports m))).

(** One call, given an interpreter [run] for the callee's body.
    The caller's stack keeps [skipn n stk]; the callee's results
    ([firstn nr]) are pushed on top of it. *)
Definition do_call (run : code -> list Z -> list Z -> store -> option res)
                   (m : module) (f : nat) (stk : list Z) (s : store) : option cres :=
  if f <? length (m_imports m) then
    match nth_error (m_imports m) f with
    | None => Some CTrap
    | Some im =>
        let n := length (sig_params (imp_sig im)) in
        let nr := length (sig_results (imp_sig im)) in
        if n <=? length stk then
          match host (imp_key im) (firstn n stk) s with
          | None => Some CTrap
          | Some (out, s') => Some (CDone (firstn nr out ++ skipn n stk) s')
          end
        else Some CTrap
    end
  else
    match nth_error (m_funcs m) (f - length (m_imports m)) with
    | None => Some CTrap
    | Some d =>
        let n := length (sig_params (f_sig d)) in
        let nr := length (sig_results (f_sig d)) in
        if n <=? length stk then
          match run (f_body d) [] (rev (firstn n stk) ++ repeat 0%Z (f_locals d)) s with
          | None => None
          | Some (RNormal out _ s') | Some (RRet out s') | Some (RBr 0 out _ s') =>
              Some (CDone (firstn nr out ++ skipn n stk) s')
          | Some (RBr (S _) _ _ _) | Some RTrap => Some CTrap
          end
        else Some CTrap
    end.

(** The interpreter. Fuel [k] bounds the call depth and the number of loop
    iterations; within one fuel level, recursion is structural on the code. *)
Fixpoint eval_seq (m : module) (k : nat) (is : code)
                  (stk locs : list Z) (s : store) {struct k} : option res :=
  match k with
  | 0 => None
  | S k' =>
    (fix go (is : code) (stk locs : list Z) (s : store) {struct is} : option res :=
      match is with
      | CNil => Some (RNormal stk locs s)
      | CCons i rest =>
        match i with
        | ILocalGet x =>
            match nth_error locs x with
            | Some v => go rest (v :: stk) locs s
            | None => Some RTrap
            end
        | ILocalSet x =>
            match stk with
            | v :: stk' =>
                if x <? length locs then go rest stk' (replace_nth locs x v) s
                else Some RTrap
            | [] => Some RTrap
            end
        | ICall f =>
            match do_call (eval_seq m k') m f stk s with
            | None => None
            | Some CTrap => Some RTrap
            | Some (CDone stk' s') => go rest stk' locs s'
            end
        | IPrim p =>
            match prim p stk locs s with
            | None => Some RTrap
            | Some (stk', locs', s') => go rest stk' locs' s'
            end
        | IBlock ar b =>
            match go b [] locs s with
            | None => None
            | Some (RNormal out l s') | Some (RBr 0 out l s') =>
                go rest (firstn ar out ++ stk) l s'
            | Some (RBr (S n) out l s') => Some (RBr n out l s')
            | Some r => Some r
            end
        | ILoop ar b =>
            match go b [] locs s with
            | None => None
            | Some (RNormal out l s') => go rest (firstn ar out ++ stk) l s'
            | Some (RBr 0 _ l s') => eval_seq m k' (CCons (ILoop ar b) rest) stk l s'
            | Some (RBr (S n) out l s') => Some (RBr n out l s')
            | Some r => Some r
            end
        | IIf ar b1 b2 =>
            match stk with
            | c :: stk' =>
                match go (if Z.eqb c 0 then b2 else b1) [] locs s with
                | None => None
                | Some (RNormal out l s') | Some (RBr 0 out l s') =>
                    go rest (firstn ar out ++ stk') l s'
                | Some (RBr (S n) out l s') => Some (RBr n out l s')
                | Some r => Some r
                end
            | [] => Some RTrap
            end
        | IBr n => Some (RBr n stk locs s)
        | IBrIf n =>
            match stk with
            | c :: stk' =>
                if Z.eqb c 0 then go rest stk' locs s else Some (RBr n stk' locs s)
            | [] => Some RTrap
            end
        | IReturn => Some (RRet stk s)
        | INop => go rest stk locs s
        | IMem p ks =>
            match mem_keys (m_mems m) ks with
            | None => Some RTrap
            | Some ys =>
                match mprim p ys stk locs s with
                | None => Some RTrap
                | Some (stk', locs', s') => go rest stk' locs' s'
                end
            end
        end
      end) is stk locs s
  end.

(** Calling function [f] with fuel [k]. *)
Definition eval_call (m : module) (k : nat) (f : nat) (stk : list Z) (s : store) : option cres :=
  do_call (eval_seq m k) m f stk s.

(** Observational equivalence of two calls. *)
Definition call_equiv (m1 : module) (f1 : nat) (m2 : module) (f2 : nat) : Prop :=
  forall stk s c,
    (exists k, eval_call m1 k f1 stk s = Some c) <->
    (exists k, eval_call m2 k f2 stk s = Some c).

(** One instruction step, with [go] the interpreter for the rest of the
    sequence and for nested bodies, at fuel [S k']. *)
Definition step (m : module) (k' : nat)
                (go : code -> list Z -> list Z -> store -> option res)
                (i : instr) (rest : code) (stk locs : list Z) (s : store) : option res :=
  match i with
  | ILocalGet x =>
      match nth_error locs x with
      | Some v => go rest (v :: stk) locs s
      | None => Some RTrap
      end
  | ILocalSet x =>
      match stk with
      | v :: stk' =>
          if x <? length locs then go rest stk' (replace_nth locs x v) s
          else Some RTrap
      | [] => Some RTrap
      end
  | ICall f =>
      match do_call (eval_seq m k') m f stk s with
      | None => None
      | Some CTrap => Some RTrap
      | Some (CDone stk' s') => go rest stk' locs s'
      end
  | IPrim p =>
      match prim p stk locs s with
      | None => Some RTrap
      | Some (stk', locs', s') => go rest stk' locs' s'
      end
  | IBlock ar b =>
      match go b [] locs s with
      | None => None
      | Some (RNormal out l s') | Some (RBr 0 out l s') =>
          go rest (firstn ar out ++ stk) l s'
      | Some (RBr (S n) out l s') => Some (RBr n out l s')
      | Some r => Some r
      end
  | ILoop ar b =>
      match go b [] locs s with
      | None => None
      | Some (RNormal out l s') => go rest (firstn ar out ++ stk) l s'
      | Some (RBr 0 _ l s') => eval_seq m k' (CCons (ILoop ar b) rest) stk l s'
      | Some (RBr (S n) out l s') => Some (RBr n out l s')
      | Some r => Some r
      end
  | IIf ar b1 b2 =>
      match stk with
      | c :: stk' =>
          match go (if Z.eqb c 0 then b2 else b1) [] locs s with
          | None => None
          | Some (RNormal out l s') | Some (RBr 0 out l s') =>
              go rest (firstn ar out ++ stk') l s'
          | Some (RBr (S n) out l s') => Some (RBr n out l s')
          | Some r => Some r
          end
      | [] => Some RTrap
      end
  | IBr n => Some (RBr n stk locs s)
  | IBrIf n =>
      match stk with
      | c :: stk' =>
          if Z.eqb c 0 then go rest stk' locs s else Some (RBr n stk' locs s)
      | [] => Some RTrap
      end
  | IReturn => Some (RRet stk s)
  | INop => go rest stk locs s
  | IMem p ks =>
      match mem_keys (m_mems m) ks with
      | None => Some RTrap
      | Some ys =>
          match mprim p ys stk locs s with
          | None => Some RTrap
          | Some (stk', locs', s') => go rest stk' locs' s'
          end
      end
  end.

Lemma eval_seq_nil : forall m k stk locs s,
  eval_seq m (S k) CNil stk locs s = Some (RNormal stk locs s).
Proof. reflexivity. Qed.

Lemma eval_seq_cons : forall m k i rest stk locs s,
  eval_seq m (S k) (CCons i rest) stk locs s
  = step m k (eval_seq m (S k)) i rest stk locs s.
Proof. intros; destruct i; reflexivity. Qed.

Ltac split_hyp :=
  repeat match goal with
  | H : match ?x with _ => _ end = Some _ |- _ =>
      let E := fresh "E" in destruct x eqn:E; try discriminate
  end.

(** The generic forward simulation, restricted to code whose calls all
    satisfy [P]. If every call to an [f] with [P f] in [m1] at fuel [k1]
    can be matched by a call to [rho f] in [m2] at fuel [k2 >= k1], then
    every such sequence [c] run in [m1] is matched by [rw_code rho c] run
    in [m2]. *)
Lemma sim_gen_on : forall (P : nat -> Prop) (m1 m2 : module) (rho : nat -> nat),
  m_mems m1 = m_mems m2 ->
  (forall k1 k2, k1 <= k2 ->
     (forall c stk locs s r, calls_c P c ->
        eval_seq m1 k1 c stk locs s = Some r ->
        eval_seq m2 k2 (rw_code rho c) stk locs s = Some r) ->
     forall f stk s cr, P f ->
       do_call (eval_seq m1 k1) m1 f stk s = Some cr ->
       do_call (eval_seq m2 k2) m2 (rho f) stk s = Some cr) ->
  forall k1 k2, k1 <= k2 ->
  forall c stk locs s r, calls_c P c ->
    eval_seq m1 k1 c stk locs s = Some r ->
    eval_seq m2 k2 (rw_code rho c) stk locs s = Some r.
Proof.
  intros P m1 m2 rho Hmem Hcall k1.
  induction k1 as [|k1 IH]; intros k2 Hle c stk locs s r Hc H; [discriminate|].
  destruct k2 as [|k2]; [lia|].
  assert (Hle' : k1 <= k2) by lia.
  specialize (IH k2 Hle').
  revert stk locs s r H. revert c Hc.
  apply (code_mut
    (fun i => forall rest, calls_i P i -> calls_c P rest ->
       (forall stk locs s r,
          eval_seq m1 (S k1) rest stk locs s = Some r ->
          eval_seq m2 (S k2) (rw_code rho rest) stk locs s = Some r) ->
       forall stk locs s r,
         eval_seq m1 (S k1) (CCons i rest) stk locs s = Some r ->
         eval_seq m2 (S k2) (rw_code rho (CCons i rest)) stk locs s = Some r)
    (fun c => calls_c P c -> forall stk locs s r,
       eval_seq m1 (S k1) c stk locs s = Some r ->
       eval_seq m2 (S k2) (rw_code rho c) stk locs s = Some r)).
  - (* local.get *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *;
      split_hyp; eauto.
  - (* local.set *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *;
      split_hyp; eauto.
  - (* call *)
    intros f rest Hf _ Hr stk locs s r H.
    cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (do_call (eval_seq m1 k1) m1 f stk s) as [cr|] eqn:E; [|discriminate].
    rewrite (Hcall _ _ Hle' IH _ _ _ _ Hf E).
    destruct cr; eauto.
  - (* prim *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *;
      split_hyp; eauto.
  - (* block *)
    intros ar b Hb rest Hcb _ Hr stk locs s r H.
    cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m1 (S k1) b [] locs s) as [rb|] eqn:E; [|discriminate].
    rewrite (Hb Hcb _ _ _ _ E).
    destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
  - (* loop *)
    intros ar b Hb rest Hcb Hcr Hr stk locs s r H.
    cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m1 (S k1) b [] locs s) as [rb|] eqn:E; [|discriminate].
    rewrite (Hb Hcb _ _ _ _ E).
    destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
    exact (IH (CCons (ILoop ar b) rest) _ _ _ _ (conj Hcb Hcr) H).
  - (* if *)
    intros ar b1 Hb1 b2 Hb2 rest [Hc1 Hc2] _ Hr stk locs s r H.
    cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|c stk']; [exact H|].
    destruct (Z.eqb c 0).
    + destruct (eval_seq m1 (S k1) b2 [] locs s) as [rb|] eqn:E; [|discriminate].
      rewrite (Hb2 Hc2 _ _ _ _ E).
      destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
    + destruct (eval_seq m1 (S k1) b1 [] locs s) as [rb|] eqn:E; [|discriminate].
      rewrite (Hb1 Hc1 _ _ _ _ E).
      destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
  - (* br *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *; eauto.
  - (* br_if *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *;
      split_hyp; eauto.
  - (* return *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *; eauto.
  - (* nop *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *; eauto.
  - (* mem *)
    intros; cbn [rw_code rw_instr] in *; rewrite eval_seq_cons in *; cbn [step] in *; rewrite ?Hmem in *;
      split_hyp; eauto.
  - (* nil *)
    intros _ stk locs s r H; rewrite eval_seq_nil in *; exact H.
  - (* cons *)
    intros i Hi rest Hrest [Hci Hcr]. exact (Hi rest Hci Hcr (Hrest Hcr)).
Qed.

(** The unrestricted forward simulation: [sim_gen_on] with every call allowed. *)
Lemma sim_gen : forall (m1 m2 : module) (rho : nat -> nat),
  m_mems m1 = m_mems m2 ->
  (forall k1 k2, k1 <= k2 ->
     (forall c stk locs s r,
        eval_seq m1 k1 c stk locs s = Some r ->
        eval_seq m2 k2 (rw_code rho c) stk locs s = Some r) ->
     forall f stk s cr,
       do_call (eval_seq m1 k1) m1 f stk s = Some cr ->
       do_call (eval_seq m2 k2) m2 (rho f) stk s = Some cr) ->
  forall k1 k2, k1 <= k2 ->
  forall c stk locs s r,
    eval_seq m1 k1 c stk locs s = Some r ->
    eval_seq m2 k2 (rw_code rho c) stk locs s = Some r.
Proof.
  intros m1 m2 rho Hmem Hcall k1 k2 Hle c stk locs s r H.
  apply (sim_gen_on (fun _ => True) m1 m2 rho Hmem) with (k1 := k1);
    [|exact Hle|apply calls_true|exact H].
  intros k1' k2' Hle' IH f stk' s' cr _ Hd.
  apply (Hcall k1' k2' Hle'); [|exact Hd].
  intros c' stk'' locs' s'' r' Hc'. exact (IH c' stk'' locs' s'' r' (calls_true c') Hc').
Qed.

Lemma do_call_mono : forall (run1 run2 : code -> list Z -> list Z -> store -> option res)
                            m f stk s cr,
  (forall c st l s0 r, run1 c st l s0 = Some r -> run2 c st l s0 = Some r) ->
  do_call run1 m f stk s = Some cr -> do_call run2 m f stk s = Some cr.
Proof.
  intros run1 run2 m f stk s cr Hrun H. unfold do_call in *.
  destruct (f <? length (m_imports m)); [exact H|].
  destruct (nth_error (m_funcs m) (f - length (m_imports m))) as [d|]; [|exact H].
  destruct (_ <=? _); [|exact H].
  destruct (run1 (f_body d) _ _ s) as [r|] eqn:E; [|discriminate].
  rewrite (Hrun _ _ _ _ _ E). exact H.
Qed.

(** Fuel monotonicity: more fuel never changes a result that was reached. *)
Lemma mono_le : forall m k1 k2, k1 <= k2 ->
  forall c stk locs s r,
    eval_seq m k1 c stk locs s = Some r -> eval_seq m k2 c stk locs s = Some r.
Proof.
  intros m k1 k2 Hle c stk locs s r H.
  rewrite <- (rw_code_id c).
  revert k1 k2 Hle c stk locs s r H.
  apply (sim_gen m m (fun f => f) eq_refl).
  intros k1 k2 _ IH f stk s cr H.
  eapply do_call_mono; [|exact H].
  intros c st l s0 r Hc. rewrite <- (rw_code_id c). exact (IH _ _ _ _ _ Hc).
Qed.

Lemma do_call_mono_le : forall m k1 k2 f stk s cr, k1 <= k2 ->
  do_call (eval_seq m k1) m f stk s = Some cr ->
  do_call (eval_seq m k2) m f stk s = Some cr.
Proof.
  intros. eapply do_call_mono; [|eassumption].
  intros; eapply mono_le; eassumption.
Qed.

Ltac lift E K :=
  match type of E with
  | eval_seq ?m ?k _ _ _ _ = _ =>
      let Hk := fresh "Hk" in
      assert (Hk : k <= K) by lia;
      rewrite (mono_le m k K Hk _ _ _ _ _ E); clear Hk
  end.

(** The generic backward simulation, with fuel existentially quantified,
    restricted to code whose calls all satisfy [P]. *)
Lemma sim_back_gen_on : forall (P : nat -> Prop) (m1 m2 : module) (rho : nat -> nat),
  m_mems m1 = m_mems m2 ->
  (forall k2,
     (forall c stk locs s r, calls_c P c ->
        eval_seq m2 k2 (rw_code rho c) stk locs s = Some r ->
        exists k1, eval_seq m1 k1 c stk locs s = Some r) ->
     forall f stk s cr, P f ->
       do_call (eval_seq m2 k2) m2 (rho f) stk s = Some cr ->
       exists k1, do_call (eval_seq m1 k1) m1 f stk s = Some cr) ->
  forall k2 c stk locs s r, calls_c P c ->
    eval_seq m2 k2 (rw_code rho c) stk locs s = Some r ->
    exists k1, eval_seq m1 k1 c stk locs s = Some r.
Proof.
  intros P m1 m2 rho Hmem Hback k2.
  induction k2 as [|k2 IH]; intros c stk locs s r Hc H; [discriminate|].
  revert stk locs s r H. revert c Hc.
  apply (code_mut
    (fun i => forall rest, calls_i P i -> calls_c P rest ->
       (forall stk locs s r,
          eval_seq m2 (S k2) (rw_code rho rest) stk locs s = Some r ->
          exists k1, eval_seq m1 k1 rest stk locs s = Some r) ->
       forall stk locs s r,
         eval_seq m2 (S k2) (rw_code rho (CCons i rest)) stk locs s = Some r ->
         exists k1, eval_seq m1 k1 (CCons i rest) stk locs s = Some r)
    (fun c => calls_c P c -> forall stk locs s r,
       eval_seq m2 (S k2) (rw_code rho c) stk locs s = Some r ->
       exists k1, eval_seq m1 k1 c stk locs s = Some r));
  cbn [rw_code rw_instr].
  - (* local.get *)
    intros n rest _ _ H stk locs s r H0.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (nth_error locs n) eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* local.set *)
    intros n rest _ _ H stk locs s r H0.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H0|].
    destruct (n <? length locs) eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* call *)
    intros n rest Hn _ H stk locs s r H0.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (do_call (eval_seq m2 k2) m2 (rho n) stk s) as [cr|] eqn:E; [|discriminate].
    destruct (Hback k2 IH _ _ _ _ Hn E) as [k1 E1].
    destruct cr as [stk' s'|].
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S (k1 + k3)).
      rewrite eval_seq_cons; cbn [step].
      rewrite (do_call_mono_le _ k1 (k1 + k3) _ _ _ _ ltac:(lia) E1).
      lift E3 (S (k1 + k3)); reflexivity.
    + exists (S k1). rewrite eval_seq_cons; cbn [step]. rewrite E1. exact H0.
  - (* prim *)
    intros n rest _ _ H stk locs s r H0.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (prim n stk locs s) as [[[stk' locs'] s']|] eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* block *)
    intros n c H rest Hcb _ H0 stk locs s r H1. specialize (H Hcb).
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m2 (S k2) (rw_code rho c) [] locs s) as [rb|] eqn:E; [|discriminate].
    destruct (H _ _ _ _ E) as [kb Eb].
    destruct rb as [out l s'|[|n'] out l s'|out s'|].
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
  - (* loop *)
    intros n c H rest Hcb Hcr H0 stk locs s r H1. specialize (H Hcb).
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m2 (S k2) (rw_code rho c) [] locs s) as [rb|] eqn:E; [|discriminate].
    destruct (H _ _ _ _ E) as [kb Eb].
    destruct rb as [out l s'|[|n'] out l s'|out s'|].
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + change (eval_seq m2 k2 (rw_code rho (CCons (ILoop n c) rest)) stk l s' = Some r) in H1.
      destruct (IH (CCons (ILoop n c) rest) _ _ _ _ (conj Hcb Hcr) H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (kb + k3). reflexivity.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
  - (* if *)
    intros n c H c0 H0 rest [Hc1 Hc2] _ H1 stk locs s r H2. specialize (H Hc1). specialize (H0 Hc2).
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H2|].
    destruct (Z.eqb v 0) eqn:Ez.
    + destruct (eval_seq m2 (S k2) (rw_code rho c0) [] locs s) as [rb|] eqn:E; [|discriminate].
      destruct (H0 _ _ _ _ E) as [kb Eb].
      destruct rb as [out l s'|[|n'] out l s'|out s'|].
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
    + destruct (eval_seq m2 (S k2) (rw_code rho c) [] locs s) as [rb|] eqn:E; [|discriminate].
      destruct (H _ _ _ _ E) as [kb Eb].
      destruct rb as [out l s'|[|n'] out l s'|out s'|].
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
  - (* br *)
    intros n rest _ _ H stk locs s r H0.
    exists 1. rewrite eval_seq_cons in *; cbn [step] in *. exact H0.
  - (* br_if *)
    intros n rest _ _ H stk locs s r H0.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H0|].
    destruct (Z.eqb v 0) eqn:Ez.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ez. exact H0.
  - (* return *)
    intros rest _ _ H stk locs s r H0.
    exists 1. rewrite eval_seq_cons in *; cbn [step] in *. exact H0.
  - (* nop *)
    intros rest _ _ H stk locs s r H0.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
    rewrite eval_seq_cons; cbn [step]. lift E3 (S k3); reflexivity.
  - (* mem *)
    intros n l rest _ _ H stk locs s r H0.
    rewrite eval_seq_cons in *; cbn [step] in *.
    rewrite <- Hmem in H0.
    destruct (mem_keys (m_mems m1) l) as [ys|] eqn:Ek.
    + destruct (mprim n ys stk locs s) as [[[stk' locs'] s']|] eqn:E.
      * destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
        rewrite eval_seq_cons; cbn [step]; rewrite Ek, E. lift E3 (S k3); reflexivity.
      * exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ek, E. exact H0.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ek. exact H0.
  - (* nil *)
    intros _ stk locs s r H.
    exists 1. exact H.
  - (* cons *)
    intros i Hi rest Hrest [Hci Hcr]. exact (Hi rest Hci Hcr (Hrest Hcr)).
Qed.

(** The unrestricted backward simulation: [sim_back_gen_on] with every call
    allowed. *)
Lemma sim_back_gen : forall (m1 m2 : module) (rho : nat -> nat),
  m_mems m1 = m_mems m2 ->
  (forall k2,
     (forall c stk locs s r,
        eval_seq m2 k2 (rw_code rho c) stk locs s = Some r ->
        exists k1, eval_seq m1 k1 c stk locs s = Some r) ->
     forall f stk s cr,
       do_call (eval_seq m2 k2) m2 (rho f) stk s = Some cr ->
       exists k1, do_call (eval_seq m1 k1) m1 f stk s = Some cr) ->
  forall k2 c stk locs s r,
    eval_seq m2 k2 (rw_code rho c) stk locs s = Some r ->
    exists k1, eval_seq m1 k1 c stk locs s = Some r.
Proof.
  intros m1 m2 rho Hmem Hback k2 c stk locs s r H.
  apply (sim_back_gen_on (fun _ => True) m1 m2 rho Hmem) with (k2 := k2);
    [|apply calls_true|exact H].
  intros k2' IH f stk' s' cr _ Hd.
  apply (Hback k2'); [|exact Hd].
  intros c' stk'' locs' s'' r' Hc'. exact (IH c' stk'' locs' s'' r' (calls_true c') Hc').
Qed.

(** *** Lemmas about calls *)

Lemma skipn_nth : forall (l : list Z) j v,
  nth_error l j = Some v -> skipn j l = v :: skipn (S j) l.
Proof.
  induction l as [|x l IH]; intros [|j] v H; cbn in *; try discriminate.
  - congruence.
  - apply IH; exact H.
Qed.

Lemma gets_run : forall m k n j tail stk locs s,
  j + n <= length locs ->
  eval_seq m (S k) (gets j n tail) stk locs s
  = eval_seq m (S k) tail (rev (firstn n (skipn j locs)) ++ stk) locs s.
Proof.
  intros m k n; induction n as [|n IH]; intros j tail stk locs s Hlen; [reflexivity|].
  cbn [gets]. rewrite eval_seq_cons; cbn [step].
  destruct (nth_error locs j) as [v|] eqn:E.
  - rewrite IH by lia. rewrite (skipn_nth _ _ _ E). cbn [firstn rev].
    rewrite <- app_assoc. reflexivity.
  - apply nth_error_None in E. lia.
Qed.

(** A call with [n] parameters only reads the top [n] stack values. *)
Lemma do_call_frame : forall run m t sg stk s,
  sig_of m t = Some sg ->
  length (sig_params sg) <= length stk ->
  do_call run m t stk s =
  match do_call run m t (firstn (length (sig_params sg)) stk) s with
  | Some (CDone x s') =>
      Some (CDone (firstn (length (sig_results sg)) x ++ skipn (length (sig_params sg)) stk) s')
  | o => o
  end.
Proof.
  intros run m t sg stk s Hsig Hlen.
  set (n := length (sig_params sg)) in *.
  assert (Hfl : length (firstn n stk) = n) by (rewrite length_firstn; lia).
  assert (Hff : firstn n (firstn n stk) = firstn n stk) by (rewrite firstn_firstn; f_equal; lia).
  assert (Hsk : skipn n (firstn n stk) = []) by (apply skipn_all2; lia).
  unfold sig_of, do_call in *.
  destruct (t <? length (m_imports m)).
  - destruct (nth_error (m_imports m) t) as [im|]; [|discriminate].
    cbn in Hsig. injection Hsig as <-. fold n.
    destruct (n <=? length stk) eqn:E1; [|apply Nat.leb_gt in E1; lia].
    destruct (n <=? length (firstn n stk)) eqn:E2; [|apply Nat.leb_gt in E2; lia].
    rewrite Hff, Hsk.
    destruct (host (imp_key im) (firstn n stk) s) as [[out s']|]; [|reflexivity].
    rewrite app_nil_r, firstn_firstn, Nat.min_id. reflexivity.
  - destruct (nth_error (m_funcs m) (t - length (m_imports m))) as [d|]; [|discriminate].
    cbn in Hsig. injection Hsig as <-. fold n.
    destruct (n <=? length stk) eqn:E1; [|apply Nat.leb_gt in E1; lia].
    destruct (n <=? length (firstn n stk)) eqn:E2; [|apply Nat.leb_gt in E2; lia].
    rewrite Hff, Hsk.
    destruct (run _ _ _ s) as [[out l s'|[|?] out l s'|out s'|]|]; try reflexivity;
      rewrite app_nil_r, firstn_firstn, Nat.min_id; reflexivity.
Qed.

Lemma do_call_underflow : forall run m t sg stk s,
  sig_of m t = Some sg ->
  length stk < length (sig_params sg) ->
  do_call run m t stk s = Some CTrap.
Proof.
  intros run m t sg stk s Hsig Hlen. unfold sig_of, do_call in *.
  destruct (t <? length (m_imports m)).
  - destruct (nth_error (m_imports m) t) as [im|]; [|discriminate].
    cbn in Hsig. injection Hsig as <-.
    destruct (_ <=? _) eqn:E; [apply Nat.leb_le in E; lia|reflexivity].
  - destruct (nth_error (m_funcs m) (t - length (m_imports m))) as [d|]; [|discriminate].
    cbn in Hsig. injection Hsig as <-.
    destruct (_ <=? _) eqn:E; [apply Nat.leb_le in E; lia|reflexivity].
Qed.

(** One adapter hop costs exactly one unit of fuel and changes nothing else. *)
Lemma adapter_unfold : forall m a d t k stk s,
  length (m_imports m) <= a ->
  nth_error (m_funcs m) (a - length (m_imports m)) = Some d ->
  trivial_adapter d t ->
  sig_of m t = Some (f_sig d) ->
  eval_call m (S k) a stk s = eval_call m k t stk s.
Proof.
  intros m a d t k stk s Himp Hd [Hloc Hbody] Hsig.
  unfold eval_call.
  set (n := length (sig_params (f_sig d))).
  destruct (Nat.le_gt_cases n (length stk)) as [Hle|Hgt].
  - rewrite (do_call_frame _ _ t _ stk s Hsig Hle). fold n.
    unfold do_call at 1.
    destruct (a <? length (m_imports m)) eqn:Ea; [apply Nat.ltb_lt in Ea; lia|].
    rewrite Hd. fold n.
    destruct (n <=? length stk) eqn:E1; [|apply Nat.leb_gt in E1; lia].
    rewrite Hbody, Hloc. cbn [repeat]. rewrite app_nil_r. fold n.
    rewrite gets_run by (rewrite length_rev, length_firstn; lia).
    cbn [skipn].
    rewrite firstn_all2 by (rewrite length_rev, length_firstn; lia).
    rewrite rev_involutive, app_nil_r.
    rewrite eval_seq_cons; cbn [step].
    destruct (do_call (eval_seq m k) m t (firstn n stk) s) as [[x s'|]|]; reflexivity.
  - rewrite (do_call_underflow _ _ t _ stk s Hsig Hgt).
    unfold do_call.
    destruct (a <? length (m_imports m)) eqn:Ea; [apply Nat.ltb_lt in Ea; lia|].
    rewrite Hd. fold n.
    destruct (n <=? length stk) eqn:E1; [apply Nat.leb_le in E1; lia|reflexivity].
Qed.

Lemma eval_call_mono_le : forall m k1 k2 f stk s c, k1 <= k2 ->
  eval_call m k1 f stk s = Some c -> eval_call m k2 f stk s = Some c.
Proof. intros; eapply do_call_mono_le; eassumption. Qed.

(** *** Adapter chains

    [chain m f g]: following trivial adapters from [f] reaches [g]. Every
    hop requires the target to have the adapter's signature, as the
    [retain] in [find_adapters] checks. *)
Inductive chain (m : module) : nat -> nat -> Prop :=
  | chain_refl : forall f, chain m f f
  | chain_step : forall a d t g,
      length (m_imports m) <= a ->
      nth_error (m_funcs m) (a - length (m_imports m)) = Some d ->
      trivial_adapter d t ->
      sig_of m t = Some (f_sig d) ->
      chain m t g ->
      chain m a g.

Lemma chain_fwd : forall m f g, chain m f g ->
  forall k stk s c, eval_call m k f stk s = Some c -> eval_call m k g stk s = Some c.
Proof.
  induction 1 as [f|a d t g Himp Hd Htriv Hsig _ IH]; intros k stk s c H; [exact H|].
  apply IH.
  rewrite <- (adapter_unfold m a d t k stk s Himp Hd Htriv Hsig).
  eapply eval_call_mono_le; [|exact H]. lia.
Qed.

Lemma chain_back : forall m f g, chain m f g ->
  forall k stk s c, eval_call m k g stk s = Some c ->
  exists k', eval_call m k' f stk s = Some c.
Proof.
  induction 1 as [f|a d t g Himp Hd Htriv Hsig _ IH]; intros k stk s c H; [exists k; exact H|].
  destruct (IH k stk s c H) as [k' Hk'].
  exists (S k'). rewrite (adapter_unfold m a d t k' stk s Himp Hd Htriv Hsig). exact Hk'.
Qed.

(** A trivial adapter is observationally equivalent to its target. *)
Theorem trivial_adapter_equiv : forall m a d t,
  length (m_imports m) <= a ->
  nth_error (m_funcs m) (a - length (m_imports m)) = Some d ->
  trivial_adapter d t ->
  sig_of m t = Some (f_sig d) ->
  call_equiv m a m t.
Proof.
  intros m a d t Himp Hd Htriv Hsig stk s c.
  assert (Hc : chain m a t) by (eapply chain_step; eauto using chain_refl).
  split; intros [k Hk].
  - exists k. exact (chain_fwd _ _ _ Hc _ _ _ _ Hk).
  - exact (chain_back _ _ _ Hc _ _ _ _ Hk).
Qed.

(** *** Devirtualization *)

Lemma do_call_rw : forall rho m k1 k2 g stk s cr,
  (forall c stk locs s r,
     eval_seq m k1 c stk locs s = Some r ->
     eval_seq (rw_module rho m) k2 (rw_code rho c) stk locs s = Some r) ->
  do_call (eval_seq m k1) m g stk s = Some cr ->
  do_call (eval_seq (rw_module rho m) k2) (rw_module rho m) g stk s = Some cr.
Proof.
  intros rho m k1 k2 g stk s cr Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs rw_module] in *.
  destruct (g <? length (m_imports m)); [exact H|].
  rewrite nth_error_map.
  destruct (nth_error (m_funcs m) (g - length (m_imports m))) as [d|]; [|exact H].
  cbn [option_map rw_func f_sig f_locals f_body].
  destruct (_ <=? _); [|exact H].
  destruct (eval_seq m k1 (f_body d) _ _ s) as [r|] eqn:E; [|discriminate].
  rewrite (Hsim _ _ _ _ _ E). exact H.
Qed.

Lemma do_call_rw_back : forall rho m k2 g stk s cr,
  (forall c stk locs s r,
     eval_seq (rw_module rho m) k2 (rw_code rho c) stk locs s = Some r ->
     exists k1, eval_seq m k1 c stk locs s = Some r) ->
  do_call (eval_seq (rw_module rho m) k2) (rw_module rho m) g stk s = Some cr ->
  exists k1, do_call (eval_seq m k1) m g stk s = Some cr.
Proof.
  intros rho m k2 g stk s cr Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs rw_module] in *.
  destruct (g <? length (m_imports m)); [exists 0; exact H|].
  rewrite nth_error_map in H.
  destruct (nth_error (m_funcs m) (g - length (m_imports m))) as [d|]; [|exists 0; exact H].
  cbn [option_map rw_func f_sig f_locals f_body] in H.
  destruct (_ <=? _); [|exists 0; exact H].
  destruct (eval_seq (rw_module rho m) k2 (rw_code rho (f_body d)) _ _ s) as [r|] eqn:E;
    [|discriminate].
  destruct (Hsim _ _ _ _ _ E) as [k1 E1].
  exists k1. rewrite E1. exact H.
Qed.

(** Devirtualization is correct. If [rho] sends every function index to
    one reachable from it along trivial adapters, then rewriting every
    [call f] to [call (rho f)] in every body leaves every function
    observationally unchanged, imports included, for any host and any
    semantics of the unmodelled instructions. *)
Theorem devirtualize_correct : forall m rho,
  (forall f, chain m f (rho f)) ->
  forall f, call_equiv m f (rw_module rho m) f.
Proof.
  intros m rho Hrho f stk s c. split.
  - intros [k Hk]. exists k.
    eapply do_call_rw; [|exact Hk].
    apply sim_gen; [reflexivity| |lia].
    intros k1 k2 Hle IH g stk' s' cr H.
    eapply do_call_rw; [exact IH|].
    exact (chain_fwd _ _ _ (Hrho g) _ _ _ _ H).
  - intros [k Hk].
    eapply do_call_rw_back; [|exact Hk].
    apply sim_back_gen; [reflexivity|].
    intros k2 IH g stk' s' cr H.
    destruct (do_call_rw_back _ _ _ _ _ _ _ IH H) as [k1 E1].
    exact (chain_back _ _ _ (Hrho g) _ _ _ _ E1).
Qed.

(** *** Trivial-call elimination *)

Lemma nop_run : forall m k c stk locs s, nop_code c ->
  eval_seq m (S k) c stk locs s = Some (RNormal stk locs s).
Proof.
  intros m k c stk locs s Hc; induction Hc as [|c _ IH]; [reflexivity|].
  rewrite eval_seq_cons; cbn [step]. exact IH.
Qed.

(** A call to a trivial function returns the caller's stack and store
    unchanged, given one unit of fuel for the callee's body. *)
Lemma trivial_call_id : forall m f k stk s, trivial_fn m f ->
  do_call (eval_seq m (S k)) m f stk s = Some (CDone stk s).
Proof.
  intros m f k stk s [Himp [d [Hd [Hp [Hr Hb]]]]].
  unfold do_call.
  destruct (f <? length (m_imports m)) eqn:Ef; [apply Nat.ltb_lt in Ef; lia|].
  rewrite Hd, Hp, Hr. cbn [length Nat.leb firstn skipn rev app].
  rewrite (nop_run _ _ _ _ _ _ Hb). reflexivity.
Qed.

Lemma trivial_call_inv : forall m f k stk s cr, trivial_fn m f ->
  do_call (eval_seq m k) m f stk s = Some cr -> cr = CDone stk s.
Proof.
  intros m f [|k] stk s cr Ht H.
  - destruct Ht as [Himp [d [Hd [Hp [Hr Hb]]]]].
    unfold do_call in H.
    destruct (f <? length (m_imports m)) eqn:Ef; [apply Nat.ltb_lt in Ef; lia|].
    rewrite Hd, Hp in H. cbn in H. discriminate.
  - rewrite (trivial_call_id _ _ _ _ _ Ht) in H. congruence.
Qed.

Lemma do_call_del : forall T m k1 k2 g stk s cr,
  (forall c stk locs s r,
     eval_seq m k1 c stk locs s = Some r ->
     eval_seq (del_module T m) k2 (del_code T c) stk locs s = Some r) ->
  do_call (eval_seq m k1) m g stk s = Some cr ->
  do_call (eval_seq (del_module T m) k2) (del_module T m) g stk s = Some cr.
Proof.
  intros T m k1 k2 g stk s cr Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs del_module] in *.
  destruct (g <? length (m_imports m)); [exact H|].
  rewrite nth_error_map.
  destruct (nth_error (m_funcs m) (g - length (m_imports m))) as [d|]; [|exact H].
  cbn [option_map del_func f_sig f_locals f_body].
  destruct (_ <=? _); [|exact H].
  destruct (eval_seq m k1 (f_body d) _ _ s) as [r|] eqn:E; [|discriminate].
  rewrite (Hsim _ _ _ _ _ E). exact H.
Qed.

Lemma do_call_del_back : forall T m k2 g stk s cr,
  (forall c stk locs s r,
     eval_seq (del_module T m) k2 (del_code T c) stk locs s = Some r ->
     exists k1, eval_seq m k1 c stk locs s = Some r) ->
  do_call (eval_seq (del_module T m) k2) (del_module T m) g stk s = Some cr ->
  exists k1, do_call (eval_seq m k1) m g stk s = Some cr.
Proof.
  intros T m k2 g stk s cr Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs del_module] in *.
  destruct (g <? length (m_imports m)); [exists 0; exact H|].
  rewrite nth_error_map in H.
  destruct (nth_error (m_funcs m) (g - length (m_imports m))) as [d|]; [|exists 0; exact H].
  cbn [option_map del_func f_sig f_locals f_body] in H.
  destruct (_ <=? _); [|exists 0; exact H].
  destruct (eval_seq (del_module T m) k2 (del_code T (f_body d)) _ _ s) as [r|] eqn:E;
    [|discriminate].
  destruct (Hsim _ _ _ _ _ E) as [k1 E1].
  exists k1. rewrite E1. exact H.
Qed.

(** Forward simulation: the rewritten module needs no more fuel. *)
Lemma sim_del : forall T m, (forall f, T f = true -> trivial_fn m f) ->
  forall k1 k2, k1 <= k2 ->
  forall c stk locs s r,
    eval_seq m k1 c stk locs s = Some r ->
    eval_seq (del_module T m) k2 (del_code T c) stk locs s = Some r.
Proof.
  intros T m HT k1.
  induction k1 as [|k1 IH]; intros k2 Hle c stk locs s r H; [discriminate|].
  destruct k2 as [|k2]; [lia|].
  assert (Hle' : k1 <= k2) by lia.
  specialize (IH k2 Hle').
  revert c stk locs s r H.
  apply (code_mut
    (fun i => forall rest,
       (forall stk locs s r,
          eval_seq m (S k1) rest stk locs s = Some r ->
          eval_seq (del_module T m) (S k2) (del_code T rest) stk locs s = Some r) ->
       forall stk locs s r,
         eval_seq m (S k1) (CCons i rest) stk locs s = Some r ->
         eval_seq (del_module T m) (S k2) (del_code T (CCons i rest)) stk locs s = Some r)
    (fun c => forall stk locs s r,
       eval_seq m (S k1) c stk locs s = Some r ->
       eval_seq (del_module T m) (S k2) (del_code T c) stk locs s = Some r));
  cbn [del_code del_instr];
  try (intros; rewrite eval_seq_cons in *; cbn [step m_mems del_module] in *;
       split_hyp; eauto; fail).
  - (* call *)
    intros f rest Hr stk locs s r H.
    rewrite eval_seq_cons in H; cbn [step] in H.
    destruct (do_call (eval_seq m k1) m f stk s) as [cr|] eqn:E; [|discriminate].
    destruct (T f) eqn:ET.
    + (* deleted: the call returned the stack and store unchanged *)
      rewrite (trivial_call_inv _ _ _ _ _ _ (HT f ET) E) in H. exact (Hr _ _ _ _ H).
    + rewrite eval_seq_cons; cbn [step].
      rewrite (do_call_del _ _ _ _ _ _ _ _ IH E).
      destruct cr; eauto.
  - (* block *)
    intros ar b Hb rest Hr stk locs s r H.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m (S k1) b [] locs s) as [rb|] eqn:E; [|discriminate].
    rewrite (Hb _ _ _ _ E).
    destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
  - (* loop *)
    intros ar b Hb rest Hr stk locs s r H.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m (S k1) b [] locs s) as [rb|] eqn:E; [|discriminate].
    rewrite (Hb _ _ _ _ E).
    destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
    apply IH in H; exact H.
  - (* if *)
    intros ar b1 Hb1 b2 Hb2 rest Hr stk locs s r H.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|c stk']; [exact H|].
    destruct (Z.eqb c 0).
    + destruct (eval_seq m (S k1) b2 [] locs s) as [rb|] eqn:E; [|discriminate].
      rewrite (Hb2 _ _ _ _ E).
      destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
    + destruct (eval_seq m (S k1) b1 [] locs s) as [rb|] eqn:E; [|discriminate].
      rewrite (Hb1 _ _ _ _ E).
      destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
  - (* nil *)
    intros; rewrite eval_seq_nil in *; exact H.
Qed.

(** Backward simulation: each deleted call costs the original one more
    unit of fuel, so the fuel is existential. *)
Lemma sim_del_back : forall T m, (forall f, T f = true -> trivial_fn m f) ->
  forall k2 c stk locs s r,
    eval_seq (del_module T m) k2 (del_code T c) stk locs s = Some r ->
    exists k1, eval_seq m k1 c stk locs s = Some r.
Proof.
  intros T m HT k2.
  induction k2 as [|k2 IH]; intros c stk locs s r H; [discriminate|].
  revert c stk locs s r H.
  apply (code_mut
    (fun i => forall rest,
       (forall stk locs s r,
          eval_seq (del_module T m) (S k2) (del_code T rest) stk locs s = Some r ->
          exists k1, eval_seq m k1 rest stk locs s = Some r) ->
       forall stk locs s r,
         eval_seq (del_module T m) (S k2) (del_code T (CCons i rest)) stk locs s = Some r ->
         exists k1, eval_seq m k1 (CCons i rest) stk locs s = Some r)
    (fun c => forall stk locs s r,
       eval_seq (del_module T m) (S k2) (del_code T c) stk locs s = Some r ->
       exists k1, eval_seq m k1 c stk locs s = Some r));
  cbn [del_code del_instr]; intros.
  - (* local.get *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (nth_error locs n) eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* local.set *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H0|].
    destruct (n <? length locs) eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* call *)
    destruct (T n) eqn:ET.
    + (* deleted: give the callee one unit of fuel for its nop body *)
      destruct (H _ _ _ _ H0) as [k3 E3]. exists (S (S k3)).
      rewrite eval_seq_cons; cbn [step].
      rewrite (trivial_call_id _ _ k3 _ _ (HT n ET)).
      lift E3 (S (S k3)); reflexivity.
    + rewrite eval_seq_cons in *; cbn [step] in *.
      destruct (do_call (eval_seq (del_module T m) k2) (del_module T m) n stk s) as [cr|] eqn:E;
        [|discriminate].
      destruct (do_call_del_back _ _ _ _ _ _ _ IH E) as [k1 E1].
      destruct cr as [stk' s'|].
      * destruct (H _ _ _ _ H0) as [k3 E3]. exists (S (k1 + k3)).
        rewrite eval_seq_cons; cbn [step].
        rewrite (do_call_mono_le _ k1 (k1 + k3) _ _ _ _ ltac:(lia) E1).
        lift E3 (S (k1 + k3)); reflexivity.
      * exists (S k1). rewrite eval_seq_cons; cbn [step]. rewrite E1. exact H0.
  - (* prim *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (prim n stk locs s) as [[[stk' locs'] s']|] eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* block *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq (del_module T m) (S k2) (del_code T c) [] locs s) as [rb|] eqn:E; [|discriminate].
    destruct (H _ _ _ _ E) as [kb Eb].
    destruct rb as [out l s'|[|n'] out l s'|out s'|].
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
  - (* loop *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq (del_module T m) (S k2) (del_code T c) [] locs s) as [rb|] eqn:E; [|discriminate].
    destruct (H _ _ _ _ E) as [kb Eb].
    destruct rb as [out l s'|[|n'] out l s'|out s'|].
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + change (eval_seq (del_module T m) k2 (del_code T (CCons (ILoop n c) rest)) stk l s' = Some r) in H1.
      destruct (IH _ _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (kb + k3). reflexivity.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
  - (* if *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H2|].
    destruct (Z.eqb v 0) eqn:Ez.
    + destruct (eval_seq (del_module T m) (S k2) (del_code T c0) [] locs s) as [rb|] eqn:E; [|discriminate].
      destruct (H0 _ _ _ _ E) as [kb Eb].
      destruct rb as [out l s'|[|n'] out l s'|out s'|].
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
    + destruct (eval_seq (del_module T m) (S k2) (del_code T c) [] locs s) as [rb|] eqn:E; [|discriminate].
      destruct (H _ _ _ _ E) as [kb Eb].
      destruct rb as [out l s'|[|n'] out l s'|out s'|].
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
  - (* br *)
    exists 1. rewrite eval_seq_cons in *; cbn [step] in *. exact H0.
  - (* br_if *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H0|].
    destruct (Z.eqb v 0) eqn:Ez.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ez. exact H0.
  - (* return *)
    exists 1. rewrite eval_seq_cons in *; cbn [step] in *. exact H0.
  - (* nop *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
    rewrite eval_seq_cons; cbn [step]. lift E3 (S k3); reflexivity.
  - (* mem *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    cbn [m_mems del_module] in H0.
    destruct (mem_keys (m_mems m) l) as [ys|] eqn:Ek.
    + destruct (mprim n ys stk locs s) as [[[stk' locs'] s']|] eqn:E.
      * destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
        rewrite eval_seq_cons; cbn [step]; rewrite Ek, E. lift E3 (S k3); reflexivity.
      * exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ek, E. exact H0.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ek. exact H0.
  - (* nil *)
    exists 1. exact H.
  - (* cons *)
    apply H; assumption.
Qed.

(** Trivial-call elimination is correct. If every function in [T] is a
    defined [() -> ()] function whose body is [nop]s only, then deleting
    every [call f] with [f] in [T] from every body leaves every function
    observationally unchanged, for any host and any semantics of the
    unmodelled instructions. *)
Theorem trivial_call_elim_correct : forall m T,
  (forall f, T f = true -> trivial_fn m f) ->
  forall f, call_equiv m f (del_module T m) f.
Proof.
  intros m T HT f stk s c. split.
  - intros [k Hk]. exists k.
    eapply do_call_del; [|exact Hk].
    intros; eapply sim_del; eauto.
  - intros [k Hk].
    eapply do_call_del_back; [|exact Hk].
    intros; eapply sim_del_back; eauto.
Qed.

(** *** Import deduplication *)

Lemma do_call_dedup : forall rho imps' m k1 k2 f stk s cr,
  dedup_ok m imps' rho ->
  (forall c stk locs s r,
     eval_seq m k1 c stk locs s = Some r ->
     eval_seq (dedup_module rho imps' m) k2 (rw_code rho c) stk locs s = Some r) ->
  do_call (eval_seq m k1) m f stk s = Some cr ->
  do_call (eval_seq (dedup_module rho imps' m) k2) (dedup_module rho imps' m) (rho f) stk s
  = Some cr.
Proof.
  intros rho imps' m k1 k2 f stk s cr [Himp Hdef] Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs dedup_module] in *.
  destruct (f <? length (m_imports m)) eqn:Ef.
  - apply Nat.ltb_lt in Ef. destruct (Himp f Ef) as [Hlt Hnth].
    apply Nat.ltb_lt in Hlt. rewrite Hlt, Hnth. exact H.
  - apply Nat.ltb_ge in Ef. rewrite (Hdef f Ef).
    destruct (f - length (m_imports m) + length imps' <? length imps') eqn:E';
      [apply Nat.ltb_lt in E'; lia|].
    replace (f - length (m_imports m) + length imps' - length imps')
      with (f - length (m_imports m)) by lia.
    rewrite nth_error_map.
    destruct (nth_error (m_funcs m) (f - length (m_imports m))) as [d|]; [|exact H].
    cbn [option_map rw_func f_sig f_locals f_body].
    destruct (_ <=? _); [|exact H].
    destruct (eval_seq m k1 (f_body d) _ _ s) as [r|] eqn:E; [|discriminate].
    rewrite (Hsim _ _ _ _ _ E). exact H.
Qed.

Lemma do_call_dedup_back : forall rho imps' m k2 f stk s cr,
  dedup_ok m imps' rho ->
  (forall c stk locs s r,
     eval_seq (dedup_module rho imps' m) k2 (rw_code rho c) stk locs s = Some r ->
     exists k1, eval_seq m k1 c stk locs s = Some r) ->
  do_call (eval_seq (dedup_module rho imps' m) k2) (dedup_module rho imps' m) (rho f) stk s
  = Some cr ->
  exists k1, do_call (eval_seq m k1) m f stk s = Some cr.
Proof.
  intros rho imps' m k2 f stk s cr [Himp Hdef] Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs dedup_module] in *.
  destruct (f <? length (m_imports m)) eqn:Ef.
  - apply Nat.ltb_lt in Ef. destruct (Himp f Ef) as [Hlt Hnth].
    apply Nat.ltb_lt in Hlt. rewrite Hlt, Hnth in H. exists 0. exact H.
  - apply Nat.ltb_ge in Ef. rewrite (Hdef f Ef) in H.
    destruct (f - length (m_imports m) + length imps' <? length imps') eqn:E';
      [apply Nat.ltb_lt in E'; lia|].
    replace (f - length (m_imports m) + length imps' - length imps')
      with (f - length (m_imports m)) in H by lia.
    rewrite nth_error_map in H.
    destruct (nth_error (m_funcs m) (f - length (m_imports m))) as [d|]; [|exists 0; exact H].
    cbn [option_map rw_func f_sig f_locals f_body] in H.
    destruct (_ <=? _); [|exists 0; exact H].
    destruct (eval_seq (dedup_module rho imps' m) k2 (rw_code rho (f_body d)) _ _ s)
      as [r|] eqn:E; [|discriminate].
    destruct (Hsim _ _ _ _ _ E) as [k1 E1].
    exists k1. rewrite E1. exact H.
Qed.

(** Import deduplication is correct. If [dedup_ok m imps' rho], then
    function [f] of [m] and function [rho f] of the deduplicated module
    are observationally equivalent, imports and defined functions alike,
    for any host and any semantics of the unmodelled instructions. Exports
    and the start function are remapped by [rho] in Rust, so this is the
    statement about every exported function. The host binds by key; see
    [import]. *)
Theorem import_dedup_correct : forall m imps' rho,
  dedup_ok m imps' rho ->
  forall f, call_equiv m f (dedup_module rho imps' m) (rho f).
Proof.
  intros m imps' rho Hok f stk s c. split.
  - intros [k Hk]. exists k.
    eapply do_call_dedup; [exact Hok| |exact Hk].
    apply sim_gen; [reflexivity| |lia].
    intros k1 k2 _ IH g stk' s' cr H.
    exact (do_call_dedup _ _ _ _ _ _ _ _ _ Hok IH H).
  - intros [k Hk].
    eapply do_call_dedup_back; [exact Hok| |exact Hk].
    apply sim_back_gen; [reflexivity|].
    intros k2 IH g stk' s' cr H.
    exact (do_call_dedup_back _ _ _ _ _ _ _ _ Hok IH H).
Qed.

(** *** Memory import deduplication *)

Lemma mem_keys_map : forall m mems' sg, mdedup_ok m mems' sg ->
  forall ks, mem_keys mems' (map sg ks) = mem_keys (m_mems m) ks.
Proof.
  intros m mems' sg Hok ks; induction ks as [|k ks IH]; cbn; [reflexivity|].
  rewrite Hok, IH. reflexivity.
Qed.

Lemma do_call_mdedup : forall sg mems' m k1 k2 f stk s cr,
  (forall c stk locs s r,
     eval_seq m k1 c stk locs s = Some r ->
     eval_seq (mdedup_module sg mems' m) k2 (mrw_code sg c) stk locs s = Some r) ->
  do_call (eval_seq m k1) m f stk s = Some cr ->
  do_call (eval_seq (mdedup_module sg mems' m) k2) (mdedup_module sg mems' m) f stk s = Some cr.
Proof.
  intros sg mems' m k1 k2 f stk s cr Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs mdedup_module] in *.
  destruct (f <? length (m_imports m)); [exact H|].
  rewrite nth_error_map.
  destruct (nth_error (m_funcs m) (f - length (m_imports m))) as [d|]; [|exact H].
  cbn [option_map mrw_func f_sig f_locals f_body].
  destruct (_ <=? _); [|exact H].
  destruct (eval_seq m k1 (f_body d) _ _ s) as [r|] eqn:E; [|discriminate].
  rewrite (Hsim _ _ _ _ _ E). exact H.
Qed.

Lemma do_call_mdedup_back : forall sg mems' m k2 f stk s cr,
  (forall c stk locs s r,
     eval_seq (mdedup_module sg mems' m) k2 (mrw_code sg c) stk locs s = Some r ->
     exists k1, eval_seq m k1 c stk locs s = Some r) ->
  do_call (eval_seq (mdedup_module sg mems' m) k2) (mdedup_module sg mems' m) f stk s = Some cr ->
  exists k1, do_call (eval_seq m k1) m f stk s = Some cr.
Proof.
  intros sg mems' m k2 f stk s cr Hsim H.
  unfold do_call in *; cbn [m_imports m_funcs mdedup_module] in *.
  destruct (f <? length (m_imports m)); [exists 0; exact H|].
  rewrite nth_error_map in H.
  destruct (nth_error (m_funcs m) (f - length (m_imports m))) as [d|]; [|exists 0; exact H].
  cbn [option_map mrw_func f_sig f_locals f_body] in H.
  destruct (_ <=? _); [|exists 0; exact H].
  destruct (eval_seq (mdedup_module sg mems' m) k2 (mrw_code sg (f_body d)) _ _ s)
    as [r|] eqn:E; [|discriminate].
  destruct (Hsim _ _ _ _ _ E) as [k1 E1].
  exists k1. rewrite E1. exact H.
Qed.

(** Forward simulation for memory import deduplication. *)
Lemma sim_mdedup : forall m mems' sg, mdedup_ok m mems' sg ->
  forall k1 k2, k1 <= k2 ->
  forall c stk locs s r,
    eval_seq m k1 c stk locs s = Some r ->
    eval_seq (mdedup_module sg mems' m) k2 (mrw_code sg c) stk locs s = Some r.
Proof.
  intros m mems' sg Hok k1.
  induction k1 as [|k1 IH]; intros k2 Hle c stk locs s r H; [discriminate|].
  destruct k2 as [|k2]; [lia|].
  assert (Hle' : k1 <= k2) by lia.
  specialize (IH k2 Hle').
  revert c stk locs s r H.
  apply (code_mut
    (fun i => forall rest,
       (forall stk locs s r,
          eval_seq m (S k1) rest stk locs s = Some r ->
          eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg rest) stk locs s = Some r) ->
       forall stk locs s r,
         eval_seq m (S k1) (CCons i rest) stk locs s = Some r ->
         eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg (CCons i rest)) stk locs s
         = Some r)
    (fun c => forall stk locs s r,
       eval_seq m (S k1) c stk locs s = Some r ->
       eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg c) stk locs s = Some r));
  cbn [mrw_code mrw_instr];
  try (intros; rewrite eval_seq_cons in *; cbn [step] in *;
       try change (m_mems (mdedup_module sg mems' m)) with mems' in *;
       rewrite ?(mem_keys_map _ _ _ Hok) in *; split_hyp; eauto; fail).
  - (* call *)
    intros f rest Hr stk locs s r H.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (do_call (eval_seq m k1) m f stk s) as [cr|] eqn:E; [|discriminate].
    rewrite (do_call_mdedup _ _ _ _ _ _ _ _ _ IH E).
    destruct cr; eauto.
  - (* block *)
    intros ar b Hb rest Hr stk locs s r H.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m (S k1) b [] locs s) as [rb|] eqn:E; [|discriminate].
    rewrite (Hb _ _ _ _ E).
    destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
  - (* loop *)
    intros ar b Hb rest Hr stk locs s r H.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq m (S k1) b [] locs s) as [rb|] eqn:E; [|discriminate].
    rewrite (Hb _ _ _ _ E).
    destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
    apply IH in H; exact H.
  - (* if *)
    intros ar b1 Hb1 b2 Hb2 rest Hr stk locs s r H.
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|c stk']; [exact H|].
    destruct (Z.eqb c 0).
    + destruct (eval_seq m (S k1) b2 [] locs s) as [rb|] eqn:E; [|discriminate].
      rewrite (Hb2 _ _ _ _ E).
      destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
    + destruct (eval_seq m (S k1) b1 [] locs s) as [rb|] eqn:E; [|discriminate].
      rewrite (Hb1 _ _ _ _ E).
      destruct rb as [? ? ?|[|?] ? ? ?|? ?|]; eauto.
  - (* nil *)
    intros; rewrite eval_seq_nil in *; exact H.
Qed.

(** Backward simulation for memory import deduplication. *)
Lemma sim_mdedup_back : forall m mems' sg, mdedup_ok m mems' sg ->
  forall k2 c stk locs s r,
    eval_seq (mdedup_module sg mems' m) k2 (mrw_code sg c) stk locs s = Some r ->
    exists k1, eval_seq m k1 c stk locs s = Some r.
Proof.
  intros m mems' sg Hok k2.
  induction k2 as [|k2 IH]; intros c stk locs s r H; [discriminate|].
  revert c stk locs s r H.
  apply (code_mut
    (fun i => forall rest,
       (forall stk locs s r,
          eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg rest) stk locs s = Some r ->
          exists k1, eval_seq m k1 rest stk locs s = Some r) ->
       forall stk locs s r,
         eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg (CCons i rest)) stk locs s = Some r ->
         exists k1, eval_seq m k1 (CCons i rest) stk locs s = Some r)
    (fun c => forall stk locs s r,
       eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg c) stk locs s = Some r ->
       exists k1, eval_seq m k1 c stk locs s = Some r));
  cbn [mrw_code mrw_instr]; intros.
  - (* local.get *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (nth_error locs n) eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* local.set *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H0|].
    destruct (n <? length locs) eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* call *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (do_call (eval_seq (mdedup_module sg mems' m) k2) (mdedup_module sg mems' m) n stk s) as [cr|] eqn:E; [|discriminate].
    destruct (do_call_mdedup_back _ _ _ _ _ _ _ _ IH E) as [k1 E1].
    destruct cr as [stk' s'|].
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S (k1 + k3)).
      rewrite eval_seq_cons; cbn [step].
      rewrite (do_call_mono_le _ k1 (k1 + k3) _ _ _ _ ltac:(lia) E1).
      lift E3 (S (k1 + k3)); reflexivity.
    + exists (S k1). rewrite eval_seq_cons; cbn [step]. rewrite E1. exact H0.
  - (* prim *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (prim n stk locs s) as [[[stk' locs'] s']|] eqn:E.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite E. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite E. exact H0.
  - (* block *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg c) [] locs s) as [rb|] eqn:E; [|discriminate].
    destruct (H _ _ _ _ E) as [kb Eb].
    destruct rb as [out l s'|[|n'] out l s'|out s'|].
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
  - (* loop *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg c) [] locs s) as [rb|] eqn:E; [|discriminate].
    destruct (H _ _ _ _ E) as [kb Eb].
    destruct rb as [out l s'|[|n'] out l s'|out s'|].
    + destruct (H0 _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
    + change (eval_seq (mdedup_module sg mems' m) k2 (mrw_code sg (CCons (ILoop n c) rest)) stk l s' = Some r) in H1.
      destruct (IH _ _ _ _ _ H1) as [k3 E3]. exists (S (kb + k3)).
      rewrite eval_seq_cons; cbn [step]. lift Eb (S (kb + k3)). lift E3 (kb + k3). reflexivity.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
    + exists (S kb). rewrite eval_seq_cons; cbn [step]. lift Eb (S kb). exact H1.
  - (* if *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H2|].
    destruct (Z.eqb v 0) eqn:Ez.
    + destruct (eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg c0) [] locs s) as [rb|] eqn:E; [|discriminate].
      destruct (H0 _ _ _ _ E) as [kb Eb].
      destruct rb as [out l s'|[|n'] out l s'|out s'|].
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
    + destruct (eval_seq (mdedup_module sg mems' m) (S k2) (mrw_code sg c) [] locs s) as [rb|] eqn:E; [|discriminate].
      destruct (H _ _ _ _ E) as [kb Eb].
      destruct rb as [out l s'|[|n'] out l s'|out s'|].
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * destruct (H1 _ _ _ _ H2) as [k3 E3]. exists (S (kb + k3)).
        rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S (kb + k3)). lift E3 (S (kb + k3)). reflexivity.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
      * exists (S kb). rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift Eb (S kb). exact H2.
  - (* br *)
    exists 1. rewrite eval_seq_cons in *; cbn [step] in *. exact H0.
  - (* br_if *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct stk as [|v stk']; [exists 1; exact H0|].
    destruct (Z.eqb v 0) eqn:Ez.
    + destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
      rewrite eval_seq_cons; cbn [step]; rewrite Ez. lift E3 (S k3); reflexivity.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ez. exact H0.
  - (* return *)
    exists 1. rewrite eval_seq_cons in *; cbn [step] in *. exact H0.
  - (* nop *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
    rewrite eval_seq_cons; cbn [step]. lift E3 (S k3); reflexivity.
  - (* mem *)
    rewrite eval_seq_cons in *; cbn [step] in *.
    change (m_mems (mdedup_module sg mems' m)) with mems' in H0.
    rewrite (mem_keys_map _ _ _ Hok) in H0.
    destruct (mem_keys (m_mems m) l) as [ys|] eqn:Ek.
    + destruct (mprim n ys stk locs s) as [[[stk' locs'] s']|] eqn:E.
      * destruct (H _ _ _ _ H0) as [k3 E3]. exists (S k3).
        rewrite eval_seq_cons; cbn [step]; rewrite Ek, E. lift E3 (S k3); reflexivity.
      * exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ek, E. exact H0.
    + exists 1. rewrite eval_seq_cons; cbn [step]; rewrite Ek. exact H0.
  - (* nil *)
    exists 1. exact H.
  - (* cons *)
    apply H; assumption.
Qed.

(** Memory import deduplication is correct. If [mdedup_ok m mems' sg],
    every function of [m] and the same function of the deduplicated
    module are observationally equivalent, for any host, any semantics of
    the unmodelled and the memory instructions, and any store. Memories
    are bound by key; see [module]. *)
Theorem memory_import_dedup_correct : forall m mems' sg,
  mdedup_ok m mems' sg ->
  forall f, call_equiv m f (mdedup_module sg mems' m) f.
Proof.
  intros m mems' sg Hok f stk s c. split.
  - intros [k Hk]. exists k.
    eapply do_call_mdedup; [|exact Hk].
    intros; eapply sim_mdedup; eauto.
  - intros [k Hk].
    eapply do_call_mdedup_back; [|exact Hk].
    intros; eapply sim_mdedup_back; eauto.
Qed.

(** The instance [deduplicate_memory_imports] runs: all memory imports
    carry the first one's key, only the first is kept, and indices
    [1 .. n-1] are remapped to [0]. *)
Corollary memory_import_dedup_rust : forall m key rest,
  m_mems m = key :: rest ->
  Forall (fun x => x = key) rest ->
  forall f, call_equiv m f (mdedup_module (rust_mem_remap (S (length rest))) [key] m) f.
Proof.
  intros m key rest Hm Hall.
  apply memory_import_dedup_correct, rust_mem_remap_ok; assumption.
Qed.

(** *** Dead function elimination *)

Lemma dfe_rho_imp : forall kf ni f, f < ni -> dfe_rho kf ni f = f.
Proof. intros kf ni f H. unfold dfe_rho. apply Nat.ltb_lt in H. rewrite H. reflexivity. Qed.

Lemma dfe_rho_def : forall kf ni f, ni <= f -> dfe_rho kf ni f = ni + cnt kf 0 (f - ni).
Proof. intros kf ni f H. unfold dfe_rho. apply Nat.ltb_ge in H. rewrite H. reflexivity. Qed.

Lemma do_call_dfe : forall kf m k1 k2 f stk s cr,
  dfe_ok kf m ->
  live kf (length (m_imports m)) f ->
  (forall c stk locs s r, calls_c (live kf (length (m_imports m))) c ->
     eval_seq m k1 c stk locs s = Some r ->
     eval_seq (dfe_module kf m) k2 (rw_code (dfe_rho kf (length (m_imports m))) c) stk locs s
     = Some r) ->
  do_call (eval_seq m k1) m f stk s = Some cr ->
  do_call (eval_seq (dfe_module kf m) k2) (dfe_module kf m)
          (dfe_rho kf (length (m_imports m)) f) stk s = Some cr.
Proof.
  intros kf m k1 k2 f stk s cr Hok Hf Hsim H.
  destruct (f <? length (m_imports m)) eqn:Ef.
  { apply Nat.ltb_lt in Ef as Ef'. rewrite (dfe_rho_imp _ _ _ Ef').
    unfold do_call in *; cbn [m_imports m_funcs dfe_module] in *. rewrite Ef in *. exact H. }
  apply Nat.ltb_ge in Ef. rewrite (dfe_rho_def _ _ _ Ef).
  unfold do_call in *; cbn [m_imports m_funcs dfe_module] in *.
  assert (Ef2 : (f <? length (m_imports m)) = false) by (apply Nat.ltb_ge; exact Ef).
  rewrite Ef2 in H.
  destruct (length (m_imports m) + cnt kf 0 (f - length (m_imports m)) <? length (m_imports m))
    eqn:E'; [apply Nat.ltb_lt in E'; lia|].
  replace (length (m_imports m) + cnt kf 0 (f - length (m_imports m)) - length (m_imports m))
    with (cnt kf 0 (f - length (m_imports m))) by lia.
  rewrite nth_error_map.
  destruct (nth_error (m_funcs m) (f - length (m_imports m))) as [d|] eqn:Ed.
  - destruct Hf as [Hf|Hf]; [lia|].
    rewrite (keep_nth kf (m_funcs m) 0 _ d Hf Ed).
    cbn [option_map rw_func f_sig f_locals f_body].
    destruct (_ <=? _); [|exact H].
    destruct (eval_seq m k1 (f_body d) _ _ s) as [r|] eqn:E; [|discriminate].
    rewrite (Hsim _ _ _ _ _ (Hok _ _ Hf Ed) E). exact H.
  - apply nth_error_None in Ed.
    rewrite (keep_out kf (m_funcs m) 0 _ Ed). exact H.
Qed.

Lemma do_call_dfe_back : forall kf m k2 f stk s cr,
  dfe_ok kf m ->
  live kf (length (m_imports m)) f ->
  (forall c stk locs s r, calls_c (live kf (length (m_imports m))) c ->
     eval_seq (dfe_module kf m) k2 (rw_code (dfe_rho kf (length (m_imports m))) c) stk locs s
     = Some r ->
     exists k1, eval_seq m k1 c stk locs s = Some r) ->
  do_call (eval_seq (dfe_module kf m) k2) (dfe_module kf m)
          (dfe_rho kf (length (m_imports m)) f) stk s = Some cr ->
  exists k1, do_call (eval_seq m k1) m f stk s = Some cr.
Proof.
  intros kf m k2 f stk s cr Hok Hf Hsim H.
  destruct (f <? length (m_imports m)) eqn:Ef.
  { apply Nat.ltb_lt in Ef as Ef'. rewrite (dfe_rho_imp _ _ _ Ef') in H.
    unfold do_call in *; cbn [m_imports m_funcs dfe_module] in *. rewrite Ef in H.
    exists 0. rewrite Ef. exact H. }
  apply Nat.ltb_ge in Ef. rewrite (dfe_rho_def _ _ _ Ef) in H.
  unfold do_call in *; cbn [m_imports m_funcs dfe_module] in *.
  assert (Ef2 : (f <? length (m_imports m)) = false) by (apply Nat.ltb_ge; exact Ef).
  rewrite Ef2.
  destruct (length (m_imports m) + cnt kf 0 (f - length (m_imports m)) <? length (m_imports m))
    eqn:E'; [apply Nat.ltb_lt in E'; lia|].
  replace (length (m_imports m) + cnt kf 0 (f - length (m_imports m)) - length (m_imports m))
    with (cnt kf 0 (f - length (m_imports m))) in H by lia.
  rewrite nth_error_map in H.
  destruct (nth_error (m_funcs m) (f - length (m_imports m))) as [d|] eqn:Ed.
  - destruct Hf as [Hf|Hf]; [lia|].
    rewrite (keep_nth kf (m_funcs m) 0 _ d Hf Ed) in H.
    cbn [option_map rw_func f_sig f_locals f_body] in H.
    destruct (_ <=? _); [|exists 0; exact H].
    destruct (eval_seq (dfe_module kf m) k2 (rw_code _ (f_body d)) _ _ s)
      as [r|] eqn:E; [|discriminate].
    destruct (Hsim _ _ _ _ _ (Hok _ _ Hf Ed) E) as [k1 E1].
    exists k1. rewrite E1. exact H.
  - apply nth_error_None in Ed.
    rewrite (keep_out kf (m_funcs m) 0 _ Ed) in H. exists 0. exact H.
Qed.

(** Dead function elimination is correct. If the live set is closed under
    calls ([dfe_ok]), then every live function [f] of [m] (every import and
    every kept defined function) and function [dfe_rho kf ni f] of the
    module with the dead functions removed are observationally equivalent,
    for any host and any semantics of the unmodelled instructions.

    In fused_optimizer.rs, exports, the start function and the functions
    named by element segments are roots of the live set, and are remapped
    through the same map, so this is the statement about each of them. The
    model has no tables: that the element section is rewritten through the
    same map is checked by the tests of #494, not by this theorem. *)
Theorem dead_function_elim_correct : forall kf m,
  dfe_ok kf m ->
  forall f, live kf (length (m_imports m)) f ->
  call_equiv m f (dfe_module kf m) (dfe_rho kf (length (m_imports m)) f).
Proof.
  intros kf m Hok f Hf stk s c. split.
  - intros [k Hk]. exists k.
    eapply do_call_dfe; [exact Hok|exact Hf| |exact Hk].
    intros c' stk' locs s' r Hc' E.
    apply (sim_gen_on (live kf (length (m_imports m))) m (dfe_module kf m)
                      (dfe_rho kf (length (m_imports m))) eq_refl)
      with (k1 := k); [|lia|exact Hc'|exact E].
    intros k1 k2 _ IH g stk'' s'' cr Hg H.
    exact (do_call_dfe _ _ _ _ _ _ _ _ Hok Hg IH H).
  - intros [k Hk].
    eapply do_call_dfe_back; [exact Hok|exact Hf| |exact Hk].
    intros c' stk' locs s' r Hc' E.
    apply (sim_back_gen_on (live kf (length (m_imports m))) m (dfe_module kf m)
                           (dfe_rho kf (length (m_imports m))) eq_refl)
      with (k2 := k); [|exact Hc'|exact E].
    intros k2 IH g stk'' s'' cr Hg H.
    exact (do_call_dfe_back _ _ _ _ _ _ _ Hok Hg IH H).
Qed.

(** *** Function body deduplication *)

Lemma do_call_same : forall run m f g stk s, same_fn m f g ->
  do_call run m g stk s = do_call run m f stk s.
Proof.
  intros run m f g stk s [->|[Hf [Hg Hn]]]; [reflexivity|].
  unfold do_call.
  assert (Ef : (f <? length (m_imports m)) = false) by (apply Nat.ltb_ge; exact Hf).
  assert (Eg : (g <? length (m_imports m)) = false) by (apply Nat.ltb_ge; exact Hg).
  rewrite Ef, Eg, Hn. reflexivity.
Qed.

(** Function body deduplication is correct. If every function [f] is
    redirected to itself or to a defined function with the same signature,
    locals and body, then rewriting every call through the redirect map
    preserves every function, for any host and any semantics of the
    unmodelled instructions. The pass removes nothing; the redirected
    functions are left for dead function elimination. *)
Theorem body_dedup_correct : forall m rho,
  (forall f, same_fn m f (rho f)) ->
  forall f, call_equiv m f (rw_module rho m) f.
Proof.
  intros m rho Hrho f stk s c. split.
  - intros [k Hk]. exists k.
    eapply do_call_rw; [|exact Hk].
    apply sim_gen; [reflexivity| |lia].
    intros k1 k2 _ IH g stk' s' cr H.
    eapply do_call_rw; [exact IH|].
    rewrite (do_call_same _ _ _ _ _ _ (Hrho g)). exact H.
  - intros [k Hk].
    eapply do_call_rw_back; [|exact Hk].
    apply sim_back_gen; [reflexivity|].
    intros k2 IH g stk' s' cr H.
    destruct (do_call_rw_back _ _ _ _ _ _ _ IH H) as [k1 E1].
    exists k1. rewrite <- (do_call_same _ _ _ _ _ _ (Hrho g)). exact E1.
Qed.

End Semantics.

(** ** Concrete checks

    A concrete instance, so that the semantics is not vacuous and the stack
    convention is pinned down. [IPrim 0] is subtraction: with [b] pushed
    before [a], it leaves [b - a]. Function 0 computes [local0 - local1];
    function 1 is a trivial adapter to 0; function 2 pushes its parameters
    in the wrong order before calling 0, so it is not an adapter. *)
Module ConcreteCheck.

Definition ex_prim (p : nat) (stk locs : list Z) (s : unit)
  : option (list Z * list Z * unit) :=
  match p, stk with
  | 0, a :: b :: r => Some ((b - a)%Z :: r, locs, s)
  | _, _ => None
  end.

Definition ex_host (_ : nat) (_ : list Z) (_ : unit) : option (list Z * unit) := None.

Definition ex_mprim (_ : nat) (_ : list nat) (_ _ : list Z) (_ : unit)
  : option (list Z * list Z * unit) := None.

Definition two_to_one : sig := mkSig [0; 0] [0].

Definition ex_module : module :=
  mkModule []
    [ mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1) (CCons (IPrim 0) CNil)));
      mkFunc two_to_one 0 (gets 0 2 (CCons (ICall 0) CNil));
      mkFunc two_to_one 0 (CCons (ILocalGet 1) (CCons (ILocalGet 0) (CCons (ICall 0) CNil)));
      mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1) (CCons (ICall 1) CNil))) ] [].

Definition run (m : module) (f : nat) :=
  eval_call unit ex_prim ex_host ex_mprim m 10 f [3%Z; 10%Z] tt.

(** The caller's stack is [3; 10] with 3 on top, so parameter 0 is 10 and
    parameter 1 is 3: the target returns 10 - 3. *)
Example target_result : run ex_module 0 = Some (CDone unit [7%Z] tt).
Proof. reflexivity. Qed.

Example adapter_result : run ex_module 1 = Some (CDone unit [7%Z] tt).
Proof. reflexivity. Qed.

(** Swapping the [local.get]s changes the answer, so [trivial_adapter]'s
    ordering condition is necessary. *)
Example swapped_is_different : run ex_module 2 = Some (CDone unit [(-7)%Z] tt).
Proof. reflexivity. Qed.

Definition ex_rho (f : nat) : nat := if Nat.eqb f 1 then 0 else f.

Example devirtualized_caller :
  run (rw_module ex_rho ex_module) 3 = run ex_module 3.
Proof. reflexivity. Qed.

Lemma ex_rho_chain : forall f, chain ex_module f (ex_rho f).
Proof.
  intros f. unfold ex_rho. destruct (Nat.eqb_spec f 1) as [->|_].
  - eapply chain_step with (d := mkFunc two_to_one 0 (gets 0 2 (CCons (ICall 0) CNil))) (t := 0);
      [cbn; lia | reflexivity | split; reflexivity | reflexivity | apply chain_refl].
  - apply chain_refl.
Qed.

(** Trivial-call elimination. Function 0 is a [() -> ()] nop; function 1
    has a parameter and a nop body; function 2 is [() -> ()] but its body
    traps. Functions 3, 4 and 5 call 0, 1 and 2 respectively. *)
Definition unit_sig : sig := mkSig [] [].

Definition ex2 : module :=
  mkModule []
    [ mkFunc unit_sig 0 (CCons INop CNil);
      mkFunc (mkSig [0] []) 0 CNil;
      mkFunc unit_sig 0 (CCons (IPrim 0) CNil);
      mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1)
                          (CCons (ICall 0) (CCons (IPrim 0) CNil))));
      mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1) (CCons (ICall 1) CNil)));
      mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1)
                          (CCons (ICall 2) (CCons (IPrim 0) CNil)))) ] [].

Lemma ex2_trivial_0 : trivial_fn ex2 0.
Proof.
  split; [cbn; lia|]. eexists; split; [reflexivity|].
  repeat split; repeat constructor.
Qed.

Example nop_call_deleted :
  run (del_module (Nat.eqb 0) ex2) 3 = run ex2 3 /\ run ex2 3 = Some (CDone unit [7%Z] tt).
Proof. split; reflexivity. Qed.

(** Deleting a call to a function with a parameter changes the answer:
    the call popped the 3. So the [() -> ()] condition is necessary. *)
Example param_call_matters :
  run ex2 4 = Some (CDone unit [10%Z] tt) /\
  run (del_module (Nat.eqb 1) ex2) 4 = Some (CDone unit [3%Z] tt).
Proof. split; reflexivity. Qed.

(** Deleting a call to a [() -> ()] function whose body traps turns a
    trap into a result. So the nop-body condition is necessary. *)
Example trapping_call_matters :
  run ex2 5 = Some (CTrap unit) /\
  run (del_module (Nat.eqb 2) ex2) 5 = Some (CDone unit [7%Z] tt).
Proof. split; reflexivity. Qed.

(** Import deduplication. Host key 0 adds 1, key 1 adds 100. Imports 0
    and 1 both have key 0, import 2 has key 1; function 3 calls import 1. *)
Definition ex_host2 (k : nat) (args : list Z) (s : unit) : option (list Z * unit) :=
  match k, args with
  | 0, [a] => Some ([(a + 1)%Z], s)
  | 1, [a] => Some ([(a + 100)%Z], s)
  | _, _ => None
  end.

Definition one_one : sig := mkSig [0] [0].

Definition ex3 : module :=
  mkModule [mkImport 0 one_one; mkImport 0 one_one; mkImport 1 one_one]
    [ mkFunc one_one 0 (CCons (ILocalGet 0) (CCons (ICall 1) CNil)) ] [].

Definition ex3_imps : list import := [mkImport 0 one_one; mkImport 1 one_one].

(** What [deduplicate_imports] computes: 0 and 1 to 0, 2 to 1, and the
    defined function 3 down to 2. *)
Definition ex3_rho (f : nat) : nat :=
  match f with 0 | 1 => 0 | 2 => 1 | _ => f - 1 end.

Definition run2 (m : module) (f : nat) :=
  eval_call unit ex_prim ex_host2 ex_mprim m 10 f [3%Z; 10%Z] tt.

Lemma ex3_ok : dedup_ok ex3 ex3_imps ex3_rho.
Proof.
  split.
  - intros i Hi. cbn in Hi. destruct i as [|[|[|i]]]; [..| lia]; split; cbn; auto.
  - intros f Hf. cbn in Hf. destruct f as [|[|[|f]]]; cbn; lia.
Qed.

Example dedup_caller : run2 ex3 3 = Some (CDone unit [4%Z; 10%Z] tt) /\
  run2 (dedup_module ex3_rho ex3_imps ex3) (ex3_rho 3) = run2 ex3 3.
Proof. split; reflexivity. Qed.

(** Merging import 1 into the import with key 1 instead changes the
    answer, so the same-key condition of [dedup_ok] is necessary. *)
Definition bad_rho (f : nat) : nat :=
  match f with 0 => 0 | 1 | 2 => 1 | _ => f - 1 end.

Example wrong_key_matters :
  run2 (dedup_module bad_rho ex3_imps ex3) (bad_rho 3) = Some (CDone unit [103%Z; 10%Z] tt).
Proof. reflexivity. Qed.

(** Memory import deduplication. The store maps a memory key to one cell.
    Operation 0 stores the top of the stack into the named memory,
    operation 1 loads from it. Function 0 stores its parameter through
    memory 1 and loads through memory 0. *)
Definition mstore := nat -> Z.

Definition m_prim (_ : nat) (_ _ : list Z) (_ : mstore)
  : option (list Z * list Z * mstore) := None.

Definition m_host (_ : nat) (_ : list Z) (_ : mstore) : option (list Z * mstore) := None.

Definition m_mprim (p : nat) (ys : list nat) (stk locs : list Z) (s : mstore)
  : option (list Z * list Z * mstore) :=
  match p, ys, stk with
  | 0, [y], v :: r => Some (r, locs, fun k => if Nat.eqb k y then v else s k)
  | 1, [y], r => Some (s y :: r, locs, s)
  | _, _, _ => None
  end.

Definition alias_body : code :=
  CCons (ILocalGet 0) (CCons (IMem 0 [1]) (CCons (IMem 1 [0]) CNil)).

Definition ex4 (mems : list nat) : module :=
  mkModule [] [mkFunc one_one 0 alias_body] mems.

(** [Some (Some out)]: returned [out]; [Some None]: trapped. *)
Definition run3 (m : module) : option (option (list Z)) :=
  match eval_call mstore m_prim m_host m_mprim m 10 0 [5%Z] (fun _ => 0%Z) with
  | Some (CDone _ out _) => Some (Some out)
  | Some (CTrap _) => Some None
  | None => None
  end.

(** Two imports of key 7 are one memory: the load sees the store. *)
Example same_key_aliases : run3 (ex4 [7; 7]) = Some (Some [5%Z]).
Proof. reflexivity. Qed.

Lemma ex4_ok : mdedup_ok (ex4 [7; 7]) [7] (rust_mem_remap 2).
Proof. apply (rust_mem_remap_ok _ 7 [7]); [reflexivity | repeat constructor]. Qed.

Example same_key_dedup :
  run3 (mdedup_module (rust_mem_remap 2) [7] (ex4 [7; 7])) = Some (Some [5%Z]).
Proof. reflexivity. Qed.

(** Red control: with keys 7 and 8 the memories are distinct and the load
    sees 0. Merging them anyway would return 5, so the key check that
    [deduplicate_memory_imports] makes is needed, and [mdedup_ok] fails. *)
Example distinct_keys : run3 (ex4 [7; 8]) = Some (Some [0%Z]).
Proof. reflexivity. Qed.

Example distinct_keys_merged :
  run3 (mdedup_module (rust_mem_remap 2) [7] (ex4 [7; 8])) = Some (Some [5%Z]).
Proof. reflexivity. Qed.

Lemma distinct_keys_not_ok : ~ mdedup_ok (ex4 [7; 8]) [7] (rust_mem_remap 2).
Proof. intro H. specialize (H 1). discriminate. Qed.

(** An index past the last memory traps before and after. *)
Example out_of_range_traps :
  run3 (mkModule [] [mkFunc one_one 0 (CCons (IMem 1 [2]) CNil)] [7; 7]) = Some None /\
  run3 (mdedup_module (rust_mem_remap 2) [7]
          (mkModule [] [mkFunc one_one 0 (CCons (IMem 1 [2]) CNil)] [7; 7])) = Some None.
Proof. split; reflexivity. Qed.

(** Dead function elimination. Function 0 calls function 2; function 1
    is dead and computes [local1 - local0]; function 2 computes
    [local0 - local1]. Keeping 0 and 2 moves 2 down to 1, and the call in
    0 is rewritten to [call 1]. *)
Definition ex5 : module :=
  mkModule []
    [ mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1) (CCons (ICall 2) CNil)));
      mkFunc two_to_one 0 (CCons (ILocalGet 1) (CCons (ILocalGet 0) (CCons (IPrim 0) CNil)));
      mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1) (CCons (IPrim 0) CNil))) ] [].

Definition ex5_keep (j : nat) : bool := negb (Nat.eqb j 1).

Lemma ex5_ok : dfe_ok ex5_keep ex5.
Proof.
  intros j d Hk Hn.
  destruct j as [|[|[|j]]]; cbn in Hn; try discriminate; inversion Hn; subst; cbn;
    repeat split; unfold live; cbn; auto.
  destruct j; discriminate.
Qed.

Example dead_function_removed :
  run (dfe_module ex5_keep ex5) (dfe_rho ex5_keep 0 0) = run ex5 0 /\
  run ex5 0 = Some (CDone unit [7%Z] tt) /\
  length (m_funcs (dfe_module ex5_keep ex5)) = 2.
Proof. repeat split; reflexivity. Qed.

(** The theorem applies to this instance. *)
Example dead_function_removed_equiv :
  call_equiv unit ex_prim ex_host ex_mprim ex5 0 (dfe_module ex5_keep ex5) (dfe_rho ex5_keep 0 0).
Proof.
  apply dead_function_elim_correct; [exact ex5_ok|]. right. reflexivity.
Qed.

(** Removing the functions without rewriting the calls leaves [call 2]
    pointing past the end of the function list: the call traps. So the
    rewrite is necessary. *)
Example stale_index_matters :
  run (mkModule [] (keep_from ex5_keep 0 (m_funcs ex5)) []) 0 = Some (CTrap unit).
Proof. reflexivity. Qed.

(** Removing a function that a kept function calls changes the answer.
    So the closure condition [dfe_ok] is necessary. *)
Definition ex5_keep_bad (j : nat) : bool := Nat.eqb j 0.

Example live_callee_matters :
  ~ dfe_ok ex5_keep_bad ex5 /\
  run (dfe_module ex5_keep_bad ex5) 0 = Some (CTrap unit).
Proof.
  split; [|reflexivity].
  intros H. specialize (H 0 _ eq_refl eq_refl). cbn in H.
  destruct H as [_ [_ [[H|H] _]]]; [lia|discriminate].
Qed.

(** Function body deduplication. Functions 0 and 1 have the same record
    ([local0 - local1]); function 2 computes [local1 - local0]; function 3
    calls 1. Redirecting 1 to 0 keeps the answer; redirecting it to 2, whose
    body differs, does not, so [same_fn] is necessary. *)
Definition sub_fn : func :=
  mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1) (CCons (IPrim 0) CNil))).

Definition ex6 : module :=
  mkModule []
    [ sub_fn; sub_fn;
      mkFunc two_to_one 0 (CCons (ILocalGet 1) (CCons (ILocalGet 0) (CCons (IPrim 0) CNil)));
      mkFunc two_to_one 0 (CCons (ILocalGet 0) (CCons (ILocalGet 1) (CCons (ICall 1) CNil))) ] [].

Definition ex6_rho (f : nat) : nat := if Nat.eqb f 1 then 0 else f.

Lemma ex6_same : forall f, same_fn ex6 f (ex6_rho f).
Proof.
  intros f. unfold ex6_rho. destruct (Nat.eqb_spec f 1) as [->|_].
  - right. cbn. repeat split; lia || reflexivity.
  - left. reflexivity.
Qed.

Example body_dedup_result :
  run (rw_module ex6_rho ex6) 3 = run ex6 3 /\ run ex6 3 = Some (CDone unit [7%Z] tt).
Proof. split; reflexivity. Qed.

Example body_dedup_equiv :
  call_equiv unit ex_prim ex_host ex_mprim ex6 3 (rw_module ex6_rho ex6) 3.
Proof. apply body_dedup_correct. exact ex6_same. Qed.

Definition ex6_bad (f : nat) : nat := if Nat.eqb f 1 then 2 else f.

Example different_body_matters :
  ~ same_fn ex6 1 (ex6_bad 1) /\
  run (rw_module ex6_bad ex6) 3 = Some (CDone unit [(-7)%Z] tt).
Proof.
  split; [|reflexivity].
  intros [H|[_ [_ H]]]; cbn in H; discriminate.
Qed.

End ConcreteCheck.
