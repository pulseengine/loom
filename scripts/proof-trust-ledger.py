#!/usr/bin/env python3
"""List every place the proofs take something on trust, and ratchet the list.

An entry is `file:kind:name`. Rocq sources (proofs/**/*.v) contribute:

  Axiom, Parameter, Conjecture  - declared without proof
  Admitted                      - a proof closed without being finished;
                                  `name` is the enclosing Lemma/Theorem
  admit                         - the tactic, inside the named proof
  Variable/Hypothesis/Context   - only when stated OUTSIDE any Section:
                                  there Rocq turns them into axioms. Inside
                                  a Section they are discharged at `End` as
                                  ordinary arguments of every theorem that
                                  uses them, so they are not listed.
  Unset-check                   - `Unset Guard/Positivity/Universe Checking`
                                  or a `bypass_check` attribute

Rust sources contribute Verus `external_body`, `assume(` and `admit(`.
`kani::assume` is a harness precondition, not a trust escape, and is
skipped.

Comments are stripped first (nested `(* *)` in Rocq, `//` and `/* */` in
Rust), so prose that mentions an axiom is not counted.

Usage:
  proof-trust-ledger.py            print the current entries
  proof-trust-ledger.py --check    compare with proofs/TRUST_LEDGER.txt;
                                   exit 1 on an entry the file lacks
  proof-trust-ledger.py --write    rewrite proofs/TRUST_LEDGER.txt
"""

import os
import re
import subprocess
import sys

LEDGER = "proofs/TRUST_LEDGER.txt"

PROOF_START = {
    "Lemma", "Theorem", "Corollary", "Proposition", "Fact", "Remark",
    "Example", "Definition", "Fixpoint", "CoFixpoint", "Instance",
    "Property", "Goal",
}
AXIOM_KINDS = {
    "Axiom": "Axiom", "Axioms": "Axiom",
    "Parameter": "Parameter", "Parameters": "Parameter",
    "Conjecture": "Conjecture", "Conjectures": "Conjecture",
}
SECTION_VARS = {
    "Variable", "Variables", "Hypothesis", "Hypotheses", "Context",
}
PREFIXES = {"Local", "Global", "Program", "Polymorphic", "Monomorphic",
            "Private", "Let"}


def repo_root():
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"],
                         capture_output=True, text=True, check=True)
    return out.stdout.strip()


def tracked(pattern):
    out = subprocess.run(["git", "ls-files", "-z", "--", pattern],
                         capture_output=True, text=True, check=True)
    return sorted(p for p in out.stdout.split("\0") if p)


def strip_rocq_comments(text, path):
    """Remove nested (* *) comments; keep strings; keep line structure."""
    out = []
    i, depth, n = 0, 0, len(text)
    in_string = False
    while i < n:
        c = text[i]
        if in_string:
            if c == '"':
                if i + 1 < n and text[i + 1] == '"':
                    if depth == 0:
                        out.append('""')
                    i += 2
                    continue
                in_string = False
            if depth == 0 or c == "\n":
                out.append(c if depth == 0 else "\n")
            i += 1
            continue
        if text.startswith("(*", i):
            depth += 1
            i += 2
            continue
        if depth > 0 and text.startswith("*)", i):
            depth -= 1
            i += 2
            continue
        if c == '"':
            in_string = True
            if depth == 0:
                out.append(c)
            i += 1
            continue
        if depth == 0 or c == "\n":
            out.append(c)
        i += 1
    if depth != 0:
        raise SystemExit(f"{path}: unterminated comment")
    return "".join(out)


SENTENCE_END = re.compile(r"\.(?=\s|$)")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_']*")


def binder_names(rest):
    """Names declared by `Axiom a b : T` or `Axiom (a : T) (b : U)`."""
    rest = rest.strip()
    names = []
    if rest.startswith("("):
        for group in re.findall(r"\(([^():]*):", rest):
            names.extend(IDENT.findall(group))
    else:
        head = rest.split(":", 1)[0]
        names.extend(IDENT.findall(head))
    return names


def rocq_entries(path, text):
    body = strip_rocq_comments(text, path)
    entries = []
    sections = []  # stack of (kind, name)
    current = None  # name of the proof being written
    for raw in SENTENCE_END.split(body):
        s = raw.strip()
        # Bullets, braces and attributes in front of a command.
        s = re.sub(r"^(?:[-+*]+|[{}])\s*", "", s)
        attrs = ""
        while s.startswith("#["):
            close = s.find("]")
            if close < 0:
                break
            attrs += s[:close + 1]
            s = s[close + 1:].lstrip()
        if "bypass_check" in attrs:
            entries.append((path, "Unset-check", "bypass_check"))
        words = s.split()
        while words and words[0] in PREFIXES:
            words = words[1:]
        if not words:
            continue
        head = words[0]
        rest = s[s.find(head) + len(head):]
        if head in ("Section", "Module") and len(words) > 1:
            name = IDENT.match(words[1] if words[1] != "Type" else
                               (words[2] if len(words) > 2 else ""))
            sections.append((head, name.group(0) if name else "?"))
        elif head == "End" and len(words) > 1:
            if sections:
                sections.pop()
        elif head in AXIOM_KINDS:
            for name in binder_names(rest):
                entries.append((path, AXIOM_KINDS[head], name))
        elif head in SECTION_VARS:
            if not any(k == "Section" for k, _ in sections):
                for name in binder_names(rest):
                    entries.append((path, "Axiom(toplevel " + head + ")",
                                    name))
        elif head in PROOF_START:
            m = IDENT.match(words[1]) if len(words) > 1 else None
            current = m.group(0) if m and head != "Goal" else "Goal"
        elif head == "Admitted":
            entries.append((path, "Admitted", current or "?"))
            current = None
        elif head in ("Qed", "Defined", "Abort"):
            current = None
        elif head == "Unset" and len(words) > 2 and words[2] == "Checking":
            entries.append((path, "Unset-check", words[1] + " Checking"))
        if re.search(r"(?<![A-Za-z0-9_'.])admit(?![A-Za-z0-9_'])", s) \
                and head not in ("Admitted",):
            entries.append((path, "admit", current or "?"))
    return entries


def strip_rust_comments(text):
    text = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"),
                  text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def rust_entries(path, text):
    """Named by function, not line, so unrelated edits do not churn it."""
    entries = []
    body = strip_rust_comments(text)
    fn = re.compile(r"\bfn\s+([A-Za-z_][A-Za-z0-9_]*)")
    for m in re.finditer(r"external_body", body):
        nxt = fn.search(body, m.end())
        entries.append((path, "verus-external_body",
                        nxt.group(1) if nxt else "?"))
    for m in re.finditer(r"(?<![A-Za-z0-9_:])(assume|admit)\s*\(", body):
        prev = list(fn.finditer(body, 0, m.start()))
        entries.append((path, "verus-" + m.group(1),
                        prev[-1].group(1) if prev else "?"))
    return entries


def collect():
    entries = []
    for path in tracked("*.v"):
        with open(path, encoding="utf-8") as f:
            entries.extend(rocq_entries(path, f.read()))
    for path in tracked("*.rs"):
        with open(path, encoding="utf-8", errors="replace") as f:
            text = f.read()
        if "external_body" in text or "assume" in text or "admit" in text:
            entries.extend(rust_entries(path, text))
    return sorted({f"{p}:{k}:{n}" for p, k, n in entries})


def read_ledger():
    if not os.path.exists(LEDGER):
        raise SystemExit(f"{LEDGER} is missing; run with --write")
    with open(LEDGER, encoding="utf-8") as f:
        return {l.strip() for l in f
                if l.strip() and not l.lstrip().startswith("#")}


def main(argv):
    os.chdir(repo_root())
    current = collect()
    mode = argv[1] if len(argv) > 1 else ""
    if mode == "--write":
        with open(LEDGER, "w", encoding="utf-8") as f:
            f.write("# Trusted, unproven declarations in loom's proofs.\n"
                    "# Generated by scripts/proof-trust-ledger.py --write.\n"
                    "# CI fails on an entry missing here. Removing a line is\n"
                    "# always allowed; adding one needs a reviewed reason.\n")
            for e in current:
                f.write(e + "\n")
        print(f"wrote {len(current)} entries to {LEDGER}")
        return 0
    if mode == "--check":
        known = read_ledger()
        new = [e for e in current if e not in known]
        gone = sorted(known - set(current))
        print(f"{len(current)} trusted declarations found, "
              f"{len(known)} in {LEDGER}")
        for e in gone:
            print(f"note: no longer present, delete from {LEDGER}: {e}")
        if new:
            for e in new:
                print(f"error: not in {LEDGER}: {e}")
            return 1
        return 0
    if mode:
        print(__doc__)
        return 2
    for e in current:
        print(e)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
