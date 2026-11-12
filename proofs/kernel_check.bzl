"""Re-check compiled proofs with the Rocq kernel checker and gate on assumptions.

`rocq_kernel_check` runs `rocqchk -o` over every .vo file a rocq_library
produced. rocqchk re-checks those libraries and everything they load,
Stdlib included, independently of coqc, and prints the context summary:
the axioms the checked environment rests on, the theory flags (for example
an impredicative Set) and any definition that skipped the guard,
positivity or universe checks.

The action fails when:
  - rocqchk fails (a library does not re-check, or cannot be found);
  - the summary is missing, has no Axioms section or does not parse;
  - one of the library's own .vo files was not loaded by rocqchk;
  - the summary reports something not listed in the allowlist file.

Allowlist lines are `<section>: <item>`, exactly as in <name>.assumptions.txt.
A line prefixed `[//pkg:target] ` applies to that library only; any other
line applies to every check.

The summary is written to <name>.summary.txt and the normalised lines to
<name>.assumptions.txt.
"""

load("@rules_rocq_rust//rocq/private:rocq.bzl", "RocqInfo")

_TransitiveVo = provider(
    doc = "Every .vo file a rocq_library and its deps compiled",
    fields = {"vo": "depset of .vo files"},
)

def _transitive_vo_impl(target, ctx):
    # RocqInfo.compiled holds a library's own .vo files only, so the deps'
    # are collected here. Without them as declared inputs, rocqchk could
    # start before a dependency's .vo exists.
    deps = []
    for attr in ("deps", "actual"):
        value = getattr(ctx.rule.attr, attr, None)
        if value == None:
            continue
        for d in (value if type(value) == "list" else [value]):
            if _TransitiveVo in d:
                deps.append(d[_TransitiveVo].vo)
    own = [target[RocqInfo].compiled] if RocqInfo in target else []
    return [_TransitiveVo(vo = depset(transitive = own + deps))]

_transitive_vo = aspect(
    implementation = _transitive_vo_impl,
    attr_aspects = ["deps", "actual"],
)

def _logical_name(vo, paths):
    """Module name for `vo` under the -Q pairs `paths` (longest dir wins)."""
    best = None
    for phys, log in paths:
        if vo.dirname == phys or vo.dirname.startswith(phys + "/"):
            if best == None or len(phys) > len(best[0]):
                best = (phys, log)
    if best == None:
        return None
    rel = vo.path[len(best[0]) + 1:][:-len(".vo")].replace("/", ".")
    return best[1] + "." + rel if best[1] else rel

def _impl(ctx):
    toolchain = ctx.toolchains["@rules_rocq_rust//rocq:toolchain_type"]
    info = toolchain.rocq_info
    checker = None
    for f in info.coq_tools.to_list():
        if f.basename == "rocqchk":
            checker = f
    for f in info.coq_tools.to_list():
        if checker == None and f.basename == "coqchk":
            checker = f
    if checker == None:
        fail("rocq_kernel_check: the Rocq toolchain provides no rocqchk/coqchk")

    lib = ctx.attr.library[RocqInfo]
    own = ctx.attr.library[DefaultInfo].files.to_list()
    if not own:
        fail("rocq_kernel_check: %s produced no .vo files" % ctx.attr.library.label)

    # Use the -Q pairs coqc saw. rules_rocq_rust drops a library's own pair
    # when its directory is a parent of (or equal to) a dependency's
    # directory, and the library then compiles with no prefix. Mirror that,
    # and bind such a library to the empty prefix first, so the later pairs
    # win for the subdirectories. A wrong guess is loud: rocqchk refuses a
    # .vo whose embedded name differs.
    # The provider lists the deps' pairs, then the library's own pair last.
    if not lib.output_dir or lib.include_paths[-1][0] != lib.output_dir:
        fail("rocq_kernel_check: unexpected include_paths layout for %s" %
             ctx.attr.library.label)
    own_pair = lib.include_paths[-1]
    dep_pairs = lib.include_paths[:-1]
    own_dropped = False
    for other, _ in dep_pairs:
        if other == own_pair[0] or other.startswith(own_pair[0] + "/"):
            own_dropped = True
    pairs = []
    for pair in dep_pairs + ([] if own_dropped else [own_pair]):
        if pair not in pairs:
            pairs.append(pair)
    front = []
    modules = []
    for vo in own:
        name = _logical_name(vo, pairs)
        if name == None:
            if (vo.dirname, "") not in front:
                front.append((vo.dirname, ""))
            name = vo.basename[:-len(".vo")]
        modules.append(name)

    args = ctx.actions.args()
    args.add("-o")
    args.add_all(ctx.attr.flags)
    for phys, log in front:
        args.add_all(["-Q", phys, log])
    if info.stdlib_path:
        args.add_all(["-Q", info.stdlib_path, "Stdlib"])
    for _, logical_name, lib_path in info.extra_libs:
        if lib_path:
            args.add_all(["-Q", lib_path, logical_name])
    for phys, log in pairs:
        args.add_all(["-Q", phys, log])
    args.add_all(modules)

    inputs = [checker, ctx.file.allowlist]
    if hasattr(info, "stdlib") and info.stdlib:
        inputs += info.stdlib.to_list()
    for files, _, _ in info.extra_libs:
        inputs += files

    log_out = ctx.actions.declare_file(ctx.label.name + ".summary.txt")
    norm_out = ctx.actions.declare_file(ctx.label.name + ".assumptions.txt")
    own_paths = ctx.actions.declare_file(ctx.label.name + ".own.txt")
    ctx.actions.write(own_paths, "\n".join([vo.path for vo in own]) + "\n")

    ctx.actions.run_shell(
        inputs = depset(
            inputs + [own_paths],
            transitive = [lib.compiled, ctx.attr.library[_TransitiveVo].vo],
        ),
        outputs = [log_out, norm_out],
        arguments = [args],
        env = {"OCAMLPATH": info.ocaml_path} if info.ocaml_path else {},
        mnemonic = "RocqKernelCheck",
        progress_message = "Kernel-checking %s" % ctx.attr.library.label,
        execution_requirements = {"no-sandbox": "1"} if info.extra_libs else {},
        command = """
set -euo pipefail
chk="{chk}"; full="{full}"; norm="{norm}"; allow="{allow}"; own="{own}"
label="{label}"
if ! "$chk" "$@" > "$full" 2>&1; then
  echo "rocqchk failed for $label:" >&2
  grep -v '^\\[intern\\|^ done\\]\\|^$' "$full" | tail -40 >&2
  exit 1
fi
if ! grep -q '^CONTEXT SUMMARY' "$full" || ! grep -q '^\\* Axioms:' "$full"; then
  echo "$label: rocqchk printed no context summary" >&2
  tail -40 "$full" >&2
  exit 1
fi
while IFS= read -r vo; do
  [ -z "$vo" ] && continue
  if ! grep -qF "/$vo ..." "$full"; then
    echo "$label: rocqchk did not load $vo" >&2
    exit 1
  fi
done < "$own"
# Normalise the summary to one "<section>: <item>" line per assumption.
# Every non-empty section, known or not, becomes allowlist lines, so a
# positivity, guard or type-in-type entry (or a section a later rocqchk
# adds) fails unless listed. A line in the summary that is neither a
# section header, an item nor blank makes the parse fail.
if ! awk '
  /^CONTEXT SUMMARY/ {{ on = 1; next }}
  !on {{ next }}
  /^=+$/ || /^[ \\t]*$/ {{ next }}
  /^\\* [^:]+:/ {{
    line = substr($0, 3); i = index(line, ":")
    sec = substr(line, 1, i - 1); val = substr(line, i + 1)
    gsub(/^[ \\t]+|[ \\t]+$/, "", val)
    if (val != "" && val != "<none>") print sec ": " val
    next
  }}
  /^    [^ ]/ && sec != "" {{ v = $0; gsub(/^[ \\t]+|[ \\t]+$/, "", v); print sec ": " v; next }}
  {{ print "unparsed summary line: " $0 > "/dev/stderr"; bad = 1 }}
  END {{ exit bad }}
' "$full" > "$norm.raw"; then
  echo "$label: could not parse the rocqchk context summary" >&2
  exit 1
fi
sort -u "$norm.raw" > "$norm"
rm -f "$norm.raw"
if ! grep -q '^Theory: ' "$norm"; then
  echo "$label: could not parse the rocqchk context summary" >&2
  tail -40 "$full" >&2
  exit 1
fi
# Allowlist lines apply to every check; "[//pkg:target] line" to one.
[ -r "$allow" ] || {{ echo "$label: cannot read $allow" >&2; exit 1; }}
accepted="$norm.accepted"
sed -e '/^#/d' -e '/^[[:space:]]*$/d' "$allow" | while IFS= read -r l; do
  case "$l" in
    "[$label] "*) printf '%s\\n' "${{l#"[$label] "}}" ;;
    "["*) ;;
    *) printf '%s\\n' "$l" ;;
  esac
done > "$accepted"
set +e
unknown=$(grep -vxF -f "$accepted" "$norm")
rc=$?
set -e
rm -f "$accepted"
if [ "$rc" -gt 1 ]; then
  echo "$label: grep failed comparing against $allow" >&2
  exit 1
fi
if [ -n "$unknown" ]; then
  echo "$label: rocqchk reports assumptions missing from $allow:" >&2
  printf '%s\\n' "$unknown" | sed 's/^/  /' >&2
  exit 1
fi
""".format(
            chk = checker.path,
            full = log_out.path,
            norm = norm_out.path,
            allow = ctx.file.allowlist.path,
            own = own_paths.path,
            label = "//%s:%s" % (ctx.attr.library.label.package, ctx.attr.library.label.name),
        ),
    )
    return [DefaultInfo(files = depset([log_out, norm_out]))]

rocq_kernel_check = rule(
    implementation = _impl,
    attrs = {
        "library": attr.label(
            providers = [[RocqInfo]],
            aspects = [_transitive_vo],
            mandatory = True,
            doc = "The rocq_library whose .vo files are re-checked",
        ),
        "allowlist": attr.label(
            allow_single_file = True,
            mandatory = True,
            doc = "Accepted summary lines, one per line",
        ),
        "flags": attr.string_list(
            doc = "Extra rocqchk flags, e.g. -impredicative-set",
        ),
    },
    toolchains = ["@rules_rocq_rust//rocq:toolchain_type"],
)
