#!/usr/bin/env bash
# check_wasm_opt_version.sh -- Verify the installed wasm-opt matches the pin.
#
# Reads scripts/wasm-opt.pinned (single "version_NNN" line, comments allowed),
# compares against the installed wasm-opt's --version output, and:
#
#   - exits 0 if the versions match (no-op),
#   - exits 0 with a warning if they differ but a newer version exists
#     (the user is told how to upgrade -- we do NOT auto-install),
#   - exits 0 with a warning if wasm-opt is not installed (the bench / harness
#     proceeds and marks wasm-opt columns as n/a),
#   - exits 2 only if the pin file itself is malformed.
#
# Environment:
#   WASM_OPT     -- path/name of the wasm-opt binary (default: wasm-opt)
#   PIN_FILE     -- path to the pin (default: <repo>/scripts/wasm-opt.pinned)
#   QUIET=1      -- suppress non-error output (still prints on mismatch)
#
# Exit codes:
#   0  -- ok (match, mismatch-with-guidance, or wasm-opt absent)
#   2  -- pin file unreadable or malformed
#
# Usage:
#   bash scripts/check_wasm_opt_version.sh
#   QUIET=1 bash scripts/check_wasm_opt_version.sh
#
# Integration:
#   * scripts/measure_corpus.sh sources nothing -- call this script before it
#     when you want a strict pre-flight check.
#   * The cargo bench (`loom-testing/benches/corpus_baseline.rs`) runs the
#     same comparison in-process via std::process::Command. This shell wrapper
#     is the manual equivalent for CI / developers without `cargo bench`.

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

WASM_OPT="${WASM_OPT:-wasm-opt}"
PIN_FILE="${PIN_FILE:-${REPO_ROOT}/scripts/wasm-opt.pinned}"
QUIET="${QUIET:-0}"

log() {
  if [[ "${QUIET}" != "1" ]]; then
    echo "$@"
  fi
}

# ---- Read pinned version -------------------------------------------------
if [[ ! -f "${PIN_FILE}" ]]; then
  echo "ERROR: pin file not found: ${PIN_FILE}" >&2
  exit 2
fi

# Skip blank lines and lines starting with '#'. Pin = first non-comment line.
PINNED=""
while IFS= read -r line || [[ -n "${line}" ]]; do
  trimmed="${line#"${line%%[![:space:]]*}"}"
  trimmed="${trimmed%"${trimmed##*[![:space:]]}"}"
  [[ -z "${trimmed}" ]] && continue
  [[ "${trimmed:0:1}" == "#" ]] && continue
  PINNED="${trimmed}"
  break
done < "${PIN_FILE}"

if [[ -z "${PINNED}" ]]; then
  echo "ERROR: pin file has no version line: ${PIN_FILE}" >&2
  exit 2
fi

if [[ ! "${PINNED}" =~ ^version_[0-9]+$ ]]; then
  echo "ERROR: pin '${PINNED}' is not 'version_NNN' format (in ${PIN_FILE})" >&2
  exit 2
fi

# ---- Locate installed wasm-opt --------------------------------------------
if ! command -v "${WASM_OPT}" >/dev/null 2>&1; then
  log "WARN: ${WASM_OPT} not installed; harness will mark wasm-opt columns n/a"
  log "      Install with:"
  log "        cargo install wasm-opt --locked         # crates.io shim"
  log "        brew install binaryen                   # macOS"
  exit 0
fi

# `wasm-opt --version` typically prints e.g.:
#   wasm-opt version 116 (version_116)
RAW_VER="$("${WASM_OPT}" --version 2>&1 | head -n1 || true)"
INSTALLED=""
# Prefer the parenthesised "(version_NNN)" form.
if [[ "${RAW_VER}" =~ \((version_[0-9]+)\) ]]; then
  INSTALLED="${BASH_REMATCH[1]}"
elif [[ "${RAW_VER}" =~ (version_[0-9]+) ]]; then
  INSTALLED="${BASH_REMATCH[1]}"
elif [[ "${RAW_VER}" =~ version[[:space:]]+([0-9]+) ]]; then
  INSTALLED="version_${BASH_REMATCH[1]}"
fi

if [[ -z "${INSTALLED}" ]]; then
  log "WARN: could not parse wasm-opt version from: '${RAW_VER}'"
  log "      proceeding without strict pin check"
  exit 0
fi

# ---- Compare ---------------------------------------------------------------
if [[ "${INSTALLED}" == "${PINNED}" ]]; then
  log "OK: wasm-opt ${INSTALLED} matches pin (${PINNED})"
  exit 0
fi

# Numeric comparison to tell "newer" vs "older".
PINNED_N="${PINNED#version_}"
INSTALLED_N="${INSTALLED#version_}"

# Always print mismatches even when QUIET=1 -- they are actionable.
echo "MISMATCH: installed wasm-opt ${INSTALLED} != pin ${PINNED}" >&2
if (( INSTALLED_N > PINNED_N )); then
  echo "  installed is NEWER than the pin." >&2
  echo "  Either:" >&2
  echo "    - downgrade locally to match the pin (recommended for reproducing baselines)," >&2
  echo "    - or bump the pin in ${PIN_FILE} and regenerate the corpus baseline." >&2
elif (( INSTALLED_N < PINNED_N )); then
  echo "  installed is OLDER than the pin. Upgrade with:" >&2
  echo "    cargo install wasm-opt --locked --version <X>" >&2
  echo "    brew upgrade binaryen                                     # macOS" >&2
else
  echo "  versions compare oddly (PIN=${PINNED}, INST=${INSTALLED})." >&2
fi

# Non-fatal: bench / measure harness still runs, but reports the mismatch in
# the generated report. CI can grep for "MISMATCH:" if it wants to fail-hard.
exit 0
