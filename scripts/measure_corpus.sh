#!/usr/bin/env bash
# measure_corpus.sh -- Corpus-wide LOOM vs wasm-opt -O3 byte-delta harness.
#
# Runs the LOOM optimizer and (optionally) wasm-opt -O3 against a curated
# set of real-world WebAssembly fixtures, validates every output via
# wasm-tools, and emits a machine-checkable markdown report.
#
# Pipeline per workload:
#   1. Record baseline byte count.
#   2. Run LOOM:           loom optimize <fixture> -> <name>.loom.wasm
#   3. Run wasm-opt -O3:   wasm-opt -O3 <fixture> -> <name>.wopt.wasm
#   4. Run LOOM AFTER wasm-opt: loom optimize <name>.wopt.wasm -> <name>.wopt-loom.wasm
#   5. Validate every output via wasm-tools validate. Any failure is HARD ERROR.
#
# Output:
#   docs/measurements/v0.9.0-corpus-baseline.md
#
# Required tools: loom (built), wasm-tools.
# Optional tools: wasm-opt (skip cleanly if absent).
#
# Exit codes:
#   0 - success
#   1 - LOOM missing or required infra missing
#   2 - hard error (LOOM produced invalid wasm on a real workload)

set -uo pipefail

# --- Resolve repo root (script must run from anywhere) ---------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# --- Configuration ---------------------------------------------------------
LOOM="${LOOM:-${REPO_ROOT}/target/release/loom}"
WASM_TOOLS="${WASM_TOOLS:-wasm-tools}"
WASM_OPT="${WASM_OPT:-wasm-opt}"
TMP_DIR="${TMP_DIR:-/tmp/loom-measure-corpus}"
REPORT_PATH="${REPORT_PATH:-${REPO_ROOT}/docs/measurements/v0.9.0-corpus-baseline.md}"

mkdir -p "${TMP_DIR}"
mkdir -p "$(dirname "${REPORT_PATH}")"

# --- Sanity checks ---------------------------------------------------------
if [[ ! -x "${LOOM}" ]]; then
  echo "ERROR: loom binary not found or not executable at ${LOOM}" >&2
  echo "       Build with: Z3_SYS_Z3_HEADER=/opt/homebrew/include/z3.h \\" >&2
  echo "                    LIBRARY_PATH=/opt/homebrew/lib cargo build --release" >&2
  exit 1
fi

if ! command -v "${WASM_TOOLS}" >/dev/null 2>&1; then
  echo "ERROR: wasm-tools not found on PATH (validation is mandatory)" >&2
  exit 1
fi

HAVE_WASM_OPT=0
if command -v "${WASM_OPT}" >/dev/null 2>&1; then
  HAVE_WASM_OPT=1
fi

MELD="${MELD:-meld}"
HAVE_MELD=0
if command -v "${MELD}" >/dev/null 2>&1; then
  HAVE_MELD=1
fi

# Per-invocation timeout (seconds). Large components can take >60min in
# LOOM's Z3-verified pipeline. The harness should not hang the developer's
# machine — any single tool invocation that exceeds PER_RUN_TIMEOUT is
# killed and the column for that workload is marked "timeout".
PER_RUN_TIMEOUT="${PER_RUN_TIMEOUT:-300}"  # 5 minutes default
TIMEOUT_BIN=""
if command -v gtimeout >/dev/null 2>&1; then
  TIMEOUT_BIN="gtimeout"
elif command -v timeout >/dev/null 2>&1; then
  TIMEOUT_BIN="timeout"
fi
# Wrap a command with the timeout if available; otherwise run as-is.
with_timeout() {
  if [[ -n "${TIMEOUT_BIN}" ]]; then
    "${TIMEOUT_BIN}" "${PER_RUN_TIMEOUT}" "$@"
  else
    "$@"
  fi
}

# Detect whether a wasm file is a Component-Model component vs a core
# module. Components start with `\0asm \r\0\1\0`, core modules with
# `\0asm \1\0\0\0`. We sniff the 8-byte header.
is_component() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then return 1; fi
  local byte4
  byte4="$(xxd -l 1 -s 4 -p "${path}" 2>/dev/null || echo "")"
  # \r = 0d
  if [[ "${byte4}" == "0d" ]]; then return 0; fi
  return 1
}

# --- Workloads -------------------------------------------------------------
# Format: "<display_name>|<path-relative-to-repo-root>|<note>"
# Paths missing on disk are reported as n/a (skipped silently).
WORKLOADS=(
  "gale|scripts/mythos/gale_measure/gale_in_baseline.wasm|kernel-FFI fixture"
  "httparse|tests/corpus/httparse.wasm|HTTP parser"
  "nom_numbers|tests/corpus/nom_numbers.wasm|parser-combinator primitives"
  "state_machine|tests/corpus/state_machine.wasm|FSM kernel"
  "json_lite|tests/corpus/json_lite.wasm|minimal JSON tokenizer"
  "loom|tests/corpus/loom.wasm|LOOM self-build (dogfood target)"
  "calculator|tests/calculator.wasm|component-shaped fixture"
  "calculator_root|calculator.wasm|2.3 MB component (root, large)"
  "simple_component|loom-core/tests/component_fixtures/simple.component.wasm|tiny component (adapter-heavy)"
  "calc_component|loom-core/tests/component_fixtures/calc.component.wasm|small component (adapter-heavy)"
)

# --- Helpers ---------------------------------------------------------------
file_size() {
  # Portable wc -c. Returns 0 on missing file.
  if [[ -f "$1" ]]; then
    wc -c < "$1" | tr -d ' '
  else
    echo "0"
  fi
}

# Compute integer percent delta with one decimal place using awk.
# Args: new_size base_size  ->  prints "-1.9" or "+0.0" or "n/a"
pct_delta() {
  local new="$1"
  local base="$2"
  if [[ -z "${base}" || "${base}" == "0" || "${base}" == "n/a" || "${new}" == "n/a" ]]; then
    echo "n/a"
    return
  fi
  awk -v n="${new}" -v b="${base}" 'BEGIN { d = (n - b) * 100.0 / b; printf "%+.1f", d }'
}

# Validate a wasm file; returns 0 on success, prints reason on failure.
validate_wasm() {
  local path="$1"
  local label="$2"
  if ! "${WASM_TOOLS}" validate "${path}" >/dev/null 2>&1; then
    local err
    err="$("${WASM_TOOLS}" validate "${path}" 2>&1 || true)"
    echo "HARD ERROR: ${label} failed wasm-tools validate" >&2
    echo "  path: ${path}" >&2
    echo "  msg : ${err}" >&2
    return 2
  fi
  return 0
}

# Get the "code section" size in bytes from wasm-tools dump (best-effort).
code_section_bytes() {
  local path="$1"
  if [[ ! -f "${path}" ]]; then
    echo "n/a"
    return
  fi
  # `wasm-tools objdump` lines look like:
  #   code        | 0x3ae - 0x6d9 |  811 bytes | 28 count
  # We sum across all "code" sections (components contain multiple). Lines
  # whose first whitespace-trimmed column is exactly "code" are matched.
  # Component-Model files don't have a top-level code section per se, but
  # `objdump` recurses into embedded core modules.
  local total
  total="$("${WASM_TOOLS}" objdump "${path}" 2>/dev/null \
    | awk -F'|' '{
        col1 = $1; gsub(/^[ \t]+|[ \t]+$/, "", col1)
        if (col1 == "code") {
          col3 = $3; gsub(/^[ \t]+|[ \t]+$/, "", col3)
          n = col3 + 0
          sum += n
        }
      } END { if (NR == 0) print "n/a"; else print sum }')"
  if [[ -z "${total}" ]]; then
    echo "n/a"
  else
    echo "${total}"
  fi
}

# --- Run pipeline ----------------------------------------------------------
LOOM_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD 2>/dev/null || echo unknown)"
LOOM_BRANCH="$(git -C "${REPO_ROOT}" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
LOOM_VERSION="$("${LOOM}" --version 2>/dev/null || echo unknown)"
RUN_TIMESTAMP="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

declare -a ROWS=()        # "name|base|loom|wopt|wopt_loom|loom_pct|wopt_pct|note|missing|red"
declare -a MELD_ROWS=()   # "name|meld_base|wopt|loom|wopt_pct|loom_pct|note" — for component fixtures only
declare -a MISSING=()
declare -a HELPS=()
declare -a NEUTRAL=()
declare -a LOSES=()
declare -a RED_ROWS=()

HARD_ERROR=0

for entry in "${WORKLOADS[@]}"; do
  IFS='|' read -r NAME REL_PATH NOTE <<< "${entry}"
  FIXTURE="${REPO_ROOT}/${REL_PATH}"

  if [[ ! -f "${FIXTURE}" ]]; then
    MISSING+=("${NAME}")
    ROWS+=("${NAME}|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|${NOTE}|1|0")
    continue
  fi

  BASE_BYTES="$(file_size "${FIXTURE}")"
  if [[ "${BASE_BYTES}" == "0" ]]; then
    MISSING+=("${NAME} (zero-byte)")
    ROWS+=("${NAME}|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|n/a|${NOTE}|1|0")
    continue
  fi

  # ---- 1. baseline (record code section bytes for diagnostics) -----------
  BASE_CODE="$(code_section_bytes "${FIXTURE}")"

  # ---- 2. LOOM optimize --------------------------------------------------
  LOOM_OUT="${TMP_DIR}/${NAME}.loom.wasm"
  LOOM_LOG="${TMP_DIR}/${NAME}.loom.log"
  LOOM_BYTES="n/a"
  LOOM_OK=0
  if with_timeout "${LOOM}" optimize "${FIXTURE}" --attestation false -o "${LOOM_OUT}" >"${LOOM_LOG}" 2>&1; then
    if validate_wasm "${LOOM_OUT}" "${NAME} (LOOM output)"; then
      LOOM_BYTES="$(file_size "${LOOM_OUT}")"
      LOOM_OK=1
    else
      HARD_ERROR=2
    fi
  else
    LOOM_BYTES="error"
  fi

  # ---- 3. wasm-opt -O3 ---------------------------------------------------
  WOPT_OUT="${TMP_DIR}/${NAME}.wopt.wasm"
  WOPT_LOG="${TMP_DIR}/${NAME}.wopt.log"
  WOPT_BYTES="n/a"
  WOPT_OK=0
  if [[ "${HAVE_WASM_OPT}" -eq 1 ]]; then
    if with_timeout "${WASM_OPT}" -O3 "${FIXTURE}" -o "${WOPT_OUT}" >"${WOPT_LOG}" 2>&1; then
      if validate_wasm "${WOPT_OUT}" "${NAME} (wasm-opt output)"; then
        WOPT_BYTES="$(file_size "${WOPT_OUT}")"
        WOPT_OK=1
      else
        # wasm-opt produced invalid wasm; flag but don't hard-fail (it's
        # not LOOM's bug). Mark column as error.
        WOPT_BYTES="invalid"
      fi
    else
      WOPT_BYTES="error"
    fi
  fi

  # ---- 4. wasm-opt -> LOOM ----------------------------------------------
  WL_OUT="${TMP_DIR}/${NAME}.wopt-loom.wasm"
  WL_LOG="${TMP_DIR}/${NAME}.wopt-loom.log"
  WL_BYTES="n/a"
  if [[ "${WOPT_OK}" -eq 1 ]]; then
    if with_timeout "${LOOM}" optimize "${WOPT_OUT}" --attestation false -o "${WL_OUT}" >"${WL_LOG}" 2>&1; then
      if validate_wasm "${WL_OUT}" "${NAME} (wasm-opt -> LOOM output)"; then
        WL_BYTES="$(file_size "${WL_OUT}")"
      else
        HARD_ERROR=2
      fi
    else
      WL_BYTES="error"
    fi
  fi

  # ---- 5. compute deltas -------------------------------------------------
  # FILE bytes include type / import / export / global / custom (debug, names,
  # attestation) sections. CODE bytes are the function-body section only —
  # the optimizer-relevant content. Debug or attestation churn can move file
  # bytes without changing what was optimized, so we report both.
  LOOM_CODE="$(code_section_bytes "${LOOM_OUT}")"
  WOPT_CODE="$(code_section_bytes "${WOPT_OUT}")"
  WL_CODE="$(code_section_bytes "${WL_OUT}")"

  LOOM_PCT="$(pct_delta "${LOOM_BYTES}" "${BASE_BYTES}")"
  WOPT_PCT="$(pct_delta "${WOPT_BYTES}" "${BASE_BYTES}")"
  LOOM_CODE_PCT="$(pct_delta "${LOOM_CODE}" "${BASE_CODE}")"
  WOPT_CODE_PCT="$(pct_delta "${WOPT_CODE}" "${BASE_CODE}")"

  RED=0
  # Red rule 1: LOOM produced LARGER output than baseline.
  if [[ "${LOOM_OK}" -eq 1 ]] && (( LOOM_BYTES > BASE_BYTES )); then
    RED=1
    RED_ROWS+=("${NAME}: LOOM grew baseline by $((LOOM_BYTES - BASE_BYTES)) bytes (${LOOM_PCT}%)")
  fi
  # Red rule 2: wasm-opt beats LOOM by more than 1% absolute.
  if [[ "${LOOM_OK}" -eq 1 && "${WOPT_OK}" -eq 1 ]]; then
    GAP=$(awk -v l="${LOOM_BYTES}" -v w="${WOPT_BYTES}" -v b="${BASE_BYTES}" \
      'BEGIN { d = (l - w) * 100.0 / b; printf "%.2f", d }')
    GAP_INT=$(awk -v g="${GAP}" 'BEGIN { printf "%d", (g >= 1.0) ? 1 : 0 }')
    if [[ "${GAP_INT}" -eq 1 ]]; then
      RED=1
      RED_ROWS+=("${NAME}: wasm-opt beats LOOM by ${GAP}% of baseline -> gap analysis recommended")
    fi
  fi

  # Bucket workloads for headline.
  if [[ "${LOOM_OK}" -eq 1 && "${WOPT_OK}" -eq 1 ]]; then
    LB_CMP=$(awk -v l="${LOOM_BYTES}" -v w="${WOPT_BYTES}" 'BEGIN { print (l < w) ? "lt" : (l > w ? "gt" : "eq") }')
    if [[ "${LB_CMP}" == "lt" ]]; then
      HELPS+=("${NAME}")
    elif [[ "${LB_CMP}" == "eq" ]]; then
      NEUTRAL+=("${NAME}")
    else
      LOSES+=("${NAME}")
    fi
  fi

  ROWS+=("${NAME}|${BASE_BYTES}|${LOOM_BYTES}|${WOPT_BYTES}|${WL_BYTES}|${LOOM_PCT}|${WOPT_PCT}|${BASE_CODE}|${LOOM_CODE}|${WOPT_CODE}|${LOOM_CODE_PCT}|${WOPT_CODE_PCT}|${NOTE}|0|${RED}")

  # ---- 6. Component-only: meld → core, then wasm-opt + LOOM on the meld output -
  #
  # For Component-Model fixtures, wasm-opt can't process the component
  # directly. But meld fuses the component into a core module, which
  # wasm-opt CAN handle. The meld-output is its OWN baseline (it
  # represents the "fused core" form of the component, structurally
  # different from the original) — we don't compare it to the component
  # baseline. Deltas in this section are relative to the meld output.
  if is_component "${FIXTURE}" && [[ "${HAVE_MELD}" -eq 1 ]]; then
    MELD_OUT="${TMP_DIR}/${NAME}.melded.wasm"
    MELD_LOG="${TMP_DIR}/${NAME}.meld.log"
    MELD_BYTES="n/a"
    MELD_OK=0
    if with_timeout "${MELD}" fuse "${FIXTURE}" -o "${MELD_OUT}" --no-attestation \
         >"${MELD_LOG}" 2>&1; then
      if validate_wasm "${MELD_OUT}" "${NAME} (meld output)"; then
        MELD_BYTES="$(file_size "${MELD_OUT}")"
        MELD_OK=1
      else
        MELD_BYTES="invalid"
      fi
    else
      MELD_BYTES="error"
    fi

    MELD_WOPT="n/a"
    MELD_LOOM="n/a"
    if [[ "${MELD_OK}" -eq 1 ]]; then
      # wasm-opt on melded core
      MWO="${TMP_DIR}/${NAME}.melded.wopt.wasm"
      if [[ "${HAVE_WASM_OPT}" -eq 1 ]]; then
        if with_timeout "${WASM_OPT}" -O3 "${MELD_OUT}" -o "${MWO}" >/dev/null 2>&1; then
          if validate_wasm "${MWO}" "${NAME} (meld→wasm-opt)"; then
            MELD_WOPT="$(file_size "${MWO}")"
          fi
        fi
      fi
      # LOOM on melded core
      MLO="${TMP_DIR}/${NAME}.melded.loom.wasm"
      if with_timeout "${LOOM}" optimize "${MELD_OUT}" --attestation false \
           -o "${MLO}" >/dev/null 2>&1; then
        if validate_wasm "${MLO}" "${NAME} (meld→LOOM)"; then
          MELD_LOOM="$(file_size "${MLO}")"
        fi
      fi
    fi

    MELD_WOPT_PCT="$(pct_delta "${MELD_WOPT}" "${MELD_BYTES}")"
    MELD_LOOM_PCT="$(pct_delta "${MELD_LOOM}" "${MELD_BYTES}")"
    MELD_ROWS+=("${NAME}|${MELD_BYTES}|${MELD_WOPT}|${MELD_LOOM}|${MELD_WOPT_PCT}|${MELD_LOOM_PCT}|${NOTE}")
  fi
done

# --- Emit report -----------------------------------------------------------
{
  echo "# v0.9.0 Corpus Baseline -- LOOM vs wasm-opt -O3"
  echo
  echo "_Generated by \`scripts/measure_corpus.sh\` at \`${RUN_TIMESTAMP}\`._"
  echo
  echo "- LOOM commit: \`${LOOM_SHA}\`"
  echo "- LOOM branch: \`${LOOM_BRANCH}\`"
  echo "- LOOM version: \`${LOOM_VERSION}\`"
  if [[ "${HAVE_WASM_OPT}" -eq 1 ]]; then
    WOPT_VER="$(${WASM_OPT} --version 2>&1 | head -n1 || echo unknown)"
    echo "- wasm-opt: \`${WOPT_VER}\` (used)"
  else
    echo "- wasm-opt: NOT INSTALLED (wasm-opt columns marked n/a)"
  fi
  echo "- wasm-tools: \`$(${WASM_TOOLS} --version 2>&1 | head -n1)\`"
  echo

  # Headline summary.
  echo "## Headline"
  echo
  HELPS_S="${HELPS[*]:-}"
  NEUTRAL_S="${NEUTRAL[*]:-}"
  LOSES_S="${LOSES[*]:-}"
  echo -n "On this corpus (only workloads where both LOOM and wasm-opt produced valid output): "
  if [[ -n "${HELPS_S}" ]]; then
    echo -n "LOOM produced a **smaller** output than wasm-opt on: ${HELPS_S// /, }. "
  fi
  if [[ -n "${NEUTRAL_S}" ]]; then
    echo -n "Neutral (byte-for-byte tie) on: ${NEUTRAL_S// /, }. "
  fi
  if [[ -n "${LOSES_S}" ]]; then
    echo -n "wasm-opt beats LOOM on: ${LOSES_S// /, }. "
  fi
  if [[ -z "${HELPS_S}" && -z "${NEUTRAL_S}" && -z "${LOSES_S}" ]]; then
    echo -n "No workload produced a side-by-side LOOM/wasm-opt pair (missing fixtures and/or wasm-opt absent)."
  fi
  echo
  echo

  if [[ "${#MISSING[@]}" -gt 0 ]]; then
    echo "Missing fixtures (skipped, marked \`n/a\`):"
    for m in "${MISSING[@]}"; do echo "- \`${m}\`"; done
    echo
  fi

  if [[ "${#RED_ROWS[@]}" -gt 0 ]]; then
    echo "## Red rows"
    echo
    for r in "${RED_ROWS[@]}"; do echo "- :red_circle: ${r}"; done
    echo
  fi

  echo "## Results — file size (total bytes incl. all sections)"
  echo
  echo "_File bytes include type / import / export / global and custom sections_"
  echo "_(name, debug, attestation, dylink). These can change without code changes;_"
  echo "_see the **code-section table** below for optimizer-relevant deltas._"
  echo
  echo "| Workload | Baseline | LOOM | wasm-opt -O3 | wasm-opt → LOOM | LOOM Δ% | wasm-opt Δ% | Note |"
  echo "|---|---:|---:|---:|---:|---:|---:|---|"
  for row in "${ROWS[@]}"; do
    IFS='|' read -r NAME BASE L W WL LP WP _BC _LC _WC _LCP _WCP NOTE _MISSING RED <<< "${row}"
    PREFIX=""
    if [[ "${RED}" == "1" ]]; then
      PREFIX=":red_circle: "
    fi
    echo "| ${PREFIX}${NAME} | ${BASE} | ${L} | ${W} | ${WL} | ${LP} | ${WP} | ${NOTE} |"
  done
  echo

  echo "## Results — code section only (optimizer-relevant)"
  echo
  echo "_Bytes of the wasm code section (function bodies) only — the surface_"
  echo "_an optimizer actually changes. Use these deltas to compare optimizer_"
  echo "_effectiveness fairly (independent of debug-info / attestation noise)._"
  echo
  echo "| Workload | Baseline (code) | LOOM (code) | wasm-opt (code) | LOOM code Δ% | wasm-opt code Δ% | Note |"
  echo "|---|---:|---:|---:|---:|---:|---|"
  for row in "${ROWS[@]}"; do
    IFS='|' read -r NAME _BASE _L _W _WL _LP _WP BC LC WC LCP WCP NOTE _MISSING _RED <<< "${row}"
    echo "| ${NAME} | ${BC} | ${LC} | ${WC} | ${LCP} | ${WCP} | ${NOTE} |"
  done
  echo

  # Second table: components-via-meld baseline (only if any meld rows produced).
  if [[ "${#MELD_ROWS[@]}" -gt 0 ]]; then
    echo "## Components via meld (fused-core baseline)"
    echo
    echo "_For Component-Model fixtures, wasm-opt cannot process the component"
    echo "directly. \`meld fuse\` produces a single core module from the component;"
    echo "that fused core is its own baseline and is structurally different from the"
    echo "original component. The deltas below compare wasm-opt and LOOM against the"
    echo "**meld output** as baseline._"
    echo
    echo "| Workload | meld baseline | wasm-opt -O3 | LOOM | wasm-opt Δ% | LOOM Δ% | Note |"
    echo "|---|---:|---:|---:|---:|---:|---|"
    for row in "${MELD_ROWS[@]}"; do
      IFS='|' read -r N MB MW ML MWP MLP MN <<< "${row}"
      echo "| ${N} | ${MB} | ${MW} | ${ML} | ${MWP} | ${MLP} | ${MN} |"
    done
    echo
  fi

  echo "## Methodology"
  echo
  echo "For each workload (fixture path is relative to repo root):"
  echo "1. Record baseline byte count via \`wc -c\` and code-section size via \`wasm-tools dump\`."
  echo "2. Run \`loom optimize <fixture> -o <name>.loom.wasm\`."
  echo "3. Run \`wasm-opt -O3 <fixture> -o <name>.wopt.wasm\` (skipped if wasm-opt unavailable)."
  echo "4. Re-run LOOM on the wasm-opt output (\`wasm-opt -> LOOM\` column)."
  echo "5. Validate every output via \`wasm-tools validate\`. **A validation failure is a HARD ERROR** -- the harness aborts with exit code 2."
  echo
  echo "Conventions:"
  echo "- Δ% is \`(out - base) / base * 100\`. Negative means smaller (better)."
  echo "- A row is flagged :red_circle: if LOOM grew the file vs. baseline, or if wasm-opt beats LOOM by more than 1% of baseline."
  echo "- Outputs of every run are in \`${TMP_DIR}\` for forensic inspection."
  echo
  echo "## Reproducing"
  echo
  echo '```bash'
  echo "# Build LOOM first (Z3 verification enabled)"
  echo "Z3_SYS_Z3_HEADER=/opt/homebrew/include/z3.h \\"
  echo "  LIBRARY_PATH=/opt/homebrew/lib cargo build --release"
  echo
  echo "# Run the harness"
  echo "bash scripts/measure_corpus.sh"
  echo '```'
} > "${REPORT_PATH}"

# --- Stdout summary --------------------------------------------------------
echo "Report written to: ${REPORT_PATH}"
echo "Workloads measured: ${#WORKLOADS[@]}"
echo "Missing fixtures : ${#MISSING[@]}"
echo "Red rows         : ${#RED_ROWS[@]}"
echo "Hard error       : ${HARD_ERROR}"

if [[ "${HARD_ERROR}" -ne 0 ]]; then
  echo "" >&2
  echo "HARD ERROR encountered (invalid wasm produced by LOOM on a real workload)." >&2
  echo "See ${TMP_DIR}/*.log for per-workload logs." >&2
  exit 2
fi

exit 0
