#!/usr/bin/env bash
# Assert a linux-gnu binary runs on stock ubuntu-22.04 (#311).
#
#   check_elf_floor.sh BINARY [MAX_GLIBC [MAX_GLIBCXX]]
#
# Fails (exit 1) when:
#   - the highest GLIBC_x.y required is above MAX_GLIBC (default 2.35), or
#   - libstdc++.so.6 is a dynamic dependency AND the highest GLIBCXX_3.4.z
#     required is above MAX_GLIBCXX (default 3.4.30).
# A statically linked libstdc++ has no such requirement. Fail-closed: if the
# floor cannot be determined (no readelf, no GLIBC version found) that is an
# error, never a pass. READELF overrides the tool (e.g. llvm-readelf).
set -euo pipefail
BIN=${1:?usage: check_elf_floor.sh BINARY [MAX_GLIBC [MAX_GLIBCXX]]}
MAX_GLIBC=${2:-2.35}
MAX_GLIBCXX=${3:-3.4.30}
READELF=${READELF:-readelf}

command -v "$READELF" >/dev/null || { echo "::error::$READELF missing, cannot determine floor"; exit 1; }
test -f "$BIN" || { echo "::error::binary not found at $BIN"; exit 1; }

dyn=$("$READELF" -d "$BIN")
vers=$("$READELF" -V "$BIN")
echo "--- NEEDED"
printf '%s\n' "$dyn" | grep NEEDED || true

need_glibc=$(printf '%s' "$vers" | grep -oE 'GLIBC_2\.[0-9]+' | sed 's/GLIBC_//' | sort -V | tail -1 || true)
need_cxx=$(printf '%s' "$vers" | grep -oE 'GLIBCXX_3\.4\.[0-9]+' | sed 's/GLIBCXX_//' | sort -V | tail -1 || true)
if [ -z "${need_glibc:-}" ]; then
  echo "::error::could not determine the GLIBC requirement, refusing to pass vacuously"
  exit 1
fi
echo "--- highest GLIBC   required: ${need_glibc} (ceiling ${MAX_GLIBC})"
echo "--- highest GLIBCXX required: ${need_cxx:-none} (ceiling ${MAX_GLIBCXX})"

fail=0
newer() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | tail -1)" != "$1" ]; }  # $2 > $1
if newer "$MAX_GLIBC" "$need_glibc"; then
  echo "::error::requires GLIBC $need_glibc > $MAX_GLIBC"; fail=1
fi
if printf '%s\n' "$dyn" | grep -q 'NEEDED.*libstdc++\.so\.6'; then
  echo "libstdc++.so.6 is a dynamic dependency"
  if [ -n "${need_cxx:-}" ] && newer "$MAX_GLIBCXX" "$need_cxx"; then
    echo "::error::requires GLIBCXX $need_cxx > $MAX_GLIBCXX with dynamic libstdc++"; fail=1
  fi
else
  echo "libstdc++ is not a dynamic dependency (statically linked)"
fi
exit "$fail"
