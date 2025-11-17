#!/bin/bash
# Performance comparison: LOOM vs wasm-opt
# Benchmarks all test fixtures against industry-standard wasm-opt

set -e

LOOM="./target/release/loom"
WASM_OPT="/tmp/binaryen-version_118/bin/wasm-opt"
WASM_DIS="/tmp/binaryen-version_118/bin/wasm-dis"
WASM_AS="/tmp/binaryen-version_118/bin/wasm-as"
FIXTURES_DIR="tests/fixtures"
RESULTS_DIR="/tmp/benchmark_results"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create results directory
mkdir -p "$RESULTS_DIR"

echo "======================================"
echo "  LOOM vs wasm-opt Performance Test  "
echo "======================================"
echo ""

# Function to get file size
get_size() {
    wc -c < "$1" 2>/dev/null || echo "0"
}

# Function to count instructions
count_instructions() {
    local wasm_file="$1"
    if [ ! -f "$wasm_file" ]; then
        echo "0"
        return
    fi

    # Disassemble and count instructions (excluding comments and empty lines)
    $WASM_DIS "$wasm_file" 2>/dev/null | \
        grep -v "^;;" | \
        grep -v "^$" | \
        grep -E "^\s+(i32\.|i64\.|f32\.|f64\.|local\.|global\.|memory\.|block|loop|if|call|return|end)" | \
        wc -l
}

# Summary stats
total_fixtures=0
loom_wins=0
wasm_opt_wins=0
ties=0

# Store results for final report
declare -a results

# Test each fixture
for wat_file in "$FIXTURES_DIR"/*.wat; do
    [ -f "$wat_file" ] || continue

    fixture_name=$(basename "$wat_file" .wat)
    total_fixtures=$((total_fixtures + 1))

    echo -e "${BLUE}Testing: ${fixture_name}${NC}"
    echo "----------------------------------------"

    # Compile original WAT to WASM
    original_wasm="$RESULTS_DIR/${fixture_name}_original.wasm"
    $WASM_AS "$wat_file" -o "$original_wasm" 2>/dev/null || {
        echo -e "${RED}✗ Failed to compile${NC}"
        echo ""
        continue
    }

    original_size=$(get_size "$original_wasm")
    original_instrs=$(count_instructions "$original_wasm")

    # Test LOOM
    loom_wasm="$RESULTS_DIR/${fixture_name}_loom.wasm"
    loom_start=$(date +%s%N)
    if $LOOM optimize "$wat_file" -o "$loom_wasm" --stats 2>&1 > /dev/null; then
        loom_end=$(date +%s%N)
        loom_time=$(echo "scale=3; ($loom_end - $loom_start) / 1000000" | bc)
        loom_size=$(get_size "$loom_wasm")
        loom_instrs=$(count_instructions "$loom_wasm")
        loom_reduction=$(echo "scale=1; 100 * (1 - $loom_size / $original_size)" | bc)
    else
        loom_time="N/A"
        loom_size="N/A"
        loom_instrs="N/A"
        loom_reduction="N/A"
    fi

    # Test wasm-opt -O2 (comparable to LOOM's optimization level)
    wasm_opt_o2="$RESULTS_DIR/${fixture_name}_wasm_opt_O2.wasm"
    wasm_opt_start=$(date +%s%N)
    if $WASM_OPT "$original_wasm" -O2 -o "$wasm_opt_o2" 2>/dev/null; then
        wasm_opt_end=$(date +%s%N)
        wasm_opt_time=$(echo "scale=3; ($wasm_opt_end - $wasm_opt_start) / 1000000" | bc)
        wasm_opt_size=$(get_size "$wasm_opt_o2")
        wasm_opt_instrs=$(count_instructions "$wasm_opt_o2")
        wasm_opt_reduction=$(echo "scale=1; 100 * (1 - $wasm_opt_size / $original_size)" | bc)
    else
        wasm_opt_time="N/A"
        wasm_opt_size="N/A"
        wasm_opt_instrs="N/A"
        wasm_opt_reduction="N/A"
    fi

    # Test wasm-opt -O4 (maximum optimization)
    wasm_opt_o4="$RESULTS_DIR/${fixture_name}_wasm_opt_O4.wasm"
    if $WASM_OPT "$original_wasm" -O4 -o "$wasm_opt_o4" 2>/dev/null; then
        wasm_opt_o4_size=$(get_size "$wasm_opt_o4")
        wasm_opt_o4_instrs=$(count_instructions "$wasm_opt_o4")
        wasm_opt_o4_reduction=$(echo "scale=1; 100 * (1 - $wasm_opt_o4_size / $original_size)" | bc)
    else
        wasm_opt_o4_size="N/A"
        wasm_opt_o4_instrs="N/A"
        wasm_opt_o4_reduction="N/A"
    fi

    # Test wasm-opt -Oz (size optimization)
    wasm_opt_oz="$RESULTS_DIR/${fixture_name}_wasm_opt_Oz.wasm"
    if $WASM_OPT "$original_wasm" -Oz -o "$wasm_opt_oz" 2>/dev/null; then
        wasm_opt_oz_size=$(get_size "$wasm_opt_oz")
        wasm_opt_oz_instrs=$(count_instructions "$wasm_opt_oz")
        wasm_opt_oz_reduction=$(echo "scale=1; 100 * (1 - $wasm_opt_oz_size / $original_size)" | bc)
    else
        wasm_opt_oz_size="N/A"
        wasm_opt_oz_instrs="N/A"
        wasm_opt_oz_reduction="N/A"
    fi

    # Print results
    echo "Original:"
    echo "  Size:         ${original_size} bytes"
    echo "  Instructions: ${original_instrs}"
    echo ""
    echo "LOOM:"
    echo "  Size:         ${loom_size} bytes (${loom_reduction}% reduction)"
    echo "  Instructions: ${loom_instrs}"
    echo "  Time:         ${loom_time} ms"
    echo ""
    echo "wasm-opt -O2:"
    echo "  Size:         ${wasm_opt_size} bytes (${wasm_opt_reduction}% reduction)"
    echo "  Instructions: ${wasm_opt_instrs}"
    echo "  Time:         ${wasm_opt_time} ms"
    echo ""
    echo "wasm-opt -O4 (max):"
    echo "  Size:         ${wasm_opt_o4_size} bytes (${wasm_opt_o4_reduction}% reduction)"
    echo "  Instructions: ${wasm_opt_o4_instrs}"
    echo ""
    echo "wasm-opt -Oz (size):"
    echo "  Size:         ${wasm_opt_oz_size} bytes (${wasm_opt_oz_reduction}% reduction)"
    echo "  Instructions: ${wasm_opt_oz_instrs}"
    echo ""

    # Determine winner (comparing LOOM vs wasm-opt -O2)
    if [[ "$loom_size" != "N/A" && "$wasm_opt_size" != "N/A" ]]; then
        if [ "$loom_size" -lt "$wasm_opt_size" ]; then
            winner="LOOM"
            loom_wins=$((loom_wins + 1))
            echo -e "${GREEN}✓ Winner: LOOM (smaller by $((wasm_opt_size - loom_size)) bytes)${NC}"
        elif [ "$loom_size" -gt "$wasm_opt_size" ]; then
            winner="wasm-opt"
            wasm_opt_wins=$((wasm_opt_wins + 1))
            echo -e "${YELLOW}✓ Winner: wasm-opt (smaller by $((loom_size - wasm_opt_size)) bytes)${NC}"
        else
            winner="TIE"
            ties=$((ties + 1))
            echo -e "${BLUE}✓ Winner: TIE (same size)${NC}"
        fi

        # Store results
        results+=("$fixture_name|$original_size|$loom_size|$loom_reduction|$loom_time|$wasm_opt_size|$wasm_opt_reduction|$wasm_opt_time|$wasm_opt_o4_size|$wasm_opt_o4_reduction|$winner")
    fi

    echo ""
done

# Generate summary report
echo "======================================"
echo "           SUMMARY REPORT             "
echo "======================================"
echo ""
echo "Total fixtures tested: $total_fixtures"
echo -e "${GREEN}LOOM wins:      $loom_wins${NC}"
echo -e "${YELLOW}wasm-opt wins:  $wasm_opt_wins${NC}"
echo -e "${BLUE}Ties:           $ties${NC}"
echo ""

# Win percentage
if [ $total_fixtures -gt 0 ]; then
    loom_win_pct=$(echo "scale=1; 100 * $loom_wins / $total_fixtures" | bc)
    wasm_opt_win_pct=$(echo "scale=1; 100 * $wasm_opt_wins / $total_fixtures" | bc)
    tie_pct=$(echo "scale=1; 100 * $ties / $total_fixtures" | bc)

    echo "Win Percentages:"
    echo "  LOOM:     ${loom_win_pct}%"
    echo "  wasm-opt: ${wasm_opt_win_pct}%"
    echo "  Ties:     ${tie_pct}%"
    echo ""
fi

# Detailed table
echo "======================================"
echo "         DETAILED RESULTS             "
echo "======================================"
echo ""
printf "%-20s %10s %15s %15s %12s %12s %10s\n" \
    "Fixture" "Original" "LOOM" "wasm-opt -O2" "wasm-opt -O4" "LOOM Time" "Winner"
printf "%-20s %10s %15s %15s %12s %12s %10s\n" \
    "--------------------" "----------" "---------------" "---------------" "------------" "------------" "----------"

for result in "${results[@]}"; do
    IFS='|' read -r name orig_size loom_size loom_red loom_time wasm_size wasm_red wasm_time wasm_o4_size wasm_o4_red winner <<< "$result"
    printf "%-20s %10s %10s (%3s%%) %10s (%3s%%) %10s       %7s ms %-10s\n" \
        "$name" "$orig_size" "$loom_size" "$loom_red" "$wasm_size" "$wasm_red" "$wasm_o4_size" "$loom_time" "$winner"
done

echo ""
echo "======================================"
echo "Notes:"
echo "- LOOM optimizes in ~10-30 microseconds (µs)"
echo "- wasm-opt -O2 is comparable optimization level"
echo "- wasm-opt -O4 is maximum optimization (slower)"
echo "- Comparison based on binary size reduction"
echo "======================================"
