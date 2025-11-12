#!/bin/bash
# Analyze WebAssembly modules for instruction frequency

echo "=== WASM Instruction Frequency Analysis ==="
echo

TEMP_DIR="/var/folders/gh/b9f4gcc12733jn0q7m2gwc9w0000gn/T"

for i in 0 1 2 3; do
    MODULE="$TEMP_DIR/loom_module_$i.wasm"
    if [ -f "$MODULE" ]; then
        SIZE=$(wc -c < "$MODULE" | tr -d ' ')
        echo "Module $i ($SIZE bytes):"
        echo "---"

        # Convert to WAT and count instructions
        wasm2wat "$MODULE" 2>/dev/null | \
        grep -oE '\b(i32|i64|f32|f64)\.(const|load|store|add|sub|mul|div_[su]|rem_[su]|and|or|xor|shl|shr_[su]|rotl|rotr|clz|ctz|popcnt|eqz|eq|ne|lt_[su]|gt_[su]|le_[su]|ge_[su]|extend|wrap|trunc|convert|reinterpret|demote|promote)\b' | \
        sort | uniq -c | sort -rn | head -20

        echo
        echo "Other operations:"
        wasm2wat "$MODULE" 2>/dev/null | \
        grep -oE '\b(local\.(get|set|tee)|global\.(get|set)|memory\.(size|grow|fill|copy|init)|table\.(get|set|size|grow|fill|copy|init)|call(_indirect)?|br(_if|_table)?|return|drop|select|block|loop|if|else|end|unreachable|nop)\b' | \
        sort | uniq -c | sort -rn | head -15

        echo
        echo "================================"
        echo
    fi
done

echo "=== Summary ==="
echo "Analyzing all modules combined..."
for i in 0 1 2 3; do
    MODULE="$TEMP_DIR/loom_module_$i.wasm"
    [ -f "$MODULE" ] && wasm2wat "$MODULE" 2>/dev/null
done | \
grep -oE '\b(i32|i64|f32|f64)\.(const|load|store|add|sub|mul|div_[su]|rem_[su]|and|or|xor|shl|shr_[su]|rotl|rotr|clz|ctz|popcnt|eqz|eq|ne|lt_[su]|gt_[su]|le_[su]|ge_[su]|extend|wrap|trunc|convert|reinterpret|demote|promote)\b' | \
sort | uniq -c | sort -rn | head -30
