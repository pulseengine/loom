#!/bin/bash
# Collect WebAssembly test corpus for differential testing

set -e

CORPUS_DIR="tests/corpus"
mkdir -p "$CORPUS_DIR"

echo "📦 Collecting WebAssembly test corpus..."
echo ""

# 1. Copy LOOM's own test fixtures
echo "1️⃣  Copying LOOM test fixtures..."
if [ -d "tests/fixtures" ]; then
    mkdir -p "$CORPUS_DIR/loom-fixtures"
    find tests/fixtures -name "*.wasm" -exec cp {} "$CORPUS_DIR/loom-fixtures/" \; 2>/dev/null || true
    FIXTURE_COUNT=$(find "$CORPUS_DIR/loom-fixtures" -name "*.wasm" 2>/dev/null | wc -l)
    echo "   ✅ Copied $FIXTURE_COUNT LOOM fixtures"
else
    echo "   ⚠️  No tests/fixtures directory found"
fi

# 2. Copy component test fixtures
echo "2️⃣  Copying component test fixtures..."
if [ -d "loom-core/tests/component_fixtures" ]; then
    mkdir -p "$CORPUS_DIR/component-fixtures"
    find loom-core/tests/component_fixtures -name "*.wasm" -exec cp {} "$CORPUS_DIR/component-fixtures/" \; 2>/dev/null || true
    COMPONENT_COUNT=$(find "$CORPUS_DIR/component-fixtures" -name "*.wasm" 2>/dev/null | wc -l)
    echo "   ✅ Copied $COMPONENT_COUNT component fixtures"
else
    echo "   ⚠️  No component fixtures found"
fi

# 3. Create simple WAT examples and compile them
echo "3️⃣  Creating WAT examples..."
WAT_DIR="$CORPUS_DIR/wat-examples"
mkdir -p "$WAT_DIR"

# Simple arithmetic
cat > "$WAT_DIR/simple_add.wat" <<'EOF'
(module
  (func \$add (param \$a i32) (param \$b i32) (result i32)
    local.get \$a
    local.get \$b
    i32.add
  )
  (export "add" (func \$add))
)
EOF

# Constants
cat > "$WAT_DIR/constants.wat" <<'EOF'
(module
  (func \$calc (result i32)
    i32.const 10
    i32.const 20
    i32.add
    i32.const 5
    i32.mul
  )
  (export "calc" (func \$calc))
)
EOF

# Locals
cat > "$WAT_DIR/locals.wat" <<'EOF'
(module
  (func \$use_locals (result i32)
    (local \$temp1 i32)
    (local \$temp2 i32)
    (local \$temp3 i32)
    i32.const 10
    local.set \$temp1
    i32.const 20
    local.set \$temp2
    local.get \$temp1
    local.get \$temp2
    i32.add
    local.set \$temp3
    local.get \$temp3
  )
  (export "use_locals" (func \$use_locals))
)
EOF

# Compile WAT files to WASM if wasm-tools is available
if command -v wasm-tools &> /dev/null; then
    WAT_COUNT=0
    for wat_file in "$WAT_DIR"/*.wat; do
        if [ -f "$wat_file" ]; then
            wasm_file="${wat_file%.wat}.wasm"
            if wasm-tools parse "$wat_file" -o "$wasm_file" 2>/dev/null; then
                ((WAT_COUNT++))
            fi
        fi
    done
    echo "   ✅ Created $WAT_COUNT WAT examples"
else
    echo "   ⚠️  wasm-tools not found, skipping WAT compilation"
fi

# 4. Count total files
echo ""
echo "📊 Corpus Summary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
TOTAL_COUNT=$(find "$CORPUS_DIR" -name "*.wasm" 2>/dev/null | wc -l | tr -d ' ')
echo "Total WASM files: $TOTAL_COUNT"
echo ""
find "$CORPUS_DIR" -type d -mindepth 1 -maxdepth 1 | while read -r dir; do
    DIR_NAME=$(basename "$dir")
    COUNT=$(find "$dir" -name "*.wasm" 2>/dev/null | wc -l | tr -d ' ')
    echo "  $DIR_NAME: $COUNT files"
done
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

echo ""
echo "✅ Corpus collection complete!"
echo ""
echo "Next steps:"
echo "  1. Build LOOM: cargo build --release"
echo "  2. Run differential tests: cargo run --bin differential"
