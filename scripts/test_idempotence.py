#!/usr/bin/env python3
"""
LOOM Idempotence Test
Tests that optimizing a WASM file multiple times produces identical results.

Usage: python test_idempotence.py <input.wasm> [--passes PASSES]
"""

import subprocess
import sys
import os
from pathlib import Path
import hashlib

def get_file_hash(filepath):
    """Calculate SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for block in iter(lambda: f.read(4096), b''):
            sha256.update(block)
    return sha256.hexdigest()

def get_wasm_stats(filepath):
    """Get statistics about a WASM file using wasm-tools."""
    try:
        # Get module structure
        result = subprocess.run(
            ['wasm-tools', 'print', filepath],
            capture_output=True,
            text=True,
            check=False
        )

        if result.returncode != 0:
            return None, f"wasm-tools error: {result.stderr[:200]}"

        wat = result.stdout

        # Count different sections
        stats = {
            'types': wat.count('(type (;'),
            'functions': wat.count('(func (;'),
            'exports': wat.count('(export '),
            'imports': wat.count('(import '),
            'tables': wat.count('(table (;'),
            'memories': wat.count('(memory (;'),
            'globals': wat.count('(global (;'),
        }

        return stats, None
    except Exception as e:
        return None, str(e)

def validate_wasm(filepath):
    """Validate a WASM file using wasm-tools."""
    try:
        result = subprocess.run(
            ['wasm-tools', 'validate', filepath],
            capture_output=True,
            text=True,
            check=False
        )
        return result.returncode == 0, result.stderr
    except Exception as e:
        return False, str(e)

def optimize_wasm(input_path, output_path, loom_path='./target/debug/loom', passes=None):
    """Run LOOM optimizer on a WASM file."""
    cmd = [loom_path, 'optimize', input_path, '-o', output_path]
    if passes:
        cmd.extend(['--passes', passes])

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False
    )

    return result.returncode == 0, result.stdout, result.stderr

def main():
    if len(sys.argv) < 2:
        print("Usage: python test_idempotence.py <input.wasm> [--passes PASSES]")
        sys.exit(1)

    input_wasm = sys.argv[1]
    passes = None

    if '--passes' in sys.argv:
        idx = sys.argv.index('--passes')
        if idx + 1 < len(sys.argv):
            passes = sys.argv[idx + 1]

    if not os.path.exists(input_wasm):
        print(f"❌ Input file not found: {input_wasm}")
        sys.exit(1)

    print("=" * 80)
    print("LOOM Idempotence Test")
    print("=" * 80)
    print(f"Input: {input_wasm}")
    if passes:
        print(f"Passes: {passes}")
    else:
        print("Passes: all (default)")
    print()

    # Get input file stats
    input_size = os.path.getsize(input_wasm)
    input_hash = get_file_hash(input_wasm)
    print(f"📊 Original File:")
    print(f"   Size: {input_size:,} bytes ({input_size / 1024 / 1024:.2f} MB)")
    print(f"   Hash: {input_hash[:16]}...")

    input_valid, input_err = validate_wasm(input_wasm)
    if input_valid:
        print(f"   Valid: ✓")
    else:
        print(f"   Valid: ✗ ({input_err[:100]}...)")

    input_stats, err = get_wasm_stats(input_wasm)
    if input_stats:
        print(f"   Types: {input_stats['types']}, Functions: {input_stats['functions']}, " +
              f"Exports: {input_stats['exports']}, Tables: {input_stats['tables']}")
    print()

    # Round 1: Optimize input
    print("🔄 Round 1: Optimizing original file...")
    opt1_path = '/tmp/test_idempotence_opt1.wasm'
    success, stdout, stderr = optimize_wasm(input_wasm, opt1_path, passes=passes)

    if not success or not os.path.exists(opt1_path):
        print(f"❌ Round 1 failed!")
        print(stderr)
        sys.exit(1)

    opt1_size = os.path.getsize(opt1_path)
    opt1_hash = get_file_hash(opt1_path)
    reduction1 = (1 - opt1_size / input_size) * 100

    print(f"   Size: {opt1_size:,} bytes ({opt1_size / 1024 / 1024:.2f} MB)")
    print(f"   Reduction: {reduction1:.1f}%")
    print(f"   Hash: {opt1_hash[:16]}...")

    opt1_valid, opt1_err = validate_wasm(opt1_path)
    if opt1_valid:
        print(f"   Valid: ✓")
    else:
        print(f"   Valid: ✗")
        print(f"   Error: {opt1_err[:200]}")

    opt1_stats, err = get_wasm_stats(opt1_path)
    if opt1_stats:
        print(f"   Types: {opt1_stats['types']}, Functions: {opt1_stats['functions']}, " +
              f"Exports: {opt1_stats['exports']}, Tables: {opt1_stats['tables']}")
    print()

    # Round 2: Optimize the optimized file
    print("🔄 Round 2: Optimizing already-optimized file...")
    opt2_path = '/tmp/test_idempotence_opt2.wasm'
    success, stdout, stderr = optimize_wasm(opt1_path, opt2_path, passes=passes)

    if not success or not os.path.exists(opt2_path):
        print(f"❌ Round 2 failed!")
        print(stderr)
        sys.exit(1)

    opt2_size = os.path.getsize(opt2_path)
    opt2_hash = get_file_hash(opt2_path)

    print(f"   Size: {opt2_size:,} bytes ({opt2_size / 1024 / 1024:.2f} MB)")
    print(f"   Hash: {opt2_hash[:16]}...")

    opt2_valid, opt2_err = validate_wasm(opt2_path)
    if opt2_valid:
        print(f"   Valid: ✓")
    else:
        print(f"   Valid: ✗")
        print(f"   Error: {opt2_err[:200]}")

    opt2_stats, err = get_wasm_stats(opt2_path)
    if opt2_stats:
        print(f"   Types: {opt2_stats['types']}, Functions: {opt2_stats['functions']}, " +
              f"Exports: {opt2_stats['exports']}, Tables: {opt2_stats['tables']}")
    print()

    # Compare results
    print("=" * 80)
    print("IDEMPOTENCE CHECK")
    print("=" * 80)

    if opt1_hash == opt2_hash:
        print("✅ PASS: Files are binary identical (idempotent)")
        print(f"   Both rounds produced: {opt1_hash[:16]}...")
        return 0
    else:
        print("❌ FAIL: Files differ (not idempotent)")
        print(f"   Round 1 hash: {opt1_hash[:16]}...")
        print(f"   Round 2 hash: {opt2_hash[:16]}...")
        print()

        size_diff = opt2_size - opt1_size
        if size_diff > 0:
            print(f"   ⚠️  Round 2 is LARGER by {size_diff:,} bytes (+{size_diff/opt1_size*100:.1f}%)")
        elif size_diff < 0:
            print(f"   ⚠️  Round 2 is SMALLER by {-size_diff:,} bytes ({size_diff/opt1_size*100:.1f}%)")
        else:
            print(f"   Size is same but content differs")

        print()
        print("   Structural differences:")
        if opt1_stats and opt2_stats:
            for key in opt1_stats:
                if opt1_stats[key] != opt2_stats[key]:
                    print(f"      {key}: {opt1_stats[key]} → {opt2_stats[key]} " +
                          f"({opt2_stats[key] - opt1_stats[key]:+d})")

        return 1

if __name__ == '__main__':
    sys.exit(main())
