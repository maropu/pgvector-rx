#!/bin/bash
# Run Perl TAP tests for pgvector-rx
# Usage: ./run_tap_tests.sh [test_file_pattern]
#
# Prerequisites:
#   - cargo pgrx install must have been run
#   - PostgreSQL 18 via pgrx must be available
#
# Examples:
#   ./run_tap_tests.sh                     # Run all tests
#   ./run_tap_tests.sh 039_hnsw_cost       # Run specific test

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Find pgrx PostgreSQL installation
PG_CONFIG="$HOME/.pgrx/18.2/pgrx-install/bin/pg_config"
if [ ! -f "$PG_CONFIG" ]; then
    echo "ERROR: pg_config not found at $PG_CONFIG"
    echo "Run: cargo pgrx init --pg18 download"
    exit 1
fi

PG_BINDIR=$($PG_CONFIG --bindir)
PG_LIBDIR=$($PG_CONFIG --libdir)
PG_SRCDIR="$HOME/.pgrx/18.2/src"

# Check that extension is installed
if [ ! -f "$($PG_CONFIG --pkglibdir)/pgvector_rx.dylib" ] && \
   [ ! -f "$($PG_CONFIG --pkglibdir)/pgvector_rx.so" ]; then
    echo "ERROR: pgvector_rx extension not installed."
    echo "Run: cargo pgrx install --pg-config $PG_CONFIG"
    exit 1
fi

# Set up environment for PostgreSQL TAP tests
export PATH="$PG_BINDIR:$PATH"
export PERL5LIB="$PG_SRCDIR/test/perl:$PERL5LIB"
export PG_REGRESS="$PG_BINDIR/../lib/postgresql/pgxs/src/test/regress/pg_regress"

# Test directory
TEST_DIR="$PROJECT_ROOT/tests/t"
cd "$TEST_DIR"

# Select tests to run
if [ -n "$1" ]; then
    TESTS=$(ls ${1}*.pl 2>/dev/null || echo "")
    if [ -z "$TESTS" ]; then
        echo "ERROR: No tests matching pattern '$1'"
        exit 1
    fi
else
    TESTS=$(ls *.pl 2>/dev/null || echo "")
    if [ -z "$TESTS" ]; then
        echo "No test files found in $TEST_DIR"
        exit 0
    fi
fi

# Run tests using prove
echo "Running TAP tests..."
echo "PG_CONFIG: $PG_CONFIG"
echo "Tests: $TESTS"
echo "---"

prove -v $TESTS
