[![Build and test](https://github.com/maropu/pgvector-rx/actions/workflows/BuildAndTest.yml/badge.svg)](https://github.com/maropu/pgvector-rx/actions/workflows/BuildAndTest.yml)
[![Prebuilt binaries](https://github.com/maropu/pgvector-rx/actions/workflows/Distribution.yml/badge.svg)](https://github.com/maropu/pgvector-rx/actions/workflows/Distribution.yml)

# pgvector-rx

A Rust implementation of the [pgvector](https://github.com/pgvector/pgvector) HNSW index for PostgreSQL 18, built with [pgrx](https://github.com/pgcentralfoundation/pgrx).

> **Experimental Project Notice**
> This repository was generated as an evaluation of autonomous AI-driven code migration capabilities. It is **not intended for production use**. This project began with an interactive collaboration phase using Claude Sonnet 4.5 to establish the foundation. During this phase, three critical documents were generated: [.github/copilot-instructions.md](.github/copilot-instructions.md) containing coding standards and pgrx/HNSW-specific guidelines, [DESIGNDOC.md](DESIGNDOC.md) outlining a comprehensive 7-phase migration roadmap and architecture documentation, and [AGENT_PROMPT.md](AGENT_PROMPT.md) defining the autonomous agent's workflow instructions. Once the scaffolding was complete, the migration roadmap in DESIGNDOC.md was systematically broken down into granular GitHub Issues, with each Issue representing a single, well-scoped implementation unit. Finally, [run_agent.sh](run_agent.sh) was executed to launch an autonomous loop where Claude Opus 4.6 iteratively processed each Issue—reading context, implementing solutions, running tests, committing changes, and verifying CI—until all tasks were completed. The entire HNSW implementation (approximately 5,000 lines of C code) was autonomously ported from [pgvector v0.8.1](https://github.com/pgvector/pgvector/tree/v0.8.1), and the complete migration—from initial setup to passing all 28 HNSW test suites—was accomplished in approximately **13 hours of agent runtime**, consuming roughly **36% of Copilot's monthly Premium Request quota**.

## Overview

pgvector-rx ports the HNSW (Hierarchical Navigable Small World) vector index from C to Rust. It provides the same `hnsw` access method and operator classes as the original pgvector extension, enabling approximate nearest-neighbor search on vector embeddings stored in PostgreSQL.

**Key features:**

- Drop-in HNSW index with L2, inner product, cosine, and L1 distance operators
- Vector types: `vector`, `halfvec`, `sparsevec`, and `bit`
- Sequential index building, concurrent inserts, vacuum with graph repair
- Iterative scan modes (relaxed/strict ordering)
- Passes all 28 HNSW tests from the original pgvector test suite

## Supported Types

| Type | Description | Max Dimensions |
|------|-------------|----------------|
| `vector` | 32-bit floating-point vector | 2,000 |
| `halfvec` | 16-bit floating-point vector | 4,000 |
| `sparsevec` | Sparse vector (non-zero elements only) | 1,000,000,000 |
| `bit` | Binary vector | 64,000 |

## Distance Operators

| Operator | Description | Operator Class |
|----------|-------------|----------------|
| `<->` | L2 (Euclidean) distance | `vector_l2_ops` |
| `<#>` | Negative inner product | `vector_ip_ops` |
| `<=>` | Cosine distance | `vector_cosine_ops` |
| `<+>` | L1 (Manhattan) distance | `vector_l1_ops` |
| `<~>` | Hamming distance (bit) | `bit_hamming_ops` |
| `<%>` | Jaccard distance (bit) | `bit_jaccard_ops` |

## Requirements

- PostgreSQL 18
- Rust (stable)
- [cargo-pgrx](https://github.com/pgcentralfoundation/pgrx) 0.17.0

## Installation

```bash
# Install cargo-pgrx
cargo install cargo-pgrx --locked --version 0.17.0

# Initialize pgrx with PostgreSQL 18
cargo pgrx init --pg18 download

# Build and install the extension
cargo pgrx install --release --features pg18 --no-default-features
```

## Usage

```sql
-- Load the extension
CREATE EXTENSION pgvector_rx;

-- Create a table with a vector column
CREATE TABLE items (id serial PRIMARY KEY, embedding vector(3));

-- Insert data
INSERT INTO items (embedding) VALUES ('[1,2,3]'), ('[4,5,6]'), ('[7,8,9]');

-- Create an HNSW index
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops);

-- Find nearest neighbors
SELECT * FROM items ORDER BY embedding <-> '[3,1,2]' LIMIT 5;
```

### Index Options

```sql
-- Customize M and ef_construction
CREATE INDEX ON items USING hnsw (embedding vector_l2_ops)
  WITH (m = 16, ef_construction = 64);
```

### Search Options

```sql
-- Adjust ef_search for recall/speed trade-off
SET hnsw.ef_search = 100;

-- Enable iterative scan
SET hnsw.iterative_scan = relaxed_order;
SET hnsw.max_scan_tuples = 20000;
```

### Other Vector Types

```sql
-- Half-precision vectors
CREATE TABLE items_half (id serial, embedding halfvec(3));
CREATE INDEX ON items_half USING hnsw (embedding halfvec_l2_ops);

-- Sparse vectors
CREATE TABLE items_sparse (id serial, embedding sparsevec(1000));
CREATE INDEX ON items_sparse USING hnsw (embedding sparsevec_l2_ops);

-- Binary vectors
CREATE TABLE items_bit (id serial, embedding bit(256));
CREATE INDEX ON items_bit USING hnsw (embedding bit_hamming_ops);
```

## Development

```bash
# Run tests
cargo pgrx test pg18

# Run regression tests
cargo pgrx regress pg18 --resetdb

# Format and lint
cargo fmt
cargo clippy --no-default-features --features pg18

# Start a PostgreSQL shell with the extension
cargo pgrx run pg18
```

## HNSW Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| `m` | 16 | 2–100 | Max connections per layer |
| `ef_construction` | 64 | 4–1,000 | Build-time candidate list size |
| `hnsw.ef_search` | 40 | 1–1,000 | Search-time candidate list size |
| `hnsw.iterative_scan` | off | off/relaxed_order/strict_order | Iterative scan mode |
| `hnsw.max_scan_tuples` | 20,000 | 1–2³¹ | Max tuples for iterative scan |
| `hnsw.scan_mem_multiplier` | 1.0 | 1–1,000 | Memory multiplier for iterative scan |

## Migration Design

See [DESIGNDOC.md](DESIGNDOC.md) for the detailed C-to-Rust migration strategy, 7-phase roadmap, and implementation guidelines.

```
src/
├── lib.rs              # Extension entry point
├── hnsw_constants.rs   # HNSW parameters and constants
├── types/              # Vector type implementations
│   ├── vector.rs       # float32 vector
│   ├── halfvec.rs      # float16 vector
│   ├── sparsevec.rs    # sparse vector
│   ├── bitvec.rs       # binary vector
│   └── hnsw.rs         # HNSW page/tuple structures
├── index/              # Index access method
│   ├── handler.rs      # AM handler and cost estimation
│   ├── build.rs        # Index building (ambuild)
│   ├── scan.rs         # Index scanning (amgettuple)
│   ├── insert.rs       # Tuple insertion (aminsert)
│   ├── vacuum.rs       # Vacuum and maintenance
│   └── options.rs      # GUC variables and reloptions
└── graph/              # In-memory graph algorithms
    └── mod.rs          # Search, neighbor selection, layer traversal
```

## Scope

This implementation covers **sequential HNSW index building** only. Parallel build support is out of scope.

## Acknowledgements

- [pgvector](https://github.com/pgvector/pgvector) by Andrew Kane — the original C implementation
- [pgrx](https://github.com/pgcentralfoundation/pgrx) — Rust framework for PostgreSQL extensions