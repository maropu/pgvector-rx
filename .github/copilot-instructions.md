# pgvector-rx Development Instructions

## Project Overview

This project is migrating the HNSW (Hierarchical Navigable Small World) index implementation from C to Rust for PostgreSQL 18. The goal is to pass all HNSW-related tests from the original pgvector test suite while maintaining functional equivalence with the C implementation.

**Scope**: Sequential HNSW index building only. Parallel build support is explicitly out of scope.

## Project Structure

```
pgvector-rx/
├── src/
│   ├── lib.rs              # Main library entry point
│   ├── types/              # Vector and data type implementations
│   ├── index/              # HNSW index structures and operations
│   │   ├── build.rs        # Index building
│   │   ├── scan.rs         # Index scanning/search
│   │   ├── insert.rs       # Tuple insertion
│   │   └── vacuum.rs       # Vacuum and maintenance
│   ├── graph/              # Graph algorithms
│   └── utils/              # Utility functions (distance, memory, etc.)
├── tests/                  # Regression tests
├── DESIGNDOC.md            # Architecture and implementation plan
└── references/             # Reference implementations
    ├── pgvector/           # Original C implementation
    ├── paradedb/           # Rust extension example
    ├── plrust/             # Rust extension example
    └── postgresml/         # Rust extension example
```

## Coding Standards

### Rust Style Guide

1. **Follow Rust conventions**:
   - Use `snake_case` for functions, variables, and modules
   - Use `PascalCase` for types, structs, and enums
   - Use `SCREAMING_SNAKE_CASE` for constants and statics
   - Maximum line length: 100 characters

2. **Documentation**:
   - All public items must have doc comments (`///`)
   - Use `//!` for module-level documentation
   - Include examples in doc comments where appropriate
   - Document safety requirements for unsafe functions

3. **Error Handling**:
   - Prefer `Result<T, E>` over panics
   - Use `anyhow::Result` for functions that can have multiple error types
   - Use custom error types when specific error handling is needed
   - Always provide context with `.context()` or `.with_context()`

4. **Safety**:
   - Minimize use of `unsafe` code
   - All `unsafe` blocks must have a `SAFETY:` comment explaining why they're safe
   - Use `#![forbid(unsafe_op_in_unsafe_fn)]` where possible
   - Wrap PostgreSQL C API calls in safe Rust abstractions

### pgrx-Specific Guidelines

1. **Extension Initialization**:
```rust
use pgrx::prelude::*;

pgrx::pg_module_magic!();

#[pg_guard]
pub extern "C" fn _PG_init() {
    // Initialize GUCs, hooks, shared memory, etc.
}
```

2. **Exported Functions**:
```rust
#[pg_extern]
fn function_name(arg1: &str, arg2: i32) -> Result<String> {
    // Function implementation
}

// With custom SQL
#[pg_extern(sql = "
CREATE FUNCTION custom_name(arg1 TEXT, arg2 INTEGER) 
RETURNS TEXT 
LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
")]
fn internal_function_name(arg1: &str, arg2: i32) -> Result<String> {
    // Function implementation
}
```

3. **Index Access Method Handler**:
```rust
#[pg_extern(sql = "
CREATE FUNCTION hnsw_handler(internal) RETURNS index_am_handler 
PARALLEL SAFE IMMUTABLE STRICT COST 0.0001 
LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
CREATE ACCESS METHOD hnsw TYPE INDEX HANDLER hnsw_handler;
COMMENT ON ACCESS METHOD hnsw IS 'hnsw index access method';
")]
fn hnsw_handler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut amroutine = unsafe { 
        PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine) 
    };
    
    // Set up AM callbacks
    amroutine.ambuild = Some(ambuild);
    amroutine.aminsert = Some(aminsert);
    // ... other callbacks
    
    amroutine.into_pg_boxed()
}
```

4. **SPI (Server Programming Interface)**:
```rust
use pgrx::spi::Spi;

Spi::connect(|mut client| {
    let result = client.select(
        "SELECT * FROM table WHERE id = $1",
        Some(1),
        Some(vec![
            (PgBuiltInOids::INT4OID.oid(), id.into_datum())
        ]),
    )?;
    // Process result
    Ok(())
})
```

5. **Memory Management**:
```rust
// Use PostgreSQL memory contexts through pgrx
use pgrx::PgMemoryContexts;

let old_context = PgMemoryContexts::CurrentMemoryContext.set_as_current();
// Allocations here
old_context.set_as_current();

// For temporary allocations
PgMemoryContexts::For(pg_sys::MemoryContextData).switch_to(|_| {
    // Temporary work
});
```

6. **Type Conversion**:
```rust
// Datum conversions
use pgrx::{FromDatum, IntoDatum};

let value: i32 = unsafe { i32::from_datum(datum, false) }.unwrap();
let datum = value.into_datum();

// For custom types
impl FromDatum for Vector {
    unsafe fn from_datum(datum: Datum, is_null: bool) -> Option<Self> {
        // Deserialization logic
    }
}

impl IntoDatum for Vector {
    fn into_datum(self) -> Option<Datum> {
        // Serialization logic
    }
    
    fn type_oid() -> Oid {
        // Return custom type OID
    }
}
```

### HNSW-Specific Guidelines

1. **Constants**:
   - Define all HNSW constants in a dedicated module
   - Match original pgvector values exactly
   - Use const for compile-time constants
   - Example:
```rust
pub const HNSW_MAX_DIM: usize = 2000;
pub const HNSW_DEFAULT_M: usize = 16;
pub const HNSW_DEFAULT_EF_CONSTRUCTION: usize = 64;
pub const HNSW_DEFAULT_EF_SEARCH: usize = 40;
```

2. **Data Structures**:
   - Use `repr(C)` for structures that interact with PostgreSQL
   - Implement `Copy` and `Clone` where appropriate
   - Use `Box<[T]>` for variable-length arrays
   - Example:
```rust
#[repr(C)]
#[derive(Copy, Clone)]
pub struct HnswMetaPage {
    magic_number: u32,
    version: u32,
    m: u16,
    ef_construction: u16,
    entry_blkno: u32,
    entry_offno: u16,
}
```

3. **Distance Functions**:
   - Implement all distance functions as traits
   - Use SIMD where possible (with fallback to scalar)
   - Prefer inline for hot path functions
   - Example:
```rust
pub trait DistanceFunction {
    fn distance(&self, a: &[f32], b: &[f32]) -> f32;
}

#[inline(always)]
fn l2_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}
```

4. **Graph Operations**:
   - Keep graph algorithms close to original C implementation
   - Document algorithm steps with references to HNSW paper
   - Use clear variable names that match the paper
   - Example:
```rust
/// Select neighbors using the heuristic algorithm from the HNSW paper.
/// Returns M nearest neighbors, prioritizing closer candidates.
fn select_neighbors_heuristic(
    candidates: &[HnswCandidate],
    m: usize,
) -> Vec<HnswCandidate> {
    // Implementation matching Algorithm 4 from paper
}
```

5. **Error Messages**:
   - Prefix errors with "pgvector-rx: " for clarity
   - Include context (table name, index name, etc.)
   - Match PostgreSQL error reporting style
   - Example:
```rust
return Err(anyhow::anyhow!(
    "pgvector-rx: dimension mismatch: expected {}, got {}",
    expected_dim,
    actual_dim
));
```

## Testing Guidelines

1. **Unit Tests**:
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_l2_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let dist = l2_distance(&a, &b);
        assert!((dist - 5.196).abs() < 0.001);
    }
}
```

2. **Integration Tests (pgrx)**:
```rust
#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;
    
    #[pg_test]
    fn test_index_creation() {
        Spi::run("CREATE TABLE test (id serial, embedding vector(3))");
        Spi::run("CREATE INDEX ON test USING hnsw (embedding vector_l2_ops)");
        // Verify index was created
    }
}
```

3. **Regression Tests**:
   - Port all SQL regression tests from references/pgvector/test/sql/hnsw_*.sql
   - Verify output matches expected/*.out files exactly
   - Test all vector types: vector, halfvec, sparsevec, bit

4. **Recall Tests**:
   - Measure and track build recall, insert recall, vacuum recall
   - Target: Within 1% of C implementation
   - Use consistent test datasets

## Performance Considerations

1. **Hot Path Optimization**:
   - Profile frequently called functions
   - Use `#[inline]` or `#[inline(always)]` for small functions
   - Minimize allocations in tight loops
   - Consider SIMD for distance calculations

2. **Memory Layout**:
   - Use cache-friendly data structures
   - Align structs appropriately
   - Minimize padding in frequently accessed structs

3. **PostgreSQL Integration**:
   - Minimize context switches between Rust and PostgreSQL
   - Batch operations where possible
   - Use PostgreSQL's memory contexts appropriately
   - Release locks promptly

## Development Workflow

1. **Before Starting Implementation**:
   - Read the relevant section in DESIGNDOC.md
   - Review the corresponding C code in references/pgvector/src/
   - Check GitHub Issues for the current phase

2. **During Implementation**:
   - Write tests first (TDD approach)
   - Implement in small, testable increments
   - Run `cargo pgrx test pg18` frequently
   - Update documentation as you go

3. **Before Submitting**:
   - Ensure all tests pass: `cargo pgrx test pg18`
   - Run formatter: `cargo fmt`
   - Run clippy: `cargo clippy --all-targets --all-features`
   - Update DESIGNDOC.md if architecture changes
   - Reference the GitHub Issue in commit message

## Common Patterns

### Pattern 1: Index AM Callback Implementation
```rust
#[pg_guard]
extern "C" fn ambuild(
    heap: pg_sys::Relation,
    index: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    // Initialize build state
    // Scan heap and insert tuples
    // Write to disk
    // Return build result
}
```

### Pattern 2: Tuple Processing
```rust
fn process_tuple(
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
) -> Result<()> {
    unsafe {
        if *isnull.offset(0) {
            // Skip NULL values
            return Ok(());
        }
        
        let vector_datum = *values.offset(0);
        let vector = Vector::from_datum(vector_datum, false)?;
        
        // Process vector
    }
    Ok(())
}
```

### Pattern 3: Buffer Management
```rust
use pgrx::pg_sys;

fn read_page(relation: pg_sys::Relation, blkno: u32) -> PgBox<pg_sys::Page> {
    unsafe {
        let buffer = pg_sys::ReadBuffer(relation, blkno);
        let page = pg_sys::BufferGetPage(buffer);
        // Process page
        pg_sys::ReleaseBuffer(buffer);
        page
    }
}
```

## References

- **HNSW Paper**: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)
- **pgrx Documentation**: https://github.com/pgcentralfoundation/pgrx
- **PostgreSQL Index AM**: https://www.postgresql.org/docs/18/indexam.html
- **Original pgvector**: references/pgvector/
- **Example Extensions**: references/paradedb/, references/plrust/, references/postgresml/

## Key Success Criteria

1. All 28 HNSW test files pass (4 SQL + 24 Perl)
2. Recall within 1% of C implementation
3. Query performance within 20% of C implementation
4. Build performance within 30% of C implementation
5. No memory leaks or crashes
6. Clean CI/CD pipeline

## FAQs

**Q: When should I use `unsafe`?**
A: Only when interfacing with PostgreSQL C API. Always wrap unsafe code in safe abstractions and document safety requirements.

**Q: How do I debug PostgreSQL crashes?**
A: Use `lldb` or `gdb` with PostgreSQL. Enable core dumps and use `cargo pgrx run pg18` with `--features pg_test`.

**Q: How do I handle PostgreSQL errors?**
A: Use `pgrx::error!()` macro or return `Result` types. pgrx will convert them to PostgreSQL errors.

**Q: Should I match the C implementation exactly?**
A: Match the behavior and correctness, but use idiomatic Rust patterns. The goal is functional equivalence, not literal translation.

**Q: How do I test memory leaks?**
A: PostgreSQL has built-in memory context leak detection. Run tests with memory context checks enabled.
