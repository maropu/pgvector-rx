# pgvector HNSW Implementation Migration to Rust - Design Document

## Overview

This document outlines the plan for migrating the HNSW (Hierarchical Navigable Small World) index implementation from C to Rust in the pgvector-rx project. The original pgvector extension supports two index types (HNSW and IVFFlat), but this project will focus exclusively on HNSW.

### Final Goal

**The ultimate goal of this project is to successfully port the HNSW implementation to Rust such that all HNSW-related tests from the original pgvector test suite pass.**

This includes:
- All regression tests under `test/sql/` and `test/expected/` for HNSW
- Performance tests under `test/t/` related to HNSW
- Build recall tests (e.g., `test/t/020_hnsw_bit_build_recall.pl`)
- Insert recall tests (e.g., `test/t/013_hnsw_vector_insert_recall.pl`)
- Vacuum tests (e.g., `test/t/011_hnsw_vacuum.pl`)
- Cost estimation tests (e.g., `test/t/039_hnsw_cost.pl`)
- Iterative scan tests (e.g., `test/t/043_hnsw_iterative_scan.pl`)

The Rust implementation must demonstrate functional equivalence with the C implementation while maintaining PostgreSQL 18 compatibility.

## Source Code Analysis

### Original pgvector HNSW Implementation

The C implementation consists of approximately **5,189 lines** of code across 7 files:

| File | Lines | Purpose |
|------|-------|---------|
| `hnsw.h` | 511 | Header file with data structures, constants, and macros |
| `hnsw.c` | 336 | Index handler and initialization |
| `hnswbuild.c` | 1,141 | Index building (including parallel build) |
| `hnswinsert.c` | 791 | Insertion operations |
| `hnswscan.c` | 330 | Index scanning/search operations |
| `hnswutils.c` | 1,422 | Utility functions (distance, memory, graph operations) |
| `hnswvacuum.c` | 658 | Vacuum and maintenance operations |

### Key Data Structures

1. **HnswElementData**: Graph node representing a vector
   - Contains heap tuple IDs, level, neighbors, value pointer
   - Supports deletion marking and versioning
   - Uses locks for concurrent access

2. **HnswNeighborArray**: Array of neighbors for each layer
   - Stores candidate elements with distances
   - Dynamic allocation based on layer and M parameter

3. **HnswGraph**: Main graph structure
   - Entry point tracking
   - Index tuple statistics
   - Memory allocation state

4. **HnswBuildState**: State during index construction
   - Parallel build coordination
   - Memory context management
   - Progress tracking

5. **HnswMetaPageData**: Metadata stored on disk
   - Version information
   - Index parameters (M, efConstruction)
   - Entry point reference

### Key Parameters

- **M** (default: 16, range: 2-100): Max connections per layer
- **efConstruction** (default: 64, range: 4-1000): Dynamic candidate list size during construction
- **efSearch** (default: 40, range: 1-1000): Dynamic candidate list size during search

### Core Functionality

1. **Index Handler** (`hnsw.c`): PostgreSQL index AM (Access Method) registration
2. **Building** (`hnswbuild.c`): 
   - Sequential and parallel index construction
   - Graph initialization and tuple insertion
   - Memory management for large builds
3. **Insertion** (`hnswinsert.c`):
   - Single tuple insertion
   - Graph update with neighbor selection
   - Duplicate detection
4. **Scanning** (`hnswscan.c`):
   - k-NN search implementation
   - Priority queue management
   - Result filtering
5. **Utilities** (`hnswutils.c`):
   - Distance calculations
   - Memory allocation
   - Graph manipulation
6. **Vacuum** (`hnswvacuum.c`):
   - Dead tuple cleanup
   - Graph repair after deletions

### Dependencies

The C implementation depends heavily on PostgreSQL internals:
- Page management (`bufmgr.h`, `bufpage.h`)
- Access methods (`genam.h`, `amapi.h`)
- Memory contexts (`palloc`, `MemoryContextAlloc`)
- Lock management (`lwlock.h`, `storage/lmgr.h`)
- WAL (Write-Ahead Logging) for crash recovery
- Parallel processing (`parallel.h`)

## Migration Strategy

### Phase 1: Foundation and Data Types
**Goal**: Establish Rust project structure and basic data types

#### Tasks:
1. **Project Setup**
   - ✅ Already completed: cargo-pgrx project initialized
   - Configure pgrx for PostgreSQL 18
   - Set up module structure

2. **Vector Type Implementation**
   - Define `Vector` struct (equivalent to C's `Vector`)
   - Implement serialization/deserialization
   - Add dimension validation (max: 2000 for HNSW)
   - Create conversions to/from PostgreSQL types

3. **Core Data Structures**
   - `HnswElement`: Graph node structure
   - `HnswNeighborArray`: Neighbor array with flexible size
   - `HnswGraph`: Main graph structure
   - `HnswMetaPage`: Metadata page structure
   - Implement pointer abstractions (replacing C's relative pointers)

4. **Constants and Configuration**
   - Port HNSW parameters and constants
   - Implement `HnswOptions` for index parameters
   - Create validation functions

**Estimated complexity**: Medium (2-3 weeks)

---

### Phase 2: Utility Functions and Distance Metrics
**Goal**: Implement foundational utility functions

#### Tasks:
1. **Distance Functions**
   - L2 distance (Euclidean)
   - Inner product distance
   - Cosine distance
   - Optimized implementations using SIMD if available

2. **Memory Management**
   - Implement `HnswAllocator` abstraction
   - Page allocation utilities
   - Buffer management wrappers around PostgreSQL's buffer pool

3. **Graph Utilities**
   - Neighbor selection algorithm (select-neighbors-simple, select-neighbors-heuristic)
   - Level assignment using exponential decay
   - Hash functions for duplicate detection

4. **Support Functions**
   - `HnswSupport` initialization
   - Type information handling
   - Normalization utilities

**Estimated complexity**: Medium-High (3-4 weeks)

---

### Phase 3: Index Building (Sequential)
**Goal**: Implement basic index construction

#### Tasks:
1. **Build State Management**
   - `HnswBuildState` initialization
   - Memory context management
   - Progress reporting integration

2. **Graph Construction**
   - Insert elements into graph (in-memory)
   - Layer assignment
   - Neighbor updates using greedy search
   - Entry point management

3. **Disk Persistence**
   - Create meta page
   - Write element tuples
   - Write neighbor tuples
   - Page management

4. **Integration**
   - Implement `hnswbuild` handler
   - Connect to PostgreSQL's index build infrastructure
   - WAL logging for crash recovery

**Estimated complexity**: High (4-5 weeks)

---

### Phase 4: Search and Scanning
**Goal**: Implement k-NN search functionality

#### Tasks:
1. **Search Algorithm**
   - Implement `search-layer` algorithm
   - Priority queue management (using Rust's BinaryHeap)
   - Visited set tracking

2. **Index Scanning**
   - `hnswbeginscan`: Initialize scan
   - `hnswgettuple`: Retrieve next tuple
   - `hnswrescan`: Restart scan with new parameters
   - `hnswendscan`: Cleanup

3. **Query Options**
   - `ef_search` parameter handling
   - Iterative scan support (relaxed/strict modes)
   - Memory limit enforcement

4. **Result Filtering**
   - Heap tuple validation
   - Duplicate handling
   - NULL vector handling

**Estimated complexity**: High (4-5 weeks)

---

### Phase 5: Insertion and Updates
**Goal**: Implement single tuple insertion

#### Tasks:
1. **On-Disk Insertion**
   - Find insert page
   - Allocate space for element and neighbor tuples
   - Handle page overflow

2. **Graph Updates**
   - Find entry layer
   - Search for nearest neighbors
   - Update neighbor connections
   - Handle duplicates

3. **Concurrency**
   - Implement locking strategy
   - Update lock and scan lock management
   - Entry point synchronization

4. **Integration**
   - Implement `hnswinsert` handler
   - WAL logging for insertions

**Estimated complexity**: High (3-4 weeks)

---

### Phase 6: Vacuum and Maintenance
**Goal**: Implement cleanup and maintenance operations

#### Tasks:
1. **Vacuum Support**
   - Mark deleted elements
   - Repair graph connections
   - Reclaim dead tuples

2. **Reindexing**
   - Support for REINDEX command
   - Concurrent reindexing

3. **Statistics**
   - Index size reporting
   - Tuple count tracking
   - Performance metrics

**Estimated complexity**: Medium (2-3 weeks)

---

### Phase 7: Parallel Build Support
**Goal**: Add parallel index construction (optional for MVP)

#### Tasks:
1. **Parallel Coordination**
   - Worker process management
   - Shared memory setup
   - Dynamic shared memory (DSM) integration

2. **Work Distribution**
   - Heap scanning parallelization
   - Graph merging strategy

3. **Synchronization**
   - Lock coordination across workers
   - Progress aggregation

**Estimated complexity**: High (3-4 weeks)
**Priority**: Low (can be deferred to post-MVP)

---

### Phase 8: Testing and Optimization
**Goal**: Ensure correctness and performance

#### Tasks:
1. **Unit Tests**
   - Distance function tests
   - Data structure tests
   - Graph algorithm tests

2. **Integration Tests**
   - Index creation tests
   - Search accuracy tests (recall measurement)
   - Insert/delete tests
   - Vacuum tests

3. **Performance Testing**
   - Benchmark against original C implementation
   - Memory usage profiling
   - Query performance testing

4. **Optimization**
   - SIMD optimization for distance calculations
   - Memory layout optimization
   - Cache-friendly data structures

**Estimated complexity**: High (4-5 weeks)

---

## Migration Challenges

### 1. PostgreSQL C API Integration
- **Challenge**: pgrx provides Rust bindings, but may not cover all functions used by HNSW
- **Solution**: Use `unsafe` blocks for direct C API calls where necessary
- **Risk**: Medium - requires careful memory management

### 2. Page and Buffer Management
- **Challenge**: HNSW uses low-level PostgreSQL buffer pool and page APIs
- **Solution**: Create safe Rust wrappers around buffer management
- **Risk**: Medium - critical for data integrity

### 3. Write-Ahead Logging (WAL)
- **Challenge**: All index modifications must be WAL-logged for crash recovery
- **Solution**: Use pgrx's WAL support or implement custom WAL records
- **Risk**: High - errors can cause data corruption

### 4. Parallel Processing
- **Challenge**: PostgreSQL's parallel infrastructure is C-centric
- **Solution**: Use DSM and shared memory carefully; consider deferring parallel build
- **Risk**: High - complex synchronization

### 5. Pointer Abstractions
- **Challenge**: C code uses relative pointers for in-memory vs on-disk structures
- **Solution**: Implement Rust enums or traits to abstract pointer types
- **Risk**: Medium - affects performance and correctness

### 6. Lock Management
- **Challenge**: Fine-grained locking for concurrent access
- **Solution**: Use PostgreSQL's LWLock through pgrx
- **Risk**: Medium - deadlocks or race conditions possible

### 7. Memory Context Integration
- **Challenge**: PostgreSQL's memory context system for automatic cleanup
- **Solution**: Integrate with pgrx's memory context support
- **Risk**: Low-Medium - memory leaks if done incorrectly

## Testing Strategy

The testing strategy focuses on ensuring full compatibility with pgvector's HNSW test suite:

### 1. Regression Test Porting
**Goal**: All pgvector HNSW tests pass with Rust implementation

Test files to port from `references/pgvector/test/`:
- `sql/hnsw_vector.sql` → `expected/hnsw_vector.out`
- `sql/hnsw_halfvec.sql` → `expected/hnsw_halfvec.out`
- `sql/hnsw_sparsevec.sql` → `expected/hnsw_sparsevec.out`
- `sql/hnsw_bit.sql` → `expected/hnsw_bit.out`

Perl test scripts to port from `test/t/`:
- `t/011_hnsw_vacuum.pl` - Vacuum functionality
- `t/013_hnsw_vector_insert_recall.pl` - Insert recall measurement
- `t/020_hnsw_bit_build_recall.pl` - Build recall for bit vectors
- `t/022_hnsw_bit_vacuum_recall.pl` - Vacuum recall for bit vectors
- `t/039_hnsw_cost.pl` - Cost estimation
- `t/043_hnsw_iterative_scan.pl` - Iterative scan functionality

### 2. Recall Testing
**Goal**: Match or exceed C implementation's recall rates

- Build recall: Measure accuracy of index construction
- Search recall: Measure k-NN search accuracy at various ef_search values
- Post-vacuum recall: Ensure accuracy maintained after vacuum operations
- Use standard datasets (e.g., SIFT, GIST) for benchmarking

### 3. Unit Tests
**Goal**: Test individual Rust components in isolation

- Distance function correctness (L2, inner product, cosine)
- Graph algorithm correctness (neighbor selection, layer assignment)
- Data structure serialization/deserialization
- Memory allocation and deallocation
- Lock and concurrency primitives

### 4. Integration Tests
**Goal**: Test end-to-end workflows

- Index creation with various parameters
- Bulk insertion followed by queries
- Concurrent inserts and searches
- Vacuum after deletions
- Index rebuild (REINDEX)

### 5. Performance Tests
**Goal**: Ensure acceptable performance vs. C implementation

- Build time comparison (sequential and parallel)
- Query latency at various ef_search values
- Throughput under concurrent load
- Memory usage profiling

### 6. Stress Tests
**Goal**: Verify stability under extreme conditions

- Large-scale indexes (millions of vectors)
- High-dimensional vectors (approaching 2000 dimensions)
- Heavy concurrent write workload
- Recovery after crashes (WAL replay)

### 7. Continuous Integration
**Goal**: Automated testing on every commit

- GitHub Actions workflow running full test suite
- Test against PostgreSQL 18
- Recall threshold checks
- Performance regression detection
- Memory leak detection with valgrind/sanitizers

### Test Acceptance Criteria
- ✅ All SQL regression tests produce identical output
- ✅ All Perl integration tests pass
- ✅ Recall rates within 1% of C implementation
- ✅ No memory leaks or crashes
- ✅ Performance within acceptable thresholds

## Success Criteria

The project will be considered successful when the following criteria are met:

1. **Test Suite Compliance (PRIMARY GOAL)**: 
   - All HNSW-related regression tests from pgvector pass (SQL tests in `test/expected/hnsw_*.out`)
   - All HNSW-related Perl integration tests pass (tests in `test/t/*hnsw*.pl`)
   - Specific test categories include:
     - Vector type support tests (`hnsw_vector.out`)
     - Halfvec type support tests (`hnsw_halfvec.out`)
     - Sparsevec type support tests (`hnsw_sparsevec.out`)
     - Bit vector type support tests (`hnsw_bit.out`)
     - Build recall tests (measuring index construction accuracy)
     - Insert recall tests (measuring insertion accuracy)
     - Vacuum recall tests (measuring post-vacuum accuracy)
     - Iterative scan tests
     - Cost estimation tests

2. **Functional Correctness**: 
   - Index creation with various parameters (m, ef_construction)
   - Search with various ef_search values
   - Insert, update, and delete operations
   - Concurrent access handling
   - Vacuum and maintenance operations

3. **Performance**: 
   - Query performance within 20% of C implementation
   - Build performance within 30% of C implementation (acceptable trade-off for memory safety)
   - Memory usage comparable to C implementation

4. **Memory Safety**: 
   - No memory leaks detected by PostgreSQL memory context checks
   - No undefined behavior or crashes
   - Safe handling of all error conditions

5. **Compatibility**: 
   - Works with PostgreSQL 18
   - SQL API matches original pgvector HNSW interface
   - Index on-disk format allows pg_upgrade compatibility (stretch goal)

6. **Code Quality**:
   - Comprehensive Rust unit tests
   - Documentation for key functions and data structures
   - CI/CD integration with automated testing

## Timeline Estimate

| Phase | Weeks | Priority |
|-------|-------|----------|
| 1. Foundation | 2-3 | Critical |
| 2. Utilities | 3-4 | Critical |
| 3. Build (Sequential) | 4-5 | Critical |
| 4. Search | 4-5 | Critical |
| 5. Insertion | 3-4 | Critical |
| 6. Vacuum | 2-3 | High |
| 7. Parallel Build | 3-4 | Low |
| 8. Testing/Optimization | 4-5 | Critical |
| **Total (MVP without Parallel)** | **22-29 weeks** | |
| **Total (Full Implementation)** | **25-33 weeks** | |

## Complete Test Suite Reference

The following is the complete list of HNSW tests from pgvector that must pass:

### SQL Regression Tests
Located in `references/pgvector/test/`:
- `sql/hnsw_vector.sql` → `expected/hnsw_vector.out` - Vector type operations
- `sql/hnsw_halfvec.sql` → `expected/hnsw_halfvec.out` - Half-precision vector operations
- `sql/hnsw_sparsevec.sql` → `expected/hnsw_sparsevec.out` - Sparse vector operations
- `sql/hnsw_bit.sql` → `expected/hnsw_bit.out` - Bit vector operations

### Perl Integration Tests
Located in `references/pgvector/test/t/`:
- `010_hnsw_wal.pl` - WAL logging and recovery
- `011_hnsw_vacuum.pl` - Vacuum operations
- `012_hnsw_vector_build_recall.pl` - Vector build recall
- `013_hnsw_vector_insert_recall.pl` - Vector insert recall
- `014_hnsw_vector_vacuum_recall.pl` - Vector vacuum recall
- `015_hnsw_vector_duplicates.pl` - Duplicate handling for vectors
- `016_hnsw_inserts.pl` - General insertion tests
- `017_hnsw_filtering.pl` - Filtering operations
- `020_hnsw_bit_build_recall.pl` - Bit vector build recall
- `021_hnsw_bit_insert_recall.pl` - Bit vector insert recall
- `022_hnsw_bit_vacuum_recall.pl` - Bit vector vacuum recall
- `023_hnsw_bit_duplicates.pl` - Duplicate handling for bit vectors
- `024_hnsw_halfvec_build_recall.pl` - Half-precision build recall
- `025_hnsw_halfvec_insert_recall.pl` - Half-precision insert recall
- `026_hnsw_halfvec_vacuum_recall.pl` - Half-precision vacuum recall
- `027_hnsw_halfvec_duplicates.pl` - Duplicate handling for halfvec
- `028_hnsw_sparsevec_build_recall.pl` - Sparse vector build recall
- `029_hnsw_sparsevec_insert_recall.pl` - Sparse vector insert recall
- `030_hnsw_sparsevec_vacuum_recall.pl` - Sparse vector vacuum recall
- `031_hnsw_sparsevec_duplicates.pl` - Duplicate handling for sparsevec
- `038_hnsw_sparsevec_vacuum_insert.pl` - Sparse vector vacuum and insert
- `039_hnsw_cost.pl` - Cost estimation
- `043_hnsw_iterative_scan.pl` - Iterative scan functionality
- `044_hnsw_iterative_scan_recall.pl` - Iterative scan recall

**Total: 4 SQL test suites + 24 Perl integration tests = 28 test files**

## Next Steps

1. Create GitHub Issues for each phase with detailed subtasks
2. Set up continuous integration for Rust code
3. Create regression test suite structure
4. Begin Phase 1 implementation

## References

- Original pgvector source: `references/pgvector/src/hnsw*`
- HNSW Paper: "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs" (Malkov & Yashunin, 2018)
- pgrx documentation: https://github.com/pgcentralfoundation/pgrx
- PostgreSQL Index AM documentation: https://www.postgresql.org/docs/current/indexam.html
