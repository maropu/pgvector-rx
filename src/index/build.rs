//! HNSW index building.

use pgrx::pg_guard;
use pgrx::pg_sys;

/// Build the HNSW index.
///
/// This is the `ambuild` callback. Currently a minimal stub that creates
/// an empty index and returns zero tuples.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    _heap_relation: pg_sys::Relation,
    _index_relation: pg_sys::Relation,
    _index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    // SAFETY: palloc0 zeroes all fields; heap_tuples and index_tuples
    // default to 0.0 which is correct for an empty build.
    pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
        as *mut pg_sys::IndexBuildResult
}

/// Build an empty HNSW index (for UNLOGGED tables).
///
/// This is the `ambuildempty` callback.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuildempty(_index_relation: pg_sys::Relation) {
    // Nothing to do for an empty build
}
