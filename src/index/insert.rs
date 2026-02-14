//! HNSW index insertion.

use pgrx::pg_guard;
use pgrx::pg_sys;

/// Insert a new tuple into the HNSW index.
///
/// This is the `aminsert` callback. Currently a stub that accepts but
/// ignores all insertions.
#[pg_guard]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn aminsert(
    _index_relation: pg_sys::Relation,
    _values: *mut pg_sys::Datum,
    _isnull: *mut bool,
    _heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // Stub: accept the tuple but don't actually index it yet
    false
}
