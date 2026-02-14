//! HNSW index scanning.

use pgrx::pg_guard;
use pgrx::pg_sys;

/// Begin an HNSW index scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index_relation: pg_sys::Relation,
    nkeys: std::ffi::c_int,
    norderbys: std::ffi::c_int,
) -> pg_sys::IndexScanDesc {
    // SAFETY: RelationGetIndexScan allocates a scan descriptor in the
    // current memory context.
    pg_sys::RelationGetIndexScan(index_relation, nkeys, norderbys)
}

/// Restart an HNSW index scan with new keys/orderbys.
#[pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    nkeys: std::ffi::c_int,
    orderbys: pg_sys::ScanKey,
    norderbys: std::ffi::c_int,
) {
    let _ = (scan, keys, nkeys, orderbys, norderbys);
    // Stub: will be implemented with actual search logic
}

/// Return the next matching tuple from an HNSW index scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    _scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    // Stub: no results yet
    false
}

/// End an HNSW index scan and release resources.
#[pg_guard]
pub unsafe extern "C-unwind" fn amendscan(_scan: pg_sys::IndexScanDesc) {
    // Stub: nothing to clean up yet
}
