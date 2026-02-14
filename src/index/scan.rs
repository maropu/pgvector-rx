//! HNSW index scanning.
//!
//! Implements on-disk k-NN search over HNSW indexes using Algorithm 5 from
//! the HNSW paper (multi-layer greedy search from entry point to ground layer).

use std::collections::{BinaryHeap, HashSet};

use pgrx::pg_guard;
use pgrx::pg_sys;

use crate::hnsw_constants::*;
use crate::index::options::{HnswIterativeScan, HNSW_ITERATIVE_SCAN, HNSW_MAX_SCAN_TUPLES};
use crate::types::hnsw::*;

// ---------------------------------------------------------------------------
// Buffer / page helpers (shared with build.rs; duplicated here to keep the
// module self-contained — a shared module can be extracted later).
// ---------------------------------------------------------------------------

/// `BufferGetPage(buffer)` equivalent.
///
/// # Safety
/// `buffer` must be a valid pinned buffer.
#[inline]
pub(crate) unsafe fn buffer_get_page(buffer: pg_sys::Buffer) -> pg_sys::Page {
    pg_sys::BufferBlocks.add((buffer as usize - 1) * pg_sys::BLCKSZ as usize) as pg_sys::Page
}

/// Get a pointer to the meta page data area (const).
///
/// # Safety
/// `page` must be a valid HNSW meta page.
#[inline]
pub(crate) unsafe fn hnsw_page_get_meta(page: pg_sys::Page) -> *const HnswMetaPageData {
    let header_size = std::mem::size_of::<pg_sys::PageHeaderData>();
    (page as *const u8).add(header_size) as *const HnswMetaPageData
}

/// Get a mutable pointer to the meta page data area.
///
/// # Safety
/// `page` must be a valid HNSW meta page with exclusive lock.
#[inline]
pub(crate) unsafe fn hnsw_page_get_meta_mut(page: pg_sys::Page) -> *mut HnswMetaPageData {
    let header_size = std::mem::size_of::<pg_sys::PageHeaderData>();
    (page as *mut u8).add(header_size) as *mut HnswMetaPageData
}

// ---------------------------------------------------------------------------
// Scan-time search candidate
// ---------------------------------------------------------------------------

/// A search candidate with distance, element block/offset, and loaded data.
#[derive(Debug, Clone)]
pub(crate) struct ScanCandidate {
    pub(crate) distance: f64,
    pub(crate) blkno: pg_sys::BlockNumber,
    pub(crate) offno: pg_sys::OffsetNumber,
    /// Element level (loaded from disk).
    pub(crate) level: i32,
    /// Neighbor page block number.
    pub(crate) neighbor_page: pg_sys::BlockNumber,
    /// Neighbor tuple offset.
    pub(crate) neighbor_offno: pg_sys::OffsetNumber,
    /// Heap TIDs for this element.
    pub(crate) heaptids: Vec<pg_sys::ItemPointerData>,
    /// Version of the element tuple.
    pub(crate) version: u8,
}

// Nearest-first ordering for BinaryHeap (min-heap).
pub(crate) struct NearestSC(ScanCandidate);

impl PartialEq for NearestSC {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}
impl Eq for NearestSC {}
impl PartialOrd for NearestSC {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for NearestSC {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .0
            .distance
            .partial_cmp(&self.0.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// Furthest-first ordering for BinaryHeap (max-heap).
struct FurthestSC(ScanCandidate);

impl PartialEq for FurthestSC {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}
impl Eq for FurthestSC {}
impl PartialOrd for FurthestSC {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for FurthestSC {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ---------------------------------------------------------------------------
// On-disk element / neighbor loading
// ---------------------------------------------------------------------------

/// Read the meta page and return `(m, entry_blkno, entry_offno, entry_level)`.
///
/// # Safety
/// `index` must be a valid, open index relation.
pub(crate) unsafe fn get_meta_page_info(
    index: pg_sys::Relation,
) -> (i32, pg_sys::BlockNumber, pg_sys::OffsetNumber, i32) {
    let buf = pg_sys::ReadBuffer(index, HNSW_METAPAGE_BLKNO);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = buffer_get_page(buf);
    let metap = hnsw_page_get_meta(page);

    if (*metap).magic_number != HNSW_MAGIC_NUMBER {
        pg_sys::UnlockReleaseBuffer(buf);
        pgrx::error!("pgvector-rx: hnsw index is not valid");
    }

    let m = (*metap).m as i32;
    let entry_blkno = (*metap).entry_blkno;
    let entry_offno = (*metap).entry_offno;
    let entry_level = (*metap).entry_level as i32;

    pg_sys::UnlockReleaseBuffer(buf);
    (m, entry_blkno, entry_offno, entry_level)
}

/// Load an element tuple from disk and compute its distance to the query.
///
/// Returns `None` if `max_distance` is set and the element is too far.
///
/// # Safety
/// `index` must be valid. `query_datum`, `dist_fmgr`, `collation` must be
/// valid for calling the distance function.
pub(crate) unsafe fn load_element(
    index: pg_sys::Relation,
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
    query_datum: pg_sys::Datum,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
    max_distance: Option<f64>,
) -> Option<ScanCandidate> {
    let buf = pg_sys::ReadBuffer(index, blkno);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = buffer_get_page(buf);

    let item_id = pg_sys::PageGetItemId(page, offno);
    let etup = pg_sys::PageGetItem(page, item_id) as *const HnswElementTupleData;

    if (*etup).type_ != HNSW_ELEMENT_TUPLE_TYPE || (*etup).deleted != 0 {
        pg_sys::UnlockReleaseBuffer(buf);
        return None;
    }

    // Compute distance
    let distance = if query_datum.value() == 0 {
        0.0
    } else {
        let data_ptr = (etup as *const u8).add(std::mem::size_of::<HnswElementTupleData>());
        let value_datum = pg_sys::Datum::from(data_ptr as usize);
        let result = pg_sys::FunctionCall2Coll(dist_fmgr, collation, query_datum, value_datum);
        f64::from_bits(result.value() as u64)
    };

    // Early exit if too far
    if let Some(max_dist) = max_distance {
        if distance >= max_dist {
            pg_sys::UnlockReleaseBuffer(buf);
            return None;
        }
    }

    // Load heap TIDs
    let mut heaptids = Vec::new();
    for i in 0..HNSW_HEAPTIDS {
        if !pg_sys::ItemPointerIsValid(&(*etup).heaptids[i]) {
            break;
        }
        heaptids.push((*etup).heaptids[i]);
    }

    let level = (*etup).level as i32;
    let version = (*etup).version;
    let neighbor_page = pg_sys::ItemPointerGetBlockNumber(&(*etup).neighbortid);
    let neighbor_offno = pg_sys::ItemPointerGetOffsetNumber(&(*etup).neighbortid);

    pg_sys::UnlockReleaseBuffer(buf);

    Some(ScanCandidate {
        distance,
        blkno,
        offno,
        level,
        neighbor_page,
        neighbor_offno,
        heaptids,
        version,
    })
}

/// Load neighbor TIDs from a neighbor tuple on disk for a given layer.
///
/// Returns `None` if the neighbor tuple has been deleted/replaced.
///
/// # Safety
/// `index` must be valid.
pub(crate) unsafe fn load_neighbor_tids(
    index: pg_sys::Relation,
    neighbor_page: pg_sys::BlockNumber,
    neighbor_offno: pg_sys::OffsetNumber,
    element_level: i32,
    element_version: u8,
    m: i32,
    layer: i32,
) -> Option<Vec<pg_sys::ItemPointerData>> {
    let lm = hnsw_get_layer_m(m, layer) as usize;

    let buf = pg_sys::ReadBuffer(index, neighbor_page);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = buffer_get_page(buf);

    let item_id = pg_sys::PageGetItemId(page, neighbor_offno);
    let ntup = pg_sys::PageGetItem(page, item_id) as *const HnswNeighborTupleData;

    // Verify tuple validity
    if (*ntup).version != element_version || (*ntup).count as i32 != (element_level + 2) * m {
        pg_sys::UnlockReleaseBuffer(buf);
        return None;
    }

    // Offset into the neighbor TID array for the requested layer
    let start = ((element_level - layer) * m) as usize;
    let tids_base = (ntup as *const u8).add(std::mem::size_of::<HnswNeighborTupleData>())
        as *const pg_sys::ItemPointerData;

    let mut tids = Vec::with_capacity(lm);
    for i in 0..lm {
        let tid = *tids_base.add(start + i);
        if !pg_sys::ItemPointerIsValid(&tid) {
            break;
        }
        tids.push(tid);
    }

    pg_sys::UnlockReleaseBuffer(buf);
    Some(tids)
}

// ---------------------------------------------------------------------------
// On-disk search layer (Algorithm 2)
// ---------------------------------------------------------------------------

/// Search a single HNSW layer on disk.
///
/// When `discarded` is `Some`, candidates that are rejected (too far once
/// we have `ef` results) are pushed into that heap for later iterative
/// scan resumption. When `visited` is `Some`, the caller-owned visited
/// set is used and updated (shared across resume iterations).
///
/// `add_entry_to_visited` controls whether entry points are added to the
/// visited set (true for initial search, false for resume).
///
/// # Safety
/// All index/query state must be valid.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn search_layer_disk(
    index: pg_sys::Relation,
    entry_points: Vec<ScanCandidate>,
    ef: usize,
    layer: i32,
    m: i32,
    query_datum: pg_sys::Datum,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
    visited: Option<&mut HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>>,
    mut discarded: Option<&mut BinaryHeap<NearestSC>>,
    add_entry_to_visited: bool,
    skip_count: Option<&HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>>,
) -> Vec<ScanCandidate> {
    // Use caller-supplied visited set or create a local one.
    let mut local_visited: HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)> =
        HashSet::with_capacity(ef * m as usize * 2);
    let visited = visited.unwrap_or(&mut local_visited);

    let mut candidates: BinaryHeap<NearestSC> = BinaryHeap::new();
    let mut results: BinaryHeap<FurthestSC> = BinaryHeap::new();
    let mut w_len: usize = 0;

    // Initialize with entry points
    for ep in entry_points {
        if add_entry_to_visited {
            visited.insert((ep.blkno, ep.offno));
        }
        let should_count = skip_count.is_none_or(|sc| !sc.contains(&(ep.blkno, ep.offno)));
        candidates.push(NearestSC(ep.clone()));
        results.push(FurthestSC(ep));
        if should_count {
            w_len += 1;
        }
    }

    while let Some(NearestSC(c)) = candidates.pop() {
        let f_dist = results.peek().map(|f| f.0.distance).unwrap_or(f64::MAX);
        if c.distance > f_dist {
            // Save rejected candidate for iterative scan
            if let Some(ref mut disc) = discarded {
                disc.push(NearestSC(c));
            }
            break;
        }

        // Load neighbor TIDs for this element at the current layer
        let neighbor_tids = match load_neighbor_tids(
            index,
            c.neighbor_page,
            c.neighbor_offno,
            c.level,
            c.version,
            m,
            layer,
        ) {
            Some(tids) => tids,
            None => continue,
        };

        for tid in &neighbor_tids {
            let n_blkno = pg_sys::ItemPointerGetBlockNumber(tid);
            let n_offno = pg_sys::ItemPointerGetOffsetNumber(tid);

            if visited.contains(&(n_blkno, n_offno)) {
                continue;
            }
            visited.insert((n_blkno, n_offno));

            let always_add = w_len < ef;
            let f_dist = results.peek().map(|f| f.0.distance).unwrap_or(f64::MAX);

            // Load element
            let e = match load_element(
                index,
                n_blkno,
                n_offno,
                query_datum,
                dist_fmgr,
                collation,
                if always_add { None } else { Some(f_dist) },
            ) {
                Some(e) => e,
                None => {
                    // For iterative scan, load without distance filter to
                    // collect as discarded
                    if !always_add {
                        if let Some(ref mut disc) = discarded {
                            if let Some(e_full) = load_element(
                                index,
                                n_blkno,
                                n_offno,
                                query_datum,
                                dist_fmgr,
                                collation,
                                None,
                            ) {
                                if e_full.level >= layer {
                                    disc.push(NearestSC(e_full));
                                }
                            }
                        }
                    }
                    continue;
                }
            };

            // Skip if element level is below current layer
            if e.level < layer {
                continue;
            }

            candidates.push(NearestSC(e.clone()));
            results.push(FurthestSC(e.clone()));
            let should_count = skip_count.is_none_or(|sc| !sc.contains(&(e.blkno, e.offno)));
            if should_count {
                w_len += 1;
            }

            if w_len > ef {
                let evicted = results.pop().unwrap();
                w_len -= 1;

                // Save evicted candidate for iterative scan
                if let Some(ref mut disc) = discarded {
                    disc.push(NearestSC(evicted.0));
                }
            }
        }
    }

    // Save remaining candidates to discarded heap
    if let Some(ref mut disc) = discarded {
        while let Some(NearestSC(c)) = candidates.pop() {
            disc.push(NearestSC(c));
        }
    }

    // Collect results sorted by distance (nearest last for pop-from-back)
    let mut result: Vec<ScanCandidate> = results.into_iter().map(|f| f.0).collect();
    result.sort_by(|a, b| {
        b.distance
            .partial_cmp(&a.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

/// Run the full multi-layer scan (Algorithm 5 from the HNSW paper).
///
/// When `iterative` is true, the caller-owned `visited` and `discarded`
/// are populated for later use by `resume_scan_items`.
///
/// # Safety
/// All index/query state must be valid.
#[allow(clippy::too_many_arguments)]
unsafe fn get_scan_items(
    index: pg_sys::Relation,
    query_datum: pg_sys::Datum,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
    ef_search: usize,
    visited: Option<&mut HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>>,
    discarded: Option<&mut BinaryHeap<NearestSC>>,
) -> (Vec<ScanCandidate>, i32) {
    let (m, entry_blkno, entry_offno, _entry_level) = get_meta_page_info(index);

    // Empty index
    if entry_blkno == pg_sys::InvalidBlockNumber {
        return (Vec::new(), m);
    }

    // Load entry point
    let ep = match load_element(
        index,
        entry_blkno,
        entry_offno,
        query_datum,
        dist_fmgr,
        collation,
        None,
    ) {
        Some(ep) => ep,
        None => return (Vec::new(), m),
    };

    let ep_level = ep.level;
    let mut ep_list = vec![ep];

    // Phase 1: greedy search from top layer down to layer 1
    for lc in (1..=ep_level).rev() {
        let w = search_layer_disk(
            index,
            ep_list,
            1,
            lc,
            m,
            query_datum,
            dist_fmgr,
            collation,
            None,
            None,
            true,
            None,
        );
        ep_list = if w.is_empty() {
            return (Vec::new(), m);
        } else {
            vec![w.into_iter().last().unwrap()]
        };
    }

    // Phase 2: search ground layer with ef_search
    let results = search_layer_disk(
        index,
        ep_list,
        ef_search,
        0,
        m,
        query_datum,
        dist_fmgr,
        collation,
        visited,
        discarded,
        true,
        None,
    );
    (results, m)
}

/// Resume an iterative scan by re-entering the ground layer with
/// discarded candidates as entry points.
///
/// # Safety
/// All index/query state must be valid.
#[allow(clippy::too_many_arguments)]
unsafe fn resume_scan_items(
    index: pg_sys::Relation,
    query_datum: pg_sys::Datum,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
    ef_search: usize,
    m: i32,
    visited: &mut HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>,
    discarded: &mut BinaryHeap<NearestSC>,
) -> Vec<ScanCandidate> {
    if discarded.is_empty() {
        return Vec::new();
    }

    // Get next batch of candidates
    let batch_size = ef_search;
    let mut ep: Vec<ScanCandidate> = Vec::with_capacity(batch_size);
    for _ in 0..batch_size {
        if discarded.is_empty() {
            break;
        }
        let NearestSC(sc) = discarded.pop().unwrap();
        ep.push(sc);
    }

    search_layer_disk(
        index,
        ep,
        batch_size,
        0,
        m,
        query_datum,
        dist_fmgr,
        collation,
        Some(visited),
        Some(discarded),
        false,
        None,
    )
}

// ---------------------------------------------------------------------------
// Scan state
// ---------------------------------------------------------------------------

/// Opaque scan state stored in `scan->opaque`.
struct HnswScanState {
    /// Whether this is the first call to gettuple.
    first: bool,
    /// Distance function FmgrInfo pointer.
    dist_fmgr: *mut pg_sys::FmgrInfo,
    /// Norm function FmgrInfo pointer (for cosine normalization), or null.
    norm_fmgr: *mut pg_sys::FmgrInfo,
    /// Type-specific normalize function.
    normalize_fn: crate::index::build::NormalizeFn,
    /// Collation for the distance function.
    collation: pg_sys::Oid,
    /// Result list sorted by distance (nearest last for pop).
    results: Vec<ScanCandidate>,
    /// M parameter from the meta page (needed for resume).
    m: i32,
    /// Query datum (saved for resume).
    query_datum: pg_sys::Datum,
    /// Visited set (shared across iterative scan iterations).
    visited: HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>,
    /// Discarded candidates heap for iterative scan.
    discarded: BinaryHeap<NearestSC>,
    /// Whether iterative scan state has been initialized.
    iterative_initialized: bool,
    /// Number of tuples returned so far (for max_scan_tuples limit).
    tuples: i64,
    /// Previous distance returned (for strict_order mode).
    previous_distance: f64,
    /// Current element being iterated (for multiple heap TIDs per element).
    current_element: Option<ScanCandidate>,
}

// ---------------------------------------------------------------------------
// AM callbacks
// ---------------------------------------------------------------------------

/// Begin an HNSW index scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambeginscan(
    index_relation: pg_sys::Relation,
    nkeys: std::ffi::c_int,
    norderbys: std::ffi::c_int,
) -> pg_sys::IndexScanDesc {
    // SAFETY: RelationGetIndexScan allocates a scan descriptor in the
    // current memory context.
    let scan = pg_sys::RelationGetIndexScan(index_relation, nkeys, norderbys);

    // Get distance function
    let dist_fmgr = pg_sys::index_getprocinfo(index_relation, 1, HNSW_DISTANCE_PROC);
    let collation = (*index_relation).rd_indcollation.read();

    // Get norm function (for cosine normalization), if present.
    let norm_fmgr = if (*index_relation)
        .rd_support
        .add(HNSW_NORM_PROC as usize - 1)
        .read()
        != pg_sys::InvalidOid
    {
        pg_sys::index_getprocinfo(index_relation, 1, HNSW_NORM_PROC)
    } else {
        std::ptr::null_mut()
    };

    // Get the type-specific normalize function.
    let normalize_fn = crate::index::build::get_normalize_fn(index_relation);

    let state = Box::new(HnswScanState {
        first: true,
        dist_fmgr,
        norm_fmgr,
        normalize_fn,
        collation,
        results: Vec::new(),
        m: 0,
        query_datum: pg_sys::Datum::from(0usize),
        visited: HashSet::new(),
        discarded: BinaryHeap::new(),
        iterative_initialized: false,
        tuples: 0,
        previous_distance: f64::NEG_INFINITY,
        current_element: None,
    });

    (*scan).opaque = Box::into_raw(state) as *mut std::ffi::c_void;

    scan
}

/// Restart an HNSW index scan with new keys/orderbys.
#[pg_guard]
pub unsafe extern "C-unwind" fn amrescan(
    scan: pg_sys::IndexScanDesc,
    keys: pg_sys::ScanKey,
    _nkeys: std::ffi::c_int,
    orderbys: pg_sys::ScanKey,
    _norderbys: std::ffi::c_int,
) {
    let so = &mut *((*scan).opaque as *mut HnswScanState);
    so.first = true;
    so.results.clear();
    so.visited.clear();
    so.discarded.clear();
    so.iterative_initialized = false;
    so.tuples = 0;
    so.previous_distance = f64::NEG_INFINITY;
    so.current_element = None;

    if !keys.is_null() && (*scan).numberOfKeys > 0 {
        std::ptr::copy_nonoverlapping(
            keys as *const u8,
            (*scan).keyData as *mut u8,
            (*scan).numberOfKeys as usize * std::mem::size_of::<pg_sys::ScanKeyData>(),
        );
    }

    if !orderbys.is_null() && (*scan).numberOfOrderBys > 0 {
        std::ptr::copy_nonoverlapping(
            orderbys as *const u8,
            (*scan).orderByData as *mut u8,
            (*scan).numberOfOrderBys as usize * std::mem::size_of::<pg_sys::ScanKeyData>(),
        );
    }
}

/// Return the next matching tuple from an HNSW index scan.
#[pg_guard]
pub unsafe extern "C-unwind" fn amgettuple(
    scan: pg_sys::IndexScanDesc,
    _direction: pg_sys::ScanDirection::Type,
) -> bool {
    let so = &mut *((*scan).opaque as *mut HnswScanState);

    let iterative_scan = HNSW_ITERATIVE_SCAN.get();

    if so.first {
        // Count index scan for stats (equivalent to pgstat_count_index_scan macro)
        {
            let rel = (*scan).indexRelation;
            if !(*rel).pgstat_info.is_null() {
                (*(*rel).pgstat_info).counts.numscans += 1;
            } else if (*rel).pgstat_enabled {
                pg_sys::pgstat_assoc_relation(rel);
                if !(*rel).pgstat_info.is_null() {
                    (*(*rel).pgstat_info).counts.numscans += 1;
                }
            }
        }

        // Safety check: HNSW requires ORDER BY
        if (*scan).orderByData.is_null() {
            pgrx::error!("cannot scan hnsw index without order");
        }

        // Requires MVCC-compliant snapshot
        let snapshot = (*scan).xs_snapshot;
        if snapshot.is_null() || (*snapshot).snapshot_type != pg_sys::SnapshotType::SNAPSHOT_MVCC {
            pgrx::error!("non-MVCC snapshots are not supported with hnsw");
        }

        // Get query value from orderByData
        let order_by = &*(*scan).orderByData;
        let query_datum = if order_by.sk_flags & pg_sys::SK_ISNULL as i32 != 0 {
            pg_sys::Datum::from(0usize)
        } else {
            let value = order_by.sk_argument;
            // Normalize the query vector for cosine distance
            if !so.norm_fmgr.is_null() && value.value() != 0 {
                let normalized = (so.normalize_fn)(value.cast_mut_ptr() as *const pg_sys::varlena);
                pg_sys::Datum::from(normalized as usize)
            } else {
                value
            }
        };

        // Get ef_search GUC value
        let ef_search = crate::index::options::HNSW_EF_SEARCH.get() as usize;

        // Determine if we need iterative scan state
        let use_iterative = iterative_scan != HnswIterativeScan::Off;

        // Run the HNSW search
        let (results, m) = if use_iterative {
            get_scan_items(
                (*scan).indexRelation,
                query_datum,
                so.dist_fmgr,
                so.collation,
                ef_search,
                Some(&mut so.visited),
                Some(&mut so.discarded),
            )
        } else {
            get_scan_items(
                (*scan).indexRelation,
                query_datum,
                so.dist_fmgr,
                so.collation,
                ef_search,
                None,
                None,
            )
        };

        so.results = results;
        so.m = m;
        so.query_datum = query_datum;
        so.iterative_initialized = use_iterative;
        so.first = false;
    }

    // Return next result
    loop {
        // First, check if we have remaining heap TIDs from current element
        if let Some(ref mut current) = so.current_element {
            if !current.heaptids.is_empty() {
                let heaptid = current.heaptids.pop().unwrap();

                // Strict ordering check
                if iterative_scan == HnswIterativeScan::StrictOrder {
                    if current.distance < so.previous_distance {
                        continue;
                    }
                    so.previous_distance = current.distance;
                }

                (*scan).xs_heaptid = heaptid;
                (*scan).xs_recheck = false;
                (*scan).xs_recheckorderby = false;
                return true;
            }
            // All heap TIDs consumed, clear current element
            so.current_element = None;
        }

        if so.results.is_empty() {
            if iterative_scan == HnswIterativeScan::Off {
                return false;
            }

            // Empty index (no discarded state initialized)
            if !so.iterative_initialized {
                return false;
            }

            let ef_search = crate::index::options::HNSW_EF_SEARCH.get() as usize;
            let max_scan_tuples = HNSW_MAX_SCAN_TUPLES.get() as i64;

            // Reached max number of tuples
            if so.tuples >= max_scan_tuples {
                if so.discarded.is_empty() {
                    return false;
                }

                // Return remaining tuples one at a time from discarded
                if let Some(NearestSC(sc)) = so.discarded.pop() {
                    so.results.push(sc);
                } else {
                    return false;
                }
            } else {
                // Resume scan with discarded candidates
                so.results = resume_scan_items(
                    (*scan).indexRelation,
                    so.query_datum,
                    so.dist_fmgr,
                    so.collation,
                    ef_search,
                    so.m,
                    &mut so.visited,
                    &mut so.discarded,
                );
            }

            if so.results.is_empty() {
                return false;
            }
        }

        let sc = match so.results.pop() {
            Some(sc) => sc,
            None => return false,
        };

        // Skip elements with no valid heap TIDs
        if sc.heaptids.is_empty() {
            continue;
        }

        so.tuples += 1;

        // Set as current element to iterate through all heap TIDs
        so.current_element = Some(sc);
    }
}

/// End an HNSW index scan and release resources.
#[pg_guard]
pub unsafe extern "C-unwind" fn amendscan(scan: pg_sys::IndexScanDesc) {
    if !(*scan).opaque.is_null() {
        // SAFETY: We allocated this with Box::new in ambeginscan.
        let _ = Box::from_raw((*scan).opaque as *mut HnswScanState);
        (*scan).opaque = std::ptr::null_mut();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hnsw_scan_basic_l2() {
        Spi::run("CREATE TABLE test_scan (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_scan (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'), \
             ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX test_scan_idx ON test_scan \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Query for nearest neighbor to [1,0,0]
        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_scan \
             ORDER BY val <-> '[1,0,0]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        assert_eq!(result, "[1,0,0]");
    }

    #[pg_test]
    fn test_hnsw_scan_top_k() {
        Spi::run("CREATE TABLE test_scan2 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_scan2 (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'), \
             ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX test_scan2_idx ON test_scan2 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Query for top 3 nearest neighbors to [1,0,0]
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_scan2 \
                ORDER BY val <-> '[1,0,0]' LIMIT 3\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 3);
    }

    #[pg_test]
    fn test_hnsw_scan_empty_index() {
        Spi::run("CREATE TABLE test_scan3 (id serial, val vector(3))").unwrap();
        Spi::run(
            "CREATE INDEX test_scan3_idx ON test_scan3 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Query on empty index should return no rows
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_scan3 \
                ORDER BY val <-> '[1,0,0]' LIMIT 1\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 0);
    }

    #[pg_test]
    fn test_hnsw_scan_cosine() {
        Spi::run("CREATE TABLE test_scan4 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_scan4 (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX test_scan4_idx ON test_scan4 \
             USING hnsw (val vector_cosine_ops)",
        )
        .unwrap();

        // Query for nearest neighbor using cosine distance
        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_scan4 \
             ORDER BY val <=> '[1,0,0]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        assert_eq!(result, "[1,0,0]");
    }

    #[pg_test]
    fn test_hnsw_scan_ip() {
        Spi::run("CREATE TABLE test_scan5 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_scan5 (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX test_scan5_idx ON test_scan5 \
             USING hnsw (val vector_ip_ops)",
        )
        .unwrap();

        // Query for nearest neighbor using inner product
        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_scan5 \
             ORDER BY val <#> '[1,0,0]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        assert_eq!(result, "[1,0,0]");
    }

    #[pg_test]
    fn test_hnsw_scan_ef_search_guc() {
        Spi::run("CREATE TABLE test_scan6 (id serial, val vector(3))").unwrap();
        for i in 0..20 {
            let x = (i as f64 * 0.3).sin() as f32;
            let y = (i as f64 * 0.5).cos() as f32;
            let z = (i as f64 * 0.1) as f32 / 10.0;
            Spi::run(&format!(
                "INSERT INTO test_scan6 (val) VALUES ('[{},{},{}]')",
                x, y, z
            ))
            .unwrap();
        }
        Spi::run(
            "CREATE INDEX test_scan6_idx ON test_scan6 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Set ef_search and verify we still get results
        Spi::run("SET hnsw.ef_search = 10").unwrap();
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_scan6 \
                ORDER BY val <-> '[0,0,0]' LIMIT 5\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 5);
    }

    #[pg_test]
    fn test_hnsw_scan_with_nulls() {
        Spi::run("CREATE TABLE test_scan7 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_scan7 (val) VALUES \
             ('[1,0,0]'), (NULL), ('[0,1,0]'), (NULL), ('[0,0,1]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX test_scan7_idx ON test_scan7 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Should return 3 results (NULLs not indexed)
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_scan7 \
                WHERE val IS NOT NULL \
                ORDER BY val <-> '[1,0,0]' LIMIT 10\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 3);
    }

    #[pg_test]
    fn test_hnsw_scan_recall() {
        // Build a dataset where we know the exact nearest neighbors
        Spi::run("CREATE TABLE test_recall (id serial, val vector(2))").unwrap();

        // Insert points on a grid
        for i in 0..10 {
            for j in 0..10 {
                Spi::run(&format!(
                    "INSERT INTO test_recall (val) VALUES ('[{},{}]')",
                    i as f32, j as f32
                ))
                .unwrap();
            }
        }

        Spi::run(
            "CREATE INDEX test_recall_idx ON test_recall \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Query for nearest to [0,0] — should be [0,0] itself
        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_recall \
             ORDER BY val <-> '[0,0]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        assert_eq!(result, "[0,0]");

        // Query for top 5 — should include [0,0], [1,0], [0,1], [1,1] and
        // one of [2,0] or [0,2] (same distance)
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_recall \
                ORDER BY val <-> '[0,0]' LIMIT 5\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 5);
    }

    #[pg_test]
    fn test_hnsw_iterative_strict_order() {
        Spi::run("CREATE TABLE test_iter1 (val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_iter1 (val) VALUES \
             ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX ON test_iter1 USING hnsw (val vector_l2_ops)").unwrap();

        Spi::run("SET hnsw.iterative_scan = strict_order").unwrap();
        Spi::run("SET hnsw.ef_search = 1").unwrap();

        // Should still return all 3 results via iterative scanning
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_iter1 \
                ORDER BY val <-> '[3,3,3]'\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 3);

        Spi::run("RESET hnsw.iterative_scan").unwrap();
        Spi::run("RESET hnsw.ef_search").unwrap();
    }

    #[pg_test]
    fn test_hnsw_iterative_relaxed_order() {
        Spi::run("CREATE TABLE test_iter2 (val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_iter2 (val) VALUES \
             ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX ON test_iter2 USING hnsw (val vector_l2_ops)").unwrap();

        Spi::run("SET hnsw.iterative_scan = relaxed_order").unwrap();

        // Should return all 3 results
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_iter2 \
                ORDER BY val <-> '[3,3,3]'\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 3);

        Spi::run("RESET hnsw.iterative_scan").unwrap();
    }

    #[pg_test]
    fn test_hnsw_iterative_empty_table() {
        Spi::run("CREATE TABLE test_iter3 (val vector(3))").unwrap();
        Spi::run("CREATE INDEX ON test_iter3 USING hnsw (val vector_l2_ops)").unwrap();

        Spi::run("SET hnsw.iterative_scan = strict_order").unwrap();

        // Empty table should return 0 rows
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_iter3 \
                ORDER BY val <-> '[3,3,3]'\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 0);

        Spi::run("RESET hnsw.iterative_scan").unwrap();
    }

    #[pg_test]
    fn test_hnsw_iterative_after_truncate() {
        Spi::run("CREATE TABLE test_iter4 (val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_iter4 (val) VALUES \
             ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX ON test_iter4 USING hnsw (val vector_l2_ops)").unwrap();

        Spi::run("TRUNCATE test_iter4").unwrap();
        Spi::run("SET hnsw.iterative_scan = strict_order").unwrap();

        // Truncated table should return 0 rows
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM (\
                SELECT val FROM test_iter4 \
                ORDER BY val <-> '[3,3,3]'\
             ) t",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 0);

        Spi::run("RESET hnsw.iterative_scan").unwrap();
    }
}
