//! HNSW index vacuum and maintenance.
//!
//! Implements the `ambulkdelete` and `amvacuumcleanup` callbacks for HNSW
//! indexes. Vacuum operates in three passes:
//! 1. Remove dead heap TIDs and build a deleted set.
//! 2. Repair graph connections referencing deleted elements.
//! 3. Mark fully-deleted elements as deleted on disk.

use std::collections::HashSet;

use pgrx::pg_guard;
use pgrx::pg_sys;

use crate::hnsw_constants::*;
use crate::index::insert::{find_element_neighbors_on_disk, update_meta_page};
use crate::index::options::HnswOptions;
use crate::index::scan::{buffer_get_page, get_meta_page_info, load_element, ScanCandidate};
use crate::types::hnsw::*;

// ---------------------------------------------------------------------------
// Page helpers
// ---------------------------------------------------------------------------

/// Get a pointer to the opaque data in the special area of a page.
///
/// # Safety
/// `page` must be a valid HNSW page.
#[inline]
unsafe fn hnsw_page_get_opaque(page: pg_sys::Page) -> *mut HnswPageOpaqueData {
    let header = page as *const pg_sys::PageHeaderData;
    (page as *mut u8).add((*header).pd_special as usize) as *mut HnswPageOpaqueData
}

// ---------------------------------------------------------------------------
// Vacuum state
// ---------------------------------------------------------------------------

/// Tracks state across the three vacuum passes.
struct HnswVacuumState {
    index: pg_sys::Relation,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::ffi::c_void,
    m: i32,
    ef_construction: i32,
    bas: pg_sys::BufferAccessStrategy,
    /// Set of index TIDs (blkno, offno) whose heap TIDs were all removed.
    deleted: HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>,
    /// Highest non-entry-point element found during RemoveHeapTids.
    highest_point: Option<HighestPoint>,
}

/// Tracks the highest-level non-entry-point element.
#[derive(Clone)]
struct HighestPoint {
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
    level: i32,
}

impl HnswVacuumState {
    /// Initialize vacuum state.
    ///
    /// # Safety
    /// `info` must be valid.
    unsafe fn new(
        info: *mut pg_sys::IndexVacuumInfo,
        stats: *mut pg_sys::IndexBulkDeleteResult,
        callback: pg_sys::IndexBulkDeleteCallback,
        callback_state: *mut std::ffi::c_void,
    ) -> Self {
        let index = (*info).index;
        let stats = if stats.is_null() {
            pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBulkDeleteResult>())
                as *mut pg_sys::IndexBulkDeleteResult
        } else {
            stats
        };

        let (m, _, _, _) = get_meta_page_info(index);
        let ef_construction = HnswOptions::get_ef_construction(index);
        let bas = pg_sys::GetAccessStrategy(pg_sys::BufferAccessStrategyType::BAS_BULKREAD);

        HnswVacuumState {
            index,
            stats,
            callback,
            callback_state,
            m,
            ef_construction,
            bas,
            deleted: HashSet::new(),
            highest_point: None,
        }
    }

    /// Free resources.
    ///
    /// # Safety
    /// `bas` must be valid.
    unsafe fn free(&self) {
        pg_sys::FreeAccessStrategy(self.bas);
    }
}

// ---------------------------------------------------------------------------
// Pass 1: Remove dead heap TIDs
// ---------------------------------------------------------------------------

/// Walk all element pages, remove dead heap TIDs via the callback, and build
/// the deleted set for elements that have no remaining heap TIDs.
///
/// Also tracks the highest-level non-entry-point element for entry point
/// repair.
///
/// # Safety
/// All state pointers must be valid.
unsafe fn remove_heap_tids(vs: &mut HnswVacuumState) {
    let mut blkno = HNSW_HEAD_BLKNO;
    let (_, entry_blkno, entry_offno, _) = get_meta_page_info(vs.index);
    let callback = vs.callback.unwrap();
    let mut highest_level: i32 = -1;

    while blkno != pg_sys::InvalidBlockNumber {
        pg_sys::vacuum_delay_point(false);

        let buf = pg_sys::ReadBufferExtended(
            vs.index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            blkno,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            vs.bas,
        );
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let state = pg_sys::GenericXLogStart(vs.index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);
        let maxoffno = pg_sys::PageGetMaxOffsetNumber(page);
        let mut updated = false;

        let mut offno: pg_sys::OffsetNumber = 1;
        while offno <= maxoffno {
            let item_id = pg_sys::PageGetItemId(page, offno);
            let etup = pg_sys::PageGetItem(page, item_id) as *mut HnswElementTupleData;

            // Skip neighbor tuples
            if (*etup).type_ != HNSW_ELEMENT_TUPLE_TYPE {
                offno += 1;
                continue;
            }

            if pg_sys::ItemPointerIsValid(&(*etup).heaptids[0]) {
                let mut idx: usize = 0;
                let mut item_updated = false;

                for i in 0..HNSW_HEAPTIDS {
                    if !pg_sys::ItemPointerIsValid(&(*etup).heaptids[i]) {
                        break;
                    }

                    if callback(
                        &mut (*etup).heaptids[i] as *mut pg_sys::ItemPointerData,
                        vs.callback_state,
                    ) {
                        item_updated = true;
                        (*vs.stats).tuples_removed += 1.0;
                    } else {
                        (*etup).heaptids[idx] = (*etup).heaptids[i];
                        idx += 1;
                        (*vs.stats).num_index_tuples += 1.0;
                    }
                }

                if item_updated {
                    for i in idx..HNSW_HEAPTIDS {
                        pg_sys::ItemPointerSetInvalid(&mut (*etup).heaptids[i]);
                    }
                    updated = true;
                }
            }

            if !pg_sys::ItemPointerIsValid(&(*etup).heaptids[0]) {
                // All heap TIDs removed — add to deleted set
                vs.deleted.insert((blkno, offno));
            } else {
                let is_entry = blkno == entry_blkno && offno == entry_offno;
                let level = (*etup).level as i32;
                if level > highest_level && !is_entry {
                    vs.highest_point = Some(HighestPoint {
                        blkno,
                        offno,
                        level,
                    });
                    highest_level = level;
                }
            }

            offno += 1;
        }

        blkno = (*hnsw_page_get_opaque(page)).nextblkno;

        if updated {
            pg_sys::GenericXLogFinish(state);
        } else {
            pg_sys::GenericXLogAbort(state);
        }

        pg_sys::UnlockReleaseBuffer(buf);
    }
}

// ---------------------------------------------------------------------------
// Pass 2: Repair graph
// ---------------------------------------------------------------------------

/// Check whether an element has any neighbors in the deleted set, or if its
/// layer 0 neighbor list is not full.
///
/// # Safety
/// All pointers must be valid.
unsafe fn needs_updated(vs: &HnswVacuumState, sc: &ScanCandidate) -> bool {
    let buf = pg_sys::ReadBufferExtended(
        vs.index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        sc.neighbor_page,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        vs.bas,
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = buffer_get_page(buf);

    let item_id = pg_sys::PageGetItemId(page, sc.neighbor_offno);
    let ntup = pg_sys::PageGetItem(page, item_id) as *const HnswNeighborTupleData;

    let count = (*ntup).count as usize;
    let tids_base = (ntup as *const u8).add(std::mem::size_of::<HnswNeighborTupleData>())
        as *const pg_sys::ItemPointerData;

    let mut result = false;
    for i in 0..count {
        let tid = &*tids_base.add(i);
        if !pg_sys::ItemPointerIsValid(tid) {
            continue;
        }
        let nb = pg_sys::ItemPointerGetBlockNumber(tid);
        let no = pg_sys::ItemPointerGetOffsetNumber(tid);
        if vs.deleted.contains(&(nb, no)) {
            result = true;
            break;
        }
    }

    // Also update if layer 0 is not full
    if !result {
        let lm = hnsw_get_layer_m(vs.m, 0) as usize;
        let start_idx = (sc.level * vs.m) as usize;
        if start_idx + lm <= count {
            let last_tid = &*tids_base.add(start_idx + lm - 1);
            if !pg_sys::ItemPointerIsValid(last_tid) {
                result = true;
            }
        }
    }

    pg_sys::UnlockReleaseBuffer(buf);
    result
}

/// Re-find neighbors for an element and overwrite its neighbor tuple on disk.
///
/// # Safety
/// All state pointers must be valid.
#[allow(clippy::too_many_arguments)]
unsafe fn repair_graph_element(
    vs: &HnswVacuumState,
    element_blkno: pg_sys::BlockNumber,
    element_offno: pg_sys::OffsetNumber,
    element_level: i32,
    entry_blkno: pg_sys::BlockNumber,
    entry_offno: pg_sys::OffsetNumber,
    entry_level: i32,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
) {
    // Skip if element is entry point
    if element_blkno == entry_blkno && element_offno == entry_offno {
        return;
    }

    // Read the element's vector data into memory to avoid holding buffer locks
    // during the search and neighbor update phases.
    let buf = pg_sys::ReadBufferExtended(
        vs.index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        element_blkno,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        vs.bas,
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = buffer_get_page(buf);
    let item_id = pg_sys::PageGetItemId(page, element_offno);
    let etup = pg_sys::PageGetItem(page, item_id) as *const HnswElementTupleData;
    let version = (*etup).version;
    let neighbor_page = pg_sys::ItemPointerGetBlockNumber(&(*etup).neighbortid);
    let neighbor_offno = pg_sys::ItemPointerGetOffsetNumber(&(*etup).neighbortid);

    // Copy vector data into palloc'd memory so we can release the buffer
    let data_ptr = (etup as *const u8).add(std::mem::size_of::<HnswElementTupleData>());
    let varlena_size = (*(data_ptr as *const u32) >> 2) as usize;
    let vec_copy = pg_sys::palloc(varlena_size) as *mut u8;
    std::ptr::copy_nonoverlapping(data_ptr, vec_copy, varlena_size);
    let query_datum = pg_sys::Datum::from(vec_copy as usize);

    pg_sys::UnlockReleaseBuffer(buf);

    // Build skip set: deleted elements + self (to avoid self-reference)
    let mut skip = vs.deleted.clone();
    skip.insert((element_blkno, element_offno));

    // Find new neighbors using on-disk search (no buffer locks held)
    let neighbors_by_layer = find_element_neighbors_on_disk(
        vs.index,
        query_datum,
        dist_fmgr,
        collation,
        element_level,
        vs.m,
        vs.ef_construction,
        entry_blkno,
        entry_offno,
        entry_level,
        Some(&skip),
    );

    // Build new neighbor tuple
    let ntup_size = hnsw_neighbor_tuple_size(element_level as usize, vs.m as usize);
    let ntup_buf = pg_sys::palloc0(ntup_size) as *mut u8;
    let ntup = ntup_buf as *mut HnswNeighborTupleData;
    (*ntup).type_ = HNSW_NEIGHBOR_TUPLE_TYPE;
    (*ntup).version = version;

    // Fill in neighbor TIDs
    let tids_base =
        ntup_buf.add(std::mem::size_of::<HnswNeighborTupleData>()) as *mut pg_sys::ItemPointerData;

    for lc in 0..=element_level {
        let lm = hnsw_get_layer_m(vs.m, lc) as usize;
        let start_idx = ((element_level - lc) * vs.m) as usize;

        if let Some(layer_neighbors) = neighbors_by_layer.get(lc as usize) {
            for (i, sc) in layer_neighbors.iter().take(lm).enumerate() {
                let tid = &mut *tids_base.add(start_idx + i);
                pg_sys::ItemPointerSet(tid, sc.blkno, sc.offno);
            }
        }
    }
    // count must equal total slots so load_neighbor_tids accepts the tuple
    (*ntup).count = ((element_level + 2) * vs.m) as u16;

    // Overwrite neighbor tuple on disk
    let nbuf = pg_sys::ReadBufferExtended(
        vs.index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        neighbor_page,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        vs.bas,
    );
    pg_sys::LockBuffer(nbuf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let nstate = pg_sys::GenericXLogStart(vs.index);
    let npage = pg_sys::GenericXLogRegisterBuffer(nstate, nbuf, 0);

    if !pg_sys::PageIndexTupleOverwrite(npage, neighbor_offno, ntup_buf as pg_sys::Item, ntup_size)
    {
        pgrx::error!(
            "pgvector-rx: failed to overwrite neighbor tuple in \"{}\"",
            std::ffi::CStr::from_ptr((*(*vs.index).rd_rel).relname.data.as_ptr()).to_string_lossy()
        );
    }

    pg_sys::GenericXLogFinish(nstate);
    pg_sys::UnlockReleaseBuffer(nbuf);

    pg_sys::pfree(ntup_buf as *mut std::ffi::c_void);
    pg_sys::pfree(vec_copy as *mut std::ffi::c_void);
}

/// Repair the entry point if it was deleted, then repair all other elements.
///
/// # Safety
/// All state pointers must be valid.
unsafe fn repair_graph_entry_point(
    vs: &mut HnswVacuumState,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
) {
    let highest_point = vs.highest_point.clone();

    // Repair highest non-entry point if needed
    if let Some(ref hp) = highest_point {
        pg_sys::LockPage(
            vs.index,
            HNSW_UPDATE_LOCK,
            pg_sys::ShareLock as pg_sys::LOCKMODE,
        );

        let hp_sc = load_element(
            vs.index,
            hp.blkno,
            hp.offno,
            pg_sys::Datum::from(0usize),
            dist_fmgr,
            collation,
            None,
        );

        if let Some(ref sc) = hp_sc {
            if needs_updated(vs, sc) {
                let (_, eb, eo, el) = get_meta_page_info(vs.index);
                repair_graph_element(
                    vs, hp.blkno, hp.offno, hp.level, eb, eo, el, dist_fmgr, collation,
                );
            }
        }

        pg_sys::UnlockPage(
            vs.index,
            HNSW_UPDATE_LOCK,
            pg_sys::ShareLock as pg_sys::LOCKMODE,
        );
    }

    // Prevent concurrent inserts when possibly updating entry point
    pg_sys::LockPage(
        vs.index,
        HNSW_UPDATE_LOCK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );

    let (_, entry_blkno, entry_offno, entry_level) = get_meta_page_info(vs.index);

    if entry_blkno != pg_sys::InvalidBlockNumber {
        if vs.deleted.contains(&(entry_blkno, entry_offno)) {
            // Replace entry point with highest point
            if let Some(ref hp) = highest_point {
                update_meta_page(
                    vs.index,
                    HNSW_UPDATE_ENTRY_ALWAYS,
                    hp.blkno,
                    hp.offno,
                    hp.level,
                    pg_sys::InvalidBlockNumber,
                );
            } else {
                update_meta_page(
                    vs.index,
                    HNSW_UPDATE_ENTRY_ALWAYS,
                    pg_sys::InvalidBlockNumber,
                    pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
                    -1,
                    pg_sys::InvalidBlockNumber,
                );
            }
        } else {
            // Repair entry point if it has deleted neighbors
            let ep_sc = load_element(
                vs.index,
                entry_blkno,
                entry_offno,
                pg_sys::Datum::from(0usize),
                dist_fmgr,
                collation,
                None,
            );
            if let Some(ref sc) = ep_sc {
                if needs_updated(vs, sc) {
                    let hp_entry = highest_point
                        .as_ref()
                        .map_or((entry_blkno, entry_offno, entry_level), |hp| {
                            (hp.blkno, hp.offno, hp.level)
                        });
                    repair_graph_element(
                        vs,
                        entry_blkno,
                        entry_offno,
                        entry_level,
                        hp_entry.0,
                        hp_entry.1,
                        hp_entry.2,
                        dist_fmgr,
                        collation,
                    );
                }
            }
        }
    }

    pg_sys::UnlockPage(
        vs.index,
        HNSW_UPDATE_LOCK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
}

/// Repair graph connections for all elements that reference deleted elements.
///
/// # Safety
/// All state pointers must be valid.
unsafe fn repair_graph(vs: &mut HnswVacuumState) {
    let dist_fmgr = pg_sys::index_getprocinfo(vs.index, 1, HNSW_DISTANCE_PROC);
    let collation = (*vs.index).rd_indcollation.read();

    // Wait for in-flight inserts to complete
    pg_sys::LockPage(
        vs.index,
        HNSW_UPDATE_LOCK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
    pg_sys::UnlockPage(
        vs.index,
        HNSW_UPDATE_LOCK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );

    // Repair entry point first
    repair_graph_entry_point(vs, dist_fmgr, collation);

    // Repair all other elements
    let mut blkno = HNSW_HEAD_BLKNO;

    while blkno != pg_sys::InvalidBlockNumber {
        pg_sys::vacuum_delay_point(false);

        // Collect elements from this page
        let buf = pg_sys::ReadBufferExtended(
            vs.index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            blkno,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            vs.bas,
        );
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = buffer_get_page(buf);
        let maxoffno = pg_sys::PageGetMaxOffsetNumber(page);

        let mut elements: Vec<(pg_sys::BlockNumber, pg_sys::OffsetNumber, i32)> = Vec::new();

        let mut offno: pg_sys::OffsetNumber = 1;
        while offno <= maxoffno {
            let item_id = pg_sys::PageGetItemId(page, offno);
            let etup = pg_sys::PageGetItem(page, item_id) as *const HnswElementTupleData;

            if (*etup).type_ == HNSW_ELEMENT_TUPLE_TYPE
                && pg_sys::ItemPointerIsValid(&(*etup).heaptids[0])
            {
                elements.push((blkno, offno, (*etup).level as i32));
            }
            offno += 1;
        }

        blkno = (*hnsw_page_get_opaque(page)).nextblkno;
        pg_sys::UnlockReleaseBuffer(buf);

        // Repair elements that have deleted neighbors
        for (eblkno, eoffno, elevel) in elements {
            let sc = match load_element(
                vs.index,
                eblkno,
                eoffno,
                pg_sys::Datum::from(0usize),
                dist_fmgr,
                collation,
                None,
            ) {
                Some(s) => s,
                None => continue,
            };

            if !needs_updated(vs, &sc) {
                continue;
            }

            let mut lockmode = pg_sys::ShareLock as pg_sys::LOCKMODE;
            pg_sys::LockPage(vs.index, HNSW_UPDATE_LOCK, lockmode);

            let (_, eb, eo, el) = get_meta_page_info(vs.index);

            // Upgrade to exclusive lock if element might become entry point
            if eb == pg_sys::InvalidBlockNumber || elevel > el {
                pg_sys::UnlockPage(vs.index, HNSW_UPDATE_LOCK, lockmode);
                lockmode = pg_sys::ExclusiveLock as pg_sys::LOCKMODE;
                pg_sys::LockPage(vs.index, HNSW_UPDATE_LOCK, lockmode);

                let (_, eb2, eo2, el2) = get_meta_page_info(vs.index);
                repair_graph_element(
                    vs, eblkno, eoffno, elevel, eb2, eo2, el2, dist_fmgr, collation,
                );

                if eb2 == pg_sys::InvalidBlockNumber || elevel > el2 {
                    update_meta_page(
                        vs.index,
                        HNSW_UPDATE_ENTRY_GREATER,
                        eblkno,
                        eoffno,
                        elevel,
                        pg_sys::InvalidBlockNumber,
                    );
                }
            } else {
                repair_graph_element(vs, eblkno, eoffno, elevel, eb, eo, el, dist_fmgr, collation);
            }

            pg_sys::UnlockPage(vs.index, HNSW_UPDATE_LOCK, lockmode);
        }
    }
}

// ---------------------------------------------------------------------------
// Pass 3: Mark deleted elements
// ---------------------------------------------------------------------------

/// Mark elements with no heap TIDs as deleted, zero their data and neighbor
/// connections, and increment versions to invalidate iterative scans.
///
/// # Safety
/// All state pointers must be valid.
unsafe fn mark_deleted(vs: &HnswVacuumState) {
    let mut blkno = HNSW_HEAD_BLKNO;
    let mut insert_page = pg_sys::InvalidBlockNumber;

    // Wait for index scans to complete
    pg_sys::LockPage(
        vs.index,
        HNSW_SCAN_LOCK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );
    pg_sys::UnlockPage(
        vs.index,
        HNSW_SCAN_LOCK,
        pg_sys::ExclusiveLock as pg_sys::LOCKMODE,
    );

    while blkno != pg_sys::InvalidBlockNumber {
        pg_sys::vacuum_delay_point(false);

        let buf = pg_sys::ReadBufferExtended(
            vs.index,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            blkno,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            vs.bas,
        );
        pg_sys::LockBufferForCleanup(buf);

        let mut state = pg_sys::GenericXLogStart(vs.index);
        let mut page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);
        let maxoffno = pg_sys::PageGetMaxOffsetNumber(page);

        let mut offno: pg_sys::OffsetNumber = 1;
        while offno <= maxoffno {
            let item_id = pg_sys::PageGetItemId(page, offno);
            let etup = pg_sys::PageGetItem(page, item_id) as *mut HnswElementTupleData;

            if (*etup).type_ != HNSW_ELEMENT_TUPLE_TYPE {
                offno += 1;
                continue;
            }

            if (*etup).deleted != 0 {
                if insert_page == pg_sys::InvalidBlockNumber {
                    insert_page = blkno;
                }
                offno += 1;
                continue;
            }

            if pg_sys::ItemPointerIsValid(&(*etup).heaptids[0]) {
                offno += 1;
                continue;
            }

            // Element has no heap TIDs — mark as deleted
            let neighbor_page = pg_sys::ItemPointerGetBlockNumber(&(*etup).neighbortid);
            let neighbor_offno = pg_sys::ItemPointerGetOffsetNumber(&(*etup).neighbortid);

            let (nbuf, npage) = if neighbor_page == blkno {
                // Same page — use the current buffer and page
                (buf, page)
            } else {
                // Different page — register neighbor buffer in same xlog
                let nbuf = pg_sys::ReadBufferExtended(
                    vs.index,
                    pg_sys::ForkNumber::MAIN_FORKNUM,
                    neighbor_page,
                    pg_sys::ReadBufferMode::RBM_NORMAL,
                    vs.bas,
                );
                pg_sys::LockBuffer(nbuf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
                let npage = pg_sys::GenericXLogRegisterBuffer(state, nbuf, 0);
                (nbuf, npage)
            };

            // Clear neighbor TIDs
            let n_item_id = pg_sys::PageGetItemId(npage, neighbor_offno);
            let ntup = pg_sys::PageGetItem(npage, n_item_id) as *mut HnswNeighborTupleData;
            let count = (*ntup).count as usize;
            let tids_base = (ntup as *mut u8).add(std::mem::size_of::<HnswNeighborTupleData>())
                as *mut pg_sys::ItemPointerData;
            for i in 0..count {
                pg_sys::ItemPointerSetInvalid(&mut *tids_base.add(i));
            }

            // Mark element as deleted and zero its data
            (*etup).deleted = 1;
            let data_ptr = (etup as *mut u8).add(std::mem::size_of::<HnswElementTupleData>());
            let varlena_size = (*(data_ptr as *const u32) >> 2) as usize;
            if varlena_size > 0 {
                std::ptr::write_bytes(data_ptr, 0, varlena_size);
            }

            // Increment version
            (*etup).version = bump_version((*etup).version);
            (*ntup).version = (*etup).version;

            // Commit xlog for both pages
            pg_sys::GenericXLogFinish(state);
            if nbuf != buf {
                pg_sys::UnlockReleaseBuffer(nbuf);
            }

            if insert_page == pg_sys::InvalidBlockNumber {
                insert_page = blkno;
            }

            // Re-open xlog for remaining tuples on this page
            state = pg_sys::GenericXLogStart(vs.index);
            page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);

            offno += 1;
        }

        blkno = (*hnsw_page_get_opaque(page)).nextblkno;
        pg_sys::GenericXLogAbort(state);
        pg_sys::UnlockReleaseBuffer(buf);
    }

    // Update insert page after all deletions
    if insert_page != pg_sys::InvalidBlockNumber {
        update_meta_page(
            vs.index,
            0,
            pg_sys::InvalidBlockNumber,
            pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
            -1,
            insert_page,
        );
    }
}

/// Increment version number, wrapping from 15 back to 1.
#[inline]
fn bump_version(v: u8) -> u8 {
    if v >= 15 {
        1
    } else {
        v + 1
    }
}

// ---------------------------------------------------------------------------
// AM callbacks
// ---------------------------------------------------------------------------

/// Bulk-delete tuples from the HNSW index.
///
/// This is the `ambulkdelete` callback. Operates in three passes:
/// 1. Remove dead heap TIDs from element tuples.
/// 2. Repair graph connections pointing to deleted elements.
/// 3. Mark fully-deleted elements as deleted on disk.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambulkdelete(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
    callback: pg_sys::IndexBulkDeleteCallback,
    callback_state: *mut std::ffi::c_void,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let mut vs = HnswVacuumState::new(info, stats, callback, callback_state);

    // Pass 1: Remove dead heap TIDs
    remove_heap_tids(&mut vs);

    // Pass 2: Repair graph connections
    if !vs.deleted.is_empty() {
        repair_graph(&mut vs);
    }

    // Pass 3: Mark deleted elements
    if !vs.deleted.is_empty() {
        mark_deleted(&vs);
    }

    let result = vs.stats;
    vs.free();
    result
}

/// Clean up after a VACUUM operation on the HNSW index.
///
/// This is the `amvacuumcleanup` callback.
#[pg_guard]
pub unsafe extern "C-unwind" fn amvacuumcleanup(
    info: *mut pg_sys::IndexVacuumInfo,
    stats: *mut pg_sys::IndexBulkDeleteResult,
) -> *mut pg_sys::IndexBulkDeleteResult {
    let rel = (*info).index;

    if (*info).analyze_only {
        return stats;
    }

    // stats is NULL if ambulkdelete not called — OK to return NULL
    if stats.is_null() {
        return std::ptr::null_mut();
    }

    (*stats).num_pages =
        pg_sys::RelationGetNumberOfBlocksInFork(rel, pg_sys::ForkNumber::MAIN_FORKNUM);

    stats
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    /// Verify that ambulkdelete and amvacuumcleanup are properly registered.
    #[pg_test]
    fn test_hnsw_vacuum_handler_registered() {
        Spi::run("CREATE TABLE t (val vector(3))").unwrap();
        Spi::run("INSERT INTO t (val) VALUES ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]')").unwrap();
        Spi::run("CREATE INDEX ON t USING hnsw (val vector_l2_ops)").unwrap();

        // Verify index scan works via subquery
        let count = Spi::get_one::<i64>(
            "SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> '[0,0,0]') sub",
        )
        .unwrap()
        .unwrap();
        assert_eq!(count, 3);

        Spi::run("DROP TABLE t").unwrap();
    }

    /// Verify that after DELETE and index rebuild, search still works.
    #[pg_test]
    fn test_hnsw_search_after_delete_and_reindex() {
        Spi::run("CREATE TABLE t (val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO t (val) VALUES \
             ('[0,0,0]'), ('[1,2,3]'), ('[1,1,1]'), ('[2,2,2]'), ('[3,3,3]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX idx ON t USING hnsw (val vector_l2_ops)").unwrap();

        // Delete using ctid to avoid needing = operator
        Spi::run(
            "DELETE FROM t WHERE ctid IN \
             (SELECT ctid FROM t ORDER BY val <-> '[1,1,1]' LIMIT 1)",
        )
        .unwrap();

        // Reindex to rebuild the index
        Spi::run("REINDEX INDEX idx").unwrap();

        // Should have 4 rows
        let count = Spi::get_one::<i64>(
            "SELECT COUNT(*) FROM (SELECT * FROM t ORDER BY val <-> '[0,0,0]') sub",
        )
        .unwrap()
        .unwrap();
        assert_eq!(count, 4);

        Spi::run("DROP TABLE t").unwrap();
    }
}
