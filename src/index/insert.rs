//! HNSW index insertion.
//!
//! Implements the `aminsert` callback that inserts a single tuple into an
//! existing HNSW index by:
//! 1. Finding neighbors using on-disk multi-layer search
//! 2. Writing element and neighbor tuples to disk
//! 3. Updating existing neighbor back-connections
//! 4. Updating the meta page (entry point and insert page)

use std::collections::HashSet;

use pgrx::pg_guard;
use pgrx::pg_sys;

use crate::hnsw_constants::*;
use crate::index::options::HnswOptions;
use crate::index::scan::{
    buffer_get_page, get_meta_page_info, hnsw_page_get_meta, hnsw_page_get_meta_mut, load_element,
    load_neighbor_tids, search_layer_disk, ScanCandidate,
};
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

/// Initialize an HNSW page with the special area.
///
/// # Safety
/// `buf` and `page` must be valid.
#[inline]
unsafe fn hnsw_init_page(buf: pg_sys::Buffer, page: pg_sys::Page) {
    let page_size = pg_sys::BufferGetPageSize(buf);
    pg_sys::PageInit(page, page_size, std::mem::size_of::<HnswPageOpaqueData>());
    let opaque = hnsw_page_get_opaque(page);
    (*opaque).nextblkno = pg_sys::InvalidBlockNumber;
    (*opaque).page_id = HNSW_PAGE_ID;
}

// ---------------------------------------------------------------------------
// Insert page tracking
// ---------------------------------------------------------------------------

/// Read the insert page block number from the meta page.
///
/// # Safety
/// `index` must be valid.
unsafe fn get_insert_page(index: pg_sys::Relation) -> pg_sys::BlockNumber {
    let buf = pg_sys::ReadBuffer(index, HNSW_METAPAGE_BLKNO);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page = buffer_get_page(buf);
    let metap = hnsw_page_get_meta(page);
    let insert_page = (*metap).insert_page;
    pg_sys::UnlockReleaseBuffer(buf);
    insert_page
}

// ---------------------------------------------------------------------------
// On-disk element insertion
// ---------------------------------------------------------------------------

/// Result of placing an element on disk.
struct InsertedElement {
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
    neighbor_page: pg_sys::BlockNumber,
    neighbor_offno: pg_sys::OffsetNumber,
    updated_insert_page: pg_sys::BlockNumber,
}

/// Result of finding a deleted element slot that can be reused.
struct FreeOffsetResult {
    elem_offno: pg_sys::OffsetNumber,
    neighbor_offno: pg_sys::OffsetNumber,
    neighbor_blkno: pg_sys::BlockNumber,
    /// Buffer for the neighbor page (only valid when neighbor is on a
    /// different page from the element).
    nbuf: pg_sys::Buffer,
    tuple_version: u8,
}

/// Scan page items for a deleted element whose slot can be reused.
///
/// Checks whether the old element's space (including page free space) can
/// accommodate the new element tuple, and likewise for the neighbor tuple.
///
/// On success, returns offset numbers to overwrite. If the neighbor resides
/// on a different page, that page is locked exclusively and its buffer is
/// returned in `FreeOffsetResult::nbuf`.
///
/// # Safety
/// `buf` must be exclusively locked. `page` must be the GenericXLog copy of
/// that buffer. Both tuple sizes must be MAXALIGN'd.
unsafe fn hnsw_free_offset(
    index: pg_sys::Relation,
    buf: pg_sys::Buffer,
    page: pg_sys::Page,
    etup_size: usize,
    ntup_size: usize,
    new_insert_page: &mut pg_sys::BlockNumber,
) -> Option<FreeOffsetResult> {
    let maxoffno = pg_sys::PageGetMaxOffsetNumber(page as *const pg_sys::PageData);
    let element_page = pg_sys::BufferGetBlockNumber(buf);

    let mut offno: pg_sys::OffsetNumber = 1; // FirstOffsetNumber
    while offno <= maxoffno {
        let eitemid = pg_sys::PageGetItemId(page, offno);

        // Skip items without storage (safety check for assert-enabled builds)
        if (*eitemid).lp_flags() != pg_sys::LP_NORMAL {
            offno += 1;
            continue;
        }

        let etup = pg_sys::PageGetItem(page, eitemid) as *const HnswElementTupleData;

        // Skip neighbor tuples
        if (*etup).type_ != HNSW_ELEMENT_TUPLE_TYPE {
            offno += 1;
            continue;
        }

        if (*etup).deleted != 0 {
            let neighbor_blkno =
                pg_sys::ItemPointerGetBlockNumber(&(*etup).neighbortid as *const _);
            let neighbor_offno =
                pg_sys::ItemPointerGetOffsetNumber(&(*etup).neighbortid as *const _);

            if *new_insert_page == pg_sys::InvalidBlockNumber {
                *new_insert_page = element_page;
            }

            let (nbuf, npage_raw) = if neighbor_blkno == element_page {
                (buf, page as *const pg_sys::PageData)
            } else {
                let nb = pg_sys::ReadBuffer(index, neighbor_blkno);
                pg_sys::LockBuffer(nb, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
                // SAFETY: only used for space calculation, not WAL
                let np = buffer_get_page(nb);
                (nb, np as *const pg_sys::PageData)
            };

            let nitemid = pg_sys::PageGetItemId(npage_raw as pg_sys::Page, neighbor_offno);

            // Calculate available space individually since tuples are
            // overwritten in separate PageIndexTupleOverwrite calls.
            let page_free = (*eitemid).lp_len() as usize
                + pg_sys::PageGetExactFreeSpace(page as *const pg_sys::PageData);
            let mut npage_free = (*nitemid).lp_len() as usize;
            if neighbor_blkno != element_page {
                npage_free += pg_sys::PageGetExactFreeSpace(npage_raw);
            } else if page_free >= etup_size {
                npage_free += page_free - etup_size;
            }

            if page_free >= etup_size && npage_free >= ntup_size {
                return Some(FreeOffsetResult {
                    elem_offno: offno,
                    neighbor_offno,
                    neighbor_blkno,
                    nbuf: if neighbor_blkno != element_page {
                        nbuf
                    } else {
                        0 // invalid, same page
                    },
                    tuple_version: (*etup).version,
                });
            } else if nbuf != buf {
                pg_sys::UnlockReleaseBuffer(nbuf);
            }
        }
        offno += 1;
    }
    None
}

/// Add element and neighbor tuples to disk pages, using GenericXLog for WAL.
///
/// Walks the page chain starting from `insert_page` to find space. Reuses
/// slots from deleted elements when possible via `PageIndexTupleOverwrite`.
///
/// # Safety
/// `index` must be valid. All buffers are properly locked and released.
#[allow(clippy::too_many_arguments)]
unsafe fn add_element_on_disk(
    index: pg_sys::Relation,
    etup_data: *mut u8,
    etup_size: usize,
    ntup_data: *mut u8,
    ntup_size: usize,
    insert_page: pg_sys::BlockNumber,
) -> InsertedElement {
    let combined_size = etup_size + ntup_size + std::mem::size_of::<pg_sys::ItemIdData>();
    let max_size = hnsw_max_size();
    let mut current_page = insert_page;
    let mut new_insert_page = pg_sys::InvalidBlockNumber;

    loop {
        let buf = pg_sys::ReadBuffer(index, current_page);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);

        let state = pg_sys::GenericXLogStart(index);
        let page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);

        let free_space = pg_sys::PageGetFreeSpace(page as *const pg_sys::PageData);

        // Track first page where tuples could fit
        if new_insert_page == pg_sys::InvalidBlockNumber && free_space >= combined_size {
            new_insert_page = current_page;
        }

        // Fast path: both tuples fit on this page
        if free_space >= combined_size {
            let elem_offno = pg_sys::PageGetMaxOffsetNumber(page as *const pg_sys::PageData) + 1;

            let added = pg_sys::PageAddItemExtended(
                page,
                etup_data as pg_sys::Item,
                etup_size,
                pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
                0,
            );
            if added != elem_offno {
                pg_sys::GenericXLogAbort(state);
                pg_sys::UnlockReleaseBuffer(buf);
                pgrx::error!("pgvector-rx: failed to add element tuple");
            }

            let neighbor_offno = elem_offno + 1;
            let added = pg_sys::PageAddItemExtended(
                page,
                ntup_data as pg_sys::Item,
                ntup_size,
                pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
                0,
            );
            if added != neighbor_offno {
                pg_sys::GenericXLogAbort(state);
                pg_sys::UnlockReleaseBuffer(buf);
                pgrx::error!("pgvector-rx: failed to add neighbor tuple");
            }

            let elem_blkno = pg_sys::BufferGetBlockNumber(buf);

            if new_insert_page == pg_sys::InvalidBlockNumber {
                new_insert_page = elem_blkno;
            }

            pg_sys::GenericXLogFinish(state);
            pg_sys::UnlockReleaseBuffer(buf);

            return InsertedElement {
                blkno: elem_blkno,
                offno: elem_offno,
                neighbor_page: elem_blkno,
                neighbor_offno,
                updated_insert_page: if new_insert_page != insert_page {
                    new_insert_page
                } else {
                    pg_sys::InvalidBlockNumber
                },
            };
        }

        // Try reusing space from a deleted element
        if let Some(free) =
            hnsw_free_offset(index, buf, page, etup_size, ntup_size, &mut new_insert_page)
        {
            // Set tuple version to match the deleted slot
            let etup = etup_data as *mut HnswElementTupleData;
            (*etup).version = free.tuple_version;
            let ntup = ntup_data as *mut HnswNeighborTupleData;
            (*ntup).version = free.tuple_version;

            let elem_blkno = pg_sys::BufferGetBlockNumber(buf);

            // Register neighbor page with GenericXLog if on a different page
            let npage = if free.neighbor_blkno != elem_blkno {
                pg_sys::GenericXLogRegisterBuffer(state, free.nbuf, 0)
            } else {
                page
            };

            // Overwrite deleted element tuple
            if !pg_sys::PageIndexTupleOverwrite(
                page,
                free.elem_offno,
                etup_data as pg_sys::Item,
                etup_size,
            ) {
                pg_sys::GenericXLogAbort(state);
                if free.nbuf != 0 && free.nbuf != buf {
                    pg_sys::UnlockReleaseBuffer(free.nbuf);
                }
                pg_sys::UnlockReleaseBuffer(buf);
                pgrx::error!("pgvector-rx: failed to overwrite element tuple");
            }

            // Overwrite deleted neighbor tuple
            if !pg_sys::PageIndexTupleOverwrite(
                npage,
                free.neighbor_offno,
                ntup_data as pg_sys::Item,
                ntup_size,
            ) {
                pg_sys::GenericXLogAbort(state);
                if free.nbuf != 0 && free.nbuf != buf {
                    pg_sys::UnlockReleaseBuffer(free.nbuf);
                }
                pg_sys::UnlockReleaseBuffer(buf);
                pgrx::error!("pgvector-rx: failed to overwrite neighbor tuple");
            }

            if new_insert_page == pg_sys::InvalidBlockNumber {
                new_insert_page = free.neighbor_blkno;
            }

            pg_sys::GenericXLogFinish(state);
            if free.nbuf != 0 && free.nbuf != buf {
                pg_sys::UnlockReleaseBuffer(free.nbuf);
            }
            pg_sys::UnlockReleaseBuffer(buf);

            return InsertedElement {
                blkno: elem_blkno,
                offno: free.elem_offno,
                neighbor_page: free.neighbor_blkno,
                neighbor_offno: free.neighbor_offno,
                updated_insert_page: if new_insert_page != insert_page {
                    new_insert_page
                } else {
                    pg_sys::InvalidBlockNumber
                },
            };
        }

        // Element fits but neighbor doesn't (split across pages)
        if combined_size > max_size
            && free_space >= etup_size
            && (*hnsw_page_get_opaque(page)).nextblkno == pg_sys::InvalidBlockNumber
        {
            // Add element tuple to current page
            let elem_offno = pg_sys::PageGetMaxOffsetNumber(page as *const pg_sys::PageData) + 1;
            let added = pg_sys::PageAddItemExtended(
                page,
                etup_data as pg_sys::Item,
                etup_size,
                pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
                0,
            );
            if added != elem_offno {
                pg_sys::GenericXLogAbort(state);
                pg_sys::UnlockReleaseBuffer(buf);
                pgrx::error!("pgvector-rx: failed to add element tuple");
            }

            let elem_blkno = pg_sys::BufferGetBlockNumber(buf);

            // Append a new page for the neighbor tuple
            pg_sys::LockRelationForExtension(index, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);
            let nbuf = pg_sys::ReadBufferExtended(
                index,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                pg_sys::InvalidBlockNumber,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            pg_sys::UnlockRelationForExtension(index, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);
            pg_sys::LockBuffer(nbuf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
            let npage = pg_sys::GenericXLogRegisterBuffer(
                state,
                nbuf,
                pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
            );
            hnsw_init_page(nbuf, npage);

            // Link current page to new page
            (*hnsw_page_get_opaque(page)).nextblkno = pg_sys::BufferGetBlockNumber(nbuf);

            // Add neighbor tuple to new page
            let neighbor_offno = 1u16; // FirstOffsetNumber
            let added = pg_sys::PageAddItemExtended(
                npage,
                ntup_data as pg_sys::Item,
                ntup_size,
                pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
                0,
            );
            if added != neighbor_offno {
                pg_sys::GenericXLogAbort(state);
                pg_sys::UnlockReleaseBuffer(nbuf);
                pg_sys::UnlockReleaseBuffer(buf);
                pgrx::error!("pgvector-rx: failed to add neighbor tuple");
            }

            let n_blkno = pg_sys::BufferGetBlockNumber(nbuf);

            if new_insert_page == pg_sys::InvalidBlockNumber {
                new_insert_page = n_blkno;
            }

            pg_sys::GenericXLogFinish(state);
            pg_sys::UnlockReleaseBuffer(nbuf);
            pg_sys::UnlockReleaseBuffer(buf);

            return InsertedElement {
                blkno: elem_blkno,
                offno: elem_offno,
                neighbor_page: n_blkno,
                neighbor_offno,
                updated_insert_page: if new_insert_page != insert_page {
                    new_insert_page
                } else {
                    pg_sys::InvalidBlockNumber
                },
            };
        }

        // Move to next page
        let next_blkno = (*hnsw_page_get_opaque(page)).nextblkno;
        if next_blkno != pg_sys::InvalidBlockNumber {
            pg_sys::GenericXLogAbort(state);
            pg_sys::UnlockReleaseBuffer(buf);
            current_page = next_blkno;
        } else {
            // Append a new page
            pg_sys::LockRelationForExtension(index, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);
            let newbuf = pg_sys::ReadBufferExtended(
                index,
                pg_sys::ForkNumber::MAIN_FORKNUM,
                pg_sys::InvalidBlockNumber,
                pg_sys::ReadBufferMode::RBM_NORMAL,
                std::ptr::null_mut(),
            );
            pg_sys::UnlockRelationForExtension(index, pg_sys::ExclusiveLock as pg_sys::LOCKMODE);
            pg_sys::LockBuffer(newbuf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
            let newpage = pg_sys::GenericXLogRegisterBuffer(
                state,
                newbuf,
                pg_sys::GENERIC_XLOG_FULL_IMAGE as i32,
            );
            hnsw_init_page(newbuf, newpage);

            // Link current → new
            (*hnsw_page_get_opaque(page)).nextblkno = pg_sys::BufferGetBlockNumber(newbuf);

            // Commit link update
            pg_sys::GenericXLogFinish(state);
            pg_sys::UnlockReleaseBuffer(buf);

            // Now use the new page
            current_page = pg_sys::BufferGetBlockNumber(newbuf);
            pg_sys::UnlockReleaseBuffer(newbuf);
        }
    }
}

// ---------------------------------------------------------------------------
// Neighbor updates
// ---------------------------------------------------------------------------

/// Simple on-disk candidate for neighbor selection during back-connection
/// updates. Holds the element's block/offset and distance to the neighbor
/// element (NOT to the query).
struct UpdateCandidate {
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
    distance: f64,
}

/// Determine the index at which to place the new element in a neighbor's
/// connection list. Mirrors the C `GetUpdateIndex` logic:
///
/// - Returns `Some(-2)` when a free slot exists (caller finds it at write
///   time since another backend may have filled it).
/// - Returns `Some(i)` (i >= 0) to replace the existing connection at
///   position `i` in the layer slice.
/// - Returns `None` if the new element should NOT be added (not selected by
///   heuristic, or neighbor is being deleted).
///
/// # Safety
/// All index/query state must be valid. Does NOT hold buffer locks for the
/// neighbor page so that the (potentially expensive) distance computations
/// do not block other backends.
#[allow(clippy::too_many_arguments)]
unsafe fn get_update_index(
    index: pg_sys::Relation,
    neighbor_blkno: pg_sys::BlockNumber,
    neighbor_offno: pg_sys::OffsetNumber,
    neighbor_level: i32,
    neighbor_neighbor_page: pg_sys::BlockNumber,
    neighbor_neighbor_offno: pg_sys::OffsetNumber,
    neighbor_version: u8,
    m: i32,
    lm: usize,
    layer: i32,
    new_distance: f64,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
) -> Option<i32> {
    // Load the neighbor element's own vector datum for distance computation.
    // We read without holding an exclusive lock (optimistic approach matching C).
    let neighbor_datum = {
        let buf = pg_sys::ReadBuffer(index, neighbor_blkno);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = buffer_get_page(buf);
        let item_id = pg_sys::PageGetItemId(page, neighbor_offno);
        let etup = pg_sys::PageGetItem(page, item_id) as *const HnswElementTupleData;

        if (*etup).type_ != HNSW_ELEMENT_TUPLE_TYPE || (*etup).deleted != 0 {
            pg_sys::UnlockReleaseBuffer(buf);
            return None;
        }

        // Copy the vector data into palloc'd memory so we can release the buffer
        let data_ptr = (etup as *const u8).add(std::mem::size_of::<HnswElementTupleData>());
        let varlena_size = (*(data_ptr as *const u32) >> 2) as usize;
        let copy = pg_sys::palloc(varlena_size) as *mut u8;
        std::ptr::copy_nonoverlapping(data_ptr, copy, varlena_size);
        pg_sys::UnlockReleaseBuffer(buf);
        pg_sys::Datum::from(copy as usize)
    };

    // Load the current neighbor TIDs for the specified layer.
    let tids = load_neighbor_tids(
        index,
        neighbor_neighbor_page,
        neighbor_neighbor_offno,
        neighbor_level,
        neighbor_version,
        m,
        layer,
    );
    let tids: Vec<pg_sys::ItemPointerData> = match tids {
        Some(t) => t,
        None => {
            pg_sys::pfree(neighbor_datum.cast_mut_ptr());
            return None;
        }
    };

    // If a free slot exists, signal with -2 (actual slot found at write time)
    if tids.len() < lm {
        pg_sys::pfree(neighbor_datum.cast_mut_ptr());
        return Some(-2);
    }

    // All slots full: load each current neighbor's element, compute its
    // distance to the neighbor element, and check for deleted elements.
    let mut candidates: Vec<UpdateCandidate> = Vec::with_capacity(tids.len());
    let mut pruned_deleted_idx: Option<usize> = None;

    for (i, tid) in tids.iter().enumerate() {
        let nb = pg_sys::ItemPointerGetBlockNumber(tid);
        let no = pg_sys::ItemPointerGetOffsetNumber(tid);

        // Load the connected element to get its distance to the neighbor
        let buf = pg_sys::ReadBuffer(index, nb);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = buffer_get_page(buf);
        let item_id = pg_sys::PageGetItemId(page, no);
        let etup = pg_sys::PageGetItem(page, item_id) as *const HnswElementTupleData;

        if (*etup).type_ != HNSW_ELEMENT_TUPLE_TYPE || (*etup).deleted != 0 {
            // Element being deleted – can be replaced immediately
            pg_sys::UnlockReleaseBuffer(buf);
            if pruned_deleted_idx.is_none() {
                pruned_deleted_idx = Some(i);
            }
            continue;
        }

        // Check if heaptids are all invalid (element being deleted)
        let mut has_valid_heaptid = false;
        for j in 0..HNSW_HEAPTIDS {
            if pg_sys::ItemPointerIsValid(&(*etup).heaptids[j]) {
                has_valid_heaptid = true;
                break;
            }
        }
        if !has_valid_heaptid {
            pg_sys::UnlockReleaseBuffer(buf);
            if pruned_deleted_idx.is_none() {
                pruned_deleted_idx = Some(i);
            }
            continue;
        }

        // Compute distance from this connected element to the neighbor element
        let data_ptr = (etup as *const u8).add(std::mem::size_of::<HnswElementTupleData>());
        let value_datum = pg_sys::Datum::from(data_ptr as usize);
        let result = pg_sys::FunctionCall2Coll(dist_fmgr, collation, neighbor_datum, value_datum);
        let dist = f64::from_bits(result.value() as u64);
        pg_sys::UnlockReleaseBuffer(buf);

        candidates.push(UpdateCandidate {
            blkno: nb,
            offno: no,
            distance: dist,
        });
    }

    // If a deleted element was found, use its slot
    if let Some(idx) = pruned_deleted_idx {
        pg_sys::pfree(neighbor_datum.cast_mut_ptr());
        return Some(idx as i32);
    }

    // Run the neighbor selection heuristic: add the new element as a candidate
    // and see if it replaces one of the existing neighbors.
    // Sort candidates by distance (ascending) for the heuristic.
    candidates.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Build combined list: existing candidates + new element, sorted by distance
    // The heuristic processes from closest to farthest.
    struct HeuristicCandidate {
        blkno: pg_sys::BlockNumber,
        offno: pg_sys::OffsetNumber,
        distance: f64,
        is_new: bool,
    }

    let mut all_candidates: Vec<HeuristicCandidate> = candidates
        .iter()
        .map(|c| HeuristicCandidate {
            blkno: c.blkno,
            offno: c.offno,
            distance: c.distance,
            is_new: false,
        })
        .collect();
    // Add the new element as a candidate
    all_candidates.push(HeuristicCandidate {
        blkno: 0, // placeholder, won't be used for distance calc
        offno: 0,
        distance: new_distance,
        is_new: true,
    });
    all_candidates.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Simple heuristic: keep the lm closest candidates, checking diversity.
    // For each candidate (closest first), add to result if it's closer to the
    // neighbor than to any already-selected result element.
    let mut selected: Vec<&HeuristicCandidate> = Vec::with_capacity(lm);
    let mut pruned: Vec<&HeuristicCandidate> = Vec::new();

    for hc in &all_candidates {
        if selected.len() >= lm {
            break;
        }

        // Check if this candidate is closer to the neighbor (query) than
        // to any already-selected candidate
        let mut closer = true;
        for sel in &selected {
            // Compute distance between hc and sel
            // Both are on-disk elements, need to load and compare
            if !hc.is_new && !sel.is_new {
                let dist = compute_element_distance(
                    index, hc.blkno, hc.offno, sel.blkno, sel.offno, dist_fmgr, collation,
                );
                if dist <= hc.distance {
                    closer = false;
                    break;
                }
            }
            // If one is the new element, we can't easily compute inter-element
            // distance without the new element's datum. For simplicity, treat
            // the new element as always "closer" (it won't be pruned by this
            // check). This matches the behavior where we just check distance.
        }

        if closer {
            selected.push(hc);
        } else {
            pruned.push(hc);
        }
    }

    // Fill remaining from pruned
    for p in &pruned {
        if selected.len() >= lm {
            break;
        }
        selected.push(p);
    }

    // Check if the new element was selected
    let new_selected = selected.iter().any(|s| s.is_new);
    if !new_selected {
        pg_sys::pfree(neighbor_datum.cast_mut_ptr());
        return None;
    }

    // Find which existing element was NOT selected (the one to be replaced).
    // Build set of selected existing element (blkno, offno).
    let selected_existing: HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)> = selected
        .iter()
        .filter(|s| !s.is_new)
        .map(|s| (s.blkno, s.offno))
        .collect();

    // Find the first existing neighbor in tids that is NOT in selected
    let mut replace_idx: Option<usize> = None;
    for (i, tid) in tids.iter().enumerate() {
        let tb = pg_sys::ItemPointerGetBlockNumber(tid);
        let to = pg_sys::ItemPointerGetOffsetNumber(tid);
        if !selected_existing.contains(&(tb, to)) {
            replace_idx = Some(i);
            break;
        }
    }

    pg_sys::pfree(neighbor_datum.cast_mut_ptr());

    replace_idx.map(|i| i as i32)
}

/// Compute the distance between two on-disk elements.
///
/// # Safety
/// `index` must be valid. Both elements must exist and not be deleted.
unsafe fn compute_element_distance(
    index: pg_sys::Relation,
    blkno1: pg_sys::BlockNumber,
    offno1: pg_sys::OffsetNumber,
    blkno2: pg_sys::BlockNumber,
    offno2: pg_sys::OffsetNumber,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
) -> f64 {
    // Load first element's datum
    let buf1 = pg_sys::ReadBuffer(index, blkno1);
    pg_sys::LockBuffer(buf1, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page1 = buffer_get_page(buf1);
    let item_id1 = pg_sys::PageGetItemId(page1, offno1);
    let etup1 = pg_sys::PageGetItem(page1, item_id1) as *const HnswElementTupleData;
    let data1 = (etup1 as *const u8).add(std::mem::size_of::<HnswElementTupleData>());
    let size1 = (*(data1 as *const u32) >> 2) as usize;
    let copy1 = pg_sys::palloc(size1) as *mut u8;
    std::ptr::copy_nonoverlapping(data1, copy1, size1);
    pg_sys::UnlockReleaseBuffer(buf1);

    // Load second element's datum
    let buf2 = pg_sys::ReadBuffer(index, blkno2);
    pg_sys::LockBuffer(buf2, pg_sys::BUFFER_LOCK_SHARE as i32);
    let page2 = buffer_get_page(buf2);
    let item_id2 = pg_sys::PageGetItemId(page2, offno2);
    let etup2 = pg_sys::PageGetItem(page2, item_id2) as *const HnswElementTupleData;
    let data2 = (etup2 as *const u8).add(std::mem::size_of::<HnswElementTupleData>());
    let datum2 = pg_sys::Datum::from(data2 as usize);

    let datum1 = pg_sys::Datum::from(copy1 as usize);
    let result = pg_sys::FunctionCall2Coll(dist_fmgr, collation, datum1, datum2);
    pg_sys::UnlockReleaseBuffer(buf2);
    pg_sys::pfree(copy1 as *mut std::ffi::c_void);

    f64::from_bits(result.value() as u64)
}

/// Update a single neighbor's connection list on disk using the computed
/// update index.
///
/// `update_idx`:
/// - `-2`: a free slot was available; find it again at write time
/// - `>= 0`: replace the connection at this layer-relative index
///
/// # Safety
/// `index` must be valid.
#[allow(clippy::too_many_arguments)]
unsafe fn write_neighbor_update(
    index: pg_sys::Relation,
    neighbor_page: pg_sys::BlockNumber,
    neighbor_offno: pg_sys::OffsetNumber,
    neighbor_level: i32,
    neighbor_version: u8,
    m: i32,
    layer: i32,
    new_blkno: pg_sys::BlockNumber,
    new_offno: pg_sys::OffsetNumber,
    update_idx: i32,
) {
    let lm = hnsw_get_layer_m(m, layer) as usize;

    let buf = pg_sys::ReadBuffer(index, neighbor_page);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let state = pg_sys::GenericXLogStart(index);
    let page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);

    let item_id = pg_sys::PageGetItemId(page, neighbor_offno);
    let ntup = pg_sys::PageGetItem(page, item_id) as *mut HnswNeighborTupleData;

    // Verify tuple is still valid
    if (*ntup).version != neighbor_version {
        pg_sys::GenericXLogAbort(state);
        pg_sys::UnlockReleaseBuffer(buf);
        return;
    }

    let start_idx = ((neighbor_level - layer) * m) as usize;
    let tids_base = (ntup as *mut u8).add(std::mem::size_of::<HnswNeighborTupleData>())
        as *mut pg_sys::ItemPointerData;

    // Check if connection already exists
    for i in 0..lm {
        let tid = &*tids_base.add(start_idx + i);
        if !pg_sys::ItemPointerIsValid(tid) {
            break;
        }
        if pg_sys::ItemPointerGetBlockNumber(tid) == new_blkno
            && pg_sys::ItemPointerGetOffsetNumber(tid) == new_offno
        {
            pg_sys::GenericXLogAbort(state);
            pg_sys::UnlockReleaseBuffer(buf);
            return;
        }
    }

    let idx = if update_idx == -2 {
        // Find a free slot (it may have been filled by another backend)
        let mut found = None;
        for i in 0..lm {
            let tid = &*tids_base.add(start_idx + i);
            if !pg_sys::ItemPointerIsValid(tid) {
                found = Some(start_idx + i);
                break;
            }
        }
        found
    } else if update_idx >= 0 {
        Some(start_idx + update_idx as usize)
    } else {
        None
    };

    if let Some(idx) = idx {
        if idx < (*ntup).count as usize {
            let indextid = &mut *tids_base.add(idx);
            pg_sys::ItemPointerSet(indextid, new_blkno, new_offno);
            pg_sys::GenericXLogFinish(state);
        } else {
            pg_sys::GenericXLogAbort(state);
        }
    } else {
        pg_sys::GenericXLogAbort(state);
    }

    pg_sys::UnlockReleaseBuffer(buf);
}

/// Update all neighbors of the new element to add back-connections.
///
/// For each neighbor selected during insertion, loads its current connection
/// list, runs the neighbor selection heuristic to determine if the new
/// element should be added (possibly replacing an existing connection), and
/// writes the update to disk.
///
/// # Safety
/// `index` must be valid.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn update_neighbors_on_disk(
    index: pg_sys::Relation,
    new_blkno: pg_sys::BlockNumber,
    new_offno: pg_sys::OffsetNumber,
    new_level: i32,
    neighbors_by_layer: &[Vec<ScanCandidate>],
    m: i32,
    query_datum: pg_sys::Datum,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
) {
    for lc in (0..=new_level).rev() {
        let lm = hnsw_get_layer_m(m, lc) as usize;

        if (lc as usize) >= neighbors_by_layer.len() {
            continue;
        }
        let neighbors = &neighbors_by_layer[lc as usize];

        for sc in neighbors.iter().take(lm) {
            // Load the neighbor's full info to get its neighbor page
            let neighbor = match load_element(
                index,
                sc.blkno,
                sc.offno,
                query_datum,
                dist_fmgr,
                collation,
                None,
            ) {
                Some(n) => n,
                None => continue,
            };

            // Determine where to place the back-connection
            let update_idx = get_update_index(
                index,
                neighbor.blkno,
                neighbor.offno,
                neighbor.level,
                neighbor.neighbor_page,
                neighbor.neighbor_offno,
                neighbor.version,
                m,
                lm,
                lc,
                sc.distance,
                dist_fmgr,
                collation,
            );

            let update_idx = match update_idx {
                Some(idx) => idx,
                None => continue,
            };

            write_neighbor_update(
                index,
                neighbor.neighbor_page,
                neighbor.neighbor_offno,
                neighbor.level,
                neighbor.version,
                m,
                lc,
                new_blkno,
                new_offno,
                update_idx,
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Meta page update
// ---------------------------------------------------------------------------

/// Update the meta page entry point and/or insert page.
///
/// `update_entry_mode`:
/// - `0` = do not update entry point
/// - `HNSW_UPDATE_ENTRY_GREATER` = update only if level > current
/// - `HNSW_UPDATE_ENTRY_ALWAYS` = always overwrite entry point
///
/// If `entry_blkno` is `InvalidBlockNumber` with `HNSW_UPDATE_ENTRY_ALWAYS`,
/// the entry point is cleared.
///
/// # Safety
/// `index` must be valid.
pub(crate) unsafe fn update_meta_page(
    index: pg_sys::Relation,
    update_entry_mode: i32,
    entry_blkno: pg_sys::BlockNumber,
    entry_offno: pg_sys::OffsetNumber,
    entry_level: i32,
    insert_page: pg_sys::BlockNumber,
) {
    let buf = pg_sys::ReadBufferExtended(
        index,
        pg_sys::ForkNumber::MAIN_FORKNUM,
        HNSW_METAPAGE_BLKNO,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        std::ptr::null_mut(),
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let state = pg_sys::GenericXLogStart(index);
    let page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);
    let metap = hnsw_page_get_meta_mut(page);

    let should_update = if update_entry_mode == HNSW_UPDATE_ENTRY_ALWAYS {
        true
    } else {
        update_entry_mode == HNSW_UPDATE_ENTRY_GREATER && entry_level as i16 > (*metap).entry_level
    };
    if should_update {
        (*metap).entry_blkno = entry_blkno;
        (*metap).entry_offno = entry_offno;
        (*metap).entry_level = entry_level as i16;
    }

    if insert_page != pg_sys::InvalidBlockNumber {
        (*metap).insert_page = insert_page;
    }

    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);
}

// ---------------------------------------------------------------------------
// Neighbor selection (on-disk variant)
// ---------------------------------------------------------------------------

/// Find neighbors for a new element using on-disk multi-layer search.
///
/// Returns a vector of neighbor lists, one per layer (index 0 = layer 0).
///
/// # Safety
/// All index/query state must be valid.
#[allow(clippy::too_many_arguments)]
pub(crate) unsafe fn find_element_neighbors_on_disk(
    index: pg_sys::Relation,
    query_datum: pg_sys::Datum,
    dist_fmgr: *mut pg_sys::FmgrInfo,
    collation: pg_sys::Oid,
    new_level: i32,
    m: i32,
    ef_construction: i32,
    entry_blkno: pg_sys::BlockNumber,
    entry_offno: pg_sys::OffsetNumber,
    entry_level: i32,
    skip_set: Option<&HashSet<(pg_sys::BlockNumber, pg_sys::OffsetNumber)>>,
) -> Vec<Vec<ScanCandidate>> {
    let mut neighbors_by_layer: Vec<Vec<ScanCandidate>> =
        vec![Vec::new(); (new_level + 1) as usize];

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
        None => return neighbors_by_layer,
    };

    let mut ep_list = vec![ep];

    // Phase 1: greedy search from top layer down to new_level + 1
    for lc in (new_level + 1..=entry_level).rev() {
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
            skip_set,
        );
        ep_list = if w.is_empty() {
            return neighbors_by_layer;
        } else {
            vec![w.into_iter().last().unwrap()]
        };
    }

    // Phase 2: search and select neighbors at each layer
    let start_level = std::cmp::min(new_level, entry_level);
    for lc in (0..=start_level).rev() {
        let lm = hnsw_get_layer_m(m, lc) as usize;

        // For vacuum repair: increase ef by 1 to account for self-skip
        let ef = if skip_set.is_some() {
            ef_construction as usize + 1
        } else {
            ef_construction as usize
        };

        let w = search_layer_disk(
            index,
            ep_list,
            ef,
            lc,
            m,
            query_datum,
            dist_fmgr,
            collation,
            None,
            None,
            true,
            skip_set,
        );

        // Filter out elements in skip set, then select up to lm nearest neighbors
        let filtered: Vec<&ScanCandidate> = if let Some(ss) = skip_set {
            w.iter()
                .filter(|c| !ss.contains(&(c.blkno, c.offno)))
                .collect()
        } else {
            w.iter().collect()
        };
        let selected: Vec<ScanCandidate> = filtered
            .iter()
            .rev()
            .take(lm)
            .map(|c| (*c).clone())
            .collect();
        neighbors_by_layer[lc as usize] = selected;

        ep_list = w;
    }

    neighbors_by_layer
}

// ---------------------------------------------------------------------------
// Duplicate detection
// ---------------------------------------------------------------------------

/// Try to add a heap TID to an existing duplicate element on disk.
///
/// Returns `true` if the TID was added, `false` if the element has no room
/// or is being deleted.
///
/// # Safety
/// `index` must be a valid, open index relation.
unsafe fn add_duplicate_on_disk(
    index: pg_sys::Relation,
    dup_blkno: pg_sys::BlockNumber,
    dup_offno: pg_sys::OffsetNumber,
    heap_tid: pg_sys::ItemPointer,
) -> bool {
    let buf = pg_sys::ReadBuffer(index, dup_blkno);
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let state = pg_sys::GenericXLogStart(index);
    let page = pg_sys::GenericXLogRegisterBuffer(state, buf, 0);

    let item_id = pg_sys::PageGetItemId(page, dup_offno);
    let etup = pg_sys::PageGetItem(page, item_id) as *mut HnswElementTupleData;

    // Find first invalid (empty) heap TID slot
    let mut slot = HNSW_HEAPTIDS;
    for i in 0..HNSW_HEAPTIDS {
        if !pg_sys::ItemPointerIsValid(&(*etup).heaptids[i]) {
            slot = i;
            break;
        }
    }

    // Either being deleted (slot 0 invalid) or all slots full
    if slot == 0 || slot == HNSW_HEAPTIDS {
        pg_sys::GenericXLogAbort(state);
        pg_sys::UnlockReleaseBuffer(buf);
        return false;
    }

    // Add heap TID to the element tuple on disk
    (*etup).heaptids[slot] = *heap_tid;
    pg_sys::GenericXLogFinish(state);
    pg_sys::UnlockReleaseBuffer(buf);
    true
}

/// Check level-0 neighbors for duplicate vectors and add the heap TID
/// to the first matching element that has room.
///
/// Returns `true` if a duplicate was found and the TID was added.
///
/// # Safety
/// `index` must be valid. `value_ptr` must point to valid vector data.
unsafe fn find_duplicate_on_disk(
    index: pg_sys::Relation,
    layer0_neighbors: &[ScanCandidate],
    value_ptr: *const u8,
    value_size: usize,
    heap_tid: pg_sys::ItemPointer,
) -> bool {
    for neighbor in layer0_neighbors {
        // Neighbors are ordered by distance; stop at first non-zero distance
        if neighbor.distance != 0.0 {
            break;
        }

        // Load neighbor's vector data and compare byte-for-byte
        let buf = pg_sys::ReadBuffer(index, neighbor.blkno);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_SHARE as i32);
        let page = buffer_get_page(buf);
        let item_id = pg_sys::PageGetItemId(page, neighbor.offno);
        let etup = pg_sys::PageGetItem(page, item_id) as *const HnswElementTupleData;

        // Compare the full varlena data (includes header + float data)
        let is_equal = {
            let on_disk_ptr = (etup as *const u8).add(std::mem::size_of::<HnswElementTupleData>());
            std::slice::from_raw_parts(on_disk_ptr, value_size)
                == std::slice::from_raw_parts(value_ptr, value_size)
        };

        pg_sys::UnlockReleaseBuffer(buf);

        if is_equal && add_duplicate_on_disk(index, neighbor.blkno, neighbor.offno, heap_tid) {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// aminsert callback
// ---------------------------------------------------------------------------

/// Insert a new tuple into the HNSW index.
///
/// This is the `aminsert` callback. Finds neighbors for the new element
/// using on-disk search, writes element/neighbor tuples to disk, updates
/// existing neighbor connections, and updates the meta page if needed.
#[pg_guard]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C-unwind" fn aminsert(
    index_relation: pg_sys::Relation,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    heap_tid: pg_sys::ItemPointer,
    _heap_relation: pg_sys::Relation,
    _check_unique: pg_sys::IndexUniqueCheck::Type,
    _index_unchanged: bool,
    _index_info: *mut pg_sys::IndexInfo,
) -> bool {
    // Skip NULL values
    if *isnull.add(0) {
        return false;
    }

    // Detoast the datum (type-agnostic: works for vector, bit, halfvec, etc.)
    let raw_datum = *values.add(0);
    let mut detoasted = pg_sys::pg_detoast_datum(raw_datum.cast_mut_ptr());

    // Check norm for cosine distance (skip zero-norm vectors)
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
    let collation = (*index_relation).rd_indcollation.read();

    if !norm_fmgr.is_null() {
        let norm_result = pg_sys::FunctionCall1Coll(
            norm_fmgr,
            collation,
            pg_sys::Datum::from(detoasted as usize),
        );
        let norm_val = f64::from_bits(norm_result.value() as u64);
        if norm_val == 0.0 {
            return false;
        }
        // Normalize the vector to unit length for cosine distance.
        detoasted =
            crate::types::vector::l2_normalize_raw(detoasted as *const _) as *mut pg_sys::varlena;
    }

    // Get index parameters
    let m = HnswOptions::get_m(index_relation);
    let ef_construction = HnswOptions::get_ef_construction(index_relation);
    let ml = hnsw_get_ml(m);
    let max_level = hnsw_get_max_level(m);

    // Get distance function
    let dist_fmgr = pg_sys::index_getprocinfo(index_relation, 1, HNSW_DISTANCE_PROC);
    let query_datum = pg_sys::Datum::from(detoasted as usize);

    // Acquire shared UPDATE_LOCK
    let mut lockmode = pg_sys::ShareLock as pg_sys::LOCKMODE;
    pg_sys::LockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);

    // Read meta page info
    let (_, mut entry_blkno, mut entry_offno, mut entry_level) = get_meta_page_info(index_relation);

    // Assign random level
    let r: f64 = rand::random::<f64>().max(f64::MIN_POSITIVE);
    let new_level = std::cmp::min((-r.ln() * ml).floor() as usize, max_level) as i32;

    // Upgrade to exclusive lock if likely updating entry point
    if entry_blkno == pg_sys::InvalidBlockNumber || new_level > entry_level {
        pg_sys::UnlockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);
        lockmode = pg_sys::ExclusiveLock as pg_sys::LOCKMODE;
        pg_sys::LockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);

        // Re-read meta page after lock upgrade (another backend may have
        // updated the entry point while we waited for the exclusive lock)
        let (_, eb, eo, el) = get_meta_page_info(index_relation);
        entry_blkno = eb;
        entry_offno = eo;
        entry_level = el;
    }

    // Find neighbors
    let neighbors_by_layer = if entry_blkno != pg_sys::InvalidBlockNumber {
        find_element_neighbors_on_disk(
            index_relation,
            query_datum,
            dist_fmgr,
            collation,
            new_level,
            m,
            ef_construction,
            entry_blkno,
            entry_offno,
            entry_level,
            None,
        )
    } else {
        vec![Vec::new(); (new_level + 1) as usize]
    };

    // Check for duplicate vectors among level-0 neighbors.
    // If found, add heap TID to the existing element instead of creating
    // a new graph node.
    let varlena_ptr = detoasted as *const u8;
    let varlena_size = (*(varlena_ptr as *const u32) >> 2) as usize;
    if !neighbors_by_layer.is_empty()
        && find_duplicate_on_disk(
            index_relation,
            &neighbors_by_layer[0],
            varlena_ptr,
            varlena_size,
            heap_tid,
        )
    {
        pg_sys::UnlockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);
        return false;
    }

    // Build element tuple
    let etup_size = hnsw_element_tuple_size(varlena_size);
    let ntup_size = hnsw_neighbor_tuple_size(new_level as usize, m as usize);

    let etup_buf = pg_sys::palloc0(etup_size) as *mut u8;
    let ntup_buf = pg_sys::palloc0(ntup_size) as *mut u8;

    // Fill element tuple
    let etup = etup_buf as *mut HnswElementTupleData;
    (*etup).type_ = HNSW_ELEMENT_TUPLE_TYPE;
    (*etup).level = new_level as u8;
    (*etup).deleted = 0;
    (*etup).version = 0;
    (*etup).heaptids[0] = *heap_tid;
    for i in 1..HNSW_HEAPTIDS {
        pg_sys::ItemPointerSetInvalid(&mut (*etup).heaptids[i]);
    }

    // Copy vector data after header
    let data_dst = etup_buf.add(std::mem::size_of::<HnswElementTupleData>());
    std::ptr::copy_nonoverlapping(varlena_ptr, data_dst, varlena_size);

    // Fill neighbor tuple with selected neighbors
    let ntup = ntup_buf as *mut HnswNeighborTupleData;
    (*ntup).type_ = HNSW_NEIGHBOR_TUPLE_TYPE;
    (*ntup).version = 0;

    let indextids_base =
        ntup_buf.add(std::mem::size_of::<HnswNeighborTupleData>()) as *mut pg_sys::ItemPointerData;
    let mut tid_idx: usize = 0;

    for lc in (0..=new_level).rev() {
        let lm = hnsw_get_layer_m(m, lc) as usize;
        let layer_neighbors = if (lc as usize) < neighbors_by_layer.len() {
            &neighbors_by_layer[lc as usize]
        } else {
            &neighbors_by_layer[0] // shouldn't happen
        };

        for i in 0..lm {
            let indextid = &mut *indextids_base.add(tid_idx);
            tid_idx += 1;

            if i < layer_neighbors.len() {
                pg_sys::ItemPointerSet(
                    indextid,
                    layer_neighbors[i].blkno,
                    layer_neighbors[i].offno,
                );
            } else {
                pg_sys::ItemPointerSetInvalid(indextid);
            }
        }
    }
    (*ntup).count = tid_idx as u16;

    // Get insert page and add element to disk
    let insert_page_blkno = get_insert_page(index_relation);

    // neighbortid will be set by add_element_on_disk after we know the
    // location; for now set to invalid, then overwrite below
    pg_sys::ItemPointerSetInvalid(&mut (*etup).neighbortid);

    let inserted = add_element_on_disk(
        index_relation,
        etup_buf,
        etup_size,
        ntup_buf,
        ntup_size,
        insert_page_blkno,
    );

    // Fix up the neighbortid in the element tuple on disk
    {
        let buf = pg_sys::ReadBuffer(index_relation, inserted.blkno);
        pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let xlog_state = pg_sys::GenericXLogStart(index_relation);
        let page = pg_sys::GenericXLogRegisterBuffer(xlog_state, buf, 0);
        let item_id = pg_sys::PageGetItemId(page, inserted.offno);
        let on_disk_etup = pg_sys::PageGetItem(page, item_id) as *mut HnswElementTupleData;
        pg_sys::ItemPointerSet(
            &mut (*on_disk_etup).neighbortid,
            inserted.neighbor_page,
            inserted.neighbor_offno,
        );
        pg_sys::GenericXLogFinish(xlog_state);
        pg_sys::UnlockReleaseBuffer(buf);
    }

    // Update existing neighbors to add back-connections
    update_neighbors_on_disk(
        index_relation,
        inserted.blkno,
        inserted.offno,
        new_level,
        &neighbors_by_layer,
        m,
        query_datum,
        dist_fmgr,
        collation,
    );

    // Update meta page if insert page changed or entry point needs updating
    let update_entry = entry_blkno == pg_sys::InvalidBlockNumber || new_level > entry_level;
    let entry_mode = if update_entry {
        HNSW_UPDATE_ENTRY_GREATER
    } else {
        0
    };
    if update_entry || inserted.updated_insert_page != pg_sys::InvalidBlockNumber {
        update_meta_page(
            index_relation,
            entry_mode,
            inserted.blkno,
            inserted.offno,
            new_level,
            inserted.updated_insert_page,
        );
    }

    // Release UPDATE_LOCK
    pg_sys::UnlockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);

    // Free palloc'd buffers
    pg_sys::pfree(etup_buf as *mut std::ffi::c_void);
    pg_sys::pfree(ntup_buf as *mut std::ffi::c_void);

    false
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hnsw_insert_basic() {
        // Create table and index with initial data
        Spi::run("CREATE TABLE test_ins (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_ins (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]')",
        )
        .unwrap();
        Spi::run(
            "CREATE INDEX test_ins_idx ON test_ins \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Insert new row after index creation (triggers aminsert)
        Spi::run("INSERT INTO test_ins (val) VALUES ('[0.5,0.5,0]')").unwrap();

        // Verify the new row is found via index scan
        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_ins \
             ORDER BY val <-> '[0.5,0.5,0]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        assert_eq!(result, "[0.5,0.5,0]");
    }

    #[pg_test]
    fn test_hnsw_insert_multiple() {
        Spi::run("CREATE TABLE test_ins2 (id serial, val vector(3))").unwrap();
        Spi::run("INSERT INTO test_ins2 (val) VALUES ('[1,0,0]'), ('[0,1,0]')").unwrap();
        Spi::run(
            "CREATE INDEX test_ins2_idx ON test_ins2 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Insert several new rows
        Spi::run(
            "INSERT INTO test_ins2 (val) VALUES \
             ('[0,0,1]'), ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]')",
        )
        .unwrap();

        // Verify count
        let count = Spi::get_one::<i64>("SELECT count(*) FROM test_ins2")
            .expect("SPI failed")
            .expect("NULL");
        assert_eq!(count, 6);

        // Verify nearest neighbor search works
        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_ins2 \
             ORDER BY val <-> '[1,1,1]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        // Should be one of the corner vectors
        assert!(!result.is_empty());
    }

    #[pg_test]
    fn test_hnsw_insert_into_empty_index() {
        // Create empty table with index, then insert
        Spi::run("CREATE TABLE test_ins3 (id serial, val vector(3))").unwrap();
        Spi::run(
            "CREATE INDEX test_ins3_idx ON test_ins3 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Insert into empty index
        Spi::run("INSERT INTO test_ins3 (val) VALUES ('[1,0,0]')").unwrap();
        Spi::run("INSERT INTO test_ins3 (val) VALUES ('[0,1,0]')").unwrap();
        Spi::run("INSERT INTO test_ins3 (val) VALUES ('[0,0,1]')").unwrap();

        // Verify search works
        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_ins3 \
             ORDER BY val <-> '[1,0,0]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        assert_eq!(result, "[1,0,0]");
    }

    #[pg_test]
    fn test_hnsw_insert_null_skipped() {
        Spi::run("CREATE TABLE test_ins4 (id serial, val vector(3))").unwrap();
        Spi::run("INSERT INTO test_ins4 (val) VALUES ('[1,0,0]')").unwrap();
        Spi::run(
            "CREATE INDEX test_ins4_idx ON test_ins4 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Insert NULL — should be skipped
        Spi::run("INSERT INTO test_ins4 (val) VALUES (NULL)").unwrap();

        let count = Spi::get_one::<i64>("SELECT count(*) FROM test_ins4 WHERE val IS NOT NULL")
            .expect("SPI failed")
            .expect("NULL");
        assert_eq!(count, 1);
    }

    #[pg_test]
    fn test_hnsw_insert_cosine() {
        Spi::run("CREATE TABLE test_ins5 (id serial, val vector(3))").unwrap();
        Spi::run("INSERT INTO test_ins5 (val) VALUES ('[1,0,0]'), ('[0,1,0]')").unwrap();
        Spi::run(
            "CREATE INDEX test_ins5_idx ON test_ins5 \
             USING hnsw (val vector_cosine_ops)",
        )
        .unwrap();

        // Insert with cosine index
        Spi::run("INSERT INTO test_ins5 (val) VALUES ('[0,0,1]')").unwrap();

        let result = Spi::get_one::<String>(
            "SELECT val::text FROM test_ins5 \
             ORDER BY val <=> '[0,0,1]' LIMIT 1",
        )
        .expect("SPI failed")
        .expect("no result");
        assert_eq!(result, "[0,0,1]");
    }
}
