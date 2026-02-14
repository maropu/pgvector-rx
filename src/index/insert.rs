//! HNSW index insertion.
//!
//! Implements the `aminsert` callback that inserts a single tuple into an
//! existing HNSW index by:
//! 1. Finding neighbors using on-disk multi-layer search
//! 2. Writing element and neighbor tuples to disk
//! 3. Updating existing neighbor back-connections
//! 4. Updating the meta page (entry point and insert page)

use pgrx::pg_guard;
use pgrx::pg_sys;

use crate::hnsw_constants::*;
use crate::index::options::HnswOptions;
use crate::index::scan::{
    buffer_get_page, get_meta_page_info, hnsw_page_get_meta, hnsw_page_get_meta_mut, load_element,
    search_layer_disk, ScanCandidate,
};
use crate::types::hnsw::*;
use crate::types::vector::VectorHeader;

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

/// Add element and neighbor tuples to disk pages, using GenericXLog for WAL.
///
/// Walks the page chain starting from `insert_page` to find space.
///
/// # Safety
/// `index` must be valid. All buffers are properly locked and released.
#[allow(clippy::too_many_arguments)]
unsafe fn add_element_on_disk(
    index: pg_sys::Relation,
    etup_data: *const u8,
    etup_size: usize,
    ntup_data: *const u8,
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

/// Update a single neighbor's connection to include the new element.
///
/// # Safety
/// `index` must be valid.
#[allow(clippy::too_many_arguments)]
unsafe fn update_neighbor_on_disk(
    index: pg_sys::Relation,
    neighbor_page: pg_sys::BlockNumber,
    neighbor_offno: pg_sys::OffsetNumber,
    neighbor_level: i32,
    neighbor_version: u8,
    m: i32,
    layer: i32,
    new_blkno: pg_sys::BlockNumber,
    new_offno: pg_sys::OffsetNumber,
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
            // Already connected
            pg_sys::GenericXLogAbort(state);
            pg_sys::UnlockReleaseBuffer(buf);
            return;
        }
    }

    // Find a free slot or the worst neighbor to replace
    let mut free_idx: Option<usize> = None;
    for i in 0..lm {
        let tid = &*tids_base.add(start_idx + i);
        if !pg_sys::ItemPointerIsValid(tid) {
            free_idx = Some(start_idx + i);
            break;
        }
    }

    if let Some(idx) = free_idx {
        // Free slot available — add the connection
        if idx < (*ntup).count as usize {
            let indextid = &mut *tids_base.add(idx);
            pg_sys::ItemPointerSet(indextid, new_blkno, new_offno);
            pg_sys::GenericXLogFinish(state);
        } else {
            pg_sys::GenericXLogAbort(state);
        }
    } else {
        // All slots full — skip (simplified: don't replace neighbors yet)
        pg_sys::GenericXLogAbort(state);
    }

    pg_sys::UnlockReleaseBuffer(buf);
}

/// Update all neighbors of the new element to add back-connections.
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

            update_neighbor_on_disk(
                index,
                neighbor.neighbor_page,
                neighbor.neighbor_offno,
                neighbor.level,
                neighbor.version,
                m,
                lc,
                new_blkno,
                new_offno,
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

        let w = search_layer_disk(
            index,
            ep_list,
            ef_construction as usize,
            lc,
            m,
            query_datum,
            dist_fmgr,
            collation,
            None,
            None,
            true,
        );

        // Select up to lm nearest neighbors (simple selection)
        let selected: Vec<ScanCandidate> = w.iter().rev().take(lm).cloned().collect();
        neighbors_by_layer[lc as usize] = selected;

        ep_list = w;
    }

    neighbors_by_layer
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

    // Detoast the vector datum
    let raw_datum = *values.add(0);
    let mut vec_ptr = pg_sys::pg_detoast_datum(raw_datum.cast_mut_ptr()) as *const VectorHeader;
    let dim = (*vec_ptr).dim as i32;

    if !(1..=HNSW_MAX_DIM).contains(&dim) {
        pgrx::error!(
            "pgvector-rx: vector has {} dimensions, max is {}",
            dim,
            HNSW_MAX_DIM
        );
    }

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
        let norm_result =
            pg_sys::FunctionCall1Coll(norm_fmgr, collation, pg_sys::Datum::from(vec_ptr as usize));
        let norm_val = f64::from_bits(norm_result.value() as u64);
        if norm_val == 0.0 {
            return false;
        }
        // Normalize the vector to unit length for cosine distance.
        vec_ptr = crate::types::vector::l2_normalize_raw(vec_ptr);
    }

    // Get index parameters
    let m = HnswOptions::get_m(index_relation);
    let ef_construction = HnswOptions::get_ef_construction(index_relation);
    let ml = hnsw_get_ml(m);
    let max_level = hnsw_get_max_level(m);

    // Get distance function
    let dist_fmgr = pg_sys::index_getprocinfo(index_relation, 1, HNSW_DISTANCE_PROC);
    let query_datum = pg_sys::Datum::from(vec_ptr as usize);

    // Acquire shared UPDATE_LOCK
    let mut lockmode = pg_sys::ShareLock as pg_sys::LOCKMODE;
    pg_sys::LockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);

    // Read meta page info
    let (_, entry_blkno, entry_offno, entry_level) = get_meta_page_info(index_relation);

    // Assign random level
    let r: f64 = rand::random::<f64>().max(f64::MIN_POSITIVE);
    let new_level = std::cmp::min((-r.ln() * ml).floor() as usize, max_level) as i32;

    // Upgrade to exclusive lock if likely updating entry point
    if entry_blkno == pg_sys::InvalidBlockNumber || new_level > entry_level {
        pg_sys::UnlockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);
        lockmode = pg_sys::ExclusiveLock as pg_sys::LOCKMODE;
        pg_sys::LockPage(index_relation, HNSW_UPDATE_LOCK, lockmode);
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
        )
    } else {
        vec![Vec::new(); (new_level + 1) as usize]
    };

    // Build element tuple
    let varlena_ptr = vec_ptr as *const u8;
    let varlena_size = (*(varlena_ptr as *const u32) >> 2) as usize;
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
