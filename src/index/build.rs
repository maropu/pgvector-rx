//! HNSW index building.
//!
//! Implements the `ambuild` callback that builds an HNSW index by:
//! 1. Scanning the heap table to collect all vectors
//! 2. Building the HNSW graph in memory using the graph module
//! 3. Flushing the graph to disk (meta page, element/neighbor tuples)
//! 4. WAL-logging all written pages

use pgrx::pg_guard;
use pgrx::pg_sys;

use crate::graph::{find_element_neighbors, update_neighbor_connections, DistanceFn, GraphElement};
use crate::hnsw_constants::*;
use crate::index::options::HnswOptions;
use crate::types::hnsw::*;
use crate::types::vector::VectorHeader;

// ---------------------------------------------------------------------------
// Buffer helpers (PostgreSQL macro equivalents)
// ---------------------------------------------------------------------------

/// `BufferGetPage(buffer)` equivalent.
///
/// # Safety
/// `buffer` must be a valid pinned buffer.
#[inline]
unsafe fn buffer_get_page(buffer: pg_sys::Buffer) -> pg_sys::Page {
    // SAFETY: BufferBlocks + (buffer - 1) * BLCKSZ
    pg_sys::BufferBlocks.add((buffer as usize - 1) * pg_sys::BLCKSZ as usize) as pg_sys::Page
}

/// P_NEW equivalent for ReadBufferExtended.
const P_NEW: pg_sys::BlockNumber = pg_sys::InvalidBlockNumber;

/// Allocate a new buffer, lock it exclusively, and return it.
///
/// # Safety
/// `index` must be a valid, open index relation.
#[inline]
unsafe fn hnsw_new_buffer(
    index: pg_sys::Relation,
    fork: pg_sys::ForkNumber::Type,
) -> pg_sys::Buffer {
    let buf = pg_sys::ReadBufferExtended(
        index,
        fork,
        P_NEW,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        std::ptr::null_mut(),
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    buf
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

/// Get a pointer to the opaque data in the special area of a page.
///
/// # Safety
/// `page` must be a valid HNSW page.
#[inline]
unsafe fn hnsw_page_get_opaque(page: pg_sys::Page) -> *mut HnswPageOpaqueData {
    let header = page as *const pg_sys::PageHeaderData;
    (page as *mut u8).add((*header).pd_special as usize) as *mut HnswPageOpaqueData
}

/// Get a pointer to the meta page data area.
///
/// # Safety
/// `page` must be a valid HNSW meta page.
#[inline]
unsafe fn hnsw_page_get_meta(page: pg_sys::Page) -> *mut HnswMetaPageData {
    // Meta page data starts right after the page header
    let header_size = std::mem::size_of::<pg_sys::PageHeaderData>();
    (page as *mut u8).add(header_size) as *mut HnswMetaPageData
}

/// Append a new page during build, linking it from the current page.
///
/// # Safety
/// All pointers must be valid. The caller holds exclusive locks.
unsafe fn hnsw_build_append_page(
    index: pg_sys::Relation,
    buf: &mut pg_sys::Buffer,
    page: &mut pg_sys::Page,
    fork: pg_sys::ForkNumber::Type,
) {
    let newbuf = hnsw_new_buffer(index, fork);

    // Link current page to the new page
    let opaque = hnsw_page_get_opaque(*page);
    (*opaque).nextblkno = pg_sys::BufferGetBlockNumber(newbuf);

    // Commit current page
    pg_sys::MarkBufferDirty(*buf);
    pg_sys::UnlockReleaseBuffer(*buf);

    // Initialize new page
    *buf = newbuf;
    *page = buffer_get_page(*buf);
    hnsw_init_page(*buf, *page);
}

// ---------------------------------------------------------------------------
// Build state
// ---------------------------------------------------------------------------

/// State for in-memory HNSW index construction.
struct HnswBuildState {
    /// Graph elements arena.
    elements: Vec<GraphElement>,
    /// Raw byte arena for vector values.
    values: Vec<u8>,
    /// Heap TIDs for each element (parallel to `elements`).
    heap_tids: Vec<pg_sys::ItemPointerData>,
    /// Index of the current graph entry point, or `None`.
    entry_point: Option<usize>,
    /// Number of dimensions.
    dimensions: i32,
    /// M parameter.
    m: i32,
    /// ef_construction parameter.
    ef_construction: i32,
    /// Maximum level based on page size constraints.
    max_level: usize,
    /// Level multiplier mL = 1/ln(M).
    ml: f64,
    /// Pointer to the distance function's FmgrInfo.
    dist_fmgr: *mut pg_sys::FmgrInfo,
    /// Collation for the distance function.
    collation: pg_sys::Oid,
    /// Pointer to the norm function's FmgrInfo, or null.
    norm_fmgr: *mut pg_sys::FmgrInfo,
    /// Number of indexed tuples.
    ind_tuples: f64,
    /// Heap relation tuples (from scan).
    rel_tuples: f64,
    /// Index relation.
    index: pg_sys::Relation,
    /// Fork number for writing pages.
    fork_num: pg_sys::ForkNumber::Type,
    /// Disk locations for each element (blkno, offno, neighbor_page,
    /// neighbor_offno). Populated during flush.
    disk_locs: Vec<DiskLocation>,
}

/// On-disk location of an element after flushing.
#[derive(Clone, Copy, Default)]
struct DiskLocation {
    blkno: pg_sys::BlockNumber,
    offno: pg_sys::OffsetNumber,
    neighbor_page: pg_sys::BlockNumber,
    neighbor_offno: pg_sys::OffsetNumber,
}

impl HnswBuildState {
    /// Create a new build state from index relation options.
    ///
    /// # Safety
    /// `index` must be a valid, open index relation.
    unsafe fn new(index: pg_sys::Relation) -> Self {
        let m = HnswOptions::get_m(index);
        let ef_construction = HnswOptions::get_ef_construction(index);
        let ml = hnsw_get_ml(m);
        let max_level = hnsw_get_max_level(m);
        let fork_num = pg_sys::ForkNumber::MAIN_FORKNUM;

        // Get the distance function from the opclass support
        // SAFETY: index is valid and the opclass must define proc 1.
        let dist_fmgr = pg_sys::index_getprocinfo(index, 1, HNSW_DISTANCE_PROC);
        let collation = (*index).rd_indcollation.read();

        // Try to get the norm function (proc 2) for cosine distance.
        // It may not exist for L2/IP operator classes.
        let norm_fmgr =
            if (*index).rd_support.add(HNSW_NORM_PROC as usize - 1).read() != pg_sys::InvalidOid {
                pg_sys::index_getprocinfo(index, 1, HNSW_NORM_PROC)
            } else {
                std::ptr::null_mut()
            };

        Self {
            elements: Vec::new(),
            values: Vec::new(),
            heap_tids: Vec::new(),
            entry_point: None,
            dimensions: 0,
            m,
            ef_construction,
            max_level,
            ml,
            dist_fmgr,
            collation,
            norm_fmgr,
            ind_tuples: 0.0,
            rel_tuples: 0.0,
            index,
            fork_num,
            disk_locs: Vec::new(),
        }
    }

    /// Compute the distance between two vectors stored in the values arena.
    fn distance_fn(
        values: &[u8],
        a_offset: usize,
        a_size: usize,
        b_offset: usize,
        b_size: usize,
    ) -> f32 {
        // SAFETY: We recover the FmgrInfo pointer from thread-local state.
        // This function is only called during build, which is
        // single-threaded. The pointer is valid for the duration of
        // the build.
        unsafe {
            let state = BUILD_STATE_PTR.expect("BUILD_STATE_PTR not set");
            let fmgr = (*state).dist_fmgr;
            let collation = (*state).collation;

            let a_ptr = pg_sys::Datum::from(values.as_ptr().add(a_offset) as usize);
            let b_ptr = pg_sys::Datum::from(values.as_ptr().add(b_offset) as usize);
            let _ = (a_size, b_size);

            let result = pg_sys::FunctionCall2Coll(fmgr, collation, a_ptr, b_ptr);
            f64::from_bits(result.value() as u64) as f32
        }
    }

    /// Assign a random level for a new element using the HNSW paper's
    /// exponential distribution: floor(-ln(rand()) * mL).
    fn random_level(&self) -> i32 {
        let r: f64 = rand::random::<f64>().max(f64::MIN_POSITIVE);
        let level = (-r.ln() * self.ml).floor() as usize;
        std::cmp::min(level, self.max_level) as i32
    }
}

/// Thread-local pointer to the active build state, used by the distance
/// function callback (which cannot carry arbitrary context).
///
/// SAFETY: Only used during sequential index build, which is
/// single-threaded per backend.
static mut BUILD_STATE_PTR: Option<*mut HnswBuildState> = None;

// ---------------------------------------------------------------------------
// Build callback
// ---------------------------------------------------------------------------

/// Heap scan callback for index building.
///
/// Called once per heap tuple. Extracts the vector, normalizes if needed,
/// and inserts into the in-memory HNSW graph.
///
/// # Safety
/// All pointers are provided by PostgreSQL's table_index_build_scan and
/// are guaranteed valid.
#[pg_guard]
unsafe extern "C-unwind" fn build_callback(
    _index: pg_sys::Relation,
    tid: pg_sys::ItemPointer,
    values: *mut pg_sys::Datum,
    isnull: *mut bool,
    _tuple_is_alive: bool,
    state: *mut std::ffi::c_void,
) {
    let bs = &mut *(state as *mut HnswBuildState);

    // Skip NULL values
    if *isnull.add(0) {
        return;
    }

    // Detoast the vector datum
    let raw_datum = *values.add(0);
    let mut vec_ptr = pg_sys::pg_detoast_datum(raw_datum.cast_mut_ptr()) as *const VectorHeader;
    let dim = (*vec_ptr).dim as i32;

    // Validate dimensions
    if !(1..=HNSW_MAX_DIM).contains(&dim) {
        pgrx::error!(
            "pgvector-rx: vector has {} dimensions, max is {}",
            dim,
            HNSW_MAX_DIM
        );
    }

    // Set dimensions from first non-null vector
    if bs.dimensions == 0 {
        bs.dimensions = dim;
    } else if bs.dimensions != dim {
        pgrx::error!(
            "pgvector-rx: expected {} dimensions, not {}",
            bs.dimensions,
            dim
        );
    }

    // Check norm for cosine distance (skip zero-norm vectors)
    // and normalize the vector when the norm function is present.
    if !bs.norm_fmgr.is_null() {
        let norm_result = pg_sys::FunctionCall1Coll(
            bs.norm_fmgr,
            bs.collation,
            pg_sys::Datum::from(vec_ptr as usize),
        );
        let norm_val = f64::from_bits(norm_result.value() as u64);
        if norm_val == 0.0 {
            return;
        }
        // Normalize the vector to unit length for cosine distance.
        vec_ptr = crate::types::vector::l2_normalize_raw(vec_ptr);
    }

    // Copy the vector data into our values arena
    let varlena_ptr = vec_ptr as *const u8;
    // VARSIZE: first 4 bytes store (size << 2) for 4-byte header
    let varlena_size = (*(varlena_ptr as *const u32) >> 2) as usize;

    let value_offset = bs.values.len();
    let value_size = varlena_size;
    bs.values
        .extend_from_slice(std::slice::from_raw_parts(varlena_ptr, varlena_size));

    // Assign level and create graph element
    let level = bs.random_level();
    let elem = GraphElement::new(level, bs.m, value_offset, value_size);
    let new_idx = bs.elements.len();
    bs.elements.push(elem);

    // Insert into graph
    if let Some(entry_idx) = bs.entry_point {
        find_element_neighbors(
            &mut bs.elements,
            &bs.values,
            new_idx,
            entry_idx,
            bs.ef_construction,
            bs.m,
            HnswBuildState::distance_fn as DistanceFn,
        );
        update_neighbor_connections(
            &mut bs.elements,
            &bs.values,
            new_idx,
            bs.m,
            HnswBuildState::distance_fn as DistanceFn,
        );

        // Update entry point if new element has higher level
        if bs.elements[new_idx].level > bs.elements[entry_idx].level {
            bs.entry_point = Some(new_idx);
        }
    } else {
        // First element becomes the entry point
        bs.entry_point = Some(new_idx);
    }

    // Store heap TID for disk persistence
    bs.heap_tids.push(*tid);

    bs.ind_tuples += 1.0;
}

// ---------------------------------------------------------------------------
// Disk persistence
// ---------------------------------------------------------------------------

/// Create the meta page (block 0).
///
/// # Safety
/// `bs` must be valid. Called during build with no concurrent access.
unsafe fn create_meta_page(bs: &HnswBuildState) {
    let buf = hnsw_new_buffer(bs.index, bs.fork_num);
    let page = buffer_get_page(buf);
    hnsw_init_page(buf, page);

    let metap = hnsw_page_get_meta(page);
    (*metap).magic_number = HNSW_MAGIC_NUMBER;
    (*metap).version = HNSW_VERSION;
    (*metap).dimensions = bs.dimensions as u32;
    (*metap).m = bs.m as u16;
    (*metap).ef_construction = bs.ef_construction as u16;
    (*metap).entry_blkno = pg_sys::InvalidBlockNumber;
    (*metap).entry_offno = pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber;
    (*metap).entry_level = -1;
    (*metap).insert_page = pg_sys::InvalidBlockNumber;

    // Set pd_lower to end of meta page data
    let header = page as *mut pg_sys::PageHeaderData;
    (*header).pd_lower =
        (metap as usize + std::mem::size_of::<HnswMetaPageData>() - page as usize) as u16;

    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);
}

/// Create graph pages: write element and neighbor placeholder tuples.
///
/// Returns the insert page block number.
///
/// # Safety
/// `bs` must be valid with a built graph. Called during build.
unsafe fn create_graph_pages(bs: &mut HnswBuildState) -> pg_sys::BlockNumber {
    if bs.elements.is_empty() {
        return pg_sys::InvalidBlockNumber;
    }

    let max_size = hnsw_max_size();

    // Prepare element and neighbor tuple buffers (max possible size)
    let alloc_size = max_size;
    let etup_buf = pg_sys::palloc0(alloc_size) as *mut u8;
    let ntup_buf = pg_sys::palloc0(alloc_size) as *mut u8;

    // Allocate disk location tracking
    bs.disk_locs = vec![DiskLocation::default(); bs.elements.len()];

    // Prepare first data page
    let mut buf = hnsw_new_buffer(bs.index, bs.fork_num);
    let mut page = buffer_get_page(buf);
    hnsw_init_page(buf, page);

    for idx in 0..bs.elements.len() {
        let level = bs.elements[idx].level;
        let value_offset = bs.elements[idx].value_offset;
        let value_size = bs.elements[idx].value_size;

        // Calculate tuple sizes
        let etup_size = hnsw_element_tuple_size(value_size);
        let ntup_size = hnsw_neighbor_tuple_size(level as usize, bs.m as usize);
        let combined_size = etup_size + ntup_size + std::mem::size_of::<pg_sys::ItemIdData>();

        if etup_size > alloc_size {
            pgrx::error!("pgvector-rx: index tuple too large");
        }

        // Zero element tuple buffer
        std::ptr::write_bytes(etup_buf, 0, etup_size);

        // Fill element tuple header
        let etup = etup_buf as *mut HnswElementTupleData;
        (*etup).type_ = HNSW_ELEMENT_TUPLE_TYPE;
        (*etup).level = level as u8;
        (*etup).deleted = 0;
        (*etup).version = 0;

        // Set heap TIDs — first one is the actual TID, rest are invalid
        (*etup).heaptids[0] = bs.heap_tids[idx];
        for i in 1..HNSW_HEAPTIDS {
            pg_sys::ItemPointerSetInvalid(&mut (*etup).heaptids[i]);
        }

        // Copy vector value data after the element tuple header
        let data_dst = etup_buf.add(std::mem::size_of::<HnswElementTupleData>());
        let data_src = bs.values.as_ptr().add(value_offset);
        std::ptr::copy_nonoverlapping(data_src, data_dst, value_size);

        // Check if we need a new page for the element tuple
        let free_space = pg_sys::PageGetFreeSpace(page as *const pg_sys::PageData);
        let need_new_page =
            free_space < etup_size || (combined_size <= max_size && free_space < combined_size);

        if need_new_page {
            hnsw_build_append_page(bs.index, &mut buf, &mut page, bs.fork_num);
        }

        // Record element disk location
        let elem_blkno = pg_sys::BufferGetBlockNumber(buf);
        let elem_offno = pg_sys::PageGetMaxOffsetNumber(page as *const pg_sys::PageData) + 1;

        // Determine neighbor tuple location
        let (neighbor_page, neighbor_offno) = if combined_size <= max_size {
            (elem_blkno, elem_offno + 1)
        } else {
            (elem_blkno + 1, 1u16) // FirstOffsetNumber
        };

        bs.disk_locs[idx] = DiskLocation {
            blkno: elem_blkno,
            offno: elem_offno,
            neighbor_page,
            neighbor_offno,
        };

        // Set the neighbor TID in the element tuple
        pg_sys::ItemPointerSet(&mut (*etup).neighbortid, neighbor_page, neighbor_offno);

        // Add element tuple to page
        let added_offno = pg_sys::PageAddItemExtended(
            page,
            etup_buf as pg_sys::Item,
            etup_size,
            pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
            0, // no flags
        );
        if added_offno != elem_offno {
            pgrx::error!("pgvector-rx: failed to add element tuple to page");
        }

        // Check if we need a new page for the neighbor tuple
        let free_space = pg_sys::PageGetFreeSpace(page as *const pg_sys::PageData);
        if free_space < ntup_size {
            hnsw_build_append_page(bs.index, &mut buf, &mut page, bs.fork_num);
        }

        // Add placeholder neighbor tuple (zeroed)
        std::ptr::write_bytes(ntup_buf, 0, ntup_size);
        let ntup = ntup_buf as *mut HnswNeighborTupleData;
        (*ntup).type_ = HNSW_NEIGHBOR_TUPLE_TYPE;

        let added_offno = pg_sys::PageAddItemExtended(
            page,
            ntup_buf as pg_sys::Item,
            ntup_size,
            pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
            0,
        );
        if added_offno != neighbor_offno {
            pgrx::error!("pgvector-rx: failed to add neighbor tuple to page");
        }
    }

    let insert_page = pg_sys::BufferGetBlockNumber(buf);

    // Commit last page
    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);

    pg_sys::pfree(etup_buf as *mut std::ffi::c_void);
    pg_sys::pfree(ntup_buf as *mut std::ffi::c_void);

    insert_page
}

/// Write actual neighbor data, overwriting the placeholder tuples.
///
/// # Safety
/// Graph must be flushed (disk_locs populated).
unsafe fn write_neighbor_tuples(bs: &HnswBuildState) {
    if bs.elements.is_empty() {
        return;
    }

    let max_size = hnsw_max_size();
    let ntup_buf = pg_sys::palloc0(max_size) as *mut u8;

    for idx in 0..bs.elements.len() {
        let elem = &bs.elements[idx];
        let loc = &bs.disk_locs[idx];
        let ntup_size = hnsw_neighbor_tuple_size(elem.level as usize, bs.m as usize);

        // Zero the buffer
        std::ptr::write_bytes(ntup_buf, 0, ntup_size);

        // Fill neighbor tuple
        let ntup = ntup_buf as *mut HnswNeighborTupleData;
        (*ntup).type_ = HNSW_NEIGHBOR_TUPLE_TYPE;
        (*ntup).version = 0;

        // Serialize neighbor TIDs: iterate layers from highest to lowest
        // (matching pgvector's C code which goes from level down to 0)
        let indextids_base = ntup_buf.add(std::mem::size_of::<HnswNeighborTupleData>())
            as *mut pg_sys::ItemPointerData;
        let mut tid_idx: usize = 0;

        for lc in (0..=elem.level).rev() {
            let lm = hnsw_get_layer_m(bs.m, lc) as usize;
            let neighbors = &elem.neighbors[lc as usize];

            for i in 0..lm {
                let indextid = indextids_base.add(tid_idx);
                tid_idx += 1;

                if i < neighbors.items.len() {
                    let neighbor_idx = neighbors.items[i].idx;
                    let nloc = &bs.disk_locs[neighbor_idx];
                    pg_sys::ItemPointerSet(indextid, nloc.blkno, nloc.offno);
                } else {
                    pg_sys::ItemPointerSetInvalid(indextid);
                }
            }
        }

        (*ntup).count = tid_idx as u16;

        // Read the page, overwrite the placeholder
        let read_buf = pg_sys::ReadBufferExtended(
            bs.index,
            bs.fork_num,
            loc.neighbor_page,
            pg_sys::ReadBufferMode::RBM_NORMAL,
            std::ptr::null_mut(),
        );
        pg_sys::LockBuffer(read_buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
        let page = buffer_get_page(read_buf);

        if !pg_sys::PageIndexTupleOverwrite(
            page,
            loc.neighbor_offno,
            ntup_buf as pg_sys::Item,
            ntup_size,
        ) {
            pgrx::error!("pgvector-rx: failed to overwrite neighbor tuple");
        }

        pg_sys::MarkBufferDirty(read_buf);
        pg_sys::UnlockReleaseBuffer(read_buf);
    }

    pg_sys::pfree(ntup_buf as *mut std::ffi::c_void);
}

/// Update the meta page with the entry point and insert page.
///
/// # Safety
/// Meta page must already exist at block 0.
unsafe fn update_meta_page(bs: &HnswBuildState, insert_page: pg_sys::BlockNumber) {
    if bs.entry_point.is_none() {
        return;
    }

    let buf = pg_sys::ReadBufferExtended(
        bs.index,
        bs.fork_num,
        HNSW_METAPAGE_BLKNO,
        pg_sys::ReadBufferMode::RBM_NORMAL,
        std::ptr::null_mut(),
    );
    pg_sys::LockBuffer(buf, pg_sys::BUFFER_LOCK_EXCLUSIVE as i32);
    let page = buffer_get_page(buf);
    let metap = hnsw_page_get_meta(page);

    if let Some(ep_idx) = bs.entry_point {
        let loc = &bs.disk_locs[ep_idx];
        (*metap).entry_blkno = loc.blkno;
        (*metap).entry_offno = loc.offno;
        (*metap).entry_level = bs.elements[ep_idx].level as i16;
    }

    (*metap).insert_page = insert_page;

    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Build the HNSW index.
///
/// This is the `ambuild` callback. Scans the heap table, builds an
/// in-memory HNSW graph, and flushes it to disk.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuild(
    heap_relation: pg_sys::Relation,
    index_relation: pg_sys::Relation,
    index_info: *mut pg_sys::IndexInfo,
) -> *mut pg_sys::IndexBuildResult {
    let mut bs = HnswBuildState::new(index_relation);

    // Validate ef_construction >= 2 * m (matches original pgvector check)
    if bs.ef_construction < 2 * bs.m {
        pgrx::error!("ef_construction must be greater than or equal to 2 * m");
    }

    // Set up thread-local state for the distance function
    BUILD_STATE_PTR = Some(&mut bs as *mut HnswBuildState);

    // Scan the heap table and build the in-memory graph
    bs.rel_tuples = pg_sys::table_index_build_scan(
        heap_relation,
        index_relation,
        index_info,
        true, // allow_sync
        true, // progress
        Some(build_callback),
        &mut bs as *mut HnswBuildState as *mut std::ffi::c_void,
        std::ptr::null_mut(), // no existing scan
    );

    // Flush the graph to disk
    create_meta_page(&bs);
    let insert_page = create_graph_pages(&mut bs);
    write_neighbor_tuples(&bs);
    update_meta_page(&bs, insert_page);

    // WAL-log all pages written
    let nblocks =
        pg_sys::RelationGetNumberOfBlocksInFork(index_relation, pg_sys::ForkNumber::MAIN_FORKNUM);
    if nblocks > 0 {
        pg_sys::log_newpage_range(
            index_relation,
            pg_sys::ForkNumber::MAIN_FORKNUM,
            0,
            nblocks,
            true, // page_std
        );
    }

    // Clear thread-local state
    BUILD_STATE_PTR = None;

    // Return build result
    let result = pg_sys::palloc0(std::mem::size_of::<pg_sys::IndexBuildResult>())
        as *mut pg_sys::IndexBuildResult;
    (*result).heap_tuples = bs.rel_tuples;
    (*result).index_tuples = bs.ind_tuples;

    result
}

/// Build an empty HNSW index (for UNLOGGED tables).
///
/// This is the `ambuildempty` callback.
#[pg_guard]
pub unsafe extern "C-unwind" fn ambuildempty(index_relation: pg_sys::Relation) {
    let fork_num = pg_sys::ForkNumber::INIT_FORKNUM;

    // Create an empty meta page on the init fork
    let buf = hnsw_new_buffer(index_relation, fork_num);
    let page = buffer_get_page(buf);
    hnsw_init_page(buf, page);

    let metap = hnsw_page_get_meta(page);
    (*metap).magic_number = HNSW_MAGIC_NUMBER;
    (*metap).version = HNSW_VERSION;
    (*metap).dimensions = 0;
    (*metap).m = HNSW_DEFAULT_M as u16;
    (*metap).ef_construction = HNSW_DEFAULT_EF_CONSTRUCTION as u16;
    (*metap).entry_blkno = pg_sys::InvalidBlockNumber;
    (*metap).entry_offno = pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber;
    (*metap).entry_level = -1;
    (*metap).insert_page = pg_sys::InvalidBlockNumber;

    let header = page as *mut pg_sys::PageHeaderData;
    (*header).pd_lower =
        (metap as usize + std::mem::size_of::<HnswMetaPageData>() - page as usize) as u16;

    pg_sys::MarkBufferDirty(buf);
    pg_sys::UnlockReleaseBuffer(buf);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hnsw_create_empty_index() {
        Spi::run("CREATE TABLE test_build (id serial, val vector(3))").unwrap();
        Spi::run(
            "CREATE INDEX test_build_idx ON test_build \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Verify the index exists
        let result = Spi::get_one::<String>(
            "SELECT indexname::text FROM pg_indexes \
             WHERE indexname = 'test_build_idx'",
        )
        .expect("SPI failed")
        .expect("index not found");
        assert_eq!(result, "test_build_idx");
    }

    #[pg_test]
    fn test_hnsw_build_with_data() {
        Spi::run("CREATE TABLE test_build2 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_build2 (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'), \
             ('[1,1,0]'), ('[0,1,1]'), ('[1,0,1]')",
        )
        .unwrap();

        // This should build the index without errors
        Spi::run(
            "CREATE INDEX test_build2_idx ON test_build2 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();

        // Verify the index was created and has the right size
        let count = Spi::get_one::<i64>(
            "SELECT count(*) FROM pg_class \
             WHERE relname = 'test_build2_idx' AND relkind = 'i'",
        )
        .expect("SPI failed")
        .expect("NULL");
        assert_eq!(count, 1);
    }

    #[pg_test]
    fn test_hnsw_build_with_nulls() {
        Spi::run("CREATE TABLE test_build3 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_build3 (val) VALUES \
             ('[1,0,0]'), (NULL), ('[0,1,0]'), (NULL), ('[0,0,1]')",
        )
        .unwrap();

        // Should build successfully, skipping NULLs
        Spi::run(
            "CREATE INDEX test_build3_idx ON test_build3 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();
    }

    #[pg_test]
    fn test_hnsw_build_with_custom_params() {
        Spi::run("CREATE TABLE test_build4 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_build4 (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]')",
        )
        .unwrap();

        // Custom m and ef_construction
        Spi::run(
            "CREATE INDEX test_build4_idx ON test_build4 \
             USING hnsw (val vector_l2_ops) WITH (m = 4, ef_construction = 8)",
        )
        .unwrap();
    }

    #[pg_test]
    fn test_hnsw_build_cosine() {
        Spi::run("CREATE TABLE test_build5 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_build5 (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]')",
        )
        .unwrap();

        // Build with cosine operator class
        Spi::run(
            "CREATE INDEX test_build5_idx ON test_build5 \
             USING hnsw (val vector_cosine_ops)",
        )
        .unwrap();
    }

    #[pg_test]
    fn test_hnsw_build_ip() {
        Spi::run("CREATE TABLE test_build6 (id serial, val vector(3))").unwrap();
        Spi::run(
            "INSERT INTO test_build6 (val) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]')",
        )
        .unwrap();

        // Build with inner product operator class
        Spi::run(
            "CREATE INDEX test_build6_idx ON test_build6 \
             USING hnsw (val vector_ip_ops)",
        )
        .unwrap();
    }

    #[pg_test]
    fn test_hnsw_build_larger_dataset() {
        Spi::run("CREATE TABLE test_build7 (id serial, val vector(3))").unwrap();
        // Insert 50 rows
        for i in 0..50 {
            let x = (i as f64 * 0.1).sin() as f32;
            let y = (i as f64 * 0.2).cos() as f32;
            let z = (i as f64 * 0.15) as f32 / 10.0;
            Spi::run(&format!(
                "INSERT INTO test_build7 (val) VALUES ('[{},{},{}]')",
                x, y, z
            ))
            .unwrap();
        }

        Spi::run(
            "CREATE INDEX test_build7_idx ON test_build7 \
             USING hnsw (val vector_l2_ops)",
        )
        .unwrap();
    }

    #[pg_test]
    #[should_panic(expected = "ef_construction must be greater than or equal to 2 * m")]
    fn test_hnsw_build_ef_construction_too_small() {
        Spi::run("CREATE TABLE test_ef_check (id serial, val vector(3))").unwrap();
        Spi::run("INSERT INTO test_ef_check (val) VALUES ('[1,0,0]')").unwrap();
        // m=16 requires ef_construction >= 32, but 31 < 32
        Spi::run(
            "CREATE INDEX ON test_ef_check \
             USING hnsw (val vector_l2_ops) WITH (m = 16, ef_construction = 31)",
        )
        .unwrap();
    }
}
