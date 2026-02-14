//! HNSW on-disk page structures.
//!
//! These `repr(C)` structs mirror the on-disk format used by the original
//! pgvector C implementation so that pages written by either implementation
//! can be read by the other.

use pgrx::pg_sys;

use crate::hnsw_constants::*;

// ---------------------------------------------------------------------------
// Page-level structures
// ---------------------------------------------------------------------------

/// Opaque data stored in the special area of each HNSW page.
///
/// Matches C's `HnswPageOpaqueData`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HnswPageOpaqueData {
    /// Next block in the chain, or `InvalidBlockNumber`.
    pub nextblkno: pg_sys::BlockNumber,
    /// Reserved for future use.
    pub unused: u16,
    /// Identifier for HNSW index pages (`HNSW_PAGE_ID`).
    pub page_id: u16,
}

impl HnswPageOpaqueData {
    /// Creates a new opaque data block with default values.
    pub fn new() -> Self {
        Self {
            nextblkno: pg_sys::InvalidBlockNumber,
            unused: 0,
            page_id: HNSW_PAGE_ID,
        }
    }
}

impl Default for HnswPageOpaqueData {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// Meta page
// ---------------------------------------------------------------------------

/// Metadata stored on the first page (block 0) of an HNSW index.
///
/// Matches C's `HnswMetaPageData`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HnswMetaPageData {
    /// Magic number for format identification (`HNSW_MAGIC_NUMBER`).
    pub magic_number: u32,
    /// Index format version (`HNSW_VERSION`).
    pub version: u32,
    /// Number of dimensions in indexed vectors.
    pub dimensions: u32,
    /// Max connections per node (M parameter).
    pub m: u16,
    /// Size of the dynamic candidate list during construction.
    pub ef_construction: u16,
    /// Block number of the entry point element, or `InvalidBlockNumber`.
    pub entry_blkno: pg_sys::BlockNumber,
    /// Offset of the entry point element, or `InvalidOffsetNumber`.
    pub entry_offno: pg_sys::OffsetNumber,
    /// Level of the entry point, or -1 if no entry point.
    pub entry_level: i16,
    /// Last page used for inserting new tuples.
    pub insert_page: pg_sys::BlockNumber,
}

impl HnswMetaPageData {
    /// Creates metadata for a new, empty HNSW index.
    pub fn new(dimensions: u32, m: u16, ef_construction: u16) -> Self {
        Self {
            magic_number: HNSW_MAGIC_NUMBER,
            version: HNSW_VERSION,
            dimensions,
            m,
            ef_construction,
            entry_blkno: pg_sys::InvalidBlockNumber,
            entry_offno: pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
            entry_level: -1,
            insert_page: pg_sys::InvalidBlockNumber,
        }
    }

    /// Returns `true` if this index has an entry point.
    pub fn has_entry_point(&self) -> bool {
        self.entry_blkno != pg_sys::InvalidBlockNumber
    }
}

// ---------------------------------------------------------------------------
// Element tuple (on-disk)
// ---------------------------------------------------------------------------

/// Element tuple stored on disk.
///
/// Matches C's `HnswElementTupleData`. The trailing `data` field is a
/// variable-length vector value (a PostgreSQL varlena).
///
/// **Note**: This struct is unsized in C (uses FLEXIBLE_ARRAY_MEMBER for
/// the vector data). In Rust we represent only the fixed-size header; the
/// variable-length payload lives immediately after this header on the page.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HnswElementTupleData {
    /// Tuple type marker (`HNSW_ELEMENT_TUPLE_TYPE`).
    pub type_: u8,
    /// Level of this element in the graph.
    pub level: u8,
    /// Whether this element has been logically deleted.
    pub deleted: u8,
    /// Tuple format version.
    pub version: u8,
    /// Heap TIDs referencing the underlying table row(s).
    pub heaptids: [pg_sys::ItemPointerData; HNSW_HEAPTIDS],
    /// TID pointing to the corresponding neighbor tuple.
    pub neighbortid: pg_sys::ItemPointerData,
    /// Reserved for future use.
    pub unused: u16,
    // Followed by: Vector data (variable-length varlena)
}

/// Returns the on-disk size of an element tuple for a vector of `data_size`
/// bytes (the full varlena size including header).
///
/// Matches C's `HNSW_ELEMENT_TUPLE_SIZE(size)`.
pub const fn hnsw_element_tuple_size(data_size: usize) -> usize {
    let base = std::mem::size_of::<HnswElementTupleData>();
    maxalign(base + data_size)
}

// ---------------------------------------------------------------------------
// Neighbor tuple (on-disk)
// ---------------------------------------------------------------------------

/// Neighbor tuple stored on disk.
///
/// Matches C's `HnswNeighborTupleData`. The `indextids` array has variable
/// length depending on the number of layers and M parameter.
///
/// This struct represents only the fixed header; `indextids` are accessed
/// via pointer arithmetic from the tuple start.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HnswNeighborTupleData {
    /// Tuple type marker (`HNSW_NEIGHBOR_TUPLE_TYPE`).
    pub type_: u8,
    /// Tuple format version.
    pub version: u8,
    /// Number of valid neighbor entries.
    pub count: u16,
    // Followed by: ItemPointerData[] (variable-length)
}

/// Returns the on-disk size of a neighbor tuple for the given level and M.
///
/// Matches C's `HNSW_NEIGHBOR_TUPLE_SIZE(level, m)`.
pub const fn hnsw_neighbor_tuple_size(level: usize, m: usize) -> usize {
    let base = std::mem::size_of::<HnswNeighborTupleData>();
    let tids = (level + 2) * m * std::mem::size_of::<pg_sys::ItemPointerData>();
    maxalign(base + tids)
}

// ---------------------------------------------------------------------------
// In-memory graph structures
// ---------------------------------------------------------------------------

/// A candidate element with its distance to the query.
///
/// Matches C's `HnswCandidate`.
#[derive(Debug, Clone, Copy)]
pub struct HnswCandidate {
    /// Distance from this candidate to the query point.
    pub distance: f32,
    /// Whether this candidate is closer than its neighbors (used by the
    /// heuristic neighbor selection algorithm).
    pub closer: bool,
    /// Block number of the element on disk.
    pub element_blkno: pg_sys::BlockNumber,
    /// Offset of the element on disk.
    pub element_offno: pg_sys::OffsetNumber,
}

/// Array of neighbor candidates for one layer of an element.
///
/// Matches C's `HnswNeighborArray`. Stored in-memory during index build
/// and search operations.
#[derive(Debug)]
pub struct HnswNeighborArray {
    /// Maximum number of items (layer M or 2*M for layer 0).
    pub max_len: usize,
    /// Whether the `closer` flags have been computed.
    pub closer_set: bool,
    /// The neighbor candidates.
    pub items: Vec<HnswCandidate>,
}

impl HnswNeighborArray {
    /// Creates a new neighbor array with the given maximum capacity.
    pub fn new(max_len: usize) -> Self {
        Self {
            max_len,
            closer_set: false,
            items: Vec::with_capacity(max_len),
        }
    }

    /// Returns the number of neighbors currently stored.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Returns `true` if no neighbors are stored.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// Returns `true` if the array is at full capacity.
    pub fn is_full(&self) -> bool {
        self.items.len() >= self.max_len
    }
}

/// In-memory representation of an HNSW graph element during index build.
///
/// A simplified version of C's `HnswElementData` that avoids the complex
/// dual-pointer (absolute/relative) scheme. During build we only use
/// absolute pointers (Rust references/indices).
#[derive(Debug)]
pub struct HnswElementData {
    /// Heap TIDs referencing the underlying table row(s).
    pub heaptids: Vec<pg_sys::ItemPointerData>,
    /// Level assigned to this element.
    pub level: u8,
    /// Whether this element has been logically deleted.
    pub deleted: bool,
    /// Tuple format version.
    pub version: u8,
    /// Hash of the element's value for duplicate detection.
    pub hash: u32,
    /// Neighbor arrays, one per layer (index 0 = layer 0, etc.).
    pub neighbors: Vec<HnswNeighborArray>,
    /// Block number where this element's tuple is stored on disk.
    pub blkno: pg_sys::BlockNumber,
    /// Offset of the element tuple on disk.
    pub offno: pg_sys::OffsetNumber,
    /// Offset of the neighbor tuple on disk.
    pub neighbor_offno: pg_sys::OffsetNumber,
    /// Block number of the neighbor tuple page.
    pub neighbor_page: pg_sys::BlockNumber,
}

impl HnswElementData {
    /// Creates a new element with the given level.
    ///
    /// Allocates neighbor arrays for each layer. Layer 0 gets `2 * m`
    /// slots; higher layers get `m` slots.
    pub fn new(level: u8, m: i32) -> Self {
        let mut neighbors = Vec::with_capacity(level as usize + 1);
        for lc in 0..=level as i32 {
            let lm = hnsw_get_layer_m(m, lc) as usize;
            neighbors.push(HnswNeighborArray::new(lm));
        }

        Self {
            heaptids: Vec::new(),
            level,
            deleted: false,
            version: 0,
            hash: 0,
            neighbors,
            blkno: pg_sys::InvalidBlockNumber,
            offno: pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
            neighbor_offno: pg_sys::InvalidOffsetNumber as pg_sys::OffsetNumber,
            neighbor_page: pg_sys::InvalidBlockNumber,
        }
    }

    /// Adds a heap TID to this element, up to the maximum of
    /// `HNSW_HEAPTIDS`.
    pub fn add_heaptid(&mut self, tid: pg_sys::ItemPointerData) {
        if self.heaptids.len() < HNSW_HEAPTIDS {
            self.heaptids.push(tid);
        }
    }
}

// ---------------------------------------------------------------------------
// Query / support structures
// ---------------------------------------------------------------------------

/// Query structure used during HNSW index scans.
///
/// Matches C's `HnswQuery`.
#[derive(Debug)]
pub struct HnswQuery {
    /// The query vector value as a PostgreSQL Datum.
    pub value: pg_sys::Datum,
}

// ---------------------------------------------------------------------------
// Size helpers
// ---------------------------------------------------------------------------

/// MAXALIGN equivalent — rounds up to the platform's maximum alignment.
///
/// On 64-bit systems this is 8 bytes, matching PostgreSQL's MAXALIGN.
#[inline]
pub const fn maxalign(size: usize) -> usize {
    const ALIGN: usize = std::mem::align_of::<u64>();
    (size + (ALIGN - 1)) & !(ALIGN - 1)
}

/// Maximum usable space on an HNSW page.
///
/// Matches C's `HNSW_MAX_SIZE`:
/// `BLCKSZ - MAXALIGN(SizeOfPageHeaderData) - MAXALIGN(sizeof(HnswPageOpaqueData)) - sizeof(ItemIdData)`
pub fn hnsw_max_size() -> usize {
    let blcksz = pg_sys::BLCKSZ as usize;
    let page_header = maxalign(std::mem::size_of::<pg_sys::PageHeaderData>());
    let opaque = maxalign(std::mem::size_of::<HnswPageOpaqueData>());
    let item_id = std::mem::size_of::<pg_sys::ItemIdData>();
    blcksz - page_header - opaque - item_id
}

/// Maximum level for a given M, ensuring the neighbor tuple fits on a
/// single page and the level fits in a `u8`.
///
/// Matches C's `HnswGetMaxLevel(m)`.
pub fn hnsw_get_max_level(m: i32) -> usize {
    let blcksz = pg_sys::BLCKSZ as usize;
    let page_header = maxalign(std::mem::size_of::<pg_sys::PageHeaderData>());
    let opaque = maxalign(std::mem::size_of::<HnswPageOpaqueData>());
    let ntup_header = std::mem::size_of::<HnswNeighborTupleData>();
    let item_id = std::mem::size_of::<pg_sys::ItemIdData>();
    let tid_size = std::mem::size_of::<pg_sys::ItemPointerData>();

    let available = blcksz - page_header - opaque - ntup_header - item_id;
    let max_from_page = available / tid_size / m as usize - 2;

    std::cmp::min(max_from_page, 255)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maxalign() {
        assert_eq!(maxalign(0), 0);
        assert_eq!(maxalign(1), 8);
        assert_eq!(maxalign(7), 8);
        assert_eq!(maxalign(8), 8);
        assert_eq!(maxalign(9), 16);
    }

    #[test]
    fn test_meta_page_new() {
        let meta = HnswMetaPageData::new(128, 16, 64);
        assert_eq!(meta.magic_number, HNSW_MAGIC_NUMBER);
        assert_eq!(meta.version, HNSW_VERSION);
        assert_eq!(meta.dimensions, 128);
        assert_eq!(meta.m, 16);
        assert_eq!(meta.ef_construction, 64);
        assert!(!meta.has_entry_point());
    }

    #[test]
    fn test_page_opaque_default() {
        let opaque = HnswPageOpaqueData::new();
        assert_eq!(opaque.page_id, HNSW_PAGE_ID);
        assert_eq!(opaque.unused, 0);
    }

    #[test]
    fn test_neighbor_array_new() {
        let na = HnswNeighborArray::new(32);
        assert_eq!(na.max_len, 32);
        assert!(na.is_empty());
        assert!(!na.is_full());
        assert_eq!(na.len(), 0);
    }

    #[test]
    fn test_element_data_layers() {
        let elem = HnswElementData::new(3, 16);
        assert_eq!(elem.level, 3);
        assert_eq!(elem.neighbors.len(), 4); // layers 0, 1, 2, 3
                                             // Layer 0 has 2*M = 32 capacity
        assert_eq!(elem.neighbors[0].max_len, 32);
        // Higher layers have M = 16 capacity
        assert_eq!(elem.neighbors[1].max_len, 16);
        assert_eq!(elem.neighbors[2].max_len, 16);
        assert_eq!(elem.neighbors[3].max_len, 16);
    }

    #[test]
    fn test_element_tuple_size() {
        // Verify the element tuple size calculation is aligned
        let size = hnsw_element_tuple_size(16); // 16 bytes of vector data
        assert_eq!(size % 8, 0, "element tuple size must be MAXALIGN'd");
    }

    #[test]
    fn test_neighbor_tuple_size() {
        // level=0, m=16 → (0+2)*16 = 32 ItemPointerData entries
        let size = hnsw_neighbor_tuple_size(0, 16);
        assert_eq!(size % 8, 0, "neighbor tuple size must be MAXALIGN'd");
    }

    #[test]
    fn test_hnsw_max_size() {
        let max = hnsw_max_size();
        // Should be reasonable — several kilobytes
        assert!(max > 4000, "max size should be > 4KB");
        assert!(max < pg_sys::BLCKSZ as usize, "max size < BLCKSZ");
    }

    #[test]
    fn test_hnsw_get_max_level() {
        let max_level = hnsw_get_max_level(16);
        // Should be a reasonable number
        assert!(max_level > 0, "max level should be > 0");
        assert!(max_level <= 255, "max level should fit in u8");
    }
}
