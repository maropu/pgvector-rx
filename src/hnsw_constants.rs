//! HNSW constants matching the original pgvector C implementation.

/// Maximum number of dimensions for HNSW index.
pub const HNSW_MAX_DIM: i32 = 2000;

/// Maximum number of non-zero elements for sparse vectors in HNSW.
pub const HNSW_MAX_NNZ: i32 = 1000;

// --- Support function procedure numbers ---

/// Distance function procedure number.
pub const HNSW_DISTANCE_PROC: u16 = 1;

/// Norm function procedure number.
pub const HNSW_NORM_PROC: u16 = 2;

/// Type info function procedure number.
pub const HNSW_TYPE_INFO_PROC: u16 = 3;

// --- Versioning ---

/// HNSW index version.
pub const HNSW_VERSION: u32 = 1;

/// Magic number for HNSW index identification.
pub const HNSW_MAGIC_NUMBER: u32 = 0xA953A953;

/// Page ID for HNSW index pages.
pub const HNSW_PAGE_ID: u16 = 0xFF90;

// --- Reserved page numbers ---

/// Block number for the meta page.
pub const HNSW_METAPAGE_BLKNO: u32 = 0;

/// Block number for the first element page.
pub const HNSW_HEAD_BLKNO: u32 = 1;

// --- Lock identifiers (correspond to page numbers) ---

/// Lock ID for update operations.
pub const HNSW_UPDATE_LOCK: u32 = 0;

/// Lock ID for scan operations.
pub const HNSW_SCAN_LOCK: u32 = 1;

// --- HNSW parameters ---

/// Default value for M (max connections per layer).
pub const HNSW_DEFAULT_M: i32 = 16;

/// Minimum value for M.
pub const HNSW_MIN_M: i32 = 2;

/// Maximum value for M.
pub const HNSW_MAX_M: i32 = 100;

/// Default value for ef_construction.
pub const HNSW_DEFAULT_EF_CONSTRUCTION: i32 = 64;

/// Minimum value for ef_construction.
pub const HNSW_MIN_EF_CONSTRUCTION: i32 = 4;

/// Maximum value for ef_construction.
pub const HNSW_MAX_EF_CONSTRUCTION: i32 = 1000;

/// Default value for ef_search.
pub const HNSW_DEFAULT_EF_SEARCH: i32 = 40;

/// Minimum value for ef_search.
pub const HNSW_MIN_EF_SEARCH: i32 = 1;

/// Maximum value for ef_search.
pub const HNSW_MAX_EF_SEARCH: i32 = 1000;

// --- Tuple types ---

/// Element tuple type marker.
pub const HNSW_ELEMENT_TUPLE_TYPE: u8 = 1;

/// Neighbor tuple type marker.
pub const HNSW_NEIGHBOR_TUPLE_TYPE: u8 = 2;

/// Number of heap TIDs stored per element for robustness against non-HOT updates.
pub const HNSW_HEAPTIDS: usize = 10;

// --- Entry point update modes ---

/// Update entry point only if new point is at a greater level.
pub const HNSW_UPDATE_ENTRY_GREATER: i32 = 1;

/// Always update entry point.
pub const HNSW_UPDATE_ENTRY_ALWAYS: i32 = 2;

// --- Build phases ---

/// Build phase: loading tuples.
pub const PROGRESS_HNSW_PHASE_LOAD: i64 = 2;

/// Returns the number of connections for a given layer.
/// Layer 0 has 2*M connections; higher layers have M connections.
#[inline]
pub const fn hnsw_get_layer_m(m: i32, layer: i32) -> i32 {
    if layer == 0 {
        m * 2
    } else {
        m
    }
}

/// Returns the optimal level multiplier mL from the HNSW paper.
#[inline]
pub fn hnsw_get_ml(m: i32) -> f64 {
    1.0 / (m as f64).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_constants_match_c_values() {
        assert_eq!(HNSW_MAX_DIM, 2000);
        assert_eq!(HNSW_MAGIC_NUMBER, 0xA953A953);
        assert_eq!(HNSW_PAGE_ID, 0xFF90);
        assert_eq!(HNSW_DEFAULT_M, 16);
        assert_eq!(HNSW_DEFAULT_EF_CONSTRUCTION, 64);
        assert_eq!(HNSW_DEFAULT_EF_SEARCH, 40);
        assert_eq!(HNSW_HEAPTIDS, 10);
    }

    #[test]
    fn test_layer_m() {
        assert_eq!(hnsw_get_layer_m(16, 0), 32); // ground layer: 2*M
        assert_eq!(hnsw_get_layer_m(16, 1), 16); // higher layers: M
        assert_eq!(hnsw_get_layer_m(16, 5), 16);
    }

    #[test]
    fn test_ml() {
        let ml = hnsw_get_ml(16);
        // mL = 1/ln(16) ≈ 0.3607
        assert!((ml - 0.3607).abs() < 0.001);
    }
}
