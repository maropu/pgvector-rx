//! HNSW index options (reloptions) and GUC variables.

use pgrx::pg_sys::AsPgCStr;
use pgrx::{pg_guard, pg_sys, GucContext, GucFlags, GucRegistry};

use crate::hnsw_constants::*;

// ---------------------------------------------------------------------------
// Iterative scan mode enum for GUC
// ---------------------------------------------------------------------------

/// Iterative scan modes for HNSW search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HnswIterativeScan {
    /// Iterative scan disabled.
    Off = HNSW_ITERATIVE_SCAN_OFF as isize,
    /// Relaxed ordering mode.
    RelaxedOrder = HNSW_ITERATIVE_SCAN_RELAXED as isize,
    /// Strict ordering mode.
    StrictOrder = HNSW_ITERATIVE_SCAN_STRICT as isize,
}

/// Wrapper to safely share `config_enum_entry` arrays across threads.
///
/// SAFETY: The wrapped array contains only static string pointers and integer
/// values, which are never mutated after initialization.
struct SyncEnumEntries([pg_sys::config_enum_entry; 4]);
unsafe impl Sync for SyncEnumEntries {}

/// config_enum_entry array for `hnsw.iterative_scan`.
static HNSW_ITERATIVE_SCAN_OPTIONS: SyncEnumEntries = SyncEnumEntries([
    pg_sys::config_enum_entry {
        name: c"off".as_ptr(),
        val: HNSW_ITERATIVE_SCAN_OFF,
        hidden: false,
    },
    pg_sys::config_enum_entry {
        name: c"relaxed_order".as_ptr(),
        val: HNSW_ITERATIVE_SCAN_RELAXED,
        hidden: false,
    },
    pg_sys::config_enum_entry {
        name: c"strict_order".as_ptr(),
        val: HNSW_ITERATIVE_SCAN_STRICT,
        hidden: false,
    },
    // Sentinel entry
    pg_sys::config_enum_entry {
        name: std::ptr::null(),
        val: 0,
        hidden: false,
    },
]);

// SAFETY: The CONFIG_ENUM_ENTRY points to a valid static array with a
// null-terminated sentinel. The enum values match the ordinals exactly.
unsafe impl pgrx::GucEnum for HnswIterativeScan {
    fn from_ordinal(ordinal: i32) -> Self {
        match ordinal {
            HNSW_ITERATIVE_SCAN_OFF => HnswIterativeScan::Off,
            HNSW_ITERATIVE_SCAN_RELAXED => HnswIterativeScan::RelaxedOrder,
            HNSW_ITERATIVE_SCAN_STRICT => HnswIterativeScan::StrictOrder,
            _ => HnswIterativeScan::Off,
        }
    }

    fn to_ordinal(&self) -> i32 {
        *self as i32
    }

    const CONFIG_ENUM_ENTRY: *const pg_sys::config_enum_entry =
        HNSW_ITERATIVE_SCAN_OPTIONS.0.as_ptr();
}

// ---------------------------------------------------------------------------
// GUC variables
// ---------------------------------------------------------------------------

/// GUC variable: `hnsw.ef_search` controls the size of the dynamic
/// candidate list during search. Valid range: 1..1000.
pub static HNSW_EF_SEARCH: pgrx::GucSetting<i32> =
    pgrx::GucSetting::<i32>::new(HNSW_DEFAULT_EF_SEARCH);

/// GUC variable: `hnsw.iterative_scan` controls the iterative scan mode.
pub static HNSW_ITERATIVE_SCAN: pgrx::GucSetting<HnswIterativeScan> =
    pgrx::GucSetting::<HnswIterativeScan>::new(HnswIterativeScan::Off);

/// GUC variable: `hnsw.max_scan_tuples` sets the max number of tuples
/// to visit for iterative scans. Valid range: 1..INT_MAX.
pub static HNSW_MAX_SCAN_TUPLES: pgrx::GucSetting<i32> =
    pgrx::GucSetting::<i32>::new(HNSW_DEFAULT_MAX_SCAN_TUPLES);

/// GUC variable: `hnsw.scan_mem_multiplier` sets the multiple of
/// work_mem to use for iterative scans. Valid range: 1..1000.
pub static HNSW_SCAN_MEM_MULTIPLIER: pgrx::GucSetting<f64> =
    pgrx::GucSetting::<f64>::new(HNSW_DEFAULT_SCAN_MEM_MULTIPLIER);

/// Custom relopt_kind for HNSW index options.
///
/// SAFETY: Written once in `init_relopts()` during `_PG_init()`, before any
/// concurrent access is possible.
static mut HNSW_RELOPT_KIND: pg_sys::relopt_kind::Type = 0;

/// HNSW index reloptions stored in `rd_options`.
///
/// Layout must match the C `HnswOptions` struct:
/// ```c
/// typedef struct HnswOptions {
///     int32 vl_len_;
///     int   m;
///     int   efConstruction;
/// } HnswOptions;
/// ```
#[repr(C)]
pub struct HnswOptions {
    /// Varlena header (do not touch directly).
    pub vl_len_: i32,
    /// Max number of connections per node (M parameter).
    pub m: i32,
    /// Size of the dynamic candidate list for construction.
    pub ef_construction: i32,
}

impl HnswOptions {
    /// Returns `m` from the given relation's reloptions, or the default.
    ///
    /// # Safety
    /// `relation` must be a valid, open index relation.
    #[allow(dead_code)]
    pub unsafe fn get_m(relation: pg_sys::Relation) -> i32 {
        let rd_options = (*relation).rd_options as *const HnswOptions;
        if rd_options.is_null() {
            HNSW_DEFAULT_M
        } else {
            (*rd_options).m
        }
    }

    /// Returns `ef_construction` from the given relation's reloptions,
    /// or the default.
    ///
    /// # Safety
    /// `relation` must be a valid, open index relation.
    #[allow(dead_code)]
    pub unsafe fn get_ef_construction(relation: pg_sys::Relation) -> i32 {
        let rd_options = (*relation).rd_options as *const HnswOptions;
        if rd_options.is_null() {
            HNSW_DEFAULT_EF_CONSTRUCTION
        } else {
            (*rd_options).ef_construction
        }
    }
}

/// Registers HNSW GUC variables with PostgreSQL.
pub fn init_gucs() {
    GucRegistry::define_int_guc(
        c"hnsw.ef_search",
        c"Sets the size of the dynamic candidate list for search.",
        c"Valid range is 1..1000.",
        &HNSW_EF_SEARCH,
        HNSW_MIN_EF_SEARCH,
        HNSW_MAX_EF_SEARCH,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_enum_guc(
        c"hnsw.iterative_scan",
        c"Sets the mode for iterative scans.",
        c"",
        &HNSW_ITERATIVE_SCAN,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_int_guc(
        c"hnsw.max_scan_tuples",
        c"Sets the max number of tuples to visit for iterative scans.",
        c"",
        &HNSW_MAX_SCAN_TUPLES,
        1,
        i32::MAX,
        GucContext::Userset,
        GucFlags::default(),
    );

    GucRegistry::define_float_guc(
        c"hnsw.scan_mem_multiplier",
        c"Sets the multiple of work_mem to use for iterative scans.",
        c"",
        &HNSW_SCAN_MEM_MULTIPLIER,
        1.0,
        1000.0,
        GucContext::Userset,
        GucFlags::default(),
    );
}

/// Registers HNSW reloptions (`m`, `ef_construction`) with PostgreSQL.
///
/// Must be called during `_PG_init()`.
pub fn init_relopts() {
    unsafe {
        HNSW_RELOPT_KIND = pg_sys::add_reloption_kind();
        pg_sys::add_int_reloption(
            HNSW_RELOPT_KIND,
            "m".as_pg_cstr(),
            "Max number of connections".as_pg_cstr(),
            HNSW_DEFAULT_M,
            HNSW_MIN_M,
            HNSW_MAX_M,
            pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
        );
        pg_sys::add_int_reloption(
            HNSW_RELOPT_KIND,
            "ef_construction".as_pg_cstr(),
            "Size of the dynamic candidate list for construction".as_pg_cstr(),
            HNSW_DEFAULT_EF_CONSTRUCTION,
            HNSW_MIN_EF_CONSTRUCTION,
            HNSW_MAX_EF_CONSTRUCTION,
            pg_sys::AccessExclusiveLock as pg_sys::LOCKMODE,
        );
    }
}

/// Parse and validate HNSW reloptions.
///
/// This is the `amoptions` callback for the HNSW access method.
#[pg_guard]
pub unsafe extern "C-unwind" fn amoptions(
    reloptions: pg_sys::Datum,
    validate: bool,
) -> *mut pg_sys::bytea {
    let options: [pg_sys::relopt_parse_elt; 2] = [
        pg_sys::relopt_parse_elt {
            optname: "m".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(HnswOptions, m) as i32,
            isset_offset: 0,
        },
        pg_sys::relopt_parse_elt {
            optname: "ef_construction".as_pg_cstr(),
            opttype: pg_sys::relopt_type::RELOPT_TYPE_INT,
            offset: std::mem::offset_of!(HnswOptions, ef_construction) as i32,
            isset_offset: 0,
        },
    ];

    pg_sys::build_reloptions(
        reloptions,
        validate,
        HNSW_RELOPT_KIND,
        std::mem::size_of::<HnswOptions>(),
        options.as_ptr(),
        2,
    ) as *mut pg_sys::bytea
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_ef_search_guc_default() {
        let result = Spi::get_one::<String>("SHOW hnsw.ef_search")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "40");
    }

    #[pg_test]
    fn test_ef_search_guc_set() {
        Spi::run("SET hnsw.ef_search = 100").expect("SET failed");
        let result = Spi::get_one::<String>("SHOW hnsw.ef_search")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "100");
    }

    #[pg_test]
    fn test_ef_search_guc_boundary() {
        Spi::run("SET hnsw.ef_search = 1").expect("SET min failed");
        let result = Spi::get_one::<String>("SHOW hnsw.ef_search")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "1");

        Spi::run("SET hnsw.ef_search = 1000").expect("SET max failed");
        let result = Spi::get_one::<String>("SHOW hnsw.ef_search")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "1000");
    }

    #[pg_test]
    fn test_iterative_scan_guc_default() {
        let result = Spi::get_one::<String>("SHOW hnsw.iterative_scan")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "off");
    }

    #[pg_test]
    fn test_iterative_scan_guc_set() {
        Spi::run("SET hnsw.iterative_scan = relaxed_order").expect("SET relaxed_order failed");
        let result = Spi::get_one::<String>("SHOW hnsw.iterative_scan")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "relaxed_order");

        Spi::run("SET hnsw.iterative_scan = strict_order").expect("SET strict_order failed");
        let result = Spi::get_one::<String>("SHOW hnsw.iterative_scan")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "strict_order");

        Spi::run("SET hnsw.iterative_scan = off").expect("SET off failed");
        let result = Spi::get_one::<String>("SHOW hnsw.iterative_scan")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "off");
    }

    #[pg_test]
    fn test_max_scan_tuples_guc_default() {
        let result = Spi::get_one::<String>("SHOW hnsw.max_scan_tuples")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "20000");
    }

    #[pg_test]
    fn test_max_scan_tuples_guc_set() {
        Spi::run("SET hnsw.max_scan_tuples = 50000").expect("SET failed");
        let result = Spi::get_one::<String>("SHOW hnsw.max_scan_tuples")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "50000");
    }

    #[pg_test]
    fn test_scan_mem_multiplier_guc_default() {
        let result = Spi::get_one::<String>("SHOW hnsw.scan_mem_multiplier")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "1");
    }

    #[pg_test]
    fn test_scan_mem_multiplier_guc_set() {
        Spi::run("SET hnsw.scan_mem_multiplier = 5").expect("SET failed");
        let result = Spi::get_one::<String>("SHOW hnsw.scan_mem_multiplier")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "5");
    }
}
