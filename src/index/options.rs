//! HNSW index options (reloptions) and GUC variables.

use pgrx::{GucContext, GucFlags, GucRegistry};

use crate::hnsw_constants::*;

/// GUC variable: `hnsw.ef_search` controls the size of the dynamic
/// candidate list during search. Valid range: 1..1000.
pub static HNSW_EF_SEARCH: pgrx::GucSetting<i32> =
    pgrx::GucSetting::<i32>::new(HNSW_DEFAULT_EF_SEARCH);

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
        // Setting at min should work
        Spi::run("SET hnsw.ef_search = 1").expect("SET min failed");
        let result = Spi::get_one::<String>("SHOW hnsw.ef_search")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "1");

        // Setting at max should work
        Spi::run("SET hnsw.ef_search = 1000").expect("SET max failed");
        let result = Spi::get_one::<String>("SHOW hnsw.ef_search")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "1000");
    }
}
