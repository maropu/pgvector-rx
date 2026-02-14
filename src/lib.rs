//! pgvector-rx: HNSW vector index for PostgreSQL, implemented in Rust.

use pgrx::prelude::*;

pub mod graph;
pub mod hnsw_constants;
pub mod index;
pub mod types;
pub mod utils;

::pgrx::pg_module_magic!(name, version);

/// Extension initialization — registers GUCs and hooks.
#[pg_guard]
pub extern "C-unwind" fn _PG_init() {
    index::init_gucs();
}

#[pg_extern]
fn hello_pgvector_rx() -> &'static str {
    "Hello, pgvector_rx"
}

#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hello_pgvector_rx() {
        assert_eq!("Hello, pgvector_rx", crate::hello_pgvector_rx());
    }
}

/// This module is required by `cargo pgrx test` invocations.
/// It must be visible at the root of your extension crate.
#[cfg(test)]
pub mod pg_test {
    pub fn setup(_options: Vec<&str>) {
        // perform one-off initialization when the pg_test framework starts
    }

    #[must_use]
    pub fn postgresql_conf_options() -> Vec<&'static str> {
        // return any postgresql.conf settings that are required for your tests
        vec![]
    }
}
