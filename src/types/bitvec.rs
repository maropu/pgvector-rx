//! PostgreSQL `bit` type distance functions for HNSW indexing.
//!
//! Provides Hamming and Jaccard distance functions for PostgreSQL's built-in
//! `bit` type, enabling HNSW index support via `bit_hamming_ops` and
//! `bit_jaccard_ops` operator classes.

use pgrx::prelude::*;
use pgrx::{pg_guard, pg_sys};

use std::ptr;

/// Generates a `pg_finfo_<name>` function required by PostgreSQL to look up
/// a C-language function.
macro_rules! pg_fn_info {
    ($name:ident) => {
        paste::paste! {
            #[no_mangle]
            pub extern "C-unwind" fn [<pg_finfo_ $name>]() -> *const pg_sys::Pg_finfo_record {
                static INFO: pg_sys::Pg_finfo_record = pg_sys::Pg_finfo_record { api_version: 1 };
                &INFO
            }
        }
    };
}

/// On-disk VarBit header (matches PostgreSQL's VarBit struct).
#[repr(C)]
pub struct VarBitHeader {
    pub vl_len_: i32,
    pub bit_len: i32,
    // bit_dat[] follows immediately after
}

/// VARHDRSZ: size of varlena header (4 bytes for standard 4-byte header).
const VARHDRSZ: usize = std::mem::size_of::<i32>();
/// VARBITHDRSZ: size of the bit_len field (4 bytes).
const VARBITHDRSZ: usize = std::mem::size_of::<i32>();

/// Returns the total size of a detoasted varlena value (VARSIZE).
///
/// # Safety
/// `ptr` must point to a valid, detoasted varlena (4-byte header).
#[inline]
unsafe fn varsize(ptr: *const VarBitHeader) -> usize {
    // 4-byte header: size is stored as (size << 2) in first 4 bytes.
    ((*(ptr as *const u32)) >> 2) as usize
}

/// Returns the number of data bytes in a VarBit (VARSIZE - VARHDRSZ - VARBITHDRSZ).
///
/// # Safety
/// `varbit` must point to a valid, detoasted VarBit.
#[inline]
unsafe fn varbit_bytes(varbit: *const VarBitHeader) -> usize {
    varsize(varbit) - VARHDRSZ - VARBITHDRSZ
}

/// Returns a pointer to the bit data bytes (bit_dat field).
///
/// # Safety
/// `varbit` must point to a valid VarBit.
#[inline]
unsafe fn varbit_data(varbit: *const VarBitHeader) -> *const u8 {
    // Skip vl_len_ (4 bytes) + bit_len (4 bytes) to reach bit_dat
    (varbit as *const u8).add(VARHDRSZ + VARBITHDRSZ)
}

/// Read argument N from a FunctionCallInfo as a raw Datum.
///
/// # Safety
/// `fcinfo` must be valid and `n` must be < nargs.
#[inline]
unsafe fn fc_arg(fcinfo: pg_sys::FunctionCallInfo, n: usize) -> pg_sys::Datum {
    let args_ptr: *const pg_sys::NullableDatum = ptr::addr_of!((*fcinfo).args).cast();
    (*args_ptr.add(n)).value
}

/// Check that two VarBit values have the same bit length.
///
/// # Safety
/// Both pointers must point to valid VarBit values.
#[inline]
unsafe fn check_bit_dims(a: *const VarBitHeader, b: *const VarBitHeader) {
    if (*a).bit_len != (*b).bit_len {
        pgrx::error!(
            "different bit lengths {} and {}",
            (*a).bit_len,
            (*b).bit_len
        );
    }
}

/// Compute Hamming distance between two bit vectors.
///
/// Counts the number of positions where the bits differ (popcount of XOR).
#[inline]
fn compute_hamming_distance(nbytes: usize, a: *const u8, b: *const u8) -> u64 {
    let mut distance: u64 = 0;
    // SAFETY: Caller ensures a and b have at least nbytes valid bytes.
    unsafe {
        for i in 0..nbytes {
            distance += pg_sys::pg_number_of_ones[(*a.add(i) ^ *b.add(i)) as usize] as u64;
        }
    }
    distance
}

/// Compute Jaccard distance between two bit vectors.
///
/// Jaccard distance = 1 - |A ∩ B| / |A ∪ B|
///                  = 1 - popcount(A AND B) / (popcount(A) + popcount(B) - popcount(A AND B))
#[inline]
fn compute_jaccard_distance(nbytes: usize, a: *const u8, b: *const u8) -> f64 {
    let mut ab: u64 = 0; // popcount(A AND B)
    let mut aa: u64 = 0; // popcount(A)
    let mut bb: u64 = 0; // popcount(B)
                         // SAFETY: Caller ensures a and b have at least nbytes valid bytes.
    unsafe {
        for i in 0..nbytes {
            let ai = *a.add(i);
            let bi = *b.add(i);
            ab += pg_sys::pg_number_of_ones[(ai & bi) as usize] as u64;
            aa += pg_sys::pg_number_of_ones[ai as usize] as u64;
            bb += pg_sys::pg_number_of_ones[bi as usize] as u64;
        }
    }
    if ab == 0 {
        1.0
    } else {
        1.0 - (ab as f64 / (aa + bb - ab) as f64)
    }
}

// ---------------------------------------------------------------------------
// SQL-callable distance functions
// ---------------------------------------------------------------------------

pg_fn_info!(hamming_distance);
pg_fn_info!(jaccard_distance);

/// Hamming distance between two bit vectors.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn hamming_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VarBitHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VarBitHeader;
    check_bit_dims(a, b);
    let nbytes = varbit_bytes(a);
    let dist = compute_hamming_distance(nbytes, varbit_data(a), varbit_data(b)) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Jaccard distance between two bit vectors.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn jaccard_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VarBitHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VarBitHeader;
    check_bit_dims(a, b);
    let nbytes = varbit_bytes(a);
    let dist = compute_jaccard_distance(nbytes, varbit_data(a), varbit_data(b));
    pg_sys::Datum::from(f64::to_bits(dist))
}

// ---------------------------------------------------------------------------
// HNSW support function
// ---------------------------------------------------------------------------

pg_fn_info!(hnsw_bit_support);

/// Returns type info for bit vectors in HNSW indexes.
///
/// The maxDimensions for bit is HNSW_MAX_DIM * 32 = 64000 bits.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn hnsw_bit_support(
    _fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    static TYPE_INFO: crate::types::halfvec::HnswTypeInfo = crate::types::halfvec::HnswTypeInfo {
        max_dimensions: crate::hnsw_constants::HNSW_MAX_DIM * 32,
    };
    pg_sys::Datum::from(&TYPE_INFO as *const crate::types::halfvec::HnswTypeInfo as usize)
}

// ---------------------------------------------------------------------------
// SQL definitions
// ---------------------------------------------------------------------------

extension_sql!(
    r#"
-- Bit vector distance functions
CREATE FUNCTION hamming_distance(bit, bit) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION jaccard_distance(bit, bit) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION hnsw_bit_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Distance operators
CREATE OPERATOR <~> (
    LEFTARG = bit, RIGHTARG = bit, PROCEDURE = hamming_distance,
    COMMUTATOR = '<~>'
);

CREATE OPERATOR <%> (
    LEFTARG = bit, RIGHTARG = bit, PROCEDURE = jaccard_distance,
    COMMUTATOR = '<%>'
);
"#,
    name = "bitvec_distance_functions",
    requires = ["vector_type_definition"],
);

extension_sql!(
    r#"
-- HNSW operator classes for bit vectors
CREATE OPERATOR CLASS bit_hamming_ops
    FOR TYPE bit USING hnsw AS
    OPERATOR 1 <~> (bit, bit) FOR ORDER BY float_ops,
    FUNCTION 1 hamming_distance(bit, bit),
    FUNCTION 3 hnsw_bit_support(internal);

CREATE OPERATOR CLASS bit_jaccard_ops
    FOR TYPE bit USING hnsw AS
    OPERATOR 1 <%> (bit, bit) FOR ORDER BY float_ops,
    FUNCTION 1 jaccard_distance(bit, bit),
    FUNCTION 3 hnsw_bit_support(internal);
"#,
    name = "bitvec_hnsw_opclasses",
    requires = ["bitvec_distance_functions", hnsw_handler],
);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hamming_distance() {
        let result =
            Spi::get_one::<f64>("SELECT hamming_distance('10101010'::bit(8), '10100000'::bit(8))")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - 2.0).abs() < 0.001);
    }

    #[pg_test]
    fn test_hamming_distance_identical() {
        let result =
            Spi::get_one::<f64>("SELECT hamming_distance('11110000'::bit(8), '11110000'::bit(8))")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - 0.0).abs() < 0.001);
    }

    #[pg_test]
    fn test_jaccard_distance() {
        // A = 11000000, B = 10100000
        // A AND B = 10000000 => popcount = 1
        // popcount(A) = 2, popcount(B) = 2
        // Jaccard = 1 - 1/(2+2-1) = 1 - 1/3 ≈ 0.667
        let result =
            Spi::get_one::<f64>("SELECT jaccard_distance('11000000'::bit(8), '10100000'::bit(8))")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - (1.0 - 1.0 / 3.0)).abs() < 0.001);
    }

    #[pg_test]
    fn test_jaccard_distance_identical() {
        let result =
            Spi::get_one::<f64>("SELECT jaccard_distance('11110000'::bit(8), '11110000'::bit(8))")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - 0.0).abs() < 0.001);
    }

    #[pg_test]
    fn test_jaccard_distance_disjoint() {
        // No bits in common => Jaccard = 1.0
        let result =
            Spi::get_one::<f64>("SELECT jaccard_distance('11000000'::bit(8), '00110000'::bit(8))")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - 1.0).abs() < 0.001);
    }

    #[pg_test]
    fn test_hamming_operator() {
        let result = Spi::get_one::<f64>("SELECT '10101010'::bit(8) <~> '10100000'::bit(8)")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 2.0).abs() < 0.001);
    }

    #[pg_test]
    fn test_jaccard_operator() {
        let result = Spi::get_one::<f64>("SELECT '11000000'::bit(8) <%> '10100000'::bit(8)")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - (1.0 - 1.0 / 3.0)).abs() < 0.001);
    }

    #[pg_test]
    fn test_bit_hamming_hnsw_index() {
        Spi::run("CREATE TABLE test_bit (id serial, v bit(8))").unwrap();
        Spi::run("INSERT INTO test_bit (v) VALUES ('10101010'), ('11001100'), ('11110000'), ('00001111')").unwrap();
        Spi::run("CREATE INDEX ON test_bit USING hnsw (v bit_hamming_ops)").unwrap();

        let result = Spi::get_one::<i32>(
            "SET enable_seqscan = off; SELECT id FROM test_bit ORDER BY v <~> '10101010'::bit(8) LIMIT 1",
        )
        .expect("SPI failed")
        .expect("NULL result");
        assert_eq!(result, 1);
    }

    #[pg_test]
    fn test_bit_jaccard_hnsw_index() {
        Spi::run("CREATE TABLE test_bit_j (id serial, v bit(8))").unwrap();
        Spi::run("INSERT INTO test_bit_j (v) VALUES ('10101010'), ('11001100'), ('11110000'), ('00001111')").unwrap();
        Spi::run("CREATE INDEX ON test_bit_j USING hnsw (v bit_jaccard_ops)").unwrap();

        let result = Spi::get_one::<i32>(
            "SET enable_seqscan = off; SELECT id FROM test_bit_j ORDER BY v <%> '10101010'::bit(8) LIMIT 1",
        )
        .expect("SPI failed")
        .expect("NULL result");
        assert_eq!(result, 1);
    }

    #[pg_test]
    fn test_bit_different_lengths_error() {
        let result = std::panic::catch_unwind(|| {
            Spi::get_one::<f64>("SELECT hamming_distance('1010'::bit(4), '10101010'::bit(8))")
        });
        assert!(result.is_err());
    }
}
