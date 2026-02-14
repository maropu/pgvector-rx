//! PostgreSQL `vector` type implementation.
//!
//! Provides the `vector` data type for storing fixed-dimension float32 vectors,
//! matching the original pgvector C implementation's on-disk format.
//!
//! Because the vector type uses a C-style flexible array member, we implement
//! I/O functions as raw `extern "C-unwind"` with `#[no_mangle]` and register
//! them via `extension_sql!`.

use pgrx::prelude::*;
use pgrx::{pg_guard, pg_sys};

use std::ptr;

/// Generates a `pg_finfo_<name>` function required by PostgreSQL to look up
/// a C-language function. This is the Rust equivalent of PG_FUNCTION_INFO_V1.
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

/// Maximum number of dimensions for the `vector` type.
pub const VECTOR_MAX_DIM: i32 = 16000;

/// Size of the fixed vector header: vl_len_(4) + dim(2) + unused(2) = 8 bytes.
const VECTOR_HEADER_SIZE: usize = 8;

/// Size of a `Vector` in bytes given `dim` dimensions.
#[inline]
pub const fn vector_size(dim: i32) -> usize {
    VECTOR_HEADER_SIZE + (dim as usize) * std::mem::size_of::<f32>()
}

/// On-disk vector header (fixed portion).
/// The float array `x[dim]` follows immediately after in memory.
#[repr(C)]
pub struct VectorHeader {
    pub vl_len_: i32,
    pub dim: i16,
    pub unused: i16,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn vector_isspace(ch: u8) -> bool {
    matches!(ch, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

#[inline]
fn check_dim(dim: i32) {
    if dim < 1 {
        pgrx::error!("vector must have at least 1 dimension");
    }
    if dim > VECTOR_MAX_DIM {
        pgrx::error!("vector cannot have more than {} dimensions", VECTOR_MAX_DIM);
    }
}

#[inline]
fn check_expected_dim(typmod: i32, dim: i32) {
    if typmod != -1 && typmod != dim {
        pgrx::error!("expected {} dimensions, not {}", typmod, dim);
    }
}

#[inline]
fn check_element(val: f32) {
    if val.is_nan() {
        pgrx::error!("NaN not allowed in vector");
    }
    if val.is_infinite() {
        pgrx::error!("infinite value not allowed in vector");
    }
}

/// Allocate and initialize a zero-filled Vector with `dim` dimensions.
///
/// # Safety
/// Allocates via `palloc0` in the current memory context.
pub unsafe fn init_vector(dim: i32) -> *mut VectorHeader {
    let size = vector_size(dim);
    // SAFETY: palloc0 returns valid, zeroed memory.
    let result = pg_sys::palloc0(size) as *mut VectorHeader;
    // SET_VARSIZE: store (size << 2) in the first 4 bytes for 4-byte header
    let vl_ptr = result as *mut u32;
    *vl_ptr = (size as u32) << 2;
    (*result).dim = dim as i16;
    (*result).unused = 0;
    result
}

/// L2-normalize a vector, returning a newly allocated normalized copy.
///
/// # Safety
/// `vec` must point to a valid `VectorHeader`. Allocates via `palloc0`.
pub unsafe fn l2_normalize_raw(vec: *const VectorHeader) -> *const VectorHeader {
    let dim = (*vec).dim as usize;
    let ax = vector_data(vec);

    let mut norm: f64 = 0.0;
    for i in 0..dim {
        let v = *ax.add(i) as f64;
        norm += v * v;
    }
    norm = norm.sqrt();

    let result = init_vector(dim as i32);
    let rx = vector_data_mut(result);
    if norm > 0.0 {
        for i in 0..dim {
            *rx.add(i) = (*ax.add(i) as f64 / norm) as f32;
        }
    }

    result as *const VectorHeader
}

/// Returns a mutable pointer to the float data of a vector.
///
/// # Safety
/// `vec` must point to a valid, properly-sized `VectorHeader`.
#[inline]
pub unsafe fn vector_data_mut(vec: *mut VectorHeader) -> *mut f32 {
    (vec as *mut u8).add(VECTOR_HEADER_SIZE) as *mut f32
}

/// Returns a const pointer to the float data of a vector.
///
/// # Safety
/// `vec` must point to a valid, properly-sized `VectorHeader`.
#[inline]
pub unsafe fn vector_data(vec: *const VectorHeader) -> *const f32 {
    (vec as *const u8).add(VECTOR_HEADER_SIZE) as *const f32
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

// ---------------------------------------------------------------------------
// I/O functions
// ---------------------------------------------------------------------------

pg_fn_info!(vector_in);
pg_fn_info!(vector_out);
pg_fn_info!(vector_typmod_in);
pg_fn_info!(vector_recv);
pg_fn_info!(vector_send);
pg_fn_info!(vector_cast);
pg_fn_info!(array_to_vector);
pg_fn_info!(vector_to_float4);

/// Parse text `[1,2,3]` into a vector. Matches C's `vector_in`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_in(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let input = pg_sys::DatumGetCString(fc_arg(fcinfo, 0));
    let typmod = fc_arg(fcinfo, 2).value() as i32;

    let lit = std::ffi::CStr::from_ptr(input).to_bytes();
    let mut pos = 0usize;

    // Skip leading whitespace
    while pos < lit.len() && vector_isspace(lit[pos]) {
        pos += 1;
    }

    let lit_str = || std::str::from_utf8(lit).unwrap_or("???");

    if pos >= lit.len() || lit[pos] != b'[' {
        pgrx::error!("invalid input syntax for type vector: \"{}\"", lit_str());
    }
    pos += 1;

    while pos < lit.len() && vector_isspace(lit[pos]) {
        pos += 1;
    }

    if pos < lit.len() && lit[pos] == b']' {
        pgrx::error!("vector must have at least 1 dimension");
    }

    let mut values: Vec<f32> = Vec::new();

    loop {
        if values.len() as i32 >= VECTOR_MAX_DIM {
            pgrx::error!("vector cannot have more than {} dimensions", VECTOR_MAX_DIM);
        }

        while pos < lit.len() && vector_isspace(lit[pos]) {
            pos += 1;
        }

        if pos >= lit.len() {
            pgrx::error!("invalid input syntax for type vector: \"{}\"", lit_str());
        }

        let start = pos;
        while pos < lit.len() && lit[pos] != b',' && lit[pos] != b']' && !vector_isspace(lit[pos]) {
            pos += 1;
        }

        let num_str = std::str::from_utf8(&lit[start..pos]).unwrap_or("");
        let val: f32 = match num_str.parse() {
            Ok(v) => v,
            Err(_) => {
                pgrx::error!("invalid input syntax for type vector: \"{}\"", lit_str());
            }
        };

        check_element(val);
        values.push(val);

        while pos < lit.len() && vector_isspace(lit[pos]) {
            pos += 1;
        }

        if pos < lit.len() && lit[pos] == b',' {
            pos += 1;
        } else if pos < lit.len() && lit[pos] == b']' {
            pos += 1;
            break;
        } else {
            pgrx::error!("invalid input syntax for type vector: \"{}\"", lit_str());
        }
    }

    while pos < lit.len() && vector_isspace(lit[pos]) {
        pos += 1;
    }
    if pos < lit.len() {
        pgrx::error!("invalid input syntax for type vector: \"{}\"", lit_str());
    }

    let dim = values.len() as i32;
    check_dim(dim);
    check_expected_dim(typmod, dim);

    let result = init_vector(dim);
    let data = vector_data_mut(result);
    for (i, &v) in values.iter().enumerate() {
        *data.add(i) = v;
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert vector to text `[1,2,3]`. Matches C's `vector_out`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_out(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const VectorHeader;
    let dim = (*vec).dim as usize;
    let data = vector_data(vec);

    let mut buf = String::with_capacity(dim * 10 + 3);
    buf.push('[');
    for i in 0..dim {
        if i > 0 {
            buf.push(',');
        }
        // Format float using ryu, then strip trailing ".0" to match
        // PostgreSQL's float_to_shortest_decimal_bufn output.
        let mut ryu_buf = ryu::Buffer::new();
        let formatted = ryu_buf.format(*data.add(i));
        if let Some(stripped) = formatted.strip_suffix(".0") {
            buf.push_str(stripped);
        } else {
            buf.push_str(formatted);
        }
    }
    buf.push(']');

    let cstr = std::ffi::CString::new(buf).unwrap();
    let result = pg_sys::pstrdup(cstr.as_ptr());
    pg_sys::Datum::from(result as usize)
}

/// Parse type modifier for `vector(N)`. Matches C's `vector_typmod_in`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_typmod_in(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let ta = fc_arg(fcinfo, 0).cast_mut_ptr::<pg_sys::ArrayType>();
    let mut n: i32 = 0;
    let tl = pg_sys::ArrayGetIntegerTypmods(ta, &mut n);

    if n != 1 {
        pgrx::error!("invalid type modifier");
    }

    let dim = *tl;
    if dim < 1 {
        pgrx::error!("dimensions for type vector must be at least 1");
    }
    if dim > VECTOR_MAX_DIM {
        pgrx::error!(
            "dimensions for type vector cannot exceed {}",
            VECTOR_MAX_DIM
        );
    }

    pg_sys::Datum::from(dim as usize)
}

/// Receive vector from binary format. Matches C's `vector_recv`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_recv(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let stringinfo = fc_arg(fcinfo, 0).cast_mut_ptr::<pg_sys::StringInfoData>();
    let typmod = fc_arg(fcinfo, 2).value() as i32;

    let dim = pg_sys::pq_getmsgint(stringinfo, 2) as i16;
    let unused = pg_sys::pq_getmsgint(stringinfo, 2) as i16;

    check_dim(dim as i32);
    check_expected_dim(typmod, dim as i32);

    if unused != 0 {
        pgrx::error!("expected unused to be 0, not {}", unused);
    }

    let result = init_vector(dim as i32);
    let data = vector_data_mut(result);
    for i in 0..dim as usize {
        let val = pg_sys::pq_getmsgfloat4(stringinfo);
        check_element(val);
        *data.add(i) = val;
    }

    pg_sys::Datum::from(result as usize)
}

/// Send vector in binary format. Matches C's `vector_send`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_send(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const VectorHeader;
    let dim = (*vec).dim;
    let unused = (*vec).unused;
    let data = vector_data(vec);

    let mut buf: pg_sys::StringInfoData = std::mem::zeroed();
    pg_sys::pq_begintypsend(&mut buf);

    // dim as 2 bytes big-endian
    let dim_be = (dim as u16).to_be_bytes();
    pg_sys::pq_sendbytes(&mut buf, dim_be.as_ptr() as *const std::ffi::c_void, 2);

    // unused as 2 bytes big-endian
    let unused_be = (unused as u16).to_be_bytes();
    pg_sys::pq_sendbytes(&mut buf, unused_be.as_ptr() as *const std::ffi::c_void, 2);

    for i in 0..dim as usize {
        pg_sys::pq_sendfloat4(&mut buf, *data.add(i));
    }

    let result = pg_sys::pq_endtypsend(&mut buf);
    pg_sys::Datum::from(result as usize)
}

/// Cast vector to vector (typmod enforcement). Matches C's `vector()`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_cast(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let typmod = fc_arg(fcinfo, 1).value() as i32;

    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const VectorHeader;
    check_expected_dim(typmod, (*vec).dim as i32);

    raw
}

/// Convert a PostgreSQL array (integer[], real[], double precision[], or
/// numeric[]) to a vector. Matches C's `array_to_vector`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn array_to_vector(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    // SAFETY: DatumGetArrayTypeP is a C macro; replicate as pg_detoast_datum.
    let array =
        pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *mut pg_sys::ArrayType;
    let typmod = fc_arg(fcinfo, 1).value() as i32;

    if (*array).ndim > 1 {
        pgrx::error!("array must be 1-D");
    }

    if pg_sys::array_contains_nulls(array) {
        pgrx::error!("array must not contain nulls");
    }

    let elem_type = (*array).elemtype;
    let mut typlen: i16 = 0;
    let mut typbyval: bool = false;
    let mut typalign: i8 = 0;
    pg_sys::get_typlenbyvalalign(elem_type, &mut typlen, &mut typbyval, &mut typalign);

    let mut elems: *mut pg_sys::Datum = ptr::null_mut();
    let mut nelems: i32 = 0;
    pg_sys::deconstruct_array(
        array,
        elem_type,
        typlen as i32,
        typbyval,
        typalign,
        &mut elems,
        ptr::null_mut(),
        &mut nelems,
    );

    let dim = nelems;
    check_dim(dim);
    check_expected_dim(typmod, dim);

    let result = init_vector(dim);
    let rx = vector_data_mut(result);

    if elem_type == pg_sys::INT4OID {
        for i in 0..dim as usize {
            *rx.add(i) = pg_sys::DatumGetInt32(*elems.add(i)) as f32;
        }
    } else if elem_type == pg_sys::FLOAT8OID {
        for i in 0..dim as usize {
            *rx.add(i) = pg_sys::DatumGetFloat8(*elems.add(i)) as f32;
        }
    } else if elem_type == pg_sys::FLOAT4OID {
        for i in 0..dim as usize {
            *rx.add(i) = pg_sys::DatumGetFloat4(*elems.add(i));
        }
    } else if elem_type == pg_sys::NUMERICOID {
        for i in 0..dim as usize {
            // SAFETY: Use OidFunctionCall to invoke numeric_float4 by OID.
            *rx.add(i) = pg_sys::DatumGetFloat4(pg_sys::OidFunctionCall1Coll(
                pg_sys::Oid::from_u32(pg_sys::F_NUMERIC_FLOAT4),
                pg_sys::InvalidOid,
                *elems.add(i),
            ));
        }
    } else {
        pgrx::error!("unsupported array type");
    }

    pg_sys::pfree(elems as *mut std::ffi::c_void);

    for i in 0..dim as usize {
        check_element(*rx.add(i));
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert a vector to a float4[] array. Matches C's `vector_to_float4`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_to_float4(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const VectorHeader;
    let dim = (*vec).dim as usize;
    let data = vector_data(vec);

    let datums = pg_sys::palloc(dim * std::mem::size_of::<pg_sys::Datum>()) as *mut pg_sys::Datum;
    for i in 0..dim {
        *datums.add(i) = pg_sys::Float4GetDatum(*data.add(i));
    }

    // SAFETY: construct_array builds a valid PostgreSQL ArrayType.
    // TYPALIGN_INT = 'i' = 105
    let result = pg_sys::construct_array(
        datums,
        dim as i32,
        pg_sys::FLOAT4OID,
        std::mem::size_of::<f32>() as i32,
        true,
        105_i8, // TYPALIGN_INT = 'i'
    );

    pg_sys::pfree(datums as *mut std::ffi::c_void);

    pg_sys::Datum::from(result as usize)
}

// ---------------------------------------------------------------------------
// Distance / utility helpers (pure Rust, no PostgreSQL dependency)
// ---------------------------------------------------------------------------

/// Ensure two vectors have the same dimensions.
#[inline]
unsafe fn check_dims(a: *const VectorHeader, b: *const VectorHeader) {
    if (*a).dim != (*b).dim {
        pgrx::error!("different vector dimensions {} and {}", (*a).dim, (*b).dim);
    }
}

/// L2 squared distance: sum((a[i] - b[i])^2).
#[inline]
fn compute_l2_squared(dim: usize, ax: *const f32, bx: *const f32) -> f32 {
    let mut distance: f32 = 0.0;
    for i in 0..dim {
        // SAFETY: caller guarantees valid pointers and dim.
        let diff = unsafe { *ax.add(i) - *bx.add(i) };
        distance += diff * diff;
    }
    distance
}

/// Inner product: sum(a[i] * b[i]).
#[inline]
fn compute_inner_product(dim: usize, ax: *const f32, bx: *const f32) -> f32 {
    let mut distance: f32 = 0.0;
    for i in 0..dim {
        // SAFETY: caller guarantees valid pointers and dim.
        distance += unsafe { *ax.add(i) * *bx.add(i) };
    }
    distance
}

/// Cosine similarity: dot(a,b) / sqrt(norm(a) * norm(b)).
#[inline]
fn compute_cosine_similarity(dim: usize, ax: *const f32, bx: *const f32) -> f64 {
    let mut similarity: f32 = 0.0;
    let mut norma: f32 = 0.0;
    let mut normb: f32 = 0.0;
    for i in 0..dim {
        // SAFETY: caller guarantees valid pointers and dim.
        unsafe {
            let ai = *ax.add(i);
            let bi = *bx.add(i);
            similarity += ai * bi;
            norma += ai * ai;
            normb += bi * bi;
        }
    }
    (similarity as f64) / ((norma as f64) * (normb as f64)).sqrt()
}

/// L1 distance: sum(|a[i] - b[i]|).
#[inline]
fn compute_l1_distance(dim: usize, ax: *const f32, bx: *const f32) -> f32 {
    let mut distance: f32 = 0.0;
    for i in 0..dim {
        // SAFETY: caller guarantees valid pointers and dim.
        distance += unsafe { (*ax.add(i) - *bx.add(i)).abs() };
    }
    distance
}

// ---------------------------------------------------------------------------
// Distance / utility PostgreSQL functions
// ---------------------------------------------------------------------------

pg_fn_info!(l2_distance);
pg_fn_info!(vector_l2_squared_distance);
pg_fn_info!(inner_product);
pg_fn_info!(vector_negative_inner_product);
pg_fn_info!(cosine_distance);
pg_fn_info!(l1_distance);
pg_fn_info!(vector_dims);
pg_fn_info!(vector_norm);
pg_fn_info!(l2_normalize);

/// L2 (Euclidean) distance between two vectors.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn l2_distance(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VectorHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = (compute_l2_squared(dim, vector_data(a), vector_data(b)) as f64).sqrt();
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// L2 squared distance (avoids sqrt, used by HNSW operator class).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_l2_squared_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VectorHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = compute_l2_squared(dim, vector_data(a), vector_data(b)) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Inner product of two vectors.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn inner_product(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VectorHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = compute_inner_product(dim, vector_data(a), vector_data(b)) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Negative inner product (used by HNSW operator class for IP distance).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_negative_inner_product(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VectorHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = -(compute_inner_product(dim, vector_data(a), vector_data(b)) as f64);
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Cosine distance: 1 - cosine_similarity(a, b).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn cosine_distance(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VectorHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let similarity = compute_cosine_similarity(dim, vector_data(a), vector_data(b));
    // Clamp to [-1, 1] to handle floating point errors
    let dist = 1.0 - similarity.clamp(-1.0, 1.0);
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// L1 (Manhattan) distance between two vectors.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn l1_distance(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const VectorHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = compute_l1_distance(dim, vector_data(a), vector_data(b)) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Get the number of dimensions of a vector.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_dims(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    pg_sys::Datum::from((*a).dim as i32)
}

/// Get the L2 norm of a vector.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_norm(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let ax = vector_data(a);
    let dim = (*a).dim as usize;
    let mut norm: f64 = 0.0;
    for i in 0..dim {
        let v = *ax.add(i) as f64;
        norm += v * v;
    }
    let result = norm.sqrt();
    pg_sys::Datum::from(f64::to_bits(result))
}

/// L2-normalize a vector to unit length.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn l2_normalize(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let ax = vector_data(a);
    let dim = (*a).dim as usize;

    let mut norm: f64 = 0.0;
    for i in 0..dim {
        let v = *ax.add(i) as f64;
        norm += v * v;
    }
    norm = norm.sqrt();

    let result = init_vector(dim as i32);
    let rx = vector_data_mut(result);
    if norm > 0.0 {
        for i in 0..dim {
            *rx.add(i) = (*ax.add(i) as f64 / norm) as f32;
        }
    }

    pg_sys::Datum::from(result as usize)
}

// ---------------------------------------------------------------------------
// SQL registration
// ---------------------------------------------------------------------------

extension_sql!(
    r#"
CREATE TYPE vector;

CREATE FUNCTION vector_in(cstring, oid, integer) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_out(vector) RETURNS cstring
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_recv(internal, oid, integer) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_send(vector) RETURNS bytea
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE vector (
    INPUT     = vector_in,
    OUTPUT    = vector_out,
    TYPMOD_IN = vector_typmod_in,
    RECEIVE   = vector_recv,
    SEND      = vector_send,
    STORAGE   = external
);

CREATE FUNCTION vector_cast(vector, integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vector(integer[], integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vector(real[], integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vector(double precision[], integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_vector(numeric[], integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_to_float4(vector, integer, boolean) RETURNS real[]
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE CAST (vector AS vector)
    WITH FUNCTION vector_cast(vector, integer, boolean) AS IMPLICIT;

CREATE CAST (vector AS real[])
    WITH FUNCTION vector_to_float4(vector, integer, boolean) AS IMPLICIT;

CREATE CAST (integer[] AS vector)
    WITH FUNCTION array_to_vector(integer[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (real[] AS vector)
    WITH FUNCTION array_to_vector(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (double precision[] AS vector)
    WITH FUNCTION array_to_vector(double precision[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (numeric[] AS vector)
    WITH FUNCTION array_to_vector(numeric[], integer, boolean) AS ASSIGNMENT;
"#,
    name = "vector_type_definition",
    bootstrap,
);

extension_sql!(
    r#"
-- Distance functions
CREATE FUNCTION l2_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_l2_squared_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION inner_product(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_negative_inner_product(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l1_distance(vector, vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_dims(vector) RETURNS integer
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_norm(vector) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l2_normalize(vector) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Distance operators
CREATE OPERATOR <-> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = vector, RIGHTARG = vector,
    PROCEDURE = vector_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = cosine_distance,
    COMMUTATOR = '<=>'
);

CREATE OPERATOR <+> (
    LEFTARG = vector, RIGHTARG = vector, PROCEDURE = l1_distance,
    COMMUTATOR = '<+>'
);
"#,
    name = "vector_distance_functions",
    requires = ["vector_type_definition"],
);

extension_sql!(
    r#"
-- HNSW operator classes
CREATE OPERATOR CLASS vector_l2_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_l2_squared_distance(vector, vector);

CREATE OPERATOR CLASS vector_ip_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector);

CREATE OPERATOR CLASS vector_cosine_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 vector_negative_inner_product(vector, vector),
    FUNCTION 2 vector_norm(vector);

CREATE OPERATOR CLASS vector_l1_ops
    FOR TYPE vector USING hnsw AS
    OPERATOR 1 <+> (vector, vector) FOR ORDER BY float_ops,
    FUNCTION 1 l1_distance(vector, vector);
"#,
    name = "vector_hnsw_opclasses",
    requires = ["vector_distance_functions", hnsw_handler],
);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_vector_type_exists() {
        let result =
            Spi::get_one::<String>("SELECT typname::text FROM pg_type WHERE typname = 'vector'")
                .expect("SPI failed")
                .expect("vector type not found");
        assert_eq!(result, "vector");
    }

    #[pg_test]
    fn test_vector_in_out_basic() {
        let result = Spi::get_one::<String>("SELECT '[1,2,3]'::vector::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    fn test_vector_in_out_floats() {
        let result = Spi::get_one::<String>("SELECT '[1.5,2.25,3.125]'::vector::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1.5,2.25,3.125]");
    }

    #[pg_test]
    fn test_vector_with_typmod() {
        let result = Spi::get_one::<String>("SELECT '[1,2,3]'::vector(3)::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    #[should_panic(expected = "expected 3 dimensions, not 2")]
    fn test_vector_typmod_mismatch() {
        Spi::get_one::<String>("SELECT '[1,2]'::vector(3)::text").ok();
    }

    #[pg_test]
    #[should_panic(expected = "vector must have at least 1 dimension")]
    fn test_vector_empty() {
        Spi::get_one::<String>("SELECT '[]'::vector::text").ok();
    }

    #[pg_test]
    #[should_panic(expected = "NaN not allowed in vector")]
    fn test_vector_nan() {
        Spi::get_one::<String>("SELECT '[NaN]'::vector::text").ok();
    }

    #[pg_test]
    #[should_panic(expected = "infinite value not allowed in vector")]
    fn test_vector_inf() {
        Spi::get_one::<String>("SELECT '[Infinity]'::vector::text").ok();
    }

    #[pg_test]
    fn test_vector_in_table() {
        Spi::run("CREATE TABLE test_vec (id serial, val vector(3))").unwrap();
        Spi::run("INSERT INTO test_vec (val) VALUES ('[1,2,3]'), ('[4,5,6]')").unwrap();
        let count = Spi::get_one::<i64>("SELECT count(*) FROM test_vec")
            .expect("SPI failed")
            .expect("NULL count");
        assert_eq!(count, 2);
    }

    #[pg_test]
    fn test_vector_null_insert() {
        Spi::run("CREATE TABLE test_vec_null (id serial, val vector(3))").unwrap();
        Spi::run("INSERT INTO test_vec_null (val) VALUES (NULL)").unwrap();
        let count = Spi::get_one::<i64>("SELECT count(*) FROM test_vec_null WHERE val IS NULL")
            .expect("SPI failed")
            .expect("NULL count");
        assert_eq!(count, 1);
    }

    // ----- Distance function tests -----

    #[pg_test]
    fn test_l2_distance() {
        let result = Spi::get_one::<f64>("SELECT l2_distance('[0,0]'::vector, '[3,4]'::vector)")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_l2_distance_same() {
        let result =
            Spi::get_one::<f64>("SELECT l2_distance('[1,2,3]'::vector, '[1,2,3]'::vector)")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_inner_product() {
        let result =
            Spi::get_one::<f64>("SELECT inner_product('[1,2,3]'::vector, '[4,5,6]'::vector)")
                .expect("SPI failed")
                .expect("NULL result");
        // 1*4 + 2*5 + 3*6 = 32
        assert!((result - 32.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_cosine_distance_identical() {
        let result =
            Spi::get_one::<f64>("SELECT cosine_distance('[1,2,3]'::vector, '[1,2,3]'::vector)")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - 0.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_cosine_distance_orthogonal() {
        let result =
            Spi::get_one::<f64>("SELECT cosine_distance('[1,0]'::vector, '[0,1]'::vector)")
                .expect("SPI failed")
                .expect("NULL result");
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_l1_distance() {
        let result =
            Spi::get_one::<f64>("SELECT l1_distance('[1,2,3]'::vector, '[4,6,8]'::vector)")
                .expect("SPI failed")
                .expect("NULL result");
        // |1-4| + |2-6| + |3-8| = 3 + 4 + 5 = 12
        assert!((result - 12.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_vector_dims_fn() {
        let result = Spi::get_one::<i32>("SELECT vector_dims('[1,2,3,4,5]'::vector)")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, 5);
    }

    #[pg_test]
    fn test_vector_norm() {
        let result = Spi::get_one::<f64>("SELECT vector_norm('[3,4]'::vector)")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[pg_test]
    #[should_panic(expected = "different vector dimensions")]
    fn test_l2_distance_dim_mismatch() {
        Spi::get_one::<f64>("SELECT l2_distance('[1,2]'::vector, '[1,2,3]'::vector)").ok();
    }

    // ----- Operator tests -----

    #[pg_test]
    fn test_l2_operator() {
        let result = Spi::get_one::<f64>("SELECT '[0,0]'::vector <-> '[3,4]'::vector")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_ip_operator() {
        // <#> returns negative inner product
        let result = Spi::get_one::<f64>("SELECT '[1,2,3]'::vector <#> '[4,5,6]'::vector")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - (-32.0)).abs() < 1e-6);
    }

    #[pg_test]
    fn test_cosine_operator() {
        let result = Spi::get_one::<f64>("SELECT '[1,0]'::vector <=> '[0,1]'::vector")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 1.0).abs() < 1e-6);
    }

    #[pg_test]
    fn test_l1_operator() {
        let result = Spi::get_one::<f64>("SELECT '[1,2,3]'::vector <+> '[4,6,8]'::vector")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 12.0).abs() < 1e-6);
    }

    // ----- Operator class tests -----

    #[pg_test]
    fn test_hnsw_opclass_l2_exists() {
        let result = Spi::get_one::<String>(
            "SELECT opcname::text FROM pg_opclass WHERE opcname = 'vector_l2_ops' \
             AND opcmethod = (SELECT oid FROM pg_am WHERE amname = 'hnsw')",
        )
        .expect("SPI failed")
        .expect("opclass not found");
        assert_eq!(result, "vector_l2_ops");
    }

    #[pg_test]
    fn test_hnsw_opclass_ip_exists() {
        let result = Spi::get_one::<String>(
            "SELECT opcname::text FROM pg_opclass WHERE opcname = 'vector_ip_ops' \
             AND opcmethod = (SELECT oid FROM pg_am WHERE amname = 'hnsw')",
        )
        .expect("SPI failed")
        .expect("opclass not found");
        assert_eq!(result, "vector_ip_ops");
    }

    #[pg_test]
    fn test_hnsw_opclass_cosine_exists() {
        let result = Spi::get_one::<String>(
            "SELECT opcname::text FROM pg_opclass WHERE opcname = 'vector_cosine_ops' \
             AND opcmethod = (SELECT oid FROM pg_am WHERE amname = 'hnsw')",
        )
        .expect("SPI failed")
        .expect("opclass not found");
        assert_eq!(result, "vector_cosine_ops");
    }

    #[pg_test]
    fn test_hnsw_opclass_l1_exists() {
        let result = Spi::get_one::<String>(
            "SELECT opcname::text FROM pg_opclass WHERE opcname = 'vector_l1_ops' \
             AND opcmethod = (SELECT oid FROM pg_am WHERE amname = 'hnsw')",
        )
        .expect("SPI failed")
        .expect("opclass not found");
        assert_eq!(result, "vector_l1_ops");
    }

    #[pg_test]
    fn test_array_to_vector_float8() {
        let result = Spi::get_one::<String>(
            "SELECT (ARRAY[1.0, 2.0, 3.0]::double precision[])::vector::text",
        )
        .expect("SPI failed")
        .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    fn test_array_to_vector_float4() {
        let result = Spi::get_one::<String>("SELECT (ARRAY[1.5, 2.5, 3.5]::real[])::vector::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1.5,2.5,3.5]");
    }

    #[pg_test]
    fn test_array_to_vector_int4() {
        let result = Spi::get_one::<String>("SELECT (ARRAY[1, 2, 3]::integer[])::vector::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    fn test_vector_to_float4() {
        let result = Spi::get_one::<String>("SELECT ('[1,2,3]'::vector)::real[]::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "{1,2,3}");
    }

    #[pg_test]
    fn test_array_random_to_vector() {
        // This is the pattern used by Perl TAP tests
        let result =
            Spi::get_one::<i32>("SELECT vector_dims(ARRAY[random(), random(), random()]::vector)")
                .expect("SPI failed")
                .expect("NULL result");
        assert_eq!(result, 3);
    }

    #[pg_test]
    fn test_array_to_vector_insert() {
        Spi::run("CREATE TABLE test_arr_cast (v vector(3))").expect("create table");
        Spi::run(
            "INSERT INTO test_arr_cast SELECT ARRAY[random(), random(), random()] \
             FROM generate_series(1, 10) i",
        )
        .expect("insert");
        let count = Spi::get_one::<i64>("SELECT COUNT(*) FROM test_arr_cast")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(count, 10);
        Spi::run("DROP TABLE test_arr_cast").expect("drop");
    }
}
