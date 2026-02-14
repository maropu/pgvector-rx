//! PostgreSQL `sparsevec` type implementation.
//!
//! Provides the `sparsevec` data type for storing sparse vectors with
//! integer indices and float32 values. Matches the original pgvector C
//! implementation's on-disk format: `{idx:val, ...}/dim`.
//!
//! The on-disk format stores 0-based indices (C convention), while the
//! text representation uses 1-based indices (SQL convention).

use pgrx::prelude::*;
use pgrx::{pg_guard, pg_sys};

use std::ptr;

/// Generates a `pg_finfo_<name>` function required by PostgreSQL.
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

/// Maximum number of dimensions for the `sparsevec` type.
pub const SPARSEVEC_MAX_DIM: i32 = 1_000_000_000;

/// Maximum number of non-zero elements.
pub const SPARSEVEC_MAX_NNZ: i32 = 16000;

/// Size of the fixed header: vl_len_(4) + dim(4) + nnz(4) + unused(4) = 16 bytes.
const SPARSEVEC_HEADER_SIZE: usize = 16;

/// On-disk sparse vector header. The indices `[nnz]` and values `[nnz]` follow.
#[repr(C)]
pub struct SparseVecHeader {
    pub vl_len_: i32,
    pub dim: i32,
    pub nnz: i32,
    pub unused: i32,
    // indices: [i32; nnz] follows immediately
    // values:  [f32; nnz] follows after indices
}

/// Total size of a SparseVector in bytes given `nnz` non-zero elements.
#[inline]
pub const fn sparsevec_size(nnz: i32) -> usize {
    SPARSEVEC_HEADER_SIZE
        + (nnz as usize) * std::mem::size_of::<i32>()
        + (nnz as usize) * std::mem::size_of::<f32>()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read argument N from a FunctionCallInfo as a raw Datum.
///
/// # Safety
/// `fcinfo` must be valid and `n` must be < nargs.
#[inline]
unsafe fn fc_arg(fcinfo: pg_sys::FunctionCallInfo, n: usize) -> pg_sys::Datum {
    let args_ptr: *const pg_sys::NullableDatum = ptr::addr_of!((*fcinfo).args).cast();
    (*args_ptr.add(n)).value
}

/// Returns a pointer to the indices array of a sparse vector.
///
/// # Safety
/// `vec` must point to a valid `SparseVecHeader`.
#[inline]
pub unsafe fn sparsevec_indices(vec: *const SparseVecHeader) -> *const i32 {
    (vec as *const u8).add(SPARSEVEC_HEADER_SIZE) as *const i32
}

/// Returns a mutable pointer to the indices array of a sparse vector.
///
/// # Safety
/// `vec` must point to a valid `SparseVecHeader`.
#[inline]
pub unsafe fn sparsevec_indices_mut(vec: *mut SparseVecHeader) -> *mut i32 {
    (vec as *mut u8).add(SPARSEVEC_HEADER_SIZE) as *mut i32
}

/// Returns a pointer to the values array of a sparse vector.
///
/// # Safety
/// `vec` must point to a valid `SparseVecHeader`.
#[inline]
pub unsafe fn sparsevec_values(vec: *const SparseVecHeader) -> *const f32 {
    let nnz = (*vec).nnz as usize;
    let offset = SPARSEVEC_HEADER_SIZE + nnz * std::mem::size_of::<i32>();
    (vec as *const u8).add(offset) as *const f32
}

/// Returns a mutable pointer to the values array of a sparse vector.
///
/// # Safety
/// `vec` must point to a valid `SparseVecHeader`.
#[inline]
pub unsafe fn sparsevec_values_mut(vec: *mut SparseVecHeader) -> *mut f32 {
    let nnz = (*vec).nnz as usize;
    let offset = SPARSEVEC_HEADER_SIZE + nnz * std::mem::size_of::<i32>();
    (vec as *mut u8).add(offset) as *mut f32
}

/// Allocate and initialize a zero-filled SparseVector.
///
/// # Safety
/// Allocates via `palloc0` in the current memory context.
pub unsafe fn init_sparsevec(dim: i32, nnz: i32) -> *mut SparseVecHeader {
    let size = sparsevec_size(nnz);
    // SAFETY: palloc0 returns valid, zeroed memory.
    let result = pg_sys::palloc0(size) as *mut SparseVecHeader;
    // SET_VARSIZE
    let vl_ptr = result as *mut u32;
    *vl_ptr = (size as u32) << 2;
    (*result).dim = dim;
    (*result).nnz = nnz;
    result
}

#[inline]
fn sparsevec_isspace(ch: u8) -> bool {
    matches!(ch, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

#[inline]
fn check_dim(dim: i32) {
    if dim < 1 {
        pgrx::error!("sparsevec must have at least 1 dimension");
    }
    if dim > SPARSEVEC_MAX_DIM {
        pgrx::error!(
            "sparsevec cannot have more than {} dimensions",
            SPARSEVEC_MAX_DIM
        );
    }
}

#[inline]
fn check_expected_dim(typmod: i32, dim: i32) {
    if typmod != -1 && typmod != dim {
        pgrx::error!("expected {} dimensions, not {}", typmod, dim);
    }
}

#[inline]
fn check_nnz(nnz: i32, dim: i32) {
    if nnz < 0 {
        pgrx::error!("sparsevec cannot have negative number of elements");
    }
    if nnz > SPARSEVEC_MAX_NNZ {
        pgrx::error!(
            "sparsevec cannot have more than {} non-zero elements",
            SPARSEVEC_MAX_NNZ
        );
    }
    if nnz > dim {
        pgrx::error!("sparsevec cannot have more elements than dimensions");
    }
}

/// Validate index at position `i` in sorted indices array.
///
/// # Safety
/// `indices` must point to at least `i+1` valid i32 values.
#[inline]
unsafe fn check_index(indices: *const i32, i: usize, dim: i32) {
    let index = *indices.add(i);
    if index < 0 || index >= dim {
        pgrx::error!("sparsevec index out of bounds");
    }
    if i > 0 {
        let prev = *indices.add(i - 1);
        if index < prev {
            pgrx::error!("sparsevec indices must be in ascending order");
        }
        if index == prev {
            pgrx::error!("sparsevec indices must not contain duplicates");
        }
    }
}

#[inline]
fn check_element(value: f32) {
    if value.is_nan() {
        pgrx::error!("NaN not allowed in sparsevec");
    }
    if value.is_infinite() {
        pgrx::error!("infinite value not allowed in sparsevec");
    }
}

// ---------------------------------------------------------------------------
// I/O functions
// ---------------------------------------------------------------------------

pg_fn_info!(sparsevec_in);
pg_fn_info!(sparsevec_out);
pg_fn_info!(sparsevec_typmod_in);
pg_fn_info!(sparsevec_recv);
pg_fn_info!(sparsevec_send);
pg_fn_info!(sparsevec_cast);
pg_fn_info!(vector_to_sparsevec);
pg_fn_info!(halfvec_to_sparsevec);
pg_fn_info!(sparsevec_to_vector);
pg_fn_info!(sparsevec_to_halfvec);
pg_fn_info!(array_to_sparsevec);

/// Parse text `{idx:val, ...}/dim` into a sparse vector.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_in(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let input = pg_sys::DatumGetCString(fc_arg(fcinfo, 0));
    let typmod = fc_arg(fcinfo, 2).value() as i32;

    let lit = std::ffi::CStr::from_ptr(input).to_bytes();
    let mut pos = 0usize;
    let lit_str = || std::str::from_utf8(lit).unwrap_or("???");

    // Count commas for max nnz estimate
    let max_nnz = lit.iter().filter(|&&c| c == b',').count() as i32 + 1;
    if max_nnz > SPARSEVEC_MAX_NNZ {
        pgrx::error!(
            "sparsevec cannot have more than {} non-zero elements",
            SPARSEVEC_MAX_NNZ
        );
    }

    struct SparseInputElement {
        index: i32,
        value: f32,
    }
    let mut elements: Vec<SparseInputElement> = Vec::with_capacity(max_nnz as usize);

    // Skip whitespace
    while pos < lit.len() && sparsevec_isspace(lit[pos]) {
        pos += 1;
    }

    if pos >= lit.len() || lit[pos] != b'{' {
        pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
    }
    pos += 1;

    while pos < lit.len() && sparsevec_isspace(lit[pos]) {
        pos += 1;
    }

    if pos < lit.len() && lit[pos] == b'}' {
        pos += 1;
    } else {
        loop {
            while pos < lit.len() && sparsevec_isspace(lit[pos]) {
                pos += 1;
            }

            if pos >= lit.len() {
                pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
            }

            // Parse index (integer)
            let idx_start = pos;
            if pos < lit.len() && (lit[pos] == b'-' || lit[pos] == b'+') {
                pos += 1;
            }
            while pos < lit.len() && lit[pos].is_ascii_digit() {
                pos += 1;
            }
            if pos == idx_start
                || (pos == idx_start + 1 && (lit[idx_start] == b'-' || lit[idx_start] == b'+'))
            {
                pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
            }
            let idx_str = std::str::from_utf8(&lit[idx_start..pos]).unwrap_or("");
            let index: i64 = match idx_str.parse() {
                Ok(v) => v,
                Err(_) => {
                    // Clamp to i32 range like C
                    if idx_str.starts_with('-') {
                        (i32::MIN as i64) + 1
                    } else {
                        i32::MAX as i64
                    }
                }
            };

            while pos < lit.len() && sparsevec_isspace(lit[pos]) {
                pos += 1;
            }

            if pos >= lit.len() || lit[pos] != b':' {
                pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
            }
            pos += 1;

            while pos < lit.len() && sparsevec_isspace(lit[pos]) {
                pos += 1;
            }

            // Parse value (float)
            let val_start = pos;
            // Allow sign, digits, '.', 'e'/'E', 'i'/'n'/'I'/'N' (for inf/nan)
            while pos < lit.len()
                && (lit[pos].is_ascii_digit()
                    || lit[pos] == b'.'
                    || lit[pos] == b'-'
                    || lit[pos] == b'+'
                    || lit[pos] == b'e'
                    || lit[pos] == b'E'
                    || lit[pos] == b'i'
                    || lit[pos] == b'n'
                    || lit[pos] == b'f'
                    || lit[pos] == b'I'
                    || lit[pos] == b'N'
                    || lit[pos] == b'F'
                    || lit[pos] == b'a'
                    || lit[pos] == b'A')
            {
                pos += 1;
            }
            let val_str = std::str::from_utf8(&lit[val_start..pos]).unwrap_or("");
            if val_str.is_empty() {
                pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
            }
            let value: f32 = match val_str.parse() {
                Ok(v) => v,
                Err(_) => {
                    pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
                }
            };

            check_element(value);

            // Do not store zero values; convert 1-based (SQL) to 0-based (C)
            if value != 0.0 {
                let index_i32 = index.clamp(i32::MIN as i64 + 1, i32::MAX as i64) as i32;
                elements.push(SparseInputElement {
                    index: index_i32 - 1,
                    value,
                });
            }

            while pos < lit.len() && sparsevec_isspace(lit[pos]) {
                pos += 1;
            }

            if pos < lit.len() && lit[pos] == b',' {
                pos += 1;
            } else if pos < lit.len() && lit[pos] == b'}' {
                pos += 1;
                break;
            } else {
                pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
            }
        }
    }

    while pos < lit.len() && sparsevec_isspace(lit[pos]) {
        pos += 1;
    }

    if pos >= lit.len() || lit[pos] != b'/' {
        pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
    }
    pos += 1;

    while pos < lit.len() && sparsevec_isspace(lit[pos]) {
        pos += 1;
    }

    // Parse dimension
    let dim_start = pos;
    if pos < lit.len() && (lit[pos] == b'-' || lit[pos] == b'+') {
        pos += 1;
    }
    while pos < lit.len() && lit[pos].is_ascii_digit() {
        pos += 1;
    }
    if pos == dim_start {
        pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
    }
    let dim_str = std::str::from_utf8(&lit[dim_start..pos]).unwrap_or("");
    let dim: i32 = match dim_str.parse::<i64>() {
        Ok(v) => v.clamp(i32::MIN as i64, i32::MAX as i64) as i32,
        Err(_) => {
            if dim_str.starts_with('-') {
                i32::MIN
            } else {
                i32::MAX
            }
        }
    };

    // Only whitespace after dimension
    while pos < lit.len() && sparsevec_isspace(lit[pos]) {
        pos += 1;
    }
    if pos != lit.len() {
        pgrx::error!("invalid input syntax for type sparsevec: \"{}\"", lit_str());
    }

    check_dim(dim);
    check_expected_dim(typmod, dim);

    // Sort elements by index
    elements.sort_by_key(|e| e.index);

    let nnz = elements.len() as i32;
    let result = init_sparsevec(dim, nnz);
    let r_indices = sparsevec_indices_mut(result);
    let r_values = sparsevec_values_mut(result);

    for (i, elem) in elements.iter().enumerate() {
        *r_indices.add(i) = elem.index;
        *r_values.add(i) = elem.value;
        check_index(r_indices, i, dim);
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert sparse vector to text `{idx:val, ...}/dim`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_out(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let svec = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let indices = sparsevec_indices(svec);
    let values = sparsevec_values(svec);
    let nnz = (*svec).nnz as usize;

    // Build output string
    let mut s = String::with_capacity(nnz * 20 + 16);
    s.push('{');
    for i in 0..nnz {
        if i > 0 {
            s.push(',');
        }
        // Convert 0-based (C) to 1-based (SQL)
        let idx = *indices.add(i) + 1;
        let val = *values.add(i);
        use std::fmt::Write;
        write!(s, "{}:{}", idx, format_float(val)).unwrap();
    }
    s.push('}');
    s.push('/');
    use std::fmt::Write;
    write!(s, "{}", (*svec).dim).unwrap();

    // Allocate cstring in palloc
    let cstr = pg_sys::palloc(s.len() + 1) as *mut u8;
    ptr::copy_nonoverlapping(s.as_ptr(), cstr, s.len());
    *cstr.add(s.len()) = 0;

    pg_sys::Datum::from(cstr as usize)
}

/// Format a float matching PostgreSQL's float_to_shortest_decimal_bufn behavior.
fn format_float(v: f32) -> String {
    if v == 0.0 {
        if v.is_sign_negative() {
            return "-0".to_string();
        }
        return "0".to_string();
    }
    // Use Rust's Display which produces shortest representation
    let s = format!("{}", v);
    // Ensure no trailing dot
    s
}

/// Parse typmod for sparsevec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_typmod_in(
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
        pgrx::error!("dimensions for type sparsevec must be at least 1");
    }
    if dim > SPARSEVEC_MAX_DIM {
        pgrx::error!(
            "dimensions for type sparsevec cannot exceed {}",
            SPARSEVEC_MAX_DIM
        );
    }

    pg_sys::Datum::from(dim)
}

/// Binary receive for sparsevec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_recv(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let buf = fc_arg(fcinfo, 0).cast_mut_ptr::<pg_sys::StringInfoData>();
    let typmod = fc_arg(fcinfo, 2).value() as i32;

    let dim = pg_sys::pq_getmsgint(buf, 4) as i32;
    let nnz = pg_sys::pq_getmsgint(buf, 4) as i32;
    let unused = pg_sys::pq_getmsgint(buf, 4) as i32;

    check_dim(dim);
    check_nnz(nnz, dim);
    check_expected_dim(typmod, dim);

    if unused != 0 {
        pgrx::error!("expected unused to be 0, not {}", unused);
    }

    let result = init_sparsevec(dim, nnz);
    let r_indices = sparsevec_indices_mut(result);
    let r_values = sparsevec_values_mut(result);

    for i in 0..nnz as usize {
        *r_indices.add(i) = pg_sys::pq_getmsgint(buf, 4) as i32;
        check_index(r_indices, i, dim);
    }

    for i in 0..nnz as usize {
        *r_values.add(i) = pg_sys::pq_getmsgfloat4(buf);
        check_element(*r_values.add(i));
        if *r_values.add(i) == 0.0 {
            pgrx::error!("binary representation of sparsevec cannot contain zero values");
        }
    }

    pg_sys::Datum::from(result as usize)
}

/// Binary send for sparsevec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_send(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let svec = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let indices = sparsevec_indices(svec);
    let values = sparsevec_values(svec);

    let mut buf: pg_sys::StringInfoData = std::mem::zeroed();
    pg_sys::pq_begintypsend(&mut buf);
    pg_sys::pq_sendint32(&mut buf, (*svec).dim as u32);
    pg_sys::pq_sendint32(&mut buf, (*svec).nnz as u32);
    pg_sys::pq_sendint32(&mut buf, (*svec).unused as u32);

    for i in 0..(*svec).nnz as usize {
        pg_sys::pq_sendint32(&mut buf, *indices.add(i) as u32);
    }

    for i in 0..(*svec).nnz as usize {
        pg_sys::pq_sendfloat4(&mut buf, *values.add(i));
    }

    pg_sys::Datum::from(pg_sys::pq_endtypsend(&mut buf) as usize)
}

/// Cast sparsevec to sparsevec (checks type modifier).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_cast(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let svec = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let typmod = fc_arg(fcinfo, 1).value() as i32;
    check_expected_dim(typmod, (*svec).dim);
    pg_sys::Datum::from(svec as usize)
}

// ---------------------------------------------------------------------------
// Cast functions
// ---------------------------------------------------------------------------

/// Convert vector to sparsevec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_to_sparsevec(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    use crate::types::vector::{vector_data, VectorHeader};

    let vec = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const VectorHeader;
    let typmod = fc_arg(fcinfo, 1).value() as i32;
    let dim = (*vec).dim as i32;
    let vx = vector_data(vec);

    check_dim(dim);
    check_expected_dim(typmod, dim);

    // Count non-zero elements
    let mut nnz = 0i32;
    for i in 0..dim as usize {
        if *vx.add(i) != 0.0 {
            nnz += 1;
        }
    }

    let result = init_sparsevec(dim, nnz);
    let r_indices = sparsevec_indices_mut(result);
    let r_values = sparsevec_values_mut(result);
    let mut j = 0usize;
    for i in 0..dim as usize {
        if *vx.add(i) != 0.0 {
            *r_indices.add(j) = i as i32;
            *r_values.add(j) = *vx.add(i);
            j += 1;
        }
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert halfvec to sparsevec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_to_sparsevec(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    use crate::types::halfvec::{half_to_f32, halfvec_data, HalfVecHeader};

    let vec = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let typmod = fc_arg(fcinfo, 1).value() as i32;
    let dim = (*vec).dim as i32;
    let hx = halfvec_data(vec);

    check_dim(dim);
    check_expected_dim(typmod, dim);

    let mut nnz = 0i32;
    for i in 0..dim as usize {
        if half_to_f32(*hx.add(i)) != 0.0 {
            nnz += 1;
        }
    }

    let result = init_sparsevec(dim, nnz);
    let r_indices = sparsevec_indices_mut(result);
    let r_values = sparsevec_values_mut(result);
    let mut j = 0usize;
    for i in 0..dim as usize {
        let v = half_to_f32(*hx.add(i));
        if v != 0.0 {
            *r_indices.add(j) = i as i32;
            *r_values.add(j) = v;
            j += 1;
        }
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert sparsevec to vector.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_to_vector(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    use crate::types::vector::{init_vector, vector_data_mut};

    let svec = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let typmod = fc_arg(fcinfo, 1).value() as i32;
    let dim = (*svec).dim;
    let s_indices = sparsevec_indices(svec);
    let s_values = sparsevec_values(svec);

    // Vector has a lower max dim
    if dim < 1 {
        pgrx::error!("vector must have at least 1 dimension");
    }
    if dim > crate::types::vector::VECTOR_MAX_DIM {
        pgrx::error!(
            "vector cannot have more than {} dimensions",
            crate::types::vector::VECTOR_MAX_DIM
        );
    }
    if typmod != -1 && typmod != dim {
        pgrx::error!("expected {} dimensions, not {}", typmod, dim);
    }

    let result = init_vector(dim);
    let rx = vector_data_mut(result);
    for i in 0..(*svec).nnz as usize {
        *rx.add(*s_indices.add(i) as usize) = *s_values.add(i);
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert sparsevec to halfvec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_to_halfvec(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    use crate::types::halfvec::{f32_to_half, halfvec_data_mut, init_halfvec};

    let svec = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let typmod = fc_arg(fcinfo, 1).value() as i32;
    let dim = (*svec).dim;
    let s_indices = sparsevec_indices(svec);
    let s_values = sparsevec_values(svec);

    if dim < 1 {
        pgrx::error!("halfvec must have at least 1 dimension");
    }
    if dim > crate::types::halfvec::HALFVEC_MAX_DIM {
        pgrx::error!(
            "halfvec cannot have more than {} dimensions",
            crate::types::halfvec::HALFVEC_MAX_DIM
        );
    }
    if typmod != -1 && typmod != dim {
        pgrx::error!("expected {} dimensions, not {}", typmod, dim);
    }

    let result = init_halfvec(dim);
    let rx = halfvec_data_mut(result);
    for i in 0..(*svec).nnz as usize {
        *rx.add(*s_indices.add(i) as usize) = f32_to_half(*s_values.add(i));
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert array to sparsevec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn array_to_sparsevec(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let array =
        pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *mut pg_sys::ArrayType;
    let typmod = fc_arg(fcinfo, 1).value() as i32;

    let ndim = (*array).ndim;
    if ndim > 1 {
        pgrx::error!("array must be 1-D");
    }

    if pg_sys::array_contains_nulls(array) {
        pgrx::error!("array must not contain nulls");
    }

    let elemtype = (*array).elemtype;
    let mut typlen: i16 = 0;
    let mut typbyval: bool = false;
    let mut typalign: i8 = 0;
    pg_sys::get_typlenbyvalalign(elemtype, &mut typlen, &mut typbyval, &mut typalign);

    let mut elemsp: *mut pg_sys::Datum = ptr::null_mut();
    let mut nelemsp: i32 = 0;
    pg_sys::deconstruct_array(
        array,
        elemtype,
        typlen as i32,
        typbyval,
        typalign,
        &mut elemsp,
        ptr::null_mut(),
        &mut nelemsp,
    );

    check_dim(nelemsp);
    check_expected_dim(typmod, nelemsp);

    // Count non-zero elements and build result
    let mut nnz: i32 = 0;
    let int4oid = pg_sys::INT4OID;
    let float4oid = pg_sys::FLOAT4OID;
    let float8oid = pg_sys::FLOAT8OID;
    let numericoid = pg_sys::NUMERICOID;

    // First pass: count non-zero
    for i in 0..nelemsp as usize {
        let d = *elemsp.add(i);
        let v: f32 = if elemtype == int4oid {
            pg_sys::DatumGetInt32(d) as f32
        } else if elemtype == float4oid {
            pg_sys::DatumGetFloat4(d)
        } else if elemtype == float8oid {
            pg_sys::DatumGetFloat8(d) as f32
        } else if elemtype == numericoid {
            pg_sys::DatumGetFloat4(pg_sys::OidFunctionCall1Coll(
                pg_sys::Oid::from_u32(pg_sys::F_NUMERIC_FLOAT4),
                pg_sys::InvalidOid,
                d,
            ))
        } else {
            pgrx::error!("unsupported array type");
        };
        if v != 0.0 {
            nnz += 1;
        }
    }

    let result = init_sparsevec(nelemsp, nnz);
    let r_indices = sparsevec_indices_mut(result);
    let r_values = sparsevec_values_mut(result);

    // Second pass: fill
    let mut j: usize = 0;
    for i in 0..nelemsp as usize {
        let d = *elemsp.add(i);
        let v: f32 = if elemtype == int4oid {
            pg_sys::DatumGetInt32(d) as f32
        } else if elemtype == float4oid {
            pg_sys::DatumGetFloat4(d)
        } else if elemtype == float8oid {
            pg_sys::DatumGetFloat8(d) as f32
        } else if elemtype == numericoid {
            pg_sys::DatumGetFloat4(pg_sys::OidFunctionCall1Coll(
                pg_sys::Oid::from_u32(pg_sys::F_NUMERIC_FLOAT4),
                pg_sys::InvalidOid,
                d,
            ))
        } else {
            pgrx::error!("unsupported array type");
        };
        if v != 0.0 {
            *r_indices.add(j) = i as i32;
            *r_values.add(j) = v;
            j += 1;
        }
    }

    pg_sys::pfree(elemsp as *mut std::ffi::c_void);

    // Check elements
    for i in 0..nnz as usize {
        check_element(*r_values.add(i));
    }

    pg_sys::Datum::from(result as usize)
}

// ---------------------------------------------------------------------------
// Distance functions
// ---------------------------------------------------------------------------

pg_fn_info!(sparsevec_l2_distance);
pg_fn_info!(sparsevec_l2_squared_distance);
pg_fn_info!(sparsevec_inner_product);
pg_fn_info!(sparsevec_negative_inner_product);
pg_fn_info!(sparsevec_cosine_distance);
pg_fn_info!(sparsevec_l1_distance);
pg_fn_info!(sparsevec_l2_norm);
pg_fn_info!(sparsevec_l2_normalize);
pg_fn_info!(sparsevec_vector_dims);

/// Check that two sparse vectors have the same dimensions.
///
/// # Safety
/// Both pointers must be valid `SparseVecHeader`s.
#[inline]
unsafe fn check_dims(a: *const SparseVecHeader, b: *const SparseVecHeader) {
    if (*a).dim != (*b).dim {
        pgrx::error!(
            "different sparsevec dimensions {} and {}",
            (*a).dim,
            (*b).dim
        );
    }
}

/// L2 squared distance between two sparse vectors.
#[allow(clippy::mut_range_bound)]
unsafe fn sparse_l2_squared_distance(a: *const SparseVecHeader, b: *const SparseVecHeader) -> f32 {
    let ax = sparsevec_values(a);
    let bx = sparsevec_values(b);
    let a_idx = sparsevec_indices(a);
    let b_idx = sparsevec_indices(b);
    let a_nnz = (*a).nnz as usize;
    let b_nnz = (*b).nnz as usize;
    let mut distance: f32 = 0.0;
    let mut bpos: usize = 0;

    for i in 0..a_nnz {
        let ai = *a_idx.add(i);
        let mut bi: i32 = -1;

        for j in bpos..b_nnz {
            bi = *b_idx.add(j);

            if ai == bi {
                let diff = *ax.add(i) - *bx.add(j);
                distance += diff * diff;
            } else if ai > bi {
                distance += *bx.add(j) * *bx.add(j);
            }

            if ai >= bi {
                bpos = j + 1;
            }
            if bi >= ai {
                break;
            }
        }

        if ai != bi {
            distance += *ax.add(i) * *ax.add(i);
        }
    }

    for j in bpos..b_nnz {
        distance += *bx.add(j) * *bx.add(j);
    }

    distance
}

/// Inner product of two sparse vectors.
#[allow(clippy::mut_range_bound)]
unsafe fn sparse_inner_product(a: *const SparseVecHeader, b: *const SparseVecHeader) -> f32 {
    let ax = sparsevec_values(a);
    let bx = sparsevec_values(b);
    let a_idx = sparsevec_indices(a);
    let b_idx = sparsevec_indices(b);
    let a_nnz = (*a).nnz as usize;
    let b_nnz = (*b).nnz as usize;
    let mut distance: f32 = 0.0;
    let mut bpos: usize = 0;

    for i in 0..a_nnz {
        let ai = *a_idx.add(i);

        for j in bpos..b_nnz {
            let bi = *b_idx.add(j);

            if ai == bi {
                distance += *ax.add(i) * *bx.add(j);
            }

            if ai >= bi {
                bpos = j + 1;
            }
            if bi >= ai {
                break;
            }
        }
    }

    distance
}

/// L2 distance.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_l2_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    check_dims(a, b);
    let dist = (sparse_l2_squared_distance(a, b) as f64).sqrt();
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// L2 squared distance.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_l2_squared_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    check_dims(a, b);
    let dist = sparse_l2_squared_distance(a, b) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Inner product.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_inner_product(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    check_dims(a, b);
    let dist = sparse_inner_product(a, b) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Negative inner product.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_negative_inner_product(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    check_dims(a, b);
    let dist = -(sparse_inner_product(a, b) as f64);
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Cosine distance.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_cosine_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    check_dims(a, b);

    let ax = sparsevec_values(a);
    let bx = sparsevec_values(b);
    let a_nnz = (*a).nnz as usize;
    let b_nnz = (*b).nnz as usize;

    let mut similarity = sparse_inner_product(a, b) as f64;

    let mut norma: f32 = 0.0;
    for i in 0..a_nnz {
        norma += *ax.add(i) * *ax.add(i);
    }

    let mut normb: f32 = 0.0;
    for i in 0..b_nnz {
        normb += *bx.add(i) * *bx.add(i);
    }

    similarity /= ((norma as f64) * (normb as f64)).sqrt();
    similarity = similarity.clamp(-1.0, 1.0);

    let dist = 1.0 - similarity;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// L1 (Manhattan) distance.
#[no_mangle]
#[pg_guard]
#[allow(clippy::mut_range_bound)]
pub unsafe extern "C-unwind" fn sparsevec_l1_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    check_dims(a, b);

    let ax = sparsevec_values(a);
    let bx = sparsevec_values(b);
    let a_idx = sparsevec_indices(a);
    let b_idx = sparsevec_indices(b);
    let a_nnz = (*a).nnz as usize;
    let b_nnz = (*b).nnz as usize;
    let mut distance: f32 = 0.0;
    let mut bpos: usize = 0;

    for i in 0..a_nnz {
        let ai = *a_idx.add(i);
        let mut bi: i32 = -1;

        for j in bpos..b_nnz {
            bi = *b_idx.add(j);

            if ai == bi {
                distance += (*ax.add(i) - *bx.add(j)).abs();
            } else if ai > bi {
                distance += (*bx.add(j)).abs();
            }

            if ai >= bi {
                bpos = j + 1;
            }
            if bi >= ai {
                break;
            }
        }

        if ai != bi {
            distance += (*ax.add(i)).abs();
        }
    }

    for j in bpos..b_nnz {
        distance += (*bx.add(j)).abs();
    }

    pg_sys::Datum::from(f64::to_bits(distance as f64))
}

/// Get number of dimensions.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_vector_dims(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from((*a).dim)
}

/// L2 norm.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_l2_norm(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let ax = sparsevec_values(a);
    let nnz = (*a).nnz as usize;
    let mut norm: f64 = 0.0;
    for i in 0..nnz {
        let v = *ax.add(i) as f64;
        norm += v * v;
    }
    pg_sys::Datum::from(f64::to_bits(norm.sqrt()))
}

/// L2-normalize a sparse vector, returning a newly allocated copy.
///
/// # Safety
/// `svec` must point to a valid `SparseVecHeader`.
pub unsafe fn sparsevec_l2_normalize_raw(svec: *const SparseVecHeader) -> *const SparseVecHeader {
    let ax = sparsevec_values(svec);
    let a_indices = sparsevec_indices(svec);
    let nnz = (*svec).nnz as usize;

    let mut norm: f64 = 0.0;
    for i in 0..nnz {
        let v = *ax.add(i) as f64;
        norm += v * v;
    }
    norm = norm.sqrt();

    let result = init_sparsevec((*svec).dim, (*svec).nnz);
    let rx = sparsevec_values_mut(result);
    let r_indices = sparsevec_indices_mut(result);

    if norm > 0.0 {
        let mut zeros = 0i32;

        for i in 0..nnz {
            *r_indices.add(i) = *a_indices.add(i);
            *rx.add(i) = (*ax.add(i) as f64 / norm) as f32;

            if (*rx.add(i)).is_infinite() {
                pgrx::error!("value out of range: overflow");
            }
            if *rx.add(i) == 0.0 {
                zeros += 1;
            }
        }

        // Reallocate in the unlikely event there are zeros
        if zeros > 0 {
            let new_nnz = (*svec).nnz - zeros;
            let new_result = init_sparsevec((*svec).dim, new_nnz);
            let nx = sparsevec_values_mut(new_result);
            let n_indices = sparsevec_indices_mut(new_result);
            let mut j: usize = 0;

            for i in 0..nnz {
                if *rx.add(i) == 0.0 {
                    continue;
                }
                *n_indices.add(j) = *r_indices.add(i);
                *nx.add(j) = *rx.add(i);
                j += 1;
            }

            pg_sys::pfree(result as *mut std::ffi::c_void);
            return new_result as *const SparseVecHeader;
        }
    }

    result as *const SparseVecHeader
}

/// L2-normalize (SQL-callable).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_l2_normalize(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let result = sparsevec_l2_normalize_raw(a);
    pg_sys::Datum::from(result as usize)
}

// ---------------------------------------------------------------------------
// Comparison functions
// ---------------------------------------------------------------------------

pg_fn_info!(sparsevec_lt);
pg_fn_info!(sparsevec_le);
pg_fn_info!(sparsevec_eq);
pg_fn_info!(sparsevec_ne);
pg_fn_info!(sparsevec_ge);
pg_fn_info!(sparsevec_gt);
pg_fn_info!(sparsevec_cmp);

/// Internal comparison matching C's sparsevec_cmp_internal.
unsafe fn sparsevec_cmp_internal(a: *const SparseVecHeader, b: *const SparseVecHeader) -> i32 {
    let ax = sparsevec_values(a);
    let bx = sparsevec_values(b);
    let a_idx = sparsevec_indices(a);
    let b_idx = sparsevec_indices(b);
    let a_nnz = (*a).nnz as usize;
    let b_nnz = (*b).nnz as usize;
    let nnz = a_nnz.min(b_nnz);

    for i in 0..nnz {
        if *a_idx.add(i) < *b_idx.add(i) {
            return if *ax.add(i) < 0.0 { -1 } else { 1 };
        }
        if *a_idx.add(i) > *b_idx.add(i) {
            return if *bx.add(i) < 0.0 { 1 } else { -1 };
        }
        if *ax.add(i) < *bx.add(i) {
            return -1;
        }
        if *ax.add(i) > *bx.add(i) {
            return 1;
        }
    }

    if a_nnz < b_nnz && *b_idx.add(nnz) < (*a).dim {
        return if *bx.add(nnz) < 0.0 { 1 } else { -1 };
    }
    if a_nnz > b_nnz && *a_idx.add(nnz) < (*b).dim {
        return if *ax.add(nnz) < 0.0 { -1 } else { 1 };
    }

    if (*a).dim < (*b).dim {
        -1
    } else if (*a).dim > (*b).dim {
        1
    } else {
        0
    }
}

#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_lt(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from(sparsevec_cmp_internal(a, b) < 0)
}

#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_le(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from(sparsevec_cmp_internal(a, b) <= 0)
}

#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_eq(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from(sparsevec_cmp_internal(a, b) == 0)
}

#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_ne(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from(sparsevec_cmp_internal(a, b) != 0)
}

#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_ge(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from(sparsevec_cmp_internal(a, b) >= 0)
}

#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_gt(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from(sparsevec_cmp_internal(a, b) > 0)
}

#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn sparsevec_cmp(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const SparseVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const SparseVecHeader;
    pg_sys::Datum::from(sparsevec_cmp_internal(a, b))
}

// ---------------------------------------------------------------------------
// HNSW support function
// ---------------------------------------------------------------------------

pg_fn_info!(hnsw_sparsevec_support);

/// HNSW support function for sparsevec type.
///
/// Returns a pointer to a `HnswTypeInfo` struct. The maxDimensions field
/// is set to SPARSEVEC_MAX_DIM (1 billion), which uniquely identifies
/// the sparsevec type for normalize dispatch.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn hnsw_sparsevec_support(
    _fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    use crate::types::halfvec::HnswTypeInfo;

    static TYPE_INFO: HnswTypeInfo = HnswTypeInfo {
        max_dimensions: SPARSEVEC_MAX_DIM,
    };

    pg_sys::Datum::from(&TYPE_INFO as *const HnswTypeInfo as usize)
}

// ---------------------------------------------------------------------------
// SQL registration
// ---------------------------------------------------------------------------

extension_sql!(
    r#"
CREATE TYPE sparsevec;

CREATE FUNCTION sparsevec_in(cstring, oid, integer) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_out(sparsevec) RETURNS cstring
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_recv(internal, oid, integer) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_send(sparsevec) RETURNS bytea
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE sparsevec (
    INPUT     = sparsevec_in,
    OUTPUT    = sparsevec_out,
    TYPMOD_IN = sparsevec_typmod_in,
    RECEIVE   = sparsevec_recv,
    SEND      = sparsevec_send,
    STORAGE   = external
);

-- Cast functions
CREATE FUNCTION sparsevec(sparsevec, integer, boolean) RETURNS sparsevec
    AS 'MODULE_PATHNAME', 'sparsevec_cast' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_to_sparsevec(vector, integer, boolean) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_to_vector(sparsevec, integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_to_sparsevec(halfvec, integer, boolean) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_to_halfvec(sparsevec, integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_sparsevec(integer[], integer, boolean) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_sparsevec(real[], integer, boolean) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_sparsevec(double precision[], integer, boolean) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_sparsevec(numeric[], integer, boolean) RETURNS sparsevec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Casts
CREATE CAST (sparsevec AS sparsevec)
    WITH FUNCTION sparsevec(sparsevec, integer, boolean) AS IMPLICIT;

CREATE CAST (sparsevec AS vector)
    WITH FUNCTION sparsevec_to_vector(sparsevec, integer, boolean) AS ASSIGNMENT;

CREATE CAST (vector AS sparsevec)
    WITH FUNCTION vector_to_sparsevec(vector, integer, boolean) AS IMPLICIT;

CREATE CAST (sparsevec AS halfvec)
    WITH FUNCTION sparsevec_to_halfvec(sparsevec, integer, boolean) AS ASSIGNMENT;

CREATE CAST (halfvec AS sparsevec)
    WITH FUNCTION halfvec_to_sparsevec(halfvec, integer, boolean) AS IMPLICIT;

CREATE CAST (integer[] AS sparsevec)
    WITH FUNCTION array_to_sparsevec(integer[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (real[] AS sparsevec)
    WITH FUNCTION array_to_sparsevec(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (double precision[] AS sparsevec)
    WITH FUNCTION array_to_sparsevec(double precision[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (numeric[] AS sparsevec)
    WITH FUNCTION array_to_sparsevec(numeric[], integer, boolean) AS ASSIGNMENT;
"#,
    name = "sparsevec_type_definition",
    requires = ["vector_type_definition", "halfvec_type_definition"],
);

extension_sql!(
    r#"
-- Distance functions
CREATE FUNCTION l2_distance(sparsevec, sparsevec) RETURNS float8
    AS 'MODULE_PATHNAME', 'sparsevec_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_l2_squared_distance(sparsevec, sparsevec) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION inner_product(sparsevec, sparsevec) RETURNS float8
    AS 'MODULE_PATHNAME', 'sparsevec_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_negative_inner_product(sparsevec, sparsevec) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(sparsevec, sparsevec) RETURNS float8
    AS 'MODULE_PATHNAME', 'sparsevec_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l1_distance(sparsevec, sparsevec) RETURNS float8
    AS 'MODULE_PATHNAME', 'sparsevec_l1_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_dims(sparsevec) RETURNS integer
    AS 'MODULE_PATHNAME', 'sparsevec_vector_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l2_norm(sparsevec) RETURNS float8
    AS 'MODULE_PATHNAME', 'sparsevec_l2_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l2_normalize(sparsevec) RETURNS sparsevec
    AS 'MODULE_PATHNAME', 'sparsevec_l2_normalize'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Comparison functions
CREATE FUNCTION sparsevec_lt(sparsevec, sparsevec) RETURNS bool
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_le(sparsevec, sparsevec) RETURNS bool
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_eq(sparsevec, sparsevec) RETURNS bool
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_ne(sparsevec, sparsevec) RETURNS bool
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_ge(sparsevec, sparsevec) RETURNS bool
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_gt(sparsevec, sparsevec) RETURNS bool
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION sparsevec_cmp(sparsevec, sparsevec) RETURNS int4
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Distance operators
CREATE OPERATOR <-> (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = sparsevec, RIGHTARG = sparsevec,
    PROCEDURE = sparsevec_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = cosine_distance,
    COMMUTATOR = '<=>'
);

CREATE OPERATOR <+> (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = l1_distance,
    COMMUTATOR = '<+>'
);

-- Comparison operators
CREATE OPERATOR < (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = sparsevec_lt,
    COMMUTATOR = > , NEGATOR = >= ,
    RESTRICT = scalarltsel, JOIN = scalarltjoinsel
);

CREATE OPERATOR <= (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = sparsevec_le,
    COMMUTATOR = >= , NEGATOR = > ,
    RESTRICT = scalarlesel, JOIN = scalarlejoinsel
);

CREATE OPERATOR = (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = sparsevec_eq,
    COMMUTATOR = = , NEGATOR = <> ,
    RESTRICT = eqsel, JOIN = eqjoinsel
);

CREATE OPERATOR <> (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = sparsevec_ne,
    COMMUTATOR = <> , NEGATOR = = ,
    RESTRICT = eqsel, JOIN = eqjoinsel
);

CREATE OPERATOR >= (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = sparsevec_ge,
    COMMUTATOR = <= , NEGATOR = < ,
    RESTRICT = scalargesel, JOIN = scalargejoinsel
);

CREATE OPERATOR > (
    LEFTARG = sparsevec, RIGHTARG = sparsevec, PROCEDURE = sparsevec_gt,
    COMMUTATOR = < , NEGATOR = <= ,
    RESTRICT = scalargtsel, JOIN = scalargtjoinsel
);

-- btree opclass
CREATE OPERATOR CLASS sparsevec_ops
    DEFAULT FOR TYPE sparsevec USING btree AS
    OPERATOR 1 < ,
    OPERATOR 2 <= ,
    OPERATOR 3 = ,
    OPERATOR 4 >= ,
    OPERATOR 5 > ,
    FUNCTION 1 sparsevec_cmp(sparsevec, sparsevec);

-- HNSW support function
CREATE FUNCTION hnsw_sparsevec_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
"#,
    name = "sparsevec_functions",
    requires = ["sparsevec_type_definition"],
);

extension_sql!(
    r#"
-- HNSW operator classes for sparsevec
CREATE OPERATOR CLASS sparsevec_l2_ops
    FOR TYPE sparsevec USING hnsw AS
    OPERATOR 1 <-> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_l2_squared_distance(sparsevec, sparsevec),
    FUNCTION 3 hnsw_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_ip_ops
    FOR TYPE sparsevec USING hnsw AS
    OPERATOR 1 <#> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_negative_inner_product(sparsevec, sparsevec),
    FUNCTION 3 hnsw_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_cosine_ops
    FOR TYPE sparsevec USING hnsw AS
    OPERATOR 1 <=> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 sparsevec_negative_inner_product(sparsevec, sparsevec),
    FUNCTION 2 l2_norm(sparsevec),
    FUNCTION 3 hnsw_sparsevec_support(internal);

CREATE OPERATOR CLASS sparsevec_l1_ops
    FOR TYPE sparsevec USING hnsw AS
    OPERATOR 1 <+> (sparsevec, sparsevec) FOR ORDER BY float_ops,
    FUNCTION 1 l1_distance(sparsevec, sparsevec),
    FUNCTION 3 hnsw_sparsevec_support(internal);
"#,
    name = "sparsevec_hnsw_opclasses",
    requires = ["sparsevec_functions", hnsw_handler],
);
