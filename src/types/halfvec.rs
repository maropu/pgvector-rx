//! PostgreSQL `halfvec` type implementation.
//!
//! Provides the `halfvec` data type for storing fixed-dimension half-precision
//! (float16) vectors, matching the original pgvector C implementation's on-disk
//! format. Values are stored as 2-byte IEEE 754 half-precision floats internally,
//! with all distance computations performed in f32.

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

/// Maximum number of dimensions for the `halfvec` type.
pub const HALFVEC_MAX_DIM: i32 = 16000;

/// Size of the fixed halfvec header: vl_len_(4) + dim(2) + unused(2) = 8 bytes.
const HALFVEC_HEADER_SIZE: usize = 8;

/// Size of a `HalfVector` in bytes given `dim` dimensions.
#[inline]
pub const fn halfvec_size(dim: i32) -> usize {
    HALFVEC_HEADER_SIZE + (dim as usize) * std::mem::size_of::<u16>()
}

/// On-disk halfvec header. The u16 array `x[dim]` follows immediately after.
/// Each u16 is an IEEE 754 half-precision float.
#[repr(C)]
pub struct HalfVecHeader {
    pub vl_len_: i32,
    pub dim: i16,
    pub unused: i16,
}

// ---------------------------------------------------------------------------
// Half-precision conversion
// ---------------------------------------------------------------------------

/// Convert a half-precision float (stored as u16) to f32.
#[inline]
pub fn half_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;

    if exp == 0 {
        if mant == 0 {
            // Zero (positive or negative)
            f32::from_bits(sign << 31)
        } else {
            // Denormal: normalize to f32
            let mut m = mant;
            let mut e: i32 = -14; // half denorm exponent bias
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            let f32_exp = (e + 127) as u32;
            f32::from_bits((sign << 31) | (f32_exp << 23) | (m << 13))
        }
    } else if exp == 31 {
        // Inf or NaN
        if mant == 0 {
            f32::from_bits((sign << 31) | (0xff << 23))
        } else {
            f32::from_bits((sign << 31) | (0xff << 23) | (mant << 13))
        }
    } else {
        // Normal
        let f32_exp = (exp as i32 - 15 + 127) as u32;
        f32::from_bits((sign << 31) | (f32_exp << 23) | (mant << 13))
    }
}

/// Convert an f32 to half-precision float (stored as u16).
/// This uses round-to-nearest-even, matching the C implementation.
#[inline]
pub fn f32_to_half(f: f32) -> u16 {
    let bits = f.to_bits();
    let sign = ((bits >> 31) & 1) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x7fffff;

    if exp == 0xff {
        // Inf or NaN
        if mant == 0 {
            (sign << 15) | (0x1f << 10)
        } else {
            // NaN: preserve some mantissa bits
            (sign << 15) | (0x1f << 10) | ((mant >> 13) as u16).max(1)
        }
    } else if exp > 142 {
        // Overflow to infinity (exp - 127 + 15 > 30)
        (sign << 15) | (0x1f << 10)
    } else if exp < 103 {
        // Underflow to zero (exp - 127 + 15 < -10)
        sign << 15
    } else if exp < 113 {
        // Denormal
        let shift = 113 - exp;
        let m = (mant | 0x800000) >> (shift + 13);
        // Round
        let round_bit = (mant | 0x800000) >> (shift + 12) & 1;
        let sticky = if ((mant | 0x800000) & ((1 << (shift + 12)) - 1)) != 0 {
            1u16
        } else {
            0
        };
        let result = (sign << 15) | (m as u16);
        if round_bit != 0 && (sticky != 0 || (m & 1) != 0) {
            result + 1
        } else {
            result
        }
    } else {
        // Normal
        let half_exp = ((exp - 127 + 15) as u16) & 0x1f;
        let half_mant = (mant >> 13) as u16;
        // Round-to-nearest-even
        let round_bit = (mant >> 12) & 1;
        let sticky = mant & 0xfff;
        let result = (sign << 15) | (half_exp << 10) | half_mant;
        if round_bit != 0 && (sticky != 0 || (half_mant & 1) != 0) {
            result + 1
        } else {
            result
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

#[inline]
fn halfvec_isspace(ch: u8) -> bool {
    matches!(ch, b' ' | b'\t' | b'\n' | b'\r' | 0x0B | 0x0C)
}

#[inline]
fn check_dim(dim: i32) {
    if dim < 1 {
        pgrx::error!("halfvec must have at least 1 dimension");
    }
    if dim > HALFVEC_MAX_DIM {
        pgrx::error!(
            "halfvec cannot have more than {} dimensions",
            HALFVEC_MAX_DIM
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
fn check_element(val: f32) {
    if val.is_nan() {
        pgrx::error!("NaN not allowed in halfvec");
    }
    if val.is_infinite() {
        pgrx::error!("infinite value not allowed in halfvec");
    }
}

/// Allocate and initialize a zero-filled HalfVector with `dim` dimensions.
///
/// # Safety
/// Allocates via `palloc0` in the current memory context.
pub unsafe fn init_halfvec(dim: i32) -> *mut HalfVecHeader {
    let size = halfvec_size(dim);
    // SAFETY: palloc0 returns valid, zeroed memory.
    let result = pg_sys::palloc0(size) as *mut HalfVecHeader;
    // SET_VARSIZE: store (size << 2) in the first 4 bytes for 4-byte header
    let vl_ptr = result as *mut u32;
    *vl_ptr = (size as u32) << 2;
    (*result).dim = dim as i16;
    (*result).unused = 0;
    result
}

/// L2-normalize a halfvec, returning a newly allocated normalized copy.
///
/// # Safety
/// `vec` must point to a valid `HalfVecHeader`. Allocates via `palloc0`.
pub unsafe fn halfvec_l2_normalize_raw(vec: *const HalfVecHeader) -> *const HalfVecHeader {
    let dim = (*vec).dim as usize;
    let ax = halfvec_data(vec);

    let mut norm: f64 = 0.0;
    for i in 0..dim {
        let v = half_to_f32(*ax.add(i)) as f64;
        norm += v * v;
    }
    norm = norm.sqrt();

    let result = init_halfvec(dim as i32);
    let rx = halfvec_data_mut(result);
    if norm > 0.0 {
        for i in 0..dim {
            let normalized = half_to_f32(*ax.add(i)) as f64 / norm;
            *rx.add(i) = f32_to_half(normalized as f32);
        }
    }

    // Check for overflow (inf indicates out-of-range)
    for i in 0..dim {
        let v = half_to_f32(*rx.add(i));
        if v.is_infinite() {
            pgrx::error!("value out of range: overflow");
        }
    }

    result as *const HalfVecHeader
}

/// Returns a mutable pointer to the half data of a halfvec.
///
/// # Safety
/// `vec` must point to a valid, properly-sized `HalfVecHeader`.
#[inline]
pub unsafe fn halfvec_data_mut(vec: *mut HalfVecHeader) -> *mut u16 {
    (vec as *mut u8).add(HALFVEC_HEADER_SIZE) as *mut u16
}

/// Returns a const pointer to the half data of a halfvec.
///
/// # Safety
/// `vec` must point to a valid, properly-sized `HalfVecHeader`.
#[inline]
pub unsafe fn halfvec_data(vec: *const HalfVecHeader) -> *const u16 {
    (vec as *const u8).add(HALFVEC_HEADER_SIZE) as *const u16
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

pg_fn_info!(halfvec_in);
pg_fn_info!(halfvec_out);
pg_fn_info!(halfvec_typmod_in);
pg_fn_info!(halfvec_recv);
pg_fn_info!(halfvec_send);
pg_fn_info!(halfvec_cast);
pg_fn_info!(array_to_halfvec);
pg_fn_info!(halfvec_to_float4);
pg_fn_info!(halfvec_to_vector);
pg_fn_info!(vector_to_halfvec);

/// Parse text `[1,2,3]` into a halfvec. Matches C's `halfvec_in`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_in(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let input = pg_sys::DatumGetCString(fc_arg(fcinfo, 0));
    let typmod = fc_arg(fcinfo, 2).value() as i32;

    let lit = std::ffi::CStr::from_ptr(input).to_bytes();
    let mut pos = 0usize;

    while pos < lit.len() && halfvec_isspace(lit[pos]) {
        pos += 1;
    }

    let lit_str = || std::str::from_utf8(lit).unwrap_or("???");

    if pos >= lit.len() || lit[pos] != b'[' {
        pgrx::error!("invalid input syntax for type halfvec: \"{}\"", lit_str());
    }
    pos += 1;

    while pos < lit.len() && halfvec_isspace(lit[pos]) {
        pos += 1;
    }

    if pos < lit.len() && lit[pos] == b']' {
        pgrx::error!("halfvec must have at least 1 dimension");
    }

    let mut values: Vec<f32> = Vec::new();

    loop {
        if values.len() as i32 >= HALFVEC_MAX_DIM {
            pgrx::error!(
                "halfvec cannot have more than {} dimensions",
                HALFVEC_MAX_DIM
            );
        }

        while pos < lit.len() && halfvec_isspace(lit[pos]) {
            pos += 1;
        }

        if pos >= lit.len() {
            pgrx::error!("invalid input syntax for type halfvec: \"{}\"", lit_str());
        }

        let start = pos;
        while pos < lit.len() && lit[pos] != b',' && lit[pos] != b']' && !halfvec_isspace(lit[pos])
        {
            pos += 1;
        }

        let num_str = std::str::from_utf8(&lit[start..pos]).unwrap_or("");
        let val: f32 = match num_str.parse() {
            Ok(v) => v,
            Err(_) => {
                pgrx::error!("invalid input syntax for type halfvec: \"{}\"", lit_str());
            }
        };

        check_element(val);
        values.push(val);

        while pos < lit.len() && halfvec_isspace(lit[pos]) {
            pos += 1;
        }

        if pos < lit.len() && lit[pos] == b',' {
            pos += 1;
        } else if pos < lit.len() && lit[pos] == b']' {
            pos += 1;
            break;
        } else {
            pgrx::error!("invalid input syntax for type halfvec: \"{}\"", lit_str());
        }
    }

    while pos < lit.len() && halfvec_isspace(lit[pos]) {
        pos += 1;
    }
    if pos < lit.len() {
        pgrx::error!("invalid input syntax for type halfvec: \"{}\"", lit_str());
    }

    let dim = values.len() as i32;
    check_dim(dim);
    check_expected_dim(typmod, dim);

    let result = init_halfvec(dim);
    let data = halfvec_data_mut(result);
    for (i, &v) in values.iter().enumerate() {
        let h = f32_to_half(v);
        // Check round-trip for overflow
        let rt = half_to_f32(h);
        if rt.is_infinite() {
            pgrx::error!("\"{}\" is out of range for type halfvec", v);
        }
        *data.add(i) = h;
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert halfvec to text `[1,2,3]`. Matches C's `halfvec_out`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_out(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const HalfVecHeader;
    let dim = (*vec).dim as usize;
    let data = halfvec_data(vec);

    let mut buf = String::with_capacity(dim * 10 + 3);
    buf.push('[');
    for i in 0..dim {
        if i > 0 {
            buf.push(',');
        }
        let val = half_to_f32(*data.add(i));
        let mut ryu_buf = ryu::Buffer::new();
        let formatted = ryu_buf.format(val);
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

/// Parse type modifier for `halfvec(N)`.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_typmod_in(
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
        pgrx::error!("dimensions for type halfvec must be at least 1");
    }
    if dim > HALFVEC_MAX_DIM {
        pgrx::error!(
            "dimensions for type halfvec cannot exceed {}",
            HALFVEC_MAX_DIM
        );
    }

    pg_sys::Datum::from(dim as usize)
}

/// Receive halfvec from binary format.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_recv(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let stringinfo = fc_arg(fcinfo, 0).cast_mut_ptr::<pg_sys::StringInfoData>();
    let typmod = fc_arg(fcinfo, 2).value() as i32;

    let dim = pg_sys::pq_getmsgint(stringinfo, 2) as i16;
    let unused = pg_sys::pq_getmsgint(stringinfo, 2) as i16;

    check_dim(dim as i32);
    check_expected_dim(typmod, dim as i32);

    if unused != 0 {
        pgrx::error!("expected unused to be 0, not {}", unused);
    }

    let result = init_halfvec(dim as i32);
    let data = halfvec_data_mut(result);
    for i in 0..dim as usize {
        *data.add(i) = pg_sys::pq_getmsgint(stringinfo, 2) as u16;
    }

    pg_sys::Datum::from(result as usize)
}

/// Send halfvec in binary format.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_send(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const HalfVecHeader;
    let dim = (*vec).dim;
    let unused = (*vec).unused;
    let data = halfvec_data(vec);

    let mut buf: pg_sys::StringInfoData = std::mem::zeroed();
    pg_sys::pq_begintypsend(&mut buf);

    let dim_be = (dim as u16).to_be_bytes();
    pg_sys::pq_sendbytes(&mut buf, dim_be.as_ptr() as *const std::ffi::c_void, 2);

    let unused_be = (unused as u16).to_be_bytes();
    pg_sys::pq_sendbytes(&mut buf, unused_be.as_ptr() as *const std::ffi::c_void, 2);

    for i in 0..dim as usize {
        let val_be = (*data.add(i)).to_be_bytes();
        pg_sys::pq_sendbytes(&mut buf, val_be.as_ptr() as *const std::ffi::c_void, 2);
    }

    let result = pg_sys::pq_endtypsend(&mut buf);
    pg_sys::Datum::from(result as usize)
}

/// Cast halfvec to halfvec (typmod enforcement).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_cast(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let typmod = fc_arg(fcinfo, 1).value() as i32;

    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const HalfVecHeader;
    check_expected_dim(typmod, (*vec).dim as i32);

    raw
}

/// Convert a PostgreSQL array to a halfvec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn array_to_halfvec(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
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

    let result = init_halfvec(dim);
    let rx = halfvec_data_mut(result);

    // First collect as f32, then convert to half
    for i in 0..dim as usize {
        let val: f32 = if elem_type == pg_sys::INT4OID {
            pg_sys::DatumGetInt32(*elems.add(i)) as f32
        } else if elem_type == pg_sys::FLOAT8OID {
            pg_sys::DatumGetFloat8(*elems.add(i)) as f32
        } else if elem_type == pg_sys::FLOAT4OID {
            pg_sys::DatumGetFloat4(*elems.add(i))
        } else if elem_type == pg_sys::NUMERICOID {
            pg_sys::DatumGetFloat4(pg_sys::OidFunctionCall1Coll(
                pg_sys::Oid::from_u32(pg_sys::F_NUMERIC_FLOAT4),
                pg_sys::InvalidOid,
                *elems.add(i),
            ))
        } else {
            pgrx::error!("unsupported array type");
        };

        check_element(val);
        let h = f32_to_half(val);
        let rt = half_to_f32(h);
        if rt.is_infinite() && !val.is_infinite() {
            pgrx::error!("\"{}\" is out of range for type halfvec", val);
        }
        *rx.add(i) = h;
    }

    pg_sys::pfree(elems as *mut std::ffi::c_void);

    pg_sys::Datum::from(result as usize)
}

/// Convert a halfvec to a float4[] array.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_to_float4(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let vec = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const HalfVecHeader;
    let dim = (*vec).dim as usize;
    let data = halfvec_data(vec);

    let datums = pg_sys::palloc(dim * std::mem::size_of::<pg_sys::Datum>()) as *mut pg_sys::Datum;
    for i in 0..dim {
        *datums.add(i) = pg_sys::Float4GetDatum(half_to_f32(*data.add(i)));
    }

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

/// Convert a halfvec to a vector.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_to_vector(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let typmod = fc_arg(fcinfo, 1).value() as i32;
    let hv = pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const HalfVecHeader;
    let dim = (*hv).dim as i32;

    if typmod != -1 && typmod != dim {
        pgrx::error!("expected {} dimensions, not {}", typmod, dim);
    }

    use crate::types::vector::{init_vector, vector_data_mut};

    let result = init_vector(dim);
    let rx = vector_data_mut(result);
    let hx = halfvec_data(hv);
    for i in 0..dim as usize {
        *rx.add(i) = half_to_f32(*hx.add(i));
    }

    pg_sys::Datum::from(result as usize)
}

/// Convert a vector to a halfvec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn vector_to_halfvec(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let raw = fc_arg(fcinfo, 0);
    let typmod = fc_arg(fcinfo, 1).value() as i32;
    let vec =
        pg_sys::pg_detoast_datum(raw.cast_mut_ptr()) as *const crate::types::vector::VectorHeader;
    let dim = (*vec).dim as i32;

    if typmod != -1 && typmod != dim {
        pgrx::error!("expected {} dimensions, not {}", typmod, dim);
    }

    let vx = crate::types::vector::vector_data(vec);
    let result = init_halfvec(dim);
    let rx = halfvec_data_mut(result);
    for i in 0..dim as usize {
        let val = *vx.add(i);
        let h = f32_to_half(val);
        let rt = half_to_f32(h);
        if rt.is_infinite() && !val.is_infinite() {
            pgrx::error!("\"{}\" is out of range for type halfvec", val);
        }
        *rx.add(i) = h;
    }

    pg_sys::Datum::from(result as usize)
}

// ---------------------------------------------------------------------------
// Distance / utility helpers
// ---------------------------------------------------------------------------

/// Ensure two halfvecs have the same dimensions.
#[inline]
unsafe fn check_dims(a: *const HalfVecHeader, b: *const HalfVecHeader) {
    if (*a).dim != (*b).dim {
        pgrx::error!("different halfvec dimensions {} and {}", (*a).dim, (*b).dim);
    }
}

/// L2 squared distance for halfvecs.
#[inline]
fn compute_l2_squared(dim: usize, ax: *const u16, bx: *const u16) -> f32 {
    let mut distance: f32 = 0.0;
    for i in 0..dim {
        // SAFETY: caller guarantees valid pointers and dim.
        let diff = unsafe { half_to_f32(*ax.add(i)) - half_to_f32(*bx.add(i)) };
        distance += diff * diff;
    }
    distance
}

/// Inner product for halfvecs.
#[inline]
fn compute_inner_product(dim: usize, ax: *const u16, bx: *const u16) -> f32 {
    let mut distance: f32 = 0.0;
    for i in 0..dim {
        distance += unsafe { half_to_f32(*ax.add(i)) * half_to_f32(*bx.add(i)) };
    }
    distance
}

/// Cosine similarity for halfvecs.
#[inline]
fn compute_cosine_similarity(dim: usize, ax: *const u16, bx: *const u16) -> f64 {
    let mut similarity: f32 = 0.0;
    let mut norma: f32 = 0.0;
    let mut normb: f32 = 0.0;
    for i in 0..dim {
        unsafe {
            let ai = half_to_f32(*ax.add(i));
            let bi = half_to_f32(*bx.add(i));
            similarity += ai * bi;
            norma += ai * ai;
            normb += bi * bi;
        }
    }
    (similarity as f64) / ((norma as f64) * (normb as f64)).sqrt()
}

/// L1 distance for halfvecs.
#[inline]
fn compute_l1_distance(dim: usize, ax: *const u16, bx: *const u16) -> f32 {
    let mut distance: f32 = 0.0;
    for i in 0..dim {
        distance += unsafe { (half_to_f32(*ax.add(i)) - half_to_f32(*bx.add(i))).abs() };
    }
    distance
}

// ---------------------------------------------------------------------------
// Distance / utility PostgreSQL functions
// ---------------------------------------------------------------------------

pg_fn_info!(halfvec_l2_distance);
pg_fn_info!(halfvec_l2_squared_distance);
pg_fn_info!(halfvec_negative_inner_product);
pg_fn_info!(halfvec_cosine_distance);
pg_fn_info!(halfvec_l1_distance);
pg_fn_info!(halfvec_vector_dims);
pg_fn_info!(halfvec_l2_norm);
pg_fn_info!(halfvec_l2_normalize);

/// L2 (Euclidean) distance between two halfvecs.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_l2_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const HalfVecHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = (compute_l2_squared(dim, halfvec_data(a), halfvec_data(b)) as f64).sqrt();
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// L2 squared distance (used by HNSW operator class).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_l2_squared_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const HalfVecHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = compute_l2_squared(dim, halfvec_data(a), halfvec_data(b)) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Negative inner product (used by HNSW operator class for IP distance).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_negative_inner_product(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const HalfVecHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = -(compute_inner_product(dim, halfvec_data(a), halfvec_data(b)) as f64);
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Cosine distance: 1 - cosine_similarity(a, b).
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_cosine_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const HalfVecHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let similarity = compute_cosine_similarity(dim, halfvec_data(a), halfvec_data(b));
    let dist = 1.0 - similarity.clamp(-1.0, 1.0);
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// L1 (Manhattan) distance between two halfvecs.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_l1_distance(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let b = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 1).cast_mut_ptr()) as *const HalfVecHeader;
    check_dims(a, b);
    let dim = (*a).dim as usize;
    let dist = compute_l1_distance(dim, halfvec_data(a), halfvec_data(b)) as f64;
    pg_sys::Datum::from(f64::to_bits(dist))
}

/// Get the number of dimensions of a halfvec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_vector_dims(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    pg_sys::Datum::from((*a).dim as i32)
}

/// Get the L2 norm of a halfvec.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_l2_norm(fcinfo: pg_sys::FunctionCallInfo) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let ax = halfvec_data(a);
    let dim = (*a).dim as usize;
    let mut norm: f64 = 0.0;
    for i in 0..dim {
        let v = half_to_f32(*ax.add(i)) as f64;
        norm += v * v;
    }
    let result = norm.sqrt();
    pg_sys::Datum::from(f64::to_bits(result))
}

/// L2-normalize a halfvec to unit length.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn halfvec_l2_normalize(
    fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    let a = pg_sys::pg_detoast_datum(fc_arg(fcinfo, 0).cast_mut_ptr()) as *const HalfVecHeader;
    let result = halfvec_l2_normalize_raw(a);
    pg_sys::Datum::from(result as usize)
}

// ---------------------------------------------------------------------------
// HNSW support function
// ---------------------------------------------------------------------------

pg_fn_info!(hnsw_halfvec_support);

/// HNSW support function for halfvec type.
///
/// Returns a pointer to a `HnswTypeInfo` struct that provides the normalize
/// function for the halfvec type. This is used by the HNSW index build, insert,
/// and scan code to normalize vectors for cosine distance.
#[no_mangle]
#[pg_guard]
pub unsafe extern "C-unwind" fn hnsw_halfvec_support(
    _fcinfo: pg_sys::FunctionCallInfo,
) -> pg_sys::Datum {
    // Return a pointer to static type info.
    // The normalize field stores a raw function pointer compatible with
    // DirectFunctionCall1Coll. We use the SQL-registered halfvec_l2_normalize.
    static TYPE_INFO: HnswTypeInfo = HnswTypeInfo {
        max_dimensions: crate::hnsw_constants::HNSW_MAX_DIM * 2,
    };

    pg_sys::Datum::from(&TYPE_INFO as *const HnswTypeInfo as usize)
}

/// Type info struct returned by hnsw support functions.
/// Simplified version - we only need maxDimensions since normalize
/// is handled via the norm_fmgr mechanism.
#[repr(C)]
pub struct HnswTypeInfo {
    pub max_dimensions: i32,
}

// ---------------------------------------------------------------------------
// SQL registration
// ---------------------------------------------------------------------------

extension_sql!(
    r#"
CREATE TYPE halfvec;

CREATE FUNCTION halfvec_in(cstring, oid, integer) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_out(halfvec) RETURNS cstring
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_typmod_in(cstring[]) RETURNS integer
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_recv(internal, oid, integer) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_send(halfvec) RETURNS bytea
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE TYPE halfvec (
    INPUT     = halfvec_in,
    OUTPUT    = halfvec_out,
    TYPMOD_IN = halfvec_typmod_in,
    RECEIVE   = halfvec_recv,
    SEND      = halfvec_send,
    STORAGE   = external
);

-- Cast functions
CREATE FUNCTION halfvec(halfvec, integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'halfvec_cast' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_halfvec(integer[], integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_halfvec(real[], integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_halfvec(double precision[], integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION array_to_halfvec(numeric[], integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_to_float4(halfvec, integer, boolean) RETURNS real[]
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_to_vector(halfvec, integer, boolean) RETURNS vector
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_to_halfvec(vector, integer, boolean) RETURNS halfvec
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Casts
CREATE CAST (halfvec AS halfvec)
    WITH FUNCTION halfvec(halfvec, integer, boolean) AS IMPLICIT;

CREATE CAST (halfvec AS vector)
    WITH FUNCTION halfvec_to_vector(halfvec, integer, boolean) AS ASSIGNMENT;

CREATE CAST (vector AS halfvec)
    WITH FUNCTION vector_to_halfvec(vector, integer, boolean) AS IMPLICIT;

CREATE CAST (halfvec AS real[])
    WITH FUNCTION halfvec_to_float4(halfvec, integer, boolean) AS ASSIGNMENT;

CREATE CAST (integer[] AS halfvec)
    WITH FUNCTION array_to_halfvec(integer[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (real[] AS halfvec)
    WITH FUNCTION array_to_halfvec(real[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (double precision[] AS halfvec)
    WITH FUNCTION array_to_halfvec(double precision[], integer, boolean) AS ASSIGNMENT;

CREATE CAST (numeric[] AS halfvec)
    WITH FUNCTION array_to_halfvec(numeric[], integer, boolean) AS ASSIGNMENT;
"#,
    name = "halfvec_type_definition",
    requires = ["vector_type_definition"],
);

extension_sql!(
    r#"
-- Distance functions
CREATE FUNCTION l2_distance(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'halfvec_l2_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_l2_squared_distance(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION inner_product(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'halfvec_negative_inner_product'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION halfvec_negative_inner_product(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION cosine_distance(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'halfvec_cosine_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l1_distance(halfvec, halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'halfvec_l1_distance'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION vector_dims(halfvec) RETURNS integer
    AS 'MODULE_PATHNAME', 'halfvec_vector_dims'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l2_norm(halfvec) RETURNS float8
    AS 'MODULE_PATHNAME', 'halfvec_l2_norm'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

CREATE FUNCTION l2_normalize(halfvec) RETURNS halfvec
    AS 'MODULE_PATHNAME', 'halfvec_l2_normalize'
    LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;

-- Distance operators
CREATE OPERATOR <-> (
    LEFTARG = halfvec, RIGHTARG = halfvec, PROCEDURE = l2_distance,
    COMMUTATOR = '<->'
);

CREATE OPERATOR <#> (
    LEFTARG = halfvec, RIGHTARG = halfvec,
    PROCEDURE = halfvec_negative_inner_product,
    COMMUTATOR = '<#>'
);

CREATE OPERATOR <=> (
    LEFTARG = halfvec, RIGHTARG = halfvec, PROCEDURE = cosine_distance,
    COMMUTATOR = '<=>'
);

CREATE OPERATOR <+> (
    LEFTARG = halfvec, RIGHTARG = halfvec, PROCEDURE = l1_distance,
    COMMUTATOR = '<+>'
);

-- HNSW support function
CREATE FUNCTION hnsw_halfvec_support(internal) RETURNS internal
    AS 'MODULE_PATHNAME' LANGUAGE C IMMUTABLE STRICT PARALLEL SAFE;
"#,
    name = "halfvec_distance_functions",
    requires = ["halfvec_type_definition"],
);

extension_sql!(
    r#"
-- HNSW operator classes for halfvec
CREATE OPERATOR CLASS halfvec_l2_ops
    FOR TYPE halfvec USING hnsw AS
    OPERATOR 1 <-> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_l2_squared_distance(halfvec, halfvec),
    FUNCTION 3 hnsw_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_ip_ops
    FOR TYPE halfvec USING hnsw AS
    OPERATOR 1 <#> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_negative_inner_product(halfvec, halfvec),
    FUNCTION 3 hnsw_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_cosine_ops
    FOR TYPE halfvec USING hnsw AS
    OPERATOR 1 <=> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 halfvec_negative_inner_product(halfvec, halfvec),
    FUNCTION 2 l2_norm(halfvec),
    FUNCTION 3 hnsw_halfvec_support(internal);

CREATE OPERATOR CLASS halfvec_l1_ops
    FOR TYPE halfvec USING hnsw AS
    OPERATOR 1 <+> (halfvec, halfvec) FOR ORDER BY float_ops,
    FUNCTION 1 l1_distance(halfvec, halfvec),
    FUNCTION 3 hnsw_halfvec_support(internal);
"#,
    name = "halfvec_hnsw_opclasses",
    requires = ["halfvec_distance_functions", hnsw_handler],
);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
#[cfg(any(test, feature = "pg_test"))]
#[pg_schema]
mod tests {
    use super::*;

    #[test]
    fn test_half_roundtrip_zero() {
        assert_eq!(half_to_f32(f32_to_half(0.0)), 0.0);
    }

    #[test]
    fn test_half_roundtrip_one() {
        assert_eq!(half_to_f32(f32_to_half(1.0)), 1.0);
    }

    #[test]
    fn test_half_roundtrip_neg() {
        assert_eq!(half_to_f32(f32_to_half(-1.0)), -1.0);
    }

    #[test]
    fn test_half_roundtrip_pi() {
        let pi_half = half_to_f32(f32_to_half(std::f32::consts::PI));
        assert!((pi_half - std::f32::consts::PI).abs() < 0.002);
    }

    #[test]
    fn test_half_inf() {
        assert!(half_to_f32(f32_to_half(f32::INFINITY)).is_infinite());
        assert!(half_to_f32(f32_to_half(f32::NEG_INFINITY)).is_infinite());
    }

    #[test]
    fn test_half_nan() {
        assert!(half_to_f32(f32_to_half(f32::NAN)).is_nan());
    }

    #[pg_test]
    fn test_halfvec_type_exists() {
        let result =
            Spi::get_one::<String>("SELECT typname::text FROM pg_type WHERE typname = 'halfvec'")
                .expect("SPI failed")
                .expect("halfvec type not found");
        assert_eq!(result, "halfvec");
    }

    #[pg_test]
    fn test_halfvec_in_out_basic() {
        let result = Spi::get_one::<String>("SELECT '[1,2,3]'::halfvec::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    fn test_halfvec_with_typmod() {
        let result = Spi::get_one::<String>("SELECT '[1,2,3]'::halfvec(3)::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    #[should_panic(expected = "expected 3 dimensions, not 2")]
    fn test_halfvec_typmod_mismatch() {
        Spi::get_one::<String>("SELECT '[1,2]'::halfvec(3)::text").ok();
    }

    #[pg_test]
    fn test_halfvec_l2_distance() {
        let result = Spi::get_one::<f64>("SELECT l2_distance('[0,0]'::halfvec, '[3,4]'::halfvec)")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 5.0).abs() < 0.01);
    }

    #[pg_test]
    fn test_halfvec_l2_operator() {
        let result = Spi::get_one::<f64>("SELECT '[0,0]'::halfvec <-> '[3,4]'::halfvec")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 5.0).abs() < 0.01);
    }

    #[pg_test]
    fn test_halfvec_ip_operator() {
        let result = Spi::get_one::<f64>("SELECT '[1,2,3]'::halfvec <#> '[4,5,6]'::halfvec")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - (-32.0)).abs() < 0.1);
    }

    #[pg_test]
    fn test_halfvec_cosine_operator() {
        let result = Spi::get_one::<f64>("SELECT '[1,0]'::halfvec <=> '[0,1]'::halfvec")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 1.0).abs() < 0.01);
    }

    #[pg_test]
    fn test_halfvec_l1_operator() {
        let result = Spi::get_one::<f64>("SELECT '[1,2,3]'::halfvec <+> '[4,6,8]'::halfvec")
            .expect("SPI failed")
            .expect("NULL result");
        assert!((result - 12.0).abs() < 0.1);
    }

    #[pg_test]
    fn test_halfvec_to_vector_cast() {
        let result = Spi::get_one::<String>("SELECT ('[1,2,3]'::halfvec)::vector::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    fn test_vector_to_halfvec_cast() {
        let result = Spi::get_one::<String>("SELECT ('[1,2,3]'::vector)::halfvec::text")
            .expect("SPI failed")
            .expect("NULL result");
        assert_eq!(result, "[1,2,3]");
    }

    #[pg_test]
    fn test_array_to_halfvec() {
        let result =
            Spi::get_one::<i32>("SELECT vector_dims((ARRAY[1.0, 2.0, 3.0]::real[])::halfvec)")
                .expect("SPI failed")
                .expect("NULL result");
        assert_eq!(result, 3);
    }

    #[pg_test]
    fn test_halfvec_hnsw_l2_index() {
        Spi::run("CREATE TABLE test_hv (id serial, v halfvec(3))").unwrap();
        Spi::run(
            "INSERT INTO test_hv (v) VALUES \
             ('[1,2,3]'), ('[4,5,6]'), ('[7,8,9]'), ('[0,0,0.5]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX ON test_hv USING hnsw (v halfvec_l2_ops)").unwrap();

        let result = Spi::get_one::<i32>(
            "SET enable_seqscan = off; \
             SELECT id FROM test_hv ORDER BY v <-> '[1,2,3]'::halfvec LIMIT 1",
        )
        .expect("SPI failed")
        .expect("NULL result");
        assert_eq!(result, 1);
    }

    #[pg_test]
    fn test_halfvec_hnsw_cosine_index() {
        Spi::run("CREATE TABLE test_hv_cos (id serial, v halfvec(3))").unwrap();
        Spi::run(
            "INSERT INTO test_hv_cos (v) VALUES \
             ('[1,0,0]'), ('[0,1,0]'), ('[0,0,1]'), ('[1,1,0]')",
        )
        .unwrap();
        Spi::run("CREATE INDEX ON test_hv_cos USING hnsw (v halfvec_cosine_ops)").unwrap();

        let result = Spi::get_one::<i32>(
            "SET enable_seqscan = off; \
             SELECT id FROM test_hv_cos \
             ORDER BY v <=> '[1,0,0]'::halfvec LIMIT 1",
        )
        .expect("SPI failed")
        .expect("NULL result");
        assert_eq!(result, 1);
    }
}
