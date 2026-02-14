//! HNSW index access method handler.

use pgrx::prelude::*;
use pgrx::{pg_sys, PgBox};

use super::build::{ambuild, ambuildempty};
use super::insert::aminsert;
use super::options::{amoptions, HNSW_EF_SEARCH};
use super::scan::{ambeginscan, amendscan, amgettuple, amrescan, get_meta_page_info};
use super::vacuum::{ambulkdelete, amvacuumcleanup};
use crate::hnsw_constants::{hnsw_get_layer_m, hnsw_get_ml};

/// Cost estimate for HNSW index scans.
///
/// Uses PostgreSQL's `genericcostestimate` as a base, then adjusts costs
/// using HNSW-specific parameters (m, ef_search) to model the graph
/// traversal cost. Returns infinity when no ORDER BY is present.
#[pg_guard]
#[allow(clippy::too_many_arguments)]
unsafe extern "C-unwind" fn amcostestimate(
    root: *mut pg_sys::PlannerInfo,
    path: *mut pg_sys::IndexPath,
    loop_count: f64,
    index_startup_cost: *mut pg_sys::Cost,
    index_total_cost: *mut pg_sys::Cost,
    index_selectivity: *mut pg_sys::Selectivity,
    index_correlation: *mut f64,
    index_pages: *mut f64,
) {
    // SAFETY: All pointers are provided by the PostgreSQL planner and are
    // guaranteed to be valid.
    if path.is_null() {
        return;
    }

    // Never use index without ORDER BY — HNSW requires a distance ordering.
    if (*path).indexorderbys.is_null() {
        *index_startup_cost = f64::INFINITY;
        *index_total_cost = f64::INFINITY;
        *index_selectivity = 0.0;
        *index_correlation = 0.0;
        *index_pages = 0.0;
        (*path).path.disabled_nodes = 2;
        return;
    }

    // SAFETY: zeroed GenericCosts struct, matching C's MemSet(&costs, 0, ...).
    let mut costs: pg_sys::GenericCosts = std::mem::zeroed();

    pg_sys::genericcostestimate(root, path, loop_count, &mut costs);

    // Read m from the index meta page.
    let index = pg_sys::index_open((*(*path).indexinfo).indexoid, pg_sys::NoLock as i32);
    let (m, _, _, _) = get_meta_page_info(index);
    pg_sys::index_close(index, pg_sys::NoLock as i32);

    let ef_search = HNSW_EF_SEARCH.get() as f64;

    // Estimate the ratio of tuples scanned during HNSW traversal.
    let num_tuples = (*(*path).indexinfo).tuples;
    let ratio = if num_tuples > 0.0 {
        let scaling_factor = 0.55_f64;
        let entry_level = (num_tuples.ln() * hnsw_get_ml(m)) as i32;
        let layer0_tuples_max = hnsw_get_layer_m(m, 0) as f64 * ef_search;
        let layer0_selectivity =
            scaling_factor * num_tuples.ln() / ((m as f64).ln() * (1.0 + ef_search.ln()));

        let r =
            (entry_level as f64 * m as f64 + layer0_tuples_max * layer0_selectivity) / num_tuples;
        r.min(1.0)
    } else {
        1.0
    };

    let mut spc_seq_page_cost: f64 = 0.0;
    pg_sys::get_tablespace_page_costs(
        (*(*path).indexinfo).reltablespace,
        std::ptr::null_mut(),
        &mut spc_seq_page_cost,
    );

    // Startup cost is the cost before returning the first row.
    costs.indexStartupCost = costs.indexTotalCost * ratio;

    // Adjust cost since TOAST is not included in seq scan cost.
    let startup_pages = costs.numIndexPages * ratio;
    if startup_pages > (*(*(*path).indexinfo).rel).pages as f64 && ratio < 0.5 {
        // Change all page cost from random to sequential.
        costs.indexStartupCost -= startup_pages * (costs.spc_random_page_cost - spc_seq_page_cost);
        // Remove cost of extra pages.
        costs.indexStartupCost -=
            (startup_pages - (*(*(*path).indexinfo).rel).pages as f64) * spc_seq_page_cost;
    }

    *index_startup_cost = costs.indexStartupCost;
    *index_total_cost = costs.indexTotalCost;
    *index_selectivity = costs.indexSelectivity;
    *index_correlation = costs.indexCorrelation;
    *index_pages = costs.numIndexPages;
}

/// Validate operator class — always returns true for now.
#[pg_guard]
unsafe extern "C-unwind" fn amvalidate(_opclassoid: pg_sys::Oid) -> bool {
    true
}

/// Returns the build phase name.
#[pg_guard]
unsafe extern "C-unwind" fn ambuildphasename(phasenum: pg_sys::int64) -> *mut std::ffi::c_char {
    use crate::hnsw_constants::PROGRESS_HNSW_PHASE_LOAD;
    match phasenum {
        PROGRESS_HNSW_PHASE_LOAD => pg_sys::pstrdup(c"loading tuples".as_ptr()),
        _ => std::ptr::null_mut(),
    }
}

/// HNSW index access method handler.
///
/// Registers the `hnsw` access method and returns the `IndexAmRoutine`
/// describing all supported operations.
#[pg_extern(sql = "
CREATE FUNCTION hnsw_handler(internal) RETURNS index_am_handler
  PARALLEL SAFE IMMUTABLE STRICT COST 0.0001
  LANGUAGE c AS 'MODULE_PATHNAME', '@FUNCTION_NAME@';
CREATE ACCESS METHOD hnsw TYPE INDEX HANDLER hnsw_handler;
COMMENT ON ACCESS METHOD hnsw IS 'hnsw index access method';
")]
fn hnsw_handler(_fcinfo: pg_sys::FunctionCallInfo) -> PgBox<pg_sys::IndexAmRoutine> {
    let mut amroutine = unsafe {
        // SAFETY: alloc_node zeroes all fields and sets the NodeTag.
        PgBox::<pg_sys::IndexAmRoutine>::alloc_node(pg_sys::NodeTag::T_IndexAmRoutine)
    };

    // Capabilities
    amroutine.amstrategies = 0;
    amroutine.amsupport = 3;
    amroutine.amoptsprocnum = 0;
    amroutine.amcanorder = false;
    amroutine.amcanorderbyop = true;
    amroutine.amcanhash = false;
    amroutine.amconsistentequality = false;
    amroutine.amconsistentordering = false;
    amroutine.amcanbackward = false;
    amroutine.amcanunique = false;
    amroutine.amcanmulticol = false;
    amroutine.amoptionalkey = true;
    amroutine.amsearcharray = false;
    amroutine.amsearchnulls = false;
    amroutine.amstorage = false;
    amroutine.amclusterable = false;
    amroutine.ampredlocks = false;
    amroutine.amcanparallel = false;
    amroutine.amcanbuildparallel = false;
    amroutine.amcaninclude = false;
    amroutine.amusemaintenanceworkmem = false;
    amroutine.amsummarizing = false;
    amroutine.amparallelvacuumoptions = pg_sys::VACUUM_OPTION_PARALLEL_BULKDEL as u8;
    amroutine.amkeytype = pg_sys::InvalidOid;

    // Interface functions
    amroutine.ambuild = Some(ambuild);
    amroutine.ambuildempty = Some(ambuildempty);
    amroutine.aminsert = Some(aminsert);
    amroutine.aminsertcleanup = None;
    amroutine.ambulkdelete = Some(ambulkdelete);
    amroutine.amvacuumcleanup = Some(amvacuumcleanup);
    amroutine.amcanreturn = None;
    amroutine.amcostestimate = Some(amcostestimate);
    amroutine.amgettreeheight = None;
    amroutine.amoptions = Some(amoptions);
    amroutine.amproperty = None;
    amroutine.ambuildphasename = Some(ambuildphasename);
    amroutine.amvalidate = Some(amvalidate);
    amroutine.amadjustmembers = None;
    amroutine.ambeginscan = Some(ambeginscan);
    amroutine.amrescan = Some(amrescan);
    amroutine.amgettuple = Some(amgettuple);
    amroutine.amgetbitmap = None;
    amroutine.amendscan = Some(amendscan);
    amroutine.ammarkpos = None;
    amroutine.amrestrpos = None;

    // Parallel scan (not supported)
    amroutine.amestimateparallelscan = None;
    amroutine.aminitparallelscan = None;
    amroutine.amparallelrescan = None;

    // PG18 strategy/cmptype translation
    amroutine.amtranslatestrategy = None;
    amroutine.amtranslatecmptype = None;

    amroutine.into_pg_boxed()
}

#[cfg(any(test, feature = "pg_test"))]
#[pgrx::pg_schema]
mod tests {
    use pgrx::prelude::*;

    #[pg_test]
    fn test_hnsw_am_exists() {
        let result = Spi::get_one::<String>("SELECT amname::text FROM pg_am WHERE amname = 'hnsw'")
            .expect("SPI failed")
            .expect("hnsw access method not found");
        assert_eq!(result, "hnsw");
    }

    #[pg_test]
    fn test_hnsw_am_is_index_type() {
        let result = Spi::get_one::<String>("SELECT amtype::text FROM pg_am WHERE amname = 'hnsw'")
            .expect("SPI failed")
            .expect("hnsw access method not found");
        assert_eq!(result, "i");
    }
}
