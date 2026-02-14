//! Graph algorithms for HNSW index.
//!
//! Implements the core HNSW algorithms for in-memory graph construction:
//! - Greedy search at a specific layer
//! - Heuristic neighbor selection
//! - Connection updates

use std::collections::{BinaryHeap, HashSet};

use crate::hnsw_constants::hnsw_get_layer_m;

/// Index of an element in the in-memory element arena.
pub type ElementIdx = usize;

/// A candidate with its distance, used during search and neighbor selection.
#[derive(Debug, Clone, Copy)]
pub struct Candidate {
    /// Distance from this candidate to the reference point.
    pub distance: f32,
    /// Index of the element in the arena.
    pub idx: ElementIdx,
}

/// In-memory neighbor array for one layer of an element.
#[derive(Debug, Clone)]
pub struct NeighborArray {
    /// Maximum number of neighbors (M or 2*M for layer 0).
    pub max_len: usize,
    /// The neighbor candidates, ordered by distance.
    pub items: Vec<Candidate>,
    /// Whether the `closer` flags have been computed for caching.
    pub closer_set: bool,
}

impl NeighborArray {
    /// Creates a new neighbor array with the given capacity.
    pub fn new(max_len: usize) -> Self {
        Self {
            max_len,
            items: Vec::with_capacity(max_len),
            closer_set: false,
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
}

/// In-memory element for graph construction.
#[derive(Debug)]
pub struct GraphElement {
    /// Level assigned to this element (0-based, higher = fewer elements).
    pub level: i32,
    /// Neighbor arrays, one per layer (index 0 = layer 0).
    pub neighbors: Vec<NeighborArray>,
    /// Offset into the values arena for this element's vector data.
    pub value_offset: usize,
    /// Size of this element's vector data in bytes.
    pub value_size: usize,
}

impl GraphElement {
    /// Creates a new element with neighbor arrays for each layer.
    pub fn new(level: i32, m: i32, value_offset: usize, value_size: usize) -> Self {
        let mut neighbors = Vec::with_capacity(level as usize + 1);
        for lc in 0..=level {
            let lm = hnsw_get_layer_m(m, lc) as usize;
            neighbors.push(NeighborArray::new(lm));
        }
        Self {
            level,
            neighbors,
            value_offset,
            value_size,
        }
    }
}

/// Wrapper for nearest-first ordering in a BinaryHeap.
#[derive(Debug, Clone, Copy)]
struct NearestCandidate(Candidate);

impl PartialEq for NearestCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}
impl Eq for NearestCandidate {}

impl PartialOrd for NearestCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for NearestCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reversed: smallest distance = highest priority
        other
            .0
            .distance
            .partial_cmp(&self.0.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Wrapper for furthest-first ordering in a BinaryHeap.
#[derive(Debug, Clone, Copy)]
struct FurthestCandidate(Candidate);

impl PartialEq for FurthestCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.0.distance == other.0.distance
    }
}
impl Eq for FurthestCandidate {}

impl PartialOrd for FurthestCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FurthestCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Normal: largest distance = highest priority
        self.0
            .distance
            .partial_cmp(&other.0.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Distance function signature for in-memory graph operations.
///
/// Given the values arena and two value offsets/sizes, returns the distance.
pub type DistanceFn =
    fn(values: &[u8], a_offset: usize, a_size: usize, b_offset: usize, b_size: usize) -> f32;

/// Search a specific layer of the HNSW graph for nearest neighbors.
///
/// Implements Algorithm 2 from the HNSW paper (in-memory variant).
///
/// # Arguments
/// * `elements` - The element arena
/// * `values` - Raw byte arena containing all vector values
/// * `entry_points` - Initial entry points with their distances
/// * `ef` - Size of the dynamic candidate list (number of results)
/// * `layer` - Layer to search
/// * `query_offset` - Offset of the query vector in `values`
/// * `query_size` - Size of the query vector data
/// * `distance_fn` - Distance computation function
#[allow(clippy::too_many_arguments)]
pub fn search_layer(
    elements: &[GraphElement],
    values: &[u8],
    entry_points: &[Candidate],
    ef: usize,
    layer: i32,
    query_offset: usize,
    query_size: usize,
    distance_fn: DistanceFn,
) -> Vec<Candidate> {
    let mut visited = HashSet::with_capacity(ef * 2);
    // C = nearest candidates (min-heap by distance)
    let mut candidates: BinaryHeap<NearestCandidate> = BinaryHeap::new();
    // W = result set (max-heap by distance, so we can evict furthest)
    let mut results: BinaryHeap<FurthestCandidate> = BinaryHeap::new();
    let mut result_len: usize = 0;

    // Initialize with entry points
    for ep in entry_points {
        visited.insert(ep.idx);
        candidates.push(NearestCandidate(*ep));
        results.push(FurthestCandidate(*ep));
        result_len += 1;
    }

    while let Some(NearestCandidate(c)) = candidates.pop() {
        // Get furthest element in W
        let f_dist = results.peek().map(|f| f.0.distance).unwrap_or(f32::MAX);

        // If closest candidate is further than furthest result, stop
        if c.distance > f_dist {
            break;
        }

        let c_elem = &elements[c.idx];

        // Skip if element doesn't have this layer
        if c_elem.level < layer {
            continue;
        }

        // Load unvisited neighbors at this layer
        let neighborhood = &c_elem.neighbors[layer as usize];

        for neighbor in &neighborhood.items {
            if visited.contains(&neighbor.idx) {
                continue;
            }
            visited.insert(neighbor.idx);

            let e_elem = &elements[neighbor.idx];

            // Skip if element level is below search layer
            if e_elem.level < layer {
                continue;
            }

            let e_distance = distance_fn(
                values,
                query_offset,
                query_size,
                e_elem.value_offset,
                e_elem.value_size,
            );

            let always_add = result_len < ef;
            let f_dist = results.peek().map(|f| f.0.distance).unwrap_or(f32::MAX);

            if e_distance < f_dist || always_add {
                let e_cand = Candidate {
                    distance: e_distance,
                    idx: neighbor.idx,
                };
                candidates.push(NearestCandidate(e_cand));
                results.push(FurthestCandidate(e_cand));
                result_len += 1;

                // Evict furthest if over ef
                if result_len > ef {
                    results.pop();
                    result_len -= 1;
                }
            }
        }
    }

    // Collect results (nearest first)
    let mut result: Vec<Candidate> = results.into_iter().map(|fc| fc.0).collect();
    result.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    result
}

/// Select neighbors using the heuristic algorithm from the HNSW paper.
///
/// Implements Algorithm 4: if candidates fit within `max_neighbors`, returns
/// all. Otherwise, uses a closer/farther classification to select the best
/// subset.
///
/// # Arguments
/// * `elements` - The element arena
/// * `values` - Raw byte arena containing all vector values
/// * `candidates` - Candidate neighbors sorted by distance (nearest first)
/// * `max_neighbors` - Maximum number of neighbors to return (lm)
/// * `distance_fn` - Distance computation function
pub fn select_neighbors(
    elements: &[GraphElement],
    values: &[u8],
    candidates: &[Candidate],
    max_neighbors: usize,
    distance_fn: DistanceFn,
) -> Vec<Candidate> {
    if candidates.len() <= max_neighbors {
        return candidates.to_vec();
    }

    let mut result: Vec<Candidate> = Vec::with_capacity(max_neighbors);
    let mut discarded: Vec<Candidate> = Vec::new();

    // Process candidates in order of increasing distance
    for &e in candidates {
        if result.len() >= max_neighbors {
            break;
        }

        // Check if e is closer to query than to any element already in result
        let closer = check_element_closer(elements, values, &e, &result, distance_fn);

        if closer {
            result.push(e);
        } else {
            discarded.push(e);
        }
    }

    // Fill remaining slots from discarded candidates
    for &d in &discarded {
        if result.len() >= max_neighbors {
            break;
        }
        result.push(d);
    }

    result
}

/// Check if a candidate is closer to the query than to any element in the
/// result set.
///
/// Returns `true` if `e` is closer to the query point than to every element
/// in `r` (i.e., it should be kept as a neighbor).
fn check_element_closer(
    elements: &[GraphElement],
    values: &[u8],
    e: &Candidate,
    r: &[Candidate],
    distance_fn: DistanceFn,
) -> bool {
    let e_elem = &elements[e.idx];

    for ri in r {
        let ri_elem = &elements[ri.idx];
        let distance = distance_fn(
            values,
            e_elem.value_offset,
            e_elem.value_size,
            ri_elem.value_offset,
            ri_elem.value_size,
        );
        if distance <= e.distance {
            return false;
        }
    }

    true
}

/// Find element neighbors at all layers using the HNSW algorithm.
///
/// Implements Algorithm 1 from the HNSW paper (INSERT without disk I/O).
/// Searches from the entry point down through layers to find and set
/// neighbors for the new element.
///
/// # Arguments
/// * `elements` - The element arena
/// * `values` - Raw byte arena
/// * `new_idx` - Index of the newly inserted element
/// * `entry_idx` - Index of the current graph entry point
/// * `ef_construction` - Size of dynamic candidate list during construction
/// * `m` - Max connections parameter
/// * `distance_fn` - Distance computation function
pub fn find_element_neighbors(
    elements: &mut [GraphElement],
    values: &[u8],
    new_idx: ElementIdx,
    entry_idx: ElementIdx,
    ef_construction: i32,
    m: i32,
    distance_fn: DistanceFn,
) {
    let new_level = elements[new_idx].level;
    let entry_level = elements[entry_idx].level;
    let new_offset = elements[new_idx].value_offset;
    let new_size = elements[new_idx].value_size;

    // Calculate initial distance from new element to entry point
    let ep_elem = &elements[entry_idx];
    let ep_distance = distance_fn(
        values,
        new_offset,
        new_size,
        ep_elem.value_offset,
        ep_elem.value_size,
    );

    let mut ep = vec![Candidate {
        distance: ep_distance,
        idx: entry_idx,
    }];

    // Phase 1: Greedy search from top layer down to new element's level + 1
    for lc in (new_level + 1..=entry_level).rev() {
        let w = search_layer(
            elements,
            values,
            &ep,
            1,
            lc,
            new_offset,
            new_size,
            distance_fn,
        );
        if !w.is_empty() {
            ep = vec![w[0]]; // Take nearest as new entry point
        }
    }

    // Phase 2: Search and connect at each layer from min(new_level, entry_level) down to 0
    let start_level = std::cmp::min(new_level, entry_level);
    for lc in (0..=start_level).rev() {
        let lm = hnsw_get_layer_m(m, lc) as usize;

        // Search with ef_construction candidates
        let w = search_layer(
            elements,
            values,
            &ep,
            ef_construction as usize,
            lc,
            new_offset,
            new_size,
            distance_fn,
        );

        // Select neighbors from candidates
        let neighbors = select_neighbors(elements, values, &w, lm, distance_fn);

        // Add connections from new element to selected neighbors
        elements[new_idx].neighbors[lc as usize].items = neighbors.clone();

        // Use the search results as entry points for next layer
        ep = w;
    }
}

/// Update neighbor connections of existing elements after a new element is
/// inserted.
///
/// For each neighbor of the new element, adds a back-connection from the
/// neighbor to the new element. If the neighbor's list is full, uses
/// `select_neighbors` to prune the worst connection.
///
/// # Arguments
/// * `elements` - The element arena
/// * `values` - Raw byte arena
/// * `new_idx` - Index of the newly inserted element
/// * `m` - Max connections parameter
/// * `distance_fn` - Distance computation function
pub fn update_neighbor_connections(
    elements: &mut [GraphElement],
    values: &[u8],
    new_idx: ElementIdx,
    m: i32,
    distance_fn: DistanceFn,
) {
    let new_level = elements[new_idx].level;

    for lc in (0..=new_level).rev() {
        let lm = hnsw_get_layer_m(m, lc) as usize;

        // Collect neighbors at this layer (snapshot to avoid borrow issues)
        let neighbor_snapshot: Vec<Candidate> =
            elements[new_idx].neighbors[lc as usize].items.clone();

        for hc in &neighbor_snapshot {
            let neighbor_idx = hc.idx;

            // Build new candidate for the back-connection
            let new_candidate = Candidate {
                distance: hc.distance,
                idx: new_idx,
            };

            let neighbors = &mut elements[neighbor_idx].neighbors[lc as usize];

            if neighbors.len() < lm {
                // Room available — just append
                neighbors.items.push(new_candidate);
            } else {
                // Need to prune: add new candidate and select best subset
                let mut all_candidates: Vec<Candidate> = neighbors.items.clone();
                all_candidates.push(new_candidate);

                // Sort by distance for select_neighbors
                all_candidates.sort_by(|a, b| {
                    a.distance
                        .partial_cmp(&b.distance)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });

                let selected = select_neighbors(elements, values, &all_candidates, lm, distance_fn);
                elements[neighbor_idx].neighbors[lc as usize].items = selected;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Simple L2 squared distance for testing.
    fn test_l2_distance(
        values: &[u8],
        a_offset: usize,
        a_size: usize,
        b_offset: usize,
        b_size: usize,
    ) -> f32 {
        let dim = a_size / std::mem::size_of::<f32>();
        assert_eq!(a_size, b_size);
        let a_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                values[a_offset..a_offset + a_size].as_ptr() as *const f32,
                dim,
            )
        };
        let b_floats: &[f32] = unsafe {
            std::slice::from_raw_parts(
                values[b_offset..b_offset + b_size].as_ptr() as *const f32,
                dim,
            )
        };
        a_floats
            .iter()
            .zip(b_floats.iter())
            .map(|(x, y)| (x - y) * (x - y))
            .sum()
    }

    /// Helper to store a vector in the values arena.
    fn store_vector(values: &mut Vec<u8>, vec: &[f32]) -> (usize, usize) {
        let offset = values.len();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                vec.as_ptr() as *const u8,
                vec.len() * std::mem::size_of::<f32>(),
            )
        };
        values.extend_from_slice(bytes);
        (offset, bytes.len())
    }

    #[test]
    fn test_search_layer_basic() {
        let m = 4;
        let mut values = Vec::new();

        // Create 5 elements at positions: [0], [1], [2], [3], [4]
        let positions: Vec<[f32; 1]> = vec![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let mut elems: Vec<GraphElement> = Vec::new();
        for pos in &positions {
            let (off, sz) = store_vector(&mut values, pos);
            elems.push(GraphElement::new(0, m, off, sz));
        }

        // Manually connect: 0-1-2-3-4 (chain)
        for i in 0..4usize {
            let dist = 1.0f32;
            elems[i].neighbors[0].items.push(Candidate {
                distance: dist,
                idx: i + 1,
            });
            elems[i + 1].neighbors[0].items.push(Candidate {
                distance: dist,
                idx: i,
            });
        }

        // Query for element nearest to [3.5], starting from element 0
        let (q_off, q_sz) = store_vector(&mut values, &[3.5f32]);
        let ep_dist = test_l2_distance(
            &values,
            q_off,
            q_sz,
            elems[0].value_offset,
            elems[0].value_size,
        );
        let ep = vec![Candidate {
            distance: ep_dist,
            idx: 0,
        }];

        let results = search_layer(&elems, &values, &ep, 2, 0, q_off, q_sz, test_l2_distance);

        // Should find elements 3 and 4 as nearest
        assert_eq!(results.len(), 2);
        let result_indices: Vec<usize> = results.iter().map(|c| c.idx).collect();
        assert!(result_indices.contains(&3));
        assert!(result_indices.contains(&4));
    }

    #[test]
    fn test_select_neighbors_fits() {
        let m = 4;
        let mut values = Vec::new();
        let positions: Vec<[f32; 1]> = vec![[0.0], [1.0], [2.0]];
        let mut elems: Vec<GraphElement> = Vec::new();
        for pos in &positions {
            let (off, sz) = store_vector(&mut values, pos);
            elems.push(GraphElement::new(0, m, off, sz));
        }

        let candidates = vec![
            Candidate {
                distance: 1.0,
                idx: 1,
            },
            Candidate {
                distance: 2.0,
                idx: 2,
            },
        ];

        // max_neighbors = 4, only 2 candidates → returns all
        let selected = select_neighbors(&elems, &values, &candidates, 4, test_l2_distance);
        assert_eq!(selected.len(), 2);
    }

    #[test]
    fn test_select_neighbors_prunes() {
        let m = 4;
        let mut values = Vec::new();

        // 4 elements: query at origin, candidates at [1,0], [1.1,0], [5,0]
        let positions: Vec<[f32; 2]> = vec![[0.0, 0.0], [1.0, 0.0], [1.1, 0.0], [5.0, 0.0]];
        let mut elems: Vec<GraphElement> = Vec::new();
        for pos in &positions {
            let (off, sz) = store_vector(&mut values, pos);
            elems.push(GraphElement::new(0, m, off, sz));
        }

        let candidates = vec![
            Candidate {
                distance: 1.0,
                idx: 1,
            },
            Candidate {
                distance: 1.21,
                idx: 2,
            },
            Candidate {
                distance: 25.0,
                idx: 3,
            },
        ];

        // max_neighbors = 2: should keep idx 1 (closest) and prune idx 2
        // (too close to idx 1) and take idx 3 to fill slot
        let selected = select_neighbors(&elems, &values, &candidates, 2, test_l2_distance);
        assert_eq!(selected.len(), 2);
        assert_eq!(selected[0].idx, 1); // nearest
    }

    #[test]
    fn test_find_element_neighbors_two_elements() {
        let m = 4;
        let ef_construction = 16;
        let mut values = Vec::new();

        let (off0, sz0) = store_vector(&mut values, &[0.0f32, 0.0]);
        let (off1, sz1) = store_vector(&mut values, &[1.0f32, 0.0]);

        let mut elems = vec![
            GraphElement::new(0, m, off0, sz0),
            GraphElement::new(0, m, off1, sz1),
        ];

        // Insert element 1 with element 0 as entry point
        find_element_neighbors(
            &mut elems,
            &values,
            1,
            0,
            ef_construction,
            m,
            test_l2_distance,
        );

        // Element 1 should have element 0 as neighbor
        assert_eq!(elems[1].neighbors[0].items.len(), 1);
        assert_eq!(elems[1].neighbors[0].items[0].idx, 0);
    }

    #[test]
    fn test_update_neighbor_connections() {
        let m = 4;
        let ef_construction = 16;
        let mut values = Vec::new();

        let (off0, sz0) = store_vector(&mut values, &[0.0f32]);
        let (off1, sz1) = store_vector(&mut values, &[1.0f32]);
        let (off2, sz2) = store_vector(&mut values, &[2.0f32]);

        let mut elems = vec![
            GraphElement::new(0, m, off0, sz0),
            GraphElement::new(0, m, off1, sz1),
            GraphElement::new(0, m, off2, sz2),
        ];

        // Insert element 1 with element 0 as entry
        find_element_neighbors(
            &mut elems,
            &values,
            1,
            0,
            ef_construction,
            m,
            test_l2_distance,
        );
        update_neighbor_connections(&mut elems, &values, 1, m, test_l2_distance);

        // Element 0 should now have element 1 as neighbor (back-connection)
        assert!(!elems[0].neighbors[0].is_empty());
        assert_eq!(elems[0].neighbors[0].items[0].idx, 1);

        // Insert element 2 with element 0 as entry
        find_element_neighbors(
            &mut elems,
            &values,
            2,
            0,
            ef_construction,
            m,
            test_l2_distance,
        );
        update_neighbor_connections(&mut elems, &values, 2, m, test_l2_distance);

        // Element 1 should have both 0 and 2 as neighbors
        let n1_indices: Vec<usize> = elems[1].neighbors[0].items.iter().map(|c| c.idx).collect();
        assert!(n1_indices.contains(&0));
        assert!(n1_indices.contains(&2));
    }

    #[test]
    fn test_multi_layer_search() {
        let m = 2;
        let ef_construction = 8;
        let mut values = Vec::new();

        // Create elements at different levels
        // Element 0: level 2 at position [0]
        // Element 1: level 0 at position [3]
        // Element 2: level 1 at position [1]
        let (off0, sz0) = store_vector(&mut values, &[0.0f32]);
        let (off1, sz1) = store_vector(&mut values, &[3.0f32]);
        let (off2, sz2) = store_vector(&mut values, &[1.0f32]);

        let mut elems = vec![
            GraphElement::new(2, m, off0, sz0),
            GraphElement::new(0, m, off1, sz1),
            GraphElement::new(1, m, off2, sz2),
        ];

        // Insert element 1 with element 0 as entry
        find_element_neighbors(
            &mut elems,
            &values,
            1,
            0,
            ef_construction,
            m,
            test_l2_distance,
        );
        update_neighbor_connections(&mut elems, &values, 1, m, test_l2_distance);

        // Element 1 (level 0) should connect to element 0 at layer 0
        assert!(!elems[1].neighbors[0].is_empty());

        // Insert element 2 (level 1) with element 0 as entry
        find_element_neighbors(
            &mut elems,
            &values,
            2,
            0,
            ef_construction,
            m,
            test_l2_distance,
        );
        update_neighbor_connections(&mut elems, &values, 2, m, test_l2_distance);

        // Element 2 should have neighbors at both layer 0 and layer 1
        assert!(!elems[2].neighbors[0].is_empty());
        assert!(!elems[2].neighbors[1].is_empty());
    }
}
