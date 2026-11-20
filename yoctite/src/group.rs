//! Grouping clusters: those simplified together, as one, to make the next level of detail.
//!
//! A group's outer border is left as it is when it is simplified, so the clusters drawn on
//! either side of it, at whatever levels, always meet. The fewer of its edges are on that border
//! the more freely it can be simplified, so a group is made of clusters that share as many
//! vertices as they can.

use crate::{
    math::{morton_order, V3},
    Mesh,
};

/// Groups `clusters`, each given as its triangles (indices into `mesh`'s vertices) and its
/// middle, into groups of up to `size`. Each group is a list of indices into `clusters`.
///
/// Groups are started from clusters in Morton order, and each grown by the cluster sharing most
/// welded vertices with it so far. A cluster with no neighbour left - a clump of foliage cards -
/// is grouped with the next one along the curve, which is close by.
pub(crate) fn group(mesh: &Mesh, clusters: &[(&[[u32; 3]], V3)], size: usize) -> Vec<Vec<usize>> {
    let count = clusters.len();
    // Each cluster's welded vertices, and the clusters each welded vertex is used by.
    let mut vertices: Vec<Vec<u32>> = Vec::with_capacity(count);
    let mut users: std::collections::HashMap<u32, Vec<u32>> = Default::default();
    for (c, (triangles, _)) in clusters.iter().enumerate() {
        let mut welded: Vec<u32> = triangles
            .iter()
            .flat_map(|t| t.iter().map(|&v| mesh.weld[v as usize]))
            .collect();
        welded.sort_unstable();
        welded.dedup();
        for &w in &welded {
            users.entry(w).or_default().push(c as u32);
        }
        vertices.push(welded);
    }
    let middles: Vec<V3> = clusters.iter().map(|(_, m)| *m).collect();
    let order = morton_order(&middles);

    let mut grouped = vec![false; count];
    // How many vertices each cluster shares with the group being made.
    let mut shared = vec![0u32; count];
    let mut touched: Vec<usize> = Vec::new();
    let mut cursor = 0;
    let mut groups = Vec::new();
    loop {
        while cursor < count && grouped[order[cursor]] {
            cursor += 1;
        }
        if cursor == count {
            break;
        }
        let mut members = Vec::with_capacity(size);
        let mut next = Some(order[cursor]);
        while let Some(c) = next.take() {
            grouped[c] = true;
            members.push(c);
            for w in &vertices[c] {
                for &u in &users[w] {
                    let u = u as usize;
                    if !grouped[u] {
                        if shared[u] == 0 {
                            touched.push(u);
                        }
                        shared[u] += 1;
                    }
                }
            }
            if members.len() == size {
                break;
            }
            next = touched
                .iter()
                .copied()
                .filter(|&u| !grouped[u])
                .max_by_key(|&u| (shared[u], std::cmp::Reverse(u)));
            if next.is_none() {
                while cursor < count && grouped[order[cursor]] {
                    cursor += 1;
                }
                next = (cursor < count).then(|| order[cursor]);
            }
        }
        for u in touched.drain(..) {
            shared[u] = 0;
        }
        groups.push(members);
    }
    groups
}
