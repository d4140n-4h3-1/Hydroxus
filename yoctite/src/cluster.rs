//! Splitting triangles into clusters: up to [`MAX_TRIANGLES`] triangles over up to
//! [`MAX_VERTICES`] vertices each, close together and, where they can be, connected.

use crate::{
    math::{morton_order, V3},
    Mesh,
};

/// The most triangles a cluster has.
pub const MAX_TRIANGLES: usize = 128;
/// The most vertices a cluster's triangles use: few enough to number them in a byte.
pub const MAX_VERTICES: usize = 255;

/// Splits `triangles` (indices into `mesh`'s vertices) into clusters, each a list of indices
/// into `triangles`.
///
/// A cluster grows from a seed triangle over the triangles that share its vertices (welded by
/// position, so seams in texture coordinates do not split it), taking first the one that adds
/// fewest vertices to it and, among those, the one nearest its middle. When nothing connected is
/// left - foliage is made of many small separate cards - it goes on with the next triangle along
/// a Morton curve through their middles, which is somewhere close by, rather than leaving a
/// cluster of two triangles.
pub(crate) fn clusterize(mesh: &Mesh, triangles: &[[u32; 3]]) -> Vec<Vec<usize>> {
    let count = triangles.len();
    if count == 0 {
        return Vec::new();
    }
    let middles: Vec<V3> = triangles
        .iter()
        .map(|t| (mesh.position(t[0]) + mesh.position(t[1]) + mesh.position(t[2])) * (1.0 / 3.0))
        .collect();
    let order = morton_order(&middles);

    // Which triangles use each welded vertex, packed: those of vertex `w` are
    // `users[first[w]..first[w + 1]]`.
    let welded = mesh.welded_count();
    let mut first = vec![0u32; welded + 1];
    for t in triangles {
        for &v in t {
            first[mesh.weld[v as usize] as usize + 1] += 1;
        }
    }
    for w in 0..welded {
        first[w + 1] += first[w];
    }
    let mut fill = first.clone();
    let mut users = vec![0u32; first[welded] as usize];
    for (i, t) in triangles.iter().enumerate() {
        for &v in t {
            let w = mesh.weld[v as usize] as usize;
            users[fill[w] as usize] = i as u32;
            fill[w] += 1;
        }
    }

    // Stamps, by cluster number + 1, of the triangles in the current cluster's frontier and the
    // vertices it uses.
    let mut assigned = vec![false; count];
    let mut in_frontier = vec![0u32; count];
    let mut vertex_stamp = vec![0u32; mesh.positions.len()];
    let mut cursor = 0;
    let mut clusters = Vec::new();

    loop {
        while cursor < count && assigned[order[cursor]] {
            cursor += 1;
        }
        if cursor == count {
            break;
        }
        let stamp = clusters.len() as u32 + 1;
        let mut members: Vec<usize> = Vec::with_capacity(MAX_TRIANGLES);
        let mut vertex_count = 0;
        let mut middle_sum = V3::default();
        let mut frontier: Vec<usize> = Vec::new();

        let mut next = Some(order[cursor]);
        while let Some(t) = next.take() {
            assigned[t] = true;
            members.push(t);
            middle_sum += middles[t];
            for &v in &triangles[t] {
                if vertex_stamp[v as usize] != stamp {
                    vertex_stamp[v as usize] = stamp;
                    vertex_count += 1;
                }
                let w = mesh.weld[v as usize] as usize;
                for &u in &users[first[w] as usize..first[w + 1] as usize] {
                    let u = u as usize;
                    if !assigned[u] && in_frontier[u] != stamp {
                        in_frontier[u] = stamp;
                        frontier.push(u);
                    }
                }
            }
            if members.len() == MAX_TRIANGLES {
                break;
            }

            let middle = middle_sum * (1.0 / members.len() as f64);
            let new_vertices = |t: usize| {
                let corners = &triangles[t];
                (0..3)
                    .filter(|&k| {
                        vertex_stamp[corners[k] as usize] != stamp && !corners[..k].contains(&corners[k])
                    })
                    .count()
            };
            frontier.retain(|&u| !assigned[u]);
            let mut best: Option<(usize, f64, usize)> = None;
            for &u in &frontier {
                let new = new_vertices(u);
                if vertex_count + new > MAX_VERTICES {
                    continue;
                }
                let distance = (middles[u] - middle).length();
                if best.is_none_or(|(n, d, _)| new < n || (new == n && distance < d)) {
                    best = Some((new, distance, u));
                }
            }
            next = match best {
                Some((_, _, u)) => Some(u),
                None if frontier.is_empty() => {
                    // Nothing connected left: the next free triangle along the curve.
                    while cursor < count && assigned[order[cursor]] {
                        cursor += 1;
                    }
                    (cursor < count)
                        .then(|| order[cursor])
                        .filter(|&u| vertex_count + new_vertices(u) <= MAX_VERTICES)
                }
                // Connected triangles are left, but none fits.
                None => None,
            };
        }
        clusters.push(members);
    }
    clusters
}
