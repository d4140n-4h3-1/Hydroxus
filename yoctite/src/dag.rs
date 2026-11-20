//! Building the graph of clusters, level after level.

use crate::{
    cluster::clusterize,
    group::group,
    math::{Sphere, V3},
    simplify::simplify,
    Cluster, ClusterDag, Group, Mesh, Settings,
};
use std::collections::HashSet;

/// A cluster made from `triangles` (indices into the mesh's vertices).
fn make_cluster(mesh: &Mesh, triangles: &[[u32; 3]], level: u32, made_from: Option<u32>) -> Cluster {
    let mut vertices: Vec<u32> = Vec::new();
    let local = triangles
        .iter()
        .map(|t| {
            t.map(|v| {
                let i = vertices.iter().position(|&u| u == v).unwrap_or_else(|| {
                    vertices.push(v);
                    vertices.len() - 1
                });
                i as u8
            })
        })
        .collect();
    let bounds = Sphere::around(vertices.iter().map(|&v| mesh.position(v)));
    Cluster {
        vertices,
        triangles: local,
        bounds,
        error: 0.0,
        lod_bounds: bounds,
        parent_error: f32::INFINITY,
        parent_bounds: bounds,
        level,
        made_from,
        simplified_in: None,
    }
}

/// Splits `triangles` into clusters and adds them to `dag`, returning their indices.
fn add_clusters(
    dag: &mut ClusterDag,
    mesh: &Mesh,
    triangles: &[[u32; 3]],
    level: u32,
    made_from: Option<u32>,
) -> Vec<u32> {
    clusterize(mesh, triangles)
        .into_iter()
        .map(|members| {
            let picked: Vec<[u32; 3]> = members.iter().map(|&t| triangles[t]).collect();
            dag.clusters.push(make_cluster(mesh, &picked, level, made_from));
            dag.clusters.len() as u32 - 1
        })
        .collect()
}

/// Builds the clusters of `mesh` at every level of detail. Triangles with two corners at the
/// same position, which cover nothing, are left out.
pub fn build(mesh: &Mesh, settings: Settings) -> ClusterDag {
    let triangles: Vec<[u32; 3]> = mesh
        .indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .filter(|t| {
            let w = t.map(|v| mesh.weld[v as usize]);
            w[0] != w[1] && w[1] != w[2] && w[0] != w[2]
        })
        .collect();
    let mut dag = ClusterDag::default();
    let mut current = add_clusters(&mut dag, mesh, &triangles, 0, None);
    // The welded vertices of clusters that became roots below the top: whatever is made next
    // to them later has to keep meeting them, so they never move again.
    let mut frozen: HashSet<u32> = HashSet::new();
    let mut level = 0;

    while current.len() > 1 {
        let described: Vec<(Vec<[u32; 3]>, V3)> = current
            .iter()
            .map(|&c| {
                let cluster = &dag.clusters[c as usize];
                (cluster.mesh_triangles().collect(), V3::from_f32(cluster.bounds.center))
            })
            .collect();
        let views: Vec<(&[[u32; 3]], V3)> =
            described.iter().map(|(t, m)| (t.as_slice(), *m)).collect();
        let groups = group(mesh, &views, settings.group_size);

        // A welded vertex in more than one group is on a border between groups, which
        // simplifying either may not move.
        let mut owner: Vec<u32> = vec![u32::MAX; mesh.welded_count()];
        let mut border: HashSet<u32> = frozen.clone();
        for (g, members) in groups.iter().enumerate() {
            for &m in members {
                for t in &described[m].0 {
                    for &v in t {
                        let w = mesh.weld[v as usize];
                        let o = &mut owner[w as usize];
                        if *o == u32::MAX {
                            *o = g as u32;
                        } else if *o != g as u32 {
                            border.insert(w);
                        }
                    }
                }
            }
        }

        let mut next = Vec::new();
        for members in groups {
            let children: Vec<u32> = members.iter().map(|&m| current[m]).collect();
            let triangles: Vec<[u32; 3]> = members
                .iter()
                .flat_map(|&m| described[m].0.iter().copied())
                .collect();
            let simplified = simplify(mesh, &triangles, |w| border.contains(&w), triangles.len() / 2);
            let reduced = (simplified.triangles.len() as f32)
                <= triangles.len() as f32 * settings.least_reduction;
            if !reduced || simplified.triangles.is_empty() {
                // As simple as it gets: these stay roots.
                for &c in &children {
                    for v in dag.clusters[c as usize].vertices.iter() {
                        frozen.insert(mesh.weld[*v as usize]);
                    }
                }
                continue;
            }

            // How far from the full mesh, at most: as the simplification moved the surface, or
            // as far as the children are and the texture was dragged from them on top, as drag
            // adds up level on level.
            let mut error = simplified.error;
            let mut bounds: Option<Sphere> = None;
            for &c in &children {
                let child = &dag.clusters[c as usize];
                error = error.max(child.error).max(child.error + simplified.dragged);
                bounds = Some(bounds.map_or(child.lod_bounds, |b| b.enclosing(child.lod_bounds)));
            }
            let bounds = bounds.expect("a group has clusters");
            let g = dag.groups.len() as u32;
            let parents = add_clusters(&mut dag, mesh, &simplified.triangles, level + 1, Some(g));
            for &p in &parents {
                let parent = &mut dag.clusters[p as usize];
                parent.error = error;
                parent.lod_bounds = bounds;
            }
            for &c in &children {
                let child = &mut dag.clusters[c as usize];
                child.parent_error = error;
                child.parent_bounds = bounds;
                child.simplified_in = Some(g);
            }
            dag.groups.push(Group {
                children,
                parents: parents.clone(),
                error,
                bounds,
                level,
            });
            next.extend(parents);
        }
        if next.is_empty() {
            break;
        }
        current = next;
        level += 1;
    }
    dag
}
