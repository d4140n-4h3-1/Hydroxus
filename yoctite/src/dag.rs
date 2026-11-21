//! Building the graph of clusters, level after level.

use crate::{
    cluster::clusterize,
    group::group,
    math::{Sphere, V3},
    prune,
    simplify::simplify,
    Cluster, ClusterDag, ExtraVertex, Group, Mesh, Settings,
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

/// What simplifying one group made: its new triangles, and how far they are from the full mesh.
struct Simplified {
    children: Vec<u32>,
    triangles: Vec<[u32; 3]>,
    error: f32,
    bounds: Sphere,
}

/// Builds the clusters of `mesh` at every level of detail. Triangles with two corners at the
/// same position, which cover nothing, are left out.
///
/// Each level is made in two passes: every group simplified - its foliage cards thinned out
/// and grown, the rest collapsed - and then, the grown cards' new vertices added to the mesh,
/// the new triangles split into clusters.
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
    // The mesh's positions and texture coordinates, with the grown cards' corners added on.
    let original_count = mesh.positions.len() as u32;
    let mut positions: Vec<[f32; 3]> = mesh.positions.to_vec();
    let mut uvs: Option<Vec<[f32; 2]>> = mesh.uvs.map(<[_]>::to_vec);
    // The welded vertices of clusters that became roots below the top: whatever is made next
    // to them later has to keep meeting them, so they never move again.
    let mut frozen: HashSet<u32> = HashSet::new();
    let mut level = 0;

    while current.len() > 1 {
        // The mesh as it stands: welded anew, but a vertex keeps its welded index, as new ones
        // only ever come after the old.
        let extended = (!dag.extra_vertices.is_empty()).then(|| {
            let mut m = Mesh::new(&positions, mesh.indices);
            m.uvs = uvs.as_deref();
            m
        });
        let at = extended.as_ref().unwrap_or(mesh);
        let source = |v: u32| {
            if v < original_count {
                v
            } else {
                dag.extra_vertices[(v - original_count) as usize].source
            }
        };

        let described: Vec<(Vec<[u32; 3]>, V3)> = current
            .iter()
            .map(|&c| {
                let cluster = &dag.clusters[c as usize];
                (cluster.mesh_triangles().collect(), V3::from_f32(cluster.bounds.center))
            })
            .collect();
        let views: Vec<(&[[u32; 3]], V3)> =
            described.iter().map(|(t, m)| (t.as_slice(), *m)).collect();
        let groups = group(at, &views, settings.group_size);

        // A welded vertex in more than one group is on a border between groups, which
        // simplifying either may not move.
        let mut owner: Vec<u32> = vec![u32::MAX; at.welded_count()];
        let mut border: HashSet<u32> = frozen.clone();
        for (g, members) in groups.iter().enumerate() {
            for &m in members {
                for t in &described[m].0 {
                    for &v in t {
                        let w = at.weld[v as usize];
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

        // First pass: each group simplified.
        let mut made: Vec<Simplified> = Vec::new();
        let mut new_vertices: Vec<ExtraVertex> = Vec::new();
        for members in groups {
            let children: Vec<u32> = members.iter().map(|&m| current[m]).collect();
            let triangles: Vec<[u32; 3]> = members
                .iter()
                .flat_map(|&m| described[m].0.iter().copied())
                .collect();
            let locked = |w: u32| border.contains(&w);
            // Foliage: its cards thinned out, and whatever else there is collapsed.
            let mut grown: Vec<(u32, [f32; 3])> = Vec::new();
            let (simplified, thinned_error) = match prune::split(at, &triangles, locked) {
                Some(split) => {
                    let first_new = positions.len() as u32 + new_vertices.len() as u32;
                    let thinned = prune::thin(at, &split.cards, first_new, settings.card_error, source);
                    grown = thinned.new_vertices;
                    let mut rest = simplify(at, &split.rest, locked, split.rest.len() / 2);
                    rest.triangles.extend(thinned.triangles);
                    (rest, thinned.error)
                }
                None => (simplify(at, &triangles, locked, triangles.len() / 2), 0.0),
            };
            let reduced = (simplified.triangles.len() as f32)
                <= triangles.len() as f32 * settings.least_reduction;
            if !reduced || simplified.triangles.is_empty() {
                // As simple as it gets: these stay roots.
                for &c in &children {
                    for v in dag.clusters[c as usize].vertices.iter() {
                        frozen.insert(at.weld[*v as usize]);
                    }
                }
                continue;
            }
            new_vertices.extend(grown.into_iter().map(|(source, position)| ExtraVertex { source, position }));

            // How far from the full mesh, at most: as the simplification moved the surface, or
            // as far as the children are and the texture was dragged from them on top, as drag
            // adds up level on level; and as the cards were thinned.
            let mut error = simplified.error.max(thinned_error);
            let mut bounds: Option<Sphere> = None;
            for &c in &children {
                let child = &dag.clusters[c as usize];
                error = error.max(child.error).max(child.error + simplified.dragged);
                bounds = Some(bounds.map_or(child.lod_bounds, |b| b.enclosing(child.lod_bounds)));
            }
            made.push(Simplified {
                children,
                triangles: simplified.triangles,
                error,
                bounds: bounds.expect("a group has clusters"),
            });
        }
        drop(extended);

        // The grown cards' corners added to the mesh, with the texture coordinates of the
        // vertices they copy.
        for extra in &new_vertices {
            positions.push(extra.position);
            if let Some(uvs) = &mut uvs {
                let uv = uvs[extra.source as usize];
                uvs.push(uv);
            }
        }
        dag.extra_vertices.extend(new_vertices);
        let extended = (!dag.extra_vertices.is_empty()).then(|| {
            let mut m = Mesh::new(&positions, mesh.indices);
            m.uvs = uvs.as_deref();
            m
        });
        let at = extended.as_ref().unwrap_or(mesh);

        // Second pass: the new triangles into clusters.
        let mut next = Vec::new();
        for Simplified { children, triangles, error, bounds } in made {
            let g = dag.groups.len() as u32;
            let parents = add_clusters(&mut dag, at, &triangles, level + 1, Some(g));
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
