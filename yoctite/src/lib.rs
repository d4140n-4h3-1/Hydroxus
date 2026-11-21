//! Yoctite: virtualized, cluster-based geometry for Hydroxus.
//!
//! A mesh is split into clusters of up to 128 triangles. Neighbouring clusters are grouped,
//! each group simplified to about half its triangles - its border with the rest of the mesh left
//! as it is - and split into clusters again; and so on, level after level, down to a few
//! clusters for the whole mesh. What is built is a directed acyclic graph of clusters: each group
//! of clusters at one level can be drawn in place of the coarser clusters made from it, or they
//! in its place.
//!
//! Every cluster carries two measures of how far its surface is from the full mesh: its own
//! error, judged over its bounds, and its parent group's, over theirs (see [`Cluster`]). Both
//! only ever grow up the graph, so for any camera there is exactly one set of clusters whose own
//! error looks small enough from there and whose parent's does not ([`select`]): drawn together,
//! they cover the mesh once, with no gaps between clusters of different levels, since a group's
//! border never moves. That set is what is drawn - small, distant, or simple things in a handful
//! of coarse clusters, what is near in full detail - and the GPU picks it per cluster, every
//! frame (the next phase).
//!
//! This is the builder: pure Rust, no engine, run once per mesh and cached ([`ClusterDag::to_bytes`]).
//! The method follows Karis et al., "A Deep Dive into Nanite Virtualized Geometry" (SIGGRAPH
//! 2021).

mod cluster;
mod dag;
mod format;
mod group;
mod math;
mod prune;
mod select;
mod simplify;
#[cfg(test)]
mod tests;

pub use cluster::{MAX_TRIANGLES, MAX_VERTICES};
pub use dag::build;
pub use format::FormatError;
pub use math::Sphere;
pub use select::{projected_error, select};

use math::V3;
use std::collections::HashMap;

/// A mesh to build the clusters of: positions, and every three indices a triangle.
pub struct Mesh<'a> {
    pub positions: &'a [[f32; 3]],
    pub indices: &'a [u32],
    /// Each vertex's texture coordinates, if it has them: then how far simplifying moves the
    /// texture over the surface counts as error too (see [`Self::with_texture_coordinates`]).
    pub uvs: Option<&'a [[f32; 2]]>,
    /// Which vertices are at the same position: the same welded index for all of them.
    weld: Vec<u32>,
    welded_count: usize,
}

impl<'a> Mesh<'a> {
    /// Vertices at exactly the same position are taken as one where clusters meet and where the
    /// surface simplifies, though each keeps its own normal and texture coordinates.
    pub fn new(positions: &'a [[f32; 3]], indices: &'a [u32]) -> Self {
        let mut seen: HashMap<[u32; 3], u32> = HashMap::with_capacity(positions.len());
        // -0.0 and 0.0 are the same place.
        let key = |p: &[f32; 3]| p.map(|v| if v == 0.0 { 0 } else { v.to_bits() });
        let weld: Vec<u32> = positions
            .iter()
            .map(|p| {
                let next = seen.len() as u32;
                *seen.entry(key(p)).or_insert(next)
            })
            .collect();
        Self {
            positions,
            indices,
            uvs: None,
            weld,
            welded_count: seen.len(),
        }
    }

    /// The mesh with texture coordinates, one for each vertex. Simplifying a flat surface - a
    /// leaf card, a plank - moves none of it, but can drag its texture across it; with these,
    /// how far the texture is dragged, in the surface's own units, counts as error, so a level
    /// of detail that smears a texture is only drawn from where the smear is too small to see.
    /// Where the texture is laid flat across a plane, as on ground, dragging it moves nothing,
    /// and nothing is counted.
    pub fn with_texture_coordinates(mut self, uvs: &'a [[f32; 2]]) -> Self {
        if uvs.len() == self.positions.len() {
            self.uvs = Some(uvs);
        }
        self
    }

    pub(crate) fn position(&self, v: u32) -> V3 {
        V3::from_f32(self.positions[v as usize])
    }

    pub(crate) fn position_f32(&self, v: u32) -> [f32; 3] {
        self.positions[v as usize]
    }

    pub(crate) fn welded_count(&self) -> usize {
        self.welded_count
    }
}

/// Settings for [`build`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Settings {
    /// How many clusters are simplified together, at most.
    pub group_size: usize,
    /// A group simplified to more than this share of its triangles has gone as far as it can:
    /// its clusters are roots, drawn however far off they are.
    pub least_reduction: f32,
    /// How much of a foliage card's size counts as error when half the cards are left out and
    /// the rest grown in their place (see the `prune` module): 1 counts a card moved as far as
    /// it reaches, which keeps every card until it is smaller than a pixel; less lets the crown
    /// thin out from where its cards are a few pixels across, keeping how much it covers.
    pub card_error: f32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            group_size: 4,
            least_reduction: 0.85,
            card_error: 0.25,
        }
    }
}

/// A vertex made by simplifying: a copy of one of the mesh's own vertices - its normal,
/// texture coordinates and the rest - at a new position. Foliage cards grown in place of those
/// left out have corners of these.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExtraVertex {
    /// The mesh's vertex it is a copy of.
    pub source: u32,
    pub position: [f32; 3],
}

/// A cluster: up to [`MAX_TRIANGLES`] triangles of the mesh at some level of detail.
#[derive(Debug, Clone, PartialEq)]
pub struct Cluster {
    /// The vertices it uses, at most [`MAX_VERTICES`]: the mesh's own, and from the mesh's
    /// vertex count on, the graph's [`ClusterDag::extra_vertices`].
    pub vertices: Vec<u32>,
    /// Its triangles, as indices into [`Self::vertices`].
    pub triangles: Vec<[u8; 3]>,
    /// Bounds round its triangles, for culling.
    pub bounds: Sphere,
    /// How far, at most, its surface is from the full mesh's (0 for the full mesh's own
    /// clusters), and the bounds that error is judged over: those of the group it was made from,
    /// which take in every cluster that error stands for.
    pub error: f32,
    pub lod_bounds: Sphere,
    /// The same for the group it is simplified into, coarser than it: infinite for a root, which
    /// nothing coarser replaces.
    pub parent_error: f32,
    pub parent_bounds: Sphere,
    /// How many times simplified: 0 for the full mesh.
    pub level: u32,
    /// The group it was made by simplifying, or [`None`] at level 0.
    pub made_from: Option<u32>,
    /// The group it is simplified into, or [`None`] for a root.
    pub simplified_in: Option<u32>,
}

impl Cluster {
    /// Its triangles as indices into the mesh's vertices.
    pub fn mesh_triangles(&self) -> impl Iterator<Item = [u32; 3]> + '_ {
        self.triangles
            .iter()
            .map(|t| t.map(|i| self.vertices[i as usize]))
    }

    /// Whether nothing coarser replaces it.
    pub fn is_root(&self) -> bool {
        self.simplified_in.is_none()
    }
}

/// Clusters simplified together into coarser ones.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    /// The clusters simplified, and the clusters they were simplified into.
    pub children: Vec<u32>,
    pub parents: Vec<u32>,
    /// How far the parents' surface is from the full mesh, at most, and bounds round the
    /// children's: the parents' error and bounds, and the children's parent error and bounds.
    pub error: f32,
    pub bounds: Sphere,
    /// The children's level.
    pub level: u32,
}

/// A mesh's clusters at every level of detail, and how they were made from each other.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct ClusterDag {
    pub clusters: Vec<Cluster>,
    pub groups: Vec<Group>,
    /// Vertices its coarser clusters use besides the mesh's own, numbered on from the mesh's
    /// vertex count: see [`ExtraVertex`] and [`Self::extend_vertices`].
    pub extra_vertices: Vec<ExtraVertex>,
}

impl ClusterDag {
    /// Adds the graph's [`Self::extra_vertices`] to the mesh's own `vertices`, of whatever kind a
    /// renderer keeps: each a clone of its source, put at its position by `place`. The clusters
    /// index the vertices so extended.
    pub fn extend_vertices<T: Clone>(&self, vertices: &mut Vec<T>, mut place: impl FnMut(&mut T, [f32; 3])) {
        vertices.reserve(self.extra_vertices.len());
        for extra in &self.extra_vertices {
            let mut vertex = vertices[extra.source as usize].clone();
            place(&mut vertex, extra.position);
            vertices.push(vertex);
        }
    }

    /// How many levels of detail it has.
    pub fn levels(&self) -> u32 {
        self.clusters.iter().map(|c| c.level + 1).max().unwrap_or(0)
    }

    /// How many triangles its clusters at `level` have between them.
    pub fn triangles_at(&self, level: u32) -> usize {
        self.clusters
            .iter()
            .filter(|c| c.level == level)
            .map(|c| c.triangles.len())
            .sum()
    }
}
