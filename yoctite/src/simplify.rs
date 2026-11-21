//! Simplifying a group of triangles to fewer, by collapsing edges in the order that moves the
//! surface least (Garland and Heckbert's quadric error metric), leaving its border alone.
//!
//! Every collapse moves one vertex onto a neighbour (a half-edge collapse), so what is left is
//! made of the mesh's own vertices, with their own normals and texture coordinates: a coarser
//! level of detail needs no vertices of its own. Collapses work on vertices welded by position,
//! so seams do not hold the surface together where it should simplify; each corner of a
//! triangle keeps the vertex on its own side of a seam.
//!
//! A seam is wherever vertices at one place differ - in texture coordinates, or in normals, as
//! on every corner of a flat-shaded box. A collapse that keeps every seam whole moves the
//! surface only by what its quadrics measure. One that cannot - a corner whose side of a seam
//! has no vertex to become where it goes - is still made, if nothing cheaper is left: that
//! corner takes the vertex of the side facing most nearly its own way, and the collapse counts
//! as moving the surface by the whole length of the edge, so the levels of detail with it are
//! only drawn from where the edge is too small to see whatever its texture does.

use crate::{
    math::{Quadric, V3},
    Mesh,
};
use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap},
};

/// How much more an open border's edges weigh than a surface of the same size, holding the
/// outline of a card or a hole where it is.
const BORDER_WEIGHT: f64 = 8.0;
/// How many rings of triangles round where a vertex went are searched for the one covering its
/// place, to measure how far the texture was dragged there.
const MAX_DRAG_RINGS: usize = 4;
/// The least cosine of the angle between a triangle and a vertex's surface for the triangle to
/// stand for the vertex, measuring texture drag.
const MIN_SAME_FACING: f64 = 0.5;
/// The least cosine of the angle a triangle may be turned through by a collapse: it may tilt,
/// but not fold over.
const MIN_TURN_COSINE: f64 = 0.2;

/// The triangles left, and how far the surface was moved: the root of the largest mean squared
/// distance any collapse moved it.
pub(crate) struct Simplified {
    pub triangles: Vec<[u32; 3]>,
    pub error: f32,
    /// How far the texture was dragged across the surface from the full mesh, measured afterwards
    /// at the drag points given (see [`DragPoint`]): at each, by the triangle now covering it.
    pub dragged: f32,
    /// Where each drag point is now: a vertex at the place of the one it was moved onto.
    pub moved: Vec<u32>,
}

/// How a triangle covers a full-mesh vertex: see [`cover`].
#[derive(Debug, Clone, Copy, PartialEq)]
struct Cover {
    /// How far from the vertex the triangle comes nearest.
    pub distance: f64,
    /// How far the texture there is dragged from the vertex's own, and the texture coordinate
    /// there.
    pub drag: f64,
    pub there: [f64; 2],
    /// Whether the vertex's place falls inside the triangle, seen square on; and whether well
    /// inside, not on an edge or corner.
    pub inside: bool,
    pub within: bool,
}

/// How triangle `p`, with texture coordinates `u`, stands for a full-mesh vertex at `at` with
/// texture coordinate `own` and its surface facing `faced`. [`None`] for a triangle facing
/// another way, or with no area.
fn cover(at: V3, own: [f64; 2], faced: V3, p: [V3; 3], u: [[f64; 2]; 3]) -> Option<Cover> {
    let map = TextureMap::of(p, u)?;
    let w = barycentric(at, p)?;
    // Only a triangle facing much as the vertex's surface did stands for it: the cap at the foot
    // of a trunk, facing down, has a texture of its own, which the side of the trunk just above
    // it was never drawn with.
    let n = (p[1] - p[0]).cross(p[2] - p[0]);
    if n.dot(faced) < MIN_SAME_FACING * n.length() * faced.length() {
        return None;
    }
    let inside = w.iter().all(|&w| w >= -1e-6);
    let within = w.iter().all(|&w| w > 1e-4);
    let clamped = w.map(|w| w.max(0.0));
    let sum: f64 = clamped.iter().sum();
    let clamped = clamped.map(|w| w / sum.max(1e-30));
    let closest = p[0] * clamped[0] + p[1] * clamped[1] + p[2] * clamped[2];
    let distance = (closest - at).length();
    // The texture where the triangle comes nearest: off a triangle its mapping carried on past
    // its edges means nothing, and from a sliver it runs off without end.
    let there = [0, 1].map(|i| clamped[0] * u[0][i] + clamped[1] * u[1][i] + clamped[2] * u[2][i]);
    Some(Cover {
        distance,
        drag: map.drag(there, own),
        there,
        inside,
        within,
    })
}

/// How far the texture is dragged at a full-mesh vertex, of the ways `found` triangles cover it:
/// those it falls inside, nearest first. Where it is well inside one or more, or on edges the
/// triangles there agree on the texture across, the most they drag it: two triangles over one
/// place both show, and on a shared edge each carries the same slip over its own stretch of
/// surface. Where it is on edges they disagree on - a seam, where two sides of the texture meet -
/// the least: the side it was on. [`None`] if nothing covers it.
fn drag_of(found: &[Cover]) -> Option<f64> {
    let nearest = found.iter().filter(|f| f.inside).map(|f| f.distance).fold(f64::MAX, f64::min);
    let tolerance = nearest * 1e-3 + 1e-6;
    let near = || found.iter().filter(move |f| f.inside && f.distance <= nearest + tolerance);
    let first = near().next()?.there;
    let agree = near().all(|f| (f.there[0] - first[0]).abs() + (f.there[1] - first[1]).abs() < 1e-5);
    if agree || near().any(|f| f.within) {
        near().filter(|f| agree || f.within).map(|f| f.drag).max_by(f64::total_cmp)
    } else {
        near().map(|f| f.drag).min_by(f64::total_cmp)
    }
}

/// A vertex of the full mesh, to measure texture drag at: as it was - its place, texture
/// coordinate and the way its surface faced - against whatever level of detail now stands for
/// it. Measured against the full mesh at every level, drag never has to be added up level on
/// level, which overstates it more each time.
#[derive(Debug, Clone, Copy)]
pub(crate) struct DragPoint {
    /// The full mesh's vertex.
    pub original: u32,
    /// A vertex of the triangles being simplified at the place it has come to.
    pub at: u32,
    /// The way the full mesh's surface faced there, not of unit length.
    pub facing: V3,
}


/// How near a whole number of repeats two texture coordinates have to be apart to be taken for
/// the two sides of a seam where a repeating texture meets itself (see [`TextureMap::drag`]).
const SEAM_SLACK: f64 = 0.25;

/// How a triangle's texture lies across it: how far over the surface the texture goes for one
/// repeat of it, the same both ways - the root of the triangle's area over its area in the
/// texture. One scale for both directions is rough on a stretched texture, but it stays bounded:
/// taken apart by direction, a coarse triangle whose texture coordinates nearly line up would
/// make a slip of the texture across it a slide of any length.
struct TextureMap {
    scale: f64,
}

impl TextureMap {
    /// The map of a triangle with corners `p` and texture coordinates `u`; [`None`] if either has
    /// no area.
    fn of(p: [V3; 3], u: [[f64; 2]; 3]) -> Option<Self> {
        let area = (p[1] - p[0]).cross(p[2] - p[0]).length();
        let texture_area =
            ((u[1][0] - u[0][0]) * (u[2][1] - u[0][1]) - (u[2][0] - u[0][0]) * (u[1][1] - u[0][1])).abs();
        (area > 0.0 && texture_area > 1e-14).then(|| Self {
            scale: (area / texture_area).sqrt(),
        })
    }

    /// How far over the surface the texture at `a` is from where it is at `b`. On a repeating
    /// texture a whole number of repeats apart is the same place: bark wrapped round a branch
    /// meets itself at a seam where one side reads 0 and the other 1. Only near a whole number,
    /// though - within [`SEAM_SLACK`] - is that taken for a seam; further off, a drag of more
    /// than half a repeat is just that, not a smaller one the other way.
    fn drag(&self, a: [f64; 2], b: [f64; 2]) -> f64 {
        let d = [0, 1].map(|i| {
            let d = a[i] - b[i];
            let whole = d.round();
            if whole != 0.0 && (d - whole).abs() <= SEAM_SLACK {
                d - whole
            } else {
                d
            }
        });
        (d[0] * d[0] + d[1] * d[1]).sqrt() * self.scale
    }
}

#[derive(Clone, Copy, PartialEq)]
struct Cost(f64);

impl Eq for Cost {}

impl PartialOrd for Cost {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Cost {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

struct Vertex {
    position: V3,
    quadric: Quadric,
    /// How far, at most, the texture has been dragged across the surface by the collapses onto
    /// it so far, in the surface's units.
    dragged: f64,
    locked: bool,
    alive: bool,
    /// One of the mesh's vertices at its place.
    vertex: u32,
    /// The vertex it was moved onto, once collapsed.
    moved_onto: Option<usize>,
    /// Its triangles, some perhaps no longer alive.
    triangles: Vec<usize>,
}

struct Triangle {
    /// Its corners as welded vertices of this group.
    corners: [usize; 3],
    /// Its corners as the mesh's own vertices.
    originals: [u32; 3],
    alive: bool,
}

struct Simplifier<'a> {
    vertices: Vec<Vertex>,
    triangles: Vec<Triangle>,
    alive: usize,
    /// The mesh's texture coordinates, by its own vertices, if it has them.
    uvs: Option<&'a [[f32; 2]]>,
}

/// Where `p` is in the plane of triangle `t`, as weights of its corners: all of them at least
/// zero inside it. [`None`] for a triangle with no area.
fn barycentric(p: V3, t: [V3; 3]) -> Option<[f64; 3]> {
    let (e0, e1, e2) = (t[1] - t[0], t[2] - t[0], p - t[0]);
    let (d00, d01, d11) = (e0.dot(e0), e0.dot(e1), e1.dot(e1));
    let (d20, d21) = (e2.dot(e0), e2.dot(e1));
    let denominator = d00 * d11 - d01 * d01;
    if denominator.abs() < 1e-30 {
        return None;
    }
    let b = (d11 * d20 - d01 * d21) / denominator;
    let c = (d00 * d21 - d01 * d20) / denominator;
    Some([1.0 - b - c, b, c])
}

impl Simplifier<'_> {
    /// How far the simplified triangles drag the texture across the surface from the full mesh,
    /// at most, in the surface's units, at `points`: at each, the texture coordinate the
    /// triangle now covering its place gives there, against its own. Each point is judged by the
    /// triangles round the vertex it was moved onto, and their neighbours - where it went - not
    /// by whatever triangle happens to be nearest: on a tree that is as likely the other side of
    /// a twig, or another branch, whose texture has nothing to do with it. On a seam a point is
    /// covered from both sides, so of the triangles nearest it, the one that drags it least is
    /// taken. Without texture coordinates, nothing. `local` is each welded vertex's index here.
    /// With where each point is now.
    /// A point nothing covers any more is where the surface was taken away - the tip of a twig
    /// collapsed back along it - which is how far the surface moved, not how far its texture slid
    /// over it: that is the quadrics' to measure.
    fn measure_drag(&self, mesh: &Mesh, points: &[DragPoint], local: &HashMap<u32, usize>) -> (f64, Vec<u32>) {
        let mut moved = Vec::with_capacity(points.len());
        let mut worst: f64 = 0.0;
        let (mut candidates, mut ring, mut seen) = (Vec::new(), Vec::new(), Vec::new());
        for point in points {
            let Some(&start) = local.get(&mesh.weld[point.at as usize]) else {
                moved.push(point.at);
                continue;
            };
            let mut at_now = start;
            while let Some(next) = self.vertices[at_now].moved_onto {
                at_now = next;
            }
            moved.push(self.vertices[at_now].vertex);
            let Some(uvs) = self.uvs else {
                continue;
            };
            let uv = |v: u32| uvs[v as usize].map(f64::from);
            let (v, faced) = (point.original, point.facing);
            let (at, own) = (mesh.position(v), uv(v));
            // Ring after ring of triangles round where it went, until one covers it - its
            // place falls inside the triangle, seen square on.
            ring.clear();
            ring.push(at_now);
            seen.clear();
            seen.push(at_now);
            candidates.clear();
            let mut found: Vec<Cover> = Vec::new();
            for _ in 0..MAX_DRAG_RINGS {
                let start = candidates.len();
                for &r in &ring {
                    candidates.extend(self.live(r).filter(|t| !candidates.contains(t)).collect::<Vec<_>>());
                }
                for &t in &candidates[start..] {
                    let tri = &self.triangles[t];
                    let p = tri.corners.map(|c| self.vertices[c].position);
                    if let Some(f) = cover(at, own, faced, p, tri.originals.map(uv)) {
                        found.push(f);
                    }
                }
                if found.iter().any(|f| f.inside) {
                    break;
                }
                let next: Vec<usize> = ring
                    .iter()
                    .flat_map(|&r| self.neighbours(r))
                    .filter(|n| !seen.contains(n))
                    .collect();
                ring.clear();
                for n in next {
                    if !seen.contains(&n) {
                        seen.push(n);
                        ring.push(n);
                    }
                }
                if ring.is_empty() {
                    break;
                }
            }
            // Judged by those covering it (see `drag_of`); nothing covering it, it is left out.
            if let Some(drag) = drag_of(&found) {
                worst = worst.max(drag);
            }
        }
        (worst, moved)
    }

    /// How far collapsing `from` onto `to` moves the surface, squared: by its quadrics, by how far
    /// it drags the texture, or by the edge's length if it breaks a seam.
    fn cost(&self, from: usize, to: usize) -> f64 {
        let q = self.vertices[from].quadric + self.vertices[to].quadric;
        let moved = q.error(self.vertices[to].position) / q.weight.max(1e-30);
        let dragged = self.dragged(from, to);
        let moved = moved.max(dragged * dragged);
        if self.keeps_seams(from, to) {
            moved
        } else {
            let edge = self.vertices[to].position - self.vertices[from].position;
            moved.max(edge.dot(edge))
        }
    }

    /// How far collapsing `from` onto `to` drags the texture across the surface, added to how far
    /// collapses onto `from` dragged it already: in the surface's units. Where `from` was, the
    /// triangle that covers it once moved gives some other texture coordinate than `from`'s own;
    /// that difference, in the surface's units by the triangle's own scale between texture and
    /// surface, is how far the texture has been dragged there. Each side of a seam is judged by
    /// its own triangles. Without texture coordinates, nothing.
    fn dragged(&self, from: usize, to: usize) -> f64 {
        let Some(uvs) = self.uvs else {
            return 0.0;
        };
        let uv = |v: u32| uvs[v as usize].map(f64::from);
        let sides = self.sides(from, to);
        let Some(&(_, fallback)) = sides.first() else {
            return self.vertices[from].dragged;
        };
        let at = self.vertices[from].position;
        // For each of `from`'s own vertices - each side of a seam - the triangle covering where
        // `from` was, the most nearly: whose least weight there is greatest; and how far it drags.
        let mut covering: Vec<(u32, f64, f64)> = Vec::new();
        for t in self.live(from) {
            let tri = &self.triangles[t];
            if tri.corners.contains(&to) {
                continue;
            }
            let k = tri.corners.iter().position(|&c| c == from).unwrap();
            let own = tri.originals[k];
            let becomes = sides.iter().find(|(o, _)| *o == own).map_or(fallback, |(_, b)| *b);
            let positions = tri
                .corners
                .map(|c| if c == from { self.vertices[to].position } else { self.vertices[c].position });
            let mut texture = tri.originals.map(uv);
            texture[k] = uv(becomes);
            let Some(w) = barycentric(at, positions) else {
                continue;
            };
            let inside = w[0].min(w[1]).min(w[2]);
            // The triangle's own mapping from the surface to the texture, carried on to where
            // `from` was: exact for a texture laid linearly across a plane.
            let there = [0, 1].map(|i| w[0] * texture[0][i] + w[1] * texture[1][i] + w[2] * texture[2][i]);
            let should = uv(own);
            let Some(map) = TextureMap::of(positions, texture) else {
                continue;
            };
            let off = map.drag(there, should);
            match covering.iter_mut().find(|(o, ..)| *o == own) {
                Some(best) if best.1 >= inside => {}
                Some(best) => *best = (own, inside, off),
                None => covering.push((own, inside, off)),
            }
        }
        let worst = covering.iter().map(|&(_, _, off)| off).fold(0.0, f64::max);
        self.vertices[from].dragged + worst
    }

    /// The way triangle `t` faces, not of unit length.
    fn facing(&self, t: usize) -> V3 {
        let p = self.triangles[t].corners.map(|c| self.vertices[c].position);
        (p[1] - p[0]).cross(p[2] - p[0])
    }

    fn push(&self, heap: &mut BinaryHeap<Reverse<(Cost, usize, usize)>>, from: usize, to: usize) {
        if !self.vertices[from].locked {
            heap.push(Reverse((Cost(self.cost(from, to)), from, to)));
        }
    }

    fn live(&self, v: usize) -> impl Iterator<Item = usize> + '_ {
        self.vertices[v]
            .triangles
            .iter()
            .copied()
            .filter(|&t| self.triangles[t].alive)
    }

    fn neighbours(&self, v: usize) -> Vec<usize> {
        let mut out: Vec<usize> = self
            .live(v)
            .flat_map(|t| self.triangles[t].corners)
            .filter(|&w| w != v)
            .collect();
        out.sort_unstable();
        out.dedup();
        out
    }

    /// Whether the surface stays a surface: the only vertices next to both ends of the edge are
    /// those across the triangles on it. Otherwise collapsing it would pinch the surface.
    fn link_holds(&self, a: usize, b: usize) -> bool {
        let across: Vec<usize> = self
            .live(a)
            .filter(|&t| self.triangles[t].corners.contains(&b))
            .flat_map(|t| self.triangles[t].corners)
            .filter(|&w| w != a && w != b)
            .collect();
        let of_b = self.neighbours(b);
        self.neighbours(a)
            .into_iter()
            .filter(|w| of_b.binary_search(w).is_ok())
            .all(|w| across.contains(&w))
    }

    /// Whether moving `a` onto `b` would turn any of `a`'s other triangles over, or flatten one
    /// to nothing.
    fn flips(&self, a: usize, b: usize) -> bool {
        let to = self.vertices[b].position;
        self.live(a).any(|t| {
            let corners = self.triangles[t].corners;
            if corners.contains(&b) {
                return false;
            }
            let p = corners.map(|c| self.vertices[c].position);
            let moved = corners.map(|c| if c == a { to } else { self.vertices[c].position });
            let before = (p[1] - p[0]).cross(p[2] - p[0]);
            let after = (moved[1] - moved[0]).cross(moved[2] - moved[0]);
            let (lb, la) = (before.length(), after.length());
            if la <= lb * 1e-6 || la == 0.0 {
                return true;
            }
            before.dot(after) < MIN_TURN_COSINE * lb * la
        })
    }

    /// Whether moving `a` onto `b` would change how many triangles use any edge between two
    /// locked vertices: the group's border with the rest of the mesh, which has to stay as it is
    /// for the clusters on either side of it to meet.
    fn moves_border(&self, a: usize, b: usize) -> bool {
        let star: Vec<usize> = {
            let mut s: Vec<usize> = self.live(a).chain(self.live(b)).collect();
            s.sort_unstable();
            s.dedup();
            s
        };
        let locked = |v: usize| self.vertices[v].locked;
        let mut counts: HashMap<(usize, usize), i32> = HashMap::new();
        let mut count = |corners: [usize; 3], by: i32| {
            for k in 0..3 {
                let (x, y) = (corners[k], corners[(k + 1) % 3]);
                if locked(x) && locked(y) {
                    *counts.entry((x.min(y), x.max(y))).or_default() += by;
                }
            }
        };
        for &t in &star {
            let corners = self.triangles[t].corners;
            count(corners, -1);
            if !(corners.contains(&a) && corners.contains(&b)) {
                count(corners.map(|c| if c == a { b } else { c }), 1);
            }
        }
        counts.values().any(|&c| c != 0)
    }

    /// What each of `a`'s own vertices becomes when `a` moves onto `b`: `b`'s on the same side of
    /// any seam in texture coordinates, as a triangle on the edge between them pairs them.
    fn sides(&self, a: usize, b: usize) -> Vec<(u32, u32)> {
        let mut sides: Vec<(u32, u32)> = Vec::new();
        for t in self.live(a) {
            let tri = &self.triangles[t];
            if let (Some(ka), Some(kb)) = (
                tri.corners.iter().position(|&c| c == a),
                tri.corners.iter().position(|&c| c == b),
            ) {
                if !sides.iter().any(|(from, _)| *from == tri.originals[ka]) {
                    sides.push((tri.originals[ka], tri.originals[kb]));
                }
            }
        }
        sides
    }

    /// Whether every one of `a`'s own vertices has one of `b`'s to become, on its side of any
    /// seam. Off a seam `a` has one, which the edge pairs; on one, it can only move along the
    /// seam, onto a vertex that is on both sides of it too. Otherwise the triangles on one side
    /// take a corner from another (see the module's description).
    fn keeps_seams(&self, a: usize, b: usize) -> bool {
        let sides = self.sides(a, b);
        self.live(a).all(|t| {
            let tri = &self.triangles[t];
            let ka = tri.corners.iter().position(|&c| c == a).unwrap();
            sides.iter().any(|(from, _)| *from == tri.originals[ka])
        })
    }

    /// Moves `a` onto `b`.
    fn collapse(&mut self, a: usize, b: usize) {
        let dragged = self.dragged(a, b);
        self.vertices[b].dragged = self.vertices[b].dragged.max(dragged);
        let sides = self.sides(a, b);
        // For a corner with no vertex of its own side to become: `b`'s vertex in each of `b`'s
        // triangles, and the way that triangle faces.
        let of_b: Vec<(V3, u32)> = self
            .live(b)
            .map(|t| {
                let tri = &self.triangles[t];
                let kb = tri.corners.iter().position(|&c| c == b).unwrap();
                (self.facing(t).normalized(), tri.originals[kb])
            })
            .collect();
        let triangles: Vec<usize> = self.live(a).collect();
        for t in triangles {
            let facing = self.facing(t).normalized();
            let tri = &mut self.triangles[t];
            if tri.corners.contains(&b) {
                tri.alive = false;
                self.alive -= 1;
                continue;
            }
            let k = tri.corners.iter().position(|&c| c == a).unwrap();
            let original = tri.originals[k];
            tri.corners[k] = b;
            tri.originals[k] = sides
                .iter()
                .find(|(from, _)| *from == original)
                .map(|(_, to)| *to)
                .unwrap_or_else(|| {
                    of_b.iter()
                        .max_by(|x, y| x.0.dot(facing).total_cmp(&y.0.dot(facing)))
                        .expect("an edge has a triangle")
                        .1
                });
            self.vertices[b].triangles.push(t);
        }
        let quadric = self.vertices[a].quadric;
        self.vertices[b].quadric += quadric;
        self.vertices[a].alive = false;
        self.vertices[a].moved_onto = Some(b);
        self.vertices[a].triangles.clear();
        let triangles = &self.triangles;
        self.vertices[b].triangles.retain(|&t| triangles[t].alive);
    }
}

/// Simplifies `triangles` (indices into `mesh`'s vertices) to `target` triangles or as near as
/// it can, never moving a vertex `locked` says to keep (given its welded index); and measures
/// how far that drags the texture from the full mesh at `points`.
pub(crate) fn simplify(
    mesh: &Mesh,
    triangles: &[[u32; 3]],
    locked: impl Fn(u32) -> bool,
    target: usize,
    points: &[DragPoint],
) -> Simplified {
    let mut local: HashMap<u32, usize> = HashMap::new();
    let mut s = Simplifier {
        vertices: Vec::new(),
        triangles: Vec::with_capacity(triangles.len()),
        alive: 0,
        uvs: mesh.uvs,
    };
    for originals in triangles {
        let corners = originals.map(|v| {
            let w = mesh.weld[v as usize];
            *local.entry(w).or_insert_with(|| {
                s.vertices.push(Vertex {
                    position: mesh.position(v),
                    quadric: Quadric::default(),
                    dragged: 0.0,
                    locked: locked(w),
                    alive: true,
                    vertex: v,
                    moved_onto: None,
                    triangles: Vec::new(),
                });
                s.vertices.len() - 1
            })
        });
        if corners[0] == corners[1] || corners[1] == corners[2] || corners[0] == corners[2] {
            continue;
        }
        let t = s.triangles.len();
        for &c in &corners {
            s.vertices[c].triangles.push(t);
        }
        s.triangles.push(Triangle {
            corners,
            originals: *originals,
            alive: true,
        });
    }
    s.alive = s.triangles.len();

    // Each vertex's quadric: the planes of its triangles, weighted by their areas, and of the
    // borders through it, at right angles to their triangles. A border is an edge of one
    // triangle, or the fold of a sheet made double-sided by a reversed copy of each triangle:
    // an edge of just two triangles facing opposite ways. Without its plane a doubled card would
    // seem a closed, flat surface, which collapses away to nothing at no cost at all.
    let mut edges: HashMap<(usize, usize), (u32, usize, usize)> = HashMap::new();
    for (t, tri) in s.triangles.iter().enumerate() {
        let p = tri.corners.map(|c| s.vertices[c].position);
        let normal = (p[1] - p[0]).cross(p[2] - p[0]);
        let area = normal.length() * 0.5;
        if area > 0.0 {
            let q = Quadric::plane(normal.normalized(), p[0], area);
            for &c in &tri.corners {
                s.vertices[c].quadric += q;
            }
        }
        for k in 0..3 {
            let (x, y) = (tri.corners[k], tri.corners[(k + 1) % 3]);
            let entry = edges.entry((x.min(y), x.max(y))).or_insert((0, t, t));
            entry.0 += 1;
            entry.2 = t;
        }
    }
    let mut borders = Vec::new();
    for (&(x, y), &(uses, t, other)) in &edges {
        let normal = s.facing(t).normalized();
        let folded = uses == 2 && normal.dot(s.facing(other).normalized()) < -0.99;
        if uses != 1 && !folded {
            continue;
        }
        let (px, py) = (s.vertices[x].position, s.vertices[y].position);
        let edge = py - px;
        let across = edge.cross(normal).normalized();
        borders.push((x, y, Quadric::plane(across, px, edge.dot(edge) * BORDER_WEIGHT)));
    }
    for (x, y, q) in borders {
        s.vertices[x].quadric += q;
        s.vertices[y].quadric += q;
    }

    let mut heap = BinaryHeap::new();
    for &(x, y) in edges.keys() {
        s.push(&mut heap, x, y);
        s.push(&mut heap, y, x);
    }
    let mut worst: f64 = 0.0;
    while s.alive > target {
        let Some(Reverse((Cost(cost), a, b))) = heap.pop() else {
            break;
        };
        if !s.vertices[a].alive || !s.vertices[b].alive {
            continue;
        }
        if !s.live(a).any(|t| s.triangles[t].corners.contains(&b)) {
            continue;
        }
        // Its ends have changed since it was put in: in again at what it costs now.
        let now = s.cost(a, b);
        if now > cost * (1.0 + 1e-9) + 1e-30 {
            heap.push(Reverse((Cost(now), a, b)));
            continue;
        }
        if !s.link_holds(a, b) || s.flips(a, b) || s.moves_border(a, b) {
            continue;
        }
        s.collapse(a, b);
        worst = worst.max(now);
        for n in s.neighbours(b) {
            s.push(&mut heap, b, n);
            s.push(&mut heap, n, b);
        }
    }

    let simplified: Vec<[u32; 3]> = s.triangles.iter().filter(|t| t.alive).map(|t| t.originals).collect();
    let (dragged, moved) = s.measure_drag(mesh, points, &local);
    Simplified {
        triangles: simplified,
        error: worst.sqrt() as f32,
        dragged: dragged as f32,
        moved,
    }
}
