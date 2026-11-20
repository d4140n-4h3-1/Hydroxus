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
/// The least cosine of the angle a triangle may be turned through by a collapse: it may tilt,
/// but not fold over.
const MIN_TURN_COSINE: f64 = 0.2;

/// The triangles left, and how far the surface was moved: the root of the largest mean squared
/// distance any collapse moved it.
pub(crate) struct Simplified {
    pub triangles: Vec<[u32; 3]>,
    pub error: f32,
    /// How far the texture was dragged across the surface, measured afterwards against the
    /// triangles simplified: at every vertex they had, by the triangle now covering it. Relative
    /// to them, not to the full mesh: drag adds up, level on level.
    pub dragged: f32,
}

/// How far `after` drags the texture across the surface from `before`, at most, in the surface's
/// units: at each vertex of `before`, the texture coordinate the triangle of `after` covering it
/// gives there, against its own. On a seam a vertex is covered from both sides, so of the
/// triangles nearest it, the one that drags it least is taken. Without texture coordinates,
/// nothing.
fn measure_drag(mesh: &Mesh, before: &[[u32; 3]], after: &[[u32; 3]]) -> f64 {
    let Some(uvs) = mesh.uvs else {
        return 0.0;
    };
    let uv = |v: u32| uvs[v as usize].map(f64::from);
    // Each triangle of `after`: its corners, their texture coordinates, and its scale from
    // texture to surface; those with no area in either left out.
    let covers: Vec<([V3; 3], [[f64; 2]; 3], f64)> = after
        .iter()
        .filter_map(|t| {
            let p = t.map(|v| mesh.position(v));
            let u = t.map(uv);
            let area = (p[1] - p[0]).cross(p[2] - p[0]).length();
            let texture_area =
                ((u[1][0] - u[0][0]) * (u[2][1] - u[0][1]) - (u[2][0] - u[0][0]) * (u[1][1] - u[0][1])).abs();
            (area > 0.0 && texture_area > 1e-14).then(|| (p, u, (area / texture_area).sqrt()))
        })
        .collect();
    let mut vertices: Vec<u32> = before.iter().flatten().copied().collect();
    vertices.sort_unstable();
    vertices.dedup();
    let mut worst: f64 = 0.0;
    for v in vertices {
        let (at, own) = (mesh.position(v), uv(v));
        // Each triangle's distance from the vertex, and how far it drags the texture there.
        let mut nearest = f64::MAX;
        let mut found: Vec<(f64, f64)> = Vec::new();
        for (p, u, scale) in &covers {
            let Some(w) = barycentric(at, *p) else {
                continue;
            };
            let clamped = w.map(|w| w.max(0.0));
            let sum: f64 = clamped.iter().sum();
            let closest = (p[0] * clamped[0] + p[1] * clamped[1] + p[2] * clamped[2]) * (1.0 / sum.max(1e-30));
            let distance = (closest - at).length();
            let there = [0, 1].map(|i| w[0] * u[0][i] + w[1] * u[1][i] + w[2] * u[2][i]);
            let off = ((there[0] - own[0]).powi(2) + (there[1] - own[1]).powi(2)).sqrt() * scale;
            nearest = nearest.min(distance);
            found.push((distance, off));
        }
        let tolerance = nearest * 1e-3 + 1e-6;
        if let Some(off) = found
            .iter()
            .filter(|(d, _)| *d <= nearest + tolerance)
            .map(|&(_, off)| off)
            .min_by(f64::total_cmp)
        {
            worst = worst.max(off);
        }
    }
    worst
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
            let area = (positions[1] - positions[0]).cross(positions[2] - positions[0]).length();
            let texture_area = ((texture[1][0] - texture[0][0]) * (texture[2][1] - texture[0][1])
                - (texture[2][0] - texture[0][0]) * (texture[1][1] - texture[0][1]))
                .abs();
            if texture_area < 1e-14 || area <= 0.0 {
                continue;
            }
            let scale = (area / texture_area).sqrt();
            let off = ((there[0] - should[0]).powi(2) + (there[1] - should[1]).powi(2)).sqrt() * scale;
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
        self.vertices[a].triangles.clear();
        let triangles = &self.triangles;
        self.vertices[b].triangles.retain(|&t| triangles[t].alive);
    }
}

/// Simplifies `triangles` (indices into `mesh`'s vertices) to `target` triangles or as near as
/// it can, never moving a vertex `locked` says to keep (given its welded index).
pub(crate) fn simplify(
    mesh: &Mesh,
    triangles: &[[u32; 3]],
    locked: impl Fn(u32) -> bool,
    target: usize,
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
    let dragged = measure_drag(mesh, triangles, &simplified) as f32;
    Simplified {
        triangles: simplified,
        error: worst.sqrt() as f32,
        dragged,
    }
}
