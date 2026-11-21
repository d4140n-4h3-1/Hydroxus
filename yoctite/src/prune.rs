//! Thinning foliage: half a group's loose cards left out, and the rest made bigger to cover as
//! much as all of them did (Cook, Halstead, Planck and Ryu, "Stochastic Simplification of
//! Aggregate Detail", SIGGRAPH 2007).
//!
//! Foliage is made of many small separate cards - a leaf, a spray of needles - each a few
//! triangles with an open outline all round. Collapsing edges cannot simplify a card short of
//! losing it, which moves the surface by the whole card, so foliage would keep every card until
//! each is smaller than a pixel. Seen from far enough off, though, a crown of a thousand cards
//! looks much like one of five hundred cards each twice the size: what shows is how much of the
//! sky the leaves cover, and in what pattern, not where each leaf is. So at each coarser level
//! half the cards go, chosen at random, and those kept are each grown about their middles by
//! the square root of the area lost, keeping the cover the same.
//!
//! A grown card is a new card: its corners are copies of the mesh's own vertices, with their
//! normals and texture coordinates, at new positions ([`crate::ExtraVertex`]). How far a
//! level of thinned cards is from the full mesh is counted as a share of the cards' size
//! ([`crate::Settings::card_error`]): not the whole of it, as each card's place would be, since
//! what is kept is how the crown looks, not where each card is.

use crate::{math::V3, Mesh};
use std::collections::HashMap;

/// How many triangles a loose piece may have and still be a card. A leaf card is two, doubled
/// four; a strip of needles or a small branch card some more.
const MAX_CARD_TRIANGLES: usize = 64;
/// How many cards a group needs, and what share of its triangles they have to be, for it to be
/// thinned as foliage. Below that, a few loose pieces of something solid - bolts on a machine -
/// are simplified as the rest of it is, not thrown away.
const MIN_CARDS: usize = 8;
const MIN_CARD_SHARE: f32 = 0.5;
/// The most a card is grown by, across, at one level.
const MAX_GROWTH: f32 = 2.0;

/// A group's loose cards thinned out.
pub(crate) struct Thinned {
    /// The triangles of the cards kept, grown: indices into the mesh's vertices and, from
    /// `first_new` on, into `new_vertices`.
    pub triangles: Vec<[u32; 3]>,
    /// The grown cards' corners: each a copy of one of the mesh's vertices, at a new position.
    pub new_vertices: Vec<(u32, [f32; 3])>,
    /// How far the thinned cards are from the full mesh, by how big the cards are.
    pub error: f32,
    /// The vertex each of the kept cards' vertices became.
    pub moved: HashMap<u32, u32>,
}

/// A group's triangles split into its loose cards and the rest.
pub(crate) struct Split {
    pub cards: Vec<Vec<[u32; 3]>>,
    pub rest: Vec<[u32; 3]>,
}

/// Splits `triangles` (indices into `mesh`'s vertices) into the pieces they make, joined by
/// welded vertices, and sorts them into cards - small, and nowhere on the group's border, which
/// `locked` says (by welded vertex) - and the rest. [`None`] when the group is not foliage: too
/// few cards, or too little of it.
pub(crate) fn split(mesh: &Mesh, triangles: &[[u32; 3]], locked: impl Fn(u32) -> bool) -> Option<Split> {
    // Pieces by union-find over welded vertices.
    let mut parent: HashMap<u32, u32> = HashMap::new();
    fn find(parent: &mut HashMap<u32, u32>, w: u32) -> u32 {
        let mut root = w;
        while let Some(&p) = parent.get(&root) {
            if p == root {
                break;
            }
            root = p;
        }
        // Path halving would do; pieces are small, so a second pass to point at the root is
        // enough.
        let mut at = w;
        while at != root {
            let next = parent[&at];
            parent.insert(at, root);
            at = next;
        }
        root
    }
    for t in triangles {
        let w = t.map(|v| mesh.weld[v as usize]);
        for &x in &w {
            parent.entry(x).or_insert(x);
        }
        let (a, b, c) = (find(&mut parent, w[0]), find(&mut parent, w[1]), find(&mut parent, w[2]));
        parent.insert(b, a);
        parent.insert(c, a);
    }
    let mut pieces: HashMap<u32, (Vec<[u32; 3]>, bool)> = HashMap::new();
    for t in triangles {
        let w = t.map(|v| mesh.weld[v as usize]);
        let root = find(&mut parent, w[0]);
        let piece = pieces.entry(root).or_insert_with(|| (Vec::new(), false));
        piece.0.push(*t);
        piece.1 |= w.iter().any(|&w| locked(w));
    }
    let mut cards = Vec::new();
    let mut rest = Vec::new();
    // In a fixed order: the same mesh thins the same way every time it is built.
    let mut pieces: Vec<(Vec<[u32; 3]>, bool)> = pieces.into_values().collect();
    pieces.sort_unstable_by_key(|(t, _)| t.iter().flatten().copied().min());
    for (piece, on_border) in pieces {
        if !on_border && piece.len() <= MAX_CARD_TRIANGLES {
            cards.push(piece);
        } else {
            rest.extend(piece);
        }
    }
    let card_triangles: usize = cards.iter().map(Vec::len).sum();
    let foliage =
        cards.len() >= MIN_CARDS && card_triangles as f32 >= triangles.len() as f32 * MIN_CARD_SHARE;
    foliage.then_some(Split { cards, rest })
}

/// A number from a card's lowest vertex, the same every time, to choose which cards are kept.
fn shuffle_key(card: &[[u32; 3]]) -> u64 {
    let lowest = card.iter().flatten().copied().min().unwrap_or(0) as u64;
    let mut z = lowest.wrapping_add(0x9e37_79b9_7f4a_7c15);
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// Keeps half of `cards`, chosen at random, grown to cover what all of them did; `first_new` is
/// the index the first new vertex will have. `card_error` is how much of a card's size counts
/// as error (see [`crate::Settings::card_error`]). `source` gives the mesh vertex a vertex is
/// a copy of, for vertices that are themselves copies.
pub(crate) fn thin(
    mesh: &Mesh,
    cards: &[Vec<[u32; 3]>],
    first_new: u32,
    card_error: f32,
    source: impl Fn(u32) -> u32,
) -> Thinned {
    let area = |t: &[u32; 3]| {
        let p = t.map(|v| mesh.position(v));
        (p[1] - p[0]).cross(p[2] - p[0]).length() * 0.5
    };
    // Each card's middle, by area, and how far its corners reach from it.
    let shapes: Vec<(V3, f64, f64)> = cards
        .iter()
        .map(|card| {
            let mut middle = V3::new(0.0, 0.0, 0.0);
            let mut total = 0.0;
            for t in card {
                let a = area(t);
                let p = t.map(|v| mesh.position(v));
                middle += (p[0] + p[1] + p[2]) * (a / 3.0);
                total += a;
            }
            if total > 0.0 {
                middle = middle * (1.0 / total);
            } else {
                middle = mesh.position(card[0][0]);
            }
            let reach = card
                .iter()
                .flatten()
                .map(|&v| (mesh.position(v) - middle).length())
                .fold(0.0, f64::max);
            (middle, total, reach)
        })
        .collect();

    let mut order: Vec<usize> = (0..cards.len()).collect();
    order.sort_unstable_by_key(|&c| shuffle_key(&cards[c]));
    let keep = order.len().div_ceil(2);
    let (kept, dropped) = order.split_at(keep);
    let all: f64 = shapes.iter().map(|s| s.1).sum();
    let kept_area: f64 = kept.iter().map(|&c| shapes[c].1).sum();
    let growth = if kept_area > 0.0 {
        ((all / kept_area).sqrt() as f32).clamp(1.0, MAX_GROWTH)
    } else {
        1.0
    };

    let mut triangles = Vec::new();
    let mut new_vertices: Vec<(u32, [f32; 3])> = Vec::new();
    let mut moved: HashMap<u32, u32> = HashMap::new();
    let mut error: f64 = 0.0;
    for &c in kept {
        let (middle, _, reach) = shapes[c];
        // A grown corner moves this far.
        error = error.max(reach * (growth as f64 - 1.0));
        let middle = middle.to_f32();
        // Cards share no vertices, so one map serves them all.
        for t in &cards[c] {
            triangles.push(t.map(|v| {
                *moved.entry(v).or_insert_with(|| {
                    // In single precision from the vertex's own position, so the two sides of a
                    // doubled card, at one place, grow to one place, and stay welded.
                    let p = mesh.position_f32(v);
                    let grown = [0, 1, 2].map(|i| middle[i] + (p[i] - middle[i]) * growth);
                    new_vertices.push((source(v), grown));
                    first_new + new_vertices.len() as u32 - 1
                })
            }));
        }
    }
    for &c in dropped {
        // A card gone leaves a gap as big as itself.
        error = error.max(shapes[c].2);
    }
    Thinned {
        triangles,
        new_vertices,
        error: error as f32 * card_error,
        moved,
    }
}
