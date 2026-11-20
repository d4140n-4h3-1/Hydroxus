use crate::*;
use std::collections::HashMap;

/// An icosahedron subdivided `times` times onto the unit sphere: closed, every vertex shared.
fn icosphere(times: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let t = (1.0 + 5f32.sqrt()) / 2.0;
    let mut positions: Vec<[f32; 3]> = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ]
    .to_vec();
    let mut faces: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    let normalize = |p: [f32; 3]| {
        let l = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        [p[0] / l, p[1] / l, p[2] / l]
    };
    for p in positions.iter_mut() {
        *p = normalize(*p);
    }
    for _ in 0..times {
        let mut middles: HashMap<(u32, u32), u32> = HashMap::new();
        let mut middle = |a: u32, b: u32, positions: &mut Vec<[f32; 3]>| {
            *middles.entry((a.min(b), a.max(b))).or_insert_with(|| {
                let (pa, pb) = (positions[a as usize], positions[b as usize]);
                positions.push(normalize([pa[0] + pb[0], pa[1] + pb[1], pa[2] + pb[2]]));
                positions.len() as u32 - 1
            })
        };
        let mut next = Vec::new();
        for [a, b, c] in faces {
            let (ab, bc, ca) = (middle(a, b, &mut positions), middle(b, c, &mut positions), middle(c, a, &mut positions));
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }
    (positions, faces.into_iter().flatten().collect())
}

/// A sphere of rings and segments with a seam in its texture coordinates: the first and last
/// column of vertices are at the same places, and each pole is a row of vertices at one place,
/// as modelling tools export them. Returns the positions, indices, and each vertex's u.
fn seamed_sphere(segments: u32, rings: u32) -> (Vec<[f32; 3]>, Vec<u32>, Vec<f32>) {
    let mut positions = Vec::new();
    let mut us = Vec::new();
    for r in 0..=rings {
        let theta = std::f32::consts::PI * r as f32 / rings as f32;
        for s in 0..=segments {
            let u = s as f32 / segments as f32;
            let phi = std::f32::consts::TAU * (s % segments) as f32 / segments as f32;
            let (y, ring) = if r == 0 {
                (1.0, 0.0)
            } else if r == rings {
                (-1.0, 0.0)
            } else {
                (theta.cos(), theta.sin())
            };
            positions.push([ring * phi.cos(), y, ring * phi.sin()]);
            us.push(u);
        }
    }
    let row = segments + 1;
    let mut indices = Vec::new();
    for r in 0..rings {
        for s in 0..segments {
            let (a, b, c, d) = (r * row + s, r * row + s + 1, (r + 1) * row + s, (r + 1) * row + s + 1);
            indices.extend([a, b, d, a, d, c]);
        }
    }
    (positions, indices, us)
}

/// A cube of `n` by `n` quads a face, flat-shaded: each face has vertices of its own, so every
/// edge and corner of the cube is a seam, as hard-surface models are exported. Returns the
/// positions, indices, and each vertex's normal.
fn flat_cube(n: u32) -> (Vec<[f32; 3]>, Vec<u32>, Vec<[f32; 3]>) {
    let (mut positions, mut indices, mut normals) = (Vec::new(), Vec::new(), Vec::new());
    for axis in 0..3 {
        for sign in [-1.0f32, 1.0] {
            let normal = {
                let mut v = [0.0; 3];
                v[axis] = sign;
                v
            };
            // Two directions across the face, turning so that triangles face outwards.
            let (u, w) = ((axis + 1) % 3, (axis + 2) % 3);
            let first = positions.len() as u32;
            for j in 0..=n {
                for i in 0..=n {
                    let mut p = [0.0; 3];
                    p[axis] = sign;
                    p[u] = -1.0 + 2.0 * i as f32 / n as f32;
                    p[w] = -1.0 + 2.0 * j as f32 / n as f32;
                    positions.push(p);
                    normals.push(normal);
                }
            }
            for j in 0..n {
                for i in 0..n {
                    let (a, b, c, d) = (first + j * (n + 1) + i, first + j * (n + 1) + i + 1, first + (j + 1) * (n + 1) + i, first + (j + 1) * (n + 1) + i + 1);
                    if sign > 0.0 {
                        indices.extend([a, b, d, a, d, c]);
                    } else {
                        indices.extend([a, d, b, a, c, d]);
                    }
                }
            }
        }
    }
    (positions, indices, normals)
}

/// A flat square of `n` by `n` quads: open all round.
fn grid(n: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
    for z in 0..=n {
        for x in 0..=n {
            positions.push([x as f32, 0.0, z as f32]);
        }
    }
    let mut indices = Vec::new();
    for z in 0..n {
        for x in 0..n {
            let (a, b, c, d) = (z * (n + 1) + x, z * (n + 1) + x + 1, (z + 1) * (n + 1) + x, (z + 1) * (n + 1) + x + 1);
            indices.extend([a, c, b, b, c, d]);
        }
    }
    (positions, indices)
}

/// Separate square cards, two triangles each, scattered and turned about: foliage.
fn cards(count: u32) -> (Vec<[f32; 3]>, Vec<u32>) {
    let mut positions = Vec::new();
    let mut indices = Vec::new();
    let mut seed = 12345u32;
    let mut random = || {
        seed ^= seed << 13;
        seed ^= seed >> 17;
        seed ^= seed << 5;
        seed as f32 / u32::MAX as f32
    };
    for _ in 0..count {
        let middle = [random() * 10.0, random() * 10.0, random() * 10.0];
        let angle = random() * std::f32::consts::TAU;
        let (dx, dz) = (angle.cos() * 0.2, angle.sin() * 0.2);
        let first = positions.len() as u32;
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
            positions.push([middle[0] + sx * dx, middle[1] + sy * 0.2, middle[2] + sx * dz]);
        }
        indices.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
    }
    (positions, indices)
}

fn built(positions: &[[f32; 3]], indices: &[u32]) -> ClusterDag {
    build(&Mesh::new(positions, indices), Settings::default())
}

/// The clusters' shapes and the numbers linking them hold together.
fn check_structure(dag: &ClusterDag, vertex_count: usize) {
    for (i, c) in dag.clusters.iter().enumerate() {
        assert!(!c.triangles.is_empty(), "cluster {i} is empty");
        assert!(c.triangles.len() <= MAX_TRIANGLES, "cluster {i}: {} triangles", c.triangles.len());
        assert!(c.vertices.len() <= MAX_VERTICES, "cluster {i}: {} vertices", c.vertices.len());
        assert!(c.vertices.iter().all(|&v| (v as usize) < vertex_count));
        assert!(c.triangles.iter().flatten().all(|&k| (k as usize) < c.vertices.len()));
    }
    for (g, group) in dag.groups.iter().enumerate() {
        for &c in &group.children {
            assert_eq!(dag.clusters[c as usize].simplified_in, Some(g as u32));
        }
        for &p in &group.parents {
            assert_eq!(dag.clusters[p as usize].made_from, Some(g as u32));
        }
    }
}

/// Errors and bounds only grow up the graph, which is what makes any cut a whole one.
fn check_monotonic(dag: &ClusterDag) {
    for (i, c) in dag.clusters.iter().enumerate() {
        assert!(c.parent_error >= c.error, "cluster {i}: parent error {} < own {}", c.parent_error, c.error);
        if !c.is_root() {
            assert!(c.parent_bounds.contains(&c.lod_bounds), "cluster {i}: parent bounds do not contain its own");
        }
    }
    for (g, group) in dag.groups.iter().enumerate() {
        for &c in &group.children {
            assert!(group.error >= dag.clusters[c as usize].error, "group {g}");
            assert!(group.bounds.contains(&dag.clusters[c as usize].lod_bounds), "group {g}");
        }
        for &p in &group.parents {
            let parent = &dag.clusters[p as usize];
            assert_eq!((parent.error, parent.lod_bounds), (group.error, group.bounds), "group {g}");
        }
    }
}

/// Triangles in a canonical form: the same three corners in the same turning order.
fn canonical(t: [u32; 3]) -> [u32; 3] {
    let k = (0..3).min_by_key(|&k| t[k]).unwrap();
    [t[k], t[(k + 1) % 3], t[(k + 2) % 3]]
}

/// Level 0 is the mesh: every triangle once.
fn check_full_detail(dag: &ClusterDag, indices: &[u32], mesh: &Mesh) {
    let mut expected: Vec<[u32; 3]> = indices
        .chunks_exact(3)
        .map(|t| [t[0], t[1], t[2]])
        .filter(|t| {
            let w = t.map(|v| mesh.weld[v as usize]);
            w[0] != w[1] && w[1] != w[2] && w[0] != w[2]
        })
        .map(canonical)
        .collect();
    let mut got: Vec<[u32; 3]> = dag
        .clusters
        .iter()
        .filter(|c| c.level == 0)
        .flat_map(|c| c.mesh_triangles())
        .map(canonical)
        .collect();
    expected.sort_unstable();
    got.sort_unstable();
    assert_eq!(got, expected);
}

/// The directed welded edges of the triangles of `selected`, counted.
fn directed_edges(dag: &ClusterDag, mesh: &Mesh, selected: &[u32]) -> HashMap<(u32, u32), u32> {
    let mut edges = HashMap::new();
    for &c in selected {
        for t in dag.clusters[c as usize].mesh_triangles() {
            let w = t.map(|v| mesh.weld[v as usize]);
            for k in 0..3 {
                *edges.entry((w[k], w[(k + 1) % 3])).or_insert(0) += 1;
            }
        }
    }
    edges
}

/// Places to look from, near and far, and thresholds from fine to coarse.
fn views() -> Vec<([f32; 3], f32)> {
    let mut out = Vec::new();
    for eye in [[0.0, 0.0, 1.5], [3.0, 1.0, 0.0], [0.0, 10.0, 0.0], [40.0, 0.0, 40.0], [0.0, 0.0, 400.0]] {
        for threshold in [0.0, 0.0005, 0.002, 0.01, 0.05, 1.0] {
            out.push((eye, threshold));
        }
    }
    out
}

/// For a closed mesh, every cut is closed too: each edge once each way, nothing missing,
/// nothing twice. A curved one is drawn at different levels from different places; one that
/// simplifies with no error at all (`exact`) is drawn at its coarsest from everywhere.
fn check_closed_cuts(dag: &ClusterDag, mesh: &Mesh, exact: bool) {
    let mut levels_seen = std::collections::HashSet::new();
    for (eye, threshold) in views() {
        let selected = select(dag, eye, threshold);
        assert!(!selected.is_empty(), "nothing drawn from {eye:?} at {threshold}");
        for &c in &selected {
            levels_seen.insert(dag.clusters[c as usize].level);
        }
        let edges = directed_edges(dag, mesh, &selected);
        for (&(a, b), &n) in &edges {
            assert_eq!(n, 1, "edge {a}->{b} used {n} times from {eye:?} at {threshold}");
            assert!(
                edges.contains_key(&(b, a)),
                "crack at edge {a}->{b} from {eye:?} at {threshold}"
            );
        }
    }
    if exact {
        assert_eq!(levels_seen, [dag.levels() - 1].into(), "an exact mesh drawn finer than it need be");
    } else {
        assert!(levels_seen.len() > 2, "cuts only ever drew levels {levels_seen:?}");
    }
}

#[test]
fn an_icosphere_builds_into_whole_levels() {
    let (positions, indices) = icosphere(5);
    let mesh = Mesh::new(&positions, &indices);
    let dag = build(&mesh, Settings::default());
    check_structure(&dag, positions.len());
    check_monotonic(&dag);
    check_full_detail(&dag, &indices, &mesh);
    check_closed_cuts(&dag, &mesh, false);

    // Each level about half the one before, down to a little.
    let full = dag.triangles_at(0);
    assert_eq!(full, 20480);
    for level in 1..dag.levels() {
        let (was, now) = (dag.triangles_at(level - 1), dag.triangles_at(level));
        assert!(now < was, "level {level}: {now} triangles after {was}");
    }
    let coarsest: usize = dag.clusters.iter().filter(|c| c.is_root()).map(|c| c.triangles.len()).sum();
    assert!(coarsest * 20 < full, "the roots still have {coarsest} of {full} triangles");
    // Level 0 is about as few clusters as it can be.
    let first = dag.clusters.iter().filter(|c| c.level == 0).count();
    assert!(first <= full / MAX_TRIANGLES * 5 / 4, "{first} clusters at level 0");
}

#[test]
fn the_coarsest_icosphere_is_still_round() {
    let (positions, indices) = icosphere(5);
    let dag = built(&positions, &indices);
    // From far enough off, a handful of clusters, all of them still on the sphere's surface.
    let far = select(&dag, [0.0, 0.0, 1000.0], 0.01);
    let triangles: usize = far.iter().map(|&c| dag.clusters[c as usize].triangles.len()).sum();
    assert!(triangles < 2048, "{triangles} triangles drawn from far off");
    // Every vertex is one of the mesh's, so on the unit sphere; and the triangles' middles not
    // far inside it.
    for &c in &far {
        for t in dag.clusters[c as usize].mesh_triangles() {
            let p = t.map(|v| positions[v as usize]);
            let m: Vec<f32> = (0..3).map(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0).collect();
            let r = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
            assert!(r > 0.6, "a triangle's middle at radius {r}");
        }
    }
}

#[test]
fn a_seamed_sphere_stays_whole_and_keeps_its_seam() {
    let (positions, indices, us) = seamed_sphere(96, 64);
    let mesh = Mesh::new(&positions, &indices);
    let dag = build(&mesh, Settings::default());
    check_structure(&dag, positions.len());
    check_monotonic(&dag);
    check_full_detail(&dag, &indices, &mesh);
    check_closed_cuts(&dag, &mesh, false);
    assert!(dag.levels() > 3, "only {} levels", dag.levels());
    // No triangle at any level takes corners from both sides of the seam: each kept the vertex
    // on its own side. Then its texture coordinates map back onto where it is: the point of the
    // sphere at its corners' mean u and height lies about where its middle does, within the
    // triangle's own size. A corner taken from the other side of the seam pulls the mean u a
    // third of a turn away. Corners at the poles, at every u, are left out of the mean.
    let angle = |a: [f32; 3], b: [f32; 3]| {
        let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let la = (a[0] * a[0] + a[1] * a[1] + a[2] * a[2]).sqrt();
        let lb = (b[0] * b[0] + b[1] * b[1] + b[2] * b[2]).sqrt();
        (dot / (la * lb)).clamp(-1.0, 1.0).acos()
    };
    let (mut checked, mut crossings) = (0, 0);
    for c in &dag.clusters {
        for t in c.mesh_triangles() {
            let p = t.map(|v| positions[v as usize]);
            let off_pole: Vec<u32> = t.into_iter().filter(|&v| positions[v as usize][1].abs() < 1.0).collect();
            if off_pole.is_empty() {
                continue;
            }
            let u = off_pole.iter().map(|&v| us[v as usize]).sum::<f32>() / off_pole.len() as f32;
            let theta = p.iter().map(|p| p[1].clamp(-1.0, 1.0).acos()).sum::<f32>() / 3.0;
            let phi = u * std::f32::consts::TAU;
            let mapped = [theta.sin() * phi.cos(), theta.cos(), theta.sin() * phi.sin()];
            let middle = [0, 1, 2].map(|k| (p[0][k] + p[1][k] + p[2][k]) / 3.0);
            let size = angle(p[0], p[1]).max(angle(p[1], p[2])).max(angle(p[2], p[0]));
            checked += 1;
            if angle(mapped, middle) <= size + 0.05 {
                continue;
            }
            // Broken only where nothing cheaper was left, and counted as moving the surface by
            // an edge's length: never in a cluster whose error is smaller than the shortest
            // edges round the poles.
            crossings += 1;
            assert!(
                c.error >= 0.002,
                "a level {} triangle with error {} spans the seam: u {:?}",
                c.level,
                c.error,
                t.map(|v| us[v as usize])
            );
        }
    }
    assert!(checked > 10_000);
    assert!(crossings * 100 < checked, "{crossings} of {checked} triangles span the seam");
}

#[test]
fn a_flat_shaded_cube_simplifies_and_keeps_its_normals() {
    let (positions, indices, normals) = flat_cube(32);
    let mesh = Mesh::new(&positions, &indices);
    let dag = build(&mesh, Settings::default());
    check_structure(&dag, positions.len());
    check_monotonic(&dag);
    check_full_detail(&dag, &indices, &mesh);
    check_closed_cuts(&dag, &mesh, true);
    let full = dag.triangles_at(0);
    let coarsest: usize = dag.clusters.iter().filter(|c| c.is_root()).map(|c| c.triangles.len()).sum();
    assert!(dag.levels() >= 6, "only {} levels", dag.levels());
    assert!(coarsest * 20 < full, "the roots still have {coarsest} of {full} triangles");
    // A corner whose normal is not its triangle's own way is shaded as another face. Corners
    // take the vertex of the side facing their way, so that is rare even where the cube's edges
    // have been simplified across.
    let (mut corners, mut wrong) = (0, 0);
    for c in &dag.clusters {
        for t in c.mesh_triangles() {
            let p = t.map(|v| positions[v as usize]);
            let e1 = [0, 1, 2].map(|k| p[1][k] - p[0][k]);
            let e2 = [0, 1, 2].map(|k| p[2][k] - p[0][k]);
            let facing = [e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]];
            for v in t {
                let n = normals[v as usize];
                corners += 1;
                if n[0] * facing[0] + n[1] * facing[1] + n[2] * facing[2] <= 0.0 {
                    wrong += 1;
                }
            }
        }
    }
    assert!(wrong * 50 < corners, "{wrong} of {corners} corners shaded as another face");
}

#[test]
fn an_open_grid_keeps_its_outline() {
    let n = 96;
    let (positions, indices) = grid(n);
    let mesh = Mesh::new(&positions, &indices);
    let dag = build(&mesh, Settings::default());
    check_structure(&dag, positions.len());
    check_monotonic(&dag);
    check_full_detail(&dag, &indices, &mesh);
    assert!(dag.levels() > 3, "only {} levels", dag.levels());
    let mut first_of: HashMap<u32, usize> = HashMap::new();
    for (v, &w) in mesh.weld.iter().enumerate() {
        first_of.entry(w).or_insert(v);
    }
    let on_border = |w: u32, _: &Mesh| {
        let p = positions[first_of[&w]];
        p[0] == 0.0 || p[2] == 0.0 || p[0] == n as f32 || p[2] == n as f32
    };
    for (eye, threshold) in views() {
        let eye = [eye[0] + 48.0, eye[1] + 5.0, eye[2] + 48.0];
        let selected = select(&dag, eye, threshold);
        let edges = directed_edges(&dag, &mesh, &selected);
        let mut area = 0.0;
        for &c in &selected {
            for t in dag.clusters[c as usize].mesh_triangles() {
                let p = t.map(|v| positions[v as usize]);
                area += ((p[1][0] - p[0][0]) * (p[2][2] - p[0][2]) - (p[2][0] - p[0][0]) * (p[1][2] - p[0][2])).abs() / 2.0;
            }
        }
        // Flat, so simplified without moving the surface at all: the same square, whole.
        assert!((area - (n * n) as f32).abs() < 1e-2, "area {area} from {eye:?} at {threshold}");
        for (&(a, b), &count) in &edges {
            assert_eq!(count, 1, "edge {a}->{b} used {count} times");
            if !edges.contains_key(&(b, a)) {
                assert!(on_border(a, &mesh) && on_border(b, &mesh), "crack at {a}->{b} from {eye:?} at {threshold}");
            }
        }
    }
}

#[test]
fn foliage_cards_make_full_clusters_and_thin_out() {
    let (positions, indices) = cards(3000);
    let mesh = Mesh::new(&positions, &indices);
    let dag = build(&mesh, Settings::default());
    check_structure(&dag, positions.len());
    check_monotonic(&dag);
    check_full_detail(&dag, &indices, &mesh);
    let first = dag.clusters.iter().filter(|c| c.level == 0).count();
    assert!(first <= 6000 / MAX_TRIANGLES * 5 / 4, "{first} clusters for 6000 triangles");
    assert!(dag.levels() > 1);
    assert!(dag.triangles_at(1) < dag.triangles_at(0));
}

#[test]
fn doubled_cards_are_drawn_from_everywhere() {
    // Each card twice, once each way round, as a game makes single sheets seen from both sides.
    let (positions, mut indices) = cards(3000);
    let back: Vec<u32> = indices.chunks_exact(3).flat_map(|t| [t[0], t[2], t[1]]).collect();
    indices.extend(back);
    let mesh = Mesh::new(&positions, &indices);
    let dag = build(&mesh, Settings::default());
    check_structure(&dag, positions.len());
    check_monotonic(&dag);
    for (eye, threshold) in views() {
        let eye = [eye[0] + 5.0, eye[1] + 5.0, eye[2] + 5.0];
        let drawn: usize = select(&dag, eye, threshold).iter().map(|&c| dag.clusters[c as usize].triangles.len()).sum();
        assert!(drawn > 0, "nothing drawn from {eye:?} at {threshold}");
    }
    // Doubled, a card is no easier to lose than single: its outline holds it, as an open border
    // does, so it thins out as slowly - not to nothing at no error, as a flat closed surface would.
    let single = built(&positions, &indices[..indices.len() / 2]);
    let level_error = |dag: &ClusterDag, level: u32| {
        dag.clusters.iter().filter(|c| c.level == level).map(|c| c.error).fold(f32::INFINITY, f32::min)
    };
    for level in 1..single.levels().min(dag.levels()) {
        let (one, two) = (level_error(&single, level), level_error(&dag, level));
        assert!(two >= one * 0.5 && two > 0.0, "level {level}: doubled cards' least error {two}, single cards' {one}");
    }
}

#[test]
fn bytes_make_the_same_graph() {
    let (positions, indices) = icosphere(3);
    let dag = built(&positions, &indices);
    let bytes = dag.to_bytes();
    assert_eq!(ClusterDag::from_bytes(&bytes), Ok(dag));
    assert_eq!(ClusterDag::from_bytes(&bytes[..bytes.len() - 1]), Err(FormatError::Corrupt));
    assert_eq!(ClusterDag::from_bytes(b"GLTF...."), Err(FormatError::NotYoctite));
    let mut other = bytes.clone();
    other[4] = 9;
    assert_eq!(ClusterDag::from_bytes(&other), Err(FormatError::Version(9)));
}

#[test]
fn an_empty_mesh_has_no_clusters() {
    let dag = built(&[], &[]);
    assert!(dag.clusters.is_empty() && dag.groups.is_empty());
}
