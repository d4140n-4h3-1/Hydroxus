//! Builds the Yoctite clusters of every mesh in glTF files and says what came of them.
//!
//! `cargo run -p yoctite --release --example inspect -- [--game] model.glb [more.glb ...]`
//!
//! With `--game`, each mesh is built as a game drawing it would build it: with its texture
//! coordinates, and every primitive after a mesh's first - foliage, on the plant models - made
//! double-sided by a reversed copy of each triangle.

use std::time::Instant;
use yoctite::{build, select, ClusterDag, Mesh, Settings};

fn main() {
    let mut paths: Vec<String> = std::env::args().skip(1).collect();
    let game = paths.iter().any(|p| p == "--game");
    paths.retain(|p| p != "--game");
    if paths.is_empty() {
        eprintln!("usage: inspect <model.glb> [...]");
        std::process::exit(2);
    }
    for path in paths {
        let (document, buffers, _) = match gltf::import(&path) {
            Ok(imported) => imported,
            Err(error) => {
                eprintln!("{path}: {error}");
                continue;
            }
        };
        println!("{path}");
        let (mut total_time, mut total_triangles) = (0.0, 0);
        for mesh in document.meshes() {
            for (p, primitive) in mesh.primitives().enumerate() {
                let reader = primitive.reader(|b| Some(&buffers[b.index()]));
                let Some(positions) = reader.read_positions() else { continue };
                let positions: Vec<[f32; 3]> = positions.collect();
                let mut indices: Vec<u32> = match reader.read_indices() {
                    Some(indices) => indices.into_u32().collect(),
                    None => (0..positions.len() as u32).collect(),
                };
                let uvs: Option<Vec<[f32; 2]>> =
                    reader.read_tex_coords(0).filter(|_| game).map(|uvs| uvs.into_f32().collect());
                if game && p > 0 {
                    let back: Vec<u32> = indices.chunks_exact(3).flat_map(|t| [t[0], t[2], t[1]]).collect();
                    indices.extend(back);
                }
                let started = Instant::now();
                let mut built_mesh = Mesh::new(&positions, &indices);
                if let Some(uvs) = &uvs {
                    built_mesh = built_mesh.with_texture_coordinates(uvs);
                }
                let dag = build(&built_mesh, Settings::default());
                let took = started.elapsed().as_secs_f64();
                total_time += took;
                total_triangles += indices.len() / 3;
                report(mesh.name().unwrap_or("?"), p, &dag, indices.len() / 3, took);
            }
        }
        println!("  all meshes: {total_triangles} triangles in {:.2} s\n", total_time);
    }
}

fn report(name: &str, primitive: usize, dag: &ClusterDag, triangles: usize, took: f64) {
    let per_level: Vec<String> = (0..dag.levels()).map(|l| dag.triangles_at(l).to_string()).collect();
    // Each level's errors, least and greatest.
    let errors: Vec<String> = (1..dag.levels())
        .map(|l| {
            let at = dag.clusters.iter().filter(|c| c.level == l).map(|c| c.error);
            let (least, most) = at.fold((f32::MAX, 0.0f32), |(a, b), e| (a.min(e), b.max(e)));
            format!("{least:.3}-{most:.3}")
        })
        .collect();
    let roots: Vec<_> = dag.clusters.iter().filter(|c| c.is_root()).collect();
    let root_triangles: usize = roots.iter().map(|c| c.triangles.len()).sum();
    let level0 = dag.clusters.iter().filter(|c| c.level == 0).count();
    let fill = dag.triangles_at(0) as f64 / level0.max(1) as f64;
    // What would be drawn of it, from straight ahead, at a pixel's error on a 1080-row screen
    // with a 75 degree field of view, the further off the fewer.
    let pixel = 75f32.to_radians() / 1080.0;
    let center = roots.first().map_or([0.0; 3], |c| c.lod_bounds.center);
    let at = |d: f32| -> usize {
        select(dag, [center[0], center[1], center[2] + d], pixel)
            .iter()
            .map(|&c| dag.clusters[c as usize].triangles.len())
            .sum()
    };
    println!(
        "  {name} #{primitive}: {triangles} triangles, {} clusters ({level0} at full detail, {fill:.0} triangles each), \
         {} levels [{}], roots {} clusters / {root_triangles} triangles, max error {:.4}, {:.1} KB, {:.0} ms\n    \
         errors by level [{}]\n    drawn at 5 m: {}, 20 m: {}, 80 m: {}, 320 m: {}",
        dag.clusters.len(),
        dag.levels(),
        per_level.join(" "),
        roots.len(),
        dag.clusters.iter().map(|c| c.error).fold(0.0, f32::max),
        dag.to_bytes().len() as f64 / 1024.0,
        took * 1000.0,
        errors.join(" "),
        at(5.0),
        at(20.0),
        at(80.0),
        at(320.0),
    );
}
