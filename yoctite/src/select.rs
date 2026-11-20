//! Choosing which clusters to draw, from where: the cut through the graph the GPU will make
//! every frame, here on the CPU as its reference.

use crate::{math::V3, ClusterDag, Sphere};

/// How big `error`, judged over `bounds`, looks from `eye`: the error over the distance to the
/// nearest point of the bounds. Infinite from inside them, unless the error is none. Never
/// smaller for a bigger error or for bounds round smaller ones, so a parent never looks better
/// than its children.
pub fn projected_error(error: f32, bounds: &Sphere, eye: [f32; 3]) -> f32 {
    if error == 0.0 {
        return 0.0;
    }
    let distance = (V3::from_f32(bounds.center) - V3::from_f32(eye)).length() as f32 - bounds.radius;
    if distance <= 0.0 {
        f32::INFINITY
    } else {
        error / distance
    }
}

/// The clusters to draw seen from `eye`, where an error that looks as big as `threshold` (in
/// the units of [`projected_error`]: a pixel is about the field of view's height in radians
/// over the screen's height in pixels) is as much as may show: each cluster whose own error
/// looks no bigger than that and whose parent's does.
pub fn select(dag: &ClusterDag, eye: [f32; 3], threshold: f32) -> Vec<u32> {
    dag.clusters
        .iter()
        .enumerate()
        .filter(|(_, c)| {
            projected_error(c.error, &c.lod_bounds, eye) <= threshold
                && projected_error(c.parent_error, &c.parent_bounds, eye) > threshold
        })
        .map(|(i, _)| i as u32)
        .collect()
}
