//! Surfaces seen from both sides.
//!
//! The renderer culls faces seen from behind. That is right for a closed model, but a mesh whose
//! faces do not agree on which way is "in" - a tube meant to be walked through, say, with some
//! faces pointing into it and some out - loses walls and ceilings from one side or the other.
//! [`make_double_sided`] gives every triangle a reversed twin with flipped normals: each side then
//! has a front face with its own, correct normal, so it is both drawn and lit properly whichever
//! side it is seen from.

use fyrox::{
    core::{algebra::Vector4, math::TriangleDefinition, pool::Handle},
    fxhash::FxHashSet,
    graph::SceneGraph,
    scene::{
        graph::Graph,
        mesh::{
            buffer::{VertexAttributeUsage, VertexReadTrait, VertexWriteTrait},
            Mesh,
        },
        node::Node,
    },
};

/// Doubles every surface under `root`. Surface data shared between meshes is doubled once, and
/// `done` remembers what was, so surface data shared with an earlier call is not doubled again.
pub fn make_double_sided(graph: &mut Graph, root: Handle<Node>, done: &mut FxHashSet<u64>) {
    let meshes: Vec<Handle<Node>> = graph.traverse_handle_iter(root).collect();
    for handle in meshes {
        let Some(mesh) = graph[handle].cast::<Mesh>() else {
            continue;
        };
        for surface in mesh.surfaces() {
            let resource = surface.data();
            if !done.insert(resource.key()) {
                continue;
            }
            let mut data = resource.data_ref();
            let original_count = data.vertex_buffer.vertex_count();
            let raw = data.vertex_buffer.raw_data().to_vec();
            let vertex_size = data.vertex_buffer.vertex_size() as usize;
            {
                let mut vertices = data.vertex_buffer.modify();
                for vertex in raw.chunks_exact(vertex_size) {
                    let _ = vertices.push_vertex_raw(vertex);
                }
                // Flip the copies.
                for mut vertex in vertices.iter_mut().skip(original_count as usize) {
                    if let Ok(normal) = vertex.read_3_f32(VertexAttributeUsage::Normal) {
                        let _ = vertex.write_3_f32(VertexAttributeUsage::Normal, -normal);
                    }
                    if let Ok(t) = vertex.read_4_f32(VertexAttributeUsage::Tangent) {
                        let _ = vertex.write_4_f32(
                            VertexAttributeUsage::Tangent,
                            Vector4::new(-t.x, -t.y, -t.z, t.w),
                        );
                    }
                }
            }
            let reversed: Vec<TriangleDefinition> = data
                .geometry_buffer
                .iter()
                .map(|t| {
                    TriangleDefinition([
                        t.0[0] + original_count,
                        t.0[2] + original_count,
                        t.0[1] + original_count,
                    ])
                })
                .collect();
            let mut triangles = data.geometry_buffer.modify();
            for triangle in reversed {
                triangles.push(triangle);
            }
        }
    }
}
