//! Yoctite's geometry on the GPU: every cluster of every mesh in a few shared buffers, and every
//! copy of a mesh placed in the world, ready for the GPU to pick and draw clusters from by itself.
//!
//! A mesh's clusters, at every level of detail, come from the `yoctite` crate's builder
//! ([`yoctite::ClusterDag`]). [`GeometryPool::add_mesh`] uploads them once:
//!
//! - its vertices, [`PoolVertex`] each, into one buffer shared by all meshes;
//! - for each cluster, the vertices it uses, as indices into that buffer, and its triangles,
//!   three bytes each packed into a word, as indices into its own list of vertices;
//! - a record of each cluster ([`GpuCluster`]): where those are, its bounds, and its errors and
//!   its parent group's, which pick its level of detail;
//! - and a record of the mesh ([`GpuMesh`]): where its clusters are.
//!
//! [`GeometryPool::add_instance`] places a copy of a mesh ([`GpuInstance`]): a transform and which
//! mesh.
//!
//! A mesh can bend: [`GeometryPool::add_morphing_mesh`] uploads, with its vertices, how far each
//! of its morph targets (blend shapes) moves each of them, and each copy of it is given weights
//! for them ([`GeometryPool::set_weights`]) - a tree swaying in the wind, every tree its own way.
//! The offsets, as half floats, and each copy's weights live in the vertex buffer itself, past
//! the vertices: a pass that draws the pool needs no buffer more for them (see [`GpuInstance`]).
//! Every level of detail bends alike, since each of a coarser level's vertices is one of the
//! mesh's own or a copy of one, with its offsets; and each cluster's bounds take in as far as its
//! vertices can be moved, so culling never leaves out a bent branch. Instances come and go and move as the scene does; meshes are uploaded once and stay until
//! removed. Freed room is used again, and a buffer that runs out of room is made twice as big,
//! with what it held copied across, so nothing already in it moves.
//!
//! [`ClusterView`] draws clusters straight from the pool - given which ones, as pairs of instance
//! and cluster - in one instanced draw, each in a colour of its own: to see the clusters and the
//! levels of detail chosen. The GPU will make that list itself, culling and choosing levels of
//! detail every frame, in the next phase.

use crate::{
    framebuffer::{PendingClear, WgpuFrameBuffer},
    sampler::WgpuSampler,
    server::WgpuGraphicsServer,
    texture::WgpuTexture,
};
use fyrox_graphics::{
    error::FrameworkError,
    framebuffer::GpuFrameBuffer,
    gpu_texture::{GpuTexture, GpuTextureDescriptor, GpuTextureKind, PixelKind},
    sampler::GpuSampler,
    server::GraphicsServer,
};
use std::marker::PhantomData;
use wgpu::util::DeviceExt;
use yoctite::ClusterDag;

/// A vertex as the pool keeps it: everything the renderer shades a surface with.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PoolVertex {
    /// Where it is, in its mesh's own space.
    pub position: [f32; 3],
    /// Which way its surface faces.
    pub normal: [f32; 3],
    /// Its first texture coordinates.
    pub uv: [f32; 2],
    /// Its tangent, and in the fourth the handedness of its bitangent.
    pub tangent: [f32; 4],
}

/// How many floats a [`PoolVertex`] is: what a shader steps through the vertex buffer by.
pub const VERTEX_FLOATS: u32 = (std::mem::size_of::<PoolVertex>() / 4) as u32;
const VERTEX_BYTES: u64 = std::mem::size_of::<PoolVertex>() as u64;

/// A cluster as the GPU reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCluster {
    /// Where its vertices' indices start in the cluster vertex buffer. They index the vertex
    /// buffer as a whole: the mesh's own place in it is already added in.
    pub first_vertex: u32,
    /// Where its triangles start in the triangle buffer, a word each.
    pub first_triangle: u32,
    /// How many vertices and triangles it has.
    pub vertex_count: u32,
    /// How many triangles it has, at most [`yoctite::MAX_TRIANGLES`].
    pub triangle_count: u32,
    /// Bounds round its triangles, for culling: centre and radius, in the mesh's own space.
    pub bounds: [f32; 4],
    /// The bounds its error is judged over.
    pub lod_bounds: [f32; 4],
    /// The bounds its parent group's error is judged over.
    pub parent_bounds: [f32; 4],
    /// How far its surface is from the full mesh's, at most.
    pub error: f32,
    /// The same for its parent group: [`f32::MAX`] for a root.
    pub parent_error: f32,
    /// Which mesh it is of.
    pub mesh: u32,
    /// How many times simplified: 0 for the full mesh.
    pub level: u32,
}

/// A mesh as the GPU reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuMesh {
    /// Where its clusters start in the cluster buffer.
    pub first_cluster: u32,
    /// How many clusters it has, at every level of detail together.
    pub cluster_count: u32,
    /// Where its vertices start in the vertex buffer.
    pub first_vertex: u32,
    /// How many vertices it has.
    pub vertex_count: u32,
    /// Bounds round the whole mesh, in its own space: round its roots' error bounds, which take
    /// in every cluster.
    pub bounds: [f32; 4],
}

/// A copy of a mesh in the world, as the GPU reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuInstance {
    /// From the mesh's space to the world's: three rows of four.
    pub transform: [f32; 12],
    /// Which mesh it is a copy of.
    pub mesh: u32,
    /// [`INSTANCE_ALIVE`], or 0 for a free slot.
    pub flags: u32,
    /// Which material it is drawn with: a slot the caller numbers, from 0 (see
    /// [`CullParameters::material_slots`]).
    pub material: u32,
    /// For a copy of a mesh that bends, where its block in the vertex buffer starts, counted in
    /// words; else [`NO_MORPH`]. The block is the mesh's first vertex, where its offsets start
    /// (again in words), and how many targets it has, then the copy's weight for each target.
    /// The offsets are each vertex's, target after target, three words each: position x and y,
    /// z and normal x, normal y and z, as two half floats a word, low half first.
    pub morph: u32,
}

/// [`GpuInstance::morph`] for a copy of a mesh that does not bend.
pub const NO_MORPH: u32 = u32::MAX;

/// How a mesh bends: its morph targets' offsets, for [`GeometryPool::add_morphing_mesh`].
#[derive(Debug, Clone, Copy)]
pub struct Morphs<'a> {
    /// How many targets it has.
    pub targets: u32,
    /// For each vertex, for each target, how far it moves the vertex's position and its normal,
    /// at a weight of 1: vertex after vertex, the vertex's targets in order.
    pub offsets: &'a [[f32; 6]],
    /// The largest weight a target is given, either way: each cluster's bounds are grown by as
    /// far as its vertices can move with every target at this.
    pub most_weight: f32,
}

/// How many words of a block before the weights (see [`GpuInstance::morph`]).
const MORPH_HEADER: u32 = 3;

/// An instance slot in use.
pub const INSTANCE_ALIVE: u32 = 1;

/// A mesh in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeshId(pub u32);

/// An instance in the pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InstanceId(pub u32);

/// Hands out ranges of a buffer, first fit, and takes them back, joining free neighbours.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RangeAllocator {
    /// Free ranges below `end`, as start and length, in order, none touching.
    free: Vec<(u32, u32)>,
    /// Everything from here on is free.
    end: u32,
}

impl RangeAllocator {
    /// A range of `length`, by its start.
    pub fn allocate(&mut self, length: u32) -> u32 {
        if length == 0 {
            return 0;
        }
        if let Some(i) = self.free.iter().position(|&(_, l)| l >= length) {
            let (start, l) = self.free[i];
            if l == length {
                self.free.remove(i);
            } else {
                self.free[i] = (start + length, l - length);
            }
            return start;
        }
        // A free range at the very end grows into the rest.
        if let Some(&(start, l)) = self.free.last() {
            if start + l == self.end {
                self.free.pop();
                self.end = start + length;
                return start;
            }
        }
        let start = self.end;
        self.end += length;
        start
    }

    /// Gives back the range at `start` of `length`.
    pub fn free(&mut self, start: u32, length: u32) {
        if length == 0 {
            return;
        }
        let i = self.free.partition_point(|&(s, _)| s < start);
        self.free.insert(i, (start, length));
        // Joined with the next, then the previous.
        if i + 1 < self.free.len() && start + length == self.free[i + 1].0 {
            self.free[i].1 += self.free[i + 1].1;
            self.free.remove(i + 1);
        }
        if i > 0 && self.free[i - 1].0 + self.free[i - 1].1 == start {
            self.free[i - 1].1 += self.free[i].1;
            self.free.remove(i);
        }
        // What is free up to the end is just the end.
        if let Some(&(s, l)) = self.free.last() {
            if s + l == self.end {
                self.free.pop();
                self.end = s;
            }
        }
    }

    /// How far into the buffer anything has been handed out.
    pub fn high_water(&self) -> u32 {
        self.end
    }

    /// How much is handed out.
    pub fn used(&self) -> u32 {
        self.end - self.free.iter().map(|&(_, l)| l).sum::<u32>()
    }
}

/// A storage buffer of `T`s that ranges are handed out of, made bigger when it runs out of
/// room.
pub struct PoolBuffer<T> {
    buffer: wgpu::Buffer,
    /// How many `T`s it has room for.
    capacity: u32,
    ranges: RangeAllocator,
    label: &'static str,
    usage: wgpu::BufferUsages,
    _of: PhantomData<T>,
}

impl<T: bytemuck::Pod> PoolBuffer<T> {
    fn new(device: &wgpu::Device, label: &'static str, capacity: u32, usage: wgpu::BufferUsages) -> Self {
        let capacity = capacity.max(1);
        let usage = usage | wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        Self {
            buffer: Self::create(device, label, capacity, usage),
            capacity,
            ranges: RangeAllocator::default(),
            label,
            usage,
            _of: PhantomData,
        }
    }

    fn create(device: &wgpu::Device, label: &str, capacity: u32, usage: wgpu::BufferUsages) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: capacity as u64 * std::mem::size_of::<T>() as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Room for `count` more, by where it starts, making the buffer bigger if it has to.
    fn allocate(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, count: u32) -> u32 {
        let start = self.ranges.allocate(count);
        let needed = self.ranges.high_water();
        if needed > self.capacity {
            let capacity = needed.max(self.capacity * 2);
            let bigger = Self::create(device, self.label, capacity, self.usage);
            let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("YoctitePoolGrow"),
            });
            encoder.copy_buffer_to_buffer(
                &self.buffer,
                0,
                &bigger,
                0,
                self.capacity as u64 * std::mem::size_of::<T>() as u64,
            );
            queue.submit([encoder.finish()]);
            self.buffer = bigger;
            self.capacity = capacity;
        }
        start
    }

    fn write(&self, queue: &wgpu::Queue, start: u32, items: &[T]) {
        if !items.is_empty() {
            queue.write_buffer(
                &self.buffer,
                start as u64 * std::mem::size_of::<T>() as u64,
                bytemuck::cast_slice(items),
            );
        }
    }

    /// Writes `bytes` at `offset` bytes into the buffer, which has to be in room handed out.
    fn write_bytes(&self, queue: &wgpu::Queue, offset: u64, bytes: &[u8]) {
        if !bytes.is_empty() {
            queue.write_buffer(&self.buffer, offset, bytes);
        }
    }

    fn free(&mut self, start: u32, count: u32) {
        self.ranges.free(start, count);
    }

    /// The buffer, for binding. It is replaced when it grows: bind it afresh each frame.
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    /// How many `T`s are in use.
    pub fn used(&self) -> u32 {
        self.ranges.used()
    }

    /// How many `T`s it has room for.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }
}

/// Where a mesh's pieces are in the pool, to give them back.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MeshPlace {
    record: GpuMesh,
    cluster_vertices: (u32, u32),
    triangles: (u32, u32),
    /// Its morph targets' offsets, in vertex slots, and how many targets, if it bends.
    morphs: Option<(u32, u32, u32)>,
}

/// How many vertex slots `words` words take.
fn slots_for(words: u32) -> u32 {
    words.div_ceil(VERTEX_FLOATS)
}

/// What is in a pool, for statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PoolStats {
    /// How many meshes are in it.
    pub meshes: u32,
    /// How many instances are placed.
    pub instances: u32,
    /// How many vertices, clusters and triangles its meshes have.
    pub vertices: u32,
    /// How many clusters its meshes have, at every level of detail.
    pub clusters: u32,
    /// How many triangles its meshes' clusters have, at every level of detail.
    pub triangles: u32,
    /// How much GPU memory its buffers take.
    pub bytes: u64,
}

/// Every cluster of every mesh, and every instance of them, on the GPU.
pub struct GeometryPool {
    /// Every mesh's vertices.
    pub vertices: PoolBuffer<PoolVertex>,
    /// Each cluster's vertices, as indices into [`Self::vertices`].
    pub cluster_vertices: PoolBuffer<u32>,
    /// Each cluster's triangles, as indices into its own vertices, a byte each in a word.
    pub triangles: PoolBuffer<u32>,
    /// Every mesh's clusters.
    pub clusters: PoolBuffer<GpuCluster>,
    /// Every mesh.
    pub meshes: PoolBuffer<GpuMesh>,
    /// Every instance, by slot; free slots are marked so.
    pub instances: PoolBuffer<GpuInstance>,
    /// Each mesh slot's place, while it is in use.
    places: Vec<Option<MeshPlace>>,
    /// Each instance slot's record, while it is in use.
    placed: Vec<Option<GpuInstance>>,
    /// Each instance slot's morph block while it is in use and bends.
    blocks: Vec<Option<Block>>,
    /// Instance slots whose weights changed since the last [`GeometryPool::flush`].
    changed: Vec<u32>,
}

/// An instance's morph block (see [`GpuInstance::morph`]): where it is, in vertex slots, how
/// many, and its words as last set.
#[derive(Debug, Clone, PartialEq)]
struct Block {
    at: u32,
    slots: u32,
    words: Vec<u32>,
}

impl GeometryPool {
    /// An empty pool, with room for about `vertices` vertices to begin with; it grows as it
    /// needs to.
    pub fn new(device: &wgpu::Device, vertices: u32) -> Self {
        let none = wgpu::BufferUsages::empty();
        Self {
            vertices: PoolBuffer::new(device, "YoctiteVertices", vertices, none),
            cluster_vertices: PoolBuffer::new(device, "YoctiteClusterVertices", vertices, none),
            triangles: PoolBuffer::new(device, "YoctiteTriangles", vertices, none),
            clusters: PoolBuffer::new(device, "YoctiteClusters", vertices / 64 + 1, none),
            meshes: PoolBuffer::new(device, "YoctiteMeshes", 64, none),
            instances: PoolBuffer::new(device, "YoctiteInstances", 1024, none),
            places: Vec::new(),
            placed: Vec::new(),
            blocks: Vec::new(),
            changed: Vec::new(),
        }
    }

    /// Uploads a mesh: its `vertices`, and its clusters at every level of detail, `dag`, whose
    /// clusters index those vertices.
    pub fn add_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dag: &ClusterDag,
        vertices: &[PoolVertex],
    ) -> MeshId {
        self.add_morphing_mesh(device, queue, dag, vertices, None)
    }

    /// Uploads a mesh that bends: as [`Self::add_mesh`], with how its morph targets move its
    /// `vertices` (one for each, so for the graph's extra vertices too: see
    /// [`ClusterDag::extend_vertices`]). Without targets, or with offsets not one for each
    /// vertex and target, it is uploaded as one that does not bend.
    pub fn add_morphing_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        dag: &ClusterDag,
        vertices: &[PoolVertex],
        morphs: Option<Morphs>,
    ) -> MeshId {
        let morphs = morphs.filter(|m| {
            let fits = m.offsets.len() == vertices.len() * m.targets as usize;
            if !fits {
                fyrox_core::log::Log::warn(format!(
                    "Yoctite: {} morph offsets for {} vertices and {} targets; the mesh is drawn unbent",
                    m.offsets.len(),
                    vertices.len(),
                    m.targets
                ));
            }
            m.targets > 0 && fits
        });
        let first_vertex = self.vertices.allocate(device, queue, vertices.len() as u32);
        self.vertices.write(queue, first_vertex, vertices);

        // How far each vertex can be moved, every target at its most.
        let reach: Vec<f32> = match morphs {
            Some(m) => m
                .offsets
                .chunks_exact(m.targets as usize)
                .map(|targets| {
                    m.most_weight.abs()
                        * targets
                            .iter()
                            .map(|o| (o[0] * o[0] + o[1] * o[1] + o[2] * o[2]).sqrt())
                            .sum::<f32>()
                })
                .collect(),
            None => Vec::new(),
        };
        let morph_place = morphs.map(|m| {
            let words: Vec<u32> = m
                .offsets
                .iter()
                .flat_map(|o| {
                    let half = |v: f32| half::f16::from_f32(v).to_bits() as u32;
                    [
                        half(o[0]) | half(o[1]) << 16,
                        half(o[2]) | half(o[3]) << 16,
                        half(o[4]) | half(o[5]) << 16,
                    ]
                })
                .collect();
            let slots = slots_for(words.len() as u32);
            let start = self.vertices.allocate(device, queue, slots);
            self.vertices.write_bytes(queue, start as u64 * VERTEX_BYTES, bytemuck::cast_slice(&words));
            (start, slots, m.targets)
        });

        let vertex_total: u32 = dag.clusters.iter().map(|c| c.vertices.len() as u32).sum();
        let triangle_total: u32 = dag.clusters.iter().map(|c| c.triangles.len() as u32).sum();
        let first_cluster_vertex = self.cluster_vertices.allocate(device, queue, vertex_total);
        let first_triangle = self.triangles.allocate(device, queue, triangle_total);
        let first_cluster = self.clusters.allocate(device, queue, dag.clusters.len() as u32);
        let mesh = self.meshes.allocate(device, queue, 1);

        let mut cluster_vertices = Vec::with_capacity(vertex_total as usize);
        let mut triangles = Vec::with_capacity(triangle_total as usize);
        let mut records = Vec::with_capacity(dag.clusters.len());
        let sphere = |s: &yoctite::Sphere| [s.center[0], s.center[1], s.center[2], s.radius];
        let mut bounds: Option<yoctite::Sphere> = None;
        let mut most_reach = 0.0f32;
        for cluster in &dag.clusters {
            // Grown by as far as its vertices can be bent out of them.
            let grown = cluster
                .vertices
                .iter()
                .filter_map(|&v| reach.get(v as usize))
                .fold(0.0f32, |a, &b| a.max(b));
            most_reach = most_reach.max(grown);
            let mut grown_bounds = sphere(&cluster.bounds);
            grown_bounds[3] += grown;
            records.push(GpuCluster {
                first_vertex: first_cluster_vertex + cluster_vertices.len() as u32,
                first_triangle: first_triangle + triangles.len() as u32,
                vertex_count: cluster.vertices.len() as u32,
                triangle_count: cluster.triangles.len() as u32,
                bounds: grown_bounds,
                lod_bounds: sphere(&cluster.lod_bounds),
                parent_bounds: sphere(&cluster.parent_bounds),
                error: cluster.error,
                // Infinite in the builder; the largest float here, which every GPU compares the
                // same way.
                parent_error: cluster.parent_error.min(f32::MAX),
                mesh,
                level: cluster.level,
            });
            cluster_vertices.extend(cluster.vertices.iter().map(|&v| first_vertex + v));
            triangles.extend(cluster.triangles.iter().map(|t| pack_triangle(*t)));
            if cluster.is_root() {
                bounds = Some(bounds.map_or(cluster.lod_bounds, |b| b.enclosing(cluster.lod_bounds)));
            }
        }
        self.cluster_vertices.write(queue, first_cluster_vertex, &cluster_vertices);
        self.triangles.write(queue, first_triangle, &triangles);
        self.clusters.write(queue, first_cluster, &records);
        let mut bounds = sphere(&bounds.unwrap_or_default());
        bounds[3] += most_reach;
        let record = GpuMesh {
            first_cluster,
            cluster_count: dag.clusters.len() as u32,
            first_vertex,
            vertex_count: vertices.len() as u32,
            bounds,
        };
        self.meshes.write(queue, mesh, &[record]);

        let slot = mesh as usize;
        if self.places.len() <= slot {
            self.places.resize(slot + 1, None);
        }
        self.places[slot] = Some(MeshPlace {
            record,
            cluster_vertices: (first_cluster_vertex, vertex_total),
            triangles: (first_triangle, triangle_total),
            morphs: morph_place,
        });
        MeshId(mesh)
    }

    /// Takes a mesh out, giving back its room. Its instances have to be removed first, or they
    /// draw whatever takes its place.
    pub fn remove_mesh(&mut self, queue: &wgpu::Queue, id: MeshId) {
        let Some(place) = self.places.get_mut(id.0 as usize).and_then(Option::take) else {
            return;
        };
        let r = place.record;
        self.vertices.free(r.first_vertex, r.vertex_count);
        if let Some((start, slots, _)) = place.morphs {
            self.vertices.free(start, slots);
        }
        self.cluster_vertices.free(place.cluster_vertices.0, place.cluster_vertices.1);
        self.triangles.free(place.triangles.0, place.triangles.1);
        self.clusters.free(r.first_cluster, r.cluster_count);
        self.meshes.write(queue, id.0, &[GpuMesh::default()]);
        self.meshes.free(id.0, 1);
    }

    /// A mesh's record, while it is in the pool.
    pub fn mesh(&self, id: MeshId) -> Option<GpuMesh> {
        self.places.get(id.0 as usize).copied().flatten().map(|p| p.record)
    }

    /// Places a copy of `mesh` at `transform` (three rows of four, from its space to the world),
    /// drawn with the material in slot `material`.
    pub fn add_instance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: MeshId,
        transform: [f32; 12],
        material: u32,
    ) -> InstanceId {
        let slot = self.instances.allocate(device, queue, 1);
        // A copy of a mesh that bends gets a block of its own: the mesh's morphs, where to find
        // them, and its weights, all at nothing to begin with.
        let place = self.places.get(mesh.0 as usize).copied().flatten();
        let block = place.and_then(|p| p.morphs.map(|m| (p.record.first_vertex, m))).map(|(first, (start, _, targets))| {
            let slots = slots_for(MORPH_HEADER + targets);
            let at = self.vertices.allocate(device, queue, slots);
            let mut words = vec![0u32; (slots * VERTEX_FLOATS) as usize];
            words[0] = first;
            words[1] = start * VERTEX_FLOATS;
            words[2] = targets;
            self.vertices.write_bytes(queue, at as u64 * VERTEX_BYTES, bytemuck::cast_slice(&words));
            Block { at, slots, words }
        });
        let record = GpuInstance {
            transform,
            mesh: mesh.0,
            flags: INSTANCE_ALIVE,
            material,
            morph: block.as_ref().map_or(NO_MORPH, |b| b.at * VERTEX_FLOATS),
        };
        self.instances.write(queue, slot, &[record]);
        if self.placed.len() <= slot as usize {
            self.placed.resize(slot as usize + 1, None);
            self.blocks.resize(slot as usize + 1, None);
        }
        self.placed[slot as usize] = Some(record);
        self.blocks[slot as usize] = block;
        InstanceId(slot)
    }

    /// Sets the weights an instance of a mesh that bends gives its morph targets, in order; past
    /// the mesh's own targets they are left out. An instance of a mesh that does not bend is
    /// left as it is. They reach the GPU at the next [`Self::flush`]: a forest sets thousands a
    /// frame, and a write each would cost more than the drawing.
    pub fn set_weights(&mut self, id: InstanceId, weights: &[f32]) {
        let Some(Some(block)) = self.blocks.get_mut(id.0 as usize) else {
            return;
        };
        let targets = block.words[2] as usize;
        let weights = &weights[..weights.len().min(targets)];
        let room = &mut block.words[MORPH_HEADER as usize..MORPH_HEADER as usize + weights.len()];
        let words: &[u32] = bytemuck::cast_slice(weights);
        if room != words {
            room.copy_from_slice(words);
            self.changed.push(id.0);
        }
    }

    /// Writes the weights set since last time to the GPU: the blocks that changed, those next
    /// to each other in the buffer written together.
    pub fn flush(&mut self, queue: &wgpu::Queue) {
        if self.changed.is_empty() {
            return;
        }
        let mut changed = std::mem::take(&mut self.changed);
        changed.sort_unstable_by_key(|&slot| self.blocks[slot as usize].as_ref().map_or(0, |b| b.at));
        changed.dedup();
        let mut run: Vec<u32> = Vec::new();
        let mut run_start = 0;
        let mut run_end = 0;
        for slot in changed.drain(..) {
            let Some(block) = self.blocks.get(slot as usize).and_then(Option::as_ref) else {
                continue;
            };
            if run.is_empty() || block.at != run_end {
                if !run.is_empty() {
                    self.vertices.write_bytes(queue, run_start as u64 * VERTEX_BYTES, bytemuck::cast_slice(&run));
                    run.clear();
                }
                run_start = block.at;
            } else if block.at < run_end {
                // The same block twice: already in the run.
                continue;
            }
            run.extend_from_slice(&block.words);
            run_end = block.at + block.slots;
        }
        if !run.is_empty() {
            self.vertices.write_bytes(queue, run_start as u64 * VERTEX_BYTES, bytemuck::cast_slice(&run));
        }
        self.changed = changed;
    }

    /// Moves an instance.
    pub fn move_instance(&mut self, queue: &wgpu::Queue, id: InstanceId, transform: [f32; 12]) {
        if let Some(Some(record)) = self.placed.get_mut(id.0 as usize) {
            record.transform = transform;
            self.instances.write(queue, id.0, &[*record]);
        }
    }

    /// Takes an instance out.
    pub fn remove_instance(&mut self, queue: &wgpu::Queue, id: InstanceId) {
        if let Some(slot) = self.placed.get_mut(id.0 as usize) {
            if slot.take().is_some() {
                self.instances.write(queue, id.0, &[GpuInstance::default()]);
                self.instances.free(id.0, 1);
                if let Some(block) = self.blocks.get_mut(id.0 as usize).and_then(Option::take) {
                    self.vertices.free(block.at, block.slots);
                }
            }
        }
    }

    /// An instance's record, while it is in the pool.
    pub fn instance(&self, id: InstanceId) -> Option<GpuInstance> {
        self.placed.get(id.0 as usize).copied().flatten()
    }

    /// Every instance placed, by slot.
    pub fn placed(&self) -> impl Iterator<Item = (u32, GpuInstance)> + '_ {
        self.placed
            .iter()
            .enumerate()
            .filter_map(|(slot, record)| record.map(|record| (slot as u32, record)))
    }

    /// How many instance slots there are, in use or not: what a pass over every instance covers.
    pub fn instance_slots(&self) -> u32 {
        self.instances.ranges.high_water()
    }

    /// What is in it.
    pub fn stats(&self) -> PoolStats {
        let bytes = |capacity: u32, size: usize| capacity as u64 * size as u64;
        PoolStats {
            meshes: self.places.iter().flatten().count() as u32,
            instances: self.placed.iter().flatten().count() as u32,
            vertices: self.vertices.used(),
            clusters: self.clusters.used(),
            triangles: self.triangles.used(),
            bytes: bytes(self.vertices.capacity(), std::mem::size_of::<PoolVertex>())
                + bytes(self.cluster_vertices.capacity(), 4)
                + bytes(self.triangles.capacity(), 4)
                + bytes(self.clusters.capacity(), std::mem::size_of::<GpuCluster>())
                + bytes(self.meshes.capacity(), std::mem::size_of::<GpuMesh>())
                + bytes(self.instances.capacity(), std::mem::size_of::<GpuInstance>()),
        }
    }
}

/// A triangle's three indices into its cluster's vertices, a byte each, as one word.
pub fn pack_triangle(t: [u8; 3]) -> u32 {
    t[0] as u32 | (t[1] as u32) << 8 | (t[2] as u32) << 16
}

const VIEW_SHADER: &str = include_str!("shaders/yoctite_view.wgsl");

/// Draws clusters straight from a [`GeometryPool`], each in a colour of its own, lit by a light
/// from over the camera's shoulder, into a texture the frame's size: to see the clusters, and
/// which levels of detail are drawn.
pub struct ClusterView {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    colour: Option<GpuTexture>,
    depth: Option<(wgpu::Texture, u32, u32)>,
}

/// The view's colour target's format.
const VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
const VIEW_DEPTH: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

impl ClusterView {
    /// The view's pipeline, made once.
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteView"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage(1),
                storage(2),
                storage(3),
                storage(4),
                storage(5),
                storage(6),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("YoctiteView"),
            source: wgpu::ShaderSource::Wgsl(VIEW_SHADER.into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("YoctiteView"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("YoctiteView"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: VIEW_FORMAT,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            // Both sides: foliage cards are single sheets.
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: Some(wgpu::DepthStencilState {
                format: VIEW_DEPTH,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        Self {
            pipeline,
            layout,
            colour: None,
            depth: None,
        }
    }

    /// Draws `list` - each an instance slot and a cluster index in `pool`, chosen on the CPU -
    /// as seen through `view_projection` (column by column, with OpenGL's clip depth from -1 to
    /// 1, as the renderer makes it), into a `width` by `height` texture, which it returns. Where
    /// nothing is drawn it is clear, alpha 0.
    pub fn draw(
        &mut self,
        server: &WgpuGraphicsServer,
        pool: &GeometryPool,
        view_projection: [[f32; 4]; 4],
        width: u32,
        height: u32,
        list: &[[u32; 2]],
    ) -> Result<GpuTexture, FrameworkError> {
        self.render(server, pool, view_projection, width, height, Source::List(list))
    }

    /// As [`Self::draw`], but the clusters chosen on the GPU by `culler`, from `eye`, where an
    /// error over its distance as big as `threshold` is as much as may show: nothing is read
    /// back to choose them, and they are drawn by an indirect draw.
    #[allow(clippy::too_many_arguments)]
    pub fn draw_culled(
        &mut self,
        server: &WgpuGraphicsServer,
        pool: &GeometryPool,
        culler: &mut Culler,
        view_projection: [[f32; 4]; 4],
        eye: [f32; 3],
        threshold: f32,
        width: u32,
        height: u32,
    ) -> Result<GpuTexture, FrameworkError> {
        let parameters = CullParameters {
            view_projection,
            eye,
            threshold,
            // The view draws every material alike.
            material_slots: 1,
        };
        self.render(server, pool, view_projection, width, height, Source::Culled(culler, parameters))
    }

    fn render(
        &mut self,
        server: &WgpuGraphicsServer,
        pool: &GeometryPool,
        view_projection: [[f32; 4]; 4],
        width: u32,
        height: u32,
        source: Source,
    ) -> Result<GpuTexture, FrameworkError> {
        let (width, height) = (width.max(1), height.max(1));
        let matches = self.colour.as_ref().is_some_and(|t| {
            matches!(t.kind(), GpuTextureKind::Rectangle { width: w, height: h } if w == width as usize && h == height as usize)
        });
        if !matches {
            self.colour = Some(server.create_texture(GpuTextureDescriptor {
                name: "YoctiteView",
                kind: GpuTextureKind::Rectangle {
                    width: width as usize,
                    height: height as usize,
                },
                pixel_kind: PixelKind::RGBA8,
                ..Default::default()
            })?);
        }
        let device = &server.state.device;
        if self.depth.as_ref().is_none_or(|(_, w, h)| (*w, *h) != (width, height)) {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("YoctiteViewDepth"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: VIEW_DEPTH,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                view_formats: &[],
            });
            self.depth = Some((texture, width, height));
        }
        let colour = self.colour.clone().expect("made above");
        let Some(target) = colour.as_any().downcast_ref::<WgpuTexture>() else {
            return Err(FrameworkError::Custom("Yoctite view target".into()));
        };
        if target.wgpu_texture().format() != VIEW_FORMAT {
            return Err(FrameworkError::Custom(format!(
                "Yoctite view target is {:?}, not {VIEW_FORMAT:?}",
                target.wgpu_texture().format()
            )));
        }

        server.flush_active_pass();
        let mut encoder = server.frame_encoder.borrow_mut().take().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("YoctiteView"),
            })
        });
        // The list to draw: made here, or by the culler in the same encoder.
        let (draw_list, culler) = match source {
            Source::List(list) => {
                // Never empty: a binding needs something in it.
                let items: Vec<[u32; 2]> = if list.is_empty() { vec![[0, 0]] } else { list.to_vec() };
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("YoctiteDrawList"),
                    contents: bytemuck::cast_slice(&items),
                    usage: wgpu::BufferUsages::STORAGE,
                });
                (Some((buffer, list.len() as u32)), None)
            }
            Source::Culled(culler, parameters) => {
                culler.record(device, &mut encoder, pool, &parameters);
                (None, Some(&*culler))
            }
        };

        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteView"),
            contents: bytemuck::cast_slice(&view_projection),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
            wgpu::BindGroupEntry {
                binding,
                resource: buffer.as_entire_binding(),
            }
        }
        let list_buffer = match (&draw_list, culler) {
            (Some((buffer, _)), _) => buffer,
            (None, Some(culler)) => &culler.lists,
            (None, None) => unreachable!("one or the other"),
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteView"),
            layout: &self.layout,
            entries: &[
                entry(0, &uniforms),
                entry(1, pool.vertices.buffer()),
                entry(2, pool.cluster_vertices.buffer()),
                entry(3, pool.triangles.buffer()),
                entry(4, pool.clusters.buffer()),
                entry(5, pool.instances.buffer()),
                entry(6, list_buffer),
            ],
        });
        {
            let depth_view = self
                .depth
                .as_ref()
                .expect("made above")
                .0
                .create_view(&wgpu::TextureViewDescriptor::default());
            let colour_view = target
                .wgpu_texture()
                .create_view(&wgpu::TextureViewDescriptor::default());
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("YoctiteView"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &colour_view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            match (&draw_list, culler) {
                // Room for a whole cluster in every instance of the draw; a cluster with fewer
                // triangles makes the rest of its vertices nothing.
                (Some((_, count)), _) if *count > 0 => pass.draw(0..CLUSTER_VERTICES, 0..*count),
                (None, Some(culler)) => pass.draw_indirect(&culler.args, ALL_DRAW_OFFSET),
                _ => {}
            }
        }
        *server.frame_encoder.borrow_mut() = Some(encoder);
        Ok(colour)
    }
}

/// Where the clusters [`ClusterView`] draws come from.
enum Source<'a> {
    List(&'a [[u32; 2]]),
    Culled(&'a mut Culler, CullParameters),
}

/// How many vertices a cluster's draw has: three for every triangle a cluster can have.
const CLUSTER_VERTICES: u32 = yoctite::MAX_TRIANGLES as u32 * 3;

/// Where the camera is, for [`Culler`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CullParameters {
    /// From the world to clip space, column by column, with OpenGL's clip depth from -1 to 1.
    pub view_projection: [[f32; 4]; 4],
    /// Where the camera is, in the world.
    pub eye: [f32; 3],
    /// An error over its distance as big as may show: about a pixel's angle.
    pub threshold: f32,
    /// How many material slots instances are numbered in (see [`GpuInstance::material`]): the
    /// list is sorted by them, and each gets its own indirect draw.
    pub material_slots: u32,
}

/// What [`Culler`] chose, a frame or two ago.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct CullStats {
    /// Instances at least partly in view.
    pub instances: u32,
    /// Clusters chosen: more than were drawn if the list ran out of room.
    pub clusters: u32,
    /// Their triangles.
    pub triangles: u32,
    /// Whether the list ran out of room; it is made bigger for the frames after.
    pub overflowed: bool,
    /// Of the clusters chosen, how many - and how many triangles - were hidden behind the depth
    /// pyramid of the frame before, seen as it was (see [`DepthPyramid`]).
    pub hidden_clusters: u32,
    /// Their triangles.
    pub hidden_triangles: u32,
    /// Of the clusters chosen, how many - and how many triangles - were small enough on the
    /// screen to be drawn by the software rasterizer (see [`GBufferDrawer`]).
    pub software_clusters: u32,
    /// Their triangles.
    pub software_triangles: u32,
}

const CULL_SHADER: &str = include_str!("shaders/yoctite_cull.wgsl");
const PYRAMID_SHADER: &str = include_str!("shaders/yoctite_hzb.wgsl");
const PYRAMID_DOWN_SHADER: &str = include_str!("shaders/yoctite_hzb_down.wgsl");
const RASTER_SHADER: &str = include_str!("shaders/yoctite_raster.wgsl");
const RESOLVE_SHADER: &str = include_str!("shaders/yoctite_resolve.wgsl");

/// A depth pyramid (hierarchical Z) of a depth buffer: level by level, each texel the farthest
/// depth of the 2x2 below it, level 0 half the screen across - see `shaders/yoctite_hzb.wgsl`.
/// Something whose nearest point is farther than the pyramid's depths everywhere over its
/// rectangle on screen is hidden. Kept with the view and projection it was seen through.
pub struct DepthPyramid {
    from_depth: wgpu::ComputePipeline,
    down: wgpu::ComputePipeline,
    from_depth_layout: wgpu::BindGroupLayout,
    down_layout: wgpu::BindGroupLayout,
    /// The pyramid, the screen it was made for, and how many levels it has.
    texture: Option<(wgpu::Texture, [u32; 2], u32)>,
    view_projection: [[f32; 4]; 4],
    /// Whether it holds a depth buffer yet.
    built: bool,
}

impl DepthPyramid {
    /// The pyramid's pipelines, made once; the pyramid itself at its first build.
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::R32Float,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
            count: None,
        };
        let source = |sample_type| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Texture {
                sample_type,
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: None,
        };
        let from_depth_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctitePyramidFromDepth"),
            entries: &[source(wgpu::TextureSampleType::Depth), storage(1)],
        });
        let down_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctitePyramidDown"),
            entries: &[source(wgpu::TextureSampleType::Float { filterable: false }), storage(1)],
        });
        let pipeline = |label: &str, source: &str, entry: &str, layout: &wgpu::BindGroupLayout| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(layout)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        Self {
            from_depth: pipeline("YoctitePyramidFromDepth", PYRAMID_SHADER, "from_depth", &from_depth_layout),
            down: pipeline("YoctitePyramidDown", PYRAMID_DOWN_SHADER, "down", &down_layout),
            from_depth_layout,
            down_layout,
            texture: None,
            view_projection: [[0.0; 4]; 4],
            built: false,
        }
    }

    /// Builds the pyramid of `depth` (a depth or depth-stencil texture, the whole screen) as
    /// `view_projection` sees it, recorded into `encoder`. A depth buffer shaders cannot read
    /// leaves it without one.
    pub fn build(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        depth: &wgpu::Texture,
        view_projection: [[f32; 4]; 4],
    ) {
        if !depth.usage().contains(wgpu::TextureUsages::TEXTURE_BINDING) {
            self.built = false;
            return;
        }
        let screen = [depth.width(), depth.height()];
        if self.texture.as_ref().is_none_or(|(_, made_for, _)| *made_for != screen) {
            let size = [screen[0].div_ceil(2).max(1), screen[1].div_ceil(2).max(1)];
            let levels = 32 - size[0].max(size[1]).leading_zeros();
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some("YoctitePyramid"),
                size: wgpu::Extent3d {
                    width: size[0],
                    height: size[1],
                    depth_or_array_layers: 1,
                },
                mip_level_count: levels,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            self.texture = Some((texture, screen, levels));
            self.built = false;
        }
        let (texture, _, levels) = self.texture.as_ref().expect("made above");
        let level = |n: u32| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                base_mip_level: n,
                mip_level_count: Some(1),
                ..Default::default()
            })
        };
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("YoctitePyramid"),
            timestamp_writes: None,
        });
        for n in 0..*levels {
            let target = level(n);
            let (pipeline, layout, source) = if n == 0 {
                (&self.from_depth, &self.from_depth_layout, depth_view.clone())
            } else {
                (&self.down, &self.down_layout, level(n - 1))
            };
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("YoctitePyramid"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&source),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&target),
                    },
                ],
            });
            let width = (texture.width() >> n).max(1);
            let height = (texture.height() >> n).max(1);
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &group, &[]);
            pass.dispatch_workgroups(width.div_ceil(8), height.div_ceil(8), 1);
        }
        self.view_projection = view_projection;
        self.built = true;
    }

    /// The whole pyramid, every level, with the view it was seen through, the screen's size and
    /// how many levels it has - once it has been built.
    pub fn view(&self) -> Option<(wgpu::TextureView, [[f32; 4]; 4], [u32; 2], u32)> {
        let (texture, screen, levels) = self.texture.as_ref().filter(|_| self.built)?;
        Some((texture.create_view(&Default::default()), self.view_projection, *screen, *levels))
    }
}

/// Where in [`Culler::args`] the draw of every cluster listed is, in bytes.
pub const ALL_DRAW_OFFSET: u64 = 16;

/// Where in [`Culler::args`] the draw of the clusters of material slot `material` is, in bytes.
pub fn material_draw_offset(material: u32) -> u64 {
    32 + 16 * material as u64
}

/// The culling's uniforms, as the shader reads them.
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CullUniforms {
    planes: [[f32; 4]; 6],
    eye: [f32; 3],
    threshold: f32,
    instance_slots: u32,
    capacity: u32,
    cluster_vertices: u32,
    material_slots: u32,
    pyramid_view_projection: [[f32; 4]; 4],
    screen_size: [f32; 2],
    pyramid_levels: u32,
    pyramid_valid: u32,
    phase: u32,
    material_stride: u32,
    software_pixels: f32,
    _padding: u32,
}

/// The six planes of the view of `view_projection` (column by column, OpenGL's clip space),
/// facing in, each normal of unit length.
pub fn frustum_planes(view_projection: &[[f32; 4]; 4]) -> [[f32; 4]; 6] {
    let row = |i: usize| [0, 1, 2, 3].map(|j| view_projection[j][i]);
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    let combine = |a: [f32; 4], b: [f32; 4], sign: f32| [0, 1, 2, 3].map(|k| a[k] + sign * b[k]);
    [
        combine(r3, r0, 1.0),
        combine(r3, r0, -1.0),
        combine(r3, r1, 1.0),
        combine(r3, r1, -1.0),
        combine(r3, r2, 1.0),
        combine(r3, r2, -1.0),
    ]
    .map(|p| {
        let length = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if length > 0.0 {
            p.map(|v| v / length)
        } else {
            p
        }
    })
}

/// Where the read-back of the culling's counts is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Readback {
    /// Free: the next culling copies its counts into it.
    Idle,
    /// A culling's counts were copied into it; mapped at the next.
    Copied,
    /// Waiting for the GPU to hand it over.
    Mapping,
}

/// Chooses on the GPU which clusters of which instances to draw: instances and clusters out of
/// view left out, and for each instance the cut through its clusters whose error looks no
/// bigger than the threshold (see `shaders/yoctite_cull.wgsl`), into a list sorted by material,
/// and the indirect arguments of a draw of each material's clusters, and of them all. Nothing is
/// read back to draw them; what it chose is read back a frame or two later, for statistics and
/// to make the list bigger when it runs out of room.
pub struct Culler {
    layout: wgpu::BindGroupLayout,
    args_layout: wgpu::BindGroupLayout,
    reset: wgpu::ComputePipeline,
    instances: wgpu::ComputePipeline,
    prepare: wgpu::ComputePipeline,
    clusters: wgpu::ComputePipeline,
    occluded: wgpu::ComputePipeline,
    offsets: wgpu::ComputePipeline,
    scatter: wgpu::ComputePipeline,
    counts: wgpu::Buffer,
    /// The indirect arguments: the cluster pass's dispatch, the draw of every cluster listed at
    /// [`ALL_DRAW_OFFSET`], and each material's draw at [`material_draw_offset`]. A draw is
    /// [`CLUSTER_VERTICES`] vertices, an instance a cluster.
    pub args: wgpu::Buffer,
    /// The clusters to draw, as instance slot and pool cluster: as listed from 0, and sorted by
    /// material from [`Self::capacity`] on.
    pub lists: wgpu::Buffer,
    /// Each material's range of the sorted list: count, start, and a cursor, a word each and one
    /// spare (see `MaterialRange` in the shader).
    pub materials: wgpu::Buffer,
    capacity: u32,
    material_capacity: u32,
    visible: wgpu::Buffer,
    visible_capacity: u32,
    readback: wgpu::Buffer,
    state: Readback,
    mapped: std::sync::Arc<std::sync::atomic::AtomicBool>,
    stats: Option<CullStats>,
    /// The depth pyramid of the frame before, which the culling tests clusters against; and a
    /// texel of one standing in before there is one.
    pub pyramid: DepthPyramid,
    no_pyramid: wgpu::TextureView,
    /// How small a cluster's radius on the screen, in pixels, it lists for the software
    /// rasterizer; nothing for none (see [`Self::set_software_pixels`]).
    software_pixels: f32,
}

/// How many clusters a draw list holds to begin with.
const FIRST_CAPACITY: u32 = 1 << 16;
/// The lists' bytes for each cluster they can hold: five lists - the first phase's as listed and
/// sorted, those put aside, the second phase's as listed and sorted - of two words each.
const LIST_BYTES: u64 = 7 * 8;
/// How many counters the culling keeps (`counts` in the shader), and their bytes.
const COUNTS: usize = 12;
const COUNTS_BYTES: u64 = COUNTS as u64 * 4;

fn storage_buffer(device: &wgpu::Device, label: &str, size: u64, more: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: size.max(16),
        // Copied from too, to read them back when looking into what was drawn.
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | more,
        mapped_at_creation: false,
    })
}

impl Culler {
    /// The culling's pipelines, made once.
    pub fn new(device: &wgpu::Device) -> Self {
        let buffer = |binding, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteCull"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                buffer(1, true),
                buffer(2, true),
                buffer(3, true),
                buffer(4, false),
                buffer(5, false),
                buffer(6, false),
                buffer(7, false),
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let args_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteCullArgs"),
            entries: &[buffer(0, false)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("YoctiteCull"),
            source: wgpu::ShaderSource::Wgsl(CULL_SHADER.into()),
        });
        // The passes that write the indirect arguments bind them; those dispatched from them, not.
        let with_args = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("YoctiteCullWithArgs"),
            bind_group_layouts: &[Some(&layout), Some(&args_layout)],
            immediate_size: 0,
        });
        let without_args = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("YoctiteCull"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        let pipeline = |entry: &str, layout: &wgpu::PipelineLayout| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let material_capacity = 64;
        Self {
            reset: pipeline("reset", &with_args),
            instances: pipeline("cull_instances", &without_args),
            prepare: pipeline("prepare", &with_args),
            clusters: pipeline("cull_clusters", &without_args),
            occluded: pipeline("cull_occluded", &without_args),
            offsets: pipeline("offsets", &with_args),
            scatter: pipeline("scatter", &without_args),
            layout,
            args_layout,
            counts: storage_buffer(device, "YoctiteCounts", COUNTS_BYTES, wgpu::BufferUsages::empty()),
            args: storage_buffer(
                device,
                "YoctiteArgs",
                material_draw_offset(2 * material_capacity),
                wgpu::BufferUsages::INDIRECT,
            ),
            lists: storage_buffer(device, "YoctiteLists", FIRST_CAPACITY as u64 * LIST_BYTES, wgpu::BufferUsages::empty()),
            materials: storage_buffer(
                device,
                "YoctiteMaterialRanges",
                2 * material_capacity as u64 * 16,
                wgpu::BufferUsages::empty(),
            ),
            capacity: FIRST_CAPACITY,
            material_capacity,
            visible: storage_buffer(device, "YoctiteVisible", 1024 * 4, wgpu::BufferUsages::empty()),
            visible_capacity: 1024,
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("YoctiteCountsRead"),
                size: COUNTS_BYTES,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: Readback::Idle,
            mapped: Default::default(),
            stats: None,
            pyramid: DepthPyramid::new(device),
            no_pyramid: device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some("YoctiteNoPyramid"),
                    size: wgpu::Extent3d {
                        width: 1,
                        height: 1,
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::R32Float,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                })
                .create_view(&Default::default()),
            software_pixels: 0.0,
        }
    }

    /// Lists clusters whose radius on the screen is `pixels` or less for the software rasterizer
    /// (in their own lists: from 5 and 6 times [`Self::capacity`], for the two phases), in the
    /// two-phase culling from now on; 0 for none, as at first.
    pub fn set_software_pixels(&mut self, pixels: f32) {
        self.software_pixels = pixels.max(0.0);
    }

    /// The counters, for the software rasterizer to read how many clusters it was given.
    pub(crate) fn counts(&self) -> &wgpu::Buffer {
        &self.counts
    }

    /// What it chose, as last read back: a frame or two behind.
    pub fn stats(&self) -> Option<CullStats> {
        self.stats
    }

    /// How many clusters each of its lists holds: where the sorted list starts in
    /// [`Self::lists`], counted in entries.
    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    /// Makes the lists hold `capacity` clusters, for the next culling.
    pub fn set_capacity(&mut self, device: &wgpu::Device, capacity: u32) {
        self.capacity = capacity.max(1);
        self.lists = storage_buffer(device, "YoctiteLists", self.capacity as u64 * LIST_BYTES, wgpu::BufferUsages::empty());
    }

    /// Takes in the counts read back, if they have arrived, and starts reading back the last
    /// culling's.
    fn poll(&mut self, device: &wgpu::Device) {
        use std::sync::atomic::Ordering;
        match self.state {
            Readback::Copied => {
                let mapped = self.mapped.clone();
                mapped.store(false, Ordering::Release);
                self.readback.slice(..).map_async(wgpu::MapMode::Read, move |result| {
                    if result.is_ok() {
                        mapped.store(true, Ordering::Release);
                    }
                });
                self.state = Readback::Mapping;
            }
            Readback::Mapping if self.mapped.load(Ordering::Acquire) => {
                let counts: [u32; COUNTS] = {
                    let data = self.readback.slice(..).get_mapped_range().expect("mapped");
                    bytemuck::pod_read_unaligned(&data[..COUNTS_BYTES as usize])
                };
                self.readback.unmap();
                let software = (counts[8] + counts[10], counts[9] + counts[11]);
                let stats = CullStats {
                    instances: counts[0],
                    clusters: counts[1] + counts[6] + software.0,
                    triangles: counts[2] + counts[7] + software.1,
                    overflowed: counts[3] != 0,
                    hidden_clusters: counts[4].saturating_sub(counts[6] + counts[10]),
                    hidden_triangles: counts[5].saturating_sub(counts[7] + counts[11]),
                    software_clusters: software.0,
                    software_triangles: software.1,
                };
                // Room for what it chose, and some to spare, from the next culling on: in the
                // first phase's lists, and in the clusters put aside.
                let needed = counts[1].max(counts[4]).max(counts[8]);
                if needed > self.capacity {
                    let wanted = (needed as u64 * 5 / 4).next_power_of_two().min(u32::MAX as u64 / 2) as u32;
                    self.set_capacity(device, wanted);
                }
                self.stats = Some(stats);
                self.state = Readback::Idle;
            }
            _ => {}
        }
    }

    /// Records the culling of `pool`'s instances as `parameters` see them into `encoder`: its
    /// lists and draw arguments are ready for draws recorded after it. In one phase, with no
    /// depths to hide anything.
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GeometryPool,
        parameters: &CullParameters,
    ) {
        self.record_phase(device, encoder, pool, parameters, 0, false);
        self.copy_counts(encoder);
    }

    /// Records the first of the two phases of culling with the depths of the frame before (see
    /// `shaders/yoctite_cull.wgsl`): what they hid put aside, the rest ready for the first set of
    /// draws (at [`material_draw_offset`] of each slot, the list sorted from [`Self::capacity`]).
    /// Then build [`Self::pyramid`] of what was drawn, and record [`Self::record_second`].
    pub fn record_first(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GeometryPool,
        parameters: &CullParameters,
    ) {
        self.record_phase(device, encoder, pool, parameters, 0, true);
    }

    /// Records the second phase: what the first put aside, tried against [`Self::pyramid`] as
    /// built since, and what it does not hide made ready for the second set of draws (at
    /// [`material_draw_offset`] of [`Self::material_stride`] plus each slot, the list sorted from
    /// four times [`Self::capacity`], its material ranges [`Self::material_stride`] on).
    pub fn record_second(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GeometryPool,
        parameters: &CullParameters,
    ) {
        self.record_phase(device, encoder, pool, parameters, 1, true);
        self.copy_counts(encoder);
    }

    /// How far on the second phase's material ranges and draws are, in slots.
    pub fn material_stride(&self) -> u32 {
        self.material_capacity
    }

    fn copy_counts(&mut self, encoder: &mut wgpu::CommandEncoder) {
        if self.state == Readback::Idle {
            encoder.copy_buffer_to_buffer(&self.counts, 0, &self.readback, 0, COUNTS_BYTES);
            self.state = Readback::Copied;
        }
    }

    fn record_phase(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GeometryPool,
        parameters: &CullParameters,
        phase: u32,
        occlusion: bool,
    ) {
        let slots = pool.instance_slots();
        let material_slots = parameters.material_slots.max(1);
        if phase == 0 {
            self.poll(device);
            if slots > self.visible_capacity {
                self.visible_capacity = slots.next_power_of_two();
                self.visible =
                    storage_buffer(device, "YoctiteVisible", self.visible_capacity as u64 * 4, wgpu::BufferUsages::empty());
            }
            if material_slots > self.material_capacity {
                self.material_capacity = material_slots.next_power_of_two();
                self.args = storage_buffer(
                    device,
                    "YoctiteArgs",
                    material_draw_offset(2 * self.material_capacity),
                    wgpu::BufferUsages::INDIRECT,
                );
                self.materials = storage_buffer(
                    device,
                    "YoctiteMaterialRanges",
                    2 * self.material_capacity as u64 * 16,
                    wgpu::BufferUsages::empty(),
                );
            }
        }
        // The first phase tests against the frame before's depths, the second against those
        // drawn since.
        let pyramid = self.pyramid.view().filter(|_| occlusion);
        let uniforms = CullUniforms {
            planes: frustum_planes(&parameters.view_projection),
            eye: parameters.eye,
            threshold: parameters.threshold,
            instance_slots: slots,
            capacity: self.capacity,
            cluster_vertices: CLUSTER_VERTICES,
            material_slots,
            pyramid_view_projection: pyramid.as_ref().map_or([[0.0; 4]; 4], |p| p.1),
            screen_size: pyramid.as_ref().map_or([1.0; 2], |p| [p.2[0] as f32, p.2[1] as f32]),
            pyramid_levels: pyramid.as_ref().map_or(1, |p| p.3),
            pyramid_valid: pyramid.is_some() as u32,
            phase,
            material_stride: self.material_capacity,
            // Only drawing into the G-Buffer, in two phases, is there a software rasterizer.
            software_pixels: if occlusion { self.software_pixels } else { 0.0 },
            _padding: 0,
        };
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteCull"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = [
            &uniforms,
            pool.clusters.buffer(),
            pool.meshes.buffer(),
            pool.instances.buffer(),
            &self.visible,
            &self.counts,
            &self.lists,
            &self.materials,
        ]
        .into_iter()
        .enumerate()
        .map(|(binding, buffer)| wgpu::BindGroupEntry {
            binding: binding as u32,
            resource: buffer.as_entire_binding(),
        })
        .collect::<Vec<_>>();
        let pyramid_view = pyramid.as_ref().map_or(&self.no_pyramid, |p| &p.0);
        entries.push(wgpu::BindGroupEntry {
            binding: 8,
            resource: wgpu::BindingResource::TextureView(pyramid_view),
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteCull"),
            layout: &self.layout,
            entries: &entries,
        });
        let args = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteCullArgs"),
            layout: &self.args_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: self.args.as_entire_binding(),
            }],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("YoctiteCull"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &bind_group, &[]);
        if phase == 0 {
            pass.set_pipeline(&self.reset);
            pass.set_bind_group(1, &args, &[]);
            pass.dispatch_workgroups(material_slots.div_ceil(64), 1, 1);
            pass.set_pipeline(&self.instances);
            pass.dispatch_workgroups(slots.div_ceil(64).max(1), 1, 1);
            pass.set_pipeline(&self.prepare);
            pass.set_bind_group(1, &args, &[]);
            pass.dispatch_workgroups(1, 1, 1);
            pass.set_pipeline(&self.clusters);
            pass.dispatch_workgroups_indirect(&self.args, 0);
        } else {
            pass.set_pipeline(&self.occluded);
            pass.dispatch_workgroups(self.capacity.div_ceil(64).min(65535), 1, 1);
        }
        pass.set_pipeline(&self.offsets);
        pass.set_bind_group(1, &args, &[]);
        pass.dispatch_workgroups(1, 1, 1);
        pass.set_pipeline(&self.scatter);
        pass.dispatch_workgroups(self.capacity.div_ceil(64).min(65535), 1, 1);
    }
}

const GBUFFER_SHADER: &str = include_str!("shaders/yoctite_gbuffer.wgsl");

/// What a material's draw into the G-Buffer reads besides its textures, as the shader reads it
/// (see `shaders/yoctite_gbuffer.wgsl`): the properties of the engine's standard and glTF
/// shaders that their G-Buffer pass uses.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialParams {
    /// Times the diffuse texture: red, green, blue and alpha, from 0 to 1.
    pub diffuse_color: [f32; 4],
    /// Times the emission texture.
    pub emission_strength: [f32; 3],
    /// Times the metallic texture's channel [`Self::metallic_channel`].
    pub metallic_factor: f32,
    /// Times the texture coordinates.
    pub tex_coord_scale: [f32; 2],
    /// Times the roughness texture's channel [`Self::roughness_channel`].
    pub roughness_factor: f32,
    /// The decal layer written into the G-Buffer's decal mask.
    pub layer_index: u32,
    /// Which channel of the metallic texture is metallic: 0 red to 3 alpha.
    pub metallic_channel: u32,
    /// Which channel of the roughness texture is roughness.
    pub roughness_channel: u32,
    /// The material's slot, which its instances are numbered with (see
    /// [`GpuInstance::material`]). Filled in by [`GBufferDrawer::draw`].
    pub slot: u32,
    /// To a multiple of sixteen bytes.
    pub _padding: u32,
}

impl Default for MaterialParams {
    fn default() -> Self {
        Self {
            diffuse_color: [1.0; 4],
            emission_strength: [0.0; 3],
            metallic_factor: 1.0,
            tex_coord_scale: [1.0; 2],
            roughness_factor: 1.0,
            layer_index: 0,
            metallic_channel: 0,
            roughness_channel: 0,
            slot: 0,
            _padding: 0,
        }
    }
}

/// A material for [`GBufferDrawer::draw`]: its slot, what it reads, and its textures, each with
/// its sampler: diffuse, normal, metallic, roughness, occlusion and emission, in that order.
#[derive(Clone)]
pub struct YoctiteMaterial {
    /// The slot its instances are numbered with.
    pub slot: u32,
    /// What it reads besides its textures.
    pub params: MaterialParams,
    /// Its textures and their samplers, in the order above.
    pub textures: [(GpuTexture, GpuSampler); 6],
}

/// The formats of the G-Buffer drawn into: a pipeline is made for each set of them.
#[derive(Debug, Clone, PartialEq)]
struct TargetFormats {
    colour: Vec<wgpu::TextureFormat>,
    depth: wgpu::TextureFormat,
}

/// Draws the clusters a [`Culler`] chose into the renderer's G-Buffer, a draw for each material,
/// as the engine draws its own deferred meshes: see `shaders/yoctite_gbuffer.wgsl`.
pub struct GBufferDrawer {
    frame_layout: wgpu::BindGroupLayout,
    material_layout: wgpu::BindGroupLayout,
    module: wgpu::ShaderModule,
    pipeline: Option<(TargetFormats, wgpu::RenderPipeline)>,
    /// The pipeline clearing a capture's rectangle (see [`Self::record_capture`]).
    capture_clear: Option<(TargetFormats, wgpu::RenderPipeline)>,
    /// The software rasterizer, where the device can run it.
    software: Option<SoftwareRaster>,
    /// How small a cluster's radius on the screen, in pixels, the software rasterizer draws it
    /// (where there is one): [`SOFTWARE_PIXELS`] at first; 0 draws everything by the hardware.
    pub software_pixels: f32,
}

/// What [`GBufferDrawer::draw`] drew.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct GBufferDraws {
    /// How many draws: one for each material, in each of the two phases.
    pub draw_calls: u32,
    /// Whether the software rasterizer drew the smallest clusters.
    pub software: bool,
}

impl GBufferDrawer {
    /// The drawer's layouts and shader, made once; its pipeline is made at the first draw, for
    /// the G-Buffer's formats.
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::VERTEX,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let frame_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteGBufferFrame"),
            entries: &[
                uniform(0, wgpu::ShaderStages::VERTEX),
                storage(1),
                storage(2),
                storage(3),
                storage(4),
                storage(5),
                storage(6),
                storage(7),
            ],
        });
        let mut entries = vec![uniform(0, wgpu::ShaderStages::VERTEX_FRAGMENT)];
        for binding in 1..=6 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            });
        }
        for binding in 7..=12 {
            entries.push(wgpu::BindGroupLayoutEntry {
                binding,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            });
        }
        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteGBufferMaterial"),
            entries: &entries,
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("YoctiteGBuffer"),
            source: wgpu::ShaderSource::Wgsl(GBUFFER_SHADER.into()),
        });
        Self {
            frame_layout,
            material_layout,
            module,
            pipeline: None,
            capture_clear: None,
            software: SoftwareRaster::new(device),
            software_pixels: SOFTWARE_PIXELS,
        }
    }

    /// Records `captures` into `colour` (diffuse, normal, ambient, material, decal mask) and
    /// `depth`: each capture's rectangle cleared, then its mesh's clusters drawn into it as the
    /// G-Buffer pass draws them, unbent, with the material of its slot from `materials`. What is
    /// outside the rectangles is left as it is. For capturing meshes as seen from given views:
    /// Candelite's surface cache.
    #[allow(clippy::too_many_arguments)]
    pub fn record_capture(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GeometryPool,
        captures: &[Capture],
        colour: &[wgpu::Texture],
        depth: &wgpu::Texture,
        materials: &[ResolvedMaterial],
    ) -> Result<(), FrameworkError> {
        if colour.len() != 5 {
            return Err(FrameworkError::Custom(format!(
                "Yoctite: capturing into {} colour targets, not 5",
                colour.len()
            )));
        }
        if captures.is_empty() {
            return Ok(());
        }
        let formats = TargetFormats {
            colour: colour.iter().map(|t| t.format()).collect(),
            depth: depth.format(),
        };
        let pipeline = self.pipeline(device, &formats).clone();
        let clear = self.capture_clear(device, &formats).clone();

        // Each capture's mesh as an instance of its own, where its mesh's space is the world's,
        // drawn unbent; its clusters listed one after another; and a material range for each
        // capture at its slot: a draw each, the range's offset where its clusters start.
        let instances: Vec<GpuInstance> = captures
            .iter()
            .map(|capture| GpuInstance {
                transform: [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                mesh: capture.mesh.0,
                flags: INSTANCE_ALIVE,
                material: capture.slot,
                morph: NO_MORPH,
            })
            .collect();
        let mut list: Vec<[u32; 2]> = Vec::new();
        let mut firsts = Vec::with_capacity(captures.len());
        for (i, capture) in captures.iter().enumerate() {
            firsts.push(list.len() as u32);
            list.extend(capture.clusters.iter().map(|&c| [i as u32, c]));
        }
        if list.is_empty() {
            return Ok(());
        }
        let instances = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteCaptureInstances"),
            contents: bytemuck::cast_slice(&instances),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let list_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteCaptureList"),
            contents: bytemuck::cast_slice(&list),
            usage: wgpu::BufferUsages::STORAGE,
        });

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Frame {
            view_projection: [[f32; 4]; 4],
            sorted_start: u32,
            range_start: u32,
            _padding: [u32; 2],
        }
        let max_slot = captures.iter().map(|c| c.slot).max().unwrap_or(0) as usize;
        let colour_views: Vec<wgpu::TextureView> = colour.iter().map(|t| t.create_view(&Default::default())).collect();
        let depth_view = depth.create_view(&Default::default());
        let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = colour_views
            .iter()
            .map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        // Each capture's bind groups made before the pass.
        let mut drawn = Vec::with_capacity(captures.len());
        for (i, capture) in captures.iter().enumerate() {
            let Some(material) = materials.iter().find(|m| m.slot == capture.slot) else {
                continue;
            };
            let frame = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("YoctiteCaptureFrame"),
                contents: bytemuck::bytes_of(&Frame {
                    view_projection: capture.view_projection,
                    sorted_start: 0,
                    range_start: 0,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            // The capture's range at its slot: its clusters, from where they start in the list.
            let mut ranges = vec![[0u32; 4]; max_slot + 1];
            ranges[capture.slot as usize] = [capture.clusters.len() as u32, firsts[i], 0, 0];
            let ranges = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("YoctiteCaptureRanges"),
                contents: bytemuck::cast_slice(&ranges),
                usage: wgpu::BufferUsages::STORAGE,
            });
            let frame_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("YoctiteCaptureFrame"),
                layout: &self.frame_layout,
                entries: &[
                    entry(0, &frame),
                    entry(1, pool.vertices.buffer()),
                    entry(2, pool.cluster_vertices.buffer()),
                    entry(3, pool.triangles.buffer()),
                    entry(4, pool.clusters.buffer()),
                    entry(5, &instances),
                    entry(6, &list_buffer),
                    entry(7, &ranges),
                ],
            });
            let material_group = self.material_group(device, material);
            drawn.push((capture, frame_group, material_group));
        }

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("YoctiteCapture"),
            color_attachments: &attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: formats.depth.has_stencil_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        for (capture, frame_group, material_group) in &drawn {
            let [x, y, w, h] = capture.viewport;
            pass.set_viewport(x as f32, y as f32, w as f32, h as f32, 0.0, 1.0);
            pass.set_scissor_rect(x, y, w, h);
            pass.set_pipeline(&clear);
            pass.draw(0..3, 0..1);
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, frame_group, &[]);
            pass.set_bind_group(1, material_group, &[]);
            pass.draw(0..CLUSTER_VERTICES, 0..capture.clusters.len() as u32);
        }
        Ok(())
    }

    /// The bind group a material is drawn with: what it reads, and its textures.
    fn material_group(&self, device: &wgpu::Device, material: &ResolvedMaterial) -> wgpu::BindGroup {
        let params = MaterialParams {
            slot: material.slot,
            ..material.params
        };
        let uniform = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteMaterial"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let mut entries = vec![wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform.as_entire_binding(),
        }];
        for (i, view) in material.views.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 1 + i as u32,
                resource: wgpu::BindingResource::TextureView(view),
            });
        }
        for (i, sampler) in material.samplers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: 7 + i as u32,
                resource: wgpu::BindingResource::Sampler(sampler),
            });
        }
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteMaterial"),
            layout: &self.material_layout,
            entries: &entries,
        })
    }

    /// The pipeline that clears a capture's rectangle, for `formats`.
    fn capture_clear(&mut self, device: &wgpu::Device, formats: &TargetFormats) -> &wgpu::RenderPipeline {
        if self.capture_clear.as_ref().is_none_or(|(made, _)| made != formats) {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("YoctiteCaptureClear"),
                source: wgpu::ShaderSource::Wgsl(CAPTURE_CLEAR_SHADER.into()),
            });
            let targets: Vec<Option<wgpu::ColorTargetState>> = formats
                .colour
                .iter()
                .map(|&format| Some(wgpu::ColorTargetState::from(format)))
                .collect();
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("YoctiteCaptureClear"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &module,
                    entry_point: Some("fs_main"),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                primitive: Default::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: formats.depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Always),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview_mask: None,
                cache: None,
            });
            self.capture_clear = Some((formats.clone(), pipeline));
        }
        &self.capture_clear.as_ref().expect("made above").1
    }

    /// Whether the device can run the software rasterizer (see [`SOFTWARE_FEATURES`]).
    pub fn has_software_raster(&self) -> bool {
        self.software.is_some()
    }

    fn pipeline(&mut self, device: &wgpu::Device, formats: &TargetFormats) -> &wgpu::RenderPipeline {
        if self.pipeline.as_ref().is_none_or(|(had, _)| had != formats) {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("YoctiteGBuffer"),
                bind_group_layouts: &[Some(&self.frame_layout), Some(&self.material_layout)],
                immediate_size: 0,
            });
            let targets: Vec<Option<wgpu::ColorTargetState>> = formats
                .colour
                .iter()
                .map(|&format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect();
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("YoctiteGBuffer"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.module,
                    entry_point: Some("vs_main"),
                    buffers: &[],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &self.module,
                    entry_point: Some("fs_main"),
                    targets: &targets,
                    compilation_options: Default::default(),
                }),
                // As the engine's G-Buffer pass: faces seen from behind culled, counter-clockwise
                // in front.
                primitive: wgpu::PrimitiveState {
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode: Some(wgpu::Face::Back),
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: formats.depth,
                    depth_write_enabled: Some(true),
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: wgpu::MultisampleState::default(),
                multiview_mask: None,
                cache: None,
            });
            self.pipeline = Some((formats.clone(), pipeline));
        }
        &self.pipeline.as_ref().expect("made above").1
    }

    /// Culls `pool`'s instances as `parameters` see them with `culler`, and draws what it chose
    /// into `gbuffer` - the renderer's G-Buffer: depth and stencil, then diffuse, normal,
    /// ambient, material and decal mask - one draw for each of `materials`, on top of what is
    /// there. Instances numbered with a material slot not among `materials` are culled but not
    /// drawn.
    pub fn draw(
        &mut self,
        server: &WgpuGraphicsServer,
        pool: &GeometryPool,
        culler: &mut Culler,
        parameters: &CullParameters,
        gbuffer: &GpuFrameBuffer,
        materials: &[YoctiteMaterial],
    ) -> Result<GBufferDraws, FrameworkError> {
        let wgpu_texture = |texture: &GpuTexture| -> Result<wgpu::Texture, FrameworkError> {
            texture
                .as_any()
                .downcast_ref::<WgpuTexture>()
                .map(|t| t.wgpu_texture().clone())
                .ok_or_else(|| FrameworkError::Custom("Yoctite: a G-Buffer texture not of wgpu".into()))
        };
        let colour: Vec<wgpu::Texture> = gbuffer
            .color_attachments()
            .iter()
            .map(|a| wgpu_texture(&a.texture))
            .collect::<Result<_, _>>()?;
        let depth = wgpu_texture(
            &gbuffer
                .depth_attachment()
                .ok_or_else(|| FrameworkError::Custom("Yoctite: a G-Buffer with no depth".into()))?
                .texture,
        )?;
        let resolved = resolve_materials(materials)?;

        let device = &server.state.device;
        server.flush_active_pass();
        // When the renderer drew none of its own meshes into the G-Buffer - all of them
        // Yoctite's - its clear is still waiting for a draw: this pass applies it, or it would
        // draw over the last frame's G-Buffer, and the clear would wipe this frame's out at the
        // renderer's next draw into it.
        let clear = gbuffer
            .as_any()
            .downcast_ref::<WgpuFrameBuffer>()
            .map(WgpuFrameBuffer::take_pending_clear)
            .unwrap_or_default();
        let mut encoder = server.frame_encoder.borrow_mut().take().unwrap_or_else(|| {
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("YoctiteGBuffer"),
            })
        });
        let drawn = self.record(device, &mut encoder, pool, culler, parameters, &colour, &depth, clear, &resolved);
        *server.frame_encoder.borrow_mut() = Some(encoder);
        drawn
    }

    /// As [`Self::draw`], into `colour` (diffuse, normal, ambient, material, decal mask) and
    /// `depth`, recorded into `encoder`, applying `clear` first.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GeometryPool,
        culler: &mut Culler,
        parameters: &CullParameters,
        colour: &[wgpu::Texture],
        depth: &wgpu::Texture,
        clear: PendingClear,
        materials: &[ResolvedMaterial],
    ) -> Result<GBufferDraws, FrameworkError> {
        if colour.len() != 5 {
            return Err(FrameworkError::Custom(format!(
                "Yoctite: a G-Buffer of {} colour targets, not 5",
                colour.len()
            )));
        }
        let formats = TargetFormats {
            colour: colour.iter().map(|t| t.format()).collect(),
            depth: depth.format(),
        };
        let pipeline = self.pipeline(device, &formats).clone();
        // The smallest clusters drawn by the software rasterizer, where there is one and its
        // arrays of textures hold every material slot.
        let software = self.software.is_some()
            && self.software_pixels > 0.0
            && parameters.material_slots <= SOFTWARE_MATERIALS;
        culler.set_software_pixels(if software { self.software_pixels } else { 0.0 });
        // The first phase of culling, against the frame before's depths.
        culler.record_first(device, encoder, pool, parameters);

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Frame {
            view_projection: [[f32; 4]; 4],
            sorted_start: u32,
            range_start: u32,
            _padding: [u32; 2],
        }
        // A phase's draws read its sorted list and its material ranges.
        let frame_group = |culler: &Culler, sorted_start: u32, range_start: u32| {
            let frame = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("YoctiteGBufferFrame"),
                contents: bytemuck::bytes_of(&Frame {
                    view_projection: parameters.view_projection,
                    sorted_start,
                    range_start,
                    _padding: [0; 2],
                }),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("YoctiteGBufferFrame"),
                layout: &self.frame_layout,
                entries: &[
                    &frame,
                    pool.vertices.buffer(),
                    pool.cluster_vertices.buffer(),
                    pool.triangles.buffer(),
                    pool.clusters.buffer(),
                    pool.instances.buffer(),
                    &culler.lists,
                    &culler.materials,
                ]
                .into_iter()
                .enumerate()
                .map(|(binding, buffer)| wgpu::BindGroupEntry {
                    binding: binding as u32,
                    resource: buffer.as_entire_binding(),
                })
                .collect::<Vec<_>>(),
            })
        };
        let first_group = frame_group(culler, culler.capacity(), 0);

        // Each material's own bind group: what it reads, and its textures.
        let groups: Vec<(u32, wgpu::BindGroup)> = materials
            .iter()
            .filter(|m| m.slot < parameters.material_slots)
            .map(|material| (material.slot, self.material_group(device, material)))
            .collect();

        let colour_views: Vec<wgpu::TextureView> = colour
            .iter()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
            .collect();
        let depth_view = depth.create_view(&wgpu::TextureViewDescriptor::default());
        let has_stencil = formats.depth.has_stencil_aspect();
        // Each material's draw of a phase's clusters: the first applying `clear`, the second
        // drawing on what the first drew.
        let draw = |encoder: &mut wgpu::CommandEncoder,
                    clear: PendingClear,
                    frame_group: &wgpu::BindGroup,
                    args: &wgpu::Buffer,
                    first_slot: u32| {
            let (colour_load, depth_load, stencil_load) = clear.load_ops();
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = colour_views
                .iter()
                .map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: colour_load,
                            store: wgpu::StoreOp::Store,
                        },
                    })
                })
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("YoctiteGBuffer"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: depth_load,
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: has_stencil.then_some(wgpu::Operations {
                        load: stencil_load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&pipeline);
            pass.set_bind_group(0, frame_group, &[]);
            for (slot, group) in &groups {
                pass.set_bind_group(1, group, &[]);
                pass.draw_indirect(args, material_draw_offset(first_slot + *slot));
            }
        };
        draw(encoder, clear, &first_group, &culler.args, 0);
        let mut raster = |encoder: &mut wgpu::CommandEncoder, culler: &Culler, phase: u32| {
            if let (true, Some(raster)) = (software, self.software.as_mut()) {
                raster.record(
                    device,
                    encoder,
                    pool,
                    culler,
                    parameters.view_projection,
                    (5 + phase) * culler.capacity(),
                    8 + 2 * phase,
                    &colour_views,
                    depth,
                    &depth_view,
                    &formats,
                    materials,
                );
            }
        };
        raster(encoder, culler, 0);

        // The depths drawn so far as a pyramid; what the first phase put aside tried against it,
        // and what it does not hide drawn too. The pyramid stays for the next frame's first phase.
        culler.pyramid.build(device, encoder, depth, parameters.view_projection);
        culler.record_second(device, encoder, pool, parameters);
        let stride = culler.material_stride();
        let second_group = frame_group(culler, 4 * culler.capacity(), stride);
        draw(encoder, PendingClear::default(), &second_group, &culler.args, stride);
        raster(encoder, culler, 1);
        Ok(GBufferDraws {
            draw_calls: 2 * groups.len() as u32,
            software,
        })
    }
}

/// One mesh drawn into a rectangle of a set of targets by [`GBufferDrawer::record_capture`].
#[derive(Debug, Clone, PartialEq)]
pub struct Capture {
    /// The mesh.
    pub mesh: MeshId,
    /// Which of its clusters, as the pool numbers them: those of the full mesh, say.
    pub clusters: Vec<u32>,
    /// The slot of the material it is drawn with.
    pub slot: u32,
    /// From the mesh's space to clip space (column by column, OpenGL's).
    pub view_projection: [[f32; 4]; 4],
    /// Where it is drawn: x, y, width and height in pixels, from the top left.
    pub viewport: [u32; 4],
}

/// Clears a rectangle of the G-Buffer's targets: nothing drawn, at the farthest depth.
const CAPTURE_CLEAR_SHADER: &str = r"
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 1.0, 1.0);
}

struct Cleared {
    @location(0) colour: vec4f,
    @location(1) normal: vec4f,
    @location(2) ambient: vec4f,
    @location(3) material: vec4f,
    @location(4) decal: u32,
};

@fragment
fn fs_main() -> Cleared {
    return Cleared(vec4f(0.0), vec4f(0.0), vec4f(0.0), vec4f(0.0), 0u);
}
";

/// A bind group entry of a whole buffer.
fn entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

/// The most material slots the software rasterizer draws with: its arrays of textures are this
/// long (six to a slot for the resolve). With more slots in use, everything is drawn by the
/// hardware.
pub const SOFTWARE_MATERIALS: u32 = 256;

/// How small a cluster's radius on the screen, in pixels, is drawn by the software rasterizer, as
/// [`GBufferDrawer`] has it at first. In Wilderness's forest the G-Buffer took 7.7 ms with none,
/// 6.8 at 12, 5.3 at 32 and 48, and 5.6 with every cluster that could be: the hardware is still
/// the faster for triangles some pixels across. At 32, no triangle of a cluster can reach past
/// the rasterizer's limit of 4,096 pixels in its box.
pub const SOFTWARE_PIXELS: f32 = 32.0;

/// The GPU features the software rasterizer needs: 64-bit atomics for the visibility buffer, and
/// arrays of textures indexed per pixel.
pub const SOFTWARE_FEATURES: wgpu::Features = wgpu::Features::SHADER_INT64
    .union(wgpu::Features::SHADER_INT64_ATOMIC_MIN_MAX)
    .union(wgpu::Features::TEXTURE_BINDING_ARRAY)
    .union(wgpu::Features::SAMPLED_TEXTURE_AND_STORAGE_BUFFER_ARRAY_NON_UNIFORM_INDEXING);

/// How many workgroups the software rasterizer is dispatched with; each takes every so many
/// clusters of its list.
const RASTER_GROUPS: u32 = 8192;

/// Yoctite's software rasterizer and its visibility buffer: clusters too small on the screen for
/// the hardware - their triangles a pixel or two - drawn by a compute pass, a thread a triangle,
/// each pixel keeping the nearest triangle over it (`shaders/yoctite_raster.wgsl`); then shaded
/// into the G-Buffer from the triangle each pixel kept, with its depth, so the depth test against
/// what the hardware drew keeps whichever is nearer (`shaders/yoctite_resolve.wgsl`).
pub struct SoftwareRaster {
    raster_layout: wgpu::BindGroupLayout,
    raster_textures_layout: wgpu::BindGroupLayout,
    raster: wgpu::ComputePipeline,
    resolve_layout: wgpu::BindGroupLayout,
    resolve_textures_layout: wgpu::BindGroupLayout,
    resolve_module: wgpu::ShaderModule,
    resolve: Option<(TargetFormats, wgpu::RenderPipeline)>,
    /// A pixel each, 64 bits: see the shaders.
    visibility: Option<(wgpu::Buffer, [u32; 2])>,
    sampler: wgpu::Sampler,
    /// A white texel, for the slots of the arrays no material fills.
    blank: wgpu::TextureView,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct RasterUniforms {
    view_projection: [[f32; 4]; 4],
    screen_size: [u32; 2],
    list_start: u32,
    counted: u32,
    flip_v: u32,
    capacity: u32,
    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ResolveUniforms {
    view_projection: [[f32; 4]; 4],
    screen_size: [u32; 2],
    flip_v: u32,
    _padding: u32,
}

impl SoftwareRaster {
    /// The rasterizer, if `device` can run it: see [`SOFTWARE_FEATURES`]; and with room in its
    /// limits for the arrays of textures and the buffers bound.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        let limits = device.limits();
        if !device.features().contains(SOFTWARE_FEATURES)
            || limits.max_binding_array_elements_per_shader_stage < SOFTWARE_MATERIALS * 6
            || limits.max_storage_buffers_per_shader_stage < 9
        {
            return None;
        }
        let buffer = |binding, visibility, read_only| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform = |visibility| wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let textures = |binding, visibility, count: u32| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Texture {
                sample_type: wgpu::TextureSampleType::Float { filterable: true },
                view_dimension: wgpu::TextureViewDimension::D2,
                multisampled: false,
            },
            count: std::num::NonZeroU32::new(count),
        };
        let sampler = |binding, visibility| wgpu::BindGroupLayoutEntry {
            binding,
            visibility,
            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
            count: None,
        };
        let compute = wgpu::ShaderStages::COMPUTE;
        let raster_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteRaster"),
            entries: &[
                uniform(compute),
                buffer(1, compute, true),
                buffer(2, compute, true),
                buffer(3, compute, true),
                buffer(4, compute, true),
                buffer(5, compute, true),
                buffer(6, compute, true),
                buffer(7, compute, true),
                buffer(8, compute, false),
                buffer(9, compute, true),
            ],
        });
        // Arrays of textures may not share a bind group with a uniform buffer.
        let raster_textures_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteRasterTextures"),
            entries: &[textures(0, compute, SOFTWARE_MATERIALS), sampler(1, compute)],
        });
        let fragment = wgpu::ShaderStages::FRAGMENT;
        let resolve_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteResolve"),
            entries: &[
                uniform(fragment),
                buffer(1, fragment, true),
                buffer(2, fragment, true),
                buffer(3, fragment, true),
                buffer(4, fragment, true),
                buffer(5, fragment, true),
                buffer(6, fragment, true),
                buffer(7, fragment, true),
                buffer(8, fragment, true),
            ],
        });
        let resolve_textures_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("YoctiteResolveTextures"),
            entries: &[textures(0, fragment, SOFTWARE_MATERIALS * 6), sampler(1, fragment)],
        });
        let raster_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("YoctiteRaster"),
            source: wgpu::ShaderSource::Wgsl(RASTER_SHADER.into()),
        });
        let raster = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("YoctiteRaster"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("YoctiteRaster"),
                bind_group_layouts: &[Some(&raster_layout), Some(&raster_textures_layout)],
                immediate_size: 0,
            })),
            module: &raster_module,
            entry_point: Some("rasterize"),
            compilation_options: Default::default(),
            cache: None,
        });
        let resolve_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("YoctiteResolve"),
            source: wgpu::ShaderSource::Wgsl(RESOLVE_SHADER.into()),
        });
        let blank = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("YoctiteBlank"),
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        Some(Self {
            raster_layout,
            raster_textures_layout,
            raster,
            resolve_layout,
            resolve_textures_layout,
            resolve_module,
            resolve: None,
            visibility: None,
            sampler: device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("YoctiteSoftware"),
                address_mode_u: wgpu::AddressMode::Repeat,
                address_mode_v: wgpu::AddressMode::Repeat,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            }),
            blank: blank.create_view(&Default::default()),
        })
    }

    fn resolve_pipeline(&mut self, device: &wgpu::Device, formats: &TargetFormats) -> &wgpu::RenderPipeline {
        if self.resolve.as_ref().is_none_or(|(made, _)| made != formats) {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("YoctiteResolve"),
                bind_group_layouts: &[Some(&self.resolve_layout), Some(&self.resolve_textures_layout)],
                immediate_size: 0,
            });
            let targets: Vec<Option<wgpu::ColorTargetState>> = formats
                .colour
                .iter()
                .map(|&format| {
                    Some(wgpu::ColorTargetState {
                        format,
                        blend: None,
                        write_mask: wgpu::ColorWrites::ALL,
                    })
                })
                .collect();
            let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("YoctiteResolve"),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: &self.resolve_module,
                    entry_point: Some("vs_main"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState::default(),
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: formats.depth,
                    depth_write_enabled: Some(true),
                    // As the G-Buffer pass tests: the nearer of what the hardware drew and what
                    // the rasterizer did is kept.
                    depth_compare: Some(wgpu::CompareFunction::Less),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &self.resolve_module,
                    entry_point: Some("fs_main"),
                    compilation_options: Default::default(),
                    targets: &targets,
                }),
                multiview_mask: None,
                cache: None,
            });
            self.resolve = Some((formats.clone(), pipeline));
        }
        &self.resolve.as_ref().expect("made above").1
    }

    /// Records the software rasterizing of one phase's list - from `list_start` in `culler`'s
    /// lists, counted in counter `counted` - into the visibility buffer, cleared first, and its
    /// resolve into `colour` and `depth`: drawn on what is there.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        pool: &GeometryPool,
        culler: &Culler,
        view_projection: [[f32; 4]; 4],
        list_start: u32,
        counted: u32,
        colour_views: &[wgpu::TextureView],
        depth: &wgpu::Texture,
        depth_view: &wgpu::TextureView,
        formats: &TargetFormats,
        materials: &[ResolvedMaterial],
    ) {
        let size = [depth.width(), depth.height()];
        if self.visibility.as_ref().is_none_or(|(_, made)| *made != size) {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("YoctiteVisibility"),
                size: (size[0] as u64 * size[1] as u64 * 8).max(8),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            self.visibility = Some((buffer, size));
        }
        let visibility = &self.visibility.as_ref().expect("made above").0;
        encoder.clear_buffer(visibility, 0, None);

        // Each slot's parameters and textures, in arrays by slot.
        let mut params = vec![MaterialParams::default(); SOFTWARE_MATERIALS as usize];
        let mut diffuse: Vec<&wgpu::TextureView> = vec![&self.blank; SOFTWARE_MATERIALS as usize];
        let mut textures: Vec<&wgpu::TextureView> = vec![&self.blank; SOFTWARE_MATERIALS as usize * 6];
        for material in materials.iter().filter(|m| m.slot < SOFTWARE_MATERIALS) {
            let slot = material.slot as usize;
            params[slot] = MaterialParams {
                slot: material.slot,
                ..material.params
            };
            diffuse[slot] = &material.views[0];
            for (i, view) in material.views.iter().enumerate() {
                textures[slot * 6 + i] = view;
            }
        }
        let params = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteSoftwareMaterials"),
            contents: bytemuck::cast_slice(&params),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let raster_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteRaster"),
            contents: bytemuck::bytes_of(&RasterUniforms {
                view_projection,
                screen_size: size,
                list_start,
                counted,
                // wgpu's framebuffers are stored top row first.
                flip_v: 1,
                capacity: culler.capacity(),
                _padding: [0; 2],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let raster_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteRaster"),
            layout: &self.raster_layout,
            entries: &[
                entry(0, &raster_uniforms),
                entry(1, pool.vertices.buffer()),
                entry(2, pool.cluster_vertices.buffer()),
                entry(3, pool.triangles.buffer()),
                entry(4, pool.clusters.buffer()),
                entry(5, pool.instances.buffer()),
                entry(6, &culler.lists),
                entry(7, culler.counts()),
                entry(8, visibility),
                entry(9, &params),
            ],
        });
        let raster_textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteRasterTextures"),
            layout: &self.raster_textures_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&diffuse),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("YoctiteRaster"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.raster);
            pass.set_bind_group(0, &raster_group, &[]);
            pass.set_bind_group(1, &raster_textures, &[]);
            pass.dispatch_workgroups(RASTER_GROUPS.min(culler.capacity()).max(1), 1, 1);
        }

        let resolve_uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("YoctiteResolve"),
            contents: bytemuck::bytes_of(&ResolveUniforms {
                view_projection,
                screen_size: size,
                flip_v: 1,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let resolve_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteResolve"),
            layout: &self.resolve_layout,
            entries: &[
                entry(0, &resolve_uniforms),
                entry(1, pool.vertices.buffer()),
                entry(2, pool.cluster_vertices.buffer()),
                entry(3, pool.triangles.buffer()),
                entry(4, pool.clusters.buffer()),
                entry(5, pool.instances.buffer()),
                entry(6, &culler.lists),
                entry(7, visibility),
                entry(8, &params),
            ],
        });
        let resolve_textures = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("YoctiteResolveTextures"),
            layout: &self.resolve_textures_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureViewArray(&textures),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
            ],
        });
        let pipeline = self.resolve_pipeline(device, formats).clone();
        let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = colour_views
            .iter()
            .map(|view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })
            })
            .collect();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("YoctiteResolve"),
            color_attachments: &attachments,
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: formats.depth.has_stencil_aspect().then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &resolve_group, &[]);
        pass.set_bind_group(1, &resolve_textures, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// `materials` with their textures as wgpu's own.
pub fn resolve_materials(materials: &[YoctiteMaterial]) -> Result<Vec<ResolvedMaterial>, FrameworkError> {
    materials
        .iter()
        .map(|material| {
            let mut views = Vec::with_capacity(6);
            let mut samplers = Vec::with_capacity(6);
            for (texture, sampler) in &material.textures {
                let (Some(texture), Some(sampler)) = (
                    texture.as_any().downcast_ref::<WgpuTexture>(),
                    sampler.as_any().downcast_ref::<WgpuSampler>(),
                ) else {
                    return Err(FrameworkError::Custom("Yoctite: a material texture not of wgpu".into()));
                };
                views.push(texture.wgpu_view().clone());
                samplers.push(sampler.wgpu_sampler().clone());
            }
            Ok(ResolvedMaterial {
                slot: material.slot,
                params: material.params,
                views: views.try_into().expect("six"),
                samplers: samplers.try_into().expect("six"),
            })
        })
        .collect()
}

/// A material with its textures as wgpu's own: see [`GBufferDrawer::record`].
#[derive(Debug, Clone)]
pub struct ResolvedMaterial {
    /// The slot its instances are numbered with.
    pub slot: u32,
    /// What it reads besides its textures.
    pub params: MaterialParams,
    /// Its diffuse, normal, metallic, roughness, occlusion and emission textures.
    pub views: [wgpu::TextureView; 6],
    /// And their samplers.
    pub samplers: [wgpu::Sampler; 6],
}

/// The view's, the culling's, the G-Buffer draw's, the depth pyramid's and the software
/// rasterizer's shaders, for validation.
#[cfg(test)]
pub(crate) fn shaders() -> [&'static str; 7] {
    [VIEW_SHADER, CULL_SHADER, GBUFFER_SHADER, PYRAMID_SHADER, PYRAMID_DOWN_SHADER, RASTER_SHADER, RESOLVE_SHADER]
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn ranges_are_reused_and_joined() {
        let mut ranges = RangeAllocator::default();
        let a = ranges.allocate(10);
        let b = ranges.allocate(20);
        let c = ranges.allocate(5);
        assert_eq!((a, b, c), (0, 10, 30));
        ranges.free(b, 20);
        assert_eq!(ranges.allocate(15), 10, "the first range big enough");
        assert_eq!(ranges.allocate(5), 25, "what is left of it");
        assert_eq!(ranges.allocate(1), 35, "the end, once nothing free fits");
        ranges.free(a, 10);
        ranges.free(10, 15);
        ranges.free(25, 5);
        assert_eq!(ranges.free, vec![(0, 30)], "neighbours joined");
        ranges.free(c, 5);
        ranges.free(35, 1);
        assert_eq!((ranges.free.len(), ranges.high_water(), ranges.used()), (0, 0, 0));
    }

    #[test]
    fn a_free_end_grows_into_the_rest() {
        let mut ranges = RangeAllocator::default();
        let a = ranges.allocate(10);
        let b = ranges.allocate(10);
        let _c = ranges.allocate(10);
        ranges.free(b, 10);
        // Too big for the hole: from the end.
        assert_eq!(ranges.allocate(15), 30);
        assert_eq!(ranges.used(), 35);
        ranges.free(a, 10);
        assert_eq!(ranges.allocate(20), 0, "the two holes joined fit it");
    }

    #[test]
    fn records_are_laid_out_as_the_shaders_read_them() {
        assert_eq!(std::mem::size_of::<PoolVertex>(), 48);
        assert_eq!(std::mem::size_of::<GpuCluster>(), 80);
        assert_eq!(std::mem::size_of::<GpuMesh>(), 32);
        assert_eq!(std::mem::size_of::<GpuInstance>(), 64);
        assert_eq!(pack_triangle([1, 2, 255]), 0x00ff_0201);
    }

    #[test]
    fn the_shaders_are_valid() {
        for source in shaders() {
            let module = wgpu::naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|e| panic!("{}", e.emit_to_string(source)));
            wgpu::naga::valid::Validator::new(
                wgpu::naga::valid::ValidationFlags::all(),
                wgpu::naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
        }
    }

    /// A camera at `eye` looking at `at`, OpenGL's perspective: column by column.
    fn camera(eye: [f32; 3], at: [f32; 3], fov: f32, aspect: f32) -> [[f32; 4]; 4] {
        let sub = |a: [f32; 3], b: [f32; 3]| [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
        let cross = |a: [f32; 3], b: [f32; 3]| [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
        let dot = |a: [f32; 3], b: [f32; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let unit = |a: [f32; 3]| {
            let l = dot(a, a).sqrt();
            a.map(|v| v / l)
        };
        let f = unit(sub(at, eye));
        let r = unit(cross(f, [0.0, 1.0, 0.0]));
        let u = cross(r, f);
        // Rows of the view matrix.
        let view = [
            [r[0], r[1], r[2], -dot(r, eye)],
            [u[0], u[1], u[2], -dot(u, eye)],
            [-f[0], -f[1], -f[2], dot(f, eye)],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let (near, far) = (0.1, 1000.0);
        let t = 1.0 / (fov / 2.0).tan();
        let projection = [
            [t / aspect, 0.0, 0.0, 0.0],
            [0.0, t, 0.0, 0.0],
            [0.0, 0.0, (far + near) / (near - far), 2.0 * far * near / (near - far)],
            [0.0, 0.0, -1.0, 0.0],
        ];
        let mut out = [[0.0; 4]; 4];
        for i in 0..4 {
            for j in 0..4 {
                // Column j, row i.
                out[j][i] = (0..4).map(|k| projection[i][k] * view[k][j]).sum();
            }
        }
        out
    }

    #[test]
    fn planes_keep_what_is_in_view() {
        let view = camera([0.0, 0.0, 0.0], [0.0, 0.0, -1.0], 1.0, 1.0);
        let planes = frustum_planes(&view);
        let inside = |c: [f32; 3], r: f32| planes.iter().all(|p| p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3] >= -r);
        assert!(inside([0.0, 0.0, -10.0], 1.0));
        assert!(!inside([0.0, 0.0, 10.0], 1.0), "behind");
        assert!(!inside([50.0, 0.0, -10.0], 1.0), "off to the side");
        assert!(inside([11.0, 0.0, -10.0], 6.0), "partly in");
        assert!(!inside([0.0, 0.0, -2000.0], 1.0), "past the far plane");
    }

    /// A turn about the vertical by `yaw` and a scale, then a move: three rows of four.
    fn placed(yaw: f32, scale: f32, at: [f32; 3]) -> [f32; 12] {
        let (s, c) = yaw.sin_cos();
        [c * scale, 0.0, s * scale, at[0], 0.0, scale, 0.0, at[1], -s * scale, 0.0, c * scale, at[2]]
    }

    /// The culling worked out on the CPU, as the shader does it: each chosen instance slot and
    /// pool cluster, and for each cluster how near any of the comparisons deciding it came to
    /// going the other way, as a share.
    fn cull_on_cpu(pool: &GeometryPool, dags: &[(MeshId, &ClusterDag)], parameters: &CullParameters) -> Vec<([u32; 2], bool, f32)> {
        let planes = frustum_planes(&parameters.view_projection);
        let eye = parameters.eye;
        let mut out = Vec::new();
        for slot in 0..pool.instance_slots() {
            let Some(instance) = pool.instance(InstanceId(slot)) else { continue };
            let t = instance.transform;
            let place = |p: [f32; 3]| [0, 1, 2].map(|r| t[r * 4] * p[0] + t[r * 4 + 1] * p[1] + t[r * 4 + 2] * p[2] + t[r * 4 + 3]);
            let scale = (0..3)
                .map(|c| (t[c] * t[c] + t[4 + c] * t[4 + c] + t[8 + c] * t[8 + c]).sqrt())
                .fold(0.0, f32::max);
            // Whether a sphere is in view, and how near its nearest plane came to deciding
            // otherwise.
            let view = |c: [f32; 3], r: f32| {
                let mut nearest = f32::MAX;
                let mut inside = true;
                for p in &planes {
                    let d = p[0] * c[0] + p[1] * c[1] + p[2] * c[2] + p[3] + r;
                    nearest = nearest.min(d.abs() / (r + 1.0));
                    inside &= d >= 0.0;
                }
                (inside, nearest)
            };
            let projected = |e: f32, c: [f32; 3], r: f32| {
                if e <= 0.0 {
                    return 0.0;
                }
                let d = ((c[0] - eye[0]).powi(2) + (c[1] - eye[1]).powi(2) + (c[2] - eye[2]).powi(2)).sqrt() - r;
                if d <= 0.0 { 3.0e38 } else { e / d }
            };
            let mesh = pool.mesh(MeshId(instance.mesh)).unwrap();
            let b = mesh.bounds;
            if !view(place([b[0], b[1], b[2]]), b[3] * scale).0 {
                continue;
            }
            let dag = dags.iter().find(|(id, _)| id.0 == instance.mesh).unwrap().1;
            let th = parameters.threshold;
            for (i, c) in dag.clusters.iter().enumerate() {
                let own = projected(c.error * scale, place(c.lod_bounds.center), c.lod_bounds.radius * scale);
                let mut margin = ((own - th) / th).abs();
                let mut chosen = own <= th;
                if chosen && !c.is_root() {
                    let parent = projected(c.parent_error * scale, place(c.parent_bounds.center), c.parent_bounds.radius * scale);
                    margin = margin.min(((parent - th) / th).abs());
                    chosen = parent > th;
                }
                if chosen {
                    let (inside, near) = view(place(c.bounds.center), c.bounds.radius * scale);
                    margin = margin.min(near);
                    chosen = inside;
                }
                out.push(([slot, mesh.first_cluster + i as u32], chosen, margin));
            }
        }
        out
    }

    /// How many materials the culling tests' instances are drawn with.
    const MATERIALS: u32 = 5;

    #[test]
    fn the_gpu_chooses_the_clusters_the_cpu_does() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut pool = GeometryPool::new(&device, 1 << 12);
        let (vertices_a, dag_a) = hill(40, 6.0);
        let (vertices_b, dag_b) = hill(24, 2.0);
        let a = pool.add_mesh(&device, &queue, &dag_a, &vertices_a);
        let b = pool.add_mesh(&device, &queue, &dag_b, &vertices_b);
        let mut seed = 7u32;
        let mut random = || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            seed as f32 / u32::MAX as f32
        };
        let mut instances = Vec::new();
        for x in 0..30 {
            for z in 0..30 {
                let mesh = if (x + z) % 3 == 0 { b } else { a };
                let at = [x as f32 * 50.0 - 750.0, random() * 4.0, z as f32 * 50.0 - 750.0];
                let material = ((x * 7 + z) % MATERIALS as usize) as u32;
                instances.push(pool.add_instance(&device, &queue, mesh, placed(random() * std::f32::consts::TAU, 0.5 + random() * 1.5, at), material));
            }
        }
        // Some slots freed, which the culling has to pass over.
        for i in (0..instances.len()).step_by(17) {
            pool.remove_instance(&queue, instances[i]);
        }
        let dags = [(a, &dag_a), (b, &dag_b)];
        let mut culler = Culler::new(&device);
        let mut compared = 0;
        for (eye, at, threshold) in [
            ([0.0, 30.0, 0.0], [300.0, 0.0, 200.0], 0.001),
            ([0.0, 30.0, 0.0], [300.0, 0.0, 200.0], 0.0002),
            ([-700.0, 5.0, -700.0], [0.0, 0.0, 0.0], 0.001),
            ([10.0, 3.0, 10.0], [-200.0, 0.0, 30.0], 0.01),
        ] {
            let parameters = CullParameters {
                view_projection: camera(eye, at, 1.2, 16.0 / 9.0),
                eye,
                threshold,
                material_slots: MATERIALS,
            };
            let mut encoder = device.create_command_encoder(&Default::default());
            culler.record(&device, &mut encoder, &pool, &parameters);
            queue.submit([encoder.finish()]);
            let args: Vec<u32> = read(&device, &queue, &culler.args, (material_draw_offset(MATERIALS) / 4) as u32);
            let counts: Vec<u32> = read(&device, &queue, &culler.counts, 4);
            let all = (ALL_DRAW_OFFSET / 4) as usize;
            let drawn = args[all + 1];
            assert_eq!(args[all], CLUSTER_VERTICES);
            assert_eq!(drawn, counts[1], "every cluster chosen is drawn");
            assert_eq!(counts[3], 0, "the list held them all");
            let capacity = culler.capacity();
            let lists: Vec<[u32; 2]> = read(&device, &queue, &culler.lists, capacity * 2);
            let mut gpu: Vec<[u32; 2]> = lists[..drawn as usize].to_vec();
            gpu.sort_unstable();
            assert!(gpu.windows(2).all(|w| w[0] != w[1]), "a cluster listed twice");

            // Sorted by material: each material's draw covers its range, which holds its
            // instances' clusters and nothing else, and the ranges together are the list.
            let ranges: Vec<[u32; 4]> = read(&device, &queue, &culler.materials, MATERIALS);
            let mut sorted = Vec::new();
            for m in 0..MATERIALS {
                let draw = (material_draw_offset(m) / 4) as usize;
                let [count, offset, ..] = ranges[m as usize];
                assert_eq!((args[draw], args[draw + 1]), (CLUSTER_VERTICES, count), "material {m}'s draw");
                for item in &lists[(capacity + offset) as usize..(capacity + offset + count) as usize] {
                    assert_eq!(pool.instance(InstanceId(item[0])).unwrap().material, m, "{item:?} in material {m}'s range");
                    sorted.push(*item);
                }
            }
            sorted.sort_unstable();
            assert_eq!(sorted, gpu, "the sorted list is the list");

            let reference = cull_on_cpu(&pool, &dags, &parameters);
            let triangles: u32 = gpu
                .iter()
                .map(|&[slot, c]| {
                    let instance = pool.instance(InstanceId(slot)).unwrap();
                    let (id, dag) = dags.iter().find(|(id, _)| id.0 == instance.mesh).unwrap();
                    dag.clusters[(c - pool.mesh(*id).unwrap().first_cluster) as usize].triangles.len() as u32
                })
                .sum();
            assert_eq!(triangles, counts[2], "triangles counted as listed");
            let mut differ = 0;
            for (item, chosen, margin) in &reference {
                if gpu.binary_search(item).is_ok() != *chosen {
                    // Only where single precision on the GPU and the CPU could round apart.
                    assert!(*margin < 1e-4, "{item:?}: chosen on the CPU {chosen}, a margin of {margin}");
                    differ += 1;
                }
            }
            let chosen = reference.iter().filter(|r| r.1).count();
            assert!(chosen > 100, "only {chosen} clusters chosen from {eye:?}");
            assert!(differ * 1000 <= chosen, "{differ} of {chosen} differ");
            compared += chosen;
        }
        assert!(compared > 1000);
    }

    #[test]
    fn a_list_too_small_is_made_bigger() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut pool = GeometryPool::new(&device, 1 << 12);
        let (vertices, dag) = hill(40, 6.0);
        let mesh = pool.add_mesh(&device, &queue, &dag, &vertices);
        for i in 0..50 {
            pool.add_instance(&device, &queue, mesh, placed(0.0, 1.0, [i as f32 * 45.0, 0.0, -60.0]), 0);
        }
        let mut culler = Culler::new(&device);
        culler.set_capacity(&device, 10);
        let parameters = CullParameters {
            view_projection: camera([1000.0, 20.0, 40.0], [1000.0, 0.0, -60.0], 2.5, 1.0),
            eye: [1000.0, 20.0, 40.0],
            threshold: 0.0005,
            material_slots: 1,
        };
        let cull = |culler: &mut Culler| {
            let mut encoder = device.create_command_encoder(&Default::default());
            culler.record(&device, &mut encoder, &pool, &parameters);
            queue.submit([encoder.finish()]);
            device.poll(wgpu::PollType::wait_indefinitely()).unwrap();
        };
        cull(&mut culler);
        let all = (ALL_DRAW_OFFSET / 4) as usize;
        let args: Vec<u32> = read(&device, &queue, &culler.args, 12);
        assert_eq!(args[all + 1], 10, "no more drawn than the list holds");
        assert_eq!(args[8 + 1], 10, "nor by the material");
        // The counts arrive over the next cullings, and the list grows to hold them.
        for _ in 0..3 {
            cull(&mut culler);
        }
        let stats = culler.stats().expect("read back by now");
        assert!(stats.overflowed && stats.clusters > 10, "{stats:?}");
        assert!(culler.capacity() >= stats.clusters, "{} for {} clusters", culler.capacity(), stats.clusters);
        cull(&mut culler);
        let args: Vec<u32> = read(&device, &queue, &culler.args, 12);
        assert_eq!(args[all + 1], stats.clusters, "all drawn once it has room");
    }

    /// A 1x1 texture of one colour, and a sampler.
    fn swatch(device: &wgpu::Device, queue: &wgpu::Queue, rgba: [u8; 4]) -> (wgpu::TextureView, wgpu::Sampler) {
        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            Default::default(),
            &rgba,
        );
        (texture.create_view(&Default::default()), device.create_sampler(&Default::default()))
    }

    /// The pixels of a `size` by `size` texture of 4-byte texels (or 1-byte with `narrow`),
    /// read back, a texel a word.
    fn pixels(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, size: u32, narrow: bool) -> Vec<u32> {
        // A row's bytes, to the multiple of 256 a copy's rows are laid out in.
        let row = (size * if narrow { 1 } else { 4 }).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * size) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("mapped"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("polled");
        let data = buffer.slice(..).get_mapped_range().expect("mapped range");
        let mut out = Vec::new();
        for y in 0..size as usize {
            for x in 0..size as usize {
                let at = y * row as usize;
                out.push(if narrow {
                    data[at + x] as u32
                } else {
                    u32::from_le_bytes(data[at + x * 4..at + x * 4 + 4].try_into().unwrap())
                });
            }
        }
        out
    }

    #[test]
    fn clusters_are_drawn_into_the_gbuffer_as_the_engine_draws_meshes() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        const SIZE: u32 = 64;
        // The G-Buffer's own formats, as the renderer makes them.
        let target = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let colour = vec![
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::R8Uint),
        ];
        let depth = target(wgpu::TextureFormat::Depth24PlusStencil8);

        // Two flat squares side by side, seen from above: on the left one drawn, on the right one
        // whose diffuse texture is see-through all over.
        let mut pool = GeometryPool::new(&device, 1 << 12);
        let (vertices, dag) = hill(16, 0.0);
        let mesh = pool.add_mesh(&device, &queue, &dag, &vertices);
        pool.add_instance(&device, &queue, mesh, placed(0.0, 1.0, [-16.0, 0.0, 0.0]), 0);
        pool.add_instance(&device, &queue, mesh, placed(0.0, 1.0, [0.0, 0.0, 0.0]), 1);
        let white = swatch(&device, &queue, [255; 4]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let clear = swatch(&device, &queue, [255, 255, 255, 0]);
        let material = |slot, diffuse: &(wgpu::TextureView, wgpu::Sampler)| ResolvedMaterial {
            slot,
            params: MaterialParams {
                diffuse_color: [1.0, 0.5, 0.25, 1.0],
                emission_strength: [0.1, 0.2, 0.3],
                metallic_factor: 0.25,
                roughness_factor: 0.75,
                layer_index: 3,
                metallic_channel: 2,
                roughness_channel: 1,
                ..Default::default()
            },
            views: [diffuse.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [diffuse.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        };
        let materials = [material(0, &white), material(1, &clear)];

        let eye = [0.0, 40.0, 8.0];
        let parameters = CullParameters {
            view_projection: camera(eye, [0.0, 0.0, 7.0], 1.2, 1.0),
            eye,
            threshold: 0.001,
            material_slots: 2,
        };
        let mut culler = Culler::new(&device);
        let mut drawer = GBufferDrawer::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            // What the last frame left: red all over, and nearer than anything, which would hide
            // everything drawn - unless the renderer's clear, still waiting for a draw when
            // Yoctite has every mesh, is applied first.
            let views: Vec<_> = colour.iter().map(|t| t.create_view(&Default::default())).collect();
            let depth_view = depth.create_view(&Default::default());
            let _ = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &views
                    .iter()
                    .map(|view| {
                        Some(wgpu::RenderPassColorAttachment {
                            view,
                            resolve_target: None,
                            depth_slice: None,
                            ops: wgpu::Operations { load: wgpu::LoadOp::Clear(wgpu::Color::RED), store: wgpu::StoreOp::Store },
                        })
                    })
                    .collect::<Vec<_>>(),
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.0), store: wgpu::StoreOp::Store }),
                    stencil_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1), store: wgpu::StoreOp::Store }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        let drawn = drawer
            .record(
                &device,
                &mut encoder,
                &pool,
                &mut culler,
                &parameters,
                &colour,
                &depth,
                // As the renderer clears it.
                PendingClear {
                    color: Some(wgpu::Color::TRANSPARENT),
                    depth: Some(1.0),
                    stencil: Some(0),
                },
                &materials,
            )
            .expect("drawn");
        queue.submit([encoder.finish()]);
        // A draw for each material in each of the culling's two phases.
        assert_eq!(drawn.draw_calls, 4);

        let diffuse = pixels(&device, &queue, &colour[0], SIZE, false);
        let normal = pixels(&device, &queue, &colour[1], SIZE, false);
        let ambient = pixels(&device, &queue, &colour[2], SIZE, false);
        let surface = pixels(&device, &queue, &colour[3], SIZE, false);
        let decal = pixels(&device, &queue, &colour[4], SIZE, true);
        let at = |x: u32, y: u32| (y * SIZE + x) as usize;
        let rgba = |v: u32| v.to_le_bytes();
        let ten = |v: u32| [v & 1023, (v >> 10) & 1023, (v >> 20) & 1023].map(|c| c as f32 / 1023.0);
        let near = |a: f32, b: f32| (a - b).abs() < 0.01;

        // The left square: its colour, facing up, its factors read from their channels, its
        // emission and its decal layer.
        let left = at(SIZE / 4, SIZE / 2);
        assert_eq!(rgba(diffuse[left]), [255, 128, 64, 255], "diffuse colour times texture");
        let n = ten(normal[left]);
        assert!(near(n[0], 0.5) && near(n[1], 1.0) && near(n[2], 0.5), "normal straight up: {n:?}");
        let m = rgba(surface[left]);
        assert_eq!([m[0], m[1], m[2], m[3]], [64, 191, 255, 255], "metallic, roughness, occlusion");
        let e = ten(ambient[left]);
        assert!(near(e[0], 0.1) && near(e[1], 0.2) && near(e[2], 0.3), "emission: {e:?}");
        assert_eq!(decal[left], 3, "decal layer");

        // The right square: left out by the alpha test, the G-Buffer as cleared.
        let right = at(SIZE * 3 / 4, SIZE / 2);
        assert_eq!((diffuse[right], normal[right], surface[right], decal[right]), (0, 0, 0, 0), "see-through");
        // The left square covers what it should, whole - every level of detail of a flat square
        // is the square - and nothing of the right one is drawn. From 40 m up, with a field of
        // view of 1.2 radians across 64 pixels, a meter is 32 / (40 tan 0.6) pixels.
        let covered = |range: std::ops::Range<u32>| {
            range.map(|x| (0..SIZE).filter(|&y| diffuse[at(x, y)] != 0).count()).sum::<usize>()
        };
        let side = 16.0 * SIZE as f32 / 2.0 / (40.0 * 0.6f32.tan());
        let expected = side * side;
        let left_covered = covered(0..SIZE / 2) as f32;
        assert!((left_covered - expected).abs() < expected * 0.1, "{left_covered} pixels drawn, about {expected} expected");
        assert_eq!(covered(SIZE / 2 + 1..SIZE), 0, "the right square not");
    }

    /// A device of this machine's, if it has one: the tests that need a GPU skip without.
    #[test]
    fn what_was_hidden_is_put_aside_and_drawn_once_uncovered() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        const SIZE: u32 = 64;
        let target = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let colour = vec![
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::R8Uint),
        ];
        let depth = target(wgpu::TextureFormat::Depth24PlusStencil8);

        // A floor, seen from 40 m up, and a far bigger square 20 m above it covering the whole
        // view - the floor in a colour of its own.
        let mut pool = GeometryPool::new(&device, 1 << 12);
        let (vertices, dag) = hill(16, 0.0);
        let mesh = pool.add_mesh(&device, &queue, &dag, &vertices);
        let floor = pool.add_instance(&device, &queue, mesh, placed(0.0, 1.0, [-8.0, 0.0, -8.0]), 0);
        let roof = pool.add_instance(&device, &queue, mesh, placed(0.0, 4.0, [-32.0, 20.0, -32.0]), 1);
        let white = swatch(&device, &queue, [255; 4]);
        let red = swatch(&device, &queue, [255, 0, 0, 255]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let material = |slot, diffuse: &(wgpu::TextureView, wgpu::Sampler)| ResolvedMaterial {
            slot,
            params: MaterialParams::default(),
            views: [diffuse.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [diffuse.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        };
        let materials = [material(0, &red), material(1, &white)];
        let eye = [0.0, 40.0, 0.5];
        let parameters = CullParameters {
            view_projection: camera(eye, [0.0, 0.0, 0.0], 1.2, 1.0),
            eye,
            threshold: 0.001,
            material_slots: 2,
        };
        let mut culler = Culler::new(&device);
        let mut drawer = GBufferDrawer::new(&device);
        // A frame: the G-Buffer cleared as the renderer clears it, and drawn. With the counts:
        // first phase listed, put aside, second phase listed.
        let mut frame = |culler: &mut Culler, pool: &GeometryPool| {
            let mut encoder = device.create_command_encoder(&Default::default());
            drawer
                .record(
                    &device,
                    &mut encoder,
                    pool,
                    culler,
                    &parameters,
                    &colour,
                    &depth,
                    PendingClear {
                        color: Some(wgpu::Color::TRANSPARENT),
                        depth: Some(1.0),
                        stencil: Some(0),
                    },
                    &materials,
                )
                .expect("drawn");
            queue.submit([encoder.finish()]);
            let counts: Vec<u32> = read(&device, &queue, &culler.counts, 8);
            (counts[1], counts[4], counts[6])
        };
        let middle = (SIZE / 2 * SIZE + SIZE / 2) as usize;
        let red_in_the_middle = || pixels(&device, &queue, &colour[0], SIZE, false)[middle] == u32::from_le_bytes([255, 0, 0, 255]);

        // The first frame has no depths before it: everything is drawn.
        let (listed, aside, _) = frame(&mut culler, &pool);
        assert!(listed >= 2 && aside == 0, "first frame: {listed} listed, {aside} put aside");
        assert!(!red_in_the_middle(), "the floor is under the roof");
        // The second, the floor is put aside, and stays hidden.
        let (listed, aside, uncovered) = frame(&mut culler, &pool);
        assert!(aside >= 1 && uncovered == 0, "second frame: {listed} listed, {aside} put aside, {uncovered} uncovered");
        assert!(!red_in_the_middle());
        // The roof taken away: the frame before's depths still hide the floor, but this frame's
        // do not, and it is drawn after all.
        pool.move_instance(&queue, roof, placed(0.0, 4.0, [1000.0, 20.0, 1000.0]));
        let (_, aside, uncovered) = frame(&mut culler, &pool);
        assert!(aside >= 1 && uncovered == aside, "uncovered: {aside} put aside, {uncovered} uncovered");
        assert!(red_in_the_middle(), "the floor drawn once uncovered");
        let _ = floor;
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        // The adapter's own limits, as the renderer asks for: the G-Buffer's five targets take
        // more bytes a pixel than the defaults allow.
        block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()
    }

    /// Runs a future that is ready at once, as wgpu's are on native backends.
    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(out) = future.as_mut().poll(&mut context) {
                return out;
            }
        }
    }

    /// What `buffer` holds, read back.
    fn read<T: bytemuck::Pod>(device: &wgpu::Device, queue: &wgpu::Queue, buffer: &wgpu::Buffer, count: u32) -> Vec<T> {
        let size = (count as u64 * std::mem::size_of::<T>() as u64).max(4);
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, size);
        queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("mapped"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("polled");
        let data = staging.slice(..).get_mapped_range().expect("mapped range");
        bytemuck::cast_slice(&data[..(count as usize * std::mem::size_of::<T>())]).to_vec()
    }

    /// A grid of `n` by `n` quads, bent into a hill so its levels of detail have error.
    fn hill(n: u32, height: f32) -> (Vec<PoolVertex>, ClusterDag) {
        let mut positions = Vec::new();
        for z in 0..=n {
            for x in 0..=n {
                let (fx, fz) = (x as f32 / n as f32 - 0.5, z as f32 / n as f32 - 0.5);
                positions.push([x as f32, height * (1.0 - 4.0 * (fx * fx + fz * fz)), z as f32]);
            }
        }
        let mut indices = Vec::new();
        for z in 0..n {
            for x in 0..n {
                let (a, b, c, d) = (z * (n + 1) + x, z * (n + 1) + x + 1, (z + 1) * (n + 1) + x, (z + 1) * (n + 1) + x + 1);
                indices.extend([a, c, b, b, c, d]);
            }
        }
        let dag = yoctite::build(&yoctite::Mesh::new(&positions, &indices), Default::default());
        let vertices = positions
            .iter()
            .map(|&position| PoolVertex {
                position,
                normal: [0.0, 1.0, 0.0],
                uv: [position[0], position[2]],
                tangent: [1.0, 0.0, 0.0, 1.0],
            })
            .collect();
        (vertices, dag)
    }

    /// Every cluster of `mesh` read back from the pool's buffers, as triangles of vertex
    /// positions, compared with what was built.
    fn check_mesh(device: &wgpu::Device, queue: &wgpu::Queue, pool: &GeometryPool, mesh: MeshId, vertices: &[PoolVertex], dag: &ClusterDag) {
        let record = pool.mesh(mesh).expect("in the pool");
        let meshes: Vec<GpuMesh> = read(device, queue, pool.meshes.buffer(), pool.meshes.capacity());
        assert_eq!(meshes[mesh.0 as usize], record);
        let clusters: Vec<GpuCluster> = read(device, queue, pool.clusters.buffer(), pool.clusters.capacity());
        let cluster_vertices: Vec<u32> = read(device, queue, pool.cluster_vertices.buffer(), pool.cluster_vertices.capacity());
        let triangles: Vec<u32> = read(device, queue, pool.triangles.buffer(), pool.triangles.capacity());
        let pool_vertices: Vec<PoolVertex> = read(device, queue, pool.vertices.buffer(), pool.vertices.capacity());
        assert_eq!(record.cluster_count as usize, dag.clusters.len());
        for (i, built) in dag.clusters.iter().enumerate() {
            let gpu = clusters[(record.first_cluster as usize) + i];
            assert_eq!((gpu.triangle_count as usize, gpu.vertex_count as usize), (built.triangles.len(), built.vertices.len()));
            assert_eq!((gpu.error, gpu.parent_error, gpu.level, gpu.mesh), (built.error, built.parent_error.min(f32::MAX), built.level, mesh.0));
            for (t, expected) in built.mesh_triangles().enumerate() {
                let packed = triangles[(gpu.first_triangle as usize) + t];
                for k in 0..3 {
                    let local = (packed >> (8 * k)) & 0xff;
                    let v = cluster_vertices[(gpu.first_vertex + local) as usize];
                    assert_eq!(pool_vertices[v as usize], vertices[expected[k] as usize], "cluster {i} triangle {t}");
                }
            }
        }
    }

    #[test]
    fn bending_meshes_keep_their_offsets_and_weights_in_the_vertex_buffer() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut pool = GeometryPool::new(&device, 16);
        // Something not bending first, so the bending mesh is not at the start of the buffer.
        let (still, still_dag) = hill(10, 1.0);
        pool.add_mesh(&device, &queue, &still_dag, &still);
        let (vertices, dag) = hill(20, 3.0);
        let targets = 2;
        let offsets: Vec<[f32; 6]> = (0..vertices.len())
            .flat_map(|v| [[0.0, 1.0, 0.0, 0.5, 0.0, -0.25], [v as f32 / 64.0, 0.0, -0.125, 0.0, 0.0, 0.0]])
            .collect();
        let mesh = pool.add_morphing_mesh(
            &device,
            &queue,
            &dag,
            &vertices,
            Some(Morphs {
                targets,
                offsets: &offsets,
                most_weight: 2.0,
            }),
        );
        let instance = pool.add_instance(&device, &queue, mesh, [0.0; 12], 0);
        let unbent = pool.add_instance(&device, &queue, MeshId(0), [0.0; 12], 0);
        // Two more right after it, set out of order: written with it in one run.
        let next: Vec<InstanceId> = (0..2).map(|_| pool.add_instance(&device, &queue, mesh, [0.0; 12], 0)).collect();
        pool.set_weights(next[1], &[3.0, 4.0]);
        pool.set_weights(instance, &[0.5, -0.25, 9.0]);
        pool.set_weights(next[0], &[1.0, 2.0]);
        pool.set_weights(unbent, &[7.0]);
        pool.flush(&queue);

        let gpu: Vec<GpuInstance> = read(&device, &queue, pool.instances.buffer(), pool.instance_slots());
        assert_eq!(gpu[unbent.0 as usize].morph, NO_MORPH);
        let block = gpu[instance.0 as usize].morph as usize;
        let words: Vec<u32> = read(&device, &queue, pool.vertices.buffer(), pool.vertices.capacity() * VERTEX_FLOATS);
        let record = pool.mesh(mesh).unwrap();
        assert_eq!(words[block], record.first_vertex);
        assert_eq!(words[block + 2], targets);
        // Only as many weights as the mesh has targets.
        assert_eq!([f32::from_bits(words[block + 3]), f32::from_bits(words[block + 4])], [0.5, -0.25]);
        assert_eq!(words[block + 5], 0);
        for (id, want) in [(next[0], [1.0, 2.0]), (next[1], [3.0, 4.0])] {
            let at = gpu[id.0 as usize].morph as usize;
            assert_eq!(words[at], record.first_vertex);
            assert_eq!([f32::from_bits(words[at + 3]), f32::from_bits(words[at + 4])], want);
        }
        // Each vertex's offsets, where the shader looks for them, as two half floats a word.
        let half = |w: u32, high: bool| half::f16::from_bits((if high { w >> 16 } else { w }) as u16).to_f32();
        for v in [0usize, 7, vertices.len() - 1] {
            for t in 0..targets as usize {
                let at = words[block + 1] as usize + (v * targets as usize + t) * 3;
                let read = [
                    half(words[at], false),
                    half(words[at], true),
                    half(words[at + 1], false),
                    half(words[at + 1], true),
                    half(words[at + 2], false),
                    half(words[at + 2], true),
                ];
                for (got, want) in read.iter().zip(offsets[v * targets as usize + t]) {
                    assert!((got - want).abs() < 2e-3, "vertex {v} target {t}: {read:?}");
                }
            }
        }
        // The clusters' bounds take in the bending: at least a meter up, at weight 2 each way.
        let clusters: Vec<GpuCluster> = read(&device, &queue, pool.clusters.buffer(), pool.clusters.capacity());
        let first = &clusters[record.first_cluster as usize];
        assert!(first.bounds[3] >= dag.clusters[0].bounds.radius + 2.0);
    }

    #[test]
    fn a_bending_mesh_is_drawn_bent() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        const SIZE: u32 = 64;
        let target = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let colour = vec![
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::R8Uint),
        ];
        let depth = target(wgpu::TextureFormat::Depth24PlusStencil8);
        // A flat square on the left, whose one target slides it 16 m right, onto the right half.
        let mut pool = GeometryPool::new(&device, 1 << 12);
        let (vertices, dag) = hill(16, 0.0);
        let offsets = vec![[16.0, 0.0, 0.0, 0.0, 0.0, 0.0]; vertices.len()];
        let mesh = pool.add_morphing_mesh(
            &device,
            &queue,
            &dag,
            &vertices,
            Some(Morphs { targets: 1, offsets: &offsets, most_weight: 1.0 }),
        );
        let instance = pool.add_instance(&device, &queue, mesh, placed(0.0, 1.0, [-16.0, 0.0, 0.0]), 0);
        let white = swatch(&device, &queue, [255; 4]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let materials = [ResolvedMaterial {
            slot: 0,
            params: MaterialParams::default(),
            views: [white.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [white.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        }];
        let eye = [0.0, 40.0, 8.0];
        let parameters = CullParameters {
            view_projection: camera(eye, [0.0, 0.0, 7.0], 1.2, 1.0),
            eye,
            threshold: 0.001,
            material_slots: 1,
        };
        let mut culler = Culler::new(&device);
        let mut drawer = GBufferDrawer::new(&device);
        let at = |x: u32, y: u32| (y * SIZE + x) as usize;
        let mut draw = |pool: &GeometryPool| {
            let mut encoder = device.create_command_encoder(&Default::default());
            drawer
                .record(
                    &device,
                    &mut encoder,
                    pool,
                    &mut culler,
                    &parameters,
                    &colour,
                    &depth,
                    PendingClear {
                        color: Some(wgpu::Color::TRANSPARENT),
                        depth: Some(1.0),
                        stencil: Some(0),
                    },
                    &materials,
                )
                .expect("drawn");
            queue.submit([encoder.finish()]);
            let diffuse = pixels(&device, &queue, &colour[0], SIZE, false);
            let covered = |range: std::ops::Range<u32>| {
                range.map(|x| (0..SIZE).filter(|&y| diffuse[at(x, y)] != 0).count()).sum::<usize>()
            };
            (covered(0..SIZE / 2), covered(SIZE / 2 + 1..SIZE))
        };
        let (left, right) = draw(&pool);
        assert!(left > 100 && right == 0, "unbent, on the left: {left} and {right} pixels");
        pool.set_weights(instance, &[1.0]);
        pool.flush(&queue);
        let (left, right) = draw(&pool);
        assert!(right > 100 && left < 20, "bent, on the right: {left} and {right} pixels");
    }

    /// A device with the software rasterizer's features, where the GPU has them.
    fn software_device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        if !adapter.features().contains(SOFTWARE_FEATURES) {
            return None;
        }
        block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: SOFTWARE_FEATURES,
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()
    }

    #[test]
    fn small_clusters_are_rasterized_in_software_as_the_hardware_draws_them() {
        let Some((device, queue)) = software_device() else {
            eprintln!("no GPU with the software rasterizer's features: skipped");
            return;
        };
        const SIZE: u32 = 128;
        let target = |format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::COPY_SRC
                    | wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            })
        };
        let colour = vec![
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgb10a2Unorm),
            target(wgpu::TextureFormat::Rgba8Unorm),
            target(wgpu::TextureFormat::R8Uint),
        ];
        let depth = target(wgpu::TextureFormat::Depth24PlusStencil8);
        // A flat square on the left, which its one target slides 16 m right.
        let mut pool = GeometryPool::new(&device, 1 << 12);
        let (vertices, dag) = hill(16, 0.0);
        let offsets = vec![[16.0, 0.0, 0.0, 0.0, 0.0, 0.0]; vertices.len()];
        let mesh = pool.add_morphing_mesh(
            &device,
            &queue,
            &dag,
            &vertices,
            Some(Morphs { targets: 1, offsets: &offsets, most_weight: 1.0 }),
        );
        let instance = pool.add_instance(&device, &queue, mesh, placed(0.0, 1.0, [-16.0, 0.0, 0.0]), 0);
        // And one below the middle whose diffuse texture is see-through all over: left out by the
        // alpha test, in software as in hardware.
        pool.add_instance(&device, &queue, mesh, placed(0.0, 1.0, [-8.0, 0.0, 16.0]), 1);
        let white = swatch(&device, &queue, [255; 4]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let clear = swatch(&device, &queue, [255, 255, 255, 0]);
        let material = |slot, diffuse: &(wgpu::TextureView, wgpu::Sampler)| ResolvedMaterial {
            slot,
            params: MaterialParams {
                diffuse_color: [1.0, 0.5, 0.25, 1.0],
                layer_index: 3,
                ..Default::default()
            },
            views: [diffuse.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [diffuse.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        };
        let materials = [material(0, &white), material(1, &clear)];
        let both_solid = [material(0, &white), material(1, &white)];
        let eye = [0.0, 40.0, 8.0];
        let parameters = CullParameters {
            view_projection: camera(eye, [0.0, 0.0, 7.0], 1.2, 1.0),
            eye,
            threshold: 0.001,
            material_slots: 2,
        };
        // Drawn with every cluster software-rasterized, or none.
        let draw = |pool: &GeometryPool, software: bool, materials: &[ResolvedMaterial]| {
            let mut culler = Culler::new(&device);
            let mut drawer = GBufferDrawer::new(&device);
            assert!(drawer.has_software_raster());
            drawer.software_pixels = if software { 1.0e6 } else { 0.0 };
            let mut encoder = device.create_command_encoder(&Default::default());
            let drawn = drawer
                .record(
                    &device,
                    &mut encoder,
                    pool,
                    &mut culler,
                    &parameters,
                    &colour,
                    &depth,
                    PendingClear {
                        color: Some(wgpu::Color::TRANSPARENT),
                        depth: Some(1.0),
                        stencil: Some(0),
                    },
                    materials,
                )
                .expect("drawn");
            queue.submit([encoder.finish()]);
            assert_eq!(drawn.software, software);
            [0, 1, 4].map(|i| pixels(&device, &queue, &colour[i], SIZE, i == 4))
        };
        let at = |x: u32, y: u32| (y * SIZE + x) as usize;
        let covered = |diffuse: &[u32], range: std::ops::Range<u32>| {
            range.map(|x| (0..SIZE).filter(|&y| diffuse[at(x, y)] != 0).count()).sum::<usize>()
        };

        let [hardware, hardware_normal, hardware_decal] = draw(&pool, false, &materials);
        let [software, software_normal, software_decal] = draw(&pool, true, &materials);
        let drawn = covered(&hardware, 0..SIZE);
        assert!(drawn > 500, "the square drawn: {drawn} pixels");
        // Only the solid square: the see-through one, below it, nowhere.
        let below = |diffuse: &[u32]| (0..SIZE).map(|x| (SIZE * 3 / 4..SIZE).filter(|&y| diffuse[at(x, y)] != 0).count()).sum::<usize>();
        assert_eq!((below(&hardware), below(&software)), (0, 0), "the see-through square left out");
        // Where it would be drawn, solid.
        let [solid, ..] = draw(&pool, true, &both_solid);
        assert!(below(&solid) > 200, "the second square in view: {} pixels", below(&solid));
        // Each pixel's middle covered alike: the edges may differ by a pixel where a middle
        // lies on one.
        let differ = (0..(SIZE * SIZE) as usize).filter(|&i| (hardware[i] != 0) != (software[i] != 0)).count();
        assert!(differ * 50 < drawn, "{differ} of {drawn} pixels covered differently");
        let middle = at(SIZE / 4, SIZE / 2);
        assert_eq!(software[middle].to_le_bytes(), [255, 128, 64, 255], "diffuse colour times texture");
        assert_eq!(software[middle], hardware[middle]);
        assert_eq!(software_normal[middle], hardware_normal[middle], "the normal");
        assert_eq!(software_decal[middle], 3, "the decal layer");
        assert_eq!(hardware_decal[middle], 3);

        // Bent as the hardware bends it.
        pool.set_weights(instance, &[1.0]);
        pool.flush(&queue);
        let [bent, ..] = draw(&pool, true, &materials);
        let (left, right) = (covered(&bent, 0..SIZE / 2), covered(&bent, SIZE / 2 + 1..SIZE));
        assert!(right > 500 && left < 50, "bent, on the right: {left} and {right} pixels");
    }

    #[test]
    fn meshes_and_instances_come_and_go_on_the_gpu() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        // Tiny buffers to begin with, so they have to grow.
        let mut pool = GeometryPool::new(&device, 16);
        let (vertices_a, dag_a) = hill(40, 6.0);
        let (vertices_b, dag_b) = hill(24, 2.0);
        let a = pool.add_mesh(&device, &queue, &dag_a, &vertices_a);
        let b = pool.add_mesh(&device, &queue, &dag_b, &vertices_b);
        check_mesh(&device, &queue, &pool, a, &vertices_a, &dag_a);
        check_mesh(&device, &queue, &pool, b, &vertices_b, &dag_b);

        // Taken out, and its room used again by another, without disturbing the rest.
        pool.remove_mesh(&queue, a);
        let (vertices_c, dag_c) = hill(30, 4.0);
        let c = pool.add_mesh(&device, &queue, &dag_c, &vertices_c);
        assert_eq!(c, a, "the freed slot");
        assert_eq!(pool.mesh(c).unwrap().first_vertex, 0, "the freed vertices");
        check_mesh(&device, &queue, &pool, b, &vertices_b, &dag_b);
        check_mesh(&device, &queue, &pool, c, &vertices_c, &dag_c);

        let at = |x: f32| [1.0, 0.0, 0.0, x, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let instances: Vec<InstanceId> = (0..2000).map(|i| pool.add_instance(&device, &queue, if i % 2 == 0 { b } else { c }, at(i as f32), 0)).collect();
        pool.move_instance(&queue, instances[7], at(-1.0));
        pool.remove_instance(&queue, instances[3]);
        let gpu: Vec<GpuInstance> = read(&device, &queue, pool.instances.buffer(), pool.instance_slots());
        assert_eq!(gpu[7].transform[3], -1.0);
        assert_eq!(gpu[3].flags, 0, "a removed instance is marked free");
        assert_eq!((gpu[8].mesh, gpu[8].flags, gpu[8].transform[3]), (b.0, INSTANCE_ALIVE, 8.0));
        assert_eq!(pool.add_instance(&device, &queue, b, at(0.0), 0), instances[3], "its slot reused");

        let stats = pool.stats();
        assert_eq!((stats.meshes, stats.instances), (2, 2000));
        assert_eq!(stats.clusters as usize, dag_b.clusters.len() + dag_c.clusters.len());
    }
}
