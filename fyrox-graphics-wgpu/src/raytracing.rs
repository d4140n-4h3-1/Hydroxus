//! Ray tracing against the scene's geometry, where the hardware supports it.
//!
//! Screen-space effects can only work with what the camera sees. Ray tracing hardware works with
//! the geometry itself: the triangles are put into an acceleration structure once, and after that a
//! shader can ask "does anything block this ray?" and get an answer that includes what is behind
//! the camera or around a corner.
//!
//! The structure comes in two levels. [`RayTracedGeometry`] holds one mesh's triangles, in the
//! mesh's own space, built once; one that is posed - a skinned character - can be given its
//! vertices again and refitted. [`RayTracedScene`] places copies of those wherever they are now,
//! and is cheap to rebuild every frame, so that what moves casts its shadow from where it is.
//!
//! What lives here is the part that has to: the acceleration structure, and the pass that traces
//! against it. It is all self-contained - none of the renderer's own machinery knows about ray
//! tracing - so an effect elsewhere can use the result as an ordinary texture.
//!
//! Glass ([`WgpuGraphicsServer::build_ray_traced_glass`]) is traced as well, but rays go on
//! through it, and each copy of it says how much of each colour of light it lets through, so the
//! shadows it casts are the colour of the light that came through it.
//!
//! Cutouts ([`WgpuGraphicsServer::build_ray_traced_cutout`]) - leaves, needles, grass, a chain-link
//! fence: cards whose texture says where they are solid - stop a ray only where it crosses a
//! solid part of the texture, as the renderer draws them. Each keeps its triangles' texture
//! coordinates and a [`CutoutMask`] of which texels are solid, which the scene hands to the shader
//! alongside its acceleration structure.
//!
//! Availability: [`WgpuGraphicsServer::ray_tracing`] is false unless the adapter reported ray
//! queries, in which case everything here returns [`None`] and the caller falls back to whatever
//! it did before.

use crate::{server::WgpuGraphicsServer, texture::WgpuTexture};
use fyrox_graphics::{
    error::FrameworkError,
    gpu_texture::{GpuTexture, GpuTextureDescriptor, GpuTextureKind, PixelKind},
    server::GraphicsServer,
};
use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// Which texels of a texture are solid, a bit each, row by row: for a cutout, where a ray is
/// stopped and where it goes on through.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CutoutMask {
    width: u32,
    height: u32,
    bits: Vec<u32>,
}

impl CutoutMask {
    /// A `width` by `height` mask, solid wherever `solid(x, y)` says, with row 0 the texture's
    /// first row of texels - the one texture coordinate v = 0 samples.
    pub fn new(width: u32, height: u32, solid: impl Fn(u32, u32) -> bool) -> Self {
        let (width, height) = (width.max(1), height.max(1));
        let mut bits = vec![0u32; (width as usize * height as usize).div_ceil(32)];
        for y in 0..height {
            for x in 0..width {
                if solid(x, y) {
                    let bit = (y * width + x) as usize;
                    bits[bit / 32] |= 1 << (bit % 32);
                }
            }
        }
        Self { width, height, bits }
    }

    /// Whether the texel at `x`, `y` is solid.
    pub fn is_solid(&self, x: u32, y: u32) -> bool {
        let bit = (y.min(self.height - 1) * self.width + x.min(self.width - 1)) as usize;
        self.bits[bit / 32] & (1 << (bit % 32)) != 0
    }

    /// How many texels wide it is.
    pub fn width(&self) -> u32 {
        self.width
    }

    /// How many texels high it is.
    pub fn height(&self) -> u32 {
        self.height
    }

    /// Its bits, 32 to a word, row by row.
    pub fn words(&self) -> &[u32] {
        &self.bits
    }
}

/// What a cutout keeps for the shader: the texture coordinates of each of its triangles'
/// corners, and its mask.
struct Cutout {
    corners: CutoutCorners,
    mask: Arc<CutoutMask>,
}

impl Cutout {
    /// What tells its record apart: copies of one surface - bent each its own way, so each
    /// geometry of its own - share their corners and mask, and so one record.
    fn key(&self) -> (usize, usize) {
        (Arc::as_ptr(&self.corners) as *const u8 as usize, Arc::as_ptr(&self.mask) as *const u8 as usize)
    }
}

/// The texture coordinates of each corner of a cutout's triangles, triangle after triangle:
/// see [`WgpuGraphicsServer::cutout_corners`]. Shared by geometry built from the same surface.
pub type CutoutCorners = Arc<Vec<[f32; 2]>>;

/// How geometry stands in a ray's way.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Solid,
    Glass,
    Cutout,
}

/// Every piece of geometry gets its own number, for a scene to tell which cutouts it holds.
static NEXT_GEOMETRY: AtomicU64 = AtomicU64::new(1);

/// One mesh's triangles in a form the ray tracing hardware can trace against, in the mesh's own
/// space, for a [`RayTracedScene`] to place copies of.
pub struct RayTracedGeometry {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    blas: wgpu::Blas,
    size: wgpu::BlasTriangleGeometrySizeDescriptor,
    vertex_count: u32,
    triangle_count: u32,
    /// Whether it can be given its vertices again, and refitted round them.
    updatable: bool,
    /// Whether it is glass, which rays go on through, rather than solid.
    see_through: bool,
    /// For a cutout, where it is solid.
    cutout: Option<Cutout>,
    id: u64,
}

impl RayTracedGeometry {
    /// How many triangles it has.
    pub fn triangle_count(&self) -> u32 {
        self.triangle_count
    }

    /// How many vertices it has, three floats each.
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// Whether it is glass, which rays go on through: see
    /// [`WgpuGraphicsServer::build_ray_traced_glass`].
    pub fn is_see_through(&self) -> bool {
        self.see_through
    }

    /// Whether it is a cutout, solid only where its texture is: see
    /// [`WgpuGraphicsServer::build_ray_traced_cutout`].
    pub fn is_cutout(&self) -> bool {
        self.cutout.is_some()
    }

    fn build_entry(&self) -> wgpu::BlasBuildEntry<'_> {
        wgpu::BlasBuildEntry {
            blas: &self.blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size: &self.size,
                vertex_buffer: &self.vertices,
                first_vertex: 0,
                vertex_stride: 12,
                index_buffer: Some(&self.indices),
                first_index: Some(0),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        }
    }
}

/// A copy of some [`RayTracedGeometry`], placed in the world.
#[derive(Clone, Copy)]
pub struct RayTracedInstance<'a> {
    /// What is placed.
    pub geometry: &'a RayTracedGeometry,
    /// Where it is: an affine transform from its own space to the world's, three rows of four,
    /// row by row.
    pub transform: [f32; 12],
    /// For glass, how much of the red, green and blue of a light it lets through, each from 0 to
    /// 1, where a ray crosses it. Solid geometry, and cutouts, let nothing through whatever this
    /// says.
    pub lets_through: [f32; 3],
}

/// The bit of an instance's custom data that marks a cutout; the bits below it say which of the
/// scene's cutouts it is. Without it, the custom data is the light glass lets through.
const CUTOUT_BIT: u32 = 1 << 23;

/// `lets_through` as an instance's custom data: seven bits for each of red, green and blue, a
/// byte apart, leaving the top bit of each byte clear - and so [`CUTOUT_BIT`].
fn lets_through_bits([r, g, b]: [f32; 3]) -> u32 {
    let seven = |v: f32| (v.clamp(0.0, 1.0) * 127.0).round() as u32;
    seven(r) | (seven(g) << 8) | (seven(b) << 16)
}

/// What the shaders read about a scene's cutouts, in storage buffers: one record per cutout,
/// the texture coordinates of each of their triangles' corners, and their masks' bits.
struct CutoutBuffers {
    /// Which corners and mask each record is, in order (see [`Cutout::key`]); held, so that
    /// their addresses are not used again by others while they tell records apart.
    keys: Vec<(usize, usize)>,
    _held: Vec<(CutoutCorners, Arc<CutoutMask>)>,
    records: wgpu::Buffer,
    corners: wgpu::Buffer,
    masks: wgpu::Buffer,
}

/// A cutout's record, as the shaders read it.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CutoutRecord {
    /// Where its triangles' corners start, counted in triangles.
    first_triangle: u32,
    /// Where its mask starts, counted in 32-bit words.
    first_word: u32,
    width: u32,
    height: u32,
}

impl CutoutBuffers {
    /// Buffers for `cutouts`, in that order, each a record; a mask several share is put in once.
    /// Never empty, as a binding has to have something in it.
    fn new(device: &wgpu::Device, cutouts: &[&Cutout]) -> Self {
        let mut records = Vec::with_capacity(cutouts.len());
        let mut corners: Vec<[f32; 2]> = Vec::new();
        let mut masks: Vec<u32> = Vec::new();
        let mut placed_masks: HashMap<usize, u32> = HashMap::new();
        for cutout in cutouts {
            let first_word = *placed_masks.entry(Arc::as_ptr(&cutout.mask) as *const u8 as usize).or_insert_with(|| {
                let first = masks.len() as u32;
                masks.extend_from_slice(&cutout.mask.bits);
                first
            });
            records.push(CutoutRecord {
                first_triangle: (corners.len() / 3) as u32,
                first_word,
                width: cutout.mask.width,
                height: cutout.mask.height,
            });
            corners.extend_from_slice(&cutout.corners);
        }
        if records.is_empty() {
            records.push(<CutoutRecord as bytemuck::Zeroable>::zeroed());
        }
        if corners.is_empty() {
            corners.push([0.0; 2]);
        }
        if masks.is_empty() {
            masks.push(0);
        }
        let buffer = |label, contents: &[u8]| {
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents,
                usage: wgpu::BufferUsages::STORAGE,
            })
        };
        Self {
            keys: cutouts.iter().map(|cutout| cutout.key()).collect(),
            _held: cutouts.iter().map(|cutout| (cutout.corners.clone(), cutout.mask.clone())).collect(),
            records: buffer("CutoutRecords", bytemuck::cast_slice(&records)),
            corners: buffer("CutoutCorners", bytemuck::cast_slice(&corners)),
            masks: buffer("CutoutMasks", bytemuck::cast_slice(&masks)),
        }
    }

    /// The bindings, from `first` on, a shader reads them through.
    fn entries(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 3] {
        [
            wgpu::BindGroupEntry {
                binding: first,
                resource: self.records.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: first + 1,
                resource: self.corners.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: first + 2,
                resource: self.masks.as_entire_binding(),
            },
        ]
    }
}

/// The layout entries for [`CutoutBuffers::entries`], from binding `first` on.
pub(crate) fn cutout_layout_entries(first: u32, visibility: wgpu::ShaderStages) -> [wgpu::BindGroupLayoutEntry; 3] {
    let storage = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    [storage(first), storage(first + 1), storage(first + 2)]
}

/// The WGSL that reads a scene's cutouts, for a shader that binds them in group 0 from binding
/// `first` on: `cutout_blocks(customData, primitiveIndex, barycentrics)` says whether a candidate
/// hit on a cutout is on a solid part of it.
pub(crate) fn cutout_wgsl(first: u32) -> String {
    format!(
        r#"
const CUTOUT_BIT: u32 = 0x800000u;

struct CutoutRecord {{
    firstTriangle: u32,
    firstWord: u32,
    width: u32,
    height: u32,
}};

@group(0) @binding({records}) var<storage, read> cutoutRecords: array<CutoutRecord>;
@group(0) @binding({corners}) var<storage, read> cutoutCorners: array<vec2f>;
@group(0) @binding({masks}) var<storage, read> cutoutMasks: array<u32>;

// Whether a hit on triangle `primitive` of the cutout `data` names, at `barycentrics`, is on a
// texel its mask says is solid. Texture coordinates repeat, as the renderer samples them.
fn cutout_blocks(data: u32, primitive: u32, barycentrics: vec2f) -> bool {{
    let record = cutoutRecords[data & (CUTOUT_BIT - 1u)];
    let corner = (record.firstTriangle + primitive) * 3u;
    let uv = cutoutCorners[corner] * (1.0 - barycentrics.x - barycentrics.y)
        + cutoutCorners[corner + 1u] * barycentrics.x
        + cutoutCorners[corner + 2u] * barycentrics.y;
    let wrapped = fract(uv);
    let x = min(u32(wrapped.x * f32(record.width)), record.width - 1u);
    let y = min(u32(wrapped.y * f32(record.height)), record.height - 1u);
    let bit = y * record.width + x;
    return ((cutoutMasks[record.firstWord + bit / 32u] >> (bit % 32u)) & 1u) != 0u;
}}
"#,
        records = first,
        corners = first + 1,
        masks = first + 2,
    )
}

/// The transform that leaves geometry where it is.
pub const IDENTITY_TRANSFORM: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// Scene geometry in a form the ray tracing hardware can trace against: copies of geometry,
/// each where it is.
pub struct RayTracedScene {
    pub(crate) tlas: wgpu::Tlas,
    /// How many copies it has room for, how many it was last given, and how many triangles the
    /// ones it has come to.
    capacity: u32,
    used: u32,
    triangle_count: u32,
    /// What the shaders read about the cutouts it holds.
    cutouts: CutoutBuffers,
    /// For a scene built from triangles alone, the geometry it holds.
    _geometry: Option<RayTracedGeometry>,
}

impl RayTracedScene {
    /// How many triangles were put into the structure, counting each copy.
    pub fn triangle_count(&self) -> u32 {
        self.triangle_count
    }

    /// The bindings, from `first` on, through which a shader reads its cutouts (see
    /// [`cutout_wgsl`]).
    pub(crate) fn cutout_entries(&self, first: u32) -> [wgpu::BindGroupEntry<'_>; 3] {
        self.cutouts.entries(first)
    }
}

/// A mesh's blend shapes, on the GPU, for [`RayTracedPoser`] to bend copies of its geometry by:
/// where its vertices are unbent, and how far each shape moves each of them.
pub struct RayTracedMorph {
    buffer: wgpu::Buffer,
    vertex_count: u32,
    targets: u32,
    id: u64,
}

/// The most blend shapes a [`RayTracedMorph`] can have.
pub const MAX_POSE_TARGETS: u32 = 60;

impl RayTracedMorph {
    /// How many vertices it bends: only geometry with as many is bent by it.
    pub fn vertex_count(&self) -> u32 {
        self.vertex_count
    }

    /// A mesh's blend shapes: `positions` unbent, three floats a vertex, in its own space; and
    /// `offsets`, three floats each, `targets` to a vertex, vertex after vertex. [`None`] if they
    /// do not agree, or there are more targets than [`MAX_POSE_TARGETS`].
    pub fn new(device: &wgpu::Device, positions: &[f32], offsets: &[f32], targets: u32) -> Option<Self> {
        let vertex_count = (positions.len() / 3) as u32;
        if vertex_count == 0
            || targets == 0
            || targets > MAX_POSE_TARGETS
            || offsets.len() != vertex_count as usize * targets as usize * 3
        {
            return None;
        }
        let mut contents = Vec::with_capacity(positions.len() + offsets.len());
        contents.extend_from_slice(&positions[..vertex_count as usize * 3]);
        contents.extend_from_slice(offsets);
        Some(Self {
            buffer: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("RayTracedMorph"),
                contents: bytemuck::cast_slice(&contents),
                usage: wgpu::BufferUsages::STORAGE,
            }),
            vertex_count,
            targets,
            id: NEXT_GEOMETRY.fetch_add(1, Ordering::Relaxed),
        })
    }
}

const POSE_SHADER: &str = r"
// Bends a copy of a mesh's geometry by its blend shapes: each vertex where it is unbent, moved by
// every shape with a weight. A job is 256 bytes: how many vertices and shapes, and the weights.
struct Job {
    vertexCount: u32,
    targets: u32,
    pad0: u32,
    pad1: u32,
    weights: array<vec4f, 15>,
};

@group(0) @binding(0) var<storage, read> job: Job;
@group(0) @binding(1) var<storage, read> morph: array<f32>;
@group(0) @binding(2) var<storage, read_write> positions: array<f32>;

@compute @workgroup_size(64)
fn pose(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let v = id.y * groups.x * 64u + id.x;
    if (v >= job.vertexCount) {
        return;
    }
    var p = vec3f(morph[v * 3u], morph[v * 3u + 1u], morph[v * 3u + 2u]);
    let offsets = job.vertexCount * 3u + v * job.targets * 3u;
    for (var t = 0u; t < job.targets; t++) {
        let weight = job.weights[t / 4u][t % 4u];
        if (weight == 0.0) {
            continue;
        }
        let at = offsets + t * 3u;
        p += vec3f(morph[at], morph[at + 1u], morph[at + 2u]) * weight;
    }
    positions[v * 3u] = p.x;
    positions[v * 3u + 1u] = p.y;
    positions[v * 3u + 2u] = p.z;
}
";

/// How many bytes a job of [`RayTracedPoser`] takes: one dynamic offset apart.
const POSE_JOB_BYTES: u64 = 256;

/// Bends copies of meshes' geometry by their blend shapes on the GPU, and refits them, all
/// together: as the renderer bends the meshes, so that their shadows move with them. The CPU
/// only hands over the weights. Kept between frames, so its pipeline is made once and each
/// geometry's bindings once.
pub struct RayTracedPoser {
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    jobs: wgpu::Buffer,
    job_capacity: u32,
    /// Each geometry's bindings, by its and its morph's numbers, while the job buffer stays.
    groups: HashMap<(u64, u64), wgpu::BindGroup>,
}

impl RayTracedPoser {
    /// The poser's pipeline.
    pub fn new(device: &wgpu::Device) -> Self {
        let storage = |binding, read_only, dynamic| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: dynamic,
                min_binding_size: wgpu::BufferSize::new(if dynamic { POSE_JOB_BYTES } else { 0 }),
            },
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("RayTracedPose"),
            entries: &[storage(0, true, true), storage(1, true, false), storage(2, false, false)],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("RayTracedPose"),
            source: wgpu::ShaderSource::Wgsl(POSE_SHADER.into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("RayTracedPose"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("RayTracedPose"),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            })),
            module: &module,
            entry_point: Some("pose"),
            compilation_options: Default::default(),
            cache: None,
        });
        Self {
            layout,
            pipeline,
            jobs: Self::job_buffer(device, 64),
            job_capacity: 64,
            groups: HashMap::new(),
        }
    }

    fn job_buffer(device: &wgpu::Device, capacity: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("RayTracedPoseJobs"),
            size: capacity as u64 * POSE_JOB_BYTES,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// Bends each geometry - updatable, with as many vertices as its morph - by its morph and
    /// weights (1 a shape at full strength; past the morph's shapes left out), and refits it.
    pub fn pose(
        &mut self,
        server: &WgpuGraphicsServer,
        poses: &[(&RayTracedGeometry, &RayTracedMorph, &[f32])],
    ) -> Result<(), FrameworkError> {
        let poses: Vec<_> = poses
            .iter()
            .filter(|(geometry, morph, _)| geometry.updatable && geometry.vertex_count == morph.vertex_count)
            .collect();
        if poses.is_empty() {
            return Ok(());
        }
        let device = &server.state.device;
        let queue = &server.state.queue;
        if poses.len() as u32 > self.job_capacity {
            self.job_capacity = (poses.len() as u32).next_power_of_two();
            self.jobs = Self::job_buffer(device, self.job_capacity);
            self.groups.clear();
        }
        let mut jobs = vec![0u8; poses.len() * POSE_JOB_BYTES as usize];
        for (job, (_, morph, weights)) in jobs.chunks_exact_mut(POSE_JOB_BYTES as usize).zip(&poses) {
            let mut words = [0u32; POSE_JOB_BYTES as usize / 4];
            words[0] = morph.vertex_count;
            words[1] = morph.targets;
            for (i, w) in weights.iter().take(morph.targets as usize).enumerate() {
                words[4 + i] = w.to_bits();
            }
            job.copy_from_slice(bytemuck::cast_slice(&words));
        }
        queue.write_buffer(&self.jobs, 0, &jobs);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("RayTracedPose"),
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("RayTracedPose"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            for (i, (geometry, morph, _)) in poses.iter().enumerate() {
                let group = self.groups.entry((geometry.id, morph.id)).or_insert_with(|| {
                    device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("RayTracedPose"),
                        layout: &self.layout,
                        entries: &[
                            wgpu::BindGroupEntry {
                                binding: 0,
                                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                    buffer: &self.jobs,
                                    offset: 0,
                                    size: wgpu::BufferSize::new(POSE_JOB_BYTES),
                                }),
                            },
                            wgpu::BindGroupEntry {
                                binding: 1,
                                resource: morph.buffer.as_entire_binding(),
                            },
                            wgpu::BindGroupEntry {
                                binding: 2,
                                resource: geometry.vertices.as_entire_binding(),
                            },
                        ],
                    })
                });
                pass.set_bind_group(0, &*group, &[(i as u64 * POSE_JOB_BYTES) as u32]);
                let groups = morph.vertex_count.div_ceil(64);
                pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
            }
        }
        let entries: Vec<wgpu::BlasBuildEntry> = poses.iter().map(|(geometry, _, _)| geometry.build_entry()).collect();
        encoder.build_acceleration_structures(entries.iter(), std::iter::empty());
        queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Lets go of the bindings of geometry and morphs no longer posed, so they can be freed:
    /// those not in `kept` (by geometry).
    pub fn forget_others(&mut self, kept: &[&RayTracedGeometry]) {
        let ids: std::collections::HashSet<u64> = kept.iter().map(|g| g.id).collect();
        self.groups.retain(|(geometry, _), _| ids.contains(geometry));
    }
}

/// The light a shadow ray is traced towards.
#[derive(Debug, Clone, Copy)]
pub enum ShadowRayLight {
    /// The sun: infinitely far away, so every ray runs the same way.
    Directional {
        /// The direction light travels *towards* the surface.
        direction: [f32; 3],
        /// How far a shadow ray reaches, in meters.
        reach: f32,
        /// How big the sun looks, as the tangent of the angle from its middle to its edge. A
        /// bigger sun casts softer shadows; zero gives hard ones.
        angular_size: f32,
    },
    /// A lamp or a torch: rays run to the light's position, and pixels out of its reach are
    /// left lit, since it lights nothing there.
    Positional {
        /// Where the light is, in world space.
        position: [f32; 3],
        /// How far the light reaches, in meters.
        radius: f32,
        /// The radius of the part that glows, in meters. A bigger light casts softer shadows;
        /// zero gives hard ones.
        size: f32,
    },
}

/// What a shadow ray needs to know: where the camera was, and where the light is.
#[derive(Debug, Clone, Copy)]
pub struct ShadowRayParameters {
    /// Turns a screen position and depth back into a world position.
    pub inverse_view_projection: [[f32; 4]; 4],
    /// The light the rays are traced towards.
    pub light: ShadowRayLight,
    /// How far off the surface a ray starts, in meters, so a surface does not shadow itself.
    pub bias: f32,
    /// True when render targets are stored top row first, which is how this backend stores them.
    pub flip_v: bool,
    /// The only pixels to trace, as x, y, width and height with the origin at the top left.
    /// Pixels outside it are left lit. A lamp lights a small part of the screen, and this keeps
    /// it from paying for rays everywhere else.
    pub scissor: Option<[u32; 4]>,
    /// Rays per pixel, spread over the light's size: a light with a size casts soft-edged
    /// shadows, and the mask is then blurred to smooth out the differences between neighboring
    /// pixels. More rays, smoother edges; with [`Self::frame`], one is enough.
    pub samples: u32,
    /// Which frame this is, where the frames are averaged afterwards (temporal anti-aliasing):
    /// the pattern of rays over the light then turns from frame to frame, and one ray a pixel
    /// covers the whole light over a few of them. [`None`] keeps the pattern still.
    pub frame: Option<u32>,
    /// Whether to trace one pixel of each 2x2 block - a quarter of the rays - and fill in the
    /// rest from the traced ones on the same surface. With [`Self::frame`], a different pixel of
    /// each block is traced every frame, and the averaged frames come to the whole picture.
    pub half_resolution: bool,
}

/// The passes that trace the rays and smooth the result, kept between frames so their pipelines
/// are built once.
pub struct ShadowTracer {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    blur_pipeline: wgpu::RenderPipeline,
    blur_layout: wgpu::BindGroupLayout,
    mask: Option<GpuTexture>,
    blurred: Option<GpuTexture>,
    history_pipeline: wgpu::RenderPipeline,
    history_layout: wgpu::BindGroupLayout,
    history_sampler: wgpu::Sampler,
    /// Each light's shadows over the frames before, by the order the lights are traced in a
    /// frame; the frame those were counted in, and how many traced so far in it.
    histories: Vec<ShadowHistory>,
    frame: Option<u32>,
    traced: usize,
}

/// A light's shadows averaged over the frames before (see `shaders/shadow_history.wgsl`): two
/// textures, one read while the other is written, and the view the latest was seen through.
#[derive(Default)]
struct ShadowHistory {
    textures: [Option<GpuTexture>; 2],
    latest: usize,
    view_projection: [[f32; 4]; 4],
    inverse_view_projection: [[f32; 4]; 4],
    /// Whether the latest holds a frame's shadows yet.
    valid: bool,
}

/// How much of each frame's shadows goes into their average over frames: about the last ten
/// frames count.
const HISTORY_BLEND: f32 = 0.1;
const HISTORY_SHADER: &str = include_str!("shaders/shadow_history.wgsl");

const SHADER: &str = include_str!("shaders/shadow_rays.wgsl");
/// Where the shadow rays' shader binds the scene's cutouts from (see [`cutout_wgsl`]).
const SHADOW_CUTOUT_BINDING: u32 = 5;

/// The shadow rays' shader, with what reads the scene's cutouts.
fn shadow_rays_source() -> String {
    // The `enable` has to come before anything else.
    let (enable, rest) = SHADER.split_once('\n').unwrap_or(("", SHADER));
    format!("{enable}\n{}{rest}", cutout_wgsl(SHADOW_CUTOUT_BINDING))
}
const BLUR_SHADER: &str = include_str!("shaders/shadow_blur.wgsl");

/// How far off a pixel's plane, in meters, a neighbor may lie and still be blurred with it.
const BLUR_PLANE_TOLERANCE: f32 = 0.05;

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// A pipeline that draws one full-screen triangle into a colour target: how much of a light's
/// red, green and blue gets through.
fn mask_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    layout: &wgpu::BindGroupLayout,
) -> wgpu::RenderPipeline {
    target_pipeline(device, label, source, layout, wgpu::TextureFormat::Rgba8Unorm)
}

/// A pipeline that draws one full-screen triangle into a colour target of `format`.
fn target_pipeline(
    device: &wgpu::Device,
    label: &str,
    source: &str,
    layout: &wgpu::BindGroupLayout,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some(label),
        source: wgpu::ShaderSource::Wgsl(source.into()),
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
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
                format,
                blend: None,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview_mask: None,
        cache: None,
    })
}

/// Records a pass that clears `target` to 1 (lit) and draws a full-screen triangle into it,
/// within `scissor` if given.
fn draw_mask(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &WgpuTexture,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
    scissor: Option<[u32; 4]>,
) {
    let view = target
        .wgpu_texture()
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: &view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    if let Some([x, y, w, h]) = scissor {
        pass.set_scissor_rect(x, y, w, h);
    }
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}

/// The colour texture in `slot`, made again if it is not `width` by `height`.
fn mask_texture(
    slot: &mut Option<GpuTexture>,
    server: &WgpuGraphicsServer,
    name: &'static str,
    width: usize,
    height: usize,
) -> Result<GpuTexture, FrameworkError> {
    texture_of(slot, server, name, width, height, PixelKind::RGBA8)
}

/// The texture of `pixel_kind` in `slot`, made again if it is not `width` by `height`.
fn texture_of(
    slot: &mut Option<GpuTexture>,
    server: &WgpuGraphicsServer,
    name: &'static str,
    width: usize,
    height: usize,
    pixel_kind: PixelKind,
) -> Result<GpuTexture, FrameworkError> {
    let matches = slot.as_ref().is_some_and(|texture| {
        matches!(texture.kind(), GpuTextureKind::Rectangle { width: w, height: h } if w == width && h == height)
    });
    if !matches {
        *slot = Some(server.create_texture(GpuTextureDescriptor {
            name,
            kind: GpuTextureKind::Rectangle { width, height },
            pixel_kind,
            ..Default::default()
        })?);
    }
    slot.clone()
        .ok_or_else(|| FrameworkError::Custom("shadow mask texture".into()))
}

impl WgpuGraphicsServer {
    /// Puts triangles into an acceleration structure. Positions are in world space, three floats
    /// each; every three indices make a triangle. Returns [`None`] without ray tracing hardware.
    pub fn build_ray_traced_scene(
        &self,
        vertices: &[f32],
        indices: &[u32],
    ) -> Result<Option<RayTracedScene>, FrameworkError> {
        let Some(geometry) = self.build_ray_traced_geometry(vertices, indices, false)? else {
            return Ok(None);
        };
        let instance = RayTracedInstance {
            geometry: &geometry,
            transform: IDENTITY_TRANSFORM,
            lets_through: [0.0; 3],
        };
        let Some(mut scene) = self.build_ray_traced_instances(&[instance])? else {
            return Ok(None);
        };
        scene._geometry = Some(geometry);
        Ok(Some(scene))
    }

    /// Puts one mesh's triangles into an acceleration structure of their own, for a
    /// [`RayTracedScene`] to place copies of. Positions are in the mesh's own space, three floats
    /// each; every three indices make a triangle. `updatable` geometry can be given its vertices
    /// again later, for something posed, at some cost to how fast rays are traced against it.
    /// Returns [`None`] without ray tracing hardware, or with no triangles.
    pub fn build_ray_traced_geometry(
        &self,
        vertices: &[f32],
        indices: &[u32],
        updatable: bool,
    ) -> Result<Option<RayTracedGeometry>, FrameworkError> {
        self.build_geometry(vertices, indices, updatable, Kind::Solid, None)
    }

    /// As [`Self::build_ray_traced_geometry`], but glass: rays go on through it rather than
    /// stopping, and each copy of it lets through as much of a light as its
    /// [`RayTracedInstance::lets_through`] says, so a light shone through it takes on its colour.
    pub fn build_ray_traced_glass(
        &self,
        vertices: &[f32],
        indices: &[u32],
        updatable: bool,
    ) -> Result<Option<RayTracedGeometry>, FrameworkError> {
        self.build_geometry(vertices, indices, updatable, Kind::Glass, None)
    }

    /// As [`Self::build_ray_traced_geometry`], but a cutout: a ray is stopped only where it
    /// crosses a texel `mask` says is solid. `uvs` are the vertices' texture coordinates, two
    /// floats each, as the renderer samples the texture the mask was made from.
    pub fn build_ray_traced_cutout(
        &self,
        vertices: &[f32],
        indices: &[u32],
        uvs: &[f32],
        mask: Arc<CutoutMask>,
        updatable: bool,
    ) -> Result<Option<RayTracedGeometry>, FrameworkError> {
        let corners = Self::cutout_corners(indices, uvs);
        self.build_ray_traced_cutout_sharing(vertices, indices, corners, mask, updatable)
    }

    /// The texture coordinates of each corner of the triangles `indices` makes, from each
    /// vertex's `uvs` (two floats each): for [`Self::build_ray_traced_cutout_sharing`].
    pub fn cutout_corners(indices: &[u32], uvs: &[f32]) -> CutoutCorners {
        let uv = |i: u32| {
            let i = i as usize * 2;
            [uvs.get(i).copied().unwrap_or(0.0), uvs.get(i + 1).copied().unwrap_or(0.0)]
        };
        Arc::new(indices[..indices.len() / 3 * 3].iter().map(|&i| uv(i)).collect())
    }

    /// As [`Self::build_ray_traced_cutout`], with the corners' texture coordinates worked out
    /// already ([`Self::cutout_corners`]) and shared: geometry built from one surface again and
    /// again - a copy of its own for each plant bent its own way - sharing them shares one
    /// record of the scene's cutouts.
    pub fn build_ray_traced_cutout_sharing(
        &self,
        vertices: &[f32],
        indices: &[u32],
        corners: CutoutCorners,
        mask: Arc<CutoutMask>,
        updatable: bool,
    ) -> Result<Option<RayTracedGeometry>, FrameworkError> {
        self.build_geometry(vertices, indices, updatable, Kind::Cutout, Some(Cutout { corners, mask }))
    }

    fn build_geometry(
        &self,
        vertices: &[f32],
        indices: &[u32],
        updatable: bool,
        kind: Kind,
        cutout: Option<Cutout>,
    ) -> Result<Option<RayTracedGeometry>, FrameworkError> {
        if !self.ray_tracing || vertices.is_empty() || indices.len() < 3 {
            return Ok(None);
        }

        let device = &self.state.device;
        let vertex_count = (vertices.len() / 3) as u32;
        let index_count = (indices.len() / 3 * 3) as u32;

        let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RayTracedVertices"),
            contents: bytemuck::cast_slice(vertices),
            // Updatable geometry can be posed on the GPU, too (see [`RayTracedPoser`]).
            usage: wgpu::BufferUsages::BLAS_INPUT
                | wgpu::BufferUsages::COPY_DST
                | if updatable { wgpu::BufferUsages::STORAGE } else { wgpu::BufferUsages::empty() },
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("RayTracedIndices"),
            contents: bytemuck::cast_slice(&indices[..index_count as usize]),
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
        });

        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(index_count),
            // Glass is only ever passed through, and each pane has to be counted once. A cutout
            // is not opaque either, so that the shader is asked about each hit on it.
            flags: match kind {
                Kind::Solid => wgpu::AccelerationStructureGeometryFlags::OPAQUE,
                Kind::Glass | Kind::Cutout => {
                    wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION
                }
            },
        };
        let (flags, update_mode) = if updatable {
            (
                wgpu::AccelerationStructureFlags::ALLOW_UPDATE
                    | wgpu::AccelerationStructureFlags::PREFER_FAST_BUILD,
                wgpu::AccelerationStructureUpdateMode::PreferUpdate,
            )
        } else {
            (
                wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                wgpu::AccelerationStructureUpdateMode::Build,
            )
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("RayTracedGeometry"),
                flags,
                update_mode,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let geometry = RayTracedGeometry {
            vertices: vertex_buffer,
            indices: index_buffer,
            blas,
            size,
            vertex_count,
            triangle_count: index_count / 3,
            updatable,
            see_through: kind == Kind::Glass,
            cutout,
            id: NEXT_GEOMETRY.fetch_add(1, Ordering::Relaxed),
        };

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("BuildRayTracedGeometry"),
        });
        encoder.build_acceleration_structures([&geometry.build_entry()], std::iter::empty());
        self.state.queue.submit([encoder.finish()]);
        Ok(Some(geometry))
    }

    /// Gives each piece of updatable geometry its vertices again - as many as it was built with,
    /// in its own space - and refits it round them, all at once.
    pub fn update_ray_traced_geometry(
        &self,
        updates: &[(&RayTracedGeometry, &[f32])],
    ) -> Result<(), FrameworkError> {
        if updates.is_empty() {
            return Ok(());
        }
        for (geometry, vertices) in updates {
            if !geometry.updatable || vertices.len() != geometry.vertex_count as usize * 3 {
                return Err(FrameworkError::Custom(format!(
                    "ray traced geometry of {} vertices, updatable: {}, given {} floats",
                    geometry.vertex_count,
                    geometry.updatable,
                    vertices.len()
                )));
            }
            self.state
                .queue
                .write_buffer(&geometry.vertices, 0, bytemuck::cast_slice(vertices));
        }
        let entries: Vec<wgpu::BlasBuildEntry> =
            updates.iter().map(|(geometry, _)| geometry.build_entry()).collect();
        let mut encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("UpdateRayTracedGeometry"),
            });
        encoder.build_acceleration_structures(entries.iter(), std::iter::empty());
        self.state.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Places `instances` in a scene to trace against. Returns [`None`] without ray tracing
    /// hardware.
    pub fn build_ray_traced_instances(
        &self,
        instances: &[RayTracedInstance],
    ) -> Result<Option<RayTracedScene>, FrameworkError> {
        if !self.ray_tracing {
            return Ok(None);
        }
        let capacity = (instances.len() as u32).max(1).next_power_of_two();
        let tlas = self.state.device.create_tlas(&wgpu::CreateTlasDescriptor {
            label: Some("RayTracedScene"),
            max_instances: capacity,
            flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
            update_mode: wgpu::AccelerationStructureUpdateMode::Build,
        });
        let mut scene = RayTracedScene {
            tlas,
            capacity,
            used: 0,
            triangle_count: 0,
            cutouts: CutoutBuffers::new(&self.state.device, &[]),
            _geometry: None,
        };
        self.update_ray_traced_instances(&mut scene, instances)?;
        Ok(Some(scene))
    }

    /// Places `instances` in `scene` in place of what it had, making room for them if it has to,
    /// and giving back room it has far more of than it needs. Cheap enough to do every frame: it
    /// costs as many copies as there are, not as many as there is room for.
    pub fn update_ray_traced_instances(
        &self,
        scene: &mut RayTracedScene,
        instances: &[RayTracedInstance],
    ) -> Result<(), FrameworkError> {
        // Every slot there is room for is gone through each time the scene is built, used or
        // not, so room for many more than are placed is given back.
        let wanted = (instances.len() as u32).max(1).next_power_of_two();
        if instances.len() as u32 > scene.capacity || scene.capacity >= wanted * 4 {
            scene.tlas = self.state.device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("RayTracedScene"),
                max_instances: wanted,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            });
            scene.capacity = wanted;
            scene.used = 0;
        }
        // Each cutout placed gets a record for the shaders, in the order they first come - one
        // for all the geometry sharing its corners and mask (see [`Cutout::key`]); the records
        // are only written again when which cutouts there are changes.
        let mut cutouts: Vec<&Cutout> = Vec::new();
        let mut records: HashMap<(usize, usize), u32> = HashMap::new();
        let mut custom_data: Vec<u32> = Vec::with_capacity(instances.len());
        for instance in instances {
            custom_data.push(match &instance.geometry.cutout {
                Some(cutout) => {
                    let record = *records.entry(cutout.key()).or_insert_with(|| {
                        cutouts.push(cutout);
                        cutouts.len() as u32 - 1
                    });
                    CUTOUT_BIT | record
                }
                None => lets_through_bits(instance.lets_through),
            });
        }
        if cutouts.len() >= CUTOUT_BIT as usize {
            return Err(FrameworkError::Custom(format!(
                "{} cutouts in one ray traced scene; at most {} fit",
                cutouts.len(),
                CUTOUT_BIT - 1
            )));
        }
        let keys: Vec<(usize, usize)> = cutouts.iter().map(|cutout| cutout.key()).collect();
        if keys != scene.cutouts.keys {
            scene.cutouts = CutoutBuffers::new(&self.state.device, &cutouts);
        }

        // Only the slots given copies now or last time; those past them are empty already.
        let end = (instances.len() as u32).max(scene.used) as usize;
        scene.used = instances.len() as u32;
        let slots = scene
            .tlas
            .get_mut_slice(0..end)
            .ok_or_else(|| FrameworkError::Custom("ray traced scene instances".into()))?;
        for (slot, instance) in slots.iter_mut().zip(
            instances
                .iter()
                .zip(&custom_data)
                .map(Some)
                .chain(std::iter::repeat(None)),
        ) {
            *slot = instance.map(|(instance, &custom_data)| {
                wgpu::TlasInstance::new(&instance.geometry.blas, instance.transform, custom_data, 0xff)
            });
        }
        scene.triangle_count = instances
            .iter()
            .map(|instance| instance.geometry.triangle_count)
            .sum();
        let mut encoder = self
            .state
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("BuildRayTracedScene"),
            });
        encoder.build_acceleration_structures(std::iter::empty(), [&scene.tlas]);
        self.state.queue.submit([encoder.finish()]);
        Ok(())
    }

    /// Creates the passes that trace shadow rays, or [`None`] without ray tracing hardware.
    pub fn create_shadow_tracer(&self) -> Option<ShadowTracer> {
        if !self.ray_tracing {
            return None;
        }
        let device = &self.state.device;
        let non_filtering = wgpu::TextureSampleType::Float { filterable: false };
        let cutouts = cutout_layout_entries(SHADOW_CUTOUT_BINDING, wgpu::ShaderStages::FRAGMENT);

        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowRays"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::AccelerationStructure {
                        vertex_return: false,
                    },
                    count: None,
                },
                texture_entry(1, wgpu::TextureSampleType::Depth),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                uniform_entry(3),
                texture_entry(4, non_filtering),
                cutouts[0].clone(),
                cutouts[1].clone(),
                cutouts[2].clone(),
            ],
        });
        let pipeline = mask_pipeline(device, "ShadowRays", &shadow_rays_source(), &layout);

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowBlur"),
            entries: &[
                texture_entry(0, non_filtering),
                texture_entry(1, wgpu::TextureSampleType::Depth),
                texture_entry(2, non_filtering),
                uniform_entry(3),
            ],
        });
        let blur_pipeline = mask_pipeline(device, "ShadowBlur", BLUR_SHADER, &blur_layout);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ShadowRays"),
            ..Default::default()
        });
        let history_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("ShadowHistory"),
            entries: &[
                texture_entry(0, non_filtering),
                texture_entry(1, wgpu::TextureSampleType::Float { filterable: true }),
                texture_entry(2, wgpu::TextureSampleType::Depth),
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                uniform_entry(4),
            ],
        });
        let history_pipeline = target_pipeline(
            device,
            "ShadowHistory",
            HISTORY_SHADER,
            &history_layout,
            wgpu::TextureFormat::Rgba16Float,
        );
        let history_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("ShadowHistory"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        Some(ShadowTracer {
            pipeline,
            layout,
            sampler,
            blur_pipeline,
            blur_layout,
            mask: None,
            blurred: None,
            history_pipeline,
            history_layout,
            history_sampler,
            histories: Vec::new(),
            frame: None,
            traced: 0,
        })
    }
}

impl ShadowTracer {
    /// Blends `result`, this frame's shadows, into its light's shadows of the frames before,
    /// recorded into `encoder`; the light is told by the order it is traced in `frame`.
    #[allow(clippy::too_many_arguments)]
    fn average(
        &mut self,
        server: &WgpuGraphicsServer,
        encoder: &mut wgpu::CommandEncoder,
        parameters: &ShadowRayParameters,
        frame: u32,
        result: &GpuTexture,
        depth: &WgpuTexture,
        width: usize,
        height: usize,
    ) -> Result<GpuTexture, FrameworkError> {
        if self.frame != Some(frame) {
            self.frame = Some(frame);
            self.traced = 0;
        }
        let index = self.traced;
        self.traced += 1;
        if self.histories.len() <= index {
            self.histories.resize_with(index + 1, Default::default);
        }
        let history = &mut self.histories[index];
        let write = 1 - history.latest;
        let sized = |texture: &Option<GpuTexture>| {
            texture.as_ref().is_some_and(|t| {
                matches!(t.kind(), GpuTextureKind::Rectangle { width: w, height: h } if w == width && h == height)
            })
        };
        if !sized(&history.textures[history.latest]) {
            history.valid = false;
        }
        let target = texture_of(&mut history.textures[write], server, "RayTracedShadowHistory", width, height, PixelKind::RGBA16F)?;
        let before = history.textures[history.latest].clone().filter(|_| history.valid).unwrap_or_else(|| result.clone());

        let inverse = parameters.inverse_view_projection;
        let view_projection = fyrox_core::algebra::Matrix4::from(inverse)
            .try_inverse()
            .map_or([[0.0; 4]; 4], |m| m.into());
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            inverse_view_projection: [[f32; 4]; 4],
            previous_view_projection: [[f32; 4]; 4],
            previous_inverse_view_projection: [[f32; 4]; 4],
            screen_size: [f32; 2],
            flip_v: u32,
            blend: f32,
        }
        let uniforms = Uniforms {
            inverse_view_projection: inverse,
            previous_view_projection: history.view_projection,
            previous_inverse_view_projection: history.inverse_view_projection,
            screen_size: [width as f32, height as f32],
            flip_v: parameters.flip_v as u32,
            blend: if history.valid { HISTORY_BLEND } else { 1.0 },
        };
        let device = &server.state.device;
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowHistory"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let (Some(current), Some(before), Some(target_texture)) = (
            result.as_any().downcast_ref::<WgpuTexture>(),
            before.as_any().downcast_ref::<WgpuTexture>(),
            target.as_any().downcast_ref::<WgpuTexture>(),
        ) else {
            return Ok(result.clone());
        };
        let depth_view = depth.wgpu_texture().create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowHistory"),
            layout: &self.history_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(current.wgpu_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(before.wgpu_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&self.history_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: uniform_buffer.as_entire_binding(),
                },
            ],
        });
        draw_mask(encoder, "ShadowHistory", target_texture, &self.history_pipeline, &bind_group, None);
        history.latest = write;
        history.view_projection = view_projection;
        history.inverse_view_projection = inverse;
        history.valid = true;
        Ok(target)
    }

    /// Traces shadow rays for each pixel and returns a mask: how much of the light's red, green
    /// and blue reaches it - 1 where all of it does, 0 where the geometry blocks it, in between
    /// where it blocks part of the light, and coloured where it came through glass. The mask is a
    /// texture like any other, to be read by an effect.
    ///
    /// The same texture is returned every time, so each call overwrites the previous mask. That
    /// is fine for reading it right after, as the renderer does for one light at a time: the GPU
    /// runs the work in the order it was recorded.
    ///
    /// With [`ShadowRayParameters::frame`], the mask is each light's shadows averaged over the
    /// frames (see `shaders/shadow_history.wgsl`), the lights told apart by the order they are
    /// traced in a frame.
    pub fn trace(
        &mut self,
        server: &WgpuGraphicsServer,
        scene: &RayTracedScene,
        depth: &GpuTexture,
        normals: &GpuTexture,
        parameters: ShadowRayParameters,
    ) -> Result<Option<GpuTexture>, FrameworkError> {
        let GpuTextureKind::Rectangle { width, height } = depth.kind() else {
            return Ok(None);
        };
        let scale: usize = if parameters.half_resolution { 2 } else { 1 };
        let (mask_width, mask_height) = (width.div_ceil(scale), height.div_ceil(scale));
        let mask = mask_texture(&mut self.mask, server, "RayTracedShadowMask", mask_width, mask_height)?;
        // Which pixel of each block is traced: each in turn, frame by frame, where the frames are
        // averaged.
        let offset = if scale > 1 {
            [[0u32, 0], [1, 1], [1, 0], [0, 1]][parameters.frame.unwrap_or(0) as usize % 4]
        } else {
            [0, 0]
        };
        // A light with a size casts soft edges, which the blur smooths.
        let soft = match parameters.light {
            ShadowRayLight::Directional { angular_size, .. } => angular_size > 0.0,
            ShadowRayLight::Positional { size, .. } => size > 0.0,
        };
        // At half resolution the blur fills in the pixels not traced, soft or not.
        let blurred = if soft || scale > 1 {
            Some(mask_texture(
                &mut self.blurred,
                server,
                "RayTracedShadowMaskBlurred",
                width,
                height,
            )?)
        } else {
            None
        };
        let result = blurred.clone().unwrap_or_else(|| mask.clone());

        // Clamped to the mask, which wgpu requires of a scissor rectangle.
        let scissor = match parameters.scissor {
            Some([x, y, w, h]) => {
                let (width, height) = (width as u32, height as u32);
                let x = x.min(width);
                let y = y.min(height);
                let w = w.min(width - x);
                let h = h.min(height - y);
                if w == 0 || h == 0 {
                    // The light reaches nothing on screen, so nothing reads the mask either.
                    return Ok(Some(result));
                }
                Some([x, y, w, h])
            }
            None => None,
        };

        let (Some(depth), Some(normals), Some(target)) = (
            depth.as_any().downcast_ref::<WgpuTexture>(),
            normals.as_any().downcast_ref::<WgpuTexture>(),
            mask.as_any().downcast_ref::<WgpuTexture>(),
        ) else {
            return Ok(None);
        };

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            inverse_view_projection: [[f32; 4]; 4],
            light_direction: [f32; 3],
            reach: f32,
            screen_size: [f32; 2],
            bias: f32,
            flip_v: u32,
            light_position: [f32; 3],
            light_radius: f32,
            positional: u32,
            sample_count: u32,
            light_size: f32,
            turn: f32,
            reach_out: f32,
            scale: u32,
            offset: [u32; 2],
        }
        let (light_direction, reach, light_position, light_radius, positional, light_size) =
            match parameters.light {
                ShadowRayLight::Directional {
                    direction,
                    reach,
                    angular_size,
                } => (direction, reach, [0.0; 3], 0.0, 0, angular_size),
                ShadowRayLight::Positional {
                    position,
                    radius,
                    size,
                } => ([0.0, -1.0, 0.0], radius, position, radius, 1, size),
            };
        let screen_size = [width as f32, height as f32];
        let uniforms = Uniforms {
            inverse_view_projection: parameters.inverse_view_projection,
            light_direction,
            reach,
            screen_size,
            bias: parameters.bias,
            flip_v: parameters.flip_v as u32,
            light_position,
            light_radius,
            positional,
            sample_count: parameters.samples.max(1),
            light_size,
            // Roberts' R2 sequence, the two-dimensional golden ratio: steps that never repeat and
            // never line up with each other, so each frame tries parts of the light the last few
            // did not.
            turn: parameters.frame.map_or(0.0, |f| (f as f64 * 0.754_877_666_2).fract() as f32),
            reach_out: parameters.frame.map_or(0.0, |f| (f as f64 * 0.569_840_291_0).fract() as f32),
            scale: scale as u32,
            offset,
        };

        let device = &server.state.device;
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("ShadowRays"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let depth_view = depth
            .wgpu_texture()
            .create_view(&wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            });
        let cutouts = scene.cutout_entries(SHADOW_CUTOUT_BINDING);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("ShadowRays"),
            layout: &self.layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.tlas.as_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&depth_view),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(normals.wgpu_view()),
                },
                cutouts[0].clone(),
                cutouts[1].clone(),
                cutouts[2].clone(),
            ],
        });

        // Anything recorded so far has to reach the GPU first: the depth buffer being read here is
        // written by it.
        server.flush_active_pass();
        let mut encoder = server
            .frame_encoder
            .borrow_mut()
            .take()
            .unwrap_or_else(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("ShadowRays"),
                })
            });
        // The scissor in the mask's pixels: every block it touches.
        let mask_scissor = scissor.map(|[x, y, w, h]| {
            let s = scale as u32;
            let (x0, y0) = (x / s, y / s);
            let (x1, y1) = ((x + w).div_ceil(s).min(mask_width as u32), (y + h).div_ceil(s).min(mask_height as u32));
            [x0, y0, x1 - x0, y1 - y0]
        });
        draw_mask(
            &mut encoder,
            "ShadowRays",
            target,
            &self.pipeline,
            &bind_group,
            mask_scissor,
        );
        server.gpu_mark_in(&mut encoder, "Lighting: shadow rays");

        if let Some(blurred) = blurred.as_ref() {
            let Some(blur_target) = blurred.as_any().downcast_ref::<WgpuTexture>() else {
                *server.frame_encoder.borrow_mut() = Some(encoder);
                return Ok(Some(mask));
            };

            #[repr(C)]
            #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
            struct BlurUniforms {
                inverse_view_projection: [[f32; 4]; 4],
                screen_size: [f32; 2],
                flip_v: u32,
                plane_tolerance: f32,
                mask_size: [i32; 2],
                offset: [i32; 2],
                scale: i32,
                radius: i32,
                _padding: [u32; 2],
            }
            let blur_uniforms = BlurUniforms {
                inverse_view_projection: parameters.inverse_view_projection,
                screen_size,
                flip_v: parameters.flip_v as u32,
                plane_tolerance: BLUR_PLANE_TOLERANCE,
                mask_size: [mask_width as i32, mask_height as i32],
                offset: [offset[0] as i32, offset[1] as i32],
                scale: scale as i32,
                // Soft edges are averaged over the 4x4 turns of the pattern, 5x5 pixels; at half
                // resolution 3x3 traced ones reach about as far, the turns over the frames
                // averaged after. Hard edges are only filled in, from the traced pixels nearest.
                radius: if soft && scale == 1 { 2 } else { 1 },
                _padding: [0; 2],
            };
            let blur_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ShadowBlur"),
                contents: bytemuck::bytes_of(&blur_uniforms),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            let blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("ShadowBlur"),
                layout: &self.blur_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(target.wgpu_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&depth_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(normals.wgpu_view()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: blur_buffer.as_entire_binding(),
                    },
                ],
            });
            draw_mask(
                &mut encoder,
                "ShadowBlur",
                blur_target,
                &self.blur_pipeline,
                &blur_bind_group,
                scissor,
            );
        }
        // Where the frames are averaged, the shadows are averaged over them too, so they hold
        // still.
        let result = match parameters.frame {
            Some(frame) => self.average(server, &mut encoder, &parameters, frame, &result, depth, width, height)?,
            None => result,
        };
        *server.frame_encoder.borrow_mut() = Some(encoder);

        Ok(Some(result))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn validate(source: &str) {
        let module = wgpu::naga::front::wgsl::parse_str(source)
            .unwrap_or_else(|e| panic!("{}\n{source}", e.emit_to_string(source)));
        wgpu::naga::valid::Validator::new(
            wgpu::naga::valid::ValidationFlags::all(),
            wgpu::naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|e| panic!("{e:?}\n{source}"));
    }

    #[test]
    fn the_shadow_rays_shader_is_valid() {
        validate(&shadow_rays_source());
    }

    #[test]
    fn the_shadow_blur_shader_is_valid() {
        validate(BLUR_SHADER);
    }

    #[test]
    fn the_shadow_history_shader_is_valid() {
        validate(HISTORY_SHADER);
    }

    #[test]
    fn the_pose_shader_is_valid() {
        validate(POSE_SHADER);
    }

    #[test]
    fn a_mask_keeps_which_texels_are_solid() {
        let mask = CutoutMask::new(7, 5, |x, y| (x + y) % 3 == 0);
        for y in 0..5 {
            for x in 0..7 {
                assert_eq!(mask.is_solid(x, y), (x + y) % 3 == 0, "{x}, {y}");
            }
        }
        assert_eq!(mask.bits.len(), 2);
    }

    #[test]
    fn glass_never_looks_like_a_cutout() {
        for through in [[0.0; 3], [1.0; 3], [0.5, 0.25, 1.0]] {
            assert_eq!(lets_through_bits(through) & CUTOUT_BIT, 0);
        }
        assert_eq!(lets_through_bits([1.0, 0.0, 1.0]), 127 | (127 << 16));
    }
}
