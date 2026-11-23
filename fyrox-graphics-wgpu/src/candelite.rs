//! Candelite: dynamic global illumination, after Lumen (Wright et al., "Lumen: Real-time Global
//! Illumination in Unreal Engine 5", SIGGRAPH 2022). No Unreal code is used. Named for the
//! candela, as Yoctite is for the smallest of prefixes. See `ROADMAP_GEOMETRY_GI.md`.
//!
//! This is its first part: the **surface cache**. A ray that hits something has to learn what it
//! hit looks like - its colour, which way it faces, how much light is on it - without shading it
//! there and then, which would cost a material's worth of work for every ray. So every mesh is
//! seen beforehand from a few sides, as **cards**: here the six faces of its bounds, each looking
//! in. What each card sees - the G-Buffer's diffuse colour, normal, emission and material, and how
//! far in the surface lies - is captured once into a rectangle, a **tile**, of a shared **atlas**,
//! by drawing the mesh's full-detail clusters from Yoctite's geometry pool as the G-Buffer pass
//! draws them (`GBufferDrawer::record_capture`). A ray's hit is then found on the cards facing it,
//! and read from the atlas: the next parts light the atlas, and gather from it.
//!
//! A card's tile is as many texels across as [`TEXELS_PER_METER`] of its side, within
//! [`SMALLEST_TILE`] and [`LARGEST_TILE`]. Tiles are packed in shelves of rows a power of two
//! high; a mesh taken out gives its tiles back.
//!
//! On the GPU each card is a [`GpuCard`], six a mesh one after another, and each mesh slot of the
//! pool says where its first card is ([`SurfaceCache::mesh_cards`]), or [`NO_CARDS`].
//!
//! The second part is the **hit lookup**. Candelite traces rays against a scene of its own
//! ([`TraceScene`]): each mesh of the pool as an acceleration structure of its full-detail
//! triangles, and each instance placed in the pool a copy of it, which names the instance's slot
//! - so a hit knows its instance, so its mesh, so its cards. The hit is taken into the mesh's own
//! space, and each card facing the ray finds it on its tile; a card that captured something else
//! there (a branch in front, say) is not believed, and those that agree are averaged, each by how
//! squarely it faces the ray ([`LOOKUP_WGSL`]). [`SurfaceView`] shows it: a ray from the eye
//! through every pixel, and what the lookup says it hit. A mesh drawn with a cutout - foliage, its
//! leaves cards whose texture is see-through between them - is traced as one: a ray passes where
//! the texture is see-through, as the renderer's alpha test leaves it undrawn.

use crate::{
    raytracing::CutoutMask,
    yoctite::{Capture, GBufferDrawer, GeometryPool, MeshId, ResolvedMaterial},
};
use std::sync::Arc;
use fyrox_graphics::error::FrameworkError;
use std::collections::HashMap;
use wgpu::util::DeviceExt;

/// How many texels of a card's tile there are to a meter of what it sees.
pub const TEXELS_PER_METER: f32 = 8.0;
/// The fewest texels a tile has along a side.
pub const SMALLEST_TILE: u32 = 4;
/// The most texels a tile has along a side.
pub const LARGEST_TILE: u32 = 256;
/// How many cards a mesh has: the six faces of its bounds.
pub const CARDS_PER_MESH: u32 = 6;
/// A mesh slot's entry in [`SurfaceCache::mesh_cards`] when it has no cards.
pub const NO_CARDS: u32 = u32::MAX;

/// The formats of the atlas's layers: those of the renderer's G-Buffer - diffuse colour, normal,
/// emission ("ambient"), metallic/roughness/occlusion, decal layer - so its meshes are captured
/// exactly as the G-Buffer pass draws them.
pub const ATLAS_FORMATS: [wgpu::TextureFormat; 5] = [
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::Rgb10a2Unorm,
    wgpu::TextureFormat::Rgb10a2Unorm,
    wgpu::TextureFormat::Rgba8Unorm,
    wgpu::TextureFormat::R8Uint,
];
/// The format of the atlas's depth: how far in from its card's face each texel's surface is, as
/// a share of the bounds' depth along it; 1 where the card sees nothing.
pub const ATLAS_DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Bounds along the axes of a mesh's own space.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bounds {
    /// The least x, y and z.
    pub min: [f32; 3],
    /// The most.
    pub max: [f32; 3],
}

impl Bounds {
    /// The bounds of `points`; empty bounds for none.
    pub fn of(points: impl IntoIterator<Item = [f32; 3]>) -> Self {
        let mut bounds = Self {
            min: [f32::MAX; 3],
            max: [f32::MIN; 3],
        };
        for p in points {
            for i in 0..3 {
                bounds.min[i] = bounds.min[i].min(p[i]);
                bounds.max[i] = bounds.max[i].max(p[i]);
            }
        }
        bounds
    }
}

/// A card: a face of a mesh's bounds, looking in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Card {
    /// The middle of the face, in the mesh's own space.
    pub center: [f32; 3],
    /// Which way the face faces, out of the bounds; the card looks the other way.
    pub normal: [f32; 3],
    /// Across the card, to the right as it looks: `u` cross `v` is `normal`.
    pub u: [f32; 3],
    /// Up the card.
    pub v: [f32; 3],
    /// Half its width along `u`.
    pub half_width: f32,
    /// Half its height along `v`.
    pub half_height: f32,
    /// How deep the bounds are along `normal`.
    pub depth: f32,
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

impl Card {
    /// The six cards of `bounds`: +x, -x, +y, -y, +z, -z. Bounds flat along an axis are given
    /// a little depth, so a card never divides by none.
    pub fn of(bounds: &Bounds) -> [Card; 6] {
        let size: [f32; 3] = std::array::from_fn(|i| (bounds.max[i] - bounds.min[i]).max(1.0e-3));
        let middle: [f32; 3] = std::array::from_fn(|i| (bounds.max[i] + bounds.min[i]) * 0.5);
        let faces: [([f32; 3], usize); 6] = [
            ([1.0, 0.0, 0.0], 0),
            ([-1.0, 0.0, 0.0], 0),
            ([0.0, 1.0, 0.0], 1),
            ([0.0, -1.0, 0.0], 1),
            ([0.0, 0.0, 1.0], 2),
            ([0.0, 0.0, -1.0], 2),
        ];
        faces.map(|(normal, axis)| {
            // Up the card is up, unless the card looks along up; then it is along z, turned so
            // that u cross v is the face's normal.
            let v = match axis {
                1 => [0.0, 0.0, -normal[1]],
                _ => [0.0, 1.0, 0.0],
            };
            let u = cross(v, normal);
            let mut center = middle;
            center[axis] += normal[axis] * size[axis] * 0.5;
            let half = |d: [f32; 3]| 0.5 * (0..3).map(|i| (d[i] * size[i]).abs()).sum::<f32>();
            Card {
                center,
                normal,
                u,
                v,
                half_width: half(u),
                half_height: half(v),
                depth: size[axis],
            }
        })
    }

    /// From the mesh's space to the card's clip space (column by column, OpenGL's, as
    /// [`Capture::view_projection`] is): across and up the card from -1 to 1, and in from its face
    /// to the far side of the bounds from -1 to 1 - so the depth stored is how far in, as a share
    /// of [`Self::depth`].
    pub fn view_projection(&self) -> [[f32; 4]; 4] {
        let (u, v, n) = (self.u, self.v, self.normal);
        let row_x = [u[0] / self.half_width, u[1] / self.half_width, u[2] / self.half_width, -dot(self.center, u) / self.half_width];
        let row_y = [v[0] / self.half_height, v[1] / self.half_height, v[2] / self.half_height, -dot(self.center, v) / self.half_height];
        // Clip z: 2 d / depth - 1, where d = (center - p) . n is how far in p is.
        let k = 2.0 / self.depth;
        let row_z = [-k * n[0], -k * n[1], -k * n[2], k * dot(self.center, n) - 1.0];
        let rows = [row_x, row_y, row_z, [0.0, 0.0, 0.0, 1.0]];
        std::array::from_fn(|column| std::array::from_fn(|row| rows[row][column]))
    }

    /// How many texels its tile has across and up.
    pub fn tile_size(&self) -> [u32; 2] {
        let texels = |half: f32| ((2.0 * half * TEXELS_PER_METER).ceil() as u32).clamp(SMALLEST_TILE, LARGEST_TILE);
        [texels(self.half_width), texels(self.half_height)]
    }
}

/// A card as the GPU reads it.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuCard {
    /// The middle of its face, in the mesh's own space, and the bounds' depth along its normal.
    pub center: [f32; 4],
    /// Across it, and half its width.
    pub u: [f32; 4],
    /// Up it, and half its height. Its normal, out of the bounds, is `u` cross `v`.
    pub v: [f32; 4],
    /// Its tile in the atlas: x, y, width and height in texels, from the top left.
    pub tile: [u32; 4],
}

/// Hands out rectangles of a square atlas, in shelves: rows a power of two high, each filled from
/// the left, first fit; a rectangle given back is joined with free neighbours in its row.
#[derive(Debug, Clone, PartialEq)]
pub struct ShelfAllocator {
    size: u32,
    shelves: Vec<Shelf>,
    /// Where the next shelf would start, down the atlas.
    next: u32,
}

#[derive(Debug, Clone, PartialEq)]
struct Shelf {
    y: u32,
    height: u32,
    /// Free spans, as x and width, in order, none touching.
    free: Vec<(u32, u32)>,
}

impl ShelfAllocator {
    /// An empty atlas `size` texels square.
    pub fn new(size: u32) -> Self {
        Self {
            size,
            shelves: Vec::new(),
            next: 0,
        }
    }

    /// Room for `width` by `height`, by its top left corner; [`None`] if the atlas is full.
    pub fn allocate(&mut self, width: u32, height: u32) -> Option<[u32; 2]> {
        let width = width.max(1);
        let class = height.max(1).next_power_of_two();
        if width > self.size || class > self.size {
            return None;
        }
        for shelf in self.shelves.iter_mut().filter(|s| s.height == class) {
            if let Some(i) = shelf.free.iter().position(|&(_, w)| w >= width) {
                let (x, w) = shelf.free[i];
                if w == width {
                    shelf.free.remove(i);
                } else {
                    shelf.free[i] = (x + width, w - width);
                }
                return Some([x, shelf.y]);
            }
        }
        if self.next + class > self.size {
            return None;
        }
        let y = self.next;
        self.next += class;
        self.shelves.push(Shelf {
            y,
            height: class,
            free: if width < self.size { vec![(width, self.size - width)] } else { Vec::new() },
        });
        Some([0, y])
    }

    /// Gives back what [`Self::allocate`] gave at `at` for `width` by `height`.
    pub fn free(&mut self, at: [u32; 2], width: u32, height: u32) {
        let class = height.max(1).next_power_of_two();
        let Some(shelf) = self.shelves.iter_mut().find(|s| s.y == at[1] && s.height == class) else {
            return;
        };
        let (x, width) = (at[0], width.max(1));
        let i = shelf.free.partition_point(|&(fx, _)| fx < x);
        shelf.free.insert(i, (x, width));
        // Joined with the span after it, then with the one before.
        if i + 1 < shelf.free.len() && shelf.free[i].0 + shelf.free[i].1 == shelf.free[i + 1].0 {
            shelf.free[i].1 += shelf.free[i + 1].1;
            shelf.free.remove(i + 1);
        }
        if i > 0 && shelf.free[i - 1].0 + shelf.free[i - 1].1 == shelf.free[i].0 {
            shelf.free[i - 1].1 += shelf.free[i].1;
            shelf.free.remove(i);
        }
    }
}

/// A mesh in the surface cache: where its cards are, and its tiles.
#[derive(Debug, Clone, PartialEq)]
struct Carded {
    first_card: u32,
    tiles: [[u32; 4]; 6],
}

/// Every mesh's cards, and the atlas they are captured into: see the module's description.
pub struct SurfaceCache {
    size: u32,
    layers: Vec<wgpu::Texture>,
    depth: wgpu::Texture,
    shelves: ShelfAllocator,
    cards: Vec<GpuCard>,
    /// Runs of six cards given back, by their first.
    free_cards: Vec<u32>,
    mesh_cards: Vec<u32>,
    meshes: HashMap<u32, Carded>,
    /// The cards and the mesh index on the GPU, made afresh when they change.
    gpu: Option<(wgpu::Buffer, wgpu::Buffer)>,
    changed: bool,
}

impl SurfaceCache {
    /// An empty cache with an atlas `size` texels square.
    pub fn new(device: &wgpu::Device, size: u32) -> Self {
        let texture = |label, format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size,
                    height: size,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                    | wgpu::TextureUsages::TEXTURE_BINDING
                    | wgpu::TextureUsages::COPY_SRC,
                view_formats: &[],
            })
        };
        Self {
            size,
            layers: ATLAS_FORMATS.iter().map(|&f| texture("CandeliteAtlas", f)).collect(),
            depth: texture("CandeliteAtlasDepth", ATLAS_DEPTH_FORMAT),
            shelves: ShelfAllocator::new(size),
            cards: Vec::new(),
            free_cards: Vec::new(),
            mesh_cards: Vec::new(),
            meshes: HashMap::new(),
            gpu: None,
            changed: true,
        }
    }

    /// Gives `mesh`, of `bounds` in its own space, its cards and their tiles, and returns the
    /// captures that fill them - drawing `clusters` (the full mesh's, as the pool numbers them)
    /// with the material in `slot` - for [`Self::record`]. [`None`] if the atlas has no room left;
    /// nothing is kept then. A mesh already in it keeps what it has, and nothing is captured.
    pub fn add_mesh(&mut self, mesh: MeshId, bounds: &Bounds, clusters: &[u32], slot: u32) -> Option<Vec<Capture>> {
        if self.meshes.contains_key(&mesh.0) {
            return Some(Vec::new());
        }
        let cards = Card::of(bounds);
        let mut tiles = [[0u32; 4]; 6];
        for (i, card) in cards.iter().enumerate() {
            let [w, h] = card.tile_size();
            match self.shelves.allocate(w, h) {
                Some([x, y]) => tiles[i] = [x, y, w, h],
                None => {
                    for tile in &tiles[..i] {
                        self.shelves.free([tile[0], tile[1]], tile[2], tile[3]);
                    }
                    return None;
                }
            }
        }
        let first_card = self.free_cards.pop().unwrap_or_else(|| {
            let first = self.cards.len() as u32;
            self.cards.resize(self.cards.len() + CARDS_PER_MESH as usize, GpuCard::default());
            first
        });
        for (i, (card, tile)) in cards.iter().zip(tiles).enumerate() {
            self.cards[first_card as usize + i] = GpuCard {
                center: [card.center[0], card.center[1], card.center[2], card.depth],
                u: [card.u[0], card.u[1], card.u[2], card.half_width],
                v: [card.v[0], card.v[1], card.v[2], card.half_height],
                tile,
            };
        }
        let slot_index = mesh.0 as usize;
        if self.mesh_cards.len() <= slot_index {
            self.mesh_cards.resize(slot_index + 1, NO_CARDS);
        }
        self.mesh_cards[slot_index] = first_card;
        self.meshes.insert(mesh.0, Carded { first_card, tiles });
        self.changed = true;
        Some(
            cards
                .iter()
                .zip(tiles)
                .map(|(card, tile)| Capture {
                    mesh,
                    clusters: clusters.to_vec(),
                    slot,
                    view_projection: card.view_projection(),
                    viewport: tile,
                })
                .collect(),
        )
    }

    /// Takes `mesh` out, giving back its tiles and cards.
    pub fn remove_mesh(&mut self, mesh: MeshId) {
        let Some(carded) = self.meshes.remove(&mesh.0) else {
            return;
        };
        for tile in carded.tiles {
            self.shelves.free([tile[0], tile[1]], tile[2], tile[3]);
        }
        self.free_cards.push(carded.first_card);
        self.mesh_cards[mesh.0 as usize] = NO_CARDS;
        self.changed = true;
    }

    /// Records `captures` (from [`Self::add_mesh`]) into the atlas, drawn by `drawer` from `pool`
    /// with `materials`.
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        drawer: &mut GBufferDrawer,
        pool: &GeometryPool,
        captures: &[Capture],
        materials: &[ResolvedMaterial],
    ) -> Result<(), FrameworkError> {
        drawer.record_capture(device, encoder, pool, captures, &self.layers, &self.depth, materials)
    }

    /// The cards, and each mesh slot's first card, on the GPU, made afresh if anything changed.
    pub fn buffers(&mut self, device: &wgpu::Device) -> (&wgpu::Buffer, &wgpu::Buffer) {
        if self.changed || self.gpu.is_none() {
            let make = |label, contents: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    // A binding may not be empty.
                    contents: if contents.is_empty() { &[0; 16] } else { contents },
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                })
            };
            self.gpu = Some((
                make("CandeliteCards", bytemuck::cast_slice(&self.cards)),
                make("CandeliteMeshCards", bytemuck::cast_slice(&self.mesh_cards)),
            ));
            self.changed = false;
        }
        let (cards, mesh_cards) = self.gpu.as_ref().expect("made above");
        (cards, mesh_cards)
    }

    /// The atlas's layers, in the order of [`ATLAS_FORMATS`], and its depth.
    pub fn atlas(&self) -> (&[wgpu::Texture], &wgpu::Texture) {
        (&self.layers, &self.depth)
    }

    /// How many texels square the atlas is.
    pub fn size(&self) -> u32 {
        self.size
    }

    /// Each mesh slot's first card, or [`NO_CARDS`].
    pub fn mesh_cards(&self) -> &[u32] {
        &self.mesh_cards
    }

    /// Every card, six to a mesh.
    pub fn cards(&self) -> &[GpuCard] {
        &self.cards
    }
}

/// Whether `device` can trace rays: it needs wgpu's ray queries.
pub fn can_trace(device: &wgpu::Device) -> bool {
    device.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY)
}

/// A mesh's triangles, as an acceleration structure to trace against; and, for a cutout, the
/// texture coordinates of each triangle's corners and where its texture is solid.
struct Traced {
    blas: wgpu::Blas,
    // Kept alive while the structure is.
    _vertices: wgpu::Buffer,
    _indices: wgpu::Buffer,
    cutout: Option<(Vec<[f32; 2]>, Arc<CutoutMask>)>,
}

/// Where a mesh's cutout is, as the lookup reads it: by mesh slot, none wide for a solid mesh.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct CutoutRecord {
    /// Where its triangles' corners start, counted in triangles.
    first_triangle: u32,
    /// Where its mask starts, in words.
    first_word: u32,
    width: u32,
    height: u32,
}

/// What Candelite traces rays against: every mesh of the pool given its triangles, and every
/// instance placed in the pool, naming its slot (see the module's description).
pub struct TraceScene {
    meshes: HashMap<u32, Traced>,
    tlas: Option<wgpu::Tlas>,
    capacity: u32,
    /// What it was last built with, to build it again only when that changes: each instance's
    /// slot, mesh and transform.
    built: Vec<(u32, u32, [f32; 12])>,
    /// The meshes' cutouts on the GPU - records by mesh slot, corners, masks - made afresh when
    /// a mesh comes or goes.
    cutouts: Option<[wgpu::Buffer; 3]>,
}

impl Default for TraceScene {
    fn default() -> Self {
        Self::new()
    }
}

impl TraceScene {
    /// An empty scene.
    pub fn new() -> Self {
        Self {
            meshes: HashMap::new(),
            tlas: None,
            capacity: 0,
            built: Vec::new(),
            cutouts: None,
        }
    }

    /// Gives `mesh` its triangles: `positions` in its own space, every three `indices` a
    /// triangle; and for a cutout, each vertex's texture coordinates (as the renderer samples the
    /// texture with) and where the texture is solid. Built at once, on the queue.
    pub fn add_mesh(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        mesh: MeshId,
        positions: &[[f32; 3]],
        indices: &[u32],
        cutout: Option<(&[[f32; 2]], Arc<CutoutMask>)>,
    ) {
        let index_count = (indices.len() / 3 * 3) as u32;
        if positions.is_empty() || index_count == 0 {
            return;
        }
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteVertices"),
            contents: bytemuck::cast_slice(positions),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteIndices"),
            contents: bytemuck::cast_slice(&indices[..index_count as usize]),
            usage: wgpu::BufferUsages::BLAS_INPUT,
        });
        let size = wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: positions.len() as u32,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(index_count),
            // A cutout is not opaque, so that the lookup is asked about each hit on it.
            flags: if cutout.is_some() {
                wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION
            } else {
                wgpu::AccelerationStructureGeometryFlags::OPAQUE
            },
        };
        let blas = device.create_blas(
            &wgpu::CreateBlasDescriptor {
                label: Some("CandeliteMesh"),
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            },
            wgpu::BlasGeometrySizeDescriptors::Triangles {
                descriptors: vec![size.clone()],
            },
        );
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("CandeliteMesh"),
        });
        encoder.build_acceleration_structures(
            [&wgpu::BlasBuildEntry {
                blas: &blas,
                geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                    size: &size,
                    vertex_buffer: &vertices,
                    first_vertex: 0,
                    vertex_stride: 12,
                    index_buffer: Some(&index_buffer),
                    first_index: Some(0),
                    transform_buffer: None,
                    transform_buffer_offset: None,
                }]),
            }],
            std::iter::empty(),
        );
        queue.submit([encoder.finish()]);
        let cutout = cutout.map(|(uvs, mask)| {
            let corners = indices[..index_count as usize]
                .iter()
                .map(|&i| uvs.get(i as usize).copied().unwrap_or_default())
                .collect();
            (corners, mask)
        });
        self.meshes.insert(
            mesh.0,
            Traced {
                blas,
                _vertices: vertices,
                _indices: index_buffer,
                cutout,
            },
        );
        self.built.clear();
        self.cutouts = None;
    }

    /// Takes `mesh` out.
    pub fn remove_mesh(&mut self, mesh: MeshId) {
        if self.meshes.remove(&mesh.0).is_some() {
            self.built.clear();
            self.cutouts = None;
        }
    }

    /// The meshes' cutouts on the GPU: a record by mesh slot, each triangle's corners'
    /// texture coordinates, and the masks' words (see [`LOOKUP_WGSL`]).
    pub fn cutout_buffers(&mut self, device: &wgpu::Device) -> &[wgpu::Buffer; 3] {
        if self.cutouts.is_none() {
            let slots = self.meshes.keys().map(|&m| m as usize + 1).max().unwrap_or(1);
            let mut records = vec![CutoutRecord::default(); slots];
            let mut corners: Vec<[f32; 2]> = Vec::new();
            let mut words: Vec<u32> = Vec::new();
            // A mask shared by several meshes is put in once.
            let mut placed_masks: Vec<(*const CutoutMask, u32)> = Vec::new();
            for (&mesh, traced) in &self.meshes {
                let Some((mesh_corners, mask)) = &traced.cutout else {
                    continue;
                };
                let first_word = match placed_masks.iter().find(|(m, _)| *m == Arc::as_ptr(mask)) {
                    Some(&(_, first)) => first,
                    None => {
                        let first = words.len() as u32;
                        words.extend_from_slice(mask.words());
                        placed_masks.push((Arc::as_ptr(mask), first));
                        first
                    }
                };
                records[mesh as usize] = CutoutRecord {
                    first_triangle: (corners.len() / 3) as u32,
                    first_word,
                    width: mask.width(),
                    height: mask.height(),
                };
                corners.extend_from_slice(mesh_corners);
            }
            if corners.is_empty() {
                corners.push([0.0; 2]);
            }
            if words.is_empty() {
                words.push(0);
            }
            let buffer = |label, contents: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents,
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            self.cutouts = Some([
                buffer("CandeliteCutouts", bytemuck::cast_slice(&records)),
                buffer("CandeliteCutoutCorners", bytemuck::cast_slice(&corners)),
                buffer("CandeliteCutoutMasks", bytemuck::cast_slice(&words)),
            ]);
        }
        self.cutouts.as_ref().expect("made above")
    }

    /// Places a copy of its mesh for every instance placed in `pool` whose mesh it has, built
    /// into `encoder` if anything changed since last time. Whether there is anything to trace.
    pub fn update(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, pool: &GeometryPool) -> bool {
        let placed: Vec<(u32, u32, [f32; 12])> = pool
            .placed()
            .filter(|(_, record)| self.meshes.contains_key(&record.mesh))
            .map(|(slot, record)| (slot, record.mesh, record.transform))
            .collect();
        if placed.is_empty() {
            return false;
        }
        if placed == self.built && self.tlas.is_some() {
            return true;
        }
        let count = placed.len() as u32;
        if self.tlas.is_none() || count > self.capacity {
            self.capacity = count.next_power_of_two().max(64);
            self.tlas = Some(device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("CandeliteScene"),
                max_instances: self.capacity,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            }));
        }
        let tlas = self.tlas.as_mut().expect("made above");
        for i in 0..self.capacity as usize {
            tlas[i] = placed.get(i).map(|&(slot, mesh, transform)| {
                // The custom data holds 24 bits: slots past them would name the wrong instance.
                wgpu::TlasInstance::new(&self.meshes[&mesh].blas, transform, slot & 0x00FF_FFFF, 0xFF)
            });
        }
        encoder.build_acceleration_structures(std::iter::empty(), [&*tlas]);
        self.built = placed;
        true
    }

    /// The scene as last built.
    pub fn tlas(&self) -> Option<&wgpu::Tlas> {
        self.tlas.as_ref()
    }
}

/// The hit lookup in WGSL, for a shader to include (with `enable wgpu_ray_query;` at its top):
/// `candelite_trace(origin, direction, reach) -> SurfaceHit`. It binds, in group 0 from
/// binding 0: the scene ([`TraceScene::tlas`]), the pool's instances, each mesh slot's first card,
/// the cards ([`SurfaceCache::buffers`]), the atlas's diffuse, normal and depth layers, and the
/// meshes' cutouts ([`TraceScene::cutout_buffers`]): see [`lookup_layout`] and [`lookup_group`].
pub const LOOKUP_WGSL: &str = r#"
struct CandeliteInstance {
    transform: array<vec4f, 3>,
    mesh: u32,
    flags: u32,
    material: u32,
    morph: u32,
};

struct CandeliteCard {
    center: vec4f,
    u: vec4f,
    v: vec4f,
    tile: vec4u,
};

@group(0) @binding(0) var candeliteScene: acceleration_structure;
@group(0) @binding(1) var<storage, read> candeliteInstances: array<CandeliteInstance>;
@group(0) @binding(2) var<storage, read> candeliteMeshCards: array<u32>;
@group(0) @binding(3) var<storage, read> candeliteCards: array<CandeliteCard>;
@group(0) @binding(4) var candeliteDiffuse: texture_2d<f32>;
@group(0) @binding(5) var candeliteNormal: texture_2d<f32>;
@group(0) @binding(6) var candeliteDepth: texture_depth_2d;

// Each mesh slot's cutout, none wide for a solid mesh; each cutout triangle's corners' texture
// coordinates; and the masks' bits.
struct CandeliteCutout {
    firstTriangle: u32,
    firstWord: u32,
    width: u32,
    height: u32,
};
@group(0) @binding(7) var<storage, read> candeliteCutouts: array<CandeliteCutout>;
@group(0) @binding(8) var<storage, read> candeliteCorners: array<vec2f>;
@group(0) @binding(9) var<storage, read> candeliteMasks: array<u32>;

// The light on the cards (see `SurfaceLight`): each instance slot's first lit card, or all ones;
// each lit card's tile in the radiance atlas, and whose it is; and the atlas itself, a texel's
// light as four half floats, row after row.
struct CandeliteLitCard {
    tile: vec4u,
    slot: u32,
    face: u32,
    pad0: u32,
    pad1: u32,
};
@group(0) @binding(10) var<storage, read> candeliteInstanceLit: array<u32>;
@group(0) @binding(11) var<storage, read> candeliteLitCards: array<CandeliteLitCard>;
@group(0) @binding(12) var<storage, read> candeliteRadiance: array<vec2u>;
const CANDELITE_RADIANCE_WIDTH: u32 = 2048u;

// The atlas's diffuse colour is stored as the G-Buffer stores it, in sRGB: light works in linear.
fn candelite_linear(c: vec3f) -> vec3f {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3f(2.4)), step(vec3f(0.04045), c));
}

fn candelite_unpack(packed: vec2u) -> vec4f {
    return vec4f(unpack2x16float(packed.x), unpack2x16float(packed.y));
}

fn candelite_pack(value: vec4f) -> vec2u {
    return vec2u(pack2x16float(value.xy), pack2x16float(value.zw));
}

// The light on face `face` of instance `slot` at `s` across and `t` up it (each -1 to 1), and 1;
// or nothing, and 0, if it has no lit cards.
fn candelite_lit_at(slot: u32, face: u32, s: f32, t: f32) -> vec4f {
    if (slot >= arrayLength(&candeliteInstanceLit)) {
        return vec4f(0.0);
    }
    let first = candeliteInstanceLit[slot];
    if (first == 0xFFFFFFFFu) {
        return vec4f(0.0);
    }
    let tile = candeliteLitCards[first + face].tile;
    let within = min(vec2u(vec2f(s * 0.5 + 0.5, 0.5 - t * 0.5) * vec2f(tile.zw)), tile.zw - 1u);
    let texel = tile.xy + within;
    return vec4f(candelite_unpack(candeliteRadiance[texel.y * CANDELITE_RADIANCE_WIDTH + texel.x]).rgb, 1.0);
}

// Whether a ray crossing triangle `primitive` of `mesh` at `barycentrics` is stopped there: on a
// solid mesh, always; on a cutout, where its texture is solid. Texture coordinates repeat, as the
// renderer samples them.
fn candelite_stops(mesh: u32, primitive: u32, barycentrics: vec2f) -> bool {
    let cutout = candeliteCutouts[mesh];
    if (cutout.width == 0u) {
        return true;
    }
    let corner = (cutout.firstTriangle + primitive) * 3u;
    let uv = candeliteCorners[corner] * (1.0 - barycentrics.x - barycentrics.y)
        + candeliteCorners[corner + 1u] * barycentrics.x
        + candeliteCorners[corner + 2u] * barycentrics.y;
    let wrapped = fract(uv);
    let x = min(u32(wrapped.x * f32(cutout.width)), cutout.width - 1u);
    let y = min(u32(wrapped.y * f32(cutout.height)), cutout.height - 1u);
    let bit = y * cutout.width + x;
    return ((candeliteMasks[cutout.firstWord + bit / 32u] >> (bit % 32u)) & 1u) != 0u;
}

// What a ray found: whether it hit anything; whether the cards saw what it hit (`covered`), or
// at least something there - in front of it, as inside a crown, where every card sees the outer
// needles - from which what it looks like is taken (`seen`); how far it went, and where to; and
// what it hit looks like: its diffuse colour, and which way it faces. A hit the cards did not see
// is to be lit where it is, not from the cache, which holds the light on what they saw.
struct SurfaceHit {
    hit: bool,
    covered: bool,
    seen: bool,
    distance: f32,
    position: vec3f,
    diffuse: vec3f,
    normal: vec3f,
    // Whether the light on what was hit is known from its lit cards, and that light: what leaves
    // it towards the ray.
    lit: bool,
    radiance: vec3f,
};

// How many cutout triangles a ray asks about at most; past them, the next it crosses stops it. A
// ray across a forest at the height of its crowns crosses leaf card after leaf card, mostly
// through their see-through parts: unbounded, one such ray costs a wave as much as hundreds, and
// a frame with a few hundred of them stalls. Past this many layers of leaves, light hardly gets
// through anyway.
const CANDELITE_MOST_CUTOUTS: u32 = 32u;

// How far a card's captured surface may be from the hit, in meters, and still be taken as it:
// a few of its texels, as a surface sloping across a texel is captured at one depth.
const CANDELITE_DEPTH_SLACK: f32 = 0.3;

// The texel of a card's `tile` nearest `within` it (its texel there, counted from the tile's
// corner) that saw something, a ring of one texel round it, then of two; or all ones if none did.
fn candelite_nearest_seen(tile: vec4u, within: vec2u) -> vec2u {
    for (var ring = 1i; ring <= 2i; ring++) {
        for (var dy = -ring; dy <= ring; dy++) {
            for (var dx = -ring; dx <= ring; dx++) {
                if (max(abs(dx), abs(dy)) != ring) {
                    continue;
                }
                let at = vec2i(within) + vec2i(dx, dy);
                if (any(at < vec2i(0)) || any(at >= vec2i(tile.zw))) {
                    continue;
                }
                let texel = tile.xy + vec2u(at);
                if (textureLoad(candeliteDepth, texel, 0) < 1.0) {
                    return texel;
                }
            }
        }
    }
    return vec2u(0xFFFFFFFFu);
}

fn candelite_trace(origin: vec3f, direction: vec3f, reach: f32) -> SurfaceHit {
    var hit = SurfaceHit(false, false, false, reach, origin + direction * reach, vec3f(0.0), -direction, false, vec3f(0.0));
    var query: ray_query;
    rayQueryInitialize(&query, candeliteScene, RayDesc(0u, 0xFFu, 0.0, reach, origin, direction));
    // Only cutouts are asked about: whether the ray is stopped where it crosses them - up to
    // CANDELITE_MOST_CUTOUTS of them, past which the next stops it (see there).
    var asked = 0u;
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        let mesh = candeliteInstances[candidate.instance_custom_data].mesh;
        asked++;
        if (asked > CANDELITE_MOST_CUTOUTS || candelite_stops(mesh, candidate.primitive_index, candidate.barycentrics)) {
            rayQueryConfirmIntersection(&query);
        }
    }
    let found = rayQueryGetCommittedIntersection(&query);
    if (found.kind == RAY_QUERY_INTERSECTION_NONE) {
        return hit;
    }
    hit.hit = true;
    hit.distance = found.t;
    hit.position = origin + direction * found.t;
    let instance = candeliteInstances[found.instance_custom_data];
    let first = candeliteMeshCards[instance.mesh];
    if (first == 0xFFFFFFFFu) {
        return hit;
    }
    // In the mesh's own space, where its cards are.
    let p = found.world_to_object * vec4f(hit.position, 1.0);
    let d = normalize(found.world_to_object * vec4f(direction, 0.0));
    var diffuse = vec3f(0.0);
    var normal = vec3f(0.0);
    var weight = 0.0;
    var radiance = vec3f(0.0);
    var lit_weight = 0.0;
    // What the cards saw there, if not what was hit.
    var near_diffuse = vec3f(0.0);
    var near_normal = vec3f(0.0);
    var near_weight = 0.0;
    for (var c = 0u; c < 6u; c++) {
        let card = candeliteCards[first + c];
        let u = card.u.xyz;
        let v = card.v.xyz;
        let n = cross(u, v);
        let facing = -dot(n, d);
        if (facing <= 0.0) {
            continue;
        }
        let local = p - card.center.xyz;
        let s = dot(local, u) / card.u.w;
        let t = dot(local, v) / card.v.w;
        if (abs(s) > 1.0 || abs(t) > 1.0) {
            continue;
        }
        // Its tile's texel there: across from the left, up from the bottom.
        let size = card.tile.zw;
        let within = min(vec2u(vec2f(s * 0.5 + 0.5, 0.5 - t * 0.5) * vec2f(size)), size - 1u);
        let texel = card.tile.xy + within;
        // Where the card saw nothing, its depth is the far side's: never what was hit, however
        // thin the bounds.
        let share = textureLoad(candeliteDepth, texel, 0);
        if (share >= 1.0) {
            // A gap in what the card saw - foliage, captured a few texels to a meter, is full of
            // them: the nearest texel round it that saw something, if any, for what it looks like.
            let near = candelite_nearest_seen(card.tile, within);
            if (near.x == 0xFFFFFFFFu) {
                continue;
            }
            near_diffuse += textureLoad(candeliteDiffuse, near, 0).rgb * facing;
            near_normal += (textureLoad(candeliteNormal, near, 0).xyz * 2.0 - 1.0) * facing;
            near_weight += facing;
            continue;
        }
        let texel_diffuse = textureLoad(candeliteDiffuse, texel, 0).rgb * facing;
        let texel_normal = (textureLoad(candeliteNormal, texel, 0).xyz * 2.0 - 1.0) * facing;
        if (abs(share * card.center.w + dot(local, n)) > CANDELITE_DEPTH_SLACK) {
            near_diffuse += texel_diffuse;
            near_normal += texel_normal;
            near_weight += facing;
            continue;
        }
        diffuse += texel_diffuse;
        normal += texel_normal;
        weight += facing;
        let light = candelite_lit_at(found.instance_custom_data, c, s, t);
        radiance += light.rgb * facing;
        lit_weight += light.w * facing;
    }
    if (weight > 0.0) {
        hit.covered = true;
        hit.seen = true;
        if (lit_weight > 0.0) {
            hit.lit = true;
            hit.radiance = radiance / lit_weight;
        }
    } else if (near_weight > 0.0) {
        hit.seen = true;
        diffuse = near_diffuse;
        normal = near_normal;
        weight = near_weight;
    } else {
        return hit;
    }
    hit.diffuse = diffuse / weight;
    hit.normal = normalize(found.object_to_world * vec4f(normal / weight, 0.0));
    return hit;
}

// Whether anything stops a ray from `origin` along `direction` within `reach`: a shadow ray.
fn candelite_blocked(origin: vec3f, direction: vec3f, reach: f32) -> bool {
    var query: ray_query;
    // Ended at the first hit: which one does not matter.
    rayQueryInitialize(&query, candeliteScene, RayDesc(0x4u, 0xFFu, 0.0, reach, origin, direction));
    var asked = 0u;
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        let mesh = candeliteInstances[candidate.instance_custom_data].mesh;
        asked++;
        if (asked > CANDELITE_MOST_CUTOUTS || candelite_stops(mesh, candidate.primitive_index, candidate.barycentrics)) {
            rayQueryConfirmIntersection(&query);
        }
    }
    return rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE;
}

// The light leaving what a ray hit, back along it: from its lit cards if it has them; else lit
// where it is (hit lighting) - the sun, if a shadow ray reaches it, and half the sky; nothing if
// no card saw anything there. A ray that hit nothing sees the sky. `toward_sun` is a unit vector;
// `sun` the light a white surface facing the sun gives off; `sky` the sky's radiance.
fn candelite_hit_light(hit: SurfaceHit, toward_sun: vec3f, sun: vec3f, sky: vec3f) -> vec3f {
    if (!hit.hit) {
        return sky;
    }
    if (hit.lit) {
        return hit.radiance;
    }
    if (!hit.seen) {
        return vec3f(0.0);
    }
    var light = sky * 0.5;
    let facing = dot(hit.normal, toward_sun);
    if (facing > 0.0 && !candelite_blocked(hit.position + hit.normal * 0.05 + toward_sun * 0.15, toward_sun, 1.0e4)) {
        light += sun * facing;
    }
    return candelite_linear(hit.diffuse) * light;
}
"#;

/// How many texels square the radiance atlas is: as `CANDELITE_RADIANCE_WIDTH` in
/// [`LOOKUP_WGSL`].
pub const RADIANCE_SIZE: u32 = 2048;
/// Within how far of the eye, in meters, an instance has lit cards of its own: measured to the
/// nearest of its bounds, so a big mesh - the ground - underfoot is near however far its middle.
pub const LIT_REACH: f32 = 60.0;
/// How big an instance has to be - the radius of its bounds - for each meter its middle is from the
/// eye, to have lit cards: a tree does all through [`LIT_REACH`], a fern only within ten or twenty
/// meters. Rays from many small instances scattered through a forest are traced in no order a
/// GPU likes - lighting ten thousand floor plants' cards cost tens of milliseconds a frame - and a
/// small plant far off adds next to nothing to the light between surfaces; what hits it is lit
/// where it hit.
pub const LIT_SIZE: f32 = 0.03;

/// Within how far of the eye an instance's lit cards are as fine as its mesh's captured ones;
/// each doubling of the distance past it halves them, to [`LIT_COARSEST`] halvings.
pub const LIT_FULL: f32 = 8.0;
/// The most a lit card is halved.
pub const LIT_COARSEST: u32 = 5;
/// How many texels of the radiance atlas are lit a frame, at most: those of cards new to it
/// first, then the rest in turn.
pub const LIGHT_BUDGET: u32 = 1 << 17;
/// How many frames apart the instances' lit cards are handed out again, at most; sooner when the
/// eye has moved [`LIT_MOVE`] meters.
pub const LIT_EVERY: u32 = 30;
/// See [`LIT_EVERY`].
pub const LIT_MOVE: f32 = 2.0;

/// A lit card as the GPU reads it: its tile in the radiance atlas, the instance slot it lights,
/// and which of its mesh's cards it is.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuLitCard {
    /// x, y, width and height in texels, from the top left.
    pub tile: [u32; 4],
    /// The instance's slot in the pool.
    pub slot: u32,
    /// Which of its mesh's six cards.
    pub face: u32,
    /// Unused: keeps it a multiple of 16 bytes.
    pub _padding: [u32; 2],
}

/// The light Candelite lights the cards with.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sunlight {
    /// Towards the sun, a unit vector.
    pub toward_sun: [f32; 3],
    /// The light a white surface squarely facing the sun gives off, linear: as the renderer
    /// lights, its colour times its intensity over pi.
    pub sun: [f32; 3],
    /// The sky's radiance, the same all over, linear.
    pub sky: [f32; 3],
}

/// An instance's lit cards.
#[derive(Debug, Clone, PartialEq)]
struct Lit {
    first: u32,
    mesh: u32,
    level: u32,
    tiles: [[u32; 4]; 6],
}

const LIGHT_SHADER: &str = r#"
struct Light {
    towardSun: vec3f,
    frame: u32,
    sun: vec3f,
    blockCount: u32,
    sky: vec3f,
    pad0: u32,
};
@group(1) @binding(0) var<uniform> light: Light;
// Each texel to light: its lit card, and which texel of its tile, row by row. Packed, 64 to a
// workgroup whatever cards they are of: a ray costs a wave about as much as its slowest lane, so
// a coarse card's few texels must not each take a wave of their own.
@group(1) @binding(1) var<storage, read> blocks: array<vec2u>;
// The indirect light on each texel so far, and over how many updates it was averaged (alpha).
@group(1) @binding(2) var<storage, read_write> indirect: array<vec2u>;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Over how many updates indirect light is averaged, at most: a new sample's weight is never less
// than one over it.
const MOST_SAMPLES: f32 = 16.0;

@compute @workgroup_size(64)
fn light_cards(@builtin(global_invocation_id) id: vec3u) {
    let b = id.y * 65535u * 64u + id.x;
    if (b >= light.blockCount) {
        return;
    }
    let block = blocks[b];
    let lit = candeliteLitCards[block.x];
    let within = vec2u(block.y % lit.tile.z, block.y / lit.tile.z);
    let texel = lit.tile.xy + within;
    let index = texel.y * CANDELITE_RADIANCE_WIDTH + texel.x;
    let instance = candeliteInstances[lit.slot];
    let first = candeliteMeshCards[instance.mesh];
    if (first == 0xFFFFFFFFu) {
        candeliteRadiance[index] = vec2u(0u);
        return;
    }
    // The mesh's card under this texel, and what it captured there.
    let card = candeliteCards[first + lit.face];
    let uv = (vec2f(within) + 0.5) / vec2f(lit.tile.zw);
    let at = card.tile.xy + min(vec2u(uv * vec2f(card.tile.zw)), card.tile.zw - 1u);
    let share = textureLoad(candeliteDepth, at, 0);
    if (share >= 1.0) {
        candeliteRadiance[index] = vec2u(0u);
        indirect[index] = vec2u(0u);
        return;
    }
    // Where that is, in the mesh's space then the world's, and which way it faces.
    let s = uv.x * 2.0 - 1.0;
    let t = 1.0 - uv.y * 2.0;
    let n = cross(card.u.xyz, card.v.xyz);
    let p = card.center.xyz + card.u.xyz * (s * card.u.w) + card.v.xyz * (t * card.v.w) - n * (share * card.center.w);
    let m = instance.transform;
    let world = vec3f(dot(m[0], vec4f(p, 1.0)), dot(m[1], vec4f(p, 1.0)), dot(m[2], vec4f(p, 1.0)));
    let captured = textureLoad(candeliteNormal, at, 0).xyz * 2.0 - 1.0;
    let normal = normalize(vec3f(dot(m[0].xyz, captured), dot(m[1].xyz, captured), dot(m[2].xyz, captured)));
    let albedo = candelite_linear(textureLoad(candeliteDiffuse, at, 0).rgb);

    // The sun, if a shadow ray reaches it.
    var direct = vec3f(0.0);
    let facing = dot(normal, light.towardSun);
    if (facing > 0.0 && !candelite_blocked(world + normal * 0.05 + light.towardSun * 0.15, light.towardSun, 1.0e4)) {
        direct = albedo * light.sun * facing;
    }

    // One ray of indirect light, cosine-weighted about the normal, averaged with those before:
    // what it hit as lit (last frame's light on its cards, so light bounces again each frame), or
    // the sky.
    let a = pcg(index ^ pcg(light.frame));
    let b2 = pcg(a);
    let r1 = f32(a) / 4294967296.0;
    let r2 = f32(b2) / 4294967296.0;
    let side = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
    let tangent = normalize(cross(side, normal));
    let bitangent = cross(normal, tangent);
    let radius = sqrt(r1);
    let phi = 6.2831853 * r2;
    let direction = normalize(tangent * (radius * cos(phi)) + bitangent * (radius * sin(phi)) + normal * sqrt(max(1.0 - r1, 0.0)));
    let hit = candelite_trace(world + normal * 0.1, direction, 200.0);
    let incoming = candelite_hit_light(hit, light.towardSun, light.sun, light.sky);
    let before = candelite_unpack(indirect[index]);
    let count = min(before.a + 1.0, MOST_SAMPLES);
    let gathered = mix(before.rgb, albedo * incoming, 1.0 / count);
    indirect[index] = candelite_pack(vec4f(gathered, count));
    candeliteRadiance[index] = candelite_pack(vec4f(direct + gathered, 1.0));
}
"#;

/// The light on the cards (see the module's description): each instance near the eye given lit
/// cards of its own - its mesh's cards, in a radiance atlas, coarser the farther it is - lit by
/// the sun and by what their surfaces see, a budget of texels a frame.
pub struct SurfaceLight {
    shelves: ShelfAllocator,
    lit_cards: Vec<GpuLitCard>,
    /// Runs of six lit cards given back, by their first.
    free_lit: Vec<u32>,
    lit: HashMap<u32, Lit>,
    instance_lit: Vec<u32>,
    radiance: wgpu::Buffer,
    indirect: wgpu::Buffer,
    /// The lit cards and each slot's first, on the GPU, made afresh when they change.
    gpu: Option<(wgpu::Buffer, wgpu::Buffer)>,
    changed: bool,
    /// Runs never lit yet, by their first, lit before the rest.
    fresh: std::collections::VecDeque<u32>,
    /// Every run, by its first, and the next to light in turn.
    runs: Vec<u32>,
    cursor: usize,
    lookup_layout: wgpu::BindGroupLayout,
    light_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    frame: u32,
    handed_out: Option<([f32; 3], u32)>,
    full: bool,
}

impl SurfaceLight {
    /// The light, with nothing lit yet; [`None`] if `device` cannot trace rays.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !can_trace(device) {
            return None;
        }
        let buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: RADIANCE_SIZE as u64 * RADIANCE_SIZE as u64 * 8,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let lookup_layout = lookup_layout_with(device, wgpu::ShaderStages::COMPUTE, true);
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let light_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CandeliteLight"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, storage(true)),
                entry(2, storage(false)),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CandeliteLight"),
            source: wgpu::ShaderSource::Wgsl(format!("enable wgpu_ray_query;\n{}\n{LIGHT_SHADER}", lookup_writable()).into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CandeliteLight"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("CandeliteLight"),
                bind_group_layouts: &[Some(&lookup_layout), Some(&light_layout)],
                immediate_size: 0,
            })),
            module: &module,
            entry_point: Some("light_cards"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Self {
            shelves: ShelfAllocator::new(RADIANCE_SIZE),
            lit_cards: Vec::new(),
            free_lit: Vec::new(),
            lit: HashMap::new(),
            instance_lit: Vec::new(),
            radiance: buffer("CandeliteRadiance"),
            indirect: buffer("CandeliteIndirect"),
            gpu: None,
            changed: true,
            fresh: Default::default(),
            runs: Vec::new(),
            cursor: 0,
            lookup_layout,
            light_layout,
            pipeline,
            frame: 0,
            handed_out: None,
            full: false,
        })
    }

    /// How many halvings an instance `distance` meters from the eye has its cards lit at.
    pub fn level(distance: f32) -> u32 {
        if distance <= LIT_FULL {
            0
        } else {
            ((distance / LIT_FULL).log2().floor() as u32 + 1).min(LIT_COARSEST)
        }
    }

    /// Hands out lit cards to the instances within [`LIT_REACH`] of `eye`, nearest first, as
    /// fine as their distance calls for (see [`LIT_FULL`]), while the atlas has room; takes them
    /// from the rest. Whether it ran out of room.
    fn hand_out(&mut self, pool: &GeometryPool, cache: &SurfaceCache, eye: [f32; 3]) -> bool {
        let mesh_cards = cache.mesh_cards();
        let mut wanted: Vec<(f32, u32, u32, u32)> = pool
            .placed()
            .filter(|(_, record)| mesh_cards.get(record.mesh as usize).is_some_and(|&c| c != NO_CARDS))
            .filter_map(|(slot, record)| {
                let t = &record.transform;
                // The mesh's bounds, from its cards: +x, -x, +y, -y, +z, -z faces' middles.
                let cards = &cache.cards()[mesh_cards[record.mesh as usize] as usize..][..6];
                let middle: [f32; 3] = std::array::from_fn(|i| (cards[2 * i].center[i] + cards[2 * i + 1].center[i]) * 0.5);
                let half: [f32; 3] = std::array::from_fn(|i| (cards[2 * i].center[i] - cards[2 * i + 1].center[i]).abs() * 0.5);
                let scale = (0..3).map(|c| (t[c] * t[c] + t[4 + c] * t[4 + c] + t[8 + c] * t[8 + c]).sqrt()).fold(0.0, f32::max);
                let radius = (half[0] * half[0] + half[1] * half[1] + half[2] * half[2]).sqrt() * scale;
                let at: [f32; 3] = std::array::from_fn(|r| t[4 * r] * middle[0] + t[4 * r + 1] * middle[1] + t[4 * r + 2] * middle[2] + t[4 * r + 3]);
                let middle_distance = ((at[0] - eye[0]).powi(2) + (at[1] - eye[1]).powi(2) + (at[2] - eye[2]).powi(2)).sqrt();
                let d = (middle_distance - radius).max(0.0);
                let big_enough = radius >= middle_distance * LIT_SIZE;
                (d < LIT_REACH && big_enough).then(|| (d, slot, record.mesh, Self::level(d)))
            })
            .collect();
        wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
        let keep: HashMap<u32, (u32, u32)> = wanted.iter().map(|&(_, slot, mesh, level)| (slot, (mesh, level))).collect();
        let gone: Vec<u32> = self
            .lit
            .iter()
            .filter(|(slot, lit)| keep.get(slot) != Some(&(lit.mesh, lit.level)))
            .map(|(&slot, _)| slot)
            .collect();
        for slot in gone {
            self.release(slot);
        }
        let mut full = false;
        for (_, slot, mesh, level) in wanted {
            if self.lit.contains_key(&slot) {
                continue;
            }
            let first_card = mesh_cards[mesh as usize] as usize;
            let mut tiles = [[0u32; 4]; 6];
            let mut placed = 0;
            for (face, tile) in tiles.iter_mut().enumerate() {
                let [_, _, w, h] = cache.cards()[first_card + face].tile;
                let (w, h) = ((w >> level).max(2), (h >> level).max(2));
                match self.shelves.allocate(w, h) {
                    Some([x, y]) => {
                        *tile = [x, y, w, h];
                        placed += 1;
                    }
                    None => break,
                }
            }
            if placed < 6 {
                for tile in &tiles[..placed] {
                    self.shelves.free([tile[0], tile[1]], tile[2], tile[3]);
                }
                full = true;
                break;
            }
            let first = self.free_lit.pop().unwrap_or_else(|| {
                let first = self.lit_cards.len() as u32;
                self.lit_cards.resize(self.lit_cards.len() + 6, GpuLitCard::default());
                first
            });
            for (face, tile) in tiles.iter().enumerate() {
                self.lit_cards[first as usize + face] = GpuLitCard {
                    tile: *tile,
                    slot,
                    face: face as u32,
                    _padding: [0; 2],
                };
            }
            if self.instance_lit.len() <= slot as usize {
                self.instance_lit.resize(slot as usize + 1, NO_CARDS);
            }
            self.instance_lit[slot as usize] = first;
            self.lit.insert(slot, Lit { first, mesh, level, tiles });
            self.fresh.push_back(first);
            self.changed = true;
        }
        self.runs = self.lit.values().map(|lit| lit.first).collect();
        self.runs.sort_unstable();
        full
    }

    fn release(&mut self, slot: u32) {
        let Some(lit) = self.lit.remove(&slot) else {
            return;
        };
        for tile in lit.tiles {
            self.shelves.free([tile[0], tile[1]], tile[2], tile[3]);
        }
        self.free_lit.push(lit.first);
        self.fresh.retain(|&first| first != lit.first);
        self.instance_lit[slot as usize] = NO_CARDS;
        self.changed = true;
    }

    /// The tiles of the lit cards of the instance in `slot`, in the radiance atlas, if it has
    /// any: x, y, width and height in texels.
    pub fn lit_tiles(&self, slot: u32) -> Option<[[u32; 4]; 6]> {
        self.lit.get(&slot).map(|lit| lit.tiles)
    }

    /// The radiance atlas: each texel's light as four half floats (red, green, blue, 1), row
    /// after row, [`RADIANCE_SIZE`] to a row.
    pub fn radiance(&self) -> &wgpu::Buffer {
        &self.radiance
    }

    fn buffers(&mut self, device: &wgpu::Device) -> (&wgpu::Buffer, &wgpu::Buffer) {
        if self.changed || self.gpu.is_none() {
            let make = |label, contents: &[u8]| {
                device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: if contents.is_empty() { &[0xFF; 32] } else { contents },
                    usage: wgpu::BufferUsages::STORAGE,
                })
            };
            self.gpu = Some((
                make("CandeliteLitCards", bytemuck::cast_slice(&self.lit_cards)),
                make("CandeliteInstanceLit", bytemuck::cast_slice(&self.instance_lit)),
            ));
            self.changed = false;
        }
        let (cards, instances) = self.gpu.as_ref().expect("made above");
        (cards, instances)
    }

    /// Lights a budget of the cards ([`LIGHT_BUDGET`] texels) with `light`, recorded into
    /// `encoder`, handing out lit cards first if the eye has moved or it is time. `scene` must be
    /// up to date ([`TraceScene::update`]).
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &mut TraceScene,
        pool: &GeometryPool,
        cache: &mut SurfaceCache,
        eye: [f32; 3],
        light: &Sunlight,
    ) {
        self.frame = self.frame.wrapping_add(1);
        let due = self.handed_out.is_none_or(|(at, frame)| {
            let moved = (0..3).map(|i| (at[i] - eye[i]).powi(2)).sum::<f32>().sqrt();
            moved >= LIT_MOVE || self.frame.wrapping_sub(frame) >= LIT_EVERY
        });
        if due {
            self.full |= self.hand_out(pool, cache, eye);
            self.handed_out = Some((eye, self.frame));
        }
        // New runs first, then the rest in turn, to the budget.
        let mut chosen: Vec<u32> = Vec::new();
        let mut texels = 0u32;
        let size = |s: &Self, first: u32| (0..6).map(|f| s.lit_cards[(first + f) as usize].tile).map(|t| t[2] * t[3]).sum::<u32>();
        while let Some(&first) = self.fresh.front() {
            let cost = size(self, first);
            if texels + cost > LIGHT_BUDGET && !chosen.is_empty() {
                break;
            }
            self.fresh.pop_front();
            chosen.push(first);
            texels += cost;
        }
        let mut looked = 0;
        while looked < self.runs.len() && texels < LIGHT_BUDGET {
            self.cursor %= self.runs.len();
            let first = self.runs[self.cursor];
            self.cursor += 1;
            looked += 1;
            if chosen.contains(&first) {
                continue;
            }
            texels += size(self, first);
            chosen.push(first);
        }
        // Every texel of them, a lit card and a texel of its tile each.
        let mut blocks: Vec<[u32; 2]> = Vec::with_capacity(texels as usize);
        for first in chosen {
            for face in 0..6 {
                let card = first + face;
                let tile = self.lit_cards[card as usize].tile;
                blocks.extend((0..tile[2] * tile[3]).map(|t| [card, t]));
            }
        }
        if blocks.is_empty() {
            return;
        }
        self.buffers(device);
        let (lit_cards, instance_lit) = self.gpu.as_ref().expect("made above");
        let Some(lookup) = lookup_group_with(device, &self.lookup_layout, scene, pool, cache, [lit_cards, instance_lit, &self.radiance]) else {
            return;
        };
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            toward_sun: [f32; 3],
            frame: u32,
            sun: [f32; 3],
            block_count: u32,
            sky: [f32; 3],
            _padding: u32,
        }
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteLight"),
            contents: bytemuck::bytes_of(&Uniforms {
                toward_sun: light.toward_sun,
                frame: self.frame,
                sun: light.sun,
                block_count: blocks.len() as u32,
                sky: light.sky,
                _padding: 0,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let block_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteLightBlocks"),
            contents: bytemuck::cast_slice(&blocks),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CandeliteLight"),
            layout: &self.light_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniforms.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: block_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.indirect.as_entire_binding(),
                },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("CandeliteLight"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &lookup, &[]);
        pass.set_bind_group(1, &group, &[]);
        let groups = (blocks.len() as u32).div_ceil(64);
        pass.dispatch_workgroups(groups.min(65535), groups.div_ceil(65535), 1);
    }

    /// Whether the atlas has run out of room for every instance in reach.
    pub fn is_full(&self) -> bool {
        self.full
    }
}

/// [`LOOKUP_WGSL`] with the radiance atlas writable, for the pass that lights it.
fn lookup_writable() -> String {
    LOOKUP_WGSL.replace(
        "var<storage, read> candeliteRadiance",
        "var<storage, read_write> candeliteRadiance",
    )
}

/// How many nested grids the world radiance cache has round the eye (at most 4). A fourth, 16 m
/// apart, cost five to eight times each of the others - its probes high over the crowns and far
/// across the forest - and came in spikes that stalled frames.
pub const CACHE_LEVELS: u32 = 3;
/// How many probes each grid has along a side.
pub const CACHE_SIZE: u32 = 16;
/// How far apart the finest grid's probes are, in meters; each grid's twice the one before.
pub const CACHE_SPACING: f32 = 2.0;
/// How many of the cache's probes are traced a frame, at most: those new to it first, then the
/// rest in turn.
pub const CACHE_BUDGET: usize = 256;
/// How far the cache's probes' rays reach, at most, in meters: past it, the sky.
pub const CACHE_REACH: f32 = 64.0;
/// How far a screen probe's ray is traced, in meters, before the cache is asked what lies beyond.
pub const SCREEN_REACH: f32 = 8.0;
const CACHE_PROBES: u32 = CACHE_SIZE * CACHE_SIZE * CACHE_SIZE;

/// Directions as octahedral maps, and sampling the world radiance cache: for shaders including
/// it, with the cache bound at group `CACHE_GROUP` (see [`RadianceCache`]).
const CACHE_COMMON: &str = r#"
struct CacheLevels {
    // Each grid's first cell, in cells of its own spacing; and the spacings.
    origin: array<vec4i, 4>,
    spacing: vec4f,
};

// A unit vector to where it lies on an octahedral map, 0 to 1 each way (up the middle).
fn cache_octahedral(d: vec3f) -> vec2f {
    var p = d.xz / (abs(d.x) + abs(d.y) + abs(d.z));
    if (d.y < 0.0) {
        p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
    }
    return p * 0.5 + 0.5;
}

fn cache_direction(uv: vec2f) -> vec3f {
    var p = uv * 2.0 - 1.0;
    let y = 1.0 - abs(p.x) - abs(p.y);
    if (y < 0.0) {
        p = (1.0 - abs(p.yx)) * select(vec2f(-1.0), vec2f(1.0), p >= vec2f(0.0));
    }
    return normalize(vec3f(p.x, y, p.y));
}

// The cache's slot for `cell` of grid `level`: cells wrap round the grid.
fn cache_slot(level: u32, cell: vec3i) -> u32 {
    let s = ((cell % 16) + 16) % 16;
    return level * 4096u + u32(s.x) + 16u * (u32(s.y) + 16u * u32(s.z));
}
"#;

const CACHE_SAMPLE: &str = r#"
@group(2) @binding(0) var<uniform> cacheLevels: CacheLevels;
@group(2) @binding(1) var<storage, read> cacheCells: array<vec4i>;
@group(2) @binding(2) var<storage, read> cacheRadiance: array<vec2u>;

// The light arriving at `x` from `d`, from the probes round it of the finest grid that holds it,
// and 1; or nothing, and 0, if no grid holds it or none of the probes round it can be used.
fn cache_radiance(x: vec3f, d: vec3f) -> vec4f {
    let texel = vec2u(clamp(cache_octahedral(d) * 8.0, vec2f(0.0), vec2f(7.999)));
    let within = texel.y * 8u + texel.x;
    for (var level = 0u; level < 4u; level++) {
        let g = x / cacheLevels.spacing[level];
        let base = vec3i(floor(g));
        let origin = cacheLevels.origin[level].xyz;
        if (any(base < origin) || any(base + 1 >= origin + 16)) {
            continue;
        }
        let f = g - vec3f(base);
        var sum = vec3f(0.0);
        var weight = 0.0;
        for (var k = 0; k < 8; k++) {
            let offset = vec3i(k & 1, (k >> 1) & 1, k >> 2);
            let cell = base + offset;
            let slot = cache_slot(level, cell);
            let stored = cacheCells[slot];
            if (any(stored.xyz != cell) || stored.w != 1) {
                continue;
            }
            let w = select(1.0 - f.x, f.x, offset.x == 1) * select(1.0 - f.y, f.y, offset.y == 1) * select(1.0 - f.z, f.z, offset.z == 1);
            sum += candelite_unpack(cacheRadiance[slot * 64u + within]).rgb * w;
            weight += w;
        }
        if (weight > 0.01) {
            return vec4f(sum / weight, 1.0);
        }
    }
    return vec4f(0.0);
}
"#;

const CACHE_UPDATE: &str = r#"
struct CacheUpdate {
    levels: CacheLevels,
    towardSun: vec4f,
    sun: vec4f,
    sky: vec4f,
    frame: u32,
    jobCount: u32,
    pad0: u32,
    pad1: u32,
};
@group(1) @binding(0) var<uniform> update: CacheUpdate;
@group(1) @binding(1) var<storage, read> jobs: array<u32>;
@group(1) @binding(2) var<storage, read_write> cells: array<vec4i>;
@group(1) @binding(3) var<storage, read_write> radiance: array<vec2u>;

fn cache_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

var<workgroup> backfaces: atomic<u32>;

// A probe a workgroup, a texel of its octahedral map a lane: its ray, through a point of the
// texel turned each frame, the light it brings back averaged with what the texel had - unless the
// slot has just been given a new cell. A probe whose rays mostly hit the backs of things is inside
// something, and is not used.
@compute @workgroup_size(8, 8)
fn update_cache(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u, @builtin(local_invocation_index) lane: u32) {
    let job_index = group.y * 65535u + group.x;
    if (job_index >= update.jobCount) {
        return;
    }
    if (lane == 0u) {
        atomicStore(&backfaces, 0u);
    }
    workgroupBarrier();
    let job = jobs[job_index];
    let level = job / 4096u;
    let s = vec3i(i32(job % 16u), i32((job / 16u) % 16u), i32((job / 256u) % 16u));
    let origin = update.levels.origin[level].xyz;
    // The cell of the grid's window this slot holds now.
    let cell = origin + (((s - ((origin % 16) + 16) % 16) % 16) + 16) % 16;
    let spacing = update.levels.spacing[level];
    let position = vec3f(cell) * spacing;
    let turn = cache_pcg(job * 7919u + update.frame * 104729u + lane);
    let jitter = vec2f(f32(turn & 0xFFFFu), f32(turn >> 16u)) / 65536.0;
    let direction = cache_direction((vec2f(local.xy) + jitter) / 8.0);
    // Out to 16 of its grid's cells, but never past CACHE_REACH: past that, the next grid's
    // probes see - and the coarsest grid's long rays, from high over the crowns down through
    // them, cost many times the rest.
    let hit = candelite_trace(position, direction, min(spacing * 16.0, CACHE_REACH));
    if (hit.hit && dot(hit.normal, direction) > 0.0) {
        atomicAdd(&backfaces, 1u);
    }
    // Lit where it hit, if it has no lit cards: with a shadow ray if it is within 4 m; further,
    // without
    // - a ray from the forest's far side towards the sun crosses crown after crown, and costs
    // many times one near by - as half in the sun.
    var light: vec3f;
    if (!hit.hit || hit.lit || !hit.seen || hit.distance < 4.0) {
        light = candelite_hit_light(hit, update.towardSun.xyz, update.sun.xyz, update.sky.xyz);
    } else {
        let facing = max(dot(hit.normal, update.towardSun.xyz), 0.0);
        light = candelite_linear(hit.diffuse) * (update.sky.xyz * 0.5 + update.sun.xyz * facing * 0.5);
    }
    let stored = cells[job];
    let fresh = any(stored.xyz != cell) || stored.w == 0;
    let index = job * 64u + lane;
    var kept = light;
    if (!fresh) {
        kept = mix(candelite_unpack(radiance[index]).rgb, light, 0.25);
    }
    radiance[index] = candelite_pack(vec4f(kept, 1.0));
    workgroupBarrier();
    if (lane == 0u) {
        let inside = atomicLoad(&backfaces) > 16u;
        cells[job] = vec4i(cell, select(1, 2, inside));
    }
}
"#;

/// The bind group layout the world radiance cache is sampled through (`cache_radiance`).
fn cache_sample_layout(device: &wgpu::Device) -> wgpu::BindGroupLayout {
    let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty,
        count: None,
    };
    let storage = wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only: true },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("CandeliteCacheSample"),
        entries: &[
            entry(
                0,
                wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
            ),
            entry(1, storage),
            entry(2, storage),
        ],
    })
}

/// Candelite's world radiance cache, after Lumen's: [`CACHE_LEVELS`] nested grids of probes round
/// the eye, [`CACHE_SIZE`] a side, the finest [`CACHE_SPACING`] apart and each the next twice as
/// far; each probe keeps the light reaching it from every direction, as an 8 by 8 octahedral map,
/// traced against the lit cards and averaged over frames. Screen probes trace only
/// [`SCREEN_REACH`], and take the light from beyond it from here: long rays cost less, and the
/// light they bring back flickers less. The grids move with the eye a cell at a time; a slot
/// handed a new cell is traced before the rest, which are traced again in turn,
/// [`CACHE_BUDGET`] probes a frame.
pub struct RadianceCache {
    cells: wgpu::Buffer,
    radiance: wgpu::Buffer,
    lookup_layout: wgpu::BindGroupLayout,
    update_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    origins: [Option<[i32; 3]>; CACHE_LEVELS as usize],
    /// Slots handed new cells, waiting to be traced, and whether each is waiting.
    waiting: std::collections::VecDeque<u32>,
    queued: Vec<bool>,
    cursor: u32,
    frame: u32,
    levels: Option<wgpu::Buffer>,
}

impl RadianceCache {
    /// An empty cache; [`None`] if `device` cannot trace rays.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !can_trace(device) {
            return None;
        }
        let probes = (CACHE_LEVELS * CACHE_PROBES) as u64;
        let cells = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CandeliteCacheCells"),
            size: probes * 16,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let radiance = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("CandeliteCacheRadiance"),
            size: probes * 64 * 8,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let lookup_layout = lookup_layout(device, wgpu::ShaderStages::COMPUTE);
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let update_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CandeliteCacheUpdate"),
            entries: &[
                entry(
                    0,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(1, storage(true)),
                entry(2, storage(false)),
                entry(3, storage(false)),
            ],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CandeliteCacheUpdate"),
            source: wgpu::ShaderSource::Wgsl(format!("enable wgpu_ray_query;\n{LOOKUP_WGSL}\n{CACHE_COMMON}\nconst CACHE_REACH: f32 = {CACHE_REACH:?};\n{CACHE_UPDATE}").into()),
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CandeliteCacheUpdate"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("CandeliteCacheUpdate"),
                bind_group_layouts: &[Some(&lookup_layout), Some(&update_layout)],
                immediate_size: 0,
            })),
            module: &module,
            entry_point: Some("update_cache"),
            compilation_options: Default::default(),
            cache: None,
        });
        Some(Self {
            cells,
            radiance,
            lookup_layout,
            update_layout,
            pipeline,
            origins: [None; CACHE_LEVELS as usize],
            waiting: Default::default(),
            queued: vec![false; (CACHE_LEVELS * CACHE_PROBES) as usize],
            cursor: 0,
            frame: 0,
            levels: None,
        })
    }

    /// Each grid's spacing, in meters.
    pub fn spacing(level: u32) -> f32 {
        CACHE_SPACING * (1u32 << level) as f32
    }

    /// Each grid's first cell, for the eye at `eye`: the eye in its middle.
    fn origin(level: u32, eye: [f32; 3]) -> [i32; 3] {
        eye.map(|e| (e / Self::spacing(level)).floor() as i32 - CACHE_SIZE as i32 / 2)
    }

    fn levels_bytes(&self) -> [u8; 80] {
        let mut words = [0u32; 20];
        for (level, origin) in self.origins.iter().enumerate() {
            let o = origin.unwrap_or([i32::MIN / 2; 3]);
            for i in 0..3 {
                words[level * 4 + i] = o[i] as u32;
            }
        }
        for level in 0..CACHE_LEVELS {
            words[16 + level as usize] = Self::spacing(level).to_bits();
        }
        let mut out = [0u8; 80];
        out.copy_from_slice(bytemuck::cast_slice(&words));
        out
    }

    /// Moves the grids with `eye`, and traces [`CACHE_BUDGET`] of the probes against what `light`
    /// has lit, recorded into `encoder`.
    #[allow(clippy::too_many_arguments)]
    pub fn update(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &mut TraceScene,
        pool: &GeometryPool,
        cache: &mut SurfaceCache,
        light: &mut SurfaceLight,
        eye: [f32; 3],
        sunlight: &Sunlight,
    ) {
        self.frame = self.frame.wrapping_add(1);
        let n = CACHE_SIZE as i32;
        for level in 0..CACHE_LEVELS {
            let origin = Self::origin(level, eye);
            let before = self.origins[level as usize];
            if before == Some(origin) {
                continue;
            }
            // Every cell of the new window not in the old one: its slot is handed it.
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        let cell = [origin[0] + x, origin[1] + y, origin[2] + z];
                        let was_in = before.is_some_and(|b| (0..3).all(|i| cell[i] >= b[i] && cell[i] < b[i] + n));
                        if was_in {
                            continue;
                        }
                        let s = cell.map(|c| c.rem_euclid(n) as u32);
                        let slot = level * CACHE_PROBES + s[0] + CACHE_SIZE * (s[1] + CACHE_SIZE * s[2]);
                        if !self.queued[slot as usize] {
                            self.queued[slot as usize] = true;
                            self.waiting.push_back(slot);
                        }
                    }
                }
            }
            self.origins[level as usize] = Some(origin);
        }
        let mut jobs: Vec<u32> = Vec::with_capacity(CACHE_BUDGET);
        while jobs.len() < CACHE_BUDGET {
            let Some(slot) = self.waiting.pop_front() else {
                break;
            };
            self.queued[slot as usize] = false;
            jobs.push(slot);
        }
        let total = CACHE_LEVELS * CACHE_PROBES;
        while jobs.len() < CACHE_BUDGET {
            // In a scrambled order (an odd multiplier, the slots a power of two): probes next to
            // each other cost alike - a slab of them inside the crowns, say - and are spread
            // over many frames rather than taken together.
            let slot = self.cursor.wrapping_mul(2_654_435_761) % total;
            self.cursor = (self.cursor + 1) % total;
            if !self.queued[slot as usize] {
                jobs.push(slot);
            }
        }
        let Some(lookup) = lookup_group(device, &self.lookup_layout, scene, pool, cache, light) else {
            return;
        };
        let levels = self.levels_bytes();
        self.levels = Some(device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteCacheLevels"),
            contents: &levels,
            usage: wgpu::BufferUsages::UNIFORM,
        }));
        let mut uniform = Vec::with_capacity(144);
        uniform.extend_from_slice(&levels);
        let four = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        uniform.extend_from_slice(bytemuck::cast_slice(&four(sunlight.toward_sun)));
        uniform.extend_from_slice(bytemuck::cast_slice(&four(sunlight.sun)));
        uniform.extend_from_slice(bytemuck::cast_slice(&four(sunlight.sky)));
        uniform.extend_from_slice(bytemuck::cast_slice(&[self.frame, jobs.len() as u32, 0, 0]));
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteCacheUpdate"),
            contents: &uniform,
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let job_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteCacheJobs"),
            contents: bytemuck::cast_slice(&jobs),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CandeliteCacheUpdate"),
            layout: &self.update_layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: uniforms.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: job_buffer.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: self.radiance.as_entire_binding() },
            ],
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("CandeliteCacheUpdate"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &lookup, &[]);
        pass.set_bind_group(1, &group, &[]);
        let count = jobs.len() as u32;
        pass.dispatch_workgroups(count.min(65535), count.div_ceil(65535), 1);
    }

    /// The bind group to sample the cache through, for `layout` (see `cache_sample_layout`);
    /// [`None`] before the first [`Self::update`].
    fn sample_group(&self, device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Option<wgpu::BindGroup> {
        let levels = self.levels.as_ref()?;
        Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CandeliteCacheSample"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: levels.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: self.cells.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: self.radiance.as_entire_binding() },
            ],
        }))
    }
}

/// How many pixels apart screen probes are, each way.
pub const PROBE_SPACING: u32 = 16;
/// The format [`ScreenProbes`] draws the light between surfaces in: per pixel, what to light it
/// with in place of the ambient colour - its diffuse colour times this is the light it gives off.
pub const PROBES_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;

/// The probes' parameters and the screen's depth and normals, shared by both their shaders.
const PROBES_COMMON: &str = r#"
struct Probes {
    inverseViewProjection: mat4x4f,
    previousViewProjection: mat4x4f,
    towardSun: vec4f,
    sun: vec4f,
    sky: vec4f,
    size: vec2u,
    grid: vec2u,
    frame: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
};
// A probe: where it is (w 1 if it is on something), which way it faces (w how many frames it
// has been averaged over), and the light reaching it as L1 spherical harmonics, red, green and
// blue (each the constant term, then y, z, x).
struct Probe {
    position: vec4f,
    normal: vec4f,
    r: vec4f,
    g: vec4f,
    b: vec4f,
};
@group(1) @binding(0) var<uniform> probes: Probes;
@group(1) @binding(1) var sceneDepth: texture_depth_2d;
@group(1) @binding(2) var sceneNormal: texture_2d<f32>;

// Where the surface seen at `pixel` is, from its depth (OpenGL's clip space, depth 0 to 1).
fn probes_unproject(pixel: vec2f, depth: f32) -> vec3f {
    let uv = pixel / vec2f(probes.size);
    let clip = vec4f(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, depth * 2.0 - 1.0, 1.0);
    let p = probes.inverseViewProjection * clip;
    return p.xyz / p.w;
}

fn probes_normal(pixel: vec2u) -> vec3f {
    return normalize(textureLoad(sceneNormal, pixel, 0).xyz * 2.0 - 1.0);
}

// The irradiance at a surface facing `n`, over pi - what to light a diffuse colour with - from a
// probe's harmonics.
fn probes_irradiance(probe: Probe, n: vec3f) -> vec3f {
    let band1 = vec4f(0.0, 0.488603 * 2.0943951 * n.y, 0.488603 * 2.0943951 * n.z, 0.488603 * 2.0943951 * n.x);
    // E(n) = pi c0 Y0 + 2 pi / 3 c1 . Y1(n): the clamped cosine convolved with the harmonics.
    let constant = 0.282095 * 3.14159265;
    let r = probe.r.x * constant + dot(probe.r.yzw, band1.yzw);
    let g = probe.g.x * constant + dot(probe.g.yzw, band1.yzw);
    let b = probe.b.x * constant + dot(probe.b.yzw, band1.yzw);
    return max(vec3f(r, g, b), vec3f(0.0)) / 3.14159265;
}
"#;

const PROBES_TRACE: &str = r#"
@group(1) @binding(3) var<storage, read> history: array<Probe>;
@group(1) @binding(4) var<storage, read_write> current: array<Probe>;

fn probes_pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

var<workgroup> gatheredR: array<vec4f, 64>;
var<workgroup> gatheredG: array<vec4f, 64>;
var<workgroup> gatheredB: array<vec4f, 64>;

// Over how many frames a probe's light is averaged, at most.
const PROBE_HISTORY: f32 = 12.0;

// A probe a workgroup, a ray a lane: placed on what is seen at a pixel of its 16 by 16 tile, a
// different one each frame; its 64 rays spread evenly over the hemisphere about its normal,
// turned each frame; what each brings back projected onto the harmonics; then averaged with what
// the probe there saw last frame, if it is on the same surface.
@compute @workgroup_size(8, 8)
fn trace(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u, @builtin(local_invocation_index) lane: u32) {
    let index = group.y * probes.grid.x + group.x;
    let seed = probes_pcg(index * 9781u + probes.frame * 6271u);
    let jitter = vec2u(seed % 16u, (seed >> 8u) % 16u);
    let pixel = min(group.xy * 16u + jitter, probes.size - 1u);
    let depth = textureLoad(sceneDepth, pixel, 0);
    let on_something = depth < 1.0;
    let position = probes_unproject(vec2f(pixel) + 0.5, depth);
    let normal = probes_normal(pixel);

    var r = vec4f(0.0);
    var g = vec4f(0.0);
    var b = vec4f(0.0);
    if (on_something) {
        let turn = probes_pcg(seed ^ lane);
        let u1 = (f32(local.x) + f32(turn & 0xFFFFu) / 65536.0) / 8.0;
        let u2 = (f32(local.y) + f32(turn >> 16u) / 65536.0) / 8.0;
        // Uniform over the hemisphere: up the normal by u1, round it by u2.
        let up = u1;
        let out = sqrt(max(1.0 - up * up, 0.0));
        let phi = 6.2831853 * u2;
        let side = select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(normal.y) > 0.9);
        let tangent = normalize(cross(side, normal));
        let bitangent = cross(normal, tangent);
        let direction = normalize(tangent * (out * cos(phi)) + bitangent * (out * sin(phi)) + normal * up);
        // Traced a little way; past that, the light from beyond is the world radiance cache's,
        // where it has probes round the ray's end - else traced on.
        let start = position + normal * 0.1;
        let hit = candelite_trace(start, direction, SCREEN_REACH);
        var light: vec3f;
        if (hit.hit) {
            light = candelite_hit_light(hit, probes.towardSun.xyz, probes.sun.xyz, probes.sky.xyz);
        } else {
            let end = start + direction * SCREEN_REACH;
            let cached = cache_radiance(end, direction);
            if (cached.w > 0.0) {
                light = cached.rgb;
            } else {
                let far = candelite_trace(end, direction, 1.0e3);
                light = candelite_hit_light(far, probes.towardSun.xyz, probes.sun.xyz, probes.sky.xyz);
            }
        }
        // Each ray stands for 2 pi / 64 of the hemisphere.
        let basis = vec4f(0.282095, 0.488603 * direction.y, 0.488603 * direction.z, 0.488603 * direction.x) * (6.2831853 / 64.0);
        r = basis * light.r;
        g = basis * light.g;
        b = basis * light.b;
    }
    gatheredR[lane] = r;
    gatheredG[lane] = g;
    gatheredB[lane] = b;
    workgroupBarrier();
    if (lane != 0u) {
        return;
    }
    var probe = Probe(vec4f(position, select(0.0, 1.0, on_something)), vec4f(normal, 1.0), vec4f(0.0), vec4f(0.0), vec4f(0.0));
    if (!on_something) {
        current[index] = probe;
        return;
    }
    for (var i = 0u; i < 64u; i++) {
        probe.r += gatheredR[i];
        probe.g += gatheredG[i];
        probe.b += gatheredB[i];
    }
    // Last frame's probe where this one was then, if it was on the same surface.
    let before = probes.previousViewProjection * vec4f(position, 1.0);
    if (before.w > 0.0) {
        let ndc = before.xy / before.w;
        let then = vec2f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5) * vec2f(probes.size);
        if (all(then >= vec2f(0.0)) && all(then < vec2f(probes.size))) {
            let cell = min(vec2u(then) / 16u, probes.grid - 1u);
            let old = history[cell.y * probes.grid.x + cell.x];
            let distance = length(position - probes_unproject(vec2f(probes.size) * 0.5, 0.0));
            let apart = abs(dot(old.position.xyz - position, normal));
            if (old.position.w > 0.5 && dot(old.normal.xyz, normal) > 0.8 && apart < 0.05 + 0.02 * distance) {
                let count = min(old.normal.w + 1.0, PROBE_HISTORY);
                probe.r = mix(old.r, probe.r, 1.0 / count);
                probe.g = mix(old.g, probe.g, 1.0 / count);
                probe.b = mix(old.b, probe.b, 1.0 / count);
                probe.normal.w = count;
            }
        }
    }
    current[index] = probe;
}
"#;

const PROBES_GATHER: &str = r#"
@group(1) @binding(3) var<storage, read> current: array<Probe>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

// Each pixel lit from the probes round it: the four nearest, each by how near it is, and only
// as far as it lies on the pixel's surface and faces its way; if none does, any of the nine
// round it facing its way; if none, the sky.
@fragment
fn fs_main(@builtin(position) at: vec4f) -> @location(0) vec4f {
    let pixel = vec2u(at.xy);
    let depth = textureLoad(sceneDepth, pixel, 0);
    if (depth >= 1.0) {
        return vec4f(probes.sky.rgb, 1.0);
    }
    let p = probes_unproject(at.xy, depth);
    let n = probes_normal(pixel);
    let eye = probes_unproject(vec2f(probes.size) * 0.5, 0.0);
    let slack = 0.05 + 0.02 * length(p - eye);
    let cell = (at.xy - 8.0) / 16.0;
    let base = vec2i(floor(cell));
    let f = fract(cell);
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let offset = vec2i(k & 1, k >> 1);
        let c = clamp(base + offset, vec2i(0), vec2i(probes.grid) - 1);
        let probe = current[u32(c.y) * probes.grid.x + u32(c.x)];
        if (probe.position.w < 0.5) {
            continue;
        }
        let near = select(1.0 - f.x, f.x, offset.x == 1) * select(1.0 - f.y, f.y, offset.y == 1) + 1.0e-3;
        let plane = exp(-abs(dot(probe.position.xyz - p, n)) / slack);
        let facing = pow(max(dot(probe.normal.xyz, n), 0.0), 4.0);
        let w = near * plane * facing;
        sum += probes_irradiance(probe, n) * w;
        weight += w;
    }
    if (weight < 1.0e-4) {
        for (var dy = -1; dy <= 1; dy++) {
            for (var dx = -1; dx <= 1; dx++) {
                let c = clamp(vec2i(floor(at.xy / 16.0)) + vec2i(dx, dy), vec2i(0), vec2i(probes.grid) - 1);
                let probe = current[u32(c.y) * probes.grid.x + u32(c.x)];
                if (probe.position.w < 0.5) {
                    continue;
                }
                let w = max(dot(probe.normal.xyz, n), 0.0) + 1.0e-3;
                sum += probes_irradiance(probe, n) * w;
                weight += w;
            }
        }
    }
    if (weight <= 0.0) {
        return vec4f(probes.sky.rgb, 1.0);
    }
    return vec4f(sum / weight, 1.0);
}
"#;

/// Candelite's final gather: the light between surfaces brought to every pixel, after Lumen's
/// screen probes. A probe is placed every [`PROBE_SPACING`] pixels each way, on what the screen
/// sees there (a different pixel of its tile each frame); it traces 64 rays over the hemisphere
/// about its normal against the lit cards ([`SurfaceLight`]), and keeps what they bring back as
/// spherical harmonics, averaged with what the probe there saw last frame if it is on the same
/// surface. Each pixel is then lit from the probes round it on its surface, at its own normal:
/// what the renderer lights its diffuse colour with in place of the ambient colour.
pub struct ScreenProbes {
    lookup_layout: wgpu::BindGroupLayout,
    trace_layout: wgpu::BindGroupLayout,
    gather_layout: wgpu::BindGroupLayout,
    trace: wgpu::ComputePipeline,
    gather: wgpu::RenderPipeline,
    empty_layout: wgpu::BindGroupLayout,
    cache_layout: wgpu::BindGroupLayout,
    /// This frame's probes and last frame's, swapped each frame, with the grid they were for.
    buffers: Option<([wgpu::Buffer; 2], [u32; 2])>,
    previous_view_projection: Option<[[f32; 4]; 4]>,
    frame: u32,
}

/// What [`ScreenProbes::record`] needs of the frame.
pub struct ProbeFrame<'a> {
    /// The scene's depth, as the G-Buffer has it (OpenGL's clip space, 0 to 1): a depth texture.
    pub depth: &'a wgpu::Texture,
    /// The scene's normals, as the G-Buffer has them (each axis brought to 0 to 1).
    pub normal: &'a wgpu::Texture,
    /// From the world to clip space, column by column (OpenGL's).
    pub view_projection: [[f32; 4]; 4],
    /// And back.
    pub inverse_view_projection: [[f32; 4]; 4],
    /// Where to draw the light: of [`PROBES_FORMAT`], the size of `depth`.
    pub target: &'a wgpu::Texture,
}

impl ScreenProbes {
    /// The probes' pipelines; [`None`] if `device` cannot trace rays.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !can_trace(device) {
            return None;
        }
        let lookup_layout = lookup_layout(device, wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT);
        let both = wgpu::ShaderStages::COMPUTE | wgpu::ShaderStages::FRAGMENT;
        let entry = |binding, ty| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: both,
            ty,
            count: None,
        };
        let uniform = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let depth = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Depth,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let normal = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let storage = |read_only| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let trace_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CandeliteProbes"),
            entries: &[entry(0, uniform), entry(1, depth), entry(2, normal), entry(3, storage(true)), entry(4, storage(false))],
        });
        let gather_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CandeliteProbeGather"),
            entries: &[entry(0, uniform), entry(1, depth), entry(2, normal), entry(3, storage(true))],
        });
        let cache_layout = cache_sample_layout(device);
        let trace_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CandeliteProbes"),
            source: wgpu::ShaderSource::Wgsl(
                format!(
                    "enable wgpu_ray_query;\n{LOOKUP_WGSL}\n{PROBES_COMMON}\n{CACHE_COMMON}\n{CACHE_SAMPLE}\nconst SCREEN_REACH: f32 = {SCREEN_REACH:?};\n{PROBES_TRACE}"
                )
                .into(),
            ),
        });
        let gather_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CandeliteProbeGather"),
            source: wgpu::ShaderSource::Wgsl(format!("{PROBES_COMMON}\n{PROBES_GATHER}").into()),
        });
        let trace = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("CandeliteProbes"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("CandeliteProbes"),
                bind_group_layouts: &[Some(&lookup_layout), Some(&trace_layout), Some(&cache_layout)],
                immediate_size: 0,
            })),
            module: &trace_module,
            entry_point: Some("trace"),
            compilation_options: Default::default(),
            cache: None,
        });
        // The gather binds nothing of the lookup: group 0 is left empty.
        let empty = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: None, entries: &[] });
        let gather = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CandeliteProbeGather"),
            layout: Some(&device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("CandeliteProbeGather"),
                bind_group_layouts: &[Some(&empty), Some(&gather_layout)],
                immediate_size: 0,
            })),
            vertex: wgpu::VertexState {
                module: &gather_module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &gather_module,
                entry_point: Some("fs_main"),
                targets: &[Some(PROBES_FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Some(Self {
            lookup_layout,
            trace_layout,
            gather_layout,
            trace,
            gather,
            empty_layout: empty,
            cache_layout,
            buffers: None,
            previous_view_projection: None,
            frame: 0,
        })
    }

    /// Places and traces the probes for `frame` against what `light` has lit - beyond
    /// [`SCREEN_REACH`], taking the light from `radiance` - and draws the light they gathered into
    /// `frame.target`, recorded into `encoder`. `radiance` must have been updated
    /// ([`RadianceCache::update`]).
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &mut self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &mut TraceScene,
        pool: &GeometryPool,
        cache: &mut SurfaceCache,
        light: &mut SurfaceLight,
        radiance: &RadianceCache,
        sunlight: &Sunlight,
        frame: &ProbeFrame,
    ) {
        let size = [frame.depth.width(), frame.depth.height()];
        let grid = [size[0].div_ceil(PROBE_SPACING), size[1].div_ceil(PROBE_SPACING)];
        if self.buffers.as_ref().is_none_or(|(_, g)| *g != grid) {
            let make = || {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("CandeliteProbes"),
                    size: grid[0] as u64 * grid[1] as u64 * 80,
                    usage: wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                })
            };
            // Zeroed: no probe is on anything, so nothing is taken from them.
            self.buffers = Some(([make(), make()], grid));
            self.previous_view_projection = None;
        }
        self.frame = self.frame.wrapping_add(1);
        let Some(lookup) = lookup_group(device, &self.lookup_layout, scene, pool, cache, light) else {
            return;
        };
        let Some(cached) = radiance.sample_group(device, &self.cache_layout) else {
            return;
        };
        let ([history, current], _) = self.buffers.as_ref().expect("made above");
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            inverse_view_projection: [[f32; 4]; 4],
            previous_view_projection: [[f32; 4]; 4],
            toward_sun: [f32; 4],
            sun: [f32; 4],
            sky: [f32; 4],
            size: [u32; 2],
            grid: [u32; 2],
            frame: u32,
            _padding: [u32; 3],
        }
        let four = |v: [f32; 3]| [v[0], v[1], v[2], 0.0];
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteProbes"),
            contents: bytemuck::bytes_of(&Uniforms {
                inverse_view_projection: frame.inverse_view_projection,
                // With no last frame, a matrix that puts everything behind the eye: no history.
                previous_view_projection: self.previous_view_projection.unwrap_or([[0.0; 4]; 4]),
                toward_sun: four(sunlight.toward_sun),
                sun: four(sunlight.sun),
                sky: four(sunlight.sky),
                size,
                grid,
                frame: self.frame,
                _padding: [0; 3],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let depth = frame.depth.create_view(&wgpu::TextureViewDescriptor {
            aspect: wgpu::TextureAspect::DepthOnly,
            ..Default::default()
        });
        let normal = frame.normal.create_view(&Default::default());
        let common = |binding, resource| wgpu::BindGroupEntry { binding, resource };
        let trace_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CandeliteProbes"),
            layout: &self.trace_layout,
            entries: &[
                common(0, uniforms.as_entire_binding()),
                common(1, wgpu::BindingResource::TextureView(&depth)),
                common(2, wgpu::BindingResource::TextureView(&normal)),
                common(3, history.as_entire_binding()),
                common(4, current.as_entire_binding()),
            ],
        });
        let gather_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CandeliteProbeGather"),
            layout: &self.gather_layout,
            entries: &[
                common(0, uniforms.as_entire_binding()),
                common(1, wgpu::BindingResource::TextureView(&depth)),
                common(2, wgpu::BindingResource::TextureView(&normal)),
                common(3, current.as_entire_binding()),
            ],
        });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("CandeliteProbes"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.trace);
            pass.set_bind_group(0, &lookup, &[]);
            pass.set_bind_group(1, &trace_group, &[]);
            pass.set_bind_group(2, &cached, &[]);
            pass.dispatch_workgroups(grid[0], grid[1], 1);
        }
        let empty = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.empty_layout,
            entries: &[],
        });
        let view = frame.target.create_view(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("CandeliteProbeGather"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.gather);
            pass.set_bind_group(0, &empty, &[]);
            pass.set_bind_group(1, &gather_group, &[]);
            pass.draw(0..3, 0..1);
        }
        // This frame's probes are next frame's history.
        if let Some((buffers, _)) = self.buffers.as_mut() {
            buffers.swap(0, 1);
        }
        self.previous_view_projection = Some(frame.view_projection);
    }
}

/// What [`SurfaceView`] shows of what the rays hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewShows {
    /// The diffuse colour, as captured.
    Diffuse,
    /// The diffuse colour, lit from above and a little to one side by the normal: to see shapes.
    Shaded,
    /// The normal, as the G-Buffer stores it (each axis from -1 to 1 brought to 0 to 1).
    Normal,
    /// The light leaving what was hit: from its lit cards, or lit where it is (see
    /// [`SurfaceLight`]); brought into range by x / (1 + x).
    Lit,
}

const VIEW_SHADER: &str = r#"
struct View {
    inverseViewProjection: mat4x4f,
    eye: vec3f,
    shows: u32,
    size: vec2f,
    pad0: vec2f,
    towardSun: vec4f,
    sun: vec4f,
    sky: vec4f,
};
@group(1) @binding(0) var<uniform> view: View;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) at: vec4f) -> @location(0) vec4f {
    // Stored top row first: up the screen is down the rows.
    let ndc = vec2f(at.x / view.size.x * 2.0 - 1.0, 1.0 - at.y / view.size.y * 2.0);
    let far = view.inverseViewProjection * vec4f(ndc, 1.0, 1.0);
    let direction = normalize(far.xyz / far.w - view.eye);
    let hit = candelite_trace(view.eye, direction, 1.0e4);
    if (view.shows == 3u && hit.seen) {
        let light = candelite_hit_light(hit, view.towardSun.xyz, view.sun.xyz, view.sky.xyz);
        return vec4f(light / (1.0 + light), 1.0);
    }
    if (!hit.hit) {
        return vec4f(0.0);
    }
    if (!hit.seen) {
        // Hit, but no card saw anything there: magenta, to stand out.
        return vec4f(1.0, 0.0, 1.0, 1.0);
    }
    switch view.shows {
        case 0u: {
            return vec4f(hit.diffuse, 1.0);
        }
        case 2u: {
            return vec4f(hit.normal * 0.5 + 0.5, 1.0);
        }
        default: {
            let light = 0.35 + 0.65 * max(dot(hit.normal, normalize(vec3f(0.3, 1.0, 0.2))), 0.0);
            return vec4f(hit.diffuse * light, 1.0);
        }
    }
}
"#;

/// The format [`SurfaceView`] draws in.
pub const VIEW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

/// Shows the hit lookup at work: a ray from the eye through every pixel, and what it found - see
/// [`ViewShows`]. Where a ray hit nothing, the target is left clear; where it hit something no
/// card saw anything at, magenta.
pub struct SurfaceView {
    lookup_layout: wgpu::BindGroupLayout,
    view_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
}

impl SurfaceView {
    /// The view's pipeline, made once; [`None`] if `device` cannot trace rays.
    pub fn new(device: &wgpu::Device) -> Option<Self> {
        if !can_trace(device) {
            return None;
        }
        let lookup_layout = lookup_layout(device, wgpu::ShaderStages::FRAGMENT);
        let view_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("CandeliteView"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("CandeliteView"),
            source: wgpu::ShaderSource::Wgsl(format!("enable wgpu_ray_query;\n{LOOKUP_WGSL}\n{VIEW_SHADER}").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("CandeliteView"),
            bind_group_layouts: &[Some(&lookup_layout), Some(&view_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("CandeliteView"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &module,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &module,
                entry_point: Some("fs_main"),
                targets: &[Some(VIEW_FORMAT.into())],
                compilation_options: Default::default(),
            }),
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            multiview_mask: None,
            cache: None,
        });
        Some(Self {
            lookup_layout,
            view_layout,
            pipeline,
        })
    }

    /// Records the view from `eye`, through `inverse_view_projection` (column by column,
    /// OpenGL's clip space), into `target` - of [`VIEW_FORMAT`] - cleared first.
    #[allow(clippy::too_many_arguments)]
    pub fn record(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        scene: &mut TraceScene,
        pool: &GeometryPool,
        cache: &mut SurfaceCache,
        light: &mut SurfaceLight,
        sunlight: &Sunlight,
        inverse_view_projection: [[f32; 4]; 4],
        eye: [f32; 3],
        shows: ViewShows,
        target: &wgpu::Texture,
    ) {
        let view = target.create_view(&Default::default());
        let lookup = lookup_group(device, &self.lookup_layout, scene, pool, cache, light);
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            inverse_view_projection: [[f32; 4]; 4],
            eye: [f32; 3],
            shows: u32,
            size: [f32; 2],
            _padding: [f32; 2],
            toward_sun: [f32; 4],
            sun: [f32; 4],
            sky: [f32; 4],
        }
        let uniforms = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("CandeliteView"),
            contents: bytemuck::bytes_of(&Uniforms {
                inverse_view_projection,
                eye,
                shows: match shows {
                    ViewShows::Diffuse => 0,
                    ViewShows::Shaded => 1,
                    ViewShows::Normal => 2,
                    ViewShows::Lit => 3,
                },
                size: [target.width() as f32, target.height() as f32],
                _padding: [0.0; 2],
                toward_sun: [sunlight.toward_sun[0], sunlight.toward_sun[1], sunlight.toward_sun[2], 0.0],
                sun: [sunlight.sun[0], sunlight.sun[1], sunlight.sun[2], 0.0],
                sky: [sunlight.sky[0], sunlight.sky[1], sunlight.sky[2], 0.0],
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let view_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("CandeliteView"),
            layout: &self.view_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: uniforms.as_entire_binding(),
            }],
        });
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("CandeliteView"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        // Nothing to trace against: left clear.
        let Some(lookup) = lookup else {
            return;
        };
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, &lookup, &[]);
        pass.set_bind_group(1, &view_group, &[]);
        pass.draw(0..3, 0..1);
    }
}

/// The bind group layout of [`LOOKUP_WGSL`], for the shader stages `visibility`.
pub fn lookup_layout(device: &wgpu::Device, visibility: wgpu::ShaderStages) -> wgpu::BindGroupLayout {
    lookup_layout_with(device, visibility, false)
}

fn lookup_layout_with(device: &wgpu::Device, visibility: wgpu::ShaderStages, radiance_writable: bool) -> wgpu::BindGroupLayout {
    let storage_as = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let storage = |binding| storage_as(binding, true);
    let texture = |binding, sample_type| wgpu::BindGroupLayoutEntry {
        binding,
        visibility,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let unfiltered = wgpu::TextureSampleType::Float { filterable: false };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("CandeliteLookup"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility,
                ty: wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
                count: None,
            },
            storage(1),
            storage(2),
            storage(3),
            texture(4, unfiltered),
            texture(5, unfiltered),
            texture(6, wgpu::TextureSampleType::Depth),
            storage(7),
            storage(8),
            storage(9),
            storage(10),
            storage(11),
            storage_as(12, !radiance_writable),
        ],
    })
}

/// The bind group of [`LOOKUP_WGSL`], for `layout` (see [`lookup_layout`]), with the light on
/// the cards from `light`.
pub fn lookup_group(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene: &mut TraceScene,
    pool: &GeometryPool,
    cache: &mut SurfaceCache,
    light: &mut SurfaceLight,
) -> Option<wgpu::BindGroup> {
    light.buffers(device);
    let (lit_cards, instance_lit) = light.gpu.as_ref().expect("made above");
    lookup_group_with(device, layout, scene, pool, cache, [lit_cards, instance_lit, &light.radiance])
}

/// As [`lookup_group`], with the lit cards, each slot's first and the radiance atlas given.
fn lookup_group_with(
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    scene: &mut TraceScene,
    pool: &GeometryPool,
    cache: &mut SurfaceCache,
    [lit_cards, instance_lit, radiance]: [&wgpu::Buffer; 3],
) -> Option<wgpu::BindGroup> {
    scene.cutout_buffers(device);
    let tlas = scene.tlas.as_ref()?;
    let [cutouts, corners, masks] = scene.cutouts.as_ref().expect("made above");
    let diffuse = cache.layers[0].create_view(&Default::default());
    let normal = cache.layers[1].create_view(&Default::default());
    let depth = cache.depth.create_view(&wgpu::TextureViewDescriptor {
        aspect: wgpu::TextureAspect::DepthOnly,
        ..Default::default()
    });
    let (cards, mesh_cards) = cache.buffers(device);
    Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("CandeliteLookup"),
        layout,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::AccelerationStructure(tlas),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: pool.instances.buffer().as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: mesh_cards.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: cards.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 4,
                resource: wgpu::BindingResource::TextureView(&diffuse),
            },
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(&normal),
            },
            wgpu::BindGroupEntry {
                binding: 6,
                resource: wgpu::BindingResource::TextureView(&depth),
            },
            wgpu::BindGroupEntry {
                binding: 7,
                resource: cutouts.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 8,
                resource: corners.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 9,
                resource: masks.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 10,
                resource: instance_lit.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 11,
                resource: lit_cards.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 12,
                resource: radiance.as_entire_binding(),
            },
        ],
    }))
}

#[cfg(test)]
mod test {
    use super::*;

    fn transform(m: &[[f32; 4]; 4], p: [f32; 3]) -> [f32; 3] {
        let c = |r: usize| m[0][r] * p[0] + m[1][r] * p[1] + m[2][r] * p[2] + m[3][r];
        [c(0), c(1), c(2)]
    }

    #[test]
    fn each_card_sees_its_face_from_outside_across_the_whole_bounds() {
        let bounds = Bounds {
            min: [-1.0, 0.0, -3.0],
            max: [2.0, 5.0, 1.0],
        };
        let cards = Card::of(&bounds);
        for card in &cards {
            let n = cross(card.u, card.v);
            assert_eq!(n, card.normal, "u cross v is the normal: {card:?}");
            let m = card.view_projection();
            // Every corner of the bounds is in view, from the face (z -1) to the far side (z 1).
            let mut z_range = (f32::MAX, f32::MIN);
            for k in 0..8 {
                let corner = [0, 1, 2].map(|i| if k & (1 << i) != 0 { bounds.max[i] } else { bounds.min[i] });
                let p = transform(&m, corner);
                assert!(p.iter().all(|c| (-1.0001..=1.0001).contains(c)), "{card:?}: {corner:?} at {p:?}");
                z_range = (z_range.0.min(p[2]), z_range.1.max(p[2]));
            }
            assert!((z_range.0 + 1.0).abs() < 1.0e-4 && (z_range.1 - 1.0).abs() < 1.0e-4, "{z_range:?}");
            // The face's middle at the near plane's middle.
            let p = transform(&m, card.center);
            assert!(p.iter().zip([0.0, 0.0, -1.0]).all(|(a, b)| (a - b).abs() < 1.0e-5), "{p:?}");
            // A point just inside the face, towards the bounds, is a little farther.
            let inside = [0, 1, 2].map(|i| card.center[i] - card.normal[i] * 0.1);
            assert!(transform(&m, inside)[2] > -1.0);
        }
        // The top card's tile: 3 by 4 meters at 8 texels a meter.
        assert_eq!(cards[2].tile_size(), [24, 32]);
    }

    #[test]
    fn shelves_hand_out_rectangles_that_never_overlap_and_take_them_back() {
        let mut shelves = ShelfAllocator::new(64);
        let mut given = Vec::new();
        for (w, h) in [(10, 7), (20, 8), (5, 3), (64, 16), (30, 5), (12, 8)] {
            let at = shelves.allocate(w, h).expect("room");
            given.push((at, w, h));
        }
        for (i, a) in given.iter().enumerate() {
            assert!(a.0[0] + a.1 <= 64 && a.0[1] + a.2 <= 64);
            for b in &given[i + 1..] {
                let apart = a.0[0] + a.1 <= b.0[0] || b.0[0] + b.1 <= a.0[0] || a.0[1] + a.2 <= b.0[1] || b.0[1] + b.2 <= a.0[1];
                assert!(apart, "{a:?} and {b:?} overlap");
            }
        }
        // Full, until something is given back; then its room is used again.
        while shelves.allocate(64, 16).is_some() {}
        assert_eq!(shelves.allocate(64, 16), None);
        let (at, w, h) = given[3];
        shelves.free(at, w, h);
        assert_eq!(shelves.allocate(64, 16), Some(at));
        // Neighbours given back are joined.
        let (a, b) = (given[0], given[5]);
        shelves.free(a.0, a.1, a.2);
        shelves.free(b.0, b.1, b.2);
        assert!(shelves.allocate(10, 8).is_some());
    }

    use crate::yoctite::{MaterialParams, PoolVertex};

    fn block_on<F: std::future::Future>(future: F) -> F::Output {
        let mut future = std::pin::pin!(future);
        let mut context = std::task::Context::from_waker(std::task::Waker::noop());
        loop {
            if let std::task::Poll::Ready(out) = future.as_mut().poll(&mut context) {
                return out;
            }
        }
    }

    fn device() -> Option<(wgpu::Device, wgpu::Queue)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok()?;
        block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_limits: adapter.limits(),
            ..Default::default()
        }))
        .ok()
    }

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

    /// Four bytes of each texel of `texture`'s `aspect`.
    fn texels(device: &wgpu::Device, queue: &wgpu::Queue, texture: &wgpu::Texture, aspect: wgpu::TextureAspect, bytes: u32) -> Vec<u32> {
        let (w, h) = (texture.width(), texture.height());
        let row = (w * bytes).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: (row * h) as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect },
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row), rows_per_image: None },
            },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        queue.submit([encoder.finish()]);
        buffer.slice(..).map_async(wgpu::MapMode::Read, |r| r.expect("mapped"));
        device.poll(wgpu::PollType::wait_indefinitely()).expect("polled");
        let data = buffer.slice(..).get_mapped_range().expect("mapped range");
        let mut out = Vec::with_capacity((w * h) as usize);
        for y in 0..h as usize {
            for x in 0..w as usize {
                let at = y * row as usize + x * bytes as usize;
                // The first word of a texel wider than one.
                let n = bytes.min(4) as usize;
                let mut word = [0u8; 4];
                word[..n].copy_from_slice(&data[at..at + n]);
                out.push(u32::from_le_bytes(word));
            }
        }
        out
    }

    /// A box from `min` to `max`, each face two triangles facing out (counter-clockwise seen
    /// from outside), or only its top face, facing up.
    fn boxed(min: [f32; 3], max: [f32; 3], top_only: bool) -> (Vec<PoolVertex>, Vec<u32>) {
        let middle: [f32; 3] = std::array::from_fn(|i| (min[i] + max[i]) * 0.5);
        let half: [f32; 3] = std::array::from_fn(|i| (max[i] - min[i]) * 0.5);
        let (mut vertices, mut indices) = (Vec::new(), Vec::new());
        for card in Card::of(&Bounds { min, max }) {
            if top_only && card.normal != [0.0, 1.0, 0.0] {
                continue;
            }
            let reach = |d: [f32; 3]| -> [f32; 3] { std::array::from_fn(|i| d[i] * half[i]) };
            let (a, b, n) = (reach(card.u), reach(card.v), card.normal);
            let c: [f32; 3] = std::array::from_fn(|i| middle[i] + n[i] * half[i]);
            let first = vertices.len() as u32;
            for (s, t) in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)] {
                vertices.push(PoolVertex {
                    position: std::array::from_fn(|i| c[i] + s * a[i] + t * b[i]),
                    normal: n,
                    uv: [(s + 1.0) * 0.5, (t + 1.0) * 0.5],
                    tangent: [card.u[0], card.u[1], card.u[2], 1.0],
                });
            }
            indices.extend([first, first + 1, first + 2, first, first + 2, first + 3]);
        }
        (vertices, indices)
    }

    #[test]
    fn each_card_captures_what_it_sees_of_its_mesh() {
        let Some((device, queue)) = device() else {
            eprintln!("no GPU: skipped");
            return;
        };
        let mut pool = GeometryPool::new(&device, 1 << 10);
        let mut cache = SurfaceCache::new(&device, 128);
        let mut captures = Vec::new();
        // A closed box, and a sheet facing up only.
        let mut meshes = Vec::new();
        for (min, max, top_only) in [([-1.0, 0.0, -2.0], [1.0, 1.5, 2.0], false), ([0.0; 3], [2.0, 0.0, 2.0], true)] {
            let (vertices, indices) = boxed(min, max, top_only);
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            let dag = yoctite::build(&yoctite::Mesh::new(&positions, &indices), Default::default());
            let mesh = pool.add_mesh(&device, &queue, &dag, &vertices);
            let first = pool.mesh(mesh).unwrap().first_cluster;
            let full: Vec<u32> = (0..dag.clusters.len() as u32).filter(|&i| dag.clusters[i as usize].level == 0).map(|i| first + i).collect();
            captures.extend(cache.add_mesh(mesh, &Bounds::of(positions.iter().copied()), &full, 0).expect("room"));
            meshes.push(mesh);
        }
        let white = swatch(&device, &queue, [255; 4]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let materials = [ResolvedMaterial {
            slot: 0,
            params: MaterialParams {
                diffuse_color: [1.0, 0.5, 0.25, 1.0],
                ..Default::default()
            },
            views: [white.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [white.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        }];
        let mut drawer = GBufferDrawer::new(&device);
        let mut encoder = device.create_command_encoder(&Default::default());
        cache.record(&device, &mut encoder, &mut drawer, &pool, &captures, &materials).expect("captured");
        queue.submit([encoder.finish()]);

        let (layers, depth) = cache.atlas();
        let diffuse = texels(&device, &queue, &layers[0], wgpu::TextureAspect::All, 4);
        let normal = texels(&device, &queue, &layers[1], wgpu::TextureAspect::All, 4);
        let depth = texels(&device, &queue, depth, wgpu::TextureAspect::DepthOnly, 4);
        let size = cache.size() as usize;
        let ten = |v: u32| [v & 1023, (v >> 10) & 1023, (v >> 20) & 1023].map(|c| c as f32 / 1023.0 * 2.0 - 1.0);
        for (m, mesh) in meshes.iter().enumerate() {
            let first = cache.mesh_cards()[mesh.0 as usize] as usize;
            for c in 0..6 {
                let card = cache.cards()[first + c];
                let [x, y, w, h] = card.tile;
                let at = (y + h / 2) as usize * size + (x + w / 2) as usize;
                let n = cross([card.u[0], card.u[1], card.u[2]], [card.v[0], card.v[1], card.v[2]]);
                // The sheet is seen only from above.
                let sees = m == 0 || n == [0.0, 1.0, 0.0];
                if sees {
                    assert_eq!(diffuse[at].to_le_bytes(), [255, 128, 64, 255], "mesh {m} card {c}: its colour");
                    let got = ten(normal[at]);
                    assert!((0..3).all(|i| (got[i] - n[i]).abs() < 0.01), "mesh {m} card {c}: normal {got:?}, not {n:?}");
                    // How far in, in meters: on the face (flat bounds are given a millimetre's
                    // depth, the sheet in its middle).
                    let inside = f32::from_bits(depth[at]) * card.center[3];
                    assert!(inside < 1.0e-3, "mesh {m} card {c}: on its face, not {inside} m in");
                } else {
                    assert_eq!(diffuse[at], 0, "mesh {m} card {c}: nothing seen from below a sheet");
                    assert_eq!(f32::from_bits(depth[at]), 1.0, "mesh {m} card {c}: nothing, the far side");
                }
                // Every texel of the tile is drawn, the box filling its cards to their edges.
                if m == 0 {
                    let drawn = (y..y + h).flat_map(|ty| (x..x + w).map(move |tx| ty as usize * size + tx as usize)).filter(|&i| diffuse[i] != 0).count();
                    assert_eq!(drawn as u32, w * h, "mesh {m} card {c}: the whole tile");
                }
            }
        }
        // Taken out and in again, the room is used again.
        cache.remove_mesh(meshes[0]);
        assert!(cache.add_mesh(meshes[0], &Bounds { min: [-1.0, 0.0, -2.0], max: [1.0, 1.5, 2.0] }, &[], 0).is_some());
    }

    #[test]
    fn a_ray_finds_what_it_hit_on_the_cards() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok() else {
            eprintln!("no GPU: skipped");
            return;
        };
        if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            eprintln!("no ray tracing: skipped");
            return;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: adapter.limits(),
            // As the renderer asks for ray queries: an experimental feature of wgpu's.
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .expect("device");

        let mut pool = GeometryPool::new(&device, 1 << 10);
        let mut cache = SurfaceCache::new(&device, 256);
        let mut scene = TraceScene::new();
        let mut captures = Vec::new();
        // A box turned a quarter about up (its -x face towards +z) 10 m down -z; and a sheet
        // facing up, 3 m overhead, seen from below.
        let quarter = |t: [f32; 3]| [0.0, 0.0, 1.0, t[0], 0.0, 1.0, 0.0, t[1], -1.0, 0.0, 0.0, t[2]];
        let shifted = |t: [f32; 3]| [1.0, 0.0, 0.0, t[0], 0.0, 1.0, 0.0, t[1], 0.0, 0.0, 1.0, t[2]];
        for (min, max, top_only, transform) in [
            ([-1.0, 0.0, -2.0], [1.0, 1.5, 2.0], false, quarter([0.0, 0.0, -10.0])),
            ([-1.5, 0.0, -1.0], [1.5, 0.0, 1.0], true, shifted([0.0, 3.0, -3.0])),
        ] {
            let (vertices, indices) = boxed(min, max, top_only);
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            let dag = yoctite::build(&yoctite::Mesh::new(&positions, &indices), Default::default());
            let mesh = pool.add_mesh(&device, &queue, &dag, &vertices);
            let first = pool.mesh(mesh).unwrap().first_cluster;
            let full: Vec<u32> = (0..dag.clusters.len() as u32).filter(|&i| dag.clusters[i as usize].level == 0).map(|i| first + i).collect();
            captures.extend(cache.add_mesh(mesh, &Bounds::of(positions.iter().copied()), &full, 0).expect("room"));
            scene.add_mesh(&device, &queue, mesh, &positions, &indices, None);
            pool.add_instance(&device, &queue, mesh, transform, 0);
        }
        // A blue card 5 m out, in front of the box: drawn on its left half, see-through on its
        // right (its texture's alpha), traced as a cutout.
        let card: Vec<PoolVertex> = [(-0.5, 0.25), (0.5, 0.25), (0.5, 1.25), (-0.5, 1.25)]
            .map(|(x, y)| PoolVertex {
                position: [x, y, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [x + 0.5, 0.5],
                tangent: [1.0, 0.0, 0.0, 1.0],
            })
            .to_vec();
        let card_indices = [0, 1, 2, 0, 2, 3];
        let positions: Vec<[f32; 3]> = card.iter().map(|v| v.position).collect();
        let uvs: Vec<[f32; 2]> = card.iter().map(|v| v.uv).collect();
        let dag = yoctite::build(&yoctite::Mesh::new(&positions, &card_indices), Default::default());
        let mesh = pool.add_mesh(&device, &queue, &dag, &card);
        let first = pool.mesh(mesh).unwrap().first_cluster;
        captures.extend(cache.add_mesh(mesh, &Bounds::of(positions.iter().copied()), &[first], 1).expect("room"));
        let mask = Arc::new(CutoutMask::new(2, 1, |x, _| x == 0));
        scene.add_mesh(&device, &queue, mesh, &positions, &card_indices, Some((&uvs, mask)));
        pool.add_instance(&device, &queue, mesh, shifted([0.0, 0.0, -5.0]), 1);

        let white = swatch(&device, &queue, [255; 4]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let half = {
            let texture = device.create_texture_with_data(
                &queue,
                &wgpu::TextureDescriptor {
                    label: None,
                    size: wgpu::Extent3d { width: 2, height: 1, depth_or_array_layers: 1 },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format: wgpu::TextureFormat::Rgba8Unorm,
                    usage: wgpu::TextureUsages::TEXTURE_BINDING,
                    view_formats: &[],
                },
                Default::default(),
                &[255, 255, 255, 255, 255, 255, 255, 0],
            );
            (texture.create_view(&Default::default()), device.create_sampler(&Default::default()))
        };
        let material = |slot, colour: [f32; 4], diffuse: &(wgpu::TextureView, wgpu::Sampler)| ResolvedMaterial {
            slot,
            params: MaterialParams {
                diffuse_color: colour,
                ..Default::default()
            },
            views: [diffuse.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [diffuse.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        };
        let materials = [material(0, [1.0, 0.5, 0.25, 1.0], &white), material(1, [0.25, 0.5, 1.0, 1.0], &half)];
        let mut drawer = GBufferDrawer::new(&device);
        let view = SurfaceView::new(&device).expect("can trace");
        let mut light = SurfaceLight::new(&device).expect("can trace");
        let sunlight = Sunlight { toward_sun: [0.0, 1.0, 0.0], sun: [1.0; 3], sky: [0.0; 3] };

        // From the origin at the box's middle height, looking down -z, 90 degrees across.
        const SIZE: u32 = 64;
        let eye = [0.0, 0.75, 0.0];
        let reach = 1.0;
        let inverse = [[reach, 0.0, 0.0, 0.0], [0.0, reach, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [eye[0], eye[1], eye[2] - reach, 1.0]];
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VIEW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let mut look = |shows| {
            let mut encoder = device.create_command_encoder(&Default::default());
            cache.record(&device, &mut encoder, &mut drawer, &pool, &captures, &materials).expect("captured");
            assert!(scene.update(&device, &mut encoder, &pool));
            view.record(&device, &mut encoder, &mut scene, &pool, &mut cache, &mut light, &sunlight, inverse, eye, shows, &target);
            queue.submit([encoder.finish()]);
            texels(&device, &queue, &target, wgpu::TextureAspect::All, 4)
        };
        let diffuse = look(ViewShows::Diffuse);
        let normal = look(ViewShows::Normal);
        let at = |x: u32, y: u32| (y * SIZE + x) as usize;
        let middle = at(SIZE / 2, SIZE / 2);
        assert_eq!(diffuse[middle].to_le_bytes(), [255, 128, 64, 255], "the box's face, as captured");
        // Facing the eye, +z in the world: its -x face in its own space, turned.
        let n = normal[middle].to_le_bytes();
        assert!(n[0].abs_diff(128) <= 2 && n[1].abs_diff(128) <= 2 && n[2] >= 253, "normal {n:?}");
        // Off to the side, past the box: nothing.
        assert_eq!(diffuse[at(2, SIZE / 2)], 0, "missed");
        // Overhead, the sheet's underside: hit, but no card saw it.
        let magenta = (0..SIZE / 3).flat_map(|y| (0..SIZE).map(move |x| (x, y))).filter(|&(x, y)| diffuse[at(x, y)].to_le_bytes() == [255, 0, 255, 255]).count();
        assert!(magenta > 20, "the sheet's underside, uncovered: {magenta} pixels");
        let box_magenta = (SIZE / 2 - 3..SIZE / 2 + 3).filter(|&x| diffuse[at(x, SIZE / 2)].to_le_bytes() == [255, 0, 255, 255]).count();
        assert_eq!(box_magenta, 0, "every hit on the box found on its cards");
        // Through the card's drawn half, the card; through its see-through half, the box behind.
        assert_eq!(diffuse[at(30, SIZE / 2)].to_le_bytes(), [64, 128, 255, 255], "the card's drawn half");
        assert_eq!(diffuse[at(34, SIZE / 2)].to_le_bytes(), [255, 128, 64, 255], "the box, through the card");
    }

    #[test]
    fn the_sun_lights_the_cards_where_a_shadow_ray_reaches_it() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok() else {
            eprintln!("no GPU: skipped");
            return;
        };
        if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            eprintln!("no ray tracing: skipped");
            return;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: adapter.limits(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .expect("device");

        let mut pool = GeometryPool::new(&device, 1 << 10);
        let mut cache = SurfaceCache::new(&device, 256);
        let mut scene = TraceScene::new();
        let mut captures = Vec::new();
        let shifted = |t: [f32; 3]| [1.0, 0.0, 0.0, t[0], 0.0, 1.0, 0.0, t[1], 0.0, 0.0, 1.0, t[2]];
        // A floor 4 m square facing up, its colour orange; and over its left half, 1 m up, a slab
        // shading it from a sun straight overhead.
        for (min, max, top_only, at, slot) in [
            ([-2.0, 0.0, -2.0], [2.0, 0.0, 2.0], true, [0.0; 3], 0),
            ([-2.5, 1.0, -2.5], [0.0, 1.2, 2.5], false, [0.0; 3], 1),
        ] {
            let (vertices, indices) = boxed(min, max, top_only);
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            let dag = yoctite::build(&yoctite::Mesh::new(&positions, &indices), Default::default());
            let mesh = pool.add_mesh(&device, &queue, &dag, &vertices);
            let first = pool.mesh(mesh).unwrap().first_cluster;
            let full: Vec<u32> = (0..dag.clusters.len() as u32).filter(|&i| dag.clusters[i as usize].level == 0).map(|i| first + i).collect();
            captures.extend(cache.add_mesh(mesh, &Bounds::of(positions.iter().copied()), &full, slot).expect("room"));
            scene.add_mesh(&device, &queue, mesh, &positions, &indices, None);
            pool.add_instance(&device, &queue, mesh, shifted(at), slot);
        }
        let white = swatch(&device, &queue, [255; 4]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let material = |slot, colour: [f32; 4]| ResolvedMaterial {
            slot,
            params: MaterialParams {
                diffuse_color: colour,
                ..Default::default()
            },
            views: [white.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [white.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        };
        let materials = [material(0, [1.0, 0.5, 0.25, 1.0]), material(1, [0.2, 0.2, 0.2, 1.0])];
        let mut drawer = GBufferDrawer::new(&device);
        let mut light = SurfaceLight::new(&device).expect("can trace");
        let sunlight = Sunlight { toward_sun: [0.0, 1.0, 0.0], sun: [1.0; 3], sky: [0.0; 3] };
        // From above the floor, close enough for its cards to be lit at full detail.
        let eye = [0.0, 3.0, 0.0];
        let mut encoder = device.create_command_encoder(&Default::default());
        cache.record(&device, &mut encoder, &mut drawer, &pool, &captures, &materials).expect("captured");
        assert!(scene.update(&device, &mut encoder, &pool));
        queue.submit([encoder.finish()]);
        for _ in 0..8 {
            let mut encoder = device.create_command_encoder(&Default::default());
            light.update(&device, &mut encoder, &mut scene, &pool, &mut cache, eye, &sunlight);
            queue.submit([encoder.finish()]);
        }
        let floor = pool.placed().find(|(_, record)| record.material == 0).map(|(slot, _)| slot).expect("floor");
        // Its top card (+y), seen from above: x across it, -z up it.
        let tile = light.lit_tiles(floor).expect("lit")[2];
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: light.radiance().size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(light.radiance(), 0, &staging, 0, light.radiance().size());
        queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        let _ = device.poll(wgpu::PollType::wait_indefinitely());
        let words: Vec<u32> = bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec();
        let at = |x: u32, y: u32| {
            let i = ((tile[1] + y) * RADIANCE_SIZE + tile[0] + x) as usize * 2;
            let (a, b) = (words[i], words[i + 1]);
            let half = |h: u32| half::f16::from_bits(h as u16).to_f32();
            [half(a & 0xFFFF), half(a >> 16), half(b & 0xFFFF)]
        };
        let (w, h) = (tile[2], tile[3]);
        let row = h / 2;
        let lit = at(w * 7 / 8, row);
        let shaded = at(w / 8, row);
        // In the sun: its colour, linear (sRGB 0.5 is 0.214, 0.25 is 0.051), the sun a white
        // surface gives off as 1.
        assert!((lit[0] - 1.0).abs() < 0.05 && (lit[1] - 0.214).abs() < 0.03 && (lit[2] - 0.051).abs() < 0.02, "lit {lit:?}");
        // In the slab's shade, with no sky: only light bounced off the lit floor and the slab's
        // underside, far less.
        assert!(shaded[0] < 0.2 * lit[0], "shaded {shaded:?}, lit {lit:?}");
        assert!(shaded[0] > 0.0, "some light bounces into the shade: {shaded:?}");

        // And through the view, as a ray from above finds it: the light on the floor's lit cards,
        // brought into range.
        const SIZE: u32 = 32;
        let view = SurfaceView::new(&device).expect("can trace");
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: VIEW_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        // Looking straight down from 3 m over the floor's right half, 90 degrees across.
        let eye = [1.0, 3.0, 0.0];
        let inverse = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [eye[0], eye[1] - 1.0, eye[2], 1.0]];
        let mut encoder = device.create_command_encoder(&Default::default());
        view.record(&device, &mut encoder, &mut scene, &pool, &mut cache, &mut light, &sunlight, inverse, eye, ViewShows::Lit, &target);
        queue.submit([encoder.finish()]);
        let seen = texels(&device, &queue, &target, wgpu::TextureAspect::All, 4);
        let middle = seen[(SIZE / 2 * SIZE + SIZE / 2) as usize].to_le_bytes();
        let expected = lit.map(|c| (c / (1.0 + c) * 255.0).round() as u8);
        assert!((0..3).all(|i| middle[i].abs_diff(expected[i]) <= 3), "seen {middle:?}, lit {expected:?}");
    }

    #[test]
    fn screen_probes_light_the_open_floor_with_the_sky_and_the_shaded_floor_less() {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let Some(adapter) = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions::default())).ok() else {
            eprintln!("no GPU: skipped");
            return;
        };
        if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            eprintln!("no ray tracing: skipped");
            return;
        }
        let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: wgpu::Features::EXPERIMENTAL_RAY_QUERY,
            required_limits: adapter.limits(),
            experimental_features: unsafe { wgpu::ExperimentalFeatures::enabled() },
            ..Default::default()
        }))
        .expect("device");

        let mut pool = GeometryPool::new(&device, 1 << 10);
        let mut cache = SurfaceCache::new(&device, 256);
        let mut scene = TraceScene::new();
        let mut captures = Vec::new();
        let shifted = |t: [f32; 3]| [1.0, 0.0, 0.0, t[0], 0.0, 1.0, 0.0, t[1], 0.0, 0.0, 1.0, t[2]];
        // The floor, 4 m square; over its left half, 1 m up, a slab.
        for (min, max, top_only, slot) in [
            ([-2.0, 0.0, -2.0], [2.0, 0.0, 2.0], true, 0),
            ([-2.5, 1.0, -2.5], [0.0, 1.2, 2.5], false, 1),
        ] {
            let (vertices, indices) = boxed(min, max, top_only);
            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
            let dag = yoctite::build(&yoctite::Mesh::new(&positions, &indices), Default::default());
            let mesh = pool.add_mesh(&device, &queue, &dag, &vertices);
            let first = pool.mesh(mesh).unwrap().first_cluster;
            let full: Vec<u32> = (0..dag.clusters.len() as u32).filter(|&i| dag.clusters[i as usize].level == 0).map(|i| first + i).collect();
            captures.extend(cache.add_mesh(mesh, &Bounds::of(positions.iter().copied()), &full, slot).expect("room"));
            scene.add_mesh(&device, &queue, mesh, &positions, &indices, None);
            pool.add_instance(&device, &queue, mesh, shifted([0.0; 3]), slot);
        }
        let white = swatch(&device, &queue, [255; 4]);
        let flat = swatch(&device, &queue, [128, 128, 255, 255]);
        let material = |slot, colour: [f32; 4]| ResolvedMaterial {
            slot,
            params: MaterialParams {
                diffuse_color: colour,
                ..Default::default()
            },
            views: [white.0.clone(), flat.0.clone(), white.0.clone(), white.0.clone(), white.0.clone(), white.0.clone()],
            samplers: [white.1.clone(), flat.1.clone(), white.1.clone(), white.1.clone(), white.1.clone(), white.1.clone()],
        };
        let materials = [material(0, [0.5, 0.5, 0.5, 1.0]), material(1, [0.2, 0.2, 0.2, 1.0])];
        let mut drawer = GBufferDrawer::new(&device);
        let mut light = SurfaceLight::new(&device).expect("can trace");
        let mut probes = ScreenProbes::new(&device).expect("can trace");
        let mut radiance = RadianceCache::new(&device).expect("can trace");
        // Only the sky, of radiance 1: a floor open to all of it gathers 1.
        let sunlight = Sunlight { toward_sun: [0.0, 1.0, 0.0], sun: [0.0; 3], sky: [1.0; 3] };
        let mut encoder = device.create_command_encoder(&Default::default());
        cache.record(&device, &mut encoder, &mut drawer, &pool, &captures, &materials).expect("captured");
        assert!(scene.update(&device, &mut encoder, &pool));
        queue.submit([encoder.finish()]);

        // The screen: the floor seen from straight above, 4 m across, at depth 0.5 - as if a
        // G-Buffer had drawn it. Clip x is world x / 2, clip y is -z / 2, clip z is -y.
        const SIZE: u32 = 64;
        let texture = |format, usage| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: None,
                size: wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage,
                view_formats: &[],
            })
        };
        let depth = texture(wgpu::TextureFormat::Depth32Float, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::RENDER_ATTACHMENT);
        let normal = texture(wgpu::TextureFormat::Rgba8Unorm, wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST);
        let target = texture(PROBES_FORMAT, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC);
        let fill = |texture: &wgpu::Texture, texel: &[u8]| {
            let data: Vec<u8> = texel.iter().copied().cycle().take(texel.len() * (SIZE * SIZE) as usize).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &data,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(SIZE * texel.len() as u32), rows_per_image: None },
                wgpu::Extent3d { width: SIZE, height: SIZE, depth_or_array_layers: 1 },
            );
        };
        {
            let view = depth.create_view(&Default::default());
            let mut encoder = device.create_command_encoder(&Default::default());
            encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &view,
                    depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(0.5), store: wgpu::StoreOp::Store }),
                    stencil_ops: None,
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            queue.submit([encoder.finish()]);
        }
        fill(&normal, &[128, 255, 128, 255]);
        let inverse = [[2.0, 0.0, 0.0, 0.0], [0.0, 0.0, -2.0, 0.0], [0.0, -1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let forward = [[0.5, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, -0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]];
        let frame = ProbeFrame { depth: &depth, normal: &normal, view_projection: forward, inverse_view_projection: inverse, target: &target };
        // Every probe of the cache traced once (64 grids' worth of slots at its budget a frame),
        // then the screen probes, a few frames over.
        let eye = [0.0, 3.0, 0.0];
        for _ in 0..(CACHE_LEVELS * CACHE_PROBES) as usize / CACHE_BUDGET {
            let mut encoder = device.create_command_encoder(&Default::default());
            radiance.update(&device, &mut encoder, &mut scene, &pool, &mut cache, &mut light, eye, &sunlight);
            queue.submit([encoder.finish()]);
        }
        for _ in 0..8 {
            let mut encoder = device.create_command_encoder(&Default::default());
            radiance.update(&device, &mut encoder, &mut scene, &pool, &mut cache, &mut light, eye, &sunlight);
            probes.record(&device, &mut encoder, &mut scene, &pool, &mut cache, &mut light, &radiance, &sunlight, &frame);
            queue.submit([encoder.finish()]);
        }
        // Red of each pixel, a half float in the low half of its first word.
        let seen = texels(&device, &queue, &target, wgpu::TextureAspect::All, 8);
        let red = |x: u32, y: u32| half::f16::from_bits((seen[(y * SIZE + x) as usize] & 0xFFFF) as u16).to_f32();
        // x = 1.75 m, out in the open but for the slab's edge 1.75 m off: most of the sky.
        let open = red(60, SIZE / 2);
        // x = -1.5 m, under the slab: only the sky round its edges.
        let shaded = red(8, SIZE / 2);
        assert!(open > 0.7 && open < 1.05, "open {open}");
        assert!(shaded < 0.5 * open, "shaded {shaded}, open {open}");

        // The sky put out, the cache left as it was: a ray leaving the screen probes' reach takes
        // its light from the cache, so the open floor still gathers the sky the cache saw.
        let dark = Sunlight { sky: [0.0; 3], ..sunlight };
        for _ in 0..16 {
            let mut encoder = device.create_command_encoder(&Default::default());
            probes.record(&device, &mut encoder, &mut scene, &pool, &mut cache, &mut light, &radiance, &dark, &frame);
            queue.submit([encoder.finish()]);
        }
        let seen = texels(&device, &queue, &target, wgpu::TextureAspect::All, 8);
        let red = |x: u32, y: u32| half::f16::from_bits((seen[(y * SIZE + x) as usize] & 0xFFFF) as u16).to_f32();
        let cached = red(60, SIZE / 2);
        assert!(cached > 0.6 * open, "from the cache {cached}, open {open}");
    }
}
