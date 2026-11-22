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

use crate::yoctite::{Capture, GBufferDrawer, GeometryPool, MeshId, ResolvedMaterial};
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
                let mut word = [0u8; 4];
                word[..bytes as usize].copy_from_slice(&data[at..at + bytes as usize]);
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
}
