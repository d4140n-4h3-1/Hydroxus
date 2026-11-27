//! Grass: clumps of grass cards all over the ground round the eye, drawn straight into the
//! G-Buffer in one instanced draw, with no node or mesh of their own (see `grass.wgsl`).
//!
//! Grass as scene nodes costs too much to cover the ground with: each clump a node for the CPU, an
//! instance for Yoctite, something Candelite and the shadows trace, and, swaying, bent anew every
//! frame. Here each clump is made in the vertex shader instead - three upright cards crossed at 60
//! degrees through its middle, six triangles, as plant_life's woodland grass cards are - one to
//! each cell of a square of the ground round the eye, which card, where in the cell, which way it
//! faces, its size and shade all from a hash of the cell, so the same clumps are there every frame;
//! and bent by the wind there too, the cards' tops along it. Being in the G-Buffer, they are lit,
//! shadowed (the trees' shadows fall on them), occluded and fogged as everything else is, and
//! Candelite's light from the screen reaches them; being nowhere else, they cast no shadow and
//! nothing traces them.
//!
//! The game gives the cards - each a texture, its size and how common it is ([`GrassCard`],
//! [`GrassField::set_cards`]) - and says where grass grows, how high the ground is there and how
//! damp, with a [`GrassMap`] ([`GrassField::set_map`]); and which way and how hard the wind blows
//! ([`GrassField::set_wind`]). Until it has given both cards and a map, no grass is drawn.
//! [`crate::GraphicsEffects::with_grass`] adds it.
//!
//! A card's texture is cut out where its alpha is under [`GrassSettings::cutoff`]. Its smaller
//! mip levels are made here, each with its alpha scaled so as much of it is over the cutoff as of
//! the full-size texture: averaged plainly, a card's thin blades fade under the cutoff further off,
//! and the grass thins to nothing.

use fyrox::{
    core::log::Log,
    graphics::{error::FrameworkError, stats::RenderPassStatistics},
    renderer::{GBufferPassContext, SceneRenderPass},
    resource::texture::{TextureKind, TexturePixelKind, TextureResource},
};
use fyrox_graphics_wgpu::{framebuffer::WgpuFrameBuffer, server::WgpuGraphicsServer, texture::WgpuTexture};
use std::{any::TypeId, cell::RefCell, rc::Rc};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("grass.wgsl");
/// Vertices in a clump, as `grass.wgsl` makes it: three cards, two triangles each.
const CLUMP_VERTICES: u32 = 18;
const CLUMP_TRIANGLES: u32 = 6;
/// The most cards there may be.
pub const MOST_CARDS: usize = 4;

/// What the grass looks like, and how far round the eye it grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassSettings {
    /// How far from the eye grass is drawn, in meters.
    pub radius: f32,
    /// From what share of the radius on it thins out, to none at the radius.
    pub thin_from: f32,
    /// The side of a cell of the ground, in meters: a clump to a cell, where the grass is
    /// thickest.
    pub cell: f32,
    /// The least and the most a clump is scaled, of its card's size.
    pub scale: (f32, f32),
    /// How much bigger the clumps grow towards the radius, where fewer are drawn: 0.5 half as big
    /// again at it.
    pub grow: f32,
    /// The alpha a card's texture is cut out below.
    pub cutoff: f32,
    /// How rough it is, and how much light comes through a card, 0 to 1.
    pub roughness: f32,
    pub translucency: f32,
    /// How much each clump's shade varies, either way: 0.2 from 0.9 to 1.1 of the texture's.
    pub shade: f32,
}

impl Default for GrassSettings {
    fn default() -> Self {
        Self {
            radius: 30.0,
            thin_from: 0.5,
            cell: 0.3,
            scale: (0.8, 1.25),
            grow: 0.6,
            cutoff: 0.4,
            roughness: 0.85,
            translucency: 0.3,
            shade: 0.25,
        }
    }
}

/// A card: its texture - RGBA, alpha cut out below [`GrassSettings::cutoff`], the soil along its
/// bottom - its size in meters, and how common it is against the others, on dry ground and how
/// much more on damp ground (see [`GrassMap::damp`]). Every card's texture must be the same size.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassCard {
    pub texture: TextureResource,
    pub width: f32,
    pub height: f32,
    pub weight: f32,
    pub damp_weight: f32,
}

/// Where grass grows: a grid of samples `spacing` meters apart, `width` along x and `depth` along
/// z, the first at `origin` (world x and z), row by row along x. Each the ground's height there, in
/// meters; how thick the grass is, 0 for none to 255 for a clump in every cell; and how damp the
/// ground is, 0 to 255, for the cards that grow more there.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassMap {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub width: u32,
    pub depth: u32,
    pub heights: Vec<f32>,
    pub density: Vec<u8>,
    pub damp: Vec<u8>,
}

#[derive(Debug, Default)]
struct Field {
    map: Option<GrassMap>,
    /// How deep snow lies at each of the map's samples, in meters, if any: a clump it covers
    /// whole is not drawn.
    snow: Option<Vec<f32>>,
    cards: Vec<GrassCard>,
    /// Counts the maps and cards given, so the pass knows when to upload them.
    version: u64,
    /// Which way the wind blows (x and z), how strongly (0 to 1), and how fast its gusts run.
    wind: [f32; 4],
}

/// The field the grass grows on, for the game to fill: shared with the pass, so a copy taken
/// before the effects are handed to the executor stays connected.
#[derive(Debug, Clone, Default)]
pub struct GrassField(Rc<RefCell<Field>>);

impl GrassField {
    /// Where grass grows from now on, in place of what was before.
    pub fn set_map(&self, map: GrassMap) {
        let count = (map.width * map.depth) as usize;
        if map.heights.len() != count || map.density.len() != count || map.damp.len() != count {
            Log::err(format!(
                "Grass: a map of {} by {} with {} heights, {} densities and {} dampnesses; none drawn",
                map.width,
                map.depth,
                map.heights.len(),
                map.density.len(),
                map.damp.len()
            ));
            return;
        }
        let mut field = self.0.borrow_mut();
        field.map = Some(map);
        field.version += 1;
    }

    /// The cards the grass is made of, at most [`MOST_CARDS`], in place of what was before.
    pub fn set_cards(&self, mut cards: Vec<GrassCard>) {
        if cards.len() > MOST_CARDS {
            Log::warn(format!("Grass: {} cards, of which only the first {MOST_CARDS} are used", cards.len()));
            cards.truncate(MOST_CARDS);
        }
        let mut field = self.0.borrow_mut();
        field.cards = cards;
        field.version += 1;
    }

    /// How deep snow lies at each of the map's samples, in meters, row by row as the map's, from
    /// now on; [`None`] for none. A clump the snow covers whole - its card's top under the
    /// snow's - is not drawn; one it covers part of grows out of it.
    pub fn set_snow(&self, snow: Option<Vec<f32>>) {
        let mut field = self.0.borrow_mut();
        field.snow = snow;
        field.version += 1;
    }

    /// No grass anywhere.
    pub fn clear(&self) {
        let mut field = self.0.borrow_mut();
        field.map = None;
        field.version += 1;
    }

    /// The wind through the grass: blowing towards `direction` (world x and z), `strength` from
    /// still (0) to a gale (1), its gusts running `gusts` radians a second.
    pub fn set_wind(&self, direction: [f32; 2], strength: f32, gusts: f32) {
        let length = (direction[0] * direction[0] + direction[1] * direction[1]).sqrt().max(1.0e-6);
        self.0.borrow_mut().wind = [direction[0] / length, direction[1] / length, strength.clamp(0.0, 1.0), gusts];
    }
}

impl PartialEq for GrassField {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// The uniforms, as `grass.wgsl`'s `Frame`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Frame {
    view_projection: [[f32; 4]; 4],
    eye: [f32; 4],
    wind: [f32; 4],
    map: [f32; 4],
    grid: [f32; 4],
    shape: [f32; 4],
    look: [f32; 4],
    weights: [f32; 4],
    damp: [f32; 4],
    sizes: [[f32; 4]; MOST_CARDS],
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    sampler: wgpu::Sampler,
    /// The formats the pipeline was made for: the G-Buffer's colours and depth.
    formats: (Vec<wgpu::TextureFormat>, wgpu::TextureFormat),
    /// The map's and the cards' textures, and the bind group over them, once both are given.
    group: Option<wgpu::BindGroup>,
}

/// The grass in the renderer: see the module's description.
pub struct GrassPass {
    source_type_id: TypeId,
    settings: GrassSettings,
    field: GrassField,
    gpu: Option<Gpu>,
    /// The version of the map and cards uploaded.
    uploaded: u64,
    failed: bool,
}

impl GrassPass {
    pub(crate) fn new(source_type_id: TypeId, settings: GrassSettings, field: GrassField) -> Self {
        Self {
            source_type_id,
            settings,
            field,
            gpu: None,
            uploaded: 0,
            failed: false,
        }
    }

    /// How many cells the square round the eye is across.
    fn across(&self) -> u32 {
        (2.0 * self.settings.radius / self.settings.cell.max(0.05)).ceil() as u32 + 1
    }

    fn draw(&mut self, server: &WgpuGraphicsServer, ctx: &GBufferPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        let field = self.field.0.borrow();
        let Some(map) = field.map.as_ref() else {
            return Ok(Default::default());
        };
        if field.cards.is_empty() {
            return Ok(Default::default());
        }
        let texture = |t: &fyrox::graphics::gpu_texture::GpuTexture| -> Result<wgpu::Texture, FrameworkError> {
            t.as_any()
                .downcast_ref::<WgpuTexture>()
                .map(|t| t.wgpu_texture().clone())
                .ok_or_else(|| FrameworkError::Custom("Grass: a G-Buffer texture not of wgpu".into()))
        };
        let colour: Vec<wgpu::Texture> =
            ctx.framebuffer.color_attachments().iter().map(|a| texture(&a.texture)).collect::<Result<_, _>>()?;
        let depth = texture(
            &ctx.framebuffer
                .depth_attachment()
                .ok_or_else(|| FrameworkError::Custom("Grass: a G-Buffer with no depth".into()))?
                .texture,
        )?;
        let formats = (colour.iter().map(wgpu::Texture::format).collect::<Vec<_>>(), depth.format());
        let depth_format = formats.1;
        let device = &server.state.device;
        let queue = &server.state.queue;
        let across = self.across();
        if self.gpu.as_ref().is_none_or(|gpu| gpu.formats != formats) {
            self.gpu = Some(make(device, formats));
            self.uploaded = 0;
        }
        let gpu = self.gpu.as_mut().expect("made above");
        if self.uploaded != field.version {
            // The cards' textures may still be loading: tried again next frame.
            match cards_texture(device, queue, &field.cards, self.settings.cutoff) {
                Some(Ok(cards)) => {
                    gpu.group = Some(group(device, queue, gpu, map, field.snow.as_deref(), &cards));
                    self.uploaded = field.version;
                }
                Some(Err(why)) => {
                    Log::err(format!("Grass: {why}; none drawn"));
                    gpu.group = None;
                    self.uploaded = field.version;
                }
                None => {}
            }
        }
        let Some(bind_group) = gpu.group.as_ref() else {
            return Ok(Default::default());
        };

        let s = &self.settings;
        let mut view_projection = [[0.0; 4]; 4];
        for (column, values) in ctx.observer.position.view_projection_matrix.column_iter().zip(view_projection.iter_mut()) {
            values.copy_from_slice(column.as_slice());
        }
        let eye = ctx.observer.position.translation;
        let (mut weights, mut damp, mut sizes) = ([0.0; 4], [0.0; 4], [[0.0; 4]; MOST_CARDS]);
        for (i, card) in field.cards.iter().enumerate() {
            weights[i] = card.weight;
            damp[i] = card.damp_weight;
            sizes[i] = [card.width, card.height, 0.0, 0.0];
        }
        let frame = Frame {
            view_projection,
            eye: [eye.x, eye.y, eye.z, ctx.elapsed_time],
            wind: field.wind,
            map: [map.origin[0], map.origin[1], map.spacing, field.cards.len() as f32],
            grid: [s.cell, s.radius, across as f32, 0.0],
            shape: [s.scale.0, s.scale.1, s.thin_from, s.grow],
            look: [s.roughness, s.translucency, s.cutoff, s.shade],
            weights,
            damp,
            sizes,
        };
        queue.write_buffer(&gpu.uniforms, 0, bytemuck::bytes_of(&frame));

        server.flush_active_pass();
        // When nothing has drawn into the G-Buffer yet this frame, its clear is still waiting for
        // a draw: this one applies it.
        let clear = ctx
            .framebuffer
            .as_any()
            .downcast_ref::<WgpuFrameBuffer>()
            .map(WgpuFrameBuffer::take_pending_clear)
            .unwrap_or_default();
        let (colour_load, depth_load, stencil_load) = clear.load_ops();
        let views: Vec<wgpu::TextureView> = colour.iter().map(|t| t.create_view(&Default::default())).collect();
        let depth_view = depth.create_view(&Default::default());
        let mut encoder = server
            .frame_encoder
            .borrow_mut()
            .take()
            .unwrap_or_else(|| device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("Grass") }));
        {
            let attachments: Vec<Option<wgpu::RenderPassColorAttachment>> = views
                .iter()
                .map(|view| {
                    Some(wgpu::RenderPassColorAttachment {
                        view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations { load: colour_load, store: wgpu::StoreOp::Store },
                    })
                })
                .collect();
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Grass"),
                color_attachments: &attachments,
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations { load: depth_load, store: wgpu::StoreOp::Store }),
                    stencil_ops: depth_format.has_stencil_aspect().then_some(wgpu::Operations {
                        load: stencil_load,
                        store: wgpu::StoreOp::Store,
                    }),
                }),
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&gpu.pipeline);
            pass.set_bind_group(0, bind_group, &[]);
            pass.draw(0..CLUMP_VERTICES, 0..across * across);
        }
        server.gpu_mark_in(&mut encoder, "G-buffer: grass");
        *server.frame_encoder.borrow_mut() = Some(encoder);
        let mut stats = RenderPassStatistics::default();
        stats.draw_calls += 1;
        // At most a quarter of the square's cells are in view and within the radius.
        stats.triangles_rendered += (across * across * CLUMP_TRIANGLES) as usize / 4;
        Ok(stats)
    }
}

/// The pipeline, the uniforms and the cards' sampler, for a G-Buffer of `formats`.
fn make(device: &wgpu::Device, formats: (Vec<wgpu::TextureFormat>, wgpu::TextureFormat)) -> Gpu {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("Grass"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let map_entry = |binding| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Grass"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            map_entry(1),
            map_entry(2),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2Array,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Grass"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let targets: Vec<Option<wgpu::ColorTargetState>> = formats
        .0
        .iter()
        .map(|&format| Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL }))
        .collect();
    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("Grass"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            // Seen from both sides: a card is a single sheet.
            cull_mode: None,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: formats.1,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Less),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &targets,
        }),
        multiview_mask: None,
        cache: None,
    });
    let uniforms = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Grass"),
        size: std::mem::size_of::<Frame>() as u64,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
        label: Some("Grass"),
        address_mode_u: wgpu::AddressMode::ClampToEdge,
        address_mode_v: wgpu::AddressMode::ClampToEdge,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::MipmapFilterMode::Linear,
        anisotropy_clamp: 4,
        ..Default::default()
    });
    Gpu { pipeline, layout, uniforms, sampler, formats, group: None }
}

/// The cards' textures as one array of layers, a card a layer, with their mip levels made (see
/// the module's description); [`None`] while any is still loading, and an error if one failed or
/// they do not fit together.
fn cards_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    cards: &[GrassCard],
    cutoff: f32,
) -> Option<Result<wgpu::Texture, String>> {
    let mut images = Vec::with_capacity(cards.len());
    let mut srgb = None;
    for card in cards {
        if card.texture.is_failed_to_load() {
            return Some(Err(format!("a card's texture, {}, did not load", card.texture.kind())));
        }
        if !card.texture.is_ok() {
            return None;
        }
        let data = card.texture.data_ref();
        let TextureKind::Rectangle { width, height } = data.kind() else {
            return Some(Err("a card's texture is not flat".to_string()));
        };
        let is_srgb = match data.pixel_kind() {
            TexturePixelKind::RGBA8 => false,
            TexturePixelKind::SRGBA8 => true,
            other => return Some(Err(format!("a card's texture is {other:?}, not 8-bit RGBA"))),
        };
        if *srgb.get_or_insert(is_srgb) != is_srgb {
            return Some(Err("the cards' textures are not all sRGB, or all not".to_string()));
        }
        images.push((width, height, data.mip_level_data(0).to_vec()));
    }
    let (width, height) = (images[0].0, images[0].1);
    if images.iter().any(|(w, h, _)| (*w, *h) != (width, height)) {
        return Some(Err("the cards' textures are not all the same size".to_string()));
    }
    let levels: Vec<Vec<Vec<u8>>> = images.iter().map(|(_, _, pixels)| mip_levels(pixels, width, height, cutoff)).collect();
    let mip_count = levels[0].len() as u32;
    // Layer after layer, each its levels from the largest down.
    let data: Vec<u8> = levels.iter().flat_map(|layer| layer.iter().flatten().copied()).collect();
    Some(Ok(device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some("GrassCards"),
            size: wgpu::Extent3d { width, height, depth_or_array_layers: cards.len() as u32 },
            mip_level_count: mip_count,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: if srgb == Some(true) { wgpu::TextureFormat::Rgba8UnormSrgb } else { wgpu::TextureFormat::Rgba8Unorm },
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        &data,
    )))
}

/// The share of `pixels` (RGBA) whose alpha is `cutoff` or more, after scaling it by `scale`.
fn coverage(pixels: &[u8], cutoff: f32, scale: f32) -> f32 {
    let count = pixels.len() / 4;
    let over = pixels.chunks_exact(4).filter(|p| p[3] as f32 / 255.0 * scale >= cutoff).count();
    over as f32 / count.max(1) as f32
}

/// `pixels` (RGBA, `width` by `height`) and its smaller mip levels down to a pixel, each halved
/// from the last: its colour averaged by alpha, so the soil-coloured fringe of a blade does not
/// darken it; its alpha averaged, then scaled so as much of it is over `cutoff` as of the full-size
/// level.
fn mip_levels(pixels: &[u8], width: u32, height: u32, cutoff: f32) -> Vec<Vec<u8>> {
    let target = coverage(pixels, cutoff, 1.0);
    let mut levels = vec![pixels.to_vec()];
    let (mut w, mut h) = (width, height);
    while w > 1 || h > 1 {
        let (nw, nh) = ((w / 2).max(1), (h / 2).max(1));
        let last = levels.last().expect("one at least");
        let mut next = vec![0u8; (nw * nh * 4) as usize];
        for y in 0..nh {
            for x in 0..nw {
                let (mut rgb, mut alpha, mut n) = ([0.0f32; 3], 0.0f32, 0.0f32);
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let (sx, sy) = ((2 * x + dx).min(w - 1), (2 * y + dy).min(h - 1));
                    let p = &last[((sy * w + sx) * 4) as usize..][..4];
                    let a = p[3] as f32 / 255.0;
                    for c in 0..3 {
                        rgb[c] += p[c] as f32 * a;
                    }
                    alpha += a;
                    n += 1.0;
                }
                let o = ((y * nw + x) * 4) as usize;
                for c in 0..3 {
                    next[o + c] = if alpha > 0.0 { (rgb[c] / alpha).round() as u8 } else { 0 };
                }
                next[o + 3] = (alpha / n * 255.0).round() as u8;
            }
        }
        // The alpha scaled to keep the coverage: found by halving the range of scales.
        let (mut low, mut high) = (0.0f32, 8.0f32);
        for _ in 0..16 {
            let middle = 0.5 * (low + high);
            if coverage(&next, cutoff, middle) < target {
                low = middle;
            } else {
                high = middle;
            }
        }
        for p in next.chunks_exact_mut(4) {
            p[3] = (p[3] as f32 * high).min(255.0).round() as u8;
        }
        levels.push(next);
        (w, h) = (nw, nh);
    }
    levels
}

/// The map's textures and the cards', uploaded, and the bind group over them and the uniforms.
fn group(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &Gpu, map: &GrassMap, snow: Option<&[f32]>, cards: &wgpu::Texture) -> wgpu::BindGroup {
    let size = wgpu::Extent3d { width: map.width, height: map.depth, depth_or_array_layers: 1 };
    let texture = |label, format, data: &[u8]| {
        device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            data,
        )
    };
    // The ground's height, and the snow's depth on it.
    let snow = snow.filter(|snow| snow.len() == map.heights.len());
    let heights: Vec<f32> = map
        .heights
        .iter()
        .enumerate()
        .flat_map(|(i, &h)| [h, snow.map_or(0.0, |snow| snow[i])])
        .collect();
    let heights = texture("GrassHeights", wgpu::TextureFormat::Rg32Float, bytemuck::cast_slice(&heights));
    let ground: Vec<u8> = map.density.iter().zip(&map.damp).flat_map(|(&d, &w)| [d, w]).collect();
    let ground = texture("GrassGround", wgpu::TextureFormat::Rg8Unorm, &ground);
    let cards_view = cards.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("Grass"),
        layout: &gpu.layout,
        entries: &[
            wgpu::BindGroupEntry { binding: 0, resource: gpu.uniforms.as_entire_binding() },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&heights.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: wgpu::BindingResource::TextureView(&ground.create_view(&Default::default())),
            },
            wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(&cards_view) },
            wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(&gpu.sampler) },
        ],
    })
}

impl SceneRenderPass for GrassPass {
    fn on_gbuffer_render(&mut self, ctx: GBufferPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        if self.failed {
            return Ok(Default::default());
        }
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(Default::default());
        };
        match self.draw(server, &ctx) {
            Ok(stats) => Ok(stats),
            Err(err) => {
                Log::err(format!("Grass failed, and is drawn no more: {err:?}"));
                self.failed = true;
                Ok(Default::default())
            }
        }
    }

    fn source_type_id(&self) -> TypeId {
        self.source_type_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_shader_is_valid_wgsl() {
        let module = wgpu::naga::front::wgsl::parse_str(SHADER).unwrap_or_else(|e| panic!("{}", e.emit_to_string(SHADER)));
        wgpu::naga::valid::Validator::new(wgpu::naga::valid::ValidationFlags::all(), wgpu::naga::valid::Capabilities::all())
            .validate(&module)
            .unwrap_or_else(|e| panic!("{e:?}"));
    }

    #[test]
    fn the_uniforms_are_as_the_shader_has_them() {
        // A matrix, eight vectors of four floats, and a vector for each card.
        assert_eq!(std::mem::size_of::<Frame>(), 64 + 8 * 16 + MOST_CARDS * 16);
    }

    #[test]
    fn a_map_of_the_wrong_size_is_refused() {
        let field = GrassField::default();
        let map = |n: usize| GrassMap {
            origin: [0.0; 2],
            spacing: 1.0,
            width: 2,
            depth: 2,
            heights: vec![0.0; n],
            density: vec![255; 4],
            damp: vec![0; 4],
        };
        field.set_map(map(3));
        assert!(field.0.borrow().map.is_none());
        field.set_map(map(4));
        assert_eq!(field.0.borrow().version, 1);
    }

    #[test]
    fn smaller_levels_keep_a_thin_blades_coverage() {
        // An 8 by 8 card: a blade a pixel wide down every fourth column, solid.
        let (w, h) = (8u32, 8u32);
        let mut pixels = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in (0..w).step_by(4) {
                let o = ((y * w + x) * 4) as usize;
                pixels[o..o + 4].copy_from_slice(&[40, 120, 30, 255]);
            }
        }
        let levels = mip_levels(&pixels, w, h, 0.4);
        assert_eq!(levels.len(), 4);
        let full = coverage(&levels[0], 0.4, 1.0);
        // At 2 by 2 a pixel is a quarter blade: averaged plainly, under the cutoff, the card gone.
        assert!(coverage(&levels[2], 0.4, 1.0) > 0.0);
        for level in &levels[1..3] {
            let kept = coverage(level, 0.4, 1.0);
            assert!(kept >= full * 0.9, "{kept} of {full}");
            // The blades' colour, not darkened by the empty pixels round them.
            let p = level.chunks_exact(4).find(|p| p[3] > 0).unwrap();
            assert!(p[1] >= 110, "{p:?}");
        }
    }
}
