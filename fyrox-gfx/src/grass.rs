//! Grass: blades of it all over the ground round the eye, drawn straight into the G-Buffer in one
//! instanced draw, with no node or mesh of their own (see `grass.wgsl`).
//!
//! A field of grass of real plant models costs too much: thousands of triangles a tuft, each tuft
//! a node, traced by the shadows and by Candelite, and bent anew every frame it sways. Here the
//! blades are made in the vertex shader instead - a narrowing strip of a few quads each, a few
//! blades to each cell of a square of the ground round the eye - placed, turned, sized and shaded
//! by a hash of where they grow, so the same blades are there every frame, and bent by the wind
//! there too. Being in the G-Buffer, they are lit, shadowed (the trees' shadows fall on them),
//! occluded and fogged as everything else is, and Candelite's light from the screen reaches them;
//! being nowhere else, they cast no shadow and nothing traces them.
//!
//! The game says where grass grows, and how high the ground is there, with a [`GrassMap`] - a grid
//! of the ground's height and of how thick the grass is, 0 to 1 - through [`GrassField::set_map`],
//! and which way and how hard the wind blows through [`GrassField::set_wind`]. Until it has given
//! a map, no grass is drawn. [`crate::GraphicsEffects::with_grass`] adds it.

use fyrox::{
    core::log::Log,
    graphics::{error::FrameworkError, stats::RenderPassStatistics},
    renderer::{GBufferPassContext, SceneRenderPass},
};
use fyrox_graphics_wgpu::{framebuffer::WgpuFrameBuffer, server::WgpuGraphicsServer, texture::WgpuTexture};
use std::{any::TypeId, cell::RefCell, rc::Rc};
use wgpu::util::DeviceExt;

const SHADER: &str = include_str!("grass.wgsl");
/// Vertices in a blade, as `grass.wgsl` makes it: three quads and a triangle at the tip.
const BLADE_VERTICES: u32 = 21;
/// Triangles in a blade.
const BLADE_TRIANGLES: u32 = 7;

/// What the grass looks like, and how far round the eye it grows.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GrassSettings {
    /// How far from the eye grass is drawn, in meters.
    pub radius: f32,
    /// From what share of the radius on it thins out, to none at the radius.
    pub thin_from: f32,
    /// The side of a cell of the ground, in meters, and how many blades a cell has at most, where
    /// the grass is thickest.
    pub cell: f32,
    pub blades: u32,
    /// A blade's height and its width at the root, in meters (each blade 0.6 to 1.4 times that
    /// tall); how far it leans, at most, as a share of its height.
    pub height: f32,
    pub width: f32,
    pub lean: f32,
    /// The colour at the root and at the tip, linear.
    pub root: [f32; 3],
    pub tip: [f32; 3],
    /// How rough it is, and how much light comes through a blade, 0 to 1.
    pub roughness: f32,
    pub translucency: f32,
}

impl Default for GrassSettings {
    fn default() -> Self {
        Self {
            radius: 35.0,
            thin_from: 0.45,
            cell: 0.5,
            blades: 10,
            height: 0.32,
            width: 0.022,
            lean: 0.35,
            root: [0.035, 0.06, 0.02],
            tip: [0.16, 0.22, 0.07],
            roughness: 0.85,
            translucency: 0.3,
        }
    }
}

/// Where grass grows: a grid of samples `spacing` meters apart, `width` along x and `depth` along
/// z, the first at `origin` (world x and z), row by row along x. Each the ground's height there, in
/// meters, and how thick the grass is, 0 for none to 255 for the most.
#[derive(Debug, Clone, PartialEq)]
pub struct GrassMap {
    pub origin: [f32; 2],
    pub spacing: f32,
    pub width: u32,
    pub depth: u32,
    pub heights: Vec<f32>,
    pub density: Vec<u8>,
}

#[derive(Debug, Default)]
struct Field {
    map: Option<GrassMap>,
    /// Counts the maps given, so the pass knows when to upload one.
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
        if map.heights.len() != (map.width * map.depth) as usize || map.density.len() != map.heights.len() {
            Log::err(format!(
                "Grass: a map of {} by {} with {} heights and {} densities; none drawn",
                map.width,
                map.depth,
                map.heights.len(),
                map.density.len()
            ));
            return;
        }
        let mut field = self.0.borrow_mut();
        field.map = Some(map);
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
    root: [f32; 4],
    tip: [f32; 4],
}

struct Gpu {
    pipeline: wgpu::RenderPipeline,
    layout: wgpu::BindGroupLayout,
    uniforms: wgpu::Buffer,
    /// The formats the pipeline was made for: the G-Buffer's colours and depth.
    formats: (Vec<wgpu::TextureFormat>, wgpu::TextureFormat),
    /// The map's textures, and the bind group over them, once one is given.
    group: Option<wgpu::BindGroup>,
}

/// The grass in the renderer: see the module's description.
pub struct GrassPass {
    source_type_id: TypeId,
    settings: GrassSettings,
    field: GrassField,
    gpu: Option<Gpu>,
    /// The version of the map uploaded.
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
            gpu.group = Some(upload(device, queue, gpu, map));
            self.uploaded = field.version;
        }
        let Some(group) = gpu.group.as_ref() else {
            return Ok(Default::default());
        };

        let s = &self.settings;
        let mut view_projection = [[0.0; 4]; 4];
        for (column, values) in ctx.observer.position.view_projection_matrix.column_iter().zip(view_projection.iter_mut()) {
            values.copy_from_slice(column.as_slice());
        }
        let eye = ctx.observer.position.translation;
        let frame = Frame {
            view_projection,
            eye: [eye.x, eye.y, eye.z, ctx.elapsed_time],
            wind: field.wind,
            map: [map.origin[0], map.origin[1], map.spacing, 0.0],
            grid: [s.cell, s.radius, across as f32, s.blades as f32],
            shape: [s.height, s.width, s.thin_from, s.lean],
            root: [s.root[0], s.root[1], s.root[2], s.roughness],
            tip: [s.tip[0], s.tip[1], s.tip[2], s.translucency],
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
            pass.set_bind_group(0, group, &[]);
            pass.draw(0..BLADE_VERTICES * s.blades, 0..across * across);
        }
        server.gpu_mark_in(&mut encoder, "G-buffer: grass");
        *server.frame_encoder.borrow_mut() = Some(encoder);
        let mut stats = RenderPassStatistics::default();
        stats.draw_calls += 1;
        // At most; most cells grow fewer, and those out of view or past the radius none.
        stats.triangles_rendered += (across * across * s.blades * BLADE_TRIANGLES) as usize / 4;
        Ok(stats)
    }
}

/// The pipeline and the uniforms, for a G-Buffer of `formats`.
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
            // Seen from both sides: a blade is a single sheet.
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
    Gpu { pipeline, layout, uniforms, formats, group: None }
}

/// The map's textures, uploaded, and the bind group over them and the uniforms.
fn upload(device: &wgpu::Device, queue: &wgpu::Queue, gpu: &Gpu, map: &GrassMap) -> wgpu::BindGroup {
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
    let heights = texture("GrassHeights", wgpu::TextureFormat::R32Float, bytemuck::cast_slice(&map.heights));
    let density = texture("GrassDensity", wgpu::TextureFormat::R8Unorm, &map.density);
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
                resource: wgpu::BindingResource::TextureView(&density.create_view(&Default::default())),
            },
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
        // A matrix and seven vectors of four floats.
        assert_eq!(std::mem::size_of::<Frame>(), 64 + 7 * 16);
    }

    #[test]
    fn a_map_of_the_wrong_size_is_refused() {
        let field = GrassField::default();
        field.set_map(GrassMap { origin: [0.0; 2], spacing: 1.0, width: 2, depth: 2, heights: vec![0.0; 3], density: vec![0; 4] });
        assert!(field.0.borrow().map.is_none());
        field.set_map(GrassMap { origin: [0.0; 2], spacing: 1.0, width: 2, depth: 2, heights: vec![0.0; 4], density: vec![255; 4] });
        assert_eq!(field.0.borrow().version, 1);
    }
}
