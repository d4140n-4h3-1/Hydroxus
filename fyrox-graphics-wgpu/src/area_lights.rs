//! Area lights: rectangles that glow - a lit panel, the rim of a screen - and light what is round
//! them from all of their surface, not from one point as the renderer's lights do.
//!
//! The light is worked out per pixel from the G-buffer after the scene is lit, by sampling points
//! over each rectangle (see `shaders/area_lights.wgsl`), then smoothed into a texture for the
//! renderer to add to the frame.
//! Where the hardware traces rays, each point is checked against a [`RayTracedScene`] for anything
//! in the way - and for glass, which colours what it lets through - so the lights cast shadows,
//! soft ones, as big lights do. Elsewhere, WebGL among them, the lights shine through everything.

use crate::{raytracing::RayTracedScene, server::WgpuGraphicsServer, texture::WgpuTexture};
use fyrox_graphics::{
    error::FrameworkError,
    gpu_texture::{GpuTexture, GpuTextureDescriptor, GpuTextureKind, PixelKind},
    server::GraphicsServer,
};
use wgpu::util::DeviceExt;

/// The most area lights lit at once. Any more are left out.
pub const MAX_AREA_LIGHTS: usize = 16;

/// A glowing rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaLight {
    /// One corner, in world space.
    pub corner: [f32; 3],
    /// The two edges from that corner. The light shines from the side their cross product
    /// points to.
    pub edges: [[f32; 3]; 2],
    /// Its colour times how bright it is: what a pixel facing it, a meter away, would get from
    /// a square meter of it.
    pub colour: [f32; 3],
    /// How far from its middle it lights anything, in meters. The light fades to nothing there.
    pub reach: f32,
    /// Whether it shines from both faces.
    pub two_sided: bool,
    /// How many points along each edge the light is sampled at, 1 to 16. A long thin strip
    /// needs several along its length and one across.
    pub samples: [u32; 2],
}

/// What the area lights need to know besides the lights.
#[derive(Debug, Clone, Copy)]
pub struct AreaLightParameters {
    /// Turns a screen position and depth back into a world position.
    pub inverse_view_projection: [[f32; 4]; 4],
    /// How far off the surface a ray starts, in meters, so a surface does not shadow itself.
    pub bias: f32,
    /// True when render targets are stored top row first, which is how this backend stores them.
    pub flip_v: bool,
}

/// The passes that light with area lights, kept between frames so their pipelines are built once.
pub struct AreaLighter {
    /// Traced, when the hardware can, and not.
    traced: Option<wgpu::RenderPipeline>,
    traced_layout: Option<wgpu::BindGroupLayout>,
    plain: wgpu::RenderPipeline,
    plain_layout: wgpu::BindGroupLayout,
    sampler: wgpu::Sampler,
    blur_layout: wgpu::BindGroupLayout,
    blur: wgpu::RenderPipeline,
    gathered: Option<GpuTexture>,
    smoothed: Option<GpuTexture>,
}

const SHADER: &str = include_str!("shaders/area_lights.wgsl");
const BLUR_SHADER: &str = include_str!("shaders/area_lights_blur.wgsl");

/// Shadow rays towards each point of a light, through glass.
const TRACED: &str = r#"enable wgpu_ray_query;

@group(0) @binding(0) var sceneGeometry: acceleration_structure;

fn visible(origin: vec3f, direction: vec3f, reach: f32) -> vec3f {
    const TERMINATE_ON_FIRST_HIT = 0x4u;
    var query: ray_query;
    rayQueryInitialize(&query, sceneGeometry, RayDesc(
        TERMINATE_ON_FIRST_HIT,
        0xFFu,
        0.001,
        max(reach, 0.002),
        origin,
        direction,
    ));
    var through = vec3f(1.0);
    while (rayQueryProceed(&query)) {
        let lets = rayQueryGetCandidateIntersection(&query).instance_custom_data;
        through *= vec3f(
            f32(lets & 0xFFu),
            f32((lets >> 8u) & 0xFFu),
            f32((lets >> 16u) & 0xFFu),
        ) / 255.0;
    }
    if (rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE) {
        return vec3f(0.0);
    }
    return through;
}
"#;

/// Everything in the light's sight, where nothing is traced.
const PLAIN: &str = r#"
fn visible(origin: vec3f, direction: vec3f, reach: f32) -> vec3f {
    return vec3f(1.0);
}
"#;

/// How far off a pixel's plane, in meters, a neighbour may lie and still be blurred with it.
const BLUR_PLANE_TOLERANCE: f32 = 0.05;

fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty,
        count: None,
    }
}

fn texture(binding: u32) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
    )
}

fn sampler(binding: u32) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
    )
}

fn uniform(binding: u32) -> wgpu::BindGroupLayoutEntry {
    entry(
        binding,
        wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
    )
}

/// A pipeline drawing one full-screen triangle into a target of `format`.
fn pipeline(
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

/// The G-buffer's textures, bound after the geometry at 1 to 5.
fn gbuffer_entries(first: u32) -> [wgpu::BindGroupLayoutEntry; 5] {
    [
        texture(first),
        sampler(first + 1),
        uniform(first + 2),
        texture(first + 3),
        texture(first + 4),
    ]
}

impl WgpuGraphicsServer {
    /// Creates the passes that light with area lights. They trace shadows where the hardware
    /// can.
    pub fn create_area_lighter(&self) -> AreaLighter {
        let device = &self.state.device;
        let format = wgpu::TextureFormat::Rgba16Float;

        let plain_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AreaLights"),
            entries: &gbuffer_entries(1),
        });
        let plain = pipeline(
            device,
            "AreaLights",
            &format!("{PLAIN}{SHADER}"),
            &plain_layout,
            format,
        );

        let (traced, traced_layout) = if self.ray_tracing {
            let mut entries = vec![entry(
                0,
                wgpu::BindingType::AccelerationStructure {
                    vertex_return: false,
                },
            )];
            entries.extend(gbuffer_entries(1));
            let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("AreaLightsTraced"),
                entries: &entries,
            });
            let traced = pipeline(
                device,
                "AreaLightsTraced",
                &format!("{TRACED}{SHADER}"),
                &layout,
                format,
            );
            (Some(traced), Some(layout))
        } else {
            (None, None)
        };

        let blur_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("AreaLightsBlur"),
            entries: &[
                texture(0),
                texture(1),
                texture(2),
                sampler(3),
                uniform(4),
            ],
        });

        let blur = pipeline(device, "AreaLightsBlur", BLUR_SHADER, &blur_layout, format);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("AreaLights"),
            ..Default::default()
        });
        AreaLighter {
            traced,
            traced_layout,
            plain,
            plain_layout,
            sampler,
            blur_layout,
            blur,
            gathered: None,
            smoothed: None,
        }
    }
}

impl AreaLighter {
    /// Whether the lights cast shadows here: whether the hardware traces rays.
    pub fn traces(&self) -> bool {
        self.traced.is_some()
    }

    /// Works out the light `lights` shine on what the G-buffer holds - `depth`, `normals` and
    /// `colours` are its - and returns it as a texture the G-buffer's size, to be added to the
    /// lit frame. Shadows are traced against `scene` when given and the hardware can.
    ///
    /// The same texture is returned every time, and each call overwrites it.
    #[allow(clippy::too_many_arguments)]
    pub fn light(
        &mut self,
        server: &WgpuGraphicsServer,
        scene: Option<&RayTracedScene>,
        depth: &GpuTexture,
        normals: &GpuTexture,
        colours: &GpuTexture,
        lights: &[AreaLight],
        parameters: AreaLightParameters,
    ) -> Result<Option<GpuTexture>, FrameworkError> {
        if lights.is_empty() {
            return Ok(None);
        }
        let GpuTextureKind::Rectangle { width, height } = depth.kind() else {
            return Ok(None);
        };
        let gathered = light_texture(&mut self.gathered, server, "AreaLightsGathered", width, height)?;
        let smoothed = light_texture(&mut self.smoothed, server, "AreaLights", width, height)?;
        let (Some(depth), Some(normals), Some(colours), Some(gathered_target), Some(smoothed_target)) = (
            depth.as_any().downcast_ref::<WgpuTexture>(),
            normals.as_any().downcast_ref::<WgpuTexture>(),
            colours.as_any().downcast_ref::<WgpuTexture>(),
            gathered.as_any().downcast_ref::<WgpuTexture>(),
            smoothed.as_any().downcast_ref::<WgpuTexture>(),
        ) else {
            return Ok(None);
        };
        let gathered = gathered_target;

        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Light {
            corner: [f32; 4],
            edge_u: [f32; 4],
            edge_v: [f32; 4],
            colour: [f32; 4],
        }
        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct Uniforms {
            inverse_view_projection: [[f32; 4]; 4],
            screen_size: [f32; 2],
            flip_v: u32,
            light_count: u32,
            bias: f32,
            _padding: [u32; 3],
            lights: [Light; MAX_AREA_LIGHTS],
        }
        let mut uniforms = Uniforms {
            inverse_view_projection: parameters.inverse_view_projection,
            screen_size: [width as f32, height as f32],
            flip_v: parameters.flip_v as u32,
            light_count: lights.len().min(MAX_AREA_LIGHTS) as u32,
            bias: parameters.bias,
            _padding: [0; 3],
            lights: bytemuck::Zeroable::zeroed(),
        };
        for (slot, light) in uniforms.lights.iter_mut().zip(lights) {
            let c = light.corner;
            let [u, v] = light.edges;
            *slot = Light {
                corner: [c[0], c[1], c[2], light.reach],
                edge_u: [u[0], u[1], u[2], if light.two_sided { 1.0 } else { 0.0 }],
                edge_v: [v[0], v[1], v[2], light.samples[0].clamp(1, 16) as f32],
                colour: [
                    light.colour[0],
                    light.colour[1],
                    light.colour[2],
                    light.samples[1].clamp(1, 16) as f32,
                ],
            };
        }

        let device = &server.state.device;
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AreaLights"),
            contents: bytemuck::bytes_of(&uniforms),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let depth_view = depth
            .wgpu_texture()
            .create_view(&wgpu::TextureViewDescriptor {
                aspect: wgpu::TextureAspect::DepthOnly,
                ..Default::default()
            });
        let mut entries = vec![
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
            wgpu::BindGroupEntry {
                binding: 5,
                resource: wgpu::BindingResource::TextureView(colours.wgpu_view()),
            },
        ];
        let (pipeline, layout) = match (scene, &self.traced, &self.traced_layout) {
            (Some(scene), Some(pipeline), Some(layout)) => {
                entries.push(wgpu::BindGroupEntry {
                    binding: 0,
                    resource: scene.tlas.as_binding(),
                });
                (pipeline, layout)
            }
            _ => (&self.plain, &self.plain_layout),
        };
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("AreaLights"),
            layout,
            entries: &entries,
        });


        #[repr(C)]
        #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
        struct BlurUniforms {
            inverse_view_projection: [[f32; 4]; 4],
            screen_size: [f32; 2],
            flip_v: u32,
            plane_tolerance: f32,
        }
        let blur_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("AreaLightsBlur"),
            contents: bytemuck::bytes_of(&BlurUniforms {
                inverse_view_projection: parameters.inverse_view_projection,
                screen_size: [width as f32, height as f32],
                flip_v: parameters.flip_v as u32,
                plane_tolerance: BLUR_PLANE_TOLERANCE,
            }),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let blur_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("AreaLightsBlur"),
            layout: &self.blur_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(gathered.wgpu_view()),
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
                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: blur_buffer.as_entire_binding(),
                },
            ],
        });

        // Anything recorded so far has to reach the GPU first: the G-buffer read here is drawn by
        // it.
        server.flush_active_pass();
        let mut encoder = server
            .frame_encoder
            .borrow_mut()
            .take()
            .unwrap_or_else(|| {
                device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("AreaLights"),
                })
            });
        draw(
            &mut encoder,
            "AreaLights",
            gathered.wgpu_view(),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            pipeline,
            &bind_group,
        );
        draw(
            &mut encoder,
            "AreaLightsBlur",
            smoothed_target.wgpu_view(),
            wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
            &self.blur,
            &blur_bind_group,
        );
        *server.frame_encoder.borrow_mut() = Some(encoder);
        Ok(Some(smoothed))
    }
}

/// The light texture in `slot`, made again if it is not `width` by `height`.
fn light_texture(
    slot: &mut Option<GpuTexture>,
    server: &WgpuGraphicsServer,
    name: &'static str,
    width: usize,
    height: usize,
) -> Result<GpuTexture, FrameworkError> {
    let matches = slot.as_ref().is_some_and(|texture| {
        matches!(texture.kind(), GpuTextureKind::Rectangle { width: w, height: h } if w == width && h == height)
    });
    if !matches {
        *slot = Some(server.create_texture(GpuTextureDescriptor {
            name,
            kind: GpuTextureKind::Rectangle { width, height },
            pixel_kind: PixelKind::RGBA16F,
            ..Default::default()
        })?);
    }
    slot.clone()
        .ok_or_else(|| FrameworkError::Custom("area light texture".into()))
}

fn draw(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    pipeline: &wgpu::RenderPipeline,
    bind_group: &wgpu::BindGroup,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations {
                load,
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bind_group, &[]);
    pass.draw(0..3, 0..1);
}
