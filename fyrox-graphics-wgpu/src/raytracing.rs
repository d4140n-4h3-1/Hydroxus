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
//! Availability: [`WgpuGraphicsServer::ray_tracing`] is false unless the adapter reported ray
//! queries, in which case everything here returns [`None`] and the caller falls back to whatever
//! it did before.

use crate::{server::WgpuGraphicsServer, texture::WgpuTexture};
use fyrox_graphics::{
    error::FrameworkError,
    gpu_texture::{GpuTexture, GpuTextureDescriptor, GpuTextureKind, PixelKind},
    server::GraphicsServer,
};
use wgpu::util::DeviceExt;

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
    /// 1, where a ray crosses it. Solid geometry lets nothing through whatever this says.
    pub lets_through: [f32; 3],
}

/// `lets_through` as an instance's custom data: a byte for each of red, green and blue.
fn lets_through_bits([r, g, b]: [f32; 3]) -> u32 {
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u32;
    byte(r) | (byte(g) << 8) | (byte(b) << 16)
}

/// The transform that leaves geometry where it is.
pub const IDENTITY_TRANSFORM: [f32; 12] = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0];

/// Scene geometry in a form the ray tracing hardware can trace against: copies of geometry,
/// each where it is.
pub struct RayTracedScene {
    tlas: wgpu::Tlas,
    /// How many copies it has room for, and how many triangles the ones it has come to.
    capacity: u32,
    triangle_count: u32,
    /// For a scene built from triangles alone, the geometry it holds.
    _geometry: Option<RayTracedGeometry>,
}

impl RayTracedScene {
    /// How many triangles were put into the structure, counting each copy.
    pub fn triangle_count(&self) -> u32 {
        self.triangle_count
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
    /// Rays per pixel, spread over the light's size. One gives hard shadows. More give soft
    /// edges, and the mask is then blurred to smooth out the differences between neighboring
    /// pixels.
    pub samples: u32,
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
}

const SHADER: &str = include_str!("shaders/shadow_rays.wgsl");
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
                format: wgpu::TextureFormat::Rgba8Unorm,
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
    let matches = slot.as_ref().is_some_and(|texture| {
        matches!(texture.kind(), GpuTextureKind::Rectangle { width: w, height: h } if w == width && h == height)
    });
    if !matches {
        *slot = Some(server.create_texture(GpuTextureDescriptor {
            name,
            kind: GpuTextureKind::Rectangle { width, height },
            pixel_kind: PixelKind::RGBA8,
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
        self.build_geometry(vertices, indices, updatable, false)
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
        self.build_geometry(vertices, indices, updatable, true)
    }

    fn build_geometry(
        &self,
        vertices: &[f32],
        indices: &[u32],
        updatable: bool,
        see_through: bool,
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
            usage: wgpu::BufferUsages::BLAS_INPUT | wgpu::BufferUsages::COPY_DST,
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
            // Glass is only ever passed through, and each pane has to be counted once.
            flags: if see_through {
                wgpu::AccelerationStructureGeometryFlags::NO_DUPLICATE_ANY_HIT_INVOCATION
            } else {
                wgpu::AccelerationStructureGeometryFlags::OPAQUE
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
            see_through,
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
            triangle_count: 0,
            _geometry: None,
        };
        self.update_ray_traced_instances(&mut scene, instances)?;
        Ok(Some(scene))
    }

    /// Places `instances` in `scene` in place of what it had, making room for them if it has to.
    /// Cheap enough to do every frame.
    pub fn update_ray_traced_instances(
        &self,
        scene: &mut RayTracedScene,
        instances: &[RayTracedInstance],
    ) -> Result<(), FrameworkError> {
        if instances.len() as u32 > scene.capacity {
            let capacity = (instances.len() as u32).next_power_of_two();
            scene.tlas = self.state.device.create_tlas(&wgpu::CreateTlasDescriptor {
                label: Some("RayTracedScene"),
                max_instances: capacity,
                flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                update_mode: wgpu::AccelerationStructureUpdateMode::Build,
            });
            scene.capacity = capacity;
        }
        let slots = scene
            .tlas
            .get_mut_slice(0..scene.capacity as usize)
            .ok_or_else(|| FrameworkError::Custom("ray traced scene instances".into()))?;
        for (slot, instance) in slots.iter_mut().zip(
            instances
                .iter()
                .map(Some)
                .chain(std::iter::repeat(None)),
        ) {
            *slot = instance.map(|instance| {
                wgpu::TlasInstance::new(
                    &instance.geometry.blas,
                    instance.transform,
                    lets_through_bits(instance.lets_through),
                    0xff,
                )
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
            ],
        });
        let pipeline = mask_pipeline(device, "ShadowRays", SHADER, &layout);

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
        Some(ShadowTracer {
            pipeline,
            layout,
            sampler,
            blur_pipeline,
            blur_layout,
            mask: None,
            blurred: None,
        })
    }
}

impl ShadowTracer {
    /// Traces shadow rays for each pixel and returns a mask: how much of the light's red, green
    /// and blue reaches it - 1 where all of it does, 0 where the geometry blocks it, in between
    /// where it blocks part of the light, and coloured where it came through glass. The mask is a
    /// texture like any other, to be read by an effect.
    ///
    /// The same texture is returned every time, so each call overwrites the previous mask. That
    /// is fine for reading it right after, as the renderer does for one light at a time: the GPU
    /// runs the work in the order it was recorded.
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
        let mask = mask_texture(&mut self.mask, server, "RayTracedShadowMask", width, height)?;
        let soft = parameters.samples > 1;
        let blurred = if soft {
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
            _padding: u32,
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
            _padding: 0,
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
        draw_mask(
            &mut encoder,
            "ShadowRays",
            target,
            &self.pipeline,
            &bind_group,
            scissor,
        );

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
            }
            let blur_uniforms = BlurUniforms {
                inverse_view_projection: parameters.inverse_view_projection,
                screen_size,
                flip_v: parameters.flip_v as u32,
                plane_tolerance: BLUR_PLANE_TOLERANCE,
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
        *server.frame_encoder.borrow_mut() = Some(encoder);

        Ok(Some(result))
    }
}
