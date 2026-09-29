//! Copying depth by drawing, for WebGL. wgpu's GL backend copies a texture by reading it as a
//! framebuffer's color, which a depth texture cannot be, so WebGL refuses every depth copy. The
//! renderer copies the G-buffer's depth into the scene's framebuffers each frame, and without it
//! light volumes have nothing to test against and the scene goes unlit. Drawing the source depth
//! as the fragment depth copies it instead. Stencil is not copied: no draw can write it.
//!
//! The source is read through a sampler that does not filter. wgpu only sets a new texture's
//! filtering to nearest for formats with one kind of sample, which a depth-stencil texture is
//! not, and WebGL counts a depth texture that filters otherwise as incomplete: read without a
//! sampler of its own, every texel comes back 0, as if everything touched the camera.

use std::cell::RefCell;
use std::collections::HashMap;

const WGSL: &str = r#"
@group(0) @binding(0) var source: texture_2d<f32>;
@group(0) @binding(1) var nearest: sampler;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4f {
    // One triangle over the whole viewport.
    let corner = vec2f(f32((i << 1u) & 2u), f32(i & 2u));
    return vec4f(corner * 2.0 - 1.0, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) at: vec4f) -> @builtin(frag_depth) f32 {
    let uv = at.xy / vec2f(textureDimensions(source, 0));
    return textureSampleLevel(source, nearest, uv, 0.0).r;
}
"#;

/// The pipelines that copy depth, one for each depth format copied into.
#[derive(Default)]
pub struct DepthCopy {
    pipelines: RefCell<HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>>,
    sampler: RefCell<Option<wgpu::Sampler>>,
}

impl DepthCopy {
    /// Records a copy of `source`'s depth into `dest`'s, over `width` by `height` texels from
    /// (`x`, `y`) in both.
    #[allow(clippy::too_many_arguments)]
    pub fn copy(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        source: &wgpu::Texture,
        dest: &wgpu::Texture,
        dest_level: u32,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
    ) {
        let format = dest.format();
        let mut pipelines = self.pipelines.borrow_mut();
        let pipeline = pipelines
            .entry(format)
            .or_insert_with(|| make_pipeline(device, format));

        let source_view = source.create_view(&wgpu::TextureViewDescriptor {
            label: Some("DepthCopySource"),
            aspect: wgpu::TextureAspect::DepthOnly,
            base_mip_level: 0,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let dest_view = dest.create_view(&wgpu::TextureViewDescriptor {
            label: Some("DepthCopyDest"),
            base_mip_level: dest_level,
            mip_level_count: Some(1),
            ..Default::default()
        });
        let mut sampler = self.sampler.borrow_mut();
        let sampler = sampler.get_or_insert_with(|| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("DepthCopy"),
                ..Default::default()
            })
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("DepthCopy"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&source_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let has_stencil = format.has_stencil_aspect();
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("DepthCopy"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &dest_view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: has_stencil.then_some(wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                }),
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.set_viewport(x as f32, y as f32, width as f32, height as f32, 0.0, 1.0);
        pass.set_scissor_rect(x, y, width, height);
        pass.draw(0..3, 0..1);
    }
}

fn make_pipeline(device: &wgpu::Device, format: wgpu::TextureFormat) -> wgpu::RenderPipeline {
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("DepthCopy"),
        source: wgpu::ShaderSource::Wgsl(WGSL.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("DepthCopy"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("DepthCopy"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("DepthCopy"),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: &module,
            entry_point: Some("vs_main"),
            compilation_options: Default::default(),
            buffers: &[],
        },
        primitive: wgpu::PrimitiveState::default(),
        depth_stencil: Some(wgpu::DepthStencilState {
            format,
            depth_write_enabled: Some(true),
            depth_compare: Some(wgpu::CompareFunction::Always),
            stencil: Default::default(),
            bias: Default::default(),
        }),
        multisample: Default::default(),
        fragment: Some(wgpu::FragmentState {
            module: &module,
            entry_point: Some("fs_main"),
            compilation_options: Default::default(),
            targets: &[],
        }),
        multiview_mask: None,
        cache: None,
    })
}
