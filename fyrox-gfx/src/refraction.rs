//! Refractive glass.
//!
//! The engine draws transparent surfaces by blending them over the frame, which can tint what is
//! behind them but never bend it. This effect draws glass after the rest of the scene instead:
//!
//! 1. Once the engine's own lighting and forward passes are done, the frame is copied into a
//!    texture ([`scene_color`]).
//! 2. Every surface with the glass shader is drawn in a pass of its own, `Refraction`. The shader
//!    works out where the refracted view ray leaves the glass, projects that onto the screen, and
//!    reads the copied frame there - so whatever is behind the glass shows through, displaced the
//!    way a real pane or lens would displace it. It also tints what it lets through and adds a
//!    Fresnel reflection and highlights from nearby lights.
//!
//! Glass takes no part in the G-buffer or in shadow maps: light passes through it, and what is
//! behind it is lit as usual. Glass seen through other glass is not visible, because the copy is
//! taken before any glass is drawn.
//!
//! Not only glass: any material whose shader has a `Refraction` pass is drawn in it, after the
//! copy - water, say, bending the view of its bed. Such a shader binds the copy as `sceneColor`
//! itself ([`scene_color`]).

use fyrox::{
    asset::untyped::ResourceKind,
    core::{algebra::Vector2, color::Color, log::Log, pool::Handle, sstorage::ImmutableString},
    graph::SceneGraph,
    graphics::{
        error::FrameworkError,
        framebuffer::{Attachment, GpuFrameBuffer},
        gpu_texture::GpuTextureKind,
        stats::RenderPassStatistics,
    },
    material::{
        shader::{Shader, ShaderResource, ShaderResourceExtension},
        Material, MaterialResource,
    },
    renderer::{bundle::BundleRenderContext, SceneRenderPass, SceneRenderPassContext},
    resource::texture::{
        TextureMagnificationFilter, TextureMinificationFilter, TextureResource,
        TextureResourceExtension, TextureWrapMode,
    },
    scene::{graph::Graph, mesh::Mesh, node::Node},
};
use std::{any::TypeId, sync::LazyLock};

/// Name of the render pass glass shaders are drawn in.
pub const REFRACTION_PASS_NAME: &str = "Refraction";

/// Shader source for engines built with wgpu (Vulkan).
const GLASS_WGSL: &str = include_str!("shaders/glass_wgsl.shader");
/// Shader source for engines built with OpenGL.
const GLASS_GLSL: &str = include_str!("shaders/glass_glsl.shader");

static GLASS_SHADER: LazyLock<ShaderResource> = LazyLock::new(|| {
    let source = if engine_uses_wgsl() {
        GLASS_WGSL
    } else {
        GLASS_GLSL
    };
    ShaderResource::new_ok(
        fyrox::core::uuid::uuid!("6a4c1d2e-93b7-4f58-8e0b-5d2f7c1a9e34"),
        ResourceKind::Embedded,
        Shader::from_string(source).expect("the built-in glass shader is valid"),
    )
});

static SCENE_COLOR: LazyLock<TextureResource> = LazyLock::new(|| {
    let texture = TextureResource::new_render_target(1, 1);
    {
        let mut data = texture.data_ref();
        data.set_s_wrap_mode(TextureWrapMode::ClampToEdge);
        data.set_t_wrap_mode(TextureWrapMode::ClampToEdge);
        data.set_minification_filter(TextureMinificationFilter::Linear);
        data.set_magnification_filter(TextureMagnificationFilter::Linear);
    }
    texture
});

/// Whether the engine was built with the wgpu backend. There is no runtime flag for it, but the
/// engine's own standard shader is written in the language the backend compiles, so it tells.
pub(crate) fn engine_uses_wgsl() -> bool {
    let standard = ShaderResource::standard();
    let state = standard.data_ref();
    state
        .definition
        .passes
        .first()
        .is_some_and(|pass| pass.vertex_shader.contains("@vertex"))
}

/// The shader behind every glass material.
pub fn glass_shader() -> ShaderResource {
    GLASS_SHADER.clone()
}

/// The texture glass reads the scene behind it from. Glass materials must have it bound as
/// `sceneColor`; [`GlassMaterial`] does that.
pub fn scene_color() -> TextureResource {
    SCENE_COLOR.clone()
}

/// Builds glass materials. The defaults are clear window glass.
#[derive(Debug, Clone, PartialEq)]
pub struct GlassMaterial {
    /// Color of the glass, in sRGB.
    pub tint: Color,
    /// How strongly the glass colors what is behind it, from 0 (not at all) to 1.
    pub tint_strength: f32,
    /// Index of refraction: 1.0 is air (no bending), 1.33 water, 1.5 window glass.
    pub index_of_refraction: f32,
    /// How far behind the surface the refracted image is taken from, in meters. Larger values
    /// displace more, as thicker glass would.
    pub distortion: f32,
    /// How much the surface ripples, from 0 (flat) to about 0.3 (strongly warped).
    pub waviness: f32,
    /// How tightly packed the ripples are, per meter.
    pub wave_scale: f32,
    /// Frosting: how far apart, in pixels, the blur samples are.
    pub blur: f32,
    /// Strength of the highlights from lights.
    pub specular_strength: f32,
    /// Sharpness of the highlights.
    pub shininess: f32,
    /// How much the glass reflects at grazing angles, from 0 to 1.
    pub reflectivity: f32,
    /// Light the glass gives off by itself, in sRGB - a lit lamp behind a pane, for instance.
    pub emission: Color,
    /// How brightly [`Self::emission`] shines. 0 turns it off.
    pub emission_strength: f32,
    /// Optional normal map for patterned glass.
    pub normal_map: Option<TextureResource>,
    /// Texture coordinate scale for the normal map.
    pub tex_coord_scale: Vector2<f32>,
    /// Whether light shone through it takes on its colour, where shadows are traced: as much of
    /// it as [`Self::tint_strength`] says. Otherwise light goes through it as if it were not
    /// there.
    pub tints_light: bool,
}

impl Default for GlassMaterial {
    fn default() -> Self {
        Self {
            tint: Color::WHITE,
            tint_strength: 0.35,
            index_of_refraction: 1.5,
            distortion: 0.3,
            waviness: 0.0,
            wave_scale: 4.0,
            blur: 0.0,
            specular_strength: 1.0,
            shininess: 96.0,
            reflectivity: 0.6,
            emission: Color::BLACK,
            emission_strength: 0.0,
            normal_map: None,
            tex_coord_scale: Vector2::new(1.0, 1.0),
            tints_light: true,
        }
    }
}

impl GlassMaterial {
    /// Glass of the given color.
    pub fn tinted(tint: Color) -> Self {
        Self {
            tint,
            ..Default::default()
        }
    }

    /// Glass that glows, for a lamp's cover or a screen. The glow is added on top of what the
    /// glass lets through; it lights nothing else by itself, so put a light behind it as well.
    pub fn lit(tint: Color, emission: Color, strength: f32) -> Self {
        Self {
            tint,
            emission,
            emission_strength: strength,
            ..Default::default()
        }
    }

    pub fn build(&self) -> Material {
        let mut material = Material::from_shader(glass_shader());
        material.bind("sceneColor", scene_color());
        if let Some(normal_map) = self.normal_map.clone() {
            material.bind("normalTexture", normal_map);
        }
        material.set_property("tint", self.tint);
        material.set_property("tintStrength", self.tint_strength);
        material.set_property("indexOfRefraction", self.index_of_refraction);
        material.set_property("distortion", self.distortion);
        material.set_property("waviness", self.waviness);
        material.set_property("waveScale", self.wave_scale);
        material.set_property("blur", self.blur);
        material.set_property("specularStrength", self.specular_strength);
        material.set_property("shininess", self.shininess);
        material.set_property("reflectivity", self.reflectivity);
        material.set_property("emission", self.emission);
        material.set_property("emissionStrength", self.emission_strength);
        material.set_property("texCoordScale", self.tex_coord_scale);
        material.set_property("tintsLight", if self.tints_light { 1.0f32 } else { 0.0 });
        material
    }

    pub fn build_resource(&self) -> MaterialResource {
        MaterialResource::new_embedded(self.build())
    }
}

/// Puts `replacement` on every surface under `root` whose material `is_target` picks. Returns how
/// many surfaces were changed. Handy for turning the glass of an imported model into real glass.
pub fn replace_materials(
    graph: &mut Graph,
    root: Handle<Node>,
    mut is_target: impl FnMut(&Material) -> bool,
    replacement: &MaterialResource,
) -> usize {
    let handles: Vec<Handle<Node>> = graph.traverse_handle_iter(root).collect();
    let mut replaced = 0;
    for handle in handles {
        let Some(mesh) = graph[handle].cast_mut::<Mesh>() else {
            continue;
        };
        for surface in mesh.surfaces_mut() {
            let matches = {
                let material = surface.material();
                let state = material.state();
                state.data_ref().is_some_and(&mut is_target)
            };
            if matches {
                surface.set_material(replacement.clone());
                replaced += 1;
            }
        }
    }
    replaced
}

/// Whether `shader` is drawn in the refraction pass: whether it has a pass of that name (the glass
/// shader does).
pub fn refracts(shader: &ShaderResource) -> bool {
    if shader == &*GLASS_SHADER {
        return true;
    }
    let state = shader.state();
    state
        .data_ref()
        .is_some_and(|shader| shader.definition.passes.iter().any(|pass| pass.name == REFRACTION_PASS_NAME))
}

/// The render pass that draws glass, and whatever else refracts. [`crate::GraphicsEffects`]
/// installs it.
pub struct RefractionPass {
    source_type_id: TypeId,
    pass_name: ImmutableString,
    /// The copy of the frame and the frame buffer used to fill it.
    copy: Option<GpuFrameBuffer>,
}

impl std::fmt::Debug for RefractionPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RefractionPass")
            .field("has_copy", &self.copy.is_some())
            .finish()
    }
}

impl RefractionPass {
    pub fn new(source_type_id: TypeId) -> Self {
        Self {
            source_type_id,
            pass_name: ImmutableString::new(REFRACTION_PASS_NAME),
            copy: None,
        }
    }

    /// Returns a frame buffer whose texture matches the frame, creating it when the frame's size
    /// or format changed.
    fn copy_target(
        &mut self,
        ctx: &mut SceneRenderPassContext,
    ) -> Result<Option<GpuFrameBuffer>, FrameworkError> {
        let Some(frame) = ctx.framebuffer.color_attachments().first() else {
            return Ok(None);
        };
        let GpuTextureKind::Rectangle { width, height } = frame.texture.kind() else {
            return Ok(None);
        };
        let pixel_kind = frame.texture.pixel_kind();

        let up_to_date = self.copy.as_ref().is_some_and(|copy| {
            copy.color_attachments().first().is_some_and(|attachment| {
                matches!(
                    attachment.texture.kind(),
                    GpuTextureKind::Rectangle { width: w, height: h } if w == width && h == height
                ) && attachment.texture.pixel_kind() == pixel_kind
            })
        });
        if !up_to_date {
            let texture = ctx.server.create_2d_render_target(
                "FyroxGfxSceneColor",
                pixel_kind,
                width,
                height,
            )?;
            // The old texture is still registered for the scene color resource; replace it.
            ctx.texture_cache.unload(&SCENE_COLOR);
            self.copy = Some(
                ctx.server
                    .create_frame_buffer(None, vec![Attachment::color(texture)])?,
            );
        }
        Ok(self.copy.clone())
    }
}

impl SceneRenderPass for RefractionPass {
    fn on_hdr_render(
        &mut self,
        mut ctx: SceneRenderPassContext,
    ) -> Result<RenderPassStatistics, FrameworkError> {
        let is_glass = |bundle: &fyrox::renderer::bundle::RenderDataBundle| {
            let state = bundle.material.state();
            state
                .data_ref()
                .is_some_and(|material| refracts(material.shader()))
        };
        if !ctx.bundle_storage.bundles.iter().any(is_glass) {
            return Ok(Default::default());
        }

        let Some(copy) = self.copy_target(&mut ctx)? else {
            return Ok(Default::default());
        };
        let texture = copy.color_attachments()[0].texture.clone();
        let GpuTextureKind::Rectangle { width, height } = texture.kind() else {
            return Ok(Default::default());
        };
        ctx.texture_cache
            .try_register(ctx.server, &SCENE_COLOR, texture)?;
        ctx.framebuffer.blit_to(
            &copy,
            0,
            0,
            width as i32,
            height as i32,
            0,
            0,
            width as i32,
            height as i32,
            true,
            false,
            false,
        );

        let statistics = ctx.bundle_storage.render_to_frame_buffer(
            ctx.server,
            ctx.geometry_cache,
            ctx.shader_cache,
            is_glass,
            |_| true,
            BundleRenderContext {
                texture_cache: ctx.texture_cache,
                render_pass_name: &self.pass_name,
                frame_buffer: ctx.framebuffer,
                viewport: ctx.observer.viewport,
                uniform_memory_allocator: ctx.uniform_memory_allocator,
                resource_manager: ctx.resource_manager,
                use_pom: false,
                light_position: &Default::default(),
                ambient_light: ctx.scene.rendering_options.ambient_lighting_color,
                scene_depth: Some(ctx.depth_texture),
                // The glass reads its own copy of the frame (`sceneColor`), taken after the
                // forward pass; the renderer's opaque copy is for forward materials.
                scene_color: None,
                renderer_resources: ctx.renderer_resources,
            },
        );
        if let Err(err) = &statistics {
            Log::err(format!("Refraction pass failed: {err:?}"));
        }
        statistics
    }

    fn source_type_id(&self) -> TypeId {
        self.source_type_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(source: &str) {
        let shader = Shader::from_string(source).expect("shader parses");
        let definition = &shader.definition;
        assert!(definition
            .passes
            .iter()
            .any(|pass| pass.name == REFRACTION_PASS_NAME));
        for name in [
            "GBuffer",
            "Forward",
            "DirectionalShadow",
            "PointShadow",
            "SpotShadow",
        ] {
            assert!(
                definition.disabled_passes.iter().any(|p| p == name),
                "{name}"
            );
        }
        assert!(definition
            .resources
            .iter()
            .any(|r| r.name.as_str() == "sceneColor"));
    }

    #[test]
    fn wgsl_shader_is_valid() {
        check(GLASS_WGSL);
    }

    #[test]
    fn glsl_shader_is_valid() {
        check(GLASS_GLSL);
    }

    #[test]
    fn the_engine_backend_is_detected() {
        // Tests build the engine with wgpu (see the dev-dependencies).
        assert!(engine_uses_wgsl());
    }

    #[test]
    fn lit_glass_carries_its_glow() {
        let material = GlassMaterial::lit(Color::WHITE, Color::opaque(255, 200, 150), 2.0).build();
        let key = ImmutableString::new("properties");
        let Some(fyrox::material::MaterialResourceBinding::PropertyGroup(group)) =
            material.bindings().get(&key)
        else {
            panic!("no properties");
        };
        assert!(matches!(
            group.property_ref("emissionStrength"),
            Some(fyrox::material::MaterialProperty::Float(strength)) if *strength == 2.0
        ));
    }

    #[test]
    fn any_shader_with_a_refraction_pass_refracts() {
        assert!(refracts(&glass_shader()));
        // A shader of its own with a refraction pass, as water's.
        let water = GLASS_WGSL.replace("FyroxGfxGlass", "Water");
        let water = ShaderResource::new_ok(
            fyrox::core::uuid::Uuid::new_v4(),
            ResourceKind::Embedded,
            Shader::from_string(&water).expect("parses"),
        );
        assert!(refracts(&water));
        assert!(!refracts(&ShaderResource::standard()));
    }

    #[test]
    fn glass_materials_read_the_scene_copy() {
        let material = GlassMaterial::tinted(Color::opaque(170, 90, 255)).build();
        assert_eq!(material.shader(), &glass_shader());
        assert!(material
            .bindings()
            .contains_key(&ImmutableString::new("sceneColor")));
    }
}
