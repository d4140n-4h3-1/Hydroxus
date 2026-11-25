//! Ray-marched reflections.
//!
//! For every pixel of a surface that reflects, a ray is sent from that point in the direction the
//! view bounces off it, and followed through the scene a step at a time. Where the ray passes
//! behind something, that something is what the surface reflects, and its color is mixed in.
//!
//! The rays are traced against the depth buffer - the scene as the camera sees it - rather than
//! against the geometry itself, so what a reflection can show is what is on screen. A reflection
//! of something behind the camera, or hidden behind a wall, has nothing to be traced against; such
//! rays find nothing and those pixels are left alone, and reflections fade out towards the edges
//! of the screen where this starts to show. Tracing against the geometry instead needs ray tracing
//! hardware, which the engine's renderer does not set up (see the crate docs).
//!
//! Only surfaces facing upwards reflect by default ([`Reflections::min_upwards`]): a floor's
//! reflection is mostly on screen, a wall's mostly is not.
//!
//! The ray is followed in steps, each pixel starting at its own point within the first so the
//! steps do not show as bands, and where it goes behind something the gap is halved until it is
//! found where it went behind - a sharp, unbroken reflection, even of thin things. A ray that only
//! passed behind something nearer the camera goes on past it; one that hits the back of something
//! finds nothing, since nothing reflected can be its back.
//!
//! How much a surface reflects, and how sharply, is its material's: smooth surfaces reflect like
//! mirrors, rougher ones blur what they reflect - each pixel's ray goes off a little to the side
//! of the mirror's, a different way every frame, which temporal anti-aliasing averages - and fade
//! out altogether towards [`Reflections::max_roughness`]. Metals reflect more
//! ([`Reflections::metal_strength`]), and colour what they reflect with their own colour.

use crate::scene_copy::SceneCopy;
use fyrox::{
    core::{
        algebra::{Matrix4, Vector2},
        log::Log,
        sstorage::ImmutableString,
    },
    graphics::{error::FrameworkError, stats::RenderPassStatistics},
    material::shader::Shader,
    renderer::{
        cache::shader::{binding, property, PropertyGroup, RenderMaterial, RenderPassContainer},
        make_viewport_matrix, SceneRenderPass, SceneRenderPassContext,
    },
};
use std::any::TypeId;

/// Shader source for engines built with wgpu (Vulkan).
const WGSL: &str = include_str!("shaders/reflections_wgsl.shader");
/// Shader source for engines built with OpenGL.
const GLSL: &str = include_str!("shaders/reflections_glsl.shader");

/// How reflections are traced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Reflections {
    /// How far a reflected ray travels before giving up, in meters.
    pub reach: f32,
    /// How many steps it takes to get there. More steps cost more and miss less; a reflection that
    /// breaks into stripes wants either more steps or a shorter reach.
    pub steps: i32,
    /// How far behind a surface a ray may pass and still count as having hit it, in meters. Too
    /// small and reflections come out patchy, too large and they smear.
    pub thickness: f32,
    /// How much of the reflection is mixed into the surface, from 0 to 1.
    pub strength: f32,
    /// How far a surface must face upwards to reflect at all: 1 is straight up, 0 is anything.
    pub min_upwards: f32,
    /// How much of the reflection a polished metal shows, from 0 to 1, in place of
    /// [`Self::strength`].
    pub metal_strength: f32,
    /// How rough a surface can be and still reflect, from 0 to 1: reflections fade out between
    /// halfway there and this.
    pub max_roughness: f32,
}

impl Default for Reflections {
    fn default() -> Self {
        Self {
            reach: 12.0,
            steps: 24,
            thickness: 0.4,
            strength: 0.35,
            min_upwards: 0.7,
            metal_strength: 0.8,
            max_roughness: 0.9,
        }
    }
}

/// The render pass that traces them. [`crate::GraphicsEffects`] installs it.
pub struct ReflectionPass {
    source_type_id: TypeId,
    settings: Reflections,
    pass_name: ImmutableString,
    shader: Option<RenderPassContainer>,
    copy: SceneCopy,
    /// Whether to reflect at all.
    switch: crate::Switch,
    /// Frames drawn, for the noise that spreads the rays to change every frame.
    frame: u32,
}

impl std::fmt::Debug for ReflectionPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ReflectionPass")
            .field("settings", &self.settings)
            .finish()
    }
}

impl ReflectionPass {
    pub fn new(source_type_id: TypeId, settings: Reflections, switch: crate::Switch) -> Self {
        Self {
            source_type_id,
            settings,
            pass_name: ImmutableString::new("Primary"),
            shader: None,
            copy: SceneCopy::new("FyroxGfxReflectionSource"),
            switch,
            frame: 0,
        }
    }
}

impl SceneRenderPass for ReflectionPass {
    fn on_hdr_render(
        &mut self,
        mut ctx: SceneRenderPassContext,
    ) -> Result<RenderPassStatistics, FrameworkError> {
        if self.settings.strength <= 0.0 || !self.switch.is_enabled() {
            return Ok(Default::default());
        }

        if self.shader.is_none() {
            let source = if crate::refraction::engine_uses_wgsl() {
                WGSL
            } else {
                GLSL
            };
            let shader = Shader::from_string(source)
                .map_err(|e| FrameworkError::Custom(format!("reflection shader: {e:?}")))?;
            self.shader = Some(RenderPassContainer::new(ctx.server, &shader)?);
        }
        let Some(shader) = self.shader.as_ref() else {
            return Ok(Default::default());
        };

        // The scene as it stands, to trace against and to read the reflected color from.
        let Some(source) = self.copy.take_from(ctx.framebuffer, &mut ctx)? else {
            return Ok(Default::default());
        };
        let size = Vector2::new(self.copy.size().x as f32, self.copy.size().y as f32);

        let view_projection = ctx.observer.position.view_projection_matrix;
        let inverse_view_projection = view_projection
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        let camera_position = ctx.observer.position.translation;
        let world_view_projection = make_viewport_matrix(ctx.observer.viewport);
        // The noise repeats after a while, which nobody sees, and stays exact as a float.
        self.frame = (self.frame + 1) % 1024;
        let frame = self.frame as f32;

        let properties = PropertyGroup::from([
            property("worldViewProjection", &world_view_projection),
            property("viewProjection", &view_projection),
            property("inverseViewProjection", &inverse_view_projection),
            property("cameraPosition", &camera_position),
            property("screenSize", &size),
            property("reach", &self.settings.reach),
            property("steps", &self.settings.steps),
            property("thickness", &self.settings.thickness),
            property("strength", &self.settings.strength),
            property("minUpwards", &self.settings.min_upwards),
            property("metalStrength", &self.settings.metal_strength),
            property("maxRoughness", &self.settings.max_roughness),
            property("frame", &frame),
        ]);
        let material = RenderMaterial::from([
            binding(
                "sceneColor",
                (&source, &ctx.renderer_resources.linear_clamp_sampler),
            ),
            binding(
                "sceneDepth",
                (
                    ctx.depth_texture,
                    &ctx.renderer_resources.nearest_clamp_sampler,
                ),
            ),
            binding(
                "sceneNormal",
                (
                    ctx.normal_texture,
                    &ctx.renderer_resources.nearest_clamp_sampler,
                ),
            ),
            binding(
                "sceneMaterial",
                (
                    ctx.material_texture,
                    &ctx.renderer_resources.nearest_clamp_sampler,
                ),
            ),
            binding(
                "sceneDiffuse",
                (
                    ctx.diffuse_texture,
                    &ctx.renderer_resources.nearest_clamp_sampler,
                ),
            ),
            binding("properties", &properties),
        ]);

        let statistics = shader.run_pass(
            1,
            &self.pass_name,
            ctx.framebuffer,
            &ctx.renderer_resources.quad,
            ctx.observer.viewport,
            &material,
            ctx.uniform_buffer_cache,
            Default::default(),
            None,
        );
        match statistics {
            Ok(statistics) => {
                let mut total = RenderPassStatistics::default();
                total += statistics;
                Ok(total)
            }
            Err(err) => {
                Log::err(format!("Reflection pass failed: {err:?}"));
                Err(err)
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

    fn check(source: &str) {
        let shader = Shader::from_string(source).expect("shader parses");
        assert!(shader.definition.passes.iter().any(|p| p.name == "Primary"));
        for name in ["sceneColor", "sceneDepth", "sceneNormal", "sceneMaterial", "sceneDiffuse"] {
            assert!(
                shader
                    .definition
                    .resources
                    .iter()
                    .any(|r| r.name.as_str() == name),
                "{name}"
            );
        }
    }

    #[test]
    fn wgsl_shader_is_valid() {
        check(WGSL);
    }

    #[test]
    fn glsl_shader_is_valid() {
        check(GLSL);
    }

    #[test]
    fn reflections_are_off_when_their_strength_is_zero() {
        // The pass returns early rather than copying the frame for nothing.
        let settings = Reflections {
            strength: 0.0,
            ..Default::default()
        };
        assert_eq!(settings.strength, 0.0);
        assert!(Reflections::default().strength > 0.0);
    }
}
