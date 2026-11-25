//! Blurring what is far off.
//!
//! Detail thinner than a pixel - a line of light on a tower a few hundred meters away - lands on
//! different pixels as the view moves, and breaks up into dashes and dots that crawl. Anti-aliasing
//! smooths it while the view is still, but in fast motion the temporal pass leans on the frame
//! alone (see [`crate::temporal`]), and the crawl comes back. Blurring it instead - more the further
//! off it is, as a camera's far focus would - turns the dashes into a soft glow that holds still.
//!
//! Each pixel is softened by how far off its surface is, from the depth buffer: not at all nearer
//! than [`DistanceBlur::start`], more and more out to [`DistanceBlur::full`], where it is spread
//! over [`DistanceBlur::radius`] of the screen's height. It gathers from a disc round it, and
//! takes a sample only as far out as that sample would itself be spread - so nothing near and
//! sharp bleeds into what is far off round it, nor the far off over its edges.
//!
//! It runs before the temporal pass, which then smooths the softened frame. A [`crate::Switch`]
//! turns it on and off: a game wants it where far detail is thin and bright - a city traced in
//! light - and not everywhere.

use crate::{scene_copy::SceneCopy, Switch};
use fyrox::{
    core::{
        algebra::{Matrix4, Vector2},
        log::Log,
        sstorage::ImmutableString,
    },
    graphics::{error::FrameworkError, gpu_texture::GpuTextureKind, stats::RenderPassStatistics},
    material::shader::Shader,
    renderer::{
        cache::shader::{binding, property, PropertyGroup, RenderMaterial, RenderPassContainer},
        make_viewport_matrix, SceneRenderPass, SceneRenderPassContext,
    },
};
use std::any::TypeId;

const WGSL: &str = include_str!("shaders/distance_blur_wgsl.shader");

/// How far off things start to blur, and how much.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistanceBlur {
    /// Nearer than this, in meters, nothing is blurred.
    pub start: f32,
    /// From here on, in meters, everything is blurred all it is.
    pub full: f32,
    /// How far it is spread there, as a part of the screen's height: 0.004 is three pixels at
    /// 768 rows.
    pub radius: f32,
}

impl Default for DistanceBlur {
    fn default() -> Self {
        Self {
            start: 80.0,
            full: 320.0,
            radius: 0.0035,
        }
    }
}

impl DistanceBlur {
    /// How many pixels a surface `distance` meters off is spread over, on a screen `height`
    /// pixels high. The shader works it out as this does.
    pub fn radius_at(&self, distance: f32, height: f32) -> f32 {
        let span = (self.full - self.start).max(1.0e-3);
        ((distance - self.start) / span).clamp(0.0, 1.0) * self.radius * height
    }
}

pub(crate) struct DistanceBlurPass {
    source_type_id: TypeId,
    pass_name: ImmutableString,
    shader: Option<RenderPassContainer>,
    copy: SceneCopy,
    settings: DistanceBlur,
    switch: Switch,
}

impl std::fmt::Debug for DistanceBlurPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DistanceBlurPass").field("settings", &self.settings).finish()
    }
}

impl DistanceBlurPass {
    pub(crate) fn new(source_type_id: TypeId, settings: DistanceBlur, switch: Switch) -> Self {
        Self {
            source_type_id,
            pass_name: ImmutableString::new("Primary"),
            shader: None,
            copy: SceneCopy::new("FyroxGfxDistanceBlurSource"),
            settings,
            switch,
        }
    }
}

impl SceneRenderPass for DistanceBlurPass {
    fn on_hdr_render(&mut self, mut ctx: SceneRenderPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        if !self.switch.is_enabled() || self.settings.radius <= 0.0 {
            return Ok(Default::default());
        }
        if self.shader.is_none() {
            let shader = Shader::from_string(WGSL)
                .map_err(|e| FrameworkError::Custom(format!("distance blur shader: {e:?}")))?;
            self.shader = Some(RenderPassContainer::new(ctx.server, &shader)?);
        }
        let Some(frame) = ctx.framebuffer.color_attachments().first() else {
            return Ok(Default::default());
        };
        let GpuTextureKind::Rectangle { width, height } = frame.texture.kind() else {
            return Ok(Default::default());
        };
        // What is drawn so far, to read from while the blur is written over it.
        let Some(source) = self.copy.take_from(ctx.framebuffer, &mut ctx)? else {
            return Ok(Default::default());
        };
        let Some(shader) = self.shader.as_ref() else {
            return Ok(Default::default());
        };
        let inverse_view_projection = ctx
            .observer
            .position
            .view_projection_matrix
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        let screen_size = Vector2::new(width as f32, height as f32);
        let camera = ctx.observer.position.translation.push(1.0);
        let settings = self.settings;
        let blur = Vector2::new(settings.start, (settings.full - settings.start).max(1.0e-3));
        let radius = settings.radius * height as f32;
        let world_view_projection = make_viewport_matrix(ctx.observer.viewport);
        let properties = PropertyGroup::from([
            property("worldViewProjection", &world_view_projection),
            property("inverseViewProjection", &inverse_view_projection),
            property("screenSize", &screen_size),
            property("cameraPosition", &camera),
            property("startAndSpan", &blur),
            property("radius", &radius),
        ]);
        let material = RenderMaterial::from([
            binding("sourceColor", (&source, &ctx.renderer_resources.linear_clamp_sampler)),
            binding("sceneDepth", (ctx.depth_texture, &ctx.renderer_resources.nearest_clamp_sampler)),
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
        let statistics = statistics.inspect_err(|err| Log::err(format!("Distance blur failed: {err:?}")))?;
        let mut total = RenderPassStatistics::default();
        total += statistics;
        Ok(total)
    }

    fn source_type_id(&self) -> TypeId {
        self.source_type_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn near_is_sharp_and_far_is_blurred_all_the_way() {
        let blur = DistanceBlur::default();
        assert_eq!(blur.radius_at(10.0, 768.0), 0.0);
        assert_eq!(blur.radius_at(blur.start, 768.0), 0.0);
        let half = blur.radius_at((blur.start + blur.full) / 2.0, 768.0);
        let all = blur.radius_at(blur.full, 768.0);
        assert!(half > 0.0 && half < all);
        assert!((all - blur.radius * 768.0).abs() < 1.0e-4);
        assert_eq!(blur.radius_at(5000.0, 768.0), all, "no more than all");
    }
}
