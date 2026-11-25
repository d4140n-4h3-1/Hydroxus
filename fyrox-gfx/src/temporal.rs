//! Temporal anti-aliasing.
//!
//! An edge that is thinner than a pixel, or that crosses pixels at a shallow angle, lands on
//! different pixels as the camera moves by less than a pixel, so it crawls and flickers. Filters
//! that look at one frame at a time (FXAA) cannot fix that: they see a different edge every frame.
//! This looks at many frames instead. Every frame the view is shifted by a different fraction of a
//! pixel ([`Renderer::set_projection_jitter`]), so over a few frames every pixel has been sampled
//! at several points across it, and each frame is blended into the frames before it - moved to
//! where the camera is now, using the depth buffer and last frame's camera. The blend settles on
//! the average over the pixel's area, which is a smooth edge that stays put.
//!
//! Anything the history shows that this frame does not - something moved, or came into view - is
//! pulled back to the colors around the pixel this frame, so it does not leave a trail.
//!
//! What the game says moves ([`crate::moving`]) is followed rather than taken to stand still: a
//! point on it is looked up where it was last frame, before it moved, and the view past its edges,
//! where the history may still show it, leans on this frame instead.
//!
//! [`Renderer::set_projection_jitter`]: fyrox::renderer::Renderer::set_projection_jitter

use crate::{
    moving::{MovingThings, MAX_MOVING_THINGS},
    scene_copy::SceneCopy,
};
use fyrox::{
    core::{
        algebra::{Matrix4, Vector2, Vector3, Vector4},
        log::Log,
        pool::Handle,
        sstorage::ImmutableString,
    },
    graphics::{
        error::FrameworkError,
        framebuffer::{Attachment, GpuFrameBuffer},
        gpu_texture::{GpuTextureKind, PixelKind},
        stats::RenderPassStatistics,
    },
    material::shader::Shader,
    renderer::{
        cache::shader::{binding, property, PropertyGroup, RenderMaterial, RenderPassContainer},
        make_viewport_matrix, SceneRenderPass, SceneRenderPassContext,
    },
    scene::{node::Node, Scene},
};
use std::any::TypeId;

const WGSL: &str = include_str!("shaders/taa_wgsl.shader");

/// How far the camera can move in one frame before the history is thrown away, in meters: past
/// this, it was put somewhere else rather than moved there.
const JUMP: f32 = 2.0;

/// How much of each new frame goes into the result, the view standing still (moving, more: see
/// the shader). Lower is smoother and slower to follow changes in lighting. Needles and leaves
/// thinner than a pixel against the sky are a different colour in each frame's shifted view, and
/// what of that each frame brings in shimmers: measured over 64 frames of a still view up into the
/// crowns, the frame changed from frame to frame by 0.0050 (of 1, colours squeezed) at 0.1, 0.0035
/// at this, 0.0030 at 0.03; a wider clamp, or the new frame counting less the more it differs,
/// did little.
const BLEND: f32 = 0.05;

/// The view's shift for frame `frame`, in pixels, each way within half a pixel of the middle.
/// A Halton sequence covers the pixel evenly in any run of consecutive frames.
pub fn jitter(frame: u64) -> Vector2<f32> {
    fn halton(mut index: u64, base: u64) -> f32 {
        let mut fraction = 1.0;
        let mut result = 0.0;
        while index > 0 {
            fraction /= base as f32;
            result += fraction * (index % base) as f32;
            index /= base;
        }
        result
    }
    let index = frame % 8 + 1;
    Vector2::new(halton(index, 2) - 0.5, halton(index, 3) - 0.5)
}

/// The pass. [`crate::GraphicsEffects`] installs it and shifts the view every frame.
pub struct TemporalPass {
    source_type_id: TypeId,
    pass_name: ImmutableString,
    shader: Option<RenderPassContainer>,
    copy: SceneCopy,
    /// The blended frames, twice: one is read while the other is written.
    history: [Option<GpuFrameBuffer>; 2],
    /// Which of `history` holds the latest blend.
    latest: usize,
    size: Vector2<usize>,
    pixel_kind: Option<PixelKind>,
    /// What the history was drawn from: which camera of which scene, its view and projection
    /// without the shift, and where it was.
    previous: Option<(Handle<Scene>, Handle<Node>, Matrix4<f32>, Vector3<f32>)>,
    /// This frame's shift of the view, in pixels, as handed to the renderer.
    jitter: Vector2<f32>,
    moving: MovingThings,
}

impl std::fmt::Debug for TemporalPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TemporalPass")
            .field("size", &self.size)
            .finish()
    }
}

impl TemporalPass {
    pub fn new(source_type_id: TypeId, moving: MovingThings) -> Self {
        Self {
            source_type_id,
            pass_name: ImmutableString::new("Primary"),
            shader: None,
            copy: SceneCopy::new("FyroxGfxTemporalCurrent"),
            history: [None, None],
            latest: 0,
            size: Vector2::new(0, 0),
            pixel_kind: None,
            previous: None,
            jitter: Vector2::zeros(),
            moving,
        }
    }

    /// The shift of the view the renderer was given for the frame about to be drawn. The blend
    /// is lined up with the view without it, so the result does not shake with it.
    pub fn set_jitter(&mut self, jitter: Vector2<f32>) {
        self.jitter = jitter;
    }

    /// Makes the history buffers match the frame; returns whether they had to be made anew.
    fn fit_history(
        &mut self,
        ctx: &SceneRenderPassContext,
        size: Vector2<usize>,
        pixel_kind: PixelKind,
    ) -> Result<bool, FrameworkError> {
        if self.size == size && self.pixel_kind == Some(pixel_kind) && self.history[0].is_some() {
            return Ok(false);
        }
        for (i, slot) in self.history.iter_mut().enumerate() {
            let name = if i == 0 {
                "FyroxGfxTemporalHistory0"
            } else {
                "FyroxGfxTemporalHistory1"
            };
            let texture = ctx
                .server
                .create_2d_render_target(name, pixel_kind, size.x, size.y)?;
            *slot = Some(
                ctx.server
                    .create_frame_buffer(None, vec![Attachment::color(texture)])?,
            );
        }
        self.size = size;
        self.pixel_kind = Some(pixel_kind);
        Ok(true)
    }
}

impl SceneRenderPass for TemporalPass {
    fn on_hdr_render(
        &mut self,
        mut ctx: SceneRenderPassContext,
    ) -> Result<RenderPassStatistics, FrameworkError> {
        if self.shader.is_none() {
            let shader = Shader::from_string(WGSL)
                .map_err(|e| FrameworkError::Custom(format!("temporal shader: {e:?}")))?;
            self.shader = Some(RenderPassContainer::new(ctx.server, &shader)?);
        }

        let Some(frame) = ctx.framebuffer.color_attachments().first() else {
            return Ok(Default::default());
        };
        let GpuTextureKind::Rectangle { width, height } = frame.texture.kind() else {
            return Ok(Default::default());
        };
        let pixel_kind = frame.texture.pixel_kind();
        let size = Vector2::new(width, height);

        let Some(current) = self.copy.take_from(ctx.framebuffer, &mut ctx)? else {
            return Ok(Default::default());
        };
        let resized = self.fit_history(&ctx, size, pixel_kind)?;

        // The camera without this frame's shift - the same shift the renderer applied (see
        // `Observer::jitter`), taken back off.
        let viewport = ctx.observer.viewport.size;
        let shift = Vector2::new(
            2.0 * self.jitter.x / viewport.x.max(1) as f32,
            2.0 * self.jitter.y / viewport.y.max(1) as f32,
        );
        let unshift = Matrix4::new_translation(&Vector3::new(-shift.x, -shift.y, 0.0));
        let view_projection = unshift * ctx.observer.position.view_projection_matrix;
        let position = ctx.observer.position.translation;
        let observer = (ctx.scene_handle, ctx.observer.handle);
        let (previous_view_projection, reset) = match self.previous {
            Some((scene, camera, matrix, was))
                if (scene, camera) == observer && (position - was).norm() < JUMP && !resized =>
            {
                (matrix, false)
            }
            _ => (view_projection, true),
        };
        self.previous = Some((observer.0, observer.1, view_projection, position));

        let (Some(shader), Some(read), Some(write)) = (
            self.shader.as_ref(),
            self.history[self.latest].as_ref(),
            self.history[1 - self.latest].as_ref(),
        ) else {
            return Ok(Default::default());
        };
        let history = read.color_attachments()[0].texture.clone();

        let inverse_view_projection = view_projection
            .try_inverse()
            .unwrap_or_else(Matrix4::identity);
        let screen_size = Vector2::new(width as f32, height as f32);
        // Where a pixel's content went in this frame, as drawn shifted: across as much, and up
        // (so down the texture, which is stored top row first) as much.
        let jitter_uv = Vector2::new(shift.x * 0.5, -shift.y * 0.5);
        let world_view_projection = make_viewport_matrix(ctx.observer.viewport);
        let camera = position.push(1.0);
        // Each moving thing as the two ends of its capsule, the radius riding along with the
        // first, and how far it moved. A radius of zero is no thing at all.
        let mut moving = [[Vector4::zeros(); 3]; MAX_MOVING_THINGS];
        for (slot, thing) in moving.iter_mut().zip(self.moving.get()) {
            *slot = [
                thing.bottom.push(thing.radius),
                thing.top.push(0.0),
                thing.moved.push(0.0),
            ];
        }
        let properties = PropertyGroup::from([
            property("worldViewProjection", &world_view_projection),
            property("inverseViewProjection", &inverse_view_projection),
            property("previousViewProjection", &previous_view_projection),
            property("screenSize", &screen_size),
            property("jitterUv", &jitter_uv),
            property("blend", &BLEND),
            property("reset", &reset),
            property("cameraPosition", &camera),
            property("moving0Bottom", &moving[0][0]),
            property("moving0Top", &moving[0][1]),
            property("moving0Moved", &moving[0][2]),
            property("moving1Bottom", &moving[1][0]),
            property("moving1Top", &moving[1][1]),
            property("moving1Moved", &moving[1][2]),
        ]);
        let material = RenderMaterial::from([
            binding(
                "currentColor",
                (&current, &ctx.renderer_resources.linear_clamp_sampler),
            ),
            binding(
                "historyColor",
                (&history, &ctx.renderer_resources.linear_clamp_sampler),
            ),
            binding(
                "sceneDepth",
                (
                    ctx.depth_texture,
                    &ctx.renderer_resources.nearest_clamp_sampler,
                ),
            ),
            binding("properties", &properties),
        ]);

        let statistics = shader.run_pass(
            1,
            &self.pass_name,
            write,
            &ctx.renderer_resources.quad,
            ctx.observer.viewport,
            &material,
            ctx.uniform_buffer_cache,
            Default::default(),
            None,
        );
        let statistics = match statistics {
            Ok(statistics) => statistics,
            Err(err) => {
                Log::err(format!("Temporal anti-aliasing failed: {err:?}"));
                return Err(err);
            }
        };

        // The blend is the frame now, and the history for the next one.
        write.blit_to(
            ctx.framebuffer,
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
        self.latest = 1 - self.latest;

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
    fn jitter_stays_within_the_pixel_and_covers_it() {
        let offsets: Vec<Vector2<f32>> = (0..8).map(jitter).collect();
        for o in &offsets {
            assert!(o.x.abs() <= 0.5 && o.y.abs() <= 0.5);
        }
        // Every quarter of the pixel gets samples.
        for (sx, sy) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            assert!(offsets.iter().any(|o| o.x * sx > 0.0 && o.y * sy > 0.0));
        }
        assert_eq!(jitter(3), jitter(11), "the pattern repeats every 8 frames");
    }
}
