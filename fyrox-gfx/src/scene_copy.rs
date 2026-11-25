//! A copy of the frame as it stands part-way through drawing it.
//!
//! Effects that read the scene while adding to it - glass, reflections - cannot read the frame
//! buffer they are drawing into, so they work from a copy taken first. The copy matches the frame
//! in size and format, and is made again whenever the frame changes shape.

use fyrox::{
    core::algebra::Vector2,
    graphics::{
        error::FrameworkError,
        framebuffer::{Attachment, GpuFrameBuffer},
        gpu_texture::{GpuTexture, GpuTextureKind},
    },
    renderer::SceneRenderPassContext,
};

pub struct SceneCopy {
    name: &'static str,
    framebuffer: Option<GpuFrameBuffer>,
    size: Vector2<usize>,
}

impl std::fmt::Debug for SceneCopy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SceneCopy")
            .field("name", &self.name)
            .field("size", &self.size)
            .finish()
    }
}

impl SceneCopy {
    pub fn new(name: &'static str) -> Self {
        Self {
            name,
            framebuffer: None,
            size: Vector2::new(0, 0),
        }
    }

    /// The size of the last copy, in pixels.
    pub fn size(&self) -> Vector2<usize> {
        self.size
    }

    /// The texture holding the copy, if one has been made.
    pub fn texture(&self) -> Option<GpuTexture> {
        self.framebuffer
            .as_ref()
            .and_then(|framebuffer| framebuffer.color_attachments().first())
            .map(|attachment| attachment.texture.clone())
    }

    /// Copies `framebuffer`'s color into this one, and returns the texture holding it.
    pub fn take_from(
        &mut self,
        framebuffer: &GpuFrameBuffer,
        ctx: &mut SceneRenderPassContext,
    ) -> Result<Option<GpuTexture>, FrameworkError> {
        let Some(frame) = framebuffer.color_attachments().first() else {
            return Ok(None);
        };
        let GpuTextureKind::Rectangle { width, height } = frame.texture.kind() else {
            return Ok(None);
        };
        let pixel_kind = frame.texture.pixel_kind();

        let up_to_date = self.size == Vector2::new(width, height)
            && self.framebuffer.as_ref().is_some_and(|copy| {
                copy.color_attachments()
                    .first()
                    .is_some_and(|attachment| attachment.texture.pixel_kind() == pixel_kind)
            });
        if !up_to_date {
            let texture = ctx
                .server
                .create_2d_render_target(self.name, pixel_kind, width, height)?;
            self.framebuffer = Some(
                ctx.server
                    .create_frame_buffer(None, vec![Attachment::color(texture)])?,
            );
            self.size = Vector2::new(width, height);
        }

        let Some(copy) = self.framebuffer.as_ref() else {
            return Ok(None);
        };
        framebuffer.blit_to(
            copy,
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
        Ok(self.texture())
    }
}
