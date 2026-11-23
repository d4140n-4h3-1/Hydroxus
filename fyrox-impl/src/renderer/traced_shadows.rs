// Copyright (c) 2019-present Dmitry Stepanov and Fyrox Engine contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Shadows that come from somewhere other than shadow maps.
//!
//! A shadow map is a picture of the scene from the light, drawn every frame for every light that
//! casts shadows, so a scene with many lamps can afford them for only a few. A [`LightShadowTracer`]
//! replaces those pictures: for each light, just before the light is drawn, it produces a mask the
//! size of the screen that says which pixels the light reaches. The light pass reads the mask where
//! it would otherwise look up its shadow map, so the shadow darkens that light's contribution alone.
//!
//! The renderer knows nothing about how the mask is made - ray tracing hardware, typically - and
//! falls back to shadow maps for any light the tracer returns nothing for.

use crate::{
    core::{algebra::Matrix4, math::Rect, pool::Handle},
    graphics::{
        error::FrameworkError, gpu_texture::GpuTexture, server::GraphicsServer, ScissorBox,
    },
    renderer::bundle::LightSource,
    scene::Scene,
};

/// Everything a tracer needs to shadow one light.
pub struct LightShadowTraceContext<'a> {
    /// The graphics server the frame is being drawn with.
    pub server: &'a dyn GraphicsServer,
    /// The scene being drawn.
    pub scene: &'a Scene,
    /// Depth of the G-buffer: where each pixel is.
    pub depth: &'a GpuTexture,
    /// World-space normals of the G-buffer, packed into 0..1.
    pub normals: &'a GpuTexture,
    /// The G-buffer's material texture: metallic, roughness, occlusion, and in alpha one minus
    /// how much light comes through from behind (leaves): such a surface facing away from the
    /// light is lit through, if nothing blocks the light on its other side.
    pub materials: &'a GpuTexture,
    /// Turns a screen position and depth back into a world position.
    pub inv_view_projection: Matrix4<f32>,
    /// The light to shadow.
    pub light: &'a LightSource,
    /// How far the light reaches, in meters, with its scale applied. Infinite for the sun.
    pub light_radius: f32,
    /// The size of the G-buffer.
    pub viewport: Rect<i32>,
    /// The part of the screen the light can reach, with the origin at the bottom left, or
    /// [`None`] for all of it. Pixels outside it are never read.
    pub scissor: Option<ScissorBox>,
}

/// Makes shadow masks for lights in place of shadow maps. See the module docs.
pub trait LightShadowTracer {
    /// Called once per scene and camera before any light is drawn, after the G-buffer is filled:
    /// the place to bring whatever the tracer traces against up to date with the scene.
    fn prepare(
        &mut self,
        server: &dyn GraphicsServer,
        scene_handle: Handle<Scene>,
        scene: &Scene,
    ) -> Result<(), FrameworkError>;

    /// Returns a mask for the light: a single-channel texture the size of the G-buffer, 1 where
    /// the light reaches and 0 where something blocks it. [`None`] leaves the light to shadow maps.
    ///
    /// The light is drawn right after this returns and before the next light is traced, so a
    /// tracer may reuse one texture for every light.
    fn trace(
        &mut self,
        ctx: LightShadowTraceContext,
    ) -> Result<Option<GpuTexture>, FrameworkError>;
}
