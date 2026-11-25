//! Area lights: rectangles that glow - a lit panel, the rim of a screen - and light what is round
//! them from all of their surface, where the renderer's own lights shine from a point.
//!
//! The game says which lights there are, every frame or whenever they change, through the list
//! [`GraphicsEffects::area_lights`](crate::GraphicsEffects::area_lights) hands out. After the
//! scene is lit, their light is added to it, worked out per pixel by sampling points over each
//! rectangle. Where shadows are traced ([`crate::raytraced_shadows`]), each point is traced too,
//! against the same scene: the lights cast soft shadows, and glass colours what they shine
//! through. Elsewhere - WebGL among them - they shine through everything, which is right in the
//! open and gives itself away only behind walls; keep their reach short there.

use crate::raytraced_shadows::SharedScene;
use fyrox::{
    core::{
        algebra::{Vector2, Vector3},
        color::Color,
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
use fyrox_graphics_wgpu::{
    area_lights::{self, AreaLightParameters, AreaLighter},
    server::WgpuGraphicsServer,
};
use std::{any::TypeId, cell::RefCell, rc::Rc};

pub use fyrox_graphics_wgpu::area_lights::MAX_AREA_LIGHTS;

/// Adds the lights' light to the frame. Drawn by the renderer, like everything else drawn into
/// the frame: WebGL loses track of the frame's own lights when it is drawn into another way.
const WGSL: &str = include_str!("shaders/area_lights_wgsl.shader");

/// A glowing rectangle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AreaLight {
    /// One corner, in world space.
    pub corner: Vector3<f32>,
    /// The two edges from that corner. It shines from the side their cross product points to,
    /// or from both if [`Self::two_sided`].
    pub edges: [Vector3<f32>; 2],
    pub colour: Color,
    /// How bright it is: what a pixel facing a square meter of it from a meter away gets, as a
    /// share of its colour.
    pub intensity: f32,
    /// How far from its middle it lights anything, in meters.
    pub reach: f32,
    pub two_sided: bool,
    /// How many points along each edge it is sampled at, 1 to 16: several along a long edge, one
    /// across a thin strip.
    pub samples: [u32; 2],
}

impl AreaLight {
    /// A rectangle from `corner` along `edges`, sampled at points about `spacing` meters apart.
    pub fn new(corner: Vector3<f32>, edges: [Vector3<f32>; 2], spacing: f32) -> Self {
        let along = |edge: &Vector3<f32>| ((edge.norm() / spacing.max(1.0e-3)).ceil() as u32).clamp(1, 16);
        Self {
            corner,
            edges,
            colour: Color::WHITE,
            intensity: 1.0,
            reach: 4.0,
            two_sided: false,
            samples: [along(&edges[0]), along(&edges[1])],
        }
    }

    pub fn with_colour(mut self, colour: Color, intensity: f32) -> Self {
        self.colour = colour;
        self.intensity = intensity;
        self
    }

    pub fn with_reach(mut self, reach: f32) -> Self {
        self.reach = reach;
        self
    }

    pub fn two_sided(mut self) -> Self {
        self.two_sided = true;
        self
    }

    fn to_backend(self) -> area_lights::AreaLight {
        let v = |v: Vector3<f32>| [v.x, v.y, v.z];
        let c = self.colour.srgb_to_linear_f32();
        area_lights::AreaLight {
            corner: v(self.corner),
            edges: [v(self.edges[0]), v(self.edges[1])],
            colour: [
                c.x * self.intensity,
                c.y * self.intensity,
                c.z * self.intensity,
            ],
            reach: self.reach,
            two_sided: self.two_sided,
            samples: self.samples,
        }
    }
}

/// The area lights to light the scene with, shared between the game and the effects.
#[derive(Debug, Clone, Default)]
pub struct AreaLights(Rc<RefCell<Vec<AreaLight>>>);

impl AreaLights {
    /// The lights from now on, in place of the ones before. Only the first
    /// [`MAX_AREA_LIGHTS`] are lit, so put the ones that matter most - nearest the camera -
    /// first.
    pub fn set(&self, lights: impl IntoIterator<Item = AreaLight>) {
        let mut list = self.0.borrow_mut();
        list.clear();
        list.extend(lights.into_iter().take(MAX_AREA_LIGHTS));
    }

    pub(crate) fn get(&self) -> Vec<AreaLight> {
        self.0.borrow().clone()
    }
}

impl PartialEq for AreaLights {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

/// Adds the area lights' light to the lit scene.
pub(crate) struct AreaLightPass {
    source_type_id: TypeId,
    lights: AreaLights,
    scene: SharedScene,
    lighter: Option<AreaLighter>,
    shader: Option<RenderPassContainer>,
    pass_name: ImmutableString,
    failed: bool,
}

impl std::fmt::Debug for AreaLightPass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AreaLightPass")
            .field("lights", &self.lights)
            .field("failed", &self.failed)
            .finish()
    }
}

impl AreaLightPass {
    pub fn new(source_type_id: TypeId, lights: AreaLights, scene: SharedScene) -> Self {
        Self {
            source_type_id,
            lights,
            scene,
            lighter: None,
            shader: None,
            pass_name: ImmutableString::new("Primary"),
            failed: false,
        }
    }
}

impl SceneRenderPass for AreaLightPass {
    fn on_hdr_render(
        &mut self,
        ctx: SceneRenderPassContext,
    ) -> Result<RenderPassStatistics, FrameworkError> {
        let lights = self.lights.get();
        if lights.is_empty() {
            return Ok(Default::default());
        }
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(Default::default());
        };
        if self.shader.is_none() {
            let shader = Shader::from_string(WGSL)
                .map_err(|e| FrameworkError::Custom(format!("area light shader: {e:?}")))?;
            self.shader = Some(RenderPassContainer::new(ctx.server, &shader)?);
        }
        let Some(shader) = self.shader.as_ref() else {
            return Ok(Default::default());
        };
        let lighter = self
            .lighter
            .get_or_insert_with(|| server.create_area_lighter());
        let inverse_view_projection = ctx
            .observer
            .position
            .view_projection_matrix
            .try_inverse()
            .unwrap_or_default();
        let mut matrix = [[0.0; 4]; 4];
        for (column, values) in inverse_view_projection.column_iter().zip(matrix.iter_mut()) {
            values.copy_from_slice(column.as_slice());
        }
        let lights = lights
            .into_iter()
            .map(AreaLight::to_backend)
            .collect::<Vec<_>>();
        let scene = self.scene.borrow();
        let light = lighter.light(
            server,
            scene.as_ref(),
            ctx.depth_texture,
            ctx.normal_texture,
            ctx.diffuse_texture,
            &lights,
            AreaLightParameters {
                inverse_view_projection: matrix,
                bias: 0.02,
                // This backend stores render targets top row first.
                flip_v: true,
            },
        );
        let light = match light {
            Ok(Some(light)) => light,
            Ok(None) => return Ok(Default::default()),
            Err(err) => {
                if !self.failed {
                    Log::err(format!("Area lights failed: {err:?}"));
                    self.failed = true;
                }
                return Ok(Default::default());
            }
        };

        let viewport = ctx.observer.viewport;
        // The light's texture is the frame's size.
        let size = match light.kind() {
            GpuTextureKind::Rectangle { width, height } => Vector2::new(width as f32, height as f32),
            _ => Vector2::new(viewport.w() as f32, viewport.h() as f32),
        };
        let world_view_projection = make_viewport_matrix(viewport);
        let properties = PropertyGroup::from([
            property("worldViewProjection", &world_view_projection),
            property("screenSize", &size),
        ]);
        let material = RenderMaterial::from([
            binding(
                "areaLight",
                (&light, &ctx.renderer_resources.nearest_clamp_sampler),
            ),
            binding("properties", &properties),
        ]);
        shader.run_pass(
            1,
            &self.pass_name,
            ctx.framebuffer,
            &ctx.renderer_resources.quad,
            viewport,
            &material,
            ctx.uniform_buffer_cache,
            Default::default(),
            None,
        )
        .map(|statistics| {
            let mut total = RenderPassStatistics::default();
            total += statistics;
            total
        })
    }

    fn source_type_id(&self) -> TypeId {
        self.source_type_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_follow_the_edges() {
        let light = AreaLight::new(
            Vector3::zeros(),
            [Vector3::x() * 1.0, Vector3::y() * 0.02],
            0.25,
        );
        assert_eq!(light.samples, [4, 1]);
    }

    #[test]
    fn a_copy_sees_what_the_game_sets() {
        let game = AreaLights::default();
        let effects = game.clone();
        let light = AreaLight::new(Vector3::zeros(), [Vector3::x(), Vector3::y()], 1.0);
        game.set(std::iter::repeat_n(light, MAX_AREA_LIGHTS + 2));
        assert_eq!(effects.get().len(), MAX_AREA_LIGHTS);
    }
}
