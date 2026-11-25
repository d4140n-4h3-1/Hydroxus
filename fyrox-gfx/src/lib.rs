//! Graphics improvements for Fyrox that live outside the engine.
//!
//! Add [`GraphicsEffects`] to the executor next to the game's own plugin, then use the effects:
//!
//! - [`refraction`]: glass that bends and tints whatever is behind it. Build a material with
//!   [`GlassMaterial`] and put it on any mesh.
//! - [`shadows`]: softer shadow edges, applied to the renderer's quality settings on startup.
//!   Pass [`GraphicsEffects::without_soft_shadows`] to keep the game's own settings.
//! - [`antialiasing`]: temporal anti-aliasing ([`temporal`]), on by default, or the renderer's
//!   FXAA.
//! - [`occlusion`]: how far ambient occlusion reaches, which is what keeps corners from going
//!   flat. On by default, at the engine's own strength.
//! - [`reflections`]: ray-marched against the depth buffer. Off by default.
//! - [`area_lights`] (feature `raytracing`): glowing rectangles that light what is round them,
//!   with soft traced shadows where shadows are traced. Off until the game sets some.
//! - [`double_sided`]: surfaces seen from both sides, for meshes whose faces disagree on which
//!   way is in - a tube to walk through, say.
//! - [`moving`]: what in the scene moves, which the game tells the effects every frame so that
//!   temporal anti-aliasing does not leave a ghost behind it.
//! - [`shadows::ShadowBudget`]: keeps shadow maps for the lights nearest the camera only, which is
//!   what makes a scene full of small lamps affordable. On by default, and idle while shadows are
//!   traced, since then there are no shadow maps to budget.
//! - [`shadows::LightBudget`]: the same for the lights themselves. Off by default, because a light
//!   going out is visible when it lit something the player can see.
//!
//! The crate works with either engine backend - OpenGL or wgpu (Vulkan) - and follows the one the
//! game was built with.
//!
//! # About ray tracing
//!
//! [`raytraced_shadows`] (feature `raytracing`) traces the shadows of every light against the
//! scene's geometry using the graphics card's ray tracing hardware, through the engine's wgpu
//! backend. It replaces the renderer's shadow maps, and leaves them in place on hardware without
//! ray tracing.
//!
//! [`reflections`] traces rays too, but against the depth buffer: the scene as the camera sees it.
//! That is what can be done from outside the engine, and it is what most games ship. It cannot
//! show what the camera cannot - a reflection of something behind the camera or around a corner
//! has nothing to trace against.
//!
//! It is cheaper than tracing geometry and needs no special hardware, which is why it is the one
//! that is on for games that want reflections everywhere.

pub mod antialiasing;
#[cfg(feature = "raytracing")]
pub mod area_lights;
pub mod distance_blur;
pub mod double_sided;
pub mod moving;
pub mod occlusion;
#[cfg(feature = "raytracing")]
pub mod raytraced_shadows;
pub mod reflections;
pub mod refraction;
mod scene_copy;
pub mod shadows;
#[cfg(feature = "raytracing")]
pub mod support;
pub mod temporal;
#[cfg(feature = "raytracing")]
pub mod yoctite;

pub use antialiasing::AntiAliasing;
pub use distance_blur::DistanceBlur;
pub use double_sided::make_double_sided;
#[cfg(feature = "raytracing")]
pub use area_lights::{AreaLight, AreaLights};
pub use moving::{MovingThing, MovingThings};
pub use occlusion::AmbientOcclusion;
#[cfg(feature = "raytracing")]
pub use raytraced_shadows::RayTracedShadows;
pub use reflections::Reflections;
pub use refraction::{replace_materials, GlassMaterial};
pub use shadows::{LightBudget, ShadowBudget, SoftShadows};

use fyrox::{
    core::{reflect::prelude::*, visitor::prelude::*},
    engine::GraphicsContext,
    plugin::{error::GameResult, Plugin, PluginContext},
};
use refraction::RefractionPass;
use std::{
    any::TypeId,
    cell::{Cell, RefCell},
    rc::Rc,
};

/// A switch the game holds to turn an effect off, and on again, while it runs: for a scene that
/// looks better without it, or to save what it costs. On until it is turned off. Shared like
/// [`MovingThings`]: a copy taken before the plugin is handed to the executor stays connected.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Switch(Rc<Cell<bool>>);

impl Switch {
    /// Turns the effect on or off, from the next frame.
    pub fn set_enabled(&self, enabled: bool) {
        self.0.set(!enabled);
    }

    pub fn is_enabled(&self) -> bool {
        !self.0.get()
    }
}

/// The switch for shadows - see [`GraphicsEffects::shadow_switch`].
pub type ShadowSwitch = Switch;

/// Installs the crate's render passes into the renderer.
#[derive(Debug, Visit, Reflect)]
#[reflect(non_cloneable, type_uuid = "b6e3f1f2-8d0a-4c1e-a1f7-2c5b9e4d7a10")]
pub struct GraphicsEffects {
    /// What Yoctite does, if anything: see [`yoctite::Mode`].
    #[cfg(feature = "raytracing")]
    #[visit(skip)]
    #[reflect(hidden)]
    yoctite: Option<yoctite::Mode>,
    /// Whether Candelite's surface view is laid over the frame: see [`Self::with_candelite_view`].
    #[cfg(feature = "raytracing")]
    candelite_view: bool,
    /// Whether that view shows the light Candelite lights the cards with: see
    /// [`Self::with_candelite_light_view`].
    #[cfg(feature = "raytracing")]
    candelite_light: bool,
    /// Whether Candelite lights the frame: see [`Self::with_candelite`].
    #[cfg(feature = "raytracing")]
    candelite: bool,
    #[visit(skip)]
    #[reflect(hidden)]
    refraction: Option<Rc<RefCell<RefractionPass>>>,
    #[visit(skip)]
    #[reflect(hidden)]
    soft_shadows: Option<SoftShadows>,
    #[visit(skip)]
    #[reflect(hidden)]
    anti_aliasing: Option<AntiAliasing>,
    #[visit(skip)]
    #[reflect(hidden)]
    shadow_budget: Option<ShadowBudget>,
    #[visit(skip)]
    #[reflect(hidden)]
    light_budget: Option<LightBudget>,
    #[visit(skip)]
    #[reflect(hidden)]
    ambient_occlusion: Option<AmbientOcclusion>,
    #[visit(skip)]
    #[reflect(hidden)]
    reflections: Option<Reflections>,
    #[visit(skip)]
    #[reflect(hidden)]
    reflection_pass: Option<Rc<RefCell<reflections::ReflectionPass>>>,
    #[visit(skip)]
    #[reflect(hidden)]
    temporal_pass: Option<Rc<RefCell<temporal::TemporalPass>>>,
    /// What moves, as the game says.
    #[visit(skip)]
    #[reflect(hidden)]
    moving: MovingThings,
    /// Frames drawn with temporal anti-aliasing, which picks each frame's shift of the view.
    #[visit(skip)]
    #[reflect(hidden)]
    temporal_frame: u64,
    #[cfg(feature = "raytracing")]
    #[visit(skip)]
    #[reflect(hidden)]
    ray_traced_shadows: Option<RayTracedShadows>,
    /// The glowing rectangles, as the game says.
    #[cfg(feature = "raytracing")]
    #[visit(skip)]
    #[reflect(hidden)]
    area_lights: AreaLights,
    #[cfg(feature = "raytracing")]
    #[visit(skip)]
    #[reflect(hidden)]
    area_light_pass: Option<Rc<RefCell<area_lights::AreaLightPass>>>,
    /// Whether the renderer's shadows are being traced, which leaves the shadow budget with
    /// nothing to do.
    #[visit(skip)]
    #[reflect(hidden)]
    shadows_traced: bool,
    /// Whether anything casts shadows, as the game says; and the renderer's shadow settings from
    /// before they were switched off, while they are.
    #[visit(skip)]
    #[reflect(hidden)]
    shadow_switch: ShadowSwitch,
    #[visit(skip)]
    #[reflect(hidden)]
    shadows_before_off: Option<[bool; 3]>,
    /// Whether surfaces reflect, as the game says.
    #[visit(skip)]
    #[reflect(hidden)]
    reflection_switch: Switch,
    /// Blurring what is far off, if wanted (see [`distance_blur`]), and whether it is on, as the
    /// game says.
    #[visit(skip)]
    #[reflect(hidden)]
    distance_blur: Option<DistanceBlur>,
    #[visit(skip)]
    #[reflect(hidden)]
    distance_blur_switch: Switch,
}

impl Default for GraphicsEffects {
    fn default() -> Self {
        Self {
            #[cfg(feature = "raytracing")]
            yoctite: None,
            #[cfg(feature = "raytracing")]
            candelite_view: false,
            #[cfg(feature = "raytracing")]
            candelite_light: false,
            #[cfg(feature = "raytracing")]
            candelite: false,
            refraction: None,
            soft_shadows: Some(SoftShadows::default()),
            anti_aliasing: Some(AntiAliasing::default()),
            shadow_budget: Some(ShadowBudget::default()),
            // Off by default: switching a light off is visible when it lights something the
            // player can see, so a game should choose the number itself.
            light_budget: None,
            ambient_occlusion: Some(AmbientOcclusion::default()),
            reflections: None,
            reflection_pass: None,
            temporal_pass: None,
            moving: MovingThings::default(),
            temporal_frame: 0,
            #[cfg(feature = "raytracing")]
            ray_traced_shadows: None,
            #[cfg(feature = "raytracing")]
            area_lights: AreaLights::default(),
            #[cfg(feature = "raytracing")]
            area_light_pass: None,
            shadows_traced: false,
            shadow_switch: ShadowSwitch::default(),
            shadows_before_off: None,
            reflection_switch: Switch::default(),
            distance_blur: None,
            distance_blur_switch: Switch::default(),
        }
    }
}

impl GraphicsEffects {
    /// The list of what moves in the scene, for the game to keep up to date every frame - see
    /// [`moving`]. The effects share it, so a copy taken before the plugin is handed to the
    /// executor stays connected.
    pub fn moving_things(&self) -> MovingThings {
        self.moving.clone()
    }

    /// The switch for shadows, on until the game turns it off. Off, no shadow maps are drawn and
    /// nothing is traced: the scene is not built for tracing, so area lights shine through
    /// everything as they do where tracing is not supported. On again, it all comes back as it
    /// was.
    pub fn shadow_switch(&self) -> Switch {
        self.shadow_switch.clone()
    }

    /// The switch for reflections, if there are any (see [`Self::with_reflections`]), on until
    /// the game turns it off. Off, no rays are marched for them.
    pub fn reflection_switch(&self) -> Switch {
        self.reflection_switch.clone()
    }

    /// The switch for blurring what is far off, if it is wanted (see
    /// [`Self::with_distance_blur`]): on until the game turns it off.
    pub fn distance_blur_switch(&self) -> Switch {
        self.distance_blur_switch.clone()
    }

    /// Blurs what is far off, more the further (see [`distance_blur`]). Off by default.
    /// Draws the scene's meshes with Yoctite, in place of the renderer: as clusters at the level
    /// of detail each needs from the camera, chosen on the GPU, into the G-Buffer, a draw for
    /// each material (see [`yoctite::YoctitePass`]). Meshes it cannot draw as the engine does stay
    /// the renderer's. Needs the wgpu backend and the `raytracing` feature.
    #[cfg(feature = "raytracing")]
    pub fn with_yoctite(mut self) -> Self {
        self.yoctite = Some(yoctite::Mode::Render);
        self
    }

    /// As [`Self::with_yoctite`], and with Candelite's surface view laid over the frame: what a
    /// ray from the eye through each pixel finds on the meshes' cards, the first part of the
    /// global illumination to come (see [`yoctite::YoctitePass::show_candelite`]). Magenta where a
    /// ray hit something no card saw.
    #[cfg(feature = "raytracing")]
    pub fn with_candelite_view(mut self) -> Self {
        self.yoctite = Some(yoctite::Mode::Render);
        self.candelite_view = true;
        self
    }

    /// As [`Self::with_yoctite`], and lit by Candelite's global illumination: the light between
    /// surfaces - the sky, and sunlight bounced off whatever it falls on, bounce after bounce -
    /// gathered at every pixel by screen probes, in place of the scene's flat ambient colour
    /// (see [`yoctite::YoctitePass::light_with_candelite`]).
    #[cfg(feature = "raytracing")]
    pub fn with_candelite(mut self) -> Self {
        self.yoctite = Some(yoctite::Mode::Render);
        self.candelite = true;
        self
    }

    /// As [`Self::with_candelite_view`], but the cards are lit - the sun with traced shadows,
    /// and the light between surfaces, bounce after bounce - and the view shows the light leaving
    /// what each ray from the eye hit (see [`yoctite::YoctitePass::show_candelite_light`]).
    #[cfg(feature = "raytracing")]
    pub fn with_candelite_light_view(mut self) -> Self {
        self.yoctite = Some(yoctite::Mode::Render);
        self.candelite_view = true;
        self.candelite_light = true;
        self
    }

    /// Draws Yoctite's cluster view over the frame instead: the scene's meshes as Yoctite's
    /// clusters, each in a colour of its own, at the level of detail each would be drawn at,
    /// chosen on the GPU.
    #[cfg(feature = "raytracing")]
    pub fn with_yoctite_view(mut self) -> Self {
        self.yoctite = Some(yoctite::Mode::View { on_gpu: true });
        self
    }

    /// As [`Self::with_yoctite_view`], but the clusters chosen on the CPU: the reference the GPU's
    /// choice is held to.
    #[cfg(feature = "raytracing")]
    pub fn with_yoctite_view_on_cpu(mut self) -> Self {
        self.yoctite = Some(yoctite::Mode::View { on_gpu: false });
        self
    }

    pub fn with_distance_blur(mut self, blur: DistanceBlur) -> Self {
        self.distance_blur = Some(blur);
        self
    }

    /// The list of area lights, for the game to fill - see [`area_lights`]. Shared like
    /// [`Self::moving_things`].
    #[cfg(feature = "raytracing")]
    pub fn area_lights(&self) -> AreaLights {
        self.area_lights.clone()
    }

    /// Leaves shadow settings as the game set them.
    pub fn without_soft_shadows(mut self) -> Self {
        self.soft_shadows = None;
        self
    }

    /// Uses the given shadow settings instead of the default ones.
    pub fn with_soft_shadows(mut self, shadows: SoftShadows) -> Self {
        self.soft_shadows = Some(shadows);
        self
    }

    /// Lets every light draw shadows, however many there are.
    pub fn without_shadow_budget(mut self) -> Self {
        self.shadow_budget = None;
        self
    }

    /// Uses the given shadow budget instead of the default one.
    pub fn with_shadow_budget(mut self, budget: ShadowBudget) -> Self {
        self.shadow_budget = Some(budget);
        self
    }

    /// Lets every light in the scene light it, however many there are. This is the default.
    pub fn without_light_budget(mut self) -> Self {
        self.light_budget = None;
        self
    }

    /// Uses the given light budget instead of the default one.
    pub fn with_light_budget(mut self, budget: LightBudget) -> Self {
        self.light_budget = Some(budget);
        self
    }

    /// Traces the shadows of every light against the scene's geometry, where the hardware can,
    /// in place of the renderer's shadow maps. Off by default.
    #[cfg(feature = "raytracing")]
    pub fn with_ray_traced_shadows(mut self, shadows: RayTracedShadows) -> Self {
        self.ray_traced_shadows = Some(shadows);
        self
    }

    /// Traces reflections on surfaces facing upwards. Off by default.
    pub fn with_reflections(mut self, reflections: Reflections) -> Self {
        self.reflections = Some(reflections);
        self
    }

    /// Leaves ambient occlusion as the game set it.
    pub fn without_ambient_occlusion(mut self) -> Self {
        self.ambient_occlusion = None;
        self
    }

    /// Uses the given ambient occlusion instead of the default.
    pub fn with_ambient_occlusion(mut self, occlusion: AmbientOcclusion) -> Self {
        self.ambient_occlusion = Some(occlusion);
        self
    }

    /// Leaves anti-aliasing as the game set it.
    pub fn without_anti_aliasing(mut self) -> Self {
        self.anti_aliasing = None;
        self
    }

    /// Uses the given anti-aliasing setting instead of the default one.
    pub fn with_anti_aliasing(mut self, anti_aliasing: AntiAliasing) -> Self {
        self.anti_aliasing = Some(anti_aliasing);
        self
    }
}

impl PartialEq for GraphicsEffects {
    fn eq(&self, other: &Self) -> bool {
        match (&self.refraction, &other.refraction) {
            (Some(a), Some(b)) => Rc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        }
    }
}

impl Plugin for GraphicsEffects {
    fn on_graphics_context_initialized(&mut self, ctx: PluginContext) -> GameResult {
        if let GraphicsContext::Initialized(graphics_context) = ctx.graphics_context {
            // The scene as traced for shadows, which the area lights trace against too.
            #[cfg(feature = "raytracing")]
            let traced_scene = raytraced_shadows::SharedScene::default();
            // Before the glass, so glass shows the light behind it.
            #[cfg(feature = "raytracing")]
            if raytraced_shadows::is_supported_backend(graphics_context.renderer.graphics_server()) {
                let pass = Rc::new(RefCell::new(area_lights::AreaLightPass::new(
                    TypeId::of::<Self>(),
                    self.area_lights.clone(),
                    traced_scene.clone(),
                )));
                graphics_context.renderer.add_render_pass(pass.clone());
                self.area_light_pass = Some(pass);
            }
            let pass = Rc::new(RefCell::new(RefractionPass::new(TypeId::of::<Self>())));
            graphics_context.renderer.add_render_pass(pass.clone());
            self.refraction = Some(pass);
            if let Some(shadows) = self.soft_shadows {
                shadows.apply(&mut graphics_context.renderer);
            }
            let mut temporal = false;
            if let Some(mut anti_aliasing) = self.anti_aliasing {
                // Temporal anti-aliasing has a wgpu shader only; on OpenGL, FXAA stands in.
                if anti_aliasing.temporal && !refraction::engine_uses_wgsl() {
                    anti_aliasing = AntiAliasing::fxaa();
                }
                anti_aliasing.apply(&mut graphics_context.renderer);
                temporal = anti_aliasing.temporal;
            }
            if let Some(occlusion) = self.ambient_occlusion {
                occlusion.apply(&mut graphics_context.renderer);
            }
            #[cfg(feature = "raytracing")]
            if let Some(shadows) = self.ray_traced_shadows {
                let renderer = &mut graphics_context.renderer;
                // Only where it can work: otherwise shadow maps, and their budget, stay in charge.
                self.shadows_traced = raytraced_shadows::is_supported(renderer.graphics_server());
                if self.shadows_traced {
                    renderer.set_light_shadow_tracer(Some(Box::new(
                        raytraced_shadows::TracedLightShadows::new(
                            shadows,
                            traced_scene,
                            self.shadow_switch.clone(),
                            temporal,
                        ),
                    )));
                }
            }
            if let Some(reflections) = self.reflections {
                // After the glass pass, so glass shows up in reflections too.
                let pass = Rc::new(RefCell::new(reflections::ReflectionPass::new(
                    TypeId::of::<Self>(),
                    reflections,
                    self.reflection_switch.clone(),
                )));
                graphics_context.renderer.add_render_pass(pass.clone());
                self.reflection_pass = Some(pass);
            }
            if let Some(blur) = self.distance_blur {
                // Before the temporal pass, which smooths the softened frame.
                let pass = Rc::new(RefCell::new(distance_blur::DistanceBlurPass::new(
                    TypeId::of::<Self>(),
                    blur,
                    self.distance_blur_switch.clone(),
                )));
                graphics_context.renderer.add_render_pass(pass);
            }
            if temporal {
                // Last, so it smooths everything drawn before it: glass and reflections too.
                let pass = Rc::new(RefCell::new(temporal::TemporalPass::new(
                    TypeId::of::<Self>(),
                    self.moving.clone(),
                )));
                graphics_context.renderer.add_render_pass(pass.clone());
                self.temporal_pass = Some(pass);
            }
            // After everything, temporal smoothing too, so its colours are its own.
            // Only where the card can run them (see [`support`]): elsewhere the renderer draws every
            // mesh itself, and lights the scene with its flat ambient colour.
            #[cfg(feature = "raytracing")]
            let support = self.yoctite.map(|_| {
                graphics_context
                    .renderer
                    .graphics_server()
                    .as_any()
                    .downcast_ref::<fyrox_graphics_wgpu::server::WgpuGraphicsServer>()
                    .map(|server| support::Support::of(&server.state.adapter, &server.state.device))
                    .unwrap_or_else(|| {
                        let why = "they need the wgpu backend".to_string();
                        support::Support { yoctite: Err(why.clone()), candelite: Err(why) }
                    })
            });
            #[cfg(feature = "raytracing")]
            let wants_candelite = self.candelite || self.candelite_view;
            #[cfg(feature = "raytracing")]
            match (self.yoctite, &support) {
                (Some(_), Some(support::Support { yoctite: Err(why), .. })) => fyrox::core::log::Log::warn(format!(
                    "Yoctite is off: {why}. The renderer draws every mesh itself{}.",
                    if wants_candelite { ", and Candelite is off with it: the scene is lit with its flat ambient colour" } else { "" }
                )),
                (Some(_), Some(support::Support { candelite: Err(why), .. })) if wants_candelite => fyrox::core::log::Log::warn(format!(
                    "Candelite is off: {why}. Yoctite draws the scene, lit with its flat ambient colour."
                )),
                (Some(_), Some(_)) => fyrox::core::log::Log::info(if wants_candelite {
                    "Yoctite draws the scene, and Candelite lights it: the graphics card runs both."
                } else {
                    "Yoctite draws the scene: the graphics card runs it."
                }),
                _ => {}
            }
            #[cfg(feature = "raytracing")]
            if let (Some(mode), Some(support::Support { yoctite: Ok(()), candelite })) = (self.yoctite, support) {
                let candelite = candelite.is_ok();
                // Drawing in place of the renderer, it tells the renderer which meshes it draws.
                let taken_over = (mode == yoctite::Mode::Render).then(|| {
                    let set = fyrox::renderer::TakenOverMeshes::default();
                    graphics_context.renderer.set_taken_over_meshes(Some(set.clone()));
                    set
                });
                let mut pass = yoctite::YoctitePass::new(TypeId::of::<Self>(), mode, taken_over);
                if candelite && self.candelite_light {
                    pass.show_candelite_light();
                } else if candelite && self.candelite_view {
                    pass.show_candelite();
                }
                if self.candelite && candelite {
                    let indirect = fyrox::renderer::IndirectDiffuse::default();
                    let specular = fyrox::renderer::IndirectSpecular::default();
                    graphics_context.renderer.set_indirect_diffuse(Some(indirect.clone()));
                    graphics_context.renderer.set_indirect_specular(Some(specular.clone()));
                    pass.light_with_candelite(indirect, specular);
                }
                let pass = Rc::new(RefCell::new(pass));
                graphics_context.renderer.add_render_pass(pass);
            }
        }
        Ok(())
    }

    fn update(&mut self, ctx: &mut PluginContext) -> GameResult {
        if let GraphicsContext::Initialized(graphics_context) = &mut *ctx.graphics_context {
            shadows::follow_switch(
                self.shadow_switch.is_enabled(),
                &mut graphics_context.renderer,
                &mut self.shadows_before_off,
            );
        }
        if self.temporal_pass.is_some() {
            if let GraphicsContext::Initialized(graphics_context) = &mut *ctx.graphics_context {
                self.temporal_frame += 1;
                let jitter = temporal::jitter(self.temporal_frame);
                graphics_context.renderer.set_projection_jitter(jitter);
                if let Some(pass) = self.temporal_pass.as_ref() {
                    pass.borrow_mut().set_jitter(jitter);
                }
            }
        }

        // Traced shadows come from the lights' cast_shadows flag too, so the budget would switch
        // them off for all but the nearest lights.
        let mut shadow_budget = self.shadow_budget.as_mut().filter(|_| !self.shadows_traced);
        if shadow_budget.is_some() || self.light_budget.is_some() {
            let handles: Vec<_> = ctx.scenes.pair_iter().map(|(handle, _)| handle).collect();
            for handle in &handles {
                let Ok(scene) = ctx.scenes.try_get_mut(*handle) else {
                    continue;
                };
                // Lights first: a light that is off does not need a shadow map either.
                if let Some(budget) = self.light_budget.as_mut() {
                    budget.apply(*handle, scene);
                }
                if let Some(budget) = shadow_budget.as_deref_mut() {
                    budget.apply(*handle, scene);
                }
            }
            if let Some(budget) = self.light_budget.as_mut() {
                budget.forget_missing(|scene| handles.contains(&scene));
            }
            if let Some(budget) = shadow_budget {
                budget.forget_missing(|scene| handles.contains(&scene));
            }
        }
        Ok(())
    }

    fn on_graphics_context_destroyed(&mut self, ctx: PluginContext) -> GameResult {
        // The renderer is gone with the context; the pass holds GPU objects of that renderer.
        let _ = ctx;
        self.refraction = None;
        self.reflection_pass = None;
        self.temporal_pass = None;
        Ok(())
    }

    fn on_deinit(&mut self, ctx: PluginContext) -> GameResult {
        if let GraphicsContext::Initialized(graphics_context) = ctx.graphics_context {
            if let Some(pass) = self.refraction.take() {
                graphics_context.renderer.remove_render_pass(pass);
            }
            if let Some(pass) = self.reflection_pass.take() {
                graphics_context.renderer.remove_render_pass(pass);
            }
            if let Some(pass) = self.temporal_pass.take() {
                graphics_context.renderer.remove_render_pass(pass);
                graphics_context
                    .renderer
                    .set_projection_jitter(Default::default());
            }
            if self.shadows_traced {
                graphics_context.renderer.set_light_shadow_tracer(None);
                self.shadows_traced = false;
            }
        }
        Ok(())
    }
}
