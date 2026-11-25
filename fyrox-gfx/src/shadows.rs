//! Softer shadow edges.
//!
//! The engine can filter shadow maps (percentage-closer filtering), which turns the hard, stair-
//! stepped edge of a shadow map lookup into a gradient a few texels wide. It is off for point and
//! spot lights in every built-in quality preset except `ultra`, which is why shadows from lamps
//! and torches tend to look cut out. [`SoftShadows`] turns it on, and gives the fade-out at the
//! edge of the shadow range enough distance to be a fade rather than a step.
//!
//! This only sets the renderer's quality settings; any of them can be set differently afterwards.

use fyrox::{
    core::{algebra::Vector3, pool::Handle},
    fxhash::FxHashMap,
    graph::SceneGraph,
    renderer::{QualitySettings, Renderer, ShadowMapPrecision},
    scene::{
        light::{point::PointLight, spot::SpotLight},
        node::Node,
        Scene,
    },
};

/// Shadow settings that favor soft edges.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SoftShadows {
    /// Filter point light shadows.
    pub point: bool,
    /// Filter spot light shadows.
    pub spot: bool,
    /// Filter the sun's cascaded shadow maps.
    pub directional: bool,
    /// Shadow map size for point and spot lights. Bigger maps have finer shadows, and a filter of
    /// the same width covers proportionally less of the scene, so the edge gets tighter.
    pub map_size: usize,
    /// Over how many meters shadows fade out at the end of their range. A short range makes
    /// shadows wink out; a few meters reads as a fade.
    pub fade_out_range: f32,
    /// How far from the camera point and spot lights still draw shadow maps. This is the main
    /// cost of having many small lights: every light inside this distance re-renders the scene
    /// (six times over, for a point light) every frame.
    pub distance: f32,
}

impl Default for SoftShadows {
    fn default() -> Self {
        Self {
            point: true,
            spot: true,
            directional: true,
            map_size: 1024,
            fade_out_range: 4.0,
            distance: 20.0,
        }
    }
}

impl SoftShadows {
    /// Applies these settings on top of the renderer's current ones.
    pub fn apply(&self, renderer: &mut Renderer) {
        let mut settings = renderer.get_quality_settings();
        self.apply_to(&mut settings);
        fyrox::core::log::Log::verify(renderer.set_quality_settings(&settings));
    }

    pub fn apply_to(&self, settings: &mut QualitySettings) {
        settings.point_soft_shadows = self.point;
        settings.spot_soft_shadows = self.spot;
        settings.csm_settings.pcf = self.directional;
        settings.point_shadow_map_size = self.map_size;
        settings.spot_shadow_map_size = self.map_size;
        settings.point_shadow_map_precision = ShadowMapPrecision::Full;
        settings.spot_shadow_map_precision = ShadowMapPrecision::Full;
        settings.point_shadows_fade_out_range = self.fade_out_range;
        settings.spot_shadows_fade_out_range = self.fade_out_range;
        settings.point_shadows_distance = self.distance;
        settings.spot_shadows_distance = self.distance;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fyrox::{
        core::algebra::Vector3,
        scene::{
            base::BaseBuilder, camera::CameraBuilder, light::point::PointLightBuilder,
            light::BaseLightBuilder, transform::TransformBuilder,
        },
    };

    fn at(position: Vector3<f32>) -> BaseBuilder {
        BaseBuilder::new().with_local_transform(
            TransformBuilder::new()
                .with_local_position(position)
                .build(),
        )
    }

    #[test]
    fn only_the_nearest_lights_stay_lit() {
        let mut scene = Scene::new();
        CameraBuilder::new(at(Vector3::new(0.0, 0.0, 0.0))).build(&mut scene.graph);
        let near = PointLightBuilder::new(BaseLightBuilder::new(at(Vector3::new(1.0, 0.0, 0.0))))
            .build(&mut scene.graph);
        let far = PointLightBuilder::new(BaseLightBuilder::new(at(Vector3::new(50.0, 0.0, 0.0))))
            .build(&mut scene.graph);
        // A light the game deliberately switched off stays off, however near it is.
        let hidden = PointLightBuilder::new(BaseLightBuilder::new(
            at(Vector3::new(0.5, 0.0, 0.0)).with_visibility(false),
        ))
        .build(&mut scene.graph);
        scene.graph.update_hierarchical_data();

        let mut budget = LightBudget::new(1);
        budget.apply(Handle::NONE, &mut scene);

        assert!(scene.graph[near].visibility());
        assert!(!scene.graph[far].visibility());
        assert!(!scene.graph[hidden].visibility(), "authored setting kept");
    }

    #[test]
    fn only_the_nearest_lights_keep_their_shadows() {
        let mut scene = Scene::new();
        CameraBuilder::new(at(Vector3::new(0.0, 0.0, 0.0))).build(&mut scene.graph);
        let near = PointLightBuilder::new(BaseLightBuilder::new(at(Vector3::new(1.0, 0.0, 0.0))))
            .build(&mut scene.graph);
        let middle = PointLightBuilder::new(BaseLightBuilder::new(at(Vector3::new(5.0, 0.0, 0.0))))
            .build(&mut scene.graph);
        let far = PointLightBuilder::new(BaseLightBuilder::new(at(Vector3::new(50.0, 0.0, 0.0))))
            .build(&mut scene.graph);
        // A light the game deliberately set up without shadows, right next to the camera.
        let unlit = PointLightBuilder::new(BaseLightBuilder::new(
            at(Vector3::new(0.5, 0.0, 0.0)).with_cast_shadows(false),
        ))
        .build(&mut scene.graph);
        scene.graph.update_hierarchical_data();

        let mut budget = ShadowBudget::new(2);
        budget.apply(Handle::NONE, &mut scene);

        assert!(scene.graph[near].cast_shadows());
        assert!(scene.graph[middle].cast_shadows());
        assert!(!scene.graph[far].cast_shadows());
        assert!(!scene.graph[unlit].cast_shadows(), "authored setting kept");

        // Walking up to the far light gives it shadows back, and takes them from the others.
        scene.graph[near.transmute::<Node>()]
            .local_transform_mut()
            .set_position(Vector3::new(100.0, 0.0, 0.0));
        scene.graph.update_hierarchical_data();
        budget.apply(Handle::NONE, &mut scene);

        assert!(!scene.graph[near].cast_shadows());
        assert!(scene.graph[middle].cast_shadows());
        assert!(scene.graph[far].cast_shadows());
    }

    #[test]
    fn soft_shadows_turn_filtering_on_without_disabling_shadows() {
        let mut settings = QualitySettings::low();
        assert!(!settings.point_soft_shadows);
        let shadows_were_on = settings.point_shadows_enabled;

        SoftShadows::default().apply_to(&mut settings);

        assert!(settings.point_soft_shadows);
        assert!(settings.spot_soft_shadows);
        assert!(settings.csm_settings.pcf);
        assert!(settings.point_shadows_fade_out_range > 1.0);
        // Whether shadows are drawn at all is the game's choice, not this effect's.
        assert_eq!(settings.point_shadows_enabled, shadows_were_on);
    }
}

/// How many lights may draw shadow maps at once.
///
/// Every point light that draws shadows re-renders the scene into six cube map faces each frame,
/// and a spot light once - so a room with a lamp every few meters spends most of its frame on
/// shadow maps for lights whose shadows are barely visible. This keeps the shadows of the lights
/// nearest the camera and turns the rest off, which is far less noticeable than it sounds: the
/// shadow of a distant lamp is small, and its light is faint by then anyway.
///
/// Lights the game set up without shadows are left alone.
#[derive(Debug, Clone, PartialEq)]
pub struct ShadowBudget {
    /// How many nearest lights keep their shadows.
    pub max_casters: usize,
    /// The lights' authored shadow settings, remembered so they can be restored.
    authored: FxHashMap<(Handle<Scene>, Handle<Node>), bool>,
}

impl Default for ShadowBudget {
    fn default() -> Self {
        Self::new(4)
    }
}

impl ShadowBudget {
    pub fn new(max_casters: usize) -> Self {
        Self {
            max_casters,
            authored: Default::default(),
        }
    }

    /// Picks which lights of `scene` draw shadows this frame.
    pub fn apply(&mut self, scene_handle: Handle<Scene>, scene: &mut Scene) {
        let Some(viewpoint) = viewpoint_of(scene) else {
            return;
        };

        // Lights the game wants shadows from, nearest first.
        let mut candidates: Vec<(Handle<Node>, f32)> = scene
            .graph
            .pair_iter()
            // Point and spot lights only: the sun's cascaded maps are drawn once for the whole
            // scene, however many other lights there are.
            .filter(|(_, node)| {
                node.cast::<PointLight>().is_some() || node.cast::<SpotLight>().is_some()
            })
            .filter_map(|(handle, node)| {
                let authored = *self
                    .authored
                    .entry((scene_handle, handle))
                    .or_insert_with(|| node.cast_shadows());
                authored.then(|| (handle, (node.global_position() - viewpoint).norm_squared()))
            })
            .collect();
        candidates.sort_by(|(_, a), (_, b)| a.total_cmp(b));

        for (rank, (handle, _)) in candidates.iter().enumerate() {
            let wanted = rank < self.max_casters;
            if let Ok(node) = scene.graph.try_get_mut(*handle) {
                if node.cast_shadows() != wanted {
                    node.set_cast_shadows(wanted);
                }
            }
        }
    }

    /// Forgets lights of scenes that no longer exist.
    pub fn forget_missing(&mut self, is_alive: impl Fn(Handle<Scene>) -> bool) {
        self.authored.retain(|(scene, _), _| is_alive(*scene));
    }
}

/// How many lights may light the scene at once.
///
/// A deferred renderer pays for every light whose volume is on screen, whether or not anything of
/// it can be seen. A corridor maze with a lamp every few meters puts dozens of them in the frustum
/// with walls in between, so most of that work never reaches the screen. This keeps the lights
/// nearest the camera and switches the rest off.
///
/// It is a blunt instrument - a distant lamp in plain sight down a long corridor can wink out - so
/// the budget should be generous enough to cover what a player can actually see at once.
#[derive(Debug, Clone, PartialEq)]
pub struct LightBudget {
    /// How many nearest lights stay lit.
    pub max_lights: usize,
    /// The lights' authored visibility, remembered so it can be restored.
    authored: FxHashMap<(Handle<Scene>, Handle<Node>), bool>,
}

impl Default for LightBudget {
    fn default() -> Self {
        Self::new(16)
    }
}

impl LightBudget {
    pub fn new(max_lights: usize) -> Self {
        Self {
            max_lights,
            authored: Default::default(),
        }
    }

    /// Picks which lights of `scene` are lit this frame.
    pub fn apply(&mut self, scene_handle: Handle<Scene>, scene: &mut Scene) {
        let Some(viewpoint) = viewpoint_of(scene) else {
            return;
        };

        let mut candidates: Vec<(Handle<Node>, f32)> = scene
            .graph
            .pair_iter()
            // The sun lights everything from nowhere in particular, so it is never budgeted.
            .filter(|(_, node)| {
                node.cast::<PointLight>().is_some() || node.cast::<SpotLight>().is_some()
            })
            .filter_map(|(handle, node)| {
                let authored = *self
                    .authored
                    .entry((scene_handle, handle))
                    .or_insert_with(|| node.visibility());
                authored.then(|| (handle, (node.global_position() - viewpoint).norm_squared()))
            })
            .collect();
        candidates.sort_by(|(_, a), (_, b)| a.total_cmp(b));

        for (rank, (handle, _)) in candidates.iter().enumerate() {
            let wanted = rank < self.max_lights;
            if let Ok(node) = scene.graph.try_get_mut(*handle) {
                if node.visibility() != wanted {
                    node.set_visibility(wanted);
                }
            }
        }
    }

    /// Forgets lights of scenes that no longer exist.
    pub fn forget_missing(&mut self, is_alive: impl Fn(Handle<Scene>) -> bool) {
        self.authored.retain(|(scene, _), _| is_alive(*scene));
    }
}

/// Where the scene is being looked at from.
fn viewpoint_of(scene: &Scene) -> Option<Vector3<f32>> {
    scene
        .graph
        .linear_iter()
        .find(|node| node.is_camera() && node.global_visibility())
        .map(|camera| camera.global_position())
}

/// Switches the renderer's shadow maps to follow `enabled`, if they do not already, keeping how
/// they were set to put them back. `applied` is what was last applied: `None` for shadows as the
/// renderer has them, or the settings they had before they were turned off.
pub(crate) fn follow_switch(enabled: bool, renderer: &mut Renderer, applied: &mut Option<[bool; 3]>) {
    let mut settings = renderer.get_quality_settings();
    match (enabled, applied.as_ref()) {
        (false, None) => {
            *applied = Some([
                settings.point_shadows_enabled,
                settings.spot_shadows_enabled,
                settings.csm_settings.enabled,
            ]);
            settings.point_shadows_enabled = false;
            settings.spot_shadows_enabled = false;
            settings.csm_settings.enabled = false;
        }
        (true, Some(&[point, spot, sun])) => {
            *applied = None;
            settings.point_shadows_enabled = point;
            settings.spot_shadows_enabled = spot;
            settings.csm_settings.enabled = sun;
        }
        _ => return,
    }
    fyrox::core::log::Log::verify(renderer.set_quality_settings(&settings));
}
