//! Ambient occlusion.
//!
//! Ambient light arrives from everywhere at once, so on its own it lights a corner exactly as much
//! as an open wall and the shapes of a room go flat. Screen-space ambient occlusion takes it away
//! again where the geometry blocks it - in corners, under ledges, where two surfaces meet - which
//! is what puts the shape back.
//!
//! The engine has it; this is where its strength is set, in meters rather than in quality presets.

use fyrox::renderer::{QualitySettings, Renderer};

/// How much ambient light the geometry blocks.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AmbientOcclusion {
    /// Whether to compute it at all.
    pub enabled: bool,
    /// How far, in meters, a surface looks around itself for something blocking the ambient light.
    /// Small values only darken tight creases; large ones shade whole corners, and start to darken
    /// surfaces that nothing is really blocking.
    pub radius: f32,
}

impl Default for AmbientOcclusion {
    fn default() -> Self {
        Self {
            enabled: true,
            radius: 0.5,
        }
    }
}

impl AmbientOcclusion {
    /// No ambient occlusion.
    pub fn off() -> Self {
        Self {
            enabled: false,
            ..Default::default()
        }
    }

    /// Ambient occlusion reaching this far, in meters.
    pub fn reaching(radius: f32) -> Self {
        Self {
            enabled: true,
            radius,
        }
    }

    /// Applies this setting on top of the renderer's current ones.
    pub fn apply(&self, renderer: &mut Renderer) {
        let mut settings = renderer.get_quality_settings();
        self.apply_to(&mut settings);
        fyrox::core::log::Log::verify(renderer.set_quality_settings(&settings));
    }

    pub fn apply_to(&self, settings: &mut QualitySettings) {
        settings.use_ssao = self.enabled;
        settings.ssao_radius = self.radius;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ambient_occlusion_sets_its_reach() {
        let mut settings = QualitySettings::low();
        AmbientOcclusion::reaching(0.9).apply_to(&mut settings);
        assert!(settings.use_ssao);
        assert_eq!(settings.ssao_radius, 0.9);

        AmbientOcclusion::off().apply_to(&mut settings);
        assert!(!settings.use_ssao);
    }
}
