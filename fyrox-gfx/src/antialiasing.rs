//! Anti-aliasing.
//!
//! Two kinds, chosen with [`AntiAliasing`]:
//!
//! - Temporal ([`crate::temporal`]): the view shifts by a fraction of a pixel every frame and the
//!   frames are blended, which smooths edges and keeps them from crawling as the camera moves. It
//!   needs the wgpu (Vulkan) backend; on OpenGL, FXAA is used instead. The default.
//! - FXAA, the engine's own: one pass over the finished frame that finds edges by luminance and
//!   blends across them. Cheap, but it sees one frame at a time, so thin geometry still flickers
//!   in motion.
//!
//! What is *not* available here:
//!
//! - MSAA. The wgpu (Vulkan) backend does not implement it, and multi-sampling a deferred
//!   renderer means multi-sampling the whole G-buffer, which is a change inside the engine rather
//!   than something a crate can add from outside.
//! - Supersampling. Rendering larger than the window and scaling down would be the simplest real
//!   improvement over FXAA, but the renderer's frame size is not public, so it cannot be driven
//!   from a plugin.

use fyrox::renderer::{QualitySettings, Renderer};

/// Which anti-aliasing the renderer should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AntiAliasing {
    /// Blend across edges found in the finished frame (FXAA). Cheap, and softens thin geometry
    /// and high-contrast edges; it cannot recover detail smaller than a pixel.
    pub fxaa: bool,
    /// Blend each frame with the ones before it, each drawn with the view shifted by a different
    /// fraction of a pixel. See [`crate::temporal`].
    pub temporal: bool,
}

impl Default for AntiAliasing {
    fn default() -> Self {
        Self {
            fxaa: false,
            temporal: true,
        }
    }
}

impl AntiAliasing {
    /// No anti-aliasing at all.
    pub fn off() -> Self {
        Self {
            fxaa: false,
            temporal: false,
        }
    }

    /// The engine's FXAA alone.
    pub fn fxaa() -> Self {
        Self {
            fxaa: true,
            temporal: false,
        }
    }

    /// Applies this setting on top of the renderer's current ones.
    pub fn apply(&self, renderer: &mut Renderer) {
        let mut settings = renderer.get_quality_settings();
        self.apply_to(&mut settings);
        fyrox::core::log::Log::verify(renderer.set_quality_settings(&settings));
    }

    pub fn apply_to(&self, settings: &mut QualitySettings) {
        settings.fxaa = self.fxaa;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn anti_aliasing_can_be_turned_on_and_off() {
        let mut settings = QualitySettings::low();
        assert!(!settings.fxaa);
        AntiAliasing::fxaa().apply_to(&mut settings);
        assert!(settings.fxaa);
        AntiAliasing::off().apply_to(&mut settings);
        assert!(!settings.fxaa);
    }

    #[test]
    fn temporal_is_the_default_and_leaves_fxaa_off() {
        let mut settings = QualitySettings::low();
        AntiAliasing::default().apply_to(&mut settings);
        assert!(AntiAliasing::default().temporal);
        assert!(!settings.fxaa, "FXAA on top would only blur the blended frames");
    }
}
