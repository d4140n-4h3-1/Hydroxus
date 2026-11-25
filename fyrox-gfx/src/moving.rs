//! Things that move, for the effects that otherwise take the scene to be standing still.
//!
//! Temporal anti-aliasing ([`crate::temporal`]) finds where each pixel was last frame from the
//! depth buffer and the camera alone, which is right for walls and floors and wrong for anything
//! that moves: a character walking along with the camera stays put on screen, but the
//! reprojection looks for it where the wall behind it was, and blends that in - a ghost. The game
//! says what moves, as capsules around it, and how far each has moved since the last frame, and
//! the effects follow them instead.
//!
//! [`GraphicsEffects::moving_things`](crate::GraphicsEffects::moving_things) hands out the list;
//! the game sets it every frame.

use fyrox::core::algebra::Vector3;
use std::{cell::RefCell, rc::Rc};

/// How many moving things the effects follow. Any more are treated as standing still.
pub const MAX_MOVING_THINGS: usize = 2;

/// Something that moves: the capsule around it, in world space, and how far it has moved since
/// the last frame.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MovingThing {
    /// The two ends of the capsule's middle line.
    pub bottom: Vector3<f32>,
    pub top: Vector3<f32>,
    pub radius: f32,
    /// How far it has moved since the frame before, in meters.
    pub moved: Vector3<f32>,
}

/// The things that move this frame, shared between the game and the effects.
#[derive(Debug, Clone, Default)]
pub struct MovingThings(Rc<RefCell<Vec<MovingThing>>>);

impl MovingThings {
    /// What moves this frame, in place of whatever did last frame.
    pub fn set(&self, things: impl IntoIterator<Item = MovingThing>) {
        let mut list = self.0.borrow_mut();
        list.clear();
        list.extend(things.into_iter().take(MAX_MOVING_THINGS));
    }

    pub(crate) fn get(&self) -> Vec<MovingThing> {
        self.0.borrow().clone()
    }
}

impl PartialEq for MovingThings {
    fn eq(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.0, &other.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn thing() -> MovingThing {
        MovingThing {
            bottom: Vector3::zeros(),
            top: Vector3::y(),
            radius: 0.5,
            moved: Vector3::x(),
        }
    }

    #[test]
    fn a_copy_sees_what_the_game_sets() {
        let game = MovingThings::default();
        let effects = game.clone();
        game.set([thing()]);
        assert_eq!(effects.get(), vec![thing()]);
        game.set([]);
        assert!(effects.get().is_empty(), "each frame replaces the last");
    }

    #[test]
    fn only_so_many_are_followed() {
        let things = MovingThings::default();
        things.set(std::iter::repeat_n(thing(), MAX_MOVING_THINGS + 3));
        assert_eq!(things.get().len(), MAX_MOVING_THINGS);
    }
}
