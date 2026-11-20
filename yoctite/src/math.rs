//! The little vector maths the builder needs: points, bounding spheres, and error quadrics.

use std::ops::{Add, AddAssign, Mul, Sub};

/// A point or direction, in double precision: simplification sums many small squared
/// distances, which single precision loses.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl V3 {
    pub fn new(x: f64, y: f64, z: f64) -> Self {
        Self { x, y, z }
    }

    pub fn from_f32(p: [f32; 3]) -> Self {
        Self::new(p[0] as f64, p[1] as f64, p[2] as f64)
    }

    pub fn to_f32(self) -> [f32; 3] {
        [self.x as f32, self.y as f32, self.z as f32]
    }

    pub fn dot(self, o: Self) -> f64 {
        self.x * o.x + self.y * o.y + self.z * o.z
    }

    pub fn cross(self, o: Self) -> Self {
        Self::new(
            self.y * o.z - self.z * o.y,
            self.z * o.x - self.x * o.z,
            self.x * o.y - self.y * o.x,
        )
    }

    pub fn length(self) -> f64 {
        self.dot(self).sqrt()
    }

    /// Unit length, or zero for a zero vector.
    pub fn normalized(self) -> Self {
        let length = self.length();
        if length > 0.0 {
            self * (1.0 / length)
        } else {
            Self::default()
        }
    }
}

impl Add for V3 {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self::new(self.x + o.x, self.y + o.y, self.z + o.z)
    }
}

impl AddAssign for V3 {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

impl Sub for V3 {
    type Output = Self;
    fn sub(self, o: Self) -> Self {
        Self::new(self.x - o.x, self.y - o.y, self.z - o.z)
    }
}

impl Mul<f64> for V3 {
    type Output = Self;
    fn mul(self, s: f64) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }
}

/// A sphere: what a cluster's bounds, and the bounds its level of detail is judged by, are.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Sphere {
    pub center: [f32; 3],
    pub radius: f32,
}

impl Sphere {
    /// A sphere round every one of `points` (Ritter's: not the smallest, but close, and quick).
    pub(crate) fn around(points: impl Iterator<Item = V3> + Clone) -> Self {
        let Some(first) = points.clone().next() else {
            return Self::default();
        };
        // The point furthest from any point, then the point furthest from that: roughly a
        // diameter.
        let far_from = |from: V3| {
            points
                .clone()
                .fold(from, |best, p| if (p - from).length() > (best - from).length() { p } else { best })
        };
        let a = far_from(first);
        let b = far_from(a);
        let mut center = (a + b) * 0.5;
        let mut radius = (b - a).length() * 0.5;
        // Grown to take in whatever it misses.
        for p in points {
            let d = (p - center).length();
            if d > radius {
                let grown = (radius + d) * 0.5;
                center = center + (p - center) * ((grown - radius) / d);
                radius = grown;
            }
        }
        Self {
            center: center.to_f32(),
            // A little more, for what rounding to single precision loses.
            radius: (radius * (1.0 + 1e-6) + 1e-7) as f32,
        }
    }

    /// The smallest sphere round both `self` and `other`.
    pub fn enclosing(self, other: Sphere) -> Sphere {
        let (a, b) = (V3::from_f32(self.center), V3::from_f32(other.center));
        let (ra, rb) = (self.radius as f64, other.radius as f64);
        let d = (b - a).length();
        if d + rb <= ra {
            return self;
        }
        if d + ra <= rb {
            return other;
        }
        let radius = (d + ra + rb) * 0.5;
        let center = a + (b - a) * ((radius - ra) / d);
        Sphere {
            center: center.to_f32(),
            radius: (radius * (1.0 + 1e-6) + 1e-7) as f32,
        }
    }

    /// Whether `other` lies wholly inside, give or take rounding.
    pub fn contains(&self, other: &Sphere) -> bool {
        let d = (V3::from_f32(self.center) - V3::from_f32(other.center)).length();
        d + other.radius as f64 <= self.radius as f64 * (1.0 + 1e-4) + 1e-5
    }
}

/// The sum of squared distances to a set of planes, each weighted: what moving a vertex costs in
/// how far the surface strays from where it was (Garland and Heckbert's quadric error metric).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub(crate) struct Quadric {
    // The symmetric 4x4 matrix's upper triangle.
    a: [f64; 10],
    /// The total weight of the planes in it, to make a mean of the sum.
    pub weight: f64,
}

impl Quadric {
    /// A plane through `point` facing `normal` (unit length), weighted by `weight`.
    pub fn plane(normal: V3, point: V3, weight: f64) -> Self {
        let (a, b, c) = (normal.x, normal.y, normal.z);
        let d = -normal.dot(point);
        let w = weight;
        Self {
            a: [
                a * a * w,
                a * b * w,
                a * c * w,
                a * d * w,
                b * b * w,
                b * c * w,
                b * d * w,
                c * c * w,
                c * d * w,
                d * d * w,
            ],
            weight,
        }
    }

    /// The weighted sum of squared distances from `p` to the planes.
    pub fn error(&self, p: V3) -> f64 {
        let q = &self.a;
        let (x, y, z) = (p.x, p.y, p.z);
        let e = q[0] * x * x
            + 2.0 * q[1] * x * y
            + 2.0 * q[2] * x * z
            + 2.0 * q[3] * x
            + q[4] * y * y
            + 2.0 * q[5] * y * z
            + 2.0 * q[6] * y
            + q[7] * z * z
            + 2.0 * q[8] * z
            + q[9];
        e.max(0.0)
    }
}

impl Add for Quadric {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        let mut a = self.a;
        for (a, b) in a.iter_mut().zip(o.a) {
            *a += b;
        }
        Self {
            a,
            weight: self.weight + o.weight,
        }
    }
}

impl AddAssign for Quadric {
    fn add_assign(&mut self, o: Self) {
        *self = *self + o;
    }
}

/// Spreads the bits of a 10-bit number three apart, for a Morton code.
fn spread(v: u32) -> u32 {
    let mut v = v & 0x3ff;
    v = (v | (v << 16)) & 0x030000ff;
    v = (v | (v << 8)) & 0x0300f00f;
    v = (v | (v << 4)) & 0x030c30c3;
    v = (v | (v << 2)) & 0x09249249;
    v
}

/// The order `points` come in along a Morton (Z-order) curve through their bounding box: points
/// near each other in that order are near each other in space.
pub(crate) fn morton_order(points: &[V3]) -> Vec<usize> {
    let (mut lo, mut hi) = (V3::new(f64::MAX, f64::MAX, f64::MAX), V3::new(f64::MIN, f64::MIN, f64::MIN));
    for p in points {
        lo = V3::new(lo.x.min(p.x), lo.y.min(p.y), lo.z.min(p.z));
        hi = V3::new(hi.x.max(p.x), hi.y.max(p.y), hi.z.max(p.z));
    }
    let extent = (hi - lo).x.max((hi - lo).y).max((hi - lo).z).max(1e-12);
    let cell = |v: f64| ((v / extent) * 1023.0).clamp(0.0, 1023.0) as u32;
    let mut keyed: Vec<(u32, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let d = *p - lo;
            (spread(cell(d.x)) | (spread(cell(d.y)) << 1) | (spread(cell(d.z)) << 2), i)
        })
        .collect();
    keyed.sort_unstable();
    keyed.into_iter().map(|(_, i)| i).collect()
}
