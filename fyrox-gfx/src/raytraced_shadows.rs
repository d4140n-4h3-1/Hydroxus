//! Shadows traced against the scene's geometry, for every light.
//!
//! A shadow map is a picture of the scene from the light's point of view, and everything about it -
//! the stair-stepped edges, the peter-panning, the cascades - comes from it being a picture at a
//! fixed resolution. It also has to be drawn again every frame for every light, six times over for
//! a lamp, which is why a scene full of lamps can only afford shadows for the nearest few, and why
//! shadows come and go as the camera moves. Tracing asks the geometry directly instead: for each
//! pixel a light reaches, is there a triangle between it and the light? The answer is exact at any
//! distance, with no maps to size, no cascades to blend, and no bias to tune beyond keeping a
//! surface from shadowing itself - and its cost is the pixels lit, not the scene redrawn.
//!
//! This needs ray tracing hardware. [`TracedLightShadows`] is handed to the renderer as its
//! [`LightShadowTracer`]: the renderer asks it for a shadow mask for each light just before
//! drawing that light, and uses the mask in place of the light's shadow map, so the shadow darkens
//! that light alone. Without the hardware it returns no masks and the renderer's shadow maps are
//! used as before.
//!
//! Lights are given a size, and a few rays per pixel are spread across it, so a shadow's edge
//! softens where only part of the light is hidden - wider the further the shadow falls from what
//! casts it, as a real one does. The mask is then blurred a little, only across the same surface,
//! to smooth out the differences between neighboring pixels. One ray per pixel gives hard shadows
//! and skips the blur.
//!
//! Each mesh's triangles are put into the hardware's structure once, in the mesh's own space -
//! once for every mesh made from the same surface data, such as a maze's repeated tiles - and every
//! frame a copy of each is placed wherever its mesh is, so what moves casts its shadow from where
//! it is. Most of a scene never moves, though, and placing thousands of copies every frame costs
//! more than tracing against them: so a mesh that has never moved, a second after it appears
//! ([`SETTLE_FRAMES`]), is put in with the others standing still in the same cube of space
//! ([`BATCH_CELL`]), as one piece of geometry already where they all are, placed once. A maze of
//! thousands of walls is traced as a few hundred pieces. When a mesh that was put in moves, or
//! goes, or a new one appears beside them, only its cube is put together again; what has once
//! moved stays a copy of its own from then on, so that what is carried about does not keep
//! putting cubes together again. When nothing has moved or changed since the last frame, the
//! scene is not placed again at all.
//!
//! A skinned mesh's triangles are skinned by its bones every frame, as the renderer draws
//! them, and refitted - within [`SKIN_REACH`] of the camera; further off, its shadow stays as it
//! last was, too small to tell. Each skinned surface's vertices, and the bones and weights they
//! hang off, are read once and kept, so that posing them is only the arithmetic; and a surface
//! whose bones have not moved since it was last posed - a body lying still, say - is not posed
//! again, nor its geometry refitted: it casts the shadow it last did, at no cost. Hidden meshes
//! still cast shadows while they stand still, since a game hides what the camera cannot see; once a mesh has moved, or if it is skinned, it casts none while
//! hidden, as with shadow maps.
//!
//! A mesh bent by its blend shapes - a tree swaying in the wind, a fern - is bent the same way
//! within [`BEND_REACH`], every frame its weights change, so its shadow moves with it: its
//! geometry, a copy of its own, is bent by its weights on the GPU, as the renderer bends the mesh,
//! and refitted (`fyrox_graphics_wgpu::raytracing::RayTracedPoser`); the CPU only hands over the
//! weights. Further off, it casts the shadow of the mesh unbent - in Wilderness the sway has all
//! but faded out there. A mesh that bends is never put in with the still meshes, so coming within
//! reach and going out of it again costs no cube put together again.
//!
//! Glass stands in light's way too, but lets it through, taking on its colour: a lamp shone
//! through red glass casts red light past it. How much of each colour gets through is the glass's
//! tint, as strongly as its tint strength; glass told not to colour light
//! ([`crate::GlassMaterial::tints_light`]) is left out, and light goes through it as if it were not
//! there. The light is only coloured, not bent: where the glass would focus it, it does not.
//!
//! Leaves, needles and grass are usually cards: a few triangles each, with a texture whose alpha
//! says where the leaf is, and the renderer leaves the rest undrawn where alpha falls below a
//! half. Traced as plain triangles they would cast the shadow of the card, a solid rectangle; so a
//! surface whose material's diffuse texture (times its diffuse colour's alpha) has any texel the
//! renderer would leave undrawn is traced as a cutout instead, stopping light only where it is
//! drawn ([`fyrox_graphics_wgpu::raytracing::CutoutMask`], made once per texture). Cutouts are
//! never put in with the still meshes: a forest is thousands of copies of a few trees, and
//! placing a copy of each shape is far less than every tree's foliage put together.

use fyrox::{
    core::{
        algebra::{Matrix4, Point3, Vector3},
        color::Color,
        log::Log,
        pool::Handle,
        sstorage::ImmutableString,
    },
    fxhash::{FxHashMap, FxHashSet},
    graph::SceneGraph,
    graphics::{error::FrameworkError, gpu_texture::GpuTexture, server::GraphicsServer},
    material::{MaterialProperty, MaterialResource, MaterialResourceBinding},
    renderer::{
        bundle::LightSourceKind,
        traced_shadows::{LightShadowTraceContext, LightShadowTracer},
    },
    scene::{
        camera::Camera,
        graph::{Graph, NodeChange},
        mesh::{
            surface::Surface,
            buffer::{VertexAttributeUsage, VertexReadTrait},
            Mesh,
        },
        node::Node,
        Scene,
    },
};
use std::{cell::RefCell, rc::Rc, sync::Arc};
use fyrox_graphics_wgpu::{
    raytracing::{
        CutoutCorners, CutoutMask, RayTracedGeometry, RayTracedInstance, RayTracedMorph, RayTracedPoser, RayTracedScene, ShadowRayLight,
        ShadowRayParameters, ShadowTracer,
    },
    server::WgpuGraphicsServer,
};

/// How shadows are traced.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayTracedShadows {
    /// How far a ray towards the sun reaches, in meters. Anything further away casts no shadow.
    /// Rays towards a lamp end at the lamp.
    pub reach: f32,
    /// How far off a surface a ray starts, in meters, so it does not shadow itself.
    pub bias: f32,
    /// Rays per pixel, spread over the light's size: soft edges for a light with a size, smoother
    /// the more rays. The rays turn over the light from frame to frame, and the frames are
    /// averaged; but the average is held close to each frame's own shadows, so a moving shadow
    /// leaves no ghost, and with one ray a frame's shadows are as noisy as that holds them to: two
    /// halve it, for little more than one costs (the rest of tracing a light costs as much).
    pub samples: u32,
    /// The radius of the glowing part of a point light, in meters. Bigger is softer.
    pub point_light_size: f32,
    /// The same for spot lights, which are usually small bulbs - a torch, a headlight.
    pub spot_light_size: f32,
    /// How big the sun looks, in degrees from its middle to its edge. The real one is about a
    /// quarter of a degree; a little more reads better at a game's scale.
    pub sun_angular_radius: f32,
    /// Whether to trace one pixel of each 2x2 block, a quarter of the rays, filling in the rest
    /// from the traced pixels on the same surface. With temporal anti-aliasing a different pixel
    /// is traced each frame and the frames are averaged, so a still view comes to full detail.
    pub half_resolution: bool,
}

impl Default for RayTracedShadows {
    fn default() -> Self {
        Self {
            reach: 200.0,
            bias: 0.02,
            samples: 2,
            point_light_size: 0.15,
            spot_light_size: 0.03,
            sun_angular_radius: 1.0,
            half_resolution: true,
        }
    }
}

impl RayTracedShadows {
    /// Sharp shadows: lights with no size, one ray per pixel, and no blur.
    pub fn hard() -> Self {
        Self {
            samples: 1,
            point_light_size: 0.0,
            spot_light_size: 0.0,
            sun_angular_radius: 0.0,
            ..Default::default()
        }
    }
}

/// Whether the graphics server can trace rays at all.
pub fn is_supported(server: &dyn GraphicsServer) -> bool {
    server
        .as_any()
        .downcast_ref::<WgpuGraphicsServer>()
        .is_some_and(|server| server.ray_tracing)
}

/// Whether the graphics server is the engine's wgpu one, which the area lights need, tracing or
/// not.
pub(crate) fn is_supported_backend(server: &dyn GraphicsServer) -> bool {
    server.as_any().is::<WgpuGraphicsServer>()
}

/// How a surface of `material` stands in the way of light: solid, glass that lets through so
/// much of a light's red, green and blue, a cutout solid only where the mask kept under this key
/// says (see [`Masks`]), or not at all.
#[derive(Debug, Clone, Copy, PartialEq)]
enum Blocks {
    Solid,
    Glass([f32; 3]),
    Cutout(u64),
}

/// The renderer's alpha test: a texel whose alpha, times the diffuse colour's, falls below this
/// is left undrawn (the standard and glTF shaders' G-buffer pass).
const ALPHA_TEST: f32 = 0.5;

/// The widest and highest a cutout mask is made, in texels: a bigger texture is sampled every
/// so many texels. Finer than this makes no difference a shadow shows.
const MASK_LIMIT: u32 = 1024;

/// The cutout masks made so far, by texture and the diffuse colour's alpha (see [`mask_key`]):
/// [`None`] for a texture with every texel drawn, which is traced solid.
///
/// Each held with a [`Pin`] on its texture: a texture's key is made from where it is in memory, so
/// a texture made after another was let go - as a game making its textures anew, a season's each
/// day, does all the time - could otherwise be given the other's mask.
pub(crate) type Masks = FxHashMap<u64, (Option<Arc<CutoutMask>>, Pin)>;

/// A weak hold on a resource, which keeps what it is made in from being given to another while
/// it is held, and so its key - which is where it is in memory - from being another's; but not
/// the resource itself, whose data goes once nothing else holds it. For a cache keyed by
/// resources' keys, so that it is never asked about another resource made where one it holds
/// was (and, once [`Self::gone`], to let it go).
#[derive(Clone, Debug)]
pub(crate) struct Pin(std::sync::Weak<fyrox::core::parking_lot::Mutex<fyrox::asset::untyped::ResourceHeader>>);

impl Pin {
    pub(crate) fn of<T: fyrox::asset::TypedResourceData>(resource: &fyrox::asset::Resource<T>) -> Self {
        Self(Arc::downgrade(&resource.clone().into_untyped().0))
    }

    /// Whether nothing holds the resource any more.
    pub(crate) fn gone(&self) -> bool {
        self.0.strong_count() == 0
    }
}

/// Lets go of the masks of textures nothing holds any more.
pub(crate) fn prune(masks: &mut Masks) {
    masks.retain(|_, (_, pin)| !pin.gone());
}

/// The key a mask of `texture`, with diffuse alpha `alpha`, is kept under.
fn mask_key(texture: u64, alpha: u8) -> u64 {
    texture.rotate_left(8) ^ alpha as u64
}

/// Which texels of `texture` the renderer draws, with every alpha first multiplied by `alpha`:
/// [`None`] if it draws them all, or if its texels cannot be read here (a compressed format, or a
/// render target, whose texels are only ever on the GPU).
fn cutout_mask(texture: &fyrox::resource::texture::Texture, alpha: f32) -> Option<CutoutMask> {
    use fyrox::resource::texture::{TextureKind, TexturePixelKind};
    let TextureKind::Rectangle { width, height } = texture.kind() else {
        return None;
    };
    // Where in each texel its alpha is, and how many bytes a texel takes.
    let (stride, at, wide) = match texture.pixel_kind() {
        TexturePixelKind::RGBA8 | TexturePixelKind::SRGBA8 | TexturePixelKind::BGRA8 => (4, 3, false),
        TexturePixelKind::LuminanceAlpha8 => (2, 1, false),
        TexturePixelKind::RGBA16 => (8, 6, true),
        _ => return None,
    };
    // The first mip level is at the start of the texture's bytes. A render target has none on the
    // CPU, so its length is checked here before slicing: `mip_level_data` would panic on it.
    let size = width as usize * height as usize * stride;
    let data = texture.data();
    if width == 0 || height == 0 || data.len() < size {
        return None;
    }
    let data = &data[..size];
    let drawn = |x: u32, y: u32| {
        let i = (y as usize * width as usize + x as usize) * stride + at;
        let texel = if wide {
            u16::from_le_bytes([data[i], data[i + 1]]) as f32 / 65535.0
        } else {
            data[i] as f32 / 255.0
        };
        texel * alpha >= ALPHA_TEST
    };
    let step = |size: u32| size.div_ceil(MASK_LIMIT).max(1);
    let (sx, sy) = (step(width), step(height));
    let mask = CutoutMask::new(width.div_ceil(sx), height.div_ceil(sy), |x, y| {
        drawn((x * sx).min(width - 1), (y * sy).min(height - 1))
    });
    let all_drawn = (0..mask.height()).all(|y| (0..mask.width()).all(|x| mask.is_solid(x, y)));
    (!all_drawn).then_some(mask)
}

/// `material`'s own value of the property `name`, if it sets one.
fn property<'a>(material: &'a fyrox::material::Material, name: &str) -> Option<&'a MaterialProperty> {
    material
        .property_group_ref(ImmutableString::new("properties"))?
        .property_ref(name)
}

/// Whether a surface of `material` is a cutout, and the key of its mask in `masks` if so, making
/// the mask if it has not been made: see [`Blocks::Cutout`]. Not while its texture is loading.
/// The cutout mask of `material`'s diffuse texture - where the renderer's alpha test draws it -
/// made once and kept in `masks`; [`None`] if it draws all of it, or it cannot be read.
pub(crate) fn material_mask(material: &MaterialResource, masks: &mut Masks) -> Option<Arc<CutoutMask>> {
    let material = material.state();
    let material = material.data_ref()?;
    let key = cutout(&material, masks)?;
    masks.get(&key).and_then(|(mask, _)| mask.clone())
}

/// The key of `material`'s cutout mask in `masks` (made and kept there the first time), if its
/// diffuse texture has see-through texels: leaves, needles, grass.
pub(crate) fn cutout(material: &fyrox::material::Material, masks: &mut Masks) -> Option<u64> {
    let texture = material.texture_ref("diffuseTexture")?.value.clone()?;
    let alpha = match property(material, "diffuseColor") {
        Some(MaterialProperty::Color(color)) => color.a,
        _ => 255,
    };
    let key = mask_key(texture.key(), alpha);
    if let Some((mask, _)) = masks.get(&key) {
        return mask.is_some().then_some(key);
    }
    let pin = Pin::of(&texture);
    let state = texture.state();
    let texture = state.data_ref()?;
    let mask = cutout_mask(&texture, alpha as f32 / 255.0).map(Arc::new);
    let is_cutout = mask.is_some();
    masks.insert(key, (mask, pin));
    is_cutout.then_some(key)
}

/// The texture coordinates of `surface`'s first `count` vertices, two floats each, as the
/// renderer samples its diffuse texture with: scaled by `material`'s `texCoordScale`, if it has
/// one.
pub(crate) fn texture_coordinates(surface: &Surface, count: usize) -> Vec<f32> {
    let scale = surface
        .material()
        .state()
        .data_ref()
        .and_then(|material| match property(&material, "texCoordScale") {
            Some(MaterialProperty::Vector2(scale)) => Some(*scale),
            _ => None,
        })
        .unwrap_or(fyrox::core::algebra::Vector2::new(1.0, 1.0));
    let data = surface.data();
    let data = data.data_ref();
    let mut out = Vec::with_capacity(count * 2);
    for vertex in data.vertex_buffer.iter().take(count) {
        let uv = vertex
            .read_2_f32(VertexAttributeUsage::TexCoord0)
            .unwrap_or_default();
        out.extend_from_slice(&[uv.x * scale.x, uv.y * scale.y]);
    }
    out
}

/// How a surface of `material` stands in the way of light, if it does. Glass lets light through,
/// taking on its tint as much as its tint strength says, unless it is told not to colour light,
/// when it is left out as if it were not there. Any other material that draws into no shadow
/// maps is left out too. One with texels the renderer leaves undrawn is a cutout.
fn blocks(material: &MaterialResource, masks: &mut Masks) -> Option<Blocks> {
    let material = material.state();
    let Some(material) = material.data_ref() else {
        return Some(Blocks::Solid);
    };
    if material.shader() == &crate::refraction::glass_shader() {
        let key = ImmutableString::new("properties");
        let Some(MaterialResourceBinding::PropertyGroup(group)) = material.bindings().get(&key)
        else {
            return None;
        };
        let float = |name: &str, default: f32| match group.property_ref(name) {
            Some(MaterialProperty::Float(v)) => *v,
            _ => default,
        };
        if float("tintsLight", 1.0) <= 0.0 {
            return None;
        }
        let strength = float("tintStrength", 0.0).clamp(0.0, 1.0);
        let tint = match group.property_ref("tint") {
            Some(MaterialProperty::Color(tint)) => *tint,
            _ => Color::WHITE,
        };
        let through = |c: u8| 1.0 - strength * (1.0 - c as f32 / 255.0);
        return Some(Blocks::Glass([through(tint.r), through(tint.g), through(tint.b)]));
    }
    let shader = material.shader().state();
    let Some(shader) = shader.data_ref() else {
        return Some(Blocks::Solid);
    };
    let opted_out = shader
        .definition
        .disabled_passes
        .iter()
        .any(|pass| pass == "PointShadow");
    if opted_out {
        return None;
    }
    Some(cutout(&material, masks).map_or(Blocks::Solid, Blocks::Cutout))
}

/// How far a mesh may drift, in meters, before it counts as having moved.
const MOVED: f32 = 1.0e-4;

/// Adds a copy of `local` vertices and their `triangles` to `vertices` and `indices`, placed by
/// `at` (see [`instance_transform`]), its indices counting on from the vertices already there.
fn append_placed(vertices: &mut Vec<f32>, indices: &mut Vec<u32>, local: &[f32], triangles: &[u32], at: &[f32; 12]) {
    let first = (vertices.len() / 3) as u32;
    for p in local.chunks_exact(3) {
        for row in 0..3 {
            let m = &at[row * 4..row * 4 + 4];
            vertices.push(m[0] * p[0] + m[1] * p[1] + m[2] * p[2] + m[3]);
        }
    }
    indices.extend(triangles.iter().map(|i| i + first));
}

/// Whether `a` and `b` put things in the same place.
fn same_place(a: &Matrix4<f32>, b: &Matrix4<f32>) -> bool {
    a.iter().zip(b.iter()).all(|(a, b)| (a - b).abs() <= MOVED)
}

/// `transform`'s top three rows, row by row, as an instance is placed with.
fn instance_transform(transform: &Matrix4<f32>) -> [f32; 12] {
    let mut out = [0.0; 12];
    for row in 0..3 {
        for column in 0..4 {
            out[row * 4 + column] = transform[(row, column)];
        }
    }
    out
}

/// How far from the camera a skinned mesh is posed afresh every frame, in meters: further off,
/// its shadow is too small for its pose to tell.
pub const SKIN_REACH: f32 = 30.0;

/// How far from the camera a mesh bent by blend shapes is bent in its shadow too, in meters.
pub const BEND_REACH: f32 = 40.0;

/// How far a mesh's shapes have to be able to move it - each at full strength, added up - for
/// its shadow to be bent too, for each meter it is from the camera. Wind seldom drives a shape
/// past a fifth of its strength, so this is about a pixel of a half-resolution shadow: a tree,
/// its limbs reaching half a meter or more, is bent all through [`BEND_REACH`]; a foxglove out to
/// about 40 m, a bilberry to about 12. Each mesh bent needs geometry of its own, which the scene
/// traced costs to put together every frame, so a meadow bent where its sway cannot be seen would
/// cost much for nothing.
pub const BEND_SHOWS: f32 = 0.02;

/// How many bent meshes are given geometry of their own a frame, at most: the floor streams in
/// plants a patch at a time, and building them all at once stalled a frame for tens of
/// milliseconds. One waiting its turn casts the shadow of its mesh unbent meanwhile.
pub const BEND_NEW_PER_FRAME: usize = 16;

/// How far, in meters, a bent mesh's vertices may have moved since its shadow's geometry was
/// last bent before it is bent again: less than a soft shadow shows.
pub const BEND_LEAST: f32 = 0.01;

/// And further off, how far for each meter between it and the camera, if more: a shadow's edge
/// moving shows by the angle it moves through - a centimetre near by is a few pixels, forty
/// metres off a fraction of one - and a leaf a few metres over the ground already casts an edge
/// some centimetres soft. Bending every plant in reach by the centimetre kept 480 refits a frame
/// going in Wilderness's forest; with this, 200.
pub const BEND_LEAST_PER_METER: f32 = 0.005;

/// The most bent meshes' geometry bent again in a frame: each is refitted, which costs the CPU a
/// few microseconds in wgpu, and a forest holds thousands within reach. Those that have moved the
/// furthest as the camera sees it go first - the nearest, mostly; the rest wait, their drift
/// growing, until it is their turn.
pub const BEND_BUDGET: usize = 256;


/// A surface with blend shapes as its data has it, read once: where its vertices are in its own
/// space, and how far each shape moves each at full strength. Handed to the GPU to bend copies
/// of its geometry by (`RayTracedMorph`).
struct Bend {
    positions: Vec<Vector3<f32>>,
    /// Shape after shape, a vertex after vertex.
    offsets: Vec<Vec<Vector3<f32>>>,
    /// How far each shape moves the vertex it moves furthest, at full strength.
    reach: Vec<f32>,
    /// Those added up: about the furthest its shapes together move it.
    sway: f32,
}

impl Bend {
    fn read(surface: &Surface) -> Option<Self> {
        let data = surface.data();
        let (positions, count) = {
            let data = data.data_ref();
            let positions: Vec<Vector3<f32>> = data
                .vertex_buffer
                .iter()
                .map(|v| v.read_3_f32(VertexAttributeUsage::Position).unwrap_or_default())
                .collect();
            let count = positions.len();
            (positions, count)
        };
        let (targets, morphs) = crate::yoctite::read_morphs(surface, count)?;
        let offsets: Vec<Vec<Vector3<f32>>> = (0..targets as usize)
            .map(|t| morphs.iter().map(|m| Vector3::new(m[t][0], m[t][1], m[t][2])).collect())
            .collect();
        let reach = offsets.iter().map(|o: &Vec<Vector3<f32>>| o.iter().map(|o| o.norm()).fold(0.0, f32::max)).collect();
        let reach: Vec<f32> = reach;
        let sway = reach.iter().sum();
        Some(Self { positions, offsets, reach, sway })
    }

    /// At most how far its vertices have moved, in its own space, going from `from` weights to
    /// `to` (none: unbent).
    fn drift(&self, from: Option<&[f32]>, to: &[f32]) -> f32 {
        let from = |t: usize| from.and_then(|f| f.get(t)).copied().unwrap_or(0.0);
        to.iter().zip(&self.reach).enumerate().map(|(t, (w, r))| (w - from(t)).abs() * r).sum()
    }

    /// Its vertices unbent, three floats each.
    fn flat_positions(&self) -> Vec<f32> {
        self.positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect()
    }

    /// How far each shape moves each vertex, three floats each, the shapes of a vertex together,
    /// vertex after vertex: as `RayTracedMorph` keeps them.
    fn flat_offsets(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.positions.len() * self.offsets.len() * 3);
        for v in 0..self.positions.len() {
            for shape in &self.offsets {
                let o = shape[v];
                out.extend_from_slice(&[o.x, o.y, o.z]);
            }
        }
        out
    }

    /// Its vertices bent by `weights`, 1 a shape at full strength, in its own space, three floats
    /// each: as the GPU bends them, from what it is handed - for tests to tell it is right.
    #[cfg(test)]
    fn pose(&self, weights: &[f32]) -> Vec<f32> {
        let (mut bent, offsets) = (self.flat_positions(), self.flat_offsets());
        let targets = self.offsets.len();
        for v in 0..self.positions.len() {
            for (t, &weight) in weights.iter().enumerate().take(targets) {
                for c in 0..3 {
                    bent[v * 3 + c] += offsets[(v * targets + t) * 3 + c] * weight;
                }
            }
        }
        bent
    }
}

/// A skinned surface's vertices as its data has them, read once: where each is in its own space,
/// and the four bones each hangs off, with how much.
struct Skin {
    positions: Vec<Vector3<f32>>,
    bones: Vec<[u8; 4]>,
    weights: Vec<[f32; 4]>,
}

impl Skin {
    fn read(surface: &Surface) -> Self {
        let data = surface.data();
        let data = data.data_ref();
        let count = data.vertex_buffer.vertex_count() as usize;
        let (mut positions, mut bones, mut weights) =
            (Vec::with_capacity(count), Vec::with_capacity(count), Vec::with_capacity(count));
        for vertex in data.vertex_buffer.iter() {
            let (Ok(position), Ok(which), Ok(weight)) = (
                vertex.read_3_f32(VertexAttributeUsage::Position),
                vertex.read_4_u8(VertexAttributeUsage::BoneIndices),
                vertex.read_4_f32(VertexAttributeUsage::BoneWeight),
            ) else {
                break;
            };
            positions.push(position);
            bones.push(which.into());
            weights.push(weight.into());
        }
        Self { positions, bones, weights }
    }

    /// Where bones posed by `matrices` (see [`bone_matrices`]) put its vertices in the world,
    /// three floats each.
    fn pose(&self, matrices: &[Matrix4<f32>]) -> Vec<f32> {
        let mut out = Vec::with_capacity(self.positions.len() * 3);
        for ((position, which), weights) in self.positions.iter().zip(&self.bones).zip(&self.weights) {
            let point = Point3::from(*position);
            let mut world = Vector3::zeros();
            for (&bone, &weight) in which.iter().zip(weights) {
                if weight > 0.0 {
                    if let Some(matrix) = matrices.get(bone as usize) {
                        world += matrix.transform_point(&point).coords * weight;
                    }
                }
            }
            out.extend_from_slice(&[world.x, world.y, world.z]);
        }
        out
    }
}

/// `surface`'s vertex positions, three floats each: in its own space, or, skinned, where its
/// bones in `graph` put them in the world, as the renderer draws it.
fn positions(graph: &Graph, surface: &Surface) -> Vec<f32> {
    let bones: Vec<Matrix4<f32>> = surface
        .bones()
        .iter()
        .map(|&bone| {
            graph.try_get_node(bone).map_or(Matrix4::identity(), |bone| {
                bone.global_transform() * bone.inv_bind_pose_transform()
            })
        })
        .collect();
    let data = surface.data();
    let data = data.data_ref();
    let mut out = Vec::with_capacity(data.vertex_buffer.vertex_count() as usize * 3);
    for vertex in data.vertex_buffer.iter() {
        let Ok(position) = vertex.read_3_f32(VertexAttributeUsage::Position) else {
            break;
        };
        let point = if bones.is_empty() {
            position
        } else {
            let (Ok(which), Ok(weights)) = (
                vertex.read_4_u8(VertexAttributeUsage::BoneIndices),
                vertex.read_4_f32(VertexAttributeUsage::BoneWeight),
            ) else {
                break;
            };
            let mut world = Vector3::zeros();
            for (&bone, &weight) in which.iter().zip(weights.iter()) {
                if let Some(matrix) = bones.get(bone as usize) {
                    world += matrix.transform_point(&Point3::from(position)).coords * weight;
                }
            }
            world
        };
        out.extend_from_slice(&[point.x, point.y, point.z]);
    }
    out
}

/// `surface`'s triangles, as indices into the first `count` of its vertices: only whole ones,
/// and each place only once. A sheet made double-sided by a reversed copy of each triangle -
/// foliage cards, for a renderer that culls faces seen from behind - has every triangle twice,
/// and a ray, which hits a triangle from either side, would test each twice: for cutouts, the
/// most of what a shadow ray costs. For glass, the twin would let through its share of the light
/// a second time.
fn triangles(surface: &Surface, count: usize) -> Vec<u32> {
    let data = surface.data();
    let data = data.data_ref();
    let count = count as u32;
    let positions: Vec<[u32; 3]> = data
        .vertex_buffer
        .iter()
        .map(|v| {
            v.read_3_f32(VertexAttributeUsage::Position)
                .map_or([0; 3], |p| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()])
        })
        .collect();
    let mut seen = FxHashSet::default();
    data.geometry_buffer
        .iter()
        .filter(|triangle| triangle.0.iter().all(|&i| i < count))
        .filter(|triangle| {
            // The same three places whichever way round, and from whichever corner.
            let mut corners = triangle.0.map(|i| positions.get(i as usize).copied().unwrap_or_default());
            corners.sort_unstable();
            seen.insert(corners)
        })
        .flat_map(|triangle| triangle.0)
        .collect()
}

/// What each of `surface`'s bones in `graph` does to the vertices hanging off it, in the world.
fn bone_matrices(graph: &Graph, surface: &Surface) -> Vec<Matrix4<f32>> {
    surface
        .bones()
        .iter()
        .map(|&bone| {
            graph.try_get_node(bone).map_or(Matrix4::identity(), |bone| {
                bone.global_transform() * bone.inv_bind_pose_transform()
            })
        })
        .collect()
}

/// Whether bones posed by `a` and by `b` put every vertex in the same place: none has moved.
fn same_pose(a: &[Matrix4<f32>], b: &[Matrix4<f32>]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(a, b)| same_place(a, b))
}

/// How long a mesh that has never moved has to have been there, in frames, before it is put in
/// with the others standing still round it ([`Batch`]).
const SETTLE_FRAMES: u64 = 60;

/// How wide the cubes of space are whose still meshes are traced as one, in meters. Smaller, a
/// mesh that moves or goes means less to put together again; bigger, fewer pieces to place.
const BATCH_CELL: f32 = 32.0;

/// Which still meshes are traced together: those in the same cube of space, standing in light's
/// way the same - solid, or glass letting through the same light.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BatchKey {
    cell: [i32; 3],
    /// The light let through, as bits, for glass.
    glass: Option<[u32; 3]>,
}

impl BatchKey {
    fn new(transform: &Matrix4<f32>, how: Blocks) -> Self {
        let cell = |i: usize| (transform[(i, 3)] / BATCH_CELL).floor() as i32;
        Self {
            cell: [cell(0), cell(1), cell(2)],
            glass: match how {
                // Cutouts are never batched; were one, it would be traced solid.
                Blocks::Solid | Blocks::Cutout(_) => None,
                Blocks::Glass(through) => Some(through.map(f32::to_bits)),
            },
        }
    }

    fn lets_through(&self) -> [f32; 3] {
        self.glass.map_or([0.0; 3], |bits| bits.map(f32::from_bits))
    }
}

/// Where a mesh was last frame, since which frame it has stood there, and whether it has ever
/// moved.
#[derive(Debug, Clone, Copy)]
struct Place {
    handle: Handle<Node>,
    transform: Matrix4<f32>,
    still_since: u64,
    moved: bool,
}

/// A still surface put in with others: which mesh and surface it is, and where; and its data,
/// so that a surface given other data is put in again.
type Fixed = (BatchKey, Member);

/// A surface in a [`Batch`]: which mesh and surface it is, where, and the data it was built from.
type Member = (Handle<Node>, usize, [f32; 12], u64);

/// Still meshes traced as one piece of geometry, already where they are in the world: the
/// hardware places one copy of it rather than one of each mesh, and nothing about it changes
/// from frame to frame. A maze's thousands of walls, floors and lamps are a few dozen of these.
struct Batch {
    members: Vec<Member>,
    geometry: RayTracedGeometry,
}

/// A surface that casts shadows: what it is traced with, where, which mesh and surface it is,
/// and how it stands in the way of light.
type Placed = (Shape, [f32; 12], Handle<Node>, usize, Blocks);

/// Which geometry a surface is traced with: what it shares with every surface made from the same
/// data, or, skinned, its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Shape {
    /// With how it is traced: the same data could be drawn solid, as glass, or as a cutout of
    /// one texture or another.
    Shared(u64, Traced),
    Posed(Handle<Node>, usize),
}

/// How a shape is traced: solid, as glass, or as a cutout with the mask kept under this key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Traced {
    Solid,
    Glass,
    Cutout(u64),
}

impl From<Blocks> for Traced {
    fn from(how: Blocks) -> Self {
        match how {
            Blocks::Solid => Self::Solid,
            Blocks::Glass(_) => Self::Glass,
            Blocks::Cutout(mask) => Self::Cutout(mask),
        }
    }
}

/// The scene as traced, shared with the area lights ([`crate::area_lights`]), which trace
/// against it too.
pub(crate) type SharedScene = Rc<RefCell<Option<RayTracedScene>>>;

/// Makes the renderer's shadows by tracing rays. [`crate::GraphicsEffects`] installs it.
pub struct TracedLightShadows {
    settings: RayTracedShadows,
    /// Whether the frames are averaged after (temporal anti-aliasing), so the rays can turn over
    /// the light from frame to frame.
    turn_over_frames: bool,
    tracer: Option<ShadowTracer>,
    scene: SharedScene,
    /// Whether to trace at all.
    switch: crate::Switch,
    /// Whether [`RayTracedShadows::half_resolution`] holds, or every pixel is traced.
    half_resolution: crate::Switch,
    /// Which scene it traces.
    scene_handle: Option<Handle<Scene>>,
    /// Each shape's geometry, built once, and posed geometry refitted every frame.
    geometry: FxHashMap<Shape, RayTracedGeometry>,
    /// Where each mesh was last frame, by the slot its node has in the graph, to tell when one
    /// moves. Kept from frame to frame, rather than made anew; a slot whose node has gone is taken
    /// by the next node put there.
    places: Vec<Option<Place>>,
    frame: u64,
    /// What each surface was traced with, where, and which surface it was, last frame - kept to
    /// fill again, and to tell whether anything has changed since.
    placed: Vec<Placed>,
    was_placed: Vec<Placed>,
    /// The meshes standing still, traced together, by where they are; and last frame's, to tell
    /// when one has come or gone.
    batches: FxHashMap<BatchKey, Batch>,
    fixed: Vec<Fixed>,
    was_fixed: Vec<Fixed>,
    /// Each skinned surface's vertices, read once, by its data; and how each posed surface's
    /// bones stood when it was last posed, so that one whose bones have not moved since is not
    /// posed again.
    skins: FxHashMap<u64, Skin>,
    poses: FxHashMap<Shape, Vec<Matrix4<f32>>>,
    /// Each surface with blend shapes, read once, by its data (or [`None`], if it has none to
    /// read). The weights each was last bent by are kept with its node (see [`Contrib`]).
    bends: FxHashMap<u64, Option<Bend>>,
    /// Each surface with blend shapes on the GPU, by its data, and what bends copies of it.
    morphs: FxHashMap<u64, Option<RayTracedMorph>>,
    poser: Option<RayTracedPoser>,
    /// Each cutout surface's corners' texture coordinates, by its data, shared by every copy of
    /// it built: so the scene keeps one record of its cutout for them all.
    corners: FxHashMap<u64, CutoutCorners>,
    /// The cutout masks made so far, kept for as long as the scene is traced.
    masks: Masks,
    unsupported_reported: bool,
    /// How far into the graph's change journal it has looked (see
    /// `fyrox::scene::graph::Graph::changes_since`); [`None`] before its first look.
    journal: Option<u64>,
    /// What each mesh node gave the scene traced when last looked at, by its slot in the graph.
    contribs: Vec<Option<Contrib>>,
    /// Nodes to look at again next frame whatever the journal says (see `prepare`).
    revisit: FxHashSet<Handle<Node>>,
    /// Bent geometry wanting bending, by node, until the budget gets to it.
    bend_wanted: FxHashMap<Handle<Node>, Vec<BendJob>>,
    /// Where the look over a slice of every node a frame has got to (see [`SWEEP`]).
    sweep: u32,
}

/// Bent geometry to bend: which, whose blend shapes (by surface data), by what, and how far it
/// has moved since it was last bent, as seen from the camera (meters over meters away).
type BendJob = (Shape, u64, Vec<f32>, f32);

/// What a mesh node gave the scene traced when last looked at: its surfaces placed and those put
/// in with the still meshes; and, for one that bends, where it was and how far off each of its
/// surfaces' bends shows (its sway times its scale), and how many showed - to look at it again
/// when the camera crosses one of those distances.
struct Contrib {
    handle: Handle<Node>,
    placed: Vec<Placed>,
    fixed: Vec<Fixed>,
    position: Vector3<f32>,
    reaches: Vec<f32>,
    class: usize,
    /// The weights each of its bent surfaces' shadow geometry was last bent by, by surface.
    bent: Vec<(usize, Vec<f32>)>,
}

/// How many of a bending mesh's surfaces show their bend from `distance` meters, given each one's
/// sway times its scale: as `prepare` decides it.
fn bend_class(reaches: &[f32], distance: f32) -> usize {
    if distance >= BEND_REACH {
        return 0;
    }
    reaches.iter().filter(|&&reach| reach >= distance * BEND_SHOWS).count()
}

/// Whether what `material` blocks is known for good: loaded, and its diffuse texture with it -
/// else it is traced solid for now, and must be looked at again once they are.
fn material_settled(material: &MaterialResource) -> bool {
    let state = material.state();
    let Some(material) = state.data_ref() else {
        return false;
    };
    material
        .texture_ref("diffuseTexture")
        .and_then(|binding| binding.value.clone())
        .is_none_or(|texture| texture.is_ok())
}

/// How many of the graph's node slots are looked at each frame besides those its change journal
/// names: changes the journal does not record - made through reflection, say - are caught within
/// a few seconds.
const SWEEP: u32 = 256;

impl std::fmt::Debug for TracedLightShadows {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TracedLightShadows")
            .field("settings", &self.settings)
            .field("built", &self.scene.borrow().is_some())
            .field("geometry", &self.geometry.len())
            .finish()
    }
}

impl TracedLightShadows {
    pub(crate) fn new(
        settings: RayTracedShadows,
        scene: SharedScene,
        switch: crate::Switch,
        half_resolution: crate::Switch,
        turn_over_frames: bool,
    ) -> Self {
        Self {
            settings,
            turn_over_frames,
            tracer: None,
            scene,
            switch,
            half_resolution,
            scene_handle: None,
            geometry: Default::default(),
            places: Default::default(),
            frame: 0,
            placed: Default::default(),
            was_placed: Default::default(),
            batches: Default::default(),
            fixed: Default::default(),
            was_fixed: Default::default(),
            skins: Default::default(),
            poses: Default::default(),
            bends: Default::default(),
            morphs: Default::default(),
            poser: None,
            corners: Default::default(),
            masks: Default::default(),
            journal: None,
            contribs: Vec::new(),
            revisit: Default::default(),
            bend_wanted: Default::default(),
            sweep: 0,
            unsupported_reported: false,
        }
    }
}

impl LightShadowTracer for TracedLightShadows {
    fn prepare(
        &mut self,
        server: &dyn GraphicsServer,
        scene_handle: Handle<Scene>,
        scene: &Scene,
    ) -> Result<(), FrameworkError> {
        let Some(server) = server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(());
        };
        if !server.ray_tracing {
            if !self.unsupported_reported {
                Log::warn(
                    "Ray traced shadows: this graphics adapter has no ray tracing, \
                     leaving shadows to shadow maps.",
                );
                self.unsupported_reported = true;
            }
            return Ok(());
        }

        // Switched off, there is nothing to trace against: everything built is let go, to be built
        // again when it is switched on.
        if !self.switch.is_enabled() {
            if self.scene_handle.take().is_some() {
                *self.scene.borrow_mut() = None;
            }
            return Ok(());
        }
        if self.scene_handle != Some(scene_handle) {
            *self.scene.borrow_mut() = None;
            self.geometry.clear();
            self.places.clear();
            self.was_placed.clear();
            self.batches.clear();
            self.was_fixed.clear();
            self.skins.clear();
            self.poses.clear();
            self.bends.clear();
            self.morphs.clear();
            self.corners.clear();
            self.masks.clear();
            self.journal = None;
            self.contribs.clear();
            self.revisit.clear();
            self.bend_wanted.clear();
            self.scene_handle = Some(scene_handle);
        }
        let graph = &scene.graph;
        // Where the camera is, to pose only the skinned meshes near enough to tell.
        let camera = graph.linear_iter().find_map(|node| {
            node.cast::<Camera>()
                .filter(|camera| camera.is_enabled())
                .map(|camera| camera.global_position())
        });

        // What each surface that casts shadows is traced with, where, and which surface it is; and
        // posed vertices.
        // Many surfaces share a material, so each material is asked once a frame what it blocks.
        let mut placed = std::mem::take(&mut self.placed);
        placed.clear();
        let mut fixed = std::mem::take(&mut self.fixed);
        fixed.clear();
        let mut posed: FxHashMap<Shape, Vec<f32>> = FxHashMap::default();
        // Bent geometry to bend this frame: which, whose blend shapes (by surface data), by what,
        // and how far it has moved, in meters, since it was last bent.
        let mut new_bent = 0;
        let mut what_blocks: FxHashMap<u64, Option<Blocks>> = FxHashMap::default();
        prune(&mut self.masks);
        self.frame += 1;
        let frame = self.frame;

        // Which nodes to look at: every one, the first time or when the graph's change journal no
        // longer reaches back to the last look; else those it names (moved, shown, hidden, bent,
        // come or gone), those to look at again (see `revisit`), bending ones the camera has
        // brought across where their bends show or stop showing, and a slice of every slot in
        // turn - looking at every node every frame, most of them standing still, took
        // milliseconds.
        let journaled = self.journal.and_then(|cursor| graph.changes_since(cursor));
        let mut visit: Vec<Handle<Node>> = Vec::new();
        match journaled {
            None => {
                self.contribs.clear();
                self.revisit.clear();
                self.bend_wanted.clear();
                visit.extend(graph.pair_iter().map(|(handle, _)| handle));
            }
            Some(changes) => {
                // A change only to a node's weights matters only where its bend shows; elsewhere
                // - most of the swaying forest - it is passed over.
                let kinds = graph.change_kinds_since(self.journal.unwrap_or(0)).unwrap_or(&[]);
                let mut set: FxHashSet<Handle<Node>> = FxHashSet::default();
                for (i, &node) in changes.iter().enumerate() {
                    if kinds.get(i) == Some(&NodeChange::Weights) {
                        let kept = self.contribs.get(node.index() as usize).and_then(Option::as_ref);
                        if kept.is_some_and(|c| c.handle == node && c.class == 0) {
                            continue;
                        }
                    }
                    set.insert(node);
                }
                set.extend(self.revisit.drain());
                for contrib in self.contribs.iter().flatten() {
                    if !contrib.reaches.is_empty() {
                        let distance = camera.map_or(0.0, |camera| (contrib.position - camera).norm());
                        if bend_class(&contrib.reaches, distance) != contrib.class {
                            set.insert(contrib.handle);
                        }
                    }
                }
                let capacity = graph.capacity();
                for _ in 0..SWEEP.min(capacity) {
                    self.sweep = (self.sweep + 1) % capacity.max(1);
                    let handle = graph.handle_from_index(self.sweep);
                    if handle.is_some() {
                        set.insert(handle);
                    }
                }
                visit.extend(set);
            }
        }
        self.journal = Some(graph.changes_end());
        for handle in visit {
            let slot = handle.index() as usize;
            if slot >= self.contribs.len() {
                self.contribs.resize_with(slot + 1, || None);
            }
            // What was kept for this slot goes, if it was this node's or a node's now gone; the
            // weights its surfaces were last bent by stay with it.
            let mut carried: Vec<(usize, Vec<f32>)> = Vec::new();
            if self.contribs[slot]
                .as_ref()
                .is_some_and(|c| c.handle == handle || graph.try_get_node(c.handle).is_err())
            {
                if let Some(old) = self.contribs[slot].take() {
                    self.bend_wanted.remove(&old.handle);
                    if old.handle == handle {
                        carried = old.bent;
                    }
                }
            }
            let Ok(node) = graph.try_get_node(handle) else {
                continue;
            };
            let contrib: Option<Contrib> = 'node: {
            let Some(mesh) = node.cast::<Mesh>() else {
                break 'node None;
            };
            let transform = node.global_transform();
            let slot = handle.index() as usize;
            if slot >= self.places.len() {
                self.places.resize(slot + 1, None);
            }
            let place = match &mut self.places[slot] {
                Some(place) if place.handle == handle => {
                    if !same_place(&place.transform, &transform) {
                        place.transform = transform;
                        place.still_since = frame;
                        place.moved = true;
                    }
                    *place
                }
                other => *other.insert(Place {
                    handle,
                    transform,
                    still_since: frame,
                    moved: false,
                }),
            };
            // Only what has never moved is put in with the rest: what has been carried about -
            // the eyes on a droid's head, say - would come and go from it, each time putting
            // its cube together again, when it lies still a while and when it goes.
            let settled = !place.moved && frame - place.still_since >= SETTLE_FRAMES;
            if !node.cast_shadows() {
                break 'node None;
            }
            let skinned = mesh.surfaces().iter().any(|s| !s.bones().is_empty());
            if (skinned || place.moved) && !node.global_visibility() {
                // Hidden: looked at again when it is shown (which the journal records).
                break 'node None;
            }
            let mut out = Contrib {
                handle,
                placed: Vec::new(),
                fixed: Vec::new(),
                position: node.global_position(),
                reaches: Vec::new(),
                class: 0,
                bent: carried,
            };
            let mut wanted: Vec<BendJob> = Vec::new();
            // Whether to look at it again next frame whatever happens: posed every frame near the
            // camera; a material still loading; bent geometry not yet given its own; settling.
            let mut again = skinned;
            // Bent by its blend shapes: its weights, 1 at full strength. A mesh with blend shapes
            // counts as bending even with them all at nothing - a tree too far off to sway, say -
            // so that its weights coming and going put no cube together again.
            let weights: Vec<f32> = mesh.blend_shapes().iter().map(|s| s.weight / 100.0).collect();
            let bending = !weights.is_empty();
            let distance = camera.map_or(0.0, |camera| (node.global_position() - camera).norm());
            let bend_near = bending && distance < BEND_REACH;
            // How much the mesh is scaled up, at most: its drift is in its own space.
            let scale = (0..3).map(|c| transform.fixed_view::<3, 1>(0, c).norm()).fold(0.0, f32::max);
            for (n, surface) in mesh.surfaces().iter().enumerate() {
                let material = surface.material();
                again |= !material_settled(material);
                let masks = &mut self.masks;
                let Some(how) = *what_blocks
                    .entry(material.key())
                    .or_insert_with(|| blocks(material, masks))
                else {
                    continue;
                };
                let key = surface.data().key();
                if bending {
                    // How far off its bend shows, kept to tell when the camera crosses it.
                    let reach = self
                        .bends
                        .entry(key)
                        .or_insert_with(|| Bend::read(surface))
                        .as_ref()
                        .map(|bend| bend.sway * scale);
                    if let Some(reach) = reach {
                        out.reaches.push(reach);
                    }
                }
                let shows = bend_near
                    && self
                        .bends
                        .entry(key)
                        .or_insert_with(|| Bend::read(surface))
                        .as_ref()
                        .is_some_and(|bend| bend.sway * scale >= distance * BEND_SHOWS);
                let has_morph = shows && {
                    let device = &server.state.device;
                    let bends = &mut self.bends;
                    self.morphs
                        .entry(key)
                        .or_insert_with(|| {
                            let bend = bends.entry(key).or_insert_with(|| Bend::read(surface)).as_ref()?;
                            RayTracedMorph::new(device, &bend.flat_positions(), &bend.flat_offsets(), bend.offsets.len() as u32)
                        })
                        .is_some()
                };
                // Geometry of its own only within this frame's share of new ones (see
                // BEND_NEW_PER_FRAME); else, for now, the shadow of its mesh unbent.
                let has_morph = has_morph
                    && (self.geometry.contains_key(&Shape::Posed(handle, n)) || {
                        new_bent += 1;
                        again |= new_bent > BEND_NEW_PER_FRAME;
                        new_bent <= BEND_NEW_PER_FRAME
                    });
                if surface.bones().is_empty() && has_morph {
                    // Built unbent the first time, then bent on the GPU once it has moved enough to show.
                    let shape = Shape::Posed(handle, n);
                    let last = out.bent.iter().find(|(m, _)| *m == n).map(|(_, w)| w.as_slice());
                    if let Some(Some(bend)) = self.bends.get(&key) {
                        let drift = bend.drift(last, &weights) * scale;
                        if drift >= BEND_LEAST.max(distance * BEND_LEAST_PER_METER) {
                            // Ranked by the angle it has moved through: the nearest first.
                            wanted.push((shape, key, weights.clone(), drift / distance.max(1.0)));
                        }
                    }
                    out.placed.push((shape, instance_transform(&transform), handle, n, how));
                    continue;
                }
                if surface.bones().is_empty() {
                    let batchable = !bending && !matches!(how, Blocks::Cutout(_));
                    if settled && batchable {
                        // Where it was last taken to be, not exactly where it is: a drift too
                        // small to count as moving should not put its cube together again.
                        let at = instance_transform(&place.transform);
                        out.fixed.push((BatchKey::new(&place.transform, how), (handle, n, at, surface.data().key())));
                        continue;
                    }
                    // Still settling: looked at again until it has, to be put with the rest.
                    again |= batchable && !place.moved;
                    let shape = Shape::Shared(surface.data().key(), how.into());
                    out.placed.push((shape, instance_transform(&transform), handle, n, how));
                } else {
                    let shape = Shape::Posed(handle, n);
                    // Posed afresh near the camera, or for the first time; further off, as it was.
                    // Either way, only if its bones have moved since it was last posed.
                    let built = self.geometry.contains_key(&shape);
                    let near = camera.is_none_or(|camera| (node.global_position() - camera).norm() < SKIN_REACH);
                    if near || !built {
                        let matrices = bone_matrices(graph, surface);
                        let still = built && self.poses.get(&shape).is_some_and(|was| same_pose(was, &matrices));
                        if !still {
                            let skin = self.skins.entry(surface.data().key()).or_insert_with(|| Skin::read(surface));
                            posed.insert(shape, skin.pose(&matrices));
                            self.poses.insert(shape, matrices);
                        }
                    }
                    let at = instance_transform(&Matrix4::identity());
                    out.placed.push((shape, at, handle, n, how));
                }
            }
            out.class = bend_class(&out.reaches, distance);
            if again {
                self.revisit.insert(handle);
            }
            if wanted.is_empty() {
                self.bend_wanted.remove(&handle);
            } else {
                self.bend_wanted.insert(handle, wanted);
            }
            Some(out)
            };
            self.contribs[slot] = contrib;
        }

        // Everything kept, in the order of the graph's slots, as when every node was looked at.
        for contrib in self.contribs.iter().flatten() {
            placed.extend_from_slice(&contrib.placed);
            fixed.extend_from_slice(&contrib.fixed);
        }
        // The bent geometry wanting bending most, this frame's budget of it below.
        let mut bend_jobs: Vec<BendJob> = self.bend_wanted.values().flatten().cloned().collect();

        // The still meshes put together again where one has come or gone.
        let mut rebuilt = 0;
        if fixed != self.was_fixed {
            let mut by_key: FxHashMap<BatchKey, Vec<Member>> = FxHashMap::default();
            for &(key, member) in &fixed {
                by_key.entry(key).or_default().push(member);
            }
            self.batches.retain(|key, _| by_key.contains_key(key));
            // Each surface's data read once, however many meshes share it.
            let mut read: FxHashMap<u64, (Vec<f32>, Vec<u32>)> = FxHashMap::default();
            for (key, members) in by_key {
                if self.batches.get(&key).is_some_and(|batch| batch.members == members) {
                    continue;
                }
                self.batches.remove(&key);
                let (mut vertices, mut indices) = (Vec::new(), Vec::new());
                for &(handle, n, at, _) in &members {
                    let Some(surface) = graph
                        .try_get_node(handle)
                        .ok()
                        .and_then(|node| node.cast::<Mesh>())
                        .and_then(|mesh| mesh.surfaces().get(n))
                    else {
                        continue;
                    };
                    let (local, triangles) = read.entry(surface.data().key()).or_insert_with(|| {
                        let local = positions(graph, surface);
                        let triangles = triangles(surface, local.len() / 3);
                        (local, triangles)
                    });
                    append_placed(&mut vertices, &mut indices, local, triangles, &at);
                }
                if indices.is_empty() {
                    continue;
                }
                let geometry = match key.glass {
                    None => server.build_ray_traced_geometry(&vertices, &indices, false)?,
                    Some(_) => server.build_ray_traced_glass(&vertices, &indices, false)?,
                };
                if let Some(geometry) = geometry {
                    self.batches.insert(key, Batch { members, geometry });
                    rebuilt += 1;
                }
            }
        }
        let fixed_same = rebuilt == 0 && fixed == self.was_fixed;
        self.fixed = std::mem::replace(&mut self.was_fixed, fixed);

        // Everything where it was last frame, and nothing posed afresh, the geometry is as it was.
        let same = placed == self.was_placed;
        let settled = same && posed.is_empty();

        // Geometry for shapes seen for the first time, and posed ones whose vertices changed in
        // number.
        let posed_vertices = |shape: &Shape| posed.get(shape);
        let mut built = 0;
        for &(shape, _, handle, n, how) in placed.iter().filter(|_| !settled) {
            let fresh = match self.geometry.get(&shape) {
                None => true,
                Some(had) => posed_vertices(&shape)
                    .is_some_and(|v| v.len() != had.vertex_count() as usize * 3),
            };
            if !fresh {
                continue;
            }
            let Some(surface) = graph
                .try_get_node(handle)
                .ok()
                .and_then(|node| node.cast::<Mesh>())
                .and_then(|mesh| mesh.surfaces().get(n))
            else {
                continue;
            };
            // A bent mesh's copy starts unbent, as its blend shapes were read: no need to read
            // its vertices again for every copy.
            let unbent = match (shape, surface.bones().is_empty()) {
                (Shape::Posed(..), true) => self.bends.get(&surface.data().key()).and_then(|b| b.as_ref()).map(Bend::flat_positions),
                _ => None,
            };
            let vertices = match (posed_vertices(&shape), unbent) {
                (Some(vertices), _) => vertices.clone(),
                (None, Some(unbent)) => unbent,
                (None, None) => positions(graph, surface),
            };
            let indices = triangles(surface, vertices.len() / 3);
            let updatable = matches!(shape, Shape::Posed(..));
            let mask = match how {
                Blocks::Cutout(key) => self.masks.get(&key).and_then(|(mask, _)| mask.clone()),
                _ => None,
            };
            let geometry = match (how, mask) {
                (Blocks::Cutout(_), Some(mask)) => {
                    let corners = self
                        .corners
                        .entry(surface.data().key())
                        .or_insert_with(|| {
                            let uvs = texture_coordinates(surface, vertices.len() / 3);
                            WgpuGraphicsServer::cutout_corners(&indices, &uvs)
                        })
                        .clone();
                    server.build_ray_traced_cutout_sharing(&vertices, &indices, corners, mask, updatable)?
                }
                (Blocks::Glass(_), _) => server.build_ray_traced_glass(&vertices, &indices, updatable)?,
                _ => server.build_ray_traced_geometry(&vertices, &indices, updatable)?,
            };
            if let Some(geometry) = geometry {
                // Built afresh, unbent: bent again from nothing.
                if let Shape::Posed(node, n) = shape {
                    if let Some(Some(contrib)) = self.contribs.get_mut(node.index() as usize) {
                        if contrib.handle == node {
                            contrib.bent.retain(|(m, _)| *m != n);
                        }
                    }
                }
                self.geometry.insert(shape, geometry);
                built += 1;
            }
        }
        // Only what is still there.
        if !same {
            let there: FxHashSet<Shape> = placed.iter().map(|(shape, ..)| *shape).collect();
            self.geometry.retain(|shape, _| there.contains(shape));
            // A pose is kept only with the geometry it was put into: built afresh, it is posed
            // afresh.
            let geometry = &self.geometry;
            self.poses.retain(|shape, _| geometry.contains_key(shape));
        }

        // Bent geometry, bent on the GPU and refitted.
        let poser = self.poser.get_or_insert_with(|| RayTracedPoser::new(&server.state.device));
        bend_jobs.retain(|(shape, ..)| self.geometry.contains_key(shape));
        if bend_jobs.len() > BEND_BUDGET {
            bend_jobs.select_nth_unstable_by(BEND_BUDGET, |a, b| b.3.total_cmp(&a.3));
            bend_jobs.truncate(BEND_BUDGET);
        }
        let mut bends: Vec<(&RayTracedGeometry, &RayTracedMorph, &[f32])> = Vec::with_capacity(bend_jobs.len());
        for (shape, key, weights, _) in &bend_jobs {
            if let (Some(geometry), Some(Some(morph))) = (self.geometry.get(shape), self.morphs.get(key)) {
                bends.push((geometry, morph, weights.as_slice()));
            }
        }
        poser.pose(server, &bends)?;
        let bent_any = !bends.is_empty();
        for (shape, _, weights, _) in bend_jobs {
            // Bent: no longer wanted, until its weights change again; and bent by these.
            if let Shape::Posed(node, n) = shape {
                if let Some(Some(contrib)) = self.contribs.get_mut(node.index() as usize) {
                    if contrib.handle == node {
                        match contrib.bent.iter_mut().find(|(m, _)| *m == n) {
                            Some((_, was)) => was.clone_from(&weights),
                            None => contrib.bent.push((n, weights.clone())),
                        }
                    }
                }
                if let Some(wanted) = self.bend_wanted.get_mut(&node) {
                    wanted.retain(|job| job.0 != shape);
                    if wanted.is_empty() {
                        self.bend_wanted.remove(&node);
                    }
                }
            }
        }
        if !same {
            let kept: Vec<&RayTracedGeometry> = self.geometry.values().collect();
            poser.forget_others(&kept);
        }

        // Posed geometry, refitted where it is this frame.
        let updates: Vec<(&RayTracedGeometry, &[f32])> = posed
            .iter()
            .filter_map(|(shape, vertices)| {
                let geometry = self.geometry.get(shape)?;
                (vertices.len() == geometry.vertex_count() as usize * 3)
                    .then_some((geometry, vertices.as_slice()))
            })
            .collect();
        server.update_ray_traced_geometry(&updates)?;

        // Nothing placed, built or refitted differently from last frame, the scene traced is
        // still right as it is.
        let unchanged = same
            && fixed_same
            && built == 0
            && updates.is_empty()
            && !bent_any
            && self.scene.borrow().is_some();
        // What was placed this frame is kept to tell against the next; last frame's, to be filled
        // again.
        self.placed = std::mem::replace(&mut self.was_placed, placed);
        if unchanged {
            if self.tracer.is_none() {
                self.tracer = server.create_shadow_tracer();
            }
            return Ok(());
        }
        let placed = &self.was_placed;

        let identity = instance_transform(&Matrix4::identity());
        let batched = self.batches.iter().map(|(key, batch)| RayTracedInstance {
            geometry: &batch.geometry,
            transform: identity,
            lets_through: key.lets_through(),
        });
        let instances: Vec<RayTracedInstance> = batched
            .chain(placed.iter().filter_map(|(shape, transform, _, _, how)| {
                let geometry = self.geometry.get(shape)?;
                // Posed geometry built solid stays so even if its surface turns to glass.
                let lets_through = match how {
                    Blocks::Glass(through) if geometry.is_see_through() => *through,
                    _ => [0.0; 3],
                };
                Some(RayTracedInstance {
                    geometry,
                    transform: *transform,
                    lets_through,
                })
            }))
            .collect();
        let mut shared = self.scene.borrow_mut();
        match shared.as_mut() {
            Some(traced) => server.update_ray_traced_instances(traced, &instances)?,
            None => *shared = server.build_ray_traced_instances(&instances)?,
        }
        if built > 0 || rebuilt > 0 {
            if let Some(traced) = shared.as_ref() {
                Log::info(format!(
                    "Ray traced shadows: {} pieces of geometry, {} of still meshes ({} put together \
                     again), {} copies, {} triangles",
                    self.geometry.len(),
                    self.batches.len(),
                    rebuilt,
                    instances.len(),
                    traced.triangle_count()
                ));
            }
        }

        if self.tracer.is_none() {
            self.tracer = server.create_shadow_tracer();
        }
        Ok(())
    }

    fn trace(
        &mut self,
        ctx: LightShadowTraceContext,
    ) -> Result<Option<GpuTexture>, FrameworkError> {
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(None);
        };
        let shared = self.scene.borrow();
        let (Some(scene), Some(tracer)) = (shared.as_ref(), self.tracer.as_mut()) else {
            return Ok(None);
        };

        let light = match ctx.light.kind {
            LightSourceKind::Directional { .. } => {
                // The renderer lights with the up vector as the direction towards the sun, so
                // the sun's light travels the other way.
                let Some(towards_sun) = ctx.light.up_vector.try_normalize(f32::EPSILON) else {
                    return Ok(None);
                };
                let direction = -towards_sun;
                ShadowRayLight::Directional {
                    direction: [direction.x, direction.y, direction.z],
                    reach: self.settings.reach,
                    angular_size: self.settings.sun_angular_radius.to_radians().tan(),
                }
            }
            LightSourceKind::Point { .. } | LightSourceKind::Spot { .. } => {
                let position = ctx.light.position;
                let size = if matches!(ctx.light.kind, LightSourceKind::Point { .. }) {
                    self.settings.point_light_size
                } else {
                    self.settings.spot_light_size
                };
                ShadowRayLight::Positional {
                    position: [position.x, position.y, position.z],
                    radius: ctx.light_radius,
                    size,
                }
            }
            LightSourceKind::Unknown => return Ok(None),
        };

        // The renderer's scissor box has its origin at the bottom left; the tracer's at the top.
        let scissor = ctx.scissor.map(|b| {
            let top = ctx.viewport.size.y - b.y - b.height;
            [
                b.x.max(0) as u32,
                top.max(0) as u32,
                b.width.max(0) as u32,
                b.height.max(0) as u32,
            ]
        });

        tracer.trace(
            server,
            scene,
            ctx.depth,
            ctx.normals,
            ctx.materials,
            ShadowRayParameters {
                inverse_view_projection: matrix_to_array(&ctx.inv_view_projection),
                light,
                bias: self.settings.bias,
                // This backend stores render targets top row first.
                flip_v: true,
                scissor,
                samples: self.settings.samples,
                frame: self.turn_over_frames.then_some(self.frame as u32),
                half_resolution: self.settings.half_resolution && self.half_resolution.is_enabled(),
                light_key: (u64::from(ctx.light.handle.generation()) << 32) | u64::from(ctx.light.handle.index()),
            },
        )
    }
}

fn matrix_to_array(matrix: &Matrix4<f32>) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (column, values) in matrix.column_iter().zip(out.iter_mut()) {
        values.copy_from_slice(column.as_slice());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_resource_is_made_with_the_key_of_one_pinned() {
        use fyrox::resource::texture::TextureResource;
        let texture = |size| fyrox::asset::Resource::new_embedded(texture(size, |_, _| 255));
        let first: TextureResource = texture(4);
        let (key, pin) = (first.key(), Pin::of(&first));
        drop(first);
        assert!(pin.gone());
        for _ in 0..1000 {
            assert_ne!(texture(4).key(), key, "a texture made where the pinned one was");
        }
    }

    /// A `size` by `size` RGBA texture, its alpha `alpha(x, y)`.
    fn texture(size: u32, alpha: impl Fn(u32, u32) -> u8) -> fyrox::resource::texture::Texture {
        use fyrox::resource::texture::{Texture, TextureKind, TexturePixelKind};
        let mut bytes = Vec::new();
        for y in 0..size {
            for x in 0..size {
                bytes.extend_from_slice(&[40, 120, 30, alpha(x, y)]);
            }
        }
        Texture::from_bytes(TextureKind::Rectangle { width: size, height: size }, TexturePixelKind::RGBA8, bytes)
            .unwrap()
    }

    #[test]
    fn a_cutout_is_solid_where_the_renderer_draws() {
        // A leaf in the left half; the right half left undrawn.
        let leaf = texture(8, |x, _| if x < 4 { 255 } else { 0 });
        let mask = cutout_mask(&leaf, 1.0).expect("a cutout");
        assert_eq!((mask.width(), mask.height()), (8, 8));
        for y in 0..8 {
            for x in 0..8 {
                assert_eq!(mask.is_solid(x, y), x < 4, "{x}, {y}");
            }
        }
        // Texels at 0.6 are drawn, but not once the diffuse colour halves them.
        let faint = texture(4, |_, _| 153);
        assert!(cutout_mask(&faint, 1.0).is_none());
        assert!(cutout_mask(&faint, 0.5).is_some());
    }

    #[test]
    fn a_sheet_made_double_sided_is_traced_once() {
        use fyrox::{
            core::{algebra::Matrix4, math::TriangleDefinition},
            scene::mesh::surface::{SurfaceData, SurfaceResource},
        };
        // A quad, and a reversed copy of each of its triangles, on the same vertices - and on
        // copies of them, as a game doubling foliage makes them.
        let mut data = SurfaceData::make_quad(&Matrix4::identity());
        let count = data.vertex_buffer.vertex_count();
        let raw = data.vertex_buffer.raw_data().to_vec();
        let size = data.vertex_buffer.vertex_size() as usize;
        {
            let mut vertices = data.vertex_buffer.modify();
            for vertex in raw.chunks_exact(size) {
                let _ = vertices.push_vertex_raw(vertex);
            }
        }
        let front: Vec<[u32; 3]> = data.geometry_buffer.iter().map(|t| t.0).collect();
        {
            let mut triangles = data.geometry_buffer.modify();
            for t in &front {
                triangles.push(TriangleDefinition([t[0], t[2], t[1]]));
                triangles.push(TriangleDefinition([t[0] + count, t[2] + count, t[1] + count]));
            }
        }
        let surface = Surface::new(SurfaceResource::new_embedded(data));
        let traced = triangles(&surface, 2 * count as usize);
        assert_eq!(traced.len(), front.len() * 3, "each place once");
        assert_eq!(&traced[..], front.iter().flatten().copied().collect::<Vec<_>>().as_slice(), "the first of each kept");
    }

    #[test]
    fn a_surface_with_blend_shapes_is_bent_by_its_weights() {
        use fyrox::{
            core::{algebra::Matrix4, algebra::Vector3 as V3, num_traits::Zero},
            fxhash::FxHashMap,
            scene::mesh::surface::{BlendShapesContainer, InputBlendShapeData, SurfaceData, SurfaceResource},
        };
        // A quad; one shape lifts every vertex a meter, the other moves the first one along x.
        let mut data = SurfaceData::make_quad(&Matrix4::identity());
        let count = data.vertex_buffer.vertex_count();
        let f = half::f16::from_f32;
        let shape = |name: &str, at: &dyn Fn(u32) -> V3<f32>| InputBlendShapeData {
            default_weight: 0.0,
            name: name.to_owned(),
            positions: (0..count).map(|v| (v, at(v).map(f))).filter(|(_, p)| !p.iter().all(|c| c.is_zero())).collect(),
            normals: FxHashMap::default(),
            tangents: FxHashMap::default(),
        };
        let shapes = [
            shape("lift", &|_| V3::new(0.0, 1.0, 0.0)),
            shape("slide", &|v| if v == 0 { V3::new(2.0, 0.0, 0.0) } else { V3::zeros() }),
        ];
        data.blend_shapes_container = Some(BlendShapesContainer::from_lists(&data.vertex_buffer, &shapes));
        let unbent: Vec<f32> = {
            let mut out = Vec::new();
            for v in data.vertex_buffer.iter() {
                let p = v.read_3_f32(VertexAttributeUsage::Position).unwrap();
                out.extend([p.x, p.y, p.z]);
            }
            out
        };
        let surface = Surface::new(SurfaceResource::new_embedded(data));
        let bend = Bend::read(&surface).expect("blend shapes read");
        assert_eq!(bend.pose(&[0.0, 0.0]), unbent, "at no weight, unbent");
        let bent = bend.pose(&[0.5, 0.25]);
        for v in 0..count as usize {
            let slide = if v == 0 { 0.5 } else { 0.0 };
            assert_eq!(bent[v * 3], unbent[v * 3] + slide, "vertex {v} x");
            assert_eq!(bent[v * 3 + 1], unbent[v * 3 + 1] + 0.5, "vertex {v} y");
            assert_eq!(bent[v * 3 + 2], unbent[v * 3 + 2], "vertex {v} z");
        }
    }

    #[test]
    fn a_texture_drawn_all_over_is_traced_solid() {
        assert!(cutout_mask(&texture(4, |_, _| 255), 1.0).is_none());
    }

    #[test]
    fn a_render_target_is_traced_solid() {
        // A camera's picture shown on a screen (Maze's race monitors): its texels are only on the
        // GPU, so there is nothing to read, and it must not panic trying.
        use fyrox::resource::texture::{TextureResource, TextureResourceExtension};
        let picture = TextureResource::new_render_target(512, 288);
        assert!(cutout_mask(&picture.data_ref(), 1.0).is_none());
    }

    #[test]
    fn a_big_texture_gives_a_smaller_mask() {
        let mask = cutout_mask(&texture(2048, |x, _| if x < 1024 { 255 } else { 0 }), 1.0).unwrap();
        assert_eq!((mask.width(), mask.height()), (1024, 1024));
        assert!(mask.is_solid(511, 3) && !mask.is_solid(512, 3));
    }

    #[test]
    fn bones_that_have_not_moved_need_no_posing_again() {
        let still = vec![Matrix4::new_translation(&Vector3::new(1.0, 2.0, 3.0)), Matrix4::identity()];
        assert!(same_pose(&still, &still.clone()));
        // A drift too small to see is no move.
        let drift = vec![Matrix4::new_translation(&Vector3::new(1.0 + MOVED * 0.5, 2.0, 3.0)), Matrix4::identity()];
        assert!(same_pose(&still, &drift));
        // But a bone that has moved is, and so is a different set of bones.
        let moved = vec![Matrix4::new_translation(&Vector3::new(1.1, 2.0, 3.0)), Matrix4::identity()];
        assert!(!same_pose(&still, &moved));
        assert!(!same_pose(&still, &still[..1]));
    }

    #[test]
    fn still_meshes_are_put_together_where_they_stand() {
        let mut vertices = Vec::new();
        let mut indices = Vec::new();
        let triangle = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let at = |x: f32| instance_transform(&Matrix4::new_translation(&Vector3::new(x, 0.0, 5.0)));
        append_placed(&mut vertices, &mut indices, &triangle, &[0, 1, 2], &at(10.0));
        append_placed(&mut vertices, &mut indices, &triangle, &[0, 1, 2], &at(20.0));
        // The second copy's indices follow on from the first's vertices, and each is moved.
        assert_eq!(indices, [0, 1, 2, 3, 4, 5]);
        assert_eq!(&vertices[..3], &[10.0, 0.0, 5.0]);
        assert_eq!(&vertices[9..12], &[20.0, 0.0, 5.0]);
        assert_eq!(&vertices[12..15], &[21.0, 0.0, 5.0]);
    }

    #[test]
    fn still_meshes_go_together_by_cube_and_by_what_they_block() {
        let at = |x: f32, y: f32| Matrix4::new_translation(&Vector3::new(x, y, -1.0));
        let solid = BatchKey::new(&at(1.0, 2.0), Blocks::Solid);
        assert_eq!(solid, BatchKey::new(&at(BATCH_CELL - 0.5, 2.0), Blocks::Solid));
        assert_ne!(solid, BatchKey::new(&at(BATCH_CELL + 0.5, 2.0), Blocks::Solid));
        // Floors over one another are apart, and so is everything below zero.
        assert_ne!(solid, BatchKey::new(&at(1.0, BATCH_CELL + 2.0), Blocks::Solid));
        assert_eq!(solid.cell[2], -1);
        // Glass is apart from what is solid, and from glass letting through other light, and
        // keeps what it lets through.
        let red = BatchKey::new(&at(1.0, 2.0), Blocks::Glass([1.0, 0.2, 0.2]));
        assert_ne!(solid, red);
        assert_ne!(red, BatchKey::new(&at(1.0, 2.0), Blocks::Glass([0.2, 1.0, 0.2])));
        assert_eq!(red.lets_through(), [1.0, 0.2, 0.2]);
        assert_eq!(solid.lets_through(), [0.0; 3]);
    }

    #[test]
    fn a_skin_is_posed_by_the_matrices_it_is_given() {
        let skin = Skin {
            positions: vec![Vector3::new(1.0, 0.0, 0.0)],
            bones: vec![[0, 1, 0, 0]],
            weights: vec![[0.5, 0.5, 0.0, 0.0]],
        };
        let matrices = [Matrix4::new_translation(&Vector3::new(0.0, 2.0, 0.0)), Matrix4::identity()];
        let posed = skin.pose(&matrices);
        assert!((posed[0] - 1.0).abs() < 1.0e-6 && (posed[1] - 1.0).abs() < 1.0e-6 && posed[2].abs() < 1.0e-6, "{posed:?}");
    }
}
