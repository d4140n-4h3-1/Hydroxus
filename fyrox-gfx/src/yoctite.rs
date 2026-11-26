//! Yoctite in the renderer: the scene's meshes kept in Yoctite's geometry pool, and drawn from it
//! cluster by cluster, at the level of detail each cluster needs from the camera.
//!
//! Every surface is built into clusters at every level of detail once (`yoctite::build`), by its
//! data, so the thousand copies of a tree share one - on threads of its own, so a scene full of
//! new meshes does not stop the frame while they build; until a mesh's surfaces are built, the
//! renderer draws it as it would without Yoctite. Its vertices and clusters go into the pool
//! (`yoctite_wgpu::GeometryPool`), and each mesh node's surfaces are placed in
//! it as instances, which move with their nodes and go when they do. Each frame, the clusters to
//! draw are chosen - instances out of the view left out, then for each the cut through its
//! clusters whose error looks no bigger than a pixel - on the GPU
//! (`yoctite_wgpu::Culler`), without the CPU seeing the list.
//!
//! [`Mode::Render`] draws them into the G-Buffer, a draw for each material, in place of the
//! renderer's own draw of those meshes (`Renderer::set_taken_over_meshes`): lit, shadowed,
//! fogged and smoothed as everything else is. Only what it can draw as the engine does is taken
//! over: visible meshes, not skinned, drawn deferred, every surface's material of the engine's
//! standard or glTF shader, without a lightmap. The rest stays the renderer's.
//!
//! Meshes with blend shapes are taken over too, and bend as the renderer would bend them: their
//! offsets go into the pool with their vertices, and each node's weights are handed to its
//! instances whenever they change - a forest swaying in the wind, at every level of detail. Their
//! bounds are grown by as far as weights of up to [`MOST_WEIGHT`] can move them.
//!
//! With Candelite's surface view ([`YoctitePass::show_candelite`]), each mesh it uploads is also
//! given Candelite's cards, captured into its surface cache, and its triangles to trace against
//! (the `candelite` crate); and over the frame is laid what a ray from the eye through
//! each pixel finds on the cards - magenta where it hit something no card saw.
//!
//! [`Mode::View`] instead lays the clusters over the frame, each in a colour of its own: to see
//! them, and which levels of detail are drawn. It can choose them on the CPU (`yoctite::select`)
//! as well, the reference the GPU's choice is held to.

use fyrox::{
    core::{
        algebra::{Matrix4, Point3, Vector2, Vector3},
        color::Color,
        log::Log,
        pool::Handle,
        sstorage::ImmutableString,
    },
    fxhash::{FxHashMap, FxHashSet},
    graph::SceneGraph,
    graphics::{
        error::FrameworkError, gpu_program::SamplerFallback, gpu_program::ShaderPropertyKind,
        gpu_program::ShaderResourceKind,
        gpu_texture::{GpuTexture, GpuTextureDescriptor, GpuTextureKind, PixelKind},
        server::GraphicsServer,
        sampler::GpuSampler, stats::RenderPassStatistics,
    },
    material::{shader::Shader, Material, MaterialProperty, MaterialResource},
    renderer::{
        cache::shader::{binding, property, PropertyGroup, RenderMaterial, RenderPassContainer},
        make_viewport_matrix,
        observer::Observer,
        resources::RendererResources,
        cache::texture::TextureCache,
        GBufferPassContext, SceneRenderPass, SceneRenderPassContext, TakenOverMeshes,
    },
    asset::manager::ResourceManager,
    scene::{
        graph::Graph,
        mesh::{
            buffer::{VertexAttributeUsage, VertexReadTrait},
            surface::Surface,
            Mesh, RenderPath,
        },
        light::{directional::DirectionalLight, BaseLight},
        node::Node,
        Scene,
    },
};
use candelite::{self, Bounds, Lamp, ProbeFrame, RadianceCache, Reflections, ScreenProbes, SurfaceCache, SurfaceLight, SurfaceView, Sunlight, TraceScene, ViewShows};
use fyrox_graphics_wgpu::{server::WgpuGraphicsServer, texture::WgpuTexture};
use yoctite_wgpu::{
    resolve_materials, Capture, ClusterView, CullParameters, Culler, GBufferDrawer, GeometryPool,
    InstanceId, MaterialParams, MeshId, Morphs, PoolVertex, YoctiteMaterial,
};
use std::{
    any::TypeId,
    collections::hash_map::Entry,
    sync::{mpsc, Arc, Mutex},
    time::Instant,
};
use yoctite::ClusterDag;

/// Lays the cluster view over the frame where it drew anything.
const WGSL: &str = include_str!("shaders/yoctite_view_wgsl.shader");

/// The largest blend shape weight the bounds of a bending mesh's clusters allow for, either way
/// (1 is a shape at full strength): a mesh bent further may have clusters culled while still in
/// view. The bounds allow for every shape at this at once, which shapes that take turns - a
/// sway's sine and cosine - never are, so a little past it is still safe. Bigger bounds cull
/// less: at 2, a swaying forest drew a sixth more triangles.
pub const MOST_WEIGHT: f32 = 1.5;

/// How many frames apart the statistics are logged.
const STATS_EVERY: u64 = 300;

/// What Yoctite does in the renderer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// Draws the meshes it can into the G-Buffer, in place of the renderer.
    Render,
    /// Lays the clusters over the frame in colours of their own, chosen on the GPU or, `on_gpu`
    /// false, on the CPU.
    View {
        /// Whether the clusters are chosen on the GPU.
        on_gpu: bool,
    },
}

/// A surface's clusters, in a scene's pool and as built.
struct Built {
    mesh: MeshId,
    dag: Arc<ClusterDag>,
}

/// A mesh node's surface placed in the pool.
struct Placed {
    instance: InstanceId,
    /// Its surface's data, by key.
    data: u64,
    /// Its material, by key, whose slot it is drawn with.
    material: u64,
}

/// A mesh node Yoctite draws, or will once its surfaces are built: what was found of it, and
/// where it was placed. Unchanged, a node costs a look at its keys and its transform each frame:
/// finding out what it is locks every surface's data and every material.
struct Drawn {
    /// Its surfaces' data and materials, by key, and how each is shaded.
    keys: Vec<(u64, u64)>,
    shadings: Vec<Shading>,
    /// Its surfaces placed in the pool, in order: none until all of them are built.
    placed: Vec<Placed>,
    /// Where it was placed, to tell when it moves.
    transform: Matrix4<f32>,
    /// The last frame it was seen in, to take out what has gone.
    seen: u64,
    /// The weights last given its blend shapes, to tell when they change.
    weights: Vec<f32>,
}

/// How a material is drawn into the G-Buffer: as the engine's glTF shader does, or its standard
/// shader.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Shading {
    Gltf,
    Standard,
}

/// A material in use, and the slot its instances are numbered with.
struct Slot {
    slot: u32,
    material: MaterialResource,
    shading: Shading,
    /// How many placed surfaces are drawn with it; the slot is given back at none.
    users: u32,
}

/// The materials drawn with, by key, and their slots.
#[derive(Default)]
struct Materials {
    by_key: FxHashMap<u64, Slot>,
    free: Vec<u32>,
    next: u32,
    /// Which diffuse textures have see-through texels: their materials are thin, and let light
    /// through (see [`LEAF_TRANSMISSION`]).
    masks: crate::raytraced_shadows::Masks,
}

/// How much of the light falling on the back of a cutout material (leaves, needles, grass) comes
/// through it: written into the G-Buffer for the sun's light pass and Candelite. A real leaf lets
/// through about as much as it reflects; this is a little under that, since a cutout card often
/// stands for several leaves.
pub const LEAF_TRANSMISSION: f32 = 0.5;

impl Materials {
    /// The slot of `material`, given one if it has none, counted as used once more.
    fn take(&mut self, material: &MaterialResource, shading: Shading) -> u32 {
        let Self { by_key, free, next, .. } = self;
        let slot = by_key.entry(material.key()).or_insert_with(|| Slot {
            slot: free.pop().unwrap_or_else(|| {
                *next += 1;
                *next - 1
            }),
            material: material.clone(),
            shading,
            users: 0,
        });
        slot.users += 1;
        slot.slot
    }

    /// Counts `key`'s material as used once less, giving its slot back at none.
    fn give_back(&mut self, key: u64) {
        if let Some(slot) = self.by_key.get_mut(&key) {
            slot.users -= 1;
            if slot.users == 0 {
                self.free.push(slot.slot);
                self.by_key.remove(&key);
            }
        }
    }

    #[cfg(test)]
    fn slot(&self, key: u64) -> u32 {
        self.by_key.get(&key).map_or(0, |slot| slot.slot)
    }

    /// How many slots there are, in use or not.
    fn slots(&self) -> u32 {
        self.next
    }
}

/// A surface's geometry, copied out to be built on another thread.
struct Job {
    key: u64,
    positions: Vec<[f32; 3]>,
    indices: Vec<u32>,
    uvs: Vec<[f32; 2]>,
}

/// A surface built: its clusters, how many triangles it had, and how long it took.
struct Done {
    key: u64,
    dag: ClusterDag,
    triangles: usize,
    seconds: f64,
}

/// The most threads surfaces are built on: building is only ever a burst as a scene loads.
const MAX_BUILDERS: usize = 4;

/// Builds surfaces' clusters on threads of its own, and keeps what it built, by the surface
/// data's key, for every scene: a surface in two scenes is built once.
struct Builder {
    jobs: mpsc::Sender<Job>,
    done: mpsc::Receiver<Done>,
    /// Each surface built, or [`None`] while it builds.
    dags: FxHashMap<u64, Option<Arc<ClusterDag>>>,
    /// Since it last had nothing to build: how many surfaces it built, of how many triangles,
    /// how long building them took the threads between them, and when it started.
    batch: (usize, usize, f64, Option<Instant>),
    threads: usize,
}

impl Builder {
    fn new() -> Self {
        let (jobs, queue) = mpsc::channel::<Job>();
        let (finished, done) = mpsc::channel();
        let queue = Arc::new(Mutex::new(queue));
        let threads = std::thread::available_parallelism()
            .map_or(1, |n| n.get().saturating_sub(1))
            .clamp(1, MAX_BUILDERS);
        for n in 0..threads {
            let queue = queue.clone();
            let finished = finished.clone();
            let spawned = std::thread::Builder::new()
                .name(format!("yoctite-build-{n}"))
                .spawn(move || loop {
                    // Ends when the pass, and the sender with it, is gone.
                    let Ok(job) = queue.lock().map_err(drop).and_then(|q| q.recv().map_err(drop)) else {
                        return;
                    };
                    let started = Instant::now();
                    // With its texture coordinates, so a level of detail that drags its texture
                    // across it counts the drag as error.
                    let dag = yoctite::build(
                        &yoctite::Mesh::new(&job.positions, &job.indices).with_texture_coordinates(&job.uvs),
                        Default::default(),
                    );
                    let done = Done {
                        key: job.key,
                        dag,
                        triangles: job.indices.len() / 3,
                        seconds: started.elapsed().as_secs_f64(),
                    };
                    if finished.send(done).is_err() {
                        return;
                    }
                });
            if let Err(err) = spawned {
                Log::err(format!("Yoctite could not start a thread to build on: {err}"));
            }
        }
        Self {
            jobs,
            done,
            dags: Default::default(),
            batch: (0, 0, 0.0, None),
            threads,
        }
    }

    /// Takes in what the threads have built since last time.
    fn collect(&mut self) {
        while let Ok(done) = self.done.try_recv() {
            self.dags.insert(done.key, Some(Arc::new(done.dag)));
            self.batch.0 += 1;
            self.batch.1 += done.triangles;
            self.batch.2 += done.seconds;
        }
        if let (count @ 1.., triangles, seconds, Some(started)) = self.batch {
            if self.dags.values().all(Option::is_some) {
                Log::info(format!(
                    "Yoctite: built {count} surfaces ({triangles} triangles) in {:.2} s, {seconds:.2} s of building \
                     on {} threads",
                    started.elapsed().as_secs_f64(),
                    self.threads
                ));
                self.batch = (0, 0, 0.0, None);
            }
        }
    }

    /// Lets go of what was built for surfaces none of `scenes` has in its pool any more; those
    /// still building are kept.
    fn forget_unused<'a>(&mut self, scenes: impl Iterator<Item = &'a SceneMeshes>) {
        let used: FxHashSet<u64> = scenes.flat_map(|scene| scene.built.keys().copied()).collect();
        self.dags.retain(|key, dag| dag.is_none() || used.contains(key));
    }

    /// `surface`'s clusters, if they are built; if not, set building.
    fn dag(&mut self, surface: &Surface) -> Option<Arc<ClusterDag>> {
        let key = surface.data().key();
        match self.dags.entry(key) {
            Entry::Occupied(built) => built.get().clone(),
            Entry::Vacant(place) => {
                let (vertices, indices) = read_surface(surface);
                let job = Job {
                    key,
                    positions: vertices.iter().map(|v| v.position).collect(),
                    uvs: vertices.iter().map(|v| v.uv).collect(),
                    indices,
                };
                if self.jobs.send(job).is_ok() {
                    place.insert(None);
                    self.batch.3.get_or_insert_with(Instant::now);
                }
                None
            }
        }
    }
}

/// One scene's meshes in Yoctite: a pool, and the choice of what to draw from it, of its own.
/// Node handles are only told apart within a scene, and what one scene's camera draws is only
/// that scene's.
#[derive(Default)]
struct SceneMeshes {
    pool: Option<GeometryPool>,
    culler: Option<Culler>,
    view: Option<ClusterView>,
    /// The surfaces in the pool, by their data's key.
    built: FxHashMap<u64, Built>,
    /// Each mesh node it draws, or will.
    nodes: FxHashMap<Handle<Node>, Drawn>,
    /// The mesh nodes placed, which the renderer leaves out of its own draw.
    taken_over: FxHashSet<Handle<Node>>,
    materials: Materials,
    /// Candelite, if it is shown: see [`YoctitePass::show_candelite`].
    candelite: Option<Candelite>,
    /// How many times the scene has been drawn.
    frame: u64,
    overflow_reported: bool,
    /// How far into the graph's change journal it has looked (see
    /// `fyrox::scene::graph::Graph::changes_since`); [`None`] before its first look.
    journal: Option<u64>,
    /// Mesh nodes waiting to be placed: their surfaces building, or their materials loading.
    waiting: FxHashSet<Handle<Node>>,
    /// Where the look over a slice of every node a frame has got to (see [`SWEEP`]).
    sweep: u32,
}

/// How many of the graph's node slots are looked at each frame besides those its change journal
/// names: changes the journal does not record - made through reflection, say - are caught within
/// a few seconds.
const SWEEP: u32 = 256;

/// A scene's Candelite: its surface cache, what it traces against, the captures waiting to be
/// drawn into the cache, and its view.
struct Candelite {
    cache: SurfaceCache,
    scene: TraceScene,
    pending: Vec<Capture>,
    view: Option<SurfaceView>,
    /// The light on the cards: see `candelite::SurfaceLight`.
    light: Option<SurfaceLight>,
    /// The screen probes, and the light between surfaces they gather, for the renderer to light
    /// with: see [`YoctitePass::light_with_candelite`].
    probes: Option<ScreenProbes>,
    radiance: Option<RadianceCache>,
    indirect: Option<GpuTexture>,
    /// The reflections, and the light they bring each pixel, for the renderer to light its
    /// specular term with.
    reflections: Option<Reflections>,
    specular: Option<GpuTexture>,
    target: Option<GpuTexture>,
    full_reported: bool,
    /// The cutout masks made so far, by texture: foliage is traced as a cutout.
    masks: crate::raytraced_shadows::Masks,
}

/// About how long lighting Candelite's cards may take on the GPU a frame, in milliseconds: its
/// budget of texels is steered towards it.
const CARDS_LIT_MS: f32 = 2.5;
/// The fewest texels Candelite's cards are lit a frame, however long they take: new cards still
/// come in.
const CARDS_LIT_LEAST: u32 = 8192;

/// About how long tracing Candelite's radiance cache may take on the GPU a frame, in milliseconds.
const CACHE_MS: f32 = 1.0;
/// The fewest probes of Candelite's radiance cache traced a frame.
const CACHE_LEAST: usize = 32;

use crate::support::ATLAS_SIZE;

/// Yoctite in the renderer: see the module's description. [`crate::GraphicsEffects::with_yoctite`]
/// and [`crate::GraphicsEffects::with_yoctite_view`] add it.
pub struct YoctitePass {
    source_type_id: TypeId,
    mode: Mode,
    /// The meshes it draws, scene by scene, which the renderer leaves out of its own draw.
    taken_over: Option<TakenOverMeshes>,
    drawer: Option<GBufferDrawer>,
    shader: Option<RenderPassContainer>,
    /// Builds every surface, once for all the scenes; made at the first frame.
    builder: Option<Builder>,
    scenes: FxHashMap<Handle<Scene>, SceneMeshes>,
    failed: bool,
    /// Whether Candelite is kept (its cards captured, its scene traced); whether its surface
    /// view is laid over the frame, and what it shows; and where the light between surfaces it
    /// gathers and the reflections it traces go, if it lights the frame.
    candelite: bool,
    candelite_overlay: bool,
    candelite_shows: ViewShows,
    /// What the light Candelite gives the renderer is multiplied by: see
    /// [`Self::set_candelite_strength`].
    candelite_strength: f32,
    indirect: Option<(fyrox::renderer::IndirectDiffuse, fyrox::renderer::IndirectSpecular)>,
    /// The game's switches for Yoctite and for Candelite, if it has them, and whether each was on
    /// as of the last frame (see [`Self::follow_switches`]).
    yoctite_switch: Option<crate::Switch>,
    yoctite_on: bool,
    candelite_switch: Option<crate::Switch>,
    candelite_on: bool,
}

impl YoctitePass {
    /// The pass, in `mode`; drawing in place of the renderer, it tells the renderer which meshes
    /// it draws through `taken_over`.
    pub(crate) fn new(source_type_id: TypeId, mode: Mode, taken_over: Option<TakenOverMeshes>) -> Self {
        Self {
            source_type_id,
            mode,
            taken_over,
            drawer: None,
            shader: None,
            builder: None,
            scenes: Default::default(),
            failed: false,
            candelite: false,
            candelite_overlay: false,
            candelite_shows: ViewShows::Shaded,
            candelite_strength: 1.0,
            indirect: None,
            yoctite_switch: None,
            yoctite_on: true,
            candelite_switch: None,
            candelite_on: true,
        }
    }

    /// Lets the game switch Yoctite off, and on again, while it runs.
    pub fn follow_yoctite_switch(&mut self, switch: crate::Switch) {
        self.yoctite_switch = Some(switch);
    }

    /// Lets the game switch Candelite off, and on again, while it runs.
    pub fn follow_candelite_switch(&mut self, switch: crate::Switch) {
        self.candelite_switch = Some(switch);
    }

    /// Follows the game's switches, if one was flipped since the last frame. Candelite is on only
    /// while Yoctite is: it lights and traces Yoctite's geometry.
    ///
    /// Yoctite off, every scene is let go - its geometry, and Candelite's cards and traced scene -
    /// and the renderer draws every mesh itself again, lit with its flat ambient colour. Candelite
    /// off, its cards and traced scenes are let go, and the light it gathered taken back from the
    /// renderer. Either on again, the scenes start again: a mesh is given its cards as it is
    /// uploaded, so every mesh is uploaded again (its clusters kept by the builder), the renderer
    /// drawing them meanwhile.
    fn follow_switches(&mut self) {
        let yoctite = self.yoctite_switch.as_ref().is_none_or(crate::Switch::is_enabled);
        let candelite = yoctite && self.candelite_switch.as_ref().is_none_or(crate::Switch::is_enabled);
        if yoctite == self.yoctite_on && candelite == self.candelite_on {
            return;
        }
        let restart = yoctite != self.yoctite_on || candelite;
        self.yoctite_on = yoctite;
        self.candelite_on = candelite;
        if let Some((indirect, specular)) = &self.indirect {
            indirect.borrow_mut().clear();
            specular.borrow_mut().clear();
        }
        if restart {
            if let Some(taken_over) = &self.taken_over {
                taken_over.borrow_mut().clear();
            }
            self.scenes.clear();
        } else {
            for scene in self.scenes.values_mut() {
                scene.candelite = None;
            }
        }
    }

    /// Turns Candelite off for good, after it failed: its cards and traced scenes dropped, and the
    /// light it gathered taken back from the renderer, which lights every scene with its flat
    /// ambient colour again. Yoctite keeps drawing.
    fn turn_off_candelite(&mut self) {
        Log::warn("Candelite is off from now on: the scene is lit with its flat ambient colour.");
        self.candelite = false;
        self.candelite_overlay = false;
        if let Some((indirect, specular)) = self.indirect.take() {
            indirect.borrow_mut().clear();
            specular.borrow_mut().clear();
        }
        for scene in self.scenes.values_mut() {
            scene.candelite = None;
        }
    }

    /// Gives each mesh drawn Candelite's cards and triangles to trace, and lays over the frame
    /// what a ray from the eye finds on the cards (see the module's description). Only drawing in
    /// place of the renderer, and where rays can be traced.
    pub fn show_candelite(&mut self) {
        self.candelite = true;
        self.candelite_overlay = true;
    }

    /// Lights the frame with Candelite's global illumination: each frame its cards are lit (the
    /// sun with traced shadows, and the light between surfaces), and screen probes gather from
    /// them the light reaching every pixel, put in `indirect` under the scene's handle for the
    /// renderer to light with in place of the ambient colour (see
    /// `fyrox::renderer::Renderer::set_indirect_diffuse`); and reflections are traced, put in
    /// `specular` likewise (see `fyrox::renderer::Renderer::set_indirect_specular`). Only drawing
    /// in place of the renderer, and where rays can be traced.
    pub fn light_with_candelite(&mut self, indirect: fyrox::renderer::IndirectDiffuse, specular: fyrox::renderer::IndirectSpecular) {
        self.candelite = true;
        self.indirect = Some((indirect, specular));
    }

    /// Multiplies the light between surfaces Candelite gives the renderer by `strength` (1, as
    /// it is, by default): see `candelite::ScreenProbes::set_strength`.
    pub fn set_candelite_strength(&mut self, strength: f32) {
        self.candelite_strength = strength;
    }

    /// As [`Self::show_candelite`], and lights the cards - the sun with traced shadows, and the
    /// light between surfaces - laying over the frame the light leaving what each ray from the eye
    /// hit: Candelite's global illumination as its rays see it.
    pub fn show_candelite_light(&mut self) {
        self.candelite = true;
        self.candelite_overlay = true;
        self.candelite_shows = ViewShows::Lit;
    }
}

/// A surface's vertices as the pool keeps them, and its triangles' indices.
fn read_surface(surface: &Surface) -> (Vec<PoolVertex>, Vec<u32>) {
    let data = surface.data();
    let data = data.data_ref();
    let vertices = data
        .vertex_buffer
        .iter()
        .map(|v| PoolVertex {
            position: v.read_3_f32(VertexAttributeUsage::Position).unwrap_or_default().into(),
            normal: v.read_3_f32(VertexAttributeUsage::Normal).unwrap_or_default().into(),
            uv: v.read_2_f32(VertexAttributeUsage::TexCoord0).unwrap_or_default().into(),
            tangent: v.read_4_f32(VertexAttributeUsage::Tangent).unwrap_or_default().into(),
        })
        .collect::<Vec<_>>();
    let count = vertices.len() as u32;
    let indices = data
        .geometry_buffer
        .iter()
        .filter(|t| t.0.iter().all(|&i| i < count))
        .flat_map(|t| t.0)
        .collect();
    (vertices, indices)
}

/// A mesh node's blend shape weights, 1 at full strength.
fn mesh_weights(node: &Node) -> Vec<f32> {
    node.cast::<Mesh>()
        .map(|mesh| mesh.blend_shapes().iter().map(|s| s.weight / 100.0).collect())
        .unwrap_or_default()
}

/// A surface's blend shapes, if it has them: how many, and for each of its first `count`
/// vertices, how far each moves its position and normal at full strength.
///
/// They are read from the volume texture the engine packs them into for its shaders
/// (`BlendShapesContainer::from_lists`): a layer a shape, in which each vertex's position, normal
/// and tangent offsets follow one another as three texels of three half floats, vertex after
/// vertex, in rows of up to 512.
pub(crate) fn read_morphs(surface: &Surface, count: usize) -> Option<(u32, Vec<Vec<[f32; 6]>>)> {
    let data = surface.data();
    let data = data.data_ref();
    let container = data.blend_shapes_container.as_ref()?;
    let storage = container.blend_shape_storage.as_ref()?;
    let n = data.vertex_buffer.vertex_count() as usize;
    let targets = container.blend_shapes.len();
    if n == 0 || targets == 0 {
        return None;
    }
    let width = n.min(512);
    let height = n.div_ceil(width);
    let texture = storage.data_ref();
    let bytes = texture.data();
    if bytes.len() < targets * width * height * 18 {
        Log::warn("Yoctite: a surface's blend shapes are not packed as expected; it is drawn unbent");
        return None;
    }
    let half_at = |at: usize| half::f16::from_le_bytes([bytes[at], bytes[at + 1]]).to_f32();
    let offsets = (0..count)
        .map(|vertex| {
            (0..targets)
                .map(|layer| {
                    if vertex >= n {
                        return [0.0; 6];
                    }
                    let at = (layer * width * height + vertex) * 18;
                    std::array::from_fn(|i| half_at(at + 2 * i))
                })
                .collect()
        })
        .collect();
    Some((targets as u32, offsets))
}

/// `m`'s top three rows, row by row, as an instance is placed with.
fn rows(m: &Matrix4<f32>) -> [f32; 12] {
    let mut out = [0.0; 12];
    for row in 0..3 {
        for column in 0..4 {
            out[row * 4 + column] = m[(row, column)];
        }
    }
    out
}

/// Whether a sphere is at least partly inside the frustum of `view_projection` (OpenGL clip
/// space), by its six planes.
fn in_view(view_projection: &Matrix4<f32>, center: Vector3<f32>, radius: f32) -> bool {
    let row = |i: usize| view_projection.row(i).transpose();
    let (r0, r1, r2, r3) = (row(0), row(1), row(2), row(3));
    [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 + r2, r3 - r2].iter().all(|plane| {
        let normal = Vector3::new(plane.x, plane.y, plane.z);
        let length = normal.norm();
        length == 0.0 || (normal.dot(&center) + plane.w) / length >= -radius
    })
}

/// How `material` is drawn into the G-Buffer, if Yoctite can draw it as the engine does: a
/// loaded material of the engine's glTF or standard shader, without a lightmap.
fn shading(material: &MaterialResource) -> Option<Shading> {
    let state = material.state();
    let material = state.data_ref()?;
    let shader = material.shader().state();
    let shader = shader.data_ref()?;
    let shading = match shader.definition.name.as_str() {
        "GLTFShader" => Shading::Gltf,
        "StandardShader" => Shading::Standard,
        _ => return None,
    };
    let lightmapped = material
        .texture_ref("lightmapTexture")
        .is_some_and(|binding| binding.value.is_some());
    (!lightmapped).then_some(shading)
}

/// `material`'s value of the property `name`: its own, or else its shader's default.
fn value_of(material: &Material, name: &str) -> Option<MaterialProperty> {
    if let Some(own) = material
        .property_group_ref(ImmutableString::new("properties"))
        .and_then(|group| group.property_ref(name))
    {
        return Some(own.clone());
    }
    let shader = material.shader().state();
    let shader = shader.data_ref()?;
    shader.definition.resources.iter().find_map(|resource| match &resource.kind {
        ShaderResourceKind::PropertyGroup(properties) => properties.iter().find(|p| p.name.as_str() == name).and_then(
            |p| match &p.kind {
                ShaderPropertyKind::Float { value } => Some(MaterialProperty::Float(*value)),
                ShaderPropertyKind::UInt { value } => Some(MaterialProperty::UInt(*value)),
                ShaderPropertyKind::Vector2 { value } => Some(MaterialProperty::Vector2(*value)),
                ShaderPropertyKind::Vector3 { value } => Some(MaterialProperty::Vector3(*value)),
                ShaderPropertyKind::Color { r, g, b, a } => Some(MaterialProperty::Color(Color::from_rgba(*r, *g, *b, *a))),
                _ => None,
            },
        ),
        _ => None,
    })
}

/// The GPU texture `material` binds as `name`, with its sampler: its own, once loaded, or else
/// the stub its shader falls back to, as the renderer does.
fn texture_of(
    material: &Material,
    name: &str,
    texture_cache: &mut TextureCache,
    server: &dyn fyrox::graphics::server::GraphicsServer,
    resource_manager: &ResourceManager,
    resources: &RendererResources,
) -> (GpuTexture, GpuSampler) {
    if let Some(texture) = material.texture_ref(name).and_then(|binding| binding.value.clone()) {
        if let Some(data) = texture_cache.get(server, resource_manager, &texture) {
            return (data.gpu_texture.clone(), data.gpu_sampler.clone());
        }
    }
    let fallback = material
        .shader()
        .state()
        .data_ref()
        .and_then(|shader| {
            shader.definition.resources.iter().find_map(|resource| match &resource.kind {
                ShaderResourceKind::Texture { fallback, .. } if resource.name.as_str() == name => Some(*fallback),
                _ => None,
            })
        })
        .unwrap_or(SamplerFallback::White);
    let stub = match fallback {
        SamplerFallback::Black => &resources.black_dummy,
        SamplerFallback::Normal => &resources.normal_dummy,
        SamplerFallback::White | SamplerFallback::Volume => &resources.white_dummy,
    };
    (stub.clone(), resources.linear_wrap_sampler.clone())
}

/// A material in a slot, as Yoctite's G-Buffer draw reads it.
fn resolve(slot: &Slot, masks: &mut crate::raytraced_shadows::Masks, ctx: &mut GBufferPassContext) -> Option<YoctiteMaterial> {
    let state = slot.material.state();
    let material = state.data_ref()?;
    let transmission = if crate::raytraced_shadows::cutout(material, masks).is_some() { LEAF_TRANSMISSION } else { 0.0 };
    let float = |name, default: f32| match value_of(material, name) {
        Some(MaterialProperty::Float(v)) => v,
        _ => default,
    };
    let diffuse_color = match value_of(material, "diffuseColor") {
        Some(MaterialProperty::Color(c)) => c.as_frgba().into(),
        _ => [1.0; 4],
    };
    let emission_strength = match value_of(material, "emissionStrength") {
        Some(MaterialProperty::Vector3(v)) => v.into(),
        _ => [0.0; 3],
    };
    let tex_coord_scale = match value_of(material, "texCoordScale") {
        Some(MaterialProperty::Vector2(v)) => v.into(),
        _ => [1.0; 2],
    };
    let layer_index = match value_of(material, "layerIndex") {
        Some(MaterialProperty::UInt(v)) => v,
        _ => 0,
    };
    // The glTF shader reads metallic and roughness from the blue and green of one texture,
    // times factors; the standard shader from the red of one each.
    let (metallic, roughness, metallic_channel, roughness_channel, metallic_factor, roughness_factor) = match slot.shading {
        Shading::Gltf => (
            "metallicRoughnessTexture",
            "metallicRoughnessTexture",
            2,
            1,
            float("metallicFactor", 1.0),
            float("roughnessFactor", 1.0),
        ),
        Shading::Standard => ("metallicTexture", "roughnessTexture", 0, 0, 1.0, 1.0),
    };
    let mut texture = |name: &str| {
        texture_of(material, name, ctx.texture_cache, ctx.server, ctx.resource_manager, ctx.renderer_resources)
    };
    let textures = [
        texture("diffuseTexture"),
        texture("normalTexture"),
        texture(metallic),
        texture(roughness),
        texture("aoTexture"),
        texture("emissionTexture"),
    ];
    Some(YoctiteMaterial {
        slot: slot.slot,
        params: MaterialParams {
            diffuse_color,
            emission_strength,
            metallic_factor,
            tex_coord_scale,
            roughness_factor,
            layer_index,
            metallic_channel,
            roughness_channel,
            transmission,
            ..Default::default()
        },
        textures,
    })
}

/// An error as big as a pixel, as the observer sees: the view's height in radians over its height
/// in pixels.
fn pixel_threshold(observer: &Observer) -> f32 {
    let fov = 2.0 * (1.0 / observer.position.projection_matrix[(1, 1)]).atan();
    fov / observer.viewport.h().max(1) as f32
}

/// The observer's view and projection, column by column.
fn view_projection_of(observer: &Observer) -> [[f32; 4]; 4] {
    let mut out = [[0.0; 4]; 4];
    for (column, values) in observer.position.view_projection_matrix.column_iter().zip(out.iter_mut()) {
        values.copy_from_slice(column.as_slice());
    }
    out
}

impl SceneMeshes {
    /// Keeps the pool in step with the scene's meshes: new surfaces uploaded once built, new
    /// mesh nodes placed once all their surfaces are, moved ones moved, gone ones taken out.
    /// Drawing in place of the renderer, only meshes whose materials it can draw are placed, and
    /// each with its material's slot; those still building are left to the renderer. Surfaces are
    /// built by `builder`, once for every scene. Whether the mesh nodes placed
    /// ([`Self::taken_over`]) changed.
    fn sync(&mut self, server: &WgpuGraphicsServer, graph: &Graph, mode: Mode, builder: &mut Builder, candelite: bool) -> bool {
        let device = &server.state.device;
        let queue = &server.state.queue;
        let rendering = mode == Mode::Render;
        if candelite && rendering && self.candelite.is_none() && candelite::can_trace(device) {
            self.candelite = Some(Candelite {
                cache: SurfaceCache::new(device, ATLAS_SIZE),
                scene: TraceScene::new(),
                pending: Vec::new(),
                view: SurfaceView::new(device),
                light: SurfaceLight::new(device),
                probes: None,
                radiance: None,
                indirect: None,
                reflections: None,
                specular: None,
                target: None,
                full_reported: false,
                masks: Default::default(),
            });
        }
        let Self {
            pool,
            built,
            nodes,
            taken_over,
            materials,
            candelite,
            frame,
            journal,
            waiting,
            sweep,
            ..
        } = self;
        let pool = pool.get_or_insert_with(|| GeometryPool::new(device, 1 << 16));
        *frame += 1;
        let frame = *frame;
        // Gives a node's surfaces' room in the pool, and their materials' slots, back.
        let release = |pool: &mut GeometryPool, materials: &mut Materials, drawn: &Drawn| {
            for placed in &drawn.placed {
                pool.remove_instance(queue, placed.instance);
                if rendering {
                    materials.give_back(placed.material);
                }
            }
        };
        let mut uploaded = 0;
        let mut bending = 0;
        let mut changed = false;
        // Which nodes to look at: every one, the first time or when the graph's change journal
        // no longer reaches back to the last look; else those it names, those waiting, and a
        // slice of every slot in turn.
        let journaled = journal.and_then(|cursor| graph.changes_since(cursor));
        let full = journaled.is_none();
        let candidates: Vec<Handle<Node>> = match journaled {
            None => graph.pair_iter().map(|(handle, _)| handle).collect(),
            Some(changes) => {
                let mut set: FxHashSet<Handle<Node>> = changes.iter().copied().collect();
                set.extend(waiting.iter().copied());
                let capacity = graph.capacity();
                for _ in 0..SWEEP.min(capacity) {
                    *sweep = (*sweep + 1) % capacity.max(1);
                    let handle = graph.handle_from_index(*sweep);
                    if handle.is_some() {
                        set.insert(handle);
                    }
                }
                set.into_iter().collect()
            }
        };
        *journal = Some(graph.changes_end());
        // Nodes looked at that are no longer drawn by it: gone, hidden, drawn forward.
        let mut gone: Vec<Handle<Node>> = Vec::new();
        for handle in candidates {
            let Ok(node) = graph.try_get_node(handle) else {
                gone.push(handle);
                continue;
            };
            let Some(mesh) = node.cast::<Mesh>() else {
                gone.push(handle);
                continue;
            };
            // Those hidden and those drawn forward stay the renderer's.
            if !node.global_visibility() || mesh.render_path() != RenderPath::Deferred {
                gone.push(handle);
                continue;
            }
            let surfaces = mesh.surfaces();
            let same = |drawn: &Drawn| {
                drawn.keys.len() == surfaces.len()
                    && drawn.keys.iter().zip(surfaces).all(|(k, s)| *k == (s.data().key(), s.material().key()))
            };
            if !nodes.get(&handle).is_some_and(same) {
                // New, or given other surfaces or materials: what was placed of it goes.
                if let Some(old) = nodes.remove(&handle) {
                    release(pool, materials, &old);
                    changed |= taken_over.remove(&handle);
                }
                // Skinned meshes stay the renderer's, and so do those with a material Yoctite
                // cannot draw as the engine does - or not yet: one still loading is looked into
                // again next frame.
                let shadings: Option<Vec<Shading>> = if surfaces.iter().any(|s| !s.bones().is_empty()) {
                    None
                } else if rendering {
                    surfaces.iter().map(|s| shading(s.material())).collect()
                } else {
                    Some(vec![Shading::Standard; surfaces.len()])
                };
                let Some(shadings) = shadings else {
                    // Skinned, or its material not drawable; or loading, to be looked into again.
                    if !surfaces.iter().any(|s| !s.bones().is_empty()) {
                        waiting.insert(handle);
                    }
                    continue;
                };
                nodes.insert(
                    handle,
                    Drawn {
                        keys: surfaces.iter().map(|s| (s.data().key(), s.material().key())).collect(),
                        shadings,
                        placed: Vec::new(),
                        transform: Matrix4::identity(),
                        seen: frame,
                        weights: Vec::new(),
                    },
                );
            }
            let drawn = nodes.get_mut(&handle).expect("found or made above");
            drawn.seen = frame;
            let transform = node.global_transform();
            // The engine's weights are in percent.
            let shapes = mesh.blend_shapes();
            if !drawn.placed.is_empty()
                && (drawn.weights.len() != shapes.len()
                    || drawn.weights.iter().zip(shapes).any(|(w, s)| *w != s.weight / 100.0))
            {
                drawn.weights = shapes.iter().map(|s| s.weight / 100.0).collect();
                for placed in &drawn.placed {
                    pool.set_weights(placed.instance, &drawn.weights);
                }

            }
            if !drawn.placed.is_empty() {
                if drawn.transform != transform {
                    for placed in &drawn.placed {
                        pool.move_instance(queue, placed.instance, rows(&transform));
                    }
                    drawn.transform = transform;
                }
                continue;
            }

            // Placed once every surface is built; each asked for, so all of a mesh's build at
            // once.
            let mut ready = true;
            for surface in surfaces {
                ready &= built.contains_key(&surface.data().key()) || builder.dag(surface).is_some();
            }
            if !ready {
                waiting.insert(handle);
                continue;
            }
            waiting.remove(&handle);
            for (n, surface) in surfaces.iter().enumerate() {
                let data = surface.data().key();
                // A mesh uploaded now, for Candelite: its triangles and full-detail clusters.
                let mut fresh = None;
                let mesh = match built.entry(data) {
                    Entry::Occupied(built) => built.get().mesh,
                    Entry::Vacant(place) => {
                        let dag = builder.dag(surface).expect("built, as checked above");
                        let (mut vertices, indices) = read_surface(surface);
                        if candelite.is_some() {
                            let positions: Vec<[f32; 3]> = vertices.iter().map(|v| v.position).collect();
                            let full: Vec<usize> =
                                (0..dag.clusters.len()).filter(|&i| dag.clusters[i].level == 0).collect();
                            fresh = Some((positions, indices, full));
                        }
                        let morphs = read_morphs(surface, vertices.len());
                        // With the vertices its coarser levels made: grown foliage cards'
                        // corners, which bend as the vertices they copy do.
                        dag.extend_vertices(&mut vertices, |v, at| v.position = at);
                        let morphs = morphs.map(|(targets, mut offsets)| {
                            dag.extend_vertices(&mut offsets, |_, _| {});
                            (targets, offsets.concat())
                        });
                        uploaded += 1;
                        bending += morphs.is_some() as usize;
                        let mesh = pool.add_morphing_mesh(
                            device,
                            queue,
                            &dag,
                            &vertices,
                            morphs.as_ref().map(|(targets, offsets)| Morphs {
                                targets: *targets,
                                offsets,
                                most_weight: MOST_WEIGHT,
                            }),
                        );
                        place.insert(Built { mesh, dag }).mesh
                    }
                };
                let (material, slot) = if rendering {
                    (surface.material().key(), materials.take(surface.material(), drawn.shadings[n]))
                } else {
                    (0, 0)
                };
                if let (Some(candelite), Some((positions, indices, full))) = (candelite.as_mut(), fresh) {
                    // Foliage traced as a cutout: a ray passes where its texture is see-through.
                    let cutout = crate::raytraced_shadows::material_mask(surface.material(), &mut candelite.masks).map(|mask| {
                        let uvs: Vec<[f32; 2]> = crate::raytraced_shadows::texture_coordinates(surface, positions.len())
                            .chunks_exact(2)
                            .map(|uv| [uv[0], uv[1]])
                            .collect();
                        (uvs, mask)
                    });
                    candelite.scene.add_mesh(
                        device,
                        queue,
                        mesh,
                        &positions,
                        &indices,
                        cutout.as_ref().map(|(uvs, mask)| (uvs.as_slice(), mask.clone())),
                    );
                    let first = pool.mesh(mesh).map_or(0, |m| m.first_cluster);
                    let clusters: Vec<u32> = full.iter().map(|&i| first + i as u32).collect();
                    match candelite.cache.add_mesh(mesh, &Bounds::of(positions), &clusters, slot) {
                        Some(captures) => candelite.pending.extend(captures),
                        None if !candelite.full_reported => {
                            Log::warn("Candelite: its atlas is full; meshes past it have no cards");
                            candelite.full_reported = true;
                        }
                        None => {}
                    }
                }
                let instance = pool.add_instance(device, queue, mesh, rows(&transform), slot);
                drawn.weights = mesh_weights(node);
                pool.set_weights(instance, &drawn.weights);
                drawn.placed.push(Placed { instance, data, material });
            }
            drawn.transform = transform;
            changed |= taken_over.insert(handle);
        }

        // The weights set, to the GPU together.
        pool.flush(queue);

        // Nodes gone, hidden or no longer drawable: theirs goes too. Looking at every node, any not
        // seen; else those looked at and found so.
        let mut removed = false;
        if full {
            nodes.retain(|handle, drawn| {
                let keep = drawn.seen == frame;
                if !keep {
                    release(pool, materials, drawn);
                    changed |= taken_over.remove(handle);
                    removed = true;
                }
                keep
            });
            waiting.retain(|handle| nodes.contains_key(handle));
        } else {
            for handle in gone {
                waiting.remove(&handle);
                if let Some(drawn) = nodes.remove(&handle) {
                    release(pool, materials, &drawn);
                    changed |= taken_over.remove(&handle);
                    removed = true;
                }
            }
        }
        if removed {
            // Meshes nothing is placed from any more go, giving back their room in the pool.
            let used: FxHashSet<u64> = nodes.values().flat_map(|d| d.placed.iter().map(|p| p.data)).collect();
            built.retain(|data, built| {
                let keep = used.contains(data);
                if !keep {
                    pool.remove_mesh(queue, built.mesh);
                    if let Some(candelite) = candelite.as_mut() {
                        candelite.cache.remove_mesh(built.mesh);
                        candelite.scene.remove_mesh(built.mesh);
                        candelite.pending.retain(|capture| capture.mesh != built.mesh);
                    }
                }
                keep
            });
        }
        if uploaded > 0 {
            let stats = pool.stats();
            Log::info(format!(
                "Yoctite: {uploaded} surfaces uploaded, {bending} of them bending; the pool holds {} meshes, {} instances, {} clusters, {:.1} MB",
                stats.meshes,
                stats.instances,
                stats.clusters,
                stats.bytes as f64 / 1048576.0
            ));
        }
        changed
    }

    /// What the GPU chose, read back a frame or two late, as clusters and triangles; warning once
    /// if its list ran out of room.
    fn chosen(&mut self, what: &str) -> Option<(usize, usize)> {
        let culler = self.culler.as_ref()?;
        culler.stats().map(|stats| {
            if stats.overflowed && !self.overflow_reported {
                Log::warn(format!(
                    "Yoctite {what}: {} clusters chosen, more than the draw list held; it holds {} from now on",
                    stats.clusters,
                    culler.capacity()
                ));
                self.overflow_reported = true;
            }
            (stats.clusters as usize, stats.triangles as usize)
        })
    }
}

/// The clusters to draw from where the camera is, chosen on the CPU, as instance slots and pool
/// clusters, and how many triangles they come to: the reference for the GPU's choice.
fn choose(
    built: &FxHashMap<u64, Built>,
    nodes: &FxHashMap<Handle<Node>, Drawn>,
    pool: &GeometryPool,
    view_projection: &Matrix4<f32>,
    eye: Vector3<f32>,
    threshold: f32,
) -> (Vec<[u32; 2]>, usize) {
    let mut list = Vec::new();
    let mut triangles = 0;
    let by_mesh: FxHashMap<MeshId, &Built> = built.values().map(|b| (b.mesh, b)).collect();
    for (drawn, placed) in nodes.values().flat_map(|d| d.placed.iter().map(move |p| (d, p))) {
        let Some(record) = pool.instance(placed.instance) else {
            continue;
        };
        let Some(built) = by_mesh.get(&MeshId(record.mesh)) else {
            continue;
        };
        let Some(mesh) = pool.mesh(built.mesh) else {
            continue;
        };
        let t = &drawn.transform;
        let scale = (0..3).map(|c| t.column(c).xyz().norm()).fold(0.0, f32::max);
        let center = t.transform_point(&Point3::new(mesh.bounds[0], mesh.bounds[1], mesh.bounds[2]));
        if !in_view(view_projection, center.coords, mesh.bounds[3] * scale) {
            continue;
        }
        // In the mesh's own space, where its errors and bounds are.
        let Some(inverse) = t.try_inverse() else {
            continue;
        };
        let local = inverse.transform_point(&Point3::from(eye));
        for c in yoctite::select(&built.dag, [local.x, local.y, local.z], threshold) {
            triangles += built.dag.clusters[c as usize].triangles.len();
            list.push([placed.instance.0, mesh.first_cluster + c]);
        }
    }
    (list, triangles)
}

impl YoctitePass {
    /// Lays Candelite's surface view over the frame, if it is shown (see
    /// [`Self::show_candelite`]).
    fn candelite_view(&mut self, ctx: SceneRenderPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        if !self.candelite_overlay || self.failed {
            return Ok(Default::default());
        }
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(Default::default());
        };
        let Some(scene) = self.scenes.get_mut(&ctx.scene_handle) else {
            return Ok(Default::default());
        };
        let (Some(candelite), Some(pool)) = (scene.candelite.as_mut(), scene.pool.as_ref()) else {
            return Ok(Default::default());
        };
        let device = &server.state.device;
        let Some(view) = candelite.view.as_ref() else {
            return Ok(Default::default());
        };
        let viewport = ctx.observer.viewport;
        let (width, height) = (viewport.w().max(1) as usize, viewport.h().max(1) as usize);
        let matches = candelite.target.as_ref().is_some_and(|t| {
            matches!(t.kind(), GpuTextureKind::Rectangle { width: w, height: h } if w == width && h == height)
        });
        if !matches {
            candelite.target = Some(server.create_texture(GpuTextureDescriptor {
                name: "CandeliteView",
                kind: GpuTextureKind::Rectangle { width, height },
                pixel_kind: PixelKind::RGBA8,
                ..Default::default()
            })?);
        }
        let drawn = candelite.target.clone().expect("made above");
        let Some(target) = drawn.as_any().downcast_ref::<WgpuTexture>() else {
            return Ok(Default::default());
        };
        let position = &ctx.observer.position;
        let Some(inverse) = position.view_projection_matrix.try_inverse() else {
            return Ok(Default::default());
        };
        let mut inverse_columns = [[0.0f32; 4]; 4];
        for (column, values) in inverse.column_iter().zip(inverse_columns.iter_mut()) {
            values.copy_from_slice(column.as_slice());
        }
        let sunlight = sunlight_of(ctx.scene);
        let eye: [f32; 3] = position.translation.into();
        server.flush_active_pass();
        let mut encoder = server.frame_encoder.borrow_mut().take().unwrap_or_else(|| {
            device.create_command_encoder(&Default::default())
        });
        let Some(light) = candelite.light.as_mut() else {
            *server.frame_encoder.borrow_mut() = Some(encoder);
            return Ok(Default::default());
        };
        // Lit already, in the G-Buffer pass, when it lights the frame.
        light.set_lamps(&lamps_of(&ctx.bundle_storage.light_sources, eye));
        if self.candelite_shows == ViewShows::Lit && self.indirect.is_none() {
            light.update(device, &mut encoder, &mut candelite.scene, pool, &mut candelite.cache, eye, &sunlight);
            if light.is_full() && !candelite.full_reported {
                Log::warn("Candelite: its radiance atlas is full; instances past it are lit where rays hit them");
                candelite.full_reported = true;
            }
        }
        view.record(
            device,
            &mut encoder,
            &mut candelite.scene,
            pool,
            &mut candelite.cache,
            light,
            &sunlight,
            inverse_columns,
            eye,
            self.candelite_shows,
            target.wgpu_texture(),
        );
        *server.frame_encoder.borrow_mut() = Some(encoder);

        if self.shader.is_none() {
            let shader = Shader::from_string(WGSL)
                .map_err(|e| FrameworkError::Custom(format!("Yoctite view shader: {e:?}")))?;
            self.shader = Some(RenderPassContainer::new(ctx.server, &shader)?);
        }
        let size = Vector2::new(viewport.w() as f32, viewport.h() as f32);
        let world_view_projection = make_viewport_matrix(viewport);
        let properties = PropertyGroup::from([
            property("worldViewProjection", &world_view_projection),
            property("screenSize", &size),
        ]);
        let material = RenderMaterial::from([
            binding("clusters", (&drawn, &ctx.renderer_resources.nearest_clamp_sampler)),
            binding("properties", &properties),
        ]);
        let shader = self.shader.as_ref().expect("made above");
        let mut stats = RenderPassStatistics::default();
        stats += shader.run_pass(
            1,
            &ImmutableString::new("Primary"),
            ctx.framebuffer,
            &ctx.renderer_resources.quad,
            viewport,
            &material,
            ctx.uniform_buffer_cache,
            Default::default(),
            None,
        )?;
        // And the rays' own draw.
        stats.draw_calls += 1;
        Ok(stats)
    }
}

impl SceneRenderPass for YoctitePass {
    fn on_gbuffer_render(&mut self, mut ctx: GBufferPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        if self.failed || self.mode != Mode::Render {
            return Ok(Default::default());
        }
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(Default::default());
        };
        self.follow_switches();
        if !self.yoctite_on {
            return Ok(Default::default());
        }
        let started = Instant::now();
        let builder = self.builder.get_or_insert_with(Builder::new);
        builder.collect();
        let scene = self.scenes.entry(ctx.scene_handle).or_default();
        let meshes = scene.built.len();
        let changed = scene.sync(server, &ctx.scene.graph, self.mode, builder, self.candelite && self.candelite_on);
        // From the next frame on the renderer leaves these out: this one, it drew them already,
        // which drawing them again changes nothing of.
        if let (true, Some(taken_over)) = (changed, &self.taken_over) {
            taken_over.borrow_mut().insert(ctx.scene_handle, scene.taken_over.clone());
        }
        if scene.built.len() < meshes {
            builder.forget_unused(self.scenes.values());
        }
        let scene = self.scenes.get_mut(&ctx.scene_handle).expect("synced above");
        let Materials { by_key, masks, .. } = &mut scene.materials;
        let materials: Vec<YoctiteMaterial> = by_key.values().filter_map(|slot| resolve(slot, masks, &mut ctx)).collect();
        let synced = started.elapsed().as_secs_f64();

        let eye = ctx.observer.position.translation;
        let parameters = CullParameters {
            view_projection: view_projection_of(ctx.observer),
            eye: eye.into(),
            threshold: pixel_threshold(ctx.observer),
            material_slots: scene.materials.slots().max(1),
        };
        let device = &server.state.device;
        let culler = scene.culler.get_or_insert_with(|| Culler::new(device));
        let drawer = self.drawer.get_or_insert_with(|| GBufferDrawer::new(device));
        let pool = scene.pool.as_ref().expect("made by sync");
        let started = Instant::now();
        let draws = match drawer.draw(server, pool, culler, &parameters, ctx.framebuffer, &materials) {
            Ok(draws) => draws,
            Err(err) => {
                Log::err(format!("Yoctite failed, leaving the meshes to the renderer: {err:?}"));
                self.failed = true;
                if let Some(taken_over) = &self.taken_over {
                    taken_over.borrow_mut().clear();
                }
                return Ok(Default::default());
            }
        };
        // Candelite's new captures, into its cache; and what it traces against, kept in step.
        let mut candelite_failed = false;
        if let Some(candelite) = scene.candelite.as_mut() {
            server.flush_active_pass();
            let mut encoder = server.frame_encoder.borrow_mut().take().unwrap_or_else(|| {
                device.create_command_encoder(&Default::default())
            });
            if !candelite.pending.is_empty() {
                let captured = resolve_materials(&materials).and_then(|resolved| {
                    candelite.cache.record(device, &mut encoder, drawer, pool, &candelite.pending, &resolved)
                });
                if let Err(err) = captured {
                    Log::err(format!("Candelite could not capture its cards: {err:?}"));
                    candelite_failed = true;
                }
                candelite.pending.clear();
            }
            candelite.scene.update(device, &mut encoder, pool);
            *server.frame_encoder.borrow_mut() = Some(encoder);
            // The light between surfaces, for the renderer to light the frame with.
            if let Some((indirect, specular)) = &self.indirect {
                if let Err(err) = gather(server, candelite, pool, &ctx, indirect, specular, self.candelite_strength) {
                    Log::err(format!("Candelite could not light the frame: {err:?}"));
                    candelite_failed = true;
                }
            }
        }
        let drew = started.elapsed().as_secs_f64();
        let (clusters, triangles) = scene.chosen("").unwrap_or_default();
        if scene.frame % STATS_EVERY == 1 {
            let hidden = scene.culler.as_ref().and_then(Culler::stats).unwrap_or_default();
            Log::info(format!(
                "Yoctite: {clusters} clusters, {triangles} triangles in {} draws ({} materials), of them {} clusters \
                 and {} triangles hidden by the frame before, and {} clusters and {} triangles drawn by the software \
                 rasterizer{}; the CPU took {:.2} ms keeping up with the scene and its materials and {:.2} ms choosing \
                 and drawing",
                draws.draw_calls,
                materials.len(),
                hidden.hidden_clusters,
                hidden.hidden_triangles,
                hidden.software_clusters,
                hidden.software_triangles,
                if draws.software { "" } else { " (off)" },
                synced * 1000.0,
                drew * 1000.0
            ));
        }
        let mut stats = RenderPassStatistics::default();
        stats.draw_calls += draws.draw_calls as usize;
        stats.triangles_rendered += triangles;
        if candelite_failed {
            self.turn_off_candelite();
        }
        Ok(stats)
    }

    fn on_hdr_render(&mut self, ctx: SceneRenderPassContext) -> Result<RenderPassStatistics, FrameworkError> {
        if !self.yoctite_on {
            return Ok(Default::default());
        }
        let Mode::View { on_gpu } = self.mode else {
            return self.candelite_view(ctx);
        };
        if self.failed {
            return Ok(Default::default());
        }
        let Some(server) = ctx.server.as_any().downcast_ref::<WgpuGraphicsServer>() else {
            return Ok(Default::default());
        };
        if self.shader.is_none() {
            let shader = Shader::from_string(WGSL)
                .map_err(|e| FrameworkError::Custom(format!("Yoctite view shader: {e:?}")))?;
            self.shader = Some(RenderPassContainer::new(ctx.server, &shader)?);
        }
        let started = Instant::now();
        let builder = self.builder.get_or_insert_with(Builder::new);
        builder.collect();
        let scene = self.scenes.entry(ctx.scene_handle).or_default();
        let meshes = scene.built.len();
        scene.sync(server, &ctx.scene.graph, self.mode, builder, false);
        if scene.built.len() < meshes {
            builder.forget_unused(self.scenes.values());
        }
        let scene = self.scenes.get_mut(&ctx.scene_handle).expect("synced above");
        let synced = started.elapsed().as_secs_f64();

        let position = &ctx.observer.position;
        let viewport = ctx.observer.viewport;
        let threshold = pixel_threshold(ctx.observer);
        let view_projection = view_projection_of(ctx.observer);
        let eye = position.translation;
        let (width, height) = (viewport.w() as u32, viewport.h() as u32);
        let view = scene.view.get_or_insert_with(|| ClusterView::new(&server.state.device));
        let pool = scene.pool.as_ref().expect("made by sync");
        let started = Instant::now();
        let (drawn, cpu_chosen) = if on_gpu {
            let culler = scene.culler.get_or_insert_with(|| Culler::new(&server.state.device));
            let drawn = view.draw_culled(server, pool, culler, view_projection, eye.into(), threshold, width, height);
            (drawn, None)
        } else {
            let (list, triangles) = choose(&scene.built, &scene.nodes, pool, &position.view_projection_matrix, eye, threshold);
            let drawn = view.draw(server, pool, view_projection, width, height, &list);
            (drawn, Some((list.len(), triangles)))
        };
        let chose = started.elapsed().as_secs_f64();
        let drawn = match drawn {
            Ok(texture) => texture,
            Err(err) => {
                Log::err(format!("Yoctite view failed: {err:?}"));
                self.failed = true;
                return Ok(Default::default());
            }
        };
        let (clusters, triangles) = cpu_chosen.or_else(|| scene.chosen("view")).unwrap_or_default();
        if scene.frame % STATS_EVERY == 1 {
            Log::info(format!(
                "Yoctite view: {} clusters, {} triangles, chosen on the {}; the pool kept up with the scene \
                 in {:.2} ms, choosing and drawing took the CPU {:.2} ms",
                clusters,
                triangles,
                if on_gpu { "GPU" } else { "CPU" },
                synced * 1000.0,
                chose * 1000.0
            ));
        }

        let size = Vector2::new(viewport.w() as f32, viewport.h() as f32);
        let world_view_projection = make_viewport_matrix(viewport);
        let properties = PropertyGroup::from([
            property("worldViewProjection", &world_view_projection),
            property("screenSize", &size),
        ]);
        let material = RenderMaterial::from([
            binding("clusters", (&drawn, &ctx.renderer_resources.nearest_clamp_sampler)),
            binding("properties", &properties),
        ]);
        let shader = self.shader.as_ref().expect("made above");
        let composited = shader.run_pass(
            1,
            &ImmutableString::new("Primary"),
            ctx.framebuffer,
            &ctx.renderer_resources.quad,
            viewport,
            &material,
            ctx.uniform_buffer_cache,
            Default::default(),
            None,
        )?;
        let mut stats = RenderPassStatistics::default();
        stats += composited;
        // And the clusters' own draw.
        stats.draw_calls += 1;
        stats.triangles_rendered += triangles;
        Ok(stats)
    }

    fn source_type_id(&self) -> TypeId {
        self.source_type_id
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fyrox::core::algebra::Perspective3;

    #[test]
    fn lamps_fade_out_as_they_near_being_left_out() {
        let lamp = |distance: f32| (distance, Lamp { position: [distance, 0.0, 0.0], reach: 10.0, light: [1.0; 3], spot: None });
        // Two more than Candelite lights with, a meter apart, out of order.
        let n = candelite::MOST_LAMPS + 2;
        let lamps: Vec<_> = (0..n).rev().map(|i| lamp(i as f32)).collect();
        let kept = faded(lamps);
        assert_eq!(kept.len(), candelite::MOST_LAMPS, "no more than it lights with");
        assert!(kept.windows(2).all(|w| w[0].position[0] < w[1].position[0]), "nearest first");
        // The nearest left out is MOST_LAMPS meters off: those LAMP_FADE or more nearer, whole;
        // those nearer it, dimmer the nearer, to nothing at it.
        let cut = candelite::MOST_LAMPS as f32;
        for lamp in &kept {
            let expected = ((cut - lamp.position[0]) / LAMP_FADE).clamp(0.0, 1.0);
            assert!((lamp.light[0] - expected).abs() < 1.0e-6, "lamp at {}: {:?}", lamp.position[0], lamp.light);
        }
        assert_eq!(kept[0].light, [1.0; 3]);
        // With no more than it lights with, none is left out, and none fades.
        let few = faded((0..3).map(|i| lamp(i as f32)).collect());
        assert!(few.iter().all(|lamp| lamp.light == [1.0; 3]));
    }

    #[test]
    fn spheres_in_and_out_of_view() {
        // Looking down -z, as OpenGL's cameras do.
        let projection = Perspective3::new(1.0, 1.0, 0.1, 100.0).to_homogeneous();
        assert!(in_view(&projection, Vector3::new(0.0, 0.0, -10.0), 1.0));
        assert!(!in_view(&projection, Vector3::new(0.0, 0.0, 10.0), 1.0), "behind");
        assert!(!in_view(&projection, Vector3::new(50.0, 0.0, -10.0), 1.0), "off to the side");
        assert!(in_view(&projection, Vector3::new(11.0, 0.0, -10.0), 6.0), "partly in");
        assert!(!in_view(&projection, Vector3::new(0.0, 0.0, -200.0), 1.0), "past the far plane");
    }

    #[test]
    fn rows_are_the_top_of_the_matrix() {
        let m = Matrix4::new_translation(&Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(rows(&m), [1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn material_slots_are_given_back_and_reused() {
        let a = MaterialResource::new_embedded(Material::standard());
        let b = MaterialResource::new_embedded(Material::standard());
        let mut materials = Materials::default();
        assert_eq!(materials.take(&a, Shading::Standard), 0);
        assert_eq!(materials.take(&a, Shading::Standard), 0, "the same material, the same slot");
        assert_eq!(materials.take(&b, Shading::Standard), 1);
        materials.give_back(a.key());
        assert_eq!(materials.slot(a.key()), 0, "still used once");
        materials.give_back(a.key());
        assert!(!materials.by_key.contains_key(&a.key()), "no longer used");
        let c = MaterialResource::new_embedded(Material::standard());
        assert_eq!(materials.take(&c, Shading::Standard), 0, "its slot used again");
        assert_eq!(materials.slots(), 2);
    }
}

/// The lamps Candelite lights with besides the sun: the point and spot lights the frame is lit
/// with, nearest `eye` first (it lights with [`candelite::MOST_LAMPS`]), each as the renderer
/// lights with it - its colour linear times its intensity over pi, falling off to nothing at its
/// reach, a spot's along its -y within its cone - faded out as it nears being left out (see
/// [`faded`]).
fn lamps_of(lights: &[fyrox::renderer::bundle::LightSource], eye: [f32; 3]) -> Vec<Lamp> {
    use fyrox::renderer::bundle::LightSourceKind;
    let lamps: Vec<(f32, Lamp)> = lights
        .iter()
        .filter_map(|source| {
            let light = linear(source.color).map(|c| c * source.intensity / std::f32::consts::PI);
            let (reach, spot) = match source.kind {
                LightSourceKind::Point { radius, .. } => (radius, None),
                LightSourceKind::Spot { full_cone_angle, hotspot_cone_angle, distance, .. } => {
                    let toward = (-source.up_vector).try_normalize(f32::EPSILON)?;
                    (distance, Some((toward.into(), (full_cone_angle * 0.5).cos(), (hotspot_cone_angle * 0.5).cos())))
                }
                _ => return None,
            };
            if reach <= 0.0 || light.iter().all(|&c| c <= 0.0) {
                return None;
            }
            let position: [f32; 3] = source.position.into();
            let distance = (0..3).map(|i| (position[i] - eye[i]).powi(2)).sum::<f32>().sqrt();
            Some((distance, Lamp { position, reach, light, spot }))
        })
        .collect();
    faded(lamps)
}

/// Over how many meters a lamp fades out as it nears being left out of the lamps Candelite lights
/// with: see [`faded`].
const LAMP_FADE: f32 = 3.0;

/// `lamps`, each with how far it is from the eye, nearest first and no more than Candelite lights
/// with ([`candelite::MOST_LAMPS`]), each faded by how near it is to being left out: to nothing
/// as it comes within [`LAMP_FADE`] of the nearest lamp left out. Cut off sharply, a lamp left
/// out switched the light bounced off what it lit off at once, and one let in on - whole patches
/// of a corridor flickering as the view went past lamp after lamp of a maze. Faded, the light
/// passes from one lamp to the next as the view moves.
fn faded(mut lamps: Vec<(f32, Lamp)>) -> Vec<Lamp> {
    lamps.sort_by(|a, b| a.0.total_cmp(&b.0));
    let cut = lamps.get(candelite::MOST_LAMPS).map_or(f32::INFINITY, |(distance, _)| *distance);
    lamps.truncate(candelite::MOST_LAMPS);
    lamps
        .into_iter()
        .map(|(distance, lamp)| {
            let fade = ((cut - distance) / LAMP_FADE).clamp(0.0, 1.0);
            Lamp { light: lamp.light.map(|c| c * fade), ..lamp }
        })
        .collect()
}

/// An sRGB colour, linear.
fn linear(c: fyrox::core::color::Color) -> [f32; 3] {
    [c.r, c.g, c.b].map(|c| {
        let c = c as f32 / 255.0;
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    })
}

/// The light Candelite lights with: the scene's first directional light that is on - its up
/// vector towards the sun, as the renderer lights with it, its colour linear times its intensity
/// over pi, as the renderer's diffuse lighting gives off - and the sky, its ambient lighting
/// colour, linear.
fn sunlight_of(scene: &Scene) -> Sunlight {
    let sky = linear(scene.rendering_options.ambient_lighting_color);
    // Only a sun that is on: a game hides its sun at night, and the renderer does not light with it.
    let on = |node: &&fyrox::scene::node::Node| node.global_visibility() && node.is_globally_enabled();
    match scene.graph.linear_iter().filter(on).find_map(|node| node.cast::<DirectionalLight>()) {
        Some(sun) => {
            let base: &BaseLight = sun.base_light_ref();
            let toward = sun.up_vector().try_normalize(f32::EPSILON).unwrap_or_else(Vector3::y);
            Sunlight {
                toward_sun: toward.into(),
                sun: linear(base.color()).map(|c| c * base.intensity() / std::f32::consts::PI),
                sky,
            }
        }
        None => Sunlight {
            toward_sun: [0.0, 1.0, 0.0],
            sun: [0.0; 3],
            sky,
        },
    }
}

/// Lights Candelite's cards and gathers with its screen probes the light between surfaces at
/// each pixel of the G-Buffer just drawn, into a texture of its own, put in `indirect` for the
/// renderer to light the frame with.
fn gather(
    server: &WgpuGraphicsServer,
    candelite: &mut Candelite,
    pool: &GeometryPool,
    ctx: &GBufferPassContext,
    indirect: &fyrox::renderer::IndirectDiffuse,
    specular: &fyrox::renderer::IndirectSpecular,
    strength: f32,
) -> Result<(), FrameworkError> {
    let device = &server.state.device;
    let (Some(light), Some(depth_attachment)) = (candelite.light.as_mut(), ctx.framebuffer.depth_attachment()) else {
        return Ok(());
    };
    let (Some(normal_attachment), Some(material_attachment)) =
        (ctx.framebuffer.color_attachments().get(1), ctx.framebuffer.color_attachments().get(3))
    else {
        return Ok(());
    };
    let Some(material) = material_attachment.texture.as_any().downcast_ref::<WgpuTexture>() else {
        return Ok(());
    };
    let (Some(depth), Some(normal)) = (
        depth_attachment.texture.as_any().downcast_ref::<WgpuTexture>(),
        normal_attachment.texture.as_any().downcast_ref::<WgpuTexture>(),
    ) else {
        return Ok(());
    };
    let (width, height) = (depth.wgpu_texture().width() as usize, depth.wgpu_texture().height() as usize);
    let fits = |t: &Option<GpuTexture>| {
        t.as_ref().is_some_and(|t| matches!(t.kind(), GpuTextureKind::Rectangle { width: w, height: h } if w == width && h == height))
    };
    for (texture, name) in [(&mut candelite.indirect, "CandeliteIndirect"), (&mut candelite.specular, "CandeliteSpecular")] {
        if !fits(texture) {
            *texture = Some(server.create_texture(GpuTextureDescriptor {
                name,
                kind: GpuTextureKind::Rectangle { width, height },
                pixel_kind: PixelKind::RGBA16F,
                ..Default::default()
            })?);
        }
    }
    let target = candelite.indirect.clone().expect("made above");
    let reflected = candelite.specular.clone().expect("made above");
    let (Some(wgpu_target), Some(wgpu_reflected)) =
        (target.as_any().downcast_ref::<WgpuTexture>(), reflected.as_any().downcast_ref::<WgpuTexture>())
    else {
        return Ok(());
    };
    if candelite.probes.is_none() {
        candelite.probes = ScreenProbes::new(device);
    }
    if candelite.radiance.is_none() {
        candelite.radiance = RadianceCache::new(device);
    }
    if candelite.reflections.is_none() {
        candelite.reflections = Reflections::new(device);
    }
    let (Some(probes), Some(radiance), Some(reflections)) =
        (candelite.probes.as_mut(), candelite.radiance.as_mut(), candelite.reflections.as_mut())
    else {
        return Ok(());
    };
    let sunlight = sunlight_of(ctx.scene);
    let position = &ctx.observer.position;
    let Some(inverse) = position.view_projection_matrix.try_inverse() else {
        return Ok(());
    };
    let mut encoder = server.frame_encoder.borrow_mut().take().unwrap_or_else(|| device.create_command_encoder(&Default::default()));
    let eye: [f32; 3] = position.translation.into();
    // Timed apart, as GPU marks: the cards lit, the radiance cache, the screen probes.
    server.gpu_mark_in(&mut encoder, "G-buffer: custom passes (Yoctite)");
    // The cards' lighting kept near CARDS_LIT_MS of GPU time: its texels' rays cost ten times as
    // much among crowns as over open ground, and walking into a wood with a fixed budget stalled
    // frames. Steered from the latest frame timed (a frame or two behind), a quarter of the way a
    // frame so as not to swing.
    if let Some(ms) = server.gpu_latest("Candelite: cards lit") {
        let scale = (CARDS_LIT_MS / ms.max(0.05)).powf(0.25).clamp(0.7, 1.1);
        light.set_budget(((light.budget() as f32 * scale) as u32).max(CARDS_LIT_LEAST));
    }
    // The radiance cache likewise: the grids stepping a cell hand it a slab of new probes at once.
    if let Some(ms) = server.gpu_latest("Candelite: radiance cache") {
        let scale = (CACHE_MS / ms.max(0.05)).powf(0.25).clamp(0.7, 1.1);
        radiance.set_budget(((radiance.budget() as f32 * scale) as usize).max(CACHE_LEAST));
    }
    light.set_lamps(&lamps_of(ctx.light_sources, eye));
    light.update(device, &mut encoder, &mut candelite.scene, pool, &mut candelite.cache, eye, &sunlight);
    server.gpu_mark_in(&mut encoder, "Candelite: cards lit");
    radiance.update(device, &mut encoder, &mut candelite.scene, pool, &mut candelite.cache, light, eye, &sunlight);
    server.gpu_mark_in(&mut encoder, "Candelite: radiance cache");
    let columns = |m: &Matrix4<f32>| {
        let mut out = [[0.0f32; 4]; 4];
        for (column, values) in m.column_iter().zip(out.iter_mut()) {
            values.copy_from_slice(column.as_slice());
        }
        out
    };
    let frame = ProbeFrame {
        depth: depth.wgpu_texture(),
        normal: normal.wgpu_texture(),
        material: material.wgpu_texture(),
        view_projection: columns(&position.view_projection_matrix),
        inverse_view_projection: columns(&inverse),
        target: wgpu_target.wgpu_texture(),
    };
    probes.set_strength(strength);
    probes.record(device, &mut encoder, &mut candelite.scene, pool, &mut candelite.cache, light, radiance, &sunlight, &frame);
    server.gpu_mark_in(&mut encoder, "Candelite: screen probes");
    reflections.record(
        device,
        &mut encoder,
        &mut candelite.scene,
        pool,
        &mut candelite.cache,
        light,
        radiance,
        &sunlight,
        &frame,
        wgpu_reflected.wgpu_texture(),
    );
    server.gpu_mark_in(&mut encoder, "Candelite: reflections");
    *server.frame_encoder.borrow_mut() = Some(encoder);
    indirect.borrow_mut().insert(ctx.scene_handle, target);
    specular.borrow_mut().insert(ctx.scene_handle, reflected);
    Ok(())
}
