# Yoctite (virtual geometry) and Candelite (dynamic global illumination)

A Rust reimplementation, for Hydroxus's wgpu (Vulkan) renderer, of the techniques behind Unreal's
Nanite (virtualized, cluster-based geometry) and Lumen (dynamic global illumination), from their
published descriptions (Karis et al., "A Deep Dive into Nanite Virtualized Geometry", SIGGRAPH
2021; Wright et al., "Lumen: Real-time Global Illumination in Unreal Engine 5", SIGGRAPH 2022).
No Unreal code is used.

Like the ray-traced shadows, both are built directly on wgpu inside `fyrox-graphics-wgpu`, behind
hooks in the renderer, with the existing path as the fallback where the hardware cannot.

Target hardware: AMD RX 7600 (RDNA3, RADV). wgpu 30 features used: `MULTI_DRAW_INDIRECT_COUNT`,
`SHADER_INT64_ATOMIC_MIN_MAX` / `TEXTURE_INT64_ATOMIC` (software raster), `EXPERIMENTAL_RAY_QUERY`
(already in use).

What each one buys:
- **Virtual geometry** is the performance one. The renderer's cost today is per draw call
  (Maze's Urbs: about 4,300 draws and 28 ms just to fill the G-buffer). Virtual geometry draws
  all static geometry in a handful of indirect draws, picks a level of detail per cluster of
  128 triangles so triangle count follows screen pixels rather than scene size, and culls
  hidden clusters on the GPU.
- **Dynamic GI** is a lighting feature and costs frame time (budget: about 4 ms at 1080p). It
  replaces the flat ambient colour with light that bounces: the sun lighting a clearing lights
  the trunks around it, and the inside of a forest is darker than its edge.

## Yoctite: virtual geometry (Nanite-style)

1. **Cluster builder** - done: the `yoctite` crate (CPU, pure Rust, no dependencies). Split each
   static mesh into clusters of up to 128 triangles with bounds; group neighbouring clusters,
   simplify each group to half (quadric error metric, half-edge collapses so every level uses
   the mesh's own vertices, group borders locked), split it again into clusters; repeat to a
   single root. The result is a DAG of levels of detail with monotonic errors and bounds, and a
   versioned binary format to cache it. A seam (texture coordinates or normals) is kept whole
   unless nothing cheaper is left; breaking one counts as error the length of the edge.
   Tests: every cut watertight from many views and thresholds (icosphere, seamed sphere, open
   grid, flat-shaded cube), errors and bounds monotonic, levels halve, seams and normals kept,
   foliage cards make full clusters, bytes round-trip.
   Measured (`cargo run -p yoctite --release --example inspect -- model.glb`): the Wilderness
   trees build in 1-20 ms a mesh, halving level by level to one root of about 75 triangles;
   Maze's Lumen race city, 1.22 M triangles in 617 meshes, in 3.9 s on one thread, down to 51 k
   at its coarsest.
   Foliage thins out (`prune`): cards cannot lose triangles without whole cards vanishing, so a
   group that is mostly small loose cards keeps half of them at each level, at random, each grown
   about its middle to keep the area they cover (Cook et al. 2007, stochastic pruning). Grown
   corners are new vertices (`ClusterDag::extra_vertices`, format version 2), copies of the
   mesh's own; their error is `Settings::card_error` (0.25) of the cards' reach, so a crown thins
   from where its cards are about eight pixels across. Texture drag is measured where each vertex
   went (rings round the vertex it was collapsed onto, facing its way, covering it), and a whole
   number of repeats apart is a seam, not a drag: measured against the nearest triangle anywhere
   it had taken other branches and seams for drag, of metres, and held whole trees at full
   detail. Drag is measured at every level against the full mesh's own vertices, followed to
   where each went, so it is never added up level on level. In Wilderness, 6.6 M triangles drawn
   became 3.4 M, and a tree builds in 20-70 ms, off the frame (fyrox-gfx builds on threads of
   its own).
2. **Geometry pool** - done (`fyrox_graphics_wgpu::yoctite::GeometryPool`): every cluster of
   every mesh in shared GPU buffers, plus an instance buffer, uploaded once and updated as
   instances come and go; `ClusterView` draws clusters from it in one instanced draw, and
   fyrox-gfx's `GraphicsEffects::with_yoctite_view` shows the scene's meshes as their clusters.
3. **GPU culling and LOD selection** - done (`Culler`, `shaders/yoctite_cull.wgsl`): five compute
   passes - instances frustum-culled, then a workgroup per visible instance choosing the cut
   (own error under a pixel, parent's not, in world space with the instance's largest scale) and
   frustum-culling clusters - into a draw list and indirect draw arguments, drawn with
   `draw_indirect`; counts read back without stalling, and the list grown when it overflows.
   Tested to choose exactly the clusters the CPU reference does. In Wilderness choosing and
   drawing costs the CPU 0.02 ms, against 1.3-1.6 ms on the CPU.
4. **Visibility buffer**: hardware-rasterize visible clusters in one indirect draw into a
   depth plus (instance, cluster, triangle) buffer; a material pass resolves it into the existing
   G-buffer, so lights, traced shadows, fog and the rest work unchanged.
5. **Two-pass occlusion culling** - done (`DepthPyramid`, `shaders/yoctite_hzb*.wgsl`,
   `Culler::record_first` / `record_second`): clusters hidden behind the frame before's depth
   pyramid, seen as it was, are put aside rather than drawn; a pyramid is made of what was drawn,
   and what was put aside is tried again against it and drawn if it is not hidden after all. A
   GPU test hides a floor under a roof, then takes the roof away. In Wilderness the forest hides
   little - needles leave gaps, and a tile with sky in it holds the sky's depth - about 15% of
   the triangles, 0.3-0.46 ms of a 3 ms G-buffer.
6. **Alpha-masked clusters** (foliage) - done with 7: the software rasterizer tests each pixel's
   texel of the diffuse texture's alpha, as the G-buffer pass does.
7. **Software rasterizer and visibility buffer** - done (`SoftwareRaster`,
   `shaders/yoctite_raster.wgsl`, `yoctite_resolve.wgsl`): the culling lists clusters whose
   radius on the screen is `SOFTWARE_PIXELS` (32) or less, and clear of the near plane, for a
   compute pass instead - a workgroup a cluster, a thread a triangle - which keeps each pixel's
   nearest triangle in a 64-bit visibility buffer (`atomicMax` of depth and triangle); a
   fullscreen pass then shades each pixel from its triangle into the G-buffer with its depth,
   tested against what the hardware drew. Both run after each culling phase's hardware draws, so
   the depth pyramid sees what they drew. Bending meshes bend in both. Needs 64-bit atomics and
   arrays of textures (`SOFTWARE_FEATURES`); without them everything is drawn by the hardware.
   A GPU test draws the same squares in software and in hardware - coverage, colour, normal,
   decal layer and the alpha test alike - and bent. In Wilderness's swaying forest the G-buffer
   went from 7.7 ms to 5.3 ms (a GPU frame of 10.6 ms to 8.2): the hardware is slow at the
   pixel-sized triangles of distant crowns, and slower still at their bending, which it does a
   corner at a time. The hardware stays faster for triangles some pixels across: everything that
   could be in software took 5.6 ms. Phase 4's single hardware visibility pass was not needed:
   the hardware draws straight into the G-buffer, a draw a material.
8. **Streaming**: cluster pages loaded and evicted by demand. Only when scenes outgrow memory.

Ray-traced shadows keep tracing the full-detail meshes. Skinned meshes stay on the existing
path; meshes that bend by blend shapes (morph targets) are drawn by Yoctite, bent at every level
of detail (`GeometryPool::add_morphing_mesh`).

## Candelite: dynamic global illumination (hardware ray tracing)

Named for the candela, the unit of luminous intensity, as Yoctite is for the smallest of
prefixes. The method follows the published description of Lumen cited above.

1. **Surface cache** - captured (`fyrox_graphics_wgpu::candelite::SurfaceCache`): each mesh of
   Yoctite's pool is given six cards, the faces of its bounds looking in, each a tile of a shared
   atlas at 8 texels a meter (4 to 256 a side, packed in shelves); what each card sees - the
   G-buffer's diffuse colour, normal, emission, material and decal layer, and how far in the
   surface is - is captured once by drawing the mesh's full-detail clusters through the card's
   orthographic view as the G-buffer pass draws them (`GBufferDrawer::record_capture`). The cards
   are on the GPU for the hit lookup (`GpuCard`, a mesh slot's first card). A GPU test captures a
   box - every card its face, colour, normal and depth, edge to edge - and a sheet seen only from
   above. fyrox-gfx captures each mesh as Yoctite uploads it.
2. **Hit lookup** - done (`TraceScene`, `LOOKUP_WGSL`, `SurfaceView`): Candelite traces against
   a scene of its own - each pool mesh's full-detail triangles, each placed instance naming its
   slot - so a hit knows its mesh and its cards. The hit is taken into the mesh's space and found
   on each card facing the ray; cards that captured it there (depth within 0.3 m) are averaged by
   how squarely they face the ray (`covered`). Foliage is traced as a cutout with the material's
   mask (rays pass its see-through texels); a needle inside a crown, which every card sees only
   behind the outer needles, takes the look of what the cards saw there, or of the nearest texel
   round a gap in what they saw (`seen`, not `covered`: such a hit is to be lit where it is, as
   Lumen's hit lighting does, not from the cache). A GPU test traces a turned box, a sheet seen
   from below, and a half see-through card in front of the box. fyrox-gfx's
   `GraphicsEffects::with_candelite_view` lays over the frame what a ray from the eye finds
   (Wilderness: WILDERNESS_CANDELITE_VIEW=1): trunks, ground and crowns, with 1.7% of what is not
   sky found on no card, mostly at thin treetops - from 46% with foliage traced solid.
3. **Lighting the surface cache** - done for the sun and sky (`SurfaceLight`): cards are captured
   per mesh, but light differs per instance, so each instance within 60 m of the eye (measured to
   its bounds) is given lit cards of its own in a 2048-square radiance atlas - its mesh's tiles,
   halved for each doubling of distance past 8 m - and a budget of texels (128k) is lit each frame,
   new cards first: the sun, if a shadow ray traced against the cutouts reaches it; and one
   cosine-weighted ray of indirect light, averaged over up to 16 updates, taking what it hit from
   that instance's lit cards (last frame's light, so light bounces again each frame), else lit
   where it hit (hit lighting: sun with a shadow ray, and half the sky), else the sky. The lookup
   returns `lit`/`radiance` from the hit's lit cards. GPU-tested: a floor under an overhead sun
   gives off its colour times the sun, and in a slab's shade only bounced light, also as the view
   sees it. fyrox-gfx `with_candelite_light_view` (Wilderness: WILDERNESS_CANDELITE_VIEW=lit).
   Lamps are still to come. Known: a mesh as big as the ground gets coarse cards (its shadows in
   metre-wide blocks), and some flowering bushes are hits no card saw (magenta since phase 2).
4. **Screen probes** - done (`ScreenProbes`): a probe every 16 pixels each way, placed on what
   the G-Buffer sees at a pixel of its tile (a different one each frame), traces 64 rays (32 since) spread
   evenly over the hemisphere about its normal (turned each frame) against the lit cards, keeps
   what they bring back as L1 spherical harmonics, and is averaged (up to 12 frames) with last
   frame's probe where it reprojects, if that lay on the same surface. Each pixel is lit from its
   four nearest probes, weighted by nearness, by how far each lies off the pixel's surface and by
   how alike their normals are (else any of nine round it), at its own normal. The renderer lights
   with that in place of the ambient colour (`Renderer::set_indirect_diffuse`, a texture per scene,
   used by the ambient pass). Candelite's light is now in linear colour, the sun as the renderer's
   diffuse lighting gives off (colour times intensity over pi). GPU-tested: floor open to a sky of
   1 gathers about 1, under a slab under half that. fyrox-gfx `GraphicsEffects::with_candelite`
   (Wilderness: WILDERNESS_CANDELITE=1): about 3.3 ms of GPU time with the cards' lighting.
   Still to do: importance sampling, spatial filtering of the probes, light through leaves (crowns
   come out darker than with the flat ambient colour).
5. **World radiance cache** - done (`RadianceCache`): three nested grids of 16 cubed probes round
   the eye, 2, 4 and 8 m apart, each probe the light reaching it from every direction as an 8 by 8
   octahedral map, averaged over frames. The grids move with the eye a cell at a time (slots wrap
   round), slots handed new cells traced first, the rest in a scrambled turn, 256 probes a frame;
   a probe whose rays mostly hit the backs of things is inside something and is not used. Screen
   probes trace 8 m (`SCREEN_REACH`; 4 m since) and take the light from beyond from the 8 probes round the
   ray's end in the finest grid holding it, else trace on. GPU-tested: with the sky put out after
   the cache saw it, open floor still gathers the sky from the cache.
   What made it affordable, and kept frames from stalling: cache rays end at 64 m or 16 cells; a
   hit without lit cards further than 4 m is lit without a shadow ray (as half in the sun); every
   ray asks about at most 32 cutout triangles (`CANDELITE_MOST_CUTOUTS`), past which the next
   stops it; a fourth grid, 16 m apart, cost five to eight times the others in spikes of tens of
   milliseconds, and was dropped. Lighting the cards packs texels 64 to a workgroup whatever cards
   they are of; and only instances big enough for their distance get lit cards (`LIT_SIZE`):
   lighting ten thousand floor plants' small cards made rays in no order a GPU likes, tens of
   milliseconds a frame while walking. In the glade: cards 2-3 ms, cache about 1 ms, screen
   probes 3.5-5.5 ms of GPU time.
6. **Reflections** - done (`Reflections`): of each 2 by 2 block of pixels, one a frame (a
   different one each frame) smoother than roughness 0.4 (`REFLECTION_ROUGHEST`, as Lumen's
   default) traces one ray along a reflection its roughness gives - a visible normal of the GGX
   distribution - against the lit cards, beyond 8 m taking the world radiance cache's light as
   the screen probes do, averaged over 3 (a mirror) to 12 frames where it reprojects onto the same
   surface. Each pixel takes the traced pixels of the nine blocks round it, weighted by nearness,
   plane distance, normal and roughness alike; from 0.3 up it gives way to the screen probes'
   light, all a rough surface reflects, which rougher pixels take alone. The renderer lights its
   specular term with it in place of the ambient colour or skybox (`Renderer::set_indirect_specular`,
   `IndirectSpecular`, as the diffuse hook; ambient pass wgpu and OpenGL). GPU-tested: a mirror
   floor reflects the sky of 1 in the open and under the slab less than 0.1; rough, exactly the
   screen probes' light. fyrox-gfx `with_candelite` turns it on with the rest. In Wilderness's glade
   it costs 0.05 ms: the forest has almost nothing smoother than 0.4, so what shows is shaded
   surfaces no longer reflecting the flat ambient colour.

Walking without stalls (since): measured a frame at a time (`WgpuGraphicsServer::gpu_latest`),
walking into the wood with Candelite took the GPU 22-39 ms a frame, and single frames 40-80 ms.
The cards' lighting and the radiance cache now each keep to a time - fyrox-gfx steers their budgets
(`SurfaceLight::set_budget`, `RadianceCache::set_budget`) towards 2.5 ms and 1 ms from the latest
frame timed - and the cards' texels are handed out to the budget exactly, a run of cards too big for
what is left carried on next frame (the ground's alone is hundreds of thousands of texels). The
single long frames were none of these: they came with the instances streaming in round the walker,
whatever the budgets, and went with instances under 1 m (`TRACE_LEAST`, half the diagonal of their
bounds) left out of what Candelite traces - rays starting near the ground crossing dense patches of
grass, ferns and small bushes. Walking, the GPU now takes 14-18 ms, and the longest frame a second
is the 35 ms of two fixed updates, as without Candelite.

Cheaper screen probes (since): 32 rays a probe (`PROBE_RAYS`) averaged over up to 20 frames
(`PROBE_HISTORY`), traced 4 m (`SCREEN_REACH`) before the radiance cache: walking, the screen
probes 7.5 ms to 2.6 ms of GPU time, the frame 37.8 to 48.8 FPS; the light within about a level of
255 of the same, a little more shimmer in deep shade.

Light through leaves (since): a material's `MaterialParams::transmission` - how much of the light
on its back comes through - is written into the G-Buffer's material alpha as one minus it (the
engine's own shaders write 1). The sun's light pass adds that share of the light on a leaf's back
(the traced shadow's rays then start off the leaf's sunlit side; `LightShadowTraceContext` carries
the material texture for it); lighting the cards does the same with its shadow ray, and its ray of
indirect light goes out the back a share of the time; a screen probe on a leaf gathers from the
whole sphere, and a leaf's pixel adds its share of the light from behind (probes keep history only
from probes of their own kind). fyrox-gfx gives every cutout material 0.5 (`LEAF_TRANSMISSION`), so
crowns seen from below are lit through rather than dark. GPU-tested: a card with half let through,
the sun straight below it, gives off half its colour on top; a leaf floor under a sky of 1 gathers
2 in the open.

## Order

Virtual geometry 1 to 4 first: that is where the frame time goes. GI follows once the visibility
buffer exists, since its card capture reuses it.

Measured since, on the GPU (`GraphicsServer::gpu_timings`, timestamps between the renderer's
sections; Wilderness logs them with WILDERNESS_STATS=1), Wilderness with Yoctite, RX 7600,
800x600: the whole G-buffer - every tree, by Yoctite - takes 1.5-3 ms, and the sun's shadows
the rest: 12 ms traced soft, 7.7 ms traced hard, 8.9 ms from shadow maps, against well under
1 ms for everything else. Occlusion culling, the visibility buffer and the software rasterizer
can only win back part of the first; the shadows are where the frame goes, and Candelite will
trace the same alpha-tested foliage.
Since: a doubled foliage card is traced once (fyrox-gfx), and soft shadows take one ray a pixel,
the pattern over the sun turning from frame to frame for temporal anti-aliasing to average -
shadow rays 9.2 ms to 3.7 ms, the GPU's frame 12 ms to 7.7. Measured with WILDERNESS_BENCH=1,
which holds the view still: with the view moving, the times swing by half.
Then: one pixel of each 2x2 block traced, a different one each frame, the rest filled in from
the traced pixels on the same surface - shadow rays 3.7 ms to 1.4 ms, the frame to 5.4 ms.
Cutting cards into pieces and leaving out the empty ones was tried and did nothing: a ray's cost
is its way through the crowns until a needle stops it, not the cards it is tested against.
