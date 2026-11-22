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
   above. Not yet captured in the renderer: that comes with the hit lookup, which uses it.
2. **Hit lookup**: a ray-query hit (instance, triangle, barycentrics) finds its card texel, so
   a ray learns the colour and light at what it hit without shading it.
3. **Lighting the surface cache**: sun and lamps with traced shadows, plus last frame's indirect
   light (multiple bounces over time), updated a budget of texels per frame.
4. **Screen probes**: diffuse GI gathered at probes every 16 pixels, rays importance-sampled,
   filtered spatially and temporally, interpolated to every pixel. Replaces the ambient colour.
5. **World radiance cache**: probes around the camera for long rays and stability.
6. **Reflections**: glossy reflections traced into the same surface cache.

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
