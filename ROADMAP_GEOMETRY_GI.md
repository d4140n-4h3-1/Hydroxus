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
5. **Two-pass occlusion culling** against a hierarchical depth buffer: last frame's visible
   clusters first, then everything newly uncovered.
6. **Alpha-masked clusters** (foliage): the visibility pass tests the texture's alpha. This
   matters most for forests.
7. **Software rasterizer** for clusters whose triangles are smaller than a pixel (compute, 64-bit
   atomics into the visibility buffer).
8. **Streaming**: cluster pages loaded and evicted by demand. Only when scenes outgrow memory.

Ray-traced shadows keep tracing the full-detail meshes. Skinned and moving meshes stay on the
existing path.

## Candelite: dynamic global illumination (hardware ray tracing)

Named for the candela, the unit of luminous intensity, as Yoctite is for the smallest of
prefixes. The method follows the published description of Lumen cited above.

1. **Surface cache**: each mesh is given a few "cards" (oriented boxes over its surface); its
   albedo, normal and emissive are captured into an atlas once.
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
