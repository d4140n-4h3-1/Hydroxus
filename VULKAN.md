# Hydroxus: the Vulkan branch

This branch - Hydroxus's `vulkan` - makes Fyrox's wgpu backend (`backend_wgpu`, Vulkan on Linux) render games the same
way the OpenGL backend does. It was driven by running Station Iapetus on both backends and
comparing the frames stage by stage (G-buffer, HDR, final image).

Build a game with `fyrox = { default-features = false, features = ["backend_wgpu"] }`, in a
workspace where nothing enables `backend_opengl`: when both are on, OpenGL wins.

## What changed and why

**Build**
- `fyrox/Cargo.toml`: `fyrox-impl` no longer pulls in its default features, which always
  enabled OpenGL, so `backend_wgpu` could never take effect.

**Coordinate conventions.** The engine's matrices and shaders follow OpenGL. wgpu clips depth
below 0 instead of -1 and stores render targets top row first.
- `fyrox-graphics-wgpu/src/vertex_depth.rs`: every vertex shader's `vs_main` is wrapped to map
  clip-space depth to wgpu's range, so depth values equal OpenGL's window depth again and the
  near half of every frustum is no longer clipped away.
- `shared.wgsl`: `S_UnProject` flips v. Shadow lookups use the new `S_ProjectToTexture`, and
  lookups into cube maps the renderer draws itself go through `S_RenderedCubeDirection`.
- The SSAO kernel and decal screen UVs flip v.
- The light volume pass (`light_volume.rs`) and the bloom bright pass (`bloom/mod.rs`) use
  `make_deferred_viewport_matrix`, so they line up with the G-buffer and the scene frame.
- UI drawn into textures is flipped (`UiRenderContext::flip_y`) to match OpenGL's layout.

**Backend correctness**
- `Queue::write_*` takes effect at the next submit, ahead of the whole batch. Recorded commands
  are now submitted before every buffer, geometry, texture and readback write. Without this,
  stale uniforms hung the GPU until the driver reset it.
- Non-normalized integer vertex attributes (bone indices) are converted to `f32` on upload, as
  OpenGL does.
- Clears are tracked per part (color, depth, stencil); a stencil-only clear used to wipe color.
- The pipeline cache is keyed on the full `DrawParameters`, and the color write mask is honored.
- The bind group and pipeline caches are keyed on wgpu handles, not wrapper addresses, and the
  bind group cache is bounded.
- Scissor rectangles are clipped to the render target. Readback rows are padded to 256 bytes.
- Render passes draw into the requested mip level; the prefiltered specular probe used to
  render every level into mip 0.
- `swap_buffers` presents a cleared frame when nothing was drawn, so a Wayland window is always
  mapped and the event loop never waits for a frame callback that cannot come.

**Also on this branch**
- `renderer/mod.rs`: the uniform memory page size is capped at 1 MB. Some drivers (AMD with Mesa)
  report 2 GB, which made every upload enormous and froze loading screens. This affects OpenGL
  too.
- `scene/animation/mod.rs`: animation tracks accept property paths saved by older engine
  versions (no leading `0.`, `Content` instead of `Value`, paths that step through inheritable
  variables). Station Iapetus's lights and doors use such paths.
- Scene and UI loading no longer log an error for files saved before `UserData` existed.
- Script errors name the failing script type, not just the node.
- `shared.wgsl` / `shared.glsl`: the soft shadow (PCF) filter divided by fewer samples than it
  took, which pushed the penumbra to full shadow early and left a hard edge. Both backends now
  divide by the real sample count.
- `renderer/mod.rs`: when FXAA is switched off, the frame is copied instead, so the number of
  full-screen passes between the scene and the back buffer stays the same. Each such pass flips
  the image on wgpu and the flips only cancel in pairs, so skipping one turned the frame upside
  down.
- `fyrox-graphics-wgpu/src/framebuffer.rs`: a pipeline no longer declares a depth-stencil state
  when the pass it runs in has no depth attachment. wgpu rejected such draws outright ("render
  pipeline targets are incompatible with render pass").
- `renderer/light.rs`: point and spot lighting is drawn with a scissor box around the light's
  screen-space bounds instead of over the whole frame. It changes nothing on screen; it saves work
  in scenes with many small lights, though in a corridor with lamps overhead (where the camera
  sits inside most light volumes) the saving is small.
- `fyrox-graphics-wgpu/src/raytracing.rs`: hardware ray tracing, where the adapter has it. The
  device is created with wgpu's experimental ray query feature, scene triangles can be put into an
  acceleration structure, and a self-contained pass traces one shadow ray per pixel into a mask
  texture. Nothing else in the renderer knows about it; an effect reads the mask as an ordinary
  texture. Off unless a game asks for it.
- `resource/gltf/mod.rs`: meshes are imported in the document's own order, which is the order
  nodes look them up in. They used to be gathered by walking the nodes, so a file whose nodes
  did not list their meshes in that order had them mixed up - a character's body put on its eyes
  node, and its eyes on the body - and a mesh shared by several nodes was imported once for each.
- `resource/gltf/surface.rs`: a mesh without texture coordinates gets tangents at right angles
  to its normals. It used to keep tangents of zero, which made the shader's normal zero too, so
  the mesh was lit by next to nothing and came out black whatever its color.
- `resource/gltf/animation.rs`: rotation keys are kept on one side of the quaternion sphere.
  Exporters may write a key as `-q` rather than `q`, the same rotation; but rotations are
  interpolated component by component, and between keys on opposite sides every component
  passes through zero, so the bone spins through poses it was never keyed to. It showed as legs
  that jerked for a frame a few times each time round a walk.

**In a browser.** On `wasm32` the backend draws with WebGL 2, through wgpu's GL backend; the game
has to enable wgpu's `webgl` feature. This path had never run, and failed in six places.
- `server.rs`: the surface is never configured at zero size. winit learns a canvas's size only
  once the page has laid it out, so the first size it reports is zero; the real one follows as a
  resize.
- `server.rs`: the device asks for the limits the adapter has, in a browser too. WebGL 2's
  minimums - 2048-pixel textures, four color attachments - are less than the G-buffer needs.
- `program.rs`: the first field of every generated uniform struct is aligned to 16, which rounds
  the struct up to a multiple of 16 bytes, as WebGL requires of a uniform block. The engine
  already pads the data it sends to match.
- `program.rs`: in a browser, depth textures are declared as float textures and each sample of
  one takes its red channel. naga writes every depth texture for GLSL as a shadow sampler, which
  can only be compared against, so WebGL rejected every shader that read depth.
- `depth_copy.rs`: in a browser, depth is copied by drawing it as the fragment depth. The GL
  backend copies a texture by reading it as a framebuffer's color, which a depth texture cannot
  be, so the G-buffer's depth never reached the scene framebuffer and light volumes lit nothing.
  Stencil is not copied.
- `framebuffer.rs`, `server.rs`: attributes a mesh lacks (bone weights and indices, a second UV
  set) are read from a zeroed dummy buffer that steps once per instance, and grows to hold an
  element for each. It used to step per vertex with a stride of zero, which Vulkan reads as the
  same element for every vertex; OpenGL reads it as tightly packed, so every vertex after the
  first read past the end of the buffer. Chromium clamps such reads; Firefox rejects the draw,
  and only the sky was drawn.

**Shadow maps, faster.** Gathering what a pass draws (`RenderDataBundleStorage::from_graph`)
walks the whole scene graph, and was most of a frame in a big scene lit with shadow maps.
- `shadow/point.rs`: a point light's shadow gathers what the light reaches once, with a box
  around its sphere, and each of the cube's six faces draws what of that is in its own view. It
  used to walk the graph for every face.
- `bundle.rs`: shadow passes no longer look for the reflection probe the observer is in, which
  cost a reflection query on every node, for an environment map shadows never use
  (`RenderDataBundleStorageOptions::collect_environment`).

In the maze game on the web, where shadows come from shadow maps, the two took the renderer's
work on a frame from about 33 ms to 12 in Chromium; in Firefox, where each draw call costs more,
from 32 ms to 26.
