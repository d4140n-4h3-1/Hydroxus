<div align="center">
  <h1>Hydroxus</h1>
  <p>A fork of the <a href="https://github.com/FyroxEngine/Fyrox">Fyrox</a> game engine, built
  around its wgpu backend: Vulkan on the desktop, WebGL 2 in a browser.</p>
</div>

Hydroxus is [Fyrox](https://fyrox.rs/) - the feature-rich 2D/3D Rust game engine by Dmitry Stepanov
and the Fyrox contributors - with its wgpu renderer made to draw games the way the OpenGL one
does, running in a browser, and taught to trace rays where the hardware can. Nearly everything
in it is Fyrox's own work; see [Credits and licence](#credits-and-licence).

## What it adds

- **The wgpu backend, made whole.** Frames match the OpenGL backend's: depth, coordinate
  conventions, shadows, bloom, SSAO, decals and UI render targets all line up. Fyrox's
  `backend_wgpu` feature takes effect instead of being overridden by OpenGL.
- **Browsers.** On `wasm32` it draws with WebGL 2, through wgpu's GL backend.
- **Hardware ray tracing.** Where the adapter has ray queries, shadows can be traced: soft and
  exact, coloured by glass the light passes through, with moving things traced where they are
  each frame.
- **Area lights.** Glowing rectangles that light what is round them from their whole surface, with
  soft traced shadows where rays can be traced.
- **Faster shadow maps**, and fixes to glTF import, animation, loading and more.

[`VULKAN.md`](VULKAN.md) describes the changes to the renderer, and why, file by file.

## Using it

The crates keep their Fyrox names, so code written for Fyrox works unchanged and upstream changes
merge cleanly. Depend on this repository instead of crates.io, and turn on the wgpu backend:

```toml
[dependencies]
fyrox = { git = "https://github.com/d4140n-4h3-1/Hydroxus.git", branch = "vulkan", default-features = false, features = ["backend_wgpu"] }
```

Nothing else in the workspace should enable `backend_opengl`: when both are on, OpenGL wins. For a
browser build, also enable wgpu's `webgl` feature.

Fyrox's own documentation applies throughout: the [Fyrox book](https://fyrox-book.github.io/),
the [API docs](https://docs.rs/fyrox/) and the
[examples](https://github.com/FyroxEngine/Fyrox-demo-projects).

## Used by

- [fyrox-gfx](https://github.com/d4140n-4h3-1/fyrox-gfx): graphics effects - traced shadows,
  area lights, glass, reflections, anti-aliasing - as render passes.
- [MazeGame](https://github.com/d4140n-4h3-1/MazeGame): the game it was made for, and its
  [web build](https://github.com/d4140n-4h3-1/MazeGame-web).

## Credits and licence

Hydroxus is a fork of [Fyrox](https://github.com/FyroxEngine/Fyrox), Copyright (c) 2019-present
Dmitry Stepanov and Fyrox Engine contributors, and like Fyrox is released under the MIT licence
([`LICENSE.md`](LICENSE.md)). It is not affiliated with or endorsed by the Fyrox project. If you
would like to support Fyrox itself, see [its README](https://github.com/FyroxEngine/Fyrox#support).
