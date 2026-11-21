// Shadows averaged over frames, so they hold still.
//
// Where the rays turn over the light from frame to frame, and at half resolution trace a
// different pixel of each block each frame (see shadow_rays.wgsl), any one frame's shadow is a
// little different from the last: averaged only by the frame's temporal anti-aliasing, which
// keeps much of each new frame, the shadows' edges crawl. Here each pixel's shadow is blended into
// what the same place in the world had in the frames before - found through the frame before's
// view - so they settle into one.
//
// The history keeps, beside each pixel's shadow, how far its surface lay from the frame before's
// near plane. A pixel whose place in the frame before was not that far off - off the screen, or a
// different surface then, as when something nearer moved away - starts again from this frame.

struct Uniforms {
    inverseViewProjection: mat4x4f,
    previousViewProjection: mat4x4f,
    previousInverseViewProjection: mat4x4f,
    screenSize: vec2f,
    flipV: u32,
    // How much of this frame goes in; 1 starts again from it.
    blend: f32,
};

@group(0) @binding(0) var current: texture_2d<f32>;
@group(0) @binding(1) var history: texture_2d<f32>;
@group(0) @binding(2) var sceneDepth: texture_depth_2d;
@group(0) @binding(3) var historySampler: sampler;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

// The world position a screen position and depth come from, through `inverse`.
fn unproject(inverse: mat4x4f, uv: vec2f, depth: f32) -> vec3f {
    let ndcY = select(uv.y * 2.0 - 1.0, 1.0 - uv.y * 2.0, uniforms.flipV != 0u);
    let world = inverse * vec4f(uv.x * 2.0 - 1.0, ndcY, depth * 2.0 - 1.0, 1.0);
    return world.xyz / world.w;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(fragCoord.xy);
    let now = textureLoad(current, pixel, 0).rgb;
    let depth = textureLoad(sceneDepth, pixel, 0);
    if (depth >= 1.0) {
        return vec4f(1.0, 1.0, 1.0, 0.0);
    }
    let uv = fragCoord.xy / uniforms.screenSize;
    let position = unproject(uniforms.inverseViewProjection, uv, depth);
    let range = distance(position, unproject(uniforms.inverseViewProjection, uv, 0.0));

    // Where this place was on the screen in the frame before.
    let clip = uniforms.previousViewProjection * vec4f(position, 1.0);
    if (uniforms.blend >= 1.0 || clip.w <= 0.0) {
        return vec4f(now, range);
    }
    let ndc = clip.xy / clip.w;
    let was = vec2f(ndc.x * 0.5 + 0.5, select(ndc.y * 0.5 + 0.5, 0.5 - ndc.y * 0.5, uniforms.flipV != 0u));
    if (any(was < vec2f(0.0)) || any(was > vec2f(1.0))) {
        return vec4f(now, range);
    }
    let before = textureSampleLevel(history, historySampler, was, 0.0);
    // The same surface then: as far from the frame before's near plane as this place is, within
    // what depth's precision and the filtering of the history allow.
    let expected = distance(position, unproject(uniforms.previousInverseViewProjection, was, 0.0));
    if (abs(before.a - expected) > 0.02 * expected + 0.05) {
        return vec4f(now, range);
    }
    return vec4f(mix(before.rgb, now, uniforms.blend), range);
}
