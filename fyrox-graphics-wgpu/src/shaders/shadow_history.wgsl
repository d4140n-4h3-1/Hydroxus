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
//
// Shadows move without the view moving: a flashlight in a walking hand, a droid walking past a
// lamp, a lamp's reach coming onto the screen as you walk up to it (outside it the mask is lit).
// Averaged over ten frames, they trailed and faded in. So the history is always held to the range
// this frame's shadow takes in the 3x3 pixels round each one: where this frame is wholly lit or
// wholly shadowed, the old shadow goes at once; in soft edges, where the rays differ from pixel to
// pixel, there is still room to average. (Held only when the light moved, a droid's shadow under a
// lamp that stood still smeared behind it.)

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
    // The four texels of the history round where it was, each taken only if it held the same
    // surface then - as far from the frame before's near plane as this place is, within what
    // depth's precision allows - and weighted bilinearly among those that did. Filtered together,
    // a texel of the ground beside a trunk or rock took in the trunk's range too, failed the test,
    // and started again from this frame's one ray: and as the view's jitter moved where it was by
    // a fraction of a pixel each frame, the shadow round the foot of trunks and rocks blinked.
    let expected = distance(position, unproject(uniforms.previousInverseViewProjection, was, 0.0));
    let size = vec2i(textureDimensions(history));
    let at = was * vec2f(size) - 0.5;
    let corner = vec2i(floor(at));
    let f = at - floor(at);
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var k = 0; k < 4; k++) {
        let offset = vec2i(k & 1, k >> 1);
        let texel = clamp(corner + offset, vec2i(0), size - 1);
        let then = textureLoad(history, texel, 0);
        if (abs(then.a - expected) > 0.02 * expected + 0.05) {
            continue;
        }
        let w = select(1.0 - f.x, f.x, offset.x == 1) * select(1.0 - f.y, f.y, offset.y == 1) + 1.0e-4;
        sum += then.rgb * w;
        weight += w;
    }
    if (weight <= 0.0) {
        return vec4f(now, range);
    }
    let currentSize = vec2i(textureDimensions(current));
    var lo = now;
    var hi = now;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let near = textureLoad(current, clamp(pixel + vec2i(x, y), vec2i(0), currentSize - 1), 0).rgb;
            lo = min(lo, near);
            hi = max(hi, near);
        }
    }
    let then = clamp(sum / weight, lo, hi);
    return vec4f(mix(then, now, uniforms.blend), range);
}
