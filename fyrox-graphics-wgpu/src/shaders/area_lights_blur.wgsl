// Smooths the light the area lights gave each pixel, for the renderer to add to the frame.
//
// Each pixel sampled its lights at points turned a little from its neighbours', in a 4x4
// pattern; averaging a 5x5 block evens that out. Only neighbours on the same surface count - near
// the pixel's plane, and facing the same way - so the light does not bleed across edges.

struct Uniforms {
    inverseViewProjection: mat4x4f,
    screenSize: vec2f,
    flipV: u32,
    planeTolerance: f32,
};

@group(0) @binding(0) var gathered: texture_2d<f32>;
@group(0) @binding(1) var sceneDepth: texture_2d<f32>;
@group(0) @binding(2) var sceneNormal: texture_2d<f32>;
@group(0) @binding(3) var nearest: sampler;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

fn world_at(uv: vec2f, depth: f32) -> vec3f {
    let ndcY = select(uv.y * 2.0 - 1.0, 1.0 - uv.y * 2.0, uniforms.flipV != 0u);
    let world = uniforms.inverseViewProjection * vec4f(uv.x * 2.0 - 1.0, ndcY, depth * 2.0 - 1.0, 1.0);
    return world.xyz / world.w;
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let uv = fragCoord.xy / uniforms.screenSize;
    let depth = textureSampleLevel(sceneDepth, nearest, uv, 0.0).r;
    if (depth >= 1.0) {
        return vec4f(0.0);
    }
    let position = world_at(uv, depth);
    let normal = normalize(textureSampleLevel(sceneNormal, nearest, uv, 0.0).xyz * 2.0 - 1.0);
    // Further away, a pixel covers more of the world, and neighbours lie further off its plane.
    let tolerance = uniforms.planeTolerance * (1.0 + distance(position, world_at(uv, 0.0)) * 0.1);
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var y = -2; y <= 2; y++) {
        for (var x = -2; x <= 2; x++) {
            let at = uv + vec2f(f32(x), f32(y)) / uniforms.screenSize;
            let d = textureSampleLevel(sceneDepth, nearest, at, 0.0).r;
            if (d >= 1.0) {
                continue;
            }
            let n = normalize(textureSampleLevel(sceneNormal, nearest, at, 0.0).xyz * 2.0 - 1.0);
            let off = abs(dot(world_at(at, d) - position, normal));
            if (off > tolerance || dot(n, normal) < 0.9) {
                continue;
            }
            sum += textureSampleLevel(gathered, nearest, at, 0.0).rgb;
            weight += 1.0;
        }
    }
    return vec4f(sum / max(weight, 1.0), 1.0);
}
