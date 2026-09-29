// Smooths a shadow mask traced with a few rays per pixel.
//
// Each pixel of the mask tested a different part of the light (see shadow_rays.wgsl), so on its own
// a soft edge is a speckle of lit and shadowed pixels. Averaging a 5x5 neighborhood covers every
// turn of the 4x4 sample pattern and turns the speckle into a gradient. The average only takes in
// neighbors on the same surface - facing the same way and lying in the same plane - so a shadow
// does not bleed across a corner or from a wall onto the floor in front of it.

struct Uniforms {
    inverseViewProjection: mat4x4f,
    screenSize: vec2f,
    flipV: u32,
    // How far off the pixel's plane, in meters, a neighbor may lie and still count as the same
    // surface.
    planeTolerance: f32,
};

@group(0) @binding(0) var mask: texture_2d<f32>;
@group(0) @binding(1) var sceneDepth: texture_depth_2d;
@group(0) @binding(2) var sceneNormal: texture_2d<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

fn world_position(pixel: vec2i, depth: f32) -> vec3f {
    let uv = (vec2f(pixel) + 0.5) / uniforms.screenSize;
    let ndcY = select(uv.y * 2.0 - 1.0, 1.0 - uv.y * 2.0, uniforms.flipV != 0u);
    let clip = vec4f(uv.x * 2.0 - 1.0, ndcY, depth * 2.0 - 1.0, 1.0);
    let world = uniforms.inverseViewProjection * clip;
    return world.xyz / world.w;
}

// How far off a position read back from the depth buffer can be (see shadow_rays.wgsl).
fn position_error(pixel: vec2i, depth: f32, position: vec3f) -> f32 {
    const DEPTH_STEP = 1.0 / 16777216.0;
    const MARGIN = 4.0;
    const SHARE = 0.001;
    let stepped = distance(world_position(pixel, min(depth + DEPTH_STEP * MARGIN, 1.0)), position);
    let range = distance(position, world_position(pixel, 0.0));
    return max(stepped, range * SHARE);
}

fn normal_at(pixel: vec2i) -> vec3f {
    return normalize(textureLoad(sceneNormal, pixel, 0).xyz * 2.0 - 1.0);
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(fragCoord.xy);
    let depth = textureLoad(sceneDepth, pixel, 0);
    if (depth >= 1.0) {
        return vec4f(1.0);
    }
    let normal = normal_at(pixel);
    let position = world_position(pixel, depth);
    let last = vec2i(uniforms.screenSize) - 1;
    // Far away, positions on one flat wall disagree by the depth buffer's precision; they are
    // still the same surface.
    let tolerance = uniforms.planeTolerance + 2.0 * position_error(pixel, depth, position);

    var sum = vec3f(0.0);
    var weights = 0.0;
    for (var dy = -2; dy <= 2; dy++) {
        for (var dx = -2; dx <= 2; dx++) {
            let neighbor = clamp(pixel + vec2i(dx, dy), vec2i(0), last);
            let neighborDepth = textureLoad(sceneDepth, neighbor, 0);
            if (neighborDepth >= 1.0) {
                continue;
            }
            let facing = pow(max(dot(normal, normal_at(neighbor)), 0.0), 8.0);
            let offPlane = abs(dot(normal, world_position(neighbor, neighborDepth) - position));
            let weight = facing * max(1.0 - offPlane / tolerance, 0.0);
            sum += textureLoad(mask, neighbor, 0).rgb * weight;
            weights += weight;
        }
    }
    // The pixel itself always counts fully, so there is always something to divide by.
    return vec4f(sum / weights, 1.0);
}
