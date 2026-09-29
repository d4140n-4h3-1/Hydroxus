enable wgpu_ray_query;

// Shadow rays, traced against the scene's geometry.
//
// The depth buffer says where each pixel is in the world. Rays are sent from there towards the
// light; if the acceleration structure reports anything in the way, that ray is in shadow. Unlike
// a shadow map this does not depend on what is on screen or on a map's resolution - the rays are
// traced against the triangles themselves.
//
// The light is either the sun, infinitely far away in one direction, or a lamp at a point, whose
// rays end where the lamp is so that nothing behind it can cast a shadow.
//
// Glass is traced too, but lets light through: its geometry is not opaque, so a ray passes on
// through it, and each copy of it carries in its custom data how much of the red, green and blue
// of the light it lets through, a byte each. A ray is lit by what is left of the light once every
// pane it crosses has taken its share - coloured by the glass - and by nothing past anything solid.
//
// A real light has a size, and a shadow's edge is soft where only part of it is hidden. With more
// than one sample, the rays spread over a disk the size of the light - or, for the sun, over the
// cone its disk fills in the sky - and the pixel gets the share of them that got through. The
// spread is turned a little from pixel to pixel in a 4x4 pattern, so that neighboring pixels test
// different parts of the light and a small blur afterwards averages them into a smooth edge.

struct Uniforms {
    inverseViewProjection: mat4x4f,
    lightDirection: vec3f,
    reach: f32,
    screenSize: vec2f,
    bias: f32,
    flipV: u32,
    lightPosition: vec3f,
    lightRadius: f32,
    positional: u32,
    sampleCount: u32,
    // The radius of a lamp in meters, or the tangent of the sun's angular radius.
    lightSize: f32,
    pad0: u32,
};

@group(0) @binding(0) var sceneGeometry: acceleration_structure;
@group(0) @binding(1) var sceneDepth: texture_depth_2d;
@group(0) @binding(2) var depthSampler: sampler;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var sceneNormal: texture_2d<f32>;

// A triangle covering the screen, from the vertex index alone.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

// How much of the light gets through along the ray within `reach`, colour by colour: none past
// anything solid - any solid hit will do, so the search stops at the first one - and through each
// pane of glass it crosses, as much as that glass lets through.
fn transmitted(origin: vec3f, direction: vec3f, reach: f32) -> vec3f {
    const TERMINATE_ON_FIRST_HIT = 0x4u;
    var query: ray_query;
    rayQueryInitialize(&query, sceneGeometry, RayDesc(
        TERMINATE_ON_FIRST_HIT,
        0xFFu,
        0.001,
        reach,
        origin,
        direction,
    ));
    var through = vec3f(1.0);
    // Glass is only a candidate, never taken for the hit: the ray goes on through it.
    while (rayQueryProceed(&query)) {
        let lets = rayQueryGetCandidateIntersection(&query).instance_custom_data;
        through *= vec3f(
            f32(lets & 0xFFu),
            f32((lets >> 8u) & 0xFFu),
            f32((lets >> 16u) & 0xFFu),
        ) / 255.0;
    }
    if (rayQueryGetCommittedIntersection(&query).kind != RAY_QUERY_INTERSECTION_NONE) {
        return vec3f(0.0);
    }
    return through;
}

// A direction at right angles to `n` (which must be unit length), without dividing by anything
// that can be zero.
fn tangent(n: vec3f) -> vec3f {
    let s = select(-1.0, 1.0, n.z >= 0.0);
    let a = -1.0 / (s + n.z);
    let b = n.x * n.y * a;
    return vec3f(1.0 + s * n.x * n.x * a, s * b, -s * n.x);
}

// How far to turn the sample pattern at this pixel, as a fraction of a full turn: every pixel of a
// 4x4 block gets a different one.
fn pattern_rotation(pixel: vec2i) -> f32 {
    var bayer = array<f32, 16>(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0
    );
    return bayer[(pixel.y & 3) * 4 + (pixel.x & 3)] / 16.0;
}

// Sample `i` of `count` on a unit disk spanned by `t` and `b`, spread evenly by the golden angle.
fn disk_sample(i: u32, count: u32, rotation: f32, t: vec3f, b: vec3f) -> vec3f {
    const GOLDEN_ANGLE = 2.39996323;
    const TAU = 6.28318531;
    let r = sqrt((f32(i) + 0.5) / f32(count));
    let theta = f32(i) * GOLDEN_ANGLE + rotation * TAU;
    return (t * cos(theta) + b * sin(theta)) * r;
}

// The world position a screen position and depth come from.
fn world_at(uv: vec2f, depth: f32) -> vec3f {
    // Render targets are stored top row first here, so screen v runs against clip-space y.
    let ndcY = select(uv.y * 2.0 - 1.0, 1.0 - uv.y * 2.0, uniforms.flipV != 0u);
    let world = uniforms.inverseViewProjection * vec4f(uv.x * 2.0 - 1.0, ndcY, depth * 2.0 - 1.0, 1.0);
    return world.xyz / world.w;
}

// How far off a position read back from the depth buffer can be: the distance a few steps of
// depth precision move it. Depth is stored with a fixed number of steps between the near and
// far planes, bunched up close to the camera, so this is well under a millimeter nearby and
// grows with the square of the distance - centimeters at a hundred meters.
fn depth_error(uv: vec2f, depth: f32, position: vec3f) -> f32 {
    const DEPTH_STEP = 1.0 / 16777216.0;
    const MARGIN = 4.0;
    return distance(world_at(uv, min(depth + DEPTH_STEP * MARGIN, 1.0)), position);
}

// How far off a position read back from the depth buffer can be, all told. Besides the depth
// buffer's own steps, turning depth back into a position in 32-bit floats loses more the
// further away it is, so the estimate never falls below a small share of the distance.
fn position_error(uv: vec2f, depth: f32, position: vec3f) -> f32 {
    // A millimeter per meter: nothing close up, where contact shadows matter, and enough far
    // away that distant walls lit at a grazing angle do not shadow themselves.
    const SHARE = 0.001;
    let range = distance(position, world_at(uv, 0.0));
    return max(depth_error(uv, depth, position), range * SHARE);
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(fragCoord.xy);
    let depth = textureLoad(sceneDepth, pixel, 0);
    // Nothing was drawn here, so nothing can be in shadow.
    if (depth >= 1.0) {
        return vec4f(1.0);
    }

    let uv = fragCoord.xy / uniforms.screenSize;
    let position = world_at(uv, depth);

    // The rays start a little way off along the surface's own normal rather than along the ray:
    // pushing them along the ray would step over anything thin standing close by, and then a
    // picket fence would cast no shadow. Far away, the position itself is only known to within
    // the depth buffer's precision, and a ray that starts behind the surface shadows the surface
    // itself - differently with every small move of the camera, so distant walls would flicker.
    // The offset covers that uncertainty as well.
    let normal = normalize(textureLoad(sceneNormal, pixel, 0).xyz * 2.0 - 1.0);
    let origin = position + normal * (uniforms.bias + position_error(uv, depth, position));

    // Where the middle of the light is, seen from here.
    var toLight = -normalize(uniforms.lightDirection);
    if (uniforms.positional != 0u) {
        let offset = uniforms.lightPosition - origin;
        let distance = length(offset);
        // Out of the lamp's reach, so it lights nothing here and there is nothing to shadow.
        if (distance >= uniforms.lightRadius || distance <= uniforms.bias) {
            return vec4f(1.0);
        }
        toLight = offset / distance;
    }
    // Facing away from the light: it lights nothing here whatever the rays would say.
    if (dot(normal, toLight) <= 0.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }

    let count = max(uniforms.sampleCount, 1u);
    let t = tangent(toLight);
    let b = cross(toLight, t);
    let rotation = pattern_rotation(pixel);
    var lit = vec3f(0.0);
    for (var i = 0u; i < count; i++) {
        var spread = vec3f(0.0);
        if (count > 1u) {
            spread = disk_sample(i, count, rotation, t, b) * uniforms.lightSize;
        }
        if (uniforms.positional != 0u) {
            // A point on the lamp. The ray stops there: anything beyond it is not between the
            // surface and the light.
            let offset = uniforms.lightPosition + spread - origin;
            let reach = length(offset);
            lit += transmitted(origin, offset / reach, reach - uniforms.bias);
        } else {
            lit += transmitted(origin, normalize(toLight + spread), uniforms.reach);
        }
    }
    return vec4f(lit / f32(count), 1.0);
}
