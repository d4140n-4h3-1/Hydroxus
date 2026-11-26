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
// of the light it lets through, seven bits each, a byte apart. A ray is lit by what is left of the
// light once every pane it crosses has taken its share - coloured by the glass - and by nothing
// past anything solid.
//
// Cutouts - leaves, needles, grass - are not opaque either. Their custom data has CUTOUT_BIT set
// and says which of the scene's cutouts they are (see `cutout_blocks`, put in ahead of this
// shader): a hit on one stops the ray only where its texture is solid there, and elsewhere the ray
// goes on, as the renderer leaves those texels undrawn.
//
// A real light has a size, and a shadow's edge is soft where only part of it is hidden. The rays
// spread over a disk the size of the light - or, for the sun, over the cone its disk fills in the
// sky - and the pixel gets the share of them that got through. The spread is turned a little from
// traced pixel to traced pixel in a 3x3 pattern (see `pattern`), and with one ray reaches out a different way too, so that
// neighboring pixels test different parts of the light and a small blur afterwards averages them
// into a smooth edge. Where the frames are averaged after (temporal anti-aliasing), the pattern
// turns from frame to frame as well, so one ray a pixel covers the whole light over a few frames:
// soft shadows for the price of hard ones.
//
// At half resolution, one pixel of each 2x2 block is traced, a different one each frame where
// frames are averaged; the blur pass (shadow_blur.wgsl) fills in the rest from the traced
// pixels on the same surface.

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
    // How far this frame turns the pattern of rays over the light, and how far it moves them
    // out from its middle, as fractions of the whole: zero where frames are not averaged.
    turn: f32,
    reachOut: f32,
    // The mask's pixels to the screen's, across (1, or 2 at half resolution), and which pixel of
    // each block this frame traces.
    scale: u32,
    offsetX: u32,
    offsetY: u32,
};

@group(0) @binding(0) var sceneGeometry: acceleration_structure;
@group(0) @binding(1) var sceneDepth: texture_depth_2d;
@group(0) @binding(2) var depthSampler: sampler;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var sceneNormal: texture_2d<f32>;
// Metallic, roughness, occlusion; alpha one minus how much light comes through from behind.
@group(0) @binding(8) var sceneMaterial: texture_2d<f32>;

// A triangle covering the screen, from the vertex index alone.
@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

// How much of the light gets through along the ray within `reach`, colour by colour: none past
// anything solid - any solid hit will do, so the search stops at the first one, a solid texel of
// a cutout included - and through each pane of glass it crosses, as much as that glass lets
// through.
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
    // Glass is only a candidate, never taken for the hit: the ray goes on through it. A cutout
    // is taken for the hit where it is solid.
    while (rayQueryProceed(&query)) {
        let candidate = rayQueryGetCandidateIntersection(&query);
        let data = candidate.instance_custom_data;
        if ((data & CUTOUT_BIT) != 0u) {
            if (cutout_blocks(data, candidate.primitive_index, candidate.barycentrics)) {
                rayQueryConfirmIntersection(&query);
            }
            continue;
        }
        through *= vec3f(
            f32(data & 0x7Fu),
            f32((data >> 8u) & 0x7Fu),
            f32((data >> 16u) & 0x7Fu),
        ) / 127.0;
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

// Where on the light the ray of this traced pixel goes, before the turn of the frame: how far
// round, as a fraction of a full turn, and which of three rings, each a third of the disk's area.
//
// Every 3x3 traced pixels - what the blur averages - hold the nine of a pattern repeating every 3:
// three rings, one for each column, and on each ring three rays a third of a turn apart, one for
// each row, each ring turned a ninth of a turn from the last. Three rays a third of a turn apart
// cancel out, whatever ring they are on and however the frame turns them, so no frame's rays lean
// to any side of the light. With a 4x4 pattern of turns instead, the 3x3 of it the blur took did
// not cancel out: each frame's shadows leant a little to one side, and as the turn from frame to
// frame swung it round their edges changed angle, or, averaged over frames, jittered.
fn pattern(cell: vec2i) -> vec2f {
    let c = vec2f(vec2i(cell.x % 3, cell.y % 3));
    return vec2f(c.x / 9.0 + c.y / 3.0, c.x);
}

// Sample `i` of `count` on a unit disk spanned by `t` and `b`, spread evenly by the golden angle,
// `out` of the way through its ring of the disk (a half for the middle of it).
fn disk_sample(i: u32, count: u32, rotation: f32, out: f32, t: vec3f, b: vec3f) -> vec3f {
    const GOLDEN_ANGLE = 2.39996323;
    const TAU = 6.28318531;
    let r = sqrt((f32(i) + out) / f32(count));
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
    // The screen's pixel this one of the mask traces.
    let block = vec2i(fragCoord.xy) * i32(uniforms.scale);
    let pixel = min(block + vec2i(i32(uniforms.offsetX), i32(uniforms.offsetY)), vec2i(uniforms.screenSize) - 1);
    let depth = textureLoad(sceneDepth, pixel, 0);
    // Nothing was drawn here, so nothing can be in shadow.
    if (depth >= 1.0) {
        return vec4f(1.0);
    }

    let uv = (vec2f(pixel) + 0.5) / uniforms.screenSize;
    let position = world_at(uv, depth);

    // The rays start a little way off along the surface's own normal rather than along the ray:
    // pushing them along the ray would step over anything thin standing close by, and then a
    // picket fence would cast no shadow. Far away, the position itself is only known to within
    // the depth buffer's precision, and a ray that starts behind the surface shadows the surface
    // itself - differently with every small move of the camera, so distant walls would flicker.
    // The offset covers that uncertainty as well.
    let normal = normalize(textureLoad(sceneNormal, pixel, 0).xyz * 2.0 - 1.0);
    var origin = position + normal * (uniforms.bias + position_error(uv, depth, position));

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
    // Facing away from the light: it lights nothing here whatever the rays would say - unless
    // light comes through it from behind (a leaf), when the rays start off its other side, the
    // side the light falls on.
    if (dot(normal, toLight) <= 0.0) {
        if (textureLoad(sceneMaterial, pixel, 0).a > 0.99) {
            return vec4f(0.0, 0.0, 0.0, 1.0);
        }
        origin = position - normal * (uniforms.bias + position_error(uv, depth, position));
    }

    let count = max(uniforms.sampleCount, 1u);
    let t = tangent(toLight);
    let b = cross(toLight, t);
    // The pattern runs over the mask's own pixels - the ones traced - not the screen's: at half
    // resolution the traced pixels of a frame all sit at the same place in their 2x2 blocks, and
    // by the screen's pixels they took only a few of its places.
    let place = pattern(vec2i(fragCoord.xy));
    let rotation = fract(place.x + uniforms.turn);
    // With one ray, how far out it goes varies too - by the pattern's rings, and within each from
    // frame to frame - or every ray would keep to one ring round the middle. Within the three rings
    // the frame's rays go s, 1 - s and halfway out, in turn: so many out as in, every frame, and
    // the nine spread as far over the light every frame. Moved out together, a frame's rays spread
    // wider or narrower than the last, and the shadows' soft edges with them.
    var out = 0.5;
    if (count == 1u) {
        let ring = u32(place.y);
        let reach = uniforms.reachOut;
        let which = (ring + u32(uniforms.turn * 3.0)) % 3u;
        let within = select(select(0.5, 1.0 - reach, which == 1u), reach, which == 0u);
        out = (place.y + within) / 3.0;
    }
    var lit = vec3f(0.0);
    for (var i = 0u; i < count; i++) {
        var spread = vec3f(0.0);
        if (uniforms.lightSize > 0.0) {
            spread = disk_sample(i, count, rotation, out, t, b) * uniforms.lightSize;
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
