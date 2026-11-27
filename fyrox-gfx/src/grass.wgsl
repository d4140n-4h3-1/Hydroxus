// Grass: clumps of grass cards drawn into the G-Buffer, round the eye, without a node or a mesh of
// their own (see grass.rs).
//
// One draw: an instance for each cell of a square of the ground round the eye, the square snapped
// to the cells so a clump stays where it grows as the eye moves; each a clump of three upright
// cards crossed at 60 degrees through its middle - as plant_life's woodland grass cards are made,
// six triangles - its texture one of the game's cards. Which card, where in its cell it stands,
// which way it faces, how big and how it is shaded all come of a hash of its cell, so the same
// clumps grow there every frame. The ground's height, how thick the grass is, and how damp the
// ground (where the bank grass grows) come from the game's map of the field. A clump not drawn -
// past the radius, out of view, where no grass grows, or thinned out with distance - has its
// vertices put outside the clip volume, and no triangle of it is drawn.
//
// The wind bends the cards' tops along it, their bases still, in gusts that run over the field.

struct Frame {
    viewProjection: mat4x4f,
    // xyz: the eye; w: the time, in seconds.
    eye: vec4f,
    // xy: which way the wind blows, along x and z; z: how strongly, 0 to 1; w: how fast the gusts
    // run.
    wind: vec4f,
    // xy: the world x and z of the map's first sample; z: meters between samples; w: how many
    // cards there are.
    map: vec4f,
    // x: a cell's side, in meters; y: the radius grass grows within; z: cells across the square.
    grid: vec4f,
    // xy: the least and the most a clump is scaled; z: from what share of the radius on it thins
    // out; w: how much bigger the clumps grow at the radius, to cover as thickly with fewer.
    shape: vec4f,
    // x: roughness; y: how much light comes through; z: the alpha a card is cut out below; w: how
    // much each clump's shade varies.
    look: vec4f,
    // How common each card is, and how much more on damp ground.
    weights: vec4f,
    damp: vec4f,
    // Each card's width and height, in meters.
    sizes: array<vec4f, 4>,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var heights: texture_2d<f32>;
// r: how thick the grass is; g: how damp the ground.
@group(0) @binding(2) var ground: texture_2d<f32>;
@group(0) @binding(3) var cards: texture_2d_array<f32>;
@group(0) @binding(4) var cardSampler: sampler;

const CARD_VERTICES: u32 = 6u;
const CLUMP_VERTICES: u32 = 18u;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) texCoord: vec2f,
    // x: how far up the card, 0 to 1; y: the clump's shade.
    @location(2) along: vec2f,
    @location(3) @interpolate(flat) card: u32,
};

fn hash(p: vec3u) -> u32 {
    var h = p.x * 0x8da6b343u ^ p.y * 0xd8163841u ^ p.z * 0xcb1ab31fu;
    h ^= h >> 15u;
    h *= 0x2c1b3c6du;
    h ^= h >> 12u;
    h *= 0x297a2d39u;
    h ^= h >> 15u;
    return h;
}

fn unit(h: u32) -> f32 {
    return f32(h & 0xffffffu) / 16777216.0;
}

// A map's value at world x and z, between its samples.
fn sample(map: texture_2d<f32>, at: vec2f) -> vec4f {
    let size = vec2i(textureDimensions(map));
    let p = (at - frame.map.xy) / frame.map.z;
    let i = vec2i(floor(p));
    let f = p - floor(p);
    let a = textureLoad(map, clamp(i, vec2i(0), size - 1), 0);
    let b = textureLoad(map, clamp(i + vec2i(1, 0), vec2i(0), size - 1), 0);
    let c = textureLoad(map, clamp(i + vec2i(0, 1), vec2i(0), size - 1), 0);
    let d = textureLoad(map, clamp(i + vec2i(1, 1), vec2i(0), size - 1), 0);
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn outside() -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4f(0.0, 0.0, 2.0, 1.0);
    output.normal = vec3f(0.0, 1.0, 0.0);
    output.texCoord = vec2f(0.0);
    output.along = vec2f(0.0);
    output.card = 0u;
    return output;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let across = u32(frame.grid.z);
    let side = frame.grid.x;
    let radius = frame.grid.y;
    let first = vec2i(floor(frame.eye.xz / side)) - vec2i(i32(across / 2u));
    let cell = first + vec2i(i32(instance % across), i32(instance / across));
    let key = vec3u(bitcast<u32>(cell.x), bitcast<u32>(cell.y), 0x67u);
    let h0 = hash(key);
    let h1 = hash(key + vec3u(0u, 0u, 0x9e37u));
    let h2 = hash(key + vec3u(0u, 0u, 0x7f4au));
    let h3 = hash(key + vec3u(0u, 0u, 0x1234u));
    let root_xz = (vec2f(cell) + vec2f(unit(h0), unit(h1))) * side;
    let distance = length(root_xz - frame.eye.xz);
    if (distance > radius) {
        return outside();
    }
    // How thick the grass grows here, thinner further off: a clump is drawn if its own number
    // falls under that.
    let ground_here = sample(ground, root_xz);
    let fade = 1.0 - smoothstep(radius * frame.shape.z, radius, distance);
    if (unit(h2) >= ground_here.r * fade) {
        return outside();
    }
    // Which card: by the cards' weights, the damp ones more where it is damp.
    let count = u32(frame.map.w);
    let weights = frame.weights + frame.damp * ground_here.g;
    var total = 0.0;
    for (var c = 0u; c < count; c++) {
        total += weights[c];
    }
    var pick = unit(h3) * total;
    var card = 0u;
    for (var c = 0u; c < count; c++) {
        if (pick < weights[c]) {
            card = c;
            break;
        }
        pick -= weights[c];
        card = c;
    }
    // Bigger further off, where fewer are drawn, so the grass looks as thick.
    let grow = 1.0 + frame.shape.w * smoothstep(radius * frame.shape.z, radius, distance);
    let scale = mix(frame.shape.x, frame.shape.y, unit(h0 >> 9u)) * grow;
    let size = frame.sizes[card].xy * scale;
    let root = vec3f(root_xz.x, sample(heights, root_xz).r, root_xz.y);
    // Out of view: its middle off the screen by more than the clump's size, give or take.
    let middle = frame.viewProjection * vec4f(root + vec3f(0.0, 0.5 * size.y, 0.0), 1.0);
    let slack = middle.w * 0.1 + size.x;
    if (any(abs(middle.xy) > vec2f(middle.w + slack)) || middle.w < -size.x) {
        return outside();
    }

    // Which corner of which card: each a quad, two triangles, across and up it.
    let k = vertex % CARD_VERTICES;
    let plane = vertex / CARD_VERTICES;
    var corners = array<vec2f, 6>(vec2f(-1.0, 0.0), vec2f(1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 0.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
    let corner = corners[k];
    let yaw = unit(h1 >> 7u) * 3.14159265 + f32(plane) * 1.04719755;
    let across_card = vec3f(cos(yaw), 0.0, sin(yaw));
    let up = corner.y;
    // The wind: gusts running over the field along it, the tops bent the more.
    let wind_dir = vec3f(frame.wind.x, 0.0, frame.wind.y);
    let gust = 0.55 + 0.45 * sin(frame.eye.w * frame.wind.w - dot(root_xz, frame.wind.xy) * 0.45 + unit(h2 >> 3u) * 2.0);
    let flutter = 0.08 * sin(frame.eye.w * 5.3 + unit(h3 >> 5u) * 6.2831853);
    let blow = frame.wind.z * (gust + flutter) * size.y * 0.55 * up;
    let position = root + across_card * (corner.x * 0.5 * size.x)
        + wind_dir * blow
        + vec3f(0.0, size.y * up * (1.0 - 0.3 * frame.wind.z * gust * up), 0.0);

    // The card's face, blended towards up: a clump of thin blades is lit more as the ground under
    // it than as each card.
    let face = vec3f(-sin(yaw), 0.0, cos(yaw));
    let normal = normalize(mix(face, vec3f(0.0, 1.0, 0.0), 0.6));

    var output: VertexOutput;
    var clip = frame.viewProjection * vec4f(position, 1.0);
    // The renderer's matrices make OpenGL's clip depth, from -w to w; wgpu's is from 0 to w.
    clip.z = (clip.z + clip.w) * 0.5;
    output.position = clip;
    output.normal = normal;
    // u across the card; v from the top of its texture down to the soil.
    output.texCoord = vec2f(corner.x * 0.5 + 0.5, 1.0 - up);
    output.along = vec2f(up, 1.0 + frame.look.w * (unit(h0 >> 17u) - 0.5));
    output.card = card;
    return output;
}

struct GBufferOutput {
    @location(0) outColor: vec4f,
    @location(1) outNormal: vec4f,
    @location(2) outAmbient: vec4f,
    @location(3) outMaterial: vec4f,
    @location(4) outDecalMask: u32,
};

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front: bool) -> GBufferOutput {
    let texel = textureSample(cards, cardSampler, input.texCoord, input.card);
    if (texel.a < frame.look.z) {
        discard;
    }
    var output: GBufferOutput;
    output.outColor = vec4f(texel.rgb * input.along.y, 1.0);
    // Seen from behind, the face turned to the eye.
    var n = normalize(input.normal);
    if (!front) {
        n = normalize(vec3f(-n.x, n.y, -n.z));
    }
    output.outNormal = vec4f(n * 0.5 + 0.5, 1.0);
    output.outAmbient = vec4f(0.0, 0.0, 0.0, 1.0);
    // Not metallic; rough; darker down at the roots; and some light through, as leaves have.
    output.outMaterial = vec4f(0.0, frame.look.x, mix(0.45, 1.0, input.along.x), 1.0 - frame.look.y);
    output.outDecalMask = 0u;
    return output;
}
