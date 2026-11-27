// Grass: blades drawn into the G-Buffer, round the eye, without a node or a mesh of their own (see
// grass.rs).
//
// One draw: an instance for each cell of a square of the ground round the eye, the square
// snapped to the cells so a blade stays where it grows as the eye moves; each cell a few blades,
// each blade BLADE_VERTICES vertices - a strip of SEGMENTS quads, narrowing, and a triangle at
// its tip. Where each blade stands, which way it faces, how tall, how far it leans and its shade
// all come of a hash of its cell and its number, so the same blades grow there every frame. The
// ground's height and how thick the grass grows on it come from the game's map of the field.
// A blade not drawn - past the radius, out of view, where no grass grows, or thinned out with
// distance - has its vertices put outside the clip volume, and no triangle of it is drawn.
//
// The wind bends each blade along its direction, more the nearer the tip, in gusts that run over
// the field. Blades are solid geometry, not cards: no fragment is thrown away, and the depth test
// keeps the ones behind from being shaded at all.

struct Frame {
    viewProjection: mat4x4f,
    // xyz: the eye; w: the time, in seconds.
    eye: vec4f,
    // xy: which way the wind blows, along x and z; z: how strongly, 0 to 1; w: how fast the gusts
    // run.
    wind: vec4f,
    // xy: the world x and z of the map's first sample; z: meters between samples.
    map: vec4f,
    // x: a cell's side, in meters; y: the radius grass grows within; z: cells across the square;
    // w: blades a cell at most.
    grid: vec4f,
    // x: a blade's height, in meters; y: its width at the root; z: from what share of the radius
    // on it thins out; w: how far it leans, at most, as a share of its height.
    shape: vec4f,
    // rgb: the colour at the root and at the tip; w: roughness, and how much light comes through.
    root: vec4f,
    tip: vec4f,
};

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var heights: texture_2d<f32>;
@group(0) @binding(2) var density: texture_2d<f32>;

const SEGMENTS: u32 = 4u;
const BLADE_VERTICES: u32 = 21u;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    // x: how far up the blade, 0 to 1; y: its shade, 0 to 1.
    @location(1) along: vec2f,
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
fn sample(map: texture_2d<f32>, at: vec2f) -> f32 {
    let size = vec2i(textureDimensions(map));
    let p = (at - frame.map.xy) / frame.map.z;
    let i = vec2i(floor(p));
    let f = p - floor(p);
    let a = textureLoad(map, clamp(i, vec2i(0), size - 1), 0).r;
    let b = textureLoad(map, clamp(i + vec2i(1, 0), vec2i(0), size - 1), 0).r;
    let c = textureLoad(map, clamp(i + vec2i(0, 1), vec2i(0), size - 1), 0).r;
    let d = textureLoad(map, clamp(i + vec2i(1, 1), vec2i(0), size - 1), 0).r;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

fn outside() -> VertexOutput {
    var output: VertexOutput;
    output.position = vec4f(0.0, 0.0, 2.0, 1.0);
    output.normal = vec3f(0.0, 1.0, 0.0);
    output.along = vec2f(0.0);
    return output;
}

@vertex
fn vs_main(@builtin(vertex_index) vertex: u32, @builtin(instance_index) instance: u32) -> VertexOutput {
    let across = u32(frame.grid.z);
    let side = frame.grid.x;
    let radius = frame.grid.y;
    // The cell, in whole cells from the world's origin.
    let first = vec2i(floor(frame.eye.xz / side)) - vec2i(i32(across / 2u));
    let cell = first + vec2i(i32(instance % across), i32(instance / across));
    let blade = vertex / BLADE_VERTICES;
    if (blade >= u32(frame.grid.w)) {
        return outside();
    }
    let key = vec3u(bitcast<u32>(cell.x), bitcast<u32>(cell.y), blade);
    let h0 = hash(key);
    let h1 = hash(key + vec3u(0u, 0u, 0x9e37u));
    let h2 = hash(key + vec3u(0u, 0u, 0x7f4au));
    let root_xz = (vec2f(cell) + vec2f(unit(h0), unit(h1))) * side;
    let distance = length(root_xz - frame.eye.xz);
    if (distance > radius) {
        return outside();
    }
    // How thick the grass grows here, and thinner further off: a blade is drawn if its own
    // number falls under that.
    let fade = 1.0 - smoothstep(radius * frame.shape.z, radius, distance);
    let thickness = sample(density, root_xz) * fade;
    if (unit(h2) >= thickness) {
        return outside();
    }
    let ground = sample(heights, root_xz);
    let root = vec3f(root_xz.x, ground, root_xz.y);
    // Out of view: the blade's root and tip both off the same side of the screen, give or take.
    let tall = frame.shape.x * (0.6 + 0.8 * unit(h0 >> 7u));
    let middle = frame.viewProjection * vec4f(root + vec3f(0.0, 0.5 * tall, 0.0), 1.0);
    let slack = middle.w * 0.1 + tall;
    if (any(abs(middle.xy) > vec2f(middle.w + slack)) || middle.w < -tall) {
        return outside();
    }

    // Which point of the blade: a quad for each segment, then the tip.
    let k = vertex % BLADE_VERTICES;
    var level: u32;
    var across_blade: f32;
    let segment = k / 6u;
    if (segment < SEGMENTS - 1u) {
        let corner = k % 6u;
        // (level up, side): the quad's two triangles.
        var levels = array<u32, 6>(0u, 0u, 1u, 1u, 0u, 1u);
        var sides = array<f32, 6>(-1.0, 1.0, -1.0, -1.0, 1.0, 1.0);
        level = segment + levels[corner];
        across_blade = sides[corner];
    } else {
        let corner = k - 6u * (SEGMENTS - 1u);
        var levels = array<u32, 3>(SEGMENTS - 1u, SEGMENTS - 1u, SEGMENTS);
        var sides = array<f32, 3>(-1.0, 1.0, 0.0);
        level = levels[min(corner, 2u)];
        across_blade = sides[min(corner, 2u)];
    }
    let t = f32(level) / f32(SEGMENTS);

    let yaw = unit(hash(key + vec3u(0u, 0u, 0x1234u))) * 6.2831853;
    let width_dir = vec3f(cos(yaw), 0.0, sin(yaw));
    let facing = vec3f(-sin(yaw), 0.0, cos(yaw));
    // Wider blades further off, where fewer are drawn, so the grass looks as thick.
    let width = frame.shape.y * (1.0 + 1.5 * distance / radius) * (1.0 - t);
    let lean = frame.shape.w * unit(h1 >> 5u) * tall;
    // The wind: gusts running over the field along it, bending the blade more towards its tip.
    let wind_dir = vec3f(frame.wind.x, 0.0, frame.wind.y);
    let gust = 0.55 + 0.45 * sin(frame.eye.w * frame.wind.w - dot(root_xz, frame.wind.xy) * 0.35 + unit(h2 >> 3u) * 1.5);
    let blow = frame.wind.z * gust * tall * 0.6;
    let bend = t * t;
    let up = tall * t * (1.0 - 0.25 * bend * (frame.wind.z * gust + frame.shape.w));
    let position = root + width_dir * (across_blade * 0.5 * width) + facing * (lean * bend) + wind_dir * (blow * bend) + vec3f(0.0, up, 0.0);

    // The blade's face, tipped back as it bends, and rounded across it; blended towards up, as a
    // field of thin blades is lit more as the ground under them than as each blade.
    let slope = 2.0 * t * (lean * facing + blow * wind_dir) / max(tall, 1.0e-3);
    let face = normalize(facing - vec3f(0.0, dot(slope, facing), 0.0) + width_dir * (across_blade * 0.35));
    let normal = normalize(mix(face, vec3f(0.0, 1.0, 0.0), 0.55));

    var output: VertexOutput;
    var clip = frame.viewProjection * vec4f(position, 1.0);
    // The renderer's matrices make OpenGL's clip depth, from -w to w; wgpu's is from 0 to w.
    clip.z = (clip.z + clip.w) * 0.5;
    output.position = clip;
    output.normal = normal;
    output.along = vec2f(t, unit(h0 >> 13u));
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
    var output: GBufferOutput;
    let shade = 0.8 + 0.4 * input.along.y;
    let colour = mix(frame.root.rgb, frame.tip.rgb, smoothstep(0.0, 1.0, input.along.x)) * shade;
    output.outColor = vec4f(colour, 1.0);
    // Seen from behind, the face turned to the eye.
    var n = normalize(input.normal);
    if (!front) {
        n = normalize(vec3f(-n.x, n.y, -n.z));
    }
    output.outNormal = vec4f(n * 0.5 + 0.5, 1.0);
    output.outAmbient = vec4f(0.0, 0.0, 0.0, 1.0);
    // Not metallic; rough; darker down among the blades; and some light through, as leaves have.
    output.outMaterial = vec4f(0.0, frame.root.w, mix(0.35, 1.0, input.along.x), 1.0 - frame.tip.w);
    output.outDecalMask = 0u;
    return output;
}
