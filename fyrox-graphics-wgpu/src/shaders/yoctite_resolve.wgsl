enable wgpu_binding_array;

// Yoctite's software-rasterized pixels shaded into the G-Buffer (see yoctite_raster.wgsl).
//
// A triangle over the screen; each pixel the visibility buffer has a triangle for finds it again
// - its cluster in the lists, its corners in the pool - and works out where in it the pixel's
// middle is, and so its texture coordinates, normal and tangent, and how fast the texture
// coordinates change across the screen, for the textures' levels. It is shaded as the G-Buffer
// pass shades it (yoctite_gbuffer.wgsl), with its material's textures from arrays of every
// material's, and written with its own depth: the depth test against what the hardware drew keeps
// whichever is nearer.
//
// A copy of a mesh that bends is bent as the rasterizer bent it, its normals turned with it.
//
// The visibility buffer is read as two words a pixel: the low, which triangle; the high, one
// less the depth's bits - 0 where nothing was drawn.

struct Cluster {
    firstVertex: u32,
    firstTriangle: u32,
    vertexCount: u32,
    triangleCount: u32,
    bounds: vec4f,
    lodBounds: vec4f,
    parentBounds: vec4f,
    error: f32,
    parentError: f32,
    mesh: u32,
    level: u32,
};

struct Instance {
    transform: array<vec4f, 3>,
    mesh: u32,
    flags: u32,
    material: u32,
    morph: u32,
};

const NO_MORPH: u32 = 0xFFFFFFFFu;

struct Material {
    diffuseColor: vec4f,
    emissionStrength: vec3f,
    metallicFactor: f32,
    texCoordScale: vec2f,
    roughnessFactor: f32,
    layerIndex: u32,
    metallicChannel: u32,
    roughnessChannel: u32,
    slot: u32,
    // How much light comes through from behind: the material alpha is one minus it.
    transmission: f32,
};

struct Resolve {
    viewProjection: mat4x4f,
    screenSize: vec2u,
    flipV: u32,
    pad0: u32,
};

// A vertex is this many floats: position, normal, texture coordinates, tangent.
const VERTEX_FLOATS: u32 = 12u;
// Textures a material has in the arrays: diffuse, normal, metallic, roughness, occlusion,
// emission.
const TEXTURES: u32 = 6u;

@group(0) @binding(0) var<uniform> resolve: Resolve;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> clusterVertices: array<u32>;
@group(0) @binding(3) var<storage, read> triangles: array<u32>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
@group(0) @binding(6) var<storage, read> lists: array<vec2u>;
@group(0) @binding(7) var<storage, read> visibility: array<vec2u>;
@group(0) @binding(8) var<storage, read> materials: array<Material>;
// Arrays of textures may not share a bind group with a uniform buffer.
@group(1) @binding(0) var textures: binding_array<texture_2d<f32>>;
@group(1) @binding(1) var textureSampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

struct GBufferOutput {
    @location(0) outColor: vec4f,
    @location(1) outNormal: vec4f,
    @location(2) outAmbient: vec4f,
    @location(3) outMaterial: vec4f,
    @location(4) outDecalMask: u32,
    @builtin(frag_depth) depth: f32,
};

fn word(at: u32) -> f32 {
    return bitcast<f32>(vertices[at]);
}

// How far `instance`'s weights move vertex `v`, and turn its normal.
struct Bend {
    position: vec3f,
    normal: vec3f,
};
fn bend_of(instance: Instance, v: u32) -> Bend {
    var bend = Bend(vec3f(0.0), vec3f(0.0));
    if (instance.morph == NO_MORPH) {
        return bend;
    }
    let block = instance.morph;
    let targets = vertices[block + 2u];
    let offsets = vertices[block + 1u] + (v - vertices[block]) * targets * 3u;
    for (var t = 0u; t < targets; t++) {
        let weight = word(block + 3u + t);
        if (weight == 0.0) {
            continue;
        }
        let a = unpack2x16float(vertices[offsets + t * 3u]);
        let b = unpack2x16float(vertices[offsets + t * 3u + 1u]);
        let c = unpack2x16float(vertices[offsets + t * 3u + 2u]);
        bend.position += vec3f(a.x, a.y, b.x) * weight;
        bend.normal += vec3f(b.y, c.x, c.y) * weight;
    }
    return bend;
}

fn place(instance: Instance, p: vec3f, w: f32) -> vec3f {
    let q = vec4f(p, w);
    return vec3f(dot(instance.transform[0], q), dot(instance.transform[1], q), dot(instance.transform[2], q));
}

fn channel(value: vec4f, which: u32) -> f32 {
    return value[min(which, 3u)];
}

// The weights of a triangle's corners at screen point `p`, each corrected for perspective by the
// corner's 1/w: what interpolates the corners' attributes there.
fn weights(screen: array<vec2f, 3>, inverseW: vec3f, p: vec2f) -> vec3f {
    let w0 = (screen[1].x - p.x) * (screen[2].y - p.y) - (screen[2].x - p.x) * (screen[1].y - p.y);
    let w1 = (screen[2].x - p.x) * (screen[0].y - p.y) - (screen[0].x - p.x) * (screen[2].y - p.y);
    let w2 = (screen[0].x - p.x) * (screen[1].y - p.y) - (screen[1].x - p.x) * (screen[0].y - p.y);
    let perspective = vec3f(w0, w1, w2) * inverseW;
    return perspective / (perspective.x + perspective.y + perspective.z);
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> GBufferOutput {
    let pixel = vec2u(fragCoord.xy);
    let key = visibility[pixel.y * resolve.screenSize.x + pixel.x];
    if (key.y == 0u) {
        discard;
    }
    let entry = key.x >> 7u;
    let item = lists[entry];
    let instance = instances[item.x];
    let cluster = clusters[item.y];
    let packed = triangles[cluster.firstTriangle + (key.x & 127u)];

    var screen: array<vec2f, 3>;
    var inverseW: vec3f;
    var uv: array<vec2f, 3>;
    var normal: array<vec3f, 3>;
    var tangent: array<vec4f, 3>;
    let size = vec2f(resolve.screenSize);
    for (var k = 0u; k < 3u; k++) {
        let vertex = clusterVertices[cluster.firstVertex + ((packed >> (8u * k)) & 0xFFu)];
        let v = vertex * VERTEX_FLOATS;
        let bend = bend_of(instance, vertex);
        let position = vec3f(word(v), word(v + 1u), word(v + 2u)) + bend.position;
        let clip = resolve.viewProjection * vec4f(place(instance, position, 1.0), 1.0);
        let n = clip.xy / clip.w;
        let y = select(n.y * 0.5 + 0.5, 0.5 - n.y * 0.5, resolve.flipV != 0u);
        screen[k] = vec2f((n.x * 0.5 + 0.5) * size.x, y * size.y);
        inverseW[k] = 1.0 / clip.w;
        normal[k] = vec3f(word(v + 3u), word(v + 4u), word(v + 5u)) + bend.normal;
        uv[k] = vec2f(word(v + 6u), word(v + 7u));
        tangent[k] = vec4f(word(v + 8u), word(v + 9u), word(v + 10u), word(v + 11u));
    }
    // At the pixel's middle, and a pixel across and down: how fast the texture moves over the
    // screen, for its levels.
    let middle = fragCoord.xy;
    let b = weights(screen, inverseW, middle);
    let across = weights(screen, inverseW, middle + vec2f(1.0, 0.0));
    let down = weights(screen, inverseW, middle + vec2f(0.0, 1.0));
    let material = materials[instance.material];
    let scale = material.texCoordScale;
    let tc = (uv[0] * b.x + uv[1] * b.y + uv[2] * b.z) * scale;
    let dx = (uv[0] * across.x + uv[1] * across.y + uv[2] * across.z) * scale - tc;
    let dy = (uv[0] * down.x + uv[1] * down.y + uv[2] * down.z) * scale - tc;

    // As the G-Buffer pass turns them: by the transform's own 3x3, normalized.
    let n = normalize(place(instance, normal[0] * b.x + normal[1] * b.y + normal[2] * b.z, 0.0));
    let t4 = tangent[0] * b.x + tangent[1] * b.y + tangent[2] * b.z;
    let t = normalize(place(instance, t4.xyz, 0.0));
    let bitangent = normalize(tangent[0].w * cross(n, t));

    let base = instance.material * TEXTURES;
    var output: GBufferOutput;
    output.outColor = material.diffuseColor * textureSampleGrad(textures[base], textureSampler, tc, dx, dy);
    // Alpha tested where rasterized.
    output.outColor.a = 1.0;
    let tangentSpace = mat3x3f(t, bitangent, n);
    let mapped = normalize(textureSampleGrad(textures[base + 1u], textureSampler, tc, dx, dy) * 2.0 - 1.0);
    output.outNormal = vec4f(normalize(tangentSpace * mapped.xyz) * 0.5 + 0.5, 1.0);
    output.outMaterial.x = material.metallicFactor
        * channel(textureSampleGrad(textures[base + 2u], textureSampler, tc, dx, dy), material.metallicChannel);
    output.outMaterial.y = material.roughnessFactor
        * channel(textureSampleGrad(textures[base + 3u], textureSampler, tc, dx, dy), material.roughnessChannel);
    output.outMaterial.z = textureSampleGrad(textures[base + 4u], textureSampler, tc, dx, dy).r;
    output.outMaterial.w = 1.0 - material.transmission;
    output.outAmbient = vec4f(
        material.emissionStrength * textureSampleGrad(textures[base + 5u], textureSampler, tc, dx, dy).rgb,
        1.0,
    );
    output.outDecalMask = material.layerIndex;
    output.depth = 1.0 - bitcast<f32>(key.y);
    return output;
}
