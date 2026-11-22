enable wgpu_binding_array;

// Yoctite's software rasterizer: clusters too small on screen for the hardware, whose triangles
// are a pixel or two - the hardware rasterizes in blocks of 2x2 pixels, and sets each triangle up
// at a fixed cost - drawn by a compute pass into a visibility buffer instead.
//
// A workgroup a cluster the culling listed for it (yoctite_cull.wgsl), a thread a triangle; the
// workgroups go through the list, each taking every so many, so a fixed dispatch covers a list of
// any length. Each
// triangle is placed and projected, turned away from the eye left out as the hardware leaves it
// out, and each pixel whose middle it covers - and whose texel of the material's diffuse texture
// is solid, as the alpha test of the G-Buffer pass has it - keeps whichever of the triangles over
// it is nearest: 64 bits a pixel, the nearness in the high 32 (one less the depth, as a float's
// bits, which order as the floats do), and which triangle in the low 32 - its cluster's place in
// the lists, times 128, plus the triangle. `atomicMax` keeps the nearest. A pass after
// (yoctite_resolve.wgsl) shades each pixel from its triangle into the G-Buffer.
//
// A copy of a mesh that bends is bent as the G-Buffer pass bends it (see yoctite_gbuffer.wgsl):
// its vertices moved by its weights, read with the offsets from the vertex buffer, as words.

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
    pad0: u32,
};

struct Raster {
    viewProjection: mat4x4f,
    screenSize: vec2u,
    // Where the phase's software list starts in the lists, which of the counts says how many it
    // listed, and how many it holds.
    listStart: u32,
    counted: u32,
    flipV: u32,
    capacity: u32,
    pad1: u32,
    pad2: u32,
};

// A vertex is this many floats: position, normal, texture coordinates, tangent.
const VERTEX_FLOATS: u32 = 12u;
// A triangle is given up past this many pixels: a cluster this small on screen has none so big
// unless it crosses the near plane, which the hardware path clips and this cannot.
const MAX_PIXELS: u32 = 4096u;

@group(0) @binding(0) var<uniform> raster: Raster;
@group(0) @binding(1) var<storage, read> vertices: array<u32>;
@group(0) @binding(2) var<storage, read> clusterVertices: array<u32>;
@group(0) @binding(3) var<storage, read> triangles: array<u32>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
@group(0) @binding(6) var<storage, read> lists: array<vec2u>;
@group(0) @binding(7) var<storage, read> counts: array<u32>;
@group(0) @binding(8) var<storage, read_write> visibility: array<atomic<u64>>;
@group(0) @binding(9) var<storage, read> materials: array<Material>;
// Arrays of textures may not share a bind group with a uniform buffer.
@group(1) @binding(0) var diffuse: binding_array<texture_2d<f32>>;
@group(1) @binding(1) var textureSampler: sampler;

fn place(instance: Instance, p: vec3f) -> vec3f {
    let q = vec4f(p, 1.0);
    return vec3f(dot(instance.transform[0], q), dot(instance.transform[1], q), dot(instance.transform[2], q));
}

fn word(at: u32) -> f32 {
    return bitcast<f32>(vertices[at]);
}

// A vertex of the pool: where `instance` has it, bent, and its texture coordinates.
fn position_of(instance: Instance, v: u32) -> vec3f {
    let at = v * VERTEX_FLOATS;
    var position = vec3f(word(at), word(at + 1u), word(at + 2u));
    if (instance.morph != NO_MORPH) {
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
            position += vec3f(a.x, a.y, b.x) * weight;
        }
    }
    return position;
}
fn uv_of(v: u32) -> vec2f {
    let at = v * VERTEX_FLOATS;
    return vec2f(word(at + 6u), word(at + 7u));
}

@compute @workgroup_size(128)
fn rasterize(
    @builtin(workgroup_id) group: vec3u,
    @builtin(num_workgroups) groups: vec3u,
    @builtin(local_invocation_index) local: u32,
) {
    let listed = min(counts[raster.counted], raster.capacity);
    for (var index = group.x; index < listed; index += groups.x) {
        draw_triangle(raster.listStart + index, local);
    }
}

// Triangle `local` of the cluster at `entry` of the lists, into the visibility buffer.
fn draw_triangle(entry: u32, local: u32) {
    let item = lists[entry];
    let instance = instances[item.x];
    let cluster = clusters[item.y];
    if (local >= cluster.triangleCount) {
        return;
    }
    let packed = triangles[cluster.firstTriangle + local];
    var screen: array<vec3f, 3>;
    var inverseW: array<f32, 3>;
    var uv: array<vec2f, 3>;
    var ndc: array<vec2f, 3>;
    let size = vec2f(raster.screenSize);
    for (var k = 0u; k < 3u; k++) {
        let v = clusterVertices[cluster.firstVertex + ((packed >> (8u * k)) & 0xFFu)];
        let clip = raster.viewProjection * vec4f(place(instance, position_of(instance, v)), 1.0);
        // Crossing the near plane, which only the hardware path clips: left out.
        if (clip.w <= 0.0) {
            return;
        }
        let n = clip.xyz / clip.w;
        ndc[k] = n.xy;
        let y = select(n.y * 0.5 + 0.5, 0.5 - n.y * 0.5, raster.flipV != 0u);
        // Depth as the renderer stores it: OpenGL's from -1 to 1 brought to 0 to 1.
        screen[k] = vec3f((n.x * 0.5 + 0.5) * size.x, y * size.y, n.z * 0.5 + 0.5);
        inverseW[k] = 1.0 / clip.w;
        uv[k] = uv_of(v);
    }
    // Facing away - clockwise as the camera sees it - left out, as the G-Buffer pass culls it.
    let facing = (ndc[1].x - ndc[0].x) * (ndc[2].y - ndc[0].y) - (ndc[2].x - ndc[0].x) * (ndc[1].y - ndc[0].y);
    if (facing <= 0.0) {
        return;
    }
    let low = max(vec2i(floor(min(min(screen[0].xy, screen[1].xy), screen[2].xy))), vec2i(0));
    let high = min(vec2i(ceil(max(max(screen[0].xy, screen[1].xy), screen[2].xy))), vec2i(raster.screenSize) - 1);
    if (any(high < low) || u32(high.x - low.x + 1) * u32(high.y - low.y + 1) > MAX_PIXELS) {
        return;
    }
    // Twice the triangle's area on screen, signed as its edges' functions are.
    let area = (screen[1].x - screen[0].x) * (screen[2].y - screen[0].y) - (screen[2].x - screen[0].x) * (screen[1].y - screen[0].y);
    if (abs(area) < 1.0e-12) {
        return;
    }

    let material = materials[instance.material];
    // The level of the diffuse texture to test: as many texels to a pixel as the triangle has.
    let texels = vec2f(textureDimensions(diffuse[instance.material]));
    let uvScale = material.texCoordScale;
    let uvArea = abs((uv[1].x - uv[0].x) * (uv[2].y - uv[0].y) - (uv[2].x - uv[0].x) * (uv[1].y - uv[0].y))
        * uvScale.x * uvScale.y * texels.x * texels.y;
    let level = max(0.5 * log2(max(uvArea / abs(area), 1.0e-8)), 0.0);
    let payload = (entry << 7u) | local;

    for (var y = low.y; y <= high.y; y++) {
        for (var x = low.x; x <= high.x; x++) {
            let p = vec2f(f32(x) + 0.5, f32(y) + 0.5);
            // The pixel's middle as weights of the corners, each from the edge across from it.
            let w0 = (screen[1].x - p.x) * (screen[2].y - p.y) - (screen[2].x - p.x) * (screen[1].y - p.y);
            let w1 = (screen[2].x - p.x) * (screen[0].y - p.y) - (screen[0].x - p.x) * (screen[2].y - p.y);
            let w2 = (screen[0].x - p.x) * (screen[1].y - p.y) - (screen[1].x - p.x) * (screen[0].y - p.y);
            let b = vec3f(w0, w1, w2) / area;
            if (any(b < vec3f(0.0))) {
                continue;
            }
            // Depth is linear on the screen; texture coordinates are not.
            let depth = dot(b, vec3f(screen[0].z, screen[1].z, screen[2].z));
            if (depth <= 0.0 || depth >= 1.0) {
                continue;
            }
            let perspective = b * vec3f(inverseW[0], inverseW[1], inverseW[2]);
            let t = (uv[0] * perspective.x + uv[1] * perspective.y + uv[2] * perspective.z)
                / (perspective.x + perspective.y + perspective.z);
            let alpha = material.diffuseColor.a
                * textureSampleLevel(diffuse[instance.material], textureSampler, t * uvScale, level).a;
            if (alpha < 0.5) {
                continue;
            }
            let near = u64(bitcast<u32>(1.0 - depth));
            atomicMax(&visibility[u32(y) * raster.screenSize.x + u32(x)], (near << 32u) | u64(payload));
        }
    }
}
