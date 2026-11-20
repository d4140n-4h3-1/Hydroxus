// Yoctite's clusters, drawn straight from the geometry pool, each in a colour of its own.
//
// One instance of the draw is one cluster: the draw list says which instance of which mesh and
// which of the pool's clusters. Its vertex `i` is corner `i % 3` of triangle `i / 3`; past the
// cluster's last triangle, every corner is put at the same point, which draws nothing.

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
    // From the mesh's space to the world's: three rows of four.
    transform: array<vec4f, 3>,
    mesh: u32,
    flags: u32,
    pad0: u32,
    pad1: u32,
};

// A vertex is this many floats: position, normal, texture coordinates, tangent.
const VERTEX_FLOATS: u32 = 12u;

@group(0) @binding(0) var<uniform> viewProjection: mat4x4f;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> clusterVertices: array<u32>;
@group(0) @binding(3) var<storage, read> triangles: array<u32>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
@group(0) @binding(6) var<storage, read> drawList: array<vec2u>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) colour: vec3f,
    @location(1) normal: vec3f,
};

// A colour from a number, bright enough to light: neighbouring numbers far apart.
fn colour_of(n: u32) -> vec3f {
    var h = n * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    h = (h >> 22u) ^ h;
    return vec3f(f32(h & 0xFFu), f32((h >> 8u) & 0xFFu), f32((h >> 16u) & 0xFFu)) / 255.0 * 0.75 + 0.25;
}

fn place(instance: Instance, p: vec3f, w: f32) -> vec3f {
    let q = vec4f(p, w);
    return vec3f(dot(instance.transform[0], q), dot(instance.transform[1], q), dot(instance.transform[2], q));
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32, @builtin(instance_index) item: u32) -> VertexOutput {
    var output: VertexOutput;
    let drawn = drawList[item];
    let instance = instances[drawn.x];
    let cluster = clusters[drawn.y];
    let triangle = index / 3u;
    if (triangle >= cluster.triangleCount) {
        output.position = vec4f(0.0, 0.0, 2.0, 1.0);
        return output;
    }
    let packed = triangles[cluster.firstTriangle + triangle];
    let local = (packed >> (8u * (index % 3u))) & 0xFFu;
    let v = clusterVertices[cluster.firstVertex + local] * VERTEX_FLOATS;
    let position = vec3f(vertices[v], vertices[v + 1u], vertices[v + 2u]);
    let normal = vec3f(vertices[v + 3u], vertices[v + 4u], vertices[v + 5u]);

    var clip = viewProjection * vec4f(place(instance, position, 1.0), 1.0);
    // The renderer's matrices make OpenGL's clip depth, from -w to w; wgpu's is from 0 to w.
    clip.z = (clip.z + clip.w) * 0.5;
    output.position = clip;
    output.colour = colour_of(drawn.y);
    output.normal = place(instance, normal, 0.0);
    return output;
}

@fragment
fn fs_main(input: VertexOutput, @builtin(front_facing) front: bool) -> @location(0) vec4f {
    // Lit from above and a little to one side; a sheet seen from behind is lit as its back.
    var n = normalize(input.normal);
    if (!front) {
        n = -n;
    }
    let light = max(dot(n, normalize(vec3f(0.4, 1.0, 0.3))), 0.0);
    return vec4f(input.colour * (0.35 + 0.65 * light), 1.0);
}
