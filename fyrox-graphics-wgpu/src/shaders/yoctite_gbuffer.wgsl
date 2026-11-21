// Yoctite's clusters drawn into the renderer's G-Buffer, as its own deferred meshes are: lit,
// shadowed and fogged after it like everything else.
//
// One draw a material: the culling sorted the clusters it chose by material, and each draw's
// instances are its material's range of that list. An instance of the draw is a cluster; its
// vertex `i` is corner `i % 3` of triangle `i / 3`, fetched through the cluster from the pool,
// and past the cluster's last triangle every corner is put at one point, which draws nothing.
//
// What is written matches the engine's standard and glTF shaders' G-Buffer pass: the diffuse
// colour (texture times colour, alpha-tested at a half), the normal from the normal map in the
// surface's tangent space, metallic, roughness and occlusion, emission, and the decal layer.
// Metallic and roughness are read from the channel each material says, times its factor: the
// glTF shader's blue and green of one texture, the standard shader's red of two.

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
    pad0: u32,
};

struct MaterialRange {
    count: u32,
    offset: u32,
    cursor: u32,
    pad0: u32,
};

struct Frame {
    viewProjection: mat4x4f,
    // Where the sorted list starts in the lists, and its material ranges in theirs: the culling's
    // first phase's, or its second's.
    sortedStart: u32,
    rangeStart: u32,
    pad1: u32,
    pad2: u32,
};

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

// A vertex is this many floats: position, normal, texture coordinates, tangent.
const VERTEX_FLOATS: u32 = 12u;

@group(0) @binding(0) var<uniform> frame: Frame;
@group(0) @binding(1) var<storage, read> vertices: array<f32>;
@group(0) @binding(2) var<storage, read> clusterVertices: array<u32>;
@group(0) @binding(3) var<storage, read> triangles: array<u32>;
@group(0) @binding(4) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(5) var<storage, read> instances: array<Instance>;
@group(0) @binding(6) var<storage, read> lists: array<vec2u>;
@group(0) @binding(7) var<storage, read> ranges: array<MaterialRange>;

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var diffuseTexture: texture_2d<f32>;
@group(1) @binding(2) var normalTexture: texture_2d<f32>;
@group(1) @binding(3) var metallicTexture: texture_2d<f32>;
@group(1) @binding(4) var roughnessTexture: texture_2d<f32>;
@group(1) @binding(5) var aoTexture: texture_2d<f32>;
@group(1) @binding(6) var emissionTexture: texture_2d<f32>;
@group(1) @binding(7) var diffuseSampler: sampler;
@group(1) @binding(8) var normalSampler: sampler;
@group(1) @binding(9) var metallicSampler: sampler;
@group(1) @binding(10) var roughnessSampler: sampler;
@group(1) @binding(11) var aoSampler: sampler;
@group(1) @binding(12) var emissionSampler: sampler;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) normal: vec3f,
    @location(1) texCoord: vec2f,
    @location(2) tangent: vec3f,
    @location(3) binormal: vec3f,
};

fn place(instance: Instance, p: vec3f, w: f32) -> vec3f {
    let q = vec4f(p, w);
    return vec3f(dot(instance.transform[0], q), dot(instance.transform[1], q), dot(instance.transform[2], q));
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32, @builtin(instance_index) item: u32) -> VertexOutput {
    var output: VertexOutput;
    let drawn = lists[frame.sortedStart + ranges[frame.rangeStart + material.slot].offset + item];
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
    let uv = vec2f(vertices[v + 6u], vertices[v + 7u]);
    let tangent = vec4f(vertices[v + 8u], vertices[v + 9u], vertices[v + 10u], vertices[v + 11u]);

    var clip = frame.viewProjection * vec4f(place(instance, position, 1.0), 1.0);
    // The renderer's matrices make OpenGL's clip depth, from -w to w; wgpu's is from 0 to w.
    clip.z = (clip.z + clip.w) * 0.5;
    output.position = clip;
    // As the engine's shaders turn them: by the transform's own 3x3, normalized.
    output.normal = normalize(place(instance, normal, 0.0));
    output.tangent = normalize(place(instance, tangent.xyz, 0.0));
    output.binormal = normalize(tangent.w * cross(output.normal, output.tangent));
    output.texCoord = uv;
    return output;
}

struct GBufferOutput {
    @location(0) outColor: vec4f,
    @location(1) outNormal: vec4f,
    @location(2) outAmbient: vec4f,
    @location(3) outMaterial: vec4f,
    @location(4) outDecalMask: u32,
};

fn channel(value: vec4f, which: u32) -> f32 {
    return value[min(which, 3u)];
}

@fragment
fn fs_main(input: VertexOutput) -> GBufferOutput {
    let tc = input.texCoord * material.texCoordScale;
    var output: GBufferOutput;
    output.outColor = material.diffuseColor * textureSample(diffuseTexture, diffuseSampler, tc);
    // Alpha test.
    if (output.outColor.a < 0.5) {
        discard;
    }
    output.outColor.a = 1.0;
    let tangentSpace = mat3x3f(input.tangent, input.binormal, input.normal);
    let n = normalize(textureSample(normalTexture, normalSampler, tc) * 2.0 - 1.0);
    output.outNormal = vec4f(normalize(tangentSpace * n.xyz) * 0.5 + 0.5, 1.0);
    output.outMaterial.x = material.metallicFactor * channel(textureSample(metallicTexture, metallicSampler, tc), material.metallicChannel);
    output.outMaterial.y = material.roughnessFactor * channel(textureSample(roughnessTexture, roughnessSampler, tc), material.roughnessChannel);
    output.outMaterial.z = textureSample(aoTexture, aoSampler, tc).r;
    output.outMaterial.w = 1.0;
    output.outAmbient = vec4f(material.emissionStrength * textureSample(emissionTexture, emissionSampler, tc).rgb, 1.0);
    output.outDecalMask = material.layerIndex;
    return output;
}
