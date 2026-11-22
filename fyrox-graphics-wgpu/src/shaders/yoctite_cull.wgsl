// Yoctite's culling: which clusters of which instances to draw this frame, chosen on the GPU, and
// sorted by the material they are drawn with.
//
// Passes, run one after the other:
//
// - `reset` clears the counters and the indirect arguments.
// - `cull_instances`, a thread an instance slot, keeps the instances whose mesh's bounds are at
//   least partly inside the view, listing them in `visible`.
// - `prepare` makes the dispatch for the next pass: a workgroup each visible instance, spread
//   over two dimensions, since one holds at most 65535 workgroups.
// - `cull_clusters`, a workgroup an instance, its threads going through the mesh's clusters,
//   lists each cluster whose own error looks no bigger than the threshold from the eye and whose
//   parent group's does - the cut through the mesh's levels of detail - and whose bounds are in
//   view, counting it for its instance's material.
// - `offsets` gives each material its range of the sorted list, and its indirect draw.
// - `scatter` copies each listed cluster into its material's range.
//
// With a depth pyramid of the frame before (see yoctite_hzb.wgsl), culling is in two phases
// (Karis et al.'s two-pass occlusion culling). In the first, the main one, a cluster that was
// hidden behind the frame before's depths, seen as they were, is not listed but put aside; what
// is listed is drawn. A pyramid is then made of the depths drawn so far, and in the second phase
// `cull_occluded` tries each cluster put aside again, against it as this frame sees it, and lists
// those not hidden after all - newly uncovered, or wrongly put aside as the camera moved - which
// `offsets` and `scatter` sort into lists of their own for a second set of draws. What the first
// phase drew is never tested against depths it did not see, so nothing is lost, whatever moved.
//
// With `softwarePixels` above nothing, a cluster listed that is that small on the screen or
// smaller - its bounds' radius, in pixels - is listed for the software rasterizer instead
// (yoctite_raster.wgsl), whose triangles a pixel or two across the hardware draws poorly: in the
// first phase from 5 * `capacity`, in the second from 6 * `capacity`, counted in `counts[8]` and
// `counts[10]`, their triangles in `counts[9]` and `counts[11]`. Not one near enough to cross
// the near plane, which only the hardware clips.
//
// Errors and bounds are judged in the world: placed by the instance's transform, and grown by
// its largest scale. A sphere round another still is, so the parent's error never looks smaller
// than its child's, and the cut is whole, as `yoctite::select` makes it in the mesh's own space.
//
// The lists are one buffer, `capacity` entries each: the first phase's as listed, from 0, and
// sorted by material, from `capacity`; the clusters put aside, from 2 * `capacity`; the second
// phase's as listed, from 3 * `capacity`, and sorted, from 4 * `capacity`. The indirect arguments
// are one buffer too: the cluster pass's dispatch at word 0, a draw of every cluster the first
// phase listed at word 4, and each material's draw at word 8 + 4 * its slot - the second
// phase's `materialStride` slots on. The material ranges are the same, the second phase's
// `materialStride` on. A dispatch cannot read its arguments from a buffer it may write, so the
// passes dispatched from them do not bind it.

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

struct Mesh {
    firstCluster: u32,
    clusterCount: u32,
    firstVertex: u32,
    vertexCount: u32,
    bounds: vec4f,
};

struct Cull {
    // The view's six planes, facing in, each normal of unit length.
    planes: array<vec4f, 6>,
    eye: vec3f,
    // An error, over its distance, as big as may show: about a pixel's angle.
    threshold: f32,
    instanceSlots: u32,
    // How many entries each list holds.
    capacity: u32,
    // How many vertices a cluster's draw has: three for every triangle a cluster can have.
    clusterVertices: u32,
    materialSlots: u32,
    // The view the depth pyramid was built from, the screen's size in pixels, how many levels
    // the pyramid has, and whether there is one.
    pyramidViewProjection: mat4x4f,
    screenSize: vec2f,
    pyramidLevels: u32,
    pyramidValid: u32,
    // Which phase: 0 the first, 1 the second; and how far on the second phase's material ranges
    // and draws are.
    phase: u32,
    materialStride: u32,
    // How small a cluster's radius on the screen, in pixels, the software rasterizer draws it;
    // nothing for none.
    softwarePixels: f32,
    pad1: u32,
};

// A material's share of the sorted list: how many of its clusters were listed, where its range
// starts, and how far the scatter has filled it.
struct MaterialRange {
    count: atomic<u32>,
    offset: u32,
    cursor: atomic<u32>,
    pad0: u32,
};

// A parent error this big or more is a root's: nothing coarser replaces it.
const ROOT_ERROR: f32 = 1.0e30;
// One dispatch dimension's most workgroups.
const MAX_GROUPS: u32 = 65535u;
const GROUP_SIZE: u32 = 64u;
// Where in the arguments the draw of every cluster, and the first material's draw, are.
const ALL_DRAW: u32 = 4u;
const MATERIAL_DRAWS: u32 = 8u;

@group(0) @binding(0) var<uniform> cull: Cull;
@group(0) @binding(1) var<storage, read> clusters: array<Cluster>;
@group(0) @binding(2) var<storage, read> meshes: array<Mesh>;
@group(0) @binding(3) var<storage, read> instances: array<Instance>;
@group(0) @binding(4) var<storage, read_write> visible: array<u32>;
// Visible instances; clusters the first phase listed (however many the list holds), their
// triangles; 1 if a list ran out of room; clusters put aside as hidden behind the frame before's
// depths, their triangles; clusters the second phase listed, their triangles; and the clusters
// each phase listed for the software rasterizer, and their triangles.
@group(0) @binding(5) var<storage, read_write> counts: array<atomic<u32>, 12>;
@group(0) @binding(6) var<storage, read_write> lists: array<vec2u>;
@group(0) @binding(7) var<storage, read_write> materials: array<MaterialRange>;
@group(0) @binding(8) var pyramid: texture_2d<f32>;
@group(1) @binding(0) var<storage, read_write> args: array<u32>;

fn place(instance: Instance, p: vec3f) -> vec3f {
    let q = vec4f(p, 1.0);
    return vec3f(dot(instance.transform[0], q), dot(instance.transform[1], q), dot(instance.transform[2], q));
}

// The most the instance stretches anything: the longest of its transform's columns.
fn largest_scale(instance: Instance) -> f32 {
    let t = instance.transform;
    let x = vec3f(t[0].x, t[1].x, t[2].x);
    let y = vec3f(t[0].y, t[1].y, t[2].y);
    let z = vec3f(t[0].z, t[1].z, t[2].z);
    return sqrt(max(dot(x, x), max(dot(y, y), dot(z, z))));
}

fn in_view(center: vec3f, radius: f32) -> bool {
    for (var i = 0u; i < 6u; i++) {
        let plane = cull.planes[i];
        if (dot(plane.xyz, center) + plane.w < -radius) {
            return false;
        }
    }
    return true;
}

// How big `error`, judged over a sphere, looks from the eye: over the distance to the sphere's
// nearest point; as big as can be from inside it, unless there is no error.
fn projected(error: f32, center: vec3f, radius: f32) -> f32 {
    if (error <= 0.0) {
        return 0.0;
    }
    let distance = length(center - cull.eye) - radius;
    if (distance <= 0.0) {
        return 3.0e38;
    }
    return error / distance;
}

// Whether the sphere at `center` of `radius` was hidden behind what the depth pyramid holds, seen
// as it was built: its nearest point farther than the pyramid's farthest depth over every pixel
// it covers. Anything reaching behind the camera, or off the screen, is not.
fn hidden(center: vec3f, radius: f32) -> bool {
    if (cull.pyramidValid == 0u) {
        return false;
    }
    var low = vec2f(1.0e9);
    var high = vec2f(-1.0e9);
    var nearest = 1.0;
    for (var k = 0u; k < 8u; k++) {
        let corner = center + radius * vec3f(
            select(-1.0, 1.0, (k & 1u) != 0u),
            select(-1.0, 1.0, (k & 2u) != 0u),
            select(-1.0, 1.0, (k & 4u) != 0u),
        );
        let clip = cull.pyramidViewProjection * vec4f(corner, 1.0);
        if (clip.w <= 0.0) {
            return false;
        }
        let ndc = clip.xyz / clip.w;
        // Stored top row first: screen v runs against clip y. Depth as the renderer stores it,
        // OpenGL's from -1 to 1 brought to 0 to 1.
        let pixel = vec2f((ndc.x + 1.0) * 0.5, (1.0 - ndc.y) * 0.5) * cull.screenSize;
        low = min(low, pixel);
        high = max(high, pixel);
        nearest = min(nearest, (ndc.z + 1.0) * 0.5);
    }
    if (nearest <= 0.0) {
        return false;
    }
    low = max(low, vec2f(0.0));
    high = min(high, cull.screenSize - 1.0);
    if (any(high < low)) {
        return false;
    }
    // The level where the rectangle spans no more than two texels each way: a texel of level n
    // covers 2^(n+1) pixels across.
    let span = max(high.x - low.x, high.y - low.y);
    let level = min(u32(max(ceil(log2(max(span, 1.0))) - 1.0, 0.0)), cull.pyramidLevels - 1u);
    let size = vec2i(textureDimensions(pyramid, level));
    let texel = f32(1u << (level + 1u));
    let first = min(vec2i(low / texel), size - 1);
    let last = min(vec2i(high / texel), size - 1);
    var farthest = 0.0;
    for (var y = first.y; y <= last.y; y++) {
        for (var x = first.x; x <= last.x; x++) {
            farthest = max(farthest, textureLoad(pyramid, vec2i(x, y), i32(level)).r);
        }
    }
    return nearest > farthest;
}

// Whether the cluster at `center` of `radius`, in the world, is for the software rasterizer:
// small on the screen, and well clear of the near plane.
fn for_software(center: vec3f, radius: f32) -> bool {
    if (cull.softwarePixels <= 0.0) {
        return false;
    }
    let distance = length(center - cull.eye) - radius;
    if (distance <= max(radius, 0.5)) {
        return false;
    }
    return radius / distance / cull.threshold <= cull.softwarePixels;
}

// Lists a cluster for the software rasterizer, in this phase's list; one past its room counts as
// overflow, as for the other lists.
fn list_for_software(item: vec2u, triangles: u32) {
    let at = atomicAdd(&counts[8u + 2u * cull.phase], 1u);
    atomicAdd(&counts[9u + 2u * cull.phase], triangles);
    if (at < cull.capacity) {
        lists[(5u + cull.phase) * cull.capacity + at] = item;
    } else {
        atomicStore(&counts[3], 1u);
    }
}

// Where the phase's listing is counted, and where its lists and material ranges start.
fn listed_count() -> u32 {
    return select(1u, 6u, cull.phase == 1u);
}
fn list_start() -> u32 {
    return select(0u, 3u * cull.capacity, cull.phase == 1u);
}
fn sorted_start() -> u32 {
    return select(cull.capacity, 4u * cull.capacity, cull.phase == 1u);
}
fn range_start() -> u32 {
    return select(0u, cull.materialStride, cull.phase == 1u);
}

// A thread a material slot; the first also clears the counters and the shared arguments.
@compute @workgroup_size(64)
fn reset(@builtin(global_invocation_id) id: vec3u) {
    if (id.x == 0u) {
        for (var i = 0u; i < 12u; i++) {
            atomicStore(&counts[i], 0u);
        }
        args[0] = 0u;
        args[1] = 1u;
        args[2] = 1u;
        args[ALL_DRAW] = cull.clusterVertices;
        args[ALL_DRAW + 1u] = 0u;
        args[ALL_DRAW + 2u] = 0u;
        args[ALL_DRAW + 3u] = 0u;
    }
    if (id.x < cull.materialSlots) {
        // Both phases' ranges and draws.
        for (var phase = 0u; phase < 2u; phase++) {
            let m = id.x + phase * cull.materialStride;
            atomicStore(&materials[m].count, 0u);
            materials[m].offset = 0u;
            atomicStore(&materials[m].cursor, 0u);
            let at = MATERIAL_DRAWS + 4u * m;
            args[at] = cull.clusterVertices;
            args[at + 1u] = 0u;
            args[at + 2u] = 0u;
            args[at + 3u] = 0u;
        }
    }
}

@compute @workgroup_size(64)
fn cull_instances(@builtin(global_invocation_id) id: vec3u) {
    let slot = id.x;
    if (slot >= cull.instanceSlots) {
        return;
    }
    let instance = instances[slot];
    if (instance.flags == 0u) {
        return;
    }
    let mesh = meshes[instance.mesh];
    if (!in_view(place(instance, mesh.bounds.xyz), mesh.bounds.w * largest_scale(instance))) {
        return;
    }
    visible[atomicAdd(&counts[0], 1u)] = slot;
}

@compute @workgroup_size(1)
fn prepare() {
    let n = atomicLoad(&counts[0]);
    args[0] = min(n, MAX_GROUPS);
    args[1] = (n + MAX_GROUPS - 1u) / MAX_GROUPS;
    args[2] = 1u;
}

@compute @workgroup_size(64)
fn cull_clusters(@builtin(workgroup_id) group: vec3u, @builtin(local_invocation_id) local: vec3u) {
    let index = group.y * MAX_GROUPS + group.x;
    if (index >= atomicLoad(&counts[0])) {
        return;
    }
    let slot = visible[index];
    let instance = instances[slot];
    let mesh = meshes[instance.mesh];
    let scale = largest_scale(instance);
    let material = min(instance.material, cull.materialSlots - 1u);
    for (var i = local.x; i < mesh.clusterCount; i += GROUP_SIZE) {
        let c = mesh.firstCluster + i;
        let cluster = clusters[c];
        let own = projected(cluster.error * scale, place(instance, cluster.lodBounds.xyz), cluster.lodBounds.w * scale);
        if (own > cull.threshold) {
            continue;
        }
        if (cluster.parentError < ROOT_ERROR) {
            let parent = projected(
                cluster.parentError * scale,
                place(instance, cluster.parentBounds.xyz),
                cluster.parentBounds.w * scale,
            );
            if (parent <= cull.threshold) {
                continue;
            }
        }
        if (!in_view(place(instance, cluster.bounds.xyz), cluster.bounds.w * scale)) {
            continue;
        }
        // Hidden behind the frame before's depths: put aside, to be tried again once this
        // frame's are drawn.
        if (hidden(place(instance, cluster.bounds.xyz), cluster.bounds.w * scale)) {
            let aside = atomicAdd(&counts[4], 1u);
            atomicAdd(&counts[5], cluster.triangleCount);
            if (aside < cull.capacity) {
                lists[2u * cull.capacity + aside] = vec2u(slot, c);
            } else {
                atomicStore(&counts[3], 1u);
            }
            continue;
        }
        if (for_software(place(instance, cluster.bounds.xyz), cluster.bounds.w * scale)) {
            list_for_software(vec2u(slot, c), cluster.triangleCount);
            continue;
        }
        let at = atomicAdd(&counts[1], 1u);
        atomicAdd(&counts[2], cluster.triangleCount);
        if (at < cull.capacity) {
            lists[at] = vec2u(slot, c);
            atomicAdd(&materials[material].count, 1u);
        }
    }
}

// The second phase: each cluster put aside tried again against this frame's depths, and listed if
// it is not hidden by them. Each thread takes every so many, as `scatter` does.
@compute @workgroup_size(64)
fn cull_occluded(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let aside = min(atomicLoad(&counts[4]), cull.capacity);
    for (var i = id.x; i < aside; i += groups.x * GROUP_SIZE) {
        let item = lists[2u * cull.capacity + i];
        let instance = instances[item.x];
        let cluster = clusters[item.y];
        let scale = largest_scale(instance);
        if (hidden(place(instance, cluster.bounds.xyz), cluster.bounds.w * scale)) {
            continue;
        }
        if (for_software(place(instance, cluster.bounds.xyz), cluster.bounds.w * scale)) {
            list_for_software(item, cluster.triangleCount);
            continue;
        }
        // No more than were put aside, so there is always room.
        let at = atomicAdd(&counts[6], 1u);
        atomicAdd(&counts[7], cluster.triangleCount);
        lists[3u * cull.capacity + at] = item;
        let material = min(instance.material, cull.materialSlots - 1u);
        atomicAdd(&materials[cull.materialStride + material].count, 1u);
    }
}

@compute @workgroup_size(1)
fn offsets() {
    let n = atomicLoad(&counts[listed_count()]);
    if (cull.phase == 0u) {
        args[ALL_DRAW + 1u] = min(n, cull.capacity);
        if (n > cull.capacity) {
            atomicStore(&counts[3], 1u);
        }
    }
    var start = 0u;
    for (var i = 0u; i < cull.materialSlots; i++) {
        let m = range_start() + i;
        let count = atomicLoad(&materials[m].count);
        materials[m].offset = start;
        args[MATERIAL_DRAWS + 4u * m + 1u] = count;
        start += count;
    }
}

// Each thread takes every so many entries, so a list of any length takes no more workgroups
// than a dispatch can have.
@compute @workgroup_size(64)
fn scatter(@builtin(global_invocation_id) id: vec3u, @builtin(num_workgroups) groups: vec3u) {
    let listed = min(atomicLoad(&counts[listed_count()]), cull.capacity);
    for (var i = id.x; i < listed; i += groups.x * GROUP_SIZE) {
        let item = lists[list_start() + i];
        let material = range_start() + min(instances[item.x].material, cull.materialSlots - 1u);
        let at = materials[material].offset + atomicAdd(&materials[material].cursor, 1u);
        lists[sorted_start() + at] = item;
    }
}
