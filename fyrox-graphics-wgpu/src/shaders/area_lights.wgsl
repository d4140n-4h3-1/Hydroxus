// Area lights: glowing rectangles - a lit panel, the rim of a screen - that light what is round
// them from all of their surface at once, rather than from a point.
//
// Each pixel of the frame is worked out from the G-buffer: where it is, which way it faces, and
// its colour. For each light, points spread evenly over the rectangle each add the light they
// shine on the pixel - as much as the pixel faces them, as much as they face the pixel, falling off
// with the square of the distance - and the sum is what the whole rectangle gives. The spread is
// turned a little from pixel to pixel in a 4x4 pattern, and a blur afterwards averages it smooth.
//
// Where the hardware traces rays, each point is also checked for anything in the way, and for
// glass, which colours what it lets through: see `visible`, which is put in front of this source
// in one of two forms - traced, or everything visible.

struct Light {
    // A corner, and how far the light reaches from its middle.
    corner: vec4f,
    // The first edge from that corner, and whether it shines from both faces.
    edgeU: vec4f,
    // The second edge, and how many points along the first.
    edgeV: vec4f,
    // Its colour, times how bright it is, and how many points along the second edge.
    colour: vec4f,
};

struct Uniforms {
    inverseViewProjection: mat4x4f,
    screenSize: vec2f,
    flipV: u32,
    lightCount: u32,
    bias: f32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
    lights: array<Light, 64>,
};

@group(0) @binding(1) var sceneDepth: texture_2d<f32>;
@group(0) @binding(2) var nearest: sampler;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var sceneNormal: texture_2d<f32>;
@group(0) @binding(5) var sceneColour: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let x = f32(i32(index) / 2) * 4.0 - 1.0;
    let y = f32(i32(index) & 1) * 4.0 - 1.0;
    return vec4f(x, y, 0.0, 1.0);
}

fn world_at(uv: vec2f, depth: f32) -> vec3f {
    let ndcY = select(uv.y * 2.0 - 1.0, 1.0 - uv.y * 2.0, uniforms.flipV != 0u);
    let world = uniforms.inverseViewProjection * vec4f(uv.x * 2.0 - 1.0, ndcY, depth * 2.0 - 1.0, 1.0);
    return world.xyz / world.w;
}

fn pattern_rotation(pixel: vec2i) -> f32 {
    var bayer = array<f32, 16>(
        0.0, 8.0, 2.0, 10.0,
        12.0, 4.0, 14.0, 6.0,
        3.0, 11.0, 1.0, 9.0,
        15.0, 7.0, 13.0, 5.0
    );
    return bayer[(pixel.y & 3) * 4 + (pixel.x & 3)] / 16.0;
}

// The G-buffer keeps colours as they are painted; light adds up in linear terms.
fn to_linear(c: vec3f) -> vec3f {
    return select(pow((c + 0.055) / 1.055, vec3f(2.4)), c / 12.92, c <= vec3f(0.04045));
}

@fragment
fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
    let pixel = vec2i(fragCoord.xy);
    let uv = fragCoord.xy / uniforms.screenSize;
    let depth = textureSampleLevel(sceneDepth, nearest, uv, 0.0).r;
    if (depth >= 1.0) {
        return vec4f(0.0, 0.0, 0.0, 1.0);
    }
    let position = world_at(uv, depth);
    let normal = normalize(textureSampleLevel(sceneNormal, nearest, uv, 0.0).xyz * 2.0 - 1.0);
    let albedo = to_linear(textureSampleLevel(sceneColour, nearest, uv, 0.0).rgb);
    // Off the surface a little, far enough that it does not shadow itself.
    let range = distance(position, world_at(uv, 0.0));
    let origin = position + normal * (uniforms.bias + range * 0.001);

    let turn = pattern_rotation(pixel);
    var light = vec3f(0.0);
    for (var l = 0u; l < min(uniforms.lightCount, 64u); l++) {
        let area = uniforms.lights[l];
        let u = area.edgeU.xyz;
        let v = area.edgeV.xyz;
        let middle = area.corner.xyz + (u + v) * 0.5;
        let reach = area.corner.w;
        let away = distance(position, middle);
        if (away >= reach) {
            continue;
        }
        // Falls to nothing at its reach, smoothly.
        let fade = pow(saturate(1.0 - pow(away / reach, 4.0)), 2.0);
        let across = cross(u, v);
        let size = length(across);
        let facing = across / max(size, 1.0e-6);
        let countU = clamp(u32(area.edgeV.w), 1u, 16u);
        let countV = clamp(u32(area.colour.w), 1u, 16u);
        let stepU = 1.0 / f32(countU);
        let stepV = 1.0 / f32(countV);
        let piece = size * stepU * stepV;
        // Each point stands for a cell of the rectangle, and close up a cell does not light as a
        // point would - without end as the distance goes to nothing - but spread over its size:
        // so the distance counts at least that much, and a surface right by the light shows
        // no bright spot at each point.
        let soft = (dot(u, u) * stepU * stepU + dot(v, v) * stepV * stepV) * 0.25;
        var gathered = vec3f(0.0);
        for (var i = 0u; i < countU; i++) {
            for (var j = 0u; j < countV; j++) {
                // Each point in its own cell of a grid over the rectangle, moved about within it
                // from pixel to pixel.
                let a = (f32(i) + fract(turn + f32(j) * 0.618)) * stepU;
                let b = (f32(j) + fract(turn * 1.7 + f32(i) * 0.382)) * stepV;
                let spot = area.corner.xyz + u * a + v * b;
                let offset = spot - origin;
                let length2 = max(dot(offset, offset), 1.0e-4);
                let dist = sqrt(length2);
                let direction = offset / dist;
                let onto = dot(normal, direction);
                var outward = -dot(facing, direction);
                if (area.edgeU.w != 0.0) {
                    outward = abs(outward);
                }
                if (onto <= 0.0 || outward <= 0.0) {
                    continue;
                }
                gathered += visible(origin, direction, dist - uniforms.bias) * onto * outward * piece / (length2 + soft);
            }
        }
        light += gathered * area.colour.rgb * fade;
    }
    return vec4f(albedo * light / 3.14159265, 1.0);
}
