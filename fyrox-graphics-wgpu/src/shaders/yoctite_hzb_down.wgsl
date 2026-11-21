// A level of Yoctite's depth pyramid from the one before it: each texel the farthest of the 2x2
// below it (see yoctite_hzb.wgsl).

@group(0) @binding(0) var below: texture_2d<f32>;
@group(0) @binding(1) var level: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn down(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(level);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let last = vec2i(textureDimensions(below)) - 1;
    let base = vec2i(id.xy) * 2;
    var farthest = 0.0;
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            farthest = max(farthest, textureLoad(below, min(base + vec2i(dx, dy), last), 0).r);
        }
    }
    textureStore(level, id.xy, vec4f(farthest));
}
