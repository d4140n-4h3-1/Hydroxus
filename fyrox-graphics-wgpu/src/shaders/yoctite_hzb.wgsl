// Yoctite's depth pyramid (hierarchical Z): the depth buffer brought down level by level, each
// texel the farthest depth of the 2x2 below it, so a texel of level n holds the farthest of the
// 2^(n+1) by 2^(n+1) screen pixels it covers. Something whose nearest point is farther than that
// everywhere over its rectangle on screen is hidden there. Levels are rounded up in size, so the
// last row and column of a level cover what is left of the screen.
//
// `from_depth` makes level 0 from the depth buffer; `down` makes each level after from the one
// before.

@group(0) @binding(0) var sourceDepth: texture_depth_2d;
@group(0) @binding(1) var level0: texture_storage_2d<r32float, write>;

@compute @workgroup_size(8, 8)
fn from_depth(@builtin(global_invocation_id) id: vec3u) {
    let size = textureDimensions(level0);
    if (id.x >= size.x || id.y >= size.y) {
        return;
    }
    let last = vec2i(textureDimensions(sourceDepth)) - 1;
    let base = vec2i(id.xy) * 2;
    var farthest = 0.0;
    for (var dy = 0; dy < 2; dy++) {
        for (var dx = 0; dx < 2; dx++) {
            farthest = max(farthest, textureLoad(sourceDepth, min(base + vec2i(dx, dy), last), 0));
        }
    }
    textureStore(level0, id.xy, vec4f(farthest));
}
