(
    name: "FyroxGfxTemporalAntiAliasing",

    resources: [
        (
            // This frame, drawn with this frame's sub-pixel shift.
            name: "currentColor",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 0
        ),
        (
            // Every frame so far, blended.
            name: "historyColor",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 1
        ),
        (
            name: "sceneDepth",
            kind: Texture(kind: DepthSampler2D, fallback: White),
            binding: 2
        ),
        (
            name: "properties",
            kind: PropertyGroup([
                (name: "worldViewProjection", kind: Matrix4()),
                (name: "inverseViewProjection", kind: Matrix4()),
                // Where the camera was last frame, to find where each point was on screen then.
                (name: "previousViewProjection", kind: Matrix4()),
                (name: "screenSize", kind: Vector2()),
                // Where each pixel's content is in this frame, which was drawn shifted by a
                // fraction of a pixel, as an offset in texture coordinates.
                (name: "jitterUv", kind: Vector2()),
                // How much of this frame goes into the result; the rest is history.
                (name: "blend", kind: Float(value: 0.1)),
                // Start over from this frame alone: the view jumped, and the history shows
                // somewhere else.
                (name: "reset", kind: Bool(value: false)),
                (name: "cameraPosition", kind: Vector4()),
                // What moves (see `moving.rs`): each capsule's ends, its radius in the w of the
                // first, and how far it moved since the frame before. A radius of zero is none.
                (name: "moving0Bottom", kind: Vector4()),
                (name: "moving0Top", kind: Vector4()),
                (name: "moving0Moved", kind: Vector4()),
                (name: "moving1Bottom", kind: Vector4()),
                (name: "moving1Top", kind: Vector4()),
                (name: "moving1Moved", kind: Vector4()),
            ]),
            binding: 0
        ),
    ],

    passes: [
        (
            name: "Primary",
            draw_parameters: DrawParameters(
                cull_face: None,
                color_write: ColorMask(
                    red: true,
                    green: true,
                    blue: true,
                    alpha: true,
                ),
                depth_write: false,
                stencil_test: None,
                depth_test: None,
                blend: None,
                stencil_op: StencilOp(
                    fail: Keep,
                    zfail: Keep,
                    zpass: Keep,
                    write_mask: 0xFFFF_FFFF,
                ),
                scissor_box: None
            ),

            vertex_shader:
                r#"
                    struct VertexInput {
                        @location(0) vertexPosition: vec3f,
                        @location(1) vertexTexCoord: vec2f,
                    };

                    struct VertexOutput {
                        @builtin(position) position: vec4f,
                        @location(0) texCoord: vec2f,
                    };

                    @vertex fn vs_main(input: VertexInput) -> VertexOutput {
                        var output: VertexOutput;
                        output.texCoord = input.vertexTexCoord;
                        output.position = properties.worldViewProjection * vec4f(input.vertexPosition, 1.0);
                        return output;
                    }
                "#,

            fragment_shader:
                r#"
                    // Blending bright and dark pixels as they are lets one bright pixel - a lamp
                    // panel, many times brighter than a wall - dominate the average, and edges
                    // against lamps would still flicker. The blending is done on colors squeezed
                    // into 0..1 instead, and the result stretched back.
                    fn squeeze(c: vec3f) -> vec3f {
                        return c / (1.0 + max(c.r, max(c.g, c.b)));
                    }

                    fn unsqueeze(c: vec3f) -> vec3f {
                        return c / max(1.0 - max(c.r, max(c.g, c.b)), 1.0e-4);
                    }

                    // How far `p` is from the segment from `a` to `b`.
                    fn toSegment(p: vec3f, a: vec3f, b: vec3f) -> f32 {
                        let ab = b - a;
                        let t = clamp(dot(p - a, ab) / max(dot(ab, ab), 1.0e-8), 0.0, 1.0);
                        return length(p - (a + ab * t));
                    }

                    // How close the segments from `p1` to `q1` and from `p2` to `q2` come.
                    fn betweenSegments(p1: vec3f, q1: vec3f, p2: vec3f, q2: vec3f) -> f32 {
                        let d1 = q1 - p1;
                        let d2 = q2 - p2;
                        let r = p1 - p2;
                        let a = dot(d1, d1);
                        let e = dot(d2, d2);
                        let f = dot(d2, r);
                        let c = dot(d1, r);
                        let b = dot(d1, d2);
                        let denominator = a * e - b * b;
                        var s = 0.0;
                        if (denominator > 1.0e-8) {
                            s = clamp((b * f - c * e) / denominator, 0.0, 1.0);
                        }
                        var t = (b * s + f) / max(e, 1.0e-8);
                        if (t < 0.0) {
                            t = 0.0;
                            s = clamp(-c / max(a, 1.0e-8), 0.0, 1.0);
                        } else if (t > 1.0) {
                            t = 1.0;
                            s = clamp((b - c) / max(a, 1.0e-8), 0.0, 1.0);
                        }
                        return length((p1 + d1 * s) - (p2 + d2 * t));
                    }

                    // Where a point on a moving thing was last frame, and whether the view to a
                    // point passes close by one: past its edges the history may still show it.
                    struct Moved {
                        world: vec3f,
                        on: bool,
                        near: bool,
                    };

                    // How far outside a moving thing's capsule the view still counts as passing
                    // close by it, in meters.
                    const NEAR_MARGIN: f32 = 0.2;

                    fn follow(moved: Moved, world: vec3f, bottom: vec4f, top: vec4f, by: vec4f) -> Moved {
                        var result = moved;
                        let radius = bottom.w;
                        if (radius <= 0.0) {
                            return result;
                        }
                        if (toSegment(world, bottom.xyz, top.xyz) < radius) {
                            result.world = world - by.xyz;
                            result.on = true;
                        } else if (betweenSegments(properties.cameraPosition.xyz, world, bottom.xyz, top.xyz) < radius + NEAR_MARGIN) {
                            result.near = true;
                        }
                        return result;
                    }

                    fn current(uv: vec2f) -> vec3f {
                        return squeeze(textureSampleLevel(currentColor_tex, currentColor_samp, uv, 0.0).rgb);
                    }

                    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
                        let texel = 1.0 / properties.screenSize;
                        let uv = fragCoord.xy * texel;
                        // The result is lined up with the view as it would be without this
                        // frame's shift, so it holds still while the shift changes. This frame's
                        // color for the pixel is where the shift moved it.
                        let here = current(uv + properties.jitterUv);

                        // The colors around this pixel this frame (the samples at pixel middles
                        // are exact, filtering or not). Whatever the history says
                        // must lie among them: where it does not, the history shows something
                        // that has moved away or come into view, and following it would leave a
                        // ghost behind.
                        var low = here;
                        var high = here;
                        for (var y = -1; y <= 1; y++) {
                            for (var x = -1; x <= 1; x++) {
                                let near = current(uv + vec2f(f32(x), f32(y)) * texel);
                                low = min(low, near);
                                high = max(high, near);
                            }
                        }

                        // Where this pixel's point was on screen last frame: from the depth buffer
                        // and the two camera positions, and for a point on something that moves,
                        // from where that was.
                        let depth = textureSampleLevel(sceneDepth_tex, sceneDepth_samp, uv + properties.jitterUv, 0);
                        let world = S_UnProject(vec3f(uv, depth), properties.inverseViewProjection);
                        var moved = Moved(world, false, false);
                        moved = follow(moved, world, properties.moving0Bottom, properties.moving0Top, properties.moving0Moved);
                        moved = follow(moved, world, properties.moving1Bottom, properties.moving1Top, properties.moving1Moved);
                        let previous = properties.previousViewProjection * vec4f(moved.world, 1.0);
                        let previousNdc = previous.xy / previous.w;
                        // Render targets are stored top row first here, so v runs against y.
                        let previousUv = vec2f(previousNdc.x * 0.5 + 0.5, 0.5 - previousNdc.y * 0.5);
                        let history = clamp(
                            squeeze(textureSampleLevel(historyColor_tex, historyColor_samp, previousUv, 0.0).rgb),
                            low,
                            high,
                        );

                        let offScreen = any(previousUv < vec2f(0.0)) || any(previousUv > vec2f(1.0));
                        let fresh = properties.reset != 0u || offScreen;
                        // The faster the view moves, the more this frame counts. Not everything
                        // stays put in the world - the flashlight's beam turns with the camera -
                        // and the history of what moved with the view would trail behind it.
                        // Fast motion hides aliasing anyway; it is smoothed again as it slows.
                        let motion = length((previousUv - uv) * properties.screenSize);
                        var blend = mix(properties.blend, 0.5, clamp(motion / 10.0, 0.0, 1.0));
                        // Past the edge of something moving, the history may still show it where
                        // it was a moment ago: this frame counts for more, so nothing trails.
                        if (moved.near) {
                            blend = max(blend, 0.5);
                        }
                        // On it, its limbs move against it too, which following it as a whole
                        // does not catch.
                        if (moved.on) {
                            blend = max(blend, 0.25);
                        }
                        let result = select(mix(history, here, blend), here, fresh);
                        return vec4f(unsqueeze(result), 1.0);
                    }
                "#,
        )
    ]
)
