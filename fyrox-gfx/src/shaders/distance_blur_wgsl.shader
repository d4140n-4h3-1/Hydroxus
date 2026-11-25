(
    name: "FyroxGfxDistanceBlur",

    resources: [
        (
            // The frame as drawn so far.
            name: "sourceColor",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 0
        ),
        (
            name: "sceneDepth",
            kind: Texture(kind: DepthSampler2D, fallback: White),
            binding: 1
        ),
        (
            name: "properties",
            kind: PropertyGroup([
                (name: "worldViewProjection", kind: Matrix4()),
                (name: "inverseViewProjection", kind: Matrix4()),
                (name: "screenSize", kind: Vector2()),
                (name: "cameraPosition", kind: Vector4()),
                // Where blurring starts, in meters, and how much further on it is all it is.
                (name: "startAndSpan", kind: Vector2()),
                // How far it is spread there, in pixels.
                (name: "radius", kind: Float(value: 3.0)),
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
                    // Samples in a disc, spread evenly by the golden angle.
                    const TAPS: i32 = 16;
                    const GOLDEN: f32 = 2.39996323;

                    // How many pixels the surface seen at `uv` is spread over, by how far off it is.
                    fn spread(uv: vec2f) -> f32 {
                        let depth = textureSampleLevel(sceneDepth_tex, sceneDepth_samp, uv, 0);
                        let world = S_UnProject(vec3f(uv, depth), properties.inverseViewProjection);
                        let distance = length(world - properties.cameraPosition.xyz);
                        let t = clamp((distance - properties.startAndSpan.x) / properties.startAndSpan.y, 0.0, 1.0);
                        return t * properties.radius;
                    }

                    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
                        let texel = 1.0 / properties.screenSize;
                        let uv = fragCoord.xy * texel;
                        let here = textureSampleLevel(sourceColor_tex, sourceColor_samp, uv, 0.0).rgb;
                        let radius = spread(uv);
                        if (radius < 0.5) {
                            return vec4f(here, 1.0);
                        }
                        var sum = here;
                        var weight = 1.0;
                        for (var i = 0; i < TAPS; i++) {
                            let angle = f32(i) * GOLDEN;
                            let out = sqrt((f32(i) + 0.5) / f32(TAPS)) * radius;
                            let at = uv + vec2f(cos(angle), sin(angle)) * out * texel;
                            // Only as far as the sample would itself be spread: something near and
                            // sharp does not bleed into the far off round it.
                            let w = clamp(spread(at) - out + 1.0, 0.0, 1.0);
                            sum += textureSampleLevel(sourceColor_tex, sourceColor_samp, at, 0.0).rgb * w;
                            weight += w;
                        }
                        return vec4f(sum / weight, 1.0);
                    }
                "#,
        )
    ]
)
