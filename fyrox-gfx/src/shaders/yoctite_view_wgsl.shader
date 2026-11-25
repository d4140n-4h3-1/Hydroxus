(
    name: "FyroxGfxYoctiteView",

    resources: [
        (
            // The clusters, drawn in their colours; alpha 0 where nothing was.
            name: "clusters",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 0
        ),
        (
            name: "properties",
            kind: PropertyGroup([
                (name: "worldViewProjection", kind: Matrix4()),
                (name: "screenSize", kind: Vector2()),
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
                    alpha: false,
                ),
                depth_write: false,
                stencil_test: None,
                depth_test: None,
                // In place of the frame where clusters were drawn, the frame left as it is
                // elsewhere.
                blend: Some(BlendParameters(
                    func: BlendFunc(
                        sfactor: SrcAlpha,
                        dfactor: OneMinusSrcAlpha,
                        alpha_sfactor: Zero,
                        alpha_dfactor: One,
                    ),
                    equation: BlendEquation(
                        rgb: Add,
                        alpha: Add
                    )
                )),
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
                    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
                        let uv = fragCoord.xy / properties.screenSize;
                        return textureSample(clusters_tex, clusters_samp, uv);
                    }
                "#,
        )
    ]
)
