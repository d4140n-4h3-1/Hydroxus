(
    name: "FyroxGfxReflections",

    resources: [
        (
            // The frame before reflections were added.
            name: "sceneColor",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 0
        ),
        (
            name: "sceneDepth",
            kind: Texture(kind: DepthSampler2D, fallback: White),
            binding: 1
        ),
        (
            name: "sceneNormal",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 2
        ),
        (
            // How metallic each surface is, in red, and how rough, in green.
            name: "sceneMaterial",
            kind: Texture(kind: Sampler2D, fallback: Black),
            binding: 3
        ),
        (
            // The colour of each surface, which a metal tints what it reflects with.
            name: "sceneDiffuse",
            kind: Texture(kind: Sampler2D, fallback: White),
            binding: 4
        ),
        (
            name: "properties",
            kind: PropertyGroup([
                (name: "worldViewProjection", kind: Matrix4()),
                (name: "viewProjection", kind: Matrix4()),
                (name: "inverseViewProjection", kind: Matrix4()),
                (name: "cameraPosition", kind: Vector3()),
                (name: "screenSize", kind: Vector2()),
                // How far a reflected ray travels, in meters.
                (name: "reach", kind: Float(value: 12.0)),
                // How many steps it takes to get there. More steps cost more and miss less.
                (name: "steps", kind: Int(value: 24)),
                // How far behind a surface the ray may pass and still count as hitting it.
                (name: "thickness", kind: Float(value: 0.4)),
                // How much of the reflection is kept.
                (name: "strength", kind: Float(value: 0.35)),
                // Only surfaces facing at least this far upwards reflect; 1 is straight up.
                (name: "minUpwards", kind: Float(value: 0.7)),
                // How much a polished metal reflects, from 0 to 1.
                (name: "metalStrength", kind: Float(value: 0.8)),
                // How rough a surface can be and still reflect at all, from 0 to 1.
                (name: "maxRoughness", kind: Float(value: 0.9)),
                // Counts the frames, for the noise that spreads the rays to change every frame.
                (name: "frame", kind: Float(value: 0.0)),
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
                // Mixed into what is already there, by how much of the surface reflects.
                blend: Some(BlendParameters(
                    func: BlendFunc(
                        sfactor: SrcAlpha,
                        dfactor: OneMinusSrcAlpha,
                        alpha_sfactor: SrcAlpha,
                        alpha_dfactor: OneMinusSrcAlpha,
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
                    // Where a world-space point lands on screen: texture coordinates, and the
                    // depth the depth buffer would hold for it.
                    fn toScreen(point: vec3f) -> vec3f {
                        let clip = properties.viewProjection * vec4f(point, 1.0);
                        let ndc = clip.xyz / clip.w;
                        // Render targets are stored top row first here, so v runs against y.
                        return vec3f(ndc.x * 0.5 + 0.5, 0.5 - ndc.y * 0.5, ndc.z * 0.5 + 0.5);
                    }

                    fn worldAt(uv: vec2f, depth: f32) -> vec3f {
                        return S_UnProject(vec3f(uv, depth), properties.inverseViewProjection);
                    }

                    fn onScreen(screen: vec3f) -> bool {
                        return screen.x >= 0.0 && screen.x <= 1.0 && screen.y >= 0.0 && screen.y <= 1.0 && screen.z <= 1.0;
                    }

                    // Noise from 0 to 1 that changes from pixel to pixel with no pattern the eye
                    // picks out, and from frame to frame, for temporal anti-aliasing to average.
                    fn noise(pixel: vec2f, frame: f32) -> f32 {
                        let p = pixel + 5.588238 * frame;
                        return fract(52.9829189 * fract(dot(p, vec2f(0.06711056, 0.00583715))));
                    }

                    // Whether the ray, `travelled` along from `start`, has gone behind what is on
                    // screen there.
                    fn behind(start: vec3f, ray: vec3f, travelled: f32) -> bool {
                        let screen = toScreen(start + ray * travelled);
                        return textureSample(sceneDepth_tex, sceneDepth_samp, screen.xy) < screen.z;
                    }

                    @fragment fn fs_main(@builtin(position) fragCoord: vec4f) -> @location(0) vec4f {
                        let uv = fragCoord.xy / properties.screenSize;
                        let depth = textureSample(sceneDepth_tex, sceneDepth_samp, uv);
                        // Nothing was drawn here, so there is nothing to reflect in.
                        if (depth >= 1.0) {
                            return vec4f(0.0);
                        }

                        let normal = normalize(textureSample(sceneNormal_tex, sceneNormal_samp, uv).xyz * 2.0 - 1.0);
                        // Reflections are kept to floors and other surfaces facing upwards: a
                        // screen-space ray has nothing to go on where the reflected view leaves
                        // the screen, which is most of the time on walls facing the camera.
                        if (normal.y < properties.minUpwards) {
                            return vec4f(0.0);
                        }
                        let material = textureSample(sceneMaterial_tex, sceneMaterial_samp, uv);
                        let metallic = clamp(material.x, 0.0, 1.0);
                        let roughness = clamp(material.y, 0.0, 1.0);
                        if (roughness >= properties.maxRoughness) {
                            return vec4f(0.0);
                        }

                        let position = worldAt(uv, depth);
                        let toCamera = normalize(properties.cameraPosition - position);
                        let mirror = reflect(-toCamera, normal);
                        // A rough surface scatters what it reflects: each pixel sends its ray off a
                        // little to the side of the mirror's, further the rougher it is, a
                        // different way every frame, and anti-aliasing averages them into a blur.
                        let spin = 6.2831853 * noise(fragCoord.xy, properties.frame);
                        let spread = roughness * roughness * sqrt(noise(fragCoord.yx + 17.0, properties.frame));
                        let side = normalize(cross(mirror, select(vec3f(0.0, 1.0, 0.0), vec3f(1.0, 0.0, 0.0), abs(mirror.y) > 0.9)));
                        let up = cross(side, mirror);
                        var ray = normalize(mirror + spread * (cos(spin) * side + sin(spin) * up));
                        if (dot(ray, normal) <= 0.0) {
                            ray = mirror;
                        }
                        if (dot(ray, normal) <= 0.0) {
                            return vec4f(0.0);
                        }

                        let stepCount = max(properties.steps, 1);
                        let stepLength = properties.reach / f32(stepCount);
                        // Each pixel starts at its own point within the first step: steps of the
                        // same length everywhere would show the steps as bands across the floor.
                        // Never at the surface itself, which would only reflect itself.
                        var travelled = stepLength * (0.25 + noise(fragCoord.xy + 31.0, properties.frame));
                        var before = 0.0;
                        var hitUv = vec2f(0.0);
                        var confidence = 0.0;

                        for (var i: i32 = 0; i < stepCount; i++) {
                            let screen = toScreen(position + ray * travelled);
                            if (!onScreen(screen)) {
                                break;
                            }
                            if (behind(position, ray, travelled)) {
                                // Somewhere between the last step and this one the ray went behind
                                // something: halve the gap to find where, as near as can be.
                                var near = before;
                                var far = travelled;
                                for (var j: i32 = 0; j < 6; j++) {
                                    let middle = 0.5 * (near + far);
                                    if (behind(position, ray, middle)) {
                                        far = middle;
                                    } else {
                                        near = middle;
                                    }
                                }
                                let point = position + ray * far;
                                let found = toScreen(point);
                                let sceneDepth = textureSample(sceneDepth_tex, sceneDepth_samp, found.xy);
                                let gap = distance(worldAt(found.xy, sceneDepth), point);
                                // Only just behind it, it hit it: unless that is its back, which
                                // nothing reflected can be. Far behind, the ray passed behind
                                // something nearer the camera, and goes on to look past it.
                                let facing = normalize(textureSample(sceneNormal_tex, sceneNormal_samp, found.xy).xyz * 2.0 - 1.0);
                                let sure = 1.0 - smoothstep(0.5 * properties.thickness, properties.thickness, gap);
                                if (sure > 0.0 && dot(facing, ray) < 0.1) {
                                    hitUv = found.xy;
                                    travelled = far;
                                    confidence = sure;
                                    break;
                                }
                            }
                            before = travelled;
                            travelled += stepLength;
                        }

                        if (confidence <= 0.0) {
                            return vec4f(0.0);
                        }

                        // Fade out where the reflection runs off the screen, and as far as the
                        // ray reaches.
                        let edge = min(
                            min(hitUv.x, 1.0 - hitUv.x),
                            min(hitUv.y, 1.0 - hitUv.y)
                        );
                        let edgeFade = smoothstep(0.0, 0.15, edge);
                        let distanceFade = 1.0 - smoothstep(0.6, 1.0, travelled / properties.reach);
                        // A surface reflects more seen at a grazing angle than head on - a metal
                        // a lot at any angle - and less the rougher it is.
                        let grazing = pow(1.0 - clamp(dot(toCamera, normal), 0.0, 1.0), 2.0);
                        let shine = mix(properties.strength * mix(0.25, 1.0, grazing), properties.metalStrength, metallic);
                        let gloss = 1.0 - smoothstep(0.35, properties.maxRoughness, roughness);

                        let amount = shine * gloss * edgeFade * distanceFade * confidence;
                        // A metal colours what it reflects with its own colour.
                        let tint = mix(vec3f(1.0), textureSample(sceneDiffuse_tex, sceneDiffuse_samp, uv).rgb, metallic);
                        let color = textureSample(sceneColor_tex, sceneColor_samp, hitUv).rgb * tint;
                        return vec4f(color, amount);
                    }
                "#,
        )
    ]
)
