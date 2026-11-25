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
            kind: Texture(kind: Sampler2D, fallback: White),
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
                    layout(location = 0) in vec3 vertexPosition;
                    layout(location = 1) in vec2 vertexTexCoord;

                    out vec2 texCoord;

                    void main()
                    {
                        texCoord = vertexTexCoord;
                        gl_Position = properties.worldViewProjection * vec4(vertexPosition, 1.0);
                    }
                "#,

            fragment_shader:
                r#"
                    out vec4 FragColor;

                    in vec2 texCoord;

                    // Where a world-space point lands on screen: texture coordinates, and the
                    // depth the depth buffer would hold for it.
                    vec3 toScreen(vec3 point) {
                        vec4 clip = properties.viewProjection * vec4(point, 1.0);
                        vec3 ndc = clip.xyz / clip.w;
                        return vec3(ndc.x * 0.5 + 0.5, ndc.y * 0.5 + 0.5, ndc.z * 0.5 + 0.5);
                    }

                    vec3 worldAt(vec2 uv, float depth) {
                        return S_UnProject(vec3(uv, depth), properties.inverseViewProjection);
                    }

                    bool onScreen(vec3 screen) {
                        return screen.x >= 0.0 && screen.x <= 1.0 && screen.y >= 0.0 && screen.y <= 1.0 && screen.z <= 1.0;
                    }

                    // Noise from 0 to 1 that changes from pixel to pixel with no pattern the eye
                    // picks out, and from frame to frame, for temporal anti-aliasing to average.
                    float noise(vec2 pixel, float frame) {
                        vec2 p = pixel + 5.588238 * frame;
                        return fract(52.9829189 * fract(dot(p, vec2(0.06711056, 0.00583715))));
                    }

                    // Whether the ray, `travelled` along from `start`, has gone behind what is on
                    // screen there.
                    bool behind(vec3 start, vec3 ray, float travelled) {
                        vec3 screen = toScreen(start + ray * travelled);
                        return texture(sceneDepth, screen.xy).r < screen.z;
                    }

                    void main()
                    {
                        vec2 uv = gl_FragCoord.xy / properties.screenSize;
                        float depth = texture(sceneDepth, uv).r;
                        // Nothing was drawn here, so there is nothing to reflect in.
                        if (depth >= 1.0) {
                            FragColor = vec4(0.0);
                            return;
                        }

                        vec3 normal = normalize(texture(sceneNormal, uv).xyz * 2.0 - 1.0);
                        // Reflections are kept to floors and other surfaces facing upwards: a
                        // screen-space ray has nothing to go on where the reflected view leaves
                        // the screen, which is most of the time on walls facing the camera.
                        if (normal.y < properties.minUpwards) {
                            FragColor = vec4(0.0);
                            return;
                        }
                        vec4 material = texture(sceneMaterial, uv);
                        float metallic = clamp(material.x, 0.0, 1.0);
                        float roughness = clamp(material.y, 0.0, 1.0);
                        if (roughness >= properties.maxRoughness) {
                            FragColor = vec4(0.0);
                            return;
                        }

                        vec3 position = worldAt(uv, depth);
                        vec3 toCamera = normalize(properties.cameraPosition - position);
                        vec3 mirror = reflect(-toCamera, normal);
                        // A rough surface scatters what it reflects: each pixel sends its ray off a
                        // little to the side of the mirror's, further the rougher it is, a
                        // different way every frame, and anti-aliasing averages them into a blur.
                        float spin = 6.2831853 * noise(gl_FragCoord.xy, properties.frame);
                        float spread = roughness * roughness * sqrt(noise(gl_FragCoord.yx + 17.0, properties.frame));
                        vec3 side = normalize(cross(mirror, abs(mirror.y) > 0.9 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0)));
                        vec3 up = cross(side, mirror);
                        vec3 ray = normalize(mirror + spread * (cos(spin) * side + sin(spin) * up));
                        if (dot(ray, normal) <= 0.0) {
                            ray = mirror;
                        }
                        if (dot(ray, normal) <= 0.0) {
                            FragColor = vec4(0.0);
                            return;
                        }

                        int stepCount = max(properties.steps, 1);
                        float stepLength = properties.reach / float(stepCount);
                        // Each pixel starts at its own point within the first step: steps of the
                        // same length everywhere would show the steps as bands across the floor.
                        // Never at the surface itself, which would only reflect itself.
                        float travelled = stepLength * (0.25 + noise(gl_FragCoord.xy + 31.0, properties.frame));
                        float before = 0.0;
                        vec2 hitUv = vec2(0.0);
                        float confidence = 0.0;

                        for (int i = 0; i < stepCount; ++i) {
                            vec3 screen = toScreen(position + ray * travelled);
                            if (!onScreen(screen)) {
                                break;
                            }
                            if (behind(position, ray, travelled)) {
                                // Somewhere between the last step and this one the ray went behind
                                // something: halve the gap to find where, as near as can be.
                                float near = before;
                                float far = travelled;
                                for (int j = 0; j < 6; ++j) {
                                    float middle = 0.5 * (near + far);
                                    if (behind(position, ray, middle)) {
                                        far = middle;
                                    } else {
                                        near = middle;
                                    }
                                }
                                vec3 hitPoint = position + ray * far;
                                vec3 found = toScreen(hitPoint);
                                float sceneDepthValue = texture(sceneDepth, found.xy).r;
                                float gap = distance(worldAt(found.xy, sceneDepthValue), hitPoint);
                                // Only just behind it, it hit it: unless that is its back, which
                                // nothing reflected can be. Far behind, the ray passed behind
                                // something nearer the camera, and goes on to look past it.
                                vec3 facing = normalize(texture(sceneNormal, found.xy).xyz * 2.0 - 1.0);
                                float sure = 1.0 - smoothstep(0.5 * properties.thickness, properties.thickness, gap);
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
                            FragColor = vec4(0.0);
                            return;
                        }

                        // Fade out where the reflection runs off the screen, and as far as the
                        // ray reaches.
                        float edge = min(min(hitUv.x, 1.0 - hitUv.x), min(hitUv.y, 1.0 - hitUv.y));
                        float edgeFade = smoothstep(0.0, 0.15, edge);
                        float distanceFade = 1.0 - smoothstep(0.6, 1.0, travelled / properties.reach);
                        // A surface reflects more seen at a grazing angle than head on - a metal
                        // a lot at any angle - and less the rougher it is.
                        float grazing = pow(1.0 - clamp(dot(toCamera, normal), 0.0, 1.0), 2.0);
                        float shine = mix(properties.strength * mix(0.25, 1.0, grazing), properties.metalStrength, metallic);
                        float gloss = 1.0 - smoothstep(0.35, properties.maxRoughness, roughness);

                        float amount = shine * gloss * edgeFade * distanceFade * confidence;
                        // A metal colours what it reflects with its own colour.
                        vec3 tint = mix(vec3(1.0), texture(sceneDiffuse, uv).rgb, metallic);
                        vec3 color = texture(sceneColor, hitUv).rgb * tint;
                        FragColor = vec4(color, amount);
                    }
                "#,
        )
    ]
)
