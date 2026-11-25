//! A glass ball and a glass pane in front of a row of colored blocks.
//!
//! `cargo run --example glass` renders with wgpu (Vulkan);
//! `cargo run --example glass --features fyrox/backend_opengl` renders with OpenGL.

use fyrox::{
    core::{
        algebra::{Matrix4, UnitQuaternion, Vector3},
        color::Color,
        reflect::prelude::*,
        visitor::prelude::*,
    },
    engine::{executor::Executor, GraphicsContextParams},
    event_loop::EventLoop,
    material::{Material, MaterialResource},
    plugin::{error::GameResult, Plugin, PluginContext},
    scene::{
        base::BaseBuilder,
        camera::CameraBuilder,
        light::{directional::DirectionalLightBuilder, point::PointLightBuilder, BaseLightBuilder},
        mesh::{
            surface::{SurfaceBuilder, SurfaceData, SurfaceResource},
            MeshBuilder,
        },
        transform::TransformBuilder,
        Scene,
    },
};
use fyrox_gfx::{GlassMaterial, GraphicsEffects};

#[derive(Default, Debug, PartialEq, Visit, Reflect)]
#[reflect(non_cloneable, type_uuid = "9f2a0d3c-51e4-4b6a-8c77-0e1d2f3a4b5c")]
struct Demo {
    time: f32,
}

fn mesh(
    scene: &mut Scene,
    data: SurfaceData,
    material: MaterialResource,
    position: Vector3<f32>,
) -> fyrox::core::pool::Handle<fyrox::scene::node::Node> {
    MeshBuilder::new(
        BaseBuilder::new().with_local_transform(
            TransformBuilder::new()
                .with_local_position(position)
                .build(),
        ),
    )
    .with_surfaces(vec![SurfaceBuilder::new(SurfaceResource::new_embedded(
        data,
    ))
    .with_material(material)
    .build()])
    .build(&mut scene.graph)
    .to_base()
}

fn solid(color: Color) -> MaterialResource {
    let mut material = Material::standard();
    material.set_property("diffuseColor", color);
    MaterialResource::new_embedded(material)
}

impl Plugin for Demo {
    fn on_graphics_context_initialized(&mut self, ctx: PluginContext) -> GameResult {
        // With rays doing the shadows, the renderer's shadow maps would only double them up.
        if std::env::var("GLASS_DEMO_RT").as_deref() == Ok("1") {
            if let fyrox::engine::GraphicsContext::Initialized(graphics_context) =
                ctx.graphics_context
            {
                let renderer = &mut graphics_context.renderer;
                let mut settings = renderer.get_quality_settings();
                settings.csm_settings.enabled = false;
                fyrox::core::log::Log::verify(renderer.set_quality_settings(&settings));
            }
        }
        Ok(())
    }

    fn init(&mut self, _scene_path: Option<&str>, ctx: PluginContext) -> GameResult {
        let mut scene = Scene::new();

        CameraBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(0.0, 1.5, -4.0))
                    .with_local_rotation(UnitQuaternion::from_axis_angle(
                        &Vector3::x_axis(),
                        10f32.to_radians(),
                    ))
                    .build(),
            ),
        )
        .build(&mut scene.graph);

        DirectionalLightBuilder::new(BaseLightBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_rotation(UnitQuaternion::from_axis_angle(
                        &Vector3::x_axis(),
                        60f32.to_radians(),
                    ))
                    .build(),
            ),
        ))
        .build(&mut scene.graph);
        PointLightBuilder::new(BaseLightBuilder::new(
            BaseBuilder::new().with_local_transform(
                TransformBuilder::new()
                    .with_local_position(Vector3::new(-1.5, 2.5, -1.5))
                    .build(),
            ),
        ))
        .with_radius(8.0)
        .build(&mut scene.graph);

        // Floor and a row of stripes behind the glass, so bending is easy to see.
        mesh(
            &mut scene,
            SurfaceData::make_cube(Matrix4::new_nonuniform_scaling(&Vector3::new(
                12.0, 0.1, 12.0,
            ))),
            solid(Color::opaque(90, 90, 100)),
            Vector3::new(0.0, -0.05, 3.0),
        );
        let colors = [
            Color::opaque(230, 60, 60),
            Color::opaque(240, 220, 60),
            Color::opaque(60, 200, 90),
            Color::opaque(60, 120, 230),
        ];
        for i in 0..12 {
            mesh(
                &mut scene,
                SurfaceData::make_cube(Matrix4::new_nonuniform_scaling(&Vector3::new(
                    0.25, 2.5, 0.25,
                ))),
                solid(colors[i % colors.len()]),
                Vector3::new(-2.75 + i as f32 * 0.5, 1.25, 3.0),
            );
        }

        let glass = GlassMaterial {
            tint: Color::opaque(170, 90, 255),
            ..Default::default()
        };
        mesh(
            &mut scene,
            SurfaceData::make_sphere(32, 32, 0.8, &Matrix4::identity()),
            glass.build_resource(),
            Vector3::new(-0.9, 1.0, 0.5),
        );
        let wavy = GlassMaterial {
            tint: Color::opaque(150, 220, 255),
            waviness: 0.2,
            ..Default::default()
        };
        mesh(
            &mut scene,
            SurfaceData::make_cube(Matrix4::new_nonuniform_scaling(&Vector3::new(
                1.2, 1.6, 0.05,
            ))),
            wavy.build_resource(),
            Vector3::new(1.0, 1.0, 0.8),
        );

        ctx.scenes.add(scene);
        Ok(())
    }

    fn update(&mut self, ctx: &mut PluginContext) -> GameResult {
        self.time += ctx.dt;
        // Stop by itself when asked to, for automated runs.
        if let Some(limit) = std::env::var("GLASS_DEMO_SECONDS")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
        {
            if self.time > limit {
                ctx.loop_controller.exit();
            }
        }
        Ok(())
    }
}

fn main() {
    let mut executor = Executor::from_params(
        Some(EventLoop::new().unwrap()),
        GraphicsContextParams {
            window_attributes: Default::default(),
            vsync: true,
            msaa_sample_count: None,
            graphics_server_constructor: Default::default(),
            named_objects: false,
        },
    );
    // GLASS_DEMO_SOFT=0 keeps the engine's own shadow settings and GLASS_DEMO_AA=0 turns
    // anti-aliasing off, both for comparison shots.
    let mut effects = GraphicsEffects::default();
    if std::env::var("GLASS_DEMO_SOFT").as_deref() == Ok("0") {
        effects = effects.without_soft_shadows();
    }
    if std::env::var("GLASS_DEMO_AA").as_deref() == Ok("0") {
        effects = effects.with_anti_aliasing(fyrox_gfx::AntiAliasing::off());
    }
    // GLASS_DEMO_RT=1 traces the sun's shadows against the geometry instead of shadow mapping.
    #[cfg(feature = "raytracing")]
    if std::env::var("GLASS_DEMO_RT").as_deref() == Ok("1") {
        effects = effects.with_ray_traced_shadows(fyrox_gfx::RayTracedShadows {
            bias: std::env::var("GLASS_DEMO_RT_BIAS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0.05),
            ..Default::default()
        });
    }
    // GLASS_DEMO_REFLECTIONS=1 traces reflections on the floor.
    if std::env::var("GLASS_DEMO_REFLECTIONS").as_deref() == Ok("1") {
        effects = effects.with_reflections(fyrox_gfx::Reflections {
            strength: 0.6,
            ..Default::default()
        });
    }
    executor.add_plugin(effects);
    executor.add_plugin(Demo::default());
    executor.run()
}
