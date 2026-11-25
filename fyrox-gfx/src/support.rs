//! Whether the graphics card can run Yoctite and Candelite, found out once, before either is
//! started: what each asks of the card is checked by name where wgpu can say (a card that draws
//! on the CPU, compute shaders, ray queries), and the rest - every limit their pipelines press on,
//! and memory for Candelite's atlases - by making their pipelines and buffers once, on the card
//! itself, with wgpu's errors caught rather than left to end the program. A card that cannot run
//! Yoctite leaves every mesh to the renderer (and Candelite with it); one that cannot run
//! Candelite lights the scene with its flat ambient colour.

use candelite::{self, RadianceCache, Reflections, ScreenProbes, SurfaceCache, SurfaceLight};
use yoctite_wgpu::{Culler, GBufferDrawer, GeometryPool};
use std::{
    future::Future,
    panic::{self, AssertUnwindSafe},
    task::{Context, Poll, Waker},
};

/// What Yoctite asks of a card beyond what the renderer does: compute shaders to choose its
/// clusters, storage buffers read in vertex shaders to pull its vertices from the pool, and
/// draws whose counts the GPU writes for itself.
const YOCTITE_NEEDS: [(wgpu::DownlevelFlags, &str); 3] = [
    (wgpu::DownlevelFlags::COMPUTE_SHADERS, "run compute shaders"),
    (wgpu::DownlevelFlags::VERTEX_STORAGE, "read storage buffers in vertex shaders"),
    (wgpu::DownlevelFlags::INDIRECT_EXECUTION, "draw indirectly"),
];

/// How many texels square Candelite's surface atlas is, as the Yoctite pass makes it.
pub(crate) const ATLAS_SIZE: u32 = 2048;

/// Whether Yoctite and Candelite can run on a card: [`Ok`], or why not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Support {
    pub yoctite: Result<(), String>,
    /// Candelite needs Yoctite too: it lights Yoctite's meshes.
    pub candelite: Result<(), String>,
}

impl Support {
    /// Whether Yoctite and Candelite can run on `device`, made from `adapter`. Makes their
    /// pipelines and buffers once, to see that the card takes them, and drops them.
    pub fn of(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Self {
        let name = adapter.get_info().name;
        let yoctite = yoctite_support(adapter, device).map_err(|why| format!("{name} {why}"));
        let candelite = match &yoctite {
            Ok(()) => candelite_support(device).map_err(|why| format!("{name} {why}")),
            Err(_) => Err("it lights what Yoctite draws, and Yoctite is off".to_string()),
        };
        Self { yoctite, candelite }
    }
}

fn yoctite_support(adapter: &wgpu::Adapter, device: &wgpu::Device) -> Result<(), String> {
    // A software renderer (llvmpipe, SwiftShader, WARP) could run it, at a frame every few seconds.
    if adapter.get_info().device_type == wgpu::DeviceType::Cpu {
        return Err("draws on the CPU, too slowly for Yoctite".to_string());
    }
    let flags = adapter.get_downlevel_capabilities().flags;
    let missing: Vec<&str> =
        YOCTITE_NEEDS.iter().filter(|(flag, _)| !flags.contains(*flag)).map(|(_, what)| *what).collect();
    if !missing.is_empty() {
        return Err(format!("cannot {}", missing.join(", or ")));
    }
    trial(device, || {
        let _ = GeometryPool::new(device, 1 << 10);
        let _ = Culler::new(device);
        let _ = GBufferDrawer::new(device);
    })
    .map_err(|why| format!("cannot make Yoctite's pipelines: {why}"))
}

fn candelite_support(device: &wgpu::Device) -> Result<(), String> {
    if !candelite::can_trace(device) {
        return Err("cannot trace rays (wgpu's ray queries)".to_string());
    }
    let made = trial(device, || {
        let _ = SurfaceCache::new(device, ATLAS_SIZE);
        SurfaceLight::new(device).is_some()
            && RadianceCache::new(device).is_some()
            && ScreenProbes::new(device).is_some()
            && Reflections::new(device).is_some()
    })
    .map_err(|why| format!("cannot make Candelite's pipelines and atlases: {why}"))?;
    if made { Ok(()) } else { Err("cannot trace rays (wgpu's ray queries)".to_string()) }
}

/// Runs `make` with every error wgpu reports on `device` - an invalid pipeline, a limit passed,
/// memory run out - caught and handed back, instead of reaching wgpu's handler for errors no one
/// caught, which ends the program; and with a panic in `make` caught likewise.
fn trial<T>(device: &wgpu::Device, make: impl FnOnce() -> T) -> Result<T, String> {
    let scopes = [wgpu::ErrorFilter::Validation, wgpu::ErrorFilter::OutOfMemory, wgpu::ErrorFilter::Internal]
        .map(|filter| device.push_error_scope(filter));
    let made = panic::catch_unwind(AssertUnwindSafe(make));
    // Innermost first, as they were pushed.
    let errors: Vec<String> =
        scopes.into_iter().rev().filter_map(|scope| ready(scope.pop()).flatten()).map(|err| first_line(&err.to_string())).collect();
    match (made, errors.is_empty()) {
        (Ok(made), true) => Ok(made),
        (Ok(_), false) => Err(errors.join("; ")),
        (Err(panic), _) => {
            let what = panic
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| panic.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "a panic".to_string());
            Err(std::iter::once(first_line(&what)).chain(errors).collect::<Vec<_>>().join("; "))
        }
    }
}

/// What wgpu's future gives at once, as it does on native backends; [`None`] if it is still to
/// come (in a browser), and an error not seen then is not held against the card.
fn ready<F: Future>(future: F) -> Option<F::Output> {
    let mut future = std::pin::pin!(future);
    match future.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(out) => Some(out),
        Poll::Pending => None,
    }
}

/// An error on one line: wgpu's run over several, a heading ("Validation Error") and then the
/// chain of causes, which say what was wrong.
fn first_line(text: &str) -> String {
    let lines: Vec<&str> = text.lines().map(str::trim).filter(|line| !line.is_empty() && *line != "Caused by:").collect();
    let line = if lines.len() > 1 { lines[1..].join(": ") } else { lines.concat() };
    line.chars().take(300).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn adapter() -> Option<wgpu::Adapter> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        ready(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            ..Default::default()
        }))?
        .ok()
    }

    /// A device on `adapter` with `features` (those it has) and `limits`, as the renderer asks
    /// for them.
    fn device(adapter: &wgpu::Adapter, features: wgpu::Features, limits: wgpu::Limits) -> wgpu::Device {
        let features = features & adapter.features();
        let ray_tracing = features.contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY);
        ready(adapter.request_device(&wgpu::DeviceDescriptor {
            required_features: features,
            required_limits: limits,
            experimental_features: if ray_tracing {
                // SAFETY: only ray queries are used, and only when the adapter has them.
                unsafe { wgpu::ExperimentalFeatures::enabled() }
            } else {
                wgpu::ExperimentalFeatures::disabled()
            },
            ..Default::default()
        }))
        .expect("ready on native")
        .expect("a device")
        .0
    }

    /// What the renderer asks for (`fyrox_graphics_wgpu::server`), where the card has it.
    fn as_the_renderer_asks() -> wgpu::Features {
        wgpu::Features::EXPERIMENTAL_RAY_QUERY
            | yoctite_wgpu::SOFTWARE_FEATURES
            | wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
    }

    #[test]
    fn a_card_with_everything_runs_both() {
        let Some(adapter) = adapter() else { return };
        if !adapter.features().contains(wgpu::Features::EXPERIMENTAL_RAY_QUERY) {
            return;
        }
        let device = device(&adapter, as_the_renderer_asks(), adapter.limits());
        let support = Support::of(&adapter, &device);
        assert_eq!(support, Support { yoctite: Ok(()), candelite: Ok(()) });
    }

    #[test]
    fn without_ray_queries_candelite_is_off_and_yoctite_stays() {
        let Some(adapter) = adapter() else { return };
        let device = device(&adapter, as_the_renderer_asks() - wgpu::Features::EXPERIMENTAL_RAY_QUERY, adapter.limits());
        let support = Support::of(&adapter, &device);
        assert_eq!(support.yoctite, Ok(()));
        assert!(support.candelite.as_ref().is_err_and(|why| why.contains("ray")), "{support:?}");
    }

    #[test]
    fn a_card_with_too_few_storage_buffers_runs_neither_and_nothing_panics() {
        let Some(adapter) = adapter() else { return };
        // Four storage buffers a shader stage, as the least wgpu runs on has (its downlevel
        // limits), where Yoctite's G-buffer pass reads seven in its vertex shader.
        let limits = wgpu::Limits { max_storage_buffers_per_shader_stage: 4, ..adapter.limits() };
        let device = device(&adapter, as_the_renderer_asks(), limits);
        let support = Support::of(&adapter, &device);
        assert!(support.yoctite.as_ref().is_err_and(|why| why.contains("pipelines") && why.contains("storage")), "{support:?}");
        assert!(support.candelite.is_err());
    }

    #[test]
    fn a_card_at_webgpus_least_runs_yoctite_but_not_candelite() {
        let Some(adapter) = adapter() else { return };
        // WebGPU's limits: eight storage buffers a stage are enough for Yoctite; Candelite's
        // passes ask for more.
        let device = device(&adapter, as_the_renderer_asks(), wgpu::Limits::default());
        let support = Support::of(&adapter, &device);
        assert_eq!(support.yoctite, Ok(()));
        assert!(support.candelite.as_ref().is_err_and(|why| why.contains("pipelines")), "{support:?}");
    }

    #[test]
    fn a_trial_hands_back_what_it_made_or_why_not() {
        let Some(adapter) = adapter() else { return };
        let device = device(&adapter, wgpu::Features::empty(), adapter.limits());
        assert_eq!(trial(&device, || 7), Ok(7));
        let too_big = trial(&device, || {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: device.limits().max_buffer_size + 4,
                usage: wgpu::BufferUsages::STORAGE,
                mapped_at_creation: false,
            })
        });
        assert!(too_big.is_err());
        assert!(trial(&device, || panic!("made nothing")).is_err_and(|why| why.contains("made nothing")));
    }
}
