//! Timing the frame on the GPU: timestamps written between the frame's sections, read back a
//! frame or two later without waiting for them, and averaged per section.
//!
//! The renderer marks the end of each section ([`fyrox_graphics::server::GraphicsServer::gpu_mark`]);
//! a section's time is from the mark before it to its own. Marks are written into the frame's
//! command encoder between render passes, which needs wgpu's
//! `TIMESTAMP_QUERY_INSIDE_ENCODERS`; without it nothing is timed.

use std::{
    cell::{Cell, RefCell},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
};

/// The most marks a frame can have; later ones are left out.
const MAX_MARKS: u32 = 64;
/// How many frames' timestamps can be on their way back at once. A frame finding none free is
/// not timed.
const IN_FLIGHT: usize = 3;

/// A readback buffer, and the marks whose timestamps it is bringing back.
struct Slot {
    buffer: wgpu::Buffer,
    labels: Vec<&'static str>,
    /// Set by wgpu once the buffer can be read.
    mapped: Arc<AtomicBool>,
    busy: bool,
}

/// Times the frame's sections on the GPU: see the module's description.
pub struct GpuTimer {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    slots: RefCell<Vec<Slot>>,
    /// This frame's marks so far.
    labels: RefCell<Vec<&'static str>>,
    /// Nanoseconds a timestamp tick.
    period: f64,
    /// Each section's time added up over the frames timed since last asked, and how many.
    totals: RefCell<Vec<(&'static str, f64)>>,
    frames: Cell<u32>,
    /// Each section's time in the latest frame to come back, kept until the next does.
    latest: RefCell<Vec<(&'static str, f32)>>,
}

impl GpuTimer {
    /// A timer, if the device can write timestamps between passes.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        let needed = wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
        if !device.features().contains(needed) {
            return None;
        }
        let bytes = MAX_MARKS as u64 * 8;
        let slots = (0..IN_FLIGHT)
            .map(|_| Slot {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("GpuTimerReadback"),
                    size: bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                labels: Vec::new(),
                mapped: Arc::new(AtomicBool::new(false)),
                busy: false,
            })
            .collect();
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("GpuTimer"),
                ty: wgpu::QueryType::Timestamp,
                count: MAX_MARKS,
            }),
            resolve: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("GpuTimerResolve"),
                size: bytes,
                usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
            slots: RefCell::new(slots),
            labels: RefCell::new(Vec::new()),
            period: queue.get_timestamp_period() as f64,
            totals: RefCell::new(Vec::new()),
            frames: Cell::new(0),
            latest: RefCell::new(Vec::new()),
        })
    }

    /// Writes a timestamp for the mark `label` into `encoder`, between its passes.
    pub fn mark(&self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        let mut labels = self.labels.borrow_mut();
        if labels.len() < MAX_MARKS as usize {
            encoder.write_timestamp(&self.queries, labels.len() as u32);
            labels.push(label);
        }
    }

    /// Ends the frame: records into `encoder`, the frame's last, the copy of its timestamps to a
    /// free readback buffer, if one is free and there are at least two marks. Returns the buffer
    /// to start reading back once `encoder` is submitted ([`Self::read_back`]).
    pub fn end_frame(&self, encoder: &mut wgpu::CommandEncoder) -> Option<usize> {
        let labels = std::mem::take(&mut *self.labels.borrow_mut());
        if labels.len() < 2 {
            return None;
        }
        let mut slots = self.slots.borrow_mut();
        let free = slots.iter().position(|s| !s.busy)?;
        let count = labels.len() as u32;
        encoder.resolve_query_set(&self.queries, 0..count, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &slots[free].buffer, 0, count as u64 * 8);
        let slot = &mut slots[free];
        slot.labels = labels;
        slot.busy = true;
        slot.mapped.store(false, Ordering::Release);
        Some(free)
    }

    /// Starts reading back the timestamps copied to readback buffer `slot`, once the encoder
    /// that copies them has been submitted.
    pub fn read_back(&self, slot: usize) {
        let slots = self.slots.borrow();
        let slot = &slots[slot];
        let mapped = slot.mapped.clone();
        let size = slot.labels.len() as u64 * 8;
        slot.buffer.slice(..size).map_async(wgpu::MapMode::Read, move |result| {
            if result.is_ok() {
                mapped.store(true, Ordering::Release);
            }
        });
    }

    /// Takes in every frame's timestamps that have come back.
    pub fn collect(&self) {
        let mut slots = self.slots.borrow_mut();
        let mut totals = self.totals.borrow_mut();
        let mut latest = self.latest.borrow_mut();
        for slot in slots.iter_mut().filter(|s| s.busy && s.mapped.load(Ordering::Acquire)) {
            let size = slot.labels.len() as u64 * 8;
            let stamps: Vec<u64> = {
                let Ok(data) = slot.buffer.slice(..size).get_mapped_range() else {
                    continue;
                };
                bytemuck::cast_slice::<u8, u64>(&data).to_vec()
            };
            slot.buffer.unmap();
            slot.busy = false;
            latest.clear();
            for (k, label) in slot.labels.iter().enumerate().skip(1) {
                let ms = stamps[k].saturating_sub(stamps[k - 1]) as f64 * self.period / 1.0e6;
                latest.push((label, ms as f32));
                match totals.iter_mut().find(|(l, _)| l == label) {
                    Some(total) => total.1 += ms,
                    None => totals.push((label, ms)),
                }
            }
            self.frames.set(self.frames.get() + 1);
        }
    }

    /// The time of the section marked `label` in the latest frame whose timestamps have come
    /// back (a frame or two ago), without disturbing [`Self::timings`]' averages.
    pub fn latest(&self, label: &str) -> Option<f32> {
        self.latest.borrow().iter().find(|(l, _)| *l == label).map(|(_, ms)| *ms)
    }

    /// Each section's time, averaged over the frames timed since last asked; see
    /// [`fyrox_graphics::server::GraphicsServer::gpu_timings`].
    pub fn timings(&self) -> Vec<(&'static str, f32)> {
        self.collect();
        let frames = self.frames.replace(0).max(1) as f64;
        let mut totals = self.totals.borrow_mut();
        totals.drain(..).map(|(label, ms)| (label, (ms / frames) as f32)).collect()
    }
}
