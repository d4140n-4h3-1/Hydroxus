// Copyright (c) 2019-present Dmitry Stepanov and Fyrox Engine contributors.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//! Sound engine module
//!
//! ## Overview
//!
//! Sound engine manages contexts, feeds output device with data.

use crate::context::SoundContext;
use crate::renderer::Renderer;
use fyrox_core::visitor::prelude::*;
use fyrox_core::SafeLock;
use std::error::Error;
use std::sync::{Arc, Mutex, MutexGuard};

/// Sound engine manages contexts, feeds output device with data. Sound engine instance can be cloned,
/// however this is always a "shallow" clone, because actual sound engine data is wrapped in Arc.
#[derive(Clone)]
pub struct SoundEngine(Arc<Mutex<State>>);

impl Default for SoundEngine {
    fn default() -> Self {
        Self::without_device(Self::DEFAULT_SAMPLE_RATE)
    }
}

/// Internal state of the sound engine.
pub struct State {
    sample_rate: u32,
    contexts: Vec<SoundContext>,
    /// A whole HRTF block rendered ahead, and how much of it has been handed out already. The HRTF
    /// renderer works only in blocks of [`State::render_buffer_len`], which is longer than the
    /// output device's own block.
    hrtf_block: Vec<(f32, f32)>,
    hrtf_block_read: usize,
    #[cfg(feature = "output")]
    output_device: Option<tinyaudio::OutputDevice>,
}

impl SoundEngine {
    /// Default sample rate of the sound engine.
    pub const DEFAULT_SAMPLE_RATE: u32 = 44100;

    /// How many samples per channel the output device is fed at a time. The device holds two of
    /// these, so this sets how late every sound is heard: 512 is about 23 ms at 44.1 kHz, where the
    /// HRTF block of 2052 is about 93 ms. In a browser the samples are handed over on the page's
    /// own thread, between frames, so there the longer block stays to keep the sound from breaking.
    #[cfg(not(target_arch = "wasm32"))]
    pub const OUTPUT_BLOCK_LEN: usize = 512;
    /// How many samples per channel the output device is fed at a time.
    #[cfg(target_arch = "wasm32")]
    pub const OUTPUT_BLOCK_LEN: usize = SoundContext::SAMPLES_PER_CHANNEL;

    /// Creates new instance of the sound engine. It is possible to have multiple engines running at
    /// the same time, but you shouldn't do this because you can create multiple contexts which
    /// should cover 99% of use cases.
    pub fn new(sample_rate: u32) -> Result<Self, Box<dyn Error>> {
        let engine = Self::without_device(sample_rate);
        engine.initialize_audio_output_device()?;
        Ok(engine)
    }

    /// Creates new instance of a sound engine without OS audio output device (so called headless mode).
    /// The user should periodically run [`State::render`] if they want to implement their own sample sending
    /// method to an output device (or a file, etc.).
    pub fn without_device(sample_rate: u32) -> Self {
        Self(Arc::new(Mutex::new(State {
            sample_rate,
            contexts: Default::default(),
            hrtf_block: Default::default(),
            hrtf_block_read: 0,
            #[cfg(feature = "output")]
            output_device: None,
        })))
    }

    /// Sets the sample rate for the sound engine and recreates the output device.
    pub fn set_sample_rate(&mut self, sample_rate: u32) -> Result<(), Box<dyn Error>> {
        self.state().sample_rate = sample_rate;
        self.initialize_audio_output_device()
    }

    /// Returns current sample rate of the sound engine.
    pub fn sample_rate(&self) -> u32 {
        self.state().sample_rate
    }

    /// Normalizes given frequency using context's sampling rate. Normalized frequency then can be used
    /// to create filters.
    pub fn normalize_frequency(&self, f: f32) -> f32 {
        f / self.sample_rate() as f32
    }

    /// Tries to initialize default audio output device.
    pub fn initialize_audio_output_device(&self) -> Result<(), Box<dyn Error>> {
        #[cfg(feature = "output")]
        {
            let sample_rate = self.sample_rate() as usize;
            let this = self.clone();

            let device = tinyaudio::run_output_device(
                tinyaudio::OutputDeviceParameters {
                    sample_rate,
                    channels_count: 2,
                    channel_sample_count: Self::OUTPUT_BLOCK_LEN,
                },
                {
                    move |buf| {
                        // SAFETY: This is safe as long as channels count above is 2.
                        let data = unsafe {
                            std::slice::from_raw_parts_mut(
                                buf.as_mut_ptr() as *mut (f32, f32),
                                buf.len() / 2,
                            )
                        };

                        this.state().render(data);
                    }
                },
            )?;

            self.state().output_device = Some(device);
        }

        Ok(())
    }

    /// Destroys current audio output device (if any).
    pub fn destroy_audio_output_device(&self) {
        #[cfg(feature = "output")]
        {
            self.state().output_device = None;
        }
    }

    /// Provides direct access to actual engine data.
    pub fn state(&self) -> MutexGuard<State> {
        self.0.safe_lock().unwrap()
    }
}

impl State {
    /// Adds new context to the engine. Each context must be added to the engine to emit
    /// sounds.
    pub fn add_context(&mut self, context: SoundContext) {
        self.contexts.push(context);
    }

    /// Removes a context from the engine. Removed context will no longer produce any sound.
    pub fn remove_context(&mut self, context: SoundContext) {
        if let Some(position) = self.contexts.iter().position(|c| c == &context) {
            self.contexts.remove(position);
        }
    }

    /// Removes all contexts from the engine.
    pub fn remove_all_contexts(&mut self) {
        self.contexts.clear()
    }

    /// Checks if a context is registered in the engine.
    pub fn has_context(&self, context: &SoundContext) -> bool {
        self.contexts
            .iter()
            .any(|c| Arc::ptr_eq(c.state.as_ref().unwrap(), context.state.as_ref().unwrap()))
    }

    /// Returns a reference to context container.
    pub fn contexts(&self) -> &[SoundContext] {
        &self.contexts
    }

    /// Returns the length of buf to be passed to [`Self::render()`].
    pub fn render_buffer_len() -> usize {
        SoundContext::SAMPLES_PER_CHANNEL
    }

    /// Renders the sound into buf. The buf must have at least [`Self::render_buffer_len()`]
    /// elements. This method must be used if and only if the engine was created via
    /// [`SoundEngine::without_device`].
    ///
    /// ## Deadlocks
    ///
    /// This method internally locks added sound contexts so it must be called when all the contexts
    /// are unlocked or you'll get a deadlock.
    pub fn render(&mut self, buf: &mut [(f32, f32)]) {
        // Without HRTF any length renders as it is asked for, and nothing waits a block ahead.
        if self.hrtf_block_read == self.hrtf_block.len() && !self.uses_hrtf() {
            buf.fill((0.0, 0.0));
            self.render_inner(buf);
            return;
        }

        let mut written = 0;
        while written < buf.len() {
            if self.hrtf_block_read == self.hrtf_block.len() {
                let mut block = std::mem::take(&mut self.hrtf_block);
                block.clear();
                block.resize(Self::render_buffer_len(), (0.0, 0.0));
                self.render_inner(&mut block);
                self.hrtf_block = block;
                self.hrtf_block_read = 0;
            }
            let count = (buf.len() - written).min(self.hrtf_block.len() - self.hrtf_block_read);
            buf[written..written + count].copy_from_slice(
                &self.hrtf_block[self.hrtf_block_read..self.hrtf_block_read + count],
            );
            written += count;
            self.hrtf_block_read += count;
        }
    }

    fn uses_hrtf(&self) -> bool {
        self.contexts
            .iter()
            .any(|context| matches!(context.state().renderer(), Renderer::HrtfRenderer(_)))
    }

    fn render_inner(&mut self, buf: &mut [(f32, f32)]) {
        for context in self.contexts.iter_mut() {
            context.state().render(self.sample_rate, buf);
        }
    }
}

impl Visit for State {
    fn visit(&mut self, name: &str, visitor: &mut Visitor) -> VisitResult {
        if visitor.is_reading() {
            self.contexts.clear();
        }

        let mut region = visitor.enter_region(name)?;

        self.contexts.visit("Contexts", &mut region)?;

        Ok(())
    }
}
