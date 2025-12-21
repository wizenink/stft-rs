/*MIT License

Copyright (c) 2025 David Maseda Neira

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(feature = "std"))]
extern crate alloc;

#[cfg(not(feature = "std"))]
use alloc::{collections::VecDeque, sync::Arc, vec, vec::Vec};

#[cfg(feature = "std")]
use std::{collections::VecDeque, sync::Arc, vec};

use core::fmt;
use core::marker::PhantomData;
use num_traits::{Float, FromPrimitive};

pub mod fft_backend;
use fft_backend::{Complex, FftBackend, FftNum, FftPlanner, FftPlannerTrait};

mod utils;
pub use utils::{apply_padding, deinterleave, deinterleave_into, interleave, interleave_into};

pub mod mel;

pub mod prelude {
    pub use crate::fft_backend::Complex;
    pub use crate::mel::{
        BatchMelSpectrogram, BatchMelSpectrogramF32, BatchMelSpectrogramF64, MelConfig,
        MelConfigF32, MelConfigF64, MelFilterbank, MelFilterbankF32, MelFilterbankF64, MelNorm,
        MelScale, MelSpectrum, MelSpectrumF32, MelSpectrumF64, StreamingMelSpectrogram,
        StreamingMelSpectrogramF32, StreamingMelSpectrogramF64,
    };
    pub use crate::utils::{
        apply_padding, deinterleave, deinterleave_into, interleave, interleave_into,
    };
    pub use crate::{
        BatchIstft, BatchIstftF32, BatchIstftF64, BatchStft, BatchStftF32, BatchStftF64,
        MultiChannelStreamingIstft, MultiChannelStreamingIstftF32, MultiChannelStreamingIstftF64,
        MultiChannelStreamingStft, MultiChannelStreamingStftF32, MultiChannelStreamingStftF64,
        PadMode, ReconstructionMode, Spectrum, SpectrumF32, SpectrumF64, SpectrumFrame,
        SpectrumFrameF32, SpectrumFrameF64, StftConfig, StftConfigBuilder, StftConfigBuilderF32,
        StftConfigBuilderF64, StftConfigF32, StftConfigF64, StreamingIstft, StreamingIstftF32,
        StreamingIstftF64, StreamingStft, StreamingStftF32, StreamingStftF64, WindowType,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReconstructionMode {
    /// Overlap-Add: normalize by sum(w), requires COLA condition
    Ola,

    /// Weighted Overlap-Add: normalize by sum(w^2), requires NOLA condition
    Wola,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WindowType {
    Hann,
    Hamming,
    Blackman,
}

#[derive(Debug, Clone)]
pub enum ConfigError<T: Float + fmt::Debug> {
    NolaViolation { min_energy: T, threshold: T },
    ColaViolation { max_deviation: T, threshold: T },
    InvalidHopSize,
    InvalidFftSize,
}

impl<T: Float + fmt::Display + fmt::Debug> fmt::Display for ConfigError<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ConfigError::NolaViolation {
                min_energy,
                threshold,
            } => {
                write!(
                    f,
                    "NOLA condition violated: min_energy={} < threshold={}",
                    min_energy, threshold
                )
            }
            ConfigError::ColaViolation {
                max_deviation,
                threshold,
            } => {
                write!(
                    f,
                    "COLA condition violated: max_deviation={} > threshold={}",
                    max_deviation, threshold
                )
            }
            ConfigError::InvalidHopSize => write!(f, "Invalid hop size"),
            ConfigError::InvalidFftSize => write!(f, "Invalid FFT size"),
        }
    }
}

#[cfg(feature = "std")]
impl<T: Float + fmt::Display + fmt::Debug> std::error::Error for ConfigError<T> {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PadMode {
    Reflect,
    Zero,
    Edge,
}

#[derive(Debug, Clone, PartialEq)]
pub struct StftConfig<T: Float> {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: WindowType,
    pub reconstruction_mode: ReconstructionMode,
    _phantom: PhantomData<T>,
}

impl<T: Float + FromPrimitive + fmt::Debug> StftConfig<T> {
    fn nola_threshold() -> T {
        T::from(1e-8).unwrap()
    }

    fn cola_relative_tolerance() -> T {
        T::from(1e-4).unwrap()
    }

    #[deprecated(
        since = "0.4.0",
        note = "Use `StftConfig::builder()` instead for a more flexible API"
    )]
    pub fn new(
        fft_size: usize,
        hop_size: usize,
        window: WindowType,
        reconstruction_mode: ReconstructionMode,
    ) -> Result<Self, ConfigError<T>> {
        if fft_size == 0 || !(cfg!(feature = "rustfft-backend") || fft_size.is_power_of_two()) {
            return Err(ConfigError::InvalidFftSize);
        }
        if hop_size == 0 || hop_size > fft_size {
            return Err(ConfigError::InvalidHopSize);
        }

        let config = Self {
            fft_size,
            hop_size,
            window,
            reconstruction_mode,
            _phantom: PhantomData,
        };

        // Validate appropriate condition based on reconstruction mode
        match reconstruction_mode {
            ReconstructionMode::Ola => config.validate_cola()?,
            ReconstructionMode::Wola => config.validate_nola()?,
        }

        Ok(config)
    }

    /// Create a new builder for StftConfig
    pub fn builder() -> StftConfigBuilder<T> {
        StftConfigBuilder::new()
    }

    /// Default: 4096 FFT, 1024 hop, Hann window, OLA mode
    #[allow(deprecated)]
    pub fn default_4096() -> Self {
        Self::new(4096, 1024, WindowType::Hann, ReconstructionMode::Ola)
            .expect("Default config should always be valid")
    }

    pub fn freq_bins(&self) -> usize {
        self.fft_size / 2 + 1
    }

    pub fn overlap_percent(&self) -> T {
        let one = T::one();
        let hundred = T::from(100.0).unwrap();
        (one - T::from(self.hop_size).unwrap() / T::from(self.fft_size).unwrap()) * hundred
    }

    fn generate_window(&self) -> Vec<T> {
        generate_window(self.window, self.fft_size)
    }

    /// Validate NOLA condition: sum(w^2) > threshold everywhere
    pub fn validate_nola(&self) -> Result<(), ConfigError<T>> {
        let window = self.generate_window();
        let num_overlaps = self.fft_size.div_ceil(self.hop_size);
        let test_len = self.fft_size + (num_overlaps - 1) * self.hop_size;
        let mut energy = vec![T::zero(); test_len];

        for i in 0..num_overlaps {
            let offset = i * self.hop_size;
            for j in 0..self.fft_size {
                if offset + j < test_len {
                    energy[offset + j] = energy[offset + j] + window[j] * window[j];
                }
            }
        }

        // Check the steady-state region (skip edges)
        let start = self.fft_size / 2;
        let end = test_len - self.fft_size / 2;
        let min_energy = energy[start..end]
            .iter()
            .copied()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or_else(T::zero);

        if min_energy < Self::nola_threshold() {
            return Err(ConfigError::NolaViolation {
                min_energy,
                threshold: Self::nola_threshold(),
            });
        }

        Ok(())
    }

    /// Validate weak COLA condition: sum(w) is constant (within relative tolerance)
    pub fn validate_cola(&self) -> Result<(), ConfigError<T>> {
        let window = self.generate_window();
        let window_len = window.len();

        let mut cola_sum_period = vec![T::zero(); self.hop_size];
        (0..window_len).for_each(|i| {
            let idx = i % self.hop_size;
            cola_sum_period[idx] = cola_sum_period[idx] + window[i];
        });

        let zero = T::zero();
        let min_sum = cola_sum_period
            .iter()
            .min_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&zero);
        let max_sum = cola_sum_period
            .iter()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(&zero);

        let epsilon = T::from(1e-9).unwrap();
        if *max_sum < epsilon {
            return Err(ConfigError::ColaViolation {
                max_deviation: T::infinity(),
                threshold: Self::cola_relative_tolerance(),
            });
        }

        let ripple = (*max_sum - *min_sum) / *max_sum;

        let is_compliant = ripple < Self::cola_relative_tolerance();

        if !is_compliant {
            return Err(ConfigError::ColaViolation {
                max_deviation: ripple,
                threshold: Self::cola_relative_tolerance(),
            });
        }
        Ok(())
    }
}

/// Builder for StftConfig with fluent API
#[derive(Debug, Clone, PartialEq)]
pub struct StftConfigBuilder<T: Float> {
    fft_size: Option<usize>,
    hop_size: Option<usize>,
    window: WindowType,
    reconstruction_mode: ReconstructionMode,
    _phantom: PhantomData<T>,
}

impl<T: Float + FromPrimitive + fmt::Debug> StftConfigBuilder<T> {
    /// Create a new builder with default values (Hann window, OLA mode)
    pub fn new() -> Self {
        Self {
            fft_size: None,
            hop_size: None,
            window: WindowType::Hann,
            reconstruction_mode: ReconstructionMode::Ola,
            _phantom: PhantomData,
        }
    }

    /// Set the FFT size (must be a power of two)
    pub fn fft_size(mut self, fft_size: usize) -> Self {
        self.fft_size = Some(fft_size);
        self
    }

    /// Set the hop size (must be > 0 and <= fft_size)
    pub fn hop_size(mut self, hop_size: usize) -> Self {
        self.hop_size = Some(hop_size);
        self
    }

    /// Set the window type (default: Hann)
    pub fn window(mut self, window: WindowType) -> Self {
        self.window = window;
        self
    }

    /// Set the reconstruction mode (default: OLA)
    pub fn reconstruction_mode(mut self, mode: ReconstructionMode) -> Self {
        self.reconstruction_mode = mode;
        self
    }

    /// Build the StftConfig, validating all parameters
    ///
    /// Returns an error if:
    /// - fft_size is not set or not a power of two
    /// - hop_size is not set, zero, or greater than fft_size
    /// - COLA/NOLA conditions are violated
    #[allow(deprecated)]
    pub fn build(self) -> Result<StftConfig<T>, ConfigError<T>> {
        let fft_size = self.fft_size.ok_or(ConfigError::InvalidFftSize)?;
        let hop_size = self.hop_size.ok_or(ConfigError::InvalidHopSize)?;

        StftConfig::new(fft_size, hop_size, self.window, self.reconstruction_mode)
    }
}

impl<T: Float + FromPrimitive + fmt::Debug> Default for StftConfigBuilder<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn generate_window<T: Float + FromPrimitive>(window_type: WindowType, size: usize) -> Vec<T> {
    let pi = T::from(core::f64::consts::PI).unwrap();
    let two = T::from(2.0).unwrap();

    match window_type {
        WindowType::Hann => (0..size)
            .map(|i| {
                let half = T::from(0.5).unwrap();
                let one = T::one();
                let i_t = T::from(i).unwrap();
                let size_t = T::from(size).unwrap(); // Use N, not N-1 for periodic window
                half * (one - (two * pi * i_t / size_t).cos())
            })
            .collect(),
        WindowType::Hamming => (0..size)
            .map(|i| {
                let i_t = T::from(i).unwrap();
                let size_t = T::from(size).unwrap(); // Use N, not N-1 for periodic window
                T::from(0.54).unwrap() - T::from(0.46).unwrap() * (two * pi * i_t / size_t).cos()
            })
            .collect(),
        WindowType::Blackman => (0..size)
            .map(|i| {
                let i_t = T::from(i).unwrap();
                let size_t = T::from(size).unwrap(); // Use N, not N-1 for periodic window
                let angle = two * pi * i_t / size_t;
                T::from(0.42).unwrap() - T::from(0.5).unwrap() * angle.cos()
                    + T::from(0.08).unwrap() * (two * angle).cos()
            })
            .collect(),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SpectrumFrame<T: Float> {
    pub freq_bins: usize,
    pub data: Vec<Complex<T>>,
}

impl<T: Float> SpectrumFrame<T> {
    pub fn new(freq_bins: usize) -> Self {
        Self {
            freq_bins,
            data: vec![Complex::new(T::zero(), T::zero()); freq_bins],
        }
    }

    pub fn from_data(data: Vec<Complex<T>>) -> Self {
        let freq_bins = data.len();
        Self { freq_bins, data }
    }

    /// Prepare frame for reuse by clearing data (keeps capacity)
    pub fn clear(&mut self) {
        for val in &mut self.data {
            *val = Complex::new(T::zero(), T::zero());
        }
    }

    /// Resize frame if needed to match freq_bins
    pub fn resize_if_needed(&mut self, freq_bins: usize) {
        if self.freq_bins != freq_bins {
            self.freq_bins = freq_bins;
            self.data
                .resize(freq_bins, Complex::new(T::zero(), T::zero()));
        }
    }

    /// Write data from a slice into this frame
    pub fn write_from_slice(&mut self, data: &[Complex<T>]) {
        self.resize_if_needed(data.len());
        self.data.copy_from_slice(data);
    }

    /// Get the magnitude of a frequency bin
    #[inline]
    pub fn magnitude(&self, bin: usize) -> T {
        let c = &self.data[bin];
        (c.re * c.re + c.im * c.im).sqrt()
    }

    /// Get the phase of a frequency bin in radians
    #[inline]
    pub fn phase(&self, bin: usize) -> T {
        let c = &self.data[bin];
        c.im.atan2(c.re)
    }

    /// Set a frequency bin from magnitude and phase
    pub fn set_magnitude_phase(&mut self, bin: usize, magnitude: T, phase: T) {
        self.data[bin] = Complex::new(magnitude * phase.cos(), magnitude * phase.sin());
    }

    /// Create a SpectrumFrame from magnitude and phase arrays
    pub fn from_magnitude_phase(magnitudes: &[T], phases: &[T]) -> Self {
        assert_eq!(
            magnitudes.len(),
            phases.len(),
            "Magnitude and phase arrays must have same length"
        );
        let freq_bins = magnitudes.len();
        let data: Vec<Complex<T>> = magnitudes
            .iter()
            .zip(phases.iter())
            .map(|(mag, phase)| Complex::new(*mag * phase.cos(), *mag * phase.sin()))
            .collect();
        Self { freq_bins, data }
    }

    /// Get all magnitudes as a Vec
    pub fn magnitudes(&self) -> Vec<T> {
        self.data
            .iter()
            .map(|c| (c.re * c.re + c.im * c.im).sqrt())
            .collect()
    }

    /// Get all phases as a Vec
    pub fn phases(&self) -> Vec<T> {
        self.data.iter().map(|c| c.im.atan2(c.re)).collect()
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Spectrum<T: Float> {
    pub num_frames: usize,
    pub freq_bins: usize,
    pub data: Vec<T>,
}

impl<T: Float> Spectrum<T> {
    pub fn new(num_frames: usize, freq_bins: usize) -> Self {
        Self {
            num_frames,
            freq_bins,
            data: vec![T::zero(); 2 * num_frames * freq_bins],
        }
    }

    #[inline]
    pub fn real(&self, frame: usize, bin: usize) -> T {
        self.data[frame * self.freq_bins + bin]
    }

    #[inline]
    pub fn imag(&self, frame: usize, bin: usize) -> T {
        let offset = self.num_frames * self.freq_bins;
        self.data[offset + frame * self.freq_bins + bin]
    }

    #[inline]
    pub fn get_complex(&self, frame: usize, bin: usize) -> Complex<T> {
        Complex::new(self.real(frame, bin), self.imag(frame, bin))
    }

    pub fn frames(&self) -> impl Iterator<Item = SpectrumFrame<T>> + '_ {
        (0..self.num_frames).map(move |frame_idx| {
            let data: Vec<Complex<T>> = (0..self.freq_bins)
                .map(|bin| self.get_complex(frame_idx, bin))
                .collect();
            SpectrumFrame::from_data(data)
        })
    }

    /// Set the real part of a bin
    #[inline]
    pub fn set_real(&mut self, frame: usize, bin: usize, value: T) {
        self.data[frame * self.freq_bins + bin] = value;
    }

    /// Set the imaginary part of a bin
    #[inline]
    pub fn set_imag(&mut self, frame: usize, bin: usize, value: T) {
        let offset = self.num_frames * self.freq_bins;
        self.data[offset + frame * self.freq_bins + bin] = value;
    }

    /// Set a bin from a complex value
    #[inline]
    pub fn set_complex(&mut self, frame: usize, bin: usize, value: Complex<T>) {
        self.set_real(frame, bin, value.re);
        self.set_imag(frame, bin, value.im);
    }

    /// Get the magnitude of a frequency bin
    #[inline]
    pub fn magnitude(&self, frame: usize, bin: usize) -> T {
        let re = self.real(frame, bin);
        let im = self.imag(frame, bin);
        (re * re + im * im).sqrt()
    }

    /// Get the phase of a frequency bin in radians
    #[inline]
    pub fn phase(&self, frame: usize, bin: usize) -> T {
        let re = self.real(frame, bin);
        let im = self.imag(frame, bin);
        im.atan2(re)
    }

    /// Set a frequency bin from magnitude and phase
    pub fn set_magnitude_phase(&mut self, frame: usize, bin: usize, magnitude: T, phase: T) {
        self.set_real(frame, bin, magnitude * phase.cos());
        self.set_imag(frame, bin, magnitude * phase.sin());
    }

    /// Get all magnitudes for a frame
    pub fn frame_magnitudes(&self, frame: usize) -> Vec<T> {
        (0..self.freq_bins)
            .map(|bin| self.magnitude(frame, bin))
            .collect()
    }

    /// Get all phases for a frame
    pub fn frame_phases(&self, frame: usize) -> Vec<T> {
        (0..self.freq_bins)
            .map(|bin| self.phase(frame, bin))
            .collect()
    }

    /// Apply a function to all bins
    pub fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, usize, Complex<T>) -> Complex<T>,
    {
        for frame in 0..self.num_frames {
            for bin in 0..self.freq_bins {
                let c = self.get_complex(frame, bin);
                let new_c = f(frame, bin, c);
                self.set_complex(frame, bin, new_c);
            }
        }
    }

    /// Apply a gain to a range of bins across all frames
    pub fn apply_gain(&mut self, bin_range: core::ops::Range<usize>, gain: T) {
        for frame in 0..self.num_frames {
            for bin in bin_range.clone() {
                if bin < self.freq_bins {
                    let c = self.get_complex(frame, bin);
                    self.set_complex(frame, bin, c * gain);
                }
            }
        }
    }

    /// Zero out a range of bins across all frames
    pub fn zero_bins(&mut self, bin_range: core::ops::Range<usize>) {
        for frame in 0..self.num_frames {
            for bin in bin_range.clone() {
                if bin < self.freq_bins {
                    self.set_complex(frame, bin, Complex::new(T::zero(), T::zero()));
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct BatchStft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    fft: Arc<dyn FftBackend<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> BatchStft<T> {
    pub fn new(config: StftConfig<T>) -> Self
    where
        FftPlanner<T>: FftPlannerTrait<T>,
    {
        let window = config.generate_window();
        let mut planner = <FftPlanner<T> as FftPlannerTrait<T>>::new();
        let fft = planner.plan_fft_forward(config.fft_size);

        Self {
            config,
            window,
            fft,
        }
    }

    pub fn process(&self, signal: &[T]) -> Spectrum<T> {
        self.process_padded(signal, PadMode::Reflect)
    }

    pub fn process_padded(&self, signal: &[T], pad_mode: PadMode) -> Spectrum<T> {
        let pad_amount = self.config.fft_size / 2;
        let padded = utils::apply_padding(signal, pad_amount, pad_mode);

        let num_frames = if padded.len() >= self.config.fft_size {
            (padded.len() - self.config.fft_size) / self.config.hop_size + 1
        } else {
            0
        };

        let freq_bins = self.config.freq_bins();
        let mut result = Spectrum::new(num_frames, freq_bins);

        let mut fft_buffer = vec![Complex::new(T::zero(), T::zero()); self.config.fft_size];

        for (frame_idx, frame_start) in (0..padded.len() - self.config.fft_size + 1)
            .step_by(self.config.hop_size)
            .enumerate()
        {
            // Apply window and prepare FFT input
            for i in 0..self.config.fft_size {
                fft_buffer[i] = Complex::new(padded[frame_start + i] * self.window[i], T::zero());
            }

            // Compute FFT
            self.fft.process(&mut fft_buffer);

            // Store positive frequencies in flat layout
            (0..freq_bins).for_each(|bin| {
                let idx = frame_idx * freq_bins + bin;
                result.data[idx] = fft_buffer[bin].re;
                result.data[num_frames * freq_bins + idx] = fft_buffer[bin].im;
            });
        }

        result
    }

    /// Process signal and write into a pre-allocated Spectrum.
    /// The spectrum must have the correct dimensions (num_frames x freq_bins).
    /// Returns true if successful, false if dimensions don't match.
    pub fn process_into(&self, signal: &[T], spectrum: &mut Spectrum<T>) -> bool {
        self.process_padded_into(signal, PadMode::Reflect, spectrum)
    }

    /// Process signal with padding and write into a pre-allocated Spectrum.
    pub fn process_padded_into(
        &self,
        signal: &[T],
        pad_mode: PadMode,
        spectrum: &mut Spectrum<T>,
    ) -> bool {
        let pad_amount = self.config.fft_size / 2;
        let padded = utils::apply_padding(signal, pad_amount, pad_mode);

        let num_frames = if padded.len() >= self.config.fft_size {
            (padded.len() - self.config.fft_size) / self.config.hop_size + 1
        } else {
            0
        };

        let freq_bins = self.config.freq_bins();

        // Check dimensions
        if spectrum.num_frames != num_frames || spectrum.freq_bins != freq_bins {
            return false;
        }

        let mut fft_buffer = vec![Complex::new(T::zero(), T::zero()); self.config.fft_size];

        for (frame_idx, frame_start) in (0..padded.len() - self.config.fft_size + 1)
            .step_by(self.config.hop_size)
            .enumerate()
        {
            // Apply window and prepare FFT input
            for i in 0..self.config.fft_size {
                fft_buffer[i] = Complex::new(padded[frame_start + i] * self.window[i], T::zero());
            }

            // Compute FFT
            self.fft.process(&mut fft_buffer);

            // Store positive frequencies in flat layout
            (0..freq_bins).for_each(|bin| {
                let idx = frame_idx * freq_bins + bin;
                spectrum.data[idx] = fft_buffer[bin].re;
                spectrum.data[num_frames * freq_bins + idx] = fft_buffer[bin].im;
            });
        }

        true
    }

    /// Process multiple channels independently.
    /// Returns one Spectrum per channel.
    ///
    /// # Arguments
    ///
    /// * `channels` - Slice of audio channels, each as a separate Vec
    ///
    /// # Panics
    ///
    /// Panics if channels is empty or if channels have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::prelude::*;
    ///
    /// let config = StftConfigF32::default_4096();
    /// let stft = BatchStftF32::new(config);
    ///
    /// let left = vec![0.0; 44100];
    /// let right = vec![0.0; 44100];
    /// let channels = vec![left, right];
    ///
    /// let spectra = stft.process_multichannel(&channels);
    /// assert_eq!(spectra.len(), 2); // One spectrum per channel
    /// ```
    pub fn process_multichannel(&self, channels: &[Vec<T>]) -> Vec<Spectrum<T>> {
        assert!(!channels.is_empty(), "channels must not be empty");

        // Validate all channels have same length
        let expected_len = channels[0].len();
        for (i, channel) in channels.iter().enumerate() {
            assert_eq!(
                channel.len(),
                expected_len,
                "Channel {} has length {}, expected {}",
                i,
                channel.len(),
                expected_len
            );
        }

        // Process each channel independently
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            channels
                .par_iter()
                .map(|channel| self.process(channel))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            channels
                .iter()
                .map(|channel| self.process(channel))
                .collect()
        }
    }

    /// Process interleaved multi-channel audio.
    /// Converts interleaved format (e.g., `[L,R,L,R,L,R,...]` for stereo)
    /// into separate Spectrum for each channel.
    ///
    /// # Arguments
    ///
    /// * `data` - Interleaved audio data
    /// * `num_channels` - Number of channels
    ///
    /// # Panics
    ///
    /// Panics if `num_channels` is 0 or if `data.len()` is not divisible by `num_channels`.
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::prelude::*;
    ///
    /// let config = StftConfigF32::default_4096();
    /// let stft = BatchStftF32::new(config);
    ///
    /// // Stereo interleaved: L,R,L,R,L,R,...
    /// let interleaved = vec![0.0; 88200]; // 2 channels * 44100 samples
    ///
    /// let spectra = stft.process_interleaved(&interleaved, 2);
    /// assert_eq!(spectra.len(), 2); // One spectrum per channel
    /// ```
    pub fn process_interleaved(&self, data: &[T], num_channels: usize) -> Vec<Spectrum<T>> {
        let channels = utils::deinterleave(data, num_channels);
        self.process_multichannel(&channels)
    }
}

#[derive(Debug, Clone)]
pub struct BatchIstft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    ifft: Arc<dyn FftBackend<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> BatchIstft<T> {
    pub fn new(config: StftConfig<T>) -> Self
    where
        FftPlanner<T>: FftPlannerTrait<T>,
    {
        let window = config.generate_window();
        let mut planner = <FftPlanner<T> as FftPlannerTrait<T>>::new();
        let ifft = planner.plan_fft_inverse(config.fft_size);

        Self {
            config,
            window,
            ifft,
        }
    }

    pub fn process(&self, spectrum: &Spectrum<T>) -> Vec<T> {
        assert_eq!(
            spectrum.freq_bins,
            self.config.freq_bins(),
            "Frequency bins mismatch"
        );

        let num_frames = spectrum.num_frames;
        let original_time_len = (num_frames - 1) * self.config.hop_size;
        let pad_amount = self.config.fft_size / 2;
        let padded_len = original_time_len + 2 * pad_amount;

        let mut overlap_buffer = vec![T::zero(); padded_len];
        let mut window_energy = vec![T::zero(); padded_len];
        let mut ifft_buffer = vec![Complex::new(T::zero(), T::zero()); self.config.fft_size];

        // Precompute window energy normalization
        for frame_idx in 0..num_frames {
            let pos = frame_idx * self.config.hop_size;
            for i in 0..self.config.fft_size {
                match self.config.reconstruction_mode {
                    ReconstructionMode::Ola => {
                        window_energy[pos + i] = window_energy[pos + i] + self.window[i];
                    }
                    ReconstructionMode::Wola => {
                        window_energy[pos + i] =
                            window_energy[pos + i] + self.window[i] * self.window[i];
                    }
                }
            }
        }

        // Process each frame
        for frame_idx in 0..num_frames {
            // Build full spectrum with conjugate symmetry
            (0..spectrum.freq_bins).for_each(|bin| {
                ifft_buffer[bin] = spectrum.get_complex(frame_idx, bin);
            });

            // Conjugate symmetry for negative frequencies (skip DC and Nyquist)
            for bin in 1..(spectrum.freq_bins - 1) {
                ifft_buffer[self.config.fft_size - bin] = ifft_buffer[bin].conj();
            }

            // Compute IFFT
            self.ifft.process(&mut ifft_buffer);

            // Overlap-add
            let pos = frame_idx * self.config.hop_size;
            for i in 0..self.config.fft_size {
                let fft_size_t = T::from(self.config.fft_size).unwrap();
                let sample = ifft_buffer[i].re / fft_size_t;

                match self.config.reconstruction_mode {
                    ReconstructionMode::Ola => {
                        // OLA: no windowing on inverse
                        overlap_buffer[pos + i] = overlap_buffer[pos + i] + sample;
                    }
                    ReconstructionMode::Wola => {
                        // WOLA: apply window on inverse
                        overlap_buffer[pos + i] = overlap_buffer[pos + i] + sample * self.window[i];
                    }
                }
            }
        }

        // Normalize by window energy
        let threshold = T::from(1e-8).unwrap();
        for i in 0..padded_len {
            if window_energy[i] > threshold {
                overlap_buffer[i] = overlap_buffer[i] / window_energy[i];
            }
        }

        // Remove padding
        overlap_buffer[pad_amount..pad_amount + original_time_len].to_vec()
    }

    /// Process spectrum and write into a pre-allocated output buffer.
    /// The output buffer will be resized if needed.
    pub fn process_into(&self, spectrum: &Spectrum<T>, output: &mut Vec<T>) {
        assert_eq!(
            spectrum.freq_bins,
            self.config.freq_bins(),
            "Frequency bins mismatch"
        );

        let num_frames = spectrum.num_frames;
        let original_time_len = (num_frames - 1) * self.config.hop_size;
        let pad_amount = self.config.fft_size / 2;
        let padded_len = original_time_len + 2 * pad_amount;

        let mut overlap_buffer = vec![T::zero(); padded_len];
        let mut window_energy = vec![T::zero(); padded_len];
        let mut ifft_buffer = vec![Complex::new(T::zero(), T::zero()); self.config.fft_size];

        // Precompute window energy normalization
        for frame_idx in 0..num_frames {
            let pos = frame_idx * self.config.hop_size;
            for i in 0..self.config.fft_size {
                match self.config.reconstruction_mode {
                    ReconstructionMode::Ola => {
                        window_energy[pos + i] = window_energy[pos + i] + self.window[i];
                    }
                    ReconstructionMode::Wola => {
                        window_energy[pos + i] =
                            window_energy[pos + i] + self.window[i] * self.window[i];
                    }
                }
            }
        }

        // Process each frame
        for frame_idx in 0..num_frames {
            // Build full spectrum with conjugate symmetry
            (0..spectrum.freq_bins).for_each(|bin| {
                ifft_buffer[bin] = spectrum.get_complex(frame_idx, bin);
            });

            // Conjugate symmetry for negative frequencies (skip DC and Nyquist)
            for bin in 1..(spectrum.freq_bins - 1) {
                ifft_buffer[self.config.fft_size - bin] = ifft_buffer[bin].conj();
            }

            // Compute IFFT
            self.ifft.process(&mut ifft_buffer);

            // Overlap-add
            let pos = frame_idx * self.config.hop_size;
            for i in 0..self.config.fft_size {
                let fft_size_t = T::from(self.config.fft_size).unwrap();
                let sample = ifft_buffer[i].re / fft_size_t;

                match self.config.reconstruction_mode {
                    ReconstructionMode::Ola => {
                        overlap_buffer[pos + i] = overlap_buffer[pos + i] + sample;
                    }
                    ReconstructionMode::Wola => {
                        overlap_buffer[pos + i] = overlap_buffer[pos + i] + sample * self.window[i];
                    }
                }
            }
        }

        // Normalize by window energy
        let threshold = T::from(1e-8).unwrap();
        for i in 0..padded_len {
            if window_energy[i] > threshold {
                overlap_buffer[i] = overlap_buffer[i] / window_energy[i];
            }
        }

        // Copy to output (resize if needed)
        output.clear();
        output.extend_from_slice(&overlap_buffer[pad_amount..pad_amount + original_time_len]);
    }

    /// Reconstruct multiple channels from their spectra.
    /// Returns one Vec per channel.
    ///
    /// # Arguments
    ///
    /// * `spectra` - Slice of Spectrum, one per channel
    ///
    /// # Panics
    ///
    /// Panics if spectra is empty.
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::prelude::*;
    ///
    /// let config = StftConfigF32::default_4096();
    /// let stft = BatchStftF32::new(config.clone());
    /// let istft = BatchIstftF32::new(config);
    ///
    /// let left = vec![0.0; 44100];
    /// let right = vec![0.0; 44100];
    /// let channels = vec![left, right];
    ///
    /// let spectra = stft.process_multichannel(&channels);
    /// let reconstructed = istft.process_multichannel(&spectra);
    ///
    /// assert_eq!(reconstructed.len(), 2); // One channel per spectrum
    /// ```
    pub fn process_multichannel(&self, spectra: &[Spectrum<T>]) -> Vec<Vec<T>> {
        assert!(!spectra.is_empty(), "spectra must not be empty");

        // Process each spectrum independently
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            spectra
                .par_iter()
                .map(|spectrum| self.process(spectrum))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            spectra
                .iter()
                .map(|spectrum| self.process(spectrum))
                .collect()
        }
    }

    /// Reconstruct multiple channels and interleave them into a single buffer.
    /// Converts separate channels back to interleaved format (e.g., `[L,R,L,R,L,R,...]` for stereo).
    ///
    /// # Arguments
    ///
    /// * `spectra` - Slice of Spectrum, one per channel
    ///
    /// # Panics
    ///
    /// Panics if spectra is empty or if channels have different lengths.
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::prelude::*;
    ///
    /// let config = StftConfigF32::default_4096();
    /// let stft = BatchStftF32::new(config.clone());
    /// let istft = BatchIstftF32::new(config);
    ///
    /// // Process interleaved stereo
    /// let interleaved = vec![0.0; 88200]; // 2 channels * 44100 samples
    /// let spectra = stft.process_interleaved(&interleaved, 2);
    ///
    /// // Reconstruct back to interleaved
    /// let output = istft.process_multichannel_interleaved(&spectra);
    /// // Output length may differ slightly due to padding/framing
    /// assert_eq!(output.len() / 2, 44032); // samples per channel after reconstruction
    /// ```
    pub fn process_multichannel_interleaved(&self, spectra: &[Spectrum<T>]) -> Vec<T> {
        let channels = self.process_multichannel(spectra);
        utils::interleave(&channels)
    }
}

#[derive(Debug, Clone)]
pub struct StreamingStft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    fft: Arc<dyn FftBackend<T>>,
    input_buffer: VecDeque<T>,
    frame_index: usize,
    fft_buffer: Vec<Complex<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> StreamingStft<T> {
    pub fn new(config: StftConfig<T>) -> Self
    where
        FftPlanner<T>: FftPlannerTrait<T>,
    {
        let window = config.generate_window();
        let mut planner = <FftPlanner<T> as FftPlannerTrait<T>>::new();
        let fft = planner.plan_fft_forward(config.fft_size);
        let fft_buffer = vec![Complex::new(T::zero(), T::zero()); config.fft_size];

        Self {
            config,
            window,
            fft,
            input_buffer: VecDeque::new(),
            frame_index: 0,
            fft_buffer,
        }
    }

    pub fn push_samples(&mut self, samples: &[T]) -> Vec<SpectrumFrame<T>> {
        self.input_buffer.extend(samples.iter().copied());

        let mut frames = Vec::new();

        while self.input_buffer.len() >= self.config.fft_size {
            // Process one frame
            for i in 0..self.config.fft_size {
                self.fft_buffer[i] = Complex::new(self.input_buffer[i] * self.window[i], T::zero());
            }

            self.fft.process(&mut self.fft_buffer);

            let freq_bins = self.config.freq_bins();
            let data: Vec<Complex<T>> = self.fft_buffer[..freq_bins].to_vec();
            frames.push(SpectrumFrame::from_data(data));

            // Advance by hop size
            self.input_buffer.drain(..self.config.hop_size);
            self.frame_index += 1;
        }

        frames
    }

    /// Push samples and write frames into a pre-allocated buffer.
    /// Returns the number of frames written.
    pub fn push_samples_into(
        &mut self,
        samples: &[T],
        output: &mut Vec<SpectrumFrame<T>>,
    ) -> usize {
        self.input_buffer.extend(samples.iter().copied());

        let initial_len = output.len();

        while self.input_buffer.len() >= self.config.fft_size {
            // Process one frame
            for i in 0..self.config.fft_size {
                self.fft_buffer[i] = Complex::new(self.input_buffer[i] * self.window[i], T::zero());
            }

            self.fft.process(&mut self.fft_buffer);

            let freq_bins = self.config.freq_bins();
            let data: Vec<Complex<T>> = self.fft_buffer[..freq_bins].to_vec();
            output.push(SpectrumFrame::from_data(data));

            // Advance by hop size
            self.input_buffer.drain(..self.config.hop_size);
            self.frame_index += 1;
        }

        output.len() - initial_len
    }

    /// Push samples and write directly into pre-existing SpectrumFrame buffers.
    /// This is a zero-allocation method - frames must be pre-allocated with correct size.
    /// Returns the number of frames written.
    ///
    /// # Example
    /// ```ignore
    /// let mut frame_pool = vec![SpectrumFrame::new(config.freq_bins()); 16];
    /// let mut frame_index = 0;
    ///
    /// let frames_written = stft.push_samples_write(chunk, &mut frame_pool, &mut frame_index);
    /// // Process frames 0..frames_written
    /// ```
    pub fn push_samples_write(
        &mut self,
        samples: &[T],
        frame_pool: &mut [SpectrumFrame<T>],
        pool_index: &mut usize,
    ) -> usize {
        self.input_buffer.extend(samples.iter().copied());

        let initial_index = *pool_index;
        let freq_bins = self.config.freq_bins();

        while self.input_buffer.len() >= self.config.fft_size && *pool_index < frame_pool.len() {
            // Process one frame
            for i in 0..self.config.fft_size {
                self.fft_buffer[i] = Complex::new(self.input_buffer[i] * self.window[i], T::zero());
            }

            self.fft.process(&mut self.fft_buffer);

            // Write directly into the pre-allocated frame
            let frame = &mut frame_pool[*pool_index];
            debug_assert_eq!(
                frame.freq_bins, freq_bins,
                "Frame pool frames must match freq_bins"
            );
            frame.data[..freq_bins].copy_from_slice(&self.fft_buffer[..freq_bins]);

            // Advance by hop size
            self.input_buffer.drain(..self.config.hop_size);
            self.frame_index += 1;
            *pool_index += 1;
        }

        *pool_index - initial_index
    }

    pub fn flush(&mut self) -> Vec<SpectrumFrame<T>> {
        // For streaming, we typically don't process partial frames
        // Could zero-pad if needed, but that changes the signal
        Vec::new()
    }

    pub fn reset(&mut self) {
        self.input_buffer.clear();
        self.frame_index = 0;
    }

    pub fn buffered_samples(&self) -> usize {
        self.input_buffer.len()
    }
}

/// Multi-channel streaming STFT processor with independent state per channel.
#[derive(Debug, Clone)]
pub struct MultiChannelStreamingStft<T: Float + FftNum> {
    processors: Vec<StreamingStft<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> MultiChannelStreamingStft<T>
where
    FftPlanner<T>: FftPlannerTrait<T>,
{
    /// Create a new multi-channel streaming STFT processor.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT configuration
    /// * `num_channels` - Number of channels
    pub fn new(config: StftConfig<T>, num_channels: usize) -> Self {
        assert!(num_channels > 0, "num_channels must be > 0");
        let processors = (0..num_channels)
            .map(|_| StreamingStft::new(config.clone()))
            .collect();
        Self { processors }
    }

    /// Push samples for all channels and get frames for each channel.
    /// Returns Vec<Vec<SpectrumFrame>>, outer Vec = channels, inner Vec = frames.
    ///
    /// # Arguments
    ///
    /// * `channels` - Slice of sample slices, one per channel
    ///
    /// # Panics
    ///
    /// Panics if channels.len() doesn't match num_channels.
    pub fn push_samples(&mut self, channels: &[&[T]]) -> Vec<Vec<SpectrumFrame<T>>> {
        assert_eq!(
            channels.len(),
            self.processors.len(),
            "Expected {} channels, got {}",
            self.processors.len(),
            channels.len()
        );

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            self.processors
                .par_iter_mut()
                .zip(channels.par_iter())
                .map(|(stft, channel)| stft.push_samples(channel))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.processors
                .iter_mut()
                .zip(channels.iter())
                .map(|(stft, channel)| stft.push_samples(channel))
                .collect()
        }
    }

    /// Flush all channels and return remaining frames.
    pub fn flush(&mut self) -> Vec<Vec<SpectrumFrame<T>>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            self.processors
                .par_iter_mut()
                .map(|stft| stft.flush())
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.processors
                .iter_mut()
                .map(|stft| stft.flush())
                .collect()
        }
    }

    /// Reset all channels.
    pub fn reset(&mut self) {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            self.processors.par_iter_mut().for_each(|stft| stft.reset());
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.processors.iter_mut().for_each(|stft| stft.reset());
        }
    }

    /// Get the number of channels.
    pub fn num_channels(&self) -> usize {
        self.processors.len()
    }
}

#[derive(Debug, Clone)]
pub struct StreamingIstft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    ifft: Arc<dyn FftBackend<T>>,
    overlap_buffer: Vec<T>,
    window_energy: Vec<T>,
    output_position: usize,
    frames_processed: usize,
    ifft_buffer: Vec<Complex<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> StreamingIstft<T> {
    pub fn new(config: StftConfig<T>) -> Self
    where
        FftPlanner<T>: FftPlannerTrait<T>,
    {
        let window = config.generate_window();
        let mut planner = <FftPlanner<T> as FftPlannerTrait<T>>::new();
        let ifft = planner.plan_fft_inverse(config.fft_size);

        // Buffer needs to hold enough samples for full overlap
        // For proper reconstruction, need at least fft_size samples
        let buffer_size = config.fft_size * 2;
        let ifft_buffer = vec![Complex::new(T::zero(), T::zero()); config.fft_size];

        Self {
            config,
            window,
            ifft,
            overlap_buffer: vec![T::zero(); buffer_size],
            window_energy: vec![T::zero(); buffer_size],
            output_position: 0,
            frames_processed: 0,
            ifft_buffer,
        }
    }

    pub fn push_frame(&mut self, frame: &SpectrumFrame<T>) -> Vec<T> {
        assert_eq!(
            frame.freq_bins,
            self.config.freq_bins(),
            "Frequency bins mismatch"
        );

        // Build full spectrum with conjugate symmetry
        for bin in 0..frame.freq_bins {
            self.ifft_buffer[bin] = frame.data[bin];
        }

        // Conjugate symmetry for negative frequencies (skip DC and Nyquist)
        for bin in 1..(frame.freq_bins - 1) {
            self.ifft_buffer[self.config.fft_size - bin] = self.ifft_buffer[bin].conj();
        }

        // Compute IFFT
        self.ifft.process(&mut self.ifft_buffer);

        // Overlap-add into buffer at the current write position
        let write_pos = self.frames_processed * self.config.hop_size;
        for i in 0..self.config.fft_size {
            let fft_size_t = T::from(self.config.fft_size).unwrap();
            let sample = self.ifft_buffer[i].re / fft_size_t;
            let buf_idx = write_pos + i;

            // Extend buffers if needed
            if buf_idx >= self.overlap_buffer.len() {
                self.overlap_buffer.resize(buf_idx + 1, T::zero());
                self.window_energy.resize(buf_idx + 1, T::zero());
            }

            match self.config.reconstruction_mode {
                ReconstructionMode::Ola => {
                    self.overlap_buffer[buf_idx] = self.overlap_buffer[buf_idx] + sample;
                    self.window_energy[buf_idx] = self.window_energy[buf_idx] + self.window[i];
                }
                ReconstructionMode::Wola => {
                    self.overlap_buffer[buf_idx] =
                        self.overlap_buffer[buf_idx] + sample * self.window[i];
                    self.window_energy[buf_idx] =
                        self.window_energy[buf_idx] + self.window[i] * self.window[i];
                }
            }
        }

        self.frames_processed += 1;

        // Calculate how many samples are "ready" (have full window energy)
        // Samples are ready when no future frames will contribute to them
        let ready_until = if self.frames_processed == 1 {
            0 // First frame: no output yet, need overlap
        } else {
            // Samples before the current frame's start position are complete
            (self.frames_processed - 1) * self.config.hop_size
        };

        // Extract ready samples
        let output_start = self.output_position;
        let output_end = ready_until;
        let mut output = Vec::new();

        let threshold = T::from(1e-8).unwrap();
        if output_end > output_start {
            for i in output_start..output_end {
                let normalized = if self.window_energy[i] > threshold {
                    self.overlap_buffer[i] / self.window_energy[i]
                } else {
                    T::zero()
                };
                output.push(normalized);
            }
            self.output_position = output_end;
        }

        output
    }

    /// Push a frame and write output samples into a pre-allocated buffer.
    /// Returns the number of samples written.
    pub fn push_frame_into(&mut self, frame: &SpectrumFrame<T>, output: &mut Vec<T>) -> usize {
        assert_eq!(
            frame.freq_bins,
            self.config.freq_bins(),
            "Frequency bins mismatch"
        );

        // Build full spectrum with conjugate symmetry
        for bin in 0..frame.freq_bins {
            self.ifft_buffer[bin] = frame.data[bin];
        }

        // Conjugate symmetry for negative frequencies (skip DC and Nyquist)
        for bin in 1..(frame.freq_bins - 1) {
            self.ifft_buffer[self.config.fft_size - bin] = self.ifft_buffer[bin].conj();
        }

        // Compute IFFT
        self.ifft.process(&mut self.ifft_buffer);

        // Overlap-add into buffer at the current write position
        let write_pos = self.frames_processed * self.config.hop_size;
        for i in 0..self.config.fft_size {
            let fft_size_t = T::from(self.config.fft_size).unwrap();
            let sample = self.ifft_buffer[i].re / fft_size_t;
            let buf_idx = write_pos + i;

            // Extend buffers if needed
            if buf_idx >= self.overlap_buffer.len() {
                self.overlap_buffer.resize(buf_idx + 1, T::zero());
                self.window_energy.resize(buf_idx + 1, T::zero());
            }

            match self.config.reconstruction_mode {
                ReconstructionMode::Ola => {
                    self.overlap_buffer[buf_idx] = self.overlap_buffer[buf_idx] + sample;
                    self.window_energy[buf_idx] = self.window_energy[buf_idx] + self.window[i];
                }
                ReconstructionMode::Wola => {
                    self.overlap_buffer[buf_idx] =
                        self.overlap_buffer[buf_idx] + sample * self.window[i];
                    self.window_energy[buf_idx] =
                        self.window_energy[buf_idx] + self.window[i] * self.window[i];
                }
            }
        }

        self.frames_processed += 1;

        // Calculate how many samples are "ready" (have full window energy)
        // Samples are ready when no future frames will contribute to them
        let ready_until = if self.frames_processed == 1 {
            0 // First frame: no output yet, need overlap
        } else {
            // Samples before the current frame's start position are complete
            (self.frames_processed - 1) * self.config.hop_size
        };

        // Extract ready samples
        let output_start = self.output_position;
        let output_end = ready_until;
        let initial_len = output.len();

        let threshold = T::from(1e-8).unwrap();
        if output_end > output_start {
            for i in output_start..output_end {
                let normalized = if self.window_energy[i] > threshold {
                    self.overlap_buffer[i] / self.window_energy[i]
                } else {
                    T::zero()
                };
                output.push(normalized);
            }
            self.output_position = output_end;
        }

        output.len() - initial_len
    }

    pub fn flush(&mut self) -> Vec<T> {
        // Return all remaining samples in buffer
        let mut output = Vec::new();
        let threshold = T::from(1e-8).unwrap();
        for i in self.output_position..self.overlap_buffer.len() {
            if self.window_energy[i] > threshold {
                output.push(self.overlap_buffer[i] / self.window_energy[i]);
            } else if i < (self.frames_processed * self.config.hop_size + self.config.fft_size) {
                output.push(T::zero()); // Sample in valid range but no window energy
            } else {
                break; // Past the end of valid data
            }
        }

        // Determine the actual end of valid data
        let valid_end =
            (self.frames_processed.saturating_sub(1)) * self.config.hop_size + self.config.fft_size;
        if output.len() > valid_end - self.output_position {
            output.truncate(valid_end - self.output_position);
        }

        self.reset();
        output
    }

    pub fn reset(&mut self) {
        self.overlap_buffer.clear();
        self.overlap_buffer
            .resize(self.config.fft_size * 2, T::zero());
        self.window_energy.clear();
        self.window_energy
            .resize(self.config.fft_size * 2, T::zero());
        self.output_position = 0;
        self.frames_processed = 0;
    }
}

/// Multi-channel streaming iSTFT processor with independent state per channel.
#[derive(Debug, Clone)]
pub struct MultiChannelStreamingIstft<T: Float + FftNum> {
    processors: Vec<StreamingIstft<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> MultiChannelStreamingIstft<T>
where
    FftPlanner<T>: FftPlannerTrait<T>,
{
    /// Create a new multi-channel streaming iSTFT processor.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT configuration
    /// * `num_channels` - Number of channels
    pub fn new(config: StftConfig<T>, num_channels: usize) -> Self {
        assert!(num_channels > 0, "num_channels must be > 0");
        let processors = (0..num_channels)
            .map(|_| StreamingIstft::new(config.clone()))
            .collect();
        Self { processors }
    }

    /// Push frames for all channels and get samples for each channel.
    /// Returns Vec<Vec<T>>, outer Vec = channels, inner Vec = samples.
    ///
    /// # Arguments
    ///
    /// * `frames` - Slice of frames, one per channel
    ///
    /// # Panics
    ///
    /// Panics if frames.len() doesn't match num_channels.
    pub fn push_frames(&mut self, frames: &[&SpectrumFrame<T>]) -> Vec<Vec<T>> {
        assert_eq!(
            frames.len(),
            self.processors.len(),
            "Expected {} channels, got {}",
            self.processors.len(),
            frames.len()
        );

        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            self.processors
                .par_iter_mut()
                .zip(frames.par_iter())
                .map(|(istft, frame)| istft.push_frame(frame))
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.processors
                .iter_mut()
                .zip(frames.iter())
                .map(|(istft, frame)| istft.push_frame(frame))
                .collect()
        }
    }

    /// Flush all channels and return remaining samples.
    pub fn flush(&mut self) -> Vec<Vec<T>> {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            self.processors
                .par_iter_mut()
                .map(|istft| istft.flush())
                .collect()
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.processors
                .iter_mut()
                .map(|istft| istft.flush())
                .collect()
        }
    }

    /// Reset all channels.
    pub fn reset(&mut self) {
        #[cfg(feature = "rayon")]
        {
            use rayon::prelude::*;
            self.processors
                .par_iter_mut()
                .for_each(|istft| istft.reset());
        }
        #[cfg(not(feature = "rayon"))]
        {
            self.processors.iter_mut().for_each(|istft| istft.reset());
        }
    }

    /// Get the number of channels.
    pub fn num_channels(&self) -> usize {
        self.processors.len()
    }
}

// Type aliases for common float types
pub type StftConfigF32 = StftConfig<f32>;
pub type StftConfigF64 = StftConfig<f64>;

pub type StftConfigBuilderF32 = StftConfigBuilder<f32>;
pub type StftConfigBuilderF64 = StftConfigBuilder<f64>;

pub type BatchStftF32 = BatchStft<f32>;
pub type BatchStftF64 = BatchStft<f64>;

pub type BatchIstftF32 = BatchIstft<f32>;
pub type BatchIstftF64 = BatchIstft<f64>;

pub type StreamingStftF32 = StreamingStft<f32>;
pub type StreamingStftF64 = StreamingStft<f64>;

pub type StreamingIstftF32 = StreamingIstft<f32>;
pub type StreamingIstftF64 = StreamingIstft<f64>;

pub type SpectrumF32 = Spectrum<f32>;
pub type SpectrumF64 = Spectrum<f64>;

pub type SpectrumFrameF32 = SpectrumFrame<f32>;
pub type SpectrumFrameF64 = SpectrumFrame<f64>;

pub type MultiChannelStreamingStftF32 = MultiChannelStreamingStft<f32>;
pub type MultiChannelStreamingStftF64 = MultiChannelStreamingStft<f64>;

pub type MultiChannelStreamingIstftF32 = MultiChannelStreamingIstft<f32>;
pub type MultiChannelStreamingIstftF64 = MultiChannelStreamingIstft<f64>;
