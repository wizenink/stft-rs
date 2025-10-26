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

use num_traits::{Float, FromPrimitive};
use rustfft::num_complex::Complex;
use rustfft::{Fft, FftNum, FftPlanner};
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

pub mod prelude {
    pub use crate::{
        BatchIstft, BatchStft, PadMode, ReconstructionMode, Spectrum, SpectrumFrame, StftConfig,
        StreamingIstft, StreamingStft, WindowType, apply_padding,
    };
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReconstructionMode {
    /// Overlap-Add: normalize by sum(w), requires COLA condition
    Ola,

    /// Weighted Overlap-Add: normalize by sum(w^2), requires NOLA condition
    Wola,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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

impl<T: Float + fmt::Display + fmt::Debug> std::error::Error for ConfigError<T> {}

#[derive(Debug, Clone, Copy)]
pub enum PadMode {
    Reflect,
    Zero,
    Edge,
}

#[derive(Clone)]
pub struct StftConfig<T: Float> {
    pub fft_size: usize,
    pub hop_size: usize,
    pub window: WindowType,
    pub reconstruction_mode: ReconstructionMode,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float + FromPrimitive + fmt::Debug> StftConfig<T> {
    fn nola_threshold() -> T {
        T::from(1e-8).unwrap()
    }

    fn cola_relative_tolerance() -> T {
        T::from(1e-4).unwrap()
    }

    pub fn new(
        fft_size: usize,
        hop_size: usize,
        window: WindowType,
        reconstruction_mode: ReconstructionMode,
    ) -> Result<Self, ConfigError<T>> {
        if fft_size == 0 || !fft_size.is_power_of_two() {
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
            _phantom: std::marker::PhantomData,
        };

        // Validate appropriate condition based on reconstruction mode
        match reconstruction_mode {
            ReconstructionMode::Ola => config.validate_cola()?,
            ReconstructionMode::Wola => config.validate_nola()?,
        }

        Ok(config)
    }

    /// Default: 4096 FFT, 1024 hop, Hann window, OLA mode
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
        let num_overlaps = (self.fft_size + self.hop_size - 1) / self.hop_size;
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
        for i in 0..window_len {
            let idx = i % self.hop_size;
            cola_sum_period[idx] = cola_sum_period[idx] + window[i];
        }

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

fn generate_window<T: Float + FromPrimitive>(window_type: WindowType, size: usize) -> Vec<T> {
    let pi = T::from(std::f64::consts::PI).unwrap();
    let two = T::from(2.0).unwrap();

    match window_type {
        WindowType::Hann => (0..size)
            .map(|i| {
                let half = T::from(0.5).unwrap();
                let one = T::one();
                let i_t = T::from(i).unwrap();
                let size_m1 = T::from(size - 1).unwrap();
                half * (one - (two * pi * i_t / size_m1).cos())
            })
            .collect(),
        WindowType::Hamming => (0..size)
            .map(|i| {
                let i_t = T::from(i).unwrap();
                let size_m1 = T::from(size - 1).unwrap();
                T::from(0.54).unwrap() - T::from(0.46).unwrap() * (two * pi * i_t / size_m1).cos()
            })
            .collect(),
        WindowType::Blackman => (0..size)
            .map(|i| {
                let i_t = T::from(i).unwrap();
                let size_m1 = T::from(size - 1).unwrap();
                let angle = two * pi * i_t / size_m1;
                T::from(0.42).unwrap() - T::from(0.5).unwrap() * angle.cos() + T::from(0.08).unwrap() * (two * angle).cos()
            })
            .collect(),
    }
}

#[derive(Clone)]
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
}

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
}

pub struct BatchStft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    fft: Arc<dyn Fft<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> BatchStft<T> {
    pub fn new(config: StftConfig<T>) -> Self {
        let window = config.generate_window();
        let mut planner = FftPlanner::new();
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
        let padded = apply_padding(signal, pad_amount, pad_mode);

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
            for bin in 0..freq_bins {
                let idx = frame_idx * freq_bins + bin;
                result.data[idx] = fft_buffer[bin].re;
                result.data[num_frames * freq_bins + idx] = fft_buffer[bin].im;
            }
        }

        result
    }
}

pub struct BatchIstft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    ifft: Arc<dyn Fft<T>>,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> BatchIstft<T> {
    pub fn new(config: StftConfig<T>) -> Self {
        let window = config.generate_window();
        let mut planner = FftPlanner::new();
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
                        window_energy[pos + i] = window_energy[pos + i] + self.window[i] * self.window[i];
                    }
                }
            }
        }

        // Process each frame
        for frame_idx in 0..num_frames {
            // Build full spectrum with conjugate symmetry
            for bin in 0..spectrum.freq_bins {
                ifft_buffer[bin] = spectrum.get_complex(frame_idx, bin);
            }

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
}

pub struct StreamingStft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    fft: Arc<dyn Fft<T>>,
    input_buffer: VecDeque<T>,
    frame_index: usize,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> StreamingStft<T> {
    pub fn new(config: StftConfig<T>) -> Self {
        let window = config.generate_window();
        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.fft_size);

        Self {
            config,
            window,
            fft,
            input_buffer: VecDeque::new(),
            frame_index: 0,
        }
    }

    pub fn push_samples(&mut self, samples: &[T]) -> Vec<SpectrumFrame<T>> {
        self.input_buffer.extend(samples.iter().copied());

        let mut frames = Vec::new();
        let mut fft_buffer = vec![Complex::new(T::zero(), T::zero()); self.config.fft_size];

        while self.input_buffer.len() >= self.config.fft_size {
            // Process one frame
            for i in 0..self.config.fft_size {
                fft_buffer[i] = Complex::new(self.input_buffer[i] * self.window[i], T::zero());
            }

            self.fft.process(&mut fft_buffer);

            let freq_bins = self.config.freq_bins();
            let data: Vec<Complex<T>> = fft_buffer[..freq_bins].to_vec();
            frames.push(SpectrumFrame::from_data(data));

            // Advance by hop size
            self.input_buffer.drain(..self.config.hop_size);
            self.frame_index += 1;
        }

        frames
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

pub struct StreamingIstft<T: Float + FftNum> {
    config: StftConfig<T>,
    window: Vec<T>,
    ifft: Arc<dyn Fft<T>>,
    overlap_buffer: Vec<T>,
    window_energy: Vec<T>,
    output_position: usize,
    frames_processed: usize,
}

impl<T: Float + FftNum + FromPrimitive + fmt::Debug> StreamingIstft<T> {
    pub fn new(config: StftConfig<T>) -> Self {
        let window = config.generate_window();
        let mut planner = FftPlanner::new();
        let ifft = planner.plan_fft_inverse(config.fft_size);

        // Buffer needs to hold enough samples for full overlap
        // For proper reconstruction, need at least fft_size samples
        let buffer_size = config.fft_size * 2;

        Self {
            config,
            window,
            ifft,
            overlap_buffer: vec![T::zero(); buffer_size],
            window_energy: vec![T::zero(); buffer_size],
            output_position: 0,
            frames_processed: 0,
        }
    }

    pub fn push_frame(&mut self, frame: &SpectrumFrame<T>) -> Vec<T> {
        assert_eq!(
            frame.freq_bins,
            self.config.freq_bins(),
            "Frequency bins mismatch"
        );

        let mut ifft_buffer = vec![Complex::new(T::zero(), T::zero()); self.config.fft_size];

        // Build full spectrum with conjugate symmetry
        for bin in 0..frame.freq_bins {
            ifft_buffer[bin] = frame.data[bin];
        }

        // Conjugate symmetry for negative frequencies (skip DC and Nyquist)
        for bin in 1..(frame.freq_bins - 1) {
            ifft_buffer[self.config.fft_size - bin] = ifft_buffer[bin].conj();
        }

        // Compute IFFT
        self.ifft.process(&mut ifft_buffer);

        // Overlap-add into buffer at the current write position
        let write_pos = self.frames_processed * self.config.hop_size;
        for i in 0..self.config.fft_size {
            let fft_size_t = T::from(self.config.fft_size).unwrap();
            let sample = ifft_buffer[i].re / fft_size_t;
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
                    self.overlap_buffer[buf_idx] = self.overlap_buffer[buf_idx] + sample * self.window[i];
                    self.window_energy[buf_idx] = self.window_energy[buf_idx] + self.window[i] * self.window[i];
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
        self.overlap_buffer.resize(self.config.fft_size * 2, T::zero());
        self.window_energy.clear();
        self.window_energy.resize(self.config.fft_size * 2, T::zero());
        self.output_position = 0;
        self.frames_processed = 0;
    }
}

/// Apply padding to a signal.
/// Streaming applications should pad manually to match batch processing quality.
pub fn apply_padding<T: Float>(signal: &[T], pad_amount: usize, mode: PadMode) -> Vec<T> {
    let total_len = signal.len() + 2 * pad_amount;
    let mut padded = vec![T::zero(); total_len];

    padded[pad_amount..pad_amount + signal.len()].copy_from_slice(signal);

    match mode {
        PadMode::Reflect => {
            for i in 0..pad_amount {
                if i + 1 < signal.len() {
                    padded[pad_amount - 1 - i] = signal[i + 1];
                }
            }

            let n = signal.len();
            for i in 0..pad_amount {
                if n >= 2 && n - 2 >= i {
                    padded[pad_amount + n + i] = signal[n - 2 - i];
                }
            }
        }
        PadMode::Zero => {}
        PadMode::Edge => {
            if !signal.is_empty() {
                for i in 0..pad_amount {
                    padded[i] = signal[0];
                }
                for i in 0..pad_amount {
                    padded[pad_amount + signal.len() + i] = signal[signal.len() - 1];
                }
            }
        }
    }

    padded
}

#[cfg(test)]
mod tests {
    use super::*;

    fn calculate_snr(original: &[f32], reconstructed: &[f32]) -> f32 {
        assert_eq!(original.len(), reconstructed.len());

        let signal_power: f32 = original.iter().map(|x| x.powi(2)).sum();
        let noise_power: f32 = original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).powi(2))
            .sum();

        if noise_power == 0.0 {
            f32::INFINITY
        } else {
            10.0 * (signal_power / noise_power).log10()
        }
    }

    fn max_abs_error(original: &[f32], reconstructed: &[f32]) -> f32 {
        original
            .iter()
            .zip(reconstructed.iter())
            .map(|(o, r)| (o - r).abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(0.0)
    }

    #[test]
    fn test_config_validation_ola() {
        let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Ola);
        assert!(config.is_ok());
    }

    #[test]
    fn test_config_validation_wola() {
        let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Wola);
        assert!(config.is_ok());
    }

    #[test]
    fn test_config_invalid_ola() {
        let config = StftConfig::<f32>::new(4096, 3000, WindowType::Hann, ReconstructionMode::Ola);
        assert!(config.is_err())
    }

    #[test]
    fn test_config_invalid_fft_size() {
        let config = StftConfig::<f32>::new(4095, 1024, WindowType::Hann, ReconstructionMode::Ola);
        assert!(matches!(config, Err(ConfigError::InvalidFftSize)));
    }

    #[test]
    fn test_config_invalid_hop_size() {
        let config = StftConfig::<f32>::new(4096, 0, WindowType::Hann, ReconstructionMode::Ola);
        assert!(matches!(config, Err(ConfigError::InvalidHopSize)));

        let config = StftConfig::<f32>::new(4096, 5000, WindowType::Hann, ReconstructionMode::Ola);
        assert!(matches!(config, Err(ConfigError::InvalidHopSize)));
    }

    #[test]
    fn test_batch_ola_roundtrip() {
        let config = StftConfig::<f32>::default_4096();
        let stft = BatchStft::new(config.clone());
        let istft = BatchIstft::new(config.clone());

        // Generate test signal (127 hops = 127 * 1024 samples)
        let signal_len = 127 * 1024;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| ((i as f32 * 0.01).sin() * 0.1))
            .collect();

        let spectrum = stft.process(&original);
        let reconstructed = istft.process(&spectrum);

        assert_eq!(original.len(), reconstructed.len());

        let snr = calculate_snr(&original, &reconstructed);
        println!("Batch OLA SNR: {:.2} dB", snr);
        assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
    }

    #[test]
    fn test_batch_wola_roundtrip() {
        let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Wola)
            .expect("Config should be valid");
        let stft = BatchStft::new(config.clone());
        let istft = BatchIstft::new(config.clone());

        let signal_len = 127 * 1024;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| ((i as f32 * 0.01).sin() * 0.1))
            .collect();

        let spectrum = stft.process(&original);
        let reconstructed = istft.process(&spectrum);

        assert_eq!(original.len(), reconstructed.len());

        let snr = calculate_snr(&original, &reconstructed);
        println!("Batch WOLA SNR: {:.2} dB", snr);
        assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
    }

    #[test]
    fn test_batch_constant_signal() {
        let config = StftConfig::<f32>::default_4096();
        let stft = BatchStft::new(config.clone());
        let istft = BatchIstft::new(config.clone());

        let signal_len = 127 * 1024;
        let original = vec![1.0; signal_len];

        let spectrum = stft.process(&original);
        let reconstructed = istft.process(&spectrum);

        let max_error = max_abs_error(&original, &reconstructed);
        println!("Constant signal max error: {:.6}", max_error);
        assert!(max_error < 0.001, "Max error too large: {:.6}", max_error);
    }

    #[test]
    fn test_streaming_ola_roundtrip() {
        let config = StftConfig::<f32>::default_4096();
        let mut stft = StreamingStft::new(config.clone());
        let mut istft = StreamingIstft::new(config.clone());

        // Generate test signal
        let signal_len = 127 * 1024;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| ((i as f32 * 0.01).sin() * 0.1))
            .collect();

        // For streaming, pad the signal to match batch behavior
        let pad_amount = config.fft_size / 2;
        let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

        // Process in chunks
        let chunk_size = 2048;
        let mut reconstructed = Vec::new();

        for chunk in padded.chunks(chunk_size) {
            let frames = stft.push_samples(chunk);
            for frame in frames {
                let samples = istft.push_frame(&frame);
                reconstructed.extend(samples);
            }
        }

        let remaining_frames = stft.flush();
        for frame in remaining_frames {
            let samples = istft.push_frame(&frame);
            reconstructed.extend(samples);
        }
        reconstructed.extend(istft.flush());

        // Remove padding from reconstruction
        let start = pad_amount.min(reconstructed.len());
        let end = (start + signal_len).min(reconstructed.len());
        let reconstructed_unpadded = &reconstructed[start..end];

        let compare_len = original.len().min(reconstructed_unpadded.len());
        let snr = calculate_snr(
            &original[..compare_len],
            &reconstructed_unpadded[..compare_len],
        );
        println!("Streaming OLA SNR: {:.2} dB", snr);
        assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
    }

    #[test]
    fn test_streaming_wola_roundtrip() {
        let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Wola)
            .expect("Config should be valid");
        let mut stft = StreamingStft::new(config.clone());
        let mut istft = StreamingIstft::new(config.clone());

        let signal_len = 127 * 1024;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| ((i as f32 * 0.01).sin() * 0.1))
            .collect();

        // Pad for streaming
        let pad_amount = config.fft_size / 2;
        let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

        let chunk_size = 2048;
        let mut reconstructed = Vec::new();

        for chunk in padded.chunks(chunk_size) {
            let frames = stft.push_samples(chunk);
            for frame in frames {
                let samples = istft.push_frame(&frame);
                reconstructed.extend(samples);
            }
        }

        let remaining_frames = stft.flush();
        for frame in remaining_frames {
            let samples = istft.push_frame(&frame);
            reconstructed.extend(samples);
        }
        reconstructed.extend(istft.flush());

        // Remove padding
        let start = pad_amount.min(reconstructed.len());
        let end = (start + signal_len).min(reconstructed.len());
        let reconstructed_unpadded = &reconstructed[start..end];

        let compare_len = original.len().min(reconstructed_unpadded.len());
        let snr = calculate_snr(
            &original[..compare_len],
            &reconstructed_unpadded[..compare_len],
        );
        println!("Streaming WOLA SNR: {:.2} dB", snr);
        assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
    }

    #[test]
    fn test_batch_vs_streaming_consistency() {
        let config = StftConfig::<f32>::default_4096();

        // Batch processing (pads internally)
        let batch_stft = BatchStft::new(config.clone());
        let signal_len = 50 * 1024;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| ((i as f32 * 0.01).sin() * 0.1))
            .collect();

        let batch_result = batch_stft.process(&original);

        // Streaming processing - need to pad manually to match batch
        let pad_amount = config.fft_size / 2;
        let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

        let mut streaming_stft = StreamingStft::new(config.clone());
        let streaming_frames = streaming_stft.push_samples(&padded);

        // Compare number of frames
        assert_eq!(batch_result.num_frames, streaming_frames.len());

        // Compare spectral content
        for (frame_idx, streaming_frame) in streaming_frames.iter().enumerate() {
            for bin in 0..batch_result.freq_bins {
                let batch_complex = batch_result.get_complex(frame_idx, bin);
                let streaming_complex = streaming_frame.data[bin];

                let diff_re = (batch_complex.re - streaming_complex.re).abs();
                let diff_im = (batch_complex.im - streaming_complex.im).abs();

                assert!(
                    diff_re < 1e-4,
                    "Real part mismatch at frame {}, bin {}: {} vs {}",
                    frame_idx,
                    bin,
                    batch_complex.re,
                    streaming_complex.re
                );
                assert!(
                    diff_im < 1e-4,
                    "Imag part mismatch at frame {}, bin {}: {} vs {}",
                    frame_idx,
                    bin,
                    batch_complex.im,
                    streaming_complex.im
                );
            }
        }
    }

    #[test]
    fn test_stft_result_accessors() {
        let config = StftConfig::<f32>::default_4096();
        let stft = BatchStft::new(config.clone());

        let signal_len = 10 * 1024;
        let original: Vec<f32> = vec![1.0; signal_len];
        let result = stft.process(&original);

        // Test accessors
        for frame in 0..result.num_frames {
            for bin in 0..result.freq_bins {
                let complex = result.get_complex(frame, bin);
                assert_eq!(complex.re, result.real(frame, bin));
                assert_eq!(complex.im, result.imag(frame, bin));
            }
        }

        // Test frame iterator
        let frames: Vec<_> = result.frames().collect();
        assert_eq!(frames.len(), result.num_frames);
        assert_eq!(frames[0].freq_bins, result.freq_bins);
    }

    #[test]
    fn test_different_windows() {
        for window_type in [WindowType::Hann, WindowType::Hamming, WindowType::Blackman] {
            let config = StftConfig::<f32>::new(4096, 1024, window_type, ReconstructionMode::Ola)
                .expect("Config should be valid");
            let stft = BatchStft::new(config.clone());
            let istft = BatchIstft::new(config.clone());

            let signal_len = 50 * 1024;
            let original: Vec<f32> = (0..signal_len)
                .map(|i| ((i as f32 * 0.01).sin() * 0.1))
                .collect();

            let spectrum = stft.process(&original);
            let reconstructed = istft.process(&spectrum);

            let snr = calculate_snr(&original, &reconstructed);
            println!("{:?} window SNR: {:.2} dB", window_type, snr);
            assert!(snr > 100.0, "{:?} SNR too low: {:.2} dB", window_type, snr);
        }
    }

    #[test]
    fn test_streaming_reset() {
        let config = StftConfig::<f32>::default_4096();
        let mut stft = StreamingStft::new(config.clone());

        let samples = vec![1.0; 5000];
        stft.push_samples(&samples);
        assert!(stft.buffered_samples() > 0);

        stft.reset();
        assert_eq!(stft.buffered_samples(), 0);
    }

    #[test]
    fn test_padding_modes() {
        let config = StftConfig::<f32>::default_4096();
        let stft = BatchStft::new(config.clone());

        let signal_len = 20 * 1024;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| ((i as f32 * 0.01).sin() * 0.1))
            .collect();

        // All padding modes should work
        for pad_mode in [PadMode::Reflect, PadMode::Zero, PadMode::Edge] {
            let result = stft.process_padded(&original, pad_mode);
            assert!(result.num_frames > 0);
            assert_eq!(result.freq_bins, config.freq_bins());
        }
    }
}
