//! Mel spectrogram computation for audio analysis.
//!
//! This module provides mel-scale frequency analysis, commonly used in speech
//! recognition and audio processing. It converts linear frequency STFT bins
//! into perceptually-motivated mel-scale bins.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "std")]
use std::vec;

use core::fmt;
use num_traits::Float;

/// Mel scale variant for frequency conversion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MelScale {
    /// HTK mel scale formula: 2595 * log10(1 + hz/700)
    Htk,
    /// Slaney mel scale: linear below 1kHz, logarithmic above
    /// Compatible with librosa (default)
    #[default]
    Slaney,
}

/// Normalization method for mel filterbank.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MelNorm {
    /// No normalization
    None,
    /// Slaney normalization: area under each filter = 1
    #[default]
    Slaney,
}

/// Configuration for mel spectrogram computation.
#[derive(Debug, Clone)]
pub struct MelConfig<T: Float> {
    /// Number of mel bands (default: 80 for speech/Whisper)
    pub n_mels: usize,
    /// Minimum frequency in Hz (default: 0.0)
    pub fmin: T,
    /// Maximum frequency in Hz (default: None = sample_rate/2)
    pub fmax: Option<T>,
    /// Mel scale variant (default: Slaney)
    pub mel_scale: MelScale,
    /// Filterbank normalization (default: Slaney)
    pub norm: MelNorm,
    /// Use power spectrum instead of magnitude (default: true)
    pub use_power: bool,
}

impl<T: Float> Default for MelConfig<T> {
    fn default() -> Self {
        MelConfig {
            n_mels: 80,
            fmin: T::zero(),
            fmax: None,
            mel_scale: MelScale::default(),
            norm: MelNorm::default(),
            use_power: true,
        }
    }
}

// Mel scale conversion functions

/// Convert frequency in Hz to mel scale using HTK formula.
///
/// Formula: 2595 * log10(1 + hz/700)
pub fn hz_to_mel_htk<T: Float>(hz: T) -> T {
    let factor = T::from(2595.0).unwrap();
    let divisor = T::from(700.0).unwrap();
    factor * (T::one() + hz / divisor).log10()
}

/// Convert mel scale to frequency in Hz using HTK formula.
///
/// Formula: 700 * (10^(mel/2595) - 1)
pub fn mel_to_hz_htk<T: Float>(mel: T) -> T {
    let factor = T::from(700.0).unwrap();
    let divisor = T::from(2595.0).unwrap();
    factor * (T::from(10.0).unwrap().powf(mel / divisor) - T::one())
}

/// Convert frequency in Hz to mel scale using Slaney formula.
///
/// Linear below 1kHz, logarithmic above (librosa-compatible).
pub fn hz_to_mel_slaney<T: Float>(hz: T) -> T {
    let f_min = T::zero();
    let f_sp = T::from(200.0 / 3.0).unwrap();
    let min_log_hz = T::from(1000.0).unwrap();
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (T::from(6.4).unwrap()).ln() / T::from(27.0).unwrap();

    if hz >= min_log_hz {
        min_log_mel + ((hz / min_log_hz).ln() / logstep)
    } else {
        (hz - f_min) / f_sp
    }
}

/// Convert mel scale to frequency in Hz using Slaney formula.
///
/// Linear below 1kHz, logarithmic above (librosa-compatible).
pub fn mel_to_hz_slaney<T: Float>(mel: T) -> T {
    let f_min = T::zero();
    let f_sp = T::from(200.0 / 3.0).unwrap();
    let min_log_hz = T::from(1000.0).unwrap();
    let min_log_mel = (min_log_hz - f_min) / f_sp;
    let logstep = (T::from(6.4).unwrap()).ln() / T::from(27.0).unwrap();

    if mel >= min_log_mel {
        min_log_hz * (logstep * (mel - min_log_mel)).exp()
    } else {
        f_min + f_sp * mel
    }
}

/// Convert frequency to mel scale based on configured scale type.
pub fn hz_to_mel<T: Float>(hz: T, scale: MelScale) -> T {
    match scale {
        MelScale::Htk => hz_to_mel_htk(hz),
        MelScale::Slaney => hz_to_mel_slaney(hz),
    }
}

/// Convert mel scale to frequency based on configured scale type.
pub fn mel_to_hz<T: Float>(mel: T, scale: MelScale) -> T {
    match scale {
        MelScale::Htk => mel_to_hz_htk(mel),
        MelScale::Slaney => mel_to_hz_slaney(mel),
    }
}

/// Mel filterbank for converting STFT bins to mel-scale bins.
///
/// Stores triangular filters as a sparse matrix where each mel bin
/// has weights for the relevant STFT frequency bins.
#[derive(Clone)]
pub struct MelFilterbank<T: Float> {
    /// Number of mel bands
    pub n_mels: usize,
    /// Number of FFT frequency bins
    pub n_freqs: usize,
    /// Sample rate in Hz
    pub sample_rate: T,
    /// Weights for each mel bin. Each Vec contains (freq_bin_index, weight) pairs.
    pub weights: Vec<Vec<(usize, T)>>,
}

impl<T: Float + fmt::Debug> MelFilterbank<T> {
    /// Create a new mel filterbank.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `n_fft` - FFT size
    /// * `config` - Mel configuration
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::mel::{MelFilterbank, MelConfig};
    ///
    /// let config = MelConfig::<f32>::default();
    /// let filterbank = MelFilterbank::new(44100.0, 4096, &config);
    /// ```
    pub fn new(sample_rate: T, n_fft: usize, config: &MelConfig<T>) -> Self {
        let n_freqs = n_fft / 2 + 1;

        // Determine fmax
        let fmax = config.fmax.unwrap_or(sample_rate / T::from(2.0).unwrap());

        // Convert min/max frequencies to mel scale
        let mel_min = hz_to_mel(config.fmin, config.mel_scale);
        let mel_max = hz_to_mel(fmax, config.mel_scale);

        // Create n_mels+2 points in mel space (including edges)
        let n_mels_plus_2 = config.n_mels + 2;
        let mel_step = (mel_max - mel_min) / T::from(n_mels_plus_2 - 1).unwrap();

        let mel_points: Vec<T> = (0..n_mels_plus_2)
            .map(|i| mel_min + T::from(i).unwrap() * mel_step)
            .collect();

        // Convert mel points to Hz
        let hz_points: Vec<T> = mel_points
            .iter()
            .map(|&mel| mel_to_hz(mel, config.mel_scale))
            .collect();

        // Convert Hz to FFT bin indices (as floats for interpolation)
        let fft_bins: Vec<T> = hz_points
            .iter()
            .map(|&hz| hz * T::from(n_fft).unwrap() / sample_rate)
            .collect();

        // Build triangular filters
        let mut weights = Vec::with_capacity(config.n_mels);

        for i in 0..config.n_mels {
            let left = fft_bins[i];
            let center = fft_bins[i + 1];
            let right = fft_bins[i + 2];

            let mut filter_weights = Vec::new();

            // Find the range of FFT bins this filter covers
            let start_bin = left.floor().to_usize().unwrap_or(0);
            let end_bin = (right.ceil().to_usize().unwrap_or(n_freqs)).min(n_freqs);

            for bin in start_bin..end_bin {
                let freq = T::from(bin).unwrap();
                let weight = if freq < center {
                    // Rising edge
                    if center > left {
                        (freq - left) / (center - left)
                    } else {
                        T::zero()
                    }
                } else {
                    // Falling edge
                    if right > center {
                        (right - freq) / (right - center)
                    } else {
                        T::zero()
                    }
                };

                if weight > T::zero() {
                    filter_weights.push((bin, weight));
                }
            }

            // Apply normalization if requested
            if config.norm == MelNorm::Slaney {
                // Slaney normalization: area under each filter = 1
                let enorm = T::from(2.0).unwrap() / (hz_points[i + 2] - hz_points[i]);
                for (_, w) in &mut filter_weights {
                    *w = *w * enorm;
                }
            }

            weights.push(filter_weights);
        }

        Self {
            n_mels: config.n_mels,
            n_freqs,
            sample_rate,
            weights,
        }
    }

    /// Apply the mel filterbank to a magnitude spectrum.
    ///
    /// # Arguments
    ///
    /// * `magnitudes` - Magnitude spectrum (n_freqs,)
    ///
    /// # Returns
    ///
    /// Mel-scale magnitudes (n_mels,)
    pub fn apply(&self, magnitudes: &[T]) -> Vec<T> {
        assert_eq!(
            magnitudes.len(),
            self.n_freqs,
            "Magnitude spectrum length mismatch"
        );

        let mut mel_mags = vec![T::zero(); self.n_mels];

        for (mel_idx, filter) in self.weights.iter().enumerate() {
            let mut sum = T::zero();
            for &(bin, weight) in filter {
                sum = sum + magnitudes[bin] * weight;
            }
            mel_mags[mel_idx] = sum;
        }

        mel_mags
    }

    /// Apply the mel filterbank to a power spectrum.
    ///
    /// # Arguments
    ///
    /// * `power` - Power spectrum (n_freqs,)
    ///
    /// # Returns
    ///
    /// Mel-scale power (n_mels,)
    #[inline]
    pub fn apply_power(&self, power: &[T]) -> Vec<T> {
        // Same as apply - the filterbank is linear
        self.apply(power)
    }
}

/// Mel-scale spectrum data structure.
///
/// Stores mel spectrogram as (num_frames x n_mels) in row-major order.
#[derive(Clone)]
pub struct MelSpectrum<T: Float> {
    /// Number of time frames
    pub num_frames: usize,
    /// Number of mel bands
    pub n_mels: usize,
    /// Mel spectrogram data (num_frames * n_mels)
    pub data: Vec<T>,
}

impl<T: Float> MelSpectrum<T> {
    /// Create a new empty mel spectrum.
    pub fn new(num_frames: usize, n_mels: usize) -> Self {
        Self {
            num_frames,
            n_mels,
            data: vec![T::zero(); num_frames * n_mels],
        }
    }

    /// Get mel value at (frame, mel_bin).
    #[inline]
    pub fn get(&self, frame: usize, mel_bin: usize) -> T {
        self.data[frame * self.n_mels + mel_bin]
    }

    /// Set mel value at (frame, mel_bin).
    #[inline]
    pub fn set(&mut self, frame: usize, mel_bin: usize, value: T) {
        self.data[frame * self.n_mels + mel_bin] = value;
    }

    /// Get a full frame as a slice.
    pub fn frame(&self, frame: usize) -> &[T] {
        let start = frame * self.n_mels;
        &self.data[start..start + self.n_mels]
    }

    /// Get a mutable reference to a full frame.
    pub fn frame_mut(&mut self, frame: usize) -> &mut [T] {
        let start = frame * self.n_mels;
        &mut self.data[start..start + self.n_mels]
    }

    /// Convert to log-mel (dB scale).
    ///
    /// # Arguments
    ///
    /// * `amin` - Minimum threshold to avoid log(0) (default: 1e-10)
    /// * `top_db` - Maximum dB range (default: 80.0)
    ///
    /// # Returns
    ///
    /// New MelSpectrum in dB scale
    pub fn to_db(&self, amin: Option<T>, top_db: Option<T>) -> Self {
        let amin = amin.unwrap_or(T::from(1e-10).unwrap());
        let top_db = top_db.unwrap_or(T::from(80.0).unwrap());
        let log10_factor = T::from(10.0).unwrap();

        let mut result = self.clone();

        // Convert to dB: 10 * log10(max(value, amin))
        for val in &mut result.data {
            let clamped = if *val < amin { amin } else { *val };
            *val = log10_factor * clamped.log10();
        }

        // Find maximum
        let max_db = result
            .data
            .iter()
            .copied()
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(T::zero());

        // Clamp to top_db range
        let threshold = max_db - top_db;
        for val in &mut result.data {
            if *val < threshold {
                *val = threshold;
            }
        }

        result
    }

    /// Apply a function to all values in-place.
    pub fn apply<F>(&mut self, mut f: F)
    where
        F: FnMut(usize, usize, T) -> T,
    {
        for frame in 0..self.num_frames {
            for mel_bin in 0..self.n_mels {
                let val = self.get(frame, mel_bin);
                self.set(frame, mel_bin, f(frame, mel_bin, val));
            }
        }
    }

    /// Compute delta (first-order derivative) features.
    ///
    /// Uses a centered difference formula with width parameter.
    ///
    /// # Arguments
    ///
    /// * `width` - Number of frames on each side to use (default: 2)
    ///
    /// # Returns
    ///
    /// New MelSpectrum containing delta features
    ///
    /// # Formula
    ///
    /// delta[t] = sum(n * (x[t+n] - x[t-n])) / (2 * sum(n^2))
    /// where n ranges from 1 to width
    pub fn delta(&self, width: Option<usize>) -> Self {
        let width = width.unwrap_or(2);
        let mut result = MelSpectrum::new(self.num_frames, self.n_mels);

        // Precompute denominator: 2 * sum(n^2) for n=1..width
        let denom =
            T::from(2).unwrap() * T::from((1..=width).map(|n| n * n).sum::<usize>()).unwrap();

        for t in 0..self.num_frames {
            for mel_bin in 0..self.n_mels {
                let mut delta_val = T::zero();

                for n in 1..=width {
                    let t_plus = (t + n).min(self.num_frames - 1);
                    let t_minus = t.saturating_sub(n);

                    let val_plus = self.get(t_plus, mel_bin);
                    let val_minus = self.get(t_minus, mel_bin);

                    delta_val = delta_val + T::from(n).unwrap() * (val_plus - val_minus);
                }

                result.set(t, mel_bin, delta_val / denom);
            }
        }

        result
    }

    /// Compute delta-delta (second-order derivative) features.
    ///
    /// Applies delta computation twice.
    ///
    /// # Arguments
    ///
    /// * `width` - Number of frames on each side to use (default: 2)
    ///
    /// # Returns
    ///
    /// New MelSpectrum containing delta-delta features
    pub fn delta_delta(&self, width: Option<usize>) -> Self {
        let delta = self.delta(width);
        delta.delta(width)
    }

    /// Compute concatenated features: [mel, delta, delta-delta].
    ///
    /// Returns a MelSpectrum with 3x the number of mel bins, where each frame
    /// contains [original, delta, delta-delta] concatenated.
    ///
    /// # Arguments
    ///
    /// * `width` - Number of frames on each side to use for delta (default: 2)
    ///
    /// # Returns
    ///
    /// MelSpectrum with n_mels*3 features per frame
    pub fn with_deltas(&self, width: Option<usize>) -> Self {
        let delta = self.delta(width);
        let delta_delta = delta.delta(width);

        let mut result = MelSpectrum::new(self.num_frames, self.n_mels * 3);

        for t in 0..self.num_frames {
            for mel_bin in 0..self.n_mels {
                // Original features
                result.set(t, mel_bin, self.get(t, mel_bin));
                // Delta features
                result.set(t, self.n_mels + mel_bin, delta.get(t, mel_bin));
                // Delta-delta features
                result.set(t, self.n_mels * 2 + mel_bin, delta_delta.get(t, mel_bin));
            }
        }

        result
    }
}

/// Batch mel spectrogram processor.
///
/// Converts STFT Spectrum to mel-scale representation.
pub struct BatchMelSpectrogram<T: Float> {
    filterbank: MelFilterbank<T>,
    use_power: bool,
}

impl<T: Float + fmt::Debug> BatchMelSpectrogram<T> {
    /// Create a new batch mel spectrogram processor.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `n_fft` - FFT size
    /// * `config` - Mel configuration
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::mel::{BatchMelSpectrogram, MelConfig};
    ///
    /// let config = MelConfig::<f32>::default();
    /// let mel_proc = BatchMelSpectrogram::new(44100.0, 4096, &config);
    /// ```
    pub fn new(sample_rate: T, n_fft: usize, config: &MelConfig<T>) -> Self {
        let filterbank = MelFilterbank::new(sample_rate, n_fft, config);
        let use_power = config.use_power;

        Self {
            filterbank,
            use_power,
        }
    }

    /// Process a Spectrum into a MelSpectrum.
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Input STFT spectrum
    ///
    /// # Returns
    ///
    /// MelSpectrum with mel-scale magnitudes or power
    pub fn process(&self, spectrum: &crate::Spectrum<T>) -> MelSpectrum<T> {
        let mut mel_spec = MelSpectrum::new(spectrum.num_frames, self.filterbank.n_mels);

        for frame_idx in 0..spectrum.num_frames {
            let frame_mags: Vec<T> = if self.use_power {
                // Compute power spectrum
                (0..spectrum.freq_bins)
                    .map(|bin| {
                        let re = spectrum.real(frame_idx, bin);
                        let im = spectrum.imag(frame_idx, bin);
                        re * re + im * im
                    })
                    .collect()
            } else {
                // Compute magnitude spectrum
                (0..spectrum.freq_bins)
                    .map(|bin| spectrum.magnitude(frame_idx, bin))
                    .collect()
            };

            let mel_frame = self.filterbank.apply(&frame_mags);
            mel_spec.frame_mut(frame_idx).copy_from_slice(&mel_frame);
        }

        mel_spec
    }

    /// Process a Spectrum and convert to log-mel (dB scale).
    ///
    /// # Arguments
    ///
    /// * `spectrum` - Input STFT spectrum
    /// * `amin` - Minimum threshold to avoid log(0) (default: 1e-10)
    /// * `top_db` - Maximum dB range (default: 80.0)
    ///
    /// # Returns
    ///
    /// MelSpectrum in dB scale
    pub fn process_db(
        &self,
        spectrum: &crate::Spectrum<T>,
        amin: Option<T>,
        top_db: Option<T>,
    ) -> MelSpectrum<T> {
        let mel_spec = self.process(spectrum);
        mel_spec.to_db(amin, top_db)
    }

    /// Get the number of mel bands.
    pub fn n_mels(&self) -> usize {
        self.filterbank.n_mels
    }
}

/// Streaming mel spectrogram processor.
///
/// Processes individual STFT frames into mel-scale frames.
pub struct StreamingMelSpectrogram<T: Float> {
    filterbank: MelFilterbank<T>,
    use_power: bool,
}

impl<T: Float + fmt::Debug> StreamingMelSpectrogram<T> {
    /// Create a new streaming mel spectrogram processor.
    ///
    /// # Arguments
    ///
    /// * `sample_rate` - Sample rate in Hz
    /// * `n_fft` - FFT size
    /// * `config` - Mel configuration
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::mel::{StreamingMelSpectrogram, MelConfig};
    ///
    /// let config = MelConfig::<f32>::default();
    /// let mel_proc = StreamingMelSpectrogram::new(44100.0, 4096, &config);
    /// ```
    pub fn new(sample_rate: T, n_fft: usize, config: &MelConfig<T>) -> Self {
        let filterbank = MelFilterbank::new(sample_rate, n_fft, config);
        let use_power = config.use_power;

        Self {
            filterbank,
            use_power,
        }
    }

    /// Process a single spectrum frame into mel-scale features.
    ///
    /// # Arguments
    ///
    /// * `frame` - Input STFT spectrum frame
    ///
    /// # Returns
    ///
    /// Vector of mel-scale magnitudes or power (length: n_mels)
    pub fn process_frame(&self, frame: &crate::SpectrumFrame<T>) -> Vec<T> {
        assert_eq!(
            frame.freq_bins, self.filterbank.n_freqs,
            "Frequency bins mismatch"
        );

        let frame_mags: Vec<T> = if self.use_power {
            // Compute power spectrum
            frame
                .data
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect()
        } else {
            // Compute magnitude spectrum
            frame.magnitudes()
        };

        self.filterbank.apply(&frame_mags)
    }

    /// Process a frame and write the result into a pre-allocated buffer.
    ///
    /// # Arguments
    ///
    /// * `frame` - Input STFT spectrum frame
    /// * `output` - Output buffer (must have length >= n_mels)
    ///
    /// # Returns
    ///
    /// Number of mel values written
    pub fn process_frame_into(&self, frame: &crate::SpectrumFrame<T>, output: &mut [T]) -> usize {
        assert_eq!(
            frame.freq_bins, self.filterbank.n_freqs,
            "Frequency bins mismatch"
        );
        assert!(
            output.len() >= self.filterbank.n_mels,
            "Output buffer too small"
        );

        let frame_mags: Vec<T> = if self.use_power {
            // Compute power spectrum
            frame
                .data
                .iter()
                .map(|c| c.re * c.re + c.im * c.im)
                .collect()
        } else {
            // Compute magnitude spectrum
            frame.magnitudes()
        };

        let mel_frame = self.filterbank.apply(&frame_mags);
        output[..self.filterbank.n_mels].copy_from_slice(&mel_frame);

        self.filterbank.n_mels
    }

    /// Get the number of mel bands.
    pub fn n_mels(&self) -> usize {
        self.filterbank.n_mels
    }
}

// Type aliases for common float types
pub type MelConfigF32 = MelConfig<f32>;
pub type MelConfigF64 = MelConfig<f64>;

pub type MelFilterbankF32 = MelFilterbank<f32>;
pub type MelFilterbankF64 = MelFilterbank<f64>;

pub type MelSpectrumF32 = MelSpectrum<f32>;
pub type MelSpectrumF64 = MelSpectrum<f64>;

pub type BatchMelSpectrogramF32 = BatchMelSpectrogram<f32>;
pub type BatchMelSpectrogramF64 = BatchMelSpectrogram<f64>;

pub type StreamingMelSpectrogramF32 = StreamingMelSpectrogram<f32>;
pub type StreamingMelSpectrogramF64 = StreamingMelSpectrogram<f64>;
