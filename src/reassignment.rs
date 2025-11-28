//! Reassignment method for sharpening time-frequency representations.
//!
//! This module implements the reassignment method, which sharpens spectrograms
//! by redistributing energy to coordinates closer to the true signal support.
//! This is particularly useful for analyzing signals with sharp transients or
//! harmonics where traditional STFT representations are blurred.
//!
//! # References
//!
//! - Auger, F., & Flandrin, P. (1995). "Improving the readability of
//!   time-frequency and time-scale representations by the reassignment method."
//!   IEEE Transactions on Signal Processing, 43(5), 1068-1089.
//! - Fulop, S. A., & Fitz, K. (2006). "Algorithms for computing the
//!   time-corrected instantaneous frequency (reassigned) spectrogram,
//!   with applications." The Journal of the Acoustical Society of America.

#[cfg(not(feature = "std"))]
use alloc::{vec, vec::Vec};

#[cfg(feature = "std")]
use std::vec;

use core::f64::consts::PI;
use num_traits::Float;

use crate::fft_backend::{Complex, FftNum, FftPlannerTrait};
use crate::{Spectrum, StftConfig, WindowType};

/// Configuration for reassignment computation.
#[derive(Debug, Clone)]
pub struct ReassignmentConfig<T: Float> {
    /// Power threshold for reassignment (relative to max power)
    /// Values below this threshold are not reassigned
    pub power_threshold: T,
    /// Whether to clip reassigned coordinates to valid ranges
    pub clip_to_bounds: bool,
}

impl<T: Float> Default for ReassignmentConfig<T> {
    fn default() -> Self {
        Self {
            power_threshold: T::from(1e-6).unwrap(),
            clip_to_bounds: true,
        }
    }
}

/// Reassigned spectrum data structure.
///
/// Contains both the magnitude spectrum and the reassignment coordinates.
#[derive(Clone)]
pub struct ReassignedSpectrum<T: Float> {
    /// Number of time frames
    pub num_frames: usize,
    /// Number of frequency bins
    pub freq_bins: usize,
    /// Sample rate in Hz
    pub sample_rate: T,
    /// Hop size in samples
    pub hop_size: usize,
    /// Magnitude values for each time-frequency point
    pub magnitudes: Vec<T>,
    /// Reassigned time coordinates (in samples from start)
    pub reassigned_times: Vec<T>,
    /// Reassigned frequency coordinates (in Hz)
    pub reassigned_freqs: Vec<T>,
}

impl<T: Float> ReassignedSpectrum<T> {
    /// Create a new reassigned spectrum.
    pub fn new(
        num_frames: usize,
        freq_bins: usize,
        sample_rate: T,
        hop_size: usize,
    ) -> Self {
        let size = num_frames * freq_bins;
        Self {
            num_frames,
            freq_bins,
            sample_rate,
            hop_size,
            magnitudes: vec![T::zero(); size],
            reassigned_times: vec![T::zero(); size],
            reassigned_freqs: vec![T::zero(); size],
        }
    }

    /// Get magnitude at (frame, bin).
    #[inline]
    pub fn magnitude(&self, frame: usize, bin: usize) -> T {
        self.magnitudes[frame * self.freq_bins + bin]
    }

    /// Get reassigned time at (frame, bin) in samples.
    #[inline]
    pub fn reassigned_time(&self, frame: usize, bin: usize) -> T {
        self.reassigned_times[frame * self.freq_bins + bin]
    }

    /// Get reassigned frequency at (frame, bin) in Hz.
    #[inline]
    pub fn reassigned_freq(&self, frame: usize, bin: usize) -> T {
        self.reassigned_freqs[frame * self.freq_bins + bin]
    }

    /// Set values at (frame, bin).
    #[inline]
    pub fn set(
        &mut self,
        frame: usize,
        bin: usize,
        magnitude: T,
        time: T,
        freq: T,
    ) {
        let idx = frame * self.freq_bins + bin;
        self.magnitudes[idx] = magnitude;
        self.reassigned_times[idx] = time;
        self.reassigned_freqs[idx] = freq;
    }

    /// Render the reassigned spectrum to a regular spectrogram grid.
    ///
    /// This maps the reassigned energy back to a regular time-frequency grid
    /// for visualization or further processing.
    pub fn render_to_grid(&self) -> Spectrum<T> {
        let mut spectrum = Spectrum::new(self.num_frames, self.freq_bins);

        for frame in 0..self.num_frames {
            for bin in 0..self.freq_bins {
                let magnitude = self.magnitude(frame, bin);

                // Skip very small values
                if magnitude < T::from(1e-10).unwrap() {
                    continue;
                }

                let reassigned_time = self.reassigned_time(frame, bin);
                let reassigned_freq = self.reassigned_freq(frame, bin);

                // Convert reassigned time to frame index
                let target_frame_f = reassigned_time / T::from(self.hop_size).unwrap();
                let target_frame = target_frame_f.round().to_usize().unwrap_or(0);

                // Convert reassigned frequency to bin index
                let freq_per_bin = self.sample_rate / T::from(self.freq_bins * 2).unwrap();
                let target_bin_f = reassigned_freq / freq_per_bin;
                let target_bin = target_bin_f.round().to_usize().unwrap_or(0);

                // Check bounds
                if target_frame < self.num_frames && target_bin < self.freq_bins {
                    // Accumulate magnitude (multiple points may map to same bin)
                    let current = spectrum.magnitude(target_frame, target_bin);
                    spectrum.set_magnitude_phase(target_frame, target_bin, current + magnitude, T::zero());
                }
            }
        }

        spectrum
    }
}

/// Generate the derivative of a window function.
///
/// This computes the numerical derivative of the window using finite differences.
pub fn window_derivative<T: Float>(window: &[T]) -> Vec<T> {
    let n = window.len();
    let mut derivative = vec![T::zero(); n];

    if n == 0 {
        return derivative;
    }

    // Forward difference for first point
    if n > 1 {
        derivative[0] = window[1] - window[0];
    }

    // Central differences for interior points
    for i in 1..n - 1 {
        derivative[i] = (window[i + 1] - window[i - 1]) / T::from(2.0).unwrap();
    }

    // Backward difference for last point
    if n > 1 {
        derivative[n - 1] = window[n - 1] - window[n - 2];
    }

    derivative
}

/// Generate a time-ramped window function.
///
/// This multiplies the window by a linear ramp centered at the window center:
/// w_t[n] = (n - center) * w[n]
pub fn window_time_ramp<T: Float>(window: &[T]) -> Vec<T> {
    let n = window.len();
    let center = T::from(n).unwrap() / T::from(2.0).unwrap();

    window
        .iter()
        .enumerate()
        .map(|(i, &w)| (T::from(i).unwrap() - center) * w)
        .collect()
}

/// Generate a window function of the specified type.
///
/// This is a helper function that creates the window coefficients.
pub fn generate_window<T: Float>(window_type: WindowType, size: usize) -> Vec<T> {
    let mut window = vec![T::zero(); size];
    let n_f = T::from(size).unwrap();
    let pi_t = T::from(PI).unwrap();

    match window_type {
        WindowType::Hann => {
            for i in 0..size {
                let i_f = T::from(i).unwrap();
                let factor = T::from(2.0).unwrap() * pi_t * i_f / n_f;
                window[i] = T::from(0.5).unwrap() * (T::one() - factor.cos());
            }
        }
        WindowType::Hamming => {
            for i in 0..size {
                let i_f = T::from(i).unwrap();
                let factor = T::from(2.0).unwrap() * pi_t * i_f / n_f;
                window[i] = T::from(0.54).unwrap() - T::from(0.46).unwrap() * factor.cos();
            }
        }
        WindowType::Blackman => {
            for i in 0..size {
                let i_f = T::from(i).unwrap();
                let factor1 = T::from(2.0).unwrap() * pi_t * i_f / n_f;
                let factor2 = T::from(4.0).unwrap() * pi_t * i_f / n_f;
                window[i] = T::from(0.42).unwrap()
                    - T::from(0.5).unwrap() * factor1.cos()
                    + T::from(0.08).unwrap() * factor2.cos();
            }
        }
    }

    window
}

/// Batch reassignment processor.
///
/// Computes the reassigned STFT by calculating three different STFTs
/// with different windows and deriving the reassignment coordinates.
pub struct BatchReassignment<T: FftNum> {
    config: StftConfig<T>,
    reassign_config: ReassignmentConfig<T>,
    window: Vec<T>,
    window_derivative: Vec<T>,
    window_time_ramp: Vec<T>,
}

impl<T: FftNum> BatchReassignment<T> {
    /// Create a new batch reassignment processor.
    ///
    /// # Arguments
    ///
    /// * `config` - STFT configuration
    /// * `reassign_config` - Reassignment-specific configuration
    ///
    /// # Example
    ///
    /// ```
    /// use stft_rs::{StftConfig, WindowType, ReconstructionMode};
    /// use stft_rs::reassignment::{BatchReassignment, ReassignmentConfig};
    ///
    /// let stft_config = StftConfig::<f32>::builder()
    ///     .fft_size(2048)
    ///     .hop_size(512)
    ///     .window(WindowType::Hann)
    ///     .reconstruction_mode(ReconstructionMode::Ola)
    ///     .build()
    ///     .unwrap();
    ///
    /// let reassign_config = ReassignmentConfig::default();
    /// let reassignment = BatchReassignment::new(stft_config, reassign_config);
    /// ```
    pub fn new(config: StftConfig<T>, reassign_config: ReassignmentConfig<T>) -> Self {
        let window = generate_window(config.window, config.fft_size);
        let window_derivative = window_derivative(&window);
        let window_time_ramp = window_time_ramp(&window);

        Self {
            config,
            reassign_config,
            window,
            window_derivative,
            window_time_ramp,
        }
    }

    /// Process a signal and compute the reassigned STFT.
    ///
    /// # Arguments
    ///
    /// * `signal` - Input signal
    /// * `sample_rate` - Sample rate in Hz
    /// * `planner` - FFT planner
    ///
    /// # Returns
    ///
    /// ReassignedSpectrum containing magnitudes and reassignment coordinates
    pub fn process<P>(
        &self,
        signal: &[T],
        sample_rate: T,
        planner: &mut P,
    ) -> ReassignedSpectrum<T>
    where
        P: FftPlannerTrait<T>,
    {
        // Compute three STFTs with different windows
        let spectrum_h = self.compute_stft(signal, &self.window, planner);
        let spectrum_dh = self.compute_stft(signal, &self.window_derivative, planner);
        let spectrum_th = self.compute_stft(signal, &self.window_time_ramp, planner);

        // Compute reassignment
        self.compute_reassignment(&spectrum_h, &spectrum_dh, &spectrum_th, sample_rate)
    }

    /// Compute STFT with a given window.
    fn compute_stft<P>(
        &self,
        signal: &[T],
        window: &[T],
        planner: &mut P,
    ) -> Spectrum<T>
    where
        P: FftPlannerTrait<T>,
    {
        let fft = planner.plan_fft_forward(self.config.fft_size);
        let hop_size = self.config.hop_size;
        let fft_size = self.config.fft_size;
        let freq_bins = fft_size / 2 + 1;

        // Calculate number of frames
        let num_frames = if signal.len() >= fft_size {
            1 + (signal.len() - fft_size) / hop_size
        } else {
            0
        };

        let mut spectrum = Spectrum::new(num_frames, freq_bins);
        let mut fft_buffer: Vec<Complex<T>> = vec![Complex::new(T::zero(), T::zero()); fft_size];

        for frame_idx in 0..num_frames {
            let start = frame_idx * hop_size;
            fft_buffer.fill(Complex::new(T::zero(), T::zero()));

            // Apply window and copy to FFT buffer
            for i in 0..fft_size.min(signal.len() - start) {
                let sample = signal[start + i];
                fft_buffer[i] = Complex {
                    re: sample * window[i],
                    im: T::zero(),
                };
            }

            // Perform FFT
            fft.process(&mut fft_buffer);

            // Store only positive frequencies (DC to Nyquist)
            for bin in 0..freq_bins {
                let c = fft_buffer[bin];
                spectrum.set_complex(frame_idx, bin, c);
            }
        }

        spectrum
    }

    /// Compute reassignment from three STFTs.
    fn compute_reassignment(
        &self,
        spectrum_h: &Spectrum<T>,
        spectrum_dh: &Spectrum<T>,
        spectrum_th: &Spectrum<T>,
        sample_rate: T,
    ) -> ReassignedSpectrum<T> {
        let num_frames = spectrum_h.num_frames;
        let freq_bins = spectrum_h.freq_bins;
        let hop_size = self.config.hop_size;
        let fft_size = self.config.fft_size;

        let mut result = ReassignedSpectrum::new(num_frames, freq_bins, sample_rate, hop_size);

        // Find maximum power for thresholding
        let mut max_power = T::zero();
        for frame in 0..num_frames {
            for bin in 0..freq_bins {
                let c = spectrum_h.get_complex(frame, bin);
                let power = c.re * c.re + c.im * c.im;
                if power > max_power {
                    max_power = power;
                }
            }
        }

        let power_threshold = max_power * self.reassign_config.power_threshold;

        // Compute reassignment for each time-frequency point
        for frame in 0..num_frames {
            let time_samples = T::from(frame * hop_size).unwrap();

            for bin in 0..freq_bins {
                // Get complex values from the three STFTs
                let s_h = spectrum_h.get_complex(frame, bin);
                let s_dh = spectrum_dh.get_complex(frame, bin);
                let s_th = spectrum_th.get_complex(frame, bin);

                // Compute magnitude
                let magnitude = (s_h.re * s_h.re + s_h.im * s_h.im).sqrt();
                let power = magnitude * magnitude;

                // Skip if below threshold
                if power < power_threshold {
                    result.set(frame, bin, T::zero(), time_samples, T::zero());
                    continue;
                }

                // Compute reassignment offsets
                // Time reassignment: t_hat = t + Re(S_th / S_h)
                // Frequency reassignment: omega_hat = omega - Im(S_dh / S_h)

                let s_h_conj = Complex {
                    re: s_h.re,
                    im: -s_h.im,
                };
                let s_h_mag_sq = s_h.re * s_h.re + s_h.im * s_h.im;

                // Avoid division by zero
                if s_h_mag_sq < T::from(1e-20).unwrap() {
                    result.set(frame, bin, magnitude, time_samples, T::zero());
                    continue;
                }

                // Compute S_th / S_h = S_th * conj(S_h) / |S_h|^2
                let ratio_th = Complex {
                    re: (s_th.re * s_h_conj.re - s_th.im * s_h_conj.im) / s_h_mag_sq,
                    im: (s_th.re * s_h_conj.im + s_th.im * s_h_conj.re) / s_h_mag_sq,
                };

                // Compute S_dh / S_h = S_dh * conj(S_h) / |S_h|^2
                let ratio_dh = Complex {
                    re: (s_dh.re * s_h_conj.re - s_dh.im * s_h_conj.im) / s_h_mag_sq,
                    im: (s_dh.re * s_h_conj.im + s_dh.im * s_h_conj.re) / s_h_mag_sq,
                };

                // Compute reassigned coordinates
                let time_offset = ratio_th.re;
                let reassigned_time = time_samples + time_offset;

                // Frequency reassignment
                let bin_freq = T::from(bin).unwrap() * sample_rate / T::from(fft_size).unwrap();
                let two_pi = T::from(2.0 * PI).unwrap();
                let freq_offset = -ratio_dh.im / two_pi;
                let reassigned_freq = bin_freq + freq_offset;

                // Apply bounds clipping if enabled
                let (final_time, final_freq) = if self.reassign_config.clip_to_bounds {
                    let max_time = T::from((num_frames - 1) * hop_size).unwrap();
                    let nyquist = sample_rate / T::from(2.0).unwrap();

                    let clipped_time = if reassigned_time < T::zero() {
                        time_samples
                    } else if reassigned_time > max_time {
                        time_samples
                    } else {
                        reassigned_time
                    };

                    let clipped_freq = if reassigned_freq < T::zero() {
                        bin_freq
                    } else if reassigned_freq > nyquist {
                        bin_freq
                    } else {
                        reassigned_freq
                    };

                    (clipped_time, clipped_freq)
                } else {
                    (reassigned_time, reassigned_freq)
                };

                result.set(frame, bin, magnitude, final_time, final_freq);
            }
        }

        result
    }
}

// Type aliases for common float types
pub type ReassignmentConfigF32 = ReassignmentConfig<f32>;
pub type ReassignmentConfigF64 = ReassignmentConfig<f64>;

pub type ReassignedSpectrumF32 = ReassignedSpectrum<f32>;
pub type ReassignedSpectrumF64 = ReassignedSpectrum<f64>;

pub type BatchReassignmentF32 = BatchReassignment<f32>;
pub type BatchReassignmentF64 = BatchReassignment<f64>;
