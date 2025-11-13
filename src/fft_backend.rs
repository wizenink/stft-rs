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

//! FFT backend abstraction layer
//!
//! This module provides a unified interface for different FFT implementations:
//! - `rustfft`: Full-featured FFT library for std environments (default)
//! - `microfft`: Lightweight no_std compatible FFT for embedded systems
//!
//! The abstraction allows stft-rs to work in both std and no_std environments
//! by selecting the appropriate backend via feature flags.

#[cfg(not(feature = "std"))]
use alloc::sync::Arc;
#[cfg(feature = "std")]
use std::sync::Arc;

use num_traits::Float;

// Re-export Complex type from rustfft's num_complex
#[cfg(feature = "rustfft-backend")]
pub use rustfft::num_complex::Complex;

// For microfft backend, we define our own Complex type
#[cfg(feature = "microfft-backend")]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Complex<T> {
    pub re: T,
    pub im: T,
}

#[cfg(feature = "microfft-backend")]
impl<T: Float> Complex<T> {
    pub fn new(re: T, im: T) -> Self {
        Self { re, im }
    }

    pub fn conj(&self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }

    pub fn norm_sqr(&self) -> T {
        self.re * self.re + self.im * self.im
    }

    pub fn norm(&self) -> T {
        self.norm_sqr().sqrt()
    }
}

#[cfg(feature = "microfft-backend")]
impl<T: Float> core::ops::Mul<T> for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        Self {
            re: self.re * rhs,
            im: self.im * rhs,
        }
    }
}

#[cfg(feature = "microfft-backend")]
impl<T: Float> core::ops::Add for Complex<T> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re + rhs.re,
            im: self.im + rhs.im,
        }
    }
}

#[cfg(feature = "microfft-backend")]
impl<T: Float> core::ops::Sub for Complex<T> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re - rhs.re,
            im: self.im - rhs.im,
        }
    }
}

#[cfg(feature = "microfft-backend")]
impl<T: Float> core::ops::Mul for Complex<T> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        Self {
            re: self.re * rhs.re - self.im * rhs.im,
            im: self.re * rhs.im + self.im * rhs.re,
        }
    }
}

#[cfg(feature = "microfft-backend")]
impl<T: Float> core::ops::Div for Complex<T> {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        let norm_sqr = rhs.norm_sqr();
        Self {
            re: (self.re * rhs.re + self.im * rhs.im) / norm_sqr,
            im: (self.im * rhs.re - self.re * rhs.im) / norm_sqr,
        }
    }
}

/// Trait for types that can be used with FFT operations
/// When rustfft backend is enabled, this also requires rustfft::FftNum
#[cfg(feature = "rustfft-backend")]
pub trait FftNum: Float + rustfft::FftNum + Send + Sync + 'static {}

#[cfg(not(feature = "rustfft-backend"))]
pub trait FftNum: Float + Send + Sync + 'static {}

#[cfg(feature = "rustfft-backend")]
impl FftNum for f32 {}
#[cfg(feature = "rustfft-backend")]
impl FftNum for f64 {}

#[cfg(not(feature = "rustfft-backend"))]
impl FftNum for f32 {}
#[cfg(not(feature = "rustfft-backend"))]
impl FftNum for f64 {}

/// Trait abstracting FFT operations for both forward and inverse transforms
pub trait FftBackend<T: FftNum>: Send + Sync {
    /// Process FFT in-place
    fn process(&self, buffer: &mut [Complex<T>]);

    /// Get the FFT size
    fn len(&self) -> usize;

    /// Check if FFT size is zero (always false for valid FFTs)
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// FFT planner trait for creating forward and inverse FFT instances
pub trait FftPlannerTrait<T: FftNum> {
    /// Create a new planner
    fn new() -> Self;

    /// Plan a forward FFT of the given size
    fn plan_fft_forward(&mut self, size: usize) -> Arc<dyn FftBackend<T>>;

    /// Plan an inverse FFT of the given size
    fn plan_fft_inverse(&mut self, size: usize) -> Arc<dyn FftBackend<T>>;
}

// ============================================================================
// RustFFT Backend Implementation (for std environments)
// ============================================================================

#[cfg(feature = "rustfft-backend")]
mod rustfft_impl {
    use super::*;
    use rustfft::{Fft, FftPlanner as RustFftPlanner};

    /// Wrapper around rustfft's Fft that implements our FftBackend trait
    struct RustFftWrapper<T: rustfft::FftNum> {
        fft: Arc<dyn Fft<T>>,
    }

    impl<T: FftNum> FftBackend<T> for RustFftWrapper<T> {
        fn process(&self, buffer: &mut [Complex<T>]) {
            // Safety: rustfft::num_complex::Complex and our re-exported Complex
            // have identical memory layout
            let buffer_ptr = buffer.as_mut_ptr() as *mut rustfft::num_complex::Complex<T>;
            let buffer_slice = unsafe { core::slice::from_raw_parts_mut(buffer_ptr, buffer.len()) };
            self.fft.process(buffer_slice);
        }

        fn len(&self) -> usize {
            self.fft.len()
        }
    }

    /// FFT planner using rustfft
    pub struct FftPlanner<T: rustfft::FftNum> {
        planner: RustFftPlanner<T>,
    }

    impl<T: FftNum> FftPlannerTrait<T> for FftPlanner<T> {
        fn new() -> Self {
            Self {
                planner: RustFftPlanner::new(),
            }
        }

        fn plan_fft_forward(&mut self, size: usize) -> Arc<dyn FftBackend<T>> {
            Arc::new(RustFftWrapper {
                fft: self.planner.plan_fft_forward(size),
            })
        }

        fn plan_fft_inverse(&mut self, size: usize) -> Arc<dyn FftBackend<T>> {
            Arc::new(RustFftWrapper {
                fft: self.planner.plan_fft_inverse(size),
            })
        }
    }
}

#[cfg(feature = "rustfft-backend")]
pub use rustfft_impl::FftPlanner;

// ============================================================================
// MicroFFT Backend Implementation (for no_std environments)
// ============================================================================

#[cfg(feature = "microfft-backend")]
mod microfft_impl {
    use super::*;

    /// Helper macro to convert slice to array reference for microfft
    macro_rules! slice_to_array {
        ($slice:expr, $size:expr) => {
            unsafe { &mut *($slice.as_mut_ptr() as *mut [microfft::Complex32; $size]) }
        };
    }

    /// Wrapper around microfft that implements our FftBackend trait
    /// Note: microfft only supports f32 and power-of-2 sizes up to 4096
    struct MicroFftForward {
        size: usize,
    }

    struct MicroFftInverse {
        size: usize,
    }

    impl FftBackend<f32> for MicroFftForward {
        fn process(&self, buffer: &mut [Complex<f32>]) {
            // microfft uses the same Complex32 type layout as ours
            // Safety: Complex<f32> and microfft::Complex32 have identical memory layout
            let buffer_ptr = buffer.as_mut_ptr() as *mut microfft::Complex32;
            let microfft_buffer =
                unsafe { core::slice::from_raw_parts_mut(buffer_ptr, buffer.len()) };

            // Use complex FFT functions from microfft
            // microfft requires exact-sized arrays, so we use unsafe casting
            // microfft functions mutate in-place and return references, but we ignore the return values
            match self.size {
                2 => {
                    let _ = microfft::complex::cfft_2(slice_to_array!(microfft_buffer, 2));
                }
                4 => {
                    let _ = microfft::complex::cfft_4(slice_to_array!(microfft_buffer, 4));
                }
                8 => {
                    let _ = microfft::complex::cfft_8(slice_to_array!(microfft_buffer, 8));
                }
                16 => {
                    let _ = microfft::complex::cfft_16(slice_to_array!(microfft_buffer, 16));
                }
                32 => {
                    let _ = microfft::complex::cfft_32(slice_to_array!(microfft_buffer, 32));
                }
                64 => {
                    let _ = microfft::complex::cfft_64(slice_to_array!(microfft_buffer, 64));
                }
                128 => {
                    let _ = microfft::complex::cfft_128(slice_to_array!(microfft_buffer, 128));
                }
                256 => {
                    let _ = microfft::complex::cfft_256(slice_to_array!(microfft_buffer, 256));
                }
                512 => {
                    let _ = microfft::complex::cfft_512(slice_to_array!(microfft_buffer, 512));
                }
                1024 => {
                    let _ = microfft::complex::cfft_1024(slice_to_array!(microfft_buffer, 1024));
                }
                2048 => {
                    let _ = microfft::complex::cfft_2048(slice_to_array!(microfft_buffer, 2048));
                }
                4096 => {
                    let _ = microfft::complex::cfft_4096(slice_to_array!(microfft_buffer, 4096));
                }
                _ => panic!("microfft only supports power-of-2 sizes from 2 to 4096"),
            }
        }

        fn len(&self) -> usize {
            self.size
        }
    }

    impl FftBackend<f32> for MicroFftInverse {
        fn process(&self, buffer: &mut [Complex<f32>]) {
            // microfft doesn't have inverse FFT, so we implement it using forward FFT
            // IFFT(x) = conj(FFT(conj(x))) / N

            // Step 1: Conjugate input
            for val in buffer.iter_mut() {
                val.im = -val.im;
            }

            // Step 2: Apply forward FFT
            let buffer_ptr = buffer.as_mut_ptr() as *mut microfft::Complex32;
            let microfft_buffer =
                unsafe { core::slice::from_raw_parts_mut(buffer_ptr, buffer.len()) };

            match self.size {
                2 => {
                    let _ = microfft::complex::cfft_2(slice_to_array!(microfft_buffer, 2));
                }
                4 => {
                    let _ = microfft::complex::cfft_4(slice_to_array!(microfft_buffer, 4));
                }
                8 => {
                    let _ = microfft::complex::cfft_8(slice_to_array!(microfft_buffer, 8));
                }
                16 => {
                    let _ = microfft::complex::cfft_16(slice_to_array!(microfft_buffer, 16));
                }
                32 => {
                    let _ = microfft::complex::cfft_32(slice_to_array!(microfft_buffer, 32));
                }
                64 => {
                    let _ = microfft::complex::cfft_64(slice_to_array!(microfft_buffer, 64));
                }
                128 => {
                    let _ = microfft::complex::cfft_128(slice_to_array!(microfft_buffer, 128));
                }
                256 => {
                    let _ = microfft::complex::cfft_256(slice_to_array!(microfft_buffer, 256));
                }
                512 => {
                    let _ = microfft::complex::cfft_512(slice_to_array!(microfft_buffer, 512));
                }
                1024 => {
                    let _ = microfft::complex::cfft_1024(slice_to_array!(microfft_buffer, 1024));
                }
                2048 => {
                    let _ = microfft::complex::cfft_2048(slice_to_array!(microfft_buffer, 2048));
                }
                4096 => {
                    let _ = microfft::complex::cfft_4096(slice_to_array!(microfft_buffer, 4096));
                }
                _ => panic!("microfft only supports power-of-2 sizes from 2 to 4096"),
            }

            // Step 3: Conjugate output (no scaling - library handles 1/N normalization)
            for val in buffer.iter_mut() {
                val.im = -val.im;
            }
        }

        fn len(&self) -> usize {
            self.size
        }
    }

    /// FFT planner for microfft (no actual planning needed, just creates wrappers)
    pub struct FftPlanner<T: FftNum> {
        _phantom: core::marker::PhantomData<T>,
    }

    impl FftPlannerTrait<f32> for FftPlanner<f32> {
        fn new() -> Self {
            Self {
                _phantom: core::marker::PhantomData,
            }
        }

        fn plan_fft_forward(&mut self, size: usize) -> Arc<dyn FftBackend<f32>> {
            // Validate size is power of 2 and within supported range
            if !size.is_power_of_two() || size < 2 || size > 4096 {
                panic!(
                    "microfft only supports power-of-2 sizes from 2 to 4096, got {}",
                    size
                );
            }
            Arc::new(MicroFftForward { size })
        }

        fn plan_fft_inverse(&mut self, size: usize) -> Arc<dyn FftBackend<f32>> {
            if !size.is_power_of_two() || size < 2 || size > 4096 {
                panic!(
                    "microfft only supports power-of-2 sizes from 2 to 4096, got {}",
                    size
                );
            }
            Arc::new(MicroFftInverse { size })
        }
    }

    // f64 is not supported by microfft
    impl FftPlannerTrait<f64> for FftPlanner<f64> {
        fn new() -> Self {
            panic!("microfft backend does not support f64, only f32");
        }

        fn plan_fft_forward(&mut self, _size: usize) -> Arc<dyn FftBackend<f64>> {
            panic!("microfft backend does not support f64, only f32");
        }

        fn plan_fft_inverse(&mut self, _size: usize) -> Arc<dyn FftBackend<f64>> {
            panic!("microfft backend does not support f64, only f32");
        }
    }
}

#[cfg(feature = "microfft-backend")]
pub use microfft_impl::FftPlanner;

// Ensure at least one backend is enabled
#[cfg(not(any(feature = "rustfft-backend", feature = "microfft-backend")))]
compile_error!("At least one FFT backend must be enabled: 'rustfft-backend' or 'microfft-backend'");

// Ensure both backends are not enabled at the same time
#[cfg(all(feature = "rustfft-backend", feature = "microfft-backend"))]
compile_error!("Cannot enable both 'rustfft-backend' and 'microfft-backend' at the same time. Choose one.");
