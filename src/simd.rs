/// SIMD-accelerated operations for STFT processing using pulp
use num_traits::Float;

#[cfg(feature = "simd")]
use pulp::Arch;

/// Apply window function to signal samples (element-wise multiplication)
/// This is used in the forward STFT to window each frame before FFT
#[inline]
pub fn apply_window<T: Float + 'static>(signal: &[T], window: &[T], output: &mut [T]) {
    debug_assert_eq!(signal.len(), window.len());
    debug_assert_eq!(signal.len(), output.len());

    #[cfg(feature = "simd")]
    {
        // Try to use SIMD if available
        let simd = pulp::Arch::new();
        match (
            std::any::TypeId::of::<T>(),
            std::any::TypeId::of::<f32>(),
            std::any::TypeId::of::<f64>(),
        ) {
            (t, f32_id, _) if t == f32_id => {
                apply_window_f32_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f32]>(signal) },
                    unsafe { std::mem::transmute::<&[T], &[f32]>(window) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f32]>(output) },
                );
                return;
            }
            (t, _, f64_id) if t == f64_id => {
                apply_window_f64_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f64]>(signal) },
                    unsafe { std::mem::transmute::<&[T], &[f64]>(window) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f64]>(output) },
                );
                return;
            }
            _ => {}
        }
    }

    // Fallback to scalar implementation
    for i in 0..signal.len() {
        output[i] = signal[i] * window[i];
    }
}

#[cfg(feature = "simd")]
fn apply_window_f32_simd(simd: Arch, signal: &[f32], window: &[f32], output: &mut [f32]) {
    simd.dispatch(|| {
        let (signal_head, signal_tail) = pulp::as_arrays::<4, _>(signal);
        let (window_head, window_tail) = pulp::as_arrays::<4, _>(window);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..signal_head.len() {
            output_head[i] = [
                signal_head[i][0] * window_head[i][0],
                signal_head[i][1] * window_head[i][1],
                signal_head[i][2] * window_head[i][2],
                signal_head[i][3] * window_head[i][3],
            ];
        }

        for i in 0..signal_tail.len() {
            output_tail[i] = signal_tail[i] * window_tail[i];
        }
    });
}

#[cfg(feature = "simd")]
fn apply_window_f64_simd(simd: Arch, signal: &[f64], window: &[f64], output: &mut [f64]) {
    simd.dispatch(|| {
        let (signal_head, signal_tail) = pulp::as_arrays::<4, _>(signal);
        let (window_head, window_tail) = pulp::as_arrays::<4, _>(window);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..signal_head.len() {
            output_head[i] = [
                signal_head[i][0] * window_head[i][0],
                signal_head[i][1] * window_head[i][1],
                signal_head[i][2] * window_head[i][2],
                signal_head[i][3] * window_head[i][3],
            ];
        }

        for i in 0..signal_tail.len() {
            output_tail[i] = signal_tail[i] * window_tail[i];
        }
    });
}

/// Fused multiply-add for overlap-add reconstruction
/// Computes: output[i] += input[i] * scale
#[inline]
pub fn fused_multiply_add<T: Float + 'static>(input: &[T], scale: T, output: &mut [T]) {
    debug_assert_eq!(input.len(), output.len());

    #[cfg(feature = "simd")]
    {
        let simd = pulp::Arch::new();
        match (
            std::any::TypeId::of::<T>(),
            std::any::TypeId::of::<f32>(),
            std::any::TypeId::of::<f64>(),
        ) {
            (t, f32_id, _) if t == f32_id => {
                fused_multiply_add_f32_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f32]>(input) },
                    unsafe { std::mem::transmute_copy::<T, f32>(&scale) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f32]>(output) },
                );
                return;
            }
            (t, _, f64_id) if t == f64_id => {
                fused_multiply_add_f64_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f64]>(input) },
                    unsafe { std::mem::transmute_copy::<T, f64>(&scale) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f64]>(output) },
                );
                return;
            }
            _ => {}
        }
    }

    // Fallback to scalar
    for i in 0..input.len() {
        output[i] = output[i] + input[i] * scale;
    }
}

#[cfg(feature = "simd")]
fn fused_multiply_add_f32_simd(simd: Arch, input: &[f32], scale: f32, output: &mut [f32]) {
    simd.dispatch(|| {
        let (input_head, input_tail) = pulp::as_arrays::<4, _>(input);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..input_head.len() {
            output_head[i][0] += input_head[i][0] * scale;
            output_head[i][1] += input_head[i][1] * scale;
            output_head[i][2] += input_head[i][2] * scale;
            output_head[i][3] += input_head[i][3] * scale;
        }

        for i in 0..input_tail.len() {
            output_tail[i] += input_tail[i] * scale;
        }
    });
}

#[cfg(feature = "simd")]
fn fused_multiply_add_f64_simd(simd: Arch, input: &[f64], scale: f64, output: &mut [f64]) {
    simd.dispatch(|| {
        let (input_head, input_tail) = pulp::as_arrays::<4, _>(input);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..input_head.len() {
            output_head[i][0] += input_head[i][0] * scale;
            output_head[i][1] += input_head[i][1] * scale;
            output_head[i][2] += input_head[i][2] * scale;
            output_head[i][3] += input_head[i][3] * scale;
        }

        for i in 0..input_tail.len() {
            output_tail[i] += input_tail[i] * scale;
        }
    });
}

/// Accumulate squared values for window energy
/// Computes: output[i] += input[i] * input[i]
#[inline]
pub fn accumulate_squared<T: Float + 'static>(input: &[T], output: &mut [T]) {
    debug_assert_eq!(input.len(), output.len());

    #[cfg(feature = "simd")]
    {
        let simd = pulp::Arch::new();
        match (
            std::any::TypeId::of::<T>(),
            std::any::TypeId::of::<f32>(),
            std::any::TypeId::of::<f64>(),
        ) {
            (t, f32_id, _) if t == f32_id => {
                accumulate_squared_f32_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f32]>(input) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f32]>(output) },
                );
                return;
            }
            (t, _, f64_id) if t == f64_id => {
                accumulate_squared_f64_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f64]>(input) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f64]>(output) },
                );
                return;
            }
            _ => {}
        }
    }

    // Fallback to scalar
    for i in 0..input.len() {
        output[i] = output[i] + input[i] * input[i];
    }
}

#[cfg(feature = "simd")]
fn accumulate_squared_f32_simd(simd: Arch, input: &[f32], output: &mut [f32]) {
    simd.dispatch(|| {
        let (input_head, input_tail) = pulp::as_arrays::<4, _>(input);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..input_head.len() {
            output_head[i][0] += input_head[i][0] * input_head[i][0];
            output_head[i][1] += input_head[i][1] * input_head[i][1];
            output_head[i][2] += input_head[i][2] * input_head[i][2];
            output_head[i][3] += input_head[i][3] * input_head[i][3];
        }

        for i in 0..input_tail.len() {
            output_tail[i] += input_tail[i] * input_tail[i];
        }
    });
}

#[cfg(feature = "simd")]
fn accumulate_squared_f64_simd(simd: Arch, input: &[f64], output: &mut [f64]) {
    simd.dispatch(|| {
        let (input_head, input_tail) = pulp::as_arrays::<4, _>(input);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..input_head.len() {
            output_head[i][0] += input_head[i][0] * input_head[i][0];
            output_head[i][1] += input_head[i][1] * input_head[i][1];
            output_head[i][2] += input_head[i][2] * input_head[i][2];
            output_head[i][3] += input_head[i][3] * input_head[i][3];
        }

        for i in 0..input_tail.len() {
            output_tail[i] += input_tail[i] * input_tail[i];
        }
    });
}

/// Compute magnitudes from complex data stored as separate real/imaginary arrays
/// magnitude[i] = sqrt(real[i]^2 + imag[i]^2)
#[inline]
pub fn compute_magnitudes<T: Float + 'static>(real: &[T], imag: &[T], output: &mut [T]) {
    debug_assert_eq!(real.len(), imag.len());
    debug_assert_eq!(real.len(), output.len());

    #[cfg(feature = "simd")]
    {
        let simd = pulp::Arch::new();
        match (
            std::any::TypeId::of::<T>(),
            std::any::TypeId::of::<f32>(),
            std::any::TypeId::of::<f64>(),
        ) {
            (t, f32_id, _) if t == f32_id => {
                compute_magnitudes_f32_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f32]>(real) },
                    unsafe { std::mem::transmute::<&[T], &[f32]>(imag) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f32]>(output) },
                );
                return;
            }
            (t, _, f64_id) if t == f64_id => {
                compute_magnitudes_f64_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f64]>(real) },
                    unsafe { std::mem::transmute::<&[T], &[f64]>(imag) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f64]>(output) },
                );
                return;
            }
            _ => {}
        }
    }

    // Fallback to scalar
    for i in 0..real.len() {
        output[i] = (real[i] * real[i] + imag[i] * imag[i]).sqrt();
    }
}

#[cfg(feature = "simd")]
fn compute_magnitudes_f32_simd(simd: Arch, real: &[f32], imag: &[f32], output: &mut [f32]) {
    simd.dispatch(|| {
        let (real_head, real_tail) = pulp::as_arrays::<4, _>(real);
        let (imag_head, imag_tail) = pulp::as_arrays::<4, _>(imag);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..real_head.len() {
            for j in 0..4 {
                output_head[i][j] =
                    (real_head[i][j] * real_head[i][j] + imag_head[i][j] * imag_head[i][j]).sqrt();
            }
        }

        for i in 0..real_tail.len() {
            output_tail[i] = (real_tail[i] * real_tail[i] + imag_tail[i] * imag_tail[i]).sqrt();
        }
    });
}

#[cfg(feature = "simd")]
fn compute_magnitudes_f64_simd(simd: Arch, real: &[f64], imag: &[f64], output: &mut [f64]) {
    simd.dispatch(|| {
        let (real_head, real_tail) = pulp::as_arrays::<4, _>(real);
        let (imag_head, imag_tail) = pulp::as_arrays::<4, _>(imag);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..real_head.len() {
            for j in 0..4 {
                output_head[i][j] =
                    (real_head[i][j] * real_head[i][j] + imag_head[i][j] * imag_head[i][j]).sqrt();
            }
        }

        for i in 0..real_tail.len() {
            output_tail[i] = (real_tail[i] * real_tail[i] + imag_tail[i] * imag_tail[i]).sqrt();
        }
    });
}

/// Scale all values in a slice by a constant factor
/// output[i] = input[i] * scale
#[inline]
pub fn scale_slice<T: Float + 'static>(input: &[T], scale: T, output: &mut [T]) {
    debug_assert_eq!(input.len(), output.len());

    #[cfg(feature = "simd")]
    {
        let simd = pulp::Arch::new();
        match (
            std::any::TypeId::of::<T>(),
            std::any::TypeId::of::<f32>(),
            std::any::TypeId::of::<f64>(),
        ) {
            (t, f32_id, _) if t == f32_id => {
                scale_slice_f32_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f32]>(input) },
                    unsafe { std::mem::transmute_copy::<T, f32>(&scale) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f32]>(output) },
                );
                return;
            }
            (t, _, f64_id) if t == f64_id => {
                scale_slice_f64_simd(
                    simd,
                    unsafe { std::mem::transmute::<&[T], &[f64]>(input) },
                    unsafe { std::mem::transmute_copy::<T, f64>(&scale) },
                    unsafe { std::mem::transmute::<&mut [T], &mut [f64]>(output) },
                );
                return;
            }
            _ => {}
        }
    }

    // Fallback to scalar
    for i in 0..input.len() {
        output[i] = input[i] * scale;
    }
}

#[cfg(feature = "simd")]
fn scale_slice_f32_simd(simd: Arch, input: &[f32], scale: f32, output: &mut [f32]) {
    simd.dispatch(|| {
        let (input_head, input_tail) = pulp::as_arrays::<4, _>(input);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..input_head.len() {
            output_head[i] = [
                input_head[i][0] * scale,
                input_head[i][1] * scale,
                input_head[i][2] * scale,
                input_head[i][3] * scale,
            ];
        }

        for i in 0..input_tail.len() {
            output_tail[i] = input_tail[i] * scale;
        }
    });
}

#[cfg(feature = "simd")]
fn scale_slice_f64_simd(simd: Arch, input: &[f64], scale: f64, output: &mut [f64]) {
    simd.dispatch(|| {
        let (input_head, input_tail) = pulp::as_arrays::<4, _>(input);
        let (output_head, output_tail) = pulp::as_arrays_mut::<4, _>(output);

        for i in 0..input_head.len() {
            output_head[i] = [
                input_head[i][0] * scale,
                input_head[i][1] * scale,
                input_head[i][2] * scale,
                input_head[i][3] * scale,
            ];
        }

        for i in 0..input_tail.len() {
            output_tail[i] = input_tail[i] * scale;
        }
    });
}

/// Divide slice by another slice element-wise (for normalization)
/// output[i] = numerator[i] / denominator[i]
/// Values where denominator < threshold are set to zero
#[inline]
pub fn divide_with_threshold<T: Float>(
    numerator: &[T],
    denominator: &[T],
    threshold: T,
    output: &mut [T],
) {
    debug_assert_eq!(numerator.len(), denominator.len());
    debug_assert_eq!(numerator.len(), output.len());

    // Scalar implementation (SIMD not as beneficial due to conditional)
    for i in 0..numerator.len() {
        output[i] = if denominator[i] > threshold {
            numerator[i] / denominator[i]
        } else {
            T::zero()
        };
    }
}
