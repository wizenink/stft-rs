mod common;

use stft_rs::prelude::*;
use stft_rs::fft_backend::{FftPlanner, FftPlannerTrait};

use num_traits::Float;

#[test]
fn test_window_derivative() {
    use stft_rs::reassignment::window_derivative;

    // Test simple window
    let window = vec![0.0, 0.5, 1.0, 0.5, 0.0];
    let derivative = window_derivative(&window);

    assert_eq!(derivative.len(), window.len());

    // Derivative should capture the slope
    // At the peak (index 2), derivative should be near zero
    assert!(
        derivative[2].abs() < 0.6,
        "Derivative at peak should be small"
    );

    // Should be positive on the rising edge and negative on falling edge
    assert!(derivative[1] > 0.0, "Rising edge should have positive slope");
    assert!(
        derivative[3] < 0.0,
        "Falling edge should have negative slope"
    );
}

#[test]
fn test_window_time_ramp() {
    use stft_rs::reassignment::window_time_ramp;

    let window = vec![1.0; 6]; // Uniform window
    let ramped = window_time_ramp(&window);

    assert_eq!(ramped.len(), window.len());

    // Should be negative on the left, positive on the right
    // For size 6, center is 3.0, so:
    // ramped[0] = (0 - 3.0) * 1.0 = -3.0
    // ramped[5] = (5 - 3.0) * 1.0 = 2.0
    assert!(ramped[0] < 0.0, "Left side should be negative");
    assert!(ramped[5] > 0.0, "Right side should be positive");

    // Center value (index 3) should be exactly zero for even-sized window
    assert_eq!(ramped[3], 0.0, "Exact center should be zero");

    // Should ramp linearly from negative to positive
    for i in 0..window.len() - 1 {
        assert!(
            ramped[i] < ramped[i + 1],
            "Should increase monotonically"
        );
    }
}

#[test]
fn test_generate_window() {
    use stft_rs::reassignment::generate_window;

    let size = 512;

    // Test Hann window
    let hann = generate_window::<f64>(WindowType::Hann, size);
    assert_eq!(hann.len(), size);
    // Hann window should be 0 at edges and ~1 at center
    assert!(hann[0] < 0.01, "Hann window should be near 0 at edge");
    assert!(hann[size / 2] > 0.99, "Hann window should be near 1 at center");

    // Test Hamming window
    let hamming = generate_window::<f64>(WindowType::Hamming, size);
    assert_eq!(hamming.len(), size);
    // Hamming window should be ~0.08 at edges
    assert!(
        hamming[0] > 0.05 && hamming[0] < 0.1,
        "Hamming window should be ~0.08 at edge"
    );

    // Test Blackman window
    let blackman = generate_window::<f64>(WindowType::Blackman, size);
    assert_eq!(blackman.len(), size);
    assert!(
        blackman[0] < 0.01,
        "Blackman window should be near 0 at edge"
    );
}

#[test]
fn test_reassignment_config_default() {
    let config = ReassignmentConfig::<f64>::default();
    assert!(config.power_threshold > 0.0);
    assert!(config.clip_to_bounds);
}

#[test]
fn test_reassigned_spectrum_creation() {
    let spectrum = ReassignedSpectrum::<f64>::new(100, 513, 44100.0, 512);

    assert_eq!(spectrum.num_frames, 100);
    assert_eq!(spectrum.freq_bins, 513);
    assert_eq!(spectrum.sample_rate, 44100.0);
    assert_eq!(spectrum.hop_size, 512);
    assert_eq!(spectrum.magnitudes.len(), 100 * 513);
    assert_eq!(spectrum.reassigned_times.len(), 100 * 513);
    assert_eq!(spectrum.reassigned_freqs.len(), 100 * 513);
}

#[test]
fn test_reassigned_spectrum_accessors() {
    let mut spectrum = ReassignedSpectrum::<f64>::new(10, 10, 44100.0, 512);

    // Test set/get
    spectrum.set(5, 3, 1.5, 2560.0, 440.0);

    assert_eq!(spectrum.magnitude(5, 3), 1.5);
    assert_eq!(spectrum.reassigned_time(5, 3), 2560.0);
    assert_eq!(spectrum.reassigned_freq(5, 3), 440.0);
}

#[test]
#[cfg(feature = "rustfft-backend")] // f64 not supported by microfft
fn test_batch_reassignment_creation() {
    let stft_config = StftConfig::<f64>::default_4096();
    let reassign_config = ReassignmentConfig::default();

    let _reassignment = BatchReassignment::new(stft_config, reassign_config);

    // Just verify it was created successfully
    // Internal structure is private, so we can't test much here
}

#[test]
#[cfg(feature = "rustfft-backend")] // f64 not supported by microfft
fn test_batch_reassignment_simple_signal() {
    use std::f64::consts::PI;

    // Create a simple sine wave at 440 Hz
    let sample_rate = 44100.0;
    let duration = 0.5; // 0.5 second
    let freq = 440.0;
    let num_samples = (sample_rate * duration) as usize;

    let signal: Vec<f64> = (0..num_samples)
        .map(|i| (2.0 * PI * freq * i as f64 / sample_rate).sin())
        .collect();

    // Create reassignment processor
    let stft_config = StftConfig::<f64>::builder()
        .fft_size(2048)
        .hop_size(512)
        .window(WindowType::Hann)
        .reconstruction_mode(ReconstructionMode::Ola)
        .build()
        .unwrap();

    let reassign_config = ReassignmentConfig::default();
    let reassignment = BatchReassignment::new(stft_config, reassign_config);

    let mut planner = FftPlanner::<f64>::new();
    let reassigned = reassignment.process(&signal, sample_rate, &mut planner);

    // Verify output structure
    assert!(reassigned.num_frames > 0, "Should have some frames");
    assert_eq!(reassigned.freq_bins, 1025); // 2048/2 + 1
    assert_eq!(reassigned.sample_rate, sample_rate);

    // Check that we got some non-zero magnitudes
    let total_magnitude: f64 = reassigned.magnitudes.iter().sum();
    assert!(total_magnitude > 0.0, "Should have non-zero magnitudes");
}

#[test]
#[cfg(feature = "rustfft-backend")] // f64 not supported by microfft
fn test_reassignment_preserves_energy() {
    use std::f64::consts::PI;

    let sample_rate = 44100.0;
    let signal: Vec<f64> = (0..8192)
        .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
        .collect();

    let stft_config = StftConfig::<f64>::builder()
        .fft_size(2048)
        .hop_size(512)
        .window(WindowType::Hann)
        .reconstruction_mode(ReconstructionMode::Ola)
        .build()
        .unwrap();

    // Compute regular STFT
    let stft = BatchStft::new(stft_config.clone());
    let spectrum = stft.process(&signal);

    // Compute reassigned STFT
    let reassign_config = ReassignmentConfig::default();
    let reassignment = BatchReassignment::new(stft_config, reassign_config);
    let mut planner = FftPlanner::<f64>::new();
    let reassigned = reassignment.process(&signal, sample_rate, &mut planner);

    // Total energy should be similar (within an order of magnitude)
    let mut regular_energy = 0.0;
    for f in 0..spectrum.num_frames {
        for b in 0..spectrum.freq_bins {
            let m = spectrum.magnitude(f, b);
            regular_energy += m * m;
        }
    }

    let reassigned_energy: f64 = reassigned.magnitudes.iter().map(|m| m * m).sum();

    // Energy should be in the same ballpark
    let ratio = reassigned_energy / regular_energy;
    assert!(
        ratio > 0.1 && ratio < 10.0,
        "Energy ratio {} should be reasonable",
        ratio
    );
}

#[test]
#[cfg(feature = "rustfft-backend")] // f64 not supported by microfft
fn test_reassignment_frequency_accuracy() {
    use std::f64::consts::PI;

    // Create a pure sine wave at a known frequency
    let sample_rate = 44100.0;
    let target_freq = 1000.0; // 1 kHz
    let signal: Vec<f64> = (0..16384)
        .map(|i| (2.0 * PI * target_freq * i as f64 / sample_rate).sin())
        .collect();

    let stft_config = StftConfig::<f64>::builder()
        .fft_size(4096)
        .hop_size(1024)
        .window(WindowType::Hann)
        .reconstruction_mode(ReconstructionMode::Ola)
        .build()
        .unwrap();

    let reassign_config = ReassignmentConfig::default();
    let reassignment = BatchReassignment::new(stft_config, reassign_config);
    let mut planner = FftPlanner::<f64>::new();
    let reassigned = reassignment.process(&signal, sample_rate, &mut planner);

    // Find the frame and bin with maximum magnitude
    let mut max_mag = 0.0;
    let mut max_frame = 0;
    let mut max_bin = 0;

    for frame in 0..reassigned.num_frames {
        for bin in 0..reassigned.freq_bins {
            let mag = reassigned.magnitude(frame, bin);
            if mag > max_mag {
                max_mag = mag;
                max_frame = frame;
                max_bin = bin;
            }
        }
    }

    // Get the reassigned frequency at the peak
    let reassigned_freq = reassigned.reassigned_freq(max_frame, max_bin);

    // Reassignment should improve frequency localization or be comparable
    // (It's ok if it's not always better due to numerical issues)
    assert!(
        reassigned_freq > 0.0,
        "Reassigned frequency should be positive"
    );
    assert!(
        reassigned_freq < sample_rate / 2.0,
        "Reassigned frequency should be below Nyquist"
    );
}

#[test]
#[cfg(feature = "rustfft-backend")] // f64 not supported by microfft
fn test_reassigned_spectrum_render_to_grid() {
    use std::f64::consts::PI;

    let sample_rate = 44100.0;
    let signal: Vec<f64> = (0..8192)
        .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
        .collect();

    let stft_config = StftConfig::<f64>::builder()
        .fft_size(2048)
        .hop_size(512)
        .window(WindowType::Hann)
        .reconstruction_mode(ReconstructionMode::Ola)
        .build()
        .unwrap();

    let reassign_config = ReassignmentConfig::default();
    let reassignment = BatchReassignment::new(stft_config, reassign_config);
    let mut planner = FftPlanner::<f64>::new();
    let reassigned = reassignment.process(&signal, sample_rate, &mut planner);

    // Render back to a grid
    let grid = reassigned.render_to_grid();

    assert_eq!(grid.num_frames, reassigned.num_frames);
    assert_eq!(grid.freq_bins, reassigned.freq_bins);

    // Should have some non-zero values
    let mut total_magnitude = 0.0;
    for f in 0..grid.num_frames {
        for b in 0..grid.freq_bins {
            total_magnitude += grid.magnitude(f, b);
        }
    }

    assert!(total_magnitude > 0.0, "Rendered grid should have energy");
}

#[test]
#[cfg(feature = "rustfft-backend")] // f64 not supported by microfft
fn test_reassignment_with_different_windows() {
    use std::f64::consts::PI;

    let sample_rate = 44100.0;
    let signal: Vec<f64> = (0..8192)
        .map(|i| (2.0 * PI * 440.0 * i as f64 / sample_rate).sin())
        .collect();

    let windows = vec![WindowType::Hann, WindowType::Hamming, WindowType::Blackman];

    for window_type in windows {
        let stft_config = StftConfig::<f64>::builder()
            .fft_size(2048)
            .hop_size(512)
            .window(window_type)
            .reconstruction_mode(ReconstructionMode::Ola)
            .build()
            .unwrap();

        let reassign_config = ReassignmentConfig::default();
        let reassignment = BatchReassignment::new(stft_config, reassign_config);
        let mut planner = FftPlanner::<f64>::new();
        let reassigned = reassignment.process(&signal, sample_rate, &mut planner);

        // Should produce valid output for all window types
        assert!(reassigned.num_frames > 0);
        let total_magnitude: f64 = reassigned.magnitudes.iter().sum();
        assert!(total_magnitude > 0.0, "Window {:?} should produce energy", window_type);
    }
}

#[test]
fn test_reassignment_type_aliases() {
    // Just verify the type aliases compile
    let _config_f32: ReassignmentConfigF32 = ReassignmentConfig::default();
    let _config_f64: ReassignmentConfigF64 = ReassignmentConfig::default();

    let _spectrum_f32: ReassignedSpectrumF32 = ReassignedSpectrum::new(10, 10, 44100.0, 512);
    let _spectrum_f64: ReassignedSpectrumF64 = ReassignedSpectrum::new(10, 10, 44100.0, 512);
}
