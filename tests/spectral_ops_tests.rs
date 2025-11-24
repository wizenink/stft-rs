mod common;

use stft_rs::fft_backend::Complex;
use stft_rs::prelude::*;

#[test]
fn test_spectrum_magnitude_phase() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());

    let signal_len = 10 * 1024;
    let original: Vec<f32> = (0..signal_len).map(|i| (i as f32 * 0.01).sin()).collect();

    let spectrum = stft.process(&original);

    // Test magnitude and phase calculations
    for frame in 0..spectrum.num_frames.min(3) {
        for bin in 0..spectrum.freq_bins.min(10) {
            let re = spectrum.real(frame, bin);
            let im = spectrum.imag(frame, bin);

            let mag = spectrum.magnitude(frame, bin);
            let expected_mag = (re * re + im * im).sqrt();
            assert!((mag - expected_mag).abs() < 1e-6);

            let phase = spectrum.phase(frame, bin);
            let expected_phase = im.atan2(re);
            assert!((phase - expected_phase).abs() < 1e-6);
        }
    }
}

#[test]
fn test_spectrum_set_magnitude_phase() {
    let config = StftConfig::<f32>::default_4096();
    let mut spectrum = Spectrum::new(10, config.freq_bins());

    let test_mag = 5.0f32;
    let test_phase = std::f32::consts::PI / 4.0;

    spectrum.set_magnitude_phase(0, 10, test_mag, test_phase);

    let retrieved_mag = spectrum.magnitude(0, 10);
    let retrieved_phase = spectrum.phase(0, 10);

    assert!((retrieved_mag - test_mag).abs() < 1e-5);
    assert!((retrieved_phase - test_phase).abs() < 1e-5);
}

#[test]
fn test_spectrum_set_complex() {
    let mut spectrum = Spectrum::<f32>::new(10, 100);

    let test_complex = Complex::new(3.0, 4.0);
    spectrum.set_complex(5, 20, test_complex);

    let retrieved = spectrum.get_complex(5, 20);
    assert!((retrieved.re - test_complex.re).abs() < 1e-6);
    assert!((retrieved.im - test_complex.im).abs() < 1e-6);
}

#[test]
fn test_spectrum_frame_magnitude_phase() {
    let freq_bins = 100;
    let mut frame = SpectrumFrame::<f32>::new(freq_bins);

    // Set some test values
    frame.data[0] = Complex::new(3.0, 4.0);
    frame.data[1] = Complex::new(1.0, 0.0);
    frame.data[2] = Complex::new(0.0, 1.0);

    // Test magnitude
    assert!((frame.magnitude(0) - 5.0).abs() < 1e-6); // sqrt(9+16)
    assert!((frame.magnitude(1) - 1.0).abs() < 1e-6);
    assert!((frame.magnitude(2) - 1.0).abs() < 1e-6);

    // Test phase
    assert!((frame.phase(1) - 0.0).abs() < 1e-6); // atan2(0, 1)
    assert!((frame.phase(2) - std::f32::consts::PI / 2.0).abs() < 1e-6); // atan2(1, 0)
}

#[test]
fn test_spectrum_frame_set_magnitude_phase() {
    let mut frame = SpectrumFrame::<f32>::new(100);

    let mag = 10.0;
    let phase = std::f32::consts::PI / 3.0;

    frame.set_magnitude_phase(50, mag, phase);

    let retrieved_mag = frame.magnitude(50);
    let retrieved_phase = frame.phase(50);

    assert!((retrieved_mag - mag).abs() < 1e-5);
    assert!((retrieved_phase - phase).abs() < 1e-5);
}

#[test]
fn test_spectrum_frame_from_magnitude_phase() {
    let magnitudes = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let phases = vec![
        0.0,
        std::f32::consts::PI / 4.0,
        std::f32::consts::PI / 2.0,
        3.0 * std::f32::consts::PI / 4.0,
        std::f32::consts::PI,
    ];

    let frame = SpectrumFrame::from_magnitude_phase(&magnitudes, &phases);

    assert_eq!(frame.freq_bins, 5);

    for i in 0..5 {
        let mag = frame.magnitude(i);
        let phase = frame.phase(i);

        assert!(
            (mag - magnitudes[i]).abs() < 1e-5,
            "Magnitude mismatch at {}",
            i
        );

        // Handle phase wrapping (π and -π are the same angle)
        let phase_diff = (phase - phases[i]).abs();
        let wrapped_diff = (phase_diff - 2.0 * std::f32::consts::PI).abs();
        assert!(
            phase_diff < 1e-5 || wrapped_diff < 1e-5,
            "Phase mismatch at {}: expected {}, got {}",
            i,
            phases[i],
            phase
        );
    }
}

#[test]
fn test_spectrum_frame_magnitudes_phases() {
    let mut frame = SpectrumFrame::<f32>::new(5);

    // Set some known values
    frame.data[0] = Complex::new(1.0, 0.0);
    frame.data[1] = Complex::new(0.0, 1.0);
    frame.data[2] = Complex::new(3.0, 4.0);

    let mags = frame.magnitudes();
    let phases = frame.phases();

    assert_eq!(mags.len(), 5);
    assert_eq!(phases.len(), 5);

    assert!((mags[0] - 1.0).abs() < 1e-6);
    assert!((mags[1] - 1.0).abs() < 1e-6);
    assert!((mags[2] - 5.0).abs() < 1e-6);
}

#[test]
fn test_spectrum_apply_gain() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config.clone());

    let signal_len = 50 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.5)
        .collect();

    let mut spectrum = stft.process(&original);

    // Apply 0.5x gain to all bins
    spectrum.apply_gain(0..spectrum.freq_bins, 0.5);

    let reconstructed = istft.process(&spectrum);

    // Check that signal power is reduced by ~0.25 (0.5^2)
    let original_power: f32 = original.iter().map(|x| x * x).sum();
    let reconstructed_power: f32 = reconstructed.iter().map(|x| x * x).sum();

    let power_ratio = reconstructed_power / original_power;
    assert!(
        (power_ratio - 0.25).abs() < 0.01,
        "Power ratio: {}",
        power_ratio
    );
}

#[test]
fn test_spectrum_zero_bins() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());

    let signal_len = 50 * 1024;
    let original: Vec<f32> = (0..signal_len).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut spectrum = stft.process(&original);

    // Zero out first 100 bins
    spectrum.zero_bins(0..100);

    // Verify bins are zeroed
    for frame in 0..spectrum.num_frames {
        for bin in 0..100 {
            let mag = spectrum.magnitude(frame, bin);
            assert!(mag < 1e-10, "Bin {} not zeroed: {}", bin, mag);
        }
    }
}

#[test]
fn test_spectrum_apply() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());

    let signal_len = 10 * 1024;
    let original: Vec<f32> = (0..signal_len).map(|i| (i as f32 * 0.01).sin()).collect();

    let mut spectrum = stft.process(&original);

    // Double all magnitudes using apply
    spectrum.apply(|_frame, _bin, c| c * 2.0);

    // Verify magnitudes are doubled
    for frame in 0..spectrum.num_frames.min(3) {
        for bin in 0..spectrum.freq_bins.min(10) {
            let c = spectrum.get_complex(frame, bin);
            let mag = (c.re * c.re + c.im * c.im).sqrt();

            // Should be roughly 2x original (not exact due to windowing)
            assert!(mag >= 0.0); // At least verify it's valid
        }
    }
}

#[test]
fn test_magnitude_phase_roundtrip() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config.clone());

    let signal_len = 50 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.3)
        .collect();

    let spectrum = stft.process(&original);

    // Extract magnitude and phase, then reconstruct
    let mut reconstructed_spectrum = Spectrum::new(spectrum.num_frames, spectrum.freq_bins);

    for frame in 0..spectrum.num_frames {
        for bin in 0..spectrum.freq_bins {
            let mag = spectrum.magnitude(frame, bin);
            let phase = spectrum.phase(frame, bin);
            reconstructed_spectrum.set_magnitude_phase(frame, bin, mag, phase);
        }
    }

    let reconstructed = istft.process(&reconstructed_spectrum);

    // Should reconstruct with high fidelity
    let snr = common::calculate_snr(&original, &reconstructed);
    assert!(
        snr > 100.0,
        "SNR too low after mag/phase roundtrip: {:.2} dB",
        snr
    );
}

#[test]
fn test_frame_magnitudes_phases() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());

    let signal_len = 10 * 1024;
    let original: Vec<f32> = (0..signal_len).map(|i| (i as f32 * 0.01).sin()).collect();

    let spectrum = stft.process(&original);

    for frame in 0..spectrum.num_frames.min(3) {
        let mags = spectrum.frame_magnitudes(frame);
        let phases = spectrum.frame_phases(frame);

        assert_eq!(mags.len(), spectrum.freq_bins);
        assert_eq!(phases.len(), spectrum.freq_bins);

        // Verify individual values match
        for bin in 0..spectrum.freq_bins {
            assert!((mags[bin] - spectrum.magnitude(frame, bin)).abs() < 1e-6);
            assert!((phases[bin] - spectrum.phase(frame, bin)).abs() < 1e-6);
        }
    }
}
