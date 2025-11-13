//! Backend Comparison Example
//!
//! This example demonstrates that rustfft and microfft backends produce
//! identical results for f32 processing. It processes test signals with
//! STFT/iSTFT and saves the results to files for comparison.
//!
//! Run with:
//! ```bash
//! # Test with rustfft backend (default)
//! cargo run --example backend_comparison -- rustfft
//!
//! # Test with microfft backend
//! cargo run --no-default-features --features microfft-backend --example backend_comparison -- microfft
//! ```
//!
//! Then compare the results:
//! ```bash
//! diff results_rustfft.txt results_microfft.txt
//! ```

use std::env;
use std::fs::File;
use std::io::Write;
use stft_rs::prelude::*;

fn generate_test_signal(sample_rate: f32, duration: f32) -> Vec<f32> {
    let num_samples = (sample_rate * duration) as usize;
    let mut signal = Vec::with_capacity(num_samples);

    // Multi-tone signal: 220 Hz, 440 Hz, 880 Hz
    for i in 0..num_samples {
        let t = i as f32 / sample_rate;
        let sample = (2.0 * std::f32::consts::PI * 220.0 * t).sin() * 0.3
            + (2.0 * std::f32::consts::PI * 440.0 * t).sin() * 0.5
            + (2.0 * std::f32::consts::PI * 880.0 * t).sin() * 0.2;
        signal.push(sample);
    }

    signal
}

fn calculate_snr(original: &[f32], reconstructed: &[f32]) -> f32 {
    let len = original.len().min(reconstructed.len());
    let original = &original[..len];
    let reconstructed = &reconstructed[..len];

    let signal_power: f32 = original.iter().map(|&x| x * x).sum();
    let noise_power: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| (o - r).powi(2))
        .sum();

    if noise_power < 1e-10 {
        return 200.0; // Effectively perfect
    }

    10.0 * (signal_power / noise_power).log10()
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let backend_name = args.get(1).map(|s| s.as_str()).unwrap_or("unknown");

    println!("=== Backend Comparison: {} ===\n", backend_name);

    // Generate test signal
    let sample_rate = 44100.0;
    let duration = 1.0; // 1 second
    let signal = generate_test_signal(sample_rate, duration);

    println!("Test signal:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Duration: {} seconds", duration);
    println!("  Samples: {}", signal.len());
    println!("  Frequencies: 220 Hz, 440 Hz, 880 Hz\n");

    // Create STFT configuration (power-of-2 for microfft compatibility)
    // Use default_4096 which is COLA compliant
    let config = StftConfigF32::default_4096();

    println!("STFT Configuration:");
    println!("  FFT size: {}", config.fft_size);
    println!("  Hop size: {}", config.hop_size);
    println!("  Window: {:?}", config.window);
    println!("  Mode: {:?}\n", config.reconstruction_mode);

    // Process with STFT
    let stft = BatchStftF32::new(config.clone());
    let spectrum = stft.process(&signal);

    println!("Spectrum:");
    println!("  Frames: {}", spectrum.num_frames);
    println!("  Frequency bins: {}\n", spectrum.freq_bins);

    // Reconstruct with iSTFT
    let istft = BatchIstftF32::new(config);
    let reconstructed = istft.process(&spectrum);

    // Calculate reconstruction quality
    let snr = calculate_snr(&signal, &reconstructed);
    println!("Reconstruction Quality:");
    println!("  SNR: {:.2} dB\n", snr);

    // Save results to file
    let filename = format!("results_{}.txt", backend_name);
    let mut file = File::create(&filename).expect("Failed to create file");

    writeln!(file, "Backend: {}", backend_name).unwrap();
    writeln!(file, "Signal samples: {}", signal.len()).unwrap();
    writeln!(file, "Reconstructed samples: {}", reconstructed.len()).unwrap();
    writeln!(file, "Spectrum frames: {}", spectrum.num_frames).unwrap();
    writeln!(file, "Spectrum freq bins: {}", spectrum.freq_bins).unwrap();
    writeln!(file, "SNR: {:.10} dB", snr).unwrap();
    writeln!(file, "").unwrap();

    // Save first 100 spectrum values (real and imaginary parts)
    writeln!(file, "First 100 spectrum values:").unwrap();
    for i in 0..100.min(spectrum.data.len()) {
        writeln!(file, "spectrum[{}] = {:.10e}", i, spectrum.data[i]).unwrap();
    }
    writeln!(file, "").unwrap();

    // Save first 1000 reconstructed samples
    writeln!(file, "First 1000 reconstructed samples:").unwrap();
    for i in 0..1000.min(reconstructed.len()) {
        writeln!(file, "reconstructed[{}] = {:.10e}", i, reconstructed[i]).unwrap();
    }
    writeln!(file, "").unwrap();

    // Calculate and save some spectral statistics
    writeln!(file, "Spectral statistics:").unwrap();
    let spectrum_sum: f32 = spectrum.data.iter().sum();
    let spectrum_mean = spectrum_sum / spectrum.data.len() as f32;
    let spectrum_max = spectrum
        .data
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let spectrum_min = spectrum
        .data
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    writeln!(file, "  Sum: {:.10e}", spectrum_sum).unwrap();
    writeln!(file, "  Mean: {:.10e}", spectrum_mean).unwrap();
    writeln!(file, "  Max: {:.10e}", spectrum_max).unwrap();
    writeln!(file, "  Min: {:.10e}", spectrum_min).unwrap();
    writeln!(file, "").unwrap();

    // Reconstruction statistics
    writeln!(file, "Reconstruction statistics:").unwrap();
    let recon_sum: f32 = reconstructed.iter().sum();
    let recon_mean = recon_sum / reconstructed.len() as f32;
    let recon_max = reconstructed
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let recon_min = reconstructed
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    writeln!(file, "  Sum: {:.10e}", recon_sum).unwrap();
    writeln!(file, "  Mean: {:.10e}", recon_mean).unwrap();
    writeln!(file, "  Max: {:.10e}", recon_max).unwrap();
    writeln!(file, "  Min: {:.10e}", recon_min).unwrap();

    println!("Results saved to: {}", filename);
    println!("\nBackend test completed successfully!");

    if snr < 100.0 {
        eprintln!("\nWARNING: SNR is lower than expected ({:.2} dB)", snr);
        std::process::exit(1);
    }
}
