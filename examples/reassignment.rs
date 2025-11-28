//! Reassignment method example
//!
//! This example demonstrates the reassignment method for sharpening
//! time-frequency representations. The reassignment method redistributes
//! energy to coordinates closer to the true signal support, resulting in
//! sharper spectrograms.
//!
//! This is particularly useful for:
//! - Analyzing signals with sharp transients
//! - Bass-heavy audio where FFT bins represent large perceptual ranges
//! - Signals with closely-spaced harmonics
//!
//! Run with: cargo run --example reassignment --features rustfft-backend

use stft_rs::fft_backend::{FftPlanner, FftPlannerTrait};
use stft_rs::prelude::*;
use std::f64::consts::PI;

fn main() {
    println!("=== Reassignment Method Example ===\n");

    // Create a test signal: two sine waves at different frequencies
    let sample_rate = 44100.0;
    let duration = 1.0; // 1 second
    let num_samples = (sample_rate * duration) as usize;

    // Frequency 1: 440 Hz (A4)
    // Frequency 2: 880 Hz (A5, one octave higher)
    let freq1 = 440.0;
    let freq2 = 880.0;

    println!("Creating test signal:");
    println!("  Sample rate: {} Hz", sample_rate);
    println!("  Duration: {} seconds", duration);
    println!("  Frequency 1: {} Hz (A4)", freq1);
    println!("  Frequency 2: {} Hz (A5)", freq2);
    println!();

    let signal: Vec<f64> = (0..num_samples)
        .map(|i| {
            let t = i as f64 / sample_rate;
            // Sum of two sine waves
            (2.0 * PI * freq1 * t).sin() * 0.5 + (2.0 * PI * freq2 * t).sin() * 0.5
        })
        .collect();

    // STFT configuration
    let stft_config = StftConfig::<f64>::builder()
        .fft_size(4096)
        .hop_size(1024)
        .window(WindowType::Hann)
        .reconstruction_mode(ReconstructionMode::Ola)
        .build()
        .unwrap();

    println!("STFT Configuration:");
    println!("  FFT size: {}", stft_config.fft_size);
    println!("  Hop size: {}", stft_config.hop_size);
    println!("  Window: {:?}", stft_config.window);
    println!();

    // Compute regular STFT
    println!("Computing regular STFT...");
    let stft = BatchStft::new(stft_config.clone());
    let spectrum = stft.process(&signal);

    println!("  Frames: {}", spectrum.num_frames);
    println!("  Frequency bins: {}", spectrum.freq_bins);
    println!();

    // Compute reassigned STFT
    println!("Computing reassigned STFT...");
    let reassign_config = ReassignmentConfig {
        power_threshold: 1e-6,
        clip_to_bounds: true,
    };

    let reassignment = BatchReassignment::new(stft_config.clone(), reassign_config);
    let mut planner = FftPlanner::<f64>::new();
    let reassigned = reassignment.process(&signal, sample_rate, &mut planner);

    println!("  Frames: {}", reassigned.num_frames);
    println!("  Frequency bins: {}", reassigned.freq_bins);
    println!();

    // Find peaks in the reassigned spectrum
    println!("Finding peaks in reassigned spectrum...");

    // Look at a middle frame to avoid edge effects
    let mid_frame = reassigned.num_frames / 2;

    // Find the two strongest peaks
    let mut peaks: Vec<(usize, f64, f64)> = Vec::new();
    for bin in 1..reassigned.freq_bins - 1 {
        let mag = reassigned.magnitude(mid_frame, bin);
        let freq = reassigned.reassigned_freq(mid_frame, bin);

        // Simple peak detection: local maximum
        let mag_prev = reassigned.magnitude(mid_frame, bin - 1);
        let mag_next = reassigned.magnitude(mid_frame, bin + 1);

        if mag > mag_prev && mag > mag_next && mag > 0.1 {
            peaks.push((bin, mag, freq));
        }
    }

    // Sort by magnitude
    peaks.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 peaks found:");
    for (i, (bin, mag, freq)) in peaks.iter().take(5).enumerate() {
        // Calculate the regular bin frequency for comparison
        let bin_freq = *bin as f64 * sample_rate / stft_config.fft_size as f64;

        println!("  Peak {}: ", i + 1);
        println!("    Bin: {}", bin);
        println!("    Magnitude: {:.4}", mag);
        println!("    Bin frequency: {:.2} Hz", bin_freq);
        println!("    Reassigned frequency: {:.2} Hz", freq);

        // Check which target frequency this is closest to
        let error1 = (freq - freq1).abs();
        let error2 = (freq - freq2).abs();

        if error1 < error2 && error1 < 50.0 {
            println!("    → Matches target frequency 1 ({:.2} Hz), error: {:.2} Hz", freq1, error1);
        } else if error2 < 50.0 {
            println!("    → Matches target frequency 2 ({:.2} Hz), error: {:.2} Hz", freq2, error2);
        }
        println!();
    }

    // Render reassigned spectrum back to a regular grid for analysis
    println!("Rendering reassigned spectrum to grid...");
    let rendered = reassigned.render_to_grid();

    // Compare energy concentration
    let regular_peak_energy = find_peak_energy(&spectrum, mid_frame, 5);
    let reassigned_peak_energy = find_peak_energy(&rendered, mid_frame, 5);

    println!("\nEnergy concentration comparison:");
    println!("  Regular STFT peak energy (5 bins): {:.4}", regular_peak_energy);
    println!("  Reassigned STFT peak energy (5 bins): {:.4}", reassigned_peak_energy);
    println!(
        "  Concentration ratio: {:.2}x",
        reassigned_peak_energy / regular_peak_energy
    );

    println!("\n=== Summary ===");
    println!("The reassignment method successfully sharpened the spectrogram,");
    println!("concentrating energy closer to the true signal frequencies.");
    println!("\nThis is particularly useful for:");
    println!("  • Improving time-frequency resolution");
    println!("  • Separating closely-spaced harmonics");
    println!("  • Analyzing transient signals");
    println!("  • Bass-heavy audio with large perceptual ranges per bin");
}

/// Find the sum of the top N peak energies in a frame
fn find_peak_energy(spectrum: &Spectrum<f64>, frame: usize, n: usize) -> f64 {
    let mut magnitudes: Vec<f64> = (0..spectrum.freq_bins)
        .map(|bin| spectrum.magnitude(frame, bin))
        .collect();

    magnitudes.sort_by(|a, b| b.partial_cmp(a).unwrap());

    magnitudes.iter().take(n).map(|m| m * m).sum()
}
