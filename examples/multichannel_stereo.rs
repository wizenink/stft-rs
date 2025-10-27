//! Example: Basic stereo multi-channel processing
//!
//! This example demonstrates:
//! - Processing stereo audio with multi-channel API
//! - Both planar and interleaved formats
//! - Applying different processing to left and right channels

use stft_rs::prelude::*;

/// Generate a test sine wave
fn generate_tone(freq: f32, duration_samples: usize, sample_rate: f32) -> Vec<f32> {
    (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

fn main() {
    println!("=== Multi-Channel Stereo Processing Example ===\n");

    // Configuration
    let sample_rate = 44100.0;
    let duration = 1.0; // 1 second
    let samples = (sample_rate * duration) as usize;

    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // === Example 1: Planar Format (Separate Channels) ===
    println!("Example 1: Planar format processing");
    println!("-----------------------------------");

    let left = generate_tone(220.0, samples, sample_rate); // A3
    let right = generate_tone(440.0, samples, sample_rate); // A4
    let channels = vec![left.clone(), right.clone()];

    println!("Input: Left channel = 220 Hz, Right channel = 440 Hz");
    println!("Processing {} samples per channel...", samples);

    // Process both channels
    let spectra = stft.process_multichannel(&channels);
    println!("Created {} spectra (one per channel)", spectra.len());
    println!(
        "Each spectrum has {} frames × {} frequency bins",
        spectra[0].num_frames, spectra[0].freq_bins
    );

    // Apply different processing to each channel
    let mut left_spectrum = spectra[0].clone();
    let mut right_spectrum = spectra[1].clone();

    // High-pass filter on left channel (remove frequencies below 200 Hz)
    let freq_bin_200hz = (200.0 * left_spectrum.freq_bins as f32 / (sample_rate / 2.0)) as usize;
    left_spectrum.zero_bins(0..freq_bin_200hz);
    println!("Applied high-pass filter to left channel (>200 Hz)");

    // Low-pass filter on right channel (remove frequencies above 500 Hz)
    let freq_bin_500hz = (500.0 * right_spectrum.freq_bins as f32 / (sample_rate / 2.0)) as usize;
    right_spectrum.zero_bins(freq_bin_500hz..right_spectrum.freq_bins);
    println!("Applied low-pass filter to right channel (<500 Hz)");

    // Reconstruct
    let reconstructed = istft.process_multichannel(&[left_spectrum, right_spectrum]);
    println!("Reconstructed {} channels", reconstructed.len());
    println!(
        "Output length: {} samples per channel\n",
        reconstructed[0].len()
    );

    // === Example 2: Interleaved Format ===
    println!("Example 2: Interleaved format processing");
    println!("----------------------------------------");

    // Create interleaved stereo: LRLRLR...
    let interleaved = interleave(&channels);
    println!(
        "Created interleaved buffer with {} samples total",
        interleaved.len()
    );

    // Process interleaved
    let spectra = stft.process_interleaved(&interleaved, 2);
    println!("Processed interleaved audio into {} spectra", spectra.len());

    // Apply gain to left channel, keep right channel unchanged
    let mut left_spectrum = spectra[0].clone();
    let right_spectrum = spectra[1].clone();

    left_spectrum.apply_gain(0..left_spectrum.freq_bins, 0.5); // Reduce left by 50%
    println!("Reduced left channel gain by 50%");

    // Reconstruct to interleaved format
    let output = istft.process_multichannel_interleaved(&[left_spectrum, right_spectrum]);
    println!(
        "Reconstructed interleaved buffer with {} samples\n",
        output.len()
    );

    // === Example 3: Channel Swapping ===
    println!("Example 3: Swap left and right channels");
    println!("---------------------------------------");

    let spectra = stft.process_multichannel(&channels);

    // Swap channels by reversing the order
    let _swapped = istft.process_multichannel(&[spectra[1].clone(), spectra[0].clone()]);
    println!("Swapped channels: left ↔ right");
    println!("Left channel now has {} Hz tone", 440.0);
    println!("Right channel now has {} Hz tone\n", 220.0);

    // === Example 4: Mono Mix ===
    println!("Example 4: Mix stereo to mono");
    println!("-----------------------------");

    let spectra = stft.process_multichannel(&channels);

    // Average left and right spectra
    let mut mono_spectrum = spectra[0].clone();
    for frame in 0..mono_spectrum.num_frames {
        for bin in 0..mono_spectrum.freq_bins {
            let left = spectra[0].get_complex(frame, bin);
            let right = spectra[1].get_complex(frame, bin);
            let avg = (left + right) * 0.5;
            mono_spectrum.set_complex(frame, bin, avg);
        }
    }

    let mono = istft.process(&mono_spectrum);
    println!("Mixed stereo to mono: {} samples", mono.len());
    println!("Mono contains both 220 Hz and 440 Hz tones\n");

    // === Summary ===
    println!("Summary");
    println!("-------");
    println!("✓ Processed stereo audio in planar format");
    println!("✓ Processed stereo audio in interleaved format");
    println!("✓ Applied per-channel filtering");
    println!("✓ Swapped channels");
    println!("✓ Mixed stereo to mono");
    println!("\nMulti-channel processing makes working with stereo audio efficient and flexible!");
}
