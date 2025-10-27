//! Example: Mid/Side Stereo Width Processing
//!
//! This example demonstrates:
//! - Converting stereo L/R to Mid/Side format
//! - Processing mid and side channels independently
//! - Adjusting stereo width by manipulating the side channel
//! - Converting back to L/R format

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

/// Convert Left/Right to Mid/Side
fn lr_to_ms(left: &[f32], right: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(left.len(), right.len());
    let mid: Vec<f32> = left.iter().zip(right).map(|(l, r)| (l + r) / 2.0).collect();
    let side: Vec<f32> = left.iter().zip(right).map(|(l, r)| (l - r) / 2.0).collect();
    (mid, side)
}

/// Convert Mid/Side back to Left/Right
fn ms_to_lr(mid: &[f32], side: &[f32]) -> (Vec<f32>, Vec<f32>) {
    assert_eq!(mid.len(), side.len());
    let left: Vec<f32> = mid.iter().zip(side).map(|(m, s)| m + s).collect();
    let right: Vec<f32> = mid.iter().zip(side).map(|(m, s)| m - s).collect();
    (left, right)
}

fn main() {
    println!("=== Mid/Side Stereo Width Processing Example ===\n");

    // Configuration
    let sample_rate = 44100.0;
    let duration = 1.0;
    let samples = (sample_rate * duration) as usize;

    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // === Step 1: Create stereo signal ===
    println!("Step 1: Create stereo test signal");
    println!("----------------------------------");

    // Left: 220 Hz (A3)
    // Right: 440 Hz (A4)
    let left = generate_tone(220.0, samples, sample_rate);
    let right = generate_tone(440.0, samples, sample_rate);

    println!("Created stereo signal:");
    println!("  Left:  220 Hz tone");
    println!("  Right: 440 Hz tone\n");

    // === Step 2: Convert to Mid/Side ===
    println!("Step 2: Convert L/R to Mid/Side");
    println!("--------------------------------");

    let (mid, side) = lr_to_ms(&left, &right);

    println!("Mid/Side encoding:");
    println!("  Mid  = (L + R) / 2  (mono sum - center information)");
    println!("  Side = (L - R) / 2  (stereo difference - width information)");
    println!("Mid contains:  Both 220 Hz and 440 Hz (center)");
    println!("Side contains: Difference between L and R (stereo width)\n");

    // === Step 3: Process Mid and Side separately ===
    println!("Step 3: Process Mid/Side with STFT");
    println!("-----------------------------------");

    let channels = vec![mid, side];
    let spectra = stft.process_multichannel(&channels);

    println!("Created {} spectra (mid and side)", spectra.len());
    println!(
        "Each spectrum: {} frames × {} bins\n",
        spectra[0].num_frames, spectra[0].freq_bins
    );

    // === Example A: Widen Stereo (boost side channel) ===
    println!("Example A: Widen stereo image (boost side by 50%)");
    println!("--------------------------------------------------");

    let mid_spec = spectra[0].clone();
    let mut side_spec = spectra[1].clone();

    // Boost side channel by 50%
    side_spec.apply_gain(0..side_spec.freq_bins, 1.5);

    println!("Applied 1.5× gain to side channel");
    println!("Effect: Stereo image will be 50% wider\n");

    // Reconstruct
    let processed = istft.process_multichannel(&[mid_spec, side_spec]);
    let (left_wide, _right_wide) = ms_to_lr(&processed[0], &processed[1]);

    println!("Reconstructed wider stereo:");
    println!("  Output length: {} samples per channel", left_wide.len());
    println!("  Stereo width: 150% of original\n");

    // === Example B: Narrow Stereo (reduce side channel) ===
    println!("Example B: Narrow stereo image (reduce side by 50%)");
    println!("----------------------------------------------------");

    let mid_spec = spectra[0].clone();
    let mut side_spec = spectra[1].clone();

    // Reduce side channel by 50%
    side_spec.apply_gain(0..side_spec.freq_bins, 0.5);

    println!("Applied 0.5× gain to side channel");
    println!("Effect: Stereo image will be 50% narrower\n");

    let processed = istft.process_multichannel(&[mid_spec, side_spec]);
    let (left_narrow, _right_narrow) = ms_to_lr(&processed[0], &processed[1]);

    println!("Reconstructed narrower stereo:");
    println!("  Output length: {} samples per channel", left_narrow.len());
    println!("  Stereo width: 50% of original\n");

    // === Example C: Mono (remove side completely) ===
    println!("Example C: Convert to mono (zero side channel)");
    println!("-----------------------------------------------");

    let mid_spec = spectra[0].clone();
    let mut side_spec = spectra[1].clone();

    // Zero out side channel completely
    side_spec.zero_bins(0..side_spec.freq_bins);

    println!("Zeroed side channel completely");
    println!("Effect: Result is mono (L = R)\n");

    let processed = istft.process_multichannel(&[mid_spec, side_spec]);
    let (left_mono, _right_mono) = ms_to_lr(&processed[0], &processed[1]);

    println!("Reconstructed mono (from mid channel only):");
    println!("  Output length: {} samples per channel", left_mono.len());
    println!("  Left and right channels are identical\n");

    // === Example D: Frequency-dependent width ===
    println!("Example D: Frequency-dependent stereo width");
    println!("--------------------------------------------");

    let mid_spec = spectra[0].clone();
    let mut side_spec = spectra[1].clone();

    // Widen high frequencies, narrow low frequencies
    let freq_bins = side_spec.freq_bins;
    let crossover_bin = freq_bins / 4; // Crossover at 1/4 of Nyquist

    // Low frequencies: reduce width (0.3×)
    side_spec.apply_gain(0..crossover_bin, 0.3);

    // High frequencies: increase width (1.8×)
    side_spec.apply_gain(crossover_bin..freq_bins, 1.8);

    println!("Applied frequency-dependent gain to side channel:");
    println!("  Low frequencies:  0.3× (narrower)");
    println!("  High frequencies: 1.8× (wider)");
    println!("Effect: Tight bass, wide highs\n");

    let processed = istft.process_multichannel(&[mid_spec, side_spec]);
    let (left_split, _right_split) = ms_to_lr(&processed[0], &processed[1]);

    println!("Reconstructed frequency-split stereo:");
    println!("  Output length: {} samples per channel", left_split.len());
    println!("  Bass is narrower, highs are wider\n");

    // === Summary ===
    println!("Summary");
    println!("-------");
    println!("✓ Converted L/R to Mid/Side format");
    println!("✓ Processed mid and side channels independently");
    println!("✓ Widened stereo (1.5× side gain)");
    println!("✓ Narrowed stereo (0.5× side gain)");
    println!("✓ Created mono (zero side)");
    println!("✓ Applied frequency-dependent width");
    println!("\nMid/Side processing is a powerful technique for stereo manipulation!");
    println!("\nUse cases:");
    println!("  - Stereo width enhancement");
    println!("  - Mono compatibility checking");
    println!("  - Separate processing of center vs. sides");
    println!("  - Mastering and mixing applications");
}
