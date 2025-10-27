use stft_rs::prelude::*;

fn main() {
    println!("Spectral Operations Example\n");

    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config.clone());

    // Generate test signal: 440 Hz sine wave
    let sample_rate = 44100;
    let duration = 1.0;
    let samples = (sample_rate as f32 * duration) as usize;
    let signal: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    println!("Processing {} samples @ {} Hz", signal.len(), sample_rate);
    println!("Fundamental frequency: 440 Hz\n");

    // Process to get spectrum
    let spectrum = stft.process(&signal);
    println!("Generated {} STFT frames", spectrum.num_frames);
    println!("Frequency bins: {}\n", spectrum.freq_bins);

    // Example 1: Get magnitude and phase
    println!("=== Example 1: Magnitude and Phase Analysis ===");
    let frame_idx = spectrum.num_frames / 2; // Middle frame

    // Find peak frequency
    let magnitudes = spectrum.frame_magnitudes(frame_idx);
    let (peak_bin, peak_mag) = magnitudes
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .unwrap();

    let freq_resolution = sample_rate as f32 / config.fft_size as f32;
    let peak_freq = peak_bin as f32 * freq_resolution;

    println!(
        "Frame {}: Peak at bin {} ({:.1} Hz)",
        frame_idx, peak_bin, peak_freq
    );
    println!("Peak magnitude: {:.4}", peak_mag);
    println!(
        "Peak phase: {:.4} radians\n",
        spectrum.phase(frame_idx, peak_bin)
    );

    // Example 2: Magnitude scaling (volume control)
    println!("=== Example 2: Apply Gain (Volume Control) ===");
    let mut spectrum_gained = spectrum.clone();
    let gain = 0.5; // Reduce volume by half

    for frame in 0..spectrum_gained.num_frames {
        for bin in 0..spectrum_gained.freq_bins {
            let mag = spectrum_gained.magnitude(frame, bin);
            let phase = spectrum_gained.phase(frame, bin);
            spectrum_gained.set_magnitude_phase(frame, bin, mag * gain, phase);
        }
    }

    let gained_signal = istft.process(&spectrum_gained);
    let original_rms = rms(&signal);
    let gained_rms = rms(&gained_signal);
    println!("Original RMS: {:.4}", original_rms);
    println!(
        "Gained RMS: {:.4} (expected {:.4})\n",
        gained_rms,
        original_rms * gain
    );

    // Example 3: High-pass filter (zero low frequencies)
    println!("=== Example 3: High-Pass Filter ===");
    let mut spectrum_hp = spectrum.clone();
    let cutoff_freq = 300.0; // Hz
    let cutoff_bin = (cutoff_freq / freq_resolution) as usize;

    spectrum_hp.zero_bins(0..cutoff_bin);
    println!(
        "Zeroed bins 0..{} (0 Hz - {:.1} Hz)",
        cutoff_bin, cutoff_freq
    );

    let hp_signal = istft.process(&spectrum_hp);
    println!("Filtered signal length: {} samples\n", hp_signal.len());

    // Example 4: Low-pass filter (attenuate high frequencies)
    println!("=== Example 4: Low-Pass Filter ===");
    let mut spectrum_lp = spectrum.clone();
    let lp_cutoff = 1000.0; // Hz
    let lp_cutoff_bin = (lp_cutoff / freq_resolution) as usize;

    spectrum_lp.zero_bins(lp_cutoff_bin..spectrum_lp.freq_bins);
    println!(
        "Zeroed bins {}..{} ({:.1} Hz - Nyquist)",
        lp_cutoff_bin, spectrum_lp.freq_bins, lp_cutoff
    );

    let lp_signal = istft.process(&spectrum_lp);
    println!("Filtered signal length: {} samples\n", lp_signal.len());

    // Example 5: Band-pass filter using apply_gain
    println!("=== Example 5: Band-Pass Filter (300 Hz - 1000 Hz) ===");
    let mut spectrum_bp = spectrum.clone();
    let low_cutoff = 300.0;
    let high_cutoff = 1000.0;
    let low_bin = (low_cutoff / freq_resolution) as usize;
    let high_bin = (high_cutoff / freq_resolution) as usize;

    // Zero below low cutoff
    spectrum_bp.zero_bins(0..low_bin);
    // Zero above high cutoff
    spectrum_bp.zero_bins(high_bin..spectrum_bp.freq_bins);

    println!(
        "Pass band: bins {}..{} ({:.1} Hz - {:.1} Hz)",
        low_bin, high_bin, low_cutoff, high_cutoff
    );

    let bp_signal = istft.process(&spectrum_bp);
    println!("Filtered signal length: {} samples\n", bp_signal.len());

    // Example 6: Custom processing with apply()
    println!("=== Example 6: Custom Processing (Phase Randomization) ===");
    let mut spectrum_phase = spectrum.clone();

    // Randomize phase while keeping magnitude (creates noise-like sound)
    use std::f32::consts::PI;
    let mut phase_counter = 0.0;
    spectrum_phase.apply(|_frame, bin, c| {
        if bin == 0 {
            c // Keep DC component
        } else {
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            phase_counter += 0.1;
            let random_phase = (phase_counter * 7.3_f32).sin() * PI;
            rustfft::num_complex::Complex::new(mag * random_phase.cos(), mag * random_phase.sin())
        }
    });

    let phase_signal = istft.process(&spectrum_phase);
    println!(
        "Phase randomized signal length: {} samples",
        phase_signal.len()
    );
    println!("(Magnitude preserved, but phase randomized)\n");

    // Example 7: Using SpectrumFrame directly
    println!("=== Example 7: SpectrumFrame Operations ===");
    let mut streaming_stft = StreamingStftF32::new(config.clone());
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&signal, pad_amount, PadMode::Reflect);

    let frames = streaming_stft.push_samples(&padded);
    if let Some(frame) = frames.first() {
        let mags = frame.magnitudes();
        let phases = frame.phases();

        println!("First frame:");
        println!("  Frequency bins: {}", frame.freq_bins);
        println!(
            "  Max magnitude: {:.4}",
            mags.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
        );
        println!(
            "  Mean magnitude: {:.4}",
            mags.iter().sum::<f32>() / mags.len() as f32
        );

        // Reconstruct from magnitude/phase
        let reconstructed_frame = SpectrumFrameF32::from_magnitude_phase(&mags, &phases);
        println!(
            "  Reconstructed frame bins: {}",
            reconstructed_frame.freq_bins
        );
    }

    println!("\n=== Summary ===");
    println!("Spectral operations available:");
    println!("  - magnitude(frame, bin) / phase(frame, bin)");
    println!("  - set_magnitude_phase(frame, bin, mag, phase)");
    println!("  - frame_magnitudes(frame) / frame_phases(frame)");
    println!("  - apply_gain(bin_range, gain)");
    println!("  - zero_bins(bin_range)");
    println!("  - apply(closure) for custom processing");
    println!("  - SpectrumFrame::from_magnitude_phase()");
}

fn rms(signal: &[f32]) -> f32 {
    let sum_squares: f32 = signal.iter().map(|x| x * x).sum();
    (sum_squares / signal.len() as f32).sqrt()
}
