use stft_rs::prelude::*;

fn main() {
    println!("Advanced Streaming with Buffer Reuse\n");

    let config = StftConfig::<f32>::default_4096();
    println!("Configuration:");
    println!(
        "  FFT: {} | Hop: {} | Overlap: {:.1}%\n",
        config.fft_size,
        config.hop_size,
        config.overlap_percent()
    );

    // Initialize processors
    let mut stft = StreamingStft::new(config.clone());
    let mut istft = StreamingIstft::new(config.clone());

    // Pre-allocate buffers for reuse
    let mut spectrum_frames = Vec::new();
    let mut output_samples = Vec::new();
    let mut final_output = Vec::new();

    // Generate test signal
    let sample_rate = 44100;
    let duration = 2.0;
    let num_samples = (sample_rate as f32 * duration) as usize;
    let signal: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let freq = 440.0 + 200.0 * t; // Frequency sweep
            0.5 * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    println!("Processing {} samples in chunks...", num_samples);

    // Apply padding for better edge reconstruction
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&signal, pad_amount, PadMode::Reflect);

    // Process in chunks with pre-allocated buffers
    let chunk_size = 512;
    let mut total_frames = 0;
    let mut chunks_processed = 0;

    for chunk in padded.chunks(chunk_size) {
        // Clear buffers but keep capacity
        spectrum_frames.clear();
        output_samples.clear();

        // Process with pre-allocated buffers
        let frames_written = stft.push_samples_into(chunk, &mut spectrum_frames);
        total_frames += frames_written;

        // Inverse transform each frame
        for frame in &spectrum_frames {
            istft.push_frame_into(frame, &mut output_samples);
        }

        // Accumulate final output
        final_output.extend_from_slice(&output_samples);

        chunks_processed += 1;
    }

    // Flush remaining data
    let remaining_frames = stft.flush();
    for frame in remaining_frames {
        output_samples.clear();
        istft.push_frame_into(&frame, &mut output_samples);
        final_output.extend_from_slice(&output_samples);
    }

    let flush_output = istft.flush();
    final_output.extend_from_slice(&flush_output);

    println!("  Processed {} chunks", chunks_processed);
    println!("  Generated {} STFT frames", total_frames);
    println!("  Output: {} samples\n", final_output.len());

    // Remove padding
    let start = pad_amount.min(final_output.len());
    let end = (start + signal.len()).min(final_output.len());
    let reconstructed = &final_output[start..end];

    // Calculate reconstruction quality
    let min_len = signal.len().min(reconstructed.len());
    let signal_power: f32 = signal[..min_len].iter().map(|x| x.powi(2)).sum();
    let noise_power: f32 = signal[..min_len]
        .iter()
        .zip(reconstructed[..min_len].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    let snr = if noise_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f32::INFINITY
    };

    println!("Reconstruction Quality:");
    println!("  Compared {} samples", min_len);
    println!("  SNR: {:.2} dB", snr);

    println!("\nAdvantages of buffer reuse:");
    println!("  - Reduced allocator pressure");
    println!("  - Better cache locality");
    println!("  - More predictable performance");
    println!("  - Lower latency variance (important for real-time)");
}
