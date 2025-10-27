use stft_rs::prelude::*;

fn main() {
    println!("Type Aliases Example\n");
    println!("Demonstrates using type aliases for cleaner code\n");

    // Without type aliases (verbose)
    let config_verbose: StftConfig<f32> = StftConfig::default_4096();
    let _stft_verbose: BatchStft<f32> = BatchStft::new(config_verbose.clone());
    let _istft_verbose: BatchIstft<f32> = BatchIstft::new(config_verbose);

    // With type aliases (concise)
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config.clone());

    // Or use builder with type aliases for custom configs
    let custom_config = StftConfigBuilderF32::new()
        .fft_size(4096)
        .hop_size(1024)
        .window(WindowType::Blackman)
        .reconstruction_mode(ReconstructionMode::Wola)
        .build()
        .expect("Valid config");

    println!("Using type aliases makes the code cleaner:");
    println!("  Instead of: StftConfig::<f32>");
    println!("  Use: StftConfigF32");
    println!();
    println!("  Instead of: BatchStft::<f32>");
    println!("  Use: BatchStftF32");
    println!();
    println!("  Builder: StftConfigBuilderF32::new()");
    println!("    .fft_size(4096).hop_size(1024)");
    println!("    .window(WindowType::Blackman).build()");
    println!();

    // Generate test signal
    let sample_rate = 44100;
    let duration = 1.0;
    let samples = (sample_rate as f32 * duration) as usize;
    let signal: Vec<f32> = (0..samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    println!("Processing {} samples...", signal.len());

    // Process with batch API
    let spectrum: SpectrumF32 = stft.process(&signal);
    let reconstructed = istft.process(&spectrum);

    println!("  Generated {} STFT frames", spectrum.num_frames);
    println!("  Frequency bins: {}", spectrum.freq_bins);
    println!("  Reconstructed {} samples", reconstructed.len());

    // Calculate SNR
    let signal_power: f32 = signal.iter().map(|x| x.powi(2)).sum();
    let noise_power: f32 = signal
        .iter()
        .zip(reconstructed.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum();

    let snr = if noise_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f32::INFINITY
    };

    println!("  SNR: {:.2} dB", snr);

    // Streaming example with type aliases
    println!("\nStreaming example:");
    let mut streaming_stft = StreamingStftF32::new(config.clone());
    let mut streaming_istft = StreamingIstftF32::new(config.clone());

    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&signal, pad_amount, PadMode::Reflect);

    let mut output = Vec::new();
    let chunk_size = 512;
    let mut frame_count = 0;

    for chunk in padded.chunks(chunk_size) {
        let frames: Vec<SpectrumFrameF32> = streaming_stft.push_samples(chunk);
        frame_count += frames.len();
        for frame in frames {
            output.extend(streaming_istft.push_frame(&frame));
        }
    }

    for frame in streaming_stft.flush() {
        output.extend(streaming_istft.push_frame(&frame));
    }
    output.extend(streaming_istft.flush());

    println!("  Processed {} chunks", padded.len() / chunk_size + 1);
    println!("  Generated {} frames", frame_count);
    println!("  Output length: {} samples", output.len());

    println!("\nAvailable type aliases:");
    println!("  Config:    StftConfigF32, StftConfigF64");
    println!("  Builder:   StftConfigBuilderF32, StftConfigBuilderF64");
    println!("  Batch:     BatchStftF32, BatchIstftF32");
    println!("  Streaming: StreamingStftF32, StreamingIstftF32");
    println!("  Data:      SpectrumF32, SpectrumFrameF32");

    println!("\nCustom config validation:");
    println!("  FFT size: {}", custom_config.fft_size);
    println!("  Hop size: {}", custom_config.hop_size);
    println!("  Window: {:?}", custom_config.window);
}
