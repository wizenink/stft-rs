use stft_rs::prelude::*;

fn main() {
    let config = StftConfig::<f32>::default_4096();

    println!("Streaming STFT Demo");
    println!(
        "FFT size: {}, Hop size: {}",
        config.fft_size, config.hop_size
    );
    println!("Processing audio in chunks...\n");

    let mut stft = StreamingStft::new(config.clone());
    let mut istft = StreamingIstft::new(config.clone());

    let sample_rate = 44100;
    let chunk_size = 512;
    let total_samples = sample_rate * 2;
    let audio: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            let freq = 200.0 + 900.0 * t;
            0.5 * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&audio, pad_amount, PadMode::Reflect);

    let mut reconstructed = Vec::new();
    let mut total_frames = 0;
    let mut chunks_processed = 0;

    for chunk in padded.chunks(chunk_size) {
        let frames = stft.push_samples(chunk);
        total_frames += frames.len();

        for frame in frames {
            let audio_chunk = istft.push_frame(&frame);
            reconstructed.extend(audio_chunk);
        }

        chunks_processed += 1;

        if chunks_processed % 20 == 0 {
            println!(
                "Processed {} chunks ({} samples), {} frames generated, {} samples reconstructed",
                chunks_processed,
                chunks_processed * chunk_size,
                total_frames,
                reconstructed.len()
            );
        }
    }

    let remaining_frames = stft.flush();
    for frame in remaining_frames {
        let audio_chunk = istft.push_frame(&frame);
        reconstructed.extend(audio_chunk);
    }
    reconstructed.extend(istft.flush());

    println!("\nStreaming processing complete:");
    println!("  Total chunks: {}", chunks_processed);
    println!("  Original samples: {}", total_samples);
    println!("  Padded samples: {}", padded.len());
    println!("  Output samples: {}", reconstructed.len());
    println!("  STFT frames: {}", total_frames);

    let start = pad_amount.min(reconstructed.len());
    let end = (start + audio.len()).min(reconstructed.len());
    let reconstructed_unpadded = &reconstructed[start..end];
    let compare_len = audio.len().min(reconstructed_unpadded.len());
    let signal_power: f32 = audio[..compare_len].iter().map(|x| x.powi(2)).sum::<f32>();

    let noise_power: f32 = audio[..compare_len]
        .iter()
        .zip(reconstructed_unpadded[..compare_len].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>();

    let snr = if noise_power > 0.0 {
        10.0 * (signal_power / noise_power).log10()
    } else {
        f32::INFINITY
    };

    println!("\nReconstruction quality:");
    println!("  Compared samples: {}", compare_len);
    println!("  SNR: {:.2} dB", snr);

    let expected_latency = config.fft_size - config.hop_size;
    println!("\nLatency:");
    println!(
        "  Algorithmic: {} samples ({:.1} ms @ {} Hz)",
        expected_latency,
        expected_latency as f32 / sample_rate as f32 * 1000.0,
        sample_rate
    );
    println!("  Padding pre-roll: {} samples", pad_amount);
    println!("  Buffered: {} samples", stft.buffered_samples());
}
