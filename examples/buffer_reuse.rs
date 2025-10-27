use std::time::Instant;
use stft_rs::prelude::*;

fn main() {
    let config = StftConfig::<f32>::default_4096();

    println!("Buffer Reuse Performance Comparison\n");
    println!("Configuration:");
    println!("  FFT size: {}", config.fft_size);
    println!("  Hop size: {}", config.hop_size);
    println!();

    // Generate test signal
    let sample_rate = 44100;
    let duration = 10.0; // 10 seconds
    let total_samples = (sample_rate as f32 * duration) as usize;
    let audio: Vec<f32> = (0..total_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    println!(
        "Signal: {} samples ({:.1}s @ {}Hz)\n",
        total_samples, duration, sample_rate
    );

    // Streaming mode comparison
    println!("=== Streaming Mode ===\n");

    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&audio, pad_amount, PadMode::Reflect);
    let chunk_size = 512;

    // Method 1: Standard push_samples (allocates on each call)
    {
        let mut stft = StreamingStft::new(config.clone());
        let mut istft = StreamingIstft::new(config.clone());

        let start = Instant::now();
        let mut reconstructed = Vec::new();

        for chunk in padded.chunks(chunk_size) {
            let frames = stft.push_samples(chunk);
            for frame in frames {
                let samples = istft.push_frame(&frame);
                reconstructed.extend(samples);
            }
        }

        let remaining_frames = stft.flush();
        for frame in remaining_frames {
            let samples = istft.push_frame(&frame);
            reconstructed.extend(samples);
        }
        reconstructed.extend(istft.flush());

        let elapsed = start.elapsed();
        println!("Standard API (allocating):");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Throughput: {:.2} samples/ms",
            total_samples as f64 / elapsed.as_millis() as f64
        );
    }

    // Method 2: Pre-allocated buffers (_into methods)
    {
        let mut stft = StreamingStft::new(config.clone());
        let mut istft = StreamingIstft::new(config.clone());

        let start = Instant::now();
        let mut frames = Vec::new();
        let mut reconstructed = Vec::new();

        for chunk in padded.chunks(chunk_size) {
            frames.clear();
            stft.push_samples_into(chunk, &mut frames);
            for frame in &frames {
                istft.push_frame_into(frame, &mut reconstructed);
            }
        }

        let elapsed = start.elapsed();
        println!("\nPre-allocated Vec buffers (_into API):");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Throughput: {:.2} samples/ms",
            total_samples as f64 / elapsed.as_millis() as f64
        );
    }

    // Method 3: Zero-allocation with frame pool
    {
        let mut stft = StreamingStft::new(config.clone());
        let mut istft = StreamingIstft::new(config.clone());

        let start = Instant::now();

        // Pre-allocate a pool of frames (estimate max needed)
        let max_frames_per_chunk = (chunk_size + config.hop_size - 1) / config.hop_size + 1;
        let mut frame_pool = vec![SpectrumFrame::new(config.freq_bins()); max_frames_per_chunk];
        let mut reconstructed = Vec::new();

        for chunk in padded.chunks(chunk_size) {
            let mut pool_index = 0;
            let frames_written = stft.push_samples_write(chunk, &mut frame_pool, &mut pool_index);

            for i in 0..frames_written {
                istft.push_frame_into(&frame_pool[i], &mut reconstructed);
            }
        }

        let elapsed = start.elapsed();
        println!("\nZero-allocation frame pool (write API):");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Throughput: {:.2} samples/ms",
            total_samples as f64 / elapsed.as_millis() as f64
        );
    }

    // Batch mode comparison
    println!("\n=== Batch Mode ===\n");

    // Method 1: Standard process (allocates)
    {
        let stft = BatchStft::new(config.clone());
        let istft = BatchIstft::new(config.clone());

        let start = Instant::now();
        let spectrum = stft.process(&audio);
        let reconstructed = istft.process(&spectrum);
        let elapsed = start.elapsed();

        println!("Standard API (allocating):");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Throughput: {:.2} samples/ms",
            total_samples as f64 / elapsed.as_millis() as f64
        );
        println!("  Output: {} samples", reconstructed.len());
    }

    // Method 2: Pre-allocated buffers
    {
        let stft = BatchStft::new(config.clone());
        let istft = BatchIstft::new(config.clone());

        // Pre-allocate output buffers
        let num_frames = (audio.len() + config.hop_size - 1) / config.hop_size;
        let mut spectrum = Spectrum::new(num_frames, config.freq_bins());
        let mut reconstructed = Vec::new();

        let start = Instant::now();
        if stft.process_into(&audio, &mut spectrum) {
            istft.process_into(&spectrum, &mut reconstructed);
        }
        let elapsed = start.elapsed();

        println!("\nPre-allocated buffers API:");
        println!("  Time: {:.2}ms", elapsed.as_secs_f64() * 1000.0);
        println!(
            "  Throughput: {:.2} samples/ms",
            total_samples as f64 / elapsed.as_millis() as f64
        );
        println!("  Output: {} samples", reconstructed.len());
    }

    println!("\n=== Summary ===");
    println!("Three allocation strategies:");
    println!("  1. Standard API: Allocates Vec for each frame (simple, good for most uses)");
    println!("  2. _into API: Reuses outer Vec but still allocates frame data");
    println!("  3. Frame pool: Zero allocations after setup (best for real-time)");
    println!();
    println!("Frame pool benefits:");
    println!("  - Predictable performance (no allocator jitter)");
    println!("  - Better cache locality");
    println!("  - Essential for hard real-time constraints");
}
