mod common;

use stft_rs::prelude::*;

#[test]
fn test_streaming_ola_roundtrip() {
    let config = StftConfig::<f32>::default_4096();
    let mut stft = StreamingStft::new(config.clone());
    let mut istft = StreamingIstft::new(config.clone());

    // Generate test signal
    let signal_len = 127 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    // For streaming, pad the signal to match batch behavior
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

    // Process in chunks
    let chunk_size = 2048;
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

    // Remove padding from reconstruction
    let start = pad_amount.min(reconstructed.len());
    let end = (start + signal_len).min(reconstructed.len());
    let reconstructed_unpadded = &reconstructed[start..end];

    let compare_len = original.len().min(reconstructed_unpadded.len());
    let snr = common::calculate_snr(
        &original[..compare_len],
        &reconstructed_unpadded[..compare_len],
    );
    println!("Streaming OLA SNR: {:.2} dB", snr);
    assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
}

#[test]
fn test_streaming_wola_roundtrip() {
    let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Wola)
        .expect("Config should be valid");
    let mut stft = StreamingStft::new(config.clone());
    let mut istft = StreamingIstft::new(config.clone());

    let signal_len = 127 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    // Pad for streaming
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

    let chunk_size = 2048;
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

    // Remove padding
    let start = pad_amount.min(reconstructed.len());
    let end = (start + signal_len).min(reconstructed.len());
    let reconstructed_unpadded = &reconstructed[start..end];

    let compare_len = original.len().min(reconstructed_unpadded.len());
    let snr = common::calculate_snr(
        &original[..compare_len],
        &reconstructed_unpadded[..compare_len],
    );
    println!("Streaming WOLA SNR: {:.2} dB", snr);
    assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
}

#[test]
fn test_batch_vs_streaming_consistency() {
    let config = StftConfig::<f32>::default_4096();

    // Batch processing (pads internally)
    let batch_stft = BatchStft::new(config.clone());
    let signal_len = 50 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    let batch_result = batch_stft.process(&original);

    // Streaming processing - need to pad manually to match batch
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

    let mut streaming_stft = StreamingStft::new(config.clone());
    let streaming_frames = streaming_stft.push_samples(&padded);

    // Compare number of frames
    assert_eq!(batch_result.num_frames, streaming_frames.len());

    // Compare spectral content
    for (frame_idx, streaming_frame) in streaming_frames.iter().enumerate() {
        for bin in 0..batch_result.freq_bins {
            let batch_complex = batch_result.get_complex(frame_idx, bin);
            let streaming_complex = streaming_frame.data[bin];

            let diff_re = (batch_complex.re - streaming_complex.re).abs();
            let diff_im = (batch_complex.im - streaming_complex.im).abs();

            assert!(
                diff_re < 1e-4,
                "Real part mismatch at frame {}, bin {}: {} vs {}",
                frame_idx,
                bin,
                batch_complex.re,
                streaming_complex.re
            );
            assert!(
                diff_im < 1e-4,
                "Imag part mismatch at frame {}, bin {}: {} vs {}",
                frame_idx,
                bin,
                batch_complex.im,
                streaming_complex.im
            );
        }
    }
}

#[test]
fn test_streaming_reset() {
    let config = StftConfig::<f32>::default_4096();
    let mut stft = StreamingStft::new(config.clone());

    let samples = vec![1.0; 5000];
    stft.push_samples(&samples);
    assert!(stft.buffered_samples() > 0);

    stft.reset();
    assert_eq!(stft.buffered_samples(), 0);
}
