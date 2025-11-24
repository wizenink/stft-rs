mod common;

use stft_rs::fft_backend::Complex;
use stft_rs::prelude::*;

#[test]
fn test_batch_process_into() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config.clone());

    let signal_len = 50 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    // Standard API
    let spectrum_standard = stft.process(&original);
    let reconstructed_standard = istft.process(&spectrum_standard);

    // _into API with pre-allocated buffers
    // Batch mode pads internally, so calculate frames from padded length
    let pad_amount = config.fft_size / 2;
    let padded_len = original.len() + 2 * pad_amount;
    let num_frames = if padded_len >= config.fft_size {
        (padded_len - config.fft_size) / config.hop_size + 1
    } else {
        0
    };
    let mut spectrum_into = Spectrum::new(num_frames, config.freq_bins());
    let mut reconstructed_into = Vec::new();

    assert!(stft.process_into(&original, &mut spectrum_into));
    istft.process_into(&spectrum_into, &mut reconstructed_into);

    // Results should be identical
    assert_eq!(spectrum_standard.num_frames, spectrum_into.num_frames);
    assert_eq!(spectrum_standard.freq_bins, spectrum_into.freq_bins);
    assert_eq!(reconstructed_standard.len(), reconstructed_into.len());

    // Compare spectral data
    for frame in 0..spectrum_standard.num_frames {
        for bin in 0..spectrum_standard.freq_bins {
            let c1 = spectrum_standard.get_complex(frame, bin);
            let c2 = spectrum_into.get_complex(frame, bin);
            assert!((c1.re - c2.re).abs() < 1e-6);
            assert!((c1.im - c2.im).abs() < 1e-6);
        }
    }

    // Compare reconstructed audio
    let snr = common::calculate_snr(&reconstructed_standard, &reconstructed_into);
    assert!(
        snr > 200.0,
        "SNR should be very high (identical): {:.2} dB",
        snr
    );
}

#[test]
fn test_streaming_push_samples_into() {
    let config = StftConfig::<f32>::default_4096();
    let mut stft_standard = StreamingStft::new(config.clone());
    let mut stft_into = StreamingStft::new(config.clone());

    let signal_len = 30 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

    let chunk_size = 2048;
    let mut frames_standard = Vec::new();
    let mut frames_into_container = Vec::new();
    let mut frames_into_all = Vec::new();

    for chunk in padded.chunks(chunk_size) {
        // Standard API
        let chunk_frames = stft_standard.push_samples(chunk);
        frames_standard.extend(chunk_frames);

        // _into API
        frames_into_container.clear();
        stft_into.push_samples_into(chunk, &mut frames_into_container);
        frames_into_all.extend_from_slice(&frames_into_container);
    }

    // Both should produce same number of frames
    assert_eq!(frames_standard.len(), frames_into_all.len());

    // Compare spectral data
    for (i, (f1, f2)) in frames_standard
        .iter()
        .zip(frames_into_all.iter())
        .enumerate()
    {
        assert_eq!(f1.freq_bins, f2.freq_bins);
        for bin in 0..f1.freq_bins {
            let diff_re = (f1.data[bin].re - f2.data[bin].re).abs();
            let diff_im = (f1.data[bin].im - f2.data[bin].im).abs();
            assert!(diff_re < 1e-6, "Frame {} bin {} real mismatch", i, bin);
            assert!(diff_im < 1e-6, "Frame {} bin {} imag mismatch", i, bin);
        }
    }
}

#[test]
fn test_streaming_push_samples_write() {
    let config = StftConfig::<f32>::default_4096();
    let mut stft_standard = StreamingStft::new(config.clone());
    let mut stft_write = StreamingStft::new(config.clone());

    let signal_len = 30 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

    let chunk_size = 2048;
    let max_frames_per_chunk = (chunk_size + config.hop_size - 1) / config.hop_size + 1;
    let mut frame_pool = vec![SpectrumFrame::new(config.freq_bins()); max_frames_per_chunk];
    let mut frames_standard = Vec::new();
    let mut frames_write = Vec::new();

    for chunk in padded.chunks(chunk_size) {
        // Standard API
        let chunk_frames = stft_standard.push_samples(chunk);
        frames_standard.extend(chunk_frames);

        // _write API
        let mut pool_index = 0;
        let frames_written = stft_write.push_samples_write(chunk, &mut frame_pool, &mut pool_index);
        for i in 0..frames_written {
            frames_write.push(frame_pool[i].clone());
        }
    }

    // Both should produce same number of frames
    assert_eq!(frames_standard.len(), frames_write.len());

    // Compare spectral data
    for (i, (f1, f2)) in frames_standard.iter().zip(frames_write.iter()).enumerate() {
        assert_eq!(f1.freq_bins, f2.freq_bins);
        for bin in 0..f1.freq_bins {
            let diff_re = (f1.data[bin].re - f2.data[bin].re).abs();
            let diff_im = (f1.data[bin].im - f2.data[bin].im).abs();
            assert!(diff_re < 1e-6, "Frame {} bin {} real mismatch", i, bin);
            assert!(diff_im < 1e-6, "Frame {} bin {} imag mismatch", i, bin);
        }
    }
}

#[test]
fn test_streaming_push_frame_into() {
    let config = StftConfig::<f32>::default_4096();
    let mut stft = StreamingStft::new(config.clone());
    let mut istft_standard = StreamingIstft::new(config.clone());
    let mut istft_into = StreamingIstft::new(config.clone());

    let signal_len = 20 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&original, pad_amount, PadMode::Reflect);

    let frames = stft.push_samples(&padded);

    let mut reconstructed_standard = Vec::new();
    let mut reconstructed_into = Vec::new();

    for frame in &frames {
        // Standard API
        let samples_standard = istft_standard.push_frame(frame);
        reconstructed_standard.extend(samples_standard);

        // _into API
        istft_into.push_frame_into(frame, &mut reconstructed_into);
    }

    // Both should produce same output
    assert_eq!(reconstructed_standard.len(), reconstructed_into.len());

    for (i, (s1, s2)) in reconstructed_standard
        .iter()
        .zip(reconstructed_into.iter())
        .enumerate()
    {
        assert!(
            (s1 - s2).abs() < 1e-6,
            "Sample {} mismatch: {} vs {}",
            i,
            s1,
            s2
        );
    }
}

#[test]
fn test_frame_pool_reuse() {
    let config = StftConfig::<f32>::default_4096();
    let mut stft = StreamingStft::new(config.clone());

    let chunk_size = 2048;
    let max_frames_per_chunk = (chunk_size + config.hop_size - 1) / config.hop_size + 1;
    let mut frame_pool = vec![SpectrumFrame::new(config.freq_bins()); max_frames_per_chunk];

    let pad_amount = config.fft_size / 2;
    let signal: Vec<f32> = (0..50000).map(|i| (i as f32 * 0.01).sin()).collect();
    let padded = apply_padding(&signal, pad_amount, PadMode::Reflect);

    let mut total_frames = 0;

    // Process multiple chunks with same pool
    for chunk in padded.chunks(chunk_size) {
        let mut pool_index = 0;
        let frames_written = stft.push_samples_write(chunk, &mut frame_pool, &mut pool_index);

        // First chunk might not produce frames yet (needs fft_size samples buffered)
        // But verify that frames_written is within bounds
        assert!(frames_written <= max_frames_per_chunk);

        // Verify each frame has valid data
        for i in 0..frames_written {
            assert_eq!(frame_pool[i].freq_bins, config.freq_bins());
            assert_eq!(frame_pool[i].data.len(), config.freq_bins());

            // Check that data is not all zeros (should have spectral content)
            let has_nonzero = frame_pool[i]
                .data
                .iter()
                .any(|c| c.re.abs() > 1e-10 || c.im.abs() > 1e-10);
            assert!(has_nonzero, "Frame {} should have spectral content", i);
        }

        total_frames += frames_written;
    }

    // Verify we produced frames overall
    assert!(total_frames > 0, "Should have produced some frames");
}

#[test]
fn test_spectrum_frame_utility_methods() {
    let freq_bins = 2049;
    let mut frame = SpectrumFrame::new(freq_bins);

    // Test clear
    frame.data[0] = Complex::new(1.0, 2.0);
    frame.data[100] = Complex::new(3.0, 4.0);
    frame.clear();
    assert!(frame.data.iter().all(|c| c.re == 0.0 && c.im == 0.0));

    // Test resize_if_needed
    frame.resize_if_needed(1025);
    assert_eq!(frame.freq_bins, 1025);
    assert_eq!(frame.data.len(), 1025);

    frame.resize_if_needed(1025); // Same size, should not change
    assert_eq!(frame.freq_bins, 1025);

    // Test write_from_slice
    let test_data: Vec<Complex<f32>> = (0..512)
        .map(|i| Complex::new(i as f32, (i * 2) as f32))
        .collect();
    frame.write_from_slice(&test_data);
    assert_eq!(frame.freq_bins, 512);
    assert_eq!(frame.data.len(), 512);
    for (i, c) in frame.data.iter().enumerate() {
        assert_eq!(c.re, i as f32);
        assert_eq!(c.im, (i * 2) as f32);
    }
}

#[test]
fn test_full_roundtrip_with_all_allocation_methods() {
    let config = StftConfig::<f32>::default_4096();

    let signal_len = 40 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| {
            let t = i as f32 / 44100.0;
            0.5 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();

    // Method 1: Standard API
    let mut stft1 = StreamingStft::new(config.clone());
    let mut istft1 = StreamingIstft::new(config.clone());
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&original, pad_amount, PadMode::Reflect);
    let mut reconstructed1 = Vec::new();

    for chunk in padded.chunks(2048) {
        let frames = stft1.push_samples(chunk);
        for frame in frames {
            reconstructed1.extend(istft1.push_frame(&frame));
        }
    }
    for frame in stft1.flush() {
        reconstructed1.extend(istft1.push_frame(&frame));
    }
    reconstructed1.extend(istft1.flush());

    // Method 2: _into API
    let mut stft2 = StreamingStft::new(config.clone());
    let mut istft2 = StreamingIstft::new(config.clone());
    let mut frames_container = Vec::new();
    let mut reconstructed2 = Vec::new();

    for chunk in padded.chunks(2048) {
        frames_container.clear();
        stft2.push_samples_into(chunk, &mut frames_container);
        for frame in &frames_container {
            istft2.push_frame_into(frame, &mut reconstructed2);
        }
    }
    for frame in stft2.flush() {
        istft2.push_frame_into(&frame, &mut reconstructed2);
    }
    reconstructed2.extend(istft2.flush());

    // Method 3: _write API
    let mut stft3 = StreamingStft::new(config.clone());
    let mut istft3 = StreamingIstft::new(config.clone());
    let max_frames = (2048 + config.hop_size - 1) / config.hop_size + 1;
    let mut frame_pool = vec![SpectrumFrame::new(config.freq_bins()); max_frames];
    let mut reconstructed3 = Vec::new();

    for chunk in padded.chunks(2048) {
        let mut pool_index = 0;
        let frames_written = stft3.push_samples_write(chunk, &mut frame_pool, &mut pool_index);
        for i in 0..frames_written {
            istft3.push_frame_into(&frame_pool[i], &mut reconstructed3);
        }
    }
    for frame in stft3.flush() {
        istft3.push_frame_into(&frame, &mut reconstructed3);
    }
    reconstructed3.extend(istft3.flush());

    // All three methods should produce identical results
    assert_eq!(reconstructed1.len(), reconstructed2.len());
    assert_eq!(reconstructed1.len(), reconstructed3.len());

    // Remove padding and compare with original
    let start = pad_amount.min(reconstructed1.len());
    let end = (start + original.len()).min(reconstructed1.len());

    let unpadded1 = &reconstructed1[start..end];
    let unpadded2 = &reconstructed2[start..end];
    let unpadded3 = &reconstructed3[start..end];

    // All methods should match exactly
    for i in 0..unpadded1.len() {
        assert!((unpadded1[i] - unpadded2[i]).abs() < 1e-6);
        assert!((unpadded1[i] - unpadded3[i]).abs() < 1e-6);
    }

    // All should have high SNR
    let compare_len = original.len().min(unpadded1.len());
    let snr1 = common::calculate_snr(&original[..compare_len], &unpadded1[..compare_len]);
    let snr2 = common::calculate_snr(&original[..compare_len], &unpadded2[..compare_len]);
    let snr3 = common::calculate_snr(&original[..compare_len], &unpadded3[..compare_len]);

    println!("Standard API SNR: {:.2} dB", snr1);
    println!("_into API SNR: {:.2} dB", snr2);
    println!("_write API SNR: {:.2} dB", snr3);

    assert!(snr1 > 100.0);
    assert!(snr2 > 100.0);
    assert!(snr3 > 100.0);
}
