mod common;

use stft_rs::prelude::*;

#[test]
fn test_batch_ola_roundtrip() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config.clone());

    // Generate test signal (127 hops = 127 * 1024 samples)
    let signal_len = 127 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    let spectrum = stft.process(&original);
    let reconstructed = istft.process(&spectrum);

    assert_eq!(original.len(), reconstructed.len());

    let snr = common::calculate_snr(&original, &reconstructed);
    println!("Batch OLA SNR: {:.2} dB", snr);
    assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
}

#[test]
fn test_batch_wola_roundtrip() {
    let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Wola)
        .expect("Config should be valid");
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config.clone());

    let signal_len = 127 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    let spectrum = stft.process(&original);
    let reconstructed = istft.process(&spectrum);

    assert_eq!(original.len(), reconstructed.len());

    let snr = common::calculate_snr(&original, &reconstructed);
    println!("Batch WOLA SNR: {:.2} dB", snr);
    assert!(snr > 100.0, "SNR too low: {:.2} dB", snr);
}

#[test]
fn test_batch_constant_signal() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config.clone());

    let signal_len = 127 * 1024;
    let original = vec![1.0; signal_len];

    let spectrum = stft.process(&original);
    let reconstructed = istft.process(&spectrum);

    let max_error = common::max_abs_error(&original, &reconstructed);
    println!("Constant signal max error: {:.6}", max_error);
    assert!(max_error < 0.001, "Max error too large: {:.6}", max_error);
}

#[test]
fn test_stft_result_accessors() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());

    let signal_len = 10 * 1024;
    let original: Vec<f32> = vec![1.0; signal_len];
    let result = stft.process(&original);

    // Test accessors
    for frame in 0..result.num_frames {
        for bin in 0..result.freq_bins {
            let complex = result.get_complex(frame, bin);
            assert_eq!(complex.re, result.real(frame, bin));
            assert_eq!(complex.im, result.imag(frame, bin));
        }
    }

    // Test frame iterator
    let frames: Vec<_> = result.frames().collect();
    assert_eq!(frames.len(), result.num_frames);
    assert_eq!(frames[0].freq_bins, result.freq_bins);
}

#[test]
fn test_different_windows() {
    for window_type in [WindowType::Hann, WindowType::Hamming, WindowType::Blackman] {
        let config = StftConfig::<f32>::new(4096, 1024, window_type, ReconstructionMode::Ola)
            .expect("Config should be valid");
        let stft = BatchStft::new(config.clone());
        let istft = BatchIstft::new(config.clone());

        let signal_len = 50 * 1024;
        let original: Vec<f32> = (0..signal_len)
            .map(|i| (i as f32 * 0.01).sin() * 0.1)
            .collect();

        let spectrum = stft.process(&original);
        let reconstructed = istft.process(&spectrum);

        let snr = common::calculate_snr(&original, &reconstructed);
        println!("{:?} window SNR: {:.2} dB", window_type, snr);
        assert!(snr > 100.0, "{:?} SNR too low: {:.2} dB", window_type, snr);
    }
}

#[test]
fn test_padding_modes() {
    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());

    let signal_len = 20 * 1024;
    let original: Vec<f32> = (0..signal_len)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();

    // All padding modes should work
    for pad_mode in [PadMode::Reflect, PadMode::Zero, PadMode::Edge] {
        let result = stft.process_padded(&original, pad_mode);
        assert!(result.num_frames > 0);
        assert_eq!(result.freq_bins, config.freq_bins());
    }
}
