use stft_rs::prelude::*;

/// Generate a test signal with a specific frequency
fn generate_tone(freq: f32, duration_samples: usize, sample_rate: f32) -> Vec<f32> {
    (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

/// Calculate SNR between two signals
/// Handles different lengths by comparing only the overlapping portion
fn calculate_snr(original: &[f32], reconstructed: &[f32]) -> f32 {
    let len = original.len().min(reconstructed.len());
    let original = &original[..len];
    let reconstructed = &reconstructed[..len];

    let signal_power: f32 = original.iter().map(|&x| x * x).sum();
    let noise_power: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(&o, &r)| (o - r).powi(2))
        .sum();

    if noise_power < 1e-10 {
        return 200.0; // Effectively perfect
    }

    10.0 * (signal_power / noise_power).log10()
}

#[test]
fn test_multichannel_vs_single_channel_stereo() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Generate stereo signals
    let left = generate_tone(220.0, 44100, 44100.0);
    let right = generate_tone(440.0, 44100, 44100.0);

    // Multi-channel approach
    let channels = vec![left.clone(), right.clone()];
    let multi_spectra = stft.process_multichannel(&channels);

    // Single-channel approach
    let single_left = stft.process(&left);
    let single_right = stft.process(&right);

    // Compare: multi-channel should match single-channel
    assert_eq!(multi_spectra.len(), 2);
    assert_eq!(multi_spectra[0].num_frames, single_left.num_frames);
    assert_eq!(multi_spectra[0].freq_bins, single_left.freq_bins);
    assert_eq!(multi_spectra[1].num_frames, single_right.num_frames);
    assert_eq!(multi_spectra[1].freq_bins, single_right.freq_bins);

    // Check data is identical
    assert_eq!(multi_spectra[0].data, single_left.data);
    assert_eq!(multi_spectra[1].data, single_right.data);

    // Test reconstruction
    let multi_reconstructed = istft.process_multichannel(&multi_spectra);
    let single_left_recon = istft.process(&single_left);
    let single_right_recon = istft.process(&single_right);

    assert_eq!(multi_reconstructed[0], single_left_recon);
    assert_eq!(multi_reconstructed[1], single_right_recon);
}

#[test]
fn test_multichannel_roundtrip_stereo() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Generate stereo signals with different frequencies
    let left = generate_tone(220.0, 44100, 44100.0);
    let right = generate_tone(440.0, 44100, 44100.0);
    let channels = vec![left.clone(), right.clone()];

    // Process
    let spectra = stft.process_multichannel(&channels);
    let reconstructed = istft.process_multichannel(&spectra);

    // Check SNR
    let snr_left = calculate_snr(&left, &reconstructed[0]);
    let snr_right = calculate_snr(&right, &reconstructed[1]);

    println!("Left channel SNR: {:.2} dB", snr_left);
    println!("Right channel SNR: {:.2} dB", snr_right);

    assert!(snr_left > 100.0, "Left channel SNR should be >100dB");
    assert!(snr_right > 100.0, "Right channel SNR should be >100dB");
}

#[test]
fn test_multichannel_roundtrip_quad() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Generate 4-channel signals with different frequencies
    let ch1 = generate_tone(220.0, 44100, 44100.0);
    let ch2 = generate_tone(330.0, 44100, 44100.0);
    let ch3 = generate_tone(440.0, 44100, 44100.0);
    let ch4 = generate_tone(550.0, 44100, 44100.0);
    let channels = vec![ch1.clone(), ch2.clone(), ch3.clone(), ch4.clone()];

    // Process
    let spectra = stft.process_multichannel(&channels);
    let reconstructed = istft.process_multichannel(&spectra);

    // Check all channels have high SNR
    for (i, (orig, recon)) in channels.iter().zip(reconstructed.iter()).enumerate() {
        let snr = calculate_snr(orig, recon);
        println!("Channel {} SNR: {:.2} dB", i, snr);
        assert!(snr > 100.0, "Channel {} SNR should be >100dB", i);
    }
}

#[test]
fn test_multichannel_roundtrip_5_1_surround() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Generate 6-channel (5.1) signals
    let channels: Vec<Vec<f32>> = (0..6)
        .map(|i| generate_tone(220.0 + (i as f32 * 55.0), 44100, 44100.0))
        .collect();

    // Process
    let spectra = stft.process_multichannel(&channels);
    assert_eq!(spectra.len(), 6);

    let reconstructed = istft.process_multichannel(&spectra);
    assert_eq!(reconstructed.len(), 6);

    // Check all channels
    for (i, (orig, recon)) in channels.iter().zip(reconstructed.iter()).enumerate() {
        let snr = calculate_snr(orig, recon);
        println!("5.1 Channel {} SNR: {:.2} dB", i, snr);
        assert!(snr > 100.0, "Channel {} SNR should be >100dB", i);
    }
}

#[test]
fn test_multichannel_roundtrip_8_channels() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Generate 8-channel signals
    let channels: Vec<Vec<f32>> = (0..8)
        .map(|i| generate_tone(220.0 + (i as f32 * 55.0), 44100, 44100.0))
        .collect();

    // Process
    let spectra = stft.process_multichannel(&channels);
    assert_eq!(spectra.len(), 8);

    let reconstructed = istft.process_multichannel(&spectra);
    assert_eq!(reconstructed.len(), 8);

    // Check all channels
    for (i, (orig, recon)) in channels.iter().zip(reconstructed.iter()).enumerate() {
        let snr = calculate_snr(orig, recon);
        println!("8-channel {} SNR: {:.2} dB", i, snr);
        assert!(snr > 100.0, "Channel {} SNR should be >100dB", i);
    }
}

#[test]
fn test_interleaved_roundtrip_stereo() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Generate stereo signals
    let left = generate_tone(220.0, 44100, 44100.0);
    let right = generate_tone(440.0, 44100, 44100.0);

    // Interleave manually
    let mut interleaved = Vec::with_capacity(88200);
    for (l, r) in left.iter().zip(right.iter()) {
        interleaved.push(*l);
        interleaved.push(*r);
    }

    // Process interleaved
    let spectra = stft.process_interleaved(&interleaved, 2);
    assert_eq!(spectra.len(), 2);

    let output = istft.process_multichannel_interleaved(&spectra);

    // Deinterleave for comparison
    let mut left_recon = Vec::new();
    let mut right_recon = Vec::new();
    for chunk in output.chunks_exact(2) {
        left_recon.push(chunk[0]);
        right_recon.push(chunk[1]);
    }

    // Check SNR
    let snr_left = calculate_snr(&left, &left_recon);
    let snr_right = calculate_snr(&right, &right_recon);

    println!("Interleaved Left SNR: {:.2} dB", snr_left);
    println!("Interleaved Right SNR: {:.2} dB", snr_right);

    assert!(snr_left > 100.0);
    assert!(snr_right > 100.0);
}

#[test]
fn test_interleaved_vs_planar() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());

    // Generate stereo signals
    let left = generate_tone(220.0, 44100, 44100.0);
    let right = generate_tone(440.0, 44100, 44100.0);

    // Planar format
    let channels = vec![left.clone(), right.clone()];
    let planar_spectra = stft.process_multichannel(&channels);

    // Interleaved format
    let interleaved = interleave(&channels);
    let interleaved_spectra = stft.process_interleaved(&interleaved, 2);

    // Should produce identical results
    assert_eq!(planar_spectra.len(), interleaved_spectra.len());
    for (planar, interleaved) in planar_spectra.iter().zip(interleaved_spectra.iter()) {
        assert_eq!(planar.data, interleaved.data);
    }
}

#[test]
fn test_channel_independence() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Generate signals: one channel is silent, one has tone
    let silent = vec![0.0; 44100];
    let tone = generate_tone(440.0, 44100, 44100.0);
    let channels = vec![silent.clone(), tone.clone()];

    // Process
    let spectra = stft.process_multichannel(&channels);
    let reconstructed = istft.process_multichannel(&spectra);

    // Silent channel should remain mostly silent
    let silent_power: f32 = reconstructed[0].iter().map(|&x| x.abs()).sum();
    assert!(
        silent_power < 0.01,
        "Silent channel should remain silent, got power: {}",
        silent_power
    );

    // Tone channel should match original
    let snr = calculate_snr(&tone, &reconstructed[1]);
    assert!(snr > 100.0);
}

#[test]
fn test_single_channel_as_multichannel() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let istft = BatchIstftF32::new(config);

    // Single channel should work with multi-channel API
    let signal = generate_tone(440.0, 44100, 44100.0);
    let channels = vec![signal.clone()];

    let spectra = stft.process_multichannel(&channels);
    assert_eq!(spectra.len(), 1);

    let reconstructed = istft.process_multichannel(&spectra);
    assert_eq!(reconstructed.len(), 1);

    let snr = calculate_snr(&signal, &reconstructed[0]);
    assert!(snr > 100.0);
}

#[test]
#[should_panic(expected = "channels must not be empty")]
fn test_multichannel_empty_channels() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config);

    let empty: Vec<Vec<f32>> = vec![];
    stft.process_multichannel(&empty);
}

#[test]
#[should_panic(expected = "Channel 1 has length")]
fn test_multichannel_mismatched_lengths() {
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config);

    let ch1 = vec![0.0; 44100];
    let ch2 = vec![0.0; 22050]; // Different length!

    stft.process_multichannel(&[ch1, ch2]);
}

#[test]
#[cfg(feature = "rustfft-backend")] // f64 not supported by microfft
fn test_multichannel_f64() {
    let config = StftConfigF64::default_4096();
    let stft = BatchStftF64::new(config.clone());
    let istft = BatchIstftF64::new(config);

    // Generate stereo f64 signals
    let left: Vec<f64> = (0..44100)
        .map(|i| (2.0 * std::f64::consts::PI * 220.0 * i as f64 / 44100.0).sin())
        .collect();
    let right: Vec<f64> = (0..44100)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
        .collect();

    let channels = vec![left.clone(), right.clone()];
    let spectra = stft.process_multichannel(&channels);
    let reconstructed = istft.process_multichannel(&spectra);

    assert_eq!(reconstructed.len(), 2);

    // Basic roundtrip check
    assert!(reconstructed[0].len() > 40000);
    assert!(reconstructed[1].len() > 40000);
}
#[test]
fn test_interleave_many_channels() {
    use stft_rs::prelude::*;

    // Test 10 channels
    let channels: Vec<Vec<f32>> = (0..10)
        .map(|i| vec![i as f32, i as f32 + 10.0, i as f32 + 20.0])
        .collect();

    let interleaved = interleave(&channels);

    // Should interleave all 10 channels
    assert_eq!(interleaved.len(), 30); // 10 channels × 3 samples

    // First sample from each channel: 0, 1, 2, ..., 9
    assert_eq!(
        &interleaved[0..10],
        &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );

    // Second sample from each channel: 10, 11, 12, ..., 19
    assert_eq!(
        &interleaved[10..20],
        &[10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0]
    );

    println!("✓ Successfully interleaved 10 channels");
}
