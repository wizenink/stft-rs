mod common;

use stft_rs::mel::*;
use stft_rs::prelude::*;

use num_traits::Float;

#[test]
fn test_htk_roundtrip() {
    let freqs = [0.0, 100.0, 1000.0, 8000.0, 22050.0];
    for &hz in &freqs {
        let mel = hz_to_mel_htk(hz);
        let hz_back = mel_to_hz_htk(mel);
        assert!(
            (hz - hz_back).abs() < 1e-6,
            "HTK roundtrip failed for {} Hz",
            hz
        );
    }
}

#[test]
fn test_slaney_roundtrip() {
    let freqs = [0.0, 100.0, 1000.0, 8000.0, 22050.0];
    for &hz in &freqs {
        let mel = hz_to_mel_slaney(hz);
        let hz_back = mel_to_hz_slaney(mel);
        assert!(
            (hz - hz_back).abs() < 1e-6,
            "Slaney roundtrip failed for {} Hz",
            hz
        );
    }
}

#[test]
fn test_mel_scale_monotonic() {
    // Test that mel scale is monotonically increasing
    let freqs = [0.0, 100.0, 500.0, 1000.0, 2000.0, 8000.0];

    let htk_mels: Vec<f64> = freqs.iter().map(|&hz| hz_to_mel_htk(hz)).collect();
    for i in 1..htk_mels.len() {
        assert!(htk_mels[i] > htk_mels[i - 1], "HTK mel scale not monotonic");
    }

    let slaney_mels: Vec<f64> = freqs.iter().map(|&hz| hz_to_mel_slaney(hz)).collect();
    for i in 1..slaney_mels.len() {
        assert!(
            slaney_mels[i] > slaney_mels[i - 1],
            "Slaney mel scale not monotonic"
        );
    }
}

#[test]
fn test_mel_filterbank_creation() {
    let config = MelConfig::<f64>::default();
    let filterbank = MelFilterbank::new(44100.0, 4096, &config);

    assert_eq!(filterbank.n_mels, 80);
    assert_eq!(filterbank.n_freqs, 2049); // 4096/2 + 1
    assert_eq!(filterbank.weights.len(), 80);

    // Each filter should have some non-zero weights
    for (i, filter) in filterbank.weights.iter().enumerate() {
        assert!(!filter.is_empty(), "Filter {} has no weights", i);
    }
}

#[test]
fn test_mel_filterbank_apply() {
    let config = MelConfig::<f64>::default();
    let filterbank = MelFilterbank::new(44100.0, 4096, &config);

    // Create a simple magnitude spectrum (all ones)
    let magnitudes = vec![1.0; 2049];
    let mel_mags = filterbank.apply(&magnitudes);

    assert_eq!(mel_mags.len(), 80);

    // All mel magnitudes should be positive for constant input
    for &mag in &mel_mags {
        assert!(mag > 0.0, "Mel magnitude should be positive");
    }
}

#[test]
fn test_mel_spectrum_creation() {
    let mel_spec = MelSpectrum::<f64>::new(100, 80);
    assert_eq!(mel_spec.num_frames, 100);
    assert_eq!(mel_spec.n_mels, 80);
    assert_eq!(mel_spec.data.len(), 8000);
}

#[test]
fn test_mel_spectrum_to_db() {
    let mut mel_spec = MelSpectrum::<f64>::new(10, 80);

    // Fill with some test values
    for i in 0..mel_spec.data.len() {
        mel_spec.data[i] = (i as f64 + 1.0) / 100.0;
    }

    let db_spec = mel_spec.to_db(None, None);

    // Check that all values are within the expected range
    let max_db = db_spec
        .data
        .iter()
        .copied()
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();
    let min_db = db_spec
        .data
        .iter()
        .copied()
        .min_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap();

    // Range should be <= top_db (80 dB default)
    assert!((max_db - min_db) <= 80.0, "dB range too large");
}

#[test]
fn test_mel_spectrum_delta() {
    let mut mel_spec = MelSpectrum::<f64>::new(10, 80);

    // Fill with linearly increasing values for easy testing
    for t in 0..10 {
        for mel_bin in 0..80 {
            mel_spec.set(t, mel_bin, t as f64);
        }
    }

    let delta = mel_spec.delta(Some(2));
    assert_eq!(delta.num_frames, 10);
    assert_eq!(delta.n_mels, 80);

    // For linearly increasing values, delta should be relatively constant
    // (except at edges where clamping occurs)
    for t in 2..8 {
        for mel_bin in 0..80 {
            let d = delta.get(t, mel_bin);
            // Delta should be positive for increasing values
            assert!(d > 0.0, "Delta should be positive for increasing values");
        }
    }
}

#[test]
fn test_mel_spectrum_with_deltas() {
    let mel_spec = MelSpectrum::<f64>::new(10, 80);
    let with_deltas = mel_spec.with_deltas(Some(2));

    assert_eq!(with_deltas.num_frames, 10);
    assert_eq!(with_deltas.n_mels, 240); // 80 * 3
}

#[test]
fn test_batch_mel_spectrogram_integration() {
    use crate::{BatchStft, StftConfig};

    // Create STFT config - use default 4096 which is COLA compliant
    let stft_config = StftConfig::<f64>::default_4096();

    // Create a test signal (1 second at 44.1kHz)
    let sample_rate = 44100.0;
    let signal: Vec<f64> = (0..44100)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sample_rate).sin())
        .collect();

    // Compute STFT
    let stft = BatchStft::new(stft_config);
    let spectrum = stft.process(&signal);

    // Create mel config
    let mel_config = MelConfig::<f64> {
        n_mels: 80,
        fmin: 0.0,
        fmax: Some(8000.0),
        mel_scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
        use_power: true,
    };

    // Process to mel spectrogram
    let mel_proc = BatchMelSpectrogram::new(sample_rate, 4096, &mel_config);
    let mel_spec = mel_proc.process(&spectrum);

    assert_eq!(mel_spec.n_mels, 80);
    assert!(mel_spec.num_frames > 0);

    // Convert to dB
    let mel_db = mel_spec.to_db(None, None);
    assert_eq!(mel_db.num_frames, mel_spec.num_frames);
    assert_eq!(mel_db.n_mels, 80);

    // Add delta features
    let with_deltas = mel_db.with_deltas(Some(2));
    assert_eq!(with_deltas.n_mels, 240); // 80 * 3
}

#[test]
fn test_streaming_mel_spectrogram_integration() {
    use crate::{StftConfig, StreamingStft};

    // Create STFT config - use default 4096 which is COLA compliant
    let stft_config = StftConfig::<f64>::default_4096();

    // Create streaming processors
    let mut stft = StreamingStft::new(stft_config);
    let mel_config = MelConfig::<f64>::default();
    let mel_proc = StreamingMelSpectrogram::new(44100.0, 4096, &mel_config);

    // Create a test signal chunk
    let chunk: Vec<f64> = (0..8192)
        .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / 44100.0).sin())
        .collect();

    // Process through streaming STFT
    let frames = stft.push_samples(&chunk);

    // Process each frame through mel processor
    let mel_frames: Vec<Vec<f64>> = frames
        .iter()
        .map(|frame| mel_proc.process_frame(frame))
        .collect();

    // Verify output
    for mel_frame in &mel_frames {
        assert_eq!(mel_frame.len(), 80);
        // Check that values are positive
        for &val in mel_frame {
            assert!(val >= 0.0, "Mel values should be non-negative");
        }
    }
}

#[test]
fn test_mel_scales_compatibility() {
    // Test that both HTK and Slaney scales work
    let configs = vec![
        MelConfig::<f64> {
            mel_scale: MelScale::Htk,
            ..Default::default()
        },
        MelConfig::<f64> {
            mel_scale: MelScale::Slaney,
            ..Default::default()
        },
    ];

    for config in configs {
        let filterbank = MelFilterbank::new(44100.0, 4096, &config);
        assert_eq!(filterbank.n_mels, 80);
        assert_eq!(filterbank.n_freqs, 2049);

        // Test that filters are valid
        for (i, filter) in filterbank.weights.iter().enumerate() {
            assert!(!filter.is_empty(), "Filter {} should have weights", i);
            // Check that weights sum to something reasonable
            let sum: f64 = filter.iter().map(|(_, w)| w).sum();
            assert!(sum > 0.0, "Filter {} weights should sum to > 0", i);
        }
    }
}
