use stft_rs::prelude::*;

/// Generate a test sine wave
fn generate_tone(freq: f32, duration_samples: usize, sample_rate: f32) -> Vec<f32> {
    (0..duration_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect()
}

#[test]
fn test_multichannel_streaming_stft() {
    let config = StftConfigF32::default_4096();
    let mut stft = MultiChannelStreamingStftF32::new(config, 2);

    assert_eq!(stft.num_channels(), 2);

    // Push some samples
    let left = vec![0.0; 512];
    let right = vec![1.0; 512];

    let frames = stft.push_samples(&[&left[..], &right[..]]);
    assert_eq!(frames.len(), 2); // 2 channels
}

#[test]
fn test_multichannel_streaming_roundtrip() {
    let config = StftConfigF32::default_4096();
    let mut stft = MultiChannelStreamingStftF32::new(config.clone(), 2);
    let mut istft = MultiChannelStreamingIstftF32::new(config.clone(), 2);

    let sample_rate = 44100.0;
    let left = generate_tone(220.0, 4096, sample_rate);
    let right = generate_tone(440.0, 4096, sample_rate);

    // Process through STFT
    let frames = stft.push_samples(&[&left[..], &right[..]]);

    // Reconstruct through iSTFT
    let mut output: Vec<Vec<f32>> = vec![Vec::new(), Vec::new()];

    for i in 0..frames[0].len() {
        let channel_frames: Vec<&SpectrumFrame<f32>> = (0..2).map(|ch| &frames[ch][i]).collect();
        let samples = istft.push_frames(&channel_frames);
        output[0].extend(&samples[0]);
        output[1].extend(&samples[1]);
    }

    // Flush remaining
    let flush_frames = stft.flush();
    for i in 0..flush_frames[0].len() {
        let channel_frames: Vec<&SpectrumFrame<f32>> =
            (0..2).map(|ch| &flush_frames[ch][i]).collect();
        let samples = istft.push_frames(&channel_frames);
        output[0].extend(&samples[0]);
        output[1].extend(&samples[1]);
    }

    let flush_samples = istft.flush();
    output[0].extend(&flush_samples[0]);
    output[1].extend(&flush_samples[1]);

    // Basic sanity checks
    assert!(output[0].len() > 3000);
    assert!(output[1].len() > 3000);
}

#[test]
fn test_multichannel_streaming_stereo() {
    let config = StftConfigF32::default_4096();
    let mut stft = MultiChannelStreamingStftF32::new(config.clone(), 2);
    let mut istft = MultiChannelStreamingIstftF32::new(config, 2);

    let chunk_size = 512;
    let left = vec![0.5; chunk_size * 10];
    let right = vec![-0.5; chunk_size * 10];

    let mut output: Vec<Vec<f32>> = vec![Vec::new(), Vec::new()];

    // Process in chunks
    for i in 0..10 {
        let left_chunk = &left[i * chunk_size..(i + 1) * chunk_size];
        let right_chunk = &right[i * chunk_size..(i + 1) * chunk_size];

        let frames = stft.push_samples(&[left_chunk, right_chunk]);

        for frame_idx in 0..frames[0].len() {
            let channel_frames: Vec<&SpectrumFrame<f32>> =
                (0..2).map(|ch| &frames[ch][frame_idx]).collect();
            let samples = istft.push_frames(&channel_frames);
            output[0].extend(&samples[0]);
            output[1].extend(&samples[1]);
        }
    }

    // Flush
    let flush_frames = stft.flush();
    for i in 0..flush_frames[0].len() {
        let channel_frames: Vec<&SpectrumFrame<f32>> =
            (0..2).map(|ch| &flush_frames[ch][i]).collect();
        let samples = istft.push_frames(&channel_frames);
        output[0].extend(&samples[0]);
        output[1].extend(&samples[1]);
    }

    let flush_samples = istft.flush();
    output[0].extend(&flush_samples[0]);
    output[1].extend(&flush_samples[1]);

    // Channels should be independent
    assert!(output[0].len() > 4000);
    assert!(output[1].len() > 4000);
}

#[test]
fn test_multichannel_streaming_reset() {
    let config = StftConfigF32::default_4096();
    let mut stft = MultiChannelStreamingStftF32::new(config.clone(), 2);

    let left = vec![0.0; 512];
    let right = vec![1.0; 512];

    let frames1 = stft.push_samples(&[&left[..], &right[..]]);

    stft.reset();

    let frames2 = stft.push_samples(&[&left[..], &right[..]]);

    // After reset, should produce same results
    assert_eq!(frames1[0].len(), frames2[0].len());
    assert_eq!(frames1[1].len(), frames2[1].len());
}

#[test]
fn test_multichannel_streaming_quad() {
    let config = StftConfigF32::default_4096();
    let mut stft = MultiChannelStreamingStftF32::new(config.clone(), 4);

    assert_eq!(stft.num_channels(), 4);

    let channels: Vec<Vec<f32>> = (0..4).map(|_| vec![0.0; 512]).collect();
    let channel_refs: Vec<&[f32]> = channels.iter().map(|c| c.as_slice()).collect();

    let frames = stft.push_samples(&channel_refs);
    assert_eq!(frames.len(), 4);
}

#[test]
#[should_panic(expected = "Expected 2 channels, got 3")]
fn test_multichannel_streaming_wrong_channel_count() {
    let config = StftConfigF32::default_4096();
    let mut stft = MultiChannelStreamingStftF32::new(config, 2);

    let ch1 = vec![0.0; 512];
    let ch2 = vec![0.0; 512];
    let ch3 = vec![0.0; 512];

    stft.push_samples(&[&ch1[..], &ch2[..], &ch3[..]]);
}

#[test]
#[should_panic(expected = "num_channels must be > 0")]
fn test_multichannel_streaming_zero_channels() {
    let config = StftConfigF32::default_4096();
    let _stft = MultiChannelStreamingStftF32::new(config, 0);
}
