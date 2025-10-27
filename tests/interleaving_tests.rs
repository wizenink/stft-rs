use stft_rs::prelude::*;

// ============================================================================
// Basic deinterleave tests
// ============================================================================

#[test]
fn test_deinterleave_stereo() {
    let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // L,R,L,R,L,R
    let channels = deinterleave(&interleaved, 2);

    assert_eq!(channels.len(), 2);
    assert_eq!(channels[0], vec![1.0, 3.0, 5.0]); // Left
    assert_eq!(channels[1], vec![2.0, 4.0, 6.0]); // Right
}

#[test]
fn test_deinterleave_multichannel() {
    let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]; // 3 channels, 3 samples each
    let channels = deinterleave(&interleaved, 3);

    assert_eq!(channels.len(), 3);
    assert_eq!(channels[0], vec![1.0, 4.0, 7.0]);
    assert_eq!(channels[1], vec![2.0, 5.0, 8.0]);
    assert_eq!(channels[2], vec![3.0, 6.0, 9.0]);
}

#[test]
fn test_deinterleave_empty() {
    let empty: Vec<f32> = vec![];
    let channels = deinterleave(&empty, 2);

    assert_eq!(channels.len(), 2);
    assert!(channels[0].is_empty());
    assert!(channels[1].is_empty());
}

#[test]
#[should_panic(expected = "num_channels must be greater than 0")]
fn test_deinterleave_zero_channels() {
    let data = vec![1.0, 2.0, 3.0];
    deinterleave(&data, 0);
}

#[test]
#[should_panic(expected = "must be divisible by")]
fn test_deinterleave_misaligned() {
    let data = vec![1.0, 2.0, 3.0]; // 3 samples, 2 channels = not divisible
    deinterleave(&data, 2);
}

// ============================================================================
// Basic interleave tests
// ============================================================================

#[test]
fn test_interleave_stereo() {
    let left = vec![1.0, 3.0, 5.0];
    let right = vec![2.0, 4.0, 6.0];
    let interleaved = interleave(&[left, right]);

    assert_eq!(interleaved, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_interleave_multichannel() {
    let ch1 = vec![1.0, 4.0, 7.0];
    let ch2 = vec![2.0, 5.0, 8.0];
    let ch3 = vec![3.0, 6.0, 9.0];
    let interleaved = interleave(&[ch1, ch2, ch3]);

    assert_eq!(
        interleaved,
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]
    );
}

#[test]
fn test_interleave_empty() {
    let ch1: Vec<f32> = vec![];
    let ch2: Vec<f32> = vec![];
    let interleaved = interleave(&[ch1, ch2]);

    assert!(interleaved.is_empty());
}

#[test]
#[should_panic(expected = "channels must not be empty")]
fn test_interleave_no_channels() {
    let empty: Vec<Vec<f32>> = vec![];
    interleave(&empty);
}

#[test]
#[should_panic(expected = "expected")]
fn test_interleave_mismatched_lengths() {
    let ch1 = vec![1.0, 2.0, 3.0];
    let ch2 = vec![4.0, 5.0]; // Different length
    interleave(&[ch1, ch2]);
}

// ============================================================================
// Roundtrip tests
// ============================================================================

#[test]
fn test_roundtrip_stereo() {
    let original_left = vec![1.0, 3.0, 5.0, 7.0];
    let original_right = vec![2.0, 4.0, 6.0, 8.0];
    let original_channels = vec![original_left.clone(), original_right.clone()];

    let interleaved = interleave(&original_channels);
    let channels = deinterleave(&interleaved, 2);

    assert_eq!(channels[0], original_left);
    assert_eq!(channels[1], original_right);
}

#[test]
fn test_roundtrip_multichannel() {
    let original = vec![
        vec![1.0, 5.0, 9.0],
        vec![2.0, 6.0, 10.0],
        vec![3.0, 7.0, 11.0],
        vec![4.0, 8.0, 12.0],
    ];

    let interleaved = interleave(&original);
    let channels = deinterleave(&interleaved, 4);

    for (i, channel) in channels.iter().enumerate() {
        assert_eq!(channel, &original[i]);
    }
}

// ============================================================================
// interleave_into tests (zero-allocation)
// ============================================================================

#[test]
fn test_interleave_into_stereo() {
    let left = vec![1.0, 3.0, 5.0];
    let right = vec![2.0, 4.0, 6.0];

    let mut output = Vec::new();
    interleave_into(&[left, right], &mut output);

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_interleave_into_with_capacity() {
    let left = vec![1.0, 3.0, 5.0];
    let right = vec![2.0, 4.0, 6.0];

    let mut output = Vec::with_capacity(6);
    interleave_into(&[left, right], &mut output);

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    assert_eq!(output.capacity(), 6); // Should not reallocate
}

#[test]
fn test_interleave_into_reuse() {
    let ch1 = vec![1.0, 4.0];
    let ch2 = vec![2.0, 5.0];
    let ch3 = vec![3.0, 6.0];

    let mut output = vec![99.0, 99.0, 99.0]; // Pre-existing data

    interleave_into(&[ch1, ch2, ch3], &mut output);

    assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn test_interleave_into_empty() {
    let ch1: Vec<f32> = vec![];
    let ch2: Vec<f32> = vec![];

    let mut output = vec![99.0];
    interleave_into(&[ch1, ch2], &mut output);

    assert!(output.is_empty());
}

#[test]
#[should_panic(expected = "channels must not be empty")]
fn test_interleave_into_no_channels() {
    let empty: Vec<Vec<f32>> = vec![];
    let mut output = Vec::new();
    interleave_into(&empty, &mut output);
}

#[test]
#[should_panic(expected = "Channel 1 has length")]
fn test_interleave_into_mismatched_lengths() {
    let ch1 = vec![1.0, 2.0, 3.0];
    let ch2 = vec![4.0, 5.0]; // Different length!
    let mut output = Vec::new();
    interleave_into(&[ch1, ch2], &mut output);
}

// ============================================================================
// deinterleave_into tests (zero-allocation)
// ============================================================================

#[test]
fn test_deinterleave_into_stereo() {
    let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![Vec::new(), Vec::new()];

    deinterleave_into(&interleaved, 2, &mut output);

    assert_eq!(output[0], vec![1.0, 3.0, 5.0]);
    assert_eq!(output[1], vec![2.0, 4.0, 6.0]);
}

#[test]
fn test_deinterleave_into_with_capacity() {
    let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let mut output = vec![Vec::with_capacity(3), Vec::with_capacity(3)];

    deinterleave_into(&interleaved, 2, &mut output);

    assert_eq!(output[0], vec![1.0, 3.0, 5.0]);
    assert_eq!(output[1], vec![2.0, 4.0, 6.0]);
    assert_eq!(output[0].capacity(), 3); // Should not reallocate
    assert_eq!(output[1].capacity(), 3);
}

#[test]
fn test_deinterleave_into_reuse() {
    let interleaved = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![
        vec![99.0, 99.0], // Pre-existing data
        vec![88.0, 88.0],
    ];

    deinterleave_into(&interleaved, 2, &mut output);

    assert_eq!(output[0], vec![1.0, 3.0]);
    assert_eq!(output[1], vec![2.0, 4.0]);
}

#[test]
fn test_deinterleave_into_empty() {
    let empty: Vec<f32> = vec![];
    let mut output = vec![vec![99.0], vec![88.0]];

    deinterleave_into(&empty, 2, &mut output);

    assert!(output[0].is_empty());
    assert!(output[1].is_empty());
}

#[test]
#[should_panic(expected = "output must have exactly")]
fn test_deinterleave_into_wrong_output_size() {
    let interleaved = vec![1.0, 2.0, 3.0, 4.0];
    let mut output = vec![Vec::new()]; // Only 1 channel, need 2
    deinterleave_into(&interleaved, 2, &mut output);
}

#[test]
#[should_panic(expected = "must be divisible by")]
fn test_deinterleave_into_misaligned() {
    let data = vec![1.0, 2.0, 3.0]; // 3 samples, 2 channels = not divisible
    let mut output = vec![Vec::new(), Vec::new()];
    deinterleave_into(&data, 2, &mut output);
}

// ============================================================================
// Roundtrip _into tests
// ============================================================================

#[test]
fn test_roundtrip_into() {
    let original = vec![
        vec![1.0, 4.0, 7.0],
        vec![2.0, 5.0, 8.0],
        vec![3.0, 6.0, 9.0],
    ];

    let mut interleaved = Vec::new();
    interleave_into(&original, &mut interleaved);

    let mut channels = vec![Vec::new(), Vec::new(), Vec::new()];
    deinterleave_into(&interleaved, 3, &mut channels);

    assert_eq!(channels, original);
}

// ============================================================================
// Additional tests
// ============================================================================

#[test]
fn test_interleave_many_channels() {
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
}

#[test]
fn test_single_channel() {
    // Single channel should work
    let mono = vec![1.0, 2.0, 3.0];
    let interleaved = interleave(&[mono.clone()]);

    assert_eq!(interleaved, mono);

    let deinterleaved = deinterleave(&interleaved, 1);
    assert_eq!(deinterleaved.len(), 1);
    assert_eq!(deinterleaved[0], mono);
}

#[test]
fn test_interleave_into_performance() {
    // Test that pre-allocated buffer doesn't reallocate
    let channels: Vec<Vec<f32>> = (0..8).map(|i| vec![i as f32; 1000]).collect();

    let mut output = Vec::with_capacity(8000);
    let capacity_before = output.capacity();

    interleave_into(&channels, &mut output);

    assert_eq!(output.len(), 8000);
    assert_eq!(output.capacity(), capacity_before); // No reallocation
}

#[test]
fn test_deinterleave_into_performance() {
    // Test that pre-allocated buffers don't reallocate
    let interleaved: Vec<f32> = (0..8000).map(|i| i as f32).collect();

    let mut output: Vec<Vec<f32>> = (0..8).map(|_| Vec::with_capacity(1000)).collect();
    let capacities_before: Vec<_> = output.iter().map(|v| v.capacity()).collect();

    deinterleave_into(&interleaved, 8, &mut output);

    for (i, channel) in output.iter().enumerate() {
        assert_eq!(channel.len(), 1000);
        assert_eq!(channel.capacity(), capacities_before[i]); // No reallocation
    }
}
