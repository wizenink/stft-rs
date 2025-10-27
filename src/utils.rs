/// Utility functions for signal processing and multi-channel audio
use num_traits::Float;

use crate::PadMode;

/// Apply padding to a signal.
/// Streaming applications should pad manually to match batch processing quality.
pub fn apply_padding<T: Float>(signal: &[T], pad_amount: usize, mode: PadMode) -> Vec<T> {
    let total_len = signal.len() + 2 * pad_amount;
    let mut padded = vec![T::zero(); total_len];

    padded[pad_amount..pad_amount + signal.len()].copy_from_slice(signal);

    match mode {
        PadMode::Reflect => {
            for i in 0..pad_amount {
                if i + 1 < signal.len() {
                    padded[pad_amount - 1 - i] = signal[i + 1];
                }
            }

            let n = signal.len();
            for i in 0..pad_amount {
                if n >= 2 && n - 2 >= i {
                    padded[pad_amount + n + i] = signal[n - 2 - i];
                }
            }
        }
        PadMode::Zero => {}
        PadMode::Edge => {
            if !signal.is_empty() {
                for i in 0..pad_amount {
                    padded[i] = signal[0];
                }
                for i in 0..pad_amount {
                    padded[pad_amount + signal.len() + i] = signal[signal.len() - 1];
                }
            }
        }
    }

    padded
}

// ============================================================================
// Multi-Channel Utilities
// ============================================================================

/// Deinterleave multi-channel audio data.
///
/// Converts interleaved format (e.g., `[L,R,L,R,L,R,...]` for stereo)
/// into separate channels (`vec![vec![L,L,L,...], vec![R,R,R,...]]`).
///
/// # Arguments
///
/// * `data` - Interleaved audio data
/// * `num_channels` - Number of channels
///
/// # Panics
///
/// Panics if `num_channels` is 0 or if `data.len()` is not divisible by `num_channels`.
///
/// # Example
///
/// ```
/// use stft_rs::deinterleave;
///
/// let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // L,R,L,R,L,R
/// let channels = deinterleave(&interleaved, 2);
///
/// assert_eq!(channels[0], vec![1.0, 3.0, 5.0]); // Left
/// assert_eq!(channels[1], vec![2.0, 4.0, 6.0]); // Right
/// ```
pub fn deinterleave<T: Float>(data: &[T], num_channels: usize) -> Vec<Vec<T>> {
    assert!(num_channels > 0, "num_channels must be greater than 0");
    assert_eq!(
        data.len() % num_channels,
        0,
        "data length ({}) must be divisible by num_channels ({})",
        data.len(),
        num_channels
    );

    if data.is_empty() {
        return vec![Vec::new(); num_channels];
    }

    let samples_per_channel = data.len() / num_channels;
    let mut channels = vec![Vec::with_capacity(samples_per_channel); num_channels];

    for (i, &sample) in data.iter().enumerate() {
        channels[i % num_channels].push(sample);
    }

    channels
}

/// Deinterleave multi-channel audio into a pre-allocated buffer (zero-allocation).
///
/// Converts interleaved format into separate channels without allocating new Vecs.
/// The output buffer must contain exactly `num_channels` `Vec<T>` elements.
/// Each Vec will be cleared and filled with samples.
///
/// # Arguments
///
/// * `data` - Interleaved audio data
/// * `num_channels` - Number of channels
/// * `output` - Pre-allocated output buffer with `num_channels` Vecs
///
/// # Panics
///
/// Panics if `num_channels` is 0, if `data.len()` is not divisible by `num_channels`,
/// or if `output.len() != num_channels`.
///
/// # Example
///
/// ```
/// use stft_rs::deinterleave_into;
///
/// let interleaved = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // L,R,L,R,L,R
/// let mut output = vec![Vec::new(), Vec::new()];
///
/// deinterleave_into(&interleaved, 2, &mut output);
///
/// assert_eq!(output[0], vec![1.0, 3.0, 5.0]); // Left
/// assert_eq!(output[1], vec![2.0, 4.0, 6.0]); // Right
/// ```
pub fn deinterleave_into<T: Float>(data: &[T], num_channels: usize, output: &mut [Vec<T>]) {
    assert!(num_channels > 0, "num_channels must be greater than 0");
    assert_eq!(
        output.len(),
        num_channels,
        "output must have exactly {} channels, got {}",
        num_channels,
        output.len()
    );
    assert_eq!(
        data.len() % num_channels,
        0,
        "data length ({}) must be divisible by num_channels ({})",
        data.len(),
        num_channels
    );

    if data.is_empty() {
        for channel in output.iter_mut() {
            channel.clear();
        }
        return;
    }

    let samples_per_channel = data.len() / num_channels;

    // Clear and reserve capacity
    for channel in output.iter_mut() {
        channel.clear();
        channel.reserve(samples_per_channel);
    }

    // Deinterleave
    for (i, &sample) in data.iter().enumerate() {
        output[i % num_channels].push(sample);
    }
}

/// Interleave multiple channels into a single buffer.
///
/// Converts separate channels (`vec![vec![L,L,L,...], vec![R,R,R,...]]`)
/// into interleaved format (e.g., `[L,R,L,R,L,R,...]` for stereo).
///
/// # Arguments
///
/// * `channels` - Vector of audio channels
///
/// # Panics
///
/// Panics if channels is empty or if channels have different lengths.
///
/// # Example
///
/// ```
/// use stft_rs::interleave;
///
/// let left = vec![1.0, 3.0, 5.0];
/// let right = vec![2.0, 4.0, 6.0];
/// let interleaved = interleave(&[left, right]);
///
/// assert_eq!(interleaved, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn interleave<T: Float>(channels: &[Vec<T>]) -> Vec<T> {
    assert!(!channels.is_empty(), "channels must not be empty");

    if channels[0].is_empty() {
        return Vec::new();
    }

    let num_channels = channels.len();
    let samples_per_channel = channels[0].len();

    // Validate all channels have same length
    for (i, channel) in channels.iter().enumerate() {
        assert_eq!(
            channel.len(),
            samples_per_channel,
            "Channel {} has length {}, expected {}",
            i,
            channel.len(),
            samples_per_channel
        );
    }

    let mut interleaved = Vec::with_capacity(samples_per_channel * num_channels);

    for sample_idx in 0..samples_per_channel {
        for channel in channels {
            interleaved.push(channel[sample_idx]);
        }
    }

    interleaved
}

/// Interleave multiple channels into a pre-allocated buffer (zero-allocation).
///
/// Converts separate channels into interleaved format without allocating.
/// The output buffer must have capacity for `samples_per_channel × num_channels`.
///
/// # Arguments
///
/// * `channels` - Slice of audio channels
/// * `output` - Pre-allocated output buffer (will be cleared and filled)
///
/// # Panics
///
/// Panics if channels is empty or if channels have different lengths.
///
/// # Example
///
/// ```
/// use stft_rs::interleave_into;
///
/// let left = vec![1.0, 3.0, 5.0];
/// let right = vec![2.0, 4.0, 6.0];
///
/// let mut output = Vec::with_capacity(6);
/// interleave_into(&[left, right], &mut output);
///
/// assert_eq!(output, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
pub fn interleave_into<T: Float>(channels: &[Vec<T>], output: &mut Vec<T>) {
    assert!(!channels.is_empty(), "channels must not be empty");

    if channels[0].is_empty() {
        output.clear();
        return;
    }

    let num_channels = channels.len();
    let samples_per_channel = channels[0].len();

    // Validate all channels have same length
    for (i, channel) in channels.iter().enumerate() {
        assert_eq!(
            channel.len(),
            samples_per_channel,
            "Channel {} has length {}, expected {}",
            i,
            channel.len(),
            samples_per_channel
        );
    }

    output.clear();
    output.reserve(samples_per_channel * num_channels);

    for sample_idx in 0..samples_per_channel {
        for channel in channels {
            output.push(channel[sample_idx]);
        }
    }
}
