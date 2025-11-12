# Mel Spectrogram Module

The `mel` module provides mel-scale frequency analysis, commonly used in speech recognition and audio processing. It converts linear frequency STFT bins into perceptually-motivated mel-scale bins.

## Quick Start

```rust
use stft_rs::prelude::*;

// Create STFT config
let stft_config = StftConfigF32::default_4096();
let stft = BatchStftF32::new(stft_config);

// Create mel spectrogram processor with defaults (80 mel bins, Slaney scale)
let mel_config = MelConfigF32::default();
let mel_proc = BatchMelSpectrogramF32::new(44100.0, 4096, &mel_config);

// Process audio to mel spectrogram
let signal: Vec<f32> = vec![0.0; 44100]; // 1 second @ 44.1kHz
let spectrum = stft.process(&signal);
let mel_spec = mel_proc.process(&spectrum);

// Convert to log-mel (dB scale)
let log_mel = mel_spec.to_db(None, None);

// Add delta features for speech recognition
let with_deltas = log_mel.with_deltas(Some(2));
// Result: 240 features per frame (80 mel + 80 delta + 80 delta-delta)
```

## Features

- **Mel Scale Variants**: HTK and Slaney (librosa-compatible, default)
- **80 Mel Bins Default**: Standard for speech/Whisper models
- **Log-Mel Conversion**: Convert to dB scale with configurable range
- **Delta Features**: First and second derivatives for speech recognition
- **Power/Magnitude**: Choose between power or magnitude spectrum
- **Batch & Streaming**: Process full spectrograms or individual frames
- **Multi-Channel**: Works with existing multi-channel STFT support
- **Zero-Copy Options**: Efficient processing with pre-allocated buffers

## Configuration

### MelConfig

```rust
use stft_rs::prelude::*;

// Default: 80 mels, Slaney scale, power spectrum
let config = MelConfigF32::default();

// Custom configuration
let config = MelConfigF32 {
    n_mels: 128,                      // Number of mel bands
    fmin: 0.0,                        // Minimum frequency (Hz)
    fmax: Some(8000.0),              // Maximum frequency (Hz), None = sr/2
    mel_scale: MelScale::Slaney,     // HTK or Slaney
    norm: MelNorm::Slaney,           // None or Slaney (area normalization)
    use_power: true,                 // true = power, false = magnitude
};
```

### Mel Scales

#### Slaney (Default)
- Linear below 1kHz, logarithmic above
- Compatible with librosa
- Better perceptual spacing

```rust
let config = MelConfigF32 {
    mel_scale: MelScale::Slaney,
    ..Default::default()
};
```

#### HTK
- Logarithmic throughout: `2595 * log10(1 + hz/700)`
- Used in HTK/Kaldi speech tools

```rust
let config = MelConfigF32 {
    mel_scale: MelScale::Htk,
    ..Default::default()
};
```

## Batch Processing

Process entire audio files at once:

```rust
use stft_rs::prelude::*;

let stft_config = StftConfigF32::default_4096();
let stft = BatchStftF32::new(stft_config);

let mel_config = MelConfigF32 {
    n_mels: 80,
    fmin: 0.0,
    fmax: Some(8000.0),
    mel_scale: MelScale::Slaney,
    norm: MelNorm::Slaney,
    use_power: true,
};

let mel_proc = BatchMelSpectrogramF32::new(44100.0, 4096, &mel_config);

// Process audio
let signal: Vec<f32> = vec![0.0; 44100];
let spectrum = stft.process(&signal);
let mel_spec = mel_proc.process(&spectrum);

// Or directly to dB scale
let log_mel = mel_proc.process_db(&spectrum, None, None);

println!("Mel spectrogram: {} frames x {} bins",
         mel_spec.num_frames, mel_spec.n_mels);
```

## Streaming Processing

Process audio frames in real-time:

```rust
use stft_rs::prelude::*;

let stft_config = StftConfigF32::default_4096();
let mut stft = StreamingStftF32::new(stft_config);

let mel_config = MelConfigF32::default();
let mel_proc = StreamingMelSpectrogramF32::new(44100.0, 4096, &mel_config);

// Process chunks
let chunk: Vec<f32> = vec![0.0; 8192];
let frames = stft.push_samples(&chunk);

// Convert each frame to mel
for frame in frames {
    let mel_frame = mel_proc.process_frame(&frame);
    // mel_frame: Vec<f32> of length 80
    println!("Mel features: {:?}", &mel_frame[..5]); // First 5 bins
}
```

### Zero-Copy Streaming

For real-time applications with minimal allocations:

```rust
let mut mel_buffer = vec![0.0f32; 80]; // Reuse this buffer

for frame in frames {
    let n = mel_proc.process_frame_into(&frame, &mut mel_buffer);
    // Process mel_buffer[..n] here
}
```

## MelSpectrum Operations

### Log-Mel (dB Scale)

Convert to logarithmic scale for better dynamic range:

```rust
let mel_spec = mel_proc.process(&spectrum);

// With defaults: amin=1e-10, top_db=80
let log_mel = mel_spec.to_db(None, None);

// Custom parameters
let log_mel = mel_spec.to_db(
    Some(1e-8),   // Minimum threshold to avoid log(0)
    Some(100.0)   // Maximum dB range
);
```

### Delta Features

Compute time derivatives for speech recognition:

```rust
let mel_spec = mel_proc.process(&spectrum);

// Delta (first derivative)
let delta = mel_spec.delta(Some(2));  // width=2 frames on each side

// Delta-delta (second derivative)
let delta_delta = mel_spec.delta_delta(Some(2));

// Concatenated features: [mel, delta, delta-delta]
let features = mel_spec.with_deltas(Some(2));
// Result: 240 features (80*3) per frame
```

**Delta Formula:**
```
delta[t] = Σ(n * (x[t+n] - x[t-n])) / (2 * Σ(n²))
where n ranges from 1 to width
```

### Accessing Data

```rust
let mel_spec = mel_proc.process(&spectrum);

// Access individual values
let value = mel_spec.get(frame_idx, mel_bin);
mel_spec.set(frame_idx, mel_bin, new_value);

// Access full frames
let frame = mel_spec.frame(frame_idx);  // &[f32]
let frame_mut = mel_spec.frame_mut(frame_idx);  // &mut [f32]

// Iterate over data
for frame_idx in 0..mel_spec.num_frames {
    let frame = mel_spec.frame(frame_idx);
    // Process frame...
}
```

### Custom Processing

```rust
// Apply function to all values
mel_spec.apply(|frame, mel_bin, value| {
    // Return modified value
    value * gain_factor
});
```

## Multi-Channel Audio

Process multi-channel audio by processing each channel separately:

```rust
use stft_rs::prelude::*;

let stft_config = StftConfigF32::default_4096();
let stft = BatchStftF32::new(stft_config);
let mel_config = MelConfigF32::default();
let mel_proc = BatchMelSpectrogramF32::new(44100.0, 4096, &mel_config);

// Process stereo
let left = vec![0.0f32; 44100];
let right = vec![0.0f32; 44100];
let channels = vec![left, right];

let spectra = stft.process_multichannel(&channels);

// Convert each channel to mel
let mel_spectra: Vec<MelSpectrumF32> = spectra
    .iter()
    .map(|spec| mel_proc.process(spec))
    .collect();

println!("Left channel: {} frames x {} mels",
         mel_spectra[0].num_frames, mel_spectra[0].n_mels);
println!("Right channel: {} frames x {} mels",
         mel_spectra[1].num_frames, mel_spectra[1].n_mels);
```

## Common Use Cases

### Speech Recognition (Whisper-style)

```rust
// Whisper uses 80 mel bins, 0-8kHz range, log-mel
let mel_config = MelConfigF32 {
    n_mels: 80,
    fmin: 0.0,
    fmax: Some(8000.0),
    mel_scale: MelScale::Slaney,
    norm: MelNorm::Slaney,
    use_power: true,
};

let mel_proc = BatchMelSpectrogramF32::new(16000.0, 400, &mel_config);
let mel_spec = mel_proc.process(&spectrum);
let log_mel = mel_spec.to_db(Some(1e-10), Some(80.0));

// For Whisper, you'd typically pad/trim to 3000 frames (30 seconds @ 50fps)
```

### Music Information Retrieval

```rust
// More mel bins, full frequency range
let mel_config = MelConfigF32 {
    n_mels: 128,
    fmin: 0.0,
    fmax: None,  // Use full Nyquist range
    mel_scale: MelScale::Slaney,
    norm: MelNorm::Slaney,
    use_power: true,
};

let mel_proc = BatchMelSpectrogramF32::new(44100.0, 4096, &mel_config);
let mel_spec = mel_proc.process(&spectrum);
let log_mel = mel_spec.to_db(None, None);
```

### Speaker Recognition

```rust
// Delta features for speaker characteristics
let mel_config = MelConfigF32::default();
let mel_proc = BatchMelSpectrogramF32::new(16000.0, 512, &mel_config);

let mel_spec = mel_proc.process(&spectrum);
let log_mel = mel_spec.to_db(None, None);

// Add deltas for speaker-specific dynamics
let features = log_mel.with_deltas(Some(2));
// Now: 240 features per frame for speaker modeling
```

## Type Aliases

For cleaner code, use type aliases:

```rust
use stft_rs::prelude::*;

// F32 aliases
let config = MelConfigF32::default();
let mel_proc = BatchMelSpectrogramF32::new(44100.0, 4096, &config);
let mel_spec: MelSpectrumF32 = mel_proc.process(&spectrum);

// F64 aliases
let config = MelConfigF64::default();
let mel_proc = BatchMelSpectrogramF64::new(44100.0, 4096, &config);
let mel_spec: MelSpectrumF64 = mel_proc.process(&spectrum);

// Available aliases:
// - MelConfigF32, MelConfigF64
// - MelFilterbankF32, MelFilterbankF64
// - MelSpectrumF32, MelSpectrumF64
// - BatchMelSpectrogramF32, BatchMelSpectrogramF64
// - StreamingMelSpectrogramF32, StreamingMelSpectrogramF64
```

## Performance Tips

1. **Reuse Buffers**: Use `process_frame_into()` in streaming mode to avoid allocations
2. **Batch Processing**: Process full files at once for better throughput
3. **Power vs Magnitude**: Power spectrum (default) is faster than magnitude
4. **Mel Bin Count**: Fewer bins = faster processing (but less frequency resolution)
5. **Rayon Parallelism**: Multi-channel processing is parallelized by default

## Implementation Details

### Mel Filterbank

- Triangular filters in mel-frequency domain
- Sparse matrix representation for efficiency
- Configurable normalization (Slaney area normalization recommended)
- Filters are precomputed at creation time

### Frequency Mappings

**Slaney Scale:**
- Linear: `mel = (hz - 0) / (200/3)` for hz < 1000
- Log: `mel = 15.0 + 27.0 * ln(hz/1000) / ln(6.4)` for hz >= 1000

**HTK Scale:**
- `mel = 2595 * log10(1 + hz/700)`

### Memory Layout

`MelSpectrum` stores data in row-major order:
```
[frame0_mel0, frame0_mel1, ..., frame0_mel79,
 frame1_mel0, frame1_mel1, ..., frame1_mel79,
 ...]
```

This layout is cache-efficient for frame-by-frame processing.

## Comparison with librosa

The default settings match librosa's mel spectrogram:

```python
# librosa (Python)
import librosa
mel_spec = librosa.feature.melspectrogram(
    y=signal, sr=44100, n_fft=4096, hop_length=1024,
    n_mels=80, fmin=0.0, fmax=8000.0,
    htk=False, norm='slaney', power=2.0
)
log_mel = librosa.power_to_db(mel_spec, ref=np.max, top_db=80)
```

```rust
// stft-rs (Rust)
let mel_config = MelConfigF32 {
    n_mels: 80,
    fmin: 0.0,
    fmax: Some(8000.0),
    mel_scale: MelScale::Slaney,  // htk=False
    norm: MelNorm::Slaney,         // norm='slaney'
    use_power: true,               // power=2.0
};

let mel_proc = BatchMelSpectrogramF32::new(44100.0, 4096, &mel_config);
let mel_spec = mel_proc.process(&spectrum);
let log_mel = mel_spec.to_db(None, Some(80.0));
```

## License

[MIT]
