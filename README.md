![crates.io](https://img.shields.io/crates/v/stft-rs.svg)
# stft-rs

High-quality, streaming-friendly STFT/iSTFT implementation in Rust working with raw slices (`&[f32]`).


> [!CAUTION]
> This crate is a WIP, expect API changes and breakage until first stable version 

## Features

- **Batch Processing**: Process entire audio buffers at once
- **Streaming Support**: Incremental processing for real-time applications
- **High Quality**: >138 dB SNR reconstruction
- **Dual Reconstruction Modes**:
  - **OLA** (Overlap-Add): Optimal for spectral processing
  - **WOLA** (Weighted Overlap-Add): Standard implementation
- **Multiple Window Functions**: Hann, Hamming, Blackman
- **NOLA/COLA Validation**: Ensures reconstruction quality
- **Flexible Buffer Management**: Three allocation strategies from simple to zero-allocation
- **Multi-Channel Audio**: Process stereo, 5.1, 7.1+ with planar or interleaved formats
- **Generic Float Support**: Works with f32, f64, and other float types
- **Type Aliases**: Convenient aliases like `StftConfigF32`, `BatchStftF32` for cleaner code
- **Spectral Operations**: Built-in helpers for magnitude/phase manipulation, filtering, and custom processing
- **No External Tensor Libraries**: Works directly with slices

## Quick Start

```rust
use stft_rs::prelude::*;

let config = StftConfig::<f32>::default_4096();

let stft = BatchStft::new(config.clone());
let istft = BatchIstft::new(config);

let signal: Vec<f32> = vec![0.0; 44100];
let spectrum = stft.process(&signal);

// Manipulate spectrum here...

let reconstructed = istft.process(&spectrum);
```

### Type Aliases for Convenience

For cleaner code, use type aliases instead of specifying generic types:

```rust
use stft_rs::prelude::*;

// Instead of StftConfig::<f32>, use:
let config = StftConfigF32::default_4096();
let stft = BatchStftF32::new(config.clone());
let istft = BatchIstftF32::new(config);

// Available aliases:
// - StftConfigF32, StftConfigF64
// - BatchStftF32, BatchIstftF32, BatchStftF64, BatchIstftF64
// - StreamingStftF32, StreamingIstftF32, StreamingStftF64, StreamingIstftF64
// - SpectrumF32, SpectrumF64
// - SpectrumFrameF32, SpectrumFrameF64
```

## Prelude

For convenience, import commonly used types with:

```rust
use stft_rs::prelude::*;
```

This exports:
- Core types: `BatchStft`, `BatchIstft`, `StreamingStft`, `StreamingIstft`, `StftConfig`, `Spectrum`, `SpectrumFrame`
- Type aliases: `StftConfigF32/F64`, `BatchStftF32/F64`, `BatchIstftF32/F64`, `StreamingStftF32/F64`, `StreamingIstftF32/F64`, `SpectrumF32/F64`, `SpectrumFrameF32/F64`
- Enums: `ReconstructionMode`, `WindowType`, `PadMode`
- Utilities: `apply_padding`, `interleave`, `deinterleave`, `interleave_into`, `deinterleave_into`

## Batch vs Streaming

### Batch API (Stateless)

Best for: Processing entire files, offline processing, ML training

```rust
use stft_rs::prelude::*;

let config = StftConfig::default_4096();
let stft = BatchStft::new(config.clone());
let istft = BatchIstft::new(config);

let spectrum = stft.process(&signal);
let reconstructed = istft.process(&spectrum);
```

### Streaming API (Stateful)

Best for: Real-time audio, low-latency processing, incremental processing

```rust
use stft_rs::prelude::*;

let config = StftConfig::default_4096();
let mut stft = StreamingStft::new(config.clone());
let mut istft = StreamingIstft::new(config.clone());

let pad_amount = config.fft_size / 2;
let padded = apply_padding(&signal, pad_amount, PadMode::Reflect);

let mut output = Vec::new();
for chunk in padded.chunks(512) {
    let frames = stft.push_samples(chunk);
    for frame in frames {
        let samples = istft.push_frame(&frame);
        output.extend(samples);
    }
}

for frame in stft.flush() {
    output.extend(istft.push_frame(&frame));
}
output.extend(istft.flush());

// Remove padding: output[pad_amount..pad_amount + signal.len()]
```

**Note on Padding in Streaming Mode:**

- **Batch mode** automatically applies reflection padding internally for optimal quality
- **Streaming mode** requires manual padding for best results (>130 dB SNR)
- Without padding, edge effects reduce quality to ~40-60 dB SNR
- Use `apply_padding()` helper function or implement custom padding
- For truly real-time applications without pre-roll, accept the edge artifacts or use fade-in/fade-out

## Buffer Management

The library provides three allocation strategies for different performance requirements:

### Level 1: Simple API (Allocates on each call)

Best for: Prototyping, one-off processing, simplicity

```rust
// Each call allocates new Vec for frames/samples
let frames = stft.push_samples(chunk);
let samples = istft.push_frame(&frame);
```

### Level 2: Reusable Containers (`_into` methods)

Best for: Repeated processing, reduced allocator pressure

```rust
// Reuse outer Vec, but still allocates frame data
let mut frames = Vec::new();
let mut output = Vec::new();

loop {
    frames.clear();  // Keeps capacity
    stft.push_samples_into(chunk, &mut frames);

    for frame in &frames {
        istft.push_frame_into(frame, &mut output);
    }
}
```

**Batch mode:**
```rust
let mut spectrum = Spectrum::new(num_frames, freq_bins);
let mut output = Vec::new();

stft.process_into(&signal, &mut spectrum);
istft.process_into(&spectrum, &mut output);
```

### Level 3: Zero-Allocation Frame Pool (`_write` methods)

Best for: Real-time audio, hard real-time constraints, minimum latency variance

```rust
// Pre-allocate frame pool once
let max_frames = (chunk_size + config.hop_size - 1) / config.hop_size + 1;
let mut frame_pool = vec![SpectrumFrame::new(config.freq_bins()); max_frames];
let mut output = Vec::new();

loop {
    let mut pool_idx = 0;
    stft.push_samples_write(chunk, &mut frame_pool, &mut pool_idx);

    for i in 0..pool_idx {
        istft.push_frame_into(&frame_pool[i], &mut output);
    }
}
```


## Configuration

### Creating Custom Configurations

```rust
use stft_rs::prelude::*;

// OLA mode
let config = StftConfig::new(
    4096,
    1024,
    WindowType::Hann,
    ReconstructionMode::Ola
).expect("Valid configuration");

// WOLA mode
let config = StftConfig::new(
    2048,
    512,
    WindowType::Hamming,
    ReconstructionMode::Wola
).expect("Valid configuration");
```

### Window Functions

- **Hann**: Smooth frequency response, good general purpose
- **Hamming**: Slightly better frequency resolution
- **Blackman**: Lower side lobes, better for spectral analysis

### Reconstruction Modes

#### OLA (Overlap-Add)
- Window applied on forward transform only
- No window on inverse transform
- Normalizes by accumulated window energy: `sum(w)`
- **Use for**: Spectral processing, modification, filtering
- **Requires**: COLA (Constant Overlap-Add) condition

#### WOLA (Weighted Overlap-Add)
- Window applied on both forward and inverse transforms
- Normalizes by accumulated window squared: `sum(w²)`
- **Use for**: Standard analysis/resynthesis
- **Requires**: NOLA (Nonzero Overlap-Add) condition

## Spectral Processing

The library provides powerful helpers for frequency domain manipulation:

```rust
let mut spectrum = stft.process(&signal);

// Get magnitude and phase
let mag = spectrum.magnitude(frame, bin);
let phase = spectrum.phase(frame, bin);

// Set from magnitude and phase
spectrum.set_magnitude_phase(frame, bin, new_mag, new_phase);

// Get all magnitudes/phases for a frame
let magnitudes = spectrum.frame_magnitudes(frame);
let phases = spectrum.frame_phases(frame);

// Apply gain to frequency range
spectrum.apply_gain(100..200, 0.5); // Attenuate bins 100-200

// Zero out frequency range
spectrum.zero_bins(0..50); // Remove DC and low frequencies

// Custom processing with closure
spectrum.apply(|frame, bin, complex| {
    // Return modified complex value
    complex * gain_factor
});
```

### Examples

#### High-Pass Filter

```rust
let mut spectrum = stft.process(&signal);

// Zero out low frequencies (simple and clean!)
spectrum.zero_bins(0..100);

let filtered = istft.process(&spectrum);
```

#### Volume Control (Spectral Domain)

```rust
let mut spectrum = stft.process(&signal);

// Apply gain in magnitude/phase domain
for frame in 0..spectrum.num_frames {
    for bin in 0..spectrum.freq_bins {
        let mag = spectrum.magnitude(frame, bin);
        let phase = spectrum.phase(frame, bin);
        spectrum.set_magnitude_phase(frame, bin, mag * 0.5, phase);
    }
}

let quieter = istft.process(&spectrum);
```

#### Band-Pass Filter

```rust
let mut spectrum = stft.process(&signal);

// Keep only frequencies between 300 Hz and 3000 Hz
let sample_rate = 44100.0;
let freq_resolution = sample_rate / config.fft_size as f32;
let low_bin = (300.0 / freq_resolution) as usize;
let high_bin = (3000.0 / freq_resolution) as usize;

spectrum.zero_bins(0..low_bin);
spectrum.zero_bins(high_bin..spectrum.freq_bins);

let filtered = istft.process(&spectrum);
```

## Multi-Channel Audio

Process stereo, 5.1, or any channel count. Supports both planar and interleaved formats:

```rust
// Planar: separate Vec per channel
let left = vec![0.0; 44100];
let right = vec![0.0; 44100];
let spectra = stft.process_multichannel(&[left, right]);

// Interleaved: L,R,L,R...
let interleaved = vec![0.0; 88200];
let spectra = stft.process_interleaved(&interleaved, 2);

// Convert between formats
let channels = deinterleave(&interleaved, 2);
let interleaved = interleave(&channels);
```

See `examples/multichannel_stereo.rs` and `examples/multichannel_midside.rs` for more.

## Performance Characteristics

- **Batch Mode**: Optimized for throughput, minimal allocations
- **Streaming Mode**: Optimized for latency, incremental output
- **Memory**: Batch allocates once, streaming uses growing buffers
- **Latency**: Streaming introduces `fft_size - hop_size` samples of latency

### Typical Performance (4096 FFT, 1024 hop)

- **Reconstruction Quality**: >138 dB SNR
- **Algorithmic Latency**: 3072 samples (69.7 ms @ 44.1kHz)
- **Throughput**: Depends on FFT implementation (rustfft)

## Examples

Run the included examples:

```bash
# Basic batch processing
cargo run --example basic_usage

# Streaming processing with chunks
cargo run --example streaming_usage

# Spectral manipulation (filtering, time-varying processing)
cargo run --example spectral_processing

# Multi-channel stereo processing
cargo run --example multichannel_stereo

# Mid/side stereo width manipulation
cargo run --example multichannel_midside

# Advanced streaming with buffer reuse patterns
cargo run --example advanced_streaming

# Performance comparison of allocation strategies
cargo run --release --example buffer_reuse

# Type aliases usage demonstration
cargo run --example type_aliases

# Spectral operations (magnitude/phase, filtering)
cargo run --example spectral_operations
```

## Implementation Details

### Critical Design Decisions

1. **Flat Data Layout**: `Spectrum` stores data as `[real_all, imag_all]` for cache efficiency
2. **Padding**: Batch mode uses reflection padding (fft_size/2 on each side)
3. **Normalization**: Per-sample normalization by accumulated window energy
4. **Conjugate Symmetry**: Automatically handled in iSTFT for real signals
5. **Streaming Latency**: Samples released only when fully reconstructed (all overlaps complete)

### STFT Formula

```
X[k,n] = Σ x[n + m] * w[m] * e^(-j2πkm/N)
```

Where:
- `x[n]`: Input signal
- `w[m]`: Window function
- `N`: FFT size
- `k`: Frequency bin
- `n`: Frame index (hop positions)

### iSTFT Reconstruction

**OLA Mode:**
```
x[n] = Σ IFFT(X[k,m]) / Σ w[n - m*hop]
```

**WOLA Mode:**
```
x[n] = Σ IFFT(X[k,m]) * w[n - m*hop] / Σ w²[n - m*hop]
```

## Testing

Run the comprehensive test suite:

```bash
cargo test --lib

# With output
cargo test --lib -- --nocapture
```

Tests verify:
-  NOLA/COLA condition validation
-  Batch OLA roundtrip (>138 dB SNR)
-  Batch WOLA roundtrip (>138 dB SNR)
-  Streaming OLA roundtrip (>138 dB SNR)
-  Streaming WOLA roundtrip (>138 dB SNR)
-  Batch vs streaming consistency
-  All window functions (Hann, Hamming, Blackman)
-  Constant signal reconstruction
-  Padding modes (reflect, zero, edge)

## Dependencies

- `rustfft`: High-performance FFT implementation
- `ndarray`: Only for internal padding operations (minimal usage)

## License

[MIT]

## Contributing

Contributions welcome! Areas for improvement:
- [ ] Additional window functions (Kaiser, Gaussian)
- [ ] SIMD optimizations
- [ ] GPU acceleration support
- [ ] Streaming multi-channel support
- [ ] More padding modes
- [ ] Overlap-save mode
