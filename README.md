# stft-rs

High-quality, streaming-friendly STFT/iSTFT implementation in Rust working with raw slices (`&[f32]`).

## Features

- **Batch Processing**: Process entire audio buffers at once
- **Streaming Support**: Incremental processing for real-time applications
- **High Quality**: >138 dB SNR reconstruction
- **Dual Reconstruction Modes**:
  - **OLA** (Overlap-Add): Optimal for spectral processing
  - **WOLA** (Weighted Overlap-Add): Standard implementation
- **Multiple Window Functions**: Hann, Hamming, Blackman
- **NOLA/COLA Validation**: Ensures reconstruction quality
- **No External Tensor Libraries**: Works directly with slices

## Quick Start

```rust
use stft_rs::prelude::*;

let config = StftConfig::default_4096();

let stft = BatchStft::new(config.clone());
let istft = BatchIstft::new(config);

let signal: Vec<f32> = vec![0.0; 44100];
let spectrum = stft.process(&signal);

// Manipulate spectrum here...

let reconstructed = istft.process(&spectrum);
```

## Prelude

For convenience, import commonly used types with:

```rust
use stft_rs::prelude::*;
```

This exports:
- `BatchStft`, `BatchIstft`
- `StreamingStft`, `StreamingIstft`
- `StftConfig`
- `Spectrum`, `SpectrumFrame`
- `ReconstructionMode`, `WindowType`, `PadMode`
- `apply_padding`

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

The library provides easy access to manipulate spectrum data:

```rust
let mut spectrum = stft.process(&signal);

// Access individual frames and bins
for frame in 0..spectrum.num_frames {
    for bin in 0..spectrum.freq_bins {
        let complex = spectrum.get_complex(frame, bin);
        let magnitude = (complex.re * complex.re + complex.im * complex.im).sqrt();
        let phase = complex.im.atan2(complex.re);

        // Modify spectrum...
    }
}

// Or iterate over frames
for frame in spectrum.frames() {
    for complex_value in &frame.data {
        // Process each frequency bin
    }
}
```

### Examples

#### High-pass Filter

```rust
let mut spectrum = stft.process(&signal);
let cutoff_bin = 100;

for frame in 0..spectrum.num_frames {
    for bin in 0..cutoff_bin {
        let idx = frame * spectrum.freq_bins + bin;
        spectrum.data[idx] = 0.0; // Zero real part
        spectrum.data[spectrum.num_frames * spectrum.freq_bins + idx] = 0.0; // Zero imag part
    }
}

let filtered = istft.process(&spectrum);
```

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
- [ ] Multi-channel support
- [ ] More padding modes
- [ ] Overlap-save mode
