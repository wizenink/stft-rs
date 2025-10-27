# Benchmarks

Comprehensive criterion benchmarks for stft-rs.

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark group
cargo bench batch_benches
cargo bench streaming_benches
cargo bench spectral_benches
cargo bench misc_benches
cargo bench multichannel_benches

# Run specific benchmark
cargo bench batch_stft_only
cargo bench allocation_strategies
cargo bench multichannel

# Save baseline for comparison
cargo bench --bench stft-bench -- --save-baseline main

# Compare against baseline
cargo bench --bench stft-bench -- --baseline main
```

## Benchmark Categories

### Batch Processing (`batch_benches`)

- **`batch_stft_only`** - Forward STFT only (1s, 5s, 10s signals)
- **`batch_roundtrip`** - Full STFT → iSTFT roundtrip (1s, 5s, 10s)
- **`batch_fft_sizes`** - Different FFT sizes (512, 1024, 2048, 4096, 8192)
- **`batch_window_types`** - Window functions (Hann, Hamming, Blackman)
- **`batch_reconstruction_modes`** - OLA vs WOLA reconstruction
- **`batch_allocation_strategies`** - Standard vs `_into` API

### Streaming Processing (`streaming_benches`)

- **`streaming_chunk_sizes`** - Different chunk sizes (128, 256, 512, 1024, 2048)
- **`allocation_strategies`** - Standard vs `_into` vs frame pool APIs

### Spectral Operations (`spectral_benches`)

- **`spectral_operations`** - All spectrum manipulation operations:
  - `magnitude_all_bins` - Calculate magnitude for all bins
  - `phase_all_bins` - Calculate phase for all bins
  - `frame_magnitudes` - Get all magnitudes for a frame
  - `frame_phases` - Get all phases for a frame
  - `apply_gain` - Apply gain to frequency range
  - `zero_bins` - Zero out frequency range
  - `apply_closure` - Custom processing with closure
  - `set_magnitude_phase_all` - Set from polar coordinates

- **`spectrum_frame_operations`** - SpectrumFrame operations:
  - `magnitudes` / `phases` - Get all values
  - `magnitude_single` / `phase_single` - Single bin access
  - `set_magnitude_phase` - Set single bin

### Miscellaneous (`misc_benches`)

- **`padding_modes`** - Reflect, Zero, Edge padding
- **`float_types`** - f32 vs f64 performance comparison

### Multi-Channel (`multichannel_benches`)

- **`multichannel`** - Multi-channel roundtrip (2, 4, 6, 8 channels)
  - Parallelized with rayon (enabled by default)
  - Benchmark with `--no-default-features` to test sequential processing

## Interpreting Results

Criterion outputs:
- **Time** - Wall-clock time per iteration
- **Throughput** - Samples processed per second
- **Change** - % change from previous run (if baseline exists)

HTML reports are generated in `target/criterion/`.

## Performance Notes

### Expected Performance (4096 FFT, 1024 hop, 5s audio @ 44.1kHz)

- **Batch STFT only**: ~20-40 ms
- **Batch roundtrip**: ~40-80 ms
- **Streaming (512 chunk)**: ~50-100 ms
- **Frame pool API**: ~2-5% faster than standard
- **f64 vs f32**: f64 ~10-20% slower

Performance heavily depends on:
1. FFT size (larger = slower per frame, but fewer frames)
2. rustfft implementation details
3. CPU architecture and cache sizes
4. Allocation overhead (reduced with `_into`/frame pool APIs)

## Adding New Benchmarks

```rust
fn bench_my_feature(c: &mut Criterion) {
    let mut group = c.benchmark_group("my_feature");

    // Setup data
    let audio = generate_signal(220500, 44100); // 5 seconds

    group.throughput(Throughput::Elements(audio.len() as u64));
    group.bench_function("my_test", |b| {
        b.iter(|| {
            // Code to benchmark
            black_box(&audio)
        });
    });

    group.finish();
}

// Add to criterion_group!
criterion_group!(my_benches, bench_my_feature);
```

## Optimization Tips

Based on benchmarks:
1. Use frame pool API for real-time streaming (predictable performance)
2. Larger FFT sizes are more efficient for throughput (but higher latency)
3. Batch mode is ~20-30% faster than streaming for offline processing
4. f32 is faster than f64 with minimal quality difference for audio
5. Spectral operations (magnitude/phase) are relatively cheap compared to FFT
