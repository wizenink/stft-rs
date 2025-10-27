use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use std::hint::black_box;
use stft_rs::prelude::*;

// ============================================================================
// Helper Functions
// ============================================================================

fn generate_signal(num_samples: usize, sample_rate: usize) -> Vec<f32> {
    (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
        })
        .collect()
}

// ============================================================================
// Batch Processing Benchmarks
// ============================================================================

fn bench_batch_stft_only(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_stft_only");

    for duration_secs in [1.0, 5.0, 10.0] {
        let sample_rate = 44100;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let audio = generate_signal(num_samples, sample_rate);

        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.0}s", duration_secs)),
            &audio,
            |b, audio| {
                let config = StftConfigF32::default_4096();
                let stft = BatchStftF32::new(config);
                b.iter(|| stft.process(black_box(audio)));
            },
        );
    }
    group.finish();
}

fn bench_batch_roundtrip(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_roundtrip");

    for duration_secs in [1.0, 5.0, 10.0] {
        let sample_rate = 44100;
        let num_samples = (sample_rate as f32 * duration_secs) as usize;
        let audio = generate_signal(num_samples, sample_rate);

        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:.0}s", duration_secs)),
            &audio,
            |b, audio| {
                let config = StftConfigF32::default_4096();
                let stft = BatchStftF32::new(config.clone());
                let istft = BatchIstftF32::new(config);
                b.iter(|| {
                    let spectrum = stft.process(black_box(audio));
                    istft.process(&spectrum)
                });
            },
        );
    }
    group.finish();
}

fn bench_batch_fft_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_fft_sizes");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);

    for fft_size in [512, 1024, 2048, 4096, 8192] {
        let hop_size = fft_size / 4;
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(BenchmarkId::from_parameter(fft_size), &audio, |b, audio| {
            // Use WOLA mode which is more flexible with hop sizes
            let config = StftConfig::new(
                fft_size,
                hop_size,
                WindowType::Hann,
                ReconstructionMode::Wola,
            )
            .unwrap();
            let stft = BatchStft::new(config);
            b.iter(|| stft.process(black_box(audio)));
        });
    }
    group.finish();
}

fn bench_batch_window_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_window_types");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);

    for window in [WindowType::Hann, WindowType::Hamming, WindowType::Blackman] {
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", window)),
            &audio,
            |b, audio| {
                let config = StftConfig::new(4096, 1024, window, ReconstructionMode::Ola).unwrap();
                let stft = BatchStft::new(config);
                b.iter(|| stft.process(black_box(audio)));
            },
        );
    }
    group.finish();
}

fn bench_batch_reconstruction_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_reconstruction_modes");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);

    for mode in [ReconstructionMode::Ola, ReconstructionMode::Wola] {
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", mode)),
            &audio,
            |b, audio| {
                let config = StftConfig::new(4096, 1024, WindowType::Hann, mode).unwrap();
                let stft = BatchStft::new(config.clone());
                let istft = BatchIstft::new(config);
                b.iter(|| {
                    let spectrum = stft.process(black_box(audio));
                    istft.process(&spectrum)
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Streaming Processing Benchmarks
// ============================================================================

fn bench_streaming_chunk_sizes(c: &mut Criterion) {
    let mut group = c.benchmark_group("streaming_chunk_sizes");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);

    let config = StftConfigF32::default_4096();
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&audio, pad_amount, PadMode::Reflect);

    for chunk_size in [128, 256, 512, 1024, 2048] {
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &padded,
            |b, padded| {
                b.iter(|| {
                    let mut stft = StreamingStftF32::new(config.clone());
                    let mut istft = StreamingIstftF32::new(config.clone());
                    let mut output = Vec::new();

                    for chunk in padded.chunks(chunk_size) {
                        let frames = stft.push_samples(black_box(chunk));
                        for frame in frames {
                            output.extend(istft.push_frame(&frame));
                        }
                    }

                    for frame in stft.flush() {
                        output.extend(istft.push_frame(&frame));
                    }
                    output.extend(istft.flush());
                    output
                });
            },
        );
    }
    group.finish();
}

// ============================================================================
// Allocation Strategy Benchmarks
// ============================================================================

fn bench_allocation_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("allocation_strategies");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);

    let config = StftConfigF32::default_4096();
    let pad_amount = config.fft_size / 2;
    let padded = apply_padding(&audio, pad_amount, PadMode::Reflect);
    let chunk_size = 512;

    // Standard API
    group.throughput(Throughput::Elements(num_samples as u64));
    group.bench_function("standard_api", |b| {
        b.iter(|| {
            let mut stft = StreamingStftF32::new(config.clone());
            let mut istft = StreamingIstftF32::new(config.clone());
            let mut output = Vec::new();

            for chunk in padded.chunks(chunk_size) {
                let frames = stft.push_samples(black_box(chunk));
                for frame in frames {
                    output.extend(istft.push_frame(&frame));
                }
            }
            output
        });
    });

    // _into API
    group.bench_function("into_api", |b| {
        b.iter(|| {
            let mut stft = StreamingStftF32::new(config.clone());
            let mut istft = StreamingIstftF32::new(config.clone());
            let mut frames = Vec::new();
            let mut output = Vec::new();

            for chunk in padded.chunks(chunk_size) {
                frames.clear();
                stft.push_samples_into(black_box(chunk), &mut frames);
                for frame in &frames {
                    istft.push_frame_into(frame, &mut output);
                }
            }
            output
        });
    });

    // Frame pool API
    group.bench_function("frame_pool", |b| {
        b.iter(|| {
            let mut stft = StreamingStftF32::new(config.clone());
            let mut istft = StreamingIstftF32::new(config.clone());
            let max_frames = (chunk_size + config.hop_size - 1) / config.hop_size + 1;
            let mut frame_pool = vec![SpectrumFrameF32::new(config.freq_bins()); max_frames];
            let mut output = Vec::new();

            for chunk in padded.chunks(chunk_size) {
                let mut pool_idx = 0;
                stft.push_samples_write(black_box(chunk), &mut frame_pool, &mut pool_idx);
                for i in 0..pool_idx {
                    istft.push_frame_into(&frame_pool[i], &mut output);
                }
            }
            output
        });
    });

    group.finish();
}

fn bench_batch_allocation_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_allocation_strategies");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);

    let config = StftConfigF32::default_4096();

    // Standard API
    group.throughput(Throughput::Elements(num_samples as u64));
    group.bench_function("standard_api", |b| {
        let stft = BatchStftF32::new(config.clone());
        let istft = BatchIstftF32::new(config.clone());
        b.iter(|| {
            let spectrum = stft.process(black_box(&audio));
            istft.process(&spectrum)
        });
    });

    // _into API
    group.bench_function("into_api", |b| {
        let stft = BatchStftF32::new(config.clone());
        let istft = BatchIstftF32::new(config.clone());

        let pad_amount = config.fft_size / 2;
        let padded_len = audio.len() + 2 * pad_amount;
        let num_frames = if padded_len >= config.fft_size {
            (padded_len - config.fft_size) / config.hop_size + 1
        } else {
            0
        };
        let mut spectrum = SpectrumF32::new(num_frames, config.freq_bins());
        let mut output = Vec::new();

        b.iter(|| {
            stft.process_into(black_box(&audio), &mut spectrum);
            output.clear();
            istft.process_into(&spectrum, &mut output);
            output.len()
        });
    });

    group.finish();
}

// ============================================================================
// Spectral Operations Benchmarks
// ============================================================================

fn bench_spectral_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_operations");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);

    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config.clone());
    let spectrum = stft.process(&audio);

    // Magnitude calculation
    group.bench_function("magnitude_all_bins", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for frame in 0..black_box(&spectrum).num_frames {
                for bin in 0..spectrum.freq_bins {
                    sum += spectrum.magnitude(frame, bin);
                }
            }
            sum
        });
    });

    // Phase calculation
    group.bench_function("phase_all_bins", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for frame in 0..black_box(&spectrum).num_frames {
                for bin in 0..spectrum.freq_bins {
                    sum += spectrum.phase(frame, bin);
                }
            }
            sum
        });
    });

    // Frame magnitudes
    group.bench_function("frame_magnitudes", |b| {
        b.iter(|| {
            let frame = black_box(spectrum.num_frames / 2);
            spectrum.frame_magnitudes(frame)
        });
    });

    // Frame phases
    group.bench_function("frame_phases", |b| {
        b.iter(|| {
            let frame = black_box(spectrum.num_frames / 2);
            spectrum.frame_phases(frame)
        });
    });

    // Apply gain
    group.bench_function("apply_gain", |b| {
        let mut spec = spectrum.clone();
        b.iter(|| {
            spec.apply_gain(black_box(100..200), black_box(0.5));
        });
    });

    // Zero bins
    group.bench_function("zero_bins", |b| {
        let mut spec = spectrum.clone();
        b.iter(|| {
            spec.zero_bins(black_box(0..100));
        });
    });

    // Apply closure
    group.bench_function("apply_closure", |b| {
        let mut spec = spectrum.clone();
        b.iter(|| {
            spec.apply(|_frame, _bin, c| c * black_box(0.9));
        });
    });

    // Set magnitude/phase
    group.bench_function("set_magnitude_phase_all", |b| {
        let mut spec = spectrum.clone();
        b.iter(|| {
            for frame in 0..spec.num_frames {
                for bin in 0..spec.freq_bins.min(100) {
                    spec.set_magnitude_phase(frame, bin, black_box(1.0), black_box(0.5));
                }
            }
        });
    });

    group.finish();
}

fn bench_spectrum_frame_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectrum_frame_operations");

    let config = StftConfigF32::default_4096();
    let mut frame = SpectrumFrameF32::new(config.freq_bins());

    // Fill with test data
    for i in 0..frame.freq_bins {
        frame.data[i] = rustfft::num_complex::Complex::new((i as f32).sin(), (i as f32).cos());
    }

    group.bench_function("magnitudes", |b| {
        b.iter(|| black_box(&frame).magnitudes());
    });

    group.bench_function("phases", |b| {
        b.iter(|| black_box(&frame).phases());
    });

    group.bench_function("magnitude_single", |b| {
        b.iter(|| black_box(&frame).magnitude(black_box(100)));
    });

    group.bench_function("phase_single", |b| {
        b.iter(|| black_box(&frame).phase(black_box(100)));
    });

    group.bench_function("set_magnitude_phase", |b| {
        let mut f = frame.clone();
        b.iter(|| {
            f.set_magnitude_phase(black_box(100), black_box(1.5), black_box(0.5));
        });
    });

    group.finish();
}

// ============================================================================
// Padding Benchmarks
// ============================================================================

fn bench_padding_modes(c: &mut Criterion) {
    let mut group = c.benchmark_group("padding_modes");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;
    let audio = generate_signal(num_samples, sample_rate);
    let pad_amount = 2048;

    for mode in [PadMode::Reflect, PadMode::Zero, PadMode::Edge] {
        group.throughput(Throughput::Elements(num_samples as u64));
        group.bench_with_input(
            BenchmarkId::from_parameter(format!("{:?}", mode)),
            &audio,
            |b, audio| {
                b.iter(|| apply_padding(black_box(audio), black_box(pad_amount), mode));
            },
        );
    }

    group.finish();
}

// ============================================================================
// Float Type Benchmarks (f32 vs f64)
// ============================================================================

fn bench_float_types(c: &mut Criterion) {
    let mut group = c.benchmark_group("float_types");

    let sample_rate = 44100;
    let duration_secs = 5.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    // f32 benchmark
    let audio_f32 = generate_signal(num_samples, sample_rate);
    group.throughput(Throughput::Elements(num_samples as u64));
    group.bench_function("f32_roundtrip", |b| {
        let config = StftConfigF32::default_4096();
        let stft = BatchStftF32::new(config.clone());
        let istft = BatchIstftF32::new(config);
        b.iter(|| {
            let spectrum = stft.process(black_box(&audio_f32));
            istft.process(&spectrum)
        });
    });

    // f64 benchmark
    let audio_f64: Vec<f64> = audio_f32.iter().map(|&x| x as f64).collect();
    group.bench_function("f64_roundtrip", |b| {
        let config = StftConfigF64::default_4096();
        let stft = BatchStftF64::new(config.clone());
        let istft = BatchIstftF64::new(config);
        b.iter(|| {
            let spectrum = stft.process(black_box(&audio_f64));
            istft.process(&spectrum)
        });
    });

    group.finish();
}

// ============================================================================
// Criterion Groups
// ============================================================================

use std::time::Duration;

fn criterion_config() -> Criterion {
    Criterion::default()
        .measurement_time(Duration::from_secs(10))
        .sample_size(50)
        .warm_up_time(Duration::from_secs(3))
}

criterion_group! {
    name = batch_benches;
    config = criterion_config();
    targets = bench_batch_stft_only,
        bench_batch_roundtrip,
        bench_batch_fft_sizes,
        bench_batch_window_types,
        bench_batch_reconstruction_modes,
        bench_batch_allocation_strategies,
}

criterion_group! {
    name = streaming_benches;
    config = criterion_config();
    targets = bench_streaming_chunk_sizes,
        bench_allocation_strategies,
}

criterion_group! {
    name = spectral_benches;
    config = criterion_config();
    targets = bench_spectral_operations,
        bench_spectrum_frame_operations,
}

criterion_group! {
    name = misc_benches;
    config = criterion_config();
    targets = bench_padding_modes,
        bench_float_types,
}

criterion_main!(
    batch_benches,
    streaming_benches,
    spectral_benches,
    misc_benches,
);
