use criterion::{Criterion, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};

use stft_rs::prelude::*;

pub fn batched_stft_bench(c: &mut Criterion) {
    let config = StftConfig::default_4096();
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config);

    let sample_rate = 44100;
    let duration_secs = 10.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    let audio: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
        })
        .collect();

    c.bench_function("stft_batch", |b| b.iter(|| stft.process(black_box(&audio))));
}

criterion_group!(benches, batched_stft_bench);
criterion_main!(benches);
