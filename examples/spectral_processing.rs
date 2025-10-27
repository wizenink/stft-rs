use stft_rs::prelude::*;

fn main() {
    println!("Spectral Processing Demo\n");

    for mode in [ReconstructionMode::Ola, ReconstructionMode::Wola] {
        println!("Testing {:?} mode:", mode);

        let config = StftConfigBuilderF32::new()
            .fft_size(4096)
            .hop_size(1024)
            .window(WindowType::Hann)
            .reconstruction_mode(mode)
            .build()
            .expect("Valid configuration");

        let stft = BatchStft::new(config.clone());
        let istft = BatchIstft::new(config);

        let sample_rate = 44100.0;
        let duration = 1.0;
        let num_samples = (sample_rate * duration) as usize;

        let audio: Vec<f32> = (0..num_samples)
            .map(|i| {
                let t = i as f32 / sample_rate;
                0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 554.0 * t).sin()
                    + 0.3 * (2.0 * std::f32::consts::PI * 659.0 * t).sin()
            })
            .collect();

        let mut spectrum = stft.process(&audio);

        let cutoff_bin = (500.0 / sample_rate * 4096.0) as usize;
        println!(
            "  Applying high-pass filter at bin {} (~500 Hz)",
            cutoff_bin
        );

        for frame in 0..spectrum.num_frames {
            for bin in 0..cutoff_bin {
                let idx = frame * spectrum.freq_bins + bin;
                spectrum.data[idx] = 0.0;
                spectrum.data[spectrum.num_frames * spectrum.freq_bins + idx] = 0.0;
            }
        }

        let reconstructed = istft.process(&spectrum);
        let min_len = audio.len().min(reconstructed.len());
        let signal_power: f32 = audio[..min_len].iter().map(|x| x.powi(2)).sum::<f32>();
        let noise_power: f32 = audio[..min_len]
            .iter()
            .zip(reconstructed[..min_len].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>();

        let snr = 10.0 * (signal_power / noise_power).log10();
        println!("  SNR after filtering: {:.2} dB\n", snr);
    }

    println!("\nTime-varying spectral manipulation");

    let config = StftConfig::<f32>::default_4096();
    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config);

    let sample_rate = 44100.0;
    let duration = 2.0;
    let num_samples = (sample_rate * duration) as usize;

    let audio: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let freq = 200.0 + 800.0 * (t / duration);
            0.5 * (2.0 * std::f32::consts::PI * freq * t).sin()
        })
        .collect();

    let mut spectrum = stft.process(&audio);

    println!("Applying time-varying spectral sculpting...");

    for frame_idx in 0..spectrum.num_frames {
        let time_pos = frame_idx as f32 / spectrum.num_frames as f32;

        for bin in 0..spectrum.freq_bins {
            let freq = bin as f32 * sample_rate / 4096.0;

            let target_freq = 200.0 + 800.0 * time_pos;
            let freq_dist = ((freq - target_freq) / 100.0).abs();
            let gain = (-freq_dist * freq_dist / 2.0).exp();

            let idx = frame_idx * spectrum.freq_bins + bin;
            let imag_idx = spectrum.num_frames * spectrum.freq_bins + idx;

            spectrum.data[idx] *= gain;
            spectrum.data[imag_idx] *= gain;
        }
    }

    let processed = istft.process(&spectrum);

    println!("Original signal length: {} samples", audio.len());
    println!("Processed signal length: {} samples", processed.len());

    let original_energy: f32 = audio.iter().map(|x| x.powi(2)).sum();
    let processed_energy: f32 = processed.iter().map(|x| x.powi(2)).sum();

    println!("Original energy: {:.2}", original_energy);
    println!("Processed energy: {:.2}", processed_energy);
    println!(
        "Energy ratio: {:.2}%",
        (processed_energy / original_energy) * 100.0
    );
}
