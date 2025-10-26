use stft_rs::prelude::*;

fn main() {
    let config = StftConfig::<f32>::default_4096();

    println!("STFT Configuration:");
    println!("  FFT size: {}", config.fft_size);
    println!("  Hop size: {}", config.hop_size);
    println!("  Overlap: {:.1}%", config.overlap_percent());
    println!("  Frequency bins: {}", config.freq_bins());
    println!("  Reconstruction: {:?}", config.reconstruction_mode);
    println!();

    let stft = BatchStft::new(config.clone());
    let istft = BatchIstft::new(config);

    let sample_rate = 44100;
    let duration_secs = 1.0;
    let num_samples = (sample_rate as f32 * duration_secs) as usize;

    let audio: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate as f32;
            0.3 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
                + 0.2 * (2.0 * std::f32::consts::PI * 880.0 * t).sin()
                + 0.1 * (2.0 * std::f32::consts::PI * 1320.0 * t).sin()
        })
        .collect();

    println!(
        "Processing {} samples ({:.2} seconds)",
        num_samples, duration_secs
    );

    let spectrum = stft.process(&audio);
    println!(
        "STFT result: {} frames x {} frequency bins",
        spectrum.num_frames, spectrum.freq_bins
    );
    let mut magnitudes: Vec<(usize, f32)> = (0..spectrum.freq_bins)
        .map(|bin| {
            let c = spectrum.get_complex(0, bin);
            let mag = (c.re * c.re + c.im * c.im).sqrt();
            (bin, mag)
        })
        .collect();

    magnitudes.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("\nTop 5 frequency components:");
    let freq_resolution = sample_rate as f32 / 4096.0;
    for i in 0..5 {
        let (bin, mag) = magnitudes[i];
        let freq = bin as f32 * freq_resolution;
        println!("  Bin {}: {:.1} Hz (magnitude: {:.2})", bin, freq, mag);
    }

    let reconstructed = istft.process(&spectrum);
    println!("\nReconstructed {} samples", reconstructed.len());
    let min_len = audio.len().min(reconstructed.len());
    let mse: f32 = audio[..min_len]
        .iter()
        .zip(reconstructed[..min_len].iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f32>()
        / min_len as f32;

    let signal_power: f32 =
        audio[..min_len].iter().map(|x| x.powi(2)).sum::<f32>() / min_len as f32;
    let snr = 10.0 * (signal_power / mse).log10();

    println!("MSE: {:.2e}", mse);
    println!("SNR: {:.2} dB", snr);
}
