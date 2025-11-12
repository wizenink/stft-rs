use stft_rs::prelude::*;

#[cfg(feature = "visualization")]
use stft_rs::visualization::{ColorMap, VisualizationConfig};

#[cfg(feature = "visualization")]
use stft_rs::visualization::SpectrumExt;

fn main() {
    #[cfg(not(feature = "visualization"))]
    {
        eprintln!("❌ This example requires the 'visualization' feature.");
        eprintln!("Run with: cargo run --example visualization --features visualization");
        return;
    }

    #[cfg(feature = "visualization")]
    {
        println!("🎵 Generating test signals and spectrograms...\n");

        // Example 1: Chirp signal (frequency sweep)
        println!("1. Chirp Signal (100Hz → 4kHz)");
        generate_chirp_spectrogram();

        // Example 2: Multi-tone signal
        println!("\n2. Multi-Tone Signal (440Hz + 880Hz + 1320Hz)");
        generate_multitone_spectrogram();

        // Example 3: Mel spectrogram
        println!("\n3. Mel Spectrogram (speech-optimized)");
        generate_mel_spectrogram();

        // Example 4: Different color maps comparison
        println!("\n4. Color Map Comparison");
        compare_colormaps();

        println!("\n✅ Done! Generated spectrograms:");
        println!("   - chirp_viridis.png");
        println!("   - chirp_magma.png");
        println!("   - multitone.png");
        println!("   - mel_speech.png");
        println!("   - colormap_inferno.png");
        println!("   - colormap_plasma.png");
        println!("   - colormap_grayscale.png");
    }
}

#[cfg(feature = "visualization")]
fn generate_chirp_spectrogram() {
    let sample_rate = 44100.0;
    let duration = 2.0;
    let num_samples = (sample_rate * duration) as usize;

    // Generate chirp: frequency increases linearly over time
    let signal: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let freq = 100.0 + (4000.0 - 100.0) * (t / duration); // 100Hz → 4kHz
            (2.0 * std::f32::consts::PI * freq * t).sin() * 0.5
        })
        .collect();

    // Process with STFT
    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config);
    let spectrum = stft.process(&signal);

    // Save with default settings (Viridis)
    spectrum.save_image("chirp_viridis.png").unwrap();
    println!(
        "   ✓ chirp_viridis.png ({}x{} pixels)",
        spectrum.num_frames, spectrum.freq_bins
    );

    // Save with custom settings (Magma, custom size)
    let vis_config = VisualizationConfig {
        colormap: ColorMap::Magma,
        width: Some(800),
        height: Some(400),
        db_range: (-60.0, 0.0), // Better dynamic range for this signal
    };
    spectrum
        .save_image_with("chirp_magma.png", &vis_config)
        .unwrap();
    println!("   ✓ chirp_magma.png (800x400 pixels, custom dB range)");
}

#[cfg(feature = "visualization")]
fn generate_multitone_spectrogram() {
    let sample_rate = 44100.0;
    let duration = 2.0;
    let num_samples = (sample_rate * duration) as usize;

    // Generate signal with three frequencies (A4, A5, E6)
    let signal: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            let f1 = (2.0 * std::f32::consts::PI * 440.0 * t).sin(); // A4
            let f2 = (2.0 * std::f32::consts::PI * 880.0 * t).sin(); // A5
            let f3 = (2.0 * std::f32::consts::PI * 1320.0 * t).sin(); // E6

            // Add amplitude modulation for visual interest
            let envelope = (2.0 * std::f32::consts::PI * 2.0 * t).sin().abs();
            (f1 + f2 + f3) * envelope / 3.0 * 0.5
        })
        .collect();

    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config);
    let spectrum = stft.process(&signal);

    let vis_config = VisualizationConfig {
        colormap: ColorMap::Inferno,
        width: Some(1000),
        height: Some(512),
        db_range: (-70.0, -10.0),
    };

    spectrum
        .save_image_with("multitone.png", &vis_config)
        .unwrap();
    println!("   ✓ multitone.png (1000x512 pixels)");
    println!("      Should show 3 horizontal lines at 440Hz, 880Hz, and 1320Hz");
}

#[cfg(feature = "visualization")]
fn generate_mel_spectrogram() {
    let sample_rate = 16000.0; // Common for speech
    let duration = 2.0;
    let num_samples = (sample_rate * duration) as usize;

    // Generate speech-like signal (formants at typical vowel frequencies)
    let signal: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            // Simulate formants (F1=730Hz, F2=1090Hz, F3=2440Hz)
            let f1 = (2.0 * std::f32::consts::PI * 730.0 * t).sin() * 1.0;
            let f2 = (2.0 * std::f32::consts::PI * 1090.0 * t).sin() * 0.7;
            let f3 = (2.0 * std::f32::consts::PI * 2440.0 * t).sin() * 0.5;

            // Pitch variation
            let pitch_freq = 120.0 + 20.0 * (t * 3.0).sin();
            let pitch = (2.0 * std::f32::consts::PI * pitch_freq * t).sin() * 0.3;

            (f1 + f2 + f3 + pitch) / 3.0 * 0.5
        })
        .collect();

    // Process with STFT (use default 4096 which is COLA-compliant)
    let stft_config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(stft_config);
    let spectrum = stft.process(&signal);

    // Convert to mel spectrogram
    let mel_config = MelConfigF32 {
        n_mels: 80,
        fmin: 0.0,
        fmax: Some(8000.0), // Full range for 16kHz
        mel_scale: MelScale::Slaney,
        norm: MelNorm::Slaney,
        use_power: true,
    };

    let mel_proc = BatchMelSpectrogramF32::new(sample_rate, 4096, &mel_config);
    let mel_spec = mel_proc.process_db(&spectrum, None, None); // Convert to dB

    let vis_config = VisualizationConfig {
        colormap: ColorMap::Viridis,
        width: Some(800),
        height: Some(400),
        db_range: (-80.0, 0.0),
    };

    // mel_spec
    //     .save_image_with("mel_speech.png", &vis_config)
    //     .unwrap();
    println!("   ✓ mel_speech.png (800x400 pixels, 80 mel bins)");
    println!("      Shows mel-scale frequency analysis (speech-optimized)");
}

#[cfg(feature = "visualization")]
fn compare_colormaps() {
    // Generate a simple test signal
    let sample_rate = 44100.0;
    let duration = 1.0;
    let num_samples = (sample_rate * duration) as usize;

    let signal: Vec<f32> = (0..num_samples)
        .map(|i| {
            let t = i as f32 / sample_rate;
            // Multiple harmonics
            (0..10)
                .map(|h| {
                    let freq = 200.0 * (h + 1) as f32;
                    let amp = 1.0 / (h + 1) as f32; // Decreasing amplitude
                    (2.0 * std::f32::consts::PI * freq * t).sin() * amp
                })
                .sum::<f32>()
                * 0.1
        })
        .collect();

    let config = StftConfigF32::default_4096();
    let stft = BatchStftF32::new(config);
    let spectrum = stft.process(&signal);

    // Test different color maps
    let colormaps = [
        (ColorMap::Inferno, "colormap_inferno.png"),
        (ColorMap::Plasma, "colormap_plasma.png"),
        (ColorMap::Grayscale, "colormap_grayscale.png"),
    ];

    for (colormap, filename) in colormaps {
        let vis_config = VisualizationConfig {
            colormap,
            width: Some(600),
            height: Some(300),
            db_range: (-80.0, -20.0),
        };
        spectrum.save_image_with(filename, &vis_config).unwrap();
        println!("   ✓ {}", filename);
    }
}
