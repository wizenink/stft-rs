use stft_rs::prelude::*;

#[test]
#[allow(deprecated)]
fn test_config_validation_ola() {
    let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Ola);
    assert!(config.is_ok());
}

#[test]
#[allow(deprecated)]
fn test_config_validation_wola() {
    let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Wola);
    assert!(config.is_ok());
}

#[test]
#[allow(deprecated)]
fn test_config_invalid_ola() {
    let config = StftConfig::<f32>::new(4096, 3000, WindowType::Hann, ReconstructionMode::Ola);
    assert!(config.is_err())
}

#[test]
#[allow(deprecated)]
fn test_config_invalid_fft_size() {
    let config = StftConfig::<f32>::new(4095, 1024, WindowType::Hann, ReconstructionMode::Ola);
    assert!(matches!(config, Err(_)));
}

#[test]
#[allow(deprecated)]
fn test_config_invalid_hop_size() {
    let config = StftConfig::<f32>::new(4096, 0, WindowType::Hann, ReconstructionMode::Ola);
    assert!(matches!(config, Err(_)));

    let config = StftConfig::<f32>::new(4096, 5000, WindowType::Hann, ReconstructionMode::Ola);
    assert!(matches!(config, Err(_)));
}

// Builder pattern tests
#[test]
fn test_builder_basic() {
    let config = StftConfig::<f32>::builder()
        .fft_size(4096)
        .hop_size(1024)
        .build();
    assert!(config.is_ok());
    let config = config.unwrap();
    assert_eq!(config.fft_size, 4096);
    assert_eq!(config.hop_size, 1024);
}

#[test]
fn test_builder_with_window() {
    let config = StftConfig::<f32>::builder()
        .fft_size(4096)
        .hop_size(1024)
        .window(WindowType::Hamming)
        .build();
    assert!(config.is_ok());
    let config = config.unwrap();
    assert_eq!(config.window, WindowType::Hamming);
}

#[test]
fn test_builder_with_reconstruction_mode() {
    let config = StftConfig::<f32>::builder()
        .fft_size(4096)
        .hop_size(1024)
        .reconstruction_mode(ReconstructionMode::Wola)
        .build();
    assert!(config.is_ok());
}

#[test]
fn test_builder_missing_fft_size() {
    let config = StftConfig::<f32>::builder()
        .hop_size(1024)
        .build();
    assert!(config.is_err());
}

#[test]
fn test_builder_missing_hop_size() {
    let config = StftConfig::<f32>::builder()
        .fft_size(4096)
        .build();
    assert!(config.is_err());
}

#[test]
fn test_builder_invalid_fft_size() {
    let config = StftConfig::<f32>::builder()
        .fft_size(4095) // Not a power of 2
        .hop_size(1024)
        .build();
    assert!(config.is_err());
}

#[test]
fn test_builder_invalid_hop_size() {
    let config = StftConfig::<f32>::builder()
        .fft_size(4096)
        .hop_size(0)
        .build();
    assert!(config.is_err());

    let config = StftConfig::<f32>::builder()
        .fft_size(4096)
        .hop_size(5000) // Greater than fft_size
        .build();
    assert!(config.is_err());
}

#[test]
fn test_builder_type_aliases() {
    let config = StftConfigBuilderF32::new()
        .fft_size(4096)
        .hop_size(1024)
        .build();
    assert!(config.is_ok());

    let config = StftConfigBuilderF64::new()
        .fft_size(4096)
        .hop_size(1024)
        .build();
    assert!(config.is_ok());
}
