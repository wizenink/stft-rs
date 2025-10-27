use stft_rs::prelude::*;

#[test]
fn test_config_validation_ola() {
    let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Ola);
    assert!(config.is_ok());
}

#[test]
fn test_config_validation_wola() {
    let config = StftConfig::<f32>::new(4096, 1024, WindowType::Hann, ReconstructionMode::Wola);
    assert!(config.is_ok());
}

#[test]
fn test_config_invalid_ola() {
    let config = StftConfig::<f32>::new(4096, 3000, WindowType::Hann, ReconstructionMode::Ola);
    assert!(config.is_err())
}

#[test]
fn test_config_invalid_fft_size() {
    let config = StftConfig::<f32>::new(4095, 1024, WindowType::Hann, ReconstructionMode::Ola);
    assert!(matches!(config, Err(_)));
}

#[test]
fn test_config_invalid_hop_size() {
    let config = StftConfig::<f32>::new(4096, 0, WindowType::Hann, ReconstructionMode::Ola);
    assert!(matches!(config, Err(_)));

    let config = StftConfig::<f32>::new(4096, 5000, WindowType::Hann, ReconstructionMode::Ola);
    assert!(matches!(config, Err(_)));
}
