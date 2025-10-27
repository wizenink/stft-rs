/// Common test utilities

pub fn calculate_snr(original: &[f32], reconstructed: &[f32]) -> f32 {
    assert_eq!(original.len(), reconstructed.len());

    let signal_power: f32 = original.iter().map(|x| x.powi(2)).sum();
    let noise_power: f32 = original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).powi(2))
        .sum();

    if noise_power == 0.0 {
        f32::INFINITY
    } else {
        10.0 * (signal_power / noise_power).log10()
    }
}

#[allow(dead_code)]
pub fn max_abs_error(original: &[f32], reconstructed: &[f32]) -> f32 {
    original
        .iter()
        .zip(reconstructed.iter())
        .map(|(o, r)| (o - r).abs())
        .max_by(|a, b| a.partial_cmp(b).unwrap())
        .unwrap_or(0.0)
}
