use std::{error::Error, path::Path};

use image::{ImageBuffer, Rgb};
use num_traits::Float;

pub enum ColorMap {
    Viridis,
    Magma,
    Inferno,
    Plasma,
    Grayscale,
}

impl ColorMap {
    fn to_gradient(&self) -> Box<dyn colorgrad::Gradient> {
        use colorgrad::preset::*;
        match self {
            ColorMap::Viridis => Box::new(viridis()),
            ColorMap::Magma => Box::new(magma()),
            ColorMap::Inferno => Box::new(inferno()),
            ColorMap::Plasma => Box::new(plasma()),
            ColorMap::Grayscale => Box::new(greys()),
        }
    }
}

pub struct VisualizationConfig {
    pub colormap: ColorMap,
    pub width: Option<u32>,   // None = 1 pixel per frame
    pub height: Option<u32>,  // None = 1 pixel per freq bin
    pub db_range: (f32, f32), // (min_db, max_db) for color mapping
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        Self {
            colormap: ColorMap::Viridis,
            width: None,
            height: None,
            db_range: (-80.0, 0.0),
        }
    }
}

pub trait SpectrumExt<T: Float> {
    fn save_image(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>>;
    fn save_image_with(
        &self,
        path: impl AsRef<Path>,
        config: &VisualizationConfig,
    ) -> Result<(), Box<dyn Error>>;
    fn to_image(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>>;
    fn to_image_with(&self, config: &VisualizationConfig) -> ImageBuffer<Rgb<u8>, Vec<u8>>;
}

impl<T: Float> SpectrumExt<T> for crate::Spectrum<T> {
    fn save_image(&self, path: impl AsRef<Path>) -> Result<(), Box<dyn Error>> {
        self.save_image_with(path, &VisualizationConfig::default())
    }

    fn save_image_with(
        &self,
        path: impl AsRef<Path>,
        config: &VisualizationConfig,
    ) -> Result<(), Box<dyn Error>> {
        let img = self.to_image_with(config);
        img.save(path)?;
        Ok(())
    }

    fn to_image(&self) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        self.to_image_with(&VisualizationConfig::default())
    }

    fn to_image_with(&self, config: &VisualizationConfig) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let gradient = config.colormap.to_gradient();

        let width = config.width.unwrap_or(self.num_frames as u32);
        let height = config.height.unwrap_or(self.freq_bins as u32);

        let mut img = ImageBuffer::new(width, height);

        let mut mag_db = Vec::with_capacity(self.num_frames * self.freq_bins);

        for frame in 0..self.num_frames {
            for bin in 0..self.freq_bins {
                let mag = self.magnitude(frame, bin);
                let db = 20.0 * mag.to_f64().unwrap_or(0.0).max(1e-10).log10();
                mag_db.push(db);
            }
        }

        let (min_db, max_db) = config.db_range;
        let range = max_db - min_db;

        for y in 0..height {
            for x in 0..width {
                let frame = (x as f32 * self.num_frames as f32 / width as f32) as usize;
                let bin =
                    ((height - 1 - y) as f32 * self.freq_bins as f32 / height as f32) as usize;
                // Flip Y axis to have low freq at the bottom

                let db = mag_db[frame * self.freq_bins + bin];
                let normalized = ((db - min_db as f64) / range as f64).clamp(0.0, 1.0);

                let color = gradient.at(normalized as f32);

                img.put_pixel(
                    x,
                    y,
                    Rgb([
                        (color.r * 255.0) as u8,
                        (color.g * 255.0) as u8,
                        (color.b * 255.0) as u8,
                    ]),
                );
            }
        }

        img
    }
}
