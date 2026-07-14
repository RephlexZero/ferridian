//! CPU-side RGBA images: PNG I/O and the perceptual diff the golden harness
//! judges renders with. Pure code — no Vulkan, unit-testable everywhere.

use std::fs;
use std::path::Path;

/// An 8-bit RGBA image in row-major order, as read back from the GPU.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RgbaImage {
    pub width: u32,
    pub height: u32,
    /// `width * height * 4` bytes, RGBA, top row first.
    pub pixels: Vec<u8>,
}

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("failed to read {path}: {source}")]
    Io {
        path: String,
        source: std::io::Error,
    },
    #[error("{path} is a git-lfs pointer, not an image — run `git lfs pull`")]
    LfsPointer { path: String },
    #[error("failed to decode {path}: {source}")]
    Decode {
        path: String,
        source: png::DecodingError,
    },
    #[error("failed to encode {path}: {source}")]
    Encode {
        path: String,
        source: png::EncodingError,
    },
    #[error("{path}: only 8-bit RGB/RGBA PNG images are supported (got {detail})")]
    UnsupportedFormat { path: String, detail: String },
}

impl RgbaImage {
    pub fn new(width: u32, height: u32) -> RgbaImage {
        RgbaImage {
            width,
            height,
            pixels: vec![0; width as usize * height as usize * 4],
        }
    }

    pub fn from_pixels(width: u32, height: u32, pixels: Vec<u8>) -> RgbaImage {
        assert_eq!(
            pixels.len(),
            width as usize * height as usize * 4,
            "pixel buffer does not match {width}x{height} RGBA"
        );
        RgbaImage {
            width,
            height,
            pixels,
        }
    }

    pub fn read_png(path: &Path) -> Result<RgbaImage, ImageError> {
        let display = path.display().to_string();
        let bytes = fs::read(path).map_err(|source| ImageError::Io {
            path: display.clone(),
            source,
        })?;
        // A golden checked out without LFS smudging is a small text pointer;
        // surface that as itself, not as a cryptic decode error.
        if bytes.starts_with(b"version https://git-lfs") {
            return Err(ImageError::LfsPointer { path: display });
        }
        let decoder = png::Decoder::new(bytes.as_slice());
        let mut reader = decoder.read_info().map_err(|source| ImageError::Decode {
            path: display.clone(),
            source,
        })?;
        let mut buffer = vec![0; reader.output_buffer_size()];
        let info = reader
            .next_frame(&mut buffer)
            .map_err(|source| ImageError::Decode {
                path: display.clone(),
                source,
            })?;
        buffer.truncate(info.buffer_size());
        let pixels = match (info.color_type, info.bit_depth) {
            (png::ColorType::Rgba, png::BitDepth::Eight) => buffer,
            (png::ColorType::Rgb, png::BitDepth::Eight) => buffer
                .chunks_exact(3)
                .flat_map(|rgb| [rgb[0], rgb[1], rgb[2], 255])
                .collect(),
            (color, depth) => {
                return Err(ImageError::UnsupportedFormat {
                    path: display,
                    detail: format!("{color:?}/{depth:?}"),
                });
            }
        };
        Ok(RgbaImage::from_pixels(info.width, info.height, pixels))
    }

    pub fn write_png(&self, path: &Path) -> Result<(), ImageError> {
        let display = path.display().to_string();
        let io_err = |source| ImageError::Io {
            path: display.clone(),
            source,
        };
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(io_err)?;
        }
        let file = fs::File::create(path).map_err(io_err)?;
        let mut encoder = png::Encoder::new(std::io::BufWriter::new(file), self.width, self.height);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        let mut writer = encoder
            .write_header()
            .map_err(|source| ImageError::Encode {
                path: display.clone(),
                source,
            })?;
        writer
            .write_image_data(&self.pixels)
            .map_err(|source| ImageError::Encode {
                path: display.clone(),
                source,
            })
    }
}

/// What "matches the golden" means. Lavapipe is pinned, so renders should be
/// bit-identical in CI; the defaults leave one quantum of headroom per channel
/// so a future Mesa bump surfaces as a *re-blessing decision*, not flakiness.
#[derive(Debug, Clone, Copy)]
pub struct DiffPolicy {
    /// Per-channel absolute difference at or below this is "the same pixel".
    pub max_channel_delta: u8,
    /// Fraction of pixels allowed to differ beyond `max_channel_delta`.
    pub max_differing_fraction: f64,
}

impl Default for DiffPolicy {
    fn default() -> Self {
        DiffPolicy {
            max_channel_delta: 1,
            max_differing_fraction: 0.0,
        }
    }
}

/// The result of comparing two same-sized images.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageDiff {
    pub total_pixels: usize,
    /// Pixels whose worst channel delta exceeds the policy's threshold.
    pub differing_pixels: usize,
    pub max_channel_delta: u8,
    pub mean_channel_delta: f64,
}

impl ImageDiff {
    pub fn differing_fraction(&self) -> f64 {
        if self.total_pixels == 0 {
            0.0
        } else {
            self.differing_pixels as f64 / self.total_pixels as f64
        }
    }
}

#[derive(Debug, thiserror::Error)]
#[error(
    "image dimensions differ: {expected_width}x{expected_height} vs {actual_width}x{actual_height}"
)]
pub struct DimensionMismatch {
    pub expected_width: u32,
    pub expected_height: u32,
    pub actual_width: u32,
    pub actual_height: u32,
}

/// Compare two images under a policy. Dimension mismatch is an error (never
/// tolerable); pixel differences come back as stats for the caller to judge.
pub fn diff_images(
    expected: &RgbaImage,
    actual: &RgbaImage,
    policy: &DiffPolicy,
) -> Result<ImageDiff, DimensionMismatch> {
    if (expected.width, expected.height) != (actual.width, actual.height) {
        return Err(DimensionMismatch {
            expected_width: expected.width,
            expected_height: expected.height,
            actual_width: actual.width,
            actual_height: actual.height,
        });
    }
    let total_pixels = expected.width as usize * expected.height as usize;
    let mut differing_pixels = 0;
    let mut max_channel_delta: u8 = 0;
    let mut delta_sum: u64 = 0;
    for (expected_px, actual_px) in expected
        .pixels
        .chunks_exact(4)
        .zip(actual.pixels.chunks_exact(4))
    {
        let mut worst = 0u8;
        for (&e, &a) in expected_px.iter().zip(actual_px) {
            let delta = e.abs_diff(a);
            worst = worst.max(delta);
            delta_sum += u64::from(delta);
        }
        max_channel_delta = max_channel_delta.max(worst);
        if worst > policy.max_channel_delta {
            differing_pixels += 1;
        }
    }
    Ok(ImageDiff {
        total_pixels,
        differing_pixels,
        max_channel_delta,
        mean_channel_delta: delta_sum as f64 / (total_pixels as f64 * 4.0).max(1.0),
    })
}

/// True when the diff is acceptable under the policy.
pub fn diff_passes(diff: &ImageDiff, policy: &DiffPolicy) -> bool {
    diff.differing_fraction() <= policy.max_differing_fraction
}

/// A visualization of where two images differ: differing pixels are amplified
/// (per-channel delta × 8, clamped), identical pixels are black. Written next
/// to failing goldens so CI artifacts are diagnosable at a glance.
pub fn diff_heatmap(expected: &RgbaImage, actual: &RgbaImage) -> RgbaImage {
    assert_eq!(
        (expected.width, expected.height),
        (actual.width, actual.height)
    );
    let pixels = expected
        .pixels
        .chunks_exact(4)
        .zip(actual.pixels.chunks_exact(4))
        .flat_map(|(e, a)| {
            [
                e[0].abs_diff(a[0]).saturating_mul(8),
                e[1].abs_diff(a[1]).saturating_mul(8),
                e[2].abs_diff(a[2]).saturating_mul(8),
                255,
            ]
        })
        .collect();
    RgbaImage::from_pixels(expected.width, expected.height, pixels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(width: u32, height: u32, rgba: [u8; 4]) -> RgbaImage {
        RgbaImage::from_pixels(
            width,
            height,
            rgba.iter()
                .copied()
                .cycle()
                .take(width as usize * height as usize * 4)
                .collect(),
        )
    }

    #[test]
    fn identical_images_pass_default_policy() {
        let policy = DiffPolicy::default();
        let a = solid(4, 4, [10, 20, 30, 255]);
        let diff = diff_images(&a, &a.clone(), &policy).unwrap();
        assert_eq!(diff.differing_pixels, 0);
        assert_eq!(diff.max_channel_delta, 0);
        assert!(diff_passes(&diff, &policy));
    }

    #[test]
    fn one_quantum_of_noise_passes_default_policy() {
        let policy = DiffPolicy::default();
        let a = solid(4, 4, [10, 20, 30, 255]);
        let b = solid(4, 4, [11, 19, 30, 255]);
        let diff = diff_images(&a, &b, &policy).unwrap();
        assert_eq!(diff.differing_pixels, 0);
        assert_eq!(diff.max_channel_delta, 1);
        assert!(diff_passes(&diff, &policy));
    }

    #[test]
    fn real_differences_fail_default_policy() {
        let policy = DiffPolicy::default();
        let a = solid(4, 4, [10, 20, 30, 255]);
        let b = solid(4, 4, [10, 20, 90, 255]);
        let diff = diff_images(&a, &b, &policy).unwrap();
        assert_eq!(diff.differing_pixels, 16);
        assert_eq!(diff.max_channel_delta, 60);
        assert!(!diff_passes(&diff, &policy));
    }

    #[test]
    fn dimension_mismatch_is_an_error() {
        let a = solid(4, 4, [0, 0, 0, 255]);
        let b = solid(2, 2, [0, 0, 0, 255]);
        assert!(diff_images(&a, &b, &DiffPolicy::default()).is_err());
    }

    #[test]
    fn png_round_trip() {
        let dir = std::env::temp_dir().join(format!("ferridian-image-{}", std::process::id()));
        let path = dir.join("roundtrip.png");
        let image = solid(3, 5, [1, 128, 255, 200]);
        image.write_png(&path).unwrap();
        assert_eq!(RgbaImage::read_png(&path).unwrap(), image);
        fs::remove_dir_all(dir).ok();
    }

    #[test]
    fn lfs_pointer_is_reported_as_such() {
        let dir = std::env::temp_dir().join(format!("ferridian-lfs-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        let path = dir.join("pointer.png");
        fs::write(
            &path,
            "version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 3\n",
        )
        .unwrap();
        assert!(matches!(
            RgbaImage::read_png(&path),
            Err(ImageError::LfsPointer { .. })
        ));
        fs::remove_dir_all(dir).ok();
    }
}
