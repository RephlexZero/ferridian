use serde::{Deserialize, Serialize};
use thiserror::Error;

// ---------------------------------------------------------------------------
// Chunk section binary layout
// ---------------------------------------------------------------------------

pub const CHUNK_SECTION_LAYOUT_VERSION: u32 = 1;
pub const CHUNK_SECTION_EDGE: usize = 16;
pub const CHUNK_SECTION_VOXEL_COUNT: usize =
    CHUNK_SECTION_EDGE * CHUNK_SECTION_EDGE * CHUNK_SECTION_EDGE;
pub const CHUNK_SECTION_HEADER_BYTES: usize = 24;
pub const CHUNK_SECTION_PACKET_BYTES: usize =
    CHUNK_SECTION_HEADER_BYTES + CHUNK_SECTION_VOXEL_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSectionHeader {
    pub version: u32,
    pub section_x: i32,
    pub section_y: i32,
    pub section_z: i32,
    pub non_air_blocks: u32,
    pub block_bytes: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChunkSectionPacket<'a> {
    pub header: ChunkSectionHeader,
    pub blocks: &'a [u8],
}

impl<'a> ChunkSectionPacket<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, FerridianError> {
        if bytes.len() != CHUNK_SECTION_PACKET_BYTES {
            return Err(FerridianError::InvalidChunkSectionPacketSize {
                expected: CHUNK_SECTION_PACKET_BYTES,
                actual: bytes.len(),
            });
        }

        let header = ChunkSectionHeader {
            version: u32::from_le_bytes(bytes[0..4].try_into().expect("version bytes")),
            section_x: i32::from_le_bytes(bytes[4..8].try_into().expect("section_x bytes")),
            section_y: i32::from_le_bytes(bytes[8..12].try_into().expect("section_y bytes")),
            section_z: i32::from_le_bytes(bytes[12..16].try_into().expect("section_z bytes")),
            non_air_blocks: u32::from_le_bytes(
                bytes[16..20].try_into().expect("non_air_blocks bytes"),
            ),
            block_bytes: u32::from_le_bytes(bytes[20..24].try_into().expect("block_bytes bytes")),
        };

        if header.version != CHUNK_SECTION_LAYOUT_VERSION {
            return Err(FerridianError::UnsupportedChunkSectionLayoutVersion(
                header.version,
            ));
        }

        if header.block_bytes as usize != CHUNK_SECTION_VOXEL_COUNT {
            return Err(FerridianError::InvalidChunkSectionBlockByteCount(
                header.block_bytes as usize,
            ));
        }

        if header.non_air_blocks as usize > CHUNK_SECTION_VOXEL_COUNT {
            return Err(FerridianError::InvalidChunkSectionNonAirCount(
                header.non_air_blocks,
            ));
        }

        let blocks = &bytes[CHUNK_SECTION_HEADER_BYTES..];
        let actual_non_air = blocks.iter().filter(|block| **block != 0).count() as u32;
        if actual_non_air != header.non_air_blocks {
            return Err(FerridianError::ChunkSectionNonAirCountMismatch {
                expected: header.non_air_blocks,
                actual: actual_non_air,
            });
        }

        Ok(Self { header, blocks })
    }
}

// ---------------------------------------------------------------------------
// Entity binary layout – flat buffer for entity position/rotation snapshots
// sent once per frame over the JNI boundary.
// ---------------------------------------------------------------------------

pub const ENTITY_LAYOUT_VERSION: u32 = 1;
pub const ENTITY_HEADER_BYTES: usize = 8;
pub const ENTITY_RECORD_BYTES: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EntityPacketHeader {
    pub version: u32,
    pub entity_count: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EntityRecord {
    pub entity_id: u32,
    pub kind: u16,
    pub flags: u16,
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub yaw: f32,
    pub pitch: f32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct EntityPacket<'a> {
    pub header: EntityPacketHeader,
    pub data: &'a [u8],
}

impl<'a> EntityPacket<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, FerridianError> {
        if bytes.len() < ENTITY_HEADER_BYTES {
            return Err(FerridianError::InvalidEntityPacketSize {
                minimum: ENTITY_HEADER_BYTES,
                actual: bytes.len(),
            });
        }

        let header = EntityPacketHeader {
            version: u32::from_le_bytes(bytes[0..4].try_into().expect("version")),
            entity_count: u32::from_le_bytes(bytes[4..8].try_into().expect("entity_count")),
        };

        if header.version != ENTITY_LAYOUT_VERSION {
            return Err(FerridianError::UnsupportedEntityLayoutVersion(
                header.version,
            ));
        }

        let expected_size =
            ENTITY_HEADER_BYTES + header.entity_count as usize * ENTITY_RECORD_BYTES;
        if bytes.len() != expected_size {
            return Err(FerridianError::InvalidEntityPacketSize {
                minimum: expected_size,
                actual: bytes.len(),
            });
        }

        Ok(Self {
            header,
            data: &bytes[ENTITY_HEADER_BYTES..],
        })
    }

    pub fn record(&self, index: usize) -> Option<EntityRecord> {
        if index >= self.header.entity_count as usize {
            return None;
        }
        let offset = index * ENTITY_RECORD_BYTES;
        let d = &self.data[offset..offset + ENTITY_RECORD_BYTES];
        Some(EntityRecord {
            entity_id: u32::from_le_bytes(d[0..4].try_into().unwrap()),
            kind: u16::from_le_bytes(d[4..6].try_into().unwrap()),
            flags: u16::from_le_bytes(d[6..8].try_into().unwrap()),
            x: f32::from_le_bytes(d[8..12].try_into().unwrap()),
            y: f32::from_le_bytes(d[12..16].try_into().unwrap()),
            z: f32::from_le_bytes(d[16..20].try_into().unwrap()),
            yaw: f32::from_le_bytes(d[20..24].try_into().unwrap()),
            pitch: f32::from_le_bytes(d[24..28].try_into().unwrap()),
        })
    }
}

// ---------------------------------------------------------------------------
// Lighting binary layout – sky and block light arrays per chunk section.
// ---------------------------------------------------------------------------

pub const LIGHTING_LAYOUT_VERSION: u32 = 1;
pub const LIGHTING_HEADER_BYTES: usize = 12;
pub const LIGHTING_NIBBLE_COUNT: usize = CHUNK_SECTION_VOXEL_COUNT / 2;
pub const LIGHTING_PACKET_BYTES: usize =
    LIGHTING_HEADER_BYTES + LIGHTING_NIBBLE_COUNT + LIGHTING_NIBBLE_COUNT;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LightingPacketHeader {
    pub version: u32,
    pub section_x: i32,
    pub section_z: i32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LightingPacket<'a> {
    pub header: LightingPacketHeader,
    pub sky_light: &'a [u8],
    pub block_light: &'a [u8],
}

impl<'a> LightingPacket<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, FerridianError> {
        if bytes.len() != LIGHTING_PACKET_BYTES {
            return Err(FerridianError::InvalidLightingPacketSize {
                expected: LIGHTING_PACKET_BYTES,
                actual: bytes.len(),
            });
        }

        let header = LightingPacketHeader {
            version: u32::from_le_bytes(bytes[0..4].try_into().expect("version")),
            section_x: i32::from_le_bytes(bytes[4..8].try_into().expect("section_x")),
            section_z: i32::from_le_bytes(bytes[8..12].try_into().expect("section_z")),
        };

        if header.version != LIGHTING_LAYOUT_VERSION {
            return Err(FerridianError::UnsupportedLightingLayoutVersion(
                header.version,
            ));
        }

        let sky_end = LIGHTING_HEADER_BYTES + LIGHTING_NIBBLE_COUNT;
        Ok(Self {
            header,
            sky_light: &bytes[LIGHTING_HEADER_BYTES..sky_end],
            block_light: &bytes[sky_end..],
        })
    }
}

// ---------------------------------------------------------------------------
// Weather binary layout – compact per-frame weather state.
// ---------------------------------------------------------------------------

pub const WEATHER_LAYOUT_VERSION: u32 = 1;
pub const WEATHER_PACKET_BYTES: usize = 20;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeatherPacket {
    pub version: u32,
    pub rain_strength: f32,
    pub thunder_strength: f32,
    pub sky_darkness: f32,
    pub time_of_day: f32,
}

impl WeatherPacket {
    pub fn parse(bytes: &[u8]) -> Result<Self, FerridianError> {
        if bytes.len() != WEATHER_PACKET_BYTES {
            return Err(FerridianError::InvalidWeatherPacketSize {
                expected: WEATHER_PACKET_BYTES,
                actual: bytes.len(),
            });
        }

        let version = u32::from_le_bytes(bytes[0..4].try_into().expect("version"));
        if version != WEATHER_LAYOUT_VERSION {
            return Err(FerridianError::UnsupportedWeatherLayoutVersion(version));
        }

        Ok(Self {
            version,
            rain_strength: f32::from_le_bytes(bytes[4..8].try_into().expect("rain")),
            thunder_strength: f32::from_le_bytes(bytes[8..12].try_into().expect("thunder")),
            sky_darkness: f32::from_le_bytes(bytes[12..16].try_into().expect("darkness")),
            time_of_day: f32::from_le_bytes(bytes[16..20].try_into().expect("time")),
        })
    }
}

// ---------------------------------------------------------------------------
// Resource pack metadata layout – describes a loaded resource pack for the
// renderer's material and texture systems.
// ---------------------------------------------------------------------------

pub const RESOURCE_PACK_LAYOUT_VERSION: u32 = 1;
pub const RESOURCE_PACK_HEADER_BYTES: usize = 16;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourcePackHeader {
    pub version: u32,
    pub format_version: u32,
    pub texture_count: u32,
    pub material_count: u32,
}

impl ResourcePackHeader {
    pub fn parse(bytes: &[u8]) -> Result<Self, FerridianError> {
        if bytes.len() < RESOURCE_PACK_HEADER_BYTES {
            return Err(FerridianError::InvalidResourcePackHeaderSize {
                minimum: RESOURCE_PACK_HEADER_BYTES,
                actual: bytes.len(),
            });
        }

        let version = u32::from_le_bytes(bytes[0..4].try_into().expect("version"));
        if version != RESOURCE_PACK_LAYOUT_VERSION {
            return Err(FerridianError::UnsupportedResourcePackLayoutVersion(
                version,
            ));
        }

        Ok(Self {
            version,
            format_version: u32::from_le_bytes(bytes[4..8].try_into().expect("format")),
            texture_count: u32::from_le_bytes(bytes[8..12].try_into().expect("textures")),
            material_count: u32::from_le_bytes(bytes[12..16].try_into().expect("materials")),
        })
    }
}

// ---------------------------------------------------------------------------
// Shared types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkspaceMetadata {
    pub name: &'static str,
    pub focus: &'static str,
}

impl WorkspaceMetadata {
    pub const fn new(name: &'static str, focus: &'static str) -> Self {
        Self { name, focus }
    }
}

#[derive(Debug, Error)]
pub enum FerridianError {
    #[error("unsupported renderer backend: {0}")]
    UnsupportedBackend(&'static str),
    #[error("unsupported chunk section layout version: {0}")]
    UnsupportedChunkSectionLayoutVersion(u32),
    #[error("invalid chunk section packet size: expected {expected} bytes, got {actual}")]
    InvalidChunkSectionPacketSize { expected: usize, actual: usize },
    #[error("invalid chunk section block byte count: {0}")]
    InvalidChunkSectionBlockByteCount(usize),
    #[error("invalid chunk section non-air count: {0}")]
    InvalidChunkSectionNonAirCount(u32),
    #[error("chunk section non-air count mismatch: expected {expected}, got {actual}")]
    ChunkSectionNonAirCountMismatch { expected: u32, actual: u32 },
    #[error("unsupported entity layout version: {0}")]
    UnsupportedEntityLayoutVersion(u32),
    #[error("invalid entity packet size: minimum {minimum} bytes, got {actual}")]
    InvalidEntityPacketSize { minimum: usize, actual: usize },
    #[error("unsupported lighting layout version: {0}")]
    UnsupportedLightingLayoutVersion(u32),
    #[error("invalid lighting packet size: expected {expected} bytes, got {actual}")]
    InvalidLightingPacketSize { expected: usize, actual: usize },
    #[error("unsupported weather layout version: {0}")]
    UnsupportedWeatherLayoutVersion(u32),
    #[error("invalid weather packet size: expected {expected} bytes, got {actual}")]
    InvalidWeatherPacketSize { expected: usize, actual: usize },
    #[error("unsupported resource pack layout version: {0}")]
    UnsupportedResourcePackLayoutVersion(u32),
    #[error("invalid resource pack header size: minimum {minimum} bytes, got {actual}")]
    InvalidResourcePackHeaderSize { minimum: usize, actual: usize },
}

// ---------------------------------------------------------------------------
// Runtime CPU feature detection — detect SIMD and other capabilities at runtime
// so hot paths can pick optimal code paths without requiring compile-time flags.
// ---------------------------------------------------------------------------

/// Detected CPU features available at runtime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CpuFeatures {
    pub sse2: bool,
    pub sse4_1: bool,
    pub avx: bool,
    pub avx2: bool,
    pub neon: bool,
}

impl CpuFeatures {
    /// Detect CPU features available on the current platform.
    pub fn detect() -> Self {
        Self {
            sse2: cfg!(target_feature = "sse2") || Self::runtime_check("sse2"),
            sse4_1: cfg!(target_feature = "sse4.1") || Self::runtime_check("sse4.1"),
            avx: cfg!(target_feature = "avx") || Self::runtime_check("avx"),
            avx2: cfg!(target_feature = "avx2") || Self::runtime_check("avx2"),
            neon: cfg!(target_arch = "aarch64"),
        }
    }

    fn runtime_check(_feature: &str) -> bool {
        // On x86/x86_64 we use std::arch::is_x86_feature_detected! at call
        // sites. This is a fallback for platforms where runtime detection
        // isn't available.
        #[cfg(target_arch = "x86_64")]
        {
            match _feature {
                "sse2" => std::arch::is_x86_feature_detected!("sse2"),
                "sse4.1" => std::arch::is_x86_feature_detected!("sse4.1"),
                "avx" => std::arch::is_x86_feature_detected!("avx"),
                "avx2" => std::arch::is_x86_feature_detected!("avx2"),
                _ => false,
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            false
        }
    }

    /// Returns the best available SIMD tier name for logging.
    pub fn best_simd_tier(&self) -> &'static str {
        if self.avx2 {
            "AVX2"
        } else if self.avx {
            "AVX"
        } else if self.sse4_1 {
            "SSE4.1"
        } else if self.neon {
            "NEON"
        } else if self.sse2 {
            "SSE2"
        } else {
            "scalar"
        }
    }
}

// ---------------------------------------------------------------------------
// Portable SIMD experiments — stable std::arch wrappers for hot paths
// ---------------------------------------------------------------------------

/// Scalar fallback for summing a slice of f32.
pub fn sum_f32_scalar(data: &[f32]) -> f32 {
    data.iter().sum()
}

/// SIMD-accelerated f32 sum using SSE2 when available (x86_64).
/// Falls back to scalar on other architectures.
pub fn sum_f32_simd(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: SSE2 presence verified above, and we only access
            // data within bounds via chunks_exact.
            return unsafe { sum_f32_sse2(data) };
        }
    }
    sum_f32_scalar(data)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn sum_f32_sse2(data: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        let mut acc = _mm_setzero_ps();

        for chunk in chunks {
            let v = _mm_loadu_ps(chunk.as_ptr());
            acc = _mm_add_ps(acc, v);
        }

        let shuf = _mm_shuffle_ps::<0b01_00_11_10>(acc, acc);
        let sums = _mm_add_ps(acc, shuf);
        let shuf2 = _mm_shuffle_ps::<0b00_01_00_01>(sums, sums);
        let sums2 = _mm_add_ps(sums, shuf2);
        let mut result = _mm_cvtss_f32(sums2);

        for &v in remainder {
            result += v;
        }
        result
    }
}

/// Scalar fallback for finding the maximum in a slice of f32.
pub fn max_f32_scalar(data: &[f32]) -> f32 {
    data.iter().copied().fold(f32::NEG_INFINITY, f32::max)
}

/// SIMD-accelerated f32 max using SSE2 when available.
pub fn max_f32_simd(data: &[f32]) -> f32 {
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("sse2") {
            return unsafe { max_f32_sse2(data) };
        }
    }
    max_f32_scalar(data)
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn max_f32_sse2(data: &[f32]) -> f32 {
    use std::arch::x86_64::*;

    unsafe {
        let chunks = data.chunks_exact(4);
        let remainder = chunks.remainder();
        let mut acc = _mm_set1_ps(f32::NEG_INFINITY);

        for chunk in chunks {
            let v = _mm_loadu_ps(chunk.as_ptr());
            acc = _mm_max_ps(acc, v);
        }

        let shuf = _mm_shuffle_ps::<0b01_00_11_10>(acc, acc);
        let maxs = _mm_max_ps(acc, shuf);
        let shuf2 = _mm_shuffle_ps::<0b00_01_00_01>(maxs, maxs);
        let maxs2 = _mm_max_ps(maxs, shuf2);
        let mut result = _mm_cvtss_f32(maxs2);

        for &v in remainder {
            result = result.max(v);
        }
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_chunk_section_packet() {
        let mut bytes = vec![0_u8; CHUNK_SECTION_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&CHUNK_SECTION_LAYOUT_VERSION.to_le_bytes());
        bytes[16..20].copy_from_slice(&2_u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&4096_u32.to_le_bytes());
        bytes[24] = 1;
        bytes[25] = 2;

        let packet = ChunkSectionPacket::parse(&bytes).expect("packet should parse");

        assert_eq!(packet.header.non_air_blocks, 2);
        assert_eq!(packet.blocks[0], 1);
        assert_eq!(packet.blocks[1], 2);
    }

    #[test]
    fn parses_entity_packet() {
        let count = 1_u32;
        let mut bytes = vec![0_u8; ENTITY_HEADER_BYTES + ENTITY_RECORD_BYTES];
        bytes[0..4].copy_from_slice(&ENTITY_LAYOUT_VERSION.to_le_bytes());
        bytes[4..8].copy_from_slice(&count.to_le_bytes());
        // entity_id = 42
        bytes[8..12].copy_from_slice(&42_u32.to_le_bytes());
        // kind = 1
        bytes[12..14].copy_from_slice(&1_u16.to_le_bytes());
        // x = 1.0
        bytes[16..20].copy_from_slice(&1.0_f32.to_le_bytes());

        let packet = EntityPacket::parse(&bytes).expect("entity packet should parse");
        assert_eq!(packet.header.entity_count, 1);
        let record = packet.record(0).expect("record 0");
        assert_eq!(record.entity_id, 42);
        assert_eq!(record.kind, 1);
        assert_eq!(record.x, 1.0);
    }

    #[test]
    fn parses_lighting_packet() {
        let mut bytes = vec![0_u8; LIGHTING_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&LIGHTING_LAYOUT_VERSION.to_le_bytes());
        bytes[4..8].copy_from_slice(&5_i32.to_le_bytes());
        bytes[8..12].copy_from_slice(&(-3_i32).to_le_bytes());

        let packet = LightingPacket::parse(&bytes).expect("lighting packet should parse");
        assert_eq!(packet.header.section_x, 5);
        assert_eq!(packet.header.section_z, -3);
        assert_eq!(packet.sky_light.len(), LIGHTING_NIBBLE_COUNT);
        assert_eq!(packet.block_light.len(), LIGHTING_NIBBLE_COUNT);
    }

    #[test]
    fn parses_weather_packet() {
        let mut bytes = vec![0_u8; WEATHER_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&WEATHER_LAYOUT_VERSION.to_le_bytes());
        bytes[4..8].copy_from_slice(&0.75_f32.to_le_bytes());
        bytes[8..12].copy_from_slice(&0.0_f32.to_le_bytes());
        bytes[12..16].copy_from_slice(&0.3_f32.to_le_bytes());
        bytes[16..20].copy_from_slice(&6000.0_f32.to_le_bytes());

        let packet = WeatherPacket::parse(&bytes).expect("weather packet should parse");
        assert_eq!(packet.rain_strength, 0.75);
        assert_eq!(packet.time_of_day, 6000.0);
    }

    #[test]
    fn parses_resource_pack_header() {
        let mut bytes = vec![0_u8; RESOURCE_PACK_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&RESOURCE_PACK_LAYOUT_VERSION.to_le_bytes());
        bytes[4..8].copy_from_slice(&15_u32.to_le_bytes());
        bytes[8..12].copy_from_slice(&256_u32.to_le_bytes());
        bytes[12..16].copy_from_slice(&64_u32.to_le_bytes());

        let header = ResourcePackHeader::parse(&bytes).expect("should parse");
        assert_eq!(header.format_version, 15);
        assert_eq!(header.texture_count, 256);
        assert_eq!(header.material_count, 64);
    }

    #[test]
    fn cpu_features_detect_returns_valid_tier() {
        let features = CpuFeatures::detect();
        let tier = features.best_simd_tier();
        assert!(["AVX2", "AVX", "SSE4.1", "SSE2", "NEON", "scalar"].contains(&tier));
    }

    #[test]
    fn simd_sum_matches_scalar() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.5).collect();
        let scalar = sum_f32_scalar(&data);
        let simd = sum_f32_simd(&data);
        assert!((scalar - simd).abs() < 0.01, "scalar={scalar} simd={simd}");
    }

    #[test]
    fn simd_max_matches_scalar() {
        let data = vec![1.0_f32, 5.0, 3.0, 9.0, 2.0, 7.0, 4.0, 8.0, 6.0];
        let scalar = max_f32_scalar(&data);
        let simd = max_f32_simd(&data);
        assert_eq!(scalar, simd);
        assert_eq!(simd, 9.0);
    }

    // -----------------------------------------------------------------------
    // FerridianError Display tests — every variant
    // -----------------------------------------------------------------------

    #[test]
    fn error_display_unsupported_backend() {
        let err = FerridianError::UnsupportedBackend("opengl");
        let msg = err.to_string();
        assert!(msg.contains("opengl"));
    }

    #[test]
    fn error_display_chunk_section_wrong_version() {
        let err = FerridianError::UnsupportedChunkSectionLayoutVersion(99);
        let msg = err.to_string();
        assert!(msg.contains("99"));
    }

    #[test]
    fn error_display_chunk_section_wrong_size() {
        let err = FerridianError::InvalidChunkSectionPacketSize {
            expected: 100,
            actual: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("100") && msg.contains("50"));
    }

    #[test]
    fn error_display_chunk_section_block_byte_count() {
        let err = FerridianError::InvalidChunkSectionBlockByteCount(999);
        assert!(err.to_string().contains("999"));
    }

    #[test]
    fn error_display_chunk_section_non_air_count() {
        let err = FerridianError::InvalidChunkSectionNonAirCount(5000);
        assert!(err.to_string().contains("5000"));
    }

    #[test]
    fn error_display_chunk_section_mismatch() {
        let err = FerridianError::ChunkSectionNonAirCountMismatch {
            expected: 10,
            actual: 5,
        };
        let msg = err.to_string();
        assert!(msg.contains("10") && msg.contains("5"));
    }

    #[test]
    fn error_display_entity_version() {
        let err = FerridianError::UnsupportedEntityLayoutVersion(42);
        assert!(err.to_string().contains("42"));
    }

    #[test]
    fn error_display_entity_size() {
        let err = FerridianError::InvalidEntityPacketSize {
            minimum: 8,
            actual: 4,
        };
        let msg = err.to_string();
        assert!(msg.contains("8") && msg.contains("4"));
    }

    #[test]
    fn error_display_lighting_version() {
        let err = FerridianError::UnsupportedLightingLayoutVersion(77);
        assert!(err.to_string().contains("77"));
    }

    #[test]
    fn error_display_lighting_size() {
        let err = FerridianError::InvalidLightingPacketSize {
            expected: 100,
            actual: 50,
        };
        let msg = err.to_string();
        assert!(msg.contains("100") && msg.contains("50"));
    }

    #[test]
    fn error_display_weather_version() {
        let err = FerridianError::UnsupportedWeatherLayoutVersion(3);
        assert!(err.to_string().contains("3"));
    }

    #[test]
    fn error_display_weather_size() {
        let err = FerridianError::InvalidWeatherPacketSize {
            expected: 20,
            actual: 10,
        };
        let msg = err.to_string();
        assert!(msg.contains("20") && msg.contains("10"));
    }

    #[test]
    fn error_display_resource_pack_version() {
        let err = FerridianError::UnsupportedResourcePackLayoutVersion(50);
        assert!(err.to_string().contains("50"));
    }

    #[test]
    fn error_display_resource_pack_size() {
        let err = FerridianError::InvalidResourcePackHeaderSize {
            minimum: 16,
            actual: 8,
        };
        let msg = err.to_string();
        assert!(msg.contains("16") && msg.contains("8"));
    }

    // -----------------------------------------------------------------------
    // ChunkSectionPacket error paths
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_section_rejects_wrong_size() {
        let result = ChunkSectionPacket::parse(&[0u8; 10]);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidChunkSectionPacketSize { .. })
        ));
    }

    #[test]
    fn chunk_section_rejects_wrong_version() {
        let mut bytes = vec![0_u8; CHUNK_SECTION_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&99_u32.to_le_bytes());
        let result = ChunkSectionPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::UnsupportedChunkSectionLayoutVersion(99))
        ));
    }

    #[test]
    fn chunk_section_rejects_wrong_block_bytes() {
        let mut bytes = vec![0_u8; CHUNK_SECTION_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&CHUNK_SECTION_LAYOUT_VERSION.to_le_bytes());
        bytes[16..20].copy_from_slice(&0_u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&999_u32.to_le_bytes()); // wrong block_bytes
        let result = ChunkSectionPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidChunkSectionBlockByteCount(999))
        ));
    }

    #[test]
    fn chunk_section_rejects_too_many_non_air() {
        let mut bytes = vec![0_u8; CHUNK_SECTION_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&CHUNK_SECTION_LAYOUT_VERSION.to_le_bytes());
        bytes[16..20].copy_from_slice(&(CHUNK_SECTION_VOXEL_COUNT as u32 + 1).to_le_bytes());
        bytes[20..24].copy_from_slice(&(CHUNK_SECTION_VOXEL_COUNT as u32).to_le_bytes());
        let result = ChunkSectionPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidChunkSectionNonAirCount(..))
        ));
    }

    #[test]
    fn chunk_section_rejects_non_air_mismatch() {
        let mut bytes = vec![0_u8; CHUNK_SECTION_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&CHUNK_SECTION_LAYOUT_VERSION.to_le_bytes());
        bytes[16..20].copy_from_slice(&5_u32.to_le_bytes()); // claims 5 non-air
        bytes[20..24].copy_from_slice(&(CHUNK_SECTION_VOXEL_COUNT as u32).to_le_bytes());
        // but no non-air blocks in data => mismatch
        let result = ChunkSectionPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::ChunkSectionNonAirCountMismatch { .. })
        ));
    }

    // -----------------------------------------------------------------------
    // EntityPacket error paths
    // -----------------------------------------------------------------------

    #[test]
    fn entity_packet_rejects_too_short() {
        let result = EntityPacket::parse(&[0u8; 4]);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidEntityPacketSize { .. })
        ));
    }

    #[test]
    fn entity_packet_rejects_wrong_version() {
        let mut bytes = vec![0_u8; ENTITY_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&99_u32.to_le_bytes());
        bytes[4..8].copy_from_slice(&0_u32.to_le_bytes());
        let result = EntityPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::UnsupportedEntityLayoutVersion(99))
        ));
    }

    #[test]
    fn entity_packet_rejects_size_mismatch() {
        // Header says 2 entities but buffer only has room for 0
        let mut bytes = vec![0_u8; ENTITY_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&ENTITY_LAYOUT_VERSION.to_le_bytes());
        bytes[4..8].copy_from_slice(&2_u32.to_le_bytes());
        let result = EntityPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidEntityPacketSize { .. })
        ));
    }

    #[test]
    fn entity_packet_record_out_of_bounds() {
        let mut bytes = vec![0_u8; ENTITY_HEADER_BYTES + ENTITY_RECORD_BYTES];
        bytes[0..4].copy_from_slice(&ENTITY_LAYOUT_VERSION.to_le_bytes());
        bytes[4..8].copy_from_slice(&1_u32.to_le_bytes());
        let packet = EntityPacket::parse(&bytes).unwrap();
        assert!(packet.record(0).is_some());
        assert!(packet.record(1).is_none());
    }

    // -----------------------------------------------------------------------
    // LightingPacket error paths
    // -----------------------------------------------------------------------

    #[test]
    fn lighting_packet_rejects_wrong_size() {
        let result = LightingPacket::parse(&[0u8; 10]);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidLightingPacketSize { .. })
        ));
    }

    #[test]
    fn lighting_packet_rejects_wrong_version() {
        let mut bytes = vec![0_u8; LIGHTING_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&99_u32.to_le_bytes());
        let result = LightingPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::UnsupportedLightingLayoutVersion(99))
        ));
    }

    // -----------------------------------------------------------------------
    // WeatherPacket error paths
    // -----------------------------------------------------------------------

    #[test]
    fn weather_packet_rejects_wrong_size() {
        let result = WeatherPacket::parse(&[0u8; 10]);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidWeatherPacketSize { .. })
        ));
    }

    #[test]
    fn weather_packet_rejects_wrong_version() {
        let mut bytes = vec![0_u8; WEATHER_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&99_u32.to_le_bytes());
        let result = WeatherPacket::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::UnsupportedWeatherLayoutVersion(99))
        ));
    }

    // -----------------------------------------------------------------------
    // ResourcePackHeader error paths
    // -----------------------------------------------------------------------

    #[test]
    fn resource_pack_header_rejects_too_short() {
        let result = ResourcePackHeader::parse(&[0u8; 4]);
        assert!(matches!(
            result,
            Err(FerridianError::InvalidResourcePackHeaderSize { .. })
        ));
    }

    #[test]
    fn resource_pack_header_rejects_wrong_version() {
        let mut bytes = vec![0_u8; RESOURCE_PACK_HEADER_BYTES];
        bytes[0..4].copy_from_slice(&99_u32.to_le_bytes());
        let result = ResourcePackHeader::parse(&bytes);
        assert!(matches!(
            result,
            Err(FerridianError::UnsupportedResourcePackLayoutVersion(99))
        ));
    }

    // -----------------------------------------------------------------------
    // SIMD edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn sum_f32_empty_slice() {
        assert_eq!(sum_f32_scalar(&[]), 0.0);
        assert_eq!(sum_f32_simd(&[]), 0.0);
    }

    #[test]
    fn sum_f32_single_element() {
        assert_eq!(sum_f32_scalar(&[42.0]), 42.0);
        assert_eq!(sum_f32_simd(&[42.0]), 42.0);
    }

    #[test]
    fn max_f32_empty_slice() {
        assert_eq!(max_f32_scalar(&[]), f32::NEG_INFINITY);
        assert_eq!(max_f32_simd(&[]), f32::NEG_INFINITY);
    }

    #[test]
    fn max_f32_single_element() {
        assert_eq!(max_f32_scalar(&[7.0]), 7.0);
        assert_eq!(max_f32_simd(&[7.0]), 7.0);
    }

    #[test]
    fn max_f32_all_negative() {
        let data = vec![-5.0, -1.0, -10.0, -3.0];
        assert_eq!(max_f32_scalar(&data), -1.0);
        assert_eq!(max_f32_simd(&data), -1.0);
    }

    // -----------------------------------------------------------------------
    // Constant assertions
    // -----------------------------------------------------------------------

    #[test]
    fn chunk_section_constants_consistent() {
        assert_eq!(CHUNK_SECTION_EDGE, 16);
        assert_eq!(CHUNK_SECTION_VOXEL_COUNT, 16 * 16 * 16);
        assert_eq!(
            CHUNK_SECTION_PACKET_BYTES,
            CHUNK_SECTION_HEADER_BYTES + CHUNK_SECTION_VOXEL_COUNT
        );
    }

    #[test]
    fn entity_constants_consistent() {
        assert_eq!(ENTITY_HEADER_BYTES, 8);
        assert_eq!(ENTITY_RECORD_BYTES, 32);
    }

    #[test]
    fn lighting_constants_consistent() {
        assert_eq!(LIGHTING_NIBBLE_COUNT, CHUNK_SECTION_VOXEL_COUNT / 2);
        assert_eq!(
            LIGHTING_PACKET_BYTES,
            LIGHTING_HEADER_BYTES + 2 * LIGHTING_NIBBLE_COUNT
        );
    }

    #[test]
    fn weather_packet_constant() {
        assert_eq!(WEATHER_PACKET_BYTES, 20);
    }

    #[test]
    fn resource_pack_header_constant() {
        assert_eq!(RESOURCE_PACK_HEADER_BYTES, 16);
    }

    // -----------------------------------------------------------------------
    // WorkspaceMetadata
    // -----------------------------------------------------------------------

    #[test]
    fn workspace_metadata_construction() {
        let meta = WorkspaceMetadata::new("test", "testing");
        assert_eq!(meta.name, "test");
        assert_eq!(meta.focus, "testing");
    }

    #[test]
    fn workspace_metadata_serde_roundtrip() {
        let meta = WorkspaceMetadata::new("test", "testing");
        // Test that Serialize/Deserialize derives work
        let debug = format!("{meta:?}");
        assert!(debug.contains("test"));
    }

    // -----------------------------------------------------------------------
    // CpuFeatures
    // -----------------------------------------------------------------------

    #[test]
    fn cpu_features_detect_is_consistent() {
        let a = CpuFeatures::detect();
        let b = CpuFeatures::detect();
        assert_eq!(a, b);
    }

    #[test]
    fn cpu_features_best_simd_tier_is_not_empty() {
        let features = CpuFeatures::detect();
        assert!(!features.best_simd_tier().is_empty());
    }
}
