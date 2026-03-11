use ferridian_utils::ChunkSectionPacket;
use jni::EnvUnowned;
use jni::errors::ThrowRuntimeExAndDefault;
use jni::objects::{JByteBuffer, JClass, JString};
use jni::sys::{jfloat, jint, jlong};
use std::collections::HashMap;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{LazyLock, Mutex};

pub const JNI_LIBRARY_NAME: &str = "ferridian_jni";
static NEXT_RENDERER_HANDLE: AtomicI64 = AtomicI64::new(1);
static LIVE_RENDERERS: LazyLock<Mutex<HashMap<i64, RendererState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// JNI lookup cache — stores resolved jclass / jmethodID / jfieldID references
// so the hot path never repeats JNI resolution.
// ---------------------------------------------------------------------------

/// Cached JNI identifiers resolved once during initialisation. Keyed by a
/// user-defined string tag so callers can look up specific classes or methods.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JniLookupCache {
    class_ids: HashMap<&'static str, jlong>,
    method_ids: HashMap<&'static str, jlong>,
    field_ids: HashMap<&'static str, jlong>,
}

impl JniLookupCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cache_class(&mut self, tag: &'static str, id: jlong) {
        self.class_ids.insert(tag, id);
    }

    pub fn cache_method(&mut self, tag: &'static str, id: jlong) {
        self.method_ids.insert(tag, id);
    }

    pub fn cache_field(&mut self, tag: &'static str, id: jlong) {
        self.field_ids.insert(tag, id);
    }

    pub fn class(&self, tag: &str) -> Option<jlong> {
        self.class_ids.get(tag).copied()
    }

    pub fn method(&self, tag: &str) -> Option<jlong> {
        self.method_ids.get(tag).copied()
    }

    pub fn field(&self, tag: &str) -> Option<jlong> {
        self.field_ids.get(tag).copied()
    }
}

// ---------------------------------------------------------------------------
// Render thread attachment model — the dedicated render thread should be
// permanently attached to the JVM so JNI crossings stay cheap.
// ---------------------------------------------------------------------------

/// Indicates whether the current thread has been permanently attached to the
/// JVM. In production the Fabric render thread calls `AttachCurrentThread`
/// once; this flag lets us assert the invariant in debug builds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ThreadAttachment {
    NotAttached,
    Attached,
}

// ---------------------------------------------------------------------------
// Renderer state
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct RendererState {
    uploaded_chunk_sections: u32,
    last_chunk_non_air_blocks: u32,
    jni_cache: JniLookupCache,
    thread_attachment: Option<ThreadAttachment>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RendererHandle(pub i64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RendererInitRequest {
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RendererResizeRequest {
    pub handle: RendererHandle,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RendererFrameRequest {
    pub handle: RendererHandle,
    pub time_seconds: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkSectionUploadRequest<'a> {
    pub handle: RendererHandle,
    pub bytes: &'a [u8],
}

pub fn bridge_status() -> &'static str {
    "ferridian-jni-ready"
}

pub fn init_renderer(request: RendererInitRequest) -> Option<RendererHandle> {
    if request.width == 0 || request.height == 0 {
        return None;
    }

    let handle = RendererHandle(NEXT_RENDERER_HANDLE.fetch_add(1, Ordering::Relaxed));
    LIVE_RENDERERS
        .lock()
        .expect("renderer state poisoned")
        .insert(handle.0, RendererState::default());
    Some(handle)
}

pub fn resize_renderer(request: RendererResizeRequest) -> bool {
    if request.width == 0 || request.height == 0 {
        return false;
    }

    LIVE_RENDERERS
        .lock()
        .expect("renderer state poisoned")
        .contains_key(&request.handle.0)
}

pub fn render_frame(request: RendererFrameRequest) -> bool {
    request.time_seconds.is_finite()
        && LIVE_RENDERERS
            .lock()
            .expect("renderer state poisoned")
            .contains_key(&request.handle.0)
}

pub fn upload_chunk_section(request: ChunkSectionUploadRequest<'_>) -> Option<u32> {
    let packet = ChunkSectionPacket::parse(request.bytes).ok()?;
    let mut renderers = LIVE_RENDERERS.lock().expect("renderer state poisoned");
    let renderer = renderers.get_mut(&request.handle.0)?;
    renderer.uploaded_chunk_sections += 1;
    renderer.last_chunk_non_air_blocks = packet.header.non_air_blocks;
    Some(packet.header.non_air_blocks)
}

/// A frame snapshot captured on the Java main thread and handed to Rust.
/// This packages camera, time, and visible chunk data into a flat structure
/// suitable for zero-copy transfer.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FrameSnapshot {
    pub camera_x: f32,
    pub camera_y: f32,
    pub camera_z: f32,
    pub yaw: f32,
    pub pitch: f32,
    pub time_seconds: f32,
    pub tick_delta: f32,
    pub visible_chunk_count: u32,
}

/// Expected byte size of a serialized FrameSnapshot header.
pub const FRAME_SNAPSHOT_HEADER_BYTES: usize = 32;

/// Submit a frame snapshot from Java-side binary data. This is the prototype
/// for per-frame camera updates over JNI.
pub fn submit_frame_snapshot(handle: RendererHandle, bytes: &[u8]) -> Option<FrameSnapshot> {
    if bytes.len() < FRAME_SNAPSHOT_HEADER_BYTES {
        return None;
    }
    let renderers = LIVE_RENDERERS.lock().expect("renderer state poisoned");
    if !renderers.contains_key(&handle.0) {
        return None;
    }

    fn read_f32(bytes: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }
    fn read_u32(bytes: &[u8], offset: usize) -> u32 {
        u32::from_le_bytes([
            bytes[offset],
            bytes[offset + 1],
            bytes[offset + 2],
            bytes[offset + 3],
        ])
    }

    Some(FrameSnapshot {
        camera_x: read_f32(bytes, 0),
        camera_y: read_f32(bytes, 4),
        camera_z: read_f32(bytes, 8),
        yaw: read_f32(bytes, 12),
        pitch: read_f32(bytes, 16),
        time_seconds: read_f32(bytes, 20),
        tick_delta: read_f32(bytes, 24),
        visible_chunk_count: read_u32(bytes, 28),
    })
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_ferridian_bridge_NativeBridge_ping<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
) -> JString<'local> {
    unowned_env
        .with_env(|env| env.new_string(bridge_status()))
        .resolve::<ThrowRuntimeExAndDefault>()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_ferridian_bridge_RendererBridge_initRendererNative<'local>(
    _env: EnvUnowned<'local>,
    _class: JClass<'local>,
    width: jint,
    height: jint,
) -> jlong {
    init_renderer(RendererInitRequest {
        width: width.max(0) as u32,
        height: height.max(0) as u32,
    })
    .map(|handle| handle.0 as jlong)
    .unwrap_or_default()
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_ferridian_bridge_RendererBridge_resizeRendererNative<'local>(
    _env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    width: jint,
    height: jint,
) {
    let _ = resize_renderer(RendererResizeRequest {
        handle: RendererHandle(handle),
        width: width.max(0) as u32,
        height: height.max(0) as u32,
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_ferridian_bridge_RendererBridge_renderFrameNative<'local>(
    _env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    time_seconds: jfloat,
) {
    let _ = render_frame(RendererFrameRequest {
        handle: RendererHandle(handle),
        time_seconds,
    });
}

#[unsafe(no_mangle)]
pub extern "system" fn Java_io_ferridian_bridge_RendererBridge_uploadChunkSectionNative<'local>(
    mut unowned_env: EnvUnowned<'local>,
    _class: JClass<'local>,
    handle: jlong,
    buffer: JByteBuffer<'local>,
) -> jint {
    let (address, capacity) = unowned_env
        .with_env(|env| {
            let address = env.get_direct_buffer_address(&buffer)?;
            let capacity = env.get_direct_buffer_capacity(&buffer)?;
            Ok::<(usize, usize), jni::errors::Error>((address as usize, capacity))
        })
        .resolve::<ThrowRuntimeExAndDefault>();

    if address == 0 || capacity == 0 {
        return 0;
    }

    let bytes = unsafe { std::slice::from_raw_parts(address as *const u8, capacity) };
    upload_chunk_section(ChunkSectionUploadRequest {
        handle: RendererHandle(handle),
        bytes,
    })
    .unwrap_or_default() as jint
}

#[cfg(test)]
mod tests {
    use super::{
        ChunkSectionUploadRequest, JNI_LIBRARY_NAME, JniLookupCache, RendererFrameRequest,
        RendererHandle, RendererInitRequest, RendererResizeRequest, ThreadAttachment,
        bridge_status, init_renderer, render_frame, resize_renderer, upload_chunk_section,
    };
    use ferridian_utils::{CHUNK_SECTION_LAYOUT_VERSION, CHUNK_SECTION_PACKET_BYTES};

    #[test]
    fn bridge_contract_is_stable() {
        assert_eq!(JNI_LIBRARY_NAME, "ferridian_jni");
        assert_eq!(bridge_status(), "ferridian-jni-ready");
    }

    #[test]
    fn renderer_api_allocates_and_accepts_frames() {
        let handle = init_renderer(RendererInitRequest {
            width: 1280,
            height: 720,
        })
        .expect("valid init request should create a handle");

        assert!(resize_renderer(RendererResizeRequest {
            handle,
            width: 800,
            height: 600,
        }));
        assert!(render_frame(RendererFrameRequest {
            handle,
            time_seconds: 0.25,
        }));
        assert!(!render_frame(RendererFrameRequest {
            handle: RendererHandle(handle.0 + 5000),
            time_seconds: 0.25,
        }));
    }

    #[test]
    fn renderer_accepts_zero_copy_chunk_section_packets() {
        let handle = init_renderer(RendererInitRequest {
            width: 1280,
            height: 720,
        })
        .expect("valid init request should create a handle");
        let mut bytes = vec![0_u8; CHUNK_SECTION_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&CHUNK_SECTION_LAYOUT_VERSION.to_le_bytes());
        bytes[16..20].copy_from_slice(&3_u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&(CHUNK_SECTION_PACKET_BYTES as u32 - 24).to_le_bytes());
        bytes[24] = 1;
        bytes[25] = 2;
        bytes[26] = 3;

        let uploaded = upload_chunk_section(ChunkSectionUploadRequest {
            handle,
            bytes: &bytes,
        });

        assert_eq!(uploaded, Some(3));
    }

    #[test]
    fn jni_lookup_cache_stores_and_retrieves_ids() {
        let mut cache = JniLookupCache::new();
        cache.cache_class("RendererBridge", 100);
        cache.cache_method("initRenderer", 200);
        cache.cache_field("handle", 300);

        assert_eq!(cache.class("RendererBridge"), Some(100));
        assert_eq!(cache.method("initRenderer"), Some(200));
        assert_eq!(cache.field("handle"), Some(300));
        assert_eq!(cache.class("Missing"), None);
    }

    #[test]
    fn thread_attachment_model_tracks_state() {
        assert_ne!(ThreadAttachment::NotAttached, ThreadAttachment::Attached);
    }

    #[test]
    fn frame_snapshot_parses_from_binary() {
        use super::{FRAME_SNAPSHOT_HEADER_BYTES, submit_frame_snapshot};

        let handle = init_renderer(RendererInitRequest {
            width: 800,
            height: 600,
        })
        .unwrap();

        let mut buf = vec![0u8; FRAME_SNAPSHOT_HEADER_BYTES];
        // camera_x = 1.0
        buf[0..4].copy_from_slice(&1.0_f32.to_le_bytes());
        // camera_y = 2.0
        buf[4..8].copy_from_slice(&2.0_f32.to_le_bytes());
        // camera_z = 3.0
        buf[8..12].copy_from_slice(&3.0_f32.to_le_bytes());
        // yaw = 0.5
        buf[12..16].copy_from_slice(&0.5_f32.to_le_bytes());
        // pitch = 0.3
        buf[16..20].copy_from_slice(&0.3_f32.to_le_bytes());
        // time_seconds = 42.0
        buf[20..24].copy_from_slice(&42.0_f32.to_le_bytes());
        // tick_delta = 0.05
        buf[24..28].copy_from_slice(&0.05_f32.to_le_bytes());
        // visible_chunk_count = 16
        buf[28..32].copy_from_slice(&16_u32.to_le_bytes());

        let snapshot = submit_frame_snapshot(handle, &buf).unwrap();
        assert!((snapshot.camera_x - 1.0).abs() < f32::EPSILON);
        assert!((snapshot.camera_y - 2.0).abs() < f32::EPSILON);
        assert!((snapshot.camera_z - 3.0).abs() < f32::EPSILON);
        assert!((snapshot.time_seconds - 42.0).abs() < f32::EPSILON);
        assert_eq!(snapshot.visible_chunk_count, 16);
    }

    #[test]
    fn frame_snapshot_rejects_short_buffer() {
        use super::{FRAME_SNAPSHOT_HEADER_BYTES, submit_frame_snapshot};

        let handle = init_renderer(RendererInitRequest {
            width: 800,
            height: 600,
        })
        .unwrap();

        let buf = vec![0u8; FRAME_SNAPSHOT_HEADER_BYTES - 1];
        assert!(submit_frame_snapshot(handle, &buf).is_none());
    }

    // -----------------------------------------------------------------------
    // init_renderer edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn init_renderer_rejects_zero_width() {
        let result = init_renderer(RendererInitRequest {
            width: 0,
            height: 720,
        });
        assert!(result.is_none());
    }

    #[test]
    fn init_renderer_rejects_zero_height() {
        let result = init_renderer(RendererInitRequest {
            width: 1280,
            height: 0,
        });
        assert!(result.is_none());
    }

    #[test]
    fn init_renderer_handles_allocate_unique() {
        let h1 = init_renderer(RendererInitRequest {
            width: 100,
            height: 100,
        })
        .unwrap();
        let h2 = init_renderer(RendererInitRequest {
            width: 100,
            height: 100,
        })
        .unwrap();
        assert_ne!(h1.0, h2.0, "handles should be unique");
    }

    // -----------------------------------------------------------------------
    // resize_renderer edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn resize_renderer_rejects_zero_dimensions() {
        let handle = init_renderer(RendererInitRequest {
            width: 800,
            height: 600,
        })
        .unwrap();
        assert!(!resize_renderer(RendererResizeRequest {
            handle,
            width: 0,
            height: 600
        }));
        assert!(!resize_renderer(RendererResizeRequest {
            handle,
            width: 800,
            height: 0
        }));
    }

    #[test]
    fn resize_renderer_unknown_handle() {
        let result = resize_renderer(RendererResizeRequest {
            handle: RendererHandle(999999),
            width: 800,
            height: 600,
        });
        assert!(!result);
    }

    // -----------------------------------------------------------------------
    // render_frame edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn render_frame_rejects_nan_time() {
        let handle = init_renderer(RendererInitRequest {
            width: 800,
            height: 600,
        })
        .unwrap();
        assert!(!render_frame(RendererFrameRequest {
            handle,
            time_seconds: f32::NAN,
        }));
    }

    #[test]
    fn render_frame_rejects_infinity() {
        let handle = init_renderer(RendererInitRequest {
            width: 800,
            height: 600,
        })
        .unwrap();
        assert!(!render_frame(RendererFrameRequest {
            handle,
            time_seconds: f32::INFINITY,
        }));
    }

    // -----------------------------------------------------------------------
    // upload_chunk_section edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn upload_chunk_section_rejects_wrong_size() {
        let handle = init_renderer(RendererInitRequest {
            width: 800,
            height: 600,
        })
        .unwrap();
        let result = upload_chunk_section(super::ChunkSectionUploadRequest {
            handle,
            bytes: &[0u8; 10],
        });
        assert!(result.is_none());
    }

    #[test]
    fn upload_chunk_section_unknown_handle() {
        let mut bytes = vec![0_u8; CHUNK_SECTION_PACKET_BYTES];
        bytes[0..4].copy_from_slice(&CHUNK_SECTION_LAYOUT_VERSION.to_le_bytes());
        bytes[16..20].copy_from_slice(&0_u32.to_le_bytes());
        bytes[20..24].copy_from_slice(&(CHUNK_SECTION_PACKET_BYTES as u32 - 24).to_le_bytes());

        let result = upload_chunk_section(super::ChunkSectionUploadRequest {
            handle: RendererHandle(888888),
            bytes: &bytes,
        });
        assert!(result.is_none());
    }

    // -----------------------------------------------------------------------
    // submit_frame_snapshot edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn submit_frame_snapshot_unknown_handle() {
        use super::{FRAME_SNAPSHOT_HEADER_BYTES, submit_frame_snapshot};
        let buf = vec![0u8; FRAME_SNAPSHOT_HEADER_BYTES];
        assert!(submit_frame_snapshot(RendererHandle(777777), &buf).is_none());
    }

    // -----------------------------------------------------------------------
    // JniLookupCache additional tests
    // -----------------------------------------------------------------------

    #[test]
    fn jni_lookup_cache_overwrites_existing() {
        let mut cache = JniLookupCache::new();
        cache.cache_class("Test", 100);
        cache.cache_class("Test", 200);
        assert_eq!(cache.class("Test"), Some(200));
    }

    #[test]
    fn jni_lookup_cache_default_is_empty() {
        let cache = JniLookupCache::default();
        assert_eq!(cache.class("any"), None);
        assert_eq!(cache.method("any"), None);
        assert_eq!(cache.field("any"), None);
    }

    // -----------------------------------------------------------------------
    // RendererHandle / request types
    // -----------------------------------------------------------------------

    #[test]
    fn renderer_handle_equality() {
        let a = RendererHandle(42);
        let b = RendererHandle(42);
        let c = RendererHandle(99);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn renderer_init_request_equality() {
        let a = RendererInitRequest {
            width: 800,
            height: 600,
        };
        let b = RendererInitRequest {
            width: 800,
            height: 600,
        };
        assert_eq!(a, b);
    }

    #[test]
    fn renderer_frame_request_debug() {
        let req = RendererFrameRequest {
            handle: RendererHandle(1),
            time_seconds: 1.5,
        };
        let debug = format!("{req:?}");
        assert!(debug.contains("RendererFrameRequest"));
    }
}
