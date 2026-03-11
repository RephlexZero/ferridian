use jni::EnvUnowned;
use jni::errors::ThrowRuntimeExAndDefault;
use jni::objects::{JClass, JString};
use jni::sys::{jfloat, jint, jlong};
use std::collections::HashSet;
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{LazyLock, Mutex};

pub const JNI_LIBRARY_NAME: &str = "ferridian_jni";
static NEXT_RENDERER_HANDLE: AtomicI64 = AtomicI64::new(1);
static LIVE_RENDERERS: LazyLock<Mutex<HashSet<i64>>> = LazyLock::new(|| Mutex::new(HashSet::new()));

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
        .insert(handle.0);
    Some(handle)
}

pub fn resize_renderer(request: RendererResizeRequest) -> bool {
    if request.width == 0 || request.height == 0 {
        return false;
    }

    LIVE_RENDERERS
        .lock()
        .expect("renderer state poisoned")
        .contains(&request.handle.0)
}

pub fn render_frame(request: RendererFrameRequest) -> bool {
    request.time_seconds.is_finite()
        && LIVE_RENDERERS
            .lock()
            .expect("renderer state poisoned")
            .contains(&request.handle.0)
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

#[cfg(test)]
mod tests {
    use super::{
        JNI_LIBRARY_NAME, RendererFrameRequest, RendererHandle, RendererInitRequest,
        RendererResizeRequest, bridge_status, init_renderer, render_frame, resize_renderer,
    };

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
}
