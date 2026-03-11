use jni::EnvUnowned;
use jni::errors::ThrowRuntimeExAndDefault;
use jni::objects::{JClass, JString};

pub const JNI_LIBRARY_NAME: &str = "ferridian_jni";

pub fn bridge_status() -> &'static str {
    "ferridian-jni-ready"
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

#[cfg(test)]
mod tests {
    use super::{JNI_LIBRARY_NAME, bridge_status};

    #[test]
    fn bridge_contract_is_stable() {
        assert_eq!(JNI_LIBRARY_NAME, "ferridian_jni");
        assert_eq!(bridge_status(), "ferridian-jni-ready");
    }
}
