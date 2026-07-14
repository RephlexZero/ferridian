//! A minimal JVM classfile parser: just enough to derive *signature
//! inventories* (class/member names + descriptors) from unobfuscated jars.
//!
//! Parsing signatures straight from the classfile beats decompiling for this
//! job: it is deterministic, needs no JVM in CI, and structurally cannot
//! reproduce Mojang source — only derived names ever leave the parser (the
//! legal guardrail, §5 of the plan).
//!
//! Jars are untrusted input, so this parser is written to be fuzzed: every
//! read is bounds-checked, every error is a value, and there is no recursion.

/// One field or method: raw access flags plus name and JVM descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Member {
    pub access_flags: u16,
    pub name: String,
    pub descriptor: String,
}

/// The subset of a parsed classfile that signature inventories need.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedClass {
    /// Binary name with slashes, e.g. `net/minecraft/client/Minecraft`.
    pub binary_name: String,
    pub super_name: Option<String>,
    pub interfaces: Vec<String>,
    pub access_flags: u16,
    pub fields: Vec<Member>,
    pub methods: Vec<Member>,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ClassfileError {
    #[error("not a classfile (bad magic)")]
    BadMagic,
    #[error("classfile truncated at byte {0}")]
    Truncated(usize),
    #[error("unknown constant pool tag {0}")]
    UnknownConstantTag(u8),
    #[error("constant pool index {0} is invalid for its use")]
    BadConstantIndex(u16),
}

/// Parse the parts of a classfile the inventory needs.
pub fn parse_class(bytes: &[u8]) -> Result<ParsedClass, ClassfileError> {
    let mut reader = Reader { bytes, offset: 0 };
    if reader.u32()? != 0xCAFE_BABE {
        return Err(ClassfileError::BadMagic);
    }
    reader.skip(4)?; // minor + major version

    let pool = ConstantPool::parse(&mut reader)?;

    let access_flags = reader.u16()?;
    let this_class = reader.u16()?;
    let super_class = reader.u16()?;
    let binary_name = pool.class_name(this_class)?;
    let super_name = if super_class == 0 {
        None
    } else {
        Some(pool.class_name(super_class)?)
    };

    let interface_count = reader.u16()?;
    let mut interfaces = Vec::with_capacity(usize::from(interface_count).min(64));
    for _ in 0..interface_count {
        interfaces.push(pool.class_name(reader.u16()?)?);
    }

    let fields = parse_members(&mut reader, &pool)?;
    let methods = parse_members(&mut reader, &pool)?;

    Ok(ParsedClass {
        binary_name,
        super_name,
        interfaces,
        access_flags,
        fields,
        methods,
    })
}

fn parse_members(
    reader: &mut Reader<'_>,
    pool: &ConstantPool,
) -> Result<Vec<Member>, ClassfileError> {
    let count = reader.u16()?;
    let mut members = Vec::with_capacity(usize::from(count).min(256));
    for _ in 0..count {
        let access_flags = reader.u16()?;
        let name = pool.utf8(reader.u16()?)?;
        let descriptor = pool.utf8(reader.u16()?)?;
        skip_attributes(reader)?;
        members.push(Member {
            access_flags,
            name,
            descriptor,
        });
    }
    Ok(members)
}

fn skip_attributes(reader: &mut Reader<'_>) -> Result<(), ClassfileError> {
    let count = reader.u16()?;
    for _ in 0..count {
        reader.skip(2)?; // attribute name index
        let length = reader.u32()?;
        reader.skip(length as usize)?;
    }
    Ok(())
}

struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Result<&[u8], ClassfileError> {
        let end = self
            .offset
            .checked_add(n)
            .filter(|&end| end <= self.bytes.len())
            .ok_or(ClassfileError::Truncated(self.offset))?;
        let slice = &self.bytes[self.offset..end];
        self.offset = end;
        Ok(slice)
    }

    fn skip(&mut self, n: usize) -> Result<(), ClassfileError> {
        self.take(n).map(|_| ())
    }

    fn u8(&mut self) -> Result<u8, ClassfileError> {
        Ok(self.take(1)?[0])
    }

    fn u16(&mut self) -> Result<u16, ClassfileError> {
        let b = self.take(2)?;
        Ok(u16::from_be_bytes([b[0], b[1]]))
    }

    fn u32(&mut self) -> Result<u32, ClassfileError> {
        let b = self.take(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
}

enum Constant {
    Utf8(String),
    Class { name_index: u16 },
    Other,
}

struct ConstantPool {
    /// Indexed 1..count; slot 0 and the second slot of long/double are `Other`.
    entries: Vec<Constant>,
}

impl ConstantPool {
    fn parse(reader: &mut Reader<'_>) -> Result<ConstantPool, ClassfileError> {
        let count = reader.u16()?;
        let mut entries = Vec::with_capacity(usize::from(count).min(1 << 12));
        entries.push(Constant::Other); // slot 0 is unused by the format
        let mut index = 1;
        while index < count {
            let tag = reader.u8()?;
            let (constant, extra_slot) = match tag {
                1 => {
                    let length = usize::from(reader.u16()?);
                    let bytes = reader.take(length)?;
                    // JVM "modified UTF-8" differs from UTF-8 only in
                    // surrogate/NUL encoding; identifiers we care about are
                    // ASCII, so lossy decoding is exact in practice.
                    (
                        Constant::Utf8(String::from_utf8_lossy(bytes).into_owned()),
                        false,
                    )
                }
                7 => (
                    Constant::Class {
                        name_index: reader.u16()?,
                    },
                    false,
                ),
                3 | 4 => {
                    reader.skip(4)?;
                    (Constant::Other, false)
                }
                5 | 6 => {
                    reader.skip(8)?;
                    (Constant::Other, true)
                }
                8 | 16 | 19 | 20 => {
                    reader.skip(2)?;
                    (Constant::Other, false)
                }
                9..=12 | 17 | 18 => {
                    reader.skip(4)?;
                    (Constant::Other, false)
                }
                15 => {
                    reader.skip(3)?;
                    (Constant::Other, false)
                }
                other => return Err(ClassfileError::UnknownConstantTag(other)),
            };
            entries.push(constant);
            index += 1;
            if extra_slot {
                entries.push(Constant::Other);
                index += 1;
            }
        }
        Ok(ConstantPool { entries })
    }

    fn utf8(&self, index: u16) -> Result<String, ClassfileError> {
        match self.entries.get(usize::from(index)) {
            Some(Constant::Utf8(text)) => Ok(text.clone()),
            _ => Err(ClassfileError::BadConstantIndex(index)),
        }
    }

    fn class_name(&self, index: u16) -> Result<String, ClassfileError> {
        match self.entries.get(usize::from(index)) {
            Some(&Constant::Class { name_index }) => self.utf8(name_index),
            _ => Err(ClassfileError::BadConstantIndex(index)),
        }
    }
}

// --- Signature rendering -------------------------------------------------

const ACC_PUBLIC: u16 = 0x0001;
const ACC_PRIVATE: u16 = 0x0002;
const ACC_PROTECTED: u16 = 0x0004;
const ACC_STATIC: u16 = 0x0008;
const ACC_FINAL: u16 = 0x0010;
const ACC_ABSTRACT: u16 = 0x0400;

fn modifiers(access_flags: u16) -> String {
    let mut out = String::new();
    if access_flags & ACC_PUBLIC != 0 {
        out.push_str("public ");
    } else if access_flags & ACC_PROTECTED != 0 {
        out.push_str("protected ");
    } else if access_flags & ACC_PRIVATE != 0 {
        out.push_str("private ");
    }
    if access_flags & ACC_STATIC != 0 {
        out.push_str("static ");
    }
    if access_flags & ACC_FINAL != 0 {
        out.push_str("final ");
    }
    if access_flags & ACC_ABSTRACT != 0 {
        out.push_str("abstract ");
    }
    out
}

/// Render a method as a Java-like signature, e.g.
/// `public void renderLevel(net.minecraft.client.DeltaTracker, boolean)`.
pub fn render_method(member: &Member) -> String {
    let (params, ret) = split_method_descriptor(&member.descriptor);
    format!(
        "{}{} {}({})",
        modifiers(member.access_flags),
        ret,
        member.name,
        params.join(", ")
    )
}

/// Render a field as a Java-like signature, e.g. `private final int frameId`.
pub fn render_field(member: &Member) -> String {
    format!(
        "{}{} {}",
        modifiers(member.access_flags),
        render_type(&mut member.descriptor.chars().peekable()),
        member.name
    )
}

fn split_method_descriptor(descriptor: &str) -> (Vec<String>, String) {
    let mut chars = descriptor.chars().peekable();
    let mut params = Vec::new();
    if chars.peek() == Some(&'(') {
        chars.next();
        while chars.peek().is_some_and(|&c| c != ')') {
            params.push(render_type(&mut chars));
        }
        chars.next(); // ')'
    }
    let ret = render_type(&mut chars);
    (params, ret)
}

/// Render one JVM type descriptor as Java source syntax. Malformed input
/// (possible — jars are untrusted) renders as `?` rather than erroring: the
/// inventory diff still works on it.
fn render_type(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    // Array dimensions iteratively: a hostile descriptor can nest `[` far
    // deeper than the JVM's 255 limit, so recursing per dimension would let
    // untrusted input pick our stack depth.
    let mut dimensions = 0usize;
    while chars.peek() == Some(&'[') {
        chars.next();
        dimensions += 1;
    }
    let mut rendered = render_scalar_type(chars);
    rendered.reserve(dimensions * 2);
    for _ in 0..dimensions {
        rendered.push_str("[]");
    }
    rendered
}

fn render_scalar_type(chars: &mut std::iter::Peekable<std::str::Chars<'_>>) -> String {
    match chars.next() {
        Some('B') => "byte".to_owned(),
        Some('C') => "char".to_owned(),
        Some('D') => "double".to_owned(),
        Some('F') => "float".to_owned(),
        Some('I') => "int".to_owned(),
        Some('J') => "long".to_owned(),
        Some('S') => "short".to_owned(),
        Some('Z') => "boolean".to_owned(),
        Some('V') => "void".to_owned(),
        Some('L') => {
            let mut name = String::new();
            for c in chars.by_ref() {
                if c == ';' {
                    break;
                }
                name.push(if c == '/' { '.' } else { c });
            }
            name
        }
        _ => "?".to_owned(),
    }
}

/// All signatures of a parsed class, sorted — the exact strings that land in
/// a committed [`crate::SignatureInventory`].
pub fn class_signatures(class: &ParsedClass) -> Vec<String> {
    let mut signatures: Vec<String> = class
        .methods
        .iter()
        .map(render_method)
        .chain(class.fields.iter().map(render_field))
        .collect();
    signatures.sort();
    signatures
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_method_descriptors() {
        let member = Member {
            access_flags: ACC_PUBLIC | ACC_STATIC,
            name: "blend".to_owned(),
            descriptor: "(Lnet/minecraft/Foo;[IF)Z".to_owned(),
        };
        assert_eq!(
            render_method(&member),
            "public static boolean blend(net.minecraft.Foo, int[], float)"
        );
    }

    #[test]
    fn renders_field_descriptors() {
        let member = Member {
            access_flags: ACC_PRIVATE | ACC_FINAL,
            name: "frames".to_owned(),
            descriptor: "[[J".to_owned(),
        };
        assert_eq!(render_field(&member), "private final long[][] frames");
    }

    #[test]
    fn hostile_array_nesting_does_not_recurse() {
        // 100k dimensions must not blow the stack (JVM caps at 255; hostile
        // jars don't care about caps).
        let member = Member {
            access_flags: 0,
            name: "deep".to_owned(),
            descriptor: format!("{}I", "[".repeat(100_000)),
        };
        let rendered = render_field(&member);
        assert!(rendered.starts_with("int[]"));
        assert_eq!(rendered.matches("[]").count(), 100_000);
    }

    #[test]
    fn malformed_descriptor_degrades_to_question_mark() {
        let member = Member {
            access_flags: 0,
            name: "x".to_owned(),
            descriptor: "(Q".to_owned(),
        };
        // Never panics on garbage — jars are untrusted input.
        assert_eq!(render_method(&member), "? x(?)");
    }

    #[test]
    fn rejects_bad_magic() {
        assert_eq!(
            parse_class(b"PK\x03\x04junk"),
            Err(ClassfileError::BadMagic)
        );
    }

    #[test]
    fn rejects_truncation_without_panicking() {
        // The magic + version + a constant pool count promising more than exists.
        let bytes = [0xCA, 0xFE, 0xBA, 0xBE, 0, 0, 0, 61, 0, 9, 1];
        assert!(matches!(
            parse_class(&bytes),
            Err(ClassfileError::Truncated(_))
        ));
    }
}
