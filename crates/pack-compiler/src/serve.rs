//! The change-detection half of `packc serve` (hot reload, M3).
//!
//! Deliberately a polling snapshot diff over std, not an OS watcher: a pack
//! is a handful of files, a dev tool can afford 200ms latency, and this way
//! the logic is a pure function of two directory scans — testable, and
//! identical on every platform.
//!
//! The reload handshake with a running engine is a file: every successful
//! rebuild atomically bumps `<out>/generation` (write-temp + rename) to a new
//! monotonic value. A consumer polls that one file and re-reads the artifact
//! when the value changes; the rename guarantees it never observes a torn
//! artifact-in-progress marker.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use crate::CompileError;

/// What one source file looked like at scan time. Modification time *and*
/// length: editors that preserve mtime on save are rare, truncate-then-write
/// races are not.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    modified: SystemTime,
    len: u64,
}

/// Every source file under a pack directory at one instant.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SourceSnapshot {
    entries: BTreeMap<PathBuf, FileStamp>,
}

impl SourceSnapshot {
    /// Scan `pack_dir` recursively, skipping `exclude` (the artifact output
    /// lives inside the pack by default — watching it would rebuild forever).
    pub fn scan(pack_dir: &Path, exclude: &Path) -> Result<SourceSnapshot, CompileError> {
        let mut entries = BTreeMap::new();
        let mut pending = vec![pack_dir.to_owned()];
        let io_err = |path: &Path, source| CompileError::Io {
            path: path.to_owned(),
            source,
        };
        while let Some(dir) = pending.pop() {
            for entry in fs::read_dir(&dir).map_err(|e| io_err(&dir, e))? {
                let entry = entry.map_err(|e| io_err(&dir, e))?;
                let path = entry.path();
                if path == exclude {
                    continue;
                }
                let metadata = match entry.metadata() {
                    Ok(metadata) => metadata,
                    // A file deleted mid-scan is just a change the next scan
                    // will settle on.
                    Err(_) => continue,
                };
                if metadata.is_dir() {
                    pending.push(path);
                } else if metadata.is_file() {
                    entries.insert(
                        path,
                        FileStamp {
                            modified: metadata.modified().map_err(|e| io_err(&dir, e))?,
                            len: metadata.len(),
                        },
                    );
                }
            }
        }
        Ok(SourceSnapshot { entries })
    }

    /// Paths that were added, removed, or modified between `self` and `next`.
    pub fn changes_since(&self, next: &SourceSnapshot) -> Vec<PathBuf> {
        let mut changed: Vec<PathBuf> = Vec::new();
        for (path, stamp) in &next.entries {
            if self.entries.get(path) != Some(stamp) {
                changed.push(path.clone());
            }
        }
        for path in self.entries.keys() {
            if !next.entries.contains_key(path) {
                changed.push(path.clone());
            }
        }
        changed.sort();
        changed
    }
}

/// Atomically publish a new artifact generation: consumers polling this file
/// re-read the artifact when the number changes, and the rename means they
/// never see a partial write.
pub fn write_generation(out_dir: &Path, generation: u64) -> Result<(), CompileError> {
    let io_err = |path: &Path, source| CompileError::Io {
        path: path.to_owned(),
        source,
    };
    fs::create_dir_all(out_dir).map_err(|e| io_err(out_dir, e))?;
    let temp = out_dir.join(".generation.tmp");
    let path = out_dir.join("generation");
    fs::write(&temp, format!("{generation}\n")).map_err(|e| io_err(&temp, e))?;
    fs::rename(&temp, &path).map_err(|e| io_err(&path, e))?;
    Ok(())
}

/// Read the published generation; `None` if never published (or unreadable —
/// a consumer treats both as "artifact not ready yet").
pub fn read_generation(out_dir: &Path) -> Option<u64> {
    fs::read_to_string(out_dir.join("generation"))
        .ok()?
        .trim()
        .parse()
        .ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// A fresh scratch pack directory per test (std-only tempdir).
    fn scratch_dir() -> PathBuf {
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        let dir = std::env::temp_dir().join(format!(
            "ferridian-serve-test-{}-{}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        fs::create_dir_all(dir.join("shaders")).expect("create scratch pack");
        dir
    }

    #[test]
    fn detects_add_modify_remove_and_ignores_excluded() {
        let pack = scratch_dir();
        let build = pack.join("build");
        fs::write(pack.join("pack.toml"), "a").unwrap();
        fs::write(pack.join("shaders/one.slang"), "one").unwrap();
        let before = SourceSnapshot::scan(&pack, &build).unwrap();
        assert!(before.changes_since(&before).is_empty());

        // Output-dir churn must be invisible to the watcher.
        fs::create_dir_all(&build).unwrap();
        fs::write(build.join("generation"), "1").unwrap();
        // Content change without a length change: mtime must carry it. File
        // mtime granularity can be coarse; force a distinct stamp.
        fs::write(pack.join("shaders/one.slang"), "two").unwrap();
        let stamp = SystemTime::now() + std::time::Duration::from_secs(2);
        let file = fs::File::options()
            .append(true)
            .open(pack.join("shaders/one.slang"))
            .unwrap();
        file.set_modified(stamp).unwrap();
        drop(file);
        fs::write(pack.join("shaders/new.slang"), "fresh").unwrap();

        let after = SourceSnapshot::scan(&pack, &build).unwrap();
        let changes = before.changes_since(&after);
        assert_eq!(
            changes,
            vec![
                pack.join("shaders/new.slang"),
                pack.join("shaders/one.slang")
            ]
        );

        fs::remove_file(pack.join("shaders/new.slang")).unwrap();
        let removed = SourceSnapshot::scan(&pack, &build).unwrap();
        assert_eq!(
            after.changes_since(&removed),
            vec![pack.join("shaders/new.slang")]
        );

        fs::remove_dir_all(&pack).ok();
    }

    #[test]
    fn generation_round_trips_and_starts_absent() {
        let out = scratch_dir().join("build");
        assert_eq!(read_generation(&out), None);
        write_generation(&out, 1).unwrap();
        assert_eq!(read_generation(&out), Some(1));
        write_generation(&out, 42).unwrap();
        assert_eq!(read_generation(&out), Some(42));
        assert!(
            !out.join(".generation.tmp").exists(),
            "temp file must not survive the rename"
        );
        fs::remove_dir_all(out.parent().unwrap()).ok();
    }
}
