use criterion::{Criterion, criterion_group, criterion_main};
use ferridian_core::{ChunkSection, Mesh};

fn bench_sequential_mesher(c: &mut Criterion) {
    let section = ChunkSection::sample_terrain();
    c.bench_function("chunk_mesh_sequential", |b| {
        b.iter(|| Mesh::from_chunk_section(&section));
    });
}

fn bench_parallel_mesher(c: &mut Criterion) {
    let section = ChunkSection::sample_terrain();
    c.bench_function("chunk_mesh_parallel", |b| {
        b.iter(|| Mesh::from_chunk_section_parallel(&section));
    });
}

criterion_group!(benches, bench_sequential_mesher, bench_parallel_mesher);
criterion_main!(benches);
