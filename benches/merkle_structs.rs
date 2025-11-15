use cmtree::{CMMap, CMTree};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const BASIC_TREE_SIZE: u64 = 100_000_000;
const BASIC_MAP_SIZE: u64 = 100_000_000;
const BATCH_BASE_SIZE: u64 = 100_000_000;
const BATCH_5K: u64 = 5_000;
const BATCH_10K: u64 = 10_000;

fn build_tree(size: u64) -> CMTree<u64> {
    let mut tree = CMTree::<u64>::new();
    for key in 0..size {
        tree.insert(key);
    }
    tree
}

fn build_map(size: u64) -> CMMap<u64, u64> {
    let mut map = CMMap::<u64, u64>::new();
    for key in 0..size {
        map.insert(key, key * 10);
    }
    map
}

fn cmtree_basic_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmtree_basic_ops");

    group.bench_function("insert_single", |b| {
        let mut tree = build_tree(BASIC_TREE_SIZE);
        let mut next_key = BASIC_TREE_SIZE;
        b.iter(|| {
            next_key += 1;
            black_box(tree.insert(next_key));
        });
    });

    group.bench_function("contains_hit", |b| {
        let tree = build_tree(BASIC_TREE_SIZE);
        let target = (BASIC_TREE_SIZE - 1) / 2;
        b.iter(|| black_box(tree.contains(&target)));
    });

    group.bench_function("contains_miss", |b| {
        let tree = build_tree(BASIC_TREE_SIZE);
        b.iter(|| black_box(tree.contains(&u64::MAX)));
    });

    group.bench_function("remove", |b| {
        let mut tree = build_tree(BASIC_TREE_SIZE);
        let target = (BASIC_TREE_SIZE - 1) / 2;
        b.iter(|| {
            let removed = tree.remove(&target);
            black_box(removed);
            tree.insert(target);
        });
    });

    group.bench_function("generate_proof", |b| {
        let tree = build_tree(BASIC_TREE_SIZE);
        let target = (BASIC_TREE_SIZE - 1) / 2;
        b.iter(|| {
            let proof = tree.generate_proof(&target).expect("proof should exist");
            black_box(proof.prefix.len());
        });
    });

    group.bench_function("root_hash", |b| {
        let tree = build_tree(BASIC_TREE_SIZE);
        b.iter(|| black_box(tree.root_hash()));
    });

    group.finish();
}

fn cmmap_basic_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmmap_basic_ops");

    group.bench_function("insert_single", |b| {
        let mut map = build_map(BASIC_MAP_SIZE);
        let mut next_key = BASIC_MAP_SIZE;
        b.iter(|| {
            next_key += 1;
            black_box(map.insert(next_key, next_key * 10));
        });
    });

    group.bench_function("get_hit", |b| {
        let map = build_map(BASIC_MAP_SIZE);
        let target = (BASIC_MAP_SIZE - 1) / 2;
        b.iter(|| black_box(map.get(&target)));
    });

    group.bench_function("remove", |b| {
        let mut map = build_map(BASIC_MAP_SIZE);
        let target = (BASIC_MAP_SIZE - 1) / 2;
        b.iter(|| {
            let removed = map.remove(&target);
            black_box(&removed);
            map.insert(target, target * 10);
        });
    });

    group.bench_function("generate_proof", |b| {
        let map = build_map(BASIC_MAP_SIZE);
        let target = (BASIC_MAP_SIZE - 1) / 2;
        b.iter(|| {
            let proof = map.generate_proof(&target).expect("proof should exist");
            black_box(proof.prefix.len());
        });
    });

    group.bench_function("root_hash", |b| {
        let map = build_map(BASIC_MAP_SIZE);
        b.iter(|| black_box(map.root_hash()));
    });

    group.finish();
}

fn batch_insert_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert_existing_1m");
    group.sample_size(10);

    let tree_batch_5k: Vec<u64> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_5K).collect();
    let tree_batch_10k: Vec<u64> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_10K).collect();
    let map_batch_5k: Vec<(u64, u64)> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_5K)
        .map(|key| (key, key * 10))
        .collect();
    let map_batch_10k: Vec<(u64, u64)> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_10K)
        .map(|key| (key, key * 10))
        .collect();

    group.bench_function(BenchmarkId::new("cmtree", "5k"), |b| {
        let mut tree = build_tree(BATCH_BASE_SIZE);
        let mut next_start = BATCH_BASE_SIZE;
        b.iter(|| {
            let batch: Vec<u64> = (next_start..next_start + BATCH_5K).collect();
            next_start += BATCH_5K;
            black_box(tree.insert_batch(batch));
        });
    });

    group.bench_function(BenchmarkId::new("cmtree", "10k"), |b| {
        let mut tree = build_tree(BATCH_BASE_SIZE);
        let mut next_start = BATCH_BASE_SIZE;
        b.iter(|| {
            let batch: Vec<u64> = (next_start..next_start + BATCH_10K).collect();
            next_start += BATCH_10K;
            black_box(tree.insert_batch(batch));
        });
    });

    group.bench_function(BenchmarkId::new("cmmap", "5k"), |b| {
        let mut map = build_map(BATCH_BASE_SIZE);
        let mut next_start = BATCH_BASE_SIZE;
        b.iter(|| {
            let batch: Vec<(u64, u64)> = (next_start..next_start + BATCH_5K)
                .map(|k| (k, k * 10))
                .collect();
            next_start += BATCH_5K;
            black_box(map.insert_batch(batch));
        });
    });

    group.bench_function(BenchmarkId::new("cmmap", "10k"), |b| {
        let mut map = build_map(BATCH_BASE_SIZE);
        let mut next_start = BATCH_BASE_SIZE;
        b.iter(|| {
            let batch: Vec<(u64, u64)> = (next_start..next_start + BATCH_10K)
                .map(|k| (k, k * 10))
                .collect();
            next_start += BATCH_10K;
            black_box(map.insert_batch(batch));
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    cmtree_basic_benches,
    cmmap_basic_benches,
    batch_insert_benches
);
criterion_main!(benches);
