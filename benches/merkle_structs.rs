use cmtree::{CMMap, CMTree};
use criterion::{BenchmarkId, Criterion, black_box, criterion_group, criterion_main};

const BASIC_TREE_SIZE: u64 = 1_000_000;
const BASIC_MAP_SIZE: u64 = 1_000_000;
const BATCH_BASE_SIZE: u64 = 1_000_000;
const BATCH_SMALL: u64 = 2_000;
const BATCH_LARGE: u64 = 10_000;
fn build_tree(size: u64) -> CMTree<u64> {
    eprintln!("Building tree of size {}...", size);
    let mut tree = CMTree::<u64>::new();
    if size > 0 {
        let batch: Vec<u64> = (0..size).collect();
        tree.insert_batch(batch);
    }
    eprintln!("Done.");
    tree
}

fn build_map(size: u64) -> CMMap<u64, u64> {
    eprintln!("Building map of size {}...", size);
    let mut map = CMMap::<u64, u64>::new();
    if size > 0 {
        let batch: Vec<(u64, u64)> = (0..size).map(|k| (k, k * 10)).collect();
        map.insert_batch(batch);
    }
    eprintln!("Done.");
    map
}

fn cmtree_basic_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmtree_basic_ops");
    let base_tree = build_tree(BASIC_TREE_SIZE);
    eprintln!("Finished building base tree for benchmarks.");
    eprintln!("Cloning base tree for insert benchmarks...");
    let mut insert_tree = base_tree.clone();
    eprintln!("Cloning base tree for contains hit benchmarks...");
    let contains_tree = base_tree.clone();
    eprintln!("Cloning base tree for contains miss benchmarks...");
    let contains_miss_tree = base_tree.clone();
    eprintln!("Cloning base tree for remove benchmarks...");
    let mut remove_tree = base_tree.clone();
    eprintln!("Cloning base tree for proof generation benchmarks...");
    let proof_tree = base_tree.clone();
    eprintln!("Cloning base tree for root hash benchmarks...");
    let root_tree = base_tree.clone();
    eprintln!("All clones complete. Starting benchmarks...");

    group.bench_function("insert_single", |b| {
        let mut next_key = BASIC_TREE_SIZE;
        b.iter(|| {
            next_key += 1;
            black_box(insert_tree.insert(next_key));
        });
    });

    group.bench_function("contains_hit", |b| {
        let target = (BASIC_TREE_SIZE - 1) / 2;
        b.iter(|| black_box(contains_tree.contains(&target)));
    });

    group.bench_function("contains_miss", |b| {
        b.iter(|| black_box(contains_miss_tree.contains(&u64::MAX)));
    });

    group.bench_function("remove", |b| {
        let target = (BASIC_TREE_SIZE - 1) / 2;
        b.iter(|| {
            let removed = remove_tree.remove(&target);
            black_box(removed);
            remove_tree.insert(target);
        });
    });

    group.bench_function("generate_proof", |b| {
        let target = (BASIC_TREE_SIZE - 1) / 2;
        b.iter(|| {
            let proof = proof_tree
                .generate_proof(&target)
                .expect("proof should exist");
            black_box(proof.prefix.len());
        });
    });

    group.bench_function("root_hash", |b| {
        b.iter(|| black_box(root_tree.root_hash()));
    });

    group.finish();
}

fn cmmap_basic_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("cmmap_basic_ops");
    let base_map = build_map(BASIC_MAP_SIZE);
    let mut insert_map = base_map.clone();
    let get_map = base_map.clone();
    let mut remove_map = base_map.clone();
    let proof_map = base_map.clone();
    let root_map = base_map.clone();

    group.bench_function("insert_single", |b| {
        let mut next_key = BASIC_MAP_SIZE;
        b.iter(|| {
            next_key += 1;
            black_box(insert_map.insert(next_key, next_key * 10));
        });
    });

    group.bench_function("get_hit", |b| {
        let target = (BASIC_MAP_SIZE - 1) / 2;
        b.iter(|| black_box(get_map.get(&target)));
    });

    group.bench_function("remove", |b| {
        let target = (BASIC_MAP_SIZE - 1) / 2;
        b.iter(|| {
            let removed = remove_map.remove(&target);
            black_box(&removed);
            remove_map.insert(target, target * 10);
        });
    });

    group.bench_function("generate_proof", |b| {
        let target = (BASIC_MAP_SIZE - 1) / 2;
        b.iter(|| {
            let proof = proof_map
                .generate_proof(&target)
                .expect("proof should exist");
            black_box(proof.prefix.len());
        });
    });

    group.bench_function("root_hash", |b| {
        b.iter(|| black_box(root_map.root_hash()));
    });

    group.finish();
}

fn batch_insert_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_insert");

    let tree_batch_small: Vec<u64> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_SMALL).collect();
    let tree_batch_large: Vec<u64> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_LARGE).collect();
    let map_batch_small: Vec<(u64, u64)> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_SMALL)
        .map(|key| (key, key * 10))
        .collect();
    let map_batch_large: Vec<(u64, u64)> = (BATCH_BASE_SIZE..BATCH_BASE_SIZE + BATCH_LARGE)
        .map(|key| (key, key * 10))
        .collect();

    let mut tree_small = build_tree(BATCH_BASE_SIZE);
    let mut tree_large = tree_small.clone();
    let mut map_small = build_map(BATCH_BASE_SIZE);
    let mut map_large = map_small.clone();

    group.bench_function(BenchmarkId::new("cmtree", "small"), |b| {
        b.iter(|| {
            black_box(tree_small.insert_batch(tree_batch_small.iter().copied()));
        });
    });

    group.bench_function(BenchmarkId::new("cmtree", "large"), |b| {
        b.iter(|| {
            black_box(tree_large.insert_batch(tree_batch_large.iter().copied()));
        });
    });

    group.bench_function(BenchmarkId::new("cmmap", "small"), |b| {
        b.iter(|| {
            black_box(map_small.insert_batch(map_batch_small.iter().cloned()));
        });
    });

    group.bench_function(BenchmarkId::new("cmmap", "large"), |b| {
        b.iter(|| {
            black_box(map_large.insert_batch(map_batch_large.iter().cloned()));
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
