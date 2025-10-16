# cmtree

[![crates.io](https://img.shields.io/crates/v/cmtree.svg)](https://crates.io/crates/cmtree)
[![docs.rs](https://img.shields.io/docsrs/cmtree/latest.svg)](https://docs.rs/cmtree)
[![CI](https://github.com/sam0x17/cmtree/actions/workflows/ci.yaml/badge.svg)](https://github.com/sam0x17/cmtree/actions/workflows/ci.yaml)
[![License](https://img.shields.io/badge/license-MIT%20%7C%20Apache--2.0-blue.svg)](./LICENSE)

Deterministic, no-std friendly Cartesian Merkle Tree for Rust. `cmtree` blends binary-search
ordering, heap-based balancing, and Merkle authentication so every node carries useful state
and proofs remain compact.

<details>
  <summary>Table of contents</summary>

  - [Features](#features)
  - [Quick start](#quick-start)
  - [Working with proofs](#working-with-proofs)
  - [Determinism & complexity](#determinism--complexity)
  - [Customization](#customization)
  - [Project status](#project-status)
</details>

## Features

- 🚀 **Deterministic treap** – keys derive their priority from a configurable digest, so the
  resulting shape is independent of insertion order.
- 🔐 **Merkle authentication** – generate membership and non-membership proofs for any key in
  `O(log n)` time, with hash ordering that matches the research paper.
- 🧵 **No-std first** – uses `alloc` only; works in embedded and wasm contexts.
- 🧩 **Pluggable hashers** – swap `Sha256` for any `Digest + Clone` such as `blake3` or `sha3`.
- 🧪 **Tested** – extensive unit, doc, and large-structure tests plus a CI pipeline covering
  `cargo fmt`, `clippy`, `doc`, and `test`.

## Quick start

```toml
[dependencies]
cmtree = "0.1"
```

```rust
use cmtree::CMTree;

fn main() {
    let mut tree = CMTree::<Vec<u8>>::new();

    tree.insert(b"alice".to_vec());
    tree.insert(b"bob".to_vec());
    tree.insert(b"carol".to_vec());

    assert!(tree.contains(&b"bob".to_vec()));
    assert_eq!(tree.len(), 3);

    let root = tree.root_hash();
    println!("Root digest: {:02x?}", root);
}
```

## Working with proofs

```rust
use cmtree::CMTree;

let mut tree = CMTree::<Vec<u8>>::new();
for key in [b"x".to_vec(), b"y".to_vec(), b"z".to_vec()] {
    tree.insert(key);
}

let root = tree.root_hash();
let proof = tree.generate_proof(&b"y".to_vec()).unwrap();

assert!(proof.existence);
assert!(proof.verify(&b"y".to_vec(), &root));

let missing = b"not-here".to_vec();
let proof = tree.generate_proof(&missing).unwrap();
assert!(!proof.existence);
assert!(proof.verify(&missing, &root));
```

Proofs follow the definition in [*Cartesian Merkle Tree* (Chystiakov et al.,
2025)](https://arxiv.org/pdf/2504.10944), storing an ordered prefix of parent key digests and
sibling hashes alongside the queried node’s children.

## Determinism & complexity

- `insert`, `remove`, `contains`, `generate_proof`: expected `O(log n)` for `n` stored keys
  (treap balancing).
- `root_hash`: `O(1)` thanks to cached node hashes.
- `Proof` length: `O(log n)`.
- Space: `O(n)`; one node (with cached key digest) per key.

Determinism stems from hashing the key to produce the heap priority. Using a strong digest
ensures priorities act like random values, maintaining balance with high probability.

## Customization

- **Alternative digests** – instantiate `CMTree::<Vec<u8>, sha3::Sha3_256>::new()`,
  `CMTree::<Vec<u8>, Sha512>::new()`, or any other `Digest + Clone` hasher.
- **Smaller priorities** – use `CMTree::<Vec<u8>, Sha256, u64>::new()` (or any
  [`Priority`](https://docs.rs/cmtree/latest/cmtree/trait.Priority.html) implementer) when
  memory pressure outweighs collision resistance.
- **Generic keys** – keys only need `Clone + Ord + Hash`; the tree hashes them into priorities
  and Merkle payloads.
- **No-std environments** – enable `alloc`, disable default features of your digest crate if
  necessary.

## Project status

The library is production-ready, enforced by:

- exhaustive unit suite (including large-tree stress tests),
- doc tests that mirror README examples,
- `cargo fmt`, `clippy -- -D warnings`, `doc`, and `test` checks in CI.

Planned enhancements:

- optional serde support for proofs,
- benchmarking utilities to compare digests,
- safe concurrency primitives for read-mostly workloads.

Feedback and contributions are welcome! Open an issue or pull request to discuss ideas or
report edge cases.
