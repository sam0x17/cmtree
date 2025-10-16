#![no_std]
#![cfg_attr(not(test), deny(missing_docs))]

//! Deterministic Cartesian Merkle Tree implementation.
//!
//! This crate provides a no-std friendly version of the Cartesian Merkle Tree described in
//! <https://arxiv.org/pdf/2504.10944>. A Cartesian Merkle Tree combines binary-search-tree
//! ordering, heap balancing, and Merkle hashing. For each key we derive a deterministic
//! priority from its hash, producing a unique tree layout regardless of insertion order. Every
//! node carries an aggregated Merkle hash, enabling succinct membership and non-membership
//! proofs.
//!
//! # Complexity
//!
//! The structure behaves similarly to a treap whose priorities are derived from the key
//! material. Assuming a strong digest and random-looking priorities, the tree remains balanced
//! with high probability, yielding:
//!
//! * `insert`, `remove`, `contains` – `O(log n)` expected time.
//! * `generate_proof` – `O(log n)` time and proof size.
//! * `root_hash` – `O(1)` time (hashes are cached on each node).
//!
//! Space consumption is `O(n)` for `n` stored keys, with a single node allocated per entry
//! plus cached digests for child subtrees.

extern crate alloc;

use alloc::boxed::Box;
use alloc::vec::Vec;
use core::cmp::{Ordering, min};
use core::hash::{Hash, Hasher};
use sha2::digest::Output;
use sha2::{Digest, Sha256};

/// Digest output for the default [`Sha256`] hasher used by [`CMTree`].
pub type Sha256Hash = Output<Sha256>;

type HashOf<H> = Output<H>;
type Link<T, H> = Option<Box<Node<T, H>>>;

struct Node<T, H>
where
    T: Clone + Ord + Hash,
    H: Digest + Clone,
{
    key: T,
    key_digest: HashOf<H>,
    priority: u128,
    hash: HashOf<H>,
    left: Link<T, H>,
    right: Link<T, H>,
}

impl<T, H> Node<T, H>
where
    T: Clone + Ord + Hash,
    H: Digest + Clone,
{
    #[inline(always)]
    fn new(key: T) -> Self {
        let key_digest = hash_key::<T, H>(&key);
        let priority = leading_u128(key_digest.as_ref());
        let zero = zero_hash::<H>();
        let hash = calculate_node_hash::<H>(&key_digest, &zero, &zero);
        Self {
            key,
            key_digest,
            priority,
            hash,
            left: None,
            right: None,
        }
    }

    #[inline(always)]
    fn left_hash(&self) -> HashOf<H> {
        self.left
            .as_ref()
            .map(|child| child.hash.clone())
            .unwrap_or_else(|| zero_hash::<H>())
    }

    #[inline(always)]
    fn right_hash(&self) -> HashOf<H> {
        self.right
            .as_ref()
            .map(|child| child.hash.clone())
            .unwrap_or_else(|| zero_hash::<H>())
    }

    #[inline(always)]
    fn left_priority(&self) -> u128 {
        self.left
            .as_ref()
            .map(|child| child.priority)
            .unwrap_or_default()
    }

    #[inline(always)]
    fn right_priority(&self) -> u128 {
        self.right
            .as_ref()
            .map(|child| child.priority)
            .unwrap_or_default()
    }

    #[inline(always)]
    fn key_digest(&self) -> &HashOf<H> {
        &self.key_digest
    }

    #[inline(always)]
    fn update_hash(&mut self) {
        let left = self.left_hash();
        let right = self.right_hash();
        self.hash = calculate_node_hash::<H>(self.key_digest(), &left, &right);
    }
}

/// Deterministic Cartesian Merkle Tree backed by a cryptographic digest.
///
/// The structure maintains ordering via the [`Ord`] implementation for the key type `T`,
/// balances using heap rotations directed by deterministic priorities, and produces Merkle
/// proofs based on the digest `H`.
///
/// # Complexity
///
/// * `insert`, `remove`, and [`contains`](Self::contains) run in expected `O(log n)` time,
///   where `n` is the number of stored keys.
/// * [`generate_proof`](Self::generate_proof) executes in `O(log n)` time and produces a proof
///   with `O(log n)` elements.
/// * [`root_hash`](Self::root_hash) reads the cached Merkle hash in `O(1)` time.
///
/// Space usage is `O(n)` for `n` keys, accounting for one node per key and the cached digests
/// for children.
///
/// # Examples
///
/// Basic insertion and membership proof verification:
///
/// ```
/// use cmtree::CMTree;
///
/// let mut tree = CMTree::<Vec<u8>>::new();
/// tree.insert(b"alice".to_vec());
/// tree.insert(b"bob".to_vec());
/// tree.insert(b"carol".to_vec());
///
/// let root = tree.root_hash();
/// let proof = tree.generate_proof(&b"bob".to_vec()).unwrap();
///
/// assert!(proof.existence);
/// assert!(proof.verify(&b"bob".to_vec(), &root));
/// ```
pub struct CMTree<T, H = Sha256>
where
    T: Clone + Ord + Hash,
    H: Digest + Clone,
{
    root: Link<T, H>,
    size: usize,
}

impl<T, H> CMTree<T, H>
where
    T: Clone + Ord + Hash,
    H: Digest + Clone,
{
    /// Creates an empty Cartesian Merkle Tree.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            root: None,
            size: 0,
        }
    }

    /// Returns the number of keys stored in the tree.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.size
    }

    /// Returns whether the tree contains no elements.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.size == 0
    }

    /// Returns the current Merkle root of the tree.
    ///
    /// When the tree is empty the zero hash of the digest is returned.
    #[inline(always)]
    pub fn root_hash(&self) -> HashOf<H> {
        self.root
            .as_ref()
            .map(|node| node.hash.clone())
            .unwrap_or_else(|| zero_hash::<H>())
    }

    /// Inserts a key into the tree.
    ///
    /// Returns `true` if the key did not previously exist.
    #[inline]
    pub fn insert(&mut self, key: T) -> bool {
        let (new_root, inserted) = Self::insert_node(self.root.take(), key);
        self.root = new_root;
        if inserted {
            self.size += 1;
        }
        inserted
    }

    /// Returns `true` if the provided key exists in the tree.
    #[inline]
    pub fn contains(&self, key: &T) -> bool {
        let mut current = self.root.as_deref();
        while let Some(node) = current {
            match key.cmp(&node.key) {
                Ordering::Less => current = node.left.as_deref(),
                Ordering::Greater => current = node.right.as_deref(),
                Ordering::Equal => return true,
            }
        }
        false
    }

    /// Removes the provided key from the tree.
    ///
    /// Returns `true` if the key was present and removed.
    #[inline]
    pub fn remove(&mut self, key: &T) -> bool {
        let (new_root, removed) = Self::remove_node(self.root.take(), key);
        if removed {
            self.size -= 1;
        }
        self.root = new_root;
        removed
    }

    /// Generates a membership or non-membership proof for the provided key.
    ///
    /// Returns `None` when the tree is empty. For membership proofs [`Proof::existence`] is
    /// `true`. For non-membership proofs the closest node encountered during the lookup is
    /// supplied as evidence alongside the queried key's child hashes.
    #[inline]
    pub fn generate_proof(&self, key: &T) -> Option<Proof<H>> {
        let mut current = self.root.as_deref()?;
        let mut path: Vec<(&Node<T, H>, Direction)> = Vec::new();
        let mut existence = false;
        let mut non_existence_key_digest: Option<HashOf<H>> = None;

        let suffix = loop {
            match key.cmp(&current.key) {
                Ordering::Less => {
                    if let Some(left_child) = current.left.as_deref() {
                        path.push((current, Direction::Left));
                        current = left_child;
                    } else {
                        non_existence_key_digest = Some(current.key_digest().clone());
                        break [current.left_hash(), current.right_hash()];
                    }
                }
                Ordering::Greater => {
                    if let Some(right_child) = current.right.as_deref() {
                        path.push((current, Direction::Right));
                        current = right_child;
                    } else {
                        non_existence_key_digest = Some(current.key_digest().clone());
                        break [current.left_hash(), current.right_hash()];
                    }
                }
                Ordering::Equal => {
                    existence = true;
                    break [current.left_hash(), current.right_hash()];
                }
            }
        };

        let mut prefix = Vec::with_capacity(path.len());
        for (node, direction) in path.into_iter().rev() {
            let sibling_hash = match direction {
                Direction::Left => node.right_hash(),
                Direction::Right => node.left_hash(),
            };
            prefix.push(ProofNode {
                parent_key_digest: node.key_digest().clone(),
                sibling_hash,
            });
        }

        Some(Proof {
            prefix,
            suffix,
            existence,
            non_existence_key_digest,
        })
    }

    #[inline]
    fn insert_node(node: Link<T, H>, key: T) -> (Link<T, H>, bool) {
        match node {
            None => (Some(Box::new(Node::new(key))), true),
            Some(mut boxed) => match key.cmp(&boxed.key) {
                Ordering::Less => {
                    let (new_left, inserted) = Self::insert_node(boxed.left.take(), key);
                    boxed.left = new_left;
                    if inserted
                        && boxed
                            .left
                            .as_ref()
                            .is_some_and(|left| left.priority > boxed.priority)
                    {
                        boxed = Self::rotate_right_owned(boxed);
                        return (Some(boxed), true);
                    }
                    boxed.update_hash();
                    (Some(boxed), inserted)
                }
                Ordering::Greater => {
                    let (new_right, inserted) = Self::insert_node(boxed.right.take(), key);
                    boxed.right = new_right;
                    if inserted
                        && boxed
                            .right
                            .as_ref()
                            .is_some_and(|right| right.priority > boxed.priority)
                    {
                        boxed = Self::rotate_left_owned(boxed);
                        return (Some(boxed), true);
                    }
                    boxed.update_hash();
                    (Some(boxed), inserted)
                }
                Ordering::Equal => (Some(boxed), false),
            },
        }
    }

    #[inline]
    fn remove_node(node: Link<T, H>, key: &T) -> (Link<T, H>, bool) {
        let mut boxed = match node {
            Some(node) => node,
            None => return (None, false),
        };

        match key.cmp(&boxed.key) {
            Ordering::Less => {
                let (new_left, removed) = Self::remove_node(boxed.left.take(), key);
                boxed.left = new_left;
                if removed {
                    boxed.update_hash();
                }
                (Some(boxed), removed)
            }
            Ordering::Greater => {
                let (new_right, removed) = Self::remove_node(boxed.right.take(), key);
                boxed.right = new_right;
                if removed {
                    boxed.update_hash();
                }
                (Some(boxed), removed)
            }
            Ordering::Equal => {
                if boxed.left.is_none() {
                    return (boxed.right.take(), true);
                }
                if boxed.right.is_none() {
                    return (boxed.left.take(), true);
                }
                if boxed.left_priority() > boxed.right_priority() {
                    boxed = Self::rotate_right_owned(boxed);
                    let (new_right, removed) = Self::remove_node(boxed.right.take(), key);
                    boxed.right = new_right;
                    boxed.update_hash();
                    (Some(boxed), removed)
                } else {
                    boxed = Self::rotate_left_owned(boxed);
                    let (new_left, removed) = Self::remove_node(boxed.left.take(), key);
                    boxed.left = new_left;
                    boxed.update_hash();
                    (Some(boxed), removed)
                }
            }
        }
    }

    #[inline]
    fn rotate_left_owned(mut node: Box<Node<T, H>>) -> Box<Node<T, H>> {
        let mut right = node
            .right
            .take()
            .expect("rotate_left_owned requires a right child");
        node.right = right.left.take();
        node.update_hash();
        right.left = Some(node);
        right.update_hash();
        right
    }

    #[inline]
    fn rotate_right_owned(mut node: Box<Node<T, H>>) -> Box<Node<T, H>> {
        let mut left = node
            .left
            .take()
            .expect("rotate_right_owned requires a left child");
        node.left = left.right.take();
        node.update_hash();
        left.right = Some(node);
        left.update_hash();
        left
    }
}

impl<T, H> Default for CMTree<T, H>
where
    T: Clone + Ord + Hash,
    H: Digest + Clone,
{
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

/// Authentication data for a single step in a Merkle proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProofNode<H>
where
    H: Digest + Clone,
{
    /// Digest of the parent node's key.
    pub parent_key_digest: HashOf<H>,
    /// Sibling subtree hash encountered on the path to the root.
    pub sibling_hash: HashOf<H>,
}

/// Membership or non-membership proof for a Cartesian Merkle Tree.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Proof<H>
where
    H: Digest + Clone,
{
    /// Path of ancestor nodes from the queried entry up to (but not including) the root.
    pub prefix: Vec<ProofNode<H>>,
    /// Left and right child hashes for the queried entry.
    pub suffix: [HashOf<H>; 2],
    /// Indicates whether this proof represents membership (`true`) or non-membership
    /// (`false`).
    pub existence: bool,
    /// Digest used to demonstrate non-membership when [`Proof::existence`] is `false`.
    pub non_existence_key_digest: Option<HashOf<H>>,
}

impl<H> Proof<H>
where
    H: Digest + Clone,
{
    /// Verifies the proof against the provided key and root hash.
    ///
    /// Membership proofs succeed when the key is present; non-membership proofs succeed when
    /// the key is absent yet the proof demonstrates the unique neighbouring node that prevents
    /// its insertion.
    ///
    /// ```
    /// use cmtree::CMTree;
    ///
    /// let mut tree = CMTree::<Vec<u8>>::new();
    /// for key in [b"a".to_vec(), b"b".to_vec(), b"c".to_vec()] {
    ///     tree.insert(key);
    /// }
    ///
    /// let root = tree.root_hash();
    /// let proof = tree.generate_proof(&b"a".to_vec()).unwrap();
    ///
    /// assert!(proof.existence);
    /// assert!(proof.verify(&b"a".to_vec(), &root));
    /// ```
    #[inline(always)]
    pub fn verify<K>(&self, key: &K, expected_root: &HashOf<H>) -> bool
    where
        K: Hash,
    {
        let key_digest = hash_key::<K, H>(key);
        self.verify_digest(&key_digest, expected_root)
    }

    #[inline(always)]
    fn verify_digest(&self, key_digest: &HashOf<H>, expected_root: &HashOf<H>) -> bool {
        let base_key = if self.existence {
            key_digest
        } else {
            match self.non_existence_key_digest.as_ref() {
                Some(d) => d,
                None => return false,
            }
        };

        let mut acc = calculate_node_hash::<H>(base_key, &self.suffix[0], &self.suffix[1]);
        for node in &self.prefix {
            acc = calculate_node_hash::<H>(&node.parent_key_digest, &acc, &node.sibling_hash);
        }

        &acc == expected_root
    }
}

enum Direction {
    Left,
    Right,
}

struct DigestHasher<H>
where
    H: Digest + Clone,
{
    digest: H,
}

impl<H> DigestHasher<H>
where
    H: Digest + Clone,
{
    #[inline(always)]
    fn new() -> Self {
        Self { digest: H::new() }
    }

    #[inline(always)]
    fn finalize(self) -> HashOf<H> {
        self.digest.finalize()
    }
}

impl<H> Hasher for DigestHasher<H>
where
    H: Digest + Clone,
{
    #[inline(always)]
    fn finish(&self) -> u64 {
        let output = self.digest.clone().finalize();
        let bytes = output.as_ref();
        let mut buf = [0u8; 8];
        let copy_len = min(buf.len(), bytes.len());
        buf[..copy_len].copy_from_slice(&bytes[..copy_len]);
        u64::from_be_bytes(buf)
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        self.digest.update(bytes);
    }
}

#[inline(always)]
fn hash_key<T, H>(key: &T) -> HashOf<H>
where
    T: Hash,
    H: Digest + Clone,
{
    let mut hasher = DigestHasher::<H>::new();
    key.hash(&mut hasher);
    hasher.finalize()
}

#[inline(always)]
fn leading_u128(bytes: &[u8]) -> u128 {
    let mut out = [0u8; 16];
    let copy_len = min(out.len(), bytes.len());
    out[..copy_len].copy_from_slice(&bytes[..copy_len]);
    u128::from_be_bytes(out)
}

#[inline(always)]
fn zero_hash<H: Digest>() -> HashOf<H> {
    Output::<H>::default()
}

#[inline(always)]
fn calculate_node_hash<H: Digest>(
    key_digest: &HashOf<H>,
    left: &HashOf<H>,
    right: &HashOf<H>,
) -> HashOf<H> {
    let (low, high) = if left.as_ref() <= right.as_ref() {
        (left, right)
    } else {
        (right, left)
    };

    let mut hasher = H::new();
    hasher.update(key_digest.as_ref());
    hasher.update(low.as_ref());
    hasher.update(high.as_ref());
    hasher.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    extern crate std;

    fn key(bytes: &[u8]) -> Vec<u8> {
        bytes.to_vec()
    }

    #[test]
    fn insert_and_contains() {
        let mut tree = CMTree::<Vec<u8>>::new();
        let k10 = key(b"10");
        let k5 = key(b"5");
        let k20 = key(b"20");

        assert!(tree.insert(k10.clone()));
        assert!(tree.insert(k5.clone()));
        assert!(tree.insert(k20.clone()));
        assert!(!tree.insert(k5.clone()));

        assert!(tree.contains(&k10));
        assert!(tree.contains(&k5));
        assert!(tree.contains(&k20));
        assert!(!tree.contains(&key(b"1")));
    }

    #[test]
    fn remove_keys() {
        let mut tree = CMTree::<Vec<u8>>::new();
        let k10 = key(b"10");
        let k5 = key(b"5");
        let k20 = key(b"20");
        let k18 = key(b"18");
        let k25 = key(b"25");

        for k in [&k10, &k5, &k20, &k18, &k25] {
            assert!(tree.insert((*k).clone()));
        }
        assert_eq!(tree.len(), 5);
        assert!(tree.remove(&k20));
        assert_eq!(tree.len(), 4);
        assert!(!tree.contains(&k20));
        assert!(tree.remove(&k10));
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn membership_proof_verifies() {
        let mut tree = CMTree::<Vec<u8>>::new();
        let keys = ["10", "5", "20", "18", "25"];
        for k in keys {
            let key_vec = key(k.as_bytes());
            tree.insert(key_vec.clone());
        }
        let root = tree.root_hash();
        let target = key(b"18");
        let proof = tree.generate_proof(&target).unwrap();
        assert!(proof.existence);
        assert!(proof.verify(&target, &root));
    }

    #[test]
    fn non_membership_proof_verifies() {
        let mut tree = CMTree::<Vec<u8>>::new();
        let keys = ["10", "5", "20", "18", "25"];
        for k in keys {
            tree.insert(key(k.as_bytes()));
        }
        let root = tree.root_hash();
        let missing = key(b"13");
        let proof = tree.generate_proof(&missing).unwrap();
        assert!(!proof.existence);
        assert!(proof.verify(&missing, &root));
    }

    #[test]
    fn empty_tree_has_zero_root_and_no_proof() {
        let tree = CMTree::<Vec<u8>>::new();
        assert_eq!(tree.root_hash(), zero_hash::<Sha256>());
        assert!(tree.generate_proof(&key(b"nonexistent")).is_none());
    }

    #[test]
    fn removing_missing_key_does_not_change_tree() {
        let mut tree = CMTree::<Vec<u8>>::new();
        for k in ["1", "2", "3", "4"] {
            assert!(tree.insert(key(k.as_bytes())));
        }
        let len_before = tree.len();
        assert!(!tree.remove(&key(b"999")));
        assert_eq!(tree.len(), len_before);
    }

    #[test]
    fn removing_root_keeps_structure_valid() {
        let mut tree = CMTree::<Vec<u8>>::new();
        let root_key = key(b"10");
        let left_key = key(b"05");
        let right_key = key(b"20");
        for k in [&root_key, &left_key, &right_key] {
            assert!(tree.insert((*k).clone()));
        }
        assert!(tree.remove(&root_key));
        assert_eq!(tree.len(), 2);
        assert!(!tree.contains(&root_key));
        assert!(tree.contains(&left_key));
        assert!(tree.contains(&right_key));
    }

    #[test]
    fn duplicate_insertions_are_idempotent() {
        let mut tree = CMTree::<Vec<u8>>::new();
        let set = ["a", "b", "c", "d", "e"];
        for k in set {
            assert!(tree.insert(key(k.as_bytes())));
        }
        let root_after_first = tree.root_hash();
        for k in set {
            assert!(!tree.insert(key(k.as_bytes())));
        }
        assert_eq!(tree.len(), set.len());
        assert_eq!(tree.root_hash(), root_after_first);
    }

    #[test]
    fn deterministic_root_for_different_insertion_orders() {
        let mut tree_a = CMTree::<Vec<u8>>::new();
        let mut tree_b = CMTree::<Vec<u8>>::new();
        let mut inputs: Vec<Vec<u8>> = ["alpha", "beta", "gamma", "delta", "epsilon"]
            .iter()
            .map(|s| key(s.as_bytes()))
            .collect();
        for k in &inputs {
            tree_a.insert(k.clone());
        }
        inputs.reverse();
        for k in &inputs {
            tree_b.insert(k.clone());
        }
        assert_eq!(tree_a.len(), tree_b.len());
        assert_eq!(tree_a.root_hash(), tree_b.root_hash());
    }

    #[test]
    fn membership_proof_rejects_when_flag_flipped() {
        let mut tree = CMTree::<Vec<u8>>::new();
        for k in ["left", "right", "root", "branch"] {
            tree.insert(key(k.as_bytes()));
        }
        let root = tree.root_hash();
        let target = key(b"branch");
        let mut proof = tree.generate_proof(&target).unwrap();
        proof.existence = false;
        assert!(!proof.verify(&target, &root));
    }

    #[test]
    fn membership_proof_rejects_with_wrong_root() {
        let mut tree_a = CMTree::<Vec<u8>>::new();
        let mut tree_b = CMTree::<Vec<u8>>::new();
        for k in ["1", "2", "3", "4"] {
            let vec_key = key(k.as_bytes());
            tree_a.insert(vec_key.clone());
            tree_b.insert(vec_key);
        }
        tree_b.insert(key(b"extra"));
        let proof = tree_a.generate_proof(&key(b"3")).unwrap();
        assert!(proof.verify(&key(b"3"), &tree_a.root_hash()));
        assert!(!proof.verify(&key(b"3"), &tree_b.root_hash()));
    }

    #[test]
    fn mutated_suffix_breaks_non_membership_proof() {
        let mut tree = CMTree::<Vec<u8>>::new();
        for k in ["10", "5", "20", "18", "25"] {
            tree.insert(key(k.as_bytes()));
        }
        let root = tree.root_hash();
        let mut proof = tree.generate_proof(&key(b"13")).unwrap();
        assert!(!proof.existence);
        proof.suffix[0] = zero_hash::<Sha256>();
        proof.suffix[1] = zero_hash::<Sha256>();
        // By also forcing existence to true we ensure mismatch of accumulator path.
        proof.existence = true;
        assert!(!proof.verify(&key(b"13"), &root));
    }

    #[test]
    fn supports_integer_keys() {
        let mut tree = CMTree::<u64>::new();
        for k in [10, 5, 20, 18, 25] {
            assert!(tree.insert(k));
        }
        assert!(tree.contains(&18u64));
        assert!(!tree.contains(&13u64));
        let root = tree.root_hash();
        let proof = tree.generate_proof(&18u64).unwrap();
        assert!(proof.existence);
        assert!(proof.verify(&18u64, &root));
    }

    #[test]
    fn large_tree_membership_and_non_membership_proofs() {
        const COUNT: usize = 5_000;
        let mut tree = CMTree::<u64>::new();
        for value in 0u64..COUNT as u64 {
            assert!(tree.insert(value));
        }
        assert_eq!(tree.len(), COUNT);

        let target = 3_456u64;
        let root = tree.root_hash();
        let proof = tree.generate_proof(&target).expect("proof should exist");
        assert!(proof.existence);
        assert!(proof.verify(&target, &root));

        let missing = COUNT as u64 + 1;
        let root_again = tree.root_hash();
        let non_membership = tree
            .generate_proof(&missing)
            .expect("proof should be generated");
        assert!(!non_membership.existence);
        assert!(non_membership.verify(&missing, &root_again));
    }

    #[test]
    fn large_tree_sequential_removals() {
        const COUNT: usize = 10_000;
        let mut tree = CMTree::<u64>::new();
        for value in 0u64..COUNT as u64 {
            assert!(tree.insert(value));
        }
        for value in 0u64..COUNT as u64 {
            assert!(tree.remove(&value));
        }
        assert!(tree.is_empty());
        assert_eq!(tree.root_hash(), zero_hash::<Sha256>());
    }
}
