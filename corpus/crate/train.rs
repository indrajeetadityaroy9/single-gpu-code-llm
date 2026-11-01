/*! Byte-level BPE training with source-attributed BPE (SA-BPE, arXiv 2604.14053 §4.1; paper §3.1). Each merge takes
    the pair with the highest score F·ln(R), where F is the pair's frequency and R the number of distinct sources
    (repositories) that contain it, so a pair seen in one source scores 0 (ln as in pchizhov/sa-bpe scoring.py). Ties
    go to the higher F, then the smaller pair. The trainer follows rustbpe, and text is split into pre-tokens by
    fancy-regex as tiktoken (openai/tiktoken) does. R is exact for every merge: the heap holds upper bounds on R,
    and an entry is recounted when it reaches the top. */

use std::cmp::Ordering;
use std::collections::BinaryHeap;
use std::hash::BuildHasher;

use fancy_regex::Regex;
use fixedbitset::FixedBitSet;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use rayon::prelude::*;
use rustc_hash::{FxBuildHasher, FxHashMap, FxHashSet};

use crate::PerWorker;

type Pair = (u32, u32);

/** A distinct pre-token ("word"): its token ids, its number of occurrences and its sorted distinct sources. */
struct Word {
    ids: Vec<u32>,
    count: i64,
    sources: Vec<u32>,
}

impl Word {
    fn contains(&self, pair: Pair) -> bool {
        self.ids.windows(2).any(|w| (w[0], w[1]) == pair)
    }

    /** Merges every non-overlapping occurrence of pair into new_id, left to right; returns the change in each
        affected pair's count (-1 removed, +1 created; as rustbpe's merge_pair). */
    fn merge_pair(&mut self, pair: Pair, new_id: u32) -> Vec<(Pair, i64)> {
        let (a, b) = pair;
        let n = self.ids.len();
        let mut out = Vec::with_capacity(n);
        let mut deltas = Vec::new();
        let mut i = 0;
        while i < n {
            if i + 1 < n && self.ids[i] == a && self.ids[i + 1] == b {
                if let Some(&x) = out.last() {
                    deltas.push(((x, a), -1));
                    deltas.push(((x, new_id), 1));
                }
                deltas.push(((a, b), -1));
                if i + 2 < n {
                    let y = self.ids[i + 2];
                    deltas.push(((b, y), -1));
                    deltas.push(((new_id, y), 1));
                }
                out.push(new_id);
                i += 2;
            } else {
                out.push(self.ids[i]);
                i += 1;
            }
        }
        self.ids = out;
        deltas
    }
}

/** A pair's heap entry: F, R (exact when `exact`, else an upper bound), the score, and the words it may occur in. */
struct Job {
    pair: Pair,
    freq: i64,
    sources: u32,
    exact: bool,
    score: f64,
    words: Vec<u32>,
}

impl Ord for Job {
    fn cmp(&self, other: &Self) -> Ordering {
        self.score.total_cmp(&other.score).then(self.freq.cmp(&other.freq)).then(other.pair.cmp(&self.pair))
    }
}

impl PartialOrd for Job {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Job {}

/** The distinct pre-tokens of the documents (document d from source sources[d]) as words of byte ids. */
fn count_words<'a>(texts: &'a [PyBackedStr], sources: &[u32], pattern: &PerWorker<Regex>) -> Vec<Word> {
    struct Chunk {
        count: i64,
        sources: Vec<u32>,
        last_doc: usize,
    }
    let shards = rayon::current_num_threads();
    let parts: Vec<Vec<FxHashMap<&'a str, Chunk>>> = (0..texts.len())
        .into_par_iter()
        .fold(
            || (0..shards).map(|_| FxHashMap::default()).collect::<Vec<_>>(),
            |mut tables, doc| {
                for piece in pattern.get().find_iter(texts[doc].as_str()) {
                    let piece = piece.unwrap().as_str();
                    let chunk = tables[FxBuildHasher.hash_one(piece) as usize % shards]
                        .entry(piece)
                        .or_insert_with(|| Chunk { count: 0, sources: Vec::new(), last_doc: usize::MAX });
                    chunk.count += 1;
                    if chunk.last_doc != doc {
                        chunk.last_doc = doc;
                        chunk.sources.push(sources[doc]);
                    }
                }
                tables
            },
        )
        .collect();
    let mut by_shard: Vec<Vec<FxHashMap<&str, Chunk>>> = (0..shards).map(|_| Vec::new()).collect();
    for part in parts {
        for (shard, table) in by_shard.iter_mut().zip(part) {
            shard.push(table);
        }
    }
    by_shard
        .into_par_iter()
        .flat_map_iter(|tables| {
            let mut merged: FxHashMap<&str, (i64, Vec<u32>)> = FxHashMap::default();
            for table in tables {
                for (piece, chunk) in table {
                    let entry = merged.entry(piece).or_default();
                    entry.0 += chunk.count;
                    entry.1.extend(chunk.sources);
                }
            }
            merged.into_iter().map(|(piece, (count, mut sources))| {
                sources.sort_unstable();
                sources.dedup();
                Word { ids: piece.bytes().map(u32::from).collect(), count, sources }
            })
        })
        .collect()
}

/** F of every pair of the words and the words each occurs in. */
fn count_pairs(words: &[Word]) -> FxHashMap<Pair, (i64, Vec<u32>)> {
    words
        .par_iter()
        .enumerate()
        .fold(FxHashMap::default, |mut pairs: FxHashMap<Pair, (i64, Vec<u32>)>, (i, word)| {
            for w in word.ids.windows(2) {
                let entry = pairs.entry((w[0], w[1])).or_default();
                entry.0 += word.count;
                if entry.1.last() != Some(&(i as u32)) {
                    entry.1.push(i as u32);
                }
            }
            pairs
        })
        .reduce(FxHashMap::default, |mut a, b| {
            for (pair, (freq, words)) in b {
                let entry = a.entry(pair).or_default();
                entry.0 += freq;
                entry.1.extend(words);
            }
            a
        })
}

/** The number of distinct sources over the given words (one bitset per parallel task of at least min_len words). */
fn count_sources(words: &[Word], of: &[u32], n_sources: usize, min_len: usize) -> u32 {
    of.par_iter()
        .with_min_len(min_len)
        .fold(|| FixedBitSet::with_capacity(n_sources), |mut bits, &w| {
            bits.extend(words[w as usize].sources.iter().map(|&s| s as usize));
            bits
        })
        .reduce(|| FixedBitSet::with_capacity(n_sources), |mut a, b| {
            a.union_with(&b);
            a
        })
        .count_ones(..) as u32
}

/** SA-BPE's score F·ln(R). */
fn score(freq: i64, sources: u32) -> f64 {
    freq as f64 * (sources as f64).ln()
}

/** The merge loop: pops the heap until its top entry's F and R are exact, then merges that pair. */
fn train_merges(texts: &[PyBackedStr], sources: &[u32], pattern: &PerWorker<Regex>, n_merges: usize) -> Vec<Pair> {
    let n_sources = *sources.iter().max().unwrap() as usize + 1;
    let mut words = count_words(texts, sources, pattern);
    let incidences: usize = words.iter().map(|w| w.sources.len()).sum();
    let min_len = (n_sources.div_ceil(u64::BITS as usize) * words.len()).div_ceil(incidences);
    let mut pair_freqs: FxHashMap<Pair, i64> = FxHashMap::default();
    let mut heap = BinaryHeap::new();
    let bound = n_sources as u32; /* every pair's R is at most the number of sources until counted */
    for (pair, (freq, of)) in count_pairs(&words) {
        pair_freqs.insert(pair, freq);
        heap.push(Job { pair, freq, sources: bound, exact: false, score: score(freq, bound), words: of });
    }
    let mut tokens: Vec<Vec<u8>> = (0..=u8::MAX).map(|b| vec![b]).collect(); /* every token's bytes, its id the index */
    let mut known: FxHashSet<Vec<u8>> = tokens.iter().cloned().collect();
    let mut merges = Vec::new();
    while merges.len() < n_merges {
        let mut job = heap.pop().unwrap();
        let freq = pair_freqs[&job.pair];
        if freq == 0 {
            continue;
        }
        if freq != job.freq {
            job.freq = freq;
            job.exact = false;
            job.score = score(freq, job.sources);
            heap.push(job);
            continue;
        }
        if !job.exact {
            job.words = job.words.into_par_iter().filter(|&w| words[w as usize].contains(job.pair)).collect();
            job.sources = count_sources(&words, &job.words, n_sources, min_len);
            job.exact = true;
            let exact = score(freq, job.sources);
            if exact != job.score {
                job.score = exact;
                heap.push(job);
                continue;
            }
        }
        let bytes = [tokens[job.pair.0 as usize].as_slice(), tokens[job.pair.1 as usize].as_slice()].concat();
        if !known.insert(bytes.clone()) {
            continue;
        }
        let new_id = tokens.len() as u32;
        tokens.push(bytes);
        merges.push(job.pair);
        let mut created: FxHashMap<Pair, Vec<u32>> = FxHashMap::default();
        for &w in &job.words {
            let word = &mut words[w as usize];
            for (pair, delta) in word.merge_pair(job.pair, new_id) {
                *pair_freqs.entry(pair).or_default() += delta * word.count;
                if delta > 0 {
                    let of = created.entry(pair).or_default();
                    if of.last() != Some(&w) {
                        of.push(w);
                    }
                }
            }
        }
        for (pair, of) in created {
            let freq = pair_freqs[&pair];
            if freq > 0 {
                heap.push(Job { pair, freq, sources: job.sources, exact: false, score: score(freq, job.sources), words: of });
            }
        }
    }
    merges
}

/** Trains n_merges merges on texts[d] from source sources[d] (dense ids from 0) within the pre-tokens of pattern.
    Merge i makes token id 256 + i; a merge whose bytes are already a token is skipped. */
#[pyfunction]
#[pyo3(signature = (texts, sources, *, pattern, n_merges))]
pub(crate) fn train(py: Python<'_>, texts: Vec<PyBackedStr>, sources: Vec<u32>, pattern: &str, n_merges: usize) -> Vec<Pair> {
    let pattern = PerWorker::new(|| Regex::new(pattern).unwrap());
    py.detach(|| train_merges(&texts, &sources, &pattern, n_merges))
}
