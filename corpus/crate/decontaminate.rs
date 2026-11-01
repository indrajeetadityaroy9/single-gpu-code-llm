/*! Benchmark decontamination (paper §2.1): finds the benchmark texts with which a training text shares an n-gram of
    whitespace-separated words (Qwen2.5-Coder's 10-gram word overlap, arXiv 2409.12186 §5; DeepSeek-Coder's 10-gram
    filter, arXiv 2401.14196 §2.4). */

use std::hash::Hasher;

use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use rayon::prelude::*;
use rustc_hash::{FxHashMap, FxHasher};

fn word_hash(word: &str) -> u64 {
    let mut hasher = FxHasher::default();
    hasher.write(word.as_bytes());
    hasher.finish()
}

fn gram_hash(words: &[u64]) -> u64 {
    let mut hasher = FxHasher::default();
    for &word in words {
        hasher.write_u64(word);
    }
    hasher.finish()
}

/** The n-grams of the benchmark texts ("tests"): each n-gram hash maps to the (text, first word) pairs that contain
    it. A hash hit is confirmed word by word, so a hash collision never counts as a match. */
#[pyclass(frozen)]
pub(crate) struct Grams {
    n: usize,
    tests: Vec<Vec<String>>,
    grams: FxHashMap<u64, Vec<(u32, u32)>>,
}

#[pymethods]
impl Grams {
    #[new]
    #[pyo3(signature = (tests, *, n))]
    fn new(tests: Vec<String>, n: usize) -> Self {
        let tests: Vec<Vec<String>> = tests.iter().map(|t| t.split_whitespace().map(str::to_owned).collect()).collect();
        let mut grams: FxHashMap<u64, Vec<(u32, u32)>> = FxHashMap::default();
        for (t, words) in tests.iter().enumerate() {
            let hashes: Vec<u64> = words.iter().map(|w| word_hash(w)).collect();
            for (i, window) in hashes.windows(n).enumerate() {
                grams.entry(gram_hash(window)).or_default().push((t as u32, i as u32));
            }
        }
        Grams { n, tests, grams }
    }

    /** For each text, the indices (ascending) of the benchmark texts that share an n-gram with it. */
    fn matches(&self, py: Python<'_>, texts: Vec<PyBackedStr>) -> Vec<Vec<u32>> {
        py.detach(|| texts.par_iter().map(|text| self.matched(text)).collect())
    }
}

impl Grams {
    fn matched(&self, text: &str) -> Vec<u32> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let hashes: Vec<u64> = words.iter().map(|w| word_hash(w)).collect();
        let mut found = Vec::new();
        for (i, window) in hashes.windows(self.n).enumerate() {
            if let Some(holders) = self.grams.get(&gram_hash(window)) {
                for &(t, at) in holders {
                    let test = &self.tests[t as usize][at as usize..at as usize + self.n];
                    if test.iter().zip(&words[i..i + self.n]).all(|(a, b)| a == b) {
                        found.push(t);
                    }
                }
            }
        }
        found.sort_unstable();
        found.dedup();
        found
    }
}
