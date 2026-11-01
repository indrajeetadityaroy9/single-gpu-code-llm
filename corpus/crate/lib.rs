/*! The corpus crate, imported in Python as sgc_corpus: SA-BPE tokenizer training (arXiv 2604.14053; paper §3.1),
    repository-level documents in dependency order (paper §2.1), benchmark decontamination by 10-gram overlap
    (paper §2.1), and the packer of training sequences with fill-in-the-middle examples (paper §3.2, §3.3). */

mod decontaminate;
mod pack;
mod repository;
mod train;

use std::sync::OnceLock;

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use tree_sitter::Language;

/** One value per worker thread of rayon's global pool, created on first use (call only from rayon workers). Needed
    because clones of a fancy-regex Regex share one scratch pool that workers would contend for. */
pub(crate) struct PerWorker<'a, T> {
    make: Box<dyn Fn() -> T + Sync + 'a>,
    values: Vec<OnceLock<T>>,
}

impl<'a, T> PerWorker<'a, T> {
    pub(crate) fn new(make: impl Fn() -> T + Sync + 'a) -> Self {
        PerWorker { make: Box::new(make), values: (0..rayon::current_num_threads()).map(|_| OnceLock::new()).collect() }
    }

    pub(crate) fn get(&self) -> &T {
        self.values[rayon::current_thread_index().unwrap()].get_or_init(&self.make)
    }
}

/** A repository source's language ("Python" or "C++"), which selects the tree-sitter grammar. */
#[derive(Clone, Copy)]
pub(crate) enum Lang {
    Python,
    Cpp,
}

impl Lang {
    pub(crate) fn language(self) -> Language {
        match self {
            Lang::Python => tree_sitter_python::LANGUAGE.into(),
            Lang::Cpp => tree_sitter_cpp::LANGUAGE.into(),
        }
    }
}

impl<'a, 'py> FromPyObject<'a, 'py> for Lang {
    type Error = PyErr;

    fn extract(name: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        match &*name.extract::<PyBackedStr>()? {
            "Python" => Ok(Lang::Python),
            "C++" => Ok(Lang::Cpp),
            other => Err(PyValueError::new_err(format!("grammar {other:?}: \"Python\" or \"C++\""))),
        }
    }
}

/** The module. rayon's global pool is built first, with the RLIMIT_STACK soft limit as each worker's stack size
    (64 MiB under scripts/lib.sh's --ulimit stack=67108864:-1) instead of std's 2 MiB: a worker waiting in a
    join runs stolen jobs on its own stack, so the nesting depth grows with concurrent pack calls. */
#[pymodule]
fn sgc_corpus(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let mut stack = libc::rlimit { rlim_cur: 0, rlim_max: 0 };
    /* SAFETY: getrlimit writes the limit into the struct it is given */
    unsafe { libc::getrlimit(libc::RLIMIT_STACK, &mut stack) };
    rayon::ThreadPoolBuilder::new().stack_size(stack.rlim_cur as usize).build_global()
        .map_err(|e| PyValueError::new_err(format!("rayon's global pool: {e}")))?;
    m.add_function(wrap_pyfunction!(train::train, m)?)?;
    m.add_function(wrap_pyfunction!(repository::topological_sort, m)?)?;
    m.add_class::<decontaminate::Grams>()?;
    m.add_function(wrap_pyfunction!(pack::pack, m)?)
}
