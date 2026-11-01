/*! Repository-level documents (paper §2.1; DeepSeek-Coder, arXiv 2401.14196 §2.2): a repository's files grouped
    into samples, the sets of files connected by imports, each ordered so that every file follows the files it
    imports. Python imports (Python Language Reference §5, §7.11) and C++ includes (C++23 [cpp.include]) are
    resolved by path within the repository. */

use std::path::{Component, Path};

use path_clean::PathClean;
use petgraph::algo::kosaraju_scc;
use petgraph::graph::{DiGraph, NodeIndex};
use petgraph::unionfind::UnionFind;
use pyo3::prelude::*;
use pyo3::pybacked::PyBackedStr;
use rayon::prelude::*;
use rustc_hash::FxHashMap;
use tree_sitter::{Node, Parser, Query, QueryCursor, StreamingIterator};

use crate::Lang;

/** tree-sitter-python import nodes: pattern 0 import, pattern 1 from-import (Python Language Reference §7.11). */
const PYTHON: &str = "(import_statement) @import (import_from_statement) @import";

/** tree-sitter-cpp include paths: pattern 0 quoted, pattern 1 angled (C++23 [cpp.include] 2-3). */
const CPP: &str = "(preproc_include path: (string_literal) @path) (preproc_include path: (system_lib_string) @path)";

/** Lookup of the repository files a name can refer to: suffix s matches path p iff p == s or p ends with "/" + s. */
struct Index<'a> {
    suffix: FxHashMap<&'a str, Vec<usize>>,
    exact: FxHashMap<&'a str, Vec<usize>>,
}

impl<'a> Index<'a> {
    fn new(paths: &'a [String]) -> Self {
        let mut suffix: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
        let mut exact: FxHashMap<&str, Vec<usize>> = FxHashMap::default();
        for (i, path) in paths.iter().enumerate() {
            exact.entry(path).or_default().push(i);
            suffix.entry(path).or_default().push(i);
            for (k, _) in path.match_indices('/') {
                suffix.entry(&path[k + 1..]).or_default().push(i);
            }
        }
        Index { suffix, exact }
    }

    fn suffix(&self, s: &str) -> &[usize] {
        self.suffix.get(s).map_or(&[], Vec::as_slice)
    }

    fn exact(&self, p: &str) -> &[usize] {
        self.exact.get(p).map_or(&[], Vec::as_slice)
    }
}

type Imports = fn(Node, usize, &str, &str, &Index, &mut Vec<usize>);

/** The identifiers of a dotted name `a.b.c` (an aliased import's name). */
fn components<'t>(node: Node, text: &'t str) -> Vec<&'t str> {
    let name = if node.kind() == "aliased_import" { node.child_by_field_name("name").unwrap() } else { node };
    let mut cursor = name.walk();
    name.named_children(&mut cursor).filter(|c| c.kind() == "identifier").map(|c| &text[c.byte_range()]).collect()
}

/** A Python import's files (Python Language Reference §5.2, §5.7, §7.11): absolute modules by suffix, relative imports
    of level l at exact paths under the importing file's l-th ancestor. */
fn python(node: Node, pattern: usize, path: &str, text: &str, index: &Index, targets: &mut Vec<usize>) {
    let mut by_suffix = |module: &str| {
        targets.extend(index.suffix(&format!("{module}.py")));
        targets.extend(index.suffix(&format!("{module}/__init__.py")));
    };
    let mut cursor = node.walk();
    let names: Vec<String> = node.children_by_field_name("name", &mut cursor).map(|name| components(name, text).join("/")).collect();
    if pattern == 0 {
        names.iter().for_each(|module| by_suffix(module));
        return;
    }
    let module = node.child_by_field_name("module_name").unwrap();
    if module.kind() == "dotted_name" {
        let module = components(module, text).join("/");
        by_suffix(&module);
        names.iter().for_each(|name| by_suffix(&format!("{module}/{name}")));
        return;
    }
    let mut cursor = module.walk();
    let (mut level, mut submodule) = (0, None);
    for child in module.named_children(&mut cursor) {
        match child.kind() {
            "import_prefix" => level = text[child.byte_range()].bytes().filter(|&b| b == b'.').count(),
            _ => submodule = Some(components(child, text)),
        }
    }
    let Some(base) = Path::new(path).ancestors().nth(level) else { return };
    let target = submodule.iter().flatten().fold(base.to_path_buf(), |target, component| target.join(component));
    let exact = |file: &Path| index.exact(file.to_str().unwrap());
    if submodule.is_some() {
        targets.extend(exact(Path::new(&format!("{}.py", target.to_str().unwrap()))));
    }
    targets.extend(exact(&target.join("__init__.py")));
    for name in names {
        targets.extend(exact(&target.join(format!("{name}.py"))));
        targets.extend(exact(&target.join(name).join("__init__.py")));
    }
}

/** A path lexically cleaned (path-clean: Go's path.Clean); None when it leaves the repository root. */
fn cleaned(path: &Path) -> Option<String> {
    let path = path.clean();
    (path.components().next() != Some(Component::ParentDir)).then(|| path.to_str().unwrap().to_string())
}

/** A C++ include's files (C++23 [cpp.include] 2-3; GCC manual "The C Preprocessor" §2.3): quoted first at the exact
    path from the including file's directory, else (and angled) by suffix. */
fn cpp(node: Node, pattern: usize, path: &str, text: &str, index: &Index, targets: &mut Vec<usize>) {
    let quoted = pattern == 0;
    /* a string literal's quotes are its first and last children */
    let name = if quoted {
        &text[node.child(0).unwrap().end_byte()..node.child(node.child_count() - 1).unwrap().start_byte()]
    } else {
        &text[node.start_byte() + 1..node.end_byte() - 1]
    };
    if quoted {
        let found = cleaned(&Path::new(path).parent().unwrap().join(name)).map_or(&[][..], |p| index.exact(&p));
        if !found.is_empty() {
            targets.extend(found);
            return;
        }
    }
    if let Some(name) = cleaned(Path::new(name)) {
        targets.extend(index.suffix(&name));
    }
}

/** The samples of one repository, given `dependencies[f]` (ascending, without f itself): the weakly connected
    components, in order of their smallest file index, each listing every file after the files it depends on
    (kosaraju_scc yields strongly connected components in reverse topological order). */
fn samples(dependencies: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = dependencies.len();
    let mut graph = DiGraph::<(), ()>::with_capacity(n, 0);
    for _ in 0..n {
        graph.add_node(());
    }
    let mut weak = UnionFind::new(n);
    for (f, targets) in dependencies.iter().enumerate() {
        for &g in targets {
            graph.add_edge(NodeIndex::new(f), NodeIndex::new(g), ());
            weak.union(f, g);
        }
    }
    let mut sample_of: Vec<Option<usize>> = vec![None; n];
    let mut samples: Vec<Vec<usize>> = Vec::new();
    for i in 0..n {
        let root = weak.find(i);
        if sample_of[root].is_none() {
            sample_of[root] = Some(samples.len());
            samples.push(Vec::new());
        }
    }
    for scc in kosaraju_scc(&graph) {
        let mut files: Vec<usize> = scc.iter().map(|v| v.index()).collect();
        files.sort_unstable();
        samples[sample_of[weak.find(files[0])].unwrap()].extend(files);
    }
    samples
}

/** The samples of a partition's repositories (arXiv 2401.14196 §2.2, Algorithm 1): repository r holds files
    ends[r - 1]..ends[r] (from 0). Each sample is a list of indices into paths; the result does not depend on the
    rayon schedule. */
#[pyfunction]
#[pyo3(signature = (paths, texts, ends, *, grammar))]
pub(crate) fn topological_sort(py: Python<'_>, paths: Vec<String>, texts: Vec<PyBackedStr>, ends: Vec<usize>, grammar: Lang) -> Vec<Vec<usize>> {
    py.detach(|| {
        let repositories: Vec<(usize, usize)> = std::iter::once(0).chain(ends.iter().copied()).zip(ends.iter().copied()).collect();
        let repository: Vec<usize> = repositories.iter().enumerate().flat_map(|(r, &(a, b))| std::iter::repeat_n(r, b - a)).collect();
        let (source, imports): (&str, Imports) = match grammar {
            Lang::Python => (PYTHON, python),
            Lang::Cpp => (CPP, cpp),
        };
        let language = grammar.language();
        let query = Query::new(&language, source).unwrap();
        let indexes: Vec<Index> = repositories.par_iter().map(|&(a, b)| Index::new(&paths[a..b])).collect();
        let parser = || {
            let mut parser = Parser::new();
            parser.set_language(&language).unwrap();
            (parser, QueryCursor::new())
        };
        /* each file's dependencies, as indices within its repository */
        let dependencies: Vec<Vec<usize>> = (0..paths.len()).into_par_iter().map_init(parser, |(parser, cursor), file| {
            let (r, mut targets) = (repository[file], Vec::new());
            let (a, b) = repositories[r];
            if b - a > 1 {
                let (path, text) = (&paths[file], &*texts[file]);
                let tree = parser.parse(text.as_bytes(), None).unwrap();
                let mut matches = cursor.matches(&query, tree.root_node(), text.as_bytes());
                while let Some(m) = matches.next() {
                    imports(m.captures()[0].node, m.pattern_index, path, text, &indexes[r], &mut targets);
                }
                targets.sort_unstable();
                targets.dedup();
                targets.retain(|&g| g != file - a);
            }
            targets
        }).collect();
        repositories.par_iter().flat_map_iter(|&(a, b)| samples(&dependencies[a..b]).into_iter().map(move |sample| sample.into_iter().map(|f| a + f).collect())).collect()
    })
}
