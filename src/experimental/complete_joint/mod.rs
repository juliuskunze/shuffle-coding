//! Perm group coset codec based on a stabilizer chains.
//! Can be used for joint shuffle coding given an objects' automorphism group.
//! Used for graphs.

use crate::autoregressive::graph::GraphPrefix;
use crate::codec::OrdSymbol;
use crate::experimental::complete_joint::aut::{AutCanonizable, Automorphisms};
use crate::graph::EdgeType;
use crate::permutable::Permutable;
use itertools::Itertools;

pub mod interleaved;
pub mod graph;
pub mod aut;
pub mod sparse_perm;

impl<N: OrdSymbol, Ty: EdgeType> AutCanonizable for GraphPrefix<N, (), Ty> {
    fn automorphisms(&self) -> Automorphisms {
        let partitions = (0..self.graph.len()).
            map(|i| self.node_label_or_index(i)).collect_vec();
        let mut a = self.graph.unlabelled_automorphisms(Some(&partitions));
        a.group.adjust_len(self.len);
        a
    }
}
