import gzip
import itertools
import os
import re

import networkx as nx
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

head_nodes = {
    "biological_process": "GO:0008150", 
    "molecular_function": "GO:0003674",
    "cellular_component": "GO:0005575",
}


def parse_group(group):
    out = {}
    out['type'] = group[0]
    out['is_a'] = []
    out['relationship'] = []

    for line in group[1:]:
        key, val = line.split(': ', 1)

        # Strip out GO names
        if '!' in val:
            val = re.sub('\ !\ .*$', '', val)

        if key == 'relationship':
            val = val.split(' ')

        # Convert to lists of GO names
        if key not in out:
            out[key] = val
        else:
            try:
                out[key] += [val]
            except TypeError:
                out[key] = [out[key], val]

    return out


add_rels = False


class Ontology:
    def __init__(self, obo_file=None, with_relationships=False, restrict_terms_file=None):
        """ Class to parse an .obo.gz file containing a gene ontology description,
        and build a networkx graph and sparse matrix, where the GO DAG is flipped (pointing downwards). 
        Allows for propogating scores and annotations to descendent nodes.
        
        obo_file: a gzipped obo file that corresponds to the ontology used in the
        training data. 
        
        with_relationships (bool): whether to include GO relationships (other than 'is_a')
        as explicit links in the dependency graph
        
        restrict_terms_file: limit the DAG to terms in the given file
        
        """

        self._ancestor_array = None
        if obo_file is None:
            obo_file = os.path.join(dir_path, 'go-basic.obo.gz')

        # this graph is built with the edges pointed downward (i.e., in reverse)
        self.G = self.create_ontology_graph(
                obo_file, with_relationships)

        if restrict_terms_file is None:
            self.terms = sorted(set(self.G.nodes))
        else:
            #term_file = os.path.join(dir_path, 'terms.csv.gz')
            self.terms = pd.read_csv(restrict_terms_file, header=None)[0]
        # make sure we include all of the ancestors of these terms?
        anc_terms = self.get_ancestors(self.terms)
        if len(anc_terms) != len(self.terms):
            print(f"WARNING: {len(anc_terms) - len(self.terms)} ancestral terms were not included in restrict_terms_file")

        self.term_index = {term: i for i, term in enumerate(self.terms)}
        terms = set(self.terms)
        num_terms_with_index = 0
        for node, data in filter(
                lambda x: x[0] in terms, self.G.nodes.items()):
            data['index'] = self.term_index[node]
            num_terms_with_index += 1

        self.total_nodes = num_terms_with_index
        if self.total_nodes != len(terms):
            print(f"WARNING: {self.total_nodes} terms found " +
                  f"among the {len(terms)} terms passed in")

        # this creates a sparse matrix where the rows are ordered according to nodelist
        self.dag_matrix = nx.to_scipy_sparse_matrix(
                self.G, nodelist=self.terms, weight=None)

    def create_ontology_graph(self, obo_file, with_relationships=False):

        G = nx.DiGraph()

        print(f"reading obo file: {obo_file}")
        with gzip.open(obo_file, mode='rt') as f:

            groups = ([l.strip() for l in g] for k, g in
                      itertools.groupby(f, lambda line: line == '\n'))

            for group in groups:
                data = parse_group(group)

                if ('is_obsolete' in data) or (data['type'] != '[Term]'):
                    continue

                G.add_node(data['id'], name=data.get('name'), namespace=data.get('namespace'))

                for target in data['is_a']:
                    G.add_edge(target, data['id'], type='is_a')

                if with_relationships:
                    for type_, target in data['relationship']:
                        G.add_edge(target, data['id'], type=type_)

        # each term in the graph will get an index which corresponds to its place in the annotation matrix
        nx.set_node_attributes(G, None, 'index')

        return G

    def terms_to_indices(self, terms):
        """ Return a sorted list of indices for the given terms, omitting
        those less common than the threshold """
        #return sorted([self.G.nodes[term]['index'] for term in terms if
        #               self.G.nodes[term]['index'] is not None])
        return sorted([self.term_index[term] for term in terms if
            term in self.term_index])

    def get_ancestors(self, terms):
        """ Includes the query terms themselves """
        if type(terms) is str:
            terms = (terms,)

        return set.union(set(terms), *(nx.ancestors(self.G, term) for term in terms))

    def get_descendants(self, terms):
        """ Includes the query term """
        if type(terms) is str:
            terms = (terms,)

        return set.union(set(terms), *(nx.descendants(self.G, term) for term in terms))

    def termlist_to_array(self, terms, dtype=bool):
        """ Propogate labels to ancestor nodes """
        arr = np.zeros(self.total_nodes, dtype=dtype)
        arr[np.asarray(self.terms_to_indices(terms))] = 1
        return arr

    def array_to_termlist(self, array):
        """ Return term ids where array evaluates to True. Uses np.where """
        return [self.term_index[i] for i in np.where(array)[1]]

    def iter_ancestor_array(self):
        """ Constructs the necessary arrays for the tensorflow segment operation.
        Returns a generator of (node_id, ancestor_id) pairs. Use via
        `segments, ids = zip(*self.iter_ancestor_array())` """

        for node, node_index in self.G.nodes(data='index'):
            if node_index is not None:
                for i, ancestor_index in enumerate(self.terms_to_indices(self.get_ancestors(node))):
                    yield ancestor_index, node_index, i

    def ancestor_array(self):
        arr = np.array(list(self.iter_ancestor_array()))
        self._ancestor_array = coo_matrix((arr[:, 0] + 1, (arr[:, 1], arr[:, 2]))).todense()
        return self._ancestor_array

    def get_head_node_indices(self):
        head_nodes = [node for node, degree 
                in self.G.in_degree if degree == 0]
        return self.terms_to_indices(head_nodes)
                                      

# notes, use nx.shortest_path_length(G, root) to find depth? score accuracy by tree depth?

# BP = ont.G.subgraph(ont.get_descendants('GO:0008150'))
# MF = ont.G.subgraph(ont.get_descendants('GO:0003674'))
# CC = ont.G.subgraph(ont.get_descendants('GO:0005575'))


def read_ann_file(ann_file, evidence_codes=None):
    """ Accepts two file formats: tsv, and gaf (gene annotation format) files
    1. tsv: three column file with the following columns: UniProtKB ID, GO ID, GO hierarchy (i.e., BP, MF, CC)
    2. gaf: a regular gaf file. An optional list of evidence codes with which to filter the annotations can be passed in
    """
    print(f"Reading {ann_file}")
    if 'gaf' in ann_file:
        read_gaf_file(ann_file, evidence_codes)
    else:
        read_ann_list_file(ann_file)


def read_gaf_file(gaf_file, evidence_codes=None):
    print(f"TODO: implement reading GAF file. Quitting")
    sys.exit()


def read_ann_list_file(ann_list_file, ont_obj, **kwargs):
    """ Read a list of annotations from a file. TODO pass arguments normally passed to pandas
    Currently expects the header "prot\tterm\thierarchy"

    :returns: pandas df and sparse matrix of annotations
    """
    df = pd.read_csv(ann_list_file, sep='\t', names=['prot', 'term', 'hierarchy'])
    leaf_ann_mat = build_ann_matrix(df, ont_obj) 
    return df, leaf_ann_mat


def build_ann_matrix(df, ont_obj):
    prots = sorted(df['prot'].unique())
    prot_index = {prot: i for i, prot in enumerate(prots)}
    p_idx_list = []
    t_idx_list = []
    for (i, prot, term) in df[['prot', 'term']].itertuples():
        p_idx_list.append(prot_index[prot])
        t_idx_list.append(ont_obj.term_index[term])
    leaf_ann_mat = coo_matrix((np.ones(len(df)), (p_idx_list, t_idx_list))).tocsr()
    return prots, leaf_ann_mat


def propagate_ann_up_dag(ann_mat, dag_mat):
    """ propagate all annotations up the DAG

    :param ann_mat: sparse matrix where rows are proteins and columns are terms 
    :param dag_mat: Assumes that the GO dag matrix has the correct orientation of the 'is_a' edges
    (pointing upwards)
    """
    # full_prop_ann_mat will get all the ancestors of every term
    full_prop_ann_mat = ann_mat.copy()
    prop_ann_mat = ann_mat.copy()
    last_prop_ann_mat = prop_ann_mat
    # keep iterating until there are no more changes, or everything is 0
    # meaning there are no more edges upward
    while True:
        prop_ann_mat = prop_ann_mat.dot(dag_mat)
        diff = (prop_ann_mat != last_prop_ann_mat)
        if diff is True or diff.nnz != 0:
            full_prop_ann_mat += prop_ann_mat
            last_prop_ann_mat = prop_ann_mat
        else:
            break
    # now change values > 1 in the full_prop_ann_mat to 1s 
    full_prop_ann_mat = (full_prop_ann_mat > 0).astype(int)
    return full_prop_ann_mat


# Copied from https://github.com/Murali-group/multi-species-GOA-prediction 
class Sparse_Annotations:
    """ An object to hold the sparse annotations, 
    the list of term IDs giving the index of each term, 
    and a mapping from term to index
    """
    def __init__(self, dag_matrix, ann_matrix, terms, prots):
        self.dag_matrix = dag_matrix
        self.ann_matrix = ann_matrix
        self.terms = terms
        # used to map from index to term and vice versa
        self.term2idx = {g: i for i, g in enumerate(terms)}
        self.prots = prots
        # used to map from node/prot to the index and vice versa
        self.node2idx = {n: i for i, n in enumerate(prots)}

    def reshape_to_prots(self, new_prots):
        """ *new_prots*: list of prots to which the cols should be changed (for example, to align to a network)
        """
        print("\treshaping %d prots to %d prots (%d in common)" % (
            len(self.prots), len(new_prots), len(set(self.prots) & set(new_prots))))
        # reshape the matrix cols to the new prots
        # put the prots on the rows to make the switch
        new_ann_mat = sp.lil_matrix((len(new_prots), self.ann_matrix.shape[0]))
        ann_matrix = self.ann_matrix.T.tocsr()
        for i, p in enumerate(new_prots):
            idx = self.node2idx.get(p)
            if idx is not None:
                new_ann_mat[i] = ann_matrix[idx]
        # now transpose back to term rows and prot cols
        self.ann_matrix = new_ann_mat.tocsc().T.tocsr()
        self.prots = new_prots
        # reset the index mapping
        self.node2idx = {n: i for i, n in enumerate(self.prots)}

    def limit_to_terms(self, terms_list):
        """ *terms_list*: list of terms. Data from rows not in this list of terms will be removed
        """
        terms_idx = [self.term2idx[t] for t in terms_list if t in self.term2idx]
        print("\tlimiting data in annotation matrix from %d terms to %d" % (len(self.terms), len(terms_idx)))
        num_pos = len((self.ann_matrix > 0).astype(int).data)
        terms = np.zeros(len(self.terms))
        terms[terms_idx] = 1
        diag = sp.diags(terms)
        self.ann_matrix = diag.dot(self.ann_matrix)
        print("\t%d pos annotations reduced to %d" % (
            num_pos, len((self.ann_matrix > 0).astype(int).data)))

    def reshape_to_terms(self, terms_list, dag_mat):
        """ 
        *terms_list*: ordered list of terms to which the rows should be changed (e.g., COMP aligned with EXPC)
        *dag_mat*: new dag matrix. Required since the terms list could contain terms which are not in this DAG
        """
        assert len(terms_list) == dag_mat.shape[0], \
            "ERROR: # terms given to reshape != the shape of the given dag matrix"
        if len(terms_list) < len(self.terms):
            # remove the extra data first to speed up indexing
            self.limit_to_terms(terms_list)
        # now move each row to the correct position in the new matrix
        new_ann_mat = sp.lil_matrix((len(terms_list), len(self.prots)))
        #terms_idx = [self.term2idx[t] for t in terms_list if t in self.term2idx]
        for idx, term in enumerate(terms_list):
            idx2 = self.term2idx.get(term)
            if idx2 is None:
                continue
            new_ann_mat[idx] = self.ann_matrix[idx2]
            #new_dag_mat[idx] = self.dag_matrix[idx2][:,terms_idx]
        self.ann_matrix = new_ann_mat.tocsr()
        self.dag_matrix = dag_mat.tocsr()
        self.terms = terms_list
        self.term2idx = {g: i for i, g in enumerate(self.terms)}

    def limit_to_prots(self, prots):
        """ *prots*: array with 1s at selected prots, 0s at other indices
        """
        diag = sp.diags(prots)
        self.ann_matrix = self.ann_matrix.dot(diag)


def create_sparse_ann_file(
        obo_file, ann_file, 
        forced=False, verbose=False, **kwargs):
    """
    Store/load the DAG, annotation matrix, terms and prots. 
    The DAG and annotation matrix will be aligned, and the prots will not be limitted to a network since the network can change.
    The DAG should be the same DAG that was used to generate the pos_neg_file
    *returns*:
        1) dag_matrix: A term by term matrix with the child -> parent relationships
        2) ann_matrix: A matrix with term rows, protein/node columns, and 1 for annotations
        3) terms: row labels
        4) prots: column labels
    """
    sparse_ann_file = ann_file + '.npz'

    if forced or not os.path.isfile(sparse_ann_file):
        # load the pos_neg_file first. Should have only one hierarchy (e.g., BP)
        ann_matrix, terms, prots = setup_sparse_annotations(ann_file)

        # now read the term hierarchy DAG
        # parse the dags first as it also sets up the term_to_category dictionary
        dag_matrix, dag_terms = setup_obo_dag_matrix(obo_file, terms)
        dag_terms2idx = {g: i for i, g in enumerate(dag_terms)}

        print("\twriting sparse annotations to %s" % (sparse_ann_file))
        # store all the data in the same file
        dag_matrix_data = get_csr_components(dag_matrix)
        ann_matrix_data = get_csr_components(ann_matrix)
        np.savez_compressed(
            sparse_ann_file, dag_matrix_data, 
            ann_matrix_data, terms, prots)
    else:
        print("\nReading annotation matrix from %s" % (sparse_ann_file))
        loaded_data = np.load(sparse_ann_file, allow_pickle=True)
        dag_matrix = make_csr_from_components(loaded_data['arr_0'])
        ann_matrix = make_csr_from_components(loaded_data['arr_1'])
        terms, prots = loaded_data['arr_2'], loaded_data['arr_3']
        #dag_matrix = make_csr_from_components(loaded_data['dag_matrix_data'])
        #ann_matrix = make_csr_from_components(loaded_data['ann_matrix_data'])
        #terms, prots = loaded_data['terms'], loaded_data['prots']

    return dag_matrix, ann_matrix, terms, prots


## small utility functions for working with the pieces of
## sparse matrices when saving to or loading from a file
#def get_csr_components(A):
#    all_data = np.asarray([A.data, A.indices, A.indptr, A.shape], dtype=object)
#    return all_data
#
#
#def make_csr_from_components(all_data):
#    return sp.csr_matrix((all_data[0], all_data[1], all_data[2]), shape=all_data[3])
