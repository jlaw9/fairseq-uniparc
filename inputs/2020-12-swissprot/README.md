
## 2021-05-27
I tried re-fine-tuning Peter's model trained with the old quickgo data, but was getting 0s for MF and CC f1 scores. 
I think the problem is that the networkx G representation of the GO changed the ordering of the nodes from when Peter first ran it,
causing a mis-matching of the indexes when compared with the ordering in the sparse matrices in the train, valid and test.npz files.
See: https://github.com/pstjohn/fairseq-uniparc/blob/main/go_annotation/ontology/ontology.py#L74

In this version of the code, I use the ordering listed in the terms file (e.g., terms-sorted.csv.gz) 
both to generate the sparse matrices and to load the indexes.
