# Reference PyTorch GraphSAGE Implementation
### Author: Huynh Thanh Trung et al.


PyTorch version of [GraphSAGE], original version with Tensorflow can be found at https://github.com/williamleif/GraphSAGE.

#### Requirements

pytorch >0.2 is required (0.4 is recommended).

networkx 1.11

python 3

#### Running examples

##### Supervised

Execute `python -m graphsage.supervised_train --prefix example_data/ppi/graphsage/ppi --multiclass True --epochs 20 --max_degree 25 --model graphsage_maxpool --cuda True ` to run with ppi dataset.

More configurations can be set by adding more arguments to the command, for example:

`--epochs 20` : Number of epochs you want to train.

`--max_degree 25` : Maximum number of neighbors sampled for each node.

`--model graphsage_maxpool` : The model you want to use, available options are graphsage_maxpool, graphsage_mean, graphsage_meanpool, graphsage_lstm

`--cuda True` : Use GPU.

`--multiclass True` : Run with multiclass data

`--prefix example_data/cora/graphsage/cora` : choose cora dataset. Note that cora is not multiclass, so when you use cora
dataset, make sure you changed the `--multiclass` argument to False.

`--embedding_dim` : set to positive value if you want to use NodeEmbedding Preprocessing

`--linear_output_dim`: set to positive value if you want to use Linear Preprocessing

##### Unsupervised

Execute `python -m graphsage.unsupervised_train --prefix example_data/cora/graphsage/cora --epochs 20 --max_degree 25 --model graphsage_mean --cuda True` to run with cora dataset.


#### Files description

inits.py: Matrices initializer with pytorch.

neigh_samplers.py: Sampler of neighbors

aggregators.py: Aggregators of neighbors' embeddings.

dataset.py: Read and extract and preprocess data.

models: SampleAndAggregate model.

prediction.py: Model to calculate loss for unsupervised train

preps.py: Preprocessing methods.

supervised_models.py: Supervised models for supervised learning.

unsupervised_models.py: Unsupervised models for unsupervised learning.

supervised_train.py: Used to train for supervised learning.

unsupervised_train.py: Used to train for unsupervisd learning.

utils.py: For printing graph detail.

#### GraphSAGE format

-G.json: json file of graph,
         "id" is the id of the nodes,
         "feature" is the feature vector of the node,
         "test/val" specifies if node is val/test node,
         "label" is the label of the node

class_map.json: json file, key is the id of the nodes, value is the class of the node

feats.npy: numpy array of the nodes' features, idx of array is the idx of the nodes

id-map.json: json file, key is the id of the nodes, value is the index of the nodes



#### data_utils:
- cora_preprocess.py: Convert cora dataset to graphsage format. (Deprecated)
- count_node_same_features: Count number of edges which have same features.
- edgelist_to_graphsage: Convert from edgelist to graphsage data (include G.json, id_map.json).
- evaluate_distance: Evaluation embedding based on link prediction.
- extend_anchor_link: Extend network.
- feature_groundtruth_checking: Checking dict features whether corrected or not, input is source dataset and target dataset (shuffle source dataset).
- feature_statistics: Print the value taken by sum all features of a network.
- filter_dataset_by_dict:
- filter_dataset_by_degree: Remove nodes which have degree < threshold.
- gen_dict: Generate dictionaries with a split value, including train.dict, test.dict and full.dict.
- generate_groundtruth: Input a txt file containing list of nodes, output a groundtruth file. This file is deprecated. Use full.dict generated by gen_dict, then rename this file to "groundtruth".  (Deprecated).
- get_sub_graph: Generate subgraphs of a network.
- graphsage_to_edgelist: Convert data from graphsage format to edgelist format.
- graphsage_to_mat: Convert data from graphsage format to .mat format.
- mat_to_graphsage: Convert data from .mat format to graphsage format.
- merge_graphs: Merge two graphs into one. Note that new groundtruth file is the groundtruth of source dataset with shuffled source dataset (this file oftens use with the input of source dataset and shuffled source dataset).
- pale_facebook_preprocess: Filter nodes in pale_facebook dataset which have degree < threshold.
- pale_random_clone: Random delete and add edges to a network based on the algorithm in IJCAI16's paper.
- pubmed_preprocess: Convert pubmed dataset to graphsage format. (Deprecated)
- random_clone: Random add and delete add edges to a network.
- random_clone_add: Random add edges to a network.
- random_clone_delete: Random delete edges to a network.
- shuffle_graph: Create new graph by shuffling a graph.
- split_dict: Split groundtruth to train and val set.
- split_embeddings: Split embeddings file to source and target embedding. This file oftens use with merge_graphs after training a merged graph.
- synthetic_graph: Create a synthetic graph.
- visualize_degree_distribution: Visualize nodes' degrees of a network.