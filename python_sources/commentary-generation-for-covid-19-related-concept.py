#!/usr/bin/env python
# coding: utf-8

# # Introduction
# 
# ## Purpose
# 
# To help medical researchers keep up with the rapid acceleration in new literature on COVID-19, we design and build this knowledge graph based commentary generation tool. When users search for something in these papers, it can generate knowledge about what these papers explain.
# 
# Unlike some traditional search engines, the tool searches for 'concept' rather than 'string'. This brings two benefits. Firstly, we can recognize the 'concept' when there are many different ways of saying the same 'concept' in different papers. Secondly, by means of knowledge graph we can find related ontologies and relations to the 'concept' rather than the location of the 'string'.
# 
# ## Approach
# 
# Steps we take to build this tool:
# 
# 1. Over the provided scholarly articles, build a knowledge graph which consists of entities like disease, symptom, gene, drug, chemical and also paper
# 2. Use graph embedding model to predict missing links in the graph.
# 3. Build a system that can visualize concepts related to target concept and generate commentary about it.

# In[ ]:


# Ignore warnings in this notebook.
import warnings
warnings.filterwarnings('ignore')


# # Build knowledge graph
# 
# ## Overall approach
# 
# We combine traditional NLP based entity and relation extraction approach to extract entities and relations from large corpus of biomedical papers with modern deep learning based approach like R-GCN to predict new relations. Our approach has the following steps and more details will be given later.
# 
# 1. Extract entities and relations that are stated in papers using [Semantic Knowledge Representation](https://semrep.nlm.nih.gov/) which is an entity and relation extraction tool based on [Unified Medical Language System](https://www.nlm.nih.gov/research/umls/index.html) project. 
# 2. Enrich relations using existing knowledges in [UMLS Metathasaurus](https://uts.nlm.nih.gov/metathesaurus.html) library.
# 3. Generate new knowledge/relations using [Relational Graphical Convoluation Network](https://arxiv.org/pdf/1703.06103.pdf).
# 
# ## Elements
# 
# **Unified Medical Language System**
# 
# The UMLS, or Unified Medical Language System, is a set of files and software sponsored by [National Library of Medicine](https://www.nlm.nih.gov/) that brings together many health and biomedical vocabularies and standards to enable interoperability between computer systems.
# 
# Getting access to UMLS and other softwares depending on UMLS(like Semantic Knowledge Representation in our work) requires a license. One can apply for the license from the [UMLS Terminology Services](https://uts.nlm.nih.gov//license.html).
# 
# **UMLS Metathasaurus**
# 
# We use UMLS Metathasaurus in this work. UMLS Metathasaurus is a major knowledge source in UMLS that consists of biomedical entities, types and relations between them. Every entity in UMLS Metathasaurus has a unique CUI, several aliases of the entity, types like Gene, Disease and some existing known relations between these entities. Let us give two entities and one relation in UMLS Metathasaurus as an example
# 
# *Entity*
# 
# | CUI | aliases | Semantic Type | 
# | --- | :--: | :--: |
# |C0034417 |quinine, Chinin etc. | Organic Chemical, Pharmacologic Substance etc. | 
# |C0024535 | Malaria, Falciparum, Subtertian malaria etc | Disease or Syndrome|
# 
# *Relation*
# 
# | CUI2 | relation | CUI1 | 
# | --- | :--: | :--: |
# |C0024535 |may_be_treated_by| C0034417| 
# 
# which stands for "Malaria may be treated by Quinine"
# 
# In our work, we explicitly use existing relations in UMLS Metathasaurus to enrich the relations we extract from biomedical papers using SemRep which is introduced later in this section.
# 
# **UMLS Semantic Network**
# 
# The Semantic Network consists of (1) a set of broad subject categories, or Semantic Types, of concepts represented in the UMLS Metathesaurus, and (2) a set of useful and important relationships, or Semantic Relations, that exist between Semantic Types. Let's also give examples of such Semantic types and Semantic relations in the Network
# 
# *Semantic Relations*
# 
# | Semantic Type | relation | Semantic Type | 
# | --- | :--: | :--: |
# |Pharmacologic Substance | treats | Injury or Poisoning	| 
# |Pharmacologic Substance | treats | Disease or Syndrome	|
# 
# We did not explicitly use the Semantic Network in our work but it's an input of the relation extraction tool SemRep below. 
# 
# 
# **Semantic Knowledge Representation (SemRep)**
# 
# SemRep is a UMLS-based program that extracts three-part propositions, called semantic predications, from sentences in biomedical text along with UMLS entities that are present in the text. The extraction is done by
# 
# 1. Recognizing UMLS Metathasaurus concepts and their semantic types in texts using another UMLS-based program [Metamap](https://metamap.nlm.nih.gov/).
# 2. Match relation stated as predicate in the text. For example, predicate such as 'treats' will be recognized.
# 3. UMLS Semantic Network is employed to verify the validity of the relation. For example, entity of type 'Pharmacologic Substance' can 'treat' entity of type 'Disease or Syndrome'
# 
# Let's also give one example of such extraction
# 
#     Input: Quinine can treat Lung cancer
#     Partial Output: 
#         entities: 
#             C0034417 Quinine
#             C0242379 Malignant neoplasm of lung 
#         relations:
#             C0034417|Quinine|TREATS|C0242379|Malignant neoplasm of lung 
# 

# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


# Utils for graph building
def get_entity_type_dict(graph):
    ''' Get entity type'''
    entity_type_dict = {}
    entity_list = set([triplet[0] for triplet in graph]) | set([triplet[2] for triplet in graph])
    for entity in entity_list:
        entity_type = entity.split('_')[0]
        entity_type_dict[entity] = entity_type   
    return entity_type_dict

def get_graph_stats(graph):
    ''' Show stats of the graph'''
    nodes = set([edge[0] for edge in graph]) | set([edge[2] for edge in graph])
    nodes_type = set([edge[0].split('_')[0] for edge in graph]) | set([edge[2].split('_')[0] for edge in graph])
    predicates = set([edge[1] for edge in graph])    
    print('Graph has {} edges, {} nodes, {} types of nodes, {} types of relations.'.          format(len(graph), len(nodes), len(nodes_type), len(predicates)))    

def deduplicate_graph(graph):
    '''Deduplicate triplets in the graph'''
    return [t.split('=>') for t in set(['=>'.join(t) for t in graph])]


# ## Extracted files
# 
# There are three data files of extracted triplets:
# 
# 1. Entity -> Entity relationships extracted from papers using [Semantic Knowledge Representation](https://semrep.nlm.nih.gov/) is in the file 'semrep_rela.csv'.
# 2. Entity -> Entity relationships as exsiting knowledge in [UMLS Metathasaurus](https://uts.nlm.nih.gov/metathesaurus.html) library is in the file 'umls_rela.csv'.
# 3. Paper -> Entity relationships extracted from papers using [Metamap](https://metamap.nlm.nih.gov/) is in the file 'semrep_entity.csv'.

# In[ ]:


# Build graph from extracted triplets files
def add_triplets_semrep_rela(graph):
    '''Entity-Entity relationship extracted from papers using semrep
    Return: update graph with Entity -> Entity relationship
    '''
    print('\nAdding realtions from semrep_rela...')    
    df_semrep = pd.read_csv('/kaggle/input/covid-challenge/semrep_rela.csv', header=0, names=['s', 'r', 'o'])
    for index, row in df_semrep.iterrows():        
        graph.append([row['s'], '1_' + row['r'], row['o']])
    return graph

def add_triplets_umls_rela(graph):
    '''Add entity-entity relationship from ULMS'''
    print('\nAdding realtions from umls_rela...')
    entity_type_dict = get_entity_type_dict(graph)
    df_umls = pd.read_csv('/kaggle/input/covid-challenge/umls_rela.csv', header=0, names=['s', 'r', 'o'])    
    for index, row in df_umls.iterrows():
        if row['s'] in entity_type_dict or row['o'] in entity_type_dict:
            graph.append([row['s'], '2_' + row['r'], row['o']])           
    return graph

def add_triplets_semrep_entity(graph):
    '''This is to add a new paper-entity relationship into the existing graph
    Return: graph for Paper -> Entity relationship
    '''
    print('\nAdding realtions from semrep_entity...')
    entity_type_dict = get_entity_type_dict(graph)
    df_semrep_entity = pd.read_csv('/kaggle/input/covid-challenge/semrep_entity.csv', header=0, names=['s', 'r', 'o'])
    for index, row in df_semrep_entity.iterrows():
        if row['o'] in entity_type_dict:
            graph.append(['Paper_' + row['s'], '0_' + row['r'] + row['o'].split('_')[0], row['o']])
    return graph


# In[ ]:


graph = []

graph = add_triplets_semrep_rela(graph)
graph = deduplicate_graph(graph)
get_graph_stats([[triplet[0], triplet[1].split('_', 1)[1], triplet[2]] for triplet in graph])

graph = add_triplets_umls_rela(graph)
graph = deduplicate_graph(graph)
get_graph_stats([[triplet[0], triplet[1].split('_', 1)[1], triplet[2]] for triplet in graph])

graph = add_triplets_semrep_entity(graph)
graph = deduplicate_graph(graph)
get_graph_stats([[triplet[0], triplet[1].split('_', 1)[1], triplet[2]] for triplet in graph])


# ## Graph structure
# 
# * Knowledge graph consists of triplets like **<subject, predicate, object>**;
# * **subject** and **object** are nodes and **predicate** is relation;
# * There are 6 types of nodes in the graph: Disease, Symptom, PharmaSub, Gene, Chemical and Paper. Each node starts with its type and ends with a global unique id, combined by '_';
# * There are 55 types of relations with direction:
#  * Entity -> Entity relations extracted from papers start with '1_'.
#  * Entity -> Entity relations from UMLS start with '2_'.
#  * Paper -> Entity relations start with '0_'.
#  
# Example as below:

# In[ ]:


df_graph = pd.DataFrame(graph, columns=['subj', 'pred', 'obj'])
df_graph.head(10)


# # Use R-GCN (Relational Graph Convolutional Network) to predict links
# 
# ## Purpose
# 
# Link Prediction (LP), the task of predicting missing facts among entities already a knowledge graph (recovery
# of missing facts, i.e. subject-predicate-object triples), is a promising and widely studied task aimed at addressing knowledge graph incompleteness. We also want to complete our knowledge graph with LP.  
# 
# Among the recent LP techniques, those based on knowledge graph embeddings (aims to embed the entities and relationships of a knowledge graph in low-dimensional vector spaces) have achieved very promising performances in some benchmarks.
# 
# 
# ## Elements
# 
# **R-GCN**
# 
# R-GCN ([Relational Graphical Convoluation Network](https://arxiv.org/pdf/1703.06103.pdf)) is related to a recent class of neural networks operating on graphs, and is developed specifically to deal with the highly multi-relational data characteristic of realistic knowledge bases.
# 
# It assumes that knowledge bases store collections of triples of the form **<subject, predicate, object>**. Consider, for example, the triple <Mikhail Baryshnikov, educated at, Vaganova Academy>, where we will refer to Baryshnikov and Vaganova Academy as entities and to educated at as a relation. Additionally, it assumes that entities are labeled with types. It is convenient to represent knowledge bases as directed labeled multigraphs with entities corresponding to nodes and triples encoded by labeled edges.
# 
# The link prediction model of R-GCN can be regarded as an autoencoder consisting of (1) an encoder: an R-GCN producing latent feature representations of entities, and (2) a decoder: a tensor factorization model exploiting these representations to predict labeled edges. Though in principle the decoder can rely on any type of factorization (or generally any scoring function), we use one of the simplest and most effective factorization methods: DistMult.
# 
# **DGL**
# 
# Deep Graph Library ([DGL](https://docs.dgl.ai/index.html)) is a Python package built for easy implementation of graph neural network model family, on top of existing DL frameworks (e.g. PyTorch, MXNet, Gluon etc.). Need to install it before using.

# In[ ]:


# Install dgl to implement R-GCN
get_ipython().system('pip install dgl')


# In[ ]:


import time

import dgl
from dgl.nn.pytorch import RelGraphConv
import torch
import torch.nn as nn
import torch.nn.functional as F

# Consists of utils for R-GCN link prediction
import utils


# In[ ]:


# Link prediction model of R-GCN
class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h
    
    
class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())
    

class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                self.num_bases, activation=act, self_loop=True,
                dropout=self.dropout)

    
class LinkPredict(nn.Module):
    def __init__(self, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.rgcn = RGCN(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss


# In[ ]:


# Main body
class KnowledgeGraph:
    '''Knowledge graph class.
    Enable us store the graph, train model, load model, predict links, search the graph and get sub-graph.'''
    def __init__(self, df_graph):
        self.df_graph_full = df_graph.copy()
        self.df_graph = df_graph[df_graph.apply(lambda row: row[1].split('_')[0], axis=1) != '0'].copy()
        self.df_graph.reset_index(inplace=True, drop=True)
        self.generate_dictionary()
        self.total_data = self.generate_dataset()
        self.num_nodes, self.num_rels, self.num_edges = self.get_stats()
        
    def generate_dictionary(self):
        print('Generating dictionaries for all entities and relations in the graph...')
        
        # entity-index and relation-index
        self.entity_dict = {}
        self.relation_dict = {}
        # index-entity and index-relation
        self.inverse_entity_dict = {}
        self.inverse_relation_dict = {}
        # count entities and relations
        entity_index = 0
        relation_index = 0
        
        for index, row in self.df_graph.iterrows():
            if row[0] not in self.entity_dict:
                self.entity_dict[row[0]] = entity_index
                self.inverse_entity_dict[entity_index] = row[0]
                entity_index += 1
                
            if row[2] not in self.entity_dict:
                self.entity_dict[row[2]] = entity_index
                self.inverse_entity_dict[entity_index] = row[2]
                entity_index += 1
                
            subj_type = row[0].split('_')[0]
            obj_type = row[2].split('_')[0]
            relation = row[1].split('_', 1)[1]
            if (subj_type, relation, obj_type) not in self.relation_dict:
                self.relation_dict[(subj_type, relation, obj_type)] = relation_index
                self.inverse_relation_dict[relation_index] = (subj_type, relation, obj_type)
                relation_index += 1
        
        print('Done.\n')
            
    def get_stats(self):
        num_nodes = len(self.entity_dict)
        num_rels = len(self.relation_dict)
        num_edges = len(self.df_graph)
        print('# entities:', num_nodes)
        print('# relations:', num_rels)
        print('# edges:', num_edges)
        
        return num_nodes, num_rels, num_edges
        
    def generate_dataset(self):
        print('Generating dataset for the model...')
        # Transfer name to index in the dataset
        triplet_list = []
        for index, row in self.df_graph.iterrows():
            triplet = []

            subj_type = row[0].split('_')[0]
            obj_type = row[2].split('_')[0]
            
            relation = row[1].split('_', 1)[1]

            triplet.append(self.entity_dict[row[0]])
            triplet.append(self.relation_dict[(subj_type, relation, obj_type)])
            triplet.append(self.entity_dict[row[2]])

            triplet_list.append(triplet)

        total_data = np.asarray(triplet_list)
        print('Done.\n')
        
        return total_data
    
    def get_train_test_data(self, n_train, n_valid, n_test):
        '''
        '''
        # Split dataset into train, valid and test
        n_train = n_train
        n_valid = n_valid
        n_test = n_test

        np.random.seed(777)
        shuffle = np.random.permutation(self.num_edges)
        train_data = self.total_data[shuffle[0:n_train]]
        valid_data = self.total_data[shuffle[n_train:n_train + n_valid]]
        test_data = self.total_data[shuffle[n_train + n_valid:n_train + n_valid + n_test]]
        
        return train_data, valid_data, test_data
    
    def create_model(self, n_hidden, num_bases, num_hidden_layers, dropout, use_cuda, reg_param):
        """"""
        model = LinkPredict(self.num_nodes,
                            n_hidden,
                            self.num_rels,
                            num_bases=num_bases,
                            num_hidden_layers=num_hidden_layers,
                            dropout=dropout,
                            use_cuda=use_cuda,
                            reg_param=reg_param)
        return model
    
    def train_model(self, model_state_file,
                    n_train=500000, n_valid=5000, n_test=5000,
                    n_hidden=500, num_bases=100, num_hidden_layers=2, dropout=0.2, use_cuda=False, reg_param=0.01,
                    lr=0.01, graph_batch_size=30000, graph_split_size=0.5,
                    negative_sample=10, edge_sampler='uniform', grad_norm=1.0,
                    evaluate_every=200, eval_batch_size=500, eval_protocol='filtered', n_epochs=800):
        """"""
        # split dataset
        train_data, valid_data, test_data = self.get_train_test_data(n_train, n_valid, n_test)
        # validation and testing triplets
        valid_data = torch.LongTensor(valid_data)
        test_data = torch.LongTensor(test_data)

        model = self.create_model(n_hidden, num_bases, num_hidden_layers, dropout, use_cuda, reg_param)
        
        # build test graph
        new_g = dgl.DGLGraph()
        test_graph, test_rel, test_norm = utils.build_test_graph(new_g, self.num_nodes, self.num_rels, train_data)
        test_deg = test_graph.in_degrees(range(test_graph.number_of_nodes())).float().view(-1,1)
        test_node_id = torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1)
        test_rel = torch.from_numpy(test_rel)
        test_norm = utils.node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))

        # build adj list and calculate degrees for sampling
        adj_list, degrees = utils.get_adj_and_degrees(self.num_nodes, train_data)

        # optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        forward_time = []
        backward_time = []
        
        # training loop
        print("start training...")

        epoch = 0
        best_mrr = 0
        while True:
            model.train()
            epoch += 1

            # perform edge neighborhood sampling to generate training graph and data
            new_g = dgl.DGLGraph()
            g, node_id, edge_type, node_norm, data, labels =                 utils.generate_sampled_graph_and_labels(
                    new_g, train_data, graph_batch_size, graph_split_size,
                    self.num_rels, adj_list, degrees, negative_sample,
                    edge_sampler)
            print("Done edge sampling")

            # set node/edge feature
            node_id = torch.from_numpy(node_id).view(-1, 1).long()
            edge_type = torch.from_numpy(edge_type)
            edge_norm = utils.node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
            data, labels = torch.from_numpy(data), torch.from_numpy(labels)
            deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

            t0 = time.time()
            embed = model(g, node_id, edge_type, edge_norm)
            loss = model.get_loss(g, embed, data, labels)
            t1 = time.time()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm) # clip gradients
            optimizer.step()
            t2 = time.time()

            forward_time.append(t1 - t0)
            backward_time.append(t2 - t1)
            print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
                  format(epoch, loss.item(), best_mrr, forward_time[-1], backward_time[-1]))

            optimizer.zero_grad()

            # validation
            if epoch % evaluate_every == 0:
                model.eval()
                print("start eval")
                embed = model(test_graph, test_node_id, test_rel, test_norm)
                mrr = utils.calc_mrr(embed, model.w_relation, torch.LongTensor(train_data),
                                     valid_data, test_data, hits=[1, 3, 10], eval_bz=eval_batch_size,
                                     eval_p=eval_protocol)
                # save best model
                if mrr >= best_mrr:
                    best_mrr = mrr
                    torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                               model_state_file)

                if epoch >= n_epochs:
                        break

        print("Training done!")
        print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
        print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))
    
    def load_model(self, model_state_file,
                   n_hidden=500, num_bases=100, num_hidden_layers=2, dropout=0.2, use_cuda=False, reg_param=0.01):
        '''
        '''
        model = self.create_model(n_hidden, num_bases, num_hidden_layers, dropout, use_cuda, reg_param)
        
        # use best model checkpoint
        checkpoint = torch.load(model_state_file)
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        print("Using best epoch: {}".format(checkpoint['epoch']))

        # build total graph
        new_g = dgl.DGLGraph()
        total_graph, total_rel, total_norm = utils.build_test_graph(new_g, self.num_nodes, self.num_rels, self.total_data)
        total_deg = total_graph.in_degrees(range(total_graph.number_of_nodes())).float().view(-1, 1)
        total_node_id = torch.arange(0, self.num_nodes, dtype=torch.long).view(-1, 1)
        total_rel = torch.from_numpy(total_rel)
        total_norm = utils.node_norm_to_edge_norm(total_graph, torch.from_numpy(total_norm).view(-1, 1))

        # get embed weights
        self.total_embed = model(total_graph, total_node_id, total_rel, total_norm)
        self.w = model.w_relation

    def save_model_with_np(self):
        ''''''
        total_embed = graph.total_embed.detach().numpy()
        np.save('total_embed.npy', total_embed)
        w = graph.w.detach().numpy()
        np.save('w.npy', w)
        
    def load_model_from_np(self):
        ''''''
        total_embed = np.load('/kaggle/input/rgcn-model/total_embed.npy')
        self.total_embed = torch.from_numpy(total_embed)
        w = np.load('/kaggle/input/rgcn-model/w.npy')
        self.w = torch.from_numpy(w)
        
    def predict_score_with_s_r_o(self, s, r, o):
        '''
        '''
        s_index = self.entity_dict[s]
        r_index = self.relation_dict[r]
        o_index = self.entity_dict[o]
        
        emb_s = self.total_embed[s_index]
        emb_r = self.w[r_index]
        emb_o = self.total_embed[o_index]

        emb_triplet = emb_s * emb_r * emb_o
        score = torch.sigmoid(torch.sum(emb_triplet))
        
        return score
    
    def get_filtered_index_list(self, entity_type):
        '''entity_type: 'Paper', 'Disease', 'Gene', 'PharmaSub', 'Symptom', 'Chemical'
        '''
        
        if entity_type not in ['Paper', 'Disease', 'Gene', 'PharmaSub', 'Symptom', 'Chemical']:
            raise Exception('entity_type should not be {:s}'.format(entity_type))
        
        filtered_index_list = []
        for entity, index in self.entity_dict.items():
            if entity.split('_')[0] == entity_type:
                filtered_index_list.append(index)
        filtered_index_list = torch.LongTensor(filtered_index_list)
        
        return filtered_index_list
    
    def get_most_possible_subject_with_relation_and_object(self, r, o):
        
        subject_entity_type = r[0]
        filtered_s = self.get_filtered_index_list(subject_entity_type)
        
        r = (r[0], r[1].split('_', 1)[1], r[2])
        r_index = self.relation_dict[r]
        o_index = self.entity_dict[o]
        
        # Load weights of the model
        emb_s = self.total_embed[filtered_s]
        emb_r = self.w[r_index]
        emb_o = self.total_embed[o_index]
        
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        
        subject_list = []
        for i in indices:
            # Get the entity index
            entity_index = int(filtered_s[int(i)])
            subject_list.append(self.inverse_entity_dict[entity_index])
            
        return subject_list
    
    def get_most_possible_object_with_subject_and_relation(self, s, r):
        
        s_index = self.entity_dict[s]
        r = (r[0], r[1].split('_', 1)[1], r[2])
        r_index = self.relation_dict[r]
        
        object_entity_type = r[2]
        filtered_o = self.get_filtered_index_list(object_entity_type)
        
        # Load weights of the model
        emb_s = self.total_embed[s_index]
        emb_r = self.w[r_index]
        emb_o = self.total_embed[filtered_o]
        
        emb_triplet = emb_s * emb_r * emb_o
        scores = torch.sigmoid(torch.sum(emb_triplet, dim=1))
        _, indices = torch.sort(scores, descending=True)
        
        object_list = []
        for i in indices:
            # Get the entity index
            entity_index = int(filtered_o[int(i)])
            object_list.append(self.inverse_entity_dict[entity_index])
            
        return object_list
    
    def get_object_with_subject(self, s):
        ''''''
        object_list = list(self.df_graph_full[self.df_graph_full.loc[:, 'subj'] == s]['obj'])
        return object_list
    
    def get_subject_with_object(self, o):
        ''''''
        subject_list = list(self.df_graph_full[self.df_graph_full.loc[:, 'obj'] == o]['subj'])
        return subject_list
    
    def get_object_with_subject_and_relation(self, s, r):
        ''''''        
        object_list = list(self.df_graph_full[(self.df_graph_full.loc[:, 'subj'] == s) &                                              (self.df_graph_full.loc[:, 'pred'] == r[1])]['obj'])
        object_list = [x for x in object_list if x.split('_')[0] == r[2]]            
        return object_list
        
    def get_subject_with_relation_and_object(self, r, o):
        ''''''
        subject_list = list(self.df_graph_full[(self.df_graph_full.loc[:, 'obj'] == o) &                                               (self.df_graph_full.loc[:, 'pred'] == r[1])]['subj'])
        subject_list = [x for x in subject_list if x.split('_')[0] == r[0]]            
        return subject_list
    
    def get_subgraph(self, df_subgraph=None, seed='Disease_C5203670', degree=1):            
        df_subgraph = pd.concat([df_subgraph, self.df_graph[(self.df_graph.loc[:, 'subj'] == seed) |                                                            (self.df_graph.loc[:, 'obj'] == seed)]])
        if degree > 1:
            seeds = list(set(df_subgraph['subj']) | set(df_subgraph['obj']))
            for seed in seeds[:10]:
                df_subgraph = self.get_subgraph(df_subgraph, seed, degree=degree-1)
        df_subgraph.reset_index(inplace=True, drop=True)
        return df_subgraph
    
    def get_subgraph_full(self, seed='Disease_C5203670', degree=1):            
        df_subgraph = self.get_subgraph(seed=seed, degree=degree)
        seeds = list(set(df_subgraph['subj']) | set(df_subgraph['obj']))
        df_subgraph_full = df_graph[(df_graph['subj'].isin(seeds)) & (df_graph['obj'].isin(seeds))]
        df_subgraph_full.reset_index(inplace=True, drop=True)
        return df_subgraph_full


# In[ ]:


# Load the graph
graph = KnowledgeGraph(df_graph)


# ## Train and load R-GCN model
# 
# Only run training code once:

# In[ ]:


# # Train
# graph.train_model('model_state.pth', n_train=130000)

# # Load full model on full graph
# graph.load_model('/kaggle/working/model_state.pth')

# # Save the vectorized information of the knowledge graph
# graph.save_model_with_np()


# After training and saving, we get two vetorized results in output: 'total_embed.npy' and 'w.npy'. They are embedded vectors representing nodes and relations. Need to move them to '/kaggle/input/rgcn-model/' for loading:

# In[ ]:


# Load the vectorized information of the knowledge graph
graph.load_model_from_np()


# # Generate Commentary
# 
# ## Approach
# 
# To generate commentary on target concept, we apply following steps:
# 
# 1. Recognize the searching terms from users and match them to concepts in our knowledge graph.
# 2. Serach the concept in the knowledge graph, find related concepts and visualize them.
# 3. Generate commentary from results in step 2, which consists of how many papers talk about the concept, which paper is the most cited, what knowledge do we know about it, what we infer by model and etc.

# ## Recognize concept
# 
# To recognize searching terms from users, we use trained NER (Named Entity Recognition) model from scispacy. The model 'en_core_sci_sm' uses full spaCy pipeline for biomedical data with a ~100k vocabulary. Then link the entity to entity code in UMLS, while it is also the entity code in our knowledge graph.
# 
# Install scispacy first:

# In[ ]:


get_ipython().system('pip install scispacy==0.2.4')


# In[ ]:


import spacy
from scispacy.umls_linking import UmlsEntityLinker

# Load NER model and link to UMLS
nlp_model = spacy.load('/kaggle/input/scispacy-model/en_core_sci_sm-0.2.4/en_core_sci_sm/en_core_sci_sm-0.2.4')
linker = UmlsEntityLinker(resolve_abbreviations=True)
nlp_model.add_pipe(linker)


# In[ ]:


# The function to recognize users' searching terms
def get_entity_information(entity_text, nlp_model, linker, entity_dict):
    '''Recognize the query string as the concept in the graph.
    '''
    entity_type_code = {'T047': 'Disease',
                        'T028': 'Gene',
                        'T121': 'PharmaSub',
                        'T103': 'Chemical',
                        #'T005': 'Virus',
                        #'T001': 'Organism',
                        #'T053': 'Behavior',
                        'T184': 'Symptom'}
    
    entity_information_list = []
    entity_list = nlp_model(entity_text).ents
    if len(entity_list) == 0:
        entity_list = nlp_model(entity_text + ' is').ents
    
    for entity in entity_list:
        entity_flag = False
        
        if len(entity._.umls_ents) == 0:
            # Not in UMLS
            if str(entity).lower() in ['remdesivir']:
                entity_code = 'PharmaSub' + '_' + 'C4726677'
                entity_name = 'Remdesivir'
                entity_definition = 'Remdesivir is an investigational nucleotide analog' +                                    'with broad-spectrum antiviral activity bothin vitroandin vivoin animal models' +                                    'against multiple emerging viral pathogens, including Ebola, Marburg, MERS and SARS.'
                entity_flag = True
            elif str(entity).lower() in ['covid-19', 'sars-cov-2', '2019-ncov', 'hcov-19',
                                         'new coronavirus', 'novel coronavirus']:
                entity_code = 'Disease' + '_' + 'C5203670'
                entity_name = 'COVID-19'
                entity_definition = 'A potentially severe acute respiratory infection' +                                    'caused by the novel coronavirus severe acute respiratory syndrome' +                                    'coronavirus 2 (SARS-CoV-2).'
                entity_flag = True
        else:
            # In UMLS
            for umls_entity, score in entity._.umls_ents:
                if score == 1:
                    cui_text = linker.umls.cui_to_entity[umls_entity]
                    cui_id = cui_text[0]
                    for tui_id in cui_text[3]:
                        if tui_id in entity_type_code:
                            # In desired entity types
                            semantic_type = entity_type_code[tui_id]
                            entity_code = semantic_type + '_' + cui_id
                            entity_name = cui_text[1]
                            entity_definition = cui_text[4]
                            if entity_code in graph.entity_dict:
                                entity_flag = True
                                break
                if entity_flag == True:
                    break
        
        if entity_flag == True:
            if (entity_code, entity_name, entity_definition) not in entity_information_list:
                entity_information_list.append((entity_code, entity_name, entity_definition))
        
    return entity_information_list


# ## Search and visualize related concepts
# 
# We use pyecharts to visualize subgraph. It enables excellent interactive experience in Jupyter Notebook.

# In[ ]:


get_ipython().system('pip install pyecharts')


# In[ ]:


from pyecharts import options as opts
from pyecharts.charts import Graph

class GraphVisualization:
    '''Visualize subgraph'''
    def __init__(self, subgraph, name_dict, title='COVID-19 knowledge graph', repulsion=80, labelShow=False):
        self.subgraph = subgraph
        self.color = {'Disease': '#FF7F50', 'Gene': '#48D1CC', 'PharmaSub': '#B3EE3A',
                      'Chemical': '#C71585', 'Symptom': '#FF0000'}
        self.cate = {'Disease': 0, 'Gene': 1, 'PharmaSub': 2,
                     'Chemical': 3, 'Symptom': 4}
        self.categories = [{'name':'Disease', 'itemStyle': {'normal': {'color': self.color['Disease']}}},
                           {'name':'Gene', 'itemStyle': {'normal': {'color': self.color['Gene']}}},
                           {'name':'PharmaSub', 'itemStyle': {'normal': {'color': self.color['PharmaSub']}}},
                           {'name':'Chemical', 'itemStyle': {'normal': {'color': self.color['Chemical']}}},
                           {'name':'Symptom', 'itemStyle': {'normal': {'color': self.color['Symptom']}}}]
        self.visual = self.visualize_graph(name_dict, title, repulsion, labelShow)
        
    def get_entity_name(self, entity_code, name_dict):
        if entity_code.split('_')[0] == 'Paper':
            entity_name = entity_code
        else:
            entity_name = name_dict[entity_code]
        return entity_name

    def get_nodes_stats(self):
        reverse_cate = dict([(v,k) for (k,v) in self.cate.items()])
        return pd.Series([reverse_cate[node['category']] for node in self.nodes]).value_counts()

    def visualize_graph(self, name_dict, title, repulsion, labelShow):

        self.nodes = []
        for entity in list(set(self.subgraph['subj']) | set(self.subgraph['obj'])):
            self.nodes.append({'name': self.get_entity_name(entity, name_dict),
                               'symbolSize': max(10, np.log1p(sum((self.subgraph['subj']==entity) |\
                                                                  (self.subgraph['obj']==entity)))*10//1),
                               'category':self.cate[entity.split('_')[0]]})
        self.links = []
        for index, row in self.subgraph.iterrows():
            self.links.append({'source': self.get_entity_name(row[0], name_dict),
                               'target': self.get_entity_name(row[2], name_dict),
                               'value': row[1]})
            
        g = (
             Graph()
             .add('', self.nodes, self.links, self.categories,
                  repulsion=repulsion, label_opts=opts.LabelOpts(is_show=labelShow))
             .set_global_opts(title_opts=opts.TitleOpts(title=title), legend_opts=opts.LegendOpts(orient='vertical',
                                                                                                  pos_left='2%',
                                                                                                  pos_top='40%',
                                                                                                  legend_icon='circle'))
             .render_notebook()
            )
        return g


# Load these entities' names to transfer them from entity code to actual name:

# In[ ]:


def generate_name_dict():
    name_dict = {}
    # Disease
    df_disease = pd.read_csv('/kaggle/input/entity-name/disease.csv', header=None, names=['name', 'code'])
    df_disease = df_disease.drop_duplicates(['code'], keep='first')
    for index, row in df_disease.iterrows():
        name_dict['Disease' + '_' + row[1]] = row[0]
    name_dict['Disease_C5203670'] = 'COVID-19'
    # PharmaSub
    df_pharma = pd.read_csv('/kaggle/input/entity-name/pharma_sub.csv', header=None, names=['name', 'code'])
    df_pharma = df_pharma.drop_duplicates(['code'], keep='first')
    for index, row in df_pharma.iterrows():
        name_dict['PharmaSub' + '_' + row[1]] = row[0]
    name_dict['PharmaSub_C4726677'] = 'Remdesivir'
    # Symptom
    df_symptom = pd.read_csv('/kaggle/input/entity-name/symptom.csv', sep=';', header=None, names=['name', 'code'])
    df_symptom = df_symptom.drop_duplicates(['code'], keep='first')
    for index, row in df_symptom.iterrows():
        name_dict['Symptom' + '_' + row[1]] = row[0]
    # Gene    
    df_gene = pd.read_csv('/kaggle/input/entity-name/gene.csv', header=None, names=['name', 'code'])
    df_gene = df_gene.drop_duplicates(['code'], keep='first')
    for index, row in df_gene.iterrows():
        name_dict['Gene' + '_' + row[1]] = row[0]
    # Chemical   
    df_chemical = pd.read_csv('/kaggle/input/entity-name/chemical.csv', header=None, names=['name', 'code'])
    df_chemical = df_chemical.drop_duplicates(['code'], keep='first')
    for index, row in df_chemical.iterrows():
        name_dict['Chemical' + '_' + row[1]] = row[0]
    return name_dict

# Load entity name dictionary
name_dict = generate_name_dict()


# Function to get the most cited paper using page rank score:

# In[ ]:


def get_most_cited_paper_title(paper_list):
    df_score = pd.read_csv('/kaggle/input/covid-challenge/df_paper_score.csv')
    
    df_part = df_score[df_score['id'].isin(paper_list)]
    if len(df_part) > 0:
        title = df_part.sort_values(by='score').iloc[-1, 1]
    else:
        title = ''
    
    return title


# ## Commentary generation
# 
# We generate different kinds of commentary for different types of concepts: Symptom, Pharmacy Substance, Disease, Gene and Chemical.
# 
# Generally, for each concept we generate commentary about:
# 1. How many papers talk about the concept. Do they also talk about COVID-19. Which of them is the most cited.
# 2. What do we know about this concept from these public papers.
# 3. What does UMLS library talk about this concept.
# 4. What do we infer by R-GCN.

# In[ ]:


#Utils
def get_commentary_paper(paper_list):
    if len(paper_list) > 0:
        commentary_paper = 'There are {:d} papers mentioned it in the dataset.'.format(len(paper_list))
        title = get_most_cited_paper_title(paper_list)
        if title != '':
            commentary_paper += ' "{:s}" is the most cited.'.format(title)
    else:
        commentary_paper = ' No paper mentioned it in the dataset.'
    return commentary_paper

def get_commentary_covid(paper_list):
    all_covid_paper = graph.get_subject_with_relation_and_object(('Paper', '0_ISABOUTDisease', 'Disease'), 'Disease_C5203670')
    covid_paper_list = list(set(paper_list).intersection(set(all_covid_paper)))
    commentary_covid = ''
    if len(covid_paper_list) > 0:
        commentary_covid = ' {:d} of these papers also mentioned COVID-19.'.format(len(covid_paper_list))
    else:
        if len(paper_list) > 0:
            commentary_covid = ' None of these papers mentioned COVID-19.'
    return commentary_covid


# Seperate functions for different types of concepts:

# In[ ]:


def generate_commentary_symptom(graph, entity_information, name_dict):
    # Unpack the information of the entity
    entity_code, entity_name, entity_definition = entity_information
    commentary_definition = '\nDefinition: ' + entity_definition + '\n'
    
    # Get the list of papers mentioned this symptom
    paper_list = graph.get_subject_with_relation_and_object(('Paper', '0_ISABOUTSymptom', 'Symptom'), entity_code)
    commentary_paper = get_commentary_paper(paper_list)
    commentary_covid = get_commentary_covid(paper_list)
    
    # Information from UMLS: disease
    disease_list = graph.get_object_with_subject_and_relation(entity_code, ('Symptom', '2_MANIFESTATION_OF', 'Disease'))
    disease_name_list = ['"' + name_dict[disease] + '"' for disease in disease_list if disease in name_dict]
    if len(disease_name_list) > 0:
        commentary_disease = '\n' + ', '.join(disease_name_list[0:5]) + ' also has this symptom.'
    else:
        commentary_disease = ''
    
    # Information from UMLS: pharma
    pharma_list = graph.get_subject_with_relation_and_object(('PharmaSub', '2_TREATS', 'Symptom'), entity_code)
    pharma_name_list = ['"' + name_dict[pharma] + '"' for pharma in pharma_list if pharma in name_dict]
    if len(pharma_name_list) > 0:
        commentary_pharma = '\n' + ', '.join(pharma_name_list[0:5]) + ' may treat this symptom.'
    else:
        commentary_pharma = ''
        
    # Information from papers: disease
    disease_list = graph.get_object_with_subject_and_relation(entity_code, ('Symptom', '1_MANIFESTATION_OF', 'Disease'))
    disease_name_list = ['"' + name_dict[disease] + '"' for disease in disease_list if disease in name_dict]
    if len(disease_name_list) > 0:
        commentary_disease_paper = '\nFrom papers, we know: ' + ', '.join(disease_name_list[0:5]) + ' also has this symptom.'
    else:
        commentary_disease_paper = ''
    
    # Information from papers: pharma
    pharma_list = graph.get_subject_with_relation_and_object(('PharmaSub', '1_TREATS', 'Symptom'), entity_code)
    pharma_name_list = ['"' + name_dict[pharma] + '"' for pharma in pharma_list if pharma in name_dict]
    if len(pharma_name_list) > 0:
        commentary_pharma_paper = '\nFrom papers, we know: ' + ', '.join(pharma_name_list[0:5]) + ' may treat this symptom.'
    else:
        commentary_pharma_paper = ''
        
    # Predict with the model: pharma
    pharma_list = graph.get_most_possible_subject_with_relation_and_object(('PharmaSub', '1_TREATS', 'Symptom'), entity_code)
    pharma_name_list = ['"' + name_dict[pharma] + '"' for pharma in pharma_list if pharma in name_dict]
    if len(pharma_name_list) > 0:
        commentary_pharma_model = '\nGraph model predicts: ' + ', '.join(pharma_name_list[0:5]) + ' may treat this symptom.'
    else:
        commentary_pharma_model = ''
        
    commentary = 'Your query is known as "{:s}", symptom.'.format(entity_name) + commentary_definition +                 commentary_paper + commentary_covid +                 commentary_disease + commentary_pharma +                 commentary_disease_paper + commentary_pharma_paper +                 commentary_pharma_model
    return commentary


# In[ ]:


def generate_commentary_pharma(graph, entity_information, name_dict):
    # Unpack the information of the entity
    entity_code, entity_name, entity_definition = entity_information
    commentary_definition = '\nDefinition: ' + entity_definition + '\n'
    
    # Get the list of papers mentioned this pharmacy
    paper_list = graph.get_subject_with_relation_and_object(('Paper', '0_ISABOUTPharmaSub', 'PharmaSub'), entity_code)
    commentary_paper = get_commentary_paper(paper_list)
    commentary_covid = get_commentary_covid(paper_list)
   
    # Information from UMLS: disease
    disease_list = graph.get_object_with_subject_and_relation(entity_code, ('PharmaSub', '2_TREATS', 'Disease'))
    disease_name_list = ['"' + name_dict[disease] + '"' for disease in disease_list if disease in name_dict]
    if len(disease_name_list) > 0:
        commentary_disease = '\n' + ', '.join(disease_name_list[0:5]) + ' may be treated by this pharmacy substance.'
    else:
        commentary_disease = ''
    
    # Information from papers: disease
    disease_list = graph.get_object_with_subject_and_relation(entity_code, ('PharmaSub', '1_TREATS', 'Disease'))
    disease_name_list = ['"' + name_dict[disease] + '"' for disease in disease_list if disease in name_dict]
    if len(disease_name_list) > 0:
        commentary_disease_paper = '\nFrom papers, we know: ' + ', '.join(disease_name_list[0:5]) +                                   ' may be treated by this pharmacy substance.'
    else:
        commentary_disease_paper = ''
        
    # Predict with the model: disease
    disease_list = graph.get_most_possible_object_with_subject_and_relation(entity_code, ('PharmaSub', '1_TREATS', 'Disease'))
    disease_name_list = ['"' + name_dict[disease] + '"' for disease in disease_list if disease in name_dict]
    if len(disease_name_list) > 0:
        commentary_disease_model = '\nGraph model predicts: ' + ', '.join(disease_name_list[0:5]) +                                   ' may be treated by this pharmacy substance.'
    else:
        commentary_disease_model = ''
    
    commentary = 'Your query is known as "{:s}", pharmacy substance.'.format(entity_name) + commentary_definition +                 commentary_paper + commentary_covid +                 commentary_disease + commentary_disease_paper +                 commentary_disease_model
    return commentary


# In[ ]:


def generate_commentary_disease(graph, entity_information, name_dict):
    # Unpack the information of the entity
    entity_code, entity_name, entity_definition = entity_information
    commentary_definition = '\nDefinition: ' + entity_definition + '\n'
    
    paper_list = graph.get_subject_with_relation_and_object(('Paper', '0_ISABOUTDisease', 'Disease'), entity_code)
    commentary_paper = get_commentary_paper(paper_list)
    commentary_covid = ''
    if entity_code != 'Disease_C5203670':
        commentary_covid = get_commentary_covid(paper_list)
    
    # Information from UMLS: symptom
    symptom_list = graph.get_subject_with_relation_and_object(('Symptom', '2_MANIFESTATION_OF', 'Disease'), entity_code)
    symptom_name_list = ['"' + name_dict[symptom] + '"' for symptom in symptom_list if symptom in name_dict]
    if len(symptom_name_list) > 0:
        commentary_symptom = '\n' + ', '.join(symptom_name_list[0:5]) + ' may be the symptoms of this disease.'
    else:
        commentary_symptom = ''
        
    # Information from UMLS: pharma
    pharma_list = graph.get_subject_with_relation_and_object(('PharmaSub', '2_TREATS', 'Disease'), entity_code)
    pharma_name_list = ['"' + name_dict[pharma] + '"' for pharma in pharma_list if pharma in name_dict]
    if len(pharma_name_list) > 0:
        commentary_pharma = '\n' + ', '.join(pharma_name_list[0:5]) + ' may treat this disease.'
    else:
        commentary_pharma = ''
        
    # Information from papers: symptom
    symptom_list = graph.get_subject_with_relation_and_object(('Symptom', '1_MANIFESTATION_OF', 'Disease'), entity_code)
    symptom_name_list = ['"' + name_dict[symptom] + '"' for symptom in symptom_list if symptom in name_dict]
    if len(symptom_name_list) > 0:
        commentary_symptom_paper = '\nFrom papers, we know: ' + ', '.join(symptom_name_list[0:5]) +                                   ' may be the symptoms of this disease.'
    else:
        commentary_symptom_paper = ''
        
    # Information from papers: pharma 
    pharma_list = graph.get_subject_with_relation_and_object(('PharmaSub', '1_TREATS', 'Disease'), entity_code)
    pharma_name_list = ['"' + name_dict[pharma] + '"' for pharma in pharma_list if pharma in name_dict]
    if len(pharma_name_list) > 0:
        commentary_pharma_paper = '\nFrom papers, we know: ' + ', '.join(pharma_name_list[0:5]) + ' may treat this disease.'
    else:
        commentary_pharma_paper = ''
        
    # Predict with the model: symptom
    symptom_list = graph.get_most_possible_subject_with_relation_and_object(('Symptom', '1_MANIFESTATION_OF', 'Disease'),
                                                                            entity_code)
    symptom_name_list = ['"' + name_dict[symptom] + '"' for symptom in symptom_list if symptom in name_dict]
    if len(symptom_name_list) > 0:
        commentary_symptom_model = '\nGraph model predicts: ' + ', '.join(symptom_name_list[0:5]) +                                   ' may be the symptoms of this disease.'
    else:
        commentary_symptom_model = ''
        
    # Predict with the model: pharma 
    pharma_list = graph.get_most_possible_subject_with_relation_and_object(('PharmaSub', '1_TREATS', 'Disease'), entity_code)
    pharma_name_list = ['"' + name_dict[pharma] + '"' for pharma in pharma_list if pharma in name_dict]
    if len(pharma_name_list) > 0:
        commentary_pharma_model = '\nGraph model predicts: ' + ', '.join(pharma_name_list[0:5]) + ' may treat this disease.'
    else:
        commentary_pharma_model = ''
    
    # Final commentary
    commentary = 'Your query is known as "{:s}", disease.'.format(entity_name) + commentary_definition +                 commentary_paper + commentary_covid +                 commentary_symptom + commentary_pharma +                 commentary_symptom_paper + commentary_pharma_paper +                 commentary_symptom_model + commentary_pharma_model
    return commentary


# In[ ]:


def generate_commentary_gene(graph, entity_information, name_dict):
    # Unpack the information of the entity
    entity_code, entity_name, entity_definition = entity_information
    commentary_definition = '\nDefinition: ' + entity_definition + '\n'
    
    paper_list = graph.get_subject_with_relation_and_object(('Paper', '0_ISABOUTGene', 'Gene'), entity_code)
    commentary_paper = get_commentary_paper(paper_list)
    commentary_covid = get_commentary_covid(paper_list)
    
    # Final commentary
    commentary = 'Your query is known as "{:s}", gene.'.format(entity_name) + commentary_definition +                 commentary_paper + commentary_covid
    
    return commentary


# In[ ]:


def generate_commentary_chemical(graph, entity_information, name_dict):
    # Unpack the information of the entity
    entity_code, entity_name, entity_definition = entity_information
    commentary_definition = '\nDefinition: ' + entity_definition + '\n'
    
    paper_list = graph.get_subject_with_relation_and_object(('Paper', '0_ISABOUTChemical', 'Chemical'), entity_code)
    commentary_paper = get_commentary_paper(paper_list)
    commentary_covid = get_commentary_covid(paper_list)
    
    # Final commentary
    commentary = 'Your query is known as "{:s}", chemical.'.format(entity_name) + commentary_definition +                 commentary_paper + commentary_covid
    
    return commentary


# In[ ]:


# Main function for commentary generation
def generate_commentary(graph, entity_information, name_dict, semantic_type):
    if semantic_type == 'Symptom':
        commentary = generate_commentary_symptom(graph, entity_information, name_dict)
    elif semantic_type == 'PharmaSub':
        commentary = generate_commentary_pharma(graph, entity_information, name_dict)
    elif semantic_type == 'Disease':
        commentary = generate_commentary_disease(graph, entity_information, name_dict)
    elif semantic_type == 'Gene':
        commentary = generate_commentary_gene(graph, entity_information, name_dict)
    elif semantic_type == 'Chemical':
        commentary = generate_commentary_chemical(graph, entity_information, name_dict)
    else:
        commentary = 'Not available now.'
    return commentary


# In[ ]:


# Main search function
def search(entity_text, nlp_model, linker, graph, name_dict):
    entity_information_list = get_entity_information(entity_text, nlp_model, linker, graph.entity_dict)
    
    if len(entity_information_list) == 0:
        raise Exception('Not an exsiting entity in the knowledge graph.')
    elif len(entity_information_list) > 1:
        raise Exception('More than one entity. Please search one at a time.')
    else:
        entity_information = entity_information_list[0]
        entity_code, entity_name, entity_definition = entity_information
        semantic_type = entity_code.split('_')[0]
        
        # Visualize the subgraph
        subgraph = graph.get_subgraph(seed=entity_code)
        g = GraphVisualization(subgraph, name_dict,
                               title='1st Degree Connection Knowledge Graph of {:s}'.format(entity_name))
        # Generate commentary
        commentary = generate_commentary(graph, entity_information, name_dict, semantic_type)
        
    return g, commentary


# # Real-world case analysis

# In[ ]:


g, commentary = search('COVID-19', nlp_model, linker, graph, name_dict)


# In[ ]:


g.visual


# In[ ]:


g.get_nodes_stats()


# In[ ]:


print(commentary)


# In[ ]:


g, commentary = search('Breathlessness', nlp_model, linker, graph, name_dict)


# In[ ]:


g.visual


# In[ ]:


g.get_nodes_stats()


# In[ ]:


print(commentary)


# In[ ]:


g, commentary = search('Remdesivir', nlp_model, linker, graph, name_dict)


# In[ ]:


g.visual


# In[ ]:


g.get_nodes_stats()


# In[ ]:


print(commentary)


# # Future Works

# In this notebook, we introduced our knowledge graph-based commentary generation tool for COVID-19 related concepts. Future works includes using graph database for our knowledge graph for more user friendly user interaction, developing a web service with better visualization and connecting to open medical library for automatical crawlling and updating.
