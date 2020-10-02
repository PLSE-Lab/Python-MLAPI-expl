#!/usr/bin/env python
# coding: utf-8

# **In my previous [kernel](https://www.kaggle.com/interneuron/randomwire) I showed how to use random graphs to build a convolutional neural network.** 
# 
# While these randomly wired networks perform quite well, they cannot (yet) compete with networks pre-trained on the ImageNet. So until someone finds a particular random net they like enough to train on ImageNet and release the weights, training from scratch is the only option. Proper references for the random network can be found in the other kernel.
# 
# Here, we will try something...interesting.
# 
# While researching about random networks, I found [this](https://github.com/vinayprabhu/Network_Science_Meets_Deep_Learning/blob/master/1_MNIST_C_Elegans.ipynb) intriguing work. Using the techniques from the RandomWire paper for converting a graph into a functional cnn, one can take, say, the known wiring diagram of the nervous system of the nematode worm *Caenorhabditis elegans*, and turn it into a convolutional network. I simply added functionality to plug the graph of the worm's nervous system into the network builder.
# 
# I find this idea quite weird. 
# 
# While *C. elegans* does posess some light-sensitive cells to help it avoid dangerous stuff like UV radiation, it has no eyes, lives in the dark, and completely lacks what we understand as a visual system. What we are going to do here is repourpose the animals *entire* nervous system for image classification. 
# 
# Wtf?
# 
# Nevertheless, it is a network architecture I have not seen used before and I hope others will find it as interesting as I do.

# **First we get the graph of the nervous system**

# In[ ]:


get_ipython().system(' wget https://www.cise.ufl.edu/research/sparse/MM/Newman/celegansneural.tar.gz')
get_ipython().system(' tar -xvzf celegansneural.tar.gz')


# In[ ]:


import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import networkx as nx

celegans = scipy.io.mmread('celegansneural/celegansneural.mtx').toarray()
celegans


# In[ ]:


celegans.shape


# In[ ]:


vfunc = np.vectorize(lambda a : 1 if a != 0 else 0)
celegansbin = vfunc(celegans)


# In[ ]:


celegans_g = nx.from_numpy_array(celegansbin)
nx.draw(celegans_g, node_size=7)


# Looks like a mess, but there must be something to it as the worms have been successfully using it for quite some time.

# In[ ]:


celegans_g.number_of_edges()


# In[ ]:


celegans_g.number_of_nodes()


# In[ ]:


#!pip install torchviz


# In[ ]:


import pandas as pd, numpy as np, os, sys
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.nn.functional as F
import torchvision as vision
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from PIL import Image
from contextlib import contextmanager
import networkx as nx
from joblib import Parallel, delayed
from tqdm import tqdm
#import torchviz
#from torchviz import make_dot, make_dot_from_trace
#print(os.listdir("../input"))


# In[ ]:





# Another difference from the previous kernel, I added a function to make a second kind of **small-world** graph: the **newman_watts_strogatz_graph**.

# In[ ]:


class RandomGraph(object):
    def __init__(self, node_num, p, k=4, m=5, graph_mode="WS"):
        self.node_num = node_num
        self.p = p
        self.k = k
        self.m = m
        self.graph_mode = graph_mode

    def make_graph(self):
        # reference
        # https://networkx.github.io/documentation/networkx-1.9/reference/generators.html

        # Code details,
        # In the case of the nx.random_graphs module, we can give the random seeds as a parameter.
        # But I have implemented it to handle it in the module.
        if self.graph_mode is "ER":
            graph = nx.random_graphs.erdos_renyi_graph(self.node_num, self.p)
        elif self.graph_mode is "WS":
            graph = nx.random_graphs.watts_strogatz_graph(self.node_num, self.k, self.p)
        elif self.graph_mode is "NWS":
            graph = nx.random_graphs.newman_watts_strogatz_graph(self.node_num, self.k, self.p)
        elif self.graph_mode is "BA":
            graph = nx.random_graphs.barabasi_albert_graph(self.node_num, self.m)
        elif self.graph_mode is 'worm':
            graph = celegans_g

        return graph

    def get_graph_info(self, graph):
        in_edges = {}
        in_edges[0] = []
        nodes = [0]
        end = []
        if self.graph_mode is 'worm':
            self.node_num = len(graph.nodes())
        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            neighbors.sort()

            edges = []
            check = []
            for neighbor in neighbors:
                if node > neighbor:
                    edges.append(neighbor + 1)
                    check.append(neighbor)
            if not edges:
                edges.append(0)
            in_edges[node + 1] = edges
            if check == neighbors:
                end.append(node + 1)
            nodes.append(node + 1)
        in_edges[self.node_num + 1] = end
        nodes.append(self.node_num + 1)

        return nodes, in_edges
    

    def save_random_graph(self, graph, path):
        if not os.path.isdir("saved_graph"):
            os.mkdir("saved_graph")
        nx.write_yaml(graph, "./saved_graph/" + path)

    def load_random_graph(self, path):
        return nx.read_yaml("./saved_graph/" + path)
    
def _save_random_graph(graph, path):
    if not os.path.isdir("saved_graph"):
        os.mkdir("saved_graph")
    nx.write_yaml(graph, "./saved_graph/" + path)

def _load_random_graph(path):
    return nx.read_yaml("./saved_graph/" + path)

def _get_graph_info(graph):
    in_edges = {}
    in_edges[0] = []
    nodes = [0]
    end = []
    node_num = len(graph.nodes())
    for node in graph.nodes():
        neighbors = list(graph.neighbors(node))
        neighbors.sort()

        edges = []
        check = []
        for neighbor in neighbors:
            if node > neighbor:
                edges.append(neighbor + 1)
                check.append(neighbor)
        if not edges:
            edges.append(0)
        in_edges[node + 1] = edges
        if check == neighbors:
            end.append(node + 1)
        nodes.append(node + 1)
    in_edges[node_num + 1] = end
    nodes.append(node_num + 1)

    return nodes, in_edges


# 
# 
# **In the previous kernel I showed below a simple WS graph. In the cell below it get the same sort of representation from the worm. Uncomment the cell with in_edges to see, its quite a bit more complex than the little WS graph... **
# 

# In[ ]:


rg = RandomGraph(8, 0.75)
gf = rg.make_graph()
rg.get_graph_info(gf)


# In[ ]:


nodes, in_edges = _get_graph_info(celegans_g)


# In[ ]:


#in_edges


# In[ ]:


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# reference, Thank you.
# https://github.com/tstandley/Xception-PyTorch/blob/master/xception.py
# Reporting 1,
# I don't know which one is better, between 'bias=False' and 'bias=True'
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels, bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

        # self.apply(weights_init)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise(x)
        return x


# ReLU-convolution-BN triplet
class Unit(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Unit, self).__init__()

        self.dropout_rate = 0.2

        self.unit = nn.Sequential(
            nn.ReLU(),
            SeparableConv2d(in_channels, out_channels, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.dropout_rate)
        )

    def forward(self, x):
        return self.unit(x)


# Reporting 2,
# In the paper, they said "The aggregation is done by weighted sum with learnable positive weights".
class Node(nn.Module):
    def __init__(self, in_degree, in_channels, out_channels, stride=1):
        super(Node, self).__init__()
        self.in_degree = in_degree
        if len(self.in_degree) > 1:
            # self.weights = nn.Parameter(torch.zeros(len(self.in_degree), requires_grad=True))
            self.weights = nn.Parameter(torch.ones(len(self.in_degree), requires_grad=True))
        self.unit = Unit(in_channels, out_channels, stride=stride)

    def forward(self, *input):
        if len(self.in_degree) > 1:
            x = (input[0] * torch.sigmoid(self.weights[0]))
            for index in range(1, len(input)):
                x += (input[index] * torch.sigmoid(self.weights[index]))
            out = self.unit(x)

            # different paper, add identity mapping
            # out += x
        else:
            out = self.unit(input[0])
        return out


class RandWire(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, is_train, name):
        super(RandWire, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.is_train = is_train
        self.name = name

        # get graph nodes and in edges
        if self.graph_mode is 'worm':
            graph_node = celegans_g
        else:
            graph_node = RandomGraph(self.node_num, self.p, graph_mode=graph_mode)
        if self.is_train:
            if self.graph_mode is 'worm':
                graph = graph_node
                self.nodes, self.in_edges = _get_graph_info(graph)
                _save_random_graph(graph, name)
            else: 
                graph = graph_node.make_graph()
                self.nodes, self.in_edges = get_graph_info(graph)
                graph_node.save_random_graph(graph, name)
        else:
            if self.graph_mode is 'worm':
                graph = _load_random_graph(name)
                self.nodes, self.in_edges = _get_graph_info(graph)
            else:
                graph = graph_node.load_random_graph(name)
                self.nodes, self.in_edges = graph_node.get_graph_info(graph)

        # define input Node
        self.module_list = nn.ModuleList([Node(self.in_edges[0], self.in_channels, self.out_channels, stride=2)])
        # define the rest Node
        self.module_list.extend([Node(self.in_edges[node], self.out_channels, self.out_channels) 
                                 for node in self.nodes if node > 0])

    def forward(self, x):
        memory = {}
        # start vertex
        out = self.module_list[0].forward(x)
        memory[0] = out

        # the rest vertex
        for node in range(1, len(self.nodes) - 1):
            # print(node, self.in_edges[node][0], self.in_edges[node])
            if len(self.in_edges[node]) > 1:
                out = self.module_list[node].forward(*[memory[in_vertex] for in_vertex in self.in_edges[node]])
            else:
                out = self.module_list[node].forward(memory[self.in_edges[node][0]])
            memory[node] = out

        # Reporting 3,
        # How do I handle the last part?
        # It has two kinds of methods.
        # first, Think of the last module as a Node and collect the data by proceeding in the same way as the previous operation.
        # second, simply sum the data and export the output.

        # My Opinion
        # out = self.module_list[self.node_num + 1].forward(*[memory[in_vertex] for in_vertex in self.in_edges[self.node_num + 1]])

        # In paper
        # print("self.in_edges: ", self.in_edges[self.node_num + 1], self.in_edges[self.node_num + 1][0])
        out = memory[self.in_edges[self.node_num + 1][0]]
        for in_vertex_index in range(1, len(self.in_edges[self.node_num + 1])):
            out += memory[self.in_edges[self.node_num + 1][in_vertex_index]]
        out = out / len(self.in_edges[self.node_num + 1])
        return out


# In[ ]:


class RandNN(nn.Module):
    def __init__(self, node_num, p, in_channels, out_channels, graph_mode, model_mode, dataset_mode, is_train):
        super(RandNN, self).__init__()
        self.node_num = node_num
        self.p = p
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.graph_mode = graph_mode
        self.model_mode = model_mode
        self.is_train = is_train
        self.dataset_mode = dataset_mode

        self.num_classes = 1103
        self.dropout_rate = 0.2

        if self.dataset_mode is "met":
            self.num_classes = 1103

        if self.model_mode is "met":
            self.REGULAR_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels // 2)
            )
            self.REGULAR_conv2 = nn.Sequential(
                RandWire(self.node_num // 2, self.p, self.in_channels // 2, self.out_channels, self.graph_mode, self.is_train, name="REGULAR_conv2")
            )
            self.REGULAR_conv3 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels, self.out_channels * 2, self.graph_mode, 
                         self.is_train, name="REGULAR_conv3")
            )
            self.REGULAR_conv4 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels * 2, self.out_channels * 4, self.graph_mode, 
                         self.is_train, name="REGULAR_conv4")
            )
            self.REGULAR_conv5 = nn.Sequential(
                RandWire(self.node_num, self.p, self.in_channels * 4, self.out_channels * 8, self.graph_mode, 
                         self.is_train, name="REGULAR_conv5")
            )
            self.REGULAR_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 8, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )

#        self.output = nn.Sequential(
#            nn.Dropout(self.dropout_rate),
#            nn.Linear(1280, self.num_classes)
#        )
        
        
        elif self.model_mode is "worm":
            self.worm_conv1 = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=self.out_channels // 2, kernel_size=3, padding=1),
                nn.BatchNorm2d(self.out_channels // 2)
            )
            
            self.worm_conv2 = nn.Sequential(
                RandWire(self.node_num // 2, self.p, self.in_channels // 2, self.out_channels, 
                         self.graph_mode, self.is_train, name="worm_conv2")
            )
            
            self.worm_classifier = nn.Sequential(
                nn.Conv2d(self.in_channels * 8, 1280, kernel_size=1),
                nn.BatchNorm2d(1280)
            )
            
        self.output = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(1280, self.num_classes)
        )

    def forward(self, x):
        if self.model_mode is "met":
            out = self.REGULAR_conv1(x)
            out = self.REGULAR_conv2(out)
            out = self.REGULAR_conv3(out)
            out = self.REGULAR_conv4(out)
            out = self.REGULAR_conv5(out)
            out = self.REGULAR_classifier(out)
        elif self.model_mode is 'worm':
            out = self.worm_conv1(x)
            out = self.worm_conv2(out)
            out = self.worm_classifier(out)

        # global average pooling
        out = F.avg_pool2d(out, kernel_size=x.size()[2:])
        out = torch.squeeze(out)
        out = self.output(out)

        return out


# In[ ]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# In[ ]:


def rw(f=None):
    m = RandNN(1, 0.75, 32, 32, 'worm', 'met', 'met', 'train')
    return m


# In[ ]:


m = rw()


# In[ ]:


count_parameters(m)


# In[ ]:


tr = pd.read_csv('../input/train.csv')
tr = tr.sample(frac=0.1).reset_index(drop=True)
te = pd.read_csv('../input/sample_submission.csv')


# In[ ]:


import fastai
from fastai.vision import *
path = Path('../input/')


# In[ ]:


SZ = 128
BS = 16

train, test = [ImageList.from_df(df, path=path, cols='id', folder=folder, suffix='.png') 
               for df, folder in zip([tr, te], ['train', 'test'])]
data = (train.split_by_rand_pct(0.1, seed=42)
        .label_from_df(cols='attribute_ids', label_delim=' ')
        .add_test(test)
        .transform(get_transforms(), size=SZ, resize_method=ResizeMethod.PAD, padding_mode='border',)
        .databunch(path=Path('.'), bs=BS).normalize(imagenet_stats))


# In[ ]:


# Source: https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/78109
class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma

    def forward(self, logit, target):
        target = target.float()
        max_val = (-logit).clamp(min=0)
        loss = logit - logit * target + max_val +                ((-max_val).exp() + (-logit - max_val).exp()).log()

        invprobs = F.logsigmoid(-logit * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        if len(loss.size())==2:
            loss = loss.sum(dim=1)
        return loss.mean()


# In[ ]:


learn = cnn_learner(data, 
                    base_arch=rw, 
                    cut = 5,
                    loss_func=FocalLoss(), 
                    metrics=fbeta)

learn = learn.to_fp16(loss_scale=64, dynamic=True)


# In[ ]:


#learn.lr_find()
#learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(8, slice(1e-3,2e-2))


# In[ ]:


learn.recorder.plot()
learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


def find_best_fixed_threshold(preds, targs, do_plot=True):
    score = []
    thrs = np.arange(0, 0.5, 0.01)
    for thr in progress_bar(thrs):
        score.append(fbeta(valid_preds[0],valid_preds[1], thresh=thr))
    score = np.array(score)
    pm = score.argmax()
    best_thr, best_score = thrs[pm], score[pm].item()
    print(f'thr={best_thr:.3f}', f'F2={best_score:.3f}')
    if do_plot:
        plt.plot(thrs, score)
        plt.vlines(x=best_thr, ymin=score.min(), ymax=score.max())
        plt.text(best_thr+0.03, best_score-0.01, f'$F_{2}=${best_score:.3f}', fontsize=14);
        plt.show()
    return best_thr

i2c = np.array([[i, c] for c, i in learn.data.train_ds.y.c2i.items()]).astype(int) # indices to class number correspondence

def join_preds(preds, thr):
    return [' '.join(i2c[np.where(t==1)[0],1].astype(str)) for t in (preds[0].sigmoid()>thr).long()]


# In[ ]:


# Validation predictions
valid_preds = learn.get_preds(DatasetType.Valid)
best_thr = find_best_fixed_threshold(*valid_preds)


# In[ ]:


test_preds = learn.get_preds(DatasetType.Test)
te.attribute_ids = join_preds(test_preds, best_thr)
te.head()


# In[ ]:


te.to_csv('submission.csv', index=False)


# In[ ]:


x = torch.randn(2,3,64,64).half().cuda()
make_dot(learn.model(x), params=dict(learn.model.named_parameters()))


# In[ ]:





# In[ ]:


x = torch.randn(2,3,64,64).half().cuda()
make_dot(learn.model[0][0:2](x), params=dict(learn.model[0][0:2].named_parameters()))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




