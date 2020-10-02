#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install torchviz')


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
import torchviz
from torchviz import make_dot, make_dot_from_trace
print(os.listdir("../input"))


# In[ ]:





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


# In[ ]:


rg = RandomGraph(8, 0.75)
gf = rg.make_graph()
rg.get_graph_info(gf)


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
                self.nodes, self.in_edges = graph_node.get_graph_info(graph)
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

        self.num_classes = 10
        self.dropout_rate = 0.2

        if self.dataset_mode is "met":
            self.num_classes = 10

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
    m = RandNN(32, 0.75, 32, 32, 'WS', 'met', 'met', 'train')
    return m


# In[ ]:


m = rw()


# In[ ]:


count_parameters(m)


# In[ ]:


tr = pd.read_csv('../input/train.csv')
#tr = tr.sample(frac=0.3).reset_index(drop=True)
te = pd.read_csv('../input/test.csv')


# In[ ]:


import fastai
from fastai.vision import *
path = Path('../input/')


# In[ ]:


tr.shape


# In[ ]:


tr.head()


# In[ ]:


class CustomImageItemList(ImageList):
    def open(self, fn):
        img = fn.reshape(28, 28)
        img = np.stack((img,)*3, axis=-1) # convert to 3 channels
        return Image(pil2tensor(img, dtype=np.float32))

    @classmethod
    def from_csv_custom(cls, path:PathOrStr, csv_name:str, imgIdx:int=1, header:str='infer', **kwargs) -> 'ItemList':
        df = pd.read_csv(Path(path)/csv_name, header=header)
        res = super().from_df(df, path=path, cols=0, **kwargs)
        # convert pixels to an ndarray
        res.items = df.iloc[:,imgIdx:].apply(lambda x: x.values / 783.0, axis=1).values
        return res


# In[ ]:


test = CustomImageItemList.from_csv_custom(path=path, csv_name='test.csv', imgIdx=0)
data = (CustomImageItemList.from_csv_custom(path=path, csv_name='train.csv')
                       .split_by_rand_pct(0.1)
                       .label_from_df(cols='label')
                       .add_test(test, label=0)
                       .databunch(bs=512, num_workers=0)
                       .normalize(imagenet_stats))


# In[ ]:


data.show_batch(rows=3, figsize=(12,9))


# In[ ]:


learn = cnn_learner(data, 
                    base_arch=rw, 
                    cut = 6,
                    metrics=accuracy,
                    model_dir='/tmp/models')

learn = learn.to_fp16(loss_scale=64, dynamic=True)


# In[ ]:


#learn.lr_find()
#learn.recorder.plot()


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(25, slice(1e-3,2e-2))


# In[ ]:


learn.recorder.plot()
learn.recorder.plot_losses()
learn.recorder.plot_metrics()


# In[ ]:


preds, y, losses = learn.get_preds(ds_type=DatasetType.Test, with_loss=True)


# In[ ]:


y = torch.argmax(preds, dim=1)


# In[ ]:


submission_df = pd.DataFrame({'ImageId': range(1, len(y) + 1), 'Label': y}, columns=['ImageId', 'Label'])
submission_df.to_csv('submission.csv', index=False)
submission_df.head()


# In[ ]:





# In[ ]:


x = torch.randn(2,3,64,64).half().cuda()
make_dot(learn.model(x), params=dict(learn.model.named_parameters()))


# In[ ]:





# In[ ]:


x = torch.randn(2,3,64,64).half().cuda()
make_dot(learn.model[0][0:2](x), params=dict(learn.model[0][0:2].named_parameters()))

