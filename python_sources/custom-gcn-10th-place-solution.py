#!/usr/bin/env python
# coding: utf-8

# In this kernel, we train a custom GCN (based on the starter kit provided by @hengck23 ) on 2JHN. For a description of our final models, you can read the following discussions: 
# https://www.kaggle.com/c/champs-scalar-coupling/discussion/106271#latest-610727 
# https://www.kaggle.com/c/champs-scalar-coupling/discussion/106293#latest-611400
# 
# As well as using the node and edge features for each molecule, we found success including some of the features used in our earlier LightGBM models. The LGB features that we included depended on the #J, because features like dihedral angles aren't so useful for 1J.

# In[ ]:


TYPE_TO_TRAIN = '2JHN'
NUM_ITERS = 20 * 1000 # change this to a high number, i.e., 100000
BATCH_SIZE = 70 #batch size over 100 usually leads to memory errors.
NUM_PROPAGATE = 2 #increasing num propagate leads to longer training times.  Make sure you adjust Num_iters correctly so the kernel doesn't time out
NUM_S2S=6

KFOLD_SEED = 2020
N_FOLD = 8
FOLD_TO_TRAIN = 0 # fold index to train, only specify from 0 to N_FOLD-1

DATA_DIR = '../input/champs-scalar-coupling/'
SPLIT_DIR = '../input/champs-scalar-coupling/'
GRAPH_DIR = '../input/gcn-data-full-v2/'
GRAPH_TRAIN_PICKLE = 'graph-train-full.pickle'
GRAPH_TEST_PICKLE = 'graph-test.pickle'

#controls which secondLayerFeat are used - uncomment the one associated with your type
#LGB_COLS = list(range(41))+[44,45,46,47,48,49]          #1J
LGB_COLS = list(range(42))+[44,45,46,47,50,51,52]       #2J
#LGB_COLS = list(range(44))+[44,45,46,47,53,54,55,56]     #3J
NODE_COLS = range(61) #controls which node features are used

RANDOM_SEED = 1991
STARTING_LR=.001


# In[ ]:


NUM_TARGET =  8

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
get_ipython().system('conda install pytorch torchvision -c pytorch --yes')
import torchvision
import torch

torch.cuda.manual_seed(1)
torch.manual_seed(12345)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
get_ipython().system('pip install torch-scatter')
get_ipython().system('pip install torch_geometric')
get_ipython().system('pip install torch_sparse')
get_ipython().system('pip install torch_cluster')

import copy, math, numpy as np, random, PIL

# torch libs
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel.data_parallel import data_parallel
from torch.nn.utils.rnn import *

# std libs
from timeit import default_timer as timer
from collections import OrderedDict
from multiprocessing import Pool
import multiprocessing as mp

#from pprintpp import pprint, pformat
import json, zipfile, csv, pandas as pd, pickle, glob, sys, time, copy, numbers, inspect, shutil, itertools 
from distutils.dir_util import copy_tree
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
from torch_scatter import *
from torch_geometric.utils import scatter_
import torch.nn as nn
from torch_geometric.nn import GATConv
# constant #
PI, INF, EPS = np.pi, np.inf, 1e-12


# In[ ]:


def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False
random_seed(RANDOM_SEED,True)       
class DecayScheduler():
    def __init__(self, base_lr, decay, step):
        super(DecayScheduler, self).__init__()
        self.step  = step
        self.decay = decay
        self.base_lr = base_lr
    def get_rate(self, epoch):
        lr = self.base_lr * (self.decay**(epoch // self.step))
        return lr

class NullScheduler():
    def __init__(self, lr=0.01 ):
        super(NullScheduler, self).__init__()
        self.lr    = lr
        self.cycle = 0
    def get_rate(self, epoch):
        return self.lr 

    
    
def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups: lr +=[ param_group['lr'] ]
    assert(len(lr)==1) #we support only one param_group
    lr = lr[0]
    return lr

def time_to_str(t, mode='min'):
    if mode=='min':
        t  = int(t)/60
        hr = t//60
        min = t%60
        return '%2d hr %02d min'%(hr,min)
    elif mode=='sec':
        t   = int(t)
        min = t//60
        sec = t%60
        return '%2d min %02d sec'%(min,sec)
    else: raise NotImplementedError
        
def compute_kaggle_metric( predict, coupling_value, coupling_type):
    mae     = [None]*NUM_COUPLING_TYPE
    log_mae = [None]*NUM_COUPLING_TYPE
    predict=((predict*targetStd)+targetMean)
    diff = np.fabs(predict-coupling_value)
    for t in range(NUM_COUPLING_TYPE):
        index = np.where(coupling_type==t)[0]
        if len(index)>0:
            m = diff[index].mean()
            log_m = np.log(m+1e-8)
            mae[t] = m
            log_mae[t] = log_m
        else: pass
    return mae, log_mae

COUPLING_TYPE_STATS=[
    #type   #mean, std, min, max
    '1JHC',  94.9761528641869,   18.27722399839607,   66.6008,   204.8800,
    '2JHC',  -0.2706244378832,    4.52360876732858,  -36.2186,    42.8192,
    '3JHC',   3.6884695895355,    3.07090647005439,  -18.5821,    76.0437,
    '1JHN',  47.4798844844683,   10.92204561670947,   24.3222,    80.4187,
    '2JHN',   3.1247536134185,    3.67345877025737,   -2.6209,    17.7436,
    '3JHN',   0.9907298624944,    1.31538940138001,   -3.1724,    10.9712,
    '2JHH', -10.2866051639817,    3.97960190019757,  -35.1761,    11.8542,
    '3JHH',   4.7710233597359,    3.70498129755812,   -3.0205,    17.4841,
]

NUM_COUPLING_TYPE = len(COUPLING_TYPE_STATS)//5
COUPLING_TYPE_MEAN = [ COUPLING_TYPE_STATS[i*5+1] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE_STD  = [ COUPLING_TYPE_STATS[i*5+2] for i in range(NUM_COUPLING_TYPE)]
COUPLING_TYPE      = [ COUPLING_TYPE_STATS[i*5  ] for i in range(NUM_COUPLING_TYPE)]

def read_pickle_from_file(pickle_file):
    with open(pickle_file, 'rb') as f:
        return(pickle.load(f))
    
class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='w'
        self.file = open(file, mode)

    def write(self, message, is_terminal=1, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass
    
def do_valid(net, valid_loader):

    valid_num = 0
    valid_predict = []
    valid_coupling_type  = []
    valid_coupling_value = []

    valid_loss = 0
    for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor,secondLayerFeat) in enumerate(valid_loader):

        net.eval()
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()

        coupling_value = coupling_value.cuda()
        coupling_index = coupling_index.cuda()

        with torch.no_grad():
            predict = net(node, edge, edge_index, node_index, coupling_index,secondLayerFeat)
            loss = criterion(predict, coupling_value)

        #---
        batch_size = len(infor)
        valid_predict.append(predict.data.cpu().numpy())
        valid_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
        valid_coupling_value.append(coupling_value.data.cpu().numpy())

        valid_loss += batch_size*loss.item()
        valid_num  += batch_size

        print('\r %8d /%8d'%(valid_num, len(valid_loader.dataset)),end='',flush=True)

        pass  #-- end of one data loader --
    assert(valid_num == len(valid_loader.dataset))
    valid_loss = valid_loss/valid_num

    #compute
    predict = np.concatenate(valid_predict)
    coupling_value = np.concatenate(valid_coupling_value)
    coupling_type  = np.concatenate(valid_coupling_type).astype(np.int32)
    mae, log_mae   = compute_kaggle_metric( predict, coupling_value, coupling_type,)

    num_target = NUM_COUPLING_TYPE
    for t in range(NUM_COUPLING_TYPE):
        if mae[t] is None:
            mae[t] = 0
            log_mae[t]  = 0
            num_target -= 1

    mae_mean, log_mae_mean = np.sum(mae)/num_target, np.sum(log_mae)/num_target

    valid_loss = log_mae + [valid_loss,mae_mean, log_mae_mean]
    return valid_loss

class Struct(object):
    def __init__(self, is_copy=False, **kwargs):
        self.add(is_copy, **kwargs)

    def add(self, is_copy=False, **kwargs):
        #self.__dict__.update(kwargs)

        if is_copy == False:
            for key, value in kwargs.items():
                setattr(self, key, value)
        else:
            for key, value in kwargs.items():
                try:
                    setattr(self, key, copy.deepcopy(value))
                    #setattr(self, key, value.copy())
                except Exception:
                    setattr(self, key, value)
    def __str__(self):
        return str(self.__dict__.keys())
    
#############################################################################

class LinearBn(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(LinearBn, self).__init__()
        self.linear = nn.Linear(in_channel, out_channel, bias=False)
        self.bn   = nn.BatchNorm1d(out_channel,eps=1e-05, momentum=0.5) #increase momentum
        self.act  = act

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        return x

class GraphConv(nn.Module):
    def __init__(self, node_dim, edge_dim ):
        super(GraphConv, self).__init__()
        self.node_dim = 128
        self.encoder = nn.Sequential(
            LinearBn(edge_dim, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 256),
            nn.ReLU(inplace=True),
            LinearBn(256, 128),
            nn.ReLU(inplace=True),
            LinearBn(128, self.node_dim * self.node_dim)
        )

        self.gru  = nn.GRU(self.node_dim, self.node_dim, batch_first=False, bidirectional=False)
        self.bias = nn.Parameter(torch.Tensor(self.node_dim))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.node_dim), 1.0 / math.sqrt(self.node_dim))


    def forward(self, node, edge_index, edge, hidden):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        edge_index = edge_index.t().contiguous()
        self.node_dim = node_dim

        #1. message :  m_j = SUM_i f(n_i, n_j, e_ij)  where i is neighbour(j)
        x_i     = torch.index_select(node, 0, edge_index[0])
        edge    = self.encoder(edge).view(-1,node_dim,node_dim)
        #message = x_i.view(-1,node_dim,1)*edge
        #message = message.sum(1)
        message = x_i.view(-1,1,node_dim)@edge
        message = message.view(-1,node_dim)
        message = scatter_('mean', message, edge_index[1], dim_size=num_node)
        message = F.relu(message +self.bias)

        #2. update: n_j = f(n_j, m_j)
        update = message

        #batch_first=True
        update, hidden = self.gru(update.view(1,-1,node_dim), hidden)
        update = update.view(-1,node_dim)

        return update, hidden

class Set2Set(torch.nn.Module):

    def softmax(self, x, index, num=None):
        x = x -  scatter_max(x, index, dim=0, dim_size=num)[0][index]
        x = x.exp()
        x = x / (scatter_add(x, index, dim=0, dim_size=num)[index] + 1e-16)
        return x

    def __init__(self, in_channel, processing_step=1):
        super(Set2Set, self).__init__()
        num_layer = 1

        self.processing_step = processing_step
        self.in_channel  = 128
        in_channel = self.in_channel
        self.out_channel = 2 * self.in_channel
        out_channel = self.out_channel
        self.num_layer   = num_layer
        self.lstm = torch.nn.LSTM(out_channel, in_channel, num_layer)
        self.lstm.reset_parameters()

    def forward(self, x, batch_index):
        batch_size = batch_index.max().item() + 1
        h = (x.new_zeros((self.num_layer, batch_size, self.in_channel)),
             x.new_zeros((self.num_layer, batch_size, self.in_channel)))
        q_star = x.new_zeros(batch_size, self.out_channel)
        for i in range(self.processing_step):
            q, h = self.lstm(q_star.unsqueeze(0), h)
            q = q.view(batch_size, -1)

            e = (x * q[batch_index]).sum(dim=-1, keepdim=True) #shape = num_node x 1
            a = self.softmax(e, batch_index, num=batch_size)   #shape = num_node x 1
            r = scatter_add(a * x, batch_index, dim=0, dim_size=batch_size) #apply attention #shape = batch_size x ...
            q_star = torch.cat([q, r], dim=-1)
        return q_star
    

#message passing
class Net(torch.nn.Module):
    def __init__(self, node_dim=13, edge_dim=5, num_target=1):
        super(Net, self).__init__()
        #very important hyper parameter - controls how many times messages are sent between the nodes
        self.num_propagate = NUM_PROPAGATE
        self.num_s2s = NUM_S2S
        #needs to be changed to the number of node features
        self.node_dim = node_dim
        
        self.preprocess = nn.Sequential(
            LinearBn(self.node_dim, 128),#128
            nn.ReLU(inplace=True),
            LinearBn(128, 128),
            nn.ReLU(inplace=True),
        )
        #140 x 13
        self.propagate = GraphConv(128, edge_dim)
        self.set2set = Set2Set(128, processing_step=self.num_s2s)

        #predict coupling constant
        self.predict = nn.Sequential(
            #[pool,node0,node1,secondLayerFeat] - pool is 128*2 in size, and node0/node1 are both 128 in size.  The secondLayerFeat 
            #are the features from the training set used to predict just the scalar coupling constant and are 44 in size.
            LinearBn(128*4+len(LGB_COLS), 1024),
            nn.ReLU(inplace=True),
            LinearBn( 1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1),
        )

    def forward(self, node, edge, edge_index, node_index, coupling_index,secondLayerFeat):
        num_node, node_dim = node.shape
        num_edge, edge_dim = edge.shape
        self.node_dim = node_dim
        node   = self.preprocess(node)
        hidden = node.view(1,num_node,-1)
        for i in range(self.num_propagate):
            node, hidden =  self.propagate(node, edge_index, edge, hidden)
        pool = self.set2set(node, node_index)

        #---
        num_coupling = len(coupling_index)
        coupling_atom0_index, coupling_atom1_index, coupling_type_index, coupling_batch_index = torch.split(coupling_index,1,dim=1)
        
        pool  = torch.index_select( pool, dim=0, index=coupling_batch_index.view(-1))
        node0 = torch.index_select( node, dim=0, index=coupling_atom0_index.view(-1))
        node1 = torch.index_select( node, dim=0, index=coupling_atom1_index.view(-1))
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        secondLayerFeat=secondLayerFeat[:,LGB_COLS].to(device)
        #backend CUDA
        predict = self.predict(torch.cat([pool,node0,node1,secondLayerFeat],-1))
        return predict.view(-1)

def criterion(predict, truth):
    predict = predict.view(-1)
    truth   = truth.view(-1)
    truth=(truth-targetMean)/targetStd
    assert(predict.shape==truth.shape)

    loss = torch.abs(predict-truth)
    loss = loss.mean()
    loss = torch.log(loss)
    return loss

class ChampsDataset(Dataset):
    def __init__(self,molecules, split, csv, mode, pickle_file, augment=None):
        self.split, self.csv, self.mode, self.augment = split, csv, mode, augment
        self.graphHolder = np.load(GRAPH_DIR + pickle_file, allow_pickle=True)
        if mode =='train': self.graphHolder = [self.graphHolder[i] for i in range(len(self.graphHolder)) if ALL_TRAIN_MOLECULES[i] in molecules]
        self.df = pd.read_csv(DATA_DIR + '%s.csv'%csv)
        self.df = self.df[self.df["molecule_name"].isin(molecules)]
        if split is not None: self.id = np.load(SPLIT_DIR + '%s'%split, allow_pickle=True)
        else: self.id = self.df.molecule_name.unique()

    def __len__(self):
        return len(self.id)
    
    def __getitem__(self, index):
        molecule_name = self.id[index]
        graph = copy.copy(self.graphHolder[index])
        assert(graph.molecule_name==molecule_name)

        ##filter only J link
        # 1JHC,     2JHC,     3JHC,     1JHN,     2JHN,     3JHN,     2JHH,     3JHH
        mask = np.zeros(len(graph.coupling.type),np.bool)
        for t in [TYPE_TO_TRAIN]:
            mask += (graph.coupling.type == COUPLING_TYPE.index(t))
        
        graph.coupling.id = graph.coupling.id [mask]
        graph.coupling.contribution = graph.coupling.contribution [mask]
        graph.coupling.index = graph.coupling.index [mask]
        graph.coupling.type = graph.coupling.type [mask]
        graph.coupling.value = graph.coupling.value [mask]
        graph.coupling.secondLayerFeat = graph.coupling.secondLayerFeat [mask]
        graph.node = np.concatenate(graph.node,-1)
        graph.node = graph.node[:,NODE_COLS]
        graph.edge = np.concatenate(graph.edge,-1)
        return graph

def null_collate(batch):
    batch_size = len(batch)
    node, edge, node_index, edge_index = [], [], [], []
    coupling_value,coupling_atom_index,coupling_type_index,coupling_batch_index = [],[],[],[]
    secondLayerFeat,infor=[],[]
    offset = 0
    for b in range(batch_size):
        graph = batch[b]
        num_node = len(graph.node)
        node.append(graph.node)
        edge.append(graph.edge)
        edge_index.append(graph.edge_index+offset)
        node_index.append(np.array([b]*num_node))

        num_coupling = len(graph.coupling.value)
        coupling_value.append(graph.coupling.value)
        coupling_atom_index.append(graph.coupling.index+offset)
        coupling_type_index.append (graph.coupling.type)
        coupling_batch_index.append(np.array([b]*num_coupling))
        secondLayerFeat.append(graph.coupling.secondLayerFeat)

        infor.append((graph.molecule_name, graph.smiles, graph.coupling.id))
        offset += num_node

    node = torch.from_numpy(np.concatenate(node)).float()
    edge = torch.from_numpy(np.concatenate(edge)).float()
    edge_index = torch.from_numpy(np.concatenate(edge_index).astype(np.int32)).long()
    node_index = torch.from_numpy(np.concatenate(node_index)).long()
    
    coupling_value = torch.from_numpy(np.concatenate(coupling_value)).float()
    coupling_index = np.concatenate([
        np.concatenate(coupling_atom_index),
        np.concatenate(coupling_type_index).reshape(-1,1),
        np.concatenate(coupling_batch_index).reshape(-1,1)
    ],-1)
    
    secondLayerFeat = torch.from_numpy(np.concatenate(secondLayerFeat)).float()
    
    coupling_index = torch.from_numpy(coupling_index).long()
    return node, edge, edge_index, node_index, coupling_value, coupling_index, infor, secondLayerFeat


# In[ ]:





# Make sure to change type and number from 3JHN and 36619

# In[ ]:


print('Loading data...')
initial_checkpoint = None
scheduler = NullScheduler(lr=STARTING_LR)
#scheduler = DecayScheduler(base_lr=STARTING_LR,decay=.5,step=25)
## setup  -----------------------------------------------------------------------------
batch_size = BATCH_SIZE 
train=pd.read_csv("../input/champs-scalar-coupling/train.csv")

ALL_TRAIN_MOLECULES = train["molecule_name"].unique()
mols=train.loc[train["type"]==TYPE_TO_TRAIN,"molecule_name"].unique()

kf = KFold(n_splits=N_FOLD, shuffle=True, random_state=KFOLD_SEED)
for ifold, (idx_tr, idx_va) in enumerate(kf.split(mols)):
    if ifold == FOLD_TO_TRAIN: break

train_dataset = ChampsDataset(molecules=mols[idx_tr],csv='train',mode ='train', pickle_file=GRAPH_TRAIN_PICKLE,split=None,augment=None)
train_loader  = DataLoader(train_dataset, sampler = RandomSampler(train_dataset),batch_size=batch_size,drop_last=True,num_workers=4,pin_memory=True,collate_fn=null_collate)
valid_dataset = ChampsDataset(molecules=mols[idx_va],csv='train',mode='train', pickle_file=GRAPH_TRAIN_PICKLE, split=None,augment=None)
valid_loader = DataLoader(valid_dataset,sampler=RandomSampler(valid_dataset),batch_size=batch_size,drop_last=False,num_workers=4,pin_memory=True,collate_fn=null_collate)

EDGE_DIM = np.sum([train_dataset.graphHolder[0].edge[j].shape[1] for j in range(len(train_dataset.graphHolder[0].edge))])
NODE_DIM = np.sum([train_dataset.graphHolder[0].node[j].shape[1] for j in range(len(train_dataset.graphHolder[0].node))])

assert(len(train_dataset)>=batch_size)
print('Loading done!')


# In[ ]:


## net ----------------------------------------
net = Net(node_dim=NODE_DIM,edge_dim=EDGE_DIM, num_target=NUM_TARGET).cuda()
if initial_checkpoint is not None:
    net.load_state_dict(torch.load(initial_checkpoint, map_location=lambda storage, loc: storage))
    
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),lr=STARTING_LR)

iter_accum  = 1
num_iters   = NUM_ITERS

iter_smooth = 50
iter_log    = 500
iter_valid  = 500
iter_save   = [0, num_iters-1]+ list(range(0, num_iters, 5000))#1*1000
start_iter = 0
start_epoch= 0
rate       = 0

train_loss   = np.zeros(20,np.float32)
valid_loss   = np.zeros(20,np.float32)
batch_loss   = np.zeros(20,np.float32)
iter = 0
i    = 0
start = timer()

bestLoss, bestDict, bestEpoch = 9999, 0, 0
print('Create Net done.')


# In[ ]:


targetMean=np.mean(train.loc[train["type"]==TYPE_TO_TRAIN,"scalar_coupling_constant"])
targetStd=np.std(train.loc[train["type"]==TYPE_TO_TRAIN,"scalar_coupling_constant"])


# In[ ]:


log = Logger()
log.open('log_train.txt',mode='a')
## start training here! ##############################################
log.write('** start training here! **\n')
log.write('   batch_size =%d,  iter_accum=%d\n'%(batch_size,iter_accum))
log.write('                      |--------------- VALID ----------------------------------------------------------------|-- TRAIN/BATCH ---------\n')
log.write('                      |std %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f    %4.1f   %4.1f  |                    |        | \n'%tuple(COUPLING_TYPE_STD))
log.write('rate     iter   epoch |    1JHC,   2JHC,   3JHC,   1JHN,   2JHN,   3JHN,   2JHH,   3JHH |  loss  mae log_mae | loss   | time          \n')
log.write('--------------------------------------------------------------------------------------------------------------------------------------\n')
rate=STARTING_LR
train_loss=5
while  iter<=num_iters:
    sum_train_loss = np.zeros(20,np.float32)
    sum = 0

    optimizer.zero_grad()
    for node, edge, edge_index, node_index, coupling_value, coupling_index, infor, secondLayerFeat in train_loader:

        batch_size = len(infor)
        iter  = i + start_iter
        epoch = (iter-start_iter)*batch_size/len(train_dataset) + start_epoch   

        if (iter % iter_valid==0):
            valid_loss = do_valid(net, valid_loader) #
            #reduceScheduler.step(valid_loss[8])
            if valid_loss[10]<bestLoss:
                bestEpoch=epoch
                bestLoss=valid_loss[10]
                bestDict=copy.deepcopy(net.state_dict())
        
        if (iter % iter_log==0):
            print('\r',end='',flush=True)
            asterisk = '*' if iter in iter_save else ' '
            log.write('%0.5f  %5.1f%s %5.1f |  %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f, %+0.3f | %+5.3f %5.2f %+0.2f | %+5.3f | %s' % (                             rate, iter/1000, asterisk, epoch, *valid_loss[:11], train_loss, time_to_str((timer() - start),'min')))
            log.write('\n')
            
        if iter in iter_save:
                    torch.save(net.state_dict(),'model.pth')
                    torch.save({
                        'optimizer': optimizer.state_dict(),
                        'iter'     : iter,
                        'epoch'    : epoch,
                    },'optimizer.pth')

        rate=scheduler.get_rate(epoch)
        adjust_learning_rate(optimizer, rate)
        
        net.train()  
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()
        coupling_value = coupling_value.cuda()
        coupling_index = coupling_index.cuda()

        predict = net(node, edge, edge_index, node_index, coupling_index, secondLayerFeat)
        loss = criterion(predict, coupling_value)
        
        (loss/iter_accum).backward()
        if (iter % iter_accum)==0:
            optimizer.step()
            optimizer.zero_grad()

        # print statistics  ------------
        batch_loss[:1] = [loss.item()]
        sum_train_loss += batch_loss
        sum += 1
        if iter%iter_smooth == 0:
            train_loss = loss
            sum_train_loss = np.zeros(20,np.float32)
            sum = 0

        i += 1
        pass  
    pass 


# In[ ]:


log.write("lr:"+str(scheduler.get_rate(epoch)))
print("bestLoss: "+str(bestLoss))
print("bestEpoch: "+str(bestEpoch))


# In[ ]:


net.load_state_dict(bestDict) 
test=pd.read_csv("../input/champs-scalar-coupling/test.csv")
#test_mols=test.loc[test["type"]==TYPE_TO_TRAIN,"molecule_name"].unique()
test_mols=test["molecule_name"].unique()
test_dataset = ChampsDataset(molecules=test_mols,csv='test',mode='test',pickle_file=GRAPH_TEST_PICKLE,split=None,augment=None,)
test_loader = DataLoader(test_dataset,sampler=SequentialSampler(test_dataset),batch_size=BATCH_SIZE,drop_last=False,num_workers=4,pin_memory=True,collate_fn=null_collate)

test_num = 0
test_predict = []
test_coupling_type  = []
test_coupling_value = []
test_id = []
test_loss = 0

start = timer()
for b, (node, edge, edge_index, node_index, coupling_value, coupling_index, infor,secondLayerFeat) in enumerate(test_loader):
    net.eval()
    with torch.no_grad():
        node = node.cuda()
        edge = edge.cuda()
        edge_index = edge_index.cuda()
        node_index = node_index.cuda()
        coupling_index = coupling_index.cuda()
        coupling_value = coupling_value.cuda()
        predict = net(node, edge, edge_index, node_index, coupling_index,secondLayerFeat)
        loss = criterion(predict, coupling_value)

        #---
    batch_size = len(infor)
    test_id.extend(list(np.concatenate([infor[b][2] for b in range(batch_size)])))
    test_predict.append(predict.data.cpu().numpy())
    test_coupling_type.append(coupling_index[:,2].data.cpu().numpy())
    test_coupling_value.append(coupling_value.data.cpu().numpy())
    test_loss += loss.item()*batch_size
    test_num += batch_size

    print('\r %8d/%8d     %0.2f  %s'%(
        test_num, len(test_dataset),test_num/len(test_dataset),
        time_to_str(timer()-start,'min')),end='',flush=True)
    pass  #-- end of one data loader --

assert(test_num == len(test_dataset))
print('\n')

id  = test_id
predict  = np.concatenate(test_predict)
predict=((predict*targetStd)+targetMean)
df = pd.DataFrame(list(zip(id, predict)), columns = ['id', 'scalar_coupling_constant'])

filename = "submission_" + TYPE_TO_TRAIN + '_batchsize' + str(BATCH_SIZE) + '_nbiters' + str(NUM_ITERS) + '_seed' + str(KFOLD_SEED) + '_fold' + str(FOLD_TO_TRAIN) + '_bestLoss' + str(np.round(bestLoss, 5))
df.to_csv(filename+".csv",index=False)

