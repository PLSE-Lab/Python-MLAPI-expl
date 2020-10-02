#!/usr/bin/env python
# coding: utf-8

# # Hazard modeling of russing plays

# The yardage gained on the rushing play in american football distributes to a one-dimenstional distribution which depends on rushing plays information. This notebook adopts **Cox proportional hazard model** to express this distribution and shows the advantage of suvival analysis approach to infer player's contribution on each play.

# # Introduction

# Survival analysis mainly focus on explaining the duration of time until some events happen. In survival analysis, our aim is to express the hazard function instead of probability density function. Let $y \in [-99, 99]$ be obtained yardage. Since the obtained yardage takes a discrete value, our aim is to estimate the discrete hazard function
# 
# $$
# \begin{aligned}
# h(y) = \mathrm{Pr}(y \leq Y < y + 1 \mid Y \geq y),
# \end{aligned}
# $$
# 
# which explains the probability of rusher being stopped at $y$.
# 
# In a rushing play, defensive players try to stop a rushing player, whereas offensive players prevent defensive players from stopping the rushing player. In terms of the hazard function, defensive players increase $h(r)$ and offensive players decrease $h(r)$. 
# 
# Our aim in this notebook is to estimate how each player affects on this hazard. It is not sufficient to evaluate players contribution of rushing play by simple stats like obtained yards, touchdowns and tuckles. Even one running back archieved highest rushing yards, it is unclear how other offensive players contribute to the rushing yards. 
# On the other hand, using hazard enables us to compare players' performance across positions similar to WPA used in baseball.

# # Model definition

# We adopt **Cox proportional model** to express the hazard function. Cox proportional model divides this hazard function into two parts as
# 
# $$
# \begin{aligned}
#     h(y) = h_0(y) \cdot \exp(\phi(x, y)), \quad y \in [-99, 99], \, \text{$x$ : covariates.}
# \end{aligned}
# $$
# 
# 
# The former part indicates how the hazard function depends on time and the latter part indicates how it depends on the covariate information.  Estimating the former part in empirical manner enables us to express the complex distribution easily.
# 
# Hense, we define $\phi(x, y)$ as the summation of player's contribution.
# $$
# \begin{aligned}
#     \phi(x, y) =  \phi_\mathrm{R}(x_\mathrm{R} \mid x) \, s_\mathrm{R}(y) + \sum_{i=1}^{10} \phi_\mathrm{O}(x^{(i)}_\mathrm{O} \mid x) \, s_\mathrm{O}(y) + \sum_{j=1}^{11} \phi_\mathrm{D}(x^{(j)}_\mathrm{D} \mid x) \, s_\mathrm{D}(y).
# \end{aligned}
# $$
# Here, $x_\mathrm{R}, \{x^{(i)}_\mathrm{O}\}_{i=1, \cdots, 10} \,\, ,  \{x^{(j)}_\mathrm{D}\}_{j=1, \cdots, 11} \, \, $ are player's information for rusher, offensive players and defensive players. $\phi_\mathrm{R}(\cdot), \phi_\mathrm{O}(\cdot), \phi_\mathrm{D}(\cdot)$ are scalar functions and $s_\mathrm{R}(\cdot), s_\mathrm{O}(\cdot), s_\mathrm{D}(\cdot)$ are temporal coefficients.
# 
# Each player's contribution to hazard  is also affected by nearby players. We express these interactions as graph expression. 
# 
# We firstly obtain Delaunay diagram from player's location those who directly involve rushing play (e.g. RB, OL, DL, LB). Then, omitting some edges which may not influence on the rusher, we construct a graph indicating pairs of block players and tackle players.
# Under this graph, we adopt **Gated Graph Neural Network** for expressing $\phi_\mathrm{R}(\cdot), \phi_\mathrm{O}(\cdot), \phi_\mathrm{D}(\cdot)$. 

# In[ ]:


from IPython.display import Image
Image("../input/gatedgnn/GatedGNN.png")


# # Setup

# In[ ]:


import numpy as np
from scipy.spatial import Delaunay
import pandas as pd
import tensorflow as tf
import os
from tqdm import tqdm_notebook as tqdm

import networkx as nx
import matplotlib.pyplot as plt


# ## Define function for etracting features from raw dataset 

# In[ ]:


def extract_feature(play, is_training=True):
    
    playDirection, fieldPosition, possessionTeam, yardLine = play.PlayDirection.iloc[0], play.FieldPosition.iloc[0], play.PossessionTeam.iloc[0], play.YardLine.iloc[0]
    homeTeamAbbr, visitorTeamAbbr = play.HomeTeamAbbr.iloc[0], play.VisitorTeamAbbr.iloc[0]
    nflIdRusher = play.NflIdRusher.iloc[0]

    if playDirection == 'right':
        direction = 1
    else:
        direction = -1

    home, away = play.Team.values == 'home', play.Team.values == 'away'
    isRusher = play.NflId.values == nflIdRusher

    if fieldPosition == possessionTeam:
        yardToEnd = 100 - yardLine
        start = np.array([120 + (yardLine + 10) * direction, 53.3 / 2]) % 120
    else:
        yardToEnd = yardLine
        start = np.array([120 - (yardLine + 10) * direction, 53.3 / 2]) % 120

    rad = np.nan_to_num(2 * np.pi * (90 - play.Dir.values) / 360)
    X, Y = play.X.values, play.Y.values
    S = play.Dis.values * np.logical_not(np.isnan(play.Dir.values)) / 0.1

    loc = np.vstack([(X - start[0]) * direction, (Y - start[1]) * direction]).T
    loc[:, 1] = loc[:, 1] - loc[isRusher, 1]
    vel = (S * np.vstack([np.cos(rad), np.sin(rad)])).T * direction
    
    x = np.hstack([loc, vel])
    inBox = np.array([position in ['T', 'C', 'G', 'OG', 'OT', 'RB', 'FB', 'HB', 'TE', 'DE', 'DL', 'DT', 'NT', 'ILB', 'LB', 'MLB', 'OLB'] for position in play.Position])

    if possessionTeam == homeTeamAbbr:

        x = np.vstack([x[isRusher], x[home * np.logical_not(isRusher)], x[away]])
        inBox = np.hstack([inBox[isRusher], inBox[home * np.logical_not(isRusher)], inBox[away]])

    elif possessionTeam == visitorTeamAbbr:

        x = np.vstack([x[isRusher], x[away * np.logical_not(isRusher)], x[home]])
        inBox = np.hstack([inBox[isRusher], inBox[away * np.logical_not(isRusher)], inBox[home]])
      
    inBox[0] = True
    inBox = inBox * (x[0, 0] <= x[:, 0])
    
    locInBox = x[inBox, :2]
    ind = np.arange(22)[inBox]
    
    try:
        delau = Delaunay(locInBox)
        adj = np.zeros((22, 22))

        for i in range(delau.simplices.shape[0]):
            for j in delau.simplices[i]:
                adj[ind[j], ind[delau.simplices[i]]] = 1

        adj *= 1 - np.eye(22)

        loc = x[:, :2]
        D = loc[:, np.newaxis] - loc
        dist = np.linalg.norm(D, axis=2)

        adj[:1, 11:], adj[11:, :1], adj[1:11, 1:11], adj[11:, 11:] = 0, 0, 0, 0
        adj[1:11, 11:] = adj[1:11, 11:] * ((D[1:11, 11:] * D[1:11, :1]).sum(2) < 0)
        adj[11:, 1:11] = adj[1:11, 11:].T
        
    except:
        adj = np.zeros((22, 22))
    
    offset = - 99
    threshold = np.minimum(np.floor(loc[:, 0].max()), yardToEnd) - offset
    
    other = np.stack([(play.Season.values == 2017).astype(np.float)]).T
    x = np.hstack([other, x])

    if is_training:

        yard = play.Yards.iloc[0] - offset
        
        c = yard < threshold
        y = np.minimum(yard, threshold)

        return x, adj, y, c, yardToEnd

    else:
        return x, adj, yardToEnd


# # Load data

# In[ ]:


data = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', low_memory=False)

data.loc[data.VisitorTeamAbbr == "ARI", 'VisitorTeamAbbr'] = "ARZ"
data.loc[data.HomeTeamAbbr == "ARI", 'HomeTeamAbbr'] = "ARZ"

data.loc[data.VisitorTeamAbbr == "BAL", 'VisitorTeamAbbr'] = "BLT"
data.loc[data.HomeTeamAbbr == "BAL", 'HomeTeamAbbr'] = "BLT"

data.loc[data.VisitorTeamAbbr == "CLE", 'VisitorTeamAbbr'] = "CLV"
data.loc[data.HomeTeamAbbr == "CLE", 'HomeTeamAbbr'] = "CLV"

data.loc[data.VisitorTeamAbbr == "HOU", 'VisitorTeamAbbr'] = "HST"
data.loc[data.HomeTeamAbbr == "HOU", 'HomeTeamAbbr'] = "HST"


# # Extract features from training dataset

# In[ ]:


inds = list(data[data.Season != 2019].groupby('PlayId').groups.values())
n = len(inds)

inds_test = list(data[data.Season == 2019].groupby('PlayId').groups.values())
n_test = len(inds_test)

yards_index = np.arange(-99, 100)


# In[ ]:


xs, adjs, ys, cs, hs = [], [], [], [], []

for i in tqdm(range(n)):

    play = data.loc[inds[i]]
    x, adj, y, c, yardToEnd = extract_feature(play)
    h = (yards_index >= play['Yards'].iloc[0]).astype(np.float)

    xs.append(x)
    ys.append(y)
    cs.append(c)
    hs.append(h)
    adjs.append(adj)

xs, adjs, ys, cs, hs = np.stack(xs), np.stack(adjs), np.hstack(ys), np.array(cs).astype(np.int), np.vstack(hs)

# Flip Y
xs, adjs, ys, cs, hs = np.vstack([xs, xs * np.array([1, 1, -1, 1, -1])]), np.vstack([adjs, adjs]), np.hstack([ys, ys]), np.hstack([cs, cs]), np.vstack([hs, hs])
n *= 2

xs, adjs, ys, cs, hs = xs[np.argsort(ys)], adjs[np.argsort(ys)], np.sort(ys), cs[np.argsort(ys)], hs[np.argsort(ys)]


# In[ ]:


ys_unique, ys_index, ys_inverse, ys_count = np.unique(ys, return_index=True, return_inverse=True, return_counts=True)
ys_mask = tf.constant(np.arange(199)[:, np.newaxis] == ys_unique, dtype=tf.float32)

cs_count = []
for j in ys_unique:
    cs_count.append(cs[ys == j].sum())

cs_count = np.array(cs_count)

cs_mask = np.zeros((n, ys_unique.shape[0]))
for j, index, count in zip(range(ys_unique.shape[0]), ys_index, ys_count):
    cs_mask[index:index+count, j] = 1.
cs_mask = tf.constant(cs_mask, dtype=tf.float32)

mask = tf.constant(ys_index <= np.arange(n)[:, np.newaxis], dtype=tf.float32)
inf_array = - tf.ones_like(mask, dtype=tf.float32) * np.inf

ys_succ_ind = [j in ys_unique for j in np.arange(ys_unique.min(), ys_unique.max()+1)]
ys_succ_ind = np.arange(len(ys_succ_ind))[ys_succ_ind]
m = np.int(ys_unique[-1] - ys_unique[0] + 1)


# In[ ]:


xs_test, adjs_test, ys_test, cs_test, hs_test = [], [], [], [], []
yardToEnds_test = []

for i in tqdm(range(n_test)):

    play = data.loc[inds_test[i]]
    x, adj, y, c, yardToEnd = extract_feature(play)
    h = (yards_index >= play['Yards'].iloc[0]).astype(np.float)

    xs_test.append(x)
    ys_test.append(y)
    cs_test.append(c)
    hs_test.append(h)
    adjs_test.append(adj)
    
    yardToEnds_test.append(yardToEnd)

xs_test, adjs_test, ys_test, cs_test, hs_test = np.stack(xs_test), np.stack(adjs_test), np.hstack(ys_test), np.array(cs_test).astype(np.int), np.vstack(hs_test)
yardToEnds_test = np.hstack([yardToEnds_test])


# # Define Cox proportional model

# In[ ]:


class GGNN(tf.keras.Model):

    def __init__(self):

        super(GGNN, self).__init__()

        self.gru_L = 3

        self.denseGRU_R, self.denseGRU_O, self.denseGRU_D = tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.tanh), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.tanh), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.tanh)

        self.update_R, self.update_O, self.update_D = tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.sigmoid, use_bias=False), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.sigmoid, use_bias=False), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.sigmoid, use_bias=False)
        self.reset_R, self.reset_O, self.reset_D = tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.sigmoid, use_bias=False), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.sigmoid, use_bias=False), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.sigmoid, use_bias=False)
        self.modify_R, self.modify_O, self.modify_D = tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.tanh, use_bias=True), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.tanh, use_bias=True), tf.keras.layers.Dense(n_layerGRU, activation=tf.nn.tanh, use_bias=True)
        
        self.dropoutGRU_R, self.dropoutGRU_O, self.dropoutGRU_D = tf.keras.layers.Dropout(dropout_rate), tf.keras.layers.Dropout(dropout_rate), tf.keras.layers.Dropout(dropout_rate)
        self.dropoutGRU_neighbor_R, self.dropoutGRU_neighbor_O, self.dropoutGRU_neighbor_D = tf.keras.layers.Dropout(dropout_rate), tf.keras.layers.Dropout(dropout_rate), tf.keras.layers.Dropout(dropout_rate)
        
        self.dense1_R, self.dense1_O, self.dense1_D = tf.keras.layers.Dense(n_layer1, activation=tf.nn.tanh), tf.keras.layers.Dense(n_layer1, activation=tf.nn.tanh), tf.keras.layers.Dense(n_layer1, activation=tf.nn.tanh)
        self.dropout1_R, self.dropout1_O, self.dropout1_D = tf.keras.layers.Dropout(dropout_rate), tf.keras.layers.Dropout(dropout_rate), tf.keras.layers.Dropout(dropout_rate)
        
        self.dense2_R, self.dense2_O, self.dense2_D = tf.keras.layers.Dense(1, use_bias=False), tf.keras.layers.Dense(1, use_bias=False), tf.keras.layers.Dense(1, use_bias=False)
      
        self.temporal_R, self.temporal_O, self.temporal_D = tf.Variable(tf.zeros(m)), tf.Variable(tf.zeros(m)), tf.Variable(tf.zeros(m))
        self.forward_R, self.forward_O, self.forward_D = tf.keras.layers.GRU(1, return_sequences=True), tf.keras.layers.GRU(1, return_sequences=True), tf.keras.layers.GRU(1, return_sequences=True)
        self.backward_R, self.backward_O, self.backward_D = tf.keras.layers.GRU(1, return_sequences=True, go_backwards=True), tf.keras.layers.GRU(1, return_sequences=True, go_backwards=True), tf.keras.layers.GRU(1, return_sequences=True, go_backwards=True)
        self.output_R, self.output_O, self.output_D = tf.keras.layers.Dense(1, use_bias=False), tf.keras.layers.Dense(1, use_bias=False), tf.keras.layers.Dense(1, use_bias=False)
        
    @tf.function  
    def call(self, X, A, is_training=False):
        
        X_R = tf.slice(X, [0, 0, 0], [-1, 1, -1])
        X_O = tf.concat([tf.tile(X_R, (1, 10, 1)), tf.slice(X, [0, 1, n_down_info], [-1, 10, -1])], axis=2)
        X_D = tf.concat([tf.tile(X_R, (1, 11, 1)), tf.slice(X, [0, 11, n_down_info], [-1, 11, -1])], axis=2)
        other = tf.squeeze(tf.slice(X, [0, 0, 0], [-1, 1, n_down_info]), axis=1)

        A_R = tf.slice(A, [0, 0, 1], [-1, 1, -1])
        A_O = tf.concat([tf.slice(A, [0, 1, 0], [-1, 10, 1]), tf.slice(A, [0, 1, 11], [-1, 10, -1])], 2)
        A_D = tf.slice(A, [0, 11, 0], [-1, -1, -1])
        
        layerGRU_R, layerGRU_O, layerGRU_D = self.denseGRU_R(X_R), self.denseGRU_O(X_O), self.denseGRU_D(X_D)
        
        maskGRU_update_R, maskGRU_update_O, maskGRU_update_D = self.dropoutGRU_R(tf.ones_like(layerGRU_R), is_training), self.dropoutGRU_O(tf.ones_like(layerGRU_O), is_training), self.dropoutGRU_D(tf.ones_like(layerGRU_D), is_training)
        maskGRU_reset_R, maskGRU_reset_O, maskGRU_reset_D = self.dropoutGRU_R(tf.ones_like(layerGRU_R), is_training), self.dropoutGRU_O(tf.ones_like(layerGRU_O), is_training), self.dropoutGRU_D(tf.ones_like(layerGRU_D), is_training)
        maskGRU_modify_R, maskGRU_modify_O, maskGRU_modify_D = self.dropoutGRU_R(tf.ones_like(layerGRU_R), is_training), self.dropoutGRU_O(tf.ones_like(layerGRU_O), is_training), self.dropoutGRU_D(tf.ones_like(layerGRU_D), is_training)
        
        maskGRU_update_neighbor_R, maskGRU_update_neighbor_O, maskGRU_update_neighbor_D = self.dropoutGRU_neighbor_R(tf.ones(tf.shape(layerGRU_R)), is_training), self.dropoutGRU_neighbor_O(tf.ones(tf.shape(layerGRU_O) * (1, 1, 2)), is_training), self.dropoutGRU_neighbor_D(tf.ones(tf.shape(layerGRU_D)), is_training)
        maskGRU_reset_neighbor_R, maskGRU_reset_neighbor_O, maskGRU_reset_neighbor_D = self.dropoutGRU_neighbor_R(tf.ones(tf.shape(layerGRU_R)), is_training), self.dropoutGRU_neighbor_O(tf.ones(tf.shape(layerGRU_O) * (1, 1, 2)), is_training), self.dropoutGRU_neighbor_D(tf.ones(tf.shape(layerGRU_D)), is_training)
        maskGRU_modify_neighbor_R, maskGRU_modify_neighbor_O, maskGRU_modify_neighbor_D = self.dropoutGRU_neighbor_R(tf.ones(tf.shape(layerGRU_R)), is_training), self.dropoutGRU_neighbor_O(tf.ones(tf.shape(layerGRU_O) * (1, 1, 2)), is_training), self.dropoutGRU_neighbor_D(tf.ones(tf.shape(layerGRU_D)), is_training)

        for l in range(self.gru_L):

            layerGRU_neighbor_R = tf.matmul(tf.slice(A_R, [0, 0, 0], [-1, -1, 10]), layerGRU_O)
            layerGRU_neighbor_O = tf.concat([tf.matmul(tf.slice(A_O, [0, 0, 0], [-1, -1, 1]), layerGRU_R), tf.matmul(tf.slice(A_O, [0, 0, 1], [-1, -1, -1]), layerGRU_D)], 2)
            layerGRU_neighbor_D = tf.matmul(tf.slice(A_D, [0, 0, 1], [-1, -1, 10]), layerGRU_O)
            
            z_R = self.update_R(tf.concat([layerGRU_R * maskGRU_update_R, layerGRU_neighbor_R * maskGRU_update_neighbor_R], 2))
            r_R = self.reset_R(tf.concat([layerGRU_R * maskGRU_reset_R, layerGRU_neighbor_R * maskGRU_reset_neighbor_R], 2))
            layerGRU_modified_R = self.modify_R(tf.concat([layerGRU_R * r_R * maskGRU_modify_R, layerGRU_neighbor_R * maskGRU_modify_neighbor_R], 2))

            z_O = self.update_O(tf.concat([layerGRU_O * maskGRU_update_O, layerGRU_neighbor_O * maskGRU_update_neighbor_O], 2))
            r_O = self.reset_O(tf.concat([layerGRU_O * maskGRU_reset_O, layerGRU_neighbor_O * maskGRU_reset_neighbor_O], 2))
            layerGRU_modified_O = self.modify_O(tf.concat([layerGRU_O * r_O * maskGRU_modify_O, layerGRU_neighbor_O * maskGRU_modify_neighbor_O], 2))

            z_D = self.update_D(tf.concat([layerGRU_D * maskGRU_update_D, layerGRU_neighbor_D * maskGRU_update_neighbor_D], 2))
            r_D = self.reset_D(tf.concat([layerGRU_D * maskGRU_reset_D, layerGRU_neighbor_D * maskGRU_reset_neighbor_D], 2))
            layerGRU_modified_D = self.modify_D(tf.concat([layerGRU_D * r_D * maskGRU_modify_D, layerGRU_neighbor_D * maskGRU_modify_neighbor_D], 2))

            layerGRU_R = (1. - z_R) * layerGRU_R + z_R * layerGRU_modified_R
            layerGRU_O = (1. - z_O) * layerGRU_O + z_O * layerGRU_modified_O
            layerGRU_D = (1. - z_D) * layerGRU_D + z_D * layerGRU_modified_D
            
        layer1_R, layer1_O, layer1_D = self.dense1_R(layerGRU_R), self.dense1_O(layerGRU_O), self.dense1_D(layerGRU_D)
        layer1_R, layer1_O, layer1_D = self.dropout1_R(layer1_R), self.dropout1_O(layer1_O), self.dropout1_D(layer1_D)
        
        layer2_R, layer2_O, layer2_D = self.dense2_R(layer1_R), self.dense2_O(layer1_O), self.dense2_D(layer1_D)
        
        temporal_R = tf.transpose(tf.gather_nd(tf.nn.softmax(tf.transpose(tf.squeeze(self.output_R(self.backward_R(self.forward_R(self.temporal_R[tf.newaxis, :, tf.newaxis]))), 2)), 0), ys_succ_ind[np.newaxis].T))
        temporal_O = tf.transpose(tf.gather_nd(tf.nn.softmax(tf.transpose(tf.squeeze(self.output_O(self.backward_O(self.forward_O(self.temporal_O[tf.newaxis, :, tf.newaxis]))), 2)), 0), ys_succ_ind[np.newaxis].T))
        temporal_D = tf.transpose(tf.gather_nd(tf.nn.softmax(tf.transpose(tf.squeeze(self.output_D(self.backward_D(self.forward_D(self.temporal_D[tf.newaxis, :, tf.newaxis]))), 2)), 0), ys_succ_ind[np.newaxis].T))
        
        out = tf.squeeze(layer2_R, 1) * temporal_R + tf.reduce_sum(layer2_O, 1) * temporal_O + tf.reduce_sum(layer2_D, 1) * temporal_D
        
        return out
    
    def call_players(self, X, A):
        
        X_R = tf.slice(X, [0, 0, 0], [-1, 1, -1])
        X_O = tf.concat([tf.tile(X_R, (1, 10, 1)), tf.slice(X, [0, 1, n_down_info], [-1, 10, -1])], axis=2)
        X_D = tf.concat([tf.tile(X_R, (1, 11, 1)), tf.slice(X, [0, 11, n_down_info], [-1, 11, -1])], axis=2)
        other = tf.squeeze(tf.slice(X, [0, 0, 0], [-1, 1, n_down_info]), axis=1)

        A_R = tf.slice(A, [0, 0, 1], [-1, 1, -1])
        A_O = tf.concat([tf.slice(A, [0, 1, 0], [-1, 10, 1]), tf.slice(A, [0, 1, 11], [-1, 10, -1])], 2)
        A_D = tf.slice(A, [0, 11, 0], [-1, -1, -1])
        
        layerGRU_R, layerGRU_O, layerGRU_D = self.denseGRU_R(X_R), self.denseGRU_O(X_O), self.denseGRU_D(X_D)
        
        for l in range(self.gru_L):

            layerGRU_neighbor_R = tf.matmul(tf.slice(A_R, [0, 0, 0], [-1, -1, 10]), layerGRU_O)
            layerGRU_neighbor_O = tf.concat([tf.matmul(tf.slice(A_O, [0, 0, 0], [-1, -1, 1]), layerGRU_R), tf.matmul(tf.slice(A_O, [0, 0, 1], [-1, -1, -1]), layerGRU_D)], 2)
            layerGRU_neighbor_D = tf.matmul(tf.slice(A_D, [0, 0, 1], [-1, -1, 10]), layerGRU_O)
            
            z_R = self.update_R(tf.concat([layerGRU_R, layerGRU_neighbor_R], 2))
            r_R = self.reset_R(tf.concat([layerGRU_R, layerGRU_neighbor_R], 2))
            layerGRU_modified_R = self.modify_R(tf.concat([layerGRU_R * r_R, layerGRU_neighbor_R], 2))

            z_O = self.update_O(tf.concat([layerGRU_O, layerGRU_neighbor_O], 2))
            r_O = self.reset_O(tf.concat([layerGRU_O, layerGRU_neighbor_O], 2))
            layerGRU_modified_O = self.modify_O(tf.concat([layerGRU_O * r_O, layerGRU_neighbor_O], 2))

            z_D = self.update_D(tf.concat([layerGRU_D, layerGRU_neighbor_D], 2))
            r_D = self.reset_D(tf.concat([layerGRU_D, layerGRU_neighbor_D], 2))
            layerGRU_modified_D = self.modify_D(tf.concat([layerGRU_D * r_D, layerGRU_neighbor_D], 2))

            layerGRU_R = (1. - z_R) * layerGRU_R + z_R * layerGRU_modified_R
            layerGRU_O = (1. - z_O) * layerGRU_O + z_O * layerGRU_modified_O
            layerGRU_D = (1. - z_D) * layerGRU_D + z_D * layerGRU_modified_D
            
        layer1_R, layer1_O, layer1_D = self.dense1_R(layerGRU_R), self.dense1_O(layerGRU_O), self.dense1_D(layerGRU_D)
        layer2_R, layer2_O, layer2_D = self.dense2_R(layer1_R), self.dense2_O(layer1_O), self.dense2_D(layer1_D)
        
        temporal_R = tf.nn.softmax(tf.squeeze(self.output_R(self.backward_R(self.forward_R(self.temporal_R[tf.newaxis, :, tf.newaxis]))), 2), 1)
        temporal_O = tf.nn.softmax(tf.squeeze(self.output_O(self.backward_O(self.forward_O(self.temporal_O[tf.newaxis, :, tf.newaxis]))), 2), 1)
        temporal_D = tf.nn.softmax(tf.squeeze(self.output_D(self.backward_D(self.forward_D(self.temporal_D[tf.newaxis, :, tf.newaxis]))), 2), 1)
        
        return tf.concat([layer2_R * temporal_R, layer2_O * temporal_O, layer2_D * temporal_D], axis=1)
    
@tf.function
def compute_cost(model, X, A):

    out = model.call(X, A, True)
    
    out_max = tf.reduce_max(tf.where(tf.cast(mask, tf.bool), out, inf_array), 0)    
    exp_sum = tf.reduce_sum(tf.exp(out - out_max) * mask, 0)
    den = (out_max + tf.math.log(exp_sum)) * cs_count

    cost = - tf.reduce_sum(tf.reduce_sum(out * cs_mask, 1) * cs) + tf.reduce_sum(den)
    
    return cost

@tf.function
def compute_gradients(model, X, A):

    with tf.GradientTape() as tape:
        cost = compute_cost(model, X, A)

    return tape.gradient(cost, model.trainable_variables), cost

@tf.function
def apply_gradients(optimizer, gradients, variables):

    optimizer.apply_gradients(zip(gradients, variables))


def compute_baseline_hazard(model, X, A):

    out = model.call(X, A, False)

    out_max = tf.reduce_max(tf.where(tf.cast(mask, tf.bool), out, inf_array), 0)    
    exp_sum = tf.reduce_sum(tf.exp(out - out_max) * mask, 0)
    den = (out_max + tf.math.log(exp_sum)) * cs_count

    baseline_hazard = np.sum(ys_mask * tf.exp(- out_max) * exp_sum.numpy() ** -1 * cs_count, axis=1)

    return baseline_hazard


def compute_hazard_ratio(model, X, A):

    out = model.call(X, A, False)
    hazard_ratio = np.dot(tf.exp(out).numpy(), tf.transpose(ys_mask))

    return hazard_ratio


# # Estimate Cox proportional model

# In[ ]:


X, A = tf.constant(xs, dtype=tf.float32), tf.constant(adjs, dtype=tf.float32)
X_test, A_test = tf.constant(xs_test, dtype=tf.float32), tf.constant(adjs_test, dtype=tf.float32)


# In[ ]:


learning_rate = 0.01
dropout_rate = 0.5
n_layerGRU = 64
n_layer1 = 16
n_down_info, n_player_info = 1, 4
n_ties = ys_unique.shape[0]


# In[ ]:


model = GGNN()
optimizer = tf.keras.optimizers.Adam(learning_rate)

traning_epochs = 500
best_CRPS = np.inf

for epoch in range(traning_epochs):
    
    gradients, cost_epoch = compute_gradients(model, X, A)
    apply_gradients(optimizer, gradients, model.variables)

    if epoch % 50 == 0:

        print('epoch ' + str(epoch) + ': ' + str(cost_epoch.numpy()))

        baseline_hazard = compute_baseline_hazard(model, X, A)
        hazard_ratio = compute_hazard_ratio(model, X_test, A_test)

        preds = 1 - np.exp(- np.cumsum(baseline_hazard * hazard_ratio, 1))
        preds[yards_index > yardToEnds_test[:, np.newaxis]] = 0.
        preds = preds / preds.max(1)[:, np.newaxis]
        preds[yards_index >= yardToEnds_test[:, np.newaxis]] = 1.
        CRPS = np.square(hs_test - preds).mean()

        print('test CRPS: ' + str(CRPS))
        
        if CRPS <= best_CRPS:
            model.save_weights('best_model')
            best_CRPS = CRPS
        
print('------------------------')
model.load_weights('best_model')


# # Player's contribution on each play

# In[ ]:


def draw_play(play):

    x, adj,  yardToEnd = extract_feature(play, False)
    out = model.call_players(tf.constant(x[np.newaxis], dtype=tf.float32), tf.constant(adj[np.newaxis], dtype=tf.float32))
    scale_R, scale_O, scale_D = temporal_R[0], temporal_O[0], temporal_D[0]
    
    playDirection, fieldPosition, possessionTeam, yardLine = play.PlayDirection.iloc[0], play.FieldPosition.iloc[0], play.PossessionTeam.iloc[0], play.YardLine.iloc[0]
    homeTeamAbbr, visitorTeamAbbr = play.HomeTeamAbbr.iloc[0], play.VisitorTeamAbbr.iloc[0]
    nflIdRusher = play.NflIdRusher.iloc[0]

    home, away = play.Team.values == 'home', play.Team.values == 'away'
    isRusher = play.NflId.values == nflIdRusher

    position = play.Position.values

    if possessionTeam == homeTeamAbbr:
        position = np.hstack([position[isRusher], position[home * np.logical_not(isRusher)], position[away]])

    elif possessionTeam == visitorTeamAbbr:
        position = np.hstack([position[isRusher], position[away * np.logical_not(isRusher)], position[home]])

    loc, vel = x[:, n_down_info:n_down_info+2], x[:, n_down_info+2:n_down_info+4]

    G = nx.Graph()

    G.add_nodes_from(np.arange(11), bipartite=0)
    G.add_nodes_from(np.arange(11, 22), bipartite=1)
    node_color = ['r']
    node_color.extend(['b' for i in range(10)])
    node_color.extend(['g' for i in range(11)])

    row, col = np.where(adj != 0)
    G.add_edges_from(zip(row, col))

    plt.figure(figsize=(18, 18))

    nx.draw_networkx_nodes(G, loc, node_color=node_color, node_size=1000, alpha=1.)
    nx.draw_networkx_edges(G, loc, alpha=0.5, style='dashed', edge_color='k')
    nx.draw_networkx_labels(G, loc, {i: position[i] for i in range(22)}, font_weight='bold', font_color='white')

    for i in range(22):
        plt.arrow(loc[i, 0], loc[i, 1], vel[i, 0] / 2. + 0.01, vel[i, 1] / 2. + 0.01, width=0.01,head_width=0.1,head_length=0.1,length_includes_head=True, color='k', alpha=0.4)

    for i in range(22):
        if (-8 < loc[i, 0] < 8) & (-8 < loc[i, 1] < 8):
            if i == 0:
                score = (out[0, i].numpy().sum() - out_players_mean[i]) * scale_R
            elif i > 10:
                score = (out[0, i].numpy().sum() - out_players_mean[i]) * scale_D
            else:
                score = (out[0, i].numpy().sum() - out_players_mean[i]) * scale_O
                
            plt.text(loc[i, 0]+0.3, loc[i, 1]+0.2, np.around(np.exp(score), 2))
            
    plt.text(-7.5, 7, "Offense:", fontsize=30)
    score_R = (tf.reduce_sum(out[:, :1]) - tf.reduce_sum(out_players_mean[:1])).numpy() * scale_R
    score_O = (tf.reduce_sum(out[:, 1:11]) - tf.reduce_sum(out_players_mean[1:11])).numpy() * scale_O
    plt.text(-5, 7, np.around(np.exp(score_R + score_O), 2), fontsize=30)

    plt.text(-7.5, 6, "Defense:", fontsize=30)
    score_D = (tf.reduce_sum(out[:, 11:]) - tf.reduce_sum(out_players_mean[11:])).numpy() * scale_D
    plt.text(-5, 6, np.around(np.exp(score_D), 2), fontsize=30)

    plt.vlines(0, -8, 8, linestyle='solid', alpha=0.2)
    plt.vlines(play.Down.iloc[0], -8, 8, color='goldenrod', linestyle='solid', alpha=0.5)

    plt.xlim(-8, 8)
    plt.ylim(-8, 8)


# In[ ]:


out_players = model.call_players(X_test, A_test)

out_rusher_mean = tf.reduce_mean(tf.reduce_sum(out_players[:, :1], 2, keepdims=True))
out_offense_mean = tf.reduce_mean(tf.reduce_sum(out_players[:, 1:11], 2, keepdims=True))
out_defense_mean = tf.reduce_mean(tf.reduce_sum(out_players[:, 11:22], 2, keepdims=True))
out_players_mean = tf.concat([out_rusher_mean * tf.ones(1), out_offense_mean * tf.ones(10), out_defense_mean * tf.ones(11)], axis=0)

temporal_R = pd.Series((out_players[0, 0] / tf.reduce_sum(out_players[0, 0])).numpy(), index=np.arange(ys_unique[0]-99, ys_unique[-1]-98))
temporal_O = pd.Series((out_players[0, 1] / tf.reduce_sum(out_players[0, 1])).numpy(), index=np.arange(ys_unique[0]-99, ys_unique[-1]-98))
temporal_D = pd.Series((out_players[0, 11] / tf.reduce_sum(out_players[0, 11])).numpy(), index=np.arange(ys_unique[0]-99, ys_unique[-1]-98))

scale_R, scale_O, scale_D = temporal_R[0], temporal_O[0], temporal_D[0]


# Each value indicates hazard ratio at 0 yard relative to mean hazard of 2019 Season. 
# For example, if a defensive player have 1.5, then the hazard at 0 yard on this play is 1.5 times larger than the mean hazard due to this player.

# In[ ]:


l = 75
play = data.loc[inds[l]]
print('Gain yard: ' + str(play.Yards.iloc[0]))
draw_play(play)


# # Player's contribution in 2019 Season

# In[ ]:


score = (tf.reduce_sum(out_players, 2) - out_players_mean).numpy() 
scale = np.hstack([scale_R, scale_O * np.ones(10), scale_D * np.ones(11)])
score *= scale


# In[ ]:


positions, names = [], []
offense_teams, defense_teams = [], []

for l in tqdm(range(len(inds_test))):
    
    play = data.loc[inds_test[l]]

    possessionTeam = play.PossessionTeam.iloc[0]
    homeTeamAbbr, visitorTeamAbbr = play.HomeTeamAbbr.iloc[0], play.VisitorTeamAbbr.iloc[0]
    nflIdRusher = play.NflIdRusher.iloc[0]

    home, away = play.Team.values == 'home', play.Team.values == 'away'
    isRusher = play.NflId.values == nflIdRusher

    position, displayName = play.Position.values, play.DisplayName.values

    if possessionTeam == homeTeamAbbr:
        position = np.hstack([position[isRusher], position[home * np.logical_not(isRusher)], position[away]])
        displayName = np.hstack([displayName[isRusher], displayName[home * np.logical_not(isRusher)], displayName[away]])
        
    elif possessionTeam == visitorTeamAbbr:
        position = np.hstack([position[isRusher], position[away * np.logical_not(isRusher)], position[home]])
        displayName = np.hstack([displayName[isRusher], displayName[away * np.logical_not(isRusher)], displayName[home]])
    
    position_replaced = []

    for pos in position:

        if pos in ['QB']:
            position_replaced.append('QB')
        elif pos in  ['C', 'T', 'OT', 'G', 'OG', 'TE']:
            position_replaced.append('OL')
        elif pos in  ['RB', 'HB', 'FB']:
            position_replaced.append('RB')
        elif pos in  ['WR']:
            position_replaced.append('WR')

        elif pos in ['DE', 'DT', 'DL', 'NT']:
            position_replaced.append('DL')
        elif pos in ['LB', 'ILB', 'OLB', 'MLB']:
            position_replaced.append('LB')
        elif pos in ['DB', 'CB', 'FS', 'SS', 'S', 'SAF']:
            position_replaced.append('DB')
    
    position_replaced = np.array(position_replaced)
            
    names.append(displayName)
    positions.append(position_replaced)
    
    if possessionTeam == homeTeamAbbr:
        offense_teams.append(homeTeamAbbr)
        defense_teams.append(visitorTeamAbbr)
    else:
        offense_teams.append(visitorTeamAbbr)
        defense_teams.append(homeTeamAbbr)
        
    
names = np.vstack(names)
positions = np.vstack(positions)
offense_teams = np.hstack(offense_teams)
defense_teams = np.hstack(defense_teams)


# In[ ]:


def calculate_score_players(position):
    
    threshold = 100
    
    if position in ['RB', 'QB', 'WR']:
        ind = np.arange(1)
    
    elif position == 'FB':
        ind = np.arange(1, 11)
        position = 'RB'
        
    elif position in ['OL']:
        ind = np.arange(1, 11)
        
    elif position in ['DL', 'LB', 'DB'] :
        ind = np.arange(11, 22)
         
    players = list(set(names[:, ind][positions[:, ind] == position].flatten().tolist()))

    score_players = {}
    for player in players:
        if (names[:, ind] == player).sum() > threshold:
            score_players[player] = score[:, ind][names[:, ind] == player].mean()

    score_players = pd.Series(score_players)
    score_players = score_players.sort_values()
    
    return score_players


def calculate_score_teams():
    
    teams = list(set(offense_teams))

    score_offense = {}
    score_defense = {}
    
    for team in teams:
        score_offense[team] = score[offense_teams == team].mean()
        score_defense[team] = score[defense_teams == team].mean()

    score_offense = pd.Series(score_offense)
    score_offense = score_offense.sort_values()
    
    score_defense = pd.Series(score_defense)
    score_defense = score_defense.sort_values()
    
    return score_offense, score_defense


# In[ ]:


score_offense, score_defense = calculate_score_teams()
score_OL = calculate_score_players('OL')
score_LB = calculate_score_players('LB')


# Below score is the mean value of log hazard ratio of defensive players at 0 yard in 2019 season.
# Larger value indicates the better rushing defense.

# In[ ]:


score_defense[::-1]


# Below score is the mean value of log hazard ratio of offensive players at 0 yard in 2019 season.
# Smaller value indicates the better rushing offense.

# In[ ]:


score_offense


# Below score is the mean value of log hazard ratio of offense line players at 0 yard in 2019 season.
# Smaller value indicates the better rushing offense.

# In[ ]:


score_OL.head(20)


# Below score is the mean value of log hazard ratio of linebackers at 0 yard in 2019 season.
# Larger value indicates the better rushing defense.

# In[ ]:


score_LB.tail(20)[::-1]


# Note that these values only indicate the goodness of offensive and defensive formation at handoff.
# Hense, these values do not reflect the ability of players, such as rushing speed of RB, tackle rate of LB and so on. 

# # Predict and submit score

# In[ ]:


import warnings
warnings.filterwarnings("ignore")

from kaggle.competitions import nflrush
env = nflrush.make_env()
iter_test = env.iter_test()


# In[ ]:


baseline_hazard = compute_baseline_hazard(model, X, A)

for (play, prediction_df) in tqdm(iter_test):
        
    play.loc[play.HomeTeamAbbr.values == "ARI", 'HomeTeamAbbr'] = "ARZ"
    play.loc[play.HomeTeamAbbr.values == "BAL", 'HomeTeamAbbr'] = "BLT"
    play.loc[play.HomeTeamAbbr.values == "CLE", 'HomeTeamAbbr'] = "CLV"
    play.loc[play.HomeTeamAbbr.values == "HOU", 'HomeTeamAbbr'] = "HST"
    
    x, adj, yardToEnd = extract_feature(play, False)
    x, adj = tf.constant(x[np.newaxis], dtype=tf.float32), tf.constant(adj[np.newaxis], dtype=tf.float32)

    hazard_ratio = compute_hazard_ratio(model, x, adj)

    pred = 1 - np.exp(- np.cumsum(baseline_hazard * hazard_ratio))
    pred[yards_index > yardToEnd] = 0.
    pred = pred / pred.max()
    pred[yards_index >= yardToEnd] = 1.
    
    prediction_df = pd.DataFrame(pred[np.newaxis], columns=prediction_df.columns)
    env.predict(prediction_df)

env.write_submission_file()

