#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


pip install snap-stanford


# In[ ]:


import snap


# Notebook was created with using [HW1-bundle](https://docs.google.com/uc?export=download&id=1xoPFKf78PrdGEELQYtomXpkFD8QFj6sI) template by cs224w

# ## 1 Network Characteristics

# ### 1.1 Degree Distribution

# In[ ]:


erdosRenyi = None
smallWorld = None
collabNet = None


# In[ ]:


edges = 0
def genErdosRenyi(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Erdos-Renyi graph with N nodes and E edges
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph.New(N, E)
    for i in range(N):
        Graph.AddNode(i)
        
    edges = 0
    Rnd = snap.TRnd(42)
    Rnd.Randomize()
    while edges < E:
        SrcNId = Graph.GetRndNId(Rnd)
        DstNId = Graph.GetRndNId(Rnd)
        if SrcNId != DstNId and not Graph.IsEdge(SrcNId, DstNId): # check it's not self-loop and nonexistent edge
            Graph.AddEdge(SrcNId, DstNId)
            edges += 1
    ############################################################################
    return Graph


# In[ ]:


def genCircle(N=5242):
    """
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Circle graph with N nodes and N edges. Imagine the nodes form a
        circle and each node is connected to its two direct neighbors.
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.TUNGraph.New(N, N)
    for i in range(N):
        Graph.AddNode(i)
    
    for i in range(1, N):
        Graph.AddEdge(i-1, i)
    Graph.AddEdge(N-1, 0)
    ############################################################################
    return Graph


# In[ ]:


def connectNbrOfNbr(Graph, N=5242):
    """
    :param - Graph: snap.PUNGraph object representing a circle graph on N nodes
    :param - N: number of nodes

    return type: snap.PUNGraph
    return: Graph object with additional N edges added by connecting each node
        to the neighbors of its neighbors
    """
    ############################################################################
    # TODO: Your code here!
    for i in range(N):
        Graph.AddEdge(i, (i + 2) % N)
    ############################################################################
    return Graph


# In[ ]:


def connectRandomNodes(Graph, M=4000):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph
    :param - M: number of edges to be added

    return type: snap.PUNGraph
    return: Graph object with additional M edges added by connecting M randomly
        selected pairs of nodes not already connected.
    """
    ############################################################################
    # TODO: Your code here!
    pairs = []
    for i in range(Graph.GetNodes()):
        for j in range(i+1, Graph.GetNodes()): # iterate from (i+1) for avoid self-loop
            if not Graph.IsEdge(i, j): # check for nonexistense of edge
                pairs.append((i, j))
                
    indexs = np.random.randint(0, len(pairs), M) # generate indexs for random pairs of nodes
    pairs = np.array(pairs)
    randPairs = pairs[indexs]
    
    for pair in randPairs:
        Graph.AddEdge(int(pair[0]), int(pair[1])) # add picked-out edges
    ############################################################################
    return Graph


# In[ ]:


def genSmallWorld(N=5242, E=14484):
    """
    :param - N: number of nodes
    :param - E: number of edges

    return type: snap.PUNGraph
    return: Small-World graph with N nodes and E edges
    """
    Graph = genCircle(N)
    Graph = connectNbrOfNbr(Graph, N)
    Graph = connectRandomNodes(Graph, 4000)
    return Graph


# In[ ]:


def loadCollabNet(path):
    """
    :param - path: path to edge list file

    return type: snap.PUNGraph
    return: Graph loaded from edge list at `path and self edges removed

    Do not forget to remove the self edges!
    """
    ############################################################################
    # TODO: Your code here!
    Graph = snap.LoadEdgeList(snap.PUNGraph, '/kaggle/input/ml-in-graphs-hw1/ca-GrQc.txt', 0, 1)
    for node in Graph.Nodes(): # remove the self edges
        if Graph.IsEdge(node.GetId(), node.GetId()):
            Graph.DelEdge(node.GetId(), node.GetId())
    ############################################################################
    return Graph


# In[ ]:


def getDataPointsToPlot(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return values:
    X: list of degrees
    Y: list of frequencies: Y[i] = fraction of nodes with degree X[i]
    """
    ############################################################################
    # TODO: Your code here!
    X, Y = [], []
    DegToCntV = snap.TIntPrV() # a vector of (degree, number of nodes of such degree) pairs
    snap.GetDegCnt(Graph, DegToCntV) # computes a degree histogram: a vector of pairs (degree, number of nodes of such degree)
    for item in DegToCntV:
        X.append(item.GetVal1()) # list of degrees
        Y.append(np.float(item.GetVal2()) / np.float(Graph.GetNodes())) # list of frequencies
    ############################################################################
    return X, Y


# In[ ]:


def Q1_1():
    """
    Code for HW1 Q1.1
    """
    global erdosRenyi, smallWorld, collabNet
    erdosRenyi = genErdosRenyi(5242, 14484)
    smallWorld = genSmallWorld(5242, 14484)
    collabNet = loadCollabNet("ca-GrQc.txt")
    
    plt.figure(figsize=(10,7))
    
    x_erdosRenyi, y_erdosRenyi = getDataPointsToPlot(erdosRenyi)
    plt.loglog(x_erdosRenyi, y_erdosRenyi, color = 'y', label = 'Erdos Renyi Network')

    x_smallWorld, y_smallWorld = getDataPointsToPlot(smallWorld)
    plt.loglog(x_smallWorld, y_smallWorld, linestyle = 'dashed', color = 'r', label = 'Small World Network')

    x_collabNet, y_collabNet = getDataPointsToPlot(collabNet)
    plt.loglog(x_collabNet, y_collabNet, linestyle = 'dotted', color = 'b', label = 'Collaboration Network')
    
    plt.grid(True, which="both", color="0.75")
    plt.xlabel('Node Degree (log)')
    plt.ylabel('Proportion of Nodes with a Given Degree (log)')
    plt.title('Degree Distribution of Erdos Renyi, Small World, and Collaboration Networks')
    plt.legend()
    plt.show()


# In[ ]:


# Execute code for Q1.1
Q1_1()


# ### 1.2 Clustering Coefficient

# In[ ]:


def calcClusteringCoefficientSingleNode(Node, Graph):
    """
    :param - Node: node from snap.PUNGraph object. Graph.Nodes() will give an
                   iterable of nodes in a graph
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: local clustering coeffient of Node
    """
    ############################################################################
    # TODO: Your code here!
    C = 0.0
    if Node.GetDeg() < 2: # if we have NodeDegree < 2 return 0
        return C
    
    for i in range(Node.GetDeg()):
        for j in range(i + 1, Node.GetDeg()):
            if Graph.IsEdge(Node.GetNbrNId(i), Node.GetNbrNId(j)): 
                # check existense of edge between i-th and j-th neighboring node.
                C += 1
    
    C = 2. * C / (Node.GetDeg() * (Node.GetDeg() - 1)) 
    #  = 2 * numbEdgesBetweenNeibors / (ki * (ki - 1)) for each node, ki - Node-i's degree
    ############################################################################
    return C


# In[ ]:


def calcClusteringCoefficient(Graph):
    """
    :param - Graph: snap.PUNGraph object representing an undirected graph

    return type: float
    returns: clustering coeffient of Graph
    """
    ############################################################################
    # TODO: Your code here! If you filled out calcClusteringCoefficientSingleNode,
    #       you'll probably want to call it in a loop here
    C = 0.0
    for node in Graph.Nodes():
        C_ = calcClusteringCoefficientSingleNode(node, Graph)
        C += C_
    C = C / np.float(Graph.GetNodes()) # averaging for all nodes
    ############################################################################
    return C


# In[ ]:


def Q1_2():
    """
    Code for Q1.2
    """
    C_erdosRenyi = calcClusteringCoefficient(erdosRenyi)
    C_smallWorld = calcClusteringCoefficient(smallWorld)
    C_collabNet = calcClusteringCoefficient(collabNet)

    print('Clustering Coefficient for Erdos Renyi Network: %f' % C_erdosRenyi)
    print('Clustering Coefficient for Small World Network: %f' % C_smallWorld)
    print('Clustering Coefficient for Collaboration Network: %f' % C_collabNet)


# In[ ]:


# Execute code for Q1.2
Q1_2()

