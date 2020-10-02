# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
print("np version ",np.__version__)
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
print("pd version ",pd.__version__)
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
import networkx as nx
print("nx version ",nx.__version__)
import matplotlib.pyplot as plt
import matplotlib
print("matplotlib version ",matplotlib.__version__)
import pylab 
print("pylab version ",matplotlib.__version__)

#load graph
G = nx.read_edgelist(path="../input/dataengmaxconnectedcomponent/MaxConn.edgelist", delimiter=":")
#load layout
forceAtlas2=pd.read_csv("../input/dataengforceatlas2layout/force_atlas_2_layout.csv",header=None)
forceAtlas2Dict={}
for i in forceAtlas2.values:
    forceAtlas2Dict[str(i[0])]=np.fromstring(i[1][1:-1],sep=',')

#dpi*figsize[0] should less than 2^16
def save_graph(graph,file_name,pos=None,node_color='#1f78b4',edge_colors='k',node_size=80,width=0.5, dpi=300):
    #initialze Figure
    plt.figure(num=None, figsize=(200, 200), dpi=dpi)
    plt.axis('off')
    fig = plt.figure(1)
    if pos==None:
        pos = nx.spring_layout(graph)
    
    nx.draw_networkx_nodes(graph,pos
                            ,node_color=node_color
                            ,node_size=node_size
                            ,linewidths=width)
    nx.draw_networkx_edges(graph,pos
                            ,node_size=node_size
                            ,edge_color=edge_colors
                            ,width=width
                            ,alpha=0.3
                          )
#     nx.draw_networkx_labels(graph,pos)

    cut = 1.00
    xmax = cut * max(xx for xx, yy in pos.values())
    xmin = cut * min(xx for xx, yy in pos.values())
    ymax = cut * max(yy for xx, yy in pos.values())
    ymin = cut * min(yy for xx, yy in pos.values())
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.show()
    plt.savefig(file_name,bbox_inches="tight",dpi=dpi)
    pylab.close()
    del fig

#Assuming that the graph g has nodes and edges entered
save_graph(G,"force_atlas2.pdf",pos=forceAtlas2Dict)#    ,node_color=node_colors