import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as pt
import torch

from child_model import OPERATION_NAMES as op_names
from child_model import ChildModel

colors = [
   (.9, .2, .2),
   (.9, .6, .3),
   (.8, .9, .4),
   (.5, .9, .5),
   (.2, .5, .9),
   (.6, .4, .6),
]

def draw_graph(ops, skips, aspect=100.2):
    fig = plt.figure(frameon=False, figsize=(10,5))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    #nodes
    for i,op in enumerate(ops):
        label = "node " + str(i) + '\n' + op_names[op]
        
        x = i / len(ops)
        bbox_props = dict(boxstyle="round", fc=colors[op], ec="k", lw=2)
        ax.text(x, 0, label, ha="center", va="center", size=10, bbox=bbox_props)
    
    #edges
    N = len(ops)
    for j in range(N):
        hood = j*(j-1)//2 
        links = skips[hood: hood+j]
        
        right = (j+.5) / len(ops)
        for i in range(j):
            if links[i]:
                left = i / len(ops)
                center = (left+right)/2
                width = right-left
                if j%2 == 0:
                    arc = pt.Arc( (center,0), width, aspect*width, theta1=0, theta2=180 )
                else:
                    arc = pt.Arc( (center,0), width, aspect*width, theta1=180, theta2=0 )
                
                ax.add_patch(arc)
     
        plt.plot( [j/len(ops), (j+1)/len(ops)], [0,0], color='k')
    plt.plot( [-.1, 1/len(ops)], [0,0], color='k')
    plt.show()

def to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    else: return x
    
def draw_child(child):
    assert type(child) is ChildModel, "only draws child models!"
    ops = [ int( to_numpy(op) ) for op in child.ops]
    skips = to_numpy( child.skips )
    draw_graph(ops,skips)

if __name__ == '__main__':
    ops = [0,1,2,3,4,5,2]
    skips = [0, 1,0, 0,0,1, 0,1,0,0, 0,1,0,0,0, 0,0,1,1,0,0, 0,0,1,1,0,0,1]
    draw_graph(ops, skips)