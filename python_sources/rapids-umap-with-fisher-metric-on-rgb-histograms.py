#!/usr/bin/env python
# coding: utf-8

# # Based on a kernel by Bojan Tunguz: https://www.kaggle.com/tunguz/melanoma-tsne-and-umap-embeddings-with-rapids

# [Rapids](https://rapids.ai) is an open-source GPU accelerated Data Science and Machine Learning library, developed and mainatained by [Nvidia](https://www.nvidia.com). It is designed to be compatible with many existing CPU tools, such as Pandas, scikit-learn, numpy, etc. It enables **massive** acceleration of many data-science and machine learning tasks, oftentimes by a factor fo 100X, or more. 
# 
# Rapids is still undergoing developemnt, and as of right now it's not availabel in the Kaggle Docker environment. If you are interested in installing and running Rapids locally on your own machine, then you should [refer to the followong instructions](https://rapids.ai/start.html).

# # Overview
# 
# The idea is to visualize the space of RGB **histograms** of the tumor images. The distance/dissimilarity between the histograms is measured with the Fisher metric (more info inside the notebook). This space is embedded in 3D using UMAP, and displayed with plotly. 
# 
# # View of data
# ![](https://www.googleapis.com/download/storage/v1/b/kaggle-user-content/o/inbox%2F466630%2F0e09c8550d8320bca4a362ba80ce82dc%2Fumap_with_histograms.png?generation=1592226072209541&alt=media)
# 
# # More information
# https://www.kaggle.com/c/siim-isic-melanoma-classification/discussion/158829
# 
# # Interaction 
# Clicking on a point on the plot below shows 5 closest images (by the UMAP coords) and their histograms.

# In[ ]:


get_ipython().run_cell_magic('time', '', 'try:\n    import cudf, cuml\n    print(\'rapids already installed\')\nexcept:\n    # INSTALL RAPIDS OFFLINE (FROM KAGGLE DATASET). TAKES 1 MINUTE :-)\n    print(\'installing rapids (should take ~80sec)\')\n    import sys\n    !cp ../input/rapids/rapids.0.13.0 /opt/conda/envs/rapids.tar.gz  2>/dev/null\n    \n    !cd /opt/conda/envs/ && tar -xzf rapids.tar.gz \n    sys.path = ["/opt/conda/envs/rapids/lib"] + ["/opt/conda/envs/rapids/lib/python3.6"] + ["/opt/conda/envs/rapids/lib/python3.6/site-packages"] + sys.path\n    !cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/  2>/dev/null\n    print(\'done installing rapids\')\n    import cudf, cuml')


# In[ ]:


import cupy as cp
import numpy as np
import pandas as pd
import os
from cuml.manifold import TSNE, UMAP
import matplotlib.pyplot as plt
from matplotlib.pyplot import ylim, xlim

import plotly.express as px

import plotly.graph_objs as go
from ipywidgets import Output, VBox

get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from PIL import Image

def central_crop(im):
    w = im.shape[0]
    nw = int(np.floor(w/np.sqrt(2)))
    d = (w-nw)//2
    return im[d:-d, d:-d, :]

def to_grayscale(IM):
    return np.asarray(Image.fromarray(IM).convert('L'))    


# In[ ]:


get_ipython().run_cell_magic('time', '', 'train_df = pd.read_csv("../input/siim-isic-melanoma-classification/train.csv")\nIMS = np.load(\'../input/siimisic-melanoma-resized-images/x_train_64.npy\')\n\nMERGE = True\n\nif MERGE:\n    IMS_test = np.load(\'../input/siimisic-melanoma-resized-images/x_test_64.npy\')\n    IMS = np.concatenate([IMS, IMS_test])\n    test_df = pd.read_csv("../input/siim-isic-melanoma-classification/test.csv")\n    \n    train_df[\'is_train\'] = True\n    train_df = train_df.append(test_df)\n    train_df.is_train.fillna(False, inplace = True)\n    train_df.target.fillna(-1, inplace = True)\n    train_df.reset_index(inplace = True, drop = True)\n\nIMS = list(map(central_crop, IMS))')


# In[ ]:


get_ipython().run_cell_magic('time', '', 'def rgb_hists(IMS):\n    HS = []\n    k = 3 if len(IMS[0].shape)==3 else 1\n    for im in IMS:            \n        comps = [np.bincount(im.reshape(-1,k)[:,i], minlength=256) for i in range(k)]    \n        HS.append(np.concatenate(comps))\n        \n    return np.asarray(HS)\n    \nhists = rgb_hists(IMS)')


# # Fisher metric
# 
# Ideally we'd use something like the KL divergence to compare the histograms, but this version of UMAP works only with the Eclidean metric it seems. We use a sqrt-transform which maps the Fisher manifold isometrically onto the Euclidean space. In other words, computing the Euclidean distance in the transformed space (train) corresponds to computing the Fisher metric in the original space (hists). 
# 
# The fisher Metric is an approximation (of the sqrt) of the KL divergence. (BTW: This sqrt is not related to the other sqrt.) More precisely, the geodesics (between two histograms/discrete probability distributions) in the Fisher space minimize the integrals of KL divergences along the curve.

# In[ ]:


train = np.sqrt(hists)
#train = hists


# In[ ]:


MEANS = np.median( np.asarray(IMS).reshape(len(IMS), -1, 3), axis = 1)

train_df['mean_col'] = [(r/255,g/255,b/255) for r,g,b in MEANS]
train_df['mean_lum'] = [r/255 + g/255 + b/255 for r,g,b in MEANS]
train_df['mean_r'] = [r/255 for r,g,b in MEANS]


# In[ ]:


def plot_rgb_hist(ax, hist):
    hs = np.split(hist,3)
    ax.plot(hs[0], 'r')
    ax.plot(hs[1], 'g')
    ax.plot(hs[2], 'b')


# In[ ]:


st = 59


# In[ ]:


print(f'using random_state = {st}')
umap = UMAP(n_components=3, random_state=st, n_neighbors = 12, n_epochs = 1_000)

xyz = umap.fit_transform(train)

train_df['emb_x'] = xyz[:, 0]
train_df['emb_y'] = xyz[:, 1]
train_df['emb_z'] = xyz[:, 2]

train_df.to_csv('tabular_with_umap_coords', index = False)


# In[ ]:


import plotly.graph_objs as go
import plotly.express as px
from ipywidgets import Output, VBox
from scipy.spatial import KDTree

def show_interactive_embedding(train_df, colors = None, sizes = None):
    X = train_df[['emb_x', 'emb_y', 'emb_z']].values
    KD = KDTree(X)
    
    train_df['ind'] = train_df.index    

    sc = px.scatter_3d(train_df, x = 'emb_x', y = 'emb_y', z = 'emb_z', 
                  size = sizes,               
                  #size = (PRED != q.target)*10 + 0.5,
                  color = colors,
                  #symbol = 'is_train',
                  hover_data = train_df.columns,
                  width = 1200, height = 1200,
                  )

    fig = go.FigureWidget(data=sc)

    out = Output()

    def same_patients(sel):
        id = train_df[train_df.index == sel].patient_id.values[0]
        ALL = train_df[train_df.patient_id == id].index.values
        ALL[list(ALL).index(sel)], ALL[0] = ALL[0], ALL[list(ALL).index(sel)]
        return ALL

    def similar_in_umap(ind, k = 20):    
        ns = KD.query(X[ind], k = k)[1]
        return ns

    @out.capture(clear_output=True)
    def handle_click(trace, points, state):
        if not points.point_inds:
            print('handle_click received empty selection, probably a bug in plotly...')
            return

        sel = points.point_inds[0]    
        #ALL = same_patients(sel)    
        ALL = similar_in_umap(sel, 50)

        _, axs = plt.subplots(len(ALL), 2, figsize = (10,len(ALL)*2))
        #axs = axs.ravel()

        new_sizes = sizes.copy()
        for i,x in enumerate(ALL):
            axs[i,1].imshow(IMS[x])
            plot_rgb_hist(axs[i,0], hists[x])        
            axs[i,0].set_title(f"{train_df.at[x, 'target']}, {train_df.at[x,'ind']} {train_df.at[x, 'diagnosis']} ")
            axs[i,0].axes.get_xaxis().set_visible(False)        
            new_sizes[x] = 15    

        fig.update_traces(marker=dict(size=new_sizes))    

        plt.show()        

        
    # bug in plotly -- if colors are specified something is wrong with selections...
    if colors is None:
        fig.data[0].on_click(handle_click)    

    return VBox([fig, out])


# In[ ]:





# In[ ]:


q = train_df[(train_df.emb_x > 5) & (train_df.emb_y > -2) & (train_df.emb_z < -1)]


# In[ ]:


sizes_by_target = (train_df.target>0).values*10 + 0.5


# In[ ]:


train_df.is_train.mean()


# In[ ]:


q.is_train.mean()


# In[ ]:


#show_interactive_embedding(train_df, sizes = sizes_by_target, colors = train_df.is_train.values)
show_interactive_embedding(train_df, sizes = sizes_by_target, colors = None)


# In[ ]:


train_df[(train_df.emb_y >= 3) & (train_df.target>=1)].target.count()


# In[ ]:


#show_interactive_embedding(train_df, sizes = None, colors = train_df.is_train.values)


# In[ ]:


#show_interactive_embedding(train_df, sizes = sizes_by_target, colors = train_df.sex.fillna('n/a').values)

