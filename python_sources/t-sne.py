#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import seaborn as sns
import matplotlib
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.decomposition import PCA, RandomizedPCA
from sklearn.manifold import TSNE

sns.set_style("darkgrid", {"axes.facecolor": ".95"})
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'svg'")


# In[ ]:


data = pd.read_csv('../input/train.csv')


# In[ ]:


data.head()


# In[ ]:


X = data.loc[:,"pixel0":"pixel783"]
y = data.label


# In[ ]:


data.shape


# In[ ]:


pca = PCA(0.95)
X_pca = pca.fit_transform(data)


# In[ ]:


pca.components_.shape


# In[ ]:


pca.explained_variance_ratio_


# In[ ]:


tsne = TSNE()
X_tsne = tsne.fit_transform(X_pca[:10000])  


# In[ ]:


matplotlib.rcParams['figure.figsize'] = (10.0, 10.0)
proj = pd.DataFrame(X_tsne)
proj.columns = ["comp_1", "comp_2"]
proj["labels"] = y
sns.lmplot("comp_1", "comp_2", hue = "labels", data = proj.sample(5000) ,fit_reg=False)

