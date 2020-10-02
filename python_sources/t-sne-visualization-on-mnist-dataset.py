#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import matplotlib.pyplot as plt
print(os.listdir("../input"))


# In[ ]:


from sklearn.manifold import TSNE
df = pd.read_csv("../input/train.csv")
df = df[:1000]


# In[ ]:


label = df.label
df.drop("label", axis=1, inplace=True)


# In[ ]:


from sklearn.preprocessing import StandardScaler
standardized_data = StandardScaler().fit_transform(df)
standardized_data.shape


# In[ ]:


model = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200, n_iter=1000)
reduced_data = model.fit_transform(standardized_data)


# In[ ]:


reduced_df = np.vstack((reduced_data.T, label)).T
reduced_df = pd.DataFrame(data=reduced_df, columns=["X", "Y", "label"])
reduced_df.label = reduced_df.label.astype(np.int)
reduced_df.head()


# In[ ]:


import seaborn as sns
reduced_df.dtypes


# In[ ]:


g = sns.FacetGrid(reduced_df, hue='label', size=6).map(plt.scatter, 'X', 'Y').add_legend()


# Converted 2 dimensions are **X** and **Y** and visualization is shown in 2-D.
# Some more points anout t-sne:
# * Perplexity ensures the number of neighbours t-sne is preserving on the basis of distance metric.
# * Try perplexity value in the range of 5 to 50, as suggested in research paper.
# * Try multiple value of steps untill changes in cluster is saturated.
# * Distance between clusters may not be useful.
# * With varying value of perplexity and steps, we may see some shapes of cluster, these are just meaningless.
# * t-sne generally expands dense cluster and shrink sparse cluster.
# Read amazing blog on interpretation of t-sne results [here](https://distill.pub/2016/misread-tsne)
