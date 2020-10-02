#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats


# In[ ]:


sns.set(color_codes=True)


# In[ ]:


df = pd.read_csv('../input//moths-data/moths.csv')
df


# In[ ]:


x = df['meters']
sns.kdeplot(x)
sns.kdeplot(x, bw=.2, label="bw: 0.2")
sns.kdeplot(x, bw=2, label="bw: 2")
plt.legend();


# In[ ]:


sns.kdeplot(x, shade=True, cut=0)
sns.rugplot(x);


# In[ ]:


x = df['meters']
sns.distplot(x, kde=False, fit=stats.gamma);


# In[ ]:


sns.jointplot(x="meters", y="A", data=df);


# In[ ]:


sns.jointplot(x="A", y="P", data=df);


# In[ ]:


with sns.axes_style("white"):
    x=df['A']
    y=df['P']
    sns.jointplot(x=x, y=y, kind="hex", color="k")


# In[ ]:


sns.jointplot(x="A", y="P", data=df, kind="kde");


# In[ ]:


f, ax = plt.subplots(figsize=(6, 6))
sns.kdeplot(df.A, df.P, ax=ax)
sns.rugplot(df.A, color="g", ax=ax)
sns.rugplot(df.P, vertical=True, ax=ax);


# In[ ]:


sns.pairplot(df)


# In[ ]:


g = sns.PairGrid(df)
g.map_diag(sns.kdeplot)
g.map_offdiag(sns.kdeplot, n_levels=6);

