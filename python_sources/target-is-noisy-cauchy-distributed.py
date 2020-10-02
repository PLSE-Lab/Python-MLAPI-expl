#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import cauchy


# In[ ]:


target = pd.read_csv('../input/train.csv',usecols=['target'],squeeze=True)


# Look at frequency of target

# In[ ]:


target.value_counts().head(10)


# Isolate the top 4 values.. is there a different process involved in their assignment?

# In[ ]:


top_targets = list(target.value_counts().head(4).index)


# Sample data, bin by 2 decimal places

# In[ ]:


def sample_targets(n):
    sample_plot_data=target[~target.isin(top_targets)].sample(n)
    sample_plot_data=sample_plot_data.groupby(np.round(sample_plot_data,2)).agg('count')
    sample_plot_data=sample_plot_data.rename('count').reset_index()
    return sample_plot_data


# Plot 50 random samples of 10,000 records, <font color='green'>rounded data observations in green</font>,  <font color='red'>cauchy PDF  of sample in red</font> :

# In[ ]:


k=50
fig = plt.figure(figsize=(25,k*5))
for i in range(k) : 
    if i==0:
        ax = fig.add_subplot(k,1,i+1)
    else:
        ax = fig.add_subplot(k,1,i+1,sharex=ax,sharey=ax)
    sample_plot_data = sample_targets(10000)
    ax.plot(sample_plot_data['target'],sample_plot_data['count'],linewidth=.2,color='g')
    plt.plot(sample_plot_data['target'],cauchy.pdf(sample_plot_data['target'])*120,color='r')
    plt.xticks(np.linspace(-15,15,15+15+1),np.linspace(-15,15,15+15+1))
    plt.xlabel(f'Mean : %.2f, Std : %.2f' % (np.mean(sample_plot_data.target),np.std(sample_plot_data.target)) ) 
plt.show()

