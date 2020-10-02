#!/usr/bin/env python
# coding: utf-8

# Script for plotting the distribution of the percentage of satisfied respondents, depending on the value of each of 8 scales.
# 
# Imports:

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Load dataset and a quick look to data structure:

# In[ ]:


df = pd.read_csv('/kaggle/input/kpmi-mbti-mod-test/kpmi_data.csv', sep=';')

df.head()


# In[ ]:


scales = ['e', 'i', 's', 'n', 't', 'f', 'j', 'p']

# Preparing 4x2 grid for charts.
fig, axs = plt.subplots(4, 2, figsize=(13,13), sharey=True, tight_layout=True)

for i, scale_name in enumerate(scales):
    # Grouping data by scale values and calculating percent of satisfied with each value.
    g = df.groupby(f'scale_{scales[i]}').agg({'satisfied':[lambda x: round(100*x.sum()/ x.count(), 2)]})
    
    g.columns = g.columns.map('_'.join).str.replace('<lambda>','percent')
    # Background grid.
    axs[i//2, i%2].grid()
    # Draw grid behind other graph elemnts.
    axs[i//2, i%2].set_axisbelow(True)
    axs[i//2, i%2].bar(g.index, g['satisfied_percent'])
    axs[i//2, i%2].set_title(f'Scale {scales[i].upper()}')
    plt.setp(axs[i//2, i%2], xlabel='Scale raw value')
    if (i+1)%2:
        # Inverting X-axis values of every first graph in a row for better visual perception.
        axs[i//2, i%2].invert_xaxis()
        plt.setp(axs[i//2, i%2], ylabel='Percent of satisfied')

