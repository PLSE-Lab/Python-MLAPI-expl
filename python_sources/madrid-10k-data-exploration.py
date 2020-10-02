#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[ ]:


data = pd.read_csv("/kaggle/input/madrid-10k-new-years-eve-popular-race/madrid_10k_20191231.csv", sep=",")


# In[ ]:


#Here is a little method to print distribution over one variable. Hope it helps!
def plot_histogram(df_data, s_title, s_x, s_y, i_bins, norm=True):
    d_plots = {}
    for value in df_data[s_y].unique():
        d_plots[value]=df_data[df_data[s_y]==value][s_x]

    f_min, f_max = df_data[s_x].min(), df_data[s_x].max() 
    bins = np.linspace(f_min, f_max, i_bins)

    for key, value in d_plots.items():
        pyplot.hist(value, bins, alpha=0.5, label=key, density=norm)
    
    pyplot.legend(loc='upper right')
    plt.title(s_title)
    pyplot.show()


# In[ ]:


plot_histogram(df_data=data, s_title='category distribution', s_x='total_seconds', s_y='age_category', i_bins=100, norm=False)


# In[ ]:


plot_histogram(df_data=data, s_title='gender distribution', s_x='total_seconds', s_y='sex', i_bins=100, norm=False)

