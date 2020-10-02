#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


from IPython.display import IFrame
display(IFrame(src = 'https://www.theregister.co.uk/2019/11/07/python_java_github_javascript/', width=1000, height=700))


# In[ ]:


from IPython.display import YouTubeVideo
YouTubeVideo("HaSpqsKaRbo")


# In[ ]:


documentation = IFrame(src = 'https://ipywidgets.readthedocs.io/en/latest/index.html', width=1000, height=600)
display(documentation)


# In[ ]:


import ipywidgets
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


5*5


# In[ ]:


def f1(x):
    
    y = x * x
    print("{0} * {0} = {1}".format(x, y))


# In[ ]:


f1(3)


# In[ ]:


ipywidgets.interact(f1, x = 10)


# In[ ]:


def myPlot(frequency = 2, color = 'blue', lw = 4, grid = True):
    """
    plots cos(pi * f * x)
    """
    
    x = np.linspace(-3, 3, 1000)
    fig, ax = plt.subplots(1, 1, figsize = (6, 4))
    ax.plot(x, np.cos(np.pi * frequency * x), lw = lw, color = color)
    ax.grid(grid)
    plt.title("plot of cos(pi * f * x)", fontdict = {"size" : 20})
    
myPlot()


# In[ ]:


ipywidgets.interact(myPlot, color = ['blue', 'red', 'green'], lw = (1, 10));


# In[ ]:


filterColumn("RM", 5)


# In[ ]:


from sklearn.datasets import load_boston
boston = load_boston()

print(boston.DESCR)

boston_df = pd.DataFrame(boston.data, columns = boston.feature_names)
boston_df["PRICE"] = boston.target
boston_df.head(10)


# In[ ]:


def filterColumn(column, threshold):
    
    boston_df_select = boston_df.loc[boston_df[column] > threshold]
    msg = "There are {:,} records for {} > {:,}".format(boston_df_select.shape[0], column, threshold)
    
    print("{0}\n{1}\n{0}".format("=" * len(msg), msg))
    display(boston_df_select.head(10))

