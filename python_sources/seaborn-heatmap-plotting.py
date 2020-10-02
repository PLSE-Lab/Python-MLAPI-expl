#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
#seaborn.heatmap (data, vmin=None, vmax=None, cmap=None, center=None, robust=False,
#                annot=None, fmt='.2g', annot_kws=None, linewidths=0, linecolor='white', 
#                cbar=True, cbar_kws=None, cbar_ax=None, square=False, ax=None, xticklabels=True, 
#                yticklabels=True, mask=None, **kwargs)


# In[ ]:


#data
uniform = np.random.randn(10,10)
sb.heatmap(uniform)
plt.show()


# In[ ]:


# data , vmin, vmax
uniform = np.random.randn(10,10)

sb.heatmap(uniform, vmin=-0.5, vmax=0.5)
plt.show()


# In[ ]:


#working with real dataset
flights = sb.load_dataset("flights")
print flights.info()
flights = flights.pivot("month", "year", "passengers")
sb.heatmap(flights)
plt.show()


# In[ ]:


#center
#it define the center coloring which will become white, upper will heat lower will cold data
sb.heatmap(flights, center=300)
plt.show()


# In[ ]:


# annot, fmt
# annot is bool , represent value of each data 
# fmt is formate/dtype of data 
sb.heatmap(flights, annot=True, fmt="d")
plt.show()


# In[ ]:


#linewidths : float,Width of the lines that will divide each cell.
sb.heatmap(flights, linewidths=2)
plt.show()


# In[ ]:


#linecolor : color, Color of the lines that will divide each cell.
sb.heatmap(flights, linewidths=2.5, linecolor='blue')
plt.show()


# In[ ]:


#cbar=True, cbar_kws=None, cbar_ax=None, 
#whether to draw the colorbar or not and in which axes and kws is dict of key-value mapping
ax = sb.heatmap(flights,  cbar_kws={"orientation": "horizontal"})
plt.show()


# In[ ]:


#square , cell size
#ax matplotlib axes i.e. use for multiple plotting
f, (ax,cbar_ax) = plt.subplots(2,1)
sb.heatmap(flights, ax=ax, annot=True, fmt='d')
sb.heatmap(uniform, ax=cbar_ax)
plt.show()


# In[ ]:


# if (int) gap between values, (list) only those value will appear in xlabel, (bool) label show or not
#xticklabels=True , for columns name , dtype=(int,list,bool)
#yticklabels=True , for row name , dtype=(int,list,bool)
data = np.random.randn(10,10)
sb.heatmap(data, xticklabels=5, yticklabels=False)
plt.show()


# In[ ]:


corr = np.corrcoef(np.random.randn(10, 10))
print corr
mask = np.zeros_like(corr)
print mask
#upper triange np.triu_indices_from
mask[np.tril_indices_from(mask)] = True
print mask
sb.heatmap(corr, mask=mask, vmax=.3, square=True)
plt.show()    


# In[ ]:




