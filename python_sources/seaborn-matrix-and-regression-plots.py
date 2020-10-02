#!/usr/bin/env python
# coding: utf-8

# # Matrix Plots

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')


# In[ ]:


tips.head()


# In[ ]:


flights.head()


# In[ ]:


tc = tips.corr()


# In[ ]:


tc


# In[ ]:


sns.heatmap(tc)


# In[ ]:


sns.heatmap(tc,annot = True,cmap = 'coolwarm')


# In[ ]:


flights.head()


# In[ ]:


ft = flights.pivot_table(index = 'month',columns = 'year',values = 'passengers')


# In[ ]:


sns.heatmap(ft)


# In[ ]:


sns.heatmap(ft,cmap = 'coolwarm',linewidth = 3,linecolor = 'white')


# In[ ]:


sns.clustermap(ft)


# In[ ]:


sns.clustermap(ft,cmap = 'coolwarm')


# # Regression Plots

# In[ ]:


tips.head()


# In[ ]:


sns.lmplot(x = 'total_bill',y = 'tip',data = tips,hue = 'smoker')


# In[ ]:


sns.lmplot(x = 'total_bill',y = 'tip',data = tips,hue = 'sex',markers = ['o','v'])


# In[ ]:


sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'sex')


# In[ ]:


sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'sex',row = 'time')


# In[ ]:


sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'day',hue = 'sex')


# In[ ]:


sns.lmplot(x = 'total_bill',y = 'tip',data = tips,col = 'day',hue = 'sex',aspect = 0.6,height = 8)


# # Grids

# In[ ]:


import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
iris = sns.load_dataset('iris')
iris.head()


# In[ ]:


sns.pairplot(iris)


# In[ ]:


sns.pairplot(iris,hue = 'species')


# In[ ]:


#use of pair grid
import matplotlib.pyplot as plt


# In[ ]:


g = sns.PairGrid(iris)
g.map(plt.scatter)


# In[ ]:


g = sns.PairGrid(iris)
g.map_diag(sns.distplot)
g.map_upper(plt.scatter)
g.map_lower(sns.kdeplot)


# In[ ]:


tips.head()


# In[ ]:


g = sns.FacetGrid(data = tips, col = 'time',row = 'smoker')
g.map(sns.distplot,'total_bill')


# In[ ]:


g = sns.FacetGrid(data = tips, col = 'time',row = 'smoker')
g.map(sns.scatterplot,'total_bill','tip')


# # Style and Color

# In[ ]:


tips.head()


# In[ ]:


sns.set_style('darkgrid')
sns.countplot(x = 'sex',data = tips)
sns.despine()


# In[ ]:


sns.set_context('notebook')
sns.countplot(x = 'sex',data = tips)


# In[ ]:


sns.lmplot('total_bill','tip',data = tips,hue = 'sex',palette = 'inferno')


# In[ ]:


sns.lmplot('total_bill','tip',data = tips,hue = 'sex',palette = 'seismic')


# In[ ]:


#refer to matplotlib colormap section on google to get more insights on the changing patterns and getting differnt colormaps.

