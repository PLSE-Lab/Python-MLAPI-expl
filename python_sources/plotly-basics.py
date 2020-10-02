#!/usr/bin/env python
# coding: utf-8

# # Data Visualization

# ### Plotly
# plotly is an interactive visualization library.
# cufflinks connects plotly wih pandas

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


from plotly import __version__


# In[ ]:


print(__version__)


# In[ ]:


import cufflinks as cf


# In[ ]:


from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[ ]:


init_notebook_mode(connected = True)


# In[ ]:


cf.go_offline()


# In[ ]:


# Data
df = pd.DataFrame(np.random.randn(100,4),columns='A B C D'.split())
df.head()


# In[ ]:


df2 = pd.DataFrame({'Category':['A','B','C'],'Values':[32,43,50]})
df2


# In[ ]:


df.plot()


# In[ ]:


df.iplot()


# In[ ]:


df.iplot(kind='scatter',x='A',y='B',mode='markers')


# In[ ]:


df2.iplot(kind='bar',x='Category',y='Values')


# In[ ]:


df.iplot(kind='box')


# In[ ]:


df3=pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[500,400,300,200,100]})
df3


# In[ ]:


df3.iplot(kind='surface')


# In[ ]:


df4=pd.DataFrame({'x':[1,2,3,4,5],'y':[10,20,30,20,10],'z':[5,4,3,2,1]})
df4.iplot(kind='surface')


# In[ ]:


df['A'].iplot(kind='hist',bins=50)


# In[ ]:


df.iplot(kind="hist")


# In[ ]:


df[['A','B']].iplot(kind='spread')


# In[ ]:


df.iplot(kind='bubble',x='A',y='B',size='C')


# In[ ]:


df.scatter_matrix()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




