#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import plotly.plotly as py
import plotly.graph_objs as go
import plotly
#plotly.tools.set_credentials_file(username='username', api_key='api_key')
#To run this code in your own notebook you have to change 'username' and 'api_key' by your own values after having an account on https://plot.ly/


# In[ ]:


data =  pd.read_csv('../input/BCT_statistque.csv',sep =",")
data.head()


# ## In order to run the cell below, you have to sigin to your plotly account
# 
# ### Link to the output
# 
# https://plot.ly/~geeks/0?share_key=wUoqqJ2DK3a4T1tqQi7ggT

# In[ ]:


# Create traces
variables = data.columns.values
p = []
for i in range(1,len(variables)):
    a = go.Scatter(
        x = data.Date,
        y = data[variables[i]],
        mode = 'lines+markers',
        name = variables[i])
    p.append(a)

#py.iplot(p, filename='line-mode')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




