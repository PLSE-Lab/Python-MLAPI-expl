#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd


# In[ ]:


data = pd.read_csv(r'/kaggle/input/2016-american-election-dataset/america.csv')
data.head()


# In[ ]:


Ndata = pd.DataFrame(data[['state_code','Trump_vote','Cliton_vote']])
Ndata.head()


# In[ ]:


Ndata.tail()


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


Ndata.set_index('state_code').plot(figsize = (20,8),grid= True)


# In[ ]:


Ndata.plot(x= "state_code", y=["Trump_vote","Cliton_vote"], kind="bar",figsize = (20,8))


# In[ ]:


Trumpsum = Ndata['Trump_vote'].sum()
Clitonsum = Ndata['Cliton_vote'].sum()
Trumpsum


# In[ ]:


Clitonsum


# In[ ]:


labels = 'Trump', 'Cliton'
sizes = [62883925, 66753516]
colors = ['lightcoral', 'lightskyblue']
explode = (0.1, 0)

# Plot
plt.pie(sizes, explode=explode, labels=labels, colors=colors,
autopct='%1.1f%%', shadow=True, startangle=140)

plt.axis('equal')
plt.show()


# In[ ]:


#Please rate me in the comment section

