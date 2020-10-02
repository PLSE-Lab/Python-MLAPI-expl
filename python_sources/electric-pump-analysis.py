#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import os
print(os.listdir("../input"))


# In[ ]:


data=pd.read_csv("../input/hotel LA1 - hotel LA1.csv")


# In[ ]:


pd.set_option('display.max_columns', None)  
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)
data.head()


# In[ ]:


data.shape


# In[ ]:


del data['Unnamed: 0']
del data['Unnamed: 1']


# In[ ]:


data.index


# In[ ]:


data.head()


# ### So, we can assume by this that 

# 1. This is a star connection as phase and line current are not given separately 
# 2. Supply is unbalanced
# 3. Too much heat will be generated as there is no supply in 1 line 

# ### Further,

# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


data.describe()


# In[ ]:


data.head()


# In[ ]:


for col in data.columns: 
    print(col) 


# In[ ]:


#data.rename(columns = {'Unnamed: 2':'Wh'}, inplace = True) 
#data.rename(columns = {"line to line Voltage Red phase to yellow phase":"Voltage ry"}, inplace = True)
#data.rename(columns = {'line to line Voltage yellow phase to blue phase':'Voltage yb'}, inplace = True)
#data.rename(columns = {'line to line Voltage blue phase to red phase':'Voltage br'}, inplace = True)
#data.rename(columns = {'phase Voltage Red':'Voltage R'}, inplace = True)
#data.rename(columns = {'phase Voltage Yellow':'Voltage Y'}, inplace = True)
#data.rename(columns = {'phase Voltage blue':'Voltage B'}, inplace = True)
#data.rename(columns = {'Unnamed: 9':'Current R'}, inplace = True)
#data.rename(columns = {'Unnamed: 10':'Current Y'}, inplace = True)
#data.rename(columns = {'Unnamed: 11':'Current B'}, inplace = True)
#data.rename(index=str, columns={"line to line Voltage Red phase to yellow phase":"Voltage ry"})


# In[ ]:


new_header = data.iloc[0] #grab the first row for the header
data = data[1:] #take the data less the header row
data.columns = new_header


# In[ ]:


for col in data.columns: 
    print(col) 


# In[ ]:


data.head()


#  ### More current lead to less Wh

# In[ ]:


data.describe()


# #### Voltage (line to line Voltage Red phase to yellow phase) is always 0

# In[ ]:


plt.plot(data.describe())


# In[ ]:




