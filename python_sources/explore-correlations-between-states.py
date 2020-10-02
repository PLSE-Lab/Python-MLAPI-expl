#!/usr/bin/env python
# coding: utf-8

# It is mostly followed from Sentdex

# 

# 

# In[ ]:


import numpy as np
import pandas as pd


# In[ ]:


df = pd.read_csv("../input/Minimum Wage Data.csv", encoding="Latin")
df.head()


# In[ ]:


grp = df.groupby("State")
Alb = grp.get_group("Alaska").set_index("Year")
Alb.head()


# ### *Pandas is like a list of dictionaries. Like: [{All column1 cells},{All column2 cells},...]
# 
# 
# 

# In[ ]:


min_wage_2018 = pd.DataFrame()
for name, group in grp:
  if min_wage_2018.empty:
    min_wage_2018 = group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name})
  else:
    min_wage_2018=min_wage_2018.join(group.set_index("Year")[["Low.2018"]].rename(columns={"Low.2018":name}))
min_wage_2018.head(10)


# In[ ]:


min_wage_2018.std()


# In[ ]:


df.set_index("Year")[["Low.2018"]]


# In[ ]:


df[df["State"]=="Alabama"].set_index("Year").head()


# **Correlation**: corr() shows tha changes according to each other. Shows us a correlation matrix. Each coefficent in correlation matrix is between **-1 and +1** 
# 
# **+1 means** that there is perfect dorrelation between these two sets. If one is increases the other increases at the same time with same percentage.
# 
# **0 means** that there is no correlation
# 
# **-1 means** that there is perfect negative correlation. If the one is increases the other decreases at the same time and same percentage.

# In[ ]:


coor_matrix = min_wage_2018.corr()
coor_matrix


# In[ ]:


list_of_list = []
for column in coor_matrix:
    #list_of_list.append(list(min_wage_2018[column].values))
    list_of_list.insert(0, list(coor_matrix[column].values))
list_of_list


# In[ ]:


get_ipython().system('pip install plotly')


# In[ ]:


list(reversed(list(coor_matrix.columns)))


# In[ ]:


import plotly
import plotly.plotly as py
import plotly.graph_objs as go

#plotting the heated correlation matrix

trace = go.Heatmap(z=list_of_list,
                   x=list(coor_matrix.columns),
                   y=list(reversed(list(coor_matrix.columns))))
data=[trace]
py.iplot(data, filename='labelled-heatmap')


# There are a lot of white rows and columns. They are comes from zero `Low.2018` values. If we can get rid of 0 values. It will be more elegant.

# In[ ]:


# Which states Low.2018 column is 0 at least in one year.
df[df["Low.2018"]==0]["State"].unique()


# In[ ]:


#dropna(axis=0) drops all the row if a cell is NaN in that row. Which is default.
#dropna(axis=1) drops all the column if a cell is NaN in that column.
min_wage_2018.replace(0, np.NaN).dropna(axis=1).corr()


# In[ ]:




