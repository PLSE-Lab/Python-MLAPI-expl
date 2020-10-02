#!/usr/bin/env python
# coding: utf-8

# ## Heatmap Visualization of votes among all parties in all constituency for all election years

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input/election-data-wrangling"))


# ### Using Zeeshan's combined data file after data wranggling
# ### Adding a new field called Seat which is a combination of Constituency, Year and Seat. This will be used for Indexing and Visualization

# In[ ]:


## 2002 Elections ## 
NA_All = pd.read_csv("../input/election-data-wrangling/NA2002-18.csv", encoding = "ISO-8859-1")
NA_Less = pd.DataFrame([])
NA_Less['Seat'] = NA_All['ConstituencyTitle'] + '-' + NA_All['Year'].astype(str) + '-' + NA_All['Seat']
NA_Less['Party'] = NA_All['Party']
NA_Less['Votes'] = NA_All['Votes']
NA_Less['Year'] = NA_All['Year']


# In[ ]:


CombinedDF = NA_Less
CombinedDF.shape


# In[ ]:


CombinedDF = CombinedDF.sort_values(by=['Seat','Year'])
CombinedDF.shape


# ### I had to come up with some logic in order to consider all 'Independent' candidates as one entry per Constituency. 
# ### Therefore I sum up votes of all Independent candidates and call it Independent party per Contituency

# In[ ]:


CombinedDF2 = CombinedDF.groupby(['Seat', 'Party'])['Votes'].sum().reset_index()
#CombinedDF2


# In[ ]:


len(CombinedDF2['Seat'].unique())


# ### Adding Indexing for faster search and lookup

# In[ ]:


#CombinedDF2.reset_index()
CombinedDF3 = CombinedDF2.set_index(['Seat','Party'])
print(CombinedDF3.index.names)


# ### creating list of unique seat names and party names for heatmap entry and visualization

# In[ ]:


seat_names=CombinedDF3.index.levels[0]
party_names=CombinedDF3.index.levels[1]


# ### Creating an empty matrix for storing heatmap data

# In[ ]:


#matrix = pd.DataFrame(index=seat_names,columns=party_names)
#matrix.shape
#matrix.iloc[0,0]=50
#matrix
matrix = np.zeros((len(seat_names),len(party_names)))


# In[ ]:


#CombinedDF3.loc[('NA-1-2002-PESHAWAR-I','Muttahidda Majlis-e-Amal Pakistan')].item()


# ### inserting values of votes in each constitueny for each party - takes couple of minutes to finish

# In[ ]:


for s in range(0,len(seat_names)):
    for p in range (0,len(party_names)):
        try:
            matrix[s,p] = CombinedDF3.loc[(seat_names[s], party_names[p])].item()
            #matrix.iloc[s,p] = CombinedDF3.loc[(seat_names[s], party_names[p])].item()
        except KeyError:
            continue
        #without loopup - very slow
        #matrix[s,p] = CombinedDF2.loc[(CombinedDF2['Seat'] == seat_names[s]) & (CombinedDF2['Party'] == party_names[p]), ['Votes']]


# ### Below is the heatmap of total votes in each constituency for each party. Yellow bright color represents large number of votes where as dark green shows small number of votes
# 
# ### It would be nice if we can export this heatmap as image or pdf to make it a poster. I did not spend much time on it but there should be some way to do it. Happy Coding !

# In[ ]:


matrix2 = pd.DataFrame(matrix, index=seat_names, columns=party_names)
#matrix2 = matrix
#matrix2 = matrix2.fillna(0)
#matrix2 = matrix2.astype(int)
matrix2.style.background_gradient(cmap='summer',axis=1)

