#!/usr/bin/env python
# coding: utf-8

# # GENRE Similarity
# 
# Calculate the genre similarity and save it as pkl file.
# 
# ## Import library

# In[ ]:


import numpy as np
import pandas as pd


# ## FUNCTION Definition

# In[ ]:


def WordVectorSimilarity(ar1,ar2):
    d1 = []    
    for s1 in ar1:
        if s1 != "":
            s1 = s1.lower()
            d1.append(s1)
    d2 = []    
    for s2 in ar2:
        if s2 != "":
            s2 = s2.lower()
            d2.append(s2)
    
    intersect = list(set(d1) & set(d2))
    if len(intersect) == 0:
        count = 1
    else:
        count = len(intersect)
    return count

def ConstructMatrixGenre(data):
    index = data.iloc[:,0]
    genre = data.iloc[:,1]
    dfgenre = pd.DataFrame(0,index=index, columns=index)
    for i in genre.iteritems():
        idata = str(i[1]).split(';')
        for j in genre[0:int(i[0])+1].iteritems():
            jdata = str(j[1]).split(';')
            if i[0] == j[0]:
                dfgenre.iat[int(i[0]),int(j[0])] = -1
            else:
                dfgenre.iat[int(i[0]),int(j[0])] = WordVectorSimilarity(idata,jdata)
    mgenre = np.matrix(dfgenre)
    newgenre = mgenre + np.transpose(mgenre)
    dfnewgenre = pd.DataFrame(newgenre,index=index, columns=index)
    return dfnewgenre


# ## Calculate GENRE similarity

# In[ ]:


dgenre = pd.read_csv('../input/datagenre-all-share-new.csv',sep='|')
genre_matrix = ConstructMatrixGenre(dgenre)
print(genre_matrix.head(10))


# ## Save to temporary file

# In[ ]:


genre_matrix.to_csv('sim_genre.csv')

