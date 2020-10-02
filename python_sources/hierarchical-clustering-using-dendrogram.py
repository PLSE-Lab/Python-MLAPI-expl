#!/usr/bin/env python
# coding: utf-8

# **Unsupervised Learning : Hierarchical Clustering**
# 
# As the name suggests its an algorithm that builds hierarchy of clusters
# 
# Dataset :
# 
# I took a dataset for Indian Premier League 2018 edition's tournament, 
# which we play in our office for fun, where in every participant need to predict the winner of each match. 
# It ends with bottom Y people treating top X :)
# 
# Goal :
# 
# Form cluster(s) of all the particpating members based on their voting pattern [ Total 60 matches]
# 
# Implementation :
# 
# Hierarchical Clustering using Dendrogram [ Unsupervised Learning visualization, without needing to specify the number of clusters]
# 
# Code :
# 
# Pardon the documentation, this was written on a lazy Friday afternoon!
# 
# Outcome :
# 
# I could see that many of the members in the initial clusters are from same team,which meant they
# discussed their votes before predicting and voted same for most number of matches 
# ( Against the tournament rules :D)
# 

# In[ ]:


#Import neccessary libraries

import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
from sklearn.preprocessing import normalize


# In[ ]:


#Importing the data into pandas dataframe
ipl=pd.read_excel('../input/IPL11_FINAL.xlsx')


# In[ ]:


#Analyze the dataframe
ipl.head()
#ipl.shape
#ipl.columns


# In[ ]:


#Select only required columns
ipl1=ipl.loc[:,['Match#', 'Date', 'Time (IST)', 'Match', 'WINNER', 'Venue', 'Anand',
       'Paris', 'Megha', 'Veni', 'Priya', 'Ashok', 'G3R', 'Madhan', 'Raj',
       'Thimma', 'Surajit', 'Naveen', 'Raghu', 'Manasa', 'Anil', 'Sridhar',
       'Sanjeeth', 'Anand P', 'Diwakar', 'Manish', 'Murali']]

#Same can be achieved using iloc as shown below
#samp=ipl.iloc[:,:27]
#samp.tail(5)


# In[ ]:


#Select required rows only

ipl1=ipl1.iloc[:60,:] # Final match was 60th which was played on 27th may 2018


# In[ ]:



ipl1.columns


# In[ ]:


#Create a dictionary tag a number to corresponding team

dict= {'RCB':1,'CSK':2,'MI':3,'SRH':4,'KKR':5,'DD':6,'RR':7,'KXI':8}


# In[ ]:


ipl2= ipl1.iloc[:,6:].replace(dict)

ipl2= ipl2.transpose()


# In[ ]:


ipl2.tail(5)


# In[ ]:


#Extract the names, to be used as labels 
names=list(ipl2.index.values)

names


# In[ ]:


# Calculate the linkage: mergings
mergings = linkage(ipl2,method='complete')


# Plot the dendrogram, using names as labels
dendrogram(mergings,
           labels=names,
           leaf_rotation=60,
           leaf_font_size=10,
)
plt.title('Hierarchical Clustering')
plt.figure(figsize=(200, 60))
plt.show()


# In[ ]:




