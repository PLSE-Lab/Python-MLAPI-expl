#!/usr/bin/env python
# coding: utf-8

# ## 2019 Kaggle ML & DS Survey:

# ### This is an incomplete notebook : (Version 1)

# I am more interested on knowing what Data Science aspirants have in common.

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


mcr = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
qs = pd.read_csv('/kaggle/input/kaggle-survey-2019/questions_only.csv')
ss = pd.read_csv('/kaggle/input/kaggle-survey-2019/survey_schema.csv')
otr = pd.read_csv("/kaggle/input/kaggle-survey-2019/other_text_responses.csv")


# In[ ]:


print(mcr.shape, qs.shape, ss.shape, otr.shape)


# In[ ]:


mcr.head()


# In[ ]:


qs.head()


# In[ ]:


ss.head()


# In[ ]:


otr.head()


# The other text responses are shuffled thus cannot be used for ML models. Even survey schema does not give any important info.

# In[ ]:


# Null values in mcr data
nul = mcr.isna().sum().sum()
tot = mcr.shape[0]*mcr.shape[1]
percent = 1-(nul/tot)
print(percent*100)


# 25 % of the data is missing. We need to fill them with an outlier.

# In[ ]:


# type of data in columns
for i in mcr.columns:
    print(i, mcr[i].dtype)


# All columns are categorical. We can easily fill with any outlier and use label encoding.

# In[ ]:


mcr = mcr.fillna('-999')


# In[ ]:


mcr.isna().sum().sum()


# In[ ]:


# label encoding
X_train = mcr.copy()
X_train = X_train.drop('Time from Start to Finish (seconds)', axis = 1)
from sklearn import preprocessing
# Label Encoding
for f in X_train.columns:
    if (X_train[f].dtype=='object'): 
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(X_train[f].values))
        X_train[f] = lbl.transform(list(X_train[f].values))


# In[ ]:


X_train.head()


# In[ ]:


from sklearn.cluster import KMeans
lst_inertia = []
lx = [2,5,10,15, 20, 25, 30, 40, 50]
for kc in lx:
    kmean = KMeans(n_clusters = kc).fit(X_train)
    lst_inertia.append(kmean.inertia_)
    print("n_clusters = {} done!".format(kc))


# In[ ]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(lx,lst_inertia)
plt.scatter(lx,lst_inertia, label = 'points')

plt.legend()
plt.xlabel("n_clusters")
plt.ylabel("inertia")
plt.title("Tuning n_clusters")
plt.grid()
plt.show()


# In[ ]:


best_n_clusters = 10
kmeans = KMeans(n_clusters = best_n_clusters).fit(X_train)
cluster_label = kmeans.labels_


# In[ ]:


for j in range(best_n_clusters):
    cnt=0
    for i in range(len(X_train)):
        if cluster_label[i]==j and cnt<4:
            print(i,cluster_label[i], mcr[1:].iloc[i])
            cnt+=1
            print("="*50)


# In[ ]:


# Visualising through wodcloud
mcr1 = pd.read_csv('/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv')
X_new = mcr1[:].copy()
X_new["cluster_label"] = cluster_label
X_new = X_new.sort_values(by = ["cluster_label"])
X_new = X_new.drop('Time from Start to Finish (seconds)', axis = 1)


# In[ ]:


X_new.head()


# In[ ]:


# No. of points in each cluster
index_label = []
cnt = 0
for i in range(len(X_new)):
    if X_new["cluster_label"][i] == cnt and cnt<best_n_clusters:
        print("cluster {} starts at index {}".format(cnt, i))
        index_label.append(i)
        cnt+=1


# ### Let us see what info does these clusters contain.

# In[ ]:


word_corpus=[]
for i in X_new.columns:
    for j in list(X_new[i].values):
        word_corpus.append(j)


# In[ ]:


from wordcloud import WordCloud
word_corpus = []
def wrdcld(points):
    word_corpus = ' '
    for col in X_new.columns:
        #print(col)
        if col!='cluster_label':
            for j in list(X_new[col][points].values):
                j = str(j)
                tokens = j.split()
                for word1 in tokens:
                    if word1 != 'nan':
                        if word1 not in word_corpus:
                            word_corpus = word_corpus+word1+' '
                #print(j)
    wordcld = WordCloud(width = 1600, height = 1600,
                       background_color = 'white').generate(word_corpus)
    
    plt.imshow(wordcld)
    plt.axis("off")
    plt.show()


# In[ ]:


index_label.append(len(X_new)-1)
index_label


# In[ ]:


for i in range(len(index_label)-2):
    points = [i for i in range(int(index_label[i]), int(index_label[i+1]))]
    print("WordCloud for Cluster: ", i+1)
    wrdcld(points)


# # The wordclouds do not give much information. We can observe that different clusters have different number of instances of each answer. Words 'AI', 'data', 'new' and 'forums' appears in almost all clusters. Some have more frequency of word 'Software' and some have 'Research'. This is just a visualization using KMeans clustering. I intend to use this with other EDA techniques.

# I am publishing this notebook as it is now and will be looking for out of box EDA techniques.

# ### The inspiration behind using KMeans was the thought where I wanted to define a Machine Learning model for this Survey. I was thinking of rating users based on their responses by clustering them into groups and giving ratings based on wordclouds, but this is incomplete. I will do more.

# #### I hope that this approach will trigger some interesting notebooks 

# In[ ]:




