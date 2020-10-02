#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# Importing libraries that we'll need for now. We'll import what we need as we go.

# In[ ]:


import pandas as pd
import numpy as np


# Loading the data onto a pandas dataframe

# In[ ]:


responses = pd.read_csv( "../input/responses.csv")


# In[ ]:


responses.head()


# Making another copy of the dataframe

# In[ ]:


df = pd.DataFrame(responses)


# Recoding the categorical features written in text to numerical one in case we use them later (Feature engineering)
# The +1 is being used to make sure the coding begins with 1 instead of 0 which is the default

# In[ ]:


df['House - block of flats'] = np.where(df['House - block of flats']=="block of flats",1,2)

df['Village - town'] = np.where(df['Village - town']=="village",1,2)

df['Left - right handed'] = np.where(df['Left - right handed']=="right handed",1,2)

df['Only child'] = np.where(df['Only child']=="yes",1,2)

df['Gender'] = np.where(df['Gender']=="male",1,2) #male=1, female=2

df['Internet usage'] = df['Internet usage'].astype('category')
df['Internet usage'] = df['Internet usage'].cat.codes + 1

df['Punctuality'] = df['Punctuality'].astype('category')
df['Punctuality'] = df['Punctuality'].cat.codes + 1

df['Lying'] = df['Lying'].astype('category')
df['Lying'] = df['Lying'].cat.codes + 1

df['Smoking'] = df['Smoking'].astype('category')
df['Smoking'] = df['Smoking'].cat.codes + 1

df['Alcohol'] = df['Alcohol'].astype('category')
df['Alcohol'] = df['Alcohol'].cat.codes + 1


# In[ ]:


df.head()


# Next 'Education' is dealt with. Randomly assigning the numbers is not the best way to go about. It is better to code the labels based on the level of education, starting from 1 for the lowest qualification and 6 for the highest

# In[ ]:


def cat_to_num(x):
    if x=="currently a primary school pupil":
        return 1
    if x=="primary schoool":
        return 2
    if x=="secondary school":
        return 3
    if x=="college/bachelor degree":
        return 4
    if x=="masters degree ":
        return 5
    if x=="doctorate degree ":
        return 6
    
    
df['Education'] = df['Education'].apply(cat_to_num)


# In[ ]:


df.head()


# Now we select only the personality traits, general views on life,  features for our subsequent work

# In[ ]:


df1 = pd.DataFrame(df)
gen_df = df1.iloc[:,76:133]


# We know that there are missing values in the entire data. Let's try and visualize them in the part we've just taken.

# In[ ]:


import missingno as msno

msno.matrix(gen_df)
msno.heatmap(gen_df)


# So we see that none of the invidual features have 30-60% missing values, in which case we'd have to remove that feature from consideration altogether. Here all the features can be retained. There are 57 features for gen_df.
# 
# Since all the features are being retained, it is desirable that the missing values be imputed with some suitable value, such that we face no problems in analysis later on. For the imputation, the neutral response of 3 is chosen to be ideal, in order to introduce as little error as possible by doing so.

# In[ ]:


def myMedianimpute(data):
    for i in data.columns:
        data[i] = data[i].replace('?',np.nan).astype(float)
        data[i] = data[i].fillna((data[i].median()))
    return data

myMedianimpute(gen_df)


# checking if the imputation worked

# In[ ]:


np.sum(gen_df.isna())


# We see that the it has worked just fine.
# 
# Moving on, let's try some visualizations (Exploratory data analysis)

# In[ ]:


df2 = pd.concat([gen_df,df1.Gender],axis=1)
df2.columns


# In[ ]:


len(gen_df.columns)


# In[ ]:


import seaborn as sns
import matplotlib.pyplot as plt

for i in  df2.columns:
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))
    sns.countplot(y=i, data=df2, ax=ax[0])
    sns.countplot(y=i, hue='Gender', data=df2, ax=ax[1])
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


# Some interesting observations can be made from these frequency plots grouped by gender. For example, men tend to get angry more often (laughs). 
# On an average men show more tendency to display most of the traits shown here, it seems. Which makes it clear that personalities can be quite different based on gender (this is ofcourse obvious).

# Let's if any correlations can be spotted and whether they help us detect personality types

# In[ ]:


import matplotlib.pyplot as plt
from scipy.stats import spearmanr

spr_corr = gen_df.corr(method='spearman')

fig = plt.figure(figsize=(25,10))
ax = fig.add_subplot(111)
cax = ax.matshow(spr_corr,cmap='coolwarm',vmin=-1,vmax=1)
fig.colorbar(cax)
ticks = np.arange(0,len(gen_df.columns),1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(gen_df.columns)
ax.set_yticklabels(gen_df.columns)
plt.show()


# We see that there are some significant correlations if a threshold of 0.3 is considered. So low (0.3) is taken because in the domain of psychology,
# correlated features with even low correlations may play a significant role in our final outcomes. In other words, statistical significance may not necessarily give a very accurate picture of actual psychological significance.

# Let's see the top correlations shall we ?

# In[ ]:


def redundant_pairs(df):
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def top_correlations(df, method, n):
    au_corr = df.corr(method = method).abs().unstack()
    labels_to_drop = redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    return au_corr[0:n]

top_corr = top_correlations(gen_df,'spearman',57)
top_corr


# On the whole, we see the stronger correlated features actually do make sense to be so (eg: workaholism and prioritising workload). Inspite of this correlations may not really help that much in finding groups in personality. A very interesting pair is the last one. With respect to the real world, it is indeed something to consider.

# Now we go into the modelling. Let's begin by importing the required libraries

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kmodes.kmodes import KModes


# I'm using KModes clustering here. The data is to be clustered with respect to the rows as I want all the features to  stay and cotribute towards finding personality types. But other than this I'll be doing something quite interesting as well, which 

# Next we convert our working dataframe into a numpy array

# In[ ]:


per = np.array(gen_df)


# Let's go clustering!

# In[ ]:


km = KModes(n_clusters=500,max_iter=1000,init='Huang',n_init=2,
            n_jobs=-1)

m1 = km.fit(per)
m1.cluster_centroids_


# In[ ]:


m1.cost_


# As we see, the initial cost is quite high. We'll need to do something about that. Now comes the interesting part. Let's be very clear about something. We cannot have 500 personality types (I've given initial number of clusters = 500). Let's see what happens if we give much lesser, say 6 (it is unlikely that the number of personality types that can be found for the given data will exceed 10 at most).

# In[ ]:


km = KModes(n_clusters=6,max_iter=1000,init='Huang',n_init=2,
            n_jobs=-1)

m0 = km.fit(per)
m0.cluster_centroids_
m0.cost_


# Now we see why initially I chose something as high as 500. We wish to reduce the cost function value and look at the final clusters. 
# Hopefully those final clusters will reveal persoanlity types that make some sense.

# But the way we reduce the value of the cost function will be quite interesting. It will leave great scope of automating the process that I'm going to use in order to find some optimum path. Let's see what I'm talking about

# In[ ]:


mdl1 = m1.cluster_centroids_
km1 = KModes(n_clusters=250,max_iter=1000,init='Huang',n_init=2,
            n_jobs=-1)

m2 = km1.fit(mdl1)
m2.cost_


# The second iteration itself reduces the cost from over 14000 to about 7000. 
# Let's keep going to see what happens, this time using the cluster centroids of the second as input for the third. We'll keep doing this until we reach an acceptable value of the cost function.

# In[ ]:


mdl2 = m2.cluster_centroids_
km2 = KModes(n_clusters=125,max_iter=1000,init='Huang',n_init=2,
            n_jobs=-1)

m3= km2.fit(mdl2)
m3.cost_


# Very interesting. Further decrease. But clearly it is still too high. Let's keep going to see what happens

# In[ ]:


mdl3 = m3.cluster_centroids_
km3 = KModes(n_clusters=62,max_iter=1000,init='Huang',n_init=2,
            n_jobs=-1)

m4= km3.fit(mdl3)
m4.cost_


# In[ ]:


mdl4 = m4.cluster_centroids_
km4 = KModes(n_clusters=31,max_iter=1000,init='Huang',n_init=2,
            n_jobs=-1)

m5 = km4.fit(mdl4)
m5.cost_


# Now let's do something else for a change

# In[ ]:


mdl4 = m4.cluster_centroids_
km4 = KModes(n_clusters=31,max_iter=1000,init='Cao',n_init=2,
            n_jobs=-1)

m5 = km4.fit(mdl4)
m5.cost_


# I have changed the initialisation method to 'Cao'. Why? let's see 

# In[ ]:


mdl5 = m5.cluster_centroids_
km5 = KModes(n_clusters=15,max_iter=1000,init='Cao',n_init=2,
            n_jobs=-1)

m6 = km5.fit(mdl5)
m6.cost_


# In[ ]:


mdl5 = m5.cluster_centroids_
km5 = KModes(n_clusters=15,max_iter=1000,init='Huang',n_init=2,
            n_jobs=-1)

m6 = km5.fit(mdl5)
m6.cost_


# We see the differences in the values of the cost function. Why so?
# 
# 'Huang' is a probabilistic method using the probabilities of the attributes to choose centroid points. As the clusters begin to form and take shape with each iteration, probablities of the individual attributes become less and less important. From here on 'Cao', which uses the density of distribution of the points to find the centroids makes mores sense. Thus 'Cao' begins to give better results. One may keep checking with both till one starts giving better results.

# Let's continue

# In[ ]:


mdl6 = m6.cluster_centroids_
km6 = KModes(n_clusters=7,max_iter=1000,init='Cao',n_init=2,
            n_jobs=-1)

m7 = km6.fit(mdl6)
m7.cost_


# In[ ]:


mdl7 = m7.cluster_centroids_
km7 = KModes(n_clusters=5,max_iter=1000,init='Cao',n_init=2,
            n_jobs=-1)

m8 = km7.fit(mdl7)
m8.cost_


# In[ ]:


mdl8 = m8.cluster_centroids_
km8 = KModes(n_clusters=4,max_iter=1000,init='Cao',n_init=2,
            n_jobs=-1)

m9 = km8.fit(mdl8)
m9.cost_


# This value is quite acceptable. We may stop here. 

# In[ ]:


mdl9 = m9.cluster_centroids_
km9 = KModes(n_clusters=3,max_iter=1000,init='Cao',n_init=2,
            n_jobs=-1)

m10 = km9.fit(mdl9)
m10.cost_


# We see that there is no further reduction, so 4 may be chosen as the optimum number of expected approximate personality types in the data we selected.
# At each iteration I reduced the number of clusters by 50% approx so that it is not arbitrary. Various combinations may be attempted ofcourse.
# Let's check out if we found any feasible personality-types/trait-groups in our final cluster.

# In[ ]:


mdl9 = m9.cluster_centroids_
mdl9


# Some very interesting stuff has come out:
# 
# 1.	In general, disorganised people lack awareness about their surroundings. They are not very sure what or when to do things. May be lazy. Often disinterested in as basic things as eating. Signs of frustration, selfishness visible. Get angry less often. This is one personality type. Lazy, confused and disorganized.
# 
# 2.	Reliability coincides with keeping promises. Socially outgoing people may have strong political opinions. Extroversion
# 3. People with strong sense of responsibility may be god believers. Get angry more often.Maybe another personality type.  
# 
# 4.	Those with less patience lie less often, but have low adaptation skills.
# 
# I may do more on this data in the future. 
# 
