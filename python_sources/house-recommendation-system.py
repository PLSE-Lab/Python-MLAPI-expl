#!/usr/bin/env python
# coding: utf-8

# # Recommendation System

#  Imports

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import os
os.listdir('../input/')


# Reading datasets

# In[ ]:


prop = pd.read_json('../input/property-data/Property.json')
user = pd.read_json('../input/property-data/User.json')
prop.shape, user.shape


# In[ ]:


user.head()


# In[ ]:


prop.head()


# ## EDA

# In[ ]:


prop['price'].head()


# You can see that the price column in **string** datatype, also contains ( $ ) and ( , ) punctualtions. 
# 
# **Task** : Normalizing **$308,711.19** to **308711.19** and **str** to **float**

# In[ ]:


def removo_punc(row):
    return float(''.join(str(row).split(','))[1:])

prop['price'] = prop['price'].astype(str).map(removo_punc)
prop['price']


# In[ ]:


sns.countplot(prop['bedroom'])


# In[ ]:


sns.countplot(prop['bathroom'])


# **'bathroom'** and **'badroom'** does not require any changes. 

# In[ ]:


sns.scatterplot(prop['latitude'], prop['longitude'], alpha=0.7)


# Let's Explore **'tags'** feature

# In[ ]:


prop['tags'].head()


# values are in **list** form.  
# **Task** : encoding to **One Hot** form

# In[ ]:


def remove_space(row):
    return ['_'.join(i.split(' ')) for i in row]

prop['tags'] = prop['tags'].map(remove_space)

set1 = set()
for i in prop['tags']:
    for j in i:
        set1.add(j)
print(len(set1))
print(set1)


# **8** unique tags.

# Mapping **tag** name to **0-1** form

# In[ ]:


def tag_one_hot(row):
    dict1 = {'Bathrooms':0, 'Bedrooms':0,'Living_rooms':0, 'Location':0, 'Picture':0, 'Price':0, 'Schools':0, 'Size_of_home':0}
    for i in row:
        dict1[i] = 1
    return [i for i in dict1.values()]
tags = prop['tags'].map(tag_one_hot)

tag_cols = ['Bathrooms', 'Bedrooms','Living_rooms', 'Location', 'Picture', 'Price', 'Schools','Size_of_home']
tag_data = pd.DataFrame(tags.tolist(), columns=tag_cols)
tag_data.head()


# Removing **'picture'** and **'address'** because its not useful much for this case.

# In[ ]:


prop = prop.drop(['tags','_id','picture','address'],axis=1)

final_prop = pd.concat((prop, tag_data),axis=1)
final_prop.head()


# We got clean dataset,lets build model

# ## Modeling

# ### Using Cosine Similarity ([formulla](http://https://www.google.com/imgres?imgurl=https%3A%2F%2Fneo4j.com%2Fdocs%2Fgraph-algorithms%2Fcurrent%2Fimages%2Fcosine-similarity.png&imgrefurl=https%3A%2F%2Fneo4j.com%2Fdocs%2Fgraph-algorithms%2Fcurrent%2Fexperimental-algorithms%2Fcosine%2F&docid=AAABX5A8IsaypM&tbnid=0E6-Qrb_6Rcw5M%3A&vet=10ahUKEwjavcPdv_jkAhWKqI8KHb-6DY8QMwhNKAIwAg..i&w=800&h=208&bih=625&biw=1366&q=cosine%20similarity%20formula&ved=0ahUKEwjavcPdv_jkAhWKqI8KHb-6DY8QMwhNKAIwAg&iact=mrc&uact=8))

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
final_prop = scaler.fit_transform(final_prop)


# we have to scale the data before appling cosine_similarity.

# * 1.0 = more Similar
# * 0.0 = no similar

# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity

cosine_simi = cosine_similarity(final_prop)
plt.figure(figsize=(20,20))
sns.heatmap(cosine_simi)


# storing all cosine similaries values to dataframe

# In[ ]:


cosine_simi = pd.DataFrame(cosine_simi, columns = [i for i in range(100)], index = [i for i in range(100)])
cosine_simi.head()


# ### Prediction(recommendations)

# we need only **userSaveHomes** to find most similar property

# In[ ]:


user.head()


# below function calculates the similarity between userSaveHome and all Properties,and returning the property which has high similar value.

# In[ ]:


def recommendations(row):
    props = {}
    for i in row:
        props[cosine_simi[i].sort_values(ascending=False).index[1]] = cosine_simi[i].sort_values(ascending=False)[1]
    return [i for i,j in props.items() if (j>0.5) & (i not in row)]


# saving Recommended properties to **'Recommendation'** column.

# In[ ]:


user['Recommendation'] = user.userSaveHomes.map(recommendations)
user.head()

