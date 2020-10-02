#!/usr/bin/env python
# coding: utf-8

# #                                                 Different cuisines around the globe

# ![](https://www.metro.ca/userfiles/image/recettes-occasions/dossiers-speciaux/cuisine-du-monde/header-cuisine-monde.jpg)

# we have given some csv files so lets check what are they

# In[ ]:


import os
print(os.listdir("../input"))


# In[ ]:


import pandas as pd
import numpy as np


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


train = pd.read_json('../input/train.json')
train.head()


# In[ ]:


test = pd.read_json('../input/test.json')
test.head()


# In[ ]:


sub = pd.read_csv('../input/sample_submission.csv')
sub.head()


# In[ ]:


train.set_index('id',inplace=True)


# In[ ]:


train.head()


# Cuisine, the foods and methods of preparation traditional to a region or population. The major factors shaping a cuisine are climate, which in large measure determines the native raw materials that are available to the cook; economic conditions, which regulate trade in delicacies and imported foodstuffs; and religious or sumptuary laws, under which certain foods are required or proscribed.
# 
# Climate also affects the supply of fuel; the characteristic Chinese food preparation methods, in which food is cut into small pieces before being cooked, was shaped primarily by the need to cook food quickly to conserve scarce firewood and charcoal.

# In[ ]:


temp = train.cuisine.value_counts()
temp


# In[ ]:


plt.figure(figsize=(20,5))
sns.countplot(train.cuisine)
plt.show()


# so most of the dishes are belongs to italian cuisine,yay i love pizza. mexican and southern-us follows the 2nd and 3rd rank. 

# club 3rd and 4th row of indine cuisine and check what are the ingridents

# In[ ]:


(train.iloc[3,1]+train.iloc[4,1])


# In[ ]:


from collections import Counter
top_n = Counter([item for sublist in train.ingredients for item in sublist]).most_common(20)
top_n


# In[ ]:


ingredients = pd.DataFrame(top_n,columns=['ingredient_name','cnt'])
ingredients


# In[ ]:


plt.figure(figsize=(20,20))
sns.barplot(x = ingredients.cnt,y = ingredients.ingredient_name)
plt.show


# `ha ha... every cuisine must add salt, i know chinese people dont like salt much, lets check that

# In[ ]:


temp1 = train[train['cuisine'] == 'chinese']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()


# see **salt ** is in 3rd place for chinese dishes, ok lets check indian ingredients

# In[ ]:


temp1 = train[train['cuisine'] == 'indian']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()


# to be frank i love comb of onions and salt,  we use garam masala alot in both north and south india states, olive oil we use very less often mostly used in restaurants and hotels, so now we will check for which cuisines olive oil takes the top place

# In[ ]:


temp1 = train[train['cuisine'] == 'italian']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()


# In italy or in italian dishes they use olive oil often, garlic next to live oil. oh we havent checked mexican cuisine yet which bags 2nd place in total num of cuines, so lets check

# In[ ]:


temp1 = train[train['cuisine'] == 'mexican']
n=6714 # total ingredients in train data
top = Counter([item for sublist in temp1.ingredients for item in sublist]).most_common(n)
temp= pd.DataFrame(top)
temp.columns = ['ingredient','total_count']
temp = temp.head(20)
plt.figure(figsize=(20,20))
sns.barplot(y = temp.ingredient,x=temp.total_count)
plt.show()


# they also use onions alot like indian cuisine dishes, it would be much better if the dataset provides list of dishes for the  respective cuisines

# till now we have seen distribution of different ingredients in different  cuisines. 
# - further we will convert text data into number format using different word embending techniques and try to develop the model

# **If you like it please upvote for me **
# 
# Thank you : )
