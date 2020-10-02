#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


df = pd.read_csv('mushrooms.csv')


# In[ ]:


df.shape


# In[ ]:


df.head().T


# ### Let's rename our class column, y-variable, to avoid any confusion later on down the road.

# In[ ]:


df.rename(index=str, columns={'class':'e_or_p'}, inplace=True)


# ### Let's check for null values.  There are none!  Wow!

# In[ ]:


df.isnull().sum()


# ### These are all objects, which makes sense since they are all letters.  The first column, class, is our y-variable.  This column represents whether or not a mushroom is poisonous (p) or edible (e).

# Attribute Information: 
#     
# e_or_p: edible=e, poisonous=p
# 
# cap-shape: bell=b,conical=c,convex=x,flat=f, knobbed=k,sunken=s
# 
# cap-surface: fibrous=f,grooves=g,scaly=y,smooth=s
# 
# cap-color: brown=n,buff=b,cinnamon=c,gray=g,green=r,pink=p,purple=u,red=e,white=w,yellow=y
# 
# bruises: bruises=t,no=f
# 
# odor: almond=a,anise=l,creosote=c,fishy=y,foul=f,musty=m,none=n,pungent=p,spicy=s
# 
# gill-attachment: attached=a,descending=d,free=f,notched=n
# 
# gill-spacing: close=c,crowded=w,distant=d
# 
# gill-size: broad=b,narrow=n
# 
# gill-color: black=k,brown=n,buff=b,chocolate=h,gray=g, green=r,orange=o,pink=p,purple=u,red=e,white=w,yellow=y
# 
# stalk-shape: enlarging=e,tapering=t
# 
# stalk-root: bulbous=b,club=c,cup=u,equal=e,rhizomorphs=z,rooted=r,missing=?
# 
# stalk-surface-above-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-surface-below-ring: fibrous=f,scaly=y,silky=k,smooth=s
# 
# stalk-color-above-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# stalk-color-below-ring: brown=n,buff=b,cinnamon=c,gray=g,orange=o,pink=p,red=e,white=w,yellow=y
# 
# veil-type: partial=p,universal=u
# 
# veil-color: brown=n,orange=o,white=w,yellow=y
# 
# ring-number: none=n,one=o,two=t
# 
# ring-type: cobwebby=c,evanescent=e,flaring=f,large=l,none=n,pendant=p,sheathing=s,zone=z
# 
# spore-print-color: black=k,brown=n,buff=b,chocolate=h,green=r,orange=o,purple=u,white=w,yellow=y
# 
# population: abundant=a,clustered=c,numerous=n,scattered=s,several=v,solitary=y
# 
# habitat: grasses=g,leaves=l,meadows=m,paths=p,urban=u,waste=w,woods=d

# ### Let's go straight into our train/test splits and the explore our train set.

# In[ ]:


train, test = train_test_split(df, test_size=.3, random_state=123, stratify=df[['e_or_p']])


# In[ ]:


train.shape


# In[ ]:


test.shape


# In[ ]:


train.apply(lambda x: x.value_counts()).T.stack()


# ## Hypothesis 1: Odor can be a strong indicator of poisonous or edible.  Fishy (y), foul (f), or spicy (s) may contain more poisonous mushrooms.

# In[ ]:


train.groupby('odor')['e_or_p'].value_counts()


# ### First off: WOW!  It appears that odor is a very useful feature in distinguishing between poisnous and edible as all but one odor is classified in only one class.
# ### Second, H1 is proven to be true.
# ### Let's take a deeper dive into mushrooms with no odor to find what differences we can see between the edible and the 74 poisonous varieties.

# ### We can now see that given there is no odor and the cap-shape is conical, the mushroom is poisonous.  Additionally, given no odor and the cap-shape is sunken, it is edible.

# In[ ]:


train[train.odor == 'n'].groupby('cap-shape')['e_or_p'].value_counts()


# ### Similar to above, given the mushroom has no odor, if it's habitat is a meadow, it's poisonous. If it's habitat is path, urban, or waste (yummy!), it's edible.  Are you really going to eat a mushroom found in waste though?

# In[ ]:


train[train.odor == 'n'].groupby('habitat')['e_or_p'].value_counts()


# ### Gill-color is very helpful.  Given the mushroom is odorless, there are multiple sub-categories that take the guess work out of whether or mushroom is edible or poisonous.

# In[ ]:


train[train.odor == 'n'].groupby('gill-color')['e_or_p'].value_counts()


# ### Let's combine the use of gill-color and then habitat on the odorless mushrooms that we have.  It's really starting to narrow down as we can see.

# In[ ]:


train[train.odor == 'n'].groupby(['gill-color', 'habitat'])['e_or_p'].value_counts()


# ### Let's now go to our visualizations notebook.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




