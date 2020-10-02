#!/usr/bin/env python
# coding: utf-8

# ## EDA & WordCloud : Women's E-commerce Reviews
# 
# ### What key words should you consider when designing women's fashion?

# #### Import Data

# In[103]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[104]:


ecom = pd.read_csv("../input/Womens Clothing E-Commerce Reviews.csv", index_col=1)


# In[105]:


ecom.head()


# In[106]:


ecom.drop(columns='Unnamed: 0', inplace=True)


# In[107]:


ecom.head()


# In[108]:


print("There are total " + str(len(ecom.index.unique())) + " unique items in this dataset")


# #### What are the ages of shoppers?

# In[109]:


sns.set(style="darkgrid")
plt.figure(figsize= (14,5))
sns.distplot(ecom['Age'], hist_kws=dict(edgecolor="k")).set_title("Distribution of Age")


# #### How are the product ratings distributed?

# In[110]:


plt.figure(figsize= (14,5))
ax=sns.countplot(x='Rating', data=ecom)
ax.set_title("Distribution of Ratings", fontsize=14)

x=ecom['Rating'].value_counts()

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')


# #### How many reviews are there per product category?

# In[111]:


plt.figure(figsize= (14,5))
ax=sns.countplot(x='Department Name', data=ecom, order = ecom['Department Name'].value_counts().index)
ax.set_title("Reviews per Department", fontsize=14)
ax.set_ylabel("# of Reviews", fontsize=12)
ax.set_xlabel("Department", fontsize=12)

x=ecom['Department Name'].value_counts()

rects = ax.patches
labels = x.values
for rect, label in zip(rects, labels):
    height = rect.get_height()
    ax.text(rect.get_x() + rect.get_width()/2, height + 5, label, ha='center', va='bottom')


# In[112]:


ecom.info()


# In[113]:


ecom.isnull().sum()


# #### Remove all rows without textual reviews

# In[114]:


ecom.dropna(subset=['Review Text'], inplace=True)


# #### Add length of review as a separate column

# In[115]:


ecom['Length'] = ecom['Review Text'].apply(len)
ecom.head()


# #### Length of reviews per rating scale

# In[116]:


d = sns.FacetGrid(ecom, col='Rating')
d.map(plt.hist,'Length',bins=30)


# In[117]:


plt.figure(figsize= (14,5))
sns.boxplot(x='Rating', y='Length', data=ecom, palette='rainbow')


# ### Let's build some WordClouds!

# In[118]:


import re

def clean_data(text):
    letters_only = re.sub("[^a-zA-Z]", " ", text) 
    words = letters_only.lower().split()                            
    return( " ".join( words ))    


# In[119]:


from wordcloud import WordCloud, STOPWORDS
stopwords= set(STOPWORDS)|{'skirt', 'blouse','dress','sweater', 'shirt','bottom', 'pant', 'pants' 'jean', 'jeans','jacket', 'top', 'dresse'}

def create_cloud(rating):
    x= [i for i in rating]
    y= ' '.join(x)
    cloud = WordCloud(background_color='white',width=1600, height=800,max_words=100,stopwords= stopwords).generate(y)
    plt.figure(figsize=(15,7.5))
    plt.axis('off')
    plt.imshow(cloud)
    plt.show()


# ### Top Words in Rating = 5
# * love
# * color
# * great
# * comfortable
# * look
# * one
# * perfect
# 
# Words to consider when designing women's clothing: comfortable, soft, beautiful, well made, detail, design, versatile

# In[120]:


rating5= ecom[ecom['Rating']==5]['Review Text'].apply(clean_data)
create_cloud(rating5)


# ### Top Words in Rating = 4
# * love
# * color
# * fit
# * look
# 

# In[121]:


rating4= ecom[ecom['Rating']==4]['Review Text'].apply(clean_data)
create_cloud(rating4)


# ### Top Words in Rating = 3
# * look
# * fit
# * color
# * fabric
# * really
# * love
# 
# negative words detected: disappointed, unfortunately, return

# In[122]:


rating3= ecom[ecom['Rating']==3]['Review Text'].apply(clean_data)
create_cloud(rating3)


# ### Top Words in Rating = 2
# * look
# * fit
# * fabric
# * color
# 
# negative words detected: tight, disappointed, going back

# In[123]:


rating2= ecom[ecom['Rating']==2]['Review Text'].apply(clean_data)
create_cloud(rating2)


# ### Top Words in Rating = 1
# * look
# * fabric
# * color
# * fit
# 
# negative words detected: returned, cheap, going back

# In[124]:


rating1= ecom[ecom['Rating']==1]['Review Text'].apply(clean_data)
create_cloud(rating1)


# ### Thank you for reading! 
# #### Your feedback is very much appreciated! 

# In[ ]:




