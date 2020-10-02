#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_googleplaystore = pd.read_csv('../input/googleplaystore.csv')


# #### We need to prepare data
# We often meet mistakes and strange coloms and rows in data sets
# It is better change data and arrange all information for the best analisis

# In[ ]:


data_googleplaystore[(data_googleplaystore['App'] == 'Life Made WI-Fi Touchscreen Photo Frame')] #strange row


# In[ ]:


data_googleplaystore.drop(data_googleplaystore[(data_googleplaystore['App'] == 'Life Made WI-Fi Touchscreen Photo Frame')].index, inplace=True)


# In[ ]:


data_googleplaystore['Reviews'] = data_googleplaystore['Reviews'].fillna(0).astype(int) #change type of Reviews 


# In[ ]:


data_googleplaystore.info() #check


# In[ ]:


data_googleplaystore.sort_values(by='Reviews', ascending=False).head()#duplicates!


# In[ ]:


data_googleplaystore.drop_duplicates('App', keep='last', inplace=True) #drop_duplicates


# In[ ]:


data_googleplaystore.sort_values(by='Reviews', ascending=False).head() 


# In[ ]:


data_googleplaystore['Category'].value_counts()


# In[ ]:


data_googleplaystore['Category'].value_counts().head(10).plot(kind='bar')


# In[ ]:


data_googleplaystore.groupby(['Category','Genres'], as_index=False)['Reviews'].sum().sort_values(by='Category', ascending=False)


# #### Let's get some information about categories

# In[ ]:


data_googleplaystore.groupby(['Category'], as_index=False)['Reviews'].sum().sort_values(by='Reviews', ascending=False).head(10)


# In[ ]:


data_googleplaystore.groupby(['Category'], as_index=False).mean().sort_values(by='Rating', ascending=False).head(10)


# ##### As a conclusion, 
# category Event has the best mean score, 
# but category Social has the most sizeable quantity of reviews

# In[ ]:


data_category = data_googleplaystore.groupby(['Category'], as_index=False).mean()
data_category.head()


# In[ ]:


plt.figure(figsize = (10,5))
x = data_category.Rating
y = data_category.Reviews
z = data_category.Category
rng = np.random.RandomState(0)
colors = rng.rand(33)
sizes = 500 * rng.rand(33)

plt.scatter(x, y, c=colors, s=sizes, alpha = 0.5, cmap='magma')
plt.xlabel("Rating")
plt.ylabel("Reviews")
for i, j in zip(x, y):
    plt.text(i, j, '%.1f' % i, ha='center', va='bottom')

plt.colorbar()
plt.show()


# In[ ]:


plt.figure(figsize = (15,7))

def list_for_visio(data_row):
    list_of_smth = []
    for i in data_row:
        list_of_smth.append(i)
    return list_of_smth


categories = list_for_visio(data_category.Category)
ratings = list_for_visio(data_category.Rating)
reviews = list_for_visio(data_category.Reviews)

for j, category in enumerate(categories):
    x = ratings[j]
    y = reviews[j]
    plt.scatter(x, y,  c='green', s=400, marker='H', alpha = 0.3)
    plt.text(x, y, category, fontsize=10, horizontalalignment='center', family='monospace', color='black', rotation=15)

plt.xlabel("Rating")
plt.ylabel("Reviews")
plt.show()


# In[ ]:


data_googleplaystore.groupby(['Category'], 
                             as_index=False).get_group('DATING').sort_values(by='Reviews', 
                                                                             ascending=False).head()
#the worst category in mean rating and rewiews


# In[ ]:


data_googleplaystore.groupby(['Category'], 
                             as_index=False).get_group('EVENTS').sort_values(by='Reviews', 
                                                                             ascending=False).head()
#the best category in mean rating and rewiews


# ##### As a conclusion, 
# Comparing the categories with the best rating and the worst rating we can notice what
# Dating category has many times more reviewes than Events category.
# That can be a reason why Dating category has so low mean rating
# 
# According the next information, Dating category has almost free times more apps than Events category, but the most remarkable variete of apps has Family category.

# In[ ]:


data_googleplaystore.groupby(['Category']).size()


# In[ ]:


data_mm = data_googleplaystore.groupby(['Category']).agg([np.sum, np.mean, np.min, np.max])


# In[ ]:


data_mm.head(3) #nice, I doesn't help us at all)


# In[ ]:


data_mm['Reviews'].sort_values(by='sum',
ascending=False).head()
#interesting, but I want to see app+category!


# ##### Firstly, Top-10 apps with many reviews and rating > 4

# In[ ]:


pd.pivot_table(data_googleplaystore[data_googleplaystore['Rating'] > 4.0],
index=['Category', 'App'], aggfunc='mean').sort_values(by='Reviews',
ascending=False).head(10)


# ##### Secondly, Top-10 apps with best rating and reviews > 100 000

# In[ ]:


pd.pivot_table(data_googleplaystore[data_googleplaystore['Reviews'] > 100000], 
index=['Category', 'App'], 
aggfunc='mean').sort_values(by='Rating', 
ascending=False).head(10)


# ### Paid vs Free apps

# In[ ]:


data_googleplaystore['Type'].value_counts()


# In[ ]:


data_googleplaystore[data_googleplaystore['Type'] == 'Free'].mean()


# In[ ]:


data_googleplaystore[data_googleplaystore['Type'] == 'Paid'].mean()


# #### So, paid apps have better reting than free apps, but they have less reviews

# In[ ]:


data_free = data_googleplaystore[data_googleplaystore['Type'] == 'Free']
data_paid = data_googleplaystore[data_googleplaystore['Type'] == 'Paid']


# In[ ]:


data_paid['Installs'].value_counts()


# In[ ]:


data_free['Installs'].value_counts()


# In[ ]:


data_paid['Price'].value_counts().head(10)


# #### How many reviews have different types of payment

# In[ ]:


data_paid.groupby('Price').mean().dropna().sort_values(by='Reviews', ascending=False).head()


# #### What apps have more reviews and how much they cost?

# In[ ]:


data_paid.groupby(['Category',
'App', 'Price'], as_index=False)['Reviews',
'Rating'].mean().sort_values(by='Reviews', ascending=False).head(10)


# In[ ]:


import seaborn as sns


# In[ ]:


paid_visio = data_paid.groupby(['Category',
'App', 'Price'], as_index=False)['Reviews',
'Rating'].mean().sort_values(by='Reviews', ascending=False).head(100)


plt.figure(figsize=(8,5))
sns.scatterplot(x=paid_visio.Reviews, y=paid_visio.Rating, hue=paid_visio.Price)
plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=0.3)


# In[ ]:


free_visio = data_free.groupby(['Category',
'App', 'Price'], as_index=False)['Reviews',
'Rating'].mean().sort_values(by='Reviews', ascending=False).head(100)

plt.figure(figsize=(8,5))
sns.scatterplot(x=free_visio.Reviews, y=free_visio.Rating)
sns.scatterplot(x=paid_visio.Reviews, y=paid_visio.Rating, hue=paid_visio.Price)
plt.legend(bbox_to_anchor=(1, 1), loc=0, borderaxespad=0.3)


# In[ ]:




