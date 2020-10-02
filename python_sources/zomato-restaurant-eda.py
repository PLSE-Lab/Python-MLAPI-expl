#!/usr/bin/env python
# coding: utf-8

# # Zomato Restaurant EDA

# In this kernel we will do some basic data exploration for Zomato Restaurant Dataset. We will finish this task in some basic steps.
# 
# 1. **Importing the dataset**: Loading the required libraries and dataset
# 2. **Cleaning the dataset**: We will clean the data by removing all the inconsistencies
# 3. **Understanding the features**: After we have cleaned up our data we will then try to understand our features and we will also drop the features we don't need.
# 4. **Visualisations**: After that we will visualize our data to get some insights out of it and to understand the relationships between the different features.
# 
# So, let's get started

# ### 1: Importing the dataset

# In[ ]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[ ]:


df = pd.read_csv('../input/zomato.csv')
df.head()


# ### 2: Cleaning the dataset

# In[ ]:


# Cleaning the rate feature
data = df
data.rate = data.rate.replace("NEW", np.nan)
data.dropna(how ='any', inplace = True)


# In[ ]:


# converting it from '4.1/5' to '4.1'
data['rate'] = data.loc[:,'rate'].replace('[ ]','',regex = True)
data['rate'] = data['rate'].astype(str)
data['rate'] = data['rate'].apply(lambda r: r.replace('/5',''))
data['rate'] = data['rate'].apply(lambda r: float(r))


# In[ ]:


# Converting into integer
data['approx_cost(for two people)'] = data['approx_cost(for two people)'].str.replace(',','')
data['approx_cost(for two people)'] = data['approx_cost(for two people)'].astype(int)


# ### 3: Understanding the features

# In[ ]:


data.head()


# So as we can see from the above data is now cleaned and we have removed all of our inconsistencies. But this dataset have some features that we don't need. These features do not give us any useful information. So we will remove them.
# 
# * **url**: We don't need the url in our analysis since it doesn't gives us any good insights about the data.
# * **address and phone**: We will delete them as well for reason mentioned above.
# * **listed_in(city)**: This feature is same as **location** feature and we need just one of these, so we will delete **listed_in(city)**.

# In[ ]:


del data['url']
del data['address']
del data['phone']
del data['listed_in(city)']


# In[ ]:


# Renaming features for convenience
data.rename(columns={'approx_cost(for two people)': 'approx_cost','listed_in(type)': 'type'}, inplace=True)


# In[ ]:


data.head()


# ### 4: Visualisations

# Now we will do some visualisations to better understand the features and their relations with each other. In this section we will first start with individual features. We will explore them and get insights out of them. After we have done that we will move on to understanding how these features relate to each other.

# #### Online Order

# From the plot below we can see that there is a significantly more number of restaurants that offer online ordering service than those who don't.

# In[ ]:


_ = sns.countplot(data['online_order'])


# #### Table Booking

# In[ ]:


_ = sns.countplot(data['book_table'])


# #### Ratings

# Plot below gives us the distribution of the rate feature. We can see that most of restaurants are rated between 3.8 to 4.2.

# In[ ]:


_ = sns.distplot(data['rate'], color="m")


# #### Votes

# In[ ]:


_ = sns.distplot(data['votes'], hist=False, color="c", kde_kws={"shade": True})


# #### Locations

# In[ ]:


plt.figure(figsize=(15, 10))
p = sns.countplot(data['location'])
_ = plt.setp(p.get_xticklabels(), rotation=90)


# #### Rest Type

# In[ ]:


plt.figure(figsize=(15, 10))
p = sns.countplot(data['rest_type'])
_ = plt.setp(p.get_xticklabels(), rotation=90)


# #### Approx Cost

# Here we can see the distribution plot of the approx cost for two people in these restaurants.

# In[ ]:


_ = sns.distplot(data['approx_cost'], hist=False, color="g", kde_kws={"shade": True})


# #### Type

# Here we have plotted a basic pie chart to see the variation in the types of restaurants.

# In[ ]:


count = data['type'].value_counts().sort_values(ascending=True)
slices = [count[6], count[5], count[4], count[3], count[2], count[1], count[0]]
labels = ['Delivery ', 'Dine-out', 'Desserts', 'Cafes', 'Drinks & nightlife', 'Buffet', 'Pubs and bars']


# In[ ]:


plt.figure(figsize=(20, 10))
_ = plt.pie(slices, labels=labels, autopct='%1.1f%%', shadow=True, startangle=90)


# So far we have been analysing the isolated features and they have given us good results. But to get more deeper understanding of the data we need to understand the relationship between these features.

# #### Average Approximate Cost for each city

# In[ ]:


cost_count = {}

for idx, d in data.iterrows():
    if d['location'] not in cost_count:
        cost_count[d['location']] = [d['approx_cost']]
    else:
        cost_count[d['location']].append(d['approx_cost'])


# In[ ]:


avg_cost = {}

for key in cost_count.keys():
    avg_cost[key] = sum(cost_count[key])/len(cost_count[key])


# From the plot below we can easily tell that the Restaurants in Sankey road have a higher average approximate cost for two people as compared to restaurants in other locations

# In[ ]:


fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

p = sns.scatterplot(x=list(avg_cost.keys()), y=list(avg_cost.values()), color='r', ax=axs[0])
q = sns.barplot(x=list(avg_cost.keys()), y=list(avg_cost.values()), ax=axs[1])

_ = plt.setp(p.get_xticklabels(), rotation=90)
_ = plt.setp(q.get_xticklabels(), rotation=90)


# #### Approximate Cost, Votes and Ratings

# In[ ]:


fig, axs = plt.subplots(2, 1, figsize=(20, 10), sharex=True)
p = sns.scatterplot(x=data['approx_cost'], y=data['votes'], hue=data['rate'], ax=axs[0])
q = sns.scatterplot(x=data['approx_cost'], y=data['rate'], size=data['votes'], sizes=(10, 100), ax=axs[1])


# #### Type and Ratings

# In[ ]:


len(data['type'].unique())
types = data['type'].unique()


# In[ ]:


types_rating = {}

for idx, d in data.iterrows():
    if d['type'] not in types_rating:
        types_rating[d['type']] = [d['rate']]
    else:
        types_rating[d['type']].append(d['rate'])


# The following plot shows the rating distribution for restaurants of all different types.

# In[ ]:


fig, axs = plt.subplots(3, 3, figsize=(15, 10), sharex=True)

p = sns.distplot(types_rating[types[0]], hist=False, color="g", kde_kws={"shade": True}, ax=axs[0, 0])
_ = p.set_title(types[0])
q = sns.distplot(types_rating[types[1]], hist=False, color="r", kde_kws={"shade": True}, ax=axs[0, 1])
_ = q.set_title(types[1])
r = sns.distplot(types_rating[types[2]], hist=False, color="b", kde_kws={"shade": True}, ax=axs[0, 2])
_ = r.set_title(types[2])
s = sns.distplot(types_rating[types[3]], hist=False, color="c", kde_kws={"shade": True}, ax=axs[1, 0])
_ = s.set_title(types[3])
t = sns.distplot(types_rating[types[4]], hist=False, color="y", kde_kws={"shade": True}, ax=axs[1, 1])
_ = t.set_title(types[4])
u = sns.distplot(types_rating[types[5]], hist=False, color="m", kde_kws={"shade": True}, ax=axs[1, 2])
_ = u.set_title(types[5])
r = sns.distplot(types_rating[types[6]], hist=False, color="c", kde_kws={"shade": True}, ax=axs[2, 0])
_ = r.set_title(types[6])

