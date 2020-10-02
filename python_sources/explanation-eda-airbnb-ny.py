#!/usr/bin/env python
# coding: utf-8

# ### Loading the required libraries..

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[ ]:


df=pd.read_csv('../input/new-york-city-airbnb-open-data/AB_NYC_2019.csv')


# In[ ]:


df.head()


# Let's see the description or the overview of complete data.

# In[ ]:


df.describe(include='all')


# In[ ]:


df.dtypes


# #### We can see that there are 16 columns which give tremendous amount of information.

# In[ ]:


df.isnull().sum()


# #### We see that thre are many missing values in the columns.
# The presence of NULL values in **name** and **host_id** don't make difference as they are not important for our analysis.
# In **last_review** column, we cannot perform any operation as the values are simply not available.
# For, **reviews_per_month** we impute the value 0.0 .

# In[ ]:


df.drop(['id','host_name','last_review'], axis=1, inplace=True)


# ## Data Exploration and Data Visualization :

# ## Let's begin with analyzing names:

# To analyze this first we'll create an empty list of all the words and then count the number of occurences of each word. So that we get the most popular terms used in airbnb services.

# In[ ]:


names=[]
upper=[]
for i in df.name:
    j=str(i).split()
    upper.append(j)
for i in upper:
    for j in i:
        names.append(j.lower())
from collections import Counter
#let's see top 20 used words
top_20=Counter(names).most_common()
top_20=top_20[0:20]
top_20


# In[ ]:


nd=pd.DataFrame(top_20)
nd=nd.rename(columns={0:'Words', 1:'Count'})
plt.figure(figsize=(18,12))
sns.barplot(nd['Words'], nd['Count'], palette='Accent')


# We can clearly observe that words like `room, bedroom, private, apartment, cozy` are widely used rather than some catchy headlines that are expected to be used for marketing.

# In[ ]:


# Analysing the uniques present..
df.neighbourhood_group.unique()


# In[ ]:


# Analysing the uniques present..
df.neighbourhood.unique()


# In[ ]:


df.room_type.unique()


# #### Doing the above steps we got to know about the categorical variables.

# ### Let's examine host_id

# In[ ]:


plt.figure(1, figsize=(15, 7))
plt.title("Which was the host max no. of times")
sns.countplot(x = "host_id", order=df['host_id'].value_counts().index[0:10] ,data=df,palette='Accent')


# These are the **hosts** that take the maximum advantage of the airbnb services. We can see that top host has 327 listings in it followed by others.

# ### Let's Examine the neighbourhood_group with prices

# In[ ]:


avg_=df.groupby('neighbourhood_group',as_index=False)['price'].mean()
plt.figure(figsize=(10,6))
sns.barplot(avg_['neighbourhood_group'], avg_['price'])


# We see that **Manhattan** has the highest mean price. And the trend is:
#     * Manhattan
#     * Brooklyn
#     * Queens
#     * Staten Island
# First, we can state that Manhattan has the highest range of prices for the listings with $150 price as average observation, followed by Brooklyn with \$90 per night. Queens and Staten Island appear to have very similar distributions, Bronx is the cheapest of them all. This distribution and density of prices were completely expected; for example, as it is no secret that Manhattan is one of the most expensive places in the world to live in, where Bronx on other hand appears to have lower standards of living.

# In[ ]:


plt.figure(1, figsize=(15, 7))
plt.title("Neighbourhood Counts")
sns.countplot(x = "neighbourhood", order=df['neighbourhood'].value_counts().index[0:10] ,data=df)


# These are **top 10** neighbourhoods.

# ## Room Type:

# In[ ]:


#taking top 10 neighbourhoods
nei=df.loc[df['neighbourhood'].isin(['Williamsburg','Bedford-Stuyvesant','Harlem','Bushwick',
                                               'Upper West Side','Hell\'s Kitchen','East Village','Upper East Side',
                                               'Crown Heights','Midtown'])]
#using factorplot to represent multiple interesting attributes together and a count
plot=sns.factorplot(x='neighbourhood', hue='neighbourhood_group', col='room_type', data=nei, kind='count')
plot.set_xticklabels(rotation=90)


# We see that there are three plots. As we see that there are 2 parameters column and hue and these are the factors which are respomsible for separation of **Factor Plot**. We see that **shared room is barely available** in top 10 listing. **Manhattan and Brooklyn** are the most travelled locations. We can also observe that Bedford-Stuyvesant and Williamsburg are the most popular for Manhattan borough, and Harlem for Brooklyn.

# ## Latitude and Longitude Columns:

# #### We compare price based on location on the map.

# In[ ]:


req=df[df.price<500]
plot=req.plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price',
                  cmap=plt.get_cmap(), colorbar=True, alpha=0.4, figsize=(10,8))


# #### We clearly see the representations of Latitudes and Longitudes in the above map.

# In[ ]:


sns.scatterplot(df.longitude,df.latitude,hue=df.neighbourhood_group,palette='Accent')


# We can see the locatiom of various neighbourhood groups on map of **NYC** as shown above.

# ## No. of Reviews

# In[ ]:


top=df.nlargest(10,'number_of_reviews')
top


# In[ ]:


plt.figure(figsize=(15,10))
plt.xticks(rotation=60)
sns.barplot(top['name'], top['number_of_reviews'], palette='Accent')


# We can infer that top reviews have come from the area `Herlem` and most of the people have opted for a `Private room` and most reviews are from **Room near JFK Queen Bed** which is from Jamaica neighbourhood.

# ## Price

# In[ ]:


sns.kdeplot(df['price'],shade=True,color='r')


# As we can see that the price distribution is very unevenly distributed, with most being low priced.
# So there is no use of finding mean of the prices as it will have affect of outliers.

# In[ ]:


import scipy.stats as stats
sns.distplot(np.log1p(df['price']),color='r')


# #### We see that the price distribution follows log(1+p) distriburtion.

# ## The END.
# ### Please star/upvote if you like.

# In[ ]:




