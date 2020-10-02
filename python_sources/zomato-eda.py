#!/usr/bin/env python
# coding: utf-8

# In[7]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
path = '../input/'
pd.options.display.max_columns = 999


# #### Importing data

# In[8]:


df = pd.read_csv(path + 'zomato.csv', encoding = "ISO-8859-1")
df.head(1)


# In[10]:


df_CC = pd.read_excel(path + 'Country-Code.xlsx')
df_CC.head()


# In[11]:


df.columns


# In[12]:


df.dtypes


# In[13]:


df.shape

Let's do some analysis based on the country code.
# #### Groupby Country code.

# In[15]:


df_grp = df.groupby(['Country Code'], as_index=False).count()[['Country Code', 'Restaurant ID']]
df_grp.columns = ['Country Code', 'No of Restaurant']
res = df_grp.join(df_CC.set_index('Country Code'), on = 'Country Code')

Let's plot it. Basically the number of restaurant that are on zomato in each country.
# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(res['Country'], res['No of Restaurant'])
plt.xticks(rotation = 90)
plt.show()

Basically opeartions of zomato is centered in India and United States mostly.
# #### Average rating for every country.

# In[ ]:


df_grp_rating = df.groupby(['Country Code'], as_index=False)
ans = df_grp_rating['Aggregate rating'].agg(np.mean)


# In[ ]:


res = ans.join(df_CC.set_index('Country Code'), on = 'Country Code')


# In[ ]:


sns.set(rc={'figure.figsize':(11.7,8.27)})
sns.barplot(res['Country'], res['Aggregate rating'])
plt.xticks(rotation = 90)
plt.show()

Looks like India has the worst Aggregate rating.
# #### Rating text wise distribution of restaurant in each country

# In[ ]:


df_grp_country = df.groupby(['Country Code', 'Rating text'], as_index=False).count()[['Country Code', 'Restaurant ID', 'Rating text']]
df_grp_country.columns = ['Country Code', 'No of Restaurant', 'Rating text']
res = df_grp_country.join(df_CC.set_index('Country Code'), on = 'Country Code')
country = res['Country'].unique()
df_ratings = res[['Country', 'Rating text']]
df_ratings['percentage'] = 0.0
for c in country:
    rating_text = res[res['Country'] == c]
    ratings = rating_text['No of Restaurant']
    df_ratings.loc[df_ratings.Country == c,'percentage'] = ratings.apply(lambda x : x/sum(ratings)*100)
ans = pd.merge(res, df_ratings, on = 'Country', how = 'inner')

Distribution of Rating text for each country.
# In[ ]:


sns.set(rc={'figure.figsize':(20,11)})
sns.barplot('Country', 'percentage', data=ans, hue = 'Rating text_y')
plt.xticks(rotation = 90)
plt.show()

A common myth is that costly places generally provides good service and food. Let's check this out for India for uniformity of currency.
# In[ ]:


np.random.seed(sum(map(ord, "categorical")))
sns.set(rc={'figure.figsize':(20,11)})
india_df = df.loc[df['Country Code'] == 1]
avg_cost = india_df['Average Cost for two']
agg_rating = pd.Categorical(india_df['Rating text'], categories=["Excellent", "Very Good", "Good", "Average", "Poor", "Not rated"], ordered=True)
sns.boxplot(agg_rating, avg_cost)
plt.show()

Median "Average Cost for Two" increases with the improvement in rating text.
# In[ ]:




