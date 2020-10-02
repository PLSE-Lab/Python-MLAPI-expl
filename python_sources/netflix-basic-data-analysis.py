#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[ ]:


netflix=pd.read_csv("../input/netflix-shows-exploratory-analysis/netflix_titles.csv",parse_dates=["date_added"])


# In[ ]:


netflix.head()


# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(20,6))
ratings=sns.countplot(x='rating',data=netflix)
plt.xlabel("Rating",fontsize=18)
plt.ylabel("Count",fontsize=18)
plt.title("Ratings",fontsize=25)
plt.show()


# In[ ]:


plt.figure(figsize=(20,6))
sns.set_style("darkgrid")
country_data=netflix['country'].value_counts().sort_values(ascending=False).head(10)
country_data
countrys=sns.barplot(country_data.index,country_data.values)
countrys.set_xticklabels(rotation=30,labels=country_data.index)
plt.title("No of Movies release in each country",fontsize=25)
plt.xlabel("Country",fontsize=18)
plt.ylabel("No Of Movies",fontsize=18)
plt.show()


# In[ ]:


plt.figure(figsize=(20,6))
net_types=netflix['type'].value_counts()
plt.pie(net_types,labels=net_types.index)
plt.legend()
plt.show()


# In[ ]:


releasemovie_year=netflix[netflix['release_year'].between(2010,2018)]
sns.set(style="darkgrid")
plt.figure(figsize=(20,6))
releaseyear=sns.countplot(x='release_year',data=releasemovie_year)

plt.title("No of Movies release in Year",fontsize=25)
plt.xlabel("Release Year",fontsize=18)
plt.ylabel("No Of Movies",fontsize=18)
plt.show()


# In[ ]:


sns.set(style="darkgrid")
plt.figure(figsize=(20,6))
director=netflix.director.value_counts().sort_values(ascending=False).head(9)
director
sns.barplot(x=director.index,y=director.values)
plt.title("Most Popular Director",fontsize=25)
plt.xlabel("Director",fontsize=18)
plt.ylabel("No Of Movies",fontsize=18)
plt.show()

