#!/usr/bin/env python
# coding: utf-8

# #Person of the "Pandas"

# In[6]:


import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(color_codes=True)

get_ipython().run_line_magic('matplotlib', 'inline')


# ###Process Data

# In[7]:


time_data = pd.read_csv("../input/archive.csv", sep=",")
time_data.head()


# In[8]:


time_data['Year_Award'] = time_data['Year'] - time_data['Birth Year']
time_data.head()


# A brief insight about the categories with more awards, the politics personalities are more influencer. 

# In[9]:


category_counts = time_data['Category'].value_counts()
category = category_counts.index
c_counts = category_counts.get_values()
plt.xticks(rotation=70)
plt.xlabel('Categories of the winners')
plt.ylabel('Number of winners')
plt.title('Number of winners per category', fontsize=20)
barplot = sns.barplot(x=category, y=c_counts)


# ### Category Winner by Country

# Now we'll make a *"group by"* **Country and Category** columns and stack them. There are a lot of winners from USA so we'll discriminate them to have a better visualization. 

# In[10]:


country_category_group = time_data.groupby(['Country','Category'])

add_counts = country_category_group.size().unstack().fillna(0)
normed_subset = add_counts.div(add_counts.sum(1),axis=0)
add_counts.plot(kind='barh',legend=True,figsize=(11,7), stacked=True, colormap='cubehelix');


# ### Category Winner by Country (No USA)

# In[11]:


country_less_usa = time_data[time_data['Country'] != "United States"]

country_less_usa_category_group = country_less_usa.groupby(['Country','Category'])

add_counts = country_less_usa_category_group.size().unstack().fillna(0)
normed_subset = add_counts.div(add_counts.sum(1),axis=0)
add_counts.plot(kind='barh',legend=True,figsize=(11,7), stacked=True, colormap='cubehelix');


# ## Age of the winners by Category 

# In[12]:


x_val=time_data['Year_Award']
y_val=time_data['Category']
sns.stripplot(x=x_val, y=y_val, data=time_data, jitter=True, orient='h', size=10);


# ### Age of the winners (Average).

# In[13]:


year_award_average = time_data['Year_Award'].mean()
print ('The average is: '+ str(year_award_average) +" year")


# In[14]:


time_data['Years_Lived'] = time_data['Death Year'] - time_data['Birth Year'] 
time_data[1:3]


# ### Age Range of the winners.
# We create 4 new age-categories "Youth, YoughtAdult, MiddleAged, Senior".

# In[15]:


#Age Range
max = time_data['Year_Award'].max()
bins = [18,25,35,60,max+1]
age_range = ['Youth','YoughtAdult','MiddleAged','Senior']

time_data['Range'] = pd.cut(time_data['Year_Award'], bins)
time_data['Age_Category'] = pd.cut(time_data['Year_Award'], bins, labels=age_range)
time_data.head()


# In[16]:


def make_autopct(values):
    def my_autopct(pct):
        total = sum(values)
        val=int((pct*total/100.0)+0.5)
        return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
    return my_autopct



age_category_counts = time_data['Age_Category'].value_counts()
#age_category_counts.plot(kind='pie',legend=False,figsize=(12,5));
age_category_counts.plot(kind='pie',legend=True,figsize=(7,7), autopct=make_autopct(age_category_counts));


# ### Winners by Country

# In[17]:


country_counts = time_data['Country'].value_counts()


# In[18]:


country_counts[:6].plot(kind='pie',legend=True,figsize=(7,7), autopct=make_autopct(country_counts[:6]));


# In[19]:


sns.jointplot(x=time_data['Year'], y=time_data['Year_Award']);


# In[20]:


sns.jointplot(x=time_data['Year'], y=time_data['Year_Award'], kind="kde");


# ### Age-Category of the winners by Honor

# In[21]:


honor_age_category_group = time_data.groupby(['Honor','Age_Category'])

add_counts = honor_age_category_group.size().unstack().fillna(0)
normed_subset = add_counts.div(add_counts.sum(1),axis=0)

normed_subset.plot(kind='barh',legend=True,figsize=(10,7), stacked=True, colormap='cubehelix');

plt.legend( loc = 'center right');

