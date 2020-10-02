#!/usr/bin/env python
# coding: utf-8

# <font size="10">Suicide Analysis "1985 - 2016"</font>
# <br><br>
# Content:
# 1. Load the dataset
# 1. Describe the dataset
# 1. Overview Analysis
# 1. Finding correlation that between suicide_no and other variables
# 1. Conclusion

# **First of all, lets input all important library required on this analysis**

# In[ ]:


#import all important library
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import linregress
warnings.filterwarnings("ignore")


# > <font size="5">1. Load the dataset</font>

# In[ ]:


#read the file
data = pd.read_csv("../input/master.csv")
#print the head of the data
print(data.head(3))


# > <font size="5">2. Describe the dataset</font>

# In[ ]:


#print the preview statistic of each column
print(data.describe())


# > <font size="5">3. Overview Analysis</font>

# **Suicides based on Gender**

# In[ ]:


#total sucide based on the gender
sns.set(style = "whitegrid")
gender_suicide = sns.barplot(x="sex", y="suicides_no", data=data)


# Based on the bar chart above, we can see that most of the suicide happen on male instead female

# **Year on Year Suicides**

# In[ ]:


#year on year suicide count
yoy_suicide = data.groupby(['year']).sum().reset_index()
#print(yoy_suicide)
plt.figure(figsize=(15,5))
sns.lineplot(x='year', y='suicides_no', data=yoy_suicide, color='navy')
plt.axhline(yoy_suicide['suicides_no'].mean(), ls='--', color='red')
plt.title('Total suicides (by year)')
plt.xlim(1985,2015)
plt.show()


# **As you can see from the line chart above, over a decade after 80's the year on year suicide increase during 90's and drop again starting from early 2000**

# **Average Suicides per Country**

# In[ ]:


#total suicide by country
suicide_per_country = data.groupby(['country']).mean().sort_values('suicides_no', ascending = False).reset_index()
plt.figure(figsize=(15,20))
sns.barplot(x='suicides_no', y='country', data=suicide_per_country)
plt.axvline(x = suicide_per_country['suicides_no'].mean(), color='red', ls='--')
plt.show()


# **50% of the top 10 country with highest average suicide are coming from Europe, 30% from Asia, and 20% from America**

# **Suicides per Age range**

# In[ ]:


#total suicide by age group
suicide_per_age = data.groupby(['age']).sum().sort_values('suicides_no', ascending = False).reset_index()
plt.figure(figsize=(10,10))
sns.barplot(x='suicides_no', y='age', data=suicide_per_age)
plt.axvline(x = suicide_per_age['suicides_no'].mean(), color='red', ls='--')
plt.show()


# **Most of the suicides come from people who have age from 35 years old above**

# **Suicide per Generation**

# In[ ]:


#total suicide by generation
suicide_per_generation = data.groupby(['generation']).sum().sort_values('suicides_no', ascending = False).reset_index()
plt.figure(figsize=(10,10))
sns.barplot(x='suicides_no', y='generation', data=suicide_per_generation)
plt.axvline(x = suicide_per_generation['suicides_no'].mean(), color='red', ls='--')
plt.show()


# **As we can see from the horizontal bar chart above, Generation X, Silent, and Boomers are the top 3 generation with the highest suicide no**

# **Top 10 countries with highest suicide**

# In[ ]:


#year on year suicide of top 3 countries with highest suicide
top3_df = data.loc[(data['country']=='Russian Federation') | (data['country']=='Japan') | (data['country']=='United States') | (data['country']=='France') | (data['country']=='Ukraine') | (data['country']=='Germany') | (data['country']=='Republic of Korea') | (data['country']=='Brazil') | (data['country']=='Poland') | (data['country']=='United Kingdom')].reset_index()
#print(top3_df)
yoy_suicide_top_3 = top3_df.groupby(['year','country']).sum().reset_index()
plt.figure(figsize=(20,10))
sns.lineplot(x='year', y='suicides_no', hue='country', data=yoy_suicide_top_3)
plt.show()


# **Suicide no in both United States and Republic of Korea increase during 2000 however other top countries decrease**

# > <font size="5">4. Finding correlation between suicides_no and other variables</font>

# **As you can see from the table below, highest correlation between 2 variables are population and suicides_no**

# In[ ]:


#perason correlation between variable
data.corr(method = 'pearson')


# Now, let's find the correlation between these 2 variables per country

# In[ ]:


#correlation between suicide_no and population
ds = data.groupby('country')[['population','suicides_no']].corr().iloc[0::2,-1].reset_index()
ds = ds.sort_values('suicides_no', ascending = False)
#print(ds.head())
plt.figure(figsize=(20,20))
sns.barplot(x='suicides_no', y='country', data=ds)
#plt.axvline(x = suicide_per_generation['suicides_no'].mean(), color='red', ls='--')
plt.show()


# **Qatar and United Arab Emirates are the only 2 countries that has correlation between populaiton and suicides_no above 0.9**
# Below is the line chart that show us the total suicides vs total population year on year for Qatar

# In[ ]:


#year on year suicide count qatar
yoy_suicide_qatar = data.loc[(data['country']=='Qatar')].groupby(['year']).sum().reset_index()
#print(yoy_suicide_qatar)
#print(yoy_suicide)
plt.figure(figsize=(15,8))

plt.subplot(211)
sns.lineplot(x='year', y='suicides_no', data=yoy_suicide_qatar, color='navy')
plt.title('Qatar total suicides (by year)')
plt.xlabel("")

plt.subplot(212)
sns.lineplot(x='year', y='population', data=yoy_suicide_qatar, color='navy')
plt.title('Qatar total population (by year)')

plt.tight_layout
plt.show()


# > <font size="5">5. Conclusion</font>

# 1. Male die way more compared to Female
# 1. Total suicide increase from 1990 - 1999 and start to drop again in the early 2000
# 1. Most of the suicide happen on people who age more than 35 years old
# 1. Most of the suicide happen on the boomers generation
# 1. Total population is the variable that has the highest correlation to the suicide no
# 1. Qatar is the country that show the highest correlation (>0.9) between total population and suicides no
