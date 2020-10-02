#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings 
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
import os
print(os.listdir("../input"))   


# In[ ]:


data = pd.read_csv('../input/master.csv')
data.head()


# In[ ]:


data.tail()


# In[ ]:


data.shape


# In[ ]:


data.describe()


# In[ ]:


data.info()


# In[ ]:


data.isnull().any()


# In[ ]:


data['HDI for year'].isna().sum()


# In[ ]:


fig = plt.figure(figsize=(20,2))
sns.heatmap(data.isnull(), yticklabels = False, cbar = False, cmap = 'ocean')


# we can see that out of 27820 rows, 19456 rows have no info about HDI. So we can drop this column

# In[ ]:


data = data.drop('HDI for year',axis =1)


# In[ ]:


data.columns


# In[ ]:


data[' gdp_for_year ($) '] = data[' gdp_for_year ($) '].apply(lambda x: x.replace(',','')).astype(float)


# #### correlation of data

# In[ ]:


corr = data.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
plt.subplots(figsize=(15,10))
sns.heatmap(corr, annot = True,mask = mask, linewidth = 0.3)
plt.title('Suicide data correlation')
plt.show()


# we can see that suicide_no and population  and gdp_for_year and population are highly correlated.

# ### variation of suicide with countries

# In[ ]:


# now we can see the pattern of suicides in respect to population of countries
plt.figure(figsize=(20,10))
p= sns.barplot(x="country",y = 'suicides/100k pop', data=data, palette='colorblind')
p.set_title("variation  of suicide per 100k population with countries")
for i in p.get_xticklabels():
    i.set_rotation(90)


# ### DIstribution of suicide numbers with respect to gender,age and generation 

# In[ ]:


#now let us see how suicide rates according to gender are distributed

gender_data = data.groupby(['sex'])['suicides_no'].sum().reset_index()
sns.barplot(x ='sex',y='suicides_no',data = gender_data)
plt.show()


# we can observe that on the basis of gender, men have committed much more suicide in compare to women.
# Now, let us check how suicide rates are affected by age factor.

# In[ ]:


plt.figure(figsize=(15,10))
age_data = data.groupby(['age'])['suicides_no'].sum().reset_index()
sns.barplot(x ='age',y='suicides_no',data = age_data)
plt.show()


# we can observe from the bar plot that suicide rates are highest for age group 35-54 years and then followed by 55-74 years.
# Now, let us see the affect of generation on the suicide rate.

# In[ ]:


plt.figure(figsize=(10,6))
gen_data = data.groupby(['generation'])['suicides_no'].sum().reset_index()
gen_data = gen_data.sort_values(by='suicides_no',ascending =False)
sns.barplot(x ='generation',y='suicides_no',data = gen_data)
plt.show()


# In[ ]:


#distribution of suicide_no in generations in the form of pie plot
plt.figure(figsize=(18,8))
gen =['Boomers','Silent','Generation X','Millenials','G.I. Generation','Generation Z']
plt.pie(data['generation'].value_counts(),explode=[0.1,0.1,0.1,0.1,0.1,0.1],labels =gen, startangle=90, autopct='%.1f%%')
plt.title('Generations Count')
plt.ylabel('Count')


# We can observe that genaration "Boomers " have highest suicide rates.
# - So gender men in age group 35 - 54 and belongs to Boomer generation have highest suicide rate.

# ### Relationship of countries and suicides 

# In[ ]:


# let's see the countries having highest suicide rates
plt.figure(figsize=(15,6))
con_data = data.groupby(['country'])['suicides_no'].sum().reset_index()
con_data = con_data.sort_values(by='suicides_no',ascending =False)
con_data = con_data.head(10)
sns.barplot(x ='country',y='suicides_no',data = con_data)
plt.show()


# In[ ]:


# countries with least suicide rates
plt.figure(figsize=(15,6))
con1_data = data.groupby(['country'])['suicides_no'].sum().reset_index()
con1_data = con1_data.sort_values(by='suicides_no',ascending =False)
con1_data = con1_data.tail(10)
sns.barplot(x ='country',y='suicides_no',data = con1_data)
plt.show()


# In[ ]:


array = ['Russian Federation', 'United States', 'Japan', 'France', 'Ukraine', 'Germany', 'Republic of Korea', 'Brazil', 'Poland', 'United Kingdom']
Period = data.loc[data['country'].isin(array)]
Period = Period.groupby(['country', 'year'])['suicides_no'].sum().unstack('country').plot(figsize=(20, 7))
Period.set_title('Top suicide countries', size=15, fontweight='bold')


# In[ ]:


array = ['Russian Federation', 'United States', 'Japan', 'France', 'Ukraine', 'Germany', 'Republic of Korea', 'Brazil', 'Poland', 'United Kingdom']
gdp_Period = data.loc[data['country'].isin(array)]
gdp_Period = gdp_Period.groupby(['country', 'year'])['gdp_per_capita ($)'].sum().unstack('country').plot(figsize=(20, 7))
gdp_Period.set_title('Top per capita gdp countries', size=15, fontweight='bold')


# In[ ]:


fig=sns.jointplot(y='gdp_per_capita ($)',x='suicides_no',kind='hex',data=data[data['country']=='United States'])
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
con2_data = data.groupby(['country'])['population'].sum().reset_index()
con2_data = con2_data.sort_values(by='population',ascending =False)
con2_data = con2_data.head(10)
sns.barplot(x ='country',y='population',data = con2_data)
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
sns.barplot(x ='year',y='population',data = (data[data['country'] == 'United States']))
plt.show()


# In[ ]:


plt.figure(figsize=(15,6))
sns.barplot(x ='year',y='population',data = (data[data['country'] == 'Russian Federation']))
plt.show()


# We observed that although per capita GDP has increased but is doesn't have much impact on suicide rates but population has a played an important role in deciding suicide rates.
# - At the end, as of 2015, US has highest per capita GDP but suicide rates are also at the highest, it is because of large increase in population. 
# - In the case of russian Federation, due to very low or almost no increase of population and increase in per capita gdp has played and important role in decreasing the suicide rates.
# - This again proves our point of correlation plot which shows higher correlation between poulation and suicide numbers

# ### Distribution of suicide along years --

# In[ ]:


plt.figure(figsize=(15,7))
sns.stripplot(x="year",y='suicides/100k pop',data=data)
plt.xticks(rotation=45)
plt.show()


# In[ ]:


dfSexPeriod =data.groupby(['sex', 'year'])['suicides_no'].sum().unstack('sex').plot(figsize=(20, 7))
dfSexPeriod.set_title('Suicide per Sex', size=15, fontweight='bold')


# In[ ]:


dfAgePeriod =data.groupby(['age', 'year'])['suicides_no'].sum().unstack('age').plot(figsize=(20, 10))
dfAgePeriod.set_title('Suicide per Age', size=15, fontweight='bold')


# From the above graphs, we can observe that there has been substantial increase in suicide rates in the world since 1985.From 19995-2005 (approx.), there has been highest suicide rates and then it is decreasing slowly.

# In[ ]:




