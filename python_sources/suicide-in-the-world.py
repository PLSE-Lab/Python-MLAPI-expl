#!/usr/bin/env python
# coding: utf-8

# ## Suicide in the world - EDA and statistical analysis

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal,mannwhitneyu,ttest_ind
import os
os.chdir('/kaggle/input/')


# In[ ]:


df = pd.read_csv('master.csv')


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.columns #checking the columns


# Removing irrelevant data and HDI for year, because there're few datas

# In[ ]:


df.drop(['HDI for year','country-year',' gdp_for_year ($) '],axis=1,inplace=True)


# In[ ]:


df.head()


# In[ ]:


df['year'].unique() #unique years


# In[ ]:


plt.figure(figsize=(10,6))
sns.barplot(x = 'generation',y='suicides/100k pop',data = df,hue = 'sex')
plt.title('Profile of suicides by generation and sex')
plt.show()


# We can observe difference between sex in suicide cases. The same occurs by generation.

# In[ ]:


plt.figure(figsize=(10,6))
sns.set()
plt.title('Profile of suicides by age and sex')
sns.barplot(x = 'age',y = 'suicides/100k pop',data = df,hue='sex')
plt.show()


# The elder are more likely able to suicide. May be the solitude... 
# Man are more likely too. Why...

# In[ ]:


x = df.groupby(['sex'])['suicides/100k pop'].mean()
y = df.groupby(['generation'])['suicides/100k pop'].mean()
z = df.groupby(['sex'])['suicides/100k pop'].std()


# In[ ]:


y #mean of suicides by generation


# In[ ]:


x #mean of suicides by sex


# In[ ]:


z #standard deviation by sex. It's really big.


# In[ ]:


sns.set()
plt.figure(figsize=(9,7))
plt.title('Profile of suicide/100k pop and sex')
k = sns.boxplot(x='sex',y = 'suicides/100k pop', data = df)
k.set_xticklabels(k.get_xticklabels(), rotation=45)
plt.show()


# The difference between sex looks really different. However, the standard desviation is big too. We must use statistical tools to confirm the difference.

# In[ ]:


xy = df[df['sex'] == 'male'].iloc[:,4].values
xx = df[df['sex'] == 'female'].iloc[:,4].values


# It's not obvious, but I'll assume that distribution isn't normal. Then, I must use non-parametrics tests. In this case, the Mann-Whitney is the correct test for this situation.

# In[ ]:


stat, p = mannwhitneyu(xx,xy)


# In[ ]:


p


# The p-value is really low. That shows the significant difference between suicides in men and women. If I use the t-test, the result is the same, as we observe below.

# In[ ]:


ttest_ind(xx,xy)[1]


# In[ ]:


sns.set()
plt.figure(figsize=(9,7))
plt.title('Profile of suicide/100k pop and year')
k = sns.boxplot(x='age',y = 'suicides/100k pop', data = df)
k.set_xticklabels(k.get_xticklabels(), rotation=45)
plt.show()


# In[ ]:


z1 = df[df['age'] == '15-24 years'].iloc[:,4].values
z2 = df[df['age'] == '25-34 years'].iloc[:,4].values
z3 = df[df['age'] == '35-54 years'].iloc[:,4].values
z4 = df[df['age'] == '55-74 years'].iloc[:,4].values
z5 = df[df['age'] == '75+ years'].iloc[:,4].values


# There're five independent groups to be compared. If the normal distribution is assumed, we should use ANOVA. However, we assumed non-normal distribution. In this case, the Kruskal-Walls is the most appropriate. 

# In[ ]:


s,p = kruskal(z1,z2,z3,z4,z5)


# In[ ]:


p


# The p value is lower than 0.05. There're significant difference between groups.

# In[ ]:


a = df.groupby('year')['suicides/100k pop'].mean()


# In[ ]:


x = a.index
y = a.values


# In[ ]:


plt.figure(figsize=(9,6))
plt.title('Mean of suicides over world in the years')
plt.ylabel('Mean of suicides')
plt.xlabel('Year')
plt.plot(x,y)


# In 1995 we observe that is the top year in suicides. Why..

# In[ ]:


plt.figure(figsize=(9,6))
sns.scatterplot(x = 'suicides/100k pop',y = 'gdp_per_capita ($)',data = df)
plt.title('Suicides and GDP')
plt.show()


# It looks like the suicides increase when the GDP/per capita decrease. However it's a weak correlation

# Let's check the suicides by country.

# In[ ]:


a = df.groupby(['country'])['suicides/100k pop'].mean()


# In[ ]:


b = a.sort_values().tail(10)
b


# In[ ]:


plt.figure(figsize=(9,6))
plt.title('Mean of suicides index by country: 1987 - 2016')
plt.ylabel('Mean of suicides')
b.plot.bar()


# The most parts of countries belongs to extinct USSR. Let's check if it was because of the finish of socialism or with the capitalism crisis

# In[ ]:


c = (df.groupby(['year','country'])['suicides/100k pop'].mean())


# In[ ]:


c


# In[ ]:


d = df[df['year'] == 1985] #First year in data set
e = df[df['year'] == 1989] #Union of West Germany and East Germany 
f = df[df['year'] == 1991] #End of Soviet Union
g = df[df['year'] == 1995] #First crisis of neoliberalism in Russian Federation, Brazil, Argentina and Mexico.
h = df[df['year'] == 2008] #Crisis of capitalism
j = df[df['year'] == 2016] #Last year in data set


# In[ ]:


k = [d,e,f,g,h,j]
m = [1985,1989,1991,1995,2008,2016]
c = ['b','k','y','r','g','m']
for i in range(len(k)):
    aa = k[i].groupby(['country'])['suicides/100k pop'].mean()
    bb = aa.sort_values().tail(10)
    plt.figure(figsize=(9,6))
    plt.title('Top 10: Mean of suicides index by country: '+str(m[i]))
    plt.ylabel('Mean of suicides')
    bb.plot.bar(color=c[i])
    plt.show()


# Let's make the same, but with the Last 10 in suicides

# In[ ]:


k = [d,e,f,g,h,j]
m = [1985,1989,1991,1995,2008,2016]
c = ['b','k','y','r','g','m']
for i in range(len(k)):
    aa = k[i].groupby(['country'])['suicides/100k pop'].mean()
    bb = aa.sort_values().head(5)
    plt.figure(figsize=(9,6))
    plt.title('Lower 5: Mean of suicides index by country: '+str(m[i]))
    plt.ylabel('Mean of suicides')
    bb.plot.bar(color=c[i])
    plt.show()


# ## Conclusions

# The suicide rate is higher in men, when compared to women.

# The suicide rate is higher in elder people than in younger.

# The correlation between suicide rate and GDP per capita is observed, but is low.

# After the end of Socialism in East Europe, the suicide rate increased in this region. May be the negative changes caused by the capitalism, as unemployment and increase of homeless? South Korea and Austria appeared with high rates. 

# 1995 is the top year in mean of suicides in the world. What was happening in this year? May be the crisis in East Europe?

# In[ ]:




