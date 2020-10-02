#!/usr/bin/env python
# coding: utf-8

# # Chocolate Ratings Analysis

# This is a quick and dirty analysis of a chocolate dataset. I hope y'all like it!!

# # Import Libraries, Download CSV and Create a DataFrame.

# In[3]:


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


cacao = pd.read_csv('../input/flavors_of_cacao.csv')


# # How does the Data Look?

# In[5]:


cacao.info()


# - Only 1794 rows and 9 columns. Not a very large dataset.

# In[6]:


cacao.describe()


# - The review spans from 2006 to 2017. A score of 1 is the lowest and 5 is the highest. Average rating is 3.2

# In[7]:


cacao.head()


# - Two things. 'Cocoa Percent' doesn't seem to be an int or float type and 'Bean Type' seems to be missing some data.

# In[8]:


type(cacao['Cocoa\nPercent'][0])


# - First, Cocoa Percent is not an integer or float type. Convert it to a float or int type.

# In[9]:


cacao['Cocoa\nPercent'] = cacao['Cocoa\nPercent'].apply(lambda x: float(x.split('%')[0]))


# In[ ]:


type(cacao['Cocoa\nPercent'][0])


# - Second, how much data am I missing from this dataset?

# In[ ]:


cacao.isnull().head()


# In[ ]:


sns.heatmap(cacao.isnull(),cbar=False,yticklabels=False,cmap='viridis')


# - No visible null values in the dataframe.

# In[ ]:


cacao['Bean\nType'][0]


# - '\xa0' is a 'non-breaking-space'

# In[ ]:


len(cacao[cacao['Bean\nType']=='\xa0']) / len(cacao['Bean\nType'])


# - Nearly 50% of the info on the Bean Type column has no value.

# In[ ]:


cacao['Bean\nType'].value_counts().head(10)


# - Most bean types are a combintion of one of the three most popular beans.

# # EDA. 

# In[ ]:


plt.figure(figsize=(8,6))
cacao['Broad Bean\nOrigin'].value_counts().head(20).plot.bar()
plt.title('Top 15 Cocoa Bean Producing Nations')
plt.show()
print('Top 15 Cocoa Bean Producing Nations\n')
print(cacao['Broad Bean\nOrigin'].value_counts().head(20))


# In[ ]:


print('Least common bean location representation\n')
print(cacao['Broad Bean\nOrigin'].value_counts().tail(20))


# - Aside from Madagascar, a huge amount of cocoa beans come from Latin America.
# A good portion of bars reviewed source from a combination of nations.

# In[ ]:


plt.figure(figsize=(8,6))
cacao['Company\nLocation'].value_counts().head(20).plot.bar()
plt.title('Top 15 Nations by Number of Chocolate Bar Companies')
plt.show()
print('Top 15 Nations by Number of Chocolate Bar Companies\n')
print(cacao['Company\nLocation'].value_counts().head(20))


# - U.S.A. is by far the country with the most chocolate producers. Most other chocolate producing nations are either European, North American or Australian.

# - Brazil, Peru, Ecuador, Colombia and Venezuela are the only countries that both produce chocolate bars and cocoa beans by a good margin. Ecuador really stands out.

# In[ ]:


plt.figure(figsize=(8,6))
cacao['Company\xa0\n(Maker-if known)'].value_counts().head(20).plot.bar()
plt.title('Top Companies by # of Bars Made')
plt.show()
print('Top Companies by # of Bars Made\n')
print(cacao['Company\xa0\n(Maker-if known)'].value_counts().head(20))


# - The Canadian company Soma really stands out as a producer that has been rated many times.

# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(cacao['Rating'],kde=False,color='green')
plt.title('Ratings Distribution')
plt.show()
print(cacao['Rating'].value_counts())


# - Most chocolate bars rank between 2.5 and 3.75. Only 2 bars rank above 4. Only 17 bars below 2.

# In[ ]:


plt.figure(figsize=(8,6))
sns.distplot(cacao['Review\nDate'],bins=50,kde=False,color='green')
plt.title('Number of Reviews per Year')
plt.show()


# - The number of reviews increases until 2016, and sharply drops in 2017. 

# # More EDA

# In[ ]:


sns.pairplot(cacao)


# - In this analysis, the Rating column will be the label. The two other columns that can be used to study Ratings would be the Review Date column and the Cocoa Percent column. 

# In[ ]:


cacao.plot.scatter(x='Cocoa\nPercent',y='Rating')
plt.title('Ratings to % of Cocoa Content')
plt.show()
print('Cocoa % to Quantity ')
print('------------')
print(cacao['Cocoa\nPercent'].value_counts().head(10))


# - Very clear void of ratings above 4 and below 2.

# In[ ]:


sns.jointplot(x='Cocoa\nPercent',y='Rating',data=cacao,size=8,kind='kde',cmap='coolwarm')


# - A huge portion of chocolates reviewed contain around 70% cocoa and rank between 2.5 to 4.

# In[ ]:


cacao.plot.scatter(x='Review\nDate',y='Rating')
plt.title('Ratings by Review Date')
plt.show()
print('Year and number of ratings that year.\n')
print(cacao['Review\nDate'].value_counts().head(10))


# - No ratings above 4 after 2007 and no ratings below 2 after 2014.
# Ratings vary less as the review progresses in time.

# In[ ]:


sns.jointplot(x='Review\nDate',y='Rating',data=cacao,size=8,kind='kde',cmap='coolwarm')


# - Ranking scores become less diverse as time progresses. Many of the reviews are between 2012 and 2017.

# In[ ]:


cacao[cacao['Rating']==5]


# - Only 2 bars ranked #5. Both were Italian, both were from Amedei and both were 70% cocoa.

# In[ ]:


cacao[cacao['Rating']==1]


# - 3 of the 4 worst chocolates came from Belgium. No chocolate with a rating of 1 came after 2008.

# In[ ]:


sns.heatmap(cacao.corr(),cmap='coolwarm',annot=True)


# - There doesn't seem to be any correlation between review date, cocoa percent and rating.

# In[ ]:


koko = cacao.groupby('Company\nLocation').mean()
plt.figure(figsize=(10,8))
koko['Rating'].plot.bar()
plt.title('Average Rating by Company Location')
plt.tight_layout()
plt.show()
print('Best Average Ratings by Company Location\n')
print(koko['Rating'].sort_values(ascending=False).head(10))


# In[ ]:


print('Chile has ' + str(len(cacao[cacao['Company\nLocation']=='Chile'])) +' chocolate companies.')
print('The Netherlands have ' + str(len((cacao[cacao['Company\nLocation']=='Amsterdam'] + cacao[cacao['Company\nLocation']=='Netherlands']))) + ' chocolate companies.')
print('The Philippines have ' + str(len(cacao[cacao['Company\nLocation']=='Philippines'])) +' chocolate companies.')


# - Do not be fooled! The Dutch have more chocolates rated than the other 2 nations.

# - No surprise the Dutch are good chocolate makers! View the link below.

# https://en.wikipedia.org/wiki/Dutch_process_chocolate

# In[ ]:


plt.figure(figsize=(10,8))
koko['Cocoa\nPercent'].plot.bar()
plt.title('Average Cocoa % by Company Location')
plt.tight_layout()
plt.show()
print('Highest Average Cocoa % by company Location\n')
print(koko['Cocoa\nPercent'].sort_values(ascending=False).head(10))
print('\n')
print('Lowest Average Cocoa % by company Location\n')
print(koko['Cocoa\nPercent'].sort_values(ascending=False).tail(10))


# - Where are Eucador, The Domincan Republic and Niacragua??? The dataset seems to have spelling errors. Many of the extremes seem to be single instances. Sao Tome and Ireland both have 4 instances.

# In[ ]:


print('Top Rated by Location of Bean.\n')
print(cacao.groupby('Broad Bean\nOrigin')['Rating'].mean().sort_values(ascending=False).head(10))
print('\n')
print('Bottom Rated by Location of Bean.\n')
print(cacao.groupby('Broad Bean\nOrigin')['Rating'].mean().sort_values(ascending=False).tail(10))


# - Both top  and bottom rated chocolates source their beans from a variety of places (with a few exceptions).

# In[ ]:


plt.figure(figsize=(10,8))
bean_type = cacao.groupby('Bean\nType')
bean_type.mean()['Rating'].plot.bar()
plt.title('Rating by Bean Type')
plt.tight_layout()
plt.show()
print('Most Common Bean Type\n')
print(cacao['Bean\nType'].value_counts().head(15))
print('\n')
print('Top Rated Beans by Bean Type\n')
print(cacao.groupby(['Bean\nType'])['Rating'].mean().sort_values(ascending=False).head(15))
print('\n')
print('Bottom Rated Beans by Bean Type\n')
print(cacao.groupby(['Bean\nType'])['Rating'].mean().sort_values(ascending=False).tail(15))


# A lot of missing data for this column. Notice the most common Bean Types (Forastero, Criollo, Trinitario) rank somewhere in the middle.

# # Feed back is appreciated!! Thank you!

# In[ ]:





# In[ ]:




