#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


from matplotlib import rcParams
import warnings 
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use('fivethirtyeight')
#plt.style.use('bmh')
rcParams['axes.labelsize'] = 20
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['text.color'] = 'k'


# In[ ]:


playstore = pd.read_csv("../input/googleplaystore.csv")


# In[ ]:


user_reviews = pd.read_csv("../input/googleplaystore_user_reviews.csv")


# In[ ]:


playstore.head()


# In[ ]:


playstore.drop_duplicates(subset = ['App'],keep='first',inplace=True)


# In[ ]:


playstore.reset_index(inplace=True)


# In[ ]:


playstore.info()


# **We can see that the price is of the object type, and it also contains the dollar symbol which I think is unnecessary, so we will strip the symbol and convert it to float type and we will remove the row where the price is 'Everyone' which does not really make sense and it supposedly garbage value for now**

# In[ ]:


playstore['Price'].unique()


# In[ ]:


playstore = playstore[playstore['Price']!= 'Everyone'].reset_index()


# In[ ]:


for i in range(0,len(playstore['Price'])):
    if '$' in playstore.loc[i,'Price']:
        playstore.loc[i,'Price'] = playstore.loc[i,'Price'][1:]
    playstore.loc[i,'Price'] =  float(playstore.loc[i,'Price'])


# In[ ]:


playstore['Price'].unique()


# In[ ]:


playstore.head()


# In[ ]:


playstore['Reviews'] = playstore['Reviews'].astype('float')


# In[ ]:


playstore['Price'] = playstore['Price'].astype('float')


# In[ ]:


playstore.info()


# **We can see that the ratings columns have missing values. One way to impute the missing values in this case, is to take the average ratings in every category and then impute the missing value of that category**

# In[ ]:


playstore.head()


# In[ ]:


playstore["Rating"] = playstore.groupby("Category")['Rating'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


playstore.head()


# **Let's see what are the categories of the Apps in our dataset and the count of the apps in those categories**
# 
# We can see that the TOP most downloaded apps are in the following categories
# * Family
# * Games
# * Tools
# * Medical
# * Business
# * Productivity
# * Personalization
# * Communication
# * Sports
# * Lifestype

# In[ ]:


f,ax1 = plt.subplots(ncols=1)
sns.countplot("Category", data=playstore,ax=ax1,order=playstore['Category'].value_counts().index)
for tick in ax1.get_xticklabels():
    tick.set_rotation(90)
f.set_size_inches(25,10)
ax1.set_title("Count by Categories")


# **It would be interesting to find out the distribution of the Ratings , Reviews and Price separately. We can see that Ratings is postively skewed, whereas Reviews and Price are heavily skewed to the less hand side**

# In[ ]:


f,(ax1,ax2,ax3) = plt.subplots(ncols=3,sharey=False)
sns.distplot(playstore['Rating'],hist=True,color='b',ax=ax1)
sns.distplot(playstore['Reviews'],hist=True,color='r',ax=ax2)
sns.distplot(playstore['Price'],hist=True,color='g',ax=ax3)
f.set_size_inches(15, 5)


# In[ ]:


f,(ax1,ax2,ax3) = plt.subplots(ncols=3,sharey=False)
sns.boxplot(x='Rating',data=playstore,ax=ax1)
sns.boxplot(x='Reviews',data=playstore,ax=ax2)
sns.boxplot(x='Price',data=playstore,ax=ax3)
f.set_size_inches(15, 5)


# **We can see that most of the Ratings are between 4 and around 4.5 and 5. **
# 
# **As far as Reviews are concerned, most of the count of the reviews are around 0**
# 
# **Also for Price, most of the Apps are Free i guess, we can check this using the Type column**

# In[ ]:


sns.countplot(x='Type',data=playstore)


# In[ ]:


f,ax1 = plt.subplots(ncols=1)
sns.countplot(x = 'Installs',hue='Content Rating',data=playstore,ax=ax1)
plt.xticks(rotation=90)
f.set_size_inches(15,5)
plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


# In[ ]:


f,ax1 = plt.subplots(ncols=1)
sns.countplot(x = 'Installs',hue='Type',data=playstore,ax=ax1,order=playstore['Installs'].value_counts().index)
plt.xticks(rotation=90)
f.set_size_inches(15,5)


# In[ ]:


f,ax1 = plt.subplots(ncols=1)
sns.boxplot(x = 'Category',y='Rating',data=playstore)
plt.xticks(rotation=90)
f.set_size_inches(15,5)


# In[ ]:


f,ax1 = plt.subplots(ncols=1)
sns.boxplot(x = 'Installs',y='Rating',data=playstore)
plt.xticks(rotation=90)
f.set_size_inches(15,5)


# In[ ]:


playstore.head()


# **Well, it would be interesting to see if there is any relationship between Ratings vs Reviews. Does more number of reviews means more ratings?**
# 
# **From the below plot, we cannot say that there is a relation, it seems that irrespective of the Reviews, the ratings are majorly between 4 and 5, which we also noticed before**
# 
# **Also it is not correct to assume that rating and reviews have a relationship because reviews can be positive or negative and increase in the number of reviews does not show whether the reviews are positive or negative.**

# In[ ]:


g = sns.lmplot(x = 'Reviews',y='Rating',data=playstore)


# **It will be interesting to see if size is also one of the factors for installs, because if the App is very heavy, may be people will be hesistant and it would slow down their phone or whatsoever. Let's check if the assumption is supported by the data**
# 
# **We have the value "Varies with device", we will replace it with the mean size of the apps in that category as we did earlier for ratings**

# In[ ]:


playstore['Size'].replace('Varies with device','0',inplace=True)


# **Convert everything into one unit, let's convert everything into MB**

# In[ ]:


for i in range(0,len(playstore['Size'])):
    if 'k' in playstore.loc[i,'Size']:
        playstore.loc[i,'Size'] = playstore.loc[i,'Size'][:-1]
        playstore.loc[i,'Size'] = float(playstore.loc[i,'Size']) / 1000
    elif 'M' in playstore.loc[i,'Size'] :
        playstore.loc[i,'Size'] = playstore.loc[i,'Size'][:-1]
        playstore.loc[i,'Size'] = float(playstore.loc[i,'Size'])


# In[ ]:


playstore['Size'] = playstore['Size'].astype('float')


# In[ ]:


playstore['Size'].replace(0,np.nan,inplace=True)


# In[ ]:


playstore['Size'] = playstore.groupby(by = 'Category')['Size'].transform(lambda x: x.fillna(x.mean()))


# In[ ]:


for i in range(0,len(playstore['Installs'])):
    if '+' in playstore.loc[i,'Installs']:
        playstore.loc[i,'Installs'] = playstore.loc[i,'Installs'][:-1]
        playstore.loc[i,'Installs'] = playstore.loc[i,'Installs'].replace(',','')
        playstore.loc[i,'Installs'] = float(playstore.loc[i,'Installs'])


# In[ ]:


playstore['Installs'] = playstore['Installs'].astype('float')


# In[ ]:


g = sns.scatterplot(y='Size',x='Installs',hue = 'Category',data=playstore[playstore['Category'].isin(['FAMILY','GAMES','TOOLS','BUSINESS','MEDICAL'])])


# In[ ]:


g = sns.catplot(x = 'Category',y='Size',kind='boxen',data=playstore,height=5,aspect=2)
g.set_xticklabels(rotation=90)


# In[ ]:


g = sns.catplot(x = 'Content Rating',y='Installs',kind='boxen',data=playstore,height=5,aspect=2)
g.set_xticklabels(rotation=90)


# In[ ]:


playstore.head()


# In[ ]:


playstore.groupby('Category')['Rating'].mean().sort_values(ascending=False)


# In[ ]:




