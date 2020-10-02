#!/usr/bin/env python
# coding: utf-8

# In this Kernel, we will have a look at the insights for the data given by Incedo. Then on the Machine Learning part. 

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt  # for data vizualization
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


df_train= pd.read_csv('../input/train_file.csv')


# In[ ]:


df_train.head() # first 5 rows


# In[ ]:


df_train.info() #basic information about the whole data


# In[ ]:


df_train.describe(include='all') # explicit details about the data 


# In[ ]:


df_train.Greater_Risk_Question.value_counts()


# We can see that the teenagers are more into pot. And from the above count we can imply that teenagers who did the first abuse before 13 years are more addicted to Drug Abuse. 

# In[ ]:


df_train.isnull().sum()#checking the null values for each variable


# Only Geolocation is missing in the dataset. This could mean that people who do drugs don't stick to a same place for Drug abuse. They keep moving as there is a fear of being caught by elders or police as they are still minors.

# In[ ]:


df_test= pd.read_csv('../input/test_file.csv')
df_test.head()


# In[ ]:


df_train.columns # checking the number of columns in the dataset


# In[ ]:


df_train.nunique()#checking all the unique items in the dataset


# In[ ]:


df_train.dtypes #checking the datatypes in the data, as we will need to hotencode the data for prediction purpose


# In[ ]:


# correlation plot of the Variables in the data

f, ax = plt.subplots(figsize = (14, 10))

corr = df_train.corr()
sns.heatmap(corr, mask = np.zeros_like(corr, dtype = np.bool), 
            cmap = sns.diverging_palette(3, 3, as_cmap = True), square = True, ax = ax)


# In[ ]:


# visualising the distribution of Drug-abuse in the dataset

df_train['LocationDesc'].value_counts(normalize = True)
df_train['Greater_Risk_Question'].value_counts(dropna = False).plot.bar(color = 'c', figsize = (10, 8))

plt.title('distribution of Drug-intake & the residence')
plt.xlabel('Drug-intake')
plt.ylabel('residence')
plt.show()


# In[ ]:


# visualising the distribution of Drug-intake & the Age  in the dataset

df_train['LocationDesc'].value_counts(normalize = True)
df_train['Race'].value_counts(dropna = False).plot.bar(color = 'green', figsize = (10, 8))

plt.title('distribution of Drug-intake & the Age')
plt.xlabel('Drug-intake')
plt.ylabel('Race')
plt.show()


# The above chart shows us that Drug Abuse is more prevalant among the the natives, followed by Latinos and African Americans. This could be due to lack of proper education/parenting since very young age or could be due to influence of Social Media on young minds.

# In[ ]:


#checking the unique values in the Drug_User Column
df_train['Greater_Risk_Question'].value_counts()


# In[ ]:


#checking the count of People who are addicted in terms of their Grade
df_train['LocationDesc'].value_counts(normalize = True)
df_train['Grade'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 8))
plt.title('people addicted to drugs in terms of their education')
plt.xlabel('Grade')
plt.ylabel('count')
plt.show()


# The above illustration illustrates an alarming pattern. It shows that teenagers who are more into Drug Abuse are people who are just in their teens and thats the time when adolesence hits. Everything seems cool age.

# In[ ]:


#Visualizing the YEAR Distribution in the Dataset
df_train['YEAR'].value_counts(normalize = True)
df_train['YEAR'].value_counts(dropna = False).plot.bar(color = 'black', figsize = (7, 5))

plt.title('Distribution of 141 coutries in suicides')
plt.xlabel('YEAR')
plt.ylabel('count')
plt.show()


# In[ ]:


df_train['YEAR'].nunique()


# In[ ]:


#Visualizing the Location Distribution in the Dataset
get_ipython().run_line_magic('time', '')
df_train['LocationDesc'].value_counts(normalize = True)
df_train['LocationDesc'].value_counts(dropna = False).plot.bar(color = 'purple', figsize = (17, 10))

plt.title('Distribution of 141 coutries in suicides')
plt.xlabel('LocationDesc')
plt.ylabel('count')
plt.show()


# In[ ]:


#Visualizing the Location Distribution in the Dataset
get_ipython().run_line_magic('time', '')
df_train['GeoLocation'].value_counts(normalize = True)
df_train['GeoLocation'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (17, 10))

plt.title('Distribution of geoloaction ')
plt.xlabel('Geolocation')
plt.ylabel('count')
plt.show()


# In[ ]:


#Visualizing the Location Distribution in the Dataset
get_ipython().run_line_magic('time', '')
df_train['GeoLocation'].value_counts(normalize = True)
df_train['GeoLocation'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (17, 10))

plt.title('Distribution of geoloaction ')
plt.xlabel('Geolocation')
plt.ylabel('count')
plt.show()


# In[ ]:


#Visualizing the choice of Drug Abuse in the Dataset
get_ipython().run_line_magic('time', '')
df_train['Greater_Risk_Question'].value_counts(normalize = True)
df_train['Greater_Risk_Question'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 7))

plt.title('Distribution of Greater_Risk_Question ')
plt.xlabel('Greater_Risk_Question')
plt.ylabel('count')
plt.show()


# Above illustration shows us, people are more into Marijuana(Ganja) who have had their first drink by the age of 13. This means the more young people start drug abuse, more addicted they become in the longer run.

# In[ ]:


# Distribution of Addiction based on Gender

get_ipython().run_line_magic('time', '')
df_train['Sex'].value_counts(normalize = True)
df_train['Sex'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 7))

plt.title('Distribution of gender ')
plt.xlabel('Sex')
plt.ylabel('count')
plt.show()


# The above figure shows, Addiction is equally spread across gender. Meaning Gender doesn't matter in this case. Teenagers of either Gender are equally prone to Addiction.

# In[ ]:


#total number of addicted people in the dataset
df_train['Description'].value_counts()


# In[ ]:


# Distribution of Description of Addiction
get_ipython().run_line_magic('time', '')
df_train['Description'].value_counts(normalize = True)
df_train['Description'].value_counts(dropna = False).plot.bar(color = 'brown', figsize = (10, 7))

plt.title('Distribution of Description ')
plt.xlabel('Sex')
plt.ylabel('count')
plt.show()


# 

# The above figure illustrates that the number of people addicted to Marijuana are highest followed by Alcohol and then coke/heroin.

# In[ ]:





# 
