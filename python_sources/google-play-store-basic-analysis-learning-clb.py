#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns


# **Read Data Folders**

# In[ ]:


dataStore = pd.read_csv('../input/googleplaystore.csv') 
dataStore = dataStore.dropna(axis=0,how='any')
dataUserReview = pd.read_csv('../input/googleplaystore_user_reviews.csv')


# **Google Play Store data info**

# In[ ]:


dataStore.info()


# **Count**

# In[ ]:


dataStore.count()


# **Describe**

# In[ ]:


dataStore.describe()


# **Google Play Store User Reviews data info**

# In[ ]:


dataUserReview.info()


# **Group By Size**

# In[ ]:


dataStore.groupby('Category').size()


# **Correlation For All**

# In[ ]:


dataStore.corr()


# In[ ]:


dataUserReview.corr()


# **Correlation Map**

# In[ ]:


f,ax = plt.subplots(figsize=(3,3))
sns.heatmap(dataStore.corr(), annot=True, linewidths=.5,fmt = '.2f', ax=ax)
plt.show()


# In[ ]:


f ,ax = plt.subplots(figsize=(2,2))
sns.heatmap(dataUserReview.corr(),annot=True,linewidths=1.0,fmt='.1f',ax=ax)
plt.show()


# **Head, Tail and Columns**

# In[ ]:


dataStore.head(3)


# In[ ]:


dataUserReview.head(12)


# In[ ]:


dataStore.columns


# In[ ]:


dataUserReview.columns


# In[ ]:


dataStore.tail()


# **Shape**

# In[ ]:


dataStore.shape


# In[ ]:


dataUserReview.shape


# ***MATPLOTLIB***

# **Line Plot**

# In[ ]:


dataStore.Rating.plot(kind='line',color='brown', label='Rating',linewidth=0.5)
plt.legend(loc='lower left') 
plt.title('Rating')
plt.show()


# **Scatter Plot**

# In[ ]:


# Rating
dataStore.plot(kind='scatter' , x='Rating', y='Rating' , color = 'red')
plt.xlabel('Rating')
plt.ylabel('Rating')
plt.title('Info')
plt.show()


# **Histogram**

# In[ ]:


dataStore.Rating.plot(kind='hist',bins = 50,figsize=(5,5))
plt.show()


# **PANDAS**

# In[ ]:


ratingSeries = dataStore['Rating']
ratingDataFrame = dataStore[['Rating']]

for index,value in dataStore[['Rating']][0:5].iterrows():
    print(index," : ",value)


# **FILTRE**

# In[ ]:


filtre = dataStore['Rating'] > 4.5
dataStore[filtre]


# In[ ]:


dataStore[np.logical_and(dataStore['Rating'] > 4.7, dataStore['Category'] == 'ART_AND_DESIGN')]


# **ITERATION**

# In[ ]:


category = iter(dataStore['Category'])
print(next(category))
print(*category)


# **LIST COMPREHENSION**

# In[ ]:


dataStore["Degree"] = ["Very Good" if i > 4.5 else "Good" if i > 4.0 else "So-So"  for i in dataStore.Rating]
dataStore.loc[:100,["Degree","Rating"]]


# **Value Counts**

# In[ ]:


print(dataStore['Category'].value_counts(dropna = False))


# **Box Plots**

# In[ ]:


dataStore.boxplot(column='Rating', by='Price')
plt.show()


# **Tidy Data - melt() and pivot()**

# In[ ]:


newDataStore = dataStore.head()
melted = pd.melt(frame=newDataStore, id_vars='App', value_vars=['Reviews','Rating'])
melted


# In[ ]:


melted.pivot(index='App', columns='variable', values='value')


# **concat() - vertical and horizontal**

# *vertical*

# In[ ]:


data1 = dataStore.head()
data2 = dataStore.tail()
conc_data_row = pd.concat([data1,data2],axis=0,ignore_index=True)
conc_data_row


# *horizontal*

# In[ ]:


data1 = dataStore['Rating'].head()
data2 = dataStore['Reviews'].head()
conc_data_col = pd.concat([data1,data2],axis=1)
conc_data_col


# **Data Type Convert**

# In[ ]:


dataStore['Reviews'] = dataStore['Reviews'].astype("float")


# **MISSING DATA**

# In[ ]:


dataStore.info()


# In[ ]:


dataStore['Reviews'].value_counts(dropna=False)


# **Assert Test**

# In[ ]:


assert dataStore['Reviews'].notnull().all() # true


# In[ ]:


assert dataStore.columns[3] == 'Reviews' # true


# In[ ]:


assert dataStore.columns[3] == 'Rating' # error


# In[ ]:


assert dataStore.Rating.dtypes == np.float # true


# In[ ]:


assert dataStore.Rating.dtypes == np.int # error


# In[ ]:


dataStore.plot(kind='hist',x='Reviews',y='Rating')
plt.show()


# **Indexing and Resampling Pandas Time Series******

# In[ ]:


#close warning
import warnings
warnings.filterwarnings("ignore")
data3 = dataStore.head()
date_list = ["1992-01-10","1992-02-10","1992-03-10","1993-03-15","1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data3['_____date_____'] = datetime_object # _____date_____ fixed column for date
data3 = data3.set_index('_____date_____') 
data3


# In[ ]:


print(data3.loc["1992-03-10":"1993-03-16"])


# **Resample**

# In[ ]:


data3.resample("A").mean() # A -> year


# In[ ]:


data3.resample("M").mean() # M -> Month


# In[ ]:


data3.resample("M").first().interpolate("linear") # fill


# In[ ]:


data3.resample("M").mean().interpolate("linear") # fill with mean


# **Manipulating Data Frames with Pandas**

# In[ ]:


dataReview = pd.read_csv('../input/googleplaystore_user_reviews.csv')
# dataReview = dataReview.set_index("App")
dataReview.index.name = "index"
dataReview.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




