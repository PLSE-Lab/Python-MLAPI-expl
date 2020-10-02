#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.


# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


day_df=pd.read_csv('../input/bike-sharing-dataset/day.csv')


# In[ ]:


hour_df=pd.read_csv('../input/bike-sharing-dataset/hour.csv')


# In[ ]:


day_df.head()


# In[ ]:


hour_df.head()


# In[ ]:


day_df.info()


# In[ ]:


hour_df.info()


# In[ ]:


day_df.drop('instant',axis=1,inplace=True)


# In[ ]:


g=sns.FacetGrid(day_df, hue='yr', palette='coolwarm',size=6,aspect=2)
g=g.map(plt.hist,'cnt',alpha=0.7, edgecolor='w')
plt.legend()


# In[ ]:


plt.figure(figsize=(11,5))
sns.barplot('yr','casual',hue='season', data=day_df,palette='rainbow', ci=None)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented on Casual basis')
plt.title('Number of bikes rented per season')


# In[ ]:


plt.figure(figsize=(11,5))
sns.barplot('yr','registered',hue='season', data=day_df,palette='rainbow', ci=None)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented on Registered basis')
plt.title('Number of bikes rented per season')


# In[ ]:


plt.figure(figsize=(11,5))
sns.barplot('yr','cnt',hue='season', data=day_df,palette='rainbow', ci=None)
plt.legend(loc='upper right',bbox_to_anchor=(1.2,0.5))
plt.xlabel('Year')
plt.ylabel('Total number of bikes rented')
plt.title('Number of bikes rented per season')


# * There was a substantial increase in bike renting from year 2011 to year 2012.
# * However, the trend for number of bikes rented based on seasons is the same each year.
# * Bikes are least rented in Spring and the most during Fall.
# * After Spring, there is a sudden increase in bike renting during summer.

# In[ ]:


df_season_fall=day_df[day_df['season']==3]


# In[ ]:


df_season_fall.mnth.nunique()


# Fall happens to be 4 months long. Let's examine which month has the highest number of rents and why.

# In[ ]:


sns.factorplot('mnth','cnt',hue='workingday',data=df_season_fall, ci=None, palette='Set2')


# * The renting is the maximum during the month 6(i.e. June: beginning of Fall) and also high during the month 9(i.e. September: end of Fall), considering a holiday/weekday
# * Also, when the renting for holidays is the least in the month 8(i.e. August) there is also a increase in bikes rented on a work day.

# Let's examine how the weather is responsible.

# In[ ]:


sns.factorplot('mnth','cnt',hue='weathersit',data=df_season_fall, ci=None, palette='Set2')


# * There is a much clear weather by the end of Fall and the cloudy, misty weather finds its way back. That very well explains the previous insight on our data, that people enjoy renting and riding bikes on a holiday when the weather is clear.

# In[ ]:


sns.jointplot('temp','cnt',data=day_df,size=7)


# In[ ]:


sns.lmplot('temp','cnt',row='workingday',col='season',data=day_df,palette='RdBu_r',fit_reg=False)


# So, people prefer more and more cycling as the days get hotter.

# In[ ]:


hour_df.drop('instant',axis=1,inplace=True)


# In[ ]:


plt.figure(figsize=(20,5))
mask = np.zeros_like(hour_df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(hour_df.corr(),cmap='RdBu_r',mask=mask, annot=True)


# In[ ]:


plt.figure(figsize=(20,5))
mask = np.zeros_like(day_df.corr(), dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
sns.heatmap(day_df.corr(),cmap='RdBu_r',mask=mask, annot=True)


# In[ ]:


df=pd.merge(day_df,hour_df,how='left',left_on='dteday',right_on='dteday')


# In[ ]:


df.head()


# In[ ]:


sns.heatmap(df.isnull(),yticklabels=False,cbar=False,cmap='viridis')


# In[ ]:


plt.figure(figsize=(12,12))
sns.heatmap(df.corr(),cmap='RdBu_r')


# In[ ]:


X=df.drop(['dteday','cnt_y'],axis=1)
y=df['cnt_y']


# In[ ]:


df.columns


# **Now, let's predict the trend of renting on hourly basis**

# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


X_train, X_test, y_train, y_test= train_test_split(X,y)


# In[ ]:


from sklearn.linear_model import LinearRegression


# In[ ]:


lm=LinearRegression()


# In[ ]:


lm.fit(X_train,y_train)


# In[ ]:


predictions=lm.predict(X_test)


# In[ ]:


plt.scatter(y_test,predictions)


# In[ ]:


from sklearn import metrics


# In[ ]:


print('MAE= ', metrics.mean_absolute_error(y_test,predictions))
print('MSE= ', metrics.mean_squared_error(y_test,predictions))
print('RMS= ', np.sqrt(metrics.mean_squared_error(y_test,predictions)))


# In[ ]:


pd.DataFrame(data=lm.coef_, index=X.columns, columns=['Coefficient'])


# In[ ]:


sns.distplot(y_test-predictions,bins=30)


# In[ ]:





# In[ ]:





# In[ ]:




