#!/usr/bin/env python
# coding: utf-8

# **Loading libraries and data...**

# In[92]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))
import matplotlib.pyplot as plt
import seaborn as sns

# Any results you write to the current directory are saved as output.


# In[93]:


beers = pd.read_csv('../input/beers.csv')
breweries = pd.read_csv('../input/breweries.csv')


# In[ ]:





# In[94]:


beers.head(10)


#  **Let's just focus on abv, ibv. Since these are the two determining factors for the style of the beer.**

# In[95]:


df=beers[['abv','ibu','style']]


# We only focus on the style which has atleast 10 entries and drop all the null valued columns

# In[96]:


final=df.groupby('style').filter(lambda x: len(x) > 10)


# In[97]:


df=final.dropna()


# In[98]:


df.head()


# In[99]:


df.info()


# *Let's see who make the cut*

# In[100]:


df['style'].unique()


# Let's do some visualizations....
# First we see the line plot of different styles of beers and then the bar plot of mean alchohol content of different styles.

# In[101]:


final1=df.groupby('style').filter(lambda x: len(x) > 30)
plt.figure(figsize=(20,10)) 
g = sns.lmplot(x='abv',y='ibu',data=final1, hue='style')

plt.show(g)


# In[102]:


plt.figure(figsize=(15,15))
final2=df.groupby('style').agg({'abv':['mean']}).plot(kind='bar')
plt.show(final2)


# **Standardising the dataset**

# 

# In[103]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(df.drop('style',axis=1))
scaled_features = scaler.transform(df.drop('style',axis=1))
df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])


# In[104]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(scaled_features,df['style'],
                                                    test_size=0.20)


# ***KNN in action***

# In[105]:


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train,y_train)
pred = knn.predict(X_test)


# In[106]:


from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))
print(classification_report(y_test,pred))


# *Let's find the correct value of K*

# In[107]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', marker='o',
         markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[108]:


# NOW WITH K=5
knn = KNeighborsClassifier(n_neighbors=5)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)


# In[109]:


print('WITH K=5')
print('\n')
print(confusion_matrix(y_test,pred))
print('\n')
print(classification_report(y_test,pred))


# In[110]:



plt.scatter(y_test,pred)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')


# Thus we see only ABV and IBU are not enough to determine the style of the beer as we see from the low precision score. More features and sample data may help the case.

# In[ ]:




