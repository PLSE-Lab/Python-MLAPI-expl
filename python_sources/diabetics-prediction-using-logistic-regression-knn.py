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
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# In[ ]:


df=pd.read_csv('/kaggle/input/diabetes-dataset/diabetes2.csv')


# In[ ]:


df.head()


# In[ ]:


df.info() 


# In[ ]:


df.describe()


# In[ ]:


sns.countplot(x='Outcome',data=df) # 0 represent patients without daibetes and 1 represent patients with dabetes.


# In[ ]:


sns.distplot(df['Age'].dropna(),kde=True)


# In[ ]:


df.corr()


# In[ ]:


sns.heatmap(df.corr())


# In[ ]:


plt.subplots(figsize=(20,15))
sns.boxplot(x='Age', y='BMI', data=df)


# In[ ]:


x = df.drop('Outcome',axis=1)
y = df['Outcome']


# In[ ]:


from sklearn.model_selection import train_test_split


# In[ ]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[ ]:


from sklearn.linear_model import LogisticRegression


# In[ ]:


LR = LogisticRegression()
LR.fit(x_train,y_train)


# In[ ]:


predictions = LR.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix


# In[ ]:


print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier


# In[ ]:


error_rate = []
for i in range(1,40):    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(x_train,y_train)
    pred_i = knn.predict(x_test)
    error_rate.append(np.mean(pred_i != y_test))
    #print(error_rate)                                                  

plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed', 
         marker='o',  markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# In[ ]:


knn = KNeighborsClassifier(n_neighbors=15)


# # This Meathod will check K value from  1 to 40 and give us best value of k
#                                                                         

# In[ ]:


knn.fit(x_train,y_train)


# In[ ]:





# In[ ]:


pred = knn.predict(x_test)


# In[ ]:


print(classification_report(y_test,pred))
print(confusion_matrix(y_test,pred))


# 

# In[ ]:





# 

# 
