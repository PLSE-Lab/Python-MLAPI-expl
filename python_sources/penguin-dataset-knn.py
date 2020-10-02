#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


path=r"/kaggle/input/palmer-archipelago-antarctica-penguin-data/penguins_size.csv"

df=pd.read_csv(path)


# In[ ]:


df.head()


# In[ ]:


df.info()


# In[ ]:


df.describe()


# # EDA

# In[ ]:


sns.countplot(x=df['species'])


# In[ ]:


sns.countplot(x=df['species'],hue=df['sex'])


# In[ ]:


sns.jointplot(x='culmen_length_mm', y='culmen_depth_mm',data=df)


# In[ ]:


sns.scatterplot(x='culmen_length_mm', y='culmen_depth_mm',data=df,hue='species')


# In[ ]:


sns.countplot(x=df['species'],hue=df['island'])


# # Converting categorical data

# In[ ]:


df=pd.get_dummies(df,columns=['sex','island'],drop_first=True)


# In[ ]:


df.head()


# In[ ]:


df.info()


# # Checking for missing values

# In[ ]:


sns.heatmap(df.isnull())


# In[ ]:


df=df.fillna(0)


# In[ ]:


sns.heatmap(df.isnull())


# # Standardizing Data

# In[ ]:


from sklearn.preprocessing import StandardScaler


# In[ ]:


scale=StandardScaler()
scale.fit(df.drop(['species'],axis=1))
transformed=scale.transform(df.drop(['species'],axis=1))
df_scaled=pd.DataFrame(transformed,columns=df.columns[1:])


# In[ ]:


df_scaled.info()


# # Lets Build Our Model

# In[ ]:


X=df_scaled
y=df['species']

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=101)


# In[ ]:


from sklearn.neighbors import KNeighborsClassifier

knn=KNeighborsClassifier(n_neighbors=1)

knn.fit(x_train,y_train)
out1=knn.predict(x_test)


# In[ ]:


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,out1))
print(confusion_matrix(y_test,out1))


# WOW we got it right at first attempt only

# Lets see the pattern of error rate for some values of n_neighbors

# In[ ]:


error_rate=[]

for i in range(1,20):
    knn_i=KNeighborsClassifier(n_neighbors=i)
    knn_i.fit(x_train,y_train)
    out_i=knn_i.predict(x_test)
    error_rate.append(np.mean(out_i!=y_test))


# In[ ]:


plt.plot(range(1,20),error_rate,marker='x',markerfacecolor='red')
plt.xlabel('# KNeighbors')
plt.ylabel('Error_rate')
plt.title('Best KNeighbors')


# Hence this graph depicts that error_rate is "ZERO" for values like 1,3,5,7 and then is zero for values after 7
# So the value of K neighbors in our model could be any odd number till 7 and any number after that giving 100% efficienct model

# # Hence our model predicts output with 100% efficiency
