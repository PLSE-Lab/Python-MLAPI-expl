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


get_ipython().system('pip install ppscore')


# In[ ]:




import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import ppscore as pps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
import numpy as np



warnings.filterwarnings("ignore")



# In[ ]:


df=pd.read_csv('/kaggle/input/churn-modelling/Churn_Modelling.csv')


# In[ ]:


df


# In[ ]:


df.describe()


# In[ ]:


lb=LabelEncoder()


# In[ ]:


df['Geography']=lb.fit_transform(df['Geography'])
df['Gender']=lb.fit_transform(df['Gender'])


# In[ ]:


df.info()


# In[ ]:


plt.figure(figsize=(15,8))
sns.heatmap(pps.matrix(df),annot=True)
plt.show()


# In[ ]:


sns.countplot(df['Exited'])
plt.show()


# In[ ]:


df['Exited'].value_counts(normalize=True)*100


# imbalanced data set

# In[ ]:


df['Gender'].groupby(df['Exited']).value_counts().plot(kind='bar')
plt.show()


# In[ ]:


df['NumOfProducts'].groupby(df['Exited']).value_counts().plot(kind='bar')
plt.show()


# In[ ]:


sns.scatterplot(df['Age'],df['CreditScore'],hue=df['Exited'])
plt.show()


# most of exited people lies between age of 40 to 60

# In[ ]:


X=df[['Age','NumOfProducts','CreditScore']]


# In[ ]:


y=df['Exited']


# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)


# In[ ]:


sc=StandardScaler()


# In[ ]:


X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)


# In[ ]:


lr=LogisticRegression(class_weight='balanced')


# In[ ]:


lr.fit(X_train,y_train)


# In[ ]:


y_pred=lr.predict(X_test)


# In[ ]:


print(classification_report(y_test,y_pred))
print(lr.score(X_train,y_train))
print(lr.score(X_test,y_test))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
cm


# KNN

# In[ ]:


knn1 = KNeighborsClassifier(n_neighbors=38,p=2)


# In[ ]:


knn1.fit(X_train,y_train)


# In[ ]:


pred5 = knn1.predict(X_test)


# In[ ]:


print(classification_report(y_test,pred5))
print(knn1.score(X_train,y_train))
print(knn1.score(X_test,y_test))


# In[ ]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,pred5)
cm


# In[ ]:





# In[ ]:




