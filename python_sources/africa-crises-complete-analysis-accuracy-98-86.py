#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.


# Data Loading :

# In[ ]:


df = pd.read_csv('/kaggle/input/africa-economic-banking-and-systemic-crisis-data/african_crises.csv')


# In[ ]:


df.head()


# Checking whether there is any NaN value present in the Dataset or not

# In[ ]:


df.isnull().sum()


# In[ ]:


X = df.drop('banking_crisis',axis=1)
X.head()


# In[ ]:


y=df.banking_crisis
y.head()


# Since cc3 and country columns are in string format, so we will convert them to integers first by using LabelEncoder

# In[ ]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X.cc3 = le.fit_transform(X.cc3)
X.country = le.fit_transform(X.country)
X.head()


# Now, we will try to find Correlation between all the given features

# In[ ]:


Correlation = X.corr()
Correlation


# Let's visualize it more clearly through Heatmap

# In[ ]:


from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


plt.figure(figsize=(30,30))
sns.heatmap(Correlation,cmap='coolwarm',annot=True,square=True, fmt ='.3f',annot_kws={'size' : 20})


# From this Heatmap, it is clearly visible that there is a strong correlation between features :
# 
# 1. cc3 and case
# 2. country and case
# 3. cc3 and country
# 
# We can observe that all the 3 features i.e. cc3,case and country are strongly correlated with each other. So, we will keep only 1 feature from these 3 features. I'll choose country and drop another cc3 and case.

# In[ ]:


X.drop(['cc3','case'],axis=1,inplace=True)
X.head()


# Now, X and y are ready for us. Let's split the data into Train and Test.

# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=1)


# I'll train the model with 75% of data and test the model with 25% of data.

# Let's apply Random Forest Algorithm on the above scenario

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

model = RandomForestClassifier()

# Training the Algorithm
model.fit(X_train,y_train)

# Testing the Algorithm
y_pred = model.predict(X_test)


# Checking the Accuracy of the model :

# In[ ]:


Accuracy = accuracy_score(y_test,y_pred)*100
print('Accuracy : ',Accuracy,'%')
print('Confusion Matrix : \n')
confusion_matrix(y_test,y_pred)


# Hence, we have achieved an Accuracy of : 98.86792452830188 %

# Hope it would be helpful for you. Thank You for watching. Cheers !

# In[ ]:




