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


# ## Reading Dataset

# In[ ]:



sky=pd.read_csv('../input/Skyserver_SQL2_27_2018 6_51_39 PM.csv')
sky.head()


# ## Checking for the null values

# In[ ]:


pd.isnull(sky).sum()


# There are no null values so does not need missing value imputation.

# ## Dropping columns not relevant for training the model like IDs
# 

# In[ ]:


sky=sky.drop(['objid','rerun','specobjid','fiberid'], axis=1)


# ## Finding the correlation between the columns to fetch the best suitable features for classification.
# ### Plotting the heatmap.

# In[ ]:


import seaborn as sns
corr=sky.corr() 
sns.heatmap(corr)


# From the above correlation heatmap we can see that columns u, g, r, i, z and redshift are highly correlated. So, we'll use these columns for classification.

# In[ ]:


sky=sky[['u','g','r','i','z','redshift','class']]


# ## Finding number of unique classes in 'class' column and will check if the classes are biased or not.

# In[ ]:


sky['class'].unique()


# In[ ]:


# Here we can see that Galaxies and Stars are more as compared to Quasars and so dataset is biased towards Galaxy and Star
sns.countplot(x='class',data=sky, palette='plasma_r')


# ## Splitting data into test(30% of data) and train(70% of data)

# In[ ]:


from sklearn.model_selection import train_test_split
train, test= train_test_split(sky, test_size=0.3, random_state=100)
train_x=train.drop('class',axis=1)
train_y=train['class']
test_x=test.drop('class',axis=1)
test_y=test['class']
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)


# In[ ]:


Accuracy_df=pd.DataFrame(columns=['Classification Algo','Accuracy'])


# ## Decision Tree Classifier

# In[ ]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
dt_model=DecisionTreeClassifier(random_state=100)
dt_model.fit(train_x,train_y)
dt_test_pred=dt_model.predict(test_x)
dt_acc=round(accuracy_score(test_y, dt_test_pred)*100,2)
Accuracy_df=Accuracy_df.append({'Classification Algo':'Decision Tree','Accuracy':dt_acc}, ignore_index=True)


# ## Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier
rf_model=RandomForestClassifier(random_state=100)
rf_model.fit(train_x,train_y)
rf_test_pred=rf_model.predict(test_x)
rf_acc=round(accuracy_score(test_y,rf_test_pred)*100,2)
Accuracy_df=Accuracy_df.append({'Classification Algo':'Random Forest','Accuracy':rf_acc}, ignore_index=True)


# ## AdaBoost

# In[ ]:


from sklearn.ensemble import AdaBoostClassifier
ab_model=AdaBoostClassifier(random_state=100)
ab_model.fit(train_x,train_y)
ab_test_pred=ab_model.predict(test_x)
ab_acc=round(accuracy_score(test_y, ab_test_pred)*100,2)
Accuracy_df=Accuracy_df.append({'Classification Algo':'AdaBoost','Accuracy':ab_acc}, ignore_index=True)


# ## K Nearest Neighbors

# In[ ]:


from sklearn.neighbors import KNeighborsClassifier
knn_model=KNeighborsClassifier()
knn_model.fit(train_x, train_y)
knn_test_pred=knn_model.predict(test_x)
knn_acc=round(accuracy_score(test_y, knn_test_pred)*100,2)
Accuracy_df=Accuracy_df.append({'Classification Algo':'KNN','Accuracy':knn_acc}, ignore_index=True)


# ## Naive Bayes

# In[ ]:


from sklearn.naive_bayes import GaussianNB
nb_model=GaussianNB()
nb_model.fit(train_x,train_y)
nb_test_pred=nb_model.predict(test_x)
nb_acc=round(accuracy_score(test_y, nb_test_pred)*100,2)
Accuracy_df=Accuracy_df.append({'Classification Algo':'Naive bayes','Accuracy':nb_acc}, ignore_index=True)
Accuracy_df


# In[ ]:




