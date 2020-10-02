#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


print("reading the file...")
df = pd.read_csv("../input/advertising.csv",encoding = 'latin1')
print("reading done")


# In[ ]:


df


# In[ ]:


df_1 = df.drop(columns=['Clicked on Ad'])
df_1


# In[ ]:


label_names = np.array(['No','Yes'])
labels = df['Clicked on Ad'].values
feature_names = np.array(list(df_1))
features = np.array(df_1)


# In[ ]:


features_train,features_test,labels_train,labels_test = train_test_split(features,labels,test_size=0.33,random_state=5)


# In[ ]:


#Working on training dataset first!
#LabelEncoding the Categorical data
le = LabelEncoder()
features_train[:,4] = le.fit_transform(features_train[:,4])
features_train[:,5] = le.fit_transform(features_train[:,5])
features_train[:,7] = le.fit_transform(features_train[:,7])
features_train[:,8] = le.fit_transform(features_train[:,8])


# In[ ]:


model = tree.DecisionTreeClassifier()
model.fit(features_train,labels_train)


# In[ ]:


#Now working on test dataset
#Prediction
##test dataframe
df_features_test = pd.DataFrame(features_test,columns = list(df_1))
df_features_test


# In[ ]:


features_test1 = np.array(df_features_test) #taking the features of the test dataset into numpy ndarray


# In[ ]:


#LabelEncoding the Categorical data of the test dataframe
le = LabelEncoder()
features_test1[:,4] = le.fit_transform(features_test1[:,4])
features_test1[:,5] = le.fit_transform(features_test1[:,5])
features_test1[:,7] = le.fit_transform(features_test1[:,7])
features_test1[:,8] = le.fit_transform(features_test1[:,8])


# In[ ]:


predictions = model.predict(features_test1)
predictions


# In[ ]:


print(accuracy_score(labels_test,predictions))


# In[ ]:


model.predict_proba(features_test1)[:,1] #class probabilities

